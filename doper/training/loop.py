# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The optimization loop over one epoch and the validation metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..cad_curves import curve_heatmap_metrics
from ..config import DoperConfig
from ..console import log
from ..losses import compute_losses
from ..model import DoperNet
from ..performance import image_layout
from ..targets import (
    decode_centernet,
    decode_keypoint_heatmaps,
    decode_simcc,
    heatmap_nms,
)

# Output stride of the keypoint and curve heatmaps relative to the crop.
KEYPOINT_HEATMAP_STRIDE = 4.0
# Probability above which a mask pixel counts as foreground.
_MASK_THRESHOLD = 0.5
_SECONDS_PER_HOUR = 3600
_SECONDS_PER_MINUTE = 60


def format_duration(seconds: float) -> str:
    """Return a human-readable H:MM:SS (or M:SS) duration.

    Args:
        seconds: Duration in seconds.

    Returns:
        The formatted duration.
    """
    seconds = int(round(seconds))
    h, rem = divmod(seconds, _SECONDS_PER_HOUR)
    m, s = divmod(rem, _SECONDS_PER_MINUTE)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def move_to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    """Move every tensor of a batch to ``device`` without blocking.

    Args:
        batch: Collated batch.
        device: Target device.

    Returns:
        The batch on ``device``.
    """
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@dataclass
class _ValidationTotals:
    """Running sums of the validation metrics."""

    num_curves: int
    kp_err_sum: float = 0.0
    kp_count: float = 0.0
    iou_sum: float = 0.0
    iou_count: float = 0.0
    center_err_sum: float = 0.0
    center_count: float = 0.0
    curve_iou_sum: np.ndarray = field(init=False)
    curve_dist_sum: np.ndarray = field(init=False)
    curve_count: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.curve_iou_sum = np.zeros((self.num_curves,), dtype=np.float64)
        self.curve_dist_sum = np.zeros((self.num_curves,), dtype=np.float64)
        self.curve_count = np.zeros((self.num_curves,), dtype=np.float64)


def _decode_keypoint_pairs(
    out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], cfg: DoperConfig, kp_dark: bool
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Decode predicted and ground-truth crop keypoints for each sample."""
    if cfg.model.keypoint_head == "simcc":
        split_ratio = cfg.model.simcc_split_ratio
        px = out["simcc_x"].cpu().numpy()
        py = out["simcc_y"].cpu().numpy()
        gx = batch["simcc_x"].cpu().numpy()
        gy = batch["simcc_y"].cpu().numpy()
        return [
            (
                decode_simcc(px[i], py[i], split_ratio, dark=kp_dark)[0],
                decode_simcc(gx[i], gy[i], split_ratio, dark=kp_dark)[0],
            )
            for i in range(px.shape[0])
        ]
    pred_heatmap = out["keypoint_heatmap"].cpu().numpy()
    gt_heatmap = batch["keypoint_heatmap"].cpu().numpy()
    return [
        (
            decode_keypoint_heatmaps(
                pred_heatmap[i],
                cfg.model.crop_size,
                beta=cfg.loss.keypoint_heatmap_beta,
                dark=kp_dark,
            )[0],
            decode_keypoint_heatmaps(
                gt_heatmap[i],
                cfg.model.crop_size,
                beta=1.0,
                dark=kp_dark,
            )[0],
        )
        for i in range(pred_heatmap.shape[0])
    ]


def _add_keypoint_errors(
    totals: _ValidationTotals,
    decoded: list[tuple[np.ndarray, np.ndarray]],
    w: np.ndarray,
) -> None:
    """Accumulate visibility-weighted keypoint distances."""
    for i, (pred_xy, gt_xy) in enumerate(decoded):
        wi = w[i]
        if wi.sum() <= 0:
            continue
        d = np.linalg.norm(pred_xy - gt_xy, axis=1)
        totals.kp_err_sum += float((d * wi).sum())
        totals.kp_count += float(wi.sum())


def _add_curve_metrics(
    totals: _ValidationTotals, out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> None:
    """Accumulate per-curve IoU and distance (in crop pixels)."""
    curve_probability = torch.sigmoid(out["curve_heatmap"]).cpu().numpy()
    curve_target = batch["curve_heatmap"].cpu().numpy()
    curve_weight = (batch["curve_weight"] * batch["has_curves"].view(-1, 1)).cpu().numpy()
    for i in range(curve_probability.shape[0]):
        metrics = curve_heatmap_metrics(
            curve_probability[i],
            curve_target[i],
            curve_weight[i],
        )
        iou = metrics["iou_per_curve"]
        distance = metrics["distance_per_curve"]
        valid = np.isfinite(iou) & np.isfinite(distance)
        totals.curve_iou_sum[valid] += iou[valid]
        totals.curve_dist_sum[valid] += distance[valid] * KEYPOINT_HEATMAP_STRIDE
        totals.curve_count[valid] += 1.0


def _add_mask_iou(
    totals: _ValidationTotals, out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> None:
    """Accumulate mask IoU at feature resolution.

    The masks are reduced to per-sample pixel counts on the device, so only
    integer counts are copied to the host.
    """
    prob = torch.sigmoid(out["mask"]) > _MASK_THRESHOLD
    tgt = batch["mask_target"] > _MASK_THRESHOLD
    intersections = (prob & tgt).flatten(1).sum(dim=1).cpu().numpy()
    unions = (prob | tgt).flatten(1).sum(dim=1).cpu().numpy()
    mw = batch["mask_weight"].cpu().numpy()
    for i in range(intersections.shape[0]):
        if mw[i] <= 0:
            continue
        inter = intersections[i]
        union = unions[i]
        if union > 0:
            totals.iou_sum += inter / union
            totals.iou_count += 1


def _add_center_errors(
    totals: _ValidationTotals,
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    det_stride: int,
) -> None:
    """Accumulate the top-1 detection center error in detector pixels."""
    hm = torch.sigmoid(out["heatmap"])
    hm_nms = heatmap_nms(hm).cpu().numpy()
    wh = out["wh"].cpu().numpy()
    off = out["offset"].cpu().numpy()
    # Only the ground-truth peak is needed, so take it on the device.
    gt_shape = tuple(batch["heatmap"].shape[1:])
    gt_peaks = batch["heatmap"].flatten(1).argmax(dim=1).cpu().numpy()
    for i in range(hm_nms.shape[0]):
        boxes, _, _ = decode_centernet(hm_nms[i], wh[i], off[i], det_stride, topk=1, score_thr=0.0)
        gt_idx = np.unravel_index(gt_peaks[i], gt_shape)
        gt_cx = gt_idx[2] * det_stride
        gt_cy = gt_idx[1] * det_stride
        if boxes.shape[0] > 0:
            pcx = (boxes[0, 0] + boxes[0, 2]) / 2
            pcy = (boxes[0, 1] + boxes[0, 3]) / 2
            totals.center_err_sum += math.hypot(pcx - gt_cx, pcy - gt_cy)
            totals.center_count += 1


def _curve_results(totals: _ValidationTotals, curve_names: Sequence[str]) -> dict[str, float]:
    """Reduce the per-curve sums to summary and per-name metrics."""
    result: dict[str, float] = {}
    valid = totals.curve_count > 0
    per_curve_iou = np.full_like(totals.curve_iou_sum, np.nan)
    per_curve_distance = np.full_like(totals.curve_dist_sum, np.nan)
    per_curve_iou[valid] = totals.curve_iou_sum[valid] / totals.curve_count[valid]
    per_curve_distance[valid] = totals.curve_dist_sum[valid] / totals.curve_count[valid]
    result["curve_iou"] = float(np.nanmean(per_curve_iou)) if valid.any() else float("nan")
    result["curve_distance_px"] = (
        float(np.nanmean(per_curve_distance)) if valid.any() else float("nan")
    )
    result["curve_worst_iou"] = float(np.nanmin(per_curve_iou)) if valid.any() else float("nan")
    result["curve_worst_distance_px"] = (
        float(np.nanmax(per_curve_distance)) if valid.any() else float("nan")
    )
    for curve_id, name in enumerate(curve_names):
        result[f"curve_iou/{name}"] = float(per_curve_iou[curve_id])
        result[f"curve_distance_px/{name}"] = float(per_curve_distance[curve_id])
    return result


@torch.no_grad()
def evaluate(
    model: DoperNet,
    loader: DataLoader,
    cfg: DoperConfig,
    device: str,
    kp_dark: bool = True,
) -> dict[str, float]:
    """Compute the validation metrics of all heads.

    Args:
        model: Model to evaluate; it is switched to eval mode.
        loader: Validation loader.
        cfg: Training config.
        device: Device the model runs on.
        kp_dark: Use DARK sub-bin keypoint decoding (else plain argmax).

    Returns:
        ``kp_err_px`` (crop pixels), ``mask_iou`` and ``det_center_err_px``
        (detector pixels), plus curve metrics when the curve head is enabled.
    """
    model.eval()
    totals = _ValidationTotals(cfg.model.num_curves)
    rotation_error = 0.0
    rotation_count = 0.0
    for batch in loader:
        batch = image_layout(move_to_device(batch, device), cfg.train.channels_last)
        out = model(batch["det_image"], batch["crop_image"])
        if cfg.model.rotation_aux_enabled:
            from ..rotation_aux import rotation_metrics

            top1, _ = rotation_metrics(
                out["rotation_logits"], batch["rotation_quaternion"], out["rotation_bins"]
            )
            weights = batch["rotation_weight"]
            rotation_error += float((top1 * weights).sum())
            rotation_count += float(weights.sum())
        w = (batch["kp_weight"] * batch["has_kp"].view(-1, 1)).cpu().numpy()
        decoded = _decode_keypoint_pairs(out, batch, cfg, kp_dark)
        _add_keypoint_errors(totals, decoded, w)
        if cfg.model.curve_head_enabled:
            _add_curve_metrics(totals, out, batch)
        _add_mask_iou(totals, out, batch)
        _add_center_errors(totals, out, batch, cfg.model.det_stride)
    result = {
        "kp_err_px": totals.kp_err_sum / max(totals.kp_count, 1.0),
        "mask_iou": float(totals.iou_sum / max(totals.iou_count, 1.0)),
        "det_center_err_px": totals.center_err_sum / max(totals.center_count, 1.0),
    }
    if cfg.model.rotation_aux_enabled:
        result["rotation_top1_deg"] = (
            rotation_error / rotation_count if rotation_count else float("nan")
        )
    if cfg.model.curve_head_enabled:
        result.update(_curve_results(totals, cfg.model.curve_names))
    return result


@dataclass
class TrainingComponents:
    """Everything one optimization step needs besides the batch."""

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scaler: torch.amp.GradScaler
    scheduler: Any
    cfg: DoperConfig
    device: str
    use_amp: bool
    amp_dtype: torch.dtype


@dataclass
class LoopCounters:
    """Counters that persist across epochs."""

    global_step: int
    nan_skips: int = 0


@dataclass
class EpochStats:
    """Throughput and update counts of one training epoch."""

    images: int = 0
    microbatches: int = 0
    updates: int = 0
    skipped_updates: int = 0


def _optimizer_step(parts: TrainingComponents, counters: LoopCounters, stats: EpochStats) -> None:
    """Clip, step, update the scaler, and advance the LR schedule."""
    cfg = parts.cfg
    if cfg.train.grad_clip > 0:
        parts.scaler.unscale_(parts.optimizer)
        torch.nn.utils.clip_grad_norm_(parts.model.parameters(), cfg.train.grad_clip)
    previous_scale = parts.scaler.get_scale()
    parts.scaler.step(parts.optimizer)
    parts.scaler.update()
    skipped_update = parts.scaler.get_scale() < previous_scale
    stats.updates += int(not skipped_update)
    stats.skipped_updates += int(skipped_update)
    parts.optimizer.zero_grad(set_to_none=True)
    counters.global_step += 1
    parts.scheduler.step()


def train_one_epoch(
    parts: TrainingComponents,
    loader: DataLoader,
    epoch: int,
    iters_per_epoch: int,
    counters: LoopCounters,
) -> EpochStats:
    """Run one epoch of gradient-accumulated mixed-precision training.

    Non-finite losses skip their microbatch and drop the partially accumulated
    gradient. The final partial accumulation of an epoch still takes a step.

    Args:
        parts: Model, optimizer, scaler, scheduler and runtime settings.
        loader: Training loader.
        epoch: Zero-based epoch index (for logging).
        iters_per_epoch: Microbatches to run (``max_iters_per_epoch`` capped).
        counters: Global step and non-finite skip counters, updated in place.

    Returns:
        This epoch's image, microbatch and update counts.
    """
    model, optimizer, cfg = parts.model, parts.optimizer, parts.cfg
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulated = 0
    stats = EpochStats()
    for it, batch in enumerate(loader):
        if cfg.train.max_iters_per_epoch and it >= cfg.train.max_iters_per_epoch:
            break
        stats.microbatches += 1
        stats.images += int(batch["det_image"].shape[0])
        lr = optimizer.param_groups[0]["lr"] / optimizer.param_groups[0].get("lr_scale", 1.0)
        batch = image_layout(move_to_device(batch, parts.device), cfg.train.channels_last)
        with torch.amp.autocast(device_type="cuda", enabled=parts.use_amp, dtype=parts.amp_dtype):
            out = model(batch["det_image"], batch["crop_image"])
            loss, logs = compute_losses(
                out,
                batch,
                cfg.loss,
                split_ratio=cfg.model.simcc_split_ratio,
                keypoint_heatmap_stride=KEYPOINT_HEATMAP_STRIDE,
            )
        # Skip non-finite batches so a single spike cannot poison the
        # weights with NaN/Inf (once a weight is NaN the run is unrecoverable).
        if not torch.isfinite(loss):
            counters.nan_skips += 1
            log(f"non-finite loss at e{epoch} it{it} (skip #{counters.nan_skips}), skipping batch")
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
            continue
        parts.scaler.scale(loss / cfg.train.gradient_accumulation_steps).backward()
        accumulated += 1
        end_of_epoch = it + 1 >= iters_per_epoch
        if accumulated >= cfg.train.gradient_accumulation_steps or end_of_epoch:
            _optimizer_step(parts, counters, stats)
            accumulated = 0
        if it % cfg.train.log_interval == 0:
            msg = " ".join(f"{k}={float(v):.4f}" for k, v in logs.items())
            print(f"[e{epoch} it{it}/{iters_per_epoch}] lr={lr:.2e} {msg}")
    return stats
