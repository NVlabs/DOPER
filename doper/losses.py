# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Joint loss for the DOPER unified model.

Combines:

* CenterNet penalty-reduced focal loss on the detection heatmap,
* L1 on width/height and sub-pixel offset at positive cells,
* BCE + soft Dice on the foreground mask,
* SimCC KL-divergence loss on the keypoint x/y distributions.

Each term is masked by its availability flag so that real-world samples without
keypoints (or without masks) contribute only the supervision they actually
carry.
"""

from __future__ import annotations

import torch
from torch.nn import functional

from .config import LossConfig

# Probability clamp that keeps the focal-loss logarithms finite.
PROBABILITY_EPSILON = 1e-6
# CenterNet penalty-reduction exponents.
FOCAL_NEGATIVE_EXPONENT = 4
FOCAL_PREDICTION_EXPONENT = 2
# Laplace smoothing of the soft Dice ratio.
DICE_SMOOTH = 1.0
# Default SimCC temperature (RTMPose KLDiscretLoss beta).
DEFAULT_SIMCC_BETA = 10.0
DEFAULT_KEYPOINT_HEATMAP_BETA = 1.0
# Class-balancing bounds for the positive BCE term of the curve loss.
CURVE_POSITIVE_SCALE_MIN = 1.0
CURVE_POSITIVE_SCALE_MAX = 100.0
# Probability threshold used for the curve IoU diagnostic.
CURVE_IOU_THRESHOLD = 0.5


def neg_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    """CenterNet penalty-reduced focal loss.

    Args:
        pred: Sigmoid probabilities of shape ``(B, C, H, W)``.
        gt: Gaussian ground-truth heatmap, 1.0 at object centers.
        sample_weight: Per-sample weights of shape ``(B,)``.

    Returns:
        The scalar loss normalized by the weighted number of positive cells. When
        there are no positives (weighted count of zero), the un-normalized negative
        loss is returned instead.
    """
    # Keep the logarithms in float32 even when the surrounding forward pass is
    # under autocast.  In float16, ``1.0 - 1e-6`` rounds back to exactly 1.0;
    # a saturated sigmoid can therefore reach log(0) below and turn the loss
    # into NaN.  This is rare early in training but becomes increasingly likely
    # after repeated fine-tuning generations sharpen the detection logits.
    pred = pred.float()
    gt = gt.float()
    sample_weight = sample_weight.float()
    pos_inds = gt.eq(1.0).float()
    neg_inds = gt.lt(1.0).float()
    neg_weights = torch.pow(1.0 - gt, FOCAL_NEGATIVE_EXPONENT)
    weight = sample_weight.view(-1, 1, 1, 1)

    pred = pred.clamp(PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    positive_loss = (
        torch.log(pred) * torch.pow(1.0 - pred, FOCAL_PREDICTION_EXPONENT) * pos_inds * weight
    )
    negative_loss = (
        torch.log(1.0 - pred)
        * torch.pow(pred, FOCAL_PREDICTION_EXPONENT)
        * neg_weights
        * neg_inds
        * weight
    )

    num_pos = (pos_inds * weight).sum()
    positive_loss = positive_loss.sum()
    negative_loss = negative_loss.sum()
    if num_pos == 0:
        return -negative_loss
    return -(positive_loss + negative_loss) / num_pos


def reg_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """L1 over cells where ``weight`` (broadcast over channels) is positive.

    Args:
        pred: Predicted regression map of shape ``(B, C, H, W)``.
        target: Regression target of the same shape.
        weight: Per-cell weights of shape ``(B, 1, H, W)``.

    Returns:
        The scalar loss normalized by the weighted cell count and channels.
    """
    pred = pred.float()
    target = target.float()
    weight = weight.float()
    loss = functional.l1_loss(pred * weight, target * weight, reduction="sum")
    return loss / (weight.sum().clamp(min=1.0) * pred.shape[1])


def _mask_loss(
    pred_logits: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor
) -> torch.Tensor:
    """BCE + soft Dice on per-sample masks, weighted by availability.

    Args:
        pred_logits: Mask logits of shape ``(B, 1, H, W)``.
        target: Binary mask target of the same shape.
        sample_weight: Per-sample availability weights of shape ``(B,)``.

    Returns:
        The scalar sum of the BCE and Dice terms.
    """
    # A 256x256 mask contains 65,536 pixels, which already exceeds float16's
    # largest finite value (65,504).  A confident foreground prediction can
    # therefore overflow both Dice reductions and produce inf/inf = NaN.
    pred_logits = pred_logits.float()
    target = target.float()
    sample_weight = sample_weight.float()
    sw = sample_weight.view(-1, 1, 1, 1)
    bce = functional.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
    bce = (bce * sw).sum() / (sw.sum().clamp(min=1.0) * pred_logits[0].numel())

    prob = torch.sigmoid(pred_logits)
    dims = (1, 2, 3)
    inter = (prob * target * sw).sum(dims)
    denom = ((prob + target) * sw).sum(dims)
    dice = 1.0 - (2.0 * inter + DICE_SMOOTH) / (denom + DICE_SMOOTH)
    dice = (dice * sample_weight).sum() / sample_weight.sum().clamp(min=1.0)
    return bce + dice


def _simcc_kl(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    beta: float = DEFAULT_SIMCC_BETA,
) -> torch.Tensor:
    """KL(target || softmax(beta * pred)) summed over bins, weighted per keypoint.

    ``pred``: (B, K, bins) logits. ``target``: (B, K, bins) normalized gaussian.
    ``weight``: (B, K). ``beta`` is the RTMPose-style temperature that sharpens
    the predicted distribution before the softmax.
    """
    pred = pred.float()
    target = target.float()
    weight = weight.float()
    log_p = functional.log_softmax(pred * beta, dim=2)
    # target * (log target - log p); target log target term is constant wrt
    # the network, so the cross-entropy form is sufficient and avoids log(0).
    loss = -(target * log_p).sum(dim=2)  # (B, K)
    loss = loss * weight
    denom = weight.sum().clamp(min=1.0)
    return loss.sum() / denom


def _keypoint_heatmap_kl(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    beta: float = DEFAULT_KEYPOINT_HEATMAP_BETA,
) -> torch.Tensor:
    """KL cross-entropy for normalized joint 2D keypoint distributions.

    Args:
        pred: Heatmap logits of shape ``(B, K, H, W)``.
        target: Normalized target distributions of the same shape.
        weight: Per-keypoint weights of shape ``(B, K)``.
        beta: Temperature applied to the logits before the softmax.

    Returns:
        The scalar weighted cross-entropy.
    """
    pred = pred.float()
    target = target.float()
    weight = weight.float()
    batch_size, num_keypoints = pred.shape[:2]
    pred_flat = pred.reshape(batch_size, num_keypoints, -1)
    target_flat = target.reshape(batch_size, num_keypoints, -1)
    log_probability = functional.log_softmax(pred_flat * beta, dim=2)
    loss = -(target_flat * log_probability).sum(dim=2) * weight
    return loss.sum() / weight.sum().clamp(min=1.0)


def _curve_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> torch.Tensor:
    """Class-balanced BCE plus soft Dice, averaged over valid named channels.

    Args:
        pred_logits: Curve logits of shape ``(B, C, H, W)``.
        target: Curve targets of the same shape.
        weight: Per-curve weights of shape ``(B, C)``.
        bce_weight: Weight of the BCE term.
        dice_weight: Weight of the Dice term.

    Returns:
        The scalar weighted loss.

    Raises:
        ValueError: If the tensor shapes are inconsistent.
    """
    if pred_logits.shape != target.shape or pred_logits.ndim != 4:
        raise ValueError("curve logits and targets must have matching (B,C,H,W) shapes")
    if weight.shape != pred_logits.shape[:2]:
        raise ValueError("curve weight must have shape (B,C)")
    pred_logits = pred_logits.float()
    target = target.float()
    weight = weight.float()
    dims = (2, 3)
    positive_mass = target.sum(dims)
    negative_mass = (1.0 - target).sum(dims)
    positive_scale = (negative_mass / positive_mass.clamp(min=1.0)).clamp(
        min=CURVE_POSITIVE_SCALE_MIN, max=CURVE_POSITIVE_SCALE_MAX
    )
    bce = -(
        positive_scale[:, :, None, None] * target * functional.logsigmoid(pred_logits)
        + (1.0 - target) * functional.logsigmoid(-pred_logits)
    ).mean(dims)

    probability = torch.sigmoid(pred_logits)
    intersection = (probability * target).sum(dims)
    denominator = (probability + target).sum(dims)
    dice = 1.0 - (2.0 * intersection + DICE_SMOOTH) / (denominator + DICE_SMOOTH)
    per_channel = bce_weight * bce + dice_weight * dice
    return (per_channel * weight).sum() / weight.sum().clamp(min=1.0)


def compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    loss_cfg: LossConfig,
    split_ratio: float = 2.0,
    keypoint_heatmap_stride: float = 4.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the weighted joint loss and its scalar logs.

    Args:
        outputs: Network outputs (``heatmap``, ``wh``, ``offset``, ``mask``, the
            keypoint head outputs and optionally ``curve_heatmap``).
        batch: Collated training targets and availability weights.
        loss_cfg: Loss weights and temperatures.
        split_ratio: SimCC bins per pixel, used for the keypoint error log.
        keypoint_heatmap_stride: Crop pixels per heatmap cell, used for the
            keypoint error log of the heatmap head.

    Returns:
        ``(total, logs)`` where ``total`` is the scalar loss tensor and ``logs``
        maps log names to detached 0-d tensors. They stay on the loss device so
        a training step does not wait on the GPU; call ``float`` to read one.
    """
    hm_prob = torch.sigmoid(outputs["heatmap"])
    det_weight = batch.get(
        "det_weight",
        torch.ones(
            outputs["heatmap"].shape[0],
            dtype=outputs["heatmap"].dtype,
            device=outputs["heatmap"].device,
        ),
    )
    reg_weight = batch["reg_weight"] * det_weight.view(-1, 1, 1, 1)
    hm_loss = neg_loss(hm_prob, batch["heatmap"], det_weight)
    wh_loss = reg_l1(outputs["wh"], batch["wh"], reg_weight)
    off_loss = reg_l1(outputs["offset"], batch["offset"], reg_weight)
    mask_loss = _mask_loss(outputs["mask"], batch["mask_target"], batch["mask_weight"])

    kp_weight = batch["kp_weight"] * batch["has_kp"].view(-1, 1)
    if "simcc_x" in outputs:
        kx = _simcc_kl(
            outputs["simcc_x"],
            batch["simcc_x"],
            kp_weight,
            loss_cfg.simcc_beta,
        )
        ky = _simcc_kl(
            outputs["simcc_y"],
            batch["simcc_y"],
            kp_weight,
            loss_cfg.simcc_beta,
        )
        kp_loss = kx + ky
        with torch.no_grad():
            pcx = outputs["simcc_x"].argmax(dim=2).float()
            pcy = outputs["simcc_y"].argmax(dim=2).float()
            gcx = batch["simcc_x"].argmax(dim=2).float()
            gcy = batch["simcc_y"].argmax(dim=2).float()
            dist = torch.sqrt((pcx - gcx) ** 2 + (pcy - gcy) ** 2) / split_ratio
    else:
        kp_loss = _keypoint_heatmap_kl(
            outputs["keypoint_heatmap"],
            batch["keypoint_heatmap"],
            kp_weight,
            loss_cfg.keypoint_heatmap_beta,
        )
        with torch.no_grad():
            pred = outputs["keypoint_heatmap"]
            target = batch["keypoint_heatmap"]
            width = pred.shape[3]
            pred_idx = pred.flatten(2).argmax(dim=2)
            target_idx = target.flatten(2).argmax(dim=2)
            pcx = torch.remainder(pred_idx, width).float()
            pcy = torch.div(pred_idx, width, rounding_mode="floor").float()
            gcx = torch.remainder(target_idx, width).float()
            gcy = torch.div(target_idx, width, rounding_mode="floor").float()
            dist = torch.sqrt((pcx - gcx) ** 2 + (pcy - gcy) ** 2) * keypoint_heatmap_stride

    # Interpretable per-iter keypoint error in crop pixels. Masked to samples
    # carrying keypoints so bbox/mask-only crops do not dilute the metric.
    with torch.no_grad():
        wsum = kp_weight.sum()
        kp_err_px = torch.where(wsum > 0, (dist * kp_weight).sum() / wsum, float("nan"))

    curve_loss = outputs["heatmap"].new_zeros(())
    curve_iou = outputs["heatmap"].new_full((), float("nan"))
    if "curve_heatmap" in outputs:
        curve_weight = batch["curve_weight"] * batch["has_curves"].view(-1, 1)
        curve_loss = _curve_loss(
            outputs["curve_heatmap"],
            batch["curve_heatmap"],
            curve_weight,
            bce_weight=loss_cfg.curve_bce_weight,
            dice_weight=loss_cfg.curve_dice_weight,
        )
        with torch.no_grad():
            prediction = torch.sigmoid(outputs["curve_heatmap"]) >= CURVE_IOU_THRESHOLD
            truth = batch["curve_heatmap"] >= CURVE_IOU_THRESHOLD
            intersection = (prediction & truth).sum(dim=(2, 3)).float()
            union = (prediction | truth).sum(dim=(2, 3)).float()
            valid = (curve_weight > 0) & (union > 0)
            curve_iou = torch.where(valid, intersection / union, float("nan")).nanmean()

    rotation_loss = outputs["heatmap"].new_zeros(())
    if "rotation_logits" in outputs:
        from .rotation_aux import rotation_aux_loss

        rotation_loss = rotation_aux_loss(
            outputs["rotation_logits"],
            batch["rotation_quaternion"],
            outputs["rotation_bins"],
            sigma_deg=loss_cfg.rotation_sigma_deg,
            sample_weight=batch["rotation_weight"],
        )

    total = (
        loss_cfg.rotation_weight * rotation_loss
        + loss_cfg.heatmap_weight * hm_loss
        + loss_cfg.wh_weight * wh_loss
        + loss_cfg.offset_weight * off_loss
        + loss_cfg.mask_weight * mask_loss
        + loss_cfg.keypoint_weight * kp_loss
        + loss_cfg.curve_weight * curve_loss
    )
    logs = {
        "loss": total.detach(),
        "hm": hm_loss.detach(),
        "wh": wh_loss.detach(),
        "off": off_loss.detach(),
        "mask": mask_loss.detach(),
        "kp": kp_loss.detach(),
        "kp_px": kp_err_px,
        "curve": curve_loss.detach(),
        "curve_iou": curve_iou,
    }
    if "rotation_logits" in outputs:
        logs["rotation"] = rotation_loss.detach()
    return total, logs
