# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Joint training entry point for the DOPER unified model.

Example (a rendered COCO dataset with ``train/`` and ``val/`` splits)::

    python -m doper.train \
        --data-root data/object \
        --work-dir work_dirs/object \
        --epochs 200 --batch-size 16

Validation reports mean keypoint error (crop pixels), foreground mask IoU, and
detection center error so all three heads are tracked from one run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Sequence
from typing import Any

import torch
from torch.utils.data import DataLoader

from .checkpoint_io import atomic_torch_save, capture_rng_state, restore_rng_state
from .config import DoperConfig
from .console import log
from .dataset import DoperDataset
from .keypoints import keypoints_3d_fingerprint, load_keypoints_3d
from .model import DoperNet
from .performance import configure_runtime, loader_options, resolve_device, seed_training
from .training.checkpoints import (
    load_init_weights,
    load_resume_checkpoint,
    resumed_best_kp,
    save_validation_checkpoints,
    trim_epoch_timing,
    write_training_provenance,
)
from .training.data import (
    RatioBatchSampler,
    build_train_loader,
    build_train_set,
    build_val_loader,
    curve_supervision_count,
    resolve_train_roots,
)
from .training.dry_run import check_dry_run
from .training.loop import (
    EpochStats,
    LoopCounters,
    TrainingComponents,
    evaluate,
    format_duration,
    train_one_epoch,
)
from .training.recipe import build_config, parse_args, validate_config, validate_training_sources
from .training.schedule import build_lr_scheduler, build_optimizer, lr_at

__all__ = [
    "RatioBatchSampler",
    "build_config",
    "evaluate",
    "main",
    "resumed_best_kp",
    "save_validation_checkpoints",
    "trim_epoch_timing",
]


def _build_val_set(cfg: DoperConfig) -> DoperDataset:
    """Build the validation set from the primary root only.

    Validation runs on the primary root only so kp_err_px / mask_iou stay
    meaningful (real bbox-only roots carry neither keypoints nor masks).
    """
    val_set = DoperDataset(cfg, "val")
    if cfg.model.curve_head_enabled and curve_supervision_count(val_set) <= 0:
        log(
            "warning: validation split has no curve_file labels; "
            "curve validation metrics will be NaN"
        )
    return val_set


def _build_model(cfg: DoperConfig, args: argparse.Namespace, device: str) -> DoperNet:
    """Construct the model, apply ``--init-from`` and report the keypoint layout."""
    model = DoperNet(cfg.model).to(device)
    if cfg.train.channels_last:
        model.to(memory_format=torch.channels_last)  # pyright: ignore[reportCallIssue]
    if args.init_from:
        load_init_weights(model, cfg, args.init_from, device)
    return model


def _keypoints_fingerprint(cfg: DoperConfig) -> str | None:
    """Fingerprint the configured 3D keypoint file, if it exists."""
    kp_path = getattr(cfg.data, "keypoints_3d_json", None)
    if not (kp_path and os.path.exists(kp_path)):
        return None
    fingerprint = keypoints_3d_fingerprint(load_keypoints_3d(kp_path))
    log(f"keypoints_3d fingerprint: {fingerprint}")
    return fingerprint


def _freeze_keypoint_head(model: DoperNet) -> None:
    """Exclude the keypoint head from optimization."""
    model.kpt_head.requires_grad_(False)
    frozen = sum(parameter.numel() for parameter in model.kpt_head.parameters())
    log(f"froze keypoint head ({frozen:,} parameters)")


def _count_steps(cfg: DoperConfig, train_loader: DataLoader) -> tuple[int, int, int]:
    """Return ``(iters_per_epoch, optimizer_steps_per_epoch, total_steps)``."""
    iters_per_epoch = len(train_loader)
    if cfg.train.max_iters_per_epoch > 0:
        iters_per_epoch = min(iters_per_epoch, cfg.train.max_iters_per_epoch)
    optimizer_steps_per_epoch = math.ceil(iters_per_epoch / cfg.train.gradient_accumulation_steps)
    total_steps = optimizer_steps_per_epoch * cfg.train.epochs
    lr_at(
        0,
        total_steps,
        cfg.train.lr,
        cfg.train.warmup_iters,
        cfg.train.lr_schedule,
        cfg.train.lr_cooldown_fraction,
    )
    return iters_per_epoch, optimizer_steps_per_epoch, total_steps


def _model_metadata(cfg: DoperConfig, keypoints_fingerprint: str | None) -> dict[str, Any]:
    """Return the ``cfg`` entry stored in every checkpoint."""
    return {"model": cfg.model.__dict__, "keypoints_3d_fingerprint": keypoints_fingerprint}


def _print_epoch_eta(
    epoch: int, start_epoch: int, epochs: int, t0: float, run_start: float
) -> None:
    """Report the epoch duration, elapsed time and remaining-time estimate."""
    epochs_done = epoch - start_epoch + 1
    epochs_left = epochs - (epoch + 1)
    elapsed = time.time() - run_start
    eta = (elapsed / epochs_done) * epochs_left
    log(
        f"epoch {epoch} done in {time.time() - t0:.1f}s | "
        f"elapsed {format_duration(elapsed)} | "
        f"ETA {format_duration(eta)} ({epochs_left} epochs left)"
    )


def _append_epoch_timing(
    work_dir: str,
    epoch: int,
    stats: EpochStats,
    nonfinite: int,
    times: tuple[float, float],
    global_step: int,
) -> None:
    """Append one row to ``epoch_timing.jsonl``; ``times`` is (start, train seconds)."""
    t0, train_seconds = times
    timing = {
        "epoch": epoch,
        "images": stats.images,
        "microbatches": stats.microbatches,
        "optimizer_updates": stats.updates,
        "scaler_skipped_updates": stats.skipped_updates,
        "nonfinite_microbatches": nonfinite,
        "train_seconds": train_seconds,
        "train_images_per_second": stats.images / train_seconds,
        "epoch_seconds_including_validation_and_checkpoints": time.time() - t0,
        "global_scheduler_step": global_step,
    }
    with open(os.path.join(work_dir, "epoch_timing.jsonl"), "a") as stream:
        stream.write(json.dumps(timing) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    """Train DOPER from the command line.

    Args:
        argv: Arguments without the program name; None reads ``sys.argv``.
    """
    args = parse_args(argv)
    cfg = build_config(args)
    validate_config(cfg, args)
    if args.dry_run:
        check_dry_run(cfg, args)
        return
    if args.resume and not os.path.isfile(args.resume):
        raise FileNotFoundError(f"resume checkpoint is missing: {args.resume}")
    device = resolve_device(cfg.train.device)
    use_amp = cfg.train.amp and device.startswith("cuda")
    amp_dtype = configure_runtime(cfg.train, device)
    loader_options(cfg.train)  # Validate before dataset/model setup.
    os.makedirs(cfg.train.work_dir, exist_ok=True)
    seed_training(cfg.train)
    # Data.
    keypoint_only_roots = list(args.keypoint_only_mix_data_root or [])
    train_roots = resolve_train_roots(
        args.data_root, list(args.mix_data_root or []), keypoint_only_roots
    )
    train_set = build_train_set(cfg, train_roots, keypoint_only_roots)
    source_sets = validate_training_sources(cfg, args, train_set, train_roots, keypoint_only_roots)
    val_set = _build_val_set(cfg)
    log(f"train={len(train_set)} (roots={len(train_roots)}) val={len(val_set)} device={device}")
    training_provenance = write_training_provenance(args, cfg, train_roots, source_sets)
    train_loader = build_train_loader(cfg, train_set, train_roots, args.mix_per_batch, device)
    val_loader = build_val_loader(cfg, val_set, device)
    # Model and optimizer.
    model = _build_model(cfg, args, device)
    keypoints_fingerprint = _keypoints_fingerprint(cfg)
    if args.freeze_keypoint_head:
        _freeze_keypoint_head(model)
    optimizer = build_optimizer(model, cfg.train)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    iters_per_epoch, optimizer_steps_per_epoch, total_steps = _count_steps(cfg, train_loader)
    # Resume.
    start_epoch = 0
    best_kp = float("inf")
    resume_state = None
    if args.resume:
        start_epoch, best_kp, resume_state = load_resume_checkpoint(
            args.resume, model, optimizer, scaler, cfg, device
        )
    global_step = (resume_state or {}).get("global_step", start_epoch * optimizer_steps_per_epoch)
    scheduler = build_lr_scheduler(optimizer, cfg.train, total_steps, global_step)
    if resume_state and "rng_state" in resume_state:
        restore_rng_state(resume_state["rng_state"])
    if isinstance(train_loader.batch_sampler, RatioBatchSampler):
        train_loader.batch_sampler.set_epoch(start_epoch)
    # Epoch loop.
    parts = TrainingComponents(model, optimizer, scaler, scheduler, cfg, device, use_amp, amp_dtype)
    counters = LoopCounters(global_step=global_step)
    metadata = _model_metadata(cfg, keypoints_fingerprint)
    run_start = time.time()
    for epoch in range(start_epoch, cfg.train.epochs):
        t0 = time.time()
        epoch_nan_start = counters.nan_skips
        stats = train_one_epoch(parts, train_loader, epoch, iters_per_epoch, counters)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        train_seconds = time.time() - t0
        _print_epoch_eta(epoch, start_epoch, cfg.train.epochs, t0, run_start)
        is_last = epoch == cfg.train.epochs - 1
        if (epoch + 1) % cfg.train.val_interval == 0 or is_last:
            metrics = evaluate(model, val_loader, cfg, device, kp_dark=args.kp_dark)
            log(f"val epoch {epoch}: " + json.dumps(metrics))
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "global_step": counters.global_step,
                "rng_state": capture_rng_state(),
                "epoch": epoch,
                "cfg": metadata,
                "metrics": metrics,
                "training_provenance": training_provenance,
            }
            best_kp = save_validation_checkpoints(
                ckpt, metrics["kp_err_px"], best_kp, cfg.train.work_dir
            )
        if (epoch + 1) % cfg.train.save_interval == 0:
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "cfg": metadata,
                    "training_provenance": training_provenance,
                },
                os.path.join(cfg.train.work_dir, f"epoch_{epoch + 1}.pth"),
            )
        _append_epoch_timing(
            cfg.train.work_dir,
            epoch,
            stats,
            counters.nan_skips - epoch_nan_start,
            (t0, train_seconds),
            counters.global_step,
        )


if __name__ == "__main__":
    main()
