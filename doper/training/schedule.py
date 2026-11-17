# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optimizer construction and learning-rate schedules for DOPER training."""

from __future__ import annotations

import math
import warnings

import torch
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LambdaLR,
    LinearLR,
    SequentialLR,
)

from ..config import TrainConfig
from ..model import DoperNet

LR_SCHEDULES = ("cosine", "constant_cooldown", "cosine_tail10")


def lr_at(
    step: int,
    total_steps: int,
    base_lr: float,
    warmup: int,
    schedule: str = "cosine",
    cooldown_fraction: float = 0.2,
) -> float:
    """Return the analytical learning rate for one optimizer update.

    Args:
        step: Zero-based optimizer update index.
        total_steps: Total number of optimizer updates in the run.
        base_lr: Peak learning rate.
        warmup: Number of linear warmup updates.
        schedule: One of ``cosine``, ``constant_cooldown`` or ``cosine_tail10``.
        cooldown_fraction: Fraction of all updates spent in the final linear
            cooldown (``constant_cooldown``) or cosine tail (``cosine_tail10``).

    Returns:
        The learning rate for ``step``.

    Raises:
        ValueError: If the schedule or its parameters are invalid.
    """
    if schedule not in LR_SCHEDULES:
        raise ValueError("Unknown learning-rate schedule")
    if schedule == "constant_cooldown":
        if not 0 < cooldown_fraction <= 1 or warmup < 0:
            raise ValueError("Cooldown fraction must be in (0, 1] and warmup nonnegative")
        cooldown_steps = math.ceil(total_steps * cooldown_fraction)
        cooldown_start = total_steps - cooldown_steps
        if cooldown_steps < 2 or cooldown_start < warmup:
            raise ValueError("Cooldown needs at least two updates and must not overlap warmup")
    if schedule == "cosine_tail10":
        # Fraction of the entire update budget spent in the final cosine tail.
        # Join at 10% of peak on the last update preceding that tail.
        if not 0 < cooldown_fraction < 1 or warmup < 0:
            raise ValueError("Tail fraction must be in (0, 1) and warmup nonnegative")
        tail_steps = round(total_steps * cooldown_fraction)
        knee = total_steps - tail_steps - 1
        if tail_steps < 2 or knee <= warmup:
            raise ValueError("Cosine tail needs two updates and a join after warmup")
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    if schedule == "constant_cooldown":
        return base_lr * max(0.0, min(1.0, (total_steps - 1 - step) / (cooldown_steps - 1)))
    if schedule == "cosine_tail10":
        if step <= knee:
            progress = (step - warmup) / (knee - warmup)
            return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
        progress = min(1.0, (step - knee) / (total_steps - 1 - knee))
        return base_lr * 0.1 * 0.5 * (1.0 + math.cos(math.pi * progress))
    progress = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    total_steps: int,
    start_step: int = 0,
) -> SequentialLR:
    """Build the phase schedulers that reproduce :func:`lr_at` per update.

    Phase schedulers advance at the same update attempts as the historical loop.
    Replaying only scheduler steps restores legacy epoch-based resume without
    changing optimizer moments or requiring scheduler state in old checkpoints.
    :func:`lr_at` remains the analytical compatibility API for experiment tooling.

    Args:
        optimizer: Optimizer whose groups may carry an ``lr_scale`` entry.
        cfg: Training config (``lr``, ``warmup_iters``, ``lr_schedule`` and
            ``lr_cooldown_fraction`` are read).
        total_steps: Total number of optimizer updates in the run.
        start_step: Number of updates already taken (for resume).

    Returns:
        The sequential scheduler, already advanced by ``start_step`` steps.

    Raises:
        ValueError: If the warmup is negative or the schedule is invalid.
    """
    warmup = cfg.warmup_iters
    if warmup < 0:
        raise ValueError("Warmup must be nonnegative")
    schedule = cfg.lr_schedule
    fraction = cfg.lr_cooldown_fraction
    lr_at(0, total_steps, cfg.lr, warmup, schedule, fraction)  # shared validation
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"] = cfg.lr * group.get("lr_scale", 1.0)
    phases, milestones = [], []
    if warmup > 0:
        phases.append(
            LinearLR(
                optimizer,
                start_factor=1 / warmup,
                end_factor=1.0,
                total_iters=max(1, warmup - 1),
            )
        )
        milestones.append(warmup)
    if schedule == "cosine":
        phases.append(CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup)))
    elif schedule == "constant_cooldown":
        decay_steps = math.ceil(total_steps * fraction)
        decay_start = total_steps - decay_steps
        if decay_start > warmup:
            phases.append(ConstantLR(optimizer, factor=1.0, total_iters=decay_start - warmup))
            milestones.append(decay_start)
        phases.append(
            LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=decay_steps - 1)
        )
    else:
        knee = total_steps - round(total_steps * fraction) - 1
        # CosineAnnealingLR's absolute eta_min would break backbone/head scaling.
        phases.append(
            LambdaLR(
                optimizer,
                lambda step: 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / (knee - warmup))),
            )
        )
        milestones.append(knee)
        phases.append(
            LambdaLR(
                optimizer,
                lambda step: (
                    0.1 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / (total_steps - 1 - knee))))
                ),
            )
        )
    scheduler = SequentialLR(optimizer, schedulers=phases, milestones=milestones)
    # Reconstruction intentionally steps no optimizer; PyTorch otherwise warns on
    # the first replay step. No tensor/optimizer state is advanced here.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Detected call of `lr_scheduler.step\(\)` before `optimizer.step\(\)`",
        )
        for _ in range(start_step):
            scheduler.step()
    return scheduler


def build_optimizer(model: DoperNet, cfg: TrainConfig) -> torch.optim.AdamW:
    """Build AdamW while retaining legacy one-group resume compatibility.

    Args:
        model: Model whose trainable parameters are optimized.
        cfg: Training config (``lr``, ``weight_decay`` and
            ``backbone_lr_multiplier`` are read).

    Returns:
        A single-group optimizer when the backbone multiplier is 1, otherwise a
        backbone group and a head group carrying their ``lr_scale``.

    Raises:
        ValueError: If the backbone multiplier is not positive.
    """
    multiplier = cfg.backbone_lr_multiplier
    if multiplier <= 0:
        raise ValueError(f"backbone_lr_multiplier must be positive, got {multiplier}")
    if multiplier == 1.0:
        return torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
    backbone_ids = {id(parameter) for parameter in model.backbone.parameters()}
    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) in backbone_ids
    ]
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    return torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": cfg.lr * multiplier,
                "lr_scale": multiplier,
            },
            {"params": head_parameters, "lr": cfg.lr, "lr_scale": 1.0},
        ],
        weight_decay=cfg.weight_decay,
    )
