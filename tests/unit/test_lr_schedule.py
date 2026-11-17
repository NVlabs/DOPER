# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from doper.training.schedule import lr_at


def test_cosine_matches_historical_every_update():
    for step in range(29680):
        expected = (
            0.0005 * (step + 1) / 500
            if step < 500
            else 0.0005 * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - 500) / (29680 - 500))))
        )
        assert lr_at(step, 29680, 0.0005, 500) == expected


def test_constant_cooldown_boundaries_and_monotonicity():
    values = [lr_at(s, 29680, 0.002, 500, "constant_cooldown", 0.2) for s in range(29680)]
    assert values[0] == 0.002 / 500
    assert values[499] == values[500] == values[23744] == 0.002
    assert all(v == 0.002 for v in values[500:23745])
    assert all(a >= b >= 0 for a, b in zip(values[23744:], values[23745:], strict=False))
    assert values[-1] == 0


@pytest.mark.parametrize(
    "schedule,fraction", [("bad", 0.2), ("constant_cooldown", 0), ("constant_cooldown", 1.1)]
)
def test_invalid_schedule_is_rejected(schedule, fraction):
    with pytest.raises(ValueError):
        lr_at(0, 100, 0.001, 10, schedule, fraction)


@pytest.mark.parametrize("knee_epoch", [20, 25])
def test_cosine_tail10_exact_join_endpoint_and_warmup(knee_epoch):
    total = 35 * 742
    knee = knee_epoch * 742 - 1
    fraction = (35 - knee_epoch) / 35
    values = [lr_at(s, total, 0.0005, 500, "cosine_tail10", fraction) for s in range(total)]
    assert values[:500] == [lr_at(s, total, 0.0005, 500) for s in range(500)]
    assert values[500] == 0.0005
    assert values[knee] == pytest.approx(0.00005, abs=1e-15)
    assert values[-1] == 0
    assert all(a >= b >= 0 for a, b in zip(values[500:], values[501:], strict=False))
    assert abs(values[knee - 1] - values[knee + 1]) < 1e-9


@pytest.mark.parametrize("fraction", [0, 1, float("nan"), -0.2])
def test_cosine_tail10_rejects_invalid_or_warmup_overlapping_tail(fraction):
    with pytest.raises(ValueError):
        lr_at(0, 100, 0.0005, 50, "cosine_tail10", fraction)


@pytest.mark.parametrize(
    "schedule,fraction,total,warmup",
    [
        ("cosine", 0.2, 29680, 500),
        ("constant_cooldown", 0.2, 29680, 500),
        ("cosine_tail10", 15 / 35, 25970, 500),
        ("cosine_tail10", 10 / 35, 25970, 500),
        ("cosine", 0.2, 10, 0),
        ("cosine", 0.2, 10, 1),
        ("cosine", 0.2, 10, 20),
    ],
)
def test_pytorch_scheduler_matches_every_historical_update(schedule, fraction, total, warmup):
    from types import SimpleNamespace

    import torch

    from doper.training.schedule import build_lr_scheduler

    cfg = SimpleNamespace(
        lr=0.0005, warmup_iters=warmup, lr_schedule=schedule, lr_cooldown_fraction=fraction
    )
    optimizer = torch.optim.SGD(
        [
            {"params": [torch.nn.Parameter(torch.zeros(1))], "lr_scale": scale}
            for scale in (1.0, 0.1)
        ],
        lr=cfg.lr,
    )
    scheduler = build_lr_scheduler(optimizer, cfg, total)
    assert isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR)
    for step in range(total):
        expected = lr_at(step, total, cfg.lr, warmup, schedule, fraction)
        assert [g["lr"] for g in optimizer.param_groups] == pytest.approx(
            [expected, expected * 0.1], rel=1e-10, abs=1e-15
        )
        optimizer.step()
        scheduler.step()


def test_scheduler_resume_reconstructs_legacy_epoch_position_without_changing_optimizer_state():
    from types import SimpleNamespace

    import torch

    from doper.training.schedule import build_lr_scheduler

    cfg = SimpleNamespace(
        lr=0.0005, warmup_iters=5, lr_schedule="constant_cooldown", lr_cooldown_fraction=0.2
    )
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([{"params": [parameter], "lr_scale": 0.1}], lr=cfg.lr)
    scheduler = build_lr_scheduler(optimizer, cfg, 100)
    for _ in range(85):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
    state = optimizer.state_dict()
    expected_lr = optimizer.param_groups[0]["lr"]
    optimizer.load_state_dict(state)
    before = parameter.detach().clone()
    resumed = build_lr_scheduler(optimizer, cfg, 100, start_step=85)
    assert torch.equal(parameter, before)
    assert optimizer.state[parameter]["step"].item() == 85
    assert optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)
    for step in range(85, 100):
        assert optimizer.param_groups[0]["lr"] == pytest.approx(
            lr_at(step, 100, cfg.lr, 5, "constant_cooldown", 0.2) * 0.1
        )
        optimizer.step()
        resumed.step()
