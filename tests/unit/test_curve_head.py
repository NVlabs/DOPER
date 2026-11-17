# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from tiny_coco import curve_fields, tiny_model_config

from doper.config import LossConfig, ModelConfig
from doper.losses import _curve_loss, compute_losses
from doper.model import DoperNet, NamedCurveHead


def _model_config(keypoint_head: str, curves: bool) -> ModelConfig:
    return tiny_model_config(
        num_classes=1,
        num_keypoints=2,
        det_input_size=64,
        neck_channels=8,
        head_channels=8,
        crop_size=32,
        keypoint_head=keypoint_head,
        gau_hidden_dims=8,
        gau_s=4,
        curve_head_channels=8,
        **(curve_fields(("a", "b", "c"), "hash") if curves else {}),
    )


@pytest.mark.parametrize("keypoint_head", ["simcc", "heatmap"])
def test_curve_head_runs_beside_both_keypoint_representations(keypoint_head):
    cfg = _model_config(keypoint_head, curves=True)
    model = DoperNet(cfg).eval()

    with torch.no_grad():
        output = model(
            torch.zeros(1, 3, cfg.det_input_size, cfg.det_input_size),
            torch.zeros(1, 3, cfg.crop_size, cfg.crop_size),
        )

    assert output["curve_heatmap"].shape == (1, 3, 8, 8)
    if keypoint_head == "simcc":
        assert output["simcc_x"].shape[:2] == (1, 2)
    else:
        assert output["keypoint_heatmap"].shape == (1, 2, 8, 8)


def test_disabled_curve_head_preserves_legacy_state_and_outputs():
    model = DoperNet(_model_config("simcc", curves=False)).eval()

    assert not hasattr(model, "curve_head")
    assert not any(key.startswith("curve_head.") for key in model.state_dict())
    with torch.no_grad():
        crop_output = model.forward_crop(torch.zeros(1, 3, 32, 32))
    assert "curve_heatmap" not in crop_output


def test_curve_loss_masks_missing_channels_and_has_finite_gradients():
    logits = torch.zeros((2, 3, 8, 8), requires_grad=True)
    target = torch.zeros_like(logits)
    target[0, 0, 3, :] = 1.0
    target[0, 2, :, 5] = 1.0
    weight = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])

    loss = _curve_loss(logits, target, weight)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[0, 1]) == 0
    assert torch.count_nonzero(logits.grad[1]) == 0


def test_curve_loss_is_zero_when_batch_has_no_valid_curves():
    logits = torch.randn((2, 3, 8, 8), requires_grad=True)
    target = torch.zeros_like(logits)
    weight = torch.zeros((2, 3))

    loss = _curve_loss(logits, target, weight)

    assert loss.item() == pytest.approx(0.0)


def test_tiny_named_curve_head_overfits_independent_channels():
    torch.manual_seed(4)
    head = NamedCurveHead(
        in_channels=(4, 4, 4, 4),
        neck_channels=4,
        head_channels=4,
        num_curves=2,
    )
    features = [
        torch.randn(1, 4, 16, 16),
        torch.randn(1, 4, 8, 8),
        torch.randn(1, 4, 4, 4),
        torch.randn(1, 4, 2, 2),
    ]
    target = torch.zeros((1, 2, 16, 16))
    target[0, 0, 4, 2:14] = 1.0
    target[0, 1, 2:14, 11] = 1.0
    weight = torch.ones((1, 2))
    optimizer = torch.optim.Adam(head.parameters(), lr=2e-2)

    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        logits = head(features)
        loss = _curve_loss(
            logits,
            target,
            weight,
            bce_weight=1.0,
            dice_weight=1.0,
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    assert losses[-1] < 0.35 * losses[0]
    probability = torch.sigmoid(head(features)).detach().numpy()[0]
    assert probability[0, 4, 8] > probability[0, 8, 11]
    assert probability[1, 8, 11] > probability[1, 4, 8]


def test_joint_model_curve_loss_wiring_backpropagates():
    cfg = _model_config("heatmap", curves=True)
    model = DoperNet(cfg)
    output = model(
        torch.randn(2, 3, cfg.det_input_size, cfg.det_input_size),
        torch.randn(2, 3, cfg.crop_size, cfg.crop_size),
    )
    curve_target = torch.zeros_like(output["curve_heatmap"])
    curve_target[:, 0, 3, :] = 1.0
    curve_target[:, 1, :, 4] = 1.0
    batch = {
        "heatmap": torch.zeros_like(output["heatmap"]),
        "wh": torch.zeros_like(output["wh"]),
        "offset": torch.zeros_like(output["offset"]),
        "reg_weight": torch.zeros((2, 1, output["wh"].shape[2], output["wh"].shape[3])),
        "mask_target": torch.zeros_like(output["mask"]),
        "mask_weight": torch.zeros(2),
        "kp_weight": torch.zeros((2, cfg.num_keypoints)),
        "has_kp": torch.zeros(2),
        "keypoint_heatmap": torch.zeros_like(output["keypoint_heatmap"]),
        "curve_heatmap": curve_target,
        "curve_weight": torch.ones((2, cfg.num_curves)),
        "has_curves": torch.ones(2),
    }

    loss, logs = compute_losses(output, batch, LossConfig())
    loss.backward()

    assert torch.isfinite(loss)
    assert logs["curve"] > 0
    assert model.curve_head.out.weight.grad is not None
    assert torch.isfinite(model.curve_head.out.weight.grad).all()
