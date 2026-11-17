# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from tiny_coco import curve_fields, tiny_model_config

from doper.config import DataConfig, DoperConfig, ModelConfig, TrainConfig
from doper.infer import load_model
from doper.model import DoperNet, TimmFeatureBackbone
from doper.training.checkpoints import load_init_weights
from doper.training.schedule import build_optimizer


def _model_config(keypoint_head: str) -> ModelConfig:
    return tiny_model_config(
        num_classes=2,
        num_keypoints=3,
        det_input_size=96,
        letterbox=False,
        neck_channels=8,
        head_channels=8,
        crop_size=32,
        crop_pad_factor=1.1,
        keypoint_head=keypoint_head,
        gau_hidden_dims=8,
        gau_s=4,
        gau_expansion=2,
    )


def _config(model: ModelConfig) -> DoperConfig:
    return DoperConfig(data=DataConfig(root="."), model=model)


def _curve_model_config(keypoint_head: str = "heatmap") -> ModelConfig:
    cfg = _model_config(keypoint_head)
    for name, value in curve_fields(("rim", "seam"), "abc123", curve_head_channels=8).items():
        setattr(cfg, name, value)
    return cfg


def test_load_model_supports_legacy_simcc_checkpoint_without_cfg(tmp_path):
    source_cfg = _model_config("simcc")
    source_model = DoperNet(source_cfg)
    source_model.eval()
    checkpoint = tmp_path / "legacy_simcc.pth"
    torch.save({"model": source_model.state_dict()}, checkpoint)
    requested_cfg = _curve_model_config("heatmap")
    load_cfg = _config(requested_cfg)

    loaded = load_model(str(checkpoint), load_cfg, "cpu")

    assert load_cfg.model.keypoint_head == "simcc"
    assert load_cfg.model.curve_head_enabled is False
    assert loaded.cfg.keypoint_head == "simcc"
    assert not hasattr(loaded, "curve_head")
    assert not loaded.training
    torch.testing.assert_close(
        loaded.state_dict()["kpt_head.cls_x.weight"],
        source_model.state_dict()["kpt_head.cls_x.weight"],
    )
    det_image = torch.randn(1, 3, source_cfg.det_input_size, source_cfg.det_input_size)
    crop_image = torch.randn(1, 3, source_cfg.crop_size, source_cfg.crop_size)
    with torch.no_grad():
        expected = source_model(det_image, crop_image)
        actual = loaded(det_image, crop_image)
    assert expected.keys() == actual.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])


def test_load_model_uses_heatmap_checkpoint_cfg(tmp_path):
    source_cfg = _model_config("heatmap")
    source_model = DoperNet(source_cfg)
    source_model.eval()
    checkpoint = tmp_path / "heatmap.pth"
    torch.save(
        {
            "model": source_model.state_dict(),
            "cfg": {"model": source_cfg.__dict__},
        },
        checkpoint,
    )
    load_cfg = _config(
        tiny_model_config(
            backbone="resnet34",
            num_classes=1,
            num_keypoints=1,
            det_input_size=64,
            crop_size=64,
            keypoint_head="simcc",
        )
    )

    loaded = load_model(str(checkpoint), load_cfg, "cpu")

    assert load_cfg.model.backbone == "resnet18"
    assert load_cfg.model.keypoint_head == "heatmap"
    assert load_cfg.model.num_keypoints == 3
    assert load_cfg.model.det_input_size == 96
    assert load_cfg.model.letterbox is False
    assert load_cfg.model.crop_pad_factor == 1.1
    assert not loaded.training
    torch.testing.assert_close(
        loaded.state_dict()["kpt_head.out.weight"],
        source_model.state_dict()["kpt_head.out.weight"],
    )
    det_image = torch.randn(1, 3, source_cfg.det_input_size, source_cfg.det_input_size)
    crop_image = torch.randn(1, 3, source_cfg.crop_size, source_cfg.crop_size)
    with torch.no_grad():
        expected = source_model(det_image, crop_image)
        actual = loaded(det_image, crop_image)
    assert expected.keys() == actual.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])


def test_load_model_supports_raw_legacy_state_dict(tmp_path):
    source_cfg = _model_config("simcc")
    source_model = DoperNet(source_cfg)
    checkpoint = tmp_path / "raw_legacy.pth"
    torch.save(source_model.state_dict(), checkpoint)

    loaded = load_model(str(checkpoint), _config(_model_config("heatmap")), "cpu")

    assert loaded.cfg.keypoint_head == "simcc"
    assert loaded.cfg.curve_head_enabled is False


def test_load_model_restores_curve_checkpoint_metadata(tmp_path):
    source_cfg = _curve_model_config("heatmap")
    source_model = DoperNet(source_cfg)
    checkpoint = tmp_path / "curve_checkpoint.pth"
    torch.save(
        {
            "model": source_model.state_dict(),
            "cfg": {"model": source_cfg.__dict__},
        },
        checkpoint,
    )

    loaded = load_model(str(checkpoint), _config(_model_config("simcc")), "cpu")

    assert loaded.cfg.curve_head_enabled is True
    assert loaded.cfg.curve_names == ("rim", "seam")
    assert loaded.cfg.curve_geometry_sha256 == "abc123"
    assert loaded.state_dict()["curve_head.out.weight"].shape[0] == 2


def test_init_from_allows_simcc_to_heatmap_head_replacement(tmp_path):
    source_cfg = _model_config("simcc")
    source_model = DoperNet(source_cfg)
    checkpoint = tmp_path / "simcc_init.pth"
    torch.save(
        {
            "model": source_model.state_dict(),
            "cfg": {"model": source_cfg.__dict__},
        },
        checkpoint,
    )
    target_cfg = _config(_model_config("heatmap"))
    target_model = DoperNet(target_cfg.model)

    load_init_weights(target_model, target_cfg, str(checkpoint), "cpu")

    torch.testing.assert_close(
        target_model.state_dict()["backbone.stem.0.weight"],
        source_model.state_dict()["backbone.stem.0.weight"],
    )


def test_init_from_legacy_checkpoint_initializes_only_new_curve_head(tmp_path):
    source_cfg = _model_config("heatmap")
    source_model = DoperNet(source_cfg)
    checkpoint = tmp_path / "legacy_heatmap_init.pth"
    torch.save(
        {
            "model": source_model.state_dict(),
            "cfg": {"model": source_cfg.__dict__},
        },
        checkpoint,
    )
    target_cfg = _config(_curve_model_config("heatmap"))
    target_model = DoperNet(target_cfg.model)
    curve_before = {
        key: value.clone()
        for key, value in target_model.state_dict().items()
        if key.startswith("curve_head.")
    }

    load_init_weights(target_model, target_cfg, str(checkpoint), "cpu")

    torch.testing.assert_close(
        target_model.state_dict()["backbone.stem.0.weight"],
        source_model.state_dict()["backbone.stem.0.weight"],
    )
    for key, expected in curve_before.items():
        torch.testing.assert_close(target_model.state_dict()[key], expected)


def test_curve_checkpoint_requires_exact_named_schema(tmp_path):
    source_cfg = _curve_model_config("heatmap")
    source_model = DoperNet(source_cfg)
    checkpoint = tmp_path / "curves.pth"
    torch.save(
        {
            "model": source_model.state_dict(),
            "cfg": {"model": source_cfg.__dict__},
        },
        checkpoint,
    )
    target_cfg = _config(_curve_model_config("heatmap"))
    target_cfg.model.curve_names = ("seam", "rim")
    target_model = DoperNet(target_cfg.model)

    with pytest.raises(RuntimeError, match="curve schema/checkpoint mismatch"):
        load_init_weights(target_model, target_cfg, str(checkpoint), "cpu")


def test_init_from_rejects_checkpoint_with_no_shared_weights(tmp_path):
    target_cfg = _config(_model_config("simcc"))
    target_model = DoperNet(target_cfg.model)
    checkpoint = tmp_path / "head_only.pth"
    torch.save(
        {"model": {"kpt_head.cls_x.weight": target_model.state_dict()["kpt_head.cls_x.weight"]}},
        checkpoint,
    )

    with pytest.raises(RuntimeError, match="essentially no meaningful"):
        load_init_weights(target_model, target_cfg, str(checkpoint), "cpu")


def test_dinov3_convnext_features_have_expected_pyramid():
    backbone = TimmFeatureBackbone("convnext_tiny.dinov3_lvd1689m", pretrained=False)

    with torch.no_grad():
        features = backbone(torch.zeros(1, 3, 64, 64))

    assert backbone.out_channels == (96, 192, 384, 768)
    assert [tuple(feature.shape[-2:]) for feature in features] == [
        (16, 16),
        (8, 8),
        (4, 4),
        (2, 2),
    ]


def test_optimizer_applies_backbone_lr_multiplier():
    model = DoperNet(_model_config("simcc"))
    cfg = TrainConfig(lr=5e-4, backbone_lr_multiplier=0.1)

    optimizer = build_optimizer(model, cfg)

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-5)
    assert optimizer.param_groups[0]["lr_scale"] == pytest.approx(0.1)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(5e-4)
    assert optimizer.param_groups[1]["lr_scale"] == pytest.approx(1.0)
