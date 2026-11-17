# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""New-model recipe defaults and legacy-checkpoint metadata precedence."""

from dataclasses import asdict

import pytest
import torch

from doper.config import DataConfig, DoperConfig, ModelConfig, TrainConfig
from doper.infer import load_model
from doper.model import DoperNet

LEGACY_BACKBONE = "convnext_tiny.dinov3_lvd1689m"
LEGACY_CROP_SIZE = 256


@pytest.fixture(scope="module")
def legacy_tiny_simcc_checkpoint(tmp_path_factory):
    """Write a legacy checkpoint (Tiny, simcc, 256, no letterbox, 4 keypoints)."""
    model_cfg = ModelConfig(
        backbone=LEGACY_BACKBONE,
        pretrained=False,
        keypoint_head="simcc",
        crop_size=LEGACY_CROP_SIZE,
        det_input_size=64,
        num_keypoints=4,
        # Opposite of the current default so restoring it is observable.
        letterbox=False,
    )
    cfg = DoperConfig(data=DataConfig(root="."), model=model_cfg)
    model = DoperNet(model_cfg)
    path = tmp_path_factory.mktemp("legacy") / "legacy.pt"
    torch.save({"model": model.state_dict(), "cfg": asdict(cfg)}, path)
    return path


def test_new_defaults():
    m = ModelConfig()
    assert (
        m.backbone,
        m.keypoint_head,
        m.det_input_size,
        m.crop_size,
        m.num_keypoints,
        m.letterbox,
    ) == ("convnext_small.dinov3_lvd1689m", "heatmap", 640, 512, 64, True)
    assert TrainConfig().backbone_lr_multiplier == 0.1


def test_legacy_checkpoint_metadata_wins(legacy_tiny_simcc_checkpoint):
    model = load_model(legacy_tiny_simcc_checkpoint, DoperConfig(data=DataConfig(root=".")), "cpu")
    assert (model.cfg.backbone, model.cfg.keypoint_head, model.cfg.crop_size) == (
        LEGACY_BACKBONE,
        "simcc",
        LEGACY_CROP_SIZE,
    )
    assert (model.cfg.det_input_size, model.cfg.letterbox, model.cfg.num_keypoints) == (
        64,
        False,
        4,
    )
