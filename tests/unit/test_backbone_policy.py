# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from doper.config import (
    DEFAULT_DOPER_BACKBONE,
    DEFAULT_DOPER_BACKBONE_LR_MULTIPLIER,
    SUPPORTED_DOPER_BACKBONES,
    ModelConfig,
    TrainConfig,
)
from doper.training.recipe import validate_backbone_policy


def test_new_doper_models_default_to_dinov3_convnext_small() -> None:
    assert DEFAULT_DOPER_BACKBONE == "convnext_small.dinov3_lvd1689m"
    assert ModelConfig().backbone == DEFAULT_DOPER_BACKBONE
    assert TrainConfig().backbone_lr_multiplier == pytest.approx(
        DEFAULT_DOPER_BACKBONE_LR_MULTIPLIER
    )
    assert pytest.approx(0.1) == DEFAULT_DOPER_BACKBONE_LR_MULTIPLIER


@pytest.mark.parametrize("backbone", SUPPORTED_DOPER_BACKBONES)
def test_standard_backbone_passes_policy(backbone) -> None:
    validate_backbone_policy(backbone, False)


def test_nonstandard_backbone_requires_explicit_opt_out() -> None:
    with pytest.raises(ValueError, match="allow-nonstandard-backbone"):
        validate_backbone_policy("resnet34", False)

    validate_backbone_policy("resnet34", True)


def test_every_doper_training_recipe_states_its_backbone_policy() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    for path in sorted(config_dir.rglob("doper*.yaml")):
        recipe = yaml.safe_load(path.read_text())
        assert "backbone" in recipe, f"{path.name} must state its backbone"
        if recipe["backbone"] not in SUPPORTED_DOPER_BACKBONES:
            assert recipe.get("allow_nonstandard_backbone") is True, (
                f"{path.name} uses a nonstandard backbone without the explicit "
                "legacy/ablation opt-out"
            )
