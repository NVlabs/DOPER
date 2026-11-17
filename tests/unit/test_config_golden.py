# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The full default config equals the source defaults plus the intended deltas.

``golden/config_defaults.json`` is the source default (from the research source snapshot)
``DoperConfig`` with exactly these new-recipe changes applied:
``model.backbone = convnext_small.dinov3_lvd1689m``, ``model.keypoint_head =
heatmap`` and ``model.crop_size = 512``, plus the disabled-by-default rotation
auxiliary settings. Any other default change fails here.
"""

import json
from dataclasses import asdict
from pathlib import Path

from doper.config import DataConfig, DoperConfig

GOLDEN_PATH = Path(__file__).parent / "golden" / "config_defaults.json"


def test_full_default_config_matches_golden():
    expected = json.loads(GOLDEN_PATH.read_text())

    actual_text = json.dumps(
        asdict(DoperConfig(data=DataConfig(root="."))), indent=2, sort_keys=True
    )

    assert json.loads(actual_text) == expected
    # Text equality also catches int/float type drift.
    assert actual_text + "\n" == GOLDEN_PATH.read_text()


def test_golden_records_only_the_intended_model_deltas():
    golden = json.loads(GOLDEN_PATH.read_text())

    model = golden["model"]

    assert model["backbone"] == "convnext_small.dinov3_lvd1689m"
    assert model["keypoint_head"] == "heatmap"
    assert (model["det_input_size"], model["crop_size"], model["letterbox"]) == (640, 512, True)
