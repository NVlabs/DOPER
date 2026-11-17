# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fully hidden robustness renders must never become training targets."""

from __future__ import annotations

import numpy as np
from tiny_coco import coco_annotation, tiny_model_config, write_coco_split

from doper.config import AugConfig, DataConfig, DoperConfig
from doper.dataset import DoperDataset


def _annotation(ann_id, image_id, **extra):
    annotation = coco_annotation(ann_id, image_id, [8.0, 8.0, 16.0, 16.0], 256.0, [16.0, 16.0, 2])
    return {**annotation, **extra}


def test_annotations_without_a_visible_target_are_skipped(tmp_path):
    hidden = _annotation(
        2,
        2,
        bbox=[0, 0, 0, 0],
        area=0,
        robustness={"valid_detection_target": False, "visible_pixels": 0},
    )
    write_coco_split(
        tmp_path / "train",
        [
            {"id": 1, "file_name": "visible.png", "width": 32, "height": 32},
            {"id": 2, "file_name": "hidden.png", "width": 32, "height": 32},
        ],
        [_annotation(1, 1, robustness={"valid_detection_target": True}), hidden],
        [{"id": 1, "name": "part"}],
        {
            f"images/{name}": np.full((32, 32, 3), 127, np.uint8)
            for name in ("visible.png", "hidden.png")
        },
    )
    cfg = DoperConfig(
        data=DataConfig(root=str(tmp_path)),
        model=tiny_model_config(
            num_keypoints=1,
            det_input_size=32,
            crop_size=32,
            crop_pad_factor=1.0,
        ),
        aug=AugConfig(enabled=False),
    )
    dataset = DoperDataset(cfg, "train")
    assert dataset.ids == [1]
    assert len(dataset) == 1
