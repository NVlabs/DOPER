# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
from tiny_coco import coco_annotation, curve_fields, tiny_model_config, write_coco_split

from doper.cad_curves import ProjectedCurves, save_projected_curves
from doper.config import AugConfig, DataConfig, DoperConfig
from doper.dataset import DoperDataset


def _config(root, enabled=True):
    return DoperConfig(
        data=DataConfig(root=str(root)),
        model=tiny_model_config(
            num_keypoints=1,
            det_input_size=32,
            crop_size=32,
            crop_pad_factor=1.0,
            **(curve_fields(("line", "outside"), "schema-hash") if enabled else {}),
        ),
        aug=AugConfig(enabled=False),
    )


def _write_dataset(root, with_curves=True):
    split = root / "train"
    (split / "curves").mkdir(parents=True)
    annotation = coco_annotation(1, 1, [8.0, 8.0, 16.0, 16.0], 256.0, [16.0, 16.0, 2])
    if with_curves:
        projected = ProjectedCurves(
            points_xy=np.asarray(
                [[8.0, 16.0], [24.0, 16.0], [-20.0, -20.0], [-10.0, -10.0]],
                dtype=np.float32,
            ),
            offsets=np.asarray([0, 2, 4], dtype=np.int64),
            valid=np.asarray([1.0, 1.0], dtype=np.float32),
            schema_sha256="schema-hash",
        )
        save_projected_curves(str(split / "curves" / "00000001.npz"), projected)
        annotation["curve_file"] = "curves/00000001.npz"
        annotation["curve_schema_sha256"] = "schema-hash"
    write_coco_split(
        split,
        [{"id": 1, "file_name": "frame.png", "width": 32, "height": 32}],
        [annotation],
        [{"id": 1, "name": "part"}],
        {"images/frame.png": np.full((32, 32, 3), 127, dtype=np.uint8)},
    )


def test_dataset_warps_named_curves_with_the_crop_affine(tmp_path):
    _write_dataset(tmp_path)
    dataset = DoperDataset(_config(tmp_path), "train")

    sample = dataset[0]

    assert sample["curve_heatmap"].shape == (2, 8, 8)
    np.testing.assert_array_equal(sample["curve_weight"].numpy(), [1.0, 0.0])
    assert sample["has_curves"].item() == 1.0
    first = sample["curve_heatmap"][0].numpy()
    assert first.max() > 0.9
    assert first[4].max() > first[0].max()


def test_dataset_masks_curve_loss_when_annotation_has_no_sidecar(tmp_path):
    _write_dataset(tmp_path, with_curves=False)
    dataset = DoperDataset(_config(tmp_path), "train")

    sample = dataset[0]

    assert sample["has_curves"].item() == 0.0
    assert sample["curve_weight"].count_nonzero().item() == 0
    assert sample["curve_heatmap"].count_nonzero().item() == 0


def test_curve_disabled_dataset_keeps_legacy_sample_keys(tmp_path):
    _write_dataset(tmp_path, with_curves=False)
    dataset = DoperDataset(_config(tmp_path, enabled=False), "train")

    sample = dataset[0]

    assert "curve_heatmap" not in sample
    assert "curve_weight" not in sample
    assert "has_curves" not in sample
