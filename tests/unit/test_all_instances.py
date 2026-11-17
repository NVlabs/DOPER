# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cv2
import numpy as np
import pytest
from tiny_coco import coco_annotation, tiny_model_config, write_coco_split

from doper.config import AugConfig, DataConfig, DoperConfig
from doper.dataset import DoperDataset
from doper.training.data import supervision_counts


def dataset(
    tmp_path, mode="all", mask_mode="union", missing_mask=False, marked=False, partial_kp=False
):
    root = tmp_path / "data"
    base = root / "train"
    image = np.zeros((64, 64, 3), np.uint8)
    image[8:25, 8:25] = 80
    image[40:49, 40:49] = 200
    files = {"images/frame.png": image}
    anns = []
    for i, (x, y, w) in enumerate([(8, 8, 16), (40, 40, 8)], 1):
        mask = np.zeros((64, 64), np.uint8)
        mask[y : y + w + 1, x : x + w + 1] = 255
        files[f"masks/{i}.png"] = mask
        keypoints = [x + 2, y + 2, 2, x + w - 2, y + w - 2, 2]
        ann = coco_annotation(i, 5, [x, y, w, w], w * w, keypoints, mask_file=f"masks/{i}.png")
        if missing_mask and i == 2:
            ann.pop("mask_file")
        if partial_kp and i == 2:
            ann.pop("keypoints")
            ann.pop("num_keypoints")
        if marked and i == 2:
            ann["is_primary"] = True
        anns.append(ann)
    images = [{"id": 5, "file_name": "frame.png", "width": 64, "height": 64}]
    write_coco_split(base, images, anns, [{"id": 1}], files)
    cfg = DoperConfig(
        data=DataConfig(root=str(root), instance_sampling=mode, mask_target_mode=mask_mode),
        model=tiny_model_config(num_keypoints=2, det_input_size=64, crop_size=16),
        aug=AugConfig(enabled=False),
    )
    return DoperDataset(cfg, "train")


def test_all_instances_have_distinct_crops_all_boxes_and_union_mask(tmp_path):
    ds = dataset(tmp_path)
    assert len(ds) == 2 and ds.image_count == 1
    assert [ds.primary_annotation(i)["id"] for i in range(2)] == [1, 2]
    a, b = ds[0], ds[1]
    assert a["mask_target"][0, 3, 3] == 1 and a["mask_target"][0, 11, 11] == 1
    assert np.array_equal(a["mask_target"].numpy(), b["mask_target"].numpy())
    assert not np.array_equal(a["crop_image"].numpy(), b["crop_image"].numpy())
    assert supervision_counts(ds, 2) == (2, 2, 0)
    # Both centers are supervised, regardless of the crop's selected instance.
    assert int((a["heatmap"] == 1).sum()) == 2


def test_legacy_primary_behavior_is_unchanged(tmp_path):
    ds = dataset(tmp_path, mode="primary", mask_mode="primary")
    assert len(ds) == 1
    assert ds.primary_annotation(0)["id"] == 1
    sample = ds[0]
    assert sample["mask_target"][0, 3, 3] == 1 and sample["mask_target"][0, 11, 11] == 0


def test_partial_labels_disable_only_the_appropriate_heads(tmp_path):
    ds = dataset(tmp_path, missing_mask=True, partial_kp=True)
    assert ds[0]["mask_weight"] == 0 and ds[1]["mask_weight"] == 0
    assert ds[0]["has_kp"] == 1 and ds[1]["has_kp"] == 0
    assert supervision_counts(ds, 2) == (1, 0, 0)


def test_all_instance_supervision_counts_follow_each_crop_not_largest(tmp_path):
    ds = dataset(tmp_path, partial_kp=True)
    assert supervision_counts(ds, 2) == (1, 2, 1)
    assert ds[0]["has_kp"] == 1 and ds[1]["has_kp"] == 0
    assert ds[0]["mask_weight"] == 1 and ds[1]["mask_weight"] == 1


def test_preexpanded_marked_primary_is_not_expanded_again(tmp_path):
    ds = dataset(tmp_path, marked=True)
    assert len(ds) == 1 and ds.primary_annotation(0)["id"] == 2
    assert ds[0]["mask_target"][0, 3, 3] == 1 and ds[0]["mask_target"][0, 11, 11] == 1


def test_historical_config_defaults_do_not_hide_explicit_policy_changes():
    from doper.config import legacy_instance_defaults

    original = {"data": {"root": "old"}}
    old = legacy_instance_defaults(original)
    assert original == {"data": {"root": "old"}}
    assert (
        old["data"]["instance_sampling"] == "primary"
        and old["data"]["mask_target_mode"] == "primary"
    )
    new = legacy_instance_defaults(
        {"data": {"root": "old", "instance_sampling": "all", "mask_target_mode": "union"}}
    )
    assert new != old and new["data"]["instance_sampling"] == "all"


def test_all_instances_requires_consistent_union_masks(tmp_path):
    with pytest.raises(ValueError, match="union"):
        dataset(tmp_path, mask_mode="primary")


def test_declared_missing_mask_fails_instead_of_teaching_background(tmp_path):
    ds = dataset(tmp_path)
    (tmp_path / "data/train/masks/2.png").unlink()
    with pytest.raises(ValueError, match="missing or unreadable"):
        ds[0]


def test_small_backbone_batch_forward_backward_with_all_instance_targets(tmp_path):
    import torch
    from torch.utils.data import DataLoader

    from doper.losses import compute_losses
    from doper.model import DoperNet

    ds = dataset(tmp_path)
    ds.cfg.model.crop_size = 64
    ds.cfg.model.keypoint_head = "heatmap"
    ds.cfg.model.backbone = "convnext_small.dinov3_lvd1689m"
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            batch = next(iter(DataLoader(ds, batch_size=2)))
            model = DoperNet(ds.cfg.model).train()
            out = model(batch["det_image"], batch["crop_image"])
            loss, _ = compute_losses(out, batch, ds.cfg.loss)
            assert torch.isfinite(loss)
            loss.backward()
            assert any(p.grad is not None for p in model.parameters())
            assert all(
                torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
            )
    finally:
        torch.set_num_threads(previous)


def test_empty_visible_masks_are_valid_union_inputs(tmp_path):
    ds = dataset(tmp_path)
    cv2.imwrite(str(tmp_path / "data/train/masks/2.png"), np.zeros((64, 64), np.uint8))
    mask, weight = ds._load_union_mask(ds.anns_by_img[5], 64, 64)
    assert weight == 1 and mask[12, 12] == 1 and mask[44, 44] == 0
