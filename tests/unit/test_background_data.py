# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the hard-background data pipeline.

Tests of the SPEED+ fine-tuning script helpers wait for that
script's port.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from tiny_coco import coco_annotation, tiny_model_config, write_coco_split

import doper.background_data as background_data
from doper.background_data import (
    BackgroundImagePool,
    BackgroundNegativeDataset,
    blend_foreground,
    load_fp_boxes,
    pool_scale,
    sample_crop_window,
    soften_composite_edge,
    worker_index_rng,
)
from doper.config import (
    AugConfig,
    BackgroundConfig,
    DataConfig,
    DoperConfig,
    LossConfig,
)
from doper.dataset import DoperDataset
from doper.losses import compute_losses, neg_loss
from doper.training.data import (
    RatioBatchSampler,
    build_background_train_loader,
    build_train_set,
)


# ---------------------------------------------------------------------------
# Fixtures: tiny synthetic COCO root + tiny background dir
# ---------------------------------------------------------------------------
def _make_background_dir(root: str, names) -> None:
    os.makedirs(root, exist_ok=True)
    for i, name in enumerate(names):
        yy, xx = np.indices((80, 120))
        img = np.empty((80, 120, 3), dtype=np.uint8)
        img[..., 0] = (xx + 31 * i) % 255
        img[..., 1] = (yy * 2 + 17 * i) % 255
        img[..., 2] = (xx + yy + 23 * i) % 255
        cv2.imwrite(os.path.join(root, name), img)
    splits = {"train": list(names[:2]), "val": [names[2]], "test": [names[3]]}
    with open(os.path.join(root, "splits.json"), "w") as f:
        json.dump(splits, f)


def _make_synth_root(root: str, num_keypoints: int) -> None:
    img = np.full((64, 64, 3), 200, dtype=np.uint8)
    img[20:44, 20:44] = (30, 60, 90)  # an "object" block
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:44, 20:44] = 255
    keypoints = []
    for k in range(num_keypoints):
        keypoints += [22.0 + k % 20, 22.0 + k % 20, 2]
    annotation = coco_annotation(
        0, 0, [20.0, 20.0, 24.0, 24.0], 576.0, keypoints, iscrowd=0, mask_file="masks/000000.png"
    )
    write_coco_split(
        Path(root) / "train",
        [{"id": 0, "file_name": "000000.png", "width": 64, "height": 64}],
        [annotation],
        [{"id": 1, "name": "plug"}],
        {"images/000000.png": img, "masks/000000.png": mask},
    )


def _cfg(tmp_path, num_keypoints=2, det_size=64, crop_size=32) -> DoperConfig:
    synth = os.path.join(tmp_path, "synth")
    bg = os.path.join(tmp_path, "bg")
    _make_synth_root(synth, num_keypoints)
    _make_background_dir(bg, ["a.png", "b.png", "c.png", "d.png"])
    return DoperConfig(
        data=DataConfig(root=synth),
        model=tiny_model_config(
            num_classes=1,
            num_keypoints=num_keypoints,
            det_input_size=det_size,
            crop_size=crop_size,
            keypoint_head="heatmap",
            keypoint_heatmap_sigma=1.5,
        ),
        aug=AugConfig(enabled=False),
        background=BackgroundConfig(
            enabled=True,
            background_dir=bg,
            splits_json=os.path.join(bg, "splits.json"),
        ),
    )


# ---------------------------------------------------------------------------
# Crop windows
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hw", [(80, 120), (120, 80), (200, 200)])
def test_sample_crop_window_stays_in_bounds(hw):
    h, w = hw
    cfg_bg = BackgroundConfig()
    rng = np.random.default_rng(0)
    for _ in range(50):
        x, y, cw, ch = sample_crop_window(h, w, cfg_bg, rng)
        assert x >= 0 and y >= 0
        assert x + cw <= w and y + ch <= h
        assert cw >= 16 and ch >= 16


def test_fp_box_scaling_uses_original_and_decoded_dimensions(tmp_path):
    path = os.path.join(tmp_path, "large.png")
    original = np.zeros((101, 203, 3), dtype=np.uint8)
    cv2.imwrite(path, original)
    pool = BackgroundImagePool([path], max_side=100)

    decoded = pool.get_bgr(0)
    assert decoded.shape[:2] == (50, 100)
    scale = pool_scale(pool, 0)
    assert np.allclose(
        scale,
        np.asarray([100 / 203, 50 / 101, 100 / 203, 50 / 101]),
    )
    # The old max-box-coordinate heuristic returned 1.0 for this in-bounds box.
    assert np.allclose(
        np.asarray([4, 5, 40, 30], dtype=np.float32) * scale,
        np.asarray([400 / 203, 250 / 101, 4000 / 203, 1500 / 101]),
    )


def test_background_pool_cache_evicts_least_recently_used(tmp_path, monkeypatch):
    monkeypatch.setattr(background_data, "MAX_CACHED_BACKGROUND_BYTES", 2 * 4 * 4 * 3)
    paths = []
    for i in range(3):
        path = os.path.join(tmp_path, f"bg{i}.png")
        cv2.imwrite(path, np.full((4, 4, 3), i, dtype=np.uint8))
        paths.append(path)
    pool = BackgroundImagePool(paths)

    first = pool.get_bgr(0)
    pool.get_bgr(1)
    assert pool.get_bgr(0) is first
    pool.get_bgr(2)

    assert list(pool._cache) == [0, 2]
    assert pool.get_bgr(1)[0, 0, 0] == 1


def test_load_fp_boxes_accepts_existing_json_shape(tmp_path):
    path = os.path.join(tmp_path, "fp.json")
    with open(path, "w") as f:
        json.dump(
            {
                "records": [
                    {
                        "file": "/old/eval/path/a.png",
                        "top_bbox_xyxy": [1.0, 2.0, 30.0, 40.0],
                    }
                ],
            },
            f,
        )
    boxes = load_fp_boxes(path)
    assert set(boxes) == {"a.png"}
    assert np.array_equal(boxes["a.png"], np.asarray([1, 2, 30, 40], np.float32))


def test_worker_index_rng_is_reproducible_and_worker_aware(monkeypatch):
    class Worker:
        def __init__(self, worker_id):
            self.id = worker_id

    monkeypatch.setattr(background_data, "get_worker_info", lambda: Worker(2))
    first = worker_index_rng(17, 5, epoch=3, stream=1, occurrence=4).integers(0, 2**31, size=8)
    again = worker_index_rng(17, 5, epoch=3, stream=1, occurrence=4).integers(0, 2**31, size=8)
    monkeypatch.setattr(background_data, "get_worker_info", lambda: Worker(3))
    other_worker = worker_index_rng(17, 5, epoch=3, stream=1, occurrence=4).integers(
        0, 2**31, size=8
    )
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other_worker)


# ---------------------------------------------------------------------------
# Negative targets
# ---------------------------------------------------------------------------
def test_background_negative_targets_are_zero(tmp_path):
    cfg = _cfg(tmp_path)
    ds = BackgroundNegativeDataset(cfg)
    assert len(ds) > 0
    s = ds[0]
    assert float(s["heatmap"].max()) == 0.0
    assert float(s["reg_weight"].sum()) == 0.0
    assert float(s["mask_target"].sum()) == 0.0
    assert float(s["mask_weight"]) == 1.0
    assert float(s["has_kp"]) == 0.0
    assert float(s["kp_weight"].sum()) == 0.0
    assert float(s["keypoint_heatmap"].sum()) == 0.0
    for key in ("det_image", "crop_image"):
        assert torch.isfinite(s[key]).all()


def test_background_negative_crops_vary(tmp_path):
    cfg = _cfg(tmp_path)
    ds = BackgroundNegativeDataset(cfg)
    a = ds[0]["det_image"]
    diffs = [not torch.equal(a, ds[0]["det_image"]) for _ in range(8)]
    assert any(diffs)  # dynamic sampling => not all identical


def test_background_negative_sequence_reproducible_by_epoch(tmp_path):
    cfg = _cfg(tmp_path)
    first_ds = BackgroundNegativeDataset(cfg)
    second_ds = BackgroundNegativeDataset(cfg)

    first = [first_ds[0]["det_image"] for _ in range(4)]
    second = [second_ds[0]["det_image"] for _ in range(4)]
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    assert any(not torch.equal(first[0], sample) for sample in first[1:])

    first_ds.set_epoch(1)
    second_ds.set_epoch(1)
    next_epoch_first = first_ds[0]["det_image"]
    next_epoch_second = second_ds[0]["det_image"]
    assert torch.equal(next_epoch_first, next_epoch_second)
    assert not torch.equal(first[0], next_epoch_first)


# ---------------------------------------------------------------------------
# Compositing preserves foreground / replaces background
# ---------------------------------------------------------------------------
def test_blend_foreground_keeps_object_replaces_background():
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[..., 0] = 255  # red foreground image
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[12:20, 12:20] = 1  # central object
    background = np.zeros((32, 32, 3), dtype=np.uint8)
    background[..., 2] = 255  # blue background crop
    cfg_bg = BackgroundConfig(
        alpha_feather_sigma_min=0.0, alpha_feather_sigma_max=0.0, composite_erode_px=0
    )
    out_rgb, out_bgr = blend_foreground(
        rgb, bgr, mask, background, cfg_bg, np.random.default_rng(0)
    )
    assert out_rgb.shape == rgb.shape and out_rgb.dtype == np.uint8
    # Object center keeps the red foreground; a corner becomes the blue bg.
    assert out_rgb[15, 15, 0] > 200
    assert out_rgb[0, 0, 2] > 200 and out_rgb[0, 0, 0] < 40


def test_force_composite_dataset_keeps_positive_labels(tmp_path):
    cfg = _cfg(tmp_path)
    ds = DoperDataset(cfg, "train", force_composite=True)
    s = ds[0]
    assert float(s["has_kp"]) == 1.0
    assert float(s["heatmap"].max()) > 0.0  # detection target present
    assert float(s["reg_weight"].sum()) > 0.0
    assert float(s["kp_weight"].sum()) > 0.0
    assert torch.isfinite(s["det_image"]).all()


def test_force_composite_sequence_is_reproducible_but_dynamic(tmp_path):
    cfg = _cfg(tmp_path)
    first_ds = DoperDataset(cfg, "train", force_composite=True)
    second_ds = DoperDataset(cfg, "train", force_composite=True)
    first = [first_ds[0]["det_image"] for _ in range(4)]
    second = [second_ds[0]["det_image"] for _ in range(4)]
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    assert any(not torch.equal(first[0], sample) for sample in first[1:])


def test_stochastic_composite_preserves_labels_and_changes_pixels(tmp_path):
    cfg = _cfg(tmp_path)
    plain = DoperDataset(cfg, "train", composite_prob=0.0)[0]
    composited = DoperDataset(cfg, "train", composite_prob=1.0)[0]
    assert not torch.equal(plain["det_image"], composited["det_image"])
    for key in ("heatmap", "wh", "offset", "reg_weight", "kp_weight"):
        assert torch.equal(plain[key], composited[key])


def test_keypoint_only_dataset_disables_detection_and_mask_loss(tmp_path):
    cfg = _cfg(tmp_path)
    sample = DoperDataset(cfg, "train", keypoint_only=True)[0]
    assert float(sample["det_weight"]) == 0.0
    assert float(sample["mask_weight"]) == 0.0
    assert float(sample["kp_weight"].sum()) > 0.0
    assert float(sample["has_kp"]) == 1.0


def test_soften_composite_edge_only_changes_boundary_band():
    rng = np.random.default_rng(4)
    image = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
    mask = np.zeros((48, 48), dtype=np.uint8)
    mask[14:34, 14:34] = 1
    softened = soften_composite_edge(image, mask, sigma=1.2, width=2)
    assert np.array_equal(softened[0:8, 0:8], image[0:8, 0:8])
    assert np.array_equal(softened[20:28, 20:28], image[20:28, 20:28])
    assert not np.array_equal(softened[12:17, 12:17], image[12:17, 12:17])


def test_natural_mix_composites_primary_root_only(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.background.composite_prob = 0.08
    mix_root = os.path.join(tmp_path, "mix")
    _make_synth_root(mix_root, cfg.model.num_keypoints)
    train_set = build_train_set(cfg, [cfg.data.root, mix_root])
    assert train_set.datasets[0].composite_prob == pytest.approx(0.08)
    assert train_set.datasets[1].composite_prob == 0.0


def test_train_set_marks_only_requested_mix_root_keypoint_only(tmp_path):
    cfg = _cfg(tmp_path)
    mix_root = os.path.join(tmp_path, "keypoint_only_mix")
    _make_synth_root(mix_root, cfg.model.num_keypoints)
    train_set = build_train_set(
        cfg,
        [cfg.data.root, mix_root],
        keypoint_only_roots=[mix_root],
    )
    assert not train_set.datasets[0].keypoint_only
    assert train_set.datasets[1].keypoint_only


# ---------------------------------------------------------------------------
# Ratio batch sampler
# ---------------------------------------------------------------------------
def test_ratio_batch_sampler_exact_composition():
    sizes = [100, 20, 20]
    per_batch = [3, 1, 1]
    sampler = RatioBatchSampler(sizes, per_batch, num_batches=5, seed=0)
    offs = [0, 100, 120]
    batches = list(sampler)
    assert len(batches) == 5
    for batch in batches:
        assert len(batch) == sum(per_batch)
        n_pos = sum(offs[0] <= i < offs[1] for i in batch)
        n_neg = sum(offs[1] <= i < offs[2] for i in batch)
        n_comp = sum(offs[2] <= i < 140 for i in batch)
        assert (n_pos, n_neg, n_comp) == (3, 1, 1)
        assert all(0 <= i < 140 for i in batch)


def test_ratio_batch_sampler_rejects_empty_stream():
    with pytest.raises(ValueError):
        RatioBatchSampler([10, 0, 5], [1, 1, 1], num_batches=2)


def test_ratio_batch_sampler_varies_across_epochs():
    sampler = RatioBatchSampler([100, 20, 20], [3, 1, 1], num_batches=3, seed=0)
    first = [list(b) for b in sampler]
    second = [list(b) for b in sampler]
    assert first != second  # epoch counter reshuffles


def test_background_loader_preserves_explicit_positive_root_ratio(tmp_path):
    cfg = _cfg(tmp_path)
    mix_root = os.path.join(tmp_path, "mix")
    _make_synth_root(mix_root, cfg.model.num_keypoints)
    positive_set = build_train_set(cfg, [cfg.data.root, mix_root])
    cfg.train.batch_size = 4
    cfg.train.num_workers = 0
    cfg.background.positive_per_batch = 2
    cfg.background.negative_per_batch = 1
    cfg.background.composite_per_batch = 1

    loader = build_background_train_loader(
        cfg,
        positive_set,
        "cpu",
        positive_per_source=[1, 1],
    )

    assert isinstance(loader.batch_sampler, RatioBatchSampler)
    assert loader.batch_sampler.per_batch == [1, 1, 1, 1]
    assert len(loader.dataset.datasets) == 4
    batch = next(iter(loader))
    assert tuple(batch["det_weight"].shape) == (4,)


# ---------------------------------------------------------------------------
# Loss behavior on mixed and all-negative batches
# ---------------------------------------------------------------------------
def _fake_heatmap_batch(has_kp_flags, mask_flags, hf=8, k=2, hs=4):
    b = len(has_kp_flags)
    g = torch.Generator().manual_seed(0)
    outputs = {
        "heatmap": torch.randn(b, 1, hf, hf, generator=g),
        "wh": torch.randn(b, 2, hf, hf, generator=g),
        "offset": torch.randn(b, 2, hf, hf, generator=g),
        "mask": torch.randn(b, 1, hf, hf, generator=g),
        "keypoint_heatmap": torch.randn(b, k, hs, hs, generator=g),
    }
    heatmap_t = torch.zeros(b, 1, hf, hf)
    reg_w = torch.zeros(b, 1, hf, hf)
    kp_w = torch.zeros(b, k)
    for i, flag in enumerate(has_kp_flags):
        if flag:
            heatmap_t[i, 0, 2, 2] = 1.0
            reg_w[i, 0, 2, 2] = 1.0
            kp_w[i] = 1.0
    kp_hm = torch.zeros(b, k, hs, hs)
    kp_hm[:, :, 0, 0] = 1.0
    batch = {
        "heatmap": heatmap_t,
        "wh": torch.zeros(b, 2, hf, hf),
        "offset": torch.zeros(b, 2, hf, hf),
        "reg_weight": reg_w,
        "mask_target": torch.zeros(b, 1, hf, hf),
        "mask_weight": torch.tensor(mask_flags, dtype=torch.float32),
        "keypoint_heatmap": kp_hm,
        "kp_weight": kp_w,
        "has_kp": torch.tensor([float(f) for f in has_kp_flags]),
    }
    return outputs, batch


def test_losses_mixed_batch_is_finite():
    outputs, batch = _fake_heatmap_batch([1, 0], [1.0, 1.0])
    total, logs = compute_losses(outputs, batch, LossConfig())
    assert torch.isfinite(total)
    assert all(torch.isfinite(logs[k]) for k in ("hm", "mask", "kp"))


def test_losses_all_negative_batch_is_finite():
    outputs, batch = _fake_heatmap_batch([0, 0], [1.0, 1.0])
    total, logs = compute_losses(outputs, batch, LossConfig())
    assert torch.isfinite(total)
    assert logs["kp"] == 0.0  # no keypoint supervision
    assert logs["hm"] > 0.0  # negatives still push the heatmap down


def test_focal_loss_is_finite_for_saturated_float16_probabilities():
    pred = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float16, requires_grad=True)
    gt = torch.tensor([[[[0.0, 1.0]]]], dtype=torch.float16)
    weight = torch.ones(1, dtype=torch.float16)

    loss = neg_loss(pred, gt, weight)
    loss.backward()

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_mask_loss_is_finite_for_saturated_256px_float16_mask():
    # 256**2 = 65,536 exceeds float16's largest finite value (65,504), so Dice
    # reductions must be performed in float32.
    outputs, batch = _fake_heatmap_batch([1], [1.0])
    logits = torch.full((1, 1, 256, 256), 20.0, dtype=torch.float16, requires_grad=True)
    outputs["mask"] = logits
    batch["mask_target"] = torch.ones_like(logits)
    batch["mask_weight"] = torch.ones(1, dtype=torch.float16)

    total, logs = compute_losses(outputs, batch, LossConfig())
    total.backward()

    assert total.dtype == torch.float32
    assert torch.isfinite(total)
    assert torch.isfinite(logs["mask"])
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_keypoint_only_batch_has_only_keypoint_loss():
    outputs, batch = _fake_heatmap_batch([1, 1], [0.0, 0.0])
    batch["det_weight"] = torch.zeros(2)
    total, logs = compute_losses(outputs, batch, LossConfig())
    assert torch.isfinite(total)
    assert logs["hm"] == 0.0
    assert logs["wh"] == 0.0
    assert logs["off"] == 0.0
    assert logs["mask"] == 0.0
    assert logs["kp"] > 0.0
    assert float(total) == pytest.approx(logs["kp"])
