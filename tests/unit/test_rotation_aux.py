# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The rotation-aux head: bins, soft targets, metrics and the checkpoint payload."""

import math

import numpy as np
import pytest
import torch
from torch.nn import functional

from doper.rotation_aux import (
    CosineClassifier,
    CosineRotationHead,
    RotationAuxPayload,
    farthest_rotation_bins,
    matrix_soft_targets,
    normalize_quaternions,
    pooled_rotation_features,
    rotation_aux_loss,
    rotation_errors_deg,
    rotation_metrics,
    soft_rotation_targets,
)
from doper.speedplus.pose import quaternion_to_object_camera

SIGMA_RAD = math.radians(15.0)


def random_quaternions(count: int, seed: int) -> np.ndarray:
    """Return ``count`` random unit scalar-first quaternions."""
    return normalize_quaternions(np.random.default_rng(seed).normal(size=(count, 4)))


def source_soft_targets(
    quaternions: torch.Tensor, bin_quaternions: torch.Tensor, sigma_rad: float
) -> torch.Tensor:
    """The source formula (research source snapshot, rotation_aux lines 121-137), written out."""
    query = functional.normalize(quaternions.float(), dim=1)
    bins = functional.normalize(bin_quaternions.float(), dim=1)
    dots = torch.abs(query @ bins.T).clamp(0.0, 1.0)
    angles = 2.0 * torch.acos(dots)
    return torch.softmax(-0.5 * (angles / sigma_rad) ** 2, dim=1)


def test_soft_targets_match_source():
    query = torch.from_numpy(np.random.default_rng(1).normal(size=(6, 4)))
    bins = torch.from_numpy(random_quaternions(32, 2))
    actual = soft_rotation_targets(query, bins, SIGMA_RAD)
    assert actual.dtype == torch.float32
    assert torch.equal(actual, source_soft_targets(query, bins, SIGMA_RAD))
    assert torch.allclose(actual.sum(dim=1), torch.ones(6))
    for bad in (0.0, -1.0, math.inf):
        with pytest.raises(ValueError, match="sigma_rad"):
            soft_rotation_targets(query, bins, bad)
    with pytest.raises(ValueError, match=r"\(B, 4\)"):
        soft_rotation_targets(query[:, :3], bins, SIGMA_RAD)
    with pytest.raises(ValueError, match=r"\(K, 4\)"):
        soft_rotation_targets(query, bins[:, :3], SIGMA_RAD)


def test_matrix_and_quaternion_soft_targets_agree():
    query = random_quaternions(8, 3)
    bins = random_quaternions(64, 4)
    rotations = torch.from_numpy(
        quaternion_to_object_camera(query, normalization="float32").astype(np.float64)
    )
    bin_rotations = torch.from_numpy(
        quaternion_to_object_camera(bins, normalization="float32").astype(np.float64)
    )
    from_matrices = matrix_soft_targets(rotations, bin_rotations, SIGMA_RAD)
    from_quaternions = soft_rotation_targets(
        torch.from_numpy(query), torch.from_numpy(bins), SIGMA_RAD
    )
    assert torch.allclose(from_matrices.float(), from_quaternions, atol=1e-5)
    flipped = soft_rotation_targets(torch.from_numpy(-query), torch.from_numpy(bins), SIGMA_RAD)
    assert torch.equal(flipped, from_quaternions)


def test_kl_equals_ce_minus_entropy():
    query = torch.from_numpy(random_quaternions(5, 5))
    bins = torch.from_numpy(random_quaternions(16, 6))
    logits = torch.from_numpy(np.random.default_rng(7).normal(size=(5, 16))).float()
    targets = soft_rotation_targets(query, bins, SIGMA_RAD)
    cross_entropy = rotation_aux_loss(logits, query, bins, sigma_deg=15.0)
    entropy = -(targets * targets.log()).sum(dim=1).mean()
    kl = functional.kl_div(functional.log_softmax(logits, dim=1), targets, reduction="batchmean")
    assert torch.allclose(kl, cross_entropy - entropy, atol=1e-6)


def test_rotation_aux_loss_sample_weights():
    query = torch.from_numpy(random_quaternions(4, 8))
    bins = torch.from_numpy(random_quaternions(8, 9))
    logits = torch.zeros(4, 8)
    unweighted = rotation_aux_loss(logits, query, bins)
    assert torch.allclose(unweighted, torch.tensor(math.log(8.0)))
    weighted = rotation_aux_loss(logits, query, bins, sample_weight=torch.tensor([1, 0, 0, 0]))
    assert torch.allclose(weighted, unweighted)
    with pytest.raises(ValueError, match="sample_weight"):
        rotation_aux_loss(logits, query, bins, sample_weight=torch.ones(3))


def test_farthest_rotation_bins_are_deterministic_and_sign_invariant():
    poses = random_quaternions(200, 10)
    selected = farthest_rotation_bins(poses, 16, 20260908)
    assert selected.dtype == np.int64 and len(set(selected.tolist())) == 16
    assert np.array_equal(selected, farthest_rotation_bins(poses, 16, 20260908))
    assert np.array_equal(selected, farthest_rotation_bins(-poses, 16, 20260908))
    with pytest.raises(ValueError, match="positive"):
        farthest_rotation_bins(poses, 0, 1)
    with pytest.raises(ValueError, match="201 bins from 200"):
        farthest_rotation_bins(poses, 201, 1)


def test_rotation_errors_deg_is_geodesic():
    q = np.asarray([[1.0, 0.0, 0.0, 0.0]])
    half = math.radians(45.0)
    rotated = np.asarray([[math.cos(half), 0.0, 0.0, math.sin(half)], [-1.0, 0.0, 0.0, 0.0]])
    assert np.allclose(rotation_errors_deg(q, rotated), [[90.0, 0.0]], atol=1e-4)


def test_pooled_features_and_cosine_head():
    features = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8)
    pooled = pooled_rotation_features(features, 4)
    assert pooled.shape == (2, 48)
    assert torch.equal(pooled, functional.adaptive_avg_pool2d(features, (4, 4)).flatten(1))
    with pytest.raises(ValueError, match="B, C, H, W"):
        pooled_rotation_features(features[0])
    with pytest.raises(ValueError, match="pool_size"):
        pooled_rotation_features(features, 0)
    assert CosineClassifier is CosineRotationHead
    head = CosineClassifier(48, 10, 30.0)
    logits = head(pooled)
    assert logits.shape == (2, 10) and float(logits.detach().abs().max()) <= 30.0 + 1e-4
    assert head.scale == head.logit_scale == 30.0 and head.pool_size == 4
    for args in ((0, 10, 30.0), (48, 10, 0.0), (48, 10, math.nan)):
        with pytest.raises(ValueError):
            CosineClassifier(*args)
    with pytest.raises(ValueError, match="pool_size"):
        CosineClassifier(48, 10, 30.0, 0)


def test_rotation_metrics_top1_and_oracle_top5():
    bins = torch.from_numpy(random_quaternions(8, 11))
    query = bins[:2].clone()
    logits = torch.zeros(2, 8)
    logits[0, 0] = 10.0
    logits[1, 3] = 10.0
    top1, top5 = rotation_metrics(logits, query, bins)
    assert float(top1[0]) < 0.1 and float(top1[1]) > 0.1
    assert float(top5[0]) < 0.1


def test_payload_round_trips_with_tensor_or_ndarray_bins():
    head = CosineClassifier(12, 4, 25.0, pool_size=2)
    bins = random_quaternions(4, 12)
    payload = RotationAuxPayload.from_head(head, bins)
    stored = payload.to_checkpoint()
    assert set(stored) == {
        "weight",
        "feature_dim",
        "num_bins",
        "pool_size",
        "logit_scale",
        "bin_quaternions",
    }
    assert isinstance(stored["bin_quaternions"], torch.Tensor)
    assert stored["bin_quaternions"].dtype == torch.float32
    assert torch.equal(stored["bin_quaternions"], torch.from_numpy(normalize_quaternions(bins)))
    legacy = {**stored, "bin_quaternions": bins}
    for section, raw_bins in ((stored, stored["bin_quaternions"].numpy()), (legacy, bins)):
        # Every load re-normalizes the stored bins, as the source consumers do.
        expected = torch.from_numpy(normalize_quaternions(raw_bins))
        for payload in (section, {"rotation_aux": section, "model": {}}):
            loaded = RotationAuxPayload.from_checkpoint(payload)
            assert torch.equal(loaded.bin_quaternions, expected)
            rebuilt = loaded.build_head()
            assert torch.equal(rebuilt.weight, head.weight)
            assert (rebuilt.logit_scale, rebuilt.pool_size) == (25.0, 2)
    with pytest.raises(ValueError, match="rotation_aux"):
        RotationAuxPayload.from_checkpoint({"model": {}})
    with pytest.raises(ValueError, match="num_bins"):
        RotationAuxPayload.from_checkpoint({**stored, "num_bins": 5})
