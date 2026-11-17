# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for DOPER's 2D keypoint heatmap targets."""

from __future__ import annotations

import numpy as np
import pytest

from doper.targets import (
    decode_keypoint_heatmaps,
    encode_keypoint_heatmaps,
    keypoint_heatmap_peak_indices,
)

CROP_SIZE = 257
HEATMAP_SIZE = 65


def _encode_as_logits(keypoints: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    heatmaps, weights = encode_keypoint_heatmaps(
        keypoints,
        np.ones((keypoints.shape[0],), dtype=np.float32),
        CROP_SIZE,
        HEATMAP_SIZE,
        sigma,
    )
    assert np.all(weights == 1.0)
    return np.log(np.maximum(heatmaps, np.finfo(np.float32).tiny))


@pytest.mark.parametrize(
    "keypoint",
    [
        (0.0, 128.0),
        (256.0, 128.0),
        (128.0, 0.0),
        (128.0, 256.0),
        (0.0, 0.0),
        (256.0, 256.0),
        (3.0, 253.0),
        (253.0, 3.0),
    ],
)
def test_encode_decode_round_trip_near_crop_borders(keypoint):
    expected = np.asarray([keypoint], dtype=np.float32)
    logits = _encode_as_logits(expected)

    decoded, _ = decode_keypoint_heatmaps(logits, CROP_SIZE, dark=True)

    np.testing.assert_allclose(decoded, expected, atol=1e-4)


def test_far_edge_keypoint_is_encoded_in_last_heatmap_bin():
    keypoint = np.asarray([[256.0, 256.0]], dtype=np.float32)

    heatmaps, weights = encode_keypoint_heatmaps(
        keypoint,
        np.ones((1,), dtype=np.float32),
        CROP_SIZE,
        HEATMAP_SIZE,
        sigma=1.5,
    )

    assert weights[0] == 1.0
    assert np.unravel_index(np.argmax(heatmaps[0]), heatmaps[0].shape) == (64, 64)
    assert heatmaps[0].sum() == pytest.approx(1.0)


def test_dark_refinement_recovers_fractional_heatmap_position():
    expected = np.asarray([[101.4, 147.6]], dtype=np.float32)
    logits = _encode_as_logits(expected)

    quantized, _ = decode_keypoint_heatmaps(logits, CROP_SIZE, dark=False)
    refined, _ = decode_keypoint_heatmaps(logits, CROP_SIZE, dark=True)

    assert np.linalg.norm(refined - expected) < np.linalg.norm(quantized - expected)
    np.testing.assert_allclose(refined, expected, atol=1e-4)


def test_peak_indices_suppress_nearby_bins_but_keep_separate_modes():
    logits = np.zeros((1, 16, 16), dtype=np.float32)
    logits[0, 4, 4] = 10.0
    logits[0, 5, 5] = 9.0
    logits[0, 11, 12] = 8.0

    peaks = keypoint_heatmap_peak_indices(logits, topk=2, min_distance_bins=2)

    np.testing.assert_array_equal(peaks[0], [[4, 4], [12, 11]])


@pytest.mark.parametrize("sigma", [0.0, -1.0, np.inf, np.nan])
def test_encode_rejects_nonpositive_or_nonfinite_sigma(sigma):
    with pytest.raises(ValueError, match="sigma"):
        encode_keypoint_heatmaps(
            np.zeros((1, 2), dtype=np.float32),
            np.ones((1,), dtype=np.float32),
            CROP_SIZE,
            HEATMAP_SIZE,
            sigma,
        )


@pytest.mark.parametrize(
    ("crop_size", "heatmap_size"),
    [
        (0, HEATMAP_SIZE),
        (1, HEATMAP_SIZE),
        (CROP_SIZE, 0),
        (CROP_SIZE, 1),
        (float(CROP_SIZE), HEATMAP_SIZE),
        (CROP_SIZE, float(HEATMAP_SIZE)),
    ],
)
def test_encode_rejects_degenerate_dimensions(crop_size, heatmap_size):
    with pytest.raises(ValueError):
        encode_keypoint_heatmaps(
            np.zeros((1, 2), dtype=np.float32),
            np.ones((1,), dtype=np.float32),
            crop_size,
            heatmap_size,
            sigma=1.5,
        )


@pytest.mark.parametrize(
    ("keypoints", "visibility"),
    [
        (np.zeros((2,), dtype=np.float32), np.ones((1,), dtype=np.float32)),
        (np.zeros((1, 3), dtype=np.float32), np.ones((1,), dtype=np.float32)),
        (np.zeros((2, 2), dtype=np.float32), np.ones((1,), dtype=np.float32)),
        (np.zeros((1, 2), dtype=np.float32), np.ones((1, 1), dtype=np.float32)),
    ],
)
def test_encode_rejects_malformed_keypoint_inputs(keypoints, visibility):
    with pytest.raises(ValueError):
        encode_keypoint_heatmaps(
            keypoints,
            visibility,
            CROP_SIZE,
            HEATMAP_SIZE,
            sigma=1.5,
        )


@pytest.mark.parametrize(
    ("logits", "crop_size"),
    [
        (np.zeros((5, 5), dtype=np.float32), CROP_SIZE),
        (np.zeros((1, 4, 5), dtype=np.float32), CROP_SIZE),
        (np.zeros((1, 1, 1), dtype=np.float32), CROP_SIZE),
        (np.zeros((1, 5, 5), dtype=np.float32), 1),
        (np.zeros((0, 5, 5), dtype=np.float32), CROP_SIZE),
        (np.zeros((1, 5, 5), dtype=np.float32), float(CROP_SIZE)),
    ],
)
def test_decode_rejects_malformed_or_degenerate_dimensions(logits, crop_size):
    with pytest.raises(ValueError):
        decode_keypoint_heatmaps(logits, crop_size)
