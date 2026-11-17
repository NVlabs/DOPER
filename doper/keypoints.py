# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Identity checks for 3D keypoint layouts shared by training and inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# The 3D keypoint layout file of a dataset, a prepared object or a capture bundle.
KEYPOINTS_NAME = "keypoints_3d.json"
# Decimals kept after scale normalization; coarse enough to survive float32.
_FINGERPRINT_DECIMALS = 4
# Hex digits of the SHA-256 digest kept in the fingerprint.
_FINGERPRINT_HEX_DIGITS = 16


def keypoints_3d_fingerprint(keypoints_3d: np.ndarray) -> str:
    """Return a scale-invariant fingerprint of a 3D keypoint layout.

    Normalized by the largest absolute coordinate so meters and millimeters
    agree; what it pins down is the set and the ORDER of the points. A model
    evaluated against a different or reordered keypoint set predicts perfectly
    good 2D points and then solves a silently wrong pose, which no other check
    in the pipeline notices.

    Args:
        keypoints_3d: ``(K, 3)`` keypoint coordinates (any array-like).

    Returns:
        A 16-character hex digest, or ``"degenerate"`` for an all-zero or
        non-finite layout.
    """
    arr = np.asarray(keypoints_3d, dtype=np.float64)
    scale = np.abs(arr).max()
    if not np.isfinite(scale) or scale <= 0:
        return "degenerate"
    # Normalize for scale, then quantize coarsely enough to survive a float32
    # round trip: callers legitimately load the same file at either precision
    # (training reads float64, the eval script float32), and 6 decimals is
    # inside float32's ~1e-7 relative error so it produced false mismatches.
    normalized = np.round((arr / scale).astype(np.float32), _FINGERPRINT_DECIMALS)
    normalized = normalized.astype(np.float64)
    return hashlib.sha256(normalized.tobytes()).hexdigest()[:_FINGERPRINT_HEX_DIGITS]


def load_keypoints_3d(path: str | Path) -> np.ndarray:
    """Read the ``(K, 3)`` float64 ``keypoints_3d`` array from a JSON file."""
    with open(path) as handle:
        return np.asarray(json.load(handle)["keypoints_3d"], dtype=np.float64)


def points_sha256(points: np.ndarray) -> str:
    """The points hash of a keypoint layout: SHA-256 of its little-endian float64 values.

    Unlike the SHA-256 of the layout file, it ignores formatting and keypoint names; recipes pin
    a layout by it (``keypoints_points_sha256``).
    """
    return hashlib.sha256(np.asarray(points, dtype="<f8").tobytes()).hexdigest()


def load_layout(path: str | Path, expected_points_sha256: str) -> NDArray[np.float64]:
    """Read a ``keypoints_3d.json`` that must be the layout with the given points hash.

    Raises:
        ValueError: For another layout.
    """
    points = load_keypoints_3d(path)
    digest = points_sha256(points)
    if digest != expected_points_sha256:
        raise ValueError(
            f"{path} is not the recipe's keypoint layout (points SHA-256 {digest}, "
            f"recipe {expected_points_sha256})"
        )
    return points
