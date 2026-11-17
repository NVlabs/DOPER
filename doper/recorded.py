# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for the recorded-capture layout: ``NNNNNN_color.png`` frames and intrinsics JSON."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

_DEFAULT_DISTORTION = [0, 0, 0, 0, 0]


def intrinsics_block_to_arrays(block: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(camera_matrix, dist_coeffs)`` from one intrinsics block.

    Accepts the RealSense SDK names (``ppx``/``ppy``, ``coeffs``) and the recorder names
    (``cx``/``cy``, ``distortion_coeffs``); ``ppx``/``ppy`` and ``distortion_coeffs`` win when
    both spellings are present.

    Args:
        block: Mapping with ``fx``, ``fy``, a principal point and optional distortion.

    Returns:
        A ``(3, 3)`` float64 camera matrix and a ``(N, 1)`` float64 distortion vector (five
        zeros if the block has none).

    Raises:
        ValueError: The block has no principal point.
    """
    cx = block.get("ppx", block.get("cx"))
    cy = block.get("ppy", block.get("cy"))
    if cx is None or cy is None:
        raise ValueError("intrinsics require ppx/ppy or cx/cy")
    camera = np.asarray(
        [[float(block["fx"]), 0.0, float(cx)], [0.0, float(block["fy"]), float(cy)], [0, 0, 1.0]],
        dtype=np.float64,
    )
    distortion = np.asarray(
        block.get("distortion_coeffs", block.get("coeffs", _DEFAULT_DISTORTION)), dtype=np.float64
    ).reshape(-1, 1)
    return camera, distortion


def load_intrinsics(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(camera_matrix, dist_coeffs)`` from a recorder's RealSense ``intrinsics.json``.

    Args:
        path: JSON file with a ``color_intrinsics`` block (see ``intrinsics_block_to_arrays``).

    Returns:
        A ``(3, 3)`` float64 camera matrix and a ``(N, 1)`` float64 distortion vector (five
        zeros if the file has none).
    """
    with open(path) as f:
        data = json.load(f)
    return intrinsics_block_to_arrays(data["color_intrinsics"])


def frame_index(path: str | Path) -> int:
    """Return the integer frame index in ``NNNNNN_color.png``, or 0 if the prefix is not numeric."""
    name = os.path.basename(path).split("_")[0]
    try:
        return int(name)
    except ValueError:
        return 0


def list_color_frames(data_dir: Path, max_frames: int = 0) -> list[Path]:
    """List ``*_color.png`` frames in ``data_dir`` ordered by frame index.

    Args:
        data_dir: Recording directory.
        max_frames: Keep only the first this-many frames; 0 keeps all.

    Returns:
        Frame paths sorted by ``frame_index`` (ties keep lexicographic order).
    """
    frames = sorted(sorted(Path(data_dir).glob("*_color.png")), key=frame_index)
    return frames[:max_frames] if max_frames > 0 else frames
