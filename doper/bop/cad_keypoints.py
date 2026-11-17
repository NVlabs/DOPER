# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keypoint selection on a BOP CAD model."""

from __future__ import annotations

import os
from pathlib import Path

from ..keypoints import KEYPOINTS_NAME

STL_NAME = "object_mm.stl"


def select_cad_keypoints(mesh: Path, geometry: Path, obj: int, count: int, units: str) -> Path:
    """Select ordered keypoints on a BOP CAD model.

    The PLY is exported to a binary STL (float32 vertices) and the hull-anchored selector runs
    on that STL; the keypoints are written in meters.

    Args:
        mesh: BOP CAD model (PLY).
        geometry: Directory receiving the STL, ``keypoints_3d.json`` and ``selection.md``.
        obj: Object id (the keypoint file's object name).
        count: Keypoint count.
        units: Length unit of the CAD model.

    Returns:
        The keypoint file.
    """
    from .. import keypoint_selection
    from .audit import load_bop_mesh

    geometry.mkdir(parents=True, exist_ok=True)
    stl = geometry / STL_NAME
    load_bop_mesh(mesh).export(stl)
    out = geometry / KEYPOINTS_NAME
    staging = geometry / f".{KEYPOINTS_NAME}.{os.getpid()}"
    keypoint_selection.main(
        [
            "--mesh", str(stl),
            "--source-units", units,
            "--num-keypoints", str(count),
            "--object-name", f"obj_{obj:06d}",
            "--out-json", str(staging),
            "--out-report-md", str(geometry / "selection.md"),
        ]
    )  # fmt: skip
    os.replace(staging, out)
    return out
