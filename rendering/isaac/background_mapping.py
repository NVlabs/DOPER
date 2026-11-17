# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in spatial interpretation of Blender scene-record backgrounds.

These mappings align texture coordinates, not lighting units or PBR shading.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def blender_dome_rotation(recorded_degrees: float) -> float:
    """Convert a Blender environment lookup rotation to the Isaac dome convention."""
    return 180.0 - recorded_degrees


def blender_ground_uv(
    points_xy: ArrayLike, scene_scale: float, texture_scale: float, rotation_degrees: float
) -> np.ndarray:
    """Evaluate Blender plane UVs at world XY positions, then its Mapping node.

    Blender's plane spans +/-scene_scale and scales UVs by texture_scale/3 before rotating
    about the texture origin. Isaac's larger plane may sample outside [0, 1]; texture wrapping
    is intentional.

    Args:
        points_xy: World XY positions, shape ``(N, 2)``.
        scene_scale: Half extent of Blender's ground plane.
        texture_scale: Recorded ground texture scale.
        rotation_degrees: Recorded ground texture rotation.

    Returns:
        The ``(N, 2)`` UV coordinates.
    """
    uv = np.asarray(points_xy, dtype=float) / (2 * scene_scale) + 0.5
    uv *= texture_scale / 3.0
    angle = np.deg2rad(rotation_degrees)
    c, s = np.cos(angle), np.sin(angle)
    return uv @ np.array([[c, s], [-s, c]])
