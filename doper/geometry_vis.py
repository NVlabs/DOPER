# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry and drawing helpers for visualizing a solved object pose on a frame."""

from __future__ import annotations

import cv2
import numpy as np

# 12 edges of an axis-aligned 3D box, indexing compute_3d_bbox()'s 8 corners.
_BBOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]  # fmt: skip
_AXIS_COLORS_BGR = [(0, 0, 255), (0, 220, 0), (255, 0, 0)]
_AXIS_THICKNESS = 3
_AXIS_TIP_LENGTH = 0.18
_MAX_PIXEL = 1e6


def compute_3d_bbox(points: np.ndarray) -> np.ndarray:
    """Return the 8 corners of the axis-aligned box around ``points``.

    Args:
        points: ``(N, 3)`` points.

    Returns:
        ``(8, 3)`` float64 corners; the first four share the maximum z, the last four the
        minimum z.
    """
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    x0, y0, z0 = mins
    x1, y1, z1 = maxs
    return np.array(
        [
            [x1, y1, z1], [x0, y1, z1], [x0, y0, z1], [x1, y0, z1],
            [x1, y1, z0], [x0, y1, z0], [x0, y0, z0], [x1, y0, z0],
        ],
        dtype=np.float64,
    )  # fmt: skip


def project_points(
    points_3d: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    """Project 3D points into the image with ``cv2.projectPoints``.

    Args:
        points_3d: ``(N, 3)`` points in the object frame.
        rvec: Object-to-camera rotation vector.
        tvec: Object-to-camera translation vector.
        camera_matrix: ``(3, 3)`` intrinsics.
        dist_coeffs: Distortion coefficients.

    Returns:
        ``(N, 2)`` pixel coordinates.
    """
    pts2d, _ = cv2.projectPoints(
        points_3d.astype(np.float64), rvec, tvec, camera_matrix, dist_coeffs
    )
    return pts2d.reshape(-1, 2)


def _to_int_pt(point: np.ndarray) -> tuple[int, int]:
    """Round a 2D point to a plain-int pixel tuple, clamped to a drawable range.

    A wildly wrong PnP pose can project to finite but astronomically large coordinates whose
    int64 cast overflows what cv2 drawing accepts; clamping draws the line off-screen instead
    of crashing.
    """
    x = int(np.clip(round(float(point[0])), -_MAX_PIXEL, _MAX_PIXEL))
    y = int(np.clip(round(float(point[1])), -_MAX_PIXEL, _MAX_PIXEL))
    return x, y


def draw_axes(
    frame: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    axis_length: float,
) -> None:
    """Draw the object's x (red), y (green) and z (blue) axes onto ``frame`` in place.

    Args:
        frame: BGR image, modified in place.
        rvec: Object-to-camera rotation vector.
        tvec: Object-to-camera translation vector.
        camera_matrix: ``(3, 3)`` intrinsics.
        dist_coeffs: Distortion coefficients.
        axis_length: Axis length in the object's units.
    """
    pts = np.array(
        [[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]],
        dtype=np.float64,
    )
    projected = project_points(pts, rvec, tvec, camera_matrix, dist_coeffs)
    if not np.isfinite(projected).all():
        return
    origin = _to_int_pt(projected[0])
    for i, color in zip(range(1, 4), _AXIS_COLORS_BGR, strict=True):
        end = _to_int_pt(projected[i])
        cv2.arrowedLine(
            frame, origin, end, color, _AXIS_THICKNESS, cv2.LINE_AA, tipLength=_AXIS_TIP_LENGTH
        )


def draw_box3d(
    frame: np.ndarray,
    corners_2d: np.ndarray,
    color: tuple[int, int, int] = (255, 210, 0),
    thickness: int = 2,
) -> None:
    """Draw the 12 edges of a projected 3D box onto ``frame`` in place.

    Args:
        frame: BGR image, modified in place.
        corners_2d: ``(8, 2)`` projected corners from ``compute_3d_bbox`` order. Nothing is
            drawn if any corner is non-finite.
        color: BGR line color.
        thickness: Line thickness in pixels.
    """
    if not np.isfinite(corners_2d).all():
        return
    for i, j in _BBOX_EDGES:
        cv2.line(
            frame,
            _to_int_pt(corners_2d[i]),
            _to_int_pt(corners_2d[j]),
            color,
            thickness,
            cv2.LINE_AA,
        )
