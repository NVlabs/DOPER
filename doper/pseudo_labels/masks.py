# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mask and box geometry for pseudo-labels: projected CAD meshes, projected cuboids, tight boxes.

Boxes are COCO ``[x, y, width, height]`` lists in image pixels; masks are ``uint8`` images with
foreground 255.
"""

from __future__ import annotations

import struct
from pathlib import Path

import cv2
import numpy as np

from doper.geometry_vis import project_points

STL_HEADER_BYTES = 80
STL_COUNT_FORMAT = "<I"
STL_COUNT_BYTES = 4
_STL_TRIANGLE = np.dtype(
    [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
)
MASK_FOREGROUND = 255
_SEAM_KERNEL = np.ones((3, 3), dtype=np.uint8)
_MIN_HULL_POINTS = 3
_BOX_DECIMALS = 2


def load_binary_stl_triangles(path: str | Path) -> np.ndarray:
    """Read the triangles of a binary STL file.

    Args:
        path: Binary STL file.

    Returns:
        ``(T, 3, 3)`` float64 triangle vertices in the file's units and frame.

    Raises:
        ValueError: The file holds fewer triangles than its header declares.
    """
    with open(path, "rb") as handle:
        handle.read(STL_HEADER_BYTES)
        triangle_count = struct.unpack(STL_COUNT_FORMAT, handle.read(STL_COUNT_BYTES))[0]
        records = np.fromfile(handle, dtype=_STL_TRIANGLE, count=triangle_count)
    if len(records) != triangle_count:
        raise ValueError(f"{path} contains {len(records)} triangles, expected {triangle_count}")
    return records["vertices"].astype(np.float64, copy=True)


def projected_mesh_mask(
    triangles: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Rasterize the solid silhouette of a posed triangle mesh.

    Triangles with a vertex behind the camera or a non-finite projection are skipped. One-pixel
    rasterization seams are closed and only external contours are filled, so cavities of the
    CAD model do not become holes in the mask.

    Args:
        triangles: ``(T, 3, 3)`` mesh triangles in the object frame.
        rvec: Object-to-camera rotation vector.
        tvec: Object-to-camera translation vector.
        camera_matrix: ``(3, 3)`` intrinsics.
        dist_coeffs: Distortion coefficients.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        ``(height, width)`` uint8 mask.
    """
    flat = triangles.reshape(-1, 3)
    projected = project_points(flat, rvec, tvec, camera_matrix, dist_coeffs).reshape(-1, 3, 2)
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    camera_points = flat @ rotation.T + np.asarray(tvec, dtype=np.float64).reshape(1, 3)
    depth = camera_points[:, 2].reshape(-1, 3)
    valid = np.all(depth > 0.0, axis=1) & np.isfinite(projected).all(axis=(1, 2))
    mask = np.zeros((height, width), dtype=np.uint8)
    polygons = np.rint(projected[valid]).astype(np.int32)
    if len(polygons):
        cv2.fillPoly(mask, list(polygons), MASK_FOREGROUND, lineType=cv2.LINE_8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _SEAM_KERNEL)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid = np.zeros_like(mask)
    if contours:
        cv2.drawContours(solid, contours, -1, MASK_FOREGROUND, thickness=cv2.FILLED)
    return solid


def projected_cuboid_mask(projected_corners: np.ndarray, width: int, height: int) -> np.ndarray:
    """Rasterize the convex silhouette of projected 3D cuboid corners.

    Non-finite corners are dropped and the rest are clipped to the image; fewer than three
    remaining corners give an empty mask.

    Args:
        projected_corners: ``(8, 2)`` projected corners in pixels.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        ``(height, width)`` uint8 mask.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    corners = np.asarray(projected_corners, dtype=np.float64)
    corners = corners[np.isfinite(corners).all(axis=1)]
    if len(corners) < _MIN_HULL_POINTS:
        return mask
    corners[:, 0] = np.clip(corners[:, 0], 0, width - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, height - 1)
    hull = cv2.convexHull(np.rint(corners).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, MASK_FOREGROUND, lineType=cv2.LINE_8)
    return mask


def bbox_from_mask(mask: np.ndarray, pad_fraction: float) -> list[float]:
    """Return the padded, image-clipped box around the foreground of ``mask``.

    Args:
        mask: ``(H, W)`` mask; nonzero pixels are foreground.
        pad_fraction: Padding on each side as a fraction of the tight box width and height.

    Returns:
        COCO ``[x, y, width, height]`` box.

    Raises:
        ValueError: The mask is empty.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("projected mesh produced an empty mask")
    height, width = mask.shape
    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max() + 1)
    y2 = float(ys.max() + 1)
    pad_x = (x2 - x1) * pad_fraction
    pad_y = (y2 - y1) * pad_fraction
    x1 = max(0.0, x1 - pad_x)
    y1 = max(0.0, y1 - pad_y)
    x2 = min(float(width), x2 + pad_x)
    y2 = min(float(height), y2 + pad_y)
    return [x1, y1, x2 - x1, y2 - y1]


def bbox_from_points(points: np.ndarray, width: int, height: int, pad_frac: float) -> list[float]:
    """Return the padded, image-clipped box around the in-image points.

    Args:
        points: ``(N, 2)`` pixel coordinates; non-finite and out-of-image points are ignored.
        width: Image width in pixels.
        height: Image height in pixels.
        pad_frac: Padding on each side as a fraction of the box width and height (each at
            least one pixel).

    Returns:
        COCO ``[x, y, width, height]`` box rounded to two decimals; all zeros when no point
        lies inside the image.
    """
    finite = np.isfinite(points).all(axis=1)
    inside = (
        finite
        & (points[:, 0] >= 0)
        & (points[:, 0] < width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < height)
    )
    kept = points[inside]
    if len(kept) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x0, y0 = kept.min(axis=0)
    x1, y1 = kept.max(axis=0)
    box_width = max(1.0, x1 - x0)
    box_height = max(1.0, y1 - y0)
    x0 = max(0.0, x0 - box_width * pad_frac)
    y0 = max(0.0, y0 - box_height * pad_frac)
    x1 = min(float(width), x1 + box_width * pad_frac)
    y1 = min(float(height), y1 + box_height * pad_frac)
    return [round(value, _BOX_DECIMALS) for value in (x0, y0, x1 - x0, y1 - y0)]
