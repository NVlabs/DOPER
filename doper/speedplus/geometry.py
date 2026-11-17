# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SPEED+ geometry: the camera, the official Tango landmarks, the 64-point layout and poses.

SPEED+ stores a scalar-first quaternion ``q_vbs2tango_true`` and a camera-frame translation
``r_Vo2To_vbs_true``. The official baseline projects model points with ``R(q).T``: the
object-to-camera rotation is the transpose of the SPEED+ direction-cosine matrix. Keeping that
transpose explicit matters, because the opposite convention gives plausible overlays with
inverted poses.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.io import loadmat
from scipy.spatial.transform import Rotation

OFFICIAL_POINT_COUNT = 11
LAYOUT_64_COUNT = 64
# OpenCV accepts these distortion vector lengths.
DISTORTION_LENGTHS = (4, 5, 8, 12, 14)
# Samples per edge of the trilinear body grid the 64-point layout is drawn from.
BODY_GRID_SAMPLES = 9
CORNER_TOLERANCE_M = 1e-10
DUPLICATE_DECIMALS = 10


@dataclass(frozen=True)
class Camera:
    """A SPEED+ pinhole camera with OpenCV distortion.

    Attributes:
        K: ``(3, 3)`` intrinsics.
        dist: OpenCV distortion coefficients.
        width: Image width in pixels.
        height: Image height in pixels.
    """

    K: np.ndarray
    dist: np.ndarray
    width: int
    height: int


def _first(payload: dict[str, Any], keys: Sequence[str]) -> Any:  # noqa: ANN401
    return next((payload[key] for key in keys if key in payload), None)


def load_camera(path: str | Path) -> Camera:
    """Load a native SPEED+ ``camera.json`` or a DOPER-normalized camera file.

    Native files carry ``cameraMatrix``, ``distCoeffs``, ``Nu`` and ``Nv``; normalized files
    carry ``camera_matrix``, ``distortion_coefficients``, ``width`` and ``height``. Missing
    distortion means none.

    Raises:
        ValueError: If the matrix is missing or not 3x3, the distortion length is not an
            OpenCV one, or the image size is missing.
    """
    payload = json.loads(Path(path).read_text())
    matrix = _first(payload, ("cameraMatrix", "camera_matrix"))
    distortion = _first(payload, ("distCoeffs", "distortion_coefficients"))
    width = _first(payload, ("Nu", "width"))
    height = _first(payload, ("Nv", "height"))
    if matrix is None:
        raise ValueError(f"camera file has no camera matrix: {path}")
    matrix_arr = np.asarray(matrix, dtype=np.float64)
    distortion_arr = np.asarray(
        np.zeros(5) if distortion is None else distortion, dtype=np.float64
    ).reshape(-1)
    if matrix_arr.shape != (3, 3):
        raise ValueError(f"camera matrix must be 3x3, got {matrix_arr.shape}")
    if distortion_arr.size not in DISTORTION_LENGTHS:
        raise ValueError(f"unsupported OpenCV distortion length {distortion_arr.size}")
    if width is None or height is None:
        raise ValueError(f"camera file has no image size (Nu/Nv or width/height): {path}")
    return Camera(matrix_arr, distortion_arr, int(width), int(height))


def load_official_tango_keypoints(mat: str | Path) -> np.ndarray:
    """Load the official 3x11 ``tango3Dpoints`` MATLAB array as ``(11, 3)`` meters.

    Raises:
        ValueError: If the array is missing, has another shape or is not finite.
    """
    payload = loadmat(mat)
    if "tango3Dpoints" not in payload:
        raise ValueError(f"{mat} does not contain tango3Dpoints")
    points = np.asarray(payload["tango3Dpoints"], dtype=np.float64)
    if points.shape == (3, OFFICIAL_POINT_COUNT):
        points = points.T
    if points.shape != (OFFICIAL_POINT_COUNT, 3):
        raise ValueError(f"expected 11x3 Tango points, got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("Tango keypoints contain non-finite coordinates")
    return points


def _trilinear_hexahedron(corners: np.ndarray, uvw: np.ndarray) -> np.ndarray:
    """Interpolate the official eight body corners on a trilinear hexahedron.

    The official order is the top z face 0..3, then the bottom z face 4..7; within a face
    ``(-x,-y), (-x,+y), (+x,+y), (+x,-y)``.
    """
    result = np.zeros((len(uvw), 3), dtype=np.float64)
    for index, (u, v, w) in enumerate(uvw):
        top = (
            (1 - u) * (1 - v) * corners[0]
            + (1 - u) * v * corners[1]
            + u * v * corners[2]
            + u * (1 - v) * corners[3]
        )
        bottom = (
            (1 - u) * (1 - v) * corners[4]
            + (1 - u) * v * corners[5]
            + u * v * corners[6]
            + u * (1 - v) * corners[7]
        )
        result[index] = (1 - w) * bottom + w * top
    return result


def _body_surface(official: np.ndarray) -> np.ndarray:
    """Dense trilinear body-surface candidates, without the eight official corners."""
    grid = np.linspace(0.0, 1.0, BODY_GRID_SAMPLES)
    uvw = np.asarray(
        [
            (u, v, w)
            for u in grid
            for v in grid
            for w in grid
            if u in (0.0, 1.0) or v in (0.0, 1.0) or w in (0.0, 1.0)
        ],
        dtype=np.float64,
    )
    candidates = _trilinear_hexahedron(official[:8], uvw)
    keep = np.ones(len(candidates), dtype=bool)
    for corner in official[:8]:
        keep &= np.linalg.norm(candidates - corner, axis=1) > CORNER_TOLERANCE_M
    return candidates[keep]


def expand_tango_keypoints_64(official: np.ndarray) -> np.ndarray:
    """Expand the 11 official landmarks to the deterministic 64-point layout.

    The first 11 rows are the official points byte for byte. The other 53 are chosen by
    farthest-point sampling from a dense trilinear body surface, which keeps the SPEED+ model
    frame and the published appendage landmarks.

    Raises:
        ValueError: If ``official`` is not ``(11, 3)``.
        RuntimeError: If the layout has duplicate points.
    """
    points = np.asarray(official, dtype=np.float64)
    if points.shape != (OFFICIAL_POINT_COUNT, 3):
        raise ValueError(f"expected 11x3 official points, got {points.shape}")
    candidates = _body_surface(points)
    selected = [p.copy() for p in points]
    min_distance = np.min(
        np.linalg.norm(candidates[:, None, :] - points[None, :, :], axis=2), axis=1
    )
    while len(selected) < LAYOUT_64_COUNT:
        candidate_index = int(np.argmax(min_distance))
        point = candidates[candidate_index]
        selected.append(point.copy())
        min_distance = np.minimum(min_distance, np.linalg.norm(candidates - point, axis=1))
        min_distance[candidate_index] = -1.0
    expanded = np.asarray(selected, dtype=np.float64)
    if np.unique(np.round(expanded, decimals=DUPLICATE_DECIMALS), axis=0).shape[0] != 64:
        raise RuntimeError("expanded Tango keypoint layout contains duplicates")
    return expanded


def quaternion_to_dcm_scalar_first(q: Sequence[float] | np.ndarray) -> np.ndarray:
    """The SPEED+ direction-cosine matrix of a scalar-first quaternion (official baseline).

    Raises:
        ValueError: If the quaternion is zero or not finite.
    """
    quaternion = np.asarray(q, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("quaternion must be finite and nonzero")
    q0, q1, q2, q3 = quaternion / norm
    return np.asarray(
        [
            [2 * q0**2 - 1 + 2 * q1**2, 2 * q1 * q2 + 2 * q0 * q3, 2 * q1 * q3 - 2 * q0 * q2],
            [2 * q1 * q2 - 2 * q0 * q3, 2 * q0**2 - 1 + 2 * q2**2, 2 * q2 * q3 + 2 * q0 * q1],
            [2 * q1 * q3 + 2 * q0 * q2, 2 * q2 * q3 - 2 * q0 * q1, 2 * q0**2 - 1 + 2 * q3**2],
        ],
        dtype=np.float64,
    )


def speedplus_pose_to_opencv(
    q: Sequence[float] | np.ndarray, r: Sequence[float] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a native SPEED+ pose to the OpenCV object-to-camera ``(rvec, tvec)``, ``(3, 1)``."""
    rotation_object_to_camera = quaternion_to_dcm_scalar_first(q).T
    rvec, _ = cv2.Rodrigues(rotation_object_to_camera)
    return rvec, np.asarray(r, dtype=np.float64).reshape(3, 1)


def opencv_pose_to_speedplus(rvec: np.ndarray, tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert an OpenCV object-to-camera pose to SPEED+ form (scalar part non-negative).

    Returns:
        The scalar-first quaternion ``(4,)`` and the translation ``(3,)``.
    """
    rotation_object_to_camera, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    # SciPy is scalar-last. The SPEED+ DCM is the transpose of SciPy's active rotation matrix
    # for the same quaternion, so the object-to-camera matrix converts directly.
    q_xyzw = Rotation.from_matrix(rotation_object_to_camera).as_quat()
    q_wxyz = q_xyzw[[3, 0, 1, 2]]
    if q_wxyz[0] < 0:
        q_wxyz = -q_wxyz
    return q_wxyz, np.asarray(tvec, dtype=np.float64).reshape(3)


def project_keypoints(
    points_3d: np.ndarray,
    q: Sequence[float] | np.ndarray,
    r: Sequence[float] | np.ndarray,
    camera: Camera,
) -> tuple[np.ndarray, np.ndarray]:
    """Project model-frame points at a native SPEED+ pose through the camera.

    Returns:
        The distorted pixels ``(N, 2)`` and the camera-frame depths ``(N,)``.
    """
    points = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    rvec, tvec = speedplus_pose_to_opencv(q, r)
    pixels, _ = cv2.projectPoints(points, rvec, tvec, camera.K, camera.dist)
    rotation, _ = cv2.Rodrigues(rvec)
    camera_points = (rotation @ points.T + tvec).T
    return pixels.reshape(-1, 2), camera_points[:, 2]
