# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Similarity fits, multi-view triangulation, camera centers and rig conversions.

Conventions:

* Camera axes are OpenCV (x right, y down, z forward). ``cam_from_world`` is the 4x4 (or
  3x4) matrix that maps world coordinates into the camera frame, as COLMAP stores it.
* ``T_a_b`` maps b-coordinates into a, so ``T_world_camera`` is the camera's pose (camera to
  world). The one exception is the NuRec rig calibration ``T_sensor_rig``, which keeps NuRec's
  own name: it maps sensor (OpenCV camera) coordinates into the rig frame, and NuRec composes
  ``T_world_camera = T_rig_world @ T_sensor_rig`` where its ``T_rig_world`` is the rig's pose
  (rig to world). ``rig_from_camera`` and ``camera_from_rig`` take and return those poses.
* Units are whatever the inputs use; nothing here converts between meters and scene units.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

# OpenCV sensor axes (x right, y down, z forward) into the forward-left-up rig frame:
# ``v_rig = T_SENSOR_RIG_FLU @ v_sensor`` (NuRec's ``T_sensor_rig`` rotation block).
T_SENSOR_RIG_FLU: NDArray[np.float64] = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
)
T_SENSOR_RIG_FLU.setflags(write=False)
MIN_SIM3_POINTS = 3
MIN_VIEWS = 2
ROTATION_ATOL = 1e-6
# Tolerance on a rigid transform's rotation block and last row.
RIGID_ATOL = ROTATION_ATOL


def invert(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """The matrix inverse (``np.linalg.inv``), typed for the numpy stubs of every Python."""
    return np.linalg.inv(matrix)  # pyright: ignore[reportReturnType]


@dataclass(frozen=True)
class Sim3:
    """A similarity transform ``x -> scale * rotation @ x + translation``.

    Attributes:
        scale: Positive scale (destination units per source unit).
        rotation: Proper 3x3 rotation (determinant +1).
        translation: Translation in destination units.
    """

    scale: float
    rotation: NDArray[np.float64]
    translation: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate and freeze the parameters."""
        rotation = np.array(self.rotation, dtype=np.float64)
        translation = np.array(self.translation, dtype=np.float64).reshape(-1)
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError(f"Sim3 scale must be positive and finite, got {self.scale}")
        if rotation.shape != (3, 3) or not _is_rotation(rotation):
            raise ValueError("Sim3 rotation must be a proper 3x3 rotation matrix")
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError("Sim3 translation must be a finite 3-vector")
        rotation.setflags(write=False)
        translation.setflags(write=False)
        object.__setattr__(self, "scale", float(self.scale))
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    def apply(self, points: ArrayLike) -> NDArray[np.float64]:
        """Transform points of shape ``(..., 3)``."""
        points = np.asarray(points, dtype=np.float64)
        return self.scale * points @ self.rotation.T + self.translation

    def inverse(self) -> Sim3:
        """The inverse similarity."""
        rotation = self.rotation.T
        return Sim3(1.0 / self.scale, rotation, -(rotation @ self.translation) / self.scale)

    def as_matrix(self) -> NDArray[np.float64]:
        """The 4x4 homogeneous matrix ``[[scale * rotation, translation], [0, 0, 0, 1]]``."""
        matrix = np.eye(4)
        matrix[:3, :3] = self.scale * self.rotation
        matrix[:3, 3] = self.translation
        return matrix


@dataclass(frozen=True)
class CameraPose:
    """A calibrated view's extrinsics.

    Attributes:
        cam_from_world: 4x4 or 3x4 matrix mapping world coordinates into the OpenCV camera.
    """

    cam_from_world: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate and freeze the matrix."""
        matrix = np.array(self.cam_from_world, dtype=np.float64)
        if matrix.shape not in ((3, 4), (4, 4)) or not _is_rotation(matrix[:3, :3]):
            raise ValueError("cam_from_world must be a 3x4 or 4x4 rigid transform")
        matrix.setflags(write=False)
        object.__setattr__(self, "cam_from_world", matrix)


@dataclass(frozen=True)
class Observation:
    """One pixel observation of a point in one view.

    Attributes:
        point: Index of the observed point (``0..n-1`` across all observations).
        camera: Index into the ``cameras`` sequence given to ``triangulate_points``.
        uv: Pixel coordinates ``(u, v)``.
    """

    point: int
    camera: int
    uv: tuple[float, float]


def umeyama_sim3(src: ArrayLike, dst: ArrayLike) -> Sim3:
    """Least-squares similarity ``dst ~ scale * R @ src + t`` (Umeyama 1991).

    The rotation is reflection-corrected: if the best orthogonal fit is a reflection, the
    smallest singular direction is flipped so ``det(R) == +1``.

    Args:
        src: Source points, shape ``(n, 3)``.
        dst: Corresponding destination points, shape ``(n, 3)``.

    Returns:
        The fitted transform; its scale is destination units per source unit.

    Raises:
        ValueError: On mismatched shapes, fewer than three points or a degenerate source.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(
            f"src and dst must have the same shape (n, 3), got {src.shape} and {dst.shape}"
        )
    if len(src) < MIN_SIM3_POINTS:
        raise ValueError(f"umeyama_sim3 needs at least 3 points, got {len(src)}")
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    a = src - src_mean
    b = dst - dst_mean
    variance = float((a * a).sum())
    if variance <= 0.0:
        raise ValueError("umeyama_sim3: the source points are degenerate (zero variance)")
    u, singular, vt = np.linalg.svd(a.T @ b)
    sign = np.ones(3)
    if np.linalg.det(vt.T @ u.T) < 0.0:
        sign[2] = -1.0
    rotation = vt.T @ np.diag(sign) @ u.T
    scale = float((singular * sign).sum() / variance)
    return Sim3(scale, rotation, dst_mean - scale * rotation @ src_mean)


def triangulate_points(
    observations: Sequence[Observation],
    cameras: Sequence[CameraPose],
    k: ArrayLike,
    *,
    f_scale_px: float = 3.0,
    init: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Triangulate points from pixel observations in calibrated views.

    Each point starts from the linear (DLT) solution, or ``init``, and is refined by
    minimizing its reprojection residuals with ``scipy.optimize.least_squares`` under the
    ``soft_l1`` loss with ``f_scale = f_scale_px``. Residuals are ordered as the
    observations are given.

    Args:
        observations: Observations; every point index ``0..n-1`` needs at least two.
        cameras: Views indexed by ``Observation.camera``.
        k: The 3x3 intrinsics shared by all views.
        f_scale_px: Soft-L1 inlier scale in pixels.
        init: Optional start, shape ``(3,)`` (shared) or ``(n, 3)``, instead of the DLT.

    Returns:
        Points of shape ``(n, 3)`` in world coordinates.

    Raises:
        ValueError: On an unknown camera, a point with fewer than two observations or a
            malformed ``init``.
    """
    from scipy.optimize import least_squares

    k = np.asarray(k, dtype=np.float64)
    groups = _group_observations(observations, len(cameras))
    starts = None
    if init is not None:
        init = np.asarray(init, dtype=np.float64)
        if init.shape not in ((3,), (len(groups), 3)):
            raise ValueError(f"init must have shape (3,) or ({len(groups)}, 3), got {init.shape}")
        starts = np.broadcast_to(init, (len(groups), 3))
    points = np.empty((len(groups), 3))
    for j, group in enumerate(groups):
        poses = np.stack([cameras[o.camera].cam_from_world[:3] for o in group])
        uv = np.array([o.uv for o in group], dtype=np.float64)
        start = _dlt_point(k @ poses, uv) if starts is None else starts[j]
        result = least_squares(
            _reprojection_residuals, start, args=(k, poses, uv), loss="soft_l1", f_scale=f_scale_px
        )
        points[j] = result.x
    return points


def camera_centre(cam_from_world: ArrayLike) -> NDArray[np.float64]:
    """World position ``-R^T t`` of a camera given its 3x4 or 4x4 ``cam_from_world``."""
    matrix = np.asarray(cam_from_world, dtype=np.float64)
    return -matrix[:3, :3].T @ matrix[:3, 3]


def rig_from_camera(
    T_world_camera: ArrayLike,  # noqa: N803 - frame-pair name
    T_sensor_rig: ArrayLike,  # noqa: N803 - NuRec calibration name
) -> NDArray[np.float64]:
    """The rig pose (rig to world) ``T_world_camera @ inv(T_sensor_rig)``.

    Args:
        T_world_camera: 4x4 camera pose (OpenCV camera to world).
        T_sensor_rig: 3x3 or 4x4 calibration mapping sensor coordinates into the rig.

    Returns:
        The 4x4 rig pose (NuRec's ``T_rig_world``).
    """
    return _homogeneous(T_world_camera) @ np.linalg.inv(_homogeneous(T_sensor_rig))


def camera_from_rig(
    T_world_rig: ArrayLike,  # noqa: N803 - frame-pair name
    T_sensor_rig: ArrayLike,  # noqa: N803 - NuRec calibration name
) -> NDArray[np.float64]:
    """The camera pose (OpenCV camera to world) ``T_world_rig @ T_sensor_rig``.

    Args:
        T_world_rig: 4x4 rig pose (rig to world; NuRec's ``T_rig_world``).
        T_sensor_rig: 3x3 or 4x4 calibration mapping sensor coordinates into the rig.

    Returns:
        The 4x4 camera pose.
    """
    return _homogeneous(T_world_rig) @ _homogeneous(T_sensor_rig)


def project_pinhole(
    points: NDArray[np.float64], cam_from_object: NDArray[np.float64], k: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Pixels ``(n, 2)`` of ``(n, 3)`` points under a 3x4 or 4x4 ``cam_from_object`` and ``k``."""
    q = (points @ cam_from_object[:3, :3].T + cam_from_object[:3, 3]) @ k.T
    return q[:, :2] / q[:, 2:]


def rotation_geodesic_deg(
    R_a: ArrayLike,  # noqa: N803 - rotation matrix name
    R_b: ArrayLike,  # noqa: N803 - rotation matrix name
) -> float:
    """Geodesic angle in degrees between two rotation matrices."""
    product = np.asarray(R_a, dtype=np.float64) @ np.asarray(R_b, dtype=np.float64).T
    cosine = np.clip((np.trace(product) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def as_rigid(
    matrix: ArrayLike,
    where: str,
    *,
    not_finite: str = "must be a finite 4x4 transform",
    not_rigid: str = "must be a rigid 4x4 transform",
    last_row_atol: float = RIGID_ATOL,
) -> NDArray[np.float64]:
    """A read-only float64 copy of a rigid 4x4 transform, or ``ValueError``.

    Rigid means a proper rotation block (orthonormal to ``RIGID_ATOL``, det > 0) and a last
    row of ``0 0 0 1`` to ``last_row_atol``. The errors read ``f"{where} {not_finite}"``
    (wrong shape or non-finite) and ``f"{where} {not_rigid}"``.
    """
    array = np.array(matrix, dtype=np.float64)
    if array.shape != (4, 4) or not np.isfinite(array).all():
        raise ValueError(f"{where} {not_finite}")
    rigid = _is_rotation(array[:3, :3]) and np.allclose(
        array[3], [0.0, 0.0, 0.0, 1.0], atol=last_row_atol
    )
    if not rigid:
        raise ValueError(f"{where} {not_rigid}")
    array.setflags(write=False)
    return array


def _is_rotation(matrix: NDArray[np.float64]) -> bool:
    return bool(
        np.isfinite(matrix).all()
        and np.allclose(matrix @ matrix.T, np.eye(3), atol=ROTATION_ATOL)
        and np.linalg.det(matrix) > 0.0
    )


def _homogeneous(matrix: ArrayLike) -> NDArray[np.float64]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (4, 4):
        return matrix
    if matrix.shape not in ((3, 3), (3, 4)):
        raise ValueError(f"expected a 3x3, 3x4 or 4x4 transform, got shape {matrix.shape}")
    out = np.eye(4)
    out[:3, : matrix.shape[1]] = matrix
    return out


def _group_observations(
    observations: Sequence[Observation], camera_count: int
) -> list[list[Observation]]:
    groups: dict[int, list[Observation]] = {}
    for observation in observations:
        if not 0 <= observation.camera < camera_count:
            raise ValueError(
                f"observation of point {observation.point} names camera "
                f"{observation.camera}; there are {camera_count} cameras"
            )
        groups.setdefault(int(observation.point), []).append(observation)
    if any(j < 0 for j in groups):
        raise ValueError("point indices must be non-negative")
    count = max(groups, default=-1) + 1
    for j in range(count):
        seen = len(groups.get(j, []))
        if seen < MIN_VIEWS:
            raise ValueError(f"point {j} has {seen} observation(s); triangulation needs 2")
    return [groups[j] for j in range(count)]


def _dlt_point(projections: NDArray[np.float64], uv: NDArray[np.float64]) -> NDArray[np.float64]:
    """Linear triangulation from 3x4 projection matrices ``K [R | t]`` and pixels."""
    rows = np.concatenate(
        [
            uv[:, :1] * projections[:, 2] - projections[:, 0],
            uv[:, 1:] * projections[:, 2] - projections[:, 1],
        ]
    )
    _, _, vt = np.linalg.svd(rows)
    homogeneous = vt[-1]
    return homogeneous[:3] / homogeneous[3]


def _reprojection_residuals(
    point: NDArray[np.float64],
    k: NDArray[np.float64],
    poses: NDArray[np.float64],
    uv: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Pixel residuals ``(u, v)`` per view, interleaved in observation order."""
    q = (k @ (poses[:, :, :3] @ point + poses[:, :, 3]).T).T
    return (q[:, :2] / q[:, 2:] - uv).ravel()
