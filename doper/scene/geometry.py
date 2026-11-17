# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Camera, pose-sampling, and keypoint-visibility math shared with the Blender contract.

Conventions (identical to ``rendering/synth_worker.py``):

* World is Z-up, meters.
* Camera-to-world matrices use the Blender/USD/OpenGL camera frame: the camera
  looks down its local ``-Z`` with ``+Y`` up. ``camera_extrinsics`` in the COCO
  output is this matrix, row-major.
* Pixel coordinates are continuous with the image spanning ``[0, W] x [0, H]``,
  origin top-left, so ``cx = W / 2`` and ``cy = H / 2``.
* Keypoint visibility: ``0`` outside the frame or behind the camera, ``1``
  occluded, ``2`` visible.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import __version__ as scipy_version
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics in pixels for an image of ``width`` x ``height``."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def matrix(self) -> np.ndarray:
        """The 3x3 camera matrix."""
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])


def intrinsics_from_lens(
    width: int, height: int, focal_length_mm: float, sensor_width_mm: float
) -> Intrinsics:
    """Intrinsics from a lens focal length and sensor width (Blender's formula)."""
    fx = focal_length_mm * width / sensor_width_mm
    return Intrinsics(int(width), int(height), fx, fx, width / 2.0, height / 2.0)


def horizontal_fov(focal_length_mm: float, sensor_width_mm: float) -> float:
    """Horizontal field of view in radians."""
    return 2.0 * math.atan(sensor_width_mm / (2.0 * focal_length_mm))


def auto_camera_distance(
    mesh_size: float,
    focal_length_mm: float,
    sensor_width_mm: float,
    fill_factor: tuple[float, float],
) -> tuple[float, float]:
    """Camera distance range so the object fills ``fill_factor`` of the frame width."""
    half_tan = math.tan(horizontal_fov(focal_length_mm, sensor_width_mm) / 2.0)
    lo, hi = fill_factor
    return (mesh_size / 2.0) / half_tan / hi, (mesh_size / 2.0) / half_tan / lo


# --------------------------------------------------------------------------
# Rotations
# --------------------------------------------------------------------------
def quat_to_matrix(q: Sequence[float]) -> np.ndarray:
    """(w, x, y, z) unit quaternion -> 3x3 rotation matrix."""
    w, x, y, z = (float(v) for v in q)
    # SciPy defaults to scalar-last; reorder explicitly for older releases too.
    return Rotation.from_quat((x, y, z, w)).as_matrix()


def euler_xyz_to_matrix(x: float, y: float, z: float) -> np.ndarray:
    """Blender ``Euler((x, y, z), 'XYZ')`` semantics: X applied first."""
    # Lowercase denotes extrinsic rotations, matching Blender's XYZ order.
    return Rotation.from_euler("xyz", (x, y, z)).as_matrix()


def make_pose(rotation: np.ndarray, translation: ArrayLike) -> np.ndarray:
    """Compose a 4x4 pose from a 3x3 rotation and a translation."""
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = np.asarray(translation, dtype=float)
    return pose


def look_at_camera(
    position: ArrayLike, target: ArrayLike, world_up: ArrayLike = (0.0, 0.0, 1.0)
) -> np.ndarray:
    """Camera-to-world 4x4 looking from ``position`` at ``target``.

    Matches Blender's ``direction.to_track_quat('-Z', 'Y')``: the camera's
    ``-Z`` points along the view direction and its ``+Y`` is as close to
    ``world_up`` as possible.
    """
    pos = np.asarray(position, dtype=float)
    forward = np.asarray(target, dtype=float) - pos
    norm = np.linalg.norm(forward)
    if norm <= 0:
        raise ValueError("camera position equals target")
    forward /= norm
    up = np.asarray(world_up, dtype=float)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, forward)
    rot = np.stack([right, cam_up, -forward], axis=1)
    return make_pose(rot, pos)


# --------------------------------------------------------------------------
# Projection and visibility
# --------------------------------------------------------------------------
def transform_points(pose: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 pose to ``(N, 3)`` points."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    return pts @ pose[:3, :3].T + pose[:3, 3]


def project_points(
    points_world: np.ndarray, cam_to_world: np.ndarray, intr: Intrinsics
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(uv (N,2) pixels, depth (N,))`` with depth along the optical axis."""
    world_to_cam = np.linalg.inv(cam_to_world)
    cam = transform_points(world_to_cam, points_world)
    depth = -cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = intr.fx * cam[:, 0] / depth + intr.cx
        v = intr.fy * (-cam[:, 1]) / depth + intr.cy
    return np.stack([u, v], axis=1), depth


def in_frame(uv: np.ndarray, depth: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """Mask of points in front of the camera and inside the image."""
    return (
        (depth > 0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] <= intr.width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] <= intr.height)
    )


def keypoint_visibility(
    uv: np.ndarray,
    depth: np.ndarray,
    depth_image: np.ndarray | None,
    intr: Intrinsics,
    tol: float,
    neighborhood: int = 1,
) -> np.ndarray:
    """Classify keypoints as 0 (out of frame), 1 (occluded), or 2 (visible).

    ``depth_image`` is a per-pixel distance-to-image-plane map (same metric as
    ``depth``). A keypoint counts as occluded when every pixel within
    ``neighborhood`` of its projection is nearer than ``depth - tol``; the
    neighborhood makes thin silhouette edges robust to half-pixel sampling.
    Pixels with no geometry (inf/nan/<=0) never occlude.
    """
    n = len(depth)
    vis = np.zeros(n, dtype=int)
    inside = in_frame(uv, depth, intr)
    vis[inside] = 2
    if depth_image is None:
        return vis
    h, w = depth_image.shape[:2]
    for i in np.flatnonzero(inside):
        col = min(max(int(math.floor(uv[i, 0])), 0), w - 1)
        row = min(max(int(math.floor(uv[i, 1])), 0), h - 1)
        r0, r1 = max(0, row - neighborhood), min(h, row + neighborhood + 1)
        c0, c1 = max(0, col - neighborhood), min(w, col + neighborhood + 1)
        patch = np.asarray(depth_image[r0:r1, c0:c1], dtype=float)
        valid = np.isfinite(patch) & (patch > 0)
        if not valid.any():
            continue
        if np.all(patch[valid] < depth[i] - tol):
            vis[i] = 1
    return vis


# --------------------------------------------------------------------------
# Scene sampling: library rotations with application-specific pose/camera rules
# --------------------------------------------------------------------------
def rotation_sampler_identity() -> dict[str, str]:
    """Backend identity for rejecting resume across numerical-library changes."""
    return {
        "algorithm": "scipy.Rotation.random/PCG64",
        "numpy": np.__version__,
        "scipy": scipy_version,
    }


def random_rotation(rng: random.Random) -> np.ndarray:
    """Haar-uniform SO(3) rotation, sampled by SciPy from a private RNG.

    Derive a NumPy stream from the caller's seeded stream; never use global
    random state. PCG64 is explicit so NumPy's default generator can evolve
    without silently changing our choice of algorithm.
    """
    generator = np.random.Generator(np.random.PCG64(rng.getrandbits(128)))
    # This keyword also supports SciPy versions predating the rng alias.
    rotation = Rotation.random(random_state=generator)  # pyright: ignore[reportCallIssue]
    return rotation.as_matrix()


def z_facing_rotation(rng: random.Random, jiggle_deg: float) -> np.ndarray:
    """Rotation with the object +Z facing up or down, with random tilt and spin."""
    z_rot = rng.uniform(0, 2 * math.pi)
    j = math.radians(jiggle_deg)
    tilt_x, tilt_y = rng.uniform(-j, j), rng.uniform(-j, j)
    if rng.random() < 0.5:
        return euler_xyz_to_matrix(tilt_x, tilt_y, z_rot)
    return euler_xyz_to_matrix(math.pi + tilt_x, tilt_y, z_rot)


def axis_facing_rotation(rng: random.Random, axis: str, jiggle_deg: float) -> np.ndarray:
    """Rotation aligning the object ``axis`` with world Z, with random tilt and spin."""
    base = z_facing_rotation(rng, jiggle_deg)
    if axis == "z":
        return base
    if axis == "x":
        return base @ euler_xyz_to_matrix(0, -math.pi / 2, 0)
    if axis == "y":
        return base @ euler_xyz_to_matrix(math.pi / 2, 0, 0)
    raise ValueError(f"unknown axis {axis!r}")


def sample_object_rotation(
    rng: random.Random, axis_bias: dict, jiggle_deg: float
) -> tuple[np.ndarray, bool]:
    """Sample the configured mixture of axis-biased and Haar-uniform rotations."""
    r = rng.random()
    c = 0.0
    for axis in ("x", "y", "z"):
        c += float(axis_bias.get(axis, 0.0))
        if r < c:
            return axis_facing_rotation(rng, axis, jiggle_deg), True
    return random_rotation(rng), False


def top_down_camera_direction(rng: random.Random, jiggle_deg: float) -> np.ndarray:
    """Unit camera direction near +Z, tilted by up to ``jiggle_deg``."""
    if jiggle_deg <= 0:
        return np.array([0.0, 0.0, 1.0])
    tilt = rng.uniform(0, math.radians(jiggle_deg))
    az = rng.uniform(0, 2 * math.pi)
    d = np.array([math.sin(tilt) * math.cos(az), math.sin(tilt) * math.sin(az), math.cos(tilt)])
    return d / np.linalg.norm(d)


def random_camera_direction(rng: random.Random) -> np.ndarray:
    """Unit camera direction on the upper hemisphere above a minimum elevation."""
    phi = rng.uniform(0, 2 * math.pi)
    cos_t = rng.uniform(0.1, 1.0)
    sin_t = math.sqrt(1 - cos_t**2)
    return np.array([sin_t * math.cos(phi), sin_t * math.sin(phi), cos_t])


def place_camera(
    rng: random.Random,
    center: np.ndarray,
    cam_dir: np.ndarray,
    cam_dist: float,
    jitter_strength: float,
) -> np.ndarray:
    """Camera-to-world for a camera at ``center + dir * dist`` looking near ``center``."""
    cam_pos = center + cam_dir * cam_dist
    look_at = center.copy()
    if jitter_strength > 0:
        angle = rng.uniform(0, 2 * math.pi)
        right = np.cross(cam_dir, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(right) < 0.01:
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, cam_dir)
        up /= np.linalg.norm(up)
        look_at = (
            center
            + (right * math.cos(angle) + up * math.sin(angle)) * jitter_strength * cam_dist * 0.5
        )
    return look_at_camera(cam_pos, look_at)


def rest_on_ground(
    rotation: np.ndarray,
    local_bbox_corners: np.ndarray,
    clearance: float,
    xy_offset: Sequence[float],
) -> np.ndarray:
    """Object pose with the rotated AABB bottom at ``clearance`` above z=0."""
    rotated = local_bbox_corners @ rotation.T
    lift = -rotated[:, 2].min() + clearance
    return make_pose(rotation, (float(xy_offset[0]), float(xy_offset[1]), lift))


def aabb_corners(points: np.ndarray) -> np.ndarray:
    """The eight corners of the axis-aligned bounding box of ``points``."""
    lo, hi = np.min(points, axis=0), np.max(points, axis=0)
    return np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    )


def stereo_offset(cam_to_world: np.ndarray, offset_m: float) -> np.ndarray:
    """Shift the camera along its own +X (baseline direction), keeping rotation."""
    shifted = cam_to_world.copy()
    shifted[:3, 3] = cam_to_world[:3, 3] + cam_to_world[:3, 0] * offset_m
    return shifted
