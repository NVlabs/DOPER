# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Line-of-sight visibility of surface points against a triangle mesh.

Written from the ray-triangle intersection of Möller and Trumbore, "Fast, Minimum Storage
Ray/Triangle Intersection" (Journal of Graphics Tools 2(1), 1997), vectorized over points and
triangles with NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Determinants below this (relative to the edge and ray lengths) mean the ray is parallel to
# the triangle's plane; such a triangle cannot block it.
PARALLEL_EPSILON = 1e-12
# Barycentric slack: a ray through a shared edge or vertex counts as a hit on both triangles,
# so a closed mesh has no cracks along its edges.
BARYCENTRIC_EPSILON = 1e-9
DEFAULT_CHUNK = 64


@dataclass(frozen=True)
class MeshEdges:
    """A triangle mesh as the ray test reads it, computed once per mesh (``mesh_edges``).

    Attributes:
        v0: First vertex of each triangle ``(f, 3)``.
        edge1: ``v1 - v0`` per triangle.
        edge2: ``v2 - v0`` per triangle.
        scale: ``|edge1| * |edge2|`` per triangle, the parallel test's scale.
    """

    v0: NDArray[np.float64]
    edge1: NDArray[np.float64]
    edge2: NDArray[np.float64]
    scale: NDArray[np.float64]


def mesh_edges(vertices: ArrayLike, faces: ArrayLike) -> MeshEdges:
    """The ``MeshEdges`` of a mesh, for many ``ray_mesh_visible_edges`` calls.

    Raises:
        ValueError: On malformed vertices or faces.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    _check_mesh(vertices, faces)
    return _edges(vertices, faces)


def ray_mesh_visible(
    points_obj: ArrayLike,
    camera_centre_obj: ArrayLike,
    vertices: ArrayLike,
    faces: ArrayLike,
    *,
    tolerance_m: float,
    chunk: int = DEFAULT_CHUNK,
) -> NDArray[np.bool_]:
    """Whether each point is visible from the camera center past the mesh.

    A point is visible when the segment from the camera center to the point meets no
    triangle at a distance from the camera in ``(0, |point - camera| - tolerance_m)``. The
    tolerance lets points on (or just under) the surface see past their own triangles;
    intersections behind the camera are ignored.

    Args:
        points_obj: Points of shape ``(n, 3)`` in the mesh (object) frame, in meters.
        camera_centre_obj: Camera center ``(3,)`` in the same frame.
        vertices: Mesh vertices ``(v, 3)``.
        faces: Triangle vertex indices ``(f, 3)``.
        tolerance_m: Non-negative distance short of each point within which hits are ignored.
        chunk: Points tested per vectorized batch (memory is ``chunk * f``).

    Returns:
        Boolean array of shape ``(n,)``.

    Raises:
        ValueError: On malformed points, camera, vertices, faces, tolerance or chunk.
    """
    points = np.asarray(points_obj, dtype=np.float64)
    camera = np.asarray(camera_centre_obj, dtype=np.float64).reshape(-1)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    _check_points(points, camera)
    _check_mesh(vertices, faces)
    _check_settings(tolerance_m, chunk)
    return _visible(points, camera, _edges(vertices, faces), tolerance_m, chunk)


def ray_mesh_visible_edges(
    points_obj: ArrayLike,
    camera_centre_obj: ArrayLike,
    edges: MeshEdges,
    *,
    tolerance_m: float,
    chunk: int = DEFAULT_CHUNK,
) -> NDArray[np.bool_]:
    """``ray_mesh_visible`` against a mesh's precomputed ``mesh_edges``.

    Raises:
        ValueError: On malformed points, camera, tolerance or chunk.
    """
    points = np.asarray(points_obj, dtype=np.float64)
    camera = np.asarray(camera_centre_obj, dtype=np.float64).reshape(-1)
    _check_points(points, camera)
    _check_settings(tolerance_m, chunk)
    return _visible(points, camera, edges, tolerance_m, chunk)


def _edges(vertices: NDArray[np.float64], faces: NDArray) -> MeshEdges:
    v0 = vertices[faces[:, 0]]
    edge1 = vertices[faces[:, 1]] - v0
    edge2 = vertices[faces[:, 2]] - v0
    scale = np.linalg.norm(edge1, axis=1) * np.linalg.norm(edge2, axis=1)
    return MeshEdges(v0, edge1, edge2, scale)


def _visible(
    points: NDArray[np.float64],
    camera: NDArray[np.float64],
    edges: MeshEdges,
    tolerance_m: float,
    chunk: int,
) -> NDArray[np.bool_]:
    offset = camera - edges.v0
    visible = np.ones(len(points), dtype=bool)
    for start in range(0, len(points), chunk):
        stop = min(start + chunk, len(points))
        visible[start:stop] = ~_blocked(points[start:stop] - camera, offset, edges, tolerance_m)
    return visible


def _blocked(
    rays: NDArray[np.float64],
    offset: NDArray[np.float64],
    edges: MeshEdges,
    tolerance_m: float,
) -> NDArray[np.bool_]:
    """Möller-Trumbore for rays ``(m, 3)`` (camera to point) against ``f`` triangles."""
    edge1, edge2 = edges.edge1, edges.edge2
    length = np.linalg.norm(rays, axis=1)
    direction = rays / np.where(length > 0.0, length, 1.0)[:, None]
    p = np.cross(direction[:, None, :], edge2[None, :, :])
    det = np.einsum("fk,mfk->mf", edge1, p)
    usable = np.abs(det) > PARALLEL_EPSILON * edges.scale[None, :]
    inv_det = np.divide(1.0, det, out=np.zeros_like(det), where=usable)
    u = np.einsum("fk,mfk->mf", offset, p) * inv_det
    q = np.cross(offset, edge1)
    v = np.einsum("mk,fk->mf", direction, q) * inv_det
    t = np.einsum("fk,fk->f", edge2, q)[None, :] * inv_det
    inside = (
        (u >= -BARYCENTRIC_EPSILON)
        & (v >= -BARYCENTRIC_EPSILON)
        & (u + v <= 1.0 + BARYCENTRIC_EPSILON)
    )
    in_front = (t > 0.0) & (t < (length - tolerance_m)[:, None])
    return (usable & inside & in_front).any(axis=1)


def _check_points(points: NDArray[np.float64], camera: NDArray[np.float64]) -> None:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_obj must have shape (n, 3), got {points.shape}")
    if camera.shape != (3,):
        raise ValueError(f"camera_centre_obj must be a 3-vector, got shape {camera.shape}")


def _check_mesh(vertices: NDArray[np.float64], faces: NDArray) -> None:
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape (v, 3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError(f"faces must be integer indices of shape (f, 3), got {faces.shape}")
    if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("faces index vertices out of range")


def _check_settings(tolerance_m: float, chunk: int) -> None:
    if not np.isfinite(tolerance_m) or tolerance_m < 0.0:
        raise ValueError(f"tolerance_m must be non-negative, got {tolerance_m}")
    if chunk < 1:
        raise ValueError(f"chunk must be positive, got {chunk}")
