# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hull-anchored keypoint selection for a CAD mesh.

Picks ``N`` mesh vertices as 3D keypoints: a few visible convex-hull vertices anchor the layout,
surface-saliency farthest-point sampling fills the rest, and ray-cast visibility repair swaps out
any keypoint that is hidden from most viewpoints. The result is written as the keypoints JSON
that ``rendering/mesh_setup.py --keypoints-input`` and ``doper`` read.

Example:
    python -m doper.keypoint_selection --mesh object.stl --source-units mm
        --out-json keypoints.json --keypoint-prefix KP_object_ (one command line)

Coordinates are in meters. ``numba`` (``doper[keypoints]``) only speeds up the ray casting; the
NumPy path gives the same visibility mask.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import struct
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

EPS = 1e-12
SOURCE_UNIT_SCALES: dict[str, float] = {"m": 1.0, "cm": 0.01, "mm": 0.001, "in": 0.0254}
OBJECT_FRAMES = ("cad", "aabb_centered")
DEFAULT_METHOD = "surface_saliency_fps_hull_anchors"
DEFAULT_NUM_KEYPOINTS = 64
DEFAULT_NUM_HULL_ANCHORS = 8
DEFAULT_VISIBILITY_VIEWS = 48
DEFAULT_MIN_VISIBILITY_RATE = 0.10
DEFAULT_KEYPOINT_PREFIX = "KP_"
MESH_DECIMALS = 9
KEYPOINT_DECIMALS = 9
HULL_POOL_FRACTION = 0.25
HULL_POOL_MULTIPLIER = 10
FILL_POOL_FRACTION = 0.25
FILL_POOL_MULTIPLIER = 3
REPAIR_CANDIDATE_LIMIT = 256
VISIBILITY_IMAGE_SIZE = 384
VISIBILITY_FOCAL_LENGTH_PX = 480.0
VISIBILITY_DISTANCE_SCALE = 6.0
STL_HEADER_BYTES = 84
STL_RECORD_BYTES = 50

MaskKernel = Callable[..., np.ndarray]


@dataclass(frozen=True)
class SurfaceMesh:
    """Triangle mesh with deduplicated vertices.

    Attributes:
        vertices: ``(V, 3)`` float64 positions.
        faces: ``(F, 3)`` int64 vertex indices.
    """

    vertices: np.ndarray
    faces: np.ndarray


@dataclass(frozen=True)
class AnchorReplacement:
    """One keypoint that visibility repair swapped for a better-seen vertex.

    Attributes:
        position: Position of the keypoint in the layout.
        old_index: Mesh vertex index before the swap.
        new_index: Mesh vertex index after the swap.
        old_visibility_rate: Fraction of views that saw the old vertex.
        new_visibility_rate: Fraction of views that see the new vertex.
    """

    position: int
    old_index: int
    new_index: int
    old_visibility_rate: float
    new_visibility_rate: float


@dataclass(frozen=True)
class KeypointLayout:
    """A selected keypoint layout.

    Attributes:
        points: ``(N, 3)`` keypoint positions in meters.
        indices: ``(N,)`` mesh vertex indices of ``points``; the first anchors come first.
        initial_anchor_indices: Hull anchors before visibility repair.
        replaced_anchors: Anchors that visibility repair swapped out, in repair order.
        saliency: ``(N,)`` surface saliency of each keypoint, in ``[0, 1]``.
        visibility_rate: ``(N,)`` fraction of the repair views that see each keypoint.
        method: Name of the selection method.
    """

    points: np.ndarray
    indices: np.ndarray
    initial_anchor_indices: np.ndarray
    replaced_anchors: tuple[AnchorReplacement, ...]
    saliency: np.ndarray
    visibility_rate: np.ndarray
    method: str


@functools.lru_cache(maxsize=1)
def load_numba_kernel() -> MaskKernel | None:
    """Compile the ray-casting visibility kernel with numba.

    Returns:
        The jitted kernel, or None when numba is not installed.
    """
    try:
        import numba
    except ImportError:
        return None

    @numba.njit(parallel=True)
    def kernel(
        eye: np.ndarray,
        points: np.ndarray,
        tri_v0: np.ndarray,
        tri_e1: np.ndarray,
        tri_e2: np.ndarray,
        tolerance: float,
        eps: float,
    ) -> np.ndarray:
        visible = np.ones(points.shape[0], dtype=np.bool_)
        for point_idx in numba.prange(points.shape[0]):
            ray = points[point_idx] - eye
            ray_len = math.sqrt(ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2])
            if ray_len <= eps:
                visible[point_idx] = False
                continue
            direction = ray / ray_len
            occluded = False
            for tri_idx in range(tri_v0.shape[0]):
                e1 = tri_e1[tri_idx]
                e2 = tri_e2[tri_idx]
                p_x = direction[1] * e2[2] - direction[2] * e2[1]
                p_y = direction[2] * e2[0] - direction[0] * e2[2]
                p_z = direction[0] * e2[1] - direction[1] * e2[0]
                det = e1[0] * p_x + e1[1] * p_y + e1[2] * p_z
                if abs(det) <= eps:
                    continue
                inv_det = 1.0 / det
                tvec = eye - tri_v0[tri_idx]
                u = (tvec[0] * p_x + tvec[1] * p_y + tvec[2] * p_z) * inv_det
                if u < -eps or u > 1.0 + eps:
                    continue
                q_x = tvec[1] * e1[2] - tvec[2] * e1[1]
                q_y = tvec[2] * e1[0] - tvec[0] * e1[2]
                q_z = tvec[0] * e1[1] - tvec[1] * e1[0]
                v = (direction[0] * q_x + direction[1] * q_y + direction[2] * q_z) * inv_det
                if v < -eps or u + v > 1.0 + eps:
                    continue
                t = (e2[0] * q_x + e2[1] * q_y + e2[2] * q_z) * inv_det
                if t > tolerance and t < ray_len - tolerance:
                    occluded = True
                    break
            visible[point_idx] = not occluded
        return visible

    return kernel


def as_points(points: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Validate and convert to a finite ``(N, 3)`` float64 array.

    Args:
        points: Array-like of 3D points.

    Returns:
        The points as float64.

    Raises:
        ValueError: The shape is not ``(N, 3)``, the array is empty, or a value is not finite.
    """
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"expected points with shape (N, 3), got {arr.shape}")
    if len(arr) == 0:
        raise ValueError("point array is empty")
    if not np.isfinite(arr).all():
        raise ValueError("point array contains NaN or inf")
    return arr


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Scale finite 1D scores to ``[0, 1]``; constant scores become zeros."""
    low = float(np.min(scores))
    high = float(np.max(scores))
    if high - low < EPS:
        return np.zeros_like(scores)
    return (scores - low) / (high - low)


def build_surface_mesh(triangles: np.ndarray) -> SurfaceMesh:
    """Deduplicate triangle corners into an indexed mesh.

    Corners are rounded to ``MESH_DECIMALS`` decimals before matching. Vertices keep first-seen
    order and degenerate faces are dropped.

    Args:
        triangles: ``(T, 3, 3)`` triangle corner positions in meters.

    Returns:
        The indexed mesh.
    """
    rounded = np.round(triangles.reshape(-1, 3), decimals=MESH_DECIMALS)
    _, first_seen, inverse = np.unique(rounded, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first_seen)
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    faces = rank[inverse.reshape(-1)].reshape(-1, 3)
    keep = (
        (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    )
    return SurfaceMesh(vertices=rounded[np.sort(first_seen)], faces=faces[keep])


def _binary_stl_triangles(data: bytes) -> np.ndarray | None:
    if len(data) < STL_HEADER_BYTES:
        return None
    count = struct.unpack_from("<I", data, 80)[0]
    if count == 0 or STL_HEADER_BYTES + count * STL_RECORD_BYTES != len(data):
        return None
    records = np.frombuffer(data, dtype=np.uint8, offset=STL_HEADER_BYTES).reshape(count, -1)
    corners = np.ascontiguousarray(records[:, 12:48]).view("<f4")
    return corners.reshape(count, 3, 3).astype(np.float64)


def _ascii_stl_triangles(data: bytes, path: Path) -> np.ndarray:
    vertices = []
    for line in data.decode("utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise ValueError(f"could not parse STL vertices from {path}")
    if len(vertices) % 3 != 0:
        raise ValueError(f"ASCII STL vertex count is not divisible by 3: {path}")
    return np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 3)


def _read_stl(path: Path) -> np.ndarray:
    data = path.read_bytes()
    triangles = _binary_stl_triangles(data)
    return triangles if triangles is not None else _ascii_stl_triangles(data, path)


def _fan_triangulate(polygons: Sequence[Sequence[int]]) -> np.ndarray:
    return np.asarray(
        [[poly[0], poly[i], poly[i + 1]] for poly in polygons for i in range(1, len(poly) - 1)],
        dtype=np.int64,
    ).reshape(-1, 3)


def _read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    polygons: list[list[int]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v" and len(parts) >= 4:
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif parts[0] == "f" and len(parts) >= 4:
            refs = [int(part.split("/")[0]) for part in parts[1:]]
            polygons.append([ref - 1 if ref > 0 else len(vertices) + ref for ref in refs])
    return np.asarray(vertices, dtype=np.float64).reshape(-1, 3), _fan_triangulate(polygons)


PLY_DTYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
    "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
}  # fmt: skip


def _parse_ply_header(data: bytes, path: Path) -> tuple[str, list[tuple[str, int, list]], int]:
    end = data.find(b"end_header")
    if not data.startswith(b"ply") or end < 0:
        raise ValueError(f"not a PLY file: {path}")
    body_start = data.index(b"\n", end) + 1
    file_format = ""
    elements: list[tuple[str, int, list]] = []
    for line in data[:end].decode("ascii", errors="ignore").splitlines():
        parts = line.split()
        if parts[:1] == ["format"]:
            file_format = parts[1]
        elif parts[:1] == ["element"]:
            elements.append((parts[1], int(parts[2]), []))
        elif parts[:1] == ["property"]:
            elements[-1][2].append(tuple(parts[1:]))
    return file_format, elements, body_start


def _ply_vertices_ascii(tokens: list[bytes], cursor: int, count: int, props: list) -> tuple:
    names = [prop[-1] for prop in props]
    block = np.asarray(tokens[cursor : cursor + count * len(props)], dtype=np.float64)
    table = block.reshape(count, len(props))
    columns = [table[:, names.index(axis)] for axis in "xyz"]
    return np.stack(columns, axis=1).astype(np.float64), cursor + count * len(props)


def _ply_vertices_binary(data: bytes, offset: int, count: int, props: list) -> tuple:
    dtype = np.dtype([(prop[1], PLY_DTYPES[prop[0]]) for prop in props])
    table = np.frombuffer(data, dtype=dtype, count=count, offset=offset)
    columns = [np.asarray(table[axis]) for axis in "xyz"]  # type: ignore[call-overload]
    return np.stack(columns, axis=1).astype(np.float64), offset + count * dtype.itemsize


def _ply_faces_ascii(tokens: list[bytes], cursor: int, count: int, props: list) -> tuple:
    polygons: list[list[int]] = []
    for _ in range(count):
        size = int(tokens[cursor])
        polygons.append([int(tok) for tok in tokens[cursor + 1 : cursor + 1 + size]])
        cursor += 1 + size
    return polygons, cursor


def _ply_faces_binary(data: bytes, offset: int, count: int, props: list) -> tuple:
    count_type, index_type = PLY_DTYPES[props[0][1]], PLY_DTYPES[props[0][2]]
    polygons: list[list[int]] = []
    for _ in range(count):
        size = int(np.frombuffer(data, dtype=count_type, count=1, offset=offset)[0])
        offset += np.dtype(count_type).itemsize
        refs = np.frombuffer(data, dtype=index_type, count=size, offset=offset)
        polygons.append(refs.tolist())
        offset += size * np.dtype(index_type).itemsize
    return polygons, offset


def _read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    file_format, elements, offset = _parse_ply_header(data, path)
    if file_format not in ("ascii", "binary_little_endian"):
        raise ValueError(f"unsupported PLY format {file_format!r}: {path}")
    if [name for name, _, _ in elements[:2]] != ["vertex", "face"]:
        raise ValueError(f"PLY needs 'vertex' then 'face' elements: {path}")
    source: Any
    read_vertices: Callable[..., tuple]
    read_faces: Callable[..., tuple]
    if file_format == "ascii":
        source, cursor = data[offset:].split(), 0
        read_vertices, read_faces = _ply_vertices_ascii, _ply_faces_ascii
    else:
        source, cursor = data, offset
        read_vertices, read_faces = _ply_vertices_binary, _ply_faces_binary
    vertices = np.empty((0, 3))
    polygons: list[list[int]] = []
    for name, count, props in elements[:2]:
        if name == "vertex":
            names = [prop[-1] for prop in props]
            if any(len(prop) != 2 for prop in props) or not {"x", "y", "z"} <= set(names):
                raise ValueError(f"PLY vertex properties must be scalars including x, y, z: {path}")
            vertices, cursor = read_vertices(source, cursor, count, props)
        else:
            if len(props) != 1 or props[0][0] != "list":
                raise ValueError(f"PLY face element must hold one list property: {path}")
            polygons, cursor = read_faces(source, cursor, count, props)
    return vertices, _fan_triangulate(polygons)


def load_mesh(path: Path, source_units: str) -> tuple[np.ndarray, np.ndarray]:
    """Load an STL, OBJ or PLY mesh as deduplicated meter-scale vertices and faces.

    Args:
        path: Mesh file. STL may be binary or ASCII; OBJ and PLY (ASCII or binary little endian)
            polygons are fan-triangulated.
        source_units: Length unit of the file's coordinates: ``m``, ``cm``, ``mm`` or ``in``.

    Returns:
        ``(vertices, faces)``: ``(V, 3)`` float64 meters and ``(F, 3)`` int64 indices. Corners
        are rounded to nine decimals (meters) and merged; degenerate faces are dropped.

    Raises:
        ValueError: Unknown ``source_units``, unsupported file extension or unparsable file.
    """
    if source_units not in SOURCE_UNIT_SCALES:
        raise ValueError(
            f"source_units must be one of {sorted(SOURCE_UNIT_SCALES)}, got {source_units!r}"
        )
    path = Path(path)
    suffix = path.suffix.lower()
    scale = SOURCE_UNIT_SCALES[source_units]
    if suffix == ".stl":
        triangles = _read_stl(path)
    elif suffix in (".obj", ".ply"):
        vertices, faces = _read_obj(path) if suffix == ".obj" else _read_ply(path)
        triangles = vertices[faces] if len(faces) else np.empty((0, 3, 3))
    else:
        raise ValueError(f"unsupported mesh format {suffix!r}; use .stl, .obj or .ply")
    if len(triangles) == 0:
        raise ValueError(f"mesh has no triangles: {path}")
    mesh = build_surface_mesh(triangles * scale)
    return mesh.vertices, mesh.faces


def _fps(points: np.ndarray, count: int, distances: np.ndarray, selected: list[int]) -> np.ndarray:
    """Greedy farthest-point picks on validated points until ``count`` indices are selected.

    Args:
        points: ``(N, 3)`` points.
        count: Target size of ``selected``; at most ``N``.
        distances: Per-point distance to the fixed seeds; modified in place.
        selected: Indices already picked; extended in place.

    Returns:
        ``selected`` as an int64 array.
    """
    distances[selected] = -np.inf
    while len(selected) < count:
        next_idx = int(np.argmax(distances))
        selected.append(next_idx)
        distances = np.minimum(distances, np.linalg.norm(points - points[next_idx], axis=1))
        distances[selected] = -np.inf
    return np.asarray(selected, dtype=np.int64)


def _seeded_fps(points: np.ndarray, count: int, seed_points: np.ndarray | None) -> np.ndarray:
    """Farthest-point sampling of validated points, measured from validated seeds first."""
    count = min(int(count), len(points))
    if seed_points is None or len(seed_points) == 0:
        centroid = points.mean(axis=0)
        first = int(np.argmax(np.linalg.norm(points - centroid, axis=1)))
        second = int(np.argmax(np.linalg.norm(points - points[first], axis=1)))
        selected = [first]
        if second != first and count > 1:
            selected.append(second)
        seeds = points[selected]
    else:
        selected = []
        seeds = seed_points
    distances = np.full(len(points), np.inf)
    for seed in seeds:
        distances = np.minimum(distances, np.linalg.norm(points - seed, axis=1))
    return _fps(points, count, distances, selected)


def farthest_point_indices(points: np.ndarray, count: int) -> np.ndarray:
    """Greedy farthest-point sampling that starts from the diameter end points.

    Args:
        points: ``(N, 3)`` points.
        count: Number of points to pick; clipped to ``N``.

    Returns:
        Selected indices in pick order.
    """
    pts = as_points(points)
    if count <= 0:
        raise ValueError("count must be positive")
    return _seeded_fps(pts, count, None)


def farthest_point_indices_from_seed_points(
    points: np.ndarray, count: int, *, seed_points: np.ndarray | None = None
) -> np.ndarray:
    """Farthest-point sampling that measures distance from fixed seed points first.

    Seeds constrain the choice but are never returned. With no seeds this is
    ``farthest_point_indices``.

    Args:
        points: ``(N, 3)`` candidate points.
        count: Number of points to pick; clipped to ``N``.
        seed_points: Optional ``(S, 3)`` fixed points.

    Returns:
        Indices into ``points`` in pick order.
    """
    pts = as_points(points)
    if count <= 0:
        raise ValueError("count must be positive")
    seeds = None if seed_points is None or np.asarray(seed_points).size == 0 else seed_points
    return _seeded_fps(pts, count, None if seeds is None else as_points(seeds))


def face_normals_and_areas(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return unit face normals (zero for degenerate faces) and face areas."""
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(cross, axis=1)
    normals = np.zeros_like(cross)
    valid = norm > EPS
    normals[valid] = cross[valid] / norm[valid, None]
    return normals, 0.5 * norm


def surface_saliency_scores(
    mesh: SurfaceMesh,
    *,
    sharpness_weight: float = 0.55,
    normal_variation_weight: float = 0.35,
    boundary_weight: float = 0.10,
) -> np.ndarray:
    """Compute per-vertex surface saliency from geometry alone.

    The score mixes ``sharpness`` (dihedral angle times edge length over incident edges),
    ``normal_variation`` (disagreement between adjacent face normals) and ``boundary`` (open
    mesh edges). Higher means a more locally distinctive surface point.

    Args:
        mesh: Indexed triangle mesh.
        sharpness_weight: Weight of the sharpness term.
        normal_variation_weight: Weight of the normal-variation term.
        boundary_weight: Weight of the open-boundary term.

    Returns:
        ``(V,)`` saliency scores in ``[0, 1]``.
    """
    vertices, faces = mesh.vertices, mesh.faces
    normals, areas = face_normals_and_areas(vertices, faces)
    edge_faces: dict[tuple[int, int], list[int]] = {}
    vertex_faces: list[list[int]] = [[] for _ in range(len(vertices))]
    for face_idx, face in enumerate(faces):
        for vertex_idx in face:
            vertex_faces[int(vertex_idx)].append(face_idx)
        for edge in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            low, high = sorted((int(edge[0]), int(edge[1])))
            edge_faces.setdefault((low, high), []).append(face_idx)
    sharpness = np.zeros(len(vertices))
    boundary = np.zeros(len(vertices))
    for (a, b), adjacent in edge_faces.items():
        if len(adjacent) == 1:
            boundary[a] = boundary[b] = 1.0
            continue
        max_angle = 0.0
        for i, face_i in enumerate(adjacent):
            for face_j in adjacent[i + 1 :]:
                dot = float(np.clip(np.dot(normals[face_i], normals[face_j]), -1.0, 1.0))
                max_angle = max(max_angle, math.acos(dot))
        contribution = max_angle * float(np.linalg.norm(vertices[a] - vertices[b]))
        sharpness[a] += contribution
        sharpness[b] += contribution
    normal_variation = np.zeros(len(vertices))
    for vertex_idx, face_indices in enumerate(vertex_faces):
        if not face_indices:
            continue
        index_array = np.asarray(face_indices, dtype=np.int64)
        weights = areas[index_array]
        total_weight = float(weights.sum())
        if total_weight <= EPS:
            continue
        mean_norm = float(np.linalg.norm((normals[index_array] * weights[:, None]).sum(axis=0)))
        normal_variation[vertex_idx] = 1.0 - mean_norm / total_weight
    return normalize_scores(
        sharpness_weight * normalize_scores(sharpness)
        + normal_variation_weight * normalize_scores(normal_variation)
        + boundary_weight * normalize_scores(boundary)
    )


def fibonacci_sphere(count: int) -> np.ndarray:
    """Return ``count`` near-uniform unit directions."""
    directions = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(count):
        z = 1.0 - (2.0 * (i + 0.5) / count)
        radius = math.sqrt(max(0.0, 1.0 - z * z))
        theta = golden_angle * i
        directions.append([math.cos(theta) * radius, math.sin(theta) * radius, z])
    return np.asarray(directions, dtype=np.float64)


def look_at_object_to_camera(eye: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the rotation and translation taking object points into a camera at ``eye``."""
    forward = target - eye
    forward = forward / max(np.linalg.norm(forward), EPS)
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(forward, up))) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right = right / max(np.linalg.norm(right), EPS)
    down = np.cross(forward, right)
    down = down / max(np.linalg.norm(down), EPS)
    rotation = np.stack([right, down, forward], axis=0)
    return rotation, -rotation @ eye


def triangle_edge_arrays(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return each triangle's first corner and its two edge vectors, C-contiguous."""
    triangles = vertices[faces]
    tri_v0 = np.ascontiguousarray(triangles[:, 0], dtype=np.float64)
    tri_e1 = np.ascontiguousarray(triangles[:, 1] - tri_v0, dtype=np.float64)
    tri_e2 = np.ascontiguousarray(triangles[:, 2] - tri_v0, dtype=np.float64)
    return tri_v0, tri_e1, tri_e2


def ray_triangle_hits(
    origin: np.ndarray, direction: np.ndarray, v0: np.ndarray, edge1: np.ndarray, edge2: np.ndarray
) -> np.ndarray:
    """Vectorized Moller-Trumbore: ray parameters of every forward hit (``t > EPS``)."""
    pvec = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    det = np.einsum("ij,ij->i", edge1, pvec)
    valid = np.abs(det) > EPS
    if not np.any(valid):
        return np.asarray([], dtype=np.float64)
    inv_det = np.zeros_like(det)
    inv_det[valid] = 1.0 / det[valid]
    tvec = origin - v0
    u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
    valid &= (u >= -EPS) & (u <= 1.0 + EPS)
    if not np.any(valid):
        return np.asarray([], dtype=np.float64)
    qvec = np.cross(tvec, edge1)
    v = np.einsum("j,ij->i", direction, qvec) * inv_det
    valid &= (v >= -EPS) & (u + v <= 1.0 + EPS)
    if not np.any(valid):
        return np.asarray([], dtype=np.float64)
    t = np.einsum("ij,ij->i", edge2, qvec) * inv_det
    valid &= t > EPS
    return t[valid]


def visibility_mask_from_edges(
    points: np.ndarray,
    *,
    eye: np.ndarray,
    tri_v0: np.ndarray,
    tri_e1: np.ndarray,
    tri_e2: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Visibility of ``points`` from ``eye`` given precomputed triangle edges.

    A point is hidden when the ray from ``eye`` hits a triangle strictly between ``tolerance``
    and the point's distance minus ``tolerance``.
    """
    query = np.ascontiguousarray(points, dtype=np.float64)
    eye = np.ascontiguousarray(np.asarray(eye, dtype=np.float64).reshape(3))
    kernel = load_numba_kernel()
    if kernel is not None:
        return kernel(eye, query, tri_v0, tri_e1, tri_e2, float(tolerance), float(EPS))
    visible = np.ones(len(query), dtype=bool)
    for idx, point in enumerate(query):
        ray = point - eye
        ray_len = float(np.linalg.norm(ray))
        if ray_len <= EPS:
            visible[idx] = False
            continue
        hits = ray_triangle_hits(eye, ray / ray_len, tri_v0, tri_e1, tri_e2)
        if np.any((hits > tolerance) & (hits < ray_len - tolerance)):
            visible[idx] = False
    return visible


def exact_visibility_rates(
    mesh: SurfaceMesh, points: np.ndarray, *, num_views: int = DEFAULT_VISIBILITY_VIEWS
) -> np.ndarray:
    """Fraction of views in which each point is in frame and not self-occluded.

    Cameras sit on a Fibonacci sphere around the vertex centroid at six object diagonals and
    look at the centroid through a 384 px, 480 px focal pinhole.

    Args:
        mesh: Occluding mesh.
        points: ``(N, 3)`` query points.
        num_views: Number of camera directions.

    Returns:
        ``(N,)`` rates in ``[0, 1]``.
    """
    query, vertices = points, mesh.vertices
    target = vertices.mean(axis=0)
    object_scale = max(float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))), EPS)
    radius = max(object_scale * VISIBILITY_DISTANCE_SCALE, object_scale + EPS)
    tolerance = max(object_scale * 1e-4, 1e-6)
    tri_v0, tri_e1, tri_e2 = triangle_edge_arrays(vertices, mesh.faces)
    views = fibonacci_sphere(max(int(num_views), 1))
    counts = np.zeros(len(query), dtype=np.int64)
    centre = VISIBILITY_IMAGE_SIZE / 2.0
    upper = VISIBILITY_IMAGE_SIZE - 1.0
    for direction in views:
        eye = target + direction * radius
        rotation, translation = look_at_object_to_camera(eye, target)
        camera_points = (rotation @ query.T).T + translation
        depth = camera_points[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = VISIBILITY_FOCAL_LENGTH_PX * (camera_points[:, 0] / depth) + centre
            v = VISIBILITY_FOCAL_LENGTH_PX * (camera_points[:, 1] / depth) + centre
        in_frame = (depth > EPS) & (u >= 0.0) & (u <= upper) & (v >= 0.0) & (v <= upper)
        counts += (
            in_frame
            & visibility_mask_from_edges(
                query, eye=eye, tri_v0=tri_v0, tri_e1=tri_e1, tri_e2=tri_e2, tolerance=tolerance
            )
        ).astype(np.int64)
    return counts.astype(np.float64) / max(len(views), 1)


def visibility_refined_indices(
    mesh: SurfaceMesh,
    selected_indices: np.ndarray | Sequence[int],
    saliency_scores: np.ndarray | Sequence[float],
    *,
    min_visibility_rate: float = 0.25,
    num_views: int = DEFAULT_VISIBILITY_VIEWS,
    exact_candidate_limit: int = REPAIR_CANDIDATE_LIMIT,
    saliency_weight: float = 0.45,
    visibility_weight: float = 0.35,
    spread_weight: float = 0.20,
) -> tuple[np.ndarray, list[AnchorReplacement], np.ndarray]:
    """Replace selected vertices whose exact visibility is below a threshold.

    Replacement candidates are farthest-point samples of the unselected vertices, seeded by the
    keypoints that are kept. Candidates that meet the threshold are scored by saliency, visibility
    and spread, and the worst keypoints are replaced first.

    Args:
        mesh: Indexed triangle mesh.
        selected_indices: Layout vertex indices.
        saliency_scores: Per-vertex saliency.
        min_visibility_rate: Rate below which a keypoint is replaced.
        num_views: Number of camera directions.
        exact_candidate_limit: Maximum replacement candidates to ray cast.
        saliency_weight: Weight of saliency in the replacement score.
        visibility_weight: Weight of visibility in the replacement score.
        spread_weight: Weight of distance from the other keypoints in the replacement score.

    Returns:
        ``(indices, swaps, after_rates)``: the refined indices, the replacements made, and the
        visibility rates of the layout after repair.
    """
    vertices = mesh.vertices
    selected = np.asarray(selected_indices, dtype=np.int64).copy()
    saliency = np.asarray(saliency_scores, dtype=np.float64)
    if selected.ndim != 1 or len(selected) == 0:
        raise ValueError("selected_indices must be a non-empty one-dimensional array")
    if np.any(selected < 0) or np.any(selected >= len(vertices)):
        raise IndexError("selected index out of bounds")
    before = exact_visibility_rates(mesh, vertices[selected], num_views=num_views)
    bad_positions = np.flatnonzero(before < float(min_visibility_rate))
    if len(bad_positions) == 0:
        return selected, [], before
    saliency_norm = normalize_scores(saliency)
    selected_set = {int(idx) for idx in selected}
    bad_set = {int(pos) for pos in bad_positions}
    fixed = np.asarray([p for p in range(len(selected)) if p not in bad_set], dtype=np.int64)
    fixed_points = vertices[selected[fixed]] if len(fixed) else None
    pool = np.asarray(
        [idx for idx in range(len(vertices)) if idx not in selected_set], dtype=np.int64
    )
    local = _seeded_fps(
        vertices[pool], min(max(1, int(exact_candidate_limit)), len(pool)), fixed_points
    )
    shortlist = pool[local]
    candidate_rates = exact_visibility_rates(mesh, vertices[shortlist], num_views=num_views)
    meets = candidate_rates >= float(min_visibility_rate)
    eligible, eligible_rates = shortlist[meets], candidate_rates[meets]
    if len(eligible) == 0:
        warnings.warn(
            "no replacement candidate met the visibility threshold; low-visibility keypoints kept",
            RuntimeWarning,
            stacklevel=2,
        )
        return selected, [], before
    object_scale = max(float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))), EPS)
    swaps: list[AnchorReplacement] = []
    used: set[int] = set()
    for pos in bad_positions[np.argsort(before[bad_positions])]:
        others = vertices[np.delete(selected, int(pos))]
        offsets = vertices[eligible][:, None, :] - others[None, :, :]
        spread = np.clip(np.min(np.linalg.norm(offsets, axis=2), axis=1) / object_scale, 0.0, 1.0)
        combined = (
            float(saliency_weight) * saliency_norm[eligible]
            + float(visibility_weight) * eligible_rates
            + float(spread_weight) * spread
        )
        for idx, candidate in enumerate(eligible):
            if int(candidate) in selected_set or int(candidate) in used:
                combined[idx] = -np.inf
        best = int(np.argmax(combined))
        if not np.isfinite(combined[best]):
            break
        old_index, new_index = int(selected[pos]), int(eligible[best])
        selected_set.discard(old_index)
        selected_set.add(new_index)
        used.add(new_index)
        selected[pos] = new_index
        swaps.append(
            AnchorReplacement(
                position=int(pos),
                old_index=old_index,
                new_index=new_index,
                old_visibility_rate=float(before[pos]),
                new_visibility_rate=float(eligible_rates[best]),
            )
        )
    after = exact_visibility_rates(mesh, vertices[selected], num_views=num_views)
    return selected, swaps, after


def convex_hull_anchor_indices(
    mesh: SurfaceMesh,
    saliency_scores: np.ndarray | Sequence[float],
    count: int,
    *,
    min_visibility_rate: float = DEFAULT_MIN_VISIBILITY_RATE,
    num_views: int = DEFAULT_VISIBILITY_VIEWS,
    saliency_weight: float = 0.35,
    visibility_weight: float = 0.35,
    radial_weight: float = 0.30,
) -> np.ndarray:
    """Select visible convex-hull vertices to anchor a layout.

    Hull vertices are scored by saliency, exact visibility and distance from the centroid; the
    best-scoring pool is then spread with farthest-point sampling. When fewer than ``count`` hull
    vertices meet the visibility threshold, the most visible ones fill the gap and a warning is
    issued.

    Args:
        mesh: Indexed triangle mesh.
        saliency_scores: Per-vertex saliency.
        count: Number of anchors.
        min_visibility_rate: Visibility rate a hull vertex needs to be eligible.
        num_views: Number of camera directions.
        saliency_weight: Weight of saliency in the hull score.
        visibility_weight: Weight of visibility in the hull score.
        radial_weight: Weight of centroid distance in the hull score.

    Returns:
        Anchor vertex indices in pick order.
    """
    from scipy.spatial import ConvexHull

    vertices = mesh.vertices
    saliency = np.asarray(saliency_scores, dtype=np.float64)
    if count <= 0:
        raise ValueError("count must be positive")
    hull = ConvexHull(vertices, qhull_options="QJ")
    hull_indices = np.unique(hull.simplices.reshape(-1)).astype(np.int64)
    hull_visibility = exact_visibility_rates(mesh, vertices[hull_indices], num_views=num_views)
    radial = np.linalg.norm(vertices[hull_indices] - vertices.mean(axis=0), axis=1)
    score = (
        float(saliency_weight) * normalize_scores(saliency[hull_indices])
        + float(visibility_weight) * normalize_scores(hull_visibility)
        + float(radial_weight) * normalize_scores(radial)
    )
    eligible_mask = hull_visibility >= float(min_visibility_rate)
    if int(np.sum(eligible_mask)) < count:
        warnings.warn(
            f"only {int(np.sum(eligible_mask))} hull vertices met visibility threshold "
            f"{float(min_visibility_rate):.3f}; filling anchors from highest-visibility hull "
            "vertices",
            RuntimeWarning,
            stacklevel=2,
        )
        order = np.argsort(-hull_visibility, kind="stable")
        eligible_positions = order[: max(count, int(np.sum(eligible_mask)))]
    else:
        eligible_positions = np.flatnonzero(eligible_mask)
    pool_size = max(
        count,
        int(math.ceil(len(eligible_positions) * HULL_POOL_FRACTION)),
        count * HULL_POOL_MULTIPLIER,
    )
    pool_size = min(pool_size, len(eligible_positions))
    ranked = eligible_positions[np.argsort(-score[eligible_positions], kind="stable")]
    candidate_positions = ranked[:pool_size]
    candidate_indices = hull_indices[candidate_positions]
    local = _seeded_fps(vertices[candidate_indices], min(count, len(candidate_indices)), None)
    return candidate_indices[local].astype(np.int64)


def hull_anchored_saliency_fps_indices(
    mesh: SurfaceMesh,
    saliency_scores: np.ndarray | Sequence[float],
    count: int,
    anchor_indices: np.ndarray | Sequence[int],
    *,
    pool_fraction: float = FILL_POOL_FRACTION,
    min_pool_multiplier: int = FILL_POOL_MULTIPLIER,
) -> np.ndarray:
    """Fill a layout with saliency-pooled farthest-point samples after fixed anchors.

    Args:
        mesh: Indexed triangle mesh.
        saliency_scores: Per-vertex saliency.
        count: Layout size; clipped to the vertex count.
        anchor_indices: Fixed anchors that come first in the result.
        pool_fraction: Fraction of vertices (most salient first) that may be picked.
        min_pool_multiplier: The pool holds at least this many vertices per keypoint.

    Returns:
        ``(count,)`` indices: the anchors, then the filled points.
    """
    vertices = mesh.vertices
    saliency = np.asarray(saliency_scores, dtype=np.float64)
    if count <= 0:
        raise ValueError("count must be positive")
    count = min(int(count), len(vertices))
    seen: set[int] = set()
    anchors_list: list[int] = []
    for value in np.asarray(anchor_indices, dtype=np.int64).reshape(-1):
        idx = int(value)
        if 0 <= idx < len(vertices) and idx not in seen:
            seen.add(idx)
            anchors_list.append(idx)
    anchors = np.asarray(anchors_list, dtype=np.int64)
    if len(anchors) >= count:
        return anchors[:count]
    pool_size = max(
        count, int(math.ceil(len(vertices) * pool_fraction)), count * min_pool_multiplier
    )
    pool_size = min(pool_size, len(vertices))
    order = np.argsort(-saliency, kind="stable")
    pool = np.asarray(
        [int(idx) for idx in order if int(idx) not in seen][:pool_size], dtype=np.int64
    )
    local = _seeded_fps(vertices[pool], min(count - len(anchors), len(pool)), vertices[anchors])
    return np.concatenate([anchors, pool[local]]).astype(np.int64)


def select_keypoints(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_keypoints: int = DEFAULT_NUM_KEYPOINTS,
    num_hull_anchors: int = DEFAULT_NUM_HULL_ANCHORS,
    *,
    anchor_visibility_views: int = DEFAULT_VISIBILITY_VIEWS,
    repair_visibility_views: int = DEFAULT_VISIBILITY_VIEWS,
    min_visibility_rate: float = DEFAULT_MIN_VISIBILITY_RATE,
) -> KeypointLayout:
    """Select a hull-anchored, visibility-checked keypoint layout on mesh vertices.

    Args:
        vertices: ``(V, 3)`` mesh vertices in meters (see ``load_mesh``).
        faces: ``(F, 3)`` triangle vertex indices.
        num_keypoints: Layout size; clipped to the vertex count.
        num_hull_anchors: Visible convex-hull anchors that start the layout.
        anchor_visibility_views: Camera directions used to rate hull vertices.
        repair_visibility_views: Camera directions used to check and repair the final layout.
        min_visibility_rate: Visibility rate below which a hull vertex is not an anchor and a
            keypoint is replaced.

    Returns:
        The layout. ``replaced_anchors`` lists the anchors that repair had to swap out;
        replaced non-anchor keypoints are not reported.

    Raises:
        ValueError: ``num_keypoints`` or ``num_hull_anchors`` is not positive, or an array
            has the wrong shape.
    """
    if num_keypoints <= 0:
        raise ValueError("num_keypoints must be positive")
    if num_hull_anchors <= 0:
        raise ValueError("num_hull_anchors must be positive")
    points = as_points(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"expected faces with shape (F, 3), got {faces.shape}")
    mesh = SurfaceMesh(vertices=points, faces=faces)
    saliency = surface_saliency_scores(mesh)
    anchors = convex_hull_anchor_indices(
        mesh,
        saliency,
        num_hull_anchors,
        min_visibility_rate=min_visibility_rate,
        num_views=anchor_visibility_views,
    )
    initial = hull_anchored_saliency_fps_indices(mesh, saliency, num_keypoints, anchors)
    refined, swaps, after = visibility_refined_indices(
        mesh,
        initial,
        saliency,
        min_visibility_rate=min_visibility_rate,
        num_views=repair_visibility_views,
    )
    replaced = tuple(swap for swap in swaps if swap.position < len(anchors))
    return KeypointLayout(
        points=points[refined],
        indices=refined,
        initial_anchor_indices=initial[: len(anchors)],
        replaced_anchors=replaced,
        saliency=saliency[refined],
        visibility_rate=after,
        method=DEFAULT_METHOD,
    )


def coordinates_sha256(keypoints_3d: Sequence[Sequence[float]] | np.ndarray) -> str:
    """Hash keypoint coordinates independently of JSON formatting.

    Args:
        keypoints_3d: ``(N, 3)`` positions.

    Returns:
        Hex SHA-256 of the C-contiguous little-endian float64 array.
    """
    array = np.ascontiguousarray(np.asarray(keypoints_3d, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def layout_payload(
    layout: KeypointLayout,
    *,
    object_name: str,
    keypoint_prefix: str,
    source_units: str,
    object_frame: str,
    parameters: dict[str, float | int],
) -> dict:
    """Build the keypoints JSON payload for a layout.

    Args:
        layout: Selected layout.
        object_name: Object name recorded in the file.
        keypoint_prefix: Prefix of the keypoint names; it must match the render config's
            ``object.keypoint_prefix``.
        source_units: Unit of the mesh the layout was selected on (``units`` is always ``m``).
        object_frame: Frame of the mesh: ``cad`` or ``aabb_centered``.
        parameters: Selection parameters to record.

    Returns:
        The JSON-serializable payload. ``keypoints_3d`` is rounded to nine decimals and
        ``coordinates_sha256`` hashes exactly those rounded values.
    """
    points = np.round(layout.points, KEYPOINT_DECIMALS)
    count = len(layout.indices)
    return {
        "object": object_name,
        "method": layout.method,
        "num_keypoints": count,
        "keypoint_names": [f"{keypoint_prefix}{idx:03d}" for idx in range(count)],
        "indices": [int(idx) for idx in layout.indices],
        "keypoints_3d": points.tolist(),
        "units": "m",
        "source_units": source_units,
        "scale": SOURCE_UNIT_SCALES[source_units],
        "object_frame": object_frame,
        "coordinates_sha256": coordinates_sha256(points),
        "parameters": parameters,
        "initial_anchor_indices": [int(idx) for idx in layout.initial_anchor_indices],
        "replaced_anchors": [
            {
                "keypoint_position": swap.position,
                "old_index": swap.old_index,
                "new_index": swap.new_index,
                "old_visibility_rate": swap.old_visibility_rate,
                "new_visibility_rate": swap.new_visibility_rate,
            }
            for swap in layout.replaced_anchors
        ],
        "visibility_rates": [float(rate) for rate in layout.visibility_rate],
    }


def format_report(layout: KeypointLayout, mesh_path: Path, parameters: dict) -> str:
    """Render a Markdown summary of a layout.

    Args:
        layout: Selected layout.
        mesh_path: Mesh the layout was selected on.
        parameters: Selection parameters (as recorded in the JSON).

    Returns:
        The Markdown text.
    """
    rates = layout.visibility_rate
    lines = [
        "# Hull-Anchored Keypoints",
        "",
        f"Mesh: `{mesh_path.name}`",
        f"Keypoints: `{len(layout.indices)}`",
        f"Hull anchors: `{parameters['num_hull_anchors']}`",
        f"Minimum visibility rate: `{parameters['min_visibility_rate']:.3f}`",
        "",
        "| min visibility | mean visibility | p10 visibility | replaced anchors |",
        "| ---: | ---: | ---: | ---: |",
        f"| {rates.min():.3f} | {rates.mean():.3f} | {np.percentile(rates, 10):.3f} | "
        f"{len(layout.replaced_anchors)} |",
        "",
        "## Anchors",
        "",
        "| anchor | initial mesh index | final mesh index | visibility | saliency |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for position, initial in enumerate(layout.initial_anchor_indices):
        lines.append(
            f"| {position} | {int(initial)} | {int(layout.indices[position])} | "
            f"{rates[position]:.3f} | {layout.saliency[position]:.3f} |"
        )
    if layout.replaced_anchors:
        lines += ["", "Warning: one or more anchors were swapped during visibility repair."]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="python -m doper.keypoint_selection",
        description="Select hull-anchored 3D keypoints on a CAD mesh and write them as JSON.",
    )
    parser.add_argument("--mesh", required=True, help="STL, OBJ or PLY mesh")
    parser.add_argument(
        "--source-units",
        required=True,
        choices=sorted(SOURCE_UNIT_SCALES),
        help="length unit of the mesh coordinates; the output is in meters",
    )
    parser.add_argument("--out-json", required=True, help="keypoints JSON to write")
    parser.add_argument("--num-keypoints", type=int, default=DEFAULT_NUM_KEYPOINTS)
    parser.add_argument("--num-hull-anchors", type=int, default=DEFAULT_NUM_HULL_ANCHORS)
    parser.add_argument("--object-name", default=None, help="default: the mesh file stem")
    parser.add_argument(
        "--keypoint-prefix",
        default=DEFAULT_KEYPOINT_PREFIX,
        help="prefix of the keypoint names (suffix: three-digit index); it must match the render "
        "config's object.keypoint_prefix, e.g. KP_object_ for mesh name 'object'",
    )
    parser.add_argument(
        "--object-frame",
        choices=OBJECT_FRAMES,
        default="cad",
        help="frame of the mesh, recorded in the JSON: cad, or aabb_centered for the mesh "
        "centered by mesh_setup.py --center-geometry (default: %(default)s)",
    )
    parser.add_argument("--out-report-md", default=None, help="optional Markdown summary")
    parser.add_argument("--anchor-visibility-views", type=int, default=DEFAULT_VISIBILITY_VIEWS)
    parser.add_argument("--repair-visibility-views", type=int, default=DEFAULT_VISIBILITY_VIEWS)
    parser.add_argument("--min-visibility-rate", type=float, default=DEFAULT_MIN_VISIBILITY_RATE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Select keypoints on a mesh and write the keypoints JSON (and optional report).

    Args:
        argv: CLI argument strings excluding the program name, or None to read ``sys.argv[1:]``.

    Returns:
        Zero after the files were written.

    Raises:
        SystemExit: Argument parsing requests help or rejects invalid arguments.
        ValueError: The mesh cannot be read or the selection arguments are invalid.
    """
    args = build_parser().parse_args(argv)
    mesh_path = Path(args.mesh)
    vertices, faces = load_mesh(mesh_path, args.source_units)
    layout = select_keypoints(
        vertices,
        faces,
        args.num_keypoints,
        args.num_hull_anchors,
        anchor_visibility_views=args.anchor_visibility_views,
        repair_visibility_views=args.repair_visibility_views,
        min_visibility_rate=args.min_visibility_rate,
    )
    parameters = {
        "num_keypoints": args.num_keypoints,
        "num_hull_anchors": args.num_hull_anchors,
        "anchor_visibility_views": args.anchor_visibility_views,
        "repair_visibility_views": args.repair_visibility_views,
        "min_visibility_rate": args.min_visibility_rate,
    }
    payload = layout_payload(
        layout,
        object_name=args.object_name or mesh_path.stem,
        keypoint_prefix=args.keypoint_prefix,
        source_units=args.source_units,
        object_frame=args.object_frame,
        parameters=parameters,
    )
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote keypoints: {out_json}")
    if args.out_report_md:
        out_report = Path(args.out_report_md)
        out_report.parent.mkdir(parents=True, exist_ok=True)
        out_report.write_text(format_report(layout, mesh_path, parameters))
        print(f"Wrote report: {out_report}")
    rates = layout.visibility_rate
    print(
        f"visibility min/mean/p10: {rates.min():.3f}/{rates.mean():.3f}/"
        f"{np.percentile(rates, 10):.3f}"
    )
    if layout.replaced_anchors:
        print(f"Warning: {len(layout.replaced_anchors)} hull anchors were visibility-repaired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
