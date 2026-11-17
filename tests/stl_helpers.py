# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Binary STL writer and a test bracket mesh shared by the unit and rendering tests."""

import struct
from pathlib import Path

import numpy as np

STL_HEADER_BYTES = 80
GRID = 6
SIZE_MM = (60.0, 40.0, 20.0)


def write_binary_stl(path: Path, triangles: np.ndarray, header: bytes = b"") -> Path:
    """Write ``(T, 3, 3)`` triangles as a float32 binary STL with unit face normals.

    Args:
        path: Output file.
        triangles: Triangle corner positions.
        header: Text for the 80 byte header; padded with zeros.

    Returns:
        ``path``.
    """
    corners = np.asarray(triangles, dtype=np.float32)
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
    record = np.zeros(len(corners), dtype=[("data", "<f4", (12,)), ("attr", "<u2")])
    record["data"] = np.concatenate([normals[:, None], corners], axis=1).reshape(-1, 12)
    with open(path, "wb") as stream:
        stream.write(header.ljust(STL_HEADER_BYTES, b"\0"))
        stream.write(struct.pack("<I", len(corners)))
        stream.write(record.tobytes())
    return path


def bracket_triangles_mm() -> np.ndarray:
    """Return a tapered and sheared box (60 x 40 x 20 mm before distortion) as triangles."""
    steps = np.linspace(0.0, 1.0, GRID + 1)
    size = np.array(SIZE_MM)
    triangles = []
    for axis in range(3):
        u, v = [(axis + 1) % 3, (axis + 2) % 3]
        for level in (0.0, 1.0):
            for i in range(GRID):
                for j in range(GRID):
                    corners = []
                    for a, b in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
                        point = np.zeros(3)
                        point[axis], point[u], point[v] = level, steps[a], steps[b]
                        corners.append(point * size)
                    triangles += [[corners[0], corners[1], corners[2]]]
                    triangles += [[corners[0], corners[2], corners[3]]]
    tris = np.array(triangles)
    tris[..., 0] *= 1.0 - 0.2 * tris[..., 1] / SIZE_MM[1]
    tris[..., 0] += 0.15 * tris[..., 2]
    # A smooth position-dependent warp breaks the distance ties a regular grid would have.
    warp = np.stack(
        [np.sin(1.7 * tris[..., 1] + 0.9 * tris[..., 2]),
         np.sin(2.3 * tris[..., 0] + 1.1 * tris[..., 2]),
         np.sin(1.3 * tris[..., 0] + 2.9 * tris[..., 1])], axis=-1)  # fmt: skip
    tris += 0.7 * warp
    return tris
