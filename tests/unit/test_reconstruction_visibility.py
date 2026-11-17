# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ray visibility of surface points against a triangle mesh."""

import numpy as np
import pytest

from doper.reconstruction.visibility import mesh_edges, ray_mesh_visible, ray_mesh_visible_edges

# A unit cube [0, 1]^3 with outward-wound triangles.
CUBE_VERTICES = np.array(
    [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)], dtype=float
)
CUBE_FACES = np.array(
    [
        [0, 2, 3],
        [0, 3, 1],  # x = 0
        [4, 5, 7],
        [4, 7, 6],  # x = 1
        [0, 1, 5],
        [0, 5, 4],  # y = 0
        [2, 6, 7],
        [2, 7, 3],  # y = 1
        [0, 4, 6],
        [0, 6, 2],  # z = 0
        [1, 3, 7],
        [1, 7, 5],  # z = 1
    ]
)


def test_ray_visible_cube() -> None:
    camera = np.array([5.0, 0.5, 5.0])
    points = np.array(
        [
            [1.0, 1.0, 1.0],  # near-face corner
            [0.0, 0.0, 0.0],  # far-face corner, behind the cube
            [1.0, 0.5, 0.0],  # silhouette edge (bottom of the x = 1 face)
            [0.0, 0.5, 1.0],  # silhouette edge (far edge of the top face)
            [0.5, 0.5, 1.0],  # center of the top face
            [0.5, 0.5, 0.0],  # center of the bottom face, hidden
            [1.0, 0.5, 0.5],  # center of the x = 1 face
            [0.0, 0.5, 0.5],  # center of the x = 0 face, hidden
        ]
    )
    actual = ray_mesh_visible(points, camera, CUBE_VERTICES, CUBE_FACES, tolerance_m=1e-6)
    expected = [True, False, True, True, True, False, True, False]
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == bool


def test_ray_visible_tolerance() -> None:
    camera = np.array([0.5, 0.5, 5.0])
    # Just below the top face: hidden by 1 mm of material unless the tolerance exceeds it.
    point = np.array([[0.5, 0.5, 0.999]])
    assert not ray_mesh_visible(point, camera, CUBE_VERTICES, CUBE_FACES, tolerance_m=1e-4)[0]
    assert ray_mesh_visible(point, camera, CUBE_VERTICES, CUBE_FACES, tolerance_m=2e-3)[0]


def test_ray_visible_ignores_geometry_behind_the_camera() -> None:
    camera = np.array([0.5, 0.5, -1.0])
    points = np.array([[0.5, 0.5, -3.0]])
    assert ray_mesh_visible(points, camera, CUBE_VERTICES, CUBE_FACES, tolerance_m=1e-6)[0]


def test_ray_visible_shared_diagonal_is_not_a_gap() -> None:
    # The ray to the bottom-face center crosses the top face exactly on its shared diagonal.
    camera = np.array([0.5, 0.5, 5.0])
    hidden = np.array([[0.5, 0.5, 0.0], [0.25, 0.25, 0.0], [0.75, 0.75, 0.0]])
    actual = ray_mesh_visible(hidden, camera, CUBE_VERTICES, CUBE_FACES, tolerance_m=1e-6)
    assert not actual.any()


def test_ray_visible_chunks_agree() -> None:
    rng = np.random.default_rng(9)
    camera = np.array([2.0, -1.5, 3.0])
    points = rng.uniform(0.0, 1.0, size=(300, 3))
    points[:, 0] = rng.integers(0, 2, size=300).astype(float)  # onto the x = 0 or x = 1 face
    full = ray_mesh_visible(points, camera, CUBE_VERTICES, CUBE_FACES, tolerance_m=1e-6)
    small = ray_mesh_visible(points, camera, CUBE_VERTICES, CUBE_FACES, tolerance_m=1e-6, chunk=7)
    np.testing.assert_array_equal(full, small)
    edges = mesh_edges(CUBE_VERTICES, CUBE_FACES)
    reused = ray_mesh_visible_edges(points, camera, edges, tolerance_m=1e-6, chunk=7)
    np.testing.assert_array_equal(full, reused)
    # Points on x = 1 face the camera (x = 2); points on x = 0 are behind the cube.
    on_near = points[:, 0] == 1.0
    assert full[on_near].all()
    assert not full[~on_near].any()


def test_ray_visible_refuses_bad_input() -> None:
    with pytest.raises(ValueError, match="points"):
        ray_mesh_visible(np.zeros(3), np.zeros(3), CUBE_VERTICES, CUBE_FACES, tolerance_m=0.0)
    with pytest.raises(ValueError, match="faces"):
        ray_mesh_visible(
            np.zeros((1, 3)), np.ones(3), CUBE_VERTICES, CUBE_FACES + 8, tolerance_m=0.0
        )
    with pytest.raises(ValueError, match="tolerance"):
        ray_mesh_visible(np.zeros((1, 3)), np.ones(3), CUBE_VERTICES, CUBE_FACES, tolerance_m=-1.0)
    with pytest.raises(ValueError, match="faces"):
        mesh_edges(CUBE_VERTICES, CUBE_FACES + 8)
    with pytest.raises(ValueError, match="chunk"):
        ray_mesh_visible_edges(
            np.zeros((1, 3)),
            np.ones(3),
            mesh_edges(CUBE_VERTICES, CUBE_FACES),
            tolerance_m=0.0,
            chunk=0,
        )
