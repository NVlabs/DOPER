# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Similarity fits, triangulation, camera centers and rig conversions."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from doper.reconstruction.geometry import (
    T_SENSOR_RIG_FLU,
    CameraPose,
    Observation,
    Sim3,
    as_rigid,
    camera_centre,
    camera_from_rig,
    project_pinhole,
    rig_from_camera,
    rotation_geodesic_deg,
    triangulate_points,
    umeyama_sim3,
)

K = np.array([[1371.25, 0.0, 941.24], [0.0, 1371.25, 591.71], [0.0, 0.0, 1.0]])


def random_pose(rng: np.random.Generator) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = Rotation.random(random_state=rng).as_matrix()
    pose[:3, 3] = rng.normal(size=3)
    return pose


def look_at(centre: np.ndarray, target: np.ndarray) -> np.ndarray:
    """OpenCV ``cam_from_world`` for a camera at ``center`` looking at ``target``."""
    forward = (target - centre) / np.linalg.norm(target - centre)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = -rotation @ centre
    return pose


def project(camera: CameraPose, point: np.ndarray) -> tuple[float, float]:
    q = K @ (camera.cam_from_world[:3, :3] @ point + camera.cam_from_world[:3, 3])
    return float(q[0] / q[2]), float(q[1] / q[2])


def test_umeyama_recovers_known_sim3() -> None:
    rng = np.random.default_rng(3)
    rotation = Rotation.random(random_state=rng).as_matrix()
    translation = rng.normal(size=3)
    src = rng.normal(scale=0.05, size=(64, 3))
    dst = 1.085 * src @ rotation.T + translation
    fit = umeyama_sim3(src, dst)
    assert fit.scale == pytest.approx(1.085, abs=1e-9)
    np.testing.assert_allclose(fit.rotation, rotation, atol=1e-9)
    np.testing.assert_allclose(fit.translation, translation, atol=1e-9)
    np.testing.assert_allclose(fit.apply(src), dst, atol=1e-9)
    np.testing.assert_allclose(fit.inverse().apply(dst), src, atol=1e-9)
    matrix = fit.as_matrix()
    homogeneous = np.c_[src, np.ones(len(src))] @ matrix.T
    np.testing.assert_allclose(homogeneous[:, :3], dst, atol=1e-9)
    np.testing.assert_allclose(matrix[3], [0.0, 0.0, 0.0, 1.0])


def test_umeyama_reflection_case() -> None:
    rng = np.random.default_rng(4)
    src = rng.normal(size=(20, 3))
    mirrored = src * np.array([-1.0, 1.0, 1.0])
    fit = umeyama_sim3(src, mirrored)
    assert np.linalg.det(fit.rotation) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(fit.rotation @ fit.rotation.T, np.eye(3), atol=1e-12)
    # The plain Kabsch/Umeyama SVD without the sign correction yields a reflection here.
    a = src - src.mean(0)
    b = mirrored - mirrored.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    assert np.linalg.det(vt.T @ u.T) < 0.0


def test_umeyama_refuses_bad_input() -> None:
    with pytest.raises(ValueError, match="same shape"):
        umeyama_sim3(np.zeros((4, 3)), np.zeros((5, 3)))
    with pytest.raises(ValueError, match="at least 3"):
        umeyama_sim3(np.zeros((2, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="degenerate"):
        umeyama_sim3(np.ones((5, 3)), np.ones((5, 3)))


def test_sim3_refuses_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="scale"):
        Sim3(0.0, np.eye(3), np.zeros(3))
    with pytest.raises(ValueError, match="rotation"):
        Sim3(1.0, np.diag([1.0, 1.0, -1.0]), np.zeros(3))


def test_triangulate_recovers_points_with_noise() -> None:
    rng = np.random.default_rng(5)
    target = np.array([0.0, 0.0, 0.0])
    points = rng.uniform(-0.03, 0.03, size=(16, 3))
    angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    centres = np.stack([0.3 * np.cos(angles), 0.3 * np.sin(angles), np.full(6, 0.1)], axis=1)
    cameras = [CameraPose(look_at(c, target)) for c in centres]
    observations = []
    for j, point in enumerate(points):
        for i, camera in enumerate(cameras):
            uv = np.array(project(camera, point)) + rng.normal(scale=0.5, size=2)
            observations.append(Observation(point=j, camera=i, uv=(uv[0], uv[1])))
    actual = triangulate_points(observations, cameras, K)
    assert actual.shape == points.shape
    assert np.linalg.norm(actual - points, axis=1).max() < 1e-3
    noiseless = [
        Observation(o.point, o.camera, project(cameras[o.camera], points[o.point]))
        for o in observations
    ]
    exact = triangulate_points(noiseless, cameras, K, init=np.array([0.0, 0.0, 0.05]))
    np.testing.assert_allclose(exact, points, atol=1e-9)


def test_triangulate_refuses_underdetermined_points() -> None:
    cameras = [CameraPose(np.eye(4)), CameraPose(np.eye(4))]
    with pytest.raises(ValueError, match="point 1 has 1 observation"):
        triangulate_points(
            [
                Observation(0, 0, (1.0, 2.0)),
                Observation(0, 1, (1.0, 2.0)),
                Observation(1, 0, (1.0, 2.0)),
            ],
            cameras,
            K,
        )
    with pytest.raises(ValueError, match="camera 2"):
        triangulate_points([Observation(0, 2, (1.0, 2.0))], cameras, K)
    with pytest.raises(ValueError, match="init"):
        triangulate_points(
            [Observation(0, 0, (1.0, 2.0)), Observation(0, 1, (1.0, 2.0))],
            cameras,
            K,
            init=np.zeros((2, 3)),
        )


def test_camera_centre() -> None:
    rng = np.random.default_rng(6)
    centre = rng.normal(size=3)
    cam_from_world = look_at(centre, np.zeros(3))
    np.testing.assert_allclose(camera_centre(cam_from_world), centre, atol=1e-12)
    np.testing.assert_allclose(camera_centre(cam_from_world[:3]), centre, atol=1e-12)


def test_rig_round_trip() -> None:
    rng = np.random.default_rng(7)
    for _ in range(5):
        world_camera = random_pose(rng)
        sensor_rig = random_pose(rng)
        rig = rig_from_camera(world_camera, sensor_rig)
        np.testing.assert_allclose(camera_from_rig(rig, sensor_rig), world_camera, atol=1e-12)
        rig = rig_from_camera(world_camera, T_SENSOR_RIG_FLU)
        np.testing.assert_allclose(camera_from_rig(rig, T_SENSOR_RIG_FLU), world_camera, atol=1e-12)


def test_t_sensor_rig_axes() -> None:
    np.testing.assert_array_equal(T_SENSOR_RIG_FLU, [[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    # OpenCV forward (z), right (x) and down (y) become rig forward, -left and -up.
    np.testing.assert_array_equal(T_SENSOR_RIG_FLU @ [0, 0, 1], [1, 0, 0])
    np.testing.assert_array_equal(T_SENSOR_RIG_FLU @ [1, 0, 0], [0, -1, 0])
    np.testing.assert_array_equal(T_SENSOR_RIG_FLU @ [0, 1, 0], [0, 0, -1])
    assert np.linalg.det(T_SENSOR_RIG_FLU) == pytest.approx(1.0)
    # A camera at the world origin looking along world +x: its rig's forward axis is world +x.
    world_camera = np.linalg.inv(look_at(np.zeros(3), np.array([1.0, 0.0, 0.0])))
    rig = rig_from_camera(world_camera, T_SENSOR_RIG_FLU)
    np.testing.assert_allclose(rig[:3, :3], np.eye(3), atol=1e-12)


def test_rotation_geodesic_deg() -> None:
    rng = np.random.default_rng(8)
    base = Rotation.random(random_state=rng)
    turned = base * Rotation.from_rotvec([0.0, 0.0, np.deg2rad(37.0)])
    assert rotation_geodesic_deg(base.as_matrix(), turned.as_matrix()) == pytest.approx(37.0)
    assert rotation_geodesic_deg(base.as_matrix(), base.as_matrix()) == pytest.approx(0.0, abs=1e-5)
    flipped = Rotation.from_rotvec([np.pi, 0.0, 0.0]).as_matrix()
    assert rotation_geodesic_deg(np.eye(3), flipped) == pytest.approx(180.0)


def test_as_rigid_returns_a_read_only_copy_and_names_the_failure():
    source = np.eye(4)
    rigid = as_rigid(source, "pose")
    assert not rigid.flags.writeable and source.flags.writeable
    with pytest.raises(ValueError, match="^pose must be a finite 4x4 transform$"):
        as_rigid(np.eye(3), "pose")
    with pytest.raises(ValueError, match="^pose bad$"):
        as_rigid(np.diag([1.0, 1.0, -1.0, 1.0]), "pose", not_rigid="bad")
    loose_row = np.eye(4)
    loose_row[3, 0] = 1e-7
    assert as_rigid(loose_row, "pose") is not None
    with pytest.raises(ValueError, match="rigid"):
        as_rigid(loose_row, "pose", last_row_atol=1e-8)


def test_project_pinhole_takes_3x4_and_4x4_poses():
    points = np.array([[0.1, -0.2, 0.3], [0.0, 0.0, 0.0]])
    pose = np.eye(4)
    pose[:3, 3] = [0.0, 0.0, 2.0]
    expected = (points + pose[:3, 3]) @ K.T
    expected = expected[:, :2] / expected[:, 2:]
    np.testing.assert_array_equal(project_pinhole(points, pose, K), expected)
    np.testing.assert_array_equal(project_pinhole(points, pose[:3], K), expected)
