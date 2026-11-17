# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from doper.scene import geometry as G  # noqa: N812
from doper.scene import labels as coco
from rendering.config import (
    load_render_config,
    render_config_from_mapping,
    sample_object_material,
)

BASE_DIR = "/nonexistent-base"


def test_intrinsics_match_blender_formula():
    intr = G.intrinsics_from_lens(640, 640, 50.0, 36.0)
    assert intr.fx == pytest.approx(50.0 * 640 / 36.0)
    assert intr.fy == intr.fx
    assert (intr.cx, intr.cy) == (320.0, 320.0)
    lo, hi = G.auto_camera_distance(0.1, 50.0, 36.0, (0.3, 0.7))
    half_tan = math.tan(G.horizontal_fov(50.0, 36.0) / 2)
    assert lo == pytest.approx(0.05 / half_tan / 0.7)
    assert hi == pytest.approx(0.05 / half_tan / 0.3)


def test_look_at_camera_uses_opengl_convention():
    cam = G.look_at_camera((0.0, -1.0, 1.0), (0.0, 0.0, 0.0))
    rot = cam[:3, :3]
    np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-9)
    assert np.linalg.det(rot) == pytest.approx(1.0)
    forward = np.array([0.0, 1.0, -1.0]) / math.sqrt(2)
    np.testing.assert_allclose(-rot[:, 2], forward, atol=1e-9)  # camera -Z looks at target
    assert rot[2, 1] > 0  # camera +Y points up-ish
    assert rot[2, 0] == pytest.approx(0.0)  # camera +X is horizontal


def test_projection_pixel_conventions():
    intr = G.intrinsics_from_lens(640, 480, 50.0, 36.0)
    cam = G.look_at_camera((0.0, 0.0, 1.0), (0.0, 0.0, 0.0))  # top-down
    uv, depth = G.project_points(np.array([[0.0, 0.0, 0.0]]), cam, intr)
    np.testing.assert_allclose(uv[0], [320.0, 240.0])
    assert depth[0] == pytest.approx(1.0)
    # A point along the camera's +X goes right in the image; along +Y goes up (smaller v).
    right = cam[:3, 3] - cam[:3, 2] + 0.1 * cam[:3, 0]
    up = cam[:3, 3] - cam[:3, 2] + 0.1 * cam[:3, 1]
    uv, _ = G.project_points(np.stack([right, up]), cam, intr)
    assert uv[0, 0] > 320.0 and uv[0, 1] == pytest.approx(240.0)
    assert uv[1, 1] < 240.0 and uv[1, 0] == pytest.approx(320.0)


def test_projected_keypoints_recover_object_pose_with_opencv_pnp():
    cv2 = pytest.importorskip("cv2")
    rng = random.Random(3)
    intr = G.intrinsics_from_lens(640, 640, 50.0, 36.0)
    kp_local = np.array(
        [[x, y, z] for x in (-0.03, 0.03) for y in (-0.02, 0.02) for z in (-0.005, 0.005)]
    )
    rot, _ = G.sample_object_rotation(rng, {"x": 0, "y": 0, "z": 0}, 5.0)
    object_pose = G.rest_on_ground(rot, G.aabb_corners(kp_local), 0.01, (0.01, -0.02))
    center = G.transform_points(object_pose, G.aabb_corners(kp_local)).mean(axis=0)
    cam = G.place_camera(rng, center, G.random_camera_direction(rng), 0.3, 0.2)

    uv, depth = G.project_points(G.transform_points(object_pose, kp_local), cam, intr)
    assert (depth > 0).all()

    ok, rvec, tvec = cv2.solvePnP(kp_local, uv, intr.matrix, None, flags=cv2.SOLVEPNP_ITERATIVE)
    assert ok
    # OpenCV camera: x right, y down, z forward. Convert the OpenGL-style extrinsics.
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    expected = flip @ np.linalg.inv(cam) @ object_pose
    np.testing.assert_allclose(cv2.Rodrigues(rvec)[0], expected[:3, :3], atol=1e-5)
    np.testing.assert_allclose(tvec.ravel(), expected[:3, 3], atol=1e-5)


def test_keypoint_visibility_classes():
    intr = G.intrinsics_from_lens(64, 64, 50.0, 36.0)
    uv = np.array([[10.0, 10.0], [30.0, 30.0], [50.0, 50.0], [-5.0, 10.0], [10.0, 10.0]])
    depth = np.array([1.0, 1.0, 1.0, 1.0, -1.0])
    depth_img = np.full((64, 64), np.inf)
    depth_img[28:33, 28:33] = 0.5  # occluder in front of keypoint 1
    depth_img[48:53, 48:53] = 0.9995  # surface itself, within tolerance
    vis = G.keypoint_visibility(uv, depth, depth_img, intr, tol=0.002)
    assert vis.tolist() == [2, 1, 2, 0, 0]
    assert G.keypoint_visibility(uv, depth, None, intr, tol=0.002).tolist() == [2, 2, 2, 0, 0]


def test_axis_biased_rotation_aligns_axis_with_world_z():
    rng = random.Random(0)
    for axis in ("x", "y", "z"):
        idx = "xyz".index(axis)
        for _ in range(20):
            rot, biased = G.sample_object_rotation(rng, {axis: 1.0}, 5.0)
            assert biased
            np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-9)
            assert np.linalg.det(rot) == pytest.approx(1.0)
            world_dir = rot[:, idx]
            assert abs(world_dir[2]) >= math.cos(math.radians(5.0 * math.sqrt(2)) + 1e-9)
    rot, biased = G.sample_object_rotation(rng, {"x": 0, "y": 0, "z": 0}, 5.0)
    assert not biased


def test_rest_on_ground_puts_lowest_corner_at_clearance():
    rng = random.Random(1)
    corners = G.aabb_corners(np.array([[-0.05, -0.02, -0.01], [0.05, 0.02, 0.01]]))
    for _ in range(10):
        rot = G.random_rotation(rng)
        pose = G.rest_on_ground(rot, corners, 0.004, (0.0, 0.0))
        assert G.transform_points(pose, corners)[:, 2].min() == pytest.approx(0.004)


def test_stereo_offset_moves_along_camera_x():
    cam = G.look_at_camera((0.0, -1.0, 0.5), (0.0, 0.0, 0.0))
    left = G.stereo_offset(cam, -0.03)
    right = G.stereo_offset(cam, 0.03)
    np.testing.assert_allclose(right[:3, 3] - left[:3, 3], cam[:3, 0] * 0.06)
    np.testing.assert_allclose(left[:3, :3], cam[:3, :3])


def test_bbox_helpers():
    uv = np.array([[100.0, 200.0], [300.0, 400.0], [900.0, 900.0]])
    vis = np.array([2, 1, 0])
    assert coco.keypoints_bbox(uv, vis, 640, 640) == pytest.approx([90.0, 190.0, 220.0, 220.0])
    assert coco.keypoints_bbox(uv, np.zeros(3, int), 640, 640) == [0.0, 0.0, 0.0, 0.0]
    mask = np.zeros((10, 10), np.uint8)
    mask[2:5, 3:7] = 255
    assert coco.mask_bbox(mask) == [3.0, 2.0, 4.0, 3.0]


def test_render_config_defaults_and_isaac_section(tmp_path):
    path = tmp_path / "render.yaml"
    path.write_text("object:\n  num_keypoints: 17\n")
    cfg = load_render_config(path)
    assert cfg.resolution == 640 and cfg.num_keypoints == 17
    assert cfg.axis_bias == {"x": 0.0, "y": 0.0, "z": 0.2}
    assert cfg.fill_factor == (0.3, 0.7) and cfg.auto_distance
    assert cfg.render_mode == "PathTracing" and cfg.bbox_source == "keypoints"
    assert cfg.resolve("assets/x") == str(tmp_path / "assets" / "x")

    custom = render_config_from_mapping(
        {
            "camera": {"stereo": {"enabled": True, "baseline_m": 0.05}},
            "pose": {"axis_bias": {"x": 0.3, "z": 0.2}},
            "isaac": {
                "render_mode": "RayTracedLighting",
                "spp": 16,
                "bbox_source": "mask",
                "light_intensity_range": [100, 200],
            },
        },
        BASE_DIR,
    )
    assert custom.stereo_enabled and custom.images_per_sample == 2
    assert custom.axis_bias == {"x": 0.3, "y": 0.0, "z": 0.2}
    assert custom.render_mode == "RayTracedLighting" and custom.spp == 16
    assert custom.bbox_source == "mask" and custom.light_intensity == (100.0, 200.0)
    with pytest.raises(ValueError):
        render_config_from_mapping({"isaac": {"render_mode": "Nope"}}, BASE_DIR)


def test_object_material_override_samples_configured_palette():
    rng = random.Random(3)
    red_only = render_config_from_mapping(
        {"material": {"object_color_choices": [[255, 0, 0]]}}, BASE_DIR
    )
    assert red_only.object_material_mode == "override" and red_only.object_metallic == 0.0
    for _ in range(5):
        sample = sample_object_material(rng, red_only)
        assert sample["diffuse_color_constant"] == (1.0, 0.0, 0.0)
        assert sample["metallic_constant"] == 0.0

    palette = render_config_from_mapping(
        {
            "material": {
                "object_material_mode": "override",
                "object_color_choices": [[235, 64, 52], [0.1, 0.2, 0.3], "orange"],
                "object_metallic": 0.05,
                "object_roughness_min": 0.4,
                "object_roughness_max": 0.5,
            }
        },
        BASE_DIR,
    )
    seen = {sample_object_material(rng, palette)["diffuse_color_constant"] for _ in range(60)}
    assert seen == {(235 / 255, 64 / 255, 52 / 255), (0.1, 0.2, 0.3), (1.0, 0.34, 0.02)}
    sample = sample_object_material(rng, palette)
    assert (
        sample["metallic_constant"] == 0.05
        and 0.4 <= sample["reflection_roughness_constant"] <= 0.5
    )

    with pytest.raises(ValueError):
        render_config_from_mapping({"material": {"object_material_mode": "override"}}, BASE_DIR)
    with pytest.raises(ValueError):
        render_config_from_mapping({"material": {"object_color_choices": ["purple"]}}, BASE_DIR)


def test_object_material_preserve_emulates_aluminium():
    rng = random.Random(3)
    # object_metallic (as DOPEST's write_render_config emits for preserve) only applies to override
    cfg = render_config_from_mapping(
        {"material": {"base_color_min": 0.8, "base_color_max": 0.9, "object_metallic": 0.0}},
        BASE_DIR,
    )
    assert cfg.object_material_mode == "preserve" and cfg.object_color_choices == ()
    sample = sample_object_material(rng, cfg)
    assert all(0.78 <= c <= 0.92 for c in sample["diffuse_color_constant"])
    assert max(sample["diffuse_color_constant"]) - min(sample["diffuse_color_constant"]) <= 0.04
    assert sample["metallic_constant"] == 1.0


def test_quaternion_uses_scalar_first_and_normalizes():
    # A 120-degree turn about (1, 1, 1) cycles X -> Y -> Z.
    np.testing.assert_allclose(
        G.quat_to_matrix((2, 2, 2, 2)), [[0, 0, 1], [1, 0, 0], [0, 1, 0]], atol=1e-14
    )


def test_euler_xyz_applies_x_then_y_then_z():
    # X then Y quarter-turns send X -> -Z, Y -> X, Z -> -Y.
    np.testing.assert_allclose(
        G.euler_xyz_to_matrix(math.pi / 2, math.pi / 2, 0),
        [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],
        atol=1e-14,
    )


@pytest.mark.parametrize(
    "axis, expected, next_draw",
    [
        (
            "x",
            [
                [-0.06907493850901344, 0.3005704755187438, 0.9512549827025414],
                [-0.06099557743572781, -0.9530210118511617, 0.29669932676612226],
                [0.9957450438758199, -0.0375278592065192, 0.08416333750581381],
            ],
            0.5103450263087496,
        ),
    ],
)
def test_biased_rotation_preserves_seeded_pose_and_rng_stream(axis, expected, next_draw):
    # Captured before the SciPy refactor: changing RNG source/order would change
    # all later fields in an existing seed's scene, even for a valid rotation.
    rng = random.Random(29)
    actual = G.axis_facing_rotation(rng, axis, 7)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    assert rng.random() == next_draw


def test_uniform_rotation_is_reproducible_without_global_rng():
    first_rng = random.Random(29)
    expected = [G.random_rotation(first_rng) for _ in range(8)]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        random.seed(921)
        np.random.seed(819)
        replay_rng = random.Random(29)
        actual = [G.random_rotation(replay_rng) for _ in range(8)]
        np.testing.assert_array_equal(actual, expected)
        assert replay_rng.getstate() == first_rng.getstate()
        assert not np.array_equal(G.random_rotation(random.Random(30)), expected[0])
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def test_uniform_rotation_has_no_preferred_direction():
    # Fixed-seed distribution guard: rotating each basis axis must produce
    # zero mean and isotropic second moments, unlike uniform Euler angles.
    rng = random.Random(184)
    matrices = np.stack([G.random_rotation(rng) for _ in range(5000)])
    np.testing.assert_allclose(matrices.mean(axis=0), 0, atol=0.035)
    for axis in range(3):
        points = matrices[:, :, axis]
        np.testing.assert_allclose(points.T @ points / len(points), np.eye(3) / 3, atol=0.025)
