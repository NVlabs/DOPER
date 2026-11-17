# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Crop-rotation geometry tests.

Tests of the SPEED+ fine-tuning script datasets wait for that
script's port.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from tiny_coco import coco_annotation, tiny_model_config, write_coco_split

from doper import targets
from doper.augment import clahe_luminance
from doper.config import AugConfig, DataConfig, DoperConfig
from doper.dataset import DoperDataset, normalize_image

# Standard ImageNet statistics used by ``doper.dataset.normalize_image``.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _quaternion_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Independent scalar-first Hamilton quaternion to active rotation."""
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=np.float64)
    w, x, y, z = np.asarray([w, x, y, z]) / np.linalg.norm([w, x, y, z])
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _project(
    points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    camera: np.ndarray,
) -> np.ndarray:
    camera_points = points @ rotation.T + translation
    pixels_h = camera_points @ camera.T
    return pixels_h[:, :2] / pixels_h[:, 2:3]


@pytest.mark.parametrize("rotation_deg", [31.25, 90.0, -90.0, 121.25, 270.0])
def test_crop_rotation_matches_camera_roll_projection(rotation_deg: float) -> None:
    """Warped 2D projections equal projections from the rolled pose target."""
    points = np.asarray(
        [
            [-0.4, -0.2, 0.1],
            [0.3, -0.25, -0.1],
            [0.45, 0.35, 0.2],
            [-0.25, 0.4, -0.15],
        ],
        dtype=np.float64,
    )
    rotation, _ = cv2.Rodrigues(np.asarray([0.31, -0.22, 0.47]))
    translation = np.asarray([0.18, -0.12, 4.5], dtype=np.float64)
    focal = 1180.0
    camera = np.asarray(
        [[focal, 0.0, 621.3], [0.0, focal, 503.7], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    crop_center = (603.2, 488.4)
    source_size = 517.0
    output_size = 384
    warp = targets.topdown_warp_matrix(crop_center, source_size, output_size, rotation_deg)
    warped_pixels = targets.warp_points(_project(points, rotation, translation, camera), warp)

    camera_roll = targets.crop_camera_roll_matrix(rotation_deg)
    augmented_rotation = camera_roll @ rotation
    augmented_translation = camera_roll @ translation
    scale = output_size / source_size
    warped_principal = targets.warp_points(camera[:2, 2][None], warp)[0]
    crop_camera = np.asarray(
        [
            [scale * focal, 0.0, warped_principal[0]],
            [0.0, scale * focal, warped_principal[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    pose_target_pixels = _project(points, augmented_rotation, augmented_translation, crop_camera)

    np.testing.assert_allclose(warped_pixels, pose_target_pixels, atol=8e-5)
    np.testing.assert_allclose(camera_roll.T @ camera_roll, np.eye(3), atol=1e-12)
    assert np.linalg.det(camera_roll) == pytest.approx(1.0)


@pytest.mark.parametrize("rotation_deg", [31.25, 90.0, 121.25, 180.0, 270.0])
def test_crop_quaternion_update_matches_crop_camera_roll(rotation_deg: float) -> None:
    quaternion = np.asarray([0.74, -0.21, 0.37, 0.52], dtype=np.float64)

    augmented = targets.rotate_object_to_camera_quaternion_for_crop(quaternion, rotation_deg)

    expected = targets.crop_camera_roll_matrix(rotation_deg) @ _quaternion_matrix(quaternion)
    np.testing.assert_allclose(_quaternion_matrix(augmented), expected, atol=1e-12)
    assert np.linalg.norm(augmented) == pytest.approx(1.0)


def test_positive_crop_rotation_moves_image_right_toward_crop_top() -> None:
    size = 160
    center = (size / 2.0, size / 2.0)
    image = np.zeros((size, size), dtype=np.uint8)
    image[77:84, 107:114] = 255
    point = np.asarray([[110.0, 80.0]], dtype=np.float32)

    warp = targets.topdown_warp_matrix(center, size, size, 90.0)
    rotated = cv2.warpAffine(image, warp, (size, size), flags=cv2.INTER_LINEAR)
    expected = targets.warp_points(point, warp)[0]

    # In y-down image coordinates, counter-clockwise means right -> up.
    np.testing.assert_allclose(expected, [80.0, 50.0], atol=1e-5)
    assert rotated[47:54, 77:84].max() == 255
    assert rotated[107:114, 77:84].max() == 0


@pytest.mark.parametrize("quarter_turns", [1, 2, 3])
def test_quarter_turn_is_composed_with_continuous_roll_in_one_affine(
    quarter_turns: int,
) -> None:
    output_size = 384
    continuous_deg = -27.4
    base = targets.topdown_warp_matrix((311.7, 246.3), 428.2, output_size, continuous_deg)
    combined = targets.topdown_warp_matrix(
        (311.7, 246.3),
        428.2,
        output_size,
        continuous_deg + 90.0 * quarter_turns,
    )
    output_turn = targets.topdown_warp_matrix(
        (output_size / 2.0, output_size / 2.0),
        output_size,
        output_size,
        90.0 * quarter_turns,
    )
    base_h = np.vstack([base, [0.0, 0.0, 1.0]])
    output_turn_h = np.vstack([output_turn, [0.0, 0.0, 1.0]])

    # getAffineTransform solves from float32 control points, so composed and
    # directly solved matrices differ by roughly 1e-4 at their translations.
    np.testing.assert_allclose(combined, (output_turn_h @ base_h)[:2], atol=1e-4)


def _write_marker_dataset(root) -> None:
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.circle(image, (80, 64), 4, (255, 255, 255), thickness=-1)
    write_coco_split(
        root / "train",
        [{"id": 1, "file_name": "marker.png", "width": 128, "height": 128}],
        [coco_annotation(1, 1, [32.0, 32.0, 64.0, 64.0], 4096.0, [80.0, 64.0, 2])],
        [{"id": 1, "name": "marker"}],
        {"images/marker.png": image},
    )


def test_dataset_combined_crop_rotation_keeps_pixels_keypoints_and_angle_aligned(
    tmp_path,
) -> None:
    _write_marker_dataset(tmp_path)
    aug = AugConfig(
        enabled=True,
        photo_prob=0.0,
        grayscale_prob=0.0,
        blur_prob=0.0,
        motion_blur_prob=0.0,
        noise_prob=0.0,
        jpeg_prob=0.0,
        crop_scale_prob=0.0,
        crop_shift_prob=0.0,
        crop_rotate_prob=1.0,
        crop_rotate=30.0,
        crop_rot90_prob=1.0,
        det_rot90_prob=0.0,
        det_scale_prob=0.0,
        det_translate_prob=0.0,
        det_erase_prob=0.0,
    )
    cfg = DoperConfig(
        data=DataConfig(root=str(tmp_path)),
        model=tiny_model_config(
            num_keypoints=1,
            det_input_size=128,
            crop_size=128,
            crop_pad_factor=1.0,
            keypoint_head="heatmap",
        ),
        aug=aug,
    )

    sample = DoperDataset(cfg, "train")[0]
    rotation_deg = float(sample["crop_rotation_deg"])
    nearest_quarter_turn = round(rotation_deg / 90.0)
    assert nearest_quarter_turn in (1, 2, 3)
    assert abs(rotation_deg - 90.0 * nearest_quarter_turn) <= 30.0

    warp = targets.topdown_warp_matrix((64.0, 64.0), 64.0, 128, rotation_deg)
    expected = targets.warp_points(np.asarray([[80.0, 64.0]]), warp)[0]

    crop = sample["crop_image"].numpy().transpose(1, 2, 0)
    crop = (crop * IMAGENET_STD + IMAGENET_MEAN).mean(axis=2)
    bright_y, bright_x = np.nonzero(crop > 0.9)
    pixel_centroid = np.asarray([bright_x.mean(), bright_y.mean()])
    assert np.linalg.norm(pixel_centroid - expected) <= 0.75

    heatmap = sample["keypoint_heatmap"][0].numpy()
    heatmap_y, heatmap_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    decoded_peak = np.asarray([heatmap_x, heatmap_y]) * (127.0 / 31.0)
    assert np.linalg.norm(decoded_peak - expected) <= 3.0


def test_clahe_luminance_preserves_shape_and_changes_low_contrast_image() -> None:
    x = np.linspace(112, 140, 96, dtype=np.uint8)
    image = np.repeat(x[None, :, None], 80, axis=0)
    image = np.repeat(image, 3, axis=2)

    result = clahe_luminance(image, clip_limit=2.0, grid_size=8)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.mean(np.abs(result.astype(np.int16) - image.astype(np.int16))) > 0.5


def test_local_imagenet_constants_match_dataset_normalization() -> None:
    image = np.array([[[0, 128, 255]]], dtype=np.uint8)

    normalized = normalize_image(image)[:, 0, 0]

    expected = (image[0, 0].astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    np.testing.assert_array_equal(normalized, expected)
