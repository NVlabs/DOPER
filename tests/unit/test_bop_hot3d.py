# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from pathlib import Path

import numpy as np
import pytest

from doper._vendor.hot3d_camera import camera as vendored_camera
from doper.bop.hot3d import (
    camera_from_object_mm,
    camera_pair,
    clean_calibration,
    decode_rle,
    mask_bbox,
    metadata_m_to_mm,
    project_native,
    project_rectified,
    rectification_maps,
    rectify,
)

VENDOR_DIR = Path(__file__).resolve().parents[2] / "doper" / "_vendor" / "hot3d_camera"


def calibration(params=None):
    return {
        "image_width": 20,
        "image_height": 16,
        "projection_model_type": "CameraModelType.FISHEYE624",
        "projection_params": params if params is not None else [10, 9.2, 7.3] + [0] * 12,
        "label": "fixture",
        "serial_number": "fixture",
    }


def distorted_calibration():
    params = [10, 9.2, 7.3, 0.01, -0.02, 0.003, 0.001, 0.0005, -0.0004, 0.0002, 0.0001]
    return calibration(params + [0.001, -0.001, 0.0002, 0.0001])


def encode_rle(mask):
    """Official paired codec: one-based absolute start and run length."""
    pixels = np.concatenate([[0], mask.flatten(), [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return runs.tolist()


def test_metadata_lengths_and_symmetry_translations_only():
    transform_matrix = np.eye(4)
    transform_matrix[:3, 3] = [0.1, -0.2, 0.3]
    src = {
        "1": {
            "name": "test",
            "diameter": 0.4,
            "min_x": -0.2,
            "size_x": 0.3,
            "symmetries_discrete": [transform_matrix.ravel().tolist()],
            "symmetries_continuous": [{"axis": [0, 0, 1], "offset": [0.02, 0.03, 0]}],
        }
    }
    before = copy.deepcopy(src)
    result = metadata_m_to_mm(src)["1"]
    assert src == before and result["diameter"] == 400 and result["size_x"] == 300
    out = np.asarray(result["symmetries_discrete"][0]).reshape(4, 4)
    np.testing.assert_allclose(out[:3, :3], np.eye(3))
    np.testing.assert_allclose(out[:3, 3], [100, -200, 300])
    assert result["symmetries_continuous"] == [{"axis": [0, 0, 1], "offset": [20, 30, 0]}]


def test_invalid_discrete_symmetry_is_rejected():
    bad = np.eye(4)
    bad[3, 3] = 2
    with pytest.raises(ValueError):
        metadata_m_to_mm({"1": {"symmetries_discrete": [bad.ravel().tolist()]}})


def test_rle_matches_official_paired_codec_not_coco():
    mask = np.zeros((5, 8), dtype=np.uint8)
    mask[1, 2:4] = 1
    mask[3, 1:5] = 1
    runs = encode_rle(mask)
    assert runs == [11, 2, 26, 4]
    decoded = decode_rle({"height": 5, "width": 8, "rle": runs})
    np.testing.assert_array_equal(decoded > 0, mask > 0)
    assert not decode_rle({"height": 5, "width": 8, "rle": []}).any()
    with pytest.raises(ValueError):
        decode_rle({"height": 5, "width": 8, "rle": [0, 3]})
    with pytest.raises(ValueError):
        decode_rle({"height": 5, "width": 8, "rle": [39, 4]})


def test_rle_round_trip():
    rng = np.random.default_rng(3)
    for _ in range(5):
        mask = (rng.random((7, 11)) > 0.6).astype(np.uint8)
        decoded = decode_rle({"height": 7, "width": 11, "rle": encode_rle(mask)})
        assert decoded.dtype == np.uint8 and set(np.unique(decoded)) <= {0, 255}
        np.testing.assert_array_equal(decoded > 0, mask > 0)


def test_rle_rejects_malformed_input():
    with pytest.raises(ValueError):
        decode_rle({"height": 0, "width": 8, "rle": []})
    with pytest.raises(ValueError):
        decode_rle({"height": 5, "width": 8, "rle": [1, 2, 3]})
    with pytest.raises(ValueError):
        decode_rle({"height": 5, "width": 8, "rle": [1.0, 2.0]})


def test_exact_principal_point_same_eye_axes_and_no_extrinsics_in_input():
    raw = {
        "calibration": dict(calibration(), T_device_from_camera={"secret_pose": 1}),
        "T_world_from_camera": {"quaternion_wxyz": [0, 1, 0, 0], "translation_xyz": [9, 8, 7]},
    }
    cleaned = clean_calibration(raw)
    assert "T_device_from_camera" not in cleaned and "T_world_from_camera" not in cleaned
    native, pinhole, intrinsics = camera_pair(raw)
    assert intrinsics[0, 2] == 9.2 and intrinsics[1, 2] == 7.3  # not rounded to width/2, height/2
    np.testing.assert_array_equal(native.T_world_from_eye, np.eye(4))
    np.testing.assert_array_equal(pinhole.T_world_from_eye, np.eye(4))
    np.testing.assert_allclose(native.eye_to_window(np.array([[0, 0, 1.0]])), [[9.2, 7.3]])


def test_calibration_validation():
    with pytest.raises(ValueError):
        clean_calibration(dict(calibration(), projection_model_type="CameraModelType.LINEAR"))
    with pytest.raises(ValueError):
        clean_calibration(calibration([0, 9.2, 7.3] + [0] * 12))
    with pytest.raises(ValueError):
        clean_calibration(calibration([10, 9.2]))


def test_map_samples_follow_official_native_projection():
    raw = calibration()
    x, y, _ = rectification_maps(raw)
    native, pinhole, _ = camera_pair(raw)
    pixels = np.array([[0, 0], [7, 5], [19, 15]], dtype=float)
    expected = native.eye_to_window(pinhole.window_to_eye(pixels))
    got = np.array([[x[int(v), int(u)], y[int(v), int(u)]] for u, v in pixels])
    np.testing.assert_allclose(got, expected, atol=1e-6)
    assert not x.flags.writeable and not y.flags.writeable
    image = np.arange(320, dtype=np.uint8).reshape(16, 20)
    converted, _ = rectify(image, raw)
    assert converted.shape == (16, 20, 3)
    mask, _ = rectify((image > 100).astype(np.uint8) * 255, raw, mask=True)
    assert set(np.unique(mask)) <= {0, 255}
    with pytest.raises(ValueError):
        rectify(np.zeros((20, 20), dtype=np.uint8), raw)


def test_mask_rectification_is_nearest_neighbor():
    raw = distorted_calibration()
    mask = np.zeros((16, 20), dtype=np.uint8)
    mask[4:12, 5:15] = 255
    mask[8, 10] = 200  # a bilinear blend would create an intermediate value near this edge
    rectified, _ = rectify(mask, raw, mask=True)
    x, y, _ = rectification_maps(raw)
    nearest = np.rint(np.stack([y, x])).astype(int)  # cv2 rounds half to even; maps avoid ties
    inside = (x >= 0) & (x <= 19) & (y >= 0) & (y <= 15)
    expected = np.zeros_like(mask)
    sampled = mask[nearest[0].clip(0, 15), nearest[1].clip(0, 19)]
    expected[inside] = (sampled[inside] > 127).astype(np.uint8) * 255
    np.testing.assert_array_equal(rectified, expected)
    assert rectified.dtype == np.uint8 and set(np.unique(rectified)) <= {0, 255}
    assert rectified.ndim == 2


def test_pose_composition_m_to_mm_and_projection_coordinates():
    cam = {"T_world_from_camera": {"quaternion_wxyz": [1, 0, 0, 0], "translation_xyz": [4, 2, 1]}}
    obj = {"T_world_from_object": {"quaternion_wxyz": [1, 0, 0, 0], "translation_xyz": [4.1, 2, 2]}}
    rotation, t = camera_from_object_mm(cam, obj)
    np.testing.assert_allclose(rotation, np.eye(3))
    np.testing.assert_allclose(t, [100, 0, 1000])
    native, pinhole, intrinsics = camera_pair(calibration())
    points = np.array([[0, 0, 0], [0, 20, 0]], dtype=float)
    uv, z = project_rectified(points, rotation, t, intrinsics)
    np.testing.assert_allclose(uv, [[10.2, 7.3], [10.2, 7.5]])
    np.testing.assert_allclose(z, [1000, 1000])
    expected = native.eye_to_window((points + t) / 1000)
    np.testing.assert_allclose(project_native(points, rotation, t, calibration()), expected)


def test_transform_rejects_invalid_se3():
    from doper.bop.hot3d import transform

    with pytest.raises(ValueError):
        transform({"quaternion_wxyz": [0, 0, 0, 0], "translation_xyz": [0, 0, 0]})
    with pytest.raises(ValueError):
        transform({"quaternion_wxyz": [1, 0, 0, 0], "translation_xyz": [0, np.nan, 0]})


def test_native_projection_matches_source():
    """Native projection equals the vendored fisheye model evaluated directly (source contract)."""
    raw = distorted_calibration()
    rng = np.random.default_rng(0)
    points = rng.uniform(-60, 60, size=(12, 3))
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t = np.array([5.0, -3.0, 400.0])
    model = vendored_camera.from_json(
        {
            "calibration": clean_calibration(raw),
            "T_world_from_camera": {"quaternion_wxyz": [1, 0, 0, 0], "translation_xyz": [0, 0, 0]},
        }
    )
    expected = model.eye_to_window((points @ rotation.T + t) / 1000)
    actual = project_native(points, rotation, t, raw)
    np.testing.assert_array_equal(actual, expected)
    assert np.abs(actual - project_rectified(points, rotation, t, camera_pair(raw)[2])[0]).max() > 0


def test_mask_bbox():
    mask = np.zeros((6, 6), dtype=np.uint8)
    assert mask_bbox(mask) is None
    mask[2:4, 1:5] = 255
    assert mask_bbox(mask) == [1, 2, 4, 2]


def test_vendored_package_keeps_license_and_notice():
    notice = (VENDOR_DIR / "NOTICE").read_text()
    assert "bc628e9286c18444b47a0971a89d075ea6f00747" in notice and "Apache-2.0" in notice
    assert "Apache License" in (VENDOR_DIR / "LICENSE").read_text()
    for name in ("camera.py", "camera_distortion.py", "math_utils.py"):
        assert "Licensed under the Apache License, Version 2.0" in (VENDOR_DIR / name).read_text()
