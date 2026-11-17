# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SPEED+ geometry: camera files, the official Tango points, the 64-point layout and poses."""

import json

import cv2
import numpy as np
import pytest
from scipy.io import savemat
from speedplus_helpers import DIST, OFFICIAL_POINTS, K

from doper.speedplus import geometry


def test_expand_64_keeps_official_and_is_deterministic():
    expanded = geometry.expand_tango_keypoints_64(OFFICIAL_POINTS)
    again = geometry.expand_tango_keypoints_64(OFFICIAL_POINTS.copy())
    assert expanded.shape == (64, 3) and expanded.dtype == np.float64
    np.testing.assert_array_equal(expanded[:11], OFFICIAL_POINTS)
    np.testing.assert_array_equal(expanded, again)
    assert np.unique(np.round(expanded, 10), axis=0).shape[0] == 64
    with pytest.raises(ValueError, match="11x3"):
        geometry.expand_tango_keypoints_64(OFFICIAL_POINTS[:10])


def test_pose_round_trip_scalar_first():
    quaternion = np.asarray([0.72, -0.21, 0.51, 0.42])
    quaternion /= np.linalg.norm(quaternion)
    translation = np.asarray([0.13, -0.08, 5.4])
    rvec, tvec = geometry.speedplus_pose_to_opencv(quaternion, translation)
    assert rvec.shape == (3, 1) and tvec.shape == (3, 1)
    # The OpenCV object-to-camera rotation is the transpose of the SPEED+ DCM.
    rotation = cv2.Rodrigues(rvec)[0]
    dcm = geometry.quaternion_to_dcm_scalar_first(quaternion)
    np.testing.assert_allclose(rotation, dcm.T, atol=1e-12)
    recovered_q, recovered_t = geometry.opencv_pose_to_speedplus(rvec, tvec)
    np.testing.assert_allclose(recovered_q, quaternion, atol=1e-12)
    np.testing.assert_array_equal(recovered_t, translation)
    flipped_q, _ = geometry.opencv_pose_to_speedplus(
        *geometry.speedplus_pose_to_opencv(-quaternion, translation)
    )
    assert flipped_q[0] > 0  # q and -q are one rotation; the scalar part comes back positive


def test_dcm_refuses_zero_quaternion():
    with pytest.raises(ValueError, match="nonzero"):
        geometry.quaternion_to_dcm_scalar_first([0.0, 0.0, 0.0, 0.0])


def test_load_camera_native_and_normalized(tmp_path):
    native = tmp_path / "camera.json"
    native.write_text(json.dumps({"Nu": 1920, "Nv": 1200, "cameraMatrix": K, "distCoeffs": DIST}))
    camera = geometry.load_camera(native)
    np.testing.assert_array_equal(camera.K, np.asarray(K))
    np.testing.assert_array_equal(camera.dist, np.asarray(DIST))
    assert (camera.width, camera.height) == (1920, 1200)
    normalized = tmp_path / "camera_intrinsics.json"
    normalized.write_text(json.dumps({"camera_matrix": K, "width": 640, "height": 400}))
    camera = geometry.load_camera(normalized)
    np.testing.assert_array_equal(camera.dist, np.zeros(5))
    assert (camera.width, camera.height) == (640, 400)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"Nu": 1920, "Nv": 1200}, "camera matrix"),
        ({"Nu": 1920, "Nv": 1200, "cameraMatrix": [[1.0, 0.0], [0.0, 1.0]]}, "3x3"),
        ({"Nu": 1920, "Nv": 1200, "cameraMatrix": K, "distCoeffs": [0.0] * 3}, "distortion"),
        ({"cameraMatrix": K}, "image size"),
    ],
)
def test_load_camera_refuses_bad_files(tmp_path, payload, message):
    path = tmp_path / "camera.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        geometry.load_camera(path)


def test_load_official_tango_keypoints(tmp_path):
    path = tmp_path / "tangoPoints.mat"
    savemat(path, {"tango3Dpoints": OFFICIAL_POINTS.T})
    np.testing.assert_array_equal(geometry.load_official_tango_keypoints(path), OFFICIAL_POINTS)
    savemat(path, {"other": OFFICIAL_POINTS.T})
    with pytest.raises(ValueError, match="tango3Dpoints"):
        geometry.load_official_tango_keypoints(path)
    bad = OFFICIAL_POINTS.copy()
    bad[0, 0] = np.nan
    savemat(path, {"tango3Dpoints": bad.T})
    with pytest.raises(ValueError, match="non-finite"):
        geometry.load_official_tango_keypoints(path)
