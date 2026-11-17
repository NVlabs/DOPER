# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry tests for the camera-ephemeris-aware SHIRT smoother, on a synthetic trajectory."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from synthetic_shirt import (
    CAMERA_MATRIX,
    DISTORTION,
    make_keypoints_3d,
    make_metadata,
    make_truth_pose,
)

from doper.filtering.pose_basis import rotation_from_basis
from doper.filtering.shirt_ephemeris import (
    ShirtEphemerisPoseRTSSmoother,
    quaternion_to_rotation,
)


def _initialized_smoother() -> ShirtEphemerisPoseRTSSmoother:
    metadata = make_metadata()
    smoother = ShirtEphemerisPoseRTSSmoother(metadata)
    rvec, tvec = make_truth_pose(metadata, 0)
    smoother.update(rvec, tvec, 0)
    return smoother


def test_quaternion_to_rotation_is_orthonormal_and_normalizes_input() -> None:
    rotation = quaternion_to_rotation(np.array([2.0, 0.4, -0.6, 0.8]))
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1.0e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    np.testing.assert_allclose(quaternion_to_rotation(np.array([1.0, 0.0, 0.0, 0.0])), np.eye(3))


def test_pose_round_trip_and_metadata_geometry() -> None:
    metadata = make_metadata()
    smoother = ShirtEphemerisPoseRTSSmoother(metadata)
    for index in (0, 3, 7, 11):
        rvec, tvec = make_truth_pose(metadata, index)
        raw_rotation, _ = cv2.Rodrigues(rvec)
        measurement = smoother.pose_to_measurement(rvec, tvec, index)
        state = np.zeros(16)
        state[0] = index * smoother.cam_step_s
        state[1:7] = measurement[0:6]
        state[7:10] = measurement[6:9]
        recovered_rvec, recovered_tvec = smoother.state_to_camera_pose(state, index)
        recovered_rotation, _ = cv2.Rodrigues(recovered_rvec)
        np.testing.assert_allclose(recovered_rotation, raw_rotation, atol=1.0e-8)
        np.testing.assert_allclose(recovered_tvec.reshape(3), tvec.reshape(3), atol=1.0e-8)
        expected_rotation = smoother._rotation_eci_from_servicer[index] @ quaternion_to_rotation(
            np.asarray(metadata["tRelState"]["q_spri2tpri"][index])
        )
        measured_rotation = rotation_from_basis(measurement[0:3], measurement[3:6])
        np.testing.assert_allclose(measured_rotation, expected_rotation, atol=1.0e-8)


def test_prediction_only_advances_without_reusing_a_measurement() -> None:
    smoother = _initialized_smoother()
    preview = smoother.preview_prediction(1)
    predicted = smoother.predict_only(1)
    np.testing.assert_allclose(predicted[0], preview[0])
    np.testing.assert_allclose(predicted[1], preview[1])
    assert smoother._times_s == [0.0, smoother.cam_step_s]
    smoothed = smoother.apply_rts(num_passes=1)
    assert len(smoothed) == 2
    with pytest.raises(RuntimeError, match="iterative RTS replay"):
        smoother.apply_rts()


def test_pose_candidate_nis_uses_predicted_pose_and_covariance() -> None:
    smoother = _initialized_smoother()
    predicted_pose = smoother.preview_prediction(1)
    displaced_tvec = predicted_pose[1] + np.asarray([[1.0], [0.0], [0.0]])
    statistics = smoother.preview_pose_innovations(
        [predicted_pose, (predicted_pose[0], displaced_tvec)], 1
    )
    assert statistics[0]["nis"] < 1.0e-10
    assert statistics[1]["nis"] > statistics[0]["nis"]
    np.testing.assert_allclose(
        statistics[0]["innovation_covariance"], statistics[1]["innovation_covariance"]
    )
    covariance = statistics[0]["innovation_covariance"]
    np.testing.assert_allclose(covariance, covariance.T, atol=1.0e-12)
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)


def test_keypoint_update_gates_landmarks_independently() -> None:
    points_3d = make_keypoints_3d()
    smoother = _initialized_smoother()
    predicted = smoother._predicted_state(1)
    observations = smoother._keypoint_observation(
        predicted.mean(), 1, points_3d, CAMERA_MATRIX, DISTORTION
    ).reshape(-1, 2)
    observations[3] += 1000.0
    covariances = np.repeat(np.eye(2)[None] * 4.0, len(points_3d), axis=0)
    _, diagnostics = smoother.update_keypoints(
        observations, covariances, points_3d, CAMERA_MATRIX, DISTORTION, 1
    )
    assert diagnostics["updated"]
    assert int(np.sum(diagnostics["accepted"])) == len(points_3d) - 1
    assert not diagnostics["accepted"][3]
    assert diagnostics["nis"][3] > 9.21034037197618

    rejected_smoother = _initialized_smoother()
    predicted_pose = rejected_smoother.preview_prediction(1)
    _, rejected = rejected_smoother.update_keypoints(
        observations + 2000.0, covariances, points_3d, CAMERA_MATRIX, DISTORTION, 1
    )
    assert not rejected["updated"]
    assert not rejected["accepted"].any()
    committed_pose = rejected_smoother.state_to_camera_pose(
        rejected_smoother._filter.state.mean(), 1
    )
    np.testing.assert_allclose(committed_pose[0], predicted_pose[0])
    np.testing.assert_allclose(committed_pose[1], predicted_pose[1])
