# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the constant-velocity SHIRT pose RTS smoother."""

from __future__ import annotations

import cv2
import numpy as np

from doper.filtering.shirt_pose import ShirtPoseRTSSmoother


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    first_rotation, _ = cv2.Rodrigues(first.reshape(3, 1))
    second_rotation, _ = cv2.Rodrigues(second.reshape(3, 1))
    relative = first_rotation @ second_rotation.T
    return float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))))


def test_constant_velocity_rts_tracks_rotation_and_translation() -> None:
    smoother = ShirtPoseRTSSmoother()
    expected = []
    for index in range(30):
        time_s = index * 5.0
        rvec = np.array([0.0, 0.0, 0.015 * time_s], dtype=np.float64)
        tvec = np.array([0.002 * time_s, -0.001 * time_s, 7.0], dtype=np.float64)
        expected.append((rvec, tvec))
        smoother.update(rvec, tvec, time_s)

    poses = smoother.apply_rts()
    assert len(poses) == len(expected)
    assert (
        max(
            _rotation_error_deg(actual[0], target[0])
            for actual, target in zip(poses[3:], expected[3:], strict=True)
        )
        < 0.2
    )
    assert (
        max(
            float(np.linalg.norm(actual[1].reshape(3) - target[1]))
            for actual, target in zip(poses[3:], expected[3:], strict=True)
        )
        < 0.01
    )


def test_rts_rejects_isolated_180_degree_measurement_flip() -> None:
    smoother = ShirtPoseRTSSmoother(rot_measurement_std=0.05)
    truth = []
    for index in range(41):
        time_s = index * 5.0
        rvec = np.array([0.0, 0.0, 0.012 * time_s], dtype=np.float64)
        truth.append(rvec)
        measured = rvec.copy()
        if index == 20:
            rotation, _ = cv2.Rodrigues(measured.reshape(3, 1))
            flipped, _ = cv2.Rodrigues(np.array([np.pi, 0.0, 0.0], dtype=np.float64).reshape(3, 1))
            measured, _ = cv2.Rodrigues(flipped @ rotation)
            measured = measured.reshape(3)
        smoother.update(measured, np.array([0.0, 0.0, 7.0]), time_s)

    poses = smoother.apply_rts()
    assert _rotation_error_deg(poses[20][0], truth[20]) < 10.0


def test_repeated_unscented_rts_passes_return_one_pose_per_measurement() -> None:
    smoother = ShirtPoseRTSSmoother()
    for index in range(20):
        time_s = index * 5.0
        smoother.update(
            np.array([0.0, 0.01 * time_s, 0.0]),
            np.array([0.001 * time_s, 0.0, 7.0]),
            time_s,
        )
    poses = smoother.apply_rts(num_passes=3)
    assert len(poses) == 20
    assert len(smoother.last_forward_state_means) == 20
    assert len(smoother.last_smoothed_state_means) == 20
    assert all(np.isfinite(np.concatenate(pose)).all() for pose in poses)
    assert len(smoother._measurements) == 20
    for index, (rvec, tvec) in enumerate(smoother._measurements):
        time_s = index * 5.0
        np.testing.assert_allclose(rvec.reshape(3), [0.0, 0.01 * time_s, 0.0])
        np.testing.assert_allclose(tvec.reshape(3), [0.001 * time_s, 0.0, 7.0])
