# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic servicer/target metadata in the layout ``ShirtEphemerisPoseRTSSmoother`` reads."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from doper.filtering.shirt_ephemeris import quaternion_to_rotation

CAMERA_STEP_S = 5.0
SAMPLE_COUNT = 12
ORBIT_RADIUS_M = 7.0e6
EARTH_MU_M3_S2 = 3.986004418e14
SERVICER_SPIN_RAD_S = 1.0e-3
CAMERA_MATRIX = np.array([[800.0, 0.0, 512.0], [0.0, 800.0, 512.0], [0.0, 0.0, 1.0]])
DISTORTION = np.zeros(5)


def _unit_quaternion(values: list[float]) -> list[float]:
    quaternion = np.asarray(values, dtype=np.float64)
    return (quaternion / np.linalg.norm(quaternion)).tolist()


def _target_attitude_quaternions() -> list[list[float]]:
    """Return one distinct unit quaternion per sample (servicer principal to target principal)."""
    rng = np.random.default_rng(3)
    return [_unit_quaternion(rng.normal(size=4).tolist()) for _ in range(SAMPLE_COUNT)]


def make_metadata() -> dict[str, Any]:
    """Return circular-orbit servicer states, a slowly spinning servicer and a rigid target."""
    mean_motion = float(np.sqrt(EARTH_MU_M3_S2 / ORBIT_RADIUS_M**3))
    times = np.arange(SAMPLE_COUNT) * CAMERA_STEP_S
    angle = mean_motion * times
    position = ORBIT_RADIUS_M * np.stack([np.cos(angle), np.sin(angle), 0.0 * angle], axis=1)
    velocity = (
        ORBIT_RADIUS_M
        * mean_motion
        * np.stack([-np.sin(angle), np.cos(angle), 0.0 * angle], axis=1)
    )
    spin = SERVICER_SPIN_RAD_S * times
    attitude = [[float(np.cos(a / 2)), 0.0, 0.0, float(np.sin(a / 2))] for a in spin]
    inertia = [[10.0, 0.2, 0.1], [0.2, 8.0, 0.3], [0.1, 0.3, 5.0]]
    return {
        "pSim": {"cam_step": CAMERA_STEP_S},
        "sAbsState": {
            "rv_eci2com_eci": np.concatenate([position, velocity], axis=1).tolist(),
            "q_eci2pri": attitude,
        },
        "pMdl": {
            "x1": {
                "r_pri2cam_pri": [0.1, 0.2, 0.3],
                "q_pri2cam": _unit_quaternion([0.9, 0.1, -0.2, 0.3]),
            },
            "x2": {
                "r_pri2bdy_pri": [0.05, -0.02, 0.01],
                "q_pri2bdy": _unit_quaternion([0.8, -0.3, 0.2, 0.1]),
                "Ipri": inertia,
            },
        },
        "tRelState": {"q_spri2tpri": _target_attitude_quaternions()},
    }


def make_truth_pose(metadata: dict[str, Any], index: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the camera-frame body pose implied by ``q_spri2tpri`` and a fixed relative offset.

    The chain is built forward from the metadata (servicer attitude, camera mounting, target
    principal attitude, body offset), independently of the smoother's inverse conversion.
    """
    servicer = metadata["sAbsState"]
    rotation_eci_servicer = quaternion_to_rotation(np.asarray(servicer["q_eci2pri"][index]))
    mounting = metadata["pMdl"]["x1"]
    rotation_servicer_camera = quaternion_to_rotation(np.asarray(mounting["q_pri2cam"]))
    rotation_eci_camera = rotation_eci_servicer @ rotation_servicer_camera
    camera_offset = rotation_eci_servicer @ np.asarray(mounting["r_pri2cam_pri"])
    target = metadata["pMdl"]["x2"]
    rotation_servicer_target = quaternion_to_rotation(
        np.asarray(metadata["tRelState"]["q_spri2tpri"][index])
    )
    rotation_eci_principal = rotation_eci_servicer @ rotation_servicer_target
    rotation_principal_body = quaternion_to_rotation(np.asarray(target["q_pri2bdy"]))
    relative_eci = rotation_eci_camera @ np.array([0.2, -0.1, 7.0]) + camera_offset
    body_offset = rotation_eci_principal @ np.asarray(target["r_pri2bdy_pri"])
    body_camera = rotation_eci_camera.T @ (relative_eci + body_offset - camera_offset)
    rotation_camera_body = rotation_eci_camera.T @ rotation_eci_principal @ rotation_principal_body
    rvec, _ = cv2.Rodrigues(rotation_camera_body)
    return rvec, body_camera


def make_keypoints_3d() -> np.ndarray:
    """Return eight body-frame landmarks around the origin, in meters."""
    rng = np.random.default_rng(5)
    return rng.uniform(-0.4, 0.4, size=(8, 3))
