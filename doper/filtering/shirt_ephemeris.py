# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Camera-ephemeris-aware UKF/RTS pose tracking for the SHIRT benchmark.

Only the released servicer state and fixed camera mounting are treated as known.
The target principal-frame pose, inertial relative velocity, and body angular
velocity are inferred from keypoint PnP measurements. Translation is propagated
with differential two-body gravity about the released servicer ephemeris and
attitude with torque-free rigid-body dynamics before conversion back into the
moving camera frame.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from doper.filtering._bayesfilter import (
    RTS,
    BayesianFilter,
    Gaussian,
    Observation,
    StateTransitionModel,
    propagate_gaussian_cross_cov,
)
from doper.filtering.pose_basis import (
    orthonormalize_state,
    rotation_from_basis,
    rotation_to_rvec,
    rvec_to_rotation,
)


def quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    """Return the active rotation used by the scalar-first SHIRT quaternions."""
    q0, q1, q2, q3 = np.asarray(quaternion, dtype=np.float64)
    q0, q1, q2, q3 = np.asarray([q0, q1, q2, q3]) / np.linalg.norm(quaternion)
    passive = np.asarray(
        [
            [
                2 * q0**2 - 1 + 2 * q1**2,
                2 * q1 * q2 + 2 * q0 * q3,
                2 * q1 * q3 - 2 * q0 * q2,
            ],
            [
                2 * q1 * q2 - 2 * q0 * q3,
                2 * q0**2 - 1 + 2 * q2**2,
                2 * q2 * q3 + 2 * q0 * q1,
            ],
            [
                2 * q1 * q3 + 2 * q0 * q2,
                2 * q2 * q3 - 2 * q0 * q1,
                2 * q0**2 - 1 + 2 * q3**2,
            ],
        ],
        dtype=np.float64,
    )
    return passive.T


class ShirtEphemerisPoseRTSSmoother:
    """Unscented camera-aware SHIRT tracker with iterative RTS initialization."""

    _STATE_SIZE = 16
    _MU_EARTH_M3_S2 = 3.986004418e14

    def __init__(
        self,
        metadata: dict | Path,
        *,
        rot_measurement_std: float = 0.05,
        trans_measurement_std_m: float = 0.10,
        angular_accel_std_rad_s2: float = 1.0e-5,
        linear_accel_std_m_s2: float = 1.0e-5,
        initial_angular_velocity_std_rad_s: float = 0.03,
        initial_relative_velocity_std_m_s: float = 0.02,
    ) -> None:
        """Create the tracker from SHIRT trajectory metadata.

        Args:
            metadata: Parsed ``metadata.json`` dict, or the path to that file. Only the
                servicer ephemeris and attitude, camera mounting and target mass model are read.
            rot_measurement_std: Measurement std per rotation-basis component.
            trans_measurement_std_m: Measurement std per translation component, meters.
            angular_accel_std_rad_s2: Angular-acceleration process noise std.
            linear_accel_std_m_s2: Linear-acceleration process noise std.
            initial_angular_velocity_std_rad_s: Prior std of the initial angular velocity.
            initial_relative_velocity_std_m_s: Prior std of the initial relative velocity.
        """
        if isinstance(metadata, Path):
            parsed: dict = json.loads(metadata.read_text())
        else:
            parsed = metadata
        self.metadata = parsed
        self.cam_step_s = float(parsed["pSim"]["cam_step"])
        servicer_state = np.asarray(parsed["sAbsState"]["rv_eci2com_eci"], dtype=np.float64)
        self._sample_times_s = np.arange(len(servicer_state)) * self.cam_step_s
        self._servicer_position_eci = servicer_state[:, 0:3]
        self._servicer_velocity_eci = servicer_state[:, 3:6]
        self._rotation_eci_from_servicer = np.asarray(
            [quaternion_to_rotation(q) for q in parsed["sAbsState"]["q_eci2pri"]]
        )

        servicer_model = parsed["pMdl"]["x1"]
        target_model = parsed["pMdl"]["x2"]
        self._camera_offset_servicer = np.asarray(servicer_model["r_pri2cam_pri"], dtype=np.float64)
        self._rotation_servicer_from_camera = quaternion_to_rotation(
            np.asarray(servicer_model["q_pri2cam"], dtype=np.float64)
        )
        self._target_body_offset_principal = np.asarray(
            target_model["r_pri2bdy_pri"], dtype=np.float64
        )
        self._rotation_principal_from_body = quaternion_to_rotation(
            np.asarray(target_model["q_pri2bdy"], dtype=np.float64)
        )
        self._target_inertia = np.asarray(target_model["Ipri"], dtype=np.float64)
        self._target_inertia_inv = np.linalg.inv(self._target_inertia)

        self._rot_measurement_std = float(rot_measurement_std)
        self._trans_measurement_std_m = float(trans_measurement_std_m)
        self._angular_accel_std = float(angular_accel_std_rad_s2)
        self._linear_accel_std = float(linear_accel_std_m_s2)
        self._initial_angular_velocity_std = float(initial_angular_velocity_std_rad_s)
        self._initial_relative_velocity_std = float(initial_relative_velocity_std_m_s)
        self._measurement_covariance = np.diag(
            [self._rot_measurement_std**2] * 6 + [self._trans_measurement_std_m**2] * 3
        )
        self._transition_model = StateTransitionModel(
            self._transition,
            np.eye(self._STATE_SIZE, dtype=np.float64) * 1.0e-12,
        )
        self._filter: BayesianFilter | None = None
        self._states: list[Gaussian] = []
        self._times_s: list[float] = []
        self._measurements: list[tuple[np.ndarray, np.ndarray, int]] = []
        self._last_forward_states: list[Gaussian] = []
        self._last_smoothed_states: list[Gaussian] = []
        self._contains_prediction_only_steps = False

    @classmethod
    def _orthonormalize(cls, state: np.ndarray) -> np.ndarray:
        return orthonormalize_state(state, offset=1)

    @staticmethod
    def _positive_definite(covariance: np.ndarray) -> np.ndarray:
        """Remove round-off negative modes introduced by constrained SO(3)."""
        symmetric = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
        eigenvalues = np.maximum(eigenvalues, scale * 1.0e-12)
        return (eigenvectors * eigenvalues) @ eigenvectors.T

    def _ephemeris(self, time_s: float) -> tuple[np.ndarray, np.ndarray]:
        position = np.asarray(
            [
                np.interp(time_s, self._sample_times_s, self._servicer_position_eci[:, i])
                for i in range(3)
            ]
        )
        velocity = np.asarray(
            [
                np.interp(time_s, self._sample_times_s, self._servicer_velocity_eci[:, i])
                for i in range(3)
            ]
        )
        return position, velocity

    @classmethod
    def _gravity(cls, position_eci: np.ndarray) -> np.ndarray:
        return -cls._MU_EARTH_M3_S2 * position_eci / np.linalg.norm(position_eci) ** 3

    def _relative_orbital_derivative(self, state: np.ndarray, time_s: float) -> np.ndarray:
        relative_position = state[0:3]
        relative_velocity = state[3:6]
        servicer_position, _ = self._ephemeris(time_s)
        # Using differential gravity cancels the common-mode perturbations in
        # SHIRT's high-fidelity servicer ephemeris.  Propagating Tango with a
        # standalone two-body model and then subtracting that ephemeris would
        # spuriously interpret the omitted perturbations as relative motion.
        relative_acceleration = self._gravity(
            servicer_position + relative_position
        ) - self._gravity(servicer_position)
        return np.concatenate((relative_velocity, relative_acceleration))

    def _propagate_relative_orbit(
        self,
        relative_position: np.ndarray,
        relative_velocity: np.ndarray,
        time_s: float,
        delta_t_s: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        value = np.concatenate((relative_position, relative_velocity))
        k1 = self._relative_orbital_derivative(value, time_s)
        k2 = self._relative_orbital_derivative(
            value + 0.5 * delta_t_s * k1, time_s + 0.5 * delta_t_s
        )
        k3 = self._relative_orbital_derivative(
            value + 0.5 * delta_t_s * k2, time_s + 0.5 * delta_t_s
        )
        k4 = self._relative_orbital_derivative(value + delta_t_s * k3, time_s + delta_t_s)
        propagated = value + delta_t_s * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        return propagated[0:3], propagated[3:6]

    def _angular_derivative(self, angular_velocity: np.ndarray) -> np.ndarray:
        momentum = self._target_inertia @ angular_velocity
        return -self._target_inertia_inv @ np.cross(angular_velocity, momentum)

    def _propagate_attitude(
        self, rotation: np.ndarray, angular_velocity: np.ndarray, delta_t_s: float
    ) -> tuple[np.ndarray, np.ndarray]:
        k1 = self._angular_derivative(angular_velocity)
        k2 = self._angular_derivative(angular_velocity + 0.5 * delta_t_s * k1)
        k3 = self._angular_derivative(angular_velocity + 0.5 * delta_t_s * k2)
        k4 = self._angular_derivative(angular_velocity + delta_t_s * k3)
        next_velocity = angular_velocity + delta_t_s * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        average_velocity = 0.5 * (angular_velocity + next_velocity)
        increment = rvec_to_rotation(average_velocity * delta_t_s)
        return rotation @ increment, next_velocity

    def _transition(self, state: np.ndarray, delta_t_s: float) -> np.ndarray:
        state = self._orthonormalize(state)
        time_s = float(state[0])
        relative_position, relative_velocity = self._propagate_relative_orbit(
            state[7:10], state[13:16], time_s, delta_t_s
        )
        rotation = rotation_from_basis(state[1:4], state[4:7])
        rotation, angular_velocity = self._propagate_attitude(rotation, state[10:13], delta_t_s)
        out = state.copy()
        out[0] = time_s + delta_t_s
        out[1:4] = rotation[:, 0]
        out[4:7] = rotation[:, 1]
        out[7:10] = relative_position
        out[10:13] = angular_velocity
        out[13:16] = relative_velocity
        return out

    @staticmethod
    def _observation(state: np.ndarray) -> np.ndarray:
        return np.concatenate((state[1:7], state[7:10]))

    def _process_noise(self, delta_t_s: float) -> np.ndarray:
        dt = max(float(delta_t_s), 1.0e-6)
        qa = self._angular_accel_std**2
        ql = self._linear_accel_std**2
        covariance = np.eye(self._STATE_SIZE) * 1.0e-12
        covariance[0, 0] = 1.0e-16
        covariance[1:7, 1:7] += np.eye(6) * qa * dt**3 / 3.0
        covariance[10:13, 10:13] += np.eye(3) * qa * dt
        covariance[7:10, 7:10] += np.eye(3) * ql * dt**3 / 3.0
        covariance[13:16, 13:16] += np.eye(3) * ql * dt
        return covariance

    def _camera_transform(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        rotation_eci_from_servicer = self._rotation_eci_from_servicer[index]
        rotation_eci_from_camera = rotation_eci_from_servicer @ self._rotation_servicer_from_camera
        camera_position_eci = (
            self._servicer_position_eci[index]
            + rotation_eci_from_servicer @ self._camera_offset_servicer
        )
        return rotation_eci_from_camera, camera_position_eci

    def pose_to_measurement(self, rvec: np.ndarray, tvec: np.ndarray, index: int) -> np.ndarray:
        """Convert a camera-frame body pose to the ECI principal-frame measurement vector."""
        rotation_camera_from_body = rvec_to_rotation(rvec)
        rotation_camera_from_principal = (
            rotation_camera_from_body @ self._rotation_principal_from_body.T
        )
        target_com_camera = (
            np.asarray(tvec, dtype=np.float64).reshape(3)
            - rotation_camera_from_principal @ self._target_body_offset_principal
        )
        rotation_eci_from_camera, camera_position_eci = self._camera_transform(index)
        rotation_eci_from_principal = rotation_eci_from_camera @ rotation_camera_from_principal
        target_com_eci = camera_position_eci + rotation_eci_from_camera @ target_com_camera
        relative_position_eci = target_com_eci - self._servicer_position_eci[index]
        return np.concatenate(
            (
                rotation_eci_from_principal[:, 0],
                rotation_eci_from_principal[:, 1],
                relative_position_eci,
            )
        )

    def state_to_camera_pose(self, state: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Convert a filter state to the camera-frame body pose ``(rvec, tvec)``."""
        state = self._orthonormalize(state)
        rotation_eci_from_principal = rotation_from_basis(state[1:4], state[4:7])
        rotation_eci_from_body = rotation_eci_from_principal @ self._rotation_principal_from_body
        target_com_eci = self._servicer_position_eci[index] + state[7:10]
        target_body_eci = (
            target_com_eci + rotation_eci_from_principal @ self._target_body_offset_principal
        )
        rotation_eci_from_camera, camera_position_eci = self._camera_transform(index)
        rotation_camera_from_body = rotation_eci_from_camera.T @ rotation_eci_from_body
        target_body_camera = rotation_eci_from_camera.T @ (target_body_eci - camera_position_eci)
        return rotation_to_rvec(rotation_camera_from_body), target_body_camera.reshape(3, 1)

    def _initial_state(self, measurement: np.ndarray, time_s: float) -> Gaussian:
        mean = np.zeros(self._STATE_SIZE)
        mean[0] = time_s
        mean[1:7] = measurement[0:6]
        mean[7:10] = measurement[6:9]
        covariance = np.eye(self._STATE_SIZE) * 1.0e-12
        covariance[1:10, 1:10] = self._measurement_covariance
        covariance[10:13, 10:13] = np.eye(3) * self._initial_angular_velocity_std**2
        covariance[13:16, 13:16] = np.eye(3) * self._initial_relative_velocity_std**2
        return Gaussian(self._orthonormalize(mean), covariance)

    def update(
        self, rvec: np.ndarray, tvec: np.ndarray, index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Initialize from, or fuse, one pose measurement and return the forward pose.

        Args:
            rvec: Rodrigues vector of the measured camera-frame body rotation.
            tvec: Measured camera-frame body translation, meters.
            index: Camera sample index; time is ``index * cam_step_s``.

        Returns:
            The forward-filtered camera-frame ``(rvec, tvec)``.
        """
        time_s = index * self.cam_step_s
        measurement = self.pose_to_measurement(rvec, tvec, index)
        if self._filter is None:
            state = self._initial_state(measurement, time_s)
            self._filter = BayesianFilter(self._transition_model, state)
        else:
            predicted = self._predicted_state(index)
            observation = Observation(
                measurement,
                self._measurement_covariance,
                self._observation,
            )
            updated = self._filter.update(observation, predicted, use_jacobian=False)
            mean = self._orthonormalize(updated.mean())
            covariance = self._positive_definite(updated.covariance())
            self._filter.set_state(Gaussian(mean, covariance))
        self._states.append(
            Gaussian(
                self._filter.state.mean().copy(),
                self._filter.state.covariance().copy(),
            )
        )
        self._times_s.append(time_s)
        self._measurements.append(
            (
                np.asarray(rvec).reshape(3, 1).copy(),
                np.asarray(tvec).reshape(3, 1).copy(),
                index,
            )
        )
        return self.state_to_camera_pose(self._filter.state.mean(), index)

    def _predicted_state(self, index: int) -> Gaussian:
        if self._filter is None or not self._times_s:
            raise RuntimeError("the filter must be initialized before prediction")
        time_s = index * self.cam_step_s
        delta_t_s = time_s - self._times_s[-1]
        if delta_t_s <= 0.0:
            raise ValueError("prediction index must advance time")
        self._transition_model.transition_noise_covariance = self._process_noise(delta_t_s)
        predicted = self._filter.predict(self._filter.state, delta_t_s, use_jacobian=False)
        return Gaussian(
            self._orthonormalize(predicted.mean()),
            self._positive_definite(predicted.covariance()),
        )

    def preview_prediction(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the next prior pose without changing filter state."""
        predicted = self._predicted_state(index)
        return self.state_to_camera_pose(predicted.mean(), index)

    def _keypoint_observation(
        self,
        state: np.ndarray,
        index: int,
        points_3d: np.ndarray,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
    ) -> np.ndarray:
        rvec, tvec = self.state_to_camera_pose(state, index)
        projected, _ = cv2.projectPoints(
            np.asarray(points_3d, dtype=np.float64),
            rvec,
            tvec,
            np.asarray(camera_matrix, dtype=np.float64),
            np.asarray(distortion, dtype=np.float64),
        )
        return projected.reshape(-1).astype(np.float64)

    def update_keypoints(
        self,
        keypoints_image: np.ndarray,
        keypoint_covariances: np.ndarray,
        points_3d: np.ndarray,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
        index: int,
        *,
        valid: np.ndarray | None = None,
        nis_gate: float = 9.21034037197618,
        covariance_scale: float = 1.0,
    ) -> tuple[tuple[np.ndarray, np.ndarray], dict[str, Any]]:
        """Gate 2D landmarks independently and jointly update the UKF.

        The nonlinear keypoint observation is propagated through the predicted
        state exactly once.  Marginal 2D innovation covariances are used for
        independent chi-square gates, after which one joint update retains the
        state-induced cross-correlations between all accepted landmarks.
        """
        if self._filter is None:
            raise RuntimeError("initialize the filter from a pose before keypoint updates")
        observed = np.asarray(keypoints_image, dtype=np.float64)
        covariances = np.asarray(keypoint_covariances, dtype=np.float64)
        points = np.asarray(points_3d, dtype=np.float64)
        if observed.shape != (len(points), 2):
            raise ValueError(f"keypoints must have shape {(len(points), 2)}, got {observed.shape}")
        if covariances.shape != (len(points), 2, 2):
            raise ValueError(
                "keypoint covariances must have shape "
                f"{(len(points), 2, 2)}, got {covariances.shape}"
            )
        if covariance_scale <= 0.0 or nis_gate <= 0.0:
            raise ValueError("covariance scale and NIS gate must be positive")
        if valid is None:
            valid_mask = np.ones(len(points), dtype=bool)
        else:
            valid_mask = np.asarray(valid, dtype=bool)
            if valid_mask.shape != (len(points),):
                raise ValueError(f"valid must have shape {(len(points),)}")
        valid_mask &= np.isfinite(observed).all(axis=1)
        valid_mask &= np.isfinite(covariances).all(axis=(1, 2))

        predicted = self._predicted_state(index)

        def observation_function(state: np.ndarray) -> np.ndarray:
            return self._keypoint_observation(state, index, points, camera_matrix, distortion)

        predicted_mean, predicted_covariance, cross_covariance = propagate_gaussian_cross_cov(
            predicted.mean(), predicted.covariance(), observation_function
        )
        predicted_points = predicted_mean.reshape(-1, 2)
        innovation = observed - predicted_points
        scaled_covariances = covariances * float(covariance_scale)
        nis = np.full(len(points), np.inf, dtype=np.float64)
        accepted = np.zeros(len(points), dtype=bool)
        for keypoint_index in np.flatnonzero(valid_mask):
            sl = slice(2 * keypoint_index, 2 * keypoint_index + 2)
            innovation_covariance = (
                predicted_covariance[sl, sl] + scaled_covariances[keypoint_index]
            )
            innovation_covariance = self._positive_definite(innovation_covariance)
            try:
                whitened = np.linalg.solve(innovation_covariance, innovation[keypoint_index])
            except np.linalg.LinAlgError:
                continue
            nis[keypoint_index] = float(innovation[keypoint_index] @ whitened)
            accepted[keypoint_index] = nis[keypoint_index] <= nis_gate

        accepted_indices = np.flatnonzero(accepted)
        if not len(accepted_indices):
            self._commit_predicted_state(predicted, index)
            pose = self.state_to_camera_pose(predicted.mean(), index)
            return pose, {
                "accepted": accepted,
                "nis": nis,
                "predicted_keypoints": predicted_points,
                "innovation": innovation,
                "updated": False,
            }

        dimensions = np.ravel(np.column_stack((2 * accepted_indices, 2 * accepted_indices + 1)))
        measurement_covariance = np.zeros((len(dimensions), len(dimensions)), dtype=np.float64)
        for output_index, keypoint_index in enumerate(accepted_indices):
            sl = slice(2 * output_index, 2 * output_index + 2)
            measurement_covariance[sl, sl] = scaled_covariances[keypoint_index]
        joint_covariance = (
            predicted_covariance[np.ix_(dimensions, dimensions)] + measurement_covariance
        )
        joint_covariance = self._positive_definite(joint_covariance)
        selected_cross_covariance = cross_covariance[:, dimensions]
        try:
            kalman_gain = np.linalg.solve(joint_covariance.T, selected_cross_covariance.T).T
        except np.linalg.LinAlgError:
            kalman_gain = selected_cross_covariance @ np.linalg.pinv(joint_covariance)
        selected_residual = innovation[accepted].reshape(-1)
        updated_mean = self._orthonormalize(predicted.mean() + kalman_gain @ selected_residual)
        updated_covariance = self._positive_definite(
            predicted.covariance() - kalman_gain @ joint_covariance @ kalman_gain.T
        )
        updated = Gaussian(updated_mean, updated_covariance)
        self._filter.set_state(updated)
        self._states.append(Gaussian(updated_mean.copy(), updated_covariance.copy()))
        self._times_s.append(index * self.cam_step_s)
        pose = self.state_to_camera_pose(updated_mean, index)
        self._measurements.append((*pose, index))
        return pose, {
            "accepted": accepted,
            "nis": nis,
            "predicted_keypoints": predicted_points,
            "innovation": innovation,
            "updated": True,
        }

    @staticmethod
    def _pose_tangent_residual(
        reference_state: np.ndarray, candidate_measurement: np.ndarray
    ) -> np.ndarray:
        """Return local SO(3) and ECI-position residual for a pose measurement."""
        reference_rotation = rotation_from_basis(reference_state[1:4], reference_state[4:7])
        candidate_rotation = rotation_from_basis(
            candidate_measurement[0:3], candidate_measurement[3:6]
        )
        rotation_residual = rotation_to_rvec(reference_rotation.T @ candidate_rotation)
        return np.concatenate(
            (
                rotation_residual.reshape(3),
                candidate_measurement[6:9] - reference_state[7:10],
            )
        )

    @classmethod
    def _pose_tangent_jacobian(
        cls, reference_state: np.ndarray, *, epsilon: float = 1.0e-6
    ) -> np.ndarray:
        """Numerically project the redundant state covariance into SE(3)."""
        reference = cls._orthonormalize(reference_state)
        reference_measurement = cls._observation(reference)
        jacobian = np.zeros((6, cls._STATE_SIZE), dtype=np.float64)
        for column in range(cls._STATE_SIZE):
            perturbed = reference.copy()
            perturbed[column] += epsilon
            perturbed = cls._orthonormalize(perturbed)
            jacobian[:, column] = (
                cls._pose_tangent_residual(reference, cls._observation(perturbed))
                - cls._pose_tangent_residual(reference, reference_measurement)
            ) / epsilon
        return jacobian

    def preview_pose_innovations(
        self,
        poses: list[tuple[np.ndarray, np.ndarray]],
        index: int,
    ) -> list[dict[str, np.ndarray | float]]:
        """Score candidate camera poses against the next UKF prior using NIS.

        The innovation is represented by the local SO(3) logarithm followed by
        ECI translation.  The predicted redundant state covariance is projected
        into this six-dimensional tangent space and combined with the pose
        observation noise before calculating the normalized innovation squared.
        This method does not alter filter state.
        """
        predicted = self._predicted_state(index)
        predicted_mean = self._orthonormalize(predicted.mean())
        jacobian = self._pose_tangent_jacobian(predicted_mean)
        predicted_pose_covariance = jacobian @ predicted.covariance() @ jacobian.T
        measurement_covariance = np.diag(
            [self._rot_measurement_std**2] * 3 + [self._trans_measurement_std_m**2] * 3
        )
        innovation_covariance = self._positive_definite(
            predicted_pose_covariance + measurement_covariance
        )
        results = []
        for rvec, tvec in poses:
            measurement = self.pose_to_measurement(rvec, tvec, index)
            innovation = self._pose_tangent_residual(predicted_mean, measurement)
            whitened = np.linalg.solve(innovation_covariance, innovation)
            results.append(
                {
                    "innovation": innovation,
                    "innovation_covariance": innovation_covariance.copy(),
                    "nis": float(innovation @ whitened),
                }
            )
        return results

    def predict_only(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Advance without a measurement, as required after outlier rejection.

        Iterative RTS is intentionally disabled after this operation because it
        would need an explicit missing-observation representation on replay.
        """
        predicted = self._predicted_state(index)
        self._commit_predicted_state(predicted, index)
        return self.state_to_camera_pose(predicted.mean(), index)

    def _commit_predicted_state(self, predicted: Gaussian, index: int) -> None:
        """Commit an already-computed prediction without an observation."""
        assert self._filter is not None
        self._filter.set_state(predicted)
        self._states.append(Gaussian(predicted.mean().copy(), predicted.covariance().copy()))
        self._times_s.append(index * self.cam_step_s)
        predicted_pose = self.state_to_camera_pose(predicted.mean(), index)
        self._measurements.append((*predicted_pose, index))
        self._contains_prediction_only_steps = True

    def _standard_rts(self) -> list[Gaussian]:
        self._transition_model.transition_noise_covariance = self._process_noise(self.cam_step_s)
        smoother = RTS(BayesianFilter(self._transition_model, self._states[0]))
        return smoother.apply(self._states, self._times_s, use_jacobian=False)

    def _new_like(self) -> ShirtEphemerisPoseRTSSmoother:
        return ShirtEphemerisPoseRTSSmoother(
            self.metadata,
            rot_measurement_std=self._rot_measurement_std,
            trans_measurement_std_m=self._trans_measurement_std_m,
            angular_accel_std_rad_s2=self._angular_accel_std,
            linear_accel_std_m_s2=self._linear_accel_std,
            initial_angular_velocity_std_rad_s=self._initial_angular_velocity_std,
            initial_relative_velocity_std_m_s=self._initial_relative_velocity_std,
        )

    def _seed(self, state: Gaussian, time_s: float) -> None:
        copied = Gaussian(
            self._orthonormalize(state.mean()),
            self._positive_definite(state.covariance()),
        )
        self._filter = BayesianFilter(self._transition_model, copied)
        self._states = [copied]
        self._times_s = [time_s]

    def apply_rts(self, *, num_passes: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
        """Run iterative unscented RTS passes and return one camera pose per forward step.

        Args:
            num_passes: Number of RTS passes, at least one. More than one pass is not
                supported after :meth:`predict_only` steps.

        Returns:
            Smoothed ``(rvec, tvec)`` camera-frame body poses.

        Raises:
            ValueError: If ``num_passes`` is below one.
            RuntimeError: If more than one pass is requested after predict-only steps.
        """
        if num_passes < 1:
            raise ValueError("num_passes must be at least one")
        if self._contains_prediction_only_steps and num_passes > 1:
            raise RuntimeError("iterative RTS replay after predict-only steps is not implemented")
        original_measurements = list(self._measurements)
        current = self
        smoothed_states: list[Gaussian] = []
        for pass_index in range(num_passes):
            smoothed_states = current._standard_rts()
            if pass_index + 1 == num_passes:
                break
            next_pass = self._new_like()
            next_pass._seed(smoothed_states[0], self._times_s[0])
            for rvec, tvec, index in original_measurements[1:]:
                next_pass.update(rvec, tvec, index)
            current = next_pass
        self._last_forward_states = current._states
        self._last_smoothed_states = smoothed_states
        return [
            current.state_to_camera_pose(state.mean(), index)
            for state, (_, _, index) in zip(smoothed_states, original_measurements, strict=True)
        ]

    @property
    def first_forward_poses(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Camera poses of the first forward pass."""
        return [
            self.state_to_camera_pose(state.mean(), index)
            for state, (_, _, index) in zip(self._states, self._measurements, strict=True)
        ]

    @property
    def last_forward_poses(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Camera poses of the final forward pass of the last :meth:`apply_rts` call."""
        return [
            self.state_to_camera_pose(state.mean(), index)
            for state, (_, _, index) in zip(
                self._last_forward_states, self._measurements, strict=False
            )
        ]
