# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constant-velocity pose filtering and RTS smoothing for SHIRT trajectories.

The live :mod:`doper.filtering.pose_basis` filter is intentionally a random-walk/constant-pose
model. SHIRT is different: consecutive images are five seconds apart and the target undergoes
smooth, persistent translation and rotation. This module therefore tracks

``[R[:, 0], R[:, 1], translation, angular_velocity, linear_velocity]``

and propagates it with constant camera-frame angular and linear velocity. The orientation is
advanced on SO(3), rather than by adding Rodrigues vectors. Small angular- and
linear-acceleration noise allows gradual departures from the constant-velocity model.
"""

from __future__ import annotations

import numpy as np

from doper.filtering._bayesfilter import (
    RTS,
    BayesianFilter,
    Gaussian,
    Observation,
    StateTransitionModel,
)
from doper.filtering.pose_basis import (
    orthonormalize_state,
    pose_to_state,
    rotation_from_basis,
    rotation_to_rvec,
    rvec_to_rotation,
)

_STATE_SIZE = 15
_POSE_SIZE = 9
_TRANSITION_JITTER = 1.0e-12
_MIN_TIMESTEP_S = 1.0e-6

Pose = tuple[np.ndarray, np.ndarray]


class ShirtPoseRTSSmoother:
    """UKF/RTS smoother tailored to the cadence and motion in SHIRT.

    Rotation measurements are represented by the first two columns of the object-to-camera
    rotation matrix. Angular velocity is camera-frame and left-multiplies the rotation during
    propagation. Translation and linear velocity are in the camera frame and meters and
    meters-per-second.
    """

    _STATE_SIZE = _STATE_SIZE

    def __init__(
        self,
        *,
        rot_measurement_std: float = 0.05,
        trans_measurement_std_m: float = 0.10,
        angular_accel_std_rad_s2: float = 5.0e-5,
        linear_accel_std_m_s2: float = 5.0e-5,
        initial_angular_velocity_std_rad_s: float = 0.05,
        initial_linear_velocity_std_m_s: float = 0.01,
    ) -> None:
        """Create the smoother.

        Args:
            rot_measurement_std: Measurement std per rotation-basis component.
            trans_measurement_std_m: Measurement std per translation component, meters.
            angular_accel_std_rad_s2: Angular-acceleration process noise std.
            linear_accel_std_m_s2: Linear-acceleration process noise std.
            initial_angular_velocity_std_rad_s: Prior std of the initial angular velocity.
            initial_linear_velocity_std_m_s: Prior std of the initial linear velocity.
        """
        self._angular_accel_std = float(angular_accel_std_rad_s2)
        self._linear_accel_std = float(linear_accel_std_m_s2)
        self._initial_angular_velocity_std = float(initial_angular_velocity_std_rad_s)
        self._initial_linear_velocity_std = float(initial_linear_velocity_std_m_s)
        self._rot_measurement_std = float(rot_measurement_std)
        self._trans_measurement_std_m = float(trans_measurement_std_m)
        self._measurement_covariance = np.diag(
            [self._rot_measurement_std**2] * 6 + [self._trans_measurement_std_m**2] * 3
        ).astype(np.float64)
        self._transition_model = StateTransitionModel(
            transition_func=self._transition,
            transition_noise_covariance=np.eye(self._STATE_SIZE, dtype=np.float64)
            * _TRANSITION_JITTER,
        )
        self._filter: BayesianFilter | None = None
        self._last_time_s: float | None = None
        self._states: list[Gaussian] = []
        self._times_s: list[float] = []
        self._measurements: list[tuple[np.ndarray, np.ndarray]] = []
        self._last_forward_state_means: list[np.ndarray] = []
        self._last_smoothed_state_means: list[np.ndarray] = []

    @classmethod
    def _orthonormalize(cls, state: np.ndarray) -> np.ndarray:
        return orthonormalize_state(state)

    @classmethod
    def _transition(cls, state: np.ndarray, delta_t_s: float) -> np.ndarray:
        state = cls._orthonormalize(state)
        dt = float(delta_t_s)
        rotation = rotation_from_basis(state[0:3], state[3:6])
        angular_velocity = state[9:12]
        delta_rotation = rvec_to_rotation(angular_velocity * dt)
        propagated_rotation = delta_rotation @ rotation
        out = state.copy()
        out[0:3] = propagated_rotation[:, 0]
        out[3:6] = propagated_rotation[:, 1]
        out[6:9] += state[12:15] * dt
        return out

    @staticmethod
    def _observation(state: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=np.float64)[0:_POSE_SIZE].copy()

    @classmethod
    def _observation_jacobian(cls, state: np.ndarray) -> np.ndarray:
        del state
        jacobian = np.zeros((_POSE_SIZE, cls._STATE_SIZE), dtype=np.float64)
        jacobian[:, 0:_POSE_SIZE] = np.eye(_POSE_SIZE, dtype=np.float64)
        return jacobian

    def _process_noise(self, delta_t_s: float) -> np.ndarray:
        dt = max(float(delta_t_s), _MIN_TIMESTEP_S)
        angular_variance = self._angular_accel_std**2
        linear_variance = self._linear_accel_std**2
        covariance = np.eye(self._STATE_SIZE, dtype=np.float64) * _TRANSITION_JITTER
        # Continuous white-acceleration model. Basis-vector uncertainty is an
        # isotropic small-angle approximation; SO(3) projection follows every
        # predict/update and every RTS output.
        covariance[0:6, 0:6] += np.eye(6) * angular_variance * dt**3 / 3.0
        covariance[9:12, 9:12] += np.eye(3) * angular_variance * dt
        covariance[6:9, 6:9] += np.eye(3) * linear_variance * dt**3 / 3.0
        covariance[12:15, 12:15] += np.eye(3) * linear_variance * dt
        return covariance

    def _set_process_noise(self, delta_t_s: float) -> None:
        self._transition_model.transition_noise_covariance = self._process_noise(delta_t_s)

    @classmethod
    def pose_to_measurement(cls, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """Return the 9-vector ``[R[:, 0], R[:, 1], t]`` measurement of a pose."""
        return pose_to_state(rvec, tvec)

    @classmethod
    def state_to_pose(cls, state: np.ndarray) -> Pose:
        """Convert a state vector to ``(rvec, tvec)`` with shapes ``(3, 1)``.

        Kept per class: unlike the other filters it orthonormalizes first and returns ``t`` as a
        view, and the outputs must stay bit-identical to the original.
        """
        state = cls._orthonormalize(state)
        rotation = rotation_from_basis(state[0:3], state[3:6])
        return rotation_to_rvec(rotation), state[6:9].reshape(3, 1)

    def update(self, rvec: np.ndarray, tvec: np.ndarray, now_s: float) -> Pose:
        """Add one pose measurement and return the forward-UKF pose.

        Args:
            rvec: Rodrigues vector of the measured object-to-camera rotation.
            tvec: Measured translation, meters.
            now_s: Measurement time in seconds.

        Returns:
            The forward-filtered ``(rvec, tvec)``.
        """
        measurement = self.pose_to_measurement(rvec, tvec)
        if self._filter is None:
            mean = np.zeros(self._STATE_SIZE, dtype=np.float64)
            mean[0:_POSE_SIZE] = measurement
            covariance = np.zeros((self._STATE_SIZE, self._STATE_SIZE), dtype=np.float64)
            covariance[0:_POSE_SIZE, 0:_POSE_SIZE] = self._measurement_covariance
            covariance[9:12, 9:12] = np.eye(3) * self._initial_angular_velocity_std**2
            covariance[12:15, 12:15] = np.eye(3) * self._initial_linear_velocity_std**2
            state = Gaussian(mean=self._orthonormalize(mean), covariance=covariance)
            self._filter = BayesianFilter(self._transition_model, state)
        else:
            assert self._last_time_s is not None
            delta_t_s = max(float(now_s) - self._last_time_s, _MIN_TIMESTEP_S)
            self._set_process_noise(delta_t_s)
            predicted = self._filter.predict(self._filter.state, delta_t_s, use_jacobian=False)
            observation = Observation(
                observation=measurement,
                noise_covariance=self._measurement_covariance,
                observation_func=self._observation,
                jacobian_func=self._observation_jacobian,
            )
            updated = self._filter.update(observation, predicted, use_jacobian=False)
            mean = self._orthonormalize(updated.mean())
            covariance = 0.5 * (updated.covariance() + updated.covariance().T)
            self._filter.set_state(Gaussian(mean=mean, covariance=covariance))
        self._last_time_s = float(now_s)
        self._measurements.append(
            (
                np.asarray(rvec, dtype=np.float64).reshape(3, 1).copy(),
                np.asarray(tvec, dtype=np.float64).reshape(3, 1).copy(),
            )
        )
        self._states.append(
            Gaussian(
                mean=self._filter.state.mean().copy(),
                covariance=self._filter.state.covariance().copy(),
            )
        )
        self._times_s.append(float(now_s))
        return self.state_to_pose(self._filter.state.mean())

    def _standard_bayesfilter_rts(self) -> list[Gaussian]:
        """Apply one standard unscented RTS pass to the recorded UKF states."""
        if len(self._states) == 1:
            return [self._states[0]]
        self._set_process_noise(self._times_s[1] - self._times_s[0])
        smoother = RTS(BayesianFilter(self._transition_model, self._states[0]))
        return smoother.apply(self._states, self._times_s, use_jacobian=False)

    def _new_like(self) -> ShirtPoseRTSSmoother:
        return ShirtPoseRTSSmoother(
            rot_measurement_std=self._rot_measurement_std,
            trans_measurement_std_m=self._trans_measurement_std_m,
            angular_accel_std_rad_s2=self._angular_accel_std,
            linear_accel_std_m_s2=self._linear_accel_std,
            initial_angular_velocity_std_rad_s=self._initial_angular_velocity_std,
            initial_linear_velocity_std_m_s=self._initial_linear_velocity_std,
        )

    def _seed_from_smoothed_state(self, state: Gaussian, time_s: float) -> None:
        """Initialize a new forward pass from one inferred state distribution."""
        mean = self._orthonormalize(state.mean())
        covariance = 0.5 * (state.covariance() + state.covariance().T)
        copied = Gaussian(mean=mean, covariance=covariance)
        self._filter = BayesianFilter(self._transition_model, copied)
        self._last_time_s = float(time_s)
        self._states = [Gaussian(mean=mean.copy(), covariance=covariance.copy())]
        self._times_s = [float(time_s)]

    def apply_rts(self, *, num_passes: int = 1) -> list[Pose]:
        """Run repeated standard unscented RTS passes.

        The first pass smooths the UKF states produced from the original pose measurements.
        Each additional pass initializes a fresh forward UKF from the preceding pass's first
        smoothed state, then assimilates the *original* measurements from frame 1 onward.
        Smoothed intermediate poses are never used as measurements.

        Args:
            num_passes: Number of RTS passes, at least one.

        Returns:
            One smoothed ``(rvec, tvec)`` per measurement.

        Raises:
            ValueError: If ``num_passes`` is below one.
        """
        if not self._states:
            return []
        if num_passes < 1:
            raise ValueError("num_passes must be at least one")
        original_measurements = list(self._measurements)
        original_times_s = list(self._times_s)
        current = self
        smoothed_states: list[Gaussian] = []
        for pass_index in range(num_passes):
            smoothed_states = current._standard_bayesfilter_rts()
            if pass_index + 1 == num_passes:
                break
            next_pass = current._new_like()
            next_pass._seed_from_smoothed_state(smoothed_states[0], original_times_s[0])
            for time_s, measurement in zip(
                original_times_s[1:], original_measurements[1:], strict=True
            ):
                next_pass.update(measurement[0], measurement[1], time_s)
            current = next_pass
        self._last_forward_state_means = [state.mean().copy() for state in current._states]
        self._last_smoothed_state_means = [state.mean().copy() for state in smoothed_states]
        return [current.state_to_pose(state.mean()) for state in smoothed_states]

    @property
    def last_forward_state_means(self) -> list[np.ndarray]:
        """Forward-UKF state means from the final iterative pass."""
        return [state.copy() for state in self._last_forward_state_means]

    @property
    def last_smoothed_state_means(self) -> list[np.ndarray]:
        """RTS state means from the final iterative pass."""
        return [state.copy() for state in self._last_smoothed_state_means]
