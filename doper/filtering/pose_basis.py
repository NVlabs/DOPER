# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rotation-basis pose math shared by the filters, and the random-walk pose Kalman filter.

State is the 9-vector ``[R[:, 0], R[:, 1], t]`` (the first two columns of the rotation matrix
plus the translation). :class:`PoseBasisKalmanFilter` uses a constant-pose (identity)
transition with random-walk process noise, so it behaves as an exponential smoother whose
stiffness is set by the measurement-vs-process noise ratio. Rotation is re-orthonormalized
after every step (Gram-Schmidt on the two basis vectors), keeping the state on SO(3) without
an explicit manifold filter.

Because the update is a linear blend between the predicted (recent-history) pose and the
measured pose, a single-frame ~180 degree flip that disagrees with the trajectory is strongly
down-weighted, so the filter doubles as a cheap temporal flip and jitter suppressor.

The module-level helpers (:func:`rvec_to_rotation`, :func:`rotation_to_rvec`,
:func:`pose_to_state`, :func:`orthonormalize_state`, :func:`rotation_from_two_vectors` and
:func:`rotation_from_basis`) are also used by the SHIRT smoothers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from doper.filtering._bayesfilter import (
    RTS,
    BayesianFilter,
    Gaussian,
    Observation,
    StateTransitionModel,
)

_STATE_SIZE = 9
_MIN_TIMESTEP_S = 1.0e-3
_TRANSITION_JITTER = 1.0e-12
_SINGULAR_NORM = 1.0e-9
_SEED_ALIGNMENT_LIMIT = 0.95
_JACOBIAN_EPSILON = 1.0e-6
_DEFAULT_RAY_SIGMA_M = 0.02

RotationFromBasis = Callable[[np.ndarray, np.ndarray], np.ndarray]
Pose = tuple[np.ndarray, np.ndarray]


def rvec_to_rotation(rvec: np.ndarray) -> np.ndarray:
    """Convert a Rodrigues vector to a ``(3, 3)`` rotation matrix.

    Args:
        rvec: Rodrigues vector with three elements, any shape.

    Returns:
        The rotation matrix.
    """
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return rotation


def rotation_to_rvec(rotation: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a ``(3, 1)`` Rodrigues vector.

    Args:
        rotation: ``(3, 3)`` rotation matrix.

    Returns:
        The Rodrigues vector.
    """
    rvec, _ = cv2.Rodrigues(rotation)
    return rvec.reshape(3, 1)


def pose_to_state(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Return the 9-vector ``[R[:, 0], R[:, 1], t]`` of a pose.

    Args:
        rvec: Rodrigues vector of the object-to-camera rotation.
        tvec: Translation with three elements, any shape.

    Returns:
        The float64 basis state.
    """
    rotation = rvec_to_rotation(rvec)
    state = np.empty(_STATE_SIZE, dtype=np.float64)
    state[0:3] = rotation[:, 0]
    state[3:6] = rotation[:, 1]
    state[6:9] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return state


def rotation_from_basis(
    first: np.ndarray, second: np.ndarray, *, identity_if_degenerate: bool = False
) -> np.ndarray:
    """Gram-Schmidt rotation from two basis columns.

    The two original filters differ only for a near-zero ``first``: the random-walk filter
    returned the identity, while the SHIRT smoothers kept going with a ``1e-12`` norm clamp.
    The flag selects between them; results are bit-identical for every other input.

    Args:
        first: First rotation column, three elements.
        second: Seed for the second rotation column, three elements.
        identity_if_degenerate: Return the identity when ``|first| < 1e-9``.

    Returns:
        A right-handed ``(3, 3)`` rotation matrix.
    """
    x_axis = np.asarray(first, dtype=np.float64).reshape(3).copy()
    y_seed = np.asarray(second, dtype=np.float64).reshape(3)
    x_norm = float(np.linalg.norm(x_axis))
    if identity_if_degenerate and x_norm < _SINGULAR_NORM:
        return np.eye(3, dtype=np.float64)
    x_axis /= max(x_norm, 1.0e-12)
    y_axis = y_seed - x_axis * float(np.dot(x_axis, y_seed))
    if float(np.linalg.norm(y_axis)) < _SINGULAR_NORM:
        seed = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(seed, x_axis))) > _SEED_ALIGNMENT_LIMIT:
            seed = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        y_axis = seed - x_axis * float(np.dot(x_axis, seed))
    y_axis /= max(float(np.linalg.norm(y_axis)), 1.0e-12)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= max(float(np.linalg.norm(z_axis)), 1.0e-12)
    return np.column_stack((x_axis, np.cross(z_axis, x_axis), z_axis))


def rotation_from_two_vectors(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """:func:`rotation_from_basis` with the identity for a degenerate ``first``."""
    return rotation_from_basis(first, second, identity_if_degenerate=True)


def orthonormalize_state(
    state: np.ndarray,
    *,
    offset: int = 0,
    rotation_from: RotationFromBasis = rotation_from_basis,
) -> np.ndarray:
    """Project the two rotation-basis vectors inside ``state`` back onto SO(3).

    Args:
        state: State vector holding the basis vectors at ``offset:offset + 6``.
        offset: Index of the first basis element in ``state``.
        rotation_from: Basis-to-rotation function, one of the two module variants.

    Returns:
        A float64 copy of ``state`` with orthonormal basis vectors.
    """
    out = np.asarray(state, dtype=np.float64).copy()
    rotation = rotation_from(out[offset : offset + 3], out[offset + 3 : offset + 6])
    out[offset : offset + 3] = rotation[:, 0]
    out[offset + 3 : offset + 6] = rotation[:, 1]
    return out


@dataclass
class _RecordedSegment:
    """One continuous forward-filter chain (split on reset / long dropout)."""

    times_s: list[float] = field(default_factory=list)
    states: list[Gaussian] = field(default_factory=list)
    step_indices: list[int] = field(default_factory=list)


class PoseBasisKalmanFilter:
    """Kalman smoother over ``[R[:, 0], R[:, 1], t]`` using ``bayesfilter``."""

    def __init__(
        self,
        *,
        rot_measurement_std: float = 0.05,
        trans_measurement_std_m: float = 0.004,
        rot_process_std: float = 0.2,
        trans_process_std_m: float = 0.01,
        reset_after_s: float = 1.0,
    ) -> None:
        """Create the filter.

        Args:
            rot_measurement_std: Measurement std for each rotation-basis component (unitless
                direction cosines). Smaller trusts each new pose more.
            trans_measurement_std_m: Measurement std for each translation component, meters.
            rot_process_std: Rotation random-walk process std per ``sqrt(second)``. Larger lets
                the pose move faster between frames (less smoothing, less lag).
            trans_process_std_m: Translation random-walk process std per ``sqrt(second)``.
            reset_after_s: Re-seed the filter after this many seconds without a valid pose.
                ``0`` disables the timeout (the filter is only reset explicitly).
        """
        self._transition_model = StateTransitionModel(
            transition_func=self._identity_transition,
            transition_noise_covariance=np.eye(_STATE_SIZE, dtype=np.float64) * _TRANSITION_JITTER,
            transition_jacobian_func=self._identity_transition_jacobian,
        )
        self._measurement_covariance = np.diag(
            [float(rot_measurement_std) ** 2] * 6 + [float(trans_measurement_std_m) ** 2] * 3
        ).astype(np.float64)
        self._rot_process_std = float(rot_process_std)
        self._trans_process_std_m = float(trans_process_std_m)
        self._reset_after_s = float(reset_after_s)
        self._filter: BayesianFilter | None = None
        self._last_measurement_s: float | None = None
        self._last_propagation_s: float | None = None
        # Optional forward history for offline RTS smoothing (recorded playback).
        self._record = False
        self._steps: list[Gaussian | None] = []
        self._step_times: list[float] = []
        self._segments: list[_RecordedSegment] = []
        self._active_segment: _RecordedSegment | None = None

    # -- transition / observation models (linear identity) ----------------
    @staticmethod
    def _identity_transition(state: np.ndarray, delta_t_s: float) -> np.ndarray:
        return state.copy()

    @staticmethod
    def _identity_transition_jacobian(state: np.ndarray, delta_t_s: float) -> np.ndarray:
        return np.eye(len(state), dtype=np.float64)

    @staticmethod
    def _identity_observation(state: np.ndarray) -> np.ndarray:
        return state.copy()

    @staticmethod
    def _identity_observation_jacobian(state: np.ndarray) -> np.ndarray:
        return np.eye(len(state), dtype=np.float64)

    def reset(self) -> None:
        """Drop the filter state; a recorded chain in progress is closed as a segment."""
        if self._record:
            self._finalize_segment()
        self._filter = None
        self._last_measurement_s = None
        self._last_propagation_s = None

    @property
    def is_active(self) -> bool:
        """Whether a filter state is currently alive."""
        return self._filter is not None

    @property
    def recording(self) -> bool:
        """Whether forward states are being recorded for RTS."""
        return self._record

    def enable_recording(self) -> None:
        """Start recording forward-filter states for a later :meth:`apply_rts` pass."""
        self._record = True
        self._steps = []
        self._step_times = []
        self._segments = []
        self._active_segment = None

    @property
    def num_recorded_steps(self) -> int:
        """Number of recorded forward steps (one per :meth:`update` call)."""
        return len(self._steps)

    def _begin_segment_if_needed(self) -> None:
        if self._active_segment is None:
            self._active_segment = _RecordedSegment()

    def _finalize_segment(self) -> None:
        if self._active_segment is not None and self._active_segment.states:
            self._segments.append(self._active_segment)
        self._active_segment = None

    @staticmethod
    def _copy_gaussian(state: Gaussian) -> Gaussian:
        return Gaussian(mean=state.mean().copy(), covariance=state.covariance().copy())

    def _record_step(self, state: Gaussian | None, now_s: float) -> int:
        """Append one forward-filter step; returns its index."""
        idx = len(self._steps)
        self._steps.append(state)
        self._step_times.append(float(now_s))
        if state is not None:
            self._begin_segment_if_needed()
            assert self._active_segment is not None
            self._active_segment.times_s.append(float(now_s))
            self._active_segment.states.append(self._copy_gaussian(state))
            self._active_segment.step_indices.append(idx)
        return idx

    def _set_process_noise(self, delta_t_s: float) -> None:
        q_rot = (self._rot_process_std**2) * delta_t_s
        q_trans = (self._trans_process_std_m**2) * delta_t_s
        self._transition_model.transition_noise_covariance = np.diag(
            [q_rot] * 6 + [q_trans] * 3
        ).astype(np.float64)

    def apply_rts(self, *, use_jacobian: bool = True) -> list[Pose | None]:
        """RTS-smooth recorded forward-filter states.

        Requires :meth:`enable_recording` before the forward pass.

        Args:
            use_jacobian: Use the transition Jacobian instead of unscented propagation.

        Returns:
            One ``(rvec, tvec)`` per :meth:`update` call, in recording order. Steps with no
            active filter state stay ``None``.
        """
        if self._record:
            self._finalize_segment()
        if not self._steps:
            return []
        out: list[Pose | None] = [None] * len(self._steps)
        for seg in self._segments:
            if not seg.states:
                continue
            if len(seg.states) == 1:
                idx = seg.step_indices[0]
                mean = self.orthonormalize_state(seg.states[0].mean())
                out[idx] = self.state_to_pose(mean)
                continue
            rts = RTS(BayesianFilter(self._transition_model, seg.states[0]))
            smoothed = rts.apply(seg.states, seg.times_s, use_jacobian=use_jacobian)
            for idx, sm_state in zip(seg.step_indices, smoothed, strict=True):
                mean = self.orthonormalize_state(sm_state.mean())
                out[idx] = self.state_to_pose(mean)
        return out

    def _record_current(self, now_s: float, alive: bool = True) -> None:
        """Record the current filter state (or ``None``) when recording is enabled."""
        if not self._record:
            return
        state = self._filter.state if self._filter is not None and alive else None
        self._record_step(state, now_s)

    def update(
        self,
        rvec: np.ndarray | None,
        tvec: np.ndarray | None,
        now_s: float,
        centroid_obj: np.ndarray | None = None,
        rays: list | None = None,
    ) -> Pose | None:
        """Fuse a new pose measurement and return the smoothed ``(rvec, tvec)``.

        Pass ``rvec=None``/``tvec=None`` for a missed detection: if the filter is alive and
        still within ``reset_after_s``, the last smoothed pose is held (coasted) and returned;
        otherwise the filter resets and ``None`` is returned. When a valid measurement is
        supplied the smoothed pose is always returned (falling back to the raw measurement on
        numerical failure).

        Optional mask-centroid ray constraint: if ``centroid_obj`` (the object-frame center of
        mass of the keypoints, a 3-vector) and ``rays`` are given, each ray adds an extra EKF
        update that pulls the projected keypoint centroid onto a camera-to-mask-centroid ray,
        i.e. it drives the perpendicular distance between the 3D keypoint centroid and the ray
        to zero. Each ray is a dict ``{"dir": unit 3-vector (ray direction in that camera's
        frame), "R": 3x3 (rotates a reference-frame point into that camera), "t": 3-vector
        (translation, meters), "sigma_m": float}``. For the reference camera use ``R = I``,
        ``t = 0``. Two cameras' rays triangulate the centroid (constraining depth too); a
        single ray constrains only the lateral position. This is an independent cue (the mask
        head vs the keypoint head), so it stabilises translation and rejects poses whose
        keypoints drift off the object.

        Args:
            rvec: Rodrigues vector of the measured pose, or ``None`` for a missed detection.
            tvec: Measured translation, or ``None`` for a missed detection.
            now_s: Measurement time in seconds.
            centroid_obj: Optional object-frame keypoint centroid for the ray constraint.
            rays: Optional list of ray dicts described above.

        Returns:
            The smoothed ``(rvec, tvec)``, or ``None`` when there is no pose to report.
        """
        if rvec is None or tvec is None:
            result = self._coast(now_s)
            self._record_current(now_s, alive=result is not None)
            return result
        measurement = pose_to_state(rvec, tvec)
        if not np.isfinite(measurement).all():
            result = self._coast(now_s)
            self._record_current(now_s, alive=result is not None)
            return result
        # (Re)seed on first pose or after a long dropout.
        if (
            self._filter is None
            or self._last_measurement_s is None
            or (
                self._reset_after_s > 0.0 and now_s - self._last_measurement_s > self._reset_after_s
            )
        ):
            if self._record and self._filter is not None:
                self._finalize_segment()
            state = Gaussian(
                mean=self.orthonormalize_state(measurement),
                covariance=self._measurement_covariance.copy(),
            )
            self._filter = BayesianFilter(self._transition_model, state)
            self._last_measurement_s = now_s
            self._last_propagation_s = now_s
            mean = self._apply_ray_updates(centroid_obj, rays)
            result = self.state_to_pose(mean)
            self._record_current(now_s)
            return result
        assert self._last_propagation_s is not None
        delta_t_s = max(_MIN_TIMESTEP_S, now_s - self._last_propagation_s)
        self._set_process_noise(delta_t_s)
        observation = Observation(
            observation=measurement,
            noise_covariance=self._measurement_covariance,
            observation_func=self._identity_observation,
            jacobian_func=self._identity_observation_jacobian,
        )
        try:
            predicted = self._filter.predict(self._filter.state, delta_t_s, use_jacobian=True)
            updated = self._filter.update(observation, predicted, use_jacobian=True)
        except np.linalg.LinAlgError:
            self.reset()
            result = self.state_to_pose(measurement)
            if self._record:
                self._record_step(None, now_s)
            return result
        mean = self.orthonormalize_state(updated.mean())
        covariance = 0.5 * (updated.covariance() + updated.covariance().T)
        self._filter.set_state(Gaussian(mean=mean, covariance=covariance))
        self._last_measurement_s = now_s
        self._last_propagation_s = now_s
        mean = self._apply_ray_updates(centroid_obj, rays)
        result = self.state_to_pose(mean)
        self._record_current(now_s)
        return result

    def _apply_ray_updates(self, centroid_obj: np.ndarray | None, rays: list | None) -> np.ndarray:
        """Condition the current state on each mask-centroid ray, in turn.

        Returns the (orthonormalized) state mean after all ray updates.
        """
        if self._filter is None:
            raise RuntimeError("filter not initialized")
        if centroid_obj is None or not rays:
            return self.orthonormalize_state(self._filter.state.mean())
        c_obj = np.asarray(centroid_obj, dtype=np.float64).reshape(3)
        for ray in rays:
            observation = self._ray_observation(c_obj, ray)
            try:
                updated = self._filter.update(observation, self._filter.state, use_jacobian=True)
            except np.linalg.LinAlgError:
                continue
            mean = self.orthonormalize_state(updated.mean())
            cov = 0.5 * (updated.covariance() + updated.covariance().T)
            self._filter.set_state(Gaussian(mean=mean, covariance=cov))
        return self.orthonormalize_state(self._filter.state.mean())

    def _ray_observation(self, c_obj: np.ndarray, ray: dict) -> Observation:
        """Build the perpendicular-distance observation for one mask-centroid ray."""
        d = np.asarray(ray["dir"], dtype=np.float64).reshape(3)
        d = d / max(float(np.linalg.norm(d)), _SINGULAR_NORM)
        R_ref2cam = np.asarray(ray.get("R", np.eye(3)), dtype=np.float64).reshape(3, 3)  # noqa: N806
        t_ref2cam = np.asarray(ray.get("t", np.zeros(3)), dtype=np.float64).reshape(3)
        sigma = float(ray.get("sigma_m", _DEFAULT_RAY_SIGMA_M))

        def perp(state: np.ndarray) -> np.ndarray:
            rot = self.rotation_from_two_vectors(state[0:3], state[3:6])
            c_ref = rot @ c_obj + state[6:9]
            c_cam = R_ref2cam @ c_ref + t_ref2cam
            return c_cam - float(np.dot(c_cam, d)) * d

        def perp_jacobian(state: np.ndarray) -> np.ndarray:
            eps = _JACOBIAN_EPSILON
            jac = np.zeros((3, _STATE_SIZE), dtype=np.float64)
            for i in range(_STATE_SIZE):
                sp = state.copy()
                sp[i] += eps
                sm = state.copy()
                sm[i] -= eps
                jac[:, i] = (perp(sp) - perp(sm)) / (2.0 * eps)
            return jac

        return Observation(
            observation=np.zeros(3, dtype=np.float64),
            noise_covariance=np.eye(3, dtype=np.float64) * (sigma**2),
            observation_func=perp,
            jacobian_func=perp_jacobian,
        )

    def _coast(self, now_s: float) -> Pose | None:
        if self._filter is None or self._last_measurement_s is None:
            return None
        if self._reset_after_s > 0.0 and now_s - self._last_measurement_s > self._reset_after_s:
            self.reset()
            return None
        # During RTS recording, advance the forward chain with a predict-only step
        # so the backward pass has a proper transition between timestamps.
        if self._record:
            assert self._last_propagation_s is not None
            delta_t_s = max(_MIN_TIMESTEP_S, now_s - self._last_propagation_s)
            self._set_process_noise(delta_t_s)
            try:
                predicted = self._filter.predict(self._filter.state, delta_t_s, use_jacobian=True)
                self._filter.set_state(predicted)
                self._last_propagation_s = now_s
                return self.state_to_pose(predicted.mean())
            except np.linalg.LinAlgError:
                self.reset()
                return None
        return self.state_to_pose(self._filter.state.mean())

    # -- state <-> pose helpers ------------------------------------------
    @staticmethod
    def pose_to_state(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """Return the 9-vector ``[R[:, 0], R[:, 1], t]`` of a pose."""
        return pose_to_state(rvec, tvec)

    @classmethod
    def orthonormalize_state(cls, state: np.ndarray) -> np.ndarray:
        """Project the rotation-basis part of ``state`` back onto SO(3)."""
        return orthonormalize_state(state, rotation_from=cls.rotation_from_two_vectors)

    @staticmethod
    def rotation_from_two_vectors(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        """Gram-Schmidt rotation from two basis columns (see the module function)."""
        return rotation_from_two_vectors(first, second)

    @classmethod
    def state_to_pose(cls, state: np.ndarray) -> Pose:
        """Convert a state vector to ``(rvec, tvec)`` with shapes ``(3, 1)``."""
        rotation = cls.rotation_from_two_vectors(state[0:3], state[3:6])
        return rotation_to_rvec(rotation), state[6:9].reshape(3, 1).astype(np.float64)
