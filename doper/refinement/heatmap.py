# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keypoint-heatmap likelihood pose refinement over one or more calibrated views.

Starting from a PnP pose, the object pose is refined so that its projected keypoints maximize the
bilinearly sampled log-probability of the network's keypoint heatmaps in every view. The pose
change is a rotation about the keypoint centroid plus a translation, bounded to +/-45 degrees
(per rotation-vector component) and +/-30 mm. A keypoint behind a camera (z <= 1e-5 m) or
outside its heatmap costs ``-log(1e-12)``.

Two optimizers are available:

* ``lbfgsb`` (default): SciPy L-BFGS-B on the mean negative log-probability with the exact
  piecewise-bilinear gradient from a Numba kernel (needs ``doper[refine]``).
* ``powell``: SciPy Powell on the summed negative log-probability (SciPy only). This is the
  optimizer behind the paper's heatmap rows; the two differ by -0.1..+0.2 AP.

Both accept the result only if the final loss is finite, strictly below the initial loss, and
every keypoint lies in front of every camera; otherwise the input pose is returned unchanged.
Preparation turns logits into float64 log-softmax maps, either with NumPy (default) or with a
channel-parallel Numba kernel (``parallel``, L-BFGS-B only; within ~1e-12 of NumPy).
"""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Literal, NamedTuple

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

Optimizer = Literal["lbfgsb", "powell"]
Preparation = Literal["numpy", "parallel"]

ROTATION_BOUND_RAD = np.pi / 4
TRANSLATION_BOUND_M = 0.03
# Normalized parameter u in [-1, 1]^6 maps to delta = u * SCALE (rotation vector rad, meters).
SCALE = np.array([ROTATION_BOUND_RAD] * 3 + [TRANSLATION_BOUND_M] * 3)
INVALID_PROBABILITY = 1e-12
INVALID = float(np.log(INVALID_PROBABILITY))
DEPTH_EPSILON_M = 1e-5
BOUND_HIT_TOLERANCE = 1e-6
MM_PER_M = 1000
LBFGSB_OPTIONS: dict[str, float] = {
    "maxiter": 200,
    "maxfun": 10000,
    "ftol": 1e-9,
    "gtol": 1e-5,
    "maxls": 30,
}
POWELL_OPTIONS: dict[str, float] = {"maxiter": 200, "xtol": 1e-5, "ftol": 1e-5}
LBFGSB_AUDIT_VERSION = 2
DEFAULT_PREPARATION_THREADS = 4
REFINE_EXTRA_MESSAGE = (
    "the L-BFGS-B heatmap optimizer and the parallel preparation need numba; "
    "install doper[refine] or use optimizer='powell'"
)


@dataclass(frozen=True)
class HeatmapObservation:
    """One view of the object: raw keypoint logits plus the geometry to project into them.

    Attributes:
        logits: ``(K, H, W)`` raw keypoint-heatmap logits of the crop head (float32 from
            ``doper.infer``), one channel per keypoint.
        crop_to_image: ``(2, 3)`` affine from crop pixels to image pixels.
        K: ``(3, 3)`` pinhole intrinsics whose last row is ``[0, 0, 1]``.
        relative_mm: ``(4, 4)`` reference-camera to this-camera transform, translation in mm.
            The identity for the reference view. It is stored as float64 before the translation
            is divided by 1000.
    """

    logits: np.ndarray
    crop_to_image: np.ndarray
    K: np.ndarray  # noqa: N815
    relative_mm: np.ndarray

    def __post_init__(self) -> None:
        """Validate shapes and store the geometry as float64 copies."""
        logits = np.asarray(self.logits)
        if logits.ndim != 3:
            raise ValueError(f"logits must be (K, H, W), got shape {logits.shape}")
        geometry = {
            "crop_to_image": (2, 3),
            "K": (3, 3),
            "relative_mm": (4, 4),
        }
        for name, shape in geometry.items():
            value = np.array(getattr(self, name), dtype=np.float64)
            if value.shape != shape:
                raise ValueError(f"{name} must be {shape}, got shape {value.shape}")
            object.__setattr__(self, name, value)
        if not np.allclose(self.K[2], [0, 0, 1]):
            raise ValueError(f"K must have last row [0, 0, 1], got {self.K[2].tolist()}")
        object.__setattr__(self, "logits", logits)


@dataclass(frozen=True)
class RefineSettings:
    """Optimizer and preparation choice.

    Attributes:
        optimizer: ``lbfgsb`` (default, Numba) or ``powell`` (SciPy only).
        preparation: ``numpy`` (default) or ``parallel`` (Numba, L-BFGS-B only).
        threads: Upper bound on Numba threads for the parallel preparation.
    """

    optimizer: Optimizer = "lbfgsb"
    preparation: Preparation = "numpy"
    threads: int = DEFAULT_PREPARATION_THREADS

    def __post_init__(self) -> None:
        """Reject unknown values and Powell with the parallel preparation."""
        if self.optimizer not in ("lbfgsb", "powell"):
            raise ValueError(f"unknown optimizer {self.optimizer!r}; use 'lbfgsb' or 'powell'")
        if self.preparation not in ("numpy", "parallel"):
            raise ValueError(f"unknown preparation {self.preparation!r}; use 'numpy' or 'parallel'")
        if self.optimizer == "powell" and self.preparation != "numpy":
            raise ValueError("Powell uses its original NumPy preparation; set preparation='numpy'")
        if self.threads < 1:
            raise ValueError(f"threads must be >= 1, got {self.threads}")


@dataclass(frozen=True)
class LbfgsbAudit:
    """L-BFGS-B outcome (audit version 2). NLLs are sums over keypoints x views.

    Attributes:
        initial_nll: Negative log-likelihood at the input pose.
        final_nll: Negative log-likelihood at the optimizer's pose.
        accepted: Whether the refined pose was returned.
        success: SciPy's success flag (not required for acceptance).
        status: SciPy's status code.
        message: SciPy's message.
        gradient_inf_norm: Infinity norm of the final gradient (normalized parameters).
        iterations: L-BFGS-B iterations.
        evaluations: Objective evaluations (``nfev``).
        views: Number of views.
        delta_normalized: Final normalized parameter ``x`` in ``[-1, 1]^6``.
        delta_rotation_rad: Rotation-vector change, radians.
        delta_translation_mm: Translation change, mm.
        bound_hits: Per parameter, whether ``|x| >= 1 - 1e-6``.
        any_bound_hit: Any parameter at its bound.
        rotation_bound_hit: Any rotation parameter at its bound.
        translation_bound_hit: Any translation parameter at its bound.
        positive_depth: Every keypoint lies in front of every camera.
        finite_objective: The final loss is finite.
        improved: The final loss is strictly below the initial loss.
        projected_gradient_inf_norm: Infinity norm of the gradient projected onto the box.
        rejection_reasons: ``nonfinite_objective``, ``no_improvement``, ``nonpositive_depth``.
    """

    initial_nll: float
    final_nll: float
    accepted: bool
    success: bool
    status: int
    message: str
    gradient_inf_norm: float
    iterations: int
    evaluations: int
    views: int
    delta_normalized: tuple[float, ...]
    delta_rotation_rad: tuple[float, ...]
    delta_translation_mm: tuple[float, ...]
    bound_hits: tuple[bool, ...]
    any_bound_hit: bool
    rotation_bound_hit: bool
    translation_bound_hit: bool
    positive_depth: bool
    finite_objective: bool
    improved: bool
    projected_gradient_inf_norm: float
    rejection_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """The audit dict (``audit_version`` 2, sequences as lists)."""
        record: dict[str, Any] = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            record[field.name] = list(value) if isinstance(value, tuple) else value
            if field.name == "views":
                record["audit_version"] = LBFGSB_AUDIT_VERSION
        return record


@dataclass(frozen=True)
class PowellAudit:
    """Powell outcome. NLLs are sums over keypoints x views.

    Attributes:
        initial_nll: Negative log-likelihood at the input pose.
        final_nll: Negative log-likelihood at the optimizer's pose.
        accepted: Whether the refined pose was returned.
        success: SciPy's success flag (not required for acceptance).
        iterations: Powell iterations.
        evaluations: Objective evaluations.
        views: Number of views.
    """

    initial_nll: float
    final_nll: float
    accepted: bool
    success: bool
    iterations: int
    evaluations: int
    views: int

    def to_dict(self) -> dict[str, Any]:
        """The audit dict."""
        return dataclasses.asdict(self)


RefinementAudit = LbfgsbAudit | PowellAudit


class PreparedView(NamedTuple):
    """One view ready for the L-BFGS-B objective.

    Attributes:
        relative_m: ``(4, 4)`` reference-to-view transform, translation in meters.
        projection: ``(3, 3)`` camera-to-heatmap projection (heatmap warp times ``K``).
        log_probabilities: ``(K, H, W)`` float64 log-softmax maps.
    """

    relative_m: np.ndarray
    projection: np.ndarray
    log_probabilities: np.ndarray


def _load_kernels() -> ModuleType:
    try:
        return importlib.import_module("doper.refinement._numba_kernels")
    except ImportError as err:
        raise ImportError(REFINE_EXTRA_MESSAGE) from err


def _log_softmax_inplace(logits: np.ndarray, beta: float) -> np.ndarray:
    """L-BFGS-B NumPy preparation: float64 ``log_softmax(beta * logits)`` per channel.

    Subtracts the max, then the log-sum-exp, in place. Powell keeps its own form
    (``_log_softmax``); the two differ in rounding, so each optimizer keeps its own.
    """
    scaled = np.array(logits, dtype=np.float64, copy=True)
    scaled *= float(beta)
    flat = scaled.reshape(scaled.shape[0], -1)
    flat -= flat.max(axis=1, keepdims=True)
    scratch = np.empty_like(flat)
    np.exp(flat, out=scratch)
    flat -= np.log(scratch.sum(axis=1, keepdims=True))
    return scaled


def _prepared_view(
    observation: HeatmapObservation, log_probabilities: np.ndarray, crop_size: int
) -> PreparedView:
    relative = np.asarray(observation.relative_mm, float).copy()
    relative[:3, 3] /= MM_PER_M
    warp = np.eye(3)
    warp[:2] = cv2.invertAffineTransform(np.asarray(observation.crop_to_image, float))
    warp[:2] *= (log_probabilities.shape[-1] - 1) / float(crop_size - 1)
    camera = np.asarray(observation.K, float)
    return PreparedView(relative, warp @ camera, log_probabilities)


def prepare_observations(
    observations: Sequence[HeatmapObservation],
    crop_size: int,
    beta: float,
    *,
    preparation: Preparation = "numpy",
    threads: int = DEFAULT_PREPARATION_THREADS,
) -> list[PreparedView]:
    """Turn observations into the float64 log-probability maps and projections L-BFGS-B uses.

    Args:
        observations: Views, reference first.
        crop_size: Crop side in pixels (``cfg.model.crop_size``).
        beta: Softmax temperature (``cfg.loss.keypoint_heatmap_beta``).
        preparation: ``numpy`` or ``parallel`` (Numba, ``doper[refine]``).
        threads: Upper bound on Numba threads for ``parallel``; the previous count is restored.

    Returns:
        One ``PreparedView`` per observation.
    """
    if preparation == "numpy":
        return [
            _prepared_view(obs, _log_softmax_inplace(obs.logits, beta), crop_size)
            for obs in observations
        ]
    if preparation != "parallel":
        raise ValueError(f"unknown preparation {preparation!r}; use 'numpy' or 'parallel'")
    kernels = _load_kernels()
    with kernels.limited_threads(threads):
        return [
            _prepared_view(
                obs, kernels.fused_log_softmax(np.asarray(obs.logits), float(beta)), crop_size
            )
            for obs in observations
        ]


class HeatmapObjective:
    """Mean negative log-probability over keypoints x views and its gradient in ``u``.

    ``u`` in ``[-1, 1]^6`` is the normalized pose change: a rotation vector about the keypoint
    centroid (``u[:3] * pi/4``) and a translation (``u[3:] * 0.03`` m).
    """

    def __init__(
        self,
        rvec: np.ndarray,
        tvec: np.ndarray,
        keypoints_mm: np.ndarray,
        prepared: Sequence[PreparedView],
    ) -> None:
        """Center the keypoints and allocate the evaluation buffers.

        Args:
            rvec: Initial rotation vector (object to reference camera).
            tvec: Initial translation, mm.
            keypoints_mm: ``(N, 3)`` object keypoints, mm.
            prepared: Prepared views from ``prepare_observations``.
        """
        self._kernels = _load_kernels()
        self.points = np.asarray(keypoints_mm, float) / MM_PER_M
        self.rotation = cv2.Rodrigues(np.asarray(rvec, float))[0]
        self.centroid = self.points.mean(0)
        self.center = self.rotation @ self.centroid + np.asarray(tvec).reshape(3) / MM_PER_M
        self.q = (self.points - self.centroid) @ self.rotation.T
        self.prepared = prepared
        self.div = len(self.points) * len(prepared)
        self.xyz = np.empty_like(self.q)
        self.dxyz = np.empty((3, len(self.points), 3))
        self.gradient = np.zeros(6)

    def __call__(self, u: np.ndarray) -> tuple[float, np.ndarray]:
        """Return the mean loss and its gradient with respect to ``u``."""
        delta = u * SCALE
        rotation, jacobian = cv2.Rodrigues(delta[:3])
        self.xyz[:] = self.q @ rotation.T + self.center + delta[3:]
        np.einsum("kij,nj->kni", jacobian.reshape(3, 3, 3), self.q, out=self.dxyz)  # pyright: ignore[reportCallIssue, reportArgumentType]
        self.gradient.fill(0)
        value = 0.0
        for relative, projection, logs in self.prepared:
            value += self._kernels.sample_gradient(
                self.xyz, self.dxyz, relative, projection, logs, INVALID, self.gradient
            )
        return value / self.div, self.gradient * SCALE / self.div

    def pose(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rotation matrix and translation (meters) after the change ``u``."""
        delta = u * SCALE
        rotation = cv2.Rodrigues(delta[:3])[0] @ self.rotation
        translation = self.center + delta[3:] - rotation @ self.centroid
        return rotation, translation


def _rejection_reasons(finite: bool, improved: bool, positive: bool) -> tuple[str, ...]:
    reasons = [] if finite else ["nonfinite_objective"]
    reasons += [] if improved else ["no_improvement"]
    reasons += [] if positive else ["nonpositive_depth"]
    return tuple(reasons)


def _lbfgsb_audit(
    result: Any,  # noqa: ANN401
    initial: float,
    div: int,
    positive: bool,
    views: int,
) -> LbfgsbAudit:
    finite = bool(np.isfinite(result.fun))
    improved = bool(result.fun < initial)
    accepted = bool(np.isfinite(result.fun) and result.fun < initial and positive)
    bound_hits = np.abs(result.x) >= 1.0 - BOUND_HIT_TOLERANCE
    projected_gradient = np.array(result.jac, copy=True)
    projected_gradient[(result.x <= -1.0 + BOUND_HIT_TOLERANCE) & (projected_gradient > 0)] = 0
    projected_gradient[(result.x >= 1.0 - BOUND_HIT_TOLERANCE) & (projected_gradient < 0)] = 0
    return LbfgsbAudit(
        initial_nll=initial * div,
        final_nll=float(result.fun) * div,
        accepted=accepted,
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        gradient_inf_norm=float(np.max(np.abs(result.jac))),
        iterations=int(result.nit),
        evaluations=int(result.nfev),
        views=views,
        delta_normalized=tuple(result.x.tolist()),
        delta_rotation_rad=tuple((result.x[:3] * SCALE[:3]).tolist()),
        delta_translation_mm=tuple((result.x[3:] * SCALE[3:] * MM_PER_M).tolist()),
        bound_hits=tuple(bound_hits.tolist()),
        any_bound_hit=bool(bound_hits.any()),
        rotation_bound_hit=bool(bound_hits[:3].any()),
        translation_bound_hit=bool(bound_hits[3:].any()),
        positive_depth=bool(positive),
        finite_objective=finite,
        improved=improved,
        projected_gradient_inf_norm=float(np.max(np.abs(projected_gradient))),
        rejection_reasons=() if accepted else _rejection_reasons(finite, improved, positive),
    )


def _refine_lbfgsb(
    rvec: np.ndarray,
    tvec: np.ndarray,
    keypoints_mm: np.ndarray,
    prepared: Sequence[PreparedView],
    views: int,
) -> tuple[np.ndarray, np.ndarray, LbfgsbAudit]:
    objective = HeatmapObjective(rvec, tvec, keypoints_mm, prepared)
    initial = objective(np.zeros(6))[0]
    result = minimize(
        objective,
        np.zeros(6),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-1, 1)] * 6,
        options=LBFGSB_OPTIONS,
    )
    rotation, translation = objective.pose(result.x)
    xyz = objective.points @ rotation.T + translation
    positive = all(
        np.all((xyz @ rel[:3, :3].T + rel[:3, 3])[:, 2] > DEPTH_EPSILON_M) for rel, _, _ in prepared
    )
    audit = _lbfgsb_audit(result, initial, objective.div, positive, views)
    if not audit.accepted:
        return rvec, tvec, audit
    return cv2.Rodrigues(rotation)[0], translation.reshape(3, 1) * MM_PER_M, audit


def _log_softmax(logits: np.ndarray, beta: float) -> np.ndarray:
    """Powell preparation: float64 ``log_softmax(beta * logits)`` per channel."""
    scaled = np.asarray(logits, dtype=np.float64) * float(beta)
    flat = scaled.reshape(scaled.shape[0], -1)
    maximum = flat.max(axis=1, keepdims=True)
    log_normalizer = maximum + np.log(np.exp(flat - maximum).sum(axis=1, keepdims=True))
    return (flat - log_normalizer).reshape(scaled.shape)


def _bilinear_channel_samples(
    maps: np.ndarray, points_xy: np.ndarray, invalid_value: float
) -> np.ndarray:
    """Sample channel ``k`` of ``maps`` at ``points_xy[k]``; ``invalid_value`` outside."""
    keypoints, height, width = maps.shape
    values = np.full(keypoints, invalid_value, dtype=np.float64)
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    valid = (
        np.isfinite(points_xy).all(axis=1)
        & (x >= 0.0)
        & (x <= width - 1)
        & (y >= 0.0)
        & (y <= height - 1)
    )
    indices = np.flatnonzero(valid)
    if not len(indices):
        return values
    x0 = np.floor(x[indices]).astype(np.int64)
    y0 = np.floor(y[indices]).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    dx = x[indices] - x0
    dy = y[indices] - y0
    values[indices] = (
        (1.0 - dx) * (1.0 - dy) * maps[indices, y0, x0]
        + dx * (1.0 - dy) * maps[indices, y0, x1]
        + (1.0 - dx) * dy * maps[indices, y1, x0]
        + dx * dy * maps[indices, y1, x1]
    )
    return values


def _project_to_heatmap(
    keypoints_3d: np.ndarray,
    transform_camera_from_object: np.ndarray,
    camera_matrix: np.ndarray,
    crop_to_image: np.ndarray,
    crop_size: int,
    heatmap_size: int,
) -> np.ndarray:
    """Heatmap coordinates of the keypoints; NaN for a keypoint with z <= 1e-5."""
    rotation = transform_camera_from_object[:3, :3]
    translation = transform_camera_from_object[:3, 3]
    points_camera = (rotation @ keypoints_3d.T).T + translation
    depth_valid = points_camera[:, 2] > DEPTH_EPSILON_M
    projected = np.full((len(keypoints_3d), 2), np.nan, dtype=np.float64)
    projected[depth_valid] = (camera_matrix @ points_camera[depth_valid].T).T[
        :, :2
    ] / points_camera[depth_valid, 2:3]
    image_to_crop = cv2.invertAffineTransform(np.asarray(crop_to_image, dtype=np.float64))
    homogeneous = np.concatenate(
        [projected, np.ones((len(projected), 1), dtype=np.float64)],
        axis=1,
    )
    points_crop = homogeneous @ image_to_crop.T
    scale = (heatmap_size - 1) / float(crop_size - 1)
    return points_crop * scale


def _refine_powell(
    rvec: np.ndarray,
    tvec: np.ndarray,
    keypoints_mm: np.ndarray,
    observations: Sequence[HeatmapObservation],
    crop_size: int,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, PowellAudit]:
    points = np.asarray(keypoints_mm, dtype=np.float64) / MM_PER_M
    rotation = cv2.Rodrigues(rvec)[0]
    centroid = points.mean(axis=0)
    center = rotation @ centroid + np.asarray(tvec).reshape(3) / MM_PER_M
    prepared = []
    for obs in observations:
        relative = obs.relative_mm.copy()
        relative[:3, 3] /= MM_PER_M
        prepared.append((relative, obs.K, obs.crop_to_image, _log_softmax(obs.logits, beta)))

    def transform(delta: np.ndarray) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_rotvec(delta[:3]).as_matrix() @ rotation
        pose[:3, 3] = center + delta[3:] - pose[:3, :3] @ centroid
        return pose

    def objective(delta: np.ndarray) -> float:
        pose = transform(delta)
        value = 0.0
        for relative, camera, crop_to_image, logs in prepared:
            xy = _project_to_heatmap(
                points, relative @ pose, camera, crop_to_image, crop_size, logs.shape[-1]
            )
            value -= _bilinear_channel_samples(logs, xy, INVALID).sum()
        return float(value)

    initial = objective(np.zeros(6))
    result = minimize(
        objective,
        np.zeros(6),
        method="Powell",
        bounds=[(-ROTATION_BOUND_RAD, ROTATION_BOUND_RAD)] * 3
        + [(-TRANSLATION_BOUND_M, TRANSLATION_BOUND_M)] * 3,
        options=POWELL_OPTIONS,
    )
    pose = transform(result.x)
    positive = all(
        np.all(((rel @ pose)[:3, :3] @ points.T + (rel @ pose)[:3, 3:4])[2] > DEPTH_EPSILON_M)
        for rel, _, _, _ in prepared
    )
    accepted = bool(np.isfinite(result.fun) and result.fun < initial and positive)
    audit = PowellAudit(
        initial_nll=initial,
        final_nll=float(result.fun),
        accepted=accepted,
        success=bool(result.success),
        iterations=int(result.nit),
        evaluations=int(result.nfev),
        views=len(observations),
    )
    if not accepted:
        return rvec, tvec, audit
    return cv2.Rodrigues(pose[:3, :3])[0], pose[:3, 3:4] * MM_PER_M, audit


def refine_pose(
    rvec: np.ndarray,
    tvec: np.ndarray,
    keypoints_mm: np.ndarray,
    observations: Sequence[HeatmapObservation],
    crop_size: int,
    beta: float,
    settings: RefineSettings | None = None,
) -> tuple[np.ndarray, np.ndarray, RefinementAudit]:
    """Refine an object pose against keypoint heatmaps in one or more views.

    Args:
        rvec: Initial rotation vector, object to reference camera.
        tvec: Initial translation, mm.
        keypoints_mm: ``(K, 3)`` object keypoints in mm, in heatmap channel order.
        observations: Views, reference first (its ``relative_mm`` is the identity).
        crop_size: Crop side in pixels (``cfg.model.crop_size``).
        beta: Softmax temperature (``cfg.loss.keypoint_heatmap_beta``).
        settings: Optimizer and preparation; defaults to L-BFGS-B with NumPy preparation.

    Returns:
        ``(rvec, tvec, audit)``: the refined ``(3, 1)`` rotation vector and translation (mm)
        when accepted, otherwise the inputs unchanged, plus the optimizer's audit.

    Raises:
        ImportError: L-BFGS-B or the parallel preparation without numba (``doper[refine]``).
    """
    settings = settings or RefineSettings()
    if settings.optimizer == "powell":
        return _refine_powell(rvec, tvec, keypoints_mm, observations, crop_size, beta)
    prepared = prepare_observations(
        observations,
        crop_size,
        beta,
        preparation=settings.preparation,
        threads=settings.threads,
    )
    return _refine_lbfgsb(rvec, tvec, keypoints_mm, prepared, len(observations))


def observation_from_detection(
    det: Any,  # noqa: ANN401
    K: np.ndarray,  # noqa: N803
    relative_mm: np.ndarray | None = None,
) -> HeatmapObservation:
    """Build an observation from a ``doper.infer`` detection.

    Args:
        det: A ``Detection`` from ``infer_image(..., return_raw_keypoint_outputs=True)``,
            which carries ``keypoint_heatmap`` and ``crop_to_image`` from the same forward pass.
        K: ``(3, 3)`` intrinsics of the image the detection came from.
        relative_mm: Reference-to-this-camera transform (mm); the identity when omitted.

    Returns:
        The observation.

    Raises:
        ValueError: The detection carries no raw heatmap or crop affine.
    """
    if det.keypoint_heatmap is None or det.crop_to_image is None:
        raise ValueError(
            "detection has no keypoint_heatmap/crop_to_image; run inference with "
            "return_raw_keypoint_outputs=True"
        )
    return HeatmapObservation(
        logits=det.keypoint_heatmap,
        crop_to_image=det.crop_to_image,
        K=K,
        relative_mm=np.eye(4) if relative_mm is None else relative_mm,
    )
