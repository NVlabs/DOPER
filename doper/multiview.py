# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-camera helpers: rigid-transform math, joint multiview PnP and detection association.

A rig of two rigidly mounted cameras gives each object one pose per camera, related by a
constant reference-to-camera transform. ``joint_pnp`` solves a single pose that reprojects in
every camera at once, and ``associate_detection_pairs`` uses it to pick which detection in
camera A and which in camera B show the same physical instance.

The geometry here is torch-free; only ``infer_all`` imports the model code, on demand.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

import cv2
import numpy as np
from scipy.optimize import least_squares

from .association_gate import AssociationGate as AssociationGate
from .pnp import PnpMethod, pnp_ransac

if TYPE_CHECKING:
    import torch

    from .config import DoperConfig
    from .infer import Detection
    from .model import DoperNet

_MIN_PNP_POINTS = 4
# EPnP RANSAC iterations of the paper's multi-view recipes (``pnp_method: epnp``).
_EPNP_ITERATIONS_SINGLE_VIEW = 100
_EPNP_ITERATIONS_STRICT = 200
_MIN_JOINT_OBSERVATIONS = 6
_JOINT_MAX_NFEV = 100
_DEFAULT_INITIAL_TVEC = (0.0, 0.0, 0.3)


class KeypointDetection(Protocol):
    """What association reads from a detection (``doper.infer.Detection`` satisfies it)."""

    @property
    def score(self) -> float:
        """Detection confidence."""
        ...

    @property
    def keypoints(self) -> np.ndarray:
        """``(K, 2)`` keypoints in pixels."""
        ...

    @property
    def keypoint_scores(self) -> np.ndarray:
        """``(K,)`` keypoint confidences."""
        ...

    @property
    def rvec(self) -> np.ndarray | None:
        """Single-view PnP rotation vector, if solved."""
        ...

    @property
    def tvec(self) -> np.ndarray | None:
        """Single-view PnP translation vector, if solved."""
        ...


@dataclass(frozen=True, eq=False)
class CameraView:
    """Intrinsics of one camera.

    Attributes:
        camera_matrix: ``(3, 3)`` pinhole matrix.
        dist_coeffs: Distortion coefficients (an empty array for none).
    """

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray


@dataclass(frozen=True, eq=False)
class RigidTransform:
    """A rigid transform in OpenCV ``(rvec, tvec)`` form.

    Attributes:
        rvec: Rotation vector.
        tvec: Translation vector.
    """

    rvec: np.ndarray
    tvec: np.ndarray


@dataclass(frozen=True)
class InferSettings:
    """Per-frame inference options for ``infer_all``.

    Attributes:
        score_thr: Minimum detection score.
        max_det: Maximum detections kept per frame.
        box_pad: Fractional padding applied to each detected box before the keypoint crop.
        rot_tta: Use 90-degree rotation test-time augmentation (about 4x slower).
        min_depth: Smallest plausible depth in meters, used to reject flipped TTA solutions.
        max_depth: Largest plausible depth in meters, used to reject flipped TTA solutions.
    """

    score_thr: float
    max_det: int
    box_pad: float
    rot_tta: bool
    min_depth: float
    max_depth: float


@dataclass(frozen=True, eq=False)
class CameraObservation:
    """One camera's keypoints for ``joint_pnp``.

    Attributes:
        observations: ``(K, 2)`` detected keypoints in pixels.
        scores: ``(K,)`` keypoint confidences.
        camera_matrix: ``(3, 3)`` intrinsics.
        dist_coeffs: Distortion coefficients.
        rvec_ref2cam: Reference-to-this-camera rotation vector, or None for the reference camera.
        tvec_ref2cam: Reference-to-this-camera translation vector, or None for the reference.
    """

    observations: np.ndarray
    scores: np.ndarray
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rvec_ref2cam: np.ndarray | None = None
    tvec_ref2cam: np.ndarray | None = None


@dataclass(frozen=True, eq=False)
class JointPnPResult:
    """Output of ``joint_pnp``.

    Attributes:
        rvec: ``(3, 1)`` object rotation vector in the reference camera frame.
        tvec: ``(3, 1)`` object translation vector in the reference camera frame.
        per_cam_reproj_px: Per-camera median reprojection error in pixels (NaN for a camera
            with no kept keypoints).
        n_obs: Total keypoints used across cameras.
    """

    rvec: np.ndarray
    tvec: np.ndarray
    per_cam_reproj_px: list[float]
    n_obs: int


DetectionT = TypeVar("DetectionT", bound=KeypointDetection)


@dataclass(frozen=True, eq=False)
class DetectionPair(Generic[DetectionT]):
    """A scored pairing of one detection per camera.

    Attributes:
        detection_a: Detection in camera A (the reference camera).
        detection_b: Detection in camera B.
        index_a: Index of ``detection_a`` in the camera A list.
        index_b: Index of ``detection_b`` in the camera B list.
        rvec_a: Joint pose rotation in camera A.
        tvec_a: Joint pose translation in camera A.
        rvec_b: Joint pose rotation in camera B.
        tvec_b: Joint pose translation in camera B.
        reprojection_a: Median reprojection error of the joint pose in camera A, pixels.
        reprojection_b: Median reprojection error of the joint pose in camera B, pixels.
        observations: Keypoints used by the joint solve.
        accepted: Whether both reprojection errors are within the gate.
    """

    detection_a: DetectionT
    detection_b: DetectionT
    index_a: int
    index_b: int
    rvec_a: np.ndarray
    tvec_a: np.ndarray
    rvec_b: np.ndarray
    tvec_b: np.ndarray
    reprojection_a: float
    reprojection_b: float
    observations: int
    accepted: bool


def invert_rt(rvec: np.ndarray, tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Invert a rigid transform given as ``(rvec, tvec)``.

    Returns:
        ``(rvec_inv, t_inv)``, both ``(3, 1)``.
    """
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    rotation_inv = rotation.T
    t_inv = -rotation_inv @ tvec.reshape(3, 1)
    rvec_inv, _ = cv2.Rodrigues(rotation_inv)
    return rvec_inv, t_inv


def _compose(
    rvec1: np.ndarray, tvec1: np.ndarray, rvec2: np.ndarray, tvec2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compose two rigid transforms (apply 1 then 2) and return ``(rvec, tvec)``."""
    rvec, tvec = cv2.composeRT(rvec1, tvec1, rvec2, tvec2)[:2]
    return rvec, tvec


def single_view_inliers(
    points_3d: np.ndarray,
    observations: np.ndarray,
    mask: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    ransac_px: float,
    pnp_method: PnpMethod = "usac",
) -> np.ndarray:
    """Flag geometric outlier keypoints with a per-view RANSAC PnP.

    Args:
        points_3d: ``(K, 3)`` object keypoints.
        observations: ``(K, 2)`` detected keypoints.
        mask: ``(K,)`` boolean mask of the confident keypoints to test.
        camera_matrix: ``(3, 3)`` intrinsics.
        dist_coeffs: Distortion coefficients.
        ransac_px: RANSAC reprojection threshold in pixels.
        pnp_method: ``usac``, or ``epnp`` for the paper's recipes.

    Returns:
        A boolean mask over all keypoints of the confident points RANSAC kept as inliers;
        ``mask`` itself if RANSAC cannot solve.
    """
    idx = np.where(mask)[0]
    if idx.size < _MIN_PNP_POINTS:
        return mask
    try:
        ok, _rvec, _tvec, inliers = pnp_ransac(
            points_3d[idx],
            observations[idx],
            camera_matrix,
            dist_coeffs,
            ransac_px,
            method=pnp_method,
            epnp_iterations=_EPNP_ITERATIONS_SINGLE_VIEW,
        )
    except cv2.error:
        return mask
    if not ok or inliers is None or len(inliers) < _MIN_PNP_POINTS:
        return mask
    keep = np.zeros_like(mask)
    keep[idx[np.asarray(inliers, dtype=np.intp).ravel()]] = True
    return keep


def _project_in_camera(
    points_3d: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, camera: CameraObservation
) -> np.ndarray:
    """Project reference-frame points through ``camera``, composing its rig transform."""
    if camera.rvec_ref2cam is None or camera.tvec_ref2cam is None:
        rvec_cam, tvec_cam = rvec, tvec
    else:
        rvec_cam, tvec_cam = _compose(rvec, tvec, camera.rvec_ref2cam, camera.tvec_ref2cam)
    projected, _ = cv2.projectPoints(
        points_3d, rvec_cam, tvec_cam, camera.camera_matrix, camera.dist_coeffs
    )
    return projected.reshape(-1, 2)


def joint_pnp(
    points_3d: np.ndarray,
    cameras: list[CameraObservation],
    init_rvec: np.ndarray,
    init_tvec: np.ndarray,
    score_thr: float,
    f_scale: float = 2.0,
    ransac_px: float = 8.0,
    drop_outliers: bool = True,
    loss: str = "huber",
    pnp_method: PnpMethod = "usac",
) -> JointPnPResult | None:
    """Solve one object pose (in the reference camera frame) that satisfies all cameras.

    A flip that fits one view but not the other cannot minimize the stacked residual, so joint
    PnP rejects it by design. Before fusing, each view's keypoints go through a single-view
    RANSAC PnP and geometric outliers are hard-dropped (``drop_outliers``), so a confidently
    wrong keypoint cannot bias the fused pose; ``loss`` then only softly down-weights the
    remaining noise. Residuals are unweighted so ``f_scale`` stays in pixel units.

    Args:
        points_3d: ``(K, 3)`` object keypoints.
        cameras: One observation per camera; the reference camera has no rig transform.
        init_rvec: Initial rotation vector in the reference frame.
        init_tvec: Initial translation vector in the reference frame.
        score_thr: Minimum keypoint confidence to use.
        f_scale: Soft margin of the robust loss, in pixels.
        ransac_px: Single-view RANSAC threshold in pixels.
        drop_outliers: Hard-drop single-view RANSAC outliers before the fused solve.
        loss: ``scipy.optimize.least_squares`` loss name.
        pnp_method: Single-view RANSAC method: ``usac``, or ``epnp`` for the paper's recipes.

    Returns:
        The fused pose, or None if fewer than six confident correspondences remain.
    """
    masks = [camera.scores >= score_thr for camera in cameras]
    if drop_outliers:
        masks = [
            single_view_inliers(
                points_3d, c.observations, m, c.camera_matrix, c.dist_coeffs, ransac_px, pnp_method
            )
            for c, m in zip(cameras, masks, strict=True)
        ]
    n_obs = int(sum(int(m.sum()) for m in masks))
    if n_obs < _MIN_JOINT_OBSERVATIONS:
        return None

    def residuals(x: np.ndarray) -> np.ndarray:
        rvec = x[:3].reshape(3, 1)
        tvec = x[3:].reshape(3, 1)
        res: list[np.ndarray] = []
        for camera, m in zip(cameras, masks, strict=True):
            if not m.any():
                continue
            proj = _project_in_camera(points_3d[m], rvec, tvec, camera)
            res.append((proj - camera.observations[m]).ravel())
        return np.concatenate(res) if res else np.zeros(1)

    x0 = np.concatenate([np.asarray(init_rvec).ravel(), np.asarray(init_tvec).ravel()])
    sol = least_squares(residuals, x0, loss=loss, f_scale=f_scale, max_nfev=_JOINT_MAX_NFEV)
    rvec = sol.x[:3].reshape(3, 1)
    tvec = sol.x[3:].reshape(3, 1)
    per_cam_reproj: list[float] = []
    for camera, m in zip(cameras, masks, strict=True):
        if not m.any():
            per_cam_reproj.append(float("nan"))
            continue
        proj = _project_in_camera(points_3d[m], rvec, tvec, camera)
        per_cam_reproj.append(
            float(np.median(np.linalg.norm(proj - camera.observations[m], axis=1)))
        )
    return JointPnPResult(rvec=rvec, tvec=tvec, per_cam_reproj_px=per_cam_reproj, n_obs=n_obs)


def infer_all(
    model: DoperNet,
    image: np.ndarray,
    cfg: DoperConfig,
    device: str | torch.device,
    settings: InferSettings,
    view: CameraView,
    keypoints_3d: np.ndarray,
) -> list[Detection]:
    """Return all detections in ``image``, with PnP poses.

    Args:
        model: Loaded model.
        image: BGR image.
        cfg: Model config.
        device: Torch device.
        settings: Score, box padding and TTA options.
        view: Camera intrinsics used for PnP.
        keypoints_3d: ``(K, 3)`` object keypoints.

    Returns:
        Detections, best-first as ``infer_image`` orders them.
    """
    from .infer import infer_image, infer_image_tta

    if settings.rot_tta:
        return infer_image_tta(
            model,
            image,
            cfg,
            str(device),
            score_thr=settings.score_thr,
            max_det=settings.max_det,
            keypoints_3d=keypoints_3d,
            camera_matrix=view.camera_matrix,
            dist_coeffs=view.dist_coeffs,
            box_pad=settings.box_pad,
            depth_range=(settings.min_depth, settings.max_depth),
        )
    return infer_image(
        model,
        image,
        cfg,
        str(device),
        score_thr=settings.score_thr,
        max_det=settings.max_det,
        keypoints_3d=keypoints_3d,
        camera_matrix=view.camera_matrix,
        dist_coeffs=view.dist_coeffs,
        box_pad=settings.box_pad,
    )


def _initial_pose(
    detection_a: KeypointDetection,
    detection_b: KeypointDetection,
    transform_a_to_b: RigidTransform,
) -> tuple[np.ndarray, np.ndarray]:
    """Initial joint pose in camera A from whichever detection already has a PnP pose."""
    if detection_a.rvec is not None:
        return (
            np.asarray(detection_a.rvec, dtype=np.float64).reshape(3, 1),
            np.asarray(detection_a.tvec, dtype=np.float64).reshape(3, 1),
        )
    if detection_b.rvec is not None:
        rvec_b_to_a, tvec_b_to_a = invert_rt(transform_a_to_b.rvec, transform_a_to_b.tvec)
        return _compose(
            np.asarray(detection_b.rvec, dtype=np.float64).reshape(3, 1),
            np.asarray(detection_b.tvec, dtype=np.float64).reshape(3, 1),
            rvec_b_to_a,
            tvec_b_to_a,
        )
    return np.zeros((3, 1), dtype=np.float64), np.array(_DEFAULT_INITIAL_TVEC).reshape(3, 1)


def associate_detection_pairs(
    detections_a: Sequence[DetectionT],
    detections_b: Sequence[DetectionT],
    keypoints_3d: np.ndarray,
    view_a: CameraView,
    view_b: CameraView,
    transform_a_to_b: RigidTransform,
    gate: AssociationGate,
) -> DetectionPair[DetectionT] | None:
    """Evaluate every A/B instance pairing and return the best joint-PnP pair.

    Pairs are ranked by worst per-camera reprojection, then summed reprojection, then more
    observations and confident keypoints, then higher detection scores. The best pair is
    returned even if it fails the reprojection gate (``accepted`` is then False).

    Args:
        detections_a: Camera A (reference) detections.
        detections_b: Camera B detections.
        keypoints_3d: ``(K, 3)`` object keypoints.
        view_a: Camera A intrinsics.
        view_b: Camera B intrinsics.
        transform_a_to_b: Rig transform from camera A to camera B.
        gate: Acceptance thresholds.

    Returns:
        The best pair, or None if no pair solves with plausible depth.
    """
    candidates: list[tuple[tuple[float, ...], DetectionPair[DetectionT]]] = []
    for index_a, detection_a in enumerate(detections_a):
        n_confident_a = int((detection_a.keypoint_scores >= gate.kp_score_thr).sum())
        if n_confident_a < gate.min_keypoints:
            continue
        for index_b, detection_b in enumerate(detections_b):
            n_confident_b = int((detection_b.keypoint_scores >= gate.kp_score_thr).sum())
            if n_confident_b < gate.min_keypoints:
                continue
            initial_rvec, initial_tvec = _initial_pose(detection_a, detection_b, transform_a_to_b)
            cameras = [
                CameraObservation(
                    observations=detection_a.keypoints.astype(np.float64),
                    scores=detection_a.keypoint_scores.astype(np.float64),
                    camera_matrix=view_a.camera_matrix,
                    dist_coeffs=view_a.dist_coeffs,
                ),
                CameraObservation(
                    observations=detection_b.keypoints.astype(np.float64),
                    scores=detection_b.keypoint_scores.astype(np.float64),
                    camera_matrix=view_b.camera_matrix,
                    dist_coeffs=view_b.dist_coeffs,
                    rvec_ref2cam=transform_a_to_b.rvec,
                    tvec_ref2cam=transform_a_to_b.tvec,
                ),
            ]
            result = joint_pnp(
                keypoints_3d,
                cameras,
                initial_rvec,
                initial_tvec,
                gate.kp_score_thr,
                ransac_px=gate.ransac_px,
                pnp_method=gate.pnp_method,
            )
            if result is None:
                continue
            reprojection_a, reprojection_b = result.per_cam_reproj_px
            if not np.isfinite([reprojection_a, reprojection_b]).all():
                continue
            rvec_b, tvec_b = _compose(
                result.rvec, result.tvec, transform_a_to_b.rvec, transform_a_to_b.tvec
            )
            depth_a = float(np.asarray(result.tvec).reshape(3)[2])
            depth_b = float(np.asarray(tvec_b).reshape(3)[2])
            if not (
                gate.min_depth <= depth_a <= gate.max_depth
                and gate.min_depth <= depth_b <= gate.max_depth
            ):
                continue
            maximum_reprojection = max(reprojection_a, reprojection_b)
            candidates.append(
                (
                    (
                        maximum_reprojection,
                        reprojection_a + reprojection_b,
                        -result.n_obs,
                        -min(n_confident_a, n_confident_b),
                        -(detection_a.score + detection_b.score),
                    ),
                    DetectionPair(
                        detection_a=detection_a,
                        detection_b=detection_b,
                        index_a=index_a,
                        index_b=index_b,
                        rvec_a=result.rvec,
                        tvec_a=result.tvec,
                        rvec_b=rvec_b,
                        tvec_b=tvec_b,
                        reprojection_a=float(reprojection_a),
                        reprojection_b=float(reprojection_b),
                        observations=int(result.n_obs),
                        accepted=maximum_reprojection <= gate.max_reprojection_px,
                    ),
                )
            )
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def strict_ransac_inlier_count(
    detection: Detection,
    keypoints_3d: np.ndarray,
    view: CameraView,
    keypoint_score_threshold: float,
    ransac_px: float,
    pnp_method: PnpMethod = "usac",
) -> int:
    """Count fixed-threshold PnP RANSAC inliers of ``detection``; failures count as zero.

    ``pnp_method`` is ``usac``, or ``epnp`` for the paper's recipes.
    """
    confident = detection.keypoint_scores >= keypoint_score_threshold
    indices = np.where(confident)[0]
    if indices.size < _MIN_PNP_POINTS:
        return 0
    try:
        ok, _rvec, _tvec, inliers = pnp_ransac(
            keypoints_3d[indices],
            detection.keypoints[indices],
            view.camera_matrix,
            view.dist_coeffs,
            ransac_px,
            method=pnp_method,
            epnp_iterations=_EPNP_ITERATIONS_STRICT,
        )
    except cv2.error:
        return 0
    return int(len(inliers)) if ok and inliers is not None else 0


def select_fallback_detection(
    detection_a: Detection | None,
    detection_b: Detection | None,
    keypoints_3d: np.ndarray,
    view_a: CameraView,
    view_b: CameraView,
    keypoint_score_threshold: float,
    ransac_px: float,
    pnp_method: PnpMethod = "usac",
) -> tuple[str, Detection, int, int] | None:
    """Select a single-view fallback by RANSAC support, then reprojection.

    ``pnp_method`` is the RANSAC method of the support count: ``usac``, or ``epnp`` for the
    paper's recipes.

    Returns:
        ``(camera_label, detection, confident_keypoints, ransac_inliers)`` for the best
        posed detection, or None if neither detection has a pose.
    """
    candidates: list[tuple[tuple[float, ...], tuple[str, Detection, int, int]]] = []
    for label, detection, view in (("A", detection_a, view_a), ("B", detection_b, view_b)):
        if detection is None or detection.rvec is None:
            continue
        confident = int((detection.keypoint_scores >= keypoint_score_threshold).sum())
        inliers = strict_ransac_inlier_count(
            detection, keypoints_3d, view, keypoint_score_threshold, ransac_px, pnp_method
        )
        reprojection = (
            float(detection.reproj_error_px)
            if detection.reproj_error_px is not None
            else float("inf")
        )
        candidates.append(
            (
                (-inliers, reprojection, -confident, -float(detection.score)),
                (label, detection, confident, inliers),
            )
        )
    return min(candidates, key=lambda item: item[0])[1] if candidates else None
