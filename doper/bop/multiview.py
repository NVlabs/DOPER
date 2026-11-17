# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BOP multi-view fusion: per-dataset support views and calibrated fusion of detections.

Each reference test image is fused with a few calibrated support images of the same object
model. ``support_groups`` picks them per dataset (``MultiviewSpec``):

* ``nearest``: the ``nearest_k`` nearest camera centers, ties broken by image id. T-LESS
  searches among the same scene's test targets (``scope: test_targets``); HOPEv2 among all
  frames of the scene (``scope: scene``) and only in scenes 41-47 (``scenes``); every other
  reference stays single-view.
* ``streams``: every synchronized stream at the reference timestamp, in ``streams`` order
  (HOT3D; ``doper.bop.hot3d_staging.stage_hot3d_support`` stages them).

``fuse_reference`` then fuses one reference image's detections with its supports on top of
``doper.multiview`` (association and joint PnP). Per reference detection that has a
single-view pose, in detection order:

1. Transform the original pose into each support camera (``W_support @ inv(W_ref)``). With
   ``skip_nonpositive_support_depth`` a support whose transformed depth is not positive is
   skipped (HOPEv2, HOT3D).
2. Keep the unused support detections whose median distance between their keypoints and the
   projected object keypoints (every keypoint, any confidence) is below ``prefilter_px``.
3. Pick the best of them with ``associate_detection_pairs``; if it passes the gate, mark it used
   (greedy within the reference image) and add its keypoints.
4. With at least one match, re-solve ``joint_pnp`` from the original pose over every matched
   view and accept it iff every per-view median error is finite and at most
   ``gate.max_reprojection_px`` and its depth is positive. Otherwise the original pose stays.

Lengths are millimeters (keypoints, camera translations, gate depths). The camera
``cam_R_w2c``/``cam_t_w2c`` fields are world-to-camera for T-LESS (``w2c``); HOPEv2 and the
staged HOT3D views store camera-to-world under those names (``c2w_named_w2c``), so they are
inverted. Detections cross a float64 boundary on input; a pose that is missing or not finite
counts as no pose, as in the evaluated detection caches.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import cv2
import numpy as np

from ..multiview import (
    CameraObservation,
    CameraView,
    KeypointDetection,
    RigidTransform,
    associate_detection_pairs,
    joint_pnp,
)
from ..recipes.schema import (
    CONVENTIONS,
    DEFAULT_GATE,
    NEAREST_K,
    PREFILTER_PX,
    SUPPORT_MODES,
    SUPPORT_SCOPES,
    CameraConvention,
    MultiviewSpec,
    SupportMode,
    SupportScope,
)

__all__ = [
    "CONVENTIONS",
    "DEFAULT_GATE",
    "FUSION_SEED",
    "NEAREST_K",
    "NO_DISTORTION",
    "PREFILTER_PX",
    "SUPPORT_MODES",
    "SUPPORT_SCOPES",
    "CameraConvention",
    "FusedPose",
    "ImageKey",
    "MultiviewSpec",
    "SupportMode",
    "SupportScope",
    "camera_center",
    "fuse_reference",
    "scene_cameras",
    "support_groups",
    "target_images",
    "world_to_camera",
]

FUSION_SEED = 42
# BOP intrinsics carry no distortion; OpenCV treats an empty array like no coefficients.
NO_DISTORTION = np.empty(0)


class ImageKey(NamedTuple):
    """One image: BOP scene and image id, and the native stream for multi-stream devices.

    Attributes:
        scene_id: BOP scene id.
        im_id: BOP image id (the timestamp of a HOT3D clip frame).
        stream: Native stream id (HOT3D, e.g. ``1201-1``); empty for single-camera datasets.
    """

    scene_id: int
    im_id: int
    stream: str = ""


@dataclass(frozen=True, eq=False)
class FusedPose:
    """The fused pose of one reference detection.

    Attributes:
        reference_index: Index of the detection in the reference image's detection list.
        score: Its detector score (fusion never rescores).
        R: ``(3, 3)`` object rotation in the reference camera.
        t: ``(3,)`` object translation in the reference camera, millimeters.
        matched: Each matched support image and the index of its detection there, in group
            order (the per-detection heatmap lookups use these indices).
        accepted: Whether the joint fit replaced the original single-view pose.
        reprojection_px: Per-view median reprojection errors of the accepted joint fit,
            reference first; empty unless ``accepted``.
        observations: Keypoints the final joint fit used (0 without a fit).
    """

    reference_index: int
    score: float
    R: np.ndarray
    t: np.ndarray
    matched: tuple[tuple[ImageKey, int], ...]
    accepted: bool
    reprojection_px: tuple[float, ...]
    observations: int


@dataclass(frozen=True, eq=False)
class _Posed:
    """A detection at the float64 boundary (``KeypointDetection``)."""

    score: float
    keypoints: np.ndarray
    keypoint_scores: np.ndarray
    rvec: np.ndarray | None
    tvec: np.ndarray | None


def _as_float64(detection: KeypointDetection) -> _Posed:
    rvec, tvec = detection.rvec, detection.tvec
    posed = rvec is not None and tvec is not None
    if posed:
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        posed = bool(np.isfinite(rvec).all() and np.isfinite(tvec).all())
    return _Posed(
        score=float(detection.score),
        keypoints=np.asarray(detection.keypoints, dtype=np.float64),
        keypoint_scores=np.asarray(detection.keypoint_scores, dtype=np.float64),
        rvec=rvec if posed else None,
        tvec=tvec if posed else None,
    )


def world_to_camera(camera: Mapping[str, Any], convention: CameraConvention) -> np.ndarray:
    """The 4x4 world-to-camera matrix of a camera record under ``convention``."""
    matrix = np.eye(4)
    matrix[:3, :3] = np.array(camera["cam_R_w2c"]).reshape(3, 3)
    matrix[:3, 3] = np.array(camera["cam_t_w2c"]).reshape(3)
    return np.linalg.inv(matrix) if convention == "c2w_named_w2c" else matrix


def camera_center(camera: Mapping[str, Any], convention: CameraConvention) -> np.ndarray:
    """The camera center in world coordinates, ``-R^T t`` of its world-to-camera transform."""
    matrix = world_to_camera(camera, convention)
    return -matrix[:3, :3].T @ matrix[:3, 3]


def target_images(rows: Iterable[Mapping[str, Any]]) -> list[ImageKey]:
    """The sorted unique images of BOP test targets; object ids and counts are discarded."""
    return sorted({ImageKey(int(row["scene_id"]), int(row["im_id"])) for row in rows})


def scene_cameras(scene_id: int, scene_camera: Mapping[str, Any]) -> dict[ImageKey, Any]:
    """Key a ``scene_camera.json`` mapping by ``ImageKey``."""
    return {ImageKey(scene_id, int(im)): camera for im, camera in scene_camera.items()}


def _nearest(
    key: ImageKey,
    candidates: Sequence[ImageKey],
    cameras: Mapping[ImageKey, Mapping[str, Any]],
    spec: MultiviewSpec,
) -> tuple[ImageKey, ...]:
    center = camera_center(cameras[key], spec.convention)

    def distance(other: ImageKey) -> tuple[float, int]:
        offset = camera_center(cameras[other], spec.convention) - center
        return float(np.linalg.norm(offset)), other.im_id

    ranked = sorted((c for c in candidates if c != key), key=distance)
    return (key, *ranked[: spec.nearest_k])


def _has_extrinsics(camera: Mapping[str, Any]) -> bool:
    return "cam_R_w2c" in camera and "cam_t_w2c" in camera


def _stream_group(
    key: ImageKey, cameras: Mapping[ImageKey, Mapping[str, Any]], spec: MultiviewSpec
) -> tuple[ImageKey, ...]:
    if key.stream not in spec.streams:
        raise ValueError(f"reference stream {key.stream!r} is not a multiview stream")
    others = (ImageKey(key.scene_id, key.im_id, s) for s in spec.streams if s != key.stream)
    return (key, *(other for other in others if other in cameras))


def support_groups(
    population: Iterable[ImageKey],
    scene_cameras: Mapping[ImageKey, Mapping[str, Any]],
    spec: MultiviewSpec,
) -> dict[ImageKey, tuple[ImageKey, ...]]:
    """Choose each reference image's support views.

    Args:
        population: Reference images (test targets); duplicates are ignored.
        scene_cameras: Camera record (``cam_K``, ``cam_R_w2c``, ``cam_t_w2c``) of every image
            that may serve as a reference or a support.
        spec: The dataset's multi-view policy.

    Returns:
        Reference to its group, the reference first and then its supports in order (nearest
        first, ties by image id; or ``spec.streams`` order). Sorted by reference.

    Raises:
        ValueError: If a multi-view reference or candidate has no extrinsics, or a stream
            reference is not one of ``spec.streams``.
    """
    references = sorted(set(population))
    if spec.support == "streams":
        return {key: _stream_group(key, scene_cameras, spec) for key in references}
    pool = references if spec.scope == "test_targets" else sorted(scene_cameras)
    by_scene: dict[tuple[int, str], list[ImageKey]] = {}
    for key in pool:
        by_scene.setdefault((key.scene_id, key.stream), []).append(key)
    groups: dict[ImageKey, tuple[ImageKey, ...]] = {}
    for key in references:
        if spec.scenes is not None and key.scene_id not in spec.scenes:
            groups[key] = (key,)
            continue
        candidates = by_scene.get((key.scene_id, key.stream), [])
        if not all(_has_extrinsics(scene_cameras[c]) for c in [key, *candidates]):
            raise ValueError(f"scene {key.scene_id}: multiview needs camera extrinsics")
        groups[key] = _nearest(key, candidates, scene_cameras, spec)
    return groups


def _intrinsics(camera: Mapping[str, Any]) -> np.ndarray:
    return np.array(camera["cam_K"]).reshape(3, 3)


@dataclass(frozen=True, eq=False)
class _Support:
    key: ImageKey
    view: CameraView
    transform: RigidTransform
    detections: list[_Posed]


def _supports(
    support_dets: Mapping[ImageKey, Sequence[KeypointDetection]],
    cameras: Mapping[ImageKey, Mapping[str, Any]],
    spec: MultiviewSpec,
) -> list[_Support]:
    keys = list(cameras)
    if list(support_dets) != keys[1:]:
        raise ValueError("cameras must list the reference and then the supports in group order")
    if len(keys) == 1:
        return []  # a single-view group needs no extrinsics
    reference_inverse = np.linalg.inv(world_to_camera(cameras[keys[0]], spec.convention))
    supports = []
    for key in keys[1:]:
        relative = world_to_camera(cameras[key], spec.convention) @ reference_inverse
        supports.append(
            _Support(
                key=key,
                view=CameraView(camera_matrix=_intrinsics(cameras[key]), dist_coeffs=NO_DISTORTION),
                transform=RigidTransform(
                    rvec=cv2.Rodrigues(relative[:3, :3])[0], tvec=relative[:3, 3:4]
                ),
                detections=[_as_float64(d) for d in support_dets[key]],
            )
        )
    return supports


def _observation(
    detection: _Posed, view: CameraView, transform: RigidTransform | None = None
) -> CameraObservation:
    return CameraObservation(
        observations=detection.keypoints,
        scores=detection.keypoint_scores,
        camera_matrix=view.camera_matrix,
        dist_coeffs=view.dist_coeffs,
        rvec_ref2cam=None if transform is None else transform.rvec,
        tvec_ref2cam=None if transform is None else transform.tvec,
    )


def _match(
    detection: _Posed,
    reference_view: CameraView,
    support: _Support,
    used: set[int],
    keypoints_mm: np.ndarray,
    spec: MultiviewSpec,
) -> int | None:
    """Index of the support detection associated with ``detection``, or None."""
    assert detection.rvec is not None and detection.tvec is not None
    rvec, tvec = cv2.composeRT(
        detection.rvec, detection.tvec, support.transform.rvec, support.transform.tvec
    )[:2]
    if spec.skip_nonpositive_support_depth and tvec[2, 0] <= 0:
        return None
    projected = cv2.projectPoints(
        keypoints_mm, rvec, tvec, support.view.camera_matrix, support.view.dist_coeffs
    )[0].reshape(-1, 2)
    candidates = [
        (j, other)
        for j, other in enumerate(support.detections)
        if j not in used
        and np.median(np.linalg.norm(other.keypoints - projected, axis=1)) < spec.prefilter_px
    ]
    if not candidates:
        return None
    pair = associate_detection_pairs(
        [detection],
        [other for _, other in candidates],
        keypoints_mm,
        reference_view,
        support.view,
        support.transform,
        spec.gate,
    )
    if pair is None or not pair.accepted:
        return None
    return candidates[pair.index_b][0]


def _fuse_detection(
    index: int,
    detection: _Posed,
    reference_view: CameraView,
    supports: list[_Support],
    used: dict[ImageKey, set[int]],
    keypoints_mm: np.ndarray,
    spec: MultiviewSpec,
) -> FusedPose:
    assert detection.rvec is not None and detection.tvec is not None
    observations = [_observation(detection, reference_view)]
    matched: list[tuple[ImageKey, int]] = []
    for support in supports:
        j = _match(detection, reference_view, support, used[support.key], keypoints_mm, spec)
        if j is None:
            continue
        used[support.key].add(j)
        observations.append(_observation(support.detections[j], support.view, support.transform))
        matched.append((support.key, j))
    rvec, tvec = detection.rvec, detection.tvec
    accepted, errors, count = False, (), 0
    if len(observations) > 1:
        result = joint_pnp(
            keypoints_mm,
            observations,
            rvec,
            tvec,
            spec.gate.kp_score_thr,
            ransac_px=spec.gate.ransac_px,
            pnp_method=spec.gate.pnp_method,
        )
        if result is not None:
            count = result.n_obs
            per_view = result.per_cam_reproj_px
            if (
                np.isfinite(per_view).all()
                and max(per_view) <= spec.gate.max_reprojection_px
                and result.tvec[2, 0] > 0
            ):
                rvec, tvec = result.rvec, result.tvec
                errors = tuple(float(e) for e in per_view)
                accepted = True
    return FusedPose(
        reference_index=index,
        score=detection.score,
        R=cv2.Rodrigues(rvec)[0],
        t=np.asarray(tvec, dtype=np.float64).ravel(),
        matched=tuple(matched),
        accepted=accepted,
        reprojection_px=errors,
        observations=int(count),
    )


def fuse_reference(
    reference_dets: Sequence[KeypointDetection],
    support_dets: Mapping[ImageKey, Sequence[KeypointDetection]],
    cameras: Mapping[ImageKey, Mapping[str, Any]],
    keypoints_mm: np.ndarray,
    spec: MultiviewSpec,
) -> list[FusedPose]:
    """Fuse one reference image's detections of one object with its support views.

    Seeds OpenCV's RNG with ``FUSION_SEED`` first, so every reference image is reproducible
    on its own.

    Args:
        reference_dets: The reference image's detections, best first (``infer_image`` order).
        support_dets: Each support image's detections, in group order.
        cameras: Camera records of the group in group order, the reference first (as
            ``support_groups`` returns the group); ``cam_K`` and the w2c-named extrinsics.
        keypoints_mm: ``(K, 3)`` object keypoints in millimeters.
        spec: The dataset's multi-view policy.

    Returns:
        One fused pose per reference detection with a single-view pose, in detection order.

    Raises:
        ValueError: If ``cameras`` is not the reference followed by ``support_dets``' keys.
    """
    cv2.setRNGSeed(FUSION_SEED)
    keypoints = np.asarray(keypoints_mm, dtype=np.float64)
    supports = _supports(support_dets, cameras, spec)
    reference = next(iter(cameras.values()))
    reference_view = CameraView(camera_matrix=_intrinsics(reference), dist_coeffs=NO_DISTORTION)
    used: dict[ImageKey, set[int]] = {support.key: set() for support in supports}
    fused = []
    for index, raw in enumerate(reference_dets):
        detection = _as_float64(raw)
        if detection.rvec is None:
            continue
        fused.append(
            _fuse_detection(index, detection, reference_view, supports, used, keypoints, spec)
        )
    return fused
