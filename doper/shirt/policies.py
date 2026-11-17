# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SHIRT temporal filters: the PnP UKF, UKF + RTS and keypoint UKF rows of Table 2.

All three run the camera-ephemeris UKF ``doper.filtering.shirt_ephemeris`` with the process
noise ``ANGULAR_ACCEL_STD_RAD_S2`` / ``LINEAR_ACCEL_STD_M_S2`` (1e-7 rad/s^2, 1e-6 m/s^2) and
its other settings at their defaults. That noise was tuned on SHIRT itself (a process-noise
sweep over the trajectories), so the filtered rows are benchmark-tuned.

* :func:`nis_rank` (the ``nis_rank_p99`` policy, "PnP UKF"): frame 0 initializes from
  its raw candidate; every later frame scores each pose candidate by its 6-D normalized
  innovation squared (NIS) against the UKF prior and updates with the minimum-NIS candidate
  among those with NIS at most the gate (chi-square 6-DoF p99), or predicts only when none
  passes.
* :func:`rts` ("UKF + RTS", offline): one unscented RTS pass over a causal run.
* :func:`keypoint_ukf` ("Keypoint UKF"): frame 0 initializes from the raw candidate; every later
  frame with a detection updates once from the heatmap landmarks whose marginal 2-D NIS passes
  the gate (chi-square 2-DoF p99), predicting only when none passes or nothing was detected.
  The landmarks and their covariances come from the heatmap cache that
  :func:`cache_heatmap_measurements` writes (``heatmap_measurements.npz``).

The filters read the candidates' and landmarks' poses only: ground truth never enters a
decision, and :func:`pose_metrics` scores the filtered poses afterwards. Table 2 reports the
means over **every** frame (2,371 per trajectory) of the geodesic rotation error (degrees) and
the Euclidean translation error (meters in the summaries, centimeters in the table); these are
not the second-orbit metrics (:func:`paper_style_components`).

The frame index is the time base (``index * metadata.pSim.cam_step``), so the filters refuse
records whose indices are not exactly ``0, 1, ..., N - 1``. A later frame whose raw PnP failed
has no candidates (``doper.shirt.candidates``): the PnP filters predict through it and count
it (``empty_candidate_frames``); a frame 0 without candidates is refused.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..speedplus.definitions import LayoutPins
from ..speedplus.metrics import pose_score
from ..speedplus.pose import quaternion_to_object_camera
from . import PAPER_DOMAIN, data
from .inputs import GROUND_TRUTH_NORMALIZATION, RECORDS_NAME, SUMMARY_NAME, load_records

CHI2_6DOF: dict[str, float] = {
    "p90": 10.644640675668422,
    "p95": 12.591587243743977,
    "p99": 16.811893829770927,
    "p999": 22.457744484825323,
}
CHI2_2DOF_P99 = 9.21034037197618
# SHIRT-tuned process noise of every Table 2 filter (benchmark-tuned; see the module docstring).
ANGULAR_ACCEL_STD_RAD_S2 = 1.0e-7
LINEAR_ACCEL_STD_M_S2 = 1.0e-6
RTS_PASSES = 1
COVARIANCE_SCALE = 1.0
COVARIANCE_FLOOR_PX = 1.0
# Reporting threshold of the flip diagnostics (degrees from the ground truth).
FLIP_ROTATION_DEG = 30.0
GROSS_ROTATION_DEG = (30.0, 60.0, 120.0, 150.0)
PRACTICAL = (5.0, 0.02)
LOOSE = (10.0, 0.05)
NIS_RANK = "pnp-ukf"
NIS_RANK_RTS = "pnp-ukf-rts"
KEYPOINT_UKF = "keypoint-ukf"
METHODS = (NIS_RANK, NIS_RANK_RTS, KEYPOINT_UKF)
CACHE_NAME = "heatmap_measurements.npz"
CACHE_KEYS = ("filenames", "points", "covariances", "confidence", "boxes", "detection_scores",
              "available")  # fmt: skip
# The identity :func:`cache_heatmap_measurements` records in every cache it writes.
CACHE_IDENTITY_KEYS = ("checkpoint_sha256", "layout_sha256", "layout_points_sha256")
# The score definition of every filter summary: the paper SPEED score, no deadband.
SCORE_DEFINITION = "paper"
INITIALIZATION = "pnp_initialization"
POSE_UPDATE = "pose"
KEYPOINT_UPDATE = "direct_keypoint"
PREDICTION_ONLY = "prediction_only"
NO_CANDIDATES = "prediction_only_no_candidates"
NO_DETECTION = "prediction_only_no_detection"

Pose = tuple[np.ndarray, np.ndarray]


# --- metrics and summaries ------------------------------------------------------------------------


def pose_metrics(
    rvec: np.ndarray,
    tvec: np.ndarray,
    ground_truth_quaternion: np.ndarray,
    ground_truth_translation: np.ndarray,
) -> dict[str, Any]:
    """Errors of one camera-frame pose against the SHIRT ground truth.

    Args:
        rvec: Estimated object-to-camera Rodrigues vector.
        tvec: Estimated object-to-camera translation, meters.
        ground_truth_quaternion: Scalar-first ``q_vbs2tango_true`` (normalized in float64).
        ground_truth_translation: ``r_Vo2To_vbs_true``, meters.

    Returns:
        ``rvec``, ``tvec``, the geodesic ``rotation_deg``, ``translation_m`` (Euclidean),
        ``translation_relative``, the raw SPEED ``pose_score`` and the practical (5 deg, 2 %) and
        loose (10 deg, 5 %) successes.
    """
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    truth = quaternion_to_object_camera(
        np.asarray(ground_truth_quaternion, dtype=np.float64),
        normalization=GROUND_TRUTH_NORMALIZATION,
    )
    cosine = float(np.clip((np.trace(rotation @ truth.T) - 1.0) / 2.0, -1.0, 1.0))
    rotation_deg = math.degrees(math.acos(cosine))
    true_t = np.asarray(ground_truth_translation, dtype=np.float64)
    translation_m = float(np.linalg.norm(np.asarray(tvec).reshape(3) - true_t))
    relative = translation_m / float(np.linalg.norm(true_t))
    return {
        "rvec": np.asarray(rvec).reshape(3).astype(float).tolist(),
        "tvec": np.asarray(tvec).reshape(3).astype(float).tolist(),
        "rotation_deg": rotation_deg,
        "translation_m": translation_m,
        "translation_relative": relative,
        "pose_score": pose_score(math.radians(rotation_deg), relative, definition=SCORE_DEFINITION),
        "practical_success": rotation_deg < PRACTICAL[0] and relative < PRACTICAL[1],
        "loose_success": rotation_deg < LOOSE[0] and relative < LOOSE[1],
    }


def _distribution(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def summarize(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    """Distributions, recalls and gross-error counts of the ``key`` pose metrics of records."""
    poses = [record[key] for record in records]
    rotation = np.asarray([float(pose["rotation_deg"]) for pose in poses])
    summary: dict[str, Any] = {
        "count": len(poses),
        "rotation_deg": _distribution(rotation),
        "translation_relative": _distribution([float(p["translation_relative"]) for p in poses]),
        "pose_score": _distribution([float(p["pose_score"]) for p in poses]),
        "practical_recall_5deg_2pct": float(np.mean([bool(p["practical_success"]) for p in poses])),
        "loose_recall_10deg_5pct": float(np.mean([bool(p["loose_success"]) for p in poses])),
    }
    for threshold in GROSS_ROTATION_DEG:
        summary[f"rotation_gt{threshold:.0f}_count"] = int(np.sum(rotation > threshold))
    return summary


def summarize_pose_values(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The Table 2 summary of pose metrics: count, rotation and translation means and more.

    ``rotation_deg_mean`` and ``translation_m_mean`` over every frame are the Table 2 numbers.
    """
    rotation = np.asarray([value["rotation_deg"] for value in values])
    translation = np.asarray([value["translation_m"] for value in values])
    relative = np.asarray([value["translation_relative"] for value in values])
    return {
        "count": len(values),
        "rotation_deg_mean": float(np.mean(rotation)),
        "rotation_deg_median": float(np.median(rotation)),
        "rotation_deg_p95": float(np.percentile(rotation, 95)),
        "rotation_deg_max": float(np.max(rotation)),
        "translation_m_mean": float(np.mean(translation)),
        "translation_relative_mean": float(np.mean(relative)),
        "pose_score_mean": float(np.mean([value["pose_score"] for value in values])),
        "practical_recall_5deg_2pct": float(np.mean([v["practical_success"] for v in values])),
        "loose_recall_10deg_5pct": float(np.mean([value["loose_success"] for value in values])),
        "rotation_gt30_count": int(np.sum(rotation > 30.0)),
        "rotation_gt60_count": int(np.sum(rotation > 60.0)),
        "rotation_gt150_count": int(np.sum(rotation > 150.0)),
    }


def _mean_std(values: Sequence[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    return [float(np.mean(array)), float(np.std(array))]


def method_summary(records: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    """:func:`summarize` plus the mean and population std of rotation, translation and score."""
    poses = [record[method] for record in records]
    return {
        **summarize(records, method),
        "rotation_deg_mean_std": _mean_std([pose["rotation_deg"] for pose in poses]),
        "translation_m_mean_std": _mean_std([pose["translation_m"] for pose in poses]),
        "pose_score_mean_std": _mean_std([pose["pose_score"] for pose in poses]),
    }


def _axis_summary(values: np.ndarray) -> dict[str, list[float]]:
    return {
        axis: [float(np.mean(values[:, index])), float(np.std(values[:, index]))]
        for index, axis in enumerate(("x", "y", "z"))
    }


def paper_style_components(records: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    """Second-orbit camera-axis errors of the ``method`` poses (``records[len // 2:]``).

    Signed and absolute translation and rotation-vector components (degrees) plus the Euclidean
    translation and geodesic rotation mean and std. This is the comparison with Park and D'Amico's
    second-orbit figure; Table 2 does not use it. Each record needs ``ground_truth_rvec`` and
    ``ground_truth_translation``.
    """
    start = len(records) // 2
    second_orbit = records[start:]
    translation_errors, rotation_errors = [], []
    for record in second_orbit:
        estimate = record[method]
        translation_errors.append(
            np.asarray(estimate["tvec"], dtype=np.float64)
            - np.asarray(record["ground_truth_translation"], dtype=np.float64)
        )
        predicted, _ = cv2.Rodrigues(np.asarray(estimate["rvec"], dtype=np.float64).reshape(3, 1))
        truth, _ = cv2.Rodrigues(
            np.asarray(record["ground_truth_rvec"], dtype=np.float64).reshape(3, 1)
        )
        error_vector, _ = cv2.Rodrigues(predicted @ truth.T)  # pyright: ignore[reportCallIssue, reportArgumentType]
        rotation_errors.append(np.degrees(error_vector.reshape(3)))
    translation = np.asarray(translation_errors)
    rotation = np.asarray(rotation_errors)
    norms = np.linalg.norm(translation, axis=1)
    return {
        "frame_count": len(second_orbit),
        "starts_at_frame": start,
        "translation_camera_m_signed_mean_std": _axis_summary(translation),
        "translation_camera_m_absolute_mean_std": _axis_summary(np.abs(translation)),
        "rotation_vector_camera_deg_signed_mean_std": _axis_summary(rotation),
        "rotation_vector_camera_deg_absolute_mean_std": _axis_summary(np.abs(rotation)),
        "euclidean_translation_m_mean_std": [float(np.mean(norms)), float(np.std(norms))],
        "geodesic_rotation_deg_mean_std": _mean_std(
            [record[method]["rotation_deg"] for record in second_orbit]
        ),
    }


def _finite_summary(values: Sequence[float]) -> dict[str, Any]:
    finite = np.asarray([value for value in values if np.isfinite(value)])
    if not len(finite):
        return {"count": 0}
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def _nan_distribution(values: Sequence[float]) -> dict[str, Any]:
    """Count, mean, min, p5, median, p95 and max of the finite values; None without any."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, **dict.fromkeys(("mean", "min", "p5", "median", "p95", "max"))}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "min": float(np.min(array)),
        "p5": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


# --- the filters --------------------------------------------------------------------------------


@dataclass
class FilterRun:
    """One trajectory through a SHIRT filter.

    Attributes:
        method: ``pnp-ukf``, ``pnp-ukf-rts`` or ``keypoint-ukf``.
        records: The candidate records the filter read, in frame order.
        poses: One filtered (or smoothed) camera-frame ``(rvec, tvec)`` per frame.
        frames: One decision record per frame: ``index``, ``update_kind`` and the method's
            selection (``selected_index`` and ``candidate_nis``) or landmark gate
            (``accepted``, ``keypoint_nis``) diagnostics.
        smoother: The ``ShirtEphemerisPoseRTSSmoother`` that produced the poses.
        settings: The filter settings, for provenance.
        causal_poses: For a smoothed run, the causal poses it smoothed.
    """

    method: str
    records: list[dict[str, Any]]
    poses: list[Pose]
    frames: list[dict[str, Any]]
    smoother: Any
    settings: dict[str, Any]
    causal_poses: list[Pose] | None = field(default=None)


def check_frame_indices(records: Sequence[Mapping[str, Any]]) -> None:
    """Refuse records whose frame indices are not exactly ``0, 1, ..., N - 1``.

    The filters' time base is the frame index, so a gap, a duplicate or an unsorted index would
    silently shift every later timestamp.

    Raises:
        ValueError: Without records, or naming the first index out of place.
    """
    if not records:
        raise ValueError("a SHIRT filter needs at least one frame")
    for position, record in enumerate(records):
        index = record["index"]
        if index != position:
            raise ValueError(
                f"SHIRT frame indices must be contiguous from 0 in order: position {position} "
                f"holds index {index} (missing, duplicate or unsorted frames)"
            )


def _new_filter(metadata: Mapping[str, Any] | Path) -> Any:  # noqa: ANN401
    from ..filtering.shirt_ephemeris import ShirtEphemerisPoseRTSSmoother

    source = metadata if isinstance(metadata, Path) else dict(metadata)
    return ShirtEphemerisPoseRTSSmoother(
        source,
        angular_accel_std_rad_s2=ANGULAR_ACCEL_STD_RAD_S2,
        linear_accel_std_m_s2=LINEAR_ACCEL_STD_M_S2,
    )


def _pose_of(candidate: Mapping[str, Any]) -> Pose:
    return (
        np.asarray(candidate["rvec"], dtype=np.float64),
        np.asarray(candidate["tvec"], dtype=np.float64),
    )


def _initial_candidate(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not records[0]["candidates"]:
        raise ValueError("no initial pose for frame 0: its candidate list is empty")
    return records[0]["candidates"][0]


def _settings(**extra: Any) -> dict[str, Any]:  # noqa: ANN401
    return {
        "angular_accel_std_rad_s2": ANGULAR_ACCEL_STD_RAD_S2,
        "linear_accel_std_m_s2": LINEAR_ACCEL_STD_M_S2,
        "process_noise": "SHIRT-tuned (benchmark-tuned process-noise sweep)",
        **extra,
    }


def _nis_rank_step(smoother: Any, candidates: Sequence[Mapping[str, Any]], index: int,  # noqa: ANN401
                   gate: float) -> tuple[Pose, dict[str, Any]]:  # fmt: skip
    """One frame after frame 0: the minimum-NIS candidate inside the gate, or a prediction."""
    if not candidates:
        frame = {"update_kind": NO_CANDIDATES, "selected_index": None, "candidate_nis": []}
        return smoother.predict_only(index), frame
    statistics = smoother.preview_pose_innovations([_pose_of(c) for c in candidates], index)
    nis = [float(value["nis"]) for value in statistics]
    plausible = [i for i, value in enumerate(nis) if value <= gate]
    selected = min(plausible, key=lambda i: nis[i]) if plausible else None
    frame = {"selected_index": selected, "candidate_nis": nis}
    if selected is None:
        return smoother.predict_only(index), {**frame, "update_kind": PREDICTION_ONLY}
    pose = smoother.update(*_pose_of(candidates[selected]), index)
    return pose, {**frame, "update_kind": POSE_UPDATE}


def nis_rank(
    candidate_records: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any] | Path,
    *,
    gate: float = CHI2_6DOF["p99"],
) -> FilterRun:
    """The PnP UKF: minimum-NIS pose candidates inside a chi-square gate (``nis_rank_p99``).

    Args:
        candidate_records: One record per frame, indices ``0..N-1`` in order, each with its
            ``candidates`` (raw first); a later frame may have none.
        metadata: The trajectory's parsed ``metadata.json``, or its path.
        gate: 6-D NIS gate; Table 2 uses chi-square 6-DoF p99.

    Returns:
        The causal run.

    Raises:
        ValueError: For non-contiguous frame indices or a frame 0 without candidates.
    """
    check_frame_indices(candidate_records)
    initial = _initial_candidate(candidate_records)
    smoother = _new_filter(metadata)
    poses = [smoother.update(*_pose_of(initial), 0)]
    frames = [{"index": 0, "update_kind": INITIALIZATION, "selected_index": 0,
               "candidate_nis": [math.nan] * len(candidate_records[0]["candidates"])}]  # fmt: skip
    for record in candidate_records[1:]:
        index = int(record["index"])
        pose, frame = _nis_rank_step(smoother, record["candidates"], index, gate)
        poses.append(pose)
        frames.append({"index": index, **frame})
    return FilterRun(
        NIS_RANK, [dict(r) for r in candidate_records], poses, frames, smoother,
        _settings(nis_gate=gate, nis_gate_label=_gate_label(gate)),
    )  # fmt: skip


def _gate_label(gate: float) -> str | None:
    return next((label for label, value in CHI2_6DOF.items() if value == gate), None)


def rts(run: FilterRun, *, num_passes: int = RTS_PASSES) -> FilterRun:
    """Smooth a causal run offline with unscented RTS (Table 2: one pass).

    More than one pass is refused after prediction-only frames (the filter cannot replay them).

    Returns:
        A run with the smoothed poses; ``causal_poses`` keeps the causal ones.
    """
    smoothed = run.smoother.apply_rts(num_passes=num_passes)
    return replace(
        run,
        method=NIS_RANK_RTS if run.method == NIS_RANK else f"{run.method}-rts",
        poses=smoothed,
        causal_poses=list(run.poses),
        settings={**run.settings, "rts_passes": num_passes},
    )


def _check_cache(cache: Mapping[str, np.ndarray], records: Sequence[Mapping[str, Any]],
                 floor_px: float) -> None:  # fmt: skip
    """Refuse a heatmap cache of other frames, or one built with another covariance floor."""
    filenames = [str(name) for name in np.asarray(cache["filenames"]).tolist()]
    expected = [str(record["filename"]) for record in records]
    if filenames != expected:
        raise ValueError(
            f"the heatmap cache holds {len(filenames)} frames that do not match the "
            f"{len(expected)} candidate records frame for frame"
        )
    if "covariance_floor_image_px" in cache:
        recorded = float(np.asarray(cache["covariance_floor_image_px"]))
        if recorded != floor_px:
            raise ValueError(
                f"the heatmap cache was built with covariance floor {recorded} px, not {floor_px}"
            )


def _empty_keypoint_frame(count: int, kind: str, updated: bool) -> dict[str, Any]:
    return {
        "update_kind": kind,
        "accepted": np.zeros(count, dtype=bool),
        "keypoint_nis": np.full(count, np.nan),
        "updated": updated,
    }


def keypoint_ukf(
    candidate_records: Sequence[Mapping[str, Any]],
    heatmap_cache: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any] | Path,
    points_3d: np.ndarray,
    camera: Any,  # noqa: ANN401 - doper.speedplus.geometry.Camera (K, dist)
    *,
    landmark_gate: float = CHI2_2DOF_P99,
    covariance_scale: float = COVARIANCE_SCALE,
    floor_px: float = COVARIANCE_FLOOR_PX,
) -> FilterRun:
    """The keypoint UKF: one joint update per frame from the gated heatmap landmarks.

    Frame 0 initializes from its raw candidate. Each later frame with a detection propagates
    the UKF sigma points through the camera projection, gates every landmark on its marginal
    2-D NIS and updates once with the accepted subset (prediction only if none is accepted);
    a frame without a detection is prediction only.

    Args:
        candidate_records: One record per frame, indices ``0..N-1`` in order; only frame 0's
            raw candidate is read.
        heatmap_cache: ``heatmap_measurements.npz`` contents for the same frames
            (:func:`cache_heatmap_measurements`).
        metadata: The trajectory's parsed ``metadata.json``, or its path.
        points_3d: ``(K, 3)`` body landmarks of the checkpoint's layout.
        camera: Intrinsics ``K`` and distortion ``dist``.
        landmark_gate: Per-landmark 2-D NIS gate; Table 2 uses chi-square 2-DoF p99.
        covariance_scale: Scale of the cached landmark covariances.
        floor_px: The covariance floor the cache must have been built with (checked when the
            cache records it).

    Raises:
        ValueError: For non-contiguous frame indices, a frame 0 without candidates, or a cache
            of other frames or another covariance floor.
    """
    check_frame_indices(candidate_records)
    _check_cache(heatmap_cache, candidate_records, floor_px)
    initial = _initial_candidate(candidate_records)
    points_3d = np.asarray(points_3d, dtype=np.float64)
    smoother = _new_filter(metadata)
    poses = [smoother.update(*_pose_of(initial), 0)]
    frames = [{"index": 0, **_empty_keypoint_frame(len(points_3d), INITIALIZATION, True)}]
    available = np.asarray(heatmap_cache["available"])
    for record in candidate_records[1:]:
        index = int(record["index"])
        if not available[index]:
            poses.append(smoother.predict_only(index))
            frames.append(
                {"index": index, **_empty_keypoint_frame(len(points_3d), NO_DETECTION, False)}
            )
            continue
        pose, diagnostics = smoother.update_keypoints(
            heatmap_cache["points"][index], heatmap_cache["covariances"][index], points_3d,
            camera.K, camera.dist, index, nis_gate=landmark_gate,
            covariance_scale=covariance_scale,
        )  # fmt: skip
        poses.append(pose)
        frames.append({
            "index": index,
            "update_kind": KEYPOINT_UPDATE if diagnostics["updated"] else PREDICTION_ONLY,
            "accepted": diagnostics["accepted"], "keypoint_nis": diagnostics["nis"],
            "updated": diagnostics["updated"],
        })  # fmt: skip
    settings = _settings(landmark_nis_gate_2d=landmark_gate, covariance_scale=covariance_scale,
                         covariance_floor_image_px=floor_px)  # fmt: skip
    return FilterRun(KEYPOINT_UKF, [dict(r) for r in candidate_records], poses, frames, smoother,
                     settings)  # fmt: skip


# --- per-frame outputs and summaries --------------------------------------------------------------


def _ground_truth(record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(record["ground_truth_quaternion"], dtype=np.float64),
        np.asarray(record["ground_truth_translation"], dtype=np.float64),
    )


def frame_metrics(run: FilterRun, poses: Sequence[Pose] | None = None) -> list[dict[str, Any]]:
    """:func:`pose_metrics` of every frame's pose (``run.poses`` by default)."""
    poses = run.poses if poses is None else poses
    return [
        pose_metrics(*pose, *_ground_truth(record))
        for pose, record in zip(poses, run.records, strict=True)
    ]


def _flip(value: float | None) -> bool:
    return value is not None and value > FLIP_ROTATION_DEG


@dataclass
class _NisDiagnostics:
    """The NIS-rank diagnostics, guarded against empty frames."""

    sources: dict[str, int] = field(default_factory=dict)
    rejected: int = 0
    selected_gt30: int = 0
    introduced_gt30: int = 0
    flip_good: int = 0
    flip_rejected: int = 0
    flip_bad: int = 0
    raw_nis_good: list[float] = field(default_factory=list)
    raw_nis_flips: list[float] = field(default_factory=list)
    selected_nis: list[float] = field(default_factory=list)
    minimum_nis: list[float] = field(default_factory=list)

    def add(self, record: Mapping[str, Any], frame: Mapping[str, Any]) -> None:
        nis = frame["candidate_nis"]
        raw_is_flip = _flip(record["raw_rotation_deg"])
        if frame["index"] > 0 and nis:
            self.minimum_nis.append(min(nis))
            if record["raw_rotation_deg"] is not None:
                (self.raw_nis_flips if raw_is_flip else self.raw_nis_good).append(nis[0])
        selected_index = frame["selected_index"]
        if selected_index is None:
            self.rejected += frame["update_kind"] == PREDICTION_ONLY
            self.flip_rejected += raw_is_flip
            return
        selected = record["candidates"][selected_index]
        self.sources[selected["source"]] = self.sources.get(selected["source"], 0) + 1
        selected_is_flip = _flip(selected["rotation_deg"])
        self.selected_gt30 += selected_is_flip
        self.introduced_gt30 += not raw_is_flip and selected_is_flip
        if raw_is_flip:
            self.flip_bad += selected_is_flip
            self.flip_good += not selected_is_flip
        if frame["index"] > 0:
            self.selected_nis.append(nis[selected_index])


def pnp_ukf_outputs(run: FilterRun) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per-frame records and the summary of a :func:`nis_rank` run.

    ``rejected_measurements`` counts the frames whose candidates all failed the gate;
    ``empty_candidate_frames`` lists the frames that had none (raw PnP failed); both predict
    only (``prediction_only_frames``).
    """
    metrics = frame_metrics(run)
    diagnostics = _NisDiagnostics()
    records = []
    for record, frame, metric in zip(run.records, run.frames, metrics, strict=True):
        diagnostics.add(record, frame)
        candidates = record["candidates"]
        selected = frame["selected_index"]
        records.append({
            "index": record["index"], "filename": record["filename"],
            "raw_rotation_deg": record["raw_rotation_deg"],
            "candidate_sources": [c["source"] for c in candidates],
            "candidate_rotation_deg": [c["rotation_deg"] for c in candidates],
            "candidate_nis": frame["candidate_nis"], "selected_index": selected,
            "selected_source": None if selected is None else candidates[selected]["source"],
            "update_kind": frame["update_kind"], "filtered": metric,
        })  # fmt: skip
    empty = [frame["index"] for frame in run.frames if frame["update_kind"] == NO_CANDIDATES]
    summary = {
        **summarize_pose_values(metrics),
        "method": "nis_rank",
        "nis_gate": run.settings["nis_gate"],
        "rejected_measurements": diagnostics.rejected,
        "empty_candidate_frames": empty,
        "prediction_only_frames": sum(f["selected_index"] is None for f in run.frames),
        "selected_sources": diagnostics.sources,
        "selected_candidate_gt30_count": diagnostics.selected_gt30,
        "introduced_candidate_gt30_count": diagnostics.introduced_gt30,
        "raw_flip_count": int(sum(_flip(r["raw_rotation_deg"]) for r in run.records)),
        "raw_flip_selected_good_candidate": diagnostics.flip_good,
        "raw_flip_rejected": diagnostics.flip_rejected,
        "raw_flip_selected_bad_candidate": diagnostics.flip_bad,
        "raw_candidate_nis_on_good_frames": _finite_summary(diagnostics.raw_nis_good),
        "raw_candidate_nis_on_flip_frames": _finite_summary(diagnostics.raw_nis_flips),
        "selected_candidate_nis": _finite_summary(diagnostics.selected_nis),
        "minimum_candidate_nis": _finite_summary(diagnostics.minimum_nis),
    }
    return records, summary


def rts_outputs(run: FilterRun) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per-frame causal and RTS records and the summary of a :func:`rts` run.

    ``summary["rts"]["rotation_deg_mean_std"]`` and ``["translation_m_mean_std"]`` hold the
    Table 2 UKF + RTS row; ``second_orbit`` the second-orbit comparison.

    Raises:
        ValueError: For a run that :func:`rts` did not smooth.
    """
    if run.causal_poses is None:
        raise ValueError("rts_outputs needs a run smoothed by rts()")
    causal = frame_metrics(run, run.causal_poses)
    smoothed = frame_metrics(run)
    records = []
    for record, frame, causal_metric, rts_metric in zip(
        run.records, run.frames, causal, smoothed, strict=True
    ):
        quaternion, _ = _ground_truth(record)
        truth = quaternion_to_object_camera(quaternion, normalization=GROUND_TRUTH_NORMALIZATION)
        selected = frame["selected_index"]
        nis = frame["candidate_nis"]
        records.append({
            "index": record["index"], "filename": record["filename"],
            "ground_truth_quaternion": record["ground_truth_quaternion"],
            "ground_truth_rvec": cv2.Rodrigues(truth)[0].reshape(3).tolist(),
            "ground_truth_translation": record["ground_truth_translation"],
            "selected_index": selected,
            "selected_source": None if selected is None else
            record["candidates"][selected]["source"],
            "selected_nis": None if selected is None else nis[selected],
            "update_kind": frame["update_kind"],
            "causal": causal_metric, "rts": rts_metric,
        })  # fmt: skip
    diagnostics = _NisDiagnostics()
    for record, frame in zip(run.records, run.frames, strict=True):
        diagnostics.add(record, frame)
    summary = {
        "gate_label": run.settings["nis_gate_label"],
        "nis_gate": run.settings["nis_gate"],
        "rts_passes": run.settings["rts_passes"],
        "rejected_measurements": diagnostics.rejected,
        "empty_candidate_frames": [
            f["index"] for f in run.frames if f["update_kind"] == NO_CANDIDATES
        ],
        "prediction_only_frames": sum(f["selected_index"] is None for f in run.frames),
        "selected_sources": diagnostics.sources,
        "causal": method_summary(records, "causal"),
        "rts": method_summary(records, "rts"),
        "second_orbit": {m: paper_style_components(records, m) for m in ("causal", "rts")},
    }
    return records, summary


def keypoint_ukf_outputs(
    run: FilterRun, baseline: FilterRun, heatmap_cache: Mapping[str, np.ndarray]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per-frame records and the summary of a keypoint run beside its PnP UKF baseline.

    ``direct_keypoint_ukf`` is the Table 2 keypoint UKF row and ``baseline_minimum_nis_pnp`` the PnP
    UKF row.
    """
    direct, base = frame_metrics(run), frame_metrics(baseline)
    records = []
    for frame, base_frame, record, metric, base_metric in zip(
        run.frames, baseline.frames, run.records, direct, base, strict=True
    ):
        accepted = np.asarray(frame["accepted"], dtype=bool)
        nis = np.asarray(frame["keypoint_nis"], dtype=np.float64)
        finite = nis[np.isfinite(nis)]
        records.append({
            "index": int(record["index"]), "filename": record["filename"],
            "baseline_selected_index": base_frame["selected_index"],
            "baseline_candidate_nis": base_frame["candidate_nis"],
            "direct_update_kind": frame["update_kind"],
            "accepted_keypoint_count": int(np.sum(accepted)),
            "accepted_keypoints": accepted.astype(int).tolist(),
            "keypoint_nis": nis.tolist(),
            "keypoint_nis_finite_mean": float(np.mean(finite)) if len(finite) else None,
            "baseline": base_metric, "direct_keypoint": metric,
        })  # fmt: skip
    available = np.asarray(heatmap_cache["available"])
    covariances = np.asarray(heatmap_cache["covariances"])[available.astype(bool)]
    covariance_std = np.sqrt(np.linalg.eigvalsh(covariances)).reshape(-1)
    accepted_nis: list[float] = []
    rejected_nis: list[float] = []
    for record in records[1:]:
        nis = np.asarray(record["keypoint_nis"], dtype=np.float64)
        accepted = np.asarray(record["accepted_keypoints"], dtype=bool)
        accepted_nis.extend(nis[accepted].tolist())
        rejected_nis.extend(nis[np.isfinite(nis) & ~accepted].tolist())
    summary = {
        "frames": len(records),
        "baseline_minimum_nis_pnp": summarize_pose_values(base),
        "direct_keypoint_ukf": summarize_pose_values(direct),
        "baseline_prediction_only_frames": int(
            np.sum([r["baseline_selected_index"] is None for r in records])
        ),
        "direct_prediction_only_frames": int(
            np.sum([r["direct_update_kind"].startswith(PREDICTION_ONLY) for r in records])
        ),
        "detector_missing_frames_after_initialization": int(np.sum(available[1:] == 0)),
        "accepted_keypoint_count": _nan_distribution(
            [r["accepted_keypoint_count"] for r in records[1:]]
        ),
        "accepted_keypoint_nis": _nan_distribution(accepted_nis),
        "rejected_keypoint_nis": _nan_distribution(rejected_nis),
        "heatmap_covariance_principal_std_image_px": _nan_distribution(covariance_std.tolist()),
    }
    return records, summary


# --- the heatmap measurement cache (keypoint UKF stage 1) --------------------------------------


def load_candidate_records(
    candidates_root: Path, roe: str, domain: str, limit: int = 0
) -> list[dict[str, Any]]:
    """The candidate records of one trajectory, sorted by frame (``doper.shirt.candidates``).

    Raises:
        FileNotFoundError: Without candidate records.
        ValueError: Without records.
    """
    return load_records(Path(candidates_root), roe, domain, limit, kind="candidate")


def load_heatmap_cache(path: Path) -> dict[str, np.ndarray]:
    """Load a ``heatmap_measurements.npz`` (no pickled objects).

    Raises:
        ValueError: When a cache key is missing.
    """
    with np.load(path, allow_pickle=False) as cached:
        cache = {name: cached[name] for name in cached.files}
    missing = [key for key in CACHE_KEYS if key not in cache]
    if missing:
        raise ValueError(f"heatmap cache {path} lacks {missing}")
    return cache


def check_heatmap_cache_identity(
    cache: Mapping[str, np.ndarray],
    path: Path,
    layout: LayoutPins,
    checkpoint_sha256: str | None = None,
) -> bool:
    """Refuse a heatmap cache of another keypoint layout or checkpoint.

    The layout identity is the points hash: a cache whose points hash matches ``layout`` but
    whose layout file SHA-256 differs is a formatting deviation, accepted.

    Args:
        cache: The cache contents.
        path: The cache file, for the messages.
        layout: The identity of the layout the keypoint UKF will project.
        checkpoint_sha256: The checkpoint the candidate records came from, when known.

    Returns:
        Whether the cache's layout is a formatting deviation of ``layout`` (or was built from a
        checkpoint whose layout was one).

    Raises:
        ValueError: For a cache without its identity, or of another layout or checkpoint,
            naming both SHA-256s.
    """
    missing = [key for key in CACHE_IDENTITY_KEYS if key not in cache]
    if missing:
        raise ValueError(
            f"heatmap cache {path} records no {missing}; rebuild it with "
            "cache_heatmap_measurements (`doper evaluate shirt --method keypoint-ukf --force`)"
        )
    points = str(cache["layout_points_sha256"])
    if points != layout.points_sha256:
        raise ValueError(
            f"heatmap cache {path} was built with keypoint layout points {points} (file "
            f"{cache['layout_sha256']}), but layout points {layout.points_sha256} (file "
            f"{layout.sha256}) were given"
        )
    recorded = str(cache["checkpoint_sha256"])
    if checkpoint_sha256 is not None and recorded != checkpoint_sha256:
        raise ValueError(
            f"heatmap cache {path} was built from checkpoint {recorded}, but the candidate "
            f"records come from checkpoint {checkpoint_sha256}"
        )
    built_deviation = bool(cache.get("layout_formatting_deviation", False))
    return built_deviation or str(cache["layout_sha256"]) != layout.sha256


def heatmap_cache_problem(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    layout: LayoutPins,
    checkpoint_sha256: str,
    floor_px: float = COVARIANCE_FLOOR_PX,
) -> str | None:
    """Why a heatmap cache cannot serve these records, layout and checkpoint; None if it can.

    Returns:
        ``"missing"`` without a cache, a description of the first mismatch (frames, layout,
        checkpoint or covariance floor), or None for a reusable cache.
    """
    if not Path(path).is_file():
        return "missing"
    try:
        cache = load_heatmap_cache(Path(path))
        check_heatmap_cache_identity(cache, Path(path), layout, checkpoint_sha256)
        _check_cache(cache, records, floor_px)
    except ValueError as error:
        return str(error)
    return None


def _frame_measurement(
    detection: Any,  # noqa: ANN401 - doper.infer.Detection or None
    crop_size: int,
    beta: float,
    floor_px: float,
) -> dict[str, Any] | None:
    from ..filtering.measurements import heatmap_keypoint_measurements

    if detection is None:
        return None
    if detection.keypoint_heatmap is None or detection.crop_to_image is None:
        raise RuntimeError("the heatmap checkpoint did not return raw heatmaps")
    measurement = heatmap_keypoint_measurements(
        detection.keypoint_heatmap, detection.crop_to_image, crop_size, beta=beta,
        covariance_floor_image_px=floor_px,
    )  # fmt: skip
    return {**measurement, "box": detection.bbox_xyxy, "score": detection.score}


def _empty_cache(frames: int, keypoints: int) -> dict[str, np.ndarray]:
    return {
        "points": np.full((frames, keypoints, 2), np.nan, dtype=np.float32),
        "covariances": np.full((frames, keypoints, 2, 2), np.nan, dtype=np.float32),
        "confidence": np.full((frames, keypoints), np.nan, dtype=np.float32),
        "boxes": np.full((frames, 4), np.nan, dtype=np.float32),
        "detection_scores": np.full(frames, np.nan, dtype=np.float32),
        "available": np.zeros(frames, dtype=np.uint8),
    }


def _measure_trajectory(
    model: Any,  # noqa: ANN401
    cfg: Any,  # noqa: ANN401
    shirt_root: Path,
    roe: str,
    domain: str,
    records: Sequence[Mapping[str, Any]],
    device: str,
    floor_px: float,
) -> dict[str, np.ndarray]:
    from .. import infer
    from .inputs import DETECTION_SCORE_THRESHOLD, MAX_DETECTIONS, read_image

    cache = _empty_cache(len(records), int(cfg.model.num_keypoints))
    for index, record in enumerate(records):
        image = read_image(shirt_root, roe, domain, record["filename"])
        detections = infer.infer_image(
            model, image, cfg, device, score_thr=DETECTION_SCORE_THRESHOLD,
            max_det=MAX_DETECTIONS, return_raw_keypoint_outputs=True,
        )  # fmt: skip
        measured = _frame_measurement(
            detections[0] if detections else None, int(cfg.model.crop_size),
            float(cfg.loss.keypoint_heatmap_beta), floor_px,
        )  # fmt: skip
        if measured is None:
            continue
        cache["points"][index] = measured["coordinates_image"]
        cache["covariances"][index] = measured["covariance_image"]
        cache["confidence"][index] = measured["confidence"]
        cache["boxes"][index] = measured["box"]
        cache["detection_scores"][index] = measured["score"]
        cache["available"][index] = 1
    return cache


def cache_path(out: Path, roe: str, domain: str) -> Path:
    """``<out>/<roe>/<domain>/heatmap_measurements.npz``."""
    return Path(out) / roe / domain / CACHE_NAME


def cache_heatmap_measurements(
    checkpoint: str | Path,
    shirt_root: str | Path,
    layout: str | Path,
    candidates_root: str | Path,
    out: str | Path,
    *,
    trajectories: Sequence[str],
    domain: str = PAPER_DOMAIN,
    recipe: Any = None,  # noqa: ANN401 - ShirtRecipe
    allow_unpinned_checkpoint: bool = False,
    limit: int = 0,
    device: str = "cpu",
    floor_px: float = COVARIANCE_FLOOR_PX,
    force: bool = False,
) -> dict[str, Any]:
    """Cache each frame's heatmap landmarks and covariances for the keypoint UKF.

    For every frame of the candidate records, the top-1 detection (``score_thr=0.0``, ``max_det=1``;
    a fresh full-image pass, not the re-crops) returns its raw heatmaps;
    ``doper.filtering.measurements.heatmap_keypoint_measurements`` turns them into image-space
    landmarks and covariances (softmax mass, second moment about the DARK peak, the checkpoint's
    heatmap beta, a ``floor_px`` floor). Writes ``<out>/<roe>/<domain>/heatmap_measurements.npz``
    with the landmarks and covariances plus the checkpoint SHA-256, the layout's file SHA-256 and
    points hash, whether the layout was a formatting deviation, the floor and the heatmap beta
    (:func:`check_heatmap_cache_identity` checks them when the keypoint UKF reads the cache).

    The checkpoint pin and layout rules are ``doper.shirt.inputs.check_model_inputs``'s; the
    candidate records must come from the same checkpoint.

    Returns:
        ``{roe: {"path", "frames", "available"}}``.

    Raises:
        ValueError: For non-contiguous candidate records, an unpinned checkpoint without
            ``allow_unpinned_checkpoint``, candidates of another checkpoint, a checkpoint not
            trained on ``layout``, or one without a heatmap keypoint head.
        FileExistsError: If a cache exists and ``force`` is false.
    """
    from . import inputs

    recipe = recipe if recipe is not None else inputs.packaged_recipe()
    shirt_root, out = Path(shirt_root), Path(out)
    inputs.check_request(trajectories, domain, recipe)
    existing = [cache_path(out, roe, domain) for roe in trajectories]
    if any(path.exists() for path in existing) and not force:
        raise FileExistsError(f"heatmap cache {existing[0]} exists; pass --force to rebuild it")
    sequences = {
        roe: load_candidate_records(Path(candidates_root), roe, domain, limit)
        for roe in trajectories
    }
    for records in sequences.values():
        check_frame_indices(records)
    checked = inputs.check_model_inputs(checkpoint, layout, recipe, allow_unpinned_checkpoint)
    inputs.recorded_checkpoint(Path(candidates_root), "candidate", checked.checkpoint_sha256)
    model, cfg = inputs.load_model(checked.checkpoint, len(checked.layout.points), device)
    if cfg.model.keypoint_head != "heatmap":
        raise ValueError(
            f"landmark covariances need a heatmap keypoint head, not {cfg.model.keypoint_head!r}"
        )
    beta = inputs.heatmap_beta(checked.payload, cfg)
    result: dict[str, Any] = {}
    for roe, records in sequences.items():
        cache = _measure_trajectory(model, cfg, shirt_root, roe, domain, records, device, floor_px)
        path = cache_path(out, roe, domain)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {
            "filenames": np.asarray([r["filename"] for r in records]), **cache,
            "checkpoint_sha256": np.asarray(checked.checkpoint_sha256),
            "layout_sha256": np.asarray(checked.layout.sha256),
            "layout_points_sha256": np.asarray(checked.layout.points_sha256),
            "layout_formatting_deviation": np.asarray(checked.match.formatting_deviation),
            "covariance_floor_image_px": np.asarray(floor_px),
            "keypoint_heatmap_beta": np.asarray(beta),
        }  # fmt: skip
        np.savez_compressed(path, **arrays)
        result[roe] = {"path": str(path), "frames": len(records),
                       "available": int(cache["available"].sum())}  # fmt: skip
    return result


# --- the trajectory drivers behind `doper evaluate shirt --method {pnp-ukf,...}` ---------------


def plain_json(value: Any) -> Any:  # noqa: ANN401
    """``value`` with numpy scalars and arrays as Python values and NaN as None (strict JSON)."""
    if isinstance(value, Mapping):
        return {key: plain_json(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return plain_json(value.tolist())
    if isinstance(value, (list, tuple)):
        return [plain_json(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_sequence(folder: Path, records: Sequence[Mapping[str, Any]], summary: Any) -> None:  # noqa: ANN401
    from ..bop.sources import publish_json, publish_text

    lines = "".join(json.dumps(plain_json(r), allow_nan=False) + "\n" for r in records)
    publish_text(folder / RECORDS_NAME, lines)
    publish_json(folder / SUMMARY_NAME, plain_json(summary))


@dataclass(frozen=True)
class _KeypointInputs:
    """The keypoint UKF's layout and the checkpoint its caches must come from."""

    points_3d: np.ndarray
    layout: LayoutPins
    checkpoint_sha256: str | None
    camera: Any


def filter_trajectories(
    method: str,
    shirt_root: str | Path,
    candidates_root: str | Path,
    out: str | Path,
    *,
    trajectories: Sequence[str],
    domain: str = PAPER_DOMAIN,
    layout: str | Path | None = None,
    caches_root: str | Path | None = None,
    limit: int = 0,
    force: bool = False,
) -> dict[str, Any]:
    """Run one Table 2 filter over the candidate records of each trajectory.

    ``pnp-ukf`` writes :func:`pnp_ukf_outputs`, ``pnp-ukf-rts`` :func:`rts_outputs` and
    ``keypoint-ukf`` :func:`keypoint_ukf_outputs` (it reads the heatmap caches under
    ``caches_root`` and the ``layout``) to ``<out>/<roe>/<domain>/{records.jsonl,summary.json}``,
    and the settings, input SHA-256s and each trajectory's summary to ``<out>/summary.json``.
    The keypoint UKF refuses a cache built with another layout (points hash) or from another
    checkpoint than the candidates' (:func:`check_heatmap_cache_identity`). The summary records
    ``layout_formatting_deviation`` when the candidates' checkpoint or the caches matched their
    layout by points hash only.

    Raises:
        ValueError: For an unknown method or trajectory, a keypoint run without its layout or
            caches, a cache of another layout or checkpoint, or records the filters refuse.
        FileExistsError: If outputs exist and ``force`` is false.
    """
    from ..bop.sources import publish_json, sha256_file
    from . import check_request
    from .inputs import recorded_summary

    if method not in METHODS:
        raise ValueError(f"unknown SHIRT filter {method!r}; known: {METHODS}")
    check_request(trajectories, domain)
    if method == KEYPOINT_UKF and (layout is None or caches_root is None):
        raise ValueError("the keypoint UKF needs the layout and the heatmap caches")
    shirt_root, out, candidates_root = Path(shirt_root), Path(out), Path(candidates_root)
    if (out / SUMMARY_NAME).exists() and not force:
        raise FileExistsError(f"{method} outputs already exist ({out}); pass --force")
    candidates_summary = recorded_summary(candidates_root) or {}
    deviation = candidates_summary.get("layout_formatting_deviation")
    keypoint = None
    if method == KEYPOINT_UKF:
        from ..speedplus.checkpoints import load_layout

        layout_file = load_layout(layout)  # type: ignore[arg-type]
        keypoint = _KeypointInputs(layout_file.points, layout_file.identity,
                                   candidates_summary.get("checkpoint_sha256"),
                                   data.load_camera(shirt_root))  # fmt: skip
    summaries: dict[str, Any] = {}
    inputs: dict[str, Any] = {}
    combined: list[dict[str, Any]] = []
    for roe in trajectories:
        records = load_candidate_records(candidates_root, roe, domain, limit)
        metadata_path = shirt_root / roe / data.METADATA_NAME
        metadata = data.load_metadata(shirt_root / roe)
        rows, summary, key, cache_deviation = _run_method(
            method, records, metadata, keypoint, caches_root, roe, domain
        )
        if keypoint is not None:
            deviation = bool(deviation) or cache_deviation
        _write_sequence(out / roe / domain, rows, summary)
        combined.extend(row[key] for row in rows)
        summaries[roe] = {domain: summary}
        inputs[roe] = {
            "candidate_records_sha256": sha256_file(candidates_root / roe / domain /
                                                    RECORDS_NAME),
            "metadata_sha256": sha256_file(metadata_path),
        }  # fmt: skip
    result = {
        "method": method,
        "domain": domain,
        "score_definition": SCORE_DEFINITION,
        "settings": _method_settings(method),
        "metrics": "means over every frame: geodesic rotation (deg), Euclidean translation (m); "
        "not second-orbit metrics",
        "ground_truth_used_by_filters_or_gates": False,
        "candidates_checkpoint_sha256": candidates_summary.get("checkpoint_sha256"),
        "layout_formatting_deviation": deviation,
        "inputs": inputs,
        "limit": limit,
        "trajectories": summaries,
        "combined": summarize_pose_values(combined),
    }
    if keypoint is not None:
        result["layout_sha256"] = keypoint.layout.sha256
        result["layout_points_sha256"] = keypoint.layout.points_sha256
        result["camera_sha256"] = sha256_file(shirt_root / data.CAMERA_NAME)
    publish_json(out / SUMMARY_NAME, plain_json(result))
    return result


def _method_settings(method: str) -> dict[str, Any]:
    settings = _settings(nis_gate=CHI2_6DOF["p99"], nis_gate_label="p99")
    if method == NIS_RANK_RTS:
        settings["rts_passes"] = RTS_PASSES
    if method == KEYPOINT_UKF:
        settings.update(landmark_nis_gate_2d=CHI2_2DOF_P99, covariance_scale=COVARIANCE_SCALE,
                        covariance_floor_image_px=COVARIANCE_FLOOR_PX)  # fmt: skip
    return settings


def _run_method(
    method: str,
    records: list[dict[str, Any]],
    metadata: Mapping[str, Any],
    keypoint: _KeypointInputs | None,
    caches_root: str | Path | None,
    roe: str,
    domain: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], str, bool]:
    """The records, summary, per-frame metric key and cache formatting deviation of ``method``."""
    causal = nis_rank(records, metadata)
    if method == NIS_RANK:
        rows, summary = pnp_ukf_outputs(causal)
        return rows, summary, "filtered", False
    if method == NIS_RANK_RTS:
        rows, summary = rts_outputs(rts(causal))
        return rows, summary, "rts", False
    if keypoint is None or caches_root is None:
        raise ValueError("the keypoint UKF needs the layout and the heatmap caches")
    path = cache_path(Path(caches_root), roe, domain)
    cache = load_heatmap_cache(path)
    deviation = check_heatmap_cache_identity(
        cache, path, keypoint.layout, keypoint.checkpoint_sha256
    )
    direct = keypoint_ukf(records, cache, metadata, keypoint.points_3d, keypoint.camera)
    rows, summary = keypoint_ukf_outputs(direct, causal, cache)
    return rows, summary, "direct_keypoint", deviation
