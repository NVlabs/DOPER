# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Relative camera-pose evaluation of a capture bundle against its robot reference.

The metric. Each capture gets a single-view DOPER pose ``P_i =
T_camera_object`` (top-1 detection with score at least ``score_thr``, ``max_det=1``, no TTA, the
bundle's rectified K with zero distortion, PnP as in ``doper.infer``). A pose is rejected
(``None``) when there is no detection, no PnP solution, a non-finite value or ``t_z <= 0``.

For every pair ``i < j`` (capture order) whose two poses are valid, the camera motion the robot
recorded, ``ref = inv(H_i) @ H_j`` with ``H = T_base_camera``
(``doper.nurec.capture.reference_camera_poses``), is compared with the motion DOPER implies for
a static object, ``est = P_i @ inv(P_j)`` (both are ``T_Ci_Cj``). The errors are the rotation
geodesic in degrees and ``1000 * ||t_est - t_ref||`` in millimeters; the latter assumes the
keypoint layout is in meters. There is no alignment, scale fit or outlier filtering: every error
is conditioned on both endpoint poses being valid. ``summarize_pairs`` reports the count,
``np.median`` and ``np.percentile(..., 95)`` over all pairs and over adjacent pairs
(``j == i + 1`` in capture order).

Captures named in an ``exclude`` list (the bundle's ``exclude.json`` and the recipe's
``evaluation.exclude``) are listed in ``frames.json`` as excluded and enter no pair.

A summary records the capture bundle's identity (``bundle_sha256``), and ``evaluate_bundle``
refuses a bundle that is not the one the recipe's profile pins.

``compare_runs`` compares two runs over the same captures of the same bundle (and the same score
threshold, ``max_det`` and capture flags) on the pairs valid in **both**:
each run's median and p95 and the median paired change ``median(adapted - baseline)``, with
each run's failed captures. In its keys, ``finetune`` is the adapted run.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..reconstruction.geometry import rotation_geodesic_deg

FRAMES_NAME = "frames.json"
PAIRS_NAME = "pairs.json"
SUMMARY_NAME = "summary.json"
COMPARISON_NAME = "paired_comparison.json"
OUTPUT_NAMES = (FRAMES_NAME, PAIRS_NAME, SUMMARY_NAME)
ROTATION_KEY = "rotation_error_deg"
TRANSLATION_KEY = "translation_error_mm_assuming_keypoints_metres"
METRICS = (ROTATION_KEY, TRANSLATION_KEY)
DEFAULT_SCORE_THRESHOLD = 0.3
MAX_DETECTIONS = 1
PERCENTILE = 95
MM_PER_M = 1000.0
DISTORTION = np.zeros(5)
# The caveats every ``summary.json`` records; the third follows ``capture_flags``'s
# ``handeye_camera_mismatch``.
CAVEAT_SOURCE_CAPTURE = (
    "Source-capture diagnostic, not independent accuracy: the captures are the "
    "reconstruction's sources; checkpoints selected on synthetic or NuRec validation; no real "
    "checkpoint selection"
)
CAVEAT_NO_FITTING = (
    "No pose or scale fitting; metric translation assumes bundled CAD coordinates in meters, "
    "physical scale not independently verified"
)
HANDEYE_CAVEATS: dict[bool | None, str] = {
    True: "Hand-eye calibration recorded at a different camera mode; transfer unvalidated",
    False: "Hand-eye calibration recorded at the capture camera mode",
    None: "Hand-eye calibration camera mode not recorded; transfer unvalidated",
}
CAVEAT_CORRELATED_PAIRS = (
    "Pair errors conditioned on valid endpoint poses; overlapping pairs correlated"
)
COMPARISON_SCOPE = (
    "Both runs checked to share the capture bundle (bundle_sha256: images, K, robot poses and "
    "hand-eye calibration), the images and exclusions in order, the score threshold, max_det "
    "and the capture flags; no alignment or error filtering. Common valid pairs plus explicit "
    "missing-pose failures. Source-capture diagnostic, not independent accuracy. Wrong-target "
    "detections require visual review."
)
# The summary keys two compared runs must share (``compare_runs``).
COMPARED_SUMMARY_KEYS = ("bundle_sha256", "score_thr", "max_det", "capture_flags")
COMPARISON_ROLES = ("baseline", "finetune")


@dataclass(frozen=True)
class FramePose:
    """One capture's DOPER result.

    Attributes:
        image: Capture name.
        sha256: SHA-256 of the capture.
        detections: Detections returned (0 or 1).
        score: Detection score.
        keypoints: ``(K, 2)`` keypoints, pixels.
        keypoint_scores: ``(K,)`` keypoint scores.
        bbox: ``(4,)`` box ``xyxy``, pixels.
        T_camera_object: Object pose in the camera (OpenCV axes, keypoint units), or None when
            rejected.
        reprojection_error_px: Mean PnP reprojection error of a valid pose, pixels.
        excluded: Whether the capture is excluded from the metric (not inferred).
    """

    image: str
    sha256: str
    detections: int = 0
    score: float | None = None
    keypoints: NDArray[np.float64] | None = None
    keypoint_scores: NDArray[np.float64] | None = None
    bbox: NDArray[np.float64] | None = None
    T_camera_object: NDArray[np.float64] | None = None  # noqa: N815 - frame-named transform
    reprojection_error_px: float | None = None
    excluded: bool = False

    @property
    def pose_success(self) -> bool:
        """Whether the capture has a valid pose."""
        return self.T_camera_object is not None

    def as_record(self, T_base_camera: NDArray[np.float64]) -> dict[str, Any]:  # noqa: N803
        """The ``frames.json`` row, with the capture's reference camera pose."""
        record: dict[str, Any] = {
            "image": self.image,
            "sha256": self.sha256,
            "detections": self.detections,
            "T_base_camera": np.asarray(T_base_camera).tolist(),
        }
        if self.excluded:
            record["excluded"] = True
        if self.detections:
            record.update(
                score=self.score,
                keypoints=_list(self.keypoints),
                keypoint_scores=_list(self.keypoint_scores),
                bbox=_list(self.bbox),
            )
        if self.T_camera_object is not None:
            record.update(
                T_camera_object=self.T_camera_object.tolist(),
                reprojection_error_px=self.reprojection_error_px,
            )
        record["pose_success"] = self.pose_success
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> FramePose:
        """Rebuild a frame from its ``frames.json`` row."""
        pose = record.get("T_camera_object")
        return cls(
            image=record["image"],
            sha256=record["sha256"],
            detections=int(record["detections"]),
            score=record.get("score"),
            keypoints=_array(record.get("keypoints")),
            keypoint_scores=_array(record.get("keypoint_scores")),
            bbox=_array(record.get("bbox")),
            T_camera_object=_array(pose),
            reprojection_error_px=record.get("reprojection_error_px"),
            excluded=bool(record.get("excluded", False)),
        )


@dataclass(frozen=True)
class PairError:
    """The relative-pose error of one capture pair.

    Attributes:
        i: Capture index of the first view (capture order).
        j: Capture index of the second view, ``j > i``.
        rotation_error_deg: Rotation geodesic between estimated and reference motion, degrees.
        translation_error_mm: ``1000 * ||t_est - t_ref||``, millimeters if keypoints are meters.
        baseline_mm: ``1000 * ||t_ref||``, the robot-recorded camera baseline, millimeters.
    """

    i: int
    j: int
    rotation_error_deg: float
    translation_error_mm: float
    baseline_mm: float

    def as_record(self) -> dict[str, Any]:
        """The ``pairs.json`` row."""
        return {
            "i": self.i,
            "j": self.j,
            ROTATION_KEY: self.rotation_error_deg,
            TRANSLATION_KEY: self.translation_error_mm,
            "baseline_mm": self.baseline_mm,
        }


def _list(value: NDArray[np.float64] | None) -> list | None:
    return None if value is None else np.asarray(value).tolist()


def _array(value: Any) -> NDArray[np.float64] | None:  # noqa: ANN401
    return None if value is None else np.asarray(value, dtype=np.float64)


def pose_from_detection(detection: Any) -> NDArray[np.float64] | None:  # noqa: ANN401
    """``T_camera_object`` of a detection's PnP solution, or None when it is rejected.

    Rejected: no detection, no PnP solution, a non-finite value or ``t_z <= 0``.
    """
    import cv2

    if detection is None or detection.rvec is None:
        return None
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(np.asarray(detection.rvec, dtype=np.float64))[0]
    pose[:3, 3] = np.asarray(detection.tvec, dtype=np.float64).ravel()
    if not np.isfinite(pose).all() or pose[2, 3] <= 0:
        return None
    return pose


def _read_capture(path: Path, sha256: str, size: tuple[int, int]) -> NDArray[Any]:
    """Decode one capture (BGR) after checking its SHA-256 and size."""
    import hashlib

    import cv2

    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f"{path} SHA-256 differs from the bundle")
    pixels = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if pixels is None or (pixels.shape[1], pixels.shape[0]) != size:
        raise ValueError(f"{path} is not a {size[0]}x{size[1]} image")
    return pixels


def _frame_of(image: Any, detections: Sequence[Any]) -> FramePose:  # noqa: ANN401
    if not detections:
        return FramePose(image.name, image.sha256)
    det = detections[0]
    pose = pose_from_detection(det)
    return FramePose(
        image=image.name,
        sha256=image.sha256,
        detections=len(detections),
        score=float(det.score),
        keypoints=np.asarray(det.keypoints),
        keypoint_scores=np.asarray(det.keypoint_scores),
        bbox=np.asarray(det.bbox_xyxy),
        T_camera_object=pose,
        reprojection_error_px=None if pose is None else det.reproj_error_px,
    )


def estimate_frames(
    model: Any,  # noqa: ANN401
    image_set: Any,  # noqa: ANN401
    keypoints_3d: NDArray[np.float64],
    *,
    cfg: Any,  # noqa: ANN401
    score_thr: float = DEFAULT_SCORE_THRESHOLD,
    device: str = "cpu",
    exclude: Sequence[str] = (),
) -> list[FramePose]:
    """Single-view DOPER poses of every capture of an image set, in capture order.

    Args:
        model: The DOPER model (``doper.infer.load_model``).
        image_set: The capture images (``doper.nurec.capture.ImageSet``).
        keypoints_3d: ``(K, 3)`` keypoint layout of the model.
        cfg: The config ``load_model`` filled.
        score_thr: Detection score threshold.
        device: Torch device.
        exclude: Capture names to skip, besides the image set's own ``exclude``.

    Returns:
        One frame per capture; an excluded capture is not inferred.

    Raises:
        ValueError: For a capture whose SHA-256 or size differs from the bundle.
    """
    from .. import infer

    skipped = set(image_set.exclude) | set(exclude)
    camera = image_set.camera
    frames = []
    for image in image_set.images:
        if image.name in skipped:
            frames.append(FramePose(image.name, image.sha256, excluded=True))
            continue
        pixels = _read_capture(
            image_set.image_path(image.name), image.sha256, (camera.width, camera.height)
        )
        detections = infer.infer_image(
            model,
            pixels,
            cfg,
            device,
            score_thr=score_thr,
            max_det=MAX_DETECTIONS,
            keypoints_3d=keypoints_3d,
            camera_matrix=camera.matrix,
            dist_coeffs=DISTORTION,
        )
        frames.append(_frame_of(image, detections))
    return frames


def relative_pose_pairs(
    frames: Sequence[FramePose], references: Mapping[str, NDArray[np.float64]]
) -> list[PairError]:
    """The errors of every capture pair whose two poses are valid.

    Args:
        frames: Frames in capture order (``i``, ``j`` index this list).
        references: Capture name to ``T_base_camera`` (4x4, meters).

    Returns:
        The pair errors, ``(i, j)`` in lexicographic order.
    """
    poses = {i: f.T_camera_object for i, f in enumerate(frames) if f.T_camera_object is not None}
    if len(poses) < 2:
        return []
    # Each inverse once per frame, not once per pair (the same values either way).
    reference_inverses = {i: np.linalg.inv(references[frames[i].image]) for i in poses}
    pose_inverses = {i: np.linalg.inv(pose) for i, pose in poses.items()}
    pairs = []
    for i, j in itertools.combinations(poses, 2):
        reference = reference_inverses[i] @ references[frames[j].image]
        estimate = poses[i] @ pose_inverses[j]
        pairs.append(
            PairError(
                i,
                j,
                rotation_geodesic_deg(estimate[:3, :3], reference[:3, :3]),
                float(MM_PER_M * np.linalg.norm(estimate[:3, 3] - reference[:3, 3])),
                float(MM_PER_M * np.linalg.norm(reference[:3, 3])),
            )
        )
    return pairs


def _stats(pairs: Sequence[PairError]) -> dict[str, Any]:
    if not pairs:
        return {"count": 0}
    values = {
        ROTATION_KEY: [p.rotation_error_deg for p in pairs],
        TRANSLATION_KEY: [p.translation_error_mm for p in pairs],
    }
    return {
        "count": len(pairs),
        **{
            key: {"median": float(np.median(v)), "p95": float(np.percentile(v, PERCENTILE))}
            for key, v in values.items()
        },
    }


def summarize_pairs(pairs: Sequence[PairError], frames: Sequence[FramePose]) -> dict[str, Any]:
    """Counts, medians and p95s over all pairs and adjacent pairs.

    Args:
        pairs: The pair errors.
        frames: The frames the pairs index (excluded ones count in no total).

    Returns:
        ``frames`` (evaluated captures), ``pose_successes``, ``all_pairs``, ``adjacent_pairs``
        (each ``{count, <metric>: {median, p95}}``, only ``count`` when empty),
        ``total_all_pairs`` and ``total_adjacent_pairs`` (the pairs and adjacent pairs of the
        evaluated captures, valid or not).
    """
    evaluated = [index for index, frame in enumerate(frames) if not frame.excluded]
    kept = set(evaluated)
    return {
        "frames": len(evaluated),
        "pose_successes": sum(frames[index].pose_success for index in evaluated),
        "all_pairs": _stats(pairs),
        "adjacent_pairs": _stats([p for p in pairs if p.j == p.i + 1]),
        "total_all_pairs": len(evaluated) * (len(evaluated) - 1) // 2,
        "total_adjacent_pairs": sum(index + 1 in kept for index in evaluated),
    }


def caveats(capture_flags: Mapping[str, Any]) -> list[str]:
    """The four caveats of a summary; the hand-eye one follows ``capture_flags``."""
    mismatch = capture_flags.get("handeye_camera_mismatch")
    return [
        CAVEAT_SOURCE_CAPTURE,
        CAVEAT_NO_FITTING,
        HANDEYE_CAVEATS[mismatch],
        CAVEAT_CORRELATED_PAIRS,
    ]


def build_summary(
    pairs: Sequence[PairError],
    frames: Sequence[FramePose],
    *,
    model: str,
    settings: Mapping[str, Any],
    checkpoint_sha256: str,
    keypoints_sha256: str,
    capture_flags: Mapping[str, Any],
    exclude: Sequence[str],
    score_thr: float,
    bundle_sha256: str,
    bundle_files_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """The ``summary.json`` of a run.

    Beside the pose statistics it records ``score_thr``, ``max_det``, ``exclude``,
    ``capture_flags`` and the capture bundle's identity (``bundle_sha256``,
    ``capture.bundle_sha256``) with its per-file hashes (``bundle_files_sha256``, None when not
    known), which ``compare_runs`` requires equal.
    """
    return {
        "model": model,
        **summarize_pairs(pairs, frames),
        "settings": dict(settings),
        "checkpoint_sha256": checkpoint_sha256,
        "keypoints_sha256": keypoints_sha256,
        "caveats": caveats(capture_flags),
        "score_thr": score_thr,
        "max_det": MAX_DETECTIONS,
        "exclude": list(exclude),
        "capture_flags": capture_flags,
        "bundle_sha256": bundle_sha256,
        "bundle_files_sha256": None if bundle_files_sha256 is None else dict(bundle_files_sha256),
    }


def write_run(
    out: Path,
    frames: Sequence[FramePose],
    references: Mapping[str, NDArray[np.float64]],
    pairs: Sequence[PairError],
    summary: Mapping[str, Any],
) -> None:
    """Write ``frames.json``, ``pairs.json`` and, last, ``summary.json`` to ``out``."""
    from ..bop.sources import publish_json

    publish_json(out / FRAMES_NAME, [f.as_record(references[f.image]) for f in frames])
    publish_json(out / PAIRS_NAME, [p.as_record() for p in pairs])
    publish_json(out / SUMMARY_NAME, dict(summary))


def load_evaluation_model(checkpoint: Path, device: str) -> tuple[Any, Any, dict[str, Any]]:
    """Load a checkpoint for evaluation.

    Returns:
        The model, the config ``load_model`` filled and the model settings recorded in the
        summary.
    """
    import dataclasses

    from ..config import DataConfig, DoperConfig
    from ..infer import load_model

    cfg = DoperConfig(data=DataConfig(root=""))
    model = load_model(checkpoint, cfg, device)
    return model, cfg, dataclasses.asdict(cfg.model)


@dataclass(frozen=True)
class EvaluationRequest:
    """One relative-pose evaluation.

    Attributes:
        bundle: Capture bundle with a robot reference and ``keypoints_3d.json``.
        checkpoint: DOPER checkpoint.
        out: Output directory.
        keypoints_points_sha256: The layout's points hash the keypoints must have.
        profile: The recipe's profile, named when the bundle is not its pinned one.
        score_thr: Detection score threshold.
        exclude: Capture names excluded besides the bundle's ``exclude.json``.
        device: Torch device.
        force: Replace existing outputs.
        bundle_pin: The bundle identity the profile pins (``capture.profile_bundle_pin``);
            a bundle with another identity is refused. None checks no identity.
    """

    bundle: Path
    checkpoint: Path
    out: Path
    keypoints_points_sha256: str
    profile: str
    score_thr: float = DEFAULT_SCORE_THRESHOLD
    exclude: tuple[str, ...] = ()
    device: str = "cpu"
    force: bool = False
    bundle_pin: str | None = None


def _clear_outputs(out: Path, force: bool) -> None:
    existing = [out / name for name in OUTPUT_NAMES if (out / name).exists()]
    if existing and not force:
        raise FileExistsError(
            f"{out} holds an evaluation ({existing[0].name}); pass --force to replace it"
        )
    for path in existing:
        path.unlink()


def evaluate_bundle(request: EvaluationRequest) -> dict[str, Any]:
    """Evaluate a checkpoint on a capture bundle; write frames, pairs and summary.

    Returns:
        The summary.

    Raises:
        FileNotFoundError: For a bundle without a robot reference or keypoints.
        FileExistsError: When ``out`` holds an evaluation and ``force`` is false.
        ValueError: For a bundle that is not ``bundle_pin``, another keypoint layout,
            an exclusion of an unknown capture, or a capture whose SHA-256 or size differs.
    """
    from ..bop.sources import sha256_file
    from ..keypoints import load_layout
    from ..nurec import capture

    bundle_sha256 = capture.verify_bundle_pin(request.bundle, request.bundle_pin, request.profile)
    # ``estimate_frames`` checks each image it reads (SHA-256, size) as it decodes it.
    image_set = capture.load_image_set(request.bundle, verify=False)
    reference = capture.load_robot_reference(request.bundle, image_set)
    unknown = sorted(set(request.exclude) - set(image_set.names))
    if unknown:
        raise ValueError(f"evaluation.exclude names captures not in the bundle: {unknown}")
    keypoints_path = request.bundle / capture.KEYPOINTS_NAME
    keypoints = load_layout(keypoints_path, request.keypoints_points_sha256)
    _clear_outputs(request.out, request.force)
    exclude = [
        name for name in image_set.names if name in set(request.exclude) | set(image_set.exclude)
    ]
    model, cfg, settings = load_evaluation_model(request.checkpoint, request.device)
    frames = estimate_frames(
        model,
        image_set,
        keypoints,
        cfg=cfg,
        score_thr=request.score_thr,
        device=request.device,
        exclude=exclude,
    )
    references = capture.reference_camera_poses(reference)
    pairs = relative_pose_pairs(frames, references)
    summary = build_summary(
        pairs,
        frames,
        model=request.checkpoint.name,
        settings=settings,
        checkpoint_sha256=sha256_file(request.checkpoint),
        keypoints_sha256=sha256_file(keypoints_path),
        capture_flags=capture.capture_flags(image_set, reference),
        exclude=exclude,
        score_thr=request.score_thr,
        bundle_sha256=bundle_sha256,
        bundle_files_sha256=capture.bundle_files_sha256(request.bundle),
    )
    write_run(request.out, frames, references, pairs, summary)
    return summary


def _read(path: Path) -> Any:  # noqa: ANN401
    with path.open() as stream:
        return json.load(stream)


def _run_record(frames: Sequence[Mapping[str, Any]], run: Path) -> dict[str, Any]:
    record = {
        "images": len(frames),
        "pose_successes": sum(bool(f["pose_success"]) for f in frames),
        "failed_images": [
            f["image"] for f in frames if not f["pose_success"] and not f.get("excluded")
        ],
    }
    excluded = [f["image"] for f in frames if f.get("excluded")]
    if excluded:
        record["excluded_images"] = excluded
    if (run / SUMMARY_NAME).is_file():
        record["checkpoint_sha256"] = _read(run / SUMMARY_NAME).get("checkpoint_sha256")
    return record


def _metric_comparison(x: NDArray[np.float64], y: NDArray[np.float64]) -> dict[str, float] | None:
    if not len(x):
        return None
    return {
        "baseline_median": float(np.median(x)),
        "finetune_median": float(np.median(y)),
        "baseline_p95": float(np.percentile(x, PERCENTILE)),
        "finetune_p95": float(np.percentile(y, PERCENTILE)),
        "median_paired_change": float(np.median(y - x)),
    }


def _comparison_identity(run: Path) -> dict[str, Any]:
    """The ``COMPARED_SUMMARY_KEYS`` of a run's summary.

    Raises:
        FileNotFoundError: Without ``summary.json``.
        ValueError: For a summary without the capture bundle identity.
    """
    path = run / SUMMARY_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: compare two runs of `doper evaluate`")
    summary = _read(path)
    if not summary.get("bundle_sha256"):
        raise ValueError(f"{path} records no capture bundle identity (bundle_sha256); rerun "
                         "`doper evaluate` to record it")  # fmt: skip
    return {key: summary.get(key) for key in COMPARED_SUMMARY_KEYS}


def compare_runs(baseline_dir: Path, adapted_dir: Path) -> dict[str, Any]:
    """Compare two evaluations of the same captures on their common valid pairs.

    Args:
        baseline_dir: The baseline run (``frames.json``, ``pairs.json``).
        adapted_dir: The adapted run.

    Returns:
        Per scope (``all_pairs``, ``adjacent_pairs``) ``common_valid_pairs`` and per metric the
        baseline and adapted (``finetune``) median and p95 and ``median_paired_change``
        (``median(adapted - baseline)``; None without common pairs); per run its images, pose
        successes, failed (and excluded) captures and checkpoint SHA-256; and ``scope``.

    Raises:
        FileNotFoundError: Without a run's ``summary.json``.
        ValueError: When the runs are not over the same captures (images, exclusions and order)
            of the same capture bundle with the same detection settings and capture flags
            (``COMPARED_SUMMARY_KEYS``).
    """
    runs = dict(zip(COMPARISON_ROLES, (Path(baseline_dir), Path(adapted_dir)), strict=True))
    summaries = {role: _comparison_identity(run) for role, run in runs.items()}
    differ = [key for key in COMPARED_SUMMARY_KEYS
              if summaries["baseline"][key] != summaries["finetune"][key]]  # fmt: skip
    if differ:
        raise ValueError(f"{baseline_dir} and {adapted_dir} must evaluate the same capture bundle "
                         f"with the same settings; their summaries differ in {differ}")  # fmt: skip
    frames = {role: _read(run / FRAMES_NAME) for role, run in runs.items()}
    identity = {
        role: [(f["image"], f["sha256"], bool(f.get("excluded"))) for f in rows]
        for role, rows in frames.items()
    }
    if identity["baseline"] != identity["finetune"]:
        raise ValueError(
            f"{baseline_dir} and {adapted_dir} must evaluate the same images (and "
            "exclusions) in the same order"
        )
    pairs = {
        role: {(p["i"], p["j"]): p for p in _read(run / PAIRS_NAME)} for role, run in runs.items()
    }
    keys = sorted(pairs["baseline"].keys() & pairs["finetune"].keys())
    result: dict[str, Any] = {}
    for scope, selected in (
        ("all_pairs", keys),
        ("adjacent_pairs", [k for k in keys if k[1] == k[0] + 1]),
    ):
        result[scope] = {"common_valid_pairs": len(selected)}
        for metric in METRICS:
            x, y = (
                np.array([pairs[role][k][metric] for k in selected], dtype=np.float64)
                for role in COMPARISON_ROLES
            )
            result[scope][metric] = _metric_comparison(x, y)
    for role, run in runs.items():
        result[role] = _run_record(frames[role], run)
    result["scope"] = COMPARISON_SCOPE
    return result


def write_comparison(baseline_dir: Path, adapted_dir: Path, out: Path) -> dict[str, Any]:
    """Write ``compare_runs`` to ``out/paired_comparison.json``; return it."""
    from ..bop.sources import publish_json

    result = compare_runs(baseline_dir, adapted_dir)
    publish_json(Path(out) / COMPARISON_NAME, result)
    return result


__all__ = [
    "CAVEAT_CORRELATED_PAIRS",
    "CAVEAT_NO_FITTING",
    "CAVEAT_SOURCE_CAPTURE",
    "COMPARISON_NAME",
    "COMPARISON_SCOPE",
    "COMPARED_SUMMARY_KEYS",
    "FRAMES_NAME",
    "HANDEYE_CAVEATS",
    "PAIRS_NAME",
    "SUMMARY_NAME",
    "EvaluationRequest",
    "FramePose",
    "PairError",
    "build_summary",
    "caveats",
    "compare_runs",
    "estimate_frames",
    "evaluate_bundle",
    "load_evaluation_model",
    "pose_from_detection",
    "relative_pose_pairs",
    "summarize_pairs",
    "write_comparison",
    "write_run",
]
