# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation average recall (AR) for BOP datasets, with jitter and GT-box diagnostics.

Three protocols:

* ``top1`` / pinhole (``HANDAL-single-target-top1``): one target instance per frame; the
  highest-scoring detection is scored, even when its PnP failed;
* ``top5-greedy`` / pinhole (``HOPE-multi-instance-top5-score-greedy-v1``): up to five
  score-ranked detections per image, matched greedily and one-to-one to the GT instances in
  confidence order; every GT instance is in the recall denominator;
* ``top5-greedy`` / native (``HOT3D-native-fisheye-top5-score-greedy-MSSDmm-MSPDwidth640-v1``):
  as above on rectified images, with MSPD projected through the original fisheye calibration
  (``doper.bop.hot3d.project_native``).

MSSD uses millimeter meshes and thresholds of 5..50 % of the diameter; MSPD is normalized to a
640 px wide image (``x 640 / width``) with thresholds of 5..50 px. Errors and symmetries follow
``bop_toolkit_lib`` (``max_sym_disc_step=0.01``) through the ``toolkit_*`` functions of
``doper.evaluation.bop_metrics``. A miss or failed PnP fails every threshold; there is no
visibility filter and GT counts never bound the prediction budget.

Arms: ``single`` (single-pass inference) always; ``jitter`` (``box_dither.apply_box_jitter`` on
each detection) with a ``JitterSpec``; ``gtbox`` (keypoints and PnP inside each GT box, a paired
per-instance oracle that is not matched) with ``gt_box=True``.

The validation population is a BOP split directory or a frozen evaluation manifest
(``doper.bop.population``).
"""

from __future__ import annotations

import dataclasses
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import cv2
import numpy as np
import torch

from ..box_dither import (
    JITTER_BATCH_SIZE,
    JITTER_TRIGGER_REPROJECTION_PX,
    apply_box_jitter,
    keypoints_from_boxes,
)
from ..config import DoperConfig
from ..evaluation.bop_metrics import (
    read_ply_vertices,
    toolkit_mspd,
    toolkit_mssd,
    toolkit_symmetry_transforms,
)
from ..infer import Detection, assert_keypoints_match, infer_image, infer_keypoints_in_box
from ..model import DoperNet
from . import hot3d
from .hot3d import project_native
from .population import CameraName, Frame, FrameLoader, load_population
from .sources import sha256_file

ProtocolName = Literal["top1", "top5-greedy"]

PROTOCOL_TOP1 = "HANDAL-single-target-top1"
PROTOCOL_TOP5 = "HOPE-multi-instance-top5-score-greedy-v1"
PROTOCOL_HOT3D_NATIVE = "HOT3D-native-fisheye-top5-score-greedy-MSSDmm-MSPDwidth640-v1"
PROTOCOL_IDS: dict[tuple[str, str], str] = {
    ("top1", "pinhole"): PROTOCOL_TOP1,
    ("top5-greedy", "pinhole"): PROTOCOL_TOP5,
    ("top5-greedy", "native"): PROTOCOL_HOT3D_NATIVE,
}
SCORE_THRESHOLD = 0.15
MAX_DETECTIONS = 5
MSPD_REFERENCE_WIDTH_PX = 640
MSSD_STEP_FRACTION = 0.05
MSPD_STEP_PX = 5.0
THRESHOLD_COUNT = 10
ARMS = ("single", "jitter", "gtbox")
DEVICES = ("Aria", "Quest3")


@dataclass(frozen=True)
class JitterSpec:
    """Box-jitter arm settings (``box_dither.apply_box_jitter``).

    Attributes:
        threshold_px: Baseline reprojection error above which the jitter grid runs.
        batch_size: Crops per forward pass.
    """

    threshold_px: float = JITTER_TRIGGER_REPROJECTION_PX
    batch_size: int = JITTER_BATCH_SIZE


class Predictor(Protocol):
    """Inference the scorer needs; ``ModelPredictor`` implements it for a model."""

    def detect(self, image: np.ndarray, camera_matrix: np.ndarray) -> list[Detection]:
        """Return single-pass detections with poses (translations in millimeters)."""
        ...

    def predict_in_box(
        self, image: np.ndarray, box_xyxy: np.ndarray, camera_matrix: np.ndarray
    ) -> Detection:
        """Return keypoints and pose predicted inside a supplied box."""
        ...

    def refine(
        self,
        image: np.ndarray,
        detection: Detection | None,
        camera_matrix: np.ndarray,
        spec: JitterSpec,
    ) -> tuple[Detection | None, dict[str, Any]]:
        """Return the box-jitter selection for ``detection`` and its audit."""
        ...


@dataclass(frozen=True)
class ModelPredictor:
    """``Predictor`` over ``doper.infer`` with the validation score threshold and budget.

    Attributes:
        model: Heatmap-head model.
        cfg: Run configuration.
        device: Torch device string.
        keypoints_3d_mm: ``(K, 3)`` object keypoints in millimeters.
    """

    model: DoperNet
    cfg: DoperConfig
    device: str
    keypoints_3d_mm: np.ndarray

    def detect(self, image: np.ndarray, camera_matrix: np.ndarray) -> list[Detection]:
        """Run single-pass inference at score 0.15 with at most five detections."""
        return infer_image(
            self.model,
            image,
            self.cfg,
            self.device,
            score_thr=SCORE_THRESHOLD,
            max_det=MAX_DETECTIONS,
            keypoints_3d=self.keypoints_3d_mm,
            camera_matrix=camera_matrix,
        )

    def predict_in_box(
        self, image: np.ndarray, box_xyxy: np.ndarray, camera_matrix: np.ndarray
    ) -> Detection:
        """Run the keypoint head and PnP inside ``box_xyxy``."""
        with torch.no_grad():
            return infer_keypoints_in_box(
                self.model,
                image,
                self.cfg,
                self.device,
                box_xyxy,
                keypoints_3d=self.keypoints_3d_mm,
                camera_matrix=camera_matrix,
            )

    def refine(
        self,
        image: np.ndarray,
        detection: Detection | None,
        camera_matrix: np.ndarray,
        spec: JitterSpec,
    ) -> tuple[Detection | None, dict[str, Any]]:
        """Apply the box-jitter selection to ``detection``."""
        return apply_box_jitter(
            self.model,
            image,
            self.cfg,
            self.device,
            detection,
            self.keypoints_3d_mm,
            camera_matrix,
            threshold=spec.threshold_px,
            batch_size=spec.batch_size,
        )


@dataclass(frozen=True)
class ArmResult:
    """Recalls and AR (percent) of one arm.

    Attributes:
        frames: Scored frames.
        gt_instances: GT instances in the recall denominator (frames for ``top1``).
        ar_mssd: Mean MSSD recall, percent.
        ar_mspd: Mean MSPD recall, percent.
        ar_mean: Mean of the two, percent.
        mssd_recalls: Recall at 5..50 % of the diameter, fractions.
        mspd_recalls: Recall at 5..50 px (640 px width), fractions.
        pose_rate: Fraction of frames with a scored pose (``top1`` only).
    """

    frames: int
    gt_instances: int
    ar_mssd: float
    ar_mspd: float
    ar_mean: float
    mssd_recalls: tuple[float, ...]
    mspd_recalls: tuple[float, ...]
    pose_rate: float | None = None

    @classmethod
    def from_summary(cls, summary: Mapping[str, Any]) -> ArmResult:
        """Build from a ``recall_summary`` or ``aggregate`` arm entry."""
        return cls(
            frames=summary["frames"],
            gt_instances=summary.get("GT_instances", summary["frames"]),
            ar_mssd=summary["AR_MSSD"],
            ar_mspd=summary["AR_MSPD"],
            ar_mean=summary["AR_mean"],
            mssd_recalls=tuple(summary["mssd_recalls"]),
            mspd_recalls=tuple(summary["mspd_recalls"]),
            pose_rate=summary.get("pose_rate"),
        )


@dataclass(frozen=True)
class ArResult:
    """Validation AR of one object model.

    Attributes:
        obj_id: BOP object id.
        protocol: Protocol id (``PROTOCOL_IDS``); selection compares it across candidates.
        camera: ``pinhole`` or ``native``.
        status: ``scored``, or ``not_evaluable`` when the object has no validation GT.
        population_sha256: SHA-256 of the manifest file, or of the canonical population built
            from a split directory; candidates are comparable only when it matches.
        frames: Frames containing the object.
        gt_instances: GT instances of the object in those frames.
        arms: Per-arm results; ``single`` is always present when scored.
        by_device: Per-device arm results (native camera only).
        gtbox_keypoints: GT-box keypoint diagnostic (mean in-frame pixel error and point count),
            when the ``gtbox`` arm ran.
        rows: Per-frame records (errors, predictions, jitter audits) in journal form.
        checkpoint_sha256: SHA-256 of the scored checkpoint, when the caller supplied it;
            selection requires it to match the candidate.
        geometry_sha256: SHA-256 of the ordered keypoint file the score used.
        camera_code_sha256: SHA-256 of the native camera code (``doper/bop/hot3d.py``) for the
            native camera, else None.
    """

    obj_id: int
    protocol: str
    camera: CameraName
    status: Literal["scored", "not_evaluable"]
    population_sha256: str
    frames: int
    gt_instances: int
    arms: Mapping[str, ArmResult] = field(default_factory=dict)
    by_device: Mapping[str, Mapping[str, ArmResult]] = field(default_factory=dict)
    gtbox_keypoints: Mapping[str, Any] | None = None
    rows: tuple[Mapping[str, Any], ...] = ()
    checkpoint_sha256: str | None = None
    geometry_sha256: str | None = None
    camera_code_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON form that ``from_dict`` restores."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ArResult:
        """Rebuild a result from its ``to_dict`` form."""

        def arm(values: Mapping[str, Any]) -> ArmResult:
            recalls = {k: tuple(values[k]) for k in ("mssd_recalls", "mspd_recalls")}
            return ArmResult(**{**values, **recalls})

        return cls(
            **{
                **data,
                "arms": {k: arm(v) for k, v in data["arms"].items()},
                "by_device": {
                    d: {k: arm(v) for k, v in arms.items()} for d, arms in data["by_device"].items()
                },
                "rows": tuple(data["rows"]),
            }
        )


def gt_box_keypoints(
    model: DoperNet,
    image: np.ndarray,
    gt_boxes: Sequence[Sequence[float]],
    cfg: DoperConfig,
    device: str,
    batch_size: int = JITTER_BATCH_SIZE,
) -> list[Detection]:
    """Predict keypoints inside ground-truth boxes, bypassing the detector.

    Everything downstream of the box matches regular inference (crop warp at the configured pad
    factor, heatmap decode), so pose quality can be read independently of detection quality.
    The caller solves PnP with ``doper.infer.solve_pnp``.

    Args:
        model: Heatmap-head model.
        image: BGR uint8 image.
        gt_boxes: Boxes ``x1, y1, x2, y2`` in image pixels.
        cfg: Run configuration.
        device: Torch device string.
        batch_size: Crops per forward pass.

    Returns:
        One pose-free ``Detection`` per box, with score 1.0.

    Raises:
        ValueError: If ``gt_boxes`` is empty.
    """
    if len(gt_boxes) == 0:
        raise ValueError("gt_box_keypoints needs at least one box")
    return keypoints_from_boxes(model, image, cfg, device, gt_boxes, batch_size=batch_size)


def recall_summary(records: Sequence[Mapping[str, Any]], diameter: float) -> dict[str, Any]:
    """Summarize ``top1`` records (one per frame) into recalls and AR.

    Args:
        records: Per-frame ``{"mssd_mm", "mspd_px640"}``; None is a miss.
        diameter: Object diameter in millimeters.

    Returns:
        ``frames``, ``pose_rate``, ``AR_MSSD``, ``AR_MSPD``, ``AR_mean`` (percent) and the ten
        recalls per metric.

    Raises:
        ValueError: If ``records`` is empty (an empty population is not AR zero).
    """
    mssd = [r["mssd_mm"] if r["mssd_mm"] is not None else float("inf") for r in records]
    mspd = [r["mspd_px640"] if r["mspd_px640"] is not None else float("inf") for r in records]
    if not records:
        raise ValueError("empty population is not AR zero")
    steps = range(1, THRESHOLD_COUNT + 1)
    a = [float(np.mean(np.asarray(mssd) < diameter * i * MSSD_STEP_FRACTION)) for i in steps]
    b = [float(np.mean(np.asarray(mspd) < i * MSPD_STEP_PX)) for i in steps]
    return {
        "frames": len(records),
        "pose_rate": sum(r["mssd_mm"] is not None for r in records) / len(records),
        "AR_MSSD": 100 * float(np.mean(a)),
        "AR_MSPD": 100 * float(np.mean(b)),
        "AR_mean": 50 * (float(np.mean(a)) + float(np.mean(b))),
        "mssd_recalls": a,
        "mspd_recalls": b,
    }


def _passes(error: float | None, threshold: float) -> bool:
    return error is not None and math.isfinite(error) and error < threshold


def matching_hits(matrix: Sequence[Sequence[float | None]], threshold: float, gt_count: int) -> int:
    """Count greedy one-to-one matches below ``threshold``.

    Rows are predictions in confidence order; each takes the lowest-error unmatched GT
    instance (ties to the lower index) and every prediction or instance matches at most once.

    Args:
        matrix: Prediction-by-instance errors; None or non-finite never matches.
        threshold: Strict upper bound on a matching error.
        gt_count: Number of GT instances (columns).

    Returns:
        Number of matched instances.

    Raises:
        ValueError: If a row does not have ``gt_count`` entries.
    """
    available = set(range(gt_count))
    hits = 0
    for errors in matrix:
        if len(errors) != gt_count:
            raise ValueError("prediction/GT matrix shape differs")
        candidates = [i for i in available if _passes(errors[i], threshold)]
        if candidates:
            best = min(candidates, key=lambda i: (errors[i], i))
            available.remove(best)
            hits += 1
    return hits


def aggregate(
    rows: Sequence[Mapping[str, Any]], diameter: float, *, arms: Sequence[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Aggregate ``top5-greedy`` rows into per-arm recalls and AR.

    The ``gtbox`` arm is a paired per-instance oracle and is counted per instance; the other
    arms use ``matching_hits``.

    Args:
        rows: Per-frame records with ``gt_indices`` and per-arm ``mssd_mm`` / ``mspd_px640``.
        diameter: Object diameter in millimeters.
        arms: Arms to aggregate; defaults to those of ``ARMS`` present in the first row.

    Returns:
        Per arm: ``frames``, ``GT_instances``, ``AR_MSSD``, ``AR_MSPD``, ``AR_mean`` (percent)
        and the ten recalls per metric.

    Raises:
        ValueError: If there are no GT instances or a GT-box row lost instances.
    """
    n = sum(len(r["gt_indices"]) for r in rows)
    if not n:
        raise ValueError("missing GT is N/A, not zero")
    if arms is None:
        arms = [arm for arm in ARMS if arm in rows[0]["arms"]]
    result = {}
    for arm in arms:
        recall = {}
        for metric, step in (
            ("mssd_mm", diameter * MSSD_STEP_FRACTION),
            ("mspd_px640", MSPD_STEP_PX),
        ):
            values = []
            for i in range(1, THRESHOLD_COUNT + 1):
                hits = 0
                for row in rows:
                    errors = row["arms"][arm][metric]
                    if arm == "gtbox":
                        if len(errors) != len(row["gt_indices"]):
                            raise ValueError("oracle population changed")
                        hits += sum(_passes(e, step * i) for e in errors)
                    else:
                        hits += matching_hits(errors, step * i, len(row["gt_indices"]))
                values.append(hits / n)
            recall[metric] = values
        a = 100 * statistics.mean(recall["mssd_mm"])
        b = 100 * statistics.mean(recall["mspd_px640"])
        result[arm] = {
            "frames": len(rows),
            "GT_instances": n,
            "AR_MSSD": a,
            "AR_MSPD": b,
            "AR_mean": (a + b) / 2,
            "mssd_recalls": recall["mssd_mm"],
            "mspd_recalls": recall["mspd_px640"],
        }
    return result


def _independent_hits(row: Mapping[str, Any], arm: str, metric: str, threshold: float) -> int:
    errors = row["arms"][arm][metric]
    if arm == "gtbox":
        return sum(e is not None and e < threshold for e in errors)
    used: set[int] = set()
    hits = 0
    for prediction in errors:
        candidates = sorted(
            (e, i)
            for i, e in enumerate(prediction)
            if i not in used and e is not None and e < threshold
        )
        if candidates:
            used.add(candidates[0][1])
            hits += 1
    return hits


def independent_scores(result: ArResult, diameter: float) -> dict[str, float]:
    """Re-aggregate a ``top5-greedy`` result with an independent loop and check it.

    Checks the fixed five-prediction budget and the prediction/instance matrix shapes, then
    recomputes each arm's ``AR_mean``.

    Args:
        result: A scored ``top5-greedy`` result.
        diameter: Object diameter in millimeters.

    Returns:
        ``AR_mean`` per arm, in percent.

    Raises:
        ValueError: If a row breaks the budget or shapes, or an AR differs by more than 1e-8.
    """
    n = sum(len(r["gt_indices"]) for r in result.rows)
    for row in result.rows:
        count = len(row["gt_indices"])
        for arm in set(row["arms"]) - {"gtbox"}:
            predictions = row["arms"][arm]["predictions"]
            if len(predictions) > MAX_DETECTIONS:
                raise ValueError("fixed prediction budget exceeded")
            for metric in ("mssd_mm", "mspd_px640"):
                matrix = row["arms"][arm][metric]
                if len(matrix) != len(predictions) or any(len(x) != count for x in matrix):
                    raise ValueError("prediction/instance matrix changed")
        if "gtbox" in row["arms"] and any(
            len(row["arms"]["gtbox"][key]) != count
            for key in ("predictions", "mssd_mm", "mspd_px640")
        ):
            raise ValueError("oracle instance population changed")
    scores = {}
    for arm, summary in result.arms.items():
        hits = 0
        for metric, scale in (
            ("mssd_mm", diameter * MSSD_STEP_FRACTION),
            ("mspd_px640", MSPD_STEP_PX),
        ):
            for threshold in range(1, THRESHOLD_COUNT + 1):
                hits += sum(
                    _independent_hits(r, arm, metric, scale * threshold) for r in result.rows
                )
        value = 100 * hits / (2 * THRESHOLD_COUNT * n)
        if not math.isclose(value, summary.ar_mean, abs_tol=1e-8):
            raise ValueError("independent re-aggregation differs")
        scores[arm] = value
    return scores


def _prediction(det: Detection | None) -> dict[str, Any] | None:
    """Pose of a detection as BOP row-major R and t (mm), or None without a finite pose."""
    if det is None or det.rvec is None or det.tvec is None:
        return None
    rotation, _ = cv2.Rodrigues(det.rvec)
    translation = np.asarray(det.tvec).reshape(3, 1)
    if not (np.isfinite(rotation).all() and np.isfinite(translation).all()):
        return None
    if not math.isfinite(det.score):
        return None
    return {
        "R": rotation.ravel().tolist(),
        "t": translation.ravel().tolist(),
        "score": float(det.score),
    }


class _Scorer:
    """MSSD and (pinhole or native) MSPD of predictions against one object's GT instances."""

    def __init__(
        self, points: np.ndarray, syms: list[np.ndarray], keypoints_mm: np.ndarray
    ) -> None:
        self.points = points
        self.syms = syms
        self.keypoints_mm = keypoints_mm

    def gt_projections(self, frame: Frame) -> list[list[np.ndarray] | None]:
        """Native-pixel projections of every symmetric GT pose, per instance (native only)."""
        if frame.native is None:
            return [None] * len(frame.matches)
        result = []
        for _, gt in frame.matches:
            rotation = np.asarray(gt["cam_R_m2c"]).reshape(3, 3)
            translation = np.asarray(gt["cam_t_m2c"]).reshape(3)
            per_symmetry = []
            for sym in self.syms:
                sym_rotation = rotation @ np.ascontiguousarray(sym[:3, :3])
                sym_translation = (rotation @ np.ascontiguousarray(sym[:3, 3:4])).reshape(3)
                per_symmetry.append(
                    self._native_uv(frame, sym_rotation, sym_translation + translation)
                )
            result.append(per_symmetry)
        return result

    def _native_uv(self, frame: Frame, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
        assert frame.calibration is not None
        return project_native(
            self.points, rotation, translation, dict(frame.calibration), native=frame.native
        )

    def errors(
        self,
        prediction: Mapping[str, Any] | None,
        gt: Mapping[str, Any],
        frame: Frame,
        gt_pixels: list[np.ndarray] | None,
    ) -> tuple[float | None, float | None]:
        """Return ``(MSSD mm, MSPD px at 640 width)``, or Nones without a finite pose."""
        if prediction is None:
            return None, None
        rotation = np.asarray(prediction["R"]).reshape(3, 3)
        translation = np.asarray(prediction["t"]).reshape(3, 1)
        gt_rotation = np.asarray(gt["cam_R_m2c"]).reshape(3, 3)
        gt_translation = np.asarray(gt["cam_t_m2c"]).reshape(3, 1)
        a = toolkit_mssd(rotation, translation, gt_rotation, gt_translation, self.points, self.syms)
        if gt_pixels is None:
            b = toolkit_mspd(
                rotation, translation, gt_rotation, gt_translation, frame.camera_matrix,
                self.points, self.syms,
            ) * MSPD_REFERENCE_WIDTH_PX / frame.image.shape[1]  # fmt: skip
        else:
            uv = self._native_uv(frame, rotation, translation)
            native_errors = [float(np.linalg.norm(uv - e, axis=1).max()) for e in gt_pixels]
            b = min(native_errors) * MSPD_REFERENCE_WIDTH_PX / frame.native.width
        return (a, b) if np.isfinite([a, b]).all() else (None, None)

    def keypoint_error(
        self, det: Detection, gt: Mapping[str, Any], frame: Frame
    ) -> tuple[float, int]:
        """Sum and count of GT-box keypoint errors over in-frame GT projections (pixels)."""
        rotation = np.asarray(gt["cam_R_m2c"]).reshape(3, 3)
        translation = np.asarray(gt["cam_t_m2c"]).reshape(3)
        xyz = self.keypoints_mm @ rotation.T + translation
        if frame.native is None:
            uvh = xyz @ frame.camera_matrix.T
            uv = uvh[:, :2] / uvh[:, 2:3]
            height, width = frame.image.shape[:2]
            predicted = det.keypoints
        else:
            assert frame.calibration is not None
            uv = project_native(
                self.keypoints_mm,
                rotation,
                translation,
                dict(frame.calibration),
                native=frame.native,
            )
            height, width = frame.native.height, frame.native.width
            predicted = frame.native.eye_to_window(frame.pinhole.window_to_eye(det.keypoints))
        valid = (xyz[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0)
        valid &= uv[:, 1] < height
        if frame.native is not None:
            valid &= np.isfinite(uv).all(axis=1)
        return float(np.linalg.norm(predicted[valid] - uv[valid], axis=1).sum()), int(valid.sum())


@dataclass
class _Run:
    """Per-object scoring state shared by the frame loops."""

    predictor: Predictor
    scorer: _Scorer
    jitter: JitterSpec | None
    gt_box: bool

    def top1_entry(
        self, det: Detection | None, gt: Mapping[str, Any], frame: Frame
    ) -> dict[str, Any]:
        prediction = _prediction(det)
        a, b = self.scorer.errors(prediction, gt, frame, None)
        if a is None:
            return {"mssd_mm": None, "mspd_px640": None, "prediction": None}
        return {"mssd_mm": a, "mspd_px640": b, "prediction": prediction}

    def top5_entry(
        self, dets: Sequence[Detection], frame: Frame, gt_pixels: list[list[np.ndarray] | None]
    ) -> dict[str, Any]:
        predictions = [_prediction(d) for d in dets]
        values = [
            [
                self.scorer.errors(p, gt, frame, uv)
                for (_, gt), uv in zip(frame.matches, gt_pixels, strict=True)
            ]
            for p in predictions
        ]
        return {
            "predictions": predictions,
            "mssd_mm": [[e[0] for e in row] for row in values],
            "mspd_px640": [[e[1] for e in row] for row in values],
        }

    def frame_row(self, frame: Frame) -> dict[str, Any]:
        record = frame.record
        return {
            "scene": record["scene"],
            "frame": record["frame"],
            "image_sha256": record["sha256"],
        }

    def score_top1(self, frame: Frame) -> dict[str, Any]:
        (_, gt), box = frame.matches[0], frame.boxes[0]
        dets = self.predictor.detect(frame.image, frame.camera_matrix)
        det = dets[0] if dets else None
        row = self.frame_row(frame)
        chosen: dict[str, Detection | None] = {"single": det}
        if self.jitter is not None:
            chosen["jitter"], row["jitter"] = self.predictor.refine(
                frame.image, det, frame.camera_matrix, self.jitter
            )
        if self.gt_box:
            oracle = None
            if box is not None:
                oracle = self.predictor.predict_in_box(frame.image, box, frame.camera_matrix)
            error_sum, count = 0.0, 0
            if oracle is not None:
                error_sum, count = self.scorer.keypoint_error(oracle, gt, frame)
            row["gtbox_kp_error_sum_native_px"], row["gtbox_kp_count"] = error_sum, count
            chosen["gtbox"] = oracle
        row["arms"] = {arm: self.top1_entry(d, gt, frame) for arm, d in chosen.items()}
        return row

    def score_top5(self, frame: Frame, indices: list[int]) -> dict[str, Any]:
        gt_pixels = self.scorer.gt_projections(frame)
        dets = self.predictor.detect(frame.image, frame.camera_matrix)
        dets = sorted(dets, key=lambda d: d.score, reverse=True)[:MAX_DETECTIONS]
        row = self.frame_row(frame)
        row["gt_indices"] = indices
        for key in ("device", "stream"):
            if key in frame.record:
                row[key] = frame.record[key]
        arms = {"single": dets}
        if self.jitter is not None:
            refined, audits = [], []
            for det in dets:
                selected, audit = self.predictor.refine(
                    frame.image, det, frame.camera_matrix, self.jitter
                )
                refined.append(selected)
                audits.append(audit)
            kept = [d for d in refined if d is not None]
            arms["jitter"] = sorted(kept, key=lambda d: d.score, reverse=True)[:MAX_DETECTIONS]
            row["jitter"] = audits
        row["arms"] = {arm: self.top5_entry(d, frame, gt_pixels) for arm, d in arms.items()}
        if self.gt_box:
            row["arms"]["gtbox"] = self._top5_gt_box(frame, gt_pixels, row)
        return row

    def _top5_gt_box(
        self, frame: Frame, gt_pixels: list[list[np.ndarray] | None], row: dict[str, Any]
    ) -> dict[str, Any]:
        oracle: dict[str, list[Any]] = {"predictions": [], "mssd_mm": [], "mspd_px640": []}
        error_sum, count = 0.0, 0
        for (_, gt), box, uv in zip(frame.matches, frame.boxes, gt_pixels, strict=True):
            det = None
            if box is not None:
                det = self.predictor.predict_in_box(frame.image, box, frame.camera_matrix)
            prediction = _prediction(det)
            a, b = self.scorer.errors(prediction, gt, frame, uv)
            oracle["predictions"].append(prediction)
            oracle["mssd_mm"].append(a)
            oracle["mspd_px640"].append(b)
            if det is not None:
                frame_sum, frame_count = self.scorer.keypoint_error(det, gt, frame)
                error_sum += frame_sum
                count += frame_count
        row["gtbox_kp_error_sum_native_px"], row["gtbox_kp_count"] = error_sum, count
        return oracle


@dataclass(frozen=True)
class _Lineage:
    checkpoint: str | None
    geometry: str | None
    camera_code: str | None


def camera_code_sha256() -> str:
    """SHA-256 of the native camera code that native validation and test inputs depend on."""
    assert hot3d.__file__ is not None
    return sha256_file(hot3d.__file__)


def _as_predictor(
    model_or_predictor: DoperNet | torch.nn.Module | Predictor,
    cfg: DoperConfig | None,
    device: str,
    keypoints_mm: np.ndarray,
) -> Predictor:
    if not isinstance(model_or_predictor, torch.nn.Module):
        return model_or_predictor
    if cfg is None:
        raise ValueError("scoring a model needs its cfg")
    assert_keypoints_match(model_or_predictor, keypoints_mm)  # type: ignore[arg-type]
    return ModelPredictor(model_or_predictor, cfg, device, keypoints_mm)  # type: ignore[arg-type]


def _gtbox_keypoints(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = sum(r["gtbox_kp_count"] for r in rows)
    total = sum(r["gtbox_kp_error_sum_native_px"] for r in rows)
    return {"error_native_px": total / count if count else None, "points": count}


def _summaries(
    rows: list[dict[str, Any]], protocol: ProtocolName, camera: CameraName, diameter: float
) -> tuple[dict[str, ArmResult], dict[str, dict[str, ArmResult]]]:
    if protocol == "top1":
        summaries = {
            arm: recall_summary([r["arms"][arm] for r in rows], diameter) for arm in rows[0]["arms"]
        }
        return {arm: ArmResult.from_summary(s) for arm, s in summaries.items()}, {}
    arms = {arm: ArmResult.from_summary(s) for arm, s in aggregate(rows, diameter).items()}
    by_device = {}
    if camera == "native":
        for device in DEVICES:
            subset = [r for r in rows if r.get("device") == device]
            if subset:
                by_device[device] = {
                    arm: ArmResult.from_summary(s) for arm, s in aggregate(subset, diameter).items()
                }
    return arms, by_device


def score_model(
    model_or_predictor: DoperNet | torch.nn.Module | Predictor,
    split_root: str | Path,
    *,
    obj_id: int,
    keypoints_3d_mm: np.ndarray,
    models_dir: str | Path,
    protocol: ProtocolName,
    camera: CameraName = "pinhole",
    jitter: JitterSpec | None = None,
    gt_box: bool = False,
    cfg: DoperConfig | None = None,
    device: str = "cpu",
    checkpoint_sha256: str | None = None,
    geometry_sha256: str | None = None,
) -> ArResult:
    """Score one object model's validation AR under one of the ``PROTOCOL_IDS`` protocols.

    Args:
        model_or_predictor: A model (with ``cfg`` and ``device``), or a ``Predictor``.
        split_root: BOP split directory, or an evaluation manifest JSON (see the module doc).
        obj_id: BOP object id.
        keypoints_3d_mm: ``(K, 3)`` object keypoints in millimeters (PnP and the GT-box
            keypoint diagnostic).
        models_dir: Directory with ``obj_XXXXXX.ply`` (mm) and ``models_info.json``.
        protocol: ``top1`` or ``top5-greedy``.
        camera: ``pinhole`` (MSPD through ``cam_K``) or ``native`` (HOT3D fisheye MSPD).
        jitter: Box-jitter arm settings; None skips the arm.
        gt_box: Whether to run the GT-box oracle arm and keypoint diagnostic.
        cfg: Run configuration, required with a model.
        device: Torch device string, with a model.
        checkpoint_sha256: SHA-256 of the scored checkpoint, recorded for selection lineage.
        geometry_sha256: SHA-256 of the ordered keypoint file, recorded for selection lineage.

    Returns:
        The object's ``ArResult``; ``not_evaluable`` when no frame contains the object.

    Raises:
        ValueError: For a protocol/camera pair with no defined protocol, a model without
            ``cfg``, or a population whose images, scene files or GT layout do not validate.
    """
    protocol_id = PROTOCOL_IDS.get((protocol, camera))
    if protocol_id is None:
        raise ValueError(f"no validation protocol scores {protocol!r} with the {camera} camera")
    keypoints = np.asarray(keypoints_3d_mm, dtype=np.float64)
    lineage = _Lineage(
        checkpoint_sha256, geometry_sha256, camera_code_sha256() if camera == "native" else None
    )
    predictor = _as_predictor(model_or_predictor, cfg, device, keypoints)
    manifest, population_sha256 = load_population(split_root)
    targets = [f for f in manifest["frames"] if obj_id in f["objects"]]
    if not targets:
        return ArResult(
            obj_id,
            protocol_id,
            camera,
            "not_evaluable",
            population_sha256,
            0,
            0,
            checkpoint_sha256=lineage.checkpoint,
            geometry_sha256=lineage.geometry,
            camera_code_sha256=lineage.camera_code,
        )
    models = Path(models_dir)
    points = read_ply_vertices(str(models / f"obj_{obj_id:06d}.ply"))
    info = json.loads((models / "models_info.json").read_text())[str(obj_id)]
    run = _Run(
        predictor, _Scorer(points, toolkit_symmetry_transforms(info), keypoints), jitter, gt_box
    )
    loader = FrameLoader(manifest, obj_id, camera)
    rows = []
    for record in targets:
        frame = loader.load(record, single_target=protocol == "top1")
        if protocol == "top1":
            rows.append(run.score_top1(frame))
        else:
            indices = [i for i, o in enumerate(record["objects"]) if o == obj_id]
            rows.append(run.score_top5(frame, indices))
    arms, by_device = _summaries(rows, protocol, camera, info["diameter"])
    return ArResult(
        obj_id=obj_id,
        protocol=protocol_id,
        camera=camera,
        status="scored",
        population_sha256=population_sha256,
        frames=len(rows),
        gt_instances=sum(len(r.get("gt_indices", [0])) for r in rows),
        arms=arms,
        by_device=by_device,
        gtbox_keypoints=_gtbox_keypoints(rows) if gt_box else None,
        rows=tuple(rows),
        checkpoint_sha256=lineage.checkpoint,
        geometry_sha256=lineage.geometry,
        camera_code_sha256=lineage.camera_code,
    )
