# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export high-confidence two-camera DOPER poses as COCO pseudo-labels with QA overlays.

Replays a recorded two-camera capture (per-camera ``NNNNNN_color.png`` frames and
``intrinsics.json``, plus an ``R_AB``/``t_AB`` rig ``--extrinsic`` in camera A's frame), runs the
model on each view and solves one joint multiview PnP pose per frame.

A frame pair is *accepted* when the joint solve succeeds, the per-camera median reprojection
error is below ``--max-reproj-px`` in **both** views, and each view has more than
``--min-inliers-per-camera`` RANSAC inliers. For each accepted pair the fused object pose is
projected through ``keypoints_3d.json`` into both cameras to synthesize per-view keypoint and box
pseudo-labels (optionally a projected cuboid or CAD-mesh mask), and a side-by-side overlay is
written for manual QA. With ``--also-ir-left`` (default) the pose is also transformed into each
camera's left-IR frame (``ir_left_intrinsics`` and ``T_color_irleft`` from ``intrinsics.json``)
for every ``NNNNNN_ir_left.png`` frame.

Output (``--out-dir``)::

    images/A_000123.png, images/B_000123.png     copied real frames
    masks/A_000123_mask.png                      optional masks
    overlays/000123.jpg                          projected keypoints, boxes and reprojection
    annotations.json                             COCO (both views, keypoints + bbox)
    summary.json                                 acceptance statistics and settings

The COCO ``file_name`` fields are relative to ``images/``. Annotations carry
``"pseudo_label": true`` and the per-frame multiview reprojection errors for later re-filtering.

Example::

    python -m doper.pseudo_labels.multiview \
        --cam-a-dir capture/cam_a --cam-b-dir capture/cam_b \
        --extrinsic capture/rig_extrinsic.json --checkpoint runs/best.pth \
        --keypoints-3d data/keypoints_3d.json --out-dir pseudo_labels/capture
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from doper.box_dither import (
    PROPOSAL_OFFSETS,
    baseline_candidate,
    cross_view_metrics,
    infer_boxes,
    proposals,
)
from doper.config import DataConfig, DoperConfig, ModelConfig
from doper.console import log
from doper.evaluation.io import write_json
from doper.geometry_vis import compute_3d_bbox, draw_axes, draw_box3d, project_points
from doper.infer import Detection, infer_keypoints_in_box, load_model
from doper.multiview import (
    AssociationGate,
    CameraView,
    DetectionPair,
    InferSettings,
    RigidTransform,
    associate_detection_pairs,
    infer_all,
    invert_rt,
    single_view_inliers,
)
from doper.performance import resolve_device
from doper.pnp import PNP_METHODS
from doper.pseudo_labels._outdir import check_output_dir, replace_output_dir
from doper.pseudo_labels.masks import (
    bbox_from_mask,
    bbox_from_points,
    load_binary_stl_triangles,
    projected_cuboid_mask,
    projected_mesh_mask,
)
from doper.recorded import (
    frame_index,
    intrinsics_block_to_arrays,
    list_color_frames,
    load_intrinsics,
)

_MODEL_DEFAULTS = ModelConfig()
VIEWS = ("A", "B")
GREEN = (0, 220, 0)
AMBER = (0, 210, 255)
SAM3_COLOR = (255, 255, 0)
MIN_DEPTH_M = 0.03
MAX_DEPTH_M = 1.0
VISIBLE_CONFIDENT = 2
VISIBLE_UNCERTAIN = 1
KEYPOINT_DECIMALS = 3
REPROJECTION_DECIMALS = 4
POSE_DECIMALS = 8
AREA_DECIMALS = 2
FRACTION_DECIMALS = 4
MASK_KEEP = 0.65
MASK_TINT = 0.35
KEYPOINT_RADIUS = 2
JPEG_QUALITY = 92
TITLE_ORIGIN = (10, 24)
TITLE_SCALE = 0.6
TITLE_THICKNESS = 2
SAM3_LABEL_MIN_Y = 18
SAM3_LABEL_OFFSET = 5
SAM3_LABEL_SCALE = 0.5
SAM3_BOX_THICKNESS = 2
HOMOGENEOUS_ROW = (0.0, 0.0, 0.0, 1.0)
HOMOGENEOUS_ATOL = 1.0e-8
MESH_MASK_TYPE = "projected_mesh_triangle_silhouette"
MESH_MASK_GEOMETRY = "projected_cad_triangle_silhouette"
CUBOID_MASK_TYPE = "projected_3d_cuboid_convex_silhouette"
DITHER_FIELDS = ("scale", "shift_x_frac", "shift_y_frac")
DITHER_LABELS = ("candidate_a", "candidate_b")


def _build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser of ``main``."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Architecture options are fallbacks for checkpoints without cfg metadata.",
    )
    parser.add_argument("--cam-a-dir", required=True)
    parser.add_argument("--cam-b-dir", required=True)
    parser.add_argument("--extrinsic", required=True, help="rig json with R_AB / t_AB (frame A)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--keypoints-3d", required=True)
    parser.add_argument("--intrinsics-a", default=None)
    parser.add_argument("--intrinsics-b", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--force", action="store_true", help="overwrite an existing out-dir")
    for view in ("a", "b"):
        parser.add_argument(
            f"--sam3-results-{view}",
            default=None,
            help=f"optional SAM3 results.json for cam {view.upper()}; uses its per-frame "
            "accepted bbox instead of detection",
        )
    parser.add_argument(
        "--min-sam3-score",
        type=float,
        default=0.0,
        help="minimum accepted SAM3 instance score when external SAM3 boxes are used",
    )
    parser.add_argument("--backbone", default=_MODEL_DEFAULTS.backbone)
    parser.add_argument("--num-keypoints", type=int, default=_MODEL_DEFAULTS.num_keypoints)
    parser.add_argument("--num-classes", type=int, default=_MODEL_DEFAULTS.num_classes)
    parser.add_argument("--det-size", type=int, default=_MODEL_DEFAULTS.det_input_size)
    parser.add_argument(
        "--letterbox", action=argparse.BooleanOptionalAction, default=_MODEL_DEFAULTS.letterbox
    )
    parser.add_argument("--crop-size", type=int, default=_MODEL_DEFAULTS.crop_size)
    parser.add_argument("--crop-pad-factor", type=float, default=_MODEL_DEFAULTS.crop_pad_factor)
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--kp-score-thr", type=float, default=0.3)
    parser.add_argument("--max-det", type=int, default=5)
    parser.add_argument("--box-pad", type=float, default=0.0)
    parser.add_argument("--rot-tta", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--joint-min-kp", type=int, default=4)
    parser.add_argument("--joint-ransac-px", type=float, default=8.0)
    parser.add_argument(
        "--pnp-method",
        choices=PNP_METHODS,
        default="usac",
        help="PnP RANSAC method: usac (default) or epnp (the paper's labels)",
    )
    parser.add_argument(
        "--box-dither-proposals",
        type=int,
        default=1,
        choices=range(1, len(PROPOSAL_OFFSETS) + 1),
        metavar=f"[1-{len(PROPOSAL_OFFSETS)}]",
        help="deterministic crop proposals per camera; 1 disables dithering",
    )
    parser.add_argument(
        "--box-dither-top-k",
        type=int,
        default=3,
        help="jointly solve this many bilateral pairs ranked by single-view "
        "cross-camera CAD agreement",
    )
    parser.add_argument(
        "--max-reproj-px",
        type=float,
        default=2.0,
        help="accept a pair only if BOTH cameras' joint reproj < this (px)",
    )
    parser.add_argument(
        "--min-inliers-per-camera",
        type=int,
        default=0,
        help="accept only when each camera has more than this many RANSAC inliers",
    )
    parser.add_argument(
        "--frame-ids",
        default=None,
        help="optional comma-separated frame IDs to process (for example 12,14,15)",
    )
    parser.add_argument(
        "--bbox-pad-frac",
        type=float,
        default=0.05,
        help="pad the projected-keypoint bbox by this fraction of w/h",
    )
    parser.add_argument(
        "--filled-projected-box-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use the projected 3D cuboid's convex silhouette as a filled mask",
    )
    parser.add_argument(
        "--mesh-mask-stl",
        default=None,
        help="optional binary STL in the keypoint/object frame; rasterize its "
        "projected triangles into an object-silhouette mask",
    )
    parser.add_argument(
        "--category-name", default=None, help="defaults to the keypoints_3d 'object'"
    )
    parser.add_argument("--category-id", type=int, default=1)
    parser.add_argument("--axis-length", type=float, default=0.03)
    parser.add_argument(
        "--also-ir-left",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also emit left-IR pseudo-labels by projecting the fused pose through each "
        "camera's ir_left intrinsics + T_color_irleft (needs NNNNNN_ir_left.png + IR "
        "calibration in intrinsics.json)",
    )
    parser.add_argument("--device", default="cuda")
    return parser


def load_sam3_boxes(path: str | Path, min_score: float) -> dict[int, tuple[np.ndarray, float]]:
    """Load the highest-scoring accepted SAM3 box of each frame.

    Args:
        path: SAM3 ``results.json`` with ``frames[*].file_name`` and
            ``frames[*].accepted_instances[*].{bbox_xyxy, sam_score}``.
        min_score: Instances below this ``sam_score`` are ignored.

    Returns:
        Frame index to ``(bbox_xyxy float32 (4,), sam_score)``; frames without an instance at
        or above ``min_score`` are absent.
    """
    with open(path) as handle:
        payload = json.load(handle)
    boxes: dict[int, tuple[np.ndarray, float]] = {}
    for record in payload.get("frames", []):
        instances = [
            item
            for item in record.get("accepted_instances", [])
            if float(item.get("sam_score", 0.0)) >= min_score
        ]
        if not instances:
            continue
        best = max(instances, key=lambda item: float(item.get("sam_score", 0.0)))
        boxes[frame_index(record["file_name"])] = (
            np.asarray(best["bbox_xyxy"], dtype=np.float32),
            float(best["sam_score"]),
        )
    return boxes


def projected_to_coco(
    projected: np.ndarray, scores: np.ndarray, width: int, height: int, kp_thr: float
) -> tuple[list[float], int]:
    """Encode projected keypoints as a flat COCO ``[x, y, v, ...]`` list.

    Args:
        projected: ``(K, 2)`` projected keypoints in pixels.
        scores: Detected keypoint confidences; a missing score counts as zero.
        width: Image width in pixels.
        height: Image height in pixels.
        kp_thr: Keypoints in the image get visibility 2 at or above this score, else 1.
            Non-finite and out-of-image keypoints are written as ``0, 0, 0``.

    Returns:
        The flat keypoint list (coordinates rounded to three decimals) and the number of
        in-image keypoints.
    """
    flat: list[float] = []
    n_visible = 0
    for i, (x_value, y_value) in enumerate(projected):
        x = float(x_value)
        y = float(y_value)
        if not np.isfinite([x, y]).all() or x < 0 or x >= width or y < 0 or y >= height:
            flat.extend([0.0, 0.0, 0])
            continue
        score = float(scores[i]) if i < len(scores) else 0.0
        visibility = VISIBLE_CONFIDENT if score >= kp_thr else VISIBLE_UNCERTAIN
        flat.extend([round(x, KEYPOINT_DECIMALS), round(y, KEYPOINT_DECIMALS), visibility])
        n_visible += 1
    return flat, n_visible


def load_ir_left_calibration(intrinsics_path: str | Path) -> tuple[np.ndarray, ...]:
    """Read the left-IR intrinsics and color-to-IR transform from ``intrinsics.json``.

    Returns:
        ``(camera_matrix, dist_coeffs, T_color_irleft)``; the ``(4, 4)`` transform maps
        color-camera points into the left-IR camera frame.

    Raises:
        ValueError: The calibration is missing or the transform is not a finite rigid
            homogeneous matrix.
    """
    with open(intrinsics_path) as handle:
        payload = json.load(handle)
    if "ir_left_intrinsics" not in payload or "T_color_irleft" not in payload:
        raise ValueError(f"{intrinsics_path} needs ir_left_intrinsics and T_color_irleft")
    camera, distortion = intrinsics_block_to_arrays(payload["ir_left_intrinsics"])
    transform = np.asarray(payload["T_color_irleft"], dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{intrinsics_path} has an invalid T_color_irleft")
    if not np.allclose(transform[3], HOMOGENEOUS_ROW, atol=HOMOGENEOUS_ATOL):
        raise ValueError(f"{intrinsics_path} has an invalid T_color_irleft bottom row")
    return camera, distortion, transform


def color_pose_to_ir_left(
    rvec_color: np.ndarray, tvec_color: np.ndarray, transform_color_irleft: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Transform an object-to-color pose into the left-IR camera frame.

    Returns:
        ``(rvec, tvec)`` of the object in the left-IR frame, both ``(3, 1)``.
    """
    rotation_color_object, _ = cv2.Rodrigues(np.asarray(rvec_color, dtype=np.float64).reshape(3, 1))
    translation_color_object = np.asarray(tvec_color, dtype=np.float64).reshape(3, 1)
    transform = np.asarray(transform_color_irleft, dtype=np.float64)
    rotation_ir_color = transform[:3, :3]
    translation_ir_color = transform[:3, 3].reshape(3, 1)
    rotation_ir_object = rotation_ir_color @ rotation_color_object
    translation_ir_object = rotation_ir_color @ translation_color_object + translation_ir_color
    rvec_ir, _ = cv2.Rodrigues(rotation_ir_object)
    return rvec_ir.reshape(3, 1), translation_ir_object


def _add_mask_overlay(image: np.ndarray, mask: np.ndarray) -> None:
    """Tint a mask green in place while retaining the underlying pixels."""
    selected = mask > 0
    if not np.any(selected):
        return
    tint = np.asarray(GREEN, dtype=np.float32)
    image[selected] = np.clip(
        image[selected].astype(np.float32) * MASK_KEEP + tint * MASK_TINT, 0, 255
    ).astype(np.uint8)


def _draw_keypoints(image: np.ndarray, projected: np.ndarray) -> None:
    for px, py in projected:
        if np.isfinite([px, py]).all():
            center = (int(round(px)), int(round(py)))
            cv2.circle(image, center, KEYPOINT_RADIUS, GREEN, -1, cv2.LINE_AA)


def _draw_bbox(image: np.ndarray, bbox: Sequence[float]) -> None:
    x, y, w, h = [int(round(v)) for v in bbox]
    cv2.rectangle(image, (x, y), (x + w, y + h), AMBER, 1)


def _side_by_side(left: np.ndarray, right: np.ndarray, title: str) -> np.ndarray:
    """Stack two overlays horizontally (right resized to the left height) under a title."""
    if right.shape[0] != left.shape[0]:
        scale = left.shape[0] / right.shape[0]
        right = cv2.resize(right, (int(right.shape[1] * scale), left.shape[0]))
    combo = np.hstack([left, right])
    cv2.putText(combo, title, TITLE_ORIGIN, cv2.FONT_HERSHEY_SIMPLEX, TITLE_SCALE, GREEN,
                TITLE_THICKNESS, cv2.LINE_AA)  # fmt: skip
    return combo


def _write_jpeg(path: str, image: np.ndarray) -> None:
    cv2.imwrite(path, image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])


def _round_vector(values: np.ndarray) -> list[float]:
    return [round(float(v), POSE_DECIMALS) for v in np.asarray(values).reshape(3)]


@dataclass
class _Exporter:
    """Loaded inputs and running output state of one export run."""

    args: argparse.Namespace
    model: Any
    cfg: DoperConfig
    device: str
    keypoints_3d: np.ndarray
    views: dict[str, CameraView]
    a_to_b: RigidTransform
    transform_b_from_a: np.ndarray
    frames: dict[str, dict[int, Path]]
    out_dir: str
    mesh_triangles: np.ndarray | None = None
    sam3_boxes: dict[str, dict[int, tuple[np.ndarray, float]]] | None = None
    ir_calibrations: dict[str, tuple[np.ndarray, ...] | None] = field(default_factory=dict)
    images: list[dict[str, Any]] = field(default_factory=list)
    annotations: list[dict[str, Any]] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    accepted_ids: list[int] = field(default_factory=list)
    box3d: np.ndarray = field(init=False)
    gate: AssociationGate = field(init=False)
    settings: InferSettings = field(init=False)

    def __post_init__(self) -> None:
        self.box3d = compute_3d_bbox(self.keypoints_3d)
        self.gate = AssociationGate(
            self.args.kp_score_thr, self.args.joint_min_kp, self.args.joint_ransac_px,
            self.args.max_reproj_px, MIN_DEPTH_M, MAX_DEPTH_M, self.args.pnp_method,
        )  # fmt: skip
        self.settings = InferSettings(
            self.args.score_thr, self.args.max_det, self.args.box_pad, self.args.rot_tta,
            MIN_DEPTH_M, MAX_DEPTH_M,
        )  # fmt: skip
        for key in ("total", "accepted", "dither_attempted", "dither_selected"):
            self.counts.setdefault(key, 0)

    @property
    def masked(self) -> bool:
        """Whether masks are written."""
        return self.mesh_triangles is not None or bool(self.args.filled_projected_box_mask)

    def detect(self, view: str, image: np.ndarray, frame: int) -> list[Detection]:
        """Detections of one view: inside the SAM3 box if given, else by the detector."""
        camera = self.views[view]
        if self.sam3_boxes is not None:
            box = self.sam3_boxes[view][frame][0]
            return [
                infer_keypoints_in_box(
                    self.model, image, self.cfg, self.device, box, self.keypoints_3d,
                    camera.camera_matrix, camera.dist_coeffs, box_pad=self.args.box_pad,
                )
            ]  # fmt: skip
        return infer_all(
            self.model, image, self.cfg, self.device, self.settings, camera, self.keypoints_3d
        )

    def associate(
        self, detections_a: list[Detection], detections_b: list[Detection]
    ) -> DetectionPair | None:
        """Best joint-PnP pairing of the two views' detections."""
        return associate_detection_pairs(
            detections_a, detections_b, self.keypoints_3d, self.views["A"], self.views["B"],
            self.a_to_b, self.gate,
        )  # fmt: skip

    def inliers(self, view: str, detection: Detection) -> int:
        """Single-view RANSAC inliers among the confident keypoints of ``detection``."""
        camera = self.views[view]
        return int(
            single_view_inliers(
                self.keypoints_3d, detection.keypoints.astype(np.float64),
                detection.keypoint_scores >= self.args.kp_score_thr,
                camera.camera_matrix, camera.dist_coeffs, self.args.joint_ransac_px,
                self.args.pnp_method,
            ).sum()
        )  # fmt: skip


def _association_rank(association: DetectionPair | None) -> tuple[float, ...]:
    """Sort key: accepted first, then lower worst and summed reprojection, more observations."""
    if association is None:
        return (1, float("inf"), float("inf"), 0)
    worst = max(association.reprojection_a, association.reprojection_b)
    return (
        0 if association.accepted else 1,
        worst,
        association.reprojection_a + association.reprojection_b,
        -association.observations,
    )


def _candidate_detection(candidate: dict[str, Any], detector_score: float) -> Detection:
    return Detection(
        bbox_xyxy=np.asarray(candidate["box_xyxy"], dtype=np.float32),
        score=float(detector_score),
        keypoints=np.asarray(candidate["keypoints"], dtype=np.float32),
        keypoint_scores=np.asarray(candidate["keypoint_scores"], dtype=np.float32),
        rvec=candidate["rvec"],
        tvec=candidate["tvec"],
        reproj_error_px=candidate["pnp_reprojection_px"],
    )


def _recover_with_box_dither(
    run: _Exporter, images: dict[str, np.ndarray], baseline: DetectionPair | None
) -> tuple[DetectionPair | None, dict[str, Any] | None]:
    """Re-solve the pair on dithered crops; return the best association and its dither info.

    The baseline pair competes with the ``--box-dither-top-k`` dithered pairs that agree best
    across cameras as single-view poses; the metadata is None when the baseline wins.
    """
    args = run.args
    if args.box_dither_proposals <= 1 or baseline is None:
        return baseline, None
    baseline_detections = {"A": baseline.detection_a, "B": baseline.detection_b}
    candidate_sets = []
    for view in VIEWS:
        detection = baseline_detections[view]
        proposed = proposals(detection.bbox_xyxy)[: args.box_dither_proposals]
        additional = infer_boxes(
            run.model, images[view], run.cfg, run.device, proposed[1:], run.keypoints_3d,
            run.views[view].camera_matrix, batch_size=max(1, len(proposed) - 1),
            dist_coeffs=run.views[view].dist_coeffs,
        )  # fmt: skip
        candidate_sets.append([baseline_candidate(detection), *additional])
    ranked_pairs = []
    for index_a, candidate_a in enumerate(candidate_sets[0]):
        for index_b, candidate_b in enumerate(candidate_sets[1]):
            agreement = cross_view_metrics(
                candidate_a, candidate_b, run.transform_b_from_a, run.keypoints_3d
            )
            rms = agreement["cad_keypoint_rms_mm"]
            if rms is not None:
                ranked_pairs.append((float(rms), index_a, index_b, agreement))
    ranked_pairs.sort(key=lambda item: item[0])
    solved: list[tuple[DetectionPair | None, dict[str, Any] | None]] = [(baseline, None)]
    for _, index_a, index_b, agreement in ranked_pairs[: max(1, args.box_dither_top_k)]:
        candidate_a = candidate_sets[0][index_a]
        candidate_b = candidate_sets[1][index_b]
        association = run.associate(
            [_candidate_detection(candidate_a, baseline.detection_a.score)],
            [_candidate_detection(candidate_b, baseline.detection_b.score)],
        )
        metadata = {
            "index_a": index_a,
            "index_b": index_b,
            "candidate_a": {key: candidate_a[key] for key in DITHER_FIELDS},
            "candidate_b": {key: candidate_b[key] for key in DITHER_FIELDS},
            "single_view_agreement": agreement,
        }
        solved.append((association, metadata))
    return min(solved, key=lambda item: _association_rank(item[0]))


@dataclass
class _ViewLabel:
    """Projected labels of one accepted color view."""

    view: str
    image: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    reprojection: float
    inliers: int
    keypoint_scores: np.ndarray
    projected: np.ndarray
    box_projected: np.ndarray
    keypoints: list[float]
    n_visible: int
    bbox: list[float]
    mask: np.ndarray | None


def _view_label(
    run: _Exporter, view: str, image: np.ndarray, pair: DetectionPair, inliers: int
) -> _ViewLabel:
    """Project the fused pose into one view and build its keypoints, box and mask."""
    args = run.args
    camera = run.views[view]
    detection = pair.detection_a if view == "A" else pair.detection_b
    rvec, tvec = (pair.rvec_a, pair.tvec_a) if view == "A" else (pair.rvec_b, pair.tvec_b)
    height, width = image.shape[:2]
    pose_args = (rvec, tvec, camera.camera_matrix, camera.dist_coeffs)
    projected = project_points(run.keypoints_3d, *pose_args)
    box_projected = project_points(run.box3d, *pose_args)
    keypoints, n_visible = projected_to_coco(
        projected, detection.keypoint_scores, width, height, args.kp_score_thr
    )
    bbox_points = box_projected if args.filled_projected_box_mask else projected
    bbox = bbox_from_points(bbox_points, width, height, args.bbox_pad_frac)
    mask = None
    if run.mesh_triangles is not None:
        mask = projected_mesh_mask(run.mesh_triangles, *pose_args, width, height)
        bbox = bbox_from_mask(mask, args.bbox_pad_frac)
    elif args.filled_projected_box_mask:
        mask = projected_cuboid_mask(box_projected, width, height)
    reprojection = pair.reprojection_a if view == "A" else pair.reprojection_b
    return _ViewLabel(view, image, rvec, tvec, reprojection, inliers, detection.keypoint_scores,
                      projected, box_projected, keypoints, n_visible, bbox, mask)  # fmt: skip


def _append_record(run: _Exporter, image: dict[str, Any], annotation: dict[str, Any]) -> None:
    """Append an image and its annotation with the next shared id."""
    image_id = len(run.images)
    run.images.append({"id": image_id, **image})
    run.annotations.append({"id": image_id, "image_id": image_id, **annotation})


def _label_fields(
    run: _Exporter, label: _ViewLabel, frame: int, modality: str, rvec: np.ndarray, tvec: np.ndarray
) -> dict[str, Any]:
    """Annotation fields shared by the color and left-IR labels of one view."""
    return {
        "category_id": run.args.category_id, "iscrowd": 0, "pseudo_label": True,
        "camera_label": label.view, "frame_id": int(frame), "modality": modality,
        "multiview_reproj_px": round(float(label.reprojection), REPROJECTION_DECIMALS),
        "rvec": _round_vector(rvec), "tvec": _round_vector(tvec),
        "ransac_inliers": int(label.inliers),
    }  # fmt: skip


def _image_fields(file_name: str, width: int, height: int, frame: int, view: str,
                  modality: str) -> dict[str, Any]:  # fmt: skip
    return {
        "file_name": file_name, "width": int(width), "height": int(height),
        "frame_id": int(frame), "camera_label": view, "view": "real_pseudo",
        "modality": modality,
    }  # fmt: skip


def _write_mask(run: _Exporter, file_name: str, mask: np.ndarray) -> str:
    """Write ``mask`` next to ``file_name`` under ``masks/``; return its relative path."""
    mask_name = f"{os.path.splitext(file_name)[0]}_mask.png"
    cv2.imwrite(os.path.join(run.out_dir, "masks", mask_name), mask)
    return f"masks/{mask_name}"


def _write_color_label(
    run: _Exporter, label: _ViewLabel, frame: int, dither: dict[str, Any] | None
) -> None:
    file_name = f"{label.view}_{frame:06d}.png"
    cv2.imwrite(os.path.join(run.out_dir, "images", file_name), label.image)
    height, width = label.image.shape[:2]
    bbox = label.bbox
    annotation = {
        **_label_fields(run, label, frame, "color", label.rvec, label.tvec),
        "keypoints": label.keypoints,
        "num_keypoints": int(label.n_visible),
        "bbox": bbox,
        "area": (
            int(np.count_nonzero(label.mask))
            if label.mask is not None
            else round(float(bbox[2] * bbox[3]), AREA_DECIMALS)
        ),
        "box_dither_selected": dither is not None,
    }
    if dither is not None:
        annotation["box_dither"] = dither[DITHER_LABELS[VIEWS.index(label.view)]]
    if label.mask is not None:
        annotation["mask_file"] = _write_mask(run, file_name, label.mask)
        annotation["mask_type"] = (
            MESH_MASK_TYPE if run.mesh_triangles is not None else CUBOID_MASK_TYPE
        )
    _append_record(
        run, _image_fields(file_name, width, height, frame, label.view, "color"), annotation
    )


def _write_ir_label(run: _Exporter, label: _ViewLabel, frame: int) -> np.ndarray | None:
    """Write the left-IR image and label of one view; return its overlay, or None if skipped."""
    calibration = run.ir_calibrations.get(label.view)
    if not run.args.also_ir_left or calibration is None:
        return None
    ir_path = str(run.frames[label.view][frame]).replace("_color.png", "_ir_left.png")
    ir_image = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    if ir_image is None:
        return None
    args = run.args
    height, width = ir_image.shape[:2]
    camera_matrix, dist_coeffs, transform = calibration
    rvec, tvec = color_pose_to_ir_left(label.rvec, label.tvec, transform)
    projected = project_points(run.keypoints_3d, rvec, tvec, camera_matrix, dist_coeffs)
    box_projected = project_points(run.box3d, rvec, tvec, camera_matrix, dist_coeffs)
    keypoints, n_visible = projected_to_coco(
        projected, label.keypoint_scores, width, height, args.kp_score_thr
    )
    bbox_points = box_projected if args.filled_projected_box_mask else projected
    bbox = bbox_from_points(bbox_points, width, height, args.bbox_pad_frac)
    file_name = f"{label.view}_{frame:06d}_ir_left.png"
    cv2.imwrite(os.path.join(run.out_dir, "images", file_name), ir_image)
    mask = None
    if args.filled_projected_box_mask:
        mask = projected_cuboid_mask(box_projected, width, height)
    overlay = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
    if mask is not None:
        _add_mask_overlay(overlay, mask)
    _draw_keypoints(overlay, projected)
    draw_box3d(overlay, box_projected, color=GREEN)
    _draw_bbox(overlay, bbox)
    annotation = {
        **_label_fields(run, label, frame, "ir_left", rvec, tvec),
        "keypoints": keypoints,
        "num_keypoints": int(n_visible),
        "bbox": bbox,
        "area": round(float(bbox[2] * bbox[3]), AREA_DECIMALS),
        "reprojection_source_modality": "color",
    }
    if mask is not None:
        annotation["mask_file"] = _write_mask(run, file_name, mask)
    _append_record(
        run, _image_fields(file_name, width, height, frame, label.view, "ir_left"), annotation
    )
    return overlay


def _color_overlay(run: _Exporter, label: _ViewLabel, frame: int) -> np.ndarray:
    camera = run.views[label.view]
    overlay = label.image.copy()
    if label.mask is not None:
        _add_mask_overlay(overlay, label.mask)
    _draw_keypoints(overlay, label.projected)
    draw_box3d(overlay, label.box_projected, color=GREEN)
    draw_axes(overlay, label.rvec, label.tvec, camera.camera_matrix, camera.dist_coeffs,
              run.args.axis_length)  # fmt: skip
    _draw_bbox(overlay, label.bbox)
    if run.sam3_boxes is not None:
        sam_box, sam_score = run.sam3_boxes[label.view][frame]
        sx1, sy1, sx2, sy2 = [int(round(v)) for v in sam_box]
        cv2.rectangle(overlay, (sx1, sy1), (sx2, sy2), SAM3_COLOR, SAM3_BOX_THICKNESS)
        origin = (sx1, max(SAM3_LABEL_MIN_Y, sy1 - SAM3_LABEL_OFFSET))
        cv2.putText(overlay, f"SAM3 {sam_score:.3f}", origin, cv2.FONT_HERSHEY_SIMPLEX,
                    SAM3_LABEL_SCALE, SAM3_COLOR, 1, cv2.LINE_AA)  # fmt: skip
    return overlay


def _accepted_pair(
    run: _Exporter, images: dict[str, np.ndarray], frame: int
) -> tuple[DetectionPair, dict[str, Any] | None, dict[str, int]] | None:
    """Detect, associate and gate one frame pair; return the accepted pair or None."""
    args = run.args
    detections = {view: run.detect(view, images[view], frame) for view in VIEWS}
    baseline = run.associate(detections["A"], detections["B"])
    if args.box_dither_proposals > 1 and baseline is not None:
        run.counts["dither_attempted"] += 1
    pair, dither = _recover_with_box_dither(run, images, baseline)
    if pair is None or not pair.accepted:
        return None
    if dither is not None:
        run.counts["dither_selected"] += 1
    paired = {"A": pair.detection_a, "B": pair.detection_b}
    for detection in paired.values():
        confident = int((detection.keypoint_scores >= args.kp_score_thr).sum())
        if confident < args.joint_min_kp:
            return None
    inliers = {view: run.inliers(view, paired[view]) for view in VIEWS}
    if min(inliers.values()) <= args.min_inliers_per_camera:
        return None
    return pair, dither, inliers


def _export_frame(run: _Exporter, frame: int) -> None:
    """Export one synchronized frame pair if it passes every gate."""
    read = {view: cv2.imread(str(run.frames[view][frame]), cv2.IMREAD_COLOR) for view in VIEWS}
    images = {view: image for view, image in read.items() if image is not None}
    if len(images) != len(VIEWS):
        return
    run.counts["total"] += 1
    accepted = _accepted_pair(run, images, frame)
    if accepted is None:
        return
    pair, dither, inliers = accepted
    run.counts["accepted"] += 1
    run.accepted_ids.append(int(frame))
    labels = [_view_label(run, view, images[view], pair, inliers[view]) for view in VIEWS]
    ir_overlays = {}
    for label in labels:
        _write_color_label(run, label, frame, dither)
        ir_overlay = _write_ir_label(run, label, frame)
        if ir_overlay is not None:
            ir_overlays[label.view] = ir_overlay
    overlays = [_color_overlay(run, label, frame) for label in labels]
    title = (
        f"frame {frame:06d}  reproj A={pair.reprojection_a:.2f} B={pair.reprojection_b:.2f}px  "
        f"inliers={inliers['A']}/{inliers['B']}"
    )
    overlay_dir = os.path.join(run.out_dir, "overlays")
    _write_jpeg(
        os.path.join(overlay_dir, f"{frame:06d}.jpg"),
        _side_by_side(overlays[0], overlays[1], title),
    )
    if all(view in ir_overlays for view in VIEWS):
        title = f"frame {frame:06d}  IR-left  (pose from color multiview)"
        combo = _side_by_side(ir_overlays["A"], ir_overlays["B"], title)
        _write_jpeg(os.path.join(overlay_dir, f"{frame:06d}_ir_left.jpg"), combo)


def _load_rig(path: str) -> tuple[RigidTransform, np.ndarray]:
    """Return the camera A-to-B transform and the ``(4, 4)`` B-from-A matrix of a rig JSON."""
    with open(path) as handle:
        extrinsic = json.load(handle)
    rotation_ab = np.asarray(extrinsic["R_AB"], dtype=np.float64)
    translation_ab = np.asarray(extrinsic["t_AB"], dtype=np.float64).reshape(3, 1)
    rvec_ab, _ = cv2.Rodrigues(rotation_ab)
    rvec_ba, tvec_ba = invert_rt(rvec_ab.reshape(3, 1), translation_ab)
    transform_a_from_b = np.eye(4, dtype=np.float64)
    transform_a_from_b[:3, :3] = rotation_ab
    transform_a_from_b[:3, 3] = translation_ab.reshape(3)
    return RigidTransform(rvec_ba, tvec_ba), np.linalg.inv(transform_a_from_b)


def _load_ir_calibrations(
    args: argparse.Namespace, intrinsics: dict[str, str]
) -> dict[str, tuple[np.ndarray, ...] | None]:
    """Left-IR calibration of every view that has IR frames (``--also-ir-left``)."""
    calibrations: dict[str, tuple[np.ndarray, ...] | None] = dict.fromkeys(VIEWS)
    if not args.also_ir_left:
        return calibrations
    for view, camera_dir in zip(VIEWS, (args.cam_a_dir, args.cam_b_dir), strict=True):
        if not glob.glob(os.path.join(camera_dir, "*_ir_left.png")):
            continue
        try:
            calibrations[view] = load_ir_left_calibration(intrinsics[view])
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise SystemExit(f"cannot export {view} left-IR frames: {error}") from error
    calibrated = [view for view, value in calibrations.items() if value is not None]
    log(
        "--also-ir-left: projecting color-frame poses through T_color_irleft + IR intrinsics "
        f"for views {calibrated or ['none']}; missing IR frames are skipped."
    )
    return calibrations


def _select_frames(
    args: argparse.Namespace,
    frames: dict[str, dict[int, Path]],
    sam3_boxes: dict[str, dict[int, tuple[np.ndarray, float]]] | None,
) -> tuple[list[int], set[int] | None]:
    """Synchronized frame indices to process, and the ``--frame-ids`` request (or None)."""
    indices = sorted(set(frames["A"]) & set(frames["B"]))
    if sam3_boxes is not None:
        indices = sorted(set(indices) & set(sam3_boxes["A"]) & set(sam3_boxes["B"]))
        log(
            f"SAM3 bbox mode: {len(sam3_boxes['A'])} cam-A and {len(sam3_boxes['B'])} cam-B "
            f"frames; {len(indices)} synchronized pairs"
        )
    requested = None
    if args.frame_ids:
        requested = {int(value.strip()) for value in args.frame_ids.split(",") if value.strip()}
        indices = [index for index in indices if index in requested]
    if not indices:
        raise SystemExit("no common frame indices between the two cameras")
    return indices, requested


def _input_paths(args: argparse.Namespace) -> list[str | None]:
    """Every input file or directory of a run, for the output-overlap check."""
    return [
        args.cam_a_dir, args.cam_b_dir, args.extrinsic, args.checkpoint, args.keypoints_3d,
        args.intrinsics_a, args.intrinsics_b, args.mesh_mask_stl, args.sam3_results_a,
        args.sam3_results_b,
    ]  # fmt: skip


def _prepare_out_dir(out_dir: str, args: argparse.Namespace, masked: bool) -> None:
    """Create the output tree, replacing an existing ``out_dir`` only with ``--force``."""
    replace_output_dir(out_dir, args.force, _input_paths(args))
    os.makedirs(os.path.join(out_dir, "images"))
    os.makedirs(os.path.join(out_dir, "overlays"))
    if masked:
        os.makedirs(os.path.join(out_dir, "masks"))


def _build_config(args: argparse.Namespace) -> DoperConfig:
    """Fallback config from the architecture options (checkpoint metadata overrides it)."""
    model_cfg = ModelConfig(
        backbone=args.backbone, pretrained=False, num_classes=args.num_classes,
        num_keypoints=args.num_keypoints, det_input_size=args.det_size, crop_size=args.crop_size,
        letterbox=args.letterbox, crop_pad_factor=args.crop_pad_factor,
    )  # fmt: skip
    return DoperConfig(data=DataConfig(root="."), model=model_cfg)


def _mask_geometry(run: _Exporter) -> str | None:
    if run.mesh_triangles is not None:
        return MESH_MASK_GEOMETRY
    return CUBOID_MASK_TYPE if run.args.filled_projected_box_mask else None


def _write_outputs(run: _Exporter, category_name: str, keypoint_names: list[str],
                   requested: set[int] | None) -> None:  # fmt: skip
    """Write ``annotations.json`` and ``summary.json`` and log the totals."""
    args = run.args
    mesh = os.path.abspath(args.mesh_mask_stl) if args.mesh_mask_stl else None
    common = {
        "max_reproj_px": args.max_reproj_px,
        "min_inliers_per_camera_exclusive": args.min_inliers_per_camera,
        "mask_geometry": _mask_geometry(run),
        "mask_mesh": mesh,
        "checkpoint": os.path.abspath(args.checkpoint),
    }
    coco = {
        "info": {
            **common,
            "description": "DOPER multiview pseudo-labels (joint PnP, both-view reproj gated).",
            "keypoints_3d": os.path.abspath(args.keypoints_3d),
            "extrinsic": os.path.abspath(args.extrinsic),
            "ir_left_geometry": (
                "object-to-color pose transformed by T_color_irleft, then projected with "
                "ir_left_intrinsics"
                if args.also_ir_left
                else None
            ),
        },
        "images": run.images,
        "annotations": run.annotations,
        "categories": [
            {"id": args.category_id, "name": category_name, "keypoints": keypoint_names,
             "skeleton": []}
        ],
    }  # fmt: skip
    write_json(Path(run.out_dir) / "annotations.json", coco)
    counts = run.counts
    n_color = sum(1 for image in run.images if image.get("modality") == "color")
    n_ir = sum(1 for image in run.images if image.get("modality") == "ir_left")
    sam3 = bool(args.sam3_results_a)
    summary = {
        **common,
        "out_dir": run.out_dir,
        "frames_total": counts["total"],
        "frames_accepted": counts["accepted"],
        "requested_frame_ids": sorted(requested) if requested is not None else None,
        "accepted_frame_ids": run.accepted_ids,
        "accepted_fraction": round(counts["accepted"] / max(counts["total"], 1), 4),
        "images_written": len(run.images),
        "annotations_written": len(run.annotations),
        "color_images": n_color,
        "ir_left_images": n_ir,
        "also_ir_left": bool(args.also_ir_left),
        "ir_left_projection": (
            "T_color_irleft_then_ir_left_intrinsics" if args.also_ir_left else None
        ),
        "box_dither_proposals": args.box_dither_proposals,
        "box_dither_top_k": args.box_dither_top_k,
        "box_dither_attempted": counts["dither_attempted"],
        "box_dither_selected": counts["dither_selected"],
        "mask_files": sum(1 for ann in run.annotations if ann.get("mask_file")),
        "cam_a_dir": os.path.abspath(args.cam_a_dir),
        "cam_b_dir": os.path.abspath(args.cam_b_dir),
        "sam3_results_a": os.path.abspath(args.sam3_results_a) if sam3 else None,
        "sam3_results_b": os.path.abspath(args.sam3_results_b) if sam3 else None,
        "min_sam3_score": args.min_sam3_score if sam3 else None,
    }
    write_json(Path(run.out_dir) / "summary.json", summary)
    total = max(counts["total"], 1)
    log(f"accepted {counts['accepted']}/{counts['total']} pairs "
        f"({100 * counts['accepted'] / total:.1f}%)")  # fmt: skip
    log(f"images: {len(run.images)} (color={n_color} ir_left={n_ir})  "
        f"annotations: {len(run.annotations)}")  # fmt: skip
    log(f"COCO:     {os.path.join(run.out_dir, 'annotations.json')}")
    log(f"overlays: {os.path.join(run.out_dir, 'overlays')}")
    log(f"summary:  {os.path.join(run.out_dir, 'summary.json')}")


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point of the multiview pseudo-label exporter.

    Args:
        argv: CLI argument strings excluding the program name, or None to read
            ``sys.argv[1:]``.

    Returns:
        Zero after the export completed.

    Raises:
        SystemExit: Invalid arguments, an existing ``--out-dir`` without ``--force``, missing
            left-IR calibration, or no synchronized frames.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.mesh_mask_stl and args.filled_projected_box_mask:
        parser.error("--mesh-mask-stl and --filled-projected-box-mask are mutually exclusive")
    if bool(args.sam3_results_a) != bool(args.sam3_results_b):
        parser.error("--sam3-results-a and --sam3-results-b must be provided together")
    out_dir = os.path.abspath(args.out_dir)
    check_output_dir(out_dir, args.force, _input_paths(args))
    device = resolve_device(args.device)
    with open(args.keypoints_3d) as handle:
        keypoint_payload = json.load(handle)
    keypoints_3d = np.asarray(keypoint_payload["keypoints_3d"], dtype=np.float64)
    mesh_triangles = None
    if args.mesh_mask_stl:
        mesh_triangles = load_binary_stl_triangles(args.mesh_mask_stl)
        log(f"mesh masks: {args.mesh_mask_stl} ({len(mesh_triangles)} triangles)")
    keypoint_names = keypoint_payload.get(
        "keypoint_names", [f"kp_{i:03d}" for i in range(len(keypoints_3d))]
    )
    category_name = args.category_name or keypoint_payload.get("object", "object")
    a_to_b, transform_b_from_a = _load_rig(args.extrinsic)
    intrinsics = {
        "A": args.intrinsics_a or os.path.join(args.cam_a_dir, "intrinsics.json"),
        "B": args.intrinsics_b or os.path.join(args.cam_b_dir, "intrinsics.json"),
    }
    views = {view: CameraView(*load_intrinsics(intrinsics[view])) for view in VIEWS}
    ir_calibrations = _load_ir_calibrations(args, intrinsics)
    cfg = _build_config(args)
    model = load_model(args.checkpoint, cfg, device)
    frames = {
        view: {frame_index(path): path for path in list_color_frames(Path(directory))}
        for view, directory in zip(VIEWS, (args.cam_a_dir, args.cam_b_dir), strict=True)
    }
    sam3_boxes = None
    if args.sam3_results_a:
        sam3_boxes = {
            view: load_sam3_boxes(path, args.min_sam3_score)
            for view, path in zip(VIEWS, (args.sam3_results_a, args.sam3_results_b), strict=True)
        }
    indices, requested = _select_frames(args, frames, sam3_boxes)
    run = _Exporter(
        args, model, cfg, device, keypoints_3d, views, a_to_b, transform_b_from_a, frames,
        out_dir, mesh_triangles, sam3_boxes, ir_calibrations,
    )  # fmt: skip
    _prepare_out_dir(out_dir, args, run.masked)
    log(
        f"{len(indices)} synchronized pairs; accepting joint reproj < {args.max_reproj_px}px "
        f"and > {args.min_inliers_per_camera} inliers in both cams"
    )
    for frame in indices:
        _export_frame(run, frame)
    _write_outputs(run, category_name, keypoint_names, requested)
    return 0


if __name__ == "__main__":
    sys.exit(main())
