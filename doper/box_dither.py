# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Box-dither proposals and cross-view scoring helpers.

Dithering re-runs the keypoint head on slightly scaled and shifted copies of a detected box and
scores the resulting poses, either per view or across a calibrated pair of views.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
import torch

from .config import DoperConfig
from .dataset import normalize_image
from .infer import DEFAULT_SCORE_THRESHOLD, LETTERBOX_PAD_VALUE, Detection, solve_pnp
from .model import DoperNet
from .targets import bbox_to_center_size, decode_keypoint_heatmaps, topdown_warp_matrix

# (scale, x shift fraction, y shift fraction); the unscaled, unshifted baseline comes first.
PROPOSAL_OFFSETS: tuple[tuple[float, float, float], ...] = (
    (1.0, 0.0, 0.0),
    (0.9, -0.10, -0.05),
    (1.1, 0.10, 0.05),
    (0.9, 0.10, -0.05),
    (1.1, -0.10, 0.05),
    (0.9, 0.0, 0.0),
    (1.1, 0.0, 0.0),
    (1.0, -0.10, 0.0),
    (1.0, 0.10, 0.0),
    (1.0, 0.0, -0.05),
    (1.0, 0.0, 0.05),
    (1.0, -0.10, -0.05),
    (1.0, 0.10, 0.05),
    (1.0, 0.10, -0.05),
    (1.0, -0.10, 0.05),
)
# Fixed jitter grid: 5 scales x 5 y offsets x 5 x offsets = 125 crops, offsets as box fractions.
JITTER_OFFSETS: tuple[float, ...] = (-0.1, -0.05, 0.0, 0.05, 0.1)
JITTER_SCALES: tuple[float, ...] = (0.85, 0.9, 1.0, 1.1, 1.15)
JITTER_CONSENSUS_WEIGHT = 0.5
JITTER_TRIGGER_REPROJECTION_PX = 8.0
JITTER_BATCH_SIZE = 16
MEDIAN_SCORE_WEIGHT = 0.75
LOW_SCORE_WEIGHT = 0.25
LOW_SCORE_PERCENTILE = 10
PNP_RANSAC_SEED = 20260724
DISTORTION_COEFFICIENT_COUNT = 5
METERS_TO_MILLIMETERS = 1000.0


def proposals(box: np.ndarray) -> list[dict[str, Any]]:
    """Return the deterministic nested box-dither proposals for one box.

    Args:
        box: ``(4,)`` box ``x1, y1, x2, y2`` in image pixels.

    Returns:
        One dict per entry of ``PROPOSAL_OFFSETS`` with ``scale``, ``shift_x_frac``,
        ``shift_y_frac`` and the float32 ``box_xyxy``. The first proposal is the baseline box.
    """
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float64)
    width, height = x2 - x1, y2 - y1
    center_x, center_y = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    output = []
    for scale, shift_x, shift_y in PROPOSAL_OFFSETS:
        proposal_width = scale * width
        proposal_height = scale * height
        shifted_x = center_x + shift_x * width
        shifted_y = center_y + shift_y * height
        output.append(
            {
                "scale": scale,
                "shift_x_frac": shift_x,
                "shift_y_frac": shift_y,
                "box_xyxy": np.asarray(
                    [
                        shifted_x - 0.5 * proposal_width,
                        shifted_y - 0.5 * proposal_height,
                        shifted_x + 0.5 * proposal_width,
                        shifted_y + 0.5 * proposal_height,
                    ],
                    dtype=np.float32,
                ),
            }
        )
    return output


def baseline_candidate(detection: Detection) -> dict[str, Any]:
    """Convert a regular detection into the candidate dict used by the scoring helpers."""
    return {
        "scale": 1.0,
        "shift_x_frac": 0.0,
        "shift_y_frac": 0.0,
        "box_xyxy": np.asarray(detection.bbox_xyxy, dtype=np.float32),
        "mean_keypoint_score": float(np.mean(detection.keypoint_scores)),
        "median_keypoint_score": float(np.median(detection.keypoint_scores)),
        "keypoints": np.asarray(detection.keypoints, dtype=np.float32),
        "keypoint_scores": np.asarray(detection.keypoint_scores, dtype=np.float32),
        "pnp_reprojection_px": (
            float(detection.reproj_error_px) if detection.reproj_error_px is not None else None
        ),
        "rvec": detection.rvec,
        "tvec": detection.tvec,
    }


def _warp_proposal_crops(
    image: np.ndarray,
    cfg: DoperConfig,
    proposals_: Sequence[dict[str, Any]],
    python_floats: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return normalized crops and inverse warps for every proposal box.

    With ``python_floats`` the box width and height are formed in the box dtype and then widened
    to Python floats before the crop geometry, as the jitter and ground-truth-box paths do.
    """
    crop_size = cfg.model.crop_size
    crops = []
    inverse_warps = []
    for proposal in proposals_:
        x1, y1, x2, y2 = proposal["box_xyxy"]
        box_xywh = (x1, y1, x2 - x1, y2 - y1)
        if python_floats:
            box_xywh = (float(x1), float(y1), float(x2 - x1), float(y2 - y1))
        center, source_size = bbox_to_center_size(box_xywh, cfg.model.crop_pad_factor)
        warp = topdown_warp_matrix(center, source_size, crop_size)
        crop = cv2.warpAffine(
            image,
            warp,
            (crop_size, crop_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(LETTERBOX_PAD_VALUE,) * 3,
        )
        crops.append(normalize_image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        inverse_warps.append(cv2.invertAffineTransform(warp))
    return crops, inverse_warps


def _predict_heatmaps(
    model: DoperNet, crops: Sequence[np.ndarray], device: str, batch_size: int
) -> np.ndarray:
    """Run the keypoint head over ``crops`` in batches and stack the heatmap logits."""
    outputs = []
    for start in range(0, len(crops), batch_size):
        batch = torch.from_numpy(np.stack(crops[start : start + batch_size])).to(device)
        logits = model.forward_kpt(batch)
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError("box dithering requires a 2D heatmap checkpoint")
        outputs.append(logits.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _decode_image_keypoints(
    heatmap: np.ndarray, inverse_warp: np.ndarray, cfg: DoperConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Decode one crop heatmap into float32 image-pixel keypoints and their scores."""
    crop_points, scores = decode_keypoint_heatmaps(
        heatmap, cfg.model.crop_size, beta=cfg.loss.keypoint_heatmap_beta, dark=True
    )
    homogeneous = np.concatenate(
        [crop_points, np.ones((len(crop_points), 1), dtype=np.float32)], axis=1
    )
    return (homogeneous @ inverse_warp.T).astype(np.float32), scores


@torch.inference_mode()
def keypoints_from_boxes(
    model: DoperNet,
    image: np.ndarray,
    cfg: DoperConfig,
    device: str,
    boxes: Sequence[Sequence[float]] | np.ndarray,
    batch_size: int = JITTER_BATCH_SIZE,
) -> list[Detection]:
    """Crop each box, run the keypoint head and map the keypoints back to the image.

    This is the pose-free half of ``infer_boxes``: no PnP is solved, so callers choose which
    result to pose.

    Args:
        model: Heatmap-head model.
        image: BGR uint8 image.
        cfg: Run configuration.
        device: Torch device string.
        boxes: ``(N, 4)`` boxes ``x1, y1, x2, y2`` in image pixels.
        batch_size: Crops per forward pass.

    Returns:
        One ``Detection`` per box with score 1.0, the float32 box, image-pixel ``keypoints`` and
        float32 ``keypoint_scores``; no pose.

    Raises:
        RuntimeError: If the model does not produce 2D keypoint heatmaps.
    """
    boxed = [{"box_xyxy": np.asarray(box)} for box in boxes]
    crops, inverse_warps = _warp_proposal_crops(image, cfg, boxed, python_floats=True)
    heatmaps = _predict_heatmaps(model, crops, device, batch_size)
    detections = []
    for item, inverse, heatmap in zip(boxed, inverse_warps, heatmaps, strict=True):
        points, scores = _decode_image_keypoints(heatmap, inverse, cfg)
        detections.append(
            Detection(
                bbox_xyxy=np.asarray(item["box_xyxy"], dtype=np.float32),
                score=1.0,
                keypoints=points,
                keypoint_scores=scores.astype(np.float32),
            )
        )
    return detections


@torch.inference_mode()
def infer_boxes(
    model: DoperNet,
    image: np.ndarray,
    cfg: DoperConfig,
    device: str,
    proposals: Sequence[dict[str, Any]],
    keypoints_3d: np.ndarray,
    camera_matrix: np.ndarray,
    batch_size: int,
    dist_coeffs: np.ndarray | None = None,
    kp_score_thr: float = DEFAULT_SCORE_THRESHOLD,
) -> list[dict[str, Any]]:
    """Run the keypoint head and PnP on every proposal box.

    Args:
        model: Heatmap-head model.
        image: BGR uint8 image.
        cfg: Run configuration.
        device: Torch device string.
        proposals: Proposal dicts from ``proposals``.
        keypoints_3d: ``(K, 3)`` object-frame keypoints.
        camera_matrix: ``(3, 3)`` camera intrinsics.
        batch_size: Crops per forward pass.
        dist_coeffs: Distortion coefficients, or None for zeros.
        kp_score_thr: Minimum keypoint score used as a PnP correspondence.

    Returns:
        One candidate per proposal: the proposal fields plus ``mean_keypoint_score``,
        ``median_keypoint_score``, ``keypoints``, ``keypoint_scores``, ``pnp_reprojection_px``,
        ``rvec`` and ``tvec`` (the last three None when PnP failed).

    Raises:
        RuntimeError: If the model does not produce 2D keypoint heatmaps.
    """
    crops, inverse_warps = _warp_proposal_crops(image, cfg, proposals)
    heatmaps = _predict_heatmaps(model, crops, device, batch_size)
    distortion = (
        np.zeros((DISTORTION_COEFFICIENT_COUNT, 1), dtype=np.float64)
        if dist_coeffs is None
        else np.asarray(dist_coeffs, dtype=np.float64)
    )
    results = []
    for proposal, inverse, heatmap in zip(proposals, inverse_warps, heatmaps, strict=True):
        image_points, scores = _decode_image_keypoints(heatmap, inverse, cfg)
        detection = Detection(
            bbox_xyxy=proposal["box_xyxy"],
            score=1.0,
            keypoints=image_points,
            keypoint_scores=scores,
        )
        # Isolate crop dithering from OpenCV RANSAC randomness. Without resetting this global
        # RNG, proposal order changes the PnP sample sequence and can make an identical
        # baseline crop decode to a different pose.
        cv2.setRNGSeed(PNP_RANSAC_SEED)
        solve_pnp(detection, keypoints_3d, camera_matrix, distortion, score_thr=kp_score_thr)
        results.append(
            {
                **proposal,
                "mean_keypoint_score": float(np.mean(scores)),
                "median_keypoint_score": float(np.median(scores)),
                "keypoints": image_points,
                "keypoint_scores": scores,
                "pnp_reprojection_px": (
                    float(detection.reproj_error_px)
                    if detection.reproj_error_px is not None
                    else None
                ),
                "rvec": detection.rvec,
                "tvec": detection.tvec,
            }
        )
    return results


def pose_matrix(candidate: dict[str, Any]) -> np.ndarray | None:
    """Return the ``(4, 4)`` object-to-camera transform of a candidate, or None without a pose."""
    if candidate["rvec"] is None or candidate["tvec"] is None:
        return None
    rotation, _ = cv2.Rodrigues(np.asarray(candidate["rvec"], dtype=np.float64).reshape(3, 1))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(candidate["tvec"], dtype=np.float64).reshape(3)
    return transform


def rotation_errors(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the geodesic angle in degrees between broadcast ``(..., 3, 3)`` rotations."""
    relative = first @ np.swapaxes(second, -1, -2)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def cross_view_metrics(
    candidate_a: dict[str, Any],
    candidate_b: dict[str, Any],
    transform_b_from_a: np.ndarray,
    keypoints_3d: np.ndarray,
) -> dict[str, float | None]:
    """Score the agreement of two single-view poses through a known rig transform.

    Args:
        candidate_a: Candidate from view A.
        candidate_b: Candidate from view B.
        transform_b_from_a: ``(4, 4)`` transform taking view A camera points to view B.
        keypoints_3d: ``(K, 3)`` object-frame keypoints in meters.

    Returns:
        ``rotation_deg``, ``translation_mm`` and ``cad_keypoint_rms_mm``; all None when either
        candidate has no pose.
    """
    pose_a = pose_matrix(candidate_a)
    pose_b = pose_matrix(candidate_b)
    if pose_a is None or pose_b is None:
        return {"rotation_deg": None, "translation_mm": None, "cad_keypoint_rms_mm": None}
    pose_a_in_b = transform_b_from_a @ pose_a
    rotation = float(rotation_errors(pose_a_in_b[None, :3, :3], pose_b[None, :3, :3])[0])
    translation = float(METERS_TO_MILLIMETERS * np.linalg.norm(pose_a_in_b[:3, 3] - pose_b[:3, 3]))
    points_a = (pose_a_in_b[:3, :3] @ keypoints_3d.T).T + pose_a_in_b[:3, 3]
    points_b = (pose_b[:3, :3] @ keypoints_3d.T).T + pose_b[:3, 3]
    rms = float(
        METERS_TO_MILLIMETERS * np.sqrt(np.mean(np.sum((points_a - points_b) ** 2, axis=1)))
    )
    return {"rotation_deg": rotation, "translation_mm": translation, "cad_keypoint_rms_mm": rms}


def jitter_boxes(
    box_xyxy: np.ndarray, offsets: Sequence[float], scales: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Return the deterministic grid of jittered boxes around ``box_xyxy`` (no RNG).

    The grid is ordered scale first, then y offset, then x offset. Offsets are fractions of the
    box width and height applied to the box center; scales resize both box sides.

    Args:
        box_xyxy: ``(4,)`` box ``x1, y1, x2, y2`` in image pixels.
        offsets: Center offsets as box-size fractions, used for both axes.
        scales: Box side scale factors.

    Returns:
        ``(N, 4)`` float32 boxes and ``(N, 3)`` float64 parameters whose columns are the x offset,
        the y offset and the scale of each box.
    """
    x1, y1, x2, y2 = np.asarray(box_xyxy, dtype=np.float32)
    width, height = float(x2 - x1), float(y2 - y1)
    center_x, center_y = float((x1 + x2) * 0.5), float((y1 + y2) * 0.5)
    boxes = []
    parameters = []
    for scale in scales:
        half_width, half_height = 0.5 * width * float(scale), 0.5 * height * float(scale)
        for offset_y in offsets:
            for offset_x in offsets:
                proposal_x = center_x + float(offset_x) * width
                proposal_y = center_y + float(offset_y) * height
                boxes.append(
                    [
                        proposal_x - half_width,
                        proposal_y - half_height,
                        proposal_x + half_width,
                        proposal_y + half_height,
                    ]
                )
                parameters.append([float(offset_x), float(offset_y), float(scale)])
    return np.asarray(boxes, dtype=np.float32), np.asarray(parameters, dtype=np.float64)


def proposal_metrics(
    detections: Sequence[Detection], reference_box: np.ndarray, consensus_weight: float
) -> list[dict[str, float]]:
    """Score each proposal by keypoint confidence minus its disagreement with the others.

    Args:
        detections: Pose-free detections from ``keypoints_from_boxes``.
        reference_box: ``(4,)`` box whose diagonal normalizes the disagreement.
        consensus_weight: Weight of the normalized disagreement in ``selection_score``.

    Returns:
        Per detection ``confidence``, ``kp_score_median``, ``kp_score_p10``, ``disagreement_px``
        (median over the other proposals of the median keypoint distance),
        ``disagreement_normalized`` and ``selection_score``.
    """
    points = np.stack([detection.keypoints for detection in detections])
    pairwise = np.linalg.norm(points[:, None, :, :] - points[None, :, :, :], axis=-1)
    disagreement_px = np.median(np.median(pairwise, axis=-1), axis=1)
    x1, y1, x2, y2 = np.asarray(reference_box, dtype=np.float32)
    box_diagonal = max(1.0, float(np.hypot(x2 - x1, y2 - y1)))
    metrics = []
    for index, detection in enumerate(detections):
        scores = np.asarray(detection.keypoint_scores, dtype=np.float64)
        confidence = MEDIAN_SCORE_WEIGHT * float(np.median(scores)) + LOW_SCORE_WEIGHT * float(
            np.percentile(scores, LOW_SCORE_PERCENTILE)
        )
        disagreement = float(disagreement_px[index])
        normalized = disagreement / box_diagonal
        metrics.append(
            {
                "confidence": confidence,
                "kp_score_median": float(np.median(scores)),
                "kp_score_p10": float(np.percentile(scores, LOW_SCORE_PERCENTILE)),
                "disagreement_px": disagreement,
                "disagreement_normalized": normalized,
                "selection_score": confidence - float(consensus_weight) * normalized,
            }
        )
    return metrics


def reprojection_error(detection: Detection | None) -> float:
    """Return the PnP reprojection error of ``detection`` in pixels, or ``inf`` without a pose."""
    if detection is None or detection.rvec is None or detection.tvec is None:
        return math.inf
    value = detection.reproj_error_px
    return float(value) if value is not None and math.isfinite(value) else math.inf


def apply_box_jitter(
    model: DoperNet,
    image: np.ndarray,
    cfg: DoperConfig,
    device: str,
    baseline: Detection | None,
    keypoints_mm: np.ndarray,
    camera_matrix: np.ndarray,
    threshold: float = JITTER_TRIGGER_REPROJECTION_PX,
    batch_size: int = JITTER_BATCH_SIZE,
) -> tuple[Detection | None, dict[str, Any]]:
    """Replace a weak baseline pose by the best-ranked jittered crop when that pose is better.

    The baseline is kept when it is missing or its reprojection error is at most ``threshold``.
    Otherwise the 125-crop grid around the baseline box is run, the crop with the highest
    confidence/consensus ``selection_score`` is posed, and it is adopted only when its
    reprojection error is strictly lower than the baseline's (a missing pose counts as infinite).

    Args:
        model: Heatmap-head model.
        image: BGR uint8 image.
        cfg: Run configuration.
        device: Torch device string.
        baseline: Single-pass detection, or None when the detector found nothing.
        keypoints_mm: ``(K, 3)`` object-frame keypoints, in the units of the translation wanted.
        camera_matrix: ``(3, 3)`` camera intrinsics.
        threshold: Baseline reprojection error in pixels above which jitter is attempted.
        batch_size: Crops per forward pass.

    Returns:
        The selected detection and an audit dict with ``attempted``, ``adopted`` and
        ``proposal_count``, plus ``selected_index``, ``selected_parameters``,
        ``selection_metrics`` and ``selected_reprojection_px`` when jitter was attempted.
    """
    audit: dict[str, Any] = {"attempted": False, "adopted": False, "proposal_count": 0}
    if baseline is None or reprojection_error(baseline) <= threshold:
        return baseline, audit
    boxes, parameters = jitter_boxes(baseline.bbox_xyxy, JITTER_OFFSETS, JITTER_SCALES)
    proposals_ = keypoints_from_boxes(model, image, cfg, device, boxes, batch_size=batch_size)
    metrics = proposal_metrics(proposals_, baseline.bbox_xyxy, JITTER_CONSENSUS_WEIGHT)
    index = max(range(len(metrics)), key=lambda i: metrics[i]["selection_score"])
    candidate = proposals_[index]
    candidate.score = baseline.score
    solve_pnp(candidate, keypoints_mm, camera_matrix, None)
    candidate_error = reprojection_error(candidate)
    adopted = candidate_error < reprojection_error(baseline)
    offset_x, offset_y, scale = (float(value) for value in parameters[index])
    audit.update(
        attempted=True,
        adopted=adopted,
        proposal_count=len(proposals_),
        selected_index=index,
        selected_parameters={"offset_x": offset_x, "offset_y": offset_y, "scale": scale},
        selection_metrics=metrics[index],
        selected_reprojection_px=candidate_error if math.isfinite(candidate_error) else None,
    )
    return (candidate if adopted else baseline), audit
