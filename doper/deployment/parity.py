# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Latency summaries and output/pose parity between a deployed engine and PyTorch."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from doper.infer import Detection

POSE_GATE_TRANSLATION_M = 0.001
POSE_GATE_ROTATION_DEG = 1.0
_PERCENTILE = 95
_RAW_PERCENTILE = 99
_MS_PER_SECOND = 1000.0
_MIN_NORM = 1e-12

PoseFrames = Sequence[Sequence[dict[str, Any]]]


def summarize(ms: Sequence[float]) -> dict[str, float | int]:
    """Summarize latency samples in milliseconds.

    Args:
        ms: One-dimensional, nonempty, finite, positive samples.

    Returns:
        ``n``, ``mean_ms``, ``median_ms``, ``p95_ms`` and ``fps_from_mean``.

    Raises:
        ValueError: If the samples are empty, not 1-D, non-finite or not positive.
    """
    a = np.asarray(ms, dtype=np.float64)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all() or (a <= 0).any():
        raise ValueError("Expected finite positive latency samples")
    return {
        "n": len(a),
        "mean_ms": float(a.mean()),
        "median_ms": float(np.median(a)),
        "p95_ms": float(np.percentile(a, _PERCENTILE)),
        "fps_from_mean": float(_MS_PER_SECOND / a.mean()),
    }


def compare_outputs(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Compare a raw output tensor with its reference.

    Returns:
        ``max_abs``, ``p99_abs`` and ``relative_l2`` (difference norm over reference norm).

    Raises:
        ValueError: If the shapes differ, either tensor is empty or holds non-finite values.
    """
    a, r = np.asarray(actual, np.float64), np.asarray(reference, np.float64)
    if a.shape != r.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(r).all():
        raise ValueError("Parity requires equal, nonempty, finite tensors")
    d = np.abs(a - r)
    return {
        "max_abs": float(d.max()),
        "p99_abs": float(np.percentile(d, _RAW_PERCENTILE)),
        "relative_l2": float(np.linalg.norm(d.ravel()) / max(np.linalg.norm(r.ravel()), _MIN_NORM)),
    }


def detections_json(dets: Sequence[Detection]) -> list[dict[str, Any]]:
    """Return JSON-ready box, score, keypoints and pose of each detection."""
    return [
        {
            "box": d.bbox_xyxy.tolist(),
            "score": float(d.score),
            "keypoints": d.keypoints.tolist(),
            "keypoint_scores": d.keypoint_scores.tolist(),
            "rvec": None if d.rvec is None else d.rvec.ravel().tolist(),
            "tvec": None if d.tvec is None else d.tvec.ravel().tolist(),
        }
        for d in dets
    ]


def _rotation_difference_deg(rvec_a: Sequence[float], rvec_b: Sequence[float]) -> float:
    ra = cv2.Rodrigues(np.asarray(rvec_a, np.float64))[0]
    rb = cv2.Rodrigues(np.asarray(rvec_b, np.float64))[0]
    cosine = np.clip((np.trace(ra @ rb.T) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cosine)))


def pose_parity(actual: PoseFrames, reference: PoseFrames) -> dict[str, Any]:
    """Match detections per frame by nearest box and measure keypoint and pose differences.

    This is parity with a reference pipeline, not accuracy: detections are paired by
    minimum total box distance (Hungarian), rotations are compared strictly (no symmetry
    discount).

    Args:
        actual: Per-frame :func:`detections_json` lists of the pipeline under test.
        reference: Per-frame lists of the reference pipeline.

    Returns:
        Detection-count and PnP-validity differences, matched count, keypoint pixel p95 and
        the maximum translation (m) and rotation (deg) differences (``None`` when no pair
        has a pose on both sides).

    Raises:
        ValueError: If the frame counts differ.
    """
    if len(actual) != len(reference):
        raise ValueError("Pose reference frame count mismatch")
    errors: list[float] = []
    translations: list[float] = []
    rotations: list[float] = []
    mismatch = validity = matched = 0
    for frame_actual, frame_reference in zip(actual, reference, strict=True):
        mismatch += len(frame_actual) != len(frame_reference)
        if not frame_actual or not frame_reference:
            continue
        costs = np.array(
            [
                [np.linalg.norm(np.subtract(a["box"], r["box"])) for r in frame_reference]
                for a in frame_actual
            ]
        )
        for i, j in zip(*linear_sum_assignment(costs), strict=True):
            a, r = frame_actual[i], frame_reference[j]
            matched += 1
            errors.extend(np.linalg.norm(np.subtract(a["keypoints"], r["keypoints"]), axis=1))
            validity += (a["tvec"] is None) != (r["tvec"] is None)
            if a["tvec"] is not None and r["tvec"] is not None:
                translations.append(float(np.linalg.norm(np.subtract(a["tvec"], r["tvec"]))))
                rotations.append(_rotation_difference_deg(a["rvec"], r["rvec"]))
    return {
        "frames_with_detection_count_difference": int(mismatch),
        "matched_detections": matched,
        "pnp_validity_differences": int(validity),
        "keypoint_pixel_p95": float(np.percentile(errors, _PERCENTILE)) if errors else None,
        "translation_difference_m_max": max(translations, default=None),
        "rotation_difference_deg_max": max(rotations, default=None),
        "scope": "Nearest-box matched parity, not accuracy; default infer_image PnP, no tracking",
    }


def pose_gate_passed(result: dict) -> bool:
    """Return whether a :func:`pose_parity` result passes the deployment pose gate.

    The gate requires equal detection counts and PnP validity on every frame, at least one
    matched detection, and finite maximum differences within 0.001 m and 1 degree.
    """
    translation = result.get("translation_difference_m_max")
    rotation = result.get("rotation_difference_deg_max")
    if translation is None or rotation is None:
        return False
    if not (math.isfinite(translation) and math.isfinite(rotation)):
        return False
    return (
        result.get("frames_with_detection_count_difference") == 0
        and result.get("pnp_validity_differences") == 0
        and result.get("matched_detections", 0) > 0
        and translation <= POSE_GATE_TRANSLATION_M
        and rotation <= POSE_GATE_ROTATION_DEG
    )
