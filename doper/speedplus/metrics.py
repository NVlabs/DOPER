# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SPEED score and the SPEED+ real-domain evaluation summary.

Two score definitions, both recorded in every summary (``score_definition``):

* ``paper`` (default): ``e_R[rad] + ||t_hat - t|| / ||t||`` with no deadband, averaged over the
  **solved** frames only. This is the score of the paper's tables (``pose_score.mean``).
* ``official``: the SPEC2021 per-term deadbands (0.169 degrees, 0.002173 relative), each term
  zeroed independently. SPEC2021 scores every image, but a frame without a pose has no defined
  official penalty: the summary lists those frames (``no_pose_filenames``) and is
  ``comparable_to_spec2021`` only when every requested frame was solved.

Official mode and missing poses: ``pose_score`` is the mean over the solved frames. With no
missing pose it is exactly the mean over every requested frame, as SPEC2021 scores it. With a
missing pose the official value is undefined (SPEC2021 has no penalty for a frame without a
pose), so the summary sets ``comparable_to_spec2021: false`` and lists ``no_pose_filenames``
instead of inventing one.

Ground-truth rotations carry a float32 floor: ``doper.speedplus.pose.candidate_record`` builds
the ground-truth matrix from a float32-normalized quaternion, as the source does, so a perfect
pose reports a rotation error of about 0.01 to 0.07 degrees. That is kept for parity; it stays
under the official 0.169 degree deadband.

The summary keeps a fixed key set, including ``ransac_inliers``, ``inliers_ge21_*`` and
``rotation_gt30_*``, which the recorded generation history carries.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .definitions import SCORE_DEFINITIONS, ScoreDefinition

__all__ = ["SCORE_DEFINITIONS", "ScoreDefinition", "pose_score", "summarize_domain"]
ROTATION_DEADBAND_DEG = 0.169
TRANSLATION_DEADBAND_RELATIVE = 0.002173
# Pseudo-label admission gate the summary reports (``inliers_ge21_*``).
ADMISSION_MIN_INLIERS = 21
GROSS_ROTATION_DEG = 30.0
FLIP_ROTATION_DEG = 60.0


def check_definition(definition: str) -> None:
    """Refuse a score definition other than ``paper`` and ``official``.

    Raises:
        ValueError: For an unknown definition.
    """
    if definition not in SCORE_DEFINITIONS:
        raise ValueError(f"unknown score definition {definition!r}; known: {SCORE_DEFINITIONS}")


def pose_score(e_r_rad: float, e_t_rel: float, *, definition: str = "paper") -> float:
    """The SPEED score of one pose.

    Args:
        e_r_rad: Rotation error in radians.
        e_t_rel: Translation error relative to the true range.
        definition: ``paper`` (no deadband) or ``official`` (per-term deadbands).

    Raises:
        ValueError: For an unknown definition.
    """
    check_definition(definition)
    if definition == "paper":
        return e_r_rad + e_t_rel
    translation_term = 0.0 if e_t_rel < TRANSLATION_DEADBAND_RELATIVE else e_t_rel
    rotation_term = 0.0 if math.degrees(e_r_rad) < ROTATION_DEADBAND_DEG else e_r_rad
    return translation_term + rotation_term


def distribution(values: np.ndarray) -> dict[str, float] | None:
    """Mean, median, p90, p95, p99 and max of ``values``; None when there are none."""
    if len(values) == 0:
        return None
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def _frame_score(row: Mapping[str, Any], definition: str) -> float:
    """The score of one solved frame: its record's paper score, or the official deadband score.

    Frames without a pose have no score under either definition; ``summarize_domain`` counts
    and lists them instead.
    """
    if definition == "paper":
        return row["pose_score"]
    return pose_score(
        math.radians(row["rotation_deg"]), row["translation_relative"], definition=definition
    )


def summarize_domain(
    records: Sequence[Mapping[str, Any]],
    requested: int,
    *,
    definition: str,
    inference_ms: Sequence[float] = (),
) -> dict[str, Any]:
    """Summarize one real domain's per-frame records (fixed summary key set plus the definition).

    Args:
        records: One record per requested frame, with ``filename`` and ``baseline`` (a
            ``doper.speedplus.pose.candidate_record`` or None when no pose was found).
        requested: Number of requested frames; every rate is over it.
        definition: Score definition of ``pose_score`` (``paper`` or ``official``). Either
            way ``pose_score`` is over the solved frames; in ``official`` mode that is the
            SPEC2021 every-frame mean when no frame lacks a pose (``comparable_to_spec2021``)
            and undefined otherwise (the flag is false and ``no_pose_filenames`` lists them).
        inference_ms: Per-frame inference latencies; ``inference_latency_ms`` is None without.

    Returns:
        The summary. Distributions over no values are None.

    Raises:
        ValueError: For an unknown definition, or when ``requested`` is not positive or differs
            from the number of records.
    """
    check_definition(definition)
    if requested <= 0 or len(records) != requested:
        raise ValueError(f"requested={requested} must be positive and equal {len(records)} records")
    solved = [row["baseline"] for row in records if row["baseline"] is not None]
    no_pose = [row["filename"] for row in records if row["baseline"] is None]
    rotation = np.asarray([row["rotation_deg"] for row in solved])
    translation = np.asarray([row["translation_relative"] for row in solved])
    score = np.asarray([_frame_score(row, definition) for row in solved])
    inliers = np.asarray([row.get("ransac_inliers", 0) for row in solved], dtype=np.float64)
    gross = requested - len(solved) + np.sum(rotation > GROSS_ROTATION_DEG)
    return {
        "requested": requested,
        "solved": len(solved),
        "solver_return_rate": len(solved) / requested,
        "no_pose_count": requested - len(solved),
        "no_pose_rate": (requested - len(solved)) / requested,
        "practical_recall_5deg_2pct": sum(row["practical_success"] for row in solved) / requested,
        "loose_recall_10deg_5pct": sum(row["loose_success"] for row in solved) / requested,
        "rotation_deg": distribution(rotation),
        "translation_relative": distribution(translation),
        "pose_score": distribution(score),
        "ransac_inliers": distribution(inliers),
        "inliers_ge21_count": int(np.sum(inliers >= ADMISSION_MIN_INLIERS)),
        "inliers_ge21_rate_requested": float(np.sum(inliers >= ADMISSION_MIN_INLIERS) / requested),
        "rotation_gt30_count": int(np.sum(rotation > GROSS_ROTATION_DEG)),
        "rotation_gt30_rate_requested": float(np.sum(rotation > GROSS_ROTATION_DEG) / requested),
        "rotation_gt60_count": int(np.sum(rotation > FLIP_ROTATION_DEG)),
        "rotation_gt60_rate_requested": float(np.sum(rotation > FLIP_ROTATION_DEG) / requested),
        "gross_no_pose_or_rotation_gt30_count": int(gross),
        "gross_no_pose_or_rotation_gt30_rate": float(gross / requested),
        "inference_latency_ms": distribution(np.asarray(inference_ms, dtype=np.float64)),
        "score_definition": definition,
        "comparable_to_spec2021": definition == "official" and not no_pose,
        "no_pose_filenames": no_pose,
    }
