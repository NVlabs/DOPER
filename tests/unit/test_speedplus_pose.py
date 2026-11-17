# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SPEED+ USAC pose candidates, their records, projected crops and the SPEED score."""

import math

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from speedplus_helpers import DIST, OFFICIAL_POINTS, K

from doper.speedplus import geometry, metrics, pose

QUATERNION = np.asarray([0.82, 0.21, -0.31, 0.43]) / np.linalg.norm([0.82, 0.21, -0.31, 0.43])
TRANSLATION = np.asarray([0.18, -0.11, 6.25])
# Every key of a recorded evaluation summary (round_history_gen08.json,
# ``evaluation.normal.<domain>``), which uses the research summary key set.
RECORDED_SUMMARY_KEYS = {
    "requested", "solved", "solver_return_rate", "no_pose_count", "no_pose_rate",
    "practical_recall_5deg_2pct", "loose_recall_10deg_5pct", "rotation_deg",
    "translation_relative", "pose_score", "ransac_inliers", "inliers_ge21_count",
    "inliers_ge21_rate_requested", "rotation_gt30_count", "rotation_gt30_rate_requested",
    "rotation_gt60_count", "rotation_gt60_rate_requested",
    "gross_no_pose_or_rotation_gt30_count", "gross_no_pose_or_rotation_gt30_rate",
    "inference_latency_ms",
}  # fmt: skip
# The five keys the research source snapshot dropped from the research summary.
RESEARCH_ONLY_KEYS = {
    "ransac_inliers", "inliers_ge21_count", "inliers_ge21_rate_requested",
    "rotation_gt30_count", "rotation_gt30_rate_requested",
}  # fmt: skip
DISTRIBUTION_KEYS = ["mean", "median", "p90", "p95", "p99", "max"]


def _points() -> np.ndarray:
    return geometry.expand_tango_keypoints_64(OFFICIAL_POINTS)


def _projection(points: np.ndarray) -> np.ndarray:
    rvec, tvec = geometry.speedplus_pose_to_opencv(QUATERNION, TRANSLATION)
    pixels, _ = cv2.projectPoints(points, rvec, tvec, np.asarray(K), np.asarray(DIST))
    return pixels.reshape(-1, 2)


def test_quaternion_to_object_camera_is_the_transposed_dcm():
    expected = geometry.quaternion_to_dcm_scalar_first(QUATERNION).T
    single = pose.quaternion_to_object_camera(QUATERNION, normalization="float32")
    batch = pose.quaternion_to_object_camera(
        np.stack([QUATERNION, -QUATERNION]), normalization="float32"
    )
    assert single.shape == (3, 3) and batch.shape == (2, 3, 3)
    np.testing.assert_allclose(single, expected, atol=1e-6)
    np.testing.assert_array_equal(batch[0], single)
    np.testing.assert_array_equal(batch[1], single)


def _angle(rotation: np.ndarray, expected: np.ndarray) -> float:
    return float(Rotation.from_matrix(rotation @ expected.T).magnitude())


def test_oracle_projection_pnp_recovers_pose():
    points = _points()
    keypoints = _projection(points)
    expected = geometry.quaternion_to_dcm_scalar_first(QUATERNION).T
    # Converged PnP on the oracle projection recovers the native pose: the DCM transpose and
    # the scalar-first order are right.
    _, rvec, tvec = cv2.solvePnP(points, keypoints, np.asarray(K), np.asarray(DIST))
    assert _angle(cv2.Rodrigues(rvec)[0], expected) < 1e-6
    assert np.linalg.norm(tvec.ravel() - TRANSLATION) / np.linalg.norm(TRANSLATION) < 1e-6
    assert (
        _angle(pose.quaternion_to_object_camera(QUATERNION, normalization="float32"), expected)
        < 1e-6
    )
    # The USAC candidate (no final refinement, as in the source) lands within its tolerance.
    candidate = pose.argmax_usac_candidate(
        keypoints, np.ones(len(points)), points, np.asarray(K), np.asarray(DIST)
    )
    assert candidate is not None and candidate.source == "argmax-usac4k"
    assert candidate.ransac_inliers == candidate.ransac_correspondences == 64
    assert _angle(cv2.Rodrigues(candidate.rvec)[0], expected) < 1e-5
    record = pose.candidate_record(candidate, QUATERNION, TRANSLATION, normalization="float32")
    assert record["translation_relative"] < 1e-5 and record["within_hil_deadband"]
    # A missing DCM transpose (the inverse rotation) is caught, not silently accepted.
    inverse = QUATERNION * np.asarray([1.0, -1.0, -1.0, -1.0])
    assert (
        pose.candidate_record(candidate, inverse, TRANSLATION, normalization="float32")[
            "rotation_deg"
        ]
        > 30.0
    )


def test_usac_returns_inlier_count_and_drops_nonfinite():
    points = _points()
    keypoints = _projection(points)
    scores = np.full(len(points), 0.9)
    keypoints[:8] = np.nan  # non-finite keypoints
    scores[8:10] = np.nan  # non-finite scores
    scores[10:13] = 0.29  # below the 0.30 threshold
    keypoints[13:18] += 150.0  # confident outliers
    candidate = pose.argmax_usac_candidate(
        keypoints, scores, points, np.asarray(K), np.asarray(DIST)
    )
    assert candidate is not None
    assert candidate.ransac_correspondences == 64 - 13
    assert candidate.ransac_inliers == 64 - 18
    record = pose.candidate_record(candidate, QUATERNION, TRANSLATION, normalization="float32")
    assert record["ransac_inliers"] == 46 and record["ransac_correspondences"] == 51
    assert record["ransac_inlier_fraction"] == 46 / 51
    assert (
        record["pose_score"]
        == math.radians(record["rotation_deg"]) + record["translation_relative"]
    )
    scores[:] = 0.0
    scores[20:23] = 1.0
    assert pose.argmax_usac_candidate(keypoints, scores, points, np.asarray(K), None) is None


def test_projected_pseudo_box_pads_and_clips():
    points = _points()
    rvec, tvec = geometry.speedplus_pose_to_opencv(QUATERNION, TRANSLATION)
    pixels = _projection(points)
    lo, hi = pixels.min(axis=0), pixels.max(axis=0)
    box = pose.projected_pseudo_box(
        rvec.ravel(), tvec.ravel(), points, np.asarray(K), np.asarray(DIST), image_size=(1920, 1200)
    )
    assert box.dtype == np.float32
    np.testing.assert_allclose(box, np.concatenate([lo, hi]), rtol=1e-6)
    padded = pose.projected_pseudo_box(
        rvec, tvec, points, np.asarray(K), np.asarray(DIST), image_size=(1920, 1200), pad_frac=0.1
    )
    np.testing.assert_allclose(padded[:2], lo - 0.1 * (hi - lo), rtol=1e-6)
    clipped = pose.projected_pseudo_box(
        rvec, tvec, points, np.asarray(K), np.asarray(DIST), image_size=(1000, 650), pad_frac=1.0
    )
    np.testing.assert_array_equal(clipped[2:], [999.0, 649.0])


def test_pose_projected_crop_warps_with_grey_border():
    from doper.dataset import normalize_image

    image = np.zeros((200, 300, 3), np.uint8)
    image[:] = (10, 20, 30)  # BGR
    box = np.asarray([100.0, 60.0, 160.0, 140.0], np.float32)
    crop, warp = pose.pose_projected_crop(
        image, box, crop_size=64, crop_pad_factor=1.25, angle_deg=0.0
    )
    assert crop.shape == (3, 64, 64) and crop.dtype == np.float32
    np.testing.assert_allclose(warp @ [130.0, 100.0, 1.0], [32.0, 32.0], atol=1e-4)
    rgb = normalize_image(np.full((1, 1, 3), (30, 20, 10), np.uint8))[:, 0, 0]
    np.testing.assert_allclose(crop[:, 32, 32], rgb)
    corner = np.asarray([0.0, 0.0, 40.0, 40.0], np.float32)
    crop, warp = pose.pose_projected_crop(
        image, corner, crop_size=64, crop_pad_factor=1.25, angle_deg=180.0
    )
    grey = normalize_image(np.full((1, 1, 3), 114, np.uint8))[:, 0, 0]
    np.testing.assert_allclose(crop[:, 63, 63], grey)
    np.testing.assert_allclose(warp @ [21.0, 20.0, 1.0], [32.0 - 64 / 50, 32.0], atol=1e-4)


def _solved(rotation_deg: float, translation_relative: float, inliers: int = 30) -> dict:
    score = math.radians(rotation_deg) + translation_relative
    return {
        "rotation_deg": rotation_deg,
        "translation_relative": translation_relative,
        "pose_score": score,
        "ransac_inliers": inliers,
        "practical_success": rotation_deg < 5.0 and translation_relative < 0.02,
        "loose_success": rotation_deg < 10.0 and translation_relative < 0.05,
    }


def _records(*baselines: dict | None) -> list[dict]:
    return [
        {"index": i, "filename": f"img{i:06d}.jpg", "baseline": b} for i, b in enumerate(baselines)
    ]


def test_paper_score_counts_solved_frames_only():
    records = _records(_solved(2.0, 0.01), None, _solved(40.0, 0.03, inliers=12))
    summary = metrics.summarize_domain(records, 3, definition="paper")
    expected = np.mean([math.radians(2.0) + 0.01, math.radians(40.0) + 0.03])
    assert summary["pose_score"]["mean"] == pytest.approx(expected, abs=1e-15)
    assert summary["solved"] == 2 and summary["no_pose_count"] == 1
    assert summary["practical_recall_5deg_2pct"] == 1 / 3
    assert summary["inliers_ge21_count"] == 1 and summary["rotation_gt30_count"] == 1
    assert summary["gross_no_pose_or_rotation_gt30_count"] == 2
    assert summary["score_definition"] == "paper"
    assert summary["comparable_to_spec2021"] is False
    assert summary["no_pose_filenames"] == ["img000001.jpg"]
    assert metrics.pose_score(math.radians(0.1), 0.001) == math.radians(0.1) + 0.001


def test_official_score_zeroes_each_term_independently():
    small_r, large_r = math.radians(0.1), math.radians(1.0)
    assert metrics.pose_score(small_r, 0.5, definition="official") == 0.5
    assert metrics.pose_score(large_r, 0.001, definition="official") == large_r
    assert metrics.pose_score(small_r, 0.001, definition="official") == 0.0
    assert metrics.pose_score(large_r, 0.5, definition="official") == 0.5 + large_r
    with pytest.raises(ValueError, match="definition"):
        metrics.pose_score(small_r, 0.5, definition="spec")
    all_solved = _records(_solved(0.1, 0.5), _solved(1.0, 0.001))
    summary = metrics.summarize_domain(all_solved, 2, definition="official")
    assert summary["pose_score"]["mean"] == (0.5 + large_r) / 2
    assert summary["score_definition"] == "official"
    assert summary["comparable_to_spec2021"] is True and summary["no_pose_filenames"] == []
    # With no missing pose the solved-frame mean is the mean over every requested frame.
    every_frame = [metrics.pose_score(math.radians(r["baseline"]["rotation_deg"]),
                                      r["baseline"]["translation_relative"], definition="official")
                   for r in all_solved]  # fmt: skip
    assert summary["solved"] == summary["requested"] == len(every_frame)
    assert summary["pose_score"]["mean"] == float(np.mean(every_frame))
    missing = metrics.summarize_domain([*all_solved, *_records(None)], 3, definition="official")
    assert missing["comparable_to_spec2021"] is False and missing["no_pose_count"] == 1


def test_summary_has_research_summary_keys():
    records = _records(_solved(2.0, 0.01), None)
    summary = metrics.summarize_domain(records, 2, definition="paper", inference_ms=[20.0, 21.0])
    assert RESEARCH_ONLY_KEYS <= RECORDED_SUMMARY_KEYS <= set(summary)
    for key in ("rotation_deg", "translation_relative", "pose_score", "ransac_inliers"):
        assert list(summary[key]) == DISTRIBUTION_KEYS
    assert summary["inference_latency_ms"]["mean"] == 20.5
    empty = metrics.summarize_domain(_records(None), 1, definition="paper")
    assert empty["pose_score"] is None and empty["inference_latency_ms"] is None
    with pytest.raises(ValueError, match="requested"):
        metrics.summarize_domain(records, 3, definition="paper")


def test_normalization_is_a_required_keyword():
    """No caller inherits a normalization silently: SPEED+ passes float32, SHIRT float64."""
    with pytest.raises(TypeError, match="normalization"):
        pose.quaternion_to_object_camera(QUATERNION)  # type: ignore[call-arg]
    candidate = pose.Candidate("x", np.zeros((3, 1)), np.ones((3, 1)))
    with pytest.raises(TypeError, match="normalization"):
        pose.candidate_record(candidate, QUATERNION, TRANSLATION)  # type: ignore[call-arg]
