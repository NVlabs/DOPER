# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Box-dither proposal and cross-view scoring helpers."""

import cv2
import numpy as np
import pytest
import torch
from tiny_coco import tiny_model_config

from doper.box_dither import (
    JITTER_OFFSETS,
    JITTER_SCALES,
    PROPOSAL_OFFSETS,
    apply_box_jitter,
    cross_view_metrics,
    infer_boxes,
    jitter_boxes,
    keypoints_from_boxes,
    pose_matrix,
    proposal_metrics,
    proposals,
    reprojection_error,
    rotation_errors,
)
from doper.config import DataConfig, DoperConfig
from doper.infer import Detection, solve_pnp
from doper.model import DoperNet

NUM_KEYPOINTS = 6
IMAGE_SIZE = 96
SHARPEN = 40.0
BOX = np.array([20.0, 30.0, 60.0, 70.0])


def _candidate(rvec, tvec):
    return {"rvec": np.asarray(rvec, float).reshape(3, 1), "tvec": np.asarray(tvec, float)}


def _rig():
    rvec_b = np.array([0.0, 0.03, 0.0])
    tvec_b = np.array([-0.10, 0.0, 0.005])
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(rvec_b)[0]
    transform[:3, 3] = tvec_b
    return transform


def test_proposals_include_unscaled_baseline_first():
    output = proposals(BOX)
    assert len(output) == len(PROPOSAL_OFFSETS)
    first = output[0]
    assert (first["scale"], first["shift_x_frac"], first["shift_y_frac"]) == (1.0, 0.0, 0.0)
    assert first["box_xyxy"].dtype == np.float32
    np.testing.assert_allclose(first["box_xyxy"], BOX)
    scaled = next(p for p in output if p["scale"] == 0.9 and p["shift_x_frac"] == 0.0)
    assert scaled["box_xyxy"][2] - scaled["box_xyxy"][0] == pytest.approx(0.9 * 40.0, rel=1e-5)


def test_rotation_errors_zero_for_identical_and_180_for_flip():
    identity = np.eye(3)
    flip = cv2.Rodrigues(np.array([np.pi, 0.0, 0.0]))[0]
    assert rotation_errors(identity, identity) == pytest.approx(0.0, abs=1e-6)
    assert rotation_errors(identity, flip) == pytest.approx(180.0, abs=1e-4)
    batch = rotation_errors(np.stack([identity, flip]), identity[None])
    np.testing.assert_allclose(batch, [0.0, 180.0], atol=1e-4)


def test_pose_matrix_none_without_pose():
    assert pose_matrix({"rvec": None, "tvec": None}) is None
    matrix = pose_matrix(_candidate([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]))
    expected = np.eye(4)
    expected[:3, 3] = (1.0, 2.0, 3.0)
    np.testing.assert_allclose(matrix, expected)


def test_cross_view_metrics_zero_for_consistent_pair():
    transform = _rig()
    keypoints = np.random.default_rng(0).uniform(-0.05, 0.05, (NUM_KEYPOINTS, 3))
    rvec_a, tvec_a = np.array([0.1, -0.2, 0.3]), np.array([0.02, -0.01, 0.6])
    pose_a = pose_matrix(_candidate(rvec_a, tvec_a))
    pose_b = transform @ pose_a
    candidate_b = _candidate(cv2.Rodrigues(pose_b[:3, :3])[0], pose_b[:3, 3])
    metrics = cross_view_metrics(_candidate(rvec_a, tvec_a), candidate_b, transform, keypoints)
    assert metrics["rotation_deg"] == pytest.approx(0.0, abs=1e-4)
    assert metrics["translation_mm"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["cad_keypoint_rms_mm"] == pytest.approx(0.0, abs=1e-6)


def test_cross_view_metrics_none_without_pose():
    missing = {"rvec": None, "tvec": None}
    metrics = cross_view_metrics(missing, missing, np.eye(4), np.zeros((1, 3)))
    assert metrics == {"rotation_deg": None, "translation_mm": None, "cad_keypoint_rms_mm": None}


def test_infer_boxes_returns_one_candidate_per_proposal():
    model_cfg = tiny_model_config(
        num_classes=1,
        num_keypoints=NUM_KEYPOINTS,
        det_input_size=IMAGE_SIZE,
        letterbox=True,
        neck_channels=8,
        head_channels=8,
        crop_size=32,
        keypoint_head="heatmap",
    )
    cfg = DoperConfig(data=DataConfig(root="."), model=model_cfg)
    torch.manual_seed(0)
    model = DoperNet(model_cfg)
    with torch.no_grad():
        model.kpt_head.out.weight.mul_(SHARPEN)
    model.eval()
    keypoints = np.random.default_rng(0).uniform(-0.05, 0.05, (NUM_KEYPOINTS, 3))
    focal = 1.2 * IMAGE_SIZE
    camera = np.array([[focal, 0, IMAGE_SIZE / 2], [0, focal, IMAGE_SIZE / 2], [0, 0, 1.0]])
    image = np.random.default_rng(1).integers(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    inputs = proposals(BOX)
    candidates = infer_boxes(model, image, cfg, "cpu", inputs, keypoints, camera, batch_size=4)
    assert len(candidates) == len(inputs)
    for candidate, proposal in zip(candidates, inputs, strict=True):
        assert candidate["scale"] == proposal["scale"]
        assert candidate["keypoints"].shape == (NUM_KEYPOINTS, 2)
        assert candidate["keypoint_scores"].shape == (NUM_KEYPOINTS,)
        assert 0.0 <= candidate["mean_keypoint_score"] <= 1.0


def _tiny_setup():
    model_cfg = tiny_model_config(
        num_classes=1,
        num_keypoints=NUM_KEYPOINTS,
        det_input_size=IMAGE_SIZE,
        letterbox=True,
        neck_channels=8,
        head_channels=8,
        crop_size=32,
        keypoint_head="heatmap",
    )
    cfg = DoperConfig(data=DataConfig(root="."), model=model_cfg)
    torch.manual_seed(0)
    model = DoperNet(model_cfg)
    with torch.no_grad():
        model.kpt_head.out.weight.mul_(SHARPEN)
    model.eval()
    keypoints = np.random.default_rng(0).uniform(-0.05, 0.05, (NUM_KEYPOINTS, 3))
    focal = 1.2 * IMAGE_SIZE
    camera = np.array([[focal, 0, IMAGE_SIZE / 2], [0, focal, IMAGE_SIZE / 2], [0, 0, 1.0]])
    image = np.random.default_rng(1).integers(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    return model, cfg, keypoints, camera, image


def _baseline(reprojection_px, solved=True):
    detection = Detection(
        bbox_xyxy=BOX.astype(np.float32),
        score=0.8,
        keypoints=np.zeros((NUM_KEYPOINTS, 2), np.float32),
        keypoint_scores=np.ones(NUM_KEYPOINTS, np.float32),
    )
    if solved:
        detection.rvec = np.zeros((3, 1))
        detection.tvec = np.array([[0.0], [0.0], [0.5]])
        detection.reproj_error_px = reprojection_px
    return detection


def test_jitter_boxes_grid_order_and_parameters():
    boxes, parameters = jitter_boxes(BOX, JITTER_OFFSETS, JITTER_SCALES)
    assert boxes.shape == (125, 4) and boxes.dtype == np.float32
    assert parameters.shape == (125, 3)
    # Scale is the slowest axis, then y offset, then x offset (columns: x offset, y offset, scale).
    np.testing.assert_allclose(parameters[0], (-0.1, -0.1, 0.85))
    np.testing.assert_allclose(parameters[1], (-0.05, -0.1, 0.85))
    np.testing.assert_allclose(parameters[5], (-0.1, -0.05, 0.85))
    np.testing.assert_allclose(parameters[25], (-0.1, -0.1, 0.9))
    np.testing.assert_allclose(parameters[-1], (0.1, 0.1, 1.15))
    centered = 25 * 2 + 12
    np.testing.assert_allclose(parameters[centered], (0.0, 0.0, 1.0))
    np.testing.assert_allclose(boxes[centered], BOX, atol=1e-4)
    width = boxes[0, 2] - boxes[0, 0]
    assert width == pytest.approx(0.85 * 40.0, rel=1e-5)
    assert 0.5 * (boxes[0, 0] + boxes[0, 2]) == pytest.approx(40.0 - 0.1 * 40.0, rel=1e-5)


def test_proposal_metrics_confidence_and_consensus():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    near = np.zeros((4, 2), np.float32)
    far = np.full((4, 2), 30.0, np.float32)
    detections = [Detection(BOX, 1.0, keypoints, scores) for keypoints in (near, near.copy(), far)]
    metrics = proposal_metrics(detections, np.array([0.0, 0.0, 30.0, 40.0]), 0.5)
    confidence = 0.75 * np.median(scores) + 0.25 * np.percentile(scores, 10)
    assert metrics[0]["confidence"] == pytest.approx(confidence)
    assert metrics[0]["disagreement_px"] == pytest.approx(0.0)
    diagonal = 50.0
    expected_far = np.median([np.hypot(30, 30)] * 2 + [0.0])
    assert metrics[2]["disagreement_px"] == pytest.approx(expected_far)
    assert metrics[2]["disagreement_normalized"] == pytest.approx(expected_far / diagonal)
    assert metrics[2]["selection_score"] == pytest.approx(
        confidence - 0.5 * expected_far / diagonal
    )
    assert metrics[0]["selection_score"] > metrics[2]["selection_score"]


def test_jitter_grid_order_ranking_and_adoption():
    model, cfg, keypoints, camera, image = _tiny_setup()
    good = _baseline(2.0)
    kept, audit = apply_box_jitter(model, image, cfg, "cpu", good, keypoints, camera)
    assert kept is good and audit == {"attempted": False, "adopted": False, "proposal_count": 0}
    kept, audit = apply_box_jitter(model, image, cfg, "cpu", None, keypoints, camera)
    assert kept is None and audit["attempted"] is False
    boxes, parameters = jitter_boxes(BOX, JITTER_OFFSETS, JITTER_SCALES)
    expected = keypoints_from_boxes(model, image, cfg, "cpu", boxes, batch_size=16)
    metrics = proposal_metrics(expected, BOX, 0.5)
    index = int(np.argmax([m["selection_score"] for m in metrics]))
    for baseline in (_baseline(9.0), _baseline(0.0, solved=False)):
        selected, audit = apply_box_jitter(model, image, cfg, "cpu", baseline, keypoints, camera)
        reference = expected[index]
        reference.score = baseline.score
        solve_pnp(reference, keypoints, camera, None)
        adopted = reprojection_error(reference) < reprojection_error(baseline)
        assert audit["attempted"] and audit["proposal_count"] == 125
        assert audit["selected_index"] == index
        assert audit["selected_parameters"] == {
            "offset_x": parameters[index, 0],
            "offset_y": parameters[index, 1],
            "scale": parameters[index, 2],
        }
        assert audit["selection_metrics"] == metrics[index]
        assert audit["adopted"] is adopted
        assert (selected is baseline) is (not adopted)
        if adopted:
            np.testing.assert_allclose(selected.keypoints, reference.keypoints)
            assert selected.score == baseline.score
    assert reprojection_error(_baseline(0.0, solved=False)) == float("inf")


def test_apply_box_jitter_threshold_is_exclusive():
    model, cfg, keypoints, camera, image = _tiny_setup()
    _, audit = apply_box_jitter(model, image, cfg, "cpu", _baseline(8.0), keypoints, camera)
    assert audit["attempted"] is False
    _, audit = apply_box_jitter(
        model, image, cfg, "cpu", _baseline(8.0), keypoints, camera, threshold=7.9
    )
    assert audit["attempted"] is True
