# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every public ``doper.infer`` entry point runs once on a tiny random model."""

import cv2
import numpy as np
import pytest
import torch
from tiny_coco import tiny_model_config

from doper.config import DataConfig, DoperConfig
from doper.infer import (
    Detection,
    infer_image,
    infer_image_tta,
    infer_keypoints_in_box,
    refine_detection_pose_with_curves,
    solve_pnp,
)
from doper.model import DoperNet

NUM_KEYPOINTS = 6
IMAGE_SIZE = 96
SHARPEN = 40.0


@pytest.fixture(scope="module")
def setup():
    """Return (model, cfg, keypoints_3d, camera_matrix, image) for a tiny CPU model."""
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
        model.wh_head.bias.add_(6.0)  # random-init boxes would otherwise be degenerate
        model.kpt_head.out.weight.mul_(SHARPEN)
    model.eval()
    keypoints_3d = np.random.default_rng(0).uniform(-0.05, 0.05, (NUM_KEYPOINTS, 3))
    focal = 1.2 * IMAGE_SIZE
    camera = np.array([[focal, 0, IMAGE_SIZE / 2], [0, focal, IMAGE_SIZE / 2], [0, 0, 1.0]])
    image = np.random.default_rng(1).integers(0, 256, (IMAGE_SIZE, 128, 3), dtype=np.uint8)
    return model, cfg, keypoints_3d, camera, image


def assert_detection(det: Detection) -> None:
    """Check the types and shapes of one detection."""
    assert det.bbox_xyxy.shape == (4,)
    assert isinstance(det.score, float)
    assert det.keypoints.shape == (NUM_KEYPOINTS, 2)
    assert det.keypoint_scores.shape == (NUM_KEYPOINTS,)


def test_infer_image_returns_detections(setup):
    model, cfg, _, _, image = setup
    dets = infer_image(model, image, cfg, "cpu", score_thr=0.0, max_det=2)
    assert isinstance(dets, list) and 1 <= len(dets) <= 2
    for det in dets:
        assert_detection(det)
    dets, mask = infer_image(model, image, cfg, "cpu", score_thr=0.0, return_mask=True)
    assert mask.shape == image.shape[:2] and mask.dtype == np.float32


@pytest.mark.parametrize("return_mask", [False, True])
def test_infer_image_tta_returns_single_detection(setup, return_mask):
    model, cfg, keypoints_3d, camera, image = setup
    result = infer_image_tta(
        model,
        image,
        cfg,
        "cpu",
        score_thr=0.0,
        keypoints_3d=keypoints_3d,
        camera_matrix=camera,
        return_mask=return_mask,
    )
    dets, mask = result if return_mask else (result, None)
    assert len(dets) == 1
    assert_detection(dets[0])
    if return_mask:
        assert mask.shape == image.shape[:2]
    else:
        assert isinstance(result, list)


def test_infer_keypoints_in_box_runs_in_supplied_box(setup):
    model, cfg, keypoints_3d, camera, image = setup
    box = np.array([10.0, 10.0, 80.0, 70.0], dtype=np.float32)
    det = infer_keypoints_in_box(
        model, image, cfg, "cpu", box, keypoints_3d=keypoints_3d, camera_matrix=camera
    )
    assert_detection(det)
    assert det.score == 1.0
    assert np.array_equal(det.bbox_xyxy, box)
    with pytest.raises(ValueError, match="shape"):
        infer_keypoints_in_box(model, image, cfg, "cpu", box[:3])


def test_solve_pnp_fills_pose_and_skips_few_points(setup):
    _, _, keypoints_3d, camera, _ = setup
    rvec = np.array([0.1, -0.2, 0.3])
    tvec = np.array([0.0, 0.0, 0.5])
    projected = cv2.projectPoints(keypoints_3d, rvec, tvec, camera, np.zeros(5))[0].reshape(-1, 2)
    det = Detection(
        bbox_xyxy=np.zeros(4, np.float32),
        score=1.0,
        keypoints=projected.astype(np.float32),
        keypoint_scores=np.ones(NUM_KEYPOINTS, np.float32),
    )
    solve_pnp(det, keypoints_3d, camera, None)
    assert det.rvec is not None and det.rvec.size == 3
    assert det.tvec is not None and det.tvec.size == 3
    assert det.reproj_error_px is not None and det.reproj_error_px < 1.0
    det.keypoint_scores = np.zeros(NUM_KEYPOINTS, np.float32)
    det.rvec = det.tvec = det.reproj_error_px = None
    solve_pnp(det, keypoints_3d, camera, None)
    assert det.rvec is None and det.tvec is None


def test_refine_with_curves_guards(setup):
    _, _, _, camera, _ = setup
    det = Detection(
        bbox_xyxy=np.zeros(4, np.float32),
        score=1.0,
        keypoints=np.zeros((NUM_KEYPOINTS, 2), np.float32),
        keypoint_scores=np.ones(NUM_KEYPOINTS, np.float32),
    )
    schema = object()
    with pytest.raises(ValueError, match="initial PnP pose"):
        refine_detection_pose_with_curves(det, schema, camera, 32)  # type: ignore[arg-type]
    det.rvec = np.zeros(3)
    det.tvec = np.array([0.0, 0.0, 0.5])
    with pytest.raises(ValueError, match="curve maps"):
        refine_detection_pose_with_curves(det, schema, camera, 32)  # type: ignore[arg-type]
