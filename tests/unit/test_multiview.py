# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Association and joint-PnP tests, ported from the source multiview association tests."""

import cv2
import numpy as np
import pytest

from doper import multiview
from doper.infer import Detection
from doper.multiview import (
    AssociationGate,
    CameraObservation,
    CameraView,
    RigidTransform,
    associate_detection_pairs,
    joint_pnp,
    select_fallback_detection,
    strict_ransac_inlier_count,
)


def _project(points, rvec, tvec, camera):
    return (
        cv2.projectPoints(points, rvec, tvec, camera, np.zeros((5, 1)))[0]
        .reshape(-1, 2)
        .astype(np.float32)
    )


def _detection(points, score, rvec, tvec):
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    return Detection(
        bbox_xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
        score=score,
        keypoints=points,
        keypoint_scores=np.ones(len(points), dtype=np.float32),
        rvec=np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        tvec=np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        reproj_error_px=0.5,
    )


def _camera():
    return np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])


def _view():
    return CameraView(camera_matrix=_camera(), dist_coeffs=np.zeros((5, 1)))


def _gate(ransac_px, max_reprojection_px):
    return AssociationGate(
        kp_score_thr=0.05,
        min_keypoints=4,
        ransac_px=ransac_px,
        max_reprojection_px=max_reprojection_px,
        min_depth=0.03,
        max_depth=1.0,
    )


def test_association_selects_lower_score_geometrically_consistent_instance():
    rng = np.random.default_rng(7)
    object_points = rng.uniform(-0.04, 0.04, size=(32, 3)).astype(np.float64)
    camera = _camera()
    rvec_a = np.array([[0.12], [-0.08], [0.04]])
    tvec_a = np.array([[0.01], [-0.015], [0.45]])
    rvec_a_to_b = np.array([[0.0], [0.03], [0.0]])
    tvec_a_to_b = np.array([[-0.10], [0.0], [0.005]])
    rvec_b, tvec_b = cv2.composeRT(rvec_a, tvec_a, rvec_a_to_b, tvec_a_to_b)[:2]
    keypoints_a = _project(object_points, rvec_a, tvec_a, camera)
    keypoints_b = _project(object_points, rvec_b, tvec_b, camera)
    wrong_keypoints_a = keypoints_a + np.array([85.0, -25.0], dtype=np.float32)
    high_score_wrong = _detection(wrong_keypoints_a, 0.95, rvec_a, tvec_a)
    lower_score_correct = _detection(keypoints_a, 0.70, rvec_a, tvec_a)
    detection_b = _detection(keypoints_b, 0.80, rvec_b, tvec_b)
    association = associate_detection_pairs(
        [high_score_wrong, lower_score_correct],
        [detection_b],
        object_points,
        _view(),
        _view(),
        RigidTransform(rvec=rvec_a_to_b, tvec=tvec_a_to_b),
        _gate(ransac_px=8.0, max_reprojection_px=2.0),
    )
    assert association is not None
    assert association.index_a == 1
    assert association.index_b == 0
    assert association.accepted
    assert max(association.reprojection_a, association.reprojection_b) < 0.1


def test_association_keeps_best_pair_but_marks_failed_gate():
    rng = np.random.default_rng(8)
    object_points = rng.uniform(-0.04, 0.04, size=(24, 3)).astype(np.float64)
    camera = _camera()
    rvec = np.zeros((3, 1))
    tvec = np.array([[0.0], [0.0], [0.5]])
    rvec_a_to_b = np.zeros((3, 1))
    tvec_a_to_b = np.array([[-0.10], [0.0], [0.0]])
    rvec_b, tvec_b = cv2.composeRT(rvec, tvec, rvec_a_to_b, tvec_a_to_b)[:2]
    points_a = _project(object_points, rvec, tvec, camera)
    points_b = _project(object_points, rvec_b, tvec_b, camera)
    points_b[:, 1] += np.linspace(-30.0, 30.0, len(points_b))
    association = associate_detection_pairs(
        [_detection(points_a, 0.9, rvec, tvec)],
        [_detection(points_b, 0.9, rvec_b, tvec_b)],
        object_points,
        _view(),
        _view(),
        RigidTransform(rvec=rvec_a_to_b, tvec=tvec_a_to_b),
        _gate(ransac_px=100.0, max_reprojection_px=0.01),
    )
    assert association is not None
    assert not association.accepted


def test_fallback_prefers_ransac_inliers_over_confident_keypoint_count(monkeypatch):
    points = np.zeros((64, 2), dtype=np.float32)
    pose_rvec = np.zeros((3, 1))
    pose_tvec = np.array([[0.0], [0.0], [0.5]])
    detection_a = _detection(points, 0.8, pose_rvec, pose_tvec)
    detection_b = _detection(points, 0.9, pose_rvec, pose_tvec)
    detection_a.keypoint_scores[-1] = 0.0  # A: 63 confident, B: 64

    def fake_inlier_count(detection, *_args, **_kwargs):
        return 20 if detection is detection_a else 5

    monkeypatch.setattr(multiview, "strict_ransac_inlier_count", fake_inlier_count)
    camera = np.eye(3, dtype=np.float64)
    view = CameraView(camera_matrix=camera, dist_coeffs=np.zeros(5))
    selected = select_fallback_detection(
        detection_a,
        detection_b,
        np.zeros((64, 3), dtype=np.float64),
        view,
        view,
        keypoint_score_threshold=0.05,
        ransac_px=3.0,
    )
    assert selected is not None
    camera_label, detection, confident, inliers = selected
    assert camera_label == "A"
    assert detection is detection_a
    assert confident == 63
    assert inliers == 20


def test_strict_ransac_inlier_count_accepts_exact_correspondences():
    rng = np.random.default_rng(9)
    object_points = rng.uniform(-0.04, 0.04, size=(32, 3)).astype(np.float64)
    camera = _camera()
    rvec = np.array([[0.1], [-0.05], [0.02]])
    tvec = np.array([[0.01], [-0.01], [0.5]])
    points = _project(object_points, rvec, tvec, camera)
    detection = _detection(points, 0.9, rvec, tvec)
    count = strict_ransac_inlier_count(
        detection,
        object_points,
        CameraView(camera_matrix=camera, dist_coeffs=np.zeros(5)),
        keypoint_score_threshold=0.05,
        ransac_px=3.0,
    )
    assert count == len(object_points)


def test_joint_pnp_recovers_pose_and_returns_none_without_observations():
    rng = np.random.default_rng(3)
    points = rng.uniform(-0.04, 0.04, size=(40, 3))
    camera = _camera()
    rvec = np.array([[0.1], [0.2], [-0.1]])
    tvec = np.array([[0.0], [0.01], [0.5]])
    rig = RigidTransform(rvec=np.array([[0.0], [0.1], [0.0]]), tvec=np.array([[-0.1], [0], [0]]))
    rvec_b, tvec_b = cv2.composeRT(rvec, tvec, rig.rvec, rig.tvec)[:2]
    scores = np.ones(40)
    cameras = [
        CameraObservation(_project(points, rvec, tvec, camera), scores, camera, np.zeros(5)),
        CameraObservation(
            _project(points, rvec_b, tvec_b, camera),
            scores,
            camera,
            np.zeros(5),
            rvec_ref2cam=rig.rvec,
            tvec_ref2cam=rig.tvec,
        ),
    ]
    result = joint_pnp(points, cameras, rvec + 0.05, tvec + 0.01, 0.3)
    assert result is not None
    np.testing.assert_allclose(result.tvec, tvec, atol=1e-4)
    assert result.n_obs == 80
    assert len(result.per_cam_reproj_px) == 2
    assert joint_pnp(points, cameras, rvec, tvec, 2.0) is None
    with pytest.raises(ValueError):
        joint_pnp(points, cameras, rvec, tvec, 0.3, loss="nonsense")
