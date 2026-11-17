# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cv2
import numpy as np

from doper.geometry_vis import compute_3d_bbox, draw_axes, draw_box3d, project_points


def _camera() -> np.ndarray:
    return np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])


def test_project_points_matches_cv2_projectpoints() -> None:
    rng = np.random.default_rng(0)
    points = rng.uniform(-0.05, 0.05, size=(16, 3))
    rvec = np.array([[0.1], [-0.2], [0.05]])
    tvec = np.array([[0.01], [0.02], [0.5]])
    dist = np.array([[0.01], [-0.02], [0.0], [0.0], [0.0]])
    expected, _ = cv2.projectPoints(points, rvec, tvec, _camera(), dist)
    actual = project_points(points, rvec, tvec, _camera(), dist)
    assert actual.shape == (16, 2)
    np.testing.assert_array_equal(actual, expected.reshape(-1, 2))


def test_compute_3d_bbox_returns_eight_corners_spanning_extent() -> None:
    points = np.array([[0.0, -1.0, 2.0], [3.0, 1.0, 4.0], [1.0, 0.0, 3.0]])
    corners = compute_3d_bbox(points)
    assert corners.shape == (8, 3)
    np.testing.assert_array_equal(corners.min(axis=0), points.min(axis=0))
    np.testing.assert_array_equal(corners.max(axis=0), points.max(axis=0))


def test_draw_helpers_mark_the_frame_and_skip_non_finite_projections() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    rvec = np.zeros((3, 1))
    tvec = np.array([[0.0], [0.0], [0.5]])
    draw_axes(frame, rvec, tvec, _camera(), np.zeros((5, 1)), 0.05)
    assert frame.any()
    box_frame = np.zeros_like(frame)
    corners = project_points(
        compute_3d_bbox(np.array([[-0.05] * 3, [0.05] * 3])), rvec, tvec, _camera(), np.zeros(5)
    )
    draw_box3d(box_frame, corners)
    assert box_frame.any()
    blank = np.zeros_like(frame)
    draw_box3d(blank, np.full((8, 2), np.nan))
    assert not blank.any()
