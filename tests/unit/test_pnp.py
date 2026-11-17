# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

import doper
from doper import multiview
from doper.infer import Detection, solve_pnp
from doper.pnp import (
    DEFAULT_PNP_CONFIDENCE,
    DEFAULT_PNP_MAX_ITERATIONS,
    DEFAULT_PNP_RANDOM_SEED,
    DEFAULT_PNP_REPROJECTION_ERROR_PX,
)
from doper.pseudo_labels.gate import GateSpec
from doper.recipes.schema import DEFAULT_GATE, load_recipe


def _detection(keypoints: np.ndarray) -> Detection:
    return Detection(
        bbox_xyxy=np.zeros(4, dtype=np.float32),
        score=1.0,
        keypoints=keypoints.astype(np.float32),
        keypoint_scores=np.ones(len(keypoints), dtype=np.float32),
    )


def test_solve_pnp_uses_sequential_usac_4k_by_default(monkeypatch) -> None:
    object_points = np.array(
        [
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.5, 0.5, 0.0],
            [-0.5, 0.5, 0.0],
        ],
        dtype=np.float64,
    )
    camera_matrix = np.array(
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.array([[0.0], [0.0], [5.0]], dtype=np.float64)
    image_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, np.zeros(5))
    det = _detection(image_points.reshape(-1, 2))
    captured = {}

    def fake_solve_pnp_ransac(obj, img, camera, dist, *, params):
        captured["params"] = params
        return True, camera.copy(), rvec.copy(), tvec.copy(), np.arange(4)[:, None]

    monkeypatch.setattr(cv2, "solvePnPRansac", fake_solve_pnp_ransac)

    solve_pnp(det, object_points, camera_matrix, None)

    params = captured["params"]
    assert params.maxIterations == DEFAULT_PNP_MAX_ITERATIONS == 4000
    assert params.threshold == pytest.approx(DEFAULT_PNP_REPROJECTION_ERROR_PX)
    assert params.confidence == pytest.approx(DEFAULT_PNP_CONFIDENCE)
    assert params.isParallel is False
    assert params.randomGeneratorState == DEFAULT_PNP_RANDOM_SEED
    np.testing.assert_allclose(det.rvec, rvec)
    np.testing.assert_allclose(det.tvec, tvec)
    assert det.reproj_error_px == pytest.approx(0.0, abs=1e-5)


def test_solve_pnp_recovers_pose_with_outliers() -> None:
    object_points = np.array(
        [
            [-0.8, -0.5, -0.2],
            [0.7, -0.6, -0.1],
            [0.9, 0.5, 0.0],
            [-0.6, 0.7, 0.1],
            [-0.4, -0.2, 0.8],
            [0.5, -0.3, 0.9],
            [0.6, 0.6, 0.7],
            [-0.7, 0.4, 0.6],
            [0.0, -0.8, 0.4],
            [0.8, 0.0, 0.5],
            [0.0, 0.8, 0.3],
            [-0.8, 0.0, 0.2],
        ],
        dtype=np.float64,
    )
    camera_matrix = np.array(
        [[700.0, 0.0, 320.0], [0.0, 710.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    expected_rvec = np.array([[0.15], [-0.08], [0.04]], dtype=np.float64)
    expected_tvec = np.array([[0.1], [-0.2], [7.0]], dtype=np.float64)
    image_points, _ = cv2.projectPoints(
        object_points,
        expected_rvec,
        expected_tvec,
        camera_matrix,
        np.zeros(5),
    )
    image_points = image_points.reshape(-1, 2)
    image_points[[1, 6, 10]] += np.array([[90.0, -70.0], [-110.0, 80.0], [75.0, 100.0]])
    det = _detection(image_points)

    solve_pnp(det, object_points, camera_matrix, None)

    assert det.rvec is not None
    assert det.tvec is not None
    np.testing.assert_allclose(det.rvec, expected_rvec, atol=1e-4)
    np.testing.assert_allclose(det.tvec, expected_tvec, atol=1e-4)


def test_epnp_appears_only_as_the_explicit_opt_in() -> None:
    """USAC is the default everywhere; only doper.pnp knows EPnP, behind ``method="epnp"``."""
    root = Path(doper.__file__).resolve().parent
    users = [
        path.relative_to(root).as_posix()
        for path in sorted(root.rglob("*.py"))
        if "SOLVEPNP_EPNP" in path.read_text()
    ]
    assert users == ["pnp.py"]


def test_defaults_are_usac() -> None:
    assert DEFAULT_GATE.pnp_method == "usac"
    assert GateSpec().pnp_method == "usac"
    with pytest.raises(ValueError, match="pnp_method"):
        GateSpec(pnp_method="sqpnp")  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["handal", "hot3d", "hopev2", "tless"])
def test_paper_recipes_keep_the_papers_epnp(name: str) -> None:
    recipe = load_recipe(name)
    assert recipe.adapt.gate.pnp_method == "epnp"
    if recipe.multiview is not None:
        assert recipe.multiview.gate.pnp_method == "epnp"
        assert replace(recipe.multiview.gate, pnp_method="usac") == DEFAULT_GATE


@pytest.mark.parametrize(
    ("call", "expected_px", "expected_iterations"),
    [("single_view_inliers", 8.0, 100), ("strict_ransac_inlier_count", 3.0, 200)],
)
def test_multiview_ransac_methods(monkeypatch, call, expected_px, expected_iterations) -> None:
    rng = np.random.default_rng(5)
    object_points = rng.uniform(-0.04, 0.04, size=(12, 3))
    camera_matrix = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    rvec = np.array([[0.1], [-0.05], [0.02]])
    tvec = np.array([[0.01], [-0.01], [0.5]])
    image_points = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, np.zeros(5))[0]
    image_points = image_points.reshape(-1, 2)
    captured: list[dict] = []

    def fake_solve_pnp_ransac(obj, img, camera, dist, **kwargs):
        captured.append(kwargs)
        inliers = np.arange(len(obj))[:, None]
        if "params" in kwargs:
            return True, camera.copy(), rvec.copy(), tvec.copy(), inliers
        return True, rvec.copy(), tvec.copy(), inliers

    monkeypatch.setattr(cv2, "solvePnPRansac", fake_solve_pnp_ransac)
    for method in ("usac", "epnp"):
        if call == "single_view_inliers":
            mask = np.ones(len(object_points), dtype=bool)
            multiview.single_view_inliers(
                object_points, image_points, mask, camera_matrix, np.zeros(5), 8.0, method
            )
        else:
            multiview.strict_ransac_inlier_count(
                _detection(image_points),
                object_points,
                multiview.CameraView(camera_matrix=camera_matrix, dist_coeffs=np.zeros(5)),
                keypoint_score_threshold=0.05,
                ransac_px=3.0,
                pnp_method=method,
            )

    usac, epnp = captured
    assert usac["params"].threshold == pytest.approx(expected_px)
    assert usac["params"].maxIterations == DEFAULT_PNP_MAX_ITERATIONS
    assert usac["params"].randomGeneratorState == DEFAULT_PNP_RANDOM_SEED
    assert epnp == {
        "reprojectionError": expected_px,
        "iterationsCount": expected_iterations,
        "flags": cv2.SOLVEPNP_EPNP,
    }
