# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Heatmap likelihood pose refinement (``doper.refinement``)."""

from __future__ import annotations

import dataclasses
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from doper.refinement import (
    HeatmapObservation,
    LbfgsbAudit,
    PowellAudit,
    RefineSettings,
    observation_from_detection,
    prepare_observations,
    refine_pose,
)

CROP_SIZE = 256
BINS = 64
SIGMA_BINS = 1.5
CAMERA = np.array([[900.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]])
TRUE_ROTATION = cv2.Rodrigues(np.array([0.3, -0.2, 0.1]))[0]
TRUE_TRANSLATION_MM = np.array([10.0, -5.0, 600.0])


@pytest.fixture(scope="module", autouse=True)
def numba_cache_dir(tmp_path_factory):
    """Keep Numba's on-disk kernel cache in a temp dir."""
    patch = pytest.MonkeyPatch()
    path = str(tmp_path_factory.mktemp("numba_cache"))
    patch.setenv("NUMBA_CACHE_DIR", path)
    numba = sys.modules.get("numba")
    if numba is not None:
        patch.setattr(numba.config, "CACHE_DIR", path, raising=False)
    yield path
    patch.undo()


def _support(angle_deg: float) -> np.ndarray:
    rotation = cv2.Rodrigues(np.array([0.05, np.radians(angle_deg), 0.0]))[0]
    center = np.array([0.0, 0.0, 600.0])
    relative = np.eye(4)
    relative[:3, :3] = rotation
    relative[:3, 3] = center - rotation @ center
    return relative


def _view(
    points_mm: np.ndarray,
    relative: np.ndarray,
    *,
    margin: float = 1.3,
    flat: bool = False,
    seed: int = 0,
) -> HeatmapObservation:
    """A view whose logits peak at the true pose's projected keypoints."""
    cam = (points_mm @ TRUE_ROTATION.T + TRUE_TRANSLATION_MM) @ relative[:3, :3].T
    cam = cam + relative[:3, 3]
    uv = (cam @ CAMERA.T)[:, :2] / np.maximum(cam[:, 2:3], 1.0)
    front = uv[cam[:, 2] > 1.0]
    center = front.mean(axis=0) if len(front) else np.array([320.0, 240.0])
    side = margin * (float(np.ptp(front, axis=0).max()) if len(front) else 200.0)
    crop_to_image = np.array(
        [
            [side / (CROP_SIZE - 1), 0.0, center[0] - side / 2],
            [0.0, side / (CROP_SIZE - 1), center[1] - side / 2],
        ]
    )
    image_to_crop = cv2.invertAffineTransform(crop_to_image)
    bins = (uv @ image_to_crop[:, :2].T + image_to_crop[:, 2]) * ((BINS - 1) / (CROP_SIZE - 1))
    grid = np.arange(BINS, dtype=np.float64)
    logits = -(
        (grid[None, None, :] - bins[:, 0, None, None]) ** 2
        + (grid[None, :, None] - bins[:, 1, None, None]) ** 2
    ) / (2 * SIGMA_BINS**2)
    logits += np.random.default_rng(seed).normal(0.0, 0.05, size=logits.shape)
    if flat:
        logits = np.zeros_like(logits)
    return HeatmapObservation(
        logits=logits.astype(np.float32),
        crop_to_image=crop_to_image,
        K=CAMERA,
        relative_mm=relative,
    )


@dataclasses.dataclass
class Problem:
    rvec: np.ndarray
    tvec: np.ndarray
    points_mm: np.ndarray
    observations: list[HeatmapObservation]


def _problem(
    views: int = 1, *, offset_mm=(4.0, -3.0, 8.0), margin: float = 1.3, flat: bool = False
) -> Problem:
    points = np.random.default_rng(3).uniform(-40.0, 40.0, size=(64, 3))
    observations = [
        _view(
            points,
            np.eye(4) if v == 0 else _support(25.0 * v * (-1) ** v),
            margin=margin,
            flat=flat,
            seed=v,
        )
        for v in range(views)
    ]
    rotation = cv2.Rodrigues(np.array([0.04, -0.03, 0.05]))[0] @ TRUE_ROTATION
    tvec = (TRUE_TRANSLATION_MM - np.asarray(offset_mm)).reshape(3, 1)
    return Problem(cv2.Rodrigues(rotation)[0], tvec, points, observations)


def _refine(problem: Problem, **settings) -> tuple:
    return refine_pose(
        problem.rvec,
        problem.tvec,
        problem.points_mm,
        problem.observations,
        CROP_SIZE,
        1.0,
        RefineSettings(**settings),
    )


@pytest.mark.parametrize("optimizer", ["lbfgsb", "powell"])
@pytest.mark.parametrize("views", [1, 2, 4])
def test_refinement_moves_toward_the_true_pose(optimizer, views):
    problem = _problem(views)
    rvec, tvec, audit = _refine(problem, optimizer=optimizer)
    assert audit.accepted and audit.final_nll < audit.initial_nll and audit.views == views
    before = np.linalg.norm(problem.tvec.ravel() - TRUE_TRANSLATION_MM)
    after = np.linalg.norm(np.asarray(tvec).ravel() - TRUE_TRANSLATION_MM)
    assert after < before / 4
    assert np.asarray(rvec).shape == (3, 1) and np.asarray(tvec).shape == (3, 1)


def test_lbfgsb_audit_has_every_v2_field():
    _, _, audit = _refine(_problem(1))
    assert isinstance(audit, LbfgsbAudit)
    record = audit.to_dict()
    assert record["audit_version"] == 2
    assert set(record) == {
        "initial_nll",
        "final_nll",
        "accepted",
        "success",
        "status",
        "message",
        "gradient_inf_norm",
        "iterations",
        "evaluations",
        "views",
        "audit_version",
        "delta_normalized",
        "delta_rotation_rad",
        "delta_translation_mm",
        "bound_hits",
        "any_bound_hit",
        "rotation_bound_hit",
        "translation_bound_hit",
        "positive_depth",
        "finite_objective",
        "improved",
        "projected_gradient_inf_norm",
        "rejection_reasons",
    }
    assert len(record["delta_normalized"]) == 6 and record["rejection_reasons"] == []
    assert record["delta_translation_mm"] == pytest.approx(
        [x * 30.0 for x in record["delta_normalized"][3:]]
    )


def test_powell_audit_has_only_the_source_fields():
    _, _, audit = _refine(_problem(1), optimizer="powell")
    assert isinstance(audit, PowellAudit)
    assert set(audit.to_dict()) == {
        "initial_nll",
        "final_nll",
        "accepted",
        "success",
        "iterations",
        "evaluations",
        "views",
    }


def test_translation_beyond_the_bound_hits_it():
    problem = _problem(1, offset_mm=(45.0, 0.0, 0.0))
    rvec, tvec, audit = _refine(problem)
    assert audit.accepted and audit.any_bound_hit and audit.translation_bound_hit
    assert audit.bound_hits[3] and audit.delta_translation_mm[0] == pytest.approx(30.0)
    assert np.asarray(tvec).ravel()[0] < TRUE_TRANSLATION_MM[0]


def test_flat_heatmaps_are_rejected_and_return_the_input_pose():
    problem = _problem(1, flat=True)
    rvec, tvec, audit = _refine(problem)
    assert not audit.accepted and audit.rejection_reasons == ("no_improvement",)
    assert rvec is problem.rvec and tvec is problem.tvec


def test_nonpositive_depth_in_a_support_view_rejects():
    problem = _problem(1)
    behind = np.eye(4)
    behind[:3, 3] = [0.0, 0.0, -600.0]
    problem.observations.append(_view(problem.points_mm, behind, flat=True))
    rvec, tvec, audit = _refine(problem)
    assert audit.improved and not audit.positive_depth and not audit.accepted
    assert audit.rejection_reasons == ("nonpositive_depth",)
    assert rvec is problem.rvec and tvec is problem.tvec
    _, _, powell = _refine(problem, optimizer="powell")
    assert not powell.accepted


def test_out_of_crop_keypoints_pay_the_invalid_penalty():
    tight = _problem(1, margin=0.6)
    _, _, audit = _refine(tight)
    # Every invalid keypoint costs -log(1e-12) ~ 27.6; a well-cropped view costs far less.
    _, _, wide = _refine(_problem(1))
    assert audit.initial_nll > wide.initial_nll + 10 * 27.6


def test_parallel_prep_within_1e_12_and_restores_threads(monkeypatch):
    numba = pytest.importorskip("numba")
    if numba.config.NUMBA_NUM_THREADS < 2:
        pytest.skip("needs at least two Numba threads to observe the limit")
    import importlib

    kernels = importlib.import_module("doper.refinement._numba_kernels")
    original = kernels.fused_log_softmax
    seen: list[int] = []

    def recording(logits, beta):
        seen.append(numba.get_num_threads())
        return original(logits, beta)

    monkeypatch.setattr(kernels, "fused_log_softmax", recording)
    problem = _problem(2)
    previous = numba.get_num_threads()
    numba.set_num_threads(2)
    try:
        reference = prepare_observations(problem.observations, CROP_SIZE, 1.0)
        parallel = prepare_observations(
            problem.observations, CROP_SIZE, 1.0, preparation="parallel", threads=1
        )
        assert seen == [1, 1]
        assert numba.get_num_threads() == 2
    finally:
        numba.set_num_threads(previous)
    for slow, fast in zip(reference, parallel, strict=True):
        assert fast.log_probabilities.dtype == np.float64
        np.testing.assert_array_equal(slow.relative_m, fast.relative_m)
        np.testing.assert_array_equal(slow.projection, fast.projection)
        assert np.max(np.abs(slow.log_probabilities - fast.log_probabilities)) <= 1e-12


def test_parallel_preparation_refines_like_numpy():
    problem = _problem(2)
    _, tvec_numpy, numpy_audit = _refine(problem)
    _, tvec_parallel, parallel_audit = _refine(problem, preparation="parallel")
    assert parallel_audit.accepted and numpy_audit.accepted
    np.testing.assert_allclose(tvec_parallel, tvec_numpy, atol=1e-3)


def test_analytic_gradient_matches_finite_difference():
    from doper.refinement.heatmap import HeatmapObjective

    problem = _problem(2)
    prepared = prepare_observations(problem.observations, CROP_SIZE, 1.0)
    objective = HeatmapObjective(problem.rvec, problem.tvec, problem.points_mm, prepared)
    rng = np.random.default_rng(0)
    for u in [np.zeros(6), *rng.uniform(-0.05, 0.05, size=(3, 6))]:
        _, gradient = objective(u)
        step = 1e-7
        numeric = np.array(
            [
                (objective(u + step * e)[0] - objective(u - step * e)[0]) / (2 * step)
                for e in np.eye(6)
            ]
        )
        assert np.max(np.abs(gradient - numeric)) <= 1e-7


def test_lbfgsb_without_numba_names_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "numba", None)
    monkeypatch.delitem(sys.modules, "doper.refinement._numba_kernels", raising=False)
    with pytest.raises(ImportError, match=r"doper\[refine\]"):
        _refine(_problem(1))
    with pytest.raises(ImportError, match=r"doper\[refine\]"):
        prepare_observations(_problem(1).observations, CROP_SIZE, 1.0, preparation="parallel")


def test_powell_needs_no_numba(monkeypatch):
    monkeypatch.setitem(sys.modules, "numba", None)
    monkeypatch.delitem(sys.modules, "doper.refinement._numba_kernels", raising=False)
    _, _, audit = _refine(_problem(1), optimizer="powell")
    assert audit.accepted


def test_powell_rejects_parallel_preparation():
    with pytest.raises(ValueError, match="Powell"):
        RefineSettings(optimizer="powell", preparation="parallel")


@pytest.mark.parametrize(
    "fields",
    [{"optimizer": "ceres"}, {"preparation": "gpu"}, {"threads": 0}],
)
def test_settings_reject_unknown_values(fields):
    with pytest.raises(ValueError):
        RefineSettings(**fields)


def test_observation_validates_shapes():
    good = _problem(1).observations[0]
    with pytest.raises(ValueError, match="logits"):
        dataclasses.replace(good, logits=good.logits[0])
    with pytest.raises(ValueError, match="crop_to_image"):
        dataclasses.replace(good, crop_to_image=np.eye(3))
    with pytest.raises(ValueError, match="K"):
        dataclasses.replace(good, K=np.eye(3) * 2)
    with pytest.raises(ValueError, match="relative_mm"):
        dataclasses.replace(good, relative_mm=np.eye(3))
    with pytest.raises(dataclasses.FrozenInstanceError):
        good.K = CAMERA  # type: ignore[misc]


def test_observation_from_detection_reads_raw_outputs():
    view = _problem(1).observations[0]
    det = SimpleNamespace(keypoint_heatmap=view.logits, crop_to_image=view.crop_to_image)
    observation = observation_from_detection(det, CAMERA)
    assert observation.logits is view.logits
    np.testing.assert_array_equal(observation.relative_mm, np.eye(4))
    np.testing.assert_array_equal(observation.crop_to_image, view.crop_to_image)
    support = _support(25.0)
    assert observation_from_detection(det, CAMERA, support).relative_mm is not support
    with pytest.raises(ValueError, match="return_raw_keypoint_outputs"):
        observation_from_detection(
            SimpleNamespace(keypoint_heatmap=None, crop_to_image=None), CAMERA
        )
