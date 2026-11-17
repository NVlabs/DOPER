# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SHIRT Table 2 filters (``doper.shirt.policies``): PnP UKF, UKF + RTS and keypoint UKF.

The ``*_matches_source`` tests compare the port with ``golden/source_shirt_policies.json``, what
the research filters (research source snapshot) produce on the ``shirt_tracks`` fixture (see
that module); the
parity cases ``shirt_nis_rank``, ``shirt_rts``, ``shirt_keypoint_ukf`` and ``shirt_summaries``
run both sides in one environment. ``test_exact_replay_when_artifacts_supplied`` replays the
paper's Table 2 from the original artifacts named by ``DOPER_SHIRT_ARTIFACTS`` (skipped without
it) and ``test_bayesfilter_pin_equivalence`` reruns the filters under the paper's bayesfilter
pin in the interpreter named by ``DOPER_BAYESFILTER_PIN_PYTHON`` (skipped without it).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import shirt_tracks as st
import torch
from golden_floats import approx_floats

from doper import cli, infer
from doper.shirt import data, policies
from doper.speedplus.checkpoints import LAYOUT_SHA_KEY, load_layout
from doper.speedplus.definitions import LayoutPins
from doper.speedplus.geometry import Camera

pytest.importorskip("bayesfilter")

GOLDEN = Path(__file__).parent / "golden" / st.GOLDEN_NAME
REPLAY_PINS = Path(__file__).parent / "golden" / "shirt_replay_pins.json"
REPLAY_ENV = "DOPER_SHIRT_ARTIFACTS"
PIN_PYTHON_ENV = "DOPER_BAYESFILTER_PIN_PYTHON"
PIN_COMMIT = "d895dca"
PIN_TOLERANCE = 1.0e-12
REPLAY_TOLERANCE = 1.0e-9
# Table 2 (mean rotation deg, mean translation m) per trajectory: the exact rows come from the
# research summary.json; RTS and framewise only exist rounded in the research docs, so they are
# compared at the precision written here.
REPLAY_EXACT = {
    "pnp-ukf": {"roe1": (0.9549403018, 0.1290473370), "roe2": (0.7487728713, 0.0394901646)},
    "keypoint-ukf": {"roe1": (0.7583036659, 0.1389535996), "roe2": (0.6762192271, 0.0515463273)},
}
REPLAY_ROUNDED = {
    "pnp-ukf-rts": {"roe1": ("0.5757", "0.12860"), "roe2": ("0.5469", "0.03869")},
    "framewise": {"roe1": ("7.245", "0.178"), "roe2": ("2.351", "0.0748")},
}
REPLAY_PREDICTION_ONLY = {"roe1": 60, "roe2": 23}


def _camera() -> Camera:
    return Camera(st.CAMERA_MATRIX, st.DISTORTION, st.CAMERA_SIZE, st.CAMERA_SIZE)


def _source_golden_or_skip() -> dict:
    """The filter golden without its versions; skip unless this env has the same builds."""
    golden = json.loads(GOLDEN.read_text())
    recorded, current = golden.pop("versions"), st.library_versions()
    if recorded != current:
        pytest.skip(f"the filter golden was produced with {recorded}, this env has {current}; "
                    f"{st.REGENERATE}")  # fmt: skip
    return approx_floats(golden)


@pytest.fixture(scope="module")
def golden() -> dict:
    return _source_golden_or_skip()


def _nis_rank(roe: str, records: list | None = None) -> policies.FilterRun:
    records = st.candidate_records(roe) if records is None else records
    return policies.nis_rank(records, st.make_metadata(st.SLOTS[roe]))


def _keypoint(roe: str, records: list | None = None, **kwargs) -> policies.FilterRun:
    records = st.candidate_records(roe) if records is None else records
    return policies.keypoint_ukf(
        records, st.heatmap_cache(roe), st.make_metadata(st.SLOTS[roe]), st.make_points_3d(),
        _camera(), **kwargs,
    )  # fmt: skip


def test_chi_square_gates_and_noise_are_the_paper_settings():
    brief = {"p90": 10.6446406757, "p95": 12.5915872437, "p99": 16.8118938298,
             "p999": 22.4577444848}  # fmt: skip
    assert {key: round(value, 10) for key, value in policies.CHI2_6DOF.items()} == brief
    assert policies.CHI2_2DOF_P99 == 9.21034037197618
    assert (policies.ANGULAR_ACCEL_STD_RAD_S2, policies.LINEAR_ACCEL_STD_M_S2) == (1e-7, 1e-6)
    assert policies.RTS_PASSES == 1
    assert "SHIRT-tuned" in policies._settings()["process_noise"]


@pytest.mark.source_golden
@pytest.mark.parametrize("roe", st.TRAJECTORIES)
def test_nis_rank_matches_source(golden, roe):
    run = _nis_rank(roe)
    records, summary = policies.pnp_ukf_outputs(run)
    assert st.canonical_nis(records, summary) == golden["nis_rank"][roe]
    assert summary["empty_candidate_frames"] == []
    assert summary["prediction_only_frames"] == summary["rejected_measurements"]


def test_nis_rank_rejects_flips(golden):
    """The fixture's all-flip frames predict only; raw flips select a correct re-crop."""
    run = _nis_rank("roe1")
    for index, event in st.EVENTS["roe1"].items():
        frame = run.frames[index]
        if event == "all_flip":
            assert frame["update_kind"] == policies.PREDICTION_ONLY
        elif event == "raw_flip":
            assert frame["selected_index"] == 1
    _, summary = policies.pnp_ukf_outputs(run)
    assert summary["rotation_gt30_count"] == 0


@pytest.mark.source_golden
@pytest.mark.parametrize("roe", st.TRAJECTORIES)
def test_rts_single_pass_matches_source(golden, roe):
    smoothed = policies.rts(_nis_rank(roe))
    assert smoothed.method == policies.NIS_RANK_RTS
    records, summary = policies.rts_outputs(smoothed)
    assert st.canonical_rts(records, summary) == golden["rts"][roe]
    assert summary["rts_passes"] == 1


def test_rts_needs_a_smoothed_run():
    with pytest.raises(ValueError, match="rts\\(\\)"):
        policies.rts_outputs(_nis_rank("roe1"))


@pytest.mark.source_golden
@pytest.mark.parametrize("roe", st.TRAJECTORIES)
def test_keypoint_ukf_matches_source(golden, roe):
    records, summary = policies.keypoint_ukf_outputs(
        _keypoint(roe), _nis_rank(roe), st.heatmap_cache(roe)
    )
    assert st.canonical_keypoint(records, summary) == golden["keypoint_ukf"][roe]


def test_summaries_match_source(golden):
    helpers = {name: getattr(policies, name) for name in (
        "pose_metrics", "summarize", "summarize_pose_values", "method_summary",
        "paper_style_components")}  # fmt: skip
    assert st.run_summaries(helpers) == golden["summaries"]


def test_filter_golden_skips_on_other_builds(monkeypatch):
    monkeypatch.setattr(st, "library_versions", lambda: {"numpy": "0"})
    with pytest.raises(pytest.skip.Exception, match="research source snapshot"):
        _source_golden_or_skip()


def _reindexed(records: list, indices: list[int]) -> list:
    copied = json.loads(json.dumps(records[: len(indices)]))
    for record, index in zip(copied, indices, strict=True):
        record["index"] = index
    return copied


@pytest.mark.parametrize("indices", [[0, 1, 3, 4], [0, 1, 1, 2], [0, 2, 1, 3], [1, 2, 3, 4]],
                         ids=["gap", "duplicate", "unsorted", "not-from-zero"])  # fmt: skip
def test_filters_refuse_noncontiguous_indices(indices):
    records = _reindexed(st.candidate_records("roe1"), indices)
    with pytest.raises(ValueError, match="contiguous from 0"):
        policies.nis_rank(records, st.make_metadata(0))
    cache = {key: value[: len(indices)] for key, value in st.heatmap_cache("roe1").items()}
    with pytest.raises(ValueError, match="contiguous from 0"):
        policies.keypoint_ukf(records, cache, st.make_metadata(0), st.make_points_3d(), _camera())
    with pytest.raises(ValueError, match="at least one frame"):
        policies.check_frame_indices([])


def test_empty_candidate_list_predicts_and_is_counted():
    """A later frame whose raw PnP failed predicts only and is counted."""
    roe, index = st.EMPTY_FRAME
    full = policies.pnp_ukf_outputs(_nis_rank(roe))[1]
    records = st.with_empty_frame(st.candidate_records(roe), index)
    run = _nis_rank(roe, records)
    assert run.frames[index]["update_kind"] == policies.NO_CANDIDATES
    rows, summary = policies.pnp_ukf_outputs(run)
    assert rows[index]["selected_index"] is None and rows[index]["candidate_nis"] == []
    assert summary["empty_candidate_frames"] == [index]
    assert summary["prediction_only_frames"] == full["prediction_only_frames"] + 1
    assert summary["rejected_measurements"] == full["rejected_measurements"]
    assert summary["count"] == st.FRAMES
    rts_rows, rts_summary = policies.rts_outputs(policies.rts(run))
    assert rts_summary["empty_candidate_frames"] == [index] and len(rts_rows) == st.FRAMES
    with pytest.raises(RuntimeError, match="predict-only"):
        policies.rts(_nis_rank(roe, records), num_passes=2)
    direct = _keypoint(roe, records)
    assert len(direct.poses) == st.FRAMES


def test_frame0_without_candidates_refused():
    records = st.with_empty_frame(st.candidate_records("roe1"), 0)
    with pytest.raises(ValueError, match="no initial pose for frame 0"):
        policies.nis_rank(records, st.make_metadata(0))
    with pytest.raises(ValueError, match="no initial pose for frame 0"):
        _keypoint("roe1", records)


def test_keypoint_cache_must_match_the_frames():
    cache = st.heatmap_cache("roe1")
    cache["filenames"] = cache["filenames"][::-1]
    with pytest.raises(ValueError, match="frame for frame"):
        policies.keypoint_ukf(st.candidate_records("roe1"), cache, st.make_metadata(0),
                              st.make_points_3d(), _camera())  # fmt: skip
    cache = {**st.heatmap_cache("roe1"), "covariance_floor_image_px": np.asarray(2.0)}
    with pytest.raises(ValueError, match="covariance floor"):
        policies.keypoint_ukf(st.candidate_records("roe1"), cache, st.make_metadata(0),
                              st.make_points_3d(), _camera())  # fmt: skip


def test_unavailable_frames_predict_only():
    run = _keypoint("roe2")
    for index in st.UNAVAILABLE["roe2"]:
        assert run.frames[index]["update_kind"] == policies.NO_DETECTION
    assert run.frames[0]["update_kind"] == policies.INITIALIZATION


def test_pose_metrics_matches_the_candidate_errors():
    record = st.candidate_records("roe1")[2]
    candidate = record["candidates"][0]
    metrics = policies.pose_metrics(candidate["rvec"], candidate["tvec"],
                                    record["ground_truth_quaternion"],
                                    record["ground_truth_translation"])  # fmt: skip
    assert metrics["rotation_deg"] == pytest.approx(candidate["rotation_deg"], abs=1e-9)
    assert metrics["translation_m"] == pytest.approx(candidate["translation_m"], abs=1e-12)


# --- the heatmap cache and the CLI --------------------------------------------------------------

CROP_SIZE = 64
HEATMAP_SIGMA = 1.5
HEATMAP_PAD_PX = 20.0


class StubHeatmapModel:
    """Stand-in detector: Gaussian heatmaps at each frame's true landmark projections."""

    def __init__(self) -> None:
        self.calls: list[dict] = []


def _stub_detection(roe: str, index: int) -> SimpleNamespace | None:
    if index in st.UNAVAILABLE[roe]:
        return None
    rvec, tvec = st.truth_poses(roe)[index]
    import cv2

    pixels = cv2.projectPoints(st.make_points_3d(), rvec, tvec, st.CAMERA_MATRIX,
                               st.DISTORTION)[0].reshape(-1, 2)  # fmt: skip
    low = pixels.min(axis=0) - HEATMAP_PAD_PX
    scale = float((pixels.max(axis=0) + HEATMAP_PAD_PX - low).max()) / CROP_SIZE
    crop = (pixels - low) / scale
    grid = np.arange(CROP_SIZE, dtype=np.float64)
    distance = ((grid[None, None, :] - crop[:, None, None, 0]) ** 2
                + (grid[None, :, None] - crop[:, None, None, 1]) ** 2)  # fmt: skip
    return SimpleNamespace(
        keypoint_heatmap=(-distance / (2 * HEATMAP_SIGMA**2)).astype(np.float32),
        crop_to_image=np.array([[scale, 0.0, low[0]], [0.0, scale, low[1]]]),
        bbox_xyxy=np.array([*low, *(low + scale * CROP_SIZE)], dtype=np.float32),
        score=0.8, expected=pixels,
    )  # fmt: skip


@pytest.fixture
def stubbed(monkeypatch) -> StubHeatmapModel:
    model = StubHeatmapModel()

    def load_model(checkpoint, cfg, device):
        cfg.model.crop_size = CROP_SIZE
        return model

    def infer_image(stub, image, cfg, device, **kwargs):
        stub.calls.append(dict(kwargs))
        detection = _stub_detection(*st.frame_of_code(int(image[0, 0, 0])))
        return [] if detection is None else [detection]

    monkeypatch.setattr(infer, "load_model", load_model)
    monkeypatch.setattr(infer, "infer_image", infer_image)
    return model


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def tree(tmp_path: Path) -> dict:
    paths = st.write_tree(tmp_path / "tree")
    st.write_images(paths["shirt"])
    runs = tmp_path / "runs"
    shutil.copytree(paths["candidates"], runs / "candidates")
    checkpoint = tmp_path / "epoch_10.pth"
    torch.save({"model": {"w": torch.ones(1)}, LAYOUT_SHA_KEY: _sha256(paths["layout"])},
               checkpoint)  # fmt: skip
    return {**paths, "runs": runs, "checkpoint": checkpoint}


def test_cache_heatmap_measurements(tree, stubbed, tmp_path):
    out = tmp_path / "cache"
    result = policies.cache_heatmap_measurements(
        tree["checkpoint"], tree["shirt"], tree["layout"], tree["candidates"], out,
        trajectories=st.TRAJECTORIES, allow_unpinned_checkpoint=True,
    )  # fmt: skip
    assert result["roe1"]["available"] == st.FRAMES - len(st.UNAVAILABLE["roe1"])
    assert all(call["return_raw_keypoint_outputs"] and call["max_det"] == 1
               and call["score_thr"] == 0.0 for call in stubbed.calls)  # fmt: skip
    cache = policies.load_heatmap_cache(policies.cache_path(out, "roe1", st.DOMAIN))
    assert cache["points"].dtype == np.float32 and cache["covariances"].shape[1:] == (12, 2, 2)
    assert str(cache["layout_sha256"]) == _sha256(tree["layout"])
    assert float(cache["covariance_floor_image_px"]) == 1.0
    np.testing.assert_allclose(cache["points"][1], _stub_detection("roe1", 1).expected, atol=0.6)
    assert not cache["available"][st.UNAVAILABLE["roe1"][0]]
    assert np.isnan(cache["points"][st.UNAVAILABLE["roe1"][0]]).all()
    with pytest.raises(FileExistsError):
        policies.cache_heatmap_measurements(
            tree["checkpoint"], tree["shirt"], tree["layout"], tree["candidates"], out,
            trajectories=st.TRAJECTORIES, allow_unpinned_checkpoint=True,
        )  # fmt: skip
    run = policies.keypoint_ukf(st.candidate_records("roe1"), cache, st.make_metadata(0),
                                st.make_points_3d(), _camera())  # fmt: skip
    assert run.frames[1]["update_kind"] == policies.KEYPOINT_UPDATE


def test_cache_refuses_layout_mismatch_and_unpinned(tree, stubbed, tmp_path):
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"keypoints_3d": (st.make_points_3d() * 2).tolist()}))
    with pytest.raises(ValueError, match="layout"):
        policies.cache_heatmap_measurements(
            tree["checkpoint"], tree["shirt"], other, tree["candidates"], tmp_path / "c",
            trajectories=("roe1",), allow_unpinned_checkpoint=True,
        )  # fmt: skip
    with pytest.raises(ValueError, match="allow-unpinned-checkpoint"):
        policies.cache_heatmap_measurements(
            tree["checkpoint"], tree["shirt"], tree["layout"], tree["candidates"],
            tmp_path / "c", trajectories=("roe1",),
        )  # fmt: skip


def _cli(tree: dict, method: str, *extra: str) -> list[str]:
    return ["evaluate", "shirt", "--method", method, "--root", str(tree["shirt"]),
            "--runs", str(tree["runs"]), *extra]  # fmt: skip


@pytest.mark.parametrize("method", ["pnp-ukf", "pnp-ukf-rts"])
def test_cli_pnp_filters(tree, method):
    assert cli.main([*_cli(tree, method), "--dry-run"]) == 0
    assert not (tree["runs"] / method).exists()
    assert cli.main(_cli(tree, method)) == 0
    summary = json.loads((tree["runs"] / method / "summary.json").read_text())
    assert summary["settings"]["angular_accel_std_rad_s2"] == 1e-7
    assert summary["ground_truth_used_by_filters_or_gates"] is False
    assert summary["score_definition"] == "paper"
    assert summary["layout_formatting_deviation"] is None  # the fixture has no candidates summary
    roe1 = summary["trajectories"]["roe1"]["lightbox"]
    expected = policies.pnp_ukf_outputs(_nis_rank("roe1"))[1]
    if method == "pnp-ukf":
        assert roe1["rotation_deg_mean"] == expected["rotation_deg_mean"]
    else:
        assert roe1["causal"]["rotation_deg_mean_std"][0] == expected["rotation_deg_mean"]
    lines = (tree["runs"] / method / "roe1" / "lightbox" / "records.jsonl").read_text()
    assert "NaN" not in lines and len(lines.splitlines()) == st.FRAMES
    assert cli.main(_cli(tree, method)) == 1


def test_cli_keypoint_ukf(tree, stubbed):
    args = _cli(tree, "keypoint-ukf", "--checkpoint", str(tree["checkpoint"]), "--keypoints-3d",
                str(tree["layout"]), "--device", "cpu", "--allow-unpinned-checkpoint")  # fmt: skip
    assert cli.main([*args, "--dry-run"]) == 0
    assert cli.main(args) == 0
    out = tree["runs"] / "keypoint-ukf"
    assert policies.cache_path(out, "roe2", st.DOMAIN).is_file()
    summary = json.loads((out / "summary.json").read_text())
    roe2 = summary["trajectories"]["roe2"]["lightbox"]
    assert roe2["direct_prediction_only_frames"] >= len(st.UNAVAILABLE["roe2"])
    assert summary["settings"]["landmark_nis_gate_2d"] == policies.CHI2_2DOF_P99


def test_cli_keypoint_ukf_needs_checkpoint_and_layout(tree):
    assert cli.main(_cli(tree, "keypoint-ukf")) == 1


def _keypoint_args(tree: dict, *extra: str) -> list[str]:
    model = ("--checkpoint", str(tree["checkpoint"]), "--keypoints-3d", str(tree["layout"]))
    return _cli(tree, "keypoint-ukf", *model, "--device", "cpu", "--allow-unpinned-checkpoint",
                *extra)  # fmt: skip


def _cache_bytes(tree: dict) -> dict[str, bytes]:
    out = tree["runs"] / "keypoint-ukf"
    return {roe: policies.cache_path(out, roe, st.DOMAIN).read_bytes() for roe in st.TRAJECTORIES}


def test_cli_keypoint_ukf_rerun_keeps_caches_and_detects_changes(tree, stubbed, capsys):
    args = _keypoint_args(tree)
    assert cli.main(args) == 0
    out = tree["runs"] / "keypoint-ukf"
    caches = _cache_bytes(tree)
    calls = len(stubbed.calls)
    # A re-run without --force is refused before any caching: the caches stay as they were.
    capsys.readouterr()
    assert cli.main(args) == 1
    assert "already exist" in capsys.readouterr().out
    assert _cache_bytes(tree) == caches and len(stubbed.calls) == calls
    # Without the summary, matching caches are reused, not rebuilt.
    (out / "summary.json").unlink()
    assert cli.main(args) == 0
    assert "reusing the heatmap caches" in capsys.readouterr().out
    assert _cache_bytes(tree) == caches and len(stubbed.calls) == calls
    # A --limit change is detected up front with a clear message, the caches untouched.
    (out / "summary.json").unlink()
    assert cli.main([*args, "--limit", "5"]) == 1
    message = capsys.readouterr().out
    assert "does not fit this run" in message and "--force" in message
    assert _cache_bytes(tree) == caches
    # So is another checkpoint.
    torch.save({"model": {"w": torch.zeros(1)}, LAYOUT_SHA_KEY: _sha256(tree["layout"])},
               tree["checkpoint"])  # fmt: skip
    shutil.rmtree(tree["runs"] / "candidates")
    shutil.copytree(tree["candidates"], tree["runs"] / "candidates")
    assert cli.main(args) == 1
    assert "checkpoint" in capsys.readouterr().out
    # --force rebuilds them.
    assert cli.main([*args, "--limit", "5", "--force"]) == 0
    cache = policies.load_heatmap_cache(policies.cache_path(out, "roe1", st.DOMAIN))
    assert len(cache["filenames"]) == 5 and str(cache["checkpoint_sha256"]) == _sha256(
        tree["checkpoint"]
    )


def test_keypoint_cache_problem_names_what_differs(tree, stubbed, tmp_path):
    out = tmp_path / "cache"
    policies.cache_heatmap_measurements(
        tree["checkpoint"], tree["shirt"], tree["layout"], tree["candidates"], out,
        trajectories=("roe1",), allow_unpinned_checkpoint=True,
    )  # fmt: skip
    path = policies.cache_path(out, "roe1", st.DOMAIN)
    layout = load_layout(tree["layout"]).identity
    records = st.candidate_records("roe1")
    checkpoint = _sha256(tree["checkpoint"])
    assert policies.heatmap_cache_problem(path, records, layout, checkpoint) is None
    assert policies.heatmap_cache_problem(tmp_path / "none.npz", records, layout,
                                          checkpoint) == "missing"  # fmt: skip
    assert "frame for frame" in policies.heatmap_cache_problem(path, records[:5], layout,
                                                               checkpoint)  # fmt: skip
    assert "f" * 64 in policies.heatmap_cache_problem(path, records, layout, "f" * 64)
    other = LayoutPins(layout.sha256, "e" * 64)
    assert "e" * 64 in policies.heatmap_cache_problem(path, records, other, checkpoint)
    floor = policies.heatmap_cache_problem(path, records, layout, checkpoint, floor_px=2.0)
    assert "covariance floor" in floor


def _filter(tree: dict, caches: Path, layout: Path, out: Path, **kwargs) -> dict:
    return policies.filter_trajectories(
        "keypoint-ukf", tree["shirt"], tree["runs"] / "candidates", out,
        trajectories=st.TRAJECTORIES, layout=layout, caches_root=caches, **kwargs,
    )  # fmt: skip


def test_keypoint_ukf_refuses_a_cache_of_another_layout_or_checkpoint(tree, stubbed, tmp_path):
    caches = tmp_path / "caches"
    policies.cache_heatmap_measurements(
        tree["checkpoint"], tree["shirt"], tree["layout"], tree["candidates"], caches,
        trajectories=st.TRAJECTORIES, allow_unpinned_checkpoint=True,
    )  # fmt: skip
    trained = load_layout(tree["layout"])
    # Layout L2's points with heatmaps of an L1 model: refused, naming both points hashes.
    moved = tmp_path / "moved.json"
    moved.write_text(json.dumps({"keypoints_3d": (trained.points + 0.05).tolist()}))
    with pytest.raises(ValueError, match=f"{trained.points_sha256}.*"
                       f"{load_layout(moved).points_sha256}"):  # fmt: skip
        _filter(tree, caches, moved, tmp_path / "out")
    # Caches of another checkpoint than the candidates': refused, naming both.
    (tree["runs"] / "candidates" / "summary.json").write_text(
        json.dumps({"checkpoint_sha256": "d" * 64}))  # fmt: skip
    with pytest.raises(ValueError, match=f"{_sha256(tree['checkpoint'])}.*{'d' * 64}"):
        _filter(tree, caches, tree["layout"], tmp_path / "out")
    (tree["runs"] / "candidates" / "summary.json").unlink()
    # A cache without its identity (not written by cache_heatmap_measurements): refused.
    with pytest.raises(ValueError, match="records no"):
        _filter(tree, tree["caches"], tree["layout"], tmp_path / "out")
    # The same points in another file format: accepted and recorded as a deviation.
    summary = _filter(tree, caches, tree["layout"], tmp_path / "out")
    assert summary["layout_formatting_deviation"] is False
    assert summary["layout_points_sha256"] == trained.points_sha256
    reformatted = tmp_path / "reformatted.json"
    reformatted.write_text(json.dumps(json.loads(tree["layout"].read_text()), indent=2))
    summary = _filter(tree, caches, reformatted, tmp_path / "out", force=True)
    assert summary["layout_formatting_deviation"] is True
    assert summary["score_definition"] == "paper"


def test_filters_carry_the_candidates_layout_deviation(tree, tmp_path):
    recorded = {"checkpoint_sha256": "a" * 64, "layout_formatting_deviation": True}
    (tree["runs"] / "candidates" / "summary.json").write_text(json.dumps(recorded))
    summary = policies.filter_trajectories(
        "pnp-ukf", tree["shirt"], tree["runs"] / "candidates", tmp_path / "out",
        trajectories=st.TRAJECTORIES,
    )  # fmt: skip
    assert summary["layout_formatting_deviation"] is True
    assert summary["candidates_checkpoint_sha256"] == "a" * 64
    with pytest.raises(ValueError, match="unknown SHIRT trajectories"):
        policies.filter_trajectories(
            "pnp-ukf", tree["shirt"], tree["runs"] / "candidates", tmp_path / "other",
            trajectories=("roe3",),
        )  # fmt: skip


@pytest.mark.parametrize(
    ("method", "extra", "message"),
    [
        ("pnp-ukf", ("--checkpoint", "c.pth"), "takes no --checkpoint"),
        ("pnp-ukf-rts", ("--keypoints-3d", "k.json"), "takes no --checkpoint or --keypoints-3d"),
        ("pnp-ukf", ("--limit", "-1"), "--limit"),
        ("pnp-ukf", ("--trajectories", "roe1,roe9"), "roe9"),
        ("framewise", ("--checkpoint", "c.pth", "--keypoints-3d", "k.json", "--limit", "-2"),
         "--limit"),
        ("candidates", ("--checkpoint", "c.pth", "--keypoints-3d", "k.json", "--trajectories",
                        "roe1,roe1"), "repeat"),
    ],
)  # fmt: skip
def test_cli_shirt_plan_refuses_bad_requests(tree, capsys, method, extra, message):
    assert cli.main([*_cli(tree, method, *extra), "--dry-run"]) == 1
    assert message in capsys.readouterr().out


def test_cli_filters_need_bayesfilter_at_plan_time(tree, capsys, monkeypatch):
    import importlib.util

    find_spec = importlib.util.find_spec

    def without_bayesfilter(name, *args):
        return None if name == "bayesfilter" else find_spec(name, *args)

    monkeypatch.setattr(importlib.util, "find_spec", without_bayesfilter)
    for method in ("pnp-ukf", "keypoint-ukf"):
        extra = () if method == "pnp-ukf" else ("--checkpoint", "c.pth", "--keypoints-3d", "k")
        assert cli.main([*_cli(tree, method, *extra), "--dry-run"]) == 1
        assert "doper[filter]" in capsys.readouterr().out


def test_filter_dry_run_imports_no_torch_or_bayesfilter(tmp_path):
    code = (
        "import sys; from doper import cli; "
        "rc = cli.main(['evaluate', 'shirt', '--method', 'keypoint-ukf', "
        f"'--root', {str(tmp_path)!r}, '--runs', {str(tmp_path / 'runs')!r}, "
        "'--checkpoint', 'c.pth', '--keypoints-3d', 'k.json', '--dry-run']); "
        "print(rc, 'torch' in sys.modules, 'bayesfilter' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "0 False False"


# --- exact replay of Table 2 (opt-in) ----------------------------------------------------------


def _significant_match(value: float, written: str) -> bool:
    """Whether ``value`` rounds to ``written`` at the decimals written."""
    decimals = len(written.split(".")[1]) if "." in written else 0
    return round(value, decimals) == float(written)


def _verify_artifacts(root: Path) -> None:
    """SHA-256-check every replay artifact against ``golden/shirt_replay_pins.json``.

    A null pin is a slot still waiting for its value: the observed SHA-256 is reported (add it
    to the pins file; no code change), every non-null pin must match.
    """
    pins = json.loads(REPLAY_PINS.read_text())["artifacts"]
    for name, pin in pins.items():
        observed = _sha256(root / name)
        if pin is None:
            warnings.warn(f"no SHA-256 pin for {name}; observed {observed}", stacklevel=2)
        else:
            assert observed == pin, f"{name}: SHA-256 {observed} is not the pinned {pin}"


def test_exact_replay_when_artifacts_supplied():
    """Table 2 from the original candidate records, heatmap cache and SHIRT metadata.

    ``DOPER_SHIRT_ARTIFACTS`` is a directory holding ``camera.json`` and ``keypoints_3d.json``
    and, per trajectory, ``<roe>/records.jsonl`` (the research candidate records,
    ``aux_flip_gate_v1/<roe>/lightbox``), ``<roe>/heatmap_measurements.npz``
    (``direct_keypoint_ukf_p99_v1/<roe>/lightbox``) and ``<roe>/metadata.json`` (SHIRT).
    """
    root = os.environ.get(REPLAY_ENV)
    if not root:
        pytest.skip(f"set {REPLAY_ENV} to replay Table 2 from the original artifacts")
    root_path = Path(root)
    _verify_artifacts(root_path)
    camera = data.load_camera(root_path)
    points_3d = np.asarray(json.loads((root_path / "keypoints_3d.json").read_text())
                           ["keypoints_3d"], dtype=np.float64)  # fmt: skip
    for roe in st.TRAJECTORIES:
        records = policies.load_candidate_records(root_path, roe, ".")
        metadata = data.load_metadata(root_path / roe)
        cache = policies.load_heatmap_cache(root_path / roe / policies.CACHE_NAME)
        causal = policies.nis_rank(records, metadata)
        direct = policies.keypoint_ukf(records, cache, metadata, points_3d, camera)
        rows, summary = policies.keypoint_ukf_outputs(direct, causal, cache)
        for method, key in (("pnp-ukf", "baseline_minimum_nis_pnp"),
                            ("keypoint-ukf", "direct_keypoint_ukf")):  # fmt: skip
            rotation, translation = REPLAY_EXACT[method][roe]
            assert abs(summary[key]["rotation_deg_mean"] - rotation) <= REPLAY_TOLERANCE
            assert abs(summary[key]["translation_m_mean"] - translation) <= REPLAY_TOLERANCE
        assert summary["baseline_prediction_only_frames"] == REPLAY_PREDICTION_ONLY[roe]
        _, rts_summary = policies.rts_outputs(policies.rts(causal))
        framewise = policies.summarize_pose_values([
            policies.pose_metrics(r["candidates"][0]["rvec"], r["candidates"][0]["tvec"],
                                  r["ground_truth_quaternion"], r["ground_truth_translation"])
            for r in records
        ])  # fmt: skip
        observed = {
            "pnp-ukf-rts": (rts_summary["rts"]["rotation_deg_mean_std"][0],
                            rts_summary["rts"]["translation_m_mean_std"][0]),
            "framewise": (framewise["rotation_deg_mean"], framewise["translation_m_mean"]),
        }  # fmt: skip
        for method, values in observed.items():
            for value, written in zip(values, REPLAY_ROUNDED[method][roe], strict=True):
                assert _significant_match(value, written), (method, roe, value, written)


def test_replay_pins_name_every_artifact():
    pins = json.loads(REPLAY_PINS.read_text())["artifacts"]
    expected = {"camera.json", "keypoints_3d.json"} | {
        f"{roe}/{name}" for roe in st.TRAJECTORIES
        for name in ("records.jsonl", "heatmap_measurements.npz", "metadata.json")
    }  # fmt: skip
    assert set(pins) == expected
    assert pins["keypoints_3d.json"] == (
        "a6697cafc62bdf5286fb8af453e44bae881b4ad481b8928edf288c85f77fbfa6"
    )
    assert _significant_match(0.128604, "0.12860") and not _significant_match(0.1287, "0.12860")


# --- The paper's bayesfilter pin (opt-in) -------------------------------------------

PIN_SCRIPT = """
import json, sys
from importlib import metadata
import numpy as np
sys.path.insert(0, {tests!r})
import shirt_tracks as st
from doper.shirt import policies
from doper.speedplus.geometry import Camera
arrays = {{}}
for roe in st.TRAJECTORIES:
    meta = st.make_metadata(st.SLOTS[roe])
    causal = policies.nis_rank(st.candidate_records(roe), meta)
    camera = Camera(st.CAMERA_MATRIX, st.DISTORTION, st.CAMERA_SIZE, st.CAMERA_SIZE)
    direct = policies.keypoint_ukf(st.candidate_records(roe), st.heatmap_cache(roe), meta,
                                   st.make_points_3d(), camera)
    for name, run in (("causal", causal), ("rts", policies.rts(causal)), ("direct", direct)):
        arrays[f"{{roe}}/{{name}}"] = np.stack([np.concatenate([np.ravel(r), np.ravel(t)])
                                                for r, t in run.poses])
    arrays[f"{{roe}}/nis"] = np.array([np.nan if v is None else v for f in causal.frames[1:]
                                       for v in f["candidate_nis"]])
np.savez(sys.argv[1], **arrays)
origin = metadata.distribution("bayesfilter").read_text("direct_url.json")
print(json.dumps({{"version": metadata.version("bayesfilter"), "direct_url": origin}}))
"""


def _pin_arrays(python: str, out: Path) -> tuple[dict, dict]:
    script = PIN_SCRIPT.format(tests=str(Path(__file__).parent))
    result = subprocess.run([python, "-c", script, str(out)], capture_output=True, text=True,
                            check=True)  # fmt: skip
    with np.load(out) as arrays:
        return dict(arrays), json.loads(result.stdout.strip().splitlines()[-1])


def test_bayesfilter_pin_equivalence(tmp_path):
    """The filters under bayesfilter @ d895dca (the paper's pin) equal 0.2.0 within 1e-12."""
    python = os.environ.get(PIN_PYTHON_ENV)
    if not python:
        pytest.skip(f"set {PIN_PYTHON_ENV} to a Python with bayesfilter @ git {PIN_COMMIT}")
    pinned, pinned_info = _pin_arrays(python, tmp_path / "pinned.npz")
    current, current_info = _pin_arrays(sys.executable, tmp_path / "current.npz")
    assert PIN_COMMIT in (pinned_info["direct_url"] or ""), pinned_info
    assert current_info["version"] == "0.2.0"
    assert set(pinned) == set(current)
    for key in pinned:
        np.testing.assert_allclose(pinned[key], current[key], rtol=0, atol=PIN_TOLERANCE,
                                   err_msg=key)  # fmt: skip
