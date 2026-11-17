# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Relative camera-pose evaluation of a capture bundle (``doper.evaluation.relative_pose``)."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from nurec_fixtures import (
    HANDEYE_CAMERA,
    NUREC_TEMPLATE,
    nurec_recipe_data,
    own_nurec_recipe,
    rigid,
    write_bundle,
)

from doper import cli
from doper.evaluation import relative_pose as rp
from doper.infer import Detection
from doper.nurec import capture

ROT = rp.ROTATION_KEY
TRANS = rp.TRANSLATION_KEY


def _frame(name: str, pose: np.ndarray | None, *, excluded: bool = False) -> rp.FramePose:
    return rp.FramePose(
        image=name,
        sha256="0" * 64,
        detections=int(pose is not None),
        T_camera_object=pose,
        excluded=excluded,
    )


def _static_scene(count: int, seed: int = 0) -> tuple[list[rp.FramePose], dict[str, np.ndarray]]:
    """Frames of a static object seen from ``count`` robot cameras (exact poses)."""
    base_object = rigid(seed + 500)
    names = [f"{i:06d}.jpg" for i in range(count)]
    cameras = {name: rigid(seed + i) for i, name in enumerate(names)}
    frames = [_frame(n, np.linalg.inv(cameras[n]) @ base_object) for n in names]
    return frames, cameras


def _delta(angle_deg: float, translation_m: tuple[float, float, float]) -> np.ndarray:
    delta = np.eye(4)
    delta[:3, :3] = cv2.Rodrigues(np.array([0.0, 0.0, np.radians(angle_deg)]))[0]
    delta[:3, 3] = translation_m
    return delta


def test_static_object_identity():
    frames, cameras = _static_scene(5)
    pairs = rp.relative_pose_pairs(frames, cameras)
    assert [(p.i, p.j) for p in pairs] == list(itertools.combinations(range(5), 2))
    for pair in pairs:
        assert pair.rotation_error_deg < 1e-5
        assert pair.translation_error_mm < 1e-9
        assert pair.baseline_mm > 0


def test_known_perturbation_errors():
    frames, cameras = _static_scene(3, seed=7)
    perturbed = _delta(5.0, (0.003, 0.0, 0.0)) @ frames[2].T_camera_object
    frames[2] = _frame(frames[2].image, perturbed)
    by_pair = {(p.i, p.j): p for p in rp.relative_pose_pairs(frames, cameras)}
    assert by_pair[(0, 1)].rotation_error_deg < 1e-5
    for key in ((0, 2), (1, 2)):
        assert by_pair[key].rotation_error_deg == pytest.approx(5.0, abs=1e-9)
        assert by_pair[key].translation_error_mm == pytest.approx(3.0, abs=1e-9)
    reference = np.linalg.inv(cameras[frames[0].image]) @ cameras[frames[2].image]
    assert by_pair[(0, 2)].baseline_mm == pytest.approx(1000 * np.linalg.norm(reference[:3, 3]))


def test_none_poses_skipped():
    frames, cameras = _static_scene(5, seed=3)
    frames[1] = _frame(frames[1].image, None)
    frames[3] = _frame(frames[3].image, None, excluded=True)
    pairs = rp.relative_pose_pairs(frames, cameras)
    assert [(p.i, p.j) for p in pairs] == [(0, 2), (0, 4), (2, 4)]
    summary = rp.summarize_pairs(pairs, frames)
    assert summary["frames"] == 4  # the excluded capture is not evaluated
    assert summary["pose_successes"] == 3
    assert summary["total_all_pairs"] == 6
    assert summary["all_pairs"]["count"] == 3


def test_pose_rejection_rules():
    good = np.eye(4)
    good[2, 3] = 0.5
    behind = good.copy()
    behind[2, 3] = 0.0
    nan = good.copy()
    nan[0, 3] = np.nan
    assert rp.pose_from_detection(_detection(good)) is not None
    assert rp.pose_from_detection(_detection(behind)) is None
    assert rp.pose_from_detection(_detection(nan)) is None
    assert rp.pose_from_detection(_detection(None)) is None
    assert rp.pose_from_detection(None) is None


def test_adjacent_pairs_follow_capture_order():
    frames, cameras = _static_scene(5, seed=11)
    frames[2] = _frame(frames[2].image, _delta(2.0, (0, 0, 0)) @ frames[2].T_camera_object)
    frames[3] = _frame(frames[3].image, None)
    pairs = rp.relative_pose_pairs(frames, cameras)
    summary = rp.summarize_pairs(pairs, frames)
    adjacent = [(p.i, p.j) for p in pairs if p.j == p.i + 1]
    assert adjacent == [(0, 1), (1, 2)]
    assert summary["adjacent_pairs"]["count"] == 2
    assert summary["total_adjacent_pairs"] == 4
    rotations = [p.rotation_error_deg for p in pairs if p.j == p.i + 1]
    assert summary["adjacent_pairs"][ROT]["median"] == pytest.approx(float(np.median(rotations)))
    assert summary["adjacent_pairs"][ROT]["p95"] == pytest.approx(
        float(np.percentile(rotations, 95))
    )
    every = [p.rotation_error_deg for p in pairs]
    assert summary["all_pairs"][ROT]["median"] == float(np.median(every))


def test_summary_without_pairs_has_only_counts():
    frames = [_frame("a.jpg", None), _frame("b.jpg", None)]
    summary = rp.summarize_pairs([], frames)
    assert summary["all_pairs"] == {"count": 0}
    assert summary["adjacent_pairs"] == {"count": 0}


def _write_run(
    out: Path, frames: list[rp.FramePose], cameras: dict, sha: str = "c" * 64, **summary
) -> None:
    pairs = rp.relative_pose_pairs(frames, cameras)
    record = {"checkpoint_sha256": sha, "bundle_sha256": "d" * 64, "score_thr": 0.3, "max_det": 1,
              "capture_flags": {"handeye_camera_mismatch": False}, **summary}  # fmt: skip
    rp.write_run(out, frames, cameras, pairs, record)


def test_compare_uses_common_pairs(tmp_path):
    frames, cameras = _static_scene(5, seed=5)
    baseline = list(frames)
    baseline[0] = _frame(frames[0].image, _delta(3.0, (0.002, 0, 0)) @ frames[0].T_camera_object)
    baseline[1] = _frame(frames[1].image, None)
    adapted = list(frames)
    adapted[0] = _frame(frames[0].image, _delta(1.0, (0.001, 0, 0)) @ frames[0].T_camera_object)
    adapted[4] = _frame(frames[4].image, None)
    _write_run(tmp_path / "b", baseline, cameras, "b" * 64)
    _write_run(tmp_path / "a", adapted, cameras, "a" * 64)
    result = rp.compare_runs(tmp_path / "b", tmp_path / "a")
    common = [(0, 2), (0, 3), (2, 3)]
    assert result["all_pairs"]["common_valid_pairs"] == len(common)
    assert result["adjacent_pairs"]["common_valid_pairs"] == 1
    base = {(p.i, p.j): p for p in rp.relative_pose_pairs(baseline, cameras)}
    adapt = {(p.i, p.j): p for p in rp.relative_pose_pairs(adapted, cameras)}
    x = np.array([base[k].rotation_error_deg for k in common])
    y = np.array([adapt[k].rotation_error_deg for k in common])
    rotation = result["all_pairs"][ROT]
    assert rotation["baseline_median"] == float(np.median(x))
    assert rotation["finetune_median"] == float(np.median(y))
    assert rotation["baseline_p95"] == float(np.percentile(x, 95))
    assert rotation["median_paired_change"] == float(np.median(y - x))
    assert result["all_pairs"][TRANS]["median_paired_change"] < 0
    assert result["baseline"]["failed_images"] == [frames[1].image]
    assert result["finetune"]["failed_images"] == [frames[4].image]
    assert result["baseline"]["pose_successes"] == 4
    assert result["baseline"]["checkpoint_sha256"] == "b" * 64
    assert result["scope"] == rp.COMPARISON_SCOPE


def test_compare_refuses_runs_over_other_images(tmp_path):
    frames, cameras = _static_scene(3, seed=1)
    _write_run(tmp_path / "b", frames, cameras)
    other = [_frame(f"x{f.image}", f.T_camera_object) for f in frames]
    _write_run(tmp_path / "a", other, {f"x{k}": v for k, v in cameras.items()})
    with pytest.raises(ValueError, match="same images"):
        rp.compare_runs(tmp_path / "b", tmp_path / "a")


def test_summary_records_caveats_and_exclusions():
    frames, cameras = _static_scene(4, seed=2)
    frames[3] = _frame(frames[3].image, None, excluded=True)
    pairs = rp.relative_pose_pairs(frames, cameras)
    flags = {"handeye_camera_mismatch": True, "capture_camera": {}, "handeye_camera": {}}
    summary = rp.build_summary(
        pairs,
        frames,
        model="best.pth",
        settings={"backbone": "b"},
        checkpoint_sha256="1" * 64,
        keypoints_sha256="2" * 64,
        capture_flags=flags,
        exclude=[frames[3].image],
        score_thr=0.3,
        bundle_sha256="3" * 64,
    )
    assert summary["caveats"] == [
        rp.CAVEAT_SOURCE_CAPTURE,
        rp.CAVEAT_NO_FITTING,
        rp.HANDEYE_CAVEATS[True],
        rp.CAVEAT_CORRELATED_PAIRS,
    ]
    assert rp.HANDEYE_CAVEATS[True] == (
        "Hand-eye calibration recorded at a different camera mode; transfer unvalidated"
    )
    assert summary["exclude"] == [frames[3].image]
    assert summary["capture_flags"] is flags
    assert summary["checkpoint_sha256"] == "1" * 64
    assert summary["keypoints_sha256"] == "2" * 64
    assert summary["max_det"] == 1 and summary["score_thr"] == 0.3
    assert (summary["bundle_sha256"], summary["bundle_files_sha256"]) == ("3" * 64, None)
    assert {
        "model",
        "frames",
        "pose_successes",
        "all_pairs",
        "adjacent_pairs",
        "total_all_pairs",
        "total_adjacent_pairs",
        "settings",
    } <= set(summary)
    for mismatch in (False, None):
        caveats = rp.caveats({"handeye_camera_mismatch": mismatch})
        assert caveats[2] == rp.HANDEYE_CAVEATS[mismatch]


def _detection(pose: np.ndarray | None) -> Detection:
    det = Detection(
        bbox_xyxy=np.array([1.0, 2.0, 30.0, 40.0]),
        score=0.9,
        keypoints=np.zeros((4, 2)),
        keypoint_scores=np.ones(4),
    )
    if pose is not None:
        det.rvec = cv2.Rodrigues(pose[:3, :3])[0] if np.isfinite(pose).all() else np.full(3, np.nan)
        det.tvec = pose[:3, 3].reshape(3, 1)
        det.reproj_error_px = 0.5
    return det


class _FakeInference:
    """Stands in for ``doper.infer.infer_image``: a pose per image, keyed by the image's pixels."""

    def __init__(self, poses: dict[int, np.ndarray | None]) -> None:
        self.poses = poses
        self.calls: list[dict] = []

    def __call__(self, model, image, cfg, device, **kwargs):  # noqa: ANN001, ANN204
        self.calls.append(kwargs)
        key = int(image[0, 0, 0])
        if key not in self.poses:
            return []
        return [_detection(self.poses[key])]


def _keyed_bundle(root: Path, **kwargs) -> dict:  # noqa: ANN003
    record = write_bundle(root, **kwargs)
    for i, name in enumerate(
        json.loads(line)["name"] for line in (root / "images.jsonl").read_text().splitlines()
    ):
        path = root / "images" / name
        pixels = cv2.imread(str(path))
        pixels[0, 0] = 10 * (i + 1)
        params = [cv2.IMWRITE_JPEG_QUALITY, 100] if name.endswith(".jpg") else []
        assert cv2.imwrite(str(path), pixels, params)
    rows = []
    for line in (root / "images.jsonl").read_text().splitlines():
        row = json.loads(line)
        row["sha256"] = capture.sha256_file(root / "images" / row["name"])
        rows.append(json.dumps(row) + "\n")
    (root / "images.jsonl").write_text("".join(rows))
    return record


def _pose(z: float) -> np.ndarray:
    pose = rigid(int(z * 100))
    pose[2, 3] = z
    return pose


def test_estimate_frames_keeps_capture_order_and_rejects(tmp_path, monkeypatch):
    import doper.infer

    root = tmp_path / "bundle"
    _keyed_bundle(root, names=("a.png", "b.png", "c.png", "d.png"), exclude=["d.png"])
    image_set = capture.load_image_set(root)
    keys = {}
    for i, name in enumerate(image_set.names):
        keys[name] = int(cv2.imread(str(image_set.image_path(name)))[0, 0, 0])
        assert keys[name] == 10 * (i + 1)
    behind = _pose(0.5)
    behind[2, 3] = -0.5
    fake = _FakeInference({keys["a.png"]: _pose(0.4), keys["b.png"]: behind})
    monkeypatch.setattr(doper.infer, "infer_image", fake)
    frames = rp.estimate_frames(object(), image_set, np.zeros((4, 3)), cfg=object())
    assert [f.image for f in frames] == ["a.png", "b.png", "c.png", "d.png"]
    assert [f.pose_success for f in frames] == [True, False, False, False]
    assert [f.detections for f in frames] == [1, 1, 0, 0]
    assert frames[3].excluded and not frames[2].excluded
    assert len(fake.calls) == 3  # the excluded capture is not inferred
    for kwargs in fake.calls:
        assert kwargs["max_det"] == 1 and kwargs["score_thr"] == 0.3
        assert np.array_equal(kwargs["dist_coeffs"], np.zeros(5))
        assert np.array_equal(kwargs["camera_matrix"], image_set.camera.matrix)
    np.testing.assert_allclose(frames[0].T_camera_object, _pose(0.4), atol=1e-12)
    record = frames[0].as_record(np.eye(4))
    assert record["pose_success"] and record["reprojection_error_px"] == 0.5
    assert rp.FramePose.from_record(record).image == "a.png"


def test_estimate_frames_refuses_changed_images(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    _keyed_bundle(root)
    image_set = capture.load_image_set(root)
    (image_set.image_path(image_set.names[0])).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        rp.estimate_frames(object(), image_set, np.zeros((4, 3)), cfg=object())


# --- CLI ------------------------------------------------------------------------------------


POINTS = [[0.01 * i, 0.0, 0.002 * i] for i in range(64)]


def _bundle_with_keypoints(root: Path, **kwargs) -> dict:  # noqa: ANN003
    """A keyed bundle with a 64-point layout (``recipe_file`` pins its points hash)."""
    record = _keyed_bundle(root, **kwargs)
    (root / "keypoints_3d.json").write_text(json.dumps({"keypoints_3d": POINTS}))
    return record


@pytest.fixture
def recipe_file(tmp_path: Path) -> str:
    import hashlib

    import yaml

    data = nurec_recipe_data()
    digest = hashlib.sha256(np.asarray(POINTS, dtype="<f8").tobytes()).hexdigest()
    data["object"]["keypoints_points_sha256"] = digest
    path = tmp_path / "nurec.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


def _evaluate_argv(tmp_path: Path, recipe: str, *extra: str) -> list[str]:
    return [
        "evaluate",
        recipe,
        "--root",
        str(tmp_path / "data"),
        "--runs",
        str(tmp_path / "runs"),
        "--bundle",
        str(tmp_path / "bundle"),
        "--checkpoint",
        str(tmp_path / "best.pth"),
        "--out",
        str(tmp_path / "eval"),
        *extra,
    ]


def test_evaluate_nurec_dry_run_touches_nothing(tmp_path, capsys):
    assert cli.main([*_evaluate_argv(tmp_path, str(NUREC_TEMPLATE)), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "relative camera poses" in out and "summary.json" in out
    assert not (tmp_path / "eval").exists()


@pytest.mark.parametrize("missing", ["--bundle", "--checkpoint", "--out"])
def test_evaluate_nurec_needs_its_inputs(tmp_path, capsys, missing):
    argv = _evaluate_argv(tmp_path, str(NUREC_TEMPLATE), "--dry-run")
    index = argv.index(missing)
    del argv[index : index + 2]
    assert cli.main(argv) == 1
    assert missing in capsys.readouterr().out


@pytest.mark.parametrize(
    ("extra", "flag"),
    [
        (["--split", "val"], "--split"),
        (["--method", "framewise"], "--method"),
        (["--keypoints-3d", "k.json"], "--keypoints-3d"),
        (["--limit", "3"], "--limit"),
    ],
)
def test_evaluate_nurec_refuses_other_kinds_flags(tmp_path, capsys, extra, flag):
    assert cli.main([*_evaluate_argv(tmp_path, str(NUREC_TEMPLATE), *extra), "--dry-run"]) == 1
    assert flag in capsys.readouterr().out


def test_bundle_and_out_refused_for_other_recipes(tmp_path, capsys):
    argv = ["evaluate", "tless", "--split", "val", "--root", "R", "--runs", "X", "--dry-run"]
    assert cli.main([*argv, "--bundle", "B"]) == 1
    assert "--bundle" in capsys.readouterr().out


def _fake_model(monkeypatch, poses_by_key: dict[int, np.ndarray | None]) -> None:
    import doper.infer

    monkeypatch.setattr(doper.infer, "infer_image", _FakeInference(poses_by_key))
    settings = {"backbone": "fixture"}
    monkeypatch.setattr(
        rp, "load_evaluation_model", lambda checkpoint, device: (object(), object(), settings)
    )


def _object_poses(record: dict, names: list[str]) -> dict[int, np.ndarray]:
    """Exact object poses of a static object for the bundle's robot cameras, keyed by pixel."""
    base_object = rigid(321)
    base_object[:3, 3] += [0.0, 0.0, 3.0]
    poses = {}
    for i, name in enumerate(names):
        camera = record["poses"][name] @ record["T_mount_camera"]
        poses[10 * (i + 1)] = np.linalg.inv(camera) @ base_object
    return poses


def test_evaluate_nurec_writes_frames_pairs_summary(tmp_path, monkeypatch, recipe_file):
    names = ["000000.png", "000001.png", "000002.png", "000003.png"]
    record = _bundle_with_keypoints(tmp_path / "bundle", names=names, handeye_camera=HANDEYE_CAMERA)
    poses = {k: v for k, v in _object_poses(record, names).items() if v[2, 3] > 0}
    assert len(poses) >= 2
    _fake_model(monkeypatch, poses)
    (tmp_path / "best.pth").write_bytes(b"checkpoint")
    assert cli.main(_evaluate_argv(tmp_path, recipe_file, "--device", "cpu")) == 0
    out = tmp_path / "eval"
    frames = json.loads((out / "frames.json").read_text())
    pairs = json.loads((out / "pairs.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    assert [f["image"] for f in frames] == names
    assert summary["pose_successes"] == len(poses)
    assert summary["all_pairs"]["count"] == len(pairs) == len(poses) * (len(poses) - 1) // 2
    assert all(p[ROT] < 1e-4 and p[TRANS] < 1e-6 for p in pairs)
    assert summary["capture_flags"]["handeye_camera_mismatch"] is True
    assert summary["caveats"][2] == rp.HANDEYE_CAVEATS[True]
    assert summary["checkpoint_sha256"] == capture.sha256_file(tmp_path / "best.pth")
    assert summary["keypoints_sha256"] == capture.sha256_file(
        tmp_path / "bundle" / "keypoints_3d.json"
    )
    assert summary["settings"] == {"backbone": "fixture"}
    assert summary["exclude"] == []
    assert summary["bundle_sha256"] == capture.bundle_sha256(tmp_path / "bundle")
    assert summary["bundle_files_sha256"] == capture.bundle_files_sha256(tmp_path / "bundle")
    # A rerun is refused without --force, and --force replaces the outputs.
    assert cli.main(_evaluate_argv(tmp_path, recipe_file, "--device", "cpu")) == 1
    assert cli.main(_evaluate_argv(tmp_path, recipe_file, "--device", "cpu", "--force")) == 0


def test_evaluate_nurec_refuses_bundle_without_robot_reference(
    tmp_path, monkeypatch, capsys, recipe_file
):
    _bundle_with_keypoints(tmp_path / "bundle", robot=False)
    _fake_model(monkeypatch, {})
    (tmp_path / "best.pth").write_bytes(b"checkpoint")
    assert cli.main(_evaluate_argv(tmp_path, recipe_file, "--device", "cpu")) == 1
    assert "robot reference" in capsys.readouterr().out
    assert not (tmp_path / "eval").exists()


def test_evaluate_nurec_refuses_another_keypoint_layout(tmp_path, monkeypatch, capsys):
    _bundle_with_keypoints(tmp_path / "bundle")
    _fake_model(monkeypatch, {})
    (tmp_path / "best.pth").write_bytes(b"checkpoint")
    recipe = own_nurec_recipe(tmp_path)
    assert cli.main(_evaluate_argv(tmp_path, recipe, "--device", "cpu")) == 1
    assert "keypoint layout" in capsys.readouterr().out


def test_evaluate_nurec_records_recipe_and_bundle_exclusions(tmp_path, monkeypatch, recipe_file):
    import yaml

    data = yaml.safe_load(Path(recipe_file).read_text())
    data["evaluation"]["exclude"] = ["000001.png"]
    Path(recipe_file).write_text(yaml.safe_dump(data))
    names = ["000000.png", "000001.png", "000002.png"]
    record = _bundle_with_keypoints(tmp_path / "bundle", names=names, exclude=["000002.png"])
    _fake_model(monkeypatch, _object_poses(record, names))
    (tmp_path / "best.pth").write_bytes(b"checkpoint")
    assert cli.main(_evaluate_argv(tmp_path, recipe_file, "--device", "cpu")) == 0
    summary = json.loads((tmp_path / "eval" / "summary.json").read_text())
    assert summary["exclude"] == ["000001.png", "000002.png"]
    assert summary["frames"] == 1 and summary["total_all_pairs"] == 0


def test_report_nurec_writes_paired_comparison(tmp_path, capsys):
    frames, cameras = _static_scene(4, seed=9)
    _write_run(tmp_path / "b", frames, cameras)
    _write_run(tmp_path / "a", frames, cameras)
    argv = [
        "report",
        str(NUREC_TEMPLATE),
        "--runs",
        str(tmp_path / "runs"),
        "--baseline",
        str(tmp_path / "b"),
        "--adapted",
        str(tmp_path / "a"),
    ]
    assert cli.main([*argv, "--dry-run"]) == 0
    assert "paired_comparison.json" in capsys.readouterr().out
    assert not (tmp_path / "runs").exists()
    assert cli.main(argv) == 0
    result = json.loads((tmp_path / "runs" / "paired_comparison.json").read_text())
    assert result["all_pairs"]["common_valid_pairs"] == 6
    assert result["all_pairs"][ROT]["median_paired_change"] == 0.0
    assert cli.main(argv) == 1  # an existing comparison is replaced only with --force
    assert cli.main([*argv, "--force"]) == 0


@pytest.mark.parametrize(
    "extra", [["--ap"], ["--timing"], ["--mask-audit"], ["--policy", "base"], ["--root", "R"]]
)
def test_report_nurec_refuses_bop_flags_and_needs_runs(tmp_path, capsys, extra):
    argv = ["report", str(NUREC_TEMPLATE), "--runs", "X", "--baseline", "B", "--adapted", "A"]
    assert cli.main([*argv, *extra, "--dry-run"]) == 1
    assert extra[0] in capsys.readouterr().out


def test_report_nurec_needs_both_runs(capsys):
    assert (
        cli.main(["report", str(NUREC_TEMPLATE), "--runs", "X", "--baseline", "B", "--dry-run"])
        == 1
    )
    assert "--adapted" in capsys.readouterr().out


def test_report_bop_refuses_nurec_flags(capsys):
    assert cli.main(["report", "tless", "--runs", "X", "--baseline", "B", "--dry-run"]) == 1
    assert "--baseline" in capsys.readouterr().out


def test_compare_lists_exclusions_and_handles_no_common_pairs(tmp_path):
    frames, cameras = _static_scene(3, seed=4)
    frames[2] = _frame(frames[2].image, None, excluded=True)
    baseline = [frames[0], _frame(frames[1].image, None), frames[2]]
    _write_run(tmp_path / "b", baseline, cameras)
    _write_run(tmp_path / "a", frames, cameras)
    result = rp.compare_runs(tmp_path / "b", tmp_path / "a")
    assert result["all_pairs"] == {"common_valid_pairs": 0, ROT: None, TRANS: None}
    assert result["baseline"]["failed_images"] == [frames[1].image]
    assert result["baseline"]["excluded_images"] == [frames[2].image]


def test_evaluate_refuses_unknown_recipe_exclusions(tmp_path, monkeypatch, capsys, recipe_file):
    import yaml

    data = yaml.safe_load(Path(recipe_file).read_text())
    data["evaluation"]["exclude"] = ["missing.png"]
    Path(recipe_file).write_text(yaml.safe_dump(data))
    _bundle_with_keypoints(tmp_path / "bundle")
    _fake_model(monkeypatch, {})
    assert cli.main(_evaluate_argv(tmp_path, recipe_file, "--device", "cpu")) == 1
    assert "missing.png" in capsys.readouterr().out


def test_compare_refuses_runs_of_another_bundle_or_settings(tmp_path):
    frames, cameras = _static_scene(3, seed=6)
    _write_run(tmp_path / "b", frames, cameras)
    changes = (
        ({"bundle_sha256": "e" * 64}, "bundle_sha256"),
        ({"score_thr": 0.5}, "score_thr"),
        ({"capture_flags": {"handeye_camera_mismatch": True}}, "capture_flags"),
    )
    for change, key in changes:
        _write_run(tmp_path / key, frames, cameras, **change)
        with pytest.raises(ValueError, match=key):
            rp.compare_runs(tmp_path / "b", tmp_path / key)
    # A summary without the bundle identity (an older run) is refused, not compared.
    _write_run(tmp_path / "old", frames, cameras, bundle_sha256=None)
    with pytest.raises(ValueError, match="no capture bundle identity"):
        rp.compare_runs(tmp_path / "old", tmp_path / "b")
    (tmp_path / "old" / "summary.json").unlink()
    with pytest.raises(FileNotFoundError, match="summary.json"):
        rp.compare_runs(tmp_path / "b", tmp_path / "old")
