# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DOPER framewise evaluation on SHIRT (``doper.shirt.candidates.framewise``).

``test_framewise_matches_source`` compares the port with ``golden/source_shirt_framewise.json``,
what the research framewise evaluation (research source snapshot) writes for the
``shirt_frames`` tree with the
same stub model (see that module). The parity case ``shirt_framewise`` runs both sides in one
environment.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import shirt_frames as sf
import torch
from golden_floats import approx_floats
from tiny_coco import tiny_model_config

from doper import cli, infer
from doper.recipes import load_any_recipe
from doper.shirt import candidates
from doper.speedplus import definitions
from doper.speedplus.checkpoints import LAYOUT_POINTS_SHA_KEY, LAYOUT_SHA_KEY, load_layout

GOLDEN = Path(__file__).parent / "golden" / "source_shirt_framewise.json"
GEN7_CHECKPOINT_SHA256 = "0824734e2d596adc0a5c45d8de034b67a11c2669926919e7105916974e4c4bba"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint(path: Path, layout_sha: str | None) -> Path:
    payload: dict = {"model": {"layer.weight": torch.ones(1)}}
    if layout_sha is not None:
        payload[LAYOUT_SHA_KEY] = layout_sha
    torch.save(payload, path)
    return path


@pytest.fixture
def scene(tmp_path: Path, monkeypatch) -> dict:
    """The SHIRT tree, its layout, a checkpoint recording that layout and the stub model."""
    root = tmp_path / "shirt"
    loader = sf.make_stub_loader(sf.write_shirt_tree(root))
    monkeypatch.setattr(infer, "load_model", loader)
    monkeypatch.setattr(infer, "infer_image", sf.stub_infer_image)
    layout = sf.write_layout(tmp_path / "keypoints_3d.json")
    checkpoint = _checkpoint(tmp_path / "epoch_10.pth", _sha256(layout))
    return {"root": root, "layout": layout, "checkpoint": checkpoint, "model": loader.model,
            "out": tmp_path / "out"}  # fmt: skip


def _framewise(scene: dict, **kwargs) -> dict:
    kwargs.setdefault("allow_unpinned_checkpoint", True)
    return candidates.framewise(
        scene["checkpoint"], scene["root"], scene["layout"], scene["out"],
        trajectories=sf.TRAJECTORIES, **kwargs,
    )  # fmt: skip


def _source_golden_or_skip() -> dict:
    """The framewise golden without its versions; skip unless this env has the same OpenCV."""
    golden = json.loads(GOLDEN.read_text())
    recorded, current = golden.pop("versions"), sf.library_versions()
    if recorded != current:
        pytest.skip(f"the framewise golden was produced with {recorded}, this env has "
                    f"{current}; {sf.REGENERATE}")  # fmt: skip
    return approx_floats(golden)


def test_framewise_matches_source(scene):
    summary = _framewise(scene)
    assert sf.canonical_framewise(scene["out"]) == _source_golden_or_skip()
    # One top-1 detection at score 0.0: no CLAHE, no TTA and no box pad.
    assert scene["model"].calls == [{"score_thr": 0.0, "max_det": 1}] * 2 * sf.FRAMES
    written = json.loads((scene["out"] / "summary.json").read_text())
    assert written == summary
    assert summary["method"] == "framewise" and summary["domain"] == "lightbox"
    assert summary["score_definition"] == "paper" and summary["temporal_filtering"] is False
    assert summary["layout_sha256"] == _sha256(scene["layout"])
    assert summary["layout_points_sha256"] == load_layout(scene["layout"]).points_sha256
    assert summary["layout_check"] == "recorded"
    assert summary["layout_formatting_deviation"] is False
    assert summary["trajectories"]["roe1"]["lightbox"]["no_pose_count"] == 2
    assert str(scene["root"]) not in (scene["out"] / "summary.json").read_text()
    assert (
        str(scene["root"]) not in (scene["out"] / "roe1" / "lightbox" / "records.jsonl").read_text()
    )


def test_framewise_golden_skips_on_other_builds(monkeypatch):
    monkeypatch.setattr(sf, "library_versions", lambda: {"opencv": "0"})
    with pytest.raises(pytest.skip.Exception, match="research source snapshot"):
        _source_golden_or_skip()


def test_shirt_refuses_layout_mismatch(scene, tmp_path):
    layout_sha = _sha256(scene["layout"])
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"keypoints_3d": [[0.0, 0.0, 1.0]] * 8}))
    other_sha = _sha256(other)
    scene["checkpoint"] = _checkpoint(tmp_path / "other.pth", other_sha)
    with pytest.raises(ValueError, match=f"{other_sha}.*{layout_sha}"):
        _framewise(scene)
    # A source-format checkpoint records no layout: it needs the recipe registry.
    scene["checkpoint"] = _checkpoint(tmp_path / "legacy.pth", None)
    legacy_sha = _sha256(scene["checkpoint"])
    with pytest.raises(ValueError, match=f"legacy checkpoint {legacy_sha}.*{layout_sha}"):
        _framewise(scene)
    assert not scene["out"].exists() and scene["model"].calls == []
    recipe = dataclasses.replace(
        load_any_recipe("shirt"), legacy_checkpoints={legacy_sha: layout_sha}
    )
    assert _framewise(scene, recipe=recipe)["layout_check"] == "legacy"


def _reformatted(layout: Path, path: Path) -> Path:
    """The same points in another file format: same points hash, another file SHA-256."""
    path.write_text(json.dumps(json.loads(layout.read_text()), indent=2))
    return path


def test_layout_formatting_deviation_is_recorded(scene, tmp_path, monkeypatch):
    trained = load_layout(scene["layout"])
    reformatted = _reformatted(scene["layout"], tmp_path / "reformatted.json")
    assert _sha256(reformatted) != trained.sha256
    # A new-format checkpoint recording the points hash: checked on it, the file is a deviation.
    torch.save({"model": {"layer.weight": torch.ones(1)}, LAYOUT_SHA_KEY: trained.sha256,
                LAYOUT_POINTS_SHA_KEY: trained.points_sha256}, scene["checkpoint"])  # fmt: skip
    summary = _framewise({**scene, "layout": reformatted})
    assert summary["layout_check"] == "recorded" and summary["layout_formatting_deviation"]
    # A legacy checkpoint registered with a layout whose points hash is known: the same.
    legacy = _checkpoint(tmp_path / "legacy.pth", None)
    recipe = dataclasses.replace(
        load_any_recipe("shirt"), legacy_checkpoints={_sha256(legacy): trained.sha256}
    )
    monkeypatch.setitem(definitions.KNOWN_LAYOUT_POINTS, trained.sha256, trained.points_sha256)
    summary = _framewise(
        {**scene, "checkpoint": legacy, "layout": reformatted}, recipe=recipe, force=True
    )
    assert summary["layout_check"] == "legacy" and summary["layout_formatting_deviation"]
    # Other points are refused, naming both points hashes.
    moved = tmp_path / "moved.json"
    moved.write_text(json.dumps({"keypoints_3d": (trained.points + 0.1).tolist()}))
    with pytest.raises(ValueError, match=f"{trained.points_sha256}.*"
                       f"{load_layout(moved).points_sha256}"):  # fmt: skip
        _framewise({**scene, "checkpoint": legacy, "layout": moved}, recipe=recipe, force=True)


def test_unpinned_checkpoint_needs_flag(scene):
    checkpoint_sha = _sha256(scene["checkpoint"])
    recipe = load_any_recipe("shirt")
    assert recipe.checkpoint_sha256 == GEN7_CHECKPOINT_SHA256
    with pytest.raises(ValueError, match=f"{checkpoint_sha}.*{GEN7_CHECKPOINT_SHA256}.*"
                       "--allow-unpinned-checkpoint"):  # fmt: skip
        _framewise(scene, allow_unpinned_checkpoint=False)
    assert not scene["out"].exists()
    summary = _framewise(scene)
    assert summary["checkpoint_sha256"] == checkpoint_sha
    assert summary["checkpoint_pin"] == GEN7_CHECKPOINT_SHA256
    assert summary["checkpoint_pinned"] is False and summary["allow_unpinned_checkpoint"] is True
    pinned = dataclasses.replace(recipe, checkpoint_sha256=checkpoint_sha)
    summary = _framewise(scene, recipe=pinned, allow_unpinned_checkpoint=False, force=True)
    assert summary["checkpoint_pinned"] is True and summary["allow_unpinned_checkpoint"] is False


def test_framewise_refuses_existing_output_and_bad_requests(scene):
    _framewise(scene)
    with pytest.raises(FileExistsError, match="--force"):
        _framewise(scene)
    with pytest.raises(ValueError, match="domain"):
        _framewise(scene, domain="sunlamp", force=True)
    with pytest.raises(ValueError, match="trajector"):
        candidates.framewise(
            scene["checkpoint"], scene["root"], scene["layout"], scene["out"], trajectories=(),
            force=True, allow_unpinned_checkpoint=True,
        )  # fmt: skip
    limited = _framewise(scene, limit=2, force=True)
    assert limited["trajectories"]["roe1"]["lightbox"]["requested"] == 2


def test_framewise_runs_a_real_tiny_model(tmp_path):
    """No stub: the real loader and inference on a random tiny heatmap model, on CPU."""
    from doper.model import DoperNet

    root = tmp_path / "shirt"
    sf.write_shirt_tree(root)
    layout = sf.write_layout(tmp_path / "keypoints_3d.json")
    model_cfg = tiny_model_config(
        num_keypoints=8, det_input_size=64, neck_channels=8, head_channels=8, crop_size=32,
        keypoint_head="heatmap",
    )  # fmt: skip
    torch.manual_seed(0)
    state = DoperNet(model_cfg).state_dict()
    checkpoint = tmp_path / "tiny.pth"
    torch.save({"model": state, "cfg": {"model": dataclasses.asdict(model_cfg)},
                LAYOUT_SHA_KEY: _sha256(layout)}, checkpoint)  # fmt: skip
    summary = candidates.framewise(
        checkpoint, root, layout, tmp_path / "out", trajectories=("roe1",),
        allow_unpinned_checkpoint=True, limit=2,
    )  # fmt: skip
    records = (tmp_path / "out" / "roe1" / "lightbox" / "records.jsonl").read_text().splitlines()
    assert len(records) == 2 and summary["trajectories"]["roe1"]["lightbox"]["requested"] == 2
    assert [json.loads(line)["index"] for line in records] == [0, 1]


def test_cli_evaluate_shirt_framewise(scene, tmp_path, capsys):
    runs = tmp_path / "runs"
    base = ["evaluate", "shirt", "--method", "framewise", "--root", str(scene["root"]),
            "--runs", str(runs), "--checkpoint", str(scene["checkpoint"]),
            "--keypoints-3d", str(scene["layout"]), "--device", "cpu"]  # fmt: skip
    assert cli.main([*base, "--dry-run"]) == 0
    assert not runs.exists()
    assert cli.main(base) == 1
    assert "--allow-unpinned-checkpoint" in capsys.readouterr().out
    assert cli.main([*base, "--allow-unpinned-checkpoint"]) == 0
    summary = json.loads((runs / "framewise" / "summary.json").read_text())
    assert list(summary["trajectories"]) == ["roe1", "roe2"]
    assert summary["allow_unpinned_checkpoint"] is True
    # Flags of the other kind are refused, both ways.
    assert cli.main([*base, "--split", "val", "--dry-run"]) == 1
    assert "--split" in capsys.readouterr().out
    bop = ["evaluate", "tless", "--split", "val", "--root", str(tmp_path), "--runs", str(runs)]
    assert cli.main([*bop, "--method", "framewise", "--dry-run"]) == 1
    assert "--method" in capsys.readouterr().out
    assert cli.main(["evaluate", "tless", "--root", str(tmp_path), "--runs", str(runs)]) == 1
    assert "--split" in capsys.readouterr().out


def test_shirt_dry_run_imports_no_torch(tmp_path):
    code = (
        "import sys; from doper import cli; "
        "rc = cli.main(['evaluate', 'shirt', '--method', 'framewise', "
        f"'--root', {str(tmp_path)!r}, "
        f"'--runs', {str(tmp_path / 'runs')!r}, '--checkpoint', 'c.pth', '--keypoints-3d', "
        "'k.json', '--dry-run']); print(rc, 'torch' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "0 False"


def test_shirt_ground_truth_is_normalized_in_float64():
    import numpy as np

    from doper.speedplus import pose

    q = np.asarray([0.71, -0.2, 0.33, 0.58])
    unit = q / np.linalg.norm(q)
    single = pose.quaternion_to_object_camera(q, normalization="float64")
    batch = pose.quaternion_to_object_camera(np.stack([q, 2 * q]), normalization="float64")
    np.testing.assert_array_equal(batch[0], single)
    np.testing.assert_allclose(single @ single.T, np.eye(3), atol=1e-15)
    w, x, y, z = unit
    expected_00 = 2 * w**2 - 1 + 2 * x**2
    assert single[0, 0] == expected_00
    assert not np.array_equal(single, pose.quaternion_to_object_camera(q, normalization="float32"))
    with pytest.raises(ValueError, match="normalization"):
        pose.quaternion_to_object_camera(q, normalization="float16")


def test_official_summary_and_missing_image(scene):
    _framewise(scene)
    folder = scene["out"] / "roe2" / "lightbox"
    records = [json.loads(line) for line in (folder / "records.jsonl").read_text().splitlines()]
    paper = candidates.summarize_sequence(records, [1.0] * len(records))
    official = candidates.summarize_sequence(records, [], definition="official")
    assert paper["pose_score"] == json.loads((folder / "summary.json").read_text())["pose_score"]
    assert official["pose_score"]["max"] <= paper["pose_score"]["max"]
    assert official["inference_latency_ms"] == dict.fromkeys(candidates.DISTRIBUTION_KEYS)
    with pytest.raises(ValueError, match="at least one record"):
        candidates.summarize_sequence([], [])
    (scene["root"] / "roe1" / "lightbox" / "images" / sf.filename(2)).unlink()
    with pytest.raises(FileNotFoundError, match=sf.filename(2)):
        _framewise(scene, force=True)
