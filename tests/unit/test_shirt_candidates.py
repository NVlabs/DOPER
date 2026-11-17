# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SHIRT 0/180 re-crop pose candidates (``doper.shirt.candidates.crop_candidates``).

``test_candidates_match_source`` compares the port with ``golden/source_shirt_candidates.json``,
the re-crop candidates the research candidate generation (the research source snapshot,
``evaluate_shirt_auxiliary_flip_gate``) produces for the ``shirt_crops`` tree with the same stub
heatmap model (see that module). The parity case ``shirt_candidates`` runs both sides in one
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
import shirt_crops as sc
import torch
from golden_floats import approx_floats
from tiny_coco import tiny_model_config

from doper import cli, infer
from doper.shirt import candidates
from doper.speedplus.checkpoints import LAYOUT_POINTS_SHA_KEY, LAYOUT_SHA_KEY, load_layout

GOLDEN = Path(__file__).parent / "golden" / "source_shirt_candidates.json"
AUX_RECORD_FIELDS = (
    "raw_aux_kl_by_angle", "raw_aux_kl_mean", "raw_aux_kl_max", "raw_aux_mass_within_45deg",
    "aux_top1_gt_error_deg_by_angle", "aux_top1_gt_error_deg_best", "aux_selected_index",
    "aux_selected",
)  # fmt: skip
AUX_CANDIDATE_FIELDS = ("aux_kl_mean", "aux_kl_max", "aux_mass_within_45deg")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint(path: Path, layout_sha: str, cfg: dict | None = None) -> Path:
    payload: dict = {"model": {"layer.weight": torch.ones(1)}, LAYOUT_SHA_KEY: layout_sha}
    if cfg is not None:
        payload["cfg"] = cfg
    torch.save(payload, path)
    return path


@pytest.fixture
def scene(tmp_path: Path, monkeypatch) -> dict:
    """The landmark-disc tree, framewise records, the layout, a checkpoint and the stub."""
    root = sc.write_crop_tree(tmp_path / "shirt")
    loader = sc.make_stub_loader()
    monkeypatch.setattr(infer, "load_model", loader)
    layout = sc.sf.write_layout(tmp_path / "keypoints_3d.json")
    return {"root": root, "layout": layout, "model": loader.model,
            "checkpoint": _checkpoint(tmp_path / "epoch_10.pth", _sha256(layout)),
            "records": sc.write_framewise_records(tmp_path / "framewise"),
            "out": tmp_path / "out"}  # fmt: skip


def _run(scene: dict, **kwargs) -> dict:
    kwargs.setdefault("allow_unpinned_checkpoint", True)
    return candidates.crop_candidates(
        scene["checkpoint"], scene["root"], scene["layout"], scene["records"], scene["out"],
        trajectories=sc.TRAJECTORIES, **kwargs,
    )  # fmt: skip


def _records(scene: dict, roe: str) -> list[dict]:
    path = scene["out"] / roe / sc.DOMAIN / "records.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _source_golden_or_skip() -> dict:
    """The candidate golden without its versions; skip unless this env has the same builds."""
    golden = json.loads(GOLDEN.read_text())
    recorded, current = golden.pop("versions"), sc.library_versions()
    if recorded != current:
        pytest.skip(f"the candidate golden was produced with {recorded}, this env has "
                    f"{current}; {sc.REGENERATE}")  # fmt: skip
    return approx_floats(golden)


def test_candidates_match_source(scene):
    summary = _run(scene)
    assert sc.canonical_candidates(scene["out"]) == _source_golden_or_skip()
    # The crops use the loaded checkpoint's geometry: two rolls per solved raw frame.
    crops = scene["model"].batches
    assert all(shape[1:] == [3, sc.CROP_SIZE, sc.CROP_SIZE] for shape in crops)
    assert sum(shape[0] for shape in crops) == 2 * (2 * sc.FRAMES - len(sc.FAILED_RAW_FRAMES))
    assert json.loads((scene["out"] / "summary.json").read_text()) == summary
    assert summary["method"] == "candidates" and summary["domain"] == "lightbox"
    assert summary["rotation_tta_angles_deg"] == [0.0, 180.0]
    assert summary["crop_size"] == sc.CROP_SIZE
    assert summary["crop_pad_factor"] == sc.CROP_PAD_FACTOR
    assert summary["keypoint_heatmap_beta"] == 1.0
    assert summary["layout_check"] == "recorded"
    roe2 = summary["trajectories"]["roe2"]["lightbox"]
    assert roe2["candidates_by_source"] == {"raw": 5, "crop_roll_0": 4, "crop_roll_180": 4}
    # The re-crops recover the true pose better than the perturbed raw pose.
    record = _records(scene, "roe1")[0]
    raw, *crops_ = record["candidates"]
    assert raw["rotation_deg"] == sc.RAW_ROTATION_DEG
    assert all(crop["rotation_deg"] < 1.0 for crop in crops_)
    assert str(scene["root"]) not in (scene["out"] / "summary.json").read_text()


def test_candidates_golden_skips_on_other_builds(monkeypatch):
    monkeypatch.setattr(sc, "library_versions", lambda: {"opencv": "0", "torch": "0"})
    with pytest.raises(pytest.skip.Exception, match="research source snapshot"):
        _source_golden_or_skip()


def test_frame0_failure_refused(scene, tmp_path):
    scene["records"] = sc.write_framewise_records(tmp_path / "failed0", failed=(("roe2", 0),))
    with pytest.raises(ValueError, match="no initial pose for roe2 frame 0"):
        _run(scene)
    assert not scene["out"].exists() and scene["model"].batches == []


def test_failed_raw_frame_kept_as_empty_candidates(scene):
    summary = _run(scene)
    records = _records(scene, "roe1")
    assert [record["index"] for record in records] == list(range(sc.FRAMES))
    failed = records[3]
    assert failed["candidates"] == [] and failed["raw_rotation_deg"] is None
    assert failed["filename"] == sc.sf.filename(3)
    roe1 = summary["trajectories"]["roe1"]["lightbox"]
    assert roe1["frames"] == sc.FRAMES
    assert roe1["empty_candidate_frames"] == [3]
    # A frame whose re-crops do not solve keeps its raw candidate only.
    blank = _records(scene, "roe2")[2]
    assert [candidate["source"] for candidate in blank["candidates"]] == ["raw"]


def test_aux_fields_null(scene):
    _run(scene)
    for roe in sc.TRAJECTORIES:
        for record in _records(scene, roe):
            assert all(record[field] is None for field in AUX_RECORD_FIELDS)
            for candidate in record["candidates"]:
                assert all(candidate[field] is None for field in AUX_CANDIDATE_FIELDS)


def test_heatmap_beta_comes_from_the_checkpoint(scene, tmp_path, monkeypatch):
    from doper import targets

    seen: list[float] = []
    decode = targets.decode_keypoint_heatmaps

    def spy(logits, crop_size, beta=1.0, **kwargs):
        seen.append(beta)
        return decode(logits, crop_size, beta=beta, **kwargs)

    monkeypatch.setattr(targets, "decode_keypoint_heatmaps", spy)
    scene["checkpoint"] = _checkpoint(
        tmp_path / "beta.pth", _sha256(scene["layout"]), {"loss": {"keypoint_heatmap_beta": 2.5}}
    )
    summary = _run(scene)
    assert summary["keypoint_heatmap_beta"] == 2.5 and set(seen) == {2.5}


def test_candidates_refuse_bad_inputs(scene, tmp_path):
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"keypoints_3d": [[0.0, 0.0, 1.0]] * 8}))
    scene_other = {**scene, "checkpoint": _checkpoint(tmp_path / "o.pth", _sha256(other))}
    with pytest.raises(ValueError, match=f"{_sha256(other)}.*{_sha256(scene['layout'])}"):
        _run(scene_other)
    with pytest.raises(FileNotFoundError, match="framewise"):
        _run({**scene, "records": tmp_path / "missing"})
    # Framewise records of another checkpoint are refused.
    (scene["records"] / "summary.json").write_text(json.dumps({"checkpoint_sha256": "f" * 64}))
    with pytest.raises(ValueError, match="f{64}"):
        _run(scene)
    (scene["records"] / "summary.json").unlink()
    assert not scene["out"].exists()
    _run(scene)
    with pytest.raises(FileExistsError, match="--force"):
        _run(scene)
    limited = _run(scene, limit=2, force=True)
    assert limited["trajectories"]["roe1"]["lightbox"]["frames"] == 2


def test_candidates_record_the_layout_identity(scene, tmp_path):
    trained = load_layout(scene["layout"])
    summary = _run(scene)
    assert summary["layout_points_sha256"] == trained.points_sha256
    assert summary["layout_formatting_deviation"] is False
    reformatted = tmp_path / "reformatted.json"
    reformatted.write_text(json.dumps(json.loads(scene["layout"].read_text()), indent=2))
    torch.save({"model": {"layer.weight": torch.ones(1)}, LAYOUT_SHA_KEY: trained.sha256,
                LAYOUT_POINTS_SHA_KEY: trained.points_sha256}, scene["checkpoint"])  # fmt: skip
    summary = _run({**scene, "layout": reformatted}, force=True)
    assert summary["layout_check"] == "recorded" and summary["layout_formatting_deviation"]


def test_candidates_run_a_real_tiny_model(tmp_path):
    """No stub: the real loader and the backbone/neck/tower/out forward of a tiny model."""
    from doper.model import DoperNet

    root = sc.write_crop_tree(tmp_path / "shirt")
    layout = sc.sf.write_layout(tmp_path / "keypoints_3d.json")
    model_cfg = tiny_model_config(
        num_keypoints=8, det_input_size=64, neck_channels=8, head_channels=8, crop_size=32,
        keypoint_head="heatmap",
    )  # fmt: skip
    torch.manual_seed(0)
    checkpoint = tmp_path / "tiny.pth"
    torch.save({"model": DoperNet(model_cfg).state_dict(),
                "cfg": {"model": dataclasses.asdict(model_cfg)},
                LAYOUT_SHA_KEY: _sha256(layout)}, checkpoint)  # fmt: skip
    summary = candidates.crop_candidates(
        checkpoint, root, layout, sc.write_framewise_records(tmp_path / "framewise"),
        tmp_path / "out", trajectories=("roe1",), allow_unpinned_checkpoint=True, limit=2,
    )  # fmt: skip
    assert summary["crop_size"] == 32
    records = (tmp_path / "out" / "roe1" / "lightbox" / "records.jsonl").read_text().splitlines()
    assert [json.loads(line)["index"] for line in records] == [0, 1]
    assert all(json.loads(line)["candidates"][0]["source"] == "raw" for line in records)


def test_cli_evaluate_shirt_candidates(scene, tmp_path):
    runs = tmp_path / "runs"
    sc.write_framewise_records(runs / "framewise")
    base = ["evaluate", "shirt", "--method", "candidates", "--root", str(scene["root"]),
            "--runs", str(runs), "--checkpoint", str(scene["checkpoint"]),
            "--keypoints-3d", str(scene["layout"]), "--device", "cpu",
            "--allow-unpinned-checkpoint"]  # fmt: skip
    assert cli.main([*base, "--dry-run"]) == 0
    assert not (runs / "candidates").exists()
    assert cli.main(base) == 0
    summary = json.loads((runs / "candidates" / "summary.json").read_text())
    assert list(summary["trajectories"]) == ["roe1", "roe2"]
    roe1 = (runs / "candidates" / "roe1" / "lightbox" / "records.jsonl").read_text()
    assert len(roe1.splitlines()) == sc.FRAMES


def test_candidates_dry_run_imports_no_torch(tmp_path):
    code = (
        "import sys; from doper import cli; "
        "rc = cli.main(['evaluate', 'shirt', '--method', 'candidates', "
        f"'--root', {str(tmp_path)!r}, "
        f"'--runs', {str(tmp_path / 'runs')!r}, '--checkpoint', 'c.pth', '--keypoints-3d', "
        "'k.json', '--dry-run']); print(rc, 'torch' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "0 False"


def test_roll_angles_are_the_source_labels():
    assert candidates.ROLL_ANGLES_DEG == (0.0, 180.0)
    assert [candidates.roll_source(angle) for angle in candidates.ROLL_ANGLES_DEG] == [
        "crop_roll_0", "crop_roll_180",
    ]  # fmt: skip
    assert candidates.DEFAULT_BATCH_SIZE == 32
