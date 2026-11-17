# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Blender launcher: argument contract, version gate, worker-output merge, seeded renders."""

import json
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from conftest import part_recipe, run_launcher
from PIL import Image

from doper.scene import geometry as G  # noqa: N812
from doper.scene import labels as coco
from rendering import synth_parallel
from rendering.dataset_contract import merge_annotations
from rendering.usd_mesh import keypoints_payload


@pytest.fixture
def fake_blender(tmp_path):
    """Factory for an executable that prints a Blender version line and nothing else."""

    def make(version: str) -> str:
        path = tmp_path / f"blender_{version}"
        path.write_text(f'#!/bin/sh\necho "Blender {version}"\n')
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        return str(path)

    return make


def required_args(tmp_path: Path) -> list[str]:
    return [
        "--blend-file", str(tmp_path / "object.blend"),
        "--config", str(tmp_path / "render.yaml"),
        "--outdir", str(tmp_path / "out"),
    ]  # fmt: skip


def test_paths_are_required():
    with pytest.raises(SystemExit):
        synth_parallel.main(["--num-images", "1"])


def test_old_blender_version_refused(tmp_path, fake_blender):
    with pytest.raises(SystemExit, match="4.2") as info:
        synth_parallel.main(required_args(tmp_path) + ["--blender-bin", fake_blender("3.6.5")])
    assert "3.6.5" in str(info.value)


def test_missing_blender_is_reported(tmp_path):
    args = required_args(tmp_path) + ["--blender-bin", str(tmp_path / "nope")]
    with pytest.raises(SystemExit, match="not found"):
        synth_parallel.main(args)


def test_missing_blend_file_is_reported(tmp_path, fake_blender):
    with pytest.raises(SystemExit, match="Blend file not found"):
        synth_parallel.main(required_args(tmp_path) + ["--blender-bin", fake_blender("4.2.1")])


def test_resolution_must_match_the_config(tmp_path, fake_blender):
    (tmp_path / "object.blend").touch()
    (tmp_path / "render.yaml").write_text(yaml.safe_dump({"output": {"resolution": 32}}))
    args = required_args(tmp_path) + ["--blender-bin", fake_blender("4.2.1"), "--resolution", "64"]
    with pytest.raises(SystemExit) as info:
        synth_parallel.main(args)
    assert info.value.code == 2


@pytest.mark.parametrize("suffix", [".usd", ".usdc"])
def test_preflight_rejects_unsupported_usd_before_blender(monkeypatch, suffix):
    def unexpected(*args, **kwargs):
        pytest.fail("Blender inspection started for an unsupported object layer")

    monkeypatch.setattr(synth_parallel, "inspect_blender_inputs", unexpected)
    with pytest.raises(ValueError, match=r"writer-generated \.usda"):
        synth_parallel.validate_blender_inputs(
            "unused", "object.blend", None, "object" + suffix, None
        )


def test_worker_payload_merges_with_synth_parallel(tmp_path):
    intr = G.intrinsics_from_lens(640, 640, 50.0, 36.0)
    cam = G.look_at_camera((0.0, -0.3, 0.3), (0.0, 0.0, 0.0))
    uv = np.array([[320.0, 320.0], [330.0, 310.0]])
    vis = np.array([2, 1])
    images, anns = [], []
    for image_id in range(2):
        fname = f"{image_id:06d}.png"
        images.append(
            coco.image_record(image_id, fname, 640, 640, pair_id=None, view="mono", camera_index=0)
        )
        anns.append(
            coco.annotation_record(
                image_id, uv=uv, vis=vis, bbox=coco.keypoints_bbox(uv, vis, 640, 640), intr=intr,
                cam_to_world=cam, object_pose=np.eye(4), view="mono", camera_index=0,
                mask_file=f"masks/{fname}",
            )
        )  # fmt: skip
        (tmp_path / "images").mkdir(exist_ok=True)
        (tmp_path / "images" / fname).write_bytes(b"png")
        (tmp_path / "masks").mkdir(exist_ok=True)
        (tmp_path / "masks" / fname).write_bytes(b"")
    payload = coco.worker_payload(
        worker_id=0, start_idx=0, end_idx=2, images=images, annotations=anns, render_masks=True,
        mask_dir="masks", stereo_enabled=False, stereo_baseline_m=0.063,
    )  # fmt: skip
    (tmp_path / "annotations_worker_000.json").write_text(json.dumps(payload))
    keypoints = keypoints_payload("clip", ["a", "b"], np.zeros((2, 3)))
    (tmp_path / "keypoints_3d.json").write_text(json.dumps(keypoints))
    worker = synth_parallel.Worker(0, 0, 2, tmp_path / "w.log")
    assert synth_parallel.validate_worker_outputs(tmp_path, [worker]) == []
    counts = merge_annotations(tmp_path, 1)
    assert counts == {"images": 2, "annotations": 2, "masks": True}
    merged = json.loads((tmp_path / "annotations.json").read_text())
    assert merged["info"]["mask_dir"] == "masks"
    assert merged["categories"][0]["keypoints"] == ["a", "b"]
    ann = merged["annotations"][0]
    for key in ("keypoints", "num_keypoints", "bbox", "area", "camera_intrinsics",
                "camera_extrinsics", "object_pose", "view", "camera_index", "mask_file",
                "mask_type"):  # fmt: skip
        assert key in ann
    assert ann["num_keypoints"] == 2 and len(ann["keypoints"]) == 6


def test_missing_worker_annotations_are_reported(tmp_path):
    worker = synth_parallel.Worker(3, 0, 2, tmp_path / "w.log")
    (error,) = synth_parallel.validate_worker_outputs(tmp_path, [worker])
    assert "Worker 3: missing" in error
    assert str(tmp_path) not in error and "see w.log" in error


def test_configured_object_usd_that_is_missing_is_an_error(tmp_path, fake_blender):
    (tmp_path / "object.blend").touch()
    (tmp_path / "render.yaml").write_text(yaml.safe_dump({"object": {"usd_path": "missing.usda"}}))
    with pytest.raises(SystemExit, match="Object USD not found"):
        synth_parallel.main(required_args(tmp_path) + ["--blender-bin", fake_blender("4.2.1")])


def test_object_name_mismatch_in_supplied_keypoints_is_explicit(tmp_path, monkeypatch):
    authored = {
        "points": [[0.0, 0.0, 0.0]],
        "keypoints_3d": [[0.0, 0.0, 0.0]],
        "keypoint_names": ["KP_object_000"],
    }
    monkeypatch.setattr(synth_parallel, "inspect_blender_inputs", lambda *a, **k: authored)
    monkeypatch.setattr(synth_parallel, "read_mesh_usda", lambda path: authored)
    keypoints = tmp_path / "kp.json"
    keypoints.write_text(json.dumps(dict(authored, object="bracket")))
    config = tmp_path / "render.yaml"
    config.write_text(yaml.safe_dump({"object": {"mesh_name": "object"}}))
    with pytest.raises(ValueError, match="'bracket'.*mesh_name is 'object'"):
        synth_parallel.validate_blender_inputs(
            "unused",
            "object.blend",
            synth_parallel.load_render_config(config),
            "object.usda",
            str(keypoints),
        )


def render_args(inputs, blender, out, num_images, *extra):
    config = inputs / "recipe.yaml"
    config.write_text(yaml.safe_dump(part_recipe(inputs)))
    return [
        "--num-images", str(num_images), "--num-workers", "1", "--seed", "11",
        "--blend-file", str(inputs / "part.blend"), "--config", str(config),
        "--outdir", str(out), "--usd", str(inputs / "part.usda"),
        "--keypoints-json", str(inputs / "correct_keypoints.json"),
        "--blender-bin", blender, *extra,
    ]  # fmt: skip


def render(inputs, blender, out, num_images, *extra):
    result = run_launcher(*render_args(inputs, blender, out, num_images, *extra))
    log = out / "worker_000.log"
    assert result.returncode == 0, (
        result.stdout + result.stderr + (log.read_text() if log.exists() else "")
    )


@pytest.fixture(scope="module")
def clean_run(part_inputs, blender, tmp_path_factory) -> Path:
    """The seed-11, four-image, unmerged render that several tests compare against."""
    out = tmp_path_factory.mktemp("clean_run") / "out"
    render(part_inputs, blender, out, 4, "--no-merge")
    return out


def outputs(out: Path) -> dict[str, bytes]:
    """Every rendered artifact plus the worker annotations.

    PNGs compare by decoded pixels: Blender's encoder output is not byte-stable.
    """
    files = [p for sub in ("images", "masks", "scenes") for p in sorted((out / sub).glob("*"))]
    data = {
        str(p.relative_to(out)): (
            np.asarray(Image.open(p)).tobytes() if p.suffix == ".png" else p.read_bytes()
        )
        for p in files
    }
    for path in sorted(out.glob("annotations*.json")):
        data[path.name] = path.read_bytes()
    return data


@pytest.mark.blender
def test_two_renders_with_same_seed_are_identical(part_inputs, blender, clean_run, tmp_path):
    render(part_inputs, blender, tmp_path / "again", 4, "--no-merge")
    first, second = outputs(clean_run), outputs(tmp_path / "again")
    assert any(name.startswith("images/") for name in first)
    assert any(name.startswith("scenes/") and first[name] for name in first)
    assert first == second


@pytest.mark.blender
def test_resume_completes_interrupted_render(part_inputs, blender, clean_run, tmp_path):
    broken = tmp_path / "broken"
    shutil.copytree(clean_run, broken)
    final = broken / "annotations_worker_000.json"
    payload = json.loads(final.read_text())
    keep = {"images": 2, "annotations": 2}
    lost = {img["file_name"] for img in payload["images"][2:]}
    payload.update({key: payload[key][:n] for key, n in keep.items()})
    final.unlink()
    (broken / "annotations_worker_000.partial.json").write_text(json.dumps(payload))
    for name in lost:
        (broken / "images" / name).unlink()
        (broken / "masks" / name).unlink()
    render(part_inputs, blender, broken, 4, "--no-merge", "--resume")
    manifest = json.loads((broken / "manifests/shard_00000000_00000004.json").read_text())
    assert manifest["reused_sample_indices"] == [0, 1]
    assert outputs(broken) == outputs(clean_run)


@pytest.mark.blender
def test_sharded_render_merges_with_run_manifest(part_inputs, blender, tmp_path):
    out = tmp_path / "out"
    render(part_inputs, blender, out, 2, "--no-merge")
    render(
        part_inputs, blender, out, 2, "--no-merge", "--start-idx", "2", "--worker-id-offset", "1"
    )
    assert not (out / "annotations.json").exists()
    script = Path(__file__).resolve().parents[2] / "rendering" / "run_manifest.py"
    done = subprocess.run(
        [sys.executable, str(script), "merge", str(out), "--expect-total", "4"],
        cwd=tmp_path, capture_output=True, text=True,
    )  # fmt: skip
    assert done.returncode == 0, done.stdout + done.stderr
    merged = json.loads((out / "annotations.json").read_text())
    assert len(merged["images"]) == len(merged["annotations"]) == 4
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["num_samples"] == 4


def dry_run_inputs(tmp_path: Path) -> list[str]:
    """Arguments of a valid ``--dry-run`` whose Blender binary and inputs are placeholders."""
    (tmp_path / "object.blend").touch()
    (tmp_path / "object.usda").touch()
    (tmp_path / "render.yaml").write_text(yaml.safe_dump({"output": {"resolution": 32}}))
    return [*required_args(tmp_path), "--usd", str(tmp_path / "object.usda"), "--dry-run"]


def test_dry_run_plans_without_launching_blender(tmp_path, monkeypatch, capsys):
    def unexpected(*args, **kwargs):
        pytest.fail("a dry run started a process")

    monkeypatch.setattr(subprocess, "Popen", unexpected)
    monkeypatch.setattr(subprocess, "run", unexpected)
    args = dry_run_inputs(tmp_path) + ["--num-images", "5", "--num-workers", "2", "--seed", "3"]
    assert synth_parallel.main(args) == 0
    out = capsys.readouterr().out
    assert "Worker 0: samples [0, 3) = 3" in out
    assert "Worker 1: samples [3, 5) = 2" in out
    assert "32x32" in out
    assert not (tmp_path / "out").exists()


def test_dry_run_refuses_a_missing_blend_file(tmp_path):
    args = dry_run_inputs(tmp_path)
    (tmp_path / "object.blend").unlink()
    with pytest.raises(SystemExit, match="Blend file not found") as info:
        synth_parallel.main(args)
    assert str(tmp_path / "object.blend") in str(info.value)
    assert not (tmp_path / "out").exists()


def test_dry_run_refuses_a_missing_usd(tmp_path):
    args = dry_run_inputs(tmp_path)
    (tmp_path / "object.usda").unlink()
    with pytest.raises(SystemExit, match="Object USD not found"):
        synth_parallel.main(args)


def test_dry_run_applies_the_resolution_check(tmp_path, capsys):
    with pytest.raises(SystemExit) as info:
        synth_parallel.main(dry_run_inputs(tmp_path) + ["--resolution", "64"])
    assert info.value.code == 2
    assert "--resolution 64 differs from output.resolution" in capsys.readouterr().err
