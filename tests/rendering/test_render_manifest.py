# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shard manifests are deterministic, portable and fail closed on identity changes."""

import json
import platform
import subprocess
import sys

import pytest
import yaml

from rendering.config import load_render_config
from rendering.run_manifest import (
    build_render_identity,
    check_shards,
    finish_shard,
    shard_manifest,
    shard_manifest_path,
    start_shard_attempt,
    validate_resume_identity,
)
from rendering.scene_spec import AssetCatalog, generate_scenes
from rendering.usd_mesh import write_mesh_usda

CUBE_POINTS = [(x, y, z) for x in (-0.03, 0.03) for y in (-0.02, 0.02) for z in (-0.01, 0.01)]
CUBE_FACES = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]


def write_inputs(root, *, absolute=False):
    """Write a config, a USD and a keypoints file into ``root``; return their paths.

    ``absolute`` writes the config's object and asset locations as absolute paths.
    """
    place = str if absolute else (lambda path: path.relative_to(root).as_posix())
    root.mkdir(parents=True, exist_ok=True)
    packs = {name: root / "packs" / name for name in ("hdri", "textures", "distractors")}
    for pack in packs.values():
        pack.mkdir(parents=True, exist_ok=True)
    config = {
        "output": {"resolution": 32, "render_masks": True},
        "object": {
            "num_keypoints": 2,
            "usd_path": place(root / "object.usda"),
            "keypoints_json": place(root / "keypoints_3d.json"),
        },
        "scene": {
            "num_distractors_min": 0,
            "num_distractors_max": 0,
            "hdri_dir": place(packs["hdri"]),
            "texture_dir": place(packs["textures"]),
            "distractor_dir": place(packs["distractors"]),
        },
    }
    config_path = root / "render.yaml"
    config_path.write_text(yaml.safe_dump(config))
    usd_path = root / "object.usda"
    write_mesh_usda(
        str(usd_path),
        object_name="object",
        points=CUBE_POINTS,
        face_vertex_counts=[4] * len(CUBE_FACES),
        face_vertex_indices=[i for face in CUBE_FACES for i in face],
        keypoint_names=["a", "b"],
        keypoints_3d=CUBE_POINTS[:2],
    )
    keypoints_path = root / "keypoints_3d.json"
    keypoints_path.write_text(json.dumps({"keypoints_3d": [list(p) for p in CUBE_POINTS[:2]]}))
    return config_path, usd_path, keypoints_path


def render_manifest_for(out, *, seed, inputs=None, absolute=False):
    """Shard manifest for a 4-sample shard; inputs default to ``out/inputs``."""
    out.mkdir(parents=True, exist_ok=True)
    config_path, usd_path, keypoints_path = write_inputs(
        inputs or out / "inputs", absolute=absolute
    )
    cfg = load_render_config(config_path)
    scenes_path = out / "scenes_00000000_00000004.jsonl"
    info = generate_scenes(
        cfg,
        usd_path=str(usd_path),
        keypoints_json=str(keypoints_path),
        seed=seed,
        start_idx=0,
        end_idx=4,
        out_path=str(scenes_path),
    )
    return shard_manifest(
        repo_root=str(out),
        cfg=cfg,
        config_path=str(config_path),
        usd_path=str(usd_path),
        keypoints_path=str(keypoints_path),
        num_keypoints=info["num_keypoints"],
        catalog_summary=info["catalog"].summary(),
        seed=seed,
        start_idx=0,
        end_idx=4,
        workers=[{"worker_id": 0, "gpu": 0}],
        scenes_path=str(scenes_path),
        renderer="blender",
        outdir=str(out),
    )


def strip_volatile(manifest):
    return {key: value for key, value in manifest.items() if key != "created_utc"}


@pytest.fixture(autouse=True)
def fixed_revision(monkeypatch):
    monkeypatch.setenv("DOPER_GIT_REVISION", "test-revision")
    monkeypatch.setenv("DOPER_IMAGE", "doper:test")


def test_manifest_is_deterministic(tmp_path):
    a, b = (render_manifest_for(tmp_path / n, seed=5) for n in ("a", "b"))
    assert strip_volatile(a) == strip_volatile(b)
    assert a["code"] == {
        "git_revision": "test-revision",
        "image": "doper:test",
        "python": a["code"]["python"],
    }


def test_manifest_has_no_absolute_paths(tmp_path):
    text = json.dumps(render_manifest_for(tmp_path / "out", seed=5, absolute=True))
    assert str(tmp_path) not in text
    assert platform.node() not in text


def test_paths_are_relative_inside_the_output_directory(tmp_path):
    manifest = render_manifest_for(tmp_path / "out", seed=5)
    assert manifest["config_path"] == "inputs/render.yaml"
    assert manifest["object"]["usd"] == "inputs/object.usda"
    assert manifest["object"]["keypoints_json"] == "inputs/keypoints_3d.json"


def test_paths_outside_the_output_directory_are_base_names(tmp_path):
    manifest = render_manifest_for(tmp_path / "out", seed=5, inputs=tmp_path / "elsewhere")
    assert manifest["config_path"] == "render.yaml"
    assert manifest["object"]["usd"] == "object.usda"
    assert manifest["object"]["keypoints_json"] == "keypoints_3d.json"
    assert (
        manifest["config_sha256"] == render_manifest_for(tmp_path / "o2", seed=5)["config_sha256"]
    )


def test_config_values_use_portable_paths(tmp_path):
    inside = render_manifest_for(tmp_path / "out", seed=5, absolute=True)["config_values"]
    assert inside["hdri_dir"] == "inputs/packs/hdri"
    assert inside["usd_path"] == "inputs/object.usda"
    outside = render_manifest_for(
        tmp_path / "o2", seed=5, inputs=tmp_path / "elsewhere", absolute=True
    )
    assert outside["config_values"]["texture_dir"] == "textures"
    assert outside["config_values"]["keypoints_json"] == "keypoints_3d.json"


def test_manifest_content_hashes_follow_the_inputs(tmp_path):
    a = render_manifest_for(tmp_path / "a", seed=5)
    b = render_manifest_for(tmp_path / "b", seed=6)
    assert a["object"] == b["object"]
    assert a["scenes"]["sha256"] != b["scenes"]["sha256"]


def identity_with_settings(tmp_path, settings):
    config_path, usd_path, _ = write_inputs(tmp_path / "inputs")
    cfg = load_render_config(config_path)
    return build_render_identity(
        cfg=cfg,
        catalog=AssetCatalog.scan(cfg),
        seed=1,
        renderer="blender",
        renderer_version="v1",
        usd_path=str(usd_path),
        renderer_settings=settings,
    )


def test_resume_rejects_blender_version_change(tmp_path):
    out = tmp_path / "out"
    original = identity_with_settings(tmp_path, {"blender_version": "4.2.1"})
    start_shard_attempt(shard_manifest_path(out, 0, 4), {"status": "running"}, original)
    validate_resume_identity(out, original, require_existing=True)
    upgraded = identity_with_settings(tmp_path, {"blender_version": "4.2.3"})
    assert upgraded.sha256 != original.sha256
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_resume_identity(out, upgraded)


def test_shard_io_imports_without_scipy():
    code = "import sys, rendering.shard_io; print('scipy' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def shard(start, end, status="complete"):
    return {
        "status": status,
        "shard": {"start_idx": start, "end_idx": end},
        "render_identity": {"seed": 1},
        "images_per_sample": 1,
        "attempts": [],
    }


def test_check_shards_reports_gaps_overlaps_and_unfinished_shards():
    assert check_shards([shard(0, 4), shard(4, 8)], expect_total=8) == []
    assert any("gap" in p for p in check_shards([shard(0, 4), shard(6, 8)]))
    assert any("overlap" in p for p in check_shards([shard(0, 5), shard(4, 8)]))
    assert any("status" in p for p in check_shards([shard(0, 4, "running")]))
    assert any("expected" in p for p in check_shards([shard(0, 4)], expect_total=8))


def test_finish_shard_marks_failure(tmp_path):
    identity = identity_with_settings(tmp_path, None)
    path = shard_manifest_path(tmp_path / "out", 0, 4)
    start_shard_attempt(path, {"status": "running"}, identity)
    failed = finish_shard(path, images_rendered=2, elapsed_sec=1.234, failed_workers=[1])
    assert failed["status"] == "failed" and failed["elapsed_sec"] == 1.2
