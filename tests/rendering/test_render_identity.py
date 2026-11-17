# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launcher identity gates must never relabel already rendered samples."""

import json
import shutil

import pytest

from rendering.config import render_config_from_mapping
from rendering.run_manifest import (
    build_render_identity,
    finish_shard,
    shard_manifest_path,
    start_shard_attempt,
    validate_resume_identity,
)
from rendering.scene_spec import AssetCatalog


def setup_inputs(root):
    root.mkdir(parents=True, exist_ok=True)
    for name in (
        "object.blend",
        "object.usd",
        "keypoints.json",
        "object.png",
        "assets/hdri/studio.hdr",
        "assets/textures/cco_textures/wood/albedo.png",
        "assets/distractors/clutter.usda",
        "assets/distractors/clutter.png",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    return root


ASSET_DIRS = {
    "hdri_dir": "assets/hdri",
    "texture_dir": "assets/textures/cco_textures",
    "distractor_dir": "assets/distractors",
}


def config(root, raw=None):
    raw = dict(raw or {})
    raw["scene"] = {**ASSET_DIRS, **raw.get("scene", {})}
    return render_config_from_mapping(raw, str(root))


def identity(root, *, seed=1, raw=None, version="blender-scenes-v2", settings=None):
    cfg = config(root, raw)
    return build_render_identity(
        cfg=cfg,
        catalog=AssetCatalog.scan(cfg),
        seed=seed,
        renderer="blender",
        renderer_version=version,
        blend_path=str(root / "object.blend"),
        usd_path=str(root / "object.usd"),
        keypoints_path=str(root / "keypoints.json"),
        texture_paths=[str(root / "object.png")],
        renderer_settings=settings,
    )


def save_attempt(out, ident, revision="old", workers=1):
    path = shard_manifest_path(out, 0, 10)
    start_shard_attempt(
        path,
        {"code": {"git_revision": revision}, "shard": {"workers": workers}, "status": "running"},
        ident,
    )
    return path


@pytest.mark.parametrize(
    "changed",
    [
        "object.blend",
        "object.usd",
        "keypoints.json",
        "object.png",
        "assets/hdri/studio.hdr",
        "assets/textures/cco_textures/wood/albedo.png",
        "assets/distractors/clutter.png",
    ],
)
def test_changed_bytes_rejected_without_output_mutation(tmp_path, changed):
    root = setup_inputs(tmp_path / "inputs")
    out = tmp_path / "output"
    original = identity(root)
    save_attempt(out, original)
    before = {str(p): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    (root / changed).write_bytes(b"replaced in place")
    updated = identity(root)
    assert updated.sha256 != original.sha256
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_resume_identity(out, updated)
    assert before == {str(p): p.read_bytes() for p in out.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "changes",
    [
        {"seed": 2},
        {"raw": {"render": {"exposure": 1.0}}},
        {"raw": {"scene": {"procedural_distractors": True}}},
        {"version": "blender-scenes-v3"},
        {"settings": {"blender_version": "9.0"}},
    ],
)
def test_scene_and_renderer_changes_rejected(tmp_path, changes):
    root = setup_inputs(tmp_path / "inputs")
    out = tmp_path / "output"
    save_attempt(out, identity(root))
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_resume_identity(out, identity(root, **changes))


def test_relocation_and_worker_layout_do_not_change_identity(tmp_path):
    root = setup_inputs(tmp_path / "original")
    relocated = tmp_path / "elsewhere"
    shutil.copytree(root, relocated)
    original = identity(root, raw={"workers": 1, "output": {"outdir": "/old"}})
    moved = identity(relocated, raw={"workers": 8, "output": {"outdir": "/new"}})
    assert original == moved
    assert original.sha256 == moved.sha256
    out = tmp_path / "output"
    save_attempt(out, original)
    validate_resume_identity(out, moved, require_existing=True)
    save_attempt(out, moved, revision="new", workers=8)


def test_absolute_input_paths_do_not_leak_into_identity(tmp_path):
    root = setup_inputs(tmp_path / "inputs")
    raw = {
        "object": {
            "usd_path": str(root / "object.usd"),
            "keypoints_json": str(root / "keypoints.json"),
        },
        "scene": {"hdri_dir": str(root / "assets/hdri")},
    }
    # The catalog name fingerprint remains relative to its own root.
    assert identity(root, raw=raw) == identity(root)


def test_attempt_history_retains_original_code_and_results(tmp_path):
    root = setup_inputs(tmp_path / "inputs")
    out = tmp_path / "output"
    ident = identity(root)
    path = save_attempt(out, ident)
    first = finish_shard(path, images_rendered=3, elapsed_sec=5, failed_workers=[0])
    save_attempt(out, ident, revision="second", workers=3)
    second = finish_shard(path, images_rendered=7, elapsed_sec=8, failed_workers=[])
    save_attempt(out, ident, revision="third", workers=2)
    current = json.loads(path.read_text())
    assert current["code"]["git_revision"] == "third"
    expected_first = dict(first)
    expected_first.pop("attempts")
    expected_second = dict(second)
    expected_second.pop("attempts")
    assert current["attempts"] == [expected_first, expected_second]
    assert current["attempts"][0]["render_identity"] == ident.to_dict()
    validate_resume_identity(out, ident)


@pytest.mark.parametrize(
    "artifact",
    [
        "annotations.json",
        "annotations_worker_001.partial.json",
        "images/0001.png",
        "resume/resume_0001.json",
        "scenes.jsonl",
        "manifest.json",
        "manifests/shard_00000000_00000010.json",
    ],
)
def test_legacy_output_fails_closed(tmp_path, artifact):
    root = setup_inputs(tmp_path / "inputs")
    out = tmp_path / "output"
    path = out / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    with pytest.raises(ValueError, match="legacy"):
        validate_resume_identity(out, identity(root))
    assert path.read_text() == "{}"


def test_empty_output_allowed_only_for_new_render(tmp_path):
    root = setup_inputs(tmp_path / "inputs")
    out = tmp_path / "output"
    ident = identity(root)
    validate_resume_identity(out, ident)
    with pytest.raises(ValueError, match="resume requires"):
        validate_resume_identity(out, ident, require_existing=True)
    assert not out.exists()


def test_mismatching_attempt_cannot_be_hidden_by_current_identity(tmp_path):
    root = setup_inputs(tmp_path / "inputs")
    out = tmp_path / "output"
    path = save_attempt(out, identity(root))
    payload = json.loads(path.read_text())
    payload["attempts"] = [{"render_identity": identity(root, seed=9).to_dict()}]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_resume_identity(out, identity(root))


def test_direct_attempt_write_rejects_identity_replacement(tmp_path):
    root = setup_inputs(tmp_path / "inputs")
    out = tmp_path / "output"
    path = save_attempt(out, identity(root))
    original = path.read_bytes()
    with pytest.raises(ValueError, match="identity mismatch"):
        save_attempt(out, identity(root, seed=9))
    assert path.read_bytes() == original


def test_swapping_two_external_dependencies_changes_identity(tmp_path):
    root = setup_inputs(tmp_path / "inputs")
    a, b = root / "first.png", root / "second.png"
    a.write_bytes(b"first")
    b.write_bytes(b"second")
    cfg = config(root)

    def build():
        return build_render_identity(
            cfg=cfg,
            catalog=AssetCatalog.scan(cfg),
            seed=1,
            renderer="blender",
            renderer_version="v2",
            blend_path=str(root / "object.blend"),
            texture_paths=[str(a), str(b)],
        )

    before = build()
    a.write_bytes(b"second")
    b.write_bytes(b"first")
    assert before.sha256 != build().sha256


def test_blender_override_omits_unused_usd_texture_but_keeps_explicit_dependencies(tmp_path):
    from rendering.usd_mesh import write_mesh_usda

    usd = tmp_path / "object.usda"
    texture = tmp_path / "missing.png"
    write_mesh_usda(
        str(usd),
        object_name="object",
        points=[(0, 0, 0), (1, 0, 0), (0, 1, 0)],
        face_vertex_counts=[3],
        face_vertex_indices=[0, 1, 2],
        uvs=[(0, 0), (1, 0), (0, 1)],
        texture_file=texture.name,
    )

    def build(mode, texture_paths=(), renderer="blender"):
        cfg = render_config_from_mapping(
            {"material": {"object_material_mode": mode, "object_color_choices": [[1, 0, 0]]}},
            str(tmp_path),
        )
        return build_render_identity(
            cfg=cfg,
            catalog=AssetCatalog.scan(cfg),
            seed=1,
            renderer=renderer,
            renderer_version="test",
            usd_path=str(usd),
            texture_paths=texture_paths,
        )

    with pytest.raises(FileNotFoundError):
        build("preserve")
    first = build("override")
    # A shared texture is explicitly reported by Blender inspection and remains mandatory.
    with pytest.raises(FileNotFoundError):
        build("override", [str(texture)])
    with pytest.raises(FileNotFoundError):
        build("override", renderer="isaac")
    texture.write_bytes(b"unused original material")
    assert build("override") == first
    shared = build("override", [str(texture)])
    texture.write_bytes(b"changed shared material")
    assert build("override", [str(texture)]) != shared


def test_rotation_backend_change_rejects_resume(tmp_path, monkeypatch):
    from doper.scene import geometry

    root = setup_inputs(tmp_path / "inputs")
    original = identity(root)
    assert (
        original.to_dict()["config"]["rotation_sampler"]["algorithm"]
        == "scipy.Rotation.random/PCG64"
    )
    output = tmp_path / "output"
    save_attempt(output, original)
    monkeypatch.setattr(geometry, "scipy_version", "different-runtime")
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_resume_identity(output, identity(root))
