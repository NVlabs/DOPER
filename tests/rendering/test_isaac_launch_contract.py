# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct workers cannot bypass launch provenance or mutate stale output."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from conftest import snapshot

from rendering.environment_bundle import (
    publish_bytes,
    publish_record,
    seal_bundle,
)
from rendering.isaac import isaac_parallel as launcher
from rendering.isaac.inputs import load_inputs
from rendering.isaac.launch_contract import validate_worker_launch
from rendering.shard_io import sha256_file
from rendering.usd_mesh import write_mesh_usda

RUNTIME = {"isaacsim": "6.0.1"}


@pytest.fixture
def active_launch(tmp_path, monkeypatch):
    usd = tmp_path / "object.usda"
    write_mesh_usda(
        str(usd),
        object_name="object",
        points=[(0, 0, 0), (0.1, 0, 0), (0, 0.1, 0)],
        face_vertex_counts=[3],
        face_vertex_indices=[0, 1, 2],
        keypoint_names=["KP_000"],
        keypoints_3d=[(0, 0, 0)],
    )
    texture = tmp_path / "assets/textures/cco_textures/test/albedo.png"
    texture.parent.mkdir(parents=True)
    texture.write_bytes(b"asset")
    config = tmp_path / "recipe.yaml"
    (tmp_path / "assets/hdri").mkdir(parents=True)
    config.write_text(
        yaml.safe_dump(
            {
                "object": {"usd_path": str(usd), "num_keypoints": 1},
                "scene": {
                    "num_distractors_min": 0,
                    "num_distractors_max": 0,
                    "hdri_dir": "assets/hdri",
                    "texture_dir": "assets/textures/cco_textures",
                },
            }
        )
    )
    out = tmp_path / "out"
    monkeypatch.setattr(launcher, "probe_runtime", lambda _: RUNTIME)
    monkeypatch.setattr(launcher, "run_workers", lambda *args, **kwargs: None)
    monkeypatch.setattr("rendering.run_manifest.code_revision", lambda _: "test")
    launcher.main(
        [
            "--num-images",
            "2",
            "--config",
            str(config),
            "--isaac-python",
            "isaac-python",
            "--outdir",
            str(out),
            "--no-merge",
        ]
    )
    return SimpleNamespace(
        config=str(config),
        usd=None,
        keypoints_json=None,
        start_idx=0,
        end_idx=2,
        seed=42,
        worker_id=0,
        gpu_device=0,
        scenes=str(out / "scenes/scenes_00000000_00000002.jsonl"),
        outdir=str(out),
    )


def config_snapshot(args):
    return snapshot(Path(args.config).parent)


def edit_manifest(args, edit):
    path = Path(args.outdir) / "manifests/shard_00000000_00000002.json"
    manifest = json.loads(path.read_text())
    edit(manifest)
    path.write_text(json.dumps(manifest))


def test_active_declared_worker_passes_read_only(active_launch):
    inputs = load_inputs(active_launch)
    before = config_snapshot(active_launch)
    settings = validate_worker_launch(inputs, runtime=RUNTIME)
    assert settings["runtime"] == RUNTIME
    assert config_snapshot(active_launch) == before


@pytest.mark.parametrize(("field", "value"), [("worker_id", 4), ("gpu_device", 1), ("end_idx", 1)])
def test_direct_changed_worker_declaration_rejected(active_launch, field, value):
    setattr(active_launch, field, value)
    inputs = load_inputs(active_launch)
    before = config_snapshot(active_launch)
    with pytest.raises(ValueError, match="launcher"):
        validate_worker_launch(inputs, runtime=RUNTIME)
    assert config_snapshot(active_launch) == before


@pytest.mark.parametrize("status", ["complete", "failed"])
def test_direct_resume_requires_new_running_attempt(active_launch, status):
    edit_manifest(active_launch, lambda manifest: manifest.update(status=status))
    with pytest.raises(ValueError, match="active"):
        validate_worker_launch(load_inputs(active_launch), runtime=RUNTIME)


def test_missing_manifest_rejects_direct_worker(active_launch):
    (Path(active_launch.outdir) / "manifests/shard_00000000_00000002.json").unlink()
    with pytest.raises(ValueError, match="launcher"):
        validate_worker_launch(load_inputs(active_launch), runtime=RUNTIME)


def test_rewritten_valid_scene_rejected_by_digest(active_launch):
    path = Path(active_launch.scenes)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["object"]["center"][0] += 1
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    inputs = load_inputs(active_launch)
    before = config_snapshot(active_launch)
    with pytest.raises(ValueError, match="scene file"):
        validate_worker_launch(inputs, runtime=RUNTIME)
    assert config_snapshot(active_launch) == before


@pytest.mark.parametrize("changed", ["object", "config"])
def test_changed_authored_inputs_fail_identity_before_writes(active_launch, changed):
    root = Path(active_launch.config).parent
    if changed == "object":
        with (root / "object.usda").open("a") as stream:
            stream.write("\n# changed object bytes\n")
    else:
        path = Path(active_launch.config)
        config = yaml.safe_load(path.read_text())
        config["scene"]["ground_style"] = "solid"
        path.write_text(yaml.safe_dump(config))
    inputs = load_inputs(active_launch)
    before = config_snapshot(active_launch)
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_worker_launch(inputs, runtime=RUNTIME)
    assert config_snapshot(active_launch) == before


def test_runtime_mismatch_rejected(active_launch):
    with pytest.raises(ValueError, match="runtime"):
        validate_worker_launch(load_inputs(active_launch), runtime={"isaacsim": "other"})


def test_ambiguous_worker_id_rejected(active_launch):
    edit_manifest(
        active_launch,
        lambda manifest: manifest["shard"]["workers"].append(dict(manifest["shard"]["workers"][0])),
    )
    with pytest.raises(ValueError, match="unambiguous"):
        validate_worker_launch(load_inputs(active_launch), runtime=RUNTIME)


def test_worker_reuses_asset_content_manifest_without_hashing_packs(active_launch, monkeypatch):
    def forbidden_scan(*args, **kwargs):
        raise AssertionError("workers must not rescan asset pack contents")

    monkeypatch.setattr("rendering.run_manifest.local_asset_manifest", forbidden_scan)
    inputs = load_inputs(active_launch)
    before = config_snapshot(active_launch)
    assert validate_worker_launch(inputs, runtime=RUNTIME)
    assert config_snapshot(active_launch) == before


def test_changed_catalog_rejected_without_hashing_packs(active_launch, monkeypatch):
    root = Path(active_launch.config).parent
    hdri = root / "assets/hdri/added.hdr"
    hdri.parent.mkdir(parents=True, exist_ok=True)
    hdri.write_bytes(b"new catalog member")
    monkeypatch.setattr(
        "rendering.run_manifest.local_asset_manifest",
        lambda *args: pytest.fail("worker scanned pack content"),
    )
    with pytest.raises(ValueError, match="asset catalog differs"):
        validate_worker_launch(load_inputs(active_launch), runtime=RUNTIME)


def test_missing_persisted_assets_cannot_fall_back_to_worker_scan(active_launch, monkeypatch):
    edit_manifest(active_launch, lambda manifest: manifest["render_identity"].pop("assets"))
    monkeypatch.setattr(
        "rendering.run_manifest.local_asset_manifest",
        lambda *args: pytest.fail("worker scanned pack content"),
    )
    with pytest.raises(ValueError, match="persisted asset manifest"):
        validate_worker_launch(load_inputs(active_launch), runtime=RUNTIME)


def test_environment_identity_is_bound_to_launch_and_resume(active_launch, monkeypatch, tmp_path):

    records = [json.loads(line) for line in Path(active_launch.scenes).read_text().splitlines()]
    root = tmp_path / "environments"
    source = {
        "identity_sha256": "a" * 64,
        "assets_sha256": json.loads(
            (Path(active_launch.outdir) / "manifests/shard_00000000_00000002.json").read_text()
        )["render_identity"]["assets"]["content_sha256"],
        "usd_sha256": sha256_file(tmp_path / "object.usda"),
        "blender_version": "4.2.1",
    }

    data = b"#usda 1.0\n"
    layer = "layers/" + hashlib.sha256(data).hexdigest() + ".usda"
    publish_bytes(root / layer, data)
    for record in records:
        publish_record(
            root,
            record,
            source=source,
            layer=layer,
            dependencies=[],
            light_count=1,
            ground=False,
        )
    seal_bundle(root, records, source=source)
    runtime = {"isaacsim": "6.0.1-test"}
    monkeypatch.setattr(launcher, "probe_runtime", lambda _: runtime)
    out = tmp_path / "native-out"
    launcher.main(
        [
            "--num-images",
            "2",
            "--config",
            active_launch.config,
            "--isaac-python",
            "isaac-python",
            "--outdir",
            str(out),
            "--no-merge",
            "--environment-root",
            str(root),
            "--environment-gain",
            "2000",
            "--environment-adapter",
            "blender42-isaac601",
        ]
    )
    active_launch.outdir = str(out)
    active_launch.scenes = str(out / "scenes/scenes_00000000_00000002.jsonl")
    active_launch.environment_root = str(root)
    active_launch.environment_gain = 2000
    active_launch.environment_adapter = "blender42-isaac601"
    expected = validate_worker_launch(load_inputs(active_launch), runtime=runtime)
    assert expected["environment"]["gain"] == 2000
    before = config_snapshot(active_launch)
    active_launch.environment_gain = 1000
    with pytest.raises(ValueError, match="settings"):
        validate_worker_launch(load_inputs(active_launch), runtime=runtime)
    assert config_snapshot(active_launch) == before
