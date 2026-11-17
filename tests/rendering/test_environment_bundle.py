# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable native environments fail closed on identity or dependency changes."""

import hashlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from rendering import run_manifest
from rendering.config import render_config_from_mapping
from rendering.environment_bundle import (
    BundleReader,
    ReplayInputs,
    canonical_hash,
    publish_bytes,
    publish_record,
    replay_options,
    seal_bundle,
    seal_dataset,
    source_from_manifest,
    validate_applied_readbacks,
)
from rendering.run_manifest import local_asset_manifest
from rendering.scene_spec import AssetCatalog
from rendering.shard_io import sha256_file


def test_relocation_and_tampered_texture(tmp_path, native_bundle):
    root, record, source = native_bundle
    moved = root.rename(tmp_path / "relocated")
    bundle = BundleReader(moved, usd_sha256=source["usd_sha256"], seed=42)
    assert bundle.verify_record(record)["light_count"] == 2
    (moved / "textures/asset.hdr").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        BundleReader(moved, usd_sha256=source["usd_sha256"], seed=42).verify_record(record)


def test_immutable_publication(tmp_path):
    path = tmp_path / "record.json"
    publish_bytes(path, b"one")
    publish_bytes(path, b"one")
    with pytest.raises(ValueError, match="existing"):
        publish_bytes(path, b"two")
    assert path.read_bytes() == b"one"


@pytest.mark.parametrize("change", ["scene", "seed", "object", "missing", "metadata"])
def test_mismatch_rejected(tmp_path, native_bundle, change):
    root, record, source = native_bundle
    kwargs = {"usd_sha256": source["usd_sha256"], "seed": 42}
    if change == "scene":
        record["background"]["rotation_deg"] = 30
    if change == "seed":
        kwargs["seed"] = 43
    if change == "object":
        kwargs["usd_sha256"] = "d" * 64
    if change == "missing":
        next((root / "layers").glob("*.usda")).unlink()
    if change == "metadata":
        (root / "records/000000.json").write_text("{}")
    with pytest.raises((ValueError, FileNotFoundError)):
        BundleReader(root, **kwargs).verify_record(record)


def test_seal_rejects_missing_or_mixed_source(tmp_path, native_bundle):
    root, record, source = native_bundle
    (root / "manifest.json").unlink()
    with pytest.raises(ValueError, match="source"):
        seal_bundle(root, [record], source={**source, "identity_sha256": "d" * 64})
    with pytest.raises(ValueError, match="missing"):
        seal_bundle(root, [record, {**record, "index": 1}], source=source)
    assert not (root / "manifest.json").exists()


@pytest.mark.parametrize("path", ["../escape.hdr", "/tmp/escape.hdr", "textures/../escape.hdr"])
def test_dependency_paths_cannot_escape(tmp_path, native_bundle, path):
    root, record, source = native_bundle
    with pytest.raises(ValueError, match="path"):
        publish_record(
            root,
            record,
            source=source,
            layer="layers/" + hashlib.sha256(b"#usda 1.0\n").hexdigest() + ".usda",
            dependencies=[path],
            light_count=2,
            ground=False,
        )


def test_dependency_symlink_cannot_escape(tmp_path, native_bundle):
    root, record, source = native_bundle
    outside = tmp_path / "outside.hdr"
    outside.write_bytes(b"other")
    (root / "textures/link.hdr").symlink_to(outside)
    with pytest.raises(ValueError, match="path"):
        publish_record(
            root,
            record,
            source=source,
            layer="layers/" + hashlib.sha256(b"#usda 1.0\n").hexdigest() + ".usda",
            dependencies=["textures/link.hdr"],
            light_count=2,
            ground=False,
        )


def test_canonical_scene_hash_ignores_key_order():
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})


@pytest.mark.parametrize("gain", [None, True, 0, -1, float("nan"), float("inf")])
def test_replay_gain_requires_explicit_positive_finite_value(gain):
    with pytest.raises(ValueError, match="gain"):
        replay_options("bundle", gain, "blender42-isaac601")


def test_replay_options_defaults_and_adapter():
    assert replay_options(None, None, None) is None
    with pytest.raises(ValueError):
        replay_options(None, 2000, None)
    with pytest.raises(ValueError):
        replay_options("bundle", 2000, "unknown")
    assert replay_options("bundle", 2000, "blender42-isaac601")["gain"] == 2000


def test_native_usd_packaging_uses_deduplicated_portable_assets(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Sdf

    from rendering.environment_bundle_usd import package_layer

    root = tmp_path / "bundle"
    texture = tmp_path / "source.hdr"
    texture.write_bytes(b"pixels")
    layer = Sdf.Layer.CreateNew(str(tmp_path / "native.usda"))
    prim = Sdf.CreatePrimInLayer(layer, "/BlenderEnvironment")
    prim.specifier = Sdf.SpecifierDef
    prim.typeName = "Xform"
    layer.defaultPrim = "BlenderEnvironment"
    a = Sdf.AttributeSpec(prim, "inputs:texture:file", Sdf.ValueTypeNames.Asset)
    a.default = Sdf.AssetPath(str(texture))
    layer.Save()
    packaged, deps = package_layer(layer.realPath, root)
    package_layer(layer.realPath, root)
    assert len(list((root / "textures").iterdir())) == 1
    assert len(deps) == 1
    copied = Sdf.Layer.FindOrOpen(str(root / packaged))
    name = copied.GetAttributeAtPath("/BlenderEnvironment.inputs:texture:file").default.path
    assert name.startswith("../textures/")
    assert (root / "layers" / name).read_bytes() == b"pixels"


def test_native_usd_packaging_rejects_layer_dependencies(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Sdf

    from rendering.environment_bundle_usd import package_layer

    nested = Sdf.Layer.CreateNew(str(tmp_path / "nested.usda"))
    nested.Save()
    layer = Sdf.Layer.CreateNew(str(tmp_path / "native.usda"))
    layer.subLayerPaths.append(str(nested.realPath))
    layer.Save()
    with pytest.raises(ValueError, match="nested"):
        package_layer(layer.realPath, tmp_path / "bundle")


def test_uncommitted_layer_does_not_block_retry(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Sdf

    from rendering.environment_bundle_usd import (
        package_layer,
        validate_layer_dependencies,
    )

    layer = Sdf.Layer.CreateNew(str(tmp_path / "native.usda"))
    prim = Sdf.CreatePrimInLayer(layer, "/BlenderEnvironment")
    prim.specifier = Sdf.SpecifierDef
    prim.typeName = "Xform"
    layer.defaultPrim = "BlenderEnvironment"
    layer.Save()
    root = tmp_path / "bundle"
    orphan, _ = package_layer(layer.realPath, root)
    # Interruption before metadata publication; retry changes native datablock data.
    Sdf.AttributeSpec(prim, "diagnostic", Sdf.ValueTypeNames.String).default = "retry"
    layer.Save()
    committed, deps = package_layer(layer.realPath, root)
    assert committed != orphan
    record = {"index": 0, "seed": 42}
    source = {
        "identity_sha256": "a" * 64,
        "usd_sha256": "b" * 64,
        "assets_sha256": "c" * 64,
        "blender_version": "4.2.1",
    }
    metadata = publish_record(
        root,
        record,
        source=source,
        layer=committed,
        dependencies=deps,
        light_count=1,
        ground=False,
    )
    seal_bundle(root, [record], source=source)
    assert (
        BundleReader(root, usd_sha256="b" * 64, seed=42).verify_record(record)["layer"] == committed
    )
    validate_layer_dependencies(root, metadata)


def test_native_dependency_inventory_must_be_complete(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Sdf

    from rendering.environment_bundle_usd import (
        package_layer,
        validate_layer_dependencies,
    )

    texture = tmp_path / "source.hdr"
    texture.write_bytes(b"pixels")
    layer = Sdf.Layer.CreateNew(str(tmp_path / "native.usda"))
    prim = Sdf.CreatePrimInLayer(layer, "/BlenderEnvironment")
    prim.specifier = Sdf.SpecifierDef
    prim.typeName = "Xform"
    layer.defaultPrim = "BlenderEnvironment"
    Sdf.AttributeSpec(
        prim, "inputs:texture:file", Sdf.ValueTypeNames.Asset
    ).default = Sdf.AssetPath(str(texture))
    layer.Save()
    root = tmp_path / "bundle"
    name, deps = package_layer(layer.realPath, root)
    with pytest.raises(ValueError, match="undeclared"):
        validate_layer_dependencies(root, {"layer": name, "files": {name: "unused"}})


def complete_environment_dataset(tmp_path):
    dataset = tmp_path / "dataset"
    root = dataset / "environments"
    record = {"index": 0, "seed": 42}
    identity = {
        "object": {"usd_sha256": "b" * 64},
        "assets": {"content_sha256": "c" * 64},
        "config": {
            "renderer_settings": {
                "blender_version": "Blender 4.2.1 LTS",
                "export_environment": 1,
            }
        },
    }
    manifest = {
        "render_identity": identity,
        "renderer": "blender",
        "status": "complete",
        "shard": {"start_idx": 0, "end_idx": 1},
        "kind": "render_shard",
    }
    source = source_from_manifest(manifest, "4.2.1 LTS")
    data = b"#usda 1.0\n"
    layer = "layers/" + hashlib.sha256(data).hexdigest() + ".usda"
    publish_bytes(root / layer, data)
    publish_record(
        root,
        record,
        source=source,
        layer=layer,
        dependencies=[],
        light_count=1,
        ground=False,
    )
    scene_path = dataset / "scenes/scenes.jsonl"
    publish_bytes(scene_path, json.dumps(record).encode() + b"\n")
    manifest["scenes"] = {"file": scene_path.name, "sha256": sha256_file(scene_path)}
    publish_bytes(
        dataset / "manifests/shard_00000000_00000001.json",
        json.dumps(manifest).encode(),
    )
    return dataset


def test_seal_preserves_exact_blender_runtime_label(tmp_path):
    dataset = complete_environment_dataset(tmp_path)
    assert seal_dataset(dataset)["source"]["blender_version"] == "4.2.1 LTS"


def test_replay_applies_gain_once_and_detects_changed_live_lights(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom, UsdLux

    from rendering.environment_bundle_usd import package_layer
    from rendering.isaac.environment_replay import EnvironmentReplay

    native = Usd.Stage.CreateNew(str(tmp_path / "native.usda"))
    root_prim = UsdGeom.Xform.Define(native, "/BlenderEnvironment").GetPrim()
    native.SetDefaultPrim(root_prim)
    dome = UsdLux.DomeLight.Define(native, "/BlenderEnvironment/World")
    dome.CreateIntensityAttr(1.5)
    UsdGeom.Xformable(dome).AddRotateXYZOp().Set(Gf.Vec3f(90, 0, 0))
    native.GetRootLayer().Save()
    root = tmp_path / "bundle"
    layer, deps = package_layer(native.GetRootLayer().realPath, root)
    record = {
        "index": 0,
        "seed": 42,
        "background": {"mode": "hdri", "rotation_deg": 20},
    }
    source = {
        "identity_sha256": "a" * 64,
        "usd_sha256": "b" * 64,
        "assets_sha256": "c" * 64,
        "blender_version": "4.2.1",
    }
    publish_record(
        root,
        record,
        source=source,
        layer=layer,
        dependencies=deps,
        light_count=1,
        ground=False,
    )
    seal_bundle(root, [record], source=source)
    reader = BundleReader(root, usd_sha256="b" * 64, seed=42)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    replay = EnvironmentReplay(
        stage, ReplayInputs(reader, {"adapter": "blender42-isaac601", "gain": 2000})
    )
    for _ in range(2):
        replay.apply(record)
        snapshot = replay.verify(record)
        assert snapshot["lights"][0]["intensity"] == 3000
        assert tuple(
            stage.GetPrimAtPath("/World/BlenderEnvironment/World")
            .GetAttribute("xformOp:rotateXYZ")
            .Get()
        ) == (0, 0, 160)
    stage.GetPrimAtPath("/World/BlenderEnvironment/World").GetAttribute("inputs:intensity").Set(1)
    with pytest.raises(RuntimeError, match="after capture"):
        replay.verify(record)


def test_readbacks_required_for_each_stereo_view_and_identity(tmp_path):
    cfg = SimpleNamespace(
        stereo_enabled=True, stereo_left_suffix="left", stereo_right_suffix="right"
    )
    record = {"index": 0, "seed": 42}
    identity = {"bundle_sha256": "a" * 64, "gain": 2000}
    payload = {
        **identity,
        "scene_sha256": canonical_hash(record),
        "lights": [valid_light()],
    }
    folder = tmp_path / "environment_applied"
    folder.mkdir()
    (folder / "000000_left.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="right"):
        validate_applied_readbacks(tmp_path, [record], cfg, identity)
    (folder / "000000_right.json").write_text(json.dumps(payload))
    validate_applied_readbacks(tmp_path, [record], cfg, identity)
    with pytest.raises(ValueError, match="identity"):
        validate_applied_readbacks(tmp_path, [record], cfg, {**identity, "gain": 1000})


def valid_light():
    return {
        "path": "/World/BlenderEnvironment/dome",
        "type": "DomeLight",
        "intensity": 2000,
        "exposure": 0,
        "world_matrix": [[int(i == j) for j in range(4)] for i in range(4)],
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(lights=[{}]),
        lambda p: p["lights"][0].update(intensity=float("nan")),
        lambda p: p["lights"][0].update(world_matrix=[[1, 2, 3, 4]]),
        lambda p: p["lights"][0].update(exposure=True),
        lambda p: p.update(lights=p["lights"] * 2),
        lambda p: p["lights"][0].update(type="Mesh"),
    ],
)
def test_readbacks_reject_malformed_live_lights(tmp_path, mutation):
    record = {"index": 0, "seed": 42}
    identity = {"bundle_sha256": "a" * 64, "gain": 2000}
    payload = {
        **identity,
        "scene_sha256": canonical_hash(record),
        "lights": [valid_light()],
    }
    mutation(payload)
    folder = tmp_path / "environment_applied"
    folder.mkdir()
    (folder / "000000.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="light"):
        validate_applied_readbacks(
            tmp_path, [record], SimpleNamespace(stereo_enabled=False), identity
        )


def test_readbacks_check_sealed_light_inventory(tmp_path, native_bundle):
    root, record, source = native_bundle
    bundle = BundleReader(root, usd_sha256=source["usd_sha256"], seed=42)
    identity = {"bundle_sha256": bundle.sha256, "gain": 2000}
    folder = tmp_path / "environment_applied"
    folder.mkdir()
    (folder / "000000.json").write_text(
        json.dumps(
            {
                **identity,
                "scene_sha256": canonical_hash(record),
                "lights": [valid_light()],
            }
        )
    )
    with pytest.raises(ValueError, match="inventory"):
        validate_applied_readbacks(
            tmp_path,
            [record],
            SimpleNamespace(stereo_enabled=False),
            identity,
            bundle=bundle,
        )


@pytest.mark.parametrize("default_prim", ["", "WrongRoot"])
def test_packaging_rejects_wrong_default_prim_before_publication(tmp_path, default_prim):
    pytest.importorskip("pxr")
    from pxr import Sdf

    from rendering.environment_bundle_usd import package_layer

    source = Sdf.Layer.CreateNew(str(tmp_path / "wrong.usda"))
    prim = Sdf.CreatePrimInLayer(source, "/BlenderEnvironment")
    prim.specifier = Sdf.SpecifierDef
    source.defaultPrim = default_prim
    source.Save()
    root = tmp_path / "bundle"
    with pytest.raises(ValueError, match="BlenderEnvironment"):
        package_layer(source.realPath, root)
    assert not root.exists()


@pytest.mark.parametrize("mutation", ["gap", "starts_after_zero", "incomplete", "mixed_source"])
def test_sealing_rejects_invalid_shard_sets(tmp_path, mutation):
    # Reuse the complete on-disk fixture, then invalidate only the shard contract.
    dataset = complete_environment_dataset(tmp_path)
    sealed = dataset / "environments/manifest.json"
    path = dataset / "manifests/shard_00000000_00000001.json"
    shard = json.loads(path.read_text())
    if mutation == "starts_after_zero":
        shard["shard"]["start_idx"] = 1
    elif mutation == "incomplete":
        shard["status"] = "running"
    else:
        second = json.loads(json.dumps(shard))
        second["shard"] = {"start_idx": 2 if mutation == "gap" else 1, "end_idx": 3}
        if mutation == "mixed_source":
            second["render_identity"]["assets"]["content_sha256"] = "d" * 64
        (path.parent / "shard_00000001_00000003.json").write_text(json.dumps(second))
    path.write_text(json.dumps(shard))
    with pytest.raises(ValueError, match="contiguous|different source"):
        seal_dataset(dataset)
    assert not sealed.exists()


def test_replay_asset_digest_is_portable_but_detects_changed_bytes(tmp_path):
    digests = []
    for name in ("blender", "isaac"):
        root = tmp_path / name
        texture = root / "assets/textures/cco_textures/test/albedo.png"
        texture.parent.mkdir(parents=True)
        texture.write_bytes(b"original texture")
        cfg = render_config_from_mapping(
            {"scene": {"texture_dir": "assets/textures/cco_textures"}}, str(root)
        )
        digest = local_asset_manifest(cfg, AssetCatalog.scan(cfg))["content_sha256"]
        digests.append(digest)
    replay = ReplayInputs(SimpleNamespace(source={"assets_sha256": digests[0]}), {})
    replay.validate_assets(digests[1])
    texture.write_bytes(b"changed texture at the same relative path")
    changed = local_asset_manifest(cfg, AssetCatalog.scan(cfg))["content_sha256"]
    with pytest.raises(ValueError, match="asset content"):
        replay.validate_assets(changed)


def test_distractor_texture_library_uses_size_identity_without_reading(tmp_path, monkeypatch):
    texture = tmp_path / "assets/distractors/textures/wood/albedo.png"
    texture.parent.mkdir(parents=True)
    texture.write_bytes(b"abcd")
    mesh = tmp_path / "assets/distractors/chair.usdc"
    mesh.write_bytes(b"mesh")
    cfg = render_config_from_mapping(
        {"scene": {"distractor_dir": "assets/distractors"}}, str(tmp_path)
    )
    hashed = []
    original = run_manifest.sha256_file
    monkeypatch.setattr(
        run_manifest,
        "sha256_file",
        lambda path, *a: hashed.append(os.path.basename(path)) or original(path, *a),
    )
    manifest = local_asset_manifest(cfg, AssetCatalog.scan(cfg))
    assert manifest["files"]["distractors/textures/wood/albedo.png"] == "size:4"
    assert hashed == ["chair.usdc"]
    texture.write_bytes(b"abcde")
    assert (
        local_asset_manifest(cfg, AssetCatalog.scan(cfg))["content_sha256"]
        != manifest["content_sha256"]
    )


def test_environment_bundle_imports_without_scipy():
    code = (
        "import sys\n"
        "for name in ('scipy', 'torch', 'cv2', 'pxr', 'bpy'):\n"
        "    sys.modules[name] = None\n"
        "import rendering.environment_bundle, rendering.shard_io\n"
        "assert 'rendering.run_manifest' not in sys.modules\n"
        "assert 'doper.scene.geometry' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_usd_export_without_pxr_names_extra(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pxr", None)
    from rendering.environment_bundle_usd import package_layer, validate_layer_dependencies

    with pytest.raises(ImportError, match=r"usd-core \(doper\[usd\]\)"):
        package_layer(tmp_path / "native.usda", tmp_path / "bundle")
    with pytest.raises(ImportError, match="usd-core"):
        validate_layer_dependencies(tmp_path, {"layer": "layers/x.usda", "files": {}})
    assert not (tmp_path / "bundle").exists()
