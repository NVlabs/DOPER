# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded CPU-only checks against real Blender authored state."""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from rendering.blender_inputs import inspect_blender_inputs
from rendering.shard_io import sha256_file

pytestmark = pytest.mark.blender


def create_blend(blender, tmp_path, *, parent=True, packed=False, shared=None):
    blend = tmp_path / "object.blend"
    texture = tmp_path / "texture.png"
    expr = f"""
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add()
mesh = bpy.context.object
mesh.name = "object"
kp = bpy.data.objects.new("KP_000", None)
bpy.context.collection.objects.link(kp)
if {parent!r}:
    kp.parent = mesh
kp.location = (0.2, 0.3, 0.4)
image = bpy.data.images.new("authored", width=1, height=1)
image.filepath_raw = {str(texture)!r}
image.file_format = "PNG"
image.save()
image.source = "FILE"
material = bpy.data.materials.new("material")
material.use_nodes = True
node = material.node_tree.nodes.new("ShaderNodeTexImage")
node.image = image
mesh.data.materials.append(material)
if {shared!r} == "mesh":
    other = bpy.data.objects.new("other", mesh.data)
    bpy.context.collection.objects.link(other)
if {shared!r} in ("world", "group", "target_group"):
    if {shared!r} == "world":
        world = bpy.data.worlds.new("world")
        bpy.context.scene.world = world
        world.use_nodes = True
        other_node = world.node_tree.nodes.new("ShaderNodeTexEnvironment")
        other_node.image = image
    else:
        material.node_tree.nodes.remove(node)
        group = bpy.data.node_groups.new("texture_group", "ShaderNodeTree")
        group.nodes.new("ShaderNodeTexImage").image = image
        material.node_tree.nodes.new("ShaderNodeGroup").node_tree = group
        if {shared!r} == "group":
            world = bpy.data.worlds.new("world")
            bpy.context.scene.world = world
            world.use_nodes = True
            world.node_tree.nodes.new("ShaderNodeGroup").node_tree = group
if {packed!r}:
    image.pack()
image.filepath_raw = "//texture.png"
bpy.ops.wm.save_as_mainfile(filepath={str(blend)!r})
"""
    result = subprocess.run(
        [
            blender,
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python-expr",
            expr,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return blend, texture


@pytest.fixture(scope="module")
def make_blend(blender, tmp_path_factory):
    """Factory copying a Blender-built ``object.blend`` variant into a test's directory.

    Each variant is built once per module; the image path is stored relative to the blend, so
    a copy owns its texture and tests may modify or delete it.
    """
    built: dict[tuple, Path] = {}

    def make(tmp_path, **variant):
        key = tuple(sorted(variant.items()))
        if key not in built:
            root = tmp_path_factory.mktemp("blend")
            create_blend(blender, root, **variant)
            built[key] = root
        shutil.copytree(built[key], tmp_path, dirs_exist_ok=True)
        return tmp_path / "object.blend", tmp_path / "texture.png"

    return make


def test_external_texture_bytes_and_mesh_keypoints_are_inspected(blender, make_blend, tmp_path):
    blend, texture = make_blend(tmp_path)
    before = sha256_file(blend)
    inputs = inspect_blender_inputs(blender, str(blend), "object", "KP_", 1)
    assert inputs["textures"] == [str(texture)]
    assert inputs["external_files"] == [str(texture)]
    assert len(inputs["points"]) == 8
    assert inputs["keypoint_names"] == ["KP_000"]
    assert inputs["keypoints_3d"][0] == pytest.approx([0.2, 0.3, 0.4])
    original_texture_hash = sha256_file(inputs["textures"][0])
    texture.write_bytes(b"same path different content")
    after = inspect_blender_inputs(blender, str(blend), "object", "KP_", 1)
    assert sha256_file(after["textures"][0]) != original_texture_hash
    assert sha256_file(blend) == before


def test_packed_images_need_no_external_file(blender, make_blend, tmp_path):
    blend, texture = make_blend(tmp_path, packed=True)
    texture.unlink()
    assert inspect_blender_inputs(blender, str(blend), "object", "KP_", 1)["external_files"] == []


def test_missing_texture_and_wrong_keypoint_count_fail_closed(blender, make_blend, tmp_path):
    blend, texture = make_blend(tmp_path)
    with pytest.raises(ValueError, match="names/count"):
        inspect_blender_inputs(blender, str(blend), "object", "KP_", 2)
    texture.unlink()
    with pytest.raises(ValueError, match="missing external texture"):
        inspect_blender_inputs(blender, str(blend), "object", "KP_", 1)


def test_keypoints_must_be_parented_to_mesh(blender, make_blend, tmp_path):
    blend, _ = make_blend(tmp_path, parent=False)
    with pytest.raises(ValueError, match="parented"):
        inspect_blender_inputs(blender, str(blend), "object", "KP_", 1)


def test_inspection_timeout_is_bounded(blender, tmp_path, monkeypatch):
    blend = tmp_path / "object.blend"
    blend.write_bytes(b"placeholder")

    def timeout_run(argv, **kwargs):
        assert kwargs["timeout"] == 0.1
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr("rendering.blender_inputs.subprocess.run", timeout_run)
    with pytest.raises(ValueError, match="timed out"):
        inspect_blender_inputs(blender, str(blend), "object", "KP_", 1, timeout=0.1)


@pytest.mark.parametrize("shared", [None, "target_group"])
def test_override_ignores_missing_target_only_texture(blender, make_blend, tmp_path, shared):
    blend, texture = make_blend(tmp_path, shared=shared)
    texture.unlink()
    with pytest.raises(ValueError, match="missing external texture"):
        inspect_blender_inputs(blender, str(blend), "object", "KP_", 1)
    assert (
        inspect_blender_inputs(
            blender, str(blend), "object", "KP_", 1, object_material_mode="override"
        )["external_files"]
        == []
    )


@pytest.mark.parametrize("shared", ["mesh", "world", "group"])
def test_override_still_checks_shared_textures(blender, make_blend, tmp_path, shared):
    blend, texture = make_blend(tmp_path, shared=shared)
    texture.unlink()
    with pytest.raises(ValueError, match="missing external texture"):
        inspect_blender_inputs(
            blender, str(blend), "object", "KP_", 1, object_material_mode="override"
        )


def test_temporary_directory_prefix(blender, make_blend, tmp_path, monkeypatch):
    blend, _ = make_blend(tmp_path)
    prefixes = []
    real = tempfile.TemporaryDirectory
    monkeypatch.setattr(
        "rendering.blender_inputs.tempfile.TemporaryDirectory",
        lambda prefix=None, **kw: prefixes.append(prefix) or real(prefix=prefix, **kw),
    )
    inspect_blender_inputs(blender, str(blend), "object", "KP_", 1)
    assert prefixes == ["doper-blender-inputs-"]


def test_export_mesh_usd_writes_textured_layer_and_keypoints(blender, make_blend, tmp_path):
    blend, texture = make_blend(tmp_path)
    out = tmp_path / "out" / "object.usda"
    keypoints = tmp_path / "out" / "keypoints_3d.json"
    script = Path(__file__).resolve().parents[2] / "rendering" / "export_mesh_usd.py"
    cmd = [
        blender,
        "--background",
        "--factory-startup",
        "--python-exit-code",
        "1",
        str(blend),
        "--python",
        str(script),
        "--",
        "--mesh-name",
        "object",
        "--keypoint-prefix",
        "KP_",
        "--output",
        str(out),
        "--keypoints-output",
        str(keypoints),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (out.parent / "object.texture.png").read_bytes() == texture.read_bytes()
    text = out.read_text()
    assert "KP_000" in text and "object.texture.png" in text
    assert json.loads(keypoints.read_text())["keypoint_names"] == ["KP_000"]
