# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Blender integration coverage for portable USD exports."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
from conftest import REPO_ROOT
from conftest import run_blender as run_blender_process
from PIL import Image

from rendering.usd_mesh import read_mesh_usda, usd_texture_file

pytestmark = pytest.mark.blender


def run_blender(blender, *args):
    """Run a bounded Blender command and fail on Python errors.

    Args:
        blender: Blender executable path.
        args: Additional Blender arguments, including any script arguments.

    Returns:
        Captured stdout for diagnostic assertions.
    """
    result = run_blender_process(blender, *args)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_external_keypoints_reject_wrong_object_and_count(tmp_path, blender):
    script = tmp_path / "external_labels.py"
    script.write_text(f"""import bpy, json, sys
from pathlib import Path
sys.path.insert(0, {str(REPO_ROOT)!r})
from rendering.export_mesh_usd import collect_keypoints
bpy.data.objects['Cube'].name = 'part'
path = Path({str(tmp_path / "labels.json")!r})
valid = dict(object='part', keypoint_names=['second', 'first'], keypoints_3d=[[1,2,3],[4,5,6]])
for invalid, message in [(dict(valid, object='other'), 'does not match'),
                         (dict(valid, keypoint_names=['one']), 'mismatch')]:
    path.write_text(json.dumps(invalid))
    try:
        collect_keypoints('part', keypoints_json=str(path))
    except SystemExit as error:
        assert message in str(error), error
    else:
        raise AssertionError('Accepted incompatible labels')
for payload in (valid, {{k:v for k,v in valid.items() if k != 'object'}}):
    path.write_text(json.dumps(payload))
    _, names, points = collect_keypoints('part', keypoints_json=str(path))
    assert names == valid['keypoint_names'] and points == valid['keypoints_3d']
""")
    run_blender(blender, "--python", script)


def test_packed_texture_and_frame_survive_relocation(tmp_path, blender):
    source = tmp_path / "source"
    source.mkdir()
    texture = source / "texture.png"
    Image.new("RGB", (4, 4), (220, 30, 50)).save(texture)
    ply = source / "part.ply"
    ply.write_text("""ply
format ascii 1.0
comment TextureFile texture.png
element vertex 4
property float x
property float y
property float z
property float texture_u
property float texture_v
element face 2
property list uchar int vertex_indices
end_header
1 2 3 0 0
3 2 3 1 0
3 4 3 1 1
1 4 3 0 1
3 0 1 2
3 0 2 3
""")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    script = REPO_ROOT / "rendering/mesh_setup.py"
    run_blender(
        blender,
        "--python",
        script,
        "--",
        "--input",
        ply,
        "--output",
        bundle / "part.blend",
        "--mesh-name",
        "part",
        "--source-units",
        "cm",
        "--num-keypoints",
        "4",
        "--center-geometry",
        "--frame-transform-output",
        bundle / "frame.json",
        "--usd-output",
        bundle / "part.usda",
    )
    initial = read_mesh_usda(str(bundle / "part.usda"))
    labels = json.loads((bundle / "part_keypoints_3d.json").read_text())
    assert initial["keypoint_names"] == labels["keypoint_names"]
    np.testing.assert_allclose(initial["keypoints_3d"], labels["keypoints_3d"], atol=1e-6)
    np.testing.assert_allclose(initial["extent"], [[-0.01, -0.01, 0], [0.01, 0.01, 0]], atol=1e-6)
    transform = json.loads((bundle / "frame.json").read_text())
    np.testing.assert_allclose(
        np.array(transform["model_from_original"]) @ np.array(transform["original_from_model"]),
        np.eye(4),
        atol=1e-6,
    )
    # The original external file can diverge after Blender packs the correct pixels.
    Image.new("RGB", (4, 4), (0, 240, 0)).save(texture)
    relocated = tmp_path / "relocated"
    shutil.move(str(bundle), relocated)
    exported = relocated / "fresh"
    export = REPO_ROOT / "rendering/export_mesh_usd.py"
    run_blender(
        blender,
        relocated / "part.blend",
        "--python",
        export,
        "--",
        "--mesh-name",
        "part",
        "--output",
        exported / "part.usda",
        "--keypoints-output",
        exported / "labels.json",
    )
    with Image.open(usd_texture_file(str(exported / "part.usda"))) as image:
        assert image.convert("RGB").getpixel((0, 0)) == (220, 30, 50)
    assert read_mesh_usda(str(exported / "part.usda"))["keypoints_3d"] == initial["keypoints_3d"]
    exported_labels = json.loads((exported / "labels.json").read_text())
    assert {key: labels[key] for key in exported_labels} == exported_labels
    # A second source using the same texture basename must not recolor this USD.
    run_blender(
        blender,
        "--python",
        script,
        "--",
        "--input",
        ply,
        "--output",
        exported / "other.blend",
        "--mesh-name",
        "other",
        "--source-units",
        "cm",
        "--num-keypoints",
        "4",
        "--usd-output",
        exported / "other.usda",
    )
    with Image.open(usd_texture_file(str(exported / "part.usda"))) as image:
        assert image.convert("RGB").getpixel((0, 0)) == (220, 30, 50)
    with Image.open(usd_texture_file(str(exported / "other.usda"))) as image:
        assert image.convert("RGB").getpixel((0, 0)) == (0, 240, 0)
    shutil.rmtree(source)
    run_blender(
        blender,
        relocated / "part.blend",
        "--python",
        export,
        "--",
        "--mesh-name",
        "part",
        "--output",
        exported / "without_source.usda",
    )
    with Image.open(usd_texture_file(str(exported / "without_source.usda"))) as image:
        assert image.convert("RGB").getpixel((0, 0)) == (220, 30, 50)


def test_material_override_and_import_safe_cli(tmp_path, blender):
    source = tmp_path / "authored.glb"
    fixture = tmp_path / "create.py"
    fixture.write_text(f"""import bpy
bpy.data.objects['Cube'].data.materials.clear()
material = bpy.data.materials.new('Authored')
material.use_nodes = True
bpy.data.objects['Cube'].data.materials.append(material)
bpy.ops.export_scene.gltf(filepath={str(source)!r}, export_format='GLB')
""")
    run_blender(blender, "--python", fixture)
    run_blender(
        blender,
        "--python",
        REPO_ROOT / "rendering/mesh_setup.py",
        "--",
        "--input",
        source,
        "--output",
        tmp_path / "part.blend",
        "--mesh-name",
        "part",
        "--source-units",
        "m",
        "--num-keypoints",
        "4",
        "--no-keep-material",
    )
    check = tmp_path / "check.py"
    check.write_text(f"""import bpy
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
from rendering import export_mesh_usd
assert callable(export_mesh_usd.main)
assert [m.name for m in bpy.data.objects['part'].data.materials] == ['ObjectAluminum']
""")
    run_blender(blender, tmp_path / "part.blend", "--python", check)


def test_keep_material_preserves_authored_later_slot(tmp_path, blender):
    source = tmp_path / "slots.glb"
    fixture = tmp_path / "slots.py"
    fixture.write_text(f"""import bpy
obj = bpy.data.objects['Cube']
obj.data.materials.clear()
obj.data.materials.append(None)
material = bpy.data.materials.new('AuthoredLater')
material.use_nodes = True
obj.data.materials.append(material)
for polygon in obj.data.polygons:
    polygon.material_index = polygon.index % 2
bpy.ops.export_scene.gltf(filepath={str(source)!r}, export_format='GLB')
""")
    run_blender(blender, "--python", fixture)
    output = run_blender(
        blender,
        "--python",
        REPO_ROOT / "rendering/mesh_setup.py",
        "--",
        "--input",
        source,
        "--output",
        tmp_path / "part.blend",
        "--mesh-name",
        "part",
        "--source-units",
        "m",
        "--num-keypoints",
        "4",
    )
    assert "Keeping existing material: AuthoredLater" in output
    check = tmp_path / "check_slots.py"
    check.write_text(
        "import bpy\nmaterials = bpy.data.objects['part'].data.materials\n"
        "assert materials[0] is None\n"
        "assert any(m and m.name == 'AuthoredLater' for m in materials)\n"
    )
    run_blender(blender, tmp_path / "part.blend", "--python", check)
