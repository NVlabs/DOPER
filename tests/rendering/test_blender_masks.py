# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual Blender regressions for mask passes over linked mesh instances."""

import subprocess

import pytest
from conftest import REPO_ROOT
from PIL import Image

pytestmark = pytest.mark.blender


@pytest.mark.parametrize("with_slots", [False, True])
@pytest.mark.parametrize("render_fails", [False, True])
def test_linked_mesh_mask_is_object_specific_and_restores_state(
    blender, tmp_path, with_slots, render_fails
):
    script = tmp_path / "mask.py"
    script.write_text("""
import sys
from pathlib import Path
from types import SimpleNamespace
import bpy
sys.path.insert(0, str(Path.cwd()))
from rendering.blender import annotations
from rendering.blender.context import BlenderContext
root, with_slots, render_fails = Path(sys.argv[-3]), sys.argv[-2] == 'True', sys.argv[-1] == 'True'
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
bpy.ops.mesh.primitive_cube_add(size=.8, location=(-.6, 0, 0))
target = bpy.context.object
target.name = 'A_target'
shared = target.data
background = target.copy()
background.data = shared
background.name = 'B_background'
background.location.x = .6
bpy.context.collection.objects.link(background)
if with_slots:
    red = bpy.data.materials.new('OriginalRed')
    blue = bpy.data.materials.new('OriginalBlue')
    override = bpy.data.materials.new('ObjectOverride')
    shared.materials.append(red)
    shared.materials.append(blue)
    for i, polygon in enumerate(shared.polygons):
        polygon.material_index = i % 2
    target.material_slots[0].link = 'OBJECT'
    target.material_slots[0].material = override
    background.material_slots[1].link = 'OBJECT'
    background.material_slots[1].material = override
original_materials = list(shared.materials)
original_indices = [p.material_index for p in shared.polygons]
original_slots = {o.name: [(s.link, s.material) for s in o.material_slots] for o in (target, background)}
original_meshes = set(bpy.data.meshes.keys())
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 1
scene.cycles.device = 'CPU'
scene.render.resolution_x = 64
scene.render.resolution_y = 32
scene.render.resolution_percentage = 100
camera_data = bpy.data.cameras.new('MaskCamera')
camera = bpy.data.objects.new('MaskCamera', camera_data)
scene.collection.objects.link(camera)
camera.location = (0, 0, 3)
camera_data.type = 'ORTHO'
camera_data.ortho_scale = 2.4
scene.camera = camera
scene.render.filepath = 'original-path'
scene.render.film_transparent = True
scene.render.image_settings.color_mode = 'RGBA'
original_world = scene.world
ctx = BlenderContext(args=SimpleNamespace(), settings=SimpleNamespace(), scene=scene, mesh_obj=target)
if render_fails:
    # Replace only the external render operator; all Blender data mutations run.
    def fail_render(**kwargs):
        raise RuntimeError('injected render failure')
    annotations.bpy = SimpleNamespace(data=bpy.data, context=bpy.context,
        ops=SimpleNamespace(render=SimpleNamespace(render=fail_render)))
try:
    annotations.render_visible_object_mask(ctx, str(root/'mask.png'))
except RuntimeError as exc:
    assert render_fails and str(exc) == 'injected render failure'
else:
    assert not render_fails
for obj in (target, background):
    assert obj.data == shared, 'shared mesh identity changed'
    assert [(s.link, s.material) for s in obj.material_slots] == original_slots[obj.name], 'object slot state changed'
assert list(shared.materials) == original_materials, 'mesh materials changed'
assert [p.material_index for p in shared.polygons] == original_indices, 'polygon indices changed'
assert set(bpy.data.meshes.keys()) == original_meshes, 'temporary meshes leaked'
assert scene.world == original_world
assert scene.render.filepath == 'original-path'
assert scene.render.film_transparent
assert scene.render.image_settings.color_mode == 'RGBA'
print('MASK_STATE_RESTORED')
""")
    result = subprocess.run(
        [
            blender,
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
            str(script),
            "--",
            str(tmp_path),
            str(with_slots),
            str(render_fails),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MASK_STATE_RESTORED" in result.stdout
    if not render_fails:
        with Image.open(tmp_path / "mask.png") as image:
            mask = image.convert("L")
            assert mask.getpixel((16, 16)) > 250, "target must be white"
            assert mask.getpixel((48, 16)) < 5, "linked distractor must be black"
