# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise opt-in metal shader/light realization in actual Blender."""

import pytest
from conftest import run_blender

pytestmark = pytest.mark.blender

# Shared preamble for the Blender-side scripts: imports and a recorded appearance.
PREAMBLE = """
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path.cwd()))
import bpy
from rendering.blender.robust_appearance import apply_appearance

SPEC = {
    "kind": "metal_endface", "exposure": 0.5, "world_strength": 0.12, "gray": 0.65,
    "metallic": 1.0, "roughness": 0.2, "cap_roughness": 0.35, "brush_scale": 180.0,
    "sun_energy": 3.0, "sun_angle": 0.01, "sun_direction": [1.0, 0.0, 1.0],
}
scene = bpy.context.scene
scene.world.use_nodes = True
"""


def run_in_blender(blender, tmp_path, body, marker):
    """Run ``PREAMBLE + body`` in background Blender and require ``marker`` in stdout."""
    script = tmp_path / "check.py"
    script.write_text(PREAMBLE + body + f"\nprint({marker!r})\n")
    result = run_blender(blender, "--threads", "2", script=script, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert marker in result.stdout


def test_material_controls_and_geometry_preserved(blender, tmp_path):
    run_in_blender(
        blender,
        tmp_path,
        """
# Version 1 record (no axis): a cylinder rotated so its axis is local Y.
bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.025, depth=0.045, rotation=(1.57079632679, 0, 0))
obj = bpy.context.object
bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
coords = [tuple(v.co) for v in obj.data.vertices]
ctx = SimpleNamespace(scene=scene, mesh_obj=obj, extra_lights=[])
apply_appearance(ctx, {"appearance": SPEC})
assert coords == [tuple(v.co) for v in obj.data.vertices], "geometry must be unchanged"
assert sum(p.material_index == 1 for p in obj.data.polygons) == 2, "two cap faces"
assert len(ctx.extra_lights) == 1 and ctx.extra_lights[0].data.type == "SUN"
assert abs(scene.view_settings.exposure - 0.5) < 1e-6
background = scene.world.node_tree.nodes.get("Background")
assert abs(background.inputs["Strength"].default_value - 0.12) < 1e-6
cap = obj.data.materials[1]
assert cap.node_tree.nodes.get("CapBrush")
assert cap.node_tree.nodes.get("Principled BSDF").inputs["Roughness"].is_linked
side_bsdf = obj.data.materials[0].node_tree.nodes.get("Principled BSDF")
assert side_bsdf.inputs["Metallic"].default_value == 1
""",
        "MATERIAL_GEOMETRY_CHECK_OK",
    )


def test_cap_axis_follows_record_and_textured_targets_are_rejected(blender, tmp_path):
    run_in_blender(
        blender,
        tmp_path,
        """
# Default cylinder axis is local Z; the record names it explicitly.
bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.025, depth=0.045)
obj = bpy.context.object
apply_appearance(SimpleNamespace(scene=scene, mesh_obj=obj, extra_lights=[]), {"appearance": {**SPEC, "axis": "z"}})
caps = [p for p in obj.data.polygons if p.material_index == 1]
assert len(caps) == 2 and all(abs(p.normal.z) > 0.9 for p in caps)

bpy.ops.mesh.primitive_cube_add(size=0.05)
textured = bpy.context.object
material = bpy.data.materials.new("Textured")
material.use_nodes = True
node = material.node_tree.nodes.new("ShaderNodeTexImage")
node.image = bpy.data.images.new("albedo", 4, 4)
textured.data.materials.append(material)
try:
    apply_appearance(SimpleNamespace(scene=scene, mesh_obj=textured, extra_lights=[]), {"appearance": SPEC})
except ValueError as err:
    assert "textured" in str(err)
else:
    raise AssertionError("textured target must be rejected")
assert textured.data.materials[0].name == "Textured", "rejected target keeps its material"

# A texture hidden inside a node group must be detected as well.
bpy.ops.mesh.primitive_cube_add(size=0.05)
grouped = bpy.context.object
group = bpy.data.node_groups.new("TexturedGroup", "ShaderNodeTree")
group.nodes.new("ShaderNodeTexImage").image = bpy.data.images.new("grouped", 4, 4)
material = bpy.data.materials.new("GroupTextured")
material.use_nodes = True
material.node_tree.nodes.new("ShaderNodeGroup").node_tree = group
grouped.data.materials.append(material)
try:
    apply_appearance(SimpleNamespace(scene=scene, mesh_obj=grouped, extra_lights=[]), {"appearance": SPEC})
except ValueError as err:
    assert "textured" in str(err)
else:
    raise AssertionError("texture inside a node group must be rejected")
""",
        "AXIS_TEXTURE_CHECK_OK",
    )
