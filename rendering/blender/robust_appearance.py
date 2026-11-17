# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Realize opt-in metallic stress scenes without modifying the source asset file.

The sampler in ``rendering/robust_appearance.py`` records every value in the
scene record's ``appearance`` section; this module only applies them. The
target's in-memory materials are replaced by gray metal (one material for the
sides, one for the end caps), so textured targets are rejected instead of
silently losing their texture. Geometry and the object frame are unchanged.
"""

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import bpy
import numpy as np
from mathutils import Vector

from rendering.usd_blender import image_texture

from .context import BlenderContext, BpyObject
from .materials import find_or_create_object_bsdf, remove_input_links

AXES = {"x": 0, "y": 1, "z": 2}
SIDE_MATERIAL = "RobustMetalSide"
CAP_MATERIAL = "RobustMetalCap"
SUN_NAME = "RobustSun"
# A face belongs to an end cap when its normal is within ~25 degrees of the axis.
CAP_NORMAL_MIN_COMPONENT = 0.9
# Brushed-machining noise: stretched along local Z so it reads as fine grooves.
CAP_BRUSH_STRETCH = (1.0, 1.0, 0.035)
CAP_BRUSH_DETAIL = 2.0
# Cap roughness varies within this factor band around the sampled value.
CAP_ROUGHNESS_SPREAD = (0.65, 1.35)
CAP_ROUGHNESS_LIMITS = (0.02, 0.8)


def apply_appearance(ctx: BlenderContext, record: BpyObject) -> None:
    """Apply a record's metallic appearance: exposure, materials and sun light.

    Records without an ``appearance`` section are left untouched.
    """
    spec = record.get("appearance")
    if spec is None:
        return
    if spec.get("kind") != "metal_endface":
        raise ValueError("Unsupported appearance record")
    if image_texture(ctx.mesh_obj) is not None:
        raise ValueError(
            "robust_appearance replaces the target material and does not support "
            "textured targets; disable it or use object_material_mode: override"
        )
    # Version 1 records predate the explicit axis and always meant local Y.
    axis_index = AXES[spec.get("axis", "y")]
    apply_exposure(ctx, spec)
    side, cap = configure_metal_materials(spec)
    configure_cap_brush(cap, spec)
    assign_side_and_cap(ctx.mesh_obj, side, cap, axis_index)
    add_sun(ctx, spec)


def apply_exposure(ctx: BlenderContext, spec: BpyObject) -> None:
    """Set view exposure and the world background strength."""
    ctx.scene.view_settings.exposure = float(spec["exposure"])
    world = ctx.scene.world
    if not (world and world.node_tree):
        return
    for node in world.node_tree.nodes:
        if node.type == "BACKGROUND":
            node.inputs["Strength"].default_value = float(spec["world_strength"])


def configure_metal_materials(spec: BpyObject) -> tuple[BpyObject, BpyObject]:
    """Create or reuse the side and cap materials as plain gray metal.

    Returns ``(side, cap)``. Existing links into color, metallic and roughness
    are removed so the sampled constants take effect.
    """
    side = get_or_create_material(SIDE_MATERIAL)
    cap = get_or_create_material(CAP_MATERIAL)
    gray = float(spec["gray"])
    for mat, roughness in ((side, spec["roughness"]), (cap, spec["cap_roughness"])):
        mat.use_nodes = True
        bsdf = find_or_create_object_bsdf(mat)
        for name in ("Base Color", "Metallic", "Roughness"):
            remove_input_links(mat.node_tree, bsdf.inputs[name])
        bsdf.inputs["Base Color"].default_value = (gray, gray, gray, 1)
        bsdf.inputs["Metallic"].default_value = 1
        bsdf.inputs["Roughness"].default_value = float(roughness)
    return side, cap


def get_or_create_material(name: str) -> BpyObject:
    """Return the named material, creating it on first use."""
    return bpy.data.materials.get(name) or bpy.data.materials.new(name)


def configure_cap_brush(cap: BpyObject, spec: BpyObject) -> None:
    """Drive cap roughness with stretched noise for subtle machining marks.

    The node graph is built once and reused; only its parameters change per
    record. The pattern varies roughness only, so geometry stays exact.
    """
    nodes = cap.node_tree.nodes
    if not nodes.get("CapBrush"):
        build_cap_brush_nodes(cap.node_tree)
    nodes["CapBrush"].inputs["Scale"].default_value = float(spec["brush_scale"])
    cap_roughness = float(spec["cap_roughness"])
    ramp = nodes["CapRoughness"]
    ramp.inputs["To Min"].default_value = max(
        CAP_ROUGHNESS_LIMITS[0], cap_roughness * CAP_ROUGHNESS_SPREAD[0]
    )
    ramp.inputs["To Max"].default_value = min(
        CAP_ROUGHNESS_LIMITS[1], cap_roughness * CAP_ROUGHNESS_SPREAD[1]
    )
    cap.node_tree.links.new(
        ramp.outputs["Result"], find_or_create_object_bsdf(cap).inputs["Roughness"]
    )


def build_cap_brush_nodes(node_tree: BpyObject) -> None:
    """Build object coordinates -> stretch -> noise -> roughness range nodes."""
    nodes, links = node_tree.nodes, node_tree.links
    coordinates = nodes.new("ShaderNodeTexCoord")
    coordinates.name = "CapCoordinates"
    stretch = nodes.new("ShaderNodeVectorMath")
    stretch.operation = "MULTIPLY"
    stretch.name = "CapBrushScale"
    stretch.inputs[1].default_value = CAP_BRUSH_STRETCH
    noise = nodes.new("ShaderNodeTexNoise")
    noise.name = "CapBrush"
    noise.inputs["Detail"].default_value = CAP_BRUSH_DETAIL
    ramp = nodes.new("ShaderNodeMapRange")
    ramp.name = "CapRoughness"
    links.new(coordinates.outputs["Object"], stretch.inputs[0])
    links.new(stretch.outputs["Vector"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], ramp.inputs["Value"])


def assign_side_and_cap(
    mesh_obj: BpyObject, side: BpyObject, cap: BpyObject, axis_index: int
) -> None:
    """Replace the mesh materials and mark faces facing along the axis as caps."""
    materials = mesh_obj.data.materials
    materials.clear()
    materials.append(side)
    materials.append(cap)
    polygons = mesh_obj.data.polygons
    normals = np.empty(len(polygons) * 3, dtype=np.float32)
    polygons.foreach_get("normal", normals)
    is_cap = np.abs(normals.reshape(-1, 3)[:, axis_index]) > CAP_NORMAL_MIN_COMPONENT
    polygons.foreach_set("material_index", is_cap.astype(np.int32))


def add_sun(ctx: BlenderContext, spec: BpyObject) -> None:
    """Add the recorded sun light; it is cleared with the other extra lights."""
    data = bpy.data.lights.new(SUN_NAME, "SUN")
    data.energy = float(spec["sun_energy"])
    data.angle = float(spec["sun_angle"])
    data.use_shadow = True
    light = bpy.data.objects.new(SUN_NAME, data)
    ctx.scene.collection.objects.link(light)
    # ``sun_direction`` points toward the sun; the light shines along -Z.
    light.rotation_euler = (-Vector(spec["sun_direction"])).to_track_quat("-Z", "Y").to_euler()
    ctx.extra_lights.append(light)
