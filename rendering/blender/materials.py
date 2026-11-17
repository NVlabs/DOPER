# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Target material appearance and shader-node helpers."""

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import math

import bpy

from .context import BlenderContext, BpyObject


def parse_rgb_color(value: BpyObject) -> BpyObject:
    """Parse an object color given by name or 3/4 channel values into RGBA."""
    if isinstance(value, str):
        named = {
            "orange": (1.0, 0.34, 0.02, 1.0),
            "green": (0.05, 0.62, 0.18, 1.0),
        }
        if value.lower() not in named:
            raise ValueError(f"unknown object color {value!r}")
        return named[value.lower()]

    channels = [float(channel) for channel in value]
    if len(channels) not in (3, 4):
        raise ValueError(f"object colors must have 3 or 4 channels, got {value!r}")
    if max(channels) > 1.0:
        channels = [channel / 255.0 for channel in channels]
    if len(channels) == 3:
        channels.append(1.0)
    return tuple(channels)


def remove_input_links(node_tree: BpyObject, socket: BpyObject) -> None:
    """Remove all links feeding the input ``socket`` of ``node_tree``."""
    for link in list(socket.links):
        node_tree.links.remove(link)


def find_or_create_object_bsdf(mat: BpyObject) -> BpyObject:
    """Return the Principled BSDF of ``mat``, creating one if missing."""
    if not mat.use_nodes:
        mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("MainBSDF") or nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        output = nodes.get("Material Output") or nodes.new("ShaderNodeOutputMaterial")
        mat.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return bsdf


def randomize_configured_object_material(ctx: BlenderContext) -> bool:
    """Randomize the object material per the config; return whether it was applied."""
    if ctx.settings.object_material_mode != "override" or not ctx.object_rgba_choices:
        return False

    mat = bpy.data.materials.get("ObjectOverrideMaterial")
    if mat is None:
        mat = bpy.data.materials.new("ObjectOverrideMaterial")
    mat.use_nodes = True
    ctx.mesh_obj.data.materials.clear()
    ctx.mesh_obj.data.materials.append(mat)
    for polygon in ctx.mesh_obj.data.polygons:
        polygon.material_index = 0

    bsdf = find_or_create_object_bsdf(mat)
    remove_input_links(mat.node_tree, bsdf.inputs["Base Color"])
    remove_input_links(mat.node_tree, bsdf.inputs["Metallic"])
    remove_input_links(mat.node_tree, bsdf.inputs["Roughness"])
    bsdf.inputs["Base Color"].default_value = ctx.rng.choice(ctx.object_rgba_choices)
    bsdf.inputs["Metallic"].default_value = ctx.settings.object_metallic
    bsdf.inputs["Roughness"].default_value = ctx.rng.uniform(
        ctx.settings.object_roughness_min, ctx.settings.object_roughness_max
    )
    mat.diffuse_color = bsdf.inputs["Base Color"].default_value
    return True


def randomize_aluminum_material(ctx: BlenderContext) -> None:
    """Only randomize if this is our aluminum material (has MainBSDF node)."""
    mat = ctx.mesh_obj.data.materials[0] if ctx.mesh_obj.data.materials else None
    if mat is None or not mat.node_tree:
        return

    # Skip if not our aluminum material (preserve existing materials)
    nodes = mat.node_tree.nodes
    if "MainBSDF" not in nodes:
        return  # Not our material, don't modify

    bsdf = nodes.get("MainBSDF")
    if bsdf:
        base_gray = ctx.rng.uniform(
            ctx.settings.mat_base_color_min, ctx.settings.mat_base_color_max
        )
        tint = ctx.rng.choice(["neutral", "warm", "cool"])
        if tint == "warm":
            bsdf.inputs["Base Color"].default_value = (
                base_gray + 0.02,
                base_gray,
                base_gray - 0.02,
                1.0,
            )
        elif tint == "cool":
            bsdf.inputs["Base Color"].default_value = (
                base_gray - 0.02,
                base_gray,
                base_gray + 0.03,
                1.0,
            )
        else:
            bsdf.inputs["Base Color"].default_value = (
                base_gray,
                base_gray + 0.01,
                base_gray + 0.02,
                1.0,
            )

    mapping = nodes.get("BrushMapping")
    if mapping:
        mapping.inputs["Rotation"].default_value = (0, 0, ctx.rng.uniform(0, math.pi))
        scale_factor = ctx.rng.uniform(0.5, 1.5)
        mapping.inputs["Scale"].default_value = (
            50.0 * scale_factor,
            2.0 * scale_factor,
            1.0,
        )

    noise = nodes.get("BrushNoise")
    if noise:
        noise.inputs["Scale"].default_value = ctx.rng.uniform(
            ctx.settings.mat_brush_scale_min, ctx.settings.mat_brush_scale_max
        )
        noise.inputs["Roughness"].default_value = ctx.rng.uniform(0.5, 0.9)

    wave = nodes.get("BrushWave")
    if wave:
        wave.inputs["Scale"].default_value = ctx.rng.uniform(
            ctx.settings.mat_brush_scale_min, ctx.settings.mat_brush_scale_max
        )
        wave.inputs["Distortion"].default_value = ctx.rng.uniform(1.0, 3.0)

    ramp = nodes.get("RoughnessRamp")
    if ramp:
        base_rough = ctx.rng.uniform(
            ctx.settings.mat_roughness_min, ctx.settings.mat_roughness_min + 0.1
        )
        ramp.color_ramp.elements[0].color = (base_rough, base_rough, base_rough, 1.0)
        ramp.color_ramp.elements[1].color = (
            ctx.settings.mat_roughness_max,
            ctx.settings.mat_roughness_max,
            ctx.settings.mat_roughness_max,
            1.0,
        )


def randomize_object_material(ctx: BlenderContext) -> None:
    """Randomize the object material per the configured mode."""
    if randomize_configured_object_material(ctx):
        return
    randomize_aluminum_material(ctx)
