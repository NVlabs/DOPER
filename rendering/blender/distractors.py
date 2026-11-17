# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Asset and procedural distractor creation, materials and placement."""

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import colorsys
import contextlib
import math
import os

import bpy
from mathutils import Euler, Vector

from .context import BlenderContext, BpyObject


def clear_distractors(ctx: BlenderContext, start_index: int = 0) -> None:
    """Remove loaded distractors from index ``start_index`` onward."""
    for obj in ctx.loaded_distractors[start_index:]:
        with contextlib.suppress(BaseException):
            bpy.data.objects.remove(obj, do_unlink=True)
    ctx.loaded_distractors = ctx.loaded_distractors[:start_index]


def has_valid_material(obj: BpyObject) -> bool:
    """Check if object has a valid material with working textures."""
    if not obj.data.materials:
        return False

    for mat in obj.data.materials:
        if mat is None or not mat.use_nodes:
            return False

        # Check for image textures with missing files
        for node in mat.node_tree.nodes:
            if node.type == "TEX_IMAGE":
                if node.image is None:
                    return False
                # Check if image file exists/loaded properly
                if node.image.size[0] == 0 or node.image.size[1] == 0:
                    return False

    return True


def create_random_material(ctx: BlenderContext) -> BpyObject:
    """Create a random simple material for distractors."""
    mat = bpy.data.materials.new(name=f"RandomMat_{ctx.rng.randint(0, 99999)}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes

    # Get the principled BSDF
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")

    # Random material type
    mat_type = ctx.rng.choice(["plastic", "metal", "rubber", "ceramic", "wood"])

    if mat_type == "plastic":
        # Colorful plastic
        bsdf.inputs["Base Color"].default_value = (
            ctx.rng.uniform(0.1, 1.0),
            ctx.rng.uniform(0.1, 1.0),
            ctx.rng.uniform(0.1, 1.0),
            1.0,
        )
        bsdf.inputs["Metallic"].default_value = 0.0
        bsdf.inputs["Roughness"].default_value = ctx.rng.uniform(0.2, 0.6)
        bsdf.inputs["IOR"].default_value = 1.45

    elif mat_type == "metal":
        # Metallic material
        gray = ctx.rng.uniform(0.5, 0.95)
        # Slight color tint
        tint = ctx.rng.choice(["neutral", "gold", "copper", "blue"])
        if tint == "gold":
            bsdf.inputs["Base Color"].default_value = (
                gray,
                gray * 0.85,
                gray * 0.5,
                1.0,
            )
        elif tint == "copper":
            bsdf.inputs["Base Color"].default_value = (
                gray,
                gray * 0.7,
                gray * 0.6,
                1.0,
            )
        elif tint == "blue":
            bsdf.inputs["Base Color"].default_value = (
                gray * 0.8,
                gray * 0.85,
                gray,
                1.0,
            )
        else:
            bsdf.inputs["Base Color"].default_value = (gray, gray, gray, 1.0)
        bsdf.inputs["Metallic"].default_value = 1.0
        bsdf.inputs["Roughness"].default_value = ctx.rng.uniform(0.1, 0.5)

    elif mat_type == "rubber":
        # Matte rubber
        bsdf.inputs["Base Color"].default_value = (
            ctx.rng.uniform(0.02, 0.3),
            ctx.rng.uniform(0.02, 0.3),
            ctx.rng.uniform(0.02, 0.3),
            1.0,
        )
        bsdf.inputs["Metallic"].default_value = 0.0
        bsdf.inputs["Roughness"].default_value = ctx.rng.uniform(0.7, 1.0)

    elif mat_type == "ceramic":
        # Shiny ceramic
        bsdf.inputs["Base Color"].default_value = (
            ctx.rng.uniform(0.7, 1.0),
            ctx.rng.uniform(0.7, 1.0),
            ctx.rng.uniform(0.7, 1.0),
            1.0,
        )
        bsdf.inputs["Metallic"].default_value = 0.0
        bsdf.inputs["Roughness"].default_value = ctx.rng.uniform(0.05, 0.2)
        bsdf.inputs["IOR"].default_value = 1.5

    else:  # wood
        # Wood-like color
        brown = ctx.rng.uniform(0.3, 0.7)
        bsdf.inputs["Base Color"].default_value = (
            brown,
            brown * ctx.rng.uniform(0.5, 0.8),
            brown * ctx.rng.uniform(0.3, 0.5),
            1.0,
        )
        bsdf.inputs["Metallic"].default_value = 0.0
        bsdf.inputs["Roughness"].default_value = ctx.rng.uniform(0.4, 0.8)

    return mat


def create_random_plastic_material(ctx: BlenderContext) -> BpyObject:
    """Create a colorful plastic material for procedural primitive distractors."""
    mat = bpy.data.materials.new(name=f"PlasticDistractor_{ctx.rng.randint(0, 99999)}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")

    hue = ctx.rng.random()
    if ctx.rng.random() < ctx.settings.procedural_distractor_dark_prob:
        saturation = ctx.rng.uniform(*ctx.settings.procedural_distractor_dark_saturation_range)
        value = ctx.rng.uniform(*ctx.settings.procedural_distractor_dark_value_range)
    else:
        saturation = ctx.rng.uniform(0.45, 0.95)
        value = ctx.rng.uniform(0.35, 0.95)
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    bsdf.inputs["Base Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = ctx.rng.uniform(0.30, 0.75)
    if "IOR" in bsdf.inputs:
        bsdf.inputs["IOR"].default_value = 1.45
    return mat


def apply_random_material_if_needed(ctx: BlenderContext, obj: BpyObject) -> None:
    """Apply random material to object if it has no valid material."""
    if not has_valid_material(obj):
        # Clear existing materials
        obj.data.materials.clear()
        # Add random material
        obj.data.materials.append(create_random_material(ctx))


def load_distractors(
    ctx: BlenderContext, n: int, center: BpyObject, scene_scale: float = 1.0
) -> None:
    """Load distractor objects around the main object.

    Args:
        ctx: Worker context.
        n: Number of distractors to load
        center: Center point to place distractors around
        scene_scale: Scale factor based on mesh size (1.0 = default for ~10cm objects)
    """
    clear_distractors(ctx)
    if not ctx.distractor_files:
        return

    for _i in range(min(n, len(ctx.distractor_files))):
        dist_file = ctx.rng.choice(ctx.distractor_files)
        before_objs = set(bpy.data.objects.keys())
        try:
            # Support both USD and GLB/GLTF formats
            if dist_file.lower().endswith((".glb", ".gltf")):
                bpy.ops.import_scene.gltf(filepath=os.path.abspath(dist_file))
            else:
                bpy.ops.wm.usd_import(filepath=os.path.abspath(dist_file))
        except Exception as e:
            print(f"Warning: failed to import distractor {dist_file}: {e}")
            continue

        new_objs = [bpy.data.objects[n] for n in sorted(set(bpy.data.objects.keys()) - before_objs)]
        mesh_objs = [o for o in new_objs if o.type == "MESH"]
        non_mesh = [o for o in new_objs if o.type != "MESH"]

        for obj in mesh_objs:
            angle = ctx.rng.uniform(0, 2 * math.pi)
            # Scale distractor distance with scene size
            dist = ctx.rng.uniform(0.05, 0.20) * scene_scale
            z_offset = ctx.rng.uniform(-0.01, 0.05) * scene_scale
            obj.location = center + Vector(
                (math.cos(angle) * dist, math.sin(angle) * dist, z_offset)
            )
            obj.rotation_euler = Euler(
                (
                    ctx.rng.uniform(0, 2 * math.pi),
                    ctx.rng.uniform(0, 2 * math.pi),
                    ctx.rng.uniform(0, 2 * math.pi),
                )
            )

            obj_size = max(obj.dimensions)
            if obj_size > 0:
                target_scale = ctx.rng.uniform(0.1, 0.8) * ctx.mesh_size / obj_size
                obj.scale = (target_scale, target_scale, target_scale)

            # Apply random material if original has missing textures
            apply_random_material_if_needed(ctx, obj)

            ctx.loaded_distractors.append(obj)

        for obj in non_mesh:
            with contextlib.suppress(BaseException):
                bpy.data.objects.remove(obj, do_unlink=True)


def add_procedural_primitive_distractor(
    ctx: BlenderContext, shape: BpyObject, name: BpyObject
) -> BpyObject:
    """Add a small primitive mesh and return the created object."""
    if shape == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=0.5)
    elif shape == "cone":
        bpy.ops.mesh.primitive_cone_add(vertices=24, radius1=0.5, radius2=0.0, depth=1.0)
    else:
        bpy.ops.mesh.primitive_cube_add(size=1.0)

    obj = bpy.context.object
    obj.name = name
    obj.data.name = f"{name}Mesh"
    obj.data.materials.append(create_random_plastic_material(ctx))
    return obj


def camera_lateral_basis(ctx: BlenderContext) -> BpyObject:
    """Return stable right/up vectors for image-space distractor placement."""
    quat = ctx.cam_obj.matrix_world.to_quaternion()
    right = quat @ Vector((1, 0, 0))
    up = quat @ Vector((0, 1, 0))
    if right.length < 0.001:
        right = Vector((1, 0, 0))
    if up.length < 0.001:
        up = Vector((0, 0, 1))
    return right.normalized(), up.normalized()


def place_procedural_distractor(
    ctx: BlenderContext,
    center: BpyObject,
    cam_dir: BpyObject,
    bbox_min_z: BpyObject,
    *,
    layer: BpyObject,
    index: BpyObject,
) -> None:
    """Place one primitive away from the object center to avoid total occlusion."""
    shape = ctx.rng.choice(ctx.settings.procedural_distractor_shapes)
    obj = add_procedural_primitive_distractor(
        ctx,
        shape,
        f"Procedural{layer.title()}Distractor_{index:02d}_{shape}",
    )

    target_size = ctx.rng.uniform(*ctx.settings.procedural_distractor_size_range) * ctx.mesh_size
    bpy.context.view_layer.update()
    obj_size = max(obj.dimensions)
    if obj_size > 0:
        scale = target_size / obj_size
        obj.scale = (scale, scale, scale)

    right, up = camera_lateral_basis(ctx)
    if (
        layer == "foreground"
        and ctx.rng.random() < ctx.settings.procedural_distractor_foreground_center_prob
    ):
        lateral_range = ctx.settings.procedural_distractor_foreground_center_lateral_range
    else:
        lateral_range = ctx.settings.procedural_distractor_lateral_range
    lateral_mag = ctx.rng.uniform(*lateral_range) * ctx.mesh_size
    lateral_sign = -1.0 if ctx.rng.random() < 0.5 else 1.0
    lateral = right * lateral_mag * lateral_sign
    vertical = up * ctx.rng.uniform(-0.25, 0.45) * ctx.mesh_size

    cam_dist_local = (center - ctx.cam_obj.matrix_world.translation).length
    if layer == "foreground":
        depth = (
            ctx.rng.uniform(*ctx.settings.procedural_distractor_foreground_depth_range)
            * cam_dist_local
        )
        base = center + cam_dir.normalized() * depth
    else:
        depth = (
            ctx.rng.uniform(*ctx.settings.procedural_distractor_background_depth_range)
            * ctx.mesh_size
        )
        base = center - cam_dir.normalized() * depth

    obj.location = base + lateral + vertical
    obj.location.z = max(
        bbox_min_z + 0.5 * target_size,
        obj.location.z
        + ctx.rng.uniform(*ctx.settings.procedural_distractor_vertical_jitter_range)
        * ctx.mesh_size,
    )
    obj.rotation_euler = Euler(
        (
            ctx.rng.uniform(0, 2 * math.pi),
            ctx.rng.uniform(0, 2 * math.pi),
            ctx.rng.uniform(0, 2 * math.pi),
        )
    )

    ctx.loaded_distractors.append(obj)


def load_procedural_distractors(
    ctx: BlenderContext, center: BpyObject, cam_dir: BpyObject, bbox_min_z: BpyObject
) -> None:
    """Create camera-relative foreground/background primitive distractors."""
    if ctx.settings.procedural_distractor_foreground_max > 0:
        n_fg = ctx.rng.randint(
            max(0, ctx.settings.procedural_distractor_foreground_min),
            max(
                ctx.settings.procedural_distractor_foreground_min,
                ctx.settings.procedural_distractor_foreground_max,
            ),
        )
    else:
        n_fg = 0
    if ctx.settings.procedural_distractor_background_max > 0:
        n_bg = ctx.rng.randint(
            max(0, ctx.settings.procedural_distractor_background_min),
            max(
                ctx.settings.procedural_distractor_background_min,
                ctx.settings.procedural_distractor_background_max,
            ),
        )
    else:
        n_bg = 0

    for index in range(n_fg):
        place_procedural_distractor(
            ctx, center, cam_dir, bbox_min_z, layer="foreground", index=index
        )
    for index in range(n_bg):
        place_procedural_distractor(
            ctx, center, cam_dir, bbox_min_z, layer="background", index=index
        )
