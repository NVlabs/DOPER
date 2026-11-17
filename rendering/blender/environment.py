# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""World backgrounds, ground surfaces and lighting."""

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import colorsys
import contextlib
import math
import os

import bpy
from mathutils import Vector

from rendering.appearance import sample_hdri

from .context import BlenderContext, BpyObject


def setup_hdri(ctx: BlenderContext, filepath: str) -> None:
    """Use the HDRI image at ``filepath`` as the world background."""
    world = bpy.data.worlds.get("SynthWorld")
    if world is None:
        world = bpy.data.worlds.new("SynthWorld")
    ctx.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    bg = nodes.new("ShaderNodeBackground")
    env_tex = nodes.new("ShaderNodeTexEnvironment")
    mapping = nodes.new("ShaderNodeMapping")
    tex_coord = nodes.new("ShaderNodeTexCoord")
    output = nodes.new("ShaderNodeOutputWorld")

    env_tex.image = bpy.data.images.load(filepath, check_existing=True)

    links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], env_tex.inputs["Vector"])
    links.new(env_tex.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], output.inputs["Surface"])

    rotation, strength = sample_hdri(ctx.rng)
    mapping.inputs["Rotation"].default_value = rotation
    bg.inputs["Strength"].default_value = strength


def setup_color_world_background(
    ctx: BlenderContext,
    *,
    value_range: BpyObject,
    saturation_range: BpyObject,
    strength_range: BpyObject,
) -> None:
    """Setup color world background."""
    world = bpy.data.worlds.get("SynthWorld")
    if world is None:
        world = bpy.data.worlds.new("SynthWorld")
    ctx.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    bg = nodes.new("ShaderNodeBackground")
    output = nodes.new("ShaderNodeOutputWorld")
    bg.inputs["Color"].default_value = random_flat_rgba(
        ctx,
        saturation=saturation_range,
        value=value_range,
    )
    bg.inputs["Strength"].default_value = ctx.rng.uniform(*strength_range)
    links.new(bg.outputs["Background"], output.inputs["Surface"])


def setup_random_world_background(ctx: BlenderContext) -> None:
    """Pick one of the configured world background modes at random and apply it."""
    setup_color_world_background(
        ctx,
        value_range=ctx.settings.world_background_value_range,
        saturation_range=ctx.settings.world_background_saturation_range,
        strength_range=ctx.settings.world_background_strength_range,
    )


def setup_dark_world_background(ctx: BlenderContext) -> None:
    """Apply a dark world background."""
    setup_color_world_background(
        ctx,
        value_range=ctx.settings.dark_background_value_range,
        saturation_range=ctx.settings.dark_background_saturation_range,
        strength_range=(0.25, 0.9),
    )


def setup_studio_world_background(ctx: BlenderContext) -> None:
    """Apply a neutral studio world background."""
    world = bpy.data.worlds.get("SynthWorld")
    if world is None:
        world = bpy.data.worlds.new("SynthWorld")
    ctx.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    bg = nodes.new("ShaderNodeBackground")
    output = nodes.new("ShaderNodeOutputWorld")
    bg.inputs["Color"].default_value = (0.78, 0.80, 0.82, 1.0)
    bg.inputs["Strength"].default_value = 0.8
    links.new(bg.outputs["Background"], output.inputs["Surface"])


def setup_world_background(ctx: BlenderContext) -> None:
    """Apply a world background according to the configured modes."""
    modes = [
        mode
        for mode in ctx.settings.world_background_modes
        if mode in {"studio", "random", "dark", "hdri"}
    ]
    if not modes:
        modes = ["studio"]
    mode = ctx.rng.choice(modes)
    if mode == "hdri":
        if ctx.hdri_files:
            setup_hdri(ctx, os.path.abspath(ctx.rng.choice(ctx.hdri_files)))
        else:
            setup_random_world_background(ctx)
    elif mode == "random":
        setup_random_world_background(ctx)
    elif mode == "dark":
        setup_dark_world_background(ctx)
    else:
        setup_studio_world_background(ctx)


def setup_ground_plane(
    ctx: BlenderContext,
    tex_dir: str | None,
    ground_z: BpyObject,
    scene_scale: float = 1.0,
    *,
    force_procedural: bool = False,
    use_ground_image: bool = False,
) -> None:
    """Create ground plane scaled to scene size.

    Args:
        ctx: Worker context.
        tex_dir: Directory with texture files
        ground_z: Z position for ground
        scene_scale: Scale factor based on mesh size (1.0 = default for ~10cm objects)
        force_procedural: Use a procedural material regardless of ``tex_dir``.
        use_ground_image: Use a real photo instead of a texture set.
    """
    if ctx.ground_obj is not None:
        bpy.data.objects.remove(ctx.ground_obj, do_unlink=True)

    # Scale ground plane with scene - larger objects need larger ground
    ground_size = 2.0 * scene_scale
    bpy.ops.mesh.primitive_plane_add(
        size=ground_size, location=(0, 0, ground_z - 0.001 * scene_scale)
    )
    ctx.ground_obj = bpy.context.active_object
    ctx.ground_obj.name = "GroundPlane"

    mat = bpy.data.materials.new("GroundMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")

    if use_ground_image and ctx.settings.ground_image_files and not force_procedural:
        setup_image_ground_material(ctx, mat, ctx.rng.choice(ctx.settings.ground_image_files))
        ctx.ground_obj.data.materials.append(mat)
        return

    use_procedural = force_procedural or tex_dir is None
    if use_procedural or tex_dir is None:
        setup_procedural_ground_material(ctx, mat, scene_scale)
        ctx.ground_obj.data.materials.append(mat)
        return

    color_file = normal_file = roughness_file = None
    for f in os.listdir(tex_dir):
        fp = os.path.join(tex_dir, f)
        fl = f.lower()
        if "color" in fl and fl.endswith(".jpg"):
            color_file = fp
        elif "normal" in fl and fl.endswith(".jpg"):
            normal_file = fp
        elif "roughness" in fl and fl.endswith(".jpg"):
            roughness_file = fp

    # Scale UV tiling with scene size - larger scenes need more tiling to maintain texture detail
    uv_scale = ctx.rng.uniform(1.0, 5.0) * scene_scale
    tex_coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (uv_scale, uv_scale, 1.0)
    links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])

    if color_file:
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(color_file, check_existing=True)
        links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
        links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])

    if roughness_file:
        tex_r = nodes.new("ShaderNodeTexImage")
        tex_r.image = bpy.data.images.load(roughness_file, check_existing=True)
        tex_r.image.colorspace_settings.name = "Non-Color"
        links.new(mapping.outputs["Vector"], tex_r.inputs["Vector"])
        links.new(tex_r.outputs["Color"], bsdf.inputs["Roughness"])

    if normal_file:
        tex_n = nodes.new("ShaderNodeTexImage")
        tex_n.image = bpy.data.images.load(normal_file, check_existing=True)
        tex_n.image.colorspace_settings.name = "Non-Color"
        normal_map = nodes.new("ShaderNodeNormalMap")
        links.new(mapping.outputs["Vector"], tex_n.inputs["Vector"])
        links.new(tex_n.outputs["Color"], normal_map.inputs["Color"])
        links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])

    ctx.ground_obj.data.materials.append(mat)


def setup_image_ground_material(ctx: BlenderContext, mat: BpyObject, image_file: BpyObject) -> None:
    """Use a real photo as a repeated ground plane material."""
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        output = nodes.get("Material Output") or nodes.new("ShaderNodeOutputMaterial")
        links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    image = bpy.data.images.load(image_file, check_existing=True)
    image.colorspace_settings.name = "sRGB"

    tex_coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    with contextlib.suppress(Exception):
        tex.extension = "REPEAT"

    tile_u = max(1.0, ctx.rng.uniform(*ctx.settings.ground_image_tile_range))
    tile_v = max(1.0, ctx.rng.uniform(*ctx.settings.ground_image_tile_range))
    rotation_z = math.radians(ctx.rng.uniform(*ctx.settings.ground_image_rotation_deg_range))
    mapping.inputs["Location"].default_value = (ctx.rng.random(), ctx.rng.random(), 0.0)
    mapping.inputs["Rotation"].default_value = (0.0, 0.0, rotation_z)
    mapping.inputs["Scale"].default_value = (tile_u, tile_v, 1.0)

    assign_random_ground_image_uvs(ctx)
    links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = ctx.rng.uniform(
        *ctx.settings.ground_image_roughness_range
    )


def assign_random_ground_image_uvs(ctx: BlenderContext) -> None:
    """Map the plane UVs to a random square sub-crop of the source image."""
    if ctx.ground_obj is None:
        return

    mesh = ctx.ground_obj.data
    if not mesh.uv_layers:
        mesh.uv_layers.new(name="UVMap")
    uv_layer = mesh.uv_layers.active.data

    crop_scale = max(0.05, min(1.0, ctx.rng.uniform(*ctx.settings.ground_image_crop_scale_range)))
    u0 = ctx.rng.uniform(0.0, max(0.0, 1.0 - crop_scale))
    v0 = ctx.rng.uniform(0.0, max(0.0, 1.0 - crop_scale))
    rotate_steps = ctx.rng.randrange(4)
    flip_u = ctx.rng.random() < 0.5
    flip_v = ctx.rng.random() < 0.5

    def transform_corner(u: BpyObject, v: BpyObject) -> BpyObject:
        """Map one texture-corner coordinate through the ground tiling transform."""
        if flip_u:
            u = 1.0 - u
        if flip_v:
            v = 1.0 - v
        for _ in range(rotate_steps):
            u, v = v, 1.0 - u
        return (u0 + u * crop_scale, v0 + v * crop_scale)

    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            vert = mesh.vertices[mesh.loops[loop_index].vertex_index]
            u = 1.0 if vert.co.x >= 0 else 0.0
            v = 1.0 if vert.co.y >= 0 else 0.0
            uv_layer[loop_index].uv = transform_corner(u, v)


def setup_procedural_ground_material(
    ctx: BlenderContext, mat: BpyObject, scene_scale: float = 1.0
) -> None:
    """Build a procedural ground material (color, checker or wood)."""
    modes = ctx.settings.procedural_ground_modes
    mode = ctx.rng.choice(modes)
    if mode not in {"color", "wood", "checker"}:
        mode = "color"

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        output = nodes.get("Material Output") or nodes.new("ShaderNodeOutputMaterial")
        links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    if ctx.rng.random() < ctx.settings.dark_background_prob:
        bsdf.inputs["Base Color"].default_value = random_flat_rgba(
            ctx,
            saturation=ctx.settings.dark_background_saturation_range,
            value=ctx.settings.dark_background_value_range,
        )
        bsdf.inputs["Roughness"].default_value = ctx.rng.uniform(0.45, 0.90)
        return

    if ctx.rng.random() < ctx.settings.checker_background_prob or mode == "checker":
        setup_procedural_checker_nodes(ctx, mat, bsdf, scene_scale)
        return

    if mode == "wood":
        setup_procedural_wood_nodes(ctx, mat, bsdf, scene_scale)
    else:
        bsdf.inputs["Base Color"].default_value = random_flat_rgba(
            ctx,
            saturation=ctx.settings.flat_background_saturation_range,
            value=ctx.settings.flat_background_value_range,
        )
        bsdf.inputs["Roughness"].default_value = ctx.rng.uniform(0.35, 0.85)


def setup_procedural_checker_nodes(
    ctx: BlenderContext, mat: BpyObject, bsdf: BpyObject, scene_scale: float = 1.0
) -> None:
    """Add checker shader nodes to the ground material."""
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    tex_coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    checker = nodes.new("ShaderNodeTexChecker")

    color_a, color_b = random_checker_rgba_pair(ctx)
    checker.inputs["Color1"].default_value = color_a
    checker.inputs["Color2"].default_value = color_b
    checker.inputs["Scale"].default_value = ctx.rng.uniform(
        *ctx.settings.checker_background_scale_range
    ) / max(scene_scale, 0.25)
    mapping.inputs["Rotation"].default_value = (
        0.0,
        0.0,
        ctx.rng.uniform(0.0, 2.0 * math.pi),
    )

    links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], checker.inputs["Vector"])
    links.new(checker.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = ctx.rng.uniform(0.35, 0.85)


def setup_procedural_wood_nodes(
    ctx: BlenderContext, mat: BpyObject, bsdf: BpyObject, scene_scale: float = 1.0
) -> None:
    """Add wood shader nodes to the ground material."""
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    tex_coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    wave = nodes.new("ShaderNodeTexWave")
    ramp = nodes.new("ShaderNodeValToRGB")

    with contextlib.suppress(Exception):
        wave.wave_type = ctx.rng.choice(["RINGS", "BANDS"])
    with contextlib.suppress(Exception):
        wave.bands_direction = ctx.rng.choice(["X", "Y", "DIAGONAL"])
    with contextlib.suppress(Exception):
        wave.rings_direction = ctx.rng.choice(["X", "Y", "Z"])

    if "Scale" in wave.inputs:
        wave.inputs["Scale"].default_value = ctx.rng.uniform(8.0, 26.0) / max(scene_scale, 0.25)
    if "Distortion" in wave.inputs:
        wave.inputs["Distortion"].default_value = ctx.rng.uniform(8.0, 22.0)

    mapping.inputs["Rotation"].default_value = (
        0.0,
        0.0,
        ctx.rng.uniform(0.0, 2.0 * math.pi),
    )
    mapping.inputs["Scale"].default_value = (
        ctx.rng.uniform(0.7, 2.0),
        ctx.rng.uniform(2.5, 8.0),
        1.0,
    )

    dark = random_wood_rgba(ctx, value=(0.20, 0.38))
    mid = random_wood_rgba(ctx, value=(0.42, 0.62))
    light = random_wood_rgba(ctx, value=(0.62, 0.82))
    ramp.color_ramp.elements[0].position = ctx.rng.uniform(0.18, 0.38)
    ramp.color_ramp.elements[0].color = dark
    ramp.color_ramp.elements[1].position = ctx.rng.uniform(0.70, 0.92)
    ramp.color_ramp.elements[1].color = light
    extra = ramp.color_ramp.elements.new(ctx.rng.uniform(0.42, 0.62))
    extra.color = mid

    links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], wave.inputs["Vector"])
    links.new(wave.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = ctx.rng.uniform(0.35, 0.75)


def random_flat_rgba(
    ctx: BlenderContext, *, saturation: BpyObject = (0.08, 0.65), value: BpyObject = (0.35, 0.90)
) -> BpyObject:
    """Sample a flat background color from the configured ranges."""
    hue = ctx.rng.random()
    sat = ctx.rng.uniform(*saturation)
    val = ctx.rng.uniform(*value)
    rgb = colorsys.hsv_to_rgb(hue, sat, val)
    return (rgb[0], rgb[1], rgb[2], 1.0)


def random_checker_rgba_pair(ctx: BlenderContext) -> BpyObject:
    """Sample two contrasting checker colors."""
    color_a = random_flat_rgba(
        ctx,
        saturation=ctx.settings.checker_background_saturation_range,
        value=ctx.settings.checker_background_value_range,
    )
    color_b = color_a
    for _ in range(8):
        candidate = random_flat_rgba(
            ctx,
            saturation=ctx.settings.checker_background_saturation_range,
            value=ctx.settings.checker_background_value_range,
        )
        distance = sum((candidate[i] - color_a[i]) ** 2 for i in range(3)) ** 0.5
        value_gap = abs(max(candidate[:3]) - max(color_a[:3]))
        if distance > 0.35 or value_gap > 0.25:
            color_b = candidate
            break
        color_b = candidate
    return color_a, color_b


def random_wood_rgba(ctx: BlenderContext, *, value: BpyObject = (0.25, 0.75)) -> BpyObject:
    """Sample a wood base color."""
    hue = ctx.rng.uniform(0.055, 0.115)
    sat = ctx.rng.uniform(0.35, 0.80)
    val = ctx.rng.uniform(*value)
    rgb = colorsys.hsv_to_rgb(hue, sat, val)
    return (rgb[0], rgb[1], rgb[2], 1.0)


def setup_random_lights(ctx: BlenderContext, center: BpyObject) -> None:
    """Replace the extra scene lights with randomly placed ones around ``center``."""
    for light in ctx.extra_lights:
        with contextlib.suppress(BaseException):
            bpy.data.objects.remove(light, do_unlink=True)
    ctx.extra_lights = []

    # Template banks want fixed illumination: random lights are the only
    # light source in a dark world, so their 1-15 energy range swings
    # object brightness by an order of magnitude between templates.
    n_lights = ctx.rng.randint(0, getattr(ctx.settings, "random_lights_max", 3))
    for i in range(n_lights):
        light_type = ctx.rng.choice(["POINT", "AREA", "SPOT"])
        light_data = bpy.data.lights.new(f"RandLight_{i}", type=light_type)
        light_data.energy = ctx.rng.uniform(1.0, 15.0)
        if light_type == "AREA":
            light_data.size = ctx.rng.uniform(0.1, 1.0)
        light_data.color = (
            ctx.rng.uniform(0.7, 1.0),
            ctx.rng.uniform(0.7, 1.0),
            ctx.rng.uniform(0.7, 1.0),
        )
        light_data.use_shadow = True

        light_obj = bpy.data.objects.new(f"RandLight_{i}", light_data)
        ctx.scene.collection.objects.link(light_obj)
        light_obj.location = center + Vector(
            (
                ctx.rng.uniform(-0.5, 0.5),
                ctx.rng.uniform(-0.5, 0.5),
                ctx.rng.uniform(0.1, 0.6),
            )
        )
        ctx.extra_lights.append(light_obj)


def update_random_ground(ctx: BlenderContext, ground_z: BpyObject) -> None:
    """Refresh the ground, including removing a plane when no-ground is sampled."""
    if ctx.settings.no_ground_prob > 0.0 and ctx.rng.random() < ctx.settings.no_ground_prob:
        if ctx.ground_obj is not None:
            bpy.data.objects.remove(ctx.ground_obj, do_unlink=True)
            ctx.ground_obj = None
        return
    use_ground_image = (
        bool(ctx.settings.ground_image_files)
        and ctx.settings.ground_image_prob > 0.0
        and ctx.rng.random() < ctx.settings.ground_image_prob
    )
    if use_ground_image:
        setup_ground_plane(
            ctx,
            None,
            ground_z,
            ctx.scene_scale,
            use_ground_image=True,
        )
    elif (
        ctx.settings.ground_style == "procedural"
        or ctx.settings.ground_style == "mixed"
        and (not ctx.texture_dirs or ctx.rng.random() < 0.5)
        or ctx.settings.ground_style == "texture"
        and not ctx.texture_dirs
    ):
        setup_ground_plane(ctx, None, ground_z, ctx.scene_scale, force_procedural=True)
    elif ctx.texture_dirs:
        setup_ground_plane(ctx, ctx.rng.choice(ctx.texture_dirs), ground_z, ctx.scene_scale)
