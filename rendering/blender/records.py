# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Realize deterministic scene records with cadence-independent placement."""

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import contextlib
import hashlib
import json
import math

import bpy
from mathutils import Euler, Matrix, Vector

from rendering.appearance import blender_field_seed, sample_light_energy
from rendering.paths import resolve_under

from .context import BlenderContext, BpyObject
from .distractors import clear_distractors, has_valid_material
from .environment import setup_ground_plane, setup_hdri
from .materials import (
    find_or_create_object_bsdf,
    randomize_aluminum_material,
    remove_input_links,
)
from .robust_appearance import apply_appearance


def seed_rng_for_record(
    ctx: BlenderContext, record: BpyObject, field: str = "sample", index: BpyObject = None
) -> None:
    """Seed the worker RNG for ``record`` according to the replay-ablation mode."""
    if getattr(ctx.args, "experimental_replay_ablation", "none") in ("appearance-rng", "both"):
        # Historical renderer consumed one RNG stream per sample across all fields.
        if field == "sample":
            digest = hashlib.sha256(f"{record['seed']}:blender:{record['index']}".encode()).digest()
            ctx.rng.seed(int.from_bytes(digest[:8], "big"))
        return
    index = record["index"] if index is None else index
    ctx.rng.seed(blender_field_seed(record["seed"], field, index))


def record_asset(ctx: BlenderContext, rel: str) -> str:
    """Resolve an asset path stored in a scene record against the config directory."""
    return resolve_under(ctx.settings.base_dir, rel)


def apply_record_object_pose(ctx: BlenderContext, record: BpyObject) -> None:
    """Apply the object pose stored in ``record``."""
    pose = Matrix([list(map(float, row)) for row in record["object"]["pose"]])
    ctx.mesh_obj.rotation_mode = "QUATERNION"
    ctx.mesh_obj.rotation_quaternion = pose.to_3x3().to_quaternion()
    ctx.mesh_obj.location = pose.to_translation()
    bpy.context.view_layer.update()


def apply_record_camera(ctx: BlenderContext, record: BpyObject) -> None:
    """Apply the camera pose stored in ``record``."""
    ctx.cam_obj.matrix_world = Matrix(
        [list(map(float, row)) for row in record["camera"]["cam_to_world"]]
    )
    bpy.context.view_layer.update()


def apply_record_material(ctx: BlenderContext, record: BpyObject) -> None:
    """Apply the object material stored in ``record``."""
    color = tuple(float(c) for c in record["material"]["color"]) + (1.0,)
    if ctx.settings.object_material_mode == "override":
        mat = bpy.data.materials.get("ObjectOverrideMaterial") or bpy.data.materials.new(
            "ObjectOverrideMaterial"
        )
        mat.use_nodes = True
        ctx.mesh_obj.data.materials.clear()
        ctx.mesh_obj.data.materials.append(mat)
        for polygon in ctx.mesh_obj.data.polygons:
            polygon.material_index = 0
        bsdf = find_or_create_object_bsdf(mat)
        for name in ("Base Color", "Metallic", "Roughness"):
            remove_input_links(mat.node_tree, bsdf.inputs[name])
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Metallic"].default_value = float(record["material"]["metallic"])
        bsdf.inputs["Roughness"].default_value = float(record["material"]["roughness"])
        mat.diffuse_color = color
        return
    randomize_aluminum_material(ctx)  # brush pattern from the per-sample rng
    mat = ctx.mesh_obj.data.materials[0] if ctx.mesh_obj.data.materials else None
    bsdf = mat.node_tree.nodes.get("MainBSDF") if mat and mat.node_tree else None
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color


def apply_record_background(ctx: BlenderContext, record: BpyObject) -> None:
    """Apply the world background stored in ``record``."""
    bg = record["background"]
    if bg["mode"] == "hdri":
        setup_hdri(ctx, record_asset(ctx, bg["hdri"]))
        mapping = next(n for n in ctx.scene.world.node_tree.nodes if n.type == "MAPPING")
        mapping.inputs["Rotation"].default_value = (
            0.0,
            0.0,
            math.radians(bg["rotation_deg"]),
        )
        return
    strength_range = {"dark": (0.25, 0.9), "studio": (0.8, 0.8)}.get(
        bg["mode"], ctx.settings.world_background_strength_range
    )
    world = bpy.data.worlds.get("SynthWorld") or bpy.data.worlds.new("SynthWorld")
    ctx.scene.world = world
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    nodes.clear()
    node_bg = nodes.new("ShaderNodeBackground")
    output = nodes.new("ShaderNodeOutputWorld")
    node_bg.inputs["Color"].default_value = tuple(float(c) for c in bg["color"]) + (1.0,)
    node_bg.inputs["Strength"].default_value = ctx.rng.uniform(*strength_range)
    links.new(node_bg.outputs["Background"], output.inputs["Surface"])


def apply_record_ground(ctx: BlenderContext, record: BpyObject, ground_z: BpyObject) -> None:
    """Apply the ground plane stored in ``record``."""
    ground = record["ground"]
    if not ground["enabled"]:
        if ctx.ground_obj is not None:
            bpy.data.objects.remove(ctx.ground_obj, do_unlink=True)
            ctx.ground_obj = None
        return
    if ground["kind"] == "texture":
        texture_set = record_asset(ctx, ground["texture_set"])
        setup_ground_plane(ctx, texture_set, ground_z, ctx.scene_scale)
        # Match the record's tiling: Isaac repeats the texture texture_scale times over a plane
        # three times as wide as Blender's, so the same texel size needs a third of the repeats.
        mapping = next(
            (n for n in ctx.ground_obj.data.materials[0].node_tree.nodes if n.type == "MAPPING"),
            None,
        )
        if mapping is not None:
            repeats = ground["texture_scale"] / 3.0
            mapping.inputs["Scale"].default_value = (repeats, repeats, 1.0)
            mapping.inputs["Rotation"].default_value = (
                0.0,
                0.0,
                math.radians(ground["texture_rotate_deg"]),
            )
        return
    if ctx.ground_obj is not None:
        bpy.data.objects.remove(ctx.ground_obj, do_unlink=True)
    bpy.ops.mesh.primitive_plane_add(
        size=2.0 * ctx.scene_scale, location=(0, 0, ground_z - 0.001 * ctx.scene_scale)
    )
    ctx.ground_obj = bpy.context.active_object
    ctx.ground_obj.name = "GroundPlane"
    mat = bpy.data.materials.new("GroundMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = tuple(float(c) for c in ground["color"]) + (1.0,)
    bsdf.inputs["Roughness"].default_value = float(ground["roughness"])
    ctx.ground_obj.data.materials.append(mat)


def record_fallback_material(look: BpyObject, index: BpyObject) -> BpyObject:
    """Return the fallback material for the record look at ``index``."""
    mat = bpy.data.materials.new(name=f"RecordDistractorMat_{index}")
    mat.use_nodes = True
    bsdf = find_or_create_object_bsdf(mat)
    bsdf.inputs["Base Color"].default_value = tuple(float(c) for c in look["color"]) + (1.0,)
    bsdf.inputs["Metallic"].default_value = float(look["metallic"])
    bsdf.inputs["Roughness"].default_value = float(look["roughness"])
    return mat


def load_record_distractors(ctx: BlenderContext, record: BpyObject) -> None:
    """Load the distractor objects listed in ``record``."""
    clear_distractors(ctx)
    ctx.record_distractor_instances = []
    for i, spec in enumerate(record["distractors"]):
        dist_file = record_asset(ctx, spec["file"])
        before_objs = set(bpy.data.objects.keys())
        try:
            if dist_file.lower().endswith((".glb", ".gltf")):
                bpy.ops.import_scene.gltf(filepath=dist_file)
            else:
                bpy.ops.wm.usd_import(filepath=dist_file)
        except Exception as e:
            print(f"Warning: failed to import distractor {dist_file}: {e}")
            continue
        new_objs = [bpy.data.objects[n] for n in sorted(set(bpy.data.objects.keys()) - before_objs)]
        mesh_objs = [o for o in new_objs if o.type == "MESH"]
        if not mesh_objs:
            for obj in new_objs:
                bpy.data.objects.remove(obj, do_unlink=True)
            continue
        # One rigid placement for the whole asset, matching Isaac's single reference prim.
        bpy.context.view_layer.update()
        corners = [o.matrix_world @ Vector(c) for o in mesh_objs for c in o.bound_box]
        size = max(max(v[k] for v in corners) - min(v[k] for v in corners) for k in range(3))
        scale = spec["size_frac"] * ctx.mesh_size / size if size > 0 else 1.0
        place = (
            Matrix.Translation(Vector(spec["position"]))
            @ Euler(tuple(spec["rotation_xyz"]), "XYZ").to_matrix().to_4x4()
            @ Matrix.Scale(scale, 4)
        )
        authored_worlds = [(obj, obj.matrix_world.copy()) for obj in mesh_objs]
        for obj, original_world in authored_worlds:
            # Detach imported hierarchy while retaining the authored world transform.
            obj.parent = None
            obj.matrix_world = place @ original_world
            ctx.record_distractor_instances.append((i, obj, original_world, scale))
            if not has_valid_material(obj):
                obj.data.materials.clear()
                obj.data.materials.append(record_fallback_material(spec["material"], i))
            ctx.loaded_distractors.append(obj)
        for obj in new_objs:
            if obj.type != "MESH":
                with contextlib.suppress(Exception):
                    bpy.data.objects.remove(obj, do_unlink=True)


def apply_record_transforms(ctx: BlenderContext, record: BpyObject, ground_z: BpyObject) -> None:
    """Apply sample-specific placement even when material/asset caches are reused."""
    if ctx.ground_obj is not None:
        ctx.ground_obj.location.z = ground_z - 0.001 * ctx.scene_scale
    for index, obj, original_world, scale in ctx.record_distractor_instances:
        spec = record["distractors"][index]
        place = (
            Matrix.Translation(Vector(spec["position"]))
            @ Euler(tuple(spec["rotation_xyz"]), "XYZ").to_matrix().to_4x4()
            @ Matrix.Scale(scale, 4)
        )
        obj.matrix_world = place @ original_world
    bpy.context.view_layer.update()


def apply_record_lights(ctx: BlenderContext, record: BpyObject) -> None:
    """Replace the extra lights with those listed in ``record``."""
    for light in ctx.extra_lights:
        with contextlib.suppress(Exception):
            bpy.data.objects.remove(light, do_unlink=True)
    ctx.extra_lights = []
    kinds = {"sphere": "POINT", "rect": "AREA", "disk": "SPOT"}
    for i, spec in enumerate(record["lights"]):
        light_data = bpy.data.lights.new(f"RandLight_{i}", type=kinds.get(spec["kind"], "POINT"))
        light_data.energy = sample_light_energy(ctx.rng)
        if light_data.type == "AREA":
            light_data.size = float(spec["scale"])
        light_data.color = tuple(float(c) for c in spec["color"])
        light_data.use_shadow = True
        light_obj = bpy.data.objects.new(f"RandLight_{i}", light_data)
        ctx.scene.collection.objects.link(light_obj)
        light_obj.location = Vector(spec["position"])
        if light_data.type in {"AREA", "SPOT"}:
            direction = Vector(record["object"]["center"]) - light_obj.location
            light_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        ctx.extra_lights.append(light_obj)


def load_scene_records(ctx: BlenderContext) -> None:
    """Load and validate the ``--scenes`` JSONL records into ``ctx.scene_records``."""
    records = {}
    export_environment = getattr(ctx.args, "export_environment", False)
    if ctx.args.scenes:
        with open(ctx.args.scenes) as f:
            for line_number, line in enumerate(f, 1):
                if line.strip():
                    record = json.loads(line)
                    uses_robustness = record.get("robustness") is not None
                    if uses_robustness and not ctx.settings.render_masks:
                        raise ValueError("robustness requires output.render_masks: true")
                    if export_environment and (
                        uses_robustness or record.get("appearance") is not None
                    ):
                        raise ValueError(
                            "robustness and appearance do not support environment export"
                        )
                    index = record["index"]
                    if index in records:
                        raise ValueError(
                            f"duplicate scene record index {index} in "
                            f"{ctx.args.scenes}:{line_number}"
                        )
                    records[index] = record
        print(f"Scenes: {len(records)} records from {ctx.args.scenes}")
        if ctx.settings.procedural_distractors:
            print("note: procedural_distractors are disabled in scenes mode (no record equivalent)")
        if ctx.settings.ground_image_files and ctx.settings.ground_image_prob > 0.0:
            print("note: ground images are disabled in scenes mode (no record equivalent)")

    ctx.scene_records = records


def realize_record(ctx: BlenderContext, record: BpyObject, cadence: BpyObject) -> None:
    """Apply cached appearance and per-sample transforms from one scene record."""
    view_idx = record["index"]
    seed_rng_for_record(ctx, record)
    apply_record_object_pose(ctx, record)
    current_bbox_min_z = float(record["object"]["bbox_min_z"])
    if cadence.changed("material", view_idx, ctx.settings.change_material_every):
        seed_rng_for_record(
            ctx,
            record,
            "material",
            view_idx // max(1, ctx.settings.change_material_every),
        )
        apply_record_material(ctx, record)
    if cadence.changed("background", view_idx, ctx.settings.change_hdri_every):
        seed_rng_for_record(
            ctx,
            record,
            "background",
            view_idx // max(1, ctx.settings.change_hdri_every),
        )
        apply_record_background(ctx, record)
    if cadence.changed("ground", view_idx, ctx.settings.change_ground_every):
        seed_rng_for_record(
            ctx, record, "ground", view_idx // max(1, ctx.settings.change_ground_every)
        )
        apply_record_ground(ctx, record, current_bbox_min_z)
    if cadence.changed("distractors", view_idx, ctx.settings.change_distractors_every):
        seed_rng_for_record(
            ctx,
            record,
            "distractors",
            view_idx // max(1, ctx.settings.change_distractors_every),
        )
        load_record_distractors(ctx, record)
    if getattr(ctx.args, "experimental_replay_ablation", "none") not in ("placement", "both"):
        apply_record_transforms(ctx, record, current_bbox_min_z)
    seed_rng_for_record(ctx, record, "lights")
    apply_record_lights(ctx, record)
    apply_record_camera(ctx, record)
    apply_record_robustness(ctx, record)
    apply_appearance(ctx, record)


def apply_record_robustness(ctx: BlenderContext, record: BpyObject) -> None:
    """Realize foreground capsules as mesh geometry shared by RGB, masks and ray casts.

    Proxies from the previous record are always removed first, so records without
    a ``robustness`` section render with no proxies.
    """
    for obj in ctx.robustness_objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    ctx.robustness_objects = []
    for index, spec in enumerate(record.get("robustness", {}).get("proxies", [])):
        look = {"color": spec["color"], "metallic": 0, "roughness": spec["roughness"]}
        material = record_fallback_material(look, f"grasp_{index}")
        for piece_index, obj in enumerate(add_capsule(spec)):
            obj.name = f"GraspProxy_{index}_{piece_index}"
            obj.data.materials.append(material)
            for poly in obj.data.polygons:
                poly.use_smooth = True
            ctx.robustness_objects.append(obj)
    bpy.context.view_layer.update()


def add_capsule(spec: BpyObject) -> BpyObject:
    """Add a capsule from ``spec`` start/end/radius as a cylinder plus two end spheres.

    Returns the three new objects; separate primitives avoid a mesh boolean.
    """
    start, end = Vector(spec["start"]), Vector(spec["end"])
    radius = float(spec["radius"])
    direction = end - start
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=24, radius=radius, depth=direction.length, location=(start + end) / 2
    )
    cylinder = bpy.context.object
    cylinder.rotation_mode = "QUATERNION"
    cylinder.rotation_quaternion = direction.to_track_quat("Z", "Y")
    pieces = [cylinder]
    for point in (start, end):
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=24, ring_count=12, radius=radius, location=point
        )
        pieces.append(bpy.context.object)
    return pieces
