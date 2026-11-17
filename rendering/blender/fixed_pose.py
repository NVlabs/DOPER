# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed OpenCV-pose preview rendering for existing operator workflows."""

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import json
import os

import bpy
from mathutils import Matrix, Vector

from .annotations import count_unoccluded_keypoints
from .camera import set_if_present
from .context import BlenderContext, BpyObject
from .distractors import (
    clear_distractors,
    load_distractors,
    load_procedural_distractors,
)
from .environment import (
    setup_random_lights,
    setup_world_background,
    update_random_ground,
)
from .materials import randomize_object_material


def rodrigues_to_matrix(rvec: BpyObject) -> BpyObject:
    """Convert an OpenCV rotation vector to a 3x3 rotation matrix."""
    axis = Vector((float(rvec[0]), float(rvec[1]), float(rvec[2])))
    angle = axis.length
    if angle < 1.0e-12:
        return Matrix.Identity(3)
    return Matrix.Rotation(angle, 3, axis / angle)


def setup_real_camera_intrinsics(ctx: BlenderContext, params: BpyObject) -> None:
    """Configure the camera from OpenCV pinhole intrinsics in ``params``."""
    fx_val = float(params["fx"])
    fy_val = float(params["fy"])
    cx_val = float(params["cx"])
    cy_val = float(params["cy"])
    width = int(params["width"])
    height = int(params["height"])

    ctx.cam_data.sensor_fit = "HORIZONTAL"
    ctx.cam_data.sensor_width = float(width)
    ctx.cam_data.sensor_height = float(height)
    ctx.cam_data.lens = fx_val
    ctx.cam_data.shift_x = (width / 2.0 - cx_val) / width
    ctx.cam_data.shift_y = (cy_val - height / 2.0) / width

    ctx.scene.render.resolution_x = width
    ctx.scene.render.resolution_y = height
    ctx.scene.render.resolution_percentage = 100
    ctx.scene.render.pixel_aspect_x = 1.0
    ctx.scene.render.pixel_aspect_y = fx_val / fy_val


def setup_full_scene_at_pose(
    ctx: BlenderContext,
    current_center: BpyObject,
    current_bbox_min_z: BpyObject,
    cam_dir: BpyObject,
    *,
    full_scene: bool = True,
) -> None:
    """Randomize materials/background/ground/distractors/lights around a fixed pose."""
    randomize_object_material(ctx)
    setup_world_background(ctx)

    if not full_scene:
        if ctx.ground_obj is not None:
            bpy.data.objects.remove(ctx.ground_obj, do_unlink=True)
            ctx.ground_obj = None
        clear_distractors(ctx)
        setup_random_lights(ctx, current_center)
        return

    update_random_ground(ctx, current_bbox_min_z)

    clear_distractors(ctx)
    if ctx.distractor_files:
        load_distractors(
            ctx,
            ctx.rng.randint(ctx.settings.num_distractors_min, ctx.settings.num_distractors_max),
            current_center,
            ctx.scene_scale,
        )

    setup_random_lights(ctx, current_center)

    if ctx.settings.procedural_distractors:
        base_distractor_count = len(ctx.loaded_distractors)
        attempts = max(1, ctx.settings.procedural_distractor_max_attempts)
        best_unoccluded = -1
        best_state = None
        accepted = False
        for _attempt in range(attempts):
            attempt_state = ctx.rng.getstate()
            clear_distractors(ctx, base_distractor_count)
            load_procedural_distractors(ctx, current_center, cam_dir, current_bbox_min_z)
            bpy.context.view_layer.update()
            unoccluded = count_unoccluded_keypoints(ctx)
            if unoccluded > best_unoccluded:
                best_unoccluded = unoccluded
                best_state = attempt_state
            if unoccluded >= ctx.settings.procedural_distractor_min_unoccluded_keypoints:
                accepted = True
                break
        if not accepted and best_state is not None:
            clear_distractors(ctx, base_distractor_count)
            ctx.rng.setstate(best_state)
            load_procedural_distractors(ctx, current_center, cam_dir, current_bbox_min_z)
            bpy.context.view_layer.update()


def render_fixed_opencv_pose(ctx: BlenderContext, params: BpyObject, render_out: BpyObject) -> None:
    """Render one preview of an OpenCV rvec/tvec pose described by ``params``."""
    scene_seed = int(params.get("scene_seed", ctx.args.seed))
    ctx.rng.seed(scene_seed)

    rotation = rodrigues_to_matrix(params["rvec"])
    tvec = params["tvec"]
    pose = rotation.to_4x4()
    pose.translation = Vector((float(tvec[0]), float(tvec[1]), float(tvec[2])))

    blend_cam = bpy.data.objects.get("Camera")
    if blend_cam is not None:
        blend_cam.hide_render = True

    if ctx.cam_obj is not None:
        bpy.data.objects.remove(ctx.cam_obj, do_unlink=True)
    ctx.cam_data = bpy.data.cameras.new("FixedPoseCam")
    ctx.cam_obj = bpy.data.objects.new("FixedPoseCam", ctx.cam_data)
    ctx.scene.collection.objects.link(ctx.cam_obj)
    ctx.scene.camera = ctx.cam_obj
    ctx.cam_obj.matrix_world = Matrix(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, -1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    setup_real_camera_intrinsics(ctx, params)

    corners = [pose @ Vector(corner) for corner in ctx.mesh_obj.bound_box]
    current_center = sum(corners, Vector()) / 8.0
    current_bbox_min_z = min(corner.z for corner in corners)
    cam_dir = (
        current_center.normalized() if current_center.length > 1.0e-6 else Vector((0.0, 0.0, -1.0))
    )

    full_scene = bool(params.get("full_scene", True))
    setup_full_scene_at_pose(
        ctx, current_center, current_bbox_min_z, cam_dir, full_scene=full_scene
    )

    ctx.mesh_obj.matrix_world = pose
    ctx.cam_obj.matrix_world = Matrix(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, -1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    ctx.scene.camera = ctx.cam_obj
    bpy.context.view_layer.update()

    samples = int(params.get("samples", ctx.eevee_samples))
    set_if_present(ctx.eevee, "taa_render_samples", samples)

    ctx.scene.render.film_transparent = bool(params.get("transparent", False))
    hide_mesh = bool(params.get("hide_mesh", False))
    ctx.mesh_obj.hide_render = hide_mesh
    ctx.mesh_obj.hide_viewport = hide_mesh

    ctx.scene.render.filepath = render_out
    bpy.ops.render.render(write_still=True)
    print(f"[fixed_pose] wrote {render_out} (scene_seed={scene_seed})")


def render_fixed_requests(ctx: BlenderContext) -> bool:
    """Handle one-off preview requests before entering dataset rendering."""
    if ctx.args.fixed_pose_manifest:
        with open(ctx.args.fixed_pose_manifest) as f:
            manifest_entries = json.load(f)
        if isinstance(manifest_entries, dict):
            manifest_entries = manifest_entries.get("entries", [])
        n_entries = len(manifest_entries)
        print(f"[fixed_pose] batch manifest: {n_entries} entries", flush=True)
        for entry_idx, entry in enumerate(manifest_entries):
            out_path = entry["out"]
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            render_fixed_opencv_pose(ctx, entry["params"], out_path)
            print(f"[fixed_pose] batch progress {entry_idx + 1}/{n_entries}", flush=True)
        return True

    if ctx.args.fixed_pose_params:
        if not ctx.args.render_out:
            raise SystemExit("--fixed_pose_params requires --render_out")
        with open(ctx.args.fixed_pose_params) as f:
            single_params = json.load(f)
        render_fixed_opencv_pose(ctx, single_params, ctx.args.render_out)
        return True

    return False
