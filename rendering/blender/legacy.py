# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Historical random scene sampling, kept separate from deterministic records."""

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import math

import bpy
import bpy_extras
from mathutils import Quaternion as MQuaternion
from mathutils import Vector

from .annotations import (
    count_unoccluded_keypoints,
)
from .camera import (
    axis_facing_quaternion_with_jiggle,
    get_camera_direction_top_down,
    random_camera_direction,
)
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


def legacy_random_quaternion(ctx: BlenderContext) -> MQuaternion:
    """Frozen three-draw sampler for historical in-worker random rendering only.

    Active dataset sampling uses SciPy in doper.scene.geometry. Keeping this
    implementation isolated preserves old worker RNG sequences without adding
    SciPy to Blender or pretending these legacy scenes match the current sampler.
    """
    u1, u2, u3 = ctx.rng.random(), ctx.rng.random(), ctx.rng.random()
    qx = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
    qy = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
    qz = math.sqrt(u1) * math.sin(2 * math.pi * u3)
    qw = math.sqrt(u1) * math.cos(2 * math.pi * u3)
    return MQuaternion((qw, qx, qy, qz))


def legacy_object_pose(ctx: BlenderContext) -> tuple[MQuaternion, bool]:
    """Return (quaternion, is_biased_pose).

    Samples from the categorical {x-facing, y-facing, z-facing, random}
    according to ``axis_bias`` probabilities.
    """
    r = ctx.rng.random()
    c = 0.0
    for axis in ("x", "y", "z"):
        c += ctx.settings.axis_bias[axis]
        if r < c:
            return axis_facing_quaternion_with_jiggle(ctx, axis, ctx.settings.axis_jiggle), True
    return legacy_random_quaternion(ctx), False


def realize_random_scene(ctx: BlenderContext, local_idx: int) -> None:
    """Sample a fresh legacy scene; this path deliberately does not support resume."""
    # === 1. Randomize object pose ===
    ctx.mesh_obj.location = Vector((0, 0, 0))
    ctx.mesh_obj.rotation_mode = "QUATERNION"
    pose_quat, is_biased_pose = legacy_object_pose(ctx)
    ctx.mesh_obj.rotation_quaternion = pose_quat

    bpy.context.view_layer.update()

    rotated_bbox = [ctx.mesh_obj.matrix_world @ Vector(c) for c in ctx.mesh_obj.bound_box]
    min_z_after_rotation = min(v.z for v in rotated_bbox)

    # Scale ground clearance and XY jitter with scene size
    ground_clearance = (
        ctx.rng.uniform(ctx.settings.ground_clearance_min, ctx.settings.ground_clearance_max)
        * ctx.scene_scale
    )
    lift_z = -min_z_after_rotation + ground_clearance

    scaled_xy_jitter = ctx.settings.xy_jitter * ctx.scene_scale
    obj_pos = Vector(
        (
            ctx.rng.uniform(-scaled_xy_jitter, scaled_xy_jitter),
            ctx.rng.uniform(-scaled_xy_jitter, scaled_xy_jitter),
            lift_z,
        )
    )
    ctx.mesh_obj.location = obj_pos
    bpy.context.view_layer.update()

    current_bbox = [ctx.mesh_obj.matrix_world @ Vector(c) for c in ctx.mesh_obj.bound_box]
    current_center = sum(current_bbox, Vector()) / 8.0
    current_bbox_min_z = min(v.z for v in current_bbox)

    # === 2. Randomize material ===
    if local_idx % ctx.settings.change_material_every == 0:
        randomize_object_material(ctx)

    # === 3. Random world/background ===
    if local_idx % ctx.settings.change_hdri_every == 0:
        setup_world_background(ctx)

    # === 4. Random ground plane ===
    if local_idx % ctx.settings.change_ground_every == 0:
        update_random_ground(ctx, current_bbox_min_z)

    # === 5. Random asset distractors ===
    if local_idx % ctx.settings.change_distractors_every == 0:
        clear_distractors(ctx)
        if ctx.distractor_files:
            load_distractors(
                ctx,
                ctx.rng.randint(ctx.settings.num_distractors_min, ctx.settings.num_distractors_max),
                current_center,
                ctx.scene_scale,
            )

    # === 6. Random lights ===
    setup_random_lights(ctx, current_center)

    # === 7. Camera positioning with visibility check ===
    cam_dist = ctx.rng.uniform(ctx.settings.cam_dist_min, ctx.settings.cam_dist_max)
    jitter_strength = ctx.rng.uniform(ctx.settings.image_jitter_min, ctx.settings.image_jitter_max)

    def setup_camera(cam_dir: BpyObject) -> None:
        """Place the camera for one candidate view direction."""
        cam_pos = current_center + cam_dir * cam_dist
        ctx.cam_obj.location = cam_pos

        if jitter_strength > 0:
            jitter_angle = ctx.rng.uniform(0, 2 * math.pi)
            cam_right = cam_dir.cross(Vector((0, 0, 1))).normalized()
            if cam_right.length < 0.01:
                cam_right = Vector((1, 0, 0))
            cam_up = cam_right.cross(cam_dir).normalized()
            jitter_offset = (
                (cam_right * math.cos(jitter_angle) + cam_up * math.sin(jitter_angle))
                * jitter_strength
                * cam_dist
                * 0.5
            )
            look_at = current_center + jitter_offset
        else:
            look_at = current_center

        direction = look_at - cam_pos
        ctx.cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()

    def check_visibility() -> bool:
        """Return whether enough keypoints are visible for the current camera."""
        visible_count = 0
        for kp in ctx.kp_objects:
            kp_world = kp.matrix_world.translation.copy()
            co_ndc = bpy_extras.object_utils.world_to_camera_view(ctx.scene, ctx.cam_obj, kp_world)
            if 0 < co_ndc.x < 1 and 0 < co_ndc.y < 1 and co_ndc.z > 0:
                visible_count += 1
        return visible_count >= ctx.settings.min_visible_keypoints

    if is_biased_pose:
        cam_dir = get_camera_direction_top_down(ctx, ctx.settings.top_down_jiggle)
        setup_camera(cam_dir)
        if ctx.settings.fallback_to_random and not check_visibility():
            cam_dir = random_camera_direction(ctx)
            setup_camera(cam_dir)
    else:
        cam_dir = random_camera_direction(ctx)
        setup_camera(cam_dir)

    # === 8. Camera-relative procedural foreground/background distractors ===
    if (
        local_idx % ctx.settings.change_distractors_every == 0
        and ctx.settings.procedural_distractors
    ):
        base_distractor_count = len(ctx.loaded_distractors)
        attempts = max(1, ctx.settings.procedural_distractor_max_attempts)
        best_unoccluded = -1
        best_attempt = 0
        best_state = None
        accepted = False
        for attempt in range(1, attempts + 1):
            attempt_state = ctx.rng.getstate()
            clear_distractors(ctx, base_distractor_count)
            load_procedural_distractors(ctx, current_center, cam_dir, current_bbox_min_z)
            bpy.context.view_layer.update()
            unoccluded = count_unoccluded_keypoints(ctx)
            if unoccluded > best_unoccluded:
                best_unoccluded = unoccluded
                best_attempt = attempt
                best_state = attempt_state
            if unoccluded >= ctx.settings.procedural_distractor_min_unoccluded_keypoints:
                accepted = True
                break
        if not accepted and best_state is not None:
            clear_distractors(ctx, base_distractor_count)
            ctx.rng.setstate(best_state)
            load_procedural_distractors(ctx, current_center, cam_dir, current_bbox_min_z)
            bpy.context.view_layer.update()
        if not accepted:
            print(
                f"[W{ctx.args.worker_id}] warning: procedural distractors left only "
                f"{best_unoccluded}/{ctx.settings.num_keypoints} unoccluded keypoints "
                f"after {attempts} attempts",
                flush=True,
            )
        elif best_attempt > 1:
            print(
                f"[W{ctx.args.worker_id}] resampled procedural distractors "
                f"{best_attempt}x; {best_unoccluded}/{ctx.settings.num_keypoints} "
                "keypoints unoccluded",
                flush=True,
            )
