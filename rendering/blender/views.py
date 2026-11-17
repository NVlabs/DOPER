# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset shard execution and checkpoint cadence."""

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import os

import bpy
from mathutils import Vector

from .annotations import (
    _should_check_kp_mask,
    annotate_robustness,
    capture_annotation,
    check_keypoints_against_mask,
    render_visible_object_mask,
)
from .context import BlenderContext


def render_views(ctx: BlenderContext, sample_index: int, local_index: int) -> tuple[str, list[str]]:
    """Render and annotate all views of one sample before checkpointing it."""
    base_position = ctx.cam_obj.location.copy()
    base_rotation = ctx.cam_obj.rotation_euler.copy()
    right = ctx.cam_obj.matrix_world.to_quaternion() @ Vector((1, 0, 0))
    views = [("mono", 0.0, 0)]
    if ctx.settings.stereo_enabled:
        views = [
            (ctx.settings.stereo_left_suffix, -0.5, 0),
            (ctx.settings.stereo_right_suffix, 0.5, 1),
        ]
    visible_counts = []
    for name, offset, camera_index in views:
        if ctx.settings.stereo_enabled:
            ctx.cam_obj.location = base_position + right * (ctx.settings.stereo_baseline_m * offset)
            ctx.cam_obj.rotation_euler = base_rotation
            bpy.context.view_layer.update()
        suffix = f"_{name}" if ctx.settings.stereo_enabled else ""
        filename = f"{sample_index:06d}{suffix}.png"
        ctx.scene.render.filepath = os.path.join(ctx.args.outdir, "images", filename)
        bpy.ops.render.render(write_still=True)
        mask_relative = None
        if ctx.settings.render_masks:
            mask_relative = f"{ctx.settings.mask_dir_name}/{filename}"
            mask_path = os.path.join(ctx.args.outdir, mask_relative)
            render_visible_object_mask(ctx, mask_path)
        image_id = sample_index * ctx.settings.images_per_sample + camera_index
        count = capture_annotation(
            ctx, image_id, filename, sample_index, name, camera_index, mask_relative
        )
        robust = ctx.scene_records.get(sample_index, {}).get("robustness")
        if robust is not None:
            # load_scene_records guarantees render_masks for robustness records.
            annotate_robustness(ctx, ctx.annotations[-1], mask_path, robust)
        if mask_relative is not None and _should_check_kp_mask(ctx, local_index):
            check_keypoints_against_mask(ctx, mask_path, ctx.annotations[-1]["keypoints"], filename)
        label = f"{name}:" if ctx.settings.stereo_enabled else ""
        visible_counts.append(f"{label}{count}/{ctx.settings.num_keypoints}")
    return filename, visible_counts
