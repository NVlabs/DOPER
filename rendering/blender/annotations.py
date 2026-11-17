# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visible masks, keypoint projection and COCO annotation checks."""

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import contextlib
import os

import bpy
import bpy_extras
import numpy as np

from .context import BlenderContext, BpyObject


def make_emission_material(name: BpyObject, color: BpyObject) -> BpyObject:
    """Create an emission material of a flat color for mask rendering."""
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        # Cached and reused across iterations, but sits with zero real users
        # in between (detached from every object once the mask render is
        # done), so orphans_purge() would otherwise delete it mid-run.
        mat.use_fake_user = True
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    emission = nodes.new("ShaderNodeEmission")
    output = nodes.new("ShaderNodeOutputMaterial")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = 1.0
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    mat.diffuse_color = color
    return mat


def mask_materials(ctx: BlenderContext) -> BpyObject:
    """Return the foreground and background emission materials used for masks."""
    if ctx.mask_target_material is None:
        ctx.mask_target_material = make_emission_material(
            "VisibleMask_TargetWhite", (1.0, 1.0, 1.0, 1.0)
        )
    if ctx.mask_background_material is None:
        ctx.mask_background_material = make_emission_material(
            "VisibleMask_BackgroundBlack", (0.0, 0.0, 0.0, 1.0)
        )
    return ctx.mask_target_material, ctx.mask_background_material


def assign_single_material(obj: BpyObject, material: BpyObject) -> None:
    """Replace all material slots of ``obj`` with the single ``material``."""
    obj.data.materials.clear()
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.material_index = 0


def render_visible_object_mask(ctx: BlenderContext, mask_path: BpyObject) -> None:
    """Render a visible target-object mask using the current camera and scene."""
    target_mat, background_mat = mask_materials(ctx)
    material_state = []
    render = ctx.scene.render
    image_settings = render.image_settings
    view_settings = ctx.scene.view_settings

    old_world = ctx.scene.world
    old_filepath = render.filepath
    old_file_format = image_settings.file_format
    old_color_mode = getattr(image_settings, "color_mode", None)
    old_compression = getattr(image_settings, "compression", None)
    old_film_transparent = render.film_transparent
    old_view_transform = getattr(view_settings, "view_transform", None)
    old_look = getattr(view_settings, "look", None)
    old_exposure = getattr(view_settings, "exposure", None)
    old_gamma = getattr(view_settings, "gamma", None)

    try:
        for obj in list(bpy.data.objects):
            if obj.type != "MESH":
                continue
            # Linked instances can share geometry while requiring opposite mask
            # colors. Never edit their original mesh or its polygon indices.
            original_mesh = obj.data
            slot_state = [(slot.link, slot.material) for slot in obj.material_slots]
            mask_mesh = original_mesh.copy()
            material_state.append((obj, original_mesh, slot_state, mask_mesh))
            obj.data = mask_mesh
            assign_single_material(obj, target_mat if obj == ctx.mesh_obj else background_mat)
            # An existing OBJECT override would otherwise hide the mask material.
            obj.material_slots[0].link = "DATA"

        mask_world = bpy.data.worlds.get("VisibleMaskWorld")
        if mask_world is None:
            mask_world = bpy.data.worlds.new("VisibleMaskWorld")
            # Same reuse-with-zero-real-users situation as the mask materials above.
            mask_world.use_fake_user = True
        ctx.scene.world = mask_world
        mask_world.use_nodes = True
        nodes = mask_world.node_tree.nodes
        links = mask_world.node_tree.links
        nodes.clear()
        bg = nodes.new("ShaderNodeBackground")
        output = nodes.new("ShaderNodeOutputWorld")
        bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
        bg.inputs["Strength"].default_value = 1.0
        links.new(bg.outputs["Background"], output.inputs["Surface"])

        render.filepath = mask_path
        render.film_transparent = False
        image_settings.file_format = "PNG"
        with contextlib.suppress(Exception):
            image_settings.color_mode = "BW"
        if hasattr(image_settings, "compression"):
            image_settings.compression = 15
        if old_view_transform is not None:
            with contextlib.suppress(Exception):
                view_settings.view_transform = "Standard"
        if old_look is not None:
            with contextlib.suppress(Exception):
                view_settings.look = "None"
        if old_exposure is not None:
            view_settings.exposure = 0.0
        if old_gamma is not None:
            view_settings.gamma = 1.0

        bpy.context.view_layer.update()
        bpy.ops.render.render(write_still=True)
    finally:
        for obj, original_mesh, slot_state, mask_mesh in reversed(material_state):
            if bpy.data.objects.get(obj.name) is not None:
                obj.data = original_mesh
                for slot, (link, material) in zip(obj.material_slots, slot_state, strict=False):
                    slot.link = link
                    if link == "OBJECT":
                        slot.material = material
            bpy.data.meshes.remove(mask_mesh)

        ctx.scene.world = old_world
        render.filepath = old_filepath
        render.film_transparent = old_film_transparent
        image_settings.file_format = old_file_format
        if old_color_mode is not None:
            with contextlib.suppress(Exception):
                image_settings.color_mode = old_color_mode
        if old_compression is not None and hasattr(image_settings, "compression"):
            image_settings.compression = old_compression
        if old_view_transform is not None:
            with contextlib.suppress(Exception):
                view_settings.view_transform = old_view_transform
        if old_look is not None:
            with contextlib.suppress(Exception):
                view_settings.look = old_look
        if old_exposure is not None:
            view_settings.exposure = old_exposure
        if old_gamma is not None:
            view_settings.gamma = old_gamma
        bpy.context.view_layer.update()


def _should_check_kp_mask(ctx: BlenderContext, local_idx: BpyObject) -> bool:
    if not (ctx.settings.render_masks and ctx.settings.keypoint_mask_check):
        return False
    if local_idx < ctx.settings.keypoint_mask_sample_first:
        return True
    return ctx.settings.keypoint_mask_sample_every > 0 and (
        local_idx % ctx.settings.keypoint_mask_sample_every == 0
    )


def _load_mask_foreground(ctx: BlenderContext, mask_path: BpyObject) -> BpyObject:
    """Load a rendered mask PNG as a top-down boolean HxW foreground array."""
    img = None
    try:
        img = bpy.data.images.load(mask_path, check_existing=False)
        w, h = img.size
        if w == 0 or h == 0:
            return None
        px = np.empty(w * h * img.channels, dtype=np.float32)
        img.pixels.foreach_get(px)
        px = px.reshape(h, w, img.channels)
        # Target object is rendered emission-white on black; any channel > 0.5
        # is foreground. Blender's pixel buffer is bottom-up, so flip to match
        # the top-down pixel coords produced by project_and_check.
        return (px[:, :, 0] > 0.5)[::-1]
    except Exception as exc:  # pragma: no cover - defensive I/O guard
        print(
            f"[W{ctx.args.worker_id}] mask-check: could not read {mask_path}: {exc}",
            flush=True,
        )
        return None
    finally:
        if img is not None:
            with contextlib.suppress(Exception):
                bpy.data.images.remove(img)


def check_keypoints_against_mask(
    ctx: BlenderContext, mask_path: BpyObject, keypoints_flat: BpyObject, file_name: BpyObject
) -> None:
    """Verify projected keypoints land on the target mask, and alert if not.

    A bounded sample of frames is checked (see keypoint_mask_sample_* config)
    so the guard adds negligible overhead. A low on-mask fraction almost always
    means keypoints_3d.json is in a different frame than the render mesh.
    """
    fg = _load_mask_foreground(ctx, mask_path)
    if fg is None:
        return
    h, w = fg.shape
    tol = max(0, ctx.settings.keypoint_mask_tol_px)
    on = total = 0
    for i in range(0, len(keypoints_flat), 3):
        x, y, v = keypoints_flat[i], keypoints_flat[i + 1], keypoints_flat[i + 2]
        # Occluded points are intentionally absent from the visible mask.
        if v != 2:
            continue
        total += 1
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - tol), min(w, xi + tol + 1)
        y0, y1 = max(0, yi - tol), min(h, yi + tol + 1)
        if x1 > x0 and y1 > y0 and fg[y0:y1, x0:x1].any():
            on += 1
    if total == 0:
        return
    frac = on / total
    ctx._kp_mask_stats["checked"] += 1
    ctx._kp_mask_stats["frac_sum"] += frac
    if frac < ctx._kp_mask_stats["worst_frac"]:
        ctx._kp_mask_stats["worst_frac"] = frac
        ctx._kp_mask_stats["worst_file"] = file_name
    if frac < ctx.settings.keypoint_mask_min_frac:
        ctx._kp_mask_stats["bad"] += 1
        if ctx._kp_mask_stats["warned"] < ctx.settings.keypoint_mask_max_warn:
            ctx._kp_mask_stats["warned"] += 1
            print(
                f"[W{ctx.args.worker_id}] WARNING keypoint/mask mismatch on {file_name}: "
                f"only {on}/{total} ({frac * 100:.0f}%) visible keypoints within {tol}px "
                f"of the target mask (need >= {ctx.settings.keypoint_mask_min_frac * 100:.0f}%). "
                f"Likely keypoints_3d.json vs render-mesh frame mismatch "
                f"(e.g. a 90-degree axis swap from an STL Y-up/Z-up round-trip).",
                flush=True,
            )


def report_kp_mask_check(ctx: BlenderContext) -> None:
    """Print an end-of-run summary / prominent alert for the keypoint-mask guard."""
    checked = ctx._kp_mask_stats["checked"]
    if not checked:
        return
    mean_frac = ctx._kp_mask_stats["frac_sum"] / checked
    bad = ctx._kp_mask_stats["bad"]
    print(
        f"[W{ctx.args.worker_id}] keypoint/mask check: {checked} frames sampled, "
        f"mean on-mask {mean_frac * 100:.1f}%, {bad} below "
        f"{ctx.settings.keypoint_mask_min_frac * 100:.0f}% "
        f"(worst {ctx._kp_mask_stats['worst_frac'] * 100:.0f}% "
        f"on {ctx._kp_mask_stats['worst_file']}).",
        flush=True,
    )
    if mean_frac < ctx.settings.keypoint_mask_min_frac or bad > checked // 2:
        print(
            f"[W{ctx.args.worker_id}] ALERT: keypoints do not line up with the rendered masks. "
            f"This dataset is likely mis-labeled (keypoints_3d.json frame != render mesh). "
            f"Fix the keypoint frame before training.",
            flush=True,
        )


def object_occludes_segment(
    obj: BpyObject,
    start_world: BpyObject,
    end_world: BpyObject,
    *,
    clearance: float = 0.001,
    inverse_cache: dict[str, BpyObject] | None = None,
) -> bool:
    """Return true when obj intersects the camera-to-keypoint segment first.

    ``inverse_cache`` maps object names to inverted world matrices; pass one dict for all the
    keypoints of a view (the objects do not move meanwhile) to invert each matrix only once.
    """
    if obj is None or obj.type != "MESH":
        return False

    segment = end_world - start_world
    segment_length = segment.length
    if segment_length <= 0:
        return False

    if inverse_cache is None:
        inv = obj.matrix_world.inverted()
    else:
        inv = inverse_cache.get(obj.name)
        if inv is None:
            inv = inverse_cache[obj.name] = obj.matrix_world.inverted()
    local_start = inv @ start_world
    local_end = inv @ end_world
    local_segment = local_end - local_start
    local_length = local_segment.length
    if local_length <= 0:
        return False

    result, loc, normal, face_idx = obj.ray_cast(
        local_start,
        local_segment.normalized(),
        distance=local_length,
    )
    if not result:
        return False

    hit_world = obj.matrix_world @ loc
    return (hit_world - start_world).length < segment_length - clearance


def project_and_check(
    ctx: BlenderContext,
    kp_world_pos: BpyObject,
    cam: BpyObject,
    scene: BpyObject,
    mesh_obj: BpyObject,
    inverse_cache: dict[str, BpyObject] | None = None,
) -> BpyObject:
    """Project a world-space keypoint to pixels; return ``(x_px, y_px, visibility_state)``.

    ``inverse_cache`` is forwarded to ``object_occludes_segment``.
    """
    res_x, res_y = scene.render.resolution_x, scene.render.resolution_y
    co_ndc = bpy_extras.object_utils.world_to_camera_view(scene, cam, kp_world_pos)
    x_px = co_ndc.x * res_x
    y_px = (1.0 - co_ndc.y) * res_y

    if co_ndc.x < 0 or co_ndc.x > 1 or co_ndc.y < 0 or co_ndc.y > 1 or co_ndc.z < 0:
        return x_px, y_px, 0

    cam_pos = cam.matrix_world.translation
    if object_occludes_segment(mesh_obj, cam_pos, kp_world_pos, inverse_cache=inverse_cache):
        return x_px, y_px, 1
    for distractor in [*ctx.loaded_distractors, *ctx.robustness_objects]:
        if object_occludes_segment(distractor, cam_pos, kp_world_pos, inverse_cache=inverse_cache):
            return x_px, y_px, 1

    return x_px, y_px, 2


def count_unoccluded_keypoints(ctx: BlenderContext) -> BpyObject:
    """Count keypoints that remain in frame and not covered by mesh/distractors."""
    count = 0
    inverse_cache: dict[str, BpyObject] = {}
    for kp in ctx.kp_objects:
        _, _, visibility = project_and_check(
            ctx,
            kp.matrix_world.translation.copy(),
            ctx.cam_obj,
            ctx.scene,
            ctx.mesh_obj,
            inverse_cache,
        )
        if visibility == 2:
            count += 1
    return count


def visible_bbox(xs: BpyObject, ys: BpyObject, width: BpyObject, height: BpyObject) -> BpyObject:
    """Return the padded ``[x, y, width, height]`` box of the projected points."""
    if not xs:
        return [0, 0, 0, 0]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    pad_x, pad_y = (x_max - x_min) * 0.05, (y_max - y_min) * 0.05
    left = max(0, x_min - pad_x)
    top = max(0, y_min - pad_y)
    right = min(width, x_max + pad_x)
    bottom = min(height, y_max + pad_y)
    return [left, top, right - left, bottom - top]


def capture_annotation(
    ctx: BlenderContext,
    image_id: BpyObject,
    file_name: BpyObject,
    view_idx: BpyObject,
    view_name: BpyObject,
    camera_index: BpyObject,
    mask_file: BpyObject = None,
) -> BpyObject:
    """Append the image entry and keypoint annotation for the current view to ``ctx``."""
    ctx.images.append(
        {
            "id": image_id,
            "file_name": file_name,
            "width": ctx.settings.resolution,
            "height": getattr(ctx.settings, "resolution_y", ctx.settings.resolution),
            "pair_id": view_idx if ctx.settings.stereo_enabled else None,
            "view": view_name,
            "camera_index": camera_index,
        }
    )

    keypoints_flat = []
    num_visible = 0
    xs, ys = [], []
    inverse_cache: dict[str, BpyObject] = {}

    for kp in ctx.kp_objects:
        kp_world = kp.matrix_world.translation.copy()
        x, y, v = project_and_check(
            ctx, kp_world, ctx.cam_obj, ctx.scene, ctx.mesh_obj, inverse_cache
        )
        keypoints_flat.extend([round(x, 2), round(y, 2), v])
        if v > 0:
            num_visible += 1
            xs.append(x)
            ys.append(y)

    bbox = visible_bbox(
        xs,
        ys,
        ctx.settings.resolution,
        getattr(ctx.settings, "resolution_y", ctx.settings.resolution),
    )
    ann = {
        "id": image_id,
        "image_id": image_id,
        "category_id": 1,
        "keypoints": keypoints_flat,
        "num_keypoints": num_visible,
        "bbox": [round(b, 2) for b in bbox],
        "area": round(bbox[2] * bbox[3], 2),
        "iscrowd": 0,
        "camera_intrinsics": {
            "fx": round(ctx.fx, 4),
            "fy": round(ctx.fy, 4),
            "cx": round(ctx.cx, 4),
            "cy": round(ctx.cy, 4),
        },
        "camera_extrinsics": [list(row) for row in ctx.cam_obj.matrix_world],
        "object_pose": [list(row) for row in ctx.mesh_obj.matrix_world],
        "view": view_name,
        "camera_index": camera_index,
    }
    if mask_file is not None:
        ann["mask_file"] = mask_file
        ann["mask_type"] = "visible_object"
    if ctx.settings.stereo_enabled:
        ann.update(
            {
                "pair_id": view_idx,
                "stereo_baseline_m": round(ctx.settings.stereo_baseline_m, 6),
                "stereo_mode": "parallel",
            }
        )
    ctx.annotations.append(ann)
    return num_visible


def annotate_robustness(
    ctx: BlenderContext, ann: BpyObject, mask_path: BpyObject, spec: BpyObject
) -> None:
    """Attach measured robustness metadata and use the visible mask for the box.

    The box gets the same 5% padding as keypoint boxes; the tight box stays in
    ``robustness.visible_bbox``. Fully hidden targets get a zero box and area.
    """
    measured = measure_robustness(ctx, mask_path, spec)
    measured["visible_keypoints"] = sum(v == 2 for v in ann["keypoints"][2::3])
    measured["in_frame_keypoints"] = ann["num_keypoints"]
    ann["robustness"] = measured
    if measured["valid_detection_target"]:
        x, y, w, h = measured["visible_bbox"]
        resolution = ctx.settings.resolution
        ann["bbox"] = [
            round(b, 2) for b in visible_bbox([x, x + w], [y, y + h], resolution, resolution)
        ]
    else:
        ann["bbox"] = [0, 0, 0, 0]
    ann["area"] = measured["visible_pixels"]


def measure_robustness(ctx: BlenderContext, mask_path: BpyObject, spec: BpyObject) -> BpyObject:
    """Measure capsule coverage against the same scene with only capsules hidden.

    Other distractors and clipping stay fixed, so this measures incremental proxy
    occlusion of the otherwise visible in-frame silhouette, not amodal coverage.
    Returns the metadata stored as the annotation's ``robustness`` field; the
    ``visible_bbox`` is the tight ``[x, y, w, h]`` of the visible mask.
    """
    visible = _load_mask_foreground(ctx, mask_path)
    if visible is None:
        raise RuntimeError(f"Cannot measure robustness mask: {mask_path}")
    baseline = visible
    if ctx.robustness_objects:
        baseline = _render_mask_without_proxies(ctx, mask_path + ".unoccluded.png")
    visible_pixels = int(visible.sum())
    baseline_pixels = int(baseline.sum())
    occluded_fraction = None
    if baseline_pixels:
        occluded_fraction = float(np.count_nonzero(baseline & ~visible) / baseline_pixels)
    return {
        "mode": spec["mode"],
        "clipping": spec.get("clipping"),
        "occluder_kind": spec.get("occluder_kind"),
        "visible_pixels": visible_pixels,
        "without_proxy_pixels": baseline_pixels,
        "proxy_occluded_fraction": occluded_fraction,
        "visible_bbox": _tight_mask_bbox(visible),
        "valid_detection_target": bool(visible_pixels),
        "coverage_basis": "in_frame_visible_silhouette_without_proxies",
    }


def _render_mask_without_proxies(ctx: BlenderContext, reference_path: BpyObject) -> BpyObject:
    """Render and load the visible mask with proxies hidden, then restore them.

    Proxy visibility is restored and the temporary image removed even if
    rendering fails.
    """
    hidden = [(obj, obj.hide_render) for obj in ctx.robustness_objects]
    try:
        for obj, _ in hidden:
            obj.hide_render = True
        render_visible_object_mask(ctx, reference_path)
        baseline = _load_mask_foreground(ctx, reference_path)
        if baseline is None:
            raise RuntimeError(f"Cannot measure robustness baseline: {reference_path}")
        return baseline
    finally:
        for obj, state in hidden:
            obj.hide_render = state
        bpy.context.view_layer.update()
        if os.path.exists(reference_path):
            os.remove(reference_path)


def _tight_mask_bbox(mask: BpyObject) -> BpyObject:
    """Return the inclusive pixel ``[x, y, w, h]`` of a mask, or zeros if empty."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return [0, 0, 0, 0]
    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
    ]
