# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authored-input binding, camera setup and reproducibility exports."""

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import glob
import json
import math
import os
import random
import shutil

import bpy
import numpy as np
from mathutils import Vector

from rendering.paths import resolve_under

from .camera import resolve_render_engine, set_if_present
from .context import BlenderContext, BpyObject
from .materials import parse_rgb_color


def resolve_config_asset(ctx: BlenderContext, path: str | None) -> str | None:
    """Resolve ``path`` against the config directory; ``None`` stays ``None``."""
    return resolve_under(ctx.settings.base_dir, path)


def load_keypoints_from_json(ctx: BlenderContext, path: str | None) -> None:
    """Create/update parented keypoint empties from a keypoints_3d.json."""
    path = resolve_config_asset(ctx, path)
    if path is None:
        raise ValueError("a keypoints JSON path is required")
    with open(path) as f:
        data = json.load(f)
    points = data.get("keypoints_3d")
    if not points:
        raise ValueError(f"{path} does not contain keypoints_3d")

    ctx.settings.num_keypoints = len(points)
    names = [f"{ctx.settings.kp_prefix}{i:03d}" for i in range(ctx.settings.num_keypoints)]
    wanted = set(names)

    for obj in list(bpy.data.objects):
        if obj.name.startswith(ctx.settings.kp_prefix) and obj.name not in wanted:
            bpy.data.objects.remove(obj, do_unlink=True)

    for name, coords in zip(names, points, strict=False):
        kp_obj = bpy.data.objects.get(name)
        if kp_obj is None:
            bpy.ops.object.empty_add(type="SPHERE", radius=0.001, location=(0, 0, 0))
            kp_obj = bpy.context.active_object
            kp_obj.name = name
        kp_obj.parent = ctx.mesh_obj
        kp_obj.location = Vector((float(coords[0]), float(coords[1]), float(coords[2])))
        kp_obj.empty_display_size = max(float(ctx.mesh_obj.dimensions[0]) * 0.05, 0.001)

    print(f"Loaded {ctx.settings.num_keypoints} keypoints from {path}")


def initialize(ctx: BlenderContext) -> None:
    """Initialize the worker in a fixed order before choosing a rendering mode."""
    seed_and_prepare_output(ctx)
    bind_object(ctx)
    export_input_metadata(ctx)
    configure_render(ctx)
    create_camera(ctx)
    gather_assets(ctx)
    adapt_camera_distance(ctx)
    initialize_caches(ctx)


def seed_and_prepare_output(ctx: BlenderContext) -> None:
    """Seed the RNGs for this worker and create the output directories."""
    # Random seed setup
    # ---------------------------------------------------------------------------
    worker_seed = ctx.args.seed + ctx.args.worker_id * 100000
    ctx.rng = random.Random(worker_seed)
    # NumPy's legacy RNG accepts only uint32 seeds. Keep historical in-range
    # streams unchanged while supporting worker IDs derived from large shards.
    np.random.seed(worker_seed % (2**32))

    ctx.num_views = ctx.args.end_idx - ctx.args.start_idx
    print(f"\n{'=' * 60}")
    worker_id, start_idx, end_idx = ctx.args.worker_id, ctx.args.start_idx, ctx.args.end_idx
    print(f"WORKER {worker_id}: rendering images [{start_idx} .. {end_idx})")
    num_images = ctx.num_views * ctx.settings.images_per_sample
    print(f"  {ctx.num_views} samples, {num_images} images, seed={worker_seed}")
    print(f"  Config: {ctx.settings.config_path}")
    if ctx.settings.render_masks:
        print(f"  Visible object masks: enabled → {ctx.settings.mask_dir_name}/")
    if ctx.settings.stereo_enabled:
        print(f"  Stereo: enabled, baseline={ctx.settings.stereo_baseline_m:.4f}m, mode=parallel")
    print(
        f"  Axis bias (object-axis -> world +Z): "
        f"x={ctx.settings.axis_bias['x'] * 100:.0f}%  "
        f"y={ctx.settings.axis_bias['y'] * 100:.0f}%  "
        f"z={ctx.settings.axis_bias['z'] * 100:.0f}%  "
        f"random={max(0.0, 1.0 - ctx.settings._axis_total) * 100:.0f}%  "
        f"jiggle: {ctx.settings.axis_jiggle}°"
    )
    print(f"{'=' * 60}\n")

    os.makedirs(os.path.join(ctx.args.outdir, "images"), exist_ok=True)
    if ctx.settings.render_masks:
        os.makedirs(os.path.join(ctx.args.outdir, ctx.settings.mask_dir_name), exist_ok=True)


def bind_object(ctx: BlenderContext) -> None:
    """Bind the target mesh and its keypoints from the open ``.blend``."""
    # Scene references
    # ---------------------------------------------------------------------------
    ctx.scene = bpy.context.scene
    ctx.mesh_obj = bpy.data.objects[ctx.settings.mesh_name]

    # A texture the .blend references by path but cannot find renders the object in Blender's
    # magenta "missing texture" color without any error; refuse to produce 50k such images
    # (mesh_setup.py packs object textures into the .blend for exactly this reason).
    for _mat in (
        [] if ctx.settings.object_material_mode == "override" else ctx.mesh_obj.data.materials
    ):
        if _mat is None or not _mat.use_nodes:
            continue
        for _node in _mat.node_tree.nodes:
            _img = getattr(_node, "image", None)
            if (
                _img is not None
                and _img.packed_file is None
                and not os.path.exists(bpy.path.abspath(_img.filepath))
            ):
                raise SystemExit(
                    f"object material {_mat.name!r} references missing texture {_img.filepath!r}; "
                    f"pack it into the .blend (mesh_setup.py does) or fix the path"
                )

    if ctx.settings.keypoints_json and not ctx.args.fixed_pose_params:
        load_keypoints_from_json(ctx, ctx.settings.keypoints_json)

    kp_names = [f"{ctx.settings.kp_prefix}{i:03d}" for i in range(ctx.settings.num_keypoints)]
    if ctx.args.fixed_pose_params:
        ctx.kp_objects = []
    else:
        ctx.kp_objects = [bpy.data.objects[name] for name in kp_names]
    print(f"Found {len(ctx.kp_objects)} keypoints on '{ctx.settings.mesh_name}'")


def export_input_metadata(ctx: BlenderContext) -> None:
    """Write ``keypoints_3d.json`` (worker 0 only) and the input metadata for the dataset."""
    kp_names = [f"{ctx.settings.kp_prefix}{i:03d}" for i in range(ctx.settings.num_keypoints)]
    # Export 3D keypoints (object-local) — only worker 0 writes this
    # ---------------------------------------------------------------------------
    if ctx.args.worker_id == 0:
        kp_3d_local = []
        for kp in ctx.kp_objects:
            loc = kp.location
            kp_3d_local.append([round(loc.x, 6), round(loc.y, 6), round(loc.z, 6)])

        with open(os.path.join(ctx.args.outdir, "keypoints_3d.json"), "w") as f:
            json.dump(
                {
                    "object": ctx.settings.mesh_name,
                    "num_keypoints": ctx.settings.num_keypoints,
                    "keypoint_names": kp_names,
                    "keypoints_3d": kp_3d_local,
                },
                f,
                indent=2,
            )

        # Save camera intrinsics for PnP pose estimation
        cam_lens = ctx.settings.focal_length_mm
        cam_sensor_width = ctx.settings.sensor_width_mm
        res_y = ctx.settings.resolution_y
        fx_val = cam_lens * ctx.settings.resolution / cam_sensor_width
        fy_val = fx_val  # Square pixels
        cx_val = ctx.settings.resolution / 2.0
        cy_val = res_y / 2.0

        camera_info = {
            "image_width": ctx.settings.resolution,
            "image_height": res_y,
            "fx": round(fx_val, 6),
            "fy": round(fy_val, 6),
            "cx": round(cx_val, 6),
            "cy": round(cy_val, 6),
            "camera_matrix": [
                [round(fx_val, 6), 0.0, round(cx_val, 6)],
                [0.0, round(fy_val, 6), round(cy_val, 6)],
                [0.0, 0.0, 1.0],
            ],
            "distortion_coefficients": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],  # No distortion in synthetic data
            "camera_model": "pinhole",
            "lens_mm": cam_lens,
            "sensor_width_mm": cam_sensor_width,
        }

        with open(os.path.join(ctx.args.outdir, "camera_intrinsics.json"), "w") as f:
            json.dump(camera_info, f, indent=2)

        # Copy config to output for reproducibility
        shutil.copy(
            ctx.settings.config_path,
            os.path.join(ctx.args.outdir, "render_config.yaml"),
        )


def configure_render(ctx: BlenderContext) -> None:
    """Configure the render engine, resolution and output format."""
    # EEVEE setup
    # ---------------------------------------------------------------------------

    ctx.scene.render.resolution_x = ctx.settings.resolution
    ctx.scene.render.resolution_y = ctx.settings.resolution_y
    ctx.scene.render.resolution_percentage = 100
    ctx.scene.render.image_settings.file_format = ctx.settings.cfg_output.get("image_format", "PNG")
    ctx.scene.render.image_settings.compression = ctx.settings.cfg_output.get("compression", 15)
    ctx.scene.render.film_transparent = False

    ctx.scene.render.engine = resolve_render_engine(
        ctx.scene, ctx.settings.cfg_render.get("engine", "BLENDER_EEVEE")
    )
    ctx.eevee = ctx.scene.eevee
    ctx.eevee_samples = int(
        ctx.settings.cfg_render.get("eevee_samples", ctx.settings.cfg_render.get("samples", 64))
    )
    set_if_present(ctx.eevee, "taa_render_samples", ctx.eevee_samples)
    set_if_present(ctx.eevee, "taa_samples", min(ctx.eevee_samples, 16))
    set_if_present(ctx.eevee, "use_raytracing", ctx.settings.cfg_render.get("use_raytracing", True))
    set_if_present(ctx.eevee, "shadow_ray_count", ctx.settings.shadow_rays)
    set_if_present(ctx.eevee, "use_shadows", True)
    set_if_present(ctx.eevee, "shadow_step_count", 6)
    apply_optional_cycles_settings(ctx)

    rt = getattr(ctx.eevee, "ray_tracing_options", None)
    if rt is not None:
        set_if_present(rt, "resolution_scale", "2")
        set_if_present(rt, "use_denoise", True)


def apply_optional_cycles_settings(ctx: BlenderContext) -> None:
    """Apply ``render.cycles_samples`` and ``render.threads``; absent keys keep Blender's."""
    cfg_render = ctx.settings.cfg_render
    if cfg_render.get("cycles_samples") is not None:
        ctx.scene.cycles.samples = int(cfg_render["cycles_samples"])
    if cfg_render.get("threads") is not None:
        ctx.scene.render.threads_mode = "FIXED"
        ctx.scene.render.threads = int(cfg_render["threads"])


def create_camera(ctx: BlenderContext) -> None:
    """Create the camera used for all views."""
    # Camera setup
    # ---------------------------------------------------------------------------
    ctx.cam_data = bpy.data.cameras.new("SynthCam")
    # honor the config: these were hardcoded, so camera.focal_length_mm /
    # sensor_width_mm in the render config had no effect at all
    ctx.cam_data.lens = ctx.settings.focal_length_mm
    ctx.cam_data.sensor_width = ctx.settings.sensor_width_mm
    ctx.cam_data.clip_start = 0.001
    ctx.cam_data.clip_end = 100.0
    ctx.cam_obj = bpy.data.objects.new("SynthCam", ctx.cam_data)
    ctx.scene.collection.objects.link(ctx.cam_obj)
    ctx.scene.camera = ctx.cam_obj


def glob_asset_dir(directory: BpyObject, *patterns: BpyObject) -> BpyObject:
    """Sorted matches of ``patterns`` in ``directory``; an empty directory name means none."""
    if not directory:
        return []
    return sorted(f for pattern in patterns for f in glob.glob(os.path.join(directory, pattern)))


def gather_assets(ctx: BlenderContext) -> None:
    """Collect HDRI, texture and distractor assets; refuse missing ones."""
    # Gather assets
    # ---------------------------------------------------------------------------
    ctx.hdri_files = glob_asset_dir(ctx.settings.hdri_dir, "*.hdr", "*.exr")
    print(f"Found {len(ctx.hdri_files)} HDRI files")

    ctx.texture_dirs = [
        d for d in glob_asset_dir(ctx.settings.texture_dir, "*") if os.path.isdir(d)
    ]

    # An asset directory that has gone missing is silent and expensive: the
    # renderer falls back to a flat studio background and every frame of a
    # 10k render looks like a gray screen. That cost a full overnight run and
    # 16 AR before anyone noticed, so a config that asks for assets which are
    # not there now stops rather than quietly degrading.
    missing = []
    if "hdri" in ctx.settings.world_background_modes and not ctx.hdri_files:
        missing.append(
            f"world_background_modes includes 'hdri' but no .hdr/.exr "
            f"files are in {ctx.settings.hdri_dir!r}"
        )
    if (
        ctx.settings.requires_ground_textures
        and ctx.settings.no_ground_prob < 1.0
        and not ctx.texture_dirs
    ):
        missing.append(
            f"ground planes are enabled but no texture directories are "
            f"in {ctx.settings.texture_dir!r}"
        )
    if missing and os.environ.get("SYNTH_ALLOW_MISSING_ASSETS") != "1":
        raise RuntimeError(
            "render assets are missing:\n  - "
            + "\n  - ".join(missing)
            + "\nSet SYNTH_ALLOW_MISSING_ASSETS=1 to render without them."
        )
    print(f"Found {len(ctx.texture_dirs)} texture sets")

    ctx.distractor_files = glob_asset_dir(
        ctx.settings.distractor_dir, "*.usdc", "*.usd", "*.usdz", "*.glb", "*.gltf"
    )
    print(f"Found {len(ctx.distractor_files)} distractor files (USD/GLB)")

    bbox_corners = [ctx.mesh_obj.matrix_world @ Vector(c) for c in ctx.mesh_obj.bound_box]
    ctx.mesh_size = max(
        (max(v[i] for v in bbox_corners) - min(v[i] for v in bbox_corners)) for i in range(3)
    )

    # Scene scale factor - used to scale ground plane, distractors, textures
    # Reference size is 0.1m (10cm), so a 1m object has scene_scale=10
    reference_size = 0.1  # 10cm reference object
    ctx.scene_scale = ctx.mesh_size / reference_size
    print(f"Mesh size: {ctx.mesh_size:.4f}m → scene_scale: {ctx.scene_scale:.2f}x")


def adapt_camera_distance(ctx: BlenderContext) -> None:
    """Derive the camera distance range from the object size when auto-distance is enabled."""
    # Auto-adapt camera distance based on object size
    # ---------------------------------------------------------------------------
    if ctx.settings.auto_distance_enabled:
        # Scale camera distance to keep consistent framing regardless of object size
        # Uses FOV calculation: distance = (object_size / 2) / tan(fov/2) / fill_factor
        cam_lens = ctx.settings.focal_length_mm
        cam_sensor_width = ctx.settings.sensor_width_mm
        fov_horizontal = 2 * math.atan(cam_sensor_width / (2 * cam_lens))

        # Compute ideal distance for the object to fill the frame at the desired fill factor
        # At fill_factor=0.5, object should fill ~50% of frame width
        ideal_dist_min = (
            (ctx.mesh_size / 2)
            / math.tan(fov_horizontal / 2)
            / ctx.settings.auto_distance_fill_factor_max
        )
        ideal_dist_max = (
            (ctx.mesh_size / 2)
            / math.tan(fov_horizontal / 2)
            / ctx.settings.auto_distance_fill_factor_min
        )

        # Use auto-calculated distances
        ctx.settings.cam_dist_min = ideal_dist_min
        ctx.settings.cam_dist_max = ideal_dist_max

        print(
            f"Auto-distance enabled: mesh_size={ctx.mesh_size:.4f}m → "
            f"cam_dist=[{ctx.settings.cam_dist_min:.3f}, {ctx.settings.cam_dist_max:.3f}]m"
        )
    else:
        print(
            f"Using fixed camera distance: "
            f"[{ctx.settings.cam_dist_min:.3f}, {ctx.settings.cam_dist_max:.3f}]m"
        )


def initialize_caches(ctx: BlenderContext) -> None:
    """Reset the per-worker scene state (ground, distractors, lights) before rendering."""
    # Ground plane
    # ---------------------------------------------------------------------------
    ctx.ground_obj = None

    # Distractors
    # ---------------------------------------------------------------------------
    ctx.loaded_distractors = []

    # ---------------------------------------------------------------------------
    # Lights
    # ---------------------------------------------------------------------------
    ctx.extra_lights = []

    # ---------------------------------------------------------------------------
    # Keypoint/mask consistency guard
    # ---------------------------------------------------------------------------
    ctx._kp_mask_stats = {
        "checked": 0,
        "bad": 0,
        "frac_sum": 0.0,
        "worst_frac": 1.0,
        "worst_file": None,
        "warned": 0,
    }

    # ---------------------------------------------------------------------------
    # Compute intrinsics
    # ---------------------------------------------------------------------------
    ctx.fx = ctx.cam_data.lens * ctx.settings.resolution / ctx.cam_data.sensor_width
    ctx.fy = ctx.fx
    ctx.cx = ctx.settings.resolution / 2.0
    ctx.cy = ctx.settings.resolution_y / 2.0

    ctx.record_distractor_instances = []

    ctx.object_rgba_choices = (
        [
            parse_rgb_color(color.strip())
            for color in ctx.settings.object_color_choices.split(",")
            if color.strip()
        ]
        if isinstance(ctx.settings.object_color_choices, str)
        else [parse_rgb_color(color) for color in ctx.settings.object_color_choices]
        if ctx.settings.object_color_choices
        else []
    )

    ctx.mask_target_material = None

    ctx.mask_background_material = None
