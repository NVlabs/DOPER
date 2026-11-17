# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# BlenderProc requires its import before any other statement (checked by `blenderproc run`),
# so this module has no docstring. BlenderProc and bpy exist only inside `blenderproc run`.
# pyright: reportMissingImports=false, reportMissingModuleSource=false
# pyright: reportAttributeAccessIssue=false
import blenderproc as bproc


# BlenderProc scene of one layout: the only module that imports ``blenderproc`` or ``bpy``.
#
# Run through the campaign (``python -m blenderproc run ... scene.py --assets DIR --config FILE
# --out DIR --seed N --gpu-index G``). The pre-physics draws come from
# :func:`rendering.blenderproc.plan.plan_scene` (re-exported here as ``plan_scene``);
# :func:`realize` applies them, then drops the objects with physics, samples the camera views,
# renders with Cycles and writes the BOP layout with ``write_bop``. Render parameters follow the
# source renderers: 20 target-dataset instances and two T-LESS/HB/TYOL distractors each, five 2 m
# room planes with one ambientCG material, a ceiling emission and a 200 W point light, a 3-10 s
# physics drop, views on a 0.61-1.4 m shell around a 15-object point of interest, Cycles at the
# configured samples with the Intel denoiser, Filmic / Medium High Contrast, and ``write_bop``
# JPEG quality 95 with millimeter annotations.

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import bpy
import numpy as np

# BlenderProc clears PYTHONPATH and the package parent from sys.path; restore the checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rendering.blenderproc.plan import (
    LIGHT_ENERGY_W,
    MAX_FRAMES_PER_LAYOUT,
    RenderConfig,
    ScenePlan,
    ScenePools,
    load_config,
    load_pools,
    plan_scene,
)

__all__ = ["plan_scene", "realize"]

ROOM_PLANES = (
    {},
    {"location": [0, -2, 2], "rotation": [-np.pi / 2, 0, 0]},
    {"location": [0, 2, 2], "rotation": [np.pi / 2, 0, 0]},
    {"location": [2, 0, 2], "rotation": [0, -np.pi / 2, 0]},
    {"location": [-2, 0, 2], "rotation": [0, np.pi / 2, 0]},
)
RIGID = {"friction": 100.0, "linear_damping": 0.99, "angular_damping": 0.99}
MAX_CAMERA_ATTEMPTS = 20000
POI_OBJECTS = 15
MIN_OBSTACLE_DISTANCE_M = 0.3


def sha256(path: Path | str) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_gpu(index: int) -> list[str]:
    """Render on exactly the OptiX device ``index`` and return its name."""
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.refresh_devices()
    optix = [d for d in prefs.devices if d.type == "OPTIX"]
    if not 0 <= index < len(optix):
        raise ValueError(f"OptiX device {index} not found ({len(optix)} available)")
    bproc.renderer.set_render_devices(desired_gpu_device_type="OPTIX", desired_gpu_ids=[index])
    for device in prefs.devices:
        device.use = device is optix[index]
    selected = [d.name for d in prefs.devices if d.use]
    if selected != [optix[index].name]:
        raise ValueError(f"wrong render devices: {selected}")
    return selected


def _check_units(
    obj: "bproc.types.MeshObject", info: dict[str, Any], config: RenderConfig
) -> dict[str, Any]:
    oid = obj.get_cp("category_id")
    data = obj.blender_obj.data
    vertices = np.asarray([v.co[:] for v in data.vertices]) * np.asarray(obj.get_scale()) * 1000
    low = vertices.min(axis=0)
    extent = np.ptp(vertices, axis=0)
    meta = info[str(oid)]
    check = config.unit_check
    close = np.allclose(extent, [meta[f"size_{a}"] for a in "xyz"], atol=check.atol_mm,
                        rtol=check.rtol)  # fmt: skip
    if check.mode == "min_extent":
        close &= np.allclose(low, [meta[f"min_{a}"] for a in "xyz"], atol=check.atol_mm,
                             rtol=check.rtol)  # fmt: skip
    if not close:
        raise ValueError(f"mesh origin/unit/extent mismatch for object {oid}")
    if not data.uv_layers:
        raise ValueError(f"object {oid}: texture UVs missing")
    return {"minimum_mm": low.tolist(), "extent_mm": extent.tolist(),
            "uv_layers": len(data.uv_layers)}  # fmt: skip


def _texture_hashes(obj: "bproc.types.MeshObject") -> dict[str, str]:
    hashes = {}
    for material in obj.blender_obj.data.materials:
        if material is None or not material.use_nodes:
            continue
        for node in material.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image:
                path = Path(bpy.path.abspath(node.image.filepath))
                if path.is_file():
                    hashes[str(path)] = sha256(path)
    return hashes


def _load_objects(plan: ScenePlan, config: RenderConfig, assets: Path) -> tuple[list, list]:
    targets = bproc.loader.load_bop_objs(
        str(assets / "bop" / config.dataset), obj_ids=list(plan.target_ids),
        object_model_unit="mm", move_origin_to_x_y_plane=False,
    )  # fmt: skip
    distractors = []
    for spec in config.distractors:
        distractors += bproc.loader.load_bop_objs(
            str(assets / "bop" / spec.dataset), obj_ids=list(plan.distractors[spec.dataset]),
            model_type=spec.model_type, object_model_unit="mm",
        )  # fmt: skip
    loaded = targets + distractors
    objects = [loaded[i] for i in plan.object_order]
    return targets, objects


def _apply_surfaces(objects: list, plan: ScenePlan, config: RenderConfig,
                    assets: Path) -> tuple[dict, dict]:  # fmt: skip
    info_path = assets / "bop" / config.dataset / "models" / "models_info.json"
    info = json.loads(info_path.read_text())
    geometry, hashes = {}, {}
    for obj, sample in zip(objects, plan.material_samples, strict=True):
        if (obj.get_cp("bop_dataset_name"), obj.get_cp("category_id")) != (
            sample.dataset,
            sample.id,
        ):
            raise ValueError("loaded objects differ from the scene plan")
        obj.set_shading_mode("auto")
        model = Path(obj.get_cp("model_path"))
        hashes[str(model)] = sha256(model)
        if sample.dataset == config.dataset:
            geometry[str(sample.id)] = _check_units(obj, info, config)
        hashes.update(_texture_hashes(obj))
        material = obj.get_materials()[0]
        if sample.grey is not None:
            material.set_principled_shader_value("Base Color", [sample.grey] * 3 + [1])
        material.set_principled_shader_value("Roughness", sample.roughness)
        material.set_principled_shader_value("Specular IOR Level", sample.specular_ior_level)
        obj.enable_rigidbody(True, mass=1.0, **RIGID)
    return geometry, hashes


def _build_room(plan: ScenePlan, pools: ScenePools, assets: Path) -> None:
    planes = [bproc.object.create_primitive("PLANE", scale=[2, 2, 1], **kw) for kw in ROOM_PLANES]
    for plane in planes:
        plane.enable_rigidbody(False, collision_shape="BOX", mass=1.0, **RIGID)
    materials = {m.get_name(): m for m in bproc.loader.load_ccmaterials(str(assets / "cctextures"))}
    if sorted(materials) != list(pools.room_materials):
        raise ValueError(f"loaded room materials {sorted(materials)} differ from the pool")
    for plane in planes:
        plane.replace_materials(materials[plan.room_material])
    ceiling = bproc.object.create_primitive("PLANE", scale=[3, 3, 1], location=[0, 0, 10])
    light_material = bproc.material.create("ceiling_emission")
    light_material.make_emissive(emission_strength=plan.emission,
                                 emission_color=list(plan.emission_color))  # fmt: skip
    ceiling.replace_materials(light_material)
    light = bproc.types.Light()
    light.set_energy(LIGHT_ENERGY_W)
    light.set_color(np.asarray(plan.light_color))
    light.set_location(np.asarray(plan.light_position))


def _drop_objects(objects: list) -> None:
    def sample_pose(obj: "bproc.types.MeshObject") -> None:
        low = np.random.uniform([-0.3, -0.3, 0], [-0.2, -0.2, 0])
        high = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.6])
        obj.set_location(np.random.uniform(low, high))
        obj.set_rotation_euler(bproc.sampler.uniformSO3())

    bproc.object.sample_poses(objects_to_sample=objects, sample_pose_func=sample_pose,
                              max_tries=1000)  # fmt: skip
    bproc.object.simulate_physics_and_fix_final_poses(
        min_simulation_time=3, max_simulation_time=10, check_object_interval=1,
        substeps_per_frame=20, solver_iters=25,
    )  # fmt: skip


def _sample_views(targets: list, objects: list, config: RenderConfig) -> tuple[list, int]:
    camera = config.camera
    bvh = bproc.object.create_bvh_tree_multi_objects(objects)
    K = np.asarray(camera.K).reshape(3, 3)  # noqa: N806
    bproc.camera.set_intrinsics_from_K_matrix(K, camera.width, camera.height)
    poses: list[list[list[float]]] = []
    attempts = 0
    while len(poses) < config.frames_per_layout:
        attempts += 1
        if attempts > MAX_CAMERA_ATTEMPTS:
            raise RuntimeError("camera sampler could not find enough valid views")
        position = bproc.sampler.shell(
            center=[0, 0, 0], radius_min=camera.radius_m[0], radius_max=camera.radius_m[1],
            elevation_min=camera.elevation_deg[0], elevation_max=camera.elevation_deg[1],
        )  # fmt: skip
        poi = bproc.object.compute_poi(np.random.choice(targets, size=POI_OBJECTS, replace=False))
        rotation = bproc.camera.rotation_from_forward_vec(
            poi - position, inplane_rot=np.random.uniform(*camera.inplane_rad)
        )
        pose = bproc.math.build_transformation_mat(position, rotation)
        if bproc.camera.perform_obstacle_in_view_check(pose, {"min": MIN_OBSTACLE_DISTANCE_M}, bvh):
            bproc.camera.add_camera_pose(pose, frame=len(poses))
            poses.append(np.asarray(pose).tolist())
    return poses, attempts


def _configure_renderer(config: RenderConfig) -> None:
    view = bpy.context.scene.view_settings
    view.view_transform = config.view_transform
    view.look = config.look
    view.exposure = 0.0
    view.gamma = 1.0
    bproc.renderer.set_max_amount_of_samples(config.samples)
    bproc.renderer.set_denoiser("INTEL")
    bproc.renderer.enable_depth_output(activate_antialiasing=False)


def realize(
    plan: ScenePlan,
    pools: ScenePools,
    config: RenderConfig,
    *,
    assets: Path,
    out: Path,
    gpu_index: int,
    config_path: Path,
) -> dict[str, Any]:
    """Build, simulate, render and write one planned layout.

    Args:
        plan: Pre-physics draws (:func:`plan_scene`).
        pools: Pools the plan was drawn from.
        config: Render config.
        assets: Staged asset root.
        out: Fresh attempt directory.
        gpu_index: OptiX device index.
        config_path: Config file (hashed into the record).

    Returns:
        The render receipt.
    """
    started = time.time()
    bproc.init()
    devices = select_gpu(gpu_index)
    bpy.context.scene.render.threads_mode = "FIXED"
    bpy.context.scene.render.threads = config.render_threads
    # Continue the source's single global stream after the planned draws.
    random.seed(plan.seed)
    np.random.set_state(plan.rng_state)
    targets, objects = _load_objects(plan, config, assets)
    geometry, hashes = _apply_surfaces(objects, plan, config, assets)
    _build_room(plan, pools, assets)
    _drop_objects(objects)
    poses, attempts = _sample_views(targets, objects, config)
    _configure_renderer(config)
    here = Path(__file__).resolve()
    record = {
        "status": "scene_frozen_before_render",
        **plan.as_record(),
        "target_id": config.target_object,
        "frames": config.frames_per_layout,
        "image_size": [config.camera.width, config.camera.height],
        "cam_K": list(config.camera.K),
        "recipe_sha256": sha256(config_path),
        "renderer_sha256": {p.name: sha256(p) for p in (here, here.with_name("plan.py"))},
        "blender_version": bpy.app.version_string,
        "devices": devices,
        "geometry": geometry,
        "asset_sha256": hashes,
        "asset_manifest_sha256": sha256(assets / "ASSET_MANIFEST.json"),
        "object_world_poses_m": [
            {
                "dataset": o.get_cp("bop_dataset_name"),
                "id": o.get_cp("category_id"),
                "matrix": np.asarray(o.get_local2world_mat()).tolist(),
            }
            for o in objects
        ],  # fmt: skip
        "camera_world_poses_m": poses,
        "camera_attempts": attempts,
        "provenance": config.provenance,
    }
    (out / "scene_record.json").write_text(json.dumps(record, indent=2) + "\n")
    render_start = time.time()
    data = bproc.renderer.render()
    render_seconds = time.time() - render_start
    bproc.writer.write_bop(
        str(out / "bop_data"), target_objects=targets, dataset=config.dataset,
        depths=data["depth"], colors=data["colors"], color_file_format="JPEG",
        jpg_quality=config.jpeg_quality, depth_scale=config.depth_scale_mm_per_uint,
        annotation_unit="mm", save_world2cam=True, frames_per_chunk=1000,
        calc_mask_info_coco=True, num_worker=2, ignore_dist_thres=10,
    )  # fmt: skip
    receipt = {
        "status": "render_and_bop_write_returned",
        "frames": config.frames_per_layout,
        "render_seconds": render_seconds,
        "total_seconds": time.time() - started,
        "target_id": config.target_object,
        "seed": plan.seed,
    }
    (out / "render_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    """Entry point under ``blenderproc run``."""
    parser = argparse.ArgumentParser(description="Render one BlenderProc layout.")
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    if not 1 <= config.frames_per_layout <= MAX_FRAMES_PER_LAYOUT:
        raise ValueError("one physical layout holds at most 25 views")
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError("use a fresh attempt directory; rendered data is never overwritten")
    out.mkdir(parents=True)
    assets = args.assets.resolve()
    pools = load_pools(assets, config)
    plan = plan_scene(args.seed, pools, config)
    receipt = realize(plan, pools, config, assets=assets, out=out, gpu_index=args.gpu_index,
                      config_path=args.config.resolve())  # fmt: skip
    print("LAYOUT_RENDER_RETURNED", json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
