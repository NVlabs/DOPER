# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NuRec ``render`` stage of ``doper prepare``.

``render`` imports the trained NuRec artifact given with ``--artifact``
(``doper.nurec.scene.import_scene``, against ``<root>/reconstruction``), samples the recipe's
novel views and writes ``<root>/render/{views,sampling,trajectory,command}.json`` and
``plan.json`` (CPU only), then renders them with the NRE container (``doper.nurec.render``,
the image pinned by the recipe's digest) into ``<root>/render/rgb/camera1``. Rendering needs
docker, an NVIDIA GPU and NGC access to the image; without docker or a GPU the stage keeps the
plan and refuses with that reason. With a robot reference, the robot camera poses mapped into
the scene by the robot scale audit are rejection sources too, and the rejection distance
converts with that audit's scale. ``sampling.json`` and
``plan.json`` record the capture's ``capture_flags``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from ...bop.sources import canonical_sha256, publish_json, sha256_file
from ...console import log
from ...recipes.nurec_schema import NurecRecipe
from ..common import (
    Step,
    read_json,
)
from ._names import (
    CAPTURE_DIR,
    FRAMES_DIR,
    RECONSTRUCTION_DIR,
    RENDER_CACHE_TARGET,
    RENDER_DIR,
    RENDER_PLAN_NAME,
    RENDER_RECEIPT_NAME,
    force_flags,
    force_step_keys,
)
from .nurec_capture import CaptureContext, _capture_names
from .receipts import StageOutput, clear, reuse_or_refuse


class _RenderState:
    """The scene the render plan step imports, handed to the render step."""

    scene: Any = None


def _rejection_sources(
    ctx: CaptureContext,
    scene: Any,  # noqa: ANN401
) -> tuple[list[Any], Any, str]:
    """The sampler's extra rejection sources, the scene it samples and the source of its scale.

    With a robot reference the robot camera poses, mapped by the robot
    to scene similarity (``robot_scale_audit``), join the scene cameras as rejection sources,
    and ``reject_mm`` converts with that similarity's scale; otherwise the training cameras are
    the sources and the reconstruction's anchor converts.
    """
    from dataclasses import replace

    from ...nurec.registration import robot_poses_in_scene, robot_scale_audit

    if not ctx.receipt.get("robot_reference"):
        return [], scene, "reconstruction_anchor"
    _, robot = ctx.image_set_and_robot
    anchor = robot_scale_audit(robot, scene)
    extra = list(robot_poses_in_scene(robot, anchor).values())
    return extra, replace(scene, scene_units_per_metre=anchor.scene_units_per_metre), "robot_audit"


def _render_backend(recipe: NurecRecipe, cache: Path | None) -> Any:  # noqa: ANN401
    from ...nurec.render import Mount, NreCliBackend

    mounts = [Mount(cache, RENDER_CACHE_TARGET)] if cache is not None else []
    env = {"TORCH_HOME": RENDER_CACHE_TARGET} if cache is not None else {}
    return NreCliBackend(recipe.render.image_digest, mounts=mounts, env=env)


def _render_seed_views(recipe: NurecRecipe, frames: Any) -> list[str]:  # noqa: ANN401
    """The recipe's seed views (capture indices) as capture image names.

    Raises:
        ValueError: For an index beyond the capture.
    """
    names = [frame.source for frame in frames.frames]
    return _capture_names(recipe.render.seed_views, names, "render.seed_views")


def _nurec_render_plan(
    ctx: CaptureContext, artifact: Path, cache: Path | None, force: bool, state: _RenderState,
) -> None:  # fmt: skip
    from dataclasses import asdict

    from ...nurec import capture
    from ...nurec.render import EXCLUSION, sample_novel_views, write_rig_trajectory
    from ...nurec.scene import import_metric_scene
    from ...reconstruction.metadata import METADATA_NAME

    root, recipe = ctx.root, ctx.recipe
    metadata = root / RECONSTRUCTION_DIR / METADATA_NAME
    if not metadata.is_file():
        raise FileNotFoundError(f"{metadata} is missing: run `doper prepare {recipe.name} --stage "
                                "reconstruct` first")  # fmt: skip
    # Only the name, source and lineage bindings are read: the render does not read the images.
    frames = capture.load_frame_map(root / CAPTURE_DIR / FRAMES_DIR, verify=False)
    # A visual-only scene (no metric scale) does not render: ``reject_mm`` needs meters.
    scene = import_metric_scene(artifact, metadata, frames)
    state.scene = scene
    spec = recipe.render
    extra, sampling_scene, scale_source = _rejection_sources(ctx, scene)
    flags = ctx.receipt["flags"]
    identity = {"artifact_sha256": scene.artifact_sha256,
                "reconstruction_id": scene.reconstruction_id,
                "render": canonical_sha256(asdict(spec)), "extra_sources": len(extra),
                "scale_source": scale_source,
                "reject_scene_units_per_metre": sampling_scene.scene_units_per_metre,
                "capture_flags": flags}  # fmt: skip
    out = root / RENDER_DIR
    plan_path = out / RENDER_PLAN_NAME
    output = StageOutput(out, plan_path, exact=True, refusal="was planned from another "
                         "artifact, reconstruction or render settings")  # fmt: skip
    if reuse_or_refuse(output, identity, force) is not None:
        log(f"render already planned: {spec.count} views of reconstruction "
            f"{scene.reconstruction_id[:12]}")  # fmt: skip
        return
    clear(output)
    out.mkdir(parents=True)
    seed_views = _render_seed_views(recipe, frames)
    views = sample_novel_views(
        sampling_scene, extra, seed=spec.seed, count=spec.count, seed_views=seed_views,
        perturb_deg=spec.perturb_deg, perturb_m=spec.perturb_m, reject_deg=spec.reject_deg,
        reject_mm=spec.reject_mm,
    )  # fmt: skip
    rejections = sum(view.rejected_before for view in views)
    publish_json(out / "views.json", [view.to_record() for view in views])
    publish_json(out / "sampling.json", {
        "seed": spec.seed, "count": spec.count, "near_source_rejections": rejections,
        "source_count": len(scene.cameras) + len(extra), "extra_sources": len(extra),
        "seed_views": seed_views, "perturb_deg": spec.perturb_deg,
        "perturb_m_scene_units": spec.perturb_m, "reject_deg": spec.reject_deg,
        "reject_mm": spec.reject_mm, "scene_units_per_metre": sampling_scene.scene_units_per_metre,
        "scale_source": scale_source, "exclusion": EXCLUSION,
        "reconstruction_id": scene.reconstruction_id, "artifact_sha256": scene.artifact_sha256,
        "capture_flags": flags,
    })  # fmt: skip
    trajectory = write_rig_trajectory(views, scene, scene.camera, out / "trajectory.json")
    command = _render_backend(recipe, cache).command(scene, trajectory, out)
    publish_json(out / "command.json", command)
    publish_json(plan_path, identity)
    log(f"render planned: {len(views)} novel views ({rejections} near-source rejections) of "
        f"{len(scene.cameras)} training cameras and {len(extra)} robot poses; scene units per "
        f"meter {sampling_scene.scene_units_per_metre} ({scale_source})")  # fmt: skip


def _write_render_receipt(out: Path, recipe: NurecRecipe, result: Any) -> None:  # noqa: ANN401
    """Write the render receipt: the plan, the NRE image and every frame's SHA-256.

    It holds no time or host, so a bit-identical re-render writes the same receipt and keeps
    the labels made from the earlier frames; any changed frame changes it.
    """
    publish_json(out / RENDER_RECEIPT_NAME, {
        "stage": "render", "plan_sha256": sha256_file(out / RENDER_PLAN_NAME),
        "image_digest": recipe.render.image_digest, "count": len(result.frames),
        "width": result.width, "height": result.height,
        "frames_sha256": {frame.name: sha256_file(frame) for frame in result.frames},
    })  # fmt: skip


def _nurec_render(
    root: Path, recipe: NurecRecipe, cache: Path | None, force: bool, state: _RenderState
) -> None:
    from ...nurec.render import RGB_DIR, SENSOR_NAME, RenderResult

    out = root / RENDER_DIR
    camera = state.scene.camera
    rgb = out / RGB_DIR
    # Bespoke: completeness of the frames decides reuse, not a recorded identity (the plan's).
    try:
        done = RenderResult.collect(rgb / SENSOR_NAME, count=recipe.render.count,
                                    width=camera.width, height=camera.height)  # fmt: skip
    except ValueError:
        done = None
    if done is not None and not force:
        if not (out / RENDER_RECEIPT_NAME).is_file():
            # Complete frames of this plan (a new plan clears the render) without a receipt.
            _write_render_receipt(out, recipe, done)
        log(f"already rendered: {len(done.frames)} frames in {rgb / SENSOR_NAME}")
        return
    (out / RENDER_RECEIPT_NAME).unlink(missing_ok=True)
    if rgb.exists() and force:
        shutil.rmtree(rgb)
    result = _render_backend(recipe, cache).render(state.scene, out / "trajectory.json", out)
    _write_render_receipt(out, recipe, result)
    log(f"rendered {len(result.frames)} frames ({result.width}x{result.height}) into "
        f"{rgb / SENSOR_NAME}")  # fmt: skip


def _render_receipt_sha256(root: Path) -> str:
    """The SHA-256 of the render receipt (the frames the labels read).

    Raises:
        FileNotFoundError: When the render step has not finished.
    """
    path = root / RENDER_DIR / RENDER_RECEIPT_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: the render step did not finish (rerun "
                                "`doper prepare --stage render` on a docker + GPU "
                                "host)")  # fmt: skip
    return sha256_file(path)


def _nurec_render_steps(ctx: CaptureContext, args: argparse.Namespace) -> list[Step]:
    from ...nurec.render import RGB_DIR, SENSOR_NAME

    root, recipe = ctx.root, ctx.recipe

    if args.artifact is None:
        raise ValueError("--stage render needs --artifact PATH: the trained NuRec artifact "
                         "(last.usdz) of the reconstruction")  # fmt: skip
    out, spec, state = root / RENDER_DIR, recipe.render, _RenderState()
    artifact, cache = Path(args.artifact), args.render_cache
    plan_inputs = {"artifact": artifact, "reconstruction": root / RECONSTRUCTION_DIR,
                   "frames": root / CAPTURE_DIR / FRAMES_DIR, "count": spec.count,
                   "seed": spec.seed, "seed_views": list(spec.seed_views),
                   "perturb": f"+/-{spec.perturb_deg} deg, +/-{spec.perturb_m} scene units",
                   "reject": f"within {spec.reject_deg} deg AND {spec.reject_mm} mm"}  # fmt: skip
    plan_outputs = {name: out / f"{name}.json"
                    for name in ("views", "sampling", "trajectory", "command", "plan")}  # fmt: skip
    render_inputs = {"image": spec.image_digest, "trajectory": out / "trajectory.json",
                     "host": "docker + NVIDIA GPU + NGC access to the image"}  # fmt: skip
    if cache is not None:
        render_inputs["model cache"] = f"{cache} -> {RENDER_CACHE_TARGET}"
    plan_force, frames_force = force_flags(args, force_step_keys("render", recipe))
    return [
        Step("sample novel views and write the NRE rig trajectory", plan_inputs, plan_outputs,
             lambda: _nurec_render_plan(ctx, artifact, cache, plan_force, state)),
        Step("render the novel views with the NRE CLI", render_inputs,
             {"frames": out / RGB_DIR / SENSOR_NAME, "receipt": out / RENDER_RECEIPT_NAME},
             lambda: _nurec_render(root, recipe, cache, frames_force, state)),
    ]  # fmt: skip


def _render_plan_of(root: Path, recipe: NurecRecipe, scene: Any) -> dict[str, Any]:  # noqa: ANN401
    """The render plan of ``scene`` (its ``plan.json``).

    Raises:
        FileNotFoundError: When the render stage has not planned.
        ValueError: When it planned another artifact or reconstruction.
    """
    plan_path = root / RENDER_DIR / RENDER_PLAN_NAME
    if not plan_path.is_file():
        raise FileNotFoundError(f"{plan_path} is missing: run `doper prepare {recipe.name} --stage "
                                "render --artifact PATH` first")  # fmt: skip
    plan = read_json(plan_path)
    if (plan.get("artifact_sha256"), plan.get("reconstruction_id")) != (
            scene.artifact_sha256, scene.reconstruction_id):  # fmt: skip
        raise ValueError(f"{plan_path} planned artifact {plan.get('artifact_sha256')} of "
                         f"reconstruction {plan.get('reconstruction_id')}, not --artifact "
                         f"{scene.artifact_sha256} of {scene.reconstruction_id}; rerun the "
                         "render stage")  # fmt: skip
    return plan
