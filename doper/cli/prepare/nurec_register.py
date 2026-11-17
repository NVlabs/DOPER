# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NuRec ``register`` stage of ``doper prepare``.

``register`` places the CAD model in the trained NuRec scene (``doper.nurec.registration``):
the DOPER keypoints of a ``doper evaluate`` frames.json of the capture (``--keypoint-frames``)
are triangulated with the scene cameras of ``--artifact`` and fitted from the CAD keypoints, the
scale is fixed from the robot camera centers (else the reconstruction's anchor) and the pose is
refined against image edges. It writes ``<root>/registration/candidate.json`` (with the
triangulation, robot scale audit and restart diagnostic) and nothing the labels read: a reviewer
accepts it with ``--accept-registration --reviewer-note TEXT``, which writes
``registration.json``. A capture bundle's reviewed ``registration.json`` serves instead.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from ...bop.sources import canonical_sha256, publish_json, sha256_file
from ...console import log
from ..common import (
    Step,
    read_json,
)
from ._names import CAPTURE_DIR, RECONSTRUCTION_DIR, REGISTRATION_DIR
from .nurec_capture import (
    CaptureContext,
    _bundle_keypoints,
    _bundle_mesh,
    _capture_names,
)
from .receipts import StageOutput, reuse_or_refuse


def _keypoint_frames(path: Path, image_set: Any) -> list[Any]:  # noqa: ANN401
    """The DOPER frames of ``--keypoint-frames``, each bound to a capture image by SHA-256.

    Raises:
        ValueError: For a frame of an image that is not in the capture or whose SHA-256 differs.
    """
    from ...evaluation.relative_pose import FramePose

    frames = [FramePose.from_record(row) for row in read_json(path)]
    shas = {image.name: image.sha256 for image in image_set.images}
    wrong = [f.image for f in frames if shas.get(f.image) != f.sha256]
    if wrong:
        raise ValueError(
            f"{path}: frames {wrong} are not images of the capture bundle (name and "
            "SHA-256); pass the frames.json of `doper evaluate` on this bundle"
        )
    return frames


def _registration_scale(robot: Any, scene: Any) -> tuple[float, str, Any]:  # noqa: ANN401
    """The registration's fixed scale: the robot audit's, else the reconstruction anchor's.

    Returns:
        ``(scene_units_per_metre, scale_source, anchor or None)``.

    Raises:
        MetricAlignmentError: Without a robot reference on a ``visual_only`` scene.
    """
    from ...nurec.registration import robot_scale_audit

    if robot is not None:
        anchor = robot_scale_audit(robot, scene)
        return anchor.scene_units_per_metre, "robot_audit", anchor
    return scene.scene_from_metres(1.0), "reconstruction_anchor", None


def _write_registration(
    out: Path, seed: Any, anchor: Any, candidate: Any, restarts: list[dict[str, Any]],  # noqa: ANN401
    flags: dict[str, Any],
) -> None:  # fmt: skip
    from ...nurec.registration import CANDIDATE_NAME

    # Without candidate.json the directory is an unfinished earlier run and is rebuilt.
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    publish_json(out / "triangulation.json", {**seed.to_record(), "capture_flags": flags})
    if anchor is not None:
        publish_json(out / "robot_scale_audit.json", {**anchor.to_record(), "capture_flags": flags})
    publish_json(out / "restart_audit.json", {
        "perturbations": "independent uniform per rotation-vector component (degrees) and per "
        "translation axis (scene units) around the candidate", "results": restarts,
        "candidate_cost": candidate.fit["cost"], "capture_flags": flags,
    })  # fmt: skip
    candidate.write(out / CANDIDATE_NAME)


def _nurec_register(ctx: CaptureContext, artifact: Path, frames_path: Path, force: bool) -> None:
    from dataclasses import asdict, replace

    from ...nurec.registration import (
        CANDIDATE_NAME,
        fit_keypoint_similarity,
        read_obj_mesh,
        refine_edges,
        restart_audit,
    )
    from ...nurec.scene import import_metric_scene
    from ...reconstruction.metadata import METADATA_NAME

    root, recipe = ctx.root, ctx.recipe
    metadata = root / RECONSTRUCTION_DIR / METADATA_NAME
    if not metadata.is_file():
        raise FileNotFoundError(f"{metadata} is missing: run `doper prepare {recipe.name} --stage "
                                "reconstruct` first")  # fmt: skip
    receipt, frames = ctx.receipt, ctx.frames
    image_set, robot = ctx.image_set_and_robot
    keypoints = _bundle_keypoints(image_set.root, recipe, "the registration")
    mesh_path = _bundle_mesh(image_set.root, recipe, "the registration")
    # Render and labels refuse a visual-only scene (no metric scale), so register refuses it too
    # rather than producing a candidate no later stage can use.
    scene = import_metric_scene(artifact, metadata, frames)
    spec = recipe.registration
    identity = {"artifact_sha256": scene.artifact_sha256,
                "reconstruction_id": scene.reconstruction_id,
                "bundle_sha256": receipt["bundle_sha256"],
                "keypoint_frames_sha256": sha256_file(frames_path),
                "mesh_sha256": sha256_file(mesh_path),
                "registration": canonical_sha256(spec.as_record())}  # fmt: skip
    out = root / REGISTRATION_DIR
    output = StageOutput(
        out, out / CANDIDATE_NAME, exact=True, refusal="was registered from other inputs or "
        "settings", verb="replace it (and any accepted registration)",
        read=lambda path: {"inputs": read_json(path).get("inputs")},
    )  # fmt: skip
    if reuse_or_refuse(output, {"inputs": identity}, force) is not None:
        log(f"already registered: the candidate {out / CANDIDATE_NAME} is current")
        return
    names = image_set.names
    seed = fit_keypoint_similarity(
        _keypoint_frames(frames_path, image_set), scene, keypoints, image_set.camera.matrix,
        exclude_views=_capture_names(spec.exclude_views, names, "registration.exclude_views"),
        max_reproj_px=spec.max_reproj_px,
    )  # fmt: skip
    scale, scale_source, anchor = _registration_scale(robot, scene)
    candidate = refine_edges(
        seed, image_set, scene, read_obj_mesh(mesh_path), fixed_scale=scale,
        fit_views=_capture_names(spec.fit_views, names, "registration.fit_views"),
        holdout_views=_capture_names(spec.holdout_views, names, "registration.holdout_views"),
        canny=spec.canny, boundary_px=spec.boundary_px, points=spec.points,
        f_scale=spec.f_scale, diff_step=spec.diff_step, max_nfev=spec.max_nfev,
        scale_source=scale_source,
    )  # fmt: skip
    restarts = restart_audit(candidate, **asdict(spec.restarts))
    candidate = replace(
        candidate, provenance={"inputs": identity, "capture_flags": receipt["flags"]}
    )
    _write_registration(out, seed, anchor, candidate, restarts, receipt["flags"])
    held = [v["refined_median_px"] for v in candidate.per_view_edges if not v["fit_view"]]
    lower = sum(r["cost"] < candidate.fit["cost"] for r in restarts)
    log(f"registration candidate written to {out / CANDIDATE_NAME}: seed scale "
        f"{seed.scale:.6g} (keypoint RMS {seed.rms_scene * 1000:.3g} scene mm), fixed scale "
        f"{scale:.6g} scene units per meter ({scale_source}), held-out median edge distances "
        f"{[round(h, 2) for h in held]} px, {lower} of {len(restarts)} restarts reached a lower "
        f"cost. Review it, then run `doper prepare {recipe.name} --stage register --root {root} "
        "--accept-registration --reviewer-note TEXT`")  # fmt: skip


def _nurec_accept(root: Path, note: str, force: bool) -> None:
    from ...nurec.registration import (
        CANDIDATE_NAME,
        REGISTRATION_NAME,
        Registration,
        RegistrationCandidate,
        accept_registration,
    )

    out = root / REGISTRATION_DIR
    path = out / CANDIDATE_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: run --stage register with --artifact and "
                                "--keypoint-frames first, then review its candidate")  # fmt: skip
    digest = sha256_file(path)
    accepted = out / REGISTRATION_NAME
    # Bespoke: a different candidate re-accepts without refusing, so there is no refusal to share.
    if accepted.is_file() and not force:
        existing = Registration.read(accepted)
        if existing.provenance.get("candidate_sha256") == digest:
            log(f"already accepted: {accepted} (rerun with --force to replace the note)")
            return
    registration = accept_registration(RegistrationCandidate.read(path), note,
                                       candidate_sha256=digest, out=accepted)  # fmt: skip
    log(f"registration accepted: {accepted} (scene units per meter "
        f"{registration.scene_units_per_metre:.6g}, reviewer note recorded)")  # fmt: skip


def _nurec_register_step(ctx: CaptureContext, args: argparse.Namespace) -> Step:
    from ...nurec.registration import CANDIDATE_NAME, REGISTRATION_NAME

    root, recipe = ctx.root, ctx.recipe
    out, spec = root / REGISTRATION_DIR, recipe.registration
    if args.accept_registration:
        given = [flag for name, flag in (("artifact", "--artifact"),
                                         ("keypoint_frames", "--keypoint-frames"))
                 if getattr(args, name, None)]  # fmt: skip
        if given:
            raise ValueError(f"--accept-registration accepts the candidate already in {out}; "
                             f"it takes no {given} (they recompute the candidate)")  # fmt: skip
        if not (args.reviewer_note or "").strip():
            raise ValueError(
                "--accept-registration needs --reviewer-note TEXT: what the "
                "reviewer checked (overlays, held-out edge distances, restarts)"
            )
        note = args.reviewer_note
        return Step("accept the reviewed registration candidate",
                    {"candidate": out / CANDIDATE_NAME, "reviewer note": note},
                    {"registration": out / REGISTRATION_NAME},
                    lambda: _nurec_accept(root, note, args.force))  # fmt: skip
    if args.reviewer_note is not None:
        raise ValueError("--reviewer-note goes with --accept-registration")
    missing = [flag for name, flag in (("artifact", "--artifact"),
                                       ("keypoint_frames", "--keypoint-frames"))
               if getattr(args, name, None) is None]  # fmt: skip
    if missing:
        raise ValueError(
            f"--stage register needs {' and '.join(missing)}: the trained NuRec "
            "artifact (last.usdz) and a `doper evaluate` frames.json of the capture"
        )
    artifact, frames = Path(args.artifact), Path(args.keypoint_frames)
    inputs = {"artifact": artifact, "keypoint frames": frames,
              "reconstruction": root / RECONSTRUCTION_DIR, "capture": root / CAPTURE_DIR,
              "mesh": f"<bundle>/{recipe.object.mesh}", "fit views": list(spec.fit_views),
              "holdout views": list(spec.holdout_views),
              "scale": "robot camera centers (else the reconstruction anchor)"}  # fmt: skip
    outputs = {name: out / f"{name}.json" for name in
               ("triangulation", "robot_scale_audit", "restart_audit")}  # fmt: skip
    outputs["candidate"] = out / CANDIDATE_NAME
    return Step("register the CAD model to the scene (a candidate for review)", inputs, outputs,
                lambda: _nurec_register(ctx, artifact, frames, args.force))  # fmt: skip
