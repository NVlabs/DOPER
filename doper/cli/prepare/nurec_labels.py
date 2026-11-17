# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NuRec ``labels`` stage of ``doper prepare``.

``labels`` labels the planned novel views (``doper.nurec.labels``). Its geometry step imports
``--artifact`` again, requires the render plan of that artifact and the registration
(``<root>/registration/registration.json`` once accepted, else the capture bundle's reviewed
one; either must name this artifact), and writes ``<root>/labels/geometry/``:
``geometry_labels.json`` (per view: metric ``T_camera_object_m``, keypoints, ray visibility,
unclipped CAD bbox, ``K``, ``reconstruction_id`` and ``lineage``), the half-resolution CAD
silhouettes ``cad_masks/NNNNNN.png`` and ``receipt.json`` (inputs, ``capture_flags``). The
scene must be metric (a visual-only scene is refused). With ``labels.mask_mode: sam2`` the
masks step segments the rendered frames with SAM2 (``doper.nurec.masks``; the
``nurec-sam`` extra, a CUDA GPU and ``--sam2-weights``) into ``<root>/labels/masks/``. The
qualification (``doper.nurec.package``) writes ``<root>/labels/qualified/qualified_labels.json``,
and the packaging step writes the adaptation dataset ``<root>/adaptation/dataset`` (the split by
lineage groups, else the recipe's one-reconstruction shuffle), its QC
``<root>/adaptation/dataset_qc.json`` and ``<root>/adaptation/receipt.json``. ``cad_only`` has
no masks step and its dataset has no ``mask_file`` (every receipt records
``reference_path: false``). Each step keeps outputs made from the same inputs; other inputs need
``--force``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from typing import Any

from ...bop.sources import canonical_sha256, publish_json, sha256_file
from ...console import log
from ...keypoints import KEYPOINTS_NAME
from ...recipes.nurec_schema import NurecRecipe
from ..common import (
    MissingExtraError,
    Step,
    read_json,
)
from ._names import (
    DATASET_QC_NAME,
    LABEL_WORKERS,
    LABELS_DIR,
    LABELS_GEOMETRY_DIR,
    LABELS_MASKS_DIR,
    LABELS_QUALIFIED_DIR,
    LABELS_RECEIPT_NAME,
    QUALIFIED_NAME,
    RECONSTRUCTION_DIR,
    REGISTRATION_DIR,
    RENDER_DIR,
    force_flags,
    force_step_keys,
)
from .nurec_capture import CaptureContext, _bundle_keypoints, _bundle_mesh
from .nurec_render import _render_plan_of, _render_receipt_sha256
from .receipts import StageOutput, clear, keep_step, reuse_or_refuse


def _nurec_geometry_labels(ctx: CaptureContext, artifact: Path, force: bool) -> None:
    from ...nurec.labels import (
        CAD_MASK_DIR,
        CONVENTION,
        DEFAULT_MASK_SCALE,
        GEOMETRY_LABELS_NAME,
        geometry_labels_batch,
    )
    from ...nurec.registration import read_obj_mesh, resolve_registration
    from ...nurec.render import NovelView
    from ...nurec.scene import import_metric_scene
    from ...reconstruction.metadata import METADATA_NAME

    root, recipe, receipt, frames = ctx.root, ctx.recipe, ctx.receipt, ctx.frames
    # Labels are metric: a visual-only scene (no metric scale) is refused before labeling.
    scene = import_metric_scene(artifact, root / RECONSTRUCTION_DIR / METADATA_NAME, frames)
    _render_plan_of(root, recipe, scene)
    image_set, _ = ctx.image_set_and_robot
    registration = resolve_registration(root / REGISTRATION_DIR, image_set.root, scene=scene)
    keypoints = _bundle_keypoints(image_set.root, recipe, "the labels")
    mesh_path = _bundle_mesh(image_set.root, recipe, "the labels")
    views_path = root / RENDER_DIR / "views.json"
    tolerance = recipe.labels.visibility_tolerance_m
    identity = {"artifact_sha256": scene.artifact_sha256,
                "reconstruction_id": scene.reconstruction_id,
                "views_sha256": sha256_file(views_path),
                "registration_sha256": canonical_sha256(registration.to_record()),
                "registration_status": registration.status, "mesh_sha256": sha256_file(mesh_path),
                "keypoints_points_sha256": recipe.object.keypoints_points_sha256,
                "visibility_tolerance_m": tolerance, "mask_scale": DEFAULT_MASK_SCALE,
                "camera": scene.camera.describe()}  # fmt: skip
    out = root / LABELS_DIR / LABELS_GEOMETRY_DIR
    receipt_path = out / LABELS_RECEIPT_NAME
    output = StageOutput(out, receipt_path, refusal="was labeled from another artifact, render "
                         "plan, registration, mesh or settings")  # fmt: skip
    kept = reuse_or_refuse(output, identity, force)
    if kept is not None:
        log(f"already labeled: {kept.get('count')} views in {out}")
        return
    clear(output)
    out.mkdir(parents=True)
    views = [NovelView.from_record(row) for row in read_json(views_path)]
    workers = max(1, min(LABEL_WORKERS, os.cpu_count() or 1))
    labels = geometry_labels_batch(
        views, scene, registration, read_obj_mesh(mesh_path), keypoints, scene.camera,
        tolerance_m=tolerance, mask_scale=DEFAULT_MASK_SCALE, workers=workers,
        mask_dir=out / CAD_MASK_DIR,
    )  # fmt: skip
    publish_json(out / GEOMETRY_LABELS_NAME, [label.to_record() for label in labels])
    visible = [int(label.ray_visible.sum()) for label in labels]
    publish_json(receipt_path, {
        "stage": "labels", "step": "geometry", **identity, "count": len(labels),
        "convention": CONVENTION, "mask_dir": CAD_MASK_DIR,
        "ray_visible_min": min(visible, default=0), "ray_visible_max": max(visible, default=0),
        "capture_flags": receipt["flags"],
    })  # fmt: skip
    log(f"geometry labels written for {len(labels)} views in {out} ({registration.status} "
        f"registration, {registration.scene_units_per_metre:.6g} scene units per meter; ray-"
        f"visible keypoints {min(visible, default=0)}-{max(visible, default=0)})")  # fmt: skip


def _geometry_rows(root: Path) -> tuple[list[Any], str]:
    """The geometry labels and the SHA-256 of their receipt.

    Raises:
        FileNotFoundError: When the geometry step has not finished.
    """
    from ...nurec.labels import GEOMETRY_LABELS_NAME, ViewLabel

    geometry = root / LABELS_DIR / LABELS_GEOMETRY_DIR
    receipt = geometry / LABELS_RECEIPT_NAME
    if not receipt.is_file():
        raise FileNotFoundError(f"{receipt} is missing: the geometry labels did not finish")
    rows = read_json(geometry / GEOMETRY_LABELS_NAME)
    return [ViewLabel.from_record(row) for row in rows], sha256_file(receipt)


def _render_frame(root: Path, index: int) -> Path:
    """The rendered RGB frame of novel view ``index``.

    Raises:
        FileNotFoundError: When the render stage has not rendered it.
    """
    from ...nurec.render import FRAME_PATTERN, RGB_DIR, SENSOR_NAME

    path = root / RENDER_DIR / RGB_DIR / SENSOR_NAME / FRAME_PATTERN.format(index=index)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing: the labels need the rendered novel views "
            "(`doper prepare --stage render` on a docker + GPU host)"
        )
    return path


def _sam2_version() -> str:
    """The installed sam2 version.

    Raises:
        MissingExtraError: Without the nurec-sam extra.
    """
    from importlib import metadata

    from ...nurec.masks import SAM2_EXTRA

    if importlib.util.find_spec("sam2") is None or importlib.util.find_spec("torch") is None:
        raise MissingExtraError(f"labels.mask_mode sam2 needs SAM2: install doper[{SAM2_EXTRA}] "
                                "(or set labels.mask_mode cad_only)")  # fmt: skip
    return metadata.version("sam2")


def _sam2_segmenter(weights: Path) -> Any:  # noqa: ANN401
    """The SAM2 image predictor (CUDA, bf16)."""
    from ...nurec.masks import SAM2_CONFIG, Sam2Segmenter

    return Sam2Segmenter(SAM2_CONFIG, weights=weights)


def _exclusion_record(recipe: NurecRecipe) -> dict[str, Any] | None:
    from dataclasses import asdict

    exclusion = recipe.labels.exclude_hsv
    return None if exclusion is None else asdict(exclusion)


def _nurec_sam_masks(root: Path, recipe: NurecRecipe, weights: Path, force: bool) -> None:
    from ...nurec.labels import CAD_MASK_DIR
    from ...nurec.masks import SAM2_CONFIG, SAM_SCORES_NAME, segment_views

    labels, geometry_sha = _geometry_rows(root)
    if not weights.is_file():
        raise FileNotFoundError(f"--sam2-weights {weights} is not a file")
    identity = {"geometry_receipt_sha256": geometry_sha,
                "render_receipt_sha256": _render_receipt_sha256(root),
                "sam2_config": SAM2_CONFIG, "sam2_version": _sam2_version(),
                "sam2_weights_sha256": sha256_file(weights),
                "exclude_hsv": _exclusion_record(recipe)}  # fmt: skip
    out = root / LABELS_DIR / LABELS_MASKS_DIR
    receipt_path = out / LABELS_RECEIPT_NAME
    if keep_step(out, receipt_path, identity, force, "segmented"):
        return
    frames = {label.index: _render_frame(root, label.index) for label in labels}
    cad = root / LABELS_DIR / LABELS_GEOMETRY_DIR / CAD_MASK_DIR
    rows = segment_views(labels, _sam2_segmenter(weights), image_path=lambda v: frames[v.index],
                         cad_mask_path=lambda v: cad / str(v.mask_file), out=out,
                         exclude_hsv=recipe.labels.exclude_hsv)  # fmt: skip
    publish_json(out / SAM_SCORES_NAME, [{"index": r["index"], "score": r["score"]} for r in rows])
    publish_json(out / "prompts.json", [{"index": r["index"], **r["prompts"]} for r in rows])
    scores = [r["score"] for r in rows]
    publish_json(receipt_path, {"stage": "labels", "step": "masks", **identity,
                                "mask_mode": "sam2", "reference_path": True, "count": len(rows),
                                "score_min": min(scores, default=None),
                                "score_max": max(scores, default=None)})  # fmt: skip
    log(f"SAM2 masks written for {len(rows)} views in {out}")


def _qualify_identity(root: Path, recipe: NurecRecipe, geometry_sha: str) -> dict[str, Any]:
    from dataclasses import asdict

    masks = root / LABELS_DIR / LABELS_MASKS_DIR / LABELS_RECEIPT_NAME
    sam2 = recipe.labels.mask_mode == "sam2"
    if sam2 and not masks.is_file():
        raise FileNotFoundError(f"{masks} is missing: the SAM2 masks did not finish")
    return {"geometry_receipt_sha256": geometry_sha,
            "render_receipt_sha256": _render_receipt_sha256(root),
            "masks_receipt_sha256": sha256_file(masks) if sam2 else None,
            "labels": asdict(recipe.labels)}  # fmt: skip


def _nurec_qualify(root: Path, recipe: NurecRecipe, force: bool) -> None:
    from ...nurec.labels import CAD_MASK_DIR
    from ...nurec.masks import SAM_MASK_PATTERN
    from ...nurec.package import assess_files

    labels, geometry_sha = _geometry_rows(root)
    identity = _qualify_identity(root, recipe, geometry_sha)
    out = root / LABELS_DIR / LABELS_QUALIFIED_DIR
    receipt_path = out / LABELS_RECEIPT_NAME
    if keep_step(out, receipt_path, identity, force, "qualified"):
        return
    spec, sam2 = recipe.labels, recipe.labels.mask_mode == "sam2"
    cad = root / LABELS_DIR / LABELS_GEOMETRY_DIR / CAD_MASK_DIR
    masks = root / LABELS_DIR / LABELS_MASKS_DIR
    qualified = [assess_files(
        label, image=_render_frame(root, label.index), cad_mask=cad / str(label.mask_file),
        mask=masks / SAM_MASK_PATTERN.format(index=label.index) if sam2 else None,
        erode_px=spec.erode_px, min_visible=spec.min_visible, min_coverage=spec.min_coverage,
        exclude_hsv=spec.exclude_hsv) for label in labels]  # fmt: skip
    out.mkdir(parents=True)
    publish_json(out / QUALIFIED_NAME, [item.to_record() for item in qualified])
    accepted = sum(item.accepted for item in qualified)
    publish_json(
        receipt_path,
        {
            "stage": "labels",
            "step": "qualify",
            **identity,
            "mask_mode": spec.mask_mode,
            "reference_path": sam2,
            "total": len(qualified),
            "accepted": accepted,
            "visible_counts": [q.visible_count for q in qualified],
        },
    )
    log(f"qualified {accepted} of {len(qualified)} views ({spec.mask_mode} masks)")


def _nurec_package(ctx: CaptureContext, force: bool) -> None:
    from dataclasses import asdict

    from ...nurec.labels import CAD_MASK_DIR, ViewLabel
    from ...nurec.masks import SAM_MASK_PATTERN
    from ...nurec.package import Qualified, package_dataset, qc_dataset
    from ..common import nurec_dataset_dir

    root, recipe = ctx.root, ctx.recipe
    qualified_receipt = root / LABELS_DIR / LABELS_QUALIFIED_DIR / LABELS_RECEIPT_NAME
    if not qualified_receipt.is_file():
        raise FileNotFoundError(f"{qualified_receipt} is missing: the qualification did not "
                                "finish")  # fmt: skip
    image_set, _ = ctx.image_set_and_robot
    _bundle_keypoints(image_set.root, recipe, "the adaptation dataset")
    keypoints = image_set.root / KEYPOINTS_NAME
    keypoints_sha = sha256_file(keypoints)
    spec, sam2 = recipe.labels, recipe.labels.mask_mode == "sam2"
    identity = {"qualified_receipt_sha256": sha256_file(qualified_receipt),
                "split": asdict(recipe.split), "category": recipe.object.category,
                "keypoints_sha256": keypoints_sha, "mask_mode": spec.mask_mode,
                "min_visible": spec.min_visible}  # fmt: skip
    dataset = nurec_dataset_dir(root, "adapt")
    receipt_path = dataset.parent / LABELS_RECEIPT_NAME
    if keep_step(dataset, receipt_path, identity, force, "packaged"):
        return
    # A QC summary of an earlier dataset must not stand next to this one if its QC fails.
    qc_path = dataset.parent / DATASET_QC_NAME
    qc_path.unlink(missing_ok=True)
    cad = root / LABELS_DIR / LABELS_GEOMETRY_DIR / CAD_MASK_DIR
    masks = root / LABELS_DIR / LABELS_MASKS_DIR
    rows = read_json(root / LABELS_DIR / LABELS_QUALIFIED_DIR / QUALIFIED_NAME)
    qualified = []
    for row in rows:
        label = ViewLabel.from_record(row)
        qualified.append(Qualified.from_record(
            row, image=_render_frame(root, label.index), cad_mask=cad / str(label.mask_file),
            mask=masks / SAM_MASK_PATTERN.format(index=label.index) if sam2 else None))  # fmt: skip
    split = recipe.split
    package_dataset(qualified, dataset, keypoints_file=keypoints, seed=split.seed,
                    train=split.train, val=split.val, category=recipe.object.category,
                    single_reconstruction_split=split.single_reconstruction_split,
                    mask_mode=spec.mask_mode, min_visible=spec.min_visible)  # fmt: skip
    qc = qc_dataset(dataset, keypoints_sha)
    publish_json(qc_path, qc)
    package = read_json(dataset / "package.json")
    publish_json(receipt_path, {"stage": "labels", "step": "package", **identity,
                                "package_sha256": sha256_file(dataset / "package.json"),
                                "qc_sha256": sha256_file(qc_path),
                                "reference_path": sam2, "lineage_split": package["lineage_split"],
                                "lineage_caveat": package["lineage_caveat"],
                                "qc": qc["status"], "train": package["train"],
                                "val": package["val"],
                                "capture_flags": ctx.receipt["flags"]})  # fmt: skip
    log(f"adaptation dataset written to {dataset}: {package['train']} train / {package['val']} "
        f"val views, {spec.mask_mode} masks, lineage split {package['lineage_split']}; QC "
        "passed")  # fmt: skip


def _nurec_labels_steps(ctx: CaptureContext, args: argparse.Namespace) -> list[Step]:
    """The labels stage's steps: geometry labels, SAM2 masks, qualification, packaging and QC.

    Raises:
        ValueError: Without ``--artifact``, or with ``--sam2-weights`` missing for
            ``mask_mode: sam2`` or given for ``cad_only``.
    """
    from ...nurec.labels import CAD_MASK_DIR, DEFAULT_MASK_SCALE, GEOMETRY_LABELS_NAME
    from ...nurec.masks import SAM2_CONFIG, SAM_SCORES_NAME
    from ...nurec.render import RGB_DIR, SENSOR_NAME
    from ..common import nurec_dataset_dir

    root, recipe = ctx.root, ctx.recipe
    if args.artifact is None:
        raise ValueError("--stage labels needs --artifact PATH: the trained NuRec artifact "
                         "(last.usdz) the render stage planned")  # fmt: skip
    spec = recipe.labels
    weights = getattr(args, "sam2_weights", None)
    if spec.mask_mode == "sam2" and weights is None:
        raise ValueError("labels.mask_mode sam2 needs --sam2-weights PATH (the SAM2 checkpoint, "
                         "sam2.1_hiera_small.pt)")  # fmt: skip
    if spec.mask_mode != "sam2" and weights is not None:
        raise ValueError(f"--sam2-weights is for labels.mask_mode sam2, not {spec.mask_mode}")
    artifact, out = Path(args.artifact), root / LABELS_DIR / LABELS_GEOMETRY_DIR
    inputs = {"artifact": artifact, "views": root / RENDER_DIR / "views.json",
              "registration": f"{root / REGISTRATION_DIR / 'registration.json'} (else the "
              "bundle's reviewed one)", "mesh": f"<bundle>/{recipe.object.mesh}",
              "visibility tolerance": f"{spec.visibility_tolerance_m} m",
              "mask scale": DEFAULT_MASK_SCALE}  # fmt: skip
    outputs = {"labels": out / GEOMETRY_LABELS_NAME, "masks": out / CAD_MASK_DIR,
               "receipt": out / LABELS_RECEIPT_NAME}  # fmt: skip
    frames = root / RENDER_DIR / RGB_DIR / SENSOR_NAME
    masks, qualified = (
        root / LABELS_DIR / LABELS_MASKS_DIR,
        root / LABELS_DIR / LABELS_QUALIFIED_DIR,
    )
    dataset = nurec_dataset_dir(root, "adapt")
    keys = force_step_keys("labels", recipe)
    forced = dict(zip(keys, force_flags(args, keys), strict=True))
    steps = [
        Step(
            "label the novel views from the registered CAD geometry",
            inputs,
            outputs,
            lambda: _nurec_geometry_labels(ctx, artifact, forced["geometry"]),
        )
    ]
    if spec.mask_mode == "sam2":
        steps.append(Step(
            "segment the novel views with SAM2 (CUDA)",
            {"frames": frames, "config": SAM2_CONFIG, "weights": weights,
             "exclude_hsv": _exclusion_record(recipe)},
            {"masks": masks, "scores": masks / SAM_SCORES_NAME,
             "receipt": masks / LABELS_RECEIPT_NAME},
            lambda: _nurec_sam_masks(
                root, recipe, Path(str(weights)), forced["masks"])))  # fmt: skip
    qualify_inputs: dict[str, Any] = {
        "frames": frames,
        "labels": outputs["labels"],
        "cad masks": outputs["masks"],
    }
    if spec.mask_mode == "sam2":
        qualify_inputs["masks"] = masks
    qualify_inputs.update(
        {
            "mask mode": f"{spec.mask_mode}"
            + ("" if spec.mask_mode == "sam2" else " (no mask supervision)"),
            "min visible": spec.min_visible,
            "min coverage": spec.min_coverage,
            "erode": f"{spec.erode_px} px kernel",
        }
    )
    steps.append(Step(
        "qualify the views (visible keypoints and mask coverage)",
        qualify_inputs,
        {"qualified": qualified / QUALIFIED_NAME, "receipt": qualified / LABELS_RECEIPT_NAME},
        lambda: _nurec_qualify(root, recipe, forced["qualify"])))  # fmt: skip
    split = recipe.split
    steps.append(Step(
        "package the adaptation dataset and QC it",
        {"qualified": qualified / QUALIFIED_NAME, "keypoints": "<bundle>/keypoints_3d.json",
         "split": f"seed {split.seed}: {split.train} train / {split.val} val (lineage groups, "
                  f"else one-reconstruction shuffle: {split.single_reconstruction_split})",
         "category": recipe.object.category},
        {"dataset": dataset, "qc": dataset.parent / DATASET_QC_NAME,
         "receipt": dataset.parent / LABELS_RECEIPT_NAME},
        lambda: _nurec_package(ctx, forced["package"])))  # fmt: skip
    return steps
