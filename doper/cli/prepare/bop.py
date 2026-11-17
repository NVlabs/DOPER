# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BOP recipes of ``doper prepare``: one object's steps from source to ``prepared.json``.

Order: fetch the pinned parts and record ``READY.json``; extract them; then per object resolve its
verified PBR source, select keypoints on the CAD (PLY exported to a binary STL, then the
hull-anchored selector in millimeters), audit them against the CAD, convert PBR to COCO, audit the
projections, split by whole scenes and record ``prepared.json``. HOT3D also stages its validation
and test manifests from the real clips.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any

from ...bop import sources
from ...bop.cad_keypoints import STL_NAME, select_cad_keypoints
from ...bop.extract import data_dir, zip_archives
from ...bop.sources import DatasetProfile, publish_json, sha256_file
from ...console import log
from ...keypoints import KEYPOINTS_NAME
from ...pseudo_labels._outdir import replace_output_dir
from ...recipes.inputs import (
    TrainingInputs,
    dataset_inputs,
    generated_inputs,
    generated_models,
    generated_view,
    planned_source,
)
from ...recipes.schema import Recipe, recipe_objects
from ..common import (
    DATASET_DIR,
    GENERATED_KEY,
    IDENTITY_NAME,
    PREPARED_NAME,
    Step,
    load,
    object_dir,
    read_json,
    require_resolving_links,
    verify_hashes,
)
from ._names import (
    CONVERTED_DIR,
    GEOMETRY_DIR,
    NUREC_FLAGS,
    ONBOARDING_PARTS,
    PROJECTION_NAME,
    SPEEDPLUS_FLAGS,
    TRAINING_PARTS,
    VISIBLE_DIR,
    refuse_flags,
)
from .bop_extras import (
    _extract,
    _fetch,
    _hot3d_steps,
    _keypoints_only_plan,
    _multisensor_plan,
    _onboarding_steps,
)


class _ObjectState:
    """Values one object's steps hand to each other at run time."""

    inputs: TrainingInputs | None = None
    done: bool = False


def _identity(
    recipe: Recipe, profile: DatasetProfile, root: Path, obj: int, inputs: TrainingInputs
) -> dict[str, Any]:
    mesh = inputs.models / f"obj_{obj:06d}.ply"
    identity = {
        "obj_id": obj,
        "dataset": profile.name,
        "profile_sha256": sources.profile_sha256(profile),
        # Relative to the dataset root, so a moved root keeps its prepared objects.
        "source": Path(os.path.relpath(inputs.train_pbr, root)).as_posix(),
        "source_ready_sha256": inputs.source_ready_sha256,
        "mesh_sha256": sha256_file(mesh),
        "keypoints": {"count": recipe.keypoints.count, "units": recipe.keypoints.source_units},
        "convert": {"val_scenes": recipe.convert.val_scenes, "seed": recipe.convert.seed},
    }
    if obj in recipe.generated_objects:
        spec = recipe.generated_spec(obj)
        ready = read_json(inputs.train_pbr.parent / sources.READY_NAME)
        identity["convert"] = {
            "split": "layout_modulo",
            "val_modulo": spec.val_modulo,
            "val_residue": spec.val_residue,
            "empty_masks": spec.empty_masks,
        }
        identity[GENERATED_KEY] = {
            **spec.as_record(),
            "render_identity_sha256": ready["render_identity_sha256"],
            "keypoints_sha256": ready["keypoints_sha256"],
        }
    return identity


def _resolve_source(
    recipe: Recipe,
    profile: DatasetProfile,
    root: Path,
    obj: int,
    force: bool,
    state: _ObjectState,
    render_root: Path | None = None,
) -> None:
    directory = object_dir(root, obj)
    if render_root is not None:
        inputs = generated_inputs(recipe, profile, root, obj, render_root, force=force)
    else:
        inputs = dataset_inputs(recipe, profile, root, obj)
    state.inputs = inputs
    if force and directory.exists():
        replace_output_dir(directory, True, [inputs.train_pbr, inputs.models])
    identity = _identity(recipe, profile, root, obj, inputs)
    # Bespoke: the identity file is separate from the completion receipt (prepared.json) and is
    # rewritten every run; force wipes the directory before it.
    path = directory / IDENTITY_NAME
    if path.exists() and read_json(path) != identity:
        raise ValueError(f"object {obj} identity changed; rerun with --force to rebuild it")
    publish_json(path, identity)
    if (directory / PREPARED_NAME).exists():
        verify_hashes(directory, read_json(directory / PREPARED_NAME)["hashes"])
        require_resolving_links(directory / DATASET_DIR, obj)
        state.done = True
        log(f"object {obj} already prepared and verified")


def _keypoints(
    recipe: Recipe, root: Path, obj: int, state: _ObjectState, render_root: Path | None = None
) -> None:
    from ...bop import audit, generated

    if state.done or state.inputs is None:
        return
    geometry = object_dir(root, obj) / GEOMETRY_DIR
    mesh = state.inputs.models / f"obj_{obj:06d}.ply"
    info = read_json(state.inputs.models / "models_info.json")
    keypoints = geometry / KEYPOINTS_NAME
    if not keypoints.exists():
        spec = recipe.keypoints
        staged = select_cad_keypoints(mesh, geometry, obj, spec.count, spec.source_units)
        problems = audit.geometry_audit(mesh, staged, info, obj)
        if problems:
            staged.unlink()
            raise ValueError(f"object {obj} keypoint audit failed: {problems}")
    if render_root is not None:
        # The campaign rendered and audited its frames with these points: train on the same.
        generated.require_same_layout(keypoints, Path(render_root) / generated.KEYPOINTS_PATH)
    report = audit.geometry_report(mesh, keypoints, info, obj)
    if report["problems"]:
        raise ValueError(f"object {obj} keypoint audit failed: {report['problems']}")
    publish_json(geometry / "audit.json", report)
    shutil.copy2(mesh, geometry / mesh.name)
    shutil.copy2(state.inputs.models / "models_info.json", geometry / "models_info.json")


def training_converted(recipe: Recipe, root: Path, obj: int) -> Path:
    """The converted root an object splits from.

    That is the positive-visible view (``converted_visible``) when the object excludes empty
    masks, else ``converted``.
    """
    name = VISIBLE_DIR if recipe.convert_for(obj).empty_masks == "exclude" else CONVERTED_DIR
    return object_dir(root, obj) / name


def _convert(recipe: Recipe, root: Path, obj: int, state: _ObjectState) -> None:
    from ...bop import convert, generated

    if state.done or state.inputs is None:
        return
    directory = object_dir(root, obj)
    converted = directory / CONVERTED_DIR
    if not converted.exists():
        convert.convert_pbr(
            state.inputs.train_pbr, obj, directory / GEOMETRY_DIR / KEYPOINTS_NAME, converted
        )
    if obj not in recipe.generated_objects:
        return
    policy = recipe.convert_for(obj).empty_masks
    if policy == "fail":
        generated.require_visible(converted, state.inputs.train_pbr, obj)
    elif policy == "exclude" and not (directory / VISIBLE_DIR).exists():
        summary = generated.exclude_invisible(
            converted, state.inputs.train_pbr, directory / VISIBLE_DIR, obj
        )
        log(f"object {obj}: excluded {summary['excluded_annotations']} annotations with an "
            "empty visible mask")  # fmt: skip


def _projection(recipe: Recipe, root: Path, obj: int, state: _ObjectState) -> None:
    from ...bop import audit

    if state.done or state.inputs is None:
        return
    directory = object_dir(root, obj)
    report = audit.projection_report(
        training_converted(recipe, root, obj),
        state.inputs.train_pbr,
        obj,
        # ``keep`` trains on empty masks, so the audit accepts those the GT info confirms.
        allow_source_empty_masks=recipe.convert.allow_source_empty_masks
        or recipe.convert_for(obj).empty_masks == "keep",
    )
    if report["problems"]:
        raise ValueError(f"object {obj} projection audit failed: {report['problems']}")
    publish_json(directory / PROJECTION_NAME, report)


def _split(recipe: Recipe, root: Path, obj: int, state: _ObjectState) -> None:
    from ...bop import generated, split

    if state.done:
        return
    directory = object_dir(root, obj)
    if (directory / DATASET_DIR).exists():
        return
    staging = directory / f".{DATASET_DIR}-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    converted = training_converted(recipe, root, obj)
    keypoints = directory / GEOMETRY_DIR / KEYPOINTS_NAME
    if recipe.convert_for(obj).split == "layout_modulo":
        spec = recipe.generated_spec(obj)
        generated.split_layouts(converted, staging, obj, keypoints, spec)
    else:
        split.split_scenes(
            converted,
            staging,
            obj,
            keypoints,
            val_scenes=recipe.convert.val_scenes,
            seed=recipe.convert.seed,
        )
    os.rename(staging, directory / DATASET_DIR)


def _membership(dataset: Path) -> dict[str, Any]:
    membership = {}
    for name in ("train", "val"):
        data = read_json(dataset / name / "annotations.json")
        if not data["images"] or not data["annotations"]:
            raise ValueError(f"empty {name} split; refuse training")
        membership[name] = {
            "images": len(data["images"]),
            "annotations": len(data["annotations"]),
            "scenes": sorted({a["bop"]["scene"] for a in data["annotations"]}),
        }
    if set(membership["train"]["scenes"]) & set(membership["val"]["scenes"]):
        raise ValueError("PBR scene leakage between train and val")
    return membership


def _receipt(recipe: Recipe, root: Path, obj: int, state: _ObjectState) -> None:
    from ...bop import generated

    if state.done:
        return
    directory = object_dir(root, obj)
    selection = f"{VISIBLE_DIR}/{generated.VISIBLE_SELECTION_NAME}"
    relative = [
        f"{GEOMETRY_DIR}/{KEYPOINTS_NAME}",
        f"{CONVERTED_DIR}/conversion_provenance.json",
        f"{DATASET_DIR}/train/annotations.json",
        f"{DATASET_DIR}/val/annotations.json",
        f"{DATASET_DIR}/{KEYPOINTS_NAME}",
    ]
    if training_converted(recipe, root, obj).name == VISIBLE_DIR:
        relative += [f"{VISIBLE_DIR}/annotations.json", selection]
    publish_json(
        directory / PREPARED_NAME,
        {
            "geometry": read_json(directory / GEOMETRY_DIR / "audit.json"),
            "projection": read_json(directory / PROJECTION_NAME),
            "membership": _membership(directory / DATASET_DIR),
            "hashes": {name: sha256_file(directory / name) for name in relative},
        },
    )


def _source_step(
    recipe: Recipe,
    profile: DatasetProfile,
    root: Path,
    obj: int,
    render_root: Path | None,
    run: Any,  # noqa: ANN401
) -> tuple[Step, Path]:
    """The source step of one object and the CAD directory it resolves."""
    directory = object_dir(root, obj)
    train_pbr = planned_source(recipe, profile, root, obj)
    if render_root is not None:
        spec = recipe.generated_spec(obj)
        models = generated_models(render_root, profile)
        inputs: dict[str, Any] = {
            "render root": render_root,
            "campaign": f"{spec.render_config}: {spec.layouts} layouts x "
            f"{spec.frames_per_layout} views, seed {spec.base_seed} + layout",
            "policy": "every layout receipt verified; none missing or recorded failed",
        }
        outputs: dict[str, Any] = {"view": generated_view(root, obj), "train_pbr": train_pbr}
    else:
        models = (
            train_pbr.parent / recipe.convert.models
            if recipe.convert.source_view == "hot3d"
            else data_dir(root) / recipe.convert.models
        )
        inputs = {"data": data_dir(root), "policy": recipe.convert.source_view}
        outputs = {"train_pbr": train_pbr}
    outputs.update(models=models, identity=directory / IDENTITY_NAME)
    return Step(f"source obj {obj}", inputs, outputs, run), models


def object_steps(
    recipe: Recipe,
    profile: DatasetProfile,
    root: Path,
    obj: int,
    *,
    force: bool = False,
    render_root: Path | None = None,
) -> list[Step]:
    """The steps that prepare one object's training data.

    Args:
        recipe: Dataset recipe.
        profile: The recipe's profile.
        root: Dataset root.
        obj: Object id.
        force: Rebuild a prepared object.
        render_root: Render campaign root of a generated object (``--generated-root``).
    """
    directory = object_dir(root, obj)
    geometry = directory / GEOMETRY_DIR
    keypoints = geometry / KEYPOINTS_NAME
    train_pbr = planned_source(recipe, profile, root, obj)
    convert = recipe.convert_for(obj)
    converted = training_converted(recipe, root, obj)
    state = _ObjectState()
    source, models = _source_step(
        recipe, profile, root, obj, render_root,
        lambda: _resolve_source(recipe, profile, root, obj, force, state, render_root),
    )  # fmt: skip
    split_rule: Any = convert.val_scenes
    if convert.split == "layout_modulo":
        spec = recipe.generated_spec(obj)
        split_rule = f"layout % {spec.val_modulo} == {spec.val_residue}"
    return [
        source,
        Step(
            f"keypoints obj {obj}",
            {
                "cad": models / f"obj_{obj:06d}.ply",
                "count": recipe.keypoints.count,
                "units": recipe.keypoints.source_units,
            },
            {"stl": geometry / STL_NAME, "keypoints": keypoints, "audit": geometry / "audit.json"},
            lambda: _keypoints(recipe, root, obj, state, render_root),
        ),
        Step(
            f"convert obj {obj}",
            {"train_pbr": train_pbr, "keypoints": keypoints, "empty masks": convert.empty_masks},
            {"converted": directory / CONVERTED_DIR, "training view": converted},
            lambda: _convert(recipe, root, obj, state),
        ),
        Step(
            f"projection audit obj {obj}",
            {"converted": converted, "train_pbr": train_pbr},
            {"report": directory / PROJECTION_NAME},
            lambda: _projection(recipe, root, obj, state),
        ),
        Step(
            f"split obj {obj}",
            {"converted": converted, "split": convert.split, "validation": split_rule},
            {"dataset": directory / DATASET_DIR},
            lambda: _split(recipe, root, obj, state),
        ),
        Step(
            f"receipt obj {obj}",
            {"object": directory},
            {"receipt": directory / PREPARED_NAME},
            lambda: _receipt(recipe, root, obj, state),
        ),
    ]


def plan(args: argparse.Namespace) -> list[Step]:
    """Build the ``prepare`` plan."""
    recipe, profile = load(args.recipe)
    refuse_flags(args, SPEEDPLUS_FLAGS, "SPEED+")
    refuse_flags(args, NUREC_FLAGS, "NuRec")
    if args.keypoints_only:
        return _keypoints_only_plan(recipe, profile, args)
    if args.multisensor:
        return _multisensor_plan(recipe, profile, args)
    root = Path(args.root)
    parts = list(args.parts or recipe.prepare.parts)
    render_root = args.generated_root
    # Onboarding views serve every object, generated ones too (their adaptation methods).
    objects = recipe_objects(recipe, args.obj_ids, generated=True)
    campaign = [o for o in objects if o in recipe.generated_objects] if args.obj_ids else []
    if render_root is not None and len(campaign) != 1:
        raise ValueError(
            "--generated-root prepares one generated object: name it with --obj-id "
            f"(generated objects: {sorted(recipe.generated)})"
        )
    if render_root is None and campaign and set(parts) >= TRAINING_PARTS:
        recipe_objects(recipe, campaign)  # raises: its training data needs --generated-root
    official = [o for o in objects if o in recipe.trainable_objects]
    # HOT3D fetches an object's reference tars when it stages that object.
    fetched = [p for p in parts if not (profile.name == "hot3d" and p in ONBOARDING_PARTS)]
    steps = []
    if fetched:
        steps.append(
            Step(
                "fetch and verify",
                {"repo": f"{profile.hf_repo}@{profile.revision}", "parts": fetched},
                {"files": sources.source_dir(root), "receipt": root / sources.READY_NAME},
                lambda: _fetch(profile, root, fetched, args.allow_test_inputs),
            )
        )
    hot3d_training = recipe.convert.source_view == "hot3d" and TRAINING_PARTS & set(parts)
    # HOT3D extracts its models and PBR in one step (hot3d_source.extract_hot3d_source reads
    # the archives itself and writes its own receipt), so no part goes through the generic
    # ZIP extraction: ``zipped`` is empty and ``_extract`` takes the HOT3D branch.
    zipped = [
        p for p in parts if zip_archives(profile, p) and p not in ONBOARDING_PARTS
    ] if not hot3d_training else []  # fmt: skip
    if hot3d_training or zipped:
        if hot3d_training and not set(parts) >= TRAINING_PARTS:
            raise ValueError("HOT3D extraction needs both the models and pbr parts")
        steps.append(
            Step(
                "extract",
                {"parts": sorted(TRAINING_PARTS) if hot3d_training else zipped},
                {"data": data_dir(root), "receipt": data_dir(root) / sources.READY_NAME},
                lambda: _extract(recipe, profile, root, zipped),
            )
        )
    if set(parts) >= TRAINING_PARTS:
        for obj in official:
            steps.extend(object_steps(recipe, profile, root, obj, force=args.force))
    for obj in campaign if render_root is not None else []:
        steps.extend(
            object_steps(recipe, profile, root, obj, force=args.force, render_root=render_root)
        )
    steps.extend(_hot3d_steps(recipe, profile, root, parts))
    steps.extend(_onboarding_steps(profile, root, parts, objects, args))
    return steps
