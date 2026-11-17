# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``doper train``: train one object's base model, or adapt it with one recipe method.

``--stage base``: the run directory is ``<runs>/base/obj_XXXXXX`` (``doper.bop.layout``): the
resolved training config goes to ``recipe.yaml``, the ordered keypoints to
``geometry/keypoints_3d.json`` and the checkpoints to ``work/``; ``completed.json`` records the
finished endpoint. An existing run is refused unless ``--force`` replaces it, so a finished base
is never overwritten by accident.

``--stage adapt --method NAME``: one adaptation method of the recipe
(``doper.training.adaptation.run_method``) in ``<runs>/<method>/obj_XXXXXX``: label pool,
audit, then a three-epoch student from the completed epoch-25 base, or
``adaptation_skipped.json`` below the method's minimum. A re-run verifies and continues;
``--labels-only`` stops after the verified pool; ``--force`` restarts the method's run. Methods
that read the unlabeled test inputs need ``--allow-test-inputs``.

``--stage followup --name NAME``: one follow-up series of the recipe
(``doper.training.followups.run_self_training``): rounds of the adaptation engine in
``<runs>/<name>/roundNN/obj_XXXXXX``, each from the previous round's student, until a round
accepts no labels, a strict-gate pool reaches the target or the round cap; then a labels-only
yield probe of the final student and ``<runs>/<name>/progress.json``. A review-gated round
trains only after ``doper review`` or with ``--accept-unreviewed`` (recorded). ``--force``
restarts the series.

``--stage multisensor`` (T-LESS): the object's three-sensor student
(``doper.training.adaptation.run_multisensor``) in ``<runs>/multisensor/obj_XXXXXX``, from the
completed epoch-25 base and the prepared PrimeSense, Kinect and Canon TRAIN roots
(``doper prepare tless --multisensor``). It trains only after ``doper review tless`` of the
root's nine overlays, or with ``--accept-unreviewed`` (recorded in ``completed.json``), and
refuses any training image that is a reserved ``test_primesense`` image. ``--force`` restarts it.

A NuRec recipe (``kind: nurec``) trains two stages with
``doper.recipes.nurec_schema.stage_arguments`` and no ``--obj-id``. ``--stage base`` trains the
synthetic baseline from ``<root>/synthetic/dataset`` into ``<runs>/base``; ``--stage adapt``
adapts the base run's ``work/best.pth`` on ``<root>/adaptation/dataset`` into ``<runs>/adapt``.
Before training, the dataset's prepare receipt must be next to it and current (the split step's
``synthetic/receipt.json``; the package step's ``adaptation/receipt.json`` with its QC passed),
the prepared split must hold the recipe's training image count and the dataset's keypoints the
recipe's layout; the expected optimizer updates
(``expected_updates``) are planned from that count. The full resolved arguments go to
``recipe.yaml``, the checkpoints to ``work/`` and ``completed.json`` records the counts, the
dataset receipt's SHA-256 and the checkpoint hashes. An existing run is refused unless
``--force`` replaces it.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from ..bop import layout, sources
from ..bop.sources import publish_json, sha256_file
from ..console import log
from ..keypoints import KEYPOINTS_NAME
from ..pseudo_labels._outdir import replace_output_dir
from ..recipes import BOP_KIND, load_any_recipe
from ..recipes.nurec_schema import (
    SYNTHETIC_SPLIT,
    NurecRecipe,
    Stage,
    expected_updates,
    stage_arguments,
)
from ..recipes.schema import PATH_KEYS, LaunchSpec, Recipe, recipe_objects, training_namespace
from .common import (
    COMPLETED_NAME,
    DATASET_DIR,
    NUREC_BEST,
    NUREC_WORK_DIR,
    RESOLVED_CONFIG_NAME,
    Step,
    add_recipe_argument,
    load,
    nurec_dataset_dir,
    nurec_run_dir,
    object_dir,
    prepared_receipt,
    read_json,
    route_kind,
    run_plan,
)
from .prepare import DATASET_QC_NAME, GEOMETRY_DIR, LABELS_RECEIPT_NAME, SYNTHETIC_RECEIPT_NAME

STAGES = ("base", "adapt", "followup", "multisensor")
NUREC_STAGES = ("base", "adapt")
NUREC_LAST = "last.pth"
# Flags of BOP recipes, refused for NuRec recipes: destination -> (flag, unset value).
BOP_TRAIN_FLAGS = {
    "obj_id": ("--obj-id", None),
    "override": ("--override", []),
    "method": ("--method", None),
    "labels_only": ("--labels-only", False),
    "name": ("--name", None),
    "accept_unreviewed": ("--accept-unreviewed", False),
    "consent": ("--allow-test-inputs", False),
}


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``doper train``."""
    parser = subparsers.add_parser(
        "train",
        help="train one object's model",
        description="Train one object's base model in-process from its prepared data, or run "
        "one adaptation method of the recipe from that base.",
    )
    add_recipe_argument(parser)
    parser.add_argument("--stage", choices=STAGES, required=True, help="training stage")
    parser.add_argument("--root", type=Path, required=True, help="prepared dataset root")
    parser.add_argument("--runs", type=Path, required=True, help="runs root")
    parser.add_argument(
        "--obj-id", type=int, default=None, dest="obj_id", help="object (BOP recipes; required)"
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="replace a recipe config value (YAML value; not paths or, for the base, launch "
        "keys); recorded in recipe.yaml",
    )
    parser.add_argument(
        "--method", default=None, help="adaptation method of the recipe (--stage adapt)"
    )
    parser.add_argument(
        "--labels-only",
        action="store_true",
        dest="labels_only",
        help="stop after the verified label pool (--stage adapt)",
    )
    parser.add_argument("--name", default=None, help="follow-up of the recipe (--stage followup)")
    parser.add_argument(
        "--accept-unreviewed",
        action="store_true",
        dest="accept_unreviewed",
        help="train a review-gated follow-up round or a multisensor student without a current "
        "human review; recorded in completed.json (and progress.json) (--stage followup or "
        "multisensor)",
    )
    parser.add_argument(
        "--allow-test-inputs",
        action="store_true",
        dest="consent",
        help="consent to transductive adaptation on the unlabeled test inputs (test methods)",
    )
    parser.add_argument("--force", action="store_true", help="replace an existing run")
    parser.set_defaults(run=run)


def parse_overrides(items: list[str]) -> dict[str, Any]:
    """Parse ``KEY=VALUE`` items, reading each value as YAML.

    Raises:
        ValueError: For an item without ``=``.
    """
    overrides = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise ValueError(f"--override expects KEY=VALUE, got {item!r}")
        overrides[key.replace("-", "_")] = yaml.safe_load(value)
    return overrides


def check_overrides(overrides: dict[str, Any], stage: str) -> None:
    """Refuse overrides that would bypass the run layout or the stage's launch contract.

    Dataset and run paths come from ``--root``, ``--runs`` and ``--obj-id``. The base stage
    trains from scratch as the recipe's ``launch`` section states (no ``init_from`` or
    ``resume``, the pretrained-backbone check); change those in a recipe copy instead. The
    adapt stage also keeps its launch, the base's architecture and the fixed budget
    (``doper.training.adaptation.check_adapt_overrides``).

    Raises:
        ValueError: For a path key, a launch key, or (adapt) an architecture or budget key.
    """
    if stage in ("adapt", "followup", "multisensor"):
        from ..training.adaptation import check_adapt_overrides

        check_adapt_overrides(overrides)
        return
    paths = sorted(set(overrides) & set(PATH_KEYS))
    if paths:
        raise ValueError(
            f"--override cannot set {paths}; they follow from --root, --runs and --obj-id"
        )
    launch = sorted(set(overrides) & {f.name for f in fields(LaunchSpec)})
    if stage == "base" and launch:
        raise ValueError(
            f"--override cannot set the launch keys {launch} for --stage base; edit them in a "
            "copy of the recipe"
        )


def _paths(root: Path, run_dir: Path, obj: int) -> dict[str, str]:
    prepared = object_dir(root, obj)
    return {
        "data_root": str(prepared / DATASET_DIR),
        "keypoints_3d": str(run_dir / layout.KEYPOINTS_PATH),
        "work_dir": str(run_dir / layout.CHECKPOINT_PATH.parent),
    }


def _train(
    recipe: Recipe, root: Path, run_dir: Path, obj: int, overrides: dict[str, Any], force: bool
) -> None:
    from .. import train

    receipt = prepared_receipt(root, obj)
    replace_output_dir(run_dir, force, [object_dir(root, obj)])
    resolved, _ = training_namespace(recipe.base, _paths(root, run_dir, obj), overrides)
    keypoints = run_dir / layout.KEYPOINTS_PATH
    keypoints.parent.mkdir(parents=True)
    shutil.copy2(object_dir(root, obj) / GEOMETRY_DIR / KEYPOINTS_NAME, keypoints)
    config = run_dir / RESOLVED_CONFIG_NAME
    config.write_text(yaml.safe_dump(resolved, sort_keys=False))
    train.main(["--config", str(config)])
    checkpoint = run_dir / layout.CHECKPOINT_PATH
    if not checkpoint.is_file():
        raise ValueError(f"training returned without {checkpoint}")
    publish_json(
        run_dir / COMPLETED_NAME,
        {
            "recipe": recipe.name,
            "stage": "base",
            "obj_id": obj,
            "epochs": resolved["epochs"],
            "checkpoint_sha256": sha256_file(checkpoint),
            "keypoints_sha256": sha256_file(keypoints),
            "prepared_hashes": receipt["hashes"],
        },
    )
    log(f"object {obj}: base training complete in {run_dir}")


def _adapt_plan(recipe: Recipe, args: argparse.Namespace, overrides: dict[str, Any]) -> Step:
    from ..training import adaptation

    if not args.method:
        methods = list(recipe.adapt.methods) if recipe.adapt is not None else []
        raise ValueError(f"--stage adapt needs --method; {recipe.name} methods: {methods}")
    spec = adaptation.method_spec(recipe, args.method)
    obj = args.obj_id
    if obj not in recipe.objects or not spec.applies_to(obj):
        raise ValueError(f"{args.method} does not run for object {obj}")
    if spec.transductive:
        sources.require_transductive_consent(args.consent)
    runs, root = Path(args.runs), Path(args.root)
    run_dir = layout.object_run(runs, args.method, obj)
    base = layout.object_run(runs, "base", obj)
    minimum = spec.minimum_for(obj)
    labels = run_dir / layout.LABELS_DIR
    if spec.source == "gt":
        source: Any = root / adaptation.ONBOARDING_DIR / adaptation.GT_VIEW
        labels = layout.gt_pool(runs, obj) / spec.supervision
    elif spec.source == "combined":
        source = ", ".join(str(layout.object_run(runs, c, obj)) for c in spec.components)
    else:
        source = root / adaptation.ONBOARDING_DIR / spec.source
    paths = {
        "data_root": object_dir(root, obj) / DATASET_DIR,
        "keypoints_3d": run_dir / layout.KEYPOINTS_PATH,
        "work_dir": run_dir / layout.CHECKPOINT_PATH.parent,
        "labels": labels,
        "teacher": base / layout.CHECKPOINT_PATH,
    }
    resolved, _ = adaptation.adapt_config(recipe, args.method, minimum, paths, overrides=overrides)

    def action() -> None:
        result = adaptation.run_method(
            recipe,
            args.method,
            obj,
            root,
            runs,
            predictor_factory=adaptation.load_predictor,
            consent=args.consent,
            labels_only=args.labels_only,
            force=args.force,
            overrides=overrides,
        )
        log(f"object {obj}: {args.method} {result.status} ({result.labels} labels)")

    return Step(
        f"adapt {args.method} obj {obj}",
        {
            "teacher": base / layout.CHECKPOINT_PATH,
            "source": source,
            "supervision": spec.supervision,
            "minimum labels": minimum,
            "backbone": resolved["backbone"],
            "epochs": resolved["epochs"],
            "overrides": overrides or "none",
            "stop": "after the verified labels" if args.labels_only else "completed student",
        },
        {
            "labels": labels,
            "receipt": run_dir / layout.LABELS_READY_NAME,
            "config": run_dir / RESOLVED_CONFIG_NAME,
            "checkpoint": run_dir / layout.CHECKPOINT_PATH,
            "completed": run_dir / COMPLETED_NAME,
        },
        action,
    )


def _followup_plan(recipe: Recipe, args: argparse.Namespace, overrides: dict[str, Any]) -> Step:
    from ..bop import review
    from ..training import adaptation, followups

    if args.method or args.labels_only or args.consent:
        raise ValueError("--method, --labels-only and --allow-test-inputs need --stage adapt")
    if not args.name:
        raise ValueError(
            f"--stage followup needs --name; {recipe.name} follow-ups: {list(recipe.followups)}"
        )
    spec = followups.followup_spec(recipe, args.name)
    obj = args.obj_id
    if obj != spec.obj_id:
        raise ValueError(f"follow-up {args.name} runs for object {spec.obj_id}, not {obj}")
    runs, root = Path(args.runs), Path(args.root)
    base = layout.object_run(runs, "base", obj)
    first = followups.round_dir(runs, args.name, 1, obj)
    teacher = base if spec.teacher == "base" else layout.object_run(runs, spec.teacher, obj)
    if spec.reuse_labels_from:
        seed: Any = f"{layout.object_run(runs, spec.reuse_labels_from, obj)} (reused pool)"
    else:
        seed = root / adaptation.ONBOARDING_DIR / spec.method().source
    paths = {
        "data_root": object_dir(root, obj) / DATASET_DIR,
        "keypoints_3d": first / layout.KEYPOINTS_PATH,
        "work_dir": first / layout.CHECKPOINT_PATH.parent,
        "labels": first / layout.LABELS_DIR,
        "teacher": teacher / layout.CHECKPOINT_PATH,
    }
    resolved, _ = adaptation.adapt_config(
        recipe, args.name, spec.minimum_labels, paths, overrides=overrides, spec=spec.method()
    )
    assert recipe.adapt is not None
    if spec.require_review:
        gate_note: Any = f"{first / review.REVIEWED_NAME} (doper review) before round 1 trains"
    else:
        gate_note = "none"
    rounds = {
        layout.round_name(n): followups.round_dir(runs, args.name, n, obj)
        for n in range(1, spec.rounds + 1)
    }

    def action() -> None:
        result = followups.run_self_training(
            recipe,
            args.name,
            root,
            runs,
            predictor_factory=adaptation.load_predictor,
            accept_unreviewed=args.accept_unreviewed,
            force=args.force,
            overrides=overrides,
        )
        log(f"{args.name}: {result.stop_reason}; final student {result.final_student}")

    return Step(
        f"follow-up {args.name} obj {obj}",
        {
            "round 1 teacher": teacher / layout.CHECKPOINT_PATH,
            "round 1 labels": seed,
            "round 1 gate": followups.round_gate(recipe, spec, 1).as_manifest(),
            "later rounds": "previous round's student; strict gate "
            f"{recipe.adapt.gate.as_manifest()}",
            "minimum labels": spec.minimum_labels,
            "rounds (cap)": spec.rounds,
            "target labels": spec.target_labels,
            "review": gate_note,
            "accept unreviewed": args.accept_unreviewed,
            "backbone": resolved["backbone"],
            "epochs": resolved["epochs"],
            "overrides": overrides or "none",
            "stop": "no labels, target reached (strict gate) or the round cap",
        },
        {
            **rounds,
            "final probe": followups.probe_dir(runs, args.name, obj),
            "progress": runs / args.name / layout.PROGRESS_NAME,
        },
        action,
    )


def _multisensor_plan(recipe: Recipe, args: argparse.Namespace, overrides: dict[str, Any]) -> Step:
    from ..bop import multisensor, review
    from ..bop.extract import data_dir
    from ..recipes.schema import MULTISENSOR_ARM
    from ..training import adaptation

    if args.method or args.labels_only or args.consent or args.name:
        raise ValueError(
            "--method, --labels-only and --allow-test-inputs need --stage adapt; --name needs "
            "--stage followup"
        )
    obj = args.obj_id
    if obj not in recipe.trainable_objects:
        raise ValueError(f"the multisensor stage runs for {recipe.name} objects, not {obj}")
    runs, root = Path(args.runs), Path(args.root)
    run_dir = layout.object_run(runs, MULTISENSOR_ARM, obj)
    base = layout.object_run(runs, "base", obj)
    out = multisensor.output_dir(root, obj)
    paths = adaptation.multisensor_paths(root, runs, obj)
    resolved, _ = adaptation.multisensor_config(recipe, paths, overrides=overrides)
    pbr = resolved["batch_size"] - sum(resolved["mix_per_batch"])

    def action() -> None:
        result = adaptation.run_multisensor(
            recipe,
            obj,
            root,
            runs,
            accept_unreviewed=args.accept_unreviewed,
            force=args.force,
            overrides=overrides,
        )
        log(f"object {obj}: multisensor {result.status} ({result.labels} real samples)")

    return Step(
        f"adapt multisensor obj {obj}",
        {
            "teacher": base / layout.CHECKPOINT_PATH,
            "sensors": out,
            "review": out / review.REVIEWED_NAME
            if not args.accept_unreviewed
            else "accepted unreviewed (recorded)",
            "reserved test": data_dir(root) / multisensor.TEST_SPLIT,
            "batch": f"{pbr} PBR + {resolved['mix_per_batch']} ({', '.join(multisensor.SENSORS)})",
            "epoch": "uncapped: max over the streams of ceil(samples / per batch)",
            "backbone": resolved["backbone"],
            "epochs": resolved["epochs"],
            "overrides": overrides or "none",
        },
        {
            "config": run_dir / RESOLVED_CONFIG_NAME,
            "checkpoint": run_dir / layout.CHECKPOINT_PATH,
            "completed": run_dir / COMPLETED_NAME,
        },
        action,
    )


def _nurec_paths(root: Path, runs: Path, stage: Stage) -> dict[str, Path]:
    dataset = nurec_dataset_dir(root, stage)
    paths = {
        "data_root": dataset,
        "keypoints_3d": dataset / KEYPOINTS_NAME,
        "work_dir": nurec_run_dir(runs, stage) / NUREC_WORK_DIR,
    }
    if stage == "adapt":
        paths["init_from"] = nurec_run_dir(runs, "base") / NUREC_WORK_DIR / NUREC_BEST
    return paths


# The prepare step whose receipt stands next to each stage's dataset, and that receipt's name.
NUREC_DATASET_STEPS = {
    "base": ("synthetic", "split", SYNTHETIC_RECEIPT_NAME),
    "adapt": ("labels", "package", LABELS_RECEIPT_NAME),
}


def _dataset_receipt(recipe: NurecRecipe, stage: Stage, dataset: Path) -> str:
    """Require the dataset's prepare receipt (and, to adapt, its passed QC); return its SHA-256.

    A dataset counts as prepared only when the step that wrote it finished: its receipt is
    written last, after the adaptation QC passed, and records the SHA-256 of the dataset's
    ``package.json``.

    Raises:
        FileNotFoundError: Without the receipt (or, to adapt, the QC summary).
        ValueError: For another step's receipt, a QC that did not pass or a dataset that
            changed since its receipt.
    """
    prepare_stage, step, name = NUREC_DATASET_STEPS[stage]
    path = dataset.parent / name
    rerun = f"rerun `doper prepare {recipe.name} --stage {prepare_stage}`"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: the {step} step of `doper prepare "
                                f"{recipe.name} --stage {prepare_stage}` did not finish (or its "
                                f"QC failed); {rerun} first")  # fmt: skip
    receipt = read_json(path)
    if receipt.get("step") != step:
        raise ValueError(f"{path} is the receipt of step {receipt.get('step')!r}, not {step}")
    package = dataset / "package.json"
    digest = sha256_file(package) if package.is_file() else None
    if digest is None or receipt.get("package_sha256") != digest:
        raise ValueError(f"{dataset} is not the dataset its receipt {path} records; {rerun} "
                         "--force")  # fmt: skip
    if stage == "adapt":
        qc = dataset.parent / DATASET_QC_NAME
        if receipt.get("qc") != "passed" or not qc.is_file():
            raise ValueError(f"{dataset} did not pass its QC ({qc}); {rerun} --force")
        if read_json(qc).get("status") != "passed" or sha256_file(qc) != receipt.get("qc_sha256"):
            raise ValueError(f"{qc} is not the passed QC its receipt {path} records; {rerun} "
                             "--force")  # fmt: skip
    return sha256_file(path)


def _train_count(dataset: Path) -> int:
    """Training images of a prepared COCO dataset (``train/annotations.json``)."""
    annotations = dataset / "train" / "annotations.json"
    if not annotations.is_file():
        raise FileNotFoundError(f"{annotations} is missing; prepare the dataset first")
    return len(read_json(annotations)["images"])


def _check_keypoints(recipe: NurecRecipe, path: Path) -> str:
    """Require the dataset's keypoints to be the recipe's layout; return the file SHA-256.

    Raises:
        ValueError: If the points hash differs.
    """
    from ..keypoints import load_layout

    load_layout(path, recipe.object.keypoints_points_sha256)
    return sha256_file(path)


def _base_best(runs: Path) -> str:
    """Return the SHA-256 of the completed base run's ``best.pth``.

    Raises:
        FileNotFoundError: If the base run did not complete.
        ValueError: If its ``best.pth`` changed since it completed.
    """
    base = nurec_run_dir(runs, "base")
    if not (base / COMPLETED_NAME).is_file():
        raise FileNotFoundError(f"{base / COMPLETED_NAME} is missing; train --stage base first")
    digest = sha256_file(base / NUREC_WORK_DIR / NUREC_BEST)
    if digest != read_json(base / COMPLETED_NAME)["best_sha256"]:
        raise ValueError(f"{base / NUREC_WORK_DIR / NUREC_BEST} changed since the base completed")
    return digest


def _train_nurec(
    recipe: NurecRecipe,
    stage: Stage,
    arguments: dict[str, Any],
    paths: dict[str, Path],
    planned: int,
    force: bool,
) -> None:
    from .. import train

    dataset, run_dir = paths["data_root"], paths["work_dir"].parent
    receipt_sha256 = _dataset_receipt(recipe, stage, dataset)
    count = _train_count(dataset)
    if count != planned:
        raise ValueError(
            f"{dataset} holds {count} training images; the {stage} stage plans {planned}"
        )
    updates = expected_updates(
        count,
        arguments["batch_size"],
        arguments["gradient_accumulation_steps"],
        arguments["epochs"],
    )
    keypoints_sha256 = _check_keypoints(recipe, paths["keypoints_3d"])
    teacher = _base_best(paths["work_dir"].parents[1]) if stage == "adapt" else None
    replace_output_dir(run_dir, force, [dataset, paths.get("init_from")])
    config = run_dir / RESOLVED_CONFIG_NAME
    config.parent.mkdir(parents=True)
    config.write_text(yaml.safe_dump(arguments, sort_keys=False))
    train.main(["--config", str(config)])
    work = paths["work_dir"]
    for name in (NUREC_BEST, NUREC_LAST):
        if not (work / name).is_file():
            raise ValueError(f"training returned without {work / name}")
    publish_json(
        run_dir / COMPLETED_NAME,
        {
            "recipe": recipe.name,
            "stage": stage,
            "epochs": arguments["epochs"],
            "train_images": count,
            "expected_updates": updates,
            "dataset_receipt_sha256": receipt_sha256,
            "keypoints_sha256": keypoints_sha256,
            "keypoints_points_sha256": recipe.object.keypoints_points_sha256,
            "init_from_sha256": teacher,
            "best_sha256": sha256_file(work / NUREC_BEST),
            "last_sha256": sha256_file(work / NUREC_LAST),
        },
    )
    log(f"{recipe.name}: {stage} training complete in {run_dir} ({updates} updates)")


def nurec_plan(args: argparse.Namespace) -> list[Step]:
    """Build the ``train`` plan of a NuRec recipe: ``--stage base`` or ``adapt``.

    Raises:
        ValueError: For a BOP flag or another stage.
    """
    recipe = load_any_recipe(args.recipe)
    if not isinstance(recipe, NurecRecipe):
        raise ValueError(f"{recipe.name} is not a NuRec recipe")
    used = [flag for dest, (flag, unset) in BOP_TRAIN_FLAGS.items() if getattr(args, dest) != unset]
    if used:
        raise ValueError(f"{used} are BOP flags; `doper train {recipe.name}` does not take them")
    if args.stage not in NUREC_STAGES:
        raise ValueError(
            f"`doper train {recipe.name}` takes --stage base or adapt, not {args.stage}"
        )
    stage: Stage = args.stage
    paths = _nurec_paths(Path(args.root), Path(args.runs), stage)
    arguments = stage_arguments(recipe, stage, {key: str(value) for key, value in paths.items()})
    planned = SYNTHETIC_SPLIT[0] if stage == "base" else recipe.split.train
    updates = expected_updates(
        planned,
        arguments["batch_size"],
        arguments["gradient_accumulation_steps"],
        arguments["epochs"],
    )
    run_dir = paths["work_dir"].parent
    inputs: dict[str, Any] = {"dataset": paths["data_root"], "keypoints": paths["keypoints_3d"]}
    if stage == "adapt":
        inputs["init from"] = paths["init_from"]
    inputs.update(
        {
            "backbone": arguments["backbone"],
            "epochs": arguments["epochs"],
            "batch": f"{arguments['batch_size']} x {arguments['gradient_accumulation_steps']} "
            "accumulation",
            "lr": arguments["lr"],
            "train images (planned)": planned,
            "expected updates": updates,
        }
    )
    return [
        Step(
            f"train {recipe.name} {stage}",
            inputs,
            {
                "config": run_dir / RESOLVED_CONFIG_NAME,
                "best": paths["work_dir"] / NUREC_BEST,
                "last": paths["work_dir"] / NUREC_LAST,
                "receipt": run_dir / COMPLETED_NAME,
            },
            lambda: _train_nurec(recipe, stage, arguments, paths, planned, args.force),
        )
    ]


def plan(args: argparse.Namespace) -> list[Step]:
    """Build the ``train`` plan of a BOP recipe."""
    recipe, _ = load(args.recipe)
    if args.obj_id is None:
        raise ValueError(f"`doper train {recipe.name}` needs --obj-id")
    overrides = parse_overrides(args.override)
    check_overrides(overrides, args.stage)
    if args.stage == "followup":
        return [_followup_plan(recipe, args, overrides)]
    if args.stage == "multisensor":
        return [_multisensor_plan(recipe, args, overrides)]
    if args.name or args.accept_unreviewed:
        raise ValueError(
            "--name needs --stage followup; --accept-unreviewed needs --stage followup or "
            "multisensor"
        )
    if args.stage == "adapt":
        return [_adapt_plan(recipe, args, overrides)]
    if args.method or args.labels_only or args.consent:
        raise ValueError("--method, --labels-only and --allow-test-inputs need --stage adapt")
    # A generated object trains from the data `prepare --generated-root` made, like any other.
    (obj,) = recipe_objects(recipe, [args.obj_id], generated=True)
    run_dir = layout.object_run(args.runs, args.stage, obj)
    resolved, _ = training_namespace(recipe.base, _paths(args.root, run_dir, obj), overrides)
    return [
        Step(
            f"train {args.stage} obj {obj}",
            {
                "prepared": object_dir(args.root, obj),
                "backbone": resolved["backbone"],
                "epochs": resolved["epochs"],
                "overrides": overrides or "none",
            },
            {
                "config": run_dir / RESOLVED_CONFIG_NAME,
                "checkpoint": run_dir / layout.CHECKPOINT_PATH,
                "keypoints": run_dir / layout.KEYPOINTS_PATH,
                "receipt": run_dir / COMPLETED_NAME,
            },
            lambda: _train(recipe, Path(args.root), run_dir, obj, overrides, args.force),
        )
    ]


def run(args: argparse.Namespace) -> int:
    """Run ``doper train`` (BOP and NuRec recipes)."""
    planners = {BOP_KIND: plan, NurecRecipe.kind: nurec_plan}
    return run_plan(route_kind(args, planners, "train"), dry_run=args.dry_run)
