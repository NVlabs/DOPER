# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``doper evaluate``: validation AR of each object model, or the BOP test export of a bank.

``--split val`` scores every trained object model on the recipe's validation population with the
single-pass, box-jitter and GT-box arms (``doper.bop.ar.score_model`` with ``JitterSpec()`` and
``gt_box=True``) and writes ``<runs>/<arm>/obj_XXXXXX/validation/ar.json``. A re-run keeps each
``ar.json`` that scored the same checkpoint, geometry and protocol, so it scores only the missing
objects; one that scored anything else is refused unless ``--force`` rescores it.

``--split test`` builds a bank and exports it to a BOP CSV (``doper.bop.export.run_bank``)
against the profile's frozen test population. ``--policy base`` uses each object's base model
(untimed with fixed policies: the cached fixed base bank below);
``--policy selected`` ranks every existing arm's endpoint by validation single-pass AR
(``doper.bop.selection.choose``; HOT3D requires native scores and one shared validation
population). A bank covers every object: one with official PBR, or a generated one
(``generated_objects``) once it is prepared from its render root (``doper prepare
--generated-root``). An object without a prepared source or a trained model makes the bank
partial, which is refused unless ``--allow-partial-bank``. An interrupted
export resumes from its journal; ``--force`` starts over (CSV, journal, selection and summary).
``selection.json`` and ``summary.json`` are written only after the CSV.

``--policy <name>`` exports one of the recipe's fixed policies (``test.policies``,
``doper.bop.banks``): one method's final endpoint per object, the base only where the policy
allows a disclosed fallback, written with ``MODEL_MATRIX.csv`` and ``SKIPS.json``. For a recipe
with fixed policies, an untimed ``--policy base`` is the fixed base policy too (every object's
base, same CSV name, a manifest instead of ``selection.json``). Untimed fixed banks scan model
by model through the per-model journal cache ``<runs>/test/cache``, so a model shared by
several banks (a base fallback, the base policy) is scanned once; timed banks
(``test.record_time``, HOT3D, a policy's own ``record_time``, T-LESS ``multisensor``, or
``--record-time``) rescan every model with all models
resident, and a timed base keeps the validation-selection path above.

``--variant`` exports one of the four test variants of the same bank instead
(``doper.bop.variants``): ``singleview``, ``singleheatmap``, ``multiview`` or ``multiheatmap``,
an untimed post-pass over each model's cached raw detection journal (``<runs>/test/cache``, the
same forward pass as the single-view export; the heatmap variants also fill the model's
separate heatmap store once, from that pass or a rescan that must reproduce it). It writes
``<runs>/test/<policy>/<variant>/<csv method><variant>_<dataset>-test.csv`` with the bank
manifests, ``summary.json`` and ``variants_manifest.json`` (optimizer, preparation, library
versions and the support-group hash). ``--optimizer`` (``lbfgsb`` or ``powell``) and
``--preparation`` (``numpy`` or ``parallel``) override the recipe's ``refine`` defaults. The
multi-view variants need the recipe's ``multiview`` policy (HANDAL has none) and, for HOT3D,
the support streams that ``doper prepare --parts test`` stages. A heatmap variant whose
optimizer or preparation needs numba is refused at plan time without ``doper[refine]``.

A SHIRT recipe (``kind: shirt``) takes ``--method`` instead of ``--split``. ``--method
framewise`` (``doper.shirt.candidates.framewise``) evaluates ``--checkpoint`` with its
``--keypoints-3d`` layout on the SHIRT tree ``--root``, frame by frame, and writes
``<runs>/framewise/<roe>/<domain>/{records.jsonl,summary.json}`` and
``<runs>/framewise/summary.json``. The trajectories and domain default to the recipe's
(``--trajectories``, ``--domain``). A checkpoint other than the recipe's pinned one needs
``--allow-unpinned-checkpoint`` (recorded in the summary); one not trained on the layout is
refused. ``--method candidates`` (``doper.shirt.candidates.crop_candidates``) reads those
framewise records and writes the 0/180 re-crop pose candidates of each frame to
``<runs>/candidates/<roe>/<domain>/{records.jsonl,summary.json}`` and
``<runs>/candidates/summary.json`` with the same checkpoint, layout and pin rules. The Table 2
filters (``doper.shirt.policies``) read those candidates and each trajectory's SHIRT
``metadata.json``: ``--method pnp-ukf`` (minimum-NIS candidates in a chi-square p99 gate),
``--method pnp-ukf-rts`` (plus one offline RTS pass) and ``--method keypoint-ukf`` (gated heatmap
landmarks; it first caches each frame's landmarks and covariances from ``--checkpoint`` with
``--keypoints-3d`` to ``<runs>/keypoint-ukf/<roe>/<domain>/heatmap_measurements.npz``, reusing a
cache of the same candidate frames, checkpoint and layout, and refusing any other cache without
``--force``). Each writes ``<runs>/<method>/<roe>/<domain>/{records.jsonl,summary.json}`` and
``<runs>/<method>/summary.json``. The PnP filters run no model and refuse ``--checkpoint`` and
``--keypoints-3d``; every filter needs ``doper[filter]`` (bayesfilter), checked at plan time, as
are the trajectories, the domain and ``--limit``. The flags of one recipe kind are refused for
the other.

A NuRec recipe (``kind: nurec``) takes ``--bundle``, ``--checkpoint`` and ``--out``
and evaluates relative camera poses (``doper.evaluation.relative_pose``): the checkpoint's
single-view pose of every capture of the bundle (top-1 detection with the recipe's
``evaluation.score_thr``, no TTA, the bundle's rectified K) against the robot reference
(``robot.jsonl`` and ``handeye.json``, required), with the bundle's ``keypoints_3d.json``, which
must be the recipe's layout. It writes ``<out>/{frames,pairs,summary}.json``; the summary
carries the capture flags (``handeye_camera_mismatch`` with both K), the caveats, the
checkpoint and keypoint SHA-256s and the excluded captures (the bundle's ``exclude.json`` and the
recipe's ``evaluation.exclude``). An existing evaluation in ``--out`` is replaced only with
``--force``. ``--root`` and ``--runs`` are required by every ``evaluate`` but not read.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import shutil
from pathlib import Path
from typing import Any

from ..bop import layout
from ..bop.extract import data_dir
from ..bop.layout import HEATMAP_VARIANTS, MULTIVIEW_VARIANTS, VARIANTS
from ..bop.sources import DatasetProfile, publish_json, sha256_file
from ..console import log
from ..recipes import BOP_KIND, load_any_recipe
from ..recipes.schema import (
    HOT3D_INDEX,
    HOT3D_PANEL,
    TEST_CACHE_DIR,
    Recipe,
    RefineSpec,
    followup_runs,
    numba_installed,
    recipe_arms,
    recipe_objects,
)
from ..recipes.shirt_schema import ShirtRecipe
from .common import (
    COMPLETED_NAME,
    MissingExtraError,
    Step,
    add_recipe_argument,
    add_refine_arguments,
    bank_objects,
    load,
    read_json,
    route_kind,
    run_plan,
)

SEED = 42
POLICIES = ("base", "selected")
TEST_DIR = layout.TEST_DIR
MISSING_REASON = "missing object: no candidate endpoint"
SELECTION_NAME = "selection.json"
SUMMARY_NAME = "summary.json"
CACHE_DIR = TEST_CACHE_DIR
SHIRT_METHODS = ("framewise", "candidates", "pnp-ukf", "pnp-ukf-rts", "keypoint-ukf")
SHIRT_FILTERS = ("pnp-ukf", "pnp-ukf-rts", "keypoint-ukf")
# SHIRT methods that run the model, so need --checkpoint and --keypoints-3d.
SHIRT_MODEL_METHODS = ("framewise", "candidates", "keypoint-ukf")
FRAMEWISE_DIR = "framewise"
CANDIDATES_DIR = "candidates"
# Flags of one recipe kind, refused for the other, with the default each has when not given.
BOP_FLAGS: dict[str, tuple[str, Any]] = {
    "split": ("--split", None),
    "policy": ("--policy", "selected"),
    "obj_ids": ("--obj-id", None),
    "arm": ("--arm", "base"),
    "allow_partial_bank": ("--allow-partial-bank", False),
    "record_time": ("--record-time", False),
    "variant": ("--variant", None),
    "optimizer": ("--optimizer", None),
    "preparation": ("--preparation", None),
}
SHIRT_FLAGS: dict[str, tuple[str, Any]] = {
    "method": ("--method", None),
    "checkpoint": ("--checkpoint", None),
    "keypoints_3d": ("--keypoints-3d", None),
    "trajectories": ("--trajectories", None),
    "domain": ("--domain", None),
    "allow_unpinned_checkpoint": ("--allow-unpinned-checkpoint", False),
    "limit": ("--limit", 0),
}
NUREC_FLAGS: dict[str, tuple[str, Any]] = {"bundle": ("--bundle", None), "out": ("--out", None)}


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``doper evaluate``."""
    parser = subparsers.add_parser(
        "evaluate",
        help="score validation AR or export the BOP test CSV",
        description="Score validation AR per object model (--split val) or export a model "
        "bank's BOP test CSV (--split test).",
    )
    add_recipe_argument(parser)
    parser.add_argument(
        "--split", choices=("val", "test"), default=None, help="BOP recipes: val or test"
    )
    parser.add_argument(
        "--root", type=Path, required=True, help="prepared dataset root (SHIRT: the SHIRT tree)"
    )
    parser.add_argument("--runs", type=Path, required=True, help="runs root")
    parser.add_argument(
        "--policy",
        default="selected",
        help="test bank policy: base, selected or one of the recipe's fixed policies",
    )
    parser.add_argument(
        "--obj-id", type=int, nargs="+", dest="obj_ids", help="validation objects (default: all)"
    )
    parser.add_argument(
        "--arm",
        default="base",
        help="validation arm: base, a recipe adaptation method, a selection alias "
        "(test.aliases) or a follow-up round (<followup>/roundNN)",
    )
    parser.add_argument(
        "--allow-partial-bank",
        action="store_true",
        help="export a test bank that lacks some objects (a generated object whose render "
        "root is not prepared, or an object without a trained model)",
    )
    parser.add_argument(
        "--record-time", action="store_true", help="write measured per-image seconds to the CSV"
    )
    parser.add_argument(
        "--variant",
        choices=VARIANTS,
        default=None,
        help="export one test variant (untimed post-pass): singleview, singleheatmap, multiview "
        "or multiheatmap",
    )
    add_refine_arguments(parser, suffix=" of --variant")
    parser.add_argument("--device", default=None, help="torch device (default: cuda if present)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="rescore existing validation results; restart a test export (its CSV, journal, "
        "selection and summary); overwrite SHIRT outputs",
    )
    _add_shirt_arguments(parser)
    nurec = parser.add_argument_group("NuRec recipes")
    nurec.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="capture bundle with a robot reference and keypoints_3d.json (also takes "
        "--checkpoint)",
    )
    nurec.add_argument("--out", type=Path, default=None, help="evaluation output directory")
    parser.set_defaults(run=run)


def _add_shirt_arguments(parser: argparse.ArgumentParser) -> None:
    from ..shirt import DOMAINS

    shirt = parser.add_argument_group("SHIRT recipes")
    shirt.add_argument("--method", choices=SHIRT_METHODS, default=None, help="SHIRT method")
    shirt.add_argument(
        "--checkpoint", type=Path, default=None, help="SPEED+ checkpoint (NuRec: DOPER checkpoint)"
    )
    shirt.add_argument(
        "--keypoints-3d", type=Path, default=None, help="the checkpoint's 75-point layout file"
    )
    shirt.add_argument(
        "--trajectories", default=None, help="comma-separated trajectories (default: recipe)"
    )
    shirt.add_argument("--domain", choices=DOMAINS, default=None, help="default: recipe")
    shirt.add_argument(
        "--allow-unpinned-checkpoint",
        action="store_true",
        help="evaluate a checkpoint other than the recipe's pinned one (recorded)",
    )
    shirt.add_argument(
        "--limit", type=int, default=0, help="first N frames of each trajectory (0: all)"
    )


def _refuse_flags(args: argparse.Namespace, flags: dict[str, tuple[str, Any]], kind: str) -> None:
    given = [flag for name, (flag, default) in flags.items() if getattr(args, name) != default]
    if given:
        raise ValueError(f"`doper evaluate` takes {given} only for {kind} recipes")


def _device(requested: str | None) -> str:
    import torch

    return requested or ("cuda:0" if torch.cuda.is_available() else "cpu")


def validation_population(recipe: Recipe, root: Path) -> Path | None:
    """The validation population of a recipe, or None when it has none."""
    population = recipe.validation.population
    if population == "none":
        return None
    if population == HOT3D_PANEL:
        from ..bop.hot3d_staging import VALIDATION_MANIFEST

        return Path(root) / VALIDATION_MANIFEST
    return data_dir(root) / population


def held_out_population(recipe: Recipe, root: Path) -> Path:
    """The test population of a recipe: a BOP split directory or the HOT3D test manifest."""
    if recipe.test.population == HOT3D_INDEX:
        from ..bop.hot3d_staging import TEST_DIR as HOT3D_TEST_DIR
        from ..bop.hot3d_staging import TEST_MANIFEST

        return Path(root) / HOT3D_TEST_DIR / TEST_MANIFEST
    return data_dir(root) / recipe.test.population


def _score(recipe: Recipe, root: Path, run_dir: Path, obj: int, device: str, force: bool) -> None:
    import cv2
    import torch

    from ..bop import ar
    from ..bop.export import keypoints_mm
    from ..config import DataConfig, DoperConfig, ModelConfig
    from ..infer import load_model

    out = run_dir / layout.VALIDATION_PATH
    checkpoint, keypoints = run_dir / layout.CHECKPOINT_PATH, run_dir / layout.KEYPOINTS_PATH
    current = {
        "obj_id": obj,
        "protocol": ar.PROTOCOL_IDS.get((recipe.validation.protocol, recipe.validation.camera)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "geometry_sha256": sha256_file(keypoints),
    }
    if out.exists() and not force:
        scored = read_json(out)
        if all(scored.get(key) == value for key, value in current.items()):
            log(f"object {obj}: already scored for this checkpoint and protocol ({out})")
            return
        raise SystemExit(
            f"{out} scores another checkpoint, geometry or protocol; pass --force to rescore it"
        )
    population = validation_population(recipe, root)
    assert population is not None
    cfg = DoperConfig(data=DataConfig(root="."), model=ModelConfig(pretrained=False))
    torch.manual_seed(SEED)
    cv2.setRNGSeed(SEED)
    model = load_model(checkpoint, cfg, device).eval()
    result = ar.score_model(
        model,
        population,
        obj_id=obj,
        keypoints_3d_mm=keypoints_mm(keypoints),
        models_dir=data_dir(root) / recipe.validation.models,
        protocol=recipe.validation.protocol,
        camera=recipe.validation.camera,
        jitter=ar.JitterSpec(),
        gt_box=True,
        cfg=cfg,
        device=device,
        checkpoint_sha256=current["checkpoint_sha256"],
        geometry_sha256=current["geometry_sha256"],
    )
    publish_json(out, result.to_dict())
    single = result.arms.get("single")
    score = f"{single.ar_mean:.2f}" if single else result.status
    log(f"object {obj}: validation single-pass AR {score} ({result.frames} frames)")


def validation_run(recipe: Recipe, arm: str) -> tuple[str, tuple[int, ...] | None]:
    """The run directory and objects a validation ``--arm`` scores.

    Args:
        recipe: Dataset recipe.
        arm: ``base``, a recipe method, a selection alias (``test.aliases``, scored in its run
            directory) or a follow-up round (``<followup>/roundNN``).

    Returns:
        ``(run, objects)``: the run name under ``<runs>`` and the objects it applies to (None:
        every recipe object).

    Raises:
        ValueError: If ``arm`` is none of these.
    """
    if arm in recipe_arms(recipe):
        return arm, None
    alias = recipe.test.aliases.get(arm)
    if alias is not None:
        return alias.run, alias.objects
    followup = followup_runs(recipe).get(arm)
    if followup is not None:
        return arm, (followup.obj_id,)
    names = [*recipe_arms(recipe), *recipe.test.aliases, *followup_runs(recipe)]
    raise ValueError(f"--arm {arm!r} is not a {recipe.name} arm, alias or follow-up: {names}")


def _validation_objects(recipe: Recipe, args: argparse.Namespace) -> tuple[str, list[int]]:
    run, only = validation_run(recipe, args.arm)
    objects = list(recipe_objects(recipe, args.obj_ids, generated=True))
    if only is None:
        return run, objects
    outside = sorted(set(args.obj_ids or ()) - set(only))
    if outside:
        raise ValueError(f"--arm {args.arm} applies to objects {list(only)}, not {outside}")
    return run, [o for o in objects if o in only]


def _validation_steps(recipe: Recipe, args: argparse.Namespace) -> list[Step]:
    population = validation_population(recipe, args.root)
    if population is None:
        return [Step(f"{recipe.name} has no validation split; the test bank keeps the base models")]
    run, objects = _validation_objects(recipe, args)
    steps = []
    for obj in objects:
        run_dir = layout.object_run(args.runs, run, obj)

        def action(run_dir: Path = run_dir, obj: int = obj) -> None:
            if not (run_dir / COMPLETED_NAME).is_file():
                if args.obj_ids:
                    raise FileNotFoundError(f"{run_dir} has no completed {args.arm} model")
                log(f"object {obj}: no completed {args.arm} model in {run_dir}; not scored")
                return
            _score(recipe, args.root, run_dir, obj, _device(args.device), args.force)

        steps.append(
            Step(
                f"validate {args.arm} obj {obj}",
                {
                    "model": run_dir / layout.CHECKPOINT_PATH,
                    "population": population,
                    "protocol": f"{recipe.validation.protocol}/{recipe.validation.camera}",
                    "arms": "single, jitter, gtbox",
                },
                {"result": run_dir / layout.VALIDATION_PATH},
                action,
            )
        )
    return steps


def _candidate(endpoint: Any, expected: tuple[str | None, str | None] | None) -> Any:  # noqa: ANN401
    """One endpoint (``doper.bop.selection.Endpoint``) as a selection candidate.

    ``expected`` is the (protocol id, validation population SHA-256) the recipe requires of a
    score; selection checks the stored score against it. None reads no score (base policy).
    """
    from ..bop.ar import ArResult
    from ..bop.selection import Candidate

    run_dir = endpoint.run_dir
    result_path = run_dir / layout.VALIDATION_PATH
    result = None
    if expected is not None and result_path.is_file():
        result = ArResult.from_dict(read_json(result_path))
    protocol, population = expected if expected is not None else (None, None)
    checkpoint, keypoints = run_dir / layout.CHECKPOINT_PATH, run_dir / layout.KEYPOINTS_PATH
    return Candidate(
        obj_id=endpoint.obj_id,
        arm=endpoint.name,
        endpoint_complete=(run_dir / COMPLETED_NAME).is_file(),
        checkpoint_sha256=sha256_file(checkpoint) if checkpoint.is_file() else "",
        geometry_sha256=sha256_file(keypoints) if keypoints.is_file() else "",
        validation_population_sha256=population,
        protocol=protocol,
        result=result,
        run_dir=run_dir,
    )


def expected_validation(
    recipe: Recipe, root: Path, *, scored: bool
) -> tuple[str | None, str | None]:
    """The validation protocol id and population SHA-256 a selection score must carry.

    Args:
        recipe: Dataset recipe.
        root: Prepared dataset root.
        scored: Whether any candidate has a stored score; without one the population is not
            hashed (building a directory population hashes every image).

    Returns:
        ``(protocol, population_sha256)``; the population is None when ``scored`` is false.

    Raises:
        FileNotFoundError: If scores exist but the validation population is missing.
    """
    from ..bop import ar
    from ..bop.population import load_population

    protocol = ar.PROTOCOL_IDS.get((recipe.validation.protocol, recipe.validation.camera))
    path = validation_population(recipe, root)
    if not scored or path is None:
        return protocol, None
    if not path.exists():
        raise FileNotFoundError(
            f"validation population {path} is missing; it is needed to verify the stored scores"
        )
    return protocol, load_population(path)[1]


def select_bank(
    recipe: Recipe, root: Path, runs: Path, policy: str, allow_partial_bank: bool
) -> tuple[dict[int, Path], dict[int, Any]]:
    """Choose one endpoint per bank object (``common.bank_objects``) and resolve the bank.

    With ``policy="selected"`` a stored validation score counts only when its protocol and
    population are the recipe's (``expected_validation``), its checkpoint and geometry the
    candidate's.

    Args:
        recipe: Dataset recipe.
        root: Prepared dataset root (the validation population).
        runs: Runs root (``<runs>/<arm>/obj_XXXXXX``).
        policy: ``base`` (every object's base) or ``selected`` (validation-selected).
        allow_partial_bank: Leave out objects that have no trained endpoint at all.

    Returns:
        The bank (object id to run directory) and the per-object selections.

    Raises:
        ValueError: If an object has no endpoint and partial banks are not allowed, or any
            other selection error (unfinished endpoint, lineage or hash mismatch).
    """
    from ..bop import ar, selection

    objects = bank_objects(recipe, root)
    # The recipe's arms and aliases; a method skipped below its label minimum is not a
    # candidate (skipped by design).
    arms = None if policy == "selected" else ("base",)
    endpoints = selection.recipe_endpoints(recipe, runs, objects, arms=arms)
    expected = None
    if policy == "selected":
        scored = any((e.run_dir / layout.VALIDATION_PATH).is_file() for e in endpoints)
        expected = expected_validation(recipe, root, scored=scored)
    candidates = [_candidate(endpoint, expected) for endpoint in endpoints]
    native = policy == "selected" and recipe.validation.camera == "native"
    choices = selection.choose(
        candidates,
        objects,
        protocol=ar.PROTOCOL_HOT3D_NATIVE if native else None,
        require_scored=native,
    )
    missing = sorted(o for o, s in choices.items() if s.reason == MISSING_REASON)
    if missing and not allow_partial_bank:
        raise ValueError(
            f"objects {missing} have no trained model; pass --allow-partial-bank to export a "
            "partial bank"
        )
    present = {o: s for o, s in choices.items() if o not in missing}
    if policy == "selected":
        _warn_unscored(present)
    if native:
        selection.require_shared_population(present)
    return selection.build_bank(present, runs), choices


def _warn_unscored(choices: dict[int, Any]) -> None:
    """Report candidates that exist but carry no validation score (they are not ranked)."""
    for obj, choice in sorted(choices.items()):
        if choice.status != "error" and choice.unscored:
            log(
                f"warning: object {obj}: candidates {list(choice.unscored)} have no validation "
                "score and are not ranked; score them with `doper evaluate --split val --arm NAME`"
            )


def _selection_record(choices: dict[int, Any]) -> dict[str, Any]:
    return {
        str(obj): {
            "status": s.status,
            "reason": s.reason,
            "arm": None if s.chosen is None else s.chosen.arm,
            "single_ar": None if s.chosen is None else s.chosen.single_ar,
            "unscored": list(s.unscored),
        }
        for obj, s in sorted(choices.items())
    }


def fixed_policy(recipe: Recipe, policy: str, record_time: bool) -> Any:  # noqa: ANN401
    """The fixed policy (``doper.bop.banks.Policy``) an export runs, or None for selection.

    A recipe policy always is one. ``base`` is the fixed ``base_policy`` (cached, with its
    manifest) when the recipe has fixed policies and the export is untimed; a timed base
    (HOT3D, ``--record-time``) and ``selected`` keep the validation-selection path.

    Raises:
        ValueError: For an unknown policy name.
    """
    from ..bop import banks

    if policy == "selected":
        return None
    if policy == "base":
        timed = record_time or recipe.test.record_time
        return banks.base_policy(recipe) if recipe.test.policies and not timed else None
    return banks.recipe_policy(recipe, policy)


def _fixed_objects(
    recipe: Recipe, root: Path, runs: Path, allow_partial_bank: bool
) -> tuple[int, ...]:
    """Objects of a fixed bank: every bank object, or those with a base when partial."""
    objects = bank_objects(recipe, root)
    if not allow_partial_bank:
        return objects
    return tuple(
        o for o in objects if (layout.object_run(runs, "base", o) / COMPLETED_NAME).is_file()
    )


def _outputs(out: Path) -> list[Path]:
    from ..bop import banks

    names = (SELECTION_NAME, SUMMARY_NAME, banks.MATRIX_NAME, banks.SKIPS_NAME)
    return [out, *(out.parent / name for name in names)]


@dataclasses.dataclass(frozen=True)
class _Bank:
    """A resolved test bank: the fixed policy (or None), the bank and its record."""

    policy: Any
    bank: dict[int, Path]
    fallbacks: Any = None
    choices: Any = None
    missing: list[int] = dataclasses.field(default_factory=list)


def _resolve_bank(
    recipe: Recipe, args: argparse.Namespace, policy_name: str, *, record_time: bool
) -> _Bank:
    """The bank of ``policy_name``: a fixed policy's or the validation selection; refuses a gap."""
    from ..bop import banks

    policy = fixed_policy(recipe, policy_name, record_time)
    fallbacks = choices = None
    if policy is not None:
        objects = _fixed_objects(recipe, args.root, args.runs, args.allow_partial_bank)
        bank, fallbacks = banks.fixed_bank(policy, args.runs, objects)
    else:
        bank, choices = select_bank(
            recipe, args.root, args.runs, policy_name, args.allow_partial_bank
        )
    missing = sorted(set(recipe.objects) - set(bank))
    if missing and not args.allow_partial_bank:
        unprepared = sorted(set(missing) & set(recipe.generated_objects))
        hint = (
            f" (generated objects {unprepared} have no prepared render root: run `doper "
            "prepare --obj-id OBJ --generated-root DIR`, then train them)"
            if unprepared
            else ""
        )
        raise ValueError(
            f"objects {missing} have no model{hint}; pass --allow-partial-bank to export a "
            "partial bank"
        )
    if missing:
        log(f"partial bank: objects {missing} have no model")
    return _Bank(policy, bank, fallbacks, choices, missing)


def _start_over(out: Path, force: bool, extra: list[Path]) -> None:
    """Refuse an existing CSV, or (``--force``) remove it and everything published beside it.

    ``extra`` lists further files or directories a forced export removes.
    """
    from ..bop import export

    if out.exists() and not force:
        raise SystemExit(f"{out} exists; pass --force to replace it")
    if force:
        # A forced export starts over: an old journal would refuse a retrained bank or other
        # settings, and an old selection, summary or manifest must not sit beside a failed
        # export. The per-model cache is keyed by the model hashes and stays.
        for path in [*_outputs(out), *extra]:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        shutil.rmtree(export.default_journal_dir(out), ignore_errors=True)


def _publish_bank(out: Path, resolved: _Bank, runs: Path) -> None:
    from ..bop import banks

    if resolved.policy is not None:
        banks.bank_manifest(out, resolved.policy, resolved.bank, resolved.fallbacks, runs)
    else:
        publish_json(out / SELECTION_NAME, _selection_record(resolved.choices))


def _publish_export(out: Path, resolved: _Bank, args: argparse.Namespace, summary: Any) -> None:  # noqa: ANN401
    """Publish the bank record and ``summary.json`` beside a finished export's CSV."""
    _publish_bank(out.parent, resolved, args.runs)
    record = dataclasses.asdict(summary)
    record["csv_path"] = str(record["csv_path"])
    publish_json(out.parent / SUMMARY_NAME, {**record, "policy": args.policy})


def _export(recipe: Recipe, profile: DatasetProfile, args: argparse.Namespace, out: Path) -> None:
    from ..bop import export

    _start_over(out, args.force, [])
    resolved = _resolve_bank(recipe, args, args.policy, record_time=args.record_time)
    policy, bank = resolved.policy, resolved.bank
    fixed = policy is not None
    record_time = args.record_time
    if fixed:
        record_time = record_time or recipe.test.record_time or policy.record_time
    cache = None
    if fixed and not record_time:
        cache = export.JournalCache(Path(args.runs) / TEST_DIR / CACHE_DIR)
    summary = export.run_bank(
        bank,
        held_out_population(recipe, args.root),
        out,
        camera=recipe.test.camera,
        objects=recipe.objects,
        allow_partial_bank=args.allow_partial_bank,
        record_time=record_time,
        expected_population=export.profile_population(profile),
        device=_device(args.device),
        cache=cache,
    )
    # Published only after the export succeeded, so selection, manifest and summary always
    # describe the CSV beside them.
    _publish_export(out, resolved, args, summary)
    log(f"wrote {summary.csv_path}: {summary.pose_rows} poses on {summary.images} images")


def bop_csv_path(recipe: Recipe, runs: Path, policy: str) -> Path:
    """Path of a policy's BOP CSV: ``<runs>/test/<policy>/<csv method>_<dataset>-test.csv``.

    The built-in policies use ``<test.method><policy>``; a fixed policy its ``csv_method``.
    """
    from ..bop.export import csv_name

    method = f"{recipe.test.method}{policy}"
    if policy not in POLICIES:
        from ..bop.banks import recipe_policy

        method = recipe_policy(recipe, policy).csv_method
    name = csv_name(method, recipe.test.dataset or recipe.name)
    return Path(runs) / TEST_DIR / policy / name


def variant_csv_path(recipe: Recipe, runs: Path, policy: str, variant: str) -> Path:
    """Path of a variant CSV: ``<runs>/test/<policy>/<variant>/<method>_<dataset>-test.csv``.

    The method token is ``<csv method of the policy><variant>`` (``doper.bop.variants``).
    """
    from ..bop.export import csv_name
    from ..bop.variants import csv_method

    name = csv_name(csv_method(recipe, policy, variant), recipe.test.dataset or recipe.name)
    return Path(runs) / TEST_DIR / policy / variant / name


def refine_spec(recipe: Recipe, args: argparse.Namespace) -> RefineSpec:
    """The heatmap settings of a variant export: the recipe's, overridden by the flags.

    Raises:
        ValueError: For flags without ``--variant``, a multi-view variant on a recipe without
            a multi-view policy, ``--record-time``, or Powell with the parallel preparation.
    """
    if args.variant is None:
        if args.optimizer is not None or args.preparation is not None:
            raise ValueError("--optimizer and --preparation apply to --variant exports only")
        return recipe.refine
    if args.split != "test":
        raise ValueError("--variant is a test export (--split test)")
    if args.variant in MULTIVIEW_VARIANTS and recipe.multiview is None:
        raise ValueError(
            f"{recipe.name} has no multi-view policy (recipe multiview block), so --variant "
            f"{args.variant} is refused; its test variants are singleview and singleheatmap"
        )
    if args.record_time:
        raise ValueError(
            "--variant exports are untimed post-passes (time -1); time them with the "
            "deployment timing protocol instead of --record-time"
        )
    spec = recipe.refine.with_overrides(args.optimizer, args.preparation)
    if args.variant in HEATMAP_VARIANTS:
        require_refine_extra(spec, f"--variant {args.variant}")
    return spec


def require_refine_extra(spec: RefineSpec, what: str) -> None:
    """Refuse, before any scan, heatmap refinement whose optimizer or preparation needs numba.

    Args:
        spec: The heatmap refinement settings.
        what: The refusing command, for the message (``--variant singleheatmap``).

    Raises:
        MissingExtraError: L-BFGS-B or the parallel preparation without numba installed.
    """
    if spec.needs_numba and not numba_installed():
        raise MissingExtraError(
            f"{what} with optimizer {spec.optimizer} and {spec.preparation} "
            "preparation needs numba: install doper[refine], or pass --optimizer powell "
            "--preparation numpy"
        )


def _scene_cameras(test_root: Path, scenes: set[int]) -> dict[Any, Any]:
    from ..bop.multiview import scene_cameras

    cameras = {}
    for scene in sorted(scenes):
        path = test_root / f"{scene:06d}" / "scene_camera.json"
        cameras.update(scene_cameras(scene, read_json(path)))
    return cameras


def support_manifest(recipe: Recipe, root: Path) -> Path:
    """The staged HOT3D support manifest of a streams recipe.

    Raises:
        ValueError: If the support streams are not staged.
    """
    from ..bop import hot3d_staging

    path = Path(root) / hot3d_staging.TEST_DIR / hot3d_staging.SUPPORT_MANIFEST
    if not path.is_file():
        raise ValueError(
            f"HOT3D support streams are not staged ({path} is missing): run `doper prepare "
            f"{recipe.name} --root ROOT --parts test --allow-test-inputs`, which stages them"
        )
    return path


def _support_view(path: Path, stream: str, group: Any, view: Any) -> tuple[dict[str, Any], Any]:  # noqa: ANN401
    """One support view of the staged manifest: its population row and its ``TestFrame``.

    The native image and calibration ride along (deployment timing rectifies them); the scan
    uses the staged rectified image and the population identity is the rows only.
    """
    from ..bop.export import TestFrame

    row = {
        "stream": stream,
        "scene_id": int(group["scene_id"]),
        "im_id": int(group["im_id"]),
        "path": view["path"],
        "rgb_sha256": view["rgb_sha256"],
        "cam_K": view["cam_K"],
    }
    frame = TestFrame(
        row["scene_id"],
        row["im_id"],
        path.parent / row["path"],
        row["rgb_sha256"],
        tuple(float(x) for x in row["cam_K"]),
        native_path=path.parent / view["native_path"],
        native_rgb_sha256=view["native_rgb_sha256"],
        native_calibration=view["native_calibration"],
    )
    return row, frame


def _stream_inputs(recipe: Recipe, root: Path, frames: list[Any], population: str) -> Any:  # noqa: ANN401
    """HOT3D: groups and support populations from the staged support manifest."""
    from ..bop import hot3d_staging
    from ..bop.multiview import ImageKey, support_groups
    from ..bop.sources import canonical_sha256
    from ..bop.variants import SupportPopulation, VariantInputs

    spec = recipe.multiview
    assert spec is not None
    path = support_manifest(recipe, root)
    manifest = read_json(path)
    cameras = hot3d_staging.support_cameras(manifest)
    references = {
        (int(g["scene_id"]), int(g["im_id"])): ImageKey(
            int(g["scene_id"]), int(g["im_id"]), g["reference"]["stream"]
        )
        for g in manifest["groups"]
    }
    if set(references) != {f.key for f in frames}:
        raise ValueError(f"{path} does not cover exactly the HOT3D test population")
    supports = []
    for stream in spec.streams:
        views = [
            _support_view(path, stream, g, v)
            for g in manifest["groups"]
            for v in g["supports"]
            if v["stream"] == stream
        ]
        if views:
            rows, support_frames = zip(*views, strict=True)
            supports.append(SupportPopulation(stream, support_frames, canonical_sha256(rows)))
    groups = support_groups(references.values(), cameras, spec)
    return VariantInputs(
        groups, cameras, references, tuple(supports), frames=tuple(frames), population=population
    )


def variant_inputs(recipe: Recipe, root: Path, variant: str) -> Any:  # noqa: ANN401
    """Groups, cameras and support populations of a variant export (``VariantInputs``).

    Without a multi-view variant every reference is its own group. A ``nearest`` recipe reads
    every test scene's ``scene_camera.json`` (with extrinsics); HOT3D's multi-view variants read
    the staged support manifest.

    Raises:
        ValueError: If HOT3D support streams are not staged.
    """
    from ..bop.export import load_test_frames
    from ..bop.multiview import ImageKey, support_groups
    from ..bop.variants import VariantInputs

    spec = recipe.multiview
    multiview = variant in MULTIVIEW_VARIANTS
    streams = multiview and spec is not None and spec.support == "streams"
    if streams:
        support_manifest(recipe, root)  # refuse before hashing the test population
    test_root = held_out_population(recipe, root)
    frames, population = load_test_frames(test_root, recipe.test.camera)
    if streams:
        return _stream_inputs(recipe, root, frames, population)
    references = {f.key: ImageKey(*f.key) for f in frames}
    if spec is not None and spec.support == "nearest":
        cameras = _scene_cameras(test_root, {f.scene_id for f in frames})
    else:
        cameras = {ImageKey(*f.key): {"cam_K": list(f.camera_matrix)} for f in frames}
    if multiview:
        assert spec is not None
        groups = support_groups(references.values(), cameras, spec)
    else:
        groups = {key: (key,) for key in references.values()}
    return VariantInputs(groups, cameras, references, frames=tuple(frames), population=population)


def _export_variant(
    recipe: Recipe,
    profile: DatasetProfile,
    args: argparse.Namespace,
    out: Path,
    spec: RefineSpec,
) -> None:
    from ..bop import export, variants

    results = out.with_name(f"{out.stem}{variants.OBJECTS_DIR_SUFFIX}")
    _start_over(out, args.force, [out.parent / variants.MANIFEST_NAME, results])
    resolved = _resolve_bank(recipe, args, args.policy, record_time=False)
    inputs = variant_inputs(recipe, args.root, args.variant)
    settings = spec.to_settings()
    summary = variants.run_variant(
        resolved.bank,
        held_out_population(recipe, args.root),
        out,
        variant=args.variant,
        inputs=inputs,
        spec=recipe.multiview,
        refine=settings,
        cache=export.JournalCache(Path(args.runs) / TEST_DIR / CACHE_DIR),
        camera=recipe.test.camera,
        objects=recipe.objects,
        allow_partial_bank=args.allow_partial_bank,
        expected_population=export.profile_population(profile),
        device=_device(args.device),
        manifest_extra={"policy": args.policy, "recipe": recipe.name},
    )
    _publish_export(out, resolved, args, summary)
    log(
        f"wrote {summary.csv_path}: {summary.pose_rows} {args.variant} poses on "
        f"{summary.images} images ({spec.optimizer}, {spec.preparation} preparation)"
    )


def _variant_steps(
    recipe: Recipe, profile: DatasetProfile, args: argparse.Namespace, spec: RefineSpec
) -> list[Step]:
    from ..bop.variants import MANIFEST_NAME

    out = variant_csv_path(recipe, args.runs, args.policy, args.variant)
    inputs: dict[str, Any] = {
        "runs": args.runs,
        "policy": args.policy,
        "test": held_out_population(recipe, args.root),
        "camera": recipe.test.camera,
        "model cache": Path(args.runs) / TEST_DIR / CACHE_DIR,
        "timing": "untimed (time -1)",
    }
    if args.variant in HEATMAP_VARIANTS:
        inputs["refine"] = f"optimizer {spec.optimizer}, {spec.preparation} preparation"
    if args.variant in MULTIVIEW_VARIANTS and recipe.multiview is not None:
        mv = recipe.multiview
        inputs["multiview"] = (
            f"{mv.convention}, streams {list(mv.streams)} (staged support streams)"
            if mv.support == "streams"
            else f"{mv.convention}, {mv.nearest_k} nearest ({mv.scope})"
        )
    outputs = {
        "csv": out,
        "summary": out.parent / SUMMARY_NAME,
        "variants manifest": out.parent / MANIFEST_NAME,
    }
    return [
        Step(
            f"export {args.policy} bank, {args.variant} variant",
            inputs,
            outputs,
            lambda: _export_variant(recipe, profile, args, out, spec),
        )
    ]


def _shirt_framewise(
    recipe: ShirtRecipe,
    args: argparse.Namespace,
    trajectories: tuple[str, ...],
    domain: str,
    out: Path,
) -> None:
    from ..shirt import candidates

    summary = candidates.framewise(
        args.checkpoint,
        args.root,
        args.keypoints_3d,
        out,
        trajectories=trajectories,
        domain=domain,
        recipe=recipe,
        allow_unpinned_checkpoint=args.allow_unpinned_checkpoint,
        limit=args.limit,
        device=_device(args.device),
        force=args.force,
    )
    for roe, domains in summary["trajectories"].items():
        result = domains[domain]
        log(f"SHIRT {roe}/{domain}: {result['solved']}/{result['requested']} solved, "
            f"mean rotation {result['rotation_deg']['mean']} deg")  # fmt: skip


def _shirt_candidates(
    recipe: ShirtRecipe,
    args: argparse.Namespace,
    trajectories: tuple[str, ...],
    domain: str,
    out: Path,
) -> None:
    from ..shirt import candidates

    summary = candidates.crop_candidates(
        args.checkpoint,
        args.root,
        args.keypoints_3d,
        Path(args.runs) / FRAMEWISE_DIR,
        out,
        trajectories=trajectories,
        domain=domain,
        recipe=recipe,
        allow_unpinned_checkpoint=args.allow_unpinned_checkpoint,
        limit=args.limit,
        device=_device(args.device),
        force=args.force,
    )
    for roe, domains in summary["trajectories"].items():
        result = domains[domain]
        log(f"SHIRT {roe}/{domain}: {result['frames']} frames, candidates "
            f"{result['candidates_by_source']}, "
            f"{len(result['empty_candidate_frames'])} without a raw pose")  # fmt: skip


def _caches_to_build(
    args: argparse.Namespace,
    trajectories: tuple[str, ...],
    domain: str,
    out: Path,
    candidates_root: Path,
) -> tuple[str, ...]:
    """The trajectories whose heatmap cache must be built: every one with ``--force``.

    Without ``--force``, a missing cache is built and a cache of these candidate frames, this
    checkpoint and this layout (points hash) is reused.

    Raises:
        ValueError: For an existing cache of other frames (a ``--limit`` change, new
            candidates), another checkpoint or another layout, naming what differs.
    """
    from ..bop.sources import sha256_file
    from ..shirt import policies
    from ..speedplus.checkpoints import load_layout

    if args.force:
        return trajectories
    checkpoint_sha = sha256_file(args.checkpoint)
    layout = load_layout(args.keypoints_3d).identity
    missing = []
    for roe in trajectories:
        path = policies.cache_path(out, roe, domain)
        records = policies.load_candidate_records(candidates_root, roe, domain, args.limit)
        problem = policies.heatmap_cache_problem(path, records, layout, checkpoint_sha)
        if problem == "missing":
            missing.append(roe)
        elif problem is not None:
            raise ValueError(
                f"the heatmap cache {path} does not fit this run ({problem}); pass --force "
                "to rebuild it"
            )
    return tuple(missing)


def _shirt_filter(
    recipe: ShirtRecipe,
    args: argparse.Namespace,
    trajectories: tuple[str, ...],
    domain: str,
    out: Path,
) -> None:
    from ..shirt import policies

    candidates_root = Path(args.runs) / CANDIDATES_DIR
    if (out / SUMMARY_NAME).exists() and not args.force:
        raise FileExistsError(f"{args.method} outputs already exist ({out}); pass --force")
    caches = None
    if args.method == "keypoint-ukf":
        caches = out
        build = _caches_to_build(args, trajectories, domain, out, candidates_root)
        if build:
            policies.cache_heatmap_measurements(
                args.checkpoint, args.root, args.keypoints_3d, candidates_root, out,
                trajectories=build, domain=domain, recipe=recipe,
                allow_unpinned_checkpoint=args.allow_unpinned_checkpoint, limit=args.limit,
                device=_device(args.device), force=args.force,
            )  # fmt: skip
        reused = [roe for roe in trajectories if roe not in build]
        if reused:
            log(f"SHIRT keypoint-ukf: reusing the heatmap caches of {reused} under {out}")
    summary = policies.filter_trajectories(
        args.method, args.root, candidates_root, out, trajectories=trajectories, domain=domain,
        layout=args.keypoints_3d, caches_root=caches, limit=args.limit, force=args.force,
    )  # fmt: skip
    for roe, domains in summary["trajectories"].items():
        result = domains[domain]
        if args.method == "pnp-ukf":
            means = (result["rotation_deg_mean"], result["translation_m_mean"])
        elif args.method == "pnp-ukf-rts":
            means = (result["rts"]["rotation_deg_mean_std"][0],
                     result["rts"]["translation_m_mean_std"][0])  # fmt: skip
        else:
            direct = result["direct_keypoint_ukf"]
            means = (direct["rotation_deg_mean"], direct["translation_m_mean"])
        log(f"SHIRT {args.method} {roe}/{domain}: mean rotation {means[0]} deg, "
            f"mean translation {means[1]} m")  # fmt: skip


def _check_shirt_request(
    args: argparse.Namespace, trajectories: tuple[str, ...], domain: str
) -> None:
    """Refuse, at plan time, a SHIRT request a stage would refuse after loading its inputs.

    Raises:
        ValueError: For a missing or ignored ``--checkpoint``/``--keypoints-3d``, a negative
            ``--limit``, or an unknown trajectory or domain.
        MissingExtraError: For a filter without bayesfilter (``doper[filter]``).
    """
    from ..shirt import check_request

    model_inputs = (args.checkpoint, args.keypoints_3d)
    if args.method in SHIRT_MODEL_METHODS and None in model_inputs:
        raise ValueError(f"`--method {args.method}` needs --checkpoint and --keypoints-3d")
    if args.method not in SHIRT_MODEL_METHODS and model_inputs != (None, None):
        raise ValueError(
            f"`--method {args.method}` filters the candidate records and runs no model; it "
            "takes no --checkpoint or --keypoints-3d"
        )
    if args.limit < 0:
        raise ValueError(f"--limit must be 0 (all frames) or positive, got {args.limit}")
    check_request(trajectories, domain)
    if args.method in SHIRT_FILTERS and importlib.util.find_spec("bayesfilter") is None:
        raise MissingExtraError(
            f"`--method {args.method}` needs bayesfilter: install doper[filter]"
        )


def shirt_plan(args: argparse.Namespace) -> list[Step]:
    """Build the ``evaluate`` plan of a SHIRT recipe: one ``--method``.

    Raises:
        ValueError: For a BOP flag, a missing method, a missing (or, for the PnP filters,
            ignored) checkpoint or layout, a negative limit, or an unknown trajectory or domain.
        MissingExtraError: For a filter without bayesfilter.
    """
    recipe = load_any_recipe(args.recipe)
    if not isinstance(recipe, ShirtRecipe):
        raise ValueError(f"{recipe.name} is not a SHIRT recipe")
    _refuse_flags(args, BOP_FLAGS, "BOP")
    _refuse_flags(args, NUREC_FLAGS, "NuRec")
    if args.method is None:
        raise ValueError(f"`doper evaluate {recipe.name}` needs --method, one of {SHIRT_METHODS}")
    trajectories = recipe.trajectories
    if args.trajectories is not None:
        trajectories = tuple(t.strip() for t in args.trajectories.split(",") if t.strip())
    domain = args.domain or recipe.domain
    _check_shirt_request(args, trajectories, domain)
    pin = recipe.checkpoint_sha256 or "none"
    if args.allow_unpinned_checkpoint:
        pin += " (--allow-unpinned-checkpoint)"
    inputs = {"shirt": args.root, "trajectories": list(trajectories), "domain": domain,
              "checkpoint": args.checkpoint, "checkpoint pin": pin,
              "layout": args.keypoints_3d}  # fmt: skip
    framewise_out = Path(args.runs) / FRAMEWISE_DIR
    if args.method in SHIRT_FILTERS:
        return [_shirt_filter_step(recipe, args, trajectories, domain, inputs)]
    if args.method == "framewise":
        out = framewise_out
        inputs["inference"] = "top-1 detection; no CLAHE, TTA or box pad"
        action = lambda: _shirt_framewise(recipe, args, trajectories, domain, out)  # noqa: E731
    else:
        out = Path(args.runs) / CANDIDATES_DIR
        inputs["framewise records"] = framewise_out / "<roe>" / domain / "records.jsonl"
        inputs["re-crops"] = "raw pose's projected box, rolled 0 and 180 deg; aux scoring null"
        action = lambda: _shirt_candidates(recipe, args, trajectories, domain, out)  # noqa: E731
    outputs = {"records": out / "<roe>" / domain / "records.jsonl", "summary": out / SUMMARY_NAME}
    return [Step(f"evaluate SHIRT {args.method}", inputs, outputs, action)]


def _shirt_filter_step(
    recipe: ShirtRecipe,
    args: argparse.Namespace,
    trajectories: tuple[str, ...],
    domain: str,
    inputs: dict[str, Any],
) -> Step:
    """The step of one Table 2 filter (``doper.shirt.policies``)."""
    out = Path(args.runs) / args.method
    inputs["candidate records"] = (
        Path(args.runs) / CANDIDATES_DIR / "<roe>" / domain / "records.jsonl"
    )
    inputs["metadata"] = Path(args.root) / "<roe>" / "metadata.json"
    inputs["process noise"] = "1e-7 rad/s^2, 1e-6 m/s^2 (SHIRT-tuned)"
    if args.method == "keypoint-ukf":
        inputs["landmark gate"] = "chi-square 2-DoF p99, covariance scale 1, floor 1 px"
    else:
        inputs["candidate gate"] = "chi-square 6-DoF p99, minimum NIS"
        for key in ("checkpoint", "checkpoint pin", "layout"):
            inputs.pop(key)
    if args.method == "pnp-ukf-rts":
        inputs["smoother"] = "one unscented RTS pass (offline)"
    outputs: dict[str, Any] = {"records": out / "<roe>" / domain / "records.jsonl",
                               "summary": out / SUMMARY_NAME}  # fmt: skip
    if args.method == "keypoint-ukf":
        outputs["heatmap cache"] = out / "<roe>" / domain / "heatmap_measurements.npz"
    return Step(
        f"evaluate SHIRT {args.method}", inputs, outputs,
        lambda: _shirt_filter(recipe, args, trajectories, domain, out),
    )  # fmt: skip


def plan(args: argparse.Namespace) -> list[Step]:
    """Build the ``evaluate`` plan of a BOP recipe.

    Raises:
        ValueError: For a SHIRT flag or a missing ``--split``.
    """
    _refuse_flags(args, SHIRT_FLAGS, "SHIRT")
    _refuse_flags(args, NUREC_FLAGS, "NuRec")
    if args.split is None:
        raise ValueError("`doper evaluate` of a BOP recipe needs --split val or test")
    recipe, profile = load(args.recipe)
    validation_run(recipe, args.arm)
    spec = refine_spec(recipe, args)
    if args.split == "val":
        return _validation_steps(recipe, args)
    if args.variant is not None:
        return _variant_steps(recipe, profile, args, spec)
    out = bop_csv_path(recipe, args.runs, args.policy)
    partial = sorted(recipe.generated_objects)
    inputs: dict[str, Any] = {
        "runs": args.runs,
        "objects": f"{len(recipe.trainable_objects)} with official PBR"
        + (
            f"; without official PBR: {partial} (once prepared from a render root)"
            if partial
            else ""
        ),
        "test": held_out_population(recipe, args.root),
        "camera": recipe.test.camera,
        "population": "profile frozen test population",
    }
    outputs: dict[str, Any] = {"csv": out, "summary": out.parent / SUMMARY_NAME}
    policy = fixed_policy(recipe, args.policy, args.record_time)
    if policy is None:
        outputs["selection"] = out.parent / SELECTION_NAME
    else:
        from ..bop import banks

        timed = args.record_time or recipe.test.record_time or policy.record_time
        inputs["method"] = f"{policy.method} (fallback: {policy.fallback})"
        inputs["timing"] = "timed, every model rescanned" if timed else "untimed"
        if not timed:
            outputs["model cache"] = Path(args.runs) / TEST_DIR / CACHE_DIR
        outputs["model matrix"] = out.parent / banks.MATRIX_NAME
        outputs["skips"] = out.parent / banks.SKIPS_NAME
    return [
        Step(
            f"export {args.policy} bank",
            inputs,
            outputs,
            lambda: _export(recipe, profile, args, out),
        )
    ]


def nurec_plan(args: argparse.Namespace) -> list[Step]:
    """Build the ``evaluate`` plan of a NuRec recipe: relative camera poses of one checkpoint.

    Raises:
        ValueError: For a BOP or SHIRT flag, or a missing ``--bundle``, ``--checkpoint`` or
            ``--out``.
    """
    from ..evaluation import relative_pose
    from ..nurec import capture
    from ..recipes.nurec_schema import NurecRecipe

    recipe = load_any_recipe(args.recipe)
    if not isinstance(recipe, NurecRecipe):
        raise ValueError(f"{recipe.name} is not a NuRec recipe")
    _refuse_flags(args, BOP_FLAGS, "BOP")
    _refuse_flags(args, {k: v for k, v in SHIRT_FLAGS.items() if k != "checkpoint"}, "SHIRT")
    missing = [
        flag
        for flag, value in (
            ("--bundle", args.bundle),
            ("--checkpoint", args.checkpoint),
            ("--out", args.out),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"`doper evaluate {recipe.name}` needs {', '.join(missing)}")
    bundle, out = Path(args.bundle), Path(args.out)
    pin = capture.profile_bundle_pin(recipe.profile)
    request = relative_pose.EvaluationRequest(
        bundle=bundle,
        checkpoint=Path(args.checkpoint),
        out=out,
        keypoints_points_sha256=recipe.object.keypoints_points_sha256,
        profile=recipe.profile,
        score_thr=recipe.evaluation.score_thr,
        exclude=recipe.evaluation.exclude,
        force=args.force,
        bundle_pin=pin,
    )
    inputs = {
        "bundle": bundle,
        "bundle identity": f"pinned {pin} (profile {recipe.profile})"
        if pin
        else f"recorded (profile {recipe.profile} pins none)",
        "robot reference": f"{bundle / 'robot.jsonl'}, {bundle / 'handeye.json'} (required)",
        "keypoints": bundle / "keypoints_3d.json",
        "keypoint points SHA-256": recipe.object.keypoints_points_sha256,
        "checkpoint": request.checkpoint,
        "inference": f"top-1 detection, score >= {request.score_thr}, no TTA, zero distortion",
        "exclude": f"{list(request.exclude)} and the bundle's exclude.json",
    }
    outputs = {name.removesuffix(".json"): out / name for name in relative_pose.OUTPUT_NAMES}

    def action() -> None:
        device = _device(args.device)
        summary = relative_pose.evaluate_bundle(dataclasses.replace(request, device=device))
        log(
            f"relative pose: {summary['pose_successes']}/{summary['frames']} poses, "
            f"{summary['all_pairs']['count']}/{summary['total_all_pairs']} pairs (bundle "
            f"{summary['bundle_sha256'][:12]})"
        )

    return [Step("evaluate relative camera poses", inputs, outputs, action)]


def run(args: argparse.Namespace) -> int:
    """Run ``doper evaluate`` (BOP, SHIRT and NuRec recipes)."""
    planners = {BOP_KIND: plan, "shirt": shirt_plan, "nurec": nurec_plan}
    return run_plan(route_kind(args, planners, "evaluate"), dry_run=args.dry_run)
