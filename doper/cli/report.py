# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``doper report``: summarize a recipe's validation AR and test exports, or score local AP.

``--ap`` instead computes the BOP24 AP (MSSD, MSPD) of each test CSV locally
(``doper.bop.ap``), for datasets whose test GT is public (T-LESS). It reads the GT of
``<root>/data/<test population>``, the ``models_eval`` meshes and a targets file, and writes
``<out>/ap_<recipe>_<policy>.json`` per policy (default: ``base`` and the recipe's fixed
policies) with the targets origin and the SHA-256 of the targets file and the CSV. With
``--variant`` (repeatable) it scores those Table 3 variant CSVs of each policy instead
(``<runs>/test/<policy>/<variant>/``) and writes ``<out>/ap_<recipe>_<policy>_<variant>.json``.
The targets
file is ``--targets`` (origin ``user``), else ``<root>/data/test_targets_bop24.json``
(``bop24``), else ``<root>/data/test_targets_bop19.json`` (``derived_from_bop19``; the pinned
``tless_base.zip`` ships only the BOP18 and BOP19 lists, and the detection protocol reads only
their image ids). The step fails only when none exists.

``--timing`` instead times each policy's bank under one of two protocols and writes
``<out>/timing/<policy>/<protocol>/``:

* ``--protocol direct``: the direct (cold) per-image runtime over the whole test population
  (``doper.bop.timing.measure``), ``summary.json``.
* ``--protocol deployment`` (default): the warm GB300 deployment protocol
  (``doper.bop.deployment_timing``) over the sampled request groups, ``requests.jsonl``,
  ``summary.json`` (weighted mean, median and p95 per variant) and ``manifest.json``, which
  records the optimizer and preparation (``--optimizer``/``--preparation``, default the
  recipe's ``refine`` block; the paper's GB300 rows used Powell), hardware and versions.

Both need the trained bank and the prepared test inputs on a CUDA device.

``--mask-audit`` (recipes with follow-ups, HANDAL) instead writes the post-training mask
diagnostic of every finished follow-up round (``doper.training.followups.mask_quality_audit``):
the IoU of each round's pool mask with the provided visible GT mask of the official static
onboarding archive, to ``<out>/mask_audit/<follow-up>.json`` (a round without a label pool is
recorded as ``missing``). The archive is
``--onboarding-zip``, else the one ``doper prepare`` downloaded to ``<root>/source``; either way
its SHA-256 must equal the profile's pin. The GT masks never feed back into labels, training,
selection or stopping.

A NuRec recipe (``kind: nurec``) takes ``--baseline`` and ``--adapted``, two
``doper evaluate`` relative-pose runs over the same captures of the same capture bundle (equal
``bundle_sha256``, score threshold, ``max_det`` and capture flags in both summaries), and writes
``<out>/paired_comparison.json`` (``doper.evaluation.relative_pose.compare_runs``: medians, p95s
and the median paired change on the pairs valid in both runs, and each run's failed captures).
An existing comparison is replaced only with ``--force``.
"""

from __future__ import annotations

import argparse
import dataclasses
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ..bop.ap import (
    BOP19_TARGETS_NAME,
    TARGETS_NAME,
    bop24_ap,
    require_public_gt,
    resolve_targets,
)
from ..bop.extract import data_dir
from ..bop.layout import VARIANTS
from ..bop.sources import publish_json, sha256_file, source_dir
from ..console import log
from ..recipes import BOP_KIND
from ..recipes.schema import Recipe, load_recipe
from ..reports.bop import ap_path
from .common import (
    Step,
    add_recipe_argument,
    add_refine_arguments,
    read_json,
    route_kind,
    run_plan,
)
from .evaluate import bop_csv_path, held_out_population, variant_csv_path

EVAL_MODELS_DIR = "models_eval"
TIMING_DIR = "timing"
TIMING_PROTOCOLS = ("direct", "deployment")
REQUESTS_NAME = "requests.jsonl"
TIMING_SUMMARY_NAME = "summary.json"
TIMING_MANIFEST_NAME = "manifest.json"
MASK_AUDIT_DIR = "mask_audit"
STATIC_PART = "static"
NUREC_FLAGS: dict[str, tuple[str, Any]] = {
    "baseline": ("--baseline", None),
    "adapted": ("--adapted", None),
}
# Flags of the BOP reports, refused for NuRec recipes, with the default each has.
BOP_REPORT_FLAGS: dict[str, tuple[str, Any]] = {
    "ap": ("--ap", False),
    "root": ("--root", None),
    "policies": ("--policy", None),
    "variants": ("--variant", None),
    "targets": ("--targets", None),
    "timing": ("--timing", False),
    "protocol": ("--protocol", None),
    "optimizer": ("--optimizer", None),
    "preparation": ("--preparation", None),
    "workers": ("--workers", None),
    "device": ("--device", None),
    "allow_partial_bank": ("--allow-partial-bank", False),
    "mask_audit": ("--mask-audit", False),
    "onboarding_zip": ("--onboarding-zip", None),
}


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``doper report``."""
    parser = subparsers.add_parser(
        "report",
        help="summarize validation AR and test exports, or compute local BOP AP",
        description="Write per-object validation AR tables and the test export summaries; "
        "with --ap, compute the BOP24 AP of the test CSVs of a public-GT dataset (T-LESS).",
    )
    add_recipe_argument(parser)
    parser.add_argument("--runs", type=Path, required=True, help="runs root")
    parser.add_argument("--out", type=Path, default=None, help="output directory (default: runs)")
    parser.add_argument(
        "--ap", action="store_true", help="compute local BOP24 AP of the test CSVs (T-LESS)"
    )
    parser.add_argument(
        "--root", type=Path, default=None, help="prepared dataset root (required with --ap)"
    )
    parser.add_argument(
        "--policy",
        action="append",
        dest="policies",
        help="test bank policy to score with --ap (repeatable; default: base and the recipe's "
        "fixed policies)",
    )
    parser.add_argument(
        "--variant",
        action="append",
        dest="variants",
        choices=VARIANTS,
        help="with --ap, score this test variant's CSV of each policy instead of the policy CSV "
        "(repeatable)",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=None,
        help=f"targets file for --ap (default: <root>/data/{TARGETS_NAME}, else "
        f"<root>/data/{BOP19_TARGETS_NAME}, which names the same test images)",
    )
    _add_timing_arguments(parser)
    nurec = parser.add_argument_group("NuRec recipes")
    nurec.add_argument("--baseline", type=Path, default=None, help="baseline evaluation run")
    nurec.add_argument("--adapted", type=Path, default=None, help="adapted evaluation run")
    parser.add_argument(
        "--mask-audit",
        action="store_true",
        help="post-training GT-mask IoU of every follow-up round's pool (diagnostic only)",
    )
    parser.add_argument(
        "--onboarding-zip",
        type=Path,
        default=None,
        help="official static onboarding archive for --mask-audit (default: the pinned one "
        "under <root>/source)",
    )
    parser.set_defaults(run=run)


def _add_timing_arguments(parser: argparse.ArgumentParser) -> None:
    """The ``--timing`` options (the direct and deployment timing protocols)."""
    timing = parser.add_argument_group("timing")
    timing.add_argument(
        "--timing", action="store_true", help="time each policy's bank (needs --root and CUDA)"
    )
    timing.add_argument(
        "--protocol",
        choices=TIMING_PROTOCOLS,
        default=None,
        help="timing protocol: direct (cold, every test image) or deployment (warm GB300 "
        "request sample; default)",
    )
    add_refine_arguments(timing, prefix="deployment ")
    timing.add_argument(
        "--workers", type=int, default=None, help="deployment CPU workers (default: 6)"
    )
    timing.add_argument("--device", default=None, help="torch device (default: cuda if present)")
    timing.add_argument(
        "--allow-partial-bank", action="store_true", help="time a bank that lacks some objects"
    )
    timing.add_argument(
        "--force",
        action="store_true",
        help="replace existing timing results (NuRec: the comparison)",
    )


def _ap_steps(recipe: Recipe, args: argparse.Namespace) -> list[Step]:
    """One AP step per policy CSV."""
    require_public_gt(recipe.test.dataset or recipe.name)
    if args.root is None:
        raise ValueError("--ap needs --root, the prepared dataset root")
    gt_root = held_out_population(recipe, args.root)
    dataset = data_dir(args.root)
    explicit = None if args.targets is None else Path(args.targets)
    targets = explicit or f"{dataset / TARGETS_NAME}, else {dataset / BOP19_TARGETS_NAME}"
    models = data_dir(args.root) / EVAL_MODELS_DIR
    out = Path(args.out or args.runs)
    steps = []
    for policy, variant in _ap_csvs(recipe, args):
        path = ap_path(out, recipe.name, policy, variant)
        if variant is None:
            csv, name = bop_csv_path(recipe, args.runs, policy), policy
        else:
            csv, name = variant_csv_path(recipe, args.runs, policy, variant), f"{policy} {variant}"

        def action(csv: Path = csv, path: Path = path, variant: str | None = variant) -> None:
            targets_path, origin = resolve_targets(dataset, explicit)
            result = bop24_ap(csv, gt_root, targets_path, models)
            record = {
                **({} if variant is None else {"variant": variant}),
                "csv": str(csv),
                "csv_sha256": sha256_file(csv),
                "targets": str(targets_path),
                "targets_origin": origin,
                "targets_sha256": sha256_file(targets_path),
                **result.to_json(),
            }
            publish_json(path, record)
            log(
                f"wrote {path}: AP {result.ap:.4f} (MSSD {result.ap_mssd:.4f}, MSPD "
                f"{result.ap_mspd:.4f}) over {len(result.per_object)} objects; targets {origin}"
            )

        inputs = {"csv": csv, "gt": gt_root, "targets": targets, "models": models}
        steps.append(Step(f"ap {name}", inputs, {"ap": path}, action))
    return steps


def _ap_csvs(recipe: Recipe, args: argparse.Namespace) -> list[tuple[str, str | None]]:
    """The (policy, variant) CSVs ``--ap`` scores; a None variant is the policy CSV."""
    policies = args.policies or ["base", *recipe.test.policies]
    variants: list[str | None] = list(dict.fromkeys(args.variants or ())) or [None]
    return [(policy, variant) for policy in policies for variant in variants]


def _refine_settings(recipe: Recipe, args: argparse.Namespace) -> Any:  # noqa: ANN401
    """The deployment ``RefineSettings``: the recipe's, overridden by the flags."""
    from .evaluate import require_refine_extra

    spec = recipe.refine.with_overrides(args.optimizer, args.preparation)
    settings = spec.to_settings()
    require_refine_extra(spec, "deployment timing of the heatmap variants")
    return settings


def _view_frames(inputs: Any) -> dict[Any, Any]:  # noqa: ANN401
    """Every reference and support view's ``TestFrame``, keyed by ``ImageKey``."""
    from ..bop.multiview import ImageKey

    paths = {inputs.reference_keys[f.key]: f for f in inputs.frames}
    for support in inputs.supports:
        paths.update({ImageKey(f.scene_id, f.im_id, support.stream): f for f in support.frames})
    return paths


def decode_views(
    paths: Mapping[Any, Any],
    keys: Iterable[Any],
) -> tuple[dict[Any, Any], dict[Any, Any]]:
    """Decode (and hash-check) the timed views before timing.

    A view with a native image (HOT3D) is decoded unchanged and paired with its precomputed
    rectification maps, whose intrinsics must equal the staged ``cam_K``.

    Args:
        paths: ``ImageKey`` to its ``TestFrame``.
        keys: The views to decode.

    Returns:
        The decoded images and the native views' ``(map_x, map_y)``.

    Raises:
        ValueError: For a view without a frame, changed or undecodable bytes, or rectification
            intrinsics that differ from the staged ``cam_K``.
    """
    import cv2
    import numpy as np

    from ..bop import hot3d
    from ..bop.export import read_checked_image

    images, maps = {}, {}
    for key in sorted(set(keys)):
        if key not in paths:
            raise ValueError(f"view {tuple(key)} has no image")
        frame = paths[key]
        if frame.native_calibration is None:
            images[key] = read_checked_image(
                frame.path, frame.rgb_sha256, cv2.IMREAD_COLOR, "image"
            )
            continue
        assert frame.native_path is not None
        images[key] = read_checked_image(
            frame.native_path, frame.native_rgb_sha256, cv2.IMREAD_UNCHANGED, "image"
        )
        map_x, map_y, intrinsics = hot3d.rectification_maps(dict(frame.native_calibration))
        if not np.allclose(intrinsics, np.reshape(frame.camera_matrix, (3, 3))):
            raise ValueError(f"view {tuple(key)}: rectification K differs from the staged cam_K")
        maps[key] = (map_x, map_y)
    return images, maps


def _bank_gap(missing: list[int]) -> dict[str, Any]:
    """The timing records' partial-bank fields."""
    return {"partial_bank": bool(missing), "missing_objects": missing}


def _deployment(recipe: Recipe, args: argparse.Namespace, policy: str, folder: Path) -> None:
    from ..bop import deployment_timing as deployment
    from ..bop import export
    from .evaluate import _device, _resolve_bank, variant_inputs

    refine = _refine_settings(recipe, args)
    resolved = _resolve_bank(recipe, args, policy, record_time=False)
    variant = "singleview" if recipe.multiview is None else "multiview"
    inputs = variant_inputs(recipe, args.root, variant)
    dataset = export.normalize_dataset(recipe.test.dataset or recipe.name)
    groups = deployment.sample_groups(dataset, sorted(inputs.groups), inputs.groups)
    device = _device(args.device)
    views = {k for g in groups for k in g.views}
    images, maps = decode_views(_view_frames(inputs), views)
    settings = deployment.DeploymentSettings(
        dataset=dataset,
        refine=refine,
        spec=recipe.multiview,
        cameras=inputs.cameras,
        images=images,
        maps=maps,
        device=device,
    )
    models = deployment.load_models(resolved.bank, device=device)
    result = deployment.run(
        models, groups, settings, workers=args.workers or deployment.WORKERS,
        requests_path=folder / REQUESTS_NAME,
    )  # fmt: skip
    bank = _bank_gap(resolved.missing)
    manifest = {**result.manifest, "recipe": recipe.name, "policy": policy, **bank}
    publish_json(folder / TIMING_MANIFEST_NAME, manifest)
    publish_json(folder / TIMING_SUMMARY_NAME, {
        "dataset": dataset, "policy": policy, "optimizer": refine.optimizer,
        "preparation": refine.preparation, "complete": result.manifest["complete"], **bank,
        "variants": result.summary,
    })  # fmt: skip
    for name, stats in result.summary.items():
        log(f"{name}: mean {stats['mean_s']:.3f} s, median {stats['median_s']:.3f} s, "
            f"p95 {stats['p95_s']:.3f} s over {stats['requests']} requests")  # fmt: skip


def _direct(recipe: Recipe, args: argparse.Namespace, policy: str, folder: Path) -> None:
    from ..bop import export, sources
    from ..bop import timing as direct
    from .evaluate import _device, _resolve_bank

    resolved = _resolve_bank(recipe, args, policy, record_time=True)
    profile = sources.load_profile(recipe.profile_ref())
    summary = direct.measure(
        resolved.bank,
        held_out_population(recipe, args.root),
        camera=recipe.test.camera,
        expected_population=export.profile_population(profile),
        device=_device(args.device),
    )
    record = dataclasses.asdict(summary)
    publish_json(folder / TIMING_SUMMARY_NAME, {
        **record, "policy": policy, **_bank_gap(resolved.missing),
    })  # fmt: skip
    log(f"{summary.images} images: mean {summary.mean_seconds:.3f} s, median "
        f"{summary.median_seconds:.3f} s, p95 {summary.p95_seconds:.3f} s")  # fmt: skip


def _timing_steps(recipe: Recipe, args: argparse.Namespace) -> list[Step]:
    """One timing step per policy bank."""
    if args.root is None:
        raise ValueError("--timing needs --root, the prepared dataset root")
    protocol = args.protocol or "deployment"
    if args.workers is not None and args.workers < 1:
        raise ValueError(f"--workers must be >= 1, got {args.workers}")
    if protocol == "direct" and (args.optimizer or args.preparation or args.workers):
        raise ValueError("--optimizer, --preparation and --workers apply to --protocol deployment")
    if protocol == "deployment":
        refine = _refine_settings(recipe, args)
        method = f"{refine.optimizer}, {refine.preparation} preparation"
    else:
        method = "cold, every test image"
    out = Path(args.out or args.runs)
    steps = []
    for policy in args.policies or ["base", *recipe.test.policies]:
        folder = out / TIMING_DIR / policy / protocol
        outputs = {"summary": folder / TIMING_SUMMARY_NAME}
        if protocol == "deployment":
            outputs.update(requests=folder / REQUESTS_NAME, manifest=folder / TIMING_MANIFEST_NAME)

        def action(policy: str = policy, folder: Path = folder) -> None:
            if (folder / TIMING_SUMMARY_NAME).exists() and not args.force:
                raise SystemExit(f"{folder / TIMING_SUMMARY_NAME} exists; pass --force")
            timed = _deployment if protocol == "deployment" else _direct
            timed(recipe, args, policy, folder)
            log(f"wrote {folder}")

        inputs = {"runs": args.runs, "root": args.root, "protocol": protocol, "method": method}
        steps.append(Step(f"timing {policy} ({protocol})", inputs, outputs, action))
    return steps


def _static_archive_name(recipe: Recipe) -> tuple[str, str]:
    """The static onboarding archive's file name and pinned SHA-256."""
    from ..bop import sources
    from ..bop.extract import zip_archives

    profile = sources.load_profile(recipe.profile_ref())
    archives = zip_archives(profile, STATIC_PART)
    if len(archives) != 1 or len(archives[0]) != 1:
        raise ValueError(f"{recipe.name}: expected one single-volume static onboarding archive")
    name = archives[0][0]
    return name, profile.files[name]


def _audit_series(runs: Path, name: str, archive: Path) -> list[dict[str, Any]]:
    """The mask audit of each round of one follow-up's ``progress.json``."""
    from ..bop import layout
    from ..training.followups import mask_quality_audit

    progress = read_json(Path(runs) / name / layout.PROGRESS_NAME)
    rounds = []
    for entry in progress["rounds"]:
        run_dir = Path(runs) / entry["run"]
        ready = run_dir / layout.LABELS_READY_NAME
        if not ready.is_file():
            rounds.append({"round": entry["round"], "run": entry["run"], "status": "missing"})
            continue
        labels_root = (run_dir / read_json(ready)["labels_root"]).resolve()
        audit = mask_quality_audit(labels_root, archive)
        rounds.append({"round": entry["round"], "run": entry["run"], "status": "audited", **audit})
    return rounds


def _mask_audit_steps(recipe: Recipe, args: argparse.Namespace) -> list[Step]:
    """One mask-audit step per follow-up of the recipe."""
    from ..bop import layout

    if not recipe.followups:
        raise ValueError(f"--mask-audit needs a recipe with follow-ups; {recipe.name} has none")
    if args.onboarding_zip is None and args.root is None:
        raise ValueError("--mask-audit needs --onboarding-zip or --root (its pinned archive)")
    name, pin = _static_archive_name(recipe)
    archive = Path(args.onboarding_zip or source_dir(args.root) / name)
    out = Path(args.out or args.runs) / MASK_AUDIT_DIR
    verified: list[Path] = []
    steps = []
    for followup in recipe.followups:
        path = out / f"{followup}.json"

        def action(followup: str = followup, path: Path = path) -> None:
            if not (Path(args.runs) / followup / layout.PROGRESS_NAME).is_file():
                log(f"{followup}: no progress.json; not audited")
                return
            if not verified:  # hash the multi-GB archive once for every follow-up
                if sha256_file(archive) != pin:
                    raise ValueError(f"{archive}: SHA-256 differs from the pinned {name} ({pin})")
                verified.append(archive)
            rounds = _audit_series(args.runs, followup, archive)
            protocol = next((r["protocol"] for r in rounds if "protocol" in r), None)
            publish_json(path, {"followup": followup, "archive": name, "archive_sha256": pin,
                                "protocol": protocol, "rounds": rounds})  # fmt: skip
            log(f"wrote {path}: {len(rounds)} rounds")

        inputs = {"runs": Path(args.runs) / followup, "archive": archive}
        steps.append(Step(f"mask audit {followup}", inputs, {"audit": path}, action))
    return steps


def plan(args: argparse.Namespace) -> list[Step]:
    """Build the ``report`` plan."""
    from ..reports.bop import write_report

    recipe = load_recipe(args.recipe)
    _refuse_flags(args, NUREC_FLAGS, "NuRec")
    timing_only = (args.protocol, args.optimizer, args.preparation, args.workers)
    if not args.timing and any(v is not None for v in timing_only):
        raise ValueError("--protocol, --optimizer, --preparation and --workers need --timing")
    if sum(map(bool, (args.timing, args.ap, args.mask_audit))) > 1:
        raise ValueError("--timing, --ap and --mask-audit are separate reports")
    if args.onboarding_zip is not None and not args.mask_audit:
        raise ValueError("--onboarding-zip needs --mask-audit")
    if args.variants and not args.ap:
        raise ValueError("--variant needs --ap")
    if args.mask_audit:
        return _mask_audit_steps(recipe, args)
    if args.timing:
        return _timing_steps(recipe, args)
    if args.ap:
        return _ap_steps(recipe, args)
    out = Path(args.out or args.runs)

    def action() -> None:
        json_path, md_path = write_report(recipe, args.runs, out)
        log(f"wrote {md_path} and {json_path}")

    return [
        Step(
            f"report {recipe.name}",
            {"runs": args.runs},
            {
                "json": out / f"report_{recipe.name}.json",
                "markdown": out / f"report_{recipe.name}.md",
            },
            action,
        )
    ]


def _refuse_flags(args: argparse.Namespace, flags: dict[str, tuple[str, Any]], kind: str) -> None:
    given = [flag for name, (flag, default) in flags.items() if getattr(args, name) != default]
    if given:
        raise ValueError(f"`doper report` takes {given} only for {kind} recipes")


def _compare(baseline: Path, adapted: Path, out: Path, force: bool) -> None:
    from ..evaluation import relative_pose

    path = out / relative_pose.COMPARISON_NAME
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; pass --force to replace it")
    result = relative_pose.write_comparison(baseline, adapted, out)
    rotation = result["all_pairs"][relative_pose.ROTATION_KEY] or {}
    log(
        f"{result['all_pairs']['common_valid_pairs']} common pairs; rotation median "
        f"{rotation.get('baseline_median')} -> {rotation.get('finetune_median')} deg"
    )


def nurec_plan(args: argparse.Namespace) -> list[Step]:
    """Build the ``report`` plan of a NuRec recipe: the paired relative-pose comparison.

    Raises:
        ValueError: For a BOP report flag or a missing ``--baseline`` or ``--adapted``.
    """
    from ..evaluation.relative_pose import COMPARISON_NAME

    _refuse_flags(args, BOP_REPORT_FLAGS, "BOP")
    missing = [
        flag
        for flag, value in (("--baseline", args.baseline), ("--adapted", args.adapted))
        if value is None
    ]
    if missing:
        raise ValueError(f"`doper report {args.recipe}` needs {', '.join(missing)}")
    baseline, adapted = Path(args.baseline), Path(args.adapted)
    out = Path(args.out or args.runs)
    inputs = {
        "baseline": baseline,
        "adapted": adapted,
        "pairs": "common valid pairs of both runs; no alignment or error filtering",
    }
    return [
        Step(
            "compare relative camera poses",
            inputs,
            {"comparison": out / COMPARISON_NAME},
            lambda: _compare(baseline, adapted, out, args.force),
        )
    ]


def run(args: argparse.Namespace) -> int:
    """Run ``doper report`` (BOP and NuRec recipes)."""
    planners = {BOP_KIND: plan, "nurec": nurec_plan}
    return run_plan(route_kind(args, planners, "report"), dry_run=args.dry_run)
