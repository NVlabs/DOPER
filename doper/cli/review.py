# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``doper review``: record a human review of a label pool's overlays (``REVIEWED.json``).

Some pools train only after a person has inspected every overlay (``doper.bop.review``). The
review binds the reviewer to the exact overlay bytes and to the pool's receipt, so relabeling
or a changed overlay makes it stale and training refuses it again (``--accept-unreviewed``
trains anyway and records that).

Review targets:

* a review-gated follow-up (``FollowupSpec.require_review``, HANDAL object 7's bootstrap): its
  round 1 run ``<runs>/<name>/round01/obj_XXXXXX``, whose ``labels_ready.json`` is the receipt
  and whose ``review/`` holds one overlay per label. ``doper train --stage followup`` renders
  them, then stops until the review exists;
* the T-LESS multisensor real TRAIN root of an object (``doper prepare tless --multisensor``):
  ``<root>/multisensor/obj_XXXXXX``, whose ``READY.json`` is the receipt and whose nine overlays
  (first, middle and last capture of each sensor) the review must cover. The multisensor
  adaptation verifies it (``doper.bop.multisensor.verify``). ``--runs`` is not read here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..bop import layout, review
from ..console import log
from ..recipes import BOP_KIND
from ..recipes.schema import Recipe
from .common import Step, add_recipe_argument, load, route_kind, run_plan


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``doper review``."""
    parser = subparsers.add_parser(
        "review",
        help="record a human review of label overlays",
        description="Record that a named reviewer inspected every overlay of a review-gated "
        "label pool; writes REVIEWED.json bound to the overlays and the pool receipt.",
    )
    add_recipe_argument(parser)
    parser.add_argument("--root", type=Path, required=True, help="prepared dataset root")
    parser.add_argument("--runs", type=Path, required=True, help="runs root")
    parser.add_argument("--obj-id", type=int, required=True, dest="obj_id", help="object")
    parser.add_argument("--reviewer", required=True, help="name of the person who reviewed")
    parser.add_argument(
        "--name", default=None, help="the review-gated follow-up (default: the object's only one)"
    )
    parser.set_defaults(run=run)


def _followup_target(recipe: Recipe, args: argparse.Namespace) -> tuple[str, Path]:
    """The review-gated follow-up of the object and its round 1 run directory."""
    names = [
        name
        for name, spec in recipe.followups.items()
        if spec.require_review and spec.obj_id == args.obj_id and args.name in (None, name)
    ]
    if len(names) != 1:
        gated = [n for n, s in recipe.followups.items() if s.require_review]
        raise ValueError(
            f"no single review-gated {recipe.name} follow-up for object {args.obj_id}"
            f"{f' named {args.name}' if args.name else ''}; review-gated follow-ups: {gated}"
        )
    name = names[0]
    return name, layout.followup_round(args.runs, name, 1, args.obj_id)


def _multisensor_plan(args: argparse.Namespace) -> list[Step]:
    """Review of a T-LESS object's prepared three-sensor root."""
    from ..bop import multisensor

    out = multisensor.output_dir(args.root, args.obj_id)
    ready = out / multisensor.READY_NAME

    def action() -> None:
        if not ready.is_file():
            raise ValueError(
                f"{out} is not prepared; run `doper prepare tless --multisensor --obj-id "
                f"{args.obj_id}` first"
            )
        path = multisensor.approve(out, args.reviewer)
        overlays = len(multisensor.review_overlays(out))
        log(f"{args.reviewer} reviewed {overlays} overlays of {out}: {path}")

    return [
        Step(
            f"review multisensor obj {args.obj_id}",
            {
                "overlays": out / "*" / multisensor.REVIEW_DIR,
                "receipt": ready,
                "reviewer": args.reviewer,
            },
            {"review": out / review.REVIEWED_NAME},
            action,
        )
    ]


def plan(args: argparse.Namespace) -> list[Step]:
    """Build the ``review`` plan."""
    from ..training import followups

    recipe, profile = load(args.recipe)
    if args.name is None and "sensors" in profile.meta:
        return _multisensor_plan(args)
    name, run_dir = _followup_target(recipe, args)
    ready = run_dir / layout.LABELS_READY_NAME

    def action() -> None:
        overlays = followups.review_overlays(run_dir)
        missing = [p for p in overlays if not p.is_file()]
        if missing:
            raise ValueError(
                f"{len(missing)} of {len(overlays)} overlays are missing in {run_dir / 'review'}; "
                f"run `doper train --stage followup --name {name}` to render them"
            )
        path = review.write(run_dir, ready, overlays, args.reviewer)
        log(f"{args.reviewer} reviewed {len(overlays)} overlays of {run_dir}: {path}")

    return [
        Step(
            f"review {name} obj {args.obj_id}",
            {
                "overlays": run_dir / followups.REVIEW_DIR,
                "receipt": ready,
                "reviewer": args.reviewer,
            },
            {"review": run_dir / review.REVIEWED_NAME},
            action,
        )
    ]


def run(args: argparse.Namespace) -> int:
    """Run ``doper review`` (BOP recipes only)."""
    return run_plan(route_kind(args, {BOP_KIND: plan}, "review"), dry_run=args.dry_run)
