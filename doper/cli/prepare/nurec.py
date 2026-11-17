# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NuRec recipes of ``doper prepare``: one ``--stage`` per run.

A NuRec recipe (``kind: nurec``) runs one ``--stage`` at a time; the stage modules
are ``nurec_capture``, ``nurec_register``, ``nurec_render``, ``nurec_labels`` and
``nurec_synthetic``.

The stages that read the bundle's CAD mesh (``register``, ``labels``, ``synthetic``) refuse a
mesh whose SHA-256 is not the recipe's ``object.mesh_sha256``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ...recipes import load_any_recipe
from ...recipes.nurec_schema import NurecRecipe
from ..common import (
    Step,
)
from ._names import (
    ARTIFACT_STAGES,
    BOP_FLAGS,
    DOPER_PNP_FLAGS,
    FORCE_STEPS,
    NUREC_STAGES,
    SPEEDPLUS_FLAGS,
    force_step_keys,
    refuse_flags,
    refuse_stage_flags,
)
from .nurec_capture import CaptureContext, _nurec_capture_step, _nurec_reconstruct_step
from .nurec_labels import _nurec_labels_steps
from .nurec_register import _nurec_register_step
from .nurec_render import _nurec_render_steps
from .nurec_synthetic import _nurec_synthetic_steps


def _check_force_from(args: argparse.Namespace, recipe: NurecRecipe) -> None:
    """Refuse ``--force-from`` for a stage without it or a step the stage does not run."""
    start = getattr(args, "force_from", None)
    if start is None:
        return
    if args.stage not in FORCE_STEPS:
        raise ValueError(f"`doper prepare` takes --force-from only for --stage "
                         f"{', '.join(FORCE_STEPS)} (this is --stage {args.stage})")  # fmt: skip
    keys = force_step_keys(args.stage, recipe)
    if start not in keys:
        raise ValueError(f"--force-from {start} is not a step of --stage {args.stage}; its steps "
                         f"are {', '.join(keys)}")  # fmt: skip


def nurec_plan(args: argparse.Namespace) -> list[Step]:
    """Build the ``prepare`` plan of a NuRec recipe: one ``--stage``.

    Raises:
        ValueError: For a BOP or SPEED+ flag, a missing or unknown stage, or a capture stage
            without ``--bundle``.
    """
    recipe = load_any_recipe(args.recipe)
    if not isinstance(recipe, NurecRecipe):
        raise ValueError(f"{recipe.name} is not a NuRec recipe")
    refuse_flags(args, BOP_FLAGS, "BOP")
    refuse_flags(args, {k: v for k, v in SPEEDPLUS_FLAGS.items() if k != "stage"}, "SPEED+")
    if args.stage not in NUREC_STAGES:
        raise ValueError(f"`doper prepare {recipe.name}` needs --stage, one of {NUREC_STAGES}")
    doper_pnp = args.stage == "reconstruct" and recipe.seed_mode == "doper_pnp"
    given = [flag for name, flag in DOPER_PNP_FLAGS.items() if getattr(args, name, None)]
    if given and not doper_pnp:
        raise ValueError(f"`doper prepare` takes {given} only for --stage reconstruct with "
                         f"seed_mode doper_pnp (this is --stage {args.stage}, seed_mode "
                         f"{recipe.seed_mode})")  # fmt: skip
    if doper_pnp and args.checkpoint is None:
        raise ValueError("seed_mode doper_pnp needs --checkpoint: the DOPER checkpoint whose "
                         "poses seed COLMAP")  # fmt: skip
    refuse_stage_flags(args)
    if args.bundle is not None and args.stage != "capture":
        raise ValueError("`doper prepare` takes --bundle only for --stage capture (later stages "
                         "read the bundle the capture stage verified)")  # fmt: skip
    if args.artifact is not None and args.stage not in ARTIFACT_STAGES:
        raise ValueError(f"`doper prepare` takes --artifact only for --stage "
                         f"{' or '.join(ARTIFACT_STAGES)}")  # fmt: skip
    _check_force_from(args, recipe)
    root = Path(args.root)
    if args.stage == "capture":
        return [_nurec_capture_step(root, recipe, args)]
    # One context per plan: the steps share what they read of the capture (loaded lazily).
    ctx = CaptureContext(root, recipe)
    if args.stage == "render":
        return _nurec_render_steps(ctx, args)
    if args.stage == "register":
        return [_nurec_register_step(ctx, args)]
    if args.stage == "labels":
        return _nurec_labels_steps(ctx, args)
    if args.stage == "reconstruct":
        return [_nurec_reconstruct_step(ctx, args)]
    return _nurec_synthetic_steps(ctx, args)
