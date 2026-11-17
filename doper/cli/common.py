# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared pieces of the ``doper`` subcommands: plans of steps, paths and receipts.

Every subcommand builds an ordered plan of steps, each with its resolved inputs and outputs.
``--dry-run`` prints the plan and stops; otherwise the steps run in order. Building a plan reads
only the recipe and its profile, never the dataset or run directories, and imports no torch.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..bop import sources
from ..bop.sources import DatasetProfile
from ..console import log
from ..recipes import load_any_recipe, recipe_kind
from ..recipes.schema import REFINE_OPTIMIZERS, REFINE_PREPARATIONS, Recipe

OBJECTS_DIR = "objects"
PREPARED_NAME = "prepared.json"
IDENTITY_NAME = "identity.json"
# Key of a prepared object's identity that records its render campaign (generated objects).
GENERATED_KEY = "generated"
COMPLETED_NAME = "completed.json"
RESOLVED_CONFIG_NAME = "recipe.yaml"
DATASET_DIR = "dataset"
SPLITS = ("train", "val")
# NuRec recipes: the COCO dataset each training stage reads, under ``<root>/<name>/dataset``,
# and its run directory, ``<runs>/<stage>`` with checkpoints in ``work/``.
NUREC_DATASETS = {"base": "synthetic", "adapt": "adaptation"}
NUREC_WORK_DIR = "work"
NUREC_BEST = "best.pth"


class MissingExtraError(ImportError):
    """An optional dependency a planned step needs is not installed (names the extra)."""


@dataclass(frozen=True)
class Step:
    """One planned step.

    Attributes:
        name: Step name, for example ``convert obj 2``.
        inputs: Resolved inputs, name to value.
        outputs: Resolved outputs, name to value.
        action: What the step does; None for a step that only reports (dry-run notes).
    """

    name: str
    inputs: Mapping[str, Any] = field(default_factory=dict)
    outputs: Mapping[str, Any] = field(default_factory=dict)
    action: Callable[[], None] | None = None


def run_plan(steps: Sequence[Step], *, dry_run: bool) -> int:
    """Print the plan with ``dry_run``; otherwise run each step in order.

    Returns:
        Zero.
    """
    for index, step in enumerate(steps, 1):
        log(f"step {index}/{len(steps)}: {step.name}")
        if dry_run:
            for kind, values in (("in", step.inputs), ("out", step.outputs)):
                for key, value in values.items():
                    log(f"    {kind:>3} {key}: {value}")
        elif step.action is not None:
            step.action()
    if dry_run:
        log(f"dry run: {len(steps)} steps planned, nothing was read or written")
    return 0


def add_recipe_argument(parser: argparse.ArgumentParser) -> None:
    """Add the positional ``RECIPE`` and ``--dry-run``."""
    parser.add_argument("recipe", metavar="RECIPE", help="packaged recipe name or recipe YAML")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the ordered steps; touch no data"
    )


def add_refine_arguments(
    parser: argparse.ArgumentParser | argparse._ArgumentGroup, *, prefix: str = "", suffix: str = ""
) -> None:
    """Add ``--optimizer`` and ``--preparation``, the overrides of the recipe's ``refine`` block.

    Args:
        parser: The parser or argument group.
        prefix: Help text before ``heatmap optimizer``/``heatmap preparation``.
        suffix: Help text after them.
    """
    for flag, choices in (("optimizer", REFINE_OPTIMIZERS), ("preparation", REFINE_PREPARATIONS)):
        parser.add_argument(
            f"--{flag}",
            choices=choices,
            default=None,
            help=f"{prefix}heatmap {flag}{suffix} (default: the recipe's refine.{flag})",
        )


def load(recipe_ref: str) -> tuple[Recipe, DatasetProfile]:
    """Load a BOP recipe and its pinned profile.

    Raises:
        ValueError: If the recipe is a SPEED+ or SHIRT recipe (commands that take those route
            on the kind first, ``route_kind``).
    """
    recipe = load_any_recipe(recipe_ref)
    if not isinstance(recipe, Recipe):
        raise ValueError(
            f"{recipe.name} is a {recipe_kind(recipe)} recipe; this command takes BOP recipes"
        )
    return recipe, sources.load_profile(recipe.profile_ref())


def route_kind(
    args: argparse.Namespace,
    planners: Mapping[str, Callable[[argparse.Namespace], list[Step]]],
    command: str,
) -> list[Step]:
    """Build the plan of the planner for the kind of ``args.recipe``.

    Args:
        args: Parsed arguments with ``recipe``.
        planners: Plan builder per recipe kind the command takes.
        command: Subcommand name, for the refusal.

    Raises:
        ValueError: If the command takes no recipe of that kind.
    """
    kind = recipe_kind(load_any_recipe(args.recipe))
    if kind not in planners:
        raise ValueError(
            f"`doper {command}` does not take {kind} recipes; it takes {sorted(planners)} recipes"
        )
    return planners[kind](args)


def nurec_dataset_dir(root: Path, stage: str) -> Path:
    """Return the COCO dataset a NuRec training stage reads, ``<root>/<name>/dataset``."""
    return Path(root) / NUREC_DATASETS[stage] / DATASET_DIR


def nurec_run_dir(runs: Path, stage: str) -> Path:
    """Return a NuRec training stage's run directory, ``<runs>/<stage>``."""
    return Path(runs) / stage


def object_dir(root: Path, obj: int) -> Path:
    """Return one object's prepared directory, ``<root>/objects/obj_XXXXXX``."""
    return Path(root) / OBJECTS_DIR / f"obj_{obj:06d}"


def read_json(path: Path) -> Any:  # noqa: ANN401
    """Read a JSON file."""
    return json.loads(Path(path).read_text())


def verify_hashes(base: Path, hashes: Mapping[str, str]) -> None:
    """Require every ``relative path -> SHA-256`` of a receipt to still match.

    Raises:
        ValueError: If a recorded file changed.
    """
    for relative, digest in hashes.items():
        if sources.sha256_file(base / relative) != digest:
            raise ValueError(f"prepared input changed: {base / relative}")


def prepared_receipt(root: Path, obj: int) -> dict[str, Any]:
    """Return an object's verified ``prepared.json``.

    Raises:
        FileNotFoundError: If the object was not prepared.
        ValueError: If a prepared file changed or a dataset link does not resolve.
    """
    directory = object_dir(root, obj)
    path = directory / PREPARED_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run `doper prepare` for object {obj} first")
    receipt = read_json(path)
    verify_hashes(directory, receipt["hashes"])
    require_resolving_links(directory / DATASET_DIR, obj)
    return receipt


def prepared_generated_objects(recipe: Recipe, root: Path) -> tuple[int, ...]:
    """Generated objects of a recipe prepared from a render root under ``root``.

    Such an object has a ``prepared.json`` and an identity that records its render campaign;
    it then belongs to every bank like an object with official PBR.
    """
    found = []
    for obj in recipe.generated_objects:
        directory = object_dir(root, obj)
        if not (directory / PREPARED_NAME).is_file() or not (directory / IDENTITY_NAME).is_file():
            continue
        if GENERATED_KEY in read_json(directory / IDENTITY_NAME):
            found.append(obj)
    return tuple(found)


def bank_objects(recipe: Recipe, root: Path) -> tuple[int, ...]:
    """Objects a full test bank needs a model of: official-PBR and prepared generated ones."""
    accepted = {*recipe.trainable_objects, *prepared_generated_objects(recipe, root)}
    return tuple(o for o in recipe.objects if o in accepted)


def require_resolving_links(dataset: Path, obj: int) -> None:
    """Require every image and mask link of a prepared split to resolve.

    Training would otherwise stop mid-epoch on the first broken link, for example after a
    dataset root prepared by an older release (absolute links) was moved.

    Raises:
        ValueError: If a link is broken; the message names the command that rebuilds it.
    """
    for split in SPLITS:
        for kind in ("images", "masks"):
            folder = Path(dataset) / split / kind
            if not folder.is_dir():
                continue
            for entry in folder.iterdir():
                if not entry.exists():
                    raise ValueError(
                        f"prepared link {entry} does not resolve (moved dataset root or removed "
                        f"source); rebuild it with `doper prepare --force --obj-id {obj}`"
                    )
