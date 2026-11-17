# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SPEED+ recipes of ``doper prepare``.

A SPEED+ recipe (``kind: speedplus``) runs one ``--stage`` at a time (``doper.speedplus.data``):
``fetch`` downloads and verifies SPEED+ v2 and the official Tango landmarks once the user
accepts the CC BY-NC-SA 4.0 license and the 16.9 GB download with ``--accept-license``;
``convert`` writes the 64-point COCO layout to ``<root>/derived/tango_64`` with the recipe's
``mask_mode`` (the paper's projected hulls), refusing other masks without
``--accept-non-paper-masks``. ``extract-landmarks`` runs the tracked-landmark extractor
(``doper.speedplus.keypoints``, the ``doper[speedplus]`` extra) on the synthetic train images and
the conversion's masks into ``<root>/derived/tracked_landmarks``; ``layout`` builds the 75-point
layout (``<root>/derived/tango_75``, train and validation by default) from the tracked-64 file
given with ``--tracked-keypoints``. It accepts only the recipe's paper layout (by its points
hash) unless ``--accept-new-layout`` is given, which marks the build ``paper_layout: false``.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

from ...bop import sources
from ...console import log
from ...recipes import load_any_recipe
from ...recipes.speedplus_schema import SpeedplusRecipe
from ...speedplus.definitions import (
    DERIVED_64,
    DERIVED_75,
    LAYOUT_SPLITS,
    PAPER_MASK_MODE,
    SPLIT_NAMES,
    TRACKED_DIR,
)
from ..common import (
    MissingExtraError,
    Step,
)
from ._names import BOP_FLAGS, NUREC_FLAGS, SPEEDPLUS_STAGES, refuse_flags

if TYPE_CHECKING:
    from ...speedplus.data import SpeedplusProfile


def _speedplus_fetch(root: Path, profile: SpeedplusProfile, consent: bool) -> None:
    from ...speedplus import data

    receipt = data.fetch(root, consent=consent, profile=profile)
    log(f"SPEED+ ready: {dict(receipt.image_counts)} images, {receipt.checksum_origin} "
        f"sha256 {receipt.archive_sha256}")  # fmt: skip


def _speedplus_convert(root: Path, out: Path, args: argparse.Namespace, mask_mode: str) -> None:
    from ...speedplus import data

    manifest = data.convert(
        root,
        out,
        mask_mode=mask_mode,
        splits=tuple(args.splits or SPLIT_NAMES),
        masks_root=args.masks_root,
        accept_non_paper_masks=args.accept_non_paper_masks,
        force=args.force,
    )
    log(f"SPEED+ 64-point layout: masks {dict(manifest.mask_sources)}, "
        f"paper masks: {manifest.paper_masks}")  # fmt: skip


def _speedplus_landmarks(root: Path, mask_root: Path, force: bool) -> None:
    from ...speedplus import data, keypoints

    if importlib.util.find_spec("sklearn") is None:
        raise MissingExtraError(
            "the tracked-landmark extractor needs scikit-learn: install doper[speedplus]"
        )
    ready = data.require_ready(root)
    written = keypoints.extract_tracked_landmarks(
        root / ready["extracted"], mask_root, root / TRACKED_DIR, force=force
    )
    log(f"SPEED+ tracked landmarks: {written}")


def _speedplus_layout(root: Path, args: argparse.Namespace, recipe: SpeedplusRecipe) -> None:
    from ...speedplus import keypoints

    manifest = keypoints.build_layout(
        root / DERIVED_64,
        root / DERIVED_75,
        args.tracked_keypoints,
        splits=tuple(args.splits or LAYOUT_SPLITS),
        accept_new_layout=args.accept_new_layout,
        pins=recipe.layout,
        force=args.force,
    )
    log(f"SPEED+ 75-point layout: sha256 {manifest.layout_sha256}, points "
        f"{manifest.points_sha256}, paper layout: {manifest.paper_layout}")  # fmt: skip
    if manifest.paper_layout and not manifest.layout_file_matches_paper:
        log(f"formatting deviation: the layout file SHA-256 is not the paper's "
            f"{recipe.layout.sha256}, although its points are the paper's")  # fmt: skip


def _speedplus_fetch_step(root: Path, recipe: SpeedplusRecipe, args: argparse.Namespace) -> Step:
    from ...speedplus import data

    profile = data.load_profile(recipe.profile)
    inputs = {"archive": profile.archive["url"], "md5": profile.archive["md5"],
              "tango_points": profile.tango_points["url"],
              "license": profile.license["name"]}  # fmt: skip
    outputs = {"archive": root / data.ARCHIVE_PATH, "data": root / data.SPEEDPLUS_DIR,
               "receipt": root / sources.READY_NAME}  # fmt: skip
    action = lambda: _speedplus_fetch(root, profile, args.accept_license)  # noqa: E731
    return Step("fetch and verify SPEED+", inputs, outputs, action)


def _speedplus_convert_step(root: Path, recipe: SpeedplusRecipe, args: argparse.Namespace) -> Step:
    from ...speedplus import data

    mask_mode = args.mask_mode or recipe.mask_mode
    if mask_mode not in (PAPER_MASK_MODE, "auto") and not args.accept_non_paper_masks:
        raise ValueError(
            f"mask mode {mask_mode} gives non-paper SPEED+ masks (the paper uses projected "
            "hulls); pass --accept-non-paper-masks to accept them"
        )
    out = root / DERIVED_64
    inputs = {"data": root / data.SPEEDPLUS_DIR, "receipt": root / sources.READY_NAME,
              "mask_mode": mask_mode, "splits": list(args.splits or SPLIT_NAMES)}  # fmt: skip
    outputs = {"layout": out, "manifest": out / data.MANIFEST_NAME}
    action = lambda: _speedplus_convert(root, out, args, mask_mode)  # noqa: E731
    return Step("convert SPEED+ to the 64-point layout", inputs, outputs, action)


def _speedplus_landmarks_step(
    root: Path, recipe: SpeedplusRecipe, args: argparse.Namespace
) -> Step:
    from ...speedplus import data

    mask_root = args.landmark_mask_root or root / DERIVED_64 / "train" / "masks"
    inputs = {"data": root / data.SPEEDPLUS_DIR, "masks": mask_root,
              "validation_masks": mask_root.parent.parent / "val" / "masks"}  # fmt: skip
    outputs = {"landmarks": root / TRACKED_DIR / data.KEYPOINTS_NAME}
    action = lambda: _speedplus_landmarks(root, mask_root, args.force)  # noqa: E731
    return Step("extract the SPEED+ tracked landmarks", inputs, outputs, action)


def _speedplus_layout_step(root: Path, recipe: SpeedplusRecipe, args: argparse.Namespace) -> Step:
    if args.tracked_keypoints is None:
        raise ValueError(
            "--stage layout needs --tracked-keypoints FILE: the paper's tracked-64 "
            f"keypoints_3d.json, or the extract-landmarks output <root>/{TRACKED_DIR}"
            "/keypoints_3d.json (with --accept-new-layout)"
        )
    inputs = {"tango_64": root / DERIVED_64, "tracked": args.tracked_keypoints,
              "splits": list(args.splits or LAYOUT_SPLITS),
              "paper points sha256": recipe.layout.points_sha256}  # fmt: skip
    outputs = {"layout": root / DERIVED_75 / "keypoints_3d.json"}
    action = lambda: _speedplus_layout(root, args, recipe)  # noqa: E731
    return Step("build the SPEED+ 75-point layout", inputs, outputs, action)


SPEEDPLUS_STEPS = {
    "fetch": _speedplus_fetch_step,
    "convert": _speedplus_convert_step,
    "extract-landmarks": _speedplus_landmarks_step,
    "layout": _speedplus_layout_step,
}


def speedplus_plan(args: argparse.Namespace) -> list[Step]:
    """Build the ``prepare`` plan of a SPEED+ recipe: one ``--stage``.

    Raises:
        ValueError: For a BOP flag, a missing stage, a non-paper mask mode without
            ``--accept-non-paper-masks``, or a layout stage without ``--tracked-keypoints``.
    """
    recipe = load_any_recipe(args.recipe)
    if not isinstance(recipe, SpeedplusRecipe):
        raise ValueError(f"{recipe.name} is not a SPEED+ recipe")
    refuse_flags(args, BOP_FLAGS, "BOP")
    refuse_flags(args, NUREC_FLAGS, "NuRec")
    if args.stage not in SPEEDPLUS_STAGES:
        raise ValueError(f"`doper prepare {recipe.name}` needs --stage, one of {SPEEDPLUS_STAGES}")
    return [SPEEDPLUS_STEPS[args.stage](Path(args.root), recipe, args)]
