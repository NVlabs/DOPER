# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run-directory layout shared by BOP training outputs, selection and test export.

Every arm (``base`` or an adaptation method) keeps one directory per object::

    <runs>/<arm>/obj_XXXXXX/{recipe.yaml, geometry/keypoints_3d.json, work/last.pth,
                             completed.json, validation/ar.json}

An adaptation method adds ``identity.json``, its label pool ``labels/`` with
``labels_ready.json``, and ``adaptation_skipped.json`` when the pool is below the minimum. The
paired GT pools of an object are shared by both GT methods: ``<runs>/gt_pools/obj_XXXXXX``.

A recipe follow-up (``Recipe.followups``, a single-object self-training series) keeps every
round as an adaptation run of its own, never in a method's directory::

    <runs>/<followup>/{identity.json, progress.json}
    <runs>/<followup>/roundNN/obj_XXXXXX/     # one adaptation run per round (01, 02, ...)
    <runs>/<followup>/final_probe/obj_XXXXXX/ # labels-only yield probe of the final student
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, get_args

CHECKPOINT_PATH = Path("work") / "last.pth"
KEYPOINTS_PATH = Path("geometry") / "keypoints_3d.json"
VALIDATION_PATH = Path("validation") / "ar.json"
COMPLETED_NAME = "completed.json"
RECIPE_NAME = "recipe.yaml"
IDENTITY_NAME = "identity.json"
LABELS_DIR = "labels"
LABELS_READY_NAME = "labels_ready.json"
SKIPPED_NAME = "adaptation_skipped.json"
GT_POOLS_DIR = "gt_pools"
PROGRESS_NAME = "progress.json"
FINAL_PROBE_DIR = "final_probe"
# Directories of the runs root that are not arms (follow-up names must avoid them).
TEST_DIR = "test"
RESERVED_RUN_DIRS = ("base", GT_POOLS_DIR, TEST_DIR)
# The Table 3 test variants (``doper.bop.variants``), each exported to
# ``<runs>/test/<policy>/<variant>/``.
Variant = Literal["singleview", "singleheatmap", "multiview", "multiheatmap"]
VARIANTS: tuple[Variant, ...] = get_args(Variant)
MULTIVIEW_VARIANTS: tuple[Variant, ...] = ("multiview", "multiheatmap")
HEATMAP_VARIANTS: tuple[Variant, ...] = ("singleheatmap", "multiheatmap")


def object_run(runs_root: Path, arm: str, obj_id: int) -> Path:
    """Return ``<runs_root>/<arm>/obj_XXXXXX``, one object's run directory of an arm."""
    return Path(runs_root) / arm / f"obj_{obj_id:06d}"


def gt_pool(runs_root: Path, obj_id: int) -> Path:
    """Return ``<runs_root>/gt_pools/obj_XXXXXX``, the paired GT pools of one object."""
    return Path(runs_root) / GT_POOLS_DIR / f"obj_{obj_id:06d}"


def round_name(number: int) -> str:
    """Return a follow-up round directory name, ``roundNN``."""
    return f"round{number:02d}"


def followup_round(runs_root: Path, name: str, number: int, obj_id: int) -> Path:
    """Return ``<runs_root>/<name>/roundNN/obj_XXXXXX``, one follow-up round of one object."""
    return object_run(Path(runs_root) / name, round_name(number), obj_id)
