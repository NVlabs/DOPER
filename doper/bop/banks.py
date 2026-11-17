# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-policy BOP test banks: one method's final endpoint per object, never score-ranked.

Every adaptation method is submitted as its own full-population bank. Each
object contributes the method's completed student; where the policy allows it, the object's
original base stands in, and every such fallback is disclosed:

* ``skipped``: the method was skipped below its label minimum (``adaptation_skipped.json``)
  and no student exists (HANDAL static and test TTA, HOPE static, HOT3D pseudo methods);
* ``out_of_scope``: a single-object follow-up policy keeps the base for every other object.

A fallback is the base run itself, whose checkpoint must still match its ``completed.json``,
and a skip marker must not hide a student. A student must be finished, belong to the object,
descend from the same base (``base_teacher_sha256``) and keep the base geometry. Any other
missing or unfinished endpoint is an error, never a silent base. ``bank_manifest`` writes the
bank's ``MODEL_MATRIX.csv`` and ``SKIPS.json``. This module does not import the inference stack.
"""

from __future__ import annotations

import csv
import io
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..recipes.schema import BASE_ARM, BUILTIN_POLICIES, Policy, Recipe
from .layout import CHECKPOINT_PATH, COMPLETED_NAME, KEYPOINTS_PATH, SKIPPED_NAME, object_run
from .sources import publish_json, publish_text, sha256_file

Bank = dict[int, Path]
FallbackReason = Literal["skipped", "out_of_scope"]
MATRIX_NAME = "MODEL_MATRIX.csv"
SKIPS_NAME = "SKIPS.json"
MATRIX_COLUMNS: tuple[str, ...] = (
    "policy",
    "object",
    "requested_method",
    "actual_method",
    "status",
    "labels",
    "checkpoint_sha256",
    "base_checkpoint_sha256",
    "keypoints_sha256",
    "run",
    "fallback_reason",
)
FALLBACK_TEXT = {
    "skipped": "method skipped below its label minimum; no student exists",
    "out_of_scope": "outside the single-object policy's scope",
}


@dataclass(frozen=True)
class Fallback:
    """One object whose base stands in for the policy's method.

    Attributes:
        obj_id: BOP object id.
        method: The policy's method (its run directory).
        reason: ``skipped`` or ``out_of_scope``.
        labels: Accepted labels of a skipped method; None out of scope.
        checkpoint_sha256: SHA-256 of the base checkpoint that stands in.
    """

    obj_id: int
    method: str
    reason: FallbackReason
    labels: int | None
    checkpoint_sha256: str


@dataclass(frozen=True)
class _Base:
    run: Path
    checkpoint_sha256: str
    keypoints_sha256: str


def base_policy(recipe: Recipe) -> Policy:
    """The built-in ``base`` policy as a fixed policy: every object's base model."""
    return Policy(BASE_ARM, BASE_ARM, f"{recipe.test.method}{BASE_ARM}", "none", None)


def recipe_policy(recipe: Recipe, name: str) -> Policy:
    """Return a recipe's fixed policy by name.

    Raises:
        ValueError: For a name that is neither built in nor in ``test.policies``.
    """
    if name not in recipe.test.policies:
        known = [*BUILTIN_POLICIES, *recipe.test.policies]
        raise ValueError(f"{name!r} is not a {recipe.name} test policy: {known}")
    return recipe.test.policies[name]


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def _base(runs: Path, obj: int) -> _Base:
    run = object_run(runs, BASE_ARM, obj)
    done = _read(run / COMPLETED_NAME)
    checkpoint, keypoints = run / CHECKPOINT_PATH, run / KEYPOINTS_PATH
    if done.get("obj_id") != obj or not checkpoint.is_file() or not keypoints.is_file():
        raise ValueError(f"object {obj}: no completed base model in {run}")
    current = (sha256_file(checkpoint), sha256_file(keypoints))
    if current != (done.get("checkpoint_sha256"), done.get("keypoints_sha256")):
        raise ValueError(f"object {obj}: the base checkpoint or keypoints differ from {run}")
    return _Base(run, *current)


def _skip_labels(run: Path, obj: int) -> int:
    if (run / CHECKPOINT_PATH).exists() or (run / COMPLETED_NAME).exists():
        raise ValueError(f"object {obj}: the skip marker in {run} masks a student")
    marker = _read(run / SKIPPED_NAME)
    labels, minimum = marker.get("labels"), marker.get("minimum_labels")
    if not isinstance(labels, int) or not isinstance(minimum, int) or not 0 <= labels < minimum:
        raise ValueError(f"object {obj}: {run / SKIPPED_NAME} is not below its label minimum")
    return labels


def _require_student(run: Path, obj: int, base: _Base, method: str) -> None:
    if not run.is_dir():
        raise ValueError(f"object {obj}: no {method} endpoint ({run})")
    done = _read(run / COMPLETED_NAME)
    checkpoint, keypoints = run / CHECKPOINT_PATH, run / KEYPOINTS_PATH
    if not done or not checkpoint.is_file():
        raise ValueError(f"object {obj}: unfinished {method} endpoint ({run})")
    if done.get("obj_id") != obj or sha256_file(checkpoint) != done.get("checkpoint_sha256"):
        raise ValueError(f"object {obj}: the {method} checkpoint differs from its receipt")
    if done.get("base_teacher_sha256") != base.checkpoint_sha256:
        raise ValueError(f"object {obj}: the {method} student descends from another base")
    current = sha256_file(keypoints) if keypoints.is_file() else None
    if current != base.keypoints_sha256 or done.get("keypoints_sha256") != current:
        raise ValueError(f"object {obj}: the {method} geometry differs from the base")


def _allow(policy: Policy, reason: FallbackReason, obj: int) -> None:
    if policy.fallback == "none":
        raise ValueError(
            f"policy {policy.name} allows no fallback, but object {obj} is {reason} for "
            f"{policy.method}"
        )
    if policy.fallback != reason:
        raise ValueError(
            f"policy {policy.name} allows only {policy.fallback} fallbacks; object {obj} is "
            f"{reason} for {policy.method}"
        )


def fixed_bank(policy: Policy, runs: Path, objects: Sequence[int]) -> tuple[Bank, list[Fallback]]:
    """Resolve a fixed policy to one run directory per object.

    Args:
        policy: The policy (``base_policy`` or a recipe policy).
        runs: Runs root (``<runs>/<method>/obj_XXXXXX``).
        objects: Objects of the bank; each needs a completed base.

    Returns:
        Object id to run directory (``doper.bop.export.run_bank``), and the base fallbacks in
        object order.

    Raises:
        ValueError: For a missing or changed base, a fallback the policy does not allow, a skip
            marker that is not below its minimum or hides a student, or a missing, unfinished
            or foreign student.
    """
    bank: Bank = {}
    fallbacks: list[Fallback] = []
    for obj in objects:
        base = _base(runs, obj)
        reason: FallbackReason | None = None
        labels = None
        run = object_run(runs, policy.method, obj)
        if policy.method == BASE_ARM:
            bank[obj] = base.run
            continue
        if policy.objects is not None and obj not in policy.objects:
            reason = "out_of_scope"
        elif (run / SKIPPED_NAME).is_file():
            reason, labels = "skipped", _skip_labels(run, obj)
        if reason is not None:
            _allow(policy, reason, obj)
            fallbacks.append(Fallback(obj, policy.method, reason, labels, base.checkpoint_sha256))
            bank[obj] = base.run
            continue
        _require_student(run, obj, base, policy.method)
        bank[obj] = run
    return bank, fallbacks


def _matrix_row(
    policy: Policy, obj: int, run: Path, fallback: Fallback | None, runs: Path
) -> dict[str, Any]:
    done = _read(run / COMPLETED_NAME)
    student = fallback is None and policy.method != BASE_ARM
    if student:
        status = "adapted_student"
    else:
        status = "base" if fallback is None else f"base_fallback_{fallback.reason}"
    labels = done.get("labels") if student else (fallback.labels if fallback else None)
    return {
        "policy": policy.name,
        "object": obj,
        "requested_method": policy.method,
        "actual_method": policy.method if student else BASE_ARM,
        "status": status,
        "labels": "" if labels is None else labels,
        "checkpoint_sha256": done["checkpoint_sha256"],
        "base_checkpoint_sha256": done["base_teacher_sha256"]
        if student
        else done["checkpoint_sha256"],
        "keypoints_sha256": done["keypoints_sha256"],
        "run": Path(os.path.relpath(run, runs)).as_posix(),
        "fallback_reason": FALLBACK_TEXT[fallback.reason] if fallback else "",
    }


def bank_manifest(
    directory: Path,
    policy: Policy,
    bank: Mapping[int, Path],
    fallbacks: Sequence[Fallback],
    runs: Path,
) -> tuple[Path, Path]:
    """Write a fixed bank's ``MODEL_MATRIX.csv`` (one row per object) and ``SKIPS.json``.

    Args:
        directory: Output directory (beside the bank's CSV).
        policy: The bank's policy.
        bank: Output of ``fixed_bank``.
        fallbacks: Output of ``fixed_bank``.
        runs: Runs root; matrix ``run`` paths are relative to it.

    Returns:
        The matrix and skips paths.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    by_object = {f.obj_id: f for f in fallbacks}
    matrix = directory / MATRIX_NAME
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(MATRIX_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for obj, run in sorted(bank.items()):
        writer.writerow(_matrix_row(policy, obj, Path(run), by_object.get(obj), Path(runs)))
    publish_text(matrix, stream.getvalue())
    skips = directory / SKIPS_NAME
    cases = [
        {
            "object_id": f.obj_id,
            "method": f.method,
            "reason": f.reason,
            "labels": f.labels,
            "base_checkpoint_sha256": f.checkpoint_sha256,
            "submission_policy": "explicit original base fallback",
        }
        for f in fallbacks
    ]
    record = {
        "policy": policy.name,
        "method": policy.method,
        "fallback": policy.fallback,
        "cases": cases,
    }
    publish_json(skips, record)
    return matrix, skips
