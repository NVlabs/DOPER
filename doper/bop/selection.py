# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation-selected BOP test banks: one completed endpoint per object, base by default.

The ranking: the candidate with the highest held-out validation single-pass AR wins, the base is
preferred on exact ties, and remaining ties go to method-name order. An object whose candidates have
no validation score keeps its base (``no_validation_score_base_fallback``).

The eligibility checks:

* every candidate endpoint finished, and each object has exactly one base and one endpoint per
  arm; arms skipped by design are omitted from the candidates, not passed as unfinished;
* the candidates are the recipe's arms (``recipe_endpoints``): the base and every adaptation
  method that applies to the object, under the source method names, plus the recipe's
  aliases (HANDAL object 8's ``static_iterative_round8`` and ``static_dynamic_low_label``,
  whose names decide exact ties); a method skipped below its label minimum
  (``adaptation_skipped.json``) is omitted;
* a score is finite, within [0, 100] and belongs to the candidate: same object, protocol,
  validation population, scored checkpoint and ordered geometry;
* every scored candidate shares the base's validation population, frame count, protocol,
  geometry and native camera code; every candidate shares the base geometry;
* optionally (HOT3D) one required protocol, every candidate scored, and one validation
  population, protocol and camera code across objects (``require_shared_population``).

Problems are reported per object as ``error`` selections, never dropped silently; candidates
present without a validation score are listed (``Selection.unscored``) rather than ranked;
``build_bank`` refuses a selection with any error. This module does not import the inference
stack (``ArResult`` is only a type here).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from .layout import CHECKPOINT_PATH, KEYPOINTS_PATH, SKIPPED_NAME, object_run
from .sources import sha256_file

if TYPE_CHECKING:
    from ..recipes.schema import Recipe
    from .ar import ArResult

FALLBACK_REASON = "no_validation_score_base_fallback"
SELECTED_REASON = "highest_held_out_val_single_AR_base_preferred_on_ties"
SELECTION_ARM = "single"
MAX_AR = 100.0
SHARED_FIELDS = (
    "validation_population_sha256",
    "validation_frames",
    "protocol",
    "geometry_sha256",
    "camera_code_sha256",
)
SelectionStatus = Literal["selected", "base_fallback", "error"]


@dataclass(frozen=True)
class Candidate:
    """One training endpoint of an object.

    Skipped-by-design arms are not candidates; pass only endpoints that exist, with
    ``endpoint_complete=False`` for one that started but did not finish.

    Attributes:
        obj_id: BOP object id.
        arm: Method/arm name (``base``, ``static_pseudo``, ...).
        endpoint_complete: Whether the fixed-budget endpoint finished.
        checkpoint_sha256: SHA-256 of ``work/last.pth``.
        geometry_sha256: SHA-256 of the ordered keypoint file.
        validation_population_sha256: Validation population the score was computed on.
        protocol: Validation protocol id (``doper.bop.ar.PROTOCOL_IDS``).
        result: Validation AR, or None when the endpoint was not scored.
        run_dir: The endpoint's run directory; None means ``<runs>/<arm>/obj_XXXXXX``.
    """

    obj_id: int
    arm: str
    endpoint_complete: bool
    checkpoint_sha256: str
    geometry_sha256: str
    validation_population_sha256: str | None
    protocol: str | None
    result: ArResult | None = None
    run_dir: Path | None = None

    @property
    def scored(self) -> bool:
        """Whether the candidate carries a scored validation result."""
        return self.result is not None and self.result.status == "scored"

    @property
    def single_ar(self) -> float | None:
        """Validation single-pass AR (percent), or None without a scored result."""
        if self.result is None or not self.scored:
            return None
        arm = self.result.arms.get(SELECTION_ARM)
        return None if arm is None else arm.ar_mean

    @property
    def validation_frames(self) -> int | None:
        """Frames of the single-pass validation score, or None without one."""
        if self.result is None or self.single_ar is None:
            return None
        return self.result.arms[SELECTION_ARM].frames

    @property
    def camera_code_sha256(self) -> str | None:
        """Native camera code hash of the score (None for pinhole or unscored)."""
        return None if self.result is None else self.result.camera_code_sha256


@dataclass(frozen=True)
class Selection:
    """The decision for one object.

    Attributes:
        obj_id: BOP object id.
        status: ``selected`` (the ranking chose ``chosen``, possibly the base),
            ``base_fallback`` (no candidate has a validation score) or ``error``.
        reason: The selection reason string, or the error.
        chosen: The selected candidate; None for an error.
        base: The base candidate, when exactly one exists.
        candidates: Every candidate considered for the object.
    """

    obj_id: int
    status: SelectionStatus
    reason: str
    chosen: Candidate | None
    base: Candidate | None
    candidates: tuple[Candidate, ...]

    @property
    def unscored(self) -> tuple[str, ...]:
        """Candidates present but without a validation score; they are not ranked."""
        return tuple(c.arm for c in self.candidates if c.single_ar is None)


@dataclass(frozen=True)
class Endpoint:
    """One existing selection candidate's run directory.

    Attributes:
        name: Candidate (arm) name: ``base``, a recipe method or an alias.
        obj_id: BOP object id.
        run_dir: The run directory (``<runs>/<run>/obj_XXXXXX``).
    """

    name: str
    obj_id: int
    run_dir: Path


def recipe_endpoints(
    recipe: Recipe, runs: Path, objects: Sequence[int], *, arms: Sequence[str] | None = None
) -> list[Endpoint]:
    """The existing candidate endpoints of a recipe, in object then table order.

    Every recipe arm that applies to the object (``MethodSpec.objects``) and has a run
    directory is a candidate, then every alias of the object. A run directory claimed by an
    alias is a candidate only under the alias name. A method skipped below its label minimum
    (``adaptation_skipped.json``) is omitted; an unfinished one stays, so ``choose`` refuses it.

    Args:
        recipe: Dataset recipe.
        runs: Runs root.
        objects: Objects to list.
        arms: Arms to consider (default: ``base`` and every recipe method, plus the aliases);
            given, only these arms and no aliases.

    Returns:
        The endpoints.
    """
    from ..recipes.schema import BASE_ARM, recipe_arms

    methods = recipe.adapt.methods if recipe.adapt is not None else {}
    names = tuple(arms) if arms is not None else recipe_arms(recipe)
    aliases = recipe.test.aliases if arms is None else {}
    endpoints = []
    for obj in objects:
        claimed = {object_run(runs, a.run, obj) for a in aliases.values() if obj in a.objects}
        listed = [
            Endpoint(name, obj, object_run(runs, name, obj))
            for name in names
            if name == BASE_ARM or (name in methods and methods[name].applies_to(obj))
        ]
        listed = [e for e in listed if e.run_dir not in claimed]
        listed += [
            Endpoint(name, obj, object_run(runs, alias.run, obj))
            for name, alias in aliases.items()
            if obj in alias.objects
        ]
        endpoints += [
            e for e in listed if e.run_dir.is_dir() and not (e.run_dir / SKIPPED_NAME).is_file()
        ]
    return endpoints


@dataclass(frozen=True)
class _Policy:
    fallback: str
    protocol: str | None
    require_scored: bool


def _score_problem(candidate: Candidate, policy: _Policy) -> str | None:
    if policy.protocol is not None and candidate.protocol != policy.protocol:
        return f"candidate {candidate.arm} is not scored under {policy.protocol}"
    if policy.require_scored and candidate.single_ar is None:
        return f"unrankable candidate {candidate.arm}: no scored validation result"
    score = candidate.single_ar
    if score is not None and (not math.isfinite(score) or not 0 <= score <= MAX_AR):
        return f"invalid validation score for {candidate.arm}"
    result = candidate.result
    if result is None:
        return None
    lineage = (
        (result.obj_id, candidate.obj_id),
        (result.protocol, candidate.protocol),
        (result.population_sha256, candidate.validation_population_sha256),
        (result.checkpoint_sha256, candidate.checkpoint_sha256),
        (result.geometry_sha256, candidate.geometry_sha256),
    )
    if any(scored != expected for scored, expected in lineage):
        return f"score/candidate lineage differs for {candidate.arm}"
    if result.camera == "native" and result.camera_code_sha256 is None:
        return f"native score without camera code for {candidate.arm}"
    return None


def _eligibility_problem(candidate: Candidate, base: Candidate) -> str | None:
    message = "candidate/base validation population, protocol, camera or geometry differs"
    if candidate.geometry_sha256 != base.geometry_sha256:
        return f"{message}: {candidate.arm} geometry_sha256"
    if candidate.single_ar is None:
        return None
    for name in SHARED_FIELDS:
        value, expected = getattr(candidate, name), getattr(base, name)
        if value != expected or (value is None and name != "camera_code_sha256"):
            return f"{message}: {candidate.arm} {name}"
    return None


def _problem(candidates: Sequence[Candidate], policy: _Policy) -> str | None:
    if not candidates:
        return "missing object: no candidate endpoint"
    duplicates = sorted(a for a, n in Counter(c.arm for c in candidates).items() if n > 1)
    if duplicates:
        return f"exactly one {policy.fallback} and one endpoint per arm required: {duplicates}"
    bases = [c for c in candidates if c.arm == policy.fallback]
    if len(bases) != 1:
        return f"exactly one {policy.fallback} required"
    unfinished = sorted(c.arm for c in candidates if not c.endpoint_complete)
    if unfinished:
        return f"unfinished endpoint: {unfinished}"
    for candidate in candidates:
        problem = _score_problem(candidate, policy) or _eligibility_problem(candidate, bases[0])
        if problem:
            return problem
    return None


def _choose_object(obj: int, candidates: Sequence[Candidate], policy: _Policy) -> Selection:
    bases = [c for c in candidates if c.arm == policy.fallback]
    base = bases[0] if len(bases) == 1 else None
    problem = _problem(candidates, policy)
    if problem is not None or base is None:
        return Selection(obj, "error", problem or "no base", None, base, tuple(candidates))
    eligible = [c for c in candidates if c.single_ar is not None]
    if not eligible:
        return Selection(obj, "base_fallback", FALLBACK_REASON, base, base, tuple(candidates))
    best = sorted(eligible, key=lambda c: (-(c.single_ar or 0.0), c.arm != policy.fallback, c.arm))[
        0
    ]
    return Selection(obj, "selected", SELECTED_REASON, best, base, tuple(candidates))


def choose(
    candidates: Sequence[Candidate],
    objects: Sequence[int],
    *,
    fallback: str = "base",
    protocol: str | None = None,
    require_scored: bool = False,
) -> dict[int, Selection]:
    """Select one endpoint per object by validation single-pass AR.

    Args:
        candidates: Existing endpoints of every arm for the objects (omit arms skipped by
            design; pass unfinished ones with ``endpoint_complete=False``).
        objects: The bank's objects; each gets a ``Selection``, a missing one an ``error``.
        fallback: Arm used without a validation score and preferred on exact ties.
        protocol: Required validation protocol id for every candidate (HOT3D: the native
            protocol); None only requires agreement with the base.
        require_scored: Require every candidate to carry a scored result (HOT3D rule); else an
            unscored candidate is simply not ranked.

    Returns:
        Object id to ``Selection``, for every object in ``objects``.

    Raises:
        ValueError: If a candidate's object is not in ``objects``.
    """
    foreign = sorted({c.obj_id for c in candidates} - set(objects))
    if foreign:
        raise ValueError(f"candidate objects {foreign} are not in the bank objects")
    policy = _Policy(fallback, protocol, require_scored)
    return {
        obj: _choose_object(obj, [c for c in candidates if c.obj_id == obj], policy)
        for obj in objects
    }


def build_bank(selection: Mapping[int, Selection], runs_root: Path) -> dict[int, Path]:
    """Resolve a selection to run directories, ``<runs_root>/<arm>/obj_XXXXXX``.

    Args:
        selection: Output of ``choose``.
        runs_root: Directory holding one subdirectory per arm.

    Returns:
        Object id to run directory, for ``doper.bop.export.run_bank``.

    Raises:
        ValueError: If any object has an error selection, or a run's checkpoint or keypoint
            file does not match the selected candidate's hashes.
    """
    errors = [f"object {s.obj_id}: {s.reason}" for s in selection.values() if s.status == "error"]
    if errors:
        raise ValueError("selection has errors: " + "; ".join(errors))
    bank = {}
    for obj, choice in sorted(selection.items()):
        if choice.chosen is None:
            raise ValueError(f"object {obj}: no chosen endpoint")
        chosen = choice.chosen
        run = (
            chosen.run_dir if chosen.run_dir is not None else object_run(runs_root, chosen.arm, obj)
        )
        if sha256_file(run / CHECKPOINT_PATH) != choice.chosen.checkpoint_sha256:
            raise ValueError(f"object {obj}: {run / CHECKPOINT_PATH} checkpoint hash differs")
        if sha256_file(run / KEYPOINTS_PATH) != choice.chosen.geometry_sha256:
            raise ValueError(f"object {obj}: {run / KEYPOINTS_PATH} geometry hash differs")
        bank[obj] = run
    return bank


def require_shared_population(selection: Mapping[int, Selection]) -> None:
    """Require every object's base to share one validation population, protocol and camera.

    ``select_hot3d_test_models`` applies this across objects (one frozen validation panel and
    camera code); per-object validation populations (HANDAL, HOPEv2) skip it.

    Raises:
        ValueError: If the bases differ, or an object has no base.
    """
    bases = [s.base for s in selection.values()]
    if any(b is None for b in bases):
        raise ValueError("every object needs a base to compare validation populations")
    identities = {
        (b.validation_population_sha256, b.protocol, b.camera_code_sha256)
        for b in bases
        if b is not None
    }
    if len(identities) != 1:
        raise ValueError(
            "objects do not share one frozen validation population, protocol and camera code"
        )
