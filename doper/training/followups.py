# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-object self-training follow-ups (HANDAL Table 89): rounds, stop rule, yield probe.

A recipe follow-up (``Recipe.followups``, ``FollowupSpec``) repeats the adaptation engine of
``doper.training.adaptation`` on one object's static onboarding view, each round in its own
run directory ``<runs>/<name>/roundNN/obj_XXXXXX`` (never a method's directory):

1. **Round 1** labels with the follow-up's teacher (the epoch-25 base by default), at the strict
   recipe gate unless ``gate_overrides`` relax it (object 7's bootstrap: reprojection < 5 px).
   With ``reuse_labels_from`` it trains on a copy of that method's verified pool instead
   (object 8: the original six-label ``static_pseudo`` pool, below that method's minimum of 20).
   With ``require_review`` it trains only after a human review of every pool overlay
   (``doper review``, ``REVIEWED.json`` via ``doper.bop.review``) or an explicit
   ``accept_unreviewed``; ``completed.json`` and ``progress.json`` record which.
2. **Each later round** regenerates the whole pool at the strict gate with the previous round's
   student as teacher and initialization, then trains three epochs (minimum ``minimum_labels``).
3. **Stop** (``stop_reason``): no accepted labels, a strict-gate pool of at least
   ``target_labels`` (a relaxed round 1 never counts), or the round cap. A labels-only probe of
   the final student at the strict gate measures its yield without training another round.

``progress.json`` holds every round's labels, teacher, student, label churn (``label_churn``)
and review status. Re-running replays every round idempotently (each run re-verifies its
receipts), so an interrupted series resumes where it stopped. ``mask_quality_audit`` is the
optional post-training diagnostic against provided GT masks; it never filters, supervises,
selects or stops anything.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import zipfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from ..bop import layout, review
from ..bop.sources import json_roundtrip, publish_json, sha256_file
from ..console import log
from ..pseudo_labels.gate import GateSpec
from ..recipes.schema import BASE_ARM, FollowupSpec, Recipe
from . import adaptation

StopReason = Literal[
    "no_positive_label_pool",
    "below_minimum_label_pool",
    "target_training_pool_reached",
    "round_cap_reached",
]
REVIEW_DIR = "review"
REVIEW_COLORS = {"mask": (0, 255, 0), "box": (0, 200, 255)}
REVIEW_LINE_PX = 2
MASK_THRESHOLD = 127
GOOD_MASK_IOU = 0.8
GT_MASK_MEMBER = "static/{scene}/mask_visib/{frame:06d}_000000.png"


# --------------------------------------------------------------------------------------
# Stop rule, churn and the post-training diagnostic (source handal_obj8_loop)
# --------------------------------------------------------------------------------------
def stop_reason(labels: int, round_number: int, target: float, max_rounds: int) -> str | None:
    """Why a self-training series stops after a round, or None to continue.

    Args:
        labels: Accepted labels of the round's pool.
        round_number: The round (1-based).
        target: Labels that end the series (``math.inf`` for a round that cannot reach it).
        max_rounds: The round cap.

    Returns:
        ``no_positive_label_pool``, ``target_training_pool_reached``, ``round_cap_reached`` or
        None, checked in that order.
    """
    if labels == 0:
        return "no_positive_label_pool"
    if labels >= target:
        return "target_training_pool_reached"
    if round_number >= max_rounds:
        return "round_cap_reached"
    return None


def label_churn(previous: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    """Compare two pool manifests by source frame, not by net count.

    Args:
        previous: The previous round's ``manifest.json`` (``{"records": []}`` for none).
        current: This round's ``manifest.json``.

    Returns:
        ``retained`` (count), sorted ``added`` and ``dropped`` source frames, and this pool's
        labels per sequence (``per_sequence``).
    """
    before = {r["source_relative_path"] for r in previous["records"]}
    after = {r["source_relative_path"] for r in current["records"]}
    return {
        "retained": len(before & after),
        "added": sorted(after - before),
        "dropped": sorted(before - after),
        "per_sequence": dict(Counter(r["scene"] for r in current["records"])),
    }


def mask_quality_audit(labels_root: Path, archive: Path) -> dict[str, Any]:
    """Post-training diagnostic: IoU of each pool mask with the provided visible GT mask.

    The GT masks are read from the official onboarding archive
    (``static/<scene>/mask_visib/<frame>_000000.png``). They are never used to generate,
    filter, supervise, select or stop anything; the pool is only read.

    Args:
        labels_root: A pool directory (``manifest.json``, ``train/annotations.json``).
        archive: The official static onboarding zip.

    Returns:
        The protocol note, mean and minimum mask IoU, the count with IoU >= 0.8, the number of
        GT masks found and one row per label.

    Raises:
        ValueError: If a GT mask and its pool mask differ in shape.
    """
    import cv2
    import numpy as np

    labels_root = Path(labels_root)
    manifest = json.loads((labels_root / "manifest.json").read_text())
    coco = json.loads((labels_root / "train" / "annotations.json").read_text())
    annotations = {a["image_id"]: a for a in coco["annotations"]}
    images = {im["file_name"]: im for im in coco["images"]}
    rows = []
    with zipfile.ZipFile(archive) as z:
        names = set(z.namelist())
        for r in manifest["records"]:
            name = GT_MASK_MEMBER.format(scene=r["scene"], frame=r["frame"])
            row: dict[str, Any] = {
                "source_relative_path": r["source_relative_path"],
                "score": r["score"],
                "reprojection_px": r["reproj_px"],
                "inliers": r["inliers"],
                "gt_available": name in names,
            }
            if name in names:
                truth = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), 0)
                annotation = annotations[images[r["image"]]["id"]]
                mask = cv2.imread(str(labels_root / "train" / annotation["mask_file"]), 0)
                if truth is None or mask is None:
                    raise ValueError(f"quality audit: cannot decode the masks of {name}")
                gt, pred = truth > MASK_THRESHOLD, mask > MASK_THRESHOLD
                if gt.shape != pred.shape:
                    raise ValueError("quality audit: mask shape mismatch")
                row["mask_iou"] = float((gt & pred).sum() / max(1, (gt | pred).sum()))
            rows.append(row)
    ious = [r["mask_iou"] for r in rows if r["gt_available"]]
    return {
        "protocol": "post-training diagnostic only; no GT filtering/supervision/selection",
        "mean_mask_iou": sum(ious) / len(ious) if ious else None,
        "min_mask_iou": min(ious) if ious else None,
        "mask_iou_ge_08": sum(v >= GOOD_MASK_IOU for v in ious),
        "gt_masks": len(ious),
        "rows": rows,
    }


# --------------------------------------------------------------------------------------
# Layout and gates
# --------------------------------------------------------------------------------------
def followup_spec(recipe: Recipe, name: str) -> FollowupSpec:
    """Return a recipe's follow-up.

    Raises:
        ValueError: If the recipe has no such follow-up.
    """
    if name not in recipe.followups:
        raise ValueError(
            f"{name!r} is not a {recipe.name} follow-up; follow-ups: {list(recipe.followups)}"
        )
    return recipe.followups[name]


def round_dir(runs: Path, name: str, number: int, obj_id: int) -> Path:
    """Return ``<runs>/<name>/roundNN/obj_XXXXXX``."""
    return layout.followup_round(runs, name, number, obj_id)


def probe_dir(runs: Path, name: str, obj_id: int) -> Path:
    """Return ``<runs>/<name>/final_probe/obj_XXXXXX``, the final student's yield probe."""
    return layout.object_run(Path(runs) / name, layout.FINAL_PROBE_DIR, obj_id)


def round_gate(recipe: Recipe, spec: FollowupSpec, number: int) -> GateSpec:
    """The gate of round ``number``: ``gate_overrides`` apply to round 1 only."""
    assert recipe.adapt is not None
    strict = recipe.adapt.gate
    return replace(strict, **spec.gate_overrides) if number == 1 else strict


def _relaxed(recipe: Recipe, spec: FollowupSpec, number: int) -> bool:
    assert recipe.adapt is not None
    return round_gate(recipe, spec, number) != recipe.adapt.gate


# --------------------------------------------------------------------------------------
# Review overlays
# --------------------------------------------------------------------------------------
def _pool(run_dir: Path) -> Path:
    ready = Path(run_dir) / layout.LABELS_READY_NAME
    if not ready.is_file():
        raise ValueError(
            f"no verified label pool in {run_dir}; run `doper train --stage followup` first"
        )
    return (Path(run_dir) / json.loads(ready.read_text())["labels_root"]).resolve()


def review_overlays(run_dir: Path) -> list[Path]:
    """The overlay image a review must cover for every label of a run's pool."""
    coco = json.loads((_pool(run_dir) / "train" / "annotations.json").read_text())
    folder = Path(run_dir) / REVIEW_DIR
    return [folder / f"{Path(image['file_name']).stem}.jpg" for image in coco["images"]]


def render_review_overlays(run_dir: Path) -> list[Path]:
    """Draw every label's mask contour and box on its image, once (existing files are kept).

    Returns:
        ``review_overlays(run_dir)``.

    Raises:
        ValueError: If an image or mask of the pool does not decode, or an overlay cannot be
            written.
    """
    import cv2

    pool = _pool(run_dir)
    coco = json.loads((pool / "train" / "annotations.json").read_text())
    by_image = {a["image_id"]: a for a in coco["annotations"]}
    paths = review_overlays(run_dir)
    for image, path in zip(coco["images"], paths, strict=True):
        if path.is_file():
            continue
        annotation = by_image[image["id"]]
        picture = cv2.imread(str(pool / "train" / "images" / image["file_name"]))
        mask = cv2.imread(str(pool / "train" / annotation["mask_file"]), 0)
        if picture is None or mask is None:
            raise ValueError(f"cannot decode the label {image['file_name']} of {pool}")
        contours, _ = cv2.findContours(
            (mask > MASK_THRESHOLD).astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(picture, contours, -1, REVIEW_COLORS["mask"], REVIEW_LINE_PX)
        x, y, w, h = (float(v) for v in annotation["bbox"])
        corners = (int(x), int(y)), (int(x + w), int(y + h))
        cv2.rectangle(picture, *corners, REVIEW_COLORS["box"], REVIEW_LINE_PX)
        path.parent.mkdir(exist_ok=True)
        if not cv2.imwrite(str(path), picture):
            raise ValueError(f"could not write the review overlay {path}")
    return paths


def _review_status(run_dir: Path, accept_unreviewed: bool, name: str) -> dict[str, Any]:
    """The round's review record; raise when the review is missing or stale and not waived."""
    overlays = render_review_overlays(run_dir)
    try:
        review.verify(run_dir, run_dir / layout.LABELS_READY_NAME, overlays=overlays)
    except review.ReviewError as err:
        if not accept_unreviewed:
            raise review.ReviewError(
                f"{err}. Inspect {run_dir / REVIEW_DIR}, then record the review with `doper "
                f"review RECIPE --obj-id N --reviewer NAME --name {name}` or train anyway with "
                "--accept-unreviewed (recorded)"
            ) from err
        log(f"warning: training {run_dir} without a current human review ({err})")
        return {"status": "accepted_unreviewed", "reason": str(err)}
    record = review.read(run_dir)
    return {
        "status": "reviewed",
        "reviewer": record["reviewer"],
        "review_sha256": sha256_file(run_dir / review.REVIEWED_NAME),
        "overlays": len(overlays),
    }


# --------------------------------------------------------------------------------------
# The series
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FollowupResult:
    """Where a follow-up series stopped.

    Attributes:
        name: The follow-up.
        obj_id: Its object.
        rounds: One ``progress.json`` row per round run.
        stop_reason: Why it stopped (``stop_reason``, or ``below_minimum_label_pool``).
        final_student: The last trained student (the round-1 teacher when none trained).
        final_student_sha256: Its SHA-256.
        final_student_yield: Its strict-gate labels-only yield (None when unmeasured).
        progress: The ``progress.json`` path.
    """

    name: str
    obj_id: int
    rounds: tuple[dict[str, Any], ...]
    stop_reason: str
    final_student: Path
    final_student_sha256: str
    final_student_yield: int | None
    progress: Path


@dataclass(frozen=True)
class _Series:
    recipe: Recipe
    name: str
    spec: FollowupSpec
    root: Path
    runs: Path
    predictor_factory: adaptation.PredictorFactory
    overrides: Mapping[str, Any]

    @property
    def base_run(self) -> Path:
        return layout.object_run(self.runs, BASE_ARM, self.spec.obj_id)

    def relative(self, path: Path) -> str:
        return Path(os.path.relpath(path, self.runs)).as_posix()


def _publish_series_identity(series: _Series) -> None:
    path = series.runs / series.name / layout.IDENTITY_NAME
    record = {
        "recipe": series.recipe.name,
        "followup": series.name,
        **series.spec.as_record(),
        "overrides": dict(series.overrides),
    }
    if path.is_file():
        if json.loads(path.read_text()) != json_roundtrip(record):
            raise ValueError(
                f"{path.parent} was started with another identity; pass --force to restart it"
            )
        return
    publish_json(path, record)


def _round(
    series: _Series,
    number: int,
    teacher: adaptation.TeacherIdentity,
    accept_unreviewed: bool,
) -> tuple[adaptation.MethodResult, dict[str, Any]]:
    """Run one round (labels, review, train); return its result and manifest."""
    spec = series.spec
    run_dir = round_dir(series.runs, series.name, number, spec.obj_id)
    reuse = spec.reuse_labels_from if number == 1 else None
    reviewed = spec.require_review and number == 1
    identity = {
        "followup": series.name,
        "round": number,
        "reused_labels_from": reuse,
        "review_required": reviewed,
    }
    gate = round_gate(series.recipe, spec, number)
    run = adaptation.RunSpec(series.name, spec.method(), gate, run_dir, identity, reuse)
    options: dict[str, Any] = {
        "predictor_factory": series.predictor_factory,
        "consent": False,
        "overrides": series.overrides,
    }
    args = (series.recipe, run, spec.obj_id, series.root, series.runs, teacher)
    result = adaptation.execute_run(*args, labels_only=True, **options)
    record = None
    if result.status == "labels_ready" and reviewed:
        completed = run_dir / layout.COMPLETED_NAME
        if completed.is_file():
            record = json.loads(completed.read_text()).get("review")
        else:
            record = _review_status(run_dir, accept_unreviewed, series.name)
    if result.status == "labels_ready":
        extra = {"review": record} if reviewed else None
        result = adaptation.execute_run(*args, labels_only=False, receipt_extra=extra, **options)
    manifest = json.loads((_pool(run_dir) / adaptation.POOL_MANIFEST).read_text())
    return result, {"manifest": manifest, "review": record, "gate": gate.as_manifest()}


def _probe(series: _Series, student: Path) -> int:
    """Labels-only strict-gate yield of the final student (never a further round)."""
    assert series.recipe.adapt is not None
    spec = series.spec
    teacher = adaptation.require_teacher(series.base_run, student)
    run = adaptation.RunSpec(
        series.name,
        spec.method(),
        series.recipe.adapt.gate,
        probe_dir(series.runs, series.name, spec.obj_id),
        {"followup": series.name, "round": layout.FINAL_PROBE_DIR},
    )
    result = adaptation.execute_run(
        series.recipe,
        run,
        spec.obj_id,
        series.root,
        series.runs,
        teacher,
        predictor_factory=series.predictor_factory,
        consent=False,
        labels_only=True,
        overrides=series.overrides,
    )
    return result.labels


def run_self_training(
    recipe: Recipe,
    name: str,
    root: Path,
    runs: Path,
    *,
    predictor_factory: adaptation.PredictorFactory,
    accept_unreviewed: bool = False,
    force: bool = False,
    overrides: Mapping[str, Any] | None = None,
) -> FollowupResult:
    """Run (or resume) one follow-up series to its stop, then probe the final student.

    Args:
        recipe: Dataset recipe with ``adapt`` and ``followups``.
        name: The follow-up.
        root: Prepared dataset root (``<root>/onboarding/static``, ``<root>/objects``).
        runs: Runs root; the object's completed base is ``<runs>/base/obj_XXXXXX``.
        predictor_factory: Loads a teacher checkpoint into a pseudo-label ``Predictor``
            (``adaptation.load_predictor``; tests pass an oracle).
        accept_unreviewed: Train a review-gated round without a current review (recorded in
            ``completed.json`` and ``progress.json``).
        force: Restart the whole series (``<runs>/<name>``); reused method pools are kept.
        overrides: Config keys replacing the adaptation template (every round).

    Returns:
        Where the series stopped.

    Raises:
        ValueError: For an unknown follow-up, a broken base, teacher, pool or student contract,
            a missing or stale review of a review-gated round (``review.ReviewError``), or a
            series started with another identity.
    """
    spec = followup_spec(recipe, name)
    adaptation.check_adapt_overrides(overrides or {})
    series = _Series(recipe, name, spec, Path(root), Path(runs), predictor_factory, overrides or {})
    folder = series.runs / name
    if force and folder.exists():
        shutil.rmtree(folder)
    _publish_series_identity(series)
    seed = (
        None
        if spec.teacher == BASE_ARM
        else layout.object_run(series.runs, spec.teacher, spec.obj_id)
    )
    teacher = adaptation.require_teacher(series.base_run, seed)
    if teacher.obj_id != spec.obj_id:
        raise ValueError(f"the base run of object {spec.obj_id} records object {teacher.obj_id}")
    progress = folder / layout.PROGRESS_NAME
    state: dict[str, Any] = {
        "recipe": recipe.name,
        "followup": name,
        "obj_id": spec.obj_id,
        "spec": spec.as_record(),
        "rounds": [],
        "finished": False,
    }
    previous: Mapping[str, Any] = {"records": []}
    reason: str | None = None
    final = teacher
    trained = False
    for number in range(1, spec.rounds + 1):
        state.update(current_round=number, current_stage="round")
        publish_json(progress, state)
        result, info = _round(series, number, teacher, accept_unreviewed)
        state["rounds"].append(
            {
                "round": number,
                "run": series.relative(result.run_dir),
                "status": result.status,
                "labels": result.labels,
                "gate": info["gate"],
                "teacher_kind": teacher.kind,
                "teacher_sha256": teacher.sha256,
                "student_sha256": result.student_sha256,
                "churn": label_churn(previous, info["manifest"]),
                "review": info["review"],
            }
        )
        publish_json(progress, state)
        log(f"{name} round {number}: {result.status}, {result.labels} labels")
        if result.status != "completed":
            reason = "no_positive_label_pool" if result.labels == 0 else "below_minimum_label_pool"
            trained = False
            break
        target = math.inf if _relaxed(recipe, spec, number) else spec.target_labels
        reason = stop_reason(result.labels, number, target, spec.rounds)
        final = adaptation.require_teacher(series.base_run, result.run_dir)
        trained = True
        if reason is not None:
            break
        teacher, previous = final, info["manifest"]
    assert reason is not None
    last = state["rounds"][-1]
    if trained:
        state.update(stop_reason=reason, current_stage="final_yield_probe")
        publish_json(progress, state)
        yield_ = _probe(series, final.path)
        probe = series.relative(probe_dir(series.runs, name, spec.obj_id))
    else:
        # The last pool came from the last teacher, but no new student was trained.
        strict = not _relaxed(recipe, spec, last["round"])
        yield_, probe = (last["labels"] if strict else None), None
    state.update(
        finished=True,
        current_stage="finished",
        stop_reason=reason,
        final_student=series.relative(final.path),
        final_student_sha256=final.sha256,
        final_student_yield=yield_,
        final_probe=probe,
    )
    publish_json(progress, state)
    log(f"{name}: {reason} after round {last['round']}; final student yield {yield_}")
    return FollowupResult(
        name,
        spec.obj_id,
        tuple(state["rounds"]),
        reason,
        final.path,
        final.sha256,
        yield_,
        progress,
    )
