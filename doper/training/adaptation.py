# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The adaptation engine: label pool, audit, adapt config, three-epoch student, receipt.

Every adaptation method of a recipe (``Recipe.adapt``) runs through ``run_method``, one object
at a time, in ``<runs>/<method>/obj_XXXXXX`` (``doper.bop.layout``):

1. **Teacher.** ``require_teacher`` pins the completed 25-epoch base of the object, or a
   completed same-object three-epoch student for follow-up rounds. The same checkpoint labels
   the pool and initializes the student (``init_from`` with ``require_init_from``);
   ``identity.json`` records it next to the base it descends from.
2. **Labels.** A strict-gate pseudo-label pool of the static, dynamic or (with transductive
   consent) test view (``doper.pseudo_labels.monocular``), the union of a static and a dynamic
   method's pools, or one arm of the object's paired GT pools (``doper.bop.gt_labels``,
   ``<runs>/gt_pools/obj_XXXXXX``). The pool is audited before ``labels_ready.json`` freezes
   its file hashes; a later step re-checks them.
3. **Minimum.** Fewer accepted labels than the method's minimum for the object write
   ``adaptation_skipped.json`` and no student, also with ``labels_only``, which otherwise stops
   here (so a labels-only run is unfinished only when its pool is trainable).
4. **Student.** ``adapt_config`` resolves the recipe's shared adaptation template over the base
   stage (backbone and architecture inherited): batch 16 with ``real_per_batch`` real labels,
   ``max_iters_per_epoch = ceil(labels / real_per_batch)``. Training resumes an interrupted
   ``work/last.pth``. ``verify_student`` audits the endpoint (budget, finite state, epoch log,
   real supervision) before ``completed.json`` is written.

Every step is idempotent: a re-run verifies what exists and continues; ``force`` restarts the
method's run directory (never the base or the shared GT pools). Importing this module does not
import torch.

``execute_run`` is the same engine for a run that is not a recipe method: a follow-up round
(``doper.training.followups``) brings its own run directory, gate, minimum and identity fields,
and may train on a copy of a method's verified pool (``RunSpec.reuse_from``).

``run_multisensor`` is the T-LESS three-sensor stage (``AdaptSpec.multisensor``) in
``<runs>/multisensor/obj_XXXXXX``: no label pool, but the reviewed PrimeSense, Kinect and Canon
TRAIN roots of ``doper.bop.multisensor`` as three mix roots beside PBR, an uncapped epoch
(``multisensor_batches``) and its own student audit (``verify_multisensor_student``). It shares
the teacher pin, the resumable launch with its lineage check and the atomic receipts.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml

from ..bop import layout, sources
from ..bop.sources import publish_json, publish_text, sha256_file
from ..console import log
from ..pseudo_labels.gate import GateSpec
from ..recipes.schema import (
    ADAPT_EPOCHS,
    ARCHITECTURE_KEYS,
    BASE_ARM,
    COMPUTED_ADAPT_KEYS,
    HOT3D_PANEL,
    MULTISENSOR_ARM,
    MULTISENSOR_SOURCES,
    PATH_KEYS,
    LaunchSpec,
    MethodSpec,
    MultisensorSpec,
    Recipe,
    TrainSpec,
    resolve_config,
    training_namespace,
)

if TYPE_CHECKING:
    from ..pseudo_labels.monocular import Predictor

ADAPT_PATH_KEYS = (*PATH_KEYS, "labels", "teacher")
MULTISENSOR_PATH_KEYS = (*PATH_KEYS, "teacher", "sensors")
BUDGET_KEYS = ("epochs", *COMPUTED_ADAPT_KEYS)
BASE_EPOCHS = 25
STUDENT_EPOCHS = ADAPT_EPOCHS
TEACHER_ARCHITECTURE = ("backbone", "num_keypoints", "det_input_size", "crop_size", "keypoint_head")
ONBOARDING_DIR = "onboarding"
GT_VIEW = "gt"
RESUME_NAME = "recipe_resume.yaml"
EPOCH_TIMING_NAME = "epoch_timing.jsonl"
POOL_MANIFEST = "manifest.json"
GT_POOL_RECEIPT = "POOL_READY.json"
PredictorFactory = Callable[[Path], "Predictor"]


@dataclass(frozen=True)
class TeacherIdentity:
    """The checkpoint that labels a method's pool and initializes its student.

    Attributes:
        path: The teacher checkpoint.
        sha256: Its SHA-256.
        base_sha256: SHA-256 of the object's completed 25-epoch base checkpoint.
        kind: ``base`` (the base itself) or ``student`` (a completed three-epoch student).
        obj_id: The object.
    """

    path: Path
    sha256: str
    base_sha256: str
    kind: Literal["base", "student"]
    obj_id: int


@dataclass(frozen=True)
class MethodResult:
    """Where ``run_method`` stopped.

    Attributes:
        method: Method name.
        obj_id: Object.
        status: ``labels_ready`` (``labels_only``), ``skipped`` (below the minimum) or
            ``completed`` (a verified student).
        labels: Accepted labels in the pool.
        run_dir: The method's run directory.
        student_sha256: SHA-256 of ``work/last.pth`` when completed.
    """

    method: str
    obj_id: int
    status: Literal["labels_ready", "skipped", "completed"]
    labels: int
    run_dir: Path
    student_sha256: str | None = None


@dataclass(frozen=True)
class RunSpec:
    """One adaptation run: a recipe method's, or a follow-up round's.

    Attributes:
        name: The run's method name (``identity.json`` / ``completed.json``).
        method: Where its labels come from and its minimum.
        gate: The pseudo-label gate of its pool.
        run_dir: The run directory of one object.
        identity: Extra ``identity.json`` fields (a follow-up's name and round).
        reuse_from: A recipe method whose verified pool of the object is copied instead of
            labeling (it must share this run's teacher and gate).
    """

    name: str
    method: MethodSpec
    gate: GateSpec
    run_dir: Path
    identity: Mapping[str, Any] = field(default_factory=dict)
    reuse_from: str | None = None


# --------------------------------------------------------------------------------------
# Adapt config
# --------------------------------------------------------------------------------------
def method_spec(recipe: Recipe, method: str) -> MethodSpec:
    """Return a recipe's method.

    Raises:
        ValueError: If the recipe has no such adaptation method.
    """
    methods = recipe.adapt.methods if recipe.adapt is not None else {}
    if method not in methods:
        raise ValueError(f"{method!r} is not a {recipe.name} method; methods: {list(methods)}")
    return methods[method]


def check_adapt_overrides(overrides: Mapping[str, Any]) -> None:
    """Refuse overrides that would change the paths, launch, architecture or budget.

    Raises:
        ValueError: Naming the refused keys.
    """
    refused = (
        (PATH_KEYS, "dataset paths; they follow from --root, --runs and --obj-id"),
        (tuple(f.name for f in fields(LaunchSpec)), "launch keys; the engine sets them"),
        (ARCHITECTURE_KEYS, "architecture keys; adaptation inherits the base's"),
        (BUDGET_KEYS, "fixed adaptation budget keys; the engine computes them per pool"),
    )
    for keys, reason in refused:
        hit = sorted(set(overrides) & set(keys))
        if hit:
            raise ValueError(f"--override cannot set {hit}: {reason}")


def adapt_config(
    recipe: Recipe,
    method: str,
    n_labels: int,
    paths: Mapping[str, Any],
    *,
    obj_id: int | None = None,
    overrides: Mapping[str, Any] | None = None,
    spec: MethodSpec | None = None,
) -> tuple[dict[str, Any], LaunchSpec]:
    """Resolve the training config of one adaptation student.

    The base stage's config, replaced by the recipe's adaptation template, plus the computed
    keys: ``mix_data_root`` (the pool), ``mix_per_batch`` (``real_per_batch``),
    ``max_iters_per_epoch = ceil(n_labels / real_per_batch)`` and the min-mix checks
    (``min_mix_mask_only_samples = 0 if box_mask_keypoints else 1``,
    ``min_mix_keypoint_samples = n_labels if box_mask_keypoints else 0``). The student
    initializes from the teacher (``require_init_from``).

    Args:
        recipe: Dataset recipe with an ``adapt`` section.
        method: Method name.
        n_labels: Accepted labels in the pool.
        paths: ``data_root``, ``keypoints_3d``, ``work_dir``, ``labels`` and ``teacher``.
        obj_id: The object, to enforce its minimum (None skips that check).
        overrides: Config keys that replace the template values (``check_adapt_overrides``).
        spec: The method itself, for a run that is not in the recipe's method table (a
            follow-up round); None looks ``method`` up.

    Returns:
        The resolved config-file mapping and its ``LaunchSpec``.

    Raises:
        ValueError: For an unknown method, zero labels, fewer than the object's minimum, a
            missing path, a refused override, or a backbone other than the base's.
    """
    spec = spec or method_spec(recipe, method)
    assert recipe.adapt is not None
    if n_labels < 1:
        raise ValueError("adaptation needs accepted real labels (never zero)")
    if obj_id is not None and n_labels < spec.minimum_for(obj_id):
        raise ValueError(
            f"{method} requires at least {spec.minimum_for(obj_id)} accepted labels for "
            f"object {obj_id}; got {n_labels}"
        )
    missing = sorted(set(ADAPT_PATH_KEYS) - set(paths))
    if missing:
        raise ValueError(f"adapt_config needs the paths {missing}")
    check_adapt_overrides(overrides or {})
    keypoint_arm = spec.supervision == "box_mask_keypoints"
    per_batch = recipe.adapt.real_per_batch
    config = {
        **recipe.base.config,
        **recipe.adapt.config,
        "mix_data_root": [str(paths["labels"])],
        "mix_per_batch": [per_batch],
        "max_iters_per_epoch": math.ceil(n_labels / per_batch),
        "min_mix_mask_only_samples": 0 if keypoint_arm else 1,
        "min_mix_keypoint_samples": n_labels if keypoint_arm else 0,
    }
    stage = TrainSpec(config, LaunchSpec(init_from=str(paths["teacher"]), require_init_from=True))
    train_paths = {key: str(paths[key]) for key in PATH_KEYS}
    resolved, _ = training_namespace(stage, train_paths, overrides)
    _, launch = resolve_config(stage, train_paths, overrides)
    if resolved["backbone"] != recipe.base.config["backbone"]:
        raise ValueError("the adaptation backbone must equal the base backbone")
    return resolved, launch


# --------------------------------------------------------------------------------------
# Teacher
# --------------------------------------------------------------------------------------
def _read_json(path: Path) -> Any:  # noqa: ANN401
    return json.loads(Path(path).read_text())


def _base_receipt(base_run: Path) -> dict[str, Any]:
    receipt = base_run / layout.COMPLETED_NAME
    done = _read_json(receipt) if receipt.is_file() else {}
    checkpoint, keypoints = base_run / layout.CHECKPOINT_PATH, base_run / layout.KEYPOINTS_PATH
    ok = (
        done.get("stage") == BASE_ARM
        and done.get("epochs") == BASE_EPOCHS
        and isinstance(done.get("obj_id"), int)
        and checkpoint.is_file()
        and keypoints.is_file()
        and sha256_file(checkpoint) == done.get("checkpoint_sha256")
        and sha256_file(keypoints) == done.get("keypoints_sha256")
    )
    if not ok:
        raise ValueError(
            f"{base_run} is not a completed 25-epoch base run with an unchanged checkpoint and "
            "keypoints; adaptation starts from it"
        )
    return done


def _student_checkpoint(override: Path) -> Path:
    override = Path(override)
    return override / layout.CHECKPOINT_PATH if override.is_dir() else override


def _require_student(student: Path, base_run: Path, done: Mapping[str, Any]) -> None:
    """Raise unless ``student`` is a completed same-object three-epoch student of the base."""
    from ..checkpoint_io import load_checkpoint
    from ..keypoints import keypoints_3d_fingerprint, load_keypoints_3d

    path = student.parent.parent / layout.COMPLETED_NAME
    receipt = _read_json(path) if path.is_file() and student.is_file() else {}
    if (
        receipt.get("obj_id") != done["obj_id"]
        or receipt.get("completed_epochs") != STUDENT_EPOCHS
        or receipt.get("student_sha256") != sha256_file(student)
    ):
        raise ValueError(
            f"teacher override {student} must be a completed same-object three-epoch student "
            "(completed.json with completed_epochs 3 and its student_sha256)"
        )
    if receipt.get("base_teacher_sha256") != done["checkpoint_sha256"]:
        raise ValueError(f"teacher override {student} descends from another base")
    checkpoint = load_checkpoint(student)
    cfg = checkpoint.get("cfg") or {}
    points = load_keypoints_3d(base_run / layout.KEYPOINTS_PATH)
    if checkpoint.get("epoch") != STUDENT_EPOCHS - 1:
        raise ValueError(f"teacher override {student} is not the epoch-index-2 checkpoint")
    if cfg.get("keypoints_3d_fingerprint") != keypoints_3d_fingerprint(points):
        raise ValueError(f"teacher override {student} uses another keypoint layout")
    base_model = (load_checkpoint(base_run / layout.CHECKPOINT_PATH).get("cfg") or {}).get("model")
    for key in TEACHER_ARCHITECTURE:
        if (cfg.get("model") or {}).get(key) != (base_model or {}).get(key):
            raise ValueError(f"teacher override {student} differs from the base in {key}")


def require_teacher(base_run: Path, override: Path | None = None) -> TeacherIdentity:
    """Pin the adaptation teacher of one object.

    Args:
        base_run: The object's base run; it must be the completed 25-epoch run with an
            unchanged checkpoint and keypoints.
        override: A follow-up teacher: a student run directory or its ``work/last.pth``. It
            must be a completed same-object student (``completed.json`` with
            ``completed_epochs`` 3, the checkpoint's ``student_sha256`` and the base's
            ``base_teacher_sha256``), checkpoint epoch index 2, the base's keypoint
            fingerprint and the base's architecture.

    Returns:
        The teacher; the base itself without ``override``.

    Raises:
        ValueError: If the base or the override breaks its contract.
    """
    base_run = Path(base_run)
    done = _base_receipt(base_run)
    base_sha = done["checkpoint_sha256"]
    if override is None:
        return TeacherIdentity(
            base_run / layout.CHECKPOINT_PATH, base_sha, base_sha, "base", done["obj_id"]
        )
    student = _student_checkpoint(override)
    _require_student(student, base_run, done)
    return TeacherIdentity(student, sha256_file(student), base_sha, "student", done["obj_id"])


# --------------------------------------------------------------------------------------
# Student audit
# --------------------------------------------------------------------------------------
def _timing_faults(work: Path, epochs: int) -> list[str]:
    path = work / EPOCH_TIMING_NAME
    if not path.is_file():
        return [f"no {EPOCH_TIMING_NAME}"]
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    faults = []
    if [row.get("epoch") for row in rows] != list(range(epochs)):
        faults.append(
            f"epoch log records epochs {[r.get('epoch') for r in rows]}, not 0..{epochs - 1}"
        )
    if any(row.get("nonfinite_microbatches") or row.get("scaler_skipped_updates") for row in rows):
        faults.append("epoch log records non-finite microbatches or skipped updates")
    return faults


def _supervision_faults(
    checkpoint: Mapping[str, Any], n_labels: int, spec: MethodSpec
) -> list[str]:
    sources_ = (checkpoint.get("training_provenance") or {}).get("train_sources") or []
    if len(sources_) != 2 or not sources_[0].get("keypoint_samples"):
        return ["training sources are not the PBR rehearsal plus one real pool"]
    real = sources_[1]
    expected = n_labels if spec.supervision == "box_mask_keypoints" else 0
    counts = (real.get("samples"), real.get("keypoint_samples"), real.get("mask_only_samples"))
    if counts != (n_labels, expected, n_labels - expected):
        return [
            f"real supervision (samples, keypoint, mask-only) {counts} differs from "
            f"{(n_labels, expected, n_labels - expected)} of {spec.supervision}"
        ]
    return []


def verify_student(
    run_dir: Path,
    n_labels: int,
    spec: MethodSpec,
    *,
    epochs: int = STUDENT_EPOCHS,
    real_per_batch: int = 4,
) -> list[str]:
    """Audit a trained student endpoint.

    Checks: ``work/last.pth`` is epoch index ``epochs - 1`` with
    ``global_step == epochs * ceil(n_labels / real_per_batch)``; model and optimizer are finite
    and trained (``doper.bop.audit.checkpoint_is_finite``); ``epoch_timing.jsonl`` records
    epochs ``0..epochs-1`` once each without non-finite microbatches or skipped updates;
    ``train_sources`` is the PBR rehearsal plus the real pool, whose keypoint and mask-only
    counts match the method's supervision; the keypoint fingerprint matches the run's geometry.

    Args:
        run_dir: The method's run directory of one object.
        n_labels: Accepted labels in the pool.
        spec: The method.
        epochs: The adaptation epochs.
        real_per_batch: Real labels per batch.

    Returns:
        The faults; empty for a sound student.
    """
    from ..checkpoint_io import load_checkpoint

    path = Path(run_dir) / layout.CHECKPOINT_PATH
    if not path.is_file():
        return [f"no student checkpoint {path}"]
    checkpoint = load_checkpoint(path)
    steps = epochs * math.ceil(n_labels / real_per_batch)
    faults = _endpoint_faults(path, checkpoint, epochs, steps)
    faults.extend(_supervision_faults(checkpoint, n_labels, spec))
    faults.extend(_fingerprint_faults(Path(run_dir), checkpoint))
    return faults


def _endpoint_faults(
    path: Path, checkpoint: Mapping[str, Any], epochs: int, steps: int | None
) -> list[str]:
    """Epoch, update count (None: not checkable), finite trained state and the epoch log."""
    from ..bop.audit import checkpoint_is_finite

    faults = []
    if checkpoint.get("epoch") != epochs - 1:
        faults.append(f"student stopped at epoch index {checkpoint.get('epoch')}, not {epochs - 1}")
    if steps is not None and checkpoint.get("global_step") != steps:
        faults.append(f"student made {checkpoint.get('global_step')} updates, not {steps}")
    if "optimizer" not in checkpoint or not checkpoint_is_finite(path):
        faults.append("student model or optimizer state is non-finite or untrained")
    faults.extend(_timing_faults(path.parent, epochs))
    return faults


def _fingerprint_faults(run_dir: Path, checkpoint: Mapping[str, Any]) -> list[str]:
    from ..keypoints import keypoints_3d_fingerprint, load_keypoints_3d

    geometry = run_dir / layout.KEYPOINTS_PATH
    recorded = (checkpoint.get("cfg") or {}).get("keypoints_3d_fingerprint")
    if geometry.is_file() and recorded != keypoints_3d_fingerprint(load_keypoints_3d(geometry)):
        return ["student keypoint fingerprint differs from the run geometry"]
    return []


# --------------------------------------------------------------------------------------
# Label pools
# --------------------------------------------------------------------------------------
def validation_exclusions(recipe: Recipe, root: Path) -> frozenset[str]:
    """SHA-256 of every validation image of the recipe's dataset (the pool leakage guard)."""
    profile = sources.load_profile(recipe.profile_ref())
    if recipe.validation.population == HOT3D_PANEL:
        from ..bop.onboarding_hot3d import load_hot3d_validation

        return load_hot3d_validation(Path(root), profile).hashes
    from ..bop.onboarding import load_validation

    return load_validation(Path(root), profile).hashes


def load_predictor(checkpoint: Path) -> Predictor:
    """The default ``predictor_factory``: the teacher model on CUDA when present, else CPU."""
    import torch

    from ..config import DataConfig, DoperConfig, ModelConfig
    from ..infer import load_model
    from ..pseudo_labels.monocular import ModelPredictor

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = DoperConfig(data=DataConfig(root="."), model=ModelConfig(pretrained=False))
    return ModelPredictor(load_model(checkpoint, cfg, device).eval(), cfg, device)


def _pool_hashes(labels_root: Path) -> dict[str, str]:
    """SHA-256 of every training file of a pool, its manifest and (GT) the shared receipt."""
    files = sorted(p for p in (labels_root / "train").rglob("*") if p.is_file())
    for receipt in (labels_root / POOL_MANIFEST, labels_root.parent / GT_POOL_RECEIPT):
        if receipt.is_file():
            files.append(receipt)
    return {os.path.relpath(p, labels_root): sha256_file(p) for p in files}


@dataclass(frozen=True)
class _Context:
    """One ``run_method`` call."""

    recipe: Recipe
    method: str
    spec: MethodSpec
    obj_id: int
    root: Path
    runs: Path
    run_dir: Path
    teacher: TeacherIdentity
    predictor_factory: PredictorFactory
    consent: bool
    gate: GateSpec
    identity: Mapping[str, Any] = field(default_factory=dict)
    reuse_from: str | None = None

    @property
    def base_run(self) -> Path:
        return layout.object_run(self.runs, BASE_ARM, self.obj_id)


def _fresh(out: Path) -> Path:
    """Remove an unreceipted pool left by an interrupted or refused run."""
    if out.exists():
        log(f"removing unreceipted label pool {out}")
        shutil.rmtree(out)
    return out


def _view_pool(ctx: _Context) -> tuple[Path, int, list[str]]:
    from ..pseudo_labels import monocular

    assert ctx.recipe.adapt is not None
    view = ctx.root / ONBOARDING_DIR / ctx.spec.source
    exclusions = validation_exclusions(ctx.recipe, ctx.root)
    out = _fresh(ctx.run_dir / layout.LABELS_DIR)
    summary = monocular.export_pool(
        ctx.predictor_factory(ctx.teacher.path),
        ctx.base_run / layout.KEYPOINTS_PATH,
        view,
        ctx.spec.source,
        exclusions,
        ctx.gate,
        out,
        teacher_sha256=ctx.teacher.sha256,
        obj_id=ctx.obj_id,
    )
    return out, summary.accepted, monocular.verify_pool(out, view, exclusions)


def _combined_pool(ctx: _Context) -> tuple[Path, int, list[str]]:
    from ..pseudo_labels import monocular

    assert ctx.recipe.adapt is not None
    components = {}
    for name in ctx.spec.components:
        run_method(
            ctx.recipe,
            name,
            ctx.obj_id,
            ctx.root,
            ctx.runs,
            predictor_factory=ctx.predictor_factory,
            consent=ctx.consent,
            labels_only=True,
            force=False,
            teacher=ctx.teacher.path if ctx.teacher.kind == "student" else None,
        )
        component = layout.object_run(ctx.runs, name, ctx.obj_id)
        if _read_json(component / layout.IDENTITY_NAME)["teacher_sha256"] != ctx.teacher.sha256:
            raise ValueError(f"component {name} was labeled by another teacher")
        components[ctx.recipe.adapt.methods[name].source] = component / layout.LABELS_DIR
    out = _fresh(ctx.run_dir / layout.LABELS_DIR)
    summary = monocular.merge_pools(components, out)
    return out, summary.accepted, monocular.verify_merged(out)


def _reused_pool(ctx: _Context) -> tuple[Path, int, list[str]]:
    """Copy a recipe method's verified pool of the object (its teacher and gate must match)."""
    from ..pseudo_labels import monocular

    assert ctx.reuse_from is not None
    source_spec = method_spec(ctx.recipe, ctx.reuse_from)
    run_method(
        ctx.recipe,
        ctx.reuse_from,
        ctx.obj_id,
        ctx.root,
        ctx.runs,
        predictor_factory=ctx.predictor_factory,
        consent=False,
        labels_only=True,
        force=False,
    )
    source = layout.object_run(ctx.runs, ctx.reuse_from, ctx.obj_id)
    receipt = _read_json(source / layout.LABELS_READY_NAME)
    pool = (source / receipt["labels_root"]).resolve()
    if _pool_hashes(pool) != receipt["files_sha256"]:
        raise ValueError(f"the reused {ctx.reuse_from} pool changed since its receipt")
    manifest = _read_json(pool / POOL_MANIFEST)
    if (
        receipt["teacher_sha256"] != ctx.teacher.sha256
        or manifest.get("checkpoint_sha256") != ctx.teacher.sha256
        or manifest.get("gate") != ctx.gate.as_manifest()
    ):
        raise ValueError(f"the reused {ctx.reuse_from} labels do not match this teacher/gate")
    out = _fresh(ctx.run_dir / layout.LABELS_DIR)
    shutil.copytree(pool, out)
    view = ctx.root / ONBOARDING_DIR / source_spec.source
    exclusions = validation_exclusions(ctx.recipe, ctx.root)
    return out, int(manifest["accepted"]), monocular.verify_pool(out, view, exclusions)


def _gt_pool(ctx: _Context) -> tuple[Path, int, list[str]]:
    from ..bop import gt_labels

    pool = layout.gt_pool(ctx.runs, ctx.obj_id)
    summary = gt_labels.export_gt_pool(
        ctx.root / ONBOARDING_DIR / GT_VIEW,
        ctx.obj_id,
        ctx.base_run / layout.KEYPOINTS_PATH,
        ctx.base_run,
        pool,
    )
    return pool / ctx.spec.supervision, summary.labels, gt_labels.verify_gt_pool(pool)


def _labels(ctx: _Context) -> tuple[Path, int]:
    """The method's verified label pool and its size; build and receipt it when missing."""
    ready = ctx.run_dir / layout.LABELS_READY_NAME
    if ready.is_file():
        receipt = _read_json(ready)
        labels_root = (ctx.run_dir / receipt["labels_root"]).resolve()
        if (
            receipt["teacher_sha256"] != ctx.teacher.sha256
            or _pool_hashes(labels_root) != receipt["files_sha256"]
        ):
            raise ValueError(
                f"label pool changed since {ready}; pass --force to relabel {ctx.method}"
            )
        return labels_root, int(receipt["labels"])
    build = {"combined": _combined_pool, "gt": _gt_pool}.get(ctx.spec.source, _view_pool)
    if ctx.reuse_from is not None:
        build = _reused_pool
    labels_root, n, faults = build(ctx)
    if faults:
        raise ValueError(f"label pool {labels_root} failed its audit: {faults}")
    publish_json(
        ready,
        {
            "labels": n,
            "labels_root": os.path.relpath(labels_root, ctx.run_dir),
            "source": ctx.spec.source,
            "supervision": ctx.spec.supervision,
            "teacher_sha256": ctx.teacher.sha256,
            "files_sha256": _pool_hashes(labels_root),
        },
    )
    log(f"object {ctx.obj_id}: {ctx.method} pool holds {n} verified labels")
    return labels_root.resolve(), n


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def launch_training(config: Path) -> None:
    """Train in-process from a resolved config file (``doper.train``)."""
    from .. import train

    train.main(["--config", str(config)])


def _identity_record(ctx: _Context) -> dict[str, Any]:
    assert ctx.recipe.adapt is not None
    record: dict[str, Any] = {
        "recipe": ctx.recipe.name,
        "method": ctx.method,
        "obj_id": ctx.obj_id,
        "source": ctx.spec.source,
        "supervision": ctx.spec.supervision,
        "minimum_labels": ctx.spec.minimum_for(ctx.obj_id),
        "teacher": os.path.relpath(ctx.teacher.path, ctx.run_dir),
        "teacher_kind": ctx.teacher.kind,
        "teacher_sha256": ctx.teacher.sha256,
        "base_teacher_sha256": ctx.teacher.base_sha256,
        "keypoints_sha256": sha256_file(ctx.base_run / layout.KEYPOINTS_PATH),
        "gate": ctx.gate.as_manifest() if ctx.spec.source != "gt" else None,
        "real_per_batch": ctx.recipe.adapt.real_per_batch,
        "transductive_test_adaptation": ctx.spec.transductive,
        "test_gt_used": False,
        "test_targets_used": False,
    }
    if ctx.spec.transductive:
        record["notice"] = sources.TEST_NOTICE
    record.update(ctx.identity)
    return record


def _publish_identity(ctx: _Context) -> None:
    _publish_once(ctx.run_dir, _identity_record(ctx))


def _publish_once(run_dir: Path, record: Mapping[str, Any]) -> None:
    """Write ``identity.json``, or check that the run was started with the same identity."""
    path = run_dir / layout.IDENTITY_NAME
    if path.is_file():
        stored = _read_json(path)
        changed = sorted(k for k in set(stored) | set(record) if stored.get(k) != record.get(k))
        if changed:
            raise ValueError(
                f"{run_dir} was started with another identity ({changed}); pass --force to "
                "restart it"
            )
        return
    publish_json(path, record)


def _finite(value: Any) -> Any:  # noqa: ANN401
    """``value`` with non-finite floats replaced by None (receipts are strict JSON)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def _run_keypoints(run_dir: Path, base_run: Path) -> Path:
    """The run's copy of the base keypoints (``geometry/keypoints_3d.json``)."""
    keypoints = run_dir / layout.KEYPOINTS_PATH
    base_keypoints = base_run / layout.KEYPOINTS_PATH
    if not keypoints.is_file():
        keypoints.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base_keypoints, keypoints)
    if sha256_file(keypoints) != sha256_file(base_keypoints):
        raise ValueError(f"{keypoints} differs from the base keypoints")
    return keypoints


def _publish_config(run_dir: Path, resolved: Mapping[str, Any]) -> None:
    """Write ``recipe.yaml``, or refuse a run started with another config."""
    text = yaml.safe_dump(dict(resolved), sort_keys=False)
    config = run_dir / layout.RECIPE_NAME
    if config.is_file() and config.read_text() != text:
        raise ValueError(f"{config} changed (another pool or override); pass --force to restart")
    publish_text(config, text)


def _write_config(ctx: _Context, labels_root: Path, n: int, overrides: Mapping[str, Any]) -> dict:
    from ..cli.common import DATASET_DIR, object_dir

    keypoints = _run_keypoints(ctx.run_dir, ctx.base_run)
    paths = {
        "data_root": object_dir(ctx.root, ctx.obj_id) / DATASET_DIR,
        "keypoints_3d": keypoints,
        "work_dir": ctx.run_dir / layout.CHECKPOINT_PATH.parent,
        "labels": labels_root,
        "teacher": ctx.teacher.path,
    }
    resolved, _ = adapt_config(
        ctx.recipe, ctx.method, n, paths, obj_id=ctx.obj_id, overrides=overrides, spec=ctx.spec
    )
    _publish_config(ctx.run_dir, resolved)
    return resolved


def _train(ctx: _Context, labels_root: Path, n: int, overrides: Mapping[str, Any]) -> str:
    """Train (or resume) the student; return the SHA-256 of the init checkpoint."""
    resolved = _write_config(ctx, labels_root, n, overrides)
    _launch(ctx.run_dir, resolved, ctx.teacher.sha256, ctx.method, ctx.obj_id)
    return ctx.teacher.sha256


def _launch(
    run_dir: Path, resolved: Mapping[str, Any], teacher_sha256: str, method: str, obj_id: int
) -> None:
    """Train the published ``recipe.yaml``, or resume an interrupted ``work/last.pth``."""
    from ..checkpoint_io import load_checkpoint

    epochs = int(resolved["epochs"])
    last = run_dir / layout.CHECKPOINT_PATH
    done = int(load_checkpoint(last)["epoch"]) + 1 if last.is_file() else 0
    if done > epochs:
        raise ValueError(f"{last} exceeds the {epochs}-epoch adaptation budget")
    if done < epochs:
        config = run_dir / layout.RECIPE_NAME
        if last.is_file():
            _require_lineage(teacher_sha256, method, load_checkpoint(last), last)
            log(f"object {obj_id}: resuming {method} from epoch {done}")
            config = run_dir / RESUME_NAME
            resume = {**resolved, "resume": str(last), "init_from": None}
            resume.update(require_init_from=False, require_no_init_from=False)
            publish_text(config, yaml.safe_dump(resume, sort_keys=False))
        launch_training(config)
    _require_lineage(teacher_sha256, method, load_checkpoint(last), last)


def _require_lineage(
    teacher_sha256: str, method: str, checkpoint: Mapping[str, Any], path: Path
) -> None:
    """Raise unless ``checkpoint`` descends from the teacher.

    A session started with ``init_from`` records the teacher SHA-256 as its init checkpoint. A
    resumed session records none; its predecessor was checked before it was resumed.
    """
    provenance = checkpoint.get("training_provenance") or {}
    init = (provenance.get("init_checkpoint") or {}).get("sha256")
    resumed = (provenance.get("resume_checkpoint") or {}).get("path") is not None
    if init != teacher_sha256 and not (init is None and resumed):
        raise ValueError(f"{path} was not initialized from the {method} teacher; pass --force")


def _complete(
    ctx: _Context,
    n: int,
    overrides: Mapping[str, Any],
    labels_root: Path,
    receipt_extra: Mapping[str, Any],
) -> str:
    """Train, audit and receipt the student; return its SHA-256."""
    from ..checkpoint_io import load_checkpoint
    from ..cli.common import prepared_receipt

    assert ctx.recipe.adapt is not None
    prepared = prepared_receipt(ctx.root, ctx.obj_id)
    init_sha = _train(ctx, labels_root, n, overrides)
    resolved = yaml.safe_load((ctx.run_dir / layout.RECIPE_NAME).read_text())
    faults = verify_student(
        ctx.run_dir,
        n,
        ctx.spec,
        epochs=int(resolved["epochs"]),
        real_per_batch=ctx.recipe.adapt.real_per_batch,
    )
    if faults:
        raise ValueError(f"{ctx.method} student of object {ctx.obj_id} failed its audit: {faults}")
    last = ctx.run_dir / layout.CHECKPOINT_PATH
    checkpoint = load_checkpoint(last)
    student = sha256_file(last)
    publish_json(
        ctx.run_dir / layout.COMPLETED_NAME,
        _finite(
            {
                "recipe": ctx.recipe.name,
                "stage": "adapt",
                "method": ctx.method,
                "obj_id": ctx.obj_id,
                "completed_epochs": int(resolved["epochs"]),
                "labels": n,
                "global_step": checkpoint["global_step"],
                "teacher_kind": ctx.teacher.kind,
                "teacher_sha256": ctx.teacher.sha256,
                "base_teacher_sha256": ctx.teacher.base_sha256,
                "init_from_sha256": init_sha,
                "student": str(layout.CHECKPOINT_PATH),
                "student_sha256": student,
                "checkpoint_sha256": student,
                "keypoints_sha256": sha256_file(ctx.run_dir / layout.KEYPOINTS_PATH),
                "source_supervision": checkpoint["training_provenance"]["train_sources"],
                "final_synthetic_metrics": checkpoint.get("metrics"),
                "prepared_hashes": prepared["hashes"],
                **receipt_extra,
            }
        ),
    )
    log(f"object {ctx.obj_id}: {ctx.method} student complete in {ctx.run_dir}")
    return student


def _completed_student(ctx: _Context) -> str:
    receipt = _read_json(ctx.run_dir / layout.COMPLETED_NAME)
    last = ctx.run_dir / layout.CHECKPOINT_PATH
    if (
        not last.is_file()
        or sha256_file(last) != receipt.get("student_sha256")
        or receipt.get("teacher_sha256") != ctx.teacher.sha256
    ):
        raise ValueError(f"{ctx.run_dir} student changed since completed.json; pass --force")
    return receipt["student_sha256"]


def run_method(
    recipe: Recipe,
    method: str,
    obj_id: int,
    root: Path,
    runs: Path,
    *,
    predictor_factory: PredictorFactory,
    consent: bool,
    labels_only: bool,
    force: bool,
    teacher: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> MethodResult:
    """Run one adaptation method for one object: labels, verify, skip or train, receipt.

    Args:
        recipe: Dataset recipe with an ``adapt`` section.
        method: Method name (``recipe.adapt.methods``).
        obj_id: The object; its completed base must be ``<runs>/base/obj_XXXXXX``.
        root: Prepared dataset root (``<root>/onboarding`` views, ``<root>/objects``).
        runs: Runs root.
        predictor_factory: Loads the teacher checkpoint into a pseudo-label ``Predictor``
            (``load_predictor``; tests pass an oracle).
        consent: Transductive consent, required by test-input methods.
        labels_only: Stop after the verified pool (``labels_ready.json``); a pool below the
            minimum is still marked skipped, so only a trainable pool is left unfinished.
        force: Restart the method's run directory.
        teacher: A follow-up teacher (``require_teacher``); None uses the base.
        overrides: Config keys replacing the adaptation template (``check_adapt_overrides``).

    Returns:
        Where the method stopped.

    Raises:
        ValueError: For an unknown method or object, missing consent, a broken teacher, base,
            pool or student contract, or a run started with another identity.
    """
    spec = method_spec(recipe, method)
    if obj_id not in recipe.objects or not spec.applies_to(obj_id):
        scope = list(spec.objects) if spec.objects is not None else "every recipe object"
        raise ValueError(f"{method} runs for objects {scope}, not object {obj_id}")
    if spec.transductive:
        sources.require_transductive_consent(consent)
    runs = Path(runs)
    identity = require_teacher(layout.object_run(runs, BASE_ARM, obj_id), teacher)
    if identity.obj_id != obj_id:
        raise ValueError(f"the base run of object {obj_id} records object {identity.obj_id}")
    if spec.transductive and identity.kind != "base":
        raise ValueError("test inputs adapt from the epoch-25 base, never from a student")
    if spec.source == "gt" and identity.kind != "base":
        raise ValueError("GT methods adapt from the epoch-25 base that built their pool")
    run_dir = layout.object_run(runs, method, obj_id)
    assert recipe.adapt is not None
    run = RunSpec(method, spec, recipe.adapt.gate, run_dir)
    return execute_run(
        recipe,
        run,
        obj_id,
        root,
        runs,
        identity,
        predictor_factory=predictor_factory,
        consent=consent,
        labels_only=labels_only,
        overrides=overrides,
        force=force,
    )


def execute_run(
    recipe: Recipe,
    run: RunSpec,
    obj_id: int,
    root: Path,
    runs: Path,
    teacher: TeacherIdentity,
    *,
    predictor_factory: PredictorFactory,
    consent: bool,
    labels_only: bool,
    overrides: Mapping[str, Any] | None = None,
    receipt_extra: Mapping[str, Any] | None = None,
    force: bool = False,
) -> MethodResult:
    """Run one validated adaptation run: identity, labels, skip or train, receipt.

    ``run_method`` checks the method, object, consent and teacher contract first; a follow-up
    round (``doper.training.followups``) builds its own ``RunSpec`` and teacher. The run holds
    the run directory's sibling lock (``doper.bop.sources.sibling_lock``) throughout, so a
    parallel job on the same directory (a labels-only component of a combined run, a forced
    restart) waits instead of deleting the run's work in progress.

    Args:
        recipe: Dataset recipe with an ``adapt`` section.
        run: The run.
        obj_id: The object.
        root: Prepared dataset root.
        runs: Runs root.
        teacher: The pinned teacher (``require_teacher``).
        predictor_factory: Loads the teacher into a pseudo-label ``Predictor``.
        consent: Transductive consent.
        labels_only: Stop after the verified pool (a pool below the minimum is still skipped).
        overrides: Config keys replacing the adaptation template.
        receipt_extra: Extra ``completed.json`` fields of a newly trained student.
        force: Remove the run directory first (under the run lock).

    Returns:
        Where the run stopped.

    Raises:
        ValueError: For a broken pool or student contract, or a run started with another
            identity.
    """
    spec, run_dir, method = run.method, run.run_dir, run.name
    with sources.file_lock(sources.sibling_lock(run_dir), busy=f"another job is running {run_dir}"):
        if force and run_dir.exists():
            shutil.rmtree(run_dir)
        ctx = _Context(
            recipe,
            run.name,
            run.method,
            obj_id,
            Path(root),
            Path(runs),
            run.run_dir,
            teacher,
            predictor_factory,
            consent,
            run.gate,
            run.identity,
            run.reuse_from,
        )
        _publish_identity(ctx)
        labels_root, n = _labels(ctx)
        minimum = spec.minimum_for(obj_id)
        if n < minimum:
            if (run_dir / layout.CHECKPOINT_PATH.parent).exists():
                raise ValueError(f"{run_dir} holds a student although its pool is below {minimum}")
            publish_json(
                run_dir / layout.SKIPPED_NAME,
                {
                    "reason": f"fewer than {minimum} accepted labels",
                    "labels": n,
                    "minimum_labels": minimum,
                },
            )
            log(f"object {obj_id}: {method} skipped: {n} labels, minimum {minimum}")
            return MethodResult(method, obj_id, "skipped", n, run_dir)
        if labels_only:
            return MethodResult(method, obj_id, "labels_ready", n, run_dir)
        if (run_dir / layout.COMPLETED_NAME).is_file():
            student = _completed_student(ctx)
        else:
            student = _complete(ctx, n, overrides or {}, labels_root, receipt_extra or {})
        return MethodResult(method, obj_id, "completed", n, run_dir, student)


# --------------------------------------------------------------------------------------
# T-LESS multisensor stage
# --------------------------------------------------------------------------------------
def _multisensor(recipe: Recipe) -> MultisensorSpec:
    if recipe.adapt is None or recipe.adapt.multisensor is None:
        raise ValueError(f"{recipe.name} has no multisensor stage (adapt.multisensor)")
    return recipe.adapt.multisensor


def _per_batch(spec: MultisensorSpec, batch_size: int) -> tuple[int, ...]:
    """Samples per batch of PBR, PrimeSense, Kinect and Canon."""
    pbr = int(batch_size) - sum(spec.mix_per_batch)
    if pbr < 1:
        raise ValueError(
            f"mix_per_batch {list(spec.mix_per_batch)} leaves no PBR samples per batch of "
            f"{batch_size}"
        )
    return (pbr, *spec.mix_per_batch)


def multisensor_config(
    recipe: Recipe,
    paths: Mapping[str, Any],
    *,
    overrides: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], LaunchSpec]:
    """Resolve the training config of one object's multisensor student.

    The base stage's config, replaced by ``adapt.multisensor.config``, plus the three sensor
    roots as ``mix_data_root`` (PrimeSense, Kinect, Canon), ``mix_per_batch``, the min-mix
    checks and ``max_iters_per_epoch: 0`` (uncapped). The student initializes from the teacher
    (``require_init_from``).

    Args:
        recipe: Dataset recipe with ``adapt.multisensor``.
        paths: ``data_root``, ``keypoints_3d``, ``work_dir``, ``teacher`` and ``sensors`` (the
            three prepared sensor roots in ``doper.bop.multisensor.SENSORS`` order).
        overrides: Config keys replacing the template (``check_adapt_overrides``).

    Returns:
        The resolved config-file mapping and its ``LaunchSpec``.

    Raises:
        ValueError: For a recipe without the stage, a missing path, not three sensor roots, a
            refused override, a backbone other than the base's or a batch without PBR.
    """
    spec = _multisensor(recipe)
    missing = sorted(set(MULTISENSOR_PATH_KEYS) - set(paths))
    if missing:
        raise ValueError(f"multisensor_config needs the paths {missing}")
    sensors = [str(p) for p in paths["sensors"]]
    if len(sensors) != MULTISENSOR_SOURCES:
        raise ValueError(f"the multisensor stage mixes {MULTISENSOR_SOURCES} sensor roots")
    check_adapt_overrides(overrides or {})
    config = {
        **recipe.base.config,
        **spec.config,
        "mix_data_root": sensors,
        "mix_per_batch": list(spec.mix_per_batch),
        "max_iters_per_epoch": 0,
        "min_mix_keypoint_samples": spec.min_mix_keypoint_samples,
        "min_mix_mask_only_samples": spec.min_mix_mask_only_samples,
    }
    stage = TrainSpec(config, LaunchSpec(init_from=str(paths["teacher"]), require_init_from=True))
    train_paths = {key: str(paths[key]) for key in PATH_KEYS}
    resolved, _ = training_namespace(stage, train_paths, overrides)
    _, launch = resolve_config(stage, train_paths, overrides)
    if resolved["backbone"] != recipe.base.config["backbone"]:
        raise ValueError("the adaptation backbone must equal the base backbone")
    _per_batch(spec, resolved["batch_size"])
    return resolved, launch


def multisensor_batches(
    recipe: Recipe,
    pbr_samples: int,
    sensor_samples: Sequence[int],
    *,
    batch_size: int | None = None,
) -> int:
    """Batches of one uncapped multisensor epoch: ``max_i ceil(n_i / b_i)`` over the streams.

    This is how ``doper.training.data`` sizes an all-instance mixed epoch, so every retained
    sample of every stream is seen and the smaller streams are oversampled (object 1: 19,294
    PBR samples at 10 per batch and 1,296 captures per sensor at 2 give 1,930 batches).

    Args:
        recipe: Dataset recipe with ``adapt.multisensor``.
        pbr_samples: PBR training samples.
        sensor_samples: Samples of PrimeSense, Kinect and Canon.
        batch_size: The batch size; None takes the recipe's.

    Returns:
        Batches (optimizer updates) per epoch.
    """
    spec = _multisensor(recipe)
    if batch_size is None:
        batch_size = int({**recipe.base.config, **spec.config}["batch_size"])
    per_batch = _per_batch(spec, batch_size)
    sizes = (int(pbr_samples), *(int(n) for n in sensor_samples))
    if len(sizes) != len(per_batch):
        raise ValueError(f"expected {MULTISENSOR_SOURCES} sensor sample counts, got {sizes[1:]}")
    return max(math.ceil(n / b) for n, b in zip(sizes, per_batch, strict=True))


def verify_multisensor_student(
    run_dir: Path,
    sensor_samples: Sequence[int],
    spec: MultisensorSpec,
    *,
    batch_size: int,
    epochs: int = STUDENT_EPOCHS,
    mix_roots: Sequence[str | Path] | None = None,
) -> list[str]:
    """Audit a trained multisensor student endpoint.

    Checks: ``train_sources`` holds four streams, PBR with keypoint supervision then PrimeSense,
    Kinect and Canon with the prepared capture counts, each fully keypoint-supervised (GT
    keypoints, no mask-only sample); ``work/last.pth`` is epoch index ``epochs - 1`` with
    ``global_step == epochs * max_i ceil(n_i / b_i)`` (the uncapped epoch over those streams);
    finite trained model and optimizer; the epoch log; the keypoint fingerprint. With
    ``mix_roots``, a stream that records its root must name ``mix_roots[i]`` (its full path,
    or the final component ``doper.train`` keeps in public provenance), so a reordered mix
    does not pass on counts alone.

    Args:
        run_dir: The object's multisensor run directory.
        sensor_samples: Captures of PrimeSense, Kinect and Canon (``READY.json`` frames).
        spec: The recipe's multisensor stage.
        batch_size: The batch size it trained with.
        epochs: The adaptation epochs.
        mix_roots: The config's ``mix_data_root`` (PrimeSense, Kinect, Canon).

    Returns:
        The faults; empty for a sound student.
    """
    from ..bop.multisensor import SENSORS
    from ..checkpoint_io import load_checkpoint

    path = Path(run_dir) / layout.CHECKPOINT_PATH
    if not path.is_file():
        return [f"no student checkpoint {path}"]
    checkpoint = load_checkpoint(path)
    streams = (checkpoint.get("training_provenance") or {}).get("train_sources") or []
    faults: list[str] = []
    steps = None
    if (
        len(streams) != 1 + MULTISENSOR_SOURCES
        or not streams[0].get("keypoint_samples")
        or not all(isinstance(s.get("samples"), int) for s in streams)
    ):
        faults.append(
            "training sources are not the four streams (PBR rehearsal, PrimeSense, Kinect, "
            "Canon) with recorded sample counts"
        )
    else:
        faults.extend(_stream_root_faults(streams[1:], mix_roots))
        for sensor, n, stream in zip(SENSORS, sensor_samples, streams[1:], strict=True):
            counts = (
                stream.get("samples"),
                stream.get("keypoint_samples"),
                stream.get("mask_only_samples"),
            )
            if counts != (n, n, 0):
                faults.append(
                    f"{sensor} supervision (samples, keypoint, mask-only) {counts} differs from "
                    f"the {n} GT-keypoint captures"
                )
        per_batch = _per_batch(spec, batch_size)
        sizes = (int(streams[0]["samples"]), *sensor_samples)
        steps = epochs * max(math.ceil(n / b) for n, b in zip(sizes, per_batch, strict=True))
    faults.extend(_endpoint_faults(path, checkpoint, epochs, steps))
    faults.extend(_fingerprint_faults(Path(run_dir), checkpoint))
    return faults


def _stream_root_faults(
    streams: Sequence[Mapping[str, Any]], mix_roots: Sequence[str | Path] | None
) -> list[str]:
    """Faults of real streams whose recorded root is not the configured mix root."""
    if mix_roots is None:
        return []
    faults = []
    for index, (stream, root) in enumerate(zip(streams, mix_roots, strict=True)):
        recorded = stream.get("root")
        if recorded is None:
            continue
        expected = Path(root)
        if str(recorded) not in (str(expected), expected.name):
            faults.append(f"real stream {index + 1} records root {recorded}, not {expected}")
    return faults


def _multisensor_review(
    out: Path, obj_id: int, accept_unreviewed: bool, allow_partial: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The verified ``READY.json`` of a prepared root and the review provenance."""
    from ..bop import multisensor, review

    try:
        ready = multisensor.verify(out, allow_partial=allow_partial)
    except review.ReviewError as error:
        if not accept_unreviewed:
            raise review.ReviewError(
                f"{error}; inspect the nine overlays and run `doper review tless --obj-id "
                f"{obj_id} --reviewer NAME`, or pass "
                "--accept-unreviewed (recorded in completed.json)"
            ) from error
        log(f"WARNING: training on {out} without a current review (--accept-unreviewed): {error}")
        ready = multisensor.verify_prepared(out)
        return ready, {"reviewer": None, "accepted_unreviewed": True, "reviewed_sha256": None}
    reviewed = out / review.REVIEWED_NAME
    record = {
        "reviewer": review.read(out)["reviewer"],
        "accepted_unreviewed": False,
        "reviewed_sha256": sha256_file(reviewed),
    }
    return ready, record


@dataclass(frozen=True)
class ReservedTest:
    """The reserved ``test_primesense`` population as the leakage guard reads it.

    Attributes:
        sha256: The population identity (``doper.bop.multisensor.population_sha256``).
        images: RGB SHA-256 to ``scene/frame``.
    """

    sha256: str
    images: Mapping[str, str]


def reserved_test(root: Path) -> ReservedTest:
    """Hash the reserved ``<root>/data/test_primesense`` population once (every test image).

    A caller adapting several objects computes it once and passes it to ``run_multisensor``.

    Raises:
        ValueError: If the split is missing.
    """
    from ..bop import multisensor
    from ..bop.extract import data_dir

    test_dir = data_dir(Path(root)) / multisensor.TEST_SPLIT
    if not test_dir.is_dir():
        raise ValueError(
            f"the reserved {multisensor.TEST_SPLIT} population {test_dir} is missing; the "
            "multisensor stage re-checks every training image against it"
        )
    reserved = multisensor.reserved_population(test_dir)
    images = {f["sha256"]: f"{f['scene']}/{f['frame']}" for f in reserved["frames"]}
    return ReservedTest(multisensor.population_sha256(reserved), images)


def _require_unreserved(out: Path, ready: Mapping[str, Any], reserved: ReservedTest) -> str:
    """Refuse a training image whose bytes are a reserved test image; return the population hash.

    The prepared root was checked against the reserved ``test_primesense`` population when it
    was written; this re-checks every training RGB against the population as it is now, and
    requires it to be the population the preparation saw.
    """
    from ..bop import multisensor

    hashes = reserved.images
    for sensor in multisensor.SENSORS:
        source = _read_json(out / sensor / multisensor.SOURCE_NAME)
        for row in source["records"]:
            if row["rgb_sha256"] in hashes:
                raise ValueError(
                    f"reserved {multisensor.TEST_SPLIT} image entered training: {sensor} frame "
                    f"{row['frame']} ({out / sensor / row['image_file']}) has the bytes of test "
                    f"image {hashes[row['rgb_sha256']]}"
                )
    digest = reserved.sha256
    if digest != ready["reserved_test_population_sha256"]:
        raise ValueError(
            f"the reserved {multisensor.TEST_SPLIT} population changed since {out} was prepared; "
            "prepare the root again against the current test split"
        )
    return digest


def _multisensor_identity(
    recipe: Recipe,
    obj_id: int,
    run_dir: Path,
    out: Path,
    ready: Mapping[str, Any],
    teacher: TeacherIdentity,
) -> dict[str, Any]:
    from ..bop import multisensor

    spec = _multisensor(recipe)
    return {
        "recipe": recipe.name,
        "method": MULTISENSOR_ARM,
        "obj_id": obj_id,
        "source": MULTISENSOR_ARM,
        "supervision": ready["supervision"],
        "mask_policy": ready["mask_policy"],
        "sensors": list(multisensor.SENSORS),
        "sensor_frames": {s: ready["sensors"][s]["frames"] for s in multisensor.SENSORS},
        "mix_per_batch": list(spec.mix_per_batch),
        "real_root": os.path.relpath(out, run_dir),
        "ready_sha256": sha256_file(out / multisensor.READY_NAME),
        "reserved_test_population_sha256": ready["reserved_test_population_sha256"],
        "teacher": os.path.relpath(teacher.path, run_dir),
        "teacher_kind": teacher.kind,
        "teacher_sha256": teacher.sha256,
        "base_teacher_sha256": teacher.base_sha256,
        "keypoints_sha256": ready["keypoints_sha256"],
        "transductive_test_adaptation": False,
        "test_gt_used": False,
        "test_targets_used": False,
    }


def _check_ready(ready: Mapping[str, Any], obj_id: int, base_run: Path) -> list[int]:
    """The capture counts of a prepared root of ``obj_id`` with the base's keypoints."""
    from ..bop import multisensor

    if ready.get("object") != obj_id:
        raise ValueError(f"the multisensor root records object {ready.get('object')}, not {obj_id}")
    if ready.get("supervision") != multisensor.SUPERVISION:
        raise ValueError(f"the multisensor root must carry {multisensor.SUPERVISION} supervision")
    if ready["keypoints_sha256"] != sha256_file(base_run / layout.KEYPOINTS_PATH):
        raise ValueError(
            "the multisensor root was prepared with other keypoints than the base run's; the "
            "student must keep the base's keypoint order"
        )
    return [int(ready["sensors"][s]["frames"]) for s in multisensor.SENSORS]


def multisensor_paths(root: Path, runs: Path, obj_id: int) -> dict[str, Any]:
    """The ``multisensor_config`` paths of one object (dry run and run share them).

    ``teacher`` is the object's epoch-25 base checkpoint; ``sensors`` the prepared PrimeSense,
    Kinect and Canon roots under ``<root>/multisensor/obj_XXXXXX``.
    """
    from ..bop import multisensor
    from ..cli.common import DATASET_DIR, object_dir

    run_dir = layout.object_run(Path(runs), MULTISENSOR_ARM, obj_id)
    out = multisensor.output_dir(Path(root), obj_id)
    return {
        "data_root": object_dir(Path(root), obj_id) / DATASET_DIR,
        "keypoints_3d": run_dir / layout.KEYPOINTS_PATH,
        "work_dir": run_dir / layout.CHECKPOINT_PATH.parent,
        "teacher": layout.object_run(Path(runs), BASE_ARM, obj_id) / layout.CHECKPOINT_PATH,
        "sensors": [out / sensor for sensor in multisensor.SENSORS],
    }


def run_multisensor(
    recipe: Recipe,
    obj_id: int,
    root: Path,
    runs: Path,
    *,
    accept_unreviewed: bool = False,
    force: bool = False,
    overrides: Mapping[str, Any] | None = None,
    allow_partial: bool = False,
    reserved: ReservedTest | None = None,
) -> MethodResult:
    """Train one T-LESS object's multisensor student (``<runs>/multisensor/obj_XXXXXX``).

    Before training: the completed epoch-25 base is the teacher; the prepared root
    ``<root>/multisensor/obj_XXXXXX`` needs a current human review (``doper review``) unless
    ``accept_unreviewed``, which ``completed.json`` records; its labels must be unchanged and
    use the base's keypoints; no training image may equal a reserved ``test_primesense`` image,
    and that population must be the one the preparation checked. A completed run is verified and
    returned; an interrupted one resumes.

    Args:
        recipe: Dataset recipe with ``adapt.multisensor``.
        obj_id: The object.
        root: Prepared dataset root (PBR objects, ``multisensor/``, ``data/test_primesense``).
        runs: Runs root.
        accept_unreviewed: Train without a current review (recorded).
        force: Restart the object's multisensor run.
        overrides: Config keys replacing the template (``check_adapt_overrides``).
        allow_partial: Accept a ``partial_smoke_only`` root (smoke tests only).
        reserved: The reserved test population (``reserved_test``); None hashes it here.

    Returns:
        The completed run (``labels``: real samples over the three sensors).

    Raises:
        doper.bop.review.ReviewError: For a missing or stale review without
            ``accept_unreviewed``.
        ValueError: For another recipe or object, a broken base, root or student contract, a
            reserved test image, or a run started with another identity.
    """
    from ..bop import multisensor
    from ..checkpoint_io import load_checkpoint
    from ..cli.common import prepared_receipt

    spec = _multisensor(recipe)
    if obj_id not in recipe.trainable_objects:
        raise ValueError(f"the multisensor stage runs for {recipe.name} objects, not {obj_id}")
    root, runs = Path(root), Path(runs)
    base_run = layout.object_run(runs, BASE_ARM, obj_id)
    teacher = require_teacher(base_run)
    if teacher.obj_id != obj_id:
        raise ValueError(f"the base run of object {obj_id} records object {teacher.obj_id}")
    run_dir = layout.object_run(runs, MULTISENSOR_ARM, obj_id)
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    if (run_dir / layout.COMPLETED_NAME).is_file():
        done = _read_json(run_dir / layout.COMPLETED_NAME)
        last = run_dir / layout.CHECKPOINT_PATH
        if not last.is_file() or sha256_file(last) != done.get("student_sha256"):
            raise ValueError(f"{run_dir} student changed since completed.json; pass --force")
        if done.get("teacher_sha256") != teacher.sha256:
            raise ValueError(f"{run_dir} descends from another base; pass --force")
        return MethodResult(
            MULTISENSOR_ARM, obj_id, "completed", done["labels"], run_dir, done["student_sha256"]
        )
    out = multisensor.output_dir(root, obj_id)
    ready, review_record = _multisensor_review(out, obj_id, accept_unreviewed, allow_partial)
    frames = _check_ready(ready, obj_id, base_run)
    population = _require_unreserved(out, ready, reserved or reserved_test(root))
    prepared = prepared_receipt(root, obj_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    _publish_once(run_dir, _multisensor_identity(recipe, obj_id, run_dir, out, ready, teacher))
    _run_keypoints(run_dir, base_run)
    paths = multisensor_paths(root, runs, obj_id)  # its teacher is ``teacher.path``
    resolved, _ = multisensor_config(recipe, paths, overrides=overrides)
    _publish_config(run_dir, resolved)
    _launch(run_dir, resolved, teacher.sha256, MULTISENSOR_ARM, obj_id)
    epochs, batch_size = int(resolved["epochs"]), int(resolved["batch_size"])
    faults = verify_multisensor_student(
        run_dir,
        frames,
        spec,
        batch_size=batch_size,
        epochs=epochs,
        mix_roots=resolved["mix_data_root"],
    )
    if faults:
        raise ValueError(f"multisensor student of object {obj_id} failed its audit: {faults}")
    last = run_dir / layout.CHECKPOINT_PATH
    checkpoint = load_checkpoint(last)
    streams = checkpoint["training_provenance"]["train_sources"]
    student = sha256_file(last)
    publish_json(
        run_dir / layout.COMPLETED_NAME,
        _finite(
            {
                "recipe": recipe.name,
                "stage": MULTISENSOR_ARM,
                "method": MULTISENSOR_ARM,
                "obj_id": obj_id,
                "completed_epochs": epochs,
                "labels": sum(frames),
                "sensor_samples": dict(zip(multisensor.SENSORS, frames, strict=True)),
                "batches_per_epoch": multisensor_batches(
                    recipe, streams[0]["samples"], frames, batch_size=batch_size
                ),
                "global_step": checkpoint["global_step"],
                "teacher_kind": teacher.kind,
                "teacher_sha256": teacher.sha256,
                "base_teacher_sha256": teacher.base_sha256,
                "init_from_sha256": teacher.sha256,
                "student": str(layout.CHECKPOINT_PATH),
                "student_sha256": student,
                "checkpoint_sha256": student,
                "keypoints_sha256": sha256_file(run_dir / layout.KEYPOINTS_PATH),
                "ready_sha256": sha256_file(out / multisensor.READY_NAME),
                "review": review_record,
                "reserved_test_population_sha256": population,
                "source_supervision": streams,
                "final_synthetic_metrics": checkpoint.get("metrics"),
                "prepared_hashes": prepared["hashes"],
            }
        ),
    )
    log(f"object {obj_id}: multisensor student complete in {run_dir}")
    return MethodResult(MULTISENSOR_ARM, obj_id, "completed", sum(frames), run_dir, student)
