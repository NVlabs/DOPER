# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint saving, resume, weight initialization and run provenance."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pickle
import subprocess
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from .. import __version__
from ..cad_curves import checkpoint_curve_metadata, require_matching_curve_metadata
from ..checkpoint_io import atomic_torch_save, checkpoint_configs, load_checkpoint
from ..config import DoperConfig
from ..console import log
from ..model import DoperNet
from ..rotation_aux import require_matching_rotation_metadata
from .data import supervision_counts

# Read size when hashing checkpoint and config files.
_HASH_CHUNK_BYTES = 1024 * 1024
# Minimum fraction of shared (non-head) parameters an init checkpoint must load.
_MIN_SHARED_WEIGHT_FRACTION = 0.1
# Fields that must agree between an init checkpoint and the model being trained.
_SHARED_ARCHITECTURE_FIELDS = (
    "backbone",
    "num_classes",
    "num_keypoints",
    "neck_channels",
    "head_channels",
)
# Root of the source checkout when this package is imported from one.
_REPO_ROOT = Path(__file__).resolve().parents[2]
# Command-line destinations whose values are filesystem paths.
_PATH_ARGUMENTS = frozenset(
    {
        "config",
        "data_root",
        "mix_data_root",
        "keypoint_only_mix_data_root",
        "work_dir",
        "keypoints_3d",
        "curve_schema",
        "background_dir",
        "background_splits_json",
        "composite_source_root",
        "fp_mine_json",
        "resume",
        "init_from",
    }
)
# ``(section, field)`` pairs of the resolved config that hold filesystem paths.
_PATH_CONFIG_FIELDS = (
    ("data", "root"),
    ("data", "keypoints_3d_json"),
    ("data", "curve_schema_json"),
    ("train", "work_dir"),
    ("background", "background_dir"),
    ("background", "splits_json"),
    ("background", "composite_source_root"),
    ("background", "fp_mine_json"),
)


def sha256_file(path: str | None) -> str | None:
    """Return the SHA-256 hex digest of ``path``, or None if it is not a file.

    Args:
        path: File to hash; None or a missing file yields None.

    Returns:
        The hex digest, or None.
    """
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_validation_checkpoints(
    ckpt: dict[str, object], kp_err_px: float, best_kp: float, work_dir: str | Path
) -> float:
    """Save ``last.pth`` and, on a new best keypoint error, ``best.pth``.

    ``best_kp`` is updated before ``last.pth`` is written, so resuming from it
    never forgets a best epoch and overwrites ``best.pth`` with a worse model.

    Args:
        ckpt: Checkpoint payload; its ``best_kp`` entry is set here.
        kp_err_px: Validation keypoint error of this epoch.
        best_kp: Best keypoint error before this epoch.
        work_dir: Directory receiving the checkpoints.

    Returns:
        The best keypoint error so far.
    """
    is_best = kp_err_px < best_kp
    if is_best:
        best_kp = kp_err_px
    ckpt["best_kp"] = best_kp
    atomic_torch_save(ckpt, Path(work_dir) / "last.pth")
    if is_best:
        atomic_torch_save(ckpt, Path(work_dir) / "best.pth")
        log(f"new best kp_err_px={best_kp:.3f}")
    return best_kp


def resumed_best_kp(ckpt: dict[str, Any], resume_path: str | Path) -> float:
    """Return the best keypoint error to continue from when resuming.

    Checkpoints written before :func:`save_validation_checkpoints` could store a
    stale ``best_kp`` in ``last.pth``; the ``best.pth`` beside it is authoritative,
    even when it is outside the new run's work directory. An unreadable
    ``best.pth`` is ignored with a warning so it cannot block the resume.

    Args:
        ckpt: The loaded resume checkpoint.
        resume_path: Path the checkpoint was loaded from.

    Returns:
        The best keypoint error recorded by the resumed run.
    """
    best_kp = float(ckpt.get("best_kp", float("inf")))
    best_path = Path(resume_path).resolve().with_name("best.pth")
    if best_path.is_file():
        try:
            best = load_checkpoint(best_path, "cpu")
        except (
            OSError,
            EOFError,
            RuntimeError,
            IndexError,
            ValueError,
            pickle.UnpicklingError,
        ) as exc:
            warnings.warn(f"ignoring unreadable {best_path}: {exc}", stacklevel=2)
            return best_kp
        if isinstance(best, dict):
            best_kp = min(best_kp, float(best.get("best_kp", float("inf"))))
    return best_kp


def trim_epoch_timing(path: str | Path, start_epoch: int) -> None:
    """Drop timing rows for epochs that a resumed run will repeat.

    Rows that do not parse, such as a line cut short when a run was killed
    mid-write, are dropped too so they cannot block the resume.

    Args:
        path: The ``epoch_timing.jsonl`` file; a missing file is ignored.
        start_epoch: First epoch the resumed run will train.
    """
    path = Path(path)
    if not path.is_file():
        return
    rows = []
    for line in path.read_text().splitlines(keepends=True):
        try:
            if json.loads(line)["epoch"] < start_epoch:
                rows.append(line)
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    path.write_text("".join(rows))


def _git_output(*command: str) -> str:
    """Run a read-only git command in the source checkout and return stdout."""
    return subprocess.check_output(
        ["git", *command],
        cwd=_REPO_ROOT,
        text=True,
        stderr=subprocess.DEVNULL,
    )


def provenance() -> dict[str, object]:
    """Describe the code that produced a training run.

    The commit hash and dirty flag are included only when the package is
    imported from a git checkout of the DOPER repository itself. Paths, branch
    names and remotes are never recorded.

    Returns:
        ``doper_version`` and, from a source checkout, ``git_commit`` and
        ``git_dirty``.
    """
    result: dict[str, object] = {"doper_version": __version__}
    try:
        toplevel = _git_output("rev-parse", "--show-toplevel").strip()
        if Path(toplevel).resolve() != _REPO_ROOT:
            return result
        commit = _git_output("rev-parse", "HEAD").strip()
        status = _git_output("status", "--short").splitlines()
    except (OSError, subprocess.CalledProcessError):
        return result
    result["git_commit"] = commit
    result["git_dirty"] = bool(status)
    return result


def _source_supervision(
    cfg: DoperConfig, train_roots: list[str], source_sets: Sequence[Any]
) -> list[dict[str, object]]:
    """Summarize the supervision each training source carries."""
    source_supervision = []
    for root, dataset in zip(train_roots, source_sets, strict=False):
        keypoints, masks, mask_only = supervision_counts(dataset, cfg.model.num_keypoints)
        source_supervision.append(
            {
                "root": root,
                "samples": len(dataset),
                "annotated_image_records": getattr(dataset, "image_count", None),
                "instance_sampling": cfg.data.instance_sampling,
                "mask_target_mode": cfg.data.mask_target_mode,
                "keypoint_samples": keypoints,
                "mask_samples": masks,
                "mask_only_samples": mask_only,
            }
        )
    return source_supervision


def _path_name(value: object) -> object:
    """Reduce a path (or list of paths) to its final component; pass others through."""
    if isinstance(value, (list, tuple)):
        return [_path_name(item) for item in value]
    if isinstance(value, str) and value:
        return os.path.basename(os.path.normpath(value))
    return value


def _strip_absolute_paths(value: Any) -> Any:  # noqa: ANN401
    """Recursively replace absolute or home-relative path strings with their basename."""
    if isinstance(value, dict):
        return {key: _strip_absolute_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strip_absolute_paths(item) for item in value]
    if isinstance(value, str) and (os.path.isabs(value) or value.startswith("~")):
        return _path_name(value)
    return value


def _sanitize_provenance(record: dict[str, Any]) -> dict[str, Any]:
    """Return ``record`` with path-valued entries reduced to their basename.

    Payload keys are unchanged. Known path arguments, config fields, training
    roots, dataset-summary keys and checkpoint/config file paths are reduced to
    their final component; any remaining absolute path is reduced as well.

    Args:
        record: Full training provenance record.

    Returns:
        A sanitized copy of ``record``.
    """
    sanitized = dict(record)
    sanitized["arguments"] = {
        key: _path_name(value) if key in _PATH_ARGUMENTS else value
        for key, value in record["arguments"].items()
    }
    resolved = {
        section: dict(fields) if isinstance(fields, dict) else fields
        for section, fields in record["resolved_config"].items()
    }
    for section, field in _PATH_CONFIG_FIELDS:
        if isinstance(resolved.get(section), dict) and field in resolved[section]:
            resolved[section][field] = _path_name(resolved[section][field])
    sanitized["resolved_config"] = resolved
    sanitized["train_sources"] = [
        {**source, "root": _path_name(source["root"])} for source in record["train_sources"]
    ]
    sanitized["dataset_summaries"] = {
        str(_path_name(root)): summary for root, summary in record["dataset_summaries"].items()
    }
    for key in ("init_checkpoint", "resume_checkpoint", "config_file"):
        sanitized[key] = {**record[key], "path": _path_name(record[key]["path"])}
    return _strip_absolute_paths(sanitized)


def write_training_provenance(
    args: argparse.Namespace,
    cfg: DoperConfig,
    train_roots: list[str],
    source_sets: Sequence[Any],
) -> dict[str, object]:
    """Write ``training_provenance.json`` into the work directory.

    By default path-valued entries are reduced to their basename so the record
    (also embedded in every checkpoint) carries no local filesystem layout.
    ``args.private_provenance`` keeps the full paths for internal runs.

    Args:
        args: Parsed command-line arguments.
        cfg: Resolved training config.
        train_roots: Training roots, primary first.
        source_sets: One dataset per training root.

    Returns:
        The provenance record, also embedded in every checkpoint.
    """
    source_supervision = _source_supervision(cfg, train_roots, source_sets)
    dataset_summaries = {}
    for root in train_roots:
        summary_path = os.path.join(root, "summary.json")
        if os.path.isfile(summary_path):
            with open(summary_path) as handle:
                dataset_summaries[root] = json.load(handle)
    training_provenance: dict[str, Any] = {
        "arguments": vars(args),
        "resolved_config": dataclasses.asdict(cfg),
        "train_sources": source_supervision,
        "dataset_summaries": dataset_summaries,
        "init_checkpoint": {
            "path": args.init_from,
            "sha256": sha256_file(args.init_from),
        },
        "resume_checkpoint": {
            "path": args.resume,
            "sha256": sha256_file(args.resume),
        },
        "config_file": {
            "path": args.config,
            "sha256": sha256_file(args.config),
        },
        "git": provenance(),
    }
    if not getattr(args, "private_provenance", False):
        training_provenance = _sanitize_provenance(training_provenance)
    with open(os.path.join(cfg.train.work_dir, "training_provenance.json"), "w") as handle:
        json.dump(training_provenance, handle, indent=2, default=str)
        handle.write("\n")
    return training_provenance


def keypoint_head_from_state(state: dict) -> str:
    """Infer the keypoint head type from state-dict key names.

    Args:
        state: Model state dict.

    Returns:
        ``"simcc"``, ``"heatmap"`` or ``"unknown"``.
    """
    keys = tuple(state)
    if any(key.startswith("kpt_head.cls_") for key in keys):
        return "simcc"
    if any(key.startswith(("kpt_head.neck.", "kpt_head.tower.", "kpt_head.out.")) for key in keys):
        return "heatmap"
    return "unknown"


def _read_init_checkpoint(checkpoint_path: str, device: str) -> tuple[dict, dict]:
    """Load an init checkpoint and return its model state and ``cfg.model``."""
    label = f"initialization checkpoint {checkpoint_path!r}"
    checkpoint, _, checkpoint_model_cfg = checkpoint_configs(
        load_checkpoint(checkpoint_path, device), label
    )
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"{label} has no model weights")
    return state, checkpoint_model_cfg


def _replaces_simcc_head(state: dict, checkpoint_model_cfg: dict, cfg: DoperConfig) -> bool:
    """Return whether a SimCC checkpoint initializes a heatmap-head model."""
    state_head = keypoint_head_from_state(state)
    configured_head = checkpoint_model_cfg.get("keypoint_head")
    if configured_head is not None and state_head != "unknown" and configured_head != state_head:
        raise RuntimeError(
            "initialization checkpoint cfg/model disagree on keypoint head: "
            f"cfg={configured_head!r}, weights={state_head!r}"
        )
    source_head = configured_head or state_head
    return source_head == "simcc" and cfg.model.keypoint_head == "heatmap"


def _shape_description(value: object) -> object:
    """Describe a state value by shape, or by type when it has none."""
    shape = getattr(value, "shape", None)
    return tuple(shape) if shape is not None else type(value).__name__


def _check_shared_fraction(model_state: dict, matching_state: dict) -> float:
    """Require that a meaningful share of the non-head weights was loaded."""
    shared_keys = [
        key
        for key in model_state
        if not key.startswith(("kpt_head.", "curve_head.", "rotation_head.", "rotation_bins"))
    ]
    shared_numel = sum(model_state[key].numel() for key in shared_keys)
    loaded_shared_numel = sum(
        model_state[key].numel() for key in matching_state if key in shared_keys
    )
    loaded_shared_fraction = loaded_shared_numel / max(1, shared_numel)
    if loaded_shared_numel == 0 or loaded_shared_fraction < _MIN_SHARED_WEIGHT_FRACTION:
        raise RuntimeError(
            "initialization checkpoint loaded essentially no meaningful shared "
            f"weights ({loaded_shared_numel}/{shared_numel} parameters)"
        )
    return loaded_shared_fraction


def load_init_weights(
    model: DoperNet,
    cfg: DoperConfig,
    checkpoint_path: str,
    device: str,
) -> None:
    """Load a compatible initialization checkpoint with an explicit report.

    Only shape-compatible tensors are loaded. A SimCC-to-heatmap head swap and a
    newly added curve or rotation auxiliary head are the permitted architecture differences.

    Args:
        model: Model to initialize in place.
        cfg: Config of ``model``.
        checkpoint_path: DOPER checkpoint (or bare state dict) to load from.
        device: Device the checkpoint tensors are mapped to.

    Raises:
        ValueError: If the checkpoint is malformed.
        RuntimeError: If it is architecturally incompatible with ``model``.
    """
    state, checkpoint_model_cfg = _read_init_checkpoint(checkpoint_path, device)
    source_has_rotation = bool(checkpoint_model_cfg.get("rotation_aux_enabled", False))
    add_rotation_head = cfg.model.rotation_aux_enabled and not source_has_rotation
    if source_has_rotation:
        require_matching_rotation_metadata(checkpoint_model_cfg, cfg.model)
    source_curve_metadata = checkpoint_curve_metadata(checkpoint_model_cfg, state.keys())
    source_has_curve = bool(source_curve_metadata.get("curve_head_enabled", False))
    add_curve_head = cfg.model.curve_head_enabled and not source_has_curve
    if source_has_curve:
        require_matching_curve_metadata(source_curve_metadata, cfg.model)
    replace_simcc_head = _replaces_simcc_head(state, checkpoint_model_cfg, cfg)
    config_mismatches = [
        (
            f"{field_name}: checkpoint={checkpoint_model_cfg[field_name]!r}, "
            f"model={getattr(cfg.model, field_name)!r}"
        )
        for field_name in _SHARED_ARCHITECTURE_FIELDS
        if field_name in checkpoint_model_cfg
        and checkpoint_model_cfg[field_name] != getattr(cfg.model, field_name)
    ]
    model_state = model.state_dict()
    unexpected_keys = sorted(key for key in state if key not in model_state)
    shape_mismatches = sorted(
        (f"{key}: checkpoint={_shape_description(value)}, model={tuple(model_state[key].shape)}")
        for key, value in state.items()
        if key in model_state
        and (not torch.is_tensor(value) or tuple(value.shape) != tuple(model_state[key].shape))
    )
    matching_state = {
        key: value
        for key, value in state.items()
        if key in model_state
        and torch.is_tensor(value)
        and tuple(value.shape) == tuple(model_state[key].shape)
    }
    incompatible = model.load_state_dict(matching_state, strict=False)
    missing_keys = sorted(incompatible.missing_keys)
    unexpected_keys = sorted(set(unexpected_keys).union(incompatible.unexpected_keys))
    log(f"init missing keys: {missing_keys}")
    log(f"init unexpected keys: {unexpected_keys}")
    log(f"init shape mismatches: {shape_mismatches}")
    loaded_shared_fraction = _check_shared_fraction(model_state, matching_state)

    def unexpected_for_architecture(key: str) -> bool:
        permitted_new_head = (replace_simcc_head and key.startswith("kpt_head.")) or (
            add_curve_head and key.startswith("curve_head.")
        )
        permitted_new_head |= add_rotation_head and key.startswith(
            ("rotation_head.", "rotation_bins")
        )
        return not permitted_new_head

    architecture_problems = []
    architecture_problems.extend(config_mismatches)
    architecture_problems.extend(
        f"missing {key}" for key in missing_keys if unexpected_for_architecture(key)
    )
    architecture_problems.extend(
        f"unexpected {key}" for key in unexpected_keys if unexpected_for_architecture(key)
    )
    architecture_problems.extend(
        f"shape {item}"
        for item in shape_mismatches
        if unexpected_for_architecture(item.split(":", 1)[0])
    )
    if architecture_problems:
        raise RuntimeError(
            "unexpected initialization architecture mismatch: " + "; ".join(architecture_problems)
        )
    modes = []
    if replace_simcc_head:
        modes.append("SimCC head replaced by heatmap head")
    if add_rotation_head:
        modes.append("new rotation auxiliary head initialized")
    if add_curve_head:
        modes.append("new curve head initialized")
    mode = " ({})".format(", ".join(modes)) if modes else ""
    log(
        f"initialized from {checkpoint_path}: "
        f"loaded={len(matching_state)}/{len(model_state)} keys, "
        f"shared_params={loaded_shared_fraction:.1%}{mode}"
    )


def load_resume_checkpoint(
    resume_path: str,
    model: DoperNet,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cfg: DoperConfig,
    device: str,
) -> tuple[int, float, dict[str, Any]]:
    """Restore model, optimizer and scaler state from an epoch checkpoint.

    Args:
        resume_path: Checkpoint written by a previous run.
        model: Model to restore in place.
        optimizer: Optimizer to restore in place.
        scaler: Gradient scaler to restore in place.
        cfg: Config of the resumed run.
        device: Device the checkpoint tensors are mapped to.

    Returns:
        ``(start_epoch, best_kp, checkpoint)``.

    Raises:
        ValueError: If the checkpoint ``cfg.model`` is malformed.
    """
    ckpt, _, resume_model_cfg = checkpoint_configs(
        load_checkpoint(resume_path, device), f"resume checkpoint {resume_path!r}"
    )
    require_matching_rotation_metadata(resume_model_cfg, cfg.model)
    resume_curve_metadata = checkpoint_curve_metadata(resume_model_cfg, ckpt["model"].keys())
    require_matching_curve_metadata(resume_curve_metadata, cfg.model)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_kp = resumed_best_kp(ckpt, resume_path)
    trim_epoch_timing(Path(cfg.train.work_dir) / "epoch_timing.jsonl", start_epoch)
    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    log(f"resumed from {resume_path} at epoch {start_epoch}")
    return start_epoch, best_kp, ckpt
