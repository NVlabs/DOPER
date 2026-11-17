# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resumable BlenderProc render campaign: one physical layout of up to 25 views per seed.

Commands (``python -m rendering.blenderproc.campaign ...``):

* ``stage --config NAME --assets DIR --models DIR --keypoints FILE --out DIR``: verify the fetched
  assets (``rendering.blenderproc.assets``), copy them with the target's BOP models and frozen
  keypoints into ``<out>/assets``, and write ``<out>/config.json`` and ``<out>/identity.json``.
  ``--layouts``, ``--workers`` and ``--base-seed`` override the config (defaults: 1000 layouts,
  the config's worker count, seed 60000 for HANDAL obj40 and 70000 for HOT3D obj33).
* ``run --out DIR --worker I --blender DIR --gpu-index G``: render worker ``I``'s layouts
  (``range(I, layouts, workers)``), each with seed ``base_seed + layout``, by running
  ``python -m blenderproc run --custom-blender-path <blender> --temp-dir <tmp> scene.py ...`` with
  the BlenderProc interpreter (``--blenderproc-python``). Each attempt is assembled, audited and
  published atomically to ``<out>/train_pbr/<layout:06d>/``; completed layouts are verified
  against their receipt and skipped. A layout whose visible-count correction exceeds the
  configured bound is archived to ``<out>/failed/``, recorded under ``failed`` in the worker
  status (``<out>/workers/<I>.json``), and the worker goes on with its other layouts; it exits
  nonzero if any of its layouts is recorded failed. ``count_bound_attempts`` (default 1) attempts
  are made per layout and run. Later runs skip recorded failures; ``run --retry-failed``
  re-attempts only them, with the same seed. Everything that decides the counts is
  deterministic for a seed, so a retry can only succeed through GPU (OptiX, denoiser)
  nondeterminism. Any other failure (BlenderProc exit, audit, disk) stops the worker with
  ``status: failed``.
* ``status --out DIR [--deep]``: completed, missing, recorded-failed and zero-visible layouts.

Published layout ``<out>/train_pbr/<layout:06d>/`` (a BOP scene, read by ``doper prepare``):
``rgb/``, ``depth/``, ``mask/``, ``mask_visib/``, ``scene_gt.json``, ``scene_camera.json``,
``scene_gt_info.json`` (canonical counts), the writer's other scene files, ``records/``
(``scene_record.json``, ``render_receipt.json``, ``AUDITED.json``,
``visible_count_canonicalization.json``, ``writer_raw/scene_gt_info.json``, ``camera.json``,
``audit_overlays/``) and ``COMPLETE.json``, the receipt with the SHA-256 of every other file.
Layout ``l`` with ``l % 10 == 9`` is validation (``split: val_pbr``); every view of a layout stays
in one split.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any

from doper.bop.sources import canonical_sha256, publish_json, sha256_file
from rendering.blenderproc import assets as asset_store
from rendering.blenderproc.audit import RECORDS, CountBoundError, audit_layout
from rendering.blenderproc.plan import RenderConfig, config_from_json, load_config, load_pools

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
SCENE_SCRIPT = PACKAGE_DIR / "scene.py"
CODE_FILES = ("scene.py", "plan.py", "audit.py", "campaign.py")
IDENTITY_NAME = "identity.json"
STAGED_NAME = "STAGED.json"
CONFIG_NAME = "config.json"
COMPLETE_NAME = "COMPLETE.json"
KEYPOINTS_PATH = "geometry/keypoints_3d.json"
SPLIT_DIR = "train_pbr"
SCENE_FILES = ("scene_gt.json", "scene_camera.json", "scene_gt_info.json")
METADATA_FILES = (
    *SCENE_FILES,
    f"{RECORDS}/AUDITED.json",
    f"{RECORDS}/scene_record.json",
    f"{RECORDS}/render_receipt.json",
    f"{RECORDS}/visible_count_canonicalization.json",
    f"{RECORDS}/writer_raw/scene_gt_info.json",
)
RECORD_FILES = ("scene_record.json", "render_receipt.json")
DEFAULT_MIN_FREE_GIB = 50
SCRUBBED_ENV = ("BLENDER_PROC_RANDOM_SEED", "PYTHONPATH")


def assigned_layouts(worker: int, workers: int, layouts: int) -> list[int]:
    """Return the layouts of one worker in a disjoint round-robin partition.

    Args:
        worker: Worker index in ``0..workers-1``.
        workers: Number of workers.
        layouts: Number of layouts.

    Returns:
        ``list(range(worker, layouts, workers))``.
    """
    if not 0 <= worker < workers or workers < 1 or layouts < 1:
        raise ValueError("invalid disjoint layout partition")
    return list(range(worker, layouts, workers))


def split_for_layout(layout: int, config: RenderConfig) -> str:
    """Return ``val_pbr`` for layouts ``l % modulo == residue`` (10 and 9), else ``train_pbr``."""
    return "val_pbr" if layout % config.val_modulo == config.val_residue else "train_pbr"


def layout_seed(config: RenderConfig, layout: int) -> int:
    """Return the seed of one layout: ``base_seed + layout``."""
    return config.base_seed + layout


def layout_dir(out: Path, layout: int) -> Path:
    """Published directory of one layout."""
    return out / SPLIT_DIR / f"{layout:06d}"


def render_command(
    *,
    python: str,
    blender: Path,
    temp_dir: Path,
    assets: Path,
    config: Path,
    attempt: Path,
    seed: int,
    gpu_index: int,
) -> list[str]:
    """Return the BlenderProc command line that renders one layout attempt.

    Args:
        python: Interpreter with BlenderProc 2.8.0 installed.
        blender: Blender 4.2.1 installation directory.
        temp_dir: BlenderProc temporary directory (task-local).
        assets: Staged asset root.
        config: Staged ``config.json``.
        attempt: Fresh attempt output directory.
        seed: Layout seed.
        gpu_index: Index of the OptiX device to render on.

    Returns:
        The argument list.
    """
    return [
        python, "-m", "blenderproc", "run", "--custom-blender-path", str(blender),
        "--temp-dir", str(temp_dir), str(SCENE_SCRIPT),
        "--assets", str(assets), "--config", str(config), "--out", str(attempt),
        "--seed", str(seed), "--gpu-index", str(gpu_index),
    ]  # fmt: skip


# --- staging ------------------------------------------------------------------------------------


def code_sha256() -> dict[str, str]:
    """SHA-256 of the renderer, plan, audit and campaign sources (part of the identity)."""
    return {name: sha256_file(PACKAGE_DIR / name) for name in CODE_FILES}


def make_identity(config: RenderConfig, assets_manifest_sha256: str) -> dict[str, Any]:
    """Return the campaign identity; any change requires a new ``--out``.

    Args:
        config: Effective render config.
        assets_manifest_sha256: SHA-256 of the staged ``ASSET_MANIFEST.json``.

    Returns:
        The identity record.
    """
    return {
        "schema": 1,
        "config": config.as_json(),
        "dataset": config.dataset,
        "target_object": config.target_object,
        "layouts": config.layouts,
        "frames_per_layout": config.frames_per_layout,
        "workers": config.workers,
        "base_seed": config.base_seed,
        "split": f"layout % {config.val_modulo} == {config.val_residue} is val_pbr",
        "assets_manifest_sha256": assets_manifest_sha256,
        "keypoints_sha256": config.keypoints_sha256,
        "code_sha256": code_sha256(),
        "blenderproc": config.blenderproc,
        "blender": config.blender,
        "bop_toolkit": config.bop_toolkit,
    }


def _copy_target_models(models: Path, dest: Path, config: RenderConfig) -> None:
    info = json.loads((models / "models_info.json").read_text())
    low, high = config.control_objects
    ids = sorted({*range(low, high + 1), config.target_object})
    missing = [i for i in ids if str(i) not in info or not (models / f"obj_{i:06d}.ply").is_file()]
    if missing:
        raise ValueError(f"{models}: missing models_info rows or PLY files for objects {missing}")
    dest.mkdir(parents=True)
    shutil.copyfile(models / "models_info.json", dest / "models_info.json")
    for i in ids:
        for path in sorted(models.glob(f"obj_{i:06d}*")):
            if path.is_file():
                shutil.copyfile(path, dest / path.name)


def _build_assets(args: argparse.Namespace, config: RenderConfig, staging: Path) -> None:
    asset_store.verify(args.assets)
    for spec in config.distractors:
        shutil.copytree(args.assets / "bop" / spec.dataset, staging / "bop" / spec.dataset)
    shutil.copytree(args.assets / "cctextures", staging / "cctextures")
    _copy_target_models(args.models, staging / "bop" / config.dataset / "models", config)
    (staging / KEYPOINTS_PATH).parent.mkdir(parents=True)
    shutil.copyfile(args.keypoints, staging / KEYPOINTS_PATH)
    asset_store.write_manifest(staging)
    pools = load_pools(staging, config)
    count = len(pools.room_materials)
    if count < config.room_materials_min or (
        config.room_materials_max is not None and count > config.room_materials_max
    ):
        raise ValueError(f"{count} room materials outside the configured bounds")


def effective_config(args: argparse.Namespace) -> RenderConfig:
    """Apply the ``stage`` overrides and check the frozen keypoints against the config."""
    config = load_config(args.config)
    overrides = {
        k: v
        for k, v in [("layouts", args.layouts), ("workers", args.workers),
                     ("base_seed", args.base_seed), ("keypoints_sha256", args.keypoints_sha256)]
        if v is not None
    }  # fmt: skip
    config = dataclasses.replace(config, **overrides)
    actual = sha256_file(args.keypoints)
    if actual != config.keypoints_sha256:
        raise ValueError(
            f"frozen keypoints {args.keypoints} have SHA-256 {actual}, the config expects "
            f"{config.keypoints_sha256} (pass --keypoints-sha256 for a different selection)"
        )
    return config


@contextlib.contextmanager
def locked(path: Path, *, blocking: bool = True) -> Iterator[None]:
    """Hold an exclusive ``flock`` on ``path`` (created) for the duration of the block."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def stage(args: argparse.Namespace) -> int:
    """Stage assets, config and identity under ``--out`` (idempotent for the same identity)."""
    config = effective_config(args)
    out: Path = args.out
    if args.dry_run:
        print(
            json.dumps(
                {
                    "out": str(out),
                    "config": config.name,
                    "layouts": config.layouts,
                    "workers": config.workers,
                    "base_seed": config.base_seed,
                    "frames_per_layout": config.frames_per_layout,
                },
                indent=2,
            )
        )
        return 0
    out.mkdir(parents=True, exist_ok=True)
    with locked(out / "stage.lock"):
        staged_assets = out / "assets"
        if not staged_assets.exists():
            staging = out / f"assets-staging-{uuid.uuid4().hex}"
            try:
                _build_assets(args, config, staging)
                staging.rename(staged_assets)
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        manifest = asset_store.verify(staged_assets)
        if sha256_file(staged_assets / KEYPOINTS_PATH) != config.keypoints_sha256:
            raise ValueError("staged keypoints differ from the campaign's frozen keypoints")
        identity = make_identity(config, sha256_file(staged_assets / asset_store.MANIFEST_NAME))
        identity_path = out / IDENTITY_NAME
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError(f"{out}: campaign identity changed; use a new --out")
        if not identity_path.exists() and any((out / SPLIT_DIR).glob(f"*/{COMPLETE_NAME}")):
            raise ValueError(f"{out}: layouts exist without a campaign identity")
        publish_json(identity_path, identity)
        publish_json(out / CONFIG_NAME, config.as_json())
        publish_json(
            out / STAGED_NAME,
            {
                "ok": True,
                "identity_sha256": canonical_sha256(identity),
                "asset_files": len(manifest["files"]),
            },
        )
    print(f"staged {out}")
    return 0


def read_campaign(out: Path) -> tuple[RenderConfig, dict[str, Any], str]:
    """Return the staged config, identity and identity SHA-256 of a campaign."""
    if not (out / STAGED_NAME).is_file():
        raise ValueError(f"{out}: run `stage` first")
    identity = json.loads((out / IDENTITY_NAME).read_text())
    config = config_from_json(json.loads((out / CONFIG_NAME).read_text()))
    if config.as_json() != identity["config"]:
        raise ValueError(f"{out}: config.json differs from the campaign identity")
    return config, identity, canonical_sha256(identity)


# --- publishing ---------------------------------------------------------------------------------


def assemble_layout(attempt: Path, dest: Path, config: RenderConfig) -> Path:
    """Copy a rendered attempt into the published layout form (see the module docstring).

    Args:
        attempt: BlenderProc output (``bop_data/<dataset>/train_pbr/000000``, records).
        dest: New layout directory.
        config: Render config.

    Returns:
        ``dest``.
    """
    bop = attempt / "bop_data" / config.dataset
    scene = bop / "train_pbr" / "000000"
    if not scene.is_dir():
        raise ValueError(f"{attempt}: no BOP scene written")
    shutil.copytree(scene, dest)
    records = dest / RECORDS
    records.mkdir()
    for name in RECORD_FILES:
        shutil.copyfile(attempt / name, records / name)
    if (bop / "camera.json").is_file():
        shutil.copyfile(bop / "camera.json", records / "camera.json")
    return dest


def _files(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): sha256_file(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != COMPLETE_NAME
    }


def _check_files(root: Path, files: dict[str, str], names: list[str] | None = None) -> None:
    for name in names if names is not None else list(files):
        relative = PurePosixPath(name)
        path = root / name
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{root}: unsafe receipt path {name}")
        if path.is_symlink() or not path.is_file() or sha256_file(path) != files[name]:
            raise ValueError(f"{root}: {name} differs from its receipt")


def verify_layout(
    path: Path,
    identity_sha256: str,
    layout: int,
    seed: int,
    config: RenderConfig,
    *,
    deep: bool = False,
) -> dict[str, Any]:
    """Check a published layout against its ``COMPLETE.json`` receipt.

    The fast check re-hashes the metadata files and requires every listed file to exist;
    ``deep`` re-hashes every file.

    Args:
        path: Published layout directory.
        identity_sha256: Campaign identity digest.
        layout: Layout index.
        seed: Expected seed.
        config: Render config.
        deep: Re-hash every image and mask too.

    Returns:
        The receipt.
    """
    receipt = json.loads((path / COMPLETE_NAME).read_text())
    expected = (identity_sha256, layout, seed, config.frames_per_layout,
                split_for_layout(layout, config))  # fmt: skip
    actual = (receipt["identity_sha256"], receipt["layout_id"], receipt["seed"],
              receipt["frames"], receipt["split"])  # fmt: skip
    if actual != expected:
        raise ValueError(f"{path}: receipt identity {actual} != expected {expected}")
    files = receipt["files"]
    missing = [name for name in files if not (path / name).is_file()]
    if missing:
        raise ValueError(f"{path}: files missing: {missing[:5]}")
    _check_files(path, files, None if deep else [n for n in METADATA_FILES if n in files])
    audited = json.loads((path / RECORDS / "AUDITED.json").read_text())
    if (
        not audited["ok"]
        or audited["frames"] != receipt["frames"]
        or (audited["target_positive_frames"] != receipt["target_positive_frames"])
    ):
        raise ValueError(f"{path}: audit and receipt disagree")
    return receipt


def publish_attempt(
    attempt: Path,
    out: Path,
    config: RenderConfig,
    keypoints: Path,
    *,
    layout: int,
    seed: int,
    identity_sha256: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Assemble, audit, receipt and atomically publish one rendered attempt.

    A failed audit moves the assembled attempt to ``<out>/failed/`` and re-raises.

    Args:
        attempt: BlenderProc output of the attempt.
        out: Campaign root.
        config: Render config.
        keypoints: Frozen keypoints of the target.
        layout: Layout index.
        seed: Layout seed.
        identity_sha256: Campaign identity digest.
        extra: Further receipt fields (attempt number, device).

    Returns:
        The published layout directory.
    """
    final = layout_dir(out, layout)
    if final.exists():
        raise FileExistsError(f"{final}: layout already published")
    attempt_tag = f"-a{extra['attempt']}" if extra and "attempt" in extra else ""
    pending = out / "pending" / f"{layout:06d}{attempt_tag}-{uuid.uuid4().hex}"
    pending.parent.mkdir(parents=True, exist_ok=True)
    assemble_layout(attempt, pending, config)
    try:
        result = audit_layout(pending, config, keypoints)
    except Exception:
        failed = out / "failed" / pending.name
        failed.parent.mkdir(parents=True, exist_ok=True)
        pending.rename(failed)
        raise
    files = _files(pending)
    receipt = {
        "schema": 1,
        "identity_sha256": identity_sha256,
        "layout_id": layout,
        "seed": seed,
        "split": split_for_layout(layout, config),
        "dataset": config.dataset,
        "target_object": config.target_object,
        "frames": result.frames,
        "target_positive_frames": result.target_positive_frames,
        "zero_visible_target": result.zero_visible_layout_retained,
        "max_count_correction": result.max_count_correction,
        "render_seconds": result.render_seconds,
        "total_seconds": result.total_seconds,
        "bop_bytes": result.bop_bytes,
        **(extra or {}),
        "files": files,
    }
    publish_json(pending / COMPLETE_NAME, receipt)
    verify_layout(pending, identity_sha256, layout, seed, config, deep=True)
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        raise FileExistsError(f"{final}: overlapping worker published this layout")
    pending.rename(final)
    return final


# --- rendering ----------------------------------------------------------------------------------


def _child_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in SCRUBBED_ENV}


def _render_once(
    args: argparse.Namespace,
    out: Path,
    layout: int,
    seed: int,
    scratch: Path,
    attempt_no: int,
    config: RenderConfig,
    identity_sha256: str,
) -> Path:
    local = Path(tempfile.mkdtemp(prefix=f"layout-{layout:06d}-", dir=scratch))
    try:
        attempt, temp = local / "render", local / "blender-tmp"
        temp.mkdir()
        log = out / "logs" / f"{layout:06d}-{attempt_no}-{uuid.uuid4().hex}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        command = render_command(
            python=args.blenderproc_python, blender=args.blender, temp_dir=temp,
            assets=out / "assets", config=out / CONFIG_NAME, attempt=attempt, seed=seed,
            gpu_index=args.gpu_index,
        )  # fmt: skip
        with log.open("w") as stream:
            done = subprocess.run(command, cwd=REPO_ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                  env=_child_env(), check=False)  # fmt: skip
        if done.returncode != 0:
            if attempt.exists():
                name = f"{layout:06d}-a{attempt_no}-{uuid.uuid4().hex}"
                shutil.copytree(attempt, out / "failed" / name)
            tail = log.read_text(errors="replace")[-4000:]
            raise RuntimeError(f"layout {layout}: BlenderProc exited {done.returncode}\n{tail}")
        return publish_attempt(
            attempt, out, config, out / "assets" / KEYPOINTS_PATH, layout=layout, seed=seed,
            identity_sha256=identity_sha256,
            extra={"attempt": attempt_no, "gpu_index": args.gpu_index, "log": log.name},
        )  # fmt: skip
    finally:
        shutil.rmtree(local, ignore_errors=True)


def render_layout(
    args: argparse.Namespace,
    out: Path,
    layout: int,
    config: RenderConfig,
    identity_sha256: str,
    first_attempt: int = 1,
) -> Path:
    """Render, audit and publish one layout in up to ``count_bound_attempts`` fresh attempts.

    Raises:
        CountBoundError: Every attempt exceeded the visible-count bound.
    """
    seed = layout_seed(config, layout)
    scratch = Path(args.scratch or tempfile.gettempdir())
    scratch.mkdir(parents=True, exist_ok=True)
    last = first_attempt + config.count_bound_attempts
    for attempt_no in range(first_attempt, last):
        try:
            return _render_once(args, out, layout, seed, scratch, attempt_no, config,
                                identity_sha256)  # fmt: skip
        except CountBoundError as err:
            print(f"layout {layout} attempt {attempt_no}: {err}", flush=True)
    raise CountBoundError(
        f"layout {layout}: attempts {first_attempt}..{last - 1} exceeded the "
        f"{config.count_bound_px} px count bound; archived under failed/"
    )


def _worker_status(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def recorded_failures(out: Path) -> dict[int, dict[str, Any]]:
    """Layouts recorded failed in any worker status (``layout -> record``), unpublished only."""
    failures: dict[int, dict[str, Any]] = {}
    for path in sorted((out / "workers").glob("*.json")):
        for key, record in _worker_status(path).get("failed", {}).items():
            if not layout_dir(out, int(key)).exists():
                failures[int(key)] = record
    return failures


def _todo(args: argparse.Namespace, out: Path, assigned: list[int],
          failed: dict[int, dict[str, Any]]) -> list[int]:  # fmt: skip
    if args.retry_failed:
        return [i for i in assigned if i in failed and not layout_dir(out, i).exists()]
    return [i for i in assigned if i not in failed]


def run(args: argparse.Namespace) -> int:
    """Render one worker's layouts (``--dry-run`` prints the plan, one JSON line per layout).

    Returns:
        0, or 1 when any of the worker's layouts is recorded failed.
    """
    out: Path = args.out
    config, _, identity_sha256 = read_campaign(out)
    if args.workers is not None and args.workers != config.workers:
        raise ValueError(f"campaign was staged for {config.workers} workers, not {args.workers}")
    assigned = assigned_layouts(args.worker, config.workers, config.layouts)
    status_path = out / "workers" / f"{args.worker:03d}.json"
    failed = {
        int(k): v
        for k, v in _worker_status(status_path).get("failed", {}).items()
        if not layout_dir(out, int(k)).exists()
    }
    todo = _todo(args, out, assigned, failed)
    if args.dry_run:
        return _dry_run(args, out, config, identity_sha256, todo)
    if args.blender is None or not (args.blender / "blender").is_file():
        raise ValueError(f"--blender {args.blender}: not a Blender installation directory")
    completed: list[int] = []
    rendered = 0

    def report(state: str, **fields: Any) -> dict[str, Any]:  # noqa: ANN401
        record = {"worker": args.worker, "status": state, "identity_sha256": identity_sha256,
                  "completed": completed, "assigned": len(assigned),
                  "failed": {str(k): v for k, v in sorted(failed.items())}, **fields}  # fmt: skip
        publish_json(status_path, record)
        return record

    with locked(out / "locks" / f"worker-{args.worker:03d}.lock", blocking=False):
        for layout in todo:
            final = layout_dir(out, layout)
            if final.exists():
                verify_layout(final, identity_sha256, layout, layout_seed(config, layout), config)
                completed.append(layout)
                continue
            if args.max_layouts and rendered >= args.max_layouts:
                break
            if shutil.disk_usage(out).free < args.min_free_gib * 1024**3:
                raise ValueError(f"less than {args.min_free_gib} GiB free under {out}")
            report("rendering", layout=layout)
            started = time.time()
            first = failed.get(layout, {}).get("attempts", 0) + 1
            try:
                render_layout(args, out, layout, config, identity_sha256, first_attempt=first)
            except CountBoundError as err:
                failed[layout] = {"error": "CountBoundError", "message": str(err),
                                  "seed": layout_seed(config, layout),
                                  "attempts": first - 1 + config.count_bound_attempts}  # fmt: skip
                rendered += 1
                print(f"layout {layout} recorded failed: {err}", flush=True)
                continue
            except Exception as err:
                report("failed", layout=layout, error=type(err).__name__, message=str(err)[-2000:])
                raise
            failed.pop(layout, None)
            completed.append(layout)
            rendered += 1
            print(f"layout {layout} published ({time.time() - started:.0f}s)", flush=True)
        done = all(layout_dir(out, i).exists() for i in assigned)
        state = "complete" if done else ("failed_layouts" if failed else "partial")
        summary = report(state)
    print(json.dumps(summary))
    return 1 if failed else 0


def _dry_run(
    args: argparse.Namespace,
    out: Path,
    config: RenderConfig,
    identity_sha256: str,
    assigned: list[int],
) -> int:
    scratch = Path(args.scratch or tempfile.gettempdir())
    planned = 0
    for layout in assigned:
        final = layout_dir(out, layout)
        if final.exists():
            verify_layout(final, identity_sha256, layout, layout_seed(config, layout), config)
            continue
        if args.max_layouts and planned >= args.max_layouts:
            break
        local = scratch / f"layout-{layout:06d}"
        seed = layout_seed(config, layout)
        command = render_command(
            python=args.blenderproc_python, blender=args.blender or Path("<blender>"),
            temp_dir=local / "blender-tmp",
            assets=out / "assets", config=out / CONFIG_NAME, attempt=local / "render", seed=seed,
            gpu_index=args.gpu_index,
        )  # fmt: skip
        print(
            json.dumps(
                {
                    "layout": layout,
                    "seed": seed,
                    "split": split_for_layout(layout, config),
                    "command": command,
                    "publish": str(final),
                }
            )
        )
        planned += 1
    return 0


def status(args: argparse.Namespace) -> int:
    """Print the campaign status as JSON; exit 1 if a published layout fails verification."""
    out: Path = args.out
    config, _, identity_sha256 = read_campaign(out)
    completed, zero_visible, problems = [], [], []
    splits = {"train_pbr": 0, "val_pbr": 0}
    for layout in range(config.layouts):
        final = layout_dir(out, layout)
        if not final.exists():
            continue
        try:
            receipt = verify_layout(final, identity_sha256, layout, layout_seed(config, layout),
                                    config, deep=args.deep)  # fmt: skip
        except (ValueError, OSError, KeyError) as err:
            problems.append(f"{layout}: {err}")
            continue
        completed.append(layout)
        splits[receipt["split"]] += 1
        if receipt["zero_visible_target"]:
            zero_visible.append(layout)
    failed = out / "failed"
    report = {
        "out": str(out),
        "identity_sha256": identity_sha256,
        "layouts": config.layouts,
        "completed": len(completed),
        "missing": config.layouts - len(completed),
        "splits": splits,
        "zero_visible_layouts": zero_visible,
        "failed_layouts": sorted(recorded_failures(out)),
        "failed_attempts": len(list(failed.iterdir())) if failed.is_dir() else 0,
        "problems": problems,
    }
    print(json.dumps(report, indent=2))
    return 1 if problems else 0


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description="Resumable BlenderProc render campaign.")
    commands = parser.add_subparsers(dest="command", required=True)
    staging = commands.add_parser("stage", help="stage assets, config and identity")
    staging.add_argument("--config", required=True, help="config name or JSON path")
    staging.add_argument("--assets", type=Path, required=True, help="fetched asset root")
    staging.add_argument("--models", type=Path, required=True, help="target BOP models dir (mm)")
    staging.add_argument("--keypoints", type=Path, required=True, help="frozen keypoints_3d.json")
    staging.add_argument("--keypoints-sha256", help="expected keypoints SHA-256 (default: config)")
    staging.add_argument("--out", type=Path, required=True, help="campaign root")
    staging.add_argument("--layouts", type=int)
    staging.add_argument("--workers", type=int)
    staging.add_argument("--base-seed", type=int)
    staging.add_argument("--dry-run", action="store_true")
    running = commands.add_parser("run", help="render one worker's layouts")
    running.add_argument("--out", type=Path, required=True)
    running.add_argument("--worker", type=int, required=True)
    running.add_argument("--workers", type=int, help="must equal the staged worker count")
    running.add_argument(
        "--blender", type=Path, help="Blender 4.2.1 directory (required unless --dry-run)"
    )
    running.add_argument("--gpu-index", type=int, default=0, help="OptiX device index")
    running.add_argument(
        "--blenderproc-python",
        default=sys.executable,
        help="interpreter with blenderproc==2.8.0 (default: this one)",
    )
    running.add_argument("--scratch", type=Path, help="task-local temporary directory")
    running.add_argument("--max-layouts", type=int, default=0, help="stop after N new layouts")
    running.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB)
    running.add_argument(
        "--retry-failed",
        action="store_true",
        help="re-attempt only this worker's recorded failures (same seed; helps only when the "
        "failure came from GPU nondeterminism)",
    )
    running.add_argument("--dry-run", action="store_true")
    reporting = commands.add_parser("status", help="report completed and missing layouts")
    reporting.add_argument("--out", type=Path, required=True)
    reporting.add_argument("--deep", action="store_true", help="re-hash every published file")
    reporting.add_argument("--dry-run", action="store_true", help="same as without (read-only)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    args = build_parser().parse_args(argv)
    if getattr(args, "max_layouts", 0) < 0 or getattr(args, "gpu_index", 0) < 0:
        raise SystemExit("--max-layouts and --gpu-index must be non-negative")
    handlers = {"stage": stage, "run": run, "status": status}
    try:
        return handlers[args.command](args)
    except BlockingIOError:
        print("error: this worker is already running (its lock is held)", file=sys.stderr)
        return 2
    except (ValueError, FileExistsError, RuntimeError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
