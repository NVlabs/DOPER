#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Parallel Isaac Sim launcher with identity-gated resume and bounded recovery.

Spawns one ``isaac_worker.py`` per GPU slot, each rendering a disjoint sample range of one
dataset in its own Kit process, then merges their annotations. Scenes are sampled once,
deterministically per sample index, by ``rendering.scene_spec`` (shared with the Blender
launcher), so an Isaac and a Blender render of one config and seed realize identical scenes.
Workers run under Isaac Sim's Python (``--isaac-python``, ``$DOPER_ISAAC_PYTHON``, else
``/isaac-sim/python.sh`` when it exists). Requires Isaac Sim 6.0 or newer.

Usage:
    /isaac-sim/python.sh rendering/isaac/isaac_parallel.py --config render.yaml \\
        --outdir out --usd object.usda --num-images 100 --gpus 0,1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from copy import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from types import FrameType
from typing import Any

import numpy as np

# Direct script execution must resolve shared renderer modules from this checkout.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from doper.scene import geometry, labels  # noqa: E402
from rendering.config import RenderConfig, load_render_config  # noqa: E402
from rendering.dataset_contract import merge_annotations, validate_dataset_records  # noqa: E402
from rendering.environment_bundle import (  # noqa: E402
    ReplayInputs,
    add_replay_arguments,
    include_replay_identity,
    load_replay,
    replay_options,
    validate_applied_readbacks,
)
from rendering.isaac.asset_cache import local_replay  # noqa: E402
from rendering.isaac.inputs import load_object_keypoints  # noqa: E402
from rendering.isaac.launch_contract import RENDERER_VERSION, probe_runtime  # noqa: E402
from rendering.isaac.settings import IsaacSettings, effective_settings_match  # noqa: E402
from rendering.launch_common import (  # noqa: E402
    add_shard_arguments,
    check_shard_layout,
    count_shard_images,
    require_min_version,
    split_range,
)
from rendering.output_lock import output_lock  # noqa: E402
from rendering.resume import consolidate_checkpoints, load_resume_state  # noqa: E402
from rendering.robustness import require_backend  # noqa: E402
from rendering.run_manifest import (  # noqa: E402
    RenderIdentity,
    build_render_identity,
    finish_shard,
    load_shards,
    merge_shards,
    shard_manifest,
    shard_manifest_path,
    start_shard_attempt,
    validate_resume_identity,
)
from rendering.scene_spec import AssetCatalog, generate_scenes, read_scene_records  # noqa: E402
from rendering.usd_mesh import keypoints_payload, read_mesh_usda  # noqa: E402
from rendering.watchdog import (  # noqa: E402
    StallMonitor,
    WorkerSlot,
    supervise,
    terminate_process_group,
)

ISAAC_PYTHON_ENV = "DOPER_ISAAC_PYTHON"
DEFAULT_ISAAC_PYTHON = "/isaac-sim/python.sh"
MIN_ISAAC_VERSION = (6, 0)
WORKER_SCRIPT = Path(__file__).resolve().parent / "isaac_worker.py"
# Shared dataset metadata must be readable by other users, like the images.
METADATA_MODE = 0o644


@dataclass(frozen=True)
class WorkerRange:
    """One worker's id, sample range ``[start_idx, end_idx)`` and GPU."""

    worker_id: int
    start_idx: int
    end_idx: int
    gpu: int


@dataclass(frozen=True)
class LaunchInputs:
    """Resolved launch paths, shared by the launcher and its workers."""

    outdir: Path
    scenes_path: Path
    config_path: Path
    usd_path: str
    keypoints_path: str | None


def parse_gpus(text: str) -> list[int]:
    """Parse ``--gpus``: distinct nonnegative device indices, comma separated."""
    try:
        gpus = [int(value.strip()) for value in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--gpus must contain comma-separated device indices"
        ) from exc
    if not gpus or any(gpu < 0 for gpu in gpus) or len(set(gpus)) != len(gpus):
        raise argparse.ArgumentTypeError("--gpus requires distinct nonnegative device indices")
    return gpus


def build_parser() -> argparse.ArgumentParser:
    """Build the launcher command line."""
    parser = argparse.ArgumentParser(description="Parallel Isaac Sim synthetic data generation")
    parser.add_argument("--config", required=True, help="Render config (YAML)")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument(
        "--num-images", type=int, required=True, help="Total number of samples to render"
    )
    parser.add_argument("--gpus", type=parse_gpus, default=[0], help="Comma-separated GPU indices")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument(
        "--usd", default=None, help="Object .usda (else object.usd_path from the config)"
    )
    parser.add_argument("--keypoints-json", default=None, help="Override object.keypoints_json")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--isaac-python",
        "--python-bin",
        dest="isaac_python",
        default=None,
        help=f"Isaac Sim Python for the workers (default: ${ISAAC_PYTHON_ENV}, then "
        f"{DEFAULT_ISAAC_PYTHON} when it exists)",
    )
    parser.add_argument(
        "--repo-root",
        default=str(REPO_ROOT),
        help="Checkout whose git revision is recorded in the shard manifest (default: this one)",
    )
    add_shard_arguments(parser, progress_interval_sec=10.0)
    parser.add_argument(
        "--stall-timeout-sec",
        type=float,
        default=900.0,
        help="Kill and relaunch (with --resume) a worker whose log/checkpoint did not change for "
        "this long; Kit can hang inside a render step without exiting (0 disables)",
    )
    parser.add_argument(
        "--startup-grace-sec",
        type=float,
        default=1200.0,
        help="Extra stall allowance before a worker's first frame (shader compilation)",
    )
    parser.add_argument(
        "--max-restarts", type=int, default=3, help="Stall restarts per worker before giving up"
    )
    add_replay_arguments(parser)
    return parser


def check_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject inconsistent counts, intervals and replay options."""
    try:
        replay_options(args.environment_root, args.environment_gain, args.environment_adapter)
    except ValueError as exc:
        parser.error(str(exc))
    for name in ("num_images", "workers_per_gpu"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("start_idx", "worker_id_offset", "max_restarts"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    for name in ("progress_interval_sec", "stall_timeout_sec", "startup_grace_sec"):
        value = getattr(args, name)
        positive = name == "progress_interval_sec"
        if not math.isfinite(value) or value < 0 or (positive and value == 0):
            requirement = "positive" if positive else "nonnegative"
            parser.error(f"--{name.replace('_', '-')} must be finite and {requirement}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and check the launcher command line."""
    parser = build_parser()
    args = parser.parse_args(argv)
    check_arguments(parser, args)
    args.repo_root = str(Path(args.repo_root).resolve())
    return args


def resolve_isaac_python(explicit: str | None) -> str | None:
    """Return ``--isaac-python``, then ``$DOPER_ISAAC_PYTHON``, then the container default."""
    if explicit:
        return explicit
    if os.environ.get(ISAAC_PYTHON_ENV):
        return os.environ[ISAAC_PYTHON_ENV]
    return DEFAULT_ISAAC_PYTHON if Path(DEFAULT_ISAAC_PYTHON).is_file() else None


def isaac_runtime(python_bin: str) -> dict[str, str]:
    """Probe the worker interpreter and require Isaac Sim 6.0 or newer.

    Raises:
        SystemExit: The interpreter cannot be probed or its Isaac Sim is too old.
    """
    try:
        runtime = probe_runtime(python_bin)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise SystemExit(
            f"cannot identify Isaac Sim with {python_bin!r} ({exc}); use --isaac-python or set "
            f"{ISAAC_PYTHON_ENV}"
        ) from exc
    require_min_version("Isaac Sim", runtime["isaacsim"], MIN_ISAAC_VERSION)
    return runtime


def worker_ranges(args: argparse.Namespace) -> list[WorkerRange]:
    """Split the sample range over ``workers_per_gpu`` slots per GPU; extras go first."""
    gpus = [gpu for gpu in args.gpus for _ in range(args.workers_per_gpu)]
    count = min(len(gpus), args.num_images)
    ranges = split_range(args.start_idx, args.num_images, count)
    return [
        WorkerRange(args.worker_id_offset + index, start, end, gpu)
        for index, ((start, end), gpu) in enumerate(zip(ranges, gpus, strict=False))
    ]


def validate_output_destination(
    outdir: Path, identity: RenderIdentity, args: argparse.Namespace, layout: Sequence[WorkerRange]
) -> None:
    """Read only: refuse identity, range and worker namespace collisions.

    Raises:
        ValueError: Resume or shard layout does not match the recorded provenance.
    """
    existing = load_shards(outdir)
    validate_resume_identity(outdir, identity, require_existing=args.resume)
    check_shard_layout(
        outdir,
        existing,
        start_idx=args.start_idx,
        end_idx=args.start_idx + args.num_images,
        worker_ids=[worker.worker_id for worker in layout],
        resume=args.resume,
        exists_before_collision=True,
    )


def worker_command(
    args: argparse.Namespace, worker: WorkerRange, paths: LaunchInputs, *, resume: bool
) -> list[str]:
    """Isaac Python command line of one worker."""
    command = [
        args.isaac_python,
        str(WORKER_SCRIPT),
        "--config",
        str(paths.config_path),
        "--outdir",
        str(paths.outdir),
        "--worker-id",
        str(worker.worker_id),
        "--start-idx",
        str(worker.start_idx),
        "--end-idx",
        str(worker.end_idx),
        "--seed",
        str(args.seed),
        "--gpu-device",
        str(worker.gpu),
        "--scenes",
        str(paths.scenes_path),
        "--usd",
        str(paths.usd_path),
    ]
    if getattr(args, "environment_root", None):
        command += [
            "--environment-root",
            str(Path(args.environment_root).resolve()),
            "--environment-gain",
            str(args.environment_gain),
            "--environment-adapter",
            args.environment_adapter,
        ]
    if paths.keypoints_path:
        command += ["--keypoints-json", str(paths.keypoints_path)]
    if resume:
        command.append("--resume")
    return command


def worker_settings_errors(
    outdir: Path,
    worker: WorkerRange | WorkerSlot,
    expected_settings: dict[str, Any],
) -> list[str]:
    """Compare the worker's reported settings with the launch identity."""
    path = outdir / f"settings_worker_{worker.worker_id:03d}.json"
    try:
        actual = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return [f"Worker {worker.worker_id}: unreadable effective settings: {exc}"]
    if (
        not isinstance(actual, dict)
        or actual.keys() != expected_settings.keys()
        or any(
            actual[key] != value for key, value in expected_settings.items() if key != "effective"
        )
        or not effective_settings_match(expected_settings["effective"], actual["effective"])
    ):
        return [
            f"Worker {worker.worker_id}: effective settings/runtime differ from launch identity"
        ]
    return []


def worker_output_errors(
    outdir: Path, worker: WorkerRange | WorkerSlot, cfg: RenderConfig, num_keypoints: int
) -> list[str]:
    """Validate the worker's final annotation file and the files it references."""
    path = outdir / f"annotations_worker_{worker.worker_id:03d}.json"
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return [f"Worker {worker.worker_id}: unreadable annotations: {exc}"]
    errors = validate_dataset_records(
        payload,
        outdir,
        start_idx=worker.start_idx,
        end_idx=worker.end_idx,
        images_per_sample=cfg.images_per_sample,
        render_masks=cfg.render_masks,
        num_keypoints=num_keypoints,
    )
    return [f"Worker {worker.worker_id}: {error}" for error in errors]


class WorkerPool:
    """Launch, supervise and validate the workers of one shard attempt."""

    def __init__(
        self,
        args: argparse.Namespace,
        paths: LaunchInputs,
        cfg: RenderConfig,
        num_keypoints: int,
        expected_settings: dict[str, Any],
    ) -> None:
        """Bind the launch arguments, paths and the identity the workers must report."""
        self.args, self.paths, self.cfg = args, paths, cfg
        self.num_keypoints = num_keypoints
        self.expected_settings = expected_settings
        self.environment: ReplayInputs | None = None
        self.environment_records: list[dict[str, Any]] = []
        self.slots: list[WorkerSlot] = []
        self._errors: dict[int, list[str]] = {}

    def launch(self, worker: WorkerRange | WorkerSlot, *, resume: bool, append: bool) -> Any:  # noqa: ANN401
        """Start one worker in its own session, logging to ``worker_<id>.log``."""
        self._errors.pop(worker.worker_id, None)
        layout = WorkerRange(worker.worker_id, worker.start_idx, worker.end_idx, worker.gpu)
        command = worker_command(self.args, layout, self.paths, resume=resume)
        log_path = self.paths.outdir / f"worker_{worker.worker_id:03d}.log"
        with log_path.open("a" if append else "w") as log:
            return subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )

    def output_errors(self, slot: WorkerRange | WorkerSlot) -> list[str]:
        """All artifact problems of one worker: environment evidence, annotations, settings."""
        outdir = self.paths.outdir
        environment_errors = []
        if self.environment_records:
            records = [
                record
                for record in self.environment_records
                if slot.start_idx <= record["index"] < slot.end_idx
            ]
            try:
                validate_applied_readbacks(
                    outdir,
                    records,
                    self.cfg,
                    self.expected_settings["environment"],
                    bundle=self.environment.bundle if self.environment else None,
                )
            except ValueError as exc:
                environment_errors.append(str(exc))
        return (
            environment_errors
            + worker_output_errors(outdir, slot, self.cfg, self.num_keypoints)
            + worker_settings_errors(outdir, slot, self.expected_settings)
        )

    def validate_slot(self, slot: WorkerRange | WorkerSlot) -> list[str]:
        """Return the slot's artifact problems, validating it only once per (re)launch."""
        if slot.worker_id not in self._errors:
            self._errors[slot.worker_id] = self.output_errors(slot)
        return self._errors[slot.worker_id]

    def start(self, layout: Sequence[WorkerRange]) -> None:
        """Launch every worker of ``layout`` with a fresh stall monitor."""
        for worker in layout:
            monitor = StallMonitor(
                timeout_sec=self.args.stall_timeout_sec,
                startup_grace_sec=self.args.startup_grace_sec,
            )
            monitor.launched(time.monotonic())
            self.slots.append(
                WorkerSlot(
                    worker_id=worker.worker_id,
                    start_idx=worker.start_idx,
                    end_idx=worker.end_idx,
                    gpu=worker.gpu,
                    log_path=str(self.paths.outdir / f"worker_{worker.worker_id:03d}.log"),
                    proc=self.launch(worker, resume=self.args.resume, append=False),
                    monitor=monitor,
                )
            )

    def wait(self) -> int:
        """Supervise until every worker completed or gave up; return the shard's image count."""
        args, outdir = self.args, self.paths.outdir
        end = args.start_idx + args.num_images
        while True:
            supervise(
                self.slots,
                outdir=str(outdir),
                relaunch=lambda slot: self.launch(slot, resume=True, append=True),
                max_restarts=args.max_restarts,
                now=time.monotonic(),
                validate_complete=lambda slot: not self.validate_slot(slot),
                stop=terminate_process_group,
                log=lambda message: print(message, flush=True),
            )
            done = sum(slot.completed or slot.gave_up for slot in self.slots)
            rendered = count_shard_images(outdir, args.start_idx, end)
            print(
                f"Progress: {rendered}/{args.num_images * self.cfg.images_per_sample} images, "
                f"{done}/{len(self.slots)} workers finished",
                flush=True,
            )
            if done == len(self.slots):
                return rendered
            time.sleep(args.progress_interval_sec)

    def stop_all(self) -> list[str]:
        """Stop every started worker; return cleanup problems."""
        problems = []
        for slot in self.slots:
            try:
                terminate_process_group(slot.proc)
            except Exception as exc:
                problems.append(f"Worker {slot.worker_id} cleanup: {exc}")
        return problems


def _terminate(signum: int, frame: FrameType | None) -> None:
    """SIGTERM handler: exit through the cleanup path, ignoring further signals."""
    # A second scheduler signal must not interrupt worker/cache cleanup.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise SystemExit(128 + signum)


@contextmanager
def sigterm_exits() -> Iterator[Callable[[], Any]]:
    """Turn SIGTERM into ``SystemExit`` inside the block and restore the old handler after.

    Yields:
        A function that makes the process ignore SIGTERM, for cleanup that must not be
        interrupted by a second scheduler signal.
    """
    previous_handler = signal.signal(signal.SIGTERM, _terminate)

    def ignore_sigterm() -> None:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

    try:
        yield ignore_sigterm
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGTERM, previous_handler)


def run_workers(
    args: argparse.Namespace,
    layout: Sequence[WorkerRange],
    paths: LaunchInputs,
    cfg: RenderConfig,
    num_keypoints: int,
    manifest_path: Path,
    expected_settings: dict[str, Any],
    *,
    environment: ReplayInputs | None = None,
) -> None:
    """Run and supervise the workers, then record the shard outcome in its manifest.

    The launcher owns the staged native environment until every worker has exited; SIGTERM
    stops the workers and records a failed attempt.

    Raises:
        RuntimeError: A worker failed or produced invalid artifacts.
    """
    pool = WorkerPool(args, paths, cfg, num_keypoints, expected_settings)
    end = args.start_idx + args.num_images
    failure_recorded = False
    started = time.monotonic()
    cache = ExitStack()
    with sigterm_exits() as ignore_sigterm:
        try:
            if "environment" in expected_settings:
                if environment is None:
                    environment = load_replay(args, paths.usd_path)
                pool.environment_records = read_scene_records(paths.scenes_path)
            staged = cache.enter_context(local_replay(environment, pool.environment_records))
            if staged is not None:
                pool.args = copy(args)
                pool.args.environment_root = str(staged.bundle.root)
                environment = staged
            pool.environment = environment
            pool.start(layout)
            rendered = pool.wait()
            slots = pool.slots
            failed = [slot.worker_id for slot in slots if slot.gave_up or slot.proc.returncode != 0]
            errors = [error for slot in slots for error in pool.validate_slot(slot)]
            errors.extend(
                f"Worker {slot.worker_id}: {slot.failure_reason}" for slot in slots if slot.gave_up
            )
            finish_shard(
                manifest_path,
                images_rendered=rendered,
                elapsed_sec=time.monotonic() - started,
                failed_workers=failed,
                problems=errors,
            )
            if failed or errors:
                failure_recorded = True
                raise RuntimeError("Isaac worker failure: " + "; ".join(errors or [str(failed)]))
        except BaseException as exc:
            ignore_sigterm()
            cleanup_errors = pool.stop_all()
            if not failure_recorded or cleanup_errors:
                finish_shard(
                    manifest_path,
                    images_rendered=count_shard_images(paths.outdir, args.start_idx, end),
                    elapsed_sec=time.monotonic() - started,
                    failed_workers=[worker.worker_id for worker in layout],
                    problems=[f"{type(exc).__name__}: {exc}", *cleanup_errors],
                )
            raise
        finally:
            ignore_sigterm()
            cache.close()


def publish_shared_metadata(
    outdir: Path, cfg: RenderConfig, config_path: Path, kp_names: list[str], kp_local: np.ndarray
) -> None:
    """Publish the keypoint schema, calibration and config for every shard.

    Unique temporary files let disjoint launchers with the same validated render identity
    publish identical metadata without partial-file reads.
    """
    intr = geometry.intrinsics_from_lens(
        cfg.resolution, cfg.resolution, cfg.focal_length_mm, cfg.sensor_width_mm
    )
    payloads = {
        "keypoints_3d.json": json.dumps(
            keypoints_payload(cfg.mesh_name, kp_names, kp_local.tolist())
        ).encode(),
        "camera_intrinsics.json": json.dumps(
            labels.camera_intrinsics_payload(intr, cfg.focal_length_mm, cfg.sensor_width_mm)
        ).encode(),
        "render_config.yaml": config_path.read_bytes(),
    }
    for name, data in payloads.items():
        with tempfile.NamedTemporaryFile(dir=outdir, prefix=f".{name}.", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(data)
                stream.close()
                temporary.chmod(METADATA_MODE)
                temporary.replace(outdir / name)
            finally:
                temporary.unlink(missing_ok=True)


def write_scenes(
    cfg: RenderConfig,
    args: argparse.Namespace,
    paths: LaunchInputs,
    environment: ReplayInputs | None,
) -> dict[str, Any]:
    """Generate the shard's scene records and publish them only after they validate."""
    end = args.start_idx + args.num_images
    with tempfile.TemporaryDirectory(prefix="isaac-scenes-") as temporary:
        pending = Path(temporary) / "scenes.jsonl"
        generated = generate_scenes(
            cfg,
            usd_path=paths.usd_path,
            keypoints_json=paths.keypoints_path,
            seed=args.seed,
            start_idx=args.start_idx,
            end_idx=end,
            out_path=str(pending),
        )
        if environment is not None:
            for record in read_scene_records(pending):
                environment.bundle.verify_record(record)
        paths.outdir.mkdir(parents=True, exist_ok=True)
        (paths.outdir / "images").mkdir(exist_ok=True)
        paths.scenes_path.parent.mkdir(exist_ok=True)
        shutil.copyfile(pending, paths.scenes_path)
    return generated


def reused_samples(args: argparse.Namespace, cfg: RenderConfig, outdir: Path) -> list[int]:
    """Consolidate the shard's checkpoints for resume; return the sample indices already done."""
    end = args.start_idx + args.num_images
    info = consolidate_checkpoints(
        str(outdir), start_idx=args.start_idx, end_idx=end, images_per_sample=cfg.images_per_sample
    )
    state = load_resume_state(
        str(outdir),
        args.worker_id_offset,
        start_idx=args.start_idx,
        end_idx=end,
        images_per_sample=cfg.images_per_sample,
        render_masks=cfg.render_masks,
        snapshots=[info["snapshot"]],
    )
    return sorted(state.done_indices)


def resolve_inputs(args: argparse.Namespace, cfg: RenderConfig, config_path: Path) -> LaunchInputs:
    """Resolve output, scene, object and keypoint paths (CLI paths against the current directory).

    Raises:
        SystemExit: The object USD does not exist.
    """
    outdir = Path(args.outdir).resolve()
    usd_path = os.path.abspath(args.usd) if args.usd else cfg.resolve(cfg.usd_path)
    if not usd_path or not Path(usd_path).is_file():
        raise SystemExit(f"Object USD not found: {usd_path or '(none: set --usd)'}")
    keypoints = os.path.abspath(args.keypoints_json) if args.keypoints_json else None
    end = args.start_idx + args.num_images
    return LaunchInputs(
        outdir=outdir,
        scenes_path=outdir / "scenes" / f"scenes_{args.start_idx:08d}_{end:08d}.jsonl",
        config_path=config_path,
        usd_path=usd_path,
        keypoints_path=cfg.resolve(keypoints or cfg.keypoints_json),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the launcher and return the process exit status.

    Raises:
        SystemExit: Invalid arguments, a missing input or an unsupported Isaac Sim.
        ValueError: The output directory or render identity conflicts with this run.
        RuntimeError: A worker failed.
    """
    args = parse_args(argv)
    args.isaac_python = resolve_isaac_python(args.isaac_python)
    if args.isaac_python is None:
        raise SystemExit(
            f"Isaac Sim Python not found; use --isaac-python or set {ISAAC_PYTHON_ENV}"
        )
    config_path = Path(args.config).resolve()
    cfg = load_render_config(config_path)
    require_backend(cfg.robustness, "isaac", appearance=cfg.robust_appearance)
    layout = worker_ranges(args)
    paths = resolve_inputs(args, cfg, config_path)
    kp_names, kp_local = load_object_keypoints(
        read_mesh_usda(paths.usd_path), paths.usd_path, paths.keypoints_path
    )
    settings = IsaacSettings.from_config(cfg)
    settings.validate_environment(args.environment_root)
    expected_settings = {
        "requested": settings.to_dict(),
        "effective": settings.carb_settings(),
        "runtime": isaac_runtime(args.isaac_python),
    }
    environment = load_replay(args, paths.usd_path)
    include_replay_identity(expected_settings, environment)
    identity = build_render_identity(
        cfg=cfg,
        catalog=AssetCatalog.scan(cfg),
        seed=args.seed,
        renderer="isaac",
        renderer_version=RENDERER_VERSION,
        usd_path=paths.usd_path,
        keypoints_path=paths.keypoints_path,
        renderer_settings=expected_settings,
    )
    if environment is not None:
        environment.validate_assets(identity.to_dict()["assets"]["content_sha256"])
    end = args.start_idx + args.num_images
    with output_lock(paths.outdir):
        validate_output_destination(paths.outdir, identity, args, layout)
        # Validate before publishing anything into a new or resumed dataset.
        generated = write_scenes(cfg, args, paths, environment)
        reused = reused_samples(args, cfg, paths.outdir) if args.resume else []
        manifest_path = shard_manifest_path(paths.outdir, args.start_idx, end)
        manifest = shard_manifest(
            repo_root=args.repo_root,
            cfg=cfg,
            config_path=str(config_path),
            usd_path=paths.usd_path,
            keypoints_path=paths.keypoints_path,
            num_keypoints=generated["num_keypoints"],
            catalog_summary=generated["catalog"].summary(),
            seed=args.seed,
            start_idx=args.start_idx,
            end_idx=end,
            scenes_path=str(paths.scenes_path),
            renderer="isaac",
            workers=[asdict(worker) for worker in layout],
            outdir=str(paths.outdir),
        )
        manifest["reused_sample_indices"] = reused
        manifest["renderer_settings"] = expected_settings
        start_shard_attempt(manifest_path, manifest, identity)
        publish_shared_metadata(paths.outdir, cfg, config_path, kp_names, kp_local)
    run_workers(
        args,
        layout,
        paths,
        cfg,
        generated["num_keypoints"],
        manifest_path,
        expected_settings,
        environment=environment,
    )
    if not args.no_merge:
        merge_annotations(paths.outdir, None)
        merge_shards(paths.outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
