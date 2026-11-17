#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Parallel Blender launcher for synthetic keypoint datasets.

Spawns several ``synth_worker.py`` Blender processes that render disjoint sample ranges of one
dataset, then merges their annotations. Scenes are sampled once, deterministically per sample
index, by ``rendering.scene_spec`` (shared with the Isaac launcher). Requires Blender 4.2 or newer.

Usage:
    python rendering/synth_parallel.py --blend-file object.blend --config render.yaml \\
        --outdir out --usd object.usda --num-images 100 --num-workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Direct script execution must resolve shared renderer modules from this checkout.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rendering.blender_inputs import inspect_blender_inputs  # noqa: E402
from rendering.config import RenderConfig, load_render_config  # noqa: E402
from rendering.dataset_contract import merge_annotations, validate_dataset_records  # noqa: E402
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
    portable_path,
    shard_manifest,
    shard_manifest_path,
    start_shard_attempt,
    validate_resume_identity,
)
from rendering.scene_spec import AssetCatalog, generate_scenes  # noqa: E402
from rendering.usd_mesh import read_mesh_usda  # noqa: E402

MIN_BLENDER_VERSION = (4, 2)
BLENDER_ENV = "DOPER_BLENDER"
ATOL = 1e-6


@dataclass(frozen=True)
class Worker:
    """One Blender worker and the sample range ``[start_idx, end_idx)`` it renders."""

    worker_id: int
    start_idx: int
    end_idx: int
    log_file: Path


def blender_version_line(blender_bin: str) -> str:
    """Return the first line of ``blender --version`` after checking it is 4.2 or newer.

    Raises:
        SystemExit: Blender is missing, unreadable, or older than 4.2.
    """
    try:
        result = subprocess.run([blender_bin, "--version"], capture_output=True, text=True)
    except OSError as err:
        raise SystemExit(
            f"Blender not found at {blender_bin!r} ({err}); use --blender-bin or set {BLENDER_ENV}"
        ) from err
    lines = result.stdout.splitlines()
    line = lines[0] if lines else ""
    match = re.match(r"Blender (\d+)\.(\d+)", line)
    if result.returncode != 0 or not match:
        raise SystemExit(f"cannot read the Blender version from {blender_bin!r}: {line!r}")
    require_min_version("Blender", line, MIN_BLENDER_VERSION, prefix="Blender ")
    return line


def generate_blender_scenes(
    cfg: RenderConfig,
    usd_path: str,
    keypoints_path: str | None,
    seed: int,
    start_idx: int,
    end_idx: int,
    outdir: Path,
) -> dict[str, Any]:
    """Write ``scenes/scenes_<start>_<end>.jsonl`` for the Blender workers.

    Uses the same sampler as the Isaac launcher, so a Blender and an Isaac render of one config
    and seed realize identical scenes.

    Returns:
        The ``generate_scenes`` result plus ``cfg``, ``usd_path`` and ``keypoints_path`` (used
        for the shard manifest).
    """
    scenes_dir = outdir / "scenes"
    scenes_dir.mkdir(exist_ok=True)
    scenes_path = scenes_dir / f"scenes_{start_idx:08d}_{end_idx:08d}.jsonl"
    generated = generate_scenes(
        cfg,
        usd_path=usd_path,
        keypoints_json=keypoints_path,
        seed=seed,
        start_idx=start_idx,
        end_idx=end_idx,
        out_path=str(scenes_path),
    )
    summary = generated["catalog"].summary()
    print(f"Scenes: {scenes_path.name} ({generated['count']} records, assets {summary})")
    return {**generated, "cfg": cfg, "usd_path": usd_path, "keypoints_path": keypoints_path}


def validate_blender_inputs(
    blender_bin: str,
    blend_file: str | os.PathLike[str],
    cfg: RenderConfig,
    usd_path: str,
    keypoints_json: str | None,
) -> tuple[dict[str, Any], str | None]:
    """Validate authored geometry against the object USD and any supplied keypoints.

    Returns:
        The inspected Blender inputs and the keypoint JSON path in use (or ``None``).

    Raises:
        ValueError: The USD is not a writer-generated ``.usda``, or Blender, USD and keypoints
            disagree.
    """
    if Path(usd_path).suffix.lower() != ".usda":
        raise ValueError(
            "deterministic Blender rendering requires a writer-generated .usda layer "
            "from rendering.usd_mesh.write_mesh_usda; re-export the object "
            f"instead of using {usd_path!r}"
        )
    authored = inspect_blender_inputs(
        blender_bin,
        str(blend_file),
        cfg.mesh_name,
        cfg.keypoint_prefix,
        cfg.num_keypoints,
        object_material_mode=cfg.object_material_mode,
    )
    exported = read_mesh_usda(usd_path)
    for field in ("points", "keypoints_3d"):
        actual, expected = np.asarray(authored[field]), np.asarray(exported[field])
        if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0, atol=ATOL):
            raise ValueError(f"Blender/USD {field} differ; re-export the object before rendering")
    if authored["keypoint_names"] != exported["keypoint_names"]:
        raise ValueError("Blender/USD ordered keypoint names differ")
    keypoints_path = cfg.resolve(keypoints_json or cfg.keypoints_json)
    if keypoints_path:
        supplied = json.loads(Path(keypoints_path).read_text())
        supplied_points = np.asarray(supplied["keypoints_3d"])
        if supplied.get("object", cfg.mesh_name) != cfg.mesh_name:
            raise ValueError(
                f"Supplied keypoints are for object {supplied['object']!r} but the config "
                f"object.mesh_name is {cfg.mesh_name!r}"
            )
        if (
            supplied["keypoint_names"] != authored["keypoint_names"]
            or supplied_points.shape != np.asarray(authored["keypoints_3d"]).shape
            or not np.allclose(supplied_points, authored["keypoints_3d"], rtol=0, atol=ATOL)
        ):
            raise ValueError("Supplied keypoints differ from the authored Blender object frame")
    return authored, keypoints_path


def validate_worker_outputs(
    outdir: str | os.PathLike[str],
    workers: Sequence[Worker],
    *,
    images_per_sample: int = 1,
    render_masks: bool = False,
    num_keypoints: int | None = None,
) -> list[str]:
    """Return validation errors for missing or incomplete worker annotations."""
    root = Path(outdir)
    errors: list[str] = []
    for worker in workers:
        ann_file = root / f"annotations_worker_{worker.worker_id:03d}.json"
        if not ann_file.exists():
            errors.append(
                f"Worker {worker.worker_id}: missing {portable_path(ann_file, root)}; "
                f"see {portable_path(worker.log_file, root)}"
            )
            continue
        try:
            data = json.loads(ann_file.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(
                f"Worker {worker.worker_id}: cannot read {portable_path(ann_file, root)}: {exc}; "
                f"see {portable_path(worker.log_file, root)}"
            )
            continue
        problems = validate_dataset_records(
            data,
            root,
            start_idx=worker.start_idx,
            end_idx=worker.end_idx,
            images_per_sample=images_per_sample,
            render_masks=render_masks,
            num_keypoints=num_keypoints,
        )
        errors.extend(
            f"Worker {worker.worker_id}: {p}; see {portable_path(worker.log_file, root)}"
            for p in problems
        )
    return errors


def build_parser() -> argparse.ArgumentParser:
    """Build the launcher command line."""
    parser = argparse.ArgumentParser(description="Parallel Blender synthetic data generation")
    parser.add_argument("--blend-file", required=True, help="Blend file with the object mesh")
    parser.add_argument("--config", required=True, help="Render config (YAML)")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--num-images", type=int, default=50000, help="Samples to render")
    parser.add_argument("--num-workers", type=int, default=8, help="Parallel Blender workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Must equal output.resolution of the config when given (the config is the source)",
    )
    parser.add_argument(
        "--blender-bin",
        default=None,
        help=f"Blender 4.2+ binary (default: ${BLENDER_ENV}, then 'blender' on PATH)",
    )
    parser.add_argument(
        "--usd",
        default=None,
        help="Object .usda (else object.usd_path from the config). When available, scenes are "
        "sampled deterministically per index by rendering/scene_spec.py and shared with the "
        "Isaac renderer; without it the legacy per-worker random loop is used.",
    )
    parser.add_argument(
        "--keypoints-json", default=None, help="Override object.keypoints_json for --usd"
    )
    parser.add_argument(
        "--export-environment",
        action="store_true",
        help="Export portable native environments; requires deterministic scenes",
    )
    parser.add_argument(
        "--no-scenes", action="store_true", help="Force the legacy in-worker randomization"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the config and inputs, print the plan and exit; starts no Blender and "
        "creates no output directory",
    )
    # Sharding surface shared with rendering/isaac/isaac_parallel.py.
    add_shard_arguments(
        parser,
        progress_interval_sec=float(os.environ.get("SYNTH_PROGRESS_INTERVAL_SEC", "1.0")),
        progress_help="Seconds between rendered image count updates",
    )
    return parser


def check_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject inconsistent counts and flag combinations."""
    if args.num_images <= 0 or args.num_workers <= 0:
        parser.error("positive image and worker counts required")
    if args.start_idx < 0 or args.worker_id_offset < 0:
        parser.error("nonnegative sample range and worker offsets required")
    if args.resume and args.no_scenes:
        parser.error("Legacy randomization cannot safely resume; use a fresh output directory")


def split_workers(args: argparse.Namespace, outdir: Path) -> list[Worker]:
    """Divide the sample range evenly; the first ``remainder`` workers take one extra sample."""
    workers = []
    ranges = split_range(args.start_idx, args.num_images, args.num_workers)
    for local_id, (start, end) in enumerate(ranges):
        worker_id = args.worker_id_offset + local_id
        log_file = outdir / f"worker_{worker_id:03d}.log"
        workers.append(Worker(worker_id, start, end, log_file))
    return workers


def check_output_state(
    args: argparse.Namespace,
    outdir: Path,
    identity: RenderIdentity,
    shard_end: int,
) -> None:
    """Fail closed when the output directory holds a conflicting or unidentified render.

    Raises:
        ValueError: Resume or shard layout does not match the recorded provenance.
    """
    existing = load_shards(outdir)
    if args.resume or existing:
        validate_resume_identity(outdir, identity, require_existing=args.resume)
    check_shard_layout(
        outdir,
        existing,
        start_idx=args.start_idx,
        end_idx=shard_end,
        worker_ids=range(args.worker_id_offset, args.worker_id_offset + args.num_workers),
        resume=args.resume,
        exists_before_collision=False,
    )


def consolidate_for_resume(
    args: argparse.Namespace, cfg: RenderConfig, outdir: Path, shard_end: int
) -> list[int]:
    """Fold existing checkpoints into one snapshot; return the sample indices already done.

    The snapshot lets the worker count of this run differ from the one that produced the
    checkpoints (see ``rendering/resume.py``).
    """
    info = consolidate_checkpoints(
        str(outdir),
        start_idx=args.start_idx,
        end_idx=shard_end,
        images_per_sample=cfg.images_per_sample,
    )
    state = load_resume_state(
        str(outdir),
        args.worker_id_offset,
        start_idx=args.start_idx,
        end_idx=shard_end,
        images_per_sample=cfg.images_per_sample,
        render_masks=cfg.render_masks,
        snapshots=[info["snapshot"]],
    )
    print(
        f"Resume: {info['records']} checkpointed records for this shard in "
        f"{Path(info['snapshot']).name} ({len(info['consumed'])} worker file(s) consolidated)"
    )
    return sorted(state.done_indices)


def write_shard_manifest(
    scenes: dict[str, Any],
    config_file: Path,
    args: argparse.Namespace,
    outdir: Path,
    workers: Sequence[Worker],
    *,
    identity: RenderIdentity,
    reused_samples: Sequence[int] = (),
) -> Path:
    """Persist the provenance record of this shard attempt; the merge step validates it."""
    shard_end = workers[-1].end_idx
    path = shard_manifest_path(outdir, args.start_idx, shard_end)
    manifest = shard_manifest(
        repo_root=str(REPO_ROOT),
        cfg=scenes["cfg"],
        config_path=str(config_file),
        usd_path=scenes["usd_path"],
        keypoints_path=scenes["keypoints_path"],
        num_keypoints=scenes["num_keypoints"],
        catalog_summary=scenes["catalog"].summary(),
        seed=args.seed,
        start_idx=args.start_idx,
        end_idx=shard_end,
        workers=[
            {"worker_id": w.worker_id, "gpu": 0, "start_idx": w.start_idx, "end_idx": w.end_idx}
            for w in workers
        ],
        scenes_path=scenes["path"],
        renderer="blender",
        outdir=str(outdir),
    )
    manifest["reused_sample_indices"] = list(reused_samples)
    start_shard_attempt(path, manifest, identity)
    return path


def worker_command(
    args: argparse.Namespace,
    worker: Worker,
    blender_bin: str,
    blend_file: Path,
    config_file: Path,
    outdir: Path,
    scenes_path: str | None,
    keypoints_path: str | None,
) -> list[str]:
    """Blender command line for one worker; ``synth_worker.py`` keeps its underscore flags."""
    cmd = [
        blender_bin,
        "--background",
        "--disable-autoexec",
        str(blend_file),
        "--python-exit-code",
        "1",
        "--python",
        str(Path(__file__).resolve().parent / "synth_worker.py"),
        "--",
        "--config",
        str(config_file),
        "--outdir",
        str(outdir),
        "--worker_id",
        str(worker.worker_id),
        "--start_idx",
        str(worker.start_idx),
        "--end_idx",
        str(worker.end_idx),
        "--seed",
        str(args.seed),
    ]
    if scenes_path:
        cmd += ["--scenes", scenes_path]
    if keypoints_path:
        cmd += ["--keypoints_json", keypoints_path]
    if args.export_environment:
        cmd.append("--export_environment")
    if args.resume:
        cmd.append("--resume")
    return cmd


def wait_for_workers(
    processes: list[tuple[Worker, subprocess.Popen[bytes]]],
    args: argparse.Namespace,
    cfg: RenderConfig,
    shard_end: int,
    t_start: float,
) -> None:
    """Poll until every worker exits, printing completions and image-count progress."""
    done = [False] * len(processes)
    interval = max(0.2, args.progress_interval_sec)
    total = args.num_images * cfg.images_per_sample
    outdir = processes[0][0].log_file.parent
    while not all(done):
        time.sleep(interval)
        for i, (worker, proc) in enumerate(processes):
            code = None if done[i] else proc.poll()
            if code is not None:
                done[i] = True
                status = "OK" if code == 0 else f"FAILED (code {code})"
                print(
                    f"  Worker {worker.worker_id} finished: {status} "
                    f"({(time.time() - t_start) / 60:.1f} min)"
                )
        rendered = count_shard_images(outdir, args.start_idx, shard_end)
        print(
            f"  Progress: {rendered}/{total} images rendered, {sum(done)}/{len(done)} workers "
            f"done ({(time.time() - t_start) / 60:.1f} min)",
            flush=True,
        )


def merge_outputs(
    args: argparse.Namespace, outdir: Path, shard_end: int, manifest_path: Path | None
) -> None:
    """Merge worker annotations (and shard manifests) into the dataset-level files."""
    print("\nMerging annotations...")
    counts = merge_annotations(outdir, None if args.worker_id_offset else args.num_workers)
    if manifest_path:
        merge_shards(outdir, expect_total=None if args.worker_id_offset else shard_end)
    print(f"\n{'=' * 60}\nCOMPLETE\n{'=' * 60}")
    print(f"Output: {outdir}/")
    print(f"  samples requested: {args.num_images}")
    print(f"  images/: {counts['images']} PNG files")
    if counts.get("masks"):
        print("  masks/: visible-object PNG files")
    print("  annotations.json: COCO keypoint format")


def print_dry_run_plan(
    args: argparse.Namespace,
    cfg: RenderConfig,
    config_file: Path,
    outdir: Path,
    usd_path: str | None,
    workers: Sequence[Worker],
) -> None:
    """Print the resolved render plan of a validated ``--dry-run``; writes nothing."""
    print(f"\n{'=' * 60}\nDry run: nothing was launched or written\n{'=' * 60}")
    print(f"Config: {config_file}\nObject USD: {usd_path or 'none (legacy randomization)'}")
    print(f"Output (not created): {outdir}")
    print(f"Resolution: {cfg.resolution}x{cfg.resolution}; keypoints: {cfg.num_keypoints}")
    print(f"Samples: {args.num_images} x {cfg.images_per_sample} image(s); seed {args.seed}")
    print(f"Workers: {args.num_workers}")
    for worker in workers:
        count = worker.end_idx - worker.start_idx
        print(
            f"  Worker {worker.worker_id}: samples [{worker.start_idx}, {worker.end_idx}) = {count}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the launcher and return the process exit status.

    Raises:
        SystemExit: Invalid arguments, a missing input, an unsupported Blender, or failed
            workers.
        ValueError: The output directory or render identity conflicts with this run.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    check_arguments(parser, args)
    blender_bin = args.blender_bin or os.environ.get(BLENDER_ENV) or shutil.which("blender")
    blender_bin = blender_bin or "blender"
    # The version check starts Blender, so a dry run skips it (and has no version).
    version_line = None if args.dry_run else blender_version_line(blender_bin)
    if version_line is not None:
        print(f"Using Blender: {version_line}")
    blend_file = Path(args.blend_file).resolve()
    if not blend_file.is_file():
        raise SystemExit(f"Blend file not found: {blend_file}")
    config_file = Path(args.config).resolve()
    outdir = Path(args.outdir).resolve()
    cfg = load_render_config(config_file)
    if args.resolution is not None and args.resolution != cfg.resolution:
        parser.error(
            f"--resolution {args.resolution} differs from output.resolution {cfg.resolution} "
            "in the config; set it there"
        )
    if args.usd and not Path(args.usd).is_file():
        raise SystemExit(f"Object USD not found: {args.usd}")
    usd_path = os.path.abspath(args.usd) if args.usd else cfg.resolve(cfg.usd_path)
    if usd_path and not args.no_scenes and not Path(usd_path).is_file():
        raise SystemExit(f"Object USD not found: {usd_path}")
    has_scenes = not args.no_scenes and bool(usd_path)
    if not args.no_scenes and not has_scenes:
        print(
            "WARNING: no object USD (--usd or object.usd_path): using legacy per-worker "
            "randomization; scenes will not match an Isaac render of this config."
        )
    keypoints_arg = os.path.abspath(args.keypoints_json) if args.keypoints_json else None
    require_backend(
        cfg.robustness,
        "blender",
        scenes=has_scenes,
        export_environment=args.export_environment,
        appearance=cfg.robust_appearance,
    )
    if args.export_environment and not has_scenes:
        parser.error("--export-environment requires deterministic scenes")
    if args.resume and not has_scenes:
        parser.error("Resume requires deterministic scenes and a matching recorded identity")
    workers = split_workers(args, outdir)
    shard_end = workers[-1].end_idx
    if args.dry_run:
        print_dry_run_plan(args, cfg, config_file, outdir, usd_path, workers)
        return 0
    assert version_line is not None  # only a dry run leaves it unset
    print(f"\n{'=' * 60}\nSynthetic Data Parallel Rendering\n{'=' * 60}")
    print(f"Total images: {args.num_images}\nWorkers: {args.num_workers}\nOutput: {outdir}")
    print(f"Resolution: {cfg.resolution}x{cfg.resolution}\nConfig: {config_file}\n")
    scenes: dict[str, Any] | None = None
    manifest_path: Path | None = None
    with output_lock(outdir):
        identity: RenderIdentity | None = None
        if has_scenes and usd_path:
            authored, keypoints_path = validate_blender_inputs(
                blender_bin, blend_file, cfg, usd_path, keypoints_arg
            )
            settings: dict[str, Any] = {"blender_version": version_line}
            if args.export_environment:
                settings["export_environment"] = 1
            identity = build_render_identity(
                cfg=cfg,
                catalog=AssetCatalog.scan(cfg),
                seed=args.seed,
                renderer="blender",
                renderer_version="blender-scenes-v3",
                blend_path=str(blend_file),
                usd_path=usd_path,
                keypoints_path=keypoints_path,
                texture_paths=authored["external_files"],
                renderer_settings=settings,
            )
            check_output_state(args, outdir, identity, shard_end)
        elif outdir.exists() and any(outdir.iterdir()):
            raise ValueError("Legacy rendering requires a fresh output directory")
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "images").mkdir(exist_ok=True)
        if has_scenes and usd_path:
            scenes = generate_blender_scenes(
                cfg,
                usd_path,
                cfg.resolve(keypoints_arg or cfg.keypoints_json),
                args.seed,
                args.start_idx,
                shard_end,
                outdir,
            )
            assert identity is not None  # set with the same condition above
            reused = consolidate_for_resume(args, cfg, outdir, shard_end) if args.resume else []
            manifest_path = write_shard_manifest(
                scenes, config_file, args, outdir, workers, identity=identity, reused_samples=reused
            )
    print(f"Launching {args.num_workers} workers...")
    t_start = time.time()
    processes: list[tuple[Worker, subprocess.Popen[bytes]]] = []
    for worker in workers:
        cmd = worker_command(
            args,
            worker,
            blender_bin,
            blend_file,
            config_file,
            outdir,
            str(scenes["path"]) if scenes else None,
            keypoints_arg,
        )
        count = worker.end_idx - worker.start_idx
        print(
            f"  Worker {worker.worker_id}: samples [{worker.start_idx}, {worker.end_idx}) = {count}"
        )
        with open(worker.log_file, "w") as log:
            processes.append((worker, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)))
    print(f"\nAll workers launched. Logs in {outdir}/worker_*.log\nWaiting for completion...")
    wait_for_workers(processes, args, cfg, shard_end, t_start)
    total_time = time.time() - t_start
    rendered = count_shard_images(outdir, args.start_idx, shard_end)
    # Blender can exit zero when its script raises, so annotation files are validated as well.
    failed = [(w.worker_id, p.returncode) for w, p in processes if p.returncode != 0]
    for worker_id, code in failed:
        print(f"Worker {worker_id}: exit code {code}; see {outdir}/worker_{worker_id:03d}.log")
    invalid = validate_worker_outputs(
        outdir,
        workers,
        images_per_sample=cfg.images_per_sample,
        render_masks=cfg.render_masks,
        num_keypoints=scenes["num_keypoints"] if scenes else cfg.num_keypoints,
    )
    if manifest_path:
        finish_shard(
            manifest_path,
            images_rendered=rendered,
            elapsed_sec=total_time,
            failed_workers=[w for w, _ in failed],
            problems=invalid,
        )
    if failed or invalid:
        lines = "\n".join(f"  {message}" for message in invalid)
        raise SystemExit(f"{len(failed)} workers failed; {len(invalid)} outputs invalid:\n{lines}")
    rate = args.num_images / max(total_time, 1e-9)
    if args.no_merge:
        print(
            f"\nSHARD COMPLETE: {rendered} images in {total_time / 60:.1f} min ({rate:.2f} "
            f"samples/s); merge deferred (once every shard finished, run "
            f"python rendering/run_manifest.py merge {outdir})"
        )
        return 0
    merge_outputs(args, outdir, shard_end, manifest_path)
    print(f"Total time: {total_time / 60:.1f} min; throughput {rate:.2f} samples/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
