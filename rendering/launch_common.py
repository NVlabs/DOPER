# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side helpers shared by the Blender and Isaac launchers.

Nothing here imports ``bpy`` or Kit; both launchers import it after putting the checkout on
``sys.path``.
"""

from __future__ import annotations

import argparse
import os
import re
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any


def count_shard_images(outdir: str | os.PathLike[str], start_idx: int, end_idx: int) -> int:
    """Count view files in a sample range; stereo filenames share a sample prefix."""
    count = 0
    for path in (Path(outdir) / "images").glob("*.png"):
        sample_prefix = path.stem.split("_", 1)[0]
        if sample_prefix.isdigit() and start_idx <= int(sample_prefix) < end_idx and path.is_file():
            count += 1
    return count


def split_range(start_idx: int, count: int, parts: int) -> list[tuple[int, int]]:
    """Divide ``count`` samples from ``start_idx`` evenly; the first remainder ranges get one more.

    Returns:
        ``parts`` half-open ``(start, end)`` ranges that tile ``[start_idx, start_idx + count)``.
    """
    base, remainder = divmod(count, parts)
    ranges = []
    start = start_idx
    for index in range(parts):
        end = start + base + (1 if index < remainder else 0)
        ranges.append((start, end))
        start = end
    return ranges


def add_shard_arguments(
    parser: argparse.ArgumentParser,
    *,
    progress_interval_sec: float,
    progress_help: str | None = None,
) -> None:
    """Add the sharding and resume flags both launchers expose."""
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="First sample index of this shard; [start_idx, start_idx + num_images) is rendered",
    )
    parser.add_argument(
        "--worker-id-offset",
        type=int,
        default=0,
        help="First worker id of this launcher, so several shards can write into one outdir",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Leave per-worker files unmerged (afterwards run rendering/run_manifest.py merge)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip samples whose images exist, reusing annotations from .partial.json checkpoints",
    )
    parser.add_argument(
        "--progress-interval-sec", type=float, default=progress_interval_sec, help=progress_help
    )


def check_shard_layout(
    outdir: Path,
    existing: Sequence[dict[str, Any]],
    *,
    start_idx: int,
    end_idx: int,
    worker_ids: Collection[int],
    resume: bool,
    exists_before_collision: bool,
) -> None:
    """Refuse range, worker-namespace and unidentified-output collisions (read only).

    Args:
        outdir: Output directory being launched into.
        existing: Shard manifests already recorded there.
        start_idx: First sample index of this shard.
        end_idx: One past the last sample index of this shard.
        worker_ids: Worker ids this launch will use.
        resume: Whether this launch resumes the recorded shard.
        exists_before_collision: Report an existing shard before a worker-id collision (the
            two launchers historically order these two fresh-launch checks differently).

    Raises:
        ValueError: Resume or shard layout does not match the recorded provenance.
    """
    overlapping = [
        m
        for m in existing
        if m["shard"]["start_idx"] < end_idx and m["shard"]["end_idx"] > start_idx
    ]
    if resume:
        recorded = [(m["shard"]["start_idx"], m["shard"]["end_idx"]) for m in overlapping]
        if recorded != [(start_idx, end_idx)]:
            raise ValueError(
                "Resume must retain the recorded shard range; worker layout may change"
            )
    exists_error = "Output shard already exists; use --resume or a fresh output directory"
    if exists_before_collision and not resume and overlapping:
        raise ValueError(exists_error)
    other_workers = {
        w["worker_id"] for m in existing if m not in overlapping for w in m["shard"]["workers"]
    }
    if other_workers.intersection(worker_ids):
        raise ValueError("Worker IDs overlap another shard; choose a distinct --worker-id-offset")
    if not exists_before_collision and not resume and overlapping:
        raise ValueError(exists_error)
    if not existing and outdir.exists() and any(outdir.iterdir()):
        raise ValueError("Existing output has no render identity; use a fresh output directory")


def require_min_version(
    name: str, found: str, minimum: tuple[int, int], *, prefix: str = ""
) -> None:
    """Exit unless ``found`` (after ``prefix``) starts with a ``major.minor`` >= ``minimum``.

    Raises:
        SystemExit: The version is missing or older than ``minimum``.
    """
    match = re.match(rf"{prefix}(\d+)\.(\d+)", found)
    if not match or (int(match[1]), int(match[2])) < minimum:
        required = ".".join(map(str, minimum))
        raise SystemExit(f"{name} {required} or newer is required; found {found!r}")
