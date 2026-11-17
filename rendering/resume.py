# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resume and cadence bookkeeping shared by the Blender and Isaac launchers and workers.

Workers checkpoint their COCO records to ``annotations_worker_XXX.partial.json`` and write
``annotations_worker_XXX.json`` when done. A resumed launcher first *consolidates* every record
for its shard range from all such files (whatever worker layout produced them) into one snapshot,
``resume/resume_<start>_<end>.json``, and moves the consumed files aside; its workers then skip
the samples the snapshot (plus their own fresh checkpoint, after a watchdog relaunch) proves
complete. Resume therefore does not depend on ``gpus_per_shard``/``workers_per_gpu`` matching the
original run, only on the shard tiling.

Cadence-controlled appearance and asset caches refresh when a sample's group
``index // change_every`` changes. Sample-specific transforms still apply on every
rendered sample. Keeping the group tracker here makes that boundary testable.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RESUME_DIR = "resume"


def partial_annotations_path(outdir: str, worker_id: int) -> str:
    """Path of a worker's in-progress checkpoint."""
    return os.path.join(outdir, f"annotations_worker_{worker_id:03d}.partial.json")


def final_annotations_path(outdir: str, worker_id: int) -> str:
    """Path of a worker's completed annotation file."""
    return os.path.join(outdir, f"annotations_worker_{worker_id:03d}.json")


def snapshot_path(outdir: str, start_idx: int, end_idx: int) -> str:
    """Path of the resume snapshot for the shard ``[start_idx, end_idx)``."""
    return os.path.join(outdir, RESUME_DIR, f"resume_{start_idx:08d}_{end_idx:08d}.json")


def _worker_files(outdir: str) -> list[str]:
    """Final files first (authoritative), then partial checkpoints, each sorted by worker id."""
    finals = sorted(
        p
        for p in glob.glob(os.path.join(outdir, "annotations_worker_*.json"))
        if os.path.basename(p)[len("annotations_worker_") : -len(".json")].isdigit()
    )
    partials = sorted(glob.glob(os.path.join(outdir, "annotations_worker_*.partial.json")))
    return finals + partials


def _records_in_range(
    payload: dict[str, Any], start_idx: int, end_idx: int, images_per_sample: int
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    by_image = {}
    duplicates = set()
    for ann in annotations:
        image_id = ann.get("image_id")
        if image_id in by_image:
            duplicates.add(image_id)
        by_image[image_id] = ann
    image_counts = {}
    for img in images:
        image_counts[img["id"]] = image_counts.get(img["id"], 0) + 1
    for img in images:
        image_id = img["id"]
        if (
            image_counts[image_id] == 1
            and image_id in by_image
            and image_id not in duplicates
            and start_idx <= image_id // images_per_sample < end_idx
        ):
            yield img, by_image[image_id]


def consolidate_checkpoints(
    outdir: str, *, start_idx: int, end_idx: int, images_per_sample: int
) -> dict[str, Any]:
    """Gather checkpointed records for ``[start_idx, end_idx)`` into the resume snapshot.

    Sources, in priority order: the existing snapshot, final worker files, partial checkpoints; the
    first record seen for an image id wins. Worker files whose whole range lies inside the shard are
    moved to ``resume/consumed/<stamp>/`` after the snapshot is written (atomically), so the new
    workers' files are the only ``annotations_worker_*`` left and the merge sees no duplicates.
    Files that straddle the shard boundary are left alone (they belong to a different tiling).
    """
    snapshot = snapshot_path(outdir, start_idx, end_idx)
    seen: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    sources: list[str] = ([snapshot] if os.path.exists(snapshot) else []) + _worker_files(outdir)
    consumed: list[str] = []
    for path in sources:
        try:
            with open(path) as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue  # a half-written .tmp-less checkpoint; its samples are simply re-rendered
        for img, ann in _records_in_range(payload, start_idx, end_idx, images_per_sample):
            seen.setdefault(img["id"], (img, ann))
        inside = (
            path != snapshot
            and "start_idx" in payload
            and "end_idx" in payload
            and start_idx <= payload["start_idx"]
            and payload["end_idx"] <= end_idx
        )
        if inside:
            consumed.append(path)
    ids = sorted(seen)
    os.makedirs(os.path.dirname(snapshot), exist_ok=True)
    tmp = snapshot + ".tmp"
    with open(tmp, "w") as f:
        json.dump(
            {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "images_per_sample": images_per_sample,
                "images": [seen[i][0] for i in ids],
                "annotations": [seen[i][1] for i in ids],
            },
            f,
        )
    os.replace(tmp, snapshot)
    if consumed:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        dest = os.path.join(
            outdir, RESUME_DIR, "consumed", f"{stamp}_{start_idx:08d}_{end_idx:08d}"
        )
        os.makedirs(dest, exist_ok=True)
        for path in consumed:
            shutil.move(path, os.path.join(dest, os.path.basename(path)))
    return {"snapshot": snapshot, "records": len(ids), "consumed": consumed}


def _file_under(root: Path, resolved_root: Path, name: Any) -> bool:  # noqa: ANN401 - JSON value
    """Whether ``name`` is a plain relative path to an existing file inside ``root``."""
    if not isinstance(name, str) or not name:
        return False
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        return False
    target = (root / path).resolve()
    return target.is_relative_to(resolved_root) and target.is_file()


@dataclass
class ResumeState:
    """Samples a worker can skip, with their already-written image/annotation records."""

    done_indices: set[int] = field(default_factory=set)
    images: list[dict[str, Any]] = field(default_factory=list)
    annotations: list[dict[str, Any]] = field(default_factory=list)


def load_resume_state(
    outdir: str,
    worker_id: int,
    *,
    start_idx: int,
    end_idx: int,
    images_per_sample: int,
    render_masks: bool,
    snapshots: Iterable[str] | None = None,
) -> ResumeState:
    """Reconstruct which samples of ``[start_idx, end_idx)`` a previous run finished.

    Records come from the shard resume snapshots in ``resume/`` (written by the launcher's
    ``consolidate_checkpoints``) and from this worker's own ``.partial.json`` (fresh after a
    watchdog relaunch); the first record per image id wins. A sample counts as done only when
    all of its ``images_per_sample`` records are present and every referenced image (and mask,
    when masks are rendered) exists on disk; anything else is re-rendered. Records outside the
    worker's range are dropped.
    """
    state = ResumeState()
    if snapshots is None:
        snapshots = sorted(glob.glob(os.path.join(outdir, RESUME_DIR, "resume_*.json")))
    sources = [partial_annotations_path(outdir, worker_id), *snapshots]
    seen: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    for path in sources:
        try:
            with open(path) as f:
                previous = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue  # An unreadable checkpoint cannot prove completion; re-render its samples.
        for img, ann in _records_in_range(previous, start_idx, end_idx, images_per_sample):
            seen.setdefault(img["id"], (img, ann))
    by_sample: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    output_root = Path(outdir)
    images_root = output_root / "images"
    resolved_output, resolved_images = output_root.resolve(), images_root.resolve()
    for image_id in sorted(seen):
        by_sample.setdefault(image_id // images_per_sample, []).append(seen[image_id])
    for sample_idx, pairs in sorted(by_sample.items()):
        if not start_idx <= sample_idx < end_idx or len(pairs) != images_per_sample:
            continue

        files_ok = all(
            _file_under(images_root, resolved_images, img.get("file_name"))
            and (
                not render_masks or _file_under(output_root, resolved_output, ann.get("mask_file"))
            )
            for img, ann in pairs
        )
        if not files_ok:
            continue
        state.done_indices.add(sample_idx)
        for img, ann in pairs:
            state.images.append(img)
            state.annotations.append(ann)
    return state


def sort_by_image_id(
    images: list[dict[str, Any]], annotations: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(images, annotations)`` in image-id order.

    A resume appends new records after the loaded ones, so the lists need re-sorting.
    """
    ordered = sorted(images, key=lambda image: image["id"])
    by_image = {ann["image_id"]: ann for ann in annotations}
    if len(by_image) != len(annotations) or set(by_image) != {im["id"] for im in images}:
        raise ValueError("Annotation references must match unique image IDs")
    return ordered, [by_image[im["id"]] for im in ordered]


class CadenceTracker:
    """Decide when a cadence field (material, background, ...) must be (re)applied.

    Scene records key cadence fields on ``index // change_every``; the first sample a worker
    actually renders always applies everything (a resume may start mid-group), later samples only
    pay for the prim/datablock churn when their group differs from the last applied one.
    """

    def __init__(self) -> None:
        """Start with nothing applied."""
        self._applied: dict[str, int] = {}

    def changed(self, field_name: str, index: int, change_every: int) -> bool:
        """Return True, and remember the group, if ``field_name`` must be applied at ``index``."""
        group = index // max(1, int(change_every))
        if self._applied.get(field_name) == group:
            return False
        self._applied[field_name] = group
        return True


class FlushPolicy:
    """Checkpoint after ``every_samples`` renders or ``every_seconds`` of wall clock.

    Whichever trigger fires first wins.

    Slow renderers (path tracing at ~0.1 img/s per worker) would otherwise lose many minutes of
    work per kill with a count-only cadence; ``every_samples <= 0`` disables the count trigger and
    ``every_seconds <= 0`` the time trigger.
    """

    def __init__(
        self,
        every_samples: int,
        every_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Set the triggers; ``clock`` is injectable for tests."""
        self.every_samples = int(every_samples)
        self.every_seconds = float(every_seconds)
        self._clock = clock
        self._since_flush = 0
        self._last_flush = clock()

    def should_flush(self) -> bool:
        """Count one rendered sample and return True when a checkpoint is due."""
        self._since_flush += 1
        by_count = self.every_samples > 0 and self._since_flush >= self.every_samples
        by_time = self.every_seconds > 0 and self._clock() - self._last_flush >= self.every_seconds
        return by_count or by_time

    def flushed(self) -> None:
        """Record that a checkpoint was just written."""
        self._since_flush = 0
        self._last_flush = self._clock()
