# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atomic worker annotation checkpoints."""

import json
import os

from .context import BlenderContext, BpyObject


def write_worker_annotations(ctx: BlenderContext, *, partial: BpyObject) -> BpyObject:
    """Write the per-worker COCO file atomically.

    Called periodically with partial=True (a .partial.json checkpoint that --resume reuses, so a
    crash only loses the images since the last checkpoint) and once at the end with partial=False;
    a checkpoint never masquerades as the final file the merge step consumes.
    """
    payload = {
        "worker_id": ctx.args.worker_id,
        "start_idx": ctx.args.start_idx,
        "end_idx": ctx.args.end_idx,
        "num_samples": ctx.num_views,
        "images_per_sample": ctx.settings.images_per_sample,
        "render_masks": ctx.settings.render_masks,
        "mask_dir": ctx.settings.mask_dir_name if ctx.settings.render_masks else None,
        "stereo": {
            "enabled": ctx.settings.stereo_enabled,
            "baseline_m": round(ctx.settings.stereo_baseline_m, 6)
            if ctx.settings.stereo_enabled
            else None,
            "mode": "parallel" if ctx.settings.stereo_enabled else None,
        },
        "images": ctx.images,
        "annotations": ctx.annotations,
    }
    stem = f"annotations_worker_{ctx.args.worker_id:03d}"
    path = os.path.join(ctx.args.outdir, f"{stem}.partial.json" if partial else f"{stem}.json")
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)
    if not partial:
        partial_path = os.path.join(ctx.args.outdir, f"{stem}.partial.json")
        if os.path.exists(partial_path):
            os.remove(partial_path)
    return path
