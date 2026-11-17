# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset shard execution and checkpoint cadence."""

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import time

import bpy

from rendering.resume import (
    CadenceTracker,
    FlushPolicy,
    load_resume_state,
    sort_by_image_id,
)

from .annotations import (
    report_kp_mask_check,
)
from .context import BlenderContext
from .legacy import realize_random_scene
from .output import write_worker_annotations
from .records import (
    realize_record,
)
from .views import render_views


def render_samples(ctx: BlenderContext) -> None:
    """Render this shard and checkpoint only complete samples."""
    # ---------------------------------------------------------------------------
    # Main render loop
    # ---------------------------------------------------------------------------
    ctx.annotations = []
    ctx.images = []

    done_indices = set()
    if ctx.args.resume:
        resume_state = load_resume_state(
            ctx.args.outdir,
            ctx.args.worker_id,
            start_idx=ctx.args.start_idx,
            end_idx=ctx.args.end_idx,
            images_per_sample=ctx.settings.images_per_sample,
            render_masks=ctx.settings.render_masks,
        )
        done_indices = resume_state.done_indices
        if ctx.environment_exporter is not None:
            for index in done_indices:
                ctx.environment_exporter.verify(ctx.scene_records[index])
        ctx.images.extend(resume_state.images)
        ctx.annotations.extend(resume_state.annotations)
        if done_indices:
            print(
                f"[W{ctx.args.worker_id}] resume: {len(done_indices)} samples already rendered, "
                "skipping them",
                flush=True,
            )

    t_start = time.time()
    rendered_now = 0
    # Cadence fields are keyed on index // change_every in the record; the tracker re-applies
    # them on the first rendered sample (a resume may start mid-group) and whenever the group
    # changes.
    cadence = CadenceTracker()
    flush_policy = FlushPolicy(ctx.args.flush_every, ctx.args.flush_every_sec)

    for view_idx in range(ctx.args.start_idx, ctx.args.end_idx):
        local_idx = view_idx - ctx.args.start_idx
        if view_idx in done_indices:
            continue
        t_iter_start = time.time()
        record = ctx.scene_records.get(view_idx)
        if ctx.args.scenes and record is None:
            raise SystemExit(f"no scene record for sample {view_idx} in {ctx.args.scenes}")

        if record is not None:
            realize_record(ctx, record, cadence)
        else:
            realize_random_scene(ctx, local_idx)

        # Datablocks replaced above (previous HDRI/texture images, ground materials, distractor
        # meshes) keep their memory until purged; without this a worker grows ~35 MB per sample.
        bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
        ctx.scene.frame_set(ctx.scene.frame_current)

        if ctx.environment_exporter is not None:
            ctx.environment_exporter.export(ctx, record)
        filename, visible_counts = render_views(ctx, view_idx, local_idx)

        t_iter = time.time() - t_iter_start
        rendered_now += 1
        elapsed = time.time() - t_start
        rate = rendered_now / elapsed if elapsed > 0 else 0
        remaining = (ctx.num_views - len(done_indices) - rendered_now) / rate if rate > 0 else 0

        if rendered_now % 10 == 0 or rendered_now == 1:
            print(
                f"[W{ctx.args.worker_id}] [{local_idx + 1}/{ctx.num_views}] {filename}  "
                f"vis_kp={','.join(visible_counts)}  "
                f"{t_iter:.2f}s/img  {rate:.1f} img/s  ETA {remaining / 60:.0f}min"
            )
        if flush_policy.should_flush():
            write_worker_annotations(ctx, partial=True)
            flush_policy.flushed()

    # ---------------------------------------------------------------------------
    # Save annotations
    # ---------------------------------------------------------------------------
    # Keep the final file in sample order even after a resume appended to the loaded records.
    ctx.images, ctx.annotations = sort_by_image_id(ctx.images, ctx.annotations)
    ann_path = write_worker_annotations(ctx, partial=False)

    report_kp_mask_check(ctx)

    elapsed_total = time.time() - t_start
    print(
        f"\n[W{ctx.args.worker_id}] DONE: {rendered_now} images rendered "
        f"({len(ctx.images)} total) in {elapsed_total / 60:.1f} min "
        f"({rendered_now / max(elapsed_total, 1e-9):.1f} img/s)"
    )
    print(f"[W{ctx.args.worker_id}] Saved → {ann_path}")
