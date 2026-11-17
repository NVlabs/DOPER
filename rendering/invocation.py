# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure launcher arguments for prepared Blender and Isaac render inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

RENDERERS = ("blender", "isaac")


def render_backend(value: str) -> str:
    """Normalize an explicit backend and reject unsupported names."""
    renderer = str(value).strip().lower()
    if renderer not in RENDERERS:
        raise ValueError(f"unknown renderer {value!r}; expected one of {RENDERERS}")
    return renderer


@dataclass(frozen=True)
class RenderInvocation:
    """Prepared paths and explicit worker layout, independent of a scheduler."""

    renderer: str
    repo_root: Path
    python_bin: str
    config: Path
    outdir: Path
    num_images: int
    seed: int = 2748
    blend: Path | None = None
    usd: Path | None = None
    keypoints: Path | None = None
    blender_bin: str | None = None
    workers: int = 1
    gpus: tuple[int, ...] = (0,)
    workers_per_gpu: int = 1
    start_idx: int = 0
    worker_id_offset: int = 0
    no_merge: bool = False
    resume: bool = False


def build_render_command(inv: RenderInvocation) -> list[str]:
    """Build argv without filesystem access, asset export, or process execution."""
    renderer = render_backend(inv.renderer)
    if inv.num_images < 1 or inv.start_idx < 0 or inv.worker_id_offset < 0:
        raise ValueError("invalid render sample range or worker offset")
    script = "isaac/isaac_parallel.py" if renderer == "isaac" else "synth_parallel.py"
    cmd = [
        inv.python_bin,
        "-u",
        str(inv.repo_root / "rendering" / script),
        "--config",
        str(inv.config),
        "--outdir",
        str(inv.outdir),
        "--num-images",
        str(inv.num_images),
        "--seed",
        str(inv.seed),
        "--start-idx",
        str(inv.start_idx),
        "--worker-id-offset",
        str(inv.worker_id_offset),
    ]
    if inv.usd is not None:
        cmd += ["--usd", str(inv.usd)]
    if inv.keypoints is not None:
        cmd += ["--keypoints-json", str(inv.keypoints)]
    if renderer == "isaac":
        if inv.usd is None:
            raise ValueError("Isaac requires a prepared USD path")
        if (
            not inv.gpus
            or len(set(inv.gpus)) != len(inv.gpus)
            or any(gpu < 0 for gpu in inv.gpus)
            or inv.workers_per_gpu < 1
        ):
            raise ValueError("invalid Isaac worker layout")
        cmd += [
            "--repo-root",
            str(inv.repo_root),
            "--python-bin",
            inv.python_bin,
            "--gpus",
            ",".join(map(str, inv.gpus)),
            "--workers-per-gpu",
            str(inv.workers_per_gpu),
        ]
    else:
        if inv.blend is None or inv.workers < 1:
            raise ValueError("Blender requires a prepared blend path and positive worker count")
        cmd += ["--blend-file", str(inv.blend)]
        if inv.blender_bin is not None:
            cmd += ["--blender-bin", inv.blender_bin]
        cmd += ["--num-workers", str(min(inv.workers, inv.num_images))]
    if inv.no_merge:
        cmd.append("--no-merge")
    if inv.resume:
        cmd.append("--resume")
    return cmd
