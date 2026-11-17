#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Blender CLI for synthetic datasets and fixed-pose previews.

Run with: blender --background object.blend --python synth_worker.py -- ...
Implementation lives in rendering.blender; importing this file starts no work.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and validate the worker command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_device", type=int, default=0)
    parser.add_argument(
        "--keypoints_json",
        default=None,
        help="Explicit ordered keypoint JSON overriding the render YAML",
    )
    parser.add_argument(
        "--scenes",
        default=None,
        help=(
            "scenes.jsonl from rendering.scene_spec: realize these deterministic per-sample scene "
            "records (shared with the Isaac renderer) instead of sampling in this worker"
        ),
    )
    parser.add_argument(
        "--flush_every",
        type=int,
        default=100,
        help="Write an annotations_worker_XXX.partial.json checkpoint every N samples (0 disables)",
    )
    parser.add_argument(
        "--flush_every_sec",
        type=float,
        default=60.0,
        help="Also checkpoint when this many seconds passed since the last one (0 disables)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse annotations from an existing .partial.json and skip samples whose images exist",
    )
    parser.add_argument(
        "--fixed_pose_params",
        default=None,
        help="JSON file with fixed OpenCV rvec/tvec + real camera intrinsics for a one-off render",
    )
    parser.add_argument(
        "--render_out",
        default=None,
        help="Output PNG path when using --fixed_pose_params",
    )
    parser.add_argument(
        "--fixed_pose_manifest",
        default=None,
        help=(
            'JSON file with a list of {"params": {...}, "out": "path.png"} entries. '
            "Renders every entry in a single Blender session (batch fixed-pose mode)."
        ),
    )
    parser.add_argument(
        "--export_environment",
        action="store_true",
        help="Export native lights/World/ground alongside deterministic renders",
    )
    parser.add_argument(
        "--experimental_replay_ablation",
        choices=("none", "appearance-rng", "placement", "both"),
        default="none",
        help="Diagnostic historical replay: old appearance RNG and/or cadence-only placement",
    )
    args = parser.parse_args(argv)
    if args.experimental_replay_ablation != "none" and (
        not args.scenes
        or args.resume
        or args.export_environment
        or args.fixed_pose_params
        or args.fixed_pose_manifest
    ):
        parser.error(
            "Replay ablations require uninterrupted --scenes without resume, "
            "environment export, or fixed-pose modes"
        )
    if args.experimental_replay_ablation != "none" and (Path(args.outdir) / "manifests").exists():
        parser.error(
            "Replay ablations require an isolated diagnostic output without shard manifests"
        )
    if args.export_environment and (
        not args.scenes or args.fixed_pose_params or args.fixed_pose_manifest
    ):
        parser.error("--export_environment requires deterministic dataset scenes")
    if args.resume and not args.scenes:
        parser.error("--resume requires deterministic --scenes; random sampling cannot be resumed")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Parse ``argv`` and run the Blender worker."""
    args = parse_args(argv)
    # Blender's --python script entrypoint does not add the repository root.
    package_root = str(Path(__file__).resolve().parents[1])
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    # Keep bpy-dependent imports inside Blender; parse_args also runs on hosts.
    from rendering.blender.worker import run_worker

    run_worker(args)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    main(argv)
