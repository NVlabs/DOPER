# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small orchestration entrypoint for Blender rendering modes."""

from argparse import Namespace

from .context import BlenderContext
from .fixed_pose import render_fixed_requests
from .loop import render_samples
from .records import load_scene_records
from .settings import load_settings
from .setup import initialize


def run_worker(args: Namespace) -> None:
    """Run one Blender worker: initialize, then render fixed poses or the shard samples."""
    context = BlenderContext(args=args, settings=load_settings(args))
    load_scene_records(context)
    initialize(context)
    if getattr(args, "export_environment", False):
        # USD export dependencies are optional for ordinary Blender renders.
        from .environment_export import EnvironmentExporter

        context.environment_exporter = EnvironmentExporter(context)
    if render_fixed_requests(context):
        return
    render_samples(context)
