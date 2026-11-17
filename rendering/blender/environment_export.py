# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in native environment export, preserving Blender scene and RNG state."""

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import tempfile
from pathlib import Path

import bpy

from rendering.environment_bundle import (
    publish_record,
    source_from_manifest,
    verify_unsealed_record,
)
from rendering.environment_bundle_usd import package_layer
from rendering.shard_io import load_shards

from .context import BlenderContext, BpyObject


class EnvironmentExporter:
    """Exports the native Blender environment of each rendered sample for a shard."""

    def __init__(self, ctx: BlenderContext) -> None:
        """Locate this worker's shard manifest and prepare the environments directory."""
        matches = [
            m
            for m in load_shards(ctx.args.outdir)
            if any(
                w["worker_id"] == ctx.args.worker_id
                and w["start_idx"] == ctx.args.start_idx
                and w["end_idx"] == ctx.args.end_idx
                for w in m["shard"]["workers"]
            )
        ]
        if len(matches) != 1 or matches[0]["status"] != "running":
            raise ValueError("environment export requires an active launcher-declared shard")
        manifest = matches[0]
        if (
            manifest["render_identity"]["config"]["renderer_settings"].get("export_environment")
            != 1
        ):
            raise ValueError("environment export not authorized by launch identity")
        self.source = source_from_manifest(manifest, bpy.app.version_string)
        self.root = Path(ctx.args.outdir) / "environments"
        self.root.mkdir(exist_ok=True)
        self.assets = {}
        self.verified = set()

    def verify(self, record: BpyObject) -> BpyObject:
        """Check that the exported environment for ``record`` is present and matches the scene."""
        return verify_unsealed_record(self.root, record, self.source, verified=self.verified)

    def export(self, ctx: BlenderContext, record: BpyObject) -> None:
        """Export the Blender-native environment of ``record`` unless a valid export exists."""
        if record["background"]["mode"] != "hdri":
            raise ValueError("native environment adapter currently requires an HDRI background")
        if (self.root / f"records/{record['index']:06d}.json").exists():
            self.verify(record)
            return
        lights = [obj for obj in ctx.scene.objects if obj.type == "LIGHT"]
        exported = lights + ([ctx.ground_obj] if ctx.ground_obj is not None else [])
        selected = list(bpy.context.selected_objects)
        active = bpy.context.view_layer.objects.active
        for obj in selected:
            obj.select_set(False)
        for obj in exported:
            obj.select_set(True)
        try:
            with tempfile.TemporaryDirectory(prefix=".export-", dir=self.root) as temporary:
                path = Path(temporary) / "environment.usda"
                result = bpy.ops.wm.usd_export(
                    filepath=str(path),
                    selected_objects_only=True,
                    export_materials=True,
                    export_cameras=False,
                    export_lights=True,
                    export_textures=False,
                    root_prim_path="/BlenderEnvironment",
                )
                if result != {"FINISHED"}:
                    raise RuntimeError(f"native environment export failed: {result}")
                layer, dependencies = package_layer(path, self.root, asset_cache=self.assets)
            publish_record(
                self.root,
                record,
                source=self.source,
                layer=layer,
                dependencies=dependencies,
                light_count=len(lights) + 1,
                ground=ctx.ground_obj is not None,
            )
        finally:
            for obj in exported:
                obj.select_set(False)
            for obj in selected:
                obj.select_set(True)
            bpy.context.view_layer.objects.active = active
