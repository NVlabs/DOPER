# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native environment export contracts exercised inside the pinned Blender runtime."""

import pytest
from conftest import run_in_blender

pytestmark = pytest.mark.blender

EXPORT_CHECK = r"""
import json
import random
from pathlib import Path
from types import SimpleNamespace

import bpy

from rendering.blender import environment_export
from rendering.environment_bundle import BundleReader, seal_bundle

tmp_path = Path({tmp_path!r})
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
scene.world = bpy.data.worlds.new('ExportTestWorld')
scene.world.use_nodes = True
image = bpy.data.images.new('ExportTestHDRI', width=8, height=4, float_buffer=True)
image.filepath_raw = str(tmp_path / 'world.exr')
image.file_format = 'OPEN_EXR'
image.save()
texture = scene.world.node_tree.nodes.new('ShaderNodeTexEnvironment')
texture.image = image
background = scene.world.node_tree.nodes.get('Background')
scene.world.node_tree.links.new(texture.outputs['Color'], background.inputs['Color'])
bpy.ops.object.light_add(type='SUN')
bpy.ops.mesh.primitive_plane_add()
ground = bpy.context.object
bpy.ops.mesh.primitive_cube_add()
active = bpy.context.object
selected = list(bpy.context.selected_objects)
manifests = tmp_path / 'manifests'
manifests.mkdir()
manifest = {{
    'kind': 'render_shard', 'status': 'running',
    'shard': {{'workers': [{{'worker_id': 0, 'start_idx': 0, 'end_idx': 2}}]}},
    'render_identity': {{'object': {{'usd_sha256': 'b' * 64}},
                        'assets': {{'content_sha256': 'c' * 64}},
                        'config': {{'renderer_settings': {{'export_environment': 1}}}}}},
}}
(manifests / 'shard_00000000_00000002.json').write_text(json.dumps(manifest))
ctx = SimpleNamespace(args=SimpleNamespace(outdir=str(tmp_path), worker_id=0,
                      start_idx=0, end_idx=2), scene=scene, ground_obj=ground,
                      rng=random.Random(42))
exporter = environment_export.EnvironmentExporter(ctx)
record = {{'index': 0, 'seed': 42, 'background': {{'mode': 'hdri'}}}}
rng_before = ctx.rng.getstate()
global_before = random.getstate()
exporter.export(ctx, record)
assert ctx.rng.getstate() == rng_before
assert random.getstate() == global_before
assert list(bpy.context.selected_objects) == selected
assert bpy.context.view_layer.objects.active == active
metadata = exporter.verify(record)
assert metadata['light_count'] == 2  # One authored sun plus the world dome.
assert metadata['ground'] is True
seal_bundle(exporter.root, [record], source=exporter.source)
BundleReader(exporter.root, usd_sha256='b' * 64, seed=42).verify_record(record)

def fail(*args, **kwargs):
    raise RuntimeError('injected packaging failure')

environment_export.package_layer = fail
exporter.export(ctx, record)  # Reuse validates the record and avoids export.
try:
    exporter.export(ctx, {{**record, 'seed': 43}})
except ValueError as error:
    assert 'scene mismatch' in str(error), error
else:
    raise AssertionError('mismatched scene was reused')
try:
    exporter.export(ctx, {{**record, 'index': 1}})
except RuntimeError as error:
    assert 'packaging failure' in str(error), error
else:
    raise AssertionError('packaging failure was swallowed')
assert list(bpy.context.selected_objects) == selected
assert bpy.context.view_layer.objects.active == active
print('EXPORT_CONTRACTS_PASS')
"""

RESUME_CHECK = r"""
import sys
from types import SimpleNamespace

from rendering.blender import loop

record = {{'index': 0, 'seed': 42}}
verified = []

def reject(value):
    verified.append(value)
    raise ValueError('stale resumed export')

loop.load_resume_state = lambda *a, **kw: SimpleNamespace(
    done_indices={{0}}, images=[], annotations=[])
ctx = SimpleNamespace(args=SimpleNamespace(resume=True, outdir={tmp_path!r},
                      worker_id=0, start_idx=0, end_idx=1),
                      settings=SimpleNamespace(images_per_sample=1, render_masks=True),
                      environment_exporter=SimpleNamespace(verify=reject),
                      scene_records={{0: record}})
try:
    loop.render_samples(ctx)
except ValueError as error:
    assert 'stale resumed export' in str(error), error
else:
    raise AssertionError('resume skipped a sample without verifying its export')
assert verified == [record]
print('RESUME_CONTRACT_PASS')
"""

IMPORT_CHECK = r"""
import sys
import rendering.blender.environment_export
leaked = {{'scipy', 'doper'}} & {{name.split('.')[0] for name in sys.modules}}
assert not leaked, f'environment export imported host-only packages: {{leaked}}'
print('IMPORTS_CLEAN')
"""


def run_check(blender, tmp_path, template, marker):
    result = run_in_blender(blender, template.format(tmp_path=str(tmp_path)), cwd=tmp_path)
    assert result.returncode == 0 and marker in result.stdout, result.stdout + result.stderr


def test_native_export_preserves_rng_selection_and_revalidates_reuse(blender, tmp_path):
    run_check(blender, tmp_path, EXPORT_CHECK, "EXPORT_CONTRACTS_PASS")


def test_resume_verifies_export_before_skipping_completed_sample(blender, tmp_path):
    run_check(blender, tmp_path, RESUME_CHECK, "RESUME_CONTRACT_PASS")


def test_environment_export_imports_without_scipy(blender, tmp_path):
    run_check(blender, tmp_path, IMPORT_CHECK, "IMPORTS_CLEAN")
