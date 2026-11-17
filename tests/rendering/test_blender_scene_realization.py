# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded Blender checks of sample placement across cache/worker boundaries."""

import subprocess
import sys

import pytest
from conftest import REPO_ROOT

from rendering.synth_worker import parse_args

# The worker CLI requires a config and an output directory.
REQUIRED = ["--config", "render.yaml", "--outdir", "dataset"]


@pytest.mark.blender
def test_realized_transforms_are_independent_of_previous_sample(blender, tmp_path):
    script = tmp_path / "check.py"
    script.write_text("""
import random, sys
from pathlib import Path
import bpy
from mathutils import Matrix, Vector, Euler
sys.path.insert(0, str(Path.cwd()))
from types import SimpleNamespace
from rendering.blender.context import BlenderContext
from rendering.blender import records
ctx = BlenderContext(args=SimpleNamespace(), settings=SimpleNamespace(base_dir=str(Path.cwd())),
                     rng=random.Random(), scene_scale=2.0)
bpy.ops.mesh.primitive_plane_add()
ctx.ground_obj = bpy.context.object
bpy.ops.mesh.primitive_cube_add()
obj = bpy.context.object
original = Matrix.Translation(Vector((.2, .1, .3)))
ctx.record_distractor_instances = [(0, obj, original, .4)]
a = {'seed': 42, 'index': 0, 'distractors': [{'position': [1,2,3], 'rotation_xyz': [0,.1,0]}]}
b = {'seed': 42, 'index': 3, 'distractors': [{'position': [-1,4,.5], 'rotation_xyz': [.3,0,.2]}]}
records.apply_record_transforms(ctx, a, .01)
records.apply_record_transforms(ctx, b, .03)
continuous = obj.matrix_world.copy(), ctx.ground_obj.matrix_world.copy()
# A new worker/resume starts from imported authored coordinates mid-cadence.
obj.matrix_world = original

ctx.ground_obj.location.z = 100
records.apply_record_transforms(ctx, b, .03)
assert all(abs(continuous[0][i][j]-obj.matrix_world[i][j]) < 1e-6 for i in range(4) for j in range(4))
assert all(abs(continuous[1][i][j]-ctx.ground_obj.matrix_world[i][j]) < 1e-6 for i in range(4) for j in range(4))
assert abs(ctx.ground_obj.location.z - .028) < 1e-6
# Backend-specific material draws use the group; lights always use the sample.
records.seed_rng_for_record(ctx, a, 'material', 0); first = ctx.rng.random()
records.seed_rng_for_record(ctx, b, 'material', 0); assert ctx.rng.random() == first
records.seed_rng_for_record(ctx, b, 'lights'); uninterrupted = ctx.rng.random()
ctx.rng.random(); ctx.rng.random()
records.seed_rng_for_record(ctx, b, 'lights'); assert ctx.rng.random() == uninterrupted
# Imported mesh hierarchies must retain authored world transforms before any placement.
real_bpy = bpy
ctx.mesh_size = .1
records.has_valid_material = lambda obj: True
originals = {}
def import_hierarchy(filepath):
    real_bpy.ops.mesh.primitive_cube_add(size=.2, location=(.3,0,0))
    parent = real_bpy.context.object
    parent.name = 'A_parent'
    real_bpy.ops.mesh.primitive_cube_add(size=.1)
    child = real_bpy.context.object
    child.name = 'B_child'
    child.parent = parent
    child.location = (.1,.2,0)
    real_bpy.context.view_layer.update()
    originals.update({obj.name: obj.matrix_world.copy() for obj in (parent,child)})
class OrderedSet(set):
    def __sub__(self, other):
        return sorted(super().__sub__(other))
records.set = OrderedSet
records.bpy = SimpleNamespace(data=real_bpy.data, context=real_bpy.context,
    ops=SimpleNamespace(wm=SimpleNamespace(usd_import=import_hierarchy)))
record = {'distractors':[{'file':'part.usda','position':[1,2,3],
                         'rotation_xyz':[0,0,.5],'size_frac':.2,'material':{}}]}
records.load_record_distractors(ctx, record)
real_bpy.context.view_layer.update()
assert len(ctx.record_distractor_instances) == 2
for index, obj, authored, scale in ctx.record_distractor_instances:
    assert all(abs(authored[i][j]-originals[obj.name][i][j]) < 1e-6 for i in range(4) for j in range(4))
    expected = Matrix.Translation(Vector((1,2,3))) @ Euler((0,0,.5),'XYZ').to_matrix().to_4x4() @ Matrix.Scale(scale,4) @ originals[obj.name]
    assert all(abs(obj.matrix_world[i][j]-expected[i][j]) < 1e-6 for i in range(4) for j in range(4))
print('REALIZED_TRANSFORMS_MATCH')
""")
    result = subprocess.run(
        [
            blender,
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
            str(script),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "REALIZED_TRANSFORMS_MATCH" in result.stdout


@pytest.mark.blender
def test_no_ground_draw_removes_existing_plane(blender, tmp_path):
    script = tmp_path / "ground.py"
    script.write_text("""
import sys
from pathlib import Path
from types import SimpleNamespace
import bpy
sys.path.insert(0, str(Path.cwd()))
from rendering.blender.context import BlenderContext
from rendering.blender.environment import update_random_ground
bpy.ops.mesh.primitive_plane_add()
plane = bpy.context.object
name = plane.name
ctx = BlenderContext(args=SimpleNamespace(), settings=SimpleNamespace(no_ground_prob=1.0),
                     ground_obj=plane)
assert name in bpy.context.scene.objects
update_random_ground(ctx, 0.01)
assert ctx.ground_obj is None
assert name not in bpy.data.objects
assert name not in bpy.context.scene.objects
# Consecutive no-ground draws must also be valid once the plane is absent.
update_random_ground(ctx, 0.02)
assert ctx.ground_obj is None
print('GROUND_REMOVED')
""")
    result = subprocess.run(
        [
            blender,
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
            str(script),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "GROUND_REMOVED" in result.stdout


def test_cli_import_and_resume_validation_do_not_require_blender():
    script = """
import importlib.abc
import sys
class NoBlender(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'bpy', 'mathutils'}:
            raise AssertionError('CLI imported Blender before validating arguments')
sys.meta_path.insert(0, NoBlender())
from rendering import synth_worker
assert 'rendering.blender.worker' not in sys.modules
args = synth_worker.parse_args(['--config', 'c.yaml', '--outdir', 'o', '--resume', '--scenes', 'scenes.jsonl'])
assert args.resume and args.scenes == 'scenes.jsonl'
try:
    synth_worker.main(['--config', 'c.yaml', '--outdir', 'o', '--resume'])
except SystemExit as exc:
    assert exc.code == 2
else:
    raise AssertionError('unsafe random-mode resume was accepted')
print('CLI_IMPORT_SAFE')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--resume requires deterministic --scenes" in result.stderr
    assert "CLI_IMPORT_SAFE" in result.stdout


@pytest.mark.parametrize("mode", ["appearance-rng", "placement", "both"])
def test_replay_ablation_requires_uninterrupted_scene_mode(mode):
    flag = ["--experimental_replay_ablation", mode]
    assert (
        parse_args(REQUIRED + flag + ["--scenes", "records.jsonl"]).experimental_replay_ablation
        == mode
    )
    for extra in (
        [],
        ["--scenes", "x", "--resume"],
        ["--scenes", "x", "--export_environment"],
        ["--scenes", "x", "--fixed_pose_params", "x"],
        ["--scenes", "x", "--fixed_pose_manifest", "x"],
    ):
        with pytest.raises(SystemExit) as exc:
            parse_args(REQUIRED + flag + extra)
        assert exc.value.code == 2


def test_replay_default_is_current_behavior():
    assert parse_args(REQUIRED).experimental_replay_ablation == "none"


@pytest.mark.parametrize("mode", ["appearance-rng", "placement", "both"])
def test_ablation_refuses_launcher_managed_dataset(tmp_path, mode):
    out = tmp_path / "dataset"
    (out / "manifests").mkdir(parents=True)
    path = out / "manifests/shard_00000000_00000001.json"
    path.write_text('{"kind": "render_shard"}')
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--config",
                "c.yaml",
                "--scenes",
                "records.jsonl",
                "--outdir",
                str(out),
                "--experimental_replay_ablation",
                mode,
            ]
        )
    assert path.read_text() == '{"kind": "render_shard"}'
