# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regressions for worker input validation and repeated-scene transitions."""

import json
import os

import pytest
from conftest import run_blender as run_blender_process

pytestmark = pytest.mark.blender


def run_blender(blender, tmp_path, code, *, hash_seed="0"):
    """Run ``code`` in background Blender with the checkout importable; return stdout."""
    script = tmp_path / "worker_check.py"
    script.write_text(
        "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path.cwd()))\n" + code
    )
    result = run_blender_process(
        blender, script=script, env={**os.environ, "PYTHONHASHSEED": hash_seed}, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_duplicate_scenes_fail_before_worker_initialization(blender, tmp_path):
    scenes = tmp_path / "scenes.jsonl"
    scenes.write_text('{"index": 7}\n\n{"index": 7}\n')
    config = tmp_path / "render.yaml"
    config.write_text("output:\n  resolution: 32\n")
    run_blender(
        blender,
        tmp_path,
        f"""
from types import SimpleNamespace
from rendering.blender.context import BlenderContext
from rendering.blender import records, worker
from rendering.synth_worker import parse_args
path = {str(scenes)!r}
ctx = BlenderContext(args=SimpleNamespace(scenes=path), settings=SimpleNamespace())
ctx.scene_records = {{99: {{"index": 99}}}}
try:
    records.load_scene_records(ctx)
except ValueError as exc:
    assert 'duplicate scene record index 7' in str(exc) and ':3' in str(exc)
else:
    raise AssertionError('duplicate record accepted')
assert ctx.scene_records == {{99: {{"index": 99}}}}, 'partial records published'
def unexpected(ctx):
    raise AssertionError('worker mutated output before rejecting scenes')
worker.initialize = unexpected
try:
    worker.run_worker(parse_args(['--config', {str(config)!r}, '--outdir', {str(tmp_path)!r},
                                  '--scenes', path]))
except ValueError as exc:
    assert 'duplicate scene record index 7' in str(exc)
else:
    raise AssertionError('worker accepted duplicates')
Path(path).write_text('{{"index": 8}}\\n{{"index": 7}}\\n')
ctx.settings = SimpleNamespace(procedural_distractors=False, ground_image_files=[])
records.load_scene_records(ctx)
assert list(ctx.scene_records) == [8,7]
""",
    )


def test_preview_modes_clear_previous_decorations(blender, tmp_path):
    run_blender(
        blender,
        tmp_path,
        """
import bpy
from types import SimpleNamespace
from mathutils import Vector
from rendering.blender import fixed_pose
from rendering.blender.context import BlenderContext
fixed_pose.randomize_object_material = lambda ctx: None
fixed_pose.setup_world_background = lambda ctx: None
fixed_pose.setup_random_lights = lambda ctx, center: None
settings = SimpleNamespace(no_ground_prob=0.0, ground_image_files=[], ground_image_prob=0.0,
    ground_style='procedural', procedural_ground_modes=['color'], dark_background_prob=0.0,
    checker_background_prob=0.0, flat_background_saturation_range=(.1,.2),
    flat_background_value_range=(.5,.6), num_distractors_min=1, num_distractors_max=1,
    procedural_distractors=False)
ctx = BlenderContext(args=SimpleNamespace(), settings=settings, scene_scale=1.0,
                     distractor_files=['fake.usda'])
def create_distractor(ctx, *args):
    bpy.ops.mesh.primitive_cube_add()
    ctx.loaded_distractors.append(bpy.context.object)
fixed_pose.load_distractors = create_distractor
origin = Vector((0,0,0))
direction = Vector((0,0,1))
fixed_pose.setup_full_scene_at_pose(ctx, origin, .1, direction, full_scene=True)
assert ctx.ground_obj is not None and len(ctx.loaded_distractors) == 1
names = [ctx.ground_obj.name, ctx.loaded_distractors[0].name]
fixed_pose.setup_full_scene_at_pose(ctx, origin, .1, direction, full_scene=False)
assert ctx.ground_obj is None and ctx.loaded_distractors == []
assert all(name not in bpy.data.objects for name in names)
fixed_pose.setup_full_scene_at_pose(ctx, origin, .1, direction, full_scene=True)
plane = ctx.ground_obj.name
ctx.settings.no_ground_prob = 1.0
fixed_pose.setup_full_scene_at_pose(ctx, origin, .1, direction, full_scene=True)
assert ctx.ground_obj is None and plane not in bpy.data.objects
""",
    )


def test_imported_mesh_randomization_is_stable_across_hash_seeds(blender, tmp_path):
    code = """
import bpy, json, random
from types import SimpleNamespace
from mathutils import Vector
from rendering.blender.context import BlenderContext
from rendering.blender import distractors
real_bpy = bpy
def import_meshes(filepath):
    for name in ['Z_mesh','A_mesh','K_mesh','B_mesh']:
        real_bpy.ops.mesh.primitive_cube_add(size=.1)
        real_bpy.context.object.name = name
        material = real_bpy.data.materials.new('authored')
        real_bpy.context.object.data.materials.append(material)
    real_bpy.context.view_layer.update()
distractors.bpy = SimpleNamespace(data=real_bpy.data, context=real_bpy.context,
    ops=SimpleNamespace(wm=SimpleNamespace(usd_import=import_meshes)))
ctx = BlenderContext(args=SimpleNamespace(), settings=SimpleNamespace(), mesh_size=.1,
    rng=random.Random(42), distractor_files=['fake.usda'])
# Material draw makes assignment order observable independently of transforms.
def random_material(ctx, obj):
    obj.data.materials[0].diffuse_color = (ctx.rng.random(),ctx.rng.random(),ctx.rng.random(),1)
distractors.apply_random_material_if_needed = random_material
distractors.load_distractors(ctx, 1, Vector((0,0,0)))
result = {obj.name: [list(obj.location),list(obj.rotation_euler),list(obj.scale),
                   list(obj.data.materials[0].diffuse_color)] for obj in ctx.loaded_distractors}
print('RESULT=' + json.dumps(result,sort_keys=True))
"""
    results = []
    for seed in ("1", "20"):
        output = run_blender(blender, tmp_path, code, hash_seed=seed)
        results.append(
            json.loads(next(line[7:] for line in output.splitlines() if line.startswith("RESULT=")))
        )
    assert results[0] == results[1]


def test_legacy_quaternion_keeps_historical_draws_without_scipy(blender, tmp_path):
    run_blender(
        blender,
        tmp_path,
        """
import importlib.abc
import random
from types import SimpleNamespace
class NoScipy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'scipy':
            raise AssertionError('Blender compatibility path imported SciPy')
sys.meta_path.insert(0, NoScipy())
from rendering.blender.context import BlenderContext
from rendering.blender.legacy import legacy_random_quaternion
ctx = BlenderContext(args=SimpleNamespace(), settings=SimpleNamespace(), rng=random.Random(29))
actual = legacy_random_quaternion(ctx).to_matrix()
expected = [[-0.040781729742740414, 0.08738560329296735, -0.9953394430325337],
            [-0.9311138704163748, -0.36467703592834777, 0.0061334969432615405],
            [-0.3624414584969944, 0.9270244957948193, 0.09623810762339802]]
assert max(abs(actual[i][j]-expected[i][j]) for i in range(3) for j in range(3)) < 1e-6
assert ctx.rng.random() == 0.2885974354053432
""",
    )


@pytest.mark.parametrize("worker_id", [0, 40625, 43750, 46875])
def test_worker_seed_accepts_large_shard_ids_preserving_valid_streams(blender, tmp_path, worker_id):
    run_blender(
        blender,
        tmp_path,
        f"""
import random
import numpy as np
from types import SimpleNamespace
from rendering.blender.context import BlenderContext
from rendering.blender.setup import seed_and_prepare_output
worker_id = {worker_id}
seed = 2748 + worker_id * 100000
settings = SimpleNamespace(images_per_sample=1, config_path='test', render_masks=False,
    stereo_enabled=False, axis_bias=dict(x=0,y=0,z=0), _axis_total=0, axis_jiggle=0)
ctx = BlenderContext(args=SimpleNamespace(seed=2748, worker_id=worker_id,
    start_idx=worker_id, end_idx=worker_id+1, outdir={str(tmp_path)!r}), settings=settings)
state = np.random.get_state()
try:
    seed_and_prepare_output(ctx)
    expected = np.random.RandomState(seed if seed < 2**32 else seed % 2**32)
    assert np.array_equal(np.random.randint(0, 100000, size=16), expected.randint(0, 100000, size=16))
    assert ctx.rng.random() == random.Random(seed).random()
    first = ctx.rng.random()
    seed_and_prepare_output(ctx)
    ctx.rng.random()
    assert ctx.rng.random() == first
finally:
    np.random.set_state(state)
""",
    )


def test_replay_ablation_separates_rng_from_placement(blender, tmp_path):
    run_blender(
        blender,
        tmp_path,
        """
import hashlib, random
from types import SimpleNamespace
from rendering.blender import records
from rendering.blender.context import BlenderContext
record = {'seed':2748, 'index':7, 'object':{'bbox_min_z':.1}}
settings = SimpleNamespace(change_material_every=3, change_hdri_every=3,
                           change_ground_every=3, change_distractors_every=3)
for mode in ('none','appearance-rng','placement','both'):
    ctx = BlenderContext(args=SimpleNamespace(experimental_replay_ablation=mode),
                         settings=settings, rng=random.Random())
    for changed in (True, False):
        calls = []
        draws = []
        def draw(ctx, record, *args):
            draws.append(ctx.rng.random())
        for name in ('apply_record_material','apply_record_background','apply_record_ground',
                     'load_record_distractors','apply_record_lights'):
            setattr(records,name,draw)
        records.apply_record_object_pose = lambda *args: None
        records.apply_record_camera = lambda *args: None
        records.apply_record_transforms = lambda *args: calls.append('placement')
        records.realize_record(ctx,record,SimpleNamespace(changed=lambda *args:changed))
        assert calls == ([] if mode in ('placement','both') else ['placement'])
        if mode in ('appearance-rng','both'):
            seed = int.from_bytes(hashlib.sha256(b'2748:blender:7').digest()[:8],'big')
            expected = random.Random(seed)
            assert draws == [expected.random() for _ in range(5 if changed else 1)]
        else:
            fields = [('material',2),('background',2),('ground',2),('distractors',2)] if changed else []
            fields += [('lights',7)]
            expected = [random.Random(int.from_bytes(hashlib.sha256(
                f'2748:blender-v2:{field}:{index}'.encode()).digest()[:8],'big')).random()
                for field,index in fields]
            assert draws == expected
print('ABLATION_CONTRACTS_PASS')
""",
    )


def test_worker_imports_without_scipy(blender, tmp_path):
    run_blender(
        blender,
        tmp_path,
        """
import importlib
import pkgutil
import sys
import rendering.blender
import rendering.synth_worker
for module in pkgutil.iter_modules(rendering.blender.__path__):
    importlib.import_module(f'rendering.blender.{module.name}')
leaked = {'scipy', 'doper'} & {name.split('.')[0] for name in sys.modules}
assert not leaked, f'worker imported host-only packages: {leaked}'
""",
    )


def test_cycles_keys_applied_only_when_set(blender, tmp_path):
    run_blender(
        blender,
        tmp_path,
        """
import bpy
from types import SimpleNamespace
from rendering.blender.context import BlenderContext
from rendering.blender.setup import apply_optional_cycles_settings
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
default_samples = scene.cycles.samples
default_threads_mode = scene.render.threads_mode
def apply(cfg_render):
    settings = SimpleNamespace(cfg_render=cfg_render)
    apply_optional_cycles_settings(BlenderContext(args=SimpleNamespace(), settings=settings,
                                                  scene=scene))
apply({'samples': 7, 'eevee_samples': 9, 'engine': 'CYCLES'})
assert scene.cycles.samples == default_samples
assert scene.render.threads_mode == default_threads_mode
apply({'cycles_samples': 3})
assert scene.cycles.samples == 3 and scene.render.threads_mode == default_threads_mode
apply({'threads': 2})
assert scene.render.threads_mode == 'FIXED' and scene.render.threads == 2
""",
    )
