# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real rendered geometry must agree with occlusion/truncation labels."""

import subprocess

import pytest
from conftest import REPO_ROOT

pytestmark = pytest.mark.blender


def test_capsules_occlude_rgb_mask_and_keypoints_and_clear_between_samples(blender, tmp_path):
    script = tmp_path / "check.py"
    script.write_text("""
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path.cwd()))
import bpy
from mathutils import Vector
from rendering.blender.context import BlenderContext
from rendering.blender import records, annotations, views
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
bpy.ops.mesh.primitive_cube_add(size=1)
target=bpy.context.object
scene=bpy.context.scene
scene.render.engine='CYCLES'
scene.cycles.samples=1
scene.cycles.device='CPU'
scene.render.resolution_x=128
scene.render.resolution_y=128
scene.render.resolution_percentage=100
camdata=bpy.data.cameras.new('cam')
cam=bpy.data.objects.new('cam',camdata)
scene.collection.objects.link(cam)
cam.location=(0,0,3)
camdata.type='ORTHO'
camdata.ortho_scale=2
scene.camera=cam
ctx=BlenderContext(args=SimpleNamespace(worker_id=0),settings=SimpleNamespace(),scene=scene,mesh_obj=target,cam_obj=cam)
bpy.context.view_layer.update()
p=Vector((0,0,.5))
assert annotations.project_and_check(ctx,p,cam,scene,target)[2]==2
record={'robustness': {'mode':'occlusion','proxies':[{'start':[0,-.5,1.5],'end':[0,.5,1.5],'radius':.13,'color':[.7,.4,.2],'roughness':.65}]}}
records.apply_record_robustness(ctx,record)
assert annotations.project_and_check(ctx,p,cam,scene,target)[2]==1, 'capsule must occlude center keypoint'
assert annotations.project_and_check(ctx,Vector((.4,0,.5)),cam,scene,target)[2]==2
assert annotations.project_and_check(ctx,Vector((2,0,.5)),cam,scene,target)[2]==0
path=str(Path(sys.argv[-1])/'mask.png')
annotations.render_visible_object_mask(ctx,path)
mask=annotations._load_mask_foreground(ctx,path)
assert not mask[64,64], 'proxy pixels must be removed from visible object mask'
assert mask[64,90], 'uncovered object must remain in mask'
coverage=annotations.measure_robustness(ctx,path,record['robustness'])
assert .2 < coverage['proxy_occluded_fraction'] < .35, coverage
assert coverage['visible_pixels'] == int(mask.sum())
assert coverage['valid_detection_target']
assert all(not o.hide_render for o in ctx.robustness_objects), 'coverage pass must restore proxies'
root=Path(sys.argv[-1])
(root/'images').mkdir()
(root/'masks').mkdir()
ctx.args.outdir=str(root)
ctx.settings=SimpleNamespace(stereo_enabled=False,render_masks=True,mask_dir_name='masks',images_per_sample=1,
    num_keypoints=3,resolution=128,keypoint_mask_check=False)
ctx.scene_records={0:record}
for point in [(0,0,.5),(.4,0,.5),(2,0,.5)]:
    kp=bpy.data.objects.new('KP',None)
    scene.collection.objects.link(kp)
    kp.location=point
    ctx.kp_objects.append(kp)
bpy.context.view_layer.update()
views.render_views(ctx,0,0)
ann=ctx.annotations[0]
assert ann['keypoints'][2::3]==[1,2,0]
assert ann['robustness']['visible_keypoints']==1
assert ann['num_keypoints']==2
assert ann['bbox']==[28.8,28.8,70.4,70.4], ann
assert ann['robustness']['visible_bbox']==[32,32,64,64]
assert ann['area']==ann['robustness']['visible_pixels']
# A correctly occluded keypoint must not cause the mask-alignment guard to warn.
ctx.settings.keypoint_mask_tol_px=1
ctx.settings.keypoint_mask_min_frac=.9
ctx.settings.keypoint_mask_max_warn=3
ctx._kp_mask_stats=dict(checked=0,frac_sum=0.,worst_frac=1.,worst_file='',bad=0,warned=0)
annotations.check_keypoints_against_mask(ctx,path,ann['keypoints'],'fixture.png')
assert ctx._kp_mask_stats['frac_sum']==1., ctx._kp_mask_stats

# A fully covered target keeps its label for review but is not a detection target.
cover={'robustness': {'mode':'occlusion','proxies':[{'start':[-1.5,0,1.5],'end':[1.5,0,1.5],'radius':1.2,'color':[.7,.4,.2],'roughness':.65}]}}
records.apply_record_robustness(ctx,cover)
ctx.scene_records={1:cover}
views.render_views(ctx,1,1)
hidden=ctx.annotations[-1]
assert hidden['robustness']['valid_detection_target'] is False, hidden['robustness']
assert hidden['bbox']==[0,0,0,0] and hidden['area']==0
records.apply_record_robustness(ctx,record)

# Shift camera so the target is truncated by the left frame boundary.
cam.location.x=.7
bpy.context.view_layer.update()
assert annotations.project_and_check(ctx,Vector((-.5,0,.5)),cam,scene,target)[2]==0
annotations.render_visible_object_mask(ctx,path)
mask=annotations._load_mask_foreground(ctx,path)
assert mask[:,0].any(), 'truncated visible mask must touch boundary'
records.apply_record_robustness(ctx,{})
assert not ctx.robustness_objects
cam.location.x=0
bpy.context.view_layer.update()
assert annotations.project_and_check(ctx,p,cam,scene,target)[2]==2
annotations.render_visible_object_mask(ctx,path)
assert annotations._load_mask_foreground(ctx,path)[64,64]
print('RENDERED_ROBUSTNESS_OK')
""")
    result = subprocess.run(
        [
            blender,
            "--background",
            "--factory-startup",
            "--threads",
            "2",
            "--python-exit-code",
            "1",
            "--python",
            str(script),
            "--",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RENDERED_ROBUSTNESS_OK" in result.stdout
