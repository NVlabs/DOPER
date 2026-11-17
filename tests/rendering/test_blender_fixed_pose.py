# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded checks of the OpenCV fixed-pose preview contract in actual Blender."""

import json
import subprocess

import pytest
import yaml
from conftest import REPO_ROOT
from PIL import Image

pytestmark = pytest.mark.blender


def _run(blender, *args):
    result = subprocess.run(
        [blender, "--background", "--python-exit-code", "1", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_rodrigues_preserves_rotation_vector_angle(blender, tmp_path):
    script = tmp_path / "rotation.py"
    script.write_text("""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from rendering.blender.fixed_pose import rodrigues_to_matrix
# Identity, Z quarter-turn, and a 120-degree turn cycling X -> Y -> Z.
cases = [([0,0,0], [[1,0,0],[0,1,0],[0,0,1]]),
         ([0,0,math.pi/2], [[0,-1,0],[1,0,0],[0,0,1]]),
         ([2*math.pi/(3*math.sqrt(3))]*3, [[0,0,1],[1,0,0],[0,1,0]])]
for rvec, expected in cases:
    actual = rodrigues_to_matrix(rvec)
    assert max(abs(actual[i][j]-expected[i][j]) for i in range(3) for j in range(3)) < 1e-6, (rvec, actual)
print('RODRIGUES_MATCH')
""")
    assert "RODRIGUES_MATCH" in _run(blender, "--factory-startup", "--python", str(script)).stdout


def test_single_and_manifest_fixed_pose_cli_render_images(blender, tmp_path):
    setup = tmp_path / "setup.py"
    setup.write_text("""
import bpy, sys
from pathlib import Path
root = Path(sys.argv[-1])
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
bpy.ops.mesh.primitive_cube_add(size=.1)
obj = bpy.context.object
obj.name = 'part'
for i, vertex in enumerate(obj.data.vertices):
    kp = bpy.data.objects.new(f'KP_part_{i:03d}', None)
    bpy.context.collection.objects.link(kp)
    kp.parent = obj
    kp.location = vertex.co
bpy.ops.wm.save_as_mainfile(filepath=str(root/'part.blend'))
""")
    _run(blender, "--factory-startup", "--python", str(setup), "--", str(tmp_path))
    config = tmp_path / "recipe.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "output": {"resolution": 32, "render_masks": False},
                "object": {"mesh_name": "part", "keypoint_prefix": "KP_part_", "num_keypoints": 8},
                "scene": {
                    "hdri_dir": str(tmp_path / "empty-hdri"),
                    "texture_dir": str(tmp_path / "empty-textures"),
                    "distractor_dir": str(tmp_path / "empty-distractors"),
                    "num_distractors_min": 0,
                    "num_distractors_max": 0,
                    "world_background_modes": "studio",
                    "no_ground": 1.0,
                },
                "render": {
                    "samples": 1,
                    "use_raytracing": False,
                    "cycles_samples": 4,
                    "threads": 1,
                },
            }
        )
    )
    params = {
        "rvec": [0, 0, 1.5707963267948966],
        "tvec": [0, 0, 0.3],
        "fx": 40,
        "fy": 40,
        "cx": 16,
        "cy": 16,
        "width": 32,
        "height": 32,
        "samples": 1,
        "full_scene": False,
        "transparent": True,
    }
    params_path = tmp_path / "pose.json"
    params_path.write_text(json.dumps(params))
    single = tmp_path / "single.png"
    batch = tmp_path / "batch" / "preview.png"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"entries": [{"params": params, "out": str(batch)}]}))
    command = [
        str(tmp_path / "part.blend"),
        "--python",
        "rendering/synth_worker.py",
        "--",
        "--config",
        str(config),
        "--outdir",
        str(tmp_path / "unused-dataset"),
    ]
    _run(blender, *command, "--fixed_pose_params", str(params_path), "--render_out", str(single))
    result = _run(blender, *command, "--fixed_pose_manifest", str(manifest))
    assert "batch progress 1/1" in result.stdout
    for path in (single, batch):
        with Image.open(path) as image:
            assert image.size == (32, 32)
            # Transparent world plus a visible cube must contain foreground and background.
            assert image.mode == "RGBA"
            assert image.getchannel("A").getextrema() == (0, 255)
    assert not (tmp_path / "unused-dataset" / "annotations_worker_000.json").exists()


def test_procedural_only_config_without_texture_dir_renders(blender, tmp_path):
    """A procedural ground needs no texture directory; a texture ground without one is refused."""
    setup = tmp_path / "setup.py"
    setup.write_text(
        """
import bpy
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
bpy.ops.mesh.primitive_cube_add(size=.1)
bpy.context.object.name = 'part'
bpy.ops.wm.save_as_mainfile(filepath=sys.argv[-1] + '/part.blend')
""".replace("import bpy", "import bpy, sys", 1)
    )
    _run(blender, "--factory-startup", "--python", str(setup), "--", str(tmp_path))
    params = {
        "rvec": [0, 0, 0],
        "tvec": [0, 0, 0.3],
        "fx": 40,
        "fy": 40,
        "cx": 16,
        "cy": 16,
        "width": 32,
        "height": 32,
        "samples": 1,
        "full_scene": False,
    }
    params_path = tmp_path / "pose.json"
    params_path.write_text(json.dumps(params))

    def render(ground_style):
        config = tmp_path / f"{ground_style}.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "output": {"resolution": 32, "render_masks": False},
                    "object": {
                        "mesh_name": "part",
                        "keypoint_prefix": "KP_part_",
                        "num_keypoints": 0,
                    },
                    "scene": {
                        "num_distractors_min": 0,
                        "num_distractors_max": 0,
                        "world_background_modes": "studio",
                        "ground_style": ground_style,
                    },
                    "render": {
                        "samples": 1,
                        "use_raytracing": False,
                        "cycles_samples": 4,
                        "threads": 1,
                    },
                }
            )
        )
        out = tmp_path / f"{ground_style}.png"
        return subprocess.run(
            [
                blender,
                "--background",
                "--python-exit-code",
                "1",
                str(tmp_path / "part.blend"),
                "--python",
                "rendering/synth_worker.py",
                "--",
                "--config",
                str(config),
                "--outdir",
                str(tmp_path / "dataset"),
                "--fixed_pose_params",
                str(params_path),
                "--render_out",
                str(out),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=90,
        ), out

    done, out = render("procedural")
    assert done.returncode == 0, done.stdout + done.stderr
    assert out.is_file()
    refused, _ = render("texture")
    assert (
        refused.returncode != 0 and "render assets are missing" in refused.stdout + refused.stderr
    )
