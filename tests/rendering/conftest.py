# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the rendering tests."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from rendering.environment_bundle import publish_bytes, publish_record, seal_bundle
from rendering.synth_parallel import blender_version_line

REPO_ROOT = Path(__file__).resolve().parents[2]
BLENDER_BASE_ARGS = ("--background", "--factory-startup", "--python-exit-code", "1")


def blender_bin() -> str | None:
    """Return the Blender binary (``DOPER_BLENDER``, then ``PATH``) if it is 4.2 or newer."""
    binary = os.environ.get("DOPER_BLENDER") or shutil.which("blender")
    if not binary:
        return None
    try:
        blender_version_line(binary)
    except SystemExit:
        return None
    return binary


@pytest.fixture(scope="session")
def blender() -> str:
    """Path of a usable Blender, or skip the test."""
    binary = blender_bin()
    if binary is None:
        pytest.skip("Blender >= 4.2 not found (set DOPER_BLENDER or put blender on PATH)")
    return binary


def run_blender(
    blender: str,
    *args: str | os.PathLike[str],
    script: str | os.PathLike[str] | None = None,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
    timeout: float = 90,
) -> subprocess.CompletedProcess:
    """Run background Blender with factory settings; the caller checks the return code.

    Args:
        blender: Blender executable.
        args: Further Blender arguments; with ``script``, these follow it (use ``--`` to pass
            script arguments).
        script: Python file to run (``--python``).
        cwd: Working directory (the checkout by default, so ``rendering`` is importable).
        env: Complete child environment, or ``None`` to inherit.
        timeout: Seconds before the run is aborted.
    """
    cmd = [blender, *BLENDER_BASE_ARGS]
    if script is not None:
        cmd += ["--python", str(script)]
    cmd += [str(arg) for arg in args]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env, timeout=timeout)


def run_in_blender(blender: str, code: str, *, cwd: Path) -> subprocess.CompletedProcess:
    """Run ``code`` inside background Blender with the checkout on ``sys.path``."""
    expr = f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r})\n{code}"
    cmd = [blender, "--background", "--factory-startup", "--python-expr", expr]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)


def snapshot(root: Path) -> dict[str, str]:
    """Map every file under ``root`` (relative path) to its SHA-256, to detect any change."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


PART_SETUP = """
import bpy, sys, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from rendering.usd_mesh import write_mesh_usda
root = Path(sys.argv[-1])
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
bpy.ops.mesh.primitive_cube_add(size=.1)
obj=bpy.context.object
obj.name='part'
names=[]
points=[]
for i,v in enumerate(obj.data.vertices):
    name=f'KP_part_{i:03d}'
    kp=bpy.data.objects.new(name,None)
    bpy.context.collection.objects.link(kp)
    kp.parent=obj
    kp.location=v.co
    names.append(name)
    points.append(list(v.co))
(root/'correct_keypoints.json').write_text(json.dumps({'object':'part','keypoint_names':names,'keypoints_3d':points}))
stale=[[v+.01 for v in point] for point in points]
(root/'stale_keypoints.json').write_text(json.dumps({'object':'part','keypoint_names':names,
    'keypoints_3d':stale}))
bpy.ops.wm.save_as_mainfile(filepath=str(root/'part.blend'))
write_mesh_usda(str(root/'part.usda'), object_name='part',
    points=[tuple(v.co) for v in obj.data.vertices],
    face_vertex_counts=[len(p.vertices) for p in obj.data.polygons],
    face_vertex_indices=[i for p in obj.data.polygons for i in p.vertices],
    keypoint_names=names,keypoints_3d=points)
"""


def make_part_inputs(blender: str, root: Path) -> None:
    """Write ``part.blend``, ``part.usda`` and correct/stale keypoint JSONs for a small cube."""
    setup = root / "setup.py"
    setup.write_text(PART_SETUP)
    result = run_blender(blender, "--", root, script=setup, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope="module")
def part_inputs(blender: str, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One :func:`make_part_inputs` directory per test module; tests must only read it."""
    root = tmp_path_factory.mktemp("part_inputs")
    make_part_inputs(blender, root)
    return root


def part_recipe(root: Path, *, stereo: bool = False) -> dict:
    """Small CPU-Cycles render recipe for the cube from :func:`make_part_inputs`."""
    recipe = {
        "output": {"resolution": 64, "render_masks": True},
        "object": {
            "mesh_name": "part",
            "keypoint_prefix": "KP_part_",
            "num_keypoints": 8,
            "usd_path": str(root / "part.usda"),
            "keypoints_json": str(root / "stale_keypoints.json"),
        },
        "scene": {
            "hdri_dir": "",
            "texture_dir": "",
            "distractor_dir": "",
            "num_distractors_min": 0,
            "num_distractors_max": 0,
            "ground_style": "procedural",
            "world_background_modes": "studio",
        },
        "render": {
            "engine": "CYCLES",
            "samples": 1,
            "use_raytracing": False,
            "cycles_samples": 4,
            "threads": 1,
        },
    }
    if stereo:
        recipe["camera"] = {"stereo": {"enabled": True}}
    return recipe


def run_launcher(*args: str, timeout: float = 600) -> subprocess.CompletedProcess:
    """Run ``rendering/synth_parallel.py`` with the current interpreter."""
    cmd = [sys.executable, str(REPO_ROOT / "rendering" / "synth_parallel.py"), *args]
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout)


@pytest.fixture
def native_bundle(tmp_path: Path) -> tuple[Path, dict, dict]:
    """Publish and seal a one-record native environment bundle under ``tmp_path / "bundle"``.

    Shared by the environment bundle and Isaac replay tests.

    Returns:
        ``(root, record, source)``.
    """
    root = tmp_path / "bundle"
    record = {
        "index": 0,
        "seed": 42,
        "background": {"mode": "hdri", "rotation_deg": 20},
    }
    source = {
        "identity_sha256": "a" * 64,
        "usd_sha256": "b" * 64,
        "blender_version": "4.2.1",
        "assets_sha256": "c" * 64,
    }
    layer = "layers/" + hashlib.sha256(b"#usda 1.0\n").hexdigest() + ".usda"
    publish_bytes(root / layer, b"#usda 1.0\n")
    publish_bytes(root / "textures/asset.hdr", b"pixels")
    publish_record(
        root,
        record,
        source=source,
        layer=layer,
        dependencies=["textures/asset.hdr"],
        light_count=2,
        ground=False,
    )
    seal_bundle(root, [record], source=source)
    return root, record, source
