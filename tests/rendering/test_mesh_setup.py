# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CAD and keypoint setup with explicit units (runs ``rendering/mesh_setup.py`` in Blender)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from conftest import REPO_ROOT, run_blender

from rendering.blender_inputs import inspect_blender_inputs
from rendering.usd_mesh import read_mesh_usda
from tests.stl_helpers import GRID, bracket_triangles_mm, write_binary_stl

pytestmark = pytest.mark.blender

SCRIPT = REPO_ROOT / "rendering" / "mesh_setup.py"
UNIT_FACTORS = {"m": 0.001, "mm": 1.0}


def write_bracket_stl(path: Path, units: str) -> np.ndarray:
    """Write the bracket as a float32 binary STL in ``units`` and return its float32 vertices."""
    vertices = (bracket_triangles_mm() * UNIT_FACTORS[units]).astype(np.float32)
    write_binary_stl(path, vertices, b"mesh_setup test bracket")
    return vertices.reshape(-1, 3)


def run_mesh_setup(
    blender: str, workdir: Path, units: str | None, *extra: str, num_keypoints: int = 16
) -> subprocess.CompletedProcess:
    """Run mesh_setup on the bracket in ``units`` (``None`` omits ``--source-units``)."""
    workdir.mkdir(parents=True, exist_ok=True)
    stl = workdir / "bracket.stl"
    write_bracket_stl(stl, units or "m")
    cmd = ["--", "--input", stl, "--output", workdir / "object.blend"]
    cmd += ["--num-keypoints", num_keypoints, "--keypoints-output", workdir / "keypoints.json"]
    if units:
        cmd += ["--source-units", units]
    return run_blender(blender, *cmd, *extra, script=SCRIPT, timeout=180)


def load_keypoints(workdir: Path) -> dict:
    """Read the keypoints JSON a run wrote."""
    return json.loads((workdir / "keypoints.json").read_text())


def test_source_units_required(blender, tmp_path):
    result = run_mesh_setup(blender, tmp_path, units=None)
    assert result.returncode != 0 and "--source-units" in result.stderr


def test_mm_and_m_inputs_give_identical_keypoints(blender, tmp_path):
    for units in ("m", "mm"):
        done = run_mesh_setup(blender, tmp_path / units, units)
        assert done.returncode == 0, done.stdout + done.stderr
    m = np.array(load_keypoints(tmp_path / "m")["keypoints_3d"])
    mm = np.array(load_keypoints(tmp_path / "mm")["keypoints_3d"])
    np.testing.assert_allclose(m, mm, rtol=0, atol=1e-7)  # float32 STL decode
    assert load_keypoints(tmp_path / "mm")["scale"] == 0.001
    extent_m = np.ptp(write_bracket_stl(tmp_path / "again.stl", "m"), axis=0).max()
    assert 0.06 < extent_m < 0.07
    assert 0.05 < np.ptp(m, axis=0).max() <= extent_m + 1e-6


def test_keypoints_input_authored_into_blend_and_usd(blender, tmp_path):
    first = run_mesh_setup(blender, tmp_path / "sampled", "mm", num_keypoints=8)
    assert first.returncode == 0, first.stdout + first.stderr
    sampled = load_keypoints(tmp_path / "sampled")
    points = [p for p in reversed(sampled["keypoints_3d"][:5])]
    names = [f"KP_object_{i:03d}" for i in range(len(points))]
    selection = {
        "keypoint_names": names,
        "keypoints_3d": points,
        "units": "m",
        "object_frame": "cad",
        "selector": {"method": "hand_built"},
    }
    (tmp_path / "selected.json").write_text(json.dumps(selection))
    usd = tmp_path / "authored" / "object.usda"
    done = run_mesh_setup(
        blender, tmp_path / "authored", "mm", "--keypoints-input", str(tmp_path / "selected.json"),
        "--usd-output", str(usd),
    )  # fmt: skip
    assert done.returncode == 0, done.stdout + done.stderr
    blend = inspect_blender_inputs(blender, str(tmp_path / "authored" / "object.blend"),
                                   "object", "KP_object_", len(points))  # fmt: skip
    from_usd = read_mesh_usda(str(usd))
    written = load_keypoints(tmp_path / "authored")
    for found in (blend, from_usd, written):
        assert found["keypoint_names"] == names
        np.testing.assert_allclose(found["keypoints_3d"], points, rtol=0, atol=1e-6)


def test_keypoints_input_frame_mismatch_rejected(blender, tmp_path):
    base = {"keypoint_names": ["KP_object_000"], "keypoints_3d": [[0.0, 0.0, 0.0]], "units": "m"}
    cases = [
        (dict(base, object_frame="aabb_centered"), [], "object_frame"),
        (dict(base, object_frame="cad"), ["--center-geometry"], "object_frame"),
        (dict(base, object_frame="cad", units="mm"), [], "units"),
        (
            dict(base, object_frame="cad", keypoint_names=["KP_other_000"]),
            [],
            "KP_object_",
        ),
    ]
    for index, (payload, flags, message) in enumerate(cases):
        path = tmp_path / f"in{index}.json"
        path.write_text(json.dumps(payload))
        done = run_mesh_setup(
            blender, tmp_path / f"run{index}", "mm", "--keypoints-input", str(path), *flags
        )
        assert done.returncode != 0 and message in done.stderr, done.stdout + done.stderr


def test_keypoint_json_records_units_and_hash(blender, tmp_path):
    done = run_mesh_setup(blender, tmp_path, "mm", "--center-geometry")
    assert done.returncode == 0, done.stdout + done.stderr
    payload = load_keypoints(tmp_path)
    assert {"object", "num_keypoints", "keypoint_names", "keypoints_3d"} <= set(payload)
    assert payload["units"] == "m" and payload["source_units"] == "mm"
    assert payload["scale"] == 0.001 and payload["object_frame"] == "aabb_centered"
    array = np.ascontiguousarray(np.array(payload["keypoints_3d"], dtype="<f8"))
    assert payload["coordinates_sha256"] == hashlib.sha256(array.tobytes()).hexdigest()
    centered = np.array(payload["keypoints_3d"])
    assert np.abs(centered).max() <= 0.035


def test_frame_transform_reports_meters(blender, tmp_path):
    done = run_mesh_setup(
        blender, tmp_path, "mm", "--center-geometry",
        "--frame-transform-output", str(tmp_path / "frame.json"),
    )  # fmt: skip
    assert done.returncode == 0, done.stdout + done.stderr
    frame = json.loads((tmp_path / "frame.json").read_text())
    assert frame["units"] == "m" and frame["source_units"] == "mm"
    forward, inverse = (
        np.array(frame["model_from_original"]),
        np.array(frame["original_from_model"]),
    )
    np.testing.assert_allclose(forward @ inverse, np.eye(4), atol=1e-7)


FACE_PROBE = """
import bpy
mesh = bpy.data.objects["object"].data
slots = [p.material_index for p in mesh.polygons]
names = [m.name for m in mesh.materials]
bsdf = mesh.materials[1].node_tree.nodes["Principled BSDF"]
print("PROBE", names, slots.count(0), [i for i, s in enumerate(slots) if s == 1],
      round(bsdf.inputs["Metallic"].default_value, 3))
"""


def test_face_materials_override(blender, tmp_path):
    materials = {"materials": [
        {"name": "body", "base_color": [0.1, 0.1, 0.1, 1], "metallic": 0, "roughness": 0.5,
         "faces": None},
        {"name": "pads", "base_color": [0.5, 0.3, 0.1, 1], "metallic": 0.85, "roughness": 0.3,
         "faces": [[2, 4], [10, 11]]},
    ]}  # fmt: skip
    spec = tmp_path / "materials.json"
    spec.write_text(json.dumps(materials))
    done = run_mesh_setup(blender, tmp_path / "ok", "mm", "--face-materials", str(spec))
    assert done.returncode == 0, done.stdout + done.stderr
    probe = subprocess.run(
        [blender, "--background", "--factory-startup", str(tmp_path / "ok" / "object.blend"),
         "--python-expr", FACE_PROBE], capture_output=True, text=True, timeout=120,
    )  # fmt: skip
    line = next(x for x in probe.stdout.splitlines() if x.startswith("PROBE"))
    faces = 3 * 2 * GRID * GRID * 2  # three axes, two sides, two triangles per cell
    assert line == f"PROBE ['body', 'pads'] {faces - 3} [2, 3, 10] 0.85"
    materials["materials"][1]["faces"] = [[2, 4], [3, 5]]
    spec.write_text(json.dumps(materials))
    done = run_mesh_setup(blender, tmp_path / "overlap", "mm", "--face-materials", str(spec))
    assert done.returncode != 0 and "overlap" in done.stdout + done.stderr
