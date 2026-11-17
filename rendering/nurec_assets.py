#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Blender script that authors the NuRec synthetic-baseline render assets of an object.

From the object's mesh (an OBJ in meters, CAD frame) and its 64-keypoint layout
(``keypoints_3d.json``), it runs ``mesh_setup.py`` with ``--keypoints-input`` and
``--face-materials`` and writes, into ``--out``:

* ``object.blend``: the mesh ``object`` with the keypoint empties ``KP_object_NNN`` parented to it
  and one dark body material;
* ``model.usda`` and ``keypoints_3d.json``, the inputs ``synth_parallel.py`` reads;
* ``assets_receipt.json``: the SHA-256 of every input and output and the identity checks.

The mesh is bound by its file SHA-256, the recipe's ``object.mesh_sha256`` passed as
``--mesh-sha256``. The scene holds only the mesh and its keypoint empties: the camera and sun
``mesh_setup.py`` adds for previews are removed, since the render workers open this ``.blend`` as
their scene and would keep the sun in every image.

The identity checks reopen the saved ``.blend`` and refuse the assets unless its geometry is the
source mesh's (Blender stores float32 vertices, so they must agree within ``VERTEX_ATOL_M``), the
scene has no camera, light or other object, every keypoint empty is within ``KEYPOINT_TOL_MM`` of
its pinned position and of a mesh vertex, and the ``.usda`` holds the same vertices, faces and
keypoints as the ``.blend``. The face shading is the mesh file's own normals
(``--smooth-angle-deg -1``).

Run from a checkout:
    blender --background --factory-startup --python rendering/nurec_assets.py -- \
        --mesh object.obj --mesh-sha256 <recipe object.mesh_sha256> \
        --keypoints keypoints_3d.json \
        --points-sha256 <recipe object.keypoints_points_sha256> --out assets
"""

# pyright: reportMissingImports=false, reportMissingModuleSource=false

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

script_dir = os.path.dirname(os.path.realpath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

OBJECT = "object"
KEYPOINT_PREFIX = f"KP_{OBJECT}_"
NUM_KEYPOINTS = 64
BODY_MATERIAL = {"name": f"{OBJECT}_body", "base_color": [0.02, 0.02, 0.02, 1.0],
                 "metallic": 0.0, "roughness": 0.45, "faces": None}  # fmt: skip
MATERIALS = (BODY_MATERIAL,)
# Blender stores vertices and empty locations as float32.
VERTEX_ATOL_M = 1e-8
KEYPOINT_TOL_MM = 0.001
MM_PER_M = 1000.0
BLEND_NAME = "object.blend"
USD_NAME = "model.usda"
KEYPOINTS_NAME = "keypoints_3d.json"
RECEIPT_NAME = "assets_receipt.json"
OUTPUT_NAMES = (BLEND_NAME, USD_NAME, KEYPOINTS_NAME)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the options after Blender's ``--`` separator.

    Args:
        argv: Argument strings, or None to read ``sys.argv`` after ``--``.

    Returns:
        Namespace with ``mesh``, ``keypoints`` and ``out``.
    """
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(prog="nurec_assets.py", description=__doc__)
    parser.add_argument("--mesh", required=True, help="the object mesh, an OBJ in meters")
    parser.add_argument(
        "--mesh-sha256", required=True, help="the mesh file's pin (recipe object.mesh_sha256)"
    )
    parser.add_argument("--keypoints", required=True, help="the bundle's keypoints_3d.json")
    parser.add_argument(
        "--points-sha256",
        required=True,
        help="the layout's pin: SHA-256 of its 64x3 little-endian float64 points (the recipe's "
        "object.keypoints_points_sha256)",
    )
    parser.add_argument("--out", required=True, help="new or empty output directory")
    return parser.parse_args(argv)


def fail(message: str) -> SystemExit:
    """Build the exception that stops the script with ``message`` on stderr.

    Args:
        message: What failed.

    Returns:
        A ``SystemExit`` carrying ``nurec_assets.py: <message>``.
    """
    return SystemExit(f"nurec_assets.py: {message}")


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_obj(path: str | os.PathLike[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the vertex and face arrays of an OBJ file.

    Args:
        path: OBJ file.

    Returns:
        ``(vertices, face_sizes, face_indices)``: float64 (N, 3), int64 per-face corner counts
        and the int64 zero-based corner vertex indices, in file order.
    """
    vertices, sizes, indices = [], [], []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                vertices.append([float(c) for c in parts[1:4]])
            elif parts[0] == "f":
                corners = [int(token.split("/")[0]) - 1 for token in parts[1:]]
                sizes.append(len(corners))
                indices.extend(corners)
    return (np.asarray(vertices, dtype=np.float64), np.asarray(sizes, dtype=np.int64),
            np.asarray(indices, dtype=np.int64))  # fmt: skip


def points_sha256(points: Sequence[Sequence[float]]) -> str:
    """SHA-256 of a keypoint array as C-contiguous little-endian float64."""
    return hashlib.sha256(np.ascontiguousarray(points, dtype="<f8").tobytes()).hexdigest()


def read_layout(path: str | os.PathLike[str], pin: str) -> tuple[list[str], list[list[float]]]:
    """Read the bundle's keypoint layout and check it is the pinned one.

    Args:
        path: ``keypoints_3d.json`` with ``keypoint_names`` and ``keypoints_3d`` (meters).
        pin: Expected :func:`points_sha256` of the points.

    Returns:
        Keypoint names and positions.

    Raises:
        SystemExit: Another object, names or points.
    """
    payload = json.loads(Path(path).read_text())
    names = [str(name) for name in payload["keypoint_names"]]
    points = [[float(c) for c in point] for point in payload["keypoints_3d"]]
    expected = [f"{KEYPOINT_PREFIX}{i:03d}" for i in range(NUM_KEYPOINTS)]
    if payload.get("object", OBJECT) != OBJECT or names != expected:
        raise fail(f"{path} is not the {OBJECT} layout ({NUM_KEYPOINTS} {KEYPOINT_PREFIX}NNN "
                   "keypoints)")  # fmt: skip
    if points_sha256(points) != pin:
        raise fail(f"{path} points SHA-256 {points_sha256(points)} is not the pin {pin}")
    return names, points


def check_mesh(path: str | os.PathLike[str], pin: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the mesh and check its file pin (primary) and its geometry (secondary).

    Args:
        path: OBJ file.
        pin: Expected file SHA-256 (the recipe's ``object.mesh_sha256``).

    Returns:
        The arrays of :func:`read_obj`.

    Raises:
        SystemExit: Another file or geometry.
    """
    digest = sha256_file(path)
    if digest != pin:
        raise fail(f"{path} SHA-256 {digest} is not the mesh pin {pin}")
    return read_obj(path)


def write_inputs(directory: Path, names: Sequence[str], points: Sequence[Sequence[float]]) -> \
        tuple[Path, Path]:  # fmt: skip
    """Write the ``mesh_setup.py`` keypoints input and face materials into ``directory``."""
    keypoints = directory / "keypoints_input.json"
    keypoints.write_text(json.dumps({
        "object": OBJECT, "keypoint_names": list(names), "keypoints_3d": [list(p) for p in points],
        "units": "m", "object_frame": "cad",
    }))  # fmt: skip
    materials = directory / "face_materials.json"
    materials.write_text(json.dumps({"materials": list(MATERIALS)}))
    return keypoints, materials


def run_mesh_setup(mesh: str, out: Path, keypoints: Path, materials: Path) -> None:
    """Author the ``.blend``, ``.usda`` and keypoints with ``mesh_setup.py``."""
    import bpy
    import mesh_setup

    # An empty factory scene: no default cube, materials or grease pencil data to carry over.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mesh_setup.main([
        "--input", mesh, "--output", str(out / BLEND_NAME), "--source-units", "m",
        "--mesh-name", OBJECT, "--keypoints-input", str(keypoints),
        "--keypoints-output", str(out / KEYPOINTS_NAME), "--usd-output", str(out / USD_NAME),
        "--smooth-angle-deg", "-1", "--face-materials", str(materials),
    ])  # fmt: skip
    # The OBJ importer's Y-up conversion leaves a rotation on the object; the renders set the pose,
    # so the object keeps the identity transform and its mesh-local frame.
    bpy.data.objects[OBJECT].rotation_euler = (0.0, 0.0, 0.0)
    remove_preview_objects()
    bpy.ops.wm.save_as_mainfile(filepath=str(out / BLEND_NAME))


def remove_preview_objects() -> None:
    """Delete every camera and light object and their data (mesh_setup's preview rig)."""
    import bpy

    for obj in [o for o in bpy.data.objects if o.type in ("CAMERA", "LIGHT")]:
        bpy.data.objects.remove(obj, do_unlink=True)
    for collection in (bpy.data.cameras, bpy.data.lights):
        for block in list(collection):
            collection.remove(block)


def blend_geometry(blend: Path) -> dict[str, Any]:
    """Reopen ``blend`` and read the object's vertices, faces, empties and materials."""
    import bpy

    bpy.ops.wm.open_mainfile(filepath=str(blend))
    mesh_obj = bpy.data.objects[OBJECT]
    mesh = mesh_obj.data
    vertices = np.zeros(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", vertices)
    sizes = np.zeros(len(mesh.polygons), dtype=np.int32)
    mesh.polygons.foreach_get("loop_total", sizes)
    indices = np.zeros(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", indices)
    slots = np.zeros(len(mesh.polygons), dtype=np.int32)
    mesh.polygons.foreach_get("material_index", slots)
    empties = {obj.name: (obj.parent, list(obj.location)) for obj in bpy.data.objects
               if obj.name.startswith(KEYPOINT_PREFIX)}  # fmt: skip
    materials = []
    for material in mesh.materials:
        bsdf = material.node_tree.nodes["Principled BSDF"]
        materials.append({
            "name": material.name,
            "base_color": [float(c) for c in bsdf.inputs["Base Color"].default_value],
            "metallic": float(bsdf.inputs["Metallic"].default_value),
            "roughness": float(bsdf.inputs["Roughness"].default_value),
        })  # fmt: skip
    if any(parent != mesh_obj for parent, _ in empties.values()):
        raise fail("every keypoint empty must be parented to the mesh")
    identity_transform = np.allclose(np.asarray(mesh_obj.matrix_world), np.eye(4), atol=1e-12)
    return {"vertices": vertices.reshape(-1, 3).astype(np.float64), "sizes": sizes,
            "indices": indices, "slots": slots, "materials": materials,
            "empties": {name: location for name, (_, location) in empties.items()},
            "names": sorted(obj.name for obj in bpy.data.objects),
            "object_types": sorted({obj.type for obj in bpy.data.objects}),
            "cameras": len(bpy.data.cameras), "lights": len(bpy.data.lights),
            "material_names": sorted(m.name for m in bpy.data.materials),
            "identity_transform": bool(identity_transform),
            "blender_version": bpy.app.version_string}  # fmt: skip


def face_material_counts(slots: np.ndarray, materials: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Faces per material."""
    names = [m["name"] for m in materials]
    return {"material_faces": {name: int((slots == i).sum()) for i, name in enumerate(names)}}


def measure_identity(
    blend: dict[str, Any],
    source: tuple[np.ndarray, np.ndarray, np.ndarray],
    names: Sequence[str],
    points: Sequence[Sequence[float]],
    usd: dict[str, Any],
) -> dict[str, Any]:
    """Compare the reopened ``.blend`` and the ``.usda`` with the source mesh and the layout."""
    vertices, sizes, indices = source
    pinned = np.asarray(points, dtype=np.float64)
    shape_ok = blend["vertices"].shape == vertices.shape
    vertex_error = float(np.abs(blend["vertices"] - vertices).max()) if shape_ok else float("inf")
    faces_equal = bool(np.array_equal(blend["sizes"], sizes)
                       and np.array_equal(blend["indices"], indices))  # fmt: skip
    if sorted(blend["empties"]) != sorted(names):
        raise fail("the .blend keypoint empties are not the layout's names")
    authored = np.asarray([blend["empties"][name] for name in names], dtype=np.float64)
    keypoint_mm = float(np.linalg.norm(authored - pinned, axis=1).max() * MM_PER_M)
    nearest = np.sqrt(((pinned[:, None, :] - vertices[None, :, :]) ** 2).sum(-1)).min(axis=1)
    usd_points = np.asarray(usd["points"], dtype=np.float64)
    usd_shape_ok = usd_points.shape == blend["vertices"].shape
    usd_error = float(np.abs(usd_points - blend["vertices"]).max()) if usd_shape_ok else np.inf
    usd_faces = bool(np.array_equal(usd["face_vertex_counts"], blend["sizes"])
                     and np.array_equal(usd["face_vertex_indices"], blend["indices"]))  # fmt: skip
    usd_keypoint_mm = float(
        np.linalg.norm(np.asarray(usd["keypoints_3d"]) - pinned, axis=1).max() * MM_PER_M
    )
    return {
        "vertices": int(len(blend["vertices"])), "faces": int(len(blend["sizes"])),
        "identity_transform": blend["identity_transform"],
        "object_types": blend["object_types"],
        "max_vertex_error_m": vertex_error, "faces_equal": faces_equal,
        "max_keypoint_error_mm": keypoint_mm,
        "max_nearest_vertex_mm": float(nearest.max() * MM_PER_M),
        "usd_max_point_difference_m": usd_error, "usd_faces_equal": usd_faces,
        "usd_keypoint_names_equal": list(usd["keypoint_names"]) == list(names),
        "usd_max_keypoint_error_mm": usd_keypoint_mm,
        **face_material_counts(blend["slots"], blend["materials"]),
    }  # fmt: skip


def identity_problems(identity: dict[str, Any], materials: Sequence[dict[str, Any]]) -> list[str]:
    """Return why the assets are not the source object (empty when they are)."""
    problems = []
    if not identity["identity_transform"]:
        problems.append("the object transform is not the identity")
    if identity["max_vertex_error_m"] > VERTEX_ATOL_M or not identity["faces_equal"]:
        problems.append("the .blend geometry differs from the source mesh")
    for key in ("max_keypoint_error_mm", "max_nearest_vertex_mm", "usd_max_keypoint_error_mm"):
        if not identity[key] <= KEYPOINT_TOL_MM:
            problems.append(f"{key} {identity[key]} > {KEYPOINT_TOL_MM}")
    if (identity["usd_max_point_difference_m"] > VERTEX_ATOL_M or not identity["usd_faces_equal"]
            or not identity["usd_keypoint_names_equal"]):  # fmt: skip
        problems.append("the .usda and .blend geometry differ")
    for spec, actual in zip(MATERIALS, materials, strict=True):
        expected = [*spec["base_color"], spec["metallic"], spec["roughness"]]
        values = [*actual["base_color"], actual["metallic"], actual["roughness"]]
        if actual["name"] != spec["name"] or not np.allclose(values, expected, atol=1e-6):
            problems.append(f"material {actual['name']!r} is not {spec['name']!r}")
    return problems


def check_scrubbed(blend: dict[str, Any]) -> None:
    """Refuse objects other than the mesh and its empties, and other material names."""
    expected = sorted([OBJECT, *(f"{KEYPOINT_PREFIX}{i:03d}" for i in range(NUM_KEYPOINTS))])
    if blend["object_types"] != ["EMPTY", "MESH"] or blend["cameras"] or blend["lights"]:
        raise fail(
            f"the .blend holds {blend['object_types']} objects, {blend['cameras']} "
            f"cameras and {blend['lights']} lights; only the mesh and empties belong"
        )
    if blend["names"] != expected:
        raise fail(
            f"unexpected objects in the .blend: {sorted(set(blend['names']) - set(expected))}"
        )
    if blend["material_names"] != sorted(m["name"] for m in MATERIALS):
        raise fail(f"unexpected materials in the .blend: {blend['material_names']}")


def build_receipt(args: argparse.Namespace, out: Path, identity: dict[str, Any],
                  blender_version: str) -> dict[str, Any]:  # fmt: skip
    """The receipt: inputs, material, identity measurements and output SHA-256s."""
    return {
        "object": OBJECT,
        "blender_version": blender_version,
        "inputs": {
            "mesh": {"name": Path(args.mesh).name, "sha256": args.mesh_sha256},
            "keypoints": {"sha256": sha256_file(args.keypoints),
                          "points_sha256": args.points_sha256},
        },
        "materials": list(MATERIALS),
        "tolerances": {"vertex_m": VERTEX_ATOL_M, "keypoint_mm": KEYPOINT_TOL_MM},
        "identity": identity,
        "files": {name: sha256_file(out / name) for name in OUTPUT_NAMES},
    }  # fmt: skip


def main(argv: Sequence[str] | None = None) -> None:
    """Author, reopen, verify and receipt the NuRec render assets.

    Raises:
        SystemExit: An input is not the pinned one, the output exists, or a check fails.
    """
    from usd_mesh import read_mesh_usda

    args = parse_args(argv)
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise fail(f"{out} is not empty")
    source = check_mesh(args.mesh, args.mesh_sha256)
    names, points = read_layout(args.keypoints, args.points_sha256)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nurec-assets-") as scratch:
        keypoints, materials = write_inputs(Path(scratch), names, points)
        run_mesh_setup(os.path.abspath(args.mesh), out, keypoints, materials)
    blend = blend_geometry(out / BLEND_NAME)
    check_scrubbed(blend)
    identity = measure_identity(blend, source, names, points, read_mesh_usda(str(out / USD_NAME)))
    problems = identity_problems(identity, blend["materials"])
    if problems:
        raise fail("identity check failed: " + "; ".join(problems))
    receipt = build_receipt(args, out, identity, blend["blender_version"])
    (out / RECEIPT_NAME).write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"NUREC_ASSETS {json.dumps(identity)}")


if __name__ == "__main__":
    main()
