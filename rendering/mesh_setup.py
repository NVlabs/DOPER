#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Blender script that prepares a CAD mesh and its keypoints for data generation.

It imports the mesh, converts it to meters from a declared source unit, optionally centers it on
its bounding box, assigns a material, authors keypoint empties (farthest point sampling, or the
names and coordinates of a ``--keypoints-input`` JSON) and saves the ``.blend`` together with an
optional keypoints JSON, ``.usda`` export, centered STL and frame transform.
``--face-materials`` replaces the material with Principled materials assigned by face index
range.

Run with:
    blender --background --python rendering/mesh_setup.py -- --input object.stl \
        --output object.blend --source-units mm --num-keypoints 64

``--source-units`` is required: a mesh file carries no unit, and guessing one silently produces
objects that are 1000 times too large or small. Everything written is in meters.
"""

# pyright: reportMissingImports=false, reportMissingModuleSource=false

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import bpy
import numpy as np
from mathutils import Vector

# Sibling modules are found next to this script, whatever the working directory or install path.
script_dir = os.path.dirname(os.path.realpath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
from usd_blender import mesh_uvs_and_texture  # noqa: E402
from usd_mesh import keypoints_payload, write_mesh_usda  # noqa: E402

if TYPE_CHECKING:
    from rendering.blender.context import BpyObject

UNIT_SCALES: dict[str, float] = {"m": 1.0, "cm": 0.01, "mm": 0.001, "in": 0.0254}
FRAME_CAD = "cad"
FRAME_AABB_CENTERED = "aabb_centered"
DEFAULT_NUM_KEYPOINTS = 64
DEFAULT_MESH_NAME = "object"
KEYPOINT_DECIMALS = 6
KEYPOINT_DISPLAY_FRACTION = 0.05
CAMERA_DISTANCE_FACTOR = 3.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the options after Blender's ``--`` separator.

    Args:
        argv: Argument strings, or None to read ``sys.argv`` after ``--``.

    Returns:
        Namespace of input, output, unit and keypoint settings.
    """
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(prog="mesh_setup.py", description=__doc__)
    parser.add_argument("--input", required=True, help="Input mesh (.stl, .obj, .glb, .gltf, .ply)")
    parser.add_argument("--output", required=True, help="Output .blend path")
    parser.add_argument(
        "--source-units",
        required=True,
        choices=sorted(UNIT_SCALES),
        help="Length unit of the input mesh coordinates; the output is in meters",
    )
    parser.add_argument(
        "--num-keypoints",
        type=int,
        default=DEFAULT_NUM_KEYPOINTS,
        help="Farthest-point-sampled keypoints (ignored with --keypoints-input)",
    )
    parser.add_argument("--mesh-name", default=DEFAULT_MESH_NAME, help="Mesh object name")
    parser.add_argument(
        "--keypoints-input",
        default=None,
        help="Keypoints JSON in meters and in this run's object frame, used instead of sampling",
    )
    parser.add_argument(
        "--keypoints-output",
        default=None,
        help="Keypoints JSON path (default: <mesh-name>_keypoints_3d.json next to the .blend)",
    )
    parser.add_argument(
        "--smooth-angle-deg",
        type=float,
        default=30.0,
        help="Smooth shading only across edges below this angle in degrees; the STL importer "
        "marks every face smooth, which renders sharp edges and holes as dark blotches. "
        "Negative disables.",
    )
    parser.add_argument(
        "--center-geometry",
        action="store_true",
        help="Translate the vertices so the bounding-box center is the model origin",
    )
    parser.add_argument(
        "--centered-mesh-output", default=None, help="Binary STL in the centered model frame"
    )
    parser.add_argument(
        "--frame-transform-output",
        default=None,
        help="JSON mapping between the original and the centered model frame",
    )
    parser.add_argument(
        "--usd-output",
        default=None,
        help=".usda export of the mesh and keypoints in the keypoint frame",
    )
    parser.add_argument(
        "--keep-material",
        action="store_true",
        default=True,
        help="Keep the imported material or texture (default); see --no-keep-material",
    )
    parser.add_argument(
        "--no-keep-material",
        dest="keep_material",
        action="store_false",
        help="Replace the mesh material with procedural brushed aluminum",
    )
    parser.add_argument(
        "--texture",
        default=None,
        help="Texture image (auto-detected from a PLY TextureFile comment if not set)",
    )
    parser.add_argument(
        "--face-materials",
        default=None,
        help="Material override JSON: Principled materials and the face index ranges each one "
        "covers (replaces the imported material; see read_face_materials)",
    )
    return parser.parse_args(argv)


def fail(message: str) -> SystemExit:
    """Build the exception that stops the script with a non-zero exit and ``message`` on stderr.

    Args:
        message: Explanation of what the caller must change.

    Returns:
        A ``SystemExit`` carrying ``mesh_setup.py: <message>``.
    """
    return SystemExit(f"mesh_setup.py: {message}")


def coordinates_sha256(keypoints_3d: Sequence[Sequence[float]]) -> str:
    """Hash keypoint coordinates independently of JSON formatting.

    Args:
        keypoints_3d: Keypoint positions, one ``[x, y, z]`` row per keypoint.

    Returns:
        Hex SHA-256 of the C-contiguous little-endian float64 array.
    """
    array = np.ascontiguousarray(np.asarray(keypoints_3d, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def read_keypoints_input(
    path: str, object_frame: str, mesh_name: str
) -> tuple[list[str], list[list[float]]]:
    """Read and validate a keypoints JSON for authoring.

    Fields other than ``keypoint_names``, ``keypoints_3d``, ``units``, ``object_frame`` and
    ``coordinates_sha256`` are ignored.

    Args:
        path: Keypoints JSON path.
        object_frame: Frame of this run's mesh, ``cad`` or ``aabb_centered``.
        mesh_name: Mesh object name; every keypoint name must start with ``KP_<mesh_name>_``.

    Returns:
        Keypoint names and positions in meters.

    Raises:
        SystemExit: The file is unreadable, not in meters, in another object frame,
            named with another prefix or inconsistent.
    """
    try:
        with open(path, encoding="utf-8") as stream:
            payload = json.load(stream)
        names = [str(name) for name in payload["keypoint_names"]]
        points = [[float(c) for c in point] for point in payload["keypoints_3d"]]
    except (OSError, ValueError, KeyError, TypeError) as err:
        raise fail(f"cannot read keypoints from {path}: {err!r}") from err
    if payload.get("units") != "m":
        raise fail(f"--keypoints-input units must be 'm', got {payload.get('units')!r}")
    if payload.get("object_frame") != object_frame:
        raise fail(
            f"--keypoints-input object_frame is {payload.get('object_frame')!r} but this run "
            f"produces {object_frame!r}; pass --center-geometry consistently with the selection"
        )
    if not names or len(names) != len(points) or any(len(p) != 3 for p in points):
        raise fail("--keypoints-input needs matching keypoint_names and Nx3 keypoints_3d")
    if len(set(names)) != len(names):
        raise fail("--keypoints-input keypoint_names must be unique")
    prefix = f"KP_{mesh_name}_"
    if any(not name.startswith(prefix) for name in names):
        raise fail(
            f"--keypoints-input names must start with {prefix!r} (KP_<mesh-name>_); rerun the "
            f"selector with --keypoint-prefix {prefix}"
        )
    recorded = payload.get("coordinates_sha256")
    if recorded is not None and recorded != coordinates_sha256(points):
        raise fail("--keypoints-input coordinates_sha256 does not match its keypoints_3d")
    return names, points


def clear_scene() -> None:
    """Delete the default objects and collections."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for collection in bpy.data.collections:
        bpy.data.collections.remove(collection)


def import_mesh(input_path: str, mesh_name: str) -> BpyObject:
    """Import ``input_path`` as one mesh object with its own mesh data.

    Args:
        input_path: Mesh file; the extension selects the importer.
        mesh_name: Name given to the object and its mesh data.

    Returns:
        The imported (and, if the file had several parts, joined) mesh object.

    Raises:
        SystemExit: The format is unsupported or the file holds no mesh.
    """
    lower = input_path.lower()
    if lower.endswith(".stl"):
        bpy.ops.wm.stl_import(filepath=input_path)
    elif lower.endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=input_path)
    elif lower.endswith((".glb", ".gltf")):
        bpy.ops.import_scene.gltf(filepath=input_path)
    elif lower.endswith(".ply"):
        bpy.ops.wm.ply_import(filepath=input_path)
    else:
        raise fail(f"unsupported mesh format: {input_path}")
    meshes = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not meshes:
        raise fail(f"no mesh objects were imported from {input_path}")
    if len(meshes) > 1:
        print(f"Imported {len(meshes)} mesh objects; joining into one mesh")
        bpy.ops.object.select_all(action="DESELECT")
        for obj in meshes:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.join()
        mesh_obj = bpy.context.object
    else:
        mesh_obj = meshes[0]
    mesh_obj.name = mesh_name
    mesh_obj.data.name = mesh_name
    if mesh_obj.data.users > 1:
        mesh_obj.data = mesh_obj.data.copy()
    return mesh_obj


def scale_to_meters(mesh_obj: BpyObject, scale: float) -> None:
    """Bake the unit scale into the vertices.

    Args:
        mesh_obj: Mesh object with a single-user mesh.
        scale: Meters per source unit.
    """
    mesh_obj.scale = (scale, scale, scale)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)


def aabb_center(mesh_obj: BpyObject) -> BpyObject:
    """Return the center of the vertex bounding box.

    Args:
        mesh_obj: Mesh object.

    Returns:
        Center as a ``mathutils.Vector`` in mesh-local coordinates.
    """
    coords = [vertex.co for vertex in mesh_obj.data.vertices]
    low = Vector(tuple(min(c[i] for c in coords) for i in range(3)))
    high = Vector(tuple(max(c[i] for c in coords) for i in range(3)))
    return (low + high) * 0.5


def smooth_by_angle(mesh_obj: BpyObject, angle_deg: float) -> None:
    """Shade smooth only across edges flatter than ``angle_deg``.

    Args:
        mesh_obj: Mesh object.
        angle_deg: Threshold in degrees; negative disables.

    Raises:
        SystemExit: The angle is above 180 degrees.
    """
    if angle_deg < 0.0:
        return
    if angle_deg > 180.0:
        raise fail("--smooth-angle-deg must be between 0 and 180 (negative disables)")
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.shade_smooth_by_angle(angle=math.radians(angle_deg), keep_sharp_edges=True)
    print(f"Applied smooth shading by angle: {angle_deg:.1f} degrees")


def matrix_rows(scale: float, translation: BpyObject) -> list[list[float]]:
    """Build a row-major 4x4 uniform scale plus translation.

    Args:
        scale: Uniform scale.
        translation: Translation vector with ``x``, ``y`` and ``z``.

    Returns:
        Four rows of four floats.
    """
    return [
        [scale, 0.0, 0.0, translation.x],
        [0.0, scale, 0.0, translation.y],
        [0.0, 0.0, scale, translation.z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def write_frame_transform(
    path: str, input_path: str, units: str, scale: float, center_meters: BpyObject
) -> None:
    """Write the original-to-centered frame mapping, in meters after scaling.

    Args:
        path: Output JSON path.
        input_path: Source mesh path, recorded for provenance.
        units: Source length unit.
        scale: Meters per source unit.
        center_meters: Bounding-box center of the scaled mesh before centering.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    center_original = center_meters / scale
    transform = {
        "source_mesh": input_path,
        "model_frame": FRAME_AABB_CENTERED,
        "units": "m",
        "source_units": units,
        "scale": scale,
        "aabb_center_original": [float(v) for v in center_original],
        "model_from_original": matrix_rows(scale, -center_meters),
        "original_from_model": matrix_rows(1.0 / scale, center_original),
    }
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(transform, stream, indent=2)
        stream.write("\n")
    print(f"Saved frame transform to: {path}")


# Binary STL record: normal, three vertices (12 floats) and the attribute byte count.
STL_TRIANGLE = struct.Struct("<12fH")


def write_binary_stl(path: str, mesh_obj: BpyObject) -> None:
    """Write the mesh as a binary STL in its current (meter) frame.

    Args:
        path: Output STL path.
        mesh_obj: Mesh object.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mesh = mesh_obj.data
    mesh.calc_loop_triangles()
    with open(path, "wb") as stream:
        stream.write(b"\0" * 80)
        stream.write(struct.pack("<I", len(mesh.loop_triangles)))
        for triangle in mesh.loop_triangles:
            vertices = [mesh.vertices[i].co for i in triangle.vertices]
            normal = (vertices[1] - vertices[0]).cross(vertices[2] - vertices[0]).normalized()
            stream.write(
                STL_TRIANGLE.pack(
                    normal.x,
                    normal.y,
                    normal.z,
                    *(coordinate for vertex in vertices for coordinate in vertex),
                    0,
                )
            )
    print(f"Saved centered binary STL to: {path}")


def uvs_from_ply_file(path: str) -> tuple[list[float], list[float]] | None:
    """Read per-vertex ``texture_u``/``texture_v`` straight out of an ascii PLY.

    Blender 4.0's PLY importer drops these properties entirely: the mesh arrives with no UV layer
    and no texture_u/texture_v attributes, so the image texture samples a single texel and the
    object renders flat white. Newer importers build the UV map themselves. The importer keeps
    vertex order and count, so a per-vertex read indexes correctly against ``loop.vertex_index``.

    Args:
        path: PLY file path.

    Returns:
        ``(us, vs)``, or None when the file is not an ascii PLY with those properties.
    """
    try:
        with open(path, "rb") as stream:
            fmt, element, count, props = None, None, 0, []
            while True:
                line = stream.readline()
                if not line:
                    return None
                parts = line.split()
                if not parts:
                    continue
                key = parts[0]
                if key == b"format":
                    fmt = parts[1].decode()
                elif key == b"element":
                    element = parts[1].decode()
                    if element == "vertex":
                        count = int(parts[2])
                elif key == b"property" and element == "vertex":
                    props.append(parts[-1].decode())
                elif key == b"end_header":
                    break
            if fmt != "ascii" or "texture_u" not in props or "texture_v" not in props:
                return None
            iu, iv = props.index("texture_u"), props.index("texture_v")
            us, vs = [0.0] * count, [0.0] * count
            for i in range(count):
                row = stream.readline().split()
                us[i] = float(row[iu])
                vs[i] = float(row[iv])
            return us, vs
    except (OSError, ValueError, IndexError) as err:
        print(f"  Could not read UVs from PLY: {err}")
        return None


def detect_texture(input_path: str, texture: str | None) -> str | None:
    """Find the texture image: ``--texture``, else the PLY ``TextureFile`` comment.

    Args:
        input_path: Mesh file path; relative textures resolve next to it.
        texture: Value of ``--texture``.

    Returns:
        Texture file path, or None.
    """
    directory = os.path.dirname(input_path)
    if texture:
        return texture if os.path.isabs(texture) else os.path.join(directory, texture)
    if not input_path.lower().endswith(".ply"):
        return None
    try:
        with open(input_path, "rb") as stream:
            for line in stream:
                if b"end_header" in line:
                    break
                if b"TextureFile" in line:
                    name = line.decode("ascii", errors="replace").split()[-1].strip()
                    candidate = os.path.join(directory, name)
                    return candidate if os.path.exists(candidate) else None
    except OSError:
        return None
    return None


def build_uv_map(mesh_obj: BpyObject, input_path: str) -> list[BpyObject]:
    """Make sure the mesh has a UV layer, rebuilding it from PLY vertex properties if needed.

    Args:
        mesh_obj: Mesh object.
        input_path: Mesh file path, re-read when the importer dropped the UVs.

    Returns:
        The mesh's UV layers (possibly empty).
    """
    mesh = mesh_obj.data
    if mesh.uv_layers:
        return list(mesh.uv_layers)
    attrs = {a.name for a in mesh.attributes}
    if "texture_u" not in attrs or "texture_v" not in attrs:
        file_uvs = uvs_from_ply_file(input_path)
        if file_uvs is None:
            return []
        print("  Rebuilding UV map from PLY texture_u/texture_v (importer dropped them)...")
        u_vals, v_vals = file_uvs
    else:
        print("  Converting texture_u/texture_v attributes to UV map...")
        u_vals = [0.0] * len(mesh.attributes["texture_u"].data)
        v_vals = [0.0] * len(mesh.attributes["texture_v"].data)
        mesh.attributes["texture_u"].data.foreach_get("value", u_vals)
        mesh.attributes["texture_v"].data.foreach_get("value", v_vals)
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for loop in mesh.loops:
        uv_layer.data[loop.index].uv = (u_vals[loop.vertex_index], v_vals[loop.vertex_index])
    return list(mesh.uv_layers)


def create_textured_material(mesh_obj: BpyObject, texture_path: str, input_path: str) -> None:
    """Replace the mesh materials with a Principled material sampling ``texture_path``.

    The pixels are packed into the ``.blend``: the file travels alone (render bundles, other
    machines) and an unresolved external path renders the object in Blender's magenta
    missing-texture color.

    Args:
        mesh_obj: Mesh object.
        texture_path: Image file.
        input_path: Mesh file path, for UV recovery.
    """
    material = bpy.data.materials.new(name="MeshTexture")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (400, 0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (100, 0)
    bsdf.inputs["Roughness"].default_value = 0.5
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    tex_node = nodes.new("ShaderNodeTexImage")
    tex_node.location = (-300, 0)
    tex_node.image = bpy.data.images.load(texture_path)
    tex_node.image.colorspace_settings.name = "sRGB"
    tex_node.image.pack()
    links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
    uv_layers = build_uv_map(mesh_obj, input_path)
    if uv_layers:
        uv_node = nodes.new("ShaderNodeUVMap")
        uv_node.location = (-500, 0)
        uv_node.uv_map = uv_layers[0].name
        links.new(uv_node.outputs["UV"], tex_node.inputs["Vector"])
    mesh_obj.data.materials.clear()
    mesh_obj.data.materials.append(material)
    print(f"  Applied texture material: {material.name}")


def create_brushed_aluminum_material(name: str = "BrushedAluminum") -> BpyObject:
    """Create a procedural brushed aluminum material with randomization nodes.

    Args:
        name: Material name.

    Returns:
        The new material. Its ``RandomSeed`` value node can be changed at render time.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (800, 0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (500, 0)
    bsdf.name = "MainBSDF"
    # Typical aluminum: RGB around (0.91, 0.92, 0.92) to (0.78, 0.79, 0.80).
    bsdf.inputs["Base Color"].default_value = (0.85, 0.86, 0.87, 1.0)
    bsdf.inputs["Metallic"].default_value = 1.0
    bsdf.inputs["Roughness"].default_value = 0.25
    bsdf.inputs["IOR"].default_value = 1.45
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    # Anisotropic brushed texture for roughness variation.
    tex_coord = nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-600, 200)
    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (-400, 200)
    mapping.name = "BrushMapping"
    mapping.inputs["Scale"].default_value = (50.0, 2.0, 1.0)
    links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])
    noise = nodes.new("ShaderNodeTexNoise")
    noise.location = (-200, 200)
    noise.name = "BrushNoise"
    noise.inputs["Scale"].default_value = 200.0
    noise.inputs["Detail"].default_value = 8.0
    noise.inputs["Roughness"].default_value = 0.7
    noise.inputs["Distortion"].default_value = 0.5
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    wave = nodes.new("ShaderNodeTexWave")
    wave.location = (-200, 0)
    wave.name = "BrushWave"
    wave.wave_type = "BANDS"
    wave.bands_direction = "X"
    wave.inputs["Scale"].default_value = 100.0
    wave.inputs["Distortion"].default_value = 2.0
    wave.inputs["Detail"].default_value = 5.0
    links.new(mapping.outputs["Vector"], wave.inputs["Vector"])
    mix_rgb = nodes.new("ShaderNodeMix")
    mix_rgb.location = (50, 100)
    mix_rgb.data_type = "RGBA"
    mix_rgb.blend_type = "MIX"
    mix_rgb.inputs["Factor"].default_value = 0.5
    links.new(noise.outputs["Fac"], mix_rgb.inputs["A"])
    links.new(wave.outputs["Fac"], mix_rgb.inputs["B"])
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.location = (250, 100)
    ramp.name = "RoughnessRamp"
    ramp.color_ramp.elements[0].position = 0.3
    ramp.color_ramp.elements[0].color = (0.15, 0.15, 0.15, 1.0)
    ramp.color_ramp.elements[1].position = 0.7
    ramp.color_ramp.elements[1].color = (0.35, 0.35, 0.35, 1.0)
    links.new(mix_rgb.outputs["Result"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Roughness"])
    # Subtle color variation for realism.
    voronoi = nodes.new("ShaderNodeTexVoronoi")
    voronoi.location = (-200, -200)
    voronoi.name = "SurfaceImperfections"
    voronoi.inputs["Scale"].default_value = 50.0
    links.new(tex_coord.outputs["Object"], voronoi.inputs["Vector"])
    color_mix = nodes.new("ShaderNodeMix")
    color_mix.location = (250, -100)
    color_mix.data_type = "RGBA"
    color_mix.blend_type = "OVERLAY"
    color_mix.inputs["Factor"].default_value = 0.1
    color_mix.inputs["A"].default_value = (0.85, 0.86, 0.87, 1.0)
    color_mix.inputs["B"].default_value = (0.75, 0.78, 0.82, 1.0)
    links.new(voronoi.outputs["Distance"], color_mix.inputs["Factor"])
    links.new(color_mix.outputs["Result"], bsdf.inputs["Base Color"])
    # Value node for easy randomization.
    seed_node = nodes.new("ShaderNodeValue")
    seed_node.location = (-600, -100)
    seed_node.name = "RandomSeed"
    seed_node.label = "Random Seed (change for variation)"
    seed_node.outputs[0].default_value = 0.0
    math_add = nodes.new("ShaderNodeMath")
    math_add.location = (-400, -100)
    math_add.operation = "ADD"
    math_add.name = "SeedOffset"
    links.new(seed_node.outputs["Value"], math_add.inputs[0])
    return mat


def apply_material(mesh_obj: BpyObject, args: argparse.Namespace, input_path: str) -> None:
    """Keep the authored material or texture, or fall back to brushed aluminum.

    Args:
        mesh_obj: Mesh object.
        args: Parsed options (``keep_material``, ``texture``).
        input_path: Mesh file path.
    """
    existing = next((mat for mat in mesh_obj.data.materials if mat is not None), None)
    texture = detect_texture(input_path, args.texture)
    if texture:
        print(f"\nFound texture file: {os.path.basename(texture)}")
    if existing is not None and args.keep_material:
        print(f"\nKeeping existing material: {existing.name}")
    elif args.keep_material and texture:
        print(f"\nCreating textured material from: {os.path.basename(texture)}")
        create_textured_material(mesh_obj, texture, input_path)
    else:
        # A bare STL keeps the brushed-aluminum look so untextured objects render as before.
        print("\nCreating brushed aluminum material...")
        mesh_obj.data.materials.clear()
        material = create_brushed_aluminum_material("ObjectAluminum")
        mesh_obj.data.materials.append(material)
        print(f"Created material: {material.name}")


def read_face_materials(path: str) -> list[dict[str, Any]]:
    """Read and validate a ``--face-materials`` JSON.

    The file holds ``{"materials": [...]}``. Each material has a unique ``name``, a linear
    ``base_color`` RGBA, ``metallic`` and ``roughness`` in [0, 1] and ``faces``: a list of
    half-open ``[start, end)`` face index ranges, or ``null`` for the one default material that
    covers every face no range names.

    Args:
        path: JSON path.

    Returns:
        The material records in slot order.

    Raises:
        SystemExit: The file is unreadable or malformed.
    """
    try:
        with open(path, encoding="utf-8") as stream:
            materials = json.load(stream)["materials"]
    except (OSError, ValueError, KeyError, TypeError) as err:
        raise fail(f"cannot read face materials from {path}: {err!r}") from err
    if not isinstance(materials, list) or not materials:
        raise fail("--face-materials needs a nonempty materials list")
    names = [str(material.get("name", "")) for material in materials]
    if "" in names or len(set(names)) != len(names):
        raise fail("--face-materials names must be nonempty and unique")
    if sum(material.get("faces") is None for material in materials) != 1:
        raise fail("--face-materials needs exactly one default material (faces null)")
    for material in materials:
        color = material.get("base_color")
        if not isinstance(color, list) or len(color) != 4:
            raise fail(f"material {material['name']!r}: base_color must be linear RGBA")
        values = [*color, material.get("metallic"), material.get("roughness")]
        if not all(isinstance(v, (int, float)) and 0.0 <= v <= 1.0 for v in values):
            raise fail(f"material {material['name']!r}: values must be numbers in [0, 1]")
        faces = material.get("faces")
        if faces is not None and not all(
            isinstance(r, list) and len(r) == 2 and all(isinstance(i, int) for i in r)
            and 0 <= r[0] < r[1] for r in faces
        ):  # fmt: skip
            raise fail(f"material {material['name']!r}: faces must be [start, end) ranges")
    return materials


def apply_face_materials(mesh_obj: BpyObject, materials: Sequence[dict[str, Any]]) -> None:
    """Replace the mesh materials with ``materials`` and assign their face ranges.

    Args:
        mesh_obj: Mesh object.
        materials: Records from :func:`read_face_materials`.

    Raises:
        SystemExit: A range lies outside the mesh or two ranges overlap.
    """
    polygons = mesh_obj.data.polygons
    assigned = [-1] * len(polygons)
    default = next(i for i, m in enumerate(materials) if m["faces"] is None)
    for slot, material in enumerate(materials):
        for start, end in material["faces"] or ():
            if end > len(polygons):
                raise fail(f"material {material['name']!r}: faces [{start}, {end}) exceed the "
                           f"{len(polygons)} faces of the mesh")  # fmt: skip
            if any(assigned[i] != -1 for i in range(start, end)):
                raise fail(f"material {material['name']!r}: faces [{start}, {end}) overlap")
            assigned[start:end] = [slot] * (end - start)
    mesh_obj.data.materials.clear()
    for material in materials:
        created = bpy.data.materials.new(material["name"])
        if created.name != material["name"]:
            raise fail(f"Blender cannot use material name {material['name']!r}")
        created.diffuse_color = tuple(material["base_color"])
        created.use_nodes = True
        bsdf = created.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = tuple(material["base_color"])
        bsdf.inputs["Metallic"].default_value = float(material["metallic"])
        bsdf.inputs["Roughness"].default_value = float(material["roughness"])
        mesh_obj.data.materials.append(created)
    indices = [default if slot == -1 else slot for slot in assigned]
    polygons.foreach_set("material_index", indices)
    mesh_obj.data.update()
    counts = {m["name"]: indices.count(slot) for slot, m in enumerate(materials)}
    print(f"Applied face materials: {counts}")


def farthest_point_indices(mesh_obj: BpyObject, num_points: int) -> list[int]:
    """Pick vertex indices by farthest point sampling, starting at the vertex nearest the centroid.

    Args:
        mesh_obj: Mesh object.
        num_points: Number of vertices to choose; clipped to the vertex count.

    Returns:
        Selected vertex indices in selection order.
    """
    vertices = [mesh_obj.matrix_world @ v.co for v in mesh_obj.data.vertices]
    if num_points > len(vertices):
        print(f"Warning: requested {num_points} keypoints but mesh only has {len(vertices)}")
        num_points = len(vertices)
    centroid = sum(vertices, Vector()) / len(vertices)
    first = min(range(len(vertices)), key=lambda i: (vertices[i] - centroid).length)
    selected = [first]
    distances = [(v - vertices[first]).length for v in vertices]
    for _ in range(num_points - 1):
        farthest = max(range(len(vertices)), key=distances.__getitem__)
        selected.append(farthest)
        distances = [
            min(d, (v - vertices[farthest]).length)
            for d, v in zip(distances, vertices, strict=True)
        ]
    return selected


def create_keypoint_empties(
    mesh_obj: BpyObject, names: Sequence[str], positions: Sequence[Sequence[float]]
) -> list[BpyObject]:
    """Create one sphere empty per keypoint, parented to the mesh at its local position.

    Args:
        mesh_obj: Mesh object that parents the empties.
        names: Empty names, in output-channel order.
        positions: Mesh-local positions in meters.

    Returns:
        The empties.

    Raises:
        SystemExit: Blender renamed an empty (duplicate or over-long name).
    """
    bpy.context.view_layer.objects.active = mesh_obj
    empties = []
    for name, position in zip(names, positions, strict=True):
        bpy.ops.object.empty_add(type="SPHERE", radius=0.001, location=(0, 0, 0))
        empty = bpy.context.active_object
        empty.name = name
        if empty.name != name:
            raise fail(f"Blender cannot use keypoint name {name!r} (got {empty.name!r})")
        empty.empty_display_size = mesh_obj.dimensions[0] * KEYPOINT_DISPLAY_FRACTION
        empty.parent = mesh_obj
        empty.location = tuple(position)
        empties.append(empty)
    return empties


def add_camera_and_light(mesh_size: float) -> None:
    """Add a camera framing the object and a sun lamp.

    Args:
        mesh_size: Largest bounding-box extent in meters.
    """
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 50
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    distance = mesh_size * CAMERA_DISTANCE_FACTOR
    cam_obj.location = (distance, -distance, distance)
    cam_obj.rotation_euler = (math.radians(54.7), 0, math.radians(45))
    light_data = bpy.data.lights.new("Sun", type="SUN")
    light_data.energy = 2.0
    light_obj = bpy.data.objects.new("Sun", light_data)
    bpy.context.scene.collection.objects.link(light_obj)
    light_obj.rotation_euler = (math.radians(45), math.radians(30), 0)


def write_keypoints_json(
    path: str,
    args: argparse.Namespace,
    names: Sequence[str],
    positions: Sequence[Sequence[float]],
    scale: float,
    object_frame: str,
) -> None:
    """Write the keypoints JSON: the source fields plus the unit and frame record.

    Args:
        path: Output path.
        args: Parsed options (``mesh_name``, ``source_units``).
        names: Keypoint names.
        positions: Mesh-local positions in meters.
        scale: Meters per source unit.
        object_frame: ``cad`` or ``aabb_centered``.
    """
    payload = keypoints_payload(args.mesh_name, names, positions)
    payload["keypoints_3d"] = [list(map(float, p)) for p in positions]
    payload.update(
        units="m",
        source_units=args.source_units,
        scale=scale,
        object_frame=object_frame,
        coordinates_sha256=coordinates_sha256(positions),
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    print(f"Saved keypoints to: {path}")


def write_usd(
    path: str,
    mesh_obj: BpyObject,
    args: argparse.Namespace,
    names: Sequence[str],
    positions: Sequence[Sequence[float]],
    scale: float,
) -> None:
    """Export the mesh and keypoints as a ``.usda`` in the keypoint frame.

    Args:
        path: Output path.
        mesh_obj: Mesh object.
        args: Parsed options.
        names: Keypoint names.
        positions: Mesh-local positions in meters.
        scale: Meters per source unit.
    """
    mesh = mesh_obj.data
    uvs, texture_file = mesh_uvs_and_texture(mesh_obj, path)
    prim = write_mesh_usda(
        path,
        object_name=args.mesh_name,
        points=[tuple(v.co) for v in mesh.vertices],
        face_vertex_counts=[len(p.vertices) for p in mesh.polygons],
        face_vertex_indices=[i for p in mesh.polygons for i in p.vertices],
        keypoint_names=list(names),
        keypoints_3d=[tuple(p) for p in positions],
        uvs=uvs,
        texture_file=texture_file,
        metadata={
            "keypoint_prefix": f"KP_{args.mesh_name}_",
            "source_mesh": os.path.basename(args.input),
            "source_units": args.source_units,
            "source_scale": float(scale),
            "aabb_centered": bool(args.center_geometry),
        },
    )
    print(f"Saved USD to: {path} (mesh prim {prim})")


def main(argv: Sequence[str] | None = None) -> None:
    """Prepare the mesh and keypoints and save the ``.blend`` and optional exports.

    Args:
        argv: Argument strings, or None to read Blender's script arguments.

    Raises:
        SystemExit: An option or the keypoints input is invalid.
    """
    args = parse_args(argv)
    scale = UNIT_SCALES[args.source_units]
    object_frame = FRAME_AABB_CENTERED if args.center_geometry else FRAME_CAD
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    for flag in ("frame_transform_output", "centered_mesh_output"):
        if getattr(args, flag) and not args.center_geometry:
            raise fail(f"--{flag.replace('_', '-')} requires --center-geometry")
    if args.num_keypoints <= 0 and not args.keypoints_input:
        raise fail("--num-keypoints must be positive")
    face_materials = read_face_materials(args.face_materials) if args.face_materials else None
    input_names = input_points = None
    if args.keypoints_input:
        input_names, input_points = read_keypoints_input(
            args.keypoints_input, object_frame, args.mesh_name
        )
    print(f"Input: {input_path} ({args.source_units}, x{scale:g} to meters)")
    print(f"Output: {output_path}")
    clear_scene()
    mesh_obj = import_mesh(input_path, args.mesh_name)
    scale_to_meters(mesh_obj, scale)
    center = aabb_center(mesh_obj)
    if args.center_geometry:
        print(f"Centering geometry at AABB center: {[round(v, 9) for v in center]}")
        for vertex in mesh_obj.data.vertices:
            vertex.co -= center
        mesh_obj.data.update()
    smooth_by_angle(mesh_obj, args.smooth_angle_deg)
    mesh_size = max(mesh_obj.dimensions)
    print(f"Mesh size after scaling: {mesh_size:.4f} m")
    if args.frame_transform_output:
        transform_path = os.path.abspath(args.frame_transform_output)
        write_frame_transform(transform_path, input_path, args.source_units, scale, center)
    if args.centered_mesh_output:
        write_binary_stl(os.path.abspath(args.centered_mesh_output), mesh_obj)
    if face_materials is not None:
        apply_face_materials(mesh_obj, face_materials)
    else:
        apply_material(mesh_obj, args, input_path)
    if input_names is not None and input_points is not None:
        names, positions = input_names, input_points
        print(f"\nAuthoring {len(names)} keypoints from {args.keypoints_input}...")
    else:
        print(f"\nDistributing {args.num_keypoints} keypoints...")
        indices = farthest_point_indices(mesh_obj, args.num_keypoints)
        names = [f"KP_{args.mesh_name}_{i:03d}" for i in range(len(indices))]
        points = [mesh_obj.data.vertices[i].co for i in indices]
        positions = [[round(float(c), KEYPOINT_DECIMALS) for c in p] for p in points]
    create_keypoint_empties(mesh_obj, names, positions)
    add_camera_and_light(mesh_size)
    keypoints_path = os.path.abspath(
        args.keypoints_output
        or os.path.join(os.path.dirname(output_path), f"{args.mesh_name}_keypoints_3d.json")
    )
    write_keypoints_json(keypoints_path, args, names, positions, scale, object_frame)
    if args.usd_output:
        write_usd(os.path.abspath(args.usd_output), mesh_obj, args, names, positions, scale)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=output_path)
    print(f"\nSetup complete: {output_path}")
    print(f"Mesh name: {args.mesh_name}; keypoint prefix: KP_{args.mesh_name}_")
    print(f"Number of keypoints: {len(names)}")


if __name__ == "__main__":
    main()
