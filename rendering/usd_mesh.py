# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free USDA export of an object mesh plus its keypoints.

The Isaac Sim rendering path cannot load ``.blend`` files, so the object must
be exported as USD. This module writes an ASCII ``.usda`` layer from raw
vertex/face arrays and keypoint positions so that the mesh and the keypoints
are guaranteed to share the exact same object frame as ``keypoints_3d.json``.
It intentionally avoids ``pxr`` and ``bpy`` so it can run inside Blender's
bundled Python, a plain virtualenv, or the Isaac Sim container, and so the
output can be unit tested without either.

Layout of the written stage::

    /<object>                 Xform (defaultPrim, kind=component)
    /<object>/mesh            Mesh   (points, faces, normals-free, no subdiv)
    /<object>/keypoints       Scope
    /<object>/keypoints/KP_*  Xform  (xformOp:translate = keypoint position)
    /<object>/Looks/Texture   Material (optional: UsdPreviewSurface reading the object's UV texture)

Stage metadata is Z-up, ``metersPerUnit = 1`` (the model frame is treated as
meters throughout the repo), and ``customLayerData`` records the object name,
keypoint prefix/count, and provenance so the file is self-describing.

Textured objects (BOP/HANDAL/HOPE meshes ship UV-mapped PNGs) additionally get
``primvars:st`` (faceVarying, one UV per face corner) and a ``UsdPreviewSurface``
material bound to the mesh whose ``UsdUVTexture`` references the texture by a
relative asset path, so the ``.usda`` and the texture travel together (render
bundles copy both). ``customLayerData.texture_file`` names it for tooling.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

Vec3 = Sequence[float]

USD_MESH_PRIM = "mesh"
USD_KEYPOINTS_SCOPE = "keypoints"
USD_LOOKS_SCOPE = "Looks"
USD_TEXTURE_MATERIAL = "Texture"


def sanitize_prim_name(name: str) -> str:
    """Return a valid USD prim identifier derived from a display name.

    Args:
        name: Object or keypoint display name.

    Returns:
        Sanitized prim identifier; a leading digit is prefixed with an underscore.
    """
    cleaned = re.sub(r"\W", "_", str(name))
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return cleaned


def _fmt(value: float) -> str:
    """Format a coordinate for the ASCII layer.

    Args:
        value: Numeric coordinate or metadata value.

    Returns:
        Compact decimal text without negative zero.
    """
    text = f"{float(value):.7g}"
    if "e" in text or "E" in text:
        text = f"{float(value):.9f}".rstrip("0").rstrip(".")
    return text if text not in {"-0", ""} else "0"


def _fmt_vec(vec: Vec3) -> str:
    """Format a coordinate vector as a USDA tuple.

    Args:
        vec: Coordinate components in authored order.

    Returns:
        Parenthesized, comma-separated decimal components.
    """
    return "(" + ", ".join(_fmt(v) for v in vec) + ")"


def _usd_string(value: str) -> str:
    """Quote a string for USDA string-valued attributes.

    Args:
        value: Unescaped string value.

    Returns:
        JSON-style quoted string literal.
    """
    return json.dumps(str(value))


def _custom_layer_data(items: Mapping[str, Any]) -> str:
    """Serialize scalar custom layer metadata.

    Args:
        items: Metadata keys and scalar values; None values are omitted.

    Returns:
        Indented USDA dictionary entries.
    """
    lines = []
    for key, value in items.items():
        if value is None:
            continue
        if isinstance(value, bool):
            lines.append(f"        bool {key} = {'true' if value else 'false'}")
        elif isinstance(value, int):
            lines.append(f"        int {key} = {value}")
        elif isinstance(value, float):
            lines.append(f"        double {key} = {_fmt(value)}")
        else:
            lines.append(f"        string {key} = {_usd_string(value)}")
    return "\n".join(lines)


def compute_extent(points: Iterable[Vec3]) -> list[list[float]]:
    """Compute an axis-aligned mesh-local bounding box.

    Args:
        points: Iterable of three-component mesh vertex positions.

    Returns:
        Minimum and maximum corners; both zero for an empty mesh.
    """
    pts = [tuple(float(c) for c in p) for p in points]
    if not pts:
        return [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    lo = [min(p[i] for p in pts) for i in range(3)]
    hi = [max(p[i] for p in pts) for i in range(3)]
    return [lo, hi]


def write_mesh_usda(
    path: str,
    *,
    object_name: str,
    points: Sequence[Vec3],
    face_vertex_counts: Sequence[int],
    face_vertex_indices: Sequence[int],
    keypoint_names: Sequence[str] = (),
    keypoints_3d: Sequence[Vec3] = (),
    metadata: Mapping[str, Any] | None = None,
    uvs: Sequence[Sequence[float]] | None = None,
    texture_file: str | None = None,
) -> str:
    """Write ``path`` as a USDA layer and return the prim path of the mesh.

    Args:
        path: Destination USDA path; parent directories are created.
        object_name: Source object name, sanitized for its USD root prim.
        points: Mesh-local vertex positions in meters.
        face_vertex_counts: Number of vertices per polygon.
        face_vertex_indices: Flattened polygon vertex indices in winding order.
        keypoint_names: Keypoint names in model output-channel order.
        keypoints_3d: Corresponding mesh-local keypoint positions in meters.
        metadata: Optional scalar provenance entries for customLayerData.
        uvs: Optional UV pair per face corner, in index-array order.
        texture_file: Optional relative image path next to the USDA layer.

    Returns:
        Absolute USD prim path of the authored mesh (not a filesystem path).

    Raises:
        ValueError: Array lengths, vertex references or texture inputs disagree.

    ``points`` and ``keypoints_3d`` must be expressed in the same object frame
    (the mesh-local frame that ``keypoints_3d.json`` uses). ``uvs`` are per face
    corner in ``face_vertex_indices`` order (faceVarying). ``texture_file`` is a
    path relative to the ``.usda`` (normally a bare file name next to it); it
    requires ``uvs`` and yields a bound UsdPreviewSurface material.
    """
    if len(keypoint_names) != len(keypoints_3d):
        raise ValueError(
            f"keypoint_names ({len(keypoint_names)}) and keypoints_3d "
            f"({len(keypoints_3d)}) length mismatch"
        )
    if sum(int(c) for c in face_vertex_counts) != len(face_vertex_indices):
        raise ValueError("face_vertex_counts do not sum to len(face_vertex_indices)")
    if face_vertex_indices and max(int(i) for i in face_vertex_indices) >= len(points):
        raise ValueError("face_vertex_indices reference a vertex outside points")
    if uvs is not None and len(uvs) != len(face_vertex_indices):
        raise ValueError(
            f"uvs ({len(uvs)}) must have one entry per face corner ({len(face_vertex_indices)})"
        )
    if texture_file and uvs is None:
        raise ValueError("texture_file requires uvs")
    if texture_file and os.path.isabs(texture_file):
        raise ValueError("texture_file must be relative to the .usda (copy the texture next to it)")

    for field, coordinates, arity in (
        ("points", points, 3),
        ("keypoints_3d", keypoints_3d, 3),
        ("uvs", uvs if uvs is not None else (), 2),
    ):
        for index, coordinate in enumerate(coordinates):
            if len(coordinate) != arity:
                raise ValueError(f"{field}[{index}] must have {arity} components")

    root = sanitize_prim_name(object_name)
    extent = compute_extent(points)
    layer_data = {
        "object": str(object_name),
        "model_frame": "mesh_local",
        "units": "meters",
        "num_keypoints": len(keypoint_names),
        "keypoint_scope": f"/{root}/{USD_KEYPOINTS_SCOPE}",
        "mesh_prim": f"/{root}/{USD_MESH_PRIM}",
    }
    if texture_file:
        layer_data["texture_file"] = str(texture_file)
    if metadata:
        layer_data.update({str(k): v for k, v in metadata.items()})

    counts_text = ", ".join(str(int(c)) for c in face_vertex_counts)
    indices_text = ", ".join(str(int(i)) for i in face_vertex_indices)
    points_text = ", ".join(_fmt_vec(p) for p in points)

    keypoint_blocks = []
    for name, pos in zip(keypoint_names, keypoints_3d, strict=True):
        prim = sanitize_prim_name(name)
        keypoint_blocks.append(
            f'        def Xform "{prim}"\n'
            "        {\n"
            f"            custom string keypoint_name = {_usd_string(name)}\n"
            f"            double3 xformOp:translate = {_fmt_vec(pos)}\n"
            '            uniform token[] xformOpOrder = ["xformOp:translate"]\n'
            "        }"
        )
    keypoints_text = "\n".join(keypoint_blocks)

    st_text = ""
    if uvs is not None:
        st_values = ", ".join("(" + _fmt(float(u)) + ", " + _fmt(float(v)) + ")" for u, v in uvs)
        st_text = (
            f"        texCoord2f[] primvars:st = [{st_values}] (\n"
            '            interpolation = "faceVarying"\n'
            "        )\n"
        )
    mesh_schemas_text = ""
    binding_text = ""
    looks_text = ""
    if texture_file:
        material_path = f"/{root}/{USD_LOOKS_SCOPE}/{USD_TEXTURE_MATERIAL}"
        mesh_schemas_text = ' (\n        prepend apiSchemas = ["MaterialBindingAPI"]\n    )'
        binding_text = f"        rel material:binding = <{material_path}>\n"
        looks_text = f"""
    def Scope "{USD_LOOKS_SCOPE}"
    {{
        def Material "{USD_TEXTURE_MATERIAL}"
        {{
            token outputs:surface.connect = <{material_path}/Surface.outputs:surface>

            def Shader "Surface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = <{material_path}/DiffuseTexture.outputs:rgb>
                float inputs:metallic = 0
                float inputs:roughness = 0.5
                token outputs:surface
            }}

            def Shader "DiffuseTexture"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @./{texture_file}@
                token inputs:sourceColorSpace = "sRGB"
                float2 inputs:st.connect = <{material_path}/UvReader.outputs:result>
                token inputs:wrapS = "repeat"
                token inputs:wrapT = "repeat"
                float3 outputs:rgb
            }}

            def Shader "UvReader"
            {{
                uniform token info:id = "UsdPrimvarReader_float2"
                token inputs:varname = "st"
                float2 outputs:result
            }}
        }}
    }}
"""

    content = f"""#usda 1.0
(
    defaultPrim = "{root}"
    doc = "DOPER object mesh and keypoints; frame matches keypoints_3d.json"
    metersPerUnit = 1
    upAxis = "Z"
    customLayerData = {{
{_custom_layer_data(layer_data)}
    }}
)

def Xform "{root}" (
    kind = "component"
)
{{
    def Mesh "{USD_MESH_PRIM}"{mesh_schemas_text}
    {{
        float3[] extent = [{_fmt_vec(extent[0])}, {_fmt_vec(extent[1])}]
        int[] faceVertexCounts = [{counts_text}]
        int[] faceVertexIndices = [{indices_text}]
{binding_text}        point3f[] points = [{points_text}]
{st_text}        uniform token subdivisionScheme = "none"
    }}

    def Scope "{USD_KEYPOINTS_SCOPE}"
    {{
{keypoints_text}
    }}
{looks_text}}}
"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"/{root}/{USD_MESH_PRIM}"


def _parse_vec_list(text: str) -> list[list[float]]:
    """Read numeric tuples from a writer-generated array.

    Args:
        text: USDA array contents without enclosing brackets.

    Returns:
        Coordinate lists in authored order.
    """
    return [[float(c) for c in group.split(",")] for group in re.findall(r"\(([^()]*)\)", text)]


def _parse_int_list(text: str) -> list[int]:
    """Read integers from a writer-generated array.

    Args:
        text: Comma-separated array contents.

    Returns:
        Integer values in authored order.
    """
    text = text.strip()
    return [int(t) for t in text.split(",") if t.strip()] if text else []


def _parse_layer_data(header: str) -> dict[str, Any]:
    """Read scalar metadata from a writer-generated header.

    Args:
        header: Header text inside the stage metadata parentheses.

    Returns:
        Dictionary of bool, int, float and string metadata.
    """
    layer_data: dict[str, Any] = {}
    for kind, key, raw in re.findall(r"(bool|int|double|string)\s+(\w+)\s*=\s*(.+)", header):
        raw = raw.strip()
        if kind == "bool":
            layer_data[key] = raw == "true"
        elif kind == "int":
            layer_data[key] = int(raw)
        elif kind == "double":
            layer_data[key] = float(raw)
        else:
            layer_data[key] = json.loads(raw)
    return layer_data


def read_layer_data(path: str) -> dict[str, Any]:
    """Read only the custom metadata header of a writer-generated layer.

    Args:
        path: UTF-8 USDA file produced by write_mesh_usda.

    Returns:
        Custom layer metadata without loading the full mesh arrays.
    """
    header_lines: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            header_lines.append(line)
            if line.rstrip("\n") == ")":
                break
    match = re.search(r"^#usda 1\.0\s*\((.*?)\n\)\n", "".join(header_lines), flags=re.S)
    if match is None:
        raise ValueError(f"{path} does not start with a usda header")
    return _parse_layer_data(match.group(1))


def usd_texture_file(usd_path: str) -> str | None:
    """Resolve the texture recorded in a writer-generated layer.

    Args:
        usd_path: USDA file whose header names an optional texture.

    Returns:
        Absolute texture path, or None for an untextured mesh; existence is not checked.
    """
    name = read_layer_data(usd_path).get("texture_file")
    return os.path.join(os.path.dirname(os.path.abspath(usd_path)), name) if name else None


def read_mesh_usda(path: str) -> dict[str, Any]:
    """Parse a layer written by :func:`write_mesh_usda`.

    Args:
        path: UTF-8 USDA file produced by write_mesh_usda.

    Returns:
        Mesh arrays, ordered keypoints, texture and stage metadata.

    Raises:
        ValueError: A required writer-generated attribute is absent or malformed.

    This is a purpose-built reader for the writer above (used by tests and by
    tooling that lacks ``pxr``); it is not a general USD parser.
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()

    def attr(pattern: str) -> str:
        """Extract a required attribute's first capture group.

        Args:
            pattern: Regular expression with one capture group for the value.

        Returns:
            Matched attribute text before numeric parsing.
        """
        match = re.search(pattern, text, flags=re.S)
        if match is None:
            raise ValueError(f"missing attribute matching {pattern!r} in {path}")
        return match.group(1)

    layer_data = _parse_layer_data(attr(r"^#usda 1\.0\s*\((.*?)\n\)\n"))

    keypoint_names: list[str] = []
    keypoints_3d: list[list[float]] = []
    for name, pos in re.findall(
        r'custom string keypoint_name = ("(?:\\.|[^"\\])*")\s*'
        r"double3 xformOp:translate = (\([^()]*\))",
        text,
    ):
        keypoint_names.append(json.loads(name))
        keypoints_3d.append(_parse_vec_list(pos)[0])

    st_match = re.search(r"texCoord2f\[\] primvars:st = \[(.*?)\] \(", text, flags=re.S)
    return {
        "texture_file": layer_data.get("texture_file"),
        "uvs": _parse_vec_list(st_match.group(1)) if st_match else None,
        "default_prim": attr(r'defaultPrim = "([^"]+)"'),
        "up_axis": attr(r'upAxis = "([^"]+)"'),
        "meters_per_unit": float(attr(r"metersPerUnit = ([0-9.eE+-]+)")),
        "custom_layer_data": layer_data,
        "points": _parse_vec_list(attr(r"point3f\[\] points = \[(.*?)\]\n")),
        "face_vertex_counts": _parse_int_list(attr(r"int\[\] faceVertexCounts = \[(.*?)\]")),
        "face_vertex_indices": _parse_int_list(attr(r"int\[\] faceVertexIndices = \[(.*?)\]")),
        "extent": _parse_vec_list(attr(r"float3\[\] extent = \[(.*?)\]")),
        "keypoint_names": keypoint_names,
        "keypoints_3d": keypoints_3d,
    }


def keypoints_payload(
    object_name: str, keypoint_names: Sequence[str], keypoints_3d: Sequence[Vec3]
) -> dict[str, Any]:
    """Build the repository keypoints_3d.json payload.

    Args:
        object_name: Name of the source mesh object.
        keypoint_names: Keypoint names in model output-channel order.
        keypoints_3d: Corresponding mesh-local positions in meters.

    Returns:
        JSON-compatible payload with positions rounded to six decimal places.
    """
    return {
        "object": str(object_name),
        "num_keypoints": len(keypoint_names),
        "keypoint_names": list(keypoint_names),
        "keypoints_3d": [[round(float(c), 6) for c in pos] for pos in keypoints_3d],
    }
