# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read authored Blender inputs before deciding whether existing samples are reusable."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

_INSPECT_SCRIPT = r"""
import bpy
import json
import math
import os

request = json.loads(REQUEST_JSON)
mesh = bpy.data.objects.get(request["mesh_name"])
if mesh is None or mesh.type != "MESH":
    raise ValueError("requested object is not a mesh: " + request["mesh_name"])
if any(abs(float(v) - 1.0) > 1e-6 for v in mesh.scale):
    raise ValueError("object scale must be applied before deterministic rendering")
if mesh.parent is not None or mesh.constraints:
    raise ValueError("object parenting/constraints must be baked before deterministic rendering")
if mesh.modifiers:
    raise ValueError("object modifiers must be applied before deterministic rendering")
names = [f'{request["keypoint_prefix"]}{i:03d}' for i in range(request["num_keypoints"])]
prefix = request["keypoint_prefix"]
found = sorted(obj.name for obj in bpy.data.objects if obj.name.startswith(prefix))
if found != sorted(names):
    raise ValueError("authored keypoint names/count do not match requested schema")
bpy.context.view_layer.update()
keypoints = []
for name in names:
    point = bpy.data.objects[name]
    if point.type != "EMPTY" or point.parent != mesh:
        raise ValueError("keypoint must be an empty parented to the requested mesh: " + name)
    local = mesh.matrix_world.inverted() @ point.matrix_world.translation
    # synth_worker exports point.location as the object-frame keypoint. Refuse
    # parent-inverse/constraint transforms that would make that export incorrect.
    if (local - point.location).length > 1e-6:
        raise ValueError("keypoint parent/constraint transform changes its object frame: " + name)
    keypoints.append(list(local))
# Only dependencies exclusive to replaced material slots can be ignored. Follow
# Blender's ID users through nested node groups, preserving shared/world users.
replaced_materials = set()
if request["object_material_mode"] == "override":
    replaced_materials = {slot.material for slot in mesh.material_slots if slot.material}
    for obj in bpy.data.objects:
        if obj != mesh:
            replaced_materials.difference_update(slot.material for slot in obj.material_slots)
users = bpy.data.user_map()

def only_replaced_material_users(block, visiting=None):
    if block in replaced_materials:
        return True
    if not isinstance(block, (bpy.types.Image, bpy.types.NodeTree)):
        return False
    visiting = set() if visiting is None else visiting
    if block in visiting:
        return False
    dependents = users.get(block, set())
    return bool(dependents) and all(
        only_replaced_material_users(user, visiting | {block}) for user in dependents)

textures = set()
external = set()
for image in bpy.data.images:
    if replaced_materials and only_replaced_material_users(image):
        continue
    if image.packed_file is not None or image.source in {"GENERATED", "VIEWER"}:
        continue
    if image.source != "FILE":
        raise ValueError("external image source must be packed before rendering: " + image.name)
    path = os.path.realpath(bpy.path.abspath(image.filepath, library=image.library))
    if not os.path.isfile(path):
        raise ValueError("missing external texture: " + path)
    textures.add(path)
for library in bpy.data.libraries:
    path = os.path.realpath(bpy.path.abspath(library.filepath))
    if not os.path.isfile(path):
        raise ValueError("missing linked Blender library: " + path)
    external.add(path)
points = [list(vertex.co) for vertex in mesh.data.vertices]
if not points or not all(math.isfinite(v) for point in points + keypoints for v in point):
    raise ValueError("object geometry/keypoints must be nonempty and finite")
result = {"textures": sorted(textures), "external_files": sorted(external | textures),
          "points": points, "keypoint_names": names, "keypoints_3d": keypoints}
with open(request["output"], "w") as stream:
    json.dump(result, stream, allow_nan=False)
"""


def inspect_blender_inputs(
    blender_bin: str,
    blend_path: str,
    mesh_name: str,
    keypoint_prefix: str,
    num_keypoints: int,
    *,
    object_material_mode: str = "preserve",
    timeout: float = 60,
) -> dict[str, Any]:
    """Inspect without rendering or saving the blend; fail before launcher mutation.

    Hash ``external_files`` alongside the .blend: these include unpacked image
    bytes and linked libraries. Packed images are already covered by the blend
    digest. Override mode excludes images used only by replaced target materials;
    shared material, node-group and world dependencies remain mandatory.
    The subprocess is bounded and disables authored auto-execution.
    """
    if num_keypoints <= 0 or not keypoint_prefix:
        raise ValueError("a positive keypoint count and nonempty prefix are required")
    if object_material_mode not in {"preserve", "override"}:
        raise ValueError("object_material_mode must be preserve or override")
    blend = Path(blend_path).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="doper-blender-inputs-") as directory:
        output = Path(directory) / "inputs.json"
        request = {
            "mesh_name": mesh_name,
            "keypoint_prefix": keypoint_prefix,
            "num_keypoints": num_keypoints,
            "output": str(output),
            "object_material_mode": object_material_mode,
        }
        expression = _INSPECT_SCRIPT.replace("REQUEST_JSON", repr(json.dumps(request)), 1)
        try:
            result = subprocess.run(
                [
                    str(blender_bin),
                    "--background",
                    "--factory-startup",
                    "--disable-autoexec",
                    str(blend),
                    "--python-exit-code",
                    "1",
                    "--python-expr",
                    expression,
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError(f"Blender input inspection timed out after {timeout}s") from exc
        if result.returncode:
            detail = (result.stdout + result.stderr)[-4000:]
            raise ValueError(
                f"Blender input inspection failed (exit {result.returncode}): {detail}"
            )
        try:
            payload = json.loads(output.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError("Blender did not produce valid input inspection metadata") from exc
        return payload
