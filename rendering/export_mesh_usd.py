#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export the object mesh and its keypoint empties from a .blend file as USDA.

Use this for objects whose authored ``.blend`` already exists so that the
Isaac Sim renderer gets the exact authored mesh frame and keypoint order instead of
re-sampling keypoints with ``mesh_setup.py``. The mesh vertices are exported in
mesh-local coordinates and the keypoints as their parent-local ``location``,
which is precisely the frame ``synth_worker.py`` writes to ``keypoints_3d.json``.

Run with:
    blender --background object.blend --python rendering/export_mesh_usd.py -- \
        --mesh-name bracket --output bracket.usda [--keypoints-output bracket_keypoints_3d.json]
"""

# pyright: reportMissingImports=false, reportMissingModuleSource=false

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

script_dir = os.path.dirname(os.path.realpath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
from usd_mesh import keypoints_payload, write_mesh_usda  # noqa: E402

LOGGER = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse export options after Blender's argument separator.

    Args:
        argv: Export argument strings, or None to read sys.argv after ``--``.

    Returns:
        Namespace of input mesh, keypoint and output settings.
    """
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-name", required=True, help="Mesh object name in the blend file")
    parser.add_argument("--output", required=True, help="Output .usda path")
    parser.add_argument(
        "--keypoint-prefix", default=None, help="Keypoint empty prefix (default KP_<mesh_name>_)"
    )
    parser.add_argument(
        "--keypoints-json",
        default=None,
        help="Take keypoints from this JSON instead of the scene empties",
    )
    parser.add_argument(
        "--keypoints-output",
        default=None,
        help="Also write the exported keypoints as keypoints_3d.json",
    )
    return parser.parse_args(argv)


def collect_keypoints(
    mesh_name: str, keypoint_prefix: str | None = None, keypoints_json: str | None = None
) -> tuple[Any, list[str], list[list[float]]]:
    """Read ordered keypoints in the existing mesh-local training frame.

    Args:
        mesh_name: Name of the Blender mesh object to export.
        keypoint_prefix: Scene-empty prefix; defaults to KP_<mesh_name>_.
        keypoints_json: Optional authoritative JSON file; preserves its order.

    Returns:
        Mesh object, ordered keypoint names and mesh-local coordinates.

    Raises:
        SystemExit: The mesh or keypoint empties are missing or mis-parented.
    """
    import bpy

    mesh_obj = bpy.data.objects.get(mesh_name)
    if mesh_obj is None or mesh_obj.type != "MESH":
        names = ", ".join(o.name for o in bpy.data.objects if o.type == "MESH")
        raise SystemExit(f"Mesh object '{mesh_name}' not found. Meshes in file: {names}")

    kp_prefix = keypoint_prefix or f"KP_{mesh_name}_"
    if keypoints_json:
        with open(keypoints_json, encoding="utf-8") as f:
            payload = json.load(f)
        payload_object = payload.get("object")
        if payload_object is not None and payload_object != mesh_name:
            raise SystemExit(
                f"Keypoint JSON object '{payload_object}' does not match mesh '{mesh_name}'"
            )
        if len(payload["keypoint_names"]) != len(payload["keypoints_3d"]):
            raise SystemExit("Keypoint JSON keypoint_names/keypoints_3d length mismatch")
        kp_names = list(payload["keypoint_names"])
        kp_3d = [[float(c) for c in p] for p in payload["keypoints_3d"]]
    else:
        kp_objs = sorted(
            (o for o in bpy.data.objects if o.name.startswith(kp_prefix)),
            key=lambda o: o.name,
        )
        if not kp_objs:
            raise SystemExit(f"No keypoint empties with prefix '{kp_prefix}' found")
        for kp in kp_objs:
            if kp.parent is not mesh_obj:
                raise SystemExit(f"Keypoint '{kp.name}' is not parented to '{mesh_obj.name}'")
        kp_names = [o.name for o in kp_objs]
        kp_3d = [[round(float(c), 6) for c in o.location] for o in kp_objs]

    return mesh_obj, kp_names, kp_3d


def main(argv: Sequence[str] | None = None) -> None:
    """Export the selected mesh without resampling its keypoints.

    Args:
        argv: Export argument strings, or None to read Blender's script arguments.
    """
    args = parse_args(argv)
    import bpy
    from usd_blender import mesh_uvs_and_texture

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    mesh_obj, kp_names, kp_3d = collect_keypoints(
        args.mesh_name,
        args.keypoint_prefix,
        args.keypoints_json,
    )
    mesh = mesh_obj.data
    uvs, texture_file = mesh_uvs_and_texture(mesh_obj, args.output)
    mesh_prim = write_mesh_usda(
        args.output,
        object_name=args.mesh_name,
        points=[tuple(v.co) for v in mesh.vertices],
        face_vertex_counts=[len(p.vertices) for p in mesh.polygons],
        face_vertex_indices=[i for p in mesh.polygons for i in p.vertices],
        keypoint_names=kp_names,
        keypoints_3d=kp_3d,
        uvs=uvs,
        texture_file=texture_file,
        metadata={
            "keypoint_prefix": args.keypoint_prefix or f"KP_{args.mesh_name}_",
            "source_blend": os.path.basename(bpy.data.filepath),
        },
    )
    LOGGER.info(
        "Exported %d vertices, %d faces, %d keypoints",
        len(mesh.vertices),
        len(mesh.polygons),
        len(kp_names),
    )
    LOGGER.info("USD: %s (mesh prim %s)", args.output, mesh_prim)

    if args.keypoints_output:
        Path(args.keypoints_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.keypoints_output, "w") as f:
            json.dump(keypoints_payload(args.mesh_name, kp_names, kp_3d), f, indent=2)
        LOGGER.info("Keypoints JSON: %s", args.keypoints_output)


if __name__ == "__main__":
    main()
