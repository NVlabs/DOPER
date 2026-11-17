# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Blender-side helpers for the USD export (imported by the bpy scripts only).

``mesh_uvs_and_texture`` pulls what :func:`rendering.usd_mesh.write_mesh_usda` needs to write a
textured object: the active UV layer as one UV per face corner, in the same corner order as
``[i for p in polygons for i in p.vertices]``, and the image of the first Image Texture node in the
mesh's materials, copied next to the ``.usda`` so the layer's relative asset path resolves anywhere
the two files travel together (render bundles).
"""

# pyright: reportMissingImports=false, reportMissingModuleSource=false

from __future__ import annotations

import logging
import os
import shutil

import bpy

LOGGER = logging.getLogger(__name__)


def image_texture(mesh_obj: bpy.types.Object) -> bpy.types.Image | None:
    """Find the first image texture in the object's materials.

    Args:
        mesh_obj: Blender mesh object whose material slots are searched in order.

    Returns:
        First image attached to an Image Texture node, or None. Nodes inside
        node groups are searched too, after the material's top-level nodes.
    """
    for material in mesh_obj.data.materials:
        if material is None or not material.use_nodes:
            continue
        image = _node_tree_image(material.node_tree)
        if image is not None:
            return image
    return None


def _node_tree_image(node_tree: bpy.types.NodeTree | None) -> bpy.types.Image | None:
    """Return the first image in ``node_tree``, then in its node groups.

    Blender forbids a group from containing itself, so the recursion ends.
    """
    if node_tree is None:
        return None
    for node in node_tree.nodes:
        if node.type == "TEX_IMAGE" and node.image is not None:
            return node.image
    groups = (_node_tree_image(node.node_tree) for node in node_tree.nodes if node.type == "GROUP")
    return next((image for image in groups if image is not None), None)


def mesh_uvs_and_texture(
    mesh_obj: bpy.types.Object, usd_path: str
) -> tuple[list[tuple[float, ...]] | None, str | None]:
    """Return ``(uvs, texture_file)`` for ``write_mesh_usda``; both None for untextured meshes.

    Args:
        mesh_obj: Blender mesh with an optional active UV layer and image material.
        usd_path: Destination USDA filename used to locate the adjacent texture.

    Returns:
        Face-corner UV pairs and relative texture filename, each optional.

    The texture is copied to ``dirname(usd_path)/<USD stem>.texture.<extension>`` and
    returned as that bare file name. A material texture without a UV layer is reported and skipped.
    """
    mesh = mesh_obj.data
    uv_layer = mesh.uv_layers.active
    uvs = None
    if uv_layer is not None:
        uvs = [
            tuple(float(c) for c in uv_layer.data[loop].uv)
            for p in mesh.polygons
            for loop in p.loop_indices
        ]
    image = image_texture(mesh_obj)
    if image is None:
        return uvs, None
    if uvs is None:
        LOGGER.warning("Material texture %r has no UV layer; exporting untextured", image.name)
        return None, None
    usd_dir = os.path.dirname(os.path.abspath(usd_path)) or "."
    os.makedirs(usd_dir, exist_ok=True)
    source = bpy.path.abspath(image.filepath) if image.filepath else ""
    extension = os.path.splitext(source)[1] if source else ".png"
    extension = extension or ".png"
    name = os.path.splitext(os.path.basename(usd_path))[0] + ".texture" + extension
    dest = os.path.join(usd_dir, name)
    if image.packed_file is not None or image.is_dirty:
        # The pixels in the blend are authoritative over an older external file.
        image.save(filepath=dest)
    elif source and os.path.isfile(source):
        if os.path.abspath(source) != os.path.abspath(dest):
            shutil.copy(source, dest)
    else:
        image.save(filepath=dest)  # packed or generated image
    LOGGER.info("Texture: %s -> %s", name, dest)
    return uvs, name
