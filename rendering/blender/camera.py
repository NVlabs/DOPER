# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Camera and pose sampling with Blender coordinate conventions."""

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import math

from mathutils import Euler, Vector
from mathutils import Quaternion as MQuaternion

from .context import BlenderContext, BpyObject


def set_if_present(obj: BpyObject, attr: BpyObject, value: BpyObject) -> None:
    """Set ``obj.attr`` to ``value`` when the attribute exists on this Blender version."""
    if hasattr(obj, attr):
        setattr(obj, attr, value)


def resolve_render_engine(scene: BpyObject, requested: BpyObject) -> BpyObject:
    """Return the requested render engine, mapping EEVEE names across Blender versions.

    ``CYCLES`` is accepted although the engine enum a background session reports lists only the
    built-in EEVEE; its add-on registers it when the engine is assigned.
    """
    if requested == "CYCLES":
        return requested
    available = {item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items}
    if requested in available:
        return requested
    if requested == "BLENDER_EEVEE" and "BLENDER_EEVEE_NEXT" in available:
        return "BLENDER_EEVEE_NEXT"
    if requested == "BLENDER_EEVEE_NEXT" and "BLENDER_EEVEE" in available:
        return "BLENDER_EEVEE"
    raise ValueError(f"Render engine {requested!r} not available; choices are {sorted(available)}")


def z_facing_quaternion_with_jiggle(ctx: BlenderContext, jiggle_degrees: float = 5.0) -> BpyObject:
    """Align the object's local +Z axis with world +Z (face up/down)."""
    z_rotation = ctx.rng.uniform(0, 2 * math.pi)
    jiggle_rad = math.radians(jiggle_degrees)
    tilt_x = ctx.rng.uniform(-jiggle_rad, jiggle_rad)
    tilt_y = ctx.rng.uniform(-jiggle_rad, jiggle_rad)

    if ctx.rng.random() < 0.5:
        base_euler = Euler((tilt_x, tilt_y, z_rotation), "XYZ")
    else:
        base_euler = Euler((math.pi + tilt_x, tilt_y, z_rotation), "XYZ")

    return base_euler.to_quaternion()


def axis_facing_quaternion_with_jiggle(
    ctx: BlenderContext, axis: BpyObject, jiggle_degrees: float = 5.0
) -> BpyObject:
    """Align the object's local ``axis`` (+x / +y / +z) with world +Z.

    Adds: (1) a random spin around the now-vertical axis, (2) a small
    uniform tilt perpendicular to it, and (3) a 50% chance of flipping so
    both ±axis orientations are covered.
    """
    base_q = z_facing_quaternion_with_jiggle(ctx, jiggle_degrees)
    if axis == "z":
        return base_q
    # Rotate the orientation so that the chosen local axis ends up pointing
    # where local +Z was (i.e. toward world +Z after the jiggle).
    if axis == "x":
        # Send local +X to local +Z: rotate -90 deg about local Y.
        pre = MQuaternion(Euler((0, -math.pi / 2, 0), "XYZ").to_quaternion())
    elif axis == "y":
        # Send local +Y to local +Z: rotate +90 deg about local X.
        pre = MQuaternion(Euler((math.pi / 2, 0, 0), "XYZ").to_quaternion())
    else:
        raise ValueError(f"unknown axis {axis!r}")
    return base_q @ pre


def get_camera_direction_top_down(ctx: BlenderContext, jiggle_degrees: float = 5.0) -> BpyObject:
    """Sample a camera direction around +Z, tilted by up to ``jiggle_degrees``."""
    cam_dir = Vector((0, 0, 1))
    if jiggle_degrees > 0:
        jiggle_rad = math.radians(jiggle_degrees)
        tilt = ctx.rng.uniform(0, jiggle_rad)
        azimuth = ctx.rng.uniform(0, 2 * math.pi)
        cam_dir = Vector(
            (
                math.sin(tilt) * math.cos(azimuth),
                math.sin(tilt) * math.sin(azimuth),
                math.cos(tilt),
            )
        )
    return cam_dir.normalized()


def random_camera_direction(ctx: BlenderContext) -> BpyObject:
    """Sample a random camera direction on the upper hemisphere."""
    phi = ctx.rng.uniform(0, 2 * math.pi)
    cos_theta = ctx.rng.uniform(0.1, 1.0)
    sin_theta = math.sqrt(1 - cos_theta**2)
    return Vector((sin_theta * math.cos(phi), sin_theta * math.sin(phi), cos_theta))
