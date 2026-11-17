# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in, deterministic metallic end-face and cast-shadow stress scenes.

Blender implementation only. Values are recorded explicitly; absent configuration
leaves existing scene records byte-identical. No change to object coordinates.

Sampling is split in two phases so robustness clipping can move the camera in
between: ``begin_appearance`` places the optional end-face camera, and
``finish_appearance`` aims the lighting from the final camera.

Every random draw comes from one per-index stream, in a fixed order. Reordering
draws changes every record for a given seed, so keep the order when editing.
"""

import copy
import math
import random
from collections.abc import Mapping
from typing import Any

import numpy as np

from doper.scene import geometry as G  # noqa: N812 - short alias used across the sampling code

AXES = {"x": 0, "y": 1, "z": 2}
PRESETS = ("mixed", "glare", "shadow", "diffuse")
CONCRETE_PRESETS = ["glare", "shadow", "diffuse"]
DEFAULT_END_VIEW_PROBABILITY = 0.45
# End-face cameras sit in a cone around the axis, at this angle range in degrees.
END_VIEW_CONE_DEG = (5, 28)
# Minimum upward component of the end-view direction, keeping the camera above the floor.
END_VIEW_MIN_ELEVATION = 0.12
# Neutral floor added when the sampled scene has none, so cast shadows land somewhere.
SHADOW_FLOOR = {"enabled": True, "kind": "color", "color": [0.35, 0.35, 0.35], "roughness": 0.7}
APPEARANCE_RECORD_VERSION = 2


def validate_robust_appearance(raw: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Normalize the YAML section; ``None`` means disabled.

    ``axis`` names the object-local symmetry axis whose end faces are stressed.
    It is required because the sampler and renderer cannot infer it from a mesh.
    """
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("robust_appearance must be a mapping")
    allowed = {"enabled", "end_view_probability", "preset", "axis"}
    if set(raw) - allowed:
        raise ValueError(f"unknown robust_appearance keys: {sorted(set(raw) - allowed)}")
    if not raw.get("enabled", False):
        return None
    probability = float(raw.get("end_view_probability", DEFAULT_END_VIEW_PROBABILITY))
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("robust_appearance.end_view_probability must be finite in [0,1]")
    preset = raw.get("preset", "mixed")
    if preset not in PRESETS:
        raise ValueError("robust_appearance.preset must be mixed/glare/shadow/diffuse")
    axis = raw.get("axis")
    if axis not in AXES:
        raise ValueError(
            "robust_appearance.axis must name the object-local symmetry axis: x, y or z"
        )
    return {"enabled": True, "end_view_probability": probability, "preset": preset, "axis": axis}


def begin_appearance(
    record: dict[str, Any], spec: Mapping[str, Any] | None, rng: random.Random
) -> dict[str, Any] | None:
    """Choose the preset and optional end-face camera; return state for ``finish_appearance``.

    ``rng`` is the sample's independent ``metal-stress`` stream, so enabling this
    leaves other fields unchanged. Returns ``None`` when ``spec`` is ``None``.
    """
    if spec is None:
        return None
    preset = spec["preset"]
    if preset == "mixed":
        preset = rng.choice(CONCRETE_PRESETS)
    axis = upward_object_axis(record, spec["axis"])
    end_view = rng.random() < spec["end_view_probability"]
    if end_view:
        place_end_view_camera(record, axis, rng)
    # Robustness clipping checks ground clearance, so the floor must exist first.
    if not record["ground"].get("enabled"):
        record["ground"] = copy.deepcopy(SHADOW_FLOOR)
    return {
        "rng": rng,
        "preset": preset,
        "end_view": end_view,
        "axis": spec["axis"],
        "axis_world": axis,
    }


def upward_object_axis(record: Mapping[str, Any], axis_name: str) -> np.ndarray:
    """World direction of the configured object axis, flipped to point upward."""
    axis = np.asarray(record["object"]["pose"], float)[:3, AXES[axis_name]]
    axis = axis if axis[2] >= 0 else -axis
    return axis / np.linalg.norm(axis)


def place_end_view_camera(record: dict[str, Any], axis: np.ndarray, rng: random.Random) -> None:
    """Move the camera into a cone around the upper end of ``axis``.

    The camera keeps its sampled distance and looks at the object center. The
    direction is lifted to a minimum elevation, so the final angle to the axis
    can exceed the cone when the object is strongly tilted.
    """
    tangent = np.cross(axis, [0, 0, 1])
    if np.linalg.norm(tangent) < 1e-6:
        tangent = np.cross(axis, [1, 0, 0])
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(axis, tangent)
    cone_angle = math.radians(rng.uniform(*END_VIEW_CONE_DEG))
    azimuth = rng.uniform(0, 2 * math.pi)
    direction = math.cos(cone_angle) * axis + math.sin(cone_angle) * (
        math.cos(azimuth) * tangent + math.sin(azimuth) * bitangent
    )
    direction[2] = max(END_VIEW_MIN_ELEVATION, direction[2])
    direction /= np.linalg.norm(direction)
    center = np.array(record["object"]["center"], float)
    position = center + float(record["camera"]["distance"]) * direction
    record["camera"]["cam_to_world"] = G.look_at_camera(position, center).tolist()


def finish_appearance(record: dict[str, Any], state: Mapping[str, Any] | None) -> None:
    """Sample lighting and material relative to the final camera and record them.

    Replaces the sampled scene lights with one sun. ``end_face_view`` records the
    requested sample; ``final_axis_angle_deg`` is measured from the final camera,
    after any robustness clipping translation.
    """
    if state is None:
        return
    rng = state["rng"]
    view = final_view_direction(record)
    axis_angle = math.degrees(math.acos(min(1.0, abs(float(view @ state["axis_world"])))))
    lighting = PRESET_SAMPLERS[state["preset"]](rng, view)
    sun_direction = lighting.pop("sun_direction")
    sun_direction /= np.linalg.norm(sun_direction)
    record["lights"] = []
    # Remaining draws, in this order: gray, brush scale, cap roughness factor.
    gray = rng.uniform(0.55, 0.82)
    brush_scale = rng.uniform(120.0, 250.0)
    cap_roughness = min(0.7, lighting["roughness"] * rng.uniform(1.0, 2.0))
    record["appearance"] = {
        "version": APPEARANCE_RECORD_VERSION,
        "kind": "metal_endface",
        "preset": state["preset"],
        "axis": state["axis"],
        "end_face_view": state["end_view"],
        "final_axis_angle_deg": axis_angle,
        "metallic": 1.0,
        "roughness": lighting["roughness"],
        "gray": gray,
        "world_strength": lighting["world_strength"],
        "exposure": lighting["exposure"],
        "sun_direction": sun_direction.tolist(),
        "sun_energy": lighting["sun_energy"],
        "sun_angle": lighting["sun_angle"],
        "brush_scale": brush_scale,
        "cap_roughness": cap_roughness,
    }


def final_view_direction(record: Mapping[str, Any]) -> np.ndarray:
    """Unit vector from the object center to the final camera position."""
    camera = np.array(record["camera"]["cam_to_world"], float)
    direction = camera[:3, 3] - np.array(record["object"]["center"], float)
    return direction / np.linalg.norm(direction)


def sample_glare(rng: random.Random, view: np.ndarray) -> dict[str, Any]:
    """Sun close to the camera, glossy metal and bright exposure: specular glare."""
    lift = np.array([rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25), 0.2])
    return {
        "sun_direction": view + lift,
        "roughness": rng.uniform(0.025, 0.13),
        "sun_energy": rng.uniform(3.0, 6.0),
        "world_strength": rng.uniform(0.25, 0.75),
        "exposure": rng.uniform(-0.4, 0.8),
        "sun_angle": rng.uniform(0.01, 0.10),
    }


def sample_shadow(rng: random.Random, view: np.ndarray) -> dict[str, Any]:
    """Low, hard sun with a dim world: long crisp cast shadows."""
    azimuth = rng.uniform(0, 2 * math.pi)
    return {
        "sun_direction": np.array([math.cos(azimuth), math.sin(azimuth), rng.uniform(0.25, 0.7)]),
        "roughness": rng.uniform(0.15, 0.45),
        "sun_energy": rng.uniform(2.0, 5.0),
        "world_strength": rng.uniform(0.04, 0.25),
        "exposure": rng.uniform(-0.4, 0.2),
        "sun_angle": rng.uniform(0.002, 0.02),
    }


def sample_diffuse(rng: random.Random, view: np.ndarray) -> dict[str, Any]:
    """High, soft sun with a bright world and rough metal: low contrast."""
    return {
        "sun_direction": np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), 1.0]),
        "roughness": rng.uniform(0.25, 0.65),
        "sun_energy": rng.uniform(0.5, 2.0),
        "world_strength": rng.uniform(0.5, 1.2),
        "exposure": rng.uniform(-0.5, 0.4),
        "sun_angle": rng.uniform(0.12, 0.3),
    }


PRESET_SAMPLERS = {"glare": sample_glare, "shadow": sample_shadow, "diffuse": sample_diffuse}
