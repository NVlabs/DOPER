# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in metallic appearance sampling: determinism and camera placement.

Configuration validation is covered in ``test_render_robustness``.
"""

import copy
import json

import numpy as np
import pytest

from rendering.robust_appearance import (
    begin_appearance,
    finish_appearance,
    validate_robust_appearance,
)
from rendering.scene_spec import stream


def record():
    """Minimal scene record with an identity object pose and no ground."""
    return {
        "object": {"pose": np.eye(4).tolist(), "center": [0, 0, 0.1]},
        "camera": {"cam_to_world": np.eye(4).tolist(), "distance": 0.4},
        "material": {},
        "lights": [],
        "background": {},
        "ground": {"enabled": False},
    }


def sample(spec, index=2):
    """Run both appearance phases on a fresh record, with no clipping in between."""
    result = record()
    state = begin_appearance(
        result, validate_robust_appearance(spec), stream(42, "metal-stress", index)
    )
    finish_appearance(result, state)
    return result


def view_direction(result):
    camera = np.asarray(result["camera"]["cam_to_world"])
    direction = camera[:3, 3] - result["object"]["center"]
    return camera, direction / np.linalg.norm(direction)


@pytest.mark.parametrize(
    "raw", [None, {}, {"enabled": False}, {"enabled": False, "preset": "glare"}]
)
def test_absent_or_disabled_appearance_does_not_change_record(raw):
    assert validate_robust_appearance(raw) is None
    result = record()
    before = copy.deepcopy(result)
    finish_appearance(result, begin_appearance(result, None, stream(42, "metal-stress", 2)))
    assert result == before


def test_deterministic_endface_and_shadow_geometry():
    spec = {"enabled": True, "end_view_probability": 1.0, "axis": "y"}
    result = sample(spec)
    assert result == sample(spec)
    appearance = result["appearance"]
    assert appearance["end_face_view"] and appearance["axis"] == "y"
    camera, view = view_direction(result)
    assert abs(view[1]) > 0.8 and view[2] > 0
    assert np.allclose(camera[:3, :3].T @ camera[:3, :3], np.eye(3))
    assert np.allclose(-camera[:3, 2], -view), "camera must look at the object center"
    assert 5 - 1e-6 <= appearance["final_axis_angle_deg"] <= 45
    assert result["ground"]["enabled"]
    assert result["ground"]["kind"] in ("color", "texture")
    assert 0.02 <= appearance["roughness"] <= 0.65
    assert appearance["metallic"] == 1
    json.dumps(result, allow_nan=False)


def test_end_face_camera_follows_configured_axis():
    result = sample({"enabled": True, "end_view_probability": 1.0, "axis": "z"})
    _, view = view_direction(result)
    assert view[2] > 0.8 and result["appearance"]["axis"] == "z"
