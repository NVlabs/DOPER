# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spatial contracts use independent Blender-plane landmarks."""

import numpy as np
import pytest

from rendering.config import render_config_from_mapping
from rendering.isaac.background_mapping import blender_dome_rotation, blender_ground_uv
from rendering.isaac.settings import IsaacSettings


@pytest.mark.parametrize("scene_scale", [0.01, 2.0, 17.0])
def test_ground_landmarks_match_blender_mapping_on_larger_isaac_plane(scene_scale):
    # Blender's +/-2 plane corners have UVs 0..1; scale=6 gives 2 repeats.
    points = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1), (0, 0), (-3, -3)]) * scene_scale
    expected = [(0, 0), (0, 2), (-2, 2), (-2, 0), (-1, 1), (2, -2)]
    np.testing.assert_allclose(blender_ground_uv(points, scene_scale, 6, 90), expected, atol=1e-12)


@pytest.mark.parametrize("angle,expected", [(0, 180), (90, 90), (180, 0), (270, -90)])
def test_hdri_lookup_rotation_is_inverse_dome_rotation(angle, expected):
    assert blender_dome_rotation(angle) == expected


def test_legacy_spatial_controls_are_opt_in_and_recorded(tmp_path):
    def settings(raw):
        return IsaacSettings.from_config(render_config_from_mapping({"isaac": raw}, str(tmp_path)))

    baseline = settings({"profile": "legacy"})
    assert baseline.match_blender_hdri is False
    assert baseline.match_blender_ground_uv is False
    assert "match_blender_hdri" not in baseline.to_dict()
    assert "match_blender_ground_uv" not in baseline.to_dict()
    for key in ("match_blender_hdri", "match_blender_ground_uv"):
        assert settings({key: True}).to_dict()[key] is True
        for bad in (1, "true", None):
            with pytest.raises(ValueError, match=key):
                settings({key: bad})
