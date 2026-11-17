# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in environment settings must fail closed and preserve old identities."""

import copy

import pytest

from rendering.config import render_config_from_mapping
from rendering.isaac.settings import IsaacSettings

PROFILE = {"name": "blender42_v1", "light_gain": 2000.0, "sun": None}
SUN = {
    "energy": 2.0,
    "rotation_degrees": [45.0, 30.0, 0.0],
    "angle_degrees": 0.526,
    "color": [1.0, 1.0, 1.0],
}


def settings(profile=None, **isaac):
    if profile is not None:
        isaac["environment_profile"] = profile
    cfg = render_config_from_mapping(
        {"scene": {"world_background_modes": "hdri"}, "isaac": isaac}, "."
    )
    return IsaacSettings.from_config(cfg)


def test_default_and_legacy_identity_do_not_gain_environment_fields():
    legacy = render_config_from_mapping({"isaac": {"profile": "legacy"}}, ".")
    for value in (settings(), IsaacSettings.from_config(legacy)):
        assert "environment_profile" not in value.to_dict()


def test_profile_round_trip_and_explicit_absent_sun():
    value = settings(PROFILE)
    assert value.to_dict()["environment_profile"] == PROFILE
    assert value.environment_profile is not None
    assert value.environment_profile.sun is None
    assert value.environment_profile.name == "blender42_v1"


@pytest.mark.parametrize(
    "profile",
    [
        {},
        False,
        "blender42_v1",
        {"name": "blender42_v1"},
        {**PROFILE, "name": "unknown"},
        {**PROFILE, "extra": 3},
        {**PROFILE, "light_gain": 0},
        {**PROFILE, "light_gain": True},
        {**PROFILE, "light_gain": float("nan")},
        {**PROFILE, "sun": {}},
        {**PROFILE, "sun": False},
    ],
)
def test_invalid_or_incomplete_profiles_are_rejected(profile):
    with pytest.raises(ValueError):
        settings(profile)


def test_explicit_sun_is_immutable_and_part_of_identity():
    raw = {**PROFILE, "sun": copy.deepcopy(SUN)}
    value = settings(raw)
    original = copy.deepcopy(value.to_dict())
    raw["sun"]["color"][0] = 0.2
    assert value.to_dict() == original
    assert settings(raw).to_dict() != original
    assert value.environment_profile is not None
    assert value.environment_profile.sun is not None
    assert value.environment_profile.sun.color == (1.0, 1.0, 1.0)


def test_non_hdri_profile_and_native_replay_conflict_fail_early():
    cfg = render_config_from_mapping({"isaac": {"environment_profile": PROFILE}}, ".")
    with pytest.raises(ValueError, match="HDRI"):
        IsaacSettings.from_config(cfg)
    with pytest.raises(ValueError, match="native replay"):
        settings(PROFILE).validate_environment("/bundle")
    settings(PROFILE).validate_environment(None)
    settings().validate_environment("/bundle")


@pytest.mark.parametrize("key", ["match_blender_hdri", "match_blender_ground_uv"])
def test_profile_rejects_disabled_spatial_corrections(key):
    with pytest.raises(ValueError, match="corrections"):
        settings(PROFILE, **{key: False})


@pytest.mark.parametrize(
    "field,value",
    [
        ("energy", -1),
        ("energy", True),
        ("rotation_degrees", [0, 0]),
        ("rotation_degrees", [0, float("inf"), 0]),
        ("angle_degrees", 181),
        ("color", [1, 1, 2]),
        ("color", [True, 1, 1]),
    ],
)
def test_invalid_sun_properties_fail_closed(field, value):
    sun = copy.deepcopy(SUN)
    sun[field] = value
    with pytest.raises(ValueError):
        settings({**PROFILE, "sun": sun})


def test_gain_change_changes_serialized_settings():
    assert settings(PROFILE).to_dict() != settings({**PROFILE, "light_gain": 1000.0}).to_dict()
