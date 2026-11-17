# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isaac defaults shared by config loading and renderer settings validation."""

from collections.abc import Mapping
from typing import Any

BLENDER_MATCH = {
    "spp": 32,
    "tonemap_op": "aces",
    "exposure_time": 0.0075,
    "film_iso": 100.0,
    "f_number": 5.0,
    "auto_exposure": False,
    "denoiser": True,
    "match_blender_hdri": True,
    "match_blender_ground_uv": True,
}


def resolve_settings(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve defaults without changing the source recipe or Blender settings.

    Explicit values (including null photometric controls) override the profile.
    The legacy profile retains the previous renderer-controlled defaults.
    """
    supplied = raw.get("isaac")
    if supplied is None:
        supplied = {}
    if not isinstance(supplied, Mapping):
        raise ValueError("isaac settings must be a mapping")
    profile = supplied.get("profile", "blender_match")
    if profile not in ("blender_match", "legacy"):
        raise ValueError("isaac.profile must be blender_match or legacy")
    if profile == "legacy":
        defaults = {"spp": (raw.get("render") or {}).get("samples", 64)}
    else:
        defaults = BLENDER_MATCH
    return {**defaults, **{key: value for key, value in supplied.items() if key != "profile"}}
