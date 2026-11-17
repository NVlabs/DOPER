# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned opt-in environment recipe; safe to validate before Kit startup."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

PROFILE_NAME = "blender42_v1"


def _number(
    value: Any,  # noqa: ANN401 - YAML scalar
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if (
        not math.isfinite(value)
        or (minimum is not None and value < minimum)
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"{name} is outside the supported range")
    return value


def _vector(
    value: Any,  # noqa: ANN401 - YAML sequence
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain three numbers")
    x, y, z = (_number(item, name, minimum, maximum) for item in value)
    return x, y, z


@dataclass(frozen=True)
class SunLight:
    """Explicit asset light, using Blender energy and angular-diameter inputs."""

    energy: float
    rotation_degrees: tuple[float, float, float]
    angle_degrees: float
    color: tuple[float, float, float]

    @classmethod
    def from_mapping(cls, raw: Any) -> SunLight:  # noqa: ANN401 - YAML value
        """Validate a ``sun`` mapping with exactly the four light fields.

        Raises:
            ValueError: A field is missing, extra, or out of range.
        """
        if not isinstance(raw, Mapping) or set(raw) != {
            "energy",
            "rotation_degrees",
            "angle_degrees",
            "color",
        }:
            raise ValueError("sun requires energy, rotation_degrees, angle_degrees and color")
        return cls(
            _number(raw["energy"], "sun.energy", 0),
            _vector(raw["rotation_degrees"], "sun.rotation_degrees"),
            _number(raw["angle_degrees"], "sun.angle_degrees", 0, 180),
            _vector(raw["color"], "sun.color", 0, 1),
        )


@dataclass(frozen=True)
class EnvironmentProfile:
    """The ``blender42_v1`` portable environment: light gain and an optional sun."""

    name: str
    light_gain: float
    sun: SunLight | None

    @classmethod
    def from_mapping(cls, raw: Any) -> EnvironmentProfile:  # noqa: ANN401 - YAML value
        """Validate an ``environment_profile`` mapping; ``sun`` must be given (or null).

        Raises:
            ValueError: The profile is unknown, incomplete, or out of range.
        """
        if not isinstance(raw, Mapping) or set(raw) != {"name", "light_gain", "sun"}:
            raise ValueError(
                "environment_profile requires name, light_gain and explicit sun (or null)"
            )
        if raw["name"] != PROFILE_NAME:
            raise ValueError(f"environment_profile.name must be {PROFILE_NAME}")
        gain = _number(raw["light_gain"], "environment_profile.light_gain", 0)
        if gain == 0:
            raise ValueError("environment_profile.light_gain must be positive")
        return cls(
            raw["name"],
            gain,
            None if raw["sun"] is None else SunLight.from_mapping(raw["sun"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON form recorded in the renderer settings (tuples become lists)."""
        result = asdict(self)
        if self.sun is not None:
            result["sun"]["rotation_degrees"] = list(self.sun.rotation_degrees)
            result["sun"]["color"] = list(self.sun.color)
        return result
