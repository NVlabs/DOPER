# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated Isaac backend controls; importing this module never starts Kit."""

# pyright: reportMissingImports=false, reportMissingModuleSource=false

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from rendering.config import RenderConfig
from rendering.isaac.defaults import resolve_settings
from rendering.isaac.environment_profile import EnvironmentProfile

# Verified against omni.rtx.settings.core 0.7.1 (Isaac Sim 6.0.1).
TONEMAP_OPERATORS = {
    "clamp": 0,
    "linear": 1,
    "reinhard": 2,
    "modified_reinhard": 3,
    "hejl_hable": 4,
    "hable": 5,
    "aces": 6,
    "iray": 7,
}
# USD tokens of the render product's tonemap operator, indexed like TONEMAP_OPERATORS.
TONEMAP_TOKENS = (
    "raw",
    "none",
    "reinhard",
    "modifiedReinhard",
    "hejiHableAlu",
    "hableUc2",
    "acesApproximation",
    "iray",
)
PHOTOMETRIC_PATHS = {
    "exposure_time": "/rtx/post/tonemap/exposureTime",
    "film_iso": "/rtx/post/tonemap/filmIso",
    "f_number": "/rtx/post/tonemap/fNumber",
    "auto_exposure": "/rtx/post/histogram/enabled",
    "denoiser": "/rtx/pathtracing/optixDenoiser/enabled",
}
SPATIAL_KEYS = ("match_blender_hdri", "match_blender_ground_uv")
BASE_KEYS = {
    "render_mode",
    "spp",
    "occlusion_tol_m",
    "bbox_source",
    "light_intensity_range",
    "dome_intensity_range",
    "capture_timeout_sec",
}
ALLOWED_KEYS = BASE_KEYS | {"tonemap_op", "environment_profile", *PHOTOMETRIC_PATHS, *SPATIAL_KEYS}
DEFAULT_CAPTURE_TIMEOUT_SEC = 900.0
RAYTRACED_SUBFRAMES = 8


def _photometric(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the opt-in tonemap, exposure and denoiser controls that are not null."""
    photometric = {
        key: raw[key] for key in ("tonemap_op", *PHOTOMETRIC_PATHS) if raw.get(key) is not None
    }
    if "tonemap_op" in photometric and (
        not isinstance(photometric["tonemap_op"], str)
        or photometric["tonemap_op"] not in TONEMAP_OPERATORS
    ):
        raise ValueError("isaac.tonemap_op must name a supported operator")
    for key in ("exposure_time", "film_iso", "f_number"):
        if key in photometric:
            value = photometric[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"isaac.{key} must be finite and positive")
            photometric[key] = float(value)
    for key in ("auto_exposure", "denoiser"):
        if key in photometric and not isinstance(photometric[key], bool):
            raise ValueError(f"isaac.{key} must be boolean")
    return photometric


def _check_ranges(settings: IsaacSettings, raw: Mapping[str, Any]) -> None:
    """Reject unsupported modes and nonfinite, negative or unordered numeric settings."""
    if settings.render_mode not in {"PathTracing", "RayTracedLighting"}:
        raise ValueError("unsupported isaac.render_mode")
    if settings.bbox_source not in {"keypoints", "mask"}:
        raise ValueError("isaac.bbox_source must be keypoints or mask")
    if not math.isfinite(settings.occlusion_tol_m) or settings.occlusion_tol_m < 0:
        raise ValueError("isaac.occlusion_tol_m must be finite and nonnegative")
    timeout = settings.capture_timeout_sec
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("isaac.capture_timeout_sec must be finite and positive")
    for name in ("light_intensity_range", "dome_intensity_range"):
        supplied = raw.get(name)
        if (
            isinstance(supplied, (tuple, list))
            and len(supplied) == 2
            and float(supplied[0]) > float(supplied[1])
        ):
            raise ValueError(f"isaac.{name} must be ordered")
        values = getattr(settings, name)
        if (
            len(values) != 2
            or not all(math.isfinite(v) and v >= 0 for v in values)
            or values[0] > values[1]
        ):
            raise ValueError(f"isaac.{name} must be an ordered nonnegative finite pair")


@dataclass(frozen=True)
class IsaacSettings:
    """Requested Isaac renderer settings, recorded in the render identity."""

    render_mode: str
    spp: int
    occlusion_tol_m: float
    bbox_source: str
    light_intensity_range: tuple[float, float]
    dome_intensity_range: tuple[float, float]
    capture_timeout_sec: float = DEFAULT_CAPTURE_TIMEOUT_SEC
    tonemap_op: str | None = None
    exposure_time: float | None = None
    film_iso: float | None = None
    f_number: float | None = None
    auto_exposure: bool | None = None
    denoiser: bool | None = None
    match_blender_hdri: bool = False
    match_blender_ground_uv: bool = False
    environment_profile: EnvironmentProfile | None = None

    @classmethod
    def from_config(cls, cfg: RenderConfig) -> IsaacSettings:
        """Validate the config's ``isaac`` section (with profile defaults resolved).

        Raises:
            ValueError: A setting is unknown, of the wrong type, or out of range.
        """
        raw = resolve_settings(cfg.raw)
        unknown = set(raw) - ALLOWED_KEYS
        if unknown:
            raise ValueError(f"unknown isaac setting(s): {', '.join(sorted(unknown))}")
        spp = raw["spp"]
        if isinstance(spp, bool) or not isinstance(spp, int) or spp <= 0:
            raise ValueError("isaac.spp must be a positive integer")
        photometric = _photometric(raw)
        spatial = {key: raw.get(key, False) for key in SPATIAL_KEYS}
        for key, value in spatial.items():
            if not isinstance(value, bool):
                raise ValueError(f"isaac.{key} must be boolean")
        profile = raw.get("environment_profile")
        if profile is not None:
            profile = EnvironmentProfile.from_mapping(profile)
            if set(cfg.world_background_modes) != {"hdri"}:
                raise ValueError("blender42_v1 requires HDRI-only scene backgrounds")
            if not all(spatial.values()):
                raise ValueError("blender42_v1 requires Blender HDRI and ground UV corrections")
        light_low, light_high = cfg.light_intensity
        dome_low, dome_high = cfg.dome_intensity
        result = cls(
            cfg.render_mode,
            spp,
            cfg.occlusion_tol_m,
            cfg.bbox_source,
            (light_low, light_high),
            (dome_low, dome_high),
            float(raw.get("capture_timeout_sec", DEFAULT_CAPTURE_TIMEOUT_SEC)),
            environment_profile=profile,
            **photometric,
            **spatial,
        )
        _check_ranges(result, raw)
        return result

    def validate_environment(self, native_root: str | None) -> None:
        """Reject a portable environment profile combined with native replay."""
        if self.environment_profile is not None and native_root:
            raise ValueError("environment_profile cannot be combined with native replay")

    def to_dict(self) -> dict[str, Any]:
        """JSON form recorded as the requested settings; unset controls are omitted."""
        values = asdict(self)
        if self.environment_profile is not None:
            values["environment_profile"] = self.environment_profile.to_dict()
        # Keep the settings of renders without these corrections unchanged.
        for name in SPATIAL_KEYS:
            if not values[name]:
                del values[name]
        for name in ("light_intensity_range", "dome_intensity_range"):
            values[name] = list(values[name])
        return {key: value for key, value in values.items() if value is not None}

    def app_config(self, resolution: int, gpu_device: int) -> dict[str, Any]:
        """``SimulationApp`` config whose stage-reset defaults match the requested settings."""
        return {
            "headless": True,
            "width": resolution,
            "height": resolution,
            "active_gpu": gpu_device,
            "physics_gpu": gpu_device,
            "multi_gpu": False,
            "renderer": self.carb_settings()["/rtx/rendermode"],
            "samples_per_pixel_per_frame": self.spp,
        }

    def carb_settings(self) -> dict[str, Any]:
        """Carb setting paths and the values the renderer must report back."""
        subframes = self.spp if self.render_mode == "PathTracing" else RAYTRACED_SUBFRAMES
        values: dict[str, Any] = {
            "/rtx/rendermode": self.render_mode.replace("RayTraced", "Raytraced"),
            "/rtx/pathtracing/spp": self.spp,
            "/rtx/pathtracing/totalSpp": self.spp,
            "/rtx/pathtracing/clampSpp": self.spp,
            "/omni/replicator/RTSubframes": subframes,
        }
        if self.tonemap_op is not None:
            values["/rtx/post/tonemap/op"] = TONEMAP_OPERATORS[self.tonemap_op]
        for name, path in PHOTOMETRIC_PATHS.items():
            value = getattr(self, name)
            if value is not None:
                values[path] = value
        return values

    @property
    def has_photometric_controls(self) -> bool:
        """Whether any tonemap, exposure or denoiser control is set."""
        return self.tonemap_op is not None or any(
            getattr(self, name) is not None for name in PHOTOMETRIC_PATHS
        )

    def apply_to_stage(
        self,
        camera_prim: Any,  # noqa: ANN401 - Usd.Prim
        product_prim: Any,  # noqa: ANN401 - Usd.Prim
    ) -> None:
        """Author opted-in USD controls before Hydra restores camera/product state.

        Isaac 6.0.1 treats these USD attributes as authoritative, so changing only their
        deprecated Carb aliases can be undone on the first capture.

        Raises:
            RuntimeError: An attribute could not be authored.
        """
        # Launcher configuration is also imported on hosts without USD bindings.
        from pxr import Sdf as _Sdf

        sdf: Any = _Sdf  # usd-core ships no type information.

        camera_values = {
            "exposure:time": (self.exposure_time, sdf.ValueTypeNames.Float),
            "exposure:iso": (self.film_iso, sdf.ValueTypeNames.Float),
            "exposure:fStop": (self.f_number, sdf.ValueTypeNames.Float),
            "omni:rtx:autoExposure:enabled": (self.auto_exposure, sdf.ValueTypeNames.Bool),
        }
        operator = TONEMAP_TOKENS[TONEMAP_OPERATORS[self.tonemap_op]] if self.tonemap_op else None
        product_values = {
            "omni:rtx:post:tonemap:operator": (operator, sdf.ValueTypeNames.Token),
            "omni:rtx:pt:denoising:enabled": (self.denoiser, sdf.ValueTypeNames.Bool),
        }
        for prim, values in ((camera_prim, camera_values), (product_prim, product_values)):
            for name, (value, value_type) in values.items():
                if value is None:
                    continue
                if not prim or not prim.CreateAttribute(name, value_type, custom=False).Set(value):
                    raise RuntimeError(f"could not author Isaac setting {name}")

    def verify_effective(self, getter: Callable[[str], Any]) -> dict[str, Any]:
        """Read back every Carb setting through ``getter`` and require the requested values.

        Returns:
            The effective values.

        Raises:
            RuntimeError: A value differs beyond float32 readback rounding.
        """
        expected = self.carb_settings()
        effective = {key: getter(key) for key in expected}
        if not effective_settings_match(expected, effective):
            raise RuntimeError(f"Isaac renderer settings did not apply: {effective}")
        return effective


def effective_settings_match(expected: Mapping[str, Any], actual: Any) -> bool:  # noqa: ANN401
    """Allow only float32 readback rounding; keys and nonfloats stay exact."""
    if not isinstance(actual, dict) or actual.keys() != expected.keys():
        return False
    for key, value in expected.items():
        readback = actual[key]
        if isinstance(value, float):
            if (
                not isinstance(readback, (float, int))
                or isinstance(readback, bool)
                or not math.isclose(readback, value, rel_tol=1e-6, abs_tol=0)
            ):
                return False
        elif type(readback) is not type(value) or readback != value:
            return False
    return True
