# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isaac settings fail early and verify what the renderer actually applied."""

from types import SimpleNamespace

import pytest

from rendering.config import render_config_from_mapping
from rendering.isaac.settings import IsaacSettings


def _settings(raw):
    return IsaacSettings.from_config(render_config_from_mapping({"isaac": raw}, "."))


@pytest.mark.parametrize(
    "raw",
    [
        {"unknown_knob": 3},
        {"spp": 0},
        {"spp": -1},
        {"spp": 1.5},
        {"spp": True},
        {"render_mode": "NotARenderer"},
        {"bbox_source": "bad"},
        {"capture_timeout_sec": 0},
        {"capture_timeout_sec": -1},
        {"capture_timeout_sec": float("nan")},
        {"capture_timeout_sec": float("inf")},
        {"occlusion_tol_m": -1},
        {"occlusion_tol_m": float("nan")},
        {"light_intensity_range": [-1, 10]},
        {"light_intensity_range": [20, 10]},
        {"light_intensity_range": [0, float("inf")]},
        {"dome_intensity_range": [2, 1]},
        {"dome_intensity_range": [1, 2, 3]},
    ],
)
def test_invalid_requested_settings_are_rejected(raw):
    with pytest.raises(ValueError):
        _settings(raw)


def test_settings_verify_requested_values_against_effective_renderer_values():
    settings = _settings({"spp": 3, "light_intensity_range": [10, 20], "capture_timeout_sec": 12})
    requested = settings.carb_settings()
    assert requested["/rtx/pathtracing/spp"] == 3
    assert requested["/omni/replicator/RTSubframes"] == 3
    assert settings.verify_effective(requested.get) == requested
    wrong = dict(requested, **{"/rtx/pathtracing/spp": 1})
    with pytest.raises(RuntimeError, match="did not apply"):
        settings.verify_effective(wrong.get)
    with pytest.raises(RuntimeError):
        settings.verify_effective(lambda key: None)
    assert settings.to_dict()["light_intensity_range"] == [10, 20]
    assert settings.to_dict()["capture_timeout_sec"] == 12


def test_raytraced_mode_uses_eight_subframes():
    settings = _settings({"render_mode": "RayTracedLighting", "spp": 3})
    assert settings.carb_settings()["/rtx/rendermode"] == "RaytracedLighting"
    assert settings.app_config(64, 0)["renderer"] == "RaytracedLighting"
    assert settings.carb_settings()["/omni/replicator/RTSubframes"] == 8


def test_app_reset_defaults_match_requested_backend():
    cfg = render_config_from_mapping({"isaac": {"render_mode": "PathTracing", "spp": 7}}, ".")
    settings = IsaacSettings.from_config(cfg)
    launch = settings.app_config(64, 2)
    assert launch["renderer"] == settings.carb_settings()["/rtx/rendermode"]
    assert launch["samples_per_pixel_per_frame"] == settings.spp
    assert launch["active_gpu"] == launch["physics_gpu"] == 2
    assert launch["multi_gpu"] is False


@pytest.mark.parametrize("raw", [[], False, 0, "", [1], "invalid", True, 1])
def test_nonmapping_isaac_section_rejected_by_both_entry_points(raw):
    with pytest.raises(ValueError, match="isaac settings must be a mapping"):
        render_config_from_mapping({"isaac": raw}, ".")
    with pytest.raises(ValueError, match="isaac settings must be a mapping"):
        IsaacSettings.from_config(SimpleNamespace(raw={"isaac": raw}))


@pytest.mark.parametrize("raw", [{}, {"isaac": None}, {"isaac": {}}])
def test_missing_or_null_isaac_section_retains_defaults(raw):
    cfg = render_config_from_mapping(raw, ".")
    assert IsaacSettings.from_config(cfg).render_mode == "PathTracing"


def test_legacy_photometric_controls_are_opt_in_and_verified():
    baseline = _settings({"profile": "legacy"})
    assert not any("/post/" in key or "Denoiser" in key for key in baseline.carb_settings())
    assert "tonemap_op" not in baseline.to_dict()
    candidate = _settings(
        {
            "tonemap_op": "aces",
            "exposure_time": 0.01,
            "film_iso": 100.0,
            "f_number": 4.0,
            "auto_exposure": False,
            "denoiser": True,
        }
    )
    requested = candidate.carb_settings()
    assert requested["/rtx/post/tonemap/op"] == 6
    assert requested["/rtx/post/tonemap/exposureTime"] == 0.01
    assert requested["/rtx/post/histogram/enabled"] is False
    assert requested["/rtx/pathtracing/optixDenoiser/enabled"] is True
    assert candidate.to_dict()["tonemap_op"] == "aces"
    effective = dict(requested)
    effective["/rtx/post/tonemap/exposureTime"] = 0.009999999776482582
    assert candidate.verify_effective(effective.get) == effective
    effective["/rtx/post/tonemap/exposureTime"] = 0.02
    with pytest.raises(RuntimeError, match="did not apply"):
        candidate.verify_effective(effective.get)


@pytest.mark.parametrize(
    "raw",
    [
        {"tonemap_op": "unknown"},
        {"tonemap_op": 6},
        {"exposure_time": 0},
        {"exposure_time": True},
        {"exposure_time": float("nan")},
        {"film_iso": -1},
        {"f_number": float("inf")},
        {"denoiser": "false"},
        {"auto_exposure": 0},
    ],
)
def test_photometric_controls_reject_ambiguous_or_invalid_values(raw):
    with pytest.raises(ValueError):
        _settings(raw)


def test_default_profile_matches_selected_calibration_without_changing_raw_config():
    raw = {"render": {"samples": 128}}
    cfg = render_config_from_mapping(raw, ".")
    settings = IsaacSettings.from_config(cfg)
    assert cfg.spp == settings.spp == 32
    assert settings.tonemap_op == "aces"
    assert settings.exposure_time == 0.0075
    assert settings.film_iso == 100.0
    assert settings.f_number == 5.0
    assert settings.auto_exposure is False
    assert settings.denoiser is True
    assert settings.match_blender_hdri is True
    assert settings.match_blender_ground_uv is True
    assert raw == cfg.raw == {"render": {"samples": 128}}


def test_legacy_profile_preserves_previous_defaults_and_manifest_shape():
    cfg = render_config_from_mapping(
        {"isaac": {"profile": "legacy"}, "render": {"samples": 128}}, "."
    )
    settings = IsaacSettings.from_config(cfg)
    assert cfg.spp == settings.spp == 128
    assert not settings.has_photometric_controls
    assert not settings.match_blender_hdri
    assert "profile" not in settings.to_dict()
    assert "tonemap_op" not in settings.to_dict()


def test_explicit_settings_override_default_profile():
    settings = _settings(
        {
            "spp": 8,
            "tonemap_op": "linear",
            "exposure_time": None,
            "match_blender_hdri": False,
            "denoiser": False,
        }
    )
    assert settings.spp == 8
    assert settings.tonemap_op == "linear"
    assert settings.exposure_time is None
    assert settings.match_blender_hdri is False
    assert settings.denoiser is False


def test_unknown_profile_rejected():
    with pytest.raises(ValueError, match="profile"):
        _settings({"profile": "typo"})
