# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real USD authoring of the photometric controls (needs ``pxr``, e.g. usd-core)."""

import pytest

from rendering.config import render_config_from_mapping
from rendering.isaac.settings import IsaacSettings

pxr = pytest.importorskip("pxr")


def stage_prims():
    from pxr import Usd, UsdGeom, UsdRender

    stage = Usd.Stage.CreateInMemory()
    camera = UsdGeom.Camera.Define(stage, "/Camera").GetPrim()
    product = UsdRender.Product.Define(stage, "/Product").GetPrim()
    return stage, camera, product


def test_authors_exposure_on_camera_and_tonemapping_on_product():
    _, camera, product = stage_prims()
    controls = {
        "tonemap_op": "hable",
        "exposure_time": 0.01,
        "film_iso": 80.0,
        "f_number": 4.0,
        "auto_exposure": False,
        "denoiser": False,
    }
    settings = IsaacSettings.from_config(render_config_from_mapping({"isaac": controls}, "."))
    settings.apply_to_stage(camera, product)
    assert camera.GetAttribute("exposure:time").Get() == pytest.approx(0.01)
    assert camera.GetAttribute("exposure:iso").Get() == 80.0
    assert camera.GetAttribute("exposure:fStop").Get() == 4.0
    assert camera.GetAttribute("omni:rtx:autoExposure:enabled").Get() is False
    assert product.GetAttribute("omni:rtx:post:tonemap:operator").Get() == "hableUc2"
    assert product.GetAttribute("omni:rtx:pt:denoising:enabled").Get() is False


def test_default_profile_authors_calibrated_settings():
    _, camera, product = stage_prims()
    settings = IsaacSettings.from_config(render_config_from_mapping({}, "."))
    settings.apply_to_stage(camera, product)
    assert camera.GetAttribute("exposure:time").Get() == pytest.approx(0.0075)
    assert camera.GetAttribute("exposure:iso").Get() == 100.0
    assert camera.GetAttribute("exposure:fStop").Get() == 5.0
    assert camera.GetAttribute("omni:rtx:autoExposure:enabled").Get() is False
    assert product.GetAttribute("omni:rtx:post:tonemap:operator").Get() == "acesApproximation"
    assert product.GetAttribute("omni:rtx:pt:denoising:enabled").Get() is True


def test_legacy_controls_do_not_author_or_replace_settings():
    from pxr import Sdf

    stage, camera, product = stage_prims()
    camera.CreateAttribute("exposure:time", Sdf.ValueTypeNames.Float).Set(0.25)
    before = stage.GetRootLayer().ExportToString()
    legacy = render_config_from_mapping({"isaac": {"profile": "legacy"}}, ".")
    IsaacSettings.from_config(legacy).apply_to_stage(camera, product)
    assert stage.GetRootLayer().ExportToString() == before
