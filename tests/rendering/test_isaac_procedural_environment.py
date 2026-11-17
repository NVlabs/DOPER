# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""USD authoring contracts of the portable environment (needs ``pxr``, e.g. usd-core)."""

from types import SimpleNamespace

import numpy as np
import pytest

from rendering.isaac.environment_profile import EnvironmentProfile

pytest.importorskip("pxr")

from pxr import Usd, UsdGeom, UsdLux, UsdShade  # noqa: E402

from rendering.isaac.procedural_environment import ProceduralEnvironment  # noqa: E402

SUN = {
    "energy": 2.0,
    "rotation_degrees": [45.0, 30.0, 0.0],
    "angle_degrees": 0.526,
    "color": [1.0, 1.0, 1.0],
}


def fixture(sun=None):
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World", "Xform")
    profile = EnvironmentProfile.from_mapping(
        {"name": "blender42_v1", "light_gain": 2000.0, "sun": sun}
    )
    env = ProceduralEnvironment(
        stage,
        SimpleNamespace(resolve=lambda p: p),  # type: ignore[arg-type]
        SimpleNamespace(change_hdri_every=3),  # type: ignore[arg-type]
        2.0,
        profile,
    )
    record = {
        "index": 0,
        "seed": 2751,
        "background": {"mode": "hdri", "hdri": "/missing.hdr", "rotation_deg": 30.0},
        "object": {"center": [0, 0, 0], "bbox_min_z": 0.25},
        "ground": {"enabled": True, "kind": "color", "color": [0.2, 0.3, 0.4], "roughness": 0.6},
        "lights": [
            {"kind": kind, "position": [1.0, 2.0, 3.0], "color": [1.0, 1.0, 1.0], "scale": 0.5}
            for kind in ("sphere", "rect", "disk")
        ],
    }
    return stage, env, record


def test_light_types_mapping_and_explicit_no_sun():
    stage, env, record = fixture()
    env.apply(record)
    lights = [p for p in stage.Traverse() if p.HasAPI(UsdLux.LightAPI)]
    assert len(lights) == 4
    assert not stage.GetPrimAtPath("/World/PortableEnvironment/Sun")
    dome = UsdLux.DomeLight(stage.GetPrimAtPath("/World/Dome"))
    np.testing.assert_allclose(dome.GetIntensityAttr().Get(), 0.6404328942298889 * 2000)
    point = UsdLux.SphereLight(stage.GetPrimAtPath("/World/PortableEnvironment/Light0"))
    assert point.GetTreatAsPointAttr().Get()
    assert point.GetRadiusAttr().Get() == 0
    np.testing.assert_allclose(point.GetIntensityAttr().Get(), 6012.508392333984, rtol=1e-6)
    area = UsdLux.RectLight(stage.GetPrimAtPath("/World/PortableEnvironment/Light1"))
    assert area.GetWidthAttr().Get() == 0.5
    assert area.GetHeightAttr().Get() == 0.5
    spot = stage.GetPrimAtPath("/World/PortableEnvironment/Light2")
    assert UsdLux.ShapingAPI(spot).GetShapingConeAngleAttr().Get() == 22.5
    assert all(UsdLux.LightAPI(p).GetNormalizeAttr().Get() for p in lights if p != dome.GetPrim())


def test_sun_units_and_random_access_do_not_accumulate_prims():
    stage, env, record = fixture(SUN)
    env.apply(record)
    expected = stage.GetRootLayer().ExportToString()
    record["index"] = 7
    env.apply(record)
    record["index"] = 0
    env.apply(record)
    assert stage.GetRootLayer().ExportToString() == expected
    sun = UsdLux.DistantLight(stage.GetPrimAtPath("/World/PortableEnvironment/Sun"))
    assert sun.GetIntensityAttr().Get() == 1000.0
    np.testing.assert_allclose(sun.GetAngleAttr().Get(), 0.263)


def test_ground_extent_uv_mapping_and_disable(tmp_path):
    stage, env, record = fixture()
    for name in ("Color", "Roughness", "NormalGL"):
        (tmp_path / f"{name}.jpg").touch()
    record["ground"] = {
        "enabled": True,
        "kind": "texture",
        "texture_set": str(tmp_path),
        "texture_scale": 6.0,
        "texture_rotate_deg": 90.0,
    }
    env.apply(record)
    ground = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Ground"))
    np.testing.assert_array_equal(ground.GetExtentAttr().Get(), [[-2, -2, 0], [2, 2, 0]])
    np.testing.assert_array_equal(
        UsdGeom.PrimvarsAPI(ground).GetPrimvar("st").Get(), [[0, 0], [1, 0], [1, 1], [0, 1]]
    )
    np.testing.assert_allclose(
        UsdGeom.Xformable(ground).ComputeLocalToWorldTransform(0)[3][2], 0.248
    )
    material = UsdShade.MaterialBindingAPI(ground).ComputeBoundMaterial()[0]
    root = str(material.GetPath())
    mapping = UsdShade.Shader(stage.GetPrimAtPath(root + "/Mapping"))
    assert tuple(mapping.GetInput("scale").Get()) == (2.0, 2.0)
    assert mapping.GetInput("rotation").Get() == 90.0
    normal = UsdShade.Shader(stage.GetPrimAtPath(root + "/normal"))
    assert tuple(normal.GetInput("bias").Get()) == (-1.0,) * 4
    assert normal.GetInput("sourceColorSpace").Get() == "raw"
    record["ground"]["enabled"] = False
    env.apply(record)
    assert not stage.GetPrimAtPath("/World/Ground")
    assert not stage.GetPrimAtPath(root)


def test_background_cadence_reconstructs_the_same_strength():
    stage, env, record = fixture()

    def strength():
        return stage.GetPrimAtPath("/World/Dome").GetAttribute("inputs:intensity").Get()

    env.apply(record)
    first = strength()
    record["index"] = 2
    env.apply(record)
    assert strength() == first
    record["index"] = 3
    env.apply(record)
    assert strength() != first
