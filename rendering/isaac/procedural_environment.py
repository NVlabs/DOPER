# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in Blender 4.2 appearance mapping, without a Blender export dependency.

Import after Kit startup. Light normalization follows Blender 4.2's USD exporter: energy/pi for
local lights, energy/4 for Sun. The additional gain is empirical.
"""

# pyright: reportMissingImports=false, reportMissingModuleSource=false

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
from pxr import (
    Gf as _Gf,
)
from pxr import (
    Sdf as _Sdf,
)
from pxr import (
    UsdGeom as _UsdGeom,
)
from pxr import (
    UsdLux as _UsdLux,
)
from pxr import (
    UsdShade as _UsdShade,
)

from doper.scene import geometry as G  # noqa: N812 - short alias used across the sampling code
from rendering.appearance import blender_field_seed, sample_hdri, sample_light_energy
from rendering.config import RenderConfig
from rendering.isaac.background_mapping import blender_dome_rotation
from rendering.isaac.environment_profile import EnvironmentProfile
from rendering.isaac.usd_utils import set_world_matrix, texture_set
from rendering.scene_spec import AssetCatalog

# usd-core ships no type information for its extension modules; use them untyped.
Gf: Any = _Gf
Sdf: Any = _Sdf
UsdGeom: Any = _UsdGeom
UsdLux: Any = _UsdLux
UsdShade: Any = _UsdShade

SPOT_CONE_ANGLE_DEG = 22.5
SPOT_CONE_SOFTNESS = 0.15
GROUND_OFFSET = 0.001  # Ground sits this fraction of the scene scale below the object.
GROUND_SURFACE = {
    "clearcoat": 0.0,
    "clearcoatRoughness": 0.03,
    "ior": 1.5,
    "metallic": 0.0,
    "opacity": 1.0,
    "roughness": 0.5,
    "specular": 0.5,
}


class ProceduralEnvironment:
    """Rebuild from each authoritative record, including on random-access resume."""

    def __init__(
        self,
        stage: Any,  # noqa: ANN401 - Usd.Stage
        catalog: AssetCatalog,
        cfg: RenderConfig,
        scene_scale: float,
        profile: EnvironmentProfile,
    ) -> None:
        """Bind the stage, asset catalog and profile; nothing is authored yet."""
        self.stage = stage
        self.catalog = catalog
        self.cfg = cfg
        self.scene_scale = scene_scale
        self.profile = profile

    def apply(self, record: dict[str, Any]) -> None:
        """Replace the dome, lights and ground with the ones of ``record``.

        Raises:
            ValueError: The record has no HDRI background or an unsupported light kind.
        """
        if record["background"]["mode"] != "hdri":
            raise ValueError("blender42_v1 requires an HDRI scene record")
        for path in ("/World/Dome", "/World/PortableEnvironment", "/World/Ground"):
            if self.stage.GetPrimAtPath(path):
                self.stage.RemovePrim(path)
        self.stage.DefinePrim("/World/PortableEnvironment", "Xform")
        self._lights(record)
        self._ground(record)

    def _common_light(
        self,
        light: Any,  # noqa: ANN401 - UsdLux light schema
        intensity: float,
        color: Any,  # noqa: ANN401 - RGB sequence
    ) -> None:
        api = UsdLux.LightAPI(light.GetPrim())
        api.CreateIntensityAttr().Set(intensity * self.profile.light_gain)
        api.CreateExposureAttr().Set(0.0)
        api.CreateColorAttr().Set(Gf.Vec3f(*color))
        api.CreateNormalizeAttr().Set(True)
        api.CreateDiffuseAttr().Set(1.0)
        api.CreateSpecularAttr().Set(1.0)

    def _dome(self, record: dict[str, Any]) -> None:
        group = record["index"] // max(1, self.cfg.change_hdri_every)
        rng = random.Random(blender_field_seed(record["seed"], "background", group))
        _, strength = sample_hdri(rng)
        dome = UsdLux.DomeLight.Define(self.stage, "/World/Dome")
        dome.CreateIntensityAttr().Set(float(np.float32(strength)) * self.profile.light_gain)
        dome.CreateTextureFileAttr().Set(
            Sdf.AssetPath(self.catalog.resolve(record["background"]["hdri"]))
        )
        UsdGeom.Xformable(dome).AddRotateXYZOp().Set(
            Gf.Vec3f(0, 0, blender_dome_rotation(record["background"]["rotation_deg"]))
        )

    def _sun(self) -> None:
        preset = self.profile.sun
        if preset is None:
            return
        sun = UsdLux.DistantLight.Define(self.stage, "/World/PortableEnvironment/Sun")
        sun.CreateAngleAttr().Set(preset.angle_degrees / 2.0)
        self._common_light(sun, preset.energy / 4.0, preset.color)
        UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(*preset.rotation_degrees))

    def _define_light(self, path: str, spec: dict[str, Any]) -> Any:  # noqa: ANN401 - UsdLux
        if spec["kind"] == "rect":
            light = UsdLux.RectLight.Define(self.stage, path)
            light.CreateWidthAttr().Set(spec["scale"])
            light.CreateHeightAttr().Set(spec["scale"])
            return light
        if spec["kind"] not in ("sphere", "disk"):
            raise ValueError(f"Unsupported light kind: {spec['kind']}")
        light = UsdLux.SphereLight.Define(self.stage, path)
        light.CreateRadiusAttr().Set(0.0)
        light.CreateTreatAsPointAttr().Set(True)
        if spec["kind"] == "disk":
            shaping = UsdLux.ShapingAPI.Apply(light.GetPrim())
            shaping.CreateShapingConeAngleAttr().Set(SPOT_CONE_ANGLE_DEG)
            shaping.CreateShapingConeSoftnessAttr().Set(SPOT_CONE_SOFTNESS)
        return light

    def _lights(self, record: dict[str, Any]) -> None:
        self._dome(record)
        self._sun()
        rng = random.Random(blender_field_seed(record["seed"], "lights", record["index"]))
        center = np.asarray(record["object"]["center"], dtype=np.float32)
        for index, spec in enumerate(record["lights"]):
            light = self._define_light(f"/World/PortableEnvironment/Light{index}", spec)
            # Blender stores energy as float32 before its USD intensity conversion.
            energy = float(np.float32(sample_light_energy(rng)))
            self._common_light(light, float(np.float32(energy / math.pi)), spec["color"])
            position = np.asarray(spec["position"], dtype=np.float32).astype(float)
            pose = (
                G.look_at_camera(position, center)
                if spec["kind"] in ("rect", "disk")
                else G.make_pose(np.eye(3), position)
            )
            set_world_matrix(light.GetPrim(), pose)

    def _ground(self, record: dict[str, Any]) -> None:
        spec = record["ground"]
        if not spec["enabled"]:
            return
        half = self.scene_scale
        mesh = UsdGeom.Mesh.Define(self.stage, "/World/Ground")
        mesh.CreatePointsAttr(
            [
                Gf.Vec3f(-half, -half, 0),
                Gf.Vec3f(half, -half, 0),
                Gf.Vec3f(half, half, 0),
                Gf.Vec3f(-half, half, 0),
            ]
        )
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
        mesh.CreateExtentAttr([Gf.Vec3f(-half, -half, 0), Gf.Vec3f(half, half, 0)])
        uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying
        )
        uv.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])
        height = record["object"]["bbox_min_z"] - GROUND_OFFSET * self.scene_scale
        set_world_matrix(mesh.GetPrim(), G.make_pose(np.eye(3), (0, 0, height)))
        root = "/World/PortableEnvironment/GroundMaterial"
        material = UsdShade.Material.Define(self.stage, root)
        surface = UsdShade.Shader.Define(self.stage, root + "/Surface")
        surface.CreateIdAttr("UsdPreviewSurface")
        material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
        for key, value in GROUND_SURFACE.items():
            surface.CreateInput(key, Sdf.ValueTypeNames.Float).Set(value)
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.8, 0.8, 0.8))
        if spec["kind"] == "texture":
            self._ground_textures(root, surface, spec)
        else:
            surface.GetInput("diffuseColor").Set(Gf.Vec3f(*spec["color"]))
            surface.GetInput("roughness").Set(spec["roughness"])
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    def _ground_mapping(self, root: str, spec: dict[str, Any]) -> Any:  # noqa: ANN401 - Shader
        reader = UsdShade.Shader.Define(self.stage, root + "/UV")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        mapping = UsdShade.Shader.Define(self.stage, root + "/Mapping")
        mapping.CreateIdAttr("UsdTransform2d")
        mapping.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        mapping.CreateInput("rotation", Sdf.ValueTypeNames.Float).Set(spec["texture_rotate_deg"])
        mapping.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(
            Gf.Vec2f(*([spec["texture_scale"] / 3.0] * 2))
        )
        mapping.CreateInput("translation", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(0, 0))
        mapping.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        return mapping

    def _ground_textures(
        self,
        root: str,
        surface: Any,  # noqa: ANN401 - UsdShade.Shader
        spec: dict[str, Any],
    ) -> None:
        mapping = self._ground_mapping(root, spec)
        paths = texture_set(self.catalog.resolve(spec["texture_set"]))
        for key, channel, kind, texture, color_space in zip(
            ("diffuseColor", "roughness", "normal"),
            ("rgb", "r", "rgb"),
            (Sdf.ValueTypeNames.Color3f, Sdf.ValueTypeNames.Float, Sdf.ValueTypeNames.Float3),
            paths,
            ("sRGB", "raw", "raw"),
            strict=True,
        ):
            if texture is None:
                continue
            node = UsdShade.Shader.Define(self.stage, root + "/" + key)
            node.CreateIdAttr("UsdUVTexture")
            node.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture))
            node.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(color_space)
            for axis in ("wrapS", "wrapT"):
                node.CreateInput(axis, Sdf.ValueTypeNames.Token).Set("repeat")
            node.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                mapping.ConnectableAPI(), "result"
            )
            if key == "normal":
                node.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 2))
                node.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, -1))
            node.CreateOutput(channel, kind)
            surface.CreateInput(key, kind).ConnectToSource(node.ConnectableAPI(), channel)
