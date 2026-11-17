# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay verified native Blender environments with an explicit runtime adapter."""

# pyright: reportMissingImports=false, reportMissingModuleSource=false

from __future__ import annotations

from typing import Any

from pxr import Gf as _Gf
from pxr import Usd as _Usd
from pxr import UsdGeom as _UsdGeom

from rendering.environment_bundle import (
    ReplayInputs,
    canonical_hash,
    validate_light_readbacks,
)
from rendering.environment_bundle_usd import validate_layer_dependencies
from rendering.isaac.background_mapping import blender_dome_rotation

# usd-core ships no type information for its extension modules; use them untyped.
Gf: Any = _Gf
Usd: Any = _Usd
UsdGeom: Any = _UsdGeom

ROOT = "/World/BlenderEnvironment"


class EnvironmentReplay:
    """Reference one sealed native environment per scene and verify it after capture."""

    def __init__(
        self,
        stage: Any,  # noqa: ANN401 - Usd.Stage
        inputs: ReplayInputs,
    ) -> None:
        """Bind the stage and the verified replay inputs; nothing is authored yet."""
        self.stage = stage
        self.inputs = inputs
        self.expected: list[dict[str, Any]] | None = None
        self.index: int | None = None

    def _lights(self, root: Any) -> list[Any]:  # noqa: ANN401 - Usd.Prim
        return [prim for prim in Usd.PrimRange(root) if prim.GetTypeName().endswith("Light")]

    def _readback(self) -> list[dict[str, Any]]:
        return [
            {
                "path": str(prim.GetPath()),
                "type": prim.GetTypeName(),
                "world_matrix": [
                    list(row) for row in UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
                ],
                "intensity": prim.GetAttribute("inputs:intensity").Get(),
                "exposure": prim.GetAttribute("inputs:exposure").Get(),
            }
            for prim in self._lights(self.stage.GetPrimAtPath(ROOT))
        ]

    def apply(self, record: dict[str, Any]) -> None:
        """Reference the record's native layer, rotate its dome and apply the light gain once.

        Raises:
            ValueError: The record is not an HDRI scene, or the composed light/ground inventory
                differs from the recorded one.
            RuntimeError: The layer cannot be referenced.
        """
        metadata = self.inputs.bundle.verify_record(record)
        # Isaac exposes pxr only after Kit starts. Hashes are checked in the
        # launcher; enumerate USD dependencies here before stage composition.
        validate_layer_dependencies(self.inputs.bundle.root, metadata)
        if record["background"]["mode"] != "hdri":
            raise ValueError("native environment adapter requires an HDRI background")
        self.stage.RemovePrim(ROOT)
        root = self.stage.DefinePrim(ROOT)
        path = self.inputs.bundle.root / metadata["layer"]
        if not root.GetReferences().AddReference(str(path)):
            raise RuntimeError(f"cannot reference native environment {path}")
        lights = self._lights(root)
        domes = [prim for prim in lights if prim.GetTypeName() == "DomeLight"]
        meshes = [prim for prim in Usd.PrimRange(root) if prim.IsA(UsdGeom.Mesh)]
        if (
            len(lights) != metadata["light_count"]
            or len(domes) != 1
            or bool(meshes) != metadata["ground"]
        ):
            raise ValueError(
                "realized native environment differs from recorded light/ground inventory"
            )
        transform = UsdGeom.Xformable(domes[0])
        transform.ClearXformOpOrder()
        rotation = blender_dome_rotation(record["background"]["rotation_deg"])
        transform.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, rotation))
        for light in lights:
            attr = light.GetAttribute("inputs:intensity")
            intensity = attr.Get()
            if intensity is None:
                raise ValueError("native light has no intensity")
            attr.Set(intensity * self.inputs.options["gain"])
        self.expected = self._readback()
        validate_light_readbacks(self.expected, expected_count=metadata["light_count"])
        self.index = record["index"]

    def verify(self, record: dict[str, Any]) -> dict[str, Any]:
        """Return the post-capture evidence, requiring the lights to be unchanged since apply.

        Raises:
            RuntimeError: The live lights or the applied scene changed.
        """
        actual = self._readback()
        if self.index != record["index"] or actual != self.expected:
            raise RuntimeError("native environment changed after capture")
        return {**self.inputs.identity, "scene_sha256": canonical_hash(record), "lights": actual}
