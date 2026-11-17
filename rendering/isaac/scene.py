# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kit-only scene realization. Import after ``SimulationApp`` has started.

Shared scene records are authoritative; cadence only caches assets and shading.
"""

# pyright: reportMissingImports=false, reportMissingModuleSource=false

from __future__ import annotations

import hashlib
import os
from typing import Any

import numpy as np
import omni.replicator.core as rep
import omni.usd
from pxr import Gf as _Gf
from pxr import Sdf as _Sdf
from pxr import Usd as _Usd
from pxr import UsdGeom as _UsdGeom
from pxr import UsdShade as _UsdShade

from doper.scene import geometry as G  # noqa: N812 - short alias used across the sampling code
from rendering.isaac.background_mapping import blender_dome_rotation, blender_ground_uv
from rendering.isaac.environment_replay import EnvironmentReplay
from rendering.isaac.inputs import WorkerInputs
from rendering.isaac.procedural_environment import ProceduralEnvironment
from rendering.isaac.usd_utils import set_world_matrix, texture_set
from rendering.resume import CadenceTracker
from rendering.robustness import require_backend

# usd-core ships no type information for its extension modules; use them untyped.
Gf: Any = _Gf
Sdf: Any = _Sdf
Usd: Any = _Usd
UsdGeom: Any = _UsdGeom
UsdShade: Any = _UsdShade

GROUND_HALF_EXTENT = 3.0  # Ground plane half size, in scene scales.
GROUND_OFFSET = 0.001  # Ground sits this fraction of the scene scale below the object.
CLIPPING_RANGE = (0.001, 100.0)
OBJECT_LOOK = {
    "diffuse_color_constant": (0.85, 0.85, 0.85),
    "reflection_roughness_constant": 0.25,
    "specular_level": 0.8,
}
SEED_MODULUS = 2**31


def set_material_inputs(material_prim: Any, **inputs: Any) -> None:  # noqa: ANN401 - USD values
    """Set OmniPBR shader inputs; strings are texture asset paths (``""`` clears a texture)."""
    types = {
        str: Sdf.ValueTypeNames.Asset,
        bool: Sdf.ValueTypeNames.Bool,
        float: Sdf.ValueTypeNames.Float,
    }
    for name, value in inputs.items():
        if isinstance(value, (tuple, list)):
            value = tuple(float(v) for v in value)
            attr_type = Sdf.ValueTypeNames.Float2 if len(value) == 2 else Sdf.ValueTypeNames.Color3f
        else:
            value = value if isinstance(value, (str, bool)) else float(value)
            attr_type = types[type(value)]
        rep.functional.modify.attribute(material_prim, f"inputs:{name}", value, attr_type)


def has_valid_material(prim: Any) -> bool:  # noqa: ANN401 - Usd.Prim
    """True if every mesh under ``prim`` has a bound material whose texture files resolve.

    Mirrors the Blender worker: some asset packs reference textures they do not ship, which
    would otherwise render black.
    """
    meshes = [p for p in Usd.PrimRange(prim) if p.IsA(UsdGeom.Mesh)]
    for mesh in meshes:
        material, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
        if not material:
            return False
        shaders = (p for p in Usd.PrimRange(material.GetPrim()) if p.IsA(UsdShade.Shader))
        for shader in shaders:
            for attr in shader.GetPrim().GetAttributes():
                if attr.GetTypeName() != Sdf.ValueTypeNames.Asset or not attr.HasValue():
                    continue
                resolved = attr.Get().resolvedPath
                if not resolved or not os.path.exists(resolved):
                    return False
    return True


class KitScene:
    """Own the USD stage and backend-specific caches for one render worker."""

    def __init__(self, inputs: WorkerInputs) -> None:
        """Create the stage: object, camera, annotators, materials and environment adapters."""
        self.cfg, self.args, self.catalog = inputs.cfg, inputs.args, inputs.catalog
        self.settings = inputs.settings
        self.mesh_size, self.scene_scale = inputs.mesh.mesh_size, inputs.mesh.scene_scale
        context = omni.usd.get_context()
        context.new_stage()
        self.stage = context.get_stage()
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        rep.functional.create.xform(name="World")
        for scope in ("Looks", "Distractors", "Lights"):
            rep.functional.create.scope(name=scope, parent="/World")
        self.obj_prim = rep.functional.create.reference(
            usd_path=inputs.usd_path,
            name="object",
            parent="/World",
            semantics=[("class", "object")],
        )
        self.obj_path = str(self.obj_prim.GetPath())
        self.object_material = self._object_material(inputs.usd_path, inputs.mesh_data)
        self._create_camera()
        self.ground_material = rep.functional.create.material(
            mdl="OmniPBR.mdl",
            name="GroundMat",
            parent="/World/Looks",
            diffuse_color_constant=(0.4, 0.4, 0.4),
            reflection_roughness_constant=0.7,
        )
        self.bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
        )
        self.distractor_prims: list[Any] = []
        self.light_prims: list[Any] = []
        self.distractor_instances: list[tuple[int, Any, float]] = []
        self.cadence = CadenceTracker()
        self.procedural_environment = None
        if self.settings.environment_profile is not None:
            self.procedural_environment = ProceduralEnvironment(
                self.stage,
                self.catalog,
                self.cfg,
                self.scene_scale,
                self.settings.environment_profile,
            )
        self.environment = None
        if inputs.environment is not None:
            self.environment = EnvironmentReplay(self.stage, inputs.environment)

    def _object_material(self, usd_path: str, mesh_data: dict[str, Any]) -> Any:  # noqa: ANN401
        """Keep an authored texture under ``preserve``; else bind a randomizable OmniPBR."""
        if mesh_data.get("texture_file") and self.cfg.object_material_mode == "preserve":
            texture_path = os.path.join(os.path.dirname(usd_path), mesh_data["texture_file"])
            if not os.path.isfile(texture_path):
                raise SystemExit(
                    f"object texture {texture_path} referenced by {usd_path} is missing"
                )
            print(
                f"[W{self.args.worker_id}] object texture: {mesh_data['texture_file']} "
                "(authored UsdPreviewSurface kept)"
            )
            return None
        return rep.functional.create.material(
            mdl="OmniPBR.mdl",
            name="ObjectMat",
            parent="/World/Looks",
            bind_prims=[self.obj_prim],
            bind_strength="strongerThanDescendants",
            metallic_constant=self.cfg.object_metallic,
            **OBJECT_LOOK,
        )

    def _create_camera(self) -> None:
        resolution = self.cfg.resolution
        self.cam_prim = rep.functional.create.camera(
            name="SynthCam",
            parent="/World",
            focal_length=self.cfg.focal_length_mm,
            horizontal_aperture=self.cfg.sensor_width_mm,
            clipping_range=CLIPPING_RANGE,
        )
        UsdGeom.Camera(self.cam_prim).GetVerticalApertureAttr().Set(self.cfg.sensor_width_mm)
        self.render_product = rep.create.render_product(
            str(self.cam_prim.GetPath()), (resolution, resolution)
        )
        self.rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        self.depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        self.seg_annot = rep.AnnotatorRegistry.get_annotator(
            "instance_segmentation", init_params={"colorize": False}
        )
        for annot in (self.rgb_annot, self.depth_annot, self.seg_annot):
            annot.attach([self.render_product])

    def load_distractors(self, distractors: list[dict[str, Any]]) -> None:
        """Realize the record's distractors.

        The fallback material applies only to assets whose own textures do not resolve.
        """
        self.clear_distractors()
        for i, spec in enumerate(distractors):
            path = self.catalog.resolve(spec["file"])
            prim = rep.functional.create.reference(
                usd_path=path, name=f"d{i}", parent="/World/Distractors"
            )
            if not prim.IsValid():
                raise RuntimeError(f"Failed to reference distractor {path}")
            if not has_valid_material(prim):
                look = spec["material"]
                rep.functional.create.material(
                    mdl="OmniPBR.mdl",
                    name=f"DistractorMat{i}",
                    parent="/World/Looks",
                    bind_prims=[prim],
                    bind_strength="strongerThanDescendants",
                    diffuse_color_constant=tuple(look["color"]),
                    metallic_constant=look["metallic"],
                    reflection_roughness_constant=look["roughness"],
                )
            self.bbox_cache.Clear()
            extent = self.bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            size = max(extent.GetSize()) if not extent.IsEmpty() else 0.0
            scale = spec["size_frac"] * self.mesh_size / size if size > 0 else 1.0
            rot = G.euler_xyz_to_matrix(*spec["rotation_xyz"])
            set_world_matrix(prim, G.make_pose(rot, np.asarray(spec["position"])), scale=scale)
            self.distractor_prims.append(prim)
            self.distractor_instances.append((i, prim, scale))

    def setup_world_background(self, background: dict[str, Any]) -> None:
        """Replace the dome light with the record's HDRI or plain-color background."""
        self.remove_prim("/World/Dome")
        if background["mode"] == "hdri":
            rotation = background["rotation_deg"]
            if self.settings.match_blender_hdri:
                rotation = blender_dome_rotation(rotation)
            rep.functional.create.dome_light(
                name="Dome",
                parent="/World",
                texture=self.catalog.resolve(background["hdri"]),
                intensity=background["intensity"],
                rotation=(0, 0, rotation),
            )
            return
        rep.functional.create.dome_light(
            name="Dome",
            parent="/World",
            color=tuple(background["color"]),
            intensity=background["intensity"],
        )

    def apply_object_material(self, material: dict[str, Any]) -> None:
        """Apply the record's object color, roughness and metalness (unless textured)."""
        if self.object_material is None:
            return
        set_material_inputs(
            self.object_material,
            diffuse_color_constant=tuple(material["color"]),
            reflection_roughness_constant=material["roughness"],
            metallic_constant=material["metallic"],
        )

    def _define_ground(self, ground: dict[str, Any]) -> Any:  # noqa: ANN401 - UsdGeom.Mesh
        half = GROUND_HALF_EXTENT * self.scene_scale
        corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
        mesh = UsdGeom.Mesh.Define(self.stage, "/World/Ground")
        mesh.CreatePointsAttr([Gf.Vec3f(x, y, 0) for x, y in corners])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
        mesh.CreateExtentAttr([Gf.Vec3f(-half, -half, 0), Gf.Vec3f(half, half, 0)])
        tex = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying
        )
        if self.settings.match_blender_ground_uv and ground["kind"] == "texture":
            uv = blender_ground_uv(
                corners, self.scene_scale, ground["texture_scale"], ground["texture_rotate_deg"]
            )
            tex.Set([Gf.Vec2f(float(u), float(v)) for u, v in uv])
        else:
            tex.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])
        return mesh

    def _texture_inputs(self, ground: dict[str, Any]) -> dict[str, Any]:
        color, rough, normal = texture_set(self.catalog.resolve(ground["texture_set"]))
        inputs: dict[str, Any] = {
            "diffuse_texture": color or "",
            "reflectionroughness_texture": rough or "",
            "normalmap_texture": normal or "",
            "reflection_roughness_texture_influence": 1.0 if rough else 0.0,
            "diffuse_color_constant": (1.0, 1.0, 1.0),
            "reflection_roughness_constant": 0.6,
            "texture_scale": (ground["texture_scale"],) * 2,
            "texture_rotate": ground["texture_rotate_deg"],
        }
        if self.settings.match_blender_ground_uv:
            inputs["texture_scale"] = (1.0, 1.0)
            inputs["texture_rotate"] = 0.0
        return inputs

    def setup_ground_plane(self, ground_z: float, ground: dict[str, Any]) -> None:
        """Replace the ground with the record's textured or plain-color plane (or none)."""
        self.remove_prim("/World/Ground")
        if not ground["enabled"]:
            return
        mesh = self._define_ground(ground)
        height = ground_z - GROUND_OFFSET * self.scene_scale
        set_world_matrix(mesh.GetPrim(), G.make_pose(np.eye(3), (0, 0, height)))
        rep.functional.modify.material(mesh.GetPrim(), UsdShade.Material(self.ground_material))
        if ground["kind"] == "texture":
            set_material_inputs(self.ground_material, **self._texture_inputs(ground))
            return
        set_material_inputs(
            self.ground_material,
            diffuse_color_constant=tuple(ground["color"]),
            reflection_roughness_constant=ground["roughness"],
            diffuse_texture="",
            reflectionroughness_texture="",
            normalmap_texture="",
            reflection_roughness_texture_influence=0.0,
        )

    def setup_lights(self, lights: list[dict[str, Any]], center: list[float]) -> None:
        """Replace the scene lights with the record's sphere, rect and disk lights."""
        for prim in self.light_prims:
            self.remove_prim(str(prim.GetPath()))
        self.light_prims.clear()
        target = tuple(float(v) for v in center)
        for i, spec in enumerate(lights):
            kwargs = {
                "name": f"L{i}",
                "parent": "/World/Lights",
                "position": tuple(spec["position"]),
                "color": tuple(spec["color"]),
                "intensity": spec["intensity"],
                "scale": spec["scale"],
            }
            if spec["kind"] == "sphere":
                prim = rep.functional.create.sphere_light(**kwargs)
            elif spec["kind"] == "rect":
                prim = rep.functional.create.rect_light(look_at=target, **kwargs)
            else:
                prim = rep.functional.create.disk_light(look_at=target, **kwargs)
            self.light_prims.append(prim)

    def clear_distractors(self) -> None:
        """Remove the distractor prims and their fallback materials."""
        for i, prim in enumerate(self.distractor_prims):
            self.remove_prim(str(prim.GetPath()))
            self.remove_prim(f"/World/Looks/DistractorMat{i}")
        self.distractor_prims.clear()
        self.distractor_instances.clear()

    def remove_prim(self, path: str) -> None:
        """Remove ``path`` from the stage if it exists."""
        if self.stage.GetPrimAtPath(path).IsValid():
            self.stage.RemovePrim(path)

    def realize(self, record: dict[str, Any]) -> None:
        """Refresh caches as needed, then apply every sample-specific transform."""
        require_backend(record.get("robustness"), "isaac", appearance=record.get("appearance"))
        index = record["index"]
        digest = hashlib.sha256(f"{record['seed']}:isaac:{index}".encode()).digest()
        rep.set_global_seed(int.from_bytes(digest[:4], "big") % SEED_MODULUS)
        set_world_matrix(self.obj_prim, record["object"]["pose"])
        if self.cadence.changed("material", index, self.cfg.change_material_every):
            self.apply_object_material(record["material"])
        portable = self.environment is None and self.procedural_environment is None
        if portable and self.cadence.changed("background", index, self.cfg.change_hdri_every):
            self.setup_world_background(record["background"])
        if portable and self.cadence.changed("ground", index, self.cfg.change_ground_every):
            self.setup_ground_plane(record["object"]["bbox_min_z"], record["ground"])
        if self.cadence.changed("distractors", index, self.cfg.change_distractors_every):
            self.load_distractors(record["distractors"])
        self.apply_sample_transforms(record)
        if self.procedural_environment is not None:
            self.procedural_environment.apply(record)
        elif self.environment is None:
            self.setup_lights(record["lights"], record["object"]["center"])
        else:
            self.environment.apply(record)

    def apply_sample_transforms(self, record: dict[str, Any]) -> None:
        """Move the cached ground and distractors to this record's placements."""
        ground = self.stage.GetPrimAtPath("/World/Ground")
        if ground.IsValid():
            height = record["object"]["bbox_min_z"] - GROUND_OFFSET * self.scene_scale
            set_world_matrix(ground, G.make_pose(np.eye(3), (0, 0, height)))
        for index, prim, scale in self.distractor_instances:
            spec = record["distractors"][index]
            pose = G.make_pose(G.euler_xyz_to_matrix(*spec["rotation_xyz"]), spec["position"])
            set_world_matrix(prim, pose, scale=scale)

    def set_camera(self, pose: np.ndarray) -> None:
        """Place the render camera at the camera-to-world ``pose``."""
        set_world_matrix(self.cam_prim, pose)
