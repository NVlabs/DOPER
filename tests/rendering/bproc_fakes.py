# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recording stand-ins for ``blenderproc`` and ``bpy``.

``record_draws`` runs a BlenderProc scene script with these fakes installed in ``sys.modules`` and
stops it when it reaches ``bproc.object.sample_poses``. Everything the script drew from the global
NumPy RNG before that point is captured through the values it hands to the fakes, plus a digest
of the RNG state at the stop. The fakes satisfy the scripts' unit checks by building each loaded
mesh from its ``models_info`` box. The same fakes record the source renderers and
``rendering/blenderproc/scene.py`` (``test_blenderproc_campaign``).

``shell`` follows BlenderProc 2.8.0's ``sampler.shell`` (default, not volume-uniform, mode): the
same three draws and arithmetic, so the fake consumes exactly what the real sampler does.
"""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import sys
import types
from pathlib import Path
from typing import Any
from unittest import mock

import cv2
import numpy as np


class StopAtPhysics(Exception):  # noqa: N818 - a control-flow signal, not an error
    """Raised by the fake ``sample_poses``: the pre-physics draws are complete."""


def rng_state_sha256(state: tuple) -> str:
    """Digest of a legacy NumPy RNG state (key array, position and Gaussian cache)."""
    name, keys, pos, has_gauss, cached = state
    h = hashlib.sha256()
    h.update(str(name).encode())
    h.update(np.asarray(keys, dtype=np.uint32).tobytes())
    h.update(f"{int(pos)}|{int(has_gauss)}|{float(cached)!r}".encode())
    return h.hexdigest()


def shell(
    center,
    radius_min,
    radius_max,
    elevation_min=-90,
    elevation_max=90,
    azimuth_min=-180,
    azimuth_max=180,
    uniform_volume=False,
):
    """BlenderProc 2.8.0 ``sampler.shell`` (the branch the renderers use)."""
    assert not uniform_volume
    center = np.array(center)
    el_sampled = np.deg2rad(elevation_min + (elevation_max - elevation_min) * np.random.rand())
    az_sampled = np.deg2rad(azimuth_min + (azimuth_max - azimuth_min) * np.random.rand())
    direction_vector = np.array(
        [
            np.sin(np.pi / 2 - el_sampled) * np.cos(az_sampled),
            np.sin(np.pi / 2 - el_sampled) * np.sin(az_sampled),
            np.cos(np.pi / 2 - el_sampled),
        ]
    )
    radius = np.random.uniform(radius_min, radius_max)
    return direction_vector * radius + center


class _Recorder:
    def __init__(self, scratch: Path) -> None:
        self.scratch = scratch
        self.loads: list[tuple[str, list[int]]] = []
        self.objects: list[tuple[str, int]] = []
        self.shader: dict[tuple[str, int], dict[str, Any]] = {}
        self.room: list[str] = []
        self.room_pool: list[str] = []
        self.emission: dict[str, Any] = {}
        self.light: dict[str, Any] = {}
        self.state_sha256 = ""


class _Material:
    def __init__(self, rec: _Recorder, owner: tuple[str, int] | None, name: str = "") -> None:
        self.rec, self.owner, self.name = rec, owner, name

    def set_principled_shader_value(self, key: str, value: Any) -> None:
        assert self.owner is not None
        self.rec.shader.setdefault(self.owner, {})[key] = value

    def get_name(self) -> str:
        return self.name

    def make_emissive(self, emission_strength: float, emission_color: list) -> None:
        self.rec.emission = {"strength": emission_strength, "color": list(emission_color)}


class _Vertex:
    def __init__(self, co: tuple[float, float, float]) -> None:
        self.co = co


class _Object:
    def __init__(self, rec: _Recorder, dataset: str, obj_id: int, info: dict, path: Path) -> None:
        self.rec, self.key = rec, (dataset, obj_id)
        low = np.array([info.get(f"min_{a}", 0.0) for a in "xyz"], float) / 1000
        high = low + np.array([info[f"size_{a}"] for a in "xyz"], float) / 1000
        data = types.SimpleNamespace(
            vertices=[_Vertex(tuple(low)), _Vertex(tuple(high))], uv_layers=["uv"], materials=[]
        )
        self.blender_obj = types.SimpleNamespace(data=data)
        self.cp = {"model_path": str(path), "bop_dataset_name": dataset, "category_id": obj_id}

    def get_cp(self, key: str) -> Any:
        return self.cp[key]

    def set_shading_mode(self, mode: str) -> None:
        self.rec.objects.append(self.key)

    def get_scale(self) -> list[float]:
        return [1.0, 1.0, 1.0]

    def get_materials(self) -> list[_Material]:
        return [_Material(self.rec, self.key)]

    def enable_rigidbody(self, *args: Any, **kwargs: Any) -> None:
        pass


class _Plane:
    def __init__(self, rec: _Recorder) -> None:
        self.rec = rec

    def enable_rigidbody(self, *args: Any, **kwargs: Any) -> None:
        pass

    def replace_materials(self, material: _Material) -> None:
        if material.name != "ceiling_emission":
            self.rec.room.append(material.name)


class _Light:
    def __init__(self, rec: _Recorder) -> None:
        self.rec = rec

    def set_energy(self, energy: float) -> None:
        self.rec.light["energy"] = energy

    def set_color(self, color: Any) -> None:
        self.rec.light["color"] = [float(c) for c in color]

    def set_location(self, location: Any) -> None:
        self.rec.light["location"] = [float(c) for c in location]


def _fake_modules(rec: _Recorder, gpu: str) -> dict[str, types.ModuleType]:
    model_file = rec.scratch / "model.bin"
    model_file.write_bytes(b"model")

    def load_bop_objs(
        bop_dataset_path: str, obj_ids: list[int], model_type: str = "", **_: Any
    ) -> list[_Object]:
        folder = Path(bop_dataset_path)
        info = json.loads(
            (
                folder / ("models_" + model_type if model_type else "models") / "models_info.json"
            ).read_text()
        )
        rec.loads.append((folder.name, [int(i) for i in obj_ids]))
        return [_Object(rec, folder.name, int(i), info[str(int(i))], model_file) for i in obj_ids]

    def load_ccmaterials(folder_path: str) -> list[_Material]:
        names = sorted(p.name for p in Path(folder_path).iterdir() if p.is_dir())
        rec.room_pool = names
        return [_Material(rec, None, name) for name in names]

    def sample_poses(**_: Any) -> None:
        rec.state_sha256 = rng_state_sha256(np.random.get_state())
        raise StopAtPhysics

    bproc = types.ModuleType("blenderproc")
    bproc.init = lambda *a, **k: None
    bproc.renderer = types.SimpleNamespace(set_render_devices=lambda **k: None)
    bproc.loader = types.SimpleNamespace(
        load_bop_objs=load_bop_objs, load_ccmaterials=load_ccmaterials
    )
    bproc.object = types.SimpleNamespace(
        create_primitive=lambda *a, **k: _Plane(rec), sample_poses=sample_poses
    )
    bproc.material = types.SimpleNamespace(create=lambda name: _Material(rec, None, name))
    bproc.types = types.SimpleNamespace(Light=lambda: _Light(rec))
    bproc.sampler = types.SimpleNamespace(shell=shell)
    device = types.SimpleNamespace(type="OPTIX", name=gpu, use=False)
    prefs = types.SimpleNamespace(
        compute_device_type="", devices=[device], refresh_devices=lambda: None
    )
    bpy = types.ModuleType("bpy")
    bpy.context = types.SimpleNamespace(
        preferences=types.SimpleNamespace(
            addons={"cycles": types.SimpleNamespace(preferences=prefs)}
        ),
        scene=types.SimpleNamespace(render=types.SimpleNamespace(threads_mode="", threads=0)),
    )
    bpy.app = types.SimpleNamespace(version_string="4.2.1")
    bpy.path = types.SimpleNamespace(abspath=lambda p: p)
    return {"blenderproc": bproc, "bpy": bpy}


def record_draws(script: Path, args: list[str], scratch: Path, gpu: str) -> dict[str, Any]:
    """Run ``script`` under the fakes until physics and return every pre-physics draw.

    Args:
        script: BlenderProc scene script (starts with ``import blenderproc as bproc``).
        args: Its command-line arguments (``--assets``, ``--out``, ``--seed``, ...).
        scratch: Directory for the fake model file.
        gpu: Name of the single fake OptiX device (the script's ``--gpu-name``).

    Returns:
        The draws, in the shape of ``rendering.blenderproc.plan.ScenePlan.as_record``.
    """
    rec = _Recorder(scratch)
    with (
        mock.patch.dict(sys.modules, _fake_modules(rec, gpu)),
        mock.patch.object(sys, "argv", [str(script), *args]),
        mock.patch.dict(os.environ, {}, clear=False),
    ):
        try:
            runpy.run_path(str(script), run_name="__main__")
        except StopAtPhysics:
            pass
        else:
            raise AssertionError("the scene script never reached sample_poses")
    target_dataset, target_ids = rec.loads[0]
    samples = []
    for key in rec.objects:
        values = rec.shader[key]
        grey = values.get("Base Color")
        samples.append(
            {
                "dataset": key[0],
                "id": key[1],
                "grey": None if grey is None else float(grey[0]),
                "roughness": float(values["Roughness"]),
                "specular_ior_level": float(values["Specular IOR Level"]),
            }
        )
    if len(set(rec.room)) != 1 or len(rec.room) != 5:
        raise AssertionError(f"unexpected room material calls {rec.room}")
    return {
        "dataset": target_dataset,
        "target_ids": target_ids,
        "distractors": {name: ids for name, ids in rec.loads[1:]},
        "objects": [list(k) for k in rec.objects],
        "material_samples": samples,
        "room_materials": rec.room_pool,
        "room_material": rec.room[0],
        "emission": float(rec.emission["strength"]),
        "emission_color": [float(c) for c in rec.emission["color"]],
        "point_light_energy": rec.light["energy"],
        "point_light_color": rec.light["color"],
        "point_light_position_m": rec.light["location"],
        "rng_state_sha256": rec.state_sha256,
    }


# Pools of the paper assets: BOP model ids of the distractor sets and the 16 ambientCG materials.
DISTRACTOR_IDS = {"tless": range(1, 31), "hb": range(1, 34), "tyol": range(1, 22)}
DISTRACTOR_FOLDERS = {"tless": "models_cad", "hb": "models", "tyol": "models"}
MATERIALS = (
    "Metal001",
    "Metal004",
    "Metal032",
    "Plastic001",
    "WoodFloor007",
    "WoodFloor018",
    "Bricks002",
    "Bricks020",
    "Tiles011",
    "Tiles046",
    "Concrete013",
    "Ground024",
    "Carpet004",
    "Fabric030",
    "Rocks006",
    "Marble001",
)
TARGETS = {"handal": range(1, 41), "hot3d": range(1, 34)}
SEEDS = {"handal": range(60000, 60010), "hot3d": range(70000, 70010)}


def _models_info(ids: range) -> dict[str, Any]:
    return {
        str(i): {
            "diameter": 100.0 + i,
            "min_x": -10.0 - i,
            "min_y": -20.0,
            "min_z": -5.0 * i,
            "size_x": 20.0 + 2 * i,
            "size_y": 40.0,
            "size_z": 10.0 * i,
        }
        for i in ids
    }


def write_fixture_assets(root: Path) -> Path:
    """Write a minimal asset tree (models_info files and material folders) under ``root``."""
    for dataset, ids in TARGETS.items():
        folder = root / "bop" / dataset / "models"
        folder.mkdir(parents=True)
        (folder / "models_info.json").write_text(json.dumps(_models_info(ids)))
    for dataset, ids in DISTRACTOR_IDS.items():
        folder = root / "bop" / dataset / DISTRACTOR_FOLDERS[dataset]
        folder.mkdir(parents=True)
        (folder / "models_info.json").write_text(json.dumps(_models_info(ids)))
    for name in MATERIALS:
        (root / "cctextures" / name).mkdir(parents=True)
        (root / "cctextures" / name / f"{name}_2K-JPG_Color.jpg").write_bytes(b"jpg")
    return root


SMALL_K = (40.0, 0.0, 16.0, 0.0, 40.0, 12.0, 0.0, 0.0, 1.0)


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"could not write {path}")


def make_attempt(
    root: Path,
    *,
    dataset: str,
    target: int,
    frames: int,
    size: tuple[int, int] = (32, 24),
    visible_offsets: dict[tuple[int, int], int] | None = None,
    amodal_offsets: dict[tuple[int, int], int] | None = None,
    target_visible: bool = True,
    seed: int = 70000,
) -> Path:
    """Write one rendered attempt in BlenderProc's ``write_bop`` layout (``bop_data/...``).

    Twenty instances (the target first, then objects 1..19) stand at fixed world poses in front
    of cameras moving along +z. ``visible_offsets`` adds to the writer's ``px_count_visib`` and
    ``amodal_offsets`` subtracts from its ``px_count_all`` for ``(frame, instance)``, emulating the
    writer's raster disagreement with the saved masks.

    Args:
        root: Attempt directory to create.
        dataset: BOP dataset name.
        target: Target object id.
        frames: Number of views.
        size: Image ``(width, height)``.
        visible_offsets: Visible-count perturbations.
        amodal_offsets: Amodal-count perturbations.
        target_visible: Draw the target's masks (else they stay empty).
        seed: Seed recorded in the scene record.

    Returns:
        ``root``.
    """
    width, height = size
    visible_offsets = visible_offsets or {}
    amodal_offsets = amodal_offsets or {}
    ids = [target] + [i for i in range(1, 21) if i != target][:19]
    scene_dir = root / "bop_data" / dataset / "train_pbr" / "000000"
    world = []
    for index, obj in enumerate(ids):
        matrix = np.eye(4)
        matrix[:3, 3] = [0.01 * index, -0.01 * index, 0.0]
        world.append({"dataset": dataset, "id": obj, "matrix": matrix.tolist()})
    poses, cameras, gt, info = [], {}, {}, {}
    for frame in range(frames):
        c2w = np.eye(4)
        c2w[:3, 3] = [0.0, 0.0, 1.0 + 0.1 * frame]
        poses.append(c2w.tolist())
        w2cv = np.linalg.inv(c2w @ np.diag([1, -1, -1, 1]))
        cameras[str(frame)] = {
            "cam_K": list(SMALL_K),
            "depth_scale": 1.0,
            "cam_R_w2c": w2cv[:3, :3].reshape(-1).tolist(),
            "cam_t_w2c": (w2cv[:3, 3] * 1000).tolist(),
        }
        depth = np.full((height, width), 1000, np.uint16)
        depth[:, 3] = 0
        _write_image(
            scene_dir / "rgb" / f"{frame:06d}.jpg", np.full((height, width, 3), 90, np.uint8)
        )
        _write_image(scene_dir / "depth" / f"{frame:06d}.png", depth)
        gt[str(frame)], info[str(frame)] = [], []
        for index, obj in enumerate(ids):
            matrix = np.asarray(world[index]["matrix"])
            rotation = w2cv[:3, :3] @ matrix[:3, :3]
            t = (w2cv[:3, :3] @ matrix[:3, 3] + w2cv[:3, 3]) * 1000
            gt[str(frame)].append(
                {"cam_R_m2c": rotation.reshape(-1).tolist(), "cam_t_m2c": t.tolist(), "obj_id": obj}
            )
            mask = np.zeros((height, width), np.uint8)
            visible = np.zeros_like(mask)
            x0 = 2 + index % 10
            if obj != target or target_visible:
                mask[2:8, x0 : x0 + 4] = 255
                visible[2:6, x0 : x0 + 4] = 255
            _write_image(scene_dir / "mask" / f"{frame:06d}_{index:06d}.png", mask)
            _write_image(scene_dir / "mask_visib" / f"{frame:06d}_{index:06d}.png", visible)
            n_all = int((mask > 0).sum())
            n_valid = int(((mask > 0) & (depth > 0)).sum())
            n_vis = int((visible > 0).sum())
            info[str(frame)].append(
                {
                    "bbox_obj": [x0, 2, 4, 6] if n_all else [-1, -1, -1, -1],
                    "bbox_visib": [x0, 2, 3, 3] if n_vis else [-1, -1, -1, -1],
                    "px_count_all": n_all - amodal_offsets.get((frame, index), 0),
                    "px_count_valid": n_valid,
                    "px_count_visib": n_vis + visible_offsets.get((frame, index), 0),
                    "visib_fract": n_vis / n_all if n_all else 0.0,
                }
            )
    for name, value in [("scene_gt", gt), ("scene_camera", cameras), ("scene_gt_info", info)]:
        (scene_dir / f"{name}.json").write_text(json.dumps(value, indent=2))
    (root / "bop_data" / dataset / "camera.json").write_text("{}")
    record = {
        "dataset": dataset,
        "target_id": target,
        "target_ids": ids,
        "frames": frames,
        "seed": seed,
        "image_size": [width, height],
        "cam_K": list(SMALL_K),
        "object_world_poses_m": world,
        "camera_world_poses_m": poses,
    }
    (root / "scene_record.json").write_text(json.dumps(record))
    receipt = {"render_seconds": 1.5, "total_seconds": 2.5, "frames": frames, "seed": seed}
    (root / "render_receipt.json").write_text(json.dumps(receipt))
    return root


def write_keypoints(path: Path) -> Path:
    """Write 64 small keypoints (meters) around the object origin."""
    points = np.random.default_rng(0).uniform(-0.01, 0.01, (64, 3)).tolist()
    path.write_text(json.dumps({"keypoints_3d": points}))
    return path
