# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render configuration and the pure-NumPy plan of one BlenderProc layout.

This module runs both on the host and inside BlenderProc's Blender, so it imports only the
standard library and NumPy.

A layout's random draws split in two. Everything drawn before ``sample_poses`` (the 20 target
instances, the six BOP distractors, the per-object surface values, the room material, the
ceiling emission and the point light) depends only on the seed and the asset pools.
:func:`plan_scene` makes exactly those draws, in the source renderer's order, from a legacy
``RandomState(seed)``; the stream equals the source's ``np.random.seed(seed)`` global stream.
Drop poses, physics and camera sampling depend on collisions and the BVH and stay in
``scene.realize``, which resumes the global RNG from :attr:`ScenePlan.rng_state`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
TARGET_INSTANCES = 20
MAX_FRAMES_PER_LAYOUT = 25
GREY_RANGE = (0.1, 0.9)
EMISSION_RANGE = (3.0, 6.0)
EMISSION_COLOR_RANGE = ([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0])
LIGHT_COLOR_RANGE = ([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
LIGHT_ENERGY_W = 200
LIGHT_RADIUS_M = (1.0, 1.5)
LIGHT_ELEVATION_DEG = (5, 89)
COLOR_FILE_SUFFIXES = ("_2K_Color.jpg", "_2K-JPG_Color.jpg")


@dataclass(frozen=True)
class DistractorSpec:
    """BOP distractor set: ``count`` objects drawn from ``bop/<dataset>/models[_<type>]``."""

    dataset: str
    model_type: str
    count: int

    @property
    def models_folder(self) -> str:
        """Model folder name (``models`` or ``models_<type>``)."""
        return f"models_{self.model_type}" if self.model_type else "models"


@dataclass(frozen=True)
class CameraSpec:
    """Pinhole camera and the shell the views are sampled from."""

    K: tuple[float, ...]  # noqa: N815 - standard name of the intrinsics matrix
    width: int
    height: int
    radius_m: tuple[float, float]
    elevation_deg: tuple[float, float]
    inplane_rad: tuple[float, float]


@dataclass(frozen=True)
class UnitCheck:
    """Loaded-mesh unit check against ``models_info``.

    ``extent``: the extent matches ``size_*`` (HANDAL). ``min_extent``: the minimum corner matches
    ``min_*`` too, so the origin was kept (HOT3D).
    """

    mode: str
    atol_mm: float
    rtol: float


@dataclass(frozen=True)
class RenderConfig:
    """One generated object's render recipe and campaign (``configs/<name>.json``)."""

    name: str
    dataset: str
    target_object: int
    control_objects: tuple[int, int]
    others_per_layout: int
    distractors: tuple[DistractorSpec, ...]
    blenderproc: str
    blender: str
    bop_toolkit: str
    camera: CameraSpec
    samples: int
    view_transform: str
    look: str
    jpeg_quality: int
    depth_scale_mm_per_uint: float
    render_threads: int
    layouts: int
    frames_per_layout: int
    base_seed: int
    workers: int
    val_modulo: int
    val_residue: int
    count_bound_px: int
    count_bound_attempts: int
    unit_check: UnitCheck
    room_materials_min: int
    room_materials_max: int | None
    keypoints_sha256: str
    overlay_frames: tuple[int, ...]
    provenance: str
    # ``keypoints_sha256`` pins the release selection (``doper prepare``); this is the research
    # file of the same ordered layout, whose extra report floats depend on the BLAS kernel.
    source_keypoints_sha256: str | None = None

    def __post_init__(self) -> None:
        """Validate the fields a campaign depends on."""
        low, high = self.control_objects
        pool = [i for i in range(low, high + 1) if i != self.target_object]
        if self.others_per_layout + 1 != TARGET_INSTANCES or len(pool) < self.others_per_layout:
            raise ValueError(f"{self.name}: {TARGET_INSTANCES} target instances need a larger pool")
        if not 1 <= self.frames_per_layout <= MAX_FRAMES_PER_LAYOUT:
            raise ValueError(f"{self.name}: one layout holds 1..{MAX_FRAMES_PER_LAYOUT} views")
        if self.layouts < 1 or self.workers < 1 or self.count_bound_attempts < 1:
            raise ValueError(f"{self.name}: layouts, workers and attempts must be positive")
        if self.count_bound_px < 0 or not 0 <= self.val_residue < self.val_modulo:
            raise ValueError(f"{self.name}: invalid count bound or split")
        if self.unit_check.mode not in {"extent", "min_extent"}:
            raise ValueError(f"{self.name}: unknown unit check {self.unit_check.mode!r}")
        if len(self.camera.K) != 9:
            raise ValueError(f"{self.name}: camera K must have 9 entries")

    def as_json(self) -> dict[str, Any]:
        """Return the JSON form read by :func:`load_config`."""
        data = asdict(self)
        data["color_management"] = {
            "view_transform": data.pop("view_transform"),
            "look": data.pop("look"),
        }
        data["split"] = {"modulo": data.pop("val_modulo"), "val_residue": data.pop("val_residue")}
        data["room_materials"] = {
            "minimum": data.pop("room_materials_min"),
            "maximum": data.pop("room_materials_max"),
        }
        return json.loads(json.dumps(data))


def config_from_json(data: dict[str, Any]) -> RenderConfig:
    """Build a :class:`RenderConfig` from its JSON form.

    Args:
        data: Parsed ``configs/<name>.json``.

    Returns:
        The validated config.
    """
    camera = data["camera"]
    return RenderConfig(
        name=data["name"],
        dataset=data["dataset"],
        target_object=int(data["target_object"]),
        control_objects=(int(data["control_objects"][0]), int(data["control_objects"][1])),
        others_per_layout=int(data["others_per_layout"]),
        distractors=tuple(
            DistractorSpec(d["dataset"], d["model_type"], int(d["count"]))
            for d in data["distractors"]
        ),
        blenderproc=data["blenderproc"],
        blender=data["blender"],
        bop_toolkit=data["bop_toolkit"],
        camera=CameraSpec(
            K=tuple(float(v) for v in camera["K"]),
            width=int(camera["width"]),
            height=int(camera["height"]),
            radius_m=(float(camera["radius_m"][0]), float(camera["radius_m"][1])),
            elevation_deg=(camera["elevation_deg"][0], camera["elevation_deg"][1]),
            inplane_rad=(float(camera["inplane_rad"][0]), float(camera["inplane_rad"][1])),
        ),
        samples=int(data["samples"]),
        view_transform=data["color_management"]["view_transform"],
        look=data["color_management"]["look"],
        jpeg_quality=int(data["jpeg_quality"]),
        depth_scale_mm_per_uint=float(data["depth_scale_mm_per_uint"]),
        render_threads=int(data["render_threads"]),
        layouts=int(data["layouts"]),
        frames_per_layout=int(data["frames_per_layout"]),
        base_seed=int(data["base_seed"]),
        workers=int(data["workers"]),
        val_modulo=int(data["split"]["modulo"]),
        val_residue=int(data["split"]["val_residue"]),
        count_bound_px=int(data["count_bound_px"]),
        count_bound_attempts=int(data["count_bound_attempts"]),
        unit_check=UnitCheck(**data["unit_check"]),
        room_materials_min=int(data["room_materials"]["minimum"]),
        room_materials_max=data["room_materials"]["maximum"],
        keypoints_sha256=data["keypoints_sha256"],
        overlay_frames=tuple(int(v) for v in data["overlay_frames"]),
        provenance=data["provenance"],
        source_keypoints_sha256=data.get("source_keypoints_sha256"),
    )


def load_config(name_or_path: str | Path) -> RenderConfig:
    """Load a packaged config by name (``handal_obj40``, ``hot3d_obj33``) or a JSON path.

    Args:
        name_or_path: Config name or path.

    Returns:
        The validated config.
    """
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIG_DIR / f"{name_or_path}.json"
        if not path.is_file():
            known = sorted(p.stem for p in CONFIG_DIR.glob("*.json"))
            raise ValueError(f"unknown render config {name_or_path!r}; known: {known}")
    return config_from_json(json.loads(path.read_text()))


@dataclass(frozen=True)
class ScenePools:
    """Everything a plan draws from.

    Attributes:
        target_candidates: Control objects of the target dataset (the target excluded).
        distractors: Sorted model ids per distractor dataset (from ``models_info.json``).
        room_materials: ambientCG material names, sorted, so the choice does not depend on the
            directory listing order BlenderProc loads them in.
    """

    target_candidates: tuple[int, ...]
    distractors: dict[str, tuple[int, ...]]
    room_materials: tuple[str, ...]


def material_names(folder: Path) -> tuple[str, ...]:
    """Sorted names of the ambientCG material folders (those with a 2K color texture)."""
    return tuple(
        sorted(
            p.name
            for p in folder.iterdir()
            if p.is_dir() and any((p / f"{p.name}{s}").is_file() for s in COLOR_FILE_SUFFIXES)
        )
    )


def load_pools(assets: Path, config: RenderConfig) -> ScenePools:
    """Read the pools of a staged asset tree.

    Args:
        assets: Asset root (``bop/<dataset>/...``, ``cctextures/``).
        config: Render config.

    Returns:
        The pools.
    """
    low, high = config.control_objects
    distractors = {}
    for spec in config.distractors:
        info = assets / "bop" / spec.dataset / spec.models_folder / "models_info.json"
        distractors[spec.dataset] = tuple(sorted(map(int, json.loads(info.read_text()))))
    return ScenePools(
        target_candidates=tuple(i for i in range(low, high + 1) if i != config.target_object),
        distractors=distractors,
        room_materials=material_names(assets / "cctextures"),
    )


@dataclass(frozen=True)
class MaterialSample:
    """Surface values of one loaded object (``gray`` only for T-LESS)."""

    dataset: str
    id: int
    grey: float | None
    roughness: float
    specular_ior_level: float


@dataclass(frozen=True)
class ScenePlan:
    """Every pre-physics draw of one layout.

    Attributes:
        seed: Layout seed.
        dataset: Target dataset.
        target_ids: The 20 target-dataset ids in load order (shuffled).
        distractors: Drawn ids per distractor dataset, in load order.
        object_order: Permutation of the loaded objects (targets, then distractors) giving the
            scene's object order.
        material_samples: Surface values in scene object order.
        room_material: Name of the material shared by the five room planes.
        emission: Ceiling emission strength.
        emission_color: Ceiling emission RGBA.
        light_color: Point-light RGB.
        light_position: Point-light position (m).
        rng_state: Legacy NumPy RNG state after the last draw.
    """

    seed: int
    dataset: str
    target_ids: tuple[int, ...]
    distractors: dict[str, tuple[int, ...]]
    object_order: tuple[int, ...]
    material_samples: tuple[MaterialSample, ...]
    room_material: str
    emission: float
    emission_color: tuple[float, ...]
    light_color: tuple[float, ...]
    light_position: tuple[float, ...]
    rng_state: tuple = field(repr=False, compare=False)

    @property
    def loaded(self) -> list[tuple[str, int]]:
        """``(dataset, id)`` of the objects in load order: targets, then each distractor set."""
        rows = [(self.dataset, i) for i in self.target_ids]
        return rows + [(name, i) for name, ids in self.distractors.items() for i in ids]

    def as_record(self) -> dict[str, Any]:
        """Return the JSON record written to ``scene_record.json`` (no RNG state array)."""
        loaded = self.loaded
        return {
            "seed": self.seed,
            "dataset": self.dataset,
            "target_ids": list(self.target_ids),
            "distractors": {k: list(v) for k, v in self.distractors.items()},
            "objects": [list(loaded[i]) for i in self.object_order],
            "material_samples": [asdict(s) for s in self.material_samples],
            "room_material": self.room_material,
            "emission": self.emission,
            "emission_color": list(self.emission_color),
            "point_light_energy": LIGHT_ENERGY_W,
            "point_light_color": list(self.light_color),
            "point_light_position_m": list(self.light_position),
            "rng_state_sha256": rng_state_sha256(self.rng_state),
        }


def rng_state_sha256(state: tuple) -> str:
    """Digest of a legacy NumPy RNG state (key array, position and Gaussian cache).

    Args:
        state: ``RandomState.get_state()`` tuple.

    Returns:
        Lowercase hex digest.
    """
    name, keys, pos, has_gauss, cached = state
    digest = hashlib.sha256()
    digest.update(str(name).encode())
    digest.update(np.asarray(keys, dtype=np.uint32).tobytes())
    digest.update(f"{int(pos)}|{int(has_gauss)}|{float(cached)!r}".encode())
    return digest.hexdigest()


def sample_shell(
    rng: np.random.RandomState,
    center: list[float],
    radius_min: float,
    radius_max: float,
    elevation_min: float,
    elevation_max: float,
    azimuth_min: float = -180,
    azimuth_max: float = 180,
) -> np.ndarray:
    """Sample a point between two spheres, as BlenderProc 2.8.0 ``sampler.shell`` does.

    Same draws and arithmetic as the sampler's default (not volume-uniform) mode: elevation,
    azimuth, then radius.

    Args:
        rng: Random state (consumes three draws).
        center: Shared sphere center.
        radius_min: Inner radius.
        radius_max: Outer radius.
        elevation_min: Minimum elevation (degrees).
        elevation_max: Maximum elevation (degrees).
        azimuth_min: Minimum azimuth (degrees).
        azimuth_max: Maximum azimuth (degrees).

    Returns:
        The sampled position.
    """
    elevation = np.deg2rad(elevation_min + (elevation_max - elevation_min) * rng.rand())
    azimuth = np.deg2rad(azimuth_min + (azimuth_max - azimuth_min) * rng.rand())
    direction = np.array(
        [
            np.sin(np.pi / 2 - elevation) * np.cos(azimuth),
            np.sin(np.pi / 2 - elevation) * np.sin(azimuth),
            np.cos(np.pi / 2 - elevation),
        ]
    )
    radius = rng.uniform(radius_min, radius_max)
    return direction * radius + np.array(center)


def plan_scene(seed: int, pools: ScenePools, config: RenderConfig) -> ScenePlan:
    """Make every draw of one layout up to ``sample_poses``, in the source renderer's order.

    Args:
        seed: Layout seed (``base_seed + layout``).
        pools: Asset pools; ``room_materials`` must be sorted.
        config: Render config.

    Returns:
        The plan, including the RNG state the scene resumes from.
    """
    if list(pools.room_materials) != sorted(pools.room_materials):
        raise ValueError("room materials must be sorted by name")
    rng = np.random.RandomState(seed)
    ids = [config.target_object] + rng.choice(
        list(pools.target_candidates), size=config.others_per_layout, replace=False
    ).tolist()
    rng.shuffle(ids)
    distractors = {}
    for spec in config.distractors:
        available = list(pools.distractors[spec.dataset])
        distractors[spec.dataset] = tuple(rng.choice(available, spec.count, replace=False).tolist())
    loaded = [(config.dataset, i) for i in ids]
    loaded += [(name, i) for name, picked in distractors.items() for i in picked]
    order = list(range(len(loaded)))
    rng.shuffle(order)
    samples = []
    for index in order:
        dataset, obj_id = loaded[index]
        grey = float(rng.uniform(*GREY_RANGE)) if dataset == "tless" else None
        roughness = float(rng.uniform(0, 1))
        specular = float(rng.uniform(0, 1))
        samples.append(MaterialSample(dataset, obj_id, grey, roughness, specular))
    room_material = str(rng.choice(list(pools.room_materials)))
    emission = float(rng.uniform(*EMISSION_RANGE))
    emission_color = rng.uniform(*EMISSION_COLOR_RANGE).tolist()
    light_color = rng.uniform(*LIGHT_COLOR_RANGE)
    light_position = sample_shell(rng, [0, 0, 0], *LIGHT_RADIUS_M, *LIGHT_ELEVATION_DEG)
    return ScenePlan(
        seed=seed,
        dataset=config.dataset,
        target_ids=tuple(ids),
        distractors=distractors,
        object_order=tuple(order),
        material_samples=tuple(samples),
        room_material=room_material,
        emission=emission,
        emission_color=tuple(emission_color),
        light_color=tuple(float(v) for v in light_color),
        light_position=tuple(float(v) for v in light_position),
        rng_state=rng.get_state(),  # pyright: ignore[reportArgumentType]
    )
