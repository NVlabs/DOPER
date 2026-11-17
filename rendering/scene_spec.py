# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine-independent, per-sample deterministic scene descriptions.

A scene record fully determines what a renderer has to realize for sample ``index`` of a
dataset: object pose, camera pose, object material, world background, ground, distractors
(asset, placement, fallback material) and lights. Every decision is drawn from an RNG derived
from ``(seed, field, key)``, so

* the dataset does not depend on how it is sharded across workers or GPUs (any shard layout,
  any retry of a shard, reproduces the same scenes),
* two renderers (Blender, Isaac Sim) can realize the same scenes for a like-for-like
  comparison,
* ``scenes.jsonl`` doubles as provenance for what was rendered.

Cadence options (``change_material_every`` etc.) are honored by keying those fields on
``index // change_every`` instead of ``index``: consecutive samples in the same group share the
draw, as the historical per-worker loop did, without depending on worker boundaries.

Assets are referenced by path relative to the config's base directory (``assets/hdri/x.hdr``),
so records are portable between hosts as long as the asset packs match
(``AssetCatalog.fingerprint``).
"""

from __future__ import annotations

import colorsys
import glob
import hashlib
import json
import math
import os
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from doper.scene import geometry as G  # noqa: N812 - short alias used across the sampling code
from rendering.config import RenderConfig, sample_object_material
from rendering.robust_appearance import begin_appearance, finish_appearance
from rendering.robustness import augment_scene
from rendering.usd_mesh import read_mesh_usda

# Version 3 samples uniform rotations with SciPy and a private PCG64 stream.
# This changes seeded poses and RNG consumption from v1/v2; never mix on resume.
SCENE_SPEC_VERSION = 3

HDRI_PATTERNS = ("*.hdr", "*.exr")
DISTRACTOR_PATTERNS = ("*.usd", "*.usda", "*.usdc", "*.usdz")
# Distractor size, distance and height draws, relative to the object's scene scale.
DISTRACTOR_SIZE_FRACTION = (0.1, 0.8)
DISTRACTOR_DISTANCE = (0.05, 0.20)
DISTRACTOR_HEIGHT = (-0.01, 0.05)
# Object size (meters) at which the scene scale is 1.
REFERENCE_OBJECT_SIZE_M = 0.1


def _glob_rel(root: str, sub: str, patterns: Sequence[str]) -> list[str]:
    """Sorted ``root``-relative files in ``root/sub`` matching a pattern; empty ``sub`` is none."""
    directory = os.path.join(root, sub)
    if not sub or not os.path.isdir(directory):
        return []
    files: set[str] = set()
    for pattern in patterns:
        files.update(glob.glob(os.path.join(directory, pattern)))
    return sorted(os.path.relpath(f, root) for f in files)


def _texture_sets(root: str, sub: str) -> list[str]:
    """Sorted ``root``-relative subdirectories of ``root/sub``; empty ``sub`` is none."""
    directory = os.path.join(root, sub)
    if not sub or not os.path.isdir(directory):
        return []
    found = (d for d in glob.glob(os.path.join(directory, "*")) if os.path.isdir(d))
    return sorted(os.path.relpath(d, root) for d in found)


@dataclass(frozen=True)
class AssetCatalog:
    """Sorted, root-relative asset lists the sampler chooses from."""

    root: str
    hdris: tuple[str, ...] = ()
    texture_sets: tuple[str, ...] = ()
    distractors: tuple[str, ...] = ()

    @classmethod
    def scan(cls, cfg: RenderConfig) -> AssetCatalog:
        """List the HDRIs, texture sets and distractors under the config's asset directories."""
        root = cfg.base_dir
        return cls(
            root=root,
            hdris=tuple(_glob_rel(root, cfg.hdri_dir, HDRI_PATTERNS)),
            texture_sets=tuple(_texture_sets(root, cfg.texture_dir)),
            distractors=tuple(_glob_rel(root, cfg.distractor_dir, DISTRACTOR_PATTERNS)),
        )

    @property
    def fingerprint(self) -> str:
        """SHA-256 over the three sorted name lists."""
        digest = hashlib.sha256()
        for group in (self.hdris, self.texture_sets, self.distractors):
            digest.update("\n".join(group).encode() + b"\0")
        return digest.hexdigest()

    def summary(self) -> dict[str, Any]:
        """Counts per group plus the fingerprint."""
        return {
            "hdris": len(self.hdris),
            "texture_sets": len(self.texture_sets),
            "distractors": len(self.distractors),
            "fingerprint": self.fingerprint,
        }

    def resolve(self, rel: str) -> str:
        """Absolute path of a catalog entry."""
        return os.path.join(self.root, rel)


@dataclass(frozen=True)
class MeshInfo:
    """Object geometry the sampler needs (from the USD, identical for every renderer)."""

    local_corners: np.ndarray  # (8, 3) AABB corners in the object frame
    mesh_size: float  # largest AABB extent, meters
    kp_local: np.ndarray  # (K, 3) keypoints in the object frame

    @classmethod
    def from_points(cls, points: np.ndarray, kp_local: np.ndarray) -> MeshInfo:
        """Build from mesh vertices and keypoints in the object frame."""
        points = np.asarray(points, dtype=float)
        return cls(
            local_corners=G.aabb_corners(points),
            mesh_size=float(np.max(points.max(axis=0) - points.min(axis=0))),
            kp_local=np.asarray(kp_local, dtype=float),
        )

    @property
    def scene_scale(self) -> float:
        """Mesh size relative to a 10 cm object."""
        return self.mesh_size / REFERENCE_OBJECT_SIZE_M


def stream(seed: int, name: str, key: int) -> random.Random:
    """Independent RNG for one decision of one sample (or cadence group)."""
    digest = hashlib.sha256(f"{int(seed)}:{name}:{int(key)}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _hsv(h: float, s: float, v: float) -> list[float]:
    return [float(c) for c in colorsys.hsv_to_rgb(h, s, v)]


def _mat(m: np.ndarray) -> list[list[float]]:
    return [[float(v) for v in row] for row in np.asarray(m)]


def _keypoints_in_frame(
    cam_to_world: np.ndarray, object_pose: np.ndarray, mesh: MeshInfo, intr: G.Intrinsics
) -> int:
    uv, depth = G.project_points(G.transform_points(object_pose, mesh.kp_local), cam_to_world, intr)
    return int(np.count_nonzero(G.in_frame(uv, depth, intr)))


def sample_object_and_camera(
    cfg: RenderConfig,
    index: int,
    seed: int,
    mesh: MeshInfo,
    intr: G.Intrinsics,
    cam_dist_range: Sequence[float],
) -> dict[str, Any]:
    """Object pose resting on the ground and a camera that looks at it."""
    rng = stream(seed, "pose", index)
    rotation, is_biased = G.sample_object_rotation(rng, cfg.axis_bias, cfg.axis_jiggle_deg)
    scale = mesh.scene_scale
    clearance = rng.uniform(*cfg.ground_clearance) * scale
    jitter = cfg.xy_jitter * scale
    object_pose = G.rest_on_ground(
        rotation,
        mesh.local_corners,
        clearance,
        (rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)),
    )
    world_corners = G.transform_points(object_pose, mesh.local_corners)
    center = world_corners.mean(axis=0)
    rng = stream(seed, "camera", index)
    cam_dist = rng.uniform(*cam_dist_range)
    jitter_strength = rng.uniform(*cfg.image_jitter)
    fallback = False
    if is_biased:
        direction = G.top_down_camera_direction(rng, cfg.top_down_jiggle_deg)
        cam_to_world = G.place_camera(rng, center, direction, cam_dist, jitter_strength)
        too_few = _keypoints_in_frame(cam_to_world, object_pose, mesh, intr) < cfg.min_keypoints
        if cfg.fallback_to_random and too_few:
            direction = G.random_camera_direction(rng)
            cam_to_world = G.place_camera(rng, center, direction, cam_dist, jitter_strength)
            fallback = True
    else:
        direction = G.random_camera_direction(rng)
        cam_to_world = G.place_camera(rng, center, direction, cam_dist, jitter_strength)
    return {
        "object": {
            "pose": _mat(object_pose),
            "axis_biased": bool(is_biased),
            "center": [float(v) for v in center],
            "bbox_min_z": float(world_corners[:, 2].min()),
        },
        "camera": {
            "cam_to_world": _mat(cam_to_world),
            "distance": float(cam_dist),
            "jitter_strength": float(jitter_strength),
            "fallback_to_random": fallback,
        },
    }


def sample_material(cfg: RenderConfig, index: int, seed: int) -> dict[str, Any]:
    """Object material for the cadence group containing ``index``."""
    rng = stream(seed, "material", index // cfg.change_material_every)
    inputs = sample_object_material(rng, cfg)
    return {
        "color": [float(c) for c in inputs["diffuse_color_constant"]],
        "roughness": float(inputs["reflection_roughness_constant"]),
        "metallic": float(inputs["metallic_constant"]),
    }


def sample_background(
    cfg: RenderConfig, index: int, seed: int, catalog: AssetCatalog
) -> dict[str, Any]:
    """World background (HDRI, dark, random color or studio) for the cadence group."""
    rng = stream(seed, "background", index // cfg.change_hdri_every)
    mode = rng.choice(list(cfg.world_background_modes))
    if mode == "hdri" and catalog.hdris:
        return {
            "mode": "hdri",
            "hdri": rng.choice(catalog.hdris),
            "intensity": float(rng.uniform(*cfg.dome_intensity)),
            "rotation_deg": float(rng.uniform(0, 360)),
        }
    if mode == "dark":
        color = _hsv(rng.random(), rng.uniform(0.0, 0.25), rng.uniform(0.0, 0.08))
        intensity = rng.uniform(*cfg.dome_intensity) * 0.5
    elif mode in {"random", "hdri"}:
        # hdri requested but none available -> random color dome, as the legacy worker did. The
        # record says what is rendered, so the mode becomes "random" (renderers branch on it).
        mode = "random"
        color = _hsv(rng.random(), rng.uniform(0.05, 0.35), rng.uniform(0.25, 0.85))
        intensity = rng.uniform(*cfg.dome_intensity)
    else:
        mode, color, intensity = "studio", [0.78, 0.80, 0.82], 0.8 * cfg.dome_intensity[0]
    return {"mode": mode, "color": color, "intensity": float(intensity)}


def sample_ground(
    cfg: RenderConfig, index: int, seed: int, catalog: AssetCatalog
) -> dict[str, Any]:
    """Ground plane (absent, textured or plain color) for the cadence group."""
    rng = stream(seed, "ground", index // cfg.change_ground_every)
    if cfg.no_ground_prob > 0.0 and rng.random() < cfg.no_ground_prob:
        return {"enabled": False}
    use_texture = (
        cfg.ground_style in {"texture", "mixed"}
        and catalog.texture_sets
        and (cfg.ground_style == "texture" or rng.random() < 0.5)
    )
    if use_texture:
        return {
            "enabled": True,
            "kind": "texture",
            "texture_set": rng.choice(catalog.texture_sets),
            "texture_scale": float(rng.uniform(2.0, 8.0)),
            "texture_rotate_deg": float(rng.uniform(0, 360)),
        }
    return {
        "enabled": True,
        "kind": "color",
        "color": _hsv(rng.random(), rng.uniform(0.0, 0.5), rng.uniform(0.15, 0.9)),
        "roughness": float(rng.uniform(0.3, 0.9)),
    }


def sample_distractor_material(rng: random.Random) -> dict[str, Any]:
    """Random plastic, metal, rubber, ceramic or wood look (the Blender worker's families)."""
    kind = rng.choice(["plastic", "metal", "rubber", "ceramic", "wood"])
    if kind == "plastic":
        color, metallic, rough = (
            [rng.uniform(0.1, 1.0) for _ in range(3)],
            0.0,
            rng.uniform(0.2, 0.6),
        )
    elif kind == "metal":
        g = rng.uniform(0.5, 0.95)
        tint = rng.choice(["neutral", "gold", "copper", "blue"])
        color = {
            "neutral": [g, g, g],
            "gold": [g, g * 0.85, g * 0.5],
            "copper": [g, g * 0.7, g * 0.6],
            "blue": [g * 0.8, g * 0.85, g],
        }[tint]
        metallic, rough = 1.0, rng.uniform(0.1, 0.5)
    elif kind == "rubber":
        color, metallic, rough = (
            [rng.uniform(0.02, 0.3) for _ in range(3)],
            0.0,
            rng.uniform(0.7, 1.0),
        )
    elif kind == "ceramic":
        color, metallic, rough = (
            [rng.uniform(0.7, 1.0) for _ in range(3)],
            0.0,
            rng.uniform(0.05, 0.2),
        )
    else:
        brown = rng.uniform(0.3, 0.7)
        color = [brown, brown * rng.uniform(0.5, 0.8), brown * rng.uniform(0.3, 0.5)]
        metallic, rough = 0.0, rng.uniform(0.4, 0.8)
    return {
        "kind": kind,
        "color": [float(c) for c in color],
        "metallic": float(metallic),
        "roughness": float(rough),
    }


def sample_distractors(
    cfg: RenderConfig,
    index: int,
    seed: int,
    catalog: AssetCatalog,
    center: Sequence[float],
    scene_scale: float,
) -> list[dict[str, Any]]:
    """Distractor picks and placements around ``center``.

    ``size_frac`` is the target size relative to the object (the renderer divides by the
    asset's own extent, which it only knows after loading). ``material`` is the fallback look
    for assets whose textures do not resolve; the renderer keeps authored materials when they
    are valid.
    """
    if not catalog.distractors:
        return []
    rng = stream(seed, "distractors", index // cfg.change_distractors_every)
    origin = np.asarray(center, dtype=float)
    out = []
    for _ in range(min(rng.randint(*cfg.num_distractors), len(catalog.distractors))):
        rel = rng.choice(catalog.distractors)
        size_frac = rng.uniform(*DISTRACTOR_SIZE_FRACTION)
        angle = rng.uniform(0, 2 * math.pi)
        dist = rng.uniform(*DISTRACTOR_DISTANCE) * scene_scale
        height = rng.uniform(*DISTRACTOR_HEIGHT) * scene_scale
        position = origin + np.array([math.cos(angle) * dist, math.sin(angle) * dist, height])
        rot = [rng.uniform(0, 2 * math.pi) for _ in range(3)]
        out.append(
            {
                "file": rel,
                "size_frac": float(size_frac),
                "position": [float(v) for v in position],
                "rotation_xyz": [float(v) for v in rot],
                "material": sample_distractor_material(rng),
            }
        )
    return out


def sample_lights(
    cfg: RenderConfig, index: int, seed: int, center: Sequence[float], scene_scale: float
) -> list[dict[str, Any]]:
    """Zero to three area or point lights around ``center``."""
    rng = stream(seed, "lights", index)
    origin = np.asarray(center, dtype=float)
    s = max(1.0, scene_scale)
    out = []
    for _ in range(rng.randint(0, 3)):
        kind = rng.choice(["sphere", "rect", "disk"])
        offset = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(0.1, 0.6)])
        out.append(
            {
                "kind": kind,
                "position": [float(v) for v in origin + offset * s],
                "color": [float(rng.uniform(0.7, 1.0)) for _ in range(3)],
                "intensity": float(rng.uniform(*cfg.light_intensity)),
                "scale": float(rng.uniform(0.05, 0.5) * s),
            }
        )
    return out


def sample_scene(
    cfg: RenderConfig,
    index: int,
    seed: int,
    catalog: AssetCatalog,
    mesh: MeshInfo,
    intr: G.Intrinsics,
    cam_dist_range: Sequence[float],
) -> dict[str, Any]:
    """The complete scene record for sample ``index``."""
    record: dict[str, Any] = {"version": SCENE_SPEC_VERSION, "index": int(index), "seed": int(seed)}
    record.update(sample_object_and_camera(cfg, index, seed, mesh, intr, cam_dist_range))
    center, scale = record["object"]["center"], mesh.scene_scale
    record["material"] = sample_material(cfg, index, seed)
    record["background"] = sample_background(cfg, index, seed, catalog)
    record["ground"] = sample_ground(cfg, index, seed, catalog)
    record["distractors"] = sample_distractors(cfg, index, seed, catalog, center, scale)
    record["lights"] = sample_lights(cfg, index, seed, center, scale)
    appearance = begin_appearance(
        record, cfg.robust_appearance, stream(seed, "metal-stress", index)
    )
    if cfg.robustness is not None:
        augment_scene(record, cfg.robustness, mesh, intr, stream(seed, "robustness", index))
    finish_appearance(record, appearance)
    return record


def camera_distance_range(cfg: RenderConfig, mesh: MeshInfo) -> Sequence[float]:
    """Camera distance range: framing-derived when ``auto_distance`` is set, else configured."""
    if cfg.auto_distance:
        return G.auto_camera_distance(
            mesh.mesh_size, cfg.focal_length_mm, cfg.sensor_width_mm, cfg.fill_factor
        )
    return cfg.distance


def generate_scenes(
    cfg: RenderConfig,
    *,
    usd_path: str,
    keypoints_json: str | None,
    seed: int,
    start_idx: int,
    end_idx: int,
    out_path: str,
) -> dict[str, Any]:
    """Sample and write the scene records for ``[start_idx, end_idx)``; shared by both launchers.

    Geometry comes from the object USD (the same file both renderers use), so a Blender and an
    Isaac render of the same seed get byte-identical records.

    Args:
        cfg: Render configuration.
        usd_path: Object USDA written by ``usd_mesh.write_mesh_usda``.
        keypoints_json: Keypoints file overriding the USD's keypoints, or ``None``.
        seed: Dataset seed.
        start_idx: First sample index.
        end_idx: One past the last sample index.
        out_path: JSON-lines file to write.

    Returns:
        ``path``, ``count``, the asset ``catalog``, the ``mesh`` and ``num_keypoints``.
    """
    mesh_data = read_mesh_usda(usd_path)
    kp_local = np.asarray(mesh_data["keypoints_3d"], dtype=float)
    if keypoints_json:
        with open(keypoints_json) as handle:
            kp_local = np.asarray(json.load(handle)["keypoints_3d"], dtype=float)
    mesh = MeshInfo.from_points(np.asarray(mesh_data["points"], dtype=float), kp_local)
    intr = G.intrinsics_from_lens(
        cfg.resolution, cfg.resolution, cfg.focal_length_mm, cfg.sensor_width_mm
    )
    catalog = AssetCatalog.scan(cfg)
    cam_dist = camera_distance_range(cfg, mesh)
    records = (
        sample_scene(cfg, i, seed, catalog, mesh, intr, cam_dist) for i in range(start_idx, end_idx)
    )
    count = write_scenes(out_path, records)
    return {
        "path": out_path,
        "count": count,
        "catalog": catalog,
        "mesh": mesh,
        "num_keypoints": len(kp_local),
    }


def write_scenes(path: str, records: Iterable[dict[str, Any]]) -> int:
    """Write records as JSON lines (one scene per line, in index order); return the count."""
    n = 0
    with open(path, "w") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            n += 1
    return n


def read_scene_records(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read a JSON-lines scene file into its records, in file order."""
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_scenes(path: str | os.PathLike[str]) -> dict[int, dict[str, Any]]:
    """Read a JSON-lines scene file into ``{index: record}``."""
    return {record["index"]: record for record in read_scene_records(path)}
