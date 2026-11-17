# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed view over the YAML render configs.

Mirrors the defaults used by ``rendering/synth_worker.py`` so a config that
drives the Blender renderer drives the Isaac renderer the same way. Options
that have no Isaac equivalent (EEVEE settings, procedural node materials) are
ignored; Isaac-only knobs live under an optional ``isaac:`` section.

Relative paths in a config resolve against the config file's directory
(:attr:`RenderConfig.base_dir`). Asset directories are optional: an absent or empty
key means "none" (procedural scene only), while a configured directory that does not
exist is an error.
"""

from __future__ import annotations

import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rendering.isaac.defaults import resolve_settings
from rendering.paths import resolve_under
from rendering.robust_appearance import validate_robust_appearance
from rendering.robustness import validate_robustness


def _range(
    value: Any,  # noqa: ANN401 - YAML scalar, string or sequence
    default: tuple[float, float],
) -> tuple[float, float]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return (float(value), float(value))
    if isinstance(value, str):
        parts = [p for p in value.replace("[", "").replace("]", "").split(",") if p.strip()]
        value = [float(p) for p in parts]
    seq = [float(v) for v in value]
    if len(seq) == 1:
        return (seq[0], seq[0])
    if len(seq) != 2:
        raise ValueError(f"expected a 2-element range, got {value!r}")
    return (min(seq), max(seq))


_NAMED_COLORS = {"orange": (1.0, 0.34, 0.02), "green": (0.05, 0.62, 0.18)}


def _color(value: Any) -> tuple[float, float, float]:  # noqa: ANN401 - YAML name or channels
    """One color as RGB in [0, 1], as ``synth_worker.parse_rgb_color`` (names, 0-1 or 0-255)."""
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in _NAMED_COLORS:
            raise ValueError(f"unknown object color {value!r}")
        return _NAMED_COLORS[key]
    channels = [float(c) for c in value]
    if len(channels) not in (3, 4):
        raise ValueError(f"object colors must have 3 or 4 channels, got {value!r}")
    if max(channels) > 1.0:
        channels = [c / 255.0 for c in channels]
    return (channels[0], channels[1], channels[2])


def _color_choices(
    value: Any,  # noqa: ANN401 - YAML list or comma-separated string
) -> tuple[tuple[float, float, float], ...]:
    if not value:
        return ()
    if isinstance(value, str):
        value = [v.strip() for v in value.split(",") if v.strip()]
    return tuple(_color(v) for v in value)


def _bool(value: Any, default: bool = False) -> bool:  # noqa: ANN401 - YAML scalar
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _asset_dir(scene: Mapping[str, Any], key: str) -> str:
    """Configured asset directory; an absent key or empty string means none."""
    value = scene.get(key)
    return "" if value is None else str(value)


def _check_asset_dirs(cfg: RenderConfig) -> None:
    """Raise ``FileNotFoundError`` naming the key of a configured directory that is missing."""
    for key in ("hdri_dir", "texture_dir", "distractor_dir"):
        resolved = cfg.resolve(getattr(cfg, key))
        if resolved is not None and not os.path.isdir(resolved):
            raise FileNotFoundError(f"scene.{key} does not exist: {resolved}")


@dataclass
class RenderConfig:
    """Render recipe parsed from a config file.

    Attributes:
        raw: The unparsed mapping, kept for the run manifest.
        base_dir: Directory that relative config paths resolve against.
    """

    raw: dict[str, Any]
    base_dir: str

    # output
    resolution: int = 640
    render_masks: bool = False
    mask_dir: str = "masks"
    # object
    mesh_name: str = "object"
    keypoint_prefix: str = "KP_"
    num_keypoints: int = 18
    keypoints_json: str | None = None
    usd_path: str | None = None
    # pose
    axis_bias: dict[str, float] = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.2})
    axis_jiggle_deg: float = 5.0
    ground_clearance: tuple[float, float] = (0.002, 0.022)
    xy_jitter: float = 0.02
    # camera
    focal_length_mm: float = 50.0
    sensor_width_mm: float = 36.0
    auto_distance: bool = True
    fill_factor: tuple[float, float] = (0.3, 0.7)
    distance: tuple[float, float] = (0.15, 0.35)
    top_down_jiggle_deg: float = 5.0
    image_jitter: tuple[float, float] = (0.0, 0.3)
    stereo_enabled: bool = False
    stereo_baseline_m: float = 0.063
    stereo_left_suffix: str = "left"
    stereo_right_suffix: str = "right"
    # visibility
    min_keypoints: int = 3
    fallback_to_random: bool = True
    # scene
    hdri_dir: str = ""
    texture_dir: str = ""
    distractor_dir: str = ""
    num_distractors: tuple[int, int] = (1, 3)
    change_hdri_every: int = 1
    change_ground_every: int = 3
    change_distractors_every: int = 5
    change_material_every: int = 5
    ground_style: str = "texture"
    no_ground_prob: float = 0.0
    world_background_modes: Sequence[str] = ("studio",)
    # material. ``preserve`` emulates the brushed-aluminium material mesh_setup.py authors in
    # Blender (the USD carries no material); ``override`` samples ``object_color_choices`` with
    # ``object_metallic`` like synth_worker.randomize_configured_object_material.
    object_material_mode: str = "preserve"
    object_color_choices: tuple[tuple[float, float, float], ...] = ()
    base_color: tuple[float, float] = (0.75, 0.92)
    roughness: tuple[float, float] = (0.1, 0.35)
    object_metallic: float = 0.0
    # Optional portable stress augmentation; absent preserves historical scene records.
    robustness: dict[str, Any] | None = None
    # Normalized opt-in metallic appearance stress; None when absent or disabled.
    robust_appearance: dict[str, Any] | None = None
    # isaac
    render_mode: str = "PathTracing"
    spp: int = 32
    occlusion_tol_m: float = 0.002
    bbox_source: str = "keypoints"
    light_intensity: tuple[float, float] = (500.0, 8000.0)
    dome_intensity: tuple[float, float] = (500.0, 2000.0)

    @property
    def images_per_sample(self) -> int:
        """Images written per scene: two for a stereo pair, else one."""
        return 2 if self.stereo_enabled else 1

    def resolve(self, path: str | None) -> str | None:
        """Resolve ``path`` against :attr:`base_dir`; empty or ``None`` means "none"."""
        if not path:
            return None
        return resolve_under(self.base_dir, path)


def sample_object_material(rng: random.Random, cfg: RenderConfig) -> dict[str, Any]:
    """OmniPBR inputs for one object-material randomization.

    ``override`` samples ``object_color_choices``
    (synth_worker.randomize_configured_object_material); ``preserve`` emulates the
    brushed-aluminium tint (synth_worker.randomize_aluminum_material).
    """
    if cfg.object_material_mode == "override":
        color, metallic = rng.choice(cfg.object_color_choices), cfg.object_metallic
    else:  # object_metallic only applies to override in Blender too; the material is aluminium
        grey = rng.uniform(*cfg.base_color)
        color, metallic = (
            tuple(min(1.0, max(0.0, grey + rng.uniform(-0.02, 0.02))) for _ in range(3)),
            1.0,
        )
    return {
        "diffuse_color_constant": tuple(float(c) for c in color),
        "reflection_roughness_constant": rng.uniform(*cfg.roughness),
        "metallic_constant": float(metallic),
    }


def load_render_config(path: str | Path) -> RenderConfig:
    """Load a render config; relative paths inside it resolve against its directory.

    Args:
        path: YAML render config.

    Returns:
        The parsed config with ``base_dir`` set to the file's directory.

    Raises:
        ValueError: The document is not a valid render config.
        FileNotFoundError: A configured asset directory does not exist.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    return render_config_from_mapping(raw, os.path.dirname(os.path.abspath(path)))


def render_config_from_mapping(raw: Mapping[str, Any], base_dir: str) -> RenderConfig:
    """Parse an already loaded render config.

    Args:
        raw: Config mapping.
        base_dir: Directory that relative paths in ``raw`` resolve against.

    Returns:
        The parsed config.

    Raises:
        ValueError: ``raw`` is not a valid render config.
        FileNotFoundError: A configured asset directory does not exist.
    """
    if not isinstance(raw, Mapping):
        raise ValueError(f"render config root must be a mapping, got {type(raw).__name__}")
    out = raw.get("output", {}) or {}
    obj = raw.get("object", {}) or {}
    pose = raw.get("pose", {}) or {}
    cam = raw.get("camera", {}) or {}
    vis = raw.get("visibility", {}) or {}
    scene = raw.get("scene", {}) or {}
    mat = raw.get("material", {}) or {}
    rnd = raw.get("render", {}) or {}
    isaac = resolve_settings(raw)
    stereo = cam.get("stereo", {}) or {}

    z_bias = float(pose.get("z_facing_bias", 0.2))
    axis_cfg = pose.get("axis_bias")
    if isinstance(axis_cfg, Mapping):
        axis_bias = {k: float(axis_cfg.get(k, 0.0)) for k in ("x", "y", "z")}
    else:
        axis_bias = {"x": 0.0, "y": 0.0, "z": z_bias}
    if sum(axis_bias.values()) > 1.0 + 1e-9:
        raise ValueError(f"axis_bias probabilities sum to more than 1: {axis_bias}")

    modes = scene.get("world_background_modes", "")
    if isinstance(modes, str):
        modes = modes.split(",")
    modes = [str(m).strip().lower() for m in modes]
    modes = [m for m in modes if m in {"studio", "random", "dark", "hdri"}] or ["studio"]

    no_ground = scene.get("no_ground", 0.0)
    no_ground_prob = (
        (1.0 if no_ground else 0.0) if isinstance(no_ground, bool) else float(no_ground or 0.0)
    )

    color_choices = _color_choices(mat.get("object_color_choices"))
    mode_raw = mat.get("object_material_mode")
    material_mode = (
        str(mode_raw).strip().lower()
        if mode_raw is not None
        else ("override" if color_choices else "preserve")
    )
    if material_mode not in {"preserve", "override"}:
        raise ValueError(
            f"material.object_material_mode must be 'preserve' or 'override', got {material_mode!r}"
        )
    if material_mode == "override" and not color_choices:
        raise ValueError(
            "material.object_material_mode is 'override' but object_color_choices is empty"
        )

    cfg = RenderConfig(
        raw=dict(raw),
        base_dir=base_dir,
        resolution=int(out.get("resolution", 640)),
        render_masks=_bool(out.get("render_masks", rnd.get("render_masks")), False),
        mask_dir=str(out.get("mask_dir", rnd.get("mask_dir", "masks"))).strip().strip("/")
        or "masks",
        mesh_name=str(obj.get("mesh_name", "object")),
        keypoint_prefix=str(obj.get("keypoint_prefix", "KP_")),
        num_keypoints=int(obj.get("num_keypoints", 18)),
        keypoints_json=obj.get("keypoints_json"),
        usd_path=obj.get("usd_path"),
        axis_bias=axis_bias,
        axis_jiggle_deg=float(pose.get("axis_jiggle_deg", pose.get("z_facing_jiggle_deg", 5.0))),
        ground_clearance=(
            float(pose.get("ground_clearance_min", 0.002)),
            float(pose.get("ground_clearance_max", 0.022)),
        ),
        xy_jitter=float(pose.get("xy_jitter", 0.02)),
        focal_length_mm=float(cam.get("focal_length_mm", 50.0)),
        sensor_width_mm=float(cam.get("sensor_width_mm", 36.0)),
        auto_distance=_bool(cam.get("auto_distance", True), True),
        fill_factor=(
            float(cam.get("fill_factor_min", 0.3)),
            float(cam.get("fill_factor_max", 0.7)),
        ),
        distance=(float(cam.get("distance_min", 0.15)), float(cam.get("distance_max", 0.35))),
        top_down_jiggle_deg=float(cam.get("top_down_jiggle_deg", 5.0)),
        image_jitter=(
            float(cam.get("image_jitter_min", 0.0)),
            float(cam.get("image_jitter_max", 0.3)),
        ),
        stereo_enabled=_bool(stereo.get("enabled"), False),
        stereo_baseline_m=float(stereo.get("baseline_m", 0.063)),
        stereo_left_suffix=str(stereo.get("left_suffix", "left")),
        stereo_right_suffix=str(stereo.get("right_suffix", "right")),
        min_keypoints=int(vis.get("min_keypoints", 3)),
        fallback_to_random=_bool(vis.get("fallback_to_random", True), True),
        hdri_dir=_asset_dir(scene, "hdri_dir"),
        texture_dir=_asset_dir(scene, "texture_dir"),
        distractor_dir=_asset_dir(scene, "distractor_dir"),
        num_distractors=(
            int(scene.get("num_distractors_min", 1)),
            int(scene.get("num_distractors_max", 3)),
        ),
        change_hdri_every=max(1, int(scene.get("change_hdri_every", 1))),
        change_ground_every=max(1, int(scene.get("change_ground_every", 3))),
        change_distractors_every=max(1, int(scene.get("change_distractors_every", 5))),
        change_material_every=max(1, int(scene.get("change_material_every", 5))),
        ground_style=str(scene.get("ground_style", "texture")).lower(),
        no_ground_prob=no_ground_prob,
        world_background_modes=tuple(modes),
        base_color=(float(mat.get("base_color_min", 0.75)), float(mat.get("base_color_max", 0.92))),
        roughness=(
            float(mat.get("object_roughness_min", mat.get("roughness_min", 0.1))),
            float(mat.get("object_roughness_max", mat.get("roughness_max", 0.35))),
        ),
        object_material_mode=material_mode,
        object_color_choices=color_choices,
        object_metallic=float(mat.get("object_metallic", 0.0)),
        robustness=validate_robustness(raw.get("robustness")),
        robust_appearance=validate_robust_appearance(raw.get("robust_appearance")),
        render_mode=str(isaac.get("render_mode", "PathTracing")),
        spp=int(isaac["spp"]),
        occlusion_tol_m=float(isaac.get("occlusion_tol_m", 0.002)),
        bbox_source=str(isaac.get("bbox_source", "keypoints")).lower(),
        light_intensity=_range(isaac.get("light_intensity_range"), (500.0, 8000.0)),
        dome_intensity=_range(isaac.get("dome_intensity_range"), (500.0, 2000.0)),
    )
    if cfg.robustness is not None and not cfg.render_masks:
        raise ValueError("robustness requires output.render_masks: true for realized coverage")
    if cfg.render_mode not in {"PathTracing", "RayTracedLighting"}:
        raise ValueError(
            f"isaac.render_mode must be PathTracing or RayTracedLighting, got {cfg.render_mode!r}"
        )
    if cfg.bbox_source not in {"keypoints", "mask"}:
        raise ValueError(
            f"isaac.bbox_source must be 'keypoints' or 'mask', got {cfg.bbox_source!r}"
        )
    _check_asset_dirs(cfg)
    return cfg
