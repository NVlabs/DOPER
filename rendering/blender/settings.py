# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve worker settings without touching Blender or output files.

The small YAML fallback supports Blender distributions without PyYAML. Settings
retain the historical renderer defaults; experiment recipes override them.
"""

import ast
import glob
import os
from argparse import Namespace
from dataclasses import dataclass
from typing import Any

from rendering.paths import resolve_under

# A YAML-parsed config value of unknown shape.
ConfigValue = Any


try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class WorkerSettings:
    """Resolved configuration shared by renderer components."""

    _axis_total: float
    auto_distance_enabled: bool
    auto_distance_fill_factor_max: float
    auto_distance_fill_factor_min: float
    axis_bias: dict[str, float]
    axis_jiggle: float
    cam_dist_max: float
    cam_dist_min: float
    cfg_output: dict[str, Any]
    cfg_render: dict[str, Any]
    change_distractors_every: int
    change_ground_every: int
    change_hdri_every: int
    change_material_every: int
    checker_background_prob: float
    checker_background_saturation_range: tuple[float, float]
    checker_background_scale_range: tuple[float, float]
    checker_background_value_range: tuple[float, float]
    config_path: str
    dark_background_prob: float
    dark_background_saturation_range: tuple[float, float]
    dark_background_value_range: tuple[float, float]
    distractor_dir: str
    fallback_to_random: bool
    flat_background_saturation_range: tuple[float, float]
    flat_background_value_range: tuple[float, float]
    ground_clearance_max: float
    ground_clearance_min: float
    ground_image_crop_scale_range: tuple[float, float]
    ground_image_files: list[str]
    ground_image_prob: float
    ground_image_rotation_deg_range: tuple[float, float]
    ground_image_roughness_range: tuple[float, float]
    ground_image_tile_range: tuple[float, float]
    ground_style: str
    hdri_dir: str
    image_jitter_max: float
    image_jitter_min: float
    images_per_sample: int
    keypoint_mask_check: bool
    keypoint_mask_max_warn: int
    keypoint_mask_min_frac: float
    keypoint_mask_sample_every: int
    keypoint_mask_sample_first: int
    keypoint_mask_tol_px: int
    keypoints_json: str | None
    kp_prefix: str
    mask_dir_name: str
    mat_base_color_max: float
    mat_base_color_min: float
    mat_brush_scale_max: float
    mat_brush_scale_min: float
    mat_roughness_max: float
    mat_roughness_min: float
    mesh_name: str
    min_visible_keypoints: int
    no_ground_prob: float
    num_distractors_max: int
    num_distractors_min: int
    num_keypoints: int
    object_color_choices: str | list[str | list[float]] | None
    object_material_mode: str
    object_metallic: float
    object_roughness_max: float
    object_roughness_min: float
    procedural_distractor_background_depth_range: tuple[float, float]
    procedural_distractor_background_max: int
    procedural_distractor_background_min: int
    procedural_distractor_dark_prob: float
    procedural_distractor_dark_saturation_range: tuple[float, float]
    procedural_distractor_dark_value_range: tuple[float, float]
    procedural_distractor_foreground_center_lateral_range: tuple[float, float]
    procedural_distractor_foreground_center_prob: float
    procedural_distractor_foreground_depth_range: tuple[float, float]
    procedural_distractor_foreground_max: int
    procedural_distractor_foreground_min: int
    procedural_distractor_lateral_range: tuple[float, float]
    procedural_distractor_max_attempts: int
    procedural_distractor_min_unoccluded_keypoints: int
    procedural_distractor_shapes: list[str]
    procedural_distractor_size_range: tuple[float, float]
    procedural_distractor_vertical_jitter_range: tuple[float, float]
    procedural_distractors: bool
    procedural_ground_modes: list[str]
    render_masks: bool
    base_dir: str
    resolution: int
    random_lights_max: int
    resolution_y: int
    focal_length_mm: float
    sensor_width_mm: float
    shadow_rays: int
    stereo_baseline_m: float
    stereo_enabled: bool
    stereo_left_suffix: str
    stereo_right_suffix: str
    texture_dir: str
    top_down_jiggle: float
    world_background_modes: list[str]
    world_background_saturation_range: tuple[float, float]
    world_background_strength_range: tuple[float, float]
    world_background_value_range: tuple[float, float]
    xy_jitter: float

    @property
    def requires_ground_textures(self) -> bool:
        """Whether ground planes consume texture sets (only "procedural" never does)."""
        return self.ground_style != "procedural"


def parse_scalar(value: ConfigValue) -> ConfigValue:
    """Parse the small YAML scalar subset used by render configs."""
    value = value.strip()
    if "#" in value:
        value = value.split("#", 1)[0].strip()
    if value == "":
        return ""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        return value[1:-1]
    if (value.startswith("[") and value.endswith("]")) or (
        value.startswith("{") and value.endswith("}")
    ):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_simple_yaml(path: ConfigValue) -> ConfigValue:
    """Load simple nested key/value YAML without PyYAML.

    Blender's bundled Python often lacks PyYAML, and render workers may run on
    offline machines. The repo render configs use indentation-based maps with
    scalar leaves and occasional simple lists, so this fallback keeps rendering
    self-contained.
    """
    root = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    with open(path) as f:
        lines = list(enumerate(f, start=1))

    def next_content_line(index: ConfigValue) -> ConfigValue:
        """Return the next non-blank, non-comment config line after ``index``, or None."""
        for _line_no, candidate in lines[index + 1 :]:
            if not candidate.strip() or candidate.lstrip().startswith("#"):
                continue
            return candidate
        return None

    for index, (line_no, raw_line) in enumerate(lines):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if line.startswith("- "):
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            if not isinstance(parent, list):
                raise ValueError(
                    f"Unsupported YAML list line {line_no} in {path}: {raw_line.rstrip()}"
                )
            parent.append(parse_scalar(line[2:].strip()))
            continue

        if ":" not in line:
            raise ValueError(f"Unsupported YAML line {line_no} in {path}: {raw_line.rstrip()}")

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if value == "" or value.startswith("#"):
            next_line = next_content_line(index)
            next_indent = (
                len(next_line) - len(next_line.lstrip(" ")) if next_line is not None else indent
            )
            child = (
                []
                if next_line is not None
                and next_indent > indent
                and next_line.strip().startswith("- ")
                else {}
            )
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = parse_scalar(value)

    return root


def load_yaml_config(path: ConfigValue) -> ConfigValue:
    """Load the render config, falling back to a small YAML subset parser without PyYAML."""
    if yaml is None:
        print("PyYAML not available in Blender Python; using built-in simple YAML parser.")
        return load_simple_yaml(path)

    with open(path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"render YAML root must be a mapping, got {type(config).__name__}")
    return config


def config_range(value: ConfigValue, default: ConfigValue) -> ConfigValue:
    """Parse a two-value ``(low, high)`` range from a config value, with a default."""
    if value is None:
        value = default
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if len(value) != 2:
        raise ValueError(f"expected a 2-value range, got {value!r}")
    low, high = float(value[0]), float(value[1])
    if high < low:
        raise ValueError(f"invalid range {value!r}")
    return (low, high)


def config_bool(value: ConfigValue, default: bool = False) -> ConfigValue:
    """Parse a config value as a boolean (accepts 1/true/yes/on strings)."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def resolve_config_path(path: str, base_dir: str) -> str:
    """Resolve ``path`` against ``base_dir`` unless it is absolute."""
    return resolve_under(base_dir, str(path))


def resolve_asset_dir(path: ConfigValue, base_dir: ConfigValue) -> ConfigValue:
    """Resolve an optional asset directory; ``None`` or empty means none (returns "")."""
    if path is None or not str(path).strip():
        return ""
    return resolve_config_path(path, base_dir)


def config_items(value: ConfigValue) -> ConfigValue:
    """Split a config value (list or comma-separated string) into stripped items."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def load_settings(args: Namespace) -> WorkerSettings:
    """Read the render config named by ``args.config`` into :class:`WorkerSettings`."""
    config_path = os.path.abspath(args.config)
    # Relative paths inside the config resolve against the config file's directory.
    base_dir = os.path.dirname(config_path)

    cfg = load_yaml_config(config_path)
    # Robustness and appearance values live only in launcher-generated scene records.
    uses_robustness = cfg.get("robustness") is not None
    uses_appearance = (cfg.get("robust_appearance") or {}).get("enabled", False)
    if (uses_robustness or uses_appearance) and not getattr(args, "scenes", None):
        raise ValueError("robustness requires launcher-generated --scenes")

    # Extract config sections
    cfg_output = cfg.get("output", {})
    cfg_object = cfg.get("object", {})
    cfg_pose = cfg.get("pose", {})
    cfg_camera = cfg.get("camera", {})
    cfg_visibility = cfg.get("visibility", {})
    cfg_scene = cfg.get("scene", {})
    cfg_material = cfg.get("material", {})
    cfg_render = cfg.get("render", {})

    # ---------------------------------------------------------------------------
    # Setup from config
    # ---------------------------------------------------------------------------
    resolution = cfg_output.get("resolution", 640)
    # optional non-square output: the real HANDal frames are 4:3, and a
    # square render changes the aspect distortion as well as the FOV
    random_lights_max = int(cfg_scene.get("random_lights_max", 3))
    resolution_y = int(cfg_output.get("resolution_y", resolution))
    focal_length_mm = float(cfg_camera.get("focal_length_mm", 50.0))
    sensor_width_mm = float(cfg_camera.get("sensor_width_mm", 36.0))
    mesh_name = cfg_object.get("mesh_name", "object")
    kp_prefix = cfg_object.get("keypoint_prefix", "KP_")
    num_keypoints = cfg_object.get("num_keypoints", 18)
    # A CLI path is relative to the current directory; a config path to the config's directory.
    keypoints_json = (
        os.path.abspath(args.keypoints_json)
        if args.keypoints_json
        else cfg_object.get("keypoints_json", None)
    )

    # Pose config
    z_facing_bias = cfg_pose.get("z_facing_bias", 0.2)
    z_facing_jiggle = cfg_pose.get("z_facing_jiggle_deg", 5.0)
    ground_clearance_min = cfg_pose.get("ground_clearance_min", 0.002)
    ground_clearance_max = cfg_pose.get("ground_clearance_max", 0.022)
    xy_jitter = cfg_pose.get("xy_jitter", 0.02)

    # Axis-facing bias (generalized). Probabilities of aligning the object's
    # local x / y / z axis with world +Z (the "up" direction the camera looks
    # down on). Whatever mass is left goes to a fully random pose.
    #
    # For backward compatibility: if `axis_bias` is not set, we fall back to the
    # old `z_facing_bias` knob (which only biased the local-Z axis).
    cfg_axis = cfg_pose.get("axis_bias", None)
    if cfg_axis is None:
        axis_bias = {"x": 0.0, "y": 0.0, "z": float(z_facing_bias)}
    else:
        axis_bias = {
            "x": float(cfg_axis.get("x", 0.0)),
            "y": float(cfg_axis.get("y", 0.0)),
            "z": float(cfg_axis.get("z", 0.0)),
        }
    _axis_total = axis_bias["x"] + axis_bias["y"] + axis_bias["z"]
    if _axis_total > 1.0 + 1e-6:
        # Renormalize (and drop "random pose" share to 0) so we never exceed 1.
        axis_bias = {k: v / _axis_total for k, v in axis_bias.items()}
        _axis_total = 1.0

    axis_jiggle = cfg_pose.get("axis_jiggle_deg", z_facing_jiggle)

    # Camera config
    cam_dist_min = cfg_camera.get("distance_min", 0.15)
    cam_dist_max = cfg_camera.get("distance_max", 0.35)
    top_down_jiggle = cfg_camera.get("top_down_jiggle_deg", 5.0)
    image_jitter_min = cfg_camera.get("image_jitter_min", 0.0)
    image_jitter_max = cfg_camera.get("image_jitter_max", 0.3)

    # Auto-distance config - adapts camera to object size
    auto_distance_enabled = cfg_camera.get("auto_distance", True)
    auto_distance_fill_factor_min = cfg_camera.get(
        "fill_factor_min", 0.3
    )  # Object fills 30-70% of frame
    auto_distance_fill_factor_max = cfg_camera.get("fill_factor_max", 0.7)

    # Optional paired stereo rendering. When enabled, each sampled object/scene
    # state emits a left/right image pair with a fixed baseline in the camera rig's
    # local X direction. The two cameras share rotation (parallel optical axes),
    # and each annotation stores the per-view camera extrinsics plus a common
    # pair_id so multiview pose solvers can group observations.
    cfg_stereo = cfg_camera.get("stereo", {})
    stereo_enabled = bool(cfg_stereo.get("enabled", False))
    stereo_baseline_m = float(cfg_stereo.get("baseline_m", 0.063))
    stereo_left_suffix = str(cfg_stereo.get("left_suffix", "left"))
    stereo_right_suffix = str(cfg_stereo.get("right_suffix", "right"))
    images_per_sample = 2 if stereo_enabled else 1

    # Visibility config
    min_visible_keypoints = cfg_visibility.get("min_keypoints", 3)
    fallback_to_random = cfg_visibility.get("fallback_to_random", True)

    # Scene config
    # An absent or empty asset directory means "none"; relative ones resolve against base_dir.
    hdri_dir = resolve_asset_dir(cfg_scene.get("hdri_dir"), base_dir)
    texture_dir = resolve_asset_dir(cfg_scene.get("texture_dir"), base_dir)
    distractor_dir = resolve_asset_dir(cfg_scene.get("distractor_dir"), base_dir)
    num_distractors_min = cfg_scene.get("num_distractors_min", 1)
    num_distractors_max = cfg_scene.get("num_distractors_max", 3)
    procedural_distractors = bool(cfg_scene.get("procedural_distractors", False))
    procedural_distractor_shapes = [
        item.strip().lower()
        for item in str(cfg_scene.get("procedural_distractor_shapes", "cube,sphere,cone")).split(
            ","
        )
        if item.strip()
    ]
    procedural_distractor_shapes = [
        item for item in procedural_distractor_shapes if item in {"cube", "sphere", "cone"}
    ]
    if not procedural_distractor_shapes:
        procedural_distractor_shapes = ["cube", "sphere", "cone"]
    procedural_distractor_foreground_min = int(
        cfg_scene.get("procedural_distractor_foreground_min", 0)
    )
    procedural_distractor_foreground_max = int(
        cfg_scene.get("procedural_distractor_foreground_max", 0)
    )
    procedural_distractor_background_min = int(
        cfg_scene.get("procedural_distractor_background_min", 0)
    )
    procedural_distractor_background_max = int(
        cfg_scene.get("procedural_distractor_background_max", 0)
    )
    procedural_distractor_size_range = config_range(
        cfg_scene.get("procedural_distractor_size_range"),
        (0.04, 0.12),
    )
    procedural_distractor_lateral_range = config_range(
        cfg_scene.get("procedural_distractor_lateral_range"),
        (0.35, 1.05),
    )
    procedural_distractor_foreground_center_prob = float(
        cfg_scene.get("procedural_distractor_foreground_center_prob", 0.0)
    )
    procedural_distractor_foreground_center_lateral_range = config_range(
        cfg_scene.get("procedural_distractor_foreground_center_lateral_range"),
        (0.0, 0.28),
    )
    procedural_distractor_foreground_depth_range = config_range(
        cfg_scene.get("procedural_distractor_foreground_depth_range"),
        (0.12, 0.45),
    )
    procedural_distractor_background_depth_range = config_range(
        cfg_scene.get("procedural_distractor_background_depth_range"),
        (0.15, 0.80),
    )
    procedural_distractor_vertical_jitter_range = config_range(
        cfg_scene.get("procedural_distractor_vertical_jitter_range"),
        (0.0, 0.04),
    )
    procedural_distractor_dark_prob = float(cfg_scene.get("procedural_distractor_dark_prob", 0.0))
    procedural_distractor_dark_value_range = config_range(
        cfg_scene.get("procedural_distractor_dark_value_range"),
        (0.02, 0.18),
    )
    procedural_distractor_dark_saturation_range = config_range(
        cfg_scene.get("procedural_distractor_dark_saturation_range"),
        (0.25, 0.90),
    )
    procedural_distractor_min_unoccluded_keypoints = int(
        cfg_scene.get("procedural_distractor_min_unoccluded_keypoints", 12)
    )
    procedural_distractor_max_attempts = int(cfg_scene.get("procedural_distractor_max_attempts", 5))
    change_hdri_every = cfg_scene.get("change_hdri_every", 1)
    change_ground_every = cfg_scene.get("change_ground_every", 3)
    change_distractors_every = cfg_scene.get("change_distractors_every", 5)
    change_material_every = cfg_scene.get("change_material_every", 5)
    ground_style = str(cfg_scene.get("ground_style", "texture")).lower()
    no_ground_prob = float(cfg_scene.get("no_ground", 0.0))
    if isinstance(cfg_scene.get("no_ground"), bool):
        no_ground_prob = 1.0 if cfg_scene.get("no_ground") else 0.0
    print(f"Ground plane skip probability: {no_ground_prob * 100:.0f}%")

    ground_image_files = []
    ground_image_dir_config = cfg_scene.get("ground_image_dir", None)
    if ground_image_dir_config:
        ground_image_dir = resolve_config_path(ground_image_dir_config, base_dir)
        if os.path.isdir(ground_image_dir):
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
                ground_image_files.extend(sorted(glob.glob(os.path.join(ground_image_dir, ext))))
        else:
            print(f"Warning: configured ground_image_dir not found: {ground_image_dir}")

    for ground_image_file_config in config_items(cfg_scene.get("ground_image_files", None)):
        ground_image_file = resolve_config_path(ground_image_file_config, base_dir)
        if os.path.exists(ground_image_file):
            ground_image_files.append(ground_image_file)
        else:
            print(f"Warning: configured ground_image_file not found: {ground_image_file}")

    for ground_image_file_config in config_items(cfg_scene.get("ground_image_file", None)):
        ground_image_file = resolve_config_path(ground_image_file_config, base_dir)
        if os.path.exists(ground_image_file):
            ground_image_files.append(ground_image_file)
        else:
            print(f"Warning: configured ground_image_file not found: {ground_image_file}")

    ground_image_files = sorted(dict.fromkeys(ground_image_files))
    ground_image_prob = float(
        cfg_scene.get(
            "ground_image_prob",
            1.0 if ground_style in {"image", "image_texture", "photo"} else 0.0,
        )
    )
    ground_image_crop_scale_range = config_range(
        cfg_scene.get("ground_image_crop_scale_range"),
        (0.45, 1.0),
    )
    ground_image_tile_range = config_range(
        cfg_scene.get("ground_image_tile_range"),
        (1.0, 1.0),
    )
    ground_image_rotation_deg_range = config_range(
        cfg_scene.get("ground_image_rotation_deg_range"),
        (-180.0, 180.0),
    )
    ground_image_roughness_range = config_range(
        cfg_scene.get("ground_image_roughness_range"),
        (0.38, 0.82),
    )
    procedural_ground_modes = [
        item.strip().lower()
        for item in str(cfg_scene.get("procedural_ground_modes", "color,wood")).split(",")
        if item.strip()
    ]
    procedural_world_background = bool(cfg_scene.get("procedural_world_background", False))
    world_background_modes = [
        item.strip().lower()
        for item in config_items(cfg_scene.get("world_background_modes", ""))
        if item.strip()
    ]
    dark_background_prob = float(cfg_scene.get("dark_background_prob", 0.0))
    dark_background_value_range = config_range(
        cfg_scene.get("dark_background_value_range"), (0.0, 0.08)
    )
    dark_background_saturation_range = config_range(
        cfg_scene.get("dark_background_saturation_range"), (0.0, 0.25)
    )
    checker_background_prob = float(cfg_scene.get("checker_background_prob", 0.0))
    checker_background_scale_range = config_range(
        cfg_scene.get("checker_background_scale_range"), (6.0, 18.0)
    )
    checker_background_value_range = config_range(
        cfg_scene.get("checker_background_value_range"), (0.25, 0.95)
    )
    checker_background_saturation_range = config_range(
        cfg_scene.get("checker_background_saturation_range"), (0.10, 0.85)
    )
    flat_background_value_range = config_range(
        cfg_scene.get("flat_background_value_range"), (0.35, 0.90)
    )
    flat_background_saturation_range = config_range(
        cfg_scene.get("flat_background_saturation_range"), (0.08, 0.65)
    )
    world_background_value_range = config_range(
        cfg_scene.get("world_background_value_range"), (0.25, 0.85)
    )
    world_background_saturation_range = config_range(
        cfg_scene.get("world_background_saturation_range"), (0.05, 0.35)
    )
    world_background_strength_range = config_range(
        cfg_scene.get("world_background_strength_range"), (0.4, 1.3)
    )
    if not procedural_ground_modes:
        procedural_ground_modes = ["color", "wood"]
    if not world_background_modes:
        world_background_modes = ["random"] if procedural_world_background else ["studio"]

    # Material config
    mat_base_color_min = cfg_material.get("base_color_min", 0.75)
    mat_base_color_max = cfg_material.get("base_color_max", 0.92)
    mat_roughness_min = cfg_material.get("roughness_min", 0.1)
    mat_roughness_max = cfg_material.get("roughness_max", 0.35)
    mat_brush_scale_min = cfg_material.get("brush_scale_min", 50)
    mat_brush_scale_max = cfg_material.get("brush_scale_max", 150)
    object_color_choices = cfg_material.get("object_color_choices", None)
    object_material_mode_config = cfg_material.get("object_material_mode", None)
    object_material_mode = (
        str(object_material_mode_config).strip().lower()
        if object_material_mode_config is not None
        else ("override" if object_color_choices else "preserve")
    )
    object_metallic = float(cfg_material.get("object_metallic", 0.0))
    object_roughness_min = float(cfg_material.get("object_roughness_min", mat_roughness_min))
    object_roughness_max = float(cfg_material.get("object_roughness_max", mat_roughness_max))

    # Render config
    shadow_rays = cfg_render.get("shadow_rays", 2)
    render_masks = config_bool(
        cfg_output.get("render_masks", cfg_render.get("render_masks", None)), False
    )
    mask_dir_name = (
        str(cfg_output.get("mask_dir", cfg_render.get("mask_dir", "masks"))).strip().strip("/")
    )
    if not mask_dir_name:
        mask_dir_name = "masks"
    if os.path.isabs(mask_dir_name) or any(part == ".." for part in mask_dir_name.split("/")):
        raise ValueError(f"mask_dir must be a relative output path, got {mask_dir_name!r}")

    # Keypoint/mask consistency guard. Catches a keypoints_3d.json vs render-mesh
    # frame mismatch (e.g. a 90-degree axis swap from a Y-up/Z-up STL round-trip):
    # the dataset stays self-consistent so training loss looks fine, but every
    # keypoint is projected off the actual object. We reload a sample of rendered
    # masks and check the visible keypoints land on the target foreground.
    keypoint_mask_check = config_bool(
        cfg_output.get(
            "check_keypoints_on_mask",
            cfg_visibility.get("check_keypoints_on_mask", None),
        ),
        True,
    )
    keypoint_mask_tol_px = int(
        cfg_output.get("keypoint_mask_tol_px", cfg_visibility.get("keypoint_mask_tol_px", 6))
    )
    keypoint_mask_min_frac = float(
        cfg_output.get("keypoint_mask_min_frac", cfg_visibility.get("keypoint_mask_min_frac", 0.6))
    )
    keypoint_mask_max_warn = int(
        cfg_output.get("keypoint_mask_max_warn", cfg_visibility.get("keypoint_mask_max_warn", 10))
    )
    keypoint_mask_sample_first = int(
        cfg_output.get(
            "keypoint_mask_sample_first",
            cfg_visibility.get("keypoint_mask_sample_first", 20),
        )
    )
    keypoint_mask_sample_every = int(
        cfg_output.get(
            "keypoint_mask_sample_every",
            cfg_visibility.get("keypoint_mask_sample_every", 50),
        )
    )
    return WorkerSettings(
        _axis_total=_axis_total,
        auto_distance_enabled=auto_distance_enabled,
        auto_distance_fill_factor_max=auto_distance_fill_factor_max,
        auto_distance_fill_factor_min=auto_distance_fill_factor_min,
        axis_bias=axis_bias,
        axis_jiggle=axis_jiggle,
        cam_dist_max=cam_dist_max,
        cam_dist_min=cam_dist_min,
        cfg_output=cfg_output,
        cfg_render=cfg_render,
        change_distractors_every=change_distractors_every,
        change_ground_every=change_ground_every,
        change_hdri_every=change_hdri_every,
        change_material_every=change_material_every,
        checker_background_prob=checker_background_prob,
        checker_background_saturation_range=checker_background_saturation_range,
        checker_background_scale_range=checker_background_scale_range,
        checker_background_value_range=checker_background_value_range,
        config_path=config_path,
        dark_background_prob=dark_background_prob,
        dark_background_saturation_range=dark_background_saturation_range,
        dark_background_value_range=dark_background_value_range,
        distractor_dir=distractor_dir,
        fallback_to_random=fallback_to_random,
        flat_background_saturation_range=flat_background_saturation_range,
        flat_background_value_range=flat_background_value_range,
        ground_clearance_max=ground_clearance_max,
        ground_clearance_min=ground_clearance_min,
        ground_image_crop_scale_range=ground_image_crop_scale_range,
        ground_image_files=ground_image_files,
        ground_image_prob=ground_image_prob,
        ground_image_rotation_deg_range=ground_image_rotation_deg_range,
        ground_image_roughness_range=ground_image_roughness_range,
        ground_image_tile_range=ground_image_tile_range,
        ground_style=ground_style,
        hdri_dir=hdri_dir,
        image_jitter_max=image_jitter_max,
        image_jitter_min=image_jitter_min,
        images_per_sample=images_per_sample,
        keypoint_mask_check=keypoint_mask_check,
        keypoint_mask_max_warn=keypoint_mask_max_warn,
        keypoint_mask_min_frac=keypoint_mask_min_frac,
        keypoint_mask_sample_every=keypoint_mask_sample_every,
        keypoint_mask_sample_first=keypoint_mask_sample_first,
        keypoint_mask_tol_px=keypoint_mask_tol_px,
        keypoints_json=keypoints_json,
        kp_prefix=kp_prefix,
        mask_dir_name=mask_dir_name,
        mat_base_color_max=mat_base_color_max,
        mat_base_color_min=mat_base_color_min,
        mat_brush_scale_max=mat_brush_scale_max,
        mat_brush_scale_min=mat_brush_scale_min,
        mat_roughness_max=mat_roughness_max,
        mat_roughness_min=mat_roughness_min,
        mesh_name=mesh_name,
        min_visible_keypoints=min_visible_keypoints,
        no_ground_prob=no_ground_prob,
        num_distractors_max=num_distractors_max,
        num_distractors_min=num_distractors_min,
        num_keypoints=num_keypoints,
        object_color_choices=object_color_choices,
        object_material_mode=object_material_mode,
        object_metallic=object_metallic,
        object_roughness_max=object_roughness_max,
        object_roughness_min=object_roughness_min,
        procedural_distractor_background_depth_range=procedural_distractor_background_depth_range,
        procedural_distractor_background_max=procedural_distractor_background_max,
        procedural_distractor_background_min=procedural_distractor_background_min,
        procedural_distractor_dark_prob=procedural_distractor_dark_prob,
        procedural_distractor_dark_saturation_range=procedural_distractor_dark_saturation_range,
        procedural_distractor_dark_value_range=procedural_distractor_dark_value_range,
        procedural_distractor_foreground_center_lateral_range=procedural_distractor_foreground_center_lateral_range,
        procedural_distractor_foreground_center_prob=procedural_distractor_foreground_center_prob,
        procedural_distractor_foreground_depth_range=procedural_distractor_foreground_depth_range,
        procedural_distractor_foreground_max=procedural_distractor_foreground_max,
        procedural_distractor_foreground_min=procedural_distractor_foreground_min,
        procedural_distractor_lateral_range=procedural_distractor_lateral_range,
        procedural_distractor_max_attempts=procedural_distractor_max_attempts,
        procedural_distractor_min_unoccluded_keypoints=procedural_distractor_min_unoccluded_keypoints,
        procedural_distractor_shapes=procedural_distractor_shapes,
        procedural_distractor_size_range=procedural_distractor_size_range,
        procedural_distractor_vertical_jitter_range=procedural_distractor_vertical_jitter_range,
        procedural_distractors=procedural_distractors,
        procedural_ground_modes=procedural_ground_modes,
        render_masks=render_masks,
        base_dir=base_dir,
        resolution=resolution,
        random_lights_max=random_lights_max,
        resolution_y=resolution_y,
        focal_length_mm=focal_length_mm,
        sensor_width_mm=sensor_width_mm,
        shadow_rays=shadow_rays,
        stereo_baseline_m=stereo_baseline_m,
        stereo_enabled=stereo_enabled,
        stereo_left_suffix=stereo_left_suffix,
        stereo_right_suffix=stereo_right_suffix,
        texture_dir=texture_dir,
        top_down_jiggle=top_down_jiggle,
        world_background_modes=world_background_modes,
        world_background_saturation_range=world_background_saturation_range,
        world_background_strength_range=world_background_strength_range,
        world_background_value_range=world_background_value_range,
        xy_jitter=xy_jitter,
    )
