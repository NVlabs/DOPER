# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in truncation and capsule occluders for portable scene records.

Clipping fractions refer to projected 3D AABB extent, not silhouette area.
Capsules approximate foreground grasp occlusion, not anatomical hands.

``augment_scene`` only edits the camera and adds a ``robustness`` section with
explicit proxy geometry; the Blender worker realizes exactly what is recorded.
Draws come from the caller's per-index stream in a fixed order, so reordering
them changes every record for a given seed.
"""

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from doper.scene import geometry as G  # noqa: N812 - short alias used across the sampling code


class SceneMesh(Protocol):
    """Mesh facts the augmentation reads (``scene_spec.MeshInfo`` provides them)."""

    @property
    def local_corners(self) -> np.ndarray:
        """(8, 3) AABB corners in the object frame."""
        ...

    @property
    def scene_scale(self) -> float:
        """Mesh size relative to a 10 cm object."""
        ...


EDGES = ("left", "right", "top", "bottom", "top_left", "top_right", "bottom_left", "bottom_right")
MODES = ("easy", "clipping", "occlusion", "combined")
DEFAULT_MODE_WEIGHTS = {"easy": 0.4, "clipping": 0.25, "occlusion": 0.25, "combined": 0.1}
# (key, default, exclusive lower bound, inclusive upper bound) for two-element ranges.
RANGE_OPTIONS = (
    ("clipping_fraction", [0.1, 0.35], 0, 0.8),
    ("proxy_radius_fraction", [0.04, 0.08], 0, 0.25),
    ("proxy_count", [1, 3], 0, 5),
)
# Bisection over a camera shift of up to 100x the farthest corner distance; 80
# halvings resolve it far below a pixel.
CLIP_SEARCH_SCALE = 100
CLIP_SEARCH_STEPS = 80
# Proxies sit at this fraction of the nearest target depth, i.e. in front of it.
PROXY_DEPTH_FRACTION = 0.6
# Radius is capped relative to the nearest target depth so capsules stay thin.
PROXY_MAX_RADIUS_DEPTH_FRACTION = 0.1
# Spacing between parallel capsules and along-axis jitter, relative to the short extent.
PROXY_SPACING = 0.28
PROXY_JITTER = 0.15
# Capsule half-length range, relative to the long projected extent.
PROXY_HALF_LENGTH = (0.3, 0.55)
# Skin and glove-like colors.
PROXY_COLORS = [[0.48, 0.25, 0.14], [0.75, 0.49, 0.32], [0.18, 0.22, 0.28], [0.06, 0.19, 0.42]]
PROXY_ROUGHNESS = 0.65


def validate_robustness(raw: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Normalize the YAML ``robustness`` section; ``None`` means disabled.

    Fills defaults, rejects unknown keys and requires ``mode_weights`` to be
    probabilities that sum to 1.
    """
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("robustness must be a mapping")
    allowed = {
        "mode_weights",
        "clipping_fraction",
        "clipping_edges",
        "proxy_radius_fraction",
        "proxy_count",
    }
    if set(raw) - allowed:
        raise ValueError(f"unknown robustness keys: {sorted(set(raw) - allowed)}")
    result: dict[str, Any] = {
        "mode_weights": validate_mode_weights(raw.get("mode_weights", DEFAULT_MODE_WEIGHTS))
    }
    for key, default, lower, upper in RANGE_OPTIONS:
        result[key] = validate_range(key, raw.get(key, default), lower, upper)
    if any(v != int(v) for v in result["proxy_count"]):
        raise ValueError("robustness.proxy_count must contain integers")
    result["proxy_count"] = [int(v) for v in result["proxy_count"]]
    edges = raw.get("clipping_edges", EDGES)
    if not isinstance(edges, (list, tuple)) or not edges or any(e not in EDGES for e in edges):
        raise ValueError("robustness.clipping_edges must contain valid edges or corners")
    result["clipping_edges"] = list(edges)
    return result


def validate_mode_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Return a weight for every mode, defaulting unlisted modes to 0."""
    if not isinstance(weights, Mapping) or set(weights) - set(MODES):
        raise ValueError("robustness.mode_weights has unknown modes")
    weights = {k: float(weights.get(k, 0)) for k in MODES}
    is_valid = all(math.isfinite(v) and v >= 0 for v in weights.values())
    if not is_valid or not math.isclose(sum(weights.values()), 1):
        raise ValueError(
            "robustness.mode_weights must be finite nonnegative probabilities summing to 1"
        )
    return weights


def validate_range(key: str, values: Sequence[float], lower: float, upper: float) -> list[float]:
    """Validate a ``[min, max]`` option with ``lower < min <= max <= upper``."""
    if not isinstance(values, (tuple, list)) or len(values) != 2:
        raise ValueError(f"robustness.{key} must be a two-element range")
    values = [float(v) for v in values]
    if any(not math.isfinite(v) for v in values) or not lower < values[0] <= values[1] <= upper:
        raise ValueError(f"robustness.{key} must satisfy {lower} < min <= max <= {upper}")
    return values


def _projected_box(
    points: np.ndarray, camera: np.ndarray, intr: G.Intrinsics
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the projected ``(min_uv, max_uv, depths)`` of points in front of the camera."""
    uv, depth = G.project_points(points, camera, intr)
    if (depth <= 0).any():
        raise ValueError("robustness requires target corners in front of camera")
    return uv.min(0), uv.max(0), depth


def _clip_camera(
    points: np.ndarray, camera: np.ndarray, intr: G.Intrinsics, edge: str, fraction: float
) -> tuple[np.ndarray, dict[str, Any]]:
    """Translate the camera so ``fraction`` of the projected box crosses ``edge``.

    Each image axis is solved independently by bisection on a shift parallel to
    the image plane. On an axis named by ``edge`` the weighted point between the
    box extremes is moved onto that border; on the other axis the box is centered.
    Returns the new camera and the clipping provenance for the record.
    """
    camera = camera.copy()
    for axis, negative, positive, size in (
        (0, "left", "right", intr.width),
        (1, "top", "bottom", intr.height),
    ):
        if negative in edge:
            weight, target = fraction, 0
        elif positive in edge:
            weight, target = 1 - fraction, size
        else:
            weight, target = 0.5, size / 2
        # Image +y points down, opposite to the camera's +Y axis.
        direction = camera[:3, axis] * (1 if axis == 0 else -1)
        origin = camera[:3, 3].copy()
        # Translation parallel to the image plane leaves optical depth unchanged.
        # The weighted projected extrema are monotone, even with perspective.
        limit = np.linalg.norm(points - origin, axis=1).max() * CLIP_SEARCH_SCALE
        low, high = -limit, limit
        for _ in range(CLIP_SEARCH_STEPS):
            shift = (low + high) / 2
            camera[:3, 3] = origin + direction * shift
            lo, hi, _ = _projected_box(points, camera, intr)
            value = (1 - weight) * lo[axis] + weight * hi[axis]
            if value > target:
                low = shift
            else:
                high = shift
    lo, hi, _ = _projected_box(points, camera, intr)
    extent = hi - lo
    inside = np.maximum(0, np.minimum(hi, [intr.width, intr.height]) - np.maximum(lo, 0))
    return camera, {
        "edge": edge,
        "requested_axis_fraction": fraction,
        "projected_bbox_xyxy": [*lo.tolist(), *hi.tolist()],
        "projected_bbox_outside_fraction": float(1 - np.prod(inside) / np.prod(extent)),
    }


def augment_scene(
    record: dict[str, Any],
    cfg: Mapping[str, Any],
    mesh: SceneMesh,
    intr: G.Intrinsics,
    rng: random.Random,
) -> None:
    """Attach explicit provenance and modify only the selected camera/proxy fields.

    Samples one mode per record. ``clipping`` and ``combined`` move the camera;
    ``occlusion`` and ``combined`` add capsule proxies in front of the target.
    """
    mode = rng.choices(MODES, weights=[cfg["mode_weights"][k] for k in MODES])[0]
    robust = {"version": 1, "mode": mode, "proxies": []}
    record["robustness"] = robust
    points = G.transform_points(np.asarray(record["object"]["pose"]), mesh.local_corners)
    camera = np.asarray(record["camera"]["cam_to_world"])
    if mode in ("clipping", "combined"):
        camera = apply_clipping(record, robust, cfg, mesh, intr, rng, points, camera)
    if mode in ("occlusion", "combined"):
        add_grasp_proxies(robust, cfg, intr, rng, points, camera)


def apply_clipping(
    record: dict[str, Any],
    robust: dict[str, Any],
    cfg: Mapping[str, Any],
    mesh: SceneMesh,
    intr: G.Intrinsics,
    rng: random.Random,
    points: np.ndarray,
    camera: np.ndarray,
) -> np.ndarray:
    """Sample an edge and fraction, move the camera, and return the new camera."""
    edge = rng.choice(cfg["clipping_edges"])
    fraction = rng.uniform(*cfg["clipping_fraction"])
    original_camera = camera.copy()
    camera, robust["clipping"] = _clip_camera(points, camera, intr, edge, fraction)
    if record.get("ground", {}).get("enabled", False):
        # Match the Blender plane placement. A low top-edge translation can
        # otherwise put the camera beneath an opaque floor. Retry from the
        # original view with a local half-turn, preserving the requested
        # image edge and clipping fraction. Valid scenes remain unchanged.
        ground_z = record["object"]["bbox_min_z"] - 0.001 * mesh.scene_scale
        clearance = 1e-6 * mesh.scene_scale
        if camera[2, 3] <= ground_z + clearance:
            original_camera[:3, :2] *= -1
            camera, robust["clipping"] = _clip_camera(points, original_camera, intr, edge, fraction)
            robust["clipping"]["camera_roll_fallback_deg"] = 180
            if camera[2, 3] <= ground_z + clearance:
                raise ValueError("robustness clipping camera must remain above enabled ground")
    record["camera"]["cam_to_world"] = camera.tolist()
    center = np.asarray(record["object"]["center"])
    record["camera"]["distance"] = float(np.linalg.norm(camera[:3, 3] - center))
    return camera


def add_grasp_proxies(
    robust: dict[str, Any],
    cfg: Mapping[str, Any],
    intr: G.Intrinsics,
    rng: random.Random,
    points: np.ndarray,
    camera: np.ndarray,
) -> None:
    """Add parallel finger-like capsules over the in-frame target box.

    Capsules share one random image-plane direction, are spaced across the
    shorter in-frame extent, and are placed at a single depth in front of the
    nearest target corner. Endpoints are stored in world coordinates.
    """
    robust["occluder_kind"] = "capsule_grasp_proxy"
    lo, hi, depths = _projected_box(points, camera, intr)
    lo, hi = np.maximum(lo, 0), np.minimum(hi, [intr.width, intr.height])
    extent = hi - lo
    if (extent <= 0).any():
        raise ValueError("robustness cannot place proxies over a fully offscreen target")
    depth = float(depths.min() * PROXY_DEPTH_FRACTION)
    angle = rng.uniform(-math.pi, math.pi)
    tangent = np.array([math.cos(angle), math.sin(angle)])
    normal = np.array([-tangent[1], tangent[0]])
    count = rng.randint(*cfg["proxy_count"])
    color = rng.choice(PROXY_COLORS)

    def pixel_to_world(pixel: np.ndarray) -> list[float]:
        """Back-project an image point to world space at the proxy depth."""
        local = np.array(
            [
                (pixel[0] - intr.cx) * depth / intr.fx,
                -(pixel[1] - intr.cy) * depth / intr.fy,
                -depth,
            ]
        )
        return (camera[:3, :3] @ local + camera[:3, 3]).tolist()

    for i in range(count):
        radius_px = rng.uniform(*cfg["proxy_radius_fraction"]) * extent.min()
        center = (lo + hi) / 2 + normal * (i - (count - 1) / 2) * extent.min() * PROXY_SPACING
        center += tangent * rng.uniform(-PROXY_JITTER, PROXY_JITTER) * extent.min()
        half_length = extent.max() * rng.uniform(*PROXY_HALF_LENGTH)
        radius = min(radius_px * depth / intr.fx, depths.min() * PROXY_MAX_RADIUS_DEPTH_FRACTION)
        robust["proxies"].append(
            {
                "start": pixel_to_world(center - tangent * half_length),
                "end": pixel_to_world(center + tangent * half_length),
                "radius": float(radius),
                "color": color,
                "roughness": PROXY_ROUGHNESS,
            }
        )


def require_backend(
    spec: Mapping[str, Any] | None,
    backend: str,
    *,
    scenes: bool = True,
    export_environment: bool = False,
    appearance: Mapping[str, Any] | None = None,
) -> None:
    """Fail when robustness or appearance is enabled for an unsupported render path.

    Only Blender rendering from launcher-generated scene records realizes these
    options; other paths would silently drop them.
    """
    enabled = spec is not None or appearance is not None
    if enabled and (backend != "blender" or not scenes or export_environment):
        raise ValueError(
            "robustness currently requires Blender scene-record rendering "
            "without environment export"
        )
