# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named CAD-curve schemas, projection, targets, and alignment utilities.

The curve vocabulary is fixed for a trained object model: channel ``c`` always
means curve ID ``c``. Curves are represented as sampled object-frame
polylines. Projection deliberately keeps occluded and back-facing geometry;
only portions behind the camera near plane are removed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    import torch

    from .config import ModelConfig

SCHEMA_VERSION = 1
# Model-config fields that describe a curve head; checkpoints must record all of them.
CURVE_METADATA_FIELDS = (
    "num_curves",
    "curve_head_channels",
    "curve_target_sigma",
    "curve_schema_version",
    "curve_names",
    "curve_geometry_sha256",
    "curve_object_id",
)
SIDECAR_VERSION = 1


@dataclass(frozen=True)
class CADCurve:
    """One named CAD curve as a sampled object-frame polyline.

    Attributes:
        id: Channel index; equals the curve's position in the schema.
        name: Unique curve name.
        points_xyz: ``(N, 3)`` object-frame points.
        closed: Whether the last point connects back to the first.
    """

    id: int
    name: str
    points_xyz: np.ndarray
    closed: bool = False


@dataclass(frozen=True)
class CADCurveSchema:
    """Ordered, hashed set of named CAD curves for one object model.

    Attributes:
        object_id: Object identifier.
        curves: Curves ordered by contiguous channel id.
        version: Schema version.
        geometry_sha256: Hash of the canonical curve geometry.
    """

    object_id: str
    curves: tuple[CADCurve, ...]
    version: int
    geometry_sha256: str

    @property
    def names(self) -> tuple[str, ...]:
        """Return the curve names in channel order."""
        return tuple(curve.name for curve in self.curves)

    @property
    def num_curves(self) -> int:
        """Return the number of curves."""
        return len(self.curves)


@dataclass(frozen=True)
class ProjectedCurves:
    """Flattened full-image polylines with per-channel offsets.

    NaN rows in ``points_xy`` separate disconnected pieces created by
    near-plane clipping. ``offsets`` has shape ``(C + 1,)``.
    """

    points_xy: np.ndarray
    offsets: np.ndarray
    valid: np.ndarray
    schema_sha256: str = ""

    @property
    def num_curves(self) -> int:
        """Return the number of curve channels."""
        return int(self.valid.shape[0])


@dataclass(frozen=True)
class PoseCurveRefinement:
    """Result of refining a pose against named curve maps.

    Attributes:
        rvec: Refined Rodrigues rotation vector, shape ``(3,)``.
        tvec: Refined translation vector, shape ``(3,)``.
        initial_score: Mean curve probability along the initial pose's curves.
        final_score: Same score for the refined pose.
        steps: Number of optimizer steps taken.
        initial_reprojection_error_px: Optional keypoint reprojection error before.
        final_reprojection_error_px: Optional keypoint reprojection error after.
    """

    rvec: np.ndarray
    tvec: np.ndarray
    initial_score: float
    final_score: float
    steps: int
    initial_reprojection_error_px: float | None = None
    final_reprojection_error_px: float | None = None


def _canonical_geometry_payload(
    object_id: str, curves: Sequence[CADCurve], version: int
) -> dict[str, Any]:
    return {
        "version": int(version),
        "object_id": str(object_id),
        "curves": [
            {
                "id": int(curve.id),
                "name": curve.name,
                "closed": bool(curve.closed),
                "points_xyz": np.asarray(curve.points_xyz, dtype=np.float64).tolist(),
            }
            for curve in curves
        ],
    }


def geometry_sha256(
    object_id: str, curves: Sequence[CADCurve], version: int = SCHEMA_VERSION
) -> str:
    """Hash the canonical JSON encoding of a curve set.

    Args:
        object_id: Object identifier.
        curves: Curves to hash.
        version: Schema version.

    Returns:
        Hex SHA-256 digest.
    """
    payload = _canonical_geometry_payload(object_id, curves, version)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def parse_curve_schema(payload: dict[str, Any]) -> CADCurveSchema:
    """Validate a curve-schema mapping and build a ``CADCurveSchema``.

    Args:
        payload: Mapping with ``version``, ``object_id`` and ordered ``curves``.

    Returns:
        The validated schema.

    Raises:
        ValueError: If the payload is malformed.
    """
    if not isinstance(payload, dict):
        raise ValueError("curve schema must be a mapping")
    version = int(payload.get("version", SCHEMA_VERSION))
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported curve schema version {version}; expected {SCHEMA_VERSION}")
    object_id = payload.get("object_id")
    if not isinstance(object_id, str) or not object_id:
        raise ValueError("curve schema object_id must be a non-empty string")
    raw_curves = payload.get("curves")
    if not isinstance(raw_curves, list) or not raw_curves:
        raise ValueError("curve schema curves must be a non-empty list")

    curves: list[CADCurve] = []
    names = set()
    for expected_id, raw in enumerate(raw_curves):
        if not isinstance(raw, dict):
            raise ValueError(f"curve {expected_id} must be a mapping")
        curve_id = raw.get("id")
        if not isinstance(curve_id, int) or isinstance(curve_id, bool):
            raise ValueError("curve id must be an integer")
        if curve_id != expected_id:
            raise ValueError(
                f"curve IDs must be contiguous and ordered; expected {expected_id}, got {curve_id}"
            )
        name = raw.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"curve {curve_id} name must be non-empty")
        if name in names:
            raise ValueError(f"duplicate curve name {name!r}")
        names.add(name)
        points = np.asarray(raw.get("points_xyz"), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 2:
            raise ValueError(f"curve {name!r} points_xyz must have shape (N,3), N >= 2")
        if not np.isfinite(points).all():
            raise ValueError(f"curve {name!r} contains non-finite coordinates")
        closed = raw.get("closed", False)
        if not isinstance(closed, bool):
            raise ValueError(f"curve {name!r} closed must be boolean")
        curves.append(
            CADCurve(
                id=curve_id,
                name=name,
                points_xyz=points.astype(np.float32),
                closed=closed,
            )
        )
    curve_tuple = tuple(curves)
    return CADCurveSchema(
        object_id=object_id,
        curves=curve_tuple,
        version=version,
        geometry_sha256=geometry_sha256(object_id, curve_tuple, version),
    )


def load_curve_schema(path: str) -> CADCurveSchema:
    """Load and validate a curve schema from a JSON file."""
    with open(path) as handle:
        payload = json.load(handle)
    return parse_curve_schema(payload)


def schema_metadata(schema: CADCurveSchema) -> dict[str, Any]:
    """Return the checkpoint metadata fields that describe ``schema``."""
    return {
        "curve_schema_version": schema.version,
        "curve_names": list(schema.names),
        "curve_geometry_sha256": schema.geometry_sha256,
        "curve_object_id": schema.object_id,
    }


def checkpoint_curve_metadata(
    checkpoint_model_cfg: dict[str, Any], state_keys: Iterable[str]
) -> dict[str, Any]:
    """Validate and return curve metadata encoded by a checkpoint.

    Legacy checkpoints have neither curve fields nor curve weights and resolve
    to a disabled branch. Curve-enabled checkpoints must contain both metadata
    and weights; partial or ambiguous checkpoints fail loudly.
    """
    keys = tuple(state_keys)
    has_weights = any(key.startswith("curve_head.") for key in keys)
    has_enabled_field = "curve_head_enabled" in checkpoint_model_cfg
    if not has_enabled_field:
        if has_weights:
            raise RuntimeError("checkpoint has curve_head weights but no curve schema metadata")
        return {"curve_head_enabled": False}
    enabled = bool(checkpoint_model_cfg["curve_head_enabled"])
    if enabled != has_weights:
        raise RuntimeError("checkpoint curve metadata and curve_head weights disagree")
    if not enabled:
        return {"curve_head_enabled": False}

    missing = [name for name in CURVE_METADATA_FIELDS if name not in checkpoint_model_cfg]
    if missing:
        raise RuntimeError(f"curve-enabled checkpoint lacks metadata fields: {missing}")
    num_curves = int(checkpoint_model_cfg["num_curves"])
    names = tuple(str(name) for name in checkpoint_model_cfg["curve_names"])
    schema_version = int(checkpoint_model_cfg["curve_schema_version"])
    head_channels = int(checkpoint_model_cfg["curve_head_channels"])
    target_sigma = float(checkpoint_model_cfg["curve_target_sigma"])
    geometry_hash = str(checkpoint_model_cfg["curve_geometry_sha256"])
    object_id = str(checkpoint_model_cfg["curve_object_id"])
    if num_curves <= 0 or len(names) != num_curves:
        raise RuntimeError("curve checkpoint num_curves does not match ordered curve_names")
    if (
        head_channels <= 0
        or not np.isfinite(target_sigma)
        or target_sigma <= 0
        or schema_version <= 0
        or not geometry_hash
        or not object_id
    ):
        raise RuntimeError("curve checkpoint schema metadata is incomplete")
    return {
        "curve_head_enabled": True,
        "num_curves": num_curves,
        "curve_head_channels": head_channels,
        "curve_target_sigma": target_sigma,
        "curve_schema_version": schema_version,
        "curve_names": names,
        "curve_geometry_sha256": geometry_hash,
        "curve_object_id": object_id,
    }


def require_matching_curve_metadata(
    checkpoint_metadata: dict[str, Any], model_cfg: ModelConfig
) -> None:
    """Require exact fixed-vocabulary agreement for a curve-enabled model."""
    if not checkpoint_metadata.get("curve_head_enabled", False):
        if bool(model_cfg.curve_head_enabled):
            raise RuntimeError("checkpoint has no curve head")
        return
    if not bool(model_cfg.curve_head_enabled):
        raise RuntimeError("checkpoint has a curve head but model disables it")
    mismatches = []
    for field_name in CURVE_METADATA_FIELDS:
        source = checkpoint_metadata[field_name]
        target = getattr(model_cfg, field_name)
        if field_name == "curve_names":
            source = tuple(source)
            target = tuple(target)
        if source != target:
            mismatches.append(f"{field_name}: checkpoint={source!r}, model={target!r}")
    if mismatches:
        raise RuntimeError("curve schema/checkpoint mismatch: " + "; ".join(mismatches))


def camera_matrix_from_intrinsics(intrinsics: dict[str, Any]) -> np.ndarray:
    """Build a 3x3 camera matrix from ``camera_matrix`` or ``fx/fy/cx/cy`` entries.

    Raises:
        ValueError: If the intrinsics are missing, non-finite or mis-shaped.
    """
    if "camera_matrix" in intrinsics:
        matrix = np.asarray(intrinsics["camera_matrix"], dtype=np.float64)
    else:
        try:
            matrix = np.array(
                [
                    [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
                    [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("intrinsics require camera_matrix or fx/fy/cx/cy") from exc
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("camera matrix must be finite with shape (3,3)")
    return matrix


def camera_from_object_from_annotation(annotation: dict[str, Any]) -> np.ndarray:
    """Resolve common DOPER annotation pose conventions.

    ``rvec``/``tvec`` are already OpenCV camera-from-object. Blender exports
    store world-from-camera and world-from-object matrices; the fixed axis
    conversion maps Blender camera coordinates (x right, y up, -z forward) to
    OpenCV coordinates (x right, y down, z forward).
    """
    if "rvec" in annotation and "tvec" in annotation:
        rvec = np.asarray(annotation["rvec"], dtype=np.float64).reshape(-1)
        tvec = np.asarray(annotation["tvec"], dtype=np.float64).reshape(-1)
        if rvec.shape != (3,) or tvec.shape != (3,):
            raise ValueError("rvec and tvec must each contain three values")
        rotation, _ = cv2.Rodrigues(rvec)
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = rotation
        result[:3, 3] = tvec
        return result

    if "camera_extrinsics" in annotation and "object_pose" in annotation:
        world_from_camera = np.asarray(annotation["camera_extrinsics"], dtype=np.float64)
        world_from_object = np.asarray(annotation["object_pose"], dtype=np.float64)
        if world_from_camera.shape != (4, 4) or world_from_object.shape != (4, 4):
            raise ValueError("Blender camera_extrinsics/object_pose must be 4x4")
        if not (np.isfinite(world_from_camera).all() and np.isfinite(world_from_object).all()):
            raise ValueError("pose matrices must be finite")
        opencv_from_blender_camera = np.diag([1.0, -1.0, -1.0, 1.0])
        return opencv_from_blender_camera @ np.linalg.inv(world_from_camera) @ world_from_object

    raise ValueError("annotation requires rvec/tvec or camera_extrinsics/object_pose")


def intrinsics_from_annotation(
    annotation: dict[str, Any],
    fallback: dict[str, Any] | None = None,
) -> np.ndarray:
    """Return the camera matrix of an annotation, falling back to ``fallback``.

    Raises:
        ValueError: If neither source provides intrinsics.
    """
    value = annotation.get("camera_intrinsics")
    if isinstance(value, dict):
        return camera_matrix_from_intrinsics(value)
    if fallback is not None:
        return camera_matrix_from_intrinsics(fallback)
    raise ValueError("annotation has no camera_intrinsics and no fallback was supplied")


def _clip_segment_near(
    first: np.ndarray, second: np.ndarray, near: float
) -> tuple[np.ndarray, np.ndarray] | None:
    z0 = float(first[2])
    z1 = float(second[2])
    inside0 = z0 >= near
    inside1 = z1 >= near
    if not inside0 and not inside1:
        return None
    a = first.copy()
    b = second.copy()
    if inside0 != inside1:
        denom = z1 - z0
        if abs(denom) <= 1e-15:
            return None
        t = (near - z0) / denom
        intersection = first + t * (second - first)
        intersection[2] = near
        if inside0:
            b = intersection
        else:
            a = intersection
    return a, b


def project_curve(
    curve: CADCurve,
    camera_from_object: np.ndarray,
    camera_matrix: np.ndarray,
    near: float = 1e-5,
) -> np.ndarray:
    """Project one curve as disconnected 2D segment pairs separated by NaNs."""
    transform = np.asarray(camera_from_object, dtype=np.float64)
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("camera_from_object must have shape (4,4)")
    if matrix.shape != (3, 3):
        raise ValueError("camera_matrix must have shape (3,3)")
    if not np.isfinite(near) or near <= 0:
        raise ValueError("near must be finite and positive")

    points = np.asarray(curve.points_xyz, dtype=np.float64)
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    camera = homogeneous @ transform.T
    camera = camera[:, :3]
    pairs = [(idx, idx + 1) for idx in range(points.shape[0] - 1)]
    if curve.closed:
        pairs.append((points.shape[0] - 1, 0))

    pieces: list[np.ndarray] = []
    for first_idx, second_idx in pairs:
        clipped = _clip_segment_near(camera[first_idx], camera[second_idx], float(near))
        if clipped is None:
            continue
        segment = np.stack(clipped)
        pixels_h = segment @ matrix.T
        pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
        if np.isfinite(pixels).all():
            pieces.append(pixels.astype(np.float32))
    if not pieces:
        return np.zeros((0, 2), dtype=np.float32)
    separator = np.full((1, 2), np.nan, dtype=np.float32)
    rows: list[np.ndarray] = []
    for piece_idx, piece in enumerate(pieces):
        if piece_idx:
            rows.append(separator)
        rows.append(piece)
    return np.concatenate(rows, axis=0)


def project_schema(
    schema: CADCurveSchema,
    camera_from_object: np.ndarray,
    camera_matrix: np.ndarray,
    near: float = 1e-5,
) -> ProjectedCurves:
    """Project every curve of ``schema`` into the image.

    Args:
        schema: Curve schema.
        camera_from_object: 4x4 OpenCV camera-from-object transform.
        camera_matrix: 3x3 camera matrix.
        near: Near-plane depth below which geometry is clipped.

    Returns:
        Flattened projected polylines with per-curve offsets and validity.
    """
    arrays = []
    offsets = [0]
    valid = []
    for curve in schema.curves:
        projected = project_curve(curve, camera_from_object, camera_matrix, near=near)
        arrays.append(projected)
        offsets.append(offsets[-1] + projected.shape[0])
        valid.append(float(projected.shape[0] >= 2))
    points = (
        np.concatenate(arrays, axis=0)
        if arrays and offsets[-1] > 0
        else np.zeros((0, 2), dtype=np.float32)
    )
    return ProjectedCurves(
        points_xy=points.astype(np.float32),
        offsets=np.asarray(offsets, dtype=np.int64),
        valid=np.asarray(valid, dtype=np.float32),
        schema_sha256=schema.geometry_sha256,
    )


def validate_projected_curves(
    projected: ProjectedCurves, expected_num_curves: int | None = None
) -> None:
    """Check the structure of projected curves.

    Raises:
        ValueError: If shapes, offsets or validity flags are inconsistent.
    """
    points = np.asarray(projected.points_xy)
    offsets = np.asarray(projected.offsets)
    valid = np.asarray(projected.valid)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("projected points_xy must have shape (N,2)")
    finite_or_nan = np.isfinite(points) | np.isnan(points)
    if not finite_or_nan.all() or np.isinf(points).any():
        raise ValueError("projected points may be finite or NaN separators, not Inf")
    if offsets.ndim != 1 or offsets.shape[0] != valid.shape[0] + 1:
        raise ValueError("offsets must have shape (num_curves + 1,)")
    if offsets[0] != 0 or offsets[-1] != points.shape[0]:
        raise ValueError("offsets must span points_xy")
    if np.any(np.diff(offsets) < 0):
        raise ValueError("offsets must be nondecreasing")
    if valid.ndim != 1 or not np.isin(valid, [0.0, 1.0]).all():
        raise ValueError("valid must be a one-dimensional binary array")
    if expected_num_curves is not None and valid.shape[0] != expected_num_curves:
        raise ValueError(
            f"curve sidecar has {valid.shape[0]} channels; expected {expected_num_curves}"
        )


def save_projected_curves(path: str, projected: ProjectedCurves) -> None:
    """Validate ``projected`` and write it to a compressed ``.npz`` sidecar."""
    validate_projected_curves(projected)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.savez_compressed(
        path,
        version=np.asarray([SIDECAR_VERSION], dtype=np.int32),
        points_xy=np.asarray(projected.points_xy, dtype=np.float32),
        offsets=np.asarray(projected.offsets, dtype=np.int64),
        valid=np.asarray(projected.valid, dtype=np.float32),
        schema_sha256=np.asarray(projected.schema_sha256),
    )


def load_projected_curves(
    path: str,
    expected_num_curves: int | None = None,
    expected_schema_sha256: str | None = None,
) -> ProjectedCurves:
    """Load and validate a projected-curve sidecar.

    Args:
        path: ``.npz`` sidecar path.
        expected_num_curves: Required channel count, if given.
        expected_schema_sha256: Required schema hash, if given.

    Returns:
        The loaded curves.

    Raises:
        ValueError: On a version, shape or schema-hash mismatch.
    """
    with np.load(path, allow_pickle=False) as payload:
        version = int(np.asarray(payload["version"]).reshape(-1)[0])
        if version != SIDECAR_VERSION:
            raise ValueError(
                f"unsupported curve sidecar version {version}; expected {SIDECAR_VERSION}"
            )
        schema_hash = str(np.asarray(payload["schema_sha256"]).reshape(()))
        projected = ProjectedCurves(
            points_xy=np.asarray(payload["points_xy"], dtype=np.float32),
            offsets=np.asarray(payload["offsets"], dtype=np.int64),
            valid=np.asarray(payload["valid"], dtype=np.float32),
            schema_sha256=schema_hash,
        )
    validate_projected_curves(projected, expected_num_curves)
    if expected_schema_sha256 and projected.schema_sha256 != expected_schema_sha256:
        raise ValueError(
            f"curve sidecar schema hash {projected.schema_sha256} does not match "
            f"expected {expected_schema_sha256}"
        )
    return projected


def transform_projected_curves(projected: ProjectedCurves, affine: np.ndarray) -> ProjectedCurves:
    """Apply a 2x3 affine transform to the finite points of ``projected``."""
    matrix = np.asarray(affine, dtype=np.float64)
    if matrix.shape != (2, 3):
        raise ValueError("affine must have shape (2,3)")
    points = np.asarray(projected.points_xy, dtype=np.float32).copy()
    finite = np.isfinite(points).all(axis=1)
    if finite.any():
        homogeneous = np.concatenate(
            [points[finite], np.ones((int(finite.sum()), 1), dtype=np.float32)],
            axis=1,
        )
        points[finite] = (homogeneous @ matrix.T).astype(np.float32)
    return ProjectedCurves(
        points_xy=points,
        offsets=projected.offsets.copy(),
        valid=projected.valid.copy(),
        schema_sha256=projected.schema_sha256,
    )


def _clip_segment_box(
    first: np.ndarray,
    second: np.ndarray,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Liang-Barsky clipping for a finite 2D segment."""
    delta = second - first
    p = (-delta[0], delta[0], -delta[1], delta[1])
    q = (
        first[0] - x_min,
        x_max - first[0],
        first[1] - y_min,
        y_max - first[1],
    )
    lower = 0.0
    upper = 1.0
    for pi, qi in zip(p, q, strict=True):
        if abs(float(pi)) <= 1e-12:
            if qi < 0:
                return None
            continue
        ratio = float(qi / pi)
        if pi < 0:
            lower = max(lower, ratio)
        else:
            upper = min(upper, ratio)
        if lower > upper:
            return None
    return first + lower * delta, first + upper * delta


def _iter_segments(points: np.ndarray) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    previous = None
    for point in points:
        if not np.isfinite(point).all():
            previous = None
            continue
        if previous is not None:
            yield previous, point
        previous = point


def rasterize_curve_heatmaps(
    projected_crop: ProjectedCurves,
    crop_size: int,
    heatmap_size: int,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode crop-space polylines as independent Gaussian distance ridges."""
    validate_projected_curves(projected_crop)
    if (
        not isinstance(crop_size, (int, np.integer))
        or not isinstance(heatmap_size, (int, np.integer))
        or crop_size <= 1
        or heatmap_size <= 1
    ):
        raise ValueError("crop_size and heatmap_size must be integers greater than 1")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")

    num_curves = projected_crop.num_curves
    target = np.zeros((num_curves, heatmap_size, heatmap_size), dtype=np.float32)
    weight = np.zeros((num_curves,), dtype=np.float32)
    scale = (heatmap_size - 1) / float(crop_size - 1)
    radius = max(1.0, 4.0 * float(sigma))

    for curve_id in range(num_curves):
        if projected_crop.valid[curve_id] <= 0:
            continue
        start = int(projected_crop.offsets[curve_id])
        stop = int(projected_crop.offsets[curve_id + 1])
        points = projected_crop.points_xy[start:stop] * scale
        for first, second in _iter_segments(points):
            clipped = _clip_segment_box(
                first,
                second,
                0.0,
                0.0,
                float(heatmap_size - 1),
                float(heatmap_size - 1),
            )
            if clipped is None:
                continue
            a, b = clipped
            weight[curve_id] = 1.0
            x0 = max(0, int(math.floor(min(a[0], b[0]) - radius)))
            x1 = min(
                heatmap_size - 1,
                int(math.ceil(max(a[0], b[0]) + radius)),
            )
            y0 = max(0, int(math.floor(min(a[1], b[1]) - radius)))
            y1 = min(
                heatmap_size - 1,
                int(math.ceil(max(a[1], b[1]) + radius)),
            )
            yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
            delta = b - a
            length_sq = float(np.dot(delta, delta))
            if length_sq <= 1e-12:
                distance_sq = (xx - a[0]) ** 2 + (yy - a[1]) ** 2
            else:
                fraction = ((xx - a[0]) * delta[0] + (yy - a[1]) * delta[1]) / length_sq
                fraction = np.clip(fraction, 0.0, 1.0)
                nearest_x = a[0] + fraction * delta[0]
                nearest_y = a[1] + fraction * delta[1]
                distance_sq = (xx - nearest_x) ** 2 + (yy - nearest_y) ** 2
            gaussian = np.exp(-distance_sq / (2.0 * float(sigma) ** 2)).astype(np.float32)
            region = target[curve_id, y0 : y1 + 1, x0 : x1 + 1]
            np.maximum(region, gaussian, out=region)
    return target, weight


def curve_heatmap_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Return per-channel IoU and symmetric ridge distance in heatmap pixels."""
    pred = np.asarray(prediction, dtype=np.float32)
    truth = np.asarray(target, dtype=np.float32)
    curve_weight = np.asarray(weight, dtype=np.float32)
    if pred.shape != truth.shape or pred.ndim != 3:
        raise ValueError("prediction and target must have matching (C,H,W) shapes")
    if curve_weight.shape != (pred.shape[0],):
        raise ValueError("weight must have shape (C,)")

    ious = np.full((pred.shape[0],), np.nan, dtype=np.float32)
    distances = np.full((pred.shape[0],), np.nan, dtype=np.float32)
    for curve_id in range(pred.shape[0]):
        if curve_weight[curve_id] <= 0:
            continue
        pred_mask = pred[curve_id] >= threshold
        target_mask = truth[curve_id] >= threshold
        union = np.logical_or(pred_mask, target_mask).sum()
        if union > 0:
            ious[curve_id] = np.logical_and(pred_mask, target_mask).sum() / float(union)
        if not pred_mask.any() or not target_mask.any():
            distances[curve_id] = float(max(pred.shape[1:]))
            continue
        pred_distance = cv2.distanceTransform((~pred_mask).astype(np.uint8), cv2.DIST_L2, 5)
        target_distance = cv2.distanceTransform((~target_mask).astype(np.uint8), cv2.DIST_L2, 5)
        distances[curve_id] = 0.5 * (
            float(pred_distance[target_mask].mean()) + float(target_distance[pred_mask].mean())
        )
    valid = curve_weight > 0
    return {
        "iou_per_curve": ious,
        "distance_per_curve": distances,
        "mean_iou": float(np.nanmean(ious[valid])) if valid.any() else float("nan"),
        "mean_distance": (float(np.nanmean(distances[valid])) if valid.any() else float("nan")),
    }


def overlay_named_curve_heatmaps(
    image_bgr: np.ndarray,
    curve_probabilities: np.ndarray,
    crop_to_image: np.ndarray,
    crop_size: int,
    threshold: float = 0.35,
    alpha: float = 0.65,
) -> np.ndarray:
    """Overlay named crop heatmaps on the original image with stable colors."""
    image = np.asarray(image_bgr)
    probabilities = np.asarray(curve_probabilities, dtype=np.float32)
    affine = np.asarray(crop_to_image, dtype=np.float64)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_bgr must have shape (H,W,3)")
    if probabilities.ndim != 3:
        raise ValueError("curve_probabilities must have shape (C,H,W)")
    if affine.shape != (2, 3):
        raise ValueError("crop_to_image must have shape (2,3)")
    if not 0.0 <= threshold <= 1.0 or not 0.0 <= alpha <= 1.0:
        raise ValueError("threshold and alpha must be in [0,1]")

    heatmap_scale = (crop_size - 1) / float(probabilities.shape[2] - 1)
    heatmap_to_crop = np.array(
        [[heatmap_scale, 0.0, 0.0], [0.0, heatmap_scale, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    crop_to_image_h = np.vstack([affine, [0.0, 0.0, 1.0]])
    heatmap_to_image = (crop_to_image_h @ heatmap_to_crop)[:2]
    output = image.astype(np.float32).copy()
    for curve_id in range(probabilities.shape[0]):
        hue = int((curve_id * 47) % 180)
        color = cv2.cvtColor(
            np.asarray([[[hue, 220, 255]]], dtype=np.uint8),
            cv2.COLOR_HSV2BGR,
        )[0, 0].astype(np.float32)
        warped = cv2.warpAffine(
            probabilities[curve_id],
            heatmap_to_image,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        strength = np.clip((warped - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0)
        blend = (alpha * strength)[..., None]
        output = output * (1.0 - blend) + color[None, None] * blend
    return np.clip(output, 0.0, 255.0).astype(np.uint8)


def sample_named_curve_scores(
    curve_probabilities: np.ndarray,
    projected_crop: ProjectedCurves,
    crop_size: int,
    samples_per_segment: int = 16,
) -> np.ndarray:
    """Sample each named probability channel along its projected CAD curve."""
    probabilities = np.asarray(curve_probabilities, dtype=np.float32)
    if probabilities.ndim != 3:
        raise ValueError("curve_probabilities must have shape (C,H,W)")
    validate_projected_curves(projected_crop, probabilities.shape[0])
    if samples_per_segment <= 0:
        raise ValueError("samples_per_segment must be positive")
    scale = (probabilities.shape[2] - 1) / float(crop_size - 1)
    scores = np.full((probabilities.shape[0],), np.nan, dtype=np.float32)
    for curve_id in range(probabilities.shape[0]):
        start = int(projected_crop.offsets[curve_id])
        stop = int(projected_crop.offsets[curve_id + 1])
        sampled = []
        for first, second in _iter_segments(projected_crop.points_xy[start:stop] * scale):
            fractions = np.linspace(0.0, 1.0, samples_per_segment, endpoint=True, dtype=np.float32)
            points = first[None] + fractions[:, None] * (second - first)[None]
            inside = (
                (points[:, 0] >= 0)
                & (points[:, 0] <= probabilities.shape[2] - 1)
                & (points[:, 1] >= 0)
                & (points[:, 1] <= probabilities.shape[1] - 1)
            )
            if not inside.any():
                continue
            map_x = points[inside, 0].reshape(1, -1).astype(np.float32)
            map_y = points[inside, 1].reshape(1, -1).astype(np.float32)
            values = cv2.remap(
                probabilities[curve_id],
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            sampled.extend(values.reshape(-1).tolist())
        if sampled:
            scores[curve_id] = float(np.mean(sampled))
    return scores


def pose_curve_alignment_score(
    curve_probabilities: np.ndarray,
    schema: CADCurveSchema,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    image_to_crop: np.ndarray,
    crop_size: int,
    samples_per_segment: int = 16,
) -> float:
    """Score a pose by the mean curve probability along its projected CAD curves.

    Returns:
        Mean score over curves, or ``-inf`` if no curve has any sample.
    """
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3))
    camera_from_object = np.eye(4, dtype=np.float64)
    camera_from_object[:3, :3] = rotation
    camera_from_object[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    projected = project_schema(schema, camera_from_object, camera_matrix)
    projected_crop = transform_projected_curves(projected, image_to_crop)
    scores = sample_named_curve_scores(
        curve_probabilities,
        projected_crop,
        crop_size,
        samples_per_segment=samples_per_segment,
    )
    if not np.isfinite(scores).any():
        return float("-inf")
    return float(np.nanmean(scores))


def _sample_schema_points(
    schema: CADCurveSchema, samples_per_segment: int
) -> tuple[np.ndarray, np.ndarray]:
    points = []
    curve_ids = []
    for curve in schema.curves:
        pairs = [(index, index + 1) for index in range(curve.points_xyz.shape[0] - 1)]
        if curve.closed:
            pairs.append((curve.points_xyz.shape[0] - 1, 0))
        for first_idx, second_idx in pairs:
            fractions = np.linspace(
                0.0,
                1.0,
                samples_per_segment,
                endpoint=False,
                dtype=np.float32,
            )
            first = curve.points_xyz[first_idx]
            second = curve.points_xyz[second_idx]
            sampled = first[None] + fractions[:, None] * (second - first)[None]
            points.append(sampled)
            curve_ids.extend([curve.id] * samples_per_segment)
    if not points:
        raise ValueError("curve schema contains no sampleable segments")
    return np.concatenate(points, axis=0), np.asarray(curve_ids, dtype=np.int64)


def refine_pose_with_named_curves(
    curve_probabilities: np.ndarray,
    schema: CADCurveSchema,
    initial_rvec: np.ndarray,
    initial_tvec: np.ndarray,
    camera_matrix: np.ndarray,
    image_to_crop: np.ndarray,
    crop_size: int,
    steps: int = 75,
    rotation_lr: float = 1e-3,
    translation_lr: float = 1e-4,
    samples_per_segment: int = 16,
    device: str = "cpu",
    evaluation_points_3d: np.ndarray | None = None,
    evaluation_points_2d: np.ndarray | None = None,
    dist_coeffs: np.ndarray | None = None,
) -> PoseCurveRefinement:
    """Prototype differentiable SE(3) refinement against named curve maps.

    This deliberately returns a new pose rather than mutating a ``Detection``.
    The caller can compare ``initial_score`` and ``final_score`` before deciding
    whether to accept the refinement.
    """
    import torch
    import torch.nn.functional as torch_functional

    probability_np = np.asarray(curve_probabilities, dtype=np.float32)
    if probability_np.ndim != 3 or probability_np.shape[0] != schema.num_curves:
        raise ValueError("curve probabilities must have shape (schema.num_curves,H,W)")
    if not np.isfinite(probability_np).all():
        raise ValueError("curve probabilities must be finite")
    if steps <= 0 or rotation_lr <= 0 or translation_lr <= 0:
        raise ValueError("steps and pose-refinement learning rates must be positive")
    points_np, curve_ids_np = _sample_schema_points(schema, samples_per_segment)
    matrix_np = np.asarray(camera_matrix, dtype=np.float32)
    affine_np = np.asarray(image_to_crop, dtype=np.float32)
    if matrix_np.shape != (3, 3) or affine_np.shape != (2, 3):
        raise ValueError("camera_matrix must be 3x3 and image_to_crop must be 2x3")

    probability = torch.as_tensor(probability_np, dtype=torch.float32, device=device).unsqueeze(0)
    object_points = torch.as_tensor(points_np, dtype=torch.float32, device=device)
    curve_ids = torch.as_tensor(curve_ids_np, dtype=torch.long, device=device)
    camera = torch.as_tensor(matrix_np, dtype=torch.float32, device=device)
    affine = torch.as_tensor(affine_np, dtype=torch.float32, device=device)
    rvec = torch.nn.Parameter(
        torch.as_tensor(
            np.asarray(initial_rvec, dtype=np.float32).reshape(3),
            device=device,
        ).clone()
    )
    tvec = torch.nn.Parameter(
        torch.as_tensor(
            np.asarray(initial_tvec, dtype=np.float32).reshape(3),
            device=device,
        ).clone()
    )
    optimizer = torch.optim.Adam(
        [
            {"params": [rvec], "lr": rotation_lr},
            {"params": [tvec], "lr": translation_lr},
        ]
    )

    def rotation_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
        theta = torch.linalg.vector_norm(axis_angle)
        safe_theta = theta.clamp(min=1e-8)
        x, y, z = axis_angle
        zero = axis_angle.new_zeros(())
        skew = torch.stack(
            [
                torch.stack([zero, -z, y]),
                torch.stack([z, zero, -x]),
                torch.stack([-y, x, zero]),
            ]
        )
        a = torch.where(
            theta < 1e-4,
            1.0 - theta * theta / 6.0,
            torch.sin(theta) / safe_theta,
        )
        b = torch.where(
            theta < 1e-4,
            0.5 - theta * theta / 24.0,
            (1.0 - torch.cos(theta)) / (safe_theta * safe_theta),
        )
        return torch.eye(3, device=device) + a * skew + b * (skew @ skew)

    height, width = probability_np.shape[1:]
    for _ in range(steps):
        optimizer.zero_grad()
        rotation = rotation_matrix(rvec)
        camera_points = object_points @ rotation.T + tvec[None]
        depth = camera_points[:, 2]
        projected_h = camera_points @ camera.T
        image_xy = projected_h[:, :2] / depth.clamp(min=1e-5)[:, None]
        crop_h = torch.cat(
            [
                image_xy,
                torch.ones((image_xy.shape[0], 1), device=device, dtype=image_xy.dtype),
            ],
            dim=1,
        )
        crop_xy = crop_h @ affine.T
        heatmap_x = crop_xy[:, 0] * (width - 1) / float(crop_size - 1)
        heatmap_y = crop_xy[:, 1] * (height - 1) / float(crop_size - 1)
        grid = torch.stack(
            [
                2.0 * heatmap_x / max(1, width - 1) - 1.0,
                2.0 * heatmap_y / max(1, height - 1) - 1.0,
            ],
            dim=1,
        )
        channel_losses = []
        for curve_id in range(schema.num_curves):
            selected = curve_ids == curve_id
            curve_grid = grid[selected].reshape(1, 1, -1, 2)
            sampled = torch_functional.grid_sample(
                probability[:, curve_id : curve_id + 1],
                curve_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).reshape(-1)
            channel_losses.append(-torch.log(sampled.clamp(min=1e-6)).mean())
        visibility_penalty = torch.relu(1e-4 - depth).mean() * 100.0
        loss = torch.stack(channel_losses).mean() + visibility_penalty
        loss.backward()
        optimizer.step()

    refined_rvec = rvec.detach().cpu().numpy().astype(np.float64)
    refined_tvec = tvec.detach().cpu().numpy().astype(np.float64)
    initial_score = pose_curve_alignment_score(
        probability_np,
        schema,
        initial_rvec,
        initial_tvec,
        matrix_np,
        affine_np,
        crop_size,
        samples_per_segment=samples_per_segment,
    )
    final_score = pose_curve_alignment_score(
        probability_np,
        schema,
        refined_rvec,
        refined_tvec,
        matrix_np,
        affine_np,
        crop_size,
        samples_per_segment=samples_per_segment,
    )

    initial_reprojection_error = None
    final_reprojection_error = None
    if evaluation_points_3d is not None or evaluation_points_2d is not None:
        if evaluation_points_3d is None or evaluation_points_2d is None:
            raise ValueError("evaluation_points_3d and evaluation_points_2d are required together")
        points_3d = np.asarray(evaluation_points_3d, dtype=np.float64)
        points_2d = np.asarray(evaluation_points_2d, dtype=np.float64)
        if (
            points_3d.ndim != 2
            or points_3d.shape[1] != 3
            or points_2d.shape != (points_3d.shape[0], 2)
            or points_3d.shape[0] == 0
        ):
            raise ValueError("evaluation points must have matching (N,3) and (N,2) shapes")
        distortion = (
            np.asarray(dist_coeffs, dtype=np.float64)
            if dist_coeffs is not None
            else np.zeros(5, dtype=np.float64)
        )

        def reprojection_error(eval_rvec: np.ndarray, eval_tvec: np.ndarray) -> float:
            projected, _ = cv2.projectPoints(
                points_3d,
                np.asarray(eval_rvec, dtype=np.float64).reshape(3),
                np.asarray(eval_tvec, dtype=np.float64).reshape(3),
                matrix_np.astype(np.float64),
                distortion,
            )
            return float(np.linalg.norm(projected.reshape(-1, 2) - points_2d, axis=1).mean())

        initial_reprojection_error = reprojection_error(initial_rvec, initial_tvec)
        final_reprojection_error = reprojection_error(refined_rvec, refined_tvec)
    return PoseCurveRefinement(
        rvec=refined_rvec,
        tvec=refined_tvec,
        initial_score=initial_score,
        final_score=final_score,
        steps=steps,
        initial_reprojection_error_px=initial_reprojection_error,
        final_reprojection_error_px=final_reprojection_error,
    )
