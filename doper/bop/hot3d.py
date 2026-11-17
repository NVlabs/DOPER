# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HOT3D camera, pose and mask contracts for BOP-format evaluation.

Meta's camera-only utilities are vendored under ``doper._vendor.hot3d_camera``. Native RGB or
gray images are warped to an axis-preserving pinhole camera at native size, with the official
``focal_scale=1.0`` and the exact original principal point. No upright rotation or synthetic
camera pose is introduced. Native-camera MSPD must project through the original fisheye model,
not the rectified intrinsics.
"""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from doper._vendor.hot3d_camera import camera

FOCAL_SCALE = 1.0
PROTOCOL = "HOT3D-native-reference-camera-official-pinhole-focal1-v1"
MM_PER_M = 1000
_LENGTH_KEYS = ("diameter", "min_x", "min_y", "min_z", "size_x", "size_y", "size_z")
_CALIBRATION_KEYS = (
    "image_width",
    "image_height",
    "projection_model_type",
    "projection_params",
    "label",
    "serial_number",
)
_FISHEYE_MODEL = "CameraModelType.FISHEYE624"
_FISHEYE_PARAM_COUNT = 15
_MASK_THRESHOLD = 127
_MAP_ROW_CHUNK = 128


def _discrete_symmetry_to_mm(values: Any) -> list[float]:  # noqa: ANN401
    """Scale the translation of one flattened 4x4 symmetry from meters to millimeters."""
    matrix = np.asarray(values, dtype=np.float64).reshape(4, 4).copy()
    if not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError("invalid discrete symmetry")
    matrix[:3, 3] *= MM_PER_M
    return matrix.ravel().tolist()


def metadata_m_to_mm(source: dict[str, Any]) -> dict[str, Any]:
    """Convert per-object metadata lengths from meters to millimeters.

    The official ZIP PLYs are in millimeters but the metadata lengths stay in meters. Symmetry
    translations and continuous-symmetry offsets scale too; rotations and axes do not.

    Args:
        source: Object metadata keyed by object id. It is not modified.

    Returns:
        A deep copy with lengths in millimeters.
    """
    result = copy.deepcopy(source)
    for entry in result.values():
        for key in _LENGTH_KEYS:
            if key in entry:
                entry[key] = float(entry[key]) * MM_PER_M
        if "symmetries_discrete" in entry:
            entry["symmetries_discrete"] = [
                _discrete_symmetry_to_mm(values) for values in entry["symmetries_discrete"]
            ]
        for symmetry in entry.get("symmetries_continuous", []):
            symmetry["offset"] = (np.asarray(symmetry["offset"], dtype=float) * MM_PER_M).tolist()
    return result


def decode_rle(data: dict[str, Any]) -> np.ndarray:
    """Decode the official one-based absolute start/length RLE, flattened row-major.

    Args:
        data: Mapping with ``height``, ``width`` and ``rle`` (start, length, start, length, ...).

    Returns:
        ``uint8`` mask of shape ``(height, width)`` holding 0 or 255.

    Raises:
        ValueError: If the dimensions, run pairs or run bounds are invalid.
    """
    height, width = int(data["height"]), int(data["width"])
    if height <= 0 or width <= 0:
        raise ValueError("invalid mask dimensions")
    runs = np.asarray(data["rle"])
    if runs.ndim != 1 or len(runs) % 2:
        raise ValueError("RLE requires start/length pairs")
    result = np.zeros(height * width, dtype=np.uint8)
    if not len(runs):
        return result.reshape(height, width)
    if runs.dtype.kind not in "iu":
        raise ValueError("RLE indices must be integers")
    starts = runs[::2].astype(np.int64) - 1
    lengths = runs[1::2].astype(np.int64)
    total = height * width
    if (starts < 0).any() or (starts > total).any() or (lengths < 0).any():
        raise ValueError("RLE outside mask bounds")
    if (lengths > total - starts).any():
        raise ValueError("RLE outside mask bounds")
    for start, length in zip(starts, lengths, strict=True):
        result[start : start + length] = 255
    return result.reshape(height, width)


def transform(value: dict[str, Any]) -> np.ndarray:
    """Build a 4x4 SE(3) matrix from ``quaternion_wxyz`` and ``translation_xyz``.

    Raises:
        ValueError: If the quaternion or translation is malformed, non-finite or degenerate.
    """
    quaternion = np.asarray(value["quaternion_wxyz"], dtype=np.float64)
    translation = np.asarray(value["translation_xyz"], dtype=np.float64)
    shapes_ok = quaternion.shape == (4,) and translation.shape == (3,)
    if not shapes_ok or not np.isfinite(quaternion).all() or not np.isfinite(translation).all():
        raise ValueError("invalid SE3")
    if np.linalg.norm(quaternion) < 1e-10:
        raise ValueError("invalid SE3")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    result[:3, 3] = translation
    return result


def camera_from_object_mm(
    camera_row: dict[str, Any], object_row: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Compose the object pose in the camera frame; the translation is in millimeters.

    Args:
        camera_row: Row holding ``T_world_from_camera``.
        object_row: Row holding ``T_world_from_object``.

    Returns:
        Rotation ``(3, 3)`` and translation ``(3,)`` in millimeters.
    """
    world_from_camera = transform(camera_row["T_world_from_camera"])
    world_from_object = transform(object_row["T_world_from_object"])
    pose = np.linalg.inv(world_from_camera) @ world_from_object
    return pose[:3, :3], pose[:3, 3] * MM_PER_M


def clean_calibration(raw: dict[str, Any]) -> dict[str, Any]:
    """Keep only the intrinsic calibration fields; extrinsics never enter inference inputs.

    Args:
        raw: Calibration mapping, optionally wrapped under a ``calibration`` key.

    Returns:
        Mapping with the allowed intrinsic keys.

    Raises:
        ValueError: If the model is not FISHEYE624 or the parameters are invalid.
    """
    calibration = raw.get("calibration", raw)
    result = {key: calibration[key] for key in _CALIBRATION_KEYS}
    if result["projection_model_type"] != _FISHEYE_MODEL:
        raise ValueError("unexpected HOT3D camera model")
    params = np.asarray(result["projection_params"], dtype=np.float64)
    if len(params) != _FISHEYE_PARAM_COUNT or not np.isfinite(params).all() or params[0] <= 0:
        raise ValueError("invalid FISHEYE624 calibration")
    return result


def camera_pair(raw: dict[str, Any]) -> tuple[Any, Any, np.ndarray]:
    """Build the native fisheye camera, its pinhole counterpart and the pinhole matrix.

    Extrinsics are identity: warping and projection stay in eye coordinates, and world, object
    or device poses never enter pseudo-label inference inputs.

    Args:
        raw: Calibration mapping accepted by ``clean_calibration``.

    Returns:
        ``(native, pinhole, K)`` with ``K`` the 3x3 float64 pinhole intrinsics.
    """
    cleaned = clean_calibration(raw)
    identity = {"quaternion_wxyz": [1, 0, 0, 0], "translation_xyz": [0, 0, 0]}
    native = camera.from_json({"calibration": cleaned, "T_world_from_camera": identity})
    pinhole = camera.PinholePlaneCameraModel(
        width=native.width,
        height=native.height,
        f=[native.f[0] * FOCAL_SCALE, native.f[1] * FOCAL_SCALE],
        c=native.c,
        distort_coeffs=[],
        T_world_from_eye=np.eye(4),
    )
    return native, pinhole, pinhole.uv_to_window_matrix().astype(np.float64)


@lru_cache(maxsize=4)
def _maps(calibration_json: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    native, pinhole, intrinsics = camera_pair(json.loads(calibration_json))
    # Chunk rows to bound intermediate allocations for native 1408 px images.
    map_x = np.empty((pinhole.height, pinhole.width), dtype=np.float32)
    map_y = np.empty_like(map_x)
    for start in range(0, pinhole.height, _MAP_ROW_CHUNK):
        stop = min(start + _MAP_ROW_CHUNK, pinhole.height)
        xs, ys = np.meshgrid(np.arange(pinhole.width), np.arange(start, stop))
        pixels = np.stack([xs, ys], axis=-1).astype(np.float64)
        uv = native.eye_to_window(pinhole.window_to_eye(pixels))
        valid = np.isfinite(uv).all(axis=-1)
        map_x[start:stop] = np.where(valid, uv[..., 0], -1)
        map_y[start:stop] = np.where(valid, uv[..., 1], -1)
    map_x.setflags(write=False)
    map_y.setflags(write=False)
    return map_x, map_y, intrinsics


def rectification_maps(raw: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cached read-only ``cv2.remap`` maps ``(map_x, map_y)`` and the pinhole ``K``."""
    key = json.dumps(clean_calibration(raw), sort_keys=True, separators=(",", ":"))
    return _maps(key)


def rectify(
    image: np.ndarray, raw: dict[str, Any], *, mask: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Warp a native fisheye image to the pinhole camera at native size.

    Args:
        image: Native image, ``(H, W)`` or ``(H, W, C)``, matching the calibration size.
        raw: Calibration mapping accepted by ``clean_calibration``.
        mask: Use nearest-neighbor sampling and binarize to 0 or 255 instead of bilinear.

    Returns:
        The rectified image (gray input becomes 3-channel unless ``mask``) and a copy of ``K``.

    Raises:
        ValueError: If the image size differs from the calibration.
    """
    cleaned = clean_calibration(raw)
    if image.shape[:2] != (cleaned["image_height"], cleaned["image_width"]):
        raise ValueError("native image/calibration size mismatch")
    map_x, map_y, intrinsics = rectification_maps(cleaned)
    return apply_rectification(image, map_x, map_y, mask=mask), intrinsics.copy()


def apply_rectification(
    image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, *, mask: bool = False
) -> np.ndarray:
    """Remap a native image with rectification maps (``rectification_maps``).

    Args:
        image: Native image, ``(H, W)`` or ``(H, W, C)``.
        map_x: ``cv2.remap`` x map.
        map_y: ``cv2.remap`` y map.
        mask: Use nearest-neighbor sampling and binarize to 0 or 255 instead of bilinear.

    Returns:
        The rectified image, black outside the native view; a gray image becomes 3-channel
        BGR unless ``mask``.
    """
    interpolation = cv2.INTER_NEAREST if mask else cv2.INTER_LINEAR
    result = cv2.remap(
        image, map_x, map_y, interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    if mask:
        return (result > _MASK_THRESHOLD).astype(np.uint8) * 255
    if result.ndim == 2:
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return result


def project_rectified(
    points_mm: np.ndarray, rotation: np.ndarray, translation_mm: np.ndarray, intrinsics: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project object-frame points through the rectified pinhole intrinsics.

    Returns:
        Pixel coordinates ``(N, 2)`` and camera-frame depth ``(N,)`` in millimeters.
    """
    xyz = np.asarray(points_mm) @ np.asarray(rotation).T + np.asarray(translation_mm).reshape(3)
    homogeneous = xyz @ np.asarray(intrinsics).T
    return homogeneous[:, :2] / homogeneous[:, 2:3], xyz[:, 2]


def project_native(
    points_mm: np.ndarray,
    rotation: np.ndarray,
    translation_mm: np.ndarray,
    calibration: dict[str, Any],
    *,
    native: Any | None = None,  # noqa: ANN401 - vendored camera model
) -> np.ndarray:
    """Project object-frame points through the original fisheye camera model.

    Args:
        points_mm: ``(N, 3)`` object-frame points in millimeters.
        rotation: ``(3, 3)`` object-to-camera rotation.
        translation_mm: Object-to-camera translation (3 values) in millimeters.
        calibration: Raw calibration accepted by ``camera_pair``.
        native: The native camera ``camera_pair(calibration)`` builds, when the caller already
            has it (repeated projections); ``calibration`` is then not re-parsed.

    Returns:
        Pixel coordinates ``(N, 2)`` in the native image.
    """
    camera_model = camera_pair(calibration)[0] if native is None else native
    xyz = (
        np.asarray(points_mm) @ np.asarray(rotation).T + np.asarray(translation_mm).reshape(3)
    ) / MM_PER_M
    return camera_model.eye_to_window(xyz)


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    """Return the tight ``[x, y, w, h]`` box of a mask's non-zero pixels, or ``None`` if empty."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
    ]
