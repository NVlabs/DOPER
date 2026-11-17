# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent audits of converted BOP data: CAD geometry, projections and checkpoints.

The audits do not trust the converter. :func:`geometry_audit` checks that the keypoints sit on
the original BOP CAD vertices (units and frame), :func:`projection_audit` re-projects the
original BOP poses through the original intrinsics and compares them with the stored labels and
masks, and :func:`checkpoint_is_finite` rejects diverged checkpoints. The audit functions return
the list of problems (empty when clean); the ``*_report`` functions return the measurements.
"""

from __future__ import annotations

import io
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from doper.bop.sources import sha256_file

MAX_VERTEX_DISTANCE_MM = 0.05
MIN_EXTENT_RATIO = 0.5
MAX_EXTENT_RATIO = 1.001
EXTENT_ATOL_MM = 1.0
EXTENT_RTOL = 0.01
MAX_REPROJECTION_PX = 0.1
MAX_OFF_MASK_FRACTION = 0.25
PROJECTION_SAMPLES = 64
MASK_DILATION_PX = 5
OVERLAY_LIMIT = 8


def load_bop_mesh(mesh_path: str | Path) -> Any:  # noqa: ANN401
    """Load CAD triangles with vertices preserved, tolerating ASCII PLY edge elements.

    ``trimesh`` fails on ASCII PLY files that carry a non-triangle ``edge`` element. Those rows
    are irrelevant to keypoints, so they are dropped while the vertex and face rows stay
    byte-for-byte as they are, with no centering or scaling.

    Args:
        mesh_path: PLY (or other ``trimesh`` readable) file.

    Returns:
        A ``trimesh.Trimesh`` with ``process=False``.

    Raises:
        ImportError: If ``trimesh`` is missing; install ``doper[bop]``.
        ValueError: If the ASCII PLY row count is inconsistent with its header.
    """
    try:
        import trimesh
    except ImportError as err:
        raise ImportError("trimesh is required to read BOP CAD models; install doper[bop]") from err
    try:
        return trimesh.load(mesh_path, force="mesh", process=False)
    except AttributeError as err:
        if "dtype" not in str(err):
            raise
        raw = Path(mesh_path).read_bytes()
        header, separator, body = raw.partition(b"end_header\n")
        if not separator or b"format ascii 1.0" not in header:
            raise
        fixed = _drop_edge_element(header, separator, body)
        return trimesh.load(io.BytesIO(fixed), file_type="ply", force="mesh", process=False)


def _drop_edge_element(header: bytes, separator: bytes, body: bytes) -> bytes:
    """Remove the ``edge`` element from an ASCII PLY, keeping every other row unchanged."""
    kept_header: list[bytes] = []
    elements: list[tuple[bytes, int]] = []
    skip = False
    for line in header.splitlines(keepends=True):
        words = line.split()
        if words and words[0] == b"element":
            skip = words[1] == b"edge"
            elements.append((words[1], int(words[2])))
        if not skip:
            kept_header.append(line)
    rows = body.splitlines(keepends=True)
    kept_rows: list[bytes] = []
    index = 0
    for name, count in elements:
        if name != b"edge":
            kept_rows.extend(rows[index : index + count])
        index += count
    if index != len(rows):
        raise ValueError("unexpected ASCII PLY row count; refuse geometry repair")
    return b"".join(kept_header) + separator + b"".join(kept_rows)


def geometry_report(
    mesh_path: str | Path,
    keypoints_path: str | Path,
    models_info: Mapping[str, Any],
    obj_id: int,
) -> dict[str, Any]:
    """Measure whether keypoints match the original BOP CAD vertices.

    ``models_info`` describes evaluation geometry, which can differ slightly from the full CAD,
    so it is a units sanity check and the CAD is never rescaled to it. Exact keypoint/CAD
    agreement remains strict.

    Args:
        mesh_path: Original BOP model (millimeters).
        keypoints_path: JSON with ordered ``keypoints_3d`` in meters.
        models_info: Parsed ``models_info.json``.
        obj_id: Object id keying ``models_info``.

    Returns:
        The measurements plus ``problems``, the list of failed checks (empty when clean).
    """
    from scipy.spatial import KDTree

    problems: list[str] = []
    vertices = np.asarray(load_bop_mesh(mesh_path).vertices, dtype=float)
    points = np.asarray(json.loads(Path(keypoints_path).read_text())["keypoints_3d"], dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) == 0
        or not np.isfinite(points).all()
    ):
        raise ValueError("expected finite ordered (N, 3) keypoints in meters")
    millimeters = points * 1000
    distances = KDTree(vertices).query(millimeters)[0]
    extent = np.ptp(vertices, axis=0)
    info = models_info[str(obj_id)]
    recorded = np.array([info[f"size_{axis}"] for axis in "xyz"])
    if not np.allclose(extent, recorded, atol=EXTENT_ATOL_MM, rtol=EXTENT_RTOL):
        problems.append(
            f"CAD extent {extent.tolist()} differs from models_info {recorded.tolist()}"
        )
    ratio = float(np.ptp(millimeters, axis=0).max() / extent.max())
    worst = float(distances.max())
    if worst > MAX_VERTEX_DISTANCE_MM or not MIN_EXTENT_RATIO <= ratio <= MAX_EXTENT_RATIO:
        problems.append(
            f"keypoint scale/frame mismatch: surface distance {worst} mm; extent ratio {ratio}"
        )
    return {
        "mesh_units": "mm",
        "keypoint_units": "m",
        "mesh_scale": 0.001,
        "frame": "original BOP model; no centering/offset/rotation",
        "mesh_extent_mm": extent.tolist(),
        "models_info_extent_mm": recorded.tolist(),
        "models_info_extent_delta_mm": (extent - recorded).tolist(),
        "keypoint_extent_ratio": ratio,
        "max_vertex_distance_mm": worst,
        "mesh_sha256": sha256_file(mesh_path),
        "keypoints_sha256": sha256_file(keypoints_path),
        "problems": problems,
    }


def geometry_audit(
    mesh_path: str | Path,
    keypoints_path: str | Path,
    models_info: Mapping[str, Any],
    obj_id: int,
) -> list[str]:
    """Return the problems of :func:`geometry_report` (empty when keypoints match the CAD)."""
    return geometry_report(mesh_path, keypoints_path, models_info, obj_id)["problems"]


def projection_report(
    converted: str | Path,
    source: str | Path,
    obj_id: int,
    overlay_dir: str | Path | None = None,
    *,
    allow_source_empty_masks: bool = False,
) -> dict[str, Any]:
    """Re-project the original BOP poses and compare them with the converted labels.

    A sample of the converted annotations is projected from ``scene_gt``/``scene_camera`` of the
    source (millimeters, OpenCV camera) and compared with the stored keypoints, then the visible
    keypoints are checked against the stored masks. T-LESS supplies valid all-zero visible
    masks; with ``allow_source_empty_masks`` such an annotation is kept, corroborated against
    the original mask and ``scene_gt_info``, and reported. Everything else fails closed.

    Args:
        converted: Output of :func:`doper.bop.convert.convert_pbr`.
        source: The BOP ``train_pbr`` directory the output was converted from.
        obj_id: Converted object id.
        overlay_dir: Directory for the first few contour/keypoint overlays; none when omitted.
        allow_source_empty_masks: Accept all-zero source masks that the source GT-info confirms.

    Returns:
        Statistics plus ``problems``, the list of failed checks (empty when clean).
    """
    import cv2

    converted, source = Path(converted), Path(source)
    data = json.loads((converted / "annotations.json").read_text())
    points_mm = (
        np.asarray(json.loads((converted / "keypoints_3d.json").read_text())["keypoints_3d"]) * 1000
    )
    images = {image["id"]: image for image in data["images"]}
    annotations = data["annotations"]
    sample = annotations[:: max(1, len(annotations) // PROJECTION_SAMPLES)][:PROJECTION_SAMPLES]
    problems: list[str] = []
    errors: list[float] = []
    outside: list[float] = []
    empty_masks: list[dict[str, Any]] = []
    metadata: dict[str, tuple[Any, ...]] = {}
    gt_info: dict[str, Any] = {}
    for annotation in sample:
        bop = annotation["bop"]
        scene = bop["scene"]
        if scene not in metadata:
            names = ("scene_gt.json", "scene_camera.json")
            metadata[scene] = tuple(
                json.loads((source / scene / name).read_text()) for name in names
            )
        scene_gt, scene_camera = metadata[scene]
        pose = scene_gt[str(bop["frame"])][bop["instance"]]
        label = f"annotation {annotation['id']} ({scene} frame {bop['frame']})"
        if pose["obj_id"] != obj_id:
            problems.append(f"{label}: source instance is object {pose['obj_id']}, not {obj_id}")
            continue
        camera_matrix = np.asarray(scene_camera[str(bop["frame"])]["cam_K"]).reshape(3, 3)
        rotation = np.asarray(pose["cam_R_m2c"]).reshape(3, 3)
        translation = np.asarray(pose["cam_t_m2c"]).reshape(3)
        homogeneous = (points_mm @ rotation.T + translation) @ camera_matrix.T
        uv = homogeneous[:, :2] / homogeneous[:, 2:3]
        stored = np.asarray(annotation["keypoints"]).reshape(-1, 3)
        valid = stored[:, 2] > 0
        if valid.any():
            errors.append(float(np.linalg.norm(uv[valid] - stored[valid, :2], axis=1).max()))
        record = images[annotation["image_id"]]
        image = cv2.imread(str(converted / "images" / record["file_name"]))
        if image is None or image.shape[:2] != (record["height"], record["width"]):
            problems.append(f"{label}: wrong image dimensions: {record['file_name']}")
            continue
        mask = cv2.imread(str(converted / annotation["mask_file"]), 0)
        if mask is None or mask.shape != image.shape[:2]:
            problems.append(f"{label}: missing/wrong-size visible mask: {annotation['mask_file']}")
            continue
        if not mask.any():
            reason = _check_empty_mask(
                annotation, bop, converted, source, gt_info, allow_source_empty_masks
            )
            if isinstance(reason, str):
                problems.append(f"{label}: {reason}")
                continue
            reason["converted_visible_keypoints"] = int((stored[:, 2] == 2).sum())
            empty_masks.append(reason)
        pixels = np.rint(uv[stored[:, 2] == 2]).astype(int)
        inside = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < record["width"])
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < record["height"])
        )
        grown = cv2.dilate(mask, np.ones((MASK_DILATION_PX,) * 2, np.uint8))
        on_mask = int((grown[pixels[inside, 1], pixels[inside, 0]] > 0).sum())
        if len(pixels):
            outside.append(1 - on_mask / len(pixels))
        if overlay_dir is not None and len(errors) <= OVERLAY_LIMIT:
            _write_overlay(Path(overlay_dir), annotation["id"], image, mask, pixels[inside])
    if not errors or max(errors) > MAX_REPROJECTION_PX:
        problems.append(f"BOP independent reprojection mismatch: {errors}")
    if not outside or float(np.mean(outside)) > MAX_OFF_MASK_FRACTION:
        problems.append(f"visible keypoints do not match pixels/masks: {outside}")
    report: dict[str, Any] = {
        "samples": len(sample),
        "max_reprojection_error_px": max(errors) if errors else math.nan,
        "mean_visible_off_mask_fraction_2px_tolerance": float(np.mean(outside))
        if outside
        else math.nan,
        "image_count": len(images),
        "annotation_count": len(annotations),
        "problems": problems,
    }
    if allow_source_empty_masks:
        report["source_empty_visible_masks"] = empty_masks
    return report


def _check_empty_mask(
    annotation: Mapping[str, Any],
    bop: Mapping[str, Any],
    converted: Path,
    source: Path,
    gt_info: dict[str, Any],
    allowed: bool,
) -> dict[str, Any] | str:
    """Return the evidence record of an all-zero mask, or the reason it is not acceptable."""
    if not allowed:
        return f"missing/empty/wrong-size visible mask: {annotation['mask_file']}"
    scene, frame, instance = bop["scene"], bop["frame"], bop["instance"]
    original = source / scene / "mask_visib" / f"{frame:06d}_{instance:06d}.png"
    converted_mask = converted / annotation["mask_file"]
    if not original.is_file() or sha256_file(original) != sha256_file(converted_mask):
        return f"empty visible mask differs from original: {annotation['mask_file']}"
    if scene not in gt_info:
        gt_info[scene] = json.loads((source / scene / "scene_gt_info.json").read_text())
    info = gt_info[scene][str(frame)][instance]
    if info.get("px_count_visib") != 0 or info.get("visib_fract") != 0:
        return f"empty visible mask conflicts with original GT-info: {annotation['mask_file']}"
    return {
        "annotation_id": annotation["id"],
        "bop": dict(bop),
        "mask_file": annotation["mask_file"],
        "source_mask_sha256": sha256_file(original),
        "source_gt_info": info,
        "action": "retained unchanged; included in projection/mask diagnostics",
    }


def _write_overlay(
    directory: Path, annotation_id: int, image: np.ndarray, mask: np.ndarray, pixels: np.ndarray
) -> None:
    """Write the mask contour and the visible keypoints over the image."""
    import cv2

    overlay = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
    for point in pixels:
        cv2.circle(overlay, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
    directory.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(directory / f"projection_{annotation_id:06d}.jpg"), overlay)


def projection_audit(
    converted: str | Path,
    source: str | Path,
    obj_id: int,
    overlay_dir: str | Path | None = None,
    *,
    allow_source_empty_masks: bool = False,
) -> list[str]:
    """Return the problems of :func:`projection_report` (empty when the labels are consistent)."""
    report = projection_report(
        converted, source, obj_id, overlay_dir, allow_source_empty_masks=allow_source_empty_masks
    )
    return report["problems"]


def is_finite_tree(value: object) -> bool:
    """Return whether every tensor and float inside nested containers is finite."""
    import torch

    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(is_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(is_finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def checkpoint_is_finite(path: str | Path) -> bool:
    """Return whether a checkpoint's model and optimizer state hold only finite numbers.

    Args:
        path: Checkpoint written by the training loop.

    Returns:
        False if any weight, optimizer value or recorded float is NaN or infinite, or if the
        optimizer has no state (an untrained checkpoint).
    """
    from doper.checkpoint_io import load_checkpoint

    checkpoint = load_checkpoint(path)
    optimizer = checkpoint["optimizer"]
    if not optimizer.get("state"):
        return False
    return is_finite_tree(checkpoint["model"]) and is_finite_tree(optimizer)
