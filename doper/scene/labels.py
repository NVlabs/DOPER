# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""COCO-keypoint records in the exact schema written by ``rendering/synth_worker.py``.

``rendering.dataset_contract.merge_annotations`` consumes the per-worker file
produced by :func:`worker_payload`, and ``doper/dataset.py`` consumes the
merged ``annotations.json`` plus ``masks/`` and ``keypoints_3d.json``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from doper.scene.geometry import Intrinsics


def keypoints_bbox(
    uv: np.ndarray, vis: np.ndarray, width: int, height: int, pad: float = 0.05
) -> list[float]:
    """Padded box around keypoints with ``vis > 0`` (Blender ``visible_bbox``)."""
    sel = vis > 0
    if not sel.any():
        return [0.0, 0.0, 0.0, 0.0]
    xs, ys = uv[sel, 0], uv[sel, 1]
    x_min, x_max, y_min, y_max = xs.min(), xs.max(), ys.min(), ys.max()
    pad_x, pad_y = (x_max - x_min) * pad, (y_max - y_min) * pad
    left, top = max(0.0, x_min - pad_x), max(0.0, y_min - pad_y)
    right, bottom = min(float(width), x_max + pad_x), min(float(height), y_max + pad_y)
    return [float(left), float(top), float(right - left), float(bottom - top)]


def mask_bbox(mask: np.ndarray) -> list[float]:
    """Tight box around the nonzero pixels of ``mask`` (x, y, w, h in pixels)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(xs.min()),
        float(ys.min()),
        float(xs.max() - xs.min() + 1),
        float(ys.max() - ys.min() + 1),
    ]


def flatten_keypoints(uv: np.ndarray, vis: np.ndarray) -> list[float]:
    """Flatten keypoints to the COCO ``[x, y, v, ...]`` layout."""
    flat: list[float] = []
    for (x, y), v in zip(uv, vis, strict=True):
        flat.extend([round(float(x), 2), round(float(y), 2), int(v)])
    return flat


def image_record(
    image_id: int,
    file_name: str,
    width: int,
    height: int,
    *,
    pair_id: int | None,
    view: str,
    camera_index: int,
) -> dict[str, Any]:
    """COCO image entry with the stereo pairing fields."""
    return {
        "id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height,
        "pair_id": pair_id,
        "view": view,
        "camera_index": camera_index,
    }


def annotation_record(
    image_id: int,
    *,
    uv: np.ndarray,
    vis: np.ndarray,
    bbox: Sequence[float],
    intr: Intrinsics,
    cam_to_world: np.ndarray,
    object_pose: np.ndarray,
    view: str,
    camera_index: int,
    mask_file: str | None = None,
    stereo: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """COCO annotation entry for one rendered image."""
    bbox = [round(float(b), 2) for b in bbox]
    ann: dict[str, Any] = {
        "id": image_id,
        "image_id": image_id,
        "category_id": 1,
        "keypoints": flatten_keypoints(uv, vis),
        "num_keypoints": int(np.count_nonzero(vis > 0)),
        "bbox": bbox,
        "area": round(bbox[2] * bbox[3], 2),
        "iscrowd": 0,
        "camera_intrinsics": {
            "fx": round(intr.fx, 4),
            "fy": round(intr.fy, 4),
            "cx": round(intr.cx, 4),
            "cy": round(intr.cy, 4),
        },
        "camera_extrinsics": [[float(v) for v in row] for row in np.asarray(cam_to_world)],
        "object_pose": [[float(v) for v in row] for row in np.asarray(object_pose)],
        "view": view,
        "camera_index": camera_index,
    }
    if mask_file is not None:
        ann["mask_file"] = mask_file
        ann["mask_type"] = "visible_object"
    if stereo:
        ann.update(stereo)
    return ann


def worker_payload(
    *,
    worker_id: int,
    start_idx: int,
    end_idx: int,
    images: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    render_masks: bool,
    mask_dir: str,
    stereo_enabled: bool,
    stereo_baseline_m: float,
) -> dict[str, Any]:
    """Per-worker annotation file merged by the launcher."""
    return {
        "worker_id": worker_id,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "num_samples": end_idx - start_idx,
        "images_per_sample": 2 if stereo_enabled else 1,
        "render_masks": render_masks,
        "mask_dir": mask_dir if render_masks else None,
        "stereo": {
            "enabled": stereo_enabled,
            "baseline_m": round(stereo_baseline_m, 6) if stereo_enabled else None,
            "mode": "parallel" if stereo_enabled else None,
        },
        "images": images,
        "annotations": annotations,
    }


def camera_intrinsics_payload(
    intr: Intrinsics, focal_length_mm: float, sensor_width_mm: float
) -> dict[str, Any]:
    """Camera intrinsics block recorded next to the dataset."""
    fx, fy, cx, cy = (round(v, 6) for v in (intr.fx, intr.fy, intr.cx, intr.cy))
    return {
        "image_width": intr.width,
        "image_height": intr.height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "camera_matrix": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
        "camera_model": "pinhole",
        "lens_mm": focal_length_mm,
        "sensor_width_mm": sensor_width_mm,
    }


def keypoints_payload(
    object_name: str, keypoint_names: Sequence[str], keypoints_3d: Sequence[Sequence[float]]
) -> dict[str, Any]:
    """Build the repository ``keypoints_3d.json`` payload.

    Same payload as ``rendering.usd_mesh.keypoints_payload``, which stays for Blender-side callers.

    Args:
        object_name: Name of the source mesh object.
        keypoint_names: Keypoint names in model output-channel order.
        keypoints_3d: Corresponding mesh-local positions in meters.

    Returns:
        JSON-compatible payload with positions rounded to six decimal places.
    """
    return {
        "object": str(object_name),
        "num_keypoints": len(keypoint_names),
        "keypoint_names": list(keypoint_names),
        "keypoints_3d": [[round(float(c), 6) for c in pos] for pos in keypoints_3d],
    }
