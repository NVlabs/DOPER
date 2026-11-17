# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic mini COCO dataset in the layout ``DoperDataset`` reads."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from doper.config import ModelConfig

NUM_KEYPOINTS = 64
OBJECT_EXTENT_M = 0.12
OBJECT_DEPTH_M = 0.5
FOCAL_SCALE = 1.2


def camera_matrix(size: int) -> np.ndarray:
    """Pinhole intrinsics used by the fixture for a square image of side ``size``."""
    focal = FOCAL_SCALE * size
    return np.array([[focal, 0.0, size / 2.0], [0.0, focal, size / 2.0], [0.0, 0.0, 1.0]])


def keypoints_3d(seed: int = 0) -> np.ndarray:
    """Return the deterministic ``(64, 3)`` object keypoints in meters."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-OBJECT_EXTENT_M, OBJECT_EXTENT_M, size=(NUM_KEYPOINTS, 3))


def coco_annotation(
    ann_id: int,
    image_id: int,
    bbox: Sequence[float],
    area: float,
    keypoints: Sequence[float],
    **extra: Any,
) -> dict[str, Any]:
    """Return a single-category COCO annotation; ``extra`` keys are appended or override."""
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": 1,
        "bbox": list(bbox),
        "area": area,
        "keypoints": list(keypoints),
        "num_keypoints": len(keypoints) // 3,
        **extra,
    }


def write_coco_split(
    split_dir: Path,
    images: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    categories: Sequence[Mapping[str, Any]],
    files: Mapping[str, np.ndarray] | None = None,
    indent: int | None = None,
) -> None:
    """Write ``annotations.json`` and the pixel files of one split under ``split_dir``.

    ``files`` maps split-relative paths (``images/a.png``, ``masks/a.png``) to arrays that
    are written with ``cv2.imwrite``. The ``images`` directory always exists.
    """
    split_dir = Path(split_dir)
    (split_dir / "images").mkdir(parents=True, exist_ok=True)
    for relative, array in (files or {}).items():
        path = split_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(path), array)
    coco = {"images": images, "annotations": annotations, "categories": categories}
    (split_dir / "annotations.json").write_text(json.dumps(coco, indent=indent))


def tiny_model_config(**overrides: Any) -> ModelConfig:
    """Return a CPU-friendly ``ModelConfig`` (resnet18, no pretrained weights) with overrides."""
    return ModelConfig(**{"backbone": "resnet18", "pretrained": False, **overrides})


def curve_fields(
    names: Sequence[str], geometry_sha256: str, object_id: str = "part", **overrides: Any
) -> dict[str, Any]:
    """Return ``ModelConfig`` overrides that enable the named-curve head."""
    return {
        "curve_head_enabled": True,
        "num_curves": len(names),
        "curve_schema_version": 1,
        "curve_names": tuple(names),
        "curve_geometry_sha256": geometry_sha256,
        "curve_object_id": object_id,
        **overrides,
    }


def _rotation(rng: np.random.Generator) -> np.ndarray:
    rvec = rng.uniform(-0.6, 0.6, size=3)
    return cv2.Rodrigues(rvec)[0]


def _write_split(split_dir: Path, n_images: int, size: int, seed: int, kp3d: np.ndarray) -> None:
    rng = np.random.default_rng(seed)
    intrinsics = camera_matrix(size)
    images, annotations, files = [], [], {}
    for i in range(n_images):
        pose = np.eye(4)
        pose[:3, :3] = _rotation(rng)
        pose[:3, 3] = [rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), OBJECT_DEPTH_M]
        cam = kp3d @ pose[:3, :3].T + pose[:3, 3]
        uv = (intrinsics @ cam.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        mask = np.zeros((size, size), np.uint8)
        hull = cv2.convexHull(np.round(uv).astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        image = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
        image[mask > 0] = rng.integers(0, 256, size=3, dtype=np.uint8)
        name = f"{i:04d}.png"
        files[f"images/{name}"] = image
        files[f"masks/{name}"] = mask
        x0, y0 = uv.min(axis=0)
        x1, y1 = uv.max(axis=0)
        images.append({"id": i + 1, "file_name": name, "width": size, "height": size})
        annotations.append(
            {
                "id": i + 1,
                "image_id": i + 1,
                "category_id": 1,
                "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                "area": float((x1 - x0) * (y1 - y0)),
                "is_primary": True,
                "mask_file": f"masks/{name}",
                "keypoints": np.concatenate([uv, np.full((NUM_KEYPOINTS, 1), 2.0)], axis=1)
                .reshape(-1)
                .tolist(),
                "object_pose": pose.tolist(),
                "camera_intrinsics": {
                    "fx": float(intrinsics[0, 0]),
                    "fy": float(intrinsics[1, 1]),
                    "cx": float(intrinsics[0, 2]),
                    "cy": float(intrinsics[1, 2]),
                },
            }
        )
    categories = [{"id": 1, "name": "object"}]
    write_coco_split(split_dir, images, annotations, categories, files, indent=1)


def make_mini_coco(root: Path, n_images: int = 4, size: int = 128, seed: int = 0) -> Path:
    """Write a tiny COCO dataset (train and val splits) under ``root`` and return ``root``.

    Every image holds one object whose 64 keypoints are projected from a known pose
    through known intrinsics; the mask is the keypoint convex hull so that the dataset
    consistency checks (pose reprojection, keypoint/mask agreement) pass.
    """
    root = Path(root)
    kp3d = keypoints_3d(seed)
    for split in ("train", "val"):
        _write_split(root / split, n_images, size, seed, kp3d)
    (root / "keypoints_3d.json").write_text(json.dumps({"keypoints_3d": kp3d.tolist()}))
    return root


def write_tiny_checkpoint(path: Path, model_cfg: ModelConfig, seed: int = 0) -> Path:
    """Save a random-init ``DoperNet`` with ``cfg.model`` metadata, as training would.

    The width/height head bias is raised so the untrained detector proposes boxes of a
    usable size instead of degenerate ones.
    """
    import torch

    from doper.model import DoperNet

    torch.manual_seed(seed)
    model = DoperNet(model_cfg)
    with torch.no_grad():
        model.wh_head.bias.add_(6.0)
    torch.save({"model": model.state_dict(), "cfg": {"model": asdict(model_cfg)}}, path)
    return Path(path)


DEPLOYMENT_NUM_KEYPOINTS = 6
DEPLOYMENT_DET_SIZE = 96
DEPLOYMENT_CROP_SIZE = 32


def write_deployment_inputs(root: Path, n_frames: int = 2) -> dict[str, Any]:
    """Write a tiny heatmap checkpoint, keypoints, camera and frames for bundle export.

    Returns:
        ``checkpoint``, ``keypoints_3d`` and ``camera`` paths, ``frames`` (BGR arrays,
        non-square so the detector letterboxes) and the ``model_cfg``.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    model_cfg = tiny_model_config(
        num_keypoints=DEPLOYMENT_NUM_KEYPOINTS,
        det_input_size=DEPLOYMENT_DET_SIZE,
        letterbox=True,
        neck_channels=8,
        head_channels=8,
        crop_size=DEPLOYMENT_CROP_SIZE,
        keypoint_head="heatmap",
    )
    checkpoint = write_tiny_checkpoint(root / "best.pth", model_cfg)
    points = np.random.default_rng(0).uniform(-0.05, 0.05, (DEPLOYMENT_NUM_KEYPOINTS, 3))
    keypoints = root / "keypoints_3d.json"
    keypoints.write_text(json.dumps({"keypoints_3d": points.tolist()}))
    camera = root / "camera.json"
    camera.write_text(json.dumps({"camera_matrix": camera_matrix(128).tolist()}))
    rng = np.random.default_rng(1)
    frames = [rng.integers(0, 256, (96, 128, 3), dtype=np.uint8) for _ in range(n_frames)]
    return {
        "checkpoint": checkpoint,
        "keypoints_3d": keypoints,
        "camera": camera,
        "frames": frames,
        "model_cfg": model_cfg,
    }
