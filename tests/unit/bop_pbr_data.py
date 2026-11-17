# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic BOP ``train_pbr`` scenes and keypoint files for the BOP conversion tests."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 64, 48
CAMERA_MATRIX = np.array([[80.0, 0.0, 32.0], [0.0, 80.0, 24.0], [0.0, 0.0, 1.0]])
DEPTH_SCALE = 0.1
RAW_DEPTH = 3000
Instance = tuple[int, np.ndarray, tuple[float, ...]]


def write_keypoints(
    path: Path, *, points: list[list[float]] | None = None, **extra: object
) -> Path:
    """Write a keypoint JSON in meters (default: four points around the origin)."""
    points = points or [[0, 0, 0], [0.02, 0, 0], [0, 0.02, 0], [0, 0, -0.02]]
    payload = {
        "object": "obj_000002",
        "num_keypoints": len(points),
        "keypoint_names": [f"KP_{index}" for index in range(len(points))],
        "keypoints_3d": points,
        **extra,
    }
    path.write_text(json.dumps(payload))
    return path


def write_scene(root: Path, name: str, frames: dict[int, list[Instance]]) -> Path:
    """Write one BOP scene: JPG RGB, uint16 depth, visible masks, GT and camera JSON."""
    scene = root / name
    for directory in ("rgb", "depth", "mask_visib"):
        (scene / directory).mkdir(parents=True, exist_ok=True)
    gt: dict[str, list[dict]] = {}
    cameras: dict[str, dict] = {}
    for frame, instances in frames.items():
        assert cv2.imwrite(
            str(scene / "rgb" / f"{frame:06d}.jpg"),
            np.full((HEIGHT, WIDTH, 3), 80 + frame, np.uint8),
        )
        depth = np.full((HEIGHT, WIDTH), RAW_DEPTH, np.uint16)
        depth[:, : WIDTH // 4] = RAW_DEPTH - 500
        assert cv2.imwrite(str(scene / "depth" / f"{frame:06d}.png"), depth)
        gt[str(frame)] = []
        for index, (obj_id, rotation, translation) in enumerate(instances):
            gt[str(frame)].append(
                {
                    "obj_id": obj_id,
                    "cam_R_m2c": rotation.reshape(-1).tolist(),
                    "cam_t_m2c": list(translation),
                }
            )
            mask = np.zeros((HEIGHT, WIDTH), np.uint8)
            mask[10 + index : 30, 30:50] = 255
            assert cv2.imwrite(str(scene / "mask_visib" / f"{frame:06d}_{index:06d}.png"), mask)
        cameras[str(frame)] = {
            "cam_K": CAMERA_MATRIX.reshape(-1).tolist(),
            "depth_scale": DEPTH_SCALE,
        }
    (scene / "scene_gt.json").write_text(json.dumps(gt))
    (scene / "scene_camera.json").write_text(json.dumps(cameras))
    return scene


def make_pbr(root: Path, scene_names: tuple[str, ...] = ("000000", "000001", "000002")) -> Path:
    """Write ``root/train_pbr`` with three frames per scene, two objects, and return it."""
    pbr = root / "train_pbr"
    rotation = cv2.Rodrigues(np.array([0.2, -0.1, 0.3]))[0]
    for offset, name in enumerate(scene_names):
        write_scene(
            pbr,
            name,
            {
                frame: [
                    (2, rotation, (5.0 * offset, -3.0 * frame, 300.0)),
                    (7, np.eye(3), (60.0, 0.0, 320.0)),
                ]
                for frame in (0, 1, 2)
            },
        )
    return pbr
