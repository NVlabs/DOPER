# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic run trees for the bank and selection tests (no model weights, only bytes)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from doper.bop import layout

GEOMETRY = b'{"keypoints_3d": [[0.0, 0.0, 0.0]]}'


def sha256(path: Path) -> str:
    """SHA-256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_base(runs: Path, obj: int) -> Path:
    """A completed base run: ``last.pth``, geometry and its ``completed.json``."""
    run = layout.object_run(runs, "base", obj)
    _write_weights(run, f"base{obj}".encode())
    done = {
        "stage": "base",
        "obj_id": obj,
        "epochs": 25,
        "checkpoint_sha256": sha256(run / layout.CHECKPOINT_PATH),
        "keypoints_sha256": sha256(run / layout.KEYPOINTS_PATH),
    }
    (run / layout.COMPLETED_NAME).write_text(json.dumps(done))
    return run


def write_student(runs: Path, method: str, obj: int, labels: int = 30) -> Path:
    """A completed three-epoch student of the object's base under ``<runs>/<method>``."""
    base = layout.object_run(runs, "base", obj)
    run = layout.object_run(runs, method, obj)
    _write_weights(run, f"{method}{obj}".encode())
    done = {
        "stage": "adapt",
        "method": method.split("/")[0],
        "obj_id": obj,
        "completed_epochs": 3,
        "labels": labels,
        "checkpoint_sha256": sha256(run / layout.CHECKPOINT_PATH),
        "student_sha256": sha256(run / layout.CHECKPOINT_PATH),
        "keypoints_sha256": sha256(run / layout.KEYPOINTS_PATH),
        "base_teacher_sha256": sha256(base / layout.CHECKPOINT_PATH),
    }
    (run / layout.COMPLETED_NAME).write_text(json.dumps(done))
    return run


def write_skip(runs: Path, method: str, obj: int, labels: int, minimum: int = 20) -> Path:
    """A method skipped below its minimum: ``adaptation_skipped.json`` and no student."""
    run = layout.object_run(runs, method, obj)
    run.mkdir(parents=True, exist_ok=True)
    record = {"reason": "below_minimum", "labels": labels, "minimum_labels": minimum}
    (run / layout.SKIPPED_NAME).write_text(json.dumps(record))
    return run


def write_tree(
    runs: Path,
    objects: range | list[int],
    methods: tuple[str, ...],
    skips: dict[str, dict[int, int]],
) -> None:
    """Base and every method for every object; ``skips[method][obj]`` is a skip's label count."""
    for obj in objects:
        write_base(runs, obj)
        for method in methods:
            skipped = skips.get(method, {})
            if obj in skipped:
                write_skip(runs, method, obj, skipped[obj])
            else:
                write_student(runs, method, obj)


def write_test_split(root: Path, frames: int, cam_k: list[float], size: tuple[int, int]) -> Path:
    """A one-scene BOP test split ``000001``: flat PNG frames and their ``scene_camera.json``.

    Frame ``n`` is filled with ``20 n + 1``; ``size`` is ``(width, height)``.
    """
    import cv2
    import numpy as np

    folder = root / "000001"
    (folder / "rgb").mkdir(parents=True)
    camera = {}
    for frame in range(frames):
        image = np.full((size[1], size[0], 3), 20 * frame + 1, np.uint8)
        cv2.imwrite(str(folder / "rgb" / f"{frame:06d}.png"), image)
        camera[str(frame)] = {"cam_K": cam_k}
    (folder / "scene_camera.json").write_text(json.dumps(camera))
    return root


def _write_weights(run: Path, checkpoint: bytes) -> None:
    (run / "work").mkdir(parents=True, exist_ok=True)
    (run / "geometry").mkdir(exist_ok=True)
    (run / layout.CHECKPOINT_PATH).write_bytes(checkpoint)
    (run / layout.KEYPOINTS_PATH).write_bytes(GEOMETRY)


def raw_record(record: dict) -> dict:
    """A raw-detection journal record (cache schema 3) whose valid poses are ``record``'s.

    ``record`` is a valid-pose journal record (``predictions`` with ``score``, ``R``, ``t``);
    each prediction becomes a kept, valid detection carrying that exact ``R``/``t``, so the
    derived journal reproduces ``record``'s predictions bit for bit.
    """
    import cv2
    import numpy as np

    detections = []
    for j, pose in enumerate(record["predictions"]):
        rvec = cv2.Rodrigues(np.asarray(pose["R"], dtype=np.float64).reshape(3, 3))[0]
        detections.append(
            {
                "index": j,
                "score": pose["score"],
                "bbox_xyxy": [0.0, 0.0, 1.0, 1.0],
                "keypoints": [],
                "keypoint_scores": [],
                "rvec": rvec.tolist(),
                "tvec": [[float(v)] for v in pose["t"]],
                "R": pose["R"],
                "t": pose["t"],
                "status": "valid",
            }
        )
    return {
        "scene_id": record["scene_id"],
        "im_id": record["im_id"],
        "rgb_sha256": record["rgb_sha256"],
        "objects_scanned": 1,
        "seconds": record.get("seconds", 0.25),
        "detections_returned": len(detections),
        "detections": detections,
    }
