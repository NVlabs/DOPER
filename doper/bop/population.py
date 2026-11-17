# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation populations: BOP split directories and frozen evaluation manifests.

A population is either a BOP split directory (scenes with ``scene_gt.json``, ``scene_camera.json``,
``scene_gt_info.json`` and ``rgb/``) or a frozen evaluation manifest: ``frames`` with ``scene``,
``frame``, ``path``, ``sha256``, ``objects`` and either a ``val_root`` holding the scene files
(hashes under ``metadata``) or per-frame ``GT``, ``cam_K`` and, for the native camera,
``native_calibration`` (the HOT3D manifest built by ``doper.bop.hot3d_staging``). ``FrameLoader``
validates each target frame against it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from ..evaluation.bop import rgb_path
from .hot3d import camera_pair
from .sources import canonical_sha256, sha256_file

CameraName = Literal["pinhole", "native"]
Population = Mapping[str, Any]
"""An evaluation manifest (``build_population``): ``val_root``, ``metadata``, ``frames``."""
MIN_GT_BOX_SIDE_PX = 1
SCENE_FILES = ("scene_gt.json", "scene_camera.json", "scene_gt_info.json")


def build_population(split_dir: str | Path) -> dict[str, Any]:
    """Build an evaluation manifest from a BOP split directory.

    Scenes (directories with ``scene_gt.json``) and frames are sorted numerically; ``objects``
    lists each frame's GT object ids in ``scene_gt.json`` order. Image paths are relative to the
    split directory, so the population hash does not depend on where the data lives.

    Args:
        split_dir: Directory of BOP scenes.

    Returns:
        A manifest with ``val_root``, ``metadata`` (scene-file hashes) and ``frames``.

    Raises:
        ValueError: If no scene is found or a frame has no RGB image.
    """
    root = Path(split_dir)
    scenes = sorted(p for p in root.iterdir() if (p / "scene_gt.json").is_file())
    if not scenes:
        raise ValueError(f"{root}: no BOP scenes")
    frames, metadata = [], {}
    for scene in scenes:
        metadata[scene.name] = {name: sha256_file(scene / name) for name in SCENE_FILES}
        scene_gt = json.loads((scene / "scene_gt.json").read_text())
        for frame in sorted(scene_gt, key=int):
            image = rgb_path(scene, int(frame))
            if image is None:
                raise ValueError(f"{scene}: frame {frame} has no RGB image")
            frames.append(
                {
                    "scene": scene.name,
                    "frame": int(frame),
                    "path": str(image.relative_to(root)),
                    "sha256": sha256_file(image),
                    "objects": [g["obj_id"] for g in scene_gt[frame]],
                }
            )
    return {"val_root": str(root), "metadata": metadata, "frames": frames}


def load_population(split_root: str | Path) -> tuple[dict[str, Any], str]:
    """Load a validation population and its identity hash.

    Args:
        split_root: BOP split directory, or an evaluation manifest JSON file.

    Returns:
        The manifest and its SHA-256: of the file bytes for a manifest file, of the canonical
        JSON without ``val_root`` for a population built from a directory.
    """
    path = Path(split_root)
    if path.is_file():
        data = path.read_bytes()
        manifest = json.loads(data)
        manifest.setdefault("val_root", str(path.parent))
        return manifest, hashlib.sha256(data).hexdigest()
    manifest = build_population(path)
    return manifest, canonical_sha256({k: v for k, v in manifest.items() if k != "val_root"})


@dataclass(frozen=True)
class Frame:
    """One validated target frame."""

    record: Mapping[str, Any]
    matches: list[tuple[int, Mapping[str, Any]]]
    boxes: list[np.ndarray | None]
    camera_matrix: np.ndarray
    image: np.ndarray
    calibration: Mapping[str, Any] | None = None
    native: Any = None  # noqa: ANN401 - vendored camera model
    pinhole: Any = None  # noqa: ANN401 - vendored camera model


def box_xyxy(box: Sequence[float] | None) -> np.ndarray | None:
    """GT ``[x, y, w, h]`` as an ``x1, y1, x2, y2`` array, or None when too small."""
    if box is None:
        return None
    x, y, w, h = box
    if not (w > MIN_GT_BOX_SIDE_PX and h > MIN_GT_BOX_SIDE_PX):
        return None
    return np.array([x, y, x + w, y + h])


class FrameLoader:
    """Validate manifest frames and resolve their GT instances, camera and image."""

    def __init__(self, manifest: Mapping[str, Any], obj_id: int, camera: CameraName) -> None:
        """Prepare to load ``obj_id``'s frames of ``manifest`` (``load_population``)."""
        self.root = Path(manifest["val_root"])
        self.metadata = manifest.get("metadata", {})
        self.obj_id = obj_id
        self.camera = camera
        self.scenes: dict[str, list[dict[str, Any]]] = {}

    def load(self, record: Mapping[str, Any], *, single_target: bool) -> Frame:
        """Validate one manifest frame and return it with its target GT, boxes and image.

        Args:
            record: Manifest frame.
            single_target: Require exactly one instance of the object (``top1`` protocol).

        Raises:
            ValueError: If the GT layout, scene files, image or calibration do not validate.
        """
        indices = [i for i, o in enumerate(record["objects"]) if o == self.obj_id]
        if "GT" in record:
            matches, boxes, camera_matrix = self._embedded(record, indices)
        else:
            matches, boxes, camera_matrix = self._scene_files(record, indices, single_target)
        if single_target and len(matches) != 1:
            raise ValueError("top1 protocol requires one target instance per frame")
        image = self._image(record)
        if self.camera == "pinhole":
            return Frame(record, matches, boxes, camera_matrix, image)
        if "native_calibration" not in record:
            raise ValueError("native camera scoring needs per-frame native_calibration")
        native, pinhole, exact = camera_pair(record["native_calibration"])
        if not np.array_equal(camera_matrix, exact):
            raise ValueError("rectification/calibration mismatch")
        return Frame(
            record,
            matches,
            boxes,
            camera_matrix,
            image,
            record["native_calibration"],
            native,
            pinhole,
        )

    def _embedded(
        self, record: Mapping[str, Any], indices: list[int]
    ) -> tuple[list[tuple[int, Mapping[str, Any]]], list[np.ndarray | None], np.ndarray]:
        matches = [(i, record["GT"][i]) for i in indices]
        if any(g["obj_id"] != self.obj_id for _, g in matches):
            raise ValueError("GT instance ordering mismatch")
        boxes = [box_xyxy(g.get("rectified_amodal_bbox")) for _, g in matches]
        return matches, boxes, np.asarray(record["cam_K"]).reshape(3, 3)

    def _scene_files(
        self, record: Mapping[str, Any], indices: list[int], single_target: bool
    ) -> tuple[list[tuple[int, Mapping[str, Any]]], list[np.ndarray | None], np.ndarray]:
        name = record["scene"]
        if name not in self.scenes:
            scene = self.root / name
            expected = self.metadata.get(name)
            if not expected:
                raise ValueError(f"manifest has no scene-file hashes for scene {name}")
            if any(sha256_file(scene / n) != digest for n, digest in expected.items()):
                raise ValueError("validation metadata changed")
            self.scenes[name] = [json.loads((scene / n).read_text()) for n in SCENE_FILES]
        gts, cameras, infos = self.scenes[name]
        fid = str(record["frame"])
        matches = [(i, g) for i, g in enumerate(gts[fid]) if g["obj_id"] == self.obj_id]
        if not single_target and [i for i, _ in matches] != indices:
            raise ValueError("GT instance ordering changed")
        boxes = [box_xyxy(infos[fid][i]["bbox_visib"]) for i, _ in matches]
        return matches, boxes, np.asarray(cameras[fid]["cam_K"]).reshape(3, 3)

    def _image(self, record: Mapping[str, Any]) -> np.ndarray:
        path = Path(record["path"])
        path = path if path.is_absolute() else self.root / path
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"validation image changed: {path}")
        image = cv2.imread(str(path))
        expected = (record.get("height"), record.get("width"))
        if image is None or (None not in expected and image.shape[:2] != expected):
            raise ValueError(f"bad validation image: {path}")
        return image
