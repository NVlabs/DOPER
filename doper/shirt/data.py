# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SHIRT trajectories: labels, camera, metadata and the packaged profile.

A SHIRT tree has the public release's layout::

    <shirt_root>/camera.json                       cameraMatrix (3x3), distCoeffs, Nu, Nv
    <shirt_root>/<roe>/<roe>.json                  labels, one per frame, in frame order
    <shirt_root>/<roe>/metadata.json               simulation and servicer ephemeris
    <shirt_root>/<roe>/<domain>/images/<filename>  domain: synthetic or lightbox

A label's frame index is its position in the label list (there is no timestamp; the filters
use ``index * metadata.pSim.cam_step``). Labels carry ``filename`` (or ``file_name``), the
scalar-first ``q_vbs2tango_true`` (or ``quaternion``) and the camera-frame translation
``r_Vo2To_vbs_true`` (or ``translation``) in meters; the list may be wrapped in ``images``,
``labels`` or ``poses``. The ground truth is for reporting only.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..recipes import profile_path
from ..speedplus.geometry import DISTORTION_LENGTHS, Camera

CAMERA_NAME = "camera.json"
METADATA_NAME = "metadata.json"
IMAGES_DIR = "images"
LABEL_WRAPPERS = ("images", "labels", "poses")
DEFAULT_DISTORTION_LENGTH = 5


@dataclass(frozen=True)
class Label:
    """One SHIRT frame's label.

    Attributes:
        index: Frame index, the label's position in the trajectory.
        filename: Image file name.
        quaternion: ``(4,)`` scalar-first ``q_vbs2tango_true``.
        translation: ``(3,)`` ``r_Vo2To_vbs_true`` in meters, camera frame.
    """

    index: int
    filename: str
    quaternion: np.ndarray
    translation: np.ndarray


@dataclass(frozen=True)
class ShirtProfile:
    """The packaged SHIRT profile (``doper/recipes/profiles/shirt.json``).

    Attributes:
        name: Profile name.
        trajectories: Released trajectories.
        domains: Released image domains.
        paper_domain: The domain of the paper's Table 2.
        image_size: ``(width, height)`` of the released images.
        path: The profile file.
    """

    name: str
    trajectories: tuple[str, ...]
    domains: tuple[str, ...]
    paper_domain: str
    image_size: tuple[int, int]
    path: Path


def load_profile(name_or_path: str | Path = "shirt") -> ShirtProfile:
    """Load the packaged SHIRT profile, or a profile file by path."""
    path = Path(name_or_path)
    if not path.is_file():
        path = profile_path(str(name_or_path))
    payload = json.loads(path.read_text())
    width, height = payload["image_size"]
    return ShirtProfile(
        name=payload["name"],
        trajectories=tuple(payload["trajectories"]),
        domains=tuple(payload["domains"]),
        paper_domain=payload["paper_domain"],
        image_size=(int(width), int(height)),
        path=path,
    )


def _label_list(payload: Any, path: Path) -> list[Mapping[str, Any]]:  # noqa: ANN401
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        for key in LABEL_WRAPPERS:
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"unsupported SHIRT label schema in {path}")


def label_fields(label: Mapping[str, Any]) -> tuple[str, np.ndarray, np.ndarray]:
    """The file name, scalar-first quaternion and translation of one label.

    Raises:
        ValueError: If a field is missing under both its names.
    """
    filename = label.get("filename", label.get("file_name"))
    quaternion = label.get("q_vbs2tango_true", label.get("quaternion"))
    translation = label.get("r_Vo2To_vbs_true", label.get("translation"))
    if filename is None or quaternion is None or translation is None:
        raise ValueError(f"SHIRT label is missing pose fields: {sorted(label)}")
    return (
        str(filename),
        np.asarray(quaternion, dtype=np.float64),
        np.asarray(translation, dtype=np.float64),
    )


def load_labels(roe_dir: str | Path) -> list[Label]:
    """Load ``<roe_dir>/<roe>.json``; a label's index is its position in the list.

    Raises:
        ValueError: For an unsupported schema or a label without its pose fields.
    """
    roe_dir = Path(roe_dir)
    path = roe_dir / f"{roe_dir.name}.json"
    labels = _label_list(json.loads(path.read_text()), path)
    return [Label(index, *label_fields(label)) for index, label in enumerate(labels)]


def _intrinsics(payload: Mapping[str, Any], path: Path) -> tuple[Any, Any]:  # noqa: ANN401
    """The camera matrix and distortion of either camera schema."""
    if "cameraMatrix" in payload:
        return payload["cameraMatrix"], payload.get("distCoeffs")
    if all(key in payload for key in ("fx", "fy", "cx", "cy")):
        matrix = [[payload["fx"], 0.0, payload["cx"]], [0.0, payload["fy"], payload["cy"]],
                  [0.0, 0.0, 1.0]]  # fmt: skip
        return matrix, payload.get("distCoeffs", payload.get("distortion"))
    raise ValueError(f"unsupported SHIRT camera schema in {path}: {sorted(payload)}")


def load_camera(shirt_root: str | Path) -> Camera:
    """Load ``<shirt_root>/camera.json``: the intrinsics, distortion and image size.

    The file carries ``cameraMatrix`` or ``fx``, ``fy``, ``cx`` and ``cy``, with ``distCoeffs``
    (or ``distortion``; missing means none), and ``Nu`` and ``Nv``, the image width and height
    (``Camera.width`` and ``Camera.height``) that the re-crop stages clip to.

    Raises:
        ValueError: For an unsupported schema, a non-3x3 matrix, a distortion length OpenCV
            does not take, or a missing ``Nu`` or ``Nv``.
    """
    path = Path(shirt_root) / CAMERA_NAME
    payload = json.loads(path.read_text())
    matrix, distortion = _intrinsics(payload, path)
    matrix_arr = np.asarray(matrix, dtype=np.float64)
    distortion_arr = np.asarray(
        np.zeros(DEFAULT_DISTORTION_LENGTH) if distortion is None else distortion,
        dtype=np.float64,
    ).reshape(-1)
    if matrix_arr.shape != (3, 3):
        raise ValueError(f"camera matrix must be 3x3, got {matrix_arr.shape}")
    if distortion_arr.size not in DISTORTION_LENGTHS:
        raise ValueError(f"unsupported OpenCV distortion length {distortion_arr.size}")
    if payload.get("Nu") is None or payload.get("Nv") is None:
        raise ValueError(f"SHIRT camera file has no image size (Nu and Nv): {path}")
    return Camera(matrix_arr, distortion_arr, int(payload["Nu"]), int(payload["Nv"]))


def load_metadata(roe_dir: str | Path) -> dict[str, Any]:
    """Load ``<roe_dir>/metadata.json``: the simulation step and the servicer ephemeris."""
    return json.loads((Path(roe_dir) / METADATA_NAME).read_text())


def image_path(shirt_root: str | Path, roe: str, domain: str, filename: str) -> Path:
    """The path of one frame's image, ``<shirt_root>/<roe>/<domain>/images/<filename>``."""
    return Path(shirt_root) / roe / domain / IMAGES_DIR / filename
