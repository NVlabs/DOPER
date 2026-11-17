# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A tiny native SPEED+ v2 tree, its archive and a fixture profile with lowered pins.

The tree follows the extracted Zenodo layout (``camera.json``, ``LICENSE.md``,
``synthetic/{images,train.json,validation.json}``, ``{lightbox,sunlamp}/{images,test.json}``)
with small images and a camera scaled to them. One sunlamp pose puts part of the body outside
the frame, so visibility and hull clipping are exercised. ``golden/source_speedplus_convert.json``
was produced from this tree, so its values must not change.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.io import savemat
from speedplus_helpers import DIST, OFFICIAL_POINTS

WIDTH, HEIGHT = 192, 120
CAMERA_MATRIX = [[298.86, 0.0, 96.0], [0.0, 298.83, 60.0], [0.0, 0.0, 1.0]]
# Split -> (domain, label file, [(quaternion, translation), ...]).
POSES: dict[str, tuple[str, str, list[tuple[list[float], list[float]]]]] = {
    "train": (
        "synthetic",
        "train.json",
        [
            ([0.72, -0.21, 0.51, 0.42], [0.13, -0.08, 6.4]),
            ([0.12, 0.83, -0.33, 0.43], [-0.21, 0.05, 8.1]),
        ],
    ),
    "val": ("synthetic", "validation.json", [([0.5, 0.5, 0.5, 0.5], [0.02, 0.11, 7.2])]),
    "lightbox": (
        "lightbox",
        "test.json",
        [
            ([0.91, 0.1, -0.3, 0.27], [0.3, -0.2, 9.5]),
            ([-0.2, 0.6, 0.7, -0.33], [-0.05, 0.0, 5.6]),
        ],
    ),
    # The second translation pushes part of the body past the right image edge.
    "sunlamp": ("sunlamp", "test.json", [([0.31, -0.62, 0.1, 0.71], [1.85, 0.12, 6.0])]),
}


def _normalized(quaternion: list[float]) -> list[float]:
    values = np.asarray(quaternion, dtype=np.float64)
    return (values / np.linalg.norm(values)).tolist()


def _image(seed: int) -> np.ndarray:
    x = np.linspace(0, 255, WIDTH)[None, :]
    y = np.linspace(0, 255, HEIGHT)[:, None]
    planes = [np.broadcast_to(x, (HEIGHT, WIDTH)), np.broadcast_to(y, (HEIGHT, WIDTH))]
    planes.append(np.full((HEIGHT, WIDTH), 30.0 + 10 * seed))
    return np.stack(planes, axis=2).astype(np.uint8)


def write_speedplus_tree(speedplus_dir: Path) -> dict[str, int]:
    """Write the native tree; return the image count per split."""
    camera = {"Nu": WIDTH, "Nv": HEIGHT, "cameraMatrix": CAMERA_MATRIX, "distCoeffs": DIST}
    speedplus_dir.mkdir(parents=True)
    (speedplus_dir / "camera.json").write_text(json.dumps(camera))
    (speedplus_dir / "LICENSE.md").write_text("Creative Commons Attribution 4.0 (fixture)\n")
    counts, seed = {}, 0
    for split, (domain, label, poses) in POSES.items():
        images = speedplus_dir / domain / "images"
        images.mkdir(parents=True, exist_ok=True)
        records = []
        for quaternion, translation in poses:
            seed += 1
            name = f"img{seed:06d}.jpg"
            assert cv2.imwrite(str(images / name), _image(seed))
            records.append(
                {
                    "filename": name,
                    "q_vbs2tango_true": _normalized(quaternion),
                    "r_Vo2To_vbs_true": translation,
                }
            )
        (speedplus_dir / domain / label).write_text(json.dumps(records))
        counts[split] = len(records)
    return counts


def write_tango_points(path: Path) -> Path:
    """Write a fake ``tangoPoints.mat`` holding the official 3x11 ``tango3Dpoints``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(path, {"tango3Dpoints": OFFICIAL_POINTS.T})
    return path


def zip_tree(speedplus_dir: Path, archive: Path) -> Path:
    """Zip the tree under a top ``speedplusv2/`` directory, as the Zenodo archive is."""
    with zipfile.ZipFile(archive, "w") as bundle:
        for path in sorted(speedplus_dir.rglob("*")):
            if path.is_file():
                bundle.write(path, f"speedplusv2/{path.relative_to(speedplus_dir).as_posix()}")
    return archive


def _digest(path: Path, name: str) -> str:
    return hashlib.new(name, path.read_bytes()).hexdigest()


def fixture_profile(archive: Path, mat: Path, counts: dict[str, int], out: Path) -> Path:
    """Write a profile that pins the fixture archive, points file and lowered image counts."""
    from doper.speedplus import data

    profile: dict[str, Any] = json.loads(data.profile_path("speedplus").read_text())
    profile["archive"].update(
        url="https://example.invalid/speedplusv2.zip",
        bytes=archive.stat().st_size,
        md5=_digest(archive, "md5"),
    )
    profile["tango_points"].update(
        url="https://example.invalid/tangoPoints.mat", sha256=_digest(mat, "sha256")
    )
    profile["image_counts"] = counts
    profile["image_size"] = [WIDTH, HEIGHT]
    out.write_text(json.dumps(profile))
    return out


def copying_downloader(sources: dict[str, Path], calls: list[str] | None = None):  # noqa: ANN201
    """A stand-in for the HTTP downloader that copies a local file per URL."""

    def download(url: str, destination: Path) -> None:
        if calls is not None:
            calls.append(url)
        destination.write_bytes(sources[url].read_bytes())

    return download


def staged_root(tmp_path: Path) -> tuple[Path, Path]:
    """Fetch the fixture into ``tmp_path/root`` with the fixture profile.

    Returns:
        The dataset root and the fixture profile file.
    """
    from doper.speedplus import data

    work = tmp_path / "upstream"
    counts = write_speedplus_tree(work / "speedplusv2")
    archive = zip_tree(work / "speedplusv2", work / "speedplusv2.zip")
    mat = write_tango_points(work / "tangoPoints.mat")
    profile = fixture_profile(archive, mat, counts, work / "profile.json")
    loaded = data.load_profile(profile)
    downloader = copying_downloader(
        {loaded.archive["url"]: archive, loaded.tango_points["url"]: mat}
    )
    root = tmp_path / "root"
    data.fetch(root, consent=True, downloader=downloader, profile=loaded)
    return root, profile


def _mask_digest(path: Path) -> str:
    pixels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return hashlib.sha256(f"{pixels.shape}{pixels.dtype}".encode() + pixels.tobytes()).hexdigest()


def canonical_conversion(out: Path, root: Path) -> dict[str, Any]:
    """The comparable content of a 64-point conversion, with paths made ``root``-relative.

    The free-text ``info.description`` is left out. Masks are compared as decoded pixels (PNG
    encoder bytes may differ across OpenCV builds), the manifest by its mask mode, layout hash
    and per-split results, and the camera file by its matrix and distortion (the release adds
    the image size).
    """
    manifest = json.loads((out / "conversion_manifest.json").read_text())
    camera = json.loads((out / "camera_intrinsics.json").read_text())
    splits = {}
    for result in manifest["splits"]:
        split = result["split"]
        coco = json.loads((out / split / "annotations.json").read_text())
        source_split = Path(coco["info"]["source_split"])
        if source_split.is_absolute():
            coco["info"]["source_split"] = source_split.relative_to(root).as_posix()
        coco["info"].pop("description")
        masks = sorted((out / split / "masks").glob("*.png"))
        splits[split] = {
            "annotations": coco,
            "masks": {path.name: _mask_digest(path) for path in masks},
            "result": {key: value for key, value in result.items() if key != "annotations"},
        }
    return {
        "keypoints_3d": json.loads((out / "keypoints_3d.json").read_text()),
        "camera": {key: camera[key] for key in ("camera_matrix", "distortion_coefficients")},
        "manifest": {key: manifest[key] for key in ("mask_mode", "keypoint_layout_sha256")},
        "splits": splits,
    }
