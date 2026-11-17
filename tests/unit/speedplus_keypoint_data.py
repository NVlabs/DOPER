# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SPEED+ 75-point layout fixtures: a tracked-64 file, a landmark scene and canonical forms.

``write_tracked_keypoints`` writes 64 fixed tracked landmarks in the extractor's format;
``golden/source_speedplus_layout.json`` holds what the research layout builder
(``prepare_speedplus_custom_keypoints.py --splits train val``) writes from it on the
``speedplus_raw_data`` conversion, in the canonical form of ``canonical_layout``.

The landmark scene: a known cloud of textured landmarks on a box is rendered into 12
synthetic-train views (one 8 degree step apart about the object's y axis, so neighboring views
fall in the extractor's 6-18 degree pair window) and 8 synthetic-validation views. Every
landmark is a fronto-parallel patch of random texture, painted far to near, on a black
background. The train masks are the dilated convex hull of the projected landmarks; the
validation masks are the bare hull, so landmarks triangulated just outside the body fail the
extractor's containment check. The tree has the extracted SPEED+ layout (``camera.json``,
``synthetic/{images,train.json,validation.json}``) and the masks the 64-point conversion's
layout (``<tango_64>/{train,val}/masks/<stem>.png``).

``golden/source_speedplus_tracked_landmarks.json`` holds what the research extractor
(``extract_speedplus_trackable_keypoints.py`` with its defaults) writes for this scene, in the
canonical form of ``canonical_landmarks``, with the OpenCV and scikit-learn versions it was
produced with (SIFT, JPEG and DBSCAN make it build-specific). Both goldens were produced with
this module (``python tests/unit/speedplus_keypoint_data.py <research-source-dir>``), so its
values must not change. The module needs only numpy, OpenCV and SciPy; the
``speedplus_tracked_landmarks`` and ``speedplus_layout`` comparisons load it by path.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

WIDTH, HEIGHT = 640, 400
CAMERA_MATRIX = [[800.0, 0.0, 320.0], [0.0, 800.0, 200.0], [0.0, 0.0, 1.0]]
DIST = [-0.2238, 0.5141, -0.0007, -0.0002, -0.1312]
SEED = 7
LANDMARKS = 160
MIN_SPACING_M = 0.08
HALF_EXTENT_M = (0.5, 0.5, 0.3)
PATCH_M = 0.045
TEXTURE_CELLS = 6
TRAIN_VIEWS = 12
TRAIN_STEP_DEG = 8.0
VALIDATION_VIEWS = 8
RANGE_M = 3.0
MASK_DILATE_PX = 15


def landmark_cloud(seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Landmarks on the faces of a box, at least ``MIN_SPACING_M`` apart, and their textures."""
    rng = np.random.default_rng(seed)
    half = np.asarray(HALF_EXTENT_M)
    points: list[np.ndarray] = []
    while len(points) < LANDMARKS:
        point = rng.uniform(-half, half)
        axis = rng.integers(3)
        point[axis] = half[axis] * rng.choice([-1.0, 1.0])
        if all(np.linalg.norm(point - other) >= MIN_SPACING_M for other in points):
            points.append(point)
    textures = rng.integers(20, 256, size=(LANDMARKS, TEXTURE_CELLS, TEXTURE_CELLS))
    return np.asarray(points), textures.astype(np.uint8)


def _rotation(axis: str, degrees: float) -> np.ndarray:
    angle = np.radians(degrees)
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.asarray([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _poses() -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Object-to-camera rotations and translations of the train and validation views."""
    base = _rotation("x", 25.0) @ _rotation("z", 10.0)
    train = [
        (base @ _rotation("y", -44.0 + TRAIN_STEP_DEG * index), np.asarray([0.0, 0.0, RANGE_M]))
        for index in range(TRAIN_VIEWS)
    ]
    validation = [
        (
            _rotation("x", -30.0 + 9.0 * index) @ _rotation("y", 40.0 * index),
            np.asarray([0.05 * (index % 3 - 1), 0.04 * (index % 2), RANGE_M + 0.1 * index]),
        )
        for index in range(VALIDATION_VIEWS)
    ]
    return {"train": train, "val": validation}


def _speedplus_record(name: str, rotation: np.ndarray, translation: np.ndarray) -> dict:
    """The SPEED+ label of an object-to-camera pose (scalar-first, scalar part non-negative).

    The SPEED+ DCM is the transpose of SciPy's active rotation matrix for the same quaternion,
    so the object-to-camera matrix converts directly (as ``opencv_pose_to_speedplus``).
    """
    matrix = cv2.Rodrigues(cv2.Rodrigues(rotation)[0])[0]
    q = Rotation.from_matrix(matrix).as_quat()[[3, 0, 1, 2]]
    if q[0] < 0:
        q = -q
    return {
        "filename": name,
        "q_vbs2tango_true": q.tolist(),
        "r_Vo2To_vbs_true": translation.tolist(),
    }


def _render(
    points: np.ndarray, textures: np.ndarray, rotation: np.ndarray, translation: np.ndarray,
    dilate_px: int,
) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Paint the landmark patches far to near; return the image and the hull mask."""
    camera, dist = np.asarray(CAMERA_MATRIX), np.asarray(DIST)
    rvec = cv2.Rodrigues(rotation)[0]
    pixels = cv2.projectPoints(points, rvec, translation, camera, dist)[0].reshape(-1, 2)
    depth = (rotation @ points.T).T[:, 2] + translation[2]
    image = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for index in np.argsort(-depth):
        size = int(round(PATCH_M * camera[0, 0] / depth[index]))
        patch = cv2.resize(textures[index], (size, size), interpolation=cv2.INTER_LINEAR)
        x0 = int(round(pixels[index, 0])) - size // 2
        y0 = int(round(pixels[index, 1])) - size // 2
        x1, y1 = max(x0, 0), max(y0, 0)
        x2, y2 = min(x0 + size, WIDTH), min(y0 + size, HEIGHT)
        if x2 > x1 and y2 > y1:
            image[y1:y2, x1:x2] = patch[y1 - y0 : y2 - y0, x1 - x0 : x2 - x0]
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(np.round(pixels).astype(np.int32)), 255)
    if dilate_px == 0:
        return image, mask
    kernel = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
    return image, cv2.dilate(mask, kernel)


def write_landmark_scene(speedplus_dir: Path, tango_64: Path) -> np.ndarray:
    """Write the scene's SPEED+ tree and its train and validation masks.

    Args:
        speedplus_dir: The SPEED+ directory to create (``camera.json``, ``synthetic``).
        tango_64: The conversion directory that receives ``{train,val}/masks``.

    Returns:
        The landmark cloud (object frame, meters).
    """
    points, textures = landmark_cloud()
    images = speedplus_dir / "synthetic" / "images"
    images.mkdir(parents=True)
    camera = {"Nu": WIDTH, "Nv": HEIGHT, "cameraMatrix": CAMERA_MATRIX, "distCoeffs": DIST}
    (speedplus_dir / "camera.json").write_text(json.dumps(camera))
    labels = {"train": "train.json", "val": "validation.json"}
    dilate = {"train": MASK_DILATE_PX, "val": 0}
    counter = 0
    for split, poses in _poses().items():
        masks = tango_64 / split / "masks"
        masks.mkdir(parents=True)
        records = []
        for rotation, translation in poses:
            counter += 1
            name = f"img{counter:06d}.jpg"
            image, mask = _render(points, textures, rotation, translation, dilate[split])
            assert cv2.imwrite(str(images / name), image)
            assert cv2.imwrite(str(masks / f"img{counter:06d}.png"), mask)
            records.append(_speedplus_record(name, rotation, translation))
        (speedplus_dir / "synthetic" / labels[split]).write_text(json.dumps(records))
    return points


TRACKED_SEED = 11
TRACKED_COUNT = 64
# The research extractor's report keys that hold machine paths.
REPORT_PATH_KEYS = ("speedplus_root", "mask_root", "validation_mask_root", "output_dir")


def tracked_points() -> np.ndarray:
    """64 fixed tracked landmarks (meters)."""
    return np.random.default_rng(TRACKED_SEED).uniform(-0.6, 0.6, size=(TRACKED_COUNT, 3))


def write_tracked_keypoints(path: Path, points: np.ndarray | None = None) -> Path:
    """Write tracked landmarks as the extractor writes them (``keypoints_3d`` plus metadata)."""
    points = tracked_points() if points is None else points
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "object_name": "tango",
        "keypoint_layout": "mesh_free_sift_multiview_tracks_v1",
        "keypoint_layout_sha256": hashlib.sha256(points.astype("<f8").tobytes()).hexdigest(),
        "official_keypoint_count": 0,
        "keypoints_3d": points.tolist(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _root_relative(value: str, base: Path, root: Path) -> str:
    """``value`` (absolute and resolved, or relative to ``base``) relative to ``root``."""
    path = Path(value)
    absolute = path if path.is_absolute() else Path(os.path.normpath(base.resolve() / path))
    return absolute.relative_to(root.resolve()).as_posix()


def canonical_layout(out: Path, root: Path) -> dict[str, Any]:
    """The comparable content of a 75-point layout, with paths made ``root``-relative.

    ``keypoints_3d.json`` is kept as text (its bytes are the layout's identity). The free-text
    ``info.description`` is left out; the manifest is compared by the source's keys, and every
    link by the directory it resolves to.
    """
    manifest = json.loads((out / "conversion_manifest.json").read_text())
    splits = {}
    for result in manifest["splits"]:
        split = result["split"]
        coco = json.loads((out / split / "annotations.json").read_text())
        info = coco["info"]
        info["source_annotations"] = _root_relative(info["source_annotations"], out, root)
        info.pop("description")
        links = {
            name: (out / split / name).resolve().relative_to(root.resolve()).as_posix()
            for name in ("images", "masks")
            if (out / split / name).exists()
        }
        result = {key: value for key, value in result.items() if key != "annotation_path"}
        splits[split] = {"annotations": coco, "links": links, "result": result}
    keys = ("keypoint_layout_sha256", "num_keypoints", "official_keypoints", "additional_keypoints")
    return {
        "keypoints_3d": (out / "keypoints_3d.json").read_text(),
        "camera": json.loads((out / "camera_intrinsics.json").read_text()),
        "manifest": {key: manifest[key] for key in keys},
        "splits": splits,
    }


def _pixels_digest(path: Path) -> str:
    pixels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return hashlib.sha256(f"{pixels.shape}{pixels.dtype}".encode() + pixels.tobytes()).hexdigest()


def canonical_landmarks(out: Path) -> dict[str, Any]:
    """The comparable content of an extractor output directory.

    The report's path entries are left out; the landmark candidates are a digest of their
    sorted-key JSON, the overlays digests of their decoded pixels.
    """
    report = json.loads((out / "report.json").read_text())
    for key in REPORT_PATH_KEYS:
        report["configuration"].pop(key)
    candidates = json.loads((out / "landmark_candidates.json").read_text())
    overlays = sorted((out / "projection_overlays").glob("*.jpg"))
    return {
        "keypoints_3d": json.loads((out / "keypoints_3d.json").read_text()),
        "report": report,
        "pair_diagnostics": json.loads((out / "pair_diagnostics.json").read_text()),
        "landmark_candidates": hashlib.sha256(
            json.dumps(candidates, sort_keys=True).encode()
        ).hexdigest(),
        "overlays": {path.name: _pixels_digest(path) for path in overlays},
        "contact_sheet": _pixels_digest(out / "projection_contact_sheet.jpg"),
    }


def library_versions() -> dict[str, str]:
    """The builds the extractor golden depends on (SIFT, JPEG and DBSCAN outputs)."""
    import sklearn

    return {"opencv": cv2.__version__, "scikit-learn": sklearn.__version__}


REGENERATE = (
    "regenerate both goldens with `python tests/unit/speedplus_keypoint_data.py "
    "<research-source-dir>` (an extracted copy of the "
    "research scripts)"
)


def write_source_goldens(rec_snapshot: Path, golden: Path) -> None:
    """Run the research layout builder and extractor (research source snapshot) on these fixtures.

    Writes ``source_speedplus_layout.json`` and ``source_speedplus_tracked_landmarks.json``
    (with the OpenCV and scikit-learn versions it was produced with) into ``golden``.
    """
    import contextlib
    import io
    import sys
    import tempfile
    from unittest import mock

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(rec_snapshot))
    import scripts.extract_speedplus_trackable_keypoints as extract_source
    import scripts.prepare_speedplus_custom_keypoints as layout_source
    import speedplus_raw_data as raw

    from doper.speedplus import data

    def run(main: Any, argv: list[str]) -> None:  # noqa: ANN401
        patched = mock.patch.object(sys, "argv", ["source", *argv])
        with patched, contextlib.redirect_stdout(io.StringIO()):
            main()

    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary).resolve()
        root, _ = raw.staged_root(work)
        data.convert(root, root / data.DERIVED_64)
        tracked = write_tracked_keypoints(work / "tracked" / "keypoints_3d.json")
        out = root / "derived" / "tango_75"
        run(layout_source.main, [
            "--source-data-root", str(root / data.DERIVED_64), "--output-root", str(out),
            "--tango-points-mat", str(root / data.TANGO_POINTS_PATH),
            "--additional-keypoints", str(tracked), "--splits", "train", "val",
        ])  # fmt: skip
        layout = canonical_layout(out, root)
        (golden / "source_speedplus_layout.json").write_text(
            json.dumps(layout, indent=1, sort_keys=True) + "\n"
        )
        scene = work / "scene"
        write_landmark_scene(scene / "speedplusv2", scene / "tango_64")
        out = scene / "landmarks"
        run(extract_source.main, [
            "--speedplus-root", str(scene / "speedplusv2"),
            "--mask-root", str(scene / "tango_64" / "train" / "masks"), "--output-dir", str(out),
        ])  # fmt: skip
        landmarks = {**canonical_landmarks(out), "versions": library_versions()}
        (golden / "source_speedplus_tracked_landmarks.json").write_text(
            json.dumps(landmarks, indent=1, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    import sys

    write_source_goldens(Path(sys.argv[1]), Path(__file__).resolve().parent / "golden")
