# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The 75-point SPEED+ keypoint layout and the tracked-landmark extractor.

The paper's layout stacks the 11 official Tango landmarks with 64 "tracked" landmarks that
were triangulated, without a mesh, from SPEED+ synthetic images.

:func:`build_layout` reprojects that layout into a copy of the 64-point conversion
(``doper.speedplus.data.convert``) and writes ``<root>/derived/tango_75``::

    keypoints_3d.json, camera_intrinsics.json, conversion_manifest.json
    <split>/{annotations.json, images -> the SPEED+ images, masks -> the 64-point hull masks}

Boxes and masks stay those of the 64-point hull (the "hull64" of the paper's run names). By
default only synthetic train and validation are built, as the paper launcher did, which keeps
real-domain ground-truth keypoints out of the training root. Keypoint names are
``official_XX``, then ``tracked_XX``.

A layout has two SHA-256 values. Its **points hash** (:func:`layout_points_sha256`, the source's
``keypoint_layout_sha256``) is the SHA-256 of the 75x3 little-endian float64 array: the layout's
identity, embedded in ``keypoints_3d.json``, the COCO ``info`` and the manifest. Its **file
SHA-256** (:func:`layout_sha256`) is that of ``keypoints_3d.json``, written in a fixed key order and
format; SPEED+ checkpoints record it as ``speedplus_layout_sha256``
(``doper.speedplus.checkpoints``). A build is the paper layout when its points hash equals the
recipe's pin; any other needs ``accept_new_layout`` and is marked ``paper_layout: false``. A paper
layout whose file SHA-256 differs from the pinned one is recorded as a formatting deviation, not
refused.

:func:`extract_tracked_landmarks` extracts the landmarks: masked SIFT on SPEED+ synthetic train
images, mutual ratio matching on pose-neighbor pairs, triangulation at the known poses, union-find
feature tracks, a DBSCAN merge, a mask-containment check on validation views and a quality-weighted
spatial selection of 64 landmarks. It needs the ``doper[speedplus]`` extra (scikit-learn); its 3D
scatter plot is written only when matplotlib is importable.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..bop.sources import file_lock, link_relative, sha256_file, sibling_lock
from . import geometry
from .data import ANNOTATIONS_NAME, CAMERA_NAME, KEYPOINTS_NAME, MANIFEST_NAME, keypoint_payload
from .data import layout_points_sha256 as layout_points_sha256  # re-exported: the points hash
from .definitions import DERIVED_75 as DERIVED_75  # re-exported: the layout's directory
from .definitions import LAYOUT_SPLITS, PAPER_LAYOUT, SPLIT_NAMES, TRACKED_SEED, LayoutPins
from .definitions import TRACKED_DIR as TRACKED_DIR  # re-exported: the extractor's directory

LAYOUT_75_NAME = "speedplus_official_11_plus_tracked_64"
TRACKED_LAYOUT_NAME = "mesh_free_sift_multiview_tracks_v1"
TRACKED_COUNT = 64
COCO_DESCRIPTION = "SPEED+ DOPER custom 11+64 tracked-keypoint layout"
SPEEDPLUS_EXTRA = "doper[speedplus]"


# --- The 75-point layout ------------------------------------------------------------------


@dataclass(frozen=True)
class LayoutManifest:
    """A 75-point layout build (``conversion_manifest.json``).

    Attributes:
        out: Output directory.
        layout_sha256: SHA-256 of the written ``keypoints_3d.json`` (what checkpoints record).
        points_sha256: Points hash of the 75 points (the layout's identity).
        paper_layout: Whether the points hash is the pinned paper one.
        layout_file_matches_paper: Whether the file SHA-256 is also the pinned one.
        splits_requested: The splits built.
        splits: Per-split results.
        keypoints_3d: The written ``keypoints_3d.json``.
        manifest: The manifest file.
    """

    out: Path
    layout_sha256: str
    points_sha256: str
    paper_layout: bool
    layout_file_matches_paper: bool
    splits_requested: tuple[str, ...]
    splits: tuple[Mapping[str, Any], ...]
    keypoints_3d: Path
    manifest: Path


def layout_sha256(path: str | Path) -> str:
    """SHA-256 of a ``keypoints_3d.json`` layout file: what SPEED+ checkpoints record."""
    return sha256_file(path)


def load_tracked_keypoints(path: str | Path) -> np.ndarray:
    """Load ``N x 3`` keypoints from a JSON list or a ``{"keypoints_3d": ...}`` payload.

    Raises:
        ValueError: If the points are not finite ``N x 3``.
    """
    payload = json.loads(Path(path).read_text())
    points = payload.get("keypoints_3d", payload) if isinstance(payload, dict) else payload
    result = np.asarray(points, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f"expected Nx3 keypoints in {path}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"non-finite keypoints in {path}")
    return result


def _official_points(tango_64: Path) -> np.ndarray:
    """The official landmarks, the first rows of the 64-point layout."""
    payload = json.loads((tango_64 / KEYPOINTS_NAME).read_text())
    count = int(payload["official_keypoint_count"])
    if count != geometry.OFFICIAL_POINT_COUNT:
        raise ValueError(f"{tango_64 / KEYPOINTS_NAME} has {count} official landmarks, not 11")
    return np.asarray(payload["keypoints_3d"], dtype=np.float64)[:count]


def _layout_points(official: np.ndarray, tracked: np.ndarray) -> np.ndarray:
    """Stack the official and tracked landmarks.

    Raises:
        ValueError: Unless there are 64 tracked landmarks and no two points coincide.
    """
    if len(tracked) != TRACKED_COUNT:
        raise ValueError(f"expected {TRACKED_COUNT} additional keypoints, got {len(tracked)}")
    points = np.vstack([official, tracked])
    rounded = np.round(points, decimals=geometry.DUPLICATE_DECIMALS)
    if np.unique(rounded, axis=0).shape[0] != len(points):
        raise ValueError("combined keypoint layout contains exact duplicates")
    return points


def _layout_text(points: np.ndarray, points_hash: str, official_count: int) -> str:
    """``keypoints_3d.json`` in its fixed key order and format."""
    payload = {
        "object_name": "tango",
        "keypoint_layout": LAYOUT_75_NAME,
        "keypoint_layout_sha256": points_hash,
        "official_keypoint_count": official_count,
        "tracked_keypoint_count": len(points) - official_count,
        "keypoints_3d": points.tolist(),
    }
    return json.dumps(payload, indent=2) + "\n"


@dataclass(frozen=True)
class _LayoutInputs:
    """What every split of one layout build shares."""

    tango_64: Path
    partial: Path
    out: Path
    points: np.ndarray
    points_hash: str
    official_count: int
    camera: geometry.Camera


def _keypoint_names(official_count: int, total: int) -> list[str]:
    official = [f"official_{index:02d}" for index in range(official_count)]
    return official + [f"tracked_{index:02d}" for index in range(total - official_count)]


def _reproject(annotation: Mapping[str, Any], image: Mapping[str, Any], inputs: _LayoutInputs,
               label: Path) -> tuple[dict[str, Any], int]:  # fmt: skip
    """One annotation with the layout's keypoints; also its visible count."""
    quaternion = annotation.get("speedplus_q_vbs2tango_true")
    translation = annotation.get("speedplus_r_Vo2To_vbs_true")
    if quaternion is None or translation is None:
        raise ValueError(f"{label} lacks SPEED+ pose metadata")
    pixels, depth = geometry.project_keypoints(inputs.points, quaternion, translation,
                                               inputs.camera)  # fmt: skip
    keypoints, visible = keypoint_payload(pixels, depth, int(image["width"]), int(image["height"]))
    return {**annotation, "keypoints": keypoints, "num_keypoints": visible}, visible


def _convert_split(inputs: _LayoutInputs, split: str) -> dict[str, Any]:
    """Reproject one split of the 64-point conversion; return its manifest entry."""
    source_split = inputs.tango_64 / split
    label = source_split / ANNOTATIONS_NAME
    if not label.is_file():
        raise FileNotFoundError(
            f"{label} is missing; convert the {split} split first (`--stage convert`)"
        )
    source = json.loads(label.read_text())
    images_by_id = {int(image["id"]): image for image in source["images"]}
    converted, min_visible = [], len(inputs.points)
    for annotation in source["annotations"]:
        image = images_by_id.get(int(annotation["image_id"]))
        if image is None:
            continue
        record, visible = _reproject(annotation, image, inputs, label)
        converted.append(record)
        min_visible = min(min_visible, visible)
    names = _keypoint_names(inputs.official_count, len(inputs.points))
    payload = {
        **source,
        "info": {
            **source.get("info", {}),
            "description": COCO_DESCRIPTION,
            "keypoint_layout_sha256": inputs.points_hash,
            "source_annotations": Path(os.path.relpath(label, inputs.out)).as_posix(),
        },
        "images": source["images"],
        "annotations": converted,
        "categories": [{**source["categories"][0], "keypoints": names, "skeleton": []}],
    }
    output_split = inputs.partial / split
    output_split.mkdir(parents=True)
    link_relative(output_split / "images", source_split / "images")
    if (source_split / "masks").exists():
        link_relative(output_split / "masks", source_split / "masks")
    text = json.dumps(payload, separators=(",", ":")) + "\n"
    (output_split / ANNOTATIONS_NAME).write_text(text)
    return {
        "split": split,
        "images": len(source["images"]),
        "annotations": len(converted),
        "min_visible_keypoints": int(min_visible),
        "annotation_path": f"{split}/{ANNOTATIONS_NAME}",
    }


def _check_splits(splits: Sequence[str]) -> None:
    unknown = sorted(set(splits) - set(SPLIT_NAMES))
    if unknown or not splits or len(set(splits)) != len(splits):
        raise ValueError(f"splits must be distinct names of {list(SPLIT_NAMES)}, got {splits}")


def _paper_decision(points_hash: str, pins: LayoutPins, tracked: Path, accept: bool) -> bool:
    """Whether the layout is the paper's; refuse another one unless it is accepted.

    Raises:
        ValueError: For a non-paper layout without ``accept``.
    """
    if points_hash == pins.points_sha256:
        return True
    if not accept:
        raise ValueError(
            f"the tracked keypoints {tracked} give layout points hash {points_hash}, not the "
            f"paper's {pins.points_sha256}; pass --accept-new-layout to build a non-paper "
            "layout (it is marked paper_layout: false)"
        )
    return False


def _layout_manifest(payload: Mapping[str, Any], out: Path) -> LayoutManifest:
    return LayoutManifest(
        out=out,
        layout_sha256=payload["layout_sha256"],
        points_sha256=payload["keypoint_layout_sha256"],
        paper_layout=bool(payload["paper_layout"]),
        layout_file_matches_paper=bool(payload["layout_file_matches_paper"]),
        splits_requested=tuple(payload["splits_requested"]),
        splits=tuple(payload["splits"]),
        keypoints_3d=out / KEYPOINTS_NAME,
        manifest=out / MANIFEST_NAME,
    )


def _existing_layout(out: Path, request: Mapping[str, Any], force: bool) -> LayoutManifest | None:
    """The finished build of the same request, its layout file re-checked; None to (re)build.

    Raises:
        FileExistsError: If ``out`` holds another build and ``force`` is false.
        ValueError: If the layout file no longer has its recorded SHA-256.
    """
    if force or not out.exists():
        return None
    manifest = out / MANIFEST_NAME
    payload = json.loads(manifest.read_text()) if manifest.is_file() else {}
    if {key: payload.get(key) for key in request} != request:
        raise FileExistsError(f"{out} holds another layout build; pass --force to rebuild it")
    actual = layout_sha256(out / KEYPOINTS_NAME)
    if actual != payload["layout_sha256"]:
        raise ValueError(
            f"{out / KEYPOINTS_NAME} has SHA-256 {actual}, but the build recorded "
            f"{payload['layout_sha256']}; pass --force to rebuild it"
        )
    return _layout_manifest(payload, out)


def _identity(file_sha: str, paper: bool, pins: LayoutPins) -> dict[str, Any]:
    """The manifest's layout identity entries, with a formatting deviation when one occurs."""
    matches = paper and file_sha == pins.sha256
    identity: dict[str, Any] = {
        "layout_sha256": file_sha,
        "paper_layout": paper,
        "paper_points_sha256": pins.points_sha256,
        "paper_layout_sha256": pins.sha256,
        "layout_file_matches_paper": matches,
    }
    if paper and not matches:
        identity["formatting_deviation"] = (
            f"keypoints_3d.json has SHA-256 {file_sha}, not the paper file's {pins.sha256}, "
            "although its points are the paper's"
        )
    return identity


def build_layout(
    tango_64_root: Path,
    out: Path,
    tracked: Path,
    *,
    splits: Sequence[str] = LAYOUT_SPLITS,
    accept_new_layout: bool = False,
    pins: LayoutPins = PAPER_LAYOUT,
    force: bool = False,
) -> LayoutManifest:
    """Build the 75-point layout from the 64-point conversion and 64 tracked landmarks.

    A finished build of the same splits, points and pins is returned as it is, after its layout
    file is checked against its recorded SHA-256; another one needs ``force``. The output is built
    beside ``out`` and renamed into place.

    Args:
        tango_64_root: The 64-point conversion (``<root>/derived/tango_64``).
        out: Output directory, usually ``<root>/derived/tango_75``.
        tracked: The tracked-64 ``keypoints_3d.json`` (``extract_tracked_landmarks`` output,
            or the paper's file).
        splits: Splits to build (default synthetic train and validation, as the paper).
        accept_new_layout: Build a layout whose points hash is not the pinned paper one; it
            is marked ``paper_layout: false``.
        pins: The paper layout's pins (the recipe's ``layout``).
        force: Rebuild an existing output.

    Returns:
        The manifest.

    Raises:
        FileNotFoundError: If a split was not converted.
        FileExistsError: If ``out`` holds another build and ``force`` is false.
        ValueError: For an unknown split, malformed tracked landmarks, a non-paper layout
            without ``accept_new_layout`` or an edited layout file.
    """
    _check_splits(splits)
    tango_64_root, out, tracked = Path(tango_64_root), Path(out), Path(tracked)
    official = _official_points(tango_64_root)
    points = _layout_points(official, load_tracked_keypoints(tracked))
    points_hash = layout_points_sha256(points)
    paper = _paper_decision(points_hash, pins, tracked, accept_new_layout)
    # The pins are part of the request: changed pins change paper_layout and the file check.
    request = {
        "splits_requested": list(splits),
        "keypoint_layout_sha256": points_hash,
        "paper_points_sha256": pins.points_sha256,
        "paper_layout_sha256": pins.sha256,
    }
    with file_lock(sibling_lock(out)):
        existing = _existing_layout(out, request, force)
        if existing is not None:
            return existing
        partial = out.with_name(f".{out.name}.partial")
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
        try:
            payload = _build_layout(tango_64_root, partial, out, points, request, paper, pins)
            payload["additional_keypoints_sha256"] = sha256_file(tracked)
            (partial / MANIFEST_NAME).write_text(json.dumps(payload, indent=2) + "\n")
        except BaseException:
            shutil.rmtree(partial)
            raise
        if out.exists():
            shutil.rmtree(out)
        partial.rename(out)
    return _layout_manifest(payload, out)


def _build_layout(
    tango_64: Path, partial: Path, out: Path, points: np.ndarray, request: Mapping[str, Any],
    paper: bool, pins: LayoutPins,
) -> dict[str, Any]:  # fmt: skip
    """Write the layout, camera and splits into ``partial``; return the manifest payload."""
    points_hash = request["keypoint_layout_sha256"]
    official_count = geometry.OFFICIAL_POINT_COUNT
    (partial / KEYPOINTS_NAME).write_text(_layout_text(points, points_hash, official_count))
    camera_path = tango_64 / CAMERA_NAME
    (partial / CAMERA_NAME).write_text(camera_path.read_text())
    inputs = _LayoutInputs(tango_64, partial, out, points, points_hash, official_count,
                           geometry.load_camera(camera_path))  # fmt: skip
    results = [_convert_split(inputs, split) for split in request["splits_requested"]]
    source_manifest = json.loads((tango_64 / MANIFEST_NAME).read_text())
    return {
        "dataset": "speedplus",
        "source_data_root": Path(os.path.relpath(tango_64, out)).as_posix(),
        "source_paper_masks": source_manifest.get("paper_masks"),
        "keypoint_layout": LAYOUT_75_NAME,
        "keypoint_layout_sha256": points_hash,
        **_identity(layout_sha256(partial / KEYPOINTS_NAME), paper, pins),
        "num_keypoints": len(points),
        "official_keypoints": official_count,
        "additional_keypoints": len(points) - official_count,
        "splits_requested": request["splits_requested"],
        "splits": results,
    }


# --- The tracked-landmark extractor ---------------------------------------------------------


@dataclass(frozen=True)
class ExtractorSettings:
    """The extractor's arguments, with their defaults.

    Attributes:
        anchors: Anchor images drawn (with the seed) from synthetic train.
        neighbors_per_anchor: Pose neighbors matched per anchor.
        min_view_deg: Smallest rotation between paired views.
        max_view_deg: Largest rotation between paired views.
        target_view_deg: Preferred rotation between paired views.
        max_range_relative: Largest relative range difference between paired views.
        nfeatures: SIFT features per image.
        sift_contrast_threshold: SIFT contrast threshold.
        ratio_threshold: Lowe ratio of the mutual matching.
        mask_erode_px: Mask erosion before feature detection.
        max_reprojection_px: Largest reprojection error of a triangulated match.
        min_triangulation_angle_deg: Smallest ray angle of a triangulated match.
        min_track_images: Fewest images in a feature track.
        max_track_spread_m: Largest 90th-percentile spread of a track's points.
        merge_radius_m: DBSCAN radius merging track centers.
        validation_mask_root: Validation masks (default ``<mask_root>/../../val/masks``).
        validation_views: Validation views of the containment check.
        min_mask_containment: Smallest fraction of validation views whose mask contains a
            landmark.
        num_keypoints: Landmarks selected.
        overlay_images: Anchor images that receive projection overlays.
    """

    anchors: int = 80
    neighbors_per_anchor: int = 6
    min_view_deg: float = 6.0
    max_view_deg: float = 18.0
    target_view_deg: float = 10.0
    max_range_relative: float = 0.15
    nfeatures: int = 2500
    sift_contrast_threshold: float = 0.01
    ratio_threshold: float = 0.85
    mask_erode_px: int = 5
    max_reprojection_px: float = 2.5
    min_triangulation_angle_deg: float = 1.0
    min_track_images: int = 3
    max_track_spread_m: float = 0.02
    merge_radius_m: float = 0.015
    validation_mask_root: Path | None = None
    validation_views: int = 300
    min_mask_containment: float = 0.90
    num_keypoints: int = 64
    overlay_images: int = 12


@dataclass
class ImageFeatures:
    """SIFT keypoints (full-image pixels) and descriptors of one image."""

    points: np.ndarray
    descriptors: np.ndarray


Node = tuple[int, int]


class DisjointSet:
    """Union-find over ``(image index, feature index)`` nodes, by rank with path compression."""

    def __init__(self) -> None:
        """Start empty."""
        self.parent: dict[Node, Node] = {}
        self.rank: dict[Node, int] = {}

    def find(self, item: Node) -> Node:
        """The root of ``item``'s set, adding ``item`` as a singleton when it is new."""
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, first: Node, second: Node) -> None:
        """Join the sets of two nodes."""
        root_a = self.find(first)
        root_b = self.find(second)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def _require_dbscan() -> type:
    """scikit-learn's ``DBSCAN``.

    Raises:
        ImportError: Naming the ``doper[speedplus]`` extra when scikit-learn is missing.
    """
    try:
        from sklearn.cluster import DBSCAN
    except ImportError as error:
        raise ImportError(
            f"the tracked-landmark extractor needs scikit-learn: install {SPEEDPLUS_EXTRA}"
        ) from error
    return DBSCAN


def quaternion_angle_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Rotation angles (degrees) between one unit quaternion and each row of ``second``."""
    dots = np.abs(second @ first)
    return np.degrees(2.0 * np.arccos(np.clip(dots, 0.0, 1.0)))


def choose_pairs(
    quaternions: np.ndarray, translations: np.ndarray, settings: ExtractorSettings,
    rng: np.random.Generator,
) -> tuple[list[int], list[tuple[int, int, float]]]:  # fmt: skip
    """Draw anchors and pair each with its pose neighbors closest to the target rotation.

    Returns:
        The anchor indices and the distinct ``(first, second, angle_deg)`` pairs.
    """
    count = len(quaternions)
    anchor_indices = rng.choice(count, size=min(settings.anchors, count), replace=False)
    ranges = np.linalg.norm(translations, axis=1)
    pairs: list[tuple[int, int, float]] = []
    used: set[tuple[int, int]] = set()
    for anchor in anchor_indices:
        angles = quaternion_angle_deg(quaternions[anchor], quaternions)
        range_relative = np.abs(ranges / ranges[anchor] - 1.0)
        eligible = np.flatnonzero(
            (angles >= settings.min_view_deg)
            & (angles <= settings.max_view_deg)
            & (range_relative <= settings.max_range_relative)
        )
        if not len(eligible):
            continue
        costs = (
            np.abs(angles[eligible] - settings.target_view_deg) + 10.0 * range_relative[eligible]
        )
        for neighbor in eligible[np.argsort(costs)[: settings.neighbors_per_anchor]]:
            first, second = sorted((int(anchor), int(neighbor)))
            pair = (first, second)
            if pair in used:
                continue
            used.add(pair)
            pairs.append((pair[0], pair[1], float(angles[neighbor])))
    return anchor_indices.tolist(), pairs


def _no_features() -> ImageFeatures:
    return ImageFeatures(np.empty((0, 2)), np.empty((0, 128), np.float32))


def extract_features(
    image_path: Path, mask_path: Path, sift: cv2.SIFT, erode_px: int
) -> ImageFeatures:
    """SIFT features inside the eroded mask, detected on the mask's bounding crop.

    Raises:
        FileNotFoundError: If the image or the mask cannot be read.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)
    if mask is None:
        raise FileNotFoundError(mask_path)
    if erode_px > 0:
        kernel_size = 2 * erode_px + 1
        mask = cv2.erode(mask, np.ones((kernel_size, kernel_size), np.uint8))
    nonzero = cv2.findNonZero(mask)
    if nonzero is None:
        return _no_features()
    x, y, width, height = cv2.boundingRect(nonzero)
    keypoints, descriptors = sift.detectAndCompute(
        image[y : y + height, x : x + width], mask[y : y + height, x : x + width]
    )
    if not keypoints or descriptors is None:
        return _no_features()
    points = np.asarray(
        [[keypoint.pt[0] + x, keypoint.pt[1] + y] for keypoint in keypoints], dtype=np.float64
    )
    return ImageFeatures(points, descriptors.astype(np.float32))


def mutual_ratio_matches(
    first: ImageFeatures, second: ImageFeatures, ratio: float
) -> list[tuple[int, int, float]]:
    """Matches that pass the ratio test both ways and agree: ``(query, train, distance)``."""
    if len(first.points) < 2 or len(second.points) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(first.descriptors, second.descriptors, k=2)
    reverse = matcher.knnMatch(second.descriptors, first.descriptors, k=2)
    accepted_forward = {
        match.queryIdx: (match.trainIdx, float(match.distance))
        for match, runner_up in forward
        if match.distance < ratio * runner_up.distance
    }
    accepted_reverse = {
        match.queryIdx: match.trainIdx
        for match, runner_up in reverse
        if match.distance < ratio * runner_up.distance
    }
    return [
        (query, train, distance)
        for query, (train, distance) in accepted_forward.items()
        if accepted_reverse.get(train) == query
    ]


Pose = tuple[np.ndarray, np.ndarray]


def _reprojection_check(
    xyz: np.ndarray, points: list[np.ndarray], poses: tuple[Pose, Pose], camera: geometry.Camera,
    max_reprojection_px: float,
) -> tuple[np.ndarray, list[np.ndarray]]:  # fmt: skip
    """Finite points in front of both cameras within the reprojection limit; the errors."""
    valid = np.isfinite(xyz).all(axis=1)
    errors = []
    for view, (rvec, tvec) in enumerate(poses):
        projected, _ = cv2.projectPoints(xyz, rvec, tvec, camera.K, camera.dist)
        error = np.linalg.norm(projected.reshape(-1, 2) - points[view], axis=1)
        errors.append(error)
        rotation, _ = cv2.Rodrigues(rvec)
        depth = (rotation @ xyz.T + tvec).T[:, 2]
        valid &= depth > 0.0
        valid &= error <= max_reprojection_px
    return valid, errors  # pyright: ignore[reportReturnType]


def _ray_angles_deg(xyz: np.ndarray, poses: tuple[Pose, Pose]) -> np.ndarray:
    """The angle between the two cameras' rays to each point."""
    rays = []
    for rvec, tvec in poses:
        rotation, _ = cv2.Rodrigues(rvec)
        center = (-rotation.T @ tvec).reshape(3)
        ray = xyz - center
        ray /= np.clip(np.linalg.norm(ray, axis=1, keepdims=True), 1e-12, None)
        rays.append(ray)
    ray_dot = np.sum(rays[0] * rays[1], axis=1)
    return np.degrees(np.arccos(np.clip(ray_dot, -1.0, 1.0)))


def triangulate_matches(
    matches: list[tuple[int, int, float]], features: tuple[ImageFeatures, ImageFeatures],
    poses: tuple[Pose, Pose], camera: geometry.Camera, settings: ExtractorSettings,
) -> list[dict[str, Any]]:  # fmt: skip
    """Triangulate matches at the two known poses and keep the geometrically valid ones.

    Returns:
        Per kept match: ``query``, ``train``, ``xyz``, ``reprojection_px`` (the larger of the
        two views), ``triangulation_angle_deg`` and ``descriptor_distance``.
    """
    if not matches:
        return []
    points = [
        np.asarray([view_features.points[int(match[view])] for match in matches], dtype=np.float64)
        for view, view_features in enumerate(features)
    ]
    normalized = [
        cv2.undistortPoints(point_set.reshape(-1, 1, 2), camera.K, camera.dist).reshape(-1, 2)
        for point_set in points
    ]
    projections = [np.hstack([cv2.Rodrigues(rvec)[0], tvec]) for rvec, tvec in poses]
    homogeneous = cv2.triangulatePoints(
        projections[0], projections[1], normalized[0].T, normalized[1].T
    )
    xyz = (homogeneous[:3] / homogeneous[3]).T
    valid, errors = _reprojection_check(xyz, points, poses, camera, settings.max_reprojection_px)
    angles = _ray_angles_deg(xyz, poses)
    valid &= angles >= settings.min_triangulation_angle_deg
    results = []
    for match_index in np.flatnonzero(valid):
        query, train, descriptor_distance = matches[match_index]
        results.append(
            {
                "query": query,
                "train": train,
                "xyz": xyz[match_index],
                "reprojection_px": float(max(errors[0][match_index], errors[1][match_index])),
                "triangulation_angle_deg": float(angles[match_index]),
                "descriptor_distance": descriptor_distance,
            }
        )
    return results


def _track_candidate(track_edges: list[dict[str, Any]], image_count: int,
                     spread: float, center: np.ndarray) -> dict[str, Any]:  # fmt: skip
    """A track's landmark candidate and its quality."""
    reprojection = np.asarray([edge["reprojection_px"] for edge in track_edges])
    angles = np.asarray([edge["triangulation_angle_deg"] for edge in track_edges])
    descriptor_distances = np.asarray([edge["descriptor_distance"] for edge in track_edges])
    quality = (
        math.log1p(image_count)
        + 0.5 * math.log1p(len(track_edges))
        + 0.025 * float(np.median(angles))
        - 0.6 * float(np.median(reprojection))
        - 30.0 * spread
        - 0.001 * float(np.median(descriptor_distances))
    )
    return {
        "xyz": center,
        "quality": quality,
        "images": image_count,
        "edges": len(track_edges),
        "spread_m": spread,
        "median_reprojection_px": float(np.median(reprojection)),
        "median_triangulation_angle_deg": float(np.median(angles)),
        "median_descriptor_distance": float(np.median(descriptor_distances)),
    }


def build_track_candidates(
    disjoint_set: DisjointSet, edges: list[dict[str, Any]], min_images: int, max_spread_m: float
) -> list[dict[str, Any]]:
    """One candidate per feature track seen in ``min_images`` images and tight enough."""
    grouped: dict[Node, list[dict[str, Any]]] = {}
    for edge in edges:
        root = disjoint_set.find(edge["first_node"])
        grouped.setdefault(root, []).append(edge)
    candidates = []
    for track_edges in grouped.values():
        nodes = {node for edge in track_edges for node in (edge["first_node"], edge["second_node"])}
        images = {node[0] for node in nodes}
        if len(images) < min_images:
            continue
        points = np.asarray([edge["xyz"] for edge in track_edges])
        center = np.median(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        spread = float(np.percentile(distances, 90))
        if spread > max_spread_m:
            continue
        candidates.append(_track_candidate(track_edges, len(images), spread, center))
    return candidates


def merge_candidates(
    candidates: list[dict[str, Any]], radius_m: float, dbscan: Callable[..., Any]
) -> list[dict[str, Any]]:
    """Merge candidates within ``radius_m`` (DBSCAN) into quality-weighted landmarks."""
    if not candidates:
        return []
    points = np.asarray([candidate["xyz"] for candidate in candidates])
    labels = dbscan(eps=radius_m, min_samples=1).fit_predict(points)
    merged = []
    for label in np.unique(labels):
        members = [
            candidate for candidate, value in zip(candidates, labels, strict=True) if value == label
        ]
        weights = np.asarray([max(candidate["quality"], 0.1) for candidate in members])
        xyz = np.average(
            np.asarray([candidate["xyz"] for candidate in members]), axis=0, weights=weights
        )
        representative = max(members, key=lambda candidate: candidate["quality"])
        best = max(member["quality"] for member in members)
        merged.append(
            {
                **representative,
                "xyz": xyz,
                "merged_tracks": len(members),
                "total_track_images": int(sum(member["images"] for member in members)),
                "quality": float(best + 0.2 * math.log1p(len(members))),
            }
        )
    return merged


def add_mask_containment(
    candidates: list[dict[str, Any]], labels: list[dict[str, Any]], image_indices: np.ndarray,
    mask_root: Path, camera: geometry.Camera,
) -> None:  # fmt: skip
    """Score each candidate by the fraction of validation masks its projection falls in.

    Raises:
        FileNotFoundError: If a validation mask is missing.
    """
    if not candidates:
        return
    points = np.asarray([candidate["xyz"] for candidate in candidates])
    hits = np.zeros(len(points), dtype=np.int64)
    trials = np.zeros(len(points), dtype=np.int64)
    for image_index in image_indices:
        record = labels[int(image_index)]
        mask_path = mask_root / record["filename"].replace(".jpg", ".png")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)
        rvec, tvec = geometry.speedplus_pose_to_opencv(
            record["q_vbs2tango_true"], record["r_Vo2To_vbs_true"]
        )
        projected, _ = cv2.projectPoints(points, rvec, tvec, camera.K, camera.dist)
        pixels = np.rint(projected.reshape(-1, 2)).astype(np.int64)
        inside = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < mask.shape[1])
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < mask.shape[0])
        )
        trials += inside.astype(np.int64)
        valid_indices = np.flatnonzero(inside)
        hits[valid_indices] += (
            mask[pixels[valid_indices, 1], pixels[valid_indices, 0]] > 0
        ).astype(np.int64)
    containment = hits / np.maximum(trials, 1)
    for candidate, value, hit_count, trial_count in zip(
        candidates, containment, hits, trials, strict=True
    ):
        candidate["mask_containment"] = float(value)
        candidate["mask_containment_hits"] = int(hit_count)
        candidate["mask_containment_trials"] = int(trial_count)
        candidate["quality"] += 0.5 * float(value)


def select_spatially_distributed(candidates: list[dict[str, Any]], count: int) -> list[dict]:
    """Greedy farthest-point selection weighted by quality (0.6 distance, 0.4 quality)."""
    if len(candidates) <= count:
        return sorted(candidates, key=lambda candidate: candidate["quality"], reverse=True)
    points = np.asarray([candidate["xyz"] for candidate in candidates])
    qualities = np.asarray([candidate["quality"] for candidate in candidates])
    low, high = np.percentile(points, [2.0, 98.0], axis=0)
    scale = np.maximum(high - low, 0.05)
    normalized_points = (points - low) / scale
    quality_normalized = (qualities - qualities.min()) / max(np.ptp(qualities), 1e-12)
    selected = [int(np.argmax(qualities))]
    minimum_distance = np.linalg.norm(normalized_points - normalized_points[selected[0]], axis=1)
    while len(selected) < count:
        score = 0.6 * minimum_distance + 0.4 * quality_normalized
        score[selected] = -np.inf
        index = int(np.argmax(score))
        selected.append(index)
        minimum_distance = np.minimum(
            minimum_distance, np.linalg.norm(normalized_points - normalized_points[index], axis=1)
        )
    return [candidates[index] for index in selected]


@dataclass(frozen=True)
class _Scene:
    """The SPEED+ inputs of one extraction."""

    speedplus_dir: Path
    mask_root: Path
    labels: list[dict[str, Any]]
    camera: geometry.Camera


def _write_scatter(output_dir: Path, points: np.ndarray) -> bool:
    """Write the 3D scatter plot when matplotlib is importable; return whether it was."""
    try:
        from matplotlib.figure import Figure
    except ImportError:
        return False
    figure = Figure(figsize=(9, 7))
    axis = figure.add_subplot(111, projection="3d")
    scatter = axis.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],  # pyright: ignore[reportArgumentType] (a 3D axis)
        c=np.arange(len(points)),
        cmap="turbo",
    )
    axis.set_xlabel("object x [m]")
    axis.set_ylabel("object y [m]")
    axis.set_zlabel("object z [m]")  # type: ignore[attr-defined]
    axis.set_title("Mesh-free triangulated keypoint selection")
    figure.colorbar(scatter, ax=axis, label="keypoint index")
    figure.tight_layout()
    figure.savefig(output_dir / "keypoints_3d_scatter.png", dpi=180)
    return True


def _overlay(scene: _Scene, record: Mapping[str, Any], points: np.ndarray) -> np.ndarray:
    """One image with the projected landmarks: green on the mask, red off it."""
    filename = record["filename"]
    image_path = scene.speedplus_dir / "synthetic" / "images" / filename
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    mask_path = scene.mask_root / filename.replace(".jpg", ".png")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    rvec, tvec = geometry.speedplus_pose_to_opencv(
        record["q_vbs2tango_true"], record["r_Vo2To_vbs_true"]
    )
    projected, _ = cv2.projectPoints(points, rvec, tvec, scene.camera.K, scene.camera.dist)
    for keypoint_index, (x, y) in enumerate(projected.reshape(-1, 2)):
        xi, yi = int(round(x)), int(round(y))
        inside = 0 <= xi < image.shape[1] and 0 <= yi < image.shape[0]
        on_mask = inside and mask is not None and mask[yi, xi] > 0
        color = (0, 255, 0) if on_mask else (0, 0, 255)
        if inside:
            cv2.circle(image, (xi, yi), 3, color, -1)
            cv2.putText(image, str(keypoint_index), (xi + 3, yi - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1)  # fmt: skip
    return image


def _write_overlays(output_dir: Path, scene: _Scene, points: np.ndarray,
                    image_indices: list[int]) -> None:  # fmt: skip
    """Projection overlays of the first anchors and their contact sheet."""
    overlay_dir = output_dir / "projection_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    panels = []
    for image_index in image_indices:
        record = scene.labels[image_index]
        image = _overlay(scene, record, points)
        cv2.imwrite(str(overlay_dir / record["filename"]), image)
        scale = min(640 / image.shape[1], 400 / image.shape[0])
        panels.append(cv2.resize(image, None, fx=scale, fy=scale))
    if not panels:
        return
    columns = 3
    rows = []
    for start in range(0, len(panels), columns):
        row = panels[start : start + columns]
        while len(row) < columns:
            row.append(np.zeros_like(panels[0]))
        rows.append(np.hstack(row))
    cv2.imwrite(str(output_dir / "projection_contact_sheet.jpg"), np.vstack(rows))


def _settings(source_defaults: Mapping[str, Any]) -> ExtractorSettings:
    """The extractor settings with the given overrides.

    Raises:
        TypeError: For an unknown setting.
    """
    known = {item.name for item in fields(ExtractorSettings)}
    unknown = sorted(set(source_defaults) - known)
    if unknown:
        raise TypeError(f"unknown extractor settings {unknown}; known: {sorted(known)}")
    return ExtractorSettings(**source_defaults)


def _load_scene(speedplus_dir: Path, mask_root: Path) -> tuple[_Scene, np.ndarray, np.ndarray]:
    """The synthetic train labels and camera; unit quaternions and translations."""
    labels = json.loads((speedplus_dir / "synthetic" / "train.json").read_text())
    quaternions = np.asarray([record["q_vbs2tango_true"] for record in labels], dtype=np.float64)
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    translations = np.asarray([record["r_Vo2To_vbs_true"] for record in labels], dtype=np.float64)
    camera = geometry.load_camera(speedplus_dir / "camera.json")
    return _Scene(speedplus_dir, mask_root, labels, camera), quaternions, translations


class _FeatureCache:
    """Per-image SIFT features and OpenCV poses, computed once."""

    def __init__(self, scene: _Scene, settings: ExtractorSettings) -> None:
        """Create the SIFT detector of ``settings``."""
        self.scene = scene
        self.settings = settings
        self.sift = cv2.SIFT.create(
            nfeatures=settings.nfeatures, contrastThreshold=settings.sift_contrast_threshold
        )
        self.features: dict[int, ImageFeatures] = {}
        self.poses: dict[int, Pose] = {}

    def get(self, index: int) -> ImageFeatures:
        """The features of synthetic train image ``index``."""
        if index not in self.features:
            filename = self.scene.labels[index]["filename"]
            self.features[index] = extract_features(
                self.scene.speedplus_dir / "synthetic" / "images" / filename,
                self.scene.mask_root / filename.replace(".jpg", ".png"),
                self.sift,
                self.settings.mask_erode_px,
            )
        return self.features[index]

    def pose(self, index: int) -> Pose:
        """The OpenCV object-to-camera pose of synthetic train image ``index``."""
        if index not in self.poses:
            record = self.scene.labels[index]
            self.poses[index] = geometry.speedplus_pose_to_opencv(
                record["q_vbs2tango_true"], record["r_Vo2To_vbs_true"]
            )
        return self.poses[index]


def _match_pairs(
    pairs: list[tuple[int, int, float]], cache: _FeatureCache
) -> tuple[DisjointSet, list[dict[str, Any]], list[dict[str, Any]]]:
    """Match and triangulate every pair; return the tracks, the edges and pair diagnostics."""
    disjoint_set = DisjointSet()
    edges: list[dict[str, Any]] = []
    pair_records = []
    for first_index, second_index, view_angle in pairs:
        features = (cache.get(first_index), cache.get(second_index))
        matches = mutual_ratio_matches(*features, cache.settings.ratio_threshold)
        triangulated = triangulate_matches(
            matches, features, (cache.pose(first_index), cache.pose(second_index)),
            cache.scene.camera, cache.settings,
        )  # fmt: skip
        for match in triangulated:
            first_node, second_node = (first_index, match["query"]), (second_index, match["train"])
            disjoint_set.union(first_node, second_node)
            edges.append({**match, "first_node": first_node, "second_node": second_node})
        pair_records.append(
            {
                "first": cache.scene.labels[first_index]["filename"],
                "second": cache.scene.labels[second_index]["filename"],
                "pose_angle_deg": view_angle,
                "first_features": len(features[0].points),
                "second_features": len(features[1].points),
                "mutual_ratio_matches": len(matches),
                "triangulated_matches": len(triangulated),
            }
        )
    return disjoint_set, edges, pair_records


def _validated(
    candidates: list[dict[str, Any]], scene: _Scene, settings: ExtractorSettings, seed: int,
    validation_mask_root: Path,
) -> list[dict[str, Any]]:  # fmt: skip
    """Score containment on validation views drawn with ``seed + 1``; keep the contained."""
    labels = json.loads((scene.speedplus_dir / "synthetic" / "validation.json").read_text())
    rng = np.random.default_rng(seed + 1)
    indices = rng.choice(len(labels), size=min(settings.validation_views, len(labels)),
                         replace=False)  # fmt: skip
    add_mask_containment(candidates, labels, indices, validation_mask_root, scene.camera)
    return [c for c in candidates if c["mask_containment"] >= settings.min_mask_containment]


def _relative(path: Path, out: Path) -> str:
    return Path(os.path.relpath(path, out)).as_posix()


def _report(
    counts: Mapping[str, int], selected: list[dict[str, Any]], points: np.ndarray,
    configuration: Mapping[str, Any],
) -> dict[str, Any]:  # fmt: skip
    """The extractor's ``report.json``."""
    return {
        "configuration": dict(configuration),
        **counts,
        "selected_keypoints": len(selected),
        "selected_bounds_min": points.min(axis=0).tolist(),
        "selected_bounds_max": points.max(axis=0).tolist(),
        "selected_median_track_images": float(np.median([item["images"] for item in selected])),
        "selected_median_reprojection_px": float(
            np.median([item["median_reprojection_px"] for item in selected])
        ),
        "selected_median_spread_m": float(np.median([item["spread_m"] for item in selected])),
        "selected_mean_mask_containment": float(
            np.mean([item["mask_containment"] for item in selected])
        ),
        "selected_min_mask_containment": float(
            np.min([item["mask_containment"] for item in selected])
        ),
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _tracked_payload(points: np.ndarray, seed: int) -> dict[str, Any]:
    return {
        "object_name": "tango",
        "keypoint_layout": TRACKED_LAYOUT_NAME,
        "keypoint_layout_sha256": layout_points_sha256(points),
        "official_keypoint_count": 0,
        "keypoints_3d": points.tolist(),
        "extraction": {
            "source": "SPEED+ synthetic train images and masks",
            "method": "masked SIFT, mutual ratio matching, known-pose triangulation, feature "
            "tracks, spatial quality selection",
            "seed": seed,
        },
    }


def extract_tracked_landmarks(
    root: Path,
    mask_root: Path,
    out: Path,
    *,
    seed: int = TRACKED_SEED,
    force: bool = False,
    **source_defaults: Any,  # noqa: ANN401
) -> Path:
    """Extract mesh-free tracked landmarks from masked SPEED+ synthetic train images.

    Writes ``keypoints_3d.json``, ``landmark_candidates.json``, ``pair_diagnostics.json``,
    ``report.json``, the projection overlays and, when matplotlib is importable, a 3D scatter
    plot into ``out`` (built beside it and renamed into place). Paths in the report are relative
    to ``out``.

    Args:
        root: The SPEED+ v2 directory (``camera.json``, ``synthetic/{images,train.json,
            validation.json}``), e.g. ``<root>/raw/extracted/speedplusv2``.
        mask_root: Synthetic train masks (``<stem>.png``), e.g. the 64-point conversion's
            ``train/masks``.
        out: Output directory.
        seed: Seed of the anchor draw; the validation draw uses ``seed + 1``.
        force: Replace an existing ``out``.
        **source_defaults: Overrides of :class:`ExtractorSettings` defaults.

    Returns:
        The written ``keypoints_3d.json``.

    Raises:
        ImportError: Without scikit-learn (the ``doper[speedplus]`` extra).
        TypeError: For an unknown setting.
        FileExistsError: If ``out`` exists and ``force`` is false.
        RuntimeError: If fewer landmarks than requested survive.
    """
    dbscan = _require_dbscan()
    settings = _settings(source_defaults)
    root, mask_root, out = Path(root), Path(mask_root), Path(out)
    with file_lock(sibling_lock(out)):
        if out.exists() and not force:
            raise FileExistsError(f"{out} exists; pass --force to replace it")
        partial = out.with_name(f".{out.name}.partial")
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
        try:
            _extract(root, mask_root, out, partial, settings, seed, dbscan)
        except BaseException:
            shutil.rmtree(partial)
            raise
        if out.exists():
            shutil.rmtree(out)
        partial.rename(out)
    return out / KEYPOINTS_NAME


def _extract(
    root: Path, mask_root: Path, out: Path, partial: Path, settings: ExtractorSettings,
    seed: int, dbscan: Callable[..., Any],
) -> None:  # fmt: skip
    """Run the extraction and write its outputs into ``partial``."""
    scene, quaternions, translations = _load_scene(root, mask_root)
    anchors, pairs = choose_pairs(quaternions, translations, settings,
                                  np.random.default_rng(seed))  # fmt: skip
    cache = _FeatureCache(scene, settings)
    disjoint_set, edges, pair_records = _match_pairs(pairs, cache)
    tracks = build_track_candidates(
        disjoint_set, edges, settings.min_track_images, settings.max_track_spread_m
    )
    merged = merge_candidates(tracks, settings.merge_radius_m, dbscan)
    validation_mask_root = (
        Path(settings.validation_mask_root)
        if settings.validation_mask_root is not None
        else mask_root.parent.parent / "val" / "masks"
    )
    eligible = _validated(merged, scene, settings, seed, validation_mask_root)
    selected = select_spatially_distributed(eligible, settings.num_keypoints)
    if len(selected) < settings.num_keypoints:
        raise RuntimeError(
            f"only {len(selected)} landmark candidates survived; requested {settings.num_keypoints}"
        )
    points = np.asarray([candidate["xyz"] for candidate in selected], dtype=np.float64)
    _write_json(partial / KEYPOINTS_NAME, _tracked_payload(points, seed))
    serializable = [{**candidate, "xyz": np.asarray(candidate["xyz"]).tolist()}
                    for candidate in merged]  # fmt: skip
    _write_json(partial / "landmark_candidates.json", serializable)
    _write_json(partial / "pair_diagnostics.json", pair_records)
    configuration = {
        "speedplus_root": _relative(root, out),
        "mask_root": _relative(mask_root, out),
        "output_dir": ".",
        **asdict(settings),
        "validation_mask_root": _relative(validation_mask_root, out),
        "seed": seed,
    }
    counts = {
        "candidate_pairs": len(pairs),
        "unique_images_processed": len(cache.features),
        "triangulated_edges": len(edges),
        "multi_image_tracks": len(tracks),
        "merged_landmark_candidates": len(merged),
        "mask_validated_landmark_candidates": len(eligible),
    }
    _write_json(partial / "report.json", _report(counts, selected, points, configuration))
    _write_overlays(partial, scene, points, anchors[: min(settings.overlay_images, len(anchors))])
    _write_scatter(partial, points)
