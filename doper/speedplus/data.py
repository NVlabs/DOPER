# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SPEED+ v2: fetch and verify the dataset, then convert it to the 64-point hull layout.

:func:`fetch` downloads the Zenodo archive and the official Tango landmarks (SPNv2
``tangoPoints.mat``) with the standard library, checks them and extracts the archive::

    <root>/READY.json                       receipt (license, checksums, image counts)
    <root>/raw/speedplusv2.zip              Zenodo record 5588480
    <root>/raw/tangoPoints.mat
    <root>/raw/extracted/speedplusv2/{camera.json, LICENSE.md, synthetic, lightbox, sunlamp}

Checks: the archive has its pinned size and MD5. Its SHA-256 is trusted on first use: the
receipt records it with ``checksum_origin: observed`` and a later fetch fails if the archive
differs from it. ``tangoPoints.mat`` has its pinned SHA-256. Archive members must be relative,
without ``..``, backslashes or links; members outside ``speedplusv2/`` are skipped and listed.
Each split has its pinned image count (47,966 synthetic train, 11,994 synthetic validation,
6,740 lightbox, 2,791 sunlamp) and every labeled image exists.

:func:`convert` writes the COCO supervision of the deterministic 64-point layout
(``doper.speedplus.geometry.expand_tango_keypoints_64``) to ``<root>/derived/tango_64``::

    keypoints_3d.json, camera_intrinsics.json, conversion_manifest.json
    <split>/{annotations.json, images -> the SPEED+ images, masks/<stem>.png}

The paper's masks are projected hulls (``mask_mode="hull"``): the convex hull of the in-frame
projected 64 points; the box is the mask's box. ``auto`` uses official SPNv2 masks wherever they
are installed, so it records ``mask_sources`` and is a non-paper conversion as soon as one
mask is not a projected hull; such a conversion, and ``official`` or ``none``, needs
``accept_non_paper_masks``. Annotations carry the native pose as
``speedplus_q_vbs2tango_true`` and ``speedplus_r_Vo2To_vbs_true``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np

from ..bop.sources import (
    HASH_BLOCK_BYTES,
    READY_NAME,
    canonical_sha256,
    file_lock,
    link_relative,
    publish_json,
    sha256_file,
    sibling_lock,
)
from ..bop.tless_sensors import HttpGet, http_download
from ..recipes import profile_path as profile_path  # re-exported: the packaged profile lookup
from . import geometry
from .definitions import DERIVED_64 as DERIVED_64  # re-exported: the conversion's directory
from .definitions import MASK_MODES, PAPER_MASK_MODE, SPLIT_NAMES

RAW_DIR = "raw"
ARCHIVE_PATH = "raw/speedplusv2.zip"
TANGO_POINTS_PATH = "raw/tangoPoints.mat"
EXTRACTED_DIR = "raw/extracted"
SPEEDPLUS_DIR = "raw/extracted/speedplusv2"
LOCK_NAME = "stage.lock"
MANIFEST_NAME = "conversion_manifest.json"
KEYPOINTS_NAME = "keypoints_3d.json"
CAMERA_NAME = "camera_intrinsics.json"
ANNOTATIONS_NAME = "annotations.json"
# Split -> (SPEED+ domain directory, label file), in ``SPLIT_NAMES`` order.
SPLITS: dict[str, tuple[str, str]] = dict(
    zip(
        SPLIT_NAMES,
        (
            ("synthetic", "train.json"),
            ("synthetic", "validation.json"),
            ("lightbox", "test.json"),
            ("sunlamp", "test.json"),
        ),
        strict=True,
    )
)
LAYOUT_NAME = "speedplus_official_11_plus_body_surface_fps_53"
PROJECTED_HULL = "projected_hull"
OFFICIAL = "official"
# Read every Nth source image to check that all share the first image's size.
VALIDATE_EVERY = 1000
# PNG level 3 keeps the sparse binary masks compact without spending most of a 69k-image
# conversion in maximum-effort DEFLATE.
PNG_COMPRESSION = 3
MASK_THRESHOLD = 128
MIN_HULL_POINTS = 3
MIN_BOX_POINTS = 2
VISIBLE = 2
COPY_BLOCK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class SpeedplusProfile:
    """The pinned SPEED+ sources (``doper/recipes/profiles/speedplus.json``).

    Attributes:
        name: Profile name.
        license: Dataset license: ``name``, ``url``, ``file`` (in the extraction).
        archive: Zenodo archive: ``name``, ``url``, ``bytes``, ``md5`` and ``sha256`` (null
            until pinned; trusted on first use).
        tango_points: Official landmarks: ``name``, ``url``, ``sha256``.
        extracted_dir: Top directory of the archive.
        image_counts: Images per split.
        image_size: ``(width, height)`` of every image.
        path: The profile file.
    """

    name: str
    license: Mapping[str, Any]
    archive: Mapping[str, Any]
    tango_points: Mapping[str, Any]
    extracted_dir: str
    image_counts: Mapping[str, int]
    image_size: tuple[int, int]
    path: Path


@dataclass(frozen=True)
class Receipt:
    """A fetched, verified and extracted SPEED+ root.

    Attributes:
        archive_sha256: SHA-256 of the Zenodo archive.
        checksum_origin: ``pinned`` or ``observed`` (trust on first use).
        image_counts: Images per split.
        speedplus_dir: The extracted ``speedplusv2`` directory.
        tango_points: The verified ``tangoPoints.mat``.
        receipt: ``READY.json``.
    """

    archive_sha256: str
    checksum_origin: str
    image_counts: Mapping[str, int]
    speedplus_dir: Path
    tango_points: Path
    receipt: Path


@dataclass(frozen=True)
class ConversionManifest:
    """A 64-point conversion (``conversion_manifest.json``).

    Attributes:
        out: Output directory.
        mask_mode: ``hull``, ``auto``, ``official`` or ``none``.
        mask_sources: Masks per source (``projected_hull``, ``official``, ``none``).
        paper_masks: Whether every mask is a projected hull, as in the paper.
        keypoint_layout_sha256: SHA-256 of the 64x3 little-endian float64 layout.
        splits: Per-split results.
        manifest: The manifest file.
    """

    out: Path
    mask_mode: str
    mask_sources: Mapping[str, int]
    paper_masks: bool
    keypoint_layout_sha256: str
    splits: tuple[Mapping[str, Any], ...]
    manifest: Path


def load_profile(name_or_path: str | Path = "speedplus") -> SpeedplusProfile:
    """Load the packaged SPEED+ profile, or a profile file by path.

    Raises:
        ValueError: If the counts do not name the four splits.
    """
    path = Path(name_or_path)
    if not path.is_file():
        path = profile_path(str(name_or_path))
    payload = json.loads(path.read_text())
    counts = {split: int(count) for split, count in payload["image_counts"].items()}
    if set(counts) != set(SPLITS):
        raise ValueError(f"{path}: image_counts must name {sorted(SPLITS)}")
    width, height = payload["image_size"]
    return SpeedplusProfile(
        name=payload["name"],
        license=dict(payload["license"]),
        archive=dict(payload["archive"]),
        tango_points=dict(payload["tango_points"]),
        extracted_dir=payload["extracted_dir"],
        image_counts=counts,
        image_size=(int(width), int(height)),
        path=path,
    )


def _profile_sha256(profile: SpeedplusProfile) -> str:
    fields = {k: v for k, v in profile.__dict__.items() if k != "path"}
    return canonical_sha256({**fields, "image_size": list(profile.image_size)})


def _digests(path: Path) -> tuple[str, str]:
    """MD5 and SHA-256 of a file in one pass."""
    md5 = hashlib.md5(usedforsecurity=False)
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(HASH_BLOCK_BYTES), b""):
            md5.update(block)
            sha.update(block)
    return md5.hexdigest(), sha.hexdigest()


def _download(url: str, target: Path, downloader: HttpGet | None, max_bytes: int | None) -> Path:
    """Download ``url`` to ``<target>.part`` and return that file."""
    partial = target.with_name(target.name + ".part")
    partial.unlink(missing_ok=True)
    if downloader is None:
        http_download(url, partial, max_bytes)
    else:
        downloader(url, partial)
    return partial


def _verify_archive(
    archive: Path, spec: Mapping[str, Any], recorded: str | None
) -> tuple[str, str]:
    """Check the archive's size, MD5 and SHA-256; return the SHA-256 and its origin."""
    retry = "; delete it so it is fetched again"
    size = archive.stat().st_size
    if spec.get("bytes") is not None and size != spec["bytes"]:
        raise ValueError(f"{archive} has {size} bytes; expected {spec['bytes']}{retry}")
    md5, digest = _digests(archive)
    if spec.get("md5") and md5 != spec["md5"]:
        raise ValueError(f"MD5 mismatch for {archive}: expected {spec['md5']}, got {md5}{retry}")
    pinned = spec.get("sha256")
    if pinned and digest != pinned:
        raise ValueError(f"SHA-256 {digest} of {archive} differs from the pinned {pinned}{retry}")
    if recorded is not None and digest != recorded:
        raise ValueError(
            f"SHA-256 {digest} of {archive} differs from the {recorded} recorded in "
            f"{READY_NAME}; the staged source changed"
        )
    return digest, "pinned" if pinned else "observed"


def _fetch_points(root: Path, spec: Mapping[str, Any], downloader: HttpGet | None) -> Path:
    """Download ``tangoPoints.mat`` if missing and check its pinned SHA-256."""
    target = root / TANGO_POINTS_PATH
    candidate = target if target.is_file() else _download(spec["url"], target, downloader, None)
    digest = sha256_file(candidate)
    if digest != spec["sha256"]:
        if candidate != target:
            candidate.unlink()
        raise ValueError(
            f"SHA-256 {digest} of {spec['name']} differs from the pinned {spec['sha256']}"
        )
    if candidate != target:
        os.replace(candidate, target)
    return target


def _member_path(info: zipfile.ZipInfo) -> PurePosixPath:
    path = PurePosixPath(info.filename)
    if (
        not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in info.filename
        or stat.S_ISLNK(info.external_attr >> 16)
    ):
        raise ValueError(f"unsafe SPEED+ archive member {info.filename!r}")
    return path


def _unzip(archive: Path, temporary: Path, top: str) -> list[str]:
    """Extract the members under ``top/`` into ``temporary``; return the skipped members."""
    skipped, seen = [], set()
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            path = _member_path(info)
            if info.filename in seen:
                raise ValueError(f"duplicate SPEED+ archive member {info.filename!r}")
            seen.add(info.filename)
            if path.parts[0] != top:
                skipped.append(info.filename)
                continue
            if info.is_dir():
                continue
            destination = temporary.joinpath(*path.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, destination.open("wb") as sink:
                shutil.copyfileobj(source, sink, COPY_BLOCK_BYTES)
    return skipped


def validate_image_counts(speedplus_dir: Path, counts: Mapping[str, int]) -> dict[str, int]:
    """Check each split's label count against ``counts`` and that every image exists.

    The pinned counts come from the profile; a fixture profile lowers them.

    Args:
        speedplus_dir: The extracted ``speedplusv2`` directory.
        counts: Expected images per split.

    Returns:
        The images per split.

    Raises:
        FileNotFoundError: If a label file is missing.
        ValueError: If a count differs, a labeled image is missing or a domain holds images
            no label names.
    """
    found: dict[str, int] = {}
    labelled: dict[str, set[str]] = {}
    for split, (domain, label) in SPLITS.items():
        label_path = speedplus_dir / domain / label
        if not label_path.is_file():
            raise FileNotFoundError(f"SPEED+ label file {label_path} is missing")
        names = [record["filename"] for record in json.loads(label_path.read_text())]
        if len(names) != counts[split]:
            raise ValueError(f"{split} has {len(names)} images; expected {counts[split]}")
        images = speedplus_dir / domain / "images"
        missing = [name for name in names if not (images / name).is_file()]
        if missing:
            raise ValueError(f"{split}: {len(missing)} labeled images are missing: {missing[:3]}")
        found[split] = len(names)
        labelled.setdefault(domain, set()).update(names)
    for domain, names in labelled.items():
        on_disk = {p.name for p in (speedplus_dir / domain / "images").glob("*.jpg")}
        if on_disk != names:
            raise ValueError(f"{domain}: {len(on_disk - names)} images have no label")
    return found


def _check_extraction(speedplus_dir: Path, profile: SpeedplusProfile) -> dict[str, int]:
    for name in ("camera.json", profile.license["file"]):
        if not (speedplus_dir / name).is_file():
            raise ValueError(f"extracted SPEED+ is missing {name}")
    return validate_image_counts(speedplus_dir, profile.image_counts)


def _extract(root: Path, profile: SpeedplusProfile) -> tuple[dict[str, int], list[str]]:
    """Extract into a temporary directory, check it, and rename it into place."""
    extracted = root / EXTRACTED_DIR
    extracted.mkdir(parents=True, exist_ok=True)
    temporary = extracted / f".{profile.extracted_dir}-extract"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        skipped = _unzip(root / ARCHIVE_PATH, temporary, profile.extracted_dir)
        counts = _check_extraction(temporary / profile.extracted_dir, profile)
        (temporary / profile.extracted_dir).rename(root / SPEEDPLUS_DIR)
    finally:
        shutil.rmtree(temporary)
    return counts, skipped


def _receipt(record: Mapping[str, Any], root: Path) -> Receipt:
    return Receipt(
        archive_sha256=record["archive"]["sha256"],
        checksum_origin=record["archive"]["checksum_origin"],
        image_counts=dict(record["image_counts"]),
        speedplus_dir=root / record["extracted"],
        tango_points=root / record["tango_points"]["path"],
        receipt=root / READY_NAME,
    )


def _reverify(
    root: Path, profile: SpeedplusProfile, previous: Mapping[str, Any], downloader: HttpGet | None
) -> Receipt:
    """Re-check a receipted root: the kept archive (if any), the points and the images."""
    if (
        previous.get("dataset") != profile.name
        or previous["archive"]["url"] != profile.archive["url"]
    ):
        raise ValueError(f"{root} was fetched from another SPEED+ source; use another root")
    archive = root / ARCHIVE_PATH
    if archive.is_file():
        _verify_archive(archive, profile.archive, previous["archive"]["sha256"])
    _fetch_points(root, profile.tango_points, downloader)
    _check_extraction(root / SPEEDPLUS_DIR, profile)
    return _receipt(previous, root)


def _require_consent(consent: bool, profile: SpeedplusProfile) -> None:
    if not consent:
        gigabytes = profile.archive["bytes"] / 1e9
        raise PermissionError(
            f"SPEED+ is distributed under {profile.license['name']} ({profile.license['url']}) "
            f"and fetching it downloads the {gigabytes:.1f} GB {profile.archive['name']}; "
            "accept the license and the download with --accept-license"
        )


def fetch(
    root: Path,
    *,
    consent: bool,
    downloader: HttpGet | None = None,
    profile: SpeedplusProfile | None = None,
) -> Receipt:
    """Download, verify and extract SPEED+ v2 and the official Tango landmarks.

    A root with a receipt is only re-verified: the kept archive (when it is still there) must
    have the recorded SHA-256, the landmarks their pinned SHA-256 and the extraction its
    image counts. Nothing is downloaded again.

    Args:
        root: Dataset root.
        consent: The user accepts the CC BY-NC-SA 4.0 license and the 16.9 GB download.
        downloader: ``download(url, destination)``; defaults to the standard-library
            ``doper.bop.tless_sensors.http_download``.
        profile: Pinned sources (default: the packaged ``speedplus`` profile).

    Returns:
        The receipt.

    Raises:
        PermissionError: Without ``consent``.
        FileExistsError: For an extraction without a receipt (inspect it before retrying).
        ValueError: For a size, MD5 or SHA-256 mismatch, an unsafe archive member, a wrong
            image count or a missing image.
    """
    profile = profile or load_profile()
    _require_consent(consent, profile)
    root = Path(root)
    with file_lock(root / RAW_DIR / LOCK_NAME):
        ready = root / READY_NAME
        if ready.is_file():
            return _reverify(root, profile, json.loads(ready.read_text()), downloader)
        if (root / SPEEDPLUS_DIR).exists():
            raise FileExistsError(f"{root / SPEEDPLUS_DIR} has no receipt; inspect it first")
        archive = root / ARCHIVE_PATH
        if not archive.is_file():
            spec = profile.archive
            os.replace(_download(spec["url"], archive, downloader, spec["bytes"]), archive)
        digest, origin = _verify_archive(archive, profile.archive, None)
        points = _fetch_points(root, profile.tango_points, downloader)
        counts, skipped = _extract(root, profile)
        record = {
            "dataset": profile.name,
            "profile_sha256": _profile_sha256(profile),
            "license": dict(profile.license),
            "archive": {
                **{k: profile.archive[k] for k in ("url", "zenodo_record", "bytes", "md5")
                   if k in profile.archive},
                "path": ARCHIVE_PATH,
                "sha256": digest,
                "checksum_origin": origin,
            },
            "tango_points": {
                "url": profile.tango_points["url"],
                "path": TANGO_POINTS_PATH,
                "sha256": sha256_file(points),
                "checksum_origin": "pinned",
                "license": profile.tango_points.get("license"),
            },
            "extracted": SPEEDPLUS_DIR,
            "skipped_members": skipped,
            "image_counts": counts,
            "image_size": list(profile.image_size),
        }  # fmt: skip
        publish_json(ready, record)
    return _receipt(record, root)


def require_ready(root: Path) -> dict[str, Any]:
    """The ``READY.json`` of a fetched SPEED+ root, with its landmarks re-checked.

    Raises:
        FileNotFoundError: If the root was not fetched.
        ValueError: If the receipt is not a SPEED+ one or the landmarks changed.
    """
    path = Path(root) / READY_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing; run `doper prepare speedplus --stage fetch` first"
        )
    ready = json.loads(path.read_text())
    if "tango_points" not in ready or "extracted" not in ready:
        raise ValueError(f"{path} is not a SPEED+ receipt")
    points = Path(root) / ready["tango_points"]["path"]
    if sha256_file(points) != ready["tango_points"]["sha256"]:
        raise ValueError(f"{points} differs from its receipt")
    return ready


def layout_points_sha256(points: np.ndarray) -> str:
    """The points hash of a keypoint layout: SHA-256 of its little-endian float64 values.

    This is the layout's ``keypoint_layout_sha256``; the SHA-256 of a layout file is
    ``doper.speedplus.keypoints.layout_sha256``.
    """
    from ..keypoints import points_sha256

    return points_sha256(points)


def keypoint_payload(
    pixels: np.ndarray, depth: np.ndarray, width: int, height: int
) -> tuple[list[float | int], int]:
    """COCO keypoints of projected points: ``x, y, 2`` when in frame, else ``x, y, 0``.

    A point is visible when its pixel is finite and inside ``[0, width) x [0, height)`` and its
    depth is positive. Coordinates are kept as projected, also for invisible points.

    Returns:
        The flat COCO keypoint list and the number of visible points.
    """
    output: list[float | int] = []
    count = 0
    for (x, y), z in zip(pixels, depth, strict=True):
        visible = bool(
            np.isfinite(x) and np.isfinite(y) and z > 0 and 0 <= x < width and 0 <= y < height
        )
        output.extend([float(x), float(y), VISIBLE if visible else 0])
        count += visible
    return output, count


def _image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    height, width = image.shape[:2]
    return width, height


def _official_mask(
    speedplus_dir: Path, masks_root: Path | None, domain: str, filename: str
) -> Path | None:
    names = [filename, f"{Path(filename).stem}.png"]
    roots = [speedplus_dir / domain / "masks"]
    if masks_root is not None:
        roots += [
            masks_root / domain / "masks",
            masks_root / domain,
            masks_root / f"{domain}_masks",
        ]
    return next((r / n for r in roots for n in names if (r / n).is_file()), None)


def projected_hull_mask(
    pixels: np.ndarray, depth: np.ndarray, width: int, height: int
) -> np.ndarray:
    """The filled convex hull of the in-frame projected points, as a 0/255 mask.

    Raises:
        ValueError: If fewer than three points are in frame.
    """
    valid = (
        np.isfinite(pixels).all(axis=1)
        & (depth > 0)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    if valid.sum() < MIN_HULL_POINTS:
        raise ValueError("fewer than three projectable Tango landmarks")
    hull = cv2.convexHull(np.round(pixels[valid]).astype(np.int32))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def _mask_bbox(mask: np.ndarray) -> tuple[list[float], float]:
    ys, xs = np.nonzero(mask >= MASK_THRESHOLD)
    if len(xs) == 0:
        raise ValueError("foreground mask is empty")
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], float(len(xs))


def _read_official(path: Path, width: int, height: int) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(path)
    if raw.ndim == 3:
        raw = raw[..., 0]
    if raw.shape != (height, width):
        raise ValueError(f"mask {path} has shape {raw.shape}, expected {(height, width)}")
    return np.where(raw >= MASK_THRESHOLD, 255, 0).astype(np.uint8)


def _clipped_box(pixels: np.ndarray, depth: np.ndarray, width: int, height: int) -> list[float]:
    valid = np.isfinite(pixels).all(axis=1) & (depth > 0)
    if valid.sum() < MIN_BOX_POINTS:
        raise ValueError("fewer than two projectable Tango landmarks")
    clipped = pixels[valid].copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, width - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, height - 1)
    x1, y1 = clipped.min(axis=0)
    x2, y2 = clipped.max(axis=0)
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


@dataclass(frozen=True)
class _SplitInputs:
    """What every split of one conversion shares."""

    speedplus_dir: Path
    output: Path
    masks_root: Path | None
    mask_mode: str
    camera: geometry.Camera
    points: np.ndarray
    validate_every: int
    image_size: tuple[int, int]


def _materialize_mask(
    inputs: _SplitInputs, split_root: Path, domain: str, filename: str,
    pixels: np.ndarray, depth: np.ndarray, size: tuple[int, int],
) -> tuple[str | None, list[float], float, str]:  # fmt: skip
    """Write one mask; return its file, box, area and source."""
    width, height = size
    official = _official_mask(inputs.speedplus_dir, inputs.masks_root, domain, filename)
    if inputs.mask_mode == OFFICIAL and official is None:
        raise FileNotFoundError(f"official mask not found for {domain}/{filename}")
    if official is not None and inputs.mask_mode in (OFFICIAL, "auto"):
        mask, source = _read_official(official, width, height), OFFICIAL
    elif inputs.mask_mode == "none":
        box = _clipped_box(pixels, depth, width, height)
        return None, box, float(max(0.0, box[2] * box[3])), "none"
    else:
        mask, source = projected_hull_mask(pixels, depth, width, height), PROJECTED_HULL
    path = split_root / "masks" / f"{Path(filename).stem}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION]):
        raise OSError(f"failed to write mask {path}")
    box, area = _mask_bbox(mask)
    return f"masks/{path.name}", box, area, source


def _annotation(
    image_id: int, record: Mapping[str, Any], keypoints: list[float | int], visible: int,
    mask: tuple[str | None, list[float], float, str],
) -> dict[str, Any]:  # fmt: skip
    mask_file, box, area, _ = mask
    annotation = {
        "id": image_id,
        "image_id": image_id,
        "category_id": 1,
        "bbox": box,
        "area": area,
        "iscrowd": 0,
        "keypoints": keypoints,
        "num_keypoints": visible,
        "speedplus_q_vbs2tango_true": np.asarray(record["q_vbs2tango_true"], np.float64).tolist(),
        "speedplus_r_Vo2To_vbs_true": np.asarray(record["r_Vo2To_vbs_true"], np.float64).tolist(),
    }
    if mask_file is not None:
        annotation["mask_file"] = mask_file
    return annotation


def _check_size(path: Path, index: int, size: tuple[int, int], every: int) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if every > 0 and index % every == 0 and _image_size(path) != size:
        raise ValueError(f"mixed image sizes: {path} is {_image_size(path)}, expected {size}")


def _coco(label: str, images: list, annotations: list, layout: str) -> dict[str, Any]:
    return {
        "info": {
            "description": "SPEED+ converted for DOPER",
            "source_split": label,
            "keypoint_layout_sha256": layout,
        },
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": 1,
                "name": "tango",
                "supercategory": "spacecraft",
                "keypoints": [f"tango_{index:02d}" for index in range(geometry.LAYOUT_64_COUNT)],
                "skeleton": [],
            }
        ],
    }


def _convert_split(inputs: _SplitInputs, split: str, root: Path) -> dict[str, Any]:
    """Convert one split into ``inputs.output / split``; return its manifest entry."""
    domain, label = SPLITS[split]
    label_path = inputs.speedplus_dir / domain / label
    image_root = inputs.speedplus_dir / domain / "images"
    records = json.loads(label_path.read_text())
    if not records:
        raise ValueError(f"no records in {label_path}")
    split_root = inputs.output / split
    split_root.mkdir(parents=True)
    link_relative(split_root / "images", image_root)
    size = _image_size(image_root / records[0]["filename"])
    if size != inputs.image_size:
        raise ValueError(f"{split} images are {size}; the receipt pins {inputs.image_size}")
    images, annotations, sources = [], [], {}
    min_visible = len(inputs.points)
    for index, record in enumerate(records):
        filename = record["filename"]
        _check_size(image_root / filename, index, size, inputs.validate_every)
        pixels, depth = geometry.project_keypoints(
            inputs.points, record["q_vbs2tango_true"], record["r_Vo2To_vbs_true"], inputs.camera
        )
        keypoints, visible = keypoint_payload(pixels, depth, *size)
        min_visible = min(min_visible, visible)
        mask = _materialize_mask(inputs, split_root, domain, filename, pixels, depth, size)
        sources[mask[3]] = sources.get(mask[3], 0) + 1
        image = {"id": index + 1, "file_name": filename, "width": size[0], "height": size[1]}
        images.append({**image, "speedplus_domain": domain})
        annotations.append(_annotation(index + 1, record, keypoints, visible, mask))
    relative_label = Path(os.path.relpath(label_path, root)).as_posix()
    coco = _coco(relative_label, images, annotations, layout_points_sha256(inputs.points))
    (split_root / ANNOTATIONS_NAME).write_text(json.dumps(coco, indent=2) + "\n")
    return {
        "split": split,
        "domain": domain,
        "count": len(records),
        "width": size[0],
        "height": size[1],
        "min_in_frame_keypoints": min_visible,
        "mask_sources": sources,
        "annotations": f"{split}/{ANNOTATIONS_NAME}",
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_layout(output: Path, points: np.ndarray, camera: geometry.Camera) -> None:
    _write_json(
        output / KEYPOINTS_NAME,
        {
            "object_name": "tango",
            "keypoint_layout": LAYOUT_NAME,
            "keypoint_layout_sha256": layout_points_sha256(points),
            "official_keypoint_count": geometry.OFFICIAL_POINT_COUNT,
            "keypoints_3d": points.tolist(),
        },
    )
    # The image size is part of the normalized camera (``geometry.load_camera`` needs it).
    _write_json(
        output / CAMERA_NAME,
        {
            "camera_matrix": camera.K.tolist(),
            "distortion_coefficients": camera.dist.tolist(),
            "width": camera.width,
            "height": camera.height,
        },
    )


def _non_paper(sources: Mapping[str, int]) -> str:
    return (
        f"non-paper SPEED+ masks {dict(sources)} (the paper uses projected hulls only); pass "
        "--accept-non-paper-masks to accept them"
    )


def _manifest(payload: Mapping[str, Any], out: Path) -> ConversionManifest:
    return ConversionManifest(
        out=out,
        mask_mode=payload["mask_mode"],
        mask_sources=dict(payload["mask_sources"]),
        paper_masks=bool(payload["paper_masks"]),
        keypoint_layout_sha256=payload["keypoint_layout_sha256"],
        splits=tuple(payload["splits"]),
        manifest=out / MANIFEST_NAME,
    )


def _check_request(mask_mode: str, splits: Sequence[str], accept_non_paper_masks: bool) -> None:
    if mask_mode not in MASK_MODES:
        raise ValueError(f"mask_mode must be one of {MASK_MODES}, got {mask_mode!r}")
    unknown = sorted(set(splits) - set(SPLITS))
    if unknown or not splits or len(set(splits)) != len(splits):
        raise ValueError(f"splits must be distinct names of {sorted(SPLITS)}, got {list(splits)}")
    if mask_mode in (OFFICIAL, "none") and not accept_non_paper_masks:
        raise ValueError(_non_paper({mask_mode: 0}))


def _existing(
    out: Path, request: Mapping[str, Any], force: bool, accept_non_paper_masks: bool
) -> ConversionManifest | None:
    """The finished conversion of the same request, or None when ``out`` may be (re)built."""
    manifest = out / MANIFEST_NAME
    if force or not out.exists():
        return None
    payload = json.loads(manifest.read_text()) if manifest.is_file() else {}
    if {key: payload.get(key) for key in request} != request:
        raise FileExistsError(f"{out} holds another conversion; pass --force to rebuild it")
    if not payload["paper_masks"] and not accept_non_paper_masks:
        raise ValueError(_non_paper(payload["mask_sources"]))
    return _manifest(payload, out)


def _check_output(out: Path, root: Path, speedplus_dir: Path) -> None:
    resolved, data_dir = out.resolve(), speedplus_dir.resolve()
    if (
        resolved in (root.resolve(), data_dir)
        or data_dir in resolved.parents
        or (resolved in data_dir.parents)
    ):
        raise ValueError(f"output {out} overlaps the SPEED+ data {speedplus_dir}")


def convert(
    root: Path,
    out: Path,
    *,
    mask_mode: str = PAPER_MASK_MODE,
    splits: Sequence[str] = tuple(SPLITS),
    masks_root: Path | None = None,
    accept_non_paper_masks: bool = False,
    force: bool = False,
    validate_every: int = VALIDATE_EVERY,
) -> ConversionManifest:
    """Convert a fetched SPEED+ root to the 64-point COCO layout with masks.

    A finished conversion of the same mask mode and splits is returned as it is; another one
    needs ``force``. The output is built beside ``out`` and renamed into place.

    Args:
        root: Fetched dataset root (``READY.json``).
        out: Output directory, usually ``<root>/derived/tango_64``.
        mask_mode: ``hull`` (paper), ``auto``, ``official`` or ``none``.
        splits: Splits to convert, of ``train``, ``val``, ``lightbox`` and ``sunlamp``.
        masks_root: Extra root of official SPNv2 masks (``<domain>/masks``, ``<domain>`` or
            ``<domain>_masks``); masks under the domain's ``masks`` are always found.
        accept_non_paper_masks: Accept masks that are not all projected hulls.
        force: Rebuild an existing output.
        validate_every: Check every Nth image's size against the first one's.

    Returns:
        The manifest.

    Raises:
        FileNotFoundError: If the root was not fetched or an image or mask is missing.
        FileExistsError: If ``out`` holds another conversion and ``force`` is false.
        ValueError: For an unknown mode or split, an output that overlaps the data, mixed
            image sizes, or non-paper masks that were not accepted.
    """
    _check_request(mask_mode, splits, accept_non_paper_masks)
    root, out = Path(root), Path(out)
    ready = require_ready(root)
    speedplus_dir = root / ready["extracted"]
    _check_output(out, root, speedplus_dir)
    request = {"mask_mode": mask_mode, "splits_requested": list(splits)}
    with file_lock(sibling_lock(out)):
        existing = _existing(out, request, force, accept_non_paper_masks)
        if existing is not None:
            return existing
        partial = out.with_name(f".{out.name}.partial")
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
        try:
            payload = _build(root, ready, partial, request, masks_root, validate_every)
            if not payload["paper_masks"] and not accept_non_paper_masks:
                raise ValueError(_non_paper(payload["mask_sources"]))
        except BaseException:
            shutil.rmtree(partial)
            raise
        if out.exists():
            shutil.rmtree(out)
        partial.rename(out)
    return _manifest(payload, out)


def _build(
    root: Path, ready: Mapping[str, Any], output: Path, request: Mapping[str, Any],
    masks_root: Path | None, validate_every: int,
) -> dict[str, Any]:  # fmt: skip
    """Write the layout, camera, splits and manifest into ``output``; return the manifest."""
    speedplus_dir = root / ready["extracted"]
    camera = geometry.load_camera(speedplus_dir / "camera.json")
    official = geometry.load_official_tango_keypoints(root / ready["tango_points"]["path"])
    points = geometry.expand_tango_keypoints_64(official)
    _write_layout(output, points, camera)
    inputs = _SplitInputs(
        speedplus_dir, output, masks_root.resolve() if masks_root else None,
        request["mask_mode"], camera, points, validate_every, tuple(ready["image_size"]),
    )  # fmt: skip
    results = [_convert_split(inputs, split, root) for split in request["splits_requested"]]
    sources: dict[str, int] = {}
    for result in results:
        for source, count in result["mask_sources"].items():
            sources[source] = sources.get(source, 0) + count
    payload = {
        "dataset": "speedplus",
        "speedplus_root": ready["extracted"],
        "archive_sha256": ready["archive"]["sha256"],
        "tango_points_sha256": ready["tango_points"]["sha256"],
        **request,
        "mask_sources": sources,
        "paper_masks": set(sources) == {PROJECTED_HULL},
        "keypoint_layout": LAYOUT_NAME,
        "keypoint_layout_sha256": layout_points_sha256(points),
        "splits": results,
    }
    _write_json(output / MANIFEST_NAME, payload)
    return payload
