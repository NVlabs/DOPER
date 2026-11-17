# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extraction of verified BOP ZIP parts and the ``data/READY.json`` coverage receipt.

The ZIP parts of a profile (models, PBR, validation and test splits) are extracted with Python
``zipfile`` into ``<root>/data``; split archives (``x.z01`` ... ``x.zip``) are read as one
archive (``doper.bop.hot3d_source.open_split_zip``). Member paths must be relative, without
``..``, links or duplicates, and the ZIP CRC is checked as each member is read. An archive that
nests its members under one directory (``meta.archive_prefix``: T-LESS ``tless_base.zip`` holds
``tless/...``) is extracted without it, so every part lands in the flat BOP layout, and a member
outside that directory is refused. A receipt per archive under ``<root>/extracted`` lets an
interrupted run resume. HOT3D's models and PBR use ``doper.bop.hot3d_source.extract_hot3d_source``
instead; its real-image clips are staged by ``doper.bop.hot3d_staging``.

``write_pbr_ready`` then records the extracted ``train_pbr`` coverage (frames and frames per
object) and checks it against the profile's pinned facts: the HOPEv2 PBR inventory (per-scene
GT hashes and object coverage) and the T-LESS frame count. Objects without PBR frames are
refused unless the caller acknowledges them.
"""

from __future__ import annotations

import json
import shutil
import stat
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from . import sources
from .hot3d_source import open_split_zip
from .sources import DatasetProfile, publish_json, sha256_file

DATA_DIR = "data"
RECEIPT_DIR = "extracted"
COPY_BLOCK_BYTES = 1024 * 1024
SCENE_METADATA = ("scene_gt.json", "scene_camera.json")


def data_dir(root: Path) -> Path:
    """Return the directory that receives the extracted dataset, ``<root>/data``."""
    return Path(root) / DATA_DIR


def zip_archives(profile: DatasetProfile, part: str) -> list[list[str]]:
    """Group a part's ZIP files into archives, each a list of volumes ending with ``.zip``.

    Files that are not ZIP volumes (HOT3D clip ``.tar`` files, JSON metadata) are left out.
    """
    names = [n for n in profile.parts[part] if n.endswith(".zip") or n[-4:-2] == ".z"]
    archives = []
    for name in (n for n in names if n.endswith(".zip")):
        stem = name[: -len(".zip")]
        volumes = sorted(n for n in names if n.startswith(stem + ".z") and n != name)
        archives.append([*volumes, name])
    return archives


def _safe_member(info: Any, prefix: str | None = None) -> PurePosixPath | None:  # noqa: ANN401
    """Return a file member's relative path, None for a directory; refuse unsafe members.

    With ``prefix``, every member must lie under that directory, which is stripped.
    """
    path = PurePosixPath(info.filename)
    if path.is_absolute() or ".." in path.parts or "\\" in info.filename or not path.parts:
        raise ValueError(f"unsafe archive path {info.filename!r}")
    if stat.S_ISLNK(info.external_attr >> 16):
        raise ValueError(f"archive link not allowed: {info.filename}")
    if prefix is not None:
        if path.parts[0] != prefix:
            raise ValueError(f"archive member {info.filename!r} is not under {prefix}/")
        if len(path.parts) == 1:
            return None
        path = PurePosixPath(*path.parts[1:])
    return None if info.is_dir() else path


def archive_prefix(profile: DatasetProfile, archive: str) -> str | None:
    """The directory an archive nests its members under (``meta.archive_prefix``), if any."""
    return profile.meta.get("archive_prefix", {}).get(archive)


def _extract_archive(root: Path, profile: DatasetProfile, volumes: Sequence[str]) -> int:
    """Extract one (possibly split) archive into ``root/data``; return its member count."""
    identity: dict[str, Any] = {
        "archive": list(volumes),
        "sha256": [profile.files[v] for v in volumes],
    }
    prefix = archive_prefix(profile, volumes[-1])
    if prefix is not None:
        identity["strip_prefix"] = prefix
    receipt = root / RECEIPT_DIR / f"{volumes[-1].replace('/', '_')}.json"
    if receipt.exists():
        if json.loads(receipt.read_text())["identity"] != identity:
            raise ValueError(f"extraction receipt of {volumes[-1]} belongs to another archive")
        return json.loads(receipt.read_text())["members"]
    target = data_dir(root)
    seen: set[str] = set()
    count = 0
    with open_split_zip([sources.source_dir(root) / v for v in volumes]) as archive:
        for info in archive.infolist():
            if info.filename in seen:
                raise ValueError(f"duplicate archive member {info.filename}")
            seen.add(info.filename)
            path = _safe_member(info, prefix)
            if path is None:
                continue
            destination = target / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, destination.open("wb") as sink:
                shutil.copyfileobj(source, sink, COPY_BLOCK_BYTES)
            count += 1
    publish_json(receipt, {"identity": identity, "members": count, "crc_checked": True})
    return count


def extract_parts(root: Path, profile: DatasetProfile, parts: Sequence[str]) -> Path:
    """Extract the ZIP archives of verified parts into ``root/data``.

    Args:
        root: Dataset root whose parts are recorded in ``READY.json``.
        profile: Dataset profile.
        parts: Part names to extract.

    Returns:
        The data directory ``root/data``.

    Raises:
        SourceNotReadyError: If a part was not fetched and verified.
        ValueError: For an unsafe, linked or duplicated member, or a receipt of another
            archive.
    """
    root = Path(root)
    ready = sources.require_ready(root, parts)
    if ready["profile_sha256"] != sources.profile_sha256(profile):
        raise ValueError("pinned download identity differs from the profile")
    with sources.file_lock(root / sources.LOCK_NAME):
        for part in parts:
            for volumes in zip_archives(profile, part):
                _extract_archive(root, profile, volumes)
    return data_dir(root)


def pbr_coverage(train_pbr: Path, objects: Sequence[int]) -> dict[str, Any]:
    """Frames, scenes, frames per object and scene GT hashes of a ``train_pbr`` directory.

    Raises:
        ValueError: If a scene's GT and camera frame sets differ.
    """
    counts = {str(o): 0 for o in objects}
    frames = 0
    hashes = {}
    for scene in sorted(p for p in Path(train_pbr).iterdir() if (p / "scene_gt.json").is_file()):
        gt, camera = (json.loads((scene / name).read_text()) for name in SCENE_METADATA)
        if set(gt) != set(camera):
            raise ValueError(f"{scene}: PBR camera/GT membership mismatch")
        for instances in gt.values():
            for obj in {g["obj_id"] for g in instances}:
                counts[str(obj)] = counts.get(str(obj), 0) + 1
        frames += len(gt)
        hashes[scene.name] = sha256_file(scene / "scene_gt.json")
    return {
        "frames": frames,
        "scenes": len(hashes),
        "object_frame_counts": counts,
        "scene_gt_sha256": hashes,
    }


def _check_inventory(profile: DatasetProfile, coverage: dict[str, Any]) -> None:
    """Compare the coverage with the pinned facts of the profile, where it has them."""
    if "pbr_inventory" in profile.meta.get("resources", {}):
        inventory = sources.load_resource(profile, "pbr_inventory")
        expected = {row["scene"]: row["gt_sha256"] for row in inventory["scenes"]}
        if coverage["scene_gt_sha256"] != expected:
            raise ValueError("PBR GT differs from the pinned archive inventory")
        if coverage["object_frame_counts"] != inventory["object_frame_counts"]:
            raise ValueError("PBR object coverage changed")
    frames = profile.meta.get("expected_frames", {}).get("train_pbr")
    frames = frames or profile.meta.get("counts", {}).get("pbr_frames")
    if frames is not None and coverage["frames"] != frames:
        raise ValueError(f"PBR has {coverage['frames']} frames; the profile pins {frames}")


def write_pbr_ready(
    root: Path, profile: DatasetProfile, *, allowed_missing: Sequence[int] = ()
) -> dict[str, Any]:
    """Record the extracted ``train_pbr`` coverage in ``root/data/READY.json``.

    Also writes ``train_pbr/exclusions.json`` with no excluded frames: datasets with documented
    defects (HANDAL) drop them in a filtered view (``sources.clean_source_view``) instead.

    Args:
        root: Dataset root with ``data/train_pbr`` extracted.
        profile: Dataset profile (``meta.objects`` lists the objects; default: those found).
        allowed_missing: Objects acknowledged to have no PBR frames.

    Returns:
        The receipt.

    Raises:
        ValueError: If the coverage differs from the profile's pinned facts, or an object
            other than ``allowed_missing`` has no PBR frame.
    """
    data = data_dir(root)
    objects = [int(o) for o in profile.meta.get("objects", [])]
    coverage = pbr_coverage(data / "train_pbr", objects)
    _check_inventory(profile, coverage)
    missing = sorted(int(o) for o, n in coverage["object_frame_counts"].items() if n <= 0)
    if missing != sorted(set(allowed_missing) & set(objects)):
        raise ValueError(f"PBR has no frames of objects {missing}; acknowledge them explicitly")
    publish_json(
        data / "train_pbr" / "exclusions.json",
        {
            "dataset": profile.name,
            "excluded_frames": [],
            "zero_byte_required_files": [],
            "missing_required_files": [],
        },
    )
    receipt = {
        "identity": {"dataset": profile.name, "profile_sha256": sources.profile_sha256(profile)},
        "dataset": profile.name,
        "revision": profile.revision,
        "coverage": {**coverage, "missing_objects": missing},
        "model_units": "mm",
        "pose_translation_units": "mm",
        "keypoint_units": "m",
        "test_used_for_training": False,
    }
    publish_json(data / sources.READY_NAME, receipt)
    return receipt
