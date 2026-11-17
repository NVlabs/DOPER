# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Original T-LESS TRAIN captures of the Kinect and Canon sensors: fetch, verify, receipts.

The BOP release of T-LESS carries only the PrimeSense TRAIN images. The Kinect and Canon TRAIN
captures of the same turntable sessions come from the original v2 publication, one ZIP per
sensor and object (``meta.sensors.url_template`` of the ``tless`` profile), served over plain
HTTP; no credentials are sent. :func:`fetch` downloads one archive, checks it, extracts it and
writes a receipt::

    <root>/sensors/archives/t-less_v2_train_<sensor>_<obj:02d>.zip
    <root>/sensors/receipts/<sensor>-<obj:06d>.json
    <root>/sensors/train_<sensor>/<obj:02d>/{gt.yml, info.yml, rgb/, depth/}

Upstream publishes no checksum. The profile pins the SHA-256 observed for object 1; any other
archive is trusted on first use: its SHA-256 is recorded in the receipt with
``checksum_origin: observed`` and a later fetch fails if the archive differs from it.

Checks: ZIP members must be relative, without ``..``, backslashes, links or duplicates, under a
top directory named after the object, and only ``gt.yml``, ``info.yml``, ``rgb/*`` and
``depth/*`` images; the ZIP CRC is checked as each member is read. Every object has 1296
captures (648 for objects 19 and 20), each with exactly one instance of the object, and the RGB,
GT and per-frame camera keys must be equal.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import urllib.request
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .sources import (
    IMAGE_SUFFIXES,
    DatasetProfile,
    file_lock,
    load_profile,
    publish_json,
    sha256_file,
)

SENSORS_DIR = "sensors"
ARCHIVES_DIR = "archives"
RECEIPTS_DIR = "receipts"
LOCK_NAME = "stage.lock"
SENSOR_NAMES = ("kinect", "canon")
METADATA_FILES = ("gt.yml", "info.yml")
IMAGE_DIRS = ("rgb", "depth")
COPY_BLOCK_BYTES = 1024 * 1024
HTTP_TIMEOUT_S = 120
USER_AGENT = "doper-tless-sensors"
HttpGet = Callable[[str, Path], None]
"""Downloads a URL to a file: ``download(url, destination)``."""


@dataclass(frozen=True)
class SensorReceipt:
    """A fetched, verified and extracted sensor archive.

    Attributes:
        sensor: ``kinect`` or ``canon``.
        obj_id: T-LESS object id.
        archive_sha256: SHA-256 of the full ZIP.
        checksum_origin: ``pinned`` (the profile pins it) or ``observed`` (trust on first use).
        frames: Number of captures.
        directory: Extracted object directory.
        receipt: The receipt file.
    """

    sensor: str
    obj_id: int
    archive_sha256: str
    checksum_origin: str
    frames: int
    directory: Path
    receipt: Path


def sensor_spec(profile: DatasetProfile) -> Mapping[str, Any]:
    """The ``meta.sensors`` table of a profile.

    Raises:
        ValueError: If the profile has none (only T-LESS has the extra sensors).
    """
    spec = profile.meta.get("sensors")
    if not spec:
        raise ValueError(f"profile {profile.name} has no additional TRAIN sensors")
    return spec


def expected_frames(obj_id: int, profile: DatasetProfile | None = None) -> int:
    """The number of TRAIN captures of an object (648 for objects 19 and 20, else 1296)."""
    spec = sensor_spec(profile or load_profile("tless"))
    return int(spec["frames_by_object"].get(str(obj_id), spec["frames"]))


def _check_request(obj_id: int, sensor: str, profile: DatasetProfile) -> Mapping[str, Any]:
    spec = sensor_spec(profile)
    if sensor not in spec["names"]:
        raise ValueError(f"unknown T-LESS sensor {sensor!r}; expected one of {spec['names']}")
    if obj_id not in profile.meta.get("objects", []):
        raise ValueError(f"T-LESS object id must be one of 1..30, got {obj_id}")
    return spec


def archive_name(obj_id: int, sensor: str) -> str:
    """File name of the original archive, ``t-less_v2_train_<sensor>_<obj:02d>.zip``."""
    return f"t-less_v2_train_{sensor}_{obj_id:02d}.zip"


def archive_path(root: Path, obj_id: int, sensor: str) -> Path:
    """Where the downloaded archive is kept."""
    return Path(root) / SENSORS_DIR / ARCHIVES_DIR / archive_name(obj_id, sensor)


def receipt_path(root: Path, obj_id: int, sensor: str) -> Path:
    """The receipt of one fetched archive."""
    return Path(root) / SENSORS_DIR / RECEIPTS_DIR / f"{sensor}-{obj_id:06d}.json"


def sensor_dir(root: Path, sensor: str) -> Path:
    """Directory holding a sensor's extracted objects, ``<root>/sensors/train_<sensor>``."""
    return Path(root) / SENSORS_DIR / f"train_{sensor}"


def object_dir(root: Path, obj_id: int, sensor: str) -> Path:
    """Extracted object directory, ``<root>/sensors/train_<sensor>/<obj:02d>``."""
    return sensor_dir(root, sensor) / f"{obj_id:02d}"


def http_download(url: str, destination: Path, max_bytes: int | None = None) -> None:
    """Download ``url`` to ``destination`` with the standard library, refusing a short body.

    Args:
        url: URL to read.
        destination: File to write.
        max_bytes: Refuse a body larger than this.

    Raises:
        ValueError: If the body exceeds ``max_bytes`` or is shorter than its Content-Length.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    size = 0
    with (
        urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response,
        destination.open("wb") as out,
    ):
        expected = int(response.headers.get("Content-Length") or 0)
        for block in iter(lambda: response.read(COPY_BLOCK_BYTES), b""):
            out.write(block)
            size += len(block)
            if max_bytes is not None and size > max_bytes:
                raise ValueError(f"{url}: archive larger than {max_bytes} bytes")
    if expected and size != expected:
        raise ValueError(f"{url}: truncated download ({size} of {expected} bytes)")


def read_metadata(folder: Path, bop_name: str, original_name: str) -> tuple[dict[int, Any], Path]:
    """Read per-frame metadata in BOP JSON or the original v2 YAML form.

    Exactly one of ``folder/bop_name`` (``scene_gt.json``, ``scene_camera.json``) and
    ``folder/original_name`` (``gt.yml``, ``info.yml``) must exist.

    Args:
        folder: Object directory.
        bop_name: BOP JSON file name.
        original_name: Original YAML file name.

    Returns:
        The frames keyed by integer id, and the file read.

    Raises:
        ValueError: If neither or both files exist, or two keys name the same frame.
    """
    import yaml

    candidates = [p for p in (folder / bop_name, folder / original_name) if p.is_file()]
    if len(candidates) != 1:
        raise ValueError(f"exactly one of {bop_name}/{original_name} required in {folder}")
    path = candidates[0]
    text = path.read_text()
    data = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    result = {int(key): value for key, value in data.items()}
    if len(result) != len(data):
        raise ValueError(f"{path}: duplicate numeric frame ids")
    return result, path


def _download(url: str, archive: Path, downloader: HttpGet | None, spec: Mapping[str, Any]) -> None:
    partial = archive.with_name(archive.name + ".part")
    partial.unlink(missing_ok=True)
    if downloader is None:
        http_download(url, partial, spec.get("max_archive_bytes"))
    else:
        downloader(url, partial)
    os.replace(partial, archive)


def _checksum_origin(
    archive: Path, digest: str, pinned: str | None, previous: Mapping[str, Any] | None
) -> str:
    """``pinned`` or ``observed``; raise if the archive differs from the pin or the receipt."""
    if pinned is not None and digest != pinned:
        archive.unlink()
        raise ValueError(f"{archive.name}: SHA-256 {digest} differs from the pinned {pinned}")
    if previous is not None and digest != previous["archive_sha256"]:
        raise ValueError(
            f"{archive.name}: SHA-256 {digest} differs from the {previous['checksum_origin']} "
            f"{previous['archive_sha256']} of its receipt; the staged source changed"
        )
    return "pinned" if pinned is not None else "observed"


def _member_path(info: zipfile.ZipInfo, obj_id: int) -> PurePosixPath | None:
    """The relative path of an allowed file member; None for a directory."""
    path = PurePosixPath(info.filename)
    if (
        not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in info.filename
        or path.parts[0] != f"{obj_id:02d}"
        or stat.S_ISLNK(info.external_attr >> 16)
    ):
        raise ValueError(f"unsafe original ZIP member {info.filename!r}")
    if info.is_dir():
        return None
    allowed = (len(path.parts) == 2 and path.name in METADATA_FILES) or (
        len(path.parts) == 3
        and path.parts[1] in IMAGE_DIRS
        and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not allowed:
        raise ValueError(f"unexpected original TRAIN member {info.filename!r}")
    return path


def _extract(archive: Path, obj_id: int, temporary: Path) -> tuple[list[dict[str, Any]], set[int]]:
    """Extract the allowed members into ``temporary``; return the file list and RGB frames."""
    files, rgb, seen = [], [], set()
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            path = _member_path(info, obj_id)
            if info.filename in seen:
                raise ValueError(f"duplicate ZIP member {info.filename!r}")
            seen.add(info.filename)
            if path is None:
                continue
            destination = temporary.joinpath(*path.parts[1:])
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, destination.open("wb") as sink:
                shutil.copyfileobj(source, sink, COPY_BLOCK_BYTES)
            files.append({"path": path.as_posix(), "bytes": info.file_size, "crc32": info.CRC})
            if path.parts[1] == "rgb":
                rgb.append(int(path.stem))
    if len(rgb) != len(set(rgb)):
        raise ValueError(f"{archive.name}: ambiguous RGB frame names")
    return files, set(rgb)


def _check_frames(folder: Path, obj_id: int, rgb: set[int], expected: int) -> tuple[Path, Path]:
    """Frame coverage and isolation checks; return the GT and camera files."""
    gt, gt_path = read_metadata(folder, "scene_gt.json", "gt.yml")
    cameras, camera_path = read_metadata(folder, "scene_camera.json", "info.yml")
    if len(rgb) != expected or set(gt) != rgb or set(cameras) != rgb:
        raise ValueError(
            f"object {obj_id}: expected {expected} TRAIN captures with equal RGB/GT/camera keys; "
            f"found {len(rgb)} RGB, {len(gt)} GT, {len(cameras)} camera frames"
        )
    for frame, instances in gt.items():
        if len(instances) != 1 or instances[0]["obj_id"] != obj_id:
            raise ValueError(f"object {obj_id} frame {frame}: TRAIN needs one instance of it")
    return gt_path, camera_path


def fetch(
    root: Path,
    obj_id: int,
    sensor: str,
    *,
    downloader: HttpGet | None = None,
    profile: DatasetProfile | None = None,
) -> SensorReceipt:
    """Download, verify and extract one original T-LESS TRAIN archive.

    An object with a receipt is only re-verified: the kept archive must still have the recorded
    SHA-256 and the extracted directory must exist.

    Args:
        root: Dataset root; the archive, receipt and extraction go under ``root/sensors``.
        obj_id: T-LESS object id (1..30).
        sensor: ``kinect`` or ``canon``.
        downloader: ``download(url, destination)``; defaults to :func:`http_download`.
        profile: Profile with the sensor table (default: the packaged ``tless``).

    Returns:
        The receipt.

    Raises:
        ValueError: For an unknown sensor or object, an archive that differs from its pin or
            its receipt, an unsafe or unexpected member, or a frame coverage or isolation
            failure.
        FileExistsError: For an extraction without a receipt (inspect it before retrying).
    """
    profile = profile or load_profile("tless")
    spec = _check_request(obj_id, sensor, profile)
    root = Path(root)
    archive, receipt = archive_path(root, obj_id, sensor), receipt_path(root, obj_id, sensor)
    final = object_dir(root, obj_id, sensor)
    url = spec["url_template"].format(sensor=sensor, obj=obj_id)
    pinned = spec.get("sha256", {}).get(sensor, {}).get(str(obj_id))
    archive.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(root / SENSORS_DIR / LOCK_NAME):
        previous = json.loads(receipt.read_text()) if receipt.is_file() else None
        if previous is None and final.exists():
            raise FileExistsError(f"{final} has no receipt; inspect it before retrying")
        if not archive.exists():
            if previous is not None:
                raise ValueError(f"{archive} of a receipted extraction is missing")
            _download(url, archive, downloader, spec)
        digest = sha256_file(archive)
        origin = _checksum_origin(archive, digest, pinned, previous)
        if previous is not None:
            if not final.is_dir():
                raise ValueError(f"{final} of a receipted archive is missing")
            return _result(previous, root, receipt)
        try:
            record = _stage(root, obj_id, sensor, archive, profile)
        except (ValueError, zipfile.BadZipFile) as err:
            raise ValueError(
                f"{err}; {archive} has no receipt and would be trusted again as {origin}: "
                "if the download may be truncated or corrupt, delete it so it is re-fetched"
            ) from err
        record.update(url=url, archive_sha256=digest, checksum_origin=origin)
        record["transport"] = spec.get("transport", "HTTP")
        publish_json(receipt, record)
    return _result(record, root, receipt)


def _stage(
    root: Path, obj_id: int, sensor: str, archive: Path, profile: DatasetProfile
) -> dict[str, Any]:
    """Extract into a temporary directory, check it, and rename it into place."""
    target = sensor_dir(root, sensor)
    target.mkdir(parents=True, exist_ok=True)
    temporary = target / f".obj{obj_id:02d}-extract"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        files, rgb = _extract(archive, obj_id, temporary)
        gt_path, camera_path = _check_frames(
            temporary, obj_id, rgb, expected_frames(obj_id, profile)
        )
        record = {
            "dataset": profile.name,
            "sensor": sensor,
            "object": obj_id,
            "archive": Path(os.path.relpath(archive, root)).as_posix(),
            "archive_bytes": archive.stat().st_size,
            "member_crc_verified": True,
            "frames": len(rgb),
            "gt_sha256": sha256_file(gt_path),
            "camera_sha256": sha256_file(camera_path),
            "directory": Path(os.path.relpath(object_dir(root, obj_id, sensor), root)).as_posix(),
            "files": files,
        }
    except Exception:
        shutil.rmtree(temporary)
        raise
    temporary.rename(object_dir(root, obj_id, sensor))
    return record


def _result(record: Mapping[str, Any], root: Path, receipt: Path) -> SensorReceipt:
    return SensorReceipt(
        sensor=record["sensor"],
        obj_id=int(record["object"]),
        archive_sha256=record["archive_sha256"],
        checksum_origin=record["checksum_origin"],
        frames=int(record["frames"]),
        directory=root / record["directory"],
        receipt=receipt,
    )
