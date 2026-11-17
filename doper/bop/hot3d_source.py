# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the HOT3D PBR source: extraction, unit canonicalization and defect filters.

:func:`extract_hot3d_source` extracts the pinned archives, :func:`canonicalize_units` makes the
PBR metadata millimeters, and :func:`filter_source_defects` and :func:`filter_object_masks` drop
the frames with evidenced unusable pixels, each in a separate metadata view that links the
original pixels. The defect policy is profile data (:mod:`doper.bop.sources`).
"""

from __future__ import annotations

import concurrent.futures
import copy
import io
import json
import logging
import os
import shutil
import stat
import struct
import threading
import zipfile
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np

from doper.bop import sources
from doper.bop.audit import load_bop_mesh
from doper.bop.hot3d import metadata_m_to_mm
from doper.bop.sources import IMAGE_SUFFIXES, publish_json, sha256_file

LOGGER = logging.getLogger(__name__)
MM_PER_M = 1000
RUN_LOCK = "run.lock"
MAX_EXTRACT_WORKERS = 16
MODEL_EXTENT_TOLERANCE_MM = 0.1
# Behavior versions of the three view builders. A view's identity records the SHA-256 of its
# version (the source hashed its script bytes), so an unrelated change to this module does not
# invalidate views that are tens of gigabytes. Bump a version only when that builder's output
# changes; existing views of the old version are then refused until they are deleted.
CANONICALIZER_VERSION = "hot3d-canonicalize-units-v1"
DEFECT_FILTER_VERSION = "hot3d-filter-source-defects-v1"
MASK_FILTER_VERSION = "hot3d-filter-object-masks-v1"
PIXEL_DIRECTORIES = ("rgb", "depth", "mask_visib")
SCENE_METADATA_FILES = ("scene_gt.json", "scene_camera.json", "scene_gt_info.json")
HOT3D_MODEL_BANKS = ("models", "models_eval")


def _hot3d_profile(profile: sources.DatasetProfile | None) -> sources.DatasetProfile:
    """Return ``profile`` or the packaged HOT3D profile."""
    return profile if profile is not None else sources.load_profile("hot3d")


def _version_sha256(version: str) -> str:
    """Identity hash of a view builder's behavior version."""
    return sources.canonical_sha256({"builder": version})


def _identity_changed(kind: str, destination: Path) -> ValueError:
    """The error for an existing view whose identity differs from the requested one."""
    return ValueError(
        f"{kind} identity changed: {destination} was built from other data or by another "
        "builder version; delete it and rerun `doper prepare` to rebuild it"
    )


def _read_json(path: Path) -> Any:  # noqa: ANN401
    return json.loads(Path(path).read_text())


def _link_pixels(original: Path, target: Path, names: Sequence[str]) -> None:
    """Link ``names`` of ``original`` into ``target`` read-only, skipping missing ones."""
    for name in names:
        path, link = original / name, target / name
        if path.exists() and not os.path.lexists(link):
            sources.link_relative(link, path)


def _link_models(stage: Path, raw: Path) -> None:
    """Link the model banks of ``raw`` into ``stage``."""
    for kind in HOT3D_MODEL_BANKS:
        if not os.path.lexists(stage / kind):
            sources.link_relative(stage / kind, raw / kind)


def _expect_pinned_ready(ready: Mapping[str, Any], profile: sources.DatasetProfile) -> None:
    """Require a prepared HOT3D source of exactly this profile."""
    identity = ready.get("identity", {})
    if (
        ready.get("dataset") != "hot3d"
        or ready.get("revision") != profile.revision
        or identity.get("profile_sha256") != sources.profile_sha256(profile)
    ):
        raise ValueError("unit policy applies only to the pinned official HOT3D source")


def _safe_member(info: zipfile.ZipInfo, roots: Sequence[str]) -> PurePosixPath:
    """Return the member path, refusing absolute paths, ``..``, unexpected roots and symlinks."""
    path = PurePosixPath(info.filename)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] not in roots:
        raise ValueError("unsafe/unexpected source archive path")
    if stat.S_ISLNK(info.external_attr >> 16):
        raise ValueError("archive symlink forbidden")
    return path


def _required_pbr_member(info: zipfile.ZipInfo) -> bool:
    """Return whether a ``train_pbr`` archive member is one the conversion needs."""
    path = _safe_member(info, ("train_pbr",))
    if info.is_dir():
        return False
    if len(path.parts) < 3 or len(path.parts[1]) != 6 or not path.parts[1].isdigit():
        raise ValueError("unexpected PBR member hierarchy")
    if len(path.parts) == 3 and path.name in SCENE_METADATA_FILES:
        return True
    return (
        len(path.parts) == 4
        and path.parts[2] in PIXEL_DIRECTORIES
        and path.suffix.lower() in IMAGE_SUFFIXES
    )


class _ConcatenatedFile(io.RawIOBase):
    """Read-only view of several files as one contiguous stream (split ZIP volumes)."""

    def __init__(self, paths: Sequence[Path]) -> None:
        self._files = [path.open("rb") for path in paths]
        self.sizes = [os.fstat(f.fileno()).st_size for f in self._files]
        self.starts = [sum(self.sizes[:i]) for i in range(len(self.sizes))]
        self._total = sum(self.sizes)
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        base = {os.SEEK_SET: 0, os.SEEK_CUR: self._position, os.SEEK_END: self._total}[whence]
        self._position = max(0, base + offset)
        return self._position

    def readinto(self, buffer: Any) -> int:  # noqa: ANN401
        view = memoryview(buffer).cast("B")
        done = 0
        while done < len(view) and self._position < self._total:
            index = max(i for i, start in enumerate(self.starts) if start <= self._position)
            handle = self._files[index]
            handle.seek(self._position - self.starts[index])
            room = self.sizes[index] - (self._position - self.starts[index])
            chunk = handle.read(min(len(view) - done, room))
            if not chunk:
                break
            view[done : done + len(chunk)] = chunk
            done += len(chunk)
            self._position += len(chunk)
        return done

    def close(self) -> None:
        for handle in self._files:
            handle.close()
        super().close()


EOCD_SIGNATURE = b"PK\x05\x06"
EOCD_STRUCT = struct.Struct("<4s4H2LH")
ZIP64_LOCATOR_STRUCT = struct.Struct("<4sLQL")
ZIP64_EOCD_STRUCT = struct.Struct("<4sQ2H2L4Q")
CENTRAL_STRUCT = struct.Struct("<4s4B4HL2L5H2L")
CENTRAL_SIGNATURE = b"PK\x01\x02"
MAX_COMMENT_BYTES = 65535
MAX_16 = 0xFFFF
MAX_32 = 0xFFFFFFFF
ZIP64_EXTRA_ID = 1


class _SplitZipFile(zipfile.ZipFile):
    """A ``ZipFile`` whose central directory may span several volumes (``zip -s``).

    ``zipfile`` refuses multi-disk archives, and for plain archives it adds one shift to every
    entry. Here each entry's offset (relative to the volume named by its disk-start field,
    including the zip64 extra field) becomes an absolute offset in the concatenated stream.
    """

    def _RealGetContents(self) -> None:  # noqa: N802
        stream = cast(_ConcatenatedFile, self.fp)
        starts = stream.starts
        size_cd, self.start_dir = self._read_end_records(stream, starts)
        stream.seek(self.start_dir)
        data = io.BytesIO(stream.read(size_cd))
        while data.tell() < size_cd:
            info = self._read_entry(data, starts)
            self.filelist.append(info)
            self.NameToInfo[info.filename] = info
        end = self.start_dir
        for info in sorted(self.filelist, key=lambda i: i.header_offset, reverse=True):
            info._end_offset = end  # type: ignore[attr-defined]  # noqa: SLF001
            end = info.header_offset

    def _read_end_records(self, stream: _ConcatenatedFile, starts: list[int]) -> tuple[int, int]:
        """Return ``(central directory size, absolute start)`` from the (zip64) end records."""
        stream.seek(0, os.SEEK_END)
        total = stream.tell()
        tail_start = max(0, total - EOCD_STRUCT.size - MAX_COMMENT_BYTES)
        stream.seek(tail_start)
        tail = stream.read()
        at = tail.rfind(EOCD_SIGNATURE)
        if at < 0 or len(tail) - at < EOCD_STRUCT.size:
            raise zipfile.BadZipFile("File is not a zip file")
        *_, size_cd, _, comment_len = EOCD_STRUCT.unpack_from(tail, at)
        self._comment = tail[at + EOCD_STRUCT.size : at + EOCD_STRUCT.size + comment_len]
        locator_at = at - ZIP64_LOCATOR_STRUCT.size
        if locator_at < 0 or tail[locator_at : locator_at + 4] != b"PK\x06\x07":
            # Without zip64 records the central directory ends where the end record begins;
            # some writers leave a placeholder offset in a split archive's end record.
            return size_cd, tail_start + at - size_cd
        _, record_disk, record_offset, disks = ZIP64_LOCATOR_STRUCT.unpack_from(tail, locator_at)
        if record_disk >= len(starts) or disks != len(starts):
            raise zipfile.BadZipFile("zip64 locator disagrees with the volumes present")
        stream.seek(starts[record_disk] + record_offset)
        fields = ZIP64_EOCD_STRUCT.unpack(stream.read(ZIP64_EOCD_STRUCT.size))
        if fields[0] != b"PK\x06\x06":
            raise zipfile.BadZipFile("Zip64 end of central directory record not found")
        disk_cd, size_cd, offset_cd = fields[5], fields[8], fields[9]
        if disk_cd >= len(starts):
            raise zipfile.BadZipFile("central directory starts on a missing volume")
        return size_cd, starts[disk_cd] + offset_cd

    def _read_entry(self, data: io.BytesIO, starts: list[int]) -> zipfile.ZipInfo:
        """Parse one central directory entry into a ``ZipInfo`` with an absolute offset."""
        raw = data.read(CENTRAL_STRUCT.size)
        if len(raw) != CENTRAL_STRUCT.size:
            raise zipfile.BadZipFile("Truncated central directory")
        centdir = CENTRAL_STRUCT.unpack(raw)
        if centdir[0] != CENTRAL_SIGNATURE:
            raise zipfile.BadZipFile("Bad magic number for central directory")
        name_length, extra_length, comment_length = centdir[12:15]
        name = data.read(name_length)
        flags = centdir[5]
        encoding = (
            "utf-8" if flags & 0x800 else (getattr(self, "metadata_encoding", None) or "cp437")
        )
        info = zipfile.ZipInfo(name.decode(encoding))
        info.extra = data.read(extra_length)
        info.comment = data.read(comment_length)
        (info.create_version, info.create_system, info.extract_version, info.reserved) = centdir[
            1:5
        ]
        info.flag_bits = flags
        (info.compress_type, time, date, info.CRC) = centdir[6:10]
        info.compress_size, info.file_size = centdir[10:12]
        info.volume, info.internal_attr, info.external_attr = centdir[15:18]
        info.header_offset = centdir[18]
        info._raw_time = time  # noqa: SLF001
        info.date_time = (
            (date >> 9) + 1980, (date >> 5) & 0xF, date & 0x1F,
            time >> 11, (time >> 5) & 0x3F, (time & 0x1F) * 2,
        )  # fmt: skip
        volume = _zip64_disk_start(info)
        decode_extra: Any = info._decodeExtra  # type: ignore[attr-defined]  # noqa: SLF001
        try:
            decode_extra(zlib.crc32(name))
        except TypeError:  # Python < 3.12 takes no filename checksum
            decode_extra()
        if volume is not None:
            info.volume = volume
        if info.volume >= len(starts):
            raise zipfile.BadZipFile(f"{info.filename} starts on a missing volume")
        info.header_offset += starts[info.volume]
        return info


def _zip64_disk_start(info: zipfile.ZipInfo) -> int | None:
    """Return the disk-start of an entry that stores it in the zip64 extra field, if any."""
    if info.volume != MAX_16:
        return None
    extra = info.extra
    position = 0
    while position + 4 <= len(extra):
        tag, length = struct.unpack_from("<HH", extra, position)
        if tag == ZIP64_EXTRA_ID:
            wide = sum(8 for value in (info.file_size, info.compress_size) if value == MAX_32)
            wide += 8 if info.header_offset == MAX_32 else 0
            return struct.unpack_from("<L", extra, position + 4 + wide)[0]
        position += 4 + length
    return None


def open_split_zip(paths: Sequence[Path]) -> zipfile.ZipFile:
    """Open a (possibly split, possibly zip64) ZIP archive with Python's ``zipfile``.

    Args:
        paths: Archive volumes in order (``x.z01``, ..., ``x.zip``) or one ``.zip``.

    Returns:
        An open archive; close it when done.
    """
    return _SplitZipFile(_ConcatenatedFile(paths))


def _extract_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, path: Path) -> None:
    """Extract one member; the ZIP CRC is checked on the complete read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(info) as source, path.open("wb") as target:
        shutil.copyfileobj(source, target, 1024 * 1024)


def _scene_metadata(scene: Path, objects: Sequence[int]) -> dict[str, Any]:
    """Validate one extracted scene and return its frame and object-frame counts."""
    gt = _read_json(scene / "scene_gt.json")
    cameras = _read_json(scene / "scene_camera.json")
    if not gt or set(gt) != set(cameras):
        raise ValueError("camera/GT frame population differs")
    counts = {str(obj): 0 for obj in objects}
    depths: list[float] = []
    for key, instances in gt.items():
        k = np.asarray(cameras[key]["cam_K"], dtype=float).reshape(3, 3)
        if not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
            raise ValueError("invalid PBR camera")
        for obj in {int(g["obj_id"]) for g in instances}:
            if str(obj) not in counts:
                raise ValueError("unexpected PBR object ID")
            counts[str(obj)] += 1
        for g in instances:
            translation = np.asarray(g["cam_t_m2c"], dtype=float)
            if translation.shape != (3,) or not np.isfinite(translation).all():
                raise ValueError("invalid PBR translation")
            depths.append(float(translation[2]))
    return {
        "frames": len(gt),
        "object_frame_counts": counts,
        "scene_gt_sha256": sha256_file(scene / "scene_gt.json"),
        "scene_camera_sha256": sha256_file(scene / "scene_camera.json"),
        "raw_translation_z_min_median_max": [
            float(np.min(depths)),
            float(np.median(depths)),
            float(np.max(depths)),
        ],
    }


def _canonical_models(
    archive: Path, stage: Path, marker: Path, expected_sha: str, objects: Sequence[int]
) -> dict[str, Any]:
    """Extract the model banks and write millimeter ``models_info.json`` files."""
    if marker.exists():
        receipt = _read_json(marker)
        if receipt["archive_sha256"] != expected_sha:
            raise ValueError("model archive identity changed")
        for kind, digest in receipt["canonical_info_sha256"].items():
            if sha256_file(stage / kind / "models_info.json") != digest:
                raise ValueError("canonical metadata changed")
        return receipt
    native: dict[str, Any] = {}
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            path = _safe_member(info, HOT3D_MODEL_BANKS)
            if info.is_dir():
                continue
            if path.name == "models_info.json":
                content = z.read(info)
                native[path.parts[0]] = json.loads(content)
                native_path = stage / path.parts[0] / "models_info_native_m.json"
                native_path.parent.mkdir(parents=True, exist_ok=True)
                native_path.write_bytes(content)
            else:
                _extract_member(z, info, stage / str(path))
    if set(native) != set(HOT3D_MODEL_BANKS):
        raise ValueError("missing model metadata bank")
    for kind, info_by_object in native.items():
        if set(map(int, info_by_object)) != set(objects):
            raise ValueError("wrong object bank")
        publish_json(stage / kind / "models_info.json", metadata_m_to_mm(info_by_object))
    rows = []
    for obj in objects:
        record = native["models"][str(obj)]
        mesh = load_bop_mesh(stage / "models" / f"obj_{obj:06d}.ply")
        expected = np.array([record["size_" + axis] for axis in "xyz"]) * MM_PER_M
        error = float(np.max(np.abs(mesh.extents - expected)))
        if error > MODEL_EXTENT_TOLERANCE_MM:
            raise ValueError("official PLY/native metadata unit relation differs")
        if not (stage / "models_eval" / f"obj_{obj:06d}.ply").is_file():
            raise ValueError("missing evaluation mesh")
        rows.append(
            {"object": obj, "mesh_extent_mm": mesh.extents.tolist(), "bbox_residual_mm": error}
        )
    receipt = {
        "archive_sha256": expected_sha,
        "PLY_vertices_units": "mm; unchanged",
        "input_metadata_length_units": "metres",
        "canonical_metadata_length_units": "millimetres",
        "symmetry_translations_and_offsets_scaled": True,
        "mesh_recentered_or_rescaled": False,
        "canonical_info_sha256": {
            kind: sha256_file(stage / kind / "models_info.json") for kind in native
        },
        "rows": rows,
    }
    publish_json(marker, receipt)
    return receipt


class _ThreadArchives:
    """One open archive per worker thread, all closed together."""

    def __init__(self, volumes: Sequence[Path]) -> None:
        self._volumes = list(volumes)
        self._local = threading.local()
        self._opened: list[zipfile.ZipFile] = []
        self._lock = threading.Lock()

    def get(self) -> zipfile.ZipFile:
        """Return this thread's archive, opening it on first use."""
        if not hasattr(self._local, "archive"):
            archive = open_split_zip(self._volumes)
            with self._lock:
                self._opened.append(archive)
            self._local.archive = archive
        return self._local.archive

    def close(self) -> None:
        """Close every archive opened so far."""
        for archive in self._opened:
            archive.close()


def _extract_scene(
    archives: _ThreadArchives,
    scene: str,
    members: Sequence[str],
    pbr: Path,
    objects: Sequence[int],
    parts: Mapping[str, str],
) -> dict[str, Any]:
    """Extract one scene into a private directory and publish it atomically."""
    destination = pbr / scene
    if (destination / "EXTRACTED.json").exists():
        receipt = _read_json(destination / "EXTRACTED.json")
        if receipt["parts"] != parts:
            raise ValueError("completed scene identity changed")
        LOGGER.info("PBR scene %s reused", scene)
        return receipt["metadata"]
    if destination.exists():
        raise ValueError("unreceipted published scene")
    archive = archives.get()
    temporary = pbr / f".extract-{scene}"
    temporary.mkdir(exist_ok=True)
    for name in members:
        _extract_member(
            archive, archive.getinfo(name), temporary.joinpath(*PurePosixPath(name).parts[2:])
        )
    metadata = _scene_metadata(temporary, objects)
    publish_json(
        temporary / "EXTRACTED.json",
        {
            "parts": parts,
            "metadata": metadata,
            "extracted_members": len(members),
            "CRC_checked_while_extracting": True,
            "amodal_mask_directory_not_extracted": True,
        },
    )
    os.rename(temporary, destination)
    LOGGER.info("PBR scene %s complete: %d frames", scene, metadata["frames"])
    return metadata


def extract_hot3d_source(
    root: Path,
    profile: sources.DatasetProfile | None = None,
    *,
    extract_workers: int = 4,
    allowed_missing: Sequence[int] = (),
) -> Path:
    """Extract the verified HOT3D PBR and model archives into ``root/data``.

    One pass, with CRC and path validation only as the needed members are extracted and
    atomic per-scene receipts so an interrupted run resumes. No full raw-tree audit.

    Args:
        root: Dataset root whose ``models`` and ``pbr`` parts are recorded in ``READY.json``.
        profile: HOT3D profile (default: the packaged one).
        extract_workers: Scene extraction threads.
        allowed_missing: Objects acknowledged to have no PBR coverage.

    Returns:
        The published ``root/data`` directory.

    Raises:
        ValueError: On a changed identity, unsafe archive member, invalid scene metadata or
            unacknowledged missing object coverage.
        SourceNotReadyError: If the parts were not fetched and verified.
    """
    profile = _hot3d_profile(profile)
    root = Path(root)
    objects = [int(o) for o in profile.meta["objects"]]
    if not 1 <= extract_workers <= MAX_EXTRACT_WORKERS:
        raise ValueError("invalid extraction concurrency")
    if any(obj not in objects for obj in allowed_missing):
        raise ValueError("invalid acknowledged coverage gap")
    download = sources.require_ready(root, ["models", "pbr"])
    if download["profile_sha256"] != sources.profile_sha256(profile):
        raise ValueError("pinned download identity differs")
    identity = {
        "dataset": "hot3d",
        "profile_sha256": sources.profile_sha256(profile),
        "download_receipt_sha256": sha256_file(root / sources.READY_NAME),
        "model_metadata_conversion": (
            "native lengths/symmetry translations m to mm; vertices unchanged"
        ),
        "acknowledged_uncovered_objects": sorted(set(allowed_missing)),
    }
    final = root / "data"
    with sources.file_lock(root / RUN_LOCK):
        if (final / sources.READY_NAME).exists():
            if _read_json(final / sources.READY_NAME)["identity"] != identity:
                raise ValueError("published source identity changed")
            return final
        _publish_hot3d(root, profile, objects, download, identity, extract_workers, allowed_missing)
    return final


def _publish_hot3d(
    root: Path,
    profile: sources.DatasetProfile,
    objects: Sequence[int],
    download: Mapping[str, Any],
    identity: dict[str, Any],
    extract_workers: int,
    allowed_missing: Sequence[int],
) -> None:
    """Stage models and every PBR scene, then publish ``root/data`` with its receipt."""
    stage = root / ".staging_data"
    stage.mkdir(exist_ok=True)
    files = sources.source_dir(root)
    model_archive = profile.parts["models"][0]
    pbr_files = profile.parts["pbr"]
    models = _canonical_models(
        files / model_archive,
        stage,
        root / "models_prepared.json",
        profile.files[model_archive],
        objects,
    )
    parts = {name: profile.files[name] for name in pbr_files}
    volumes = [files / name for name in pbr_files]
    pbr = stage / "train_pbr"
    pbr.mkdir(exist_ok=True)
    with open_split_zip(volumes) as listing:
        groups: dict[str, list[str]] = {}
        for info in listing.infolist():
            if _required_pbr_member(info):
                groups.setdefault(PurePosixPath(info.filename).parts[1], []).append(info.filename)
    archives = _ThreadArchives(volumes)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=extract_workers) as executor:
            futures = {
                scene: executor.submit(
                    _extract_scene, archives, scene, members, pbr, objects, parts
                )
                for scene, members in sorted(groups.items())
            }
            receipts = {scene: future.result() for scene, future in futures.items()}
    finally:
        archives.close()
    if not receipts:
        raise ValueError("empty PBR source")
    counts = {
        str(obj): sum(r["object_frame_counts"][str(obj)] for r in receipts.values())
        for obj in objects
    }
    missing = sorted(int(k) for k, n in counts.items() if n <= 0)
    if missing != sorted(set(allowed_missing)):
        raise ValueError(
            f"PBR missing object coverage {missing}; "
            "explicit matching data-gap acknowledgement required"
        )
    coverage = {
        "dataset": "hot3d",
        "frames": sum(r["frames"] for r in receipts.values()),
        "scenes": len(receipts),
        "object_frame_counts": counts,
        "missing_objects": missing,
        "available_objects": [obj for obj in objects if obj not in missing],
        "full33_object_training_coverage": not missing,
        "scene_gt_sha256": {scene: r["scene_gt_sha256"] for scene, r in receipts.items()},
    }
    publish_json(
        pbr / "exclusions.json",
        {
            "dataset": "hot3d",
            "excluded_frames": [],
            "missing_required_files": [],
            "basis": "No HANDAL exclusions reused. CRC verified during required extraction; "
            "consumed images/depth/masks are checked by conversion. No raw-tree file/hash audit.",
        },
    )
    publish_json(stage / "source_inventory.json", coverage)
    publish_json(
        stage / sources.READY_NAME,
        {
            "identity": identity,
            "dataset": "hot3d",
            "repo": profile.hf_repo,
            "revision": profile.revision,
            "coverage": coverage,
            "model_units": models,
            "real_data_staged": False,
            "test_GT_read": False,
            "test_object_targets_read": False,
            "legacy_renders_or_weights_used": False,
        },
    )
    if (root / "data").exists():
        raise ValueError("refuse to overwrite published source")
    os.rename(stage, root / "data")


# --------------------------------------------------------------------------------------
# HOT3D unit canonicalization and defect filters
# --------------------------------------------------------------------------------------
def convert_metadata(
    gt: Mapping[str, Any], cameras: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return copies of PBR ``scene_gt`` and ``scene_camera`` with translations in millimeters.

    Only ``cam_t_m2c`` and ``cam_t_w2c`` are scaled. ``cam_K``, rotations and ``depth_scale``
    stay unchanged: ``depth_scale`` is already millimeters per encoded depth unit.

    Raises:
        ValueError: If the frame keys differ or a translation or depth scale is invalid.
    """
    if set(gt) != set(cameras):
        raise ValueError("PBR pose/calibration keys differ")
    gt_mm, cameras_mm = copy.deepcopy(dict(gt)), copy.deepcopy(dict(cameras))
    for values in gt_mm.values():
        for row in values:
            row["cam_t_m2c"] = _translation_mm(row["cam_t_m2c"], "invalid PBR translation")
    for row in cameras_mm.values():
        if "cam_t_w2c" in row:
            row["cam_t_w2c"] = _translation_mm(row["cam_t_w2c"], "invalid camera translation")
        if row.get("depth_scale", 1) <= 0:
            raise ValueError("invalid depth scale")
    return gt_mm, cameras_mm


def _translation_mm(value: Any, message: str) -> list[float]:  # noqa: ANN401
    translation = np.asarray(value, dtype=float)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError(message)
    return (translation * MM_PER_M).tolist()


def canonicalize_units(
    source: Path, destination: Path, profile: sources.DatasetProfile | None = None
) -> Path:
    """Build the canonical-millimeter metadata view of the prepared HOT3D PBR source.

    Pinned HOT3D PLY coordinates and decoded ``depth * depth_scale`` are millimeters, but the
    PBR ``cam_t_m2c`` and camera ``cam_t_w2c`` are meters. Only that translation metadata is
    converted; RGB, depth and masks stay unchanged behind read-only links.

    Args:
        source: Root holding the output of :func:`extract_hot3d_source` in ``data/``.
        destination: Root of the new view; its ``data/`` receives it.
        profile: HOT3D profile (default: the packaged one).

    Returns:
        The view directory ``destination/data``.

    Raises:
        ValueError: If the source is not the pinned HOT3D source or its content changed.
    """
    profile = _hot3d_profile(profile)
    raw = Path(source).resolve() / "data"
    ready = _read_json(raw / sources.READY_NAME)
    _expect_pinned_ready(ready, profile)
    if ready["model_units"]["canonical_metadata_length_units"] != "millimetres":
        raise ValueError("CAD metadata must already be canonical mm")
    identity = {
        "dataset": "hot3d",
        "profile_sha256": sources.profile_sha256(profile),
        "raw_source_ready_sha256": sha256_file(raw / sources.READY_NAME),
        "conversion": "PBR GT/camera translation m->mm only; depth_scale already mm and unchanged",
        "canonicalizer_sha256": _version_sha256(CANONICALIZER_VERSION),
    }
    destination = Path(destination)
    final = destination / "data"
    with sources.file_lock(destination / RUN_LOCK):
        if (final / sources.READY_NAME).exists():
            if _read_json(final / sources.READY_NAME)["identity"] != identity:
                raise _identity_changed("canonical view", destination)
            return final
        _stage_canonical(raw, destination / ".staging", final, ready, identity, profile)
    return final


def _stage_canonical(
    raw: Path,
    stage: Path,
    final: Path,
    ready: dict[str, Any],
    identity: dict[str, Any],
    profile: sources.DatasetProfile,
) -> None:
    """Write the canonical metadata of every scene into ``stage`` and publish it."""
    stage.mkdir(exist_ok=True)
    pbr = stage / "train_pbr"
    pbr.mkdir(exist_ok=True)
    records: dict[str, dict[str, str]] = {}
    counts = {str(obj): 0 for obj in profile.meta["objects"]}
    frames = 0
    for scene, expected_sha in sorted(ready["coverage"]["scene_gt_sha256"].items()):
        original, target = raw / "train_pbr" / scene, pbr / scene
        target.mkdir(exist_ok=True)
        if sha256_file(original / "scene_gt.json") != expected_sha:
            raise ValueError("raw PBR GT differs from extraction receipt")
        gt = _read_json(original / "scene_gt.json")
        camera = _read_json(original / "scene_camera.json")
        gt_mm, camera_mm = convert_metadata(gt, camera)
        publish_json(target / "scene_gt.json", gt_mm)
        publish_json(target / "scene_camera.json", camera_mm)
        _link_pixels(original, target, [*PIXEL_DIRECTORIES, "scene_gt_info.json"])
        if not all((target / name).is_dir() for name in PIXEL_DIRECTORIES):
            raise ValueError("required canonical pixel link missing")
        for values in gt.values():
            for obj in {g["obj_id"] for g in values}:
                counts[str(obj)] += 1
        frames += len(gt)
        records[scene] = {
            "raw_gt_sha256": expected_sha,
            "raw_camera_sha256": sha256_file(original / "scene_camera.json"),
            "canonical_gt_sha256": sha256_file(target / "scene_gt.json"),
            "canonical_camera_sha256": sha256_file(target / "scene_camera.json"),
        }
        LOGGER.info("HOT3D PBR units: scene %s, %d frames", scene, len(gt))
    if counts != ready["coverage"]["object_frame_counts"] or frames != ready["coverage"]["frames"]:
        raise ValueError("unit conversion changed source population")
    _link_models(stage, raw)
    shutil.copyfile(raw / "train_pbr" / "exclusions.json", pbr / "exclusions.json")
    coverage = {
        **ready["coverage"],
        "scene_gt_sha256": {s: r["canonical_gt_sha256"] for s, r in records.items()},
    }
    units = {
        "raw_pose_translation_units": "metres",
        "canonical_pose_translation_units": "millimetres",
        "raw_camera_translation_units": "metres",
        "canonical_camera_translation_units": "millimetres",
        "depth_units": "millimeters after unchanged depth_scale",
        "depth_scale_multiplier": 1,
        "pose_translation_multiplier": MM_PER_M,
        "RGB_depth_mask_bytes_unchanged": True,
        "original_annotations_retained": True,
        "raw_source": str(raw),
        "scenes": records,
    }
    publish_json(stage / "source_inventory.json", coverage)
    publish_json(
        stage / sources.READY_NAME,
        {**ready, "identity": identity, "coverage": coverage, "PBR_units": units},
    )
    if final.exists():
        raise ValueError("refuse unreceipted canonical view overwrite")
    os.rename(stage, final)


def _expect_canonical_ready(ready: Mapping[str, Any], profile: sources.DatasetProfile) -> None:
    """Require the pinned canonical-millimeter HOT3D source."""
    units = ready.get("PBR_units", {})
    identity = ready.get("identity", {})
    if (
        ready.get("dataset") != "hot3d"
        or ready.get("revision") != profile.revision
        or identity.get("profile_sha256") != sources.profile_sha256(profile)
        or units.get("canonical_pose_translation_units") != "millimetres"
        or units.get("depth_scale_multiplier") != 1
    ):
        raise ValueError("pinned canonical-mm source required")


def _drop_frames(scene: Path, keys: Sequence[str]) -> tuple[dict, dict, dict | None]:
    """Return ``scene``'s GT, camera and GT-info (if any) without the frames ``keys``."""
    gt = _read_json(scene / "scene_gt.json")
    camera = _read_json(scene / "scene_camera.json")
    info_path = scene / "scene_gt_info.json"
    info = _read_json(info_path) if info_path.exists() else None
    for key in keys:
        del gt[key]
        del camera[key]
        if info is not None:
            info.pop(key, None)
    return gt, camera, info


def _write_filtered_scene(
    original: Path, target: Path, gt: dict, camera: dict, info: dict | None
) -> None:
    """Write filtered metadata and link the untouched pixels."""
    publish_json(target / "scene_gt.json", gt)
    publish_json(target / "scene_camera.json", camera)
    if info is not None:
        publish_json(target / "scene_gt_info.json", info)
    _link_pixels(original, target, PIXEL_DIRECTORIES)


def _record_filtered_scene(
    scene: str, target: Path, coverage: dict[str, Any], units: dict[str, Any]
) -> None:
    """Update the coverage and unit records with the filtered scene's metadata hashes."""
    gt_sha = sha256_file(target / "scene_gt.json")
    coverage["scene_gt_sha256"][scene] = gt_sha
    units["scenes"][scene]["canonical_gt_sha256"] = gt_sha
    units["scenes"][scene]["canonical_camera_sha256"] = sha256_file(target / "scene_camera.json")


def filter_source_defects(
    source: Path, destination: Path, profile: sources.DatasetProfile | None = None
) -> Path:
    """Build a metadata view without the frames whose required depth is unusable.

    Only fixed metadata and the known empty depth files are inspected. The canonical metadata
    and every RGB, depth and mask byte stay untouched; no depth or mask is fabricated.

    Args:
        source: Root of the canonical-millimeter view (:func:`canonicalize_units`).
        destination: Root of the new view; its ``data/`` receives it.
        profile: HOT3D profile whose depth defects are applied (default: packaged).

    Returns:
        The view directory ``destination/data``.

    Raises:
        ValueError: If the parent is not the pinned canonical source or the defect evidence
            no longer matches it.
    """
    profile = _hot3d_profile(profile)
    raw = Path(source).resolve() / "data"
    ready_path = raw / sources.READY_NAME
    ready = _read_json(ready_path)
    _expect_canonical_ready(ready, profile)
    manifest = sources.hot3d_depth_defects(profile)
    identity = {
        **ready["identity"],
        "parent_ready_sha256": sha256_file(ready_path),
        "defects_manifest_sha256": sources.hot3d_defects_sha256(profile),
        "filter_sha256": _version_sha256(DEFECT_FILTER_VERSION),
    }
    destination = Path(destination)
    final = destination / "data"
    with sources.file_lock(destination / RUN_LOCK):
        if (final / sources.READY_NAME).exists():
            if _read_json(final / sources.READY_NAME)["identity"] != identity:
                raise _identity_changed("filtered source", destination)
            return final
        stage = destination / ".staging"
        stage.mkdir(exist_ok=True)
        marker = stage / "FILTER_IDENTITY.json"
        if marker.exists() and _read_json(marker) != identity:
            raise _identity_changed("partial filtered source", stage)
        publish_json(marker, identity)
        coverage = copy.deepcopy(ready["coverage"])
        units = copy.deepcopy(ready["PBR_units"])
        records, seen = {}, set()
        for scene, expected_sha in sorted(ready["coverage"]["scene_gt_sha256"].items()):
            original, target = raw / "train_pbr" / scene, stage / "train_pbr" / scene
            target.mkdir(parents=True, exist_ok=True)
            defects = [row for row in manifest["defective_frames"] if row["scene"] == scene]
            if not defects:
                _link_pixels(original, target, [*PIXEL_DIRECTORIES, *SCENE_METADATA_FILES])
                continue
            if sha256_file(original / "scene_gt.json") != expected_sha:
                raise ValueError("canonical GT changed")
            _check_depth_defects(raw, original, defects)
            keys = [str(row["frame"]) for row in defects]
            gt, camera, info = _drop_frames(original, keys)
            for row in defects:
                coverage["frames"] -= 1
                for obj in row["objects"]:
                    coverage["object_frame_counts"][str(obj)] -= 1
                seen.add((scene, row["frame"]))
            _write_filtered_scene(original, target, gt, camera, info)
            _record_filtered_scene(scene, target, coverage, units)
            records[scene] = {
                "parent_gt_sha256": expected_sha,
                "filtered_gt_sha256": sha256_file(target / "scene_gt.json"),
            }
        if seen != {(r["scene"], r["frame"]) for r in manifest["defective_frames"]}:
            raise ValueError("not all declared defects were removed")
        _link_models(stage, raw)
        publish_json(
            stage / "train_pbr" / "exclusions.json", sources.hot3d_exclusion_record(profile)
        )
        publish_json(stage / "known_shared_pixel_defects.json", manifest)
        publish_json(stage / "source_inventory.json", coverage)
        units["excluded_frame_count"] = len(seen)
        correction = {
            "reason": "official zero-byte depth; no valid depth supervision",
            "parent": str(raw),
            "records": records,
            "original_pixels_retained": True,
        }
        publish_json(
            stage / sources.READY_NAME,
            {
                **ready,
                "identity": identity,
                "coverage": coverage,
                "PBR_units": units,
                "source_correction": correction,
            },
        )
        if final.exists():
            raise ValueError("refuse to overwrite unreceipted filtered source")
        os.rename(stage, final)
    return final


def _check_depth_defects(raw: Path, original: Path, defects: Sequence[Mapping[str, Any]]) -> None:
    """Verify that each declared defect still matches the source metadata and empty file."""
    gt = _read_json(original / "scene_gt.json")
    camera = _read_json(original / "scene_camera.json")
    for row in defects:
        key = str(row["frame"])
        if key not in gt or sorted({pose["obj_id"] for pose in gt[key]}) != row["objects"]:
            raise ValueError("defect target/frame metadata differs")
        if key not in camera:
            raise ValueError("defect target/frame metadata differs")
        for defect in row["defects"]:
            if (raw / defect["file"]).stat().st_size != 0:
                raise ValueError("declared empty depth no longer matches source")


def filter_object_masks(
    source: Path,
    destination: Path,
    obj: int,
    profile: sources.DatasetProfile | None = None,
) -> Path:
    """Build an object-scoped view that omits that object's evidenced unusable-mask frames.

    A GT list element is never removed (it would shift the BOP instance-to-mask indices); the
    whole frame is omitted from this object's view only. Other objects keep using their
    original or depth-corrected views and retain the valid samples of those frames. No pixel is
    modified and no mask is invented.

    Args:
        source: Root of the parent view: the depth-corrected one for an object with a depth
            defect, otherwise the canonical-millimeter one.
        destination: Root of the new view; its ``data/`` receives it.
        obj: Object the view is for.
        profile: HOT3D profile whose mask defects are applied (default: packaged).

    Returns:
        The view directory ``destination/data``.

    Raises:
        ValueError: If ``obj`` has no evidenced mask defect, the parent is not the right
            unscoped pinned source, or the defect evidence no longer matches it.
    """
    profile = _hot3d_profile(profile)
    records = sources.hot3d_mask_records(profile, obj)
    if not records:
        raise ValueError("object has no evidenced mask defects")
    raw = Path(source).resolve() / "data"
    ready_path = raw / sources.READY_NAME
    ready = _read_json(ready_path)
    _expect_canonical_ready(ready, profile)
    if ready.get("permitted_training_objects") is not None:
        raise ValueError("unscoped pinned canonical-mm parent required")
    parent = _read_json(raw / "train_pbr" / "exclusions.json")
    if obj in sources.hot3d_affected_objects(profile):
        if parent != sources.hot3d_exclusion_record(profile):
            raise ValueError("affected object requires depth-corrected parent")
    elif (
        parent.get("dataset") != "hot3d"
        or parent.get("excluded_frames")
        or parent.get("missing_required_files")
    ):
        raise ValueError("unexpected parent exclusions")
    identity = {
        **ready["identity"],
        "parent_ready_sha256": sha256_file(ready_path),
        "mask_policy_sha256": sources.hot3d_mask_policy_sha256(profile, obj),
        "filter_sha256": _version_sha256(MASK_FILTER_VERSION),
        "object": obj,
    }
    destination = Path(destination)
    final = destination / "data"
    with sources.file_lock(destination / RUN_LOCK):
        if (final / sources.READY_NAME).exists():
            if _read_json(final / sources.READY_NAME)["identity"] != identity:
                raise _identity_changed("object mask view", destination)
            return final
        stage = destination / ".staging"
        stage.mkdir(exist_ok=True)
        marker = stage / "FILTER_IDENTITY.json"
        if marker.exists() and _read_json(marker) != identity:
            raise _identity_changed("partial object mask view", stage)
        publish_json(marker, identity)
        _stage_object_view(raw, stage, final, ready, identity, profile, obj, records)
    return final


def _stage_object_view(
    raw: Path,
    stage: Path,
    final: Path,
    ready: dict[str, Any],
    identity: dict[str, Any],
    profile: sources.DatasetProfile,
    obj: int,
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Write the object-scoped view into ``stage`` and publish it."""
    coverage = copy.deepcopy(ready["coverage"])
    units = copy.deepcopy(ready["PBR_units"])
    seen: set[tuple[str, int]] = set()
    for scene, gt_sha in sorted(ready["coverage"]["scene_gt_sha256"].items()):
        original, target = raw / "train_pbr" / scene, stage / "train_pbr" / scene
        target.mkdir(parents=True, exist_ok=True)
        defects = [row for row in records if row["scene"] == scene]
        if not defects:
            _link_pixels(original, target, [*PIXEL_DIRECTORIES, *SCENE_METADATA_FILES])
            continue
        if sha256_file(original / "scene_gt.json") != gt_sha:
            raise ValueError("parent GT changed")
        gt_parent = _read_json(original / "scene_gt.json")
        for row in defects:
            if gt_parent[str(row["frame"])][row["instance"]]["obj_id"] != obj:
                raise ValueError("original mask instance does not belong to declared object")
            path = raw / row["file"]
            if row["reason"] == "missing_official_archive_member":
                if os.path.lexists(path):
                    raise ValueError("declared missing mask now exists")
            elif path.stat().st_size != 0:
                raise ValueError("declared empty mask does not match original source")
        frames = sorted({row["frame"] for row in defects})
        for frame in frames:
            for object_id in {g["obj_id"] for g in gt_parent[str(frame)]}:
                coverage["object_frame_counts"][str(object_id)] -= 1
            coverage["frames"] -= 1
            seen.add((scene, frame))
        gt, camera, info = _drop_frames(original, [str(f) for f in frames])
        _write_filtered_scene(original, target, gt, camera, info)
        _record_filtered_scene(scene, target, coverage, units)
    if seen != {(r["scene"], r["frame"]) for r in records}:
        raise ValueError("not all mask defects removed")
    _link_models(stage, raw)
    publish_json(
        stage / "train_pbr" / "exclusions.json", sources.hot3d_mask_exclusion_record(profile, obj)
    )
    publish_json(stage / "source_inventory.json", coverage)
    correction = {
        "object": obj,
        "parent": str(raw),
        "omitted_frames": len(seen),
        "original_pixels_retained": True,
        "original_instance_indices_preserved": True,
    }
    publish_json(
        stage / sources.READY_NAME,
        {
            **ready,
            "identity": identity,
            "coverage": coverage,
            "PBR_units": units,
            "permitted_training_objects": [obj],
            "object_mask_correction": correction,
        },
    )
    if final.exists():
        raise ValueError("refuse unreceipted mask-view overwrite")
    os.rename(stage, final)
