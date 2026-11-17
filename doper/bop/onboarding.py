# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Onboarding, test and GT views of the BOP real-image archives in one canonical layout.

Every view lives under ``<root>/onboarding``::

    <kind>/<sequence>/{rgb/*, scene_camera.json}   kind: static, dynamic or test
    gt/<sequence>/{scene_gt.json, scene_camera.json, rgb -> ../../static/<sequence>/rgb,
                   <mask_dir>/}

``mask_dir`` is ``mask_visib`` for HANDAL and HOPE and ``mask_supplied`` for HOT3D; the GT
``READY.json`` records it per frame. Sequences are staged into a temporary directory and
renamed into place, and each view's ``READY.json`` merges the sequences staged so far, so an
interrupted run resumes and a later run may add objects.

Only RGB and sanitized intrinsics reach the pseudo-label views: GT poses, masks, depth and
camera extrinsics are never staged there. The GT view is isolated and holds only frames with
a released pose, one instance of the sequence's object, and a released mask. The unlabeled
test view needs the transductive consent flag.

Validation overlap, per dataset: a view raises when an RGB byte hash equals a validation
image, except where the profile says ``overlap_policy: record`` (HANDAL): the
overlaps are then listed in ``READY.json`` for the exporter to exclude. HOT3D checks the native
and the rectified hash and always raises.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import zipfile
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np

from . import sources
from .extract import data_dir
from .population import build_population
from .sources import IMAGE_SUFFIXES, DatasetProfile, canonical_sha256, publish_json, sha256_file

ONBOARDING_DIR = "onboarding"
GT_DIR = "gt"
LOCK_NAME = ".lock"
KINDS = ("static", "dynamic", "test")
SEQUENCE_PATTERNS = {
    "static": r"obj_(\d{6})(?:_(up|down))?",
    "dynamic": r"obj_(\d{6})",
    "test": r"\d{6}",
}
CAMERA_FIELDS = ("cam_K", "width", "height")
MASK_VISIBLE = "mask_visib"
MASK_SUPPLIED = "mask_supplied"
MASK_SUFFIX = "_000000.png"
STAGING = ".staging-"
RECORD_POLICY = "record"
GT_SOURCE_KIND = "onboarding_static_ground_truth"


@dataclass(frozen=True)
class ViewSummary:
    """What one staging call produced.

    Attributes:
        kind: ``static``, ``dynamic``, ``test``, ``gt`` or ``hot3d_object``.
        sequences: Sorted names of the sequences the call covered.
        rgb_counts: RGB frames (GT frames for ``gt``) per sequence.
        excluded_validation: Staged RGB paths (relative to the view) whose bytes equal a
            validation image; the exporter excludes them.
        ready: The receipt of the view.
    """

    kind: str
    sequences: tuple[str, ...]
    rgb_counts: Mapping[str, int]
    excluded_validation: tuple[str, ...]
    ready: Path


@dataclass(frozen=True)
class Validation:
    """The validation population a view must not overlap.

    Attributes:
        hashes: SHA-256 of every validation image (HOT3D: native and rectified).
        sha256: Identity of the population (canonical JSON without ``val_root``).
        frames: Number of validation frames.
    """

    hashes: frozenset[str]
    sha256: str
    frames: int


def sanitize_camera(scene_camera: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only ``cam_K``, ``width`` and ``height`` of every frame, after validating ``K``.

    Args:
        scene_camera: A BOP ``scene_camera.json`` mapping.

    Returns:
        The sanitized mapping; extrinsics, depth scale and other fields are dropped.

    Raises:
        ValueError: If the mapping is empty or a frame has no valid pinhole ``cam_K``.
    """
    if not scene_camera:
        raise ValueError("empty camera sequence")
    cleaned = {}
    for key, row in scene_camera.items():
        k = [float(x) for x in row.get("cam_K", [])]
        if len(k) != 9 or not np.isfinite(k).all() or k[0] <= 0 or k[4] <= 0 or k[6:] != [0, 0, 1]:
            raise ValueError(f"invalid onboarding intrinsics in frame {key}")
        cleaned[key] = {field: row[field] for field in CAMERA_FIELDS if field in row}
    return cleaned


def _locked(root: str | Path) -> AbstractContextManager[None]:
    """Hold the onboarding lock of ``root`` (shared with the HOT3D stager)."""
    return sources.file_lock(Path(root) / ONBOARDING_DIR / LOCK_NAME)


def _fresh_directory(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def _publish_directory(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.rename(source, destination)


def load_validation(root: Path, profile: DatasetProfile) -> Validation:
    """Load the extracted validation split of a BOP dataset (``<root>/data/val``).

    Args:
        root: Dataset root with the ``val`` part extracted.
        profile: Dataset profile (``meta.layout.val`` names the directory, default ``val``).

    Returns:
        The validation image hashes and the population identity.

    Raises:
        ValueError: If the split is not extracted or its frame count differs from the pin.
    """
    directory = data_dir(root) / profile.meta.get("layout", {}).get("val", "val")
    try:
        manifest = build_population(directory)
    except (OSError, ValueError) as err:
        raise ValueError(f"validation split {directory} is not extracted: {err}") from err
    frames = manifest["frames"]
    expected = profile.meta.get("counts", {}).get("val_rgb")
    if expected is not None and len(frames) != expected:
        raise ValueError(f"validation split has {len(frames)} frames; the profile pins {expected}")
    identity = canonical_sha256({k: v for k, v in manifest.items() if k != "val_root"})
    return Validation(frozenset(f["sha256"] for f in frames), identity, len(frames))


def _check_names(archive: zipfile.ZipFile) -> None:
    seen: set[str] = set()
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
            raise ValueError(f"unsafe archive path {info.filename!r}")
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError(f"archive link not allowed: {info.filename}")
        if info.filename in seen:
            raise ValueError(f"duplicate archive member {info.filename}")
        seen.add(info.filename)


def _wanted_objects(
    profile: DatasetProfile, kind: str, objects: Sequence[int] | None
) -> list[int] | None:
    known = profile.meta.get(f"onboarding_{kind}_objects") or profile.meta.get("objects")
    if objects is None:
        return sorted(known) if known else None
    chosen = sorted(set(objects))
    if len(chosen) != len(objects):
        raise ValueError("object ids must be unique")
    unknown = [o for o in chosen if known and o not in known]
    if unknown:
        raise ValueError(f"object {unknown[0]} is not an onboarding object of {profile.name}")
    return chosen


def _view_archive(profile: DatasetProfile, kind: str) -> tuple[str, str]:
    files = profile.parts.get(kind, [])
    prefix = profile.meta.get("onboarding_prefix", {}).get(kind)
    if len(files) != 1 or not prefix:
        raise ValueError(f"profile {profile.name} has no single-archive {kind} onboarding part")
    return files[0], prefix


def _sequence_object(kind: str, sequence: str) -> int | None:
    match = re.fullmatch(SEQUENCE_PATTERNS[kind], sequence)
    if match is None:
        return None
    return int(match[1]) if kind != "test" else 0


def _scan(
    archive: zipfile.ZipFile, prefix: str, kind: str, wanted: Sequence[int] | None
) -> dict[str, dict[str, Any]]:
    """Group the RGB and camera members of the wanted sequences."""
    found: dict[str, dict[str, Any]] = {}
    for info in archive.infolist():
        parts = PurePosixPath(info.filename).parts
        if len(parts) < 3 or parts[0] != prefix:
            continue
        obj = _sequence_object(kind, parts[1])
        if obj is None or (kind != "test" and wanted is not None and obj not in wanted):
            continue
        entry = found.setdefault(parts[1], {"rgb": [], "camera": None, "object": obj})
        if (
            len(parts) == 4
            and parts[2] == "rgb"
            and Path(parts[3]).suffix.lower() in IMAGE_SUFFIXES
        ):
            entry["rgb"].append(info)
        elif len(parts) == 3 and parts[2] == "scene_camera.json":
            entry["camera"] = info
    return {s: e for s, e in sorted(found.items()) if e["rgb"] or e["camera"]}


def _require_inventory(
    profile: DatasetProfile,
    kind: str,
    found: Mapping[str, Mapping[str, Any]],
    wanted: Any,  # noqa: ANN401
) -> None:
    for sequence, entry in found.items():
        if not entry["rgb"] or entry["camera"] is None:
            raise ValueError(f"incomplete RGB/calibration population of {sequence}")
    if wanted is not None:
        present = {entry["object"] for entry in found.values()}
        missing = [o for o in wanted if o not in present]
        if missing:
            raise ValueError(f"object {missing[0]} has no {kind} images")
    total = sum(len(entry["rgb"]) for entry in found.values())
    if kind == "test":
        expected = profile.meta["test_population"]
        if (total, len(found)) != (expected["images"], expected["scenes"]):
            raise ValueError("unexpected test RGB/calibration inventory")


def _check_test_frame(payload: bytes, name: str, camera: Mapping[str, Any]) -> None:
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"undecodable test RGB {name}")
    first = next(iter(camera.values()))
    row = camera.get(str(int(Path(name).stem)))
    if row is None:
        if not all(
            np.allclose(r["cam_K"], first["cam_K"], rtol=1e-8, atol=1e-6) for r in camera.values()
        ):
            raise ValueError(f"test frame {name} lacks calibration and intrinsics vary")
        row = first
    height, width = image.shape[:2]
    if (row.get("width", width), row.get("height", height)) != (width, height):
        raise ValueError(f"test image {name} and calibration size mismatch")


def _stage_sequence(
    archive: zipfile.ZipFile,
    entry: Mapping[str, Any],
    destination: Path,
    kind: str,
    validation: Validation,
    overlaps: list[dict[str, str]],
    record: bool,
) -> dict[str, Any]:
    """Write one sequence's RGB and sanitized camera; return its receipt entry."""
    sequence = destination.name
    raw = archive.read(entry["camera"])
    camera = sanitize_camera(json.loads(raw))
    publish_json(destination / "scene_camera.json", camera)
    digests = {}
    for info in sorted(entry["rgb"], key=lambda i: i.filename):
        name = PurePosixPath(info.filename).name
        payload = archive.read(info)
        digest = hashlib.sha256(payload).hexdigest()
        if kind == "test":
            _check_test_frame(payload, name, camera)
        if digest in validation.hashes:
            if not record:
                raise ValueError(
                    f"onboarding RGB {sequence}/rgb/{name} overlaps the validation split"
                )
            overlaps.append({"path": f"{sequence}/rgb/{name}", "sha256": digest})
        (destination / "rgb").mkdir(exist_ok=True)
        (destination / "rgb" / name).write_bytes(payload)
        digests[name] = digest
    return {
        "rgb_count": len(digests),
        "camera_source_sha256": hashlib.sha256(raw).hexdigest(),
        "camera_sha256": sha256_file(destination / "scene_camera.json"),
        "rgb_sha256": canonical_sha256(digests),
    }


def _read_ready(view: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    path = view / sources.READY_NAME
    if not path.exists():
        return {"identity": dict(identity), "sequences": {}}
    ready = json.loads(path.read_text())
    if ready["identity"] != identity:
        raise ValueError(f"{path}: staged identity changed; remove the view to restage it")
    return ready


def _identity(
    profile: DatasetProfile, validation_sha256: str, extra: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "dataset": profile.name,
        "revision": profile.revision,
        "profile_sha256": sources.profile_sha256(profile),
        "evaluation_manifest_sha256": validation_sha256,
        **extra,
    }


def _summary(
    kind: str, sequences: Sequence[str], ready: Mapping[str, Any], path: Path
) -> ViewSummary:
    chosen = set(sequences)
    overlaps = tuple(
        o["path"] for o in ready.get("validation_overlap", []) if o["path"].split("/")[0] in chosen
    )
    counts = {s: ready["sequences"][s]["rgb_count"] for s in sorted(chosen)}
    return ViewSummary(kind, tuple(sorted(chosen)), counts, overlaps, path)


def stage_view(
    root: Path,
    profile: DatasetProfile,
    kind: str,
    *,
    objects: Sequence[int] | None = None,
    consent: bool = False,
) -> ViewSummary:
    """Stage the static, dynamic or test RGB and sanitized intrinsics of a BOP dataset.

    The archive is fetched and SHA-256 checked first (``sources.fetch``). The dataset's
    validation split must be extracted under ``<root>/data`` (the ``val`` part).

    Args:
        root: Dataset root.
        profile: HANDAL or HOPE profile with ``onboarding_prefix``.
        kind: ``static``, ``dynamic`` or ``test``.
        objects: Objects to stage (default: the profile's); not for ``test``.
        consent: Transductive consent, required for ``test``.

    Returns:
        The summary of the staged sequences.

    Raises:
        ValueError: Without consent for ``test``, for an unsafe archive, a missing object or
            calibration, an unexpected population, or a validation overlap the profile does
            not allow to be recorded.
    """
    if kind not in KINDS:
        raise ValueError(f"view kind must be one of {KINDS}")
    if kind == "test":
        sources.require_transductive_consent(consent)
        if objects is not None:
            raise ValueError("the test view has no object selection")
    name, prefix = _view_archive(profile, kind)
    wanted = _wanted_objects(profile, kind, objects) if kind != "test" else None
    root = Path(root)
    sources.fetch(profile, root, [kind], allow_transductive=consent)
    validation = load_validation(root, profile)
    record = profile.meta.get("overlap_policy") == RECORD_POLICY
    view = root / ONBOARDING_DIR / kind
    identity = _identity(
        profile, validation.sha256, {"archive": name, "archive_sha256": profile.files[name]}
    )
    with (
        _locked(root),
        zipfile.ZipFile(sources.source_dir(root) / name) as archive,
    ):
        _check_names(archive)
        found = _scan(archive, prefix, kind, wanted)
        _require_inventory(profile, kind, found, wanted if objects is None else list(objects))
        ready = _read_ready(view, identity)
        todo = [s for s in found if s not in ready["sequences"]]
        stage = _fresh_directory(view / f"{STAGING}{os.getpid()}")
        overlaps: list[dict[str, str]] = []
        try:
            for sequence in todo:
                (stage / sequence).mkdir()
                ready["sequences"][sequence] = _stage_sequence(
                    archive, found[sequence], stage / sequence, kind, validation, overlaps, record
                )
            for sequence in todo:
                _publish_directory(stage / sequence, view / sequence)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        _check_population(profile, kind, ready, objects)
        ready.update(_view_receipt(profile, kind, validation, ready, overlaps, record))
        publish_json(view / sources.READY_NAME, ready)
    return _summary(kind, list(found), ready, view / sources.READY_NAME)


def _check_population(
    profile: DatasetProfile, kind: str, ready: Mapping[str, Any], objects: Sequence[int] | None
) -> None:
    expected = profile.meta.get("counts", {}).get(f"{kind}_rgb")
    if objects is None and expected is not None and kind != "test":
        total = sum(s["rgb_count"] for s in ready["sequences"].values())
        if total != expected:
            raise ValueError(f"{kind} view has {total} RGB frames; the profile pins {expected}")


def _view_receipt(
    profile: DatasetProfile,
    kind: str,
    validation: Validation,
    ready: Mapping[str, Any],
    overlaps: Sequence[Mapping[str, str]],
    record: bool,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "dataset": profile.name,
        "dataset_revision": profile.revision,
        "source_kind": sources.SOURCE_KINDS[kind],
        "evaluation_manifest_sha256": validation.sha256,
        "validation_frames": validation.frames,
        "validation_overlap_action": "exporter excludes these hashes" if record else "refused",
        "validation_overlap": [*ready.get("validation_overlap", []), *overlaps],
        "gt_supervision_staged": False,
        "camera_extrinsics_staged": False,
    }
    if kind == "test":
        receipt.update(
            transductive_test_adaptation=True,
            test_targets_staged=False,
            notice=sources.TEST_NOTICE,
        )
    else:
        receipt["objects"] = sorted({int(s.split("_")[1]) for s in ready["sequences"]})
    return receipt


def _gt_keys(
    sequence: str, gt: Mapping[str, Any], camera: Mapping[str, Any], obj: int
) -> list[str]:
    keys = []
    for key, rows in sorted(gt.items(), key=lambda item: int(item[0])):
        if key not in camera:
            raise ValueError(f"{sequence} frame {key}: GT frame has no exact camera calibration")
        if len(rows) != 1 or rows[0]["obj_id"] != obj:
            raise ValueError(
                f"{sequence} frame {key}: expected exactly one instance of object {obj}"
            )
        keys.append(key)
    return keys


def _static_rgb(static: Path, sequence: str, frame: int) -> Path:
    paths = [static / sequence / "rgb" / f"{frame:06d}{ext}" for ext in sorted(IMAGE_SUFFIXES)]
    found = [p for p in paths if p.is_file()]
    if len(found) != 1:
        raise ValueError(f"{sequence} frame {frame}: missing or ambiguous static RGB")
    return found[0]


def _stage_gt_sequence(
    archive: zipfile.ZipFile,
    prefix: str,
    sequence: str,
    obj: int,
    static: Path,
    destination: Path,
    validation: Validation,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base = f"{prefix}/{sequence}/"
    gt_raw, camera_raw = (
        archive.read(base + "scene_gt.json"),
        archive.read(base + "scene_camera.json"),
    )
    camera = json.loads(camera_raw)
    keys = _gt_keys(sequence, json.loads(gt_raw), camera, obj)
    if not keys:
        raise ValueError(f"{sequence}: empty static GT sequence")
    (destination / MASK_VISIBLE).mkdir(parents=True)
    (destination / "scene_gt.json").write_bytes(gt_raw)
    publish_json(destination / "scene_camera.json", sanitize_camera({k: camera[k] for k in keys}))
    frames = []
    for key in keys:
        frame = int(key)
        rgb = _static_rgb(static, sequence, frame)
        digest = sha256_file(rgb)
        if digest in validation.hashes:
            raise ValueError(f"GT RGB {sequence}/rgb/{rgb.name} overlaps the validation split")
        mask_name = f"{frame:06d}{MASK_SUFFIX}"
        try:
            payload = archive.read(f"{base}{MASK_VISIBLE}/{mask_name}")
        except KeyError as err:
            raise ValueError(f"{sequence} frame {key}: missing released mask") from err
        (destination / MASK_VISIBLE / mask_name).write_bytes(payload)
        frames.append(_gt_record(obj, sequence, frame, rgb.name, digest, MASK_VISIBLE, payload))
    metadata = {
        "source_gt_sha256": hashlib.sha256(gt_raw).hexdigest(),
        "source_camera_sha256": hashlib.sha256(camera_raw).hexdigest(),
        "staged_gt_sha256": sha256_file(destination / "scene_gt.json"),
        "staged_camera_sha256": sha256_file(destination / "scene_camera.json"),
        "gt_frames": len(keys),
    }
    return metadata, frames


def _gt_record(
    obj: int, sequence: str, frame: int, rgb: str, rgb_sha256: str, mask_dir: str, mask: bytes
) -> dict[str, Any]:
    return {
        "object_id": obj,
        "sequence": sequence,
        "frame": frame,
        "rgb": f"{sequence}/rgb/{rgb}",
        "rgb_sha256": rgb_sha256,
        "mask_dir": mask_dir,
        "mask": f"{sequence}/{mask_dir}/{frame:06d}{MASK_SUFFIX}",
        "mask_sha256": hashlib.sha256(mask).hexdigest(),
    }


def _link_rgb(gt: Path, sequence: str) -> None:
    link = gt / sequence / "rgb"
    link.unlink(missing_ok=True)
    link.symlink_to(Path("..") / ".." / "static" / sequence / "rgb")


def _gt_receipt(
    ready: dict[str, Any], validation: Validation, wanted: Sequence[int] | None
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for frame in ready["frames"]:
        counts[str(frame["object_id"])] = counts.get(str(frame["object_id"]), 0) + 1
    if wanted is not None and any(str(o) not in counts for o in wanted):
        raise ValueError("an expected object lacks pose-annotated static GT")
    return {
        "source_kind": GT_SOURCE_KIND,
        "evaluation_manifest_sha256": validation.sha256,
        "gt_supervision_staged": True,
        "pose_annotated_only": True,
        "depth_available": False,
        "no_pose_interpolation": True,
        "object_frame_counts": dict(sorted(counts.items(), key=lambda item: int(item[0]))),
        "gt_field_note": "released static pose and mask annotations; not a claim of perfect "
        "manual annotation",
    }


def stage_gt_view(
    root: Path, profile: DatasetProfile, *, objects: Sequence[int] | None = None
) -> ViewSummary:
    """Stage the isolated supervised view of the static onboarding frames.

    Every frame listed in a sequence's ``scene_gt.json`` is staged and must have exactly one
    instance of the sequence's object, an exact camera entry and a released visible mask;
    otherwise staging raises. Dense frames without a GT entry are not part of this view and
    nothing is interpolated.
    ``rgb`` links to the static view, which must be staged first.

    Args:
        root: Dataset root.
        profile: HANDAL or HOPE profile.
        objects: Objects to stage (default: the profile's).

    Returns:
        The summary; ``rgb_counts`` counts GT frames.

    Raises:
        ValueError: If the static view is missing, a frame has no single matching instance,
            camera or mask, a RGB overlaps the validation split or an object has no GT.
    """
    root = Path(root)
    name, prefix = _view_archive(profile, "static")
    wanted = _wanted_objects(profile, "static", objects)
    static = root / ONBOARDING_DIR / "static"
    if not (static / sources.READY_NAME).is_file():
        raise ValueError("stage the static view before its GT view")
    sources.fetch(profile, root, ["static"])
    validation = load_validation(root, profile)
    gt = root / ONBOARDING_DIR / GT_DIR
    identity = _identity(
        profile, validation.sha256, {"archive": name, "archive_sha256": profile.files[name]}
    )
    with (
        _locked(root),
        zipfile.ZipFile(sources.source_dir(root) / name) as archive,
    ):
        _check_names(archive)
        names = set(archive.namelist())
        sequences = {
            n.split("/")[1]: _sequence_object("static", n.split("/")[1])
            for n in names
            if n.startswith(prefix + "/") and n.endswith("/scene_gt.json") and n.count("/") == 2
        }
        chosen = sorted(
            s for s, o in sequences.items() if o is not None and (wanted is None or o in wanted)
        )
        ready = _read_ready(gt, identity)
        ready.setdefault("metadata", {})
        ready.setdefault("frames", [])
        staged = set(json.loads((static / sources.READY_NAME).read_text())["sequences"])
        stage = _fresh_directory(gt / f"{STAGING}{os.getpid()}")
        todo = [s for s in chosen if s not in ready["metadata"]]
        try:
            for sequence in todo:
                if sequence not in staged:
                    raise ValueError(f"static view of {sequence} is not staged")
                metadata, frames = _stage_gt_sequence(
                    archive,
                    prefix,
                    sequence,
                    int(sequences[sequence] or 0),
                    static,
                    stage / sequence,
                    validation,
                )
                ready["metadata"][sequence] = {
                    **metadata,
                    "dense_rgb_frames": _dense(static, sequence),
                }
                ready["frames"] = [f for f in ready["frames"] if f["sequence"] != sequence] + frames
            for sequence in todo:
                _publish_directory(stage / sequence, gt / sequence)
                _link_rgb(gt, sequence)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        ready["frames"].sort(key=lambda f: (f["sequence"], f["frame"]))
        ready.update(_gt_receipt(ready, validation, wanted))
        publish_json(gt / sources.READY_NAME, ready)
    counts = {s: ready["metadata"][s]["gt_frames"] for s in chosen}
    return ViewSummary("gt", tuple(chosen), counts, (), gt / sources.READY_NAME)


def _dense(static: Path, sequence: str) -> int:
    return len(list((static / sequence / "rgb").iterdir()))
