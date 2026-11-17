# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HOT3D per-object onboarding: reference tars staged into the canonical onboarding layout.

Each object has Aria and Quest3 reference tars (static up/down, dynamic). Native frames are
rectified and numbered ``device_index * 1_000_000 + frame_id`` so both devices share one
sequence directory; static frames with a released pose and mask also enter the GT view
(``mask_supplied``). The shared receipts and helpers live in ``doper.bop.onboarding``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import hot3d, hot3d_staging, sources
from .export import encode_jpeg
from .onboarding import (
    GT_DIR,
    MASK_SUFFIX,
    MASK_SUPPLIED,
    ONBOARDING_DIR,
    STAGING,
    Validation,
    ViewSummary,
    _fresh_directory,
    _gt_receipt,
    _gt_record,
    _identity,
    _link_rgb,
    _locked,
    _publish_directory,
    _read_ready,
    _view_receipt,
)
from .sources import DatasetProfile, canonical_sha256, publish_json, sha256_file

OBJECTS_DIR = "objects"
HOT3D_NAMESPACE = 1_000_000
HOT3D_MASK_VALUES = frozenset({0, 1, 255})
HOT3D_MASK_SEMANTICS = "released_static_PNG_visibility_unasserted"
HOT3D_TAR = re.compile(
    r"object_ref_(aria|quest3)_(static|dynamic)/(obj_\d{6}(?:_(?:up|down))?)\.tar"
)


def _read_member(tar: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    stream = tar.extractfile(member)
    if stream is None:
        raise ValueError(f"unreadable tar member {member.name}")
    with stream:
        return stream.read()


def _decode(raw: bytes, what: str) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"invalid onboarding {what}")
    return image


def _encode_png(mask: np.ndarray) -> bytes:
    ok, data = cv2.imencode(".png", mask)
    if not ok:
        raise ValueError("mask encode failed")
    return data.tobytes()


@dataclass
class _Sequence:
    """One HOT3D sequence while its tars are read (Aria and Quest3 share a sequence)."""

    camera: dict[str, Any] = field(default_factory=dict)
    digests: dict[str, str] = field(default_factory=dict)
    archives: list[dict[str, str]] = field(default_factory=list)
    gt: dict[str, Any] = field(default_factory=dict)
    gt_camera: dict[str, Any] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)
    skipped: int = 0


def load_hot3d_validation(root: Path, profile: DatasetProfile) -> Validation:
    """The staged HOT3D validation panel: native and rectified image hashes and its identity.

    Raises:
        ValueError: If the staged manifest belongs to another profile.
    """
    path = root / hot3d_staging.VALIDATION_MANIFEST
    manifest = json.loads(path.read_text())
    if manifest["identity"] != hot3d_staging.validation_identity(profile):
        raise ValueError("validation manifest identity differs from this profile")
    hashes = {h for f in manifest["frames"] for h in (f["sha256"], f["native_rgb_sha256"])}
    return Validation(frozenset(hashes), sha256_file(path), len(manifest["frames"]))


def _hot3d_frame(
    tar: tarfile.TarFile,
    row: Mapping[str, tarfile.TarInfo],
    fid: int,
    spec: Mapping[str, Any],
    state: _Sequence,
    validation: Validation,
    stage: Path,
) -> None:
    if fid >= HOT3D_NAMESPACE:
        raise ValueError("native frame exceeds the reserved device namespace")
    logical = spec["device_index"] * HOT3D_NAMESPACE + fid
    key = str(logical)
    if key in state.camera:
        raise ValueError("duplicate onboarding logical frame")
    if "camera" not in row:
        raise ValueError("missing exact per-frame calibration")
    camera_row = json.loads(_read_member(tar, row["camera"]))[spec["stream"]]
    raw = _read_member(tar, row["image"])
    native_sha = hashlib.sha256(raw).hexdigest()
    rectified, intrinsics = hot3d.rectify(_decode(raw, "image"), camera_row)
    payload = encode_jpeg(rectified)
    digest = hashlib.sha256(payload).hexdigest()
    if native_sha in validation.hashes or digest in validation.hashes:
        raise ValueError(f"onboarding frame {key} overlaps the validation split")
    name = f"{logical:06d}.jpg"
    folder = stage / spec["kind"] / spec["sequence"] / "rgb"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_bytes(payload)
    calibration = {
        "cam_K": intrinsics.ravel().tolist(),
        "width": rectified.shape[1],
        "height": rectified.shape[0],
    }
    state.camera[key] = calibration
    state.digests[name] = digest
    if spec["kind"] == "static":
        _hot3d_gt_frame(tar, row, camera_row, logical, name, spec, state, stage)


def _hot3d_gt_frame(
    tar: tarfile.TarFile,
    row: Mapping[str, tarfile.TarInfo],
    camera_row: dict[str, Any],
    logical: int,
    name: str,
    spec: Mapping[str, Any],
    state: _Sequence,
    stage: Path,
) -> None:
    if "object" not in row or "mask" not in row:
        state.skipped += 1
        return
    matches = json.loads(_read_member(tar, row["object"])).get(str(spec["object"]), [])
    if len(matches) != 1:
        raise ValueError("reference GT must have exactly the indicated object instance")
    rotation, translation = hot3d.camera_from_object_mm(camera_row, matches[0])
    native = _decode(_read_member(tar, row["mask"]), "mask")
    if native.ndim != 2 or not set(np.unique(native).tolist()) <= HOT3D_MASK_VALUES:
        raise ValueError("expected a released binary single-channel static PNG")
    mask, _ = hot3d.rectify((native > 0).astype(np.uint8) * 255, camera_row, mask=True)
    encoded = _encode_png(mask)
    folder = stage / GT_DIR / spec["sequence"] / MASK_SUPPLIED
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{logical:06d}{MASK_SUFFIX}").write_bytes(encoded)
    key = str(logical)
    pose = {
        "obj_id": spec["object"],
        "cam_R_m2c": rotation.ravel().tolist(),
        "cam_t_m2c": translation.tolist(),
    }
    state.gt[key] = [pose]
    state.gt_camera[key] = dict(state.camera[key])
    record = _gt_record(
        spec["object"], spec["sequence"], logical, name, state.digests[name], MASK_SUPPLIED, encoded
    )
    state.records.append(
        {**record, "device": spec["device"], "native_frame": logical % HOT3D_NAMESPACE,
         "mask_semantics": HOT3D_MASK_SEMANTICS}
    )  # fmt: skip


def _stage_tars(
    root: Path,
    profile: DatasetProfile,
    obj_id: int,
    tars: Sequence[str],
    validation: Validation,
    stage: Path,
) -> dict[tuple[str, str], _Sequence]:
    streams = hot3d_staging.streams(profile)
    devices = list(streams)
    states: dict[tuple[str, str], _Sequence] = {}
    for name in tars:
        match = HOT3D_TAR.fullmatch(name)
        if match is None:
            raise ValueError(f"unexpected onboarding tar name {name}")
        device = {"aria": "Aria", "quest3": "Quest3"}[match[1]]
        spec = {
            "device": device,
            "device_index": devices.index(device),
            "stream": streams[device],
            "kind": match[2],
            "sequence": match[3],
            "object": obj_id,
        }
        state = states.setdefault((match[2], match[3]), _Sequence())
        state.archives.append({"name": name, "sha256": profile.files[name]})
        static = match[2] == "static"
        with tarfile.open(sources.source_dir(root) / name) as tar:
            members = hot3d_staging.native_members(
                tar, spec["stream"], gt=static, supplied_mask=static
            )
            frame_ids = sorted(f for f, r in members.items() if "image" in r)
            if not frame_ids:
                raise ValueError("empty native onboarding stream")
            for fid in frame_ids:
                _hot3d_frame(tar, members[fid], fid, spec, state, validation, stage)
    return states


def _publish_object(
    root: Path,
    profile: DatasetProfile,
    identity: Mapping[str, Any],
    states: Mapping[tuple[str, str], _Sequence],
    validation: Validation,
    stage: Path,
) -> dict[str, dict[str, int]]:
    base = root / ONBOARDING_DIR
    counts: dict[str, dict[str, int]] = {"static": {}, "dynamic": {}}
    gt_sequences = []
    for (kind, sequence), state in sorted(states.items()):
        publish_json(stage / kind / sequence / "scene_camera.json", state.camera)
        if state.gt:
            scene = stage / GT_DIR / sequence
            publish_json(scene / "scene_gt.json", state.gt)
            publish_json(scene / "scene_camera.json", state.gt_camera)
            gt_sequences.append(sequence)
        counts[kind][sequence] = len(state.digests)
    if not gt_sequences:
        raise ValueError("no static exact-pose/released-mask supervision")
    for kind in ("static", "dynamic"):
        view = base / kind
        ready = _read_ready(view, identity)
        for sequence in counts[kind]:
            state = states[(kind, sequence)]
            _publish_directory(stage / kind / sequence, view / sequence)
            ready["sequences"][sequence] = {
                "rgb_count": len(state.digests),
                "camera_sha256": sha256_file(view / sequence / "scene_camera.json"),
                "rgb_sha256": canonical_sha256(state.digests),
                "source_archives": state.archives,
            }
        ready.update(_view_receipt(profile, kind, validation, ready, [], False))
        ready["device_namespace"] = "device_index * 1000000 + native frame id"
        publish_json(view / sources.READY_NAME, ready)
    _publish_gt(base / GT_DIR, stage, identity, states, gt_sequences, validation)
    return counts


def _publish_gt(
    gt: Path,
    stage: Path,
    identity: Mapping[str, Any],
    states: Mapping[tuple[str, str], _Sequence],
    sequences: Sequence[str],
    validation: Validation,
) -> None:
    ready = _read_ready(gt, identity)
    ready.setdefault("metadata", {})
    ready["frames"] = [f for f in ready.get("frames", []) if f["sequence"] not in sequences]
    for sequence in sequences:
        state = states[("static", sequence)]
        _publish_directory(stage / GT_DIR / sequence, gt / sequence)
        _link_rgb(gt, sequence)
        ready["frames"].extend(state.records)
        ready["metadata"][sequence] = {
            "staged_gt_sha256": sha256_file(gt / sequence / "scene_gt.json"),
            "staged_camera_sha256": sha256_file(gt / sequence / "scene_camera.json"),
            "gt_frames": len(state.gt),
            "dense_rgb_frames": len(state.digests),
            "skipped_source_frames": state.skipped,
        }
    ready["frames"].sort(key=lambda f: (f["sequence"], f["frame"]))
    ready.update(_gt_receipt(ready, validation, None))
    ready["mask_semantics"] = HOT3D_MASK_SEMANTICS
    publish_json(gt / sources.READY_NAME, ready)


def _object_summary(receipt: Path) -> ViewSummary:
    data = json.loads(receipt.read_text())
    counts = data["rgb_counts"]
    return ViewSummary("hot3d_object", tuple(sorted(counts)), counts, (), receipt)


def _is_staged(root: Path, receipt: Path, identity: Mapping[str, Any]) -> bool:
    """Whether the object receipt matches and every view it lists is still receipted."""
    if not receipt.is_file():
        return False
    data = json.loads(receipt.read_text())
    if data["identity"] != identity:
        return False
    base = root / ONBOARDING_DIR
    gt_ready = base / GT_DIR / sources.READY_NAME
    if not gt_ready.is_file():
        return False
    for sequence in data["rgb_counts"]:
        kind = "dynamic" if re.fullmatch(r"obj_\d{6}", sequence) else "static"
        ready = base / kind / sources.READY_NAME
        if not ready.is_file() or sequence not in json.loads(ready.read_text())["sequences"]:
            return False
        if not (base / kind / sequence / "scene_camera.json").is_file():
            return False
    return True


def stage_hot3d_object(root: Path, profile: DatasetProfile, obj_id: int) -> ViewSummary:
    """Stage one HOT3D object's static, dynamic and GT onboarding views from its tars.

    Each native frame is rectified (JPEG quality 95; masks with nearest-neighbor sampling) and
    gets the logical id ``device_index * 1_000_000 + frame_id`` so the Aria and Quest3 frames
    of a sequence share one directory. Static frames with a released pose and mask also enter
    the GT view (``mask_supplied``).

    Args:
        root: Dataset root with the frozen validation manifest.
        profile: HOT3D profile with ``meta.onboarding_tars``.
        obj_id: Object id.

    Returns:
        The summary over the static and dynamic sequences; ``ready`` is the object receipt.

    Raises:
        ValueError: For an unknown object, a mismatching validation manifest, a frame id
            outside the device namespace, a missing calibration, a GT instance count other
            than one, or an overlap with the validation images (native or rectified hash).
    """
    root = Path(root)
    tars = profile.meta.get("onboarding_tars", {}).get(str(obj_id))
    if not tars:
        raise ValueError(f"object {obj_id} has no onboarding tars in profile {profile.name}")
    validation = load_hot3d_validation(root, profile)
    identity = _identity(profile, validation.sha256, {})
    key = f"obj_{obj_id:06d}"
    receipt = root / ONBOARDING_DIR / OBJECTS_DIR / f"{key}.json"
    sources.fetch(replace(profile, parts={**profile.parts, key: list(tars)}), root, [key])
    with _locked(root):
        if _is_staged(root, receipt, identity):
            return _object_summary(receipt)
        stage = _fresh_directory(root / ONBOARDING_DIR / f"{STAGING}{key}-{os.getpid()}")
        try:
            states = _stage_tars(root, profile, obj_id, tars, validation, stage)
            counts = _publish_object(root, profile, identity, states, validation, stage)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        publish_json(
            receipt,
            {
                "identity": identity,
                "object_id": obj_id,
                "rgb_counts": {**counts["static"], **counts["dynamic"]},
                "static_gt_frames": sum(len(s.gt) for s in states.values()),
                "val_training_images": 0,
                "test_images_used": 0,
            },
        )
    return _object_summary(receipt)
