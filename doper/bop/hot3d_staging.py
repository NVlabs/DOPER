# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HOT3D real-image staging: the frozen validation manifest and the image-only test manifest.

Validation reads only the frozen labeled TRAIN panel (profile resource ``validation_panel``):
five fixed frames of one clip per training sequence. Each frame's designated native view is
rectified to the pinhole camera (``doper.bop.hot3d.rectify``) and JPEG-encoded; its GT is the
object poses and amodal-mask boxes of that view. The merged manifest is the population that
``doper.bop.ar.score_model`` scores with the native camera.

The test manifest holds only the designated native-view RGB and its sanitized intrinsic
calibration for the frozen image index (profile resource ``test_image_index``); no object,
mask, hand or pose member of a test clip is read. ``doper.bop.export.run_bank`` exports from it.

Both stage in finite shards with per-shard (test: per-clip) ``READY.json`` receipts, so an
interrupted run resumes, and publish atomically. Manifest paths are relative to the manifest's
directory, so their hashes do not depend on where the dataset root lives.

Multi-view fusion (``doper.bop.multiview``) also needs the synchronized support streams of each
test image and every stream's camera pose. ``stage_hot3d_support`` stages them separately from
the image-only test manifest (this grouping is pending author confirmation): per device, all streams
at the reference timestamp (Aria ``214-1`` with ``1201-1``/``1201-2``; Quest3 ``1201-1`` with
``1201-2``); mono support streams are rectified and converted to BGR. Camera poses are the
clip's ``T_world_from_camera`` in millimeters, stored under the ``cam_R_w2c``/``cam_t_w2c`` names
(the ``c2w_named_w2c`` convention). Object, mask and hand members are never read.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np

from . import hot3d, sources
from .export import encode_jpeg
from .multiview import ImageKey
from .sources import DatasetProfile, publish_json, sha256_file

VALIDATION_DIR = "hot3d_validation"
VALIDATION_MANIFEST = "validation_manifest.json"
TEST_DIR = "hot3d_test"
TEST_MANIFEST = "manifest.json"
SHARDS = 16
RUN_LOCK = "run.lock"
TEST_FRAMES = (29, 59, 89, 119, 149)
QUEST3_SCENES = range(1288, 1849)
ARIA_SCENES = range(3365, 3832)
MASK_SEMANTICS = "amodal GT projection with source vignette; NOT measured visible mask"
SUPPORT_DIR = "support"
SUPPORT_MANIFEST = "support_manifest.json"
# All synchronized streams of each device, the evaluation (reference) stream first.
DEVICE_STREAMS = {"Aria": ("214-1", "1201-1", "1201-2"), "Quest3": ("1201-1", "1201-2")}
SUPPORT_GROUPING = (
    "all synchronized device streams at the reference timestamp; mono streams rectified and "
    "converted to BGR (grouping pending author confirmation)"
)
TEST_FRAME_FIELDS = frozenset(
    {
        "scene_id",
        "im_id",
        "path",
        "rgb_sha256",
        "cam_K",
        "width",
        "height",
        "native_path",
        "native_rgb_sha256",
        "native_calibration",
        "device",
        "stream",
        "source_archive_sha256",
    }
)


def _profile(profile: DatasetProfile | None) -> DatasetProfile:
    profile = profile or sources.load_profile("hot3d")
    if profile.name != "hot3d":
        raise ValueError("HOT3D staging needs the hot3d profile")
    return profile


def _read_json(path: Path) -> Any:  # noqa: ANN401
    return json.loads(path.read_text())


def streams(profile: DatasetProfile) -> dict[str, str]:
    """Return the evaluation stream of each device (``Aria`` and ``Quest3``)."""
    return dict(profile.meta["evaluation_streams"])


def native_members(
    tar: tarfile.TarFile, stream: str, *, gt: bool = False, supplied_mask: bool = False
) -> dict[int, dict[str, tarfile.TarInfo]]:
    """Index a clip's per-frame camera, image and optional annotation members of one stream.

    Args:
        tar: Open clip archive.
        stream: Native stream id, for example ``214-1``.
        gt: Also index the ``objects.json`` annotation members.
        supplied_mask: Also index the released ``mask_<stream>.png`` members of the static
            onboarding references.

    Returns:
        Frame id to role (``camera``, ``image``, ``object``, ``mask``) to member.

    Raises:
        ValueError: For an unsafe member path, a link, or a duplicate frame member.
    """
    patterns = {
        "camera": r"(\d{6})\.cameras\.json",
        "image": rf"(\d{{6}})\.image_{re.escape(stream)}\.jpg",
    }
    if gt:
        patterns["object"] = r"(\d{6})\.objects\.json"
    if supplied_mask:
        patterns["mask"] = rf"(\d{{6}})\.mask_{re.escape(stream)}\.png"
    records: dict[int, dict[str, tarfile.TarInfo]] = {}
    for member in tar.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
            raise ValueError("unsafe native tar member")
        if not member.isfile():
            continue
        for role, pattern in patterns.items():
            match = re.fullmatch(pattern, path.name)
            if match:
                entry = records.setdefault(int(match[1]), {})
                if role in entry:
                    raise ValueError("duplicate frame/stream member")
                entry[role] = member
    return records


def _read_member(tar: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    stream = tar.extractfile(member)
    if stream is None:
        raise ValueError(f"unreadable tar member {member.name}")
    with stream:
        return stream.read()


def pose_targets(
    objects: Mapping[str, Any], camera_row: Mapping[str, Any], stream: str, known: Sequence[int]
) -> list[dict[str, Any]]:
    """GT targets of one native view: every instance with a non-empty amodal mask in it.

    There is no visibility-fraction threshold; instances annotated only for other camera views
    are not targets. Boxes are those of the rectified amodal mask.

    Args:
        objects: The frame's ``objects.json``: object id to instance rows.
        camera_row: The view's camera row (calibration and ``T_world_from_camera``).
        stream: Native stream id.
        known: The dataset's object ids.

    Returns:
        Targets in object-id then instance order.

    Raises:
        ValueError: For an unknown object id.
    """
    targets = []
    for key, instances in sorted(objects.items(), key=lambda item: int(item[0])):
        obj = int(key)
        if obj not in known:
            raise ValueError("unknown HOT3D object")
        for instance, row in enumerate(instances):
            encoded = row.get("masks_amodal", {}).get(stream)
            if not encoded or not encoded["rle"]:
                continue
            native_mask = hot3d.decode_rle(encoded)
            if not native_mask.any():
                continue
            rotation, translation = hot3d.camera_from_object_mm(dict(camera_row), row)
            rectified, _ = hot3d.rectify(native_mask, dict(camera_row), mask=True)
            targets.append(
                {
                    "obj_id": obj,
                    "instance": instance,
                    "cam_R_m2c": rotation.ravel().tolist(),
                    "cam_t_m2c": translation.tolist(),
                    "rectified_amodal_bbox": hot3d.mask_bbox(rectified),
                    "native_amodal_area": int(np.count_nonzero(native_mask)),
                    "GT_mask_semantics": MASK_SEMANTICS,
                }
            )
    return targets


def _decode_native(raw: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("invalid native image")
    return image


def _camera_code_sha256() -> str:
    assert hot3d.__file__ is not None
    return sha256_file(hot3d.__file__)


def _resource_sha256(profile: DatasetProfile, key: str) -> str:
    return profile.meta["resources"][key]["sha256"]


def validation_identity(profile: DatasetProfile | None = None) -> dict[str, Any]:
    """Identity of the validation staging: pinned source, frozen panel and camera code."""
    profile = _profile(profile)
    return {
        "dataset": "hot3d",
        "revision": profile.revision,
        "profile_sha256": sources.profile_sha256(profile),
        "panel_sha256": _resource_sha256(profile, "validation_panel"),
        "camera_code_sha256": _camera_code_sha256(),
        "camera_protocol": hot3d.PROTOCOL,
    }


def _panel_rows(profile: DatasetProfile) -> list[dict[str, Any]]:
    return list(sources.load_resource(profile, "validation_panel")["clip_rows"])


def _validation_frame(
    tar: tarfile.TarFile,
    row: Mapping[str, tarfile.TarInfo],
    clip: Mapping[str, Any],
    fid: int,
    stage: Path,
    profile: DatasetProfile,
    shard_dir: str,
) -> dict[str, Any]:
    device = clip["device"]
    stream = streams(profile)[device]
    if not {"image", "camera", "object"} <= row.keys():
        raise ValueError("missing selected validation frame data")
    native_bytes = _read_member(tar, row["image"])
    camera_row = json.loads(_read_member(tar, row["camera"]))[stream]
    rectified, intrinsics = hot3d.rectify(_decode_native(native_bytes), camera_row)
    payload = encode_jpeg(rectified)
    filename = f"{device.lower()}_{clip['clip_id']:06d}_{fid:06d}.jpg"
    (stage / "images" / filename).write_bytes(payload)
    (stage / "native" / filename).write_bytes(native_bytes)
    objects = json.loads(_read_member(tar, row["object"]))
    targets = pose_targets(objects, camera_row, stream, profile.meta["objects"])
    return {
        "scene": f"{clip['clip_id']:06d}",
        "scene_id": clip["clip_id"],
        "frame": fid,
        "sequence_id": clip["sequence_id"],
        "device": device,
        "stream": stream,
        "path": f"{shard_dir}/images/{filename}",
        "native_path": f"{shard_dir}/native/{filename}",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "native_rgb_sha256": hashlib.sha256(native_bytes).hexdigest(),
        "width": rectified.shape[1],
        "height": rectified.shape[0],
        "cam_K": intrinsics.ravel().tolist(),
        "native_calibration": hot3d.clean_calibration(camera_row),
        "objects": [g["obj_id"] for g in targets],
        "GT": targets,
        "source_archive": clip["source_path"],
        "source_archive_sha256": profile.files[clip["source_path"]],
    }


def stage_validation_shard(
    root: Path, shard: int, profile: DatasetProfile | None = None, *, shards: int = SHARDS
) -> Path:
    """Rectify and receipt one shard of the frozen validation panel.

    Args:
        root: Dataset root whose ``val`` part is fetched and recorded in ``READY.json``.
        shard: Shard index; the shard holds panel rows ``shard::shards``.
        profile: HOT3D profile (default: packaged).
        shards: Shard count.

    Returns:
        The published shard directory ``root/hot3d_validation/shard_XX/data``.

    Raises:
        ValueError: For an invalid shard, a panel row outside the labeled TRAIN clips, a
            missing frame, or a changed identity.
    """
    profile = _profile(profile)
    if shard not in range(shards):
        raise ValueError("invalid validation shard")
    sources.require_ready(root, ["val"])
    identity = validation_identity(profile)
    folder = Path(root) / VALIDATION_DIR / f"shard_{shard:02d}"
    final = folder / "data"
    with sources.file_lock(folder / RUN_LOCK):
        if (final / sources.READY_NAME).exists():
            if _read_json(final / sources.READY_NAME)["identity"] != identity:
                raise ValueError("validation shard identity changed")
            return final
        stage = folder / ".staging"
        for name in ("images", "native"):
            (stage / name).mkdir(parents=True, exist_ok=True)
        entries = _panel_rows(profile)[shard::shards]
        shard_dir = f"{VALIDATION_DIR}/shard_{shard:02d}/data"
        frames = []
        for clip in entries:
            name = clip["source_path"]
            if not name.startswith("train_" + clip["device"].lower() + "/"):
                raise ValueError("validation can only use frozen labeled TRAIN clips")
            with tarfile.open(sources.source_dir(root) / name) as tar:
                members = native_members(tar, streams(profile)[clip["device"]], gt=True)
                for fid in clip["frames"]:
                    row = members.get(fid, {})
                    frames.append(_validation_frame(tar, row, clip, fid, stage, profile, shard_dir))
        if len(frames) != sum(len(clip["frames"]) for clip in entries):
            raise ValueError("validation shard lost frames")
        publish_json(
            stage / sources.READY_NAME,
            {"identity": identity, "shard": shard, "frames": frames, "optimization_inputs": False},
        )
        if final.exists():
            raise ValueError("unreceipted validation publication")
        os.rename(stage, final)
    return final


def merge_validation(
    root: Path, profile: DatasetProfile | None = None, *, shards: int = SHARDS
) -> Path:
    """Merge the validation shards into ``root/validation_manifest.json``.

    Args:
        root: Dataset root holding every staged shard.
        profile: HOT3D profile (default: packaged).
        shards: Shard count used for staging.

    Returns:
        The manifest path.

    Raises:
        ValueError: If a shard identity differs, the frame coverage is not the panel's, an
            object has no validation frame (pre-training coverage gate) or a published
            manifest would change.
    """
    profile = _profile(profile)
    root = Path(root)
    identity = validation_identity(profile)
    frames = []
    for shard in range(shards):
        path = root / VALIDATION_DIR / f"shard_{shard:02d}" / "data" / sources.READY_NAME
        receipt = _read_json(path)
        if receipt["identity"] != identity or receipt["shard"] != shard:
            raise ValueError("validation shard identity differs")
        frames.extend(receipt["frames"])
    panel = _panel_rows(profile)
    expected = {(row["clip_id"], f) for row in panel for f in row["frames"]}
    keys = [(r["scene_id"], r["frame"]) for r in frames]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("validation frame coverage differs")
    objects = [int(o) for o in profile.meta["objects"]]
    counts = {str(o): sum(o in f["objects"] for f in frames) for o in objects}
    if any(n == 0 for n in counts.values()):
        raise ValueError(
            "pre-training validation coverage gate failed; freeze a revised panel before any "
            "model training"
        )
    frames.sort(key=lambda r: (r["scene_id"], r["frame"]))
    manifest = {
        "dataset": "hot3d",
        "identity": identity,
        "frames": frames,
        "objects": objects,
        "object_frame_counts": counts,
        "images": len(frames),
        "clips": len(panel),
        "source": "frozen labeled TRAIN panel; never optimization inputs",
        "test_used": False,
        "visibility_threshold": None,
        "camera_protocol": hot3d.PROTOCOL,
    }
    path = root / VALIDATION_MANIFEST
    if path.exists() and _read_json(path) != manifest:
        raise ValueError("frozen validation publication changed")
    publish_json(path, manifest)
    return path


def reference_device(scene: int) -> str:
    """Return the designated device of a HOT3D test scene (official BOP scene mapping).

    Raises:
        ValueError: For a scene outside the frozen HOT3D test split.
    """
    if scene in QUEST3_SCENES:
        return "Quest3"
    if scene in ARIA_SCENES:
        return "Aria"
    raise ValueError("scene is not in the frozen HOT3D test split")


def held_out_test_clips(profile: DatasetProfile | None = None) -> list[dict[str, Any]]:
    """Clips of the frozen image-only test index, with their device and pinned archive.

    Raises:
        ValueError: If the index carries anything but ``scene_id``/``im_id``, a clip does not
            hold the five fixed frames, or a clip archive is not pinned in the profile.
    """
    profile = _profile(profile)
    index = sources.load_resource(profile, "test_image_index")
    if any(set(row) != {"scene_id", "im_id"} for row in index["images"]):
        raise ValueError("frozen image-only schedule changed")
    frames: dict[int, list[int]] = {}
    for row in index["images"]:
        frames.setdefault(int(row["scene_id"]), []).append(int(row["im_id"]))
    clips = []
    for scene, ids in sorted(frames.items()):
        device = reference_device(scene)
        name = f"test_{device.lower()}/clip-{scene:06d}.tar"
        if sorted(ids) != list(TEST_FRAMES) or name not in profile.parts.get("test", []):
            raise ValueError("unexpected test archive/view/schedule")
        clips.append({"clip_id": scene, "device": device, "source_path": name, "frames": ids})
    return clips


def held_out_test_identity(profile: DatasetProfile | None = None) -> dict[str, Any]:
    """Identity of the test staging: pinned source, frozen index and camera code."""
    profile = _profile(profile)
    return {
        "dataset": "hot3d",
        "revision": profile.revision,
        "profile_sha256": sources.profile_sha256(profile),
        "image_index_sha256": _resource_sha256(profile, "test_image_index"),
        "camera_code_sha256": _camera_code_sha256(),
        "camera_protocol": hot3d.PROTOCOL,
        "test_object_GT_used": False,
        "test_object_targets_or_counts_used": False,
        "test_masks_or_hands_used": False,
        "camera_extrinsics_used": False,
        "image_only_schedule_used": True,
        "training_or_adaptation": False,
    }


def validate_test_frame(frame: Mapping[str, Any], profile: DatasetProfile) -> None:
    """Refuse a test frame with foreign fields, the wrong view or changed rectification.

    Raises:
        ValueError: If any test-supervision field leaked or the calibration changed.
    """
    device_streams = streams(profile)
    if (
        set(frame) != TEST_FRAME_FIELDS
        or type(frame["scene_id"]) is not int
        or type(frame["im_id"]) is not int
        or frame["im_id"] not in TEST_FRAMES
        or frame["device"] != reference_device(frame["scene_id"])
        or frame.get("stream") != device_streams.get(frame["device"])
    ):
        raise ValueError("foreign/test-supervision fields or wrong reference camera")
    calibration = frame["native_calibration"]
    if calibration != hot3d.clean_calibration(calibration):
        raise ValueError("extrinsics or foreign camera fields leaked")
    intrinsics = np.asarray(frame["cam_K"], dtype=float).reshape(3, 3)
    if (
        not np.isfinite(intrinsics).all()
        or min(intrinsics[0, 0], intrinsics[1, 1]) <= 0
        or not np.allclose(intrinsics[2], [0, 0, 1])
    ):
        raise ValueError("invalid rectified test intrinsics")
    if not np.array_equal(intrinsics, hot3d.camera_pair(calibration)[2]):
        raise ValueError("test rectification calibration changed")
    size = (calibration["image_width"], calibration["image_height"])
    if (frame["width"], frame["height"]) != size:
        raise ValueError("rectification resolution changed")


def _extract_clip(
    archive: Path,
    clip: Mapping[str, Any],
    destination: Path,
    identity: Mapping[str, Any],
    profile: DatasetProfile,
    relative: str,
) -> list[dict[str, Any]]:
    """Stage one test clip's RGB and sanitized calibration; resume from its receipt."""
    final = destination / "data"
    archive_sha = profile.files[clip["source_path"]]
    ident = {**identity, "clip": dict(clip), "archive_sha256": archive_sha}
    if (final / sources.READY_NAME).exists():
        ready = _read_json(final / sources.READY_NAME)
        if ready["identity"] != ident:
            raise ValueError("completed test clip identity changed")
        return ready["frames"]
    stage = destination / ".staging"
    stage.mkdir(parents=True, exist_ok=True)
    stream = streams(profile)[clip["device"]]
    frames = []
    with tarfile.open(archive) as tar:
        members = native_members(tar, stream)
        for fid in clip["frames"]:
            entry = members.get(fid, {})
            if set(entry) != {"camera", "image"}:
                raise ValueError("test clip is not RGB/calibration-only")
            raw = _read_member(tar, entry["image"])
            # Sanitize before rectification; world, device and object poses never enter it.
            calibration = hot3d.clean_calibration(
                json.loads(_read_member(tar, entry["camera"]))[stream]
            )
            rectified, intrinsics = hot3d.rectify(_decode_native(raw), calibration)
            payload = encode_jpeg(rectified)
            (stage / f"{fid:06d}.jpg").write_bytes(payload)
            (stage / f"{fid:06d}.native.jpg").write_bytes(raw)
            row = {
                "scene_id": clip["clip_id"],
                "im_id": fid,
                "path": f"{relative}/{fid:06d}.jpg",
                "native_path": f"{relative}/{fid:06d}.native.jpg",
                "rgb_sha256": hashlib.sha256(payload).hexdigest(),
                "native_rgb_sha256": hashlib.sha256(raw).hexdigest(),
                "cam_K": intrinsics.ravel().tolist(),
                "native_calibration": calibration,
                "width": rectified.shape[1],
                "height": rectified.shape[0],
                "device": clip["device"],
                "stream": stream,
                "source_archive_sha256": archive_sha,
            }
            validate_test_frame(row, profile)
            frames.append(row)
    publish_json(stage / sources.READY_NAME, {"identity": ident, "frames": frames})
    if final.exists():
        raise ValueError("refuse unreceipted test clip overwrite")
    os.rename(stage, final)
    return frames


def stage_test_shard(
    root: Path, shard: int, profile: DatasetProfile | None = None, *, shards: int = SHARDS
) -> Path:
    """Stage one shard of the frozen test clips (RGB and sanitized calibration only).

    Args:
        root: Dataset root whose ``test`` part is fetched and recorded in ``READY.json``.
        shard: Shard index; the shard holds clips ``shard::shards``.
        profile: HOT3D profile (default: packaged).
        shards: Shard count.

    Returns:
        The shard receipt path.

    Raises:
        ValueError: For an invalid shard, a clip with annotation members in the staged view,
            or a changed identity.
    """
    profile = _profile(profile)
    if shard not in range(shards):
        raise ValueError("invalid finite shard")
    sources.require_ready(root, ["test"])
    identity = held_out_test_identity(profile)
    folder = Path(root) / TEST_DIR / "shards" / f"{shard:02d}"
    receipt = folder / sources.READY_NAME
    with sources.file_lock(folder / RUN_LOCK):
        if receipt.exists():
            if _read_json(receipt)["identity"] != identity:
                raise ValueError("test shard identity changed")
            return receipt
        frames = []
        for clip in held_out_test_clips(profile)[shard::shards]:
            relative = f"shards/{shard:02d}/clips/{clip['clip_id']:06d}/data"
            frames.extend(
                _extract_clip(
                    sources.source_dir(root) / clip["source_path"],
                    clip,
                    folder / "clips" / f"{clip['clip_id']:06d}",
                    identity,
                    profile,
                    relative,
                )
            )
        publish_json(receipt, {"identity": identity, "shard": shard, "frames": frames})
    return receipt


def merge_test(root: Path, profile: DatasetProfile | None = None, *, shards: int = SHARDS) -> Path:
    """Merge the test shards into ``root/hot3d_test/manifest.json``.

    Args:
        root: Dataset root holding every staged test shard.
        profile: HOT3D profile (default: packaged).
        shards: Shard count used for staging.

    Returns:
        The manifest path.

    Raises:
        ValueError: If a shard identity differs, the population is not the frozen index, or a
            published manifest would change.
    """
    profile = _profile(profile)
    folder = Path(root) / TEST_DIR
    identity = held_out_test_identity(profile)
    frames = []
    for shard in range(shards):
        receipt = _read_json(folder / "shards" / f"{shard:02d}" / sources.READY_NAME)
        if receipt["identity"] != identity or receipt["shard"] != shard:
            raise ValueError("test shard receipt differs")
        frames.extend(receipt["frames"])
    index = sources.load_resource(profile, "test_image_index")
    expected = {(r["scene_id"], r["im_id"]) for r in index["images"]}
    keys = [(r["scene_id"], r["im_id"]) for r in frames]
    if len(keys) != len(expected) or len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError("incomplete/duplicate test image population")
    for frame in frames:
        validate_test_frame(frame, profile)
    frames.sort(key=lambda r: (r["scene_id"], r["im_id"]))
    manifest = {
        "dataset": "hot3d",
        "identity": identity,
        "frames": frames,
        "scene_count": len({k[0] for k in keys}),
        "RGB_count": len(frames),
        "input_fields": ["native_RGB", "sanitized_native_calibration", "rectified_RGB", "cam_K"],
        "test_GT_read": False,
        "test_object_targets_or_counts_read": False,
        "camera_extrinsics_used": False,
        "pose_frame": "original designated native camera; identity virtual rotation; BOP t in mm",
    }
    path = folder / TEST_MANIFEST
    if path.exists() and _read_json(path) != manifest:
        raise ValueError("published test source changed")
    publish_json(path, manifest)
    return path


def support_identity(profile: DatasetProfile | None = None) -> dict[str, Any]:
    """Identity of the support-stream staging: the test identity plus extrinsics and grouping."""
    profile = _profile(profile)
    return {
        **held_out_test_identity(profile),
        "camera_extrinsics_used": True,
        "device_streams": {device: list(names) for device, names in DEVICE_STREAMS.items()},
        "grouping": SUPPORT_GROUPING,
    }


def _support_streams(profile: DatasetProfile, device: str) -> tuple[str, ...]:
    names = DEVICE_STREAMS[device]
    if names[0] != streams(profile)[device]:
        raise ValueError("the reference stream must be the device's evaluation stream")
    return names


def _camera_pose_mm(camera_row: Mapping[str, Any]) -> dict[str, list[float]]:
    """``T_world_from_camera`` in millimeters under the w2c field names (``c2w_named_w2c``)."""
    world_from_camera = hot3d.transform(dict(camera_row["T_world_from_camera"]))
    return {
        "cam_R_w2c": world_from_camera[:3, :3].ravel().tolist(),
        "cam_t_w2c": (world_from_camera[:3, 3] * hot3d.MM_PER_M).tolist(),
    }


def _support_view(
    tar: tarfile.TarFile,
    entry: Mapping[str, tarfile.TarInfo],
    camera_row: Mapping[str, Any],
    stream: str,
    stage: Path,
    relative: str,
    fid: int,
) -> dict[str, Any]:
    raw = _read_member(tar, entry["image"])
    calibration = hot3d.clean_calibration(dict(camera_row))
    rectified, intrinsics = hot3d.rectify(_decode_native(raw), calibration)
    payload = encode_jpeg(rectified)
    name = f"{fid:06d}_{stream}"
    (stage / f"{name}.jpg").write_bytes(payload)
    (stage / f"{name}.native.jpg").write_bytes(raw)
    return {
        "stream": stream,
        "path": f"{relative}/{name}.jpg",
        "native_path": f"{relative}/{name}.native.jpg",
        "rgb_sha256": hashlib.sha256(payload).hexdigest(),
        "native_rgb_sha256": hashlib.sha256(raw).hexdigest(),
        "cam_K": intrinsics.ravel().tolist(),
        **_camera_pose_mm(camera_row),
        "native_calibration": calibration,
        "width": rectified.shape[1],
        "height": rectified.shape[0],
    }


def _stage_support_clip(
    archive: Path,
    clip: Mapping[str, Any],
    destination: Path,
    identity: Mapping[str, Any],
    profile: DatasetProfile,
    relative: str,
) -> list[dict[str, Any]]:
    """Stage one clip's support streams and camera poses; resume from its receipt."""
    final = destination / "data"
    ident = {**identity, "clip": dict(clip), "archive_sha256": profile.files[clip["source_path"]]}
    if (final / sources.READY_NAME).exists():
        ready = _read_json(final / sources.READY_NAME)
        if ready["identity"] != ident:
            raise ValueError("completed support clip identity changed")
        return ready["groups"]
    stage = destination / ".staging"
    stage.mkdir(parents=True, exist_ok=True)
    reference, *supports = _support_streams(profile, clip["device"])
    groups = []
    with tarfile.open(archive) as tar:
        members = {name: native_members(tar, name) for name in (reference, *supports)}
        for fid in clip["frames"]:
            entries = {name: members[name].get(fid, {}) for name in members}
            if any(set(entry) != {"camera", "image"} for entry in entries.values()):
                raise ValueError("test clip lacks a synchronized RGB/calibration stream")
            rows = json.loads(_read_member(tar, entries[reference]["camera"]))
            views = [
                _support_view(tar, entries[name], rows[name], name, stage, relative, fid)
                for name in supports
            ]
            groups.append(
                {
                    "scene_id": clip["clip_id"],
                    "im_id": fid,
                    "device": clip["device"],
                    "reference": {
                        "stream": reference,
                        "cam_K": hot3d.camera_pair(rows[reference])[2].ravel().tolist(),
                        **_camera_pose_mm(rows[reference]),
                    },
                    "supports": views,
                }
            )
    publish_json(stage / sources.READY_NAME, {"identity": ident, "groups": groups})
    if final.exists():
        raise ValueError("refuse unreceipted support clip overwrite")
    os.rename(stage, final)
    return groups


def stage_hot3d_support(root: Path, profile: DatasetProfile | None = None) -> Path:
    """Stage the synchronized support streams and camera poses of every frozen test image.

    Args:
        root: Dataset root whose ``test`` part is fetched and recorded in ``READY.json``.
        profile: HOT3D profile (default: packaged).

    Returns:
        The manifest ``root/hot3d_test/support_manifest.json``: per test image, the reference
        stream's ``cam_K`` and pose, and each support stream's rectified image, ``cam_K`` and
        pose. ``support_cameras`` keys its cameras for ``doper.bop.multiview``.

    Raises:
        ValueError: If a stream is missing at a scheduled frame, a receipt's identity changed,
            or a published manifest would change.
    """
    profile = _profile(profile)
    sources.require_ready(root, ["test"])
    identity = support_identity(profile)
    folder = Path(root) / TEST_DIR / SUPPORT_DIR
    groups = []
    with sources.file_lock(folder / RUN_LOCK):
        for clip in held_out_test_clips(profile):
            relative = f"{SUPPORT_DIR}/clips/{clip['clip_id']:06d}/data"
            groups.extend(
                _stage_support_clip(
                    sources.source_dir(root) / clip["source_path"],
                    clip,
                    folder / "clips" / f"{clip['clip_id']:06d}",
                    identity,
                    profile,
                    relative,
                )
            )
        groups.sort(key=lambda g: (g["scene_id"], g["im_id"]))
        manifest = {"dataset": "hot3d", "identity": identity, "groups": groups}
        path = Path(root) / TEST_DIR / SUPPORT_MANIFEST
        if path.exists() and _read_json(path) != manifest:
            raise ValueError("published support manifest changed")
        publish_json(path, manifest)
    return path


def support_cameras(manifest: Mapping[str, Any]) -> dict[ImageKey, dict[str, Any]]:
    """Every staged view's camera record (``cam_K`` and pose), keyed with its stream."""
    cameras: dict[ImageKey, dict[str, Any]] = {}
    for group in manifest["groups"]:
        for view in (group["reference"], *group["supports"]):
            key = ImageKey(int(group["scene_id"]), int(group["im_id"]), view["stream"])
            cameras[key] = {name: view[name] for name in ("cam_K", "cam_R_w2c", "cam_t_w2c")}
    return cameras
