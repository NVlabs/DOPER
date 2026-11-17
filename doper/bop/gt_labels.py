# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paired GT label pools from a staged GT view: box/mask and box/mask/keypoints.

``export_gt_pool`` reads the isolated supervised view (``<root>/onboarding/gt``, see
``doper.bop.onboarding``) of one object and writes two pools that feed the GT adaptation
methods ``gt_box_mask`` and ``gt_box_mask_keypoints``::

    <out>/{box_mask,box_mask_keypoints}/{train/{images,masks,annotations.json},keypoints_3d.json}
    <out>/{identity.json,POOL_READY.json}

Both arms hold byte-identical images, masks and boxes. The box is the tight bounding box of
the released mask (``mask > 127``). The keypoint arm adds the keypoints projected with the
released pose; no depth or self-occlusion is available in static onboarding, so an in-frame
point carries COCO visibility 1 (labeled, visibility not asserted) and an out-of-frame point
0. A frame is skipped, with its reason in ``POOL_READY.json``, when its mask is empty or has a
side of at most one pixel, or when no keypoint projects into the image.

The pool starts from a completed 25-epoch base run: its ``completed.json``, checkpoint and
keypoints must still match. Images and masks are copied and every path the receipt records is
relative to the pool, so a runs tree moved as a whole still verifies. ``verify_gt_pool``
re-checks the bytes on disk before training.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from doper.bop import sources
from doper.bop.convert import CV_TO_GL, _validate_pose, bop_pose_to_meters, load_keypoints
from doper.bop.onboarding import GT_SOURCE_KIND
from doper.bop.onboarding_hot3d import HOT3D_MASK_SEMANTICS
from doper.console import log
from doper.scene import geometry
from doper.scene.labels import annotation_record, mask_bbox

GT_ARMS = ("box_mask", "box_mask_keypoints")
BASE_EPOCHS = 25
KEYPOINT_COUNT = 64
MAX_ROUNDTRIP_PX = 0.01
MASK_THRESHOLD = 127
MASK_VALUES = frozenset({0, 255})
MIN_BOX_SIDE_PX = 1
MAX_OVERLAYS = 8
MM_PER_M = 1000.0
PARTIAL_SUFFIX = ".partial"
COMPLETED_NAME = "completed.json"
CHECKPOINT_PATH = Path("work") / "last.pth"
KEYPOINTS_PATH = Path("geometry") / "keypoints_3d.json"
READY_NAME = "POOL_READY.json"
IDENTITY_NAME = "identity.json"
KEYPOINTS_NAME = "keypoints_3d.json"
BOX_KEYS = ("id", "image_id", "category_id", "bbox", "area", "iscrowd", "mask_file", "mask_type")
FORBIDDEN_BOX_KEYS = ("keypoints", "num_keypoints", "object_pose")
SKIP_MASK = "empty_or_degenerate_mask"
SKIP_KEYPOINTS = "no_in_frame_gt_keypoints"
SKIP_REASONS = (SKIP_MASK, SKIP_KEYPOINTS)
RECORD_SOURCE = "onboarding_static_gt"
POOL_SCOPE = "common exact-pose-annotated static frames"
KEYPOINT_VISIBILITY = (
    "1 for labeled in-frame projections;0 out-of-frame; no depth/self-occlusion claim"
)
EXPORTER_PATH = Path(__file__)


@dataclass(frozen=True)
class GtPoolSummary:
    """What one GT pool export produced.

    Attributes:
        root: The pool directory (both arms below it).
        labels: Frames in each arm.
        source_frames: Pose-annotated GT frames of the object that were considered.
        skipped: Frames skipped.
        skip_reasons: Skipped frames per reason.
    """

    root: Path
    labels: int
    source_frames: int
    skipped: int
    skip_reasons: Mapping[str, int]


@dataclass(frozen=True)
class _Projection:
    """One frame's GT keypoints in pixels."""

    uv: np.ndarray
    in_frame: np.ndarray
    pose: np.ndarray
    intrinsics: geometry.Intrinsics
    roundtrip_px: float


def _read_json(path: Path) -> Any:  # noqa: ANN401
    return json.loads(path.read_text())


def _require_gt_view(gt_view: Path) -> dict[str, Any]:
    """The GT view's ``READY.json``, which must describe a supervised pose-annotated view."""
    path = gt_view / sources.READY_NAME
    ready = _read_json(path) if path.is_file() else {}
    if (
        ready.get("source_kind") != GT_SOURCE_KIND
        or not ready.get("gt_supervision_staged")
        or not ready.get("pose_annotated_only")
    ):
        raise ValueError(f"{gt_view} is not an explicit supervised static-GT view")
    return ready


def _require_base(base_run: Path, obj_id: int, keypoints: Path) -> tuple[str, str]:
    """Checkpoint and keypoint digests of a completed epoch-25 base run of ``obj_id``."""
    receipt = base_run / COMPLETED_NAME
    done = _read_json(receipt) if receipt.is_file() else {}
    checkpoint, geometry_file = base_run / CHECKPOINT_PATH, base_run / KEYPOINTS_PATH
    ok = (
        done.get("obj_id") == obj_id
        and done.get("epochs") == BASE_EPOCHS
        and checkpoint.is_file()
        and geometry_file.is_file()
        and sources.sha256_file(checkpoint) == done.get("checkpoint_sha256")
        and sources.sha256_file(geometry_file) == done.get("keypoints_sha256")
    )
    if not ok:
        raise ValueError(
            f"{base_run} is not a completed epoch-25 base of object {obj_id} with an "
            "unchanged checkpoint and keypoints; GT adaptation starts from the frozen epoch-25 base"
        )
    if sources.sha256_file(keypoints) != done["keypoints_sha256"]:
        raise ValueError(f"{keypoints} is not the keypoint file of the base run {base_run}")
    return done["checkpoint_sha256"], done["keypoints_sha256"]


def _supervision(ready: Mapping[str, Any]) -> dict[str, str]:
    """How the released labels are described in the pool identity."""
    hot3d = ready.get("identity", {}).get("dataset") == "hot3d" or "mask_semantics" in ready
    if not hot3d:
        return {
            "box_definition": "tight visible-mask bbox; no teacher gates",
            "gt_annotation_source": "official released static pose/mask annotations",
        }
    if ready.get("mask_semantics") != HOT3D_MASK_SEMANTICS:
        raise ValueError("HOT3D mask semantics must identify the released static PNG masks")
    return {
        "box_definition": "tight released-static-mask bbox; no teacher gates; "
        "no measured-visibility claim",
        "gt_annotation_source": "official exact static GT poses with released static PNG masks",
        "mask_semantics": HOT3D_MASK_SEMANTICS,
    }


def _identity(
    ready: Mapping[str, Any], gt_view: Path, obj_id: int, digests: tuple[str, str]
) -> dict[str, Any]:
    return {
        "object_id": obj_id,
        "source_ready_sha256": sources.sha256_file(gt_view / sources.READY_NAME),
        "base_sha256": digests[0],
        "keypoints_sha256": digests[1],
        "exporter_sha256": sources.sha256_file(EXPORTER_PATH),
        "evaluation_manifest_sha256": ready["evaluation_manifest_sha256"],
        "pool_scope": POOL_SCOPE,
        "keypoint_visibility": KEYPOINT_VISIBILITY,
        **_supervision(ready),
    }


def _project(
    entry: Mapping[str, Any], matrix: np.ndarray, points_m: np.ndarray, size: tuple[int, int]
) -> _Projection:
    """Project the keypoints with the GT pose and cross-check the scene-geometry projection."""
    width, height = size
    rotation, translation = _validate_pose(entry, "static GT")
    pose = bop_pose_to_meters(rotation, translation)
    xyz = points_m @ rotation.T + translation / MM_PER_M
    if not np.isfinite(xyz).all() or np.any(xyz[:, 2] <= 0):
        raise ValueError("invalid GT projection depth")
    homogeneous = xyz @ matrix.T
    uv = homogeneous[:, :2] / homogeneous[:, 2:]
    in_frame = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    intrinsics = geometry.Intrinsics(
        width, height, matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
    )
    other, _ = geometry.project_points(
        geometry.transform_points(pose, points_m), CV_TO_GL, intrinsics
    )
    roundtrip = float(np.linalg.norm(uv - other, axis=1).max())
    if roundtrip > MAX_ROUNDTRIP_PX:
        raise ValueError("GT unit/frame projection mismatch")
    return _Projection(uv, in_frame, pose, intrinsics, roundtrip)


def _annotations(
    projection: _Projection, mask: np.ndarray, image_id: int, mask_file: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The box/mask-only annotation and the full one (same box, area and mask)."""
    foreground = mask > MASK_THRESHOLD
    full = annotation_record(
        image_id,
        uv=projection.uv,
        vis=projection.in_frame.astype(int),
        bbox=mask_bbox(foreground),
        intr=projection.intrinsics,
        cam_to_world=CV_TO_GL,
        object_pose=projection.pose,
        view="mono",
        camera_index=0,
        mask_file=mask_file,
    )
    full["area"] = int(foreground.sum())
    return {key: full[key] for key in BOX_KEYS}, full


def _skip_reason(box: Mapping[str, Any], in_frame_count: int) -> str | None:
    if box["area"] == 0 or min(box["bbox"][2:]) <= MIN_BOX_SIDE_PX:
        return SKIP_MASK
    return SKIP_KEYPOINTS if in_frame_count == 0 else None


def _relative(target: Path, pool: Path) -> str:
    return Path(os.path.relpath(target.resolve(), pool.resolve())).as_posix()


def _read_inputs(
    gt_view: Path, row: Mapping[str, Any]
) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    """The frame's RGB and mask files with their decoded pixels, after the hash checks."""
    rgb, mask_path = gt_view / row["rgb"], gt_view / row["mask"]
    if (
        sources.sha256_file(rgb) != row["rgb_sha256"]
        or sources.sha256_file(mask_path) != row["mask_sha256"]
    ):
        raise ValueError("source bytes changed since the GT view was staged")
    image = cv2.imread(str(rgb))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if image is None or mask is None or mask.dtype != np.uint8 or mask.shape != image.shape[:2]:
        raise ValueError(f"bad GT image/mask shape: {rgb.name}")
    if not set(np.unique(mask).tolist()) <= MASK_VALUES:
        raise ValueError(f"expected a binary released mask: {mask_path.name}")
    return rgb, mask_path, image, mask


def _camera(calibration: Mapping[str, Any], size: tuple[int, int]) -> np.ndarray:
    matrix = np.asarray(calibration["cam_K"], dtype=float).reshape(3, 3)
    valid = np.isfinite(matrix).all() and matrix[0, 0] > 0 and matrix[1, 1] > 0
    if not valid or not np.allclose(matrix[2], [0, 0, 1]):
        raise ValueError("bad GT camera")
    if (calibration.get("width", size[0]), calibration.get("height", size[1])) != size:
        raise ValueError("GT intrinsics/image resolution mismatch")
    return matrix


def _write_overlay(stage: Path, index: int, image: np.ndarray, full: Mapping[str, Any]) -> None:
    overlay = image.copy()
    x, y, w, h = (int(v) for v in full["bbox"])
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for px, py, visible in np.asarray(full["keypoints"]).reshape(-1, 3):
        if visible > 0:
            cv2.circle(overlay, (round(px), round(py)), 3, (0, 0, 255), -1)
    (stage / "overlays").mkdir(exist_ok=True)
    cv2.imwrite(str(stage / "overlays" / f"{index:03d}.jpg"), overlay)


@dataclass
class _Collected:
    images: list[dict[str, Any]]
    annotations: dict[str, list[dict[str, Any]]]
    records: list[dict[str, Any]]
    skipped: list[dict[str, Any]]


def _store_frame(stage: Path, files: tuple[Path, Path], names: tuple[str, str]) -> None:
    """Copy one accepted frame's image and mask into both arms."""
    for arm in GT_ARMS:
        shutil.copyfile(files[0], stage / arm / "train" / "images" / names[0])
        shutil.copyfile(files[1], stage / arm / "train" / names[1])


def _collect(
    gt_view: Path,
    rows: list[dict[str, Any]],
    ready: Mapping[str, Any],
    points: np.ndarray,
    stage: Path,
    out: Path,
) -> _Collected:
    collected = _Collected([], {arm: [] for arm in GT_ARMS}, [], [])
    cache: dict[str, tuple[Any, Any]] = {}
    for row in rows:
        sequence, frame = row["sequence"], str(row["frame"])
        if sequence not in cache:
            scene, expected = gt_view / sequence, ready["metadata"][sequence]
            if (
                sources.sha256_file(scene / "scene_gt.json") != expected["staged_gt_sha256"]
                or sources.sha256_file(scene / "scene_camera.json")
                != expected["staged_camera_sha256"]
            ):
                raise ValueError("staged GT pose/calibration changed")
            cache[sequence] = (
                _read_json(scene / "scene_gt.json"),
                _read_json(scene / "scene_camera.json"),
            )
        (scene_gt, cameras) = cache[sequence]
        entry = scene_gt[frame][0]
        if entry["obj_id"] != row["object_id"]:
            raise ValueError("wrong GT object")
        rgb, mask_path, image, mask = _read_inputs(gt_view, row)
        height, width = image.shape[:2]
        projection = _project(
            entry, _camera(cameras[frame], (width, height)), points, (width, height)
        )
        stem = f"{sequence}_{row['frame']:06d}"
        image_name, mask_file = stem + rgb.suffix.lower(), f"masks/{stem}.png"
        image_id = len(collected.images)
        box, full = _annotations(projection, mask, image_id, mask_file)
        in_frame = int(projection.in_frame.sum())
        reason = _skip_reason(box, in_frame)
        if reason:
            collected.skipped.append(
                {"sequence": sequence, "frame": row["frame"], "reason": reason}
            )
            continue
        collected.images.append(
            {"id": image_id, "file_name": image_name, "width": width, "height": height}
        )
        for arm, annotation in zip(GT_ARMS, (box, full), strict=True):
            annotation["bop"] = {
                "scene": sequence,
                "frame": row["frame"],
                "instance": 0,
                "source": RECORD_SOURCE,
            }
            collected.annotations[arm].append(annotation)
        _store_frame(stage, (rgb, mask_path), (image_name, mask_file))
        collected.records.append(
            {
                **row,
                "rgb_source": _relative(rgb, out),
                "mask_source": _relative(mask_path, out),
                "image": image_name,
                "pool_mask": mask_file,
                "in_frame_keypoints": in_frame,
                "projected_point_mask_support": _mask_support(projection, mask),
                "projection_roundtrip_max_px": projection.roundtrip_px,
            }
        )
        if image_id < MAX_OVERLAYS:
            _write_overlay(stage, image_id, image, full)
    return collected


def _mask_support(projection: _Projection, mask: np.ndarray) -> float | None:
    pixels = np.floor(projection.uv[projection.in_frame]).astype(int)
    if len(pixels) == 0:
        return None
    return float((mask[pixels[:, 1], pixels[:, 0]] > MASK_THRESHOLD).mean())


def _json_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sources.sha256_file(path)
        for path in sorted(root.rglob("*.json"))
        if path.name != READY_NAME
    }


def _summary(root: Path, receipt: Mapping[str, Any]) -> GtPoolSummary:
    reasons = Counter(row["reason"] for row in receipt["skipped"])
    return GtPoolSummary(
        root,
        receipt["labels"],
        receipt["pose_annotated_source_frames"],
        len(receipt["skipped"]),
        {reason: reasons[reason] for reason in SKIP_REASONS},
    )


def export_gt_pool(
    gt_view: Path, obj_id: int, keypoints: Path, base_run: Path, out: Path
) -> GtPoolSummary:
    """Write the paired ``box_mask`` / ``box_mask_keypoints`` pools of one object.

    Args:
        gt_view: ``<root>/onboarding/gt`` as ``stage_gt_view`` (or the HOT3D stager) left it.
        obj_id: The object to export.
        keypoints: ``keypoints_3d.json`` of the object (meters); it must be the base run's.
        base_run: The completed 25-epoch base run of the object; its checkpoint and keypoints
            must still match ``completed.json``.
        out: The pool directory. A new directory is published atomically; one that holds the
            same identity is verified and returned unchanged.

    Returns:
        The summary of the pool.

    Raises:
        ValueError: If the view is not a supervised GT view, the base is not a completed
            epoch-25 run, an input changed, a projection disagrees with the independent
            projection by more than 0.01 px, no frame survives, or ``out`` holds a pool with
            another identity or one that fails its audit.
    """
    gt_view, out = Path(gt_view), Path(out)
    ready = _require_gt_view(gt_view)
    digests = _require_base(Path(base_run), obj_id, Path(keypoints))
    _, points = load_keypoints(Path(keypoints))
    if len(points) != KEYPOINT_COUNT:
        raise ValueError(f"GT pools keep the exact {KEYPOINT_COUNT}-keypoint base geometry")
    identity = _identity(ready, gt_view, obj_id, digests)
    # Both GT arms share the pool and its staging directory: a second job waits, then finds the
    # published pool.
    lock = sources.sibling_lock(out)
    with sources.file_lock(lock, busy=f"another job is exporting the GT pool {out}"):
        if out.exists():
            return _existing(out, identity)
        return _build_pool(gt_view, obj_id, Path(keypoints), points, ready, identity, out)


def _build_pool(
    gt_view: Path,
    obj_id: int,
    keypoints: Path,
    points: np.ndarray,
    ready: Mapping[str, Any],
    identity: Mapping[str, Any],
    out: Path,
) -> GtPoolSummary:
    """Stage, receipt and publish a new pool (the caller holds the pool lock)."""
    rows = sorted(
        (r for r in ready["frames"] if r["object_id"] == obj_id),
        key=lambda r: (r["sequence"], r["frame"]),
    )
    stage = out.with_name(out.name + PARTIAL_SUFFIX)
    shutil.rmtree(stage, ignore_errors=True)
    for arm in GT_ARMS:
        for sub in ("images", "masks"):
            (stage / arm / "train" / sub).mkdir(parents=True)
        shutil.copyfile(keypoints, stage / arm / KEYPOINTS_NAME)
    collected = _collect(gt_view, rows, ready, points, stage, out)
    if not collected.images:
        shutil.rmtree(stage, ignore_errors=True)
        raise ValueError("zero-label GT training forbidden: no usable GT frame")
    for arm in GT_ARMS:
        sources.publish_json(
            stage / arm / "train" / "annotations.json",
            {
                "images": collected.images,
                "annotations": collected.annotations[arm],
                "categories": [{"id": 1, "name": f"obj_{obj_id:06d}"}],
            },
        )
    sources.publish_json(stage / IDENTITY_NAME, identity)
    receipt = {
        "object_id": obj_id,
        "labels": len(collected.images),
        "pose_annotated_source_frames": len(rows),
        "records": collected.records,
        "skipped": collected.skipped,
        "identity": identity,
        "file_sha256": _json_hashes(stage),
        "both_arms_identical_rgb_box_mask": True,
    }
    sources.publish_json(stage / READY_NAME, receipt)
    stage.rename(out)
    faults = verify_gt_pool(out)
    if faults:
        raise ValueError(f"GT pool failed its audit: {faults}")
    log(f"[gt] {receipt['labels']} labels of {len(rows)} GT frames; wrote {out}")
    return _summary(out, receipt)


def _existing(out: Path, identity: Mapping[str, Any]) -> GtPoolSummary:
    path = out / IDENTITY_NAME
    if not path.is_file() or _read_json(path) != identity:
        raise ValueError(f"GT pool identity changed at {out}; remove it to export again")
    faults = verify_gt_pool(out)
    if faults:
        raise ValueError(f"existing GT pool failed its audit: {faults}")
    return _summary(out, _read_json(out / READY_NAME))


def _arm_faults(pool: Path, arm: str, receipt: Mapping[str, Any]) -> list[str]:
    """Faults in one arm's images, masks and boxes."""
    faults: list[str] = []
    coco = _read_json(pool / arm / "train" / "annotations.json")
    by_image = {a["image_id"]: a for a in coco["annotations"]}
    if len(by_image) != len(receipt["records"]) or len(coco["images"]) != len(by_image):
        faults.append(f"{arm}: label counts differ from the receipt")
    for image, record in zip(coco["images"], receipt["records"], strict=False):
        annotation = by_image.get(image["id"], {})
        picture, mask_path = (
            pool / arm / "train" / "images" / image["file_name"],
            pool / arm / "train" / record["pool_mask"],
        )
        if (
            not picture.is_file()
            or not mask_path.is_file()
            or sources.sha256_file(picture) != record["rgb_sha256"]
            or sources.sha256_file(mask_path) != record["mask_sha256"]
        ):
            faults.append(f"{arm}: image or mask missing or changed ({image['file_name']})")
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        box = mask_bbox(mask > MASK_THRESHOLD) if mask is not None else None
        if (
            box is None
            or min(box[2:]) <= MIN_BOX_SIDE_PX
            or annotation.get("bbox") != [round(v, 2) for v in box]
        ):
            faults.append(f"{arm}: mask is empty or disagrees with its box ({image['file_name']})")
    return faults


def _contract_faults(pool: Path) -> list[str]:
    """Faults in the pairing of the two arms and in the keypoint visibility convention."""
    box = _read_json(pool / GT_ARMS[0] / "train" / "annotations.json")
    full = _read_json(pool / GT_ARMS[1] / "train" / "annotations.json")
    faults: list[str] = []
    if box["images"] != full["images"] or len(box["annotations"]) != len(full["annotations"]):
        faults.append("GT arm image population differs")
        return faults
    for plain, rich in zip(box["annotations"], full["annotations"], strict=True):
        if any(plain[key] != rich.get(key) for key in plain) or any(
            k in plain for k in FORBIDDEN_BOX_KEYS
        ):
            faults.append("GT box/mask targets differ between arms or pose leaked into box schema")
        visibility = np.asarray(rich["keypoints"]).reshape(-1, 3)[:, 2]
        if not set(visibility.tolist()) <= {0.0, 1.0} or rich["num_keypoints"] != int(
            visibility.sum()
        ):
            faults.append("keypoint visibility is not 1 in frame and 0 out of frame")
    return faults


def verify_gt_pool(out: Path) -> list[str]:
    """Audit a GT pool's bytes against its receipt.

    Checks: recorded JSON files unchanged; identity matches the receipt; both arms hold the
    recorded images and masks, with boxes equal to the mask boxes; both arms share images,
    boxes and masks, with no pose or keypoint field in the box arm; visibility is 0 or 1;
    source frames, when the GT view is still where the receipt points, are unchanged.

    Args:
        out: A pool directory written by ``export_gt_pool``.

    Returns:
        The faults; empty when the pool is safe to train on.
    """
    out = Path(out)
    try:
        receipt = _read_json(out / READY_NAME)
        faults = [
            f"frozen GT pool changed: {name}"
            for name, digest in receipt["file_sha256"].items()
            if not (out / name).is_file() or sources.sha256_file(out / name) != digest
        ]
        if faults:
            return faults
        if _read_json(out / IDENTITY_NAME) != receipt["identity"]:
            faults.append("identity.json differs from the receipt")
        for arm in GT_ARMS:
            faults.extend(_arm_faults(out, arm, receipt))
        faults.extend(_contract_faults(out))
        for row in receipt["records"]:
            for key, digest in (("rgb_source", "rgb_sha256"), ("mask_source", "mask_sha256")):
                source = (out / row[key]).resolve()
                if source.is_file() and sources.sha256_file(source) != row[digest]:
                    faults.append("frozen static GT source pixels changed")
        return faults
    except (ValueError, KeyError, OSError) as error:
        return [str(error)]
