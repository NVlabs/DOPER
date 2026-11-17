# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert BOP ``train_pbr`` to the DOPER COCO layout.

:func:`convert_pbr` turns selected BOP ``train_pbr`` frames into an unsplit dataset. It keeps
the BOP scene, frame and instance identities, uses the source RGB and visible-instance mask
bytes unchanged, and emits the pose schema of the renderers. BOP poses and depth are
millimeters in an OpenCV camera; emitted poses are meters with the OpenGL camera of the
renderers. ``model_offset_mm`` explicitly moves the keypoints into the BOP model frame; there
is no automatic centering or cross-dataset frame inference.
"""

from __future__ import annotations

import ctypes
import errno
import json
import logging
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from doper.bop.sources import IMAGE_SUFFIXES, sha256_file
from doper.scene import geometry, labels

LOGGER = logging.getLogger(__name__)
CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])
DEFAULT_MODEL_OFFSET_MM = (0.0, 0.0, 0.0)
DEFAULT_OCCLUSION_TOL_MM = 5.0
DEFAULT_MIN_VISIBLE_KEYPOINTS = 3
PROVENANCE_FORMAT = "doper-bop-pbr-conversion-v1"
ROTATION_ATOL = 1e-5
INTRINSICS_ATOL = 1e-8
MM_PER_M = 1000
RENAME_NOREPLACE = 1
AT_FDCWD = -100


@dataclass(frozen=True)
class ConvertSummary:
    """Outcome of :func:`convert_pbr`.

    Attributes:
        out: The published dataset directory.
        images: Number of emitted images.
        annotations: Number of emitted annotations.
    """

    out: Path
    images: int
    annotations: int


# --------------------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------------------
def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    """Load a required JSON object with input-specific diagnostics."""
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(f"missing {description}: {path}") from None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as err:
        raise ValueError(f"invalid {description} {path}: {err}") from err
    if not isinstance(payload, dict):
        raise ValueError(f"invalid {description} {path}: expected a JSON object")
    return payload


def _finite_array(value: Any, shape: tuple[int, ...], description: str) -> np.ndarray:  # noqa: ANN401
    """Convert a value to a finite float array of exactly ``shape``."""
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as err:
        raise ValueError(f"invalid {description}: expected numeric values") from err
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"invalid {description}: expected finite shape {shape}, got {array.shape}")
    return array


def load_keypoints(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    """Load and validate ordered, meter-scale keypoint metadata.

    Both the repository ``keypoints_3d.json`` layout and the keypoint selector's release JSON
    (which adds ``units``) are accepted. ``units``, when present, must be ``m``.

    Args:
        path: Keypoint JSON.

    Returns:
        The parsed payload and the ``(K, 3)`` keypoints in meters.

    Raises:
        ValueError: If the file is not a valid, finite, ordered keypoint layout in meters.
    """
    payload = _load_json_object(path, "keypoints metadata")
    names = payload.get("keypoint_names")
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(n, str) or not n for n in names)
    ):
        raise ValueError("invalid keypoint_names: expected a non-empty list of strings")
    if len(set(names)) != len(names):
        raise ValueError("invalid keypoint_names: names must be unique and ordered")
    if payload.get("units", "m") != "m":
        raise ValueError(f"invalid keypoints units {payload['units']!r}: expected 'm'")
    points = _finite_array(payload.get("keypoints_3d"), (len(names), 3), "keypoints_3d")
    count = payload.get("num_keypoints")
    if isinstance(count, bool) or not isinstance(count, int) or count != len(names):
        raise ValueError(f"invalid num_keypoints: expected {len(names)}, got {count!r}")
    if "object" in payload and (not isinstance(payload["object"], str) or not payload["object"]):
        raise ValueError("invalid keypoints object name")
    return payload, points


def _frame_number(frame_key: str, description: str) -> int:
    """Parse a non-negative BOP JSON frame key."""
    try:
        frame = int(frame_key)
    except (TypeError, ValueError) as err:
        raise ValueError(f"invalid frame key in {description}: {frame_key!r}") from err
    if frame < 0:
        raise ValueError(f"invalid frame key in {description}: {frame_key!r}")
    return frame


def _validate_intrinsics(cam: Any, description: str) -> tuple[np.ndarray, float]:  # noqa: ANN401
    """Validate zero-skew pinhole intrinsics and the required depth scale."""
    if not isinstance(cam, dict):
        raise ValueError(f"invalid calibration {description}: expected an object")
    k = _finite_array(cam.get("cam_K"), (9,), f"cam_K for {description}").reshape(3, 3)
    if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0.0, 0.0, 1.0], atol=INTRINSICS_ATOL):
        raise ValueError(
            f"invalid cam_K for {description}: expected positive focal lengths and [0,0,1] last row"
        )
    if not np.allclose([k[0, 1], k[1, 0]], [0.0, 0.0], atol=INTRINSICS_ATOL):
        raise ValueError(f"invalid cam_K for {description}: projection requires zero skew")
    scale = cam.get("depth_scale")
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not np.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError(
            f"invalid depth_scale for {description}: expected a positive finite number"
        )
    return k, float(scale)


def _validate_pose(gt: Any, description: str) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN401
    """Validate one finite BOP model-to-camera rigid pose."""
    if not isinstance(gt, dict):
        raise ValueError(f"invalid pose {description}: expected an object")
    rotation = _finite_array(gt.get("cam_R_m2c"), (9,), f"cam_R_m2c for {description}").reshape(
        3, 3
    )
    proper = np.allclose(rotation @ rotation.T, np.eye(3), atol=ROTATION_ATOL)
    if not proper or not np.isclose(np.linalg.det(rotation), 1.0, atol=ROTATION_ATOL):
        raise ValueError(f"invalid cam_R_m2c for {description}: expected a proper rotation")
    return rotation, _finite_array(gt.get("cam_t_m2c"), (3,), f"cam_t_m2c for {description}")


def bop_pose_to_meters(rotation: np.ndarray, translation_mm: np.ndarray) -> np.ndarray:
    """Return a 4x4 object-to-OpenCV-camera pose in meters."""
    pose = np.eye(4)
    pose[:3, :3] = _finite_array(rotation, (3, 3), "rotation")
    pose[:3, 3] = _finite_array(translation_mm, (3,), "translation") / MM_PER_M
    return pose


def _source_rgb(scene_dir: Path, frame: int) -> Path:
    """Resolve the frame's single required JPG or PNG RGB source."""
    candidates = [scene_dir / "rgb" / f"{frame:06d}{suffix}" for suffix in IMAGE_SUFFIXES]
    found = [path for path in candidates if path.is_file()]
    if not found:
        raise FileNotFoundError(f"missing rgb for {scene_dir.name} frame {frame}")
    if len(found) != 1:
        raise ValueError(f"invalid rgb for {scene_dir.name} frame {frame}: both JPG and PNG exist")
    return found[0]


def _read_depth_mm(scene_dir: Path, frame: int, size: tuple[int, int], scale: float) -> np.ndarray:
    """Read the uint16 depth map of a frame and scale it to millimeters."""
    width, height = size
    description = f"{scene_dir.name} frame {frame}"
    path = scene_dir / "depth" / f"{frame:06d}.png"
    if not path.is_file():
        raise FileNotFoundError(f"missing depth for {description}: {path}")
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.ndim != 2 or raw.shape != (height, width):
        raise ValueError(
            f"invalid depth for {description}: expected {width}x{height} single-channel image"
        )
    if raw.dtype != np.uint16:
        raise ValueError(f"invalid depth for {description}: expected uint16 dtype, got {raw.dtype}")
    return raw.astype(np.float32) * scale


def _check_mask(scene_dir: Path, frame: int, instance: int, size: tuple[int, int]) -> Path:
    """Validate the visible mask of one instance and return its path."""
    width, height = size
    description = f"{scene_dir.name} frame {frame} instance {instance}"
    path = scene_dir / "mask_visib" / f"{frame:06d}_{instance:06d}.png"
    if not path.is_file():
        raise FileNotFoundError(f"missing mask for {description}: {path}")
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.ndim != 2 or mask.shape != (height, width):
        raise ValueError(
            f"invalid mask for {description}: expected {width}x{height} single-channel image"
        )
    if mask.dtype != np.uint8:
        raise ValueError(f"invalid mask for {description}: expected uint8 dtype, got {mask.dtype}")
    return path


@dataclass(frozen=True)
class _FrameRequest:
    """Selection and labeling parameters shared by every frame of one conversion."""

    obj_id: int
    keypoints_m: np.ndarray
    tol_mm: float
    min_visible: int
    out_masks: Path


def _target_instances(
    scene_dir: Path, frame: int, obj_id: int, scene_gt: object
) -> list[tuple[int, dict]]:
    """Return ``(instance index, pose)`` of every instance of ``obj_id`` in a frame."""
    if not isinstance(scene_gt, list):
        raise ValueError(f"invalid scene_gt for {scene_dir.name} frame {frame}: expected a list")
    targets: list[tuple[int, dict]] = []
    for instance, gt in enumerate(scene_gt):
        value = gt.get("obj_id") if isinstance(gt, dict) else None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"invalid obj_id for {scene_dir.name} frame {frame} instance {instance}"
            )
        if value == obj_id:
            targets.append((instance, gt))
    return targets


def _frame_records(
    scene_dir: Path,
    frame_key: str,
    request: _FrameRequest,
    ids: tuple[int, int],
    scene_gt: Any,  # noqa: ANN401
    cam: Any,  # noqa: ANN401
) -> dict[str, Any] | None:
    """Build one image record and every usable target-instance annotation.

    Args:
        scene_dir: BOP scene directory.
        frame_key: Frame key of ``scene_gt.json``.
        request: Conversion parameters.
        ids: ``(image_id, first_annotation_id)`` to assign.
        scene_gt: The frame's ``scene_gt`` list.
        cam: The frame's ``scene_camera`` entry.

    Returns:
        ``image``, ``src`` and ``annotations``; None if the frame has no usable target.
    """
    frame = _frame_number(frame_key, f"scene {scene_dir.name}")
    targets = _target_instances(scene_dir, frame, request.obj_id, scene_gt)
    if not targets:
        return None
    rgb = _source_rgb(scene_dir, frame)
    header = cv2.imread(str(rgb), cv2.IMREAD_UNCHANGED)
    if header is None or header.ndim not in (2, 3):
        raise ValueError(f"invalid rgb for {scene_dir.name} frame {frame}: cannot decode {rgb}")
    height, width = header.shape[:2]
    k, depth_scale = _validate_intrinsics(cam, f"{scene_dir.name} frame {frame}")
    depth_mm = _read_depth_mm(scene_dir, frame, (width, height), depth_scale)
    intr = geometry.Intrinsics(width, height, k[0, 0], k[1, 1], k[0, 2], k[1, 2])
    validated = []
    for instance, gt in targets:
        description = f"{scene_dir.name} frame {frame} instance {instance}"
        rotation, translation = _validate_pose(gt, description)
        validated.append(
            (
                instance,
                rotation,
                translation,
                _check_mask(scene_dir, frame, instance, (width, height)),
            )
        )
    image_id, first_annotation_id = ids
    annotations: list[dict[str, Any]] = []
    for instance, rotation, translation, mask_src in validated:
        annotation = _annotation(
            scene_dir, frame, instance, (rotation, translation), mask_src, request, intr, depth_mm
        )
        if annotation is None:
            continue
        annotation["id"] = first_annotation_id + len(annotations)
        annotation["image_id"] = image_id
        annotations.append(annotation)
    if not annotations:
        return None
    file_name = f"{scene_dir.name}_{frame:06d}{rgb.suffix.lower()}"
    image = labels.image_record(
        image_id, file_name, width, height, pair_id=None, view="mono", camera_index=0
    )
    return {"image": image, "src": rgb, "annotations": annotations}


def _annotation(
    scene_dir: Path,
    frame: int,
    instance: int,
    pose: tuple[np.ndarray, np.ndarray],
    mask_src: Path,
    request: _FrameRequest,
    intr: geometry.Intrinsics,
    depth_mm: np.ndarray,
) -> dict[str, Any] | None:
    """Label one instance, copying its mask; None when too few keypoints are visible."""
    object_pose = bop_pose_to_meters(*pose)
    points_world = geometry.transform_points(object_pose, request.keypoints_m)
    uv, depth_m = geometry.project_points(points_world, CV_TO_GL, intr)
    if not np.isfinite(uv).all() or not np.isfinite(depth_m).all():
        raise ValueError(
            f"non-finite projection for {scene_dir.name} frame {frame} instance {instance}"
        )
    visibility = geometry.keypoint_visibility(
        uv, depth_m * MM_PER_M, depth_mm, intr, tol=request.tol_mm
    )
    if int(np.count_nonzero(visibility == 2)) < request.min_visible:
        return None
    mask_file = f"masks/{scene_dir.name}_{frame:06d}_{instance:06d}.png"
    shutil.copyfile(mask_src, request.out_masks.parent / mask_file)
    annotation = labels.annotation_record(
        0,
        uv=uv,
        vis=visibility,
        bbox=labels.keypoints_bbox(uv, visibility, intr.width, intr.height),
        intr=intr,
        cam_to_world=CV_TO_GL,
        object_pose=object_pose,
        view="mono",
        camera_index=0,
        mask_file=mask_file,
    )
    annotation["bop"] = {"scene": scene_dir.name, "frame": frame, "instance": instance}
    return annotation


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Publish a completed directory atomically without replacing any target."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace publication is unsupported: libc has no renameat2")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        AT_FDCWD, os.fsencode(source), AT_FDCWD, os.fsencode(destination), RENAME_NOREPLACE
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(f"output already exists: {destination}")
    unsupported = {errno.ENOSYS, errno.EINVAL, getattr(errno, "EOPNOTSUPP", errno.ENOSYS)}
    if error in unsupported:
        raise RuntimeError(
            f"atomic no-replace publication is unsupported for {destination}: {os.strerror(error)}"
        )
    raise OSError(error, os.strerror(error), str(destination))


@dataclass(frozen=True)
class _Options:
    """The validated options of one :func:`convert_pbr` call."""

    obj_id: int
    model_offset_mm: tuple[float, ...]
    occlusion_tol_mm: float
    min_visible_keypoints: int
    scenes: tuple[str, ...] | None
    max_frames: int | None
    copy_images: bool

    def validate(self, keypoint_count: int) -> None:
        """Reject bounds that would make selection or geometry ambiguous."""
        if self.obj_id <= 0:
            raise ValueError("obj-id must be positive")
        if not np.isfinite(self.occlusion_tol_mm) or self.occlusion_tol_mm < 0:
            raise ValueError("occlusion-tol-mm must be finite and non-negative")
        if not 0 <= self.min_visible_keypoints <= keypoint_count:
            raise ValueError(f"min-visible-keypoints must be between 0 and {keypoint_count}")
        if self.max_frames is not None and self.max_frames < 0:
            raise ValueError("max-frames must be non-negative")
        if not np.isfinite(np.asarray(self.model_offset_mm, dtype=float)).all():
            raise ValueError("model-offset-mm must be finite")


def _select_scenes(train_pbr: Path, scenes: Sequence[str] | None) -> dict[str, Path]:
    """Return the selected scene directories by name, in sorted order."""
    if not train_pbr.is_dir():
        raise ValueError(f"train-pbr is not a directory: {train_pbr}")
    available = {
        path.name: path
        for path in train_pbr.iterdir()
        if path.is_dir() and (path / "scene_gt.json").is_file()
    }
    if scenes is None:
        incomplete = sorted(
            path.name
            for path in train_pbr.iterdir()
            if path.is_dir() and path.name.isdigit() and not (path / "scene_gt.json").is_file()
        )
        if incomplete:
            raise ValueError(
                "numeric BOP scene directory missing scene_gt.json: " + ", ".join(incomplete)
            )
        names = sorted(available)
    else:
        if not scenes or len(set(scenes)) != len(scenes):
            raise ValueError("invalid requested scenes: provide one or more unique scene names")
        missing = [name for name in scenes if name not in available]
        if missing:
            raise ValueError(f"requested scene(s) not found: {', '.join(missing)}")
        names = sorted(scenes)
    if not names:
        raise ValueError("no BOP scenes selected")
    return {name: available[name] for name in names}


def _check_scene_gt(scene_name: str, scene_gt: Mapping[str, Any]) -> list[tuple[int, str]]:
    """Validate ``scene_gt`` keys and object ids; return ``(frame, key)`` pairs in frame order."""
    pairs = sorted(
        ((_frame_number(key, f"scene_gt {scene_name}"), key) for key in scene_gt),
        key=lambda p: p[0],
    )
    if len({frame for frame, _ in pairs}) != len(pairs):
        raise ValueError(f"invalid scene_gt for {scene_name}: duplicate numeric frame keys")
    for frame, key in pairs:
        entries = scene_gt[key]
        if not isinstance(entries, list):
            raise ValueError(f"invalid scene_gt for {scene_name} frame {frame}: expected a list")
        for instance, gt in enumerate(entries):
            value = gt.get("obj_id") if isinstance(gt, dict) else None
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"invalid obj_id for {scene_name} frame {frame} instance {instance}"
                )
    return pairs


def convert_pbr(
    train_pbr: Path,
    obj_id: int,
    keypoints: Path,
    out: Path,
    *,
    model_offset_mm: Sequence[float] = DEFAULT_MODEL_OFFSET_MM,
    occlusion_tol_mm: float = DEFAULT_OCCLUSION_TOL_MM,
    min_visible_keypoints: int = DEFAULT_MIN_VISIBLE_KEYPOINTS,
    scenes: Sequence[str] | None = None,
    max_frames: int | None = None,
    copy_images: bool = False,
) -> ConvertSummary:
    """Convert BOP ``train_pbr`` frames of one object to an unsplit DOPER COCO dataset.

    The output is staged in a sibling directory and published atomically; an existing ``out``
    is never replaced. Images are symlinked unless ``copy_images``.

    Args:
        train_pbr: BOP ``train_pbr`` directory (millimeters, OpenCV camera, uint16 depth).
        obj_id: BOP object id to convert.
        keypoints: Keypoint JSON in meters, in the model frame shifted by ``model_offset_mm``.
        out: New output directory.
        model_offset_mm: Offset added to the keypoints to move them into the BOP model frame.
        occlusion_tol_mm: Depth tolerance of the occlusion test.
        min_visible_keypoints: Minimum visible keypoints for an annotation to be kept.
        scenes: Scene names to convert (default: every scene).
        max_frames: Stop after this many frames (default: no limit).
        copy_images: Copy the RGB files instead of symlinking them.

    Returns:
        The summary.

    Raises:
        FileExistsError: If ``out`` exists.
        ValueError: If any input is invalid or no frame is emitted.
        FileNotFoundError: If a required file is missing.
    """
    options = _Options(
        obj_id=obj_id,
        model_offset_mm=tuple(float(v) for v in model_offset_mm),
        occlusion_tol_mm=occlusion_tol_mm,
        min_visible_keypoints=min_visible_keypoints,
        scenes=None if scenes is None else tuple(scenes),
        max_frames=max_frames,
        copy_images=copy_images,
    )
    out = Path(out).absolute()
    if os.path.lexists(out):
        raise FileExistsError(f"output already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = Path(tempfile.mkdtemp(prefix=f".{out.name}.tmp-", dir=out.parent))
    try:
        images, annotations = _convert(Path(train_pbr), Path(keypoints), temporary, options)
        _rename_noreplace(temporary, out)
        temporary = None
    finally:
        if temporary is not None:
            shutil.rmtree(temporary)
    return ConvertSummary(out=out, images=images, annotations=annotations)


@dataclass
class _Emitted:
    """Everything emitted so far by one conversion."""

    images: list[dict[str, Any]]
    annotations: list[dict[str, Any]]
    intrinsics: set[str]
    metadata: dict[str, dict[str, str]]
    frames: list[dict[str, Any]]
    scenes: list[str]


def _convert(train_pbr: Path, keypoints: Path, output: Path, options: _Options) -> tuple[int, int]:
    """Validate the inputs and build a complete dataset inside ``output``."""
    payload, keypoints_m = load_keypoints(keypoints)
    options.validate(len(keypoints_m))
    keypoints_m = keypoints_m + np.asarray(options.model_offset_mm, dtype=float) / MM_PER_M
    selected = _select_scenes(train_pbr, options.scenes)
    out_images, out_masks = output / "images", output / "masks"
    out_images.mkdir(parents=True)
    out_masks.mkdir()
    request = _FrameRequest(
        options.obj_id,
        keypoints_m,
        options.occlusion_tol_mm,
        options.min_visible_keypoints,
        out_masks,
    )
    emitted = _Emitted([], [], set(), {}, [], list(selected))
    limit = options.max_frames or 0
    for scene_dir in selected.values():
        _convert_scene(scene_dir, train_pbr, request, options, out_images, emitted, limit)
        if limit and len(emitted.images) >= limit:
            break
    if not emitted.images:
        raise ValueError(
            f"no frames emitted for object {options.obj_id} with the requested selection"
        )
    _write_outputs(output, train_pbr, keypoints, payload, keypoints_m, options, emitted)
    return len(emitted.images), len(emitted.annotations)


def _convert_scene(
    scene_dir: Path,
    train_pbr: Path,
    request: _FrameRequest,
    options: _Options,
    out_images: Path,
    emitted: _Emitted,
    limit: int,
) -> None:
    """Convert the target frames of one scene into ``emitted``, stopping at ``limit`` images."""
    name = scene_dir.name
    gt_path, camera_path = scene_dir / "scene_gt.json", scene_dir / "scene_camera.json"
    scene_gt = _load_json_object(gt_path, f"scene_gt metadata for scene {name}")
    scene_camera = _load_json_object(camera_path, f"scene_camera metadata for scene {name}")
    emitted.metadata[name] = {
        "scene_gt.json": sha256_file(gt_path),
        "scene_camera.json": sha256_file(camera_path),
    }
    for frame, frame_key in _check_scene_gt(name, scene_gt):
        entries = scene_gt[frame_key]
        if not any(gt["obj_id"] == options.obj_id for gt in entries):
            continue
        if frame_key not in scene_camera:
            raise ValueError(f"missing calibration for {name} frame {frame}")
        record = _frame_records(
            scene_dir,
            frame_key,
            request,
            (len(emitted.images), len(emitted.annotations)),
            entries,
            scene_camera[frame_key],
        )
        if record is None:
            continue
        destination = out_images / record["image"]["file_name"]
        if options.copy_images:
            shutil.copyfile(record["src"], destination)
        else:
            destination.symlink_to(os.path.relpath(record["src"], out_images))
        emitted.images.append(record["image"])
        emitted.annotations.extend(record["annotations"])
        cam_k = scene_camera[frame_key]["cam_K"]
        emitted.intrinsics.add(json.dumps(cam_k, separators=(",", ":"), allow_nan=False))
        emitted.frames.append(
            {
                "scene": name,
                "frame": frame,
                "rgb": str(record["src"].relative_to(train_pbr)),
                "rgb_sha256": sha256_file(record["src"]),
                "target_instances": [a["bop"]["instance"] for a in record["annotations"]],
            }
        )
        if limit and len(emitted.images) >= limit:
            return


def _write_outputs(
    output: Path,
    train_pbr: Path,
    keypoints: Path,
    payload: dict[str, Any],
    keypoints_m: np.ndarray,
    options: _Options,
    emitted: _Emitted,
) -> None:
    """Write ``keypoints_3d.json``, ``annotations.json`` and the conversion provenance."""
    object_name = payload.get("object", f"obj_{options.obj_id:06d}")
    names = list(payload["keypoint_names"])
    (output / "keypoints_3d.json").write_text(
        json.dumps(
            labels.keypoints_payload(object_name, names, keypoints_m.tolist()),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    flags = np.concatenate([np.asarray(a["keypoints"])[2::3] for a in emitted.annotations])
    merged = {
        "info": {
            "description": f"BOP train_pbr obj {options.obj_id} - {len(emitted.images)} images",
            "version": "1.0",
            "source": str(train_pbr),
            "num_samples": len(emitted.images),
            "images_per_sample": 1,
            "stereo": None,
            "render_masks": True,
            "mask_dir": "masks",
            "distinct_intrinsics": len(emitted.intrinsics),
            "keypoint_visibility": {
                "visible": int(np.count_nonzero(flags == 2)),
                "occluded": int(np.count_nonzero(flags == 1)),
            },
            "conversion_provenance": "conversion_provenance.json",
        },
        "categories": [
            {"id": 1, "name": payload.get("object", "object"), "keypoints": names, "skeleton": []}
        ],
        "images": emitted.images,
        "annotations": emitted.annotations,
    }
    (output / "annotations.json").write_text(json.dumps(merged, allow_nan=False) + "\n")
    provenance = {
        "format": PROVENANCE_FORMAT,
        "source": {"train_pbr": str(train_pbr.resolve()), "object_id": options.obj_id},
        "source_metadata": emitted.metadata,
        "keypoints": {
            "input": str(keypoints.resolve()),
            "sha256": sha256_file(keypoints),
            "object": object_name,
            "names": names,
        },
        "selection": {
            "requested_scenes": None if options.scenes is None else list(options.scenes),
            "selected_scenes": emitted.scenes,
            "max_frames": options.max_frames or 0,
        },
        "options": {
            "model_offset_mm": list(options.model_offset_mm),
            "occlusion_tol_mm": options.occlusion_tol_mm,
            "min_visible_keypoints": options.min_visible_keypoints,
            "copy_images": options.copy_images,
        },
        "emitted_frames": emitted.frames,
    }
    (output / "conversion_provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n"
    )
