# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent audit and visible-count canonicalization of one rendered BlenderProc layout.

The audit reads a layout in its published form (see ``campaign.assemble_layout``): the BOP scene
files at the top (``rgb/``, ``depth/``, ``mask/``, ``mask_visib/``, ``scene_*.json``) and the
render records under ``records/``. It checks, without trusting the BOP writer:

* the RGB/depth shape and type, the 20-instance population of every frame, the camera K, depth
  scale and world-to-camera frame;
* every object pose against the saved Blender world matrices (rotation entries 1e-4, translation
  0.1 mm, so the writer's units are millimeters) and the target keypoints reprojected through
  both poses (0.1 px);
* each mask pair: binary, visible pixels inside the amodal mask.

BlenderProc rasterizes ``scene_gt_info`` and the mask PNGs separately, so their pixel counts can
disagree at edges. The audit rewrites only the derived fields of ``scene_gt_info.json``
(``px_count_visib``, ``px_count_valid``, ``visib_fract``, ``bbox_visib`` and ``px_count_all``
bounded below by the saved amodal mask) from the saved PNGs. A correction larger than
``config.count_bound_px`` raises :class:`CountBoundError`; the campaign archives the attempt and
records the layout as failed (see ``campaign``). The raw writer file is kept under
``records/writer_raw/``.

Layouts whose target is never visible are retained and reported
(``zero_visible_layout_retained``), as the HOT3D obj33 source did. The HANDAL obj40 source audit
raised on them instead; both campaigns now share the retaining policy.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from doper.bop.sources import publish_json, sha256_file
from rendering.blenderproc.plan import TARGET_INSTANCES, RenderConfig

RECORDS = "records"
AUDITED_NAME = "AUDITED.json"
CANONICALIZATION_NAME = "visible_count_canonicalization.json"
KEYPOINT_COUNT = 64
MAX_ROTATION_ERROR = 1e-4
MAX_TRANSLATION_ERROR_MM = 0.1
MAX_REPROJECTION_ERROR_PX = 0.1
BLENDER_TO_OPENCV = np.diag([1, -1, -1, 1])


class CountBoundError(ValueError):
    """A writer pixel count differs from the saved masks by more than the configured bound."""


@dataclass(frozen=True)
class AuditResult:
    """Outcome of :func:`audit_layout` (also written to ``records/AUDITED.json``)."""

    ok: bool
    dataset: str
    object_id: int
    frames: int
    image_size: list[int]
    target_positive_frames: int
    target_visibility: list[float]
    target_bbox_sizes: list[list[int]]
    zero_visible_layout_retained: bool
    max_rotation_error: float
    max_translation_error_mm: float
    max_keypoint_reprojection_error_px: float | None
    projectable_target_frames: int
    keypoints_sha256: str
    max_count_correction: int
    count_bound_px: int
    render_seconds: float
    total_seconds: float
    bop_bytes: int
    files: dict[str, dict[str, Any]]
    annotation_units: str = "mm"
    depth_scale_mm_per_unit: float = 1.0


class _Reader:
    """Decode images from bytes while recording their SHA-256 and size."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.files: dict[str, dict[str, Any]] = {}

    def image(self, path: Path, flags: int) -> np.ndarray | None:
        raw = path.read_bytes()
        rel = str(path.relative_to(self.root))
        self.files[rel] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), flags)


def canonical_counts(
    counts: dict[str, Any], mask: np.ndarray, visible_mask: np.ndarray, depth: np.ndarray
) -> tuple[dict[str, Any], int]:
    """Derive one instance's BOP visibility fields from its saved masks.

    Args:
        counts: The writer's ``scene_gt_info`` row.
        mask: Saved amodal mask.
        visible_mask: Saved visible mask.
        depth: Saved depth image.

    Returns:
        The canonical row and the largest absolute count correction.
    """
    if not set(np.unique(mask)) <= {0, 255} or not set(np.unique(visible_mask)) <= {0, 255}:
        raise ValueError("non-binary mask")
    m = mask > 0
    v = visible_mask > 0
    if np.any(v & ~m):
        raise ValueError("visible pixels outside the amodal mask")
    visible = int(v.sum())
    valid = int((m & (depth > 0)).sum())
    all_pixels = max(counts["px_count_all"], int(m.sum()))
    if not 0 <= visible <= valid <= all_pixels:
        raise ValueError("invalid visible/valid/all pixel ordering")
    delta = max(
        abs(visible - counts["px_count_visib"]),
        abs(valid - counts["px_count_valid"]),
        abs(all_pixels - counts["px_count_all"]),
    )
    x, y, w, h = cv2.boundingRect(visible_mask)
    bbox = [x, y, w - 1, h - 1] if visible else [-1, -1, -1, -1]
    after = {
        **counts,
        "px_count_all": all_pixels,
        "px_count_valid": valid,
        "px_count_visib": visible,
        "bbox_visib": bbox,
        "visib_fract": visible / all_pixels if all_pixels else 0.0,
    }
    return after, delta


def _project(points_mm: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:  # noqa: N803
    return (R @ points_mm.T).T + t


def _overlay(image: np.ndarray, visible_mask: np.ndarray, uv: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    contours, _ = cv2.findContours(visible_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    for point in uv:
        x, y = np.rint(point).astype(int)
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(overlay, (int(x), int(y)), 3, (0, 0, 255), -1)
    return overlay


def _check_contract(scene: dict[str, Any], config: RenderConfig) -> list[int]:
    if (scene["dataset"], scene["target_id"]) != (config.dataset, config.target_object):
        raise ValueError("layout dataset/target differs from the render config")
    if scene["frames"] != config.frames_per_layout:
        raise ValueError(f"layout has {scene['frames']} views, expected {config.frames_per_layout}")
    target_ids = scene["target_ids"]
    if len(set(target_ids)) != TARGET_INSTANCES or config.target_object not in target_ids:
        raise ValueError("wrong target population")
    if list(scene["image_size"]) != [config.camera.width, config.camera.height]:
        raise ValueError("layout image size differs from the render config")
    return target_ids


def _check_camera(camera: dict[str, Any], scene: dict[str, Any], w2cv: np.ndarray) -> np.ndarray:
    K = np.asarray(camera["cam_K"]).reshape(3, 3)  # noqa: N806
    if not np.allclose(K, np.asarray(scene["cam_K"]).reshape(3, 3), atol=1e-4, rtol=1e-7):
        raise ValueError("camera K changed")
    if camera["depth_scale"] != 1:
        raise ValueError("depth scale is not 1 mm per unit")
    if "cam_R_w2c" in camera and not np.allclose(
        np.asarray(camera["cam_R_w2c"]).reshape(3, 3), w2cv[:3, :3], atol=1e-4
    ):
        raise ValueError("camera frame mismatch")
    if (
        "cam_t_w2c" in camera
        and np.linalg.norm(np.asarray(camera["cam_t_w2c"]) - w2cv[:3, 3] * 1000)
        > MAX_TRANSLATION_ERROR_MM
    ):
        raise ValueError("camera translation units mismatch")
    return K


def audit_layout(layout_dir: Path, config: RenderConfig, keypoints: Path) -> AuditResult:
    """Audit one fresh layout and canonicalize its visible counts in place.

    Args:
        layout_dir: Layout in published form (BOP scene files plus ``records/``).
        config: Render config of the campaign.
        keypoints: Frozen ``keypoints_3d.json`` of the target (64 points, meters).

    Returns:
        The audit result, also written to ``records/AUDITED.json``.

    Raises:
        CountBoundError: A count correction exceeds ``config.count_bound_px``.
        ValueError: Any other check fails, or the layout was already audited.
    """
    root = Path(layout_dir)
    records = root / RECORDS
    if (records / AUDITED_NAME).exists():
        raise ValueError("fresh layout audit only; reuse the published receipt instead")
    scene = json.loads((records / "scene_record.json").read_text())
    receipt = json.loads((records / "render_receipt.json").read_text())
    target_ids = _check_contract(scene, config)
    frames = scene["frames"]
    gt = json.loads((root / "scene_gt.json").read_text())
    cameras = json.loads((root / "scene_camera.json").read_text())
    info_path = root / "scene_gt_info.json"
    raw_info = info_path.read_bytes()
    info = json.loads(raw_info)
    expected = {str(i) for i in range(frames)}
    if set(gt) != expected or set(cameras) != expected or set(info) != expected:
        raise ValueError("incomplete BOP population")
    width, height = config.camera.width, config.camera.height
    world = {
        r["id"]: np.asarray(r["matrix"])
        for r in scene["object_world_poses_m"]
        if r["dataset"] == config.dataset
    }
    points = np.asarray(json.loads(Path(keypoints).read_text())["keypoints_3d"])
    if points.shape != (KEYPOINT_COUNT, 3) or not np.isfinite(points).all():
        raise ValueError("bad frozen keypoints")
    reader = _Reader(root)
    rot_err: list[float] = []
    trans_err: list[float] = []
    pixel_err: list[float] = []
    visibility: list[float] = []
    boxes: list[list[int]] = []
    changes: list[dict[str, Any]] = []
    max_delta = 0
    for frame in range(frames):
        key = str(frame)
        image = reader.image(root / "rgb" / f"{frame:06d}.jpg", cv2.IMREAD_COLOR)
        depth = reader.image(root / "depth" / f"{frame:06d}.png", cv2.IMREAD_UNCHANGED)
        if (
            image is None
            or image.shape[:2] != (height, width)
            or depth is None
            or depth.shape != (height, width)
            or depth.dtype != np.uint16
        ):
            raise ValueError("bad RGB/depth shape/type")
        if (
            len(gt[key]) != TARGET_INSTANCES
            or len(info[key]) != TARGET_INSTANCES
            or sorted(r["obj_id"] for r in gt[key]) != sorted(target_ids)
        ):
            raise ValueError("instance population changed")
        c2w = np.asarray(scene["camera_world_poses_m"][frame])
        w2cv = np.linalg.inv(c2w @ BLENDER_TO_OPENCV)
        K = _check_camera(cameras[key], scene, w2cv)  # noqa: N806
        for index, (pose, counts) in enumerate(zip(gt[key], info[key], strict=True)):
            H = world[pose["obj_id"]]  # noqa: N806
            scale = np.linalg.norm(H[:3, :3], axis=0)
            if not np.allclose(scale, scale[0], rtol=1e-5):
                raise ValueError("anisotropic scale")
            R = w2cv[:3, :3] @ (H[:3, :3] / scale)  # noqa: N806
            t = (w2cv[:3, :3] @ H[:3, 3] + w2cv[:3, 3]) * 1000
            stored_R = np.asarray(pose["cam_R_m2c"]).reshape(3, 3)  # noqa: N806
            stored_t = np.asarray(pose["cam_t_m2c"])
            rot_err.append(float(np.abs(R - stored_R).max()))
            trans_err.append(float(np.linalg.norm(t - stored_t)))
            name = f"{frame:06d}_{index:06d}.png"
            mask = reader.image(root / "mask" / name, cv2.IMREAD_GRAYSCALE)
            vis = reader.image(root / "mask_visib" / name, cv2.IMREAD_GRAYSCALE)
            if (
                mask is None
                or vis is None
                or mask.shape != (height, width)
                or vis.shape != mask.shape
            ):
                raise ValueError("missing/wrong-size mask")
            after, delta = canonical_counts(counts, mask, vis, depth)
            max_delta = max(max_delta, delta)
            if delta > config.count_bound_px:
                raise CountBoundError(
                    f"writer count differs from the saved masks by {delta} px (frame {frame}, "
                    f"instance {index}); bound {config.count_bound_px} px"
                )
            if after != counts:
                changes.append(
                    {"frame": frame, "instance": index, "before": dict(counts), "after": after}
                )
            info[key][index] = after
            if pose["obj_id"] != config.target_object:
                continue
            a = _project(points * 1000, R, t)
            b = _project(points * 1000, stored_R, stored_t)
            good = (a[:, 2] > 0) & (b[:, 2] > 0)
            ua = a[good] @ K.T
            ub = b[good] @ K.T
            if good.any():
                errors = np.linalg.norm(ua[:, :2] / ua[:, 2:] - ub[:, :2] / ub[:, 2:], axis=1)
                pixel_err.append(float(errors.max()))
            visibility.append(after["visib_fract"])
            boxes.append(after["bbox_obj"][2:])
            if frame in config.overlay_frames:
                overlay = _overlay(image, vis, ub[:, :2] / ub[:, 2:])
                (records / "audit_overlays").mkdir(exist_ok=True)
                cv2.imwrite(str(records / "audit_overlays" / f"{frame:06d}.jpg"), overlay)
    if (
        max(rot_err) > MAX_ROTATION_ERROR
        or max(trans_err) > MAX_TRANSLATION_ERROR_MM
        or (pixel_err and max(pixel_err) > MAX_REPROJECTION_ERROR_PX)
    ):
        raise ValueError("independent geometry/units/projection audit failed")
    raw_copy = records / "writer_raw" / "scene_gt_info.json"
    raw_copy.parent.mkdir(parents=True, exist_ok=True)
    raw_copy.write_bytes(raw_info)
    publish_json(info_path, info)
    publish_json(
        records / CANONICALIZATION_NAME,
        {
            "method": "exact saved-mask visible/valid counts and visible box; amodal area "
            "bounded below by the saved mask; raw writer info retained",
            "max_allowed_count_correction": config.count_bound_px,
            "max_observed_count_correction": max_delta,
            "changes": changes,
            "raw_sha256": sha256_file(raw_copy),
            "canonical_sha256": sha256_file(info_path),
        },
    )
    files = reader.files
    for name in ("scene_gt.json", "scene_camera.json", "scene_gt_info.json"):
        path = root / name
        files[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    result = AuditResult(
        ok=True,
        dataset=config.dataset,
        object_id=config.target_object,
        frames=frames,
        image_size=[width, height],
        target_positive_frames=sum(v > 0 for v in visibility),
        target_visibility=visibility,
        target_bbox_sizes=boxes,
        zero_visible_layout_retained=not any(v > 0 for v in visibility),
        max_rotation_error=max(rot_err),
        max_translation_error_mm=max(trans_err),
        max_keypoint_reprojection_error_px=max(pixel_err) if pixel_err else None,
        projectable_target_frames=len(pixel_err),
        keypoints_sha256=sha256_file(Path(keypoints)),
        max_count_correction=max_delta,
        count_bound_px=config.count_bound_px,
        render_seconds=receipt["render_seconds"],
        total_seconds=receipt["total_seconds"],
        bop_bytes=sum(v["bytes"] for v in files.values()),
        files=files,
    )
    publish_json(records / AUDITED_NAME, asdict(result))
    return result
