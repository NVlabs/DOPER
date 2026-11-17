# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T-LESS three-sensor TRAIN supervision: per-sensor COCO roots, projected masks, review gate.

T-LESS captured every TRAIN turntable session with three sensors. :func:`prepare` turns one
object's PrimeSense (BOP release, ``<root>/data/train_primesense``), Kinect and Canon (original
v2 release, ``<root>/sensors``, see ``doper.bop.tless_sensors``) captures into three COCO roots
the multisensor adaptation mixes with PBR::

    <root>/multisensor/obj_XXXXXX/
        <sensor>/train/{images/, masks/, annotations.json}   sensor: primesense, kinect, canon
        <sensor>/{keypoints_3d.json, SOURCE.json}
        <sensor>/review/<frame>.jpg                           first, middle and last capture
        READY.json, REVIEWED.json (``doper review``)

Supervision is the released ground truth: the 64 keypoints are projected with the released pose
and each frame's own camera matrix (visibility = inside the image). PrimeSense keeps its released
``mask_visib``; Kinect and Canon, which ship no masks, get the amodal mask of the audited CAD
model projected with the released pose (:func:`projected_cad_mask`, holes preserved, no
occlusion removed). The box is the mask's tight box.

Guards: every RGB image whose bytes equal a reserved ``test_primesense`` image is refused,
naming the frame; EXIF orientation must be 1; camera matrices may have no skew; the three
sensors' RGB populations must differ (one camera relabeled as three is refused); the capture
ids must agree across sensors; paths of test or validation data are refused; outputs are
new-only (staged, then renamed). The nine overlays must be reviewed by a person
(:func:`approve`, ``doper review tless``) before :func:`verify` lets training use the root.
Images are relative links and every recorded path is relative to the dataset root, so a moved
root still verifies.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np

from . import review, tless_sensors
from .convert import CV_TO_GL, _validate_pose, bop_pose_to_meters, load_keypoints
from .extract import data_dir
from .population import Population, build_population
from .sources import IMAGE_SUFFIXES, canonical_sha256, link_relative, publish_json, sha256_file
from .tless_sensors import read_metadata, sensor_dir

SENSORS = ("primesense", "kinect", "canon")
PRIMESENSE_DIR = "train_primesense"
TEST_SPLIT = "test_primesense"
MULTISENSOR_DIR = "multisensor"
READY_NAME = "READY.json"
SOURCE_NAME = "SOURCE.json"
KEYPOINTS_NAME = "keypoints_3d.json"
REVIEW_DIR = "review"
KEYPOINT_COUNT = 64
SUPERVISION = "gt-keypoints"
MASK_POLICY = "projected"
PROVIDED_MASK_TYPE = "provided_foreground_visibility_unasserted"
PROJECTED_MASK_TYPE = "gt_projected_amodal"
EVALUATION_PARTS = frozenset(
    {"val", "test", "validation", "test_primesense", "test_kinect", "test_canon"}
)
EXIF_ORIENTATION = 274
MAX_PIXEL = 1e7
OVERLAY_MAX_SIDE = 1200
MM_PER_M = 1000.0
MASK_VALUES = frozenset({0, 1, 255})
SOURCE_FORMAT = (
    "original v2 YAML or BOP JSON; per-frame K; original CAD pose mm; ordered landmarks m"
)


class CadMesh(NamedTuple):
    """CAD triangles in the original model frame (millimeters)."""

    vertices: np.ndarray
    faces: np.ndarray


@dataclass(frozen=True)
class MultisensorSummary:
    """What :func:`prepare` wrote.

    Attributes:
        root: The prepared object root (``<root>/multisensor/obj_XXXXXX``).
        frames: Captures per sensor.
        overlays: The review overlays, three per sensor.
        ready: The ``READY.json`` receipt.
    """

    root: Path
    frames: Mapping[str, int]
    overlays: tuple[Path, ...]
    ready: Path


def output_dir(root: Path, obj_id: int) -> Path:
    """The prepared object root, ``<root>/multisensor/obj_XXXXXX``."""
    return Path(root) / MULTISENSOR_DIR / f"obj_{obj_id:06d}"


def require_training_path(path: Path) -> None:
    """Refuse a file that resolves into evaluation data (a ``val``/``test`` directory).

    Raises:
        ValueError: If a resolved path component names an evaluation split.
    """
    if any(part.lower() in EVALUATION_PARTS for part in Path(path).resolve().parts):
        raise ValueError(f"training file {path} resolves into evaluation data")


def source_dirs(root: Path, obj_id: int) -> dict[str, Path]:
    """The TRAIN object directory of every sensor (one directory whose number is the object).

    Raises:
        ValueError: If a sensor has no or several directories of the object, or two sensors
            resolve to the same directory.
    """
    bases = {"primesense": data_dir(root) / PRIMESENSE_DIR}
    bases.update({sensor: sensor_dir(root, sensor) for sensor in SENSORS[1:]})
    found = {}
    for sensor, base in bases.items():
        matches = (
            [
                p
                for p in base.iterdir()
                if p.is_dir() and p.name.isdecimal() and int(p.name) == obj_id
            ]
            if base.is_dir()
            else []
        )
        if len(matches) != 1:
            raise ValueError(f"exactly one {sensor} TRAIN directory of object {obj_id} in {base}")
        require_training_path(matches[0])
        found[sensor] = matches[0]
    if len({p.resolve() for p in found.values()}) != len(found):
        raise ValueError("different sensors resolve to the same source directory")
    return found


def reserved_population(test_dir: Path) -> Population:
    """The reserved ``test_primesense`` population no training image may equal.

    Args:
        test_dir: The extracted BOP ``test_primesense`` split.

    Returns:
        ``population.build_population`` of the split, flagged held-out reporting only.
    """
    return {
        **build_population(test_dir),
        "split": TEST_SPLIT,
        "role": "held_out_reporting_only",
        "adaptation_allowed": False,
        "checkpoint_selection": False,
    }


def population_sha256(population: Population) -> str:
    """Identity of a population: as ``population.load_population`` hashes its directory.

    The equality holds for the directory form only (``build_population`` output, as
    :func:`reserved_population` returns); a manifest file is hashed by its bytes instead.
    """
    return canonical_sha256({k: population[k] for k in ("metadata", "frames")})


def _reserved_hashes(reserved: Population) -> dict[str, str]:
    if reserved.get("split") != TEST_SPLIT or reserved.get("adaptation_allowed") is not False:
        raise ValueError("an explicit held-out test_primesense population is required")
    return {f["sha256"]: f"{f['scene']}/{f['frame']}" for f in reserved["frames"]}


def load_audited_cad(mesh_path: Path, audit_path: Path, keypoints: Path) -> CadMesh:
    """Load the CAD model the keypoints were audited against (``prepare``'s ``audit.json``).

    Raises:
        ValueError: If the audit is not of this mesh and these keypoints in mm/m, or the mesh
            is not a finite triangle mesh.
    """
    from .audit import load_bop_mesh

    audit = json.loads(Path(audit_path).read_text())
    audit = audit.get("geometry", audit)
    if (
        audit.get("mesh_units") != "mm"
        or audit.get("keypoint_units") != "m"
        or audit.get("mesh_sha256") != sha256_file(mesh_path)
        or audit.get("keypoints_sha256") != sha256_file(keypoints)
    ):
        raise ValueError("CAD and keypoints must be the audited mm/m pair exactly")
    mesh = load_bop_mesh(mesh_path)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not np.isfinite(vertices).all()
        or faces.ndim != 2
        or faces.shape[1] != 3
        or not len(faces)
    ):
        raise ValueError("a finite triangular CAD model in its original frame is required")
    return CadMesh(vertices, faces)


def projected_cad_mask(
    mesh: CadMesh,
    camera_matrix: np.ndarray,
    rotation: np.ndarray,
    translation_mm: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Amodal mask: the union of the projected CAD triangles, geometric holes preserved.

    Each triangle is filled on its own (``cv2.fillConvexPoly``), so overlapping triangles add up
    instead of canceling; there is no hull fill, hole fill, dilation or occlusion removal.

    Args:
        mesh: CAD triangles (mm).
        camera_matrix: 3x3 matrix of this image.
        rotation: Model-to-camera rotation.
        translation_mm: Model-to-camera translation (mm).
        shape: ``(height, width)`` of the image.

    Returns:
        A uint8 mask, 255 on the object.

    Raises:
        ValueError: If the model crosses or lies behind the camera, the projection is not
            finite, or the mask is empty.
    """
    xyz = mesh.vertices @ np.asarray(rotation).T + np.asarray(translation_mm).reshape(3)
    if not np.isfinite(xyz).all() or np.any(xyz[:, 2] <= 0):
        raise ValueError("CAD crosses or lies behind the camera; inspect frame and units")
    projected = xyz @ np.asarray(camera_matrix).T
    uv = projected[:, :2] / projected[:, 2:]
    if not np.isfinite(uv).all() or np.abs(uv).max() > MAX_PIXEL:
        raise ValueError("invalid CAD pixel projection")
    polygons = np.rint(uv[mesh.faces]).astype(np.int32)
    mask = np.zeros(shape, np.uint8)
    for triangle in polygons:
        cv2.fillConvexPoly(mask, triangle, 255, lineType=cv2.LINE_8)
    if not mask.any():
        raise ValueError("GT CAD projection has no foreground")
    return mask


def _camera_matrix(camera: Mapping[str, Any]) -> np.ndarray:
    matrix = np.asarray(camera["cam_K"], dtype=float).reshape(3, 3)
    if (
        not np.isfinite(matrix).all()
        or min(matrix[0, 0], matrix[1, 1]) <= 0
        or not np.allclose(matrix[2], [0, 0, 1])
        or not np.allclose([matrix[0, 1], matrix[1, 0]], [0, 0])
    ):
        raise ValueError("invalid or unsupported skew in the per-frame camera matrix")
    return matrix


def _project_annotation(
    gt: Mapping[str, Any],
    matrix: np.ndarray,
    points_m: np.ndarray,
    size: tuple[int, int],
    image_id: int,
    bbox: Sequence[float],
    mask_file: str | None,
) -> tuple[dict[str, Any], np.ndarray]:
    """The COCO annotation of the keypoints projected with the released pose."""
    from ..scene.geometry import Intrinsics
    from ..scene.labels import annotation_record

    width, height = size
    rotation, translation = _validate_pose(gt, "released T-LESS training pose")
    xyz = points_m @ rotation.T + translation.reshape(3) / MM_PER_M
    if not np.isfinite(xyz).all() or np.any(xyz[:, 2] <= 0):
        raise ValueError("invalid released-pose projection depth")
    homogeneous = xyz @ matrix.T
    uv = homogeneous[:, :2] / homogeneous[:, 2:]
    valid = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    if not valid.any():
        raise ValueError("no keypoint projects inside the training RGB; inspect K/frame/units")
    intrinsics = Intrinsics(width, height, matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2])
    annotation = annotation_record(
        image_id,
        uv=uv,
        vis=valid.astype(int),
        bbox=bbox,
        intr=intrinsics,
        cam_to_world=CV_TO_GL,
        object_pose=bop_pose_to_meters(rotation, translation),
        view="mono",
        camera_index=0,
        mask_file=mask_file,
    )
    if mask_file:
        annotation["mask_type"] = PROVIDED_MASK_TYPE
    return annotation, uv


@dataclass
class _SensorSource:
    """One sensor's TRAIN captures, read and cross-checked."""

    sensor: str
    folder: Path
    gt: dict[int, Any]
    gt_path: Path
    cameras: dict[int, Any]
    camera_path: Path
    images: dict[int, Path]
    masks: dict[tuple[int, int], Path]


def _read_source(sensor: str, folder: Path, expected: int) -> _SensorSource:
    gt, gt_path = read_metadata(folder, "scene_gt.json", "gt.yml")
    cameras, camera_path = read_metadata(folder, "scene_camera.json", "info.yml")
    for path in (gt_path, camera_path):
        require_training_path(path)
    images: dict[int, Path] = {}
    for path in (folder / "rgb").iterdir():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if not path.stem.isdecimal() or int(path.stem) in images:
            raise ValueError(f"ambiguous RGB frame name {path}")
        images[int(path.stem)] = path
    if set(gt) != set(cameras) or set(gt) != set(images):
        raise ValueError(f"{sensor}: RGB, GT and per-frame calibration populations differ")
    if len(gt) != expected:
        raise ValueError(f"{sensor}: expected all {expected} released TRAIN frames, got {len(gt)}")
    masks: dict[tuple[int, int], Path] = {}
    if (folder / "mask_visib").is_dir():
        for path in (folder / "mask_visib").glob("*.png"):
            pieces = path.stem.split("_")
            if len(pieces) == 2 and all(p.isdecimal() for p in pieces):
                key = (int(pieces[0]), int(pieces[1]))
                if key in masks:
                    raise ValueError(f"duplicate instance mask {path}")
                masks[key] = path
    return _SensorSource(sensor, folder, gt, gt_path, cameras, camera_path, images, masks)


@dataclass
class _Frame:
    """The labels of one capture."""

    image: np.ndarray
    bbox: Any  # noqa: ANN401 - released box, list or None until resolved
    mask: np.ndarray | None
    mask_file: str | None
    mask_origin: str
    source_mask: Path | None


def _decode(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as original:
        if original.getexif().get(EXIF_ORIENTATION, 1) != 1:
            raise ValueError(
                f"{path}: EXIF rotation must be resolved with matching calibration first"
            )
    image = cv2.imread(str(path))
    if image is None or not image.shape[0] or not image.shape[1]:
        raise ValueError(f"cannot decode real RGB {path}")
    return image


def _frame_mask(
    source: _SensorSource,
    frame: int,
    pose: Mapping[str, Any],
    image: np.ndarray,
    matrix: np.ndarray,
    mesh: CadMesh,
    train: Path,
) -> _Frame:
    """The released mask if there is one, else the projected CAD mask; written to ``train``."""
    from ..scene.labels import mask_bbox

    height, width = image.shape[:2]
    mask_file = f"masks/{source.sensor}_{frame:06d}.png"
    released = source.masks.get((frame, 0))
    bbox = pose.get("obj_bb")
    if released is not None:
        require_training_path(released)
        mask = cv2.imread(str(released), cv2.IMREAD_UNCHANGED)
        if (
            mask is None
            or mask.ndim != 2
            or mask.shape != (height, width)
            or not set(np.unique(mask).tolist()) <= MASK_VALUES
        ):
            raise ValueError(f"{released}: mask must be binary and match the RGB dimensions")
        mask = (mask > 0).astype(np.uint8) * 255
        if not cv2.imwrite(str(train / mask_file), mask):
            raise ValueError("cannot write normalized mask")
        if mask.any():
            bbox = mask_bbox(mask > 0)
        return _Frame(image, bbox, mask, mask_file, "provided", released)
    rotation, translation = _validate_pose(pose, "GT CAD mask")
    mask = projected_cad_mask(mesh, matrix, rotation, translation, (height, width))
    if not cv2.imwrite(str(train / mask_file), mask):
        raise ValueError("projected mask write failed")
    return _Frame(image, mask_bbox(mask > 0), mask, mask_file, PROJECTED_MASK_TYPE, None)


def _resolve_bbox(source: _SensorSource, frame: int, labels: _Frame, cache: dict) -> list:
    """The box, falling back to the released ``scene_gt_info`` amodal box; never guessed."""
    bbox = labels.bbox
    if bbox is None:
        info = source.folder / "scene_gt_info.json"
        if not info.is_file():
            raise ValueError("released GT box or nonempty provided mask required; never guess")
        if "info" not in cache:
            require_training_path(info)
            cache["info"] = json.loads(info.read_text())
        bbox = cache["info"][str(frame)][0]["bbox_obj"]
    height, width = labels.image.shape[:2]
    if len(bbox) != 4 or not np.isfinite(bbox).all() or min(bbox[2:]) <= 0:
        raise ValueError("invalid source GT box")
    if bbox[0] >= width or bbox[1] >= height or bbox[0] + bbox[2] <= 0 or bbox[1] + bbox[3] <= 0:
        raise ValueError("GT box does not intersect the RGB image")
    return bbox


def _write_overlay(
    stage: Path, sensor: str, frame: int, labels: _Frame, bbox: Sequence[float], uv: np.ndarray
) -> str:
    """Draw box, mask contour and keypoints; return the overlay's path relative to ``stage``."""
    image = labels.image
    height, width = image.shape[:2]
    overlay = image.copy()
    x, y, box_w, box_h = map(int, bbox)
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (0, 255, 255), 2)
    if labels.mask is not None:
        contours = cv2.findContours(labels.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
    for u, v in uv:
        if 0 <= u < width and 0 <= v < height:
            cv2.circle(overlay, (round(float(u)), round(float(v))), 2, (255, 0, 255), -1)
    scale = min(1.0, OVERLAY_MAX_SIDE / max(height, width))
    overlay = cv2.resize(overlay, None, fx=scale, fy=scale)
    name = f"{sensor}/{REVIEW_DIR}/{frame:06d}.jpg"
    destination = stage / name
    destination.parent.mkdir(exist_ok=True)
    if not cv2.imwrite(str(destination), overlay):
        raise ValueError("review overlay encode failed")
    return name


@dataclass(frozen=True)
class _Context:
    """Inputs shared by every sensor of one preparation."""

    root: Path
    stage: Path
    obj_id: int
    keypoints: Path
    points: np.ndarray
    mesh: CadMesh
    mesh_identity: Mapping[str, Any]
    excluded: Mapping[str, str]


def _relative(path: Path, root: Path) -> str:
    return Path(os.path.relpath(Path(path).resolve(), Path(root).resolve())).as_posix()


def _prepare_sensor(context: _Context, source: _SensorSource) -> tuple[dict, dict[str, str]]:
    """Write one sensor's COCO root; return its READY record and its overlays."""
    sensor, obj_id = source.sensor, context.obj_id
    target = context.stage / sensor
    train = target / "train"
    (train / "images").mkdir(parents=True)
    (train / "masks").mkdir()
    rows, annotations, records, overlays, cache = [], [], [], {}, {}
    frames = sorted(source.gt)
    chosen = {frames[i] for i in sorted({0, len(frames) // 2, len(frames) - 1})}
    for index, frame in enumerate(frames):
        instances = source.gt[frame]
        if len(instances) != 1 or instances[0]["obj_id"] != obj_id:
            raise ValueError(f"{sensor} frame {frame}: TRAIN must hold one instance of the object")
        path = source.images[frame]
        require_training_path(path)
        image = _decode(path)
        matrix = _camera_matrix(source.cameras[frame])
        digest = sha256_file(path)
        if digest in context.excluded:
            raise ValueError(
                f"reserved test_primesense image entered training: {sensor} frame {frame} "
                f"({path}) has the bytes of test image {context.excluded[digest]}"
            )
        image_name = f"{sensor}_{obj_id:06d}_{frame:06d}{path.suffix.lower()}"
        link_relative(train / "images" / image_name, path)
        labels = _frame_mask(source, frame, instances[0], image, matrix, context.mesh, train)
        bbox = _resolve_bbox(source, frame, labels, cache)
        height, width = image.shape[:2]
        annotation, uv = _project_annotation(
            instances[0], matrix, context.points, (width, height), index, bbox, labels.mask_file
        )
        if labels.mask_origin == PROJECTED_MASK_TYPE:
            annotation["mask_type"] = PROJECTED_MASK_TYPE
        rows.append({"id": index, "file_name": image_name, "width": width, "height": height})
        annotations.append(annotation)
        records.append(_record(context, source, frame, path, digest, image_name, labels, train))
        if frame in chosen:
            name = _write_overlay(context.stage, sensor, frame, labels, bbox, uv)
            overlays[name] = sha256_file(context.stage / name)
    categories = [{"id": 1, "name": f"obj_{obj_id:06d}"}]
    publish_json(
        train / "annotations.json",
        {"images": rows, "annotations": annotations, "categories": categories},
    )
    shutil.copyfile(context.keypoints, target / KEYPOINTS_NAME)
    masks = sum(r["mask_file"] is not None for r in records)
    projected = sum(r["mask_origin"] == PROJECTED_MASK_TYPE for r in records)
    publish_json(
        target / SOURCE_NAME,
        {
            "sensor": sensor,
            "object": obj_id,
            "split": "official_real_train",
            "supervision": SUPERVISION,
            "mask_policy": MASK_POLICY,
            "masks": masks,
            "projected_masks": projected,
            "mask_geometry": context.mesh_identity,
            "frames": len(rows),
            "source_gt_sha256": sha256_file(source.gt_path),
            "source_camera_sha256": sha256_file(source.camera_path),
            "records": records,
            "synthetic": False,
            "test_images_used": False,
        },
    )
    population = json.dumps([(r["frame"], r["rgb_sha256"]) for r in records])
    record = {
        "frames": len(rows),
        "masks": masks,
        "projected_masks": projected,
        "annotations_sha256": sha256_file(train / "annotations.json"),
        "source_sha256": sha256_file(target / SOURCE_NAME),
        "keypoints_sha256": sha256_file(context.keypoints),
        "rgb_population_sha256": hashlib.sha256(population.encode()).hexdigest(),
    }
    return record, overlays


def _record(
    context: _Context,
    source: _SensorSource,
    frame: int,
    path: Path,
    digest: str,
    image_name: str,
    labels: _Frame,
    train: Path,
) -> dict[str, Any]:
    mask_file = labels.mask_file
    return {
        "frame": frame,
        "source_rgb": _relative(path, context.root),
        "rgb_sha256": digest,
        "image_file": "train/images/" + image_name,
        "mask_file": "train/" + mask_file if mask_file else None,
        "prepared_mask_sha256": sha256_file(train / mask_file) if mask_file else None,
        "camera_sha256": sha256_file(source.camera_path),
        "source_mask_sha256": sha256_file(labels.source_mask) if labels.source_mask else None,
        "mask_origin": labels.mask_origin,
    }


def prepare(
    root: Path,
    obj_id: int,
    *,
    keypoints: Path,
    cad_model: Path,
    cad_audit: Path,
    reserved: Population,
    expected_frames: int | None = None,
) -> MultisensorSummary:
    """Write the three-sensor supervision roots of one T-LESS object.

    Args:
        root: Dataset root with ``data/train_primesense`` and ``sensors/train_{kinect,canon}``.
        obj_id: T-LESS object id.
        keypoints: The object's ordered 64 keypoints (meters), ``geometry/keypoints_3d.json``.
        cad_model: The CAD model the keypoints were audited on (``geometry/obj_XXXXXX.ply``).
        cad_audit: Its audit (``geometry/audit.json``).
        reserved: The reserved ``test_primesense`` population (:func:`reserved_population`).
        expected_frames: Captures every sensor must have (default: the full release, 1296, or
            648 for objects 19 and 20). Fewer marks the root ``partial_smoke_only``.

    Returns:
        What was written.

    Raises:
        ValueError: For any failed guard (see the module docstring).
        FileExistsError: If the output or its staging directory exists.
    """
    root = Path(root)
    full = tless_sensors.expected_frames(obj_id)
    expected = full if expected_frames is None else expected_frames
    out = output_dir(root, obj_id)
    stage = out.with_name(out.name + ".staging")
    if out.exists():
        raise FileExistsError(f"{out} exists; prepared roots are never overwritten")
    if stage.exists():
        raise FileExistsError(f"uncompleted staging {stage} exists; inspect it before retrying")
    _, points = load_keypoints(Path(keypoints))
    if len(points) != KEYPOINT_COUNT:
        raise ValueError(f"exactly {KEYPOINT_COUNT} ordered keypoints in meters are required")
    mesh = load_audited_cad(cad_model, cad_audit, keypoints)
    excluded = _reserved_hashes(reserved)
    sources_by_sensor = {
        sensor: _read_source(sensor, folder, expected)
        for sensor, folder in source_dirs(root, obj_id).items()
    }
    captures = {frozenset(s.gt) for s in sources_by_sensor.values()}
    if len(captures) != 1:
        raise ValueError("synchronized TRAIN capture ids differ across sensors")
    mesh_identity = {
        "path": _relative(cad_model, root),
        "sha256": sha256_file(cad_model),
        "audit_sha256": sha256_file(cad_audit),
        "mesh_and_pose_units": "mm",
        "mask_type": PROJECTED_MASK_TYPE,
        "preserve_holes": True,
        "external_occlusion_removed": False,
    }
    context = _Context(root, stage, obj_id, Path(keypoints), points, mesh, mesh_identity, excluded)
    stage.mkdir(parents=True)
    try:
        records, overlays = {}, {}
        for sensor in SENSORS:
            records[sensor], written = _prepare_sensor(context, sources_by_sensor[sensor])
            overlays.update(written)
        if len({r["rgb_population_sha256"] for r in records.values()}) != len(SENSORS):
            raise ValueError("sensor RGB populations are byte-identical; one camera is not three")
        publish_json(
            stage / READY_NAME,
            {
                "dataset": "tless",
                "object": obj_id,
                "sensors": records,
                "supervision": SUPERVISION,
                "mask_policy": MASK_POLICY,
                "expected_frames": expected,
                "partial_smoke_only": expected != full,
                "keypoints_sha256": sha256_file(keypoints),
                "mask_geometry": mesh_identity,
                "reserved_test_population_sha256": population_sha256(reserved),
                "synchronized_sensors_not_independent_scenes": True,
                "preparation_code_sha256": sha256_file(Path(__file__)),
                "review_overlays": overlays,
                "source_format": SOURCE_FORMAT,
            },
        )
    except BaseException:
        shutil.rmtree(stage)
        raise
    stage.rename(out)
    return MultisensorSummary(
        root=out,
        frames={sensor: records[sensor]["frames"] for sensor in SENSORS},
        overlays=tuple(out / name for name in overlays),
        ready=out / READY_NAME,
    )


def read_ready(root: Path) -> dict[str, Any]:
    """The ``READY.json`` of a prepared object root."""
    path = Path(root) / READY_NAME
    if not path.is_file():
        raise ValueError(f"{root} is not a prepared multisensor root (no {READY_NAME})")
    return json.loads(path.read_text())


def review_overlays(root: Path) -> list[Path]:
    """The overlays a review must cover (all nine the preparation wrote)."""
    return [Path(root) / name for name in read_ready(root)["review_overlays"]]


def _check_overlays(root: Path, ready: Mapping[str, Any]) -> None:
    for name, digest in ready["review_overlays"].items():
        path = Path(root) / name
        if not path.is_file() or sha256_file(path) != digest:
            raise review.ReviewError(f"review overlay {path} changed since preparation")


def approve(root: Path, reviewer: str) -> Path:
    """Record that ``reviewer`` inspected the nine overlays (``REVIEWED.json``).

    Raises:
        ValueError: For an empty reviewer, or an overlay that changed since preparation.
    """
    root = Path(root)
    ready = read_ready(root)
    _check_overlays(root, ready)
    return review.write(root, root / READY_NAME, review_overlays(root), reviewer)


def verify_prepared(root: Path) -> dict[str, Any]:
    """Re-check the prepared labels byte for byte against ``READY.json`` and ``SOURCE.json``.

    Returns:
        The ``READY.json`` content.

    Raises:
        ValueError: If a sensor is missing or an annotation, source record, keypoint file,
            linked image or mask changed.
    """
    root = Path(root)
    ready = read_ready(root)
    if ready.get("dataset") != "tless" or set(ready["sensors"]) != set(SENSORS):
        raise ValueError("all three real TRAIN sensors are required")
    for sensor, record in ready["sensors"].items():
        folder = root / sensor
        if (
            sha256_file(folder / "train" / "annotations.json") != record["annotations_sha256"]
            or sha256_file(folder / SOURCE_NAME) != record["source_sha256"]
            or sha256_file(folder / KEYPOINTS_NAME) != ready["keypoints_sha256"]
        ):
            raise ValueError(f"{sensor}: real labels, source record or keypoints changed")
        for row in json.loads((folder / SOURCE_NAME).read_text())["records"]:
            if sha256_file(folder / row["image_file"]) != row["rgb_sha256"]:
                raise ValueError(f"{sensor} frame {row['frame']}: prepared real RGB changed")
            mask = row["mask_file"]
            if mask and sha256_file(folder / mask) != row["prepared_mask_sha256"]:
                raise ValueError(f"{sensor} frame {row['frame']}: prepared real mask changed")
    return ready


def verify(root: Path, *, allow_partial: bool = False) -> dict[str, Any]:
    """Require a current human review and unchanged labels before training on a root.

    Args:
        root: Prepared object root.
        allow_partial: Accept a ``partial_smoke_only`` root (smoke tests only).

    Returns:
        The ``READY.json`` content.

    Raises:
        doper.bop.review.ReviewError: If the review is missing, stale or does not cover all
            nine overlays, or an overlay changed.
        ValueError: If the root is partial without ``allow_partial`` or a label changed.
    """
    root = Path(root)
    ready = read_ready(root)
    if ready["partial_smoke_only"] and not allow_partial:
        raise ValueError(f"{root} holds a partial smoke preparation")
    if not ready["review_overlays"]:
        raise review.ReviewError("the preparation wrote no review overlays")
    review.verify(root, root / READY_NAME, overlays=review_overlays(root))
    _check_overlays(root, ready)
    return verify_prepared(root)
