# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict-gate monocular pseudo-label pools from unlabeled onboarding and test views.

``export_pool`` labels the RGB frames of one staged view (``<root>/onboarding/<kind>``, see
``doper.bop.onboarding``) with a teacher model and keeps a frame only when it passes the
strict single-view gate: the top detection scores above ``GateSpec.score``, its PnP
reprojection error is below ``max_reprojection_px``, more than ``min_inliers`` confident
keypoints survive a RANSAC PnP, and the mask head covers at least ``min_area_fraction`` of
the image inside the padded detection box. An accepted frame carries a box and a mask and
nothing else: the teacher pose only goes to the audit records of ``manifest.json``, because a
monocular pose can be self-consistent and still wrong through symmetry or a single-view
ambiguity.

The exporter never sees validation labels. Frames whose bytes hash into ``exclusions`` (the
validation population; plus any overlap the view's ``READY.json`` recorded) are skipped, which
is the leakage guard for datasets whose staging records overlaps instead of refusing them.

``verify_pool`` audits the bytes on disk before training. ``merge_pools`` unions a static and
a dynamic pool of the same teacher, keypoint layout, evaluation population and gate, keeping
the first of any duplicate source image; ``verify_merged`` re-derives the merge from the
component pools it records.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from doper.bop import sources
from doper.bop.onboarding import SEQUENCE_PATTERNS
from doper.console import log
from doper.infer import Detection
from doper.multiview import single_view_inliers
from doper.pseudo_labels.gate import STRICT_GATE, GateSpec, strict_thresholds

SOURCE_KINDS = ("static", "dynamic", "test")
TEST_NOTICE = sources.TEST_NOTICE
BANNED_KEYS = ("keypoints", "num_keypoints", "rvec", "tvec", "object_pose")
MERGE_KINDS = ("static", "dynamic")
MERGED_SOURCE_KIND = "onboarding_static_dynamic"
REJECT_REASONS = ("no_det", "no_pose", "score", "reproj", "inliers", "mask", "reserved_eval")
DUPLICATE_REASON = "duplicate_image"
MAX_OVERLAYS = 12
PROGRESS_EVERY = 50
MASK_ON = 255
MASK_CHECK_THRESHOLD = 127
MASK_BOX_SLACK = 0.1
MASK_BOX_SLACK_PX = 2
MM_PER_M = 1000.0
IMAGE_SUFFIXES = sources.IMAGE_SUFFIXES
PARTIAL_SUFFIX = ".partial"
IDENTITY_KEYS = (
    "obj_id",
    "checkpoint_sha256",
    "keypoints_sha256",
    "evaluation_manifest_sha256",
    "gate",
)


class Predictor(Protocol):
    """One teacher inference: the call ``doper.infer.infer_image`` makes, with the model bound."""

    def __call__(
        self,
        image_bgr: np.ndarray,
        camera_matrix: np.ndarray,
        keypoints_mm: np.ndarray,
        score_threshold: float,
    ) -> tuple[list[Detection], np.ndarray | None]:
        """Detect on one image.

        Args:
            image_bgr: ``(H, W, 3)`` BGR image.
            camera_matrix: ``(3, 3)`` intrinsics.
            keypoints_mm: ``(K, 3)`` object keypoints in millimeters.
            score_threshold: Detector score floor.

        Returns:
            The detections (best first, with a PnP pose when one was solved) and the
            ``(H, W)`` foreground probability map, or ``None`` without a mask head.
        """
        ...


@dataclass(frozen=True)
class PoolSummary:
    """What one export or merge produced.

    Attributes:
        root: The pool directory.
        frames_seen: Source frames considered.
        accepted: Labels in the pool.
        rejected: Frames rejected (merge: including duplicates).
        reject_reasons: Rejected frames per reason.
    """

    root: Path
    frames_seen: int
    accepted: int
    rejected: int
    reject_reasons: Mapping[str, int]


class ModelPredictor:
    """``Predictor`` over a loaded teacher model (``doper.infer.infer_image`` with the mask)."""

    def __init__(self, model: Any, cfg: Any, device: str) -> None:  # noqa: ANN401
        """Bind a model from ``doper.infer.load_model``, its config and its device."""
        self.model, self.cfg, self.device = model, cfg, device

    def __call__(
        self,
        image_bgr: np.ndarray,
        camera_matrix: np.ndarray,
        keypoints_mm: np.ndarray,
        score_threshold: float,
    ) -> tuple[list[Detection], np.ndarray | None]:
        """Run ``infer_image`` and return its detections and mask probabilities."""
        from doper.infer import infer_image

        return infer_image(
            self.model,
            image_bgr,
            self.cfg,
            device=self.device,
            keypoints_3d=keypoints_mm,
            camera_matrix=camera_matrix,
            score_thr=score_threshold,
            return_mask=True,
        )


def _sequences(view_root: Path, kind: str, obj_id: int | None) -> list[tuple[Path, dict, int]]:
    """Staged sequences of ``kind`` (for one object when given), sorted, with their cameras."""
    pattern = re.compile(SEQUENCE_PATTERNS[kind])
    found = []
    for sequence in sorted(view_root.iterdir()):
        match = pattern.fullmatch(sequence.name)
        if not sequence.is_dir() or not match:
            continue
        sequence_obj = int(match[1]) if kind != "test" else -1
        if obj_id is not None and kind != "test" and sequence_obj != obj_id:
            continue
        cameras = json.loads((sequence / "scene_camera.json").read_text())
        if not cameras or not (sequence / "rgb").is_dir():
            raise ValueError(f"incomplete {kind} sequence {sequence}")
        found.append((sequence, cameras, sequence_obj))
    if not found:
        raise ValueError(f"no {kind} onboarding sequences in {view_root} for obj {obj_id}")
    return found


def _view_object(kind: str, sequences: list[tuple[Path, dict, int]], obj_id: int | None) -> int:
    if obj_id is not None:
        return obj_id
    objects = {obj for _, _, obj in sequences}
    if kind == "test":
        raise ValueError("a test view has no object; pass obj_id (the object to adapt)")
    if len(objects) != 1:
        raise ValueError(f"{kind} view holds objects {sorted(objects)}; pass obj_id to pick one")
    return objects.pop()


def _camera_for_frame(cameras: Mapping[str, Any], frame: int) -> tuple[np.ndarray, Mapping]:
    """Frame calibration, or the verified constant intrinsics for an extra RGB frame."""
    if str(frame) in cameras:
        record = cameras[str(frame)]
    else:
        record = next(iter(cameras.values()))
        if not all(
            np.allclose(row["cam_K"], record["cam_K"], rtol=1e-8, atol=1e-6)
            for row in cameras.values()
        ):
            raise ValueError(f"no calibration for frame {frame}; sequence intrinsics vary")
    k = np.asarray(record["cam_K"], dtype=np.float64).reshape(3, 3)
    if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
        raise ValueError("invalid camera intrinsics")
    return k, record


def _ready(view_root: Path) -> dict[str, Any]:
    path = view_root / sources.READY_NAME
    return json.loads(path.read_text()) if path.is_file() else {}


def _padded_mask(
    prob: np.ndarray, box: tuple[float, float, float, float], gate: GateSpec
) -> np.ndarray:
    """Mask probabilities at or above the gate inside the box padded by ``box_pad``."""
    height, width = prob.shape[:2]
    x0, y0, x1, y1 = box
    pad_x, pad_y = (x1 - x0) * gate.box_pad, (y1 - y0) * gate.box_pad
    keep = np.zeros((height, width), bool)
    keep[
        max(0, int(y0 - pad_y)) : min(height, int(y1 + pad_y) + 1),
        max(0, int(x0 - pad_x)) : min(width, int(x1 + pad_x) + 1),
    ] = True
    return ((prob >= gate.mask_probability) & keep).astype(np.uint8) * MASK_ON


def _reject_reason(
    detections: list[Detection],
    prob: np.ndarray | None,
    keypoints_mm: np.ndarray,
    camera_matrix: np.ndarray,
    shape: tuple[int, int],
    gate: GateSpec,
) -> tuple[str | None, int, np.ndarray | None]:
    """Apply the gate; return ``(reason or None, inliers, mask)``."""
    if not detections:
        return "no_det", 0, None
    det = detections[0]
    if det.rvec is None:
        return "no_pose", 0, None
    if not np.isfinite(det.score) or det.score <= gate.score:
        return "score", 0, None
    reprojection = det.reproj_error_px
    if (
        reprojection is None
        or not np.isfinite(reprojection)
        or reprojection >= gate.max_reprojection_px
    ):
        return "reproj", 0, None
    inliers = int(
        single_view_inliers(
            keypoints_mm,
            det.keypoints,
            det.keypoint_scores >= gate.keypoint_score,
            camera_matrix,
            np.zeros(5),
            gate.ransac_px,
            gate.pnp_method,
        ).sum()
    )
    if inliers <= gate.min_inliers:
        return "inliers", inliers, None
    x0, y0, x1, y1 = (float(v) for v in det.bbox_xyxy)
    if prob is None or not np.isfinite([x0, y0, x1, y1]).all() or x1 <= x0 or y1 <= y0:
        return "mask", inliers, None
    mask = _padded_mask(prob, (x0, y0, x1, y1), gate)
    height, width = shape
    if int((mask > 0).sum()) < gate.min_area_fraction * height * width:
        return "mask", inliers, None
    return None, inliers, mask


def _write_overlay(
    out: Path, index: int, image: np.ndarray, mask: np.ndarray, box: tuple[float, ...]
) -> None:
    overlay = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    x0, y0, x1, y1 = (float(v) for v in box)
    cv2.rectangle(overlay, (int(x0), int(y0)), (int(x1), int(y1)), (0, 200, 255), 2)
    (out / "overlays").mkdir(exist_ok=True)
    cv2.imwrite(str(out / "overlays" / f"{index:03d}.jpg"), overlay)


def _calibration_hashes(sequences: list[tuple[Path, dict, int]]) -> dict[str, str]:
    return {seq.name: sources.sha256_file(seq / "scene_camera.json") for seq, _, _ in sequences}


def _transduction(kind: str) -> str:
    if kind == "test":
        return TEST_NOTICE
    return f"dedicated {kind} onboarding; validation excluded"


@dataclass
class _Accepted:
    images: list[dict[str, Any]]
    annotations: list[dict[str, Any]]
    audit: list[dict[str, Any]]


def export_pool(
    predictor: Predictor,
    keypoints: Path,
    view_root: Path,
    kind: str,
    exclusions: frozenset[str],
    gate: GateSpec,
    out: Path,
    *,
    teacher_sha256: str,
    obj_id: int | None = None,
) -> PoolSummary:
    """Label one staged view with the strict gate and write a box/mask-only COCO pool.

    Args:
        predictor: The teacher, wrapping ``doper.infer.infer_image`` (``ModelPredictor``).
        keypoints: ``keypoints_3d.json`` of the object (meters).
        view_root: ``<root>/onboarding/<kind>``; frames are ``<sequence>/rgb/*.{jpg,png}``,
            ordered sorted ``*.jpg`` then sorted ``*.png`` per sequence.
        kind: ``static``, ``dynamic`` or ``test``.
        exclusions: SHA-256 of every validation image; matching frames are skipped (together
            with the overlaps ``view_root/READY.json`` recorded).
        gate: The gate thresholds.
        out: New pool directory (``train/images``, ``train/masks``, ``train/annotations.json``,
            ``manifest.json``); it must not exist.
        teacher_sha256: SHA-256 of the teacher checkpoint, recorded as its identity.
        obj_id: The object the teacher was trained for. It selects that object's sequences of a
            static or dynamic view (and may be omitted when the view holds exactly one); a
            test view stages scenes, not objects, so it must be given there.

    Returns:
        The summary of the pool.

    Raises:
        ValueError: For an unknown kind, an ambiguous or empty view, no validation exclusions,
            unusable images or calibration, or inputs that change during the export.
        FileExistsError: If ``out`` exists.
    """
    if kind not in SOURCE_KINDS:
        raise ValueError(f"view kind must be one of {SOURCE_KINDS}")
    view_root, out = Path(view_root).resolve(), Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists; remove it or choose another pool directory")
    sequences = _sequences(view_root, kind, obj_id)
    object_id = _view_object(kind, sequences, obj_id)
    ready = _ready(view_root)
    reserved = set(exclusions) | {o["sha256"] for o in ready.get("validation_overlap", [])}
    if not reserved:
        raise ValueError("empty validation exclusion population; the leakage guard needs it")
    calibration = _calibration_hashes(sequences)
    keypoint_digest = sources.sha256_file(keypoints)
    keypoints_mm = (
        np.array(json.loads(Path(keypoints).read_text())["keypoints_3d"], dtype=np.float64)
        * MM_PER_M
    )
    stage = out.with_name(out.name + PARTIAL_SUFFIX)
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "train" / "images").mkdir(parents=True)
    (stage / "train" / "masks").mkdir(parents=True)
    accepted = _Accepted([], [], [])
    reasons: dict[str, int] = dict.fromkeys(REJECT_REASONS, 0)
    seen = 0
    started = time.time()
    for sequence, cameras, _ in sequences:
        rgb = sequence / "rgb"
        for path in [p for suffix in IMAGE_SUFFIXES for p in sorted(rgb.glob(f"*{suffix}"))]:
            seen += 1
            digest = sources.sha256_file(path)
            if digest in reserved:
                reasons["reserved_eval"] += 1
                continue
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"cannot decode source image {path}")
            height, width = image.shape[:2]
            matrix, record = _camera_for_frame(cameras, int(path.stem))
            if (record.get("width", width), record.get("height", height)) != (width, height):
                raise ValueError(f"camera/image dimensions differ: {path}")
            detections, prob = predictor(image, matrix, keypoints_mm, gate.detector_threshold)
            reason, inliers, mask = _reject_reason(
                detections, prob, keypoints_mm, matrix, (height, width), gate
            )
            if reason is not None or mask is None:
                reasons[reason or "mask"] += 1
                continue
            _accept(stage, accepted, path, sequence, image, mask, detections[0], digest, inliers)
            if len(accepted.images) % PROGRESS_EVERY == 0:
                elapsed = time.time() - started
                log(f"  accepted {len(accepted.images)} / seen {seen} ({elapsed:.0f}s)")
    if sources.sha256_file(keypoints) != keypoint_digest:
        raise ValueError("teacher or keypoint file changed during export")
    if _calibration_hashes(sequences) != calibration:
        raise ValueError("source calibration changed during export")
    manifest = _pool_manifest(
        kind, object_id, view_root, teacher_sha256, keypoint_digest, calibration, ready, gate
    )
    manifest.update(
        frames_seen=seen,
        accepted=len(accepted.images),
        rejected=sum(reasons.values()),
        reject_reasons=reasons,
        records=accepted.audit,
    )
    coco = {
        "images": accepted.images,
        "annotations": accepted.annotations,
        "categories": [{"id": 1, "name": f"obj_{object_id:06d}"}],
    }
    (stage / "train" / "annotations.json").write_text(json.dumps(coco))
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=1))
    stage.rename(out)
    log(f"[export] {len(accepted.images)} accepted of {seen} {kind} source frames; wrote {out}")
    return PoolSummary(out, seen, len(accepted.images), manifest["rejected"], reasons)


def _accept(
    stage: Path,
    accepted: _Accepted,
    path: Path,
    sequence: Path,
    image: np.ndarray,
    mask: np.ndarray,
    det: Detection,
    digest: str,
    inliers: int,
) -> None:
    """Write one accepted frame's image and mask and append its label and audit record."""
    x0, y0, x1, y1 = (float(v) for v in det.bbox_xyxy)
    stem = f"scene{sequence.name}_frame{path.stem}"
    image_name = f"{stem}{path.suffix.lower()}"
    shutil.copyfile(path, stage / "train" / "images" / image_name)
    if not cv2.imwrite(str(stage / "train" / "masks" / f"{stem}.png"), mask):
        raise OSError("could not write pseudo mask")
    index = len(accepted.images)
    if index < MAX_OVERLAYS:
        _write_overlay(stage, index, image, mask, (x0, y0, x1, y1))
    area = int((mask > 0).sum())
    height, width = image.shape[:2]
    accepted.images.append({"id": index, "file_name": image_name, "width": width, "height": height})
    accepted.annotations.append(
        {
            "id": index,
            "image_id": index,
            "category_id": 1,
            "bbox": [x0, y0, x1 - x0, y1 - y0],
            "area": area,
            "iscrowd": 0,
            "segmentation": f"{stem}.png",
            "mask_file": f"masks/{stem}.png",
        }
    )
    accepted.audit.append(
        {
            "image": image_name,
            "scene": sequence.name,
            "source_relative_path": f"{sequence.name}/rgb/{path.name}",
            "source_image_sha256": digest,
            "frame": int(path.stem),
            "score": float(det.score),
            "reproj_px": float(det.reproj_error_px or 0.0),
            "inliers": inliers,
            "rvec": [float(v) for v in np.asarray(det.rvec).ravel()],
            "tvec": [float(v) for v in np.asarray(det.tvec).ravel()],
            "mask_area": area,
        }
    )


def _pool_manifest(
    kind: str,
    obj_id: int,
    view_root: Path,
    teacher_sha256: str,
    keypoint_digest: str,
    calibration: Mapping[str, str],
    ready: Mapping[str, Any],
    gate: GateSpec,
) -> dict[str, Any]:
    return {
        "obj_id": obj_id,
        "checkpoint_sha256": teacher_sha256,
        "keypoints_sha256": keypoint_digest,
        "source_kind": sources.SOURCE_KINDS[kind],
        "source_root": str(view_root),
        "transductive_test_adaptation": kind == "test",
        "test_gt_used": False,
        "test_targets_used": False,
        "source_calibration_sha256": dict(calibration),
        "evaluation_manifest_sha256": ready.get("evaluation_manifest_sha256"),
        "gt_pose_or_mask_used": False,
        "pose_units": "mm",
        "includes_annotated_frames": False,
        "transduction": _transduction(kind),
        "gate": gate.as_manifest(),
    }


def _read_pool(pool: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = pool / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    return json.loads((pool / "train" / "annotations.json").read_text()), manifest


def _verify_provenance(
    pool: Path,
    view_root: Path,
    images: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    exclusions: frozenset[str],
) -> list[str]:
    """Faults in the onboarding provenance recorded by the manifest."""
    faults: list[str] = []
    kind = next(
        (k for k in SOURCE_KINDS if sources.SOURCE_KINDS[k] == manifest.get("source_kind")), ""
    )
    if not kind:
        return ["wrong source/supervision contract"]
    if kind == "test" and (
        manifest.get("transductive_test_adaptation") is not True
        or manifest.get("test_gt_used") is not False
        or manifest.get("test_targets_used") is not False
    ):
        faults.append("test transduction/supervision contract missing or invalid")
    if manifest.get("gt_pose_or_mask_used") is not False:
        faults.append("wrong source/supervision contract")
    recorded = _ready(view_root).get("evaluation_manifest_sha256")
    if recorded is not None and manifest.get("evaluation_manifest_sha256") != recorded:
        faults.append("evaluation manifest identity mismatch")
    if not exclusions:
        faults.append("empty evaluation exclusion manifest")
    for sequence, digest in manifest.get("source_calibration_sha256", {}).items():
        path = view_root / sequence / "scene_camera.json"
        if not path.is_file() or sources.sha256_file(path) != digest:
            faults.append("static calibration changed")
    records = {row["image"]: row for row in manifest.get("records", [])}
    if len(records) != len(images) or manifest.get("accepted") != len(images):
        faults.append("label/audit record counts differ")
    for image in images:
        faults.extend(_verify_record(pool, view_root, image, records, exclusions))
    return faults


def _verify_record(
    pool: Path,
    view_root: Path,
    image: Mapping[str, Any],
    records: Mapping[str, Any],
    exclusions: frozenset[str],
) -> list[str]:
    record = records.get(image["file_name"])
    if not record:
        return ["image missing audit record"]
    source = (view_root / record["source_relative_path"]).resolve()
    if view_root not in source.parents or not source.is_file():
        return ["source outside static root or missing"]
    staged = pool / "train" / "images" / image["file_name"]
    if not staged.is_file():
        return []
    faults = []
    digest = sources.sha256_file(staged)
    if digest != record["source_image_sha256"] or digest != sources.sha256_file(source):
        faults.append("source/output image changed")
    if digest in exclusions:
        faults.append("validation image present in adaptation labels")
    return faults


def _mask_fault(pool: Path, image: Mapping[str, Any], annotation: Mapping[str, Any] | None) -> str:
    """``gone``, ``shape``, ``mask`` or ``""`` for one image's files and mask geometry."""
    mask_file = (annotation or {}).get("mask_file", "")
    picture = cv2.imread(str(pool / "train" / "images" / image["file_name"]))
    if picture is None:
        return "gone"
    if (picture.shape[1], picture.shape[0]) != (image["width"], image["height"]):
        return "shape"
    mask = cv2.imread(str(pool / "train" / mask_file), 0)
    if mask is None or mask.shape != picture.shape[:2] or not (mask > MASK_CHECK_THRESHOLD).any():
        return "mask"
    ys, xs = np.nonzero(mask > MASK_CHECK_THRESHOLD)
    x, y, w, h = (annotation or {})["bbox"]
    slack = MASK_BOX_SLACK_PX
    outside = (
        xs.min() < x - w * MASK_BOX_SLACK - slack
        or xs.max() > x + w * (1 + MASK_BOX_SLACK) + slack
        or ys.min() < y - h * MASK_BOX_SLACK - slack
        or ys.max() > y + h * (1 + MASK_BOX_SLACK) + slack
    )
    return "mask" if outside else ""


def verify_pool(pool: Path, view_root: Path, exclusions: frozenset[str]) -> list[str]:
    """Audit a pool's bytes against its view and the validation population.

    Checks, each a fault: no validation image among the labels; annotations carry a box and a
    mask file and none of ``BANNED_KEYS``; images and masks exist, decode and match their
    records; masks are non-empty and sit inside their box (+/-10 percent + 2 px); the manifest
    records the gate and a consistent source and supervision contract; every label traces to
    an unchanged source image and calibration under ``view_root``.

    Args:
        pool: A pool directory written by ``export_pool``.
        view_root: The view the pool was exported from.
        exclusions: SHA-256 of every validation image.

    Returns:
        The faults; empty when the pool is safe to train on.
    """
    pool, view_root = Path(pool), Path(view_root).resolve()
    coco, manifest = _read_pool(pool)
    images, annotations = coco["images"], coco["annotations"]
    faults: list[str] = []
    if not manifest:
        faults.append("no manifest.json; the gate used is unrecorded")
    elif "gate" not in manifest:
        faults.append("manifest.json records no gate")
    if manifest:
        faults.extend(_verify_provenance(pool, view_root, images, manifest, exclusions))
    bad = [a["id"] for a in annotations if any(b in a for b in BANNED_KEYS)]
    if bad:
        faults.append(
            f"{len(bad)} annotations carry pose or keypoint fields (ids {bad[:5]}); "
            "monocular labels are box/mask only"
        )
    missing = [a["id"] for a in annotations if not a.get("bbox") or not a.get("mask_file")]
    if missing:
        faults.append(f"{len(missing)} annotations lack a bbox or mask_file")
    by_image = {a["image_id"]: a for a in annotations}
    counts = Counter(
        _mask_fault(pool, im, by_image.get(im["id"]))
        for im in images
        if by_image.get(im["id"]) and by_image[im["id"]].get("bbox")
    )
    if counts["gone"]:
        faults.append(f"{counts['gone']} of {len(images)} checked images or masks are missing")
    if counts["shape"]:
        faults.append(f"{counts['shape']} images disagree with their width/height record")
    if counts["mask"]:
        faults.append(f"{counts['mask']} masks are empty or fall outside their box")
    return faults


def _contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise ValueError("missing file or path outside label root")
    return path


def _component_identity(kind: str, root: Path) -> tuple[dict, dict, dict]:
    """Validate one component pool's contract; return its manifest, COCO and identity."""
    manifest = json.loads((root / "manifest.json").read_text())
    coco = json.loads((root / "train" / "annotations.json").read_text())
    identity = {key: manifest.get(key) for key in IDENTITY_KEYS}
    if (
        manifest.get("source_kind") != f"onboarding_{kind}"
        or manifest.get("gt_pose_or_mask_used") is not False
        or manifest.get("includes_annotated_frames")
        or strict_thresholds(identity["gate"] or {}) != STRICT_GATE
    ):
        raise ValueError("source/object/gate/supervision contract mismatch")
    return manifest, coco, identity


def _merged_description(
    components: Mapping[str, Path], base: Path
) -> tuple[dict[str, Any], dict[str, Any], list[tuple[Path, str]]]:
    """The merged manifest, COCO and (source, relative target) files of a static+dynamic union."""
    if set(components) != set(MERGE_KINDS):
        raise ValueError("merge needs exactly the static and dynamic component pools")
    images, annotations, records, files, duplicates = [], [], [], [], []
    counts: dict[str, Any] = {}
    parents: dict[str, Any] = {}
    seen: set[str] = set()
    common: dict[str, Any] | None = None
    for kind in MERGE_KINDS:
        root = Path(components[kind])
        manifest, coco, identity = _component_identity(kind, root)
        if common is not None and identity != common:
            raise ValueError("component teacher/keypoint/evaluation/gate identity mismatch")
        common = identity
        by_image = {a["image_id"]: a for a in coco["annotations"]}
        audit = {r["image"]: r for r in manifest["records"]}
        sizes = (len(by_image), len(coco["annotations"]), len(coco["images"]), len(audit))
        if len(set(sizes)) != 1 or sizes[0] != manifest["accepted"]:
            raise ValueError("component counts differ")
        counts[kind] = {
            "frames_seen": manifest["frames_seen"],
            "accepted_before_dedup": manifest["accepted"],
            "accepted_for_training": 0,
            "reject_reasons": manifest["reject_reasons"],
        }
        parents[kind] = {
            "root": os.path.relpath(root.resolve(), base.resolve()),
            "manifest_sha256": sources.sha256_file(root / "manifest.json"),
            "annotations_sha256": sources.sha256_file(root / "train" / "annotations.json"),
        }
        for image in coco["images"]:
            annotation, record = by_image[image["id"]], audit[image["file_name"]]
            if any(key in annotation for key in BANNED_KEYS):
                raise ValueError("real pose/keypoint supervision is forbidden")
            source_image = _contained(root / "train" / "images", image["file_name"])
            source_mask = _contained(root / "train", annotation["mask_file"])
            digest = sources.sha256_file(source_image)
            if digest != record["source_image_sha256"]:
                raise ValueError("component image changed")
            if digest in seen:
                duplicates.append({"source": kind, "image": image["file_name"], "sha256": digest})
                continue
            seen.add(digest)
            index = len(images)
            filename = f"{kind}__{Path(image['file_name']).name}"
            maskname = f"{kind}__{Path(annotation['mask_file']).name}"
            images.append({**copy.deepcopy(image), "id": index, "file_name": filename})
            annotations.append(
                {
                    **copy.deepcopy(annotation),
                    "id": index,
                    "image_id": index,
                    "mask_file": f"masks/{maskname}",
                    "segmentation": maskname,
                }
            )
            records.append(
                {
                    **copy.deepcopy(record),
                    "image": filename,
                    "onboarding_source": kind,
                    "component_image": image["file_name"],
                }
            )
            files.extend(
                [
                    (source_image, f"train/images/{filename}"),
                    (source_mask, f"train/masks/{maskname}"),
                ]
            )
            counts[kind]["accepted_for_training"] += 1
    rejected: Counter[str] = Counter()
    for count in counts.values():
        rejected.update(count["reject_reasons"])
    rejected[DUPLICATE_REASON] = len(duplicates)
    manifest = dict(
        common or {},
        source_kind=MERGED_SOURCE_KIND,
        gt_pose_or_mask_used=False,
        includes_annotated_frames=False,
        components=parents,
        source_counts=counts,
        frames_seen=sum(c["frames_seen"] for c in counts.values()),
        accepted=len(images),
        rejected=sum(rejected.values()),
        reject_reasons=dict(rejected),
        deduplicated=duplicates,
        records=records,
        files_sha256={relative: sources.sha256_file(source) for source, relative in files},
    )
    obj_id = (common or {}).get("obj_id")
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": f"obj_{obj_id or 0:06d}"}],
    }
    return manifest, coco, files


def merge_pools(components: Mapping[str, Path], out: Path) -> PoolSummary:
    """Union a static and a dynamic pool of one teacher into a training pool.

    Static labels come first. A source image whose SHA-256 was already taken is dropped (first
    wins); kept files are renamed ``{kind}__name`` and ids are reindexed. Both components must
    carry the strict gate and share teacher, keypoint, evaluation and gate identity.

    Args:
        components: Pool directories keyed ``static`` and ``dynamic``; the manifest records
            their paths relative to ``out``.
        out: New merged pool directory; it must not exist.

    Returns:
        The summary of the merged pool.

    Raises:
        ValueError: If a component breaks the contract or the identities differ.
        FileExistsError: If ``out`` exists.
    """
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists; remove it or choose another pool directory")
    manifest, coco, files = _merged_description(components, out)
    stage = out.with_name(out.name + PARTIAL_SUFFIX)
    shutil.rmtree(stage, ignore_errors=True)
    for source, relative in files:
        (stage / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, stage / relative)
    (stage / "train").mkdir(parents=True, exist_ok=True)
    sources.publish_json(stage / "train" / "annotations.json", coco)
    sources.publish_json(stage / "manifest.json", manifest)
    stage.rename(out)
    faults = verify_merged(out)
    if faults:
        raise ValueError(f"merged pool failed its audit: {faults}")
    return PoolSummary(
        out,
        manifest["frames_seen"],
        manifest["accepted"],
        manifest["rejected"],
        manifest["reject_reasons"],
    )


def verify_merged(out: Path) -> list[str]:
    """Re-derive a merged pool from the component pools it records and compare.

    The manifest records each component's path relative to the merged pool, so a runs tree
    that is moved as a whole still verifies.

    Args:
        out: A pool directory written by ``merge_pools``.

    Returns:
        The faults (component identity or contract mismatch, changed metadata, changed
        images or masks); empty when the merge is intact.
    """
    out = Path(out)
    stored = json.loads((out / "manifest.json").read_text())
    try:
        components = {
            kind: (out / info["root"]).resolve() for kind, info in stored["components"].items()
        }
        manifest, coco, files = _merged_description(components, out)
        faults = []
        if (
            stored != manifest
            or json.loads((out / "train" / "annotations.json").read_text()) != coco
        ):
            faults.append("merged metadata changed")
        for source, relative in files:
            if sources.sha256_file(_contained(out, relative)) != sources.sha256_file(source):
                faults.append("merged image/mask changed")
        return faults
    except (ValueError, KeyError, OSError) as error:
        return [str(error)]
