# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SHIRT pose candidates: the DOPER framewise evaluation and the 0/180 re-crop candidates.

``framewise`` runs the SPEED+ model on every frame of each trajectory independently, before any
temporal filter: the top-1 detection (``score_thr=0.0``, ``max_det=1``; no CLAHE, no TTA and no
box pad), then ``doper.speedplus.pose.argmax_usac_candidate`` on the keypoints scoring at least
0.30 (sequential USAC, 4,000 iterations, 3 px, seed 42). It writes, under ``out``::

    <roe>/<domain>/records.jsonl   one record per frame, in frame order
    <roe>/<domain>/summary.json    the sequence summary
    summary.json                   every sequence summary, the checkpoint and layout identity

Every frame keeps its record: a frame without a detection or a pose has ``baseline: null``, so
the frame index stays the label index. The ground-truth errors in the records and summaries
(SHIRT normalizes the ground-truth quaternion in float64) are reporting fields that no decision
reads. No overlays or error plots are written.

The checkpoint must be the one the recipe pins (``checkpoint_sha256``) unless
``allow_unpinned_checkpoint``, which the summary records, and it must have been trained on the
given 75-point layout (``doper.shirt.inputs.check_model_inputs``): another points hash is
refused naming both SHA-256s, and the same points in another file are recorded as
``layout_formatting_deviation``. The records carry SHIRT ground truth and stay user-local.

``crop_candidates`` turns the framewise records into each frame's pose candidates for the temporal
filters: the raw pose, then two fresh crops around the raw pose's projected 75-keypoint box, rolled
by 0 and 180 degrees, each decoded and solved with the same USAC. The crop geometry and the DARK
heatmap beta come from the loaded checkpoint. Rotation-aux KL scoring is not done: the ``aux_*``
fields are present and null. A frame whose raw PnP failed is kept with no candidates, so the frame
index stays the time base; a frame 0 without a raw pose is refused, as the filters cannot start
without it.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..bop.sources import publish_json, publish_text, sha256_file
from ..recipes.shirt_schema import ShirtRecipe
from ..speedplus.metrics import check_definition, pose_score
from ..speedplus.pose import (
    DEFAULT_SCORE_THRESHOLD,
    argmax_usac_candidate,
    candidate_record,
    pose_projected_crop,
    projected_pseudo_box,
    quaternion_to_object_camera,
)
from . import PAPER_DOMAIN, data
from .inputs import (
    DETECTION_SCORE_THRESHOLD,
    GROUND_TRUTH_NORMALIZATION,
    MAX_DETECTIONS,
    RECORDS_NAME,
    SUMMARY_NAME,
    check_model_inputs,
    check_request,
    load_model,
    load_records,
    packaged_recipe,
    read_image,
    recorded_checkpoint,
    sequence_dir,
)
from .inputs import check_checkpoint_pin as check_checkpoint_pin  # re-exported
from .inputs import heatmap_beta as heatmap_beta  # re-exported

METHOD = "framewise"
GROSS_ROTATION_DEG = (30.0, 60.0, 120.0, 150.0)
PRACTICAL_FIELD = "practical_success"
LOOSE_FIELD = "loose_success"
DISTRIBUTION_KEYS = ("mean", "median", "p90", "p95", "p99", "max")
POSE_PIPELINE = (
    "top-1 detection (score_thr 0.0, max_det 1); keypoint score >= 0.30; "
    "USAC 4000 iterations / 3 px / 0.99 / seed 42; no CLAHE, TTA or box pad"
)


def distribution(values: Sequence[float] | np.ndarray) -> dict[str, float | None]:
    """Mean, median, p90, p95, p99 and max of the finite ``values``; None for each without."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return dict.fromkeys(DISTRIBUTION_KEYS)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    """Geodesic angle in degrees between two rotation matrices."""
    relative = first @ second.T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _ground_truth_rotation(record: Mapping[str, Any]) -> np.ndarray:
    quaternion = np.asarray(record["ground_truth_quaternion"], dtype=np.float64)
    return quaternion_to_object_camera(quaternion, normalization=GROUND_TRUTH_NORMALIZATION)


def temporal_increment_errors(records: Sequence[Mapping[str, Any]]) -> list[float]:
    """Rotation error of each frame-to-frame increment between consecutive solved frames."""
    values: list[float] = []
    for previous, current in zip(records, records[1:], strict=False):
        if previous["baseline"] is None or current["baseline"] is None:
            continue
        predicted_previous, _ = cv2.Rodrigues(
            np.asarray(previous["baseline"]["rvec"], dtype=np.float64)
        )
        predicted_current, _ = cv2.Rodrigues(
            np.asarray(current["baseline"]["rvec"], dtype=np.float64)
        )
        predicted = predicted_current @ predicted_previous.T
        truth = _ground_truth_rotation(current) @ _ground_truth_rotation(previous).T
        values.append(rotation_distance_deg(predicted, truth))
    return values


def frame_record(
    label: data.Label,
    detection: Any,  # noqa: ANN401 - doper.infer.Detection or None
    points_3d: np.ndarray,
    camera: Any,  # noqa: ANN401 - doper.speedplus.geometry.Camera
) -> dict[str, Any]:
    """One frame's record: its label, detection score and USAC pose (``baseline``) or None."""
    candidate = (
        None
        if detection is None
        else argmax_usac_candidate(
            detection.keypoints,
            detection.keypoint_scores,
            points_3d,
            camera.K,
            camera.dist,
            score_thr=DEFAULT_SCORE_THRESHOLD,
        )
    )
    baseline = (
        None
        if candidate is None
        else candidate_record(
            candidate, label.quaternion, label.translation, normalization=GROUND_TRUTH_NORMALIZATION
        )
    )
    return {
        "index": label.index,
        "filename": label.filename,
        "ground_truth_quaternion": label.quaternion.tolist(),
        "ground_truth_translation": label.translation.tolist(),
        "detection_score": None if detection is None else float(detection.score),
        "baseline": baseline,
    }


def _pose_scores(solved: Sequence[Mapping[str, Any]], definition: str) -> list[float]:
    if definition == "paper":
        return [row["pose_score"] for row in solved]
    return [
        pose_score(
            math.radians(row["rotation_deg"]), row["translation_relative"], definition=definition
        )
        for row in solved
    ]


def summarize_sequence(
    records: Sequence[Mapping[str, Any]],
    inference_ms: Sequence[float],
    *,
    definition: str = "paper",
) -> dict[str, Any]:
    """The framewise summary of one trajectory and domain.

    Rates are over every requested frame; the distributions are over the solved ones. In
    ``official`` mode ``pose_score`` holds the per-frame official (deadband) scores.

    Raises:
        ValueError: Without records, or for an unknown score definition.
    """
    check_definition(definition)
    requested = len(records)
    if not requested:
        raise ValueError("a framewise summary needs at least one record")
    solved = [record["baseline"] for record in records if record["baseline"] is not None]
    rotation = np.asarray([row["rotation_deg"] for row in solved], dtype=np.float64)
    no_pose = requested - len(solved)
    summary: dict[str, Any] = {
        "requested": requested,
        "solved": len(solved),
        "solver_return_rate": len(solved) / requested,
        "no_pose_count": no_pose,
        "no_pose_rate": no_pose / requested,
        "practical_recall_5deg_2pct": sum(row[PRACTICAL_FIELD] for row in solved) / requested,
        "loose_recall_10deg_5pct": sum(row[LOOSE_FIELD] for row in solved) / requested,
        "rotation_deg": distribution(rotation),
        "translation_relative": distribution([row["translation_relative"] for row in solved]),
        "pose_score": distribution(_pose_scores(solved, definition)),
        "ransac_inliers": distribution([row["ransac_inliers"] for row in solved]),
    }
    for threshold in GROSS_ROTATION_DEG:
        summary[f"rotation_gt{threshold:.0f}_count"] = int(np.sum(rotation > threshold))
    summary["gross_no_pose_or_rotation_gt30_count"] = int(
        no_pose + np.sum(rotation > GROSS_ROTATION_DEG[0])
    )
    summary["temporal_rotation_increment_error_deg"] = distribution(
        temporal_increment_errors(records)
    )
    summary["inference_latency_ms"] = distribution(inference_ms)
    return summary


def _refuse_existing(
    out: Path, trajectories: Sequence[str], domain: str, force: bool, method: str = METHOD
) -> None:
    written = [out / SUMMARY_NAME] + [
        sequence_dir(out, roe, domain) / name
        for roe in trajectories
        for name in (RECORDS_NAME, SUMMARY_NAME)
    ]
    existing = [path for path in written if path.exists()]
    if existing and not force:
        raise FileExistsError(
            f"{method} outputs already exist ({existing[0]}); pass --force to overwrite them"
        )


def _infer_frame(
    model: Any,  # noqa: ANN401
    image: np.ndarray,
    cfg: Any,  # noqa: ANN401
    device: str,
) -> tuple[Any, float]:
    """The top-1 detection of one image and the inference time in milliseconds."""
    import torch

    from .. import infer

    cuda = device.startswith("cuda")
    if cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    detections = infer.infer_image(
        model, image, cfg, device, score_thr=DETECTION_SCORE_THRESHOLD, max_det=MAX_DETECTIONS
    )
    if cuda:
        torch.cuda.synchronize()
    return (detections[0] if detections else None), (time.perf_counter() - start) * 1000.0


def _evaluate_sequence(
    model: Any,  # noqa: ANN401
    cfg: Any,  # noqa: ANN401
    shirt_root: Path,
    roe: str,
    domain: str,
    labels: Sequence[data.Label],
    points_3d: np.ndarray,
    camera: Any,  # noqa: ANN401
    device: str,
) -> tuple[list[dict[str, Any]], list[float]]:
    records: list[dict[str, Any]] = []
    inference_ms: list[float] = []
    for label in labels:
        image = read_image(shirt_root, roe, domain, label.filename)
        detection, milliseconds = _infer_frame(model, image, cfg, device)
        inference_ms.append(milliseconds)
        records.append(frame_record(label, detection, points_3d, camera))
    return records, inference_ms


def framewise(
    checkpoint: str | Path,
    shirt_root: str | Path,
    layout: str | Path,
    out: str | Path,
    *,
    trajectories: Sequence[str],
    domain: str = PAPER_DOMAIN,
    recipe: ShirtRecipe | None = None,
    allow_unpinned_checkpoint: bool = False,
    limit: int = 0,
    device: str = "cpu",
    force: bool = False,
) -> dict[str, Any]:
    """Evaluate a SPEED+ checkpoint framewise on SHIRT trajectories.

    Args:
        checkpoint: SPEED+ checkpoint (new format, or a legacy one in the recipe registry).
        shirt_root: The SHIRT tree.
        layout: The 75-point ``keypoints_3d.json`` the checkpoint was trained on.
        out: Output directory.
        trajectories: Trajectories to evaluate, for example ``("roe1", "roe2")``.
        domain: Image domain; Table 2 is ``lightbox``.
        recipe: The SHIRT recipe (checkpoint pin, legacy registry, score definition); the
            packaged ``shirt`` recipe by default.
        allow_unpinned_checkpoint: Evaluate a checkpoint other than the recipe's pin.
        limit: Evaluate only the first ``limit`` frames of each trajectory (0: all).
        device: Torch device.
        force: Overwrite existing outputs.

    Returns:
        The top-level summary, also written to ``<out>/summary.json``.

    Raises:
        ValueError: For no trajectories, an unknown domain, an unpinned checkpoint without
            ``allow_unpinned_checkpoint``, a checkpoint not trained on ``layout`` (naming both
            SHA-256s), or a trajectory without labels.
        FileExistsError: If outputs exist and ``force`` is false.
    """
    recipe = recipe if recipe is not None else packaged_recipe()
    shirt_root, out = Path(shirt_root), Path(out)
    check_request(trajectories, domain, recipe)
    _refuse_existing(out, trajectories, domain, force)
    checked = check_model_inputs(checkpoint, layout, recipe, allow_unpinned_checkpoint)
    points_3d = checked.layout.points
    camera = data.load_camera(shirt_root)
    sequences = {roe: _labels(shirt_root / roe, limit) for roe in trajectories}
    model, cfg = load_model(checked.checkpoint, len(points_3d), device)
    summaries: dict[str, dict[str, Any]] = {}
    for roe, labels in sequences.items():
        records, inference_ms = _evaluate_sequence(
            model, cfg, shirt_root, roe, domain, labels, points_3d, camera, device
        )
        summary = summarize_sequence(records, inference_ms, definition=recipe.score)
        folder = sequence_dir(out, roe, domain)
        publish_text(folder / RECORDS_NAME, "".join(json.dumps(r) + "\n" for r in records))
        publish_json(folder / SUMMARY_NAME, summary)
        summaries[roe] = {domain: summary}
    result = {
        "method": METHOD,
        "domain": domain,
        "score_definition": recipe.score,
        "temporal_filtering": False,
        "pose_pipeline": POSE_PIPELINE,
        **checked.summary_fields(),
        "camera_sha256": sha256_file(shirt_root / data.CAMERA_NAME),
        "limit": limit,
        "trajectories": summaries,
    }
    publish_json(out / SUMMARY_NAME, result)
    return result


def _labels(roe_dir: Path, limit: int) -> list[data.Label]:
    labels = data.load_labels(roe_dir)
    if limit > 0:
        labels = labels[:limit]
    if not labels:
        raise ValueError(f"SHIRT trajectory {roe_dir.name} has no labels")
    return labels


# --- 0/180 re-crop candidates -------------------------------------------------------------------

CANDIDATES_METHOD = "candidates"
RAW_SOURCE = "raw"
# The "0/180 TTA": two crop rolls in degrees, each a fresh crop around the raw pose.
ROLL_ANGLES_DEG: tuple[float, ...] = (0.0, 180.0)
DEFAULT_BATCH_SIZE = 32
CROP_BOX_PAD_FRAC = 0.0
CROP_BOX = (
    "75-keypoint extent projected from the raw PnP pose (pad 0, clipped to Nu/Nv), squared by "
    "the checkpoint's crop_pad_factor; gray-114 border; no CLAHE"
)
# Rotation-aux scoring is not ported: these record and candidate fields stay null.
RECORD_AUX_FIELDS = (
    "raw_aux_kl_by_angle",
    "raw_aux_kl_mean",
    "raw_aux_kl_max",
    "raw_aux_mass_within_45deg",
    "aux_top1_gt_error_deg_by_angle",
    "aux_top1_gt_error_deg_best",
)
CANDIDATE_AUX_FIELDS = ("aux_kl_mean", "aux_kl_max", "aux_mass_within_45deg")
CANDIDATE_ERROR_FIELDS = (
    "rotation_deg",
    "translation_m",
    "translation_relative",
    "pose_score",
    "practical_success",
    "loose_success",
)


def roll_source(angle_deg: float) -> str:
    """The candidate source label of a crop roll, ``crop_roll_0`` or ``crop_roll_180``."""
    return f"crop_roll_{angle_deg:g}"


def _candidate_entry(source: str, pose: Mapping[str, Any]) -> dict[str, Any]:
    """One candidate entry, with null rotation-aux fields."""
    return {
        "source": source,
        "rvec": pose["rvec"],
        "tvec": pose["tvec"],
        "ransac_inliers": pose["ransac_inliers"],
        **dict.fromkeys(CANDIDATE_AUX_FIELDS),
        **{field: pose[field] for field in CANDIDATE_ERROR_FIELDS},
    }


def candidate_frame_record(
    framewise_record: Mapping[str, Any], crop_entries: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """One frame's candidate record: ``[raw, crop_roll_*]``, or ``[]`` when the raw PnP failed.

    The re-crop entries are those that solved; a frame without a raw pose keeps its record
    with no candidates, so the frame index stays the time base.
    """
    baseline = framewise_record["baseline"]
    candidates = [] if baseline is None else [_candidate_entry(RAW_SOURCE, baseline)]
    candidates += [dict(entry) for entry in crop_entries]
    return {
        "index": framewise_record["index"],
        "filename": framewise_record["filename"],
        "ground_truth_quaternion": framewise_record["ground_truth_quaternion"],
        "ground_truth_translation": framewise_record["ground_truth_translation"],
        "raw_rotation_deg": None if baseline is None else baseline["rotation_deg"],
        **dict.fromkeys(RECORD_AUX_FIELDS),
        "candidates": candidates,
        "aux_selected_index": None,
        "aux_selected": None,
    }


def summarize_candidates(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Frame count, the frames without candidates and the candidate count of each source."""
    sources = [RAW_SOURCE, *(roll_source(angle) for angle in ROLL_ANGLES_DEG)]
    counts = dict.fromkeys(sources, 0)
    for record in records:
        for candidate in record["candidates"]:
            counts[candidate["source"]] = counts.get(candidate["source"], 0) + 1
    return {
        "frames": len(records),
        "empty_candidate_frames": [r["index"] for r in records if not r["candidates"]],
        "candidates_by_source": counts,
    }


def load_framewise_records(
    framewise_root: Path, roe: str, domain: str, limit: int = 0
) -> list[dict[str, Any]]:
    """The framewise records of one trajectory, in frame order; frame 0 must have a pose.

    Raises:
        FileNotFoundError: Without framewise records.
        ValueError: Without records, or when frame 0 has no raw pose (the filters cannot
            start).
    """
    records = load_records(framewise_root, roe, domain, limit, kind="framewise")
    if records[0]["baseline"] is None:
        raise ValueError(f"no initial pose for {roe} frame 0")
    return records


@dataclass(frozen=True)
class CropGeometry:
    """How the re-crops are cut and decoded (all read from the loaded checkpoint).

    Attributes:
        crop_size: Crop side in pixels (``cfg.model.crop_size``).
        crop_pad_factor: Square padding of the box (``cfg.model.crop_pad_factor``).
        beta: DARK decode temperature (``cfg.loss.keypoint_heatmap_beta``).
    """

    crop_size: int
    crop_pad_factor: float
    beta: float


def _frame_crops(
    image: np.ndarray,
    baseline: Mapping[str, Any],
    points_3d: np.ndarray,
    camera: Any,  # noqa: ANN401 - doper.speedplus.geometry.Camera
    geometry: CropGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """The rolled crops around the raw pose's projected box and their float32 warps."""
    box = projected_pseudo_box(
        np.asarray(baseline["rvec"], dtype=np.float64),
        np.asarray(baseline["tvec"], dtype=np.float64),
        points_3d,
        camera.K,
        camera.dist,
        image_size=(camera.width, camera.height),
        pad_frac=CROP_BOX_PAD_FRAC,
    )
    crops, warps = [], []
    for angle in ROLL_ANGLES_DEG:
        crop, warp = pose_projected_crop(
            image,
            box,
            crop_size=geometry.crop_size,
            crop_pad_factor=geometry.crop_pad_factor,
            angle_deg=angle,
        )
        crops.append(crop)
        warps.append(warp.astype(np.float32))
    return np.stack(crops), np.stack(warps)


def keypoint_logits(model: Any, crops: np.ndarray, device: str) -> np.ndarray:  # noqa: ANN401
    """Heatmap logits of a crop batch: backbone, ``kpt_head.neck``, then ``out(tower(.))``."""
    import torch

    with torch.inference_mode():
        batch = torch.from_numpy(crops).to(device)
        fused = model.kpt_head.neck(model.backbone(batch))
        logits = model.kpt_head.out(model.kpt_head.tower(fused))
        return logits.float().cpu().numpy()


def crop_candidate_entry(
    logits: np.ndarray,
    warp: np.ndarray,
    angle_deg: float,
    framewise_record: Mapping[str, Any],
    points_3d: np.ndarray,
    camera: Any,  # noqa: ANN401
    geometry: CropGeometry,
) -> dict[str, Any] | None:
    """Decode one rolled crop's heatmaps back to the image and solve them with USAC.

    Returns:
        The candidate entry, or None when the solve fails.
    """
    from .. import targets

    coordinates, scores = targets.decode_keypoint_heatmaps(
        logits, geometry.crop_size, beta=geometry.beta, dark=True
    )
    inverse = cv2.invertAffineTransform(warp)
    homogeneous = np.concatenate(
        (coordinates, np.ones((len(coordinates), 1), dtype=np.float32)), axis=1
    )
    keypoints = (homogeneous @ inverse.T).astype(np.float32)
    candidate = argmax_usac_candidate(
        keypoints,
        scores.astype(np.float32),
        points_3d,
        camera.K,
        camera.dist,
        score_thr=DEFAULT_SCORE_THRESHOLD,
    )
    if candidate is None:
        return None
    record = candidate_record(
        candidate,
        np.asarray(framewise_record["ground_truth_quaternion"], dtype=np.float64),
        np.asarray(framewise_record["ground_truth_translation"], dtype=np.float64),
        normalization=GROUND_TRUTH_NORMALIZATION,
    )
    return _candidate_entry(roll_source(angle_deg), record)


def _crop_entries(
    model: Any,  # noqa: ANN401
    sequences: Mapping[str, Sequence[Mapping[str, Any]]],
    shirt_root: Path,
    domain: str,
    points_3d: np.ndarray,
    camera: Any,  # noqa: ANN401
    geometry: CropGeometry,
    batch_size: int,
    device: str,
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    """The solved re-crop entries of every frame with a raw pose, keyed by (roe, index).

    The frames are batched in order (trajectory, then frame; raw-failed frames skipped), so a batch
    can span trajectories, as in the source data loader.
    """
    items = [
        (roe, record)
        for roe, records in sequences.items()
        for record in records
        if record["baseline"] is not None
    ]
    entries: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        prepared = [
            _frame_crops(
                read_image(shirt_root, roe, domain, record["filename"]),
                record["baseline"],
                points_3d,
                camera,
                geometry,
            )
            for roe, record in batch
        ]
        logits = keypoint_logits(model, np.concatenate([crops for crops, _ in prepared]), device)
        logits = logits.reshape(len(batch), len(ROLL_ANGLES_DEG), *logits.shape[1:])
        for item, ((roe, record), (_, warps)) in enumerate(zip(batch, prepared, strict=True)):
            solved = (
                crop_candidate_entry(
                    logits[item, roll], warps[roll], angle, record, points_3d, camera, geometry
                )
                for roll, angle in enumerate(ROLL_ANGLES_DEG)
            )
            entries[(roe, record["index"])] = [entry for entry in solved if entry is not None]
    return entries


def crop_candidates(
    checkpoint: str | Path,
    shirt_root: str | Path,
    layout: str | Path,
    framewise_records: str | Path,
    out: str | Path,
    *,
    trajectories: Sequence[str],
    domain: str = PAPER_DOMAIN,
    recipe: ShirtRecipe | None = None,
    allow_unpinned_checkpoint: bool = False,
    limit: int = 0,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = "cpu",
    force: bool = False,
) -> dict[str, Any]:
    """Generate the 0/180 re-crop pose candidates of each SHIRT frame.

    For every frame with a raw (framewise) pose, the image is cropped twice around the raw
    pose's projected 75-keypoint box, rolled by 0 and 180 degrees, with the checkpoint's crop
    geometry; each crop's heatmaps are DARK-decoded with the checkpoint's heatmap beta, mapped
    back to the image and solved with the shared USAC. Each record holds ``candidates =
    [raw, crop_roll_0, crop_roll_180]`` (a re-crop that does not solve is left out), the
    ``aux_*`` fields null (the rotation-aux scoring is not ported) and the ground-truth errors
    as reporting fields. A frame whose raw PnP failed keeps its record with no candidates.
    Writes ``<out>/<roe>/<domain>/{records.jsonl,summary.json}`` and ``<out>/summary.json``.

    Args:
        checkpoint: SPEED+ heatmap checkpoint (new format, or a legacy one in the registry).
        shirt_root: The SHIRT tree.
        layout: The 75-point ``keypoints_3d.json`` the checkpoint was trained on.
        framewise_records: Output directory of :func:`framewise` with the same checkpoint.
        out: Output directory.
        trajectories: Trajectories, for example ``("roe1", "roe2")``.
        domain: Image domain; Table 2 is ``lightbox``.
        recipe: The SHIRT recipe; the packaged ``shirt`` recipe by default.
        allow_unpinned_checkpoint: Use a checkpoint other than the recipe's pin.
        limit: Only the first ``limit`` frames of each trajectory (0: all).
        batch_size: Frames per forward batch (default 32).
        device: Torch device.
        force: Overwrite existing outputs.

    Returns:
        The top-level summary, also written to ``<out>/summary.json``.

    Raises:
        ValueError: For no trajectories, an unknown domain, a frame 0 without a raw pose
            (``no initial pose for <roe> frame 0``), an unpinned checkpoint without
            ``allow_unpinned_checkpoint``, framewise records of another checkpoint, a checkpoint
            not trained on ``layout``, or a checkpoint without a heatmap keypoint head.
        FileNotFoundError: Without framewise records or an image.
        FileExistsError: If outputs exist and ``force`` is false.
    """
    recipe = recipe if recipe is not None else packaged_recipe()
    shirt_root, out = Path(shirt_root), Path(out)
    framewise_root = Path(framewise_records)
    check_request(trajectories, domain, recipe)
    _refuse_existing(out, trajectories, domain, force, CANDIDATES_METHOD)
    sequences = {
        roe: load_framewise_records(framewise_root, roe, domain, limit) for roe in trajectories
    }
    checked = check_model_inputs(checkpoint, layout, recipe, allow_unpinned_checkpoint)
    framewise_checkpoint = recorded_checkpoint(
        framewise_root, "framewise", checked.checkpoint_sha256
    )
    points_3d = checked.layout.points
    camera = data.load_camera(shirt_root)
    model, cfg = load_model(checked.checkpoint, len(points_3d), device)
    if cfg.model.keypoint_head != "heatmap":
        raise ValueError(
            f"re-crop candidates need a heatmap keypoint head, not {cfg.model.keypoint_head!r}"
        )
    geometry = CropGeometry(
        int(cfg.model.crop_size),
        float(cfg.model.crop_pad_factor),
        heatmap_beta(checked.payload, cfg),
    )
    entries = _crop_entries(
        model, sequences, shirt_root, domain, points_3d, camera, geometry, batch_size, device
    )
    summaries: dict[str, dict[str, Any]] = {}
    for roe, framewise_rows in sequences.items():
        records = [
            candidate_frame_record(row, entries.get((roe, row["index"]), []))
            for row in framewise_rows
        ]
        summary = summarize_candidates(records)
        folder = sequence_dir(out, roe, domain)
        publish_text(folder / RECORDS_NAME, "".join(json.dumps(r) + "\n" for r in records))
        publish_json(folder / SUMMARY_NAME, summary)
        summaries[roe] = {domain: summary}
    result = {
        "method": CANDIDATES_METHOD,
        "domain": domain,
        "rotation_tta_angles_deg": list(ROLL_ANGLES_DEG),
        "crop_box": CROP_BOX,
        "crop_size": geometry.crop_size,
        "crop_pad_factor": geometry.crop_pad_factor,
        "keypoint_heatmap_beta": geometry.beta,
        "keypoint_score_threshold": DEFAULT_SCORE_THRESHOLD,
        "rotation_aux_scoring": None,
        "ground_truth_used_for_candidates": False,
        **checked.summary_fields(),
        "framewise_checkpoint_sha256": framewise_checkpoint,
        "camera_sha256": sha256_file(shirt_root / data.CAMERA_NAME),
        "limit": limit,
        "batch_size": batch_size,
        "trajectories": summaries,
    }
    publish_json(out / SUMMARY_NAME, result)
    return result
