# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic SHIRT candidate tracks for the Table 2 filters, with injected 180-degree flips.

Each trajectory has ``FRAMES`` frames over a circular-orbit servicer with a slowly spinning
target (``make_metadata``); the true camera-frame pose of frame ``i`` follows
``synthetic_shirt.make_truth_pose``. Every frame carries candidate records in the layout of
``doper.shirt.candidates.crop_candidates`` (``[raw, crop_roll_0, crop_roll_180]``, each the
true pose plus noise) with scripted events (``EVENTS``):

* ``raw_flip``: the raw pose and ``crop_roll_180`` are rotated by 180 degrees about the body
  x axis, ``crop_roll_0`` is right;
* ``crop_flip``: only ``crop_roll_180`` is flipped;
* ``all_flip``: every candidate is flipped, so the NIS gate rejects them all;
* ``missing_crop``: ``crop_roll_180`` did not solve and is left out.

The keypoint cache (the ``heatmap_measurements.npz`` layout) holds each frame's projected
``LANDMARKS`` body landmarks plus noise with per-landmark covariances, an outlier landmark on
every third frame and no detection on ``UNAVAILABLE`` frames.

The research filters (research source snapshot:
``evaluate_shirt_mahalanobis_candidates.evaluate_policy``,
``evaluate_shirt_mahalanobis_rts.process_sequence`` and
``evaluate_shirt_direct_keypoint_ukf.evaluate_trajectory``) and the research summary helpers
run on this fixture through ``run_source``; ``golden/source_shirt_policies.json`` holds their
canonical outputs with the library versions they were produced with. Regenerate it with
``python tests/unit/shirt_tracks.py <research-source-dir>``, an extracted copy of the research
source snapshot. The ``shirt_*`` filter comparisons load this module by path. The module needs
numpy, OpenCV, SciPy and ``synthetic_shirt`` (which imports the release filter only for
``quaternion_to_rotation``).
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _sibling(name: str) -> Any:  # noqa: ANN401
    """Import the test module ``name`` next to this file by path (for loading by path)."""
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sf = _sibling("shirt_frames")
shirt = sf.synthetic_shirt()

TRAJECTORIES = ("roe1", "roe2")
DOMAIN = "lightbox"
FRAMES = 24
CAMERA_STEP_S = 5.0
CAMERA_SIZE = 1024
CAMERA_MATRIX = shirt.CAMERA_MATRIX
DISTORTION = np.array([-0.05, 0.01, 0.0, 0.0, 0.0])
LANDMARKS = 12
TARGET_SPIN_RAD_S = 2.0e-3
RAW_NOISE_DEG = 0.5
CROP_NOISE_DEG = 0.3
TRANSLATION_NOISE_M = 0.01
KEYPOINT_NOISE_PX = 0.8
KEYPOINT_STD_PX = (1.0, 2.5)
OUTLIER_EVERY = 3
OUTLIER_PX = (35.0, -20.0)
FLIP = np.diag([1.0, -1.0, -1.0])
SLOTS = {"roe1": 0, "roe2": 1}
EVENTS: dict[str, dict[int, str]] = {
    "roe1": {4: "raw_flip", 7: "crop_flip", 9: "missing_crop", 11: "raw_flip", 14: "all_flip",
             17: "raw_flip"},
    "roe2": {3: "raw_flip", 8: "all_flip", 12: "crop_flip", 15: "raw_flip", 20: "all_flip"},
}  # fmt: skip
UNAVAILABLE: dict[str, tuple[int, ...]] = {"roe1": (10,), "roe2": (6,)}
# A release-only scenario: a later frame whose raw PnP failed has no candidates.
EMPTY_FRAME = ("roe1", 13)
RAW, CROP_0, CROP_180 = "raw", "crop_roll_0", "crop_roll_180"
NULL_RECORD_FIELDS = (
    "raw_aux_kl_by_angle", "raw_aux_kl_mean", "raw_aux_kl_max", "raw_aux_mass_within_45deg",
    "aux_top1_gt_error_deg_by_angle", "aux_top1_gt_error_deg_best", "aux_selected_index",
    "aux_selected",
)  # fmt: skip
GOLDEN_NAME = "source_shirt_policies.json"
REGENERATE = (
    "regenerate it with `python tests/unit/shirt_tracks.py <research-source-dir>` (the research "
    "SHIRT filters in the research source snapshot)"
)
SUMMARY_POSES = 40
GATE_LABEL = "p99"


def _unit(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array / np.linalg.norm(array)


def _hamilton(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = first
    w2, x2, y2, z2 = second
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])  # fmt: skip


def make_metadata(slot: int) -> dict[str, Any]:
    """Servicer ephemeris of ``FRAMES`` samples and a target spinning about a fixed axis."""
    base = shirt.make_metadata()
    mean_motion = float(np.sqrt(shirt.EARTH_MU_M3_S2 / shirt.ORBIT_RADIUS_M**3))
    times = np.arange(FRAMES) * CAMERA_STEP_S
    angle = mean_motion * times + 0.1 * slot
    zero = 0.0 * angle
    radius = shirt.ORBIT_RADIUS_M
    position = radius * np.stack([np.cos(angle), np.sin(angle), zero], axis=1)
    velocity = radius * mean_motion * np.stack([-np.sin(angle), np.cos(angle), zero], axis=1)
    spin = shirt.SERVICER_SPIN_RAD_S * times
    rng = np.random.default_rng(40 + slot)
    start, axis = _unit(rng.normal(size=4)), _unit(rng.normal(size=3))
    target = [
        _hamilton(start, np.array([math.cos(a / 2), *(math.sin(a / 2) * axis)])).tolist()
        for a in TARGET_SPIN_RAD_S * times
    ]
    base["pSim"] = {"cam_step": CAMERA_STEP_S}
    base["sAbsState"] = {
        "rv_eci2com_eci": np.concatenate([position, velocity], axis=1).tolist(),
        "q_eci2pri": [[float(np.cos(a / 2)), 0.0, 0.0, float(np.sin(a / 2))] for a in spin],
    }
    base["tRelState"] = {"q_spri2tpri": target}
    return base


def make_points_3d() -> np.ndarray:
    """``LANDMARKS`` body-frame landmarks around the origin, in meters."""
    return np.random.default_rng(7).uniform(-0.5, 0.5, size=(LANDMARKS, 3))


def _rotate(rvec: np.ndarray, degrees: float, rng: np.random.Generator) -> np.ndarray:
    axis = _unit(rng.normal(size=3))
    noise = cv2.Rodrigues(axis * math.radians(degrees) * rng.uniform(0.3, 1.0))[0]
    return noise @ cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]


def _errors(rotation: np.ndarray, tvec: np.ndarray, truth: tuple[np.ndarray, np.ndarray]
            ) -> dict[str, Any]:  # fmt: skip
    """The candidate error fields of a pose against the truth (reporting only)."""
    relative = rotation @ cv2.Rodrigues(truth[0])[0].T
    rotation_deg = math.degrees(math.acos(float(np.clip((np.trace(relative) - 1) / 2, -1, 1))))
    translation_m = float(np.linalg.norm(tvec - truth[1].reshape(3)))
    relative_t = translation_m / float(np.linalg.norm(truth[1]))
    return {
        "rotation_deg": rotation_deg, "translation_m": translation_m,
        "translation_relative": relative_t, "pose_score": math.radians(rotation_deg) + relative_t,
        "practical_success": rotation_deg < 5.0 and relative_t < 0.02,
        "loose_success": rotation_deg < 10.0 and relative_t < 0.05,
    }  # fmt: skip


def _candidate(source: str, truth: tuple[np.ndarray, np.ndarray], flipped: bool,
               noise_deg: float, rng: np.random.Generator) -> dict[str, Any]:  # fmt: skip
    rotation = _rotate(truth[0], noise_deg, rng)
    if flipped:
        rotation = rotation @ FLIP
    tvec = truth[1].reshape(3) + rng.normal(scale=TRANSLATION_NOISE_M, size=3)
    return {
        "source": source,
        "rvec": cv2.Rodrigues(rotation)[0].reshape(3).tolist(),
        "tvec": tvec.tolist(),
        "ransac_inliers": int(rng.integers(8, LANDMARKS + 1)),
        "aux_kl_mean": None, "aux_kl_max": None, "aux_mass_within_45deg": None,
        **_errors(rotation, tvec, truth),
    }  # fmt: skip


def _frame_candidates(event: str | None, truth: tuple[np.ndarray, np.ndarray],
                      rng: np.random.Generator) -> list[dict[str, Any]]:  # fmt: skip
    flips = {
        None: (False, False, False), "raw_flip": (True, False, True),
        "crop_flip": (False, False, True), "all_flip": (True, True, True),
        "missing_crop": (False, False, False),
    }[event]  # fmt: skip
    noise = (RAW_NOISE_DEG, CROP_NOISE_DEG, CROP_NOISE_DEG)
    entries = [
        _candidate(source, truth, flipped, degrees, rng)
        for source, flipped, degrees in zip((RAW, CROP_0, CROP_180), flips, noise, strict=True)
    ]
    return entries[:2] if event == "missing_crop" else entries


def truth_poses(roe: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """The true camera-frame ``(rvec (3, 1), tvec (3,))`` of every frame of ``roe``."""
    metadata = make_metadata(SLOTS[roe])
    return [shirt.make_truth_pose(metadata, index) for index in range(FRAMES)]


def candidate_records(roe: str) -> list[dict[str, Any]]:
    """The candidate records of ``roe`` in frame order."""
    rng = np.random.default_rng(1000 + SLOTS[roe])
    records = []
    for index, truth in enumerate(truth_poses(roe)):
        candidates = _frame_candidates(EVENTS[roe].get(index), truth, rng)
        records.append({
            "index": index,
            "filename": sf.filename(index),
            "ground_truth_quaternion": sf.speed_quaternion(truth[0]).tolist(),
            "ground_truth_translation": np.asarray(truth[1]).reshape(3).tolist(),
            "raw_rotation_deg": candidates[0]["rotation_deg"],
            **dict.fromkeys(NULL_RECORD_FIELDS),
            "candidates": candidates,
        })  # fmt: skip
    return records


def with_empty_frame(records: list[dict[str, Any]], index: int) -> list[dict[str, Any]]:
    """A copy of ``records`` whose frame ``index`` lost its raw pose (no candidates)."""
    copied = json.loads(json.dumps(records))
    copied[index]["candidates"] = []
    copied[index]["raw_rotation_deg"] = None
    return copied


def heatmap_cache(roe: str) -> dict[str, np.ndarray]:
    """The keypoint cache of ``roe`` in the ``heatmap_measurements.npz`` layout."""
    rng = np.random.default_rng(2000 + SLOTS[roe])
    points_3d = make_points_3d()
    points = np.full((FRAMES, LANDMARKS, 2), np.nan, dtype=np.float32)
    covariances = np.full((FRAMES, LANDMARKS, 2, 2), np.nan, dtype=np.float32)
    confidence = np.full((FRAMES, LANDMARKS), np.nan, dtype=np.float32)
    boxes = np.full((FRAMES, 4), np.nan, dtype=np.float32)
    scores = np.full(FRAMES, np.nan, dtype=np.float32)
    available = np.zeros(FRAMES, dtype=np.uint8)
    for index, (rvec, tvec) in enumerate(truth_poses(roe)):
        if index in UNAVAILABLE[roe]:
            continue
        projected = cv2.projectPoints(points_3d, rvec, tvec, CAMERA_MATRIX, DISTORTION)[0]
        pixels = projected.reshape(-1, 2) + rng.normal(scale=KEYPOINT_NOISE_PX, size=(LANDMARKS, 2))
        if index % OUTLIER_EVERY == 0:
            pixels[index % LANDMARKS] += OUTLIER_PX
        std = rng.uniform(*KEYPOINT_STD_PX, size=(LANDMARKS, 2))
        correlation = rng.uniform(-0.3, 0.3, size=LANDMARKS)
        cross = correlation * std[:, 0] * std[:, 1]
        covariances[index] = np.stack(
            [np.stack([std[:, 0] ** 2, cross], -1), np.stack([cross, std[:, 1] ** 2], -1)], 1
        )
        points[index] = pixels
        confidence[index] = rng.uniform(0.2, 0.9, size=LANDMARKS)
        boxes[index] = [*pixels.min(axis=0), *pixels.max(axis=0)]
        scores[index] = 0.9
        available[index] = 1
    return {
        "filenames": np.asarray([sf.filename(index) for index in range(FRAMES)]),
        "points": points, "covariances": covariances, "confidence": confidence,
        "boxes": boxes, "detection_scores": scores, "available": available,
    }  # fmt: skip


def write_tree(root: Path) -> dict[str, Path]:
    """Write the SHIRT camera and metadata, candidate records, caches and the layout.

    Returns:
        ``shirt`` (the SHIRT tree), ``candidates`` (records at ``<roe>/<domain>/records.jsonl``),
        ``caches`` (``<roe>/<domain>/heatmap_measurements.npz``) and ``layout``.
    """
    paths = {name: root / name for name in ("shirt", "candidates", "caches")}
    paths["layout"] = root / "keypoints_3d.json"
    paths["shirt"].mkdir(parents=True, exist_ok=True)
    camera = {"cameraMatrix": CAMERA_MATRIX.tolist(), "distCoeffs": DISTORTION.tolist(),
              "Nu": CAMERA_SIZE, "Nv": CAMERA_SIZE}  # fmt: skip
    (paths["shirt"] / "camera.json").write_text(json.dumps(camera))
    paths["layout"].write_text(json.dumps({"keypoints_3d": make_points_3d().tolist()}))
    for roe in TRAJECTORIES:
        (paths["shirt"] / roe).mkdir(exist_ok=True)
        (paths["shirt"] / roe / "metadata.json").write_text(json.dumps(make_metadata(SLOTS[roe])))
        records = paths["candidates"] / roe / DOMAIN / "records.jsonl"
        records.parent.mkdir(parents=True, exist_ok=True)
        records.write_text("".join(json.dumps(r) + "\n" for r in candidate_records(roe)))
        cache = paths["caches"] / roe / DOMAIN / "heatmap_measurements.npz"
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **heatmap_cache(roe))
    return paths


IMAGE_SIDE = 16
CODE_BASE, CODE_FRAME_STEP, CODE_TRAJECTORY_STEP = 5, 5, 120


def image_code(roe: str, index: int) -> int:
    """Gray level of frame ``index`` of ``roe`` in the images ``write_images`` writes."""
    return CODE_BASE + CODE_FRAME_STEP * index + CODE_TRAJECTORY_STEP * SLOTS[roe]


def frame_of_code(code: int) -> tuple[str, int]:
    """The ``(roe, index)`` an image gray level codes."""
    slot, rest = divmod(code - CODE_BASE, CODE_TRAJECTORY_STEP)
    return TRAJECTORIES[slot], rest // CODE_FRAME_STEP


def write_images(shirt_root: Path) -> None:
    """Uniform images coding each frame (for a stub detector), under the SHIRT layout."""
    for roe in TRAJECTORIES:
        images = shirt_root / roe / DOMAIN / "images"
        images.mkdir(parents=True, exist_ok=True)
        for index in range(FRAMES):
            image = np.full((IMAGE_SIDE, IMAGE_SIDE, 3), image_code(roe, index), dtype=np.uint8)
            cv2.imwrite(str(images / sf.filename(index)), image)


def summary_inputs() -> list[dict[str, Any]]:
    """Fixed pose records for the summary helpers: estimated and true poses, some flipped."""
    rng = np.random.default_rng(31)
    rows = []
    for index in range(SUMMARY_POSES):
        truth_rvec = rng.normal(size=3)
        truth_t = np.array([0.2, -0.1, 7.0]) + rng.normal(scale=0.5, size=3)
        rotation = _rotate(truth_rvec, 2.0 + 4.0 * (index % 5), rng)
        if index % 7 == 3:
            rotation = rotation @ FLIP
        rows.append({
            "rvec": cv2.Rodrigues(rotation)[0].reshape(3), "tvec": truth_t + rng.normal(
                scale=0.05 + 0.1 * (index % 3), size=3),
            "ground_truth_quaternion": sf.speed_quaternion(truth_rvec),
            "ground_truth_rvec": truth_rvec,
            "ground_truth_translation": truth_t,
        })  # fmt: skip
    return rows


# --- canonical forms shared by the unit test, the golden and the parity cases ------------------


def clean(value: Any) -> Any:  # noqa: ANN401
    """``value`` as plain JSON: NaN becomes None, numpy scalars and arrays become Python."""
    if isinstance(value, Mapping):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [
            clean(item) for item in (value.tolist() if isinstance(value, np.ndarray) else value)
        ]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(value) else float(value)
    return value


NIS_FRAME_KEYS = ("index", "selected_index", "selected_source", "candidate_nis", "filtered")
NIS_SUMMARY_KEYS = (
    "count", "rotation_deg_mean", "rotation_deg_median", "rotation_deg_p95", "rotation_deg_max",
    "translation_m_mean", "translation_relative_mean", "pose_score_mean",
    "practical_recall_5deg_2pct", "loose_recall_10deg_5pct", "rotation_gt30_count",
    "rotation_gt60_count", "rotation_gt150_count", "nis_gate", "rejected_measurements",
    "selected_sources", "selected_candidate_gt30_count", "introduced_candidate_gt30_count",
    "raw_flip_count", "raw_flip_selected_good_candidate", "raw_flip_rejected",
    "raw_flip_selected_bad_candidate", "raw_candidate_nis_on_good_frames",
    "raw_candidate_nis_on_flip_frames", "selected_candidate_nis", "minimum_candidate_nis",
)  # fmt: skip
RTS_RECORD_KEYS = ("index", "ground_truth_rvec", "selected_index", "selected_source",
                   "selected_nis", "causal", "rts")  # fmt: skip
KEYPOINT_RECORD_KEYS = (
    "index", "baseline_selected_index", "baseline_candidate_nis", "direct_update_kind",
    "accepted_keypoint_count", "accepted_keypoints", "keypoint_nis", "keypoint_nis_finite_mean",
    "baseline", "direct_keypoint",
)  # fmt: skip


def _pick(row: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return clean({key: row[key] for key in keys})


def canonical_nis(frames: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> dict:
    """The compared part of a minimum-NIS run: per-frame decisions and poses, the summary."""
    return {"frames": [_pick(f, NIS_FRAME_KEYS) for f in frames],
            "summary": _pick(summary, NIS_SUMMARY_KEYS)}  # fmt: skip


# Release-only summary fields (the empty-frame count and the pass count).
RELEASE_ONLY_RTS_KEYS = ("rts_passes", "empty_candidate_frames", "prediction_only_frames")


def canonical_rts(records: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> dict:
    """The compared part of a causal plus one-pass RTS run (records and the summary)."""
    kept = {key: value for key, value in summary.items() if key not in RELEASE_ONLY_RTS_KEYS}
    return {"records": [_pick(r, RTS_RECORD_KEYS) for r in records], "summary": clean(kept)}


def canonical_keypoint(records: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> dict:
    """The compared part of a keypoint-UKF run (records and the whole summary)."""
    return {"records": [_pick(r, KEYPOINT_RECORD_KEYS) for r in records],
            "summary": clean(summary)}  # fmt: skip


def run_summaries(helpers: Mapping[str, Callable[..., Any]]) -> dict[str, Any]:
    """The summary helpers on ``summary_inputs``: ``pose_metrics``, ``summarize``,
    ``summarize_pose_values``, ``method_summary`` and ``paper_style_components``."""
    rows = summary_inputs()
    metrics = [
        helpers["pose_metrics"](row["rvec"], row["tvec"], row["ground_truth_quaternion"],
                                row["ground_truth_translation"])
        for row in rows
    ]  # fmt: skip
    records = [
        {"m": metric, "ground_truth_translation": row["ground_truth_translation"].tolist(),
         "ground_truth_rvec": row["ground_truth_rvec"].tolist()}
        for metric, row in zip(metrics, rows, strict=True)
    ]  # fmt: skip
    return clean({
        "pose_metrics": metrics,
        "summarize": helpers["summarize"](records, "m"),
        "summarize_pose_values": helpers["summarize_pose_values"](metrics),
        "method_summary": helpers["method_summary"](records, "m"),
        "paper_style_components": helpers["paper_style_components"](records, "m"),
    })  # fmt: skip


# --- the research side ----------------------------------------------------------------------


def source_nis_rank(evaluate_policy: Callable[..., Any], gate: float, paths: Mapping[str, Path]
                    ) -> dict[str, Any]:  # fmt: skip
    """The research ``nis_rank_p99`` policy of every trajectory, canonical."""
    spec = {"name": "nis_rank_p99", "method": "nis_rank", "gate": gate}
    result = {}
    for roe in TRAJECTORIES:
        summary, frames = evaluate_policy(
            candidate_records(roe), paths["shirt"] / roe / "metadata.json", spec, 0.4,
            keep_frame_records=True,
        )  # fmt: skip
        result[roe] = canonical_nis(frames, summary)
    return result


def source_rts(process_sequence: Callable[..., Any], gate: float, paths: Mapping[str, Path],
               work: Path) -> dict[str, Any]:  # fmt: skip
    """The research causal plus one-pass RTS run of every trajectory, canonical."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    result = {}
    for roe in TRAJECTORIES:
        out = work / "rts" / GATE_LABEL / roe
        summary = process_sequence(
            paths["candidates"] / roe / DOMAIN / "records.jsonl",
            paths["shirt"] / roe / "metadata.json", out, GATE_LABEL, gate,
        )  # fmt: skip
        records = [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()]
        result[roe] = canonical_rts(records, summary)
    return result


def source_keypoint(evaluate_trajectory: Callable[..., Any], gates: tuple[float, float],
                    paths: Mapping[str, Path], work: Path) -> dict[str, Any]:  # fmt: skip
    """The research direct keypoint UKF (and its minimum-NIS baseline), canonical."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    args = Namespace(shirt_root=paths["shirt"], output_dir=work / "keypoint", domain=DOMAIN,
                     overlay_count=0, baseline_nis_gate=gates[0], landmark_nis_gate=gates[1],
                     covariance_scale=1.0)  # fmt: skip
    result = {}
    for roe in TRAJECTORIES:
        summary, records = evaluate_trajectory(
            roe, candidate_records(roe), heatmap_cache(roe), make_points_3d(), CAMERA_MATRIX,
            DISTORTION, args,
        )  # fmt: skip
        result[roe] = canonical_keypoint(records, summary)
    return result


def library_versions() -> dict[str, str]:
    """The builds the golden depends on (the filter's linear algebra and OpenCV)."""
    import scipy

    return {"numpy": np.__version__, "opencv": cv2.__version__, "scipy": scipy.__version__}


def run_source(source_root: Path, work: Path) -> dict[str, Any]:
    """Run the research filters and summary helpers on the fixture."""
    import contextlib
    import io

    sys.path.insert(0, str(source_root))
    os.environ.setdefault("MPLBACKEND", "Agg")
    import scripts.evaluate_shirt_auxiliary_flip_gate as flip_gate
    import scripts.evaluate_shirt_direct_keypoint_ukf as keypoint
    import scripts.evaluate_shirt_ephemeris_smoother as ephemeris
    import scripts.evaluate_shirt_mahalanobis_candidates as candidates
    import scripts.evaluate_shirt_mahalanobis_rts as rts
    import scripts.evaluate_shirt_rts_smoother as rts_smoother

    gate = candidates.CHI2_6DOF["p99"]
    paths = write_tree(work / "tree")
    with contextlib.redirect_stdout(io.StringIO()):
        payload = {
            "nis_rank": source_nis_rank(candidates.evaluate_policy, gate, paths),
            "rts": source_rts(rts.process_sequence, gate, paths, work),
            "keypoint_ukf": source_keypoint(
                keypoint.evaluate_trajectory, (gate, keypoint.CHI2_2DOF_P99), paths, work),
            "summaries": run_summaries({
                "pose_metrics": rts_smoother.pose_metrics, "summarize": rts_smoother.summarize,
                "summarize_pose_values": flip_gate.summarize_pose_values,
                "method_summary": rts.method_summary,
                "paper_style_components": ephemeris.paper_style_components,
            }),
        }  # fmt: skip
    return payload


def write_source_golden(source_root: Path, golden: Path) -> None:
    """Write ``source_shirt_policies.json`` from the research filters."""
    import tempfile

    with tempfile.TemporaryDirectory() as temporary:
        payload = {**run_source(source_root, Path(temporary).resolve()),
                   "versions": library_versions()}  # fmt: skip
    (golden / GOLDEN_NAME).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    write_source_golden(Path(sys.argv[1]), Path(__file__).resolve().parent / "golden")
