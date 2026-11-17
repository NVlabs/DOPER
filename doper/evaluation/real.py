# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Label-free real-world validation scorecard for a DOPER checkpoint.

Synthetic validation (``kp_err_px`` in training) only measures how well the head fits the
*synthetic* keypoint layout; it is blind to the sim-to-real gap and to discrete pose flips, the
two things that break on real captures. Real keypoint labels rarely exist, so this scores a
recorded capture with *geometry* instead of labels:

* detection rate: did the detector find the object at all,
* PnP-solve rate: did enough keypoints agree for a pose,
* reprojection error (px): 2D self-consistency of the keypoints,
* depth-plausibility rate: fraction of poses with a physically sane z,
* temporal flip / jitter rate: frame-to-frame pose jumps, the proxy for the discrete flips the
  reprojection term cannot see (a flipped pose can still reproject well).

None of these need ground truth. Reprojection alone is flip-blind, which is why the
depth-plausibility and temporal-jump terms are reported alongside it. Run on a *held-out*
capture (not one you have been tuning against) to avoid overfitting to a single recording.

Example::

    python -m doper.evaluation.real \
        --checkpoint runs/best.pth --data-dir recordings/cam_0 \
        --keypoints-3d data/keypoints_3d.json
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from doper.config import DataConfig, DoperConfig, ModelConfig
from doper.console import log
from doper.infer import Detection, infer_image, infer_image_tta, load_model
from doper.keypoints import load_keypoints_3d
from doper.model import DoperNet
from doper.performance import resolve_device
from doper.recorded import frame_index, list_color_frames, load_intrinsics

from .io import write_json

MAX_DETECTIONS = 5
CENTIMETERS_PER_METER = 100.0
_MODEL_DEFAULTS = ModelConfig()


@dataclass(frozen=True)
class ScorecardOptions:
    """Thresholds of the scorecard.

    Attributes:
        score_thr: Minimum detection score.
        kp_score_thr: Minimum keypoint score counted as a confident keypoint.
        box_pad: Fractional padding of the detected box before the keypoint crop.
        rot_tta: Evaluate with 90-degree rotation test-time augmentation.
        depth_range: Plausible depth window ``(min, max)`` in meters.
        flip_rot_deg: Frame-to-frame rotation jump (degrees) counted as a flip.
        jump_depth_cm: Frame-to-frame depth jump (centimeters) counted as a jump.
    """

    score_thr: float = 0.3
    kp_score_thr: float = 0.3
    box_pad: float = 0.0
    rot_tta: bool = False
    depth_range: tuple[float, float] = (0.03, 1.0)
    flip_rot_deg: float = 60.0
    jump_depth_cm: float = 5.0


def relative_rotation_deg(rvec_a: np.ndarray, rvec_b: np.ndarray) -> float:
    """Return the geodesic angle (degrees) between two Rodrigues rotation vectors."""
    rot_a, _ = cv2.Rodrigues(rvec_a.reshape(3, 1))
    rot_b, _ = cv2.Rodrigues(rvec_b.reshape(3, 1))
    relative = rot_a.T @ rot_b
    cos = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _percent(count: int, total: int) -> float:
    """Return ``100 * count / total``, or 0 for an empty population."""
    return 100.0 * count / total if total > 0 else 0.0


def _median(values: Sequence[float]) -> float | None:
    """Return the median of ``values``, or None when empty."""
    return float(np.median(values)) if values else None


def _mean(values: Sequence[float]) -> float | None:
    """Return the mean of ``values``, or None when empty."""
    return float(np.mean(values)) if values else None


def _frame_record(det: Detection | None, index: int, kp_score_thr: float) -> dict[str, Any]:
    """Summarize the best detection of one frame (or its absence) as a per-frame record."""
    has_pose = det is not None and det.rvec is not None
    kp_scores = det.keypoint_scores if det is not None else None
    reproj = det.reproj_error_px if det is not None else None
    return {
        "idx": index,
        "detected": det is not None,
        "pnp_ok": has_pose,
        "reproj_px": None if reproj is None else float(reproj),
        "depth_cm": (
            float(det.tvec.reshape(3)[2] * CENTIMETERS_PER_METER)
            if det is not None and det.tvec is not None
            else None
        ),
        "rvec": det.rvec.reshape(3).tolist() if det is not None and det.rvec is not None else None,
        # Label-free sim-to-real localization signal: how many keypoints the model fires
        # confidently, and its mean confidence. Flip-blind, but a good proxy for "does the
        # model even see the object clearly on real imagery".
        "n_kp_conf": int((kp_scores >= kp_score_thr).sum()) if kp_scores is not None else 0,
        "kp_conf_mean": float(kp_scores.mean()) if kp_scores is not None else None,
    }


def _detect_frame(
    model: DoperNet,
    image: np.ndarray,
    cfg: DoperConfig,
    device: str,
    keypoints_3d: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    options: ScorecardOptions,
) -> Detection | None:
    """Return the highest-scoring detection of ``image``, or None when nothing is detected."""
    common: dict[str, Any] = {
        "score_thr": options.score_thr,
        "max_det": MAX_DETECTIONS,
        "keypoints_3d": keypoints_3d,
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "box_pad": options.box_pad,
    }
    if options.rot_tta:
        dets = infer_image_tta(model, image, cfg, device, depth_range=options.depth_range, **common)
    else:
        dets = infer_image(model, image, cfg, device, **common)
    return max(dets, key=lambda d: d.score) if dets else None


def _temporal_steps(
    per_frame: Sequence[dict[str, Any]], options: ScorecardOptions
) -> tuple[list[float], list[float], int]:
    """Return rotation steps, depth steps and the flip count over consecutive posed frames."""
    rot_steps: list[float] = []
    depth_steps: list[float] = []
    n_flip = 0
    for a, b in zip(per_frame[:-1], per_frame[1:], strict=True):
        if not (a["pnp_ok"] and b["pnp_ok"]):
            continue
        rot = relative_rotation_deg(np.asarray(a["rvec"]), np.asarray(b["rvec"]))
        depth = abs(b["depth_cm"] - a["depth_cm"])
        rot_steps.append(rot)
        depth_steps.append(depth)
        if rot > options.flip_rot_deg or depth > options.jump_depth_cm:
            n_flip += 1
    return rot_steps, depth_steps, n_flip


def _summarize(
    per_frame: Sequence[dict[str, Any]], num_keypoints: int, options: ScorecardOptions
) -> dict[str, Any]:
    """Aggregate per-frame records into the scorecard metrics."""
    depth_lo, depth_hi = options.depth_range
    n = len(per_frame)
    detected = [r for r in per_frame if r["detected"]]
    posed = [r for r in per_frame if r["pnp_ok"]]
    reprojs = [r["reproj_px"] for r in posed]
    n_depth_ok = sum(
        1
        for r in posed
        if depth_lo * CENTIMETERS_PER_METER <= r["depth_cm"] <= depth_hi * CENTIMETERS_PER_METER
    )
    rot_steps, depth_steps, n_flip = _temporal_steps(per_frame, options)
    n_pairs = len(rot_steps)
    kp_conf = [r["kp_conf_mean"] for r in detected if r["kp_conf_mean"] is not None]
    return {
        "rot_tta": bool(options.rot_tta),
        "frames": n,
        "det_rate_pct": _percent(len(detected), n),
        "pnp_rate_pct": _percent(len(posed), n),
        "reproj_px_median": _median(reprojs),
        "reproj_px_mean": _mean(reprojs),
        "mean_kp_above_conf": float(np.mean([r["n_kp_conf"] for r in detected]))
        if detected
        else 0.0,
        "kp_conf_median": _median(kp_conf),
        "num_keypoints": num_keypoints,
        "depth_plausible_pct_of_frames": _percent(n_depth_ok, n),
        "depth_plausible_pct_of_pnp": _percent(n_depth_ok, len(posed)),
        "temporal_pairs": n_pairs,
        "flip_jump_rate_pct": _percent(n_flip, n_pairs),
        "rot_step_deg_median": _median(rot_steps),
        "depth_step_cm_median": _median(depth_steps),
    }


def evaluate_recording(
    model: DoperNet,
    cfg: DoperConfig,
    device: str,
    frames: Sequence[Path],
    keypoints_3d: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    options: ScorecardOptions | None = None,
) -> dict[str, Any]:
    """Score a model on recorded frames with label-free geometric metrics.

    Frames without a detection stay in the denominator of every rate; only frames that
    cannot be read from disk are skipped.

    Args:
        model: Loaded model (see ``doper.infer.load_model``).
        cfg: Config the model was loaded with.
        device: Torch device string of the model.
        frames: Ordered ``NNNNNN_color.png`` frame paths.
        keypoints_3d: ``(K, 3)`` object keypoints for PnP.
        camera_matrix: ``(3, 3)`` camera intrinsics.
        dist_coeffs: Distortion coefficients.
        options: Scorecard thresholds; defaults to ``ScorecardOptions()``.

    Returns:
        ``{"scorecard": {...}, "per_frame": [...]}`` where ``scorecard`` holds the aggregate
        metrics (rates in percent, ``None`` for undefined medians) and ``per_frame`` one
        record per readable frame.
    """
    options = options or ScorecardOptions()
    per_frame = []
    unreadable = 0
    for path in frames:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            unreadable += 1
            continue
        det = _detect_frame(
            model, image, cfg, device, keypoints_3d, camera_matrix, dist_coeffs, options
        )
        per_frame.append(_frame_record(det, frame_index(path), options.kp_score_thr))
    if unreadable:
        log(f"WARNING: skipped {unreadable} unreadable frame(s); rates cover readable frames only")
    return {
        "scorecard": _summarize(per_frame, cfg.model.num_keypoints, options),
        "per_frame": per_frame,
    }


def _log_scorecard(scorecard: dict[str, Any], options: ScorecardOptions) -> None:
    """Print the human-readable scorecard."""
    depth_lo, depth_hi = options.depth_range
    log("=== real scorecard ===")
    log(f"frames              : {scorecard['frames']}")
    log(f"detection rate      : {scorecard['det_rate_pct']:.1f}%")
    log(f"PnP solve rate      : {scorecard['pnp_rate_pct']:.1f}%")
    if scorecard["reproj_px_median"] is not None:
        log(
            f"reproj px           : median={scorecard['reproj_px_median']:.2f} "
            f"mean={scorecard['reproj_px_mean']:.2f}"
        )
    kp_conf = scorecard["kp_conf_median"]
    line = (
        f"kp above conf>={options.kp_score_thr:.2f}  : "
        f"{scorecard['mean_kp_above_conf']:.1f} / {scorecard['num_keypoints']}"
    )
    log(line if kp_conf is None else f"{line} (mean conf median={kp_conf:.3f})")
    log(
        f"depth plausible     : {scorecard['depth_plausible_pct_of_frames']:.1f}% of frames "
        f"({scorecard['depth_plausible_pct_of_pnp']:.1f}% of poses) in "
        f"[{depth_lo * 100:.0f},{depth_hi * 100:.0f}]cm"
    )
    if scorecard["temporal_pairs"]:
        log(
            f"flip/jump rate      : {scorecard['flip_jump_rate_pct']:.1f}% "
            f"(>{options.flip_rot_deg:.0f}deg or >{options.jump_depth_cm:.0f}cm "
            "between adjacent poses)"
        )
        log(
            f"temporal jitter     : rot_step median={scorecard['rot_step_deg_median']:.1f}deg "
            f"depth_step median={scorecard['depth_step_cm_median']:.2f}cm"
        )


def _build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser of ``main``."""
    parser = argparse.ArgumentParser(
        description="Label-free real-world scorecard for DOPER",
        epilog="Architecture options are fallbacks for checkpoints without cfg metadata.",
    )
    defaults = ScorecardOptions()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--data-dir", required=True, help="folder of NNNNNN_color.png + intrinsics.json"
    )
    parser.add_argument(
        "--keypoints-3d", required=True, help="keypoints_3d.json (required for PnP metrics)"
    )
    parser.add_argument(
        "--intrinsics",
        default=None,
        help="override intrinsics.json (else <data-dir>/intrinsics.json)",
    )
    parser.add_argument(
        "--out-json", default=None, help="scorecard json (default <data-dir>/real_eval.json)"
    )
    parser.add_argument("--backbone", default=_MODEL_DEFAULTS.backbone)
    parser.add_argument("--num-keypoints", type=int, default=_MODEL_DEFAULTS.num_keypoints)
    parser.add_argument("--num-classes", type=int, default=_MODEL_DEFAULTS.num_classes)
    parser.add_argument("--det-size", type=int, default=_MODEL_DEFAULTS.det_input_size)
    parser.add_argument(
        "--letterbox",
        action=argparse.BooleanOptionalAction,
        default=_MODEL_DEFAULTS.letterbox,
        help="aspect-preserving letterbox; must match how the checkpoint was trained",
    )
    parser.add_argument("--crop-size", type=int, default=_MODEL_DEFAULTS.crop_size)
    parser.add_argument("--crop-pad-factor", type=float, default=_MODEL_DEFAULTS.crop_pad_factor)
    parser.add_argument("--score-thr", type=float, default=defaults.score_thr)
    parser.add_argument("--kp-score-thr", type=float, default=defaults.kp_score_thr)
    parser.add_argument("--box-pad", type=float, default=defaults.box_pad)
    parser.add_argument(
        "--rot-tta",
        action=argparse.BooleanOptionalAction,
        default=defaults.rot_tta,
        help="evaluate WITH 90-degree rotation TTA (default off: measures the raw model)",
    )
    parser.add_argument(
        "--depth-range",
        type=float,
        nargs=2,
        default=defaults.depth_range,
        metavar=("MIN_M", "MAX_M"),
        help="plausible depth window in meters for the depth-plausibility metric",
    )
    parser.add_argument(
        "--flip-rot-deg",
        type=float,
        default=defaults.flip_rot_deg,
        help="frame-to-frame rotation jump (deg) counted as a flip/jump",
    )
    parser.add_argument(
        "--jump-depth-cm",
        type=float,
        default=defaults.jump_depth_cm,
        help="frame-to-frame depth jump (cm) counted as a jump",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="cap number of frames (0 = all)")
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Score a checkpoint on a recorded capture and write the scorecard JSON.

    Args:
        argv: CLI argument strings excluding the program name, or None to read
            ``sys.argv[1:]``.

    Returns:
        Zero after the scorecard was computed and written.

    Raises:
        SystemExit: Argument parsing requests help or rejects invalid arguments, or the data
            directory holds no ``*_color.png`` frames.
    """
    args = _build_parser().parse_args(argv)
    device = resolve_device(args.device)
    data_dir = Path(args.data_dir)
    camera_matrix, dist_coeffs = load_intrinsics(args.intrinsics or data_dir / "intrinsics.json")
    keypoints_3d = load_keypoints_3d(args.keypoints_3d)
    model_cfg = ModelConfig(
        backbone=args.backbone,
        pretrained=False,
        num_classes=args.num_classes,
        num_keypoints=args.num_keypoints,
        det_input_size=args.det_size,
        crop_size=args.crop_size,
        letterbox=args.letterbox,
        crop_pad_factor=args.crop_pad_factor,
    )
    cfg = DoperConfig(data=DataConfig(root="."), model=model_cfg)
    model = load_model(args.checkpoint, cfg, device)
    frames = list_color_frames(data_dir, args.max_frames)
    if not frames:
        raise SystemExit(f"no *_color.png frames found in {data_dir}")
    options = ScorecardOptions(
        score_thr=args.score_thr,
        kp_score_thr=args.kp_score_thr,
        box_pad=args.box_pad,
        rot_tta=args.rot_tta,
        depth_range=(float(args.depth_range[0]), float(args.depth_range[1])),
        flip_rot_deg=args.flip_rot_deg,
        jump_depth_cm=args.jump_depth_cm,
    )
    log(args.checkpoint)
    log(f"{len(frames)} frames from {data_dir} (rot_tta={options.rot_tta})")
    result = evaluate_recording(
        model, cfg, device, frames, keypoints_3d, camera_matrix, dist_coeffs, options
    )
    scorecard = {"checkpoint": args.checkpoint, "data_dir": args.data_dir, **result["scorecard"]}
    _log_scorecard(scorecard, options)
    out_json = Path(args.out_json) if args.out_json else data_dir / "real_eval.json"
    write_json(out_json, {"scorecard": scorecard, "per_frame": result["per_frame"]})
    log(f"scorecard json      : {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
