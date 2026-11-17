# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a DOPER checkpoint over a recorded single-camera capture and visualize it.

Given a capture folder of ``NNNNNN_color.png`` frames plus the RealSense ``intrinsics.json``
written by the recorder, this loads the model once, runs the full detect -> top-down crop ->
keypoint path on every frame, solves PnP against the object's 3D keypoints, and writes
per-frame overlay PNGs, an MP4 and a per-frame JSON so you can scrub through the result.
Single camera only.

Example::

    python -m doper.infer_recorded \
        --checkpoint runs/best.pth --data-dir recordings/cam_0 \
        --keypoints-3d data/keypoints_3d.json
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from doper.config import DataConfig, DoperConfig, ModelConfig
from doper.console import log
from doper.evaluation.io import write_json
from doper.geometry_vis import compute_3d_bbox, draw_axes, draw_box3d, project_points
from doper.infer import Detection, infer_image, infer_image_tta, infer_keypoints_in_box, load_model
from doper.keypoints import load_keypoints_3d
from doper.performance import resolve_device
from doper.recorded import frame_index, list_color_frames, load_intrinsics

MIN_DETECTIONS = 5
CENTIMETERS_PER_METER = 100.0
STATUS_ORIGIN = (8, 22)
STATUS_SCALE = 0.55
STATUS_COLOR = (0, 255, 0)
MASK_COLOR = (0, 180, 0)
NO_POSE = {"pnp_ok": False, "reproj_px": None, "depth_cm": None}
_MODEL_DEFAULTS = ModelConfig()
# BGR box/keypoint colors cycled across multiple drawn detections per frame.
_DET_COLORS = [
    (0, 210, 255),  # amber (best)
    (255, 120, 0),  # blue
    (0, 255, 120),  # green
    (255, 0, 255),  # magenta
    (0, 255, 255),  # yellow
]


def _build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser of ``main``."""
    parser = argparse.ArgumentParser(
        description="Run DOPER over a recorded single-camera capture",
        epilog="Architecture options are fallbacks for checkpoints without cfg metadata.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--data-dir", required=True, help="folder of NNNNNN_color.png + intrinsics.json"
    )
    parser.add_argument(
        "--keypoints-3d", default=None, help="keypoints_3d.json (enables PnP/pose overlay)"
    )
    parser.add_argument(
        "--intrinsics",
        default=None,
        help="override intrinsics.json (else <data-dir>/intrinsics.json)",
    )
    parser.add_argument("--out-dir", default=None, help="output dir (default <data-dir>/doper_out)")
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
    parser.add_argument(
        "--crop-pad-factor",
        type=float,
        default=_MODEL_DEFAULTS.crop_pad_factor,
        help="crop_pad_factor for the keypoint crop; ~1.0 can restore the training framing "
        "when the detector emits looser boxes than the training ground truth",
    )
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--kp-score-thr", type=float, default=0.3)
    parser.add_argument(
        "--draw-topk",
        type=int,
        default=1,
        help="draw up to this many detections per frame (default 1 = best only)",
    )
    parser.add_argument(
        "--box-pad",
        type=float,
        default=0.0,
        help="expand the detected box by this fraction before the keypoint crop "
        "(e.g. 0.2 = +20%% w/h); 0 = off",
    )
    parser.add_argument(
        "--fixed-bbox",
        type=float,
        nargs=4,
        default=None,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="fixed image-space box used for every frame; bypasses detection",
    )
    parser.add_argument(
        "--show-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="overlay the predicted foreground mask (green tint) on each frame",
    )
    parser.add_argument("--mask-thr", type=float, default=0.5, help="foreground prob threshold")
    parser.add_argument("--mask-alpha", type=float, default=0.45, help="mask overlay strength")
    parser.add_argument(
        "--rot-tta",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="90-degree rotation test-time ensembling (~4x slower); helps discrete pose flips",
    )
    parser.add_argument(
        "--rot-tta-depth-range",
        type=float,
        nargs=2,
        default=(0.03, 1.0),
        metavar=("MIN_M", "MAX_M"),
        help="plausible depth window (meters) used to reject flipped PnP solutions",
    )
    parser.add_argument("--axis-length", type=float, default=0.03)
    parser.add_argument("--max-frames", type=int, default=0, help="cap number of frames (0 = all)")
    parser.add_argument("--fps", type=float, default=10.0, help="output video frame rate")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--dump-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write per_frame.json with per-frame detected/pnp_ok/reproj/depth",
    )
    parser.add_argument("--device", default="cuda")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject inconsistent ``--fixed-bbox`` options via ``parser.error``."""
    if args.fixed_bbox is None:
        return
    x1, y1, x2, y2 = args.fixed_bbox
    if x2 <= x1 or y2 <= y1:
        parser.error("--fixed-bbox must satisfy X2 > X1 and Y2 > Y1")
    if args.rot_tta:
        parser.error("--rot-tta is not supported with --fixed-bbox")


def _build_config(args: argparse.Namespace) -> DoperConfig:
    """Build the fallback config from the architecture options."""
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
    return DoperConfig(data=DataConfig(root="."), model=model_cfg)


def _detect(
    model: Any,  # noqa: ANN401 - DoperNet or a test double
    image: np.ndarray,
    cfg: DoperConfig,
    device: str,
    args: argparse.Namespace,
    geometry: dict[str, Any],
) -> tuple[list[Detection], np.ndarray | None]:
    """Run detection (or the fixed box) on one frame; return detections and optional mask."""
    if args.fixed_bbox is not None:
        box = np.asarray(args.fixed_bbox, dtype=np.float32)
        det = infer_keypoints_in_box(
            model, image, cfg, device, box, box_pad=args.box_pad, **geometry
        )
        return [det], None
    common: dict[str, Any] = {
        "score_thr": args.score_thr,
        "max_det": max(MIN_DETECTIONS, args.draw_topk),
        "box_pad": args.box_pad,
        "return_mask": args.show_mask,
        **geometry,
    }
    if args.rot_tta:
        common["depth_range"] = tuple(args.rot_tta_depth_range)
        result = infer_image_tta(model, image, cfg, device, **common)
    else:
        result = infer_image(model, image, cfg, device, **common)
    return result if args.show_mask else (result, None)


def _tint_mask(vis: np.ndarray, mask: np.ndarray | None, args: argparse.Namespace) -> np.ndarray:
    """Blend the foreground mask over ``vis`` (returned unchanged without a mask)."""
    if mask is None:
        return vis
    foreground = mask >= args.mask_thr
    if not foreground.any():
        return vis
    overlay = vis.copy()
    overlay[foreground] = MASK_COLOR
    return cv2.addWeighted(overlay, args.mask_alpha, vis, 1.0 - args.mask_alpha, 0)


def _reproj_px(det: Detection) -> float:
    """Return the reprojection error of a posed detection (NaN if the solver recorded none)."""
    return float("nan") if det.reproj_error_px is None else float(det.reproj_error_px)


def _pose_fields(det: Detection) -> dict[str, Any]:
    """Return the ``pnp_ok``/``reproj_px``/``depth_cm`` fields of one detection."""
    if det.rvec is None or det.tvec is None:
        return dict(NO_POSE)
    return {
        "pnp_ok": True,
        "reproj_px": _reproj_px(det),
        "depth_cm": float(det.tvec.reshape(3)[2] * CENTIMETERS_PER_METER),
    }


def _draw_detection(
    vis: np.ndarray,
    det: Detection,
    color: tuple[int, int, int],
    camera: tuple[np.ndarray, np.ndarray],
    box3d: np.ndarray | None,
    args: argparse.Namespace,
) -> None:
    """Draw one detection's box, confident keypoints and pose geometry onto ``vis``."""
    camera_matrix, dist = camera
    x1, y1, x2, y2 = det.bbox_xyxy.astype(int)
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    for (kx, ky), score in zip(det.keypoints, det.keypoint_scores, strict=True):
        if score >= args.kp_score_thr and np.isfinite([kx, ky]).all():
            cv2.circle(vis, (int(kx), int(ky)), 2, color, -1, cv2.LINE_AA)
    if det.rvec is None or det.tvec is None:
        return
    if box3d is not None:
        draw_box3d(vis, project_points(box3d, det.rvec, det.tvec, camera_matrix, dist))
    draw_axes(vis, det.rvec, det.tvec, camera_matrix, dist, args.axis_length)


def _frame_status(draw_list: Sequence[Detection], best: Detection) -> str:
    """Return the overlay status text, tracking the best detection."""
    status = (f"{len(draw_list)} det | " if len(draw_list) > 1 else "") + f"score={best.score:.2f}"
    if best.rvec is not None and best.tvec is not None:
        z = float(best.tvec.reshape(3)[2])
        status += f" reproj={_reproj_px(best):.1f}px z={z * 100:.1f}cm"
    return status


def _frame_record(path: Path, best: Detection | None, args: argparse.Namespace) -> dict[str, Any]:
    """Return the per-frame JSON record for the best detection (or its absence)."""
    record: dict[str, Any] = {
        "idx": frame_index(path),
        "frame": path.name,
        "fixed_bbox": args.fixed_bbox is not None,
        "detected": best is not None,
        "score": float(best.score) if best is not None else None,
        "bbox_xyxy": [float(v) for v in best.bbox_xyxy] if best is not None else None,
        "detections": [],
    }
    record.update(_pose_fields(best) if best is not None else NO_POSE)
    return record


def _summarize(per_frame: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-frame records into detection and pose statistics."""
    posed = [r for r in per_frame if r["pnp_ok"]]
    reproj = [r["reproj_px"] for r in posed]
    depth = [r["depth_cm"] for r in posed]
    return {
        "frames": len(per_frame),
        "detections": sum(1 for r in per_frame if r["detected"]),
        "poses": len(posed),
        "reproj_px_mean": float(np.mean(reproj)) if reproj else None,
        "reproj_px_median": float(np.median(reproj)) if reproj else None,
        "depth_cm_mean": float(np.mean(depth)) if depth else None,
        "depth_cm_min": float(np.min(depth)) if depth else None,
        "depth_cm_max": float(np.max(depth)) if depth else None,
    }


def _log_summary(summary: dict[str, Any], overlay_dir: Path) -> None:
    """Print the human-readable run summary."""
    n = max(1, summary["frames"])
    log("=== summary ===")
    log(f"frames           : {summary['frames']}")
    log(f"detections       : {summary['detections']} ({100.0 * summary['detections'] / n:.1f}%)")
    log(f"poses (PnP ok)   : {summary['poses']} ({100.0 * summary['poses'] / n:.1f}%)")
    if summary["poses"]:
        log(
            f"reproj err px    : mean={summary['reproj_px_mean']:.2f} "
            f"median={summary['reproj_px_median']:.2f}"
        )
        log(
            f"depth cm         : mean={summary['depth_cm_mean']:.1f} "
            f"min={summary['depth_cm_min']:.1f} max={summary['depth_cm_max']:.1f}"
        )
    log(f"overlays         : {overlay_dir}")


def _process_frame(
    path: Path,
    image: np.ndarray,
    model: Any,  # noqa: ANN401 - DoperNet or a test double
    cfg: DoperConfig,
    device: str,
    args: argparse.Namespace,
    scene: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, str]:
    """Infer one frame; return its JSON record, annotated overlay and status text."""
    dets, mask = _detect(model, image, cfg, device, args, scene["geometry"])
    vis = _tint_mask(image.copy(), mask, args)
    ranked = sorted(dets, key=lambda d: d.score, reverse=True)
    draw_list = ranked[: max(1, args.draw_topk)]
    best = ranked[0] if ranked else None
    record = _frame_record(path, best, args)
    status = "no det"
    if best is not None:
        for i, det in enumerate(draw_list):
            color = _DET_COLORS[i % len(_DET_COLORS)]
            _draw_detection(vis, det, color, scene["camera"], scene["box3d"], args)
            record["detections"].append(
                {
                    "score": float(det.score),
                    "bbox_xyxy": [float(v) for v in det.bbox_xyxy],
                    **_pose_fields(det),
                }
            )
        status = _frame_status(draw_list, best)
    text = f"{path.name} | {status}"
    cv2.putText(
        vis, text, STATUS_ORIGIN, cv2.FONT_HERSHEY_SIMPLEX, STATUS_SCALE, STATUS_COLOR, 2,
        cv2.LINE_AA,
    )  # fmt: skip
    return record, vis, status


def _load_scene(args: argparse.Namespace, data_dir: Path) -> dict[str, Any]:
    """Load intrinsics and optional 3D keypoints into the per-run geometry bundle."""
    intrinsics_path = Path(args.intrinsics) if args.intrinsics else data_dir / "intrinsics.json"
    camera_matrix, dist = load_intrinsics(intrinsics_path)
    log(
        f"intrinsics {intrinsics_path}: fx={camera_matrix[0, 0]:.1f} fy={camera_matrix[1, 1]:.1f} "
        f"cx={camera_matrix[0, 2]:.1f} cy={camera_matrix[1, 2]:.1f}"
    )
    keypoints_3d = load_keypoints_3d(args.keypoints_3d) if args.keypoints_3d else None
    return {
        "camera": (camera_matrix, dist),
        "box3d": compute_3d_bbox(keypoints_3d) if keypoints_3d is not None else None,
        "geometry": {
            "keypoints_3d": keypoints_3d,
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist,
        },
    }


def _check_keypoint_count(keypoints_3d: np.ndarray | None, num_keypoints: int) -> None:
    """Fail clearly when the 3D keypoints do not match the loaded model's keypoint count."""
    if keypoints_3d is not None and keypoints_3d.shape[0] != num_keypoints:
        raise SystemExit(
            f"--keypoints-3d has {keypoints_3d.shape[0]} keypoints but the checkpoint predicts "
            f"{num_keypoints}"
        )


def _open_writer(args: argparse.Namespace, first_frame: Path, out_dir: Path) -> Any:  # noqa: ANN401
    """Return an MP4 writer sized to ``first_frame``, or None with ``--no-video``."""
    if args.no_video:
        return None
    first = cv2.imread(str(first_frame), cv2.IMREAD_COLOR)
    if first is None:
        raise SystemExit(f"cannot read first frame {first_frame}")
    height, width = first.shape[:2]
    path = out_dir / "doper_overlay.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), args.fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"cannot open an mp4v video writer for {path}; pass --no-video")
    return writer


def run_recorded(args: argparse.Namespace) -> dict[str, Any]:
    """Run the model over a recorded capture and write overlays, video and per-frame JSON.

    Args:
        args: Parsed options of ``main`` (see ``--help``).

    Returns:
        ``{"per_frame": [...], "summary": {...}}`` with one JSON record per readable frame and
        aggregate detection/pose statistics. Overlay PNGs (``<out-dir>/overlays``), the MP4
        (unless ``--no-video``) and ``per_frame.json`` (unless ``--no-dump-json``) are written
        as side effects.

    Raises:
        SystemExit: The data directory holds no ``*_color.png`` frames.
    """
    device = resolve_device(args.device)
    data_dir = Path(args.data_dir)
    scene = _load_scene(args, data_dir)
    cfg = _build_config(args)
    model = load_model(args.checkpoint, cfg, device)
    log(f"loaded {args.checkpoint} on {device}")
    _check_keypoint_count(scene["geometry"]["keypoints_3d"], cfg.model.num_keypoints)
    frames = list_color_frames(data_dir, args.max_frames)
    if not frames:
        raise SystemExit(f"no *_color.png frames found in {data_dir}")
    log(f"{len(frames)} frames")
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "doper_out"
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    writer = _open_writer(args, frames[0], out_dir)
    per_frame: list[dict[str, Any]] = []
    unreadable = 0
    for position, path in enumerate(frames):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            unreadable += 1
            continue
        record, vis, status = _process_frame(path, image, model, cfg, device, args, scene)
        per_frame.append(record)
        cv2.imwrite(str(overlay_dir / f"{frame_index(path):06d}.png"), vis)
        if writer is not None:
            writer.write(vis)
        if position % 25 == 0 or position == len(frames) - 1:
            log(f"{position + 1}/{len(frames)} {path.name} | {status}")
    if writer is not None:
        writer.release()
    if unreadable:
        log(f"WARNING: skipped {unreadable} unreadable frame(s); rates cover readable frames only")
    summary = _summarize(per_frame)
    _log_summary(summary, overlay_dir)
    if args.dump_json:
        write_json(out_dir / "per_frame.json", per_frame)
        log(f"per-frame json   : {out_dir / 'per_frame.json'}")
    return {"per_frame": per_frame, "summary": summary}


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point of the recorded-sequence inference tool.

    Args:
        argv: CLI argument strings excluding the program name, or None to read
            ``sys.argv[1:]``.

    Returns:
        Zero after the run completed.

    Raises:
        SystemExit: Invalid arguments, or the data directory holds no frames.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    run_recorded(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
