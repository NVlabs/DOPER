# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BOP-style pose evaluation of DOPER checkpoints on a BOP ``val``/``test`` split (single object).

For every frame that contains ``--obj-id`` the model is run through ``infer.infer_image``
(detection, keypoints, PnP in millimeters with the frame's ``cam_K``), the best-scoring detection
is matched to the GT instance, and the two official BOP pose errors (MSSD and MSPD, see
``doper.evaluation.bop_metrics``) are computed with the ``models_eval`` mesh. The per-frame BOP CSV
is written as well so the BOP toolkit can re-score if wanted.

Missing or corrupt images and multiple target instances fail evaluation. Images are preflighted
once and checked against their content hashes for every checkpoint; non-detections remain
failures in the unchanged scoring population.

Example::

    python -m doper.evaluation.bop --obj-id 2 --val-dir data/val \
        --models-dir data/models/models_eval --keypoints keypoints_3d.json \
        --checkpoint first=runs/a/best.pth --checkpoint second=runs/b/best.pth \
        --out-json bop_eval.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from doper.config import DataConfig, DoperConfig, ModelConfig
from doper.console import log as console_log
from doper.infer import infer_image, load_model
from doper.model import DoperNet

from .bop_metrics import mspd, mssd, read_ply_vertices, summarize, symmetry_transforms
from .io import parse_named_inputs, write_json

MAX_DETECTIONS = 5
MSPD_REFERENCE_WIDTH_PX = 640.0
DEFAULT_SCORE_THRESHOLD = 0.15
MILLIMETERS_PER_METER = 1000.0


def val_frames(val_dir: Path, obj_id: int) -> list[tuple[Path, int, dict, dict]]:
    """Collect frames containing exactly one target instance.

    Args:
        val_dir: BOP split directory containing scene folders with scene_gt.json
            and scene_camera.json.
        obj_id: BOP object identifier to select from each frame.

    Returns:
        Ordered (scene_dir, frame_id, ground_truth, camera) tuples, sorted by
        scene directory and numeric frame ID. Frames without the object are omitted.

    Raises:
        ValueError: A frame contains multiple instances of the target object.
    """
    out = []
    scene_dirs = sorted(
        p for p in val_dir.iterdir() if p.is_dir() and (p / "scene_gt.json").is_file()
    )
    for scene_dir in scene_dirs:
        gt_all = json.loads((scene_dir / "scene_gt.json").read_text())
        cam_all = json.loads((scene_dir / "scene_camera.json").read_text())
        for frame_key in sorted(gt_all, key=int):
            instances = [g for g in gt_all[frame_key] if g["obj_id"] == obj_id]
            if len(instances) > 1:
                raise ValueError(
                    f"{scene_dir.name}/{int(frame_key):06d}: single-instance evaluation "
                    f"requires one instance of object {obj_id}, found {len(instances)}"
                )
            if instances:
                out.append((scene_dir, int(frame_key), instances[0], cam_all[frame_key]))
    return out


def rgb_path(scene_dir: Path, frame: int) -> Path | None:
    """Find a frame image, preferring PNG over JPEG.

    Args:
        scene_dir: BOP scene directory containing an rgb subdirectory.
        frame: Numeric frame ID used as a six-digit filename stem.

    Returns:
        The .png or .jpg path, or None when neither exists.
    """
    for ext in (".png", ".jpg"):
        path = scene_dir / "rgb" / f"{frame:06d}{ext}"
        if path.exists():
            return path
    return None


@dataclass(frozen=True)
class EvaluationFrame:
    """A preflighted frame with its image content pinned by a digest.

    Attributes:
        scene_dir: BOP scene directory; its basename identifies the scene.
        frame: Numeric frame ID within the scene.
        gt: Target BOP annotation with obj_id, flattened cam_R_m2c rotation,
            and cam_t_m2c translation in millimeters.
        camera: BOP camera entry with flattened (3, 3) cam_K pixel intrinsics.
        image_path: Absolute resolved image path recorded during preflight.
        image_sha256: Hex SHA-256 digest of the encoded image bytes.
    """

    scene_dir: Path
    frame: int
    gt: dict
    camera: dict
    image_path: Path
    image_sha256: str

    def manifest_entry(self, *, include_local_paths: bool = False) -> dict:
        """Describe the image content, pose and camera used for this frame.

        Args:
            include_local_paths: Emit the resolved local image path when True.
                By default, emit an rgb/<frame>.<extension> scene-relative identifier.

        Returns:
            A dictionary containing scene, frame, image_path, image_sha256, gt
            and camera. The annotation dictionaries are shared with this frame.
        """
        relative = f"rgb/{self.frame:06d}{self.image_path.suffix}"
        return {
            "scene": self.scene_dir.name,
            "frame": self.frame,
            "image_path": str(self.image_path) if include_local_paths else relative,
            "image_sha256": self.image_sha256,
            "gt": self.gt,
            "camera": self.camera,
        }


def _read_rgb(path: Path) -> tuple[np.ndarray, str]:
    """Decode a color image and hash the same encoded bytes.

    Args:
        path: Path to the encoded evaluation image.

    Returns:
        A pair of the decoded (H, W, 3) uint8 BGR image and the hexadecimal
        SHA-256 digest of its encoded bytes.

    Raises:
        ValueError: The image cannot be read or decoded.
    """
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read evaluation image {path}: {exc}") from exc
    try:
        image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ValueError(f"cannot decode evaluation image {path}") from exc
    if image is None:
        raise ValueError(f"cannot decode evaluation image {path}")
    return image, hashlib.sha256(content).hexdigest()


def prepare_frames(
    frames: Sequence[tuple[Path, int, dict, dict]],
) -> tuple[EvaluationFrame, ...]:
    """Validate the scoring population and pin image paths and content.

    Images are decoded here, then read again while scoring to avoid retaining
    the entire split in memory. Every checkpoint must see the preflight bytes.

    Args:
        frames: Ordered (scene_dir, frame_id, ground_truth, camera) tuples
            returned by val_frames.

    Returns:
        An ordered tuple of EvaluationFrame records with resolved paths and
        image hashes; decoded images are not retained.

    Raises:
        ValueError: The population is empty, or an image is missing, unreadable
            or corrupt.
    """
    if not frames:
        raise ValueError("no evaluation frames contain the requested object")
    prepared = []
    for scene_dir, frame, gt, cam in frames:
        path = rgb_path(scene_dir, frame)
        if path is None:
            raise ValueError(f"{scene_dir.name}/{frame:06d}: missing evaluation RGB image")
        path = path.resolve()
        _, digest = _read_rgb(path)
        prepared.append(EvaluationFrame(scene_dir, frame, gt, cam, path, digest))
    return tuple(prepared)


def _score_frame(
    model: DoperNet,
    cfg: DoperConfig,
    item: EvaluationFrame,
    *,
    kp_mm: np.ndarray,
    pts_mm: np.ndarray,
    syms: Sequence[np.ndarray],
    score_thr: float,
    device: str,
) -> tuple[dict, list | None]:
    """Run inference on one pinned frame and compute its pose errors.

    Args:
        model: Loaded model in eval mode.
        cfg: Config the model was loaded with.
        item: Preflighted frame; its image must still match ``image_sha256``.
        kp_mm: (K, 3) model-frame keypoints in millimeters.
        pts_mm: (N, 3) evaluation mesh vertices in model-frame millimeters.
        syms: Model-frame (4, 4) symmetry transforms.
        score_thr: Detection confidence threshold.
        device: PyTorch inference device string.

    Returns:
        The per-frame record and the pose pieces ``(score, rot_est, t_est)`` needed for the CSV
        row, or None for the latter when no detection produced a pose.

    Raises:
        ValueError: The image is unreadable, corrupt or changed since preflight.
    """
    name = f"{item.scene_dir.name}/{item.frame:06d}"
    image, digest = _read_rgb(item.image_path)
    if digest != item.image_sha256:
        raise ValueError(f"{name}: evaluation image changed after preflight: {item.image_path}")
    camera_matrix = np.asarray(item.camera["cam_K"], dtype=float).reshape(3, 3)
    rot_gt = np.asarray(item.gt["cam_R_m2c"], dtype=float).reshape(3, 3)
    t_gt = np.asarray(item.gt["cam_t_m2c"], dtype=float).reshape(3)
    with torch.no_grad():
        dets = infer_image(
            model,
            image,
            cfg,
            device,
            score_thr=score_thr,
            max_det=MAX_DETECTIONS,
            keypoints_3d=kp_mm,
            camera_matrix=camera_matrix,
        )
    posed = [d for d in dets if d.rvec is not None and d.tvec is not None]
    record = {
        "scene": item.scene_dir.name,
        "frame": item.frame,
        "score": 0.0,
        "mssd_mm": None,
        "mspd_px640": None,
    }
    if not posed:
        return record, None
    best = max(posed, key=lambda d: d.score)
    rot_est, _ = cv2.Rodrigues(np.asarray(best.rvec, dtype=float))
    t_est = np.asarray(best.tvec, dtype=float).reshape(3)
    width_scale = MSPD_REFERENCE_WIDTH_PX / image.shape[1]
    reproj = None if best.reproj_error_px is None else float(best.reproj_error_px)
    record.update(
        score=float(best.score),
        mssd_mm=mssd(rot_est, t_est, rot_gt, t_gt, pts_mm, syms),
        mspd_px640=mspd(rot_est, t_est, rot_gt, t_gt, camera_matrix, pts_mm, syms) * width_scale,
        reproj_error_px=reproj,
    )
    return record, [float(best.score), rot_est, t_est]


def _csv_row(item: EvaluationFrame, obj_id: int, pose: list) -> list:
    """Format one BOP result CSV row (scene_id, im_id, obj_id, score, R, t, time)."""
    score, rot_est, t_est = pose
    return [
        int(item.scene_dir.name),
        item.frame,
        obj_id,
        f"{score:.4f}",
        " ".join(f"{v:.8f}" for v in rot_est.flatten()),
        " ".join(f"{v:.4f}" for v in t_est),
        -1,
    ]


def evaluate_checkpoint(
    checkpoint: str,
    frames: Sequence[EvaluationFrame],
    *,
    obj_id: int,
    kp_mm: np.ndarray,
    pts_mm: np.ndarray,
    syms: Sequence[np.ndarray],
    diameter_mm: float,
    score_thr: float,
    device: str,
    csv_path: Path | None,
    log: Callable[[str], None] = console_log,
) -> dict:
    """Score a checkpoint, retaining non-detections as recall failures.

    Args:
        checkpoint: Path to the DOPER checkpoint loaded for inference.
        frames: Ordered preflighted scoring population from prepare_frames.
        obj_id: BOP object identifier written to CSV rows.
        kp_mm: (K, 3) model-frame keypoints in millimeters, ordered to match
            the checkpoint keypoint head.
        pts_mm: (N, 3) evaluation mesh vertices in model-frame millimeters.
        syms: Model-frame (4, 4) symmetry transforms with millimeter translations.
        diameter_mm: Object diameter in millimeters for MSSD recall thresholds.
        score_thr: Detection confidence threshold passed to inference.
        device: PyTorch inference device string, for example cuda or cpu.
        csv_path: Destination for BOP pose CSV rows, or None to disable CSV output.
            Parent directories are created when needed.
        log: Callable accepting a single progress-summary string; defaults to the
            package console logger.

    Returns:
        A dictionary with summary statistics and ordered per-frame records.
        Frame errors are mssd_mm and mspd_px640 (pixels at 640-pixel width),
        or None for missed poses. Summary fps includes image reads and scoring
        but excludes model loading and output writing.

    Raises:
        ValueError: An image is unreadable, corrupt or changed since preflight.
    """
    model_cfg = ModelConfig(pretrained=False, num_keypoints=len(kp_mm))
    cfg = DoperConfig(data=DataConfig(root="."), model=model_cfg)
    model = load_model(checkpoint, cfg, device)
    rows, results = [], []
    start = time.time()
    for item in frames:
        record, pose = _score_frame(
            model,
            cfg,
            item,
            kp_mm=kp_mm,
            pts_mm=pts_mm,
            syms=syms,
            score_thr=score_thr,
            device=device,
        )
        if pose is not None:
            rows.append(_csv_row(item, obj_id, pose))
        results.append(record)
    elapsed = time.time() - start
    summary = summarize(results, diameter_mm=diameter_mm)
    summary["fps"] = len(results) / max(elapsed, 1e-9)
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["scene_id", "im_id", "obj_id", "score", "R", "t", "time"])
            writer.writerows(rows)
    log(
        f"{checkpoint}: {summary['frames']} frames, det {summary['det_rate']:.1%}, "
        f"AR_MSSD {summary['ar_mssd']:.3f} AR_MSPD {summary['ar_mspd']:.3f} "
        f"AR {summary['ar_mean']:.3f} (median MSSD {summary['mssd_median_mm']:.1f} mm, "
        f"MSPD {summary['mspd_median_px640']:.1f} px) {summary['fps']:.1f} fps"
    )
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {"summary": summary, "frames": results}


def format_table(results: dict[str, dict]) -> str:
    """Format checkpoint recall, detection rate and median pose errors.

    Args:
        results: Mapping of checkpoint labels to evaluate_checkpoint results,
            each containing a summary dictionary.

    Returns:
        A fixed-width console table with one row per checkpoint. Median MSSD
        is in millimeters and median MSPD is in pixels at 640-pixel image width.
    """
    cols = [
        ("model", 14),
        ("AR_MSSD", 8),
        ("AR_MSPD", 8),
        ("AR", 8),
        ("det", 7),
        ("MSSD_med", 9),
        ("MSPD_med", 9),
    ]
    lines = [" ".join(f"{name:>{w}}" for name, w in cols)]
    for name, r in results.items():
        s = r["summary"]
        lines.append(
            " ".join(
                [
                    f"{name:>14}",
                    f"{s['ar_mssd']:>8.3f}",
                    f"{s['ar_mspd']:>8.3f}",
                    f"{s['ar_mean']:>8.3f}",
                    f"{s['det_rate']:>7.1%}",
                    f"{s['mssd_median_mm']:>9.1f}",
                    f"{s['mspd_median_px640']:>9.1f}",
                ]
            )
        )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the BOP evaluator."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--obj-id", type=int, required=True)
    parser.add_argument(
        "--val-dir", type=Path, required=True, help="BOP split directory with scene folders"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        required=True,
        help="BOP models_eval dir (obj_XXXXXX.ply + models_info.json)",
    )
    parser.add_argument(
        "--keypoints",
        type=Path,
        required=True,
        help="keypoints_3d.json in meters, BOP model frame",
    )
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument(
        "--score-thr",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        help="detection score threshold (BOP: lower catches more)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=None,
        help="write BOP result CSVs here (one per checkpoint)",
    )
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument(
        "--include-local-paths",
        action="store_true",
        help="include local input/image paths in JSON; default uses basenames and "
        "scene-relative IDs",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0, help="evaluate only the first N frames (smoke test)"
    )
    return parser


def _build_payload(
    args: argparse.Namespace,
    checkpoints: list[tuple[str, str]],
    frames: Sequence[EvaluationFrame],
    keypoints_bytes: bytes,
    results: dict[str, dict],
) -> dict:
    """Assemble the JSON result document, hiding local paths unless requested."""
    local = args.include_local_paths
    return {
        "obj_id": args.obj_id,
        "val_dir": str(args.val_dir) if local else args.val_dir.name,
        "score_thr": args.score_thr,
        "checkpoints": {name: path if local else Path(path).name for name, path in checkpoints},
        "evaluation": {
            "schema_version": 1,
            "device": args.device,
            "max_frames": args.max_frames,
            "models_dir": str(args.models_dir) if local else args.models_dir.name,
            "keypoints": str(args.keypoints) if local else args.keypoints.name,
            "keypoints_sha256": hashlib.sha256(keypoints_bytes).hexdigest(),
            "frames": [frame.manifest_entry(include_local_paths=local) for frame in frames],
        },
        "results": results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Validate inputs, score checkpoints and write BOP results.

    Args:
        argv: CLI argument strings excluding the program name, or None to read
            sys.argv[1:].

    Returns:
        Zero after successful evaluation and requested CSV/JSON writes.

    Raises:
        SystemExit: Argument parsing requests help or rejects invalid arguments.
        ValueError: Evaluation inputs fail validation.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.max_frames < 0:
        parser.error("--max-frames must be nonnegative")
    try:
        checkpoints = parse_named_inputs(args.checkpoint, "checkpoint", prefix="model")
    except ValueError as exc:
        parser.error(str(exc))
    info = json.loads((args.models_dir / "models_info.json").read_text())[str(args.obj_id)]
    pts_mm = read_ply_vertices(str(args.models_dir / f"obj_{args.obj_id:06d}.ply"))
    syms = symmetry_transforms(info)
    keypoints_bytes = args.keypoints.read_bytes()
    kp_mm = (
        np.asarray(json.loads(keypoints_bytes)["keypoints_3d"], dtype=np.float64)
        * MILLIMETERS_PER_METER
    )
    selected = val_frames(args.val_dir, args.obj_id)
    if args.max_frames:
        selected = selected[: args.max_frames]
    frames = prepare_frames(selected)
    console_log(
        f"obj {args.obj_id}: {len(frames)} frames, {len(pts_mm)} eval vertices, "
        f"{len(syms)} symmetry transform(s), diameter {info['diameter']:.1f} mm, "
        f"{len(kp_mm)} keypoints"
    )
    results = {}
    for name, checkpoint in checkpoints:
        csv_name = f"{name}_obj{args.obj_id:06d}_{args.val_dir.name}.csv"
        results[name] = evaluate_checkpoint(
            checkpoint,
            frames,
            obj_id=args.obj_id,
            kp_mm=kp_mm,
            pts_mm=pts_mm,
            syms=syms,
            diameter_mm=float(info["diameter"]),
            score_thr=args.score_thr,
            device=args.device,
            csv_path=args.csv_dir / csv_name if args.csv_dir else None,
        )
    print("\n" + format_table(results))
    if args.out_json:
        write_json(
            args.out_json, _build_payload(args, checkpoints, frames, keypoints_bytes, results)
        )
        console_log(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
