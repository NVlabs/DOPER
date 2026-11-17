# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate strict per-camera box and mask pseudo-labels for an HDF5 episode split.

Every episode ``<dataset-root>/episode_<id>.hdf5`` stores RGB frames per camera under
``obs/images/<camera>``; ``--split-json`` maps ``train``/``validation``/``test`` to episode ids.
All ``--eval-splits`` are evaluated, but only ``--label-splits`` write training images,
projected-cuboid masks and COCO annotations, so validation and test pseudo-labels stay
physically separate from the training root. A frame is labeled when its best detection scores
above ``--score-thr``, its PnP reprojection error is below ``--max-reproj-px`` and it has more
than ``--min-inliers`` RANSAC inliers. ``--cameras`` names the image datasets to read (default
``wrist_left,wrist_right``); every camera is undistorted pinhole with the required ``--fx``,
``--fy``, ``--cx``, ``--cy`` intrinsics, which have no default because a wrong calibration
silently corrupts every pose, mask and box.

Needs the ``hdf5`` extra (``h5py``).

Example::

    python -m doper.pseudo_labels.hdf5_episodes \
        --dataset-root episodes --split-json episodes/split.json \
        --checkpoint runs/best.pth --keypoints-3d data/keypoints_3d.json \
        --out-root pseudo_labels/episodes --fx 600 --fy 600 --cx 320 --cy 240
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from doper.config import DataConfig, DoperConfig, ModelConfig
from doper.console import log
from doper.evaluation.io import write_json
from doper.geometry_vis import compute_3d_bbox, project_points
from doper.infer import Detection, infer_image, load_model
from doper.multiview import single_view_inliers
from doper.performance import resolve_device
from doper.pnp import PNP_METHODS
from doper.pseudo_labels._outdir import replace_output_dir
from doper.pseudo_labels.masks import bbox_from_points, projected_cuboid_mask

SPLIT_NAMES = frozenset({"train", "validation", "test"})
DEFAULT_CAMERAS = "wrist_left,wrist_right"
DETECTION_SCORE_FLOOR = 0.1
MAX_DETECTIONS = 5
DISTORTION_COEFFICIENT_COUNT = 5
AREA_DECIMALS = 2
H5PY_HINT = "h5py is required for doper.pseudo_labels.hdf5_episodes; install doper[hdf5]"


def _build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser of ``main``."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--keypoints-3d", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--label-splits",
        default="train",
        help="comma-separated splits allowed to write training labels",
    )
    parser.add_argument(
        "--eval-splits",
        default="train,validation,test",
        help="comma-separated splits to evaluate",
    )
    parser.add_argument("--score-thr", type=float, default=0.5)
    parser.add_argument("--kp-score-thr", type=float, default=0.3)
    parser.add_argument("--max-reproj-px", type=float, default=3.0)
    parser.add_argument("--min-inliers", type=int, default=50)
    parser.add_argument("--ransac-px", type=float, default=8.0)
    parser.add_argument(
        "--pnp-method",
        choices=PNP_METHODS,
        default="usac",
        help="PnP RANSAC method: usac (default) or epnp (the paper's labels)",
    )
    for name in ("fx", "fy", "cx", "cy"):
        parser.add_argument(f"--{name}", type=float, required=True, help="pinhole intrinsics (px)")
    parser.add_argument(
        "--cameras",
        default=DEFAULT_CAMERAS,
        help="comma-separated camera names read from obs/images/<camera> in every episode",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true", help="replace an existing out-root")
    return parser


def _import_h5py() -> Any:  # noqa: ANN401 - the h5py module
    """Import h5py, or raise an ``ImportError`` naming the ``hdf5`` extra."""
    try:
        import h5py
    except ImportError as error:
        raise ImportError(H5PY_HINT) from error
    return h5py


def _split_names(value: str) -> list[str]:
    return [name.strip() for name in value.split(",") if name.strip()]


class _EpisodeLabeler:
    """Model, geometry and accumulated labels of one export run."""

    def __init__(self, args: argparse.Namespace, keypoints_3d: np.ndarray) -> None:
        self.args = args
        self.device = resolve_device(args.device)
        self.keypoints_3d = keypoints_3d
        self.box3d = compute_3d_bbox(keypoints_3d)
        self.camera_matrix = np.asarray(
            [[args.fx, 0.0, args.cx], [0.0, args.fy, args.cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        self.distortion = np.zeros((DISTORTION_COEFFICIENT_COUNT, 1), dtype=np.float64)
        model_cfg = ModelConfig(pretrained=False, num_classes=1, num_keypoints=len(keypoints_3d))
        self.cfg = DoperConfig(data=DataConfig(root="."), model=model_cfg)
        self.model = load_model(str(args.checkpoint), self.cfg, self.device)
        self.image_dir = args.out_root / "train" / "images"
        self.mask_dir = args.out_root / "train" / "masks"
        self.images: list[dict[str, Any]] = []
        self.annotations: list[dict[str, Any]] = []

    def best_detection(self, image: np.ndarray) -> Detection | None:
        """Highest-scoring posed detection of one BGR frame, or None."""
        detections = infer_image(
            self.model, image, self.cfg, self.device, score_thr=DETECTION_SCORE_FLOOR,
            max_det=MAX_DETECTIONS, keypoints_3d=self.keypoints_3d,
            camera_matrix=self.camera_matrix, dist_coeffs=self.distortion,
        )  # fmt: skip
        return max(detections, key=lambda item: item.score) if detections else None

    def passes_pose_gates(self, detection: Detection) -> bool:
        """Whether the detection's score and PnP reprojection pass the gates."""
        args = self.args
        return not (
            detection.score <= args.score_thr
            or detection.rvec is None
            or detection.reproj_error_px is None
            or detection.reproj_error_px >= args.max_reproj_px
        )

    def inliers(self, detection: Detection) -> int:
        """Single-view RANSAC inliers among the confident keypoints of ``detection``."""
        return int(
            single_view_inliers(
                self.keypoints_3d, detection.keypoints.astype(np.float64),
                detection.keypoint_scores >= self.args.kp_score_thr, self.camera_matrix,
                self.distortion, self.args.ransac_px, self.args.pnp_method,
            ).sum()
        )  # fmt: skip

    def write_label(self, image: np.ndarray, detection: Detection, inliers: int,
                    origin: dict[str, Any]) -> None:  # fmt: skip
        """Write the image, projected-cuboid mask and COCO records of an accepted frame.

        Args:
            image: BGR frame.
            detection: Accepted detection with a pose.
            inliers: RANSAC inlier count of the detection.
            origin: ``episode``, ``camera_label`` and ``frame_id`` of the frame.
        """
        rvec, tvec, reprojection = detection.rvec, detection.tvec, detection.reproj_error_px
        if rvec is None or tvec is None or reprojection is None:
            raise ValueError("only posed detections can be labeled")
        height, width = image.shape[:2]
        projected = project_points(self.box3d, rvec, tvec, self.camera_matrix, self.distortion)
        mask = projected_cuboid_mask(projected, width, height)
        bbox = bbox_from_points(projected, width, height, 0.0)
        stem = f"episode_{origin['episode']}_{origin['camera_label']}_{origin['frame_id']:06d}"
        image_name = f"{stem}.png"
        mask_name = f"{stem}_mask.png"
        cv2.imwrite(str(self.image_dir / image_name), image)
        cv2.imwrite(str(self.mask_dir / mask_name), mask)
        identifier = len(self.images) + 1
        self.images.append(
            {"id": identifier, "file_name": image_name, "width": int(width),
             "height": int(height), "view": "real_pseudo", "modality": "color", **origin}
        )  # fmt: skip
        self.annotations.append(
            {"id": identifier, "image_id": identifier, "category_id": 1,
             "bbox": [float(value) for value in bbox],
             "area": round(float(bbox[2] * bbox[3]), AREA_DECIMALS), "iscrowd": 0,
             "mask_file": f"masks/{mask_name}", "pseudo_label": True,
             "detector_score": float(detection.score),
             "reproj_px": float(reprojection), "ransac_inliers": inliers, **origin}
        )  # fmt: skip

    def label_camera(self, frames: Any, origin: dict[str, Any], write: bool) -> dict[str, Any]:  # noqa: ANN401
        """Evaluate every frame of one camera; label accepted frames when ``write``.

        Returns:
            ``frames``, ``score_gt`` (best detections above the score gate), ``accepted`` and
            ``accepted_frame_ids`` of the camera.
        """
        accepted_ids: list[int] = []
        scores_over_gate = 0
        for frame_id, frame_rgb in enumerate(frames):
            image = cv2.cvtColor(np.asarray(frame_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            detection = self.best_detection(image)
            if detection is None:
                continue
            scores_over_gate += int(detection.score > self.args.score_thr)
            if not self.passes_pose_gates(detection):
                continue
            inliers = self.inliers(detection)
            if inliers <= self.args.min_inliers:
                continue
            accepted_ids.append(frame_id)
            if write:
                self.write_label(image, detection, inliers, {**origin, "frame_id": frame_id})
        return {
            "frames": int(len(frames)),
            "score_gt": scores_over_gate,
            "accepted": len(accepted_ids),
            "accepted_frame_ids": accepted_ids,
        }


def _prepare_out_root(args: argparse.Namespace) -> None:
    """Create ``train/images`` and ``train/masks``; replace ``--out-root`` only with ``--force``."""
    inputs = [args.dataset_root, args.split_json, args.checkpoint, args.keypoints_3d]
    replace_output_dir(args.out_root, args.force, inputs)
    out_root = args.out_root
    (out_root / "train" / "images").mkdir(parents=True)
    (out_root / "train" / "masks").mkdir()


def _category(payload: dict[str, Any], keypoint_count: int) -> dict[str, Any]:
    names = payload.get("keypoint_names", [f"kp_{index:03d}" for index in range(keypoint_count)])
    return {"id": 1, "name": payload.get("object", "object"), "keypoints": names, "skeleton": []}


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point of the HDF5 episode pseudo-labeler.

    Args:
        argv: CLI argument strings excluding the program name, or None to read
            ``sys.argv[1:]``.

    Returns:
        Zero after ``metrics.json`` and ``train/annotations.json`` were written.

    Raises:
        ImportError: h5py is not installed (install the ``hdf5`` extra).
        SystemExit: Unknown split names, label splits outside the evaluated ones, or an
            existing ``--out-root`` without ``--force``.
    """
    args = _build_parser().parse_args(argv)
    h5py = _import_h5py()
    label_splits = set(_split_names(args.label_splits))
    eval_splits = _split_names(args.eval_splits)
    cameras = _split_names(args.cameras)
    split = json.loads(args.split_json.read_text())
    unknown = (label_splits | set(eval_splits)) - SPLIT_NAMES
    if unknown:
        raise SystemExit(f"unknown split names: {sorted(unknown)}")
    if not label_splits.issubset(set(eval_splits)):
        raise SystemExit("--label-splits must be a subset of --eval-splits")
    _prepare_out_root(args)
    payload = json.loads(args.keypoints_3d.read_text())
    keypoints_3d = np.asarray(payload["keypoints_3d"], dtype=np.float64)
    labeler = _EpisodeLabeler(args, keypoints_3d)
    metrics: dict[str, Any] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "split_json": str(args.split_json.resolve()),
        "gates": {
            "score_gt": args.score_thr,
            "reprojection_lt_px": args.max_reproj_px,
            "ransac_inliers_gt": args.min_inliers,
            "ransac_px": args.ransac_px,
            "pnp_method": args.pnp_method,
        },
        "intrinsics": {
            "fx": args.fx,
            "fy": args.fy,
            "cx": args.cx,
            "cy": args.cy,
            "distortion": [0.0] * DISTORTION_COEFFICIENT_COUNT,
        },  # fmt: skip
        "splits": {},
    }
    for split_name in eval_splits:
        episodes: dict[str, Any] = {}
        metrics["splits"][split_name] = {"episodes": episodes}
        for episode_id in split[split_name]:
            episode_metrics: dict[str, Any] = {}
            episodes[str(episode_id)] = episode_metrics
            with h5py.File(args.dataset_root / f"episode_{episode_id}.hdf5", "r") as hdf5:
                for camera in cameras:
                    origin = {"episode": int(episode_id), "camera_label": camera}
                    stats = labeler.label_camera(
                        hdf5[f"obs/images/{camera}"], origin, split_name in label_splits
                    )
                    episode_metrics[camera] = stats
                    log(f"{split_name} episode={episode_id} camera={camera} "
                        f"accepted={stats['accepted']}/{stats['frames']}")  # fmt: skip
            write_json(args.out_root / "metrics.json", metrics)
    coco = {
        "info": {
            "description": (
                "Strict per-camera projected box/mask pseudo labels. No keypoint supervision."
            ),
            "mask_only": True,
            "checkpoint": str(args.checkpoint.resolve()),
            "labeled_splits": sorted(label_splits),
            "unique_samples": len(labeler.images),
        },
        "images": labeler.images,
        "annotations": labeler.annotations,
        "categories": [_category(payload, len(keypoints_3d))],
    }
    write_json(args.out_root / "train" / "annotations.json", coco)
    log(f"complete: labels={len(labeler.images)} out={args.out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
