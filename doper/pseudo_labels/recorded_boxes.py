# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export reviewed recorded detections as color and colocated left-IR box pseudo-labels.

The detection boxes are in the coordinates of an (optionally center-cropped) color recording.
Color frames are symlinked into a ``train/`` mix root; the matching full-resolution left-IR
frames are cropped with the same window (``--crop-json``) and receive the exact same boxes.

``--detections-json`` holds a list of records ``{"idx", "frame", "detected", "bbox_xyxy",
"score"}``, where ``frame`` is the color file name in ``--cropped-color-dir``.

Example::

    python -m doper.pseudo_labels.recorded_boxes \
        --detections-json review/detections.json --cropped-color-dir capture/cropped \
        --source-ir-dir capture/cam_a --crop-json capture/cropped/crop.json \
        --out-root pseudo_labels/boxes
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2

from doper.console import log
from doper.evaluation.io import write_json
from doper.pseudo_labels._outdir import replace_output_dir

MIN_BOX_SIDE_PX = 1.0


def _build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser of ``main``."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--detections-json", required=True)
    parser.add_argument("--cropped-color-dir", required=True)
    parser.add_argument("--source-ir-dir", required=True)
    parser.add_argument(
        "--crop-json",
        default=None,
        help="crop metadata; omit when color and IR frames are already full-size",
    )
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--category-name", default="object")
    parser.add_argument("--camera-label", default="A")
    parser.add_argument(
        "--frame-ids", default=None, help="optional comma-separated frame IDs to export"
    )
    parser.add_argument("--force", action="store_true", help="replace an existing out-root")
    return parser


def _crop_window(
    args: argparse.Namespace, records: list[dict[str, Any]]
) -> tuple[tuple[int, int, int, int], tuple[int, int], dict[str, Any]]:
    """Return the crop ``(x0, y0, x1, y1)``, the output ``(width, height)`` and crop metadata.

    Without ``--crop-json`` the window is the full size of the first detected color frame.
    """
    if args.crop_json:
        with open(args.crop_json) as handle:
            crop_info = json.load(handle)
        x0, y0, x1, y1 = [int(v) for v in crop_info["crop_xyxy"]]
        out_w, out_h = [int(v) for v in crop_info["output_size"]]
        return (x0, y0, x1, y1), (out_w, out_h), crop_info
    first_frame = next(
        (
            os.path.join(args.cropped_color_dir, record["frame"])
            for record in records
            if record.get("detected")
        ),
        None,
    )
    first = cv2.imread(first_frame, cv2.IMREAD_COLOR) if first_frame else None
    if first is None:
        raise SystemExit("could not read a detected color frame to infer image size")
    out_h, out_w = first.shape[:2]
    crop_info = {
        "source": os.path.abspath(args.cropped_color_dir),
        "edge_frac": 0.0,
        "source_size": [out_w, out_h],
        "crop_xyxy": [0, 0, out_w, out_h],
        "output_size": [out_w, out_h],
    }
    return (0, 0, out_w, out_h), (out_w, out_h), crop_info


def _clipped_bbox(bbox_xyxy: Sequence[float], width: int, height: int) -> list[float]:
    """Clip an ``xyxy`` box to the image (at least one pixel per side) as COCO ``xywh``."""
    bx1, by1, bx2, by2 = [float(v) for v in bbox_xyxy]
    bx1 = min(max(bx1, 0.0), float(width - 1))
    by1 = min(max(by1, 0.0), float(height - 1))
    bx2 = min(max(bx2, bx1 + MIN_BOX_SIDE_PX), float(width))
    by2 = min(max(by2, by1 + MIN_BOX_SIDE_PX), float(height))
    return [bx1, by1, bx2 - bx1, by2 - by1]


def _write_ir_crop(
    args: argparse.Namespace, frame_stem: str, images_dir: str, window: tuple[int, ...],
    size: tuple[int, int],
) -> str:  # fmt: skip
    """Crop the left-IR frame of ``frame_stem`` into ``images_dir``; return its file name."""
    x0, y0, x1, y1 = window
    ir_src = os.path.join(args.source_ir_dir, f"{frame_stem}_ir_left.png")
    ir = cv2.imread(ir_src, cv2.IMREAD_GRAYSCALE)
    if ir is None:
        raise SystemExit(f"missing left-IR frame: {ir_src}")
    ir_crop = ir[y0:y1, x0:x1]
    if ir_crop.shape[:2] != (size[1], size[0]):
        raise SystemExit(f"unexpected IR crop shape for {ir_src}: {ir_crop.shape[:2]}")
    ir_name = f"{frame_stem}_ir_left.png"
    if not cv2.imwrite(os.path.join(images_dir, ir_name), ir_crop):
        raise SystemExit(f"failed to write cropped IR frame {ir_name}")
    return ir_name


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point of the recorded-box pseudo-label exporter.

    Args:
        argv: CLI argument strings excluding the program name, or None to read
            ``sys.argv[1:]``.

    Returns:
        Zero after ``<out-root>/train/annotations.json`` was written.

    Raises:
        SystemExit: An existing ``--out-root`` without ``--force``, a missing or unreadable
            frame, a wrongly sized IR crop, or a requested frame without an exportable box.
    """
    args = _build_parser().parse_args(argv)
    out_root = os.path.abspath(os.path.expanduser(args.out_root))
    split_dir = os.path.join(out_root, "train")
    images_dir = os.path.join(split_dir, "images")
    inputs = [args.detections_json, args.cropped_color_dir, args.source_ir_dir, args.crop_json]
    replace_output_dir(out_root, args.force, inputs)
    os.makedirs(images_dir)
    with open(args.detections_json) as handle:
        records = json.load(handle)
    requested = (
        {int(value) for value in args.frame_ids.split(",") if value.strip()}
        if args.frame_ids
        else None
    )
    window, (out_w, out_h), crop_info = _crop_window(args, records)
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    accepted = 0
    for record in records:
        if not record.get("detected") or record.get("bbox_xyxy") is None:
            continue
        if requested is not None and int(record["idx"]) not in requested:
            continue
        frame_name = record["frame"]
        frame_stem = os.path.splitext(frame_name)[0].replace("_color", "")
        bbox = _clipped_bbox(record["bbox_xyxy"], out_w, out_h)
        color_src = os.path.abspath(os.path.join(args.cropped_color_dir, frame_name))
        if not os.path.isfile(color_src):
            raise SystemExit(f"missing cropped color frame: {color_src}")
        color_name = f"{frame_stem}_color.png"
        os.symlink(color_src, os.path.join(images_dir, color_name))
        ir_name = _write_ir_crop(args, frame_stem, images_dir, window, (out_w, out_h))
        for file_name, modality in ((color_name, "color"), (ir_name, "ir_left")):
            common = {
                "frame_id": int(record["idx"]),
                "camera_label": args.camera_label,
                "modality": modality,
            }
            identifier = len(images) + 1
            images.append(
                {"id": identifier, "file_name": file_name, "width": out_w, "height": out_h,
                 "view": "real_pseudo", **common}
            )  # fmt: skip
            annotations.append(
                {"id": identifier, "image_id": identifier, "category_id": 1, "bbox": bbox,
                 "area": bbox[2] * bbox[3], "iscrowd": 0, "pseudo_label": True,
                 "detector_score": float(record["score"]), **common}
            )  # fmt: skip
        accepted += 1
    if requested is not None:
        missing = sorted(requested - {int(image["frame_id"]) for image in images})
        if missing:
            raise SystemExit(f"requested frames lacked exportable detections: {missing}")
    coco = {
        "info": {
            "description": (
                "Reviewed detector boxes on recorded color frames, transferred unchanged to "
                "colocated left IR."
            ),
            "source_detections": os.path.abspath(args.detections_json),
            "crop": crop_info,
        },
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": args.category_name}],
    }
    write_json(Path(split_dir) / "annotations.json", coco)
    log(f"accepted detections: {accepted}")
    log(f"images/annotations: {len(images)} ({accepted} color + {accepted} ir_left)")
    log(f"wrote {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
