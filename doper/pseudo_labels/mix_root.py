# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert a flat COCO export into a training mix-root split by symlinking its images.

The exporters (e.g. ``doper.pseudo_labels.multiview``) write a flat directory::

    <src>/annotations.json
    <src>/images/<file_name>
    <src>/masks/<mask_file>            (optional)

while training expects each mix root to expose a split subdirectory::

    <dst>/train/annotations.json
    <dst>/train/images/<file_name>     (symlinks, no copies)
    <dst>/train/masks/<mask_file>      (symlinks)

so a freshly exported pseudo-label set can be added straight to a training config as another
``mix_data_root``.

Example::

    python -m doper.pseudo_labels.mix_root --src pseudo_labels/capture --dst mix/capture
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from doper.console import log
from doper.evaluation.io import write_json
from doper.pseudo_labels._outdir import replace_output_dir

POSE_FIELDS = frozenset(
    {
        "keypoints",
        "num_keypoints",
        "rvec",
        "tvec",
        "multiview_reproj_px",
        "ransac_inliers",
        "box_dither",
        "box_dither_selected",
        "reprojection_source_modality",
    }
)
MISSING_EXAMPLES = 3


def _build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser of ``main``."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--src", required=True, help="flat export dir (annotations.json + images/)")
    parser.add_argument(
        "--dst", required=True, help="mix-root dir to create (gets a train/ subdir)"
    )
    parser.add_argument("--split", default="train", help="split subdir name (default: train)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing dst split")
    parser.add_argument(
        "--mask-only",
        action="store_true",
        help="strip keypoints and pose metadata while retaining boxes and masks",
    )
    return parser


def strip_pose_fields(coco: dict[str, Any]) -> None:
    """Remove keypoint and pose metadata from every annotation in place and mark the info.

    Args:
        coco: COCO payload; its ``annotations`` lose every ``POSE_FIELDS`` key and its
            ``info`` gets ``mask_only`` and a matching description.
    """
    for annotation in coco["annotations"]:
        for name in POSE_FIELDS:
            annotation.pop(name, None)
    info = dict(coco.get("info", {}))
    info["mask_only"] = True
    info["description"] = (
        "Mask-and-bounding-box-only derivative; keypoint and pose metadata were "
        "intentionally removed."
    )
    coco["info"] = info


def _link_images(coco: dict[str, Any], src_images: str, dst_images: str) -> int:
    """Symlink every referenced image; return the count or exit listing missing images."""
    referenced = {image["file_name"] for image in coco["images"]}
    linked = 0
    missing = []
    for file_name in sorted(referenced):
        source = os.path.abspath(os.path.join(src_images, file_name))
        if not os.path.exists(source):
            missing.append(file_name)
            continue
        os.symlink(source, os.path.join(dst_images, file_name))
        linked += 1
    if missing:
        raise SystemExit(
            f"[mixroot] {len(missing)} referenced images missing, e.g. {missing[:MISSING_EXAMPLES]}"
        )
    return linked


def _link_masks(coco: dict[str, Any], src: str, split_dir: str) -> int:
    """Symlink every referenced mask file at its relative path; return the count."""
    mask_files = sorted({ann["mask_file"] for ann in coco["annotations"] if ann.get("mask_file")})
    for relative in mask_files:
        source = os.path.abspath(os.path.join(src, relative))
        if not os.path.isfile(source):
            raise SystemExit(f"[mixroot] referenced mask missing: {relative}")
        destination = os.path.join(split_dir, relative)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        os.symlink(source, destination)
    return len(mask_files)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point of the mix-root converter.

    Args:
        argv: CLI argument strings excluding the program name, or None to read
            ``sys.argv[1:]``.

    Returns:
        Zero after ``<dst>/<split>/annotations.json`` was written.

    Raises:
        SystemExit: The source lacks ``annotations.json`` or ``images/``, the destination
            split exists without ``--force``, or a referenced image or mask is missing.
    """
    args = _build_parser().parse_args(argv)
    src_annotations = os.path.join(args.src, "annotations.json")
    src_images = os.path.join(args.src, "images")
    if not os.path.isfile(src_annotations):
        raise SystemExit(f"[mixroot] no annotations.json in {args.src}")
    if not os.path.isdir(src_images):
        raise SystemExit(f"[mixroot] no images/ dir in {args.src}")
    split_dir = os.path.join(args.dst, args.split)
    replace_output_dir(split_dir, args.force, [args.src])
    dst_images = os.path.join(split_dir, "images")
    os.makedirs(dst_images)
    with open(src_annotations) as handle:
        coco = json.load(handle)
    if args.mask_only:
        strip_pose_fields(coco)
    linked = _link_images(coco, src_images, dst_images)
    linked_masks = _link_masks(coco, args.src, split_dir)
    write_json(Path(split_dir) / "annotations.json", coco)
    n_color = sum(1 for image in coco["images"] if image.get("modality") == "color")
    n_ir = sum(1 for image in coco["images"] if image.get("modality") == "ir_left")
    log(split_dir)
    log(
        f"linked {linked} images (color={n_color} ir_left={n_ir}), {linked_masks} masks, "
        f"{len(coco['annotations'])} annotations" + (" (mask-only)" if args.mask_only else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
