# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure record and declared-artifact validation shared by render backends.

Expected coverage comes from the caller's requested range and configuration,
never from metadata reported by a worker. No directory-wide image scan is used.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, TypeGuard


def is_integer(value: Any) -> TypeGuard[int]:  # noqa: ANN401 - CLI or JSON value
    """Whether ``value`` is an ``int`` other than a ``bool``."""
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value: Any) -> bool:  # noqa: ANN401 - JSON value
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _artifact(root: Path, name: Any, label: str, errors: list[str]) -> None:  # noqa: ANN401 - JSON value
    if not isinstance(name, str) or not name or "\\" in name:
        errors.append(f"{label}: missing or unsafe relative path")
        return
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        errors.append(f"{label}: unsafe relative path {name!r}")
        return
    try:
        target = (root / path).resolve()
        if not target.is_relative_to(root.resolve()):
            errors.append(f"{label}: path escapes artifact directory")
        elif not target.is_file():
            errors.append(f"{label}: missing file {name!r}")
    except (OSError, ValueError, RuntimeError):
        errors.append(f"{label}: invalid artifact path")


def validate_dataset_records(
    payload: Any,  # noqa: ANN401 - parsed JSON of unknown shape
    outdir: str | os.PathLike[str],
    *,
    start_idx: int,
    end_idx: int,
    images_per_sample: int,
    render_masks: bool,
    num_keypoints: int | None = None,
) -> list[str]:
    """Return errors for incomplete/corrupt single-object render records.

    ``file_name`` is relative to ``outdir/images``; ``mask_file`` is relative
    to ``outdir``. Stereo view names may be configured, but camera indices and
    pair IDs must match the deterministic image IDs.
    """
    if (
        not is_integer(start_idx)
        or not is_integer(end_idx)
        or start_idx < 0
        or end_idx < start_idx
        or not is_integer(images_per_sample)
        or images_per_sample not in (1, 2)
    ):
        raise ValueError("invalid expected sample range or images_per_sample")
    if num_keypoints is not None and (not is_integer(num_keypoints) or num_keypoints < 0):
        raise ValueError("num_keypoints must be a nonnegative integer")
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["dataset payload must be an object"]
    images, annotations = payload.get("images"), payload.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        return ["images and annotations must be lists"]
    expected = set(range(start_idx * images_per_sample, end_idx * images_per_sample))
    root = Path(outdir)
    image_ids, annotation_ids, references = [], [], []
    image_by_id = {}
    paths = []
    views_by_sample: dict[int, list[str]] = {}
    for index, record in enumerate(images):
        label = f"image[{index}]"
        if not isinstance(record, dict):
            errors.append(f"{label}: must be an object")
            continue
        image_id = record.get("id")
        if not is_integer(image_id):
            errors.append(f"{label}: invalid id")
        else:
            image_ids.append(image_id)
            image_by_id[image_id] = record
            if images_per_sample == 2:
                if (
                    not is_integer(record.get("pair_id"))
                    or not is_integer(record.get("camera_index"))
                    or record["pair_id"] != image_id // 2
                    or record["camera_index"] != image_id % 2
                ):
                    errors.append(f"{label}: stereo pair_id/camera_index does not match image id")
                view = record.get("view")
                if not isinstance(view, str) or not view:
                    errors.append(f"{label}: missing stereo view")
                else:
                    views_by_sample.setdefault(image_id // 2, []).append(view)
        name = record.get("file_name")
        _artifact(root / "images", name, f"{label} RGB", errors)
        if isinstance(name, str):
            paths.append(name)
        for dimension in ("width", "height"):
            if not is_integer(record.get(dimension)) or record[dimension] <= 0:
                errors.append(f"{label}: invalid {dimension}")
    if len(set(paths)) != len(paths):
        errors.append("duplicate RGB file paths")
    for sample, views in views_by_sample.items():
        if len(views) != 2 or len(set(views)) != 2:
            errors.append(f"sample {sample}: missing or duplicate stereo views")
    for index, record in enumerate(annotations):
        label = f"annotation[{index}]"
        if not isinstance(record, dict):
            errors.append(f"{label}: must be an object")
            continue
        ann_id, image_id = record.get("id"), record.get("image_id")
        if not is_integer(ann_id):
            errors.append(f"{label}: invalid id")
        else:
            annotation_ids.append(ann_id)
        if not is_integer(image_id):
            errors.append(f"{label}: invalid image_id reference")
        else:
            references.append(image_id)
            if image_id not in image_by_id:
                errors.append(f"{label}: image_id references missing image {image_id}")
            elif images_per_sample == 2:
                image = image_by_id[image_id]
                for field in ("pair_id", "camera_index", "view"):
                    if record.get(field) != image.get(field):
                        errors.append(f"{label}: stereo {field} disagrees with image")
        keypoints = record.get("keypoints")
        if (
            not isinstance(keypoints, list)
            or len(keypoints) % 3
            or (num_keypoints is not None and len(keypoints) != num_keypoints * 3)
            or not all(_finite(v) for v in keypoints)
        ):
            errors.append(f"{label}: invalid keypoints shape or nonfinite values")
        else:
            visibility = keypoints[2::3]
            if any(v not in (0, 1, 2) for v in visibility):
                errors.append(f"{label}: invalid keypoint visibility")
            if not is_integer(record.get("num_keypoints")) or record["num_keypoints"] != sum(
                v > 0 for v in visibility
            ):
                errors.append(f"{label}: num_keypoints disagrees with visible keypoints")
        bbox = record.get("bbox")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(_finite(v) for v in bbox)
            or any(v < 0 for v in bbox[2:])
        ):
            errors.append(f"{label}: invalid bbox")
        if "area" in record and (not _finite(record["area"]) or record["area"] < 0):
            errors.append(f"{label}: invalid area")
        for field in ("camera_extrinsics", "object_pose"):
            if field in record:
                matrix = record[field]
                if (
                    not isinstance(matrix, list)
                    or len(matrix) != 4
                    or any(
                        not isinstance(row, list)
                        or len(row) != 4
                        or not all(_finite(v) for v in row)
                        for row in matrix
                    )
                ):
                    errors.append(f"{label}: invalid {field} matrix")
        if "camera_intrinsics" in record:
            intrinsics = record["camera_intrinsics"]
            if (
                not isinstance(intrinsics, dict)
                or not all(_finite(intrinsics.get(k)) for k in ("fx", "fy", "cx", "cy"))
                or intrinsics["fx"] <= 0
                or intrinsics["fy"] <= 0
            ):
                errors.append(f"{label}: invalid camera_intrinsics")
        if render_masks or record.get("mask_file") is not None:
            _artifact(root, record.get("mask_file"), f"{label} mask", errors)
    for label, values in (("image IDs", image_ids), ("annotation IDs", annotation_ids)):
        if len(set(values)) != len(values):
            errors.append(f"duplicate {label}")
    if set(image_ids) != expected:
        errors.append("image IDs do not match expected sample range")
    counts = Counter(references)
    if set(counts) != expected or any(count != 1 for count in counts.values()):
        errors.append("expected exactly one annotation per image in requested range")
    return errors


def merge_annotations(
    outdir: str | os.PathLike[str], num_workers: int | None = None
) -> dict[str, Any]:
    """Merge per-worker annotation files into single COCO format file.

    ``num_workers=None`` merges every ``annotations_worker_*.json`` present, which is what a
    sharded render (several launchers writing into one directory) needs.

    Args:
        outdir: Dataset directory holding the worker files and ``keypoints_3d.json``.
        num_workers: Merge exactly workers ``0..num_workers-1``, or every final file if ``None``.

    Returns:
        Image and annotation counts and whether masks were rendered.

    Raises:
        ValueError: Image or annotation IDs are duplicated or do not pair one to one.
    """
    outdir = Path(outdir)

    all_images = []
    all_annotations = []
    keypoint_names = None
    mesh_name = None
    images_per_sample = 1
    stereo = None
    render_masks = False
    mask_dir = None

    if num_workers is None:
        # Final files only: annotations_worker_NNN.json, never .partial.json or .tmp writes.
        ann_files = sorted(
            p
            for p in outdir.glob("annotations_worker_*.json")
            if p.stem.removeprefix("annotations_worker_").isdigit()
        )
    else:
        ann_files = [
            outdir / f"annotations_worker_{worker_id:03d}.json" for worker_id in range(num_workers)
        ]
    for ann_file in ann_files:
        if ann_file.exists():
            with open(ann_file) as f:
                data = json.load(f)
            all_images.extend(data.get("images", []))
            all_annotations.extend(data.get("annotations", []))
            images_per_sample = max(images_per_sample, int(data.get("images_per_sample", 1) or 1))
            if data.get("stereo", {}).get("enabled"):
                stereo = data.get("stereo")
            if data.get("render_masks"):
                render_masks = True
                mask_dir = data.get("mask_dir") or mask_dir or "masks"

    # Reject ambiguous ownership before publishing a merged artifact.
    for label, records, key in (("image", all_images, "id"), ("annotation", all_annotations, "id")):
        ids = [record.get(key) for record in records]
        if any(not is_integer(value) for value in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"invalid or duplicate {label} IDs in worker annotations")
    image_ids = {record["id"] for record in all_images}
    references = [record.get("image_id") for record in all_annotations]
    if (
        any(not is_integer(value) for value in references)
        or set(references) != image_ids
        or len(references) != len(set(references))
    ):
        raise ValueError("expected exactly one annotation referencing each merged image")

    # Load keypoints info
    kp_file = outdir / "keypoints_3d.json"
    if kp_file.exists():
        with open(kp_file) as f:
            kp_data = json.load(f)
        keypoint_names = kp_data.get("keypoint_names", [])
        mesh_name = kp_data.get("object", "object")

    # Sort by image ID
    all_images.sort(key=lambda x: x["id"])
    all_annotations.sort(key=lambda x: x["id"])

    # Write merged file
    merged = {
        "info": {
            "description": f"DOPER synthetic data - {len(all_images)} images",
            "version": "1.0",
            "num_samples": len(all_images) // images_per_sample
            if images_per_sample
            else len(all_images),
            "images_per_sample": images_per_sample,
            "stereo": stereo,
            "render_masks": render_masks,
            "mask_dir": mask_dir if render_masks else None,
        },
        "categories": [
            {
                "id": 1,
                "name": mesh_name or "object",
                "keypoints": keypoint_names or [],
                "skeleton": [],
            }
        ],
        "images": all_images,
        "annotations": all_annotations,
    }

    with open(outdir / "annotations.json", "w") as f:
        json.dump(merged, f)

    print(f"  Merged {len(all_images)} images, {len(all_annotations)} annotations")
    return {"images": len(all_images), "annotations": len(all_annotations), "masks": render_masks}
