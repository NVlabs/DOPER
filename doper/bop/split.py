# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hold out whole scenes of a converted BOP ``train_pbr`` root as a validation split.

PBR scenes are rendered as continuous camera sweeps, so a random frame split would put
near-duplicate frames on both sides and the validation number would stop meaning anything.
The output has ``train/`` and ``val/`` directories in the layout ``DoperDataset`` reads, with
images and masks symlinked rather than copied, and ``keypoints_3d.json`` at the top. The links
are relative, so a dataset root can be moved as a whole.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_SEED = 42
DEFAULT_VAL_SCENES = 5


@dataclass(frozen=True)
class SplitSummary:
    """Outcome of :func:`split_scenes`.

    Attributes:
        out: The split root.
        train_scenes: Scenes in ``train/``.
        val_scenes: Held-out scenes in ``val/``.
        train_images: Images in ``train/``.
        val_images: Images in ``val/``.
        train_annotations: Annotations in ``train/``.
        val_annotations: Annotations in ``val/``.
        multi_instance_images: Images that carry more than one instance of the object.
    """

    out: Path
    train_scenes: tuple[str, ...]
    val_scenes: tuple[str, ...]
    train_images: int
    val_images: int
    train_annotations: int
    val_annotations: int
    multi_instance_images: int


def resolve_root(path: Path) -> Path:
    """Return the directory holding ``annotations.json``: ``path`` or ``path/train``.

    Raises:
        FileNotFoundError: If neither has an ``annotations.json``.
    """
    for candidate in (path, path / "train"):
        if (candidate / "annotations.json").is_file():
            return candidate
    raise FileNotFoundError(f"no annotations.json under {path} or {path / 'train'}")


def scene_of(image: dict[str, Any], annotations: list[dict[str, Any]]) -> str:
    """Return the scene id of an image.

    The converter records it under ``bop``. The file name is a far weaker guarantee, so it is
    used only when no annotation records the scene.
    """
    for annotation in annotations:
        scene = (annotation.get("bop") or {}).get("scene")
        if scene is not None:
            return str(scene)
    return str(image["file_name"]).split("_")[0]


def split_scenes(
    src: Path,
    dst: Path,
    obj_id: int,
    keypoints: Path,
    *,
    val_scenes: int = DEFAULT_VAL_SCENES,
    seed: int = DEFAULT_SEED,
    force: bool = False,
) -> SplitSummary:
    """Split a converted root into ``train/`` and ``val/`` by whole scenes.

    The held-out scenes are a seeded sample of the sorted scene ids. Every instance of the
    object in a frame is kept: a frame can hold several, each its own annotation sharing one
    image.

    Args:
        src: Converted root (or a root whose ``train/`` holds the converted data).
        dst: Output root.
        obj_id: BOP object id the data was converted for.
        keypoints: ``keypoints_3d.json`` copied to ``dst``.
        val_scenes: Number of scenes to hold out (at most all of them).
        seed: Seed of the scene sample.
        force: Replace an existing ``dst``.

    Returns:
        The split summary.

    Raises:
        ValueError: If ``obj_id`` is not positive or ``val_scenes`` is negative.
        FileExistsError: If ``dst`` exists and ``force`` is false.
        FileNotFoundError: If the source has no annotations or ``keypoints`` is missing.
    """
    if obj_id <= 0:
        raise ValueError("obj_id must be positive")
    if val_scenes < 0:
        raise ValueError("val_scenes must be non-negative")
    data = json.loads((resolve_root(Path(src)) / "annotations.json").read_text())
    by_image = _by_image(data)
    scenes = sorted({scene_of(im, by_image.get(im["id"], [])) for im in data["images"]})
    held_out = set(random.Random(seed).sample(scenes, min(val_scenes, len(scenes))))
    return split_by(src, dst, obj_id, keypoints, held_out.__contains__, force=force)


def _by_image(data: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in data["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    return by_image


def split_by(
    src: Path,
    dst: Path,
    obj_id: int,
    keypoints: Path,
    is_validation: Callable[[str], bool],
    *,
    order: Callable[[str], Any] | None = None,
    force: bool = False,
) -> SplitSummary:
    """Split a converted root into ``train/`` and ``val/`` by a rule on whole scenes.

    Args:
        src: Converted root (or a root whose ``train/`` holds the converted data).
        dst: Output root.
        obj_id: BOP object id the data was converted for.
        keypoints: ``keypoints_3d.json`` copied to ``dst``.
        is_validation: Whether a scene id goes to ``val/``.
        order: Sort key of a scene id; images are listed by it (stable, so a scene keeps the
            converter's frame order). None keeps the converter's order.
        force: Replace an existing ``dst``.

    Returns:
        The split summary.

    Raises:
        ValueError: If ``obj_id`` is not positive.
        FileExistsError: If ``dst`` exists and ``force`` is false.
        FileNotFoundError: If the source has no annotations or ``keypoints`` is missing.
    """
    if obj_id <= 0:
        raise ValueError("obj_id must be positive")
    root = resolve_root(Path(src))
    keypoints = Path(keypoints)
    if not keypoints.is_file():
        raise FileNotFoundError(f"keypoints file not found: {keypoints}")
    dst = Path(dst)
    if dst.exists() and not force:
        raise FileExistsError(f"{dst} exists; pass force=True")
    data = json.loads((root / "annotations.json").read_text())
    by_image = _by_image(data)
    if order is not None:
        key = order
        data["images"] = sorted(
            data["images"], key=lambda im: key(scene_of(im, by_image.get(im["id"], [])))
        )
    scenes = sorted({scene_of(im, by_image.get(im["id"], [])) for im in data["images"]})
    held_out = {scene for scene in scenes if is_validation(scene)}
    if dst.exists():
        shutil.rmtree(dst)
    counts = {
        split: _write_split(root, dst, split, data, by_image, held_out)
        for split in ("train", "val")
    }
    shutil.copyfile(keypoints, dst / "keypoints_3d.json")
    return SplitSummary(
        out=dst,
        train_scenes=tuple(s for s in scenes if s not in held_out),
        val_scenes=tuple(sorted(held_out)),
        train_images=counts["train"][0],
        val_images=counts["val"][0],
        train_annotations=counts["train"][1],
        val_annotations=counts["val"][1],
        multi_instance_images=sum(1 for group in by_image.values() if len(group) > 1),
    )


def _write_split(
    root: Path,
    dst: Path,
    split: str,
    data: dict[str, Any],
    by_image: dict[int, list[dict[str, Any]]],
    held_out: set[str],
) -> tuple[int, int]:
    """Write one split's annotations and links; return its image and annotation counts."""
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    (dst / split / "images").mkdir(parents=True)
    (dst / split / "masks").mkdir(parents=True)
    for image in data["images"]:
        group = by_image.get(image["id"], [])
        if not group or (scene_of(image, group) in held_out) != (split == "val"):
            continue
        new_id = len(images)
        images.append({**image, "id": new_id})
        links = [("images", image["file_name"])]
        for annotation in group:
            # Some roots carry only ``segmentation``; the trainer reads ``mask_file``.
            mask_name = Path(annotation.get("mask_file") or annotation["segmentation"]).name
            annotations.append(
                {
                    **annotation,
                    "id": len(annotations),
                    "image_id": new_id,
                    "mask_file": f"masks/{mask_name}",
                }
            )
            links.append(("masks", mask_name))
        for subdirectory, name in links:
            target = (root / subdirectory / name).resolve()
            link = dst / split / subdirectory / name
            if target.exists() and not link.exists():
                os.symlink(os.path.relpath(target, link.parent.resolve()), link)
    payload = {"images": images, "annotations": annotations, "categories": data["categories"]}
    (dst / split / "annotations.json").write_text(json.dumps(payload))
    return len(images), len(annotations)
