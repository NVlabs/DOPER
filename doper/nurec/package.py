# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualification, packaging and QC of the NuRec adaptation dataset.

**Qualification** (``assess``/``qualify``). A keypoint of a view is
visible when it is ray-visible (the geometry label), its rounded pixel is inside the image and
inside the mask eroded with an ``erode_px`` x ``erode_px`` kernel (5 x 5 by default, about
2 px), and is not of the excluded color (recipe ``labels.exclude_hsv``). The mask coverage is
``|mask and CAD| / |CAD|``. A view qualifies with at least ``min_visible`` visible keypoints
and a coverage of at least ``min_coverage``. In the ``cad_only`` mask mode there is no mask:
visibility is the ray test and the color exclusion, and the coverage is not checked.

**Packaging** (``package_dataset``). The accepted views are
split into train and validation:

* with two or more capture lineages, whole lineage groups are assigned (the groups shuffled by
  ``default_rng(seed)``, validation filled first), so no lineage straddles the split;
* with one lineage, only when ``single_reconstruction_split`` is set:
  ``default_rng(seed).shuffle`` of the views, the first ``train`` to training and the next
  ``val`` to validation. Validation then shares the scene with training; the receipt records
  ``lineage_split: false`` with that caveat.

Each split is a COCO directory the DOPER trainer reads (``images/``, ``masks/``,
``annotations.json``): images hard-linked (copied across file systems) as
``nurec_NNNNNN.png``, one annotation per image with the keypoints as ``(x, y, v)``,
``v`` in {0, 2} and invisible keypoints zeroed, the CAD bbox clipped to the image, the camera
``K`` and the metric ``T_camera_object_m``. With ``sam2`` masks the visible-mask target is
SAM2 and CAD (``mask_file``). A ``cad_only`` dataset writes **no** ``mask_file``: the CAD
silhouette is amodal (it ignores anything in front of the object), so it supervises
boxes and keypoints only and never the visible-mask head. The dataset also holds the keypoints
file, ``pose_manifest.json``, ``selection_manifest.json``, ``sha256.json`` (every file's
SHA-256) and the receipt ``package.json``.

**QC** (``qc_dataset``) re-checks a packaged dataset
and raises ``DatasetQcError`` naming the first failed check.

Frames and units: OpenCV camera axes; ``T_camera_object_m`` maps object meters into the camera;
pixels are ``(x, y)``.
"""

from __future__ import annotations

import collections
import json
import shutil
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from ..bop.sources import publish_json, sha256_file, sha256_files
from ..keypoints import KEYPOINTS_NAME
from ..recipes.nurec_schema import MASK_MODES, HsvExclusion
from ._files import link_or_copy, read_image
from .labels import ViewLabel
from .masks import excluded_pixels

VISIBLE = 2
IMAGE_PATTERN = "nurec_{index:06d}.png"
IMAGES_DIR = "images"
MASKS_DIR = "masks"
ANNOTATIONS_NAME = "annotations.json"
POSE_MANIFEST_NAME = "pose_manifest.json"
SELECTION_NAME = "selection_manifest.json"
SHA256_NAME = "sha256.json"
PACKAGE_NAME = "package.json"
SPLITS = ("train", "val")
CATEGORY_ID = 1
# The QC tolerances: rotation orthonormality and determinant, and the
# reprojection of the CAD keypoints through the stored pose against the stored keypoints.
ROTATION_ATOL = 1e-6
REPROJECTION_MAX_PX = 1e-6
LINEAGE_CAVEAT = ("validation views come from the same reconstruction (one capture lineage) as "
                  "the training views and share its scene: a reference-reproduction exception, not "
                  "an independent validation")  # fmt: skip


@dataclass(frozen=True)
class Qualified:
    """The qualification of one view (accepted or not).

    Attributes:
        label: The geometry label (without its mask array).
        visibility: ``(n,)`` visible keypoints (ray, eroded mask, color).
        coverage: ``|mask and CAD| / |CAD|``, or None without a mask (``cad_only``).
        accepted: Whether the view qualifies.
        reason: ``accepted`` or why not.
        image: The view's render, when assessed from files.
        mask: Its SAM2 mask, when assessed from files with one.
        cad_mask: Its CAD silhouette, when assessed from files.
    """

    label: ViewLabel
    visibility: NDArray[np.bool_]
    coverage: float | None
    accepted: bool
    reason: str
    image: Path | None = None
    mask: Path | None = None
    cad_mask: Path | None = None

    @property
    def visible_count(self) -> int:
        """Visible keypoints."""
        return int(self.visibility.sum())

    def to_record(self) -> dict[str, Any]:
        """The ``qualified_labels.json`` row: the label's row and the qualification."""
        return {**self.label.to_record(),
                "visibility": [VISIBLE if v else 0 for v in self.visibility],
                "visible_count": self.visible_count, "cad_mask_coverage": self.coverage,
                "accepted": self.accepted, "reason": self.reason}  # fmt: skip

    @classmethod
    def from_record(cls, record: Mapping[str, Any], *, image: str | Path | None = None,
                    mask: str | Path | None = None,
                    cad_mask: str | Path | None = None) -> Qualified:  # fmt: skip
        """Rebuild a qualification from its ``qualified_labels.json`` row and its files."""
        return cls(label=ViewLabel.from_record(record),
                   visibility=np.array(record["visibility"]) > 0,
                   coverage=record["cad_mask_coverage"], accepted=bool(record["accepted"]),
                   reason=str(record["reason"]), image=None if image is None else Path(image),
                   mask=None if mask is None else Path(mask),
                   cad_mask=None if cad_mask is None else Path(cad_mask))  # fmt: skip


def _binary(mask: NDArray[Any]) -> NDArray[np.uint8]:
    return (np.asarray(mask) > 0).astype(np.uint8)


def assess(
    label: ViewLabel,
    mask: NDArray[Any] | None,
    image: NDArray[np.uint8],
    *,
    erode_px: int = 5,
    min_visible: int = 12,
    min_coverage: float = 0.5,
    exclude_hsv: HsvExclusion | None = None,
) -> Qualified:
    """Qualify one view, accepted or not.

    Args:
        label: The geometry label; it needs its ``cad_mask`` array when ``mask`` is given.
        mask: The view's object mask (SAM2; nonzero is object), or None (``cad_only``).
        image: ``(h, w, 3)`` BGR render.
        erode_px: Side of the square erosion kernel applied to the mask (0 or 1: none).
        min_visible: Fewest visible keypoints.
        min_coverage: Smallest mask coverage of the CAD silhouette.
        exclude_hsv: The excluded color, or None.

    Returns:
        The qualification (``label`` without its mask array).

    Raises:
        ValueError: For an image or mask of another size, or a mask without the CAD mask.
    """
    width, height = label.image_size
    if image.shape[:2] != (height, width):
        raise ValueError(f"view {label.index}: image {image.shape[:2]} is not {(height, width)}")
    excluded = excluded_pixels(image, exclude_hsv)
    eroded = None
    coverage = None
    if mask is not None:
        binary = _binary(mask)
        if label.cad_mask is None or binary.shape != (height, width):
            raise ValueError(f"view {label.index}: the mask must be {(height, width)} and the "
                             "label needs its cad_mask array")  # fmt: skip
        kernel = np.ones((erode_px, erode_px), np.uint8)
        eroded = cv2.erode(binary, kernel) if erode_px > 1 else binary
        cad = np.asarray(label.cad_mask) > 0
        coverage = float((binary.astype(bool) & cad).sum() / max(1, cad.sum()))
    inside = np.zeros(len(label.keypoints_xy), dtype=bool)
    for i, point in enumerate(label.keypoints_xy):
        x, y = np.round(point).astype(int)
        inside[i] = (0 <= x < width and 0 <= y < height and not excluded[y, x]
                     and (eroded is None or eroded[y, x] > 0))  # fmt: skip
    visibility = np.asarray(label.ray_visible, dtype=bool) & inside
    reasons = []
    if int(visibility.sum()) < min_visible:
        reasons.append(f"fewer_than_{min_visible}_visible")
    if coverage is not None and coverage < min_coverage:
        reasons.append(f"mask_coverage_below_{min_coverage}")
    return Qualified(label=replace(label, cad_mask=None), visibility=visibility,
                     coverage=coverage, accepted=not reasons,
                     reason="accepted" if not reasons else "+".join(reasons))  # fmt: skip


def qualify(
    label: ViewLabel,
    mask: NDArray[Any] | None,
    image: NDArray[np.uint8],
    *,
    erode_px: int = 5,
    min_visible: int = 12,
    min_coverage: float = 0.5,
    exclude_hsv: HsvExclusion | None = None,
) -> Qualified | None:
    """``assess`` returning only an accepted view (None when the view does not qualify)."""
    result = assess(label, mask, image, erode_px=erode_px, min_visible=min_visible,
                    min_coverage=min_coverage, exclude_hsv=exclude_hsv)  # fmt: skip
    return result if result.accepted else None


def assess_files(
    label: ViewLabel, *, image: str | Path, cad_mask: str | Path, mask: str | Path | None = None,
    **rules: Any,  # noqa: ANN401
) -> Qualified:  # fmt: skip
    """``assess`` a view from its files; the result keeps the paths for ``package_dataset``.

    Args:
        label: The geometry label (its mask array is read from ``cad_mask``).
        image: The render (BGR PNG).
        cad_mask: The CAD silhouette PNG.
        mask: The SAM2 mask PNG, or None (``cad_only``).
        **rules: ``assess`` keywords.

    Raises:
        FileNotFoundError: For an unreadable file.
    """
    image, cad_mask = Path(image), Path(cad_mask)
    mask = None if mask is None else Path(mask)
    with_cad = replace(label, cad_mask=read_image(cad_mask, cv2.IMREAD_GRAYSCALE))
    result = assess(with_cad, None if mask is None else read_image(mask, cv2.IMREAD_GRAYSCALE),
                    read_image(image, cv2.IMREAD_COLOR), **rules)  # fmt: skip
    return replace(result, image=image, mask=mask, cad_mask=cad_mask)


@dataclass(frozen=True)
class _Split:
    train: list[Qualified]
    val: list[Qualified]
    unselected: list[Qualified]
    lineage_split: bool


def _split(accepted: list[Qualified], seed: int, train: int, val: int,
           single_reconstruction_split: bool) -> _Split:  # fmt: skip
    groups: dict[str, list[Qualified]] = {}
    for item in accepted:
        groups.setdefault(item.label.lineage, []).append(item)
    rng = np.random.default_rng(seed)
    if len(groups) < 2:
        if not single_reconstruction_split:
            raise ValueError("every accepted view comes from one capture lineage, so validation "
                             "would share the scene with training; set "
                             "split.single_reconstruction_split to accept that split")  # fmt: skip
        views = list(accepted)
        rng.shuffle(views)  # type: ignore[arg-type]
        return _Split(views[:train], views[train : train + val], views[train + val :], False)
    order = sorted(groups)
    rng.shuffle(order)  # type: ignore[arg-type]
    val_pool: list[Qualified] = []
    val_groups = 0
    while len(val_pool) < val and val_groups < len(order):
        val_pool.extend(groups[order[val_groups]])
        val_groups += 1
    train_pool = [item for name in order[val_groups:] for item in groups[name]]
    if len(val_pool) < val or len(train_pool) < train:
        raise ValueError(f"the {len(groups)} lineage groups give {len(train_pool)} training and "
                         f"{len(val_pool)} validation views without splitting a group; "
                         f"{train} and {val} are needed")  # fmt: skip
    rng.shuffle(val_pool)  # type: ignore[arg-type]
    rng.shuffle(train_pool)  # type: ignore[arg-type]
    return _Split(train_pool[:train], val_pool[:val], val_pool[val:] + train_pool[train:], True)


def _bbox(label: ViewLabel) -> list[float]:
    """The unclipped CAD bbox clipped to the image (``[x, y, w, h]``)."""
    x, y, w, h = label.cad_bbox
    width, height = label.image_size
    x0, y0, x1, y1 = max(0, x), max(0, y), min(width, x + w), min(height, y + h)
    return [x0, y0, x1 - x0, y1 - y0]


def _annotation(item: Qualified, number: int, split_dir: Path, name: str, mask_mode: str,
                min_visible: int) -> dict[str, Any]:  # fmt: skip
    label, mask = item.label, item.mask if mask_mode == "sam2" else None
    if item.image is None or item.cad_mask is None or (mask_mode == "sam2" and mask is None):
        raise ValueError(f"view {label.index} has no files to package (use assess_files)")
    link_or_copy(item.image, split_dir / IMAGES_DIR / name)
    cad = read_image(item.cad_mask, cv2.IMREAD_GRAYSCALE) > 0
    if mask is not None:
        final = (read_image(mask, cv2.IMREAD_GRAYSCALE) > 0) & cad
        if not cv2.imwrite(str(split_dir / MASKS_DIR / name), final.astype(np.uint8) * 255):
            raise OSError(f"could not write {split_dir / MASKS_DIR / name}")
        area = int(final.sum())
    else:
        area = int(cad.sum())
    visibility = np.where(item.visibility, VISIBLE, 0)
    uv = np.array(label.keypoints_xy, dtype=np.float64)
    uv[visibility == 0] = 0
    flat = np.c_[uv, visibility].ravel()
    if item.visible_count < min_visible or not np.isfinite(flat).all():
        raise ValueError(f"view {label.index}: {item.visible_count} visible keypoints (needs "
                         f"{min_visible}) or non-finite keypoints")  # fmt: skip
    pose = label.T_camera_object_m.tolist()
    annotation = {"id": number, "image_id": number, "category_id": CATEGORY_ID,
                  "bbox": _bbox(label), "area": area, "iscrowd": 0, "keypoints": flat.tolist(),
                  "num_keypoints": int((visibility > 0).sum())}  # fmt: skip
    if mask_mode == "sam2":
        annotation["mask_file"] = f"{MASKS_DIR}/{name}"
    return {**annotation, "camera_intrinsics": label.K.tolist(), "camera_extrinsics": pose,
            "T_camera_object_m": pose, "source_render_index": label.index}  # fmt: skip


def _keypoint_names(path: Path) -> tuple[list[str], int]:
    data = json.loads(path.read_text())
    count = len(data["keypoints_3d"])
    names = data.get("keypoint_names") or [f"kp_{i:02d}" for i in range(count)]
    if len(names) != count:
        raise ValueError(f"{path}: {len(names)} keypoint names for {count} keypoints")
    return [str(name) for name in names], count


def _write_split(out: Path, split: str, chosen: Sequence[Qualified], category: dict[str, Any],
                 mask_mode: str, min_visible: int) -> list[dict[str, Any]]:  # fmt: skip
    split_dir = out / split
    (split_dir / IMAGES_DIR).mkdir(parents=True)
    if mask_mode == "sam2":
        (split_dir / MASKS_DIR).mkdir()
    images, annotations, poses = [], [], []
    for number, item in enumerate(chosen, 1):
        name = IMAGE_PATTERN.format(index=item.label.index)
        width, height = item.label.image_size
        annotations.append(_annotation(item, number, split_dir, name, mask_mode, min_visible))
        images.append({"id": number, "width": width, "height": height, "file_name": name})
        poses.append({"split": split, "file_name": name, **item.to_record()})
    publish_json(split_dir / ANNOTATIONS_NAME, {"images": images, "annotations": annotations,
                                                "categories": [category]})  # fmt: skip
    return poses


def package_dataset(
    qualified: Sequence[Qualified],
    out: str | Path,
    *,
    keypoints_file: str | Path,
    seed: int = 817,
    train: int = 1024,
    val: int = 128,
    category: str = "object",
    single_reconstruction_split: bool,
    mask_mode: str = "sam2",
    min_visible: int = 12,
) -> Path:
    """Split the accepted views and write the COCO adaptation dataset.

    Args:
        qualified: Every assessed view (``assess_files``), accepted or not; the rejected ones
            are recorded in ``selection_manifest.json``.
        out: The dataset directory (created; it must not exist or be empty).
        keypoints_file: The object's ``keypoints_3d.json`` (copied verbatim; its
            ``keypoint_names`` name the category's keypoints).
        seed: Split seed.
        train: Training views.
        val: Validation views.
        category: COCO category name.
        single_reconstruction_split: Allow the split of views of one capture lineage.
        mask_mode: ``sam2`` (writes the SAM2-and-CAD masks) or ``cad_only`` (no mask file).
        min_visible: Fewest visible keypoints of a packaged view (recorded for the QC).

    Returns:
        ``out``.

    Raises:
        ValueError: For an unknown mask mode, too few accepted views, a one-lineage split
            without ``single_reconstruction_split``, a non-empty ``out`` or duplicate views.
    """
    if mask_mode not in MASK_MODES:
        raise ValueError(f"mask_mode must be one of {MASK_MODES}, got {mask_mode!r}")
    out, keypoints_file = Path(out), Path(keypoints_file)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty")
    accepted = [item for item in qualified if item.accepted]
    if len(accepted) < train + val:
        raise ValueError(f"only {len(accepted)} of {len(qualified)} views are accepted, "
                         f"{train + val} are needed: render more novel views")  # fmt: skip
    split = _split(accepted, seed, train, val, single_reconstruction_split)
    selected = [*split.train, *split.val]
    names = [IMAGE_PATTERN.format(index=item.label.index) for item in selected]
    poses = {item.label.T_camera_nre.tobytes() for item in selected}
    if len(set(names)) != len(names) or len(poses) != len(selected):
        raise ValueError("the selected views repeat a render index or a camera pose")
    keypoint_names, count = _keypoint_names(keypoints_file)
    if any(len(item.label.keypoints_xy) != count for item in selected):
        raise ValueError(f"the labels do not have the {count} keypoints of {keypoints_file}")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(keypoints_file, out / KEYPOINTS_NAME)
    record = {"id": CATEGORY_ID, "name": category, "keypoints": keypoint_names, "skeleton": []}
    manifest_rows = [
        row
        for name, chosen in (("train", split.train), ("val", split.val))
        for row in _write_split(out, name, chosen, record, mask_mode, min_visible)
    ]
    publish_json(out / POSE_MANIFEST_NAME, manifest_rows)
    publish_json(out / SELECTION_NAME, {
        "total_candidates": len(qualified), "accepted_candidates": len(accepted),
        "selected": len(selected),
        "rejected": [{"index": item.label.index, "reason": item.reason,
                      "visible_count": item.visible_count, "cad_mask_coverage": item.coverage}
                     for item in qualified if not item.accepted],
        "unselected_valid": [item.label.index for item in split.unselected],
    })  # fmt: skip
    files = sorted(p for p in out.rglob("*") if p.is_file())
    hashes = dict(zip((str(p.relative_to(out)) for p in files), sha256_files(files), strict=True))
    publish_json(out / SHA256_NAME, hashes)
    publish_json(out / PACKAGE_NAME, {
        "mask_mode": mask_mode, "reference_path": mask_mode == "sam2",
        "lineage_split": split.lineage_split,
        "lineage_caveat": None if split.lineage_split else LINEAGE_CAVEAT,
        "lineages": sorted({item.label.lineage for item in accepted}),
        "reconstruction_ids": sorted({item.label.reconstruction_id for item in accepted}),
        "seed": seed, "train": len(split.train), "val": len(split.val), "category": category,
        "min_visible": min_visible, "keypoints_sha256": hashes[KEYPOINTS_NAME],
        "candidate_seed_views": sorted({item.label.seed_view for item in qualified}),
        "sha256_manifest": sha256_file(out / SHA256_NAME),
    })  # fmt: skip
    return out


class DatasetQcError(ValueError):
    """A packaged dataset failed a QC check.

    Attributes:
        check: The failed check's name.
    """

    def __init__(self, check: str, detail: str) -> None:
        """Name the check and what failed."""
        super().__init__(f"dataset QC check {check} failed: {detail}")
        self.check = check


def _require(ok: bool, check: str, detail: str) -> None:
    if not ok:
        raise DatasetQcError(check, detail)


def _qc_annotation(root: Path, split: str, image: Mapping[str, Any], ann: Mapping[str, Any],
                   points: NDArray[np.float64], min_visible: int,
                   mask_required: bool | None) -> tuple[float, int]:  # fmt: skip
    """Check one annotation; return its largest reprojection error and visible count.

    ``mask_required`` says whether the annotation must carry a ``mask_file`` (a ``sam2``
    dataset) or must not (``cad_only``); None when the dataset records no mask mode.
    """
    where = f"{split}/{image.get('file_name')}"
    _require(image["id"] == ann["image_id"], "image_id", f"{where}: image and annotation ids")
    flat = np.array(ann["keypoints"], dtype=np.float64)
    _require(flat.size == 3 * len(points), "keypoint_count", f"{where}: {flat.size // 3} keypoints")
    keypoints = flat.reshape(-1, 3)
    _require(bool(np.isfinite(keypoints).all()), "finite", f"{where}: non-finite keypoints")
    visible = keypoints[:, 2] > 0
    _require(
        set(keypoints[:, 2]) <= {0, VISIBLE}, "visibility_values", f"{where}: v not in {{0, 2}}"
    )
    _require(int(visible.sum()) >= min_visible, "min_visible",
             f"{where}: {int(visible.sum())} visible, fewer than {min_visible}")  # fmt: skip
    _require(bool(np.all(keypoints[~visible, :2] == 0)), "invisible_zeroed",
             f"{where}: an invisible keypoint is not (0, 0)")  # fmt: skip
    _require(ann["num_keypoints"] == int(visible.sum()), "num_keypoints", f"{where}: num_keypoints")
    pose = np.array(ann["T_camera_object_m"], dtype=np.float64)
    rotation = pose[:3, :3]
    _require(float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))) < ROTATION_ATOL
             and abs(float(np.linalg.det(rotation)) - 1) < ROTATION_ATOL, "rotation",
             f"{where}: the rotation is not orthonormal with determinant 1")  # fmt: skip
    projected = (points @ rotation.T + pose[:3, 3]) @ np.array(ann["camera_intrinsics"]).T
    uv = projected[:, :2] / projected[:, 2:]
    errors = np.linalg.norm(uv[visible] - keypoints[visible, :2], axis=1)
    error = float(errors.max(initial=0.0))
    _require(error < REPROJECTION_MAX_PX, "reprojection",
             f"{where}: keypoints differ from the pose's projection by {error:.3g} px")  # fmt: skip
    if mask_required is not None:
        _require(bool(ann.get("mask_file")) == mask_required, "mask_file",
                 f"{where}: mask_file {'missing' if mask_required else 'present'} in a "
                 f"{'sam2' if mask_required else 'cad_only'} dataset")  # fmt: skip
    if ann.get("mask_file"):
        mask = cv2.imread(str(root / split / ann["mask_file"]), cv2.IMREAD_GRAYSCALE)
        shape = None if mask is None else mask.shape
        _require(shape == (image["height"], image["width"]), "mask_shape",
                 f"{where}: mask missing or not {image['height']}x{image['width']}")  # fmt: skip
        area = int((mask > 0).sum()) if mask is not None else -1
        _require(area == ann["area"], "mask_area", f"{where}: area")
    _require(all(z > 0 for z in ann["bbox"][2:]), "bbox_positive", f"{where}: empty bbox")
    return error, int(visible.sum())


def qc_dataset(
    root: str | Path,
    keypoints_sha: str,
    *,
    splits: Mapping[str, int] | None = None,
    min_visible: int | None = None,
    seed_views: Collection[str] | None = None,
) -> dict[str, Any]:
    """QC a packaged dataset.

    Checks: the keypoints file's SHA-256; per split the image and annotation counts and the
    category's keypoint count; per annotation the image id, finite keypoints, ``v`` in
    {0, 2}, at least ``min_visible`` visible, invisible keypoints at (0, 0), ``num_keypoints``,
    an orthonormal rotation with determinant 1, the CAD keypoints reprojected through
    ``T_camera_object_m`` and ``K`` within 1e-6 px of the visible keypoints, a ``mask_file``
    exactly when ``package.json`` records ``mask_mode: sam2``, the mask's shape and ``area``
    (when it has a mask) and a non-empty bbox; then unique image names and image hashes, every
    candidate seed view represented, the source exclusion flags (when recorded; otherwise the
    summary says the exclusion was enforced at sampling) and every file of ``sha256.json``.

    Args:
        root: The dataset directory.
        keypoints_sha: SHA-256 of the expected ``keypoints_3d.json``.
        splits: Expected images per split (default: ``package.json``).
        min_visible: Fewest visible keypoints (default: ``package.json``).
        seed_views: Seed views the selection must cover (default: ``package.json``).

    Returns:
        The QC summary (``status: passed``).

    Raises:
        DatasetQcError: On the first failed check, named.
        FileNotFoundError: Without ``package.json`` and the expectations it gives.
    """
    root = Path(root)
    receipt = (
        json.loads((root / PACKAGE_NAME).read_text()) if (root / PACKAGE_NAME).is_file() else {}
    )
    if not receipt and (splits is None or min_visible is None or seed_views is None):
        raise FileNotFoundError(f"{root / PACKAGE_NAME} is missing: pass splits, "
                                "min_visible and seed_views")  # fmt: skip
    splits = splits or {name: int(receipt[name]) for name in SPLITS}
    minimum = int(receipt["min_visible"] if min_visible is None else min_visible)
    seeds = {
        str(v) for v in (receipt["candidate_seed_views"] if seed_views is None else seed_views)
    }
    mask_mode = receipt.get("mask_mode")
    mask_required = None if mask_mode is None else mask_mode == "sam2"
    hashes = json.loads((root / SHA256_NAME).read_text())
    found = sha256_file(root / KEYPOINTS_NAME)
    _require(found == keypoints_sha, "keypoints_sha256", f"{found} is not {keypoints_sha}")
    points = np.array(json.loads((root / KEYPOINTS_NAME).read_text())["keypoints_3d"],
                      dtype=np.float64)  # fmt: skip
    names, image_hashes, visible, errors = [], [], [], []
    for split, expected in splits.items():
        data = json.loads((root / split / ANNOTATIONS_NAME).read_text())
        _require(len(data["images"]) == len(data["annotations"]) == expected, "split_counts",
                 f"{split}: {len(data['images'])} images, {len(data['annotations'])} "
                 f"annotations, {expected} expected")  # fmt: skip
        _require(len(data["categories"][0]["keypoints"]) == len(points), "keypoint_count",
                 f"{split}: the category names {len(data['categories'][0]['keypoints'])} "
                 f"keypoints, the keypoints file {len(points)}")  # fmt: skip
        for image, ann in zip(data["images"], data["annotations"], strict=True):
            error, count = _qc_annotation(root, split, image, ann, points, minimum, mask_required)
            names.append(image["file_name"])
            image_hashes.append(hashes.get(f"{split}/{IMAGES_DIR}/{image['file_name']}"))
            errors.append(error)
            visible.append(count)
    _require(len(names) == len(set(names)), "unique_names", "image names repeat")
    _require(None not in image_hashes and len(set(image_hashes)) == len(image_hashes),
             "unique_image_hashes", "an image is unlisted or two images are identical")  # fmt: skip
    rows = json.loads((root / POSE_MANIFEST_NAME).read_text())
    hist = collections.Counter(str(row["seed_view"]) for row in rows)
    missing = sorted(seeds - set(hist))
    _require(not missing, "seed_neighbourhoods", f"no selected view of seed views {missing}")
    flagged = any("source_exclusion_pass" in row for row in rows)
    _require(all(row.get("source_exclusion_pass", True) for row in rows), "source_exclusion",
             "a selected view failed the source exclusion")  # fmt: skip
    current = dict(zip(hashes, sha256_files(root / name for name in hashes), strict=True))
    changed = sorted(name for name, digest in hashes.items() if current[name] != digest)
    _require(not changed, "file_sha256", f"files differ from {SHA256_NAME}: {changed[:5]}")
    selection = json.loads((root / SELECTION_NAME).read_text())
    return {
        "status": "passed", "splits": dict(splits), "min_visible": min(visible),
        "median_visible": float(np.median(visible)),
        "max_projection_consistency_error_px": max(errors),
        "unique_names": len(set(names)), "unique_image_hashes": len(set(image_hashes)),
        "source_neighborhood_counts": dict(hist),
        "candidate_count": selection["total_candidates"],
        "accepted_candidate_count": selection["accepted_candidates"],
        "rejected_candidate_count": len(selection["rejected"]),
        "mask_mode": receipt.get("mask_mode"), "lineage_split": receipt.get("lineage_split"),
        "source_exclusion": "checked per view" if flagged else "enforced at sampling (the views "
        "record no exclusion flag; the novel-view sampler rejects views near a source camera)",
        "keypoints_sha256": found,
    }  # fmt: skip


__all__ = [
    "LINEAGE_CAVEAT",
    "PACKAGE_NAME",
    "SHA256_NAME",
    "DatasetQcError",
    "Qualified",
    "assess",
    "assess_files",
    "package_dataset",
    "qc_dataset",
    "qualify",
]
