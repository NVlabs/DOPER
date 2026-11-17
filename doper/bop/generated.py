# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training data of generated objects: a verified BlenderProc campaign as a BOP source.

HANDAL object 40 and HOT3D object 33 have no official PBR. Their bases train on a campaign
rendered by ``rendering/blenderproc/campaign.py`` (checkout only; ``doper`` never imports it),
which publishes layout ``l`` as the BOP scene ``<render root>/train_pbr/<l:06d>/`` with a
``COMPLETE.json`` receipt. This module re-implements the receipt check of that campaign
(``verify_layout``), so ``doper prepare --generated-root`` can refuse a campaign with any
missing, recorded-failed or changed layout before it converts anything:

* ``source_view`` verifies the campaign identity against the recipe's ``GeneratedSpec`` and
  every layout's receipt, then links layout ``l`` as scene ``l:06d`` of a ``train_pbr`` view;
* ``require_same_layout`` binds the keypoints ``doper`` selects to the frozen keypoints the
  campaign rendered and audited with (the same ordered layout);
* ``split_layouts`` keeps the campaign's split: layout ``l`` is validation when
  ``l % val_modulo == val_residue``, whole layouts only, with the images in conversion-shard order;
* ``select_visible`` / ``exclude_invisible`` drop annotations whose authoritative visible
  pixel count is zero, after proving the mask empty (``empty_masks: exclude``); ``require_visible``
  refuses them (``empty_masks: fail``); ``empty_masks: keep`` trains on them unchanged (as for
  HANDAL object 40).

The receipt contract is pinned on both sides by ``tests/unit/generated_render.py``, which the
campaign's own verifier checks in ``tests/rendering/test_blenderproc_campaign.py``.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from .sources import READY_NAME, canonical_sha256, publish_json, sha256_file
from .split import SplitSummary, split_by

if TYPE_CHECKING:
    from ..recipes.schema import GeneratedSpec

RENDER_SCHEMA = 1
IDENTITY_NAME = "identity.json"
STAGED_NAME = "STAGED.json"
CONFIG_NAME = "config.json"
COMPLETE_NAME = "COMPLETE.json"
SPLIT_DIR = "train_pbr"
WORKERS_DIR = "workers"
RECORDS = "records"
KEYPOINTS_PATH = Path("assets") / "geometry" / "keypoints_3d.json"
MODELS_DIR = Path("assets") / "bop"
# Files the fast receipt check re-hashes (``METADATA_FILES`` of the campaign).
METADATA_FILES = (
    "scene_gt.json",
    "scene_camera.json",
    "scene_gt_info.json",
    f"{RECORDS}/AUDITED.json",
    f"{RECORDS}/scene_record.json",
    f"{RECORDS}/render_receipt.json",
    f"{RECORDS}/visible_count_canonicalization.json",
    f"{RECORDS}/writer_raw/scene_gt_info.json",
)
TRAIN_SPLIT, VAL_SPLIT = "train_pbr", "val_pbr"
SOURCE_KIND = "generated_blenderproc_render"
LAYOUT_FIELDS = ("object", "keypoint_names", "indices", "keypoints_3d")
VISIBLE_POLICY = "positive-COCO-requires-nonempty-visible-mask-v1"
VISIBLE_SELECTION_NAME = "VISIBLE_SELECTION.json"
EXCLUDED_REASON = "original_visible_count_zero"


@dataclass(frozen=True)
class Campaign:
    """A staged campaign whose identity matches the recipe.

    Attributes:
        root: The render root.
        identity: ``identity.json``.
        identity_sha256: Canonical SHA-256 of the identity (every receipt carries it).
        keypoints: The frozen keypoints the campaign rendered and audited with.
        models: The staged BOP models of the dataset (the target's CAD in millimeters).
    """

    root: Path
    identity: dict[str, Any]
    identity_sha256: str
    keypoints: Path
    models: Path


def _read(path: Path) -> Any:  # noqa: ANN401
    return json.loads(Path(path).read_text())


def split_for_layout(layout: int, spec: GeneratedSpec) -> str:
    """Return ``val_pbr`` when ``layout % val_modulo == val_residue``, else ``train_pbr``."""
    return VAL_SPLIT if layout % spec.val_modulo == spec.val_residue else TRAIN_SPLIT


def _expected_identity(spec: GeneratedSpec, dataset: str, obj: int) -> dict[str, Any]:
    return {
        "schema": RENDER_SCHEMA,
        "dataset": dataset,
        "target_object": obj,
        "layouts": spec.layouts,
        "frames_per_layout": spec.frames_per_layout,
        "base_seed": spec.base_seed,
        "split": f"layout % {spec.val_modulo} == {spec.val_residue} is {VAL_SPLIT}",
    }


def read_campaign(render_root: Path, spec: GeneratedSpec, *, dataset: str, obj: int) -> Campaign:
    """Read a staged campaign and check its identity against the recipe.

    Args:
        render_root: Campaign root (the ``--out`` of the campaign's ``stage``).
        spec: The object's render campaign in the recipe.
        dataset: Dataset (profile) name.
        obj: The generated object.

    Returns:
        The campaign.

    Raises:
        ValueError: If the campaign is not staged, its config or keypoints differ from its
            identity, or the identity is not the recipe's campaign.
    """
    root = Path(render_root)
    if not (root / STAGED_NAME).is_file():
        raise ValueError(f"{root} is not a staged render campaign (no {STAGED_NAME})")
    identity = _read(root / IDENTITY_NAME)
    digest = canonical_sha256(identity)
    if _read(root / STAGED_NAME).get("identity_sha256") != digest:
        raise ValueError(f"{root}: {STAGED_NAME} does not match {IDENTITY_NAME}")
    config = identity.get("config", {})
    if _read(root / CONFIG_NAME) != config:
        raise ValueError(f"{root}: {CONFIG_NAME} differs from the campaign identity")
    expected = {
        **_expected_identity(spec, dataset, obj),
        "render_config": spec.render_config,
        "config split": {"modulo": spec.val_modulo, "val_residue": spec.val_residue},
    }
    actual = {key: identity.get(key) for key in _expected_identity(spec, dataset, obj)}
    actual.update({"render_config": config.get("name"), "config split": config.get("split")})
    wrong = [f"{k} {actual[k]!r} != {v!r}" for k, v in expected.items() if actual[k] != v]
    if wrong:
        raise ValueError(f"{root} is not the recipe's render campaign: {'; '.join(wrong)}")
    keypoints = root / KEYPOINTS_PATH
    digest_kp = sha256_file(keypoints)
    if not digest_kp == identity.get("keypoints_sha256") == config.get("keypoints_sha256"):
        raise ValueError(f"{keypoints} differs from the campaign's frozen keypoints")
    if spec.keypoints_sha256 is not None and digest_kp != spec.keypoints_sha256:
        raise ValueError(
            f"{root} was staged with keypoints {digest_kp}, not the recipe's pinned selection "
            f"{spec.keypoints_sha256} (`doper prepare --keypoints-only`)"
        )
    return Campaign(root, identity, digest, keypoints, root / MODELS_DIR / dataset / "models")


def _check_files(path: Path, files: Mapping[str, str], names: list[str]) -> None:
    for name in names:
        relative = PurePosixPath(name)
        target = path / name
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{path}: unsafe receipt path {name}")
        if target.is_symlink() or not target.is_file() or sha256_file(target) != files[name]:
            raise ValueError(f"{path}: {name} differs from its receipt")


def verify_layout(
    path: Path, campaign: Campaign, layout: int, spec: GeneratedSpec, *, deep: bool = False
) -> dict[str, Any]:
    """Check one published layout against its ``COMPLETE.json`` receipt.

    The identity fields must match, every listed file must exist, the metadata files must
    re-hash (every file with ``deep``) and ``records/AUDITED.json`` must be ``ok`` with the
    receipt's frame counts: the campaign's own fast check (``verify_layout``).

    Args:
        path: Published layout directory.
        campaign: The verified campaign.
        layout: Layout index.
        spec: The object's render campaign.
        deep: Re-hash every image and mask too.

    Returns:
        The receipt.

    Raises:
        ValueError: If the layout differs from its receipt or the campaign.
    """
    receipt = _read(path / COMPLETE_NAME)
    expected = (
        campaign.identity_sha256,
        layout,
        spec.base_seed + layout,
        spec.frames_per_layout,
        split_for_layout(layout, spec),
    )
    actual = (
        receipt.get("identity_sha256"),
        receipt.get("layout_id"),
        receipt.get("seed"),
        receipt.get("frames"),
        receipt.get("split"),
    )
    if actual != expected:
        raise ValueError(f"{path}: receipt identity {actual} != expected {expected}")
    files = receipt["files"]
    missing = [name for name in files if not (path / name).is_file()]
    if missing:
        raise ValueError(f"{path}: files missing: {missing[:5]}")
    _check_files(path, files, list(files) if deep else [n for n in METADATA_FILES if n in files])
    audited = _read(path / RECORDS / "AUDITED.json")
    if (
        not audited.get("ok")
        or audited.get("frames") != receipt["frames"]
        or audited.get("target_positive_frames") != receipt.get("target_positive_frames")
    ):
        raise ValueError(f"{path}: audit and receipt disagree")
    return receipt


def recorded_failures(render_root: Path) -> list[int]:
    """Layouts recorded failed in a worker status and not published since."""
    failed = set()
    for path in sorted((Path(render_root) / WORKERS_DIR).glob("*.json")):
        for key in _read(path).get("failed", {}):
            if not (Path(render_root) / SPLIT_DIR / f"{int(key):06d}").exists():
                failed.add(int(key))
    return sorted(failed)


def verify_campaign(
    render_root: Path, spec: GeneratedSpec, *, dataset: str, obj: int
) -> tuple[Campaign, dict[int, dict[str, Any]]]:
    """Verify the campaign identity and every layout receipt.

    Returns:
        The campaign and every layout's receipt.

    Raises:
        ValueError: Naming every missing, recorded-failed or changed layout.
    """
    campaign = read_campaign(render_root, spec, dataset=dataset, obj=obj)
    failed = recorded_failures(campaign.root)
    receipts, missing, problems = {}, [], []
    for layout in range(spec.layouts):
        path = campaign.root / SPLIT_DIR / f"{layout:06d}"
        if not (path / COMPLETE_NAME).is_file():
            if layout not in failed:
                missing.append(layout)
            continue
        try:
            receipts[layout] = verify_layout(path, campaign, layout, spec)
        except (ValueError, KeyError, OSError) as err:
            problems.append(f"{layout:06d}: {err}")
    errors = []
    if failed:
        errors.append(
            f"recorded failed layouts {failed} (re-render them with `campaign run --retry-failed`)"
        )
    if missing:
        errors.append(f"missing layouts {missing}")
    if problems:
        errors.append("changed layouts: " + "; ".join(problems))
    if errors:
        raise ValueError(
            f"{campaign.root} is not a complete campaign of {spec.layouts} "
            f"layouts: {'; '.join(errors)}"
        )
    return campaign, receipts


def _view_ready(
    campaign: Campaign, spec: GeneratedSpec, receipts: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    identity = campaign.identity
    return {
        "source_kind": SOURCE_KIND,
        "dataset": identity["dataset"],
        "target_object": identity["target_object"],
        "render_root": str(campaign.root.resolve()),
        "render_identity_sha256": campaign.identity_sha256,
        "render_config": spec.render_config,
        "layouts": spec.layouts,
        "frames_per_layout": spec.frames_per_layout,
        "base_seed": spec.base_seed,
        "split": {"modulo": spec.val_modulo, "val_residue": spec.val_residue},
        "validation_layouts": [i for i in receipts if split_for_layout(i, spec) == VAL_SPLIT],
        "receipts_sha256": canonical_sha256(
            {
                f"{i:06d}": sha256_file(campaign.root / SPLIT_DIR / f"{i:06d}" / COMPLETE_NAME)
                for i in receipts
            }
        ),
        "target_positive_frames": sum(r["target_positive_frames"] for r in receipts.values()),
        "zero_visible_layouts": [i for i, r in receipts.items() if r.get("zero_visible_target")],
        "keypoints_sha256": identity["keypoints_sha256"],
        "annotation_units": "mm",
        "depth_scale_mm_per_uint": 1,
        "official_PBR_source": False,
        "real_or_test_images_used": False,
    }


def source_view(
    render_root: Path, spec: GeneratedSpec, destination: Path, *, dataset: str, obj: int
) -> Path:
    """Verify a campaign and link layout ``l`` as scene ``l:06d`` of a ``train_pbr`` view.

    The view (``<destination>/train_pbr``, with ``<destination>/READY.json``) is built once;
    later calls verify the campaign again and require the same receipts and links.

    Args:
        render_root: Campaign root.
        spec: The object's render campaign.
        destination: View directory (``<root>/views/generated-objXXXXXX``).
        dataset: Dataset (profile) name.
        obj: The generated object.

    Returns:
        The view's ``train_pbr`` directory.

    Raises:
        ValueError: If the campaign is incomplete or changed (``verify_campaign``), or the view
            was built from another render root or campaign.
    """
    campaign, receipts = verify_campaign(render_root, spec, dataset=dataset, obj=obj)
    ready = _view_ready(campaign, spec, receipts)
    destination = Path(destination)
    view = destination / SPLIT_DIR
    if (destination / READY_NAME).is_file():
        if _read(destination / READY_NAME) != ready:
            raise ValueError(
                f"{destination}: render root changed since the view was built; "
                "rebuild the object with `doper prepare --force`"
            )
        for layout in receipts:
            link = view / f"{layout:06d}"
            if link.resolve() != (campaign.root / SPLIT_DIR / f"{layout:06d}").resolve():
                raise ValueError(f"{link}: scene link changed")
        return view
    staging = destination.parent / f".{destination.name}-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    (staging / SPLIT_DIR).mkdir(parents=True)
    for layout in receipts:
        target = (campaign.root / SPLIT_DIR / f"{layout:06d}").resolve()
        (staging / SPLIT_DIR / f"{layout:06d}").symlink_to(target, target_is_directory=True)
    publish_json(staging / READY_NAME, ready)
    if destination.exists():
        shutil.rmtree(destination)
    os.rename(staging, destination)
    return view


def require_same_layout(selected: Path, frozen: Path) -> None:
    """Require the selected keypoints to be the campaign's frozen layout.

    The object name, ordered names, mesh vertex indices and coordinates must be equal; the
    files may differ in their other (report) fields.

    Raises:
        ValueError: Naming the fields that differ.
    """
    mine, theirs = _read(selected), _read(frozen)
    differ = [key for key in LAYOUT_FIELDS if mine.get(key) != theirs.get(key)]
    if differ:
        raise ValueError(
            f"selected keypoints {selected} differ from the campaign's frozen keypoints "
            f"{frozen} in {differ}; the renders were audited with another layout"
        )


def split_layouts(
    converted: Path, dst: Path, obj_id: int, keypoints: Path, spec: GeneratedSpec
) -> SplitSummary:
    """Split a converted campaign by layout: ``l % val_modulo == val_residue`` is validation.

    Each split lists its images in conversion order: shard by shard (shard ``s`` holds the layouts
    ``range(s, layouts, conversion_shards)``), each shard in layout and frame order.

    Args:
        converted: Converted root (``doper.bop.convert.convert_pbr``) of the source view.
        dst: Output root (``train/``, ``val/``, ``keypoints_3d.json``).
        obj_id: Converted object.
        keypoints: ``keypoints_3d.json`` copied to ``dst``.
        spec: The object's render campaign.

    Returns:
        The split summary.
    """

    def is_validation(scene: str) -> bool:
        return split_for_layout(int(scene), spec) == VAL_SPLIT

    def shard_major(scene: str) -> tuple[int, int]:
        return int(scene) % spec.conversion_shards, int(scene)

    return split_by(converted, dst, obj_id, keypoints, is_validation, order=shard_major)


# --- empty visible masks --------------------------------------------------------------------


Lookup = Callable[[Mapping[str, Any]], tuple[Mapping[str, Any], Mapping[str, Any]]]
ZeroProof = Callable[[Mapping[str, Any], Mapping[str, Any]], dict[str, Any]]


def scene_lookup(train_pbr: Path) -> Lookup:
    """Return ``lookup(bop) -> (scene_gt row, scene_gt_info row)`` over a ``train_pbr`` root."""
    cache: dict[str, tuple[Any, Any]] = {}

    def lookup(ref: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        scene = ref["scene"]
        if scene not in cache:
            folder = Path(train_pbr) / scene
            cache[scene] = (_read(folder / "scene_gt.json"), _read(folder / "scene_gt_info.json"))
        gt, info = cache[scene]
        return gt[str(ref["frame"])][ref["instance"]], info[str(ref["frame"])][ref["instance"]]

    return lookup


def zero_mask_proof(
    converted: Path, train_pbr: Path, ann: Mapping[str, Any], image: Mapping[str, Any]
) -> dict[str, Any]:
    """Prove an annotation's visible mask empty: source and converted masks equal, all zero.

    Raises:
        ValueError: If either mask is missing, they differ, or the mask is not an all-zero
            ``uint8`` image of the image's size.
    """
    import cv2

    ref = ann["bop"]
    raw = (
        Path(train_pbr)
        / ref["scene"]
        / "mask_visib"
        / f"{ref['frame']:06d}_{ref['instance']:06d}.png"
    )
    copy = Path(converted) / ann["mask_file"]
    if not raw.is_file() or not copy.is_file():
        raise ValueError(f"missing mask must not be treated as invisibility: {raw}")
    if sha256_file(raw) != sha256_file(copy):
        raise ValueError(f"raw/converted mask mismatch: {copy}")
    mask = cv2.imread(str(copy), cv2.IMREAD_UNCHANGED)
    if (
        mask is None
        or mask.dtype.name != "uint8"
        or mask.shape != (image["height"], image["width"])
        or mask.any()
    ):
        raise ValueError(f"an invisible target requires a correct-size uint8 zero mask: {copy}")
    return {
        "raw_sha256": sha256_file(raw),
        "converted_sha256": sha256_file(copy),
        "width": image["width"],
        "height": image["height"],
        "nonzero_pixels": 0,
    }


def select_visible(
    data: Mapping[str, Any], lookup: Lookup, verify_zero: ZeroProof, obj_id: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Drop the annotations whose authoritative visible pixel count is zero.

    Retained records and IDs are kept verbatim; an image is kept while it has a retained
    annotation. Every dropped annotation needs a proof that its mask is empty (``verify_zero``).

    Args:
        data: Converted COCO annotations.
        lookup: ``bop`` reference to its ``scene_gt`` and ``scene_gt_info`` rows.
        verify_zero: ``(annotation, image)`` to the proof of an empty mask (or raises).
        obj_id: Converted object.

    Returns:
        The filtered annotations and the exclusion records.

    Raises:
        ValueError: For duplicate IDs, an unknown image, a source row of another object or an
            invalid visible count, or when nothing visible remains.
    """
    images = {image["id"]: image for image in data["images"]}
    ids = {a["id"] for a in data["annotations"]}
    if len(images) != len(data["images"]) or len(ids) != len(data["annotations"]):
        raise ValueError("duplicate COCO IDs")
    kept, excluded = [], []
    for ann in data["annotations"]:
        if ann["image_id"] not in images:
            raise ValueError("unknown image")
        gt, info = lookup(ann["bop"])
        count = info["px_count_visib"]
        if (
            gt["obj_id"] != obj_id
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise ValueError(f"invalid authoritative visibility of annotation {ann['id']}")
        if count:
            kept.append(ann)
            continue
        proof = verify_zero(ann, images[ann["image_id"]])
        excluded.append(
            {
                "annotation_id": ann["id"],
                "image_id": ann["image_id"],
                "bop": ann["bop"],
                "reason": EXCLUDED_REASON,
                "mask_proof": proof,
            }
        )
    used = {a["image_id"] for a in kept}
    selected = [image for image in data["images"] if image["id"] in used]
    if not kept or not selected:
        raise ValueError("empty positive dataset")
    info = {
        **data.get("info", {}),
        "description": f"BOP obj{obj_id} positive-visible selection: {len(selected)} images",
        "num_samples": len(selected),
        "conversion_provenance": VISIBLE_SELECTION_NAME,
        "parent_conversion_provenance": "original_conversion_provenance.json",
        "selection_policy": VISIBLE_POLICY,
    }
    if all("keypoints" in a for a in kept):
        flags = [v for a in kept for v in a["keypoints"][2::3]]
        info["keypoint_visibility"] = {
            "visible": sum(v == 2 for v in flags),
            "occluded": sum(v == 1 for v in flags),
        }
    return {**data, "info": info, "images": selected, "annotations": kept}, excluded


def _zero_rows(data: Mapping[str, Any], lookup: Lookup) -> Iterator[Mapping[str, Any]]:
    for ann in data["annotations"]:
        if lookup(ann["bop"])[1]["px_count_visib"] == 0:
            yield ann


def require_visible(converted: Path, train_pbr: Path, obj_id: int) -> None:
    """Refuse converted annotations whose source visible mask is empty (``empty_masks: fail``).

    Raises:
        ValueError: Naming the annotations with an empty visible mask.
    """
    data = _read(Path(converted) / "annotations.json")
    lookup = scene_lookup(train_pbr)
    empty = [a["bop"] for a in _zero_rows(data, lookup)]
    if empty:
        raise ValueError(
            f"{len(empty)} converted annotations of object {obj_id} have an empty visible mask "
            f"(first: {empty[:3]}); the recipe's empty_masks policy is fail"
        )


def exclude_invisible(converted: Path, train_pbr: Path, out: Path, obj_id: int) -> dict[str, Any]:
    """Write the positive-visible view of a converted root (``empty_masks: exclude``).

    ``out`` links the parent's ``images``, ``masks`` and ``keypoints_3d.json``, links its
    conversion provenance as ``original_conversion_provenance.json``, and holds the filtered
    ``annotations.json`` and ``VISIBLE_SELECTION.json`` (the exclusions with their proofs).

    Returns:
        The selection summary (``VISIBLE_SELECTION.json`` without the exclusion records).
    """
    converted, out = Path(converted), Path(out)
    data = _read(converted / "annotations.json")
    filtered, excluded = select_visible(
        data,
        scene_lookup(train_pbr),
        lambda ann, image: zero_mask_proof(converted, train_pbr, ann, image),
        obj_id,
    )
    staging = out.parent / f".{out.name}-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    for name in ("images", "masks", "keypoints_3d.json"):
        (staging / name).symlink_to(os.path.relpath(converted / name, staging))
    (staging / "original_conversion_provenance.json").symlink_to(
        os.path.relpath(converted / "conversion_provenance.json", staging)
    )
    (staging / "annotations.json").write_text(json.dumps(filtered) + "\n")
    summary = {
        "policy": VISIBLE_POLICY,
        "parent_annotations_sha256": sha256_file(converted / "annotations.json"),
        "parent_conversion_provenance_sha256": sha256_file(
            converted / "conversion_provenance.json"
        ),
        "input_images": len(data["images"]),
        "output_images": len(filtered["images"]),
        "input_annotations": len(data["annotations"]),
        "output_annotations": len(filtered["annotations"]),
        "excluded_annotations": len(excluded),
        "excluded_images": len(data["images"]) - len(filtered["images"]),
        "retained_records_unmodified": True,
        "raw_renders_unchanged": True,
        "test_or_real_validation_used_for_selection": False,
    }
    publish_json(staging / VISIBLE_SELECTION_NAME, {**summary, "excluded": excluded})
    if out.exists():
        shutil.rmtree(out)
    os.rename(staging, out)
    return summary
