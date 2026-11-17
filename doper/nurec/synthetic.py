# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The synthetic baseline dataset of a NuRec recipe: render config, audit, split.

**Render config** (``render_config``). ``rendering/nurec_render.yaml`` holds the
render's settings (640 px, 64 keypoints, masks, 50 mm lens, texture ground, HDRI world, 0-8
distractors changed every image). Its ``hdri_dir``, ``texture_dir`` and ``distractor_dir`` are
empty: they are required user inputs, and the resolved copy the render reads fills them, the
object ``usd_path`` and ``keypoints_json`` with absolute paths.

**Audit** (``audit_render``). Before a render is used, every sample is checked: the merged
``manifest.json`` is complete, without problems, of ``count`` samples and ``seed``, of the
expected config and object (when given); the records pass
``rendering.dataset_contract.validate_dataset_records`` (64 keypoints, masks); one annotation
per image; every image and mask decodes at ``resolution`` x ``resolution``; every mask has a
foreground pixel (value >= 128); every keypoint is finite with visibility in {0, 1, 2}.

**Split** (``split_synthetic``). Images are sorted by the SHA-256 of the image id, the first
``train`` go to training and the next ``val`` to validation. The split is deterministic and
independent of the record order.

**Package** (``package_synthetic``). Each split is a COCO directory the DOPER trainer reads
(``images/``, ``masks/``, ``annotations.json``), the files hard-linked (copied across file
systems) from the render, beside the render's ``keypoints_3d.json``, ``sha256.json`` and the
receipt ``package.json``.

``rendering`` is a checkout-only tree; the audit imports its dataset contract from the
checkout that holds this package.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from ..bop.sources import publish_json, sha256_file, sha256_files
from ..keypoints import KEYPOINTS_NAME
from ._files import link_or_copy

# The synthetic render: count, seed, resolution and keypoints.
RENDER_COUNT = 10000
RENDER_SEED = 142
RENDER_RESOLUTION = 640
NUM_KEYPOINTS = 64
# The asset directories the render config leaves to the user.
ASSET_KEYS = ("hdri_dir", "texture_dir", "distractor_dir")
# Characters a path may not hold: the Blender-side fallback YAML reader cuts at ``#`` and
# does not unescape quotes.
UNSAFE_PATH_CHARACTERS = ("#", '"', "'", "\n", "\r")
MASK_FOREGROUND = 128
VISIBILITY_VALUES = (0, 1, 2)
VISIBLE = 2
# Problems listed in an audit result (the count is always complete).
MAX_LISTED_PROBLEMS = 100
CHECKOUT = Path(__file__).resolve().parents[2]
MANIFEST_NAME = "manifest.json"
ANNOTATIONS_NAME = "annotations.json"
IMAGES_DIR = "images"
SHA256_NAME = "sha256.json"
PACKAGE_NAME = "package.json"
SPLIT_RULE = "sha256(str(image_id)) ascending: first train, next val"
# ``render_identity.object`` keys of the merged manifest.
OBJECT_HASH_KEYS = ("blend_sha256", "usd_sha256", "keypoints_sha256")


def checkout_file(relative: str) -> Path:
    """A file of the DOPER checkout this package was imported from.

    Raises:
        FileNotFoundError: When the package is not imported from a checkout that has it.
    """
    path = CHECKOUT / relative
    if not path.is_file():
        raise FileNotFoundError(
            f"{relative} is missing: the synthetic stage runs from a DOPER "
            "source checkout (rendering/ is not part of the package)"
        )
    return path


def _flow_lists_dumper() -> type:
    import yaml

    class FlowListDumper(yaml.SafeDumper):
        """Block mappings with flow-style lists (Blender's fallback YAML reader)."""

    FlowListDumper.add_representer(
        list, lambda dumper, data: dumper.represent_sequence("tag:yaml.org,2002:seq", data, True)
    )
    return FlowListDumper


def _safe_path(path: str | os.PathLike[str], what: str) -> str:
    text = str(Path(path).resolve())
    if any(character in text for character in UNSAFE_PATH_CHARACTERS):
        raise ValueError(f"{what} {text!r} holds one of {UNSAFE_PATH_CHARACTERS!r}; move or "
                         "link it to a plain path")  # fmt: skip
    return text


def render_config(
    template: str | os.PathLike[str],
    out: str | os.PathLike[str],
    *,
    asset_dirs: Mapping[str, str | os.PathLike[str] | None],
    usd_path: str | os.PathLike[str],
    keypoints_json: str | os.PathLike[str],
) -> dict[str, Any]:
    """Write the resolved render config: the template with the asset and object paths.

    Args:
        template: The render config with empty asset directories
            (``rendering/nurec_render.yaml``).
        out: Resolved config path.
        asset_dirs: ``hdri_dir``, ``texture_dir`` and ``distractor_dir``: existing, nonempty
            directories (required user inputs).
        usd_path: The object ``.usda``.
        keypoints_json: The object's ``keypoints_3d.json``.

    Returns:
        The written mapping.

    Raises:
        ValueError: An asset directory is missing, not a nonempty directory or an unsafe path,
            or the template sets an asset directory itself.
    """
    import yaml

    config = yaml.safe_load(Path(template).read_text())
    scene = config.setdefault("scene", {})
    preset = [key for key in ASSET_KEYS if scene.get(key)]
    if preset:
        raise ValueError(f"{template} sets {preset}: the asset directories are user inputs")
    missing = [key for key in ASSET_KEYS if not asset_dirs.get(key)]
    if missing:
        flags = ", ".join("--" + key.replace("_", "-") for key in missing)
        raise ValueError(f"the synthetic render needs the asset directories {flags} (e.g. CC0 "
                         "HDRIs, PBR texture sets and distractor meshes)")  # fmt: skip
    for key in ASSET_KEYS:
        directory = Path(str(asset_dirs[key]))
        if not directory.is_dir() or not any(directory.iterdir()):
            raise ValueError(f"--{key.replace('_', '-')} {directory} is not a nonempty directory")
        scene[key] = _safe_path(directory, f"--{key.replace('_', '-')}")
    config.setdefault("object", {}).update(
        usd_path=_safe_path(usd_path, "the object USD"),
        keypoints_json=_safe_path(keypoints_json, "the object keypoints"),
    )
    text = yaml.dump(config, Dumper=_flow_lists_dumper(), sort_keys=False, width=1 << 16)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text)
    return config


def split_synthetic(
    records: Sequence[Mapping[str, Any]],
    *,
    train: int = 9750,
    val: int = 250,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Split image records by the SHA-256 of their id.

    Args:
        records: COCO image records (``id``), exactly ``train + val`` of them.
        train: Training images.
        val: Validation images.

    Returns:
        ``(train_records, val_records)``, each in digest order.

    Raises:
        ValueError: For another record count or a repeated id.
    """
    if len(records) != train + val:
        raise ValueError(f"{len(records)} images cannot be split {train}/{val}")
    ids = [record["id"] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("the image ids repeat")
    ordered = sorted(records, key=lambda r: hashlib.sha256(str(r["id"]).encode()).hexdigest())
    return ordered[:train], ordered[train:]


def _dataset_contract() -> ModuleType:
    try:
        from rendering import dataset_contract
    except ImportError:
        checkout_file("rendering/dataset_contract.py")
        sys.path.insert(0, str(CHECKOUT))
        from rendering import dataset_contract
    return dataset_contract


def _manifest_problems(
    manifest: Mapping[str, Any],
    count: int,
    seed: int,
    config_sha256: str | None,
    object_hashes: Mapping[str, str] | None,
) -> list[str]:
    problems = []
    if manifest.get("status") != "complete" or manifest.get("problems"):
        problems.append(f"manifest status {manifest.get('status')!r} with problems "
                        f"{manifest.get('problems')}")  # fmt: skip
    if not manifest.get("num_images") == manifest.get("num_samples") == count:
        problems.append(f"manifest has {manifest.get('num_images')} images / "
                        f"{manifest.get('num_samples')} samples, not {count}")  # fmt: skip
    if manifest.get("seed") != seed:
        problems.append(f"manifest seed {manifest.get('seed')} is not {seed}")
    if config_sha256 is not None and manifest.get("config_sha256") != config_sha256:
        problems.append("the render config differs from the planned one")
    recorded = (manifest.get("render_identity") or {}).get("object") or {}
    expected = object_hashes or {}
    for key in OBJECT_HASH_KEYS if object_hashes is not None else ():
        if recorded.get(key) != expected[key]:
            problems.append(f"render_identity.object.{key} differs from the authored object")
    return problems


def _keypoint_problems(root: Path, keypoints: Path | None) -> list[str]:
    if keypoints is None:
        return []
    path = root / KEYPOINTS_NAME
    if not path.is_file():
        return [f"{KEYPOINTS_NAME} is missing"]
    actual, expected = json.loads(path.read_text()), json.loads(keypoints.read_text())
    same = actual.get("keypoint_names") == expected["keypoint_names"] and np.allclose(
        actual.get("keypoints_3d"), expected["keypoints_3d"], rtol=0, atol=1e-6)  # fmt: skip
    return [] if same else [f"{KEYPOINTS_NAME} is not the authored layout"]


def _sample_problems(
    root: Path, image: Mapping[str, Any], ann: Mapping[str, Any], resolution: int,
    num_keypoints: int,
) -> tuple[list[str], int, int]:  # fmt: skip
    import cv2

    label = f"image {image.get('id')}"
    rgb = cv2.imread(str(root / IMAGES_DIR / str(image.get("file_name"))), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(root / str(ann.get("mask_file"))), cv2.IMREAD_GRAYSCALE)
    problems = []
    decoded = int(rgb is not None) + int(mask is not None)
    size = (resolution, resolution)
    if rgb is None or rgb.shape[:2] != size:
        problems.append(f"{label}: RGB is not a decodable {resolution}x{resolution} image")
    if mask is None or mask.shape != size or mask.dtype != np.uint8:
        problems.append(f"{label}: mask is not a decodable {resolution}x{resolution} 8-bit image")
    elif not (mask >= MASK_FOREGROUND).any():
        problems.append(f"{label}: mask has no foreground pixel (>= {MASK_FOREGROUND})")
    try:
        keypoints = np.asarray(ann.get("keypoints"), dtype=np.float64).reshape(num_keypoints, 3)
    except (TypeError, ValueError):
        return [*problems, f"{label}: keypoints are not {num_keypoints} (x, y, v)"], 0, decoded
    if not np.isfinite(keypoints).all() or not np.isin(keypoints[:, 2], VISIBILITY_VALUES).all():
        problems.append(f"{label}: keypoints not finite or visibility not in {VISIBILITY_VALUES}")
    return problems, int((keypoints[:, 2] == VISIBLE).sum()), decoded


def audit_render(
    root: str | os.PathLike[str],
    *,
    count: int = RENDER_COUNT,
    seed: int = RENDER_SEED,
    resolution: int = RENDER_RESOLUTION,
    num_keypoints: int = NUM_KEYPOINTS,
    config_sha256: str | None = None,
    object_hashes: Mapping[str, str] | None = None,
    keypoints: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Audit every sample of a merged render (``rendering/synth_parallel.py`` output).

    Args:
        root: Render output directory (``manifest.json``, ``annotations.json``, ``images/``,
            masks).
        count: Expected samples (one image each).
        seed: Expected render seed.
        resolution: Expected square image side in pixels.
        num_keypoints: Keypoints per annotation.
        config_sha256: Expected SHA-256 of the render config, when known.
        object_hashes: Expected ``blend_sha256``, ``usd_sha256`` and ``keypoints_sha256`` of
            the object, when known.
        keypoints: The authored ``keypoints_3d.json`` the render's copy must match.

    Returns:
        ``status`` (``passed`` or ``failed``), ``problem_count``, the first
        ``MAX_LISTED_PROBLEMS`` ``problems``, the image and decoded media counts, the minimum
        and median number of visible keypoints and ``data_sha256`` (every image, mask and
        dataset file).
    """
    root = Path(root)
    problems: list[str] = []
    manifest_path, annotations_path = root / MANIFEST_NAME, root / ANNOTATIONS_NAME
    if not manifest_path.is_file() or not annotations_path.is_file():
        problems.append(f"{MANIFEST_NAME} or {ANNOTATIONS_NAME} is missing: the render did not "
                        "finish and merge")  # fmt: skip
        return _audit_result(problems, 0, [], 0, {}, count, seed, resolution)
    manifest = json.loads(manifest_path.read_text())
    problems += _manifest_problems(manifest, count, seed, config_sha256, object_hashes)
    problems += _keypoint_problems(root, Path(keypoints) if keypoints is not None else None)
    payload = json.loads(annotations_path.read_text())
    problems += _dataset_contract().validate_dataset_records(
        payload, root, start_idx=0, end_idx=count, images_per_sample=1, render_masks=True,
        num_keypoints=num_keypoints)  # fmt: skip
    images, annotations = payload.get("images") or [], payload.get("annotations") or []
    by_image = {ann.get("image_id"): ann for ann in annotations}
    if len(by_image) != len(annotations) or len(by_image) != count or len(images) != count:
        problems.append(f"{len(images)} images and {len(annotations)} annotations are not one "
                        f"annotation for each of {count} images")  # fmt: skip
    visible, hashed, decoded = [], [], 0
    for image in images:
        ann = by_image.get(image.get("id"))
        if ann is None:
            continue
        found, count_visible, media = _sample_problems(root, image, ann, resolution, num_keypoints)
        decoded += media
        problems += found
        visible.append(count_visible)
        for path in (
            root / IMAGES_DIR / str(image.get("file_name")),
            root / str(ann.get("mask_file")),
        ):
            if path.is_file():
                hashed.append(path)
    hashed += [root / name for name in (ANNOTATIONS_NAME, KEYPOINTS_NAME, MANIFEST_NAME)
               if (root / name).is_file()]  # fmt: skip
    names = (path.relative_to(root).as_posix() for path in hashed)
    hashes = dict(zip(names, sha256_files(hashed), strict=True))
    return _audit_result(problems, len(images), visible, decoded, hashes, count, seed, resolution)


def _audit_result(problems: list[str], images: int, visible: list[int], decoded: int,
                  hashes: dict[str, str], count: int, seed: int,
                  resolution: int) -> dict[str, Any]:  # fmt: skip
    return {
        "status": "failed" if problems else "passed",
        "problem_count": len(problems),
        "problems": problems[:MAX_LISTED_PROBLEMS],
        "count": count, "seed": seed, "resolution": resolution,
        "images": images, "decoded_media": decoded,
        "visible_keypoints_min": min(visible) if visible else None,
        "visible_keypoints_median": float(np.median(visible)) if visible else None,
        "data_sha256": hashes,
    }  # fmt: skip


def _write_split(render: Path, out: Path, records: Sequence[Mapping[str, Any]],
                 by_image: Mapping[Any, Mapping[str, Any]], categories: Any) -> None:  # noqa: ANN401  # fmt: skip
    annotations = [by_image[record["id"]] for record in records]
    for record, ann in zip(records, annotations, strict=True):
        name = str(record["file_name"])
        link_or_copy(render / IMAGES_DIR / name, out / IMAGES_DIR / name)
        link_or_copy(render / str(ann["mask_file"]), out / str(ann["mask_file"]))
    publish_json(out / ANNOTATIONS_NAME, {"images": list(records), "annotations": annotations,
                                          "categories": categories})  # fmt: skip


def package_synthetic(
    render: str | os.PathLike[str],
    out: str | os.PathLike[str],
    *,
    train: int = 9750,
    val: int = 250,
    audit_sha256: str | None = None,
) -> dict[str, Any]:
    """Split an audited render into the ``train``/``val`` COCO dataset the trainer reads.

    Args:
        render: Audited render directory.
        out: New or empty dataset directory.
        train: Training images.
        val: Validation images.
        audit_sha256: SHA-256 of the audit record, kept in the receipt.

    Returns:
        The receipt written to ``out/package.json``.

    Raises:
        ValueError: ``out`` is not empty, or the render cannot be split.
    """
    render, out = Path(render), Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty")
    payload = json.loads((render / ANNOTATIONS_NAME).read_text())
    train_records, val_records = split_synthetic(payload["images"], train=train, val=val)
    by_image = {ann["image_id"]: ann for ann in payload["annotations"]}
    out.mkdir(parents=True, exist_ok=True)
    for split, records in (("train", train_records), ("val", val_records)):
        _write_split(render, out / split, records, by_image, payload["categories"])
    shutil.copyfile(render / KEYPOINTS_NAME, out / KEYPOINTS_NAME)
    files = [p for p in sorted(out.rglob("*")) if p.is_file()]
    hashes = dict(zip((p.relative_to(out).as_posix() for p in files), sha256_files(files),
                      strict=True))  # fmt: skip
    publish_json(out / SHA256_NAME, hashes)
    receipt = {
        "split_rule": SPLIT_RULE, "train": len(train_records),
        "val": len(val_records), "keypoints_sha256": hashes[KEYPOINTS_NAME],
        "annotations_sha256": sha256_file(render / ANNOTATIONS_NAME),
        "audit_sha256": audit_sha256, "sha256_manifest": sha256_file(out / SHA256_NAME),
    }  # fmt: skip
    publish_json(out / PACKAGE_NAME, receipt)
    return receipt
