# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-dataset training inputs of a recipe: the verified PBR source and CAD of one object.

``dataset_inputs`` resolves the per-dataset training source. It keeps each dataset's source
identity separate and fails closed:

* HANDAL trains from a filtered view without the documented zero-byte frames
  (``sources.clean_source_view``), object-scoped when the object has a missing mask;
* HOPEv2 and T-LESS train from the extracted ``train_pbr`` with no exclusions;
* HOT3D trains from the canonical-millimeter view, through the pinned depth-defect filter for
  affected objects and the object-scoped mask filter for objects with mask defects.

Every source must carry the extraction receipt of this profile with PBR frames of the object.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..bop import sources
from ..bop.extract import data_dir
from ..bop.sources import DatasetProfile, sha256_file
from .schema import Recipe

VIEWS_DIR = "views"
CANONICAL_VIEW = "canonical"
DEPTH_VIEW = "canonical-depthfix"
GENERATED_VIEW = "generated-obj{obj:06d}"


@dataclass(frozen=True)
class TrainingInputs:
    """The verified PBR source of one object.

    Attributes:
        train_pbr: BOP ``train_pbr`` directory (or filtered view) the converter reads.
        models: CAD directory with ``obj_XXXXXX.ply`` (mm) and ``models_info.json``.
        source_ready_sha256: SHA-256 of the source's ``READY.json`` receipt (None for the
            HANDAL view, whose identity file records its source instead).
    """

    train_pbr: Path
    models: Path
    source_ready_sha256: str | None


def views_dir(root: Path) -> Path:
    """Return the directory of the filtered source views, ``<root>/views``."""
    return Path(root) / VIEWS_DIR


def hot3d_view_names(profile: DatasetProfile, obj: int) -> list[str]:
    """Names of the HOT3D views an object's source is built through, parent first."""
    names = [CANONICAL_VIEW]
    if obj in sources.hot3d_affected_objects(profile):
        names.append(DEPTH_VIEW)
    if sources.hot3d_mask_records(profile, obj):
        generation = sources.hot3d_mask_generation(profile, obj)
        kind = "maskfix" if generation == 1 else f"maskfix-v{generation}"
        names.append(f"canonical-{kind}-obj{obj:06d}")
    return names


def generated_view(root: Path, obj: int) -> Path:
    """The source view of a generated object, ``<root>/views/generated-objXXXXXX``."""
    return views_dir(root) / GENERATED_VIEW.format(obj=obj)


def planned_source(recipe: Recipe, profile: DatasetProfile, root: Path, obj: int) -> Path:
    """The ``train_pbr`` directory ``dataset_inputs`` will return, without touching data."""
    if obj in recipe.generated_objects:
        return generated_view(root, obj) / "train_pbr"
    if recipe.convert.source_view == "exclusions":
        rows = profile.defects.get("missing_required_files", [])
        missing = [r for r in rows if r["obj_id"] == obj]
        return views_dir(root) / (f"source_view_obj{obj:06d}" if missing else "source_view")
    if recipe.convert.source_view == "hot3d":
        return views_dir(root) / hot3d_view_names(profile, obj)[-1] / "data" / "train_pbr"
    return data_dir(root) / "train_pbr"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _require_coverage(ready: dict[str, Any], obj: int, name: str) -> None:
    if ready["coverage"]["object_frame_counts"].get(str(obj), 0) <= 0:
        raise ValueError(f"{name} source has no PBR frames of object {obj}")


def _extracted_inputs(recipe: Recipe, profile: DatasetProfile, root: Path, obj: int) -> Path:
    """HOPEv2 / T-LESS: the extracted PBR with no exclusions."""
    data = data_dir(root)
    ready = _read(data / sources.READY_NAME)
    if (
        ready.get("dataset") != profile.name
        or ready["identity"]["profile_sha256"] != sources.profile_sha256(profile)
        or ready["model_units"] != "mm"
        or ready["pose_translation_units"] != "mm"
        or ready["test_used_for_training"]
    ):
        raise ValueError(f"{recipe.name} source identity/units mismatch")
    _require_coverage(ready, obj, recipe.name)
    exclusions = _read(data / "train_pbr" / "exclusions.json")
    if (
        exclusions.get("dataset") != profile.name
        or exclusions["excluded_frames"]
        or exclusions["missing_required_files"]
    ):
        raise ValueError(f"{recipe.name} source defects require an explicit separate policy")
    return data / "train_pbr"


def _handal_inputs(profile: DatasetProfile, root: Path, obj: int) -> Path:
    """HANDAL: the view without the documented defective frames."""
    data = data_dir(root)
    ready = _read(data / sources.READY_NAME)
    if ready["identity"]["profile_sha256"] != sources.profile_sha256(profile):
        raise ValueError("HANDAL source identity mismatch")
    _require_coverage(ready, obj, "HANDAL")
    return sources.clean_source_view(data / "train_pbr", views_dir(root), profile, obj)


def hot3d_views(profile: DatasetProfile, root: Path, obj: int) -> Path:
    """Build (or reuse) the HOT3D views of one object and return the last one's ``data``."""
    from ..bop import hot3d_source

    parent = Path(root)
    for name in hot3d_view_names(profile, obj):
        destination = views_dir(root) / name
        if name == CANONICAL_VIEW:
            hot3d_source.canonicalize_units(parent, destination, profile)
        elif name == DEPTH_VIEW:
            hot3d_source.filter_source_defects(parent, destination, profile)
        else:
            hot3d_source.filter_object_masks(parent, destination, obj, profile)
        parent = destination
    return parent / "data"


def _hot3d_inputs(profile: DatasetProfile, root: Path, obj: int) -> Path:
    """HOT3D: the canonical-millimeter view with the object's pinned defect filters."""
    view = hot3d_views(profile, root, obj)
    ready = _read(view / sources.READY_NAME)
    units = ready.get("PBR_units", {})
    if (
        ready.get("dataset") != "hot3d"
        or ready["model_units"]["canonical_metadata_length_units"] != "millimetres"
        or units.get("canonical_pose_translation_units") != "millimetres"
        or units.get("depth_scale_multiplier") != 1
        or ready.get("legacy_renders_or_weights_used")
    ):
        raise ValueError("HOT3D source identity/coverage/unit mismatch")
    _require_coverage(ready, obj, "HOT3D")
    exclusions = _read(view / "train_pbr" / "exclusions.json")
    sources.validate_hot3d_exclusions(profile, exclusions, ready, obj)
    return view / "train_pbr"


def generated_models(render_root: Path, profile: DatasetProfile) -> Path:
    """The BOP models a campaign staged (``<render root>/assets/bop/<dataset>/models``)."""
    from ..bop import generated

    return Path(render_root) / generated.MODELS_DIR / profile.name / "models"


def generated_inputs(
    recipe: Recipe,
    profile: DatasetProfile,
    root: Path,
    obj: int,
    render_root: Path,
    *,
    force: bool = False,
) -> TrainingInputs:
    """Verify a generated object's render campaign and return its source view and CAD.

    Args:
        recipe: Dataset recipe; ``obj`` must have a render campaign (``generated``).
        profile: The recipe's profile.
        root: Dataset root (the view goes to ``<root>/views/generated-objXXXXXX``).
        obj: The generated object.
        render_root: The campaign root.
        force: Rebuild the view (after the render root moved or changed).

    Returns:
        The training inputs: the view's ``train_pbr``, the staged CAD and the view receipt.

    Raises:
        ValueError: If the object has no render campaign, or the campaign is incomplete, failed
            or differs from the recipe (``doper.bop.generated.source_view``).
    """
    from ..bop import generated

    spec = recipe.generated_spec(obj)
    view = generated_view(root, obj)
    if force and view.exists():
        shutil.rmtree(view)
    train_pbr = generated.source_view(render_root, spec, view, dataset=profile.name, obj=obj)
    return TrainingInputs(
        train_pbr=train_pbr,
        models=generated_models(render_root, profile),
        source_ready_sha256=sha256_file(view / sources.READY_NAME),
    )


def dataset_inputs(recipe: Recipe, profile: DatasetProfile, root: Path, obj: int) -> TrainingInputs:
    """Resolve and verify the PBR source and CAD one object trains from.

    Args:
        recipe: Dataset recipe.
        profile: The recipe's profile.
        root: Dataset root (extracted under ``data/``).
        obj: Object id; must be a trainable recipe object.

    Returns:
        The training inputs; filtered views are built on first use.

    Raises:
        ValueError: If the object is not trainable, a source receipt is incomplete, or the
            source identity, units, coverage or exclusions do not match the recipe's dataset
            policy.
    """
    if obj not in recipe.trainable_objects:
        raise ValueError(f"object {obj} is not a trainable {recipe.name} object")
    view = recipe.convert.source_view
    try:
        if view == "exclusions":
            train_pbr = _handal_inputs(profile, root, obj)
        elif view == "hot3d":
            train_pbr = _hot3d_inputs(profile, root, obj)
        else:
            train_pbr = _extracted_inputs(recipe, profile, root, obj)
    except KeyError as err:
        raise ValueError(
            f"{recipe.name} source receipt under {root} lacks {err}; it was not written by "
            "this release, so rerun `doper prepare` on a fresh root"
        ) from None
    models = data_dir(root) / recipe.convert.models
    if view == "hot3d":
        models = train_pbr.parent / recipe.convert.models
    ready = train_pbr.parent / sources.READY_NAME
    return TrainingInputs(
        train_pbr=train_pbr,
        models=models,
        source_ready_sha256=sha256_file(ready) if ready.is_file() else None,
    )
