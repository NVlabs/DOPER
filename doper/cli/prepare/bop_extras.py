# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BOP recipes of ``doper prepare``: dataset, generated-object and multisensor steps.

A generated object (no official PBR: HANDAL 40, HOT3D 33) first gets its frozen keypoints with
``--obj-id OBJ --keypoints-only``: the official CAD is read from the verified models archive,
the keypoints are selected as for any object and checked against the recipe's pinned SHA-256,
and the render campaign is staged with that file. It is then prepared from the campaign with
``--obj-id OBJ --generated-root DIR``: every layout receipt is verified
(``doper.bop.generated.source_view``), the selected keypoints must be the campaign's frozen
layout, annotations with an empty visible mask are kept, excluded or refused per the recipe, and the
split keeps the campaign's layout rule. The prepared object then trains like any other.

T-LESS objects also get three-sensor real TRAIN supervision with ``--multisensor`` (after their
base preparation and ``--parts test --allow-test-inputs``, which extracts the reserved
``test_primesense`` split): the PrimeSense TRAIN part is fetched and extracted, each object's
original Kinect and Canon TRAIN archives are fetched (``doper.bop.tless_sensors``), and
``doper.bop.multisensor.prepare`` writes ``<root>/multisensor/obj_XXXXXX`` from them with the
object's audited CAD and keypoints, refusing any image of the reserved test split. The root then
needs ``doper review tless --obj-id OBJ`` before the multisensor adaptation trains on it.

The ``static`` and ``dynamic`` parts are not extracted: ``doper.bop.onboarding`` stages them
(with the static GT view) into ``<root>/onboarding``, per object for HOT3D, whose reference
tars are fetched per object. The ``test`` part is also staged as an onboarding view when the
``val`` part is given too (the overlap check needs the extracted validation split). For HOT3D
the ``test`` part also stages the synchronized support streams and camera poses of every test
image (``doper.bop.hot3d_staging.stage_hot3d_support``) that the multi-view test variants fuse.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ...bop import sources
from ...bop.cad_keypoints import STL_NAME, select_cad_keypoints
from ...bop.extract import data_dir
from ...bop.sources import DatasetProfile, sha256_file
from ...console import log
from ...keypoints import KEYPOINTS_NAME
from ...recipes.schema import HOT3D_INDEX, HOT3D_PANEL, Recipe, recipe_objects
from ..common import (
    Step,
    object_dir,
)
from ._names import GEOMETRY_DIR, ONBOARDING_DIR, ONBOARDING_PARTS


def _fetch(profile: DatasetProfile, root: Path, parts: Sequence[str], allow_test: bool) -> None:
    if sources.TRANSDUCTIVE_PARTS.intersection(parts) and not allow_test:
        raise ValueError(
            "the test part holds unlabeled test inputs (RGB and intrinsics only); pass "
            "--allow-test-inputs to fetch them"
        )
    sources.fetch(profile, root, parts, allow_transductive=allow_test)
    sources.write_ready(root, profile, parts)


def _extract(recipe: Recipe, profile: DatasetProfile, root: Path, parts: Sequence[str]) -> None:
    from ...bop import extract, hot3d_source

    if recipe.convert.source_view == "hot3d":
        hot3d_source.extract_hot3d_source(root, profile, allowed_missing=recipe.generated_objects)
        return
    extract.extract_parts(root, profile, parts)
    if "pbr" in parts:
        extract.write_pbr_ready(root, profile, allowed_missing=recipe.generated_objects)


def _hot3d_steps(
    recipe: Recipe, profile: DatasetProfile, root: Path, parts: Sequence[str]
) -> list[Step]:
    from ...bop import hot3d_staging as staging

    steps = []
    if "val" in parts and recipe.validation.population == HOT3D_PANEL:

        def validation() -> None:
            for shard in range(staging.SHARDS):
                staging.stage_validation_shard(root, shard, profile)
            staging.merge_validation(root, profile)

        steps.append(
            Step(
                "stage HOT3D validation panel",
                {"clips": sources.source_dir(root), "panel": "profile resource validation_panel"},
                {"manifest": root / staging.VALIDATION_MANIFEST},
                validation,
            )
        )
    if "test" in parts and recipe.test.population == HOT3D_INDEX:

        def test() -> None:
            for shard in range(staging.SHARDS):
                staging.stage_test_shard(root, shard, profile)
            staging.merge_test(root, profile)

        steps.append(
            Step(
                "stage HOT3D test images",
                {"clips": sources.source_dir(root), "index": "profile resource test_image_index"},
                {"manifest": root / staging.TEST_DIR / staging.TEST_MANIFEST},
                test,
            )
        )
        if recipe.multiview is not None and recipe.multiview.support == "streams":
            # The multi-view test variants fuse every synchronized stream of a test image; the
            # support streams are staged with the test part (images and camera poses only).
            def support() -> None:
                staging.stage_hot3d_support(root, profile)

            steps.append(
                Step(
                    "stage HOT3D support streams",
                    {
                        "clips": sources.source_dir(root),
                        "streams": list(recipe.multiview.streams),
                    },
                    {"manifest": root / staging.TEST_DIR / staging.SUPPORT_MANIFEST},
                    support,
                )
            )
    return steps


def _stage_view(root: Path, profile: DatasetProfile, kind: str, **kwargs: Any) -> None:  # noqa: ANN401
    from ...bop import onboarding

    onboarding.stage_view(root, profile, kind, **kwargs)


def _stage_gt_view(root: Path, profile: DatasetProfile, **kwargs: Any) -> None:  # noqa: ANN401
    from ...bop import onboarding

    onboarding.stage_gt_view(root, profile, **kwargs)


def _stage_hot3d_object(root: Path, profile: DatasetProfile, obj: int) -> None:
    from ...bop import onboarding_hot3d

    onboarding_hot3d.stage_hot3d_object(root, profile, obj)


def _view_step(name: str, profile: DatasetProfile, root: Path, kind: str, run: Any) -> Step:  # noqa: ANN401
    return Step(
        name,
        {"archive": profile.parts[kind], "validation": data_dir(root) / "val"},
        {"ready": root / ONBOARDING_DIR / kind / sources.READY_NAME},
        run,
    )


def _onboarding_steps(
    profile: DatasetProfile,
    root: Path,
    parts: Sequence[str],
    objects: Sequence[int],
    args: Any,  # noqa: ANN401
) -> list[Step]:
    steps = []
    selected = args.obj_ids
    if profile.name == "hot3d":
        for obj in objects if ONBOARDING_PARTS & set(parts) else []:
            steps.append(
                Step(
                    f"stage HOT3D onboarding obj {obj}",
                    {"tars": profile.meta["onboarding_tars"].get(str(obj), []), "val": "manifest"},
                    {"receipt": root / ONBOARDING_DIR / "objects" / f"obj_{obj:06d}.json"},
                    lambda obj=obj: _stage_hot3d_object(root, profile, obj),
                )
            )
        return steps
    for kind in ("static", "dynamic"):
        if kind in parts:
            steps.append(
                _view_step(
                    f"stage {kind} onboarding",
                    profile,
                    root,
                    kind,
                    lambda kind=kind: _stage_view(root, profile, kind, objects=selected),
                )
            )
    if "static" in parts:
        steps.append(
            Step(
                "stage static GT onboarding",
                {"static": root / ONBOARDING_DIR / "static"},
                {"ready": root / ONBOARDING_DIR / "gt" / sources.READY_NAME},
                lambda: _stage_gt_view(root, profile, objects=selected),
            )
        )
    if {"test", "val"} <= set(parts) and "test" in profile.meta.get("onboarding_prefix", {}):
        steps.append(
            _view_step(
                "stage test onboarding",
                profile,
                root,
                "test",
                lambda: _stage_view(root, profile, "test", consent=args.allow_test_inputs),
            )
        )
    return steps


def _read_cad(profile: DatasetProfile, root: Path, member: str, out: Path) -> Path:
    """Copy one CAD member of the verified models archives to ``out``."""
    import zipfile

    for name in profile.parts["models"]:
        with zipfile.ZipFile(sources.source_dir(root) / name) as archive:
            if member in archive.namelist():
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(archive.read(member))
                return out
    raise ValueError(f"no {member} in the models part {profile.parts['models']}")


def _select_pinned(recipe: Recipe, profile: DatasetProfile, root: Path, obj: int) -> None:
    import tempfile

    spec = recipe.generated_spec(obj)
    geometry = object_dir(root, obj) / GEOMETRY_DIR
    member = f"{recipe.convert.models}/obj_{obj:06d}.ply"
    with tempfile.TemporaryDirectory() as scratch:
        mesh = _read_cad(profile, root, member, Path(scratch) / f"obj_{obj:06d}.ply")
        spec_keypoints = recipe.keypoints
        out = select_cad_keypoints(
            mesh, geometry, obj, spec_keypoints.count, spec_keypoints.source_units
        )
    digest = sha256_file(out)
    if spec.keypoints_sha256 is not None and digest != spec.keypoints_sha256:
        out.unlink()
        raise ValueError(
            f"object {obj}: selected keypoints {digest} differ from the pinned "
            f"{spec.keypoints_sha256}; the CAD or the selector changed"
        )
    log(f"object {obj}: {out} keypoints_sha256 {digest}")


def _keypoints_only_plan(
    recipe: Recipe, profile: DatasetProfile, args: argparse.Namespace
) -> list[Step]:
    """``--keypoints-only``: the pinned keypoints of one generated object."""
    root = Path(args.root)
    objects = recipe_objects(recipe, args.obj_ids, generated=True) if args.obj_ids else ()
    if args.generated_root is not None or len(objects) != 1 or objects[0] not in recipe.generated:
        raise ValueError(
            "--keypoints-only selects the keypoints of one generated object before its "
            f"campaign: name it with --obj-id (generated objects: {sorted(recipe.generated)}), "
            "without --generated-root"
        )
    (obj,) = objects
    spec = recipe.generated_spec(obj)
    geometry = object_dir(root, obj) / GEOMETRY_DIR
    archives = profile.parts["models"]

    def fetch_models() -> None:
        sources.fetch(profile, root, ["models"])

    return [
        Step(
            "fetch and verify models",
            {"repo": f"{profile.hf_repo}@{profile.revision}", "archives": archives},
            {"files": sources.source_dir(root)},
            fetch_models,
        ),
        Step(
            f"keypoints obj {obj}",
            {
                "cad": f"{archives}: {recipe.convert.models}/obj_{obj:06d}.ply",
                "count": recipe.keypoints.count,
                "units": recipe.keypoints.source_units,
                "pinned sha256": spec.keypoints_sha256 or "none",
            },
            {"keypoints": geometry / KEYPOINTS_NAME, "stl": geometry / STL_NAME},
            lambda: _select_pinned(recipe, profile, root, obj),
        ),
    ]


def _prepare_multisensor(root: Path, obj: int) -> None:
    """One object's three-sensor roots; an existing root is only re-verified."""
    from ...bop import multisensor

    out = multisensor.output_dir(root, obj)
    if (out / multisensor.READY_NAME).exists():
        multisensor.verify_prepared(out)
        log(f"object {obj}: multisensor root already prepared and verified: {out}")
        return
    test = data_dir(root) / multisensor.TEST_SPLIT
    if not test.is_dir():
        raise ValueError(
            f"{test} is missing: the reserved test split is needed to exclude its images; run "
            "`doper prepare tless --root ROOT --parts test --allow-test-inputs` first"
        )
    geometry = object_dir(root, obj) / GEOMETRY_DIR
    summary = multisensor.prepare(
        root,
        obj,
        keypoints=geometry / KEYPOINTS_NAME,
        cad_model=geometry / f"obj_{obj:06d}.ply",
        cad_audit=geometry / "audit.json",
        reserved=multisensor.reserved_population(test),
    )
    log(f"object {obj}: {summary.root} prepared; review its {len(summary.overlays)} overlays "
        f"with `doper review tless --obj-id {obj}`")  # fmt: skip


def _multisensor_plan(
    recipe: Recipe, profile: DatasetProfile, args: argparse.Namespace
) -> list[Step]:
    """``--multisensor``: PrimeSense TRAIN, then per object Kinect, Canon and the roots."""
    from ...bop import multisensor, tless_sensors

    spec = tless_sensors.sensor_spec(profile)
    if args.keypoints_only or args.generated_root is not None or args.parts:
        raise ValueError("--multisensor takes no --parts, --keypoints-only or --generated-root")
    # Bespoke: prepared multisensor roots are never overwritten, so --force is refused outright.
    if args.force:
        raise ValueError(
            "--multisensor takes no --force: prepared roots are never overwritten; to rebuild "
            "an object, delete ROOT/multisensor/obj_XXXXXX (this discards its REVIEWED.json, so "
            "it must be reviewed again) and rerun"
        )
    root = Path(args.root)
    objects = recipe_objects(recipe, args.obj_ids)
    static = list(profile.parts["static"])
    steps = [
        Step(
            "fetch and verify",
            {"repo": f"{profile.hf_repo}@{profile.revision}", "parts": ["static"]},
            {"files": sources.source_dir(root), "receipt": root / sources.READY_NAME},
            lambda: _fetch(profile, root, ["static"], False),
        ),
        Step(
            "extract",
            {"parts": ["static"], "archives": static},
            {"data": data_dir(root) / multisensor.PRIMESENSE_DIR},
            lambda: _extract_static(profile, root),
        ),
    ]
    for obj in objects:
        for sensor in spec["names"]:
            steps.append(
                Step(
                    f"fetch {sensor} obj {obj}",
                    {
                        "url": spec["url_template"].format(sensor=sensor, obj=obj),
                        "sha256": spec["sha256"].get(sensor, {}).get(str(obj), "observed"),
                    },
                    {
                        "archive": tless_sensors.archive_path(root, obj, sensor),
                        "receipt": tless_sensors.receipt_path(root, obj, sensor),
                        "data": tless_sensors.object_dir(root, obj, sensor),
                    },
                    lambda obj=obj, sensor=sensor: _fetch_sensor(root, obj, sensor, profile),
                )
            )
        geometry = object_dir(root, obj) / GEOMETRY_DIR
        steps.append(
            Step(
                f"prepare multisensor obj {obj}",
                {
                    "keypoints": geometry / KEYPOINTS_NAME,
                    "cad": geometry / f"obj_{obj:06d}.ply",
                    "reserved": data_dir(root) / multisensor.TEST_SPLIT,
                },
                {"root": multisensor.output_dir(root, obj)},
                lambda obj=obj: _prepare_multisensor(root, obj),
            )
        )
    return steps


def _extract_static(profile: DatasetProfile, root: Path) -> None:
    from ...bop import extract

    extract.extract_parts(root, profile, ["static"])


def _fetch_sensor(root: Path, obj: int, sensor: str, profile: DatasetProfile) -> None:
    from ...bop import tless_sensors

    receipt = tless_sensors.fetch(root, obj, sensor, profile=profile)
    log(f"{sensor} obj {obj}: {receipt.frames} frames, {receipt.checksum_origin} "
        f"sha256 {receipt.archive_sha256}")  # fmt: skip
