# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Names, stage tuples and flag tables shared by the ``doper prepare`` modules."""

from __future__ import annotations

import argparse
from typing import Any

GEOMETRY_DIR = "geometry"


CONVERTED_DIR = "converted"


VISIBLE_DIR = "converted_visible"


PROJECTION_NAME = "projection_audit.json"


TRAINING_PARTS = frozenset({"models", "pbr"})


ONBOARDING_PARTS = frozenset({"static", "dynamic"})


ONBOARDING_DIR = "onboarding"  # doper.bop.onboarding, which pulls torch: imported when run


SPEEDPLUS_STAGES = ("fetch", "convert", "extract-landmarks", "layout")


NUREC_STAGES = ("capture", "reconstruct", "register", "render", "labels", "synthetic")


# Flags of one recipe kind, refused for the other.
BOP_FLAGS = {
    "parts": "--parts",
    "obj_ids": "--obj-id",
    "allow_test_inputs": "--allow-test-inputs",
    "generated_root": "--generated-root",
    "keypoints_only": "--keypoints-only",
    "multisensor": "--multisensor",
}


# Flags only ``--stage render`` takes.
RENDER_FLAGS = {"render_cache": "--render-cache"}


# Flags only ``--stage register`` takes.
REGISTER_FLAGS = {"keypoint_frames": "--keypoint-frames",
                  "accept_registration": "--accept-registration",
                  "reviewer_note": "--reviewer-note"}  # fmt: skip


# Stages that import the trained NuRec artifact (``--artifact``).
ARTIFACT_STAGES = ("register", "render", "labels")


# Flags only ``--stage reconstruct`` with ``seed_mode: doper_pnp`` takes.
DOPER_PNP_FLAGS = {"checkpoint": "--checkpoint", "device": "--device"}


CAPTURE_DIR = "capture"


RECONSTRUCTION_DIR = "reconstruction"


FRAMES_DIR = "frames"


RENDER_DIR = "render"


REGISTRATION_DIR = "registration"


RENDER_PLAN_NAME = "plan.json"


# The render step's receipt: the frames' SHA-256 manifest, which the labels steps bind to.
RENDER_RECEIPT_NAME = "receipt.json"


# The labels stage: ``<root>/labels/<step>/`` (the geometry step here; masks, qualification and
# packaging add their own steps beside it).
LABELS_DIR = "labels"


LABELS_GEOMETRY_DIR = "geometry"


LABELS_RECEIPT_NAME = "receipt.json"


LABELS_MASKS_DIR = "masks"


LABELS_QUALIFIED_DIR = "qualified"


QUALIFIED_NAME = "qualified_labels.json"


DATASET_QC_NAME = "dataset_qc.json"


# Flags only ``--stage labels`` takes.
LABELS_FLAGS = {"sam2_weights": "--sam2-weights"}


# Flags only ``--stage synthetic`` takes.
SYNTHETIC_FLAGS = {
    "hdri_dir": "--hdri-dir",
    "texture_dir": "--texture-dir",
    "distractor_dir": "--distractor-dir",
    "render_workers": "--render-workers",
}


# Flags of NuRec recipes, refused for the other kinds: the per-stage flags plus the shared ones.
NUREC_FLAGS = {
    "bundle": "--bundle",
    **DOPER_PNP_FLAGS,
    "artifact": "--artifact",
    **RENDER_FLAGS,
    **REGISTER_FLAGS,
    **LABELS_FLAGS,
    **SYNTHETIC_FLAGS,
    "force_from": "--force-from",
}


# Flags that only one ``--stage`` takes.
STAGE_ONLY_FLAGS = {
    "render": RENDER_FLAGS,
    "register": REGISTER_FLAGS,
    "labels": LABELS_FLAGS,
    "synthetic": SYNTHETIC_FLAGS,
}


# The synthetic stage: ``<root>/synthetic/{assets,render,dataset}`` and its records.
SYNTHETIC_ASSETS_DIR = "assets"


SYNTHETIC_RENDER_DIR = "render"


SYNTHETIC_CONFIG_NAME = "render.yaml"


SYNTHETIC_COMMAND_NAME = "command.json"


SYNTHETIC_AUDIT_NAME = "audit.json"


SYNTHETIC_RECEIPT_NAME = "receipt.json"


# Blender workers of the synthetic render unless --render-workers says otherwise.
SYNTHETIC_RENDER_WORKERS = 8


BLENDER_ENV = "DOPER_BLENDER"


# Worker processes of the geometry labels; the labels do not depend on it.
LABEL_WORKERS = 8


# Where the render container mounts its model cache (``--render-cache``).
RENDER_CACHE_TARGET = "/workdir/cache"


SPEEDPLUS_FLAGS = {
    "stage": "--stage",
    "accept_license": "--accept-license",
    "mask_mode": "--mask-mode",
    "accept_non_paper_masks": "--accept-non-paper-masks",
    "masks_root": "--masks-root",
    "splits": "--splits",
    "tracked_keypoints": "--tracked-keypoints",
    "accept_new_layout": "--accept-new-layout",
    "landmark_mask_root": "--landmark-mask-root",
}


def refuse_flags(args: argparse.Namespace, flags: dict[str, str], kind: str) -> None:
    given = [flag for name, flag in flags.items() if getattr(args, name, None)]
    if given:
        raise ValueError(f"`doper prepare` takes {given} only for {kind} recipes")


def refuse_stage_flags(args: argparse.Namespace) -> None:
    """Refuse each stage-only flag unless ``--stage`` is the stage that takes it."""
    for stage, flags in STAGE_ONLY_FLAGS.items():
        given = [flag for name, flag in flags.items() if getattr(args, name, None)]
        if given and args.stage != stage:
            raise ValueError(f"`doper prepare` takes {given} only for --stage {stage}")


# The steps ``--force-from`` names, per stage, in the order the stage runs them.
FORCE_STEPS = {
    "render": ("plan", "frames"),
    "labels": ("geometry", "masks", "qualify", "package"),
    "synthetic": ("assets", "render", "audit", "split"),
}


def force_step_keys(stage: str, recipe: Any) -> tuple[str, ...]:  # noqa: ANN401
    """The steps of ``stage`` that ``recipe`` runs, in order (no ``masks`` without sam2)."""
    keys = FORCE_STEPS.get(stage, ())
    if stage == "labels" and recipe.labels.mask_mode != "sam2":
        return tuple(key for key in keys if key != "masks")
    return keys


def force_flags(args: argparse.Namespace, keys: tuple[str, ...]) -> list[bool]:
    """Per step of ``keys``: whether it is forced (``--force``, or at/after ``--force-from``)."""
    start = getattr(args, "force_from", None)
    first = keys.index(start) if start in keys else len(keys)
    return [bool(args.force) or index >= first for index in range(len(keys))]
