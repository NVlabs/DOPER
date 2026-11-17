# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``doper prepare``: fetch and verify a dataset, then build each object's training data.

One module per recipe family: ``bop`` and ``bop_extras`` (BOP datasets), ``speedplus``,
and ``nurec`` with its stage modules ``nurec_capture``, ``nurec_register``,
``nurec_render``, ``nurec_labels`` and ``nurec_synthetic``. ``_names`` holds the shared
directory and file names and the flag tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ...bop.cad_keypoints import STL_NAME, select_cad_keypoints
from ...recipes import BOP_KIND
from ...speedplus.definitions import (
    DERIVED_64,
    LAYOUT_SPLITS,
    MASK_MODES,
    PAPER_MASK_MODE,
    SPLIT_NAMES,
)
from ..common import add_recipe_argument, route_kind, run_plan
from ._names import (
    CAPTURE_DIR,
    DATASET_QC_NAME,
    GEOMETRY_DIR,
    LABELS_RECEIPT_NAME,
    NUREC_STAGES,
    REGISTRATION_DIR,
    RENDER_CACHE_TARGET,
    SPEEDPLUS_STAGES,
    SYNTHETIC_RECEIPT_NAME,
    SYNTHETIC_RENDER_WORKERS,
)
from .bop import object_steps, plan, training_converted
from .nurec import nurec_plan
from .speedplus import speedplus_plan


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``doper prepare``."""
    parser = subparsers.add_parser(
        "prepare",
        help="fetch, verify and convert a dataset",
        description="Fetch and verify the pinned dataset parts, then select keypoints, convert "
        "PBR to COCO, audit and split each object (HOT3D: also stage the validation and test "
        "manifests).",
    )
    add_recipe_argument(parser)
    parser.add_argument("--root", type=Path, required=True, help="dataset root")
    parser.add_argument(
        "--parts", nargs="+", default=None, help="profile parts (default: the recipe's)"
    )
    parser.add_argument(
        "--obj-id",
        type=int,
        nargs="+",
        dest="obj_ids",
        help="objects (default: all trainable); also the HOT3D objects whose onboarding tars "
        "are fetched",
    )
    parser.add_argument(
        "--allow-test-inputs",
        action="store_true",
        help="allow fetching the unlabeled test part (RGB and intrinsics only); also consent "
        "to its transductive onboarding view",
    )
    parser.add_argument(
        "--generated-root",
        type=Path,
        default=None,
        help="render campaign root of the generated object named by --obj-id (rendering/"
        "blenderproc); every layout must be published and verified",
    )
    parser.add_argument(
        "--keypoints-only",
        action="store_true",
        help="fetch only the models part and write the pinned keypoints of the generated object "
        "named by --obj-id (the file its render campaign is staged with)",
    )
    parser.add_argument(
        "--multisensor",
        action="store_true",
        help="T-LESS: fetch the PrimeSense, Kinect and Canon TRAIN captures of the --obj-id "
        "objects (default: all) and write their supervision roots for human review. Disk: even "
        "one object downloads and extracts the full PrimeSense TRAIN part (~2.5 GB), and every "
        "Kinect/Canon archive is kept for re-verification (up to ~2 GB each; all 30 objects "
        "~60+ GB)",
    )
    forcing = parser.add_mutually_exclusive_group()
    forcing.add_argument("--force", action="store_true", help="rebuild prepared objects")
    forcing.add_argument(
        "--force-from",
        metavar="STEP",
        default=None,
        help="NuRec render, labels, synthetic: rebuild from STEP onward and keep the earlier "
        "steps (render: plan, frames; labels: geometry, masks, qualify, package; synthetic: "
        "assets, render, audit, split)",
    )
    speedplus = parser.add_argument_group("SPEED+ and NuRec recipes")
    speedplus.add_argument(
        "--stage",
        choices=(*SPEEDPLUS_STAGES, *NUREC_STAGES),
        help=f"stage to run: SPEED+ {', '.join(SPEEDPLUS_STAGES)}; NuRec {', '.join(NUREC_STAGES)}",
    )
    speedplus.add_argument(
        "--accept-license",
        action="store_true",
        help="fetch: accept the SPEED+ CC BY-NC-SA 4.0 license and the 16.9 GB download",
    )
    speedplus.add_argument(
        "--mask-mode",
        choices=MASK_MODES,
        default=None,
        help=f"convert: masks (default: the recipe's; the paper's is {PAPER_MASK_MODE})",
    )
    speedplus.add_argument(
        "--accept-non-paper-masks",
        action="store_true",
        help="convert: accept masks that are not all projected hulls (official SPNv2 masks)",
    )
    speedplus.add_argument(
        "--masks-root", type=Path, default=None, help="convert: root of official SPNv2 masks"
    )
    speedplus.add_argument(
        "--splits",
        nargs="+",
        choices=SPLIT_NAMES,
        default=None,
        help=f"convert, layout: splits (default: all for convert, {' '.join(LAYOUT_SPLITS)} for "
        "layout, as the paper)",
    )
    speedplus.add_argument(
        "--tracked-keypoints",
        type=Path,
        default=None,
        help="layout: the tracked-64 keypoints_3d.json (the paper's, or extract-landmarks output)",
    )
    speedplus.add_argument(
        "--accept-new-layout",
        action="store_true",
        help="layout: accept a layout that is not the recipe's paper layout (marked non-paper)",
    )
    speedplus.add_argument(
        "--landmark-mask-root",
        type=Path,
        default=None,
        help=f"extract-landmarks: synthetic train masks (default: <root>/{DERIVED_64}/train/masks)",
    )
    nurec = parser.add_argument_group("NuRec recipes")
    nurec.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="capture: the capture bundle (session.json, images.jsonl, images/; robot.jsonl "
        f"and handeye.json for robot seeding and evaluation); verified and normalized into "
        f"<root>/{CAPTURE_DIR}",
    )
    nurec.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="reconstruct with seed_mode doper_pnp (experimental): the DOPER checkpoint whose "
        "poses seed COLMAP",
    )
    nurec.add_argument(
        "--device",
        default=None,
        help="reconstruct with seed_mode doper_pnp: torch device (default: cuda if present)",
    )
    nurec.add_argument(
        "--artifact",
        type=Path,
        default=None,
        help="register, render, labels: the trained NuRec artifact (last.usdz) of the "
        "reconstruction",
    )
    nurec.add_argument(
        "--keypoint-frames",
        type=Path,
        default=None,
        help="register: a `doper evaluate` frames.json of the capture (DOPER keypoints and PnP "
        "errors per image) whose keypoints seed the registration",
    )
    nurec.add_argument(
        "--accept-registration",
        action="store_true",
        help=f"register: accept the reviewed <root>/{REGISTRATION_DIR}/candidate.json as the "
        "registration the labels use (needs --reviewer-note)",
    )
    nurec.add_argument(
        "--reviewer-note",
        default=None,
        help="register with --accept-registration: what the reviewer checked (recorded)",
    )
    nurec.add_argument(
        "--render-cache",
        type=Path,
        default=None,
        help=f"render: host model cache mounted at {RENDER_CACHE_TARGET} (TORCH_HOME) in the NRE "
        "container",
    )
    nurec.add_argument(
        "--sam2-weights",
        type=Path,
        default=None,
        help="labels with labels.mask_mode sam2: the SAM2 checkpoint (sam2.1_hiera_small.pt for "
        "the default config; needs the nurec-sam extra and a CUDA GPU)",
    )
    for flag, what in (("--hdri-dir", "HDRI environment maps (.hdr/.exr)"),
                       ("--texture-dir", "PBR ground texture sets"),
                       ("--distractor-dir", "distractor meshes")):  # fmt: skip
        nurec.add_argument(flag, type=Path, default=None,
                           help=f"synthetic (required): directory of {what} for the "
                           "domain-randomized render")  # fmt: skip
    nurec.add_argument(
        "--render-workers",
        type=int,
        default=None,
        help=f"synthetic: parallel Blender workers (default {SYNTHETIC_RENDER_WORKERS})",
    )
    parser.set_defaults(run=run)


def run(args: argparse.Namespace) -> int:
    """Run ``doper prepare`` (BOP, SPEED+ and NuRec recipes)."""
    planners = {BOP_KIND: plan, "speedplus": speedplus_plan, "nurec": nurec_plan}
    return run_plan(route_kind(args, planners, "prepare"), dry_run=args.dry_run)


__all__ = [
    "DATASET_QC_NAME",
    "GEOMETRY_DIR",
    "LABELS_RECEIPT_NAME",
    "STL_NAME",
    "SYNTHETIC_RECEIPT_NAME",
    "add_parser",
    "nurec_plan",
    "object_steps",
    "plan",
    "run",
    "select_cad_keypoints",
    "speedplus_plan",
    "training_converted",
]
