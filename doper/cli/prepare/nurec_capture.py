# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NuRec ``capture`` and ``reconstruct`` stages of ``doper prepare``.

``capture`` verifies the
capture bundle given with ``--bundle`` (``doper.nurec.capture``: image SHA-256s and sizes, the
rectified camera, and the robot reference when the bundle has one), writes the normalized
copies with their ``frames.json`` to ``<root>/capture/frames`` and records
``<root>/capture/READY.json`` (bundle identity, lineages, exclusions, camera flags). A rerun
with the same bundle verifies and keeps it; another bundle needs ``--force``.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from functools import cached_property
from pathlib import Path
from typing import Any

from ...bop import sources
from ...bop.sources import canonical_sha256, publish_json, sha256_file
from ...console import log
from ...keypoints import KEYPOINTS_NAME
from ...recipes.nurec_schema import NurecRecipe
from ..common import (
    Step,
    read_json,
)
from ._names import CAPTURE_DIR, FRAMES_DIR, RECONSTRUCTION_DIR
from .receipts import StageOutput, clear, reuse_or_refuse


def _capture_receipt(bundle: Path, recipe: NurecRecipe) -> tuple[dict[str, Any], Any]:
    """A capture bundle's receipt (without the frame map) and its image set, images unverified.

    The caller verifies the images: ``normalize_images`` checks each as it reads it, otherwise
    ``verify_images``.
    """
    from ...nurec import capture

    image_set = capture.load_image_set(bundle, verify=False)
    reference = None
    if capture.has_robot_reference(bundle):
        reference = capture.load_robot_reference(bundle, image_set)
    pin = capture.profile_bundle_pin(recipe.profile)
    identity = capture.verify_bundle_pin(bundle, pin, recipe.profile)
    lineages = list(dict.fromkeys(image.lineage for image in image_set.images))
    receipt = {
        "stage": "capture",
        "bundle_sha256": identity,
        "bundle_files_sha256": capture.bundle_files_sha256(bundle),
        "pinned": pin is not None,
        "image_count": len(image_set.images),
        "lineages": lineages,
        "exclude": list(image_set.exclude),
        "robot_reference": reference is not None,
        "flags": capture.capture_flags(image_set, reference),
        "max_long_edge": capture.NORMALIZED_LONG_EDGE,
        "interpolation": capture.INTERPOLATION,
    }
    return receipt, image_set


def _nurec_capture(bundle: Path, out: Path, recipe: NurecRecipe, force: bool) -> None:
    from ...nurec import capture

    receipt, image_set = _capture_receipt(bundle, recipe)
    path = out / sources.READY_NAME
    frames = out / FRAMES_DIR
    output = StageOutput(out, path, refusal="was prepared from another bundle or settings",
                         verb="rebuild it", clear_out=False)  # fmt: skip
    if path.exists() and not force:
        capture.verify_images(image_set)
    if reuse_or_refuse(output, receipt, force) is not None:
        capture.load_frame_map(frames)
        log(f"capture already prepared and verified: {receipt['image_count']} images")
        return
    # Without a receipt the frames are an unfinished earlier run: overwritten, not cleared.
    clear(output)
    capture.normalize_images(image_set, frames, force=True)
    publish_json(path, {**receipt, "bundle": str(bundle.resolve()), "frames": FRAMES_DIR,
                        "frames_sha256": sha256_file(frames / capture.FRAMES_NAME)})  # fmt: skip
    flags = receipt["flags"]
    log(f"capture ready: {receipt['image_count']} images, robot reference "
        f"{receipt['robot_reference']}, hand-eye camera mismatch "
        f"{flags['handeye_camera_mismatch']}")  # fmt: skip


def _nurec_capture_step(root: Path, recipe: NurecRecipe, args: argparse.Namespace) -> Step:
    if args.bundle is None:
        raise ValueError(
            "--stage capture needs --bundle DIR: the capture bundle (session.json, images.jsonl"
            ", images/; robot.jsonl and handeye.json for robot seeding and evaluation)"
        )
    bundle, out = Path(args.bundle), root / CAPTURE_DIR
    inputs = {"bundle": bundle, "seed_mode": recipe.seed_mode,
              "normalize": "long edge <= 1280, INTER_AREA"}  # fmt: skip
    outputs = {"frames": out / FRAMES_DIR, "receipt": out / sources.READY_NAME}
    action = lambda: _nurec_capture(bundle, out, recipe, args.force)  # noqa: E731
    return Step("verify and normalize the capture bundle", inputs, outputs, action)


class CaptureContext:
    """What the stages after ``capture`` read of the capture, loaded once per prepare run.

    Built once by ``nurec_plan`` and shared by its steps. Building it reads nothing (so
    ``--dry-run`` stays read-free); each member is loaded on first use and kept. The bundle's
    images are checked against their pinned hashes once, whichever steps need them.
    """

    def __init__(self, root: Path, recipe: NurecRecipe) -> None:
        """Remember the prepare root and the recipe; nothing is read."""
        self.root = root
        self.recipe = recipe

    @cached_property
    def receipt(self) -> dict[str, Any]:
        """The capture stage's receipt.

        Raises:
            FileNotFoundError: When the capture stage has not run.
        """
        path = self.root / CAPTURE_DIR / sources.READY_NAME
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing: run `doper prepare {self.recipe.name} "
                                    "--stage capture --bundle DIR` first")  # fmt: skip
        return read_json(path)

    @cached_property
    def frames(self) -> Any:  # noqa: ANN401
        """The capture's frame map, its normalized frames verified."""
        from ...nurec import capture

        return capture.load_frame_map(self.root / CAPTURE_DIR / FRAMES_DIR)

    @cached_property
    def image_set_and_robot(self) -> tuple[Any, Any]:
        """The capture images (verified) and robot reference a seeded strategy reads.

        Returns:
            ``(image_set, robot)``; ``robot`` is None when the bundle has no robot reference.

        Raises:
            ValueError: When the bundle the capture stage read is gone or changed since.
        """
        from ...nurec import capture

        receipt = self.receipt
        bundle = Path(receipt["bundle"])
        if not bundle.is_dir() or capture.bundle_sha256(bundle) != receipt["bundle_sha256"]:
            raise ValueError(f"the capture bundle {bundle} is missing or changed since the "
                             f"capture stage; rerun `doper prepare {self.recipe.name} --stage "
                             "capture --bundle DIR --force`")  # fmt: skip
        image_set = capture.load_image_set(bundle)
        robot = None
        if capture.has_robot_reference(bundle):
            robot = capture.load_robot_reference(bundle, image_set)
        return image_set, robot

    @cached_property
    def bundle(self) -> Path:
        """The verified capture bundle directory."""
        return self.image_set_and_robot[0].root

    def reconstruct_inputs(self) -> dict[str, str]:
        """The input hashes a reconstruction starts from (needs the capture stage's output).

        Raises:
            FileNotFoundError: When the capture stage has not run.
        """
        from ...nurec import capture

        receipt, root = self.receipt, self.root
        _ = self.frames  # the frame map is loaded and verified before its hash is read
        return {
            "bundle_sha256": receipt["bundle_sha256"],
            "capture_receipt": sha256_file(root / CAPTURE_DIR / sources.READY_NAME),
            "frames": sha256_file(root / CAPTURE_DIR / FRAMES_DIR / capture.FRAMES_NAME),
            "colmap_flags": canonical_sha256(dict(self.recipe.colmap)),
        }


def _bundle_keypoints(
    bundle: Path, recipe: NurecRecipe, purpose: str = "seed_mode doper_pnp"
) -> Any:  # noqa: ANN401
    """The bundle's ``keypoints_3d.json``, which must be the recipe's layout (points hash).

    Raises:
        FileNotFoundError: When the bundle has no keypoints.
        ValueError: For another layout.
    """
    from ...keypoints import load_layout

    path = bundle / KEYPOINTS_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{purpose} needs the capture bundle's {KEYPOINTS_NAME} (the "
                                f"CAD keypoints of the DOPER keypoints): {path}")  # fmt: skip
    return load_layout(path, recipe.object.keypoints_points_sha256)


def _bundle_mesh(bundle: Path, recipe: NurecRecipe, purpose: str) -> Path:
    """The bundle's CAD mesh (recipe ``object.mesh``), which must be the recipe's file.

    Raises:
        FileNotFoundError: When the bundle has no mesh.
        ValueError: When its SHA-256 is not the recipe's ``object.mesh_sha256``.
    """
    path = bundle / recipe.object.mesh
    if not path.is_file():
        raise FileNotFoundError(f"{purpose} needs the capture bundle's CAD mesh "
                                f"{recipe.object.mesh} (recipe object.mesh): {path}")  # fmt: skip
    digest = sha256_file(path)
    if digest != recipe.object.mesh_sha256:
        raise ValueError(f"{path} is not the recipe's mesh (SHA-256 {digest}, recipe "
                         f"object.mesh_sha256 {recipe.object.mesh_sha256})")  # fmt: skip
    return path


def _doper_pnp_seed(bundle: Path, recipe: NurecRecipe, checkpoint: Path, device: str | None) -> Any:  # noqa: ANN401
    """Mode (c)'s strategy: the checkpoint's model, the bundle keypoints, the recipe gates."""
    from ...evaluation import relative_pose
    from ...reconstruction.seeding import DoperPnpSeed

    keypoints = _bundle_keypoints(bundle, recipe)
    if device is None:
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, cfg, _ = relative_pose.load_evaluation_model(checkpoint, device)
    return DoperPnpSeed(model, keypoints, recipe.doper_pnp, cfg=cfg, device=device)


def _reconstruction_identity(path: Path) -> dict[str, Any]:
    """The recorded identity of a reconstruction: its inputs and seeding strategy."""
    from ...reconstruction.metadata import ReconstructionMetadata

    metadata = ReconstructionMetadata.read(path)
    return {"inputs": metadata.inputs, "strategy": metadata.strategy}


def _nurec_reconstruct(
    ctx: CaptureContext, force: bool, checkpoint: Path | None, device: str | None
) -> None:
    from ...reconstruction.colmap_runner import COLMAP_ENV, ColmapCli
    from ...reconstruction.metadata import METADATA_NAME, ReconstructionMetadata
    from ...reconstruction.seeding import STRATEGIES, ColmapSfmSeed, reconstruct

    root, recipe = ctx.root, ctx.recipe
    doper_pnp = recipe.seed_mode == "doper_pnp"
    receipt, frames, inputs = ctx.receipt, ctx.frames, ctx.reconstruct_inputs()
    if doper_pnp:
        assert checkpoint is not None  # nurec_plan refuses doper_pnp without --checkpoint
        inputs["checkpoint"] = sha256_file(checkpoint)
        inputs["doper_pnp_gates"] = canonical_sha256(recipe.doper_pnp.as_record())
    out = root / RECONSTRUCTION_DIR
    output = StageOutput(
        out, out / METADATA_NAME, exact=True, refusal="was reconstructed from another capture "
        "or settings", verb="rebuild it", read=_reconstruction_identity,
    )  # fmt: skip
    identity = {"inputs": inputs, "strategy": recipe.seed_mode}
    if reuse_or_refuse(output, identity, force) is not None:
        existing = ReconstructionMetadata.read(out / METADATA_NAME)
        log(f"already reconstructed: {existing.colmap.metrics['registered_images']} of "
            f"{len(existing.images)} images registered")  # fmt: skip
        return
    image_set = robot = None
    if doper_pnp:
        assert checkpoint is not None
        image_set, robot = ctx.image_set_and_robot
        strategy = _doper_pnp_seed(image_set.root, recipe, checkpoint, device)
    else:
        strategy = STRATEGIES[recipe.seed_mode]()
    if not isinstance(strategy, ColmapSfmSeed):
        if image_set is None:
            image_set, robot = ctx.image_set_and_robot
        # Refuse before touching the output (mode (c) keeps its predictions for the run).
        strategy.seed(image_set, frames, robot)
    # Without reconstruction.json the directory is an unfinished earlier run and is rebuilt.
    clear(output)
    runner = ColmapCli(os.environ.get(COLMAP_ENV, "colmap"), recipe.colmap, out / "logs")
    metadata = reconstruct(image_set, frames, strategy, runner, out,
                           frames_dir=root / CAPTURE_DIR / FRAMES_DIR, robot=robot,
                           capture_flags=receipt["flags"], inputs=inputs)  # fmt: skip
    metrics = metadata.colmap.metrics
    scale = ""
    if metadata.scale_anchor is not None:
        anchor = metadata.scale_anchor
        scale = (f", scene units per meter {anchor.scene_units_per_metre:.6g} (seed to refined "
                 f"centers RMS {anchor.rms_mm:.3g} mm, max {anchor.max_mm:.3g} mm)")  # fmt: skip
    if metadata.scale_check is not None:
        check = metadata.scale_check
        scale += (f", CAD-keypoint scale {check.scene_units_per_metre:.6g} "
                  f"({check.relative_difference:.3%} from the anchor)")  # fmt: skip
    seeded = ""
    if doper_pnp:
        added = sum(r.registered_by == "image_registrator" for r in metadata.images)
        seeded = (f", DOPER seeded {sum(r.seeded for r in metadata.images)} images "
                  f"(image_registrator added {added})")  # fmt: skip
    warnings = f", warnings {metadata.colmap.warnings}" if metadata.colmap.warnings else ""
    log(
        f"reconstruction ready ({metadata.strategy}, metric {metadata.is_metric}): "
        f"{metrics['registered_images']} of {metrics['total_images']} images registered, "
        f"{metrics['points']} points, median reprojection {metrics['median_reprojection_px']} px"
        f"{seeded}{scale}{warnings}"
    )


def _nurec_reconstruct_step(ctx: CaptureContext, args: argparse.Namespace) -> Step:
    from ...reconstruction.colmap_runner import COLMAP_ENV, REQUIRED_VERSION

    root, recipe = ctx.root, ctx.recipe
    out = root / RECONSTRUCTION_DIR
    inputs = {"capture": root / CAPTURE_DIR / sources.READY_NAME,
              "frames": root / CAPTURE_DIR / FRAMES_DIR, "seed_mode": recipe.seed_mode,
              "colmap": f"{os.environ.get(COLMAP_ENV, 'colmap')} (COLMAP {REQUIRED_VERSION})",
              "colmap commands": sorted(recipe.colmap)}  # fmt: skip
    if recipe.seed_mode == "doper_pnp":
        inputs.update({"checkpoint": args.checkpoint, "doper_pnp": recipe.doper_pnp.as_record(),
                       "reference_path": "false (mode (c) is experimental)"})  # fmt: skip
    outputs = {"model": out / "sparse", "text model": out / "sparse-text",
               "metadata": out / "reconstruction.json"}  # fmt: skip
    action = lambda: _nurec_reconstruct(  # noqa: E731
        ctx, args.force, args.checkpoint, args.device
    )
    return Step(f"reconstruct the capture with COLMAP (seed_mode {recipe.seed_mode})", inputs,
                outputs, action)  # fmt: skip


def _capture_names(indices: Sequence[int], names: Sequence[str], where: str) -> list[str]:
    """Capture indices (capture order) as capture names.

    Raises:
        ValueError: For an index beyond the capture.
    """
    beyond = [i for i in indices if i >= len(names)]
    if beyond:
        raise ValueError(f"{where} {beyond} are beyond the {len(names)} capture images")
    return [names[i] for i in indices]
