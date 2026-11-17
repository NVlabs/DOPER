# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NuRec ``synthetic`` stage of ``doper prepare``.

``synthetic`` builds the synthetic baseline dataset (``doper.nurec.synthetic``; a source
checkout, since ``rendering/`` is not part of the package, and Blender 4.2 from
``DOPER_BLENDER`` or ``PATH``). It authors the render object from the capture bundle's mesh and
keypoint layout (``rendering/nurec_assets.py``: ``object.blend``, ``model.usda``,
``keypoints_3d.json`` and ``assets_receipt.json`` with the identity checks) into
``<root>/synthetic/assets``; writes ``<root>/synthetic/render.yaml`` from
``rendering/nurec_render.yaml`` with the required ``--hdri-dir``, ``--texture-dir`` and
``--distractor-dir`` and renders 10,000 images with seed 142 (``rendering/synth_parallel.py``,
``--render-workers`` Blender workers; ``command.json`` records the call, and an unfinished
render resumes) into ``<root>/synthetic/render``; audits every sample
(``<root>/synthetic/audit.json``) and splits the render 9,750/250 by the SHA-256 of the image
id into ``<root>/synthetic/dataset`` (``<root>/synthetic/receipt.json``).
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ...bop.sources import publish_json, sha256_file
from ...console import log
from ...keypoints import KEYPOINTS_NAME
from ...recipes.nurec_schema import SYNTHETIC_SPLIT, NurecRecipe
from ..common import (
    DATASET_DIR,
    Step,
    nurec_dataset_dir,
    read_json,
)
from ._names import (
    BLENDER_ENV,
    SYNTHETIC_ASSETS_DIR,
    SYNTHETIC_AUDIT_NAME,
    SYNTHETIC_COMMAND_NAME,
    SYNTHETIC_CONFIG_NAME,
    SYNTHETIC_RECEIPT_NAME,
    SYNTHETIC_RENDER_DIR,
    SYNTHETIC_RENDER_WORKERS,
    force_flags,
    force_step_keys,
)
from .nurec_capture import CaptureContext, _bundle_keypoints, _bundle_mesh
from .receipts import keep_step


def _blender_binary() -> str:
    """The Blender binary: ``DOPER_BLENDER``, else ``blender`` on ``PATH``.

    Raises:
        FileNotFoundError: When neither names one.
    """
    binary = os.environ.get(BLENDER_ENV) or shutil.which("blender")
    if not binary:
        raise FileNotFoundError(f"the synthetic stage needs Blender 4.2 or newer: set "
                                f"{BLENDER_ENV} or put blender on PATH")  # fmt: skip
    return binary


def _run_checked(argv: Sequence[str], what: str, *, capture: bool) -> None:
    """Run ``argv``; raise with its output's tail when it fails."""
    import subprocess

    result = subprocess.run(list(argv), capture_output=capture, text=True, check=False)
    if result.returncode:
        tail = ((result.stdout or "") + (result.stderr or ""))[-4000:] if capture else ""
        raise RuntimeError(f"{what} failed (exit {result.returncode}){': ' + tail if tail else ''}")


def _synthetic_assets_identity(
    ctx: CaptureContext,
) -> tuple[dict[str, Any], Path, Path]:
    """The asset step's identity and the bundle mesh and keypoints it reads.

    Raises:
        FileNotFoundError: Without the capture stage, the bundle mesh or the checkout scripts.
        ValueError: For a changed bundle or another keypoint layout.
    """
    from ...nurec.synthetic import checkout_file

    recipe, bundle = ctx.recipe, ctx.bundle
    mesh = _bundle_mesh(bundle, recipe, "the synthetic render")
    _bundle_keypoints(bundle, recipe, "the synthetic render")
    keypoints = bundle / KEYPOINTS_NAME
    scripts = ("rendering/nurec_assets.py", "rendering/mesh_setup.py", "rendering/usd_mesh.py")
    identity = {
        "mesh_sha256": sha256_file(mesh),
        "keypoints_sha256": sha256_file(keypoints),
        "keypoints_points_sha256": recipe.object.keypoints_points_sha256,
        "scripts": {name: sha256_file(checkout_file(name)) for name in scripts},
    }
    return identity, mesh, keypoints


def _nurec_synthetic_assets(ctx: CaptureContext, force: bool) -> None:
    from ...nurec.synthetic import checkout_file

    root, recipe = ctx.root, ctx.recipe
    identity, mesh, keypoints = _synthetic_assets_identity(ctx)
    out = nurec_dataset_dir(root, "base").parent / SYNTHETIC_ASSETS_DIR
    receipt_path = out / SYNTHETIC_RECEIPT_NAME
    if keep_step(out, receipt_path, identity, force, "authored"):
        return
    script = checkout_file("rendering/nurec_assets.py")
    _run_checked([_blender_binary(), "--background", "--factory-startup", "--python-exit-code",
                  "1", "--python", str(script), "--", "--mesh", str(mesh), "--mesh-sha256",
                  recipe.object.mesh_sha256, "--keypoints",
                  str(keypoints), "--points-sha256", recipe.object.keypoints_points_sha256,
                  "--out", str(out)], "rendering/nurec_assets.py", capture=True)  # fmt: skip
    assets = read_json(out / "assets_receipt.json")
    publish_json(
        receipt_path,
        {
            "stage": "synthetic",
            "step": "assets",
            **identity,
            "assets_receipt_sha256": sha256_file(out / "assets_receipt.json"),
            "files": assets["files"],
            "identity": assets["identity"],
        },
    )
    log(f"render object authored in {out}: max keypoint error "
        f"{assets['identity']['max_keypoint_error_mm']:.2e} mm, "
        f"{assets['identity']['material_faces']} faces per material")  # fmt: skip


def _synthetic_assets(root: Path) -> tuple[Path, dict[str, Any]]:
    """The authored assets directory and its stage receipt.

    Raises:
        FileNotFoundError: When the asset step has not finished.
    """
    out = nurec_dataset_dir(root, "base").parent / SYNTHETIC_ASSETS_DIR
    receipt = out / SYNTHETIC_RECEIPT_NAME
    if not receipt.is_file():
        raise FileNotFoundError(f"{receipt} is missing: the asset step did not finish")
    return out, read_json(receipt)


def _nurec_synthetic_render(root: Path, args: argparse.Namespace) -> None:
    import sys

    from ...nurec.synthetic import RENDER_COUNT, RENDER_SEED, checkout_file, render_config

    assets, _ = _synthetic_assets(root)
    base = assets.parent
    config, render = base / SYNTHETIC_CONFIG_NAME, base / SYNTHETIC_RENDER_DIR
    dirs = {"hdri_dir": args.hdri_dir, "texture_dir": args.texture_dir,
            "distractor_dir": args.distractor_dir}  # fmt: skip
    staged = base / f".{SYNTHETIC_CONFIG_NAME}.{os.getpid()}"
    template = checkout_file("rendering/nurec_render.yaml")
    render_config(template, staged, asset_dirs=dirs, usd_path=assets / "model.usda",
                  keypoints_json=assets / KEYPOINTS_NAME)  # fmt: skip
    # Bespoke: the identity is the config bytes, refusal only when a render exists, and an
    # unfinished render resumes instead of being cleared.
    changed = config.is_file() and config.read_bytes() != staged.read_bytes()
    if changed and render.exists() and not args.force:
        staged.unlink()
        raise ValueError(
            f"{render} was rendered with another config (asset directories or "
            "object); rerun with --force to replace it"
        )
    os.replace(staged, config)
    manifest = render / "manifest.json"
    if manifest.is_file() and read_json(manifest).get("status") == "complete" and not args.force:
        log(f"already rendered: {render}")
        return
    if render.exists() and args.force:
        shutil.rmtree(render)
    workers = args.render_workers or SYNTHETIC_RENDER_WORKERS
    command = [
        sys.executable,
        str(checkout_file("rendering/synth_parallel.py")),
        "--blend-file",
        str(assets / "object.blend"),
        "--usd",
        str(assets / "model.usda"),
        "--keypoints-json",
        str(assets / KEYPOINTS_NAME),
        "--config",
        str(config),
        "--outdir",
        str(render),
        "--num-images",
        str(RENDER_COUNT),
        "--num-workers",
        str(workers),
        "--seed",
        str(RENDER_SEED),
        "--blender-bin",
        _blender_binary(),
    ]
    if render.exists() and any(render.iterdir()):
        command.append("--resume")
    publish_json(base / SYNTHETIC_COMMAND_NAME, {"argv": command})
    _run_checked(command, "rendering/synth_parallel.py", capture=False)
    log(f"rendered {RENDER_COUNT} images into {render}")


def _nurec_synthetic_audit(root: Path, force: bool) -> None:
    from ...nurec.synthetic import RENDER_COUNT, RENDER_RESOLUTION, RENDER_SEED, audit_render

    assets, receipt = _synthetic_assets(root)
    base = assets.parent
    render, audit_path = base / SYNTHETIC_RENDER_DIR, base / SYNTHETIC_AUDIT_NAME
    manifest = render / "manifest.json"
    files = receipt["files"]
    # What the render is checked against: a recorded audit is reused only for the same render,
    # config and authored object.
    expected = {"config_sha256": sha256_file(base / SYNTHETIC_CONFIG_NAME),
                "object_hashes": {"blend_sha256": files["object.blend"],
                                  "usd_sha256": files["model.usda"],
                                  "keypoints_sha256": files[KEYPOINTS_NAME]}}  # fmt: skip
    # Bespoke: a mismatching audit is silently redone, never refused.
    if audit_path.is_file() and manifest.is_file() and not force:
        recorded = read_json(audit_path)
        same_render = recorded["data_sha256"].get("manifest.json") == sha256_file(manifest)
        if recorded["status"] == "passed" and recorded.get("expected") == expected and same_render:
            log(f"render already audited: {recorded['images']} images passed")
            return
    audit = audit_render(render, count=RENDER_COUNT, seed=RENDER_SEED,
                         resolution=RENDER_RESOLUTION, keypoints=assets / KEYPOINTS_NAME,
                         **expected)  # fmt: skip
    audit["expected"] = expected
    publish_json(audit_path, audit)
    if audit["status"] != "passed":
        raise ValueError(f"the synthetic render failed its audit ({audit['problem_count']} "
                         f"problems, see {audit_path}): {audit['problems'][:3]}")  # fmt: skip
    log(f"render audited: {audit['images']} images, visible keypoints min "
        f"{audit['visible_keypoints_min']} median {audit['visible_keypoints_median']}")  # fmt: skip


def _nurec_synthetic_split(root: Path, recipe: NurecRecipe, force: bool) -> None:
    from ...nurec.synthetic import package_synthetic

    dataset = nurec_dataset_dir(root, "base")
    audit = dataset.parent / SYNTHETIC_AUDIT_NAME
    if not audit.is_file():
        raise FileNotFoundError(f"{audit} is missing: the audit step did not finish")
    if read_json(audit)["status"] != "passed":
        raise ValueError(f"{audit} records a failed audit; fix the render and rerun --stage "
                         "synthetic")  # fmt: skip
    train, val = SYNTHETIC_SPLIT
    identity = {"audit_sha256": sha256_file(audit), "train": train, "val": val}
    receipt_path = dataset.parent / SYNTHETIC_RECEIPT_NAME
    if keep_step(dataset, receipt_path, identity, force, "split"):
        return
    package = package_synthetic(dataset.parent / SYNTHETIC_RENDER_DIR, dataset, train=train,
                                val=val, audit_sha256=identity["audit_sha256"])  # fmt: skip
    _bundle_keypoints(dataset, recipe, "the synthetic dataset")
    publish_json(
        receipt_path,
        {
            "stage": "synthetic",
            "step": "split",
            **identity,
            "split_rule": package["split_rule"],
            "package_sha256": sha256_file(dataset / "package.json"),
        },
    )
    log(f"synthetic dataset written to {dataset}: {package['train']} train / {package['val']} "
        "val images (SHA-256 of the image id)")  # fmt: skip


def _nurec_synthetic_steps(ctx: CaptureContext, args: argparse.Namespace) -> list[Step]:
    """The synthetic stage's steps: assets, render, audit and split.

    Raises:
        ValueError: Without the three asset directories, or with a non-positive worker count.
    """
    from ...nurec.synthetic import ASSET_KEYS, RENDER_COUNT, RENDER_SEED

    root, recipe = ctx.root, ctx.recipe
    missing = ["--" + key.replace("_", "-") for key in ASSET_KEYS if getattr(args, key) is None]
    if missing:
        raise ValueError(f"--stage synthetic needs {', '.join(missing)}: the HDRI, ground texture "
                         "and distractor directories of the domain-randomized render (required "
                         "user inputs, e.g. CC0 sets)")  # fmt: skip
    if args.render_workers is not None and args.render_workers < 1:
        raise ValueError("--render-workers must be positive")
    base = nurec_dataset_dir(root, "base").parent
    assets = base / SYNTHETIC_ASSETS_DIR
    train, val = SYNTHETIC_SPLIT
    keys = force_step_keys("synthetic", recipe)
    forced = dict(zip(keys, force_flags(args, keys), strict=True))
    render_args = argparse.Namespace(**{**vars(args), "force": forced["render"]})
    return [
        Step("author the render object (rendering/nurec_assets.py, Blender)",
             {"mesh": f"<bundle>/{recipe.object.mesh}", "keypoints": "<bundle>/keypoints_3d.json",
              "material": "one body material"},
             {"assets": assets, "receipt": assets / SYNTHETIC_RECEIPT_NAME},
             lambda: _nurec_synthetic_assets(ctx, forced["assets"])),
        Step("render the synthetic images (rendering/synth_parallel.py, Blender)",
             {"assets": assets, "config": "rendering/nurec_render.yaml", "hdri": args.hdri_dir,
              "textures": args.texture_dir, "distractors": args.distractor_dir,
              "images": RENDER_COUNT, "seed": RENDER_SEED,
              "workers": args.render_workers or SYNTHETIC_RENDER_WORKERS},
             {"config": base / SYNTHETIC_CONFIG_NAME, "command": base / SYNTHETIC_COMMAND_NAME,
              "render": base / SYNTHETIC_RENDER_DIR},
             lambda: _nurec_synthetic_render(root, render_args)),
        Step("audit every rendered sample",
             {"render": base / SYNTHETIC_RENDER_DIR, "config": base / SYNTHETIC_CONFIG_NAME,
              "assets": assets / SYNTHETIC_RECEIPT_NAME},
             {"audit": base / SYNTHETIC_AUDIT_NAME},
             lambda: _nurec_synthetic_audit(root, forced["audit"])),
        Step("split the render into the synthetic dataset",
             {"render": base / SYNTHETIC_RENDER_DIR, "audit": base / SYNTHETIC_AUDIT_NAME,
              "split": f"{train} train / {val} val by SHA-256 of the image id"},
             {"dataset": base / DATASET_DIR, "receipt": base / SYNTHETIC_RECEIPT_NAME},
             lambda: _nurec_synthetic_split(root, recipe, forced["split"])),
    ]  # fmt: skip
