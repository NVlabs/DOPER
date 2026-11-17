# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A BlenderProc campaign root in the published form, without rendering anything.

``write_render_root`` lays out what ``rendering.blenderproc.campaign`` publishes (see its module
docstring): ``identity.json``, ``config.json``, ``STAGED.json``, the staged ``assets/`` (target
models and frozen keypoints) and one BOP scene per layout under ``train_pbr/<layout:06d>/`` with
``records/`` and the ``COMPLETE.json`` receipt. The doper tests read it through
``doper.bop.generated``; ``tests/rendering/test_blenderproc_campaign.py`` checks the same root
with the campaign's own ``status`` verifier, so the receipt contract cannot drift on one side.

Only the standard library is imported, so both test suites can load this module by path.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "rendering" / "blenderproc" / "configs"
METADATA = (
    "AUDITED.json",
    "scene_record.json",
    "render_receipt.json",
    "visible_count_canonicalization.json",
)
SceneWriter = Callable[[Path, int], None]


def sha256(path: Path) -> str:
    """SHA-256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(value: object) -> str:
    """``doper.bop.sources.canonical_sha256``: SHA-256 of sorted, compact JSON."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def minimal_scene(scene: Path, layout: int) -> None:
    """A metadata-only BOP scene (enough for receipt verification, not for conversion)."""
    (scene / "rgb").mkdir(parents=True, exist_ok=True)
    (scene / "rgb" / "000000.jpg").write_bytes(b"jpeg %d" % layout)
    for name in ("scene_gt.json", "scene_camera.json", "scene_gt_info.json"):
        (scene / name).write_text(json.dumps({"0": [], "layout": layout}))


def render_config(name: str, **changes: object) -> dict:
    """A packaged render config (``rendering/blenderproc/configs``) with fields replaced."""
    config = json.loads((CONFIGS / f"{name}.json").read_text())
    config.update(changes)
    return config


def write_render_root(
    root: Path,
    *,
    config: dict,
    keypoints: Path,
    models: Path,
    scene: SceneWriter = minimal_scene,
    missing: Iterable[int] = (),
    failed: Iterable[int] = (),
    positive_frames: int | None = None,
) -> Path:
    """Write a published campaign root.

    Args:
        root: Campaign root to create.
        config: Effective render config JSON (``render_config``); its ``keypoints_sha256`` is
            replaced by the SHA-256 of ``keypoints``.
        keypoints: Frozen keypoints staged as ``assets/geometry/keypoints_3d.json``.
        models: BOP models directory copied to ``assets/bop/<dataset>/models``.
        scene: Writes the BOP files of one layout (``scene(directory, layout)``).
        missing: Layouts left unpublished.
        failed: Layouts left unpublished and recorded failed in ``workers/000.json``.
        positive_frames: ``target_positive_frames`` of every receipt (default: all frames).

    Returns:
        ``root``.
    """
    config = {**config, "keypoints_sha256": sha256(keypoints), "workers": 1}
    dataset, frames = config["dataset"], config["frames_per_layout"]
    assets = root / "assets"
    (assets / "geometry").mkdir(parents=True)
    shutil.copyfile(keypoints, assets / "geometry" / "keypoints_3d.json")
    shutil.copytree(models, assets / "bop" / dataset / "models")
    modulo, residue = config["split"]["modulo"], config["split"]["val_residue"]
    identity = {
        "schema": 1,
        "config": config,
        "dataset": dataset,
        "target_object": config["target_object"],
        "layouts": config["layouts"],
        "frames_per_layout": frames,
        "workers": 1,
        "base_seed": config["base_seed"],
        "split": f"layout % {modulo} == {residue} is val_pbr",
        "assets_manifest_sha256": "0" * 64,
        "keypoints_sha256": config["keypoints_sha256"],
        "code_sha256": {},
        "blenderproc": config["blenderproc"],
        "blender": config["blender"],
        "bop_toolkit": config["bop_toolkit"],
    }
    digest = canonical_sha256(identity)
    (root / "identity.json").write_text(json.dumps(identity))
    (root / "config.json").write_text(json.dumps(config))
    (root / "STAGED.json").write_text(json.dumps({"ok": True, "identity_sha256": digest}))
    skipped, failures = set(missing), set(failed)
    for layout in range(config["layouts"]):
        if layout in skipped | failures:
            continue
        _publish(
            root / "train_pbr" / f"{layout:06d}",
            layout,
            config,
            digest,
            scene,
            frames if positive_frames is None else positive_frames,
        )
    if failures:
        records = {
            str(i): {
                "error": "CountBoundError",
                "message": "over the bound",
                "seed": config["base_seed"] + i,
                "attempts": 1,
            }
            for i in failures
        }
        (root / "workers").mkdir()
        (root / "workers" / "000.json").write_text(
            json.dumps({"status": "failed_layouts", "failed": records})
        )
    return root


def _publish(
    layout_dir: Path, layout: int, config: dict, digest: str, scene: SceneWriter, positive: int
) -> None:
    scene(layout_dir, layout)
    frames = config["frames_per_layout"]
    records = layout_dir / "records"
    (records / "writer_raw").mkdir(parents=True)
    shutil.copyfile(
        layout_dir / "scene_gt_info.json", records / "writer_raw" / "scene_gt_info.json"
    )
    audited = {"ok": True, "frames": frames, "target_positive_frames": positive}
    for name in METADATA:
        payload = audited if name == "AUDITED.json" else {"layout": layout}
        (records / name).write_text(json.dumps(payload))
    files = {
        str(p.relative_to(layout_dir)): sha256(p)
        for p in sorted(layout_dir.rglob("*"))
        if p.is_file()
    }
    modulo, residue = config["split"]["modulo"], config["split"]["val_residue"]
    receipt = {
        "schema": 1,
        "identity_sha256": digest,
        "layout_id": layout,
        "seed": config["base_seed"] + layout,
        "split": "val_pbr" if layout % modulo == residue else "train_pbr",
        "dataset": config["dataset"],
        "target_object": config["target_object"],
        "frames": frames,
        "target_positive_frames": positive,
        "zero_visible_target": positive == 0,
        "max_count_correction": 0,
        "render_seconds": 1.0,
        "total_seconds": 1.0,
        "bop_bytes": 1,
        "attempt": 1,
        "gpu_index": 0,
        "log": "log",
        "files": files,
    }
    (layout_dir / "COMPLETE.json").write_text(json.dumps(receipt))
