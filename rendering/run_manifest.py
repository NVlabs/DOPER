# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provenance manifests for rendered datasets.

Each render launcher writes ``manifests/shard_<start>_<end>.json`` describing exactly what
produced its slice: code revision and image, the render config (path, digest and the resolved
values), the object USD and keypoints digests, the asset catalog fingerprint, the seed, the
shard range and worker layout, and the scene-record file digest. ``merge_shards`` then checks
that every shard agrees on everything that must be identical for the shards to form one dataset
(config, object, assets, seed, scene-spec version), that the ranges tile ``[0, total)`` without
gaps or overlap, and that the images on disk match, before writing the dataset-level
``manifest.json``.

Manifests are machine-independent: no host name, and file paths are stored relative to the
output directory when they lie inside it, otherwise as a base name. SHA-256 fields identify
content.

CLI (used by the merge step of a sharded render; runs from any directory)::

    python rendering/run_manifest.py merge <dataset_dir> [--expect-total N]
    python rendering/run_manifest.py show  <dataset_dir>

``merge`` first combines the per-worker annotation files into ``annotations.json`` (when
workers left any), then validates the shard manifests and writes ``manifest.json``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Direct script execution must resolve shared renderer modules from this checkout.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from doper.scene import geometry  # noqa: E402
from rendering.config import RenderConfig  # noqa: E402
from rendering.dataset_contract import merge_annotations, validate_dataset_records  # noqa: E402
from rendering.environment_bundle import canonical_bytes, canonical_hash  # noqa: E402
from rendering.scene_spec import SCENE_SPEC_VERSION, AssetCatalog  # noqa: E402
from rendering.shard_io import MANIFEST_DIR, load_shards, sha256_file, write_json  # noqa: E402
from rendering.usd_mesh import usd_texture_file  # noqa: E402

MANIFEST_VERSION = 1
IDENTITY_VERSION = 1

# Fields every shard of one dataset must agree on.
CONSISTENT_KEYS = (
    "config_sha256",
    "config_values",
    "object",
    "assets",
    "seed",
    "scene_spec_version",
    "images_per_sample",
)
TEXTURE_LIBRARY_PREFIX = "distractors/textures/"
HASH_THREADS = min(8, os.cpu_count() or 1)
# Config sections whose raw (backend-specific) options enter the identity.
IDENTITY_SECTIONS = (
    "output",
    "object",
    "pose",
    "camera",
    "visibility",
    "scene",
    "material",
    "render",
    "isaac",
)
# Resolved config fields that hold object or asset locations.
PATH_VALUE_KEYS = ("usd_path", "keypoints_json", "hdri_dir", "texture_dir", "distractor_dir")
# Raw keys that name machine-specific paths or execution layout; they never enter the identity.
MACHINE_SPECIFIC_KEYS = {
    "object": ("usd_path", "keypoints_json", "blend_path"),
    "scene": ("hdri_dir", "texture_dir", "distractor_dir"),
    "output": ("outdir", "output_dir", "workers", "num_workers"),
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclasses.dataclass(frozen=True)
class RenderIdentity:
    """Immutable, portable rendering inputs; execution placement is deliberately separate.

    Canonical JSON strings keep nested inputs immutable as well as the value itself.
    Renderer version is a semantic compatibility version, not a git revision.
    """

    config_json: str
    object_hashes: tuple[tuple[str, str | None], ...]
    assets_json: str
    renderer: str
    renderer_version: str
    scene_spec_version: int
    seed: int
    version: int = IDENTITY_VERSION

    def to_dict(self) -> dict[str, Any]:
        """The identity as plain JSON-able data."""
        return {
            "version": self.version,
            "config": json.loads(self.config_json),
            "object": dict(self.object_hashes),
            "assets": json.loads(self.assets_json),
            "renderer": self.renderer,
            "renderer_version": self.renderer_version,
            "scene_spec_version": self.scene_spec_version,
            "seed": self.seed,
        }

    @property
    def sha256(self) -> str:
        """SHA-256 of the canonical JSON form."""
        return canonical_hash(self.to_dict())


def local_asset_manifest(cfg: RenderConfig, catalog: AssetCatalog) -> dict[str, Any]:
    """Hash local packs once in the launcher, including texture and USD sidecars.

    Relative filenames preserve catalog ordering identity while file digests detect
    same-name replacement. Absolute storage locations never enter the manifest.
    Workers consume the persisted manifest; they must not call this scanner.

    The shared distractor texture library (potentially very large) is identified by file
    size rather than content, so every launch does not reread it.
    """
    files: dict[str, str] = {}
    hashed: list[tuple[str, Path]] = []
    directories: list[str] = []
    for group, directory in (
        ("hdri", cfg.hdri_dir),
        ("textures", cfg.texture_dir),
        ("distractors", cfg.distractor_dir),
    ):
        resolved = cfg.resolve(directory)
        if resolved is None:
            continue
        root = Path(resolved)
        if not root.exists():
            continue
        if not root.is_dir():
            raise ValueError(f"asset root is not a directory: {root}")
        directories.append(group)
        for path in sorted(root.rglob("*")):
            relative = f"{group}/{path.relative_to(root).as_posix()}"
            if path.is_dir():
                directories.append(relative)
            elif path.is_file() and relative.startswith(TEXTURE_LIBRARY_PREFIX):
                files[relative] = f"size:{path.stat().st_size}"
            elif path.is_file():
                files[relative] = ""  # Keeps catalog order; the digest is filled in below.
                hashed.append((relative, path))
            else:
                raise ValueError(f"unreadable asset: {path}")
    if hashed:
        # hashlib releases the GIL, so threads overlap the file reads.
        with ThreadPoolExecutor(max_workers=min(HASH_THREADS, len(hashed))) as executor:
            digests = executor.map(sha256_file, [path for _, path in hashed])
            files.update(zip((relative for relative, _ in hashed), digests, strict=True))
    contents = {"files": files, "directories": sorted(directories)}
    return {
        "version": 1,
        "catalog": catalog.summary(),
        **contents,
        "content_sha256": canonical_hash(contents),
    }


def normalized_scene_config(cfg: RenderConfig) -> dict[str, Any]:
    """Resolved scene values plus backend options absent from the shared dataclass.

    Parsing YAML removes formatting and key-order differences. Preserve backend raw
    options conservatively: exposure or procedural settings must never disappear
    just because the engine-neutral config does not yet expose them.
    """
    values = config_values(cfg)
    for key in PATH_VALUE_KEYS:
        values.pop(key, None)  # content and catalog identity replace machine paths
    raw = {key: dict(cfg.raw[key] or {}) for key in IDENTITY_SECTIONS if key in cfg.raw}
    if cfg.robust_appearance is not None:
        raw["robust_appearance"] = dict(cfg.robust_appearance)
    for section, keys in MACHINE_SPECIFIC_KEYS.items():
        for key in keys:
            raw.get(section, {}).pop(key, None)
    return {"resolved": values, "backend_options": {k: v for k, v in raw.items() if v}}


def build_render_identity(
    *,
    cfg: RenderConfig,
    catalog: AssetCatalog,
    seed: int,
    renderer: str,
    renderer_version: str,
    blend_path: str | None = None,
    usd_path: str | None = None,
    keypoints_path: str | None = None,
    texture_paths: Sequence[str] = (),
    renderer_settings: dict[str, Any] | None = None,
    asset_manifest: dict[str, Any] | None = None,
) -> RenderIdentity:
    """Prepare an identity without writing output or changing existing provenance.

    Launchers hash packs by default. Workers pass the launcher's persisted
    ``asset_manifest`` and verify catalog identity without rereading pack bytes.
    Those pack contents must remain immutable for the duration of the attempt.
    """
    if not renderer or not renderer_version:
        raise ValueError("renderer and semantic renderer_version are required")
    hashes: dict[str, str | None] = {
        name: sha256_file(path) if path else None
        for name, path in (
            ("blend_sha256", blend_path),
            ("usd_sha256", usd_path),
            ("keypoints_sha256", keypoints_path),
        )
    }
    textures = list(texture_paths)
    # Blender replaces this material under override. Inspection still supplies
    # the same file explicitly if another authored material or world needs it.
    if usd_path and not (renderer == "blender" and cfg.object_material_mode == "override"):
        try:
            texture = usd_texture_file(usd_path)
        except (ValueError, UnicodeError):
            texture = None  # non-writer USD: caller must supply external dependencies
        if texture and texture not in textures:
            textures.append(texture)
    anchor = blend_path or usd_path
    dependency_root = Path(anchor).parent if anchor else Path(cfg.base_dir)
    for path in sorted(set(textures)):
        logical_name = os.path.relpath(path, dependency_root)
        hashes[f"dependency:{logical_name}"] = sha256_file(path)
    if asset_manifest is None:
        asset_manifest = local_asset_manifest(cfg, catalog)
    elif asset_manifest.get("catalog") != catalog.summary():
        raise ValueError("asset catalog differs from persisted launch manifest")
    config = normalized_scene_config(cfg)
    config["renderer_settings"] = renderer_settings or {}
    config["rotation_sampler"] = geometry.rotation_sampler_identity()
    return RenderIdentity(
        config_json=canonical_bytes(config).decode(),
        object_hashes=tuple(sorted(hashes.items())),
        assets_json=canonical_bytes(asset_manifest).decode(),
        renderer=renderer,
        renderer_version=str(renderer_version),
        scene_spec_version=SCENE_SPEC_VERSION,
        seed=int(seed),
    )


def validate_resume_identity(
    outdir: str | os.PathLike[str], identity: RenderIdentity, *, require_existing: bool = False
) -> None:
    """Fail closed before scene writes, consolidation, or output manifest replacement.

    Every existing shard/dataset identity and previous attempt must agree. Legacy
    output cannot be inferred safe from IDs: migrate explicitly or use a fresh dir.

    Raises:
        ValueError: An existing manifest has a different or missing identity, is unreadable,
            or output exists without any manifest.
    """
    root = Path(outdir)
    paths = sorted((root / MANIFEST_DIR).glob("shard_*.json"))
    if (root / "manifest.json").exists():
        paths.append(root / "manifest.json")
    if require_existing and not paths:
        raise ValueError(
            "resume requires an existing render identity manifest; "
            "use a fresh render or explicit migration"
        )
    expected = identity.to_dict()

    def check(payload: dict[str, Any], source: Path) -> None:
        if payload.get("render_identity") is None:
            raise ValueError(
                f"legacy output has no reliable render identity: {source}; "
                "use a fresh output directory or explicit migration"
            )
        if payload["render_identity"] != expected:
            raise ValueError(f"render identity mismatch: {source}; existing output was not changed")
        for attempt in payload.get("attempts", []):
            check(attempt, source)

    for path in paths:
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot validate render identity: {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"invalid render manifest: {path}")
        check(payload, path)
    if not paths and root.exists():
        artifacts = [
            root / "annotations.json",
            *root.glob("annotations_worker_*.json"),
            *root.glob("scenes*.jsonl"),
            *root.glob("resume/*.json"),
        ]
        if any(p.exists() for p in artifacts) or any((root / "images").glob("*")):
            raise ValueError(
                "legacy output has no reliable render identity; "
                "use a fresh output directory or explicit migration"
            )


def start_shard_attempt(
    path: str | os.PathLike[str], manifest: dict[str, Any], identity: RenderIdentity
) -> dict[str, Any]:
    """Persist a new attempt while retaining complete prior attempt provenance.

    Call validate_resume_identity before any launcher mutation. This second check
    protects direct callers from silently replacing a shard with another identity.
    """
    path = Path(path)
    payload = dict(manifest)
    payload["render_identity"] = identity.to_dict()
    history = []
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("render_identity") != identity.to_dict():
            raise ValueError(f"render identity mismatch or legacy manifest: {path}")
        history = previous.pop("attempts", [])
        history.append(previous)
    payload["attempts"] = history
    write_json(path, payload)
    return payload


def code_revision(repo_root: str | os.PathLike[str]) -> str:
    """Git revision of the running code.

    Uses the ``DOPER_GIT_REVISION`` override, then the image's ``GIT_REVISION`` file in
    ``repo_root``, then ``git``; ``-dirty`` marks uncommitted changes under ``rendering``.
    """
    if os.environ.get("DOPER_GIT_REVISION"):
        return os.environ["DOPER_GIT_REVISION"]
    marker = Path(repo_root) / "GIT_REVISION"
    if marker.is_file():
        return marker.read_text().strip()
    try:
        rev = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain", "--", "rendering"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        return f"{rev}-dirty" if dirty else rev
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def config_values(
    cfg: RenderConfig, outdir: str | os.PathLike[str] | None = None
) -> dict[str, Any]:
    """JSON-able resolved config: dataclass fields except the raw mapping and machine root.

    Absolute object and asset paths are made portable (relative to ``outdir`` when inside it,
    else their base name); relative ones are already machine-independent and stay as written.
    """
    values = dataclasses.asdict(cfg)
    values.pop("raw", None)
    values.pop("base_dir", None)
    for key in PATH_VALUE_KEYS:
        if values.get(key) and os.path.isabs(values[key]):
            values[key] = portable_path(values[key], outdir)
    for key in ("robustness", "robust_appearance"):
        if values.get(key) is None:
            values.pop(key, None)
    return json.loads(json.dumps(values, default=list))


def portable_path(path: str | os.PathLike[str], outdir: str | os.PathLike[str] | None) -> str:
    """``path`` relative to ``outdir`` when inside it, otherwise its base name."""
    absolute = Path(os.path.abspath(path))
    if outdir is not None:
        root = Path(os.path.abspath(outdir))
        if absolute.is_relative_to(root):
            return absolute.relative_to(root).as_posix()
    return absolute.name


def shard_manifest(
    *,
    repo_root: str,
    cfg: RenderConfig,
    config_path: str,
    usd_path: str,
    keypoints_path: str | None,
    num_keypoints: int,
    catalog_summary: dict[str, Any],
    seed: int,
    start_idx: int,
    end_idx: int,
    workers: Sequence[dict[str, Any]],
    scenes_path: str | None,
    renderer: str = "isaac",
    outdir: str | None = None,
) -> dict[str, Any]:
    """Build the manifest of one running shard.

    Args:
        repo_root: Checkout the code revision is read from.
        cfg: Render configuration.
        config_path: Config file; stored relative to ``outdir`` when inside it, else its name.
        usd_path: Object USD; its digest is recorded and its name stored the same way.
        keypoints_path: Keypoints JSON, or ``None``.
        num_keypoints: Keypoint count of the object.
        catalog_summary: ``AssetCatalog.summary()`` of the asset packs.
        seed: Dataset seed.
        start_idx: First sample index of the shard.
        end_idx: One past the last sample index of the shard.
        workers: Per-worker layout records.
        scenes_path: Scene-record file, or ``None``.
        renderer: ``blender`` or ``isaac``.
        outdir: Dataset directory used to relativize stored paths.

    Returns:
        The manifest, with ``status`` ``running``.
    """
    image = os.environ.get("DOPER_IMAGE") or ""
    try:
        texture = usd_texture_file(usd_path)
    except (OSError, ValueError):  # not a layer written by usd_mesh.py; the usd digest covers it
        texture = None
    if renderer == "blender" and cfg.object_material_mode == "override":
        texture = None
    return {
        "manifest_version": MANIFEST_VERSION,
        "kind": "shard",
        "status": "running",
        "created_utc": _utc_now(),
        "renderer": renderer,
        "code": {
            "git_revision": code_revision(repo_root),
            "image": image,
            "python": platform.python_version(),
        },
        "config_path": portable_path(config_path, outdir),
        "config_sha256": sha256_file(config_path),
        "config_values": config_values(cfg, outdir),
        "object": {
            "usd": portable_path(usd_path, outdir),
            "usd_sha256": sha256_file(usd_path),
            "keypoints_json": portable_path(keypoints_path, outdir) if keypoints_path else None,
            "keypoints_sha256": sha256_file(keypoints_path) if keypoints_path else None,
            "num_keypoints": int(num_keypoints),
            "texture": portable_path(texture, outdir) if texture else None,
            "texture_sha256": sha256_file(texture) if texture and os.path.isfile(texture) else None,
        },
        "assets": dict(catalog_summary),
        "seed": int(seed),
        "scene_spec_version": SCENE_SPEC_VERSION,
        "images_per_sample": int(cfg.images_per_sample),
        "shard": {"start_idx": int(start_idx), "end_idx": int(end_idx), "workers": list(workers)},
        "scenes": {
            "file": os.path.basename(scenes_path) if scenes_path else None,
            "sha256": sha256_file(scenes_path) if scenes_path else None,
        },
    }


def shard_manifest_path(outdir: str | os.PathLike[str], start_idx: int, end_idx: int) -> Path:
    """Location of the manifest for the shard ``[start_idx, end_idx)``."""
    return Path(outdir) / MANIFEST_DIR / f"shard_{start_idx:08d}_{end_idx:08d}.json"


def finish_shard(
    path: str | os.PathLike[str],
    *,
    images_rendered: int,
    elapsed_sec: float,
    failed_workers: Sequence[int],
    problems: Sequence[str] = (),
) -> dict[str, Any]:
    """Mark the shard complete, or failed when a worker exited non-zero or output was invalid."""
    manifest = json.loads(Path(path).read_text())
    failed = bool(failed_workers or problems)
    manifest.update(
        {
            "status": "failed" if failed else "complete",
            "finished_utc": _utc_now(),
            "images_rendered": int(images_rendered),
            "elapsed_sec": round(float(elapsed_sec), 1),
            "failed_workers": list(failed_workers),
            "problems": list(problems),
        }
    )
    write_json(path, manifest)
    return manifest


def check_shards(shards: Sequence[dict[str, Any]], *, expect_total: int | None = None) -> list[str]:
    """Return a list of problems (empty when the shards form one consistent dataset)."""
    problems: list[str] = []
    if not shards:
        return ["no shard manifests found"]
    first = shards[0]
    has_identity = any(s.get("render_identity") is not None for s in shards)
    keys = (
        ("render_identity", "images_per_sample") if has_identity else (*CONSISTENT_KEYS, "renderer")
    )
    for shard in shards[1:]:
        for key in keys:
            if shard.get(key) != first.get(key):
                problems.append(f"shard {_label(shard)} differs from {_label(first)} in {key}")
    for shard in shards:
        if "render_identity" in keys:
            if shard.get("render_identity") is None:
                problems.append(f"shard {_label(shard)} has no render_identity")
            for attempt in shard.get("attempts", []):
                if attempt.get("render_identity") != shard.get("render_identity"):
                    problems.append(f"shard {_label(shard)} attempt has different render_identity")
        if shard.get("status") != "complete":
            problems.append(f"shard {_label(shard)} status is {shard.get('status')!r}")
    ranges = sorted((s["shard"]["start_idx"], s["shard"]["end_idx"]) for s in shards)
    expected_start = 0
    for start, end in ranges:
        if start < expected_start:
            problems.append(f"shards overlap at index {start}")
        elif start > expected_start:
            problems.append(f"gap in shards: indices [{expected_start}, {start}) were not rendered")
        expected_start = max(expected_start, end)
    if expect_total is not None and expected_start != expect_total:
        problems.append(
            f"shards cover [0, {expected_start}) but {expect_total} samples were expected"
        )
    return problems


def shard_warnings(shards: Sequence[dict[str, Any]]) -> list[str]:
    """Differences that do not make the shards inconsistent but are worth recording.

    Render identity carries the semantic renderer version; compatible attempts may
    have different code revisions. Keep all contributing revisions visible, including
    earlier attempts whose samples were reused.
    """
    revisions = sorted(
        {attempt["code"]["git_revision"] for s in shards for attempt in [s, *s.get("attempts", [])]}
    )
    if len(revisions) <= 1:
        return []
    return [f"shards were rendered by {len(revisions)} code revisions: {', '.join(revisions)}"]


def _label(shard: dict[str, Any]) -> str:
    return f"[{shard['shard']['start_idx']}, {shard['shard']['end_idx']})"


def merge_shards(
    outdir: str | os.PathLike[str], *, expect_total: int | None = None, strict: bool = True
) -> dict[str, Any]:
    """Validate shard manifests and the images on disk, then write ``manifest.json``.

    Raises:
        SystemExit: ``strict`` and the shards or records are inconsistent.
    """
    outdir = Path(outdir)
    shards = load_shards(outdir)
    problems = check_shards(shards, expect_total=expect_total)
    warnings = shard_warnings(shards)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    total = max((s["shard"]["end_idx"] for s in shards), default=0)
    if shards:
        per_sample = shards[0]["images_per_sample"]
        annotations = outdir / "annotations.json"
        try:
            payload = json.loads(annotations.read_text())
        except (OSError, ValueError) as exc:
            problems.append(f"annotations.json is missing or unreadable: {exc}")
        else:
            problems.extend(
                validate_dataset_records(
                    payload,
                    outdir,
                    start_idx=0,
                    end_idx=total,
                    images_per_sample=per_sample,
                    render_masks=bool(
                        shards[0].get("config_values", {}).get("render_masks", False)
                    ),
                    num_keypoints=shards[0].get("object", {}).get("num_keypoints"),
                )
            )
    if problems and strict:
        raise SystemExit("dataset manifest check failed:\n  " + "\n  ".join(problems))
    common = {key: shards[0].get(key) for key in CONSISTENT_KEYS} if shards else {}
    if shards and shards[0].get("render_identity") is not None:
        common["render_identity"] = shards[0]["render_identity"]
    code = sorted(
        {
            json.dumps(attempt["code"], sort_keys=True)
            for s in shards
            for attempt in [s, *s.get("attempts", [])]
        },
        key=str,
    )
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "kind": "dataset",
        "status": "complete" if not problems else "inconsistent",
        "problems": problems,
        "warnings": warnings,
        "created_utc": _utc_now(),
        "renderer": shards[0].get("renderer") if shards else None,
        "num_samples": total,
        "num_images": total * (common.get("images_per_sample") or 1),
        **common,
        "code": [json.loads(c) for c in code],
        "shards": [
            {
                "start_idx": s["shard"]["start_idx"],
                "end_idx": s["shard"]["end_idx"],
                "workers": s["shard"]["workers"],
                "elapsed_sec": s.get("elapsed_sec"),
                "code": s["code"],
                "scenes": s["scenes"],
                "attempts": s.get("attempts", []),
            }
            for s in shards
        ],
    }
    write_json(outdir / "manifest.json", manifest)
    return manifest


def merge_worker_annotations(outdir: str | os.PathLike[str]) -> None:
    """Merge every final ``annotations_worker_*.json`` into ``annotations.json``.

    Sharded renders (``--no-merge``) leave only worker files; this is their annotation merge.
    Does nothing when the directory holds no worker files, so an already merged dataset is
    left as is.

    Raises:
        SystemExit: Worker annotations have duplicate IDs or do not pair images and annotations.
    """
    if not any(Path(outdir).glob("annotations_worker_*.json")):
        return
    try:
        merge_annotations(outdir, None)
    except ValueError as exc:
        raise SystemExit(f"cannot merge worker annotations: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point (``merge`` and ``show``)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    merge = sub.add_parser("merge", help="validate shard manifests and write manifest.json")
    merge.add_argument("outdir")
    merge.add_argument(
        "--expect-total", type=int, default=None, help="Total samples the shards must cover"
    )
    merge.add_argument(
        "--no-strict", action="store_true", help="Write manifest.json even with problems"
    )
    show = sub.add_parser("show", help="print manifest.json or the shard manifests")
    show.add_argument("outdir")
    args = parser.parse_args(argv)
    if args.command == "merge":
        merge_worker_annotations(args.outdir)
        manifest = merge_shards(
            args.outdir, expect_total=args.expect_total, strict=not args.no_strict
        )
        print(
            f"manifest.json: {manifest['num_images']} images from {len(manifest['shards'])} "
            f"shard(s), status {manifest['status']}"
        )
        return 0 if manifest["status"] == "complete" else 1
    path = Path(args.outdir) / "manifest.json"
    payload = (
        json.loads(path.read_text()) if path.is_file() else {"shards": load_shards(args.outdir)}
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
