# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned render assets of the BlenderProc campaign: BOP distractors and ambientCG materials.

``assets.json`` pins the three BOP model archives used as distractors (T-LESS, HB, TYOL, from the
public ``bop-benchmark`` Hugging Face datasets at fixed revisions), the 16 ambientCG CC0 2K-JPG
room materials and the Blender 4.2.1 tarball (see ``install.fetch_blender``). :func:`fetch`
downloads the archives, verifies every SHA-256, extracts them safely and writes
``ASSET_MANIFEST.json`` (the SHA-256 of every extracted file) and a ``READY.json`` receipt. The
target object's models are not fetched here; the campaign's ``stage`` takes them separately.

Command line: ``python -m rendering.blenderproc.assets --dest DIR [--dry-run]``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
import uuid
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from doper.bop.sources import (
    canonical_sha256,
    default_downloader,
    publish_json,
    sha256_file,
)

PINS_PATH = Path(__file__).resolve().parent / "assets.json"
MANIFEST_NAME = "ASSET_MANIFEST.json"
READY_NAME = "READY.json"
ARCHIVES = "archives"
EXTRACTED = ("bop", "cctextures")
HTTP_TIMEOUT_S = 180
HttpGet = Callable[[str, Path], None]


def load_manifest() -> dict[str, Any]:
    """Return the pinned asset table (``assets.json``)."""
    return json.loads(PINS_PATH.read_text())


def http_download(url: str, path: Path) -> None:
    """Download ``url`` to ``path`` (standard library only).

    Args:
        url: HTTPS URL.
        path: Destination file.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "doper-assets"})
    with (
        urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response,
        path.open("wb") as out,
    ):
        shutil.copyfileobj(response, out, length=8 * 1024 * 1024)


def safe_extract(archive: Path, destination: Path) -> None:
    """Extract a ZIP archive, refusing absolute, parent-relative and symlink members.

    Args:
        archive: ZIP file.
        destination: Directory to extract into (created).
    """
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            name = PurePosixPath(member.filename)
            is_link = (member.external_attr >> 16) & 0o170000 == 0o120000
            if name.is_absolute() or ".." in name.parts or "\\" in member.filename or is_link:
                raise ValueError(f"{archive.name}: unsafe member {member.filename!r}")
        destination.mkdir(parents=True, exist_ok=True)
        bundle.extractall(destination)


def _verified(path: Path, expected: str) -> Path:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{path.name}: SHA-256 {actual} != pinned {expected}")
    return path


def file_table(root: Path) -> dict[str, str]:
    """Map every file under ``root/bop`` and ``root/cctextures`` (relative path) to its SHA-256."""
    files = {}
    for top in EXTRACTED:
        for path in sorted((root / top).rglob("*")):
            if path.is_symlink():
                raise ValueError(f"{path}: symlinks are not allowed in the asset tree")
            if path.is_file():
                files[str(path.relative_to(root))] = sha256_file(path)
    return files


def write_manifest(
    root: Path, archives: dict[str, str] | None = None, pins: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Write ``ASSET_MANIFEST.json`` for an extracted asset tree.

    Args:
        root: Asset root.
        archives: Verified archive SHA-256 per asset name, if known.
        pins: Pinned table the tree came from; defaults to ``assets.json``.

    Returns:
        The manifest.
    """
    manifest = {
        "schema": 1,
        "pins_sha256": canonical_sha256(pins or load_manifest()),
        "archives": archives or {},
        "files": file_table(root),
    }
    publish_json(root / MANIFEST_NAME, manifest)
    return manifest


def verify(root: Path) -> dict[str, Any]:
    """Re-hash every file listed in ``ASSET_MANIFEST.json`` and refuse unlisted extra files.

    Args:
        root: Asset root.

    Returns:
        The manifest.
    """
    manifest = json.loads((root / MANIFEST_NAME).read_text())
    actual = file_table(root)
    problems = sorted(set(actual) ^ set(manifest["files"]))
    problems += sorted(
        k for k in actual if k in manifest["files"] and actual[k] != manifest["files"][k]
    )
    if problems:
        raise ValueError(f"{root}: asset files differ from {MANIFEST_NAME}: {problems[:10]}")
    return manifest


def _fetch_archive(path: Path, expected: str, download: Callable[[], Path]) -> Path:
    if path.is_file() and sha256_file(path) == expected:
        return path
    return _verified(download(), expected)


def _extract_all(
    pins: dict[str, Any],
    archives_dir: Path,
    staging: Path,
    downloader: Callable[[str, str, str, Path], Path],
    http_get: HttpGet,
) -> dict[str, str]:
    archives: dict[str, str] = {}
    for name, pin in pins["bop_models"].items():
        path = _fetch_archive(
            archives_dir / pin["filename"],
            pin["sha256"],
            lambda pin=pin: downloader(
                pin["hf_repo"], pin["filename"], pin["revision"], archives_dir
            ),
        )
        safe_extract(path, staging / "bop" / name)
        archives[name] = pin["sha256"]
    for name, pin in pins["materials"].items():
        path = archives_dir / f"{name}_2K-JPG.zip"

        def download(url: str = pin["url"], path: Path = path) -> Path:
            http_get(url, path)
            return path

        safe_extract(_fetch_archive(path, pin["sha256"], download), staging / "cctextures" / name)
        archives[name] = pin["sha256"]
    return archives


def fetch(
    dest: Path,
    *,
    downloader: Callable[[str, str, str, Path], Path] = default_downloader,
    http_get: HttpGet = http_download,
    pins: dict[str, Any] | None = None,
) -> Path:
    """Download, verify and extract the pinned distractors and materials into ``dest``.

    Layout: ``bop/{tless,hb,tyol}/...`` (the BOP models archives), ``cctextures/<Name>/...``,
    ``archives/`` (the verified downloads), ``ASSET_MANIFEST.json`` and ``READY.json``. A
    verified ``dest`` is returned unchanged.

    Args:
        dest: Asset root.
        downloader: Hugging Face downloader ``(repo, filename, revision, directory) -> path``.
        http_get: HTTP downloader for the materials.
        pins: Pinned table; defaults to ``assets.json``.

    Returns:
        ``dest``.
    """
    pins = pins or load_manifest()
    ready = dest / READY_NAME
    if ready.is_file() and json.loads(ready.read_text()).get("pins_sha256") == canonical_sha256(
        pins
    ):
        verify(dest)
        return dest
    archives_dir = dest / ARCHIVES
    archives_dir.mkdir(parents=True, exist_ok=True)
    staging = dest / f".extract-{uuid.uuid4().hex}"
    try:
        archives = _extract_all(pins, archives_dir, staging, downloader, http_get)
        for top in EXTRACTED:
            if (dest / top).exists():
                shutil.rmtree(dest / top)
            (staging / top).rename(dest / top)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    manifest = write_manifest(dest, archives, pins)
    publish_json(
        ready,
        {
            "schema": 1,
            "pins_sha256": canonical_sha256(pins),
            "manifest_sha256": sha256_file(dest / MANIFEST_NAME),
            "files": len(manifest["files"]),
        },
    )
    return dest


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Fetch the pinned BlenderProc campaign assets.")
    parser.add_argument("--dest", type=Path, required=True, help="asset root to create")
    parser.add_argument("--dry-run", action="store_true", help="list the pinned downloads only")
    args = parser.parse_args(argv)
    pins = load_manifest()
    if args.dry_run:
        for name, pin in pins["bop_models"].items():
            print(f"{name}: {pin['hf_repo']}@{pin['revision']} {pin['filename']} {pin['sha256']}")
        for name, pin in pins["materials"].items():
            print(f"{name}: {pin['url']} {pin['sha256']}")
        print(f"-> {args.dest}")
        return 0
    fetch(args.dest)
    print(f"assets ready: {args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
