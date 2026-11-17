# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SHA-256 manifests that pin every file of a deployment bundle folder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

MANIFEST_NAME = "manifest.json"
_READ_BLOCK_BYTES = 8 << 20


def sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of the file at ``path``."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_READ_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _bundle_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.name != MANIFEST_NAME)


def write_manifest(folder: Path, entry: dict) -> None:
    """Write ``manifest.json`` holding ``entry`` plus a ``files`` hash of every other file.

    Args:
        folder: Bundle folder; every regular file in it except the manifest is recorded.
        entry: Bundle metadata. Its ``files`` key, if any, is replaced.
    """
    folder = Path(folder)
    files = {path.name: sha256(path) for path in _bundle_files(folder)}
    (folder / MANIFEST_NAME).write_text(json.dumps({**entry, "files": files}, indent=2))


def verify(folder: Path) -> dict[str, Any]:
    """Check every recorded file hash of a bundle and return its manifest.

    Args:
        folder: Bundle folder written by :func:`write_manifest`.

    Returns:
        The parsed manifest.

    Raises:
        ValueError: ``Bundle checksum mismatch: <file>`` when a recorded file is missing
            or was edited, or ``Bundle has an unrecorded file: <file>`` when the folder holds
            a file the manifest does not record.
    """
    folder = Path(folder)
    manifest = json.loads((folder / MANIFEST_NAME).read_text())
    recorded = manifest["files"]
    for name, digest in recorded.items():
        path = folder / name
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    for path in _bundle_files(folder):
        if path.name not in recorded:
            raise ValueError(f"Bundle has an unrecorded file: {path.name}")
    return manifest
