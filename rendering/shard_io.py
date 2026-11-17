# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standard-library helpers for shard manifests, shared with Blender-side code.

Blender's bundled Python has no SciPy, so everything the environment export needs to hash
files and read or write manifests lives here instead of in ``rendering.run_manifest``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

MANIFEST_DIR = "manifests"
HASH_CHUNK_BYTES = 1 << 20


def sha256_file(path: str | os.PathLike[str], chunk: int = HASH_CHUNK_BYTES) -> str:
    """Return the hex SHA-256 digest of a file, read in ``chunk``-byte blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | os.PathLike[str], payload: dict[str, Any]) -> Path:
    """Atomically write ``payload`` as sorted, indented JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return path


def load_shards(outdir: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read every ``manifests/shard_*.json`` under ``outdir`` in file-name order."""
    return [
        json.loads(p.read_text()) for p in sorted(Path(outdir, MANIFEST_DIR).glob("shard_*.json"))
    ]
