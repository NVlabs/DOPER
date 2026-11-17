# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage immutable native replay assets outside shared-storage folder watchers."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rendering.environment_bundle import BundleReader, ReplayInputs, bundle_path

# Node-local by default, independent of a TMPDIR that may point at shared storage.
STAGING_DIR = "/tmp"


@contextmanager
def local_replay(
    environment: ReplayInputs | None,
    records: Iterable[dict[str, Any]],
    *,
    directory: str | Path = STAGING_DIR,
) -> Iterator[ReplayInputs | None]:
    """Copy only the assigned records and their dependencies; keep the sealed bundle identity.

    The launcher owns the cache lifetime because Kit's fast shutdown can bypass worker
    ``finally`` blocks. No source-directory enumeration or symlinks are used.

    Args:
        environment: Verified replay inputs, or ``None`` for no native replay.
        records: Scene records whose environments the workers will render.
        directory: Parent of the temporary staging directory.

    Yields:
        Replay inputs reading the staged copy, or ``None`` when ``environment`` is ``None``.

    Raises:
        OSError: ``directory`` lacks the space for the staged files.
        ValueError: The staged copy does not reproduce the sealed hashes.
    """
    if environment is None:
        yield None
        return
    records = list(records)
    source = environment.bundle
    names = {"manifest.json"}
    for record in records:
        metadata = source.verify_record(record)
        names.add(f"records/{record['index']:06d}.json")
        names.update(metadata["files"])
    paths = {name: bundle_path(source.root, name) for name in names}
    size = sum(path.stat().st_size for path in paths.values())
    if shutil.disk_usage(directory).free < size:
        raise OSError(f"Native environment staging needs {size} bytes in {directory}")
    with tempfile.TemporaryDirectory(prefix="isaac-environment-", dir=directory) as temporary:
        root = Path(temporary)
        print(
            f"Staging native environment: {len(records)} scenes, {len(paths)} files, "
            f"{size} bytes -> {root}",
            flush=True,
        )
        for name, path in paths.items():
            target = bundle_path(root, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        # Never reuse the source reader's hash cache for newly copied files.
        bundle = BundleReader(root, usd_sha256=source.usd_sha256, seed=source.seed)
        if bundle.sha256 != source.sha256:
            raise ValueError("Staged environment manifest hash mismatch")
        for record in records:
            bundle.verify_record(record)
        print("Native environment staging verified", flush=True)
        yield ReplayInputs(bundle, environment.options)
