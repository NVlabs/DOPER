# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""POSIX advisory locks for concurrent launchers sharing a dataset directory."""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def output_lock(
    outdir: str | os.PathLike[str], *, shard: tuple[int, int] | None = None
) -> Iterator[None]:
    """Serialize setup, or hold one shard's ownership for the task lifetime.

    Lock files live beside the output directory and are never unlinked: removing one could
    let two callers lock different inodes for the same dataset.

    Args:
        outdir: Dataset directory; the lock file is ``.<name>.<suffix>.lock`` in its parent.
        shard: ``(start_idx, end_idx)`` to lock that shard; ``None`` locks dataset setup.

    Yields:
        Nothing; the lock is held until the ``with`` block exits.
    """
    outdir = Path(outdir).resolve()
    outdir.parent.mkdir(parents=True, exist_ok=True)
    suffix = "setup" if shard is None else f"shard-{shard[0]}-{shard[1]}"
    with (outdir.parent / f".{outdir.name}.{suffix}.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
