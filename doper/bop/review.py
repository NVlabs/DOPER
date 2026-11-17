# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The human review gate: ``REVIEWED.json`` binds a reviewer to exact overlays and receipt.

Some label pools need a person to look at them before a model trains on them: HANDAL object 7's
relaxed-gate bootstrap pool and the T-LESS multisensor real data. The gate is the same for both
and knows nothing about either dataset:

* ``write`` records the reviewer, the SHA-256 of the receipt the review is about (``ready``,
  for example a pool's ``labels_ready.json`` or a prepared root's ``READY.json``) and the
  SHA-256 of every overlay image the reviewer inspected, in ``<root>/REVIEWED.json``;
* ``verify`` raises ``ReviewError`` unless that record exists, confirms the review, still
  matches the receipt and every overlay byte for byte, and (when given) covers exactly the
  overlays the caller expects.

Any change to the receipt or to an overlay therefore makes the review stale. Paths are stored
relative to ``root`` so a moved tree keeps its review. Nothing here imports torch or OpenCV.
"""

from __future__ import annotations

import datetime
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .sources import publish_json, sha256_file

REVIEWED_NAME = "REVIEWED.json"
SCHEMA_VERSION = 1


class ReviewError(ValueError):
    """A review is missing, unconfirmed, stale or incomplete."""


def _relative(root: Path, path: Path, what: str, *, must_exist: bool = True) -> str:
    """``path`` relative to ``root``; it must be inside ``root`` (and an existing file)."""
    path = Path(path)
    if must_exist and not path.is_file():
        raise ValueError(f"{what} {path} is missing")
    relative = os.path.relpath(path.resolve(), root.resolve())
    if relative.startswith("..") or os.path.isabs(relative):
        raise ReviewError(f"{what} {path} must be inside {root}")
    return Path(relative).as_posix()


def write(root: Path, ready: Path, overlays: Sequence[Path], reviewer: str) -> Path:
    """Record that ``reviewer`` inspected ``overlays`` for the receipt ``ready``.

    Args:
        root: Directory of the reviewed data; ``REVIEWED.json`` is written there.
        ready: The receipt the review is about (inside ``root``).
        overlays: Every overlay image the reviewer inspected (inside ``root``).
        reviewer: The reviewer's name.

    Returns:
        The ``REVIEWED.json`` path.

    Raises:
        ValueError: For an empty reviewer name, no overlays, or a receipt or overlay that is
            missing or outside ``root``.
    """
    root = Path(root)
    if not reviewer.strip():
        raise ValueError("a reviewer name is required")
    if not overlays:
        raise ValueError("no overlays to review; a review must cover the generated previews")
    ready_name = _relative(root, ready, "review receipt")
    reviewed = {
        _relative(root, overlay, "review overlay"): sha256_file(overlay) for overlay in overlays
    }
    path = root / REVIEWED_NAME
    publish_json(
        path,
        {
            "schema": SCHEMA_VERSION,
            "reviewer": reviewer.strip(),
            "reviewed_at_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "ready": ready_name,
            "ready_sha256": sha256_file(ready),
            "reviewed_overlays": dict(sorted(reviewed.items())),
            "operator_confirmed": True,
        },
    )
    return path


def read(root: Path) -> dict[str, Any]:
    """Return ``<root>/REVIEWED.json``.

    Raises:
        ReviewError: If it is missing.
    """
    path = Path(root) / REVIEWED_NAME
    if not path.is_file():
        raise ReviewError(f"no human review ({path} is missing)")
    return json.loads(path.read_text())


def verify(root: Path, ready: Path, *, overlays: Sequence[Path] | None = None) -> None:
    """Require a confirmed, current review of ``ready``.

    Args:
        root: Directory holding ``REVIEWED.json``.
        ready: The receipt the review must still match.
        overlays: The overlays the review must cover exactly (None: the recorded ones).

    Raises:
        ReviewError: If the review is missing or unconfirmed, ``ready`` changed since it (a stale
            review), a reviewed overlay changed or is gone, or the review does not cover
            ``overlays``.
    """
    root = Path(root)
    record = read(root)
    if record.get("operator_confirmed") is not True or not str(record.get("reviewer", "")):
        raise ReviewError(f"{root / REVIEWED_NAME} does not confirm a named human review")
    if not Path(ready).is_file() or record.get("ready_sha256") != sha256_file(ready):
        raise ReviewError(f"the review in {root} is stale: {ready} changed since it")
    reviewed = record.get("reviewed_overlays") or {}
    if not reviewed:
        raise ReviewError(f"the review in {root} covers no overlays")
    for name, digest in reviewed.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ReviewError(f"reviewed overlay {path} changed since the review")
    if overlays is not None:
        expected = {
            _relative(root, overlay, "review overlay", must_exist=False) for overlay in overlays
        }
        if expected != set(reviewed):
            raise ReviewError(
                f"the review in {root} does not cover the current overlays "
                f"({len(reviewed)} reviewed, {len(expected)} generated)"
            )
