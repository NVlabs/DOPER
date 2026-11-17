# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared output-directory policy of the pseudo-label exporters."""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterable
from pathlib import Path


def _overlaps(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def check_output_dir(out: str | Path, force: bool, inputs: Iterable[str | Path | None]) -> None:
    """Refuse an output directory that overlaps an input or exists without ``force``.

    Args:
        out: Output directory of the exporter.
        force: Whether an existing ``out`` may be replaced.
        inputs: Input files and directories of the run; ``None`` entries are ignored.

    Raises:
        SystemExit: ``out`` equals, contains or lies inside an input path, or ``out`` exists
            and ``force`` is false.
    """
    resolved = Path(os.path.expanduser(out)).resolve()
    for item in inputs:
        if item is None:
            continue
        source = Path(os.path.expanduser(item)).resolve()
        if _overlaps(resolved, source):
            raise SystemExit(f"output {resolved} overlaps input {source}; choose another output")
    if resolved.exists() and not force:
        raise SystemExit(f"{resolved} exists; pass --force to replace it")


def replace_output_dir(out: str | Path, force: bool, inputs: Iterable[str | Path | None]) -> None:
    """Apply ``check_output_dir`` and remove an existing ``out`` so it can be recreated.

    Args:
        out: Output directory of the exporter.
        force: Whether an existing ``out`` may be replaced.
        inputs: Input files and directories of the run; ``None`` entries are ignored.

    Raises:
        SystemExit: See ``check_output_dir``.
    """
    check_output_dir(out, force, inputs)
    path = Path(os.path.expanduser(out))
    if path.exists():
        shutil.rmtree(path)
