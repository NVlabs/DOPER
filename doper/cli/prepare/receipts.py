# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One receipt check shared by the NuRec prepare steps.

A step records the identity of its inputs in a receipt next to its outputs. A rerun with the
same identity keeps the outputs, a rerun with another identity refuses unless ``--force`` is
given, and a missing receipt means an unfinished earlier run that is rebuilt.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...console import log
from ..common import read_json


@dataclass(frozen=True)
class StageOutput:
    """Where a step keeps its outputs and the receipt that says what they were made from.

    Attributes:
        out: Directory a rebuild clears.
        receipt: File holding the recorded identity (may lie outside ``out``).
        refusal: Why a receipt of another identity is refused.
        verb: What ``--force`` does about it.
        read: Extracts the recorded identity from the receipt file.
        exact: Compare the whole record; otherwise only the keys of the current identity.
        clear_out: Whether a rebuild removes ``out`` (else only the receipt is unlinked).
    """

    out: Path
    receipt: Path
    refusal: str = "was made from other inputs or settings"
    verb: str = "replace it"
    read: Callable[[Path], Mapping[str, Any]] = read_json
    exact: bool = False
    clear_out: bool = True


def reuse_or_refuse(
    spec: StageOutput, identity: Mapping[str, Any], force: bool
) -> Mapping[str, Any] | None:
    """The matching record to keep, or ``None`` when the step must (re)build.

    With ``force`` the receipt is not read, so a corrupt one cannot block the rebuild.

    Raises:
        ValueError: When the receipt is of another identity and ``force`` is not set.
    """
    if force or not spec.receipt.exists():
        return None
    record = spec.read(spec.receipt)
    same = record == identity if spec.exact else {k: record.get(k) for k in identity} == identity
    if not same:
        raise ValueError(f"{spec.out} {spec.refusal}; rerun with --force to {spec.verb}")
    return record


def clear(spec: StageOutput) -> None:
    """Clear an unfinished or replaced build: remove ``out`` (if asked) and the receipt."""
    if spec.clear_out and spec.out.exists():
        shutil.rmtree(spec.out)
    spec.receipt.unlink(missing_ok=True)


def keep_step(
    out: Path, receipt: Path, identity: Mapping[str, Any], force: bool, what: str
) -> bool:
    """Whether a step keeps its outputs (same identity); otherwise clear them.

    Raises:
        ValueError: When the outputs were made from other inputs and ``force`` is not set.
    """
    spec = StageOutput(out, receipt)
    if reuse_or_refuse(spec, identity, force) is not None:
        log(f"already {what}: {out}")
        return True
    clear(spec)
    return False
