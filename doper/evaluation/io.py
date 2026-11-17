# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared input-label and JSON-output contracts for checkpoint evaluators."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path


def parse_named_inputs(
    values: Sequence[str], option: str, *, prefix: str | None = None
) -> list[tuple[str, str]]:
    """Parse ``NAME=PATH`` or bare paths into unique, filename-safe labels.

    A separator before the first equals sign identifies a bare path, including
    ``./epoch=10.pth``. Unqualified ``epoch=10.pth`` is inherently ambiguous and uses
    the explicit-name syntax. Paths after an explicit name are left untouched.

    Args:
        values: Ordered CLI input strings containing explicit labels or bare paths.
        option: CLI option name without leading dashes, used in errors and as the
            default label prefix. Must be filename-safe when used as a prefix.
        prefix: Optional filename-safe prefix for generated labels. None or an
            empty string uses ``option``; the zero-based input index is appended.

    Returns:
        Ordered ``(label, path)`` pairs with unique labels; paths are not resolved
        or checked for existence.

    Raises:
        ValueError: A label or path is empty, or a label is duplicated, contains
            a NUL character, or is a dot or double-dot component.
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for i, value in enumerate(values):
        name, sep, path = value.partition("=")
        if not sep or "/" in name or "\\" in name:
            name, path = f"{prefix or option}{i}", value
        if not name or not path:
            raise ValueError(f"--{option} expects name=path, got {value!r}")
        if name in {".", ".."} or "\x00" in name:
            raise ValueError(f"invalid --{option} name {name!r}: expected a filename component")
        if name in seen:
            raise ValueError(f"duplicate --{option} name {name!r}")
        seen.add(name)
        out.append((name, path))
    return out


def json_safe(value: object) -> object:
    """Replace non-finite floats with None for strict JSON serialization.

    Args:
        value: Result payload composed of JSON-compatible scalars, dictionaries,
            lists and tuples. Dictionary keys must already be JSON-compatible.

    Returns:
        A recursively rebuilt payload with non-finite Python floats replaced
        by None and tuples converted to lists. Dictionary keys and other values
        are preserved; arbitrary objects are not converted into JSON-compatible types.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: object) -> None:
    """Write ``payload`` as strict JSON (sorted keys, indent 2), creating parent directories.

    Non-finite floats are written as ``null`` so the file parses with any strict decoder.

    Args:
        path: Output file; missing parent directories are created.
        payload: JSON-compatible result payload (see ``json_safe``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(json_safe(payload), f, indent=2, sort_keys=True, allow_nan=False)
