# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standard-library path helpers shared by host-side and Blender-side rendering code."""

from __future__ import annotations

import os
from typing import overload


@overload
def resolve_under(base_dir: str, path: None) -> None: ...


@overload
def resolve_under(base_dir: str, path: str | os.PathLike[str]) -> str: ...


def resolve_under(base_dir: str, path: str | os.PathLike[str] | None) -> str | None:
    """Join a relative ``path`` onto ``base_dir``; absolute paths pass through, ``None`` stays.

    Callers decide what an empty path means; this joins it like any other relative path.
    """
    if path is None:
        return None
    text = os.fspath(path)
    return text if os.path.isabs(text) else os.path.join(base_dir, text)
