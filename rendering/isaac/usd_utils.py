# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""USD placement and texture lookup shared by Isaac environment implementations."""

# pyright: reportMissingImports=false, reportMissingModuleSource=false

from __future__ import annotations

import glob
import os
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from pxr import Gf as _Gf
from pxr import UsdGeom as _UsdGeom

# usd-core ships no type information for its extension modules; use them untyped.
Gf: Any = _Gf
UsdGeom: Any = _UsdGeom

TEXTURE_SUFFIXES = ("jpg", "png", "jpeg")


def texture_set(directory: str | None) -> tuple[str | None, str | None, str | None]:
    """Return ``(color, roughness, normal)`` texture paths of a CC0-style texture folder."""

    def pick(*keys: str) -> str | None:
        for key in keys:
            hits = _glob_many(directory, [f"*{key}*.{suffix}" for suffix in TEXTURE_SUFFIXES])
            if hits:
                return hits[0]
        return None

    return (
        pick("Color", "color", "albedo", "Albedo", "diff"),
        pick("Roughness", "roughness", "rough"),
        pick("NormalGL", "Normal", "normal", "nor_gl"),
    )


def _glob_many(directory: str | None, patterns: Sequence[str]) -> list[str]:
    if not directory or not os.path.isdir(directory):
        return []
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(directory, pattern)))
    return sorted({os.path.abspath(f) for f in files})


def set_world_matrix(
    prim: Any,  # noqa: ANN401 - Usd.Prim
    matrix4x4: ArrayLike,
    scale: float | None = None,
) -> None:
    """Author a single ``xformOp:transform`` so the prim's world pose is exactly ``matrix4x4``."""
    m = np.asarray(matrix4x4, dtype=float)
    if scale is not None:
        m = m @ np.diag([scale, scale, scale, 1.0])
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(Gf.Matrix4d(m.T.tolist()))
