# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native OpenUSD asset relocation for Blender's single-layer environment export.

``pxr`` (``usd-core``) is imported lazily so the module itself imports without it.
"""

# pyright: reportMissingImports=false, reportMissingModuleSource=false

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from rendering.environment_bundle import bundle_path, publish_bytes

TEXTURE_SUFFIXES = {".png", ".jpg", ".jpeg", ".hdr", ".exr", ".tif", ".tiff"}
PXR_MISSING = "pxr is required for USD environment export; install usd-core (doper[usd])"


def _load_pxr() -> tuple[Any, Any]:
    """Import ``Sdf`` and ``UsdUtils`` from ``pxr``.

    Returns:
        The ``Sdf`` and ``UsdUtils`` modules.

    Raises:
        ImportError: ``pxr`` is not installed.
    """
    try:
        from pxr import Sdf, UsdUtils
    except ImportError as exc:
        raise ImportError(PXR_MISSING) from exc
    return Sdf, UsdUtils


def package_layer(
    source_path: str | Path, root: str | Path, *, asset_cache: dict | None = None
) -> tuple[str, list[str]]:
    """Copy textures once by content hash and rewrite assets using OpenUSD APIs.

    Args:
        source_path: Single-layer native environment ``.usda`` with default prim
            ``BlenderEnvironment``.
        root: Bundle root receiving ``textures/`` and ``layers/``.
        asset_cache: Optional per-export cache of already published source textures.

    Returns:
        The bundle-relative layer name and the sorted bundle-relative texture names.

    Raises:
        ImportError: ``pxr`` is not installed.
        ValueError: The layer is nested, has the wrong default prim, or references an
            unsupported or changing asset.
    """
    sdf, usd_utils = _load_pxr()
    root = Path(root)
    source = sdf.Layer.FindOrOpen(str(source_path))
    if source is None or source.subLayerPaths:
        raise ValueError("native environment must be one layer without nested layers")
    if source.defaultPrim != "BlenderEnvironment":
        raise ValueError("native environment defaultPrim must be BlenderEnvironment")
    layer = sdf.Layer.CreateAnonymous(".usda")
    layer.TransferContent(source)
    cache = {} if asset_cache is None else asset_cache
    dependencies = set()

    def relocate(path: str) -> str:
        if not path:
            return path
        resolved = Path(sdf.ComputeAssetPathRelativeToLayer(source, path)).resolve()
        if resolved.suffix.lower() not in TEXTURE_SUFFIXES:
            raise ValueError(f"unsupported nested layer or environment asset: {path}")
        stat = resolved.stat()
        stamp = (stat.st_size, stat.st_mtime_ns)
        if resolved in cache:
            previous_stamp, name = cache[resolved]
            if previous_stamp != stamp:
                raise ValueError(f"environment asset changed during export: {resolved}")
        else:
            data = resolved.read_bytes()
            after = resolved.stat()
            if (after.st_size, after.st_mtime_ns) != stamp:
                raise ValueError(f"environment asset changed during export: {resolved}")
            digest = hashlib.sha256(data).hexdigest()
            name = f"textures/{digest}{resolved.suffix.lower()}"
            publish_bytes(bundle_path(root, name), data)
            cache[resolved] = (stamp, name)
        dependencies.add(name)
        return "../" + name

    usd_utils.ModifyAssetPaths(layer, relocate)
    data = layer.ExportToString().encode()
    name = f"layers/{hashlib.sha256(data).hexdigest()}.usda"
    publish_bytes(bundle_path(root, name), data)
    return name, sorted(dependencies)


def validate_layer_dependencies(root: str | Path, metadata: dict[str, Any]) -> None:
    """Reject undeclared or outside assets before composing the imported layer.

    Args:
        root: Sealed bundle root.
        metadata: The scene record metadata (``layer`` and ``files``).

    Raises:
        ImportError: ``pxr`` is not installed.
        ValueError: The layer is invalid or its assets differ from the declared files.
    """
    sdf, usd_utils = _load_pxr()
    root = Path(root).resolve()
    source = sdf.Layer.FindOrOpen(str(root / metadata["layer"]))
    if source is None or source.subLayerPaths or source.defaultPrim != "BlenderEnvironment":
        raise ValueError("invalid native environment root or nested layers")
    layer = sdf.Layer.CreateAnonymous(".usda")
    layer.TransferContent(source)
    dependencies = set()

    def inspect(path: str) -> str:
        if not path:
            return path
        resolved = Path(sdf.ComputeAssetPathRelativeToLayer(source, path)).resolve()
        if not resolved.is_relative_to(root) or resolved.suffix.lower() not in TEXTURE_SUFFIXES:
            raise ValueError(f"undeclared environment dependency: {path}")
        name = resolved.relative_to(root).as_posix()
        if name not in metadata["files"]:
            raise ValueError(f"undeclared environment dependency: {path}")
        dependencies.add(name)
        return path

    usd_utils.ModifyAssetPaths(layer, inspect)
    if dependencies != set(metadata["files"]) - {metadata["layer"]}:
        raise ValueError("environment dependency inventory differs from native USD")
