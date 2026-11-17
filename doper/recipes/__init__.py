# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packaged DOPER recipes and the pinned dataset profiles they use.

A recipe's ``kind`` picks its schema: ``bop`` (the default when a recipe names none,
``doper.recipes.schema``), ``speedplus`` (``speedplus_schema``), ``shirt`` (``shirt_schema``) or
``nurec`` (``nurec_schema``). ``load_any_recipe`` loads any of them; ``recipe_kind`` names the
kind of a loaded recipe. Parsing imports no torch; SPEED+ and SHIRT parsing imports no numpy
either (BOP and NuRec recipes check their training keys against ``doper.train``).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import MISSING, fields
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .nurec_schema import NurecRecipe
    from .schema import Recipe
    from .shirt_schema import ShirtRecipe
    from .speedplus_schema import SpeedplusRecipe

BOP_KIND = "bop"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def profile_directory() -> Path:
    """The directory of the packaged dataset profiles."""
    return Path(str(resources.files("doper.recipes").joinpath("profiles")))


def profile_path(name: str) -> Path:
    """The packaged profile file of ``name`` (it may not exist)."""
    return profile_directory() / f"{name}.json"


def _check_legacy_checkpoints(registry: object) -> None:
    """Require a ``legacy_checkpoints`` mapping of lowercase hex SHA-256 pairs.

    Raises:
        ValueError: For a non-mapping registry or a key or value that is not a SHA-256.
    """
    if not isinstance(registry, Mapping):
        raise ValueError(f"legacy_checkpoints must be a mapping, got {type(registry).__name__}")
    for checkpoint, layout in registry.items():
        if not isinstance(checkpoint, str) or not _SHA256.fullmatch(checkpoint):
            raise ValueError(f"legacy_checkpoints key {checkpoint!r} is not a checkpoint SHA-256")
        if not isinstance(layout, str) or not _SHA256.fullmatch(layout):
            raise ValueError(
                f"legacy_checkpoints[{checkpoint}] = {layout!r} is not a layout SHA-256"
            )


def _parse_kind_recipe(
    cls: type[SpeedplusRecipe] | type[ShirtRecipe],
    data: Mapping[str, Any],
    source: Path | None = None,
) -> SpeedplusRecipe | ShirtRecipe:
    """Build a SPEED+ or SHIRT recipe from its parsed YAML mapping.

    Raises:
        ValueError: For a wrong kind, unknown or missing keys, an unknown score definition or a
            malformed ``legacy_checkpoints`` registry.
    """
    from ..speedplus.definitions import SCORE_DEFINITIONS

    kind = cls.kind
    if data.get("kind") != kind:
        raise ValueError(f"recipe kind must be {kind!r}, got {data.get('kind')!r}")
    declared = {f.name: f for f in fields(cls) if f.name != "source"}
    values = {key: value for key, value in data.items() if key != "kind"}
    unknown = sorted(set(values) - set(declared))
    missing = sorted(
        name
        for name, f in declared.items()
        if name not in values and f.default is MISSING and f.default_factory is MISSING
    )
    if unknown or missing:
        raise ValueError(f"{kind} recipe has unknown keys {unknown} / missing keys {missing}")
    if values.get("score", "paper") not in SCORE_DEFINITIONS:
        raise ValueError(
            f"recipe score must be one of {SCORE_DEFINITIONS}, got {values['score']!r}"
        )
    _check_legacy_checkpoints(values.get("legacy_checkpoints", {}))
    return cls(**values, source=source)


def load_any_recipe(
    name_or_path: str | Path,
) -> Recipe | SpeedplusRecipe | ShirtRecipe | NurecRecipe:
    """Load a packaged recipe by name, or a recipe YAML, with the schema of its ``kind``.

    Raises:
        FileNotFoundError: If neither a packaged recipe nor a file matches.
        ValueError: For an unknown kind or an invalid recipe.
    """
    import yaml

    from .nurec_schema import NurecRecipe, parse_nurec_recipe
    from .schema import parse_recipe, recipe_path
    from .shirt_schema import ShirtRecipe
    from .speedplus_schema import SpeedplusRecipe

    path = recipe_path(name_or_path)
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, Mapping):
        raise ValueError(f"recipe {path} must be a mapping")
    kind = data.get("kind", BOP_KIND)
    if kind == BOP_KIND:
        return parse_recipe(data, source=path.resolve())
    if kind == NurecRecipe.kind:
        return parse_nurec_recipe(data, source=path.resolve())
    classes = {cls.kind: cls for cls in (SpeedplusRecipe, ShirtRecipe)}
    if kind not in classes:
        known = [BOP_KIND, *classes, NurecRecipe.kind]
        raise ValueError(f"recipe {path} has unknown kind {kind!r}; known: {known}")
    return _parse_kind_recipe(classes[kind], data, source=path.resolve())


def recipe_kind(recipe: Recipe | SpeedplusRecipe | ShirtRecipe | NurecRecipe) -> str:
    """The kind of a loaded recipe: ``bop``, ``speedplus``, ``shirt`` or ``nurec``."""
    return getattr(recipe, "kind", BOP_KIND)
