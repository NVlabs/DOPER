# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed SPEED+ recipe (``kind: speedplus``).

The SPEED+ recipe drives the SPEC2021 Tango pipeline through the ``doper`` verbs, which route
on the recipe's ``kind``. It names its dataset profile and the SPEED score definition its
evaluation summaries use (``doper.speedplus.metrics``); the stages are added stage by stage.
``doper.recipes.load_any_recipe`` parses it and refuses unknown keys. Nothing here imports
numpy or torch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from ..speedplus.definitions import (
    MASK_MODES,
    PAPER_LAYOUT,
    PAPER_MASK_MODE,
    LayoutPins,
    ScoreDefinition,
)


@dataclass(frozen=True)
class SpeedplusRecipe:
    """The SPEED+ recipe.

    Attributes:
        kind: ``speedplus`` (class constant; the YAML must carry it).
        name: Recipe name.
        profile: Packaged dataset profile name.
        score: Default SPEED score definition, ``paper`` or ``official``.
        mask_mode: Masks of the 64-point conversion (``doper.speedplus.data.convert``). The
            paper's is ``hull`` (projected hulls); any other needs ``--accept-non-paper-masks``.
        legacy_checkpoints: Source-format SPEED+ checkpoints, which record no keypoint layout,
            accepted by their file SHA-256 together with the SHA-256 of the
            ``keypoints_3d.json`` they were trained on (``doper.speedplus.checkpoints``).
        layout: The paper's 75-point layout pins (``doper.speedplus.keypoints.build_layout``):
            ``sha256`` of the ``keypoints_3d.json`` file and ``points_sha256``, the identity.
            The YAML gives them as a mapping.
        source: The recipe file, when loaded from one.
    """

    kind: ClassVar[str] = "speedplus"
    name: str
    profile: str
    score: ScoreDefinition = "paper"
    mask_mode: str = PAPER_MASK_MODE
    legacy_checkpoints: dict[str, str] = field(default_factory=dict)
    layout: LayoutPins = PAPER_LAYOUT
    source: Path | None = None

    def __post_init__(self) -> None:
        """Refuse an unknown mask mode; parse the layout pins from their YAML mapping.

        Raises:
            ValueError: If ``mask_mode`` is not one of ``MASK_MODES``, or ``layout`` is not a
                mapping of exactly ``sha256`` and ``points_sha256`` SHA-256 values.
        """
        if self.mask_mode not in MASK_MODES:
            raise ValueError(f"mask_mode must be one of {MASK_MODES}, got {self.mask_mode!r}")
        if isinstance(self.layout, Mapping):
            keys = {"sha256", "points_sha256"}
            if set(self.layout) != keys:
                raise ValueError(
                    f"layout must name exactly {sorted(keys)}, got {sorted(self.layout)}"
                )
            object.__setattr__(self, "layout", LayoutPins(**self.layout))
        elif not isinstance(self.layout, LayoutPins):
            raise ValueError(f"layout must be a mapping, got {type(self.layout).__name__}")
