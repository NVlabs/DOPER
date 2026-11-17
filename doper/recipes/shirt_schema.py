# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed SHIRT recipe (``kind: shirt``).

The SHIRT recipe evaluates the SPEED+ model on the SHIRT trajectories through the ``doper``
verbs, which route on the recipe's ``kind``. It names its dataset profile, the SPEED score
definition (``doper.speedplus.metrics``), the trajectories and image domain it evaluates and the
SHA-256 of the checkpoint it pins. The Table 2 filter settings are fixed constants of
``doper.shirt.policies`` (the paper's gates and SHIRT-tuned process noise).
``doper.recipes.load_any_recipe`` parses it and refuses unknown keys. Nothing here imports
numpy or torch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from ..shirt import DOMAINS, PAPER_DOMAIN, TRAJECTORIES
from ..speedplus.definitions import ScoreDefinition

_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ShirtRecipe:
    """The SHIRT recipe.

    Attributes:
        kind: ``shirt`` (class constant; the YAML must carry it).
        name: Recipe name.
        profile: Packaged dataset profile name.
        score: Default SPEED score definition, ``paper`` or ``official``.
        legacy_checkpoints: Source-format SPEED+ checkpoints, which record no keypoint layout,
            accepted by their file SHA-256 together with the SHA-256 of the
            ``keypoints_3d.json`` they were trained on (``doper.speedplus.checkpoints``).
        trajectories: Trajectories evaluated by default. The YAML gives a list.
        domain: Image domain evaluated by default (``synthetic`` or ``lightbox``).
        checkpoint_sha256: SHA-256 of the checkpoint the recipe pins; any other checkpoint
            needs ``--allow-unpinned-checkpoint``, which the summaries record. None pins none.
        source: The recipe file, when loaded from one.
    """

    kind: ClassVar[str] = "shirt"
    name: str
    profile: str
    score: ScoreDefinition = "paper"
    legacy_checkpoints: dict[str, str] = field(default_factory=dict)
    trajectories: tuple[str, ...] = TRAJECTORIES
    domain: str = PAPER_DOMAIN
    checkpoint_sha256: str | None = None
    source: Path | None = None

    def __post_init__(self) -> None:
        """Normalize the trajectories to a tuple and check the domain and checkpoint pin.

        Raises:
            ValueError: For an empty trajectory list or a non-string trajectory, an unknown
                domain, or a pin that is not a lowercase hex SHA-256.
        """
        trajectories = self.trajectories
        if (
            isinstance(trajectories, str)
            or not trajectories
            or not all(isinstance(t, str) and t for t in trajectories)
        ):
            raise ValueError(
                f"trajectories must be a non-empty list of names, got {trajectories!r}"
            )
        object.__setattr__(self, "trajectories", tuple(trajectories))
        if self.domain not in DOMAINS:
            raise ValueError(f"domain must be one of {DOMAINS}, got {self.domain!r}")
        pin = self.checkpoint_sha256
        if pin is not None and not (isinstance(pin, str) and _SHA256.fullmatch(pin)):
            raise ValueError(f"checkpoint_sha256 must be a lowercase hex SHA-256, got {pin!r}")
