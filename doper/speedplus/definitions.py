# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SPEED score definitions, shared by the recipes and ``doper.speedplus.metrics``.

Plain Python, so recipe parsing needs no numpy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

ScoreDefinition = Literal["paper", "official"]
SCORE_DEFINITIONS: tuple[str, ...] = ("paper", "official")

# Conversion mask modes (``doper.speedplus.data.convert``). ``hull`` is the paper's: the convex
# hull of the projected 64 points. ``auto`` uses installed official SPNv2 masks where present,
# ``official`` requires them and ``none`` writes no mask.
MASK_MODES: tuple[str, ...] = ("hull", "auto", "official", "none")
PAPER_MASK_MODE = "hull"

# SPEED+ splits: synthetic train and validation, and the two real domains.
SPLIT_NAMES: tuple[str, ...] = ("train", "val", "lightbox", "sunlamp")

# The 75-point layout (``doper.speedplus.keypoints``): the 64-point conversion's directory,
# the 75-point output, the tracked-landmark extractor's output and the extractor's seed.
DERIVED_64 = "derived/tango_64"
DERIVED_75 = "derived/tango_75"
TRACKED_DIR = "derived/tracked_landmarks"
TRACKED_SEED = 20260826
# The paper launcher built the 75-point layout for synthetic train and validation only, which
# keeps real-domain ground-truth keypoints out of the training root.
LAYOUT_SPLITS: tuple[str, ...] = ("train", "val")
# The paper's 75-point ``keypoints_3d.json``: the SHA-256 of the file (what SPEED+ checkpoints
# record as ``speedplus_layout_sha256``) and its points hash, the SHA-256 of the 75x3
# little-endian float64 array (the layout's identity, embedded in the file). Confirm the points
# hash on the first real build.
PAPER_LAYOUT_SHA256 = "a6697cafc62bdf5286fb8af453e44bae881b4ad481b8928edf288c85f77fbfa6"
PAPER_LAYOUT_POINTS_SHA256 = "4c78a38de43fbe96ed2639749d95eafaa6a5db93f1de68f8ebdb3fae97c8ef0e"
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class LayoutPins:
    """The pinned identity of a 75-point layout.

    Attributes:
        sha256: SHA-256 of the ``keypoints_3d.json`` file. A layout whose points match but
            whose file differs is a formatting deviation, recorded but not refused.
        points_sha256: SHA-256 of the 75x3 little-endian float64 points: the identity.
    """

    sha256: str
    points_sha256: str

    def __post_init__(self) -> None:
        """Require two lowercase hex SHA-256 values.

        Raises:
            ValueError: Naming the field that is not a SHA-256.
        """
        for name in ("sha256", "points_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise ValueError(f"layout {name} must be a lowercase hex SHA-256, got {value!r}")


PAPER_LAYOUT = LayoutPins(sha256=PAPER_LAYOUT_SHA256, points_sha256=PAPER_LAYOUT_POINTS_SHA256)
# The points hash of each layout file a checkpoint registry or a checkpoint can name by its file
# SHA-256 alone: file SHA-256 -> points hash. A layout whose points hash matches is the same
# layout, even when its file SHA-256 differs (a formatting deviation, recorded but not refused).
KNOWN_LAYOUT_POINTS: dict[str, str] = {PAPER_LAYOUT.sha256: PAPER_LAYOUT.points_sha256}
