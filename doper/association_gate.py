# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The thresholds of multi-view detection association (``doper.multiview``).

Kept free of OpenCV and SciPy so recipes can name a gate without importing the geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class AssociationGate:
    """Thresholds a candidate A/B detection pair must pass.

    Attributes:
        kp_score_thr: Minimum keypoint confidence for a keypoint to count as confident.
        min_keypoints: Minimum confident keypoints a detection needs to be considered.
        ransac_px: Single-view RANSAC reprojection threshold in pixels (outlier rejection).
        max_reprojection_px: Largest per-camera median reprojection error of the joint pose
            for the pair to be marked ``accepted``.
        min_depth: Smallest plausible object depth in both cameras, in the caller's length
            units (those of the keypoints and rig translations: meters for the robot rigs,
            millimeters for BOP).
        max_depth: Largest plausible object depth in both cameras, in the same units.
        pnp_method: Single-view PnP RANSAC method: ``usac`` (the default), or ``epnp`` for the
            paper's recipes.
    """

    kp_score_thr: float
    min_keypoints: int
    ransac_px: float
    max_reprojection_px: float
    min_depth: float
    max_depth: float
    pnp_method: Literal["usac", "epnp"] = "usac"

    def __post_init__(self) -> None:
        """Reject an unknown PnP method."""
        if self.pnp_method not in ("usac", "epnp"):
            raise ValueError(f"pnp_method must be usac or epnp, got {self.pnp_method!r}")
