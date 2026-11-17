# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The strict pseudo-label gate, kept free of torch and OpenCV so recipes can bind it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class GateSpec:
    """Thresholds of the strict pseudo-label gate.

    Attributes:
        detector_threshold: Score floor handed to the detector.
        score: A detection must score strictly above this.
        keypoint_score: Keypoints scoring at least this enter the RANSAC PnP.
        max_reprojection_px: PnP reprojection error must be strictly below this.
        min_inliers: RANSAC inliers must be strictly above this.
        ransac_px: RANSAC reprojection threshold.
        mask_probability: Mask probability at or above this is foreground.
        min_area_fraction: The mask must cover at least this fraction of the image.
        box_pad: Fraction of the box size the mask may extend beyond the box.
        pnp_method: RANSAC PnP method: ``usac`` (the default), or ``epnp`` for the paper's
            recipes.
    """

    detector_threshold: float = 0.1
    score: float = 0.5
    keypoint_score: float = 0.3
    max_reprojection_px: float = 3.0
    min_inliers: int = 50
    ransac_px: float = 8.0
    mask_probability: float = 0.5
    min_area_fraction: float = 0.002
    box_pad: float = 0.05
    pnp_method: Literal["usac", "epnp"] = "usac"

    def __post_init__(self) -> None:
        """Reject an unknown PnP method."""
        if self.pnp_method not in ("usac", "epnp"):
            raise ValueError(f"pnp_method must be usac or epnp, got {self.pnp_method!r}")

    def as_manifest(self) -> dict[str, Any]:
        """Return the gate as ``manifest.json`` records it (the audited thresholds and method).

        The paper's pools used EPnP and recorded no method, so an ``epnp`` gate omits it and
        those manifests still match; a ``usac`` gate records ``pnp_method``.
        """
        manifest = {
            "score_gt": self.score,
            "reprojection_lt_px": self.max_reprojection_px,
            "ransac_inliers_gt": self.min_inliers,
            "ransac_px": self.ransac_px,
            "kp_score_thr": self.keypoint_score,
        }
        if self.pnp_method != "epnp":
            manifest["pnp_method"] = self.pnp_method
        return manifest


def strict_thresholds(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return a gate manifest's thresholds without its PnP method."""
    return {key: value for key, value in manifest.items() if key != "pnp_method"}


STRICT_GATE = strict_thresholds(GateSpec().as_manifest())
