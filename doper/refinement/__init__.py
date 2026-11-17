# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pose refinement against the network's keypoint heatmaps.

``heatmap`` needs only NumPy, OpenCV and SciPy; its default L-BFGS-B optimizer and the parallel
preparation load Numba kernels on first use (``pip install doper[refine]``).
"""

from doper.refinement.heatmap import (
    HeatmapObservation,
    LbfgsbAudit,
    PowellAudit,
    PreparedView,
    RefinementAudit,
    RefineSettings,
    observation_from_detection,
    prepare_observations,
    refine_pose,
)

__all__ = [
    "HeatmapObservation",
    "LbfgsbAudit",
    "PowellAudit",
    "PreparedView",
    "RefineSettings",
    "RefinementAudit",
    "observation_from_detection",
    "prepare_observations",
    "refine_pose",
]
