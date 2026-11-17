# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single guarded import of the bayesfilter names used by the pose filters."""

from __future__ import annotations

try:
    from bayesfilter.distributions import Gaussian
    from bayesfilter.filtering import BayesianFilter
    from bayesfilter.model import StateTransitionModel
    from bayesfilter.observation import Observation
    from bayesfilter.smoothing import RTS
    from bayesfilter.unscented import propagate_gaussian_cross_cov
except ImportError as exc:
    raise ImportError("bayesfilter is required for doper.filtering; install doper[filter]") from exc

__all__ = [
    "RTS",
    "BayesianFilter",
    "Gaussian",
    "Observation",
    "StateTransitionModel",
    "propagate_gaussian_cross_cov",
]
