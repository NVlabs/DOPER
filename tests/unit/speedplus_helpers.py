# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared SPEED+ test constants: the official Tango landmarks and a SPEED+-like camera."""

import numpy as np

# The 11 official SPEED+ Tango landmarks (meters), as the public baseline publishes them.
OFFICIAL_POINTS = np.asarray(
    [
        [-0.3700, -0.3850, 0.3215],
        [-0.3700, 0.3850, 0.3215],
        [0.3700, 0.3850, 0.3215],
        [0.3700, -0.3850, 0.3215],
        [-0.3700, -0.2640, 0.0000],
        [-0.3700, 0.3040, 0.0000],
        [0.3700, 0.3040, 0.0000],
        [0.3700, -0.2640, 0.0000],
        [-0.5427, 0.4877, 0.2535],
        [0.5427, 0.4877, 0.2591],
        [0.3050, -0.5790, 0.2515],
    ]
)
K = [[2988.6, 0.0, 960.0], [0.0, 2988.3, 600.0], [0.0, 0.0, 1.0]]
DIST = [-0.2238, 0.5141, -0.0007, -0.0002, -0.1312]
