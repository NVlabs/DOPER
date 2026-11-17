# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numba kernels for heatmap refinement (the ``refine`` extra).

``doper.refinement.heatmap`` imports this module only for the L-BFGS-B optimizer or the parallel
preparation, so the rest of the package works without numba. The kernel bodies keep a fixed
operation order: changing it changes the optimization path.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

DEPTH_EPSILON_M = 1e-5


@njit(cache=True, fastmath=False)
def sample_gradient(
    xyz: np.ndarray,
    jac: np.ndarray,
    rel: np.ndarray,
    P: np.ndarray,  # noqa: N803
    maps: np.ndarray,
    invalid: float,
    gradient: np.ndarray,
) -> float:
    """Negative bilinear log-probability of one view and its exact piecewise gradient.

    Args:
        xyz: ``(N, 3)`` keypoints in the reference camera, meters.
        jac: ``(3, N, 3)`` derivative of ``xyz`` with respect to the rotation vector.
        rel: ``(4, 4)`` reference-to-view transform, meters.
        P: ``(3, 3)`` camera-to-heatmap projection.
        maps: ``(N, H, W)`` float64 log-probability maps.
        invalid: Log-probability charged to a keypoint behind the camera or outside the map.
        gradient: ``(6,)`` accumulator (rotation vector, then translation), updated in place.

    Returns:
        The summed negative log-probability of this view.
    """
    n, h, w = maps.shape
    loss = 0.0
    for i in range(n):
        c = np.empty(3)
        for a in range(3):
            c[a] = rel[a, 0] * xyz[i, 0] + rel[a, 1] * xyz[i, 1] + rel[a, 2] * xyz[i, 2] + rel[a, 3]
        if c[2] <= DEPTH_EPSILON_M:
            loss -= invalid
            continue
        u = (P[0, 0] * c[0] + P[0, 1] * c[1] + P[0, 2] * c[2]) / c[2]
        v = (P[1, 0] * c[0] + P[1, 1] * c[1] + P[1, 2] * c[2]) / c[2]
        if not np.isfinite(u) or not np.isfinite(v) or u < 0 or v < 0 or u > w - 1 or v > h - 1:
            loss -= invalid
            continue
        x0 = int(np.floor(u))
        y0 = int(np.floor(v))
        x1 = min(x0 + 1, w - 1)
        y1 = min(y0 + 1, h - 1)
        dx = u - x0
        dy = v - y0
        a = maps[i, y0, x0]
        b = maps[i, y0, x1]
        cc = maps[i, y1, x0]
        d = maps[i, y1, x1]
        loss -= (1 - dx) * (1 - dy) * a + dx * (1 - dy) * b + (1 - dx) * dy * cc + dx * dy * d
        gx = -((1 - dy) * (b - a) + dy * (d - cc))
        gy = -((1 - dx) * (cc - a) + dx * (d - b))
        dc = np.empty(3)
        dr = np.empty(3)
        for k in range(3):
            dc[k] = (gx * P[0, k] + gy * P[1, k]) / c[2]
        dc[2] -= (gx * u + gy * v) / c[2]
        for k in range(3):
            dr[k] = dc[0] * rel[0, k] + dc[1] * rel[1, k] + dc[2] * rel[2, k]
            gradient[3 + k] += dr[k]
        for k in range(3):
            gradient[k] += dr[0] * jac[k, i, 0] + dr[1] * jac[k, i, 1] + dr[2] * jac[k, i, 2]
    return loss


@njit(cache=True, parallel=True)
def fused_log_softmax(x: np.ndarray, beta: float) -> np.ndarray:
    """Per-channel ``log_softmax(beta * x)`` as float64, one channel per thread.

    Each channel's max and sum stay sequential, so the result does not depend on the thread
    count; it differs from the NumPy preparation by at most ~1e-12.
    """
    n, h, w = x.shape
    out = np.empty((n, h, w), np.float64)
    for k in prange(n):
        mx = -np.inf
        for y in range(h):
            for z in range(w):
                mx = max(mx, float(x[k, y, z]) * beta)
        s = 0.0
        for y in range(h):
            for z in range(w):
                v = float(x[k, y, z]) * beta - mx
                out[k, y, z] = v
                s += np.exp(v)
        log_sum = np.log(s)
        for y in range(h):
            for z in range(w):
                out[k, y, z] -= log_sum
    return out


@contextmanager
def limited_threads(threads: int) -> Iterator[None]:
    """Run with at most ``threads`` Numba threads, restoring the previous count afterwards."""
    previous = get_num_threads()
    set_num_threads(min(threads, previous))
    try:
        yield
    finally:
        set_num_threads(previous)
