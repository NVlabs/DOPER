# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DOPER's PnP RANSAC: OpenCV's sequential USAC with fixed, seeded parameters.

USAC is the default everywhere. EPnP RANSAC exists only as an explicit opt-in
(``pnp_method: epnp``) so the paper's multi-view and pseudo-label recipes reproduce exactly.
Torch-free, so the multi-view geometry can use it without importing the model code.
"""

from __future__ import annotations

from typing import Literal, get_args

import cv2
import numpy as np

PnpMethod = Literal["usac", "epnp"]
PNP_METHODS: tuple[str, ...] = get_args(PnpMethod)

DEFAULT_PNP_CONFIDENCE = 0.99
DEFAULT_PNP_MAX_ITERATIONS = 4000
DEFAULT_PNP_REPROJECTION_ERROR_PX = 3.0
DEFAULT_PNP_RANDOM_SEED = 42


def usac_pnp_ransac(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray | None,
    threshold_px: float = DEFAULT_PNP_REPROJECTION_ERROR_PX,
) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray | None]:
    """Run DOPER's PnP RANSAC: OpenCV's sequential USAC with fixed, seeded parameters.

    Args:
        object_points: ``(N, 3)`` object-frame points.
        image_points: ``(N, 2)`` image points, ordered like ``object_points``.
        camera_matrix: ``(3, 3)`` camera intrinsics.
        dist_coeffs: Distortion coefficients, or None for none.
        threshold_px: Inlier reprojection threshold in pixels.

    Returns:
        ``(ok, rvec, tvec, inliers)`` as returned by ``cv2.solvePnPRansac``.

    Raises:
        RuntimeError: If OpenCV lacks ``cv2.UsacParams``.
    """
    if not hasattr(cv2, "UsacParams"):
        raise RuntimeError("DOPER PnP requires an OpenCV build with cv2.UsacParams support")
    params = cv2.UsacParams()
    params.confidence = DEFAULT_PNP_CONFIDENCE
    params.maxIterations = DEFAULT_PNP_MAX_ITERATIONS
    params.threshold = float(threshold_px)
    params.isParallel = False
    params.randomGeneratorState = DEFAULT_PNP_RANDOM_SEED
    ok, _updated_camera_matrix, rvec, tvec, inliers = cv2.solvePnPRansac(
        np.asarray(object_points, dtype=np.float64),
        np.asarray(image_points, dtype=np.float64),
        np.asarray(camera_matrix, dtype=np.float64),
        dist_coeffs,
        params=params,
    )
    return bool(ok), rvec, tvec, inliers


def pnp_ransac(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray | None,
    threshold_px: float,
    method: PnpMethod = "usac",
    epnp_iterations: int = 100,
) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray | None]:
    """Run PnP RANSAC with ``method``: USAC (the default) or the paper's EPnP RANSAC.

    Args:
        object_points: ``(N, 3)`` object-frame points.
        image_points: ``(N, 2)`` image points, ordered like ``object_points``.
        camera_matrix: ``(3, 3)`` camera intrinsics.
        dist_coeffs: Distortion coefficients, or None for none.
        threshold_px: Inlier reprojection threshold in pixels.
        method: ``usac``, or ``epnp`` to reproduce the paper's recipes.
        epnp_iterations: RANSAC iterations of the ``epnp`` method.

    Returns:
        ``(ok, rvec, tvec, inliers)`` as returned by ``cv2.solvePnPRansac``.

    Raises:
        ValueError: If ``method`` is not one of ``PNP_METHODS``.
    """
    if method == "usac":
        return usac_pnp_ransac(
            object_points, image_points, camera_matrix, dist_coeffs, threshold_px=threshold_px
        )
    if method != "epnp":
        raise ValueError(f"pnp_method must be one of {PNP_METHODS}, got {method!r}")
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        np.asarray(object_points, dtype=np.float64),
        np.asarray(image_points, dtype=np.float64),
        camera_matrix,
        dist_coeffs,
        reprojectionError=float(threshold_px),
        iterationsCount=epnp_iterations,
        flags=cv2.SOLVEPNP_EPNP,
    )
    return bool(ok), rvec, tvec, inliers
