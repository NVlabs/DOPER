# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from doper.filtering.measurements import heatmap_keypoint_measurements


def test_heatmap_covariance_transforms_to_image_coordinates() -> None:
    grid_y, grid_x = np.mgrid[:17, :17]
    variance_bins = 4.0
    logits = -0.5 * ((grid_x - 8.0) ** 2 + (grid_y - 8.0) ** 2) / variance_bins
    crop_to_image = np.asarray([[2.0, 0.0, 10.0], [0.0, 3.0, 20.0]])
    result = heatmap_keypoint_measurements(
        logits[None],
        crop_to_image,
        crop_size=65,
        covariance_floor_image_px=0.0,
        dark=False,
    )

    np.testing.assert_allclose(result["coordinates_crop"], [[32.0, 32.0]])
    np.testing.assert_allclose(result["coordinates_image"], [[74.0, 116.0]])
    # Four crop pixels per heatmap bin, followed by the 2x/3x image affine.
    np.testing.assert_allclose(
        result["covariance_image"][0],
        np.diag([variance_bins * 8.0**2, variance_bins * 12.0**2]),
        rtol=2.0e-3,
        atol=0.1,
    )


def test_heatmap_covariance_applies_image_space_floor() -> None:
    logits = np.full((1, 9, 9), -100.0, dtype=np.float32)
    logits[0, 4, 4] = 100.0
    result = heatmap_keypoint_measurements(
        logits,
        np.eye(2, 3),
        crop_size=9,
        covariance_floor_image_px=1.5,
        dark=False,
    )
    np.testing.assert_allclose(np.linalg.eigvalsh(result["covariance_image"][0]), [2.25, 2.25])
