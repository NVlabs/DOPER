# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image-space keypoint measurements and uncertainty from spatial heatmaps."""

from __future__ import annotations

import numpy as np

from doper.targets import decode_keypoint_heatmaps


def heatmap_keypoint_measurements(
    logits: np.ndarray,
    crop_to_image: np.ndarray,
    crop_size: int,
    *,
    beta: float = 1.0,
    covariance_floor_image_px: float = 1.0,
    dark: bool = True,
) -> dict[str, np.ndarray]:
    """Decode heatmaps and estimate per-keypoint full-image covariance.

    Following Park and D'Amico's measurement construction, normalized heatmap
    intensity is treated as probability mass and its second moment is computed
    about the decoded peak.  The mean and covariance are then transformed from
    heatmap bins through crop coordinates into full-image pixels.

    Args:
        logits: ``(K, H, W)`` square heatmap logits.
        crop_to_image: ``(2, 3)`` affine from crop pixels to full-image pixels.
        crop_size: Side length of the square crop, in pixels.
        beta: Softmax inverse temperature applied to the logits.
        covariance_floor_image_px: Minimum image-space standard deviation, in pixels; the
            covariance eigenvalues are clamped to its square.
        dark: Whether to refine peaks with the DARK-style sub-bin offset.

    Returns:
        A dict of float32 arrays: ``coordinates_image`` and ``coordinates_crop`` ``(K, 2)``,
        ``covariance_image`` and ``covariance_heatmap`` ``(K, 2, 2)``, and ``confidence``
        ``(K,)``.

    Raises:
        ValueError: If the shapes or scalar arguments are invalid.
    """
    values = np.asarray(logits, dtype=np.float64)
    transform = np.asarray(crop_to_image, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError(f"expected square (K,H,W) heatmaps, got {values.shape}")
    if transform.shape != (2, 3):
        raise ValueError(f"crop_to_image must have shape (2,3), got {transform.shape}")
    if crop_size <= 1 or values.shape[-1] <= 1:
        raise ValueError("crop and heatmap dimensions must exceed one")
    if beta <= 0.0 or covariance_floor_image_px < 0.0:
        raise ValueError("beta must be positive and covariance floor nonnegative")

    coordinates_crop, confidence = decode_keypoint_heatmaps(
        values,
        crop_size,
        beta=beta,
        dark=dark,
    )
    homogeneous = np.concatenate(
        (
            coordinates_crop.astype(np.float64),
            np.ones((len(coordinates_crop), 1), dtype=np.float64),
        ),
        axis=1,
    )
    coordinates_image = homogeneous @ transform.T

    keypoint_count, height, width = values.shape
    flat = values.reshape(keypoint_count, -1) * beta
    flat -= flat.max(axis=1, keepdims=True)
    probability = np.exp(flat)
    probability /= np.clip(probability.sum(axis=1, keepdims=True), 1.0e-12, None)
    probability = probability.reshape(keypoint_count, height, width)

    heatmap_scale = (width - 1) / float(crop_size - 1)
    peaks_heatmap = coordinates_crop.astype(np.float64) * heatmap_scale
    grid_y, grid_x = np.mgrid[:height, :width]
    crop_scale = (crop_size - 1) / float(width - 1)
    linear = transform[:, :2] * crop_scale
    covariance_image = np.zeros((keypoint_count, 2, 2), dtype=np.float64)
    covariance_heatmap = np.zeros_like(covariance_image)
    floor_variance = covariance_floor_image_px**2
    for index in range(keypoint_count):
        delta_x = grid_x - peaks_heatmap[index, 0]
        delta_y = grid_y - peaks_heatmap[index, 1]
        mass = probability[index]
        covariance = np.asarray(
            [
                [
                    np.sum(mass * delta_x * delta_x),
                    np.sum(mass * delta_x * delta_y),
                ],
                [
                    np.sum(mass * delta_x * delta_y),
                    np.sum(mass * delta_y * delta_y),
                ],
            ],
            dtype=np.float64,
        )
        covariance_heatmap[index] = covariance
        image_covariance = linear @ covariance @ linear.T
        image_covariance = 0.5 * (image_covariance + image_covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(image_covariance)
        eigenvalues = np.maximum(eigenvalues, floor_variance)
        covariance_image[index] = (eigenvectors * eigenvalues) @ eigenvectors.T

    return {
        "coordinates_image": coordinates_image.astype(np.float32),
        "coordinates_crop": coordinates_crop.astype(np.float32),
        "covariance_image": covariance_image.astype(np.float32),
        "covariance_heatmap": covariance_heatmap.astype(np.float32),
        "confidence": confidence.astype(np.float32),
    }
