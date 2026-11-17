# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Photometric + sensor augmentation for the DOPER dataset.

These transforms target the sim-to-real *appearance* gap: brightness/contrast,
HSV jitter, Gaussian sensor noise, Gaussian/motion blur, and mild JPEG
artifacts. They are appearance-only - geometry is never changed - so boxes and
keypoints stay valid and the same routine can be applied independently to the
full detection image and to the keypoint crop.

All functions operate on HWC uint8 RGB arrays and take an explicit
``numpy.random.Generator`` so workers can be decorrelated.
"""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np

from .config import AugConfig

# Lens-flare tuning ranges, in field units relative to the (0, 1) field square
# unless noted. Intensities are in uint8 gray levels before the strength factor.
_EPS = 1e-6
_FLARE_STRENGTH_RANGE = (0.25, 1.0)
_FLARE_TOP_EDGE_PROB = 0.8
_FLARE_TOP_SOURCE_X_RANGE = (-0.15, 1.15)
_FLARE_TOP_SOURCE_Y_RANGE = (-0.30, 0.08)
_FLARE_SIDE_SOURCE_X_LEFT_RANGE = (-0.25, -0.02)
_FLARE_SIDE_SOURCE_X_RIGHT_RANGE = (1.02, 1.25)
_FLARE_SIDE_SOURCE_Y_RANGE = (-0.10, 0.85)
_FLARE_BLACK_LIFT_RANGE = (18.0, 38.0)
_FLARE_VEIL_SCALE_RANGE = (0.45, 1.0)
_FLARE_VEIL_AMPLITUDE_RANGE = (14.0, 48.0)
_FLARE_RAY_COUNT_RANGE = (2, 6)
_FLARE_RAY_ANGLE_STD = 0.18
_FLARE_RAY_WIDTH_RANGE = (0.008, 0.045)
_FLARE_RAY_AMPLITUDE_RANGE = (7.0, 30.0)
_FLARE_RAY_FALLOFF_RANGE = (0.7, 1.6)
_FLARE_GHOST_COUNT_RANGE = (1, 5)
_FLARE_GHOST_POSITION_RANGE = (0.35, 1.45)
_FLARE_GHOST_SIGMA_ALONG_RANGE = (0.025, 0.11)
_FLARE_GHOST_SIGMA_ACROSS_RANGE = (0.015, 0.07)
_FLARE_GHOST_AMPLITUDE_RANGE = (5.0, 28.0)
_FLARE_BLOOM_THRESHOLD = 175.0
_FLARE_BLOOM_SPAN = 80.0
_FLARE_BLOOM_SIGMA_RANGE = (10.0, 36.0)
_FLARE_BLOOM_AMPLITUDE_RANGE = (30.0, 85.0)
_HUE_DEGREES = 180.0


def lens_flare_augment(img: np.ndarray, cfg: AugConfig, rng: np.random.Generator) -> np.ndarray:
    """Add camera-space veiling glare without changing image geometry.

    The layer combines a lifted black level, broad veil, directional rays,
    internal-reflection ghosts, and bloom from already-bright pixels. It is
    deliberately grayscale-safe and additive: real flare raises dark pixels,
    whereas a multiplicative brightness gradient leaves black pixels black.
    """
    if not cfg.lens_flare:
        return img
    if not 0.0 <= cfg.lens_flare_prob <= 1.0:
        raise ValueError("lens_flare_prob must be in [0, 1]")
    if not 0.0 <= cfg.lens_flare_max_strength <= 1.0:
        raise ValueError("lens_flare_max_strength must be in [0, 1]")
    if not 0.0 < cfg.lens_flare_field_scale <= 1.0:
        raise ValueError("lens_flare_field_scale must be in (0, 1]")
    if rng.random() >= cfg.lens_flare_prob:
        return img
    return _lens_flare(
        img,
        cfg.lens_flare_max_strength,
        rng,
        field_scale=cfg.lens_flare_field_scale,
    )


def _lens_flare(
    img: np.ndarray,
    max_strength: float,
    rng: np.random.Generator,
    field_scale: float = 1.0,
) -> np.ndarray:
    if max_strength == 0.0:
        return img
    if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
        raise ValueError("lens flare expects an HWC uint8 RGB image")
    if not 0.0 < field_scale <= 1.0:
        raise ValueError("field_scale must be in (0, 1]")

    height, width = img.shape[:2]
    field_height = max(2, int(round(height * field_scale)))
    field_width = max(2, int(round(width * field_scale)))
    yy, xx = np.mgrid[0:field_height, 0:field_width].astype(np.float32)
    nx = xx / max(field_width - 1, 1)
    ny = yy / max(field_height - 1, 1)
    strength = float(rng.uniform(*_FLARE_STRENGTH_RANGE) * max_strength)

    # SPEED+ Sunlamp most often enters at or just beyond the upper edge.
    if rng.random() < _FLARE_TOP_EDGE_PROB:
        source_x = float(rng.uniform(*_FLARE_TOP_SOURCE_X_RANGE))
        source_y = float(rng.uniform(*_FLARE_TOP_SOURCE_Y_RANGE))
    else:
        side = int(rng.integers(0, 2))
        source_x = float(
            rng.uniform(*_FLARE_SIDE_SOURCE_X_LEFT_RANGE)
            if side == 0
            else rng.uniform(*_FLARE_SIDE_SOURCE_X_RIGHT_RANGE)
        )
        source_y = float(rng.uniform(*_FLARE_SIDE_SOURCE_Y_RANGE))

    dx = nx - source_x
    dy = ny - source_y
    radius = np.sqrt(dx * dx + dy * dy) + _EPS
    source_to_center = np.asarray([0.5 - source_x, 0.5 - source_y])
    source_to_center /= max(float(np.linalg.norm(source_to_center)), _EPS)
    central_angle = float(np.arctan2(source_to_center[1], source_to_center[0]))
    angle = np.arctan2(dy, dx)

    black_lift = float(rng.uniform(*_FLARE_BLACK_LIFT_RANGE)) * strength
    flare = np.full((field_height, field_width), black_lift, dtype=np.float32)
    veil_scale = float(rng.uniform(*_FLARE_VEIL_SCALE_RANGE))
    veil_amplitude = float(rng.uniform(*_FLARE_VEIL_AMPLITUDE_RANGE)) * strength
    flare += veil_amplitude * np.exp(-0.5 * (radius / veil_scale) ** 2)

    for _ in range(int(rng.integers(*_FLARE_RAY_COUNT_RANGE))):
        ray_angle = central_angle + float(rng.normal(0.0, _FLARE_RAY_ANGLE_STD))
        angular_distance = np.arctan2(np.sin(angle - ray_angle), np.cos(angle - ray_angle))
        width_radians = float(rng.uniform(*_FLARE_RAY_WIDTH_RANGE))
        amplitude = float(rng.uniform(*_FLARE_RAY_AMPLITUDE_RANGE)) * strength
        falloff = float(rng.uniform(*_FLARE_RAY_FALLOFF_RANGE))
        forward = np.cos(angle - ray_angle) > 0.0
        ray = np.exp(-0.5 * (angular_distance / width_radians) ** 2)
        ray *= np.exp(-radius / falloff) * forward
        flare += amplitude * ray

    axis_x, axis_y = map(float, source_to_center)
    perpendicular_x, perpendicular_y = -axis_y, axis_x
    for _ in range(int(rng.integers(*_FLARE_GHOST_COUNT_RANGE))):
        position = float(rng.uniform(*_FLARE_GHOST_POSITION_RANGE))
        center_x = source_x + position * (0.5 - source_x)
        center_y = source_y + position * (0.5 - source_y)
        along = (nx - center_x) * axis_x + (ny - center_y) * axis_y
        across = (nx - center_x) * perpendicular_x + (ny - center_y) * perpendicular_y
        sigma_along = float(rng.uniform(*_FLARE_GHOST_SIGMA_ALONG_RANGE))
        sigma_across = float(rng.uniform(*_FLARE_GHOST_SIGMA_ACROSS_RANGE))
        amplitude = float(rng.uniform(*_FLARE_GHOST_AMPLITUDE_RANGE)) * strength
        ghost = np.exp(-0.5 * ((along / sigma_along) ** 2 + (across / sigma_across) ** 2))
        flare += amplitude * ghost

    field_image = (
        img
        if field_height == height and field_width == width
        else cv2.resize(
            img,
            (field_width, field_height),
            interpolation=cv2.INTER_AREA,
        )
    )
    gray = cv2.cvtColor(field_image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    highlights = np.clip((gray - _FLARE_BLOOM_THRESHOLD) / _FLARE_BLOOM_SPAN, 0.0, 1.0)
    bloom_sigma = float(rng.uniform(*_FLARE_BLOOM_SIGMA_RANGE)) * field_scale
    bloom = cv2.GaussianBlur(highlights, (0, 0), bloom_sigma)
    flare += float(rng.uniform(*_FLARE_BLOOM_AMPLITUDE_RANGE)) * strength * bloom

    if field_height != height or field_width != width:
        flare = cv2.resize(flare, (width, height), interpolation=cv2.INTER_CUBIC)
    output = np.clip(img.astype(np.float32) + flare[..., None], 0, 255)
    return np.ascontiguousarray(np.rint(output).astype(np.uint8))


def photometric_augment(img: np.ndarray, cfg: AugConfig, rng: np.random.Generator) -> np.ndarray:
    """Apply the full appearance pipeline to an HWC uint8 RGB image."""
    out = img
    if cfg.directional_brightness:
        apply_directional_brightness = rng.random() < cfg.directional_brightness_prob
        # Draw a child seed unconditionally while the feature is enabled. This
        # keeps every later photometric/geometric RNG draw identical between a
        # zero-strength control and an active-gradient ablation.
        directional_seed = int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
        if apply_directional_brightness:
            out = _directional_brightness(out, cfg, np.random.default_rng(directional_seed))
    if rng.random() < cfg.photo_prob:
        out = _brightness_contrast(out, cfg, rng)
        out = _hsv_jitter(out, cfg, rng)
    if rng.random() < cfg.grayscale_prob:
        out = _grayscale(out)
    if rng.random() < cfg.blur_prob:
        out = _gaussian_blur(out, cfg, rng)
    if rng.random() < cfg.motion_blur_prob:
        out = _motion_blur(out, cfg, rng)
    if rng.random() < cfg.noise_prob:
        out = _gaussian_noise(out, cfg, rng)
    if rng.random() < cfg.jpeg_prob:
        out = _jpeg(out, cfg, rng)
    return np.ascontiguousarray(out)


def _directional_brightness(
    img: np.ndarray, cfg: AugConfig, rng: np.random.Generator
) -> np.ndarray:
    """Multiply RGB by a smooth, randomly oriented illumination field.

    The sampled strength ``s`` produces gains in ``[1-s, 1+s]`` across the
    image. The same gain is applied to every channel, so this adds no color
    cast. Multiplication also keeps exact-black space backgrounds black.
    """
    max_strength = float(cfg.directional_brightness_max_strength)
    if not 0.0 <= max_strength <= 1.0:
        raise ValueError(
            f"directional_brightness_max_strength must be in [0, 1], got {max_strength}"
        )
    if max_strength == 0.0:
        return img

    height, width = img.shape[:2]
    angle = rng.uniform(0.0, 2.0 * np.pi)
    direction_x = float(np.cos(angle))
    direction_y = float(np.sin(angle))
    x = np.linspace(-0.5, 0.5, width, dtype=np.float32)[None, :]
    y = np.linspace(-0.5, 0.5, height, dtype=np.float32)[:, None]
    extent = 0.5 * (abs(direction_x) + abs(direction_y))
    projected = (direction_x * x + direction_y * y) / max(extent, _EPS)
    position = np.clip((projected + 1.0) * 0.5, 0.0, 1.0)
    smooth_position = position * position * (3.0 - 2.0 * position)

    strength = float(rng.uniform(0.0, max_strength))
    gain = 1.0 + strength * (2.0 * smooth_position - 1.0)
    out = img.astype(np.float32) * gain[..., None]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def clahe_luminance(img: np.ndarray, clip_limit: float, grid_size: int) -> np.ndarray:
    """Apply CLAHE to LAB luminance while preserving RGB chroma channels."""
    if clip_limit <= 0.0:
        raise ValueError("CLAHE clip_limit must be positive")
    if grid_size < 1:
        raise ValueError("CLAHE grid_size must be at least one")
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    luminance, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(grid_size), int(grid_size)),
    )
    luminance = clahe.apply(luminance)
    return np.ascontiguousarray(
        cv2.cvtColor(
            cv2.merge((luminance, channel_a, channel_b)),
            cv2.COLOR_LAB2RGB,
        )
    )


def _brightness_contrast(img: np.ndarray, cfg: AugConfig, rng: np.random.Generator) -> np.ndarray:
    contrast = 1.0 + rng.uniform(-cfg.contrast, cfg.contrast)
    brightness = rng.uniform(-cfg.brightness, cfg.brightness) * 255.0
    out = (img.astype(np.float32) - 128.0) * contrast + 128.0 + brightness
    return np.clip(out, 0, 255).astype(np.uint8)


def _hsv_jitter(img: np.ndarray, cfg: AugConfig, rng: np.random.Generator) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-cfg.hue, cfg.hue) * _HUE_DEGREES) % _HUE_DEGREES
    hsv[..., 1] = np.clip(
        hsv[..., 1] * (1.0 + rng.uniform(-cfg.saturation, cfg.saturation)), 0, 255
    )
    hsv[..., 2] = np.clip(hsv[..., 2] * (1.0 + rng.uniform(-cfg.value, cfg.value)), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _grayscale(img: np.ndarray) -> np.ndarray:
    """Fully desaturate to luma and replicate back to 3 RGB channels."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def random_erase(img: np.ndarray, cfg: AugConfig, rng: np.random.Generator) -> np.ndarray:
    """Fill ``cfg.det_erase_count`` random rectangles with uniform noise (cutout).

    Pixels only - callers keep the boxes/mask unchanged so this teaches occlusion
    robustness, not geometry.

    Args:
        img: HWC uint8 image.
        cfg: Augmentation config.
        rng: Random generator.

    Returns:
        A copy of ``img`` with the rectangles filled.
    """
    out = img.copy()
    h, w = out.shape[:2]
    chans = out.shape[2] if out.ndim == 3 else 1
    area = float(h * w)
    for _ in range(max(1, cfg.det_erase_count)):
        target = rng.uniform(cfg.det_erase_min_area, cfg.det_erase_max_area) * area
        aspect = rng.uniform(0.3, 3.3)
        eh = int(round(np.sqrt(target / aspect)))
        ew = int(round(np.sqrt(target * aspect)))
        if eh < 1 or ew < 1 or eh >= h or ew >= w:
            continue
        cy = int(rng.integers(0, h - eh))
        cx = int(rng.integers(0, w - ew))
        out[cy : cy + eh, cx : cx + ew] = rng.integers(0, 256, size=(eh, ew, chans), dtype=np.uint8)
    return out


def _gaussian_blur(img: np.ndarray, cfg: AugConfig, rng: np.random.Generator) -> np.ndarray:
    sigma = rng.uniform(0.1, cfg.blur_max_sigma)
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)


def _motion_blur(img: np.ndarray, cfg: AugConfig, rng: np.random.Generator) -> np.ndarray:
    max_k = max(3, cfg.motion_blur_max_kernel)
    k = int(rng.integers(3, max_k + 1)) | 1  # odd kernel size
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    angle = rng.uniform(0, 180)
    rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (k, k))
    s = kernel.sum()
    if s > _EPS:
        kernel /= s
    return cv2.filter2D(img, -1, kernel)


def _gaussian_noise(img: np.ndarray, cfg: AugConfig, rng: np.random.Generator) -> np.ndarray:
    std = rng.uniform(0.0, cfg.noise_max_std)
    noise = rng.normal(0.0, std, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _jpeg(img: np.ndarray, cfg: AugConfig, rng: np.random.Generator) -> np.ndarray:
    quality = int(rng.integers(cfg.jpeg_quality_min, cfg.jpeg_quality_max + 1))
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img
    return cast(np.ndarray, cv2.imdecode(enc, cv2.IMREAD_COLOR))
