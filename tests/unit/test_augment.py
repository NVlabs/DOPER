# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

import doper.augment as augment
from doper.config import AugConfig


def _gradient_only_config(**overrides) -> AugConfig:
    values = {
        "directional_brightness": True,
        "directional_brightness_prob": 1.0,
        "directional_brightness_max_strength": 0.55,
        "photo_prob": 0.0,
        "grayscale_prob": 0.0,
        "blur_prob": 0.0,
        "motion_blur_prob": 0.0,
        "noise_prob": 0.0,
        "jpeg_prob": 0.0,
    }
    values.update(overrides)
    return AugConfig(**values)


def test_directional_brightness_is_grayscale_safe_and_keeps_black_black() -> None:
    image = np.full((101, 101, 3), 100, dtype=np.uint8)
    image[50, 50] = 0
    result = augment.photometric_augment(image, _gradient_only_config(), np.random.default_rng(7))

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert result.flags.c_contiguous
    np.testing.assert_array_equal(result[..., 0], result[..., 1])
    np.testing.assert_array_equal(result[..., 1], result[..., 2])
    np.testing.assert_array_equal(result[50, 50], np.zeros(3, dtype=np.uint8))
    assert int(result.max()) > 100
    assert int(result.min()) < 100


def test_directional_brightness_disabled_preserves_existing_pipeline() -> None:
    image = np.arange(12 * 9 * 3, dtype=np.uint8).reshape(12, 9, 3)
    cfg = _gradient_only_config(directional_brightness=False)

    result = augment.photometric_augment(image, cfg, np.random.default_rng(3))

    np.testing.assert_array_equal(result, image)


def test_directional_brightness_runs_before_other_photometric_transforms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def record(name):
        def transform(image, _cfg=None, _rng=None):
            calls.append(name)
            return image

        return transform

    monkeypatch.setattr(augment, "_directional_brightness", record("gradient"))
    monkeypatch.setattr(augment, "_brightness_contrast", record("brightness"))
    monkeypatch.setattr(augment, "_hsv_jitter", record("hsv"))
    monkeypatch.setattr(augment, "_grayscale", record("grayscale"))
    monkeypatch.setattr(augment, "_gaussian_blur", record("gaussian_blur"))
    monkeypatch.setattr(augment, "_motion_blur", record("motion_blur"))
    monkeypatch.setattr(augment, "_gaussian_noise", record("noise"))
    monkeypatch.setattr(augment, "_jpeg", record("jpeg"))
    cfg = AugConfig(
        directional_brightness=True,
        directional_brightness_prob=1.0,
        photo_prob=1.0,
        grayscale_prob=1.0,
        blur_prob=1.0,
        motion_blur_prob=1.0,
        noise_prob=1.0,
        jpeg_prob=1.0,
    )

    augment.photometric_augment(np.zeros((8, 8, 3), dtype=np.uint8), cfg, np.random.default_rng(5))

    assert calls == [
        "gradient",
        "brightness",
        "hsv",
        "grayscale",
        "gaussian_blur",
        "motion_blur",
        "noise",
        "jpeg",
    ]


def test_directional_brightness_strength_does_not_shift_parent_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downstream_draws: list[float] = []

    def record_brightness(image, _cfg, rng):
        downstream_draws.append(float(rng.random()))
        return image

    monkeypatch.setattr(augment, "_brightness_contrast", record_brightness)
    monkeypatch.setattr(augment, "_hsv_jitter", lambda image, _cfg, _rng: image)
    image = np.full((17, 19, 3), 100, dtype=np.uint8)
    common = {
        "directional_brightness": True,
        "directional_brightness_prob": 1.0,
        "photo_prob": 1.0,
        "grayscale_prob": 0.0,
        "blur_prob": 0.0,
        "motion_blur_prob": 0.0,
        "noise_prob": 0.0,
        "jpeg_prob": 0.0,
    }

    control = augment.photometric_augment(
        image,
        AugConfig(**common, directional_brightness_max_strength=0.0),
        np.random.default_rng(29),
    )
    active = augment.photometric_augment(
        image,
        AugConfig(**common, directional_brightness_max_strength=0.55),
        np.random.default_rng(29),
    )

    assert downstream_draws[0] == downstream_draws[1]
    np.testing.assert_array_equal(control, image)
    assert not np.array_equal(active, image)


@pytest.mark.parametrize("strength", [-0.01, 1.01])
def test_directional_brightness_rejects_invalid_strength(strength: float) -> None:
    cfg = _gradient_only_config(directional_brightness_max_strength=strength)

    with pytest.raises(ValueError, match="must be in"):
        augment.photometric_augment(
            np.zeros((8, 8, 3), dtype=np.uint8),
            cfg,
            np.random.default_rng(2),
        )


def test_lens_flare_is_deterministic_grayscale_safe_and_geometry_preserving() -> None:
    image = np.zeros((96, 144, 3), dtype=np.uint8)
    image[38:58, 62:82] = 240
    cfg = AugConfig(
        lens_flare=True,
        lens_flare_prob=1.0,
        lens_flare_max_strength=0.8,
    )

    first = augment.lens_flare_augment(image, cfg, np.random.default_rng(17))
    second = augment.lens_flare_augment(image, cfg, np.random.default_rng(17))

    assert first.shape == image.shape
    assert first.dtype == np.uint8
    assert first.flags.c_contiguous
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[..., 0], first[..., 1])
    np.testing.assert_array_equal(first[..., 1], first[..., 2])
    assert float(first[:20].mean()) > 0.0
    assert not np.array_equal(first, image)


def test_lens_flare_disabled_or_probability_zero_is_identity() -> None:
    image = np.arange(24 * 31 * 3, dtype=np.uint8).reshape(24, 31, 3)
    disabled = augment.lens_flare_augment(
        image, AugConfig(lens_flare=False), np.random.default_rng(2)
    )
    skipped = augment.lens_flare_augment(
        image,
        AugConfig(lens_flare=True, lens_flare_prob=0.0),
        np.random.default_rng(2),
    )

    np.testing.assert_array_equal(disabled, image)
    np.testing.assert_array_equal(skipped, image)


def test_quarter_resolution_lens_flare_preserves_native_image_geometry() -> None:
    image = np.zeros((97, 145, 3), dtype=np.uint8)
    image[20:70, 45:105] = 210
    cfg = AugConfig(
        lens_flare=True,
        lens_flare_prob=1.0,
        lens_flare_max_strength=0.75,
        lens_flare_field_scale=0.25,
    )

    result = augment.lens_flare_augment(image, cfg, np.random.default_rng(31))

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert result.flags.c_contiguous
    np.testing.assert_array_equal(result[..., 0], result[..., 1])
    np.testing.assert_array_equal(result[..., 1], result[..., 2])
    assert not np.array_equal(result, image)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("lens_flare_prob", -0.1),
        ("lens_flare_max_strength", 1.1),
        ("lens_flare_field_scale", 0.0),
        ("lens_flare_field_scale", 1.1),
    ),
)
def test_lens_flare_rejects_invalid_configuration(field: str, value: float) -> None:
    cfg = AugConfig(lens_flare=True, **{field: value})

    with pytest.raises(ValueError, match=field):
        augment.lens_flare_augment(
            np.zeros((8, 8, 3), dtype=np.uint8),
            cfg,
            np.random.default_rng(2),
        )
