# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract against captured Blender 4.2 values, not an Isaac-generated oracle."""

import random

import numpy as np

from rendering.appearance import blender_field_seed, sample_hdri, sample_light_energy


def test_blender_readback_values():
    # Seed 2751, scene 0: recorded Blender world/light properties.
    rng = random.Random(blender_field_seed(2751, "background", 0))
    _, strength = sample_hdri(rng)
    assert float(np.float32(strength)) == 0.6404328942298889
    rng = random.Random(blender_field_seed(2751, "lights", 0))
    assert float(np.float32(sample_light_energy(rng))) == 9.444429397583008


def test_hdri_preserves_draw_count_and_legacy_rng_state():
    expected = random.Random(123)
    for _ in range(4):
        expected.random()
    actual = random.Random(123)
    sample_hdri(actual)
    assert actual.getstate() == expected.getstate()
