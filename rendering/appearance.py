# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Blender appearance draws; order is part of the scene recipe contract."""

import hashlib
import math
import random


def blender_field_seed(seed: int | str, field: str, index: int) -> int:
    """Return the restart-independent seed used by Blender scene records."""
    digest = hashlib.sha256(f"{seed}:blender-v2:{field}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def sample_hdri(rng: random.Random) -> tuple[tuple[float, float, float], float]:
    """Draw rotation then strength, including rotations overridden by records."""
    rotation = (
        rng.uniform(-0.1, 0.1),
        rng.uniform(-0.1, 0.1),
        rng.uniform(0, 2 * math.pi),
    )
    return rotation, rng.uniform(0.5, 2.0)


def sample_light_energy(rng: random.Random) -> float:
    """Draw the Blender point/area/spot energy in watts."""
    return rng.uniform(1.0, 15.0)
