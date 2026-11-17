# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit per-worker state; components never share mutable module globals."""

import random
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any

from .settings import WorkerSettings

# Blender datablocks (objects, materials, scenes) have no type stubs.
BpyObject = Any


@dataclass
class BlenderContext:
    """BlenderContext."""

    args: Namespace
    settings: WorkerSettings
    _kp_mask_stats: dict = field(default_factory=dict)
    annotations: list[dict] = field(default_factory=list)
    cam_data: Any = None
    cam_obj: Any = None
    cx: float = 0.0
    cy: float = 0.0
    distractor_files: list[str] = field(default_factory=list)
    eevee: Any = None
    eevee_samples: int = 0
    extra_lights: list[Any] = field(default_factory=list)
    fx: float = 0.0
    fy: float = 0.0
    ground_obj: Any = None
    hdri_files: list[str] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)
    kp_objects: list[Any] = field(default_factory=list)
    loaded_distractors: list[Any] = field(default_factory=list)
    robustness_objects: list[Any] = field(default_factory=list)
    mask_background_material: Any = None
    mask_target_material: Any = None
    mesh_obj: Any = None
    mesh_size: float = 0.0
    num_views: int = 0
    object_rgba_choices: Any = None
    record_distractor_instances: list[tuple] = field(default_factory=list)
    rng: random.Random = field(default_factory=random.Random)
    scene: Any = None
    scene_records: dict[int, dict] = field(default_factory=dict)
    environment_exporter: Any = None
    scene_scale: float = 0.0
    texture_dirs: list[str] = field(default_factory=list)
