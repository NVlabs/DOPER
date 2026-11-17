# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DOPER: pose estimation from keypoint heatmaps.

The package exposes the config, dataset, model, and the joint loss. Names load on first access
so ``import doper`` (and ``doper.scene``) does not import torch.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import DataConfig, DoperConfig, LossConfig, ModelConfig, TrainConfig
    from .dataset import DoperDataset
    from .losses import compute_losses
    from .model import DoperNet

__version__ = "0.1.0.dev0"

_LAZY: dict[str, str] = {
    "DataConfig": ".config",
    "DoperConfig": ".config",
    "LossConfig": ".config",
    "ModelConfig": ".config",
    "TrainConfig": ".config",
    "DoperDataset": ".dataset",
    "compute_losses": ".losses",
    "DoperNet": ".model",
}

__all__ = [
    "DataConfig",
    "DoperConfig",
    "DoperDataset",
    "DoperNet",
    "LossConfig",
    "ModelConfig",
    "TrainConfig",
    "__version__",
    "compute_losses",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Import the module defining a public name on first access."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the public names, including those not loaded yet."""
    return sorted(set(globals()) | set(__all__))
