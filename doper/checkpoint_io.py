# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atomic epoch checkpoints and process RNG state for preemptible training.

This supports epoch-boundary recovery, not bitwise continuation of persistent
DataLoader workers or a partially completed epoch.
"""

from __future__ import annotations

import importlib
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

# NumPy types in the saved checkpoint payload: the RNG state from
# ``np.random.get_state()`` (``_reconstruct``, ``ndarray``, ``dtype``, ``UInt32DType``),
# NumPy scalars in the metrics (``scalar``, ``Float64DType``) and the float32 rotation bins
# of source-format SPEED+ checkpoints (``Float32DType``). Checkpoints written
# under NumPy 1.x name the helpers ``numpy.core.multiarray``; NumPy 2 renamed the package
# to ``numpy._core``, so both spellings are allowlisted.
_MULTIARRAY = importlib.import_module(
    "numpy._core.multiarray" if hasattr(np, "_core") else "numpy.core.multiarray"
)
SAFE_CHECKPOINT_GLOBALS: list[Any] = [
    np.ndarray,
    np.dtype,
    np.dtypes.UInt32DType,
    np.dtypes.Float32DType,
    np.dtypes.Float64DType,
    *(
        entry
        for helper in (_MULTIARRAY._reconstruct, _MULTIARRAY.scalar)
        for entry in (
            helper,
            (helper, f"numpy.core.multiarray.{helper.__name__}"),
            (helper, f"numpy._core.multiarray.{helper.__name__}"),
        )
    ),
]


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint with ``weights_only=True``.

    Only the NumPy types of the saved RNG state, metrics and SPEED+ rotation bins are
    allowlisted on top of the tensor and plain-container types that ``torch.load`` already
    accepts.
    There is no fallback to unrestricted unpickling.

    Args:
        path: Checkpoint file.
        map_location: Device the tensors are mapped to.

    Returns:
        The unpickled checkpoint payload.
    """
    torch.serialization.add_safe_globals(SAFE_CHECKPOINT_GLOBALS)
    return torch.load(path, map_location=map_location, weights_only=True)


def checkpoint_configs(
    checkpoint: object, label: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate a checkpoint payload and return it with its ``cfg`` and ``cfg.model``.

    Older checkpoints store ``cfg`` or ``cfg.model`` as ``None`` or omit them;
    both read as empty mappings.

    Args:
        checkpoint: Payload returned by :func:`load_checkpoint`.
        label: Description of the checkpoint used in error messages.

    Returns:
        ``(checkpoint, cfg, model_cfg)``.

    Raises:
        ValueError: If the payload, ``cfg`` or ``cfg.model`` is not a mapping.
    """
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{label} must contain a state dictionary")
    cfg = checkpoint.get("cfg", {})
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{label} has a non-mapping cfg")
    model_cfg = cfg.get("model", {})
    if model_cfg is None:
        model_cfg = {}
    if not isinstance(model_cfg, dict):
        raise ValueError(f"{label} has a non-mapping cfg.model")
    return checkpoint, cfg, model_cfg


def atomic_torch_save(payload: object, path: str | Path) -> None:
    """Publish only a fully written checkpoint; preserve the old file on failure.

    Args:
        payload: Object passed to ``torch.save``.
        path: Destination file. Its parent directory is created if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def capture_rng_state() -> dict[str, Any]:
    """Capture the parent process RNG state (worker processes are not captured).

    Returns:
        Python, NumPy, torch CPU and (when available) CUDA RNG states.
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG tensors on CPU, even when the checkpoint was loaded on CUDA.

    Args:
        state: A mapping produced by :func:`capture_rng_state`.
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
