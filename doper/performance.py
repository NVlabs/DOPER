# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training RNG initialization and opt-in performance/backend settings."""

import os
import random
from functools import partial
from typing import Any

import cv2
import numpy as np
import torch

from .config import TrainConfig
from .console import log


def _initialize_worker(worker_id: int, *, opencv_threads: int | None) -> None:
    """Seed worker OpenCV independently and optionally bound its CPU threads."""
    if opencv_threads is not None:
        cv2.setNumThreads(opencv_threads)
    cv2.setRNGSeed(torch.initial_seed() % (2**31))


def seed_training(cfg: TrainConfig) -> None:
    """Seed parent RNGs once per fresh training process."""
    torch.manual_seed(cfg.seed)  # Includes all CUDA devices.
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    cv2.setRNGSeed(cfg.seed % (2**31))


def loader_options(cfg: TrainConfig, *, stream: str = "train") -> dict[str, Any]:
    """Return worker-only options, valid also for a synchronous DataLoader.

    Args:
        cfg: Training config.
        stream: ``"train"`` or ``"val"``; selects the shuffle generator seed.

    Returns:
        Keyword arguments to pass to ``torch.utils.data.DataLoader``.

    Raises:
        ValueError: If worker/thread settings or ``stream`` are invalid.
    """
    if cfg.num_workers < 0 or cfg.prefetch_factor < 1:
        raise ValueError("num_workers must be nonnegative and prefetch_factor positive")
    if cfg.opencv_threads is not None and cfg.opencv_threads < 1:
        raise ValueError("opencv_threads must be positive when specified")
    if stream not in ("train", "val"):
        raise ValueError("loader stream must be train or val")
    options = {}
    if cfg.reproducible:
        # Isolate shuffle/worker seeds from model initialization and validation.
        options["generator"] = torch.Generator().manual_seed(cfg.seed + (stream == "val"))
    if cfg.num_workers:
        options["prefetch_factor"] = cfg.prefetch_factor
        options["worker_init_fn"] = partial(_initialize_worker, opencv_threads=cfg.opencv_threads)
    elif cfg.opencv_threads is not None:
        cv2.setNumThreads(cfg.opencv_threads)
    return options


def configure_runtime(cfg: TrainConfig, device: str) -> torch.dtype:
    """Apply only explicitly requested backends and reject unsupported BF16.

    Args:
        cfg: Training config; ``cudnn_benchmark`` is forced off when reproducible.
        device: Torch device string such as ``"cuda:0"`` or ``"cpu"``.

    Returns:
        The autocast dtype selected by ``cfg.amp_dtype``.

    Raises:
        ValueError: If the AMP dtype, CUBLAS workspace or BF16 support is invalid.
    """
    if cfg.amp_dtype not in ("float16", "bfloat16"):
        raise ValueError("amp_dtype must be float16 or bfloat16")
    if cfg.reproducible:
        if device.startswith("cuda"):
            workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            if workspace is None:
                if torch.cuda.is_initialized():
                    raise ValueError(
                        "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before "
                        "CUDA initialization for reproducible training"
                    )
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            elif workspace not in (":4096:8", ":16:8"):
                raise ValueError(
                    "Reproducible training requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8"
                )
        cfg.cudnn_benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=False)
    if cfg.amp and device.startswith("cuda") and cfg.amp_dtype == "bfloat16":
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("Requested BF16 training is unsupported on this GPU")
    if cfg.cudnn_benchmark is not None:
        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    if cfg.allow_tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = cfg.allow_tf32
        torch.backends.cudnn.allow_tf32 = cfg.allow_tf32
    return torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16


def image_layout(batch: dict[str, Any], channels_last: bool) -> dict[str, Any]:
    """Change only the two image tensors' layout, preserving targets and values.

    Args:
        batch: Collated batch containing ``det_image`` and ``crop_image``.
        channels_last: Whether to convert the images to channels-last memory format.

    Returns:
        The input batch if ``channels_last`` is false, else a shallow copy.
    """
    if not channels_last:
        return batch
    batch = dict(batch)
    for key in ("det_image", "crop_image"):
        batch[key] = batch[key].contiguous(memory_format=torch.channels_last)
    return batch


def resolve_device(requested: str) -> str:
    """Fall back to the CPU when CUDA is requested but unavailable."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        log("CUDA unavailable, falling back to CPU")
        return "cpu"
    return requested
