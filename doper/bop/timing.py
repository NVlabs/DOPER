# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct per-image BOP pose runtime: decoded RGB/K plus resident models to all final poses.

The protocol (``PROTOCOL``; ``NATIVE_PROTOCOL`` for the HOT3D native camera): one device, all object
models resident, sequential all-object passes in the original FP32 inference. There is no warmup:
the cold first image is measured and counted. One wall-clock span per image starts after the RGB is
read, hash-checked and decoded (and the models and keypoints are loaded), CUDA synchronized, and
ends when every object's final valid pose list is available, CUDA synchronized. For the native
camera the span also includes rectification and the JPEG round trip that reproduces the staged test
pixels. The scan itself is ``doper.bop.export.scan_bank``, so timed and exported poses come from the
same code. The warm (steady-state) GB300 deployment benchmark of the test variants is a separate
protocol, ``doper.bop.deployment_timing``; both are run by ``doper report --timing``.
"""

from __future__ import annotations

import math
import platform
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import export

PERCENTILE = 0.95
CPUINFO = Path("/proc/cpuinfo")
PROTOCOL: dict[str, Any] = {
    "start": "after RGB read/hash/decode, intrinsics and all models/keypoints loaded; "
    "CUDA synchronized",
    "end": "all objects processed and final valid pose lists available; CUDA synchronized",
    "includes": [
        "image preprocessing",
        "host-to-device image transfer",
        "all object detection/keypoint passes",
        "PnP",
        "pose validation and collection",
    ],
    "excludes": [
        "RGB file IO/hash/decode",
        "checkpoint loading",
        "CSV/JSON serialization and writes",
    ],
    "gpu_count": 1,
    "execution": "sequential all-object passes, all models GPU resident",
    "precision": "original FP32 inference, no AMP/compile/quantization changes",
    "explicit_warmup_passes": 0,
    "cold_first_measured_image_included": True,
    "score_threshold": export.SCORE_THRESHOLD,
    "max_detections_per_object": export.MAX_DETECTIONS,
    "no_GT_or_targets_or_presence_priors": True,
}
NATIVE_PROTOCOL: dict[str, Any] = {
    "start": "raw native RGB/calibration and all models loaded, CUDA synchronized",
    "end": "all object poses available, CUDA synchronized",
    "includes": [
        "native-camera rectification",
        "JPEG roundtrip matching frozen staged/validation pixels",
        "rectified content verification",
        "preprocessing/image GPU transfer",
        "all model passes",
        "PnP/pose collection",
    ],
    "excludes": [
        "native image file IO/hash/decode",
        "checkpoint loading",
        "output serialization/writing",
    ],
    "warmup_frames_discarded": 0,
    "models_GPU_resident": True,
    "GPUs": 1,
    "fixed_top5_score_threshold": export.SCORE_THRESHOLD,
    "test_training": False,
    "GT_targets_masks_hands_used": False,
}


@dataclass(frozen=True)
class TimingSummary:
    """Direct per-image runtime of a bank.

    Attributes:
        protocol: The timing protocol (``PROTOCOL`` or ``NATIVE_PROTOCOL``).
        camera: ``pinhole`` or ``native``.
        images: Images timed (every image, the cold first one included).
        objects_per_image: Models run on each image.
        mean_seconds: Mean image time.
        median_seconds: Median image time.
        p95_seconds: 95th percentile (linear interpolation).
        min_seconds: Fastest image.
        max_seconds: Slowest image.
        sum_seconds: Total timed seconds.
        pose_rows: Valid poses collected.
        images_without_poses: Images with no valid pose.
        seconds: Per-image times in population order.
        hardware: Device, library and threading settings of the run.
    """

    protocol: Mapping[str, Any]
    camera: str
    images: int
    objects_per_image: int
    mean_seconds: float
    median_seconds: float
    p95_seconds: float
    min_seconds: float
    max_seconds: float
    sum_seconds: float
    pose_rows: int
    images_without_poses: int
    seconds: tuple[float, ...]
    hardware: Mapping[str, Any]


def stats(values: Sequence[float]) -> dict[str, float]:
    """Runtime statistics of per-image seconds (linearly interpolated p95).

    Args:
        values: Per-image seconds.

    Returns:
        ``images``, ``mean_seconds``, ``median_seconds``, ``p95_seconds``, ``min_seconds``,
        ``max_seconds`` and ``sum_seconds``.
    """
    ordered = sorted(values)
    index = (len(ordered) - 1) * PERCENTILE
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return {
        "images": len(values),
        "mean_seconds": statistics.mean(values),
        "median_seconds": statistics.median(values),
        "p95_seconds": ordered[low] + (ordered[high] - ordered[low]) * (index - low),
        "min_seconds": min(values),
        "max_seconds": max(values),
        "sum_seconds": math.fsum(values),
    }


def cpu_model() -> str:
    """CPU model name from ``/proc/cpuinfo``, else ``platform``."""
    try:
        lines = CPUINFO.read_text().splitlines()
    except OSError:
        lines = []
    names = [line.split(":", 1)[1].strip() for line in lines if line.startswith("model name")]
    return names[0] if names else platform.processor() or platform.machine()


def hardware(device: str) -> dict[str, Any]:
    """Describe the timing device and the library settings that affect runtime."""
    import cv2
    import numpy as np
    import torch

    info: dict[str, Any] = {
        "device": device,
        "cpu": cpu_model(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "torch_threads": torch.get_num_threads(),
        "opencv_threads": cv2.getNumThreads(),
    }
    if device.startswith("cuda"):
        # cuDNN and device queries initialize CUDA, so they are made for CUDA timings only.
        properties = torch.cuda.get_device_properties(torch.device(device))
        info.update(
            cudnn=torch.backends.cudnn.version(),
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            gpu_name=properties.name,
            gpu_memory_bytes=properties.total_memory,
            compute_capability=[properties.major, properties.minor],
        )
    return info


def measure(
    bank: Mapping[int, Path],
    test_root: Path,
    *,
    camera: export.CameraName,
    score_threshold: float = export.SCORE_THRESHOLD,
    max_detections: int = export.MAX_DETECTIONS,
    journal_dir: Path | None = None,
    dataset: str | None = None,
    expected_population: export.TestPopulation | None = None,
    device: str = "cuda:0",
) -> TimingSummary:
    """Time every test image under the direct protocol (no warmup, cold first image counted).

    Args:
        bank: Object id to run directory (see ``export.run_bank``).
        test_root: BOP test split directory, or a test manifest JSON.
        camera: ``pinhole``, or ``native`` (rectification and JPEG round trip timed).
        score_threshold: Detection score threshold.
        max_detections: Detections per object and image.
        journal_dir: Resumable timing journal; its identity includes the hardware description,
            so timings from different hardware or software are never mixed. None times every
            image in this call.
        dataset: Packaged dataset whose frozen test population is required.
        expected_population: Required test population (fixtures); overrides ``dataset``.
        device: Torch device string.

    Returns:
        The timing summary.

    Raises:
        ValueError: Without a dataset or expected population, or for another population.
    """
    if expected_population is None:
        if dataset is None:
            raise ValueError("timing needs the dataset or an expected test population")
        expected_population = export.frozen_population(dataset)
    frames, population = export.load_test_frames(test_root, camera)
    export.require_population(frames, expected_population)
    models = export.load_bank(
        bank, device=device, score_threshold=score_threshold, max_detections=max_detections
    )
    protocol = NATIVE_PROTOCOL if camera == "native" else PROTOCOL
    machine = hardware(device)
    identity = export.bank_identity(
        models,
        population,
        camera=camera,
        protocol=protocol,
        hardware=machine,
        score_threshold=score_threshold,
        max_detections=max_detections,
    )
    records, _ = export.scan_bank(
        models,
        frames,
        camera=camera,
        max_detections=max_detections,
        synchronize=export.synchronizer(device),
        journal_dir=journal_dir,
        identity=identity,
    )
    rows = list(records.values())
    seconds = tuple(float(r["seconds"]) for r in rows)
    summary = stats(seconds)
    return TimingSummary(
        protocol=protocol,
        camera=camera,
        images=int(summary["images"]),
        objects_per_image=len(models),
        mean_seconds=summary["mean_seconds"],
        median_seconds=summary["median_seconds"],
        p95_seconds=summary["p95_seconds"],
        min_seconds=summary["min_seconds"],
        max_seconds=summary["max_seconds"],
        sum_seconds=summary["sum_seconds"],
        pose_rows=sum(len(r["predictions"]) for r in rows),
        images_without_poses=sum(not r["predictions"] for r in rows),
        seconds=seconds,
        hardware=machine,
    )
