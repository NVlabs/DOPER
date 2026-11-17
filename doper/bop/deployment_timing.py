# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warm deployment request latency of the BOP test variants (the GB300 Table 3 protocol).

This is a steady-state benchmark, separate from the direct (cold) per-image runtime of
``doper.bop.timing``. One device holds every object model of a bank. A request is one sampled
reference image (with its support views for the multi-view variants); its span starts with the
decoded image arrays in memory, CUDA synchronized, and ends when every model's final poses of the
variant are available, CUDA synchronized. Inside the span the inference process runs each model on
each view (four PyTorch threads, ``cv2.setRNGSeed(42)`` per view); the fusion and heatmap refinement
of each model run in one of six persistent spawn worker processes (one PyTorch/OpenCV thread each,
``OMP/OPENBLAS/MKL_NUM_THREADS=1``) and overlap the next model's inference. The span therefore
includes serialization, scheduling and waits; a multi-view request re-infers every support view.
Composition is ``doper.bop.variants`` ``compose_variants``, the same code as the variant exports.

Before any request every model is warmed on one image of each camera resolution, the crop head on
batches 1-5, and the workers with one task (an empty one, so the refinement code itself is not
exercised). Each sampled group is timed ``repetitions`` times, its variants in the order
``Random(42 + 2 * group + repeat)`` gives. Every record is appended to the request journal as it
completes. Statistics are weighted by the sample weights (``sample_groups``). Within a group and
repetition every variant must return the same number of predictions: the composed detections (finite
single-view poses) before any variant's ``t_z > 0`` filter; each variant's returned rows are
recorded separately. HANDAL has no multi-view policy, so a single-view pose with ``t_z <= 0`` is
never composed (nor refined), and its count is taken after that rejection, which every variant
applies.

HOT3D views are decoded native images (outside the span); each request rectifies them inside the
span with ``cv2.remap`` on maps precomputed from the native calibration (bilinear, black border,
gray to BGR), without the JPEG round trip of the staged evaluation inputs.

The paper's GB300 timings were measured with Powell refinement; the release default is L-BFGS-B, so
the manifest records the optimizer and preparation that ran.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import functools
import json
import multiprocessing
import os
import platform
import random
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..recipes.schema import MultiviewSpec, numba_installed
from ..refinement import RefineSettings
from . import export, hot3d
from .multiview import ImageKey
from .sources import canonical_sha256
from .variants import (
    HEATMAP_VARIANTS,
    MULTIVIEW_VARIANTS,
    VARIANTS,
    JournalDetection,
    Variant,
    VariantSettings,
    View,
    compose_variants,
    library_versions,
)

GROUP_COUNT = 24
REPETITIONS = 2
WORKERS = 6
TORCH_THREADS = 4
WORKER_THREADS = 1
SHUFFLE_SEED = 42
VIEW_SEED = 42
WARMUP_CROP_BATCHES = (1, 2, 3, 4, 5)
WORKER_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
READY_SLEEP_SECONDS = 0.1
MEDIAN_QUANTILE = 0.5
P95_QUANTILE = 0.95
SINGLE_VIEW_DATASET_VARIANTS: tuple[Variant, ...] = ("singleview", "singleheatmap")
PROTOCOL: dict[str, Any] = {
    "start": "decoded image arrays of the request's views in memory, all models resident; "
    "CUDA synchronized",
    "end": "every model's final poses of the variant available; CUDA synchronized",
    "includes": [
        "image preprocessing and host-to-device transfer",
        "every model on every view of the request (support views re-inferred)",
        "PnP",
        "multi-view fusion and heatmap refinement in the CPU workers",
        "task serialization, scheduling and waits",
    ],
    "excludes": [
        "image file IO and decode",
        "checkpoint loading",
        "warmup",
        "output serialization and writes",
    ],
    "gpu_count": 1,
    "models_resident": True,
    "precision": "original FP32 inference, no AMP/compile/quantization changes",
    "score_threshold": export.SCORE_THRESHOLD,
    "max_detections_per_object": export.MAX_DETECTIONS,
    "warmup": "every camera resolution per model, crop batches 1-5, one worker task",
    "worker_warmup_exercises_refinement": False,
    "cpu_workers": WORKERS,
    "worker_start_method": "spawn",
    "inference_torch_threads": TORCH_THREADS,
    "worker_torch_opencv_numba_threads": WORKER_THREADS,
    "repetitions": REPETITIONS,
    "variant_order": "random.Random(42 + 2 * group + repeat).shuffle",
    "sample": {
        "hopev2": "24 groups stratified uncalibrated/calibrated, evenly spaced; weight = "
        "stratum fraction / stratum groups",
        "tless": "middle scheduled image of each scene; weight = scene fraction",
        "hot3d": "24 groups stratified by view count 2/3, evenly spaced; weight = stratum "
        "fraction / stratum groups",
        "handal": "24 evenly spaced images; weight 1/24",
    },
    "statistics": "weighted mean, weighted median and weighted p95 over requests",
    "prediction_count": "composed detections (finite single-view poses) before the t_z > 0 "
    "filter on the multi-view datasets; on HANDAL after rejecting single-view poses with "
    "t_z <= 0, which are never composed nor refined (the source HANDAL follow-up rejected "
    "nonpositive initial depth in both arms); equal across the variants of a group and "
    "repetition",
    "hot3d_inputs": "decoded native images; per-request cv2.remap with precomputed maps "
    "(bilinear, black border, gray to BGR) inside the span; no JPEG round trip",
    "no_GT_or_targets_or_presence_priors": True,
}


@dataclass(frozen=True)
class WeightedGroup:
    """One sampled request: a reference image, its views and its sample weight.

    Attributes:
        reference: The reference image.
        views: The reference first, then its support views.
        weight: Sample weight (the weights of a dataset's sample sum to one).
    """

    reference: ImageKey
    views: tuple[ImageKey, ...]
    weight: float


def evenly(items: Sequence[Any], count: int) -> list[Any]:
    """``count`` evenly spaced items, first and last included."""
    return [items[int(x)] for x in np.linspace(0, len(items) - 1, min(count, len(items)))]


def _stratified(
    strata: Sequence[Sequence[ImageKey]],
    total: int,
    groups_spec: Mapping[ImageKey, Sequence[ImageKey]],
) -> list[WeightedGroup]:
    sample = []
    for keys in strata:
        count = round(GROUP_COUNT * len(keys) / total)
        fraction = len(keys) / total
        sample += [
            WeightedGroup(k, tuple(groups_spec[k]), fraction / count) for k in evenly(keys, count)
        ]
    return sample


def sample_groups(
    dataset: str,
    population: Sequence[ImageKey],
    groups_spec: Mapping[ImageKey, Sequence[ImageKey]],
) -> list[WeightedGroup]:
    """The deterministic image-only request sample of a dataset (no labels or detections).

    Args:
        dataset: ``hopev2``, ``tless``, ``hot3d`` or ``handal``.
        population: The test reference images.
        groups_spec: Reference image to its group, the reference first (``support_groups``;
            a reference without supports maps to itself).

    Returns:
        The sampled groups: HOPEv2 uncalibrated then calibrated, HOT3D two-view then
        three-view, T-LESS by scene, HANDAL in image order.

    Raises:
        ValueError: For an unknown dataset or a reference without a group.
    """
    items = sorted(set(population))
    missing = [tuple(k) for k in items if k not in groups_spec]
    if missing:
        raise ValueError(f"references {missing[:3]} have no group")
    name = export.normalize_dataset(dataset)
    if name == "hopev2":
        strata = [[k for k in items if len(groups_spec[k]) == 1],
                  [k for k in items if len(groups_spec[k]) > 1]]  # fmt: skip
        return _stratified(strata, len(items), groups_spec)
    if name == "hot3d":
        strata = [[k for k in items if len(groups_spec[k]) == n] for n in (2, 3)]
        return _stratified(strata, len(items), groups_spec)
    if name == "tless":
        sample = []
        for scene in sorted({k.scene_id for k in items}):
            frames = [k for k in items if k.scene_id == scene]
            middle = frames[len(frames) // 2]
            sample.append(
                WeightedGroup(middle, tuple(groups_spec[middle]), len(frames) / len(items))
            )
        return sample
    if name == "handal":
        return [
            WeightedGroup(k, tuple(groups_spec[k]), 1 / GROUP_COUNT)
            for k in evenly(items, GROUP_COUNT)
        ]
    raise ValueError(f"unknown deployment timing dataset {dataset!r}")


def groups_sha256(groups: Sequence[WeightedGroup]) -> str:
    """Identity of a request sample (references, views and weights, in order)."""
    return canonical_sha256(
        [[list(g.reference), [list(k) for k in g.views], g.weight] for g in groups]
    )


def variant_order(variants: Sequence[Variant], group: int, repeat: int) -> list[Variant]:
    """The variant order of one group and repetition: ``Random(42 + 2 * group + repeat)``."""
    order = list(variants)
    random.Random(SHUFFLE_SEED + 2 * group + repeat).shuffle(order)
    return order


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    index = np.searchsorted(np.cumsum(weights[order]), quantile)
    return float(values[order][min(len(values) - 1, int(index))])


def weighted_stats(samples: Sequence[tuple[float, float]]) -> dict[str, float]:
    """Weighted mean, median and p95 of ``(seconds, weight)`` samples.

    The weights are normalized; a quantile is the smallest sorted value whose cumulative weight
    reaches it (``searchsorted``, left).

    Raises:
        ValueError: Without samples.
    """
    if not samples:
        raise ValueError("no timing samples")
    values = np.array([s[0] for s in samples], dtype=np.float64)
    weights = np.array([s[1] for s in samples], dtype=np.float64)
    weights /= weights.sum()
    return {
        "mean": float(weights @ values),
        "median": _weighted_quantile(values, weights, MEDIAN_QUANTILE),
        "p95": _weighted_quantile(values, weights, P95_QUANTILE),
    }


def check_prediction_counts(records: Sequence[Mapping[str, Any]]) -> None:
    """Require every variant of a group and repetition to return the same number of poses.

    Raises:
        ValueError: Naming the first ``(group, repeat)`` whose variants disagree.
    """
    counts: dict[tuple[int, int], set[int]] = {}
    for record in records:
        counts.setdefault((record["group"], record["repeat"]), set()).add(record["predictions"])
    for key in sorted(counts):
        if len(counts[key]) != 1:
            raise ValueError(
                f"group {key[0]} repetition {key[1]}: the variants returned different "
                f"prediction counts {sorted(counts[key])}"
            )


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-variant weighted statistics of the request records.

    Returns:
        Variant to ``mean_s``, ``median_s``, ``p95_s``, ``requests``, ``reference_groups`` and
        the weighted ``mean_predictions``.

    Raises:
        ValueError: If the variants of a group and repetition disagree on prediction counts.
    """
    check_prediction_counts(records)
    summary = {}
    for name in sorted({r["variant"] for r in records}):
        rows = [r for r in records if r["variant"] == name]
        stats = weighted_stats([(r["wall_s"], r["weight"]) for r in rows])
        weights = np.array([r["weight"] for r in rows], dtype=np.float64)
        weights /= weights.sum()
        summary[name] = {
            "mean_s": stats["mean"],
            "median_s": stats["median"],
            "p95_s": stats["p95"],
            "requests": len(rows),
            "reference_groups": len({r["group"] for r in rows}),
            "mean_predictions": float(weights @ np.array([r["predictions"] for r in rows])),
        }
    return summary


@dataclass(frozen=True)
class TimedModel:
    """One resident model of the timed bank.

    Attributes:
        obj_id: BOP object id.
        detect: ``detect(image_bgr, K, raw)`` returning ``doper.infer.Detection`` objects with
            poses in millimeters; ``raw`` keeps the heatmaps of the same forward pass.
        keypoints_mm: ``(K, 3)`` object keypoints, millimeters.
        crop_size: Crop side (``cfg.model.crop_size``).
        beta: Heatmap softmax temperature (``cfg.loss.keypoint_heatmap_beta``).
        checkpoint_sha256: SHA-256 of the checkpoint.
        keypoints_sha256: SHA-256 of the keypoint file.
        keypoint_head: Keypoint head kind (the heatmap variants need ``heatmap``).
        warm_crop: Runs the crop head once on a zero batch of the given size; None to skip.
    """

    obj_id: int
    detect: Callable[[np.ndarray, np.ndarray, bool], list[Any]]
    keypoints_mm: np.ndarray
    crop_size: int
    beta: float
    checkpoint_sha256: str
    keypoints_sha256: str
    keypoint_head: str | None = None
    warm_crop: Callable[[int], None] | None = None


@dataclass(frozen=True)
class DeploymentSettings:
    """What a deployment timing run measures.

    Attributes:
        dataset: Dataset name.
        refine: Heatmap optimizer and preparation.
        spec: The dataset's multi-view policy; None for a single-view dataset (HANDAL).
        cameras: Camera record of every reference and support image.
        images: Decoded image of every reference and support view (loaded before timing): the
            network BGR image, or for a view in ``maps`` the native image.
        maps: Native views (HOT3D) to their precomputed ``(map_x, map_y)`` rectification maps
            (``doper.bop.hot3d.rectification_maps``); rectified inside the span.
        variants: Variants timed; default all four, or single view and heatmap without ``spec``.
        device: Torch device string.
        max_detections: Detections kept per model and view.
    """

    dataset: str
    refine: RefineSettings
    spec: MultiviewSpec | None
    cameras: Mapping[ImageKey, Mapping[str, Any]]
    images: Mapping[ImageKey, np.ndarray]
    maps: Mapping[ImageKey, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    variants: tuple[Variant, ...] | None = None
    device: str = "cuda:0"
    max_detections: int = export.MAX_DETECTIONS

    def timed_variants(self) -> tuple[Variant, ...]:
        """The variants of the run (the default depends on the multi-view policy)."""
        if self.variants is not None:
            return self.variants
        return VARIANTS if self.spec is not None else SINGLE_VIEW_DATASET_VARIANTS


def rectify_native(image: np.ndarray, maps: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Rectify a native image with precomputed maps, as ``doper.bop.hot3d.rectify`` does.

    Bilinear ``cv2.remap`` with a black constant border; a gray image becomes BGR.
    """
    return hot3d.apply_rectification(image, *maps)


@dataclass(frozen=True)
class TimingResult:
    """A deployment timing run.

    Attributes:
        protocol: ``PROTOCOL``.
        records: One record per request, in measurement order.
        summary: Per-variant weighted statistics (``summarize``).
        manifest: Optimizer, preparation, sample, models, hardware and versions of the run.
    """

    protocol: Mapping[str, Any]
    records: tuple[dict[str, Any], ...]
    summary: Mapping[str, Mapping[str, Any]]
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class _Heatmap:
    """A picklable heatmap loader holding the arrays of one forward pass."""

    logits: np.ndarray
    crop_to_image: np.ndarray

    def __call__(self) -> tuple[np.ndarray, np.ndarray]:
        return self.logits, self.crop_to_image


@dataclass(frozen=True)
class _Task:
    """One model's CPU part of one request, sent to a worker."""

    variant: Variant
    reference: ImageKey
    views: tuple[ImageKey, ...]
    journal: Mapping[ImageKey, View]
    keypoints_mm: np.ndarray
    settings: VariantSettings
    spec: MultiviewSpec | None = None


@dataclass(frozen=True)
class _Composed:
    """One model's CPU result of one request.

    Attributes:
        predictions: Composed detections (finite single-view poses; on HANDAL only those
            with ``t_z > 0``; the shared count, equal across variants by construction).
        rows: Rows the variant returns after its ``t_z > 0`` validity filter.
        cpu_seconds: CPU wall time of the composition.
        optimizers: The optimizers that ran (``variants.audit_optimizer``).
    """

    predictions: int
    rows: int
    cpu_seconds: float
    optimizers: tuple[str, ...]


def _compose(task: _Task) -> _Composed:
    """Compose one model's variant rows (in a worker, or in-process for ``singleview``)."""
    cv2.setRNGSeed(VIEW_SEED)
    started = time.perf_counter()
    audit: list[dict[str, Any]] = []
    rows = compose_variants(
        task.journal,
        {task.reference: task.views},
        task.keypoints_mm,
        task.settings,
        task.spec,
        audit=audit,
    )[task.variant]
    ran = sorted({entry["optimizer"] for entry in audit if entry["optimizer"] is not None})
    return _Composed(len(audit), len(rows), time.perf_counter() - started, tuple(ran))


def _init_worker() -> None:
    """One PyTorch, OpenCV and Numba thread per worker (BLAS/OpenMP come from the environment)."""
    import torch

    torch.set_num_threads(WORKER_THREADS)
    cv2.setNumThreads(WORKER_THREADS)
    if numba_installed():
        import numba

        # The parallel preparation uses min(threads, current), so it runs on one thread here.
        numba.set_num_threads(WORKER_THREADS)


def _ready(_: int) -> int:
    time.sleep(READY_SLEEP_SECONDS)
    return os.getpid()


def _journal_detections(
    detections: Sequence[Any], max_detections: int, raw: bool
) -> tuple[JournalDetection, ...]:
    """The kept detections of one view (score order, stable), as journal detections."""
    kept = sorted(detections, key=lambda d: d.score, reverse=True)[:max_detections]
    out = []
    for index, det in enumerate(kept):
        heatmap = getattr(det, "keypoint_heatmap", None)
        affine = getattr(det, "crop_to_image", None)
        out.append(
            JournalDetection.from_raw(
                export.raw_detection(det, index),
                _Heatmap(np.asarray(heatmap), np.asarray(affine))
                if raw and heatmap is not None and affine is not None
                else None,
            )
        )
    return tuple(out)


def _camera_matrix(settings: DeploymentSettings, key: ImageKey) -> np.ndarray:
    return np.asarray(settings.cameras[key]["cam_K"], dtype=np.float64).reshape(3, 3)


@dataclass
class _Runner:
    """The resident models, the worker pool and the request loop of one run."""

    models: Sequence[TimedModel]
    settings: DeploymentSettings
    pool: concurrent.futures.ProcessPoolExecutor
    synchronize: Callable[[], None]
    optimizers: set[str] = field(default_factory=set)

    def _task(self, model: TimedModel, variant: Variant, group: WeightedGroup,
              views: tuple[ImageKey, ...], journal: Mapping[ImageKey, View]) -> _Task:  # fmt: skip
        settings = VariantSettings(
            refine=self.settings.refine, crop_size=model.crop_size, beta=model.beta,
            variants=(variant,),
        )  # fmt: skip
        return _Task(variant, group.reference, views, journal, model.keypoints_mm, settings,
                     self.settings.spec)  # fmt: skip

    def network_input(self, key: ImageKey) -> np.ndarray:
        """The network image of a view: the decoded image, or the native one rectified."""
        image = self.settings.images[key]
        maps = self.settings.maps.get(key)
        return image if maps is None else rectify_native(image, maps)

    def warm(self, groups: Sequence[WeightedGroup], workers: int) -> None:
        """Warm every camera resolution and the crop head of each model, then the workers."""
        shapes: dict[tuple[int, ...], tuple[ImageKey, np.ndarray]] = {}
        for group in groups:
            for key in group.views:
                image = self.network_input(key)
                shapes.setdefault(image.shape, (key, image))
        for model in self.models:
            for key, image in shapes.values():
                model.detect(image, _camera_matrix(self.settings, key), False)
            if model.warm_crop is not None:
                for batch in WARMUP_CROP_BATCHES:
                    model.warm_crop(batch)
        self.synchronize()
        seen: set[int] = set()
        while len(seen) < workers:
            seen.update(self.pool.map(_ready, range(4 * workers)))
        group = groups[0]
        journal = {k: View(self.settings.cameras[k], ()) for k in group.views}
        variant = self.settings.timed_variants()[-1]
        self.pool.submit(_compose, self._task(self.models[0], variant, group, group.views,
                                              journal)).result()  # fmt: skip

    def request(self, variant: Variant, index: int, group: WeightedGroup,
                repeat: int) -> dict[str, Any]:  # fmt: skip
        """Time one request and return its record."""
        views = group.views if variant in MULTIVIEW_VARIANTS else group.views[:1]
        raw = variant in HEATMAP_VARIANTS
        settings = self.settings
        self.synchronize()
        start = time.perf_counter()
        images = [self.network_input(k) for k in views]
        preprocess = time.perf_counter() - start
        futures: list[concurrent.futures.Future[_Composed]] = []
        results: list[_Composed] = []
        inference = 0.0
        for model in self.models:
            began = time.perf_counter()
            journal = {}
            for key, image in zip(views, images, strict=True):
                cv2.setRNGSeed(VIEW_SEED)
                found = model.detect(image, _camera_matrix(settings, key), raw)
                kept = _journal_detections(found, settings.max_detections, raw)
                journal[key] = View(settings.cameras[key], kept)
            inference += time.perf_counter() - began
            task = self._task(model, variant, group, views, journal)
            if variant == "singleview":
                results.append(_compose(task))
            else:
                futures.append(self.pool.submit(_compose, task))
        results += [f.result() for f in futures]
        self.synchronize()
        elapsed = time.perf_counter() - start
        for result in results:
            self.optimizers.update(result.optimizers)
        return {
            "dataset": settings.dataset,
            "variant": variant,
            "group": index,
            "scene_id": group.reference.scene_id,
            "im_id": group.reference.im_id,
            "stream": group.reference.stream,
            "repeat": repeat,
            "views": len(views),
            "weight": group.weight,
            "wall_s": elapsed,
            "preprocess_s": preprocess,
            "inference_loop_wall_s": inference,
            "worker_cpu_sum_s": sum(r.cpu_seconds for r in results if variant != "singleview"),
            "predictions": sum(r.predictions for r in results),
            "rows": sum(r.rows for r in results),
        }


@contextlib.contextmanager
def _thread_settings(torch_threads: int) -> Iterator[None]:
    """One BLAS/OpenMP thread for spawned workers, ``torch_threads`` for inference; restored."""
    import torch

    saved_env = {name: os.environ.get(name) for name in WORKER_THREAD_ENV}
    saved_torch, saved_cv = torch.get_num_threads(), cv2.getNumThreads()
    os.environ.update(dict.fromkeys(WORKER_THREAD_ENV, str(WORKER_THREADS)))
    torch.set_num_threads(torch_threads)
    cv2.setNumThreads(WORKER_THREADS)
    try:
        yield
    finally:
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        torch.set_num_threads(saved_torch)
        cv2.setNumThreads(saved_cv)


def runtime_versions() -> dict[str, Any]:
    """Versions of the libraries and interpreter that affect the measured latency."""
    import torch

    return {
        **library_versions(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def timing_manifest(
    settings: DeploymentSettings,
    groups: Sequence[WeightedGroup],
    models: Sequence[TimedModel],
    *,
    records: Sequence[Mapping[str, Any]],
    optimizers_run: Sequence[str],
    repetitions: int,
    workers: int,
    torch_threads: int,
    hardware: Mapping[str, Any],
) -> dict[str, Any]:
    """The manifest of a timing run: the optimizer and preparation that ran, and the setup.

    Raises:
        ValueError: If a refinement ran with another optimizer than the declared one.
    """
    declared = settings.refine.optimizer
    if any(name != declared for name in optimizers_run):
        raise ValueError(
            f"the timing declares optimizer {declared!r} but {sorted(set(optimizers_run))} "
            "ran; a timing must never be described with another optimizer than it measured"
        )
    names = settings.timed_variants()
    return {
        "protocol": PROTOCOL,
        "dataset": settings.dataset,
        "variants": list(names),
        "optimizer": declared,
        "preparation": settings.refine.preparation,
        "threads": settings.refine.threads,
        "optimizers_run": sorted(set(optimizers_run)),
        "multiview": None if settings.spec is None else repr(settings.spec),
        "repetitions": repetitions,
        "workers": workers,
        "torch_threads": torch_threads,
        "worker_threads": WORKER_THREADS,
        "worker_thread_env": dict.fromkeys(WORKER_THREAD_ENV, str(WORKER_THREADS)),
        "worker_numba_threads": WORKER_THREADS if numba_installed() else None,
        "max_detections": settings.max_detections,
        "groups": len(groups),
        "groups_sha256": groups_sha256(groups),
        "models": [
            {
                "object_id": m.obj_id,
                "checkpoint_sha256": m.checkpoint_sha256,
                "keypoints_sha256": m.keypoints_sha256,
                "crop_size": m.crop_size,
                "beta": m.beta,
            }
            for m in models
        ],  # fmt: skip
        "requests": len(records),
        "complete": len(records) == len(groups) * repetitions * len(names),
        "hardware": dict(hardware),
        "versions": runtime_versions(),
    }


def run(
    bank: Sequence[TimedModel],
    groups: Sequence[WeightedGroup],
    settings: DeploymentSettings,
    *,
    repetitions: int = REPETITIONS,
    workers: int = WORKERS,
    torch_threads: int = TORCH_THREADS,
    requests_path: Path | None = None,
) -> TimingResult:
    """Time every sampled group under the deployment protocol (module docstring).

    Args:
        bank: The resident models (``load_models``).
        groups: The request sample (``sample_groups``).
        settings: Dataset, variants, optimizer, cameras and decoded images.
        repetitions: Timed repetitions of each group.
        workers: Persistent spawn worker processes for fusion and refinement.
        torch_threads: PyTorch threads of the inference process.
        requests_path: Request journal (JSON lines), truncated at the start; each record is
            appended as its request completes, so a late failure keeps every timing.

    Returns:
        The records, the weighted per-variant statistics and the manifest.

    Raises:
        ValueError: Without models or groups, for heatmap variants on a model without a heatmap
            head, multi-view variants without a policy, prediction counts that disagree between
            variants, or a refinement that ran with another optimizer than declared.
    """
    from .timing import hardware as describe_hardware

    names = settings.timed_variants()
    if not bank or not groups:
        raise ValueError("deployment timing needs models and sampled groups")
    if workers < 1 or repetitions < 1:
        raise ValueError(f"workers and repetitions must be >= 1, got {workers}, {repetitions}")
    if any(v in MULTIVIEW_VARIANTS for v in names) and settings.spec is None:
        raise ValueError("the multi-view variants need the dataset's multi-view policy")
    plain = [m.obj_id for m in bank if m.keypoint_head != "heatmap"]
    if any(v in HEATMAP_VARIANTS for v in names) and plain:
        raise ValueError(f"objects {plain}: the heatmap variants need a heatmap keypoint head")
    records = []
    if requests_path is not None:
        requests_path.parent.mkdir(parents=True, exist_ok=True)
        requests_path.write_text("")
    context = multiprocessing.get_context("spawn")
    with (
        _thread_settings(torch_threads),
        concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=context, initializer=_init_worker
        ) as pool,
    ):
        runner = _Runner(bank, settings, pool, export.synchronizer(settings.device))
        runner.warm(groups, workers)
        for index, group in enumerate(groups):
            for repeat in range(repetitions):
                for variant in variant_order(names, index, repeat):
                    record = runner.request(variant, index, group, repeat)
                    records.append(record)
                    if requests_path is not None:
                        with requests_path.open("a") as journal:
                            journal.write(json.dumps(record) + "\n")
        machine = describe_hardware(settings.device)
    summary = summarize(records)
    manifest = timing_manifest(
        settings, groups, bank, records=records, optimizers_run=sorted(runner.optimizers),
        repetitions=repetitions, workers=workers, torch_threads=torch_threads, hardware=machine,
    )  # fmt: skip
    return TimingResult(PROTOCOL, tuple(records), summary, manifest)


def _detect(
    model: Any,  # noqa: ANN401
    cfg: Any,  # noqa: ANN401
    device: str,
    keypoints: np.ndarray,
    score_threshold: float,
    max_detections: int,
    image: np.ndarray,
    camera_matrix: np.ndarray,
    raw: bool,
) -> list[Any]:
    import torch

    with torch.inference_mode():
        return export._detect_with_model(
            model, cfg, device, keypoints, score_threshold, max_detections, raw, image,
            camera_matrix,
        )  # fmt: skip


def _warm_crop(model: Any, crop_size: int, device: str, batch: int) -> None:  # noqa: ANN401
    import torch

    with torch.inference_mode():
        model.forward_crop(torch.zeros((batch, 3, crop_size, crop_size), device=device))


def timed_model(
    obj_id: int,
    model: Any,  # noqa: ANN401
    cfg: Any,  # noqa: ANN401
    keypoints_mm: np.ndarray,
    *,
    device: str,
    checkpoint_sha256: str,
    keypoints_sha256: str,
    score_threshold: float = export.SCORE_THRESHOLD,
    max_detections: int = export.MAX_DETECTIONS,
) -> TimedModel:
    """Wrap a loaded ``DoperNet`` and its configuration as a resident timed model."""
    config = export.model_config(cfg)
    crop_size = int(config["crop_size"])
    return TimedModel(
        obj_id=obj_id,
        detect=functools.partial(
            _detect, model, cfg, device, keypoints_mm, score_threshold, max_detections
        ),
        keypoints_mm=keypoints_mm,
        crop_size=crop_size,
        beta=float(config["keypoint_heatmap_beta"]),
        checkpoint_sha256=checkpoint_sha256,
        keypoints_sha256=keypoints_sha256,
        keypoint_head=config["keypoint_head"],
        warm_crop=functools.partial(_warm_crop, model, crop_size, device),
    )


def load_models(
    bank: Mapping[int, Path],
    *,
    device: str,
    score_threshold: float = export.SCORE_THRESHOLD,
    max_detections: int = export.MAX_DETECTIONS,
) -> list[TimedModel]:
    """Load every model of a bank onto ``device`` (all resident), as ``export.load_bank`` does.

    Args:
        bank: Object id to run directory (``work/last.pth``, ``geometry/keypoints_3d.json``).
        device: Torch device string.
        score_threshold: Detection score threshold.
        max_detections: Detections per object and image.

    Returns:
        The timed models sorted by object id.
    """
    models = []
    for obj in sorted(bank):
        run = export.load_run(bank[obj], device)
        models.append(
            timed_model(
                obj,
                run.model,
                run.cfg,
                run.keypoints,
                device=device,
                checkpoint_sha256=run.checkpoint_sha256,
                keypoints_sha256=run.keypoints_sha256,
                score_threshold=score_threshold,
                max_detections=max_detections,
            )  # fmt: skip
        )
    return models
