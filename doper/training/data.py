# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training data sources, supervision accounting and batch composition."""

from __future__ import annotations

import copy
import dataclasses
import math
import os
import weakref
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler

from ..background_data import BackgroundNegativeDataset
from ..config import DoperConfig
from ..console import log
from ..dataset import DoperDataset
from ..performance import loader_options

# supervision_counts results per dataset, keyed by keypoint count.
_SUPERVISION_COUNTS: weakref.WeakKeyDictionary[Any, dict[int, tuple[int, int, int]]] = (
    weakref.WeakKeyDictionary()
)

# What :func:`build_train_set` returns: one root's dataset, or several concatenated.
TrainSet = DoperDataset | ConcatDataset


def concat_sizes(dataset: ConcatDataset) -> list[int]:
    """Return the length of each sub-dataset of ``dataset``, in order."""
    bounds = [0, *dataset.cumulative_sizes]
    return [end - start for start, end in zip(bounds, bounds[1:], strict=False)]


def dataset_for_root(
    cfg: DoperConfig,
    root: str,
    split: str,
    composite_prob: float = 0.0,
    keypoint_only: bool = False,
) -> DoperDataset:
    """Build a DoperDataset for ``root`` reusing all other cfg settings.

    Lets us concatenate several COCO roots (e.g. synthetic keypoint data + real
    bbox-only data) into one training set. Per-sample ``has_kp`` / ``mask_weight``
    flags mean each source contributes only the supervision it actually carries.

    Args:
        cfg: Base config; only ``data.root`` is replaced.
        root: COCO root of this source.
        split: Split subdirectory to load.
        composite_prob: Probability of a stochastic background composite.
        keypoint_only: Whether this source contributes only keypoint loss.

    Returns:
        The dataset for ``root``.
    """
    sub = copy.copy(cfg)
    sub.data = dataclasses.replace(cfg.data, root=root)
    return DoperDataset(
        sub,
        split,
        composite_prob=composite_prob,
        keypoint_only=keypoint_only,
    )


def _primary_annotation(dataset: DoperDataset, index: int, annotations: list[dict]) -> dict:
    """Return the annotation that supervises sample ``index``."""
    marked = [a for a in annotations if a.get("is_primary", False)]
    if hasattr(dataset, "primary_annotation"):
        return dataset.primary_annotation(index)
    if len(marked) == 1:
        return marked[0]
    return max(annotations, key=lambda a: a.get("area", 0.0))


def supervision_counts(dataset: DoperDataset, num_keypoints: int) -> tuple[int, int, int]:
    """Count keypoint, mask, and mask-only samples without decoding images.

    The counts are memoized per dataset because the build log, the recipe
    checks and the provenance record all ask for them.

    Args:
        dataset: Dataset exposing ``ids`` and ``anns_by_img``.
        num_keypoints: Keypoints a sample needs to count as keypoint-supervised.

    Returns:
        ``(keypoint_samples, mask_samples, mask_only_samples)``.
    """
    memo = _SUPERVISION_COUNTS.setdefault(dataset, {})
    if num_keypoints in memo:
        return memo[num_keypoints]
    data_cfg = getattr(getattr(dataset, "cfg", None), "data", None)
    union_masks = getattr(data_cfg, "mask_target_mode", "primary") == "union"
    keypoint_samples = 0
    mask_samples = 0
    mask_only_samples = 0
    for index, image_id in enumerate(dataset.ids):
        annotations = dataset.anns_by_img[image_id]
        primary = _primary_annotation(dataset, index, annotations)
        has_keypoints = len(primary.get("keypoints", [])) >= 3 * num_keypoints
        if union_masks:
            has_mask = all(bool(a.get("mask_file")) for a in annotations)
        else:
            has_mask = bool(primary.get("mask_file"))
        keypoint_samples += int(has_keypoints)
        mask_samples += int(has_mask)
        mask_only_samples += int(has_mask and not has_keypoints)
    memo[num_keypoints] = (keypoint_samples, mask_samples, mask_only_samples)
    return memo[num_keypoints]


def curve_supervision_count(dataset: DoperDataset) -> int:
    """Count samples whose primary annotation carries a curve label.

    Args:
        dataset: Dataset exposing ``ids`` and ``anns_by_img``.

    Returns:
        Number of samples with a ``curve_file``.
    """
    count = 0
    for index, image_id in enumerate(dataset.ids):
        annotations = dataset.anns_by_img[image_id]
        primary = _primary_annotation(dataset, index, annotations)
        count += int(bool(primary.get("curve_file")))
    return count


def build_train_set(
    cfg: DoperConfig,
    roots: list[str],
    keypoint_only_roots: list[str] | None = None,
) -> TrainSet:
    """Build the (possibly concatenated) training set and log its supervision.

    Args:
        cfg: Training config.
        roots: Primary root followed by any mix roots.
        keypoint_only_roots: Roots that contribute keypoint loss only.

    Returns:
        A single dataset, or a ``ConcatDataset`` over ``roots`` in order.
    """
    keypoint_only = set(keypoint_only_roots or [])
    sets = [
        dataset_for_root(
            cfg,
            root,
            "train",
            composite_prob=cfg.background.composite_prob if index == 0 else 0.0,
            keypoint_only=root in keypoint_only,
        )
        for index, root in enumerate(roots)
    ]
    for r, s in zip(roots, sets, strict=True):
        keypoints, masks, mask_only = supervision_counts(s, cfg.model.num_keypoints)
        supervision = "keypoints_only" if r in keypoint_only else "all_available"
        log(
            f"  train source {r}: {len(s)} samples / {s.image_count} annotated "
            f"image records (instance_sampling={cfg.data.instance_sampling}, "
            f"mask_target={cfg.data.mask_target_mode}) "
            f"(keypoints={keypoints}, masks={masks}, mask_only={mask_only}, "
            f"curves={curve_supervision_count(s)}, "
            f"supervision={supervision})"
        )
    return sets[0] if len(sets) == 1 else ConcatDataset(sets)


class RatioBatchSampler(Sampler):
    """Yield batches with a fixed per-stream composition.

    ``stream_sizes`` are the lengths of the concatenated sub-datasets (in order)
    and ``per_batch`` the number of indices drawn from each stream per batch.
    Each stream is drawn without replacement from a reshuffled pool that refills
    when exhausted, so every positive is seen ~once per epoch while the smaller
    negative/composite streams are oversampled to their target batch share. This
    also guarantees every batch is mixed (>=1 positive), so the CenterNet loss
    never hits its all-negative normalization branch.
    """

    def __init__(
        self,
        stream_sizes: list[int],
        per_batch: list[int],
        num_batches: int,
        seed: int = 0,
    ) -> None:
        """Create the sampler.

        Args:
            stream_sizes: Length of each concatenated sub-dataset, in order.
            per_batch: Indices drawn from each stream per batch.
            num_batches: Batches per epoch.
            seed: Base seed; epoch ``e`` shuffles with ``seed + e``.

        Raises:
            ValueError: If any stream is empty.
        """
        self.stream_sizes = [int(s) for s in stream_sizes]
        self.per_batch = [int(n) for n in per_batch]
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        self._epoch = 0
        self.offsets: list[int] = []
        running = 0
        for size in self.stream_sizes:
            self.offsets.append(running)
            running += size
        if any(s <= 0 for s in self.stream_sizes):
            raise ValueError(f"all streams must be non-empty, got {self.stream_sizes}")

    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        """Restore the next epoch's shuffle after an epoch-boundary resume."""
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        """Yield one epoch of mixed batches and advance the epoch counter."""
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        pools = [
            list(rng.permutation(size) + off)
            for size, off in zip(self.stream_sizes, self.offsets, strict=True)
        ]
        cursors = [0] * len(pools)

        def draw(stream: int, count: int) -> list[int]:
            out: list[int] = []
            while count > 0:
                if cursors[stream] >= len(pools[stream]):
                    pools[stream] = list(
                        rng.permutation(self.stream_sizes[stream]) + self.offsets[stream]
                    )
                    cursors[stream] = 0
                take = min(count, len(pools[stream]) - cursors[stream])
                out.extend(pools[stream][cursors[stream] : cursors[stream] + take])
                cursors[stream] += take
                count -= take
            return out

        for _ in range(self.num_batches):
            batch: list[int] = []
            for stream, count in enumerate(self.per_batch):
                batch.extend(draw(stream, count))
            rng.shuffle(batch)
            yield batch


def _validate_positive_counts(
    positive_counts: list[int], num_streams: int, positive_per_batch: int
) -> None:
    """Reject per-source positive counts that do not match the batch composition."""
    if len(positive_counts) != num_streams:
        raise ValueError(
            f"positive per-source counts have {len(positive_counts)} values for {num_streams} roots"
        )
    if any(count <= 0 for count in positive_counts):
        raise ValueError(f"positive per-source counts must be positive: {positive_counts}")
    if sum(positive_counts) != positive_per_batch:
        raise ValueError(
            f"positive per-source counts {positive_counts} sum to {sum(positive_counts)}, "
            f"expected {positive_per_batch}"
        )


def build_background_train_loader(
    cfg: DoperConfig,
    positive_set: TrainSet,
    device: str,
    positive_per_source: Sequence[int] | None = None,
) -> DataLoader:
    """Build a fixed-ratio positive/negative/composite loader.

    ``positive_per_source`` optionally preserves an explicit ratio among the
    ordinary positive roots.  Without it, the historical behavior treats the
    entire (possibly concatenated) positive set as one naturally weighted
    stream.

    Args:
        cfg: Training config with background mixing enabled.
        positive_set: Ordinary (positive) training set.
        device: Training device; CUDA enables pinned memory.
        positive_per_source: Optional per-root positive counts per batch.

    Returns:
        A loader driven by a :class:`RatioBatchSampler`.

    Raises:
        ValueError: If the batch composition is inconsistent.
    """
    cfg_bg = cfg.background
    negatives = BackgroundNegativeDataset(cfg)
    composite_cfg = copy.copy(cfg)
    composite_cfg.data = dataclasses.replace(
        cfg.data, root=cfg_bg.composite_source_root or cfg.data.root
    )
    composites = DoperDataset(composite_cfg, cfg_bg.composite_source_split, force_composite=True)
    if positive_per_source is None:
        positive_streams = [positive_set]
        positive_sizes = [len(positive_set)]
        positive_counts = [cfg_bg.positive_per_batch]
    else:
        if not isinstance(positive_set, ConcatDataset):
            raise ValueError(
                "per-source positive background sampling requires multiple positive roots"
            )
        positive_streams = list(positive_set.datasets)
        positive_sizes = concat_sizes(positive_set)
        positive_counts = [int(count) for count in positive_per_source]
        _validate_positive_counts(positive_counts, len(positive_streams), cfg_bg.positive_per_batch)
    streams = [*positive_streams, negatives, composites]
    per_batch = [
        *positive_counts,
        cfg_bg.negative_per_batch,
        cfg_bg.composite_per_batch,
    ]
    if sum(per_batch) != cfg.train.batch_size:
        raise ValueError(
            f"background per-batch {per_batch} (sum {sum(per_batch)}) must equal "
            f"batch_size {cfg.train.batch_size}"
        )
    concat = ConcatDataset(streams)
    num_batches = max(
        1,
        max(
            math.ceil(size / max(1, count))
            for size, count in zip(positive_sizes, positive_counts, strict=True)
        ),
    )
    sampler = RatioBatchSampler(
        [*positive_sizes, len(negatives), len(composites)],
        per_batch,
        num_batches,
        seed=cfg.train.seed,
    )
    log(
        f"  background negatives: {len(negatives)} views, "
        f"composites: {len(composites)} samples, "
        f"batch streams={per_batch}, batches/epoch={num_batches}"
    )
    return DataLoader(
        concat,
        batch_sampler=sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=cfg.train.num_workers > 0,
        **loader_options(cfg.train),
    )


def resolve_train_roots(
    data_root: str, mix_data_roots: list[str], keypoint_only_roots: list[str]
) -> list[str]:
    """Order the training roots and reject ambiguous source roles.

    Args:
        data_root: Primary COCO root.
        mix_data_roots: Ordinary train-only mix roots.
        keypoint_only_roots: Train-only roots that contribute keypoint loss only.

    Returns:
        ``[data_root, *mix_data_roots, *keypoint_only_roots]``.

    Raises:
        ValueError: If a root has more than one role.
    """
    overlap = set(mix_data_roots) & set(keypoint_only_roots)
    if overlap:
        raise ValueError(
            f"roots cannot be both ordinary and keypoint-only mix sources: {sorted(overlap)}"
        )
    if data_root in keypoint_only_roots:
        raise ValueError("the primary data root cannot be keypoint-only")
    return [
        data_root,
        *mix_data_roots,
        *keypoint_only_roots,
    ]


def _check_mix_per_batch(mix_per_batch: list[int], train_roots: list[str]) -> None:
    """Require one ``--mix-per-batch`` value per mix root."""
    if len(mix_per_batch) != len(train_roots) - 1:
        raise ValueError(
            f"--mix-per-batch has {len(mix_per_batch)} values for {len(train_roots) - 1} mix roots"
        )


def _background_positive_per_source(
    cfg: DoperConfig, mix_per_batch: list[int] | None, train_roots: list[str]
) -> list[int] | None:
    """Resolve the per-root positive counts of a background batch."""
    if not mix_per_batch:
        return None
    mix_per_batch = list(mix_per_batch)
    _check_mix_per_batch(mix_per_batch, train_roots)
    primary_per_batch = cfg.background.positive_per_batch - sum(mix_per_batch)
    if primary_per_batch <= 0 or any(count <= 0 for count in mix_per_batch):
        raise ValueError(
            "invalid positive source composition for background "
            f"sampling: primary={primary_per_batch}, "
            f"mix={mix_per_batch}, positive total="
            f"{cfg.background.positive_per_batch}"
        )
    return [primary_per_batch, *mix_per_batch]


def _print_batch_composition(
    cfg: DoperConfig,
    train_roots: list[str],
    per_batch: list[int],
    source_sizes: list[int],
    num_batches: int,
) -> None:
    """Spell out what a fixed per-source batch mix resolved to."""
    # `mix_per_batch: 4` means "4 of batch_size", which is easy to misread as a
    # fraction of the background config's positive_per_batch (48) when
    # background compositing is off and that number does not apply.
    names = [os.path.basename(r.rstrip("/")) or r for r in train_roots]
    accumulation = cfg.train.gradient_accumulation_steps
    log(
        f"batch composition (effective batch "
        f"{cfg.train.batch_size * accumulation} "
        f"= {cfg.train.batch_size} x {accumulation} accum):"
    )
    total = sum(per_batch)
    for name, count, size in zip(names, per_batch, source_sizes, strict=False):
        log(f"  {count}/{total} per batch ({100 * count / total:3.0f}%)  {size:6d} samples  {name}")
    log(f"  {num_batches} batches/epoch -> {num_batches * total} samples/epoch")


def _build_mixed_train_loader(
    cfg: DoperConfig,
    train_set: TrainSet,
    train_roots: list[str],
    mix_per_batch: list[int],
    device: str,
) -> DataLoader:
    """Build a loader with a fixed number of samples per source root per batch."""
    mix_per_batch = list(mix_per_batch)
    _check_mix_per_batch(mix_per_batch, train_roots)
    primary_per_batch = cfg.train.batch_size - sum(mix_per_batch)
    if primary_per_batch <= 0 or any(count <= 0 for count in mix_per_batch):
        raise ValueError(
            f"invalid source batch composition: primary={primary_per_batch}, mix={mix_per_batch}"
        )
    if not isinstance(train_set, ConcatDataset):
        raise ValueError("--mix-per-batch requires at least one --mix-data-root")
    source_sizes = concat_sizes(train_set)
    per_batch = [primary_per_batch, *mix_per_batch]
    num_batches = max(1, math.ceil(source_sizes[0] / primary_per_batch))
    if cfg.data.instance_sampling == "all":
        num_batches = max(
            math.ceil(size / count) for size, count in zip(source_sizes, per_batch, strict=False)
        )
    sampler = RatioBatchSampler(source_sizes, per_batch, num_batches, seed=cfg.train.seed)
    _print_batch_composition(cfg, train_roots, per_batch, source_sizes, num_batches)
    return DataLoader(
        train_set,
        batch_sampler=sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=cfg.train.num_workers > 0,
        **loader_options(cfg.train),
    )


def build_train_loader(
    cfg: DoperConfig,
    train_set: TrainSet,
    train_roots: list[str],
    mix_per_batch: list[int] | None,
    device: str,
) -> DataLoader:
    """Build the training loader for the configured batch composition.

    Args:
        cfg: Training config.
        train_set: Training set built by :func:`build_train_set`.
        train_roots: Roots of ``train_set``, primary first.
        mix_per_batch: Optional fixed samples per batch from each mix root.
        device: Training device; CUDA enables pinned memory.

    Returns:
        A background-ratio, fixed-mix, or naturally shuffled loader.
    """
    if cfg.background.enabled:
        positive_per_source = _background_positive_per_source(cfg, mix_per_batch, train_roots)
        return build_background_train_loader(
            cfg,
            train_set,
            device,
            positive_per_source=positive_per_source,
        )
    if mix_per_batch:
        return _build_mixed_train_loader(cfg, train_set, train_roots, mix_per_batch, device)
    return DataLoader(
        train_set,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=cfg.data.instance_sampling != "all",
        persistent_workers=cfg.train.num_workers > 0,
        **loader_options(cfg.train),
    )


def build_val_loader(cfg: DoperConfig, val_set: Dataset, device: str) -> DataLoader:
    """Build the deterministic validation loader.

    Args:
        cfg: Training config.
        val_set: Validation dataset.
        device: Training device; CUDA enables pinned memory.

    Returns:
        An unshuffled loader over ``val_set``.
    """
    return DataLoader(
        val_set,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=cfg.train.num_workers > 0,
        **loader_options(cfg.train, stream="val"),
    )
