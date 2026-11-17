# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``python -m doper.train --dry-run``: validate a training config without starting a run."""

from __future__ import annotations

import argparse
import os

from ..config import DoperConfig
from ..console import log
from ..keypoints import load_keypoints_3d


def _require_split(cfg: DoperConfig, split_dir: str) -> str:
    """Return the annotation file of a split, which must exist with its image directory."""
    base = os.path.join(cfg.data.root, split_dir)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"split directory is missing: {base}")
    ann_path = os.path.join(base, cfg.data.ann_filename)
    if not os.path.isfile(ann_path):
        raise FileNotFoundError(f"split annotations are missing: {ann_path}")
    images = os.path.join(base, cfg.data.images_dirname)
    if not os.path.isdir(images):
        raise FileNotFoundError(f"split image directory is missing: {images}")
    return ann_path


def _check_keypoints(cfg: DoperConfig) -> str:
    """Check the 3D keypoint file against ``num_keypoints``; return a one-line description."""
    path = cfg.data.keypoints_3d_json
    if not path:
        return "not set (pass --keypoints-3d to check the layout)"
    if not os.path.isfile(path):
        raise FileNotFoundError(f"keypoints_3d file is missing: {path}")
    count = int(load_keypoints_3d(path).shape[0])
    if count != cfg.model.num_keypoints:
        raise ValueError(
            f"--num-keypoints is {cfg.model.num_keypoints} but {path} has {count} keypoints"
        )
    return f"{path} ({count} keypoints)"


def check_dry_run(cfg: DoperConfig, args: argparse.Namespace) -> None:
    """Check the data splits and keypoint file, then print the resolved plan.

    Runs after config validation and before any runtime setup, so it builds no model, downloads
    no weights and creates no ``work_dir``.

    Args:
        cfg: Validated training config.
        args: Parsed training arguments.

    Raises:
        FileNotFoundError: A split directory, its annotations or the keypoint file is missing.
        ValueError: ``num_keypoints`` differs from the keypoint file.
    """
    for split_dir in (cfg.data.train_split, cfg.data.val_split):
        _require_split(cfg, split_dir)
    keypoints = _check_keypoints(cfg)
    log("dry run: configuration and data are valid; nothing was built or written")
    log(f"  data_root: {cfg.data.root} (splits {cfg.data.train_split}, {cfg.data.val_split})")
    log(f"  keypoints_3d: {keypoints}")
    log(f"  backbone: {cfg.model.backbone} (pretrained={cfg.model.pretrained})")
    log(f"  det_input_size: {cfg.model.det_input_size}  crop_size: {cfg.model.crop_size}")
    if cfg.model.rotation_aux_enabled:
        log(
            f"  rotation_aux: {cfg.model.rotation_num_bins} bins, "
            f"loss_weight={cfg.loss.rotation_weight}; requires pose-labeled training crops"
        )
    log(f"  epochs: {cfg.train.epochs}  batch_size: {cfg.train.batch_size}")
    log(f"  max_iters_per_epoch: {cfg.train.max_iters_per_epoch}  device: {cfg.train.device}")
    log(f"  work_dir: {cfg.train.work_dir}")
