# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pseudo-label exporters: turn confident model poses or reviewed boxes into COCO training data.

Each exporter is a command-line module with ``main(argv=None) -> int``:

* ``python -m doper.pseudo_labels.multiview``: joint two-camera PnP poses projected into
  keypoint, box and optional mask labels for both cameras (and their left-IR images).
* ``python -m doper.pseudo_labels.recorded_boxes``: reviewed detector boxes on a recording,
  copied unchanged to the colocated left-IR frames.
* ``python -m doper.pseudo_labels.hdf5_episodes``: per-camera box and mask labels for an
  episode split stored as HDF5 (needs the ``hdf5`` extra).
* ``python -m doper.pseudo_labels.mix_root``: link a flat export into a training mix root.

``doper.pseudo_labels.monocular`` is a library without a command line: strict-gate box/mask pools
from the staged onboarding and test views, their audit and the static+dynamic merge. It writes
only to a new directory.

Every command-line exporter refuses to write into an existing output directory unless ``--force`` is
given, in which case that directory is replaced. An output directory that equals, contains or
lies inside one of the run's inputs is always refused, so ``--force`` never deletes inputs.
"""
