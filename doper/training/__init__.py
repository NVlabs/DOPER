# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training components behind the :mod:`doper.train` entry point.

* :mod:`.recipe` - command-line flags, config files and config validation;
* :mod:`.data` - training sources and batch composition;
* :mod:`.schedule` - optimizer and learning-rate schedules;
* :mod:`.checkpoints` - checkpoints, resume, initialization and provenance;
* :mod:`.loop` - the per-epoch optimization loop and validation metrics.
"""
