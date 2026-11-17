# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prefixed status lines for training and inference command-line output."""

from __future__ import annotations

PREFIX = "[doper]"


def log(message: str) -> None:
    """Print ``message`` to stdout behind the package prefix."""
    print(f"{PREFIX} {message}")
