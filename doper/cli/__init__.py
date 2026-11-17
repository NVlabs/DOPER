# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``doper`` command: ``prepare``, ``train``, ``review``, ``evaluate`` and ``report``.

``RECIPE`` is a packaged recipe (``handal``, ``hopev2``, ``hot3d``, ``tless``) or a recipe
YAML. Every command takes ``--dry-run``, which prints the ordered steps with their resolved
inputs and outputs and touches no data. Subcommand modules import torch and the BOP stack only
when a step runs, so ``doper --help`` and dry runs stay light.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from ..bop.sources import SourceNotReadyError
from ..console import log
from . import evaluate, prepare, report, review, train
from .common import MissingExtraError


def build_parser() -> argparse.ArgumentParser:
    """Build the ``doper`` parser."""
    parser = argparse.ArgumentParser(
        prog="doper",
        description="DOPER BOP recipes: prepare a dataset, train, review labels, evaluate, report.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for module in (prepare, train, review, evaluate, report):
        module.add_parser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``doper`` command.

    Args:
        argv: Arguments without the program name; None reads ``sys.argv``.

    Returns:
        Zero on success; one when a step refuses invalid or changed data, a missing input, a
        missing optional dependency (``MissingExtraError``) or an existing output, or fails on
        the file system (``OSError``, for example an export
        journal locked by another run). The error is logged without a traceback. Other
        exceptions propagate; argparse and existing-output refusals exit via ``SystemExit``.
    """
    args = build_parser().parse_args(argv)
    try:
        return args.run(args)
    except (ValueError, OSError, SourceNotReadyError, MissingExtraError) as err:
        log(f"error: {err}")
        return 1


__all__ = ["build_parser", "main"]
