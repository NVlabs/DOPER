# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Evaluate DOPER checkpoints on labeled (synthetic) validation splits.

The training loop reports ``kp_err_px`` / ``det_center_err_px`` / ``mask_iou`` on its own val
split; this runs the same ``evaluate`` for any checkpoint x split matrix, for example to score
models trained on different render engines against each other's validation splits. Model
architecture comes from the checkpoint (``infer.load_model``), the data layout is
``<root>/<split>/{images,annotations.json}``.

Example::

    python -m doper.evaluation.synthetic \
        --checkpoint engine_a=runs/a/best.pth --checkpoint engine_b=runs/b/best.pth \
        --data-root engine_a=data/a --data-root engine_b=data/b \
        --out-json cross_engine.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from doper.config import DataConfig, DoperConfig, ModelConfig
from doper.dataset import DoperDataset
from doper.infer import load_model
from doper.performance import resolve_device
from doper.training.loop import evaluate

from .io import parse_named_inputs, write_json

LOGGER = logging.getLogger(__name__)
REPORTED_METRICS = ("kp_err_px", "det_center_err_px", "mask_iou")

Matrix = dict[str, dict[str, dict[str, float]]]


def format_matrix(results: Matrix, metric: str) -> str:
    """Format models as rows and dataset labels as columns.

    Args:
        results: Nested mapping indexed by checkpoint label, dataset label and
            metric name, as returned by ``evaluate_matrix``.
        metric: Metric key to display, for example ``kp_err_px`` or ``mask_iou``.

    Returns:
        A fixed-width table preserving first-seen label order. Numeric cells
        use three decimal places; missing or nonnumeric cells display a dash.
    """
    models = list(results)
    splits: list[str] = []
    for per_split in results.values():
        splits += [s for s in per_split if s not in splits]
    width = max([len(metric)] + [len(m) for m in models] + [len(s) for s in splits] + [8])
    lines = [f"{metric:<{width}} " + " ".join(f"{s:>{width}}" for s in splits)]
    for model in models:
        cells = []
        for split in splits:
            value = results[model].get(split, {}).get(metric)
            numeric = isinstance(value, (int, float))
            cells.append(f"{value:>{width}.3f}" if numeric else f"{'-':>{width}}")
        lines.append(f"{model:<{width}} " + " ".join(cells))
    return "\n".join(lines)


def _score_checkpoint_on_root(
    checkpoint: str,
    root: str,
    root_name: str,
    *,
    split: str,
    batch_size: int,
    num_workers: int,
    device: str,
    kp_dark: bool,
) -> dict[str, float]:
    """Return the validation metrics (plus ``num_samples``) of one checkpoint on one root."""
    cfg = DoperConfig(
        data=DataConfig(root=root, val_split=split), model=ModelConfig(pretrained=False)
    )
    model = load_model(checkpoint, cfg, device)  # fills cfg.model from the checkpoint
    dataset = DoperDataset(cfg, "val")
    if len(dataset) == 0:
        raise ValueError(f"{root_name}/{split} contains no annotated validation samples")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )
    with torch.no_grad():
        metrics = evaluate(model, loader, cfg, device, kp_dark=kp_dark)
    scored = {k: float(v) for k, v in metrics.items()}
    scored["num_samples"] = len(dataset)
    return scored


def evaluate_matrix(
    checkpoints: Sequence[tuple[str, str]],
    roots: Sequence[tuple[str, str]],
    *,
    split: str,
    batch_size: int,
    num_workers: int,
    device: str,
    kp_dark: bool,
    log: Callable[[str], object] = LOGGER.info,
) -> Matrix:
    """Evaluate every checkpoint against every labeled dataset root.

    Args:
        checkpoints: Ordered (unique model label, checkpoint path) pairs.
        roots: Ordered (unique dataset label, root path) pairs. Each root contains
            ``<split>/images`` and ``<split>/annotations.json``.
        split: Validation subdirectory name beneath every root.
        batch_size: Number of images per validation batch.
        num_workers: DataLoader worker count; zero loads data in the main process.
        device: PyTorch device string; unavailable CUDA falls back to cpu.
        kp_dark: Whether to use DARK sub-bin keypoint decoding during validation.
        log: Callable accepting one progress string per checkpoint/dataset pair;
            defaults to this module's logger at INFO level.

    Returns:
        Nested ``results[model_label][dataset_label]`` metric dictionaries from
        ``evaluate``, plus ``num_samples``. Architecture comes from each checkpoint;
        metric values retain their training-validation units and may be non-finite.

    Raises:
        ValueError: A dataset contains no annotated validation samples.
    """
    device = resolve_device(device)
    results: Matrix = {}
    for model_name, checkpoint in checkpoints:
        results[model_name] = {}
        for root_name, root in roots:
            metrics = _score_checkpoint_on_root(
                checkpoint,
                root,
                root_name,
                split=split,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                kp_dark=kp_dark,
            )
            results[model_name][root_name] = metrics
            shown = " ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "num_samples")
            log(
                f"[eval_synth] {model_name} on {root_name}/{split} "
                f"({metrics['num_samples']} samples): {shown}"
            )
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    return results


def _build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser of ``main``."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="DOPER checkpoint (repeatable); architecture is read from it",
    )
    parser.add_argument(
        "--data-root",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Split root containing <split>/images and <split>/annotations.json (repeatable)",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--kp-dark",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DARK sub-bin keypoint decoding, as in training validation",
    )
    parser.add_argument("--out-json", default=None, help="Write results[model][split][metric] here")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run synthetic validation, log metric tables and optionally write JSON results.

    Args:
        argv: CLI argument strings excluding the program name, or None to read
            ``sys.argv[1:]``.

    Returns:
        Zero after successful evaluation and any requested JSON write.

    Raises:
        SystemExit: Argument parsing requests help or rejects invalid arguments.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        checkpoints = parse_named_inputs(args.checkpoint, "checkpoint")
        roots = parse_named_inputs(args.data_root, "data-root")
    except ValueError as exc:
        parser.error(str(exc))
    results = evaluate_matrix(
        checkpoints,
        roots,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        kp_dark=args.kp_dark,
    )
    for metric in REPORTED_METRICS:
        LOGGER.info("\n%s", format_matrix(results, metric))
    if args.out_json:
        out_json = Path(args.out_json)
        payload = {
            "split": args.split,
            "kp_dark": args.kp_dark,
            "checkpoints": dict(checkpoints),
            "data_roots": dict(roots),
            "results": results,
        }
        write_json(out_json, payload)
        LOGGER.info("wrote %s", out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
