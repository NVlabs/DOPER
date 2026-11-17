# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line recipe: flags, config files, the resolved config and its checks."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import yaml

from ..cad_curves import load_curve_schema
from ..config import (
    DEFAULT_DOPER_BACKBONE,
    DEFAULT_DOPER_BACKBONE_LR_MULTIPLIER,
    SUPPORTED_DOPER_BACKBONES,
    AugConfig,
    BackgroundConfig,
    DataConfig,
    DoperConfig,
    LossConfig,
    ModelConfig,
    TrainConfig,
)
from ..console import PREFIX, log

if TYPE_CHECKING:
    from .data import TrainSet


def load_config_file(path: str) -> dict:
    """Load a YAML/JSON training config into a flat dict of argparse defaults.

    Keys use the same names as the CLI flags with dashes replaced by underscores
    (e.g. ``data-root`` or ``data_root`` -> ``data_root``). ``mix_data_root`` may
    be a single string or a list. Unknown keys raise so typos fail loudly.

    Args:
        path: ``.yaml``/``.yml`` or JSON file.

    Returns:
        The config as argparse destination names to values.

    Raises:
        ValueError: If the file is not a mapping at the top level.
    """
    with open(path) as f:
        is_yaml = path.endswith((".yaml", ".yml"))
        raw = (yaml.safe_load(f) or {}) if is_yaml else json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"config {path} must be a mapping at the top level")
    return {str(k).replace("-", "_"): v for k, v in raw.items()}


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the dataset source flags."""
    parser.add_argument("--data-root", required=True, help="primary COCO root (drives train + val)")
    parser.add_argument(
        "--mix-data-root",
        action="append",
        default=[],
        metavar="ROOT",
        help="additional COCO root mixed into TRAIN only (repeatable); e.g. real bbox-only data",
    )
    parser.add_argument(
        "--keypoint-only-mix-data-root",
        action="append",
        default=[],
        metavar="ROOT",
        help="additional TRAIN-only COCO root whose samples contribute only "
        "keypoint loss; detection and mask losses are explicitly disabled",
    )
    parser.add_argument(
        "--mix-per-batch",
        type=int,
        nargs="*",
        default=None,
        metavar="N",
        help="fixed samples per batch from each --mix-data-root; the remainder "
        "comes from the primary root",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--work-dir", default="work_dirs/doper")
    parser.add_argument("--keypoints-3d", default=None)


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the architecture flags."""
    parser.add_argument(
        "--backbone",
        default=DEFAULT_DOPER_BACKBONE,
        help=(
            f"shared backbone; default is {DEFAULT_DOPER_BACKBONE}. Also "
            f"supported without any opt-out: {SUPPORTED_DOPER_BACKBONES}"
        ),
    )
    parser.add_argument(
        "--allow-nonstandard-backbone",
        action="store_true",
        help=(
            "explicitly opt out of the supported-backbone policy; reserved "
            "for controlled ablations and legacy reproduction"
        ),
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--num-keypoints", type=int, default=64)
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument(
        "--letterbox",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="aspect-preserving letterbox of the det input instead of a square squash",
    )
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument(
        "--keypoint-head",
        choices=("simcc", "heatmap"),
        default="heatmap",
        help="keypoint representation; heatmap uses a joint stride-4 2D distribution",
    )
    parser.add_argument(
        "--keypoint-heatmap-sigma",
        type=float,
        default=1.5,
        help="2D target Gaussian sigma in stride-4 heatmap pixels",
    )


def _add_rotation_arguments(parser: argparse.ArgumentParser) -> None:
    """Add opt-in coarse rotation supervision settings."""
    parser.add_argument(
        "--rotation-aux",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="train a coarse rotation classifier from pose-labeled crops",
    )
    parser.add_argument("--rotation-num-bins", type=int, default=256)
    parser.add_argument("--rotation-pool-size", type=int, default=4)
    parser.add_argument("--rotation-bin-seed", type=int, default=42)
    parser.add_argument("--rotation-logit-scale", type=float, default=30.0)
    parser.add_argument("--rotation-loss-weight", type=float, default=1.0)
    parser.add_argument("--rotation-sigma-deg", type=float, default=15.0)


def _add_curve_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the CAD curve head flags."""
    parser.add_argument(
        "--curve-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable one crop heatmap channel per named CAD curve",
    )
    parser.add_argument(
        "--curve-schema",
        default=None,
        help="versioned curves_3d.json defining the fixed ordered vocabulary",
    )
    parser.add_argument(
        "--num-curves",
        type=int,
        default=0,
        help="optional expected schema channel count; 0 derives it from the schema",
    )
    parser.add_argument("--curve-head-channels", type=int, default=64)
    parser.add_argument(
        "--curve-target-sigma",
        type=float,
        default=1.5,
        help="curve ridge sigma in stride-4 heatmap pixels",
    )
    parser.add_argument("--curve-loss-weight", type=float, default=1.0)
    parser.add_argument("--curve-bce-weight", type=float, default=1.0)
    parser.add_argument("--curve-dice-weight", type=float, default=1.0)
    parser.add_argument(
        "--instance-sampling",
        choices=("primary", "all"),
        default="primary",
        help="all: one sample per instance, all boxes per image; epochs become instance-weighted",
    )
    parser.add_argument(
        "--mask-target-mode",
        choices=("primary", "union"),
        default="primary",
        help="union: union of all instance masks; partially mask-labeled images have zero "
        "mask loss",
    )
    parser.add_argument("--mask-threshold", type=int, default=128)


def _add_optimization_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the schedule, optimizer and runtime flags."""
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="microbatches per optimizer step (effective batch = batch size times this)",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--warmup-iters", type=int, default=500)
    parser.add_argument(
        "--lr-schedule",
        choices=("cosine", "constant_cooldown", "cosine_tail10"),
        default="cosine",
    )
    parser.add_argument(
        "--lr-cooldown-fraction",
        type=float,
        default=0.2,
        help="Fraction of total updates in final linear cooldown or cosine_tail10 tail "
        "(10%% peak to zero)",
    )
    parser.add_argument(
        "--backbone-lr-multiplier",
        type=float,
        default=DEFAULT_DOPER_BACKBONE_LR_MULTIPLIER,
        help=(
            "backbone LR relative to head LR; defaults to 0.1 for the "
            "default DINOv3 ConvNeXt-Small backbone"
        ),
    )
    parser.add_argument(
        "--grad-clip", type=float, default=5.0, help="max global grad norm (0 disables)"
    )
    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--png-decoder", choices=("opencv", "pyspng"), default="opencv")
    parser.add_argument("--opencv-threads", type=int, default=None)
    parser.add_argument(
        "--reproducible",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Opt in to fresh-run RNG isolation and strict deterministic algorithms; "
        "disables cuDNN benchmarking; requires fixed worker count and environment.",
    )
    parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--kp-dark",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DARK sub-bin keypoint decoding for the val metric (default on). "
        "--no-kp-dark for legacy argmax.",
    )


def _add_augmentation_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the photometric augmentation flags."""
    parser.add_argument(
        "--no-aug", action="store_true", help="disable photometric/sensor augmentation"
    )
    parser.add_argument(
        "--directional-brightness",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "enable smooth multiplicative directional-brightness augmentation "
            "before the existing photometric/sensor pipeline"
        ),
    )
    parser.add_argument(
        "--directional-brightness-prob",
        type=float,
        default=0.4,
        help="per-image probability when --directional-brightness is enabled",
    )
    parser.add_argument(
        "--directional-brightness-max-strength",
        type=float,
        default=0.55,
        help="maximum gradient strength s, giving endpoint gains 1-s and 1+s",
    )
    parser.add_argument(
        "--lens-flare",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "enable additive camera-space veiling glare, flare ghosts, rays, "
            "and highlight bloom before detector resizing and keypoint cropping"
        ),
    )
    parser.add_argument(
        "--lens-flare-prob",
        type=float,
        default=0.35,
        help="per-image probability when --lens-flare is enabled",
    )
    parser.add_argument(
        "--lens-flare-max-strength",
        type=float,
        default=1.0,
        help="maximum additive flare strength in [0, 1]",
    )
    parser.add_argument(
        "--lens-flare-field-scale",
        type=float,
        default=1.0,
        help=(
            "resolution fraction for the smooth flare field in (0, 1]; the "
            "source image remains at native resolution"
        ),
    )
    parser.add_argument("--max-iters-per-epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the config against the data and keypoint files, print the plan and "
        "exit; builds no model, downloads no weights and creates no work dir",
    )


def _add_background_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the hard-background negative and composite flags."""
    parser.add_argument(
        "--background",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="mix background-photo negatives + composites via a ratio batch sampler "
        "(needs --background-dir and --background-splits-json)",
    )
    parser.add_argument("--background-dir", default="", help="directory of background photos")
    parser.add_argument(
        "--background-splits-json", default="", help="scene-grouped split file of those photos"
    )
    parser.add_argument(
        "--composite-source-root",
        default="",
        help="synthetic COCO root for composites (default: --data-root)",
    )
    parser.add_argument(
        "--fp-mine-json",
        default="",
        help="background FP eval JSON (records[].top_bbox_xyxy) for hard-negative mining",
    )
    parser.add_argument(
        "--background-fp-mine-prob",
        type=float,
        default=0.30,
        help="probability of centering a non-full-frame negative crop on a mined FP",
    )
    parser.add_argument(
        "--background-fp-fill-min",
        type=float,
        default=0.30,
        help="minimum fraction of a mined crop occupied by the false proposal",
    )
    parser.add_argument(
        "--background-fp-fill-max",
        type=float,
        default=0.70,
        help="maximum fraction of a mined crop occupied by the false proposal",
    )
    parser.add_argument("--positive-per-batch", type=int, default=48)
    parser.add_argument("--negative-per-batch", type=int, default=8)
    parser.add_argument("--composite-per-batch", type=int, default=8)
    parser.add_argument("--background-crop-min-frac", type=float, default=0.20)
    parser.add_argument("--background-crop-max-frac", type=float, default=1.0)
    parser.add_argument("--background-live-square-prob", type=float, default=0.30)
    parser.add_argument("--background-full-frame-prob", type=float, default=0.10)
    parser.add_argument(
        "--composite-prob",
        type=float,
        default=0.0,
        help="probability of replacing a primary synthetic sample's background "
        "without changing natural-rate dataset sampling",
    )
    parser.add_argument("--background-blur-sigma-min", type=float, default=0.0)
    parser.add_argument("--background-blur-sigma-max", type=float, default=0.0)
    parser.add_argument("--edge-blur-sigma", type=float, default=0.0)
    parser.add_argument("--edge-blur-width", type=int, default=0)
    parser.add_argument("--crops-per-source", type=int, default=16)


def _add_initialization_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the resume, initialization and recipe-requirement flags."""
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--private-provenance",
        action="store_true",
        help=(
            "record full filesystem paths in training_provenance.json and checkpoints; "
            "by default path-valued entries are reduced to their basename"
        ),
    )
    parser.add_argument(
        "--init-from",
        default=None,
        help="initialize matching model weights from a checkpoint without optimizer state",
    )
    parser.add_argument(
        "--freeze-keypoint-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "exclude the keypoint head from optimization; useful for box/mask-only "
            "adaptation while allowing the shared backbone to update"
        ),
    )
    parser.add_argument(
        "--require-pretrained-backbone",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="fail recipe validation unless the backbone uses ImageNet weights",
    )
    parser.add_argument(
        "--require-no-init-from",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="fail recipe validation if --init-from or --resume is set",
    )
    parser.add_argument(
        "--require-init-from",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="fail recipe validation unless --init-from is set and --resume is not",
    )
    parser.add_argument(
        "--min-mix-keypoint-samples",
        type=int,
        default=0,
        help="minimum keypoint-bearing samples required across mix roots",
    )
    parser.add_argument(
        "--min-mix-mask-only-samples",
        type=int,
        default=0,
        help="minimum mask-bearing samples without keypoints required across mix roots",
    )
    parser.add_argument(
        "--min-keypoint-only-samples",
        type=int,
        default=0,
        help="minimum keypoint-bearing samples required across roots configured "
        "with --keypoint-only-mix-data-root",
    )


def _apply_config_defaults(parser: argparse.ArgumentParser, config_defaults: dict) -> None:
    """Seed parser defaults from a config file; unknown keys exit."""
    valid = {a.dest for a in parser._actions}
    unknown = set(config_defaults) - valid
    if unknown:
        raise SystemExit(f"{PREFIX} unknown config keys: {sorted(unknown)}")
    # A required arg (data_root) supplied via config must no longer be required.
    for action in parser._actions:
        if action.dest in config_defaults and getattr(action, "required", False):
            action.required = False
    parser.set_defaults(**config_defaults)


def _config_parser() -> argparse.ArgumentParser:
    """Parser of the ``--config`` flag alone."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--config", default=None, help="YAML/JSON training config; CLI flags override its values"
    )
    return pre


def build_parser(config_defaults: dict | None = None) -> argparse.ArgumentParser:
    """Build the training parser, seeded with config-file values as defaults.

    Args:
        config_defaults: Config values keyed by argparse destination; unknown keys exit.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        description="Train the DOPER unified model", parents=[_config_parser()]
    )
    _add_data_arguments(parser)
    _add_model_arguments(parser)
    _add_curve_arguments(parser)
    _add_rotation_arguments(parser)
    _add_optimization_arguments(parser)
    _add_augmentation_arguments(parser)
    _add_background_arguments(parser)
    _add_initialization_arguments(parser)
    if config_defaults:
        _apply_config_defaults(parser, config_defaults)
    return parser


def config_keys() -> frozenset[str]:
    """Return the keys a training config file may set (the argparse destinations)."""
    return frozenset(a.dest for a in build_parser()._actions if a.dest not in {"help", "config"})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the training command line, seeded by an optional ``--config`` file.

    Config-file values become argparse defaults, so explicit flags win.

    Args:
        argv: Arguments without the program name; None reads ``sys.argv``.

    Returns:
        The parsed arguments.
    """
    argv = None if argv is None else list(argv)
    # Pre-parse only --config so its values can seed argparse defaults (CLI wins).
    pre_args, _ = _config_parser().parse_known_args(argv)
    config_defaults: dict = {}
    if pre_args.config:
        config_defaults = load_config_file(pre_args.config)
    return build_parser(config_defaults).parse_args(argv)


def build_config(args: argparse.Namespace) -> DoperConfig:
    """Resolve parsed arguments into the full training config.

    Args:
        args: Parsed training arguments.

    Returns:
        The resolved config.
    """
    reproducible = getattr(args, "reproducible", False)
    return DoperConfig(
        data=DataConfig(
            root=args.data_root,
            train_split=args.train_split,
            val_split=args.val_split,
            keypoints_3d_json=args.keypoints_3d,
            curve_schema_json=args.curve_schema,
            mask_fg_threshold=args.mask_threshold,
            png_decoder=getattr(args, "png_decoder", "opencv"),
            instance_sampling=getattr(args, "instance_sampling", "primary"),
            mask_target_mode=getattr(args, "mask_target_mode", "primary"),
        ),
        model=ModelConfig(
            backbone=args.backbone,
            pretrained=not args.no_pretrained,
            num_classes=args.num_classes,
            num_keypoints=args.num_keypoints,
            det_input_size=args.det_size,
            crop_size=args.crop_size,
            letterbox=args.letterbox,
            keypoint_head=args.keypoint_head,
            keypoint_heatmap_sigma=args.keypoint_heatmap_sigma,
            rotation_aux_enabled=args.rotation_aux,
            rotation_num_bins=args.rotation_num_bins,
            rotation_pool_size=args.rotation_pool_size,
            rotation_bin_seed=args.rotation_bin_seed,
            rotation_logit_scale=args.rotation_logit_scale,
            curve_head_enabled=args.curve_head,
            num_curves=args.num_curves,
            curve_head_channels=args.curve_head_channels,
            curve_target_sigma=args.curve_target_sigma,
        ),
        loss=LossConfig(
            rotation_weight=args.rotation_loss_weight,
            rotation_sigma_deg=args.rotation_sigma_deg,
            curve_weight=args.curve_loss_weight,
            curve_bce_weight=args.curve_bce_weight,
            curve_dice_weight=args.curve_dice_weight,
        ),
        aug=AugConfig(
            enabled=not args.no_aug,
            directional_brightness=args.directional_brightness,
            directional_brightness_prob=args.directional_brightness_prob,
            directional_brightness_max_strength=(args.directional_brightness_max_strength),
            lens_flare=args.lens_flare,
            lens_flare_prob=args.lens_flare_prob,
            lens_flare_max_strength=args.lens_flare_max_strength,
            lens_flare_field_scale=args.lens_flare_field_scale,
        ),
        background=BackgroundConfig(
            enabled=args.background,
            background_dir=args.background_dir,
            splits_json=args.background_splits_json,
            composite_source_root=args.composite_source_root or args.data_root,
            fp_mine_json=args.fp_mine_json,
            fp_mine_prob=args.background_fp_mine_prob,
            fp_fill_min=args.background_fp_fill_min,
            fp_fill_max=args.background_fp_fill_max,
            positive_per_batch=args.positive_per_batch,
            negative_per_batch=args.negative_per_batch,
            composite_per_batch=args.composite_per_batch,
            crop_min_frac=args.background_crop_min_frac,
            crop_max_frac=args.background_crop_max_frac,
            live_square_prob=args.background_live_square_prob,
            full_frame_prob=args.background_full_frame_prob,
            composite_prob=args.composite_prob,
            background_blur_sigma_min=args.background_blur_sigma_min,
            background_blur_sigma_max=args.background_blur_sigma_max,
            edge_blur_sigma=args.edge_blur_sigma,
            edge_blur_width=args.edge_blur_width,
            crops_per_source=args.crops_per_source,
            seed=args.seed,
        ),
        train=TrainConfig(
            work_dir=args.work_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_workers=args.num_workers,
            lr=args.lr,
            backbone_lr_multiplier=args.backbone_lr_multiplier,
            warmup_iters=args.warmup_iters,
            lr_schedule=getattr(args, "lr_schedule", "cosine"),
            lr_cooldown_fraction=getattr(args, "lr_cooldown_fraction", 0.2),
            grad_clip=args.grad_clip,
            val_interval=args.val_interval,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            device=args.device,
            amp=not args.no_amp,
            amp_dtype=getattr(args, "amp_dtype", "float16"),
            channels_last=getattr(args, "channels_last", False),
            prefetch_factor=getattr(args, "prefetch_factor", 2),
            opencv_threads=getattr(args, "opencv_threads", None),
            reproducible=reproducible,
            cudnn_benchmark=(False if reproducible else getattr(args, "cudnn_benchmark", None)),
            allow_tf32=getattr(args, "allow_tf32", None),
            max_iters_per_epoch=args.max_iters_per_epoch,
            seed=args.seed,
        ),
    )


def validate_backbone_policy(backbone: str, allow_nonstandard_backbone: bool) -> None:
    """Restrict DOPER training to a supported backbone unless opted out.

    Args:
        backbone: Requested backbone name.
        allow_nonstandard_backbone: Explicit opt-out for ablations.

    Raises:
        ValueError: If the backbone is unsupported and not opted out.
    """
    if backbone in SUPPORTED_DOPER_BACKBONES:
        return
    if allow_nonstandard_backbone:
        log(
            "WARNING: explicit nonstandard-backbone opt-out: "
            f"{backbone!r}; supported backbones are {SUPPORTED_DOPER_BACKBONES}"
        )
        return
    raise ValueError(
        f"DOPER training requires one of the supported backbones "
        f"{SUPPORTED_DOPER_BACKBONES}; got {backbone!r}. Anything else is "
        "reserved for controlled ablations or legacy reproduction and "
        "requires --allow-nonstandard-backbone."
    )


def configure_curve_schema(cfg: DoperConfig) -> None:
    """Resolve the fixed ordered curve vocabulary before model construction.

    Args:
        cfg: Config updated in place with the schema's curve metadata.

    Raises:
        ValueError: If the curve head and schema flags are inconsistent.
    """
    if not cfg.model.curve_head_enabled:
        if cfg.data.curve_schema_json:
            raise ValueError("--curve-schema requires --curve-head")
        return
    if not cfg.data.curve_schema_json:
        raise ValueError("--curve-head requires --curve-schema")
    schema = load_curve_schema(cfg.data.curve_schema_json)
    if cfg.model.num_curves not in (0, schema.num_curves):
        raise ValueError(
            f"--num-curves={cfg.model.num_curves} does not match schema channel count "
            f"{schema.num_curves}"
        )
    cfg.model.num_curves = schema.num_curves
    cfg.model.curve_schema_version = schema.version
    cfg.model.curve_names = schema.names
    cfg.model.curve_geometry_sha256 = schema.geometry_sha256
    cfg.model.curve_object_id = schema.object_id


def _validate_curve_settings(cfg: DoperConfig) -> None:
    """Check the curve head size, target sigma and loss weights."""
    if cfg.model.curve_head_channels <= 0:
        raise ValueError("--curve-head-channels must be positive")
    if not math.isfinite(cfg.model.curve_target_sigma) or cfg.model.curve_target_sigma <= 0:
        raise ValueError("--curve-target-sigma must be finite and positive")
    curve_loss_values = (
        cfg.loss.curve_weight,
        cfg.loss.curve_bce_weight,
        cfg.loss.curve_dice_weight,
    )
    if any(not math.isfinite(value) or value < 0 for value in curve_loss_values):
        raise ValueError("curve loss weights must be finite and nonnegative")
    if cfg.loss.curve_bce_weight + cfg.loss.curve_dice_weight <= 0:
        raise ValueError("at least one curve BCE/Dice weight must be positive")


def _require_in_range(name: str, value: float, low: float, high: float, low_open: bool) -> None:
    """Raise unless ``low <= value <= high`` (``low < value`` when ``low_open``)."""
    inside = (low < value if low_open else low <= value) and value <= high
    if not inside:
        bracket = "(" if low_open else "["
        raise ValueError(f"{name} must be in {bracket}{low:g}, {high:g}], got {value}")


def _validate_augmentation_settings(cfg: DoperConfig) -> None:
    """Check the augmentation probabilities and strengths."""
    _require_in_range(
        "--directional-brightness-prob", cfg.aug.directional_brightness_prob, 0, 1, False
    )
    _require_in_range(
        "--directional-brightness-max-strength",
        cfg.aug.directional_brightness_max_strength,
        0,
        1,
        False,
    )
    _require_in_range("--lens-flare-prob", cfg.aug.lens_flare_prob, 0, 1, False)
    _require_in_range("--lens-flare-max-strength", cfg.aug.lens_flare_max_strength, 0, 1, False)
    _require_in_range("--lens-flare-field-scale", cfg.aug.lens_flare_field_scale, 0, 1, True)


def _validate_background_settings(cfg: DoperConfig) -> None:
    """Check the composite and background-negative settings."""
    background = cfg.background
    _require_in_range("--composite-prob", background.composite_prob, 0, 1, False)
    if background.enabled and background.composite_prob > 0:
        raise ValueError(
            "--composite-prob cannot be combined with the fixed-ratio --background stream"
        )
    _require_in_range("--background-fp-mine-prob", background.fp_mine_prob, 0, 1, False)
    if not (0.0 < background.fp_fill_min <= background.fp_fill_max <= 1.0):
        raise ValueError(
            "background FP fill fractions must satisfy 0 < min <= max <= 1, "
            f"got {background.fp_fill_min}, {background.fp_fill_max}"
        )
    if (
        background.background_blur_sigma_min < 0
        or background.background_blur_sigma_max < background.background_blur_sigma_min
    ):
        raise ValueError("background blur sigma range must satisfy 0 <= min <= max")


def validate_config(cfg: DoperConfig, args: argparse.Namespace) -> None:
    """Resolve the curve schema and reject inconsistent settings before any setup.

    Args:
        cfg: Config from :func:`build_config`; the curve schema is applied in place.
        args: Parsed training arguments.

    Raises:
        ValueError: If any setting is out of range or inconsistent.
    """
    validate_backbone_policy(cfg.model.backbone, args.allow_nonstandard_backbone)
    if cfg.model.rotation_aux_enabled:
        if cfg.model.keypoint_head != "heatmap":
            raise ValueError("--rotation-aux requires --keypoint-head heatmap")
        if cfg.model.rotation_num_bins < 2 or cfg.model.rotation_pool_size <= 0:
            raise ValueError("rotation bins must be >= 2 and pool size must be positive")
        if cfg.model.rotation_bin_seed < 0:
            raise ValueError("rotation bin seed must be nonnegative")
        for value in (
            cfg.model.rotation_logit_scale,
            cfg.loss.rotation_sigma_deg,
            cfg.loss.rotation_weight,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "rotation scale, sigma and loss weight must be finite and positive"
                )
    configure_curve_schema(cfg)
    if cfg.model.curve_head_enabled:
        _validate_curve_settings(cfg)
    _validate_augmentation_settings(cfg)
    _validate_background_settings(cfg)
    if cfg.train.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive")
    if cfg.train.reproducible and args.resume:
        raise ValueError(
            "--reproducible supports fresh runs only; resume does not "
            "restore RNG, sampler or persistent-worker state"
        )


def _validate_initialization_requirements(args: argparse.Namespace, cfg: DoperConfig) -> None:
    """Enforce the recipe's backbone and checkpoint-initialization requirements."""
    if args.require_pretrained_backbone and not cfg.model.pretrained:
        raise ValueError("recipe requires an ImageNet-pretrained backbone; remove --no-pretrained")
    if args.require_no_init_from and (args.init_from or args.resume):
        raise ValueError("recipe requires no DOPER initialization/resume checkpoint")
    if args.require_init_from and (not args.init_from or args.resume):
        raise ValueError("recipe requires a DOPER --init-from checkpoint and no --resume")


def _require_minimum(found: int, required: int, label: str) -> None:
    """Raise unless at least ``required`` samples of ``label`` were found."""
    if found < required:
        raise ValueError(f"recipe requires at least {required} {label}, found {found}")


def validate_training_sources(
    cfg: DoperConfig,
    args: argparse.Namespace,
    train_set: TrainSet,
    train_roots: list[str],
    keypoint_only_roots: list[str],
) -> list[Any]:
    """Check the recipe's supervision requirements against the built sources.

    Args:
        cfg: Resolved training config.
        args: Parsed training arguments.
        train_set: Training set, a ``ConcatDataset`` when several roots are used.
        train_roots: Training roots, primary first.
        keypoint_only_roots: Roots that contribute keypoint loss only.

    Returns:
        One dataset per training root.

    Raises:
        ValueError: If a supervision or initialization requirement is unmet.
    """
    from torch.utils.data import ConcatDataset

    from .data import curve_supervision_count, supervision_counts

    source_sets: list[Any] = (
        list(train_set.datasets) if isinstance(train_set, ConcatDataset) else [train_set]
    )
    if cfg.model.curve_head_enabled:
        curve_samples = sum(curve_supervision_count(dataset) for dataset in source_sets)
        if curve_samples <= 0:
            raise ValueError("curve head is enabled but no training annotation has curve_file")
    mix_counts = [
        supervision_counts(dataset, cfg.model.num_keypoints) for dataset in source_sets[1:]
    ]
    mix_keypoint_samples = sum(counts[0] for counts in mix_counts)
    mix_mask_only_samples = sum(counts[2] for counts in mix_counts)
    keypoint_only_samples = sum(
        counts[0]
        for root, counts in zip(train_roots[1:], mix_counts, strict=False)
        if root in set(keypoint_only_roots)
    )
    _validate_initialization_requirements(args, cfg)
    _require_minimum(mix_keypoint_samples, args.min_mix_keypoint_samples, "mix keypoint samples")
    _require_minimum(mix_mask_only_samples, args.min_mix_mask_only_samples, "mix mask-only samples")
    _require_minimum(keypoint_only_samples, args.min_keypoint_only_samples, "keypoint-only samples")
    if cfg.model.rotation_aux_enabled:
        from ..rotation_aux import annotation_quaternion

        pose_count = sum(
            annotation_quaternion(dataset.primary_annotation(i)) is not None
            for dataset in source_sets
            for i in range(len(dataset))
        )
        if not pose_count:
            raise ValueError(
                "--rotation-aux requires training annotations with object_pose "
                "or speedplus_q_vbs2tango_true"
            )
        log(f"rotation supervision: {pose_count} training crops")
    doper_init = "none" if not args.init_from and not args.resume else "checkpoint"
    log(
        "recipe validated: "
        f"backbone_init={'imagenet' if cfg.model.pretrained else 'random'} "
        f"doper_init={doper_init} "
        f"mix_keypoints={mix_keypoint_samples} "
        f"mix_mask_only={mix_mask_only_samples} "
        f"keypoint_only={keypoint_only_samples}"
    )
    if cfg.background.composite_prob > 0:
        primary_size = len(source_sets[0])
        expected_overall = cfg.background.composite_prob * primary_size / max(1, len(train_set))
        log(
            "stochastic composites: "
            f"primary_prob={cfg.background.composite_prob:.3f} "
            f"expected_overall={expected_overall:.3%}; "
            "natural source rates preserved"
        )
    return source_sets
