# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration dataclasses for the DOPER unified pose model.

A single ``DoperConfig`` controls the dataset layout, the network sizes, and
the joint-loss weights so that training, inference, and evaluation all agree on
the same conventions. Defaults suit the rendered datasets of ``rendering/`` (640x640 renders,
64 keypoints, one instance per image).
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field

# Repository-wide DOPER architecture policy. ConvNeXt-Small (the paper
# recipe) is the default for new models; ConvNeXt-Tiny and ConvNeXt-Base (both
# DINOv3-pretrained) are also supported production backbones, selectable
# directly via --backbone with no extra opt-out required. Anything outside this set is a genuine
# nonstandard backbone (legacy ResNet reproduction, etc.) and needs
# --allow-nonstandard-backbone.
DEFAULT_DOPER_BACKBONE = "convnext_small.dinov3_lvd1689m"
SUPPORTED_DOPER_BACKBONES = (
    "convnext_tiny.dinov3_lvd1689m",
    "convnext_small.dinov3_lvd1689m",
    "convnext_base.dinov3_lvd1689m",
)
DEFAULT_DOPER_BACKBONE_LR_MULTIPLIER = 0.1


@dataclass
class DataConfig:
    """Where the COCO-format render splits live.

    ``root`` should contain ``train/`` and ``val/`` subdirs, each with an
    ``images/`` folder, a ``masks/`` folder, and an ``annotations.json``.
    """

    root: str
    train_split: str = "train"
    val_split: str = "val"
    images_dirname: str = "images"
    ann_filename: str = "annotations.json"
    # keypoints_3d.json is needed only for PnP at inference time.
    keypoints_3d_json: str | None = None
    # Fixed ordered CAD-curve vocabulary used by the optional crop curve head.
    curve_schema_json: str | None = None
    # Pixel value threshold that separates the target object from the
    # background / other scene content in the rendered "visible_object" mask.
    mask_fg_threshold: int = 128
    # Optional CPU PNG decoder; historical recipes retain OpenCV.
    png_decoder: str = "opencv"
    # Legacy image/primary-instance behavior remains the default for replay.
    # "all" indexes one training/evaluation sample per annotated instance;
    # every sample still carries all detection boxes from its image.
    instance_sampling: str = "primary"
    # Single foreground head: union all available instance masks when requested.
    # A partially mask-labeled image has zero mask loss in union mode.
    mask_target_mode: str = "primary"


def legacy_instance_defaults(config: dict) -> dict:
    """Normalize missing historical data-policy fields without weakening guards.

    Old config JSON meant primary-instance crops/masks. Explicit new values are
    retained, so frozen-recipe comparisons still reject all/union changes.
    """
    result = copy.deepcopy(config)
    if isinstance(result.get("data"), dict):
        result["data"].setdefault("instance_sampling", "primary")
        result["data"].setdefault("mask_target_mode", "primary")
    if isinstance(result.get("aug"), dict):
        # SPEED+ illumination augmentations postdate older recipes; all are off by default.
        for key, value in asdict(AugConfig()).items():
            if key.startswith(("directional_brightness", "lens_flare")):
                result["aug"].setdefault(key, value)
    return result


@dataclass
class ModelConfig:
    """Network architecture: backbone, detection path and keypoint/curve heads."""

    # All new DOPER models default to pretrained DINOv3 ConvNeXt-Small (paper recipe).
    # DINOv3 Tiny/Base are supported choices; inference honors checkpoint metadata.
    backbone: str = DEFAULT_DOPER_BACKBONE
    pretrained: bool = True
    num_classes: int = 1
    num_keypoints: int = 64

    # Detection / mask path (full image).
    det_input_size: int = 640
    det_stride: int = 4  # stride of the shared detection feature map
    # Aspect-preserving letterbox (gray pad) instead of an anisotropic squash to
    # the square det input. Keeps the object's apparent aspect ratio consistent
    # across render (square) and real (non-square) frames; must match between
    # training and inference. See targets.letterbox_image.
    letterbox: bool = True
    neck_channels: int = 128
    head_channels: int = 64

    # Keypoint path (top-down crop).
    crop_size: int = 512
    crop_pad_factor: float = 1.25
    # ``heatmap`` is the default for new models (paper recipe); ``simcc`` remains
    # supported and is what older checkpoints without this field are loaded as.
    # ``heatmap`` predicts a joint 2D distribution at crop stride 4, avoiding
    # invalid x/y cross-pairings from independent SimCC marginals.
    keypoint_head: str = "heatmap"
    keypoint_heatmap_sigma: float = 1.5  # in stride-4 heatmap pixels
    simcc_split_ratio: float = 2.0
    simcc_sigma: float = 6.0
    # RTMCC keypoint head: embedding dim + GAU (cross-keypoint self-attention).
    gau_hidden_dims: int = 256
    gau_s: int = 128
    gau_expansion: int = 2

    # Optional coarse rotation supervision from the crop keypoint FPN.
    rotation_aux_enabled: bool = False
    rotation_num_bins: int = 256
    rotation_pool_size: int = 4
    rotation_bin_seed: int = 42
    rotation_logit_scale: float = 30.0

    # Optional named CAD-curve path (top-down crop, independent sigmoid maps).
    # Disabled by default so legacy checkpoints retain their exact architecture.
    curve_head_enabled: bool = False
    num_curves: int = 0
    curve_head_channels: int = 64
    curve_target_sigma: float = 1.5  # in stride-4 heatmap pixels
    curve_schema_version: int = 0
    curve_names: tuple[str, ...] = ()
    curve_geometry_sha256: str = ""
    curve_object_id: str = ""


@dataclass
class AugConfig:
    """Photometric, sensor, and geometric training augmentation.

    Appearance transforms do not change labels. Geometric transforms are
    applied to pixels and their matching detection/keypoint targets together.
    The detection image and keypoint crop have independent geometry paths;
    all augmentation is train-split only.
    """

    enabled: bool = True
    # Smooth multiplicative illumination gradient across the image. This is
    # applied before the other photometric/sensor transforms so it modifies the
    # clean scene signal while leaving geometry and exact-black pixels intact.
    # Disabled by default to preserve existing recipe behavior.
    directional_brightness: bool = False
    directional_brightness_prob: float = 0.4
    directional_brightness_max_strength: float = 0.55
    # Additive, camera-space veiling glare / lens-flare augmentation. This is
    # applied once to the full source image before detector resizing and
    # keypoint cropping, so both branches see the same optical artifact.
    lens_flare: bool = False
    lens_flare_prob: float = 0.35
    lens_flare_max_strength: float = 1.0
    # Compute the smooth flare layer at this fraction of source resolution and
    # upsample before compositing. The source pixels themselves are never
    # resized. Quarter resolution is visually equivalent and much cheaper for
    # 1920x1200 SPEED+ frames; 1.0 preserves the original implementation.
    lens_flare_field_scale: float = 1.0
    # Brightness/contrast/HSV jitter, applied together with this probability.
    photo_prob: float = 0.8
    brightness: float = 0.2  # +/- fraction of full range
    contrast: float = 0.2  # +/- fraction around 1.0
    hue: float = 0.04  # +/- fraction of the 180-degree OpenCV hue range
    saturation: float = 0.3  # +/- fraction around 1.0
    value: float = 0.2  # +/- fraction around 1.0 (HSV value)
    # Gaussian blur.
    blur_prob: float = 0.2
    blur_max_sigma: float = 1.5
    # Motion blur.
    motion_blur_prob: float = 0.2
    motion_blur_max_kernel: int = 9
    # Additive Gaussian sensor noise (std in 0-255 units).
    noise_prob: float = 0.3
    noise_max_std: float = 12.0
    # Mild JPEG compression artifacts.
    jpeg_prob: float = 0.3
    jpeg_quality_min: int = 40
    jpeg_quality_max: int = 90
    # Full desaturation to grayscale (replicated to 3 channels). Helps transfer
    # to infrared streams and color-cast or poorly lit real frames.
    grayscale_prob: float = 0.15

    # Top-down keypoint-crop geometric jitter. Teaches the keypoint head to
    # tolerate the imperfect boxes the detector produces at inference (vs the
    # exact GT box it would otherwise always see). Train split only.
    crop_scale_prob: float = 1.0
    crop_scale: float = 0.25  # crop size multiplied by 1 +/- this
    crop_shift_prob: float = 1.0
    crop_shift: float = 0.1  # center shifted by +/- this fraction of crop size
    crop_rotate_prob: float = 0.6
    crop_rotate: float = 45.0  # in-plane rotation, +/- degrees

    # Right-angle rotations for full orientation coverage (the object appears
    # at all in-plane angles in the real data). For the detection image
    # boxes stay axis-aligned (w/h swap); for the keypoint crop these stack on
    # the +/-crop_rotate jitter and are folded into the same affine resampling.
    # ``*_prob`` is the chance of a non-zero k in {1,2,3}; 0.75 => approximately
    # uniform over all 4 orientations.
    det_rot90_prob: float = 0.75
    crop_rot90_prob: float = 0.75

    # Detection-image geometric jitter. Chirality-preserving on purpose: only
    # uniform scale/zoom + translation (no flips/reflections), so the keypoint
    # head's handedness is never corrupted. Applied to the full detection image
    # together with its boxes and mask; the keypoint crop is built separately
    # from the original image and is unaffected. Mirrors YOLO's scale/translate.
    det_scale_prob: float = 1.0
    det_scale: float = 0.5  # zoom by a factor in [1 - this, 1 + this]
    det_translate_prob: float = 1.0
    det_translate: float = 0.1  # shift by +/- this fraction of the det input size
    # Small continuous in-plane rotation of the detection image (camera roll).
    # Chirality-preserving (proper rotation, no reflection). Folds into the same
    # affine as scale/translate so there is only one warp. Off by default: the
    # renders already show the object at every in-plane angle and the det path
    # already has discrete 90-degree rotations, so this is largely redundant and
    # it inflates GT boxes slightly (axis-aligned enclosure of rotated corners).
    # Set the prob > 0 to A/B it in isolation.
    det_rotate_prob: float = 0.0
    det_rotate: float = 10.0  # +/- degrees
    # Random erasing / cutout on the detection image for occlusion robustness.
    # Pixels only - boxes and mask targets are left unchanged.
    det_erase_prob: float = 0.4
    det_erase_count: int = 1
    det_erase_min_area: float = 0.02  # per-rect area as a fraction of the image
    det_erase_max_area: float = 0.15


@dataclass
class LossConfig:
    """Weights and temperatures of the joint training loss."""

    rotation_weight: float = 1.0
    rotation_sigma_deg: float = 15.0
    heatmap_weight: float = 1.0
    wh_weight: float = 0.1
    offset_weight: float = 1.0
    mask_weight: float = 1.0
    keypoint_weight: float = 1.0
    # Temperature applied to the SimCC logits before softmax in the KL loss.
    # Matches RTMPose's KLDiscretLoss(beta=10): sharpens the predicted 1D
    # coordinate distribution over the (many) bins for a more precise argmax.
    simcc_beta: float = 10.0
    # The 2D heatmap head is trained directly as a normal categorical
    # distribution, so it does not need RTMPose's logit sharpening.
    keypoint_heatmap_beta: float = 1.0
    curve_weight: float = 1.0
    curve_bce_weight: float = 1.0
    curve_dice_weight: float = 1.0


@dataclass
class BackgroundConfig:
    """Hard-negative background photos + synthetic-on-background composites.

    Teaches the detector that object-free scenery is *not* the object (and, via
    composites, that the object on that same scenery still is). The negatives and
    composites are fed through the same letterbox + augmentation path as ordinary
    samples. Splits are scene-grouped in ``splits_json`` so crops from one source
    photo never leak across train/val/test.
    """

    enabled: bool = False
    # Directory of source background photos and the scene-grouped split file.
    background_dir: str = ""
    splits_json: str = ""
    train_split: str = "train"
    # Synthetic COCO root whose foreground masks/keypoints are pasted onto
    # background crops for composites. Empty => reuse the primary ``data.root``.
    composite_source_root: str = ""
    composite_source_split: str = "train"

    # Per-batch source balance (must sum to the training batch size).
    positive_per_batch: int = 48
    negative_per_batch: int = 8
    composite_per_batch: int = 8

    # Dynamic crop sampling for negatives.
    crop_min_frac: float = 0.20  # crop side as a fraction of min(H, W)
    crop_max_frac: float = 1.0
    live_square_prob: float = 0.30  # center-square crop
    full_frame_prob: float = 0.10  # whole image (full-frame letterbox geometry)
    fp_mine_prob: float = 0.30  # crop centered on a false-positive box from fp_mine_json
    fp_mine_json: str = ""  # eval JSON with records[].top_bbox_xyxy (train only)
    fp_fill_min: float = 0.30  # false box occupies this fraction of the crop
    fp_fill_max: float = 0.70

    # Synthetic object compositing.
    # ``composite_prob`` is the probability that a primary synthetic sample has
    # its rendered background replaced in-place. It uses the ordinary shuffled
    # loader, so no source is oversampled and no negative-only examples are
    # introduced. The fixed-ratio stream is controlled by ``enabled`` and the
    # per-batch fields above.
    composite_prob: float = 0.0
    alpha_feather_sigma_min: float = 1.5
    alpha_feather_sigma_max: float = 3.0
    composite_erode_px: int = 1
    background_blur_sigma_min: float = 0.0
    background_blur_sigma_max: float = 0.0
    edge_blur_sigma: float = 0.0
    edge_blur_width: int = 0

    # Decode/cache controls.
    max_image_side: int = 2048  # downscale huge source photos on load
    # Nominal virtual sizes (the ratio sampler reshuffles/refills as needed).
    crops_per_source: int = 16
    seed: int = 0


@dataclass
class TrainConfig:
    """Optimizer, schedule, logging and runtime settings for training."""

    work_dir: str = "work_dirs/doper"
    epochs: int = 200
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    num_workers: int = 8
    lr: float = 2.0e-3
    # Fine-tune the DINOv3 foundation backbone more conservatively than newly
    # initialized DOPER necks and heads. Legacy runs may override this.
    backbone_lr_multiplier: float = DEFAULT_DOPER_BACKBONE_LR_MULTIPLIER
    weight_decay: float = 0.05
    warmup_iters: int = 500
    lr_schedule: str = "cosine"
    lr_cooldown_fraction: float = 0.2
    # Max global grad norm (0 disables). Guards against loss-spike divergence
    # (the CenterNet wh/focal terms can spike and blow a weight to NaN).
    grad_clip: float = 5.0
    val_interval: int = 5
    log_interval: int = 20
    save_interval: int = 10
    seed: int = 42
    device: str = "cuda"
    amp: bool = True
    amp_dtype: str = "float16"
    channels_last: bool = False
    prefetch_factor: int = 2
    opencv_threads: int | None = None
    reproducible: bool = False
    cudnn_benchmark: bool | None = None
    allow_tf32: bool | None = None
    # Cap iterations per epoch (debug / smoke runs). 0 means use the full set.
    max_iters_per_epoch: int = 0


@dataclass
class DoperConfig:
    """Top-level configuration bundling the data, model, loss, train and aug sections."""

    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)

    @property
    def det_feat_size(self) -> tuple[int, int]:
        """Return the ``(height, width)`` of the stride-reduced detection feature map."""
        s = self.model.det_input_size // self.model.det_stride
        return (s, s)
