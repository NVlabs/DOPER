# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hard-negative backgrounds and synthetic-on-background composites.

A detector trained on renders can fire on object-free scenery of the deployment site. This
option addresses that at the *data* level, with no architecture change, by mixing two extra
streams into training:

* **negatives** - dynamic multi-scale crops of your background photos with all-zero
  detection/mask targets and masked keypoints, so the detector learns the scenery is not the
  object;
* **composites** - the rendered synthetic object (its foreground mask + 64 keypoints)
  alpha-blended onto one of those same background crops, so the network learns the *object*,
  not to reject the background's pixels.

Both share the letterbox + augmentation geometry of the ordinary dataset. Source
photos are scene-grouped in ``splits.json`` so crops never leak across splits.
This module deliberately does not import :mod:`doper.dataset` at import time to
avoid a cycle (``dataset`` imports helpers from here); the few needed helpers are
imported lazily inside methods.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from . import targets
from .augment import photometric_augment
from .config import BackgroundConfig, DoperConfig

# Smallest crop side (px) sampled from a background photo.
_MIN_CROP_SIDE = 16
# Decoded-photo bytes each worker may cache. A miss re-decodes a full photo,
# roughly tripling a background sample's cost, so the budget holds a typical
# pool (37 photos at 2048 px are ~350 MB) and only bounds very large pools.
MAX_CACHED_BACKGROUND_BYTES = 1 << 30
# Fill value for letterbox padding of RGB images.
_PAD_VALUE = 114
# Relative jitter of the false-positive-mining crop size and position.
_FP_JITTER = 0.15
# Smoothing sigma and gain applied to the composite edge band weight.
_EDGE_BAND_SIGMA = 0.8
_EDGE_BAND_GAIN = 0.65
# RNG stream used by the negative dataset (the composite stream of ``DoperDataset``).
_STREAM_NEGATIVE = 0


# ---------------------------------------------------------------------------
# Reproducible per-worker randomness
# ---------------------------------------------------------------------------
def worker_index_rng(
    seed: int,
    index: int,
    epoch: int = 0,
    stream: int = 0,
    occurrence: int = 0,
) -> np.random.Generator:
    """Return a deterministic RNG for one worker/index/epoch sample.

    ``stream`` separates independent decisions within a sample. ``occurrence``
    lets a dynamically sampled dataset produce different results when a sampler
    repeats an index, while preserving the sequence across equivalent runs.
    """
    worker = get_worker_info()
    worker_id = worker.id if worker is not None else 0
    entropy = [
        int(seed) & 0xFFFFFFFF,
        int(worker_id) & 0xFFFFFFFF,
        int(index) & 0xFFFFFFFF,
        int(epoch) & 0xFFFFFFFF,
        int(stream) & 0xFFFFFFFF,
        int(occurrence) & 0xFFFFFFFF,
    ]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def next_worker_occurrence(counters: dict[tuple[int, int], int], index: int) -> int:
    """Advance the deterministic repeat counter for this worker and index."""
    worker = get_worker_info()
    worker_id = worker.id if worker is not None else 0
    key = (int(worker_id), int(index))
    occurrence = counters.get(key, 0)
    counters[key] = occurrence + 1
    return occurrence


# ---------------------------------------------------------------------------
# Split manifest
# ---------------------------------------------------------------------------
def load_splits(splits_json: str) -> dict[str, list[str]]:
    """Load the scene-grouped split manifest (train/val/test -> filenames).

    Raises:
        ValueError: When no split file is configured.
    """
    if not splits_json:
        raise ValueError(
            "background training needs your background photos: set background.background_dir "
            "and background.splits_json (--background-dir, --background-splits-json)"
        )
    with open(splits_json) as f:
        raw = json.load(f)
    return {k: list(v) for k, v in raw.items() if k in ("train", "val", "test")}


def split_file_paths(cfg_bg: BackgroundConfig, split: str) -> list[str]:
    """Absolute paths of the background photos assigned to ``split``."""
    splits = load_splits(cfg_bg.splits_json)
    if split not in splits:
        raise KeyError(f"split {split!r} not in {cfg_bg.splits_json}")
    return [os.path.join(cfg_bg.background_dir, name) for name in splits[split]]


def load_fp_boxes(fp_mine_json: str) -> dict[str, np.ndarray]:
    """Map ``file_name -> top false-positive box [x1,y1,x2,y2]`` from an eval JSON."""
    if not fp_mine_json or not os.path.exists(fp_mine_json):
        return {}
    with open(fp_mine_json) as f:
        data = json.load(f)
    boxes: dict[str, np.ndarray] = {}
    for rec in data.get("records", []):
        box = rec.get("top_bbox_xyxy")
        if box is not None:
            boxes[os.path.basename(rec["file"])] = np.asarray(box, dtype=np.float32)
    return boxes


# ---------------------------------------------------------------------------
# Source image pool (per worker, lazily decoded + downscaled)
# ---------------------------------------------------------------------------
class BackgroundImagePool:
    """Lazily decodes background photos, downscaled to ``max_side`` on the long edge.

    Each dataloader worker gets its own least-recently-used cache of at most
    ``MAX_CACHED_BACKGROUND_BYTES`` of decoded photos.

    Args:
        paths: Background photo paths.
        max_side: Long-edge limit in pixels; non-positive disables downscaling.
    """

    def __init__(self, paths: list[str], max_side: int = 2048) -> None:
        """Create an empty cache over ``paths``."""
        self.paths = list(paths)
        self.max_side = int(max_side)
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cached_bytes = 0
        self._original_shapes: dict[int, tuple[int, int]] = {}

    def __len__(self) -> int:
        """Return the number of background photos."""
        return len(self.paths)

    def name(self, idx: int) -> str:
        """Return the file name of photo ``idx``."""
        return os.path.basename(self.paths[idx])

    def get_bgr(self, idx: int) -> np.ndarray:
        """Return photo ``idx`` as a cached, possibly downscaled BGR array.

        Raises:
            FileNotFoundError: If the photo cannot be decoded.
        """
        img = self._cache.get(idx)
        if img is not None:
            self._cache.move_to_end(idx)
            return img
        path = self.paths[idx]
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        h, w = bgr.shape[:2]
        self._original_shapes[idx] = (h, w)
        long_side = max(h, w)
        if self.max_side > 0 and long_side > self.max_side:
            scale = self.max_side / float(long_side)
            bgr = cv2.resize(
                bgr,
                (int(round(w * scale)), int(round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        self._cache[idx] = bgr
        self._cached_bytes += bgr.nbytes
        while self._cached_bytes > MAX_CACHED_BACKGROUND_BYTES and len(self._cache) > 1:
            self._cached_bytes -= self._cache.popitem(last=False)[1].nbytes
        return bgr

    def original_shape(self, idx: int) -> tuple[int, int]:
        """Original decoded ``(height, width)`` before pool downscaling."""
        self.get_bgr(idx)
        return self._original_shapes[idx]


# ---------------------------------------------------------------------------
# Crop-window sampling
# ---------------------------------------------------------------------------
def sample_crop_window(
    h: int, w: int, cfg_bg: BackgroundConfig, rng: np.random.Generator
) -> tuple[int, int, int, int]:
    """Random multi-scale crop window ``(x, y, cw, ch)`` inside the image.

    Roughly square-ish (aspect in {1, 4/3, 3/4}); side scales between
    ``crop_min_frac`` and ``crop_max_frac`` of the short image side.
    """
    short = min(h, w)
    frac = float(rng.uniform(cfg_bg.crop_min_frac, cfg_bg.crop_max_frac))
    base = max(float(_MIN_CROP_SIDE), frac * short)
    aspect = float(rng.choice([1.0, 4.0 / 3.0, 3.0 / 4.0]))
    cw = int(round(min(w, base * (aspect if aspect > 1 else 1.0))))
    ch = int(round(min(h, base / (aspect if aspect < 1 else 1.0))))
    cw = int(np.clip(cw, _MIN_CROP_SIDE, w))
    ch = int(np.clip(ch, _MIN_CROP_SIDE, h))
    x = int(rng.integers(0, max(1, w - cw + 1)))
    y = int(rng.integers(0, max(1, h - ch + 1)))
    return x, y, cw, ch


def live_square_window(h: int, w: int) -> tuple[int, int, int, int]:
    """Center square crop of the shorter side."""
    s = min(h, w)
    return (w - s) // 2, (h - s) // 2, s, s


def fp_mine_window(
    h: int,
    w: int,
    box_xyxy: np.ndarray,
    cfg_bg: BackgroundConfig,
    rng: np.random.Generator,
) -> tuple[int, int, int, int]:
    """Crop window centered on a false-positive box so the trigger fills the crop.

    The box (in the eval's original-image pixels) is scaled here into the pool's
    downscaled pixels by the caller. The crop side is chosen so the box occupies
    ``fp_fill_min..fp_fill_max`` of it, then jittered by +/-15%.
    """
    bx1, by1, bx2, by2 = [float(v) for v in box_xyxy]
    bw = max(1.0, bx2 - bx1)
    bh = max(1.0, by2 - by1)
    cx = 0.5 * (bx1 + bx2)
    cy = 0.5 * (by1 + by2)
    fill = float(rng.uniform(cfg_bg.fp_fill_min, cfg_bg.fp_fill_max))
    side = max(bw, bh) / max(fill, 1e-3)
    side *= 1.0 + float(rng.uniform(-_FP_JITTER, _FP_JITTER))
    cx += float(rng.uniform(-_FP_JITTER, _FP_JITTER)) * side
    cy += float(rng.uniform(-_FP_JITTER, _FP_JITTER)) * side
    side = float(np.clip(side, _MIN_CROP_SIDE, min(h, w)))
    x = int(np.clip(round(cx - side / 2.0), 0, max(0, w - side)))
    y = int(np.clip(round(cy - side / 2.0), 0, max(0, h - side)))
    cw = int(min(side, w - x))
    ch = int(min(side, h - y))
    return x, y, cw, ch


def sample_background_crop(
    pool: BackgroundImagePool,
    fp_boxes: dict[str, np.ndarray],
    cfg_bg: BackgroundConfig,
    rng: np.random.Generator,
    allow_fp_mine: bool = True,
) -> np.ndarray:
    """Pick a pool image + crop mode and return the crop as an HWC uint8 RGB view."""
    idx = int(rng.integers(0, len(pool)))
    bgr = pool.get_bgr(idx)
    h, w = bgr.shape[:2]
    r = float(rng.random())
    if r < cfg_bg.full_frame_prob:
        x, y, cw, ch = 0, 0, w, h
    elif r < cfg_bg.full_frame_prob + cfg_bg.live_square_prob:
        x, y, cw, ch = live_square_window(h, w)
    else:
        name = pool.name(idx)
        box = fp_boxes.get(name)
        if allow_fp_mine and box is not None and rng.random() < cfg_bg.fp_mine_prob:
            # eval boxes are in the ORIGINAL photo pixels; rescale to the pool's
            # (possibly downscaled) resolution before mining.
            scale = pool_scale(pool, idx)
            x, y, cw, ch = fp_mine_window(h, w, box * scale, cfg_bg, rng)
        else:
            x, y, cw, ch = sample_crop_window(h, w, cfg_bg, rng)
    crop_bgr = bgr[y : y + ch, x : x + cw]
    return cv2.cvtColor(np.ascontiguousarray(crop_bgr), cv2.COLOR_BGR2RGB)


def pool_scale(pool: BackgroundImagePool, idx: int) -> np.ndarray:
    """Exact x/y ratios from original photo pixels to pool image pixels.

    The ratios come from the decoded and original image shapes, never from box
    coordinates.

    Args:
        pool: Background image pool.
        idx: Photo index in ``pool``.

    Returns:
        ``[sx, sy, sx, sy]`` as float32, ready to multiply an ``xyxy`` box.
    """
    bgr = pool.get_bgr(idx)
    decoded_h, decoded_w = bgr.shape[:2]
    original_h, original_w = pool.original_shape(idx)
    scale_x = decoded_w / float(original_w)
    scale_y = decoded_h / float(original_h)
    return np.asarray([scale_x, scale_y, scale_x, scale_y], dtype=np.float32)


# ---------------------------------------------------------------------------
# Compositing the synthetic object onto a background crop
# ---------------------------------------------------------------------------
def apply_background_to_synthetic(
    rgb: np.ndarray,
    bgr: np.ndarray,
    fg_mask: np.ndarray,
    pool: BackgroundImagePool,
    fp_boxes: dict[str, np.ndarray],
    cfg_bg: BackgroundConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace the synthetic image's background with a background crop, in place-safe.

    Keeps the object foreground (``fg_mask > 0``) untouched, so the existing
    bbox / mask / 64-keypoint labels remain exactly valid. Returns updated
    ``(rgb, bgr)``; the foreground is feathered at the mask edge to avoid a hard
    composite seam.
    """
    crop_rgb = sample_background_crop(pool, fp_boxes, cfg_bg, rng, allow_fp_mine=False)
    blur_min = max(0.0, float(cfg_bg.background_blur_sigma_min))
    blur_max = max(blur_min, float(cfg_bg.background_blur_sigma_max))
    if blur_max > 0:
        sigma = float(rng.uniform(blur_min, blur_max))
        if sigma > 0:
            crop_rgb = cv2.GaussianBlur(crop_rgb, (0, 0), sigmaX=sigma, sigmaY=sigma)

    out_rgb, out_bgr = blend_foreground(rgb, bgr, fg_mask, crop_rgb, cfg_bg, rng)
    if cfg_bg.edge_blur_sigma > 0 and cfg_bg.edge_blur_width > 0:
        out_rgb = soften_composite_edge(
            out_rgb,
            fg_mask,
            cfg_bg.edge_blur_sigma,
            cfg_bg.edge_blur_width,
        )
        out_bgr = soften_composite_edge(
            out_bgr,
            fg_mask,
            cfg_bg.edge_blur_sigma,
            cfg_bg.edge_blur_width,
        )
    return out_rgb, out_bgr


def soften_composite_edge(
    composite: np.ndarray,
    fg_mask: np.ndarray,
    sigma: float,
    width: int,
) -> np.ndarray:
    """Blur only a narrow band spanning the composited silhouette."""
    if sigma <= 0 or width <= 0:
        return composite
    kernel_size = 2 * int(width) + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    foreground = (fg_mask > 0).astype(np.uint8)
    outer = cv2.dilate(foreground, kernel)
    inner = cv2.erode(foreground, kernel)
    edge_weight = (outer - inner).astype(np.float32)
    edge_weight = cv2.GaussianBlur(edge_weight, (0, 0), sigmaX=_EDGE_BAND_SIGMA)
    edge_weight = np.clip(edge_weight * _EDGE_BAND_GAIN, 0.0, 1.0)[..., None]
    blurred = cv2.GaussianBlur(composite, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    softened = (
        composite.astype(np.float32) * (1.0 - edge_weight)
        + blurred.astype(np.float32) * edge_weight
    )
    return np.clip(softened, 0, 255).astype(np.uint8)


def blend_foreground(
    rgb: np.ndarray,
    bgr: np.ndarray,
    fg_mask: np.ndarray,
    background_rgb: np.ndarray,
    cfg_bg: BackgroundConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Alpha-blend the ``fg_mask`` foreground over a resized ``background_rgb`` crop.

    Args:
        rgb: Synthetic HWC uint8 RGB image.
        bgr: The same image in BGR order.
        fg_mask: Foreground mask; nonzero pixels are kept.
        background_rgb: Background crop, resized to the image size.
        cfg_bg: Background config (erosion and feather settings).
        rng: Random generator for the feather sigma.

    Returns:
        Updated ``(rgb, bgr)``; labels are unchanged.
    """
    h, w = rgb.shape[:2]
    bg_rgb = cv2.resize(background_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    bg_bgr = cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2BGR)

    alpha = (fg_mask > 0).astype(np.float32)
    erode = max(0, int(cfg_bg.composite_erode_px))
    if erode > 0:
        kernel = np.ones((2 * erode + 1, 2 * erode + 1), np.uint8)
        alpha = cv2.erode(alpha, kernel)
    sigma = float(rng.uniform(cfg_bg.alpha_feather_sigma_min, cfg_bg.alpha_feather_sigma_max))
    if sigma > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=sigma)
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]

    out_rgb = rgb.astype(np.float32) * alpha + bg_rgb.astype(np.float32) * (1.0 - alpha)
    out_bgr = bgr.astype(np.float32) * alpha + bg_bgr.astype(np.float32) * (1.0 - alpha)
    return (
        np.clip(out_rgb, 0, 255).astype(np.uint8),
        np.clip(out_bgr, 0, 255).astype(np.uint8),
    )


# ---------------------------------------------------------------------------
# Negative dataset (annotation-free background crops)
# ---------------------------------------------------------------------------
class BackgroundNegativeDataset(Dataset):
    """Dynamic object-free crops with all-zero detection/mask targets.

    The index, worker, configured seed, epoch, and repeat occurrence determine
    each crop. Thus repeated indices still yield dynamic samples, but equivalent
    loader runs are reproducible. Detection and mask heads are supervised toward
    "nothing here"; the keypoint head is masked (``has_kp = 0``) exactly like
    real bbox-only samples.
    """

    def __init__(self, cfg: DoperConfig, split: str | None = None) -> None:
        """Build the background pool for ``split`` (default: the configured train split)."""
        self.cfg = cfg
        cfg_bg = cfg.background
        split = split or cfg_bg.train_split
        paths = split_file_paths(cfg_bg, split)
        self.pool = BackgroundImagePool(paths, cfg_bg.max_image_side)
        self.fp_boxes = load_fp_boxes(cfg_bg.fp_mine_json)
        self._length = max(1, cfg_bg.crops_per_source * max(1, len(self.pool)))
        self._epoch = 0
        self._rng_counters: dict[tuple[int, int], int] = {}

    def __len__(self) -> int:
        """Return the configured number of crops per epoch."""
        return self._length

    def set_epoch(self, epoch: int) -> None:
        """Set the RNG epoch for direct or non-persistent-worker use."""
        self._epoch = int(epoch)
        self._rng_counters.clear()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Build one background-only sample with all-zero detection and mask targets."""
        from .dataset import normalize_image  # lazy to avoid an import cycle

        mcfg = self.cfg.model
        cfg_bg = self.cfg.background
        occurrence = next_worker_occurrence(self._rng_counters, index)
        rng = worker_index_rng(
            cfg_bg.seed, index, self._epoch, stream=_STREAM_NEGATIVE, occurrence=occurrence
        )
        det_size = mcfg.det_input_size

        crop_rgb = sample_background_crop(self.pool, self.fp_boxes, cfg_bg, rng)
        if mcfg.letterbox:
            det_rgb = targets.letterbox_image(
                crop_rgb, det_size, pad_value=_PAD_VALUE, interp=cv2.INTER_LINEAR
            )
        else:
            det_rgb = cv2.resize(crop_rgb, (det_size, det_size), interpolation=cv2.INTER_LINEAR)
        if self.cfg.aug.enabled:
            det_rgb = photometric_augment(det_rgb, self.cfg.aug, rng)

        feat_h, feat_w = self.cfg.det_feat_size
        heatmap, wh, offset, reg_weight = targets.encode_centernet(
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            num_classes=mcfg.num_classes,
            out_h=feat_h,
            out_w=feat_w,
            stride=mcfg.det_stride,
        )
        mask_target = np.zeros((1, feat_h, feat_w), dtype=np.float32)

        crop_img = cv2.resize(
            crop_rgb, (mcfg.crop_size, mcfg.crop_size), interpolation=cv2.INTER_LINEAR
        )
        if self.cfg.aug.enabled:
            crop_img = photometric_augment(crop_img, self.cfg.aug, rng)

        kp_weight = np.zeros((mcfg.num_keypoints,), dtype=np.float32)
        sample = {
            "det_image": torch.from_numpy(normalize_image(det_rgb)),
            "heatmap": torch.from_numpy(heatmap),
            "wh": torch.from_numpy(wh),
            "offset": torch.from_numpy(offset),
            "reg_weight": torch.from_numpy(reg_weight),
            "mask_target": torch.from_numpy(mask_target),
            "det_weight": torch.tensor(1.0, dtype=torch.float32),
            "mask_weight": torch.tensor(1.0, dtype=torch.float32),
            "crop_image": torch.from_numpy(normalize_image(crop_img)),
            "kp_weight": torch.from_numpy(kp_weight),
            "has_kp": torch.tensor(0.0, dtype=torch.float32),
            "img_id": torch.tensor(-1, dtype=torch.long),
            # Match DoperDataset's sample schema. Backgrounds do not receive
            # a geometric crop transform or orientation supervision.
            "crop_rotation_deg": torch.tensor(0.0, dtype=torch.float32),
        }
        if mcfg.keypoint_head == "simcc":
            bins = int(round(mcfg.crop_size * mcfg.simcc_split_ratio))
            sample["simcc_x"] = torch.zeros((mcfg.num_keypoints, bins), dtype=torch.float32)
            sample["simcc_y"] = torch.zeros((mcfg.num_keypoints, bins), dtype=torch.float32)
        else:
            hs = mcfg.crop_size // 4
            sample["keypoint_heatmap"] = torch.zeros(
                (mcfg.num_keypoints, hs, hs), dtype=torch.float32
            )
        if mcfg.curve_head_enabled:
            hs = mcfg.crop_size // 4
            sample["curve_heatmap"] = torch.zeros((mcfg.num_curves, hs, hs), dtype=torch.float32)
            sample["curve_weight"] = torch.zeros((mcfg.num_curves,), dtype=torch.float32)
            sample["has_curves"] = torch.tensor(0.0, dtype=torch.float32)
        if mcfg.rotation_aux_enabled:
            sample["rotation_quaternion"] = torch.tensor([1.0, 0.0, 0.0, 0.0])
            sample["rotation_weight"] = torch.tensor(0.0)
        return sample
