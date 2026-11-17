# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Target encoders/decoders for the DOPER unified model.

``gaussian_radius`` and ``_draw_umich_gaussian`` follow the CornerNet/CenterNet
reference implementations (CenterNet is MIT licensed, CornerNet is BSD-3-Clause licensed);
see THIRD_PARTY_NOTICES.md.

This module is framework-light: it only depends on numpy and cv2 so that the
dataset can build training targets in worker processes without importing torch.

Two task families are encoded here:

* CenterNet-style detection targets (center heatmap + width/height + sub-pixel
  offset) for the full-image detection head.
* SimCC-style keypoint targets (1D coordinate classification along x and y) for
  the top-down keypoint head, together with the affine crop that maps the
  original image into a square top-down keypoint crop.

Keeping encode/decode in one place means the training targets and the inference
decode use exactly the same conventions.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

# Letterbox padding value (YOLO-style gray).
LETTERBOX_PAD_VALUE = 114.0
# Floor on a normalization mass / softmax denominator before dividing.
PROBABILITY_MASS_EPSILON = 1e-9
# Gaussian mass below which a SimCC keypoint target is dropped.
SIMCC_MASS_EPSILON = 1e-6
# Largest sub-bin offset a DARK refinement may apply.
DARK_MAX_OFFSET = 0.75
# A peak counts as concave only if its curvature is below minus this value.
DARK_CURVATURE_EPSILON = 1e-6
DARK_DETERMINANT_EPSILON = 1e-8
# Default half-width (in bins) of the keypoint score windows.
HEATMAP_SCORE_WINDOW = 2
SIMCC_SCORE_WINDOW = 10
DEFAULT_PEAK_COUNT = 3
DEFAULT_HEATMAP_PEAK_DISTANCE_BINS = 4
# CenterNet gaussian overlap and decode defaults.
GAUSSIAN_MIN_OVERLAP = 0.7
CENTERNET_DECODE_TOPK = 10
CENTERNET_SCORE_THRESHOLD = 0.2


# ---------------------------------------------------------------------------
# Top-down crop affine (high-resolution keypoint path)
# ---------------------------------------------------------------------------
def topdown_warp_matrix(
    center: tuple[float, float],
    src_size: float,
    out_size: int,
    rotation_deg: float = 0.0,
) -> np.ndarray:
    """Return a 2x3 affine mapping the original image into a square crop.

    The crop is a square of side ``src_size`` (in original-image pixels)
    centered on ``center``, optionally rotated in-plane by ``rotation_deg``, and
    resized to ``out_size`` x ``out_size``. The mapping is a similarity (uniform
    scale + rotation + translation), so it preserves the object aspect ratio -
    this is what protects keypoint precision for small objects relative to a
    low-resolution ROIAlign crop. ``rotation_deg`` defaults to 0 so inference
    callers get the plain axis-aligned crop.
    """
    cx, cy = center
    half = src_size / 2.0
    theta = np.deg2rad(rotation_deg)
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
    # Rotated source basis vectors (image coords are y-down).
    dir_x = (half * cos_t, half * sin_t)
    dir_y = (-half * sin_t, half * cos_t)
    src = np.array(
        [[cx, cy], [cx + dir_x[0], cy + dir_x[1]], [cx + dir_y[0], cy + dir_y[1]]],
        dtype=np.float32,
    )
    dst = np.array(
        [
            [out_size / 2.0, out_size / 2.0],
            [out_size, out_size / 2.0],
            [out_size / 2.0, out_size],
        ],
        dtype=np.float32,
    )
    return cv2.getAffineTransform(src, dst)


def crop_camera_roll_matrix(rotation_deg: float) -> np.ndarray:
    """Return the camera-frame roll induced by a top-down crop rotation.

    :func:`topdown_warp_matrix` uses image coordinates (``y`` points down), so
    a positive ``rotation_deg`` makes the visible image content rotate
    counter-clockwise.  For an object-to-camera pose target, the equivalent
    change of camera coordinates is therefore ``Rz(-rotation_deg)``.  If the
    original pose is ``X_cam = R @ X_obj + t``, the pose matching the rotated
    crop is ``R' = Q @ R`` and ``t' = Q @ t``, where ``Q`` is this matrix.

    Keeping this convention next to the crop affine makes it harder for an
    auxiliary pose target to silently use the opposite sign.
    """
    theta = np.deg2rad(float(rotation_deg))
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
    return np.asarray(
        [
            [cos_t, sin_t, 0.0],
            [-sin_t, cos_t, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rotate_object_to_camera_quaternion_for_crop(
    quaternion_wxyz: np.ndarray,
    rotation_deg: float,
) -> np.ndarray:
    """Update a scalar-first object-to-camera quaternion for a crop roll.

    The returned quaternion represents ``Q @ R`` using the same ``Q`` as
    :func:`crop_camera_roll_matrix`.  This is the label required when a
    camera-frame rotation classifier is trained from an augmented crop.
    """
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError(
            f"Expected scalar-first quaternion with shape (4,), got {quaternion.shape}"
        )
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Quaternion must have a finite, non-zero norm")
    w2, x2, y2, z2 = quaternion / norm
    half_theta = np.deg2rad(float(rotation_deg)) / 2.0
    # Rz(-rotation_deg), left-multiplied onto the object-to-camera pose.
    w1, x1, y1, z1 = float(np.cos(half_theta)), 0.0, 0.0, -float(np.sin(half_theta))
    result = np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )
    return result / np.linalg.norm(result)


def bbox_to_center_size(
    bbox_xywh: tuple[float, float, float, float],
    pad_factor: float,
) -> tuple[tuple[float, float], float]:
    """Convert a COCO ``[x, y, w, h]`` box into a padded square center/size."""
    x, y, w, h = bbox_xywh
    cx = x + w / 2.0
    cy = y + h / 2.0
    size = max(w, h) * pad_factor
    return (cx, cy), float(size)


def warp_points(points_xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine to an ``(N, 2)`` array of points."""
    pts = np.asarray(points_xy, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    homog = np.concatenate([pts, ones], axis=1)
    return (homog @ matrix.T).astype(np.float32)


def rot90_boxes(boxes_xywh: np.ndarray, k: int, size: float) -> np.ndarray:
    """Rotate axis-aligned ``[x, y, w, h]`` boxes by ``k*90`` deg in a square.

    Matches ``numpy.rot90`` (counter-clockwise) on a ``size`` x ``size`` image,
    so the same ``k`` can be applied to the image, its boxes, and its mask and
    stay aligned. A 90-degree rotation maps axis-aligned boxes to axis-aligned
    boxes (width/height swap), so this is exact - no enclosing-box slack.
    """
    k = int(k) % 4
    boxes = np.asarray(boxes_xywh, dtype=np.float32).reshape(-1, 4)
    if k == 0 or boxes.shape[0] == 0:
        return boxes
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x2, y2 = x + w, y + h

    def _tf(px: np.ndarray, py: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if k == 1:
            return py, size - px
        if k == 2:
            return size - px, size - py
        return size - py, px  # k == 3

    ax, ay = _tf(x, y)
    bx, by = _tf(x2, y2)
    nx = np.minimum(ax, bx)
    ny = np.minimum(ay, by)
    nw = np.abs(bx - ax)
    nh = np.abs(by - ay)
    return np.stack([nx, ny, nw, nh], axis=1).astype(np.float32)


def scale_translate_matrix(size: int, scale: float, tx: float, ty: float) -> np.ndarray:
    """2x3 affine: uniform zoom about the image center + pixel translation.

    No rotation/shear/reflection, so axis-aligned boxes stay axis-aligned and
    object chirality is preserved (the point of using this over a flip).
    """
    c = size / 2.0
    b = c - scale * c
    return np.array([[scale, 0.0, b + tx], [0.0, scale, b + ty]], dtype=np.float32)


def affine_boxes_xywh(
    boxes_xywh: np.ndarray, matrix: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a 2x3 affine to ``[x, y, w, h]`` boxes and clip them to a square.

    Boxes that fall (almost) entirely outside the ``size`` square are dropped.

    Returns ``(boxes, keep_mask)`` where ``keep_mask`` is a boolean array over
    the input boxes so caller-side labels can be filtered in lockstep.
    """
    boxes = np.asarray(boxes_xywh, dtype=np.float32).reshape(-1, 4)
    if boxes.shape[0] == 0:
        return boxes, np.zeros((0,), dtype=bool)
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    xs = np.stack([x, x + w, x, x + w], axis=1)
    ys = np.stack([y, y, y + h, y + h], axis=1)
    px = matrix[0, 0] * xs + matrix[0, 1] * ys + matrix[0, 2]
    py = matrix[1, 0] * xs + matrix[1, 1] * ys + matrix[1, 2]
    nx1 = np.clip(px.min(axis=1), 0, size)
    ny1 = np.clip(py.min(axis=1), 0, size)
    nx2 = np.clip(px.max(axis=1), 0, size)
    ny2 = np.clip(py.max(axis=1), 0, size)
    nw = nx2 - nx1
    nh = ny2 - ny1
    keep = (nw > 1.0) & (nh > 1.0)
    out = np.stack([nx1, ny1, nw, nh], axis=1).astype(np.float32)
    return out[keep], keep


# ---------------------------------------------------------------------------
# Letterbox (aspect-preserving) resize for the full-image detection path
# ---------------------------------------------------------------------------
def heatmap_nms(hm: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """Keep only local maxima of a detection heatmap.

    Args:
        hm: ``(B, C, H, W)`` heatmap probabilities.
        kernel: Max-pooling window size.

    Returns:
        ``hm`` with non-maximum pixels zeroed.
    """
    pad = (kernel - 1) // 2
    hmax = torch.nn.functional.max_pool2d(hm, kernel, stride=1, padding=pad)
    keep = (hmax == hm).float()
    return hm * keep


def letterbox_params(orig_h: int, orig_w: int, dst_size: int) -> tuple[float, int, int, int, int]:
    """Uniform-scale letterbox mapping of an image into a ``dst_size`` square.

    Returns ``(scale, pad_x, pad_y, new_w, new_h)`` where the original image is
    scaled by ``scale`` (the same factor on both axes, so aspect ratio is
    preserved) to ``new_w`` x ``new_h`` and centered in the square with
    ``pad_x``/``pad_y`` border on the left/top. A point ``(x, y)`` in original
    pixels maps to ``(x * scale + pad_x, y * scale + pad_y)`` in letterbox
    pixels; invert with ``(X - pad_x) / scale``.
    """
    scale = dst_size / float(max(orig_h, orig_w))
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    pad_x = (dst_size - new_w) // 2
    pad_y = (dst_size - new_h) // 2
    return scale, pad_x, pad_y, new_w, new_h


def letterbox_image(
    img: np.ndarray,
    dst_size: int,
    pad_value: float = LETTERBOX_PAD_VALUE,
    interp: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Letterbox an HWC or HW image into a ``dst_size`` square (aspect-preserving)."""
    h, w = img.shape[:2]
    scale, pad_x, pad_y, new_w, new_h = letterbox_params(h, w, dst_size)
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    if img.ndim == 3:
        out = np.full((dst_size, dst_size, img.shape[2]), pad_value, dtype=img.dtype)
    else:
        out = np.full((dst_size, dst_size), pad_value, dtype=img.dtype)
    out[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return out


# ---------------------------------------------------------------------------
# 2D keypoint heatmap targets
# ---------------------------------------------------------------------------
def encode_keypoint_heatmaps(
    keypoints_crop: np.ndarray,
    visibility: np.ndarray,
    crop_size: int,
    heatmap_size: int,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode crop keypoints as normalized 2D Gaussian distributions.

    Crop and heatmap coordinates are aligned at both endpoint pixel centers:
    crop coordinate ``0`` maps to heatmap coordinate ``0`` and
    ``crop_size - 1`` maps to ``heatmap_size - 1``. This keeps labeled points
    on the far row/column of the crop valid and is inverted exactly by
    :func:`decode_keypoint_heatmaps`.
    """
    keypoints_crop = np.asarray(keypoints_crop)
    visibility = np.asarray(visibility)
    if keypoints_crop.ndim != 2 or keypoints_crop.shape[1] != 2:
        raise ValueError(f"Expected keypoints_crop with shape (K,2), got {keypoints_crop.shape}")
    if visibility.ndim != 1 or visibility.shape[0] != keypoints_crop.shape[0]:
        raise ValueError("visibility must have shape (K,) matching keypoints_crop")
    if not isinstance(crop_size, (int, np.integer)) or not isinstance(
        heatmap_size, (int, np.integer)
    ):
        raise ValueError("crop_size and heatmap_size must be integer dimensions")
    if crop_size <= 1 or heatmap_size <= 1:
        raise ValueError("crop_size and heatmap_size must both be greater than 1")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be a finite positive value")

    num_keypoints = keypoints_crop.shape[0]
    target = np.zeros(
        (num_keypoints, heatmap_size, heatmap_size),
        dtype=np.float32,
    )
    weight = np.zeros((num_keypoints,), dtype=np.float32)
    scale = (heatmap_size - 1) / float(crop_size - 1)
    grid = np.arange(heatmap_size, dtype=np.float32)

    for keypoint_idx in range(num_keypoints):
        if visibility[keypoint_idx] <= 0:
            continue
        mu_x = float(keypoints_crop[keypoint_idx, 0]) * scale
        mu_y = float(keypoints_crop[keypoint_idx, 1]) * scale
        if mu_x < 0 or mu_x > heatmap_size - 1 or mu_y < 0 or mu_y > heatmap_size - 1:
            continue
        gaussian_x = np.exp(-((grid - mu_x) ** 2) / (2.0 * sigma**2))
        gaussian_y = np.exp(-((grid - mu_y) ** 2) / (2.0 * sigma**2))
        gaussian = gaussian_y[:, None] * gaussian_x[None, :]
        mass = float(gaussian.sum())
        if mass <= PROBABILITY_MASS_EPSILON:
            continue
        target[keypoint_idx] = gaussian / mass
        weight[keypoint_idx] = 1.0
    return target, weight


def decode_keypoint_heatmaps(
    logits: np.ndarray,
    crop_size: int,
    beta: float = 1.0,
    window: int = HEATMAP_SCORE_WINDOW,
    dark: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode ``(K,H,W)`` joint heatmap logits to crop-space coordinates."""
    logits = np.asarray(logits)
    if logits.ndim != 3:
        raise ValueError(f"Expected (K,H,W) heatmaps, got {logits.shape}")
    num_keypoints, height, width = logits.shape
    if height != width:
        raise ValueError("Keypoint heatmaps must be square")
    if not isinstance(crop_size, (int, np.integer)):
        raise ValueError("crop_size must be an integer dimension")
    if crop_size <= 1 or width <= 1:
        raise ValueError("crop_size and heatmap dimensions must be greater than 1")
    if num_keypoints == 0:
        raise ValueError("Keypoint heatmaps must contain at least one keypoint")
    flat = logits.reshape(num_keypoints, -1)
    peak_flat = np.argmax(flat, axis=1)
    peak_y, peak_x = np.divmod(peak_flat, width)
    pos_x = peak_x.astype(np.float32)
    pos_y = peak_y.astype(np.float32)
    if dark:
        offsets = _dark_offsets_2d(logits, peak_x, peak_y)
        pos_x += offsets[:, 0]
        pos_y += offsets[:, 1]

    scale = (crop_size - 1) / float(width - 1)
    coords = np.stack([pos_x, pos_y], axis=1).astype(np.float32) * scale

    scaled = flat * beta
    scaled -= scaled.max(axis=1, keepdims=True)
    probability = np.exp(scaled)
    probability /= np.clip(probability.sum(axis=1, keepdims=True), PROBABILITY_MASS_EPSILON, None)
    probability = probability.reshape(num_keypoints, height, width)
    scores = np.zeros((num_keypoints,), dtype=np.float32)
    for keypoint_idx in range(num_keypoints):
        x0 = max(0, int(peak_x[keypoint_idx]) - window)
        x1 = min(width, int(peak_x[keypoint_idx]) + window + 1)
        y0 = max(0, int(peak_y[keypoint_idx]) - window)
        y1 = min(height, int(peak_y[keypoint_idx]) + window + 1)
        scores[keypoint_idx] = probability[keypoint_idx, y0:y1, x0:x1].sum()
    return coords, scores


def keypoint_heatmap_peak_indices(
    logits: np.ndarray,
    topk: int = DEFAULT_PEAK_COUNT,
    min_distance_bins: int = DEFAULT_HEATMAP_PEAK_DISTANCE_BINS,
) -> np.ndarray:
    """Return top-k spatially distinct 2D peaks as ``(x, y)`` indices."""
    if logits.ndim != 3:
        raise ValueError(f"Expected (K,H,W) heatmaps, got {logits.shape}")
    num_keypoints, height, width = logits.shape
    topk = max(1, min(int(topk), height * width))
    radius = max(0, int(min_distance_bins))
    peaks = np.zeros((num_keypoints, topk, 2), dtype=np.int64)
    for keypoint_idx, heatmap in enumerate(logits):
        work = heatmap.astype(np.float64, copy=True)
        for rank in range(topk):
            flat_idx = int(np.argmax(work))
            y, x = np.unravel_index(flat_idx, work.shape)
            peaks[keypoint_idx, rank] = (x, y)
            x0, x1 = max(0, x - radius), min(width, x + radius + 1)
            y0, y1 = max(0, y - radius), min(height, y + radius + 1)
            work[y0:y1, x0:x1] = -np.inf
    return peaks


def refine_keypoint_heatmap_peak(
    logits: np.ndarray,
    x: int,
    y: int,
    dark: bool = True,
) -> np.ndarray:
    """Return one 2D peak in heatmap-bin coordinates, optionally DARK-refined."""
    position = np.asarray([float(x), float(y)], dtype=np.float32)
    if dark:
        position += _dark_offsets_2d(
            logits[None],
            np.asarray([x], dtype=np.int64),
            np.asarray([y], dtype=np.int64),
        )[0]
    return position


def _dark_offsets_2d(
    logits: np.ndarray,
    peak_x: np.ndarray,
    peak_y: np.ndarray,
) -> np.ndarray:
    """Second-order DARK refinement using the local 2D Hessian."""
    num_keypoints, height, width = logits.shape
    offsets = np.zeros((num_keypoints, 2), dtype=np.float32)
    for keypoint_idx in range(num_keypoints):
        x = int(peak_x[keypoint_idx])
        y = int(peak_y[keypoint_idx])
        if x <= 0 or x >= width - 1 or y <= 0 or y >= height - 1:
            continue
        surface = logits[keypoint_idx]
        dx = 0.5 * (surface[y, x + 1] - surface[y, x - 1])
        dy = 0.5 * (surface[y + 1, x] - surface[y - 1, x])
        dxx = surface[y, x + 1] - 2.0 * surface[y, x] + surface[y, x - 1]
        dyy = surface[y + 1, x] - 2.0 * surface[y, x] + surface[y - 1, x]
        dxy = 0.25 * (
            surface[y + 1, x + 1]
            - surface[y - 1, x + 1]
            - surface[y + 1, x - 1]
            + surface[y - 1, x - 1]
        )
        determinant = dxx * dyy - dxy * dxy
        if (
            dxx >= -DARK_CURVATURE_EPSILON
            or dyy >= -DARK_CURVATURE_EPSILON
            or determinant <= DARK_DETERMINANT_EPSILON
        ):
            continue
        offset_x = -(dyy * dx - dxy * dy) / determinant
        offset_y = -(-dxy * dx + dxx * dy) / determinant
        offsets[keypoint_idx, 0] = np.clip(offset_x, -DARK_MAX_OFFSET, DARK_MAX_OFFSET)
        offsets[keypoint_idx, 1] = np.clip(offset_y, -DARK_MAX_OFFSET, DARK_MAX_OFFSET)
    return offsets


# ---------------------------------------------------------------------------
# SimCC keypoint targets
# ---------------------------------------------------------------------------
def encode_simcc(
    keypoints_crop: np.ndarray,
    visibility: np.ndarray,
    out_size: int,
    split_ratio: float,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode crop-space keypoints into 1D SimCC gaussian targets.

    Args:
        keypoints_crop: ``(K, 2)`` keypoint coordinates in crop pixels.
        visibility: ``(K,)`` visibility flag (>0 means labeled).
        out_size: crop side length in pixels (square crop).
        split_ratio: number of SimCC bins per pixel.
        sigma: gaussian sigma in *bin* units.

    Returns:
        ``(target_x, target_y, weight)`` where ``target_x`` is ``(K, Wx)``,
        ``target_y`` is ``(K, Wy)`` (each row sums to 1 for labeled points),
        and ``weight`` is ``(K,)``.
    """
    num_kp = keypoints_crop.shape[0]
    bins = int(round(out_size * split_ratio))
    target_x = np.zeros((num_kp, bins), dtype=np.float32)
    target_y = np.zeros((num_kp, bins), dtype=np.float32)
    weight = np.zeros((num_kp,), dtype=np.float32)

    grid = np.arange(bins, dtype=np.float32)
    for k in range(num_kp):
        if visibility[k] <= 0:
            continue
        mu_x = float(keypoints_crop[k, 0]) * split_ratio
        mu_y = float(keypoints_crop[k, 1]) * split_ratio
        # Drop points whose center falls outside the crop entirely.
        if mu_x < 0 or mu_x > bins - 1 or mu_y < 0 or mu_y > bins - 1:
            continue
        gx = np.exp(-((grid - mu_x) ** 2) / (2.0 * sigma**2))
        gy = np.exp(-((grid - mu_y) ** 2) / (2.0 * sigma**2))
        sx = gx.sum()
        sy = gy.sum()
        if sx <= SIMCC_MASS_EPSILON or sy <= SIMCC_MASS_EPSILON:
            continue
        target_x[k] = gx / sx
        target_y[k] = gy / sy
        weight[k] = 1.0
    return target_x, target_y, weight


def decode_simcc(
    simcc_x: np.ndarray,
    simcc_y: np.ndarray,
    split_ratio: float,
    window: int = SIMCC_SCORE_WINDOW,
    beta: float = 1.0,
    dark: bool = True,
    method: str = "argmax",
    coord_beta: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode SimCC logits into crop-space coordinates + a calibrated score.

    Args:
        simcc_x: ``(K, Wx)`` array of logits.
        simcc_y: ``(K, Wy)`` array of logits.
        split_ratio: SimCC bins per pixel.
        window: half-width (in bins) of the peak window used for confidence.
        beta: temperature applied to the logits before the softmax used for
            scoring. Must match the training ``LossConfig.simcc_beta`` (the head
            is trained so ``softmax(beta * logits)`` is the sharp distribution;
            scoring the raw logits would read as flat/low-confidence and wrongly
            filter every keypoint).
        dark: if True (default), refine each coordinate to sub-bin precision with
            a DARK-style quadratic Taylor fit at the peak instead of taking the
            bare argmax bin (see ``_dark_offset_1d``). Removes the
            ``1 / split_ratio`` px quantization and the frame-to-frame argmax
            jitter. Purely a decode-side change: it does not touch the weights,
            so it works on any existing checkpoint. Set False for the legacy
            argmax behavior.
        method: coordinate decoder: ``"argmax"`` (the existing behavior),
            ``"softargmax"`` (expectation over every bin), or
            ``"local_softargmax"`` (expectation inside ``+/-window`` of the
            argmax). ``dark`` only applies to ``"argmax"``.
        coord_beta: softmax temperature for the two soft-argmax methods. It is
            deliberately separate from ``beta`` so coordinate decoding can be
            swept without changing confidence thresholding.

    Returns:
        ``(coords, scores)`` with ``coords`` shape ``(K, 2)`` in crop pixels and
        ``scores`` shape ``(K,)``. The score is the softmax probability mass
        within ``+/-window`` bins of the peak, which is ~1.0 for a confident
        (sharp) prediction and small for a flat/uncertain one - unlike the bare
        softmax peak, this is stable regardless of the number of bins.
    """
    x_idx = np.argmax(simcc_x, axis=1)
    y_idx = np.argmax(simcc_y, axis=1)
    if method == "argmax":
        if dark:
            x_pos = x_idx + _dark_offset_1d(simcc_x, x_idx)
            y_pos = y_idx + _dark_offset_1d(simcc_y, y_idx)
        else:
            x_pos = x_idx.astype(np.float32)
            y_pos = y_idx.astype(np.float32)
    elif method == "softargmax":
        sx_coord = _softmax(simcc_x * coord_beta)
        sy_coord = _softmax(simcc_y * coord_beta)
        x_grid = np.arange(simcc_x.shape[1], dtype=np.float32)
        y_grid = np.arange(simcc_y.shape[1], dtype=np.float32)
        x_pos = (sx_coord * x_grid[None]).sum(axis=1)
        y_pos = (sy_coord * y_grid[None]).sum(axis=1)
    elif method == "local_softargmax":
        x_pos = _local_softargmax_1d(simcc_x, x_idx, window, coord_beta)
        y_pos = _local_softargmax_1d(simcc_y, y_idx, window, coord_beta)
    else:
        raise ValueError(
            f"Unknown SimCC decode method {method!r}; expected argmax, "
            "softargmax, or local_softargmax"
        )
    coords = np.stack([x_pos, y_pos], axis=1).astype(np.float32) / split_ratio
    sx = _softmax(simcc_x * beta)
    sy = _softmax(simcc_y * beta)
    score_x = _windowed_mass(sx, x_idx, window)
    score_y = _windowed_mass(sy, y_idx, window)
    scores = np.minimum(score_x, score_y).astype(np.float32)
    return coords.astype(np.float32), scores


def _local_softargmax_1d(
    logits: np.ndarray,
    peak_idx: np.ndarray,
    window: int,
    beta: float,
) -> np.ndarray:
    out = np.zeros((logits.shape[0],), dtype=np.float32)
    for row_idx, peak in enumerate(peak_idx):
        lo = max(0, int(peak) - window)
        hi = min(logits.shape[1], int(peak) + window + 1)
        local = logits[row_idx, lo:hi] * beta
        local = local - local.max()
        prob = np.exp(local)
        prob /= np.clip(prob.sum(), PROBABILITY_MASS_EPSILON, None)
        bins = np.arange(lo, hi, dtype=np.float32)
        out[row_idx] = float((prob * bins).sum())
    return out


def _dark_offset_1d(logits: np.ndarray, peak_idx: np.ndarray) -> np.ndarray:
    """DARK-style sub-bin refinement for a 1D SimCC distribution.

    Fits a local quadratic to the (log-likelihood == logit) surface at the peak
    via a second-order Taylor expansion and returns the sub-bin offset
    ``-f'(mu) / f''(mu)`` per keypoint. SimCC is trained with Gaussian targets
    (see ``encode_simcc``), so the logits are locally parabolic and this recovers
    sub-pixel position without a high-resolution map - the 1D reduction of DARK
    (Zhang et al., CVPR 2020), where the 2D Hessian collapses to a scalar per
    axis and no matrix inversion is needed.

    The offset is applied only at interior, concave (i.e. genuine) peaks and is
    clamped to a single bin; anywhere else it is 0 (fall back to the argmax bin),
    so a flat or edge-pinned distribution can never be pushed to a wild value.
    """
    n, w = logits.shape
    if w < 3:
        return np.zeros(n, dtype=np.float32)
    ar = np.arange(n)
    kc = np.clip(peak_idx, 1, w - 2)
    lm = logits[ar, kc - 1]
    l0 = logits[ar, kc]
    lp = logits[ar, kc + 1]
    d1 = 0.5 * (lp - lm)
    d2 = lp - 2.0 * l0 + lm
    concave = d2 < -DARK_CURVATURE_EPSILON
    offset = np.where(concave, -d1 / np.where(concave, d2, -1.0), 0.0)
    offset = np.clip(offset, -DARK_MAX_OFFSET, DARK_MAX_OFFSET)
    interior = (peak_idx > 0) & (peak_idx < w - 1)
    return np.where(interior, offset, 0.0).astype(np.float32)


def _windowed_mass(prob: np.ndarray, peak_idx: np.ndarray, window: int) -> np.ndarray:
    bins = prob.shape[1]
    out = np.zeros(prob.shape[0], dtype=np.float32)
    for k in range(prob.shape[0]):
        lo = max(0, int(peak_idx[k]) - window)
        hi = min(bins, int(peak_idx[k]) + window + 1)
        out[k] = prob[k, lo:hi].sum()
    return out


def _softmax(arr: np.ndarray) -> np.ndarray:
    arr = arr - arr.max(axis=1, keepdims=True)
    e = np.exp(arr)
    return e / np.clip(e.sum(axis=1, keepdims=True), PROBABILITY_MASS_EPSILON, None)


# ---------------------------------------------------------------------------
# CenterNet detection targets
# ---------------------------------------------------------------------------
def gaussian_radius(
    det_size: tuple[float, float], min_overlap: float = GAUSSIAN_MIN_OVERLAP
) -> float:
    """Standard CornerNet/CenterNet gaussian radius from a box size."""
    height, width = det_size

    a1 = 1.0
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(max(b1**2 - 4 * a1 * c1, 0.0))
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4.0
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(max(b2**2 - 4 * a2 * c2, 0.0))
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(max(b3**2 - 4 * a3 * c3, 0.0))
    r3 = (b3 + sq3) / (2 * a3)
    return max(0.0, min(r1, r2, r3))


def _draw_umich_gaussian(heatmap: np.ndarray, center: tuple[int, int], radius: int) -> None:
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    m, n = radius, radius
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    g = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    g[g < np.finfo(g.dtype).eps * g.max()] = 0

    cx, cy = center
    h, w = heatmap.shape
    left, right = min(cx, radius), min(w - cx, radius + 1)
    top, bottom = min(cy, radius), min(h - cy, radius + 1)
    if right <= -left or bottom <= -top:
        return
    masked_hm = heatmap[cy - top : cy + bottom, cx - left : cx + right]
    masked_g = g[radius - top : radius + bottom, radius - left : radius + right]
    if masked_g.size > 0:
        np.maximum(masked_hm, masked_g, out=masked_hm)


def encode_centernet(
    boxes_xywh: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    out_h: int,
    out_w: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Encode boxes into CenterNet heatmap / wh / offset / reg-weight maps.

    Args:
        boxes_xywh: ``(N, 4)`` boxes in *original image* pixels (x, y, w, h).
        labels: ``(N,)`` zero-based class ids.
        num_classes: number of detection classes.
        out_h: output (stride-reduced) feature map height.
        out_w: output (stride-reduced) feature map width.
        stride: feature stride relative to the detection input image.

    Returns:
        ``(heatmap, wh, offset, reg_weight)`` with shapes ``(C, H, W)``,
        ``(2, H, W)``, ``(2, H, W)``, ``(1, H, W)``. ``wh`` is in feature-map
        pixels.
    """
    heatmap = np.zeros((num_classes, out_h, out_w), dtype=np.float32)
    wh = np.zeros((2, out_h, out_w), dtype=np.float32)
    offset = np.zeros((2, out_h, out_w), dtype=np.float32)
    reg_weight = np.zeros((1, out_h, out_w), dtype=np.float32)

    for box, label in zip(boxes_xywh, labels, strict=False):
        x, y, w, h = box
        fw = w / stride
        fh = h / stride
        if fw <= 0 or fh <= 0:
            continue
        cx = (x + w / 2.0) / stride
        cy = (y + h / 2.0) / stride
        cx_int = int(np.clip(cx, 0, out_w - 1))
        cy_int = int(np.clip(cy, 0, out_h - 1))
        radius = int(max(0, gaussian_radius((fh, fw))))
        _draw_umich_gaussian(heatmap[int(label)], (cx_int, cy_int), radius)
        wh[0, cy_int, cx_int] = fw
        wh[1, cy_int, cx_int] = fh
        offset[0, cy_int, cx_int] = cx - cx_int
        offset[1, cy_int, cx_int] = cy - cy_int
        reg_weight[0, cy_int, cx_int] = 1.0
    return heatmap, wh, offset, reg_weight


def decode_centernet(
    heatmap: np.ndarray,
    wh: np.ndarray,
    offset: np.ndarray,
    stride: int,
    topk: int = CENTERNET_DECODE_TOPK,
    score_thr: float = CENTERNET_SCORE_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode CenterNet maps into boxes in detection-input pixels.

    Expects ``heatmap`` already passed through sigmoid and a local-max NMS.

    Returns ``(boxes_xyxy, scores, labels)``.
    """
    num_classes, h, w = heatmap.shape
    flat = heatmap.reshape(-1)
    k = min(topk, flat.shape[0])
    idx = np.argpartition(flat, -k)[-k:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    scores = flat[idx]
    keep = scores >= score_thr
    idx = idx[keep]
    scores = scores[keep]

    labels = (idx // (h * w)).astype(np.int64)
    rem = idx % (h * w)
    ys = (rem // w).astype(np.float32)
    xs = (rem % w).astype(np.float32)

    boxes = []
    for xi, yi in zip(xs, ys, strict=False):
        xc = int(xi)
        yc = int(yi)
        ox = offset[0, yc, xc]
        oy = offset[1, yc, xc]
        bw = wh[0, yc, xc]
        bh = wh[1, yc, xc]
        cx = (xi + ox) * stride
        cy = (yi + oy) * stride
        bw_px = bw * stride
        bh_px = bh * stride
        boxes.append([cx - bw_px / 2, cy - bh_px / 2, cx + bw_px / 2, cy + bh_px / 2])
    boxes_arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return boxes_arr, scores.astype(np.float32), labels
