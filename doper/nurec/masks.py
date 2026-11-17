# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAM2 masks of NuRec novel views: prompts from the CAD geometry, the segmenter and its batch.

Each rendered novel view is segmented by SAM2 from prompts its geometry label
gives (``doper.nurec.labels.ViewLabel``):

* a box: the extent of the **rasterized CAD silhouette's nonzero pixels**, widened by
  ``BOX_MARGIN_PX`` on every side (not the float projected bbox). The box goes to SAM2 as is,
  so it may reach past the image border; only the region searched for the negative point below
  is clipped to the image;
* up to ``MAX_POSITIVE_POINTS`` positive points, evenly spaced over the keypoints that are
  ray-visible, inside the image, not of the excluded color and darker than
  ``PROMPT_MAX_VALUE`` (OpenCV HSV value); when none qualifies, the mean of all keypoints;
* a negative point at the median of the excluded-color pixels inside the box when there are
  more than ``MIN_NEGATIVE_PIXELS`` of them (for example a fixture holding the object; recipe
  ``labels.exclude_hsv``).

``Sam2Segmenter`` runs SAM2's image predictor (config ``SAM2_CONFIG``, user-supplied weights,
CUDA bf16 autocast, one mask per view). SAM2 and torch are the optional ``nurec-sam`` extra and
are imported only when a segmenter is built. The ``cad_only`` mask mode builds no segmenter.

Images are OpenCV BGR arrays; pixel coordinates are ``(x, y)``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from ..recipes.nurec_schema import HsvExclusion
from ._files import read_image
from .labels import ViewLabel

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
SAM2_EXTRA = "nurec-sam"
BOX_MARGIN_PX = 10
MAX_POSITIVE_POINTS = 5
# Positive points must be darker than this HSV value: bright specular keypoints mislead SAM2.
PROMPT_MAX_VALUE = 160
MIN_NEGATIVE_PIXELS = 10
SAM_MASK_PATTERN = "{index:06d}.png"
SAM_SCORES_NAME = "sam_scores.json"
MASK_VALUE = 255
_HSV_BOUNDS = (("h_above", 0, np.greater), ("h_below", 0, np.less),
               ("s_above", 1, np.greater), ("s_below", 1, np.less),
               ("v_above", 2, np.greater), ("v_below", 2, np.less))  # fmt: skip


def excluded_pixels(
    image: NDArray[np.uint8], exclude_hsv: HsvExclusion | None
) -> NDArray[np.bool_]:
    """Pixels of the excluded color: every bound of ``exclude_hsv`` that is set holds (strict).

    Args:
        image: ``(h, w, 3)`` BGR image.
        exclude_hsv: The range (OpenCV 8-bit HSV), or None to exclude nothing.

    Returns:
        ``(h, w)`` boolean mask.
    """
    if exclude_hsv is None:
        return np.zeros(image.shape[:2], dtype=bool)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    excluded = np.ones(image.shape[:2], dtype=bool)
    for name, channel, compare in _HSV_BOUNDS:
        bound = getattr(exclude_hsv, name)
        if bound is not None:
            excluded &= compare(hsv[:, :, channel], bound)
    return excluded


@dataclass(frozen=True)
class Prompts:
    """SAM2 prompts of one view.

    Attributes:
        box: ``(4,)`` ``[x0, y0, x1, y1]`` pixels (may reach past the image).
        points: ``(n, 2)`` point prompts, pixels.
        labels: ``(n,)`` 1 for a positive point, 0 for the negative one.
    """

    box: NDArray[np.float64]
    points: NDArray[np.float64]
    labels: NDArray[np.int32]

    def to_record(self) -> dict[str, Any]:
        """The prompts as a JSON object."""
        return {"box": self.box.tolist(), "points": self.points.tolist(),
                "labels": self.labels.tolist()}  # fmt: skip


def _positive_points(label: ViewLabel, hsv_value: NDArray[np.uint8],
                     excluded: NDArray[np.bool_]) -> NDArray[np.float64]:  # fmt: skip
    width, height = label.image_size
    eligible = []
    for point, visible in zip(label.keypoints_xy, label.ray_visible, strict=True):
        x, y = np.round(point).astype(int)
        if (visible and 0 <= x < width and 0 <= y < height and not excluded[y, x]
                and hsv_value[y, x] < PROMPT_MAX_VALUE):  # fmt: skip
            eligible.append(point)
    if not eligible:
        return label.keypoints_xy.mean(axis=0)[None, :]
    points = np.array(eligible, dtype=np.float64)
    picks = np.linspace(0, len(points) - 1, min(MAX_POSITIVE_POINTS, len(points)))
    return points[picks.round().astype(int)]


def sam2_prompts(view_label: ViewLabel, image: NDArray[np.uint8], *,
                 exclude_hsv: HsvExclusion | None) -> Prompts:  # fmt: skip
    """The SAM2 prompts of a view from its CAD silhouette and keypoints.

    Args:
        view_label: The view's geometry label with its ``cad_mask`` array.
        image: ``(h, w, 3)`` BGR render of the view.
        exclude_hsv: The excluded color (recipe ``labels.exclude_hsv``), or None.

    Returns:
        The box, the positive points and the optional negative point.

    Raises:
        ValueError: Without a CAD mask, for an empty one or an image of another size.
    """
    mask = view_label.cad_mask
    if mask is None:
        raise ValueError(f"view {view_label.index}: sam2_prompts needs the label's cad_mask array")
    width, height = view_label.image_size
    if image.shape[:2] != (height, width) or mask.shape != (height, width):
        raise ValueError(f"view {view_label.index}: image {image.shape[:2]} and CAD mask "
                         f"{mask.shape} must be {(height, width)}")  # fmt: skip
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError(f"view {view_label.index}: the CAD silhouette is empty")
    box = np.array(
        [
            xs.min() - BOX_MARGIN_PX,
            ys.min() - BOX_MARGIN_PX,
            xs.max() + BOX_MARGIN_PX,
            ys.max() + BOX_MARGIN_PX,
        ],
        dtype=np.float64,
    )
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    excluded = excluded_pixels(image, exclude_hsv)
    points = _positive_points(view_label, np.asarray(hsv[:, :, 2], np.uint8), excluded)
    labels = np.ones(len(points), np.int32)
    region = np.zeros((height, width), dtype=bool)
    region[max(0, int(box[1])):min(height, int(box[3])),
           max(0, int(box[0])):min(width, int(box[2]))] = True  # fmt: skip
    negative_y, negative_x = np.nonzero(excluded & region)
    if len(negative_x) > MIN_NEGATIVE_PIXELS:
        points = np.vstack([points, [np.median(negative_x), np.median(negative_y)]])
        labels = np.r_[labels, 0].astype(np.int32)
    return Prompts(box=box, points=points, labels=labels)  # pyright: ignore[reportArgumentType]


class Segmenter(Protocol):
    """Segments one view from its prompts."""

    def segment(
        self, image: NDArray[np.uint8], prompts: Prompts
    ) -> tuple[NDArray[np.uint8], float]:
        """``(mask of 0 and 255, score)`` of a BGR image."""
        ...


class Sam2Segmenter:
    """SAM2's image predictor: one mask per view.

    Args:
        config: The SAM2 config identifier (default ``SAM2_CONFIG``, hiera small).
        weights: The SAM2 checkpoint file (user-supplied; ``sam2.1_hiera_small.pt`` for the
            default config).
        device: Torch device; CUDA runs with bf16 autocast.

    Raises:
        ImportError: Without the ``nurec-sam`` extra (torch and sam2).
        FileNotFoundError: For missing weights.
    """

    def __init__(
        self, config: str = SAM2_CONFIG, *, weights: str | Path, device: str = "cuda"
    ) -> None:
        """Build the predictor (imports torch and sam2)."""
        try:
            import torch
            from sam2.build_sam import build_sam2  # pyright: ignore[reportMissingImports]
            from sam2.sam2_image_predictor import (  # pyright: ignore[reportMissingImports]
                SAM2ImagePredictor,
            )
        except ImportError as error:
            raise ImportError(f"SAM2 masks need the {SAM2_EXTRA} extra (torch and sam2): "
                              f"uv pip install 'doper[{SAM2_EXTRA}]'") from error  # fmt: skip
        if not Path(weights).is_file():
            raise FileNotFoundError(f"SAM2 weights not found: {weights}")
        self._torch = torch
        self.device = device
        with self._context():
            self._predictor = SAM2ImagePredictor(build_sam2(config, str(weights), device=device))

    def _context(self) -> contextlib.ExitStack:
        """Inference mode and bf16 autocast on the segmenter's device type."""
        torch, stack = self._torch, contextlib.ExitStack()
        stack.enter_context(torch.inference_mode())
        stack.enter_context(torch.autocast(str(self.device).split(":")[0], dtype=torch.bfloat16))
        return stack

    def segment(
        self, image: NDArray[np.uint8], prompts: Prompts
    ) -> tuple[NDArray[np.uint8], float]:
        """SAM2's mask (0 and 255) and score of a BGR image (``multimask_output=False``)."""
        with self._context():
            self._predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            masks, scores, _ = self._predictor.predict(
                point_coords=prompts.points, point_labels=prompts.labels, box=prompts.box,
                multimask_output=False,
            )  # fmt: skip
        return np.asarray(masks[0]).astype(np.uint8) * MASK_VALUE, float(scores[0])


def segment_views(
    labels: Sequence[ViewLabel],
    segmenter: Segmenter,
    *,
    image_path: Callable[[ViewLabel], Path],
    cad_mask_path: Callable[[ViewLabel], Path],
    out: str | Path,
    exclude_hsv: HsvExclusion | None,
) -> list[dict[str, Any]]:
    """SAM2 masks of many views, written as ``SAM_MASK_PATTERN`` PNGs (0 and 255) in ``out``.

    Args:
        labels: The geometry labels (their CAD masks are read from ``cad_mask_path``).
        segmenter: The segmenter (``Sam2Segmenter``).
        image_path: The render of a view.
        cad_mask_path: The CAD silhouette PNG of a view.
        out: Output directory (created).
        exclude_hsv: The excluded color.

    Returns:
        Per view, in order: ``index``, ``score``, ``mask_file`` and its ``prompts``.

    Raises:
        FileNotFoundError: For a missing render or CAD mask.
        OSError: When a mask cannot be written.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for label in labels:
        image = read_image(image_path(label), cv2.IMREAD_COLOR)
        cad = read_image(cad_mask_path(label), cv2.IMREAD_GRAYSCALE)
        prompts = sam2_prompts(replace(label, cad_mask=cad), image, exclude_hsv=exclude_hsv)
        mask, score = segmenter.segment(image, prompts)
        name = SAM_MASK_PATTERN.format(index=label.index)
        if not cv2.imwrite(str(out / name), np.ascontiguousarray(mask, dtype=np.uint8)):
            raise OSError(f"could not write {out / name}")
        rows.append({"index": label.index, "score": score, "mask_file": name,
                     "prompts": prompts.to_record()})  # fmt: skip
    return rows


__all__ = [
    "BOX_MARGIN_PX",
    "MAX_POSITIVE_POINTS",
    "MIN_NEGATIVE_PIXELS",
    "PROMPT_MAX_VALUE",
    "SAM2_CONFIG",
    "SAM2_EXTRA",
    "SAM_MASK_PATTERN",
    "SAM_SCORES_NAME",
    "Prompts",
    "Sam2Segmenter",
    "Segmenter",
    "excluded_pixels",
    "sam2_prompts",
    "segment_views",
]
