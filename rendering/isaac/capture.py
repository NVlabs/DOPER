# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated frame capture without importing Kit or holding annotator-owned arrays."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

RETRY_SLEEP_SEC = 0.05


class FrameCapture:
    """Step until all annotators describe a usable frame within a monotonic budget.

    The launcher watchdog must handle a step that blocks inside Kit; this class bounds retries
    and rejects frames returned after the deadline. Call :meth:`prime` once BEFORE applying the
    first saved scene/camera pose to initialize the render graph. Its discarded step costs one
    requested ``rt_subframes`` step per worker; normal captures have no extra warmup step.
    """

    def __init__(
        self,
        *,
        step: Callable[..., Any],
        rgb: Callable[[], Any],
        depth: Callable[[], Any],
        segmentation: Callable[[], Any],
        resolution: int,
        object_path: str,
        rt_subframes: int,
        timeout_sec: float,
        clock: Callable[[], float] = time.monotonic,
        prime_step: Callable[..., Any] | None = None,
    ) -> None:
        """Bind the Kit step and annotator getters.

        Args:
            step: Advances the renderer (``rep.orchestrator.step`` keyword arguments).
            rgb: Returns the RGB(A) annotator buffer.
            depth: Returns the distance-to-image-plane buffer.
            segmentation: Returns the instance segmentation ``{"data", "info"}`` mapping.
            resolution: Square image size in pixels.
            object_path: Absolute prim path of the object; its subtree forms the mask.
            rt_subframes: Subframes per step.
            timeout_sec: Budget of one :meth:`prime` or :meth:`capture` call.
            clock: Monotonic clock (injectable for tests).
            prime_step: Step used by :meth:`prime`; defaults to ``step``.

        Raises:
            ValueError: A size, budget, or object path is invalid.
        """
        if resolution <= 0 or rt_subframes <= 0:
            raise ValueError("resolution and rt_subframes must be positive")
        if not math.isfinite(timeout_sec) or timeout_sec <= 0:
            raise ValueError("timeout_sec must be finite and positive")
        if not object_path.startswith("/") or object_path == "/":
            raise ValueError("object_path must identify an absolute object prim")
        self.step = step
        self.prime_step = prime_step if prime_step is not None else step
        self.rgb = rgb
        self.depth = depth
        self.segmentation = segmentation
        self.resolution = resolution
        self.object_path = object_path.rstrip("/")
        self.rt_subframes = rt_subframes
        self.timeout_sec = timeout_sec
        self.clock = clock

    def _frame(
        self,
        rgb_data: Any,  # noqa: ANN401 - annotator buffer
        depth_data: Any,  # noqa: ANN401 - annotator buffer
        segmentation: Any,  # noqa: ANN401 - annotator mapping
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        shape = (self.resolution, self.resolution)
        rgb = np.asarray(rgb_data)
        if rgb.ndim != 3 or rgb.shape[:2] != shape or rgb.shape[2] not in (3, 4):
            raise ValueError(f"invalid RGB shape {rgb.shape}")
        if rgb.dtype != np.uint8:
            raise ValueError(f"invalid RGB dtype {rgb.dtype}; expected uint8")
        depth = np.asarray(depth_data, dtype=np.float32)
        if depth.shape != shape:
            raise ValueError(f"invalid depth shape {depth.shape}")
        if not isinstance(segmentation, Mapping):
            raise ValueError("segmentation is not a mapping")
        ids = np.asarray(segmentation.get("data"))
        info = segmentation.get("info")
        labels = info.get("idToLabels") if isinstance(info, Mapping) else None
        if ids.shape != shape or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError(f"invalid segmentation IDs {ids.shape}/{ids.dtype}")
        if not isinstance(labels, Mapping):
            raise ValueError("segmentation has no idToLabels mapping")
        mask = np.zeros(shape, dtype=bool)
        for identifier, label in labels.items():
            if not isinstance(label, str):
                raise ValueError("segmentation labels must be prim path strings")
            if label == self.object_path or label.startswith(self.object_path + "/"):
                mask |= ids == int(identifier)
        if rgb.shape[2] == 4 and mask.any() and not rgb[..., 3][mask].any():
            # Kit can publish initialized depth/IDs before its RGBA render buffer.
            # Alpha gates readiness here; fully black opaque pixels remain valid.
            raise ValueError("foreground alpha is entirely zero; RGBA capture not ready")
        if np.isnan(depth).any() or (depth < 0).any():
            raise ValueError("depth contains NaN or negative distances")
        if mask.any() and (not np.isfinite(depth[mask]).all() or (depth[mask] <= 0).any()):
            raise ValueError("foreground depth must be finite and positive")
        return rgb[..., :3].copy(), depth.copy(), mask

    def prime(self) -> None:
        """Discard one initial-scene step before the caller applies saved poses.

        Raises:
            TimeoutError: The step returned after the budget.
        """
        deadline = self.clock() + self.timeout_sec
        self.prime_step(rt_subframes=self.rt_subframes, delta_time=0.0, pause_timeline=True)
        if self.clock() >= deadline:
            raise TimeoutError(f"frame capture prime timed out after {self.timeout_sec:g}s")

    def capture(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step and read the annotators until they describe one valid frame.

        Returns:
            Copies of the RGB image (uint8), depth (float32) and object mask (bool).

        Raises:
            TimeoutError: No valid frame arrived within the budget.
        """
        deadline = self.clock() + self.timeout_sec
        attempts = 0
        reason = "annotators not ready"
        while self.clock() < deadline:
            attempts += 1
            self.step(rt_subframes=self.rt_subframes, delta_time=0.0, pause_timeline=True)
            if self.clock() >= deadline:
                break
            # Operational annotator exceptions propagate; malformed buffers retry.
            rgb, depth, segmentation = self.rgb(), self.depth(), self.segmentation()
            if self.clock() >= deadline:
                break
            try:
                return self._frame(rgb, depth, segmentation)
            except (ValueError, TypeError, OverflowError) as exc:
                reason = str(exc)
            remaining = deadline - self.clock()
            if remaining > 0:
                time.sleep(min(RETRY_SLEEP_SEC, remaining))
        raise TimeoutError(
            f"frame capture timed out after {self.timeout_sec:g}s ({attempts} steps): {reason}"
        )
