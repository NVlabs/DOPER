# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SPEED+ checkpoints: the rotation-aux payload and the keypoint-layout identity.

A SPEED+ model is trained on one 75-point ``keypoints_3d.json`` layout and is only meaningful
with that layout. A layout's identity is its **points hash** (the SHA-256 of its points as
little-endian float64, ``doper.speedplus.data.layout_points_sha256``); its file SHA-256 is
checked too. Every new SPEED+ checkpoint records the layout three times: the scale-invariant
``cfg.keypoints_3d_fingerprint`` every DOPER checkpoint carries, ``speedplus_layout_sha256``
(the file SHA-256) and ``speedplus_layout_points_sha256`` (the points hash).

Every SPEED+ and SHIRT consumer checks its checkpoint with :func:`check_checkpoint_layout` (or
loads it through :func:`load_speedplus_checkpoint`), against the :class:`LayoutFile` it is
given:

* the layout the checkpoint was trained on comes from its recorded keys or, for a legacy
  (source-format) checkpoint, which records none, from the recipe's ``legacy_checkpoints``
  registry (checkpoint file SHA-256 -> layout file SHA-256); an unregistered legacy checkpoint
  is refused;
* when that layout's points hash is known (recorded, or for a file SHA-256 in
  ``doper.speedplus.definitions.KNOWN_LAYOUT_POINTS``), the points hashes must match; a matching
  points hash with a different file SHA-256 is a **formatting deviation**, accepted and recorded
  (``layout_formatting_deviation``) in every consumer summary;
* otherwise the file SHA-256s must match.

A refusal names both SHA-256s.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from ..bop.sources import sha256_file
from ..checkpoint_io import load_checkpoint
from ..keypoints import keypoints_3d_fingerprint, load_keypoints_3d
from ..rotation_aux import RotationAuxPayload
from .data import layout_points_sha256
from .definitions import KNOWN_LAYOUT_POINTS, LayoutPins

# Checkpoint key holding the SHA-256 of the ``keypoints_3d.json`` file the model trained on.
LAYOUT_SHA_KEY = "speedplus_layout_sha256"
# Checkpoint key holding the points hash of that layout (its identity).
LAYOUT_POINTS_SHA_KEY = "speedplus_layout_points_sha256"

LayoutCheck = Literal["recorded", "legacy"]


def load_layout_points(path: str | Path) -> np.ndarray:
    """Read the ``(N, 3)`` float64 ``keypoints_3d`` points of a layout file.

    Raises:
        ValueError: When the points are not a non-empty, finite ``(N, 3)`` array.
    """
    points = load_keypoints_3d(path)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError(f"layout {path} keypoints_3d must be (N, 3), got shape {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError(f"layout {path} keypoints_3d holds non-finite values")
    return points


@dataclass(frozen=True)
class LayoutFile:
    """A ``keypoints_3d.json`` layout file, its points and its identity.

    Attributes:
        path: The layout file.
        points: ``(N, 3)`` float64 body-frame keypoints.
        identity: Its file SHA-256 and points hash.
    """

    path: Path
    points: np.ndarray
    identity: LayoutPins

    @property
    def sha256(self) -> str:
        """SHA-256 of the layout file."""
        return self.identity.sha256

    @property
    def points_sha256(self) -> str:
        """The points hash, the layout's identity."""
        return self.identity.points_sha256


def load_layout(path: str | Path) -> LayoutFile:
    """Load a layout file with its file SHA-256 and points hash.

    Raises:
        ValueError: As :func:`load_layout_points`.
    """
    points = load_layout_points(path)
    identity = LayoutPins(sha256=sha256_file(path), points_sha256=layout_points_sha256(points))
    return LayoutFile(Path(path), points, identity)


@dataclass(frozen=True)
class LayoutMatch:
    """How a checkpoint's layout was accepted.

    Attributes:
        check: ``recorded`` (the checkpoint records its layout) or ``legacy`` (accepted through
            the recipe registry).
        formatting_deviation: The points hashes match but the file SHA-256s differ.
    """

    check: LayoutCheck
    formatting_deviation: bool = False

    def summary_fields(self) -> dict[str, Any]:
        """``layout_check`` and ``layout_formatting_deviation``, for a consumer summary."""
        return {
            "layout_check": self.check,
            "layout_formatting_deviation": self.formatting_deviation,
        }


def speedplus_checkpoint_payload(
    *,
    model_state: Mapping[str, torch.Tensor],
    model_cfg: Mapping[str, Any],
    rotation_aux: RotationAuxPayload,
    keypoints_3d: str | Path,
    epoch: int,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a new-format SPEED+ checkpoint payload.

    The payload has the source trainers' keys (``model``, ``epoch``, ``cfg.model``,
    ``metrics``, ``rotation_aux``) with the bins stored as a float32 tensor, plus
    ``cfg.keypoints_3d_fingerprint``, :data:`LAYOUT_SHA_KEY` and :data:`LAYOUT_POINTS_SHA_KEY`.
    Free-form provenance (the source's ``training_provenance``) belongs in run manifests, not
    here.

    Args:
        model_state: The model ``state_dict``.
        model_cfg: The model configuration mapping.
        rotation_aux: The rotation head and its bins.
        keypoints_3d: The ``keypoints_3d.json`` layout file the model trained on.
        epoch: Epoch number.
        metrics: Epoch metrics.

    Returns:
        The payload, ready for ``doper.checkpoint_io.atomic_torch_save``.
    """
    layout = load_layout(keypoints_3d)
    return {
        "model": dict(model_state),
        "epoch": int(epoch),
        "cfg": {
            "model": dict(model_cfg),
            "keypoints_3d_fingerprint": keypoints_3d_fingerprint(layout.points),
        },
        "metrics": dict(metrics),
        "rotation_aux": rotation_aux.to_checkpoint(),
        LAYOUT_SHA_KEY: layout.sha256,
        LAYOUT_POINTS_SHA_KEY: layout.points_sha256,
    }


def _trained_layout(
    checkpoint: Mapping[str, Any],
    checkpoint_sha256: str,
    legacy_checkpoints: Mapping[str, str],
    given: LayoutPins,
) -> tuple[LayoutCheck, str | None, str | None]:
    """The check kind and the file SHA-256 and points hash of the layout a checkpoint used.

    Raises:
        ValueError: For an unregistered legacy checkpoint.
    """
    if LAYOUT_SHA_KEY in checkpoint or LAYOUT_POINTS_SHA_KEY in checkpoint:
        file_sha = checkpoint.get(LAYOUT_SHA_KEY)
        points = checkpoint.get(LAYOUT_POINTS_SHA_KEY) or KNOWN_LAYOUT_POINTS.get(str(file_sha))
        return "recorded", file_sha, points
    registered = legacy_checkpoints.get(checkpoint_sha256)
    if registered is None:
        raise ValueError(
            f"legacy checkpoint {checkpoint_sha256} records no keypoint layout and is not in "
            f"the recipe's legacy_checkpoints registry; refusing it with layout {given.sha256} "
            f"(points {given.points_sha256})"
        )
    return "legacy", registered, KNOWN_LAYOUT_POINTS.get(registered)


def check_checkpoint_layout(
    checkpoint: Mapping[str, Any],
    checkpoint_sha256: str,
    layout: LayoutPins | LayoutFile,
    legacy_checkpoints: Mapping[str, str],
) -> LayoutMatch:
    """Refuse a SPEED+ checkpoint that was not trained on the given keypoint layout.

    Args:
        checkpoint: The loaded checkpoint payload.
        checkpoint_sha256: SHA-256 of the checkpoint file.
        layout: The identity (or the loaded file) of the layout the caller will use.
        legacy_checkpoints: The recipe registry, checkpoint SHA-256 -> layout file SHA-256.

    Returns:
        How the layout was accepted, and whether it is a formatting deviation.

    Raises:
        ValueError: On a layout mismatch (naming both SHA-256s), or for an unregistered legacy
            checkpoint.
    """
    given = layout.identity if isinstance(layout, LayoutFile) else layout
    check, file_sha, points_sha = _trained_layout(
        checkpoint, checkpoint_sha256, legacy_checkpoints, given
    )
    trained = "trained on" if check == "recorded" else "registered with"
    kind = "checkpoint" if check == "recorded" else "legacy checkpoint"
    if points_sha is not None:
        if points_sha != given.points_sha256:
            raise ValueError(
                f"{kind} {checkpoint_sha256} was {trained} keypoint layout points "
                f"{points_sha} (file {file_sha}), but layout points {given.points_sha256} "
                f"(file {given.sha256}) were given"
            )
        return LayoutMatch(check, file_sha != given.sha256)
    if file_sha != given.sha256:
        raise ValueError(
            f"{kind} {checkpoint_sha256} was {trained} keypoint layout {file_sha}, but layout "
            f"{given.sha256} (points {given.points_sha256}) was given"
        )
    return LayoutMatch(check)


def load_speedplus_checkpoint(
    path: str | Path,
    keypoints_3d: str | Path,
    legacy_checkpoints: Mapping[str, str],
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a SPEED+ checkpoint (``weights_only``) and check it against a keypoint layout.

    Args:
        path: Checkpoint file.
        keypoints_3d: The ``keypoints_3d.json`` layout file the caller will use.
        legacy_checkpoints: The recipe registry, checkpoint SHA-256 -> layout file SHA-256.
        map_location: Device the tensors are mapped to.

    Returns:
        The checkpoint payload.

    Raises:
        ValueError: As :func:`check_checkpoint_layout`.
    """
    checkpoint = load_checkpoint(path, map_location)
    check_checkpoint_layout(
        checkpoint, sha256_file(path), load_layout(keypoints_3d), legacy_checkpoints
    )
    return checkpoint
