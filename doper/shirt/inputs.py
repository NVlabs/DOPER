# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the SHIRT stages share: records, the checkpoint and layout checks, and the model.

The framewise evaluation, the re-crop candidates and the heatmap cache all run the SPEED+ model
on SHIRT frames, so they check their checkpoint the same way (:func:`check_model_inputs`): the
recipe's checkpoint pin (``check_checkpoint_pin``), then the keypoint layout the checkpoint was
trained on (``doper.speedplus.checkpoints.check_checkpoint_layout``), before any model is
loaded. The later stages read the earlier stages' records (:func:`load_records`) and refuse
records of another checkpoint (:func:`recorded_checkpoint`).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..bop.sources import sha256_file
from ..checkpoint_io import load_checkpoint
from ..recipes import load_any_recipe
from ..recipes.shirt_schema import ShirtRecipe
from ..speedplus.checkpoints import LayoutFile, LayoutMatch, check_checkpoint_layout, load_layout
from ..speedplus.pose import Normalization
from . import check_request as check_names
from . import data

RECORDS_NAME = "records.jsonl"
SUMMARY_NAME = "summary.json"
# Inference: the top-1 detection at any score.
DETECTION_SCORE_THRESHOLD = 0.0
MAX_DETECTIONS = 1
# SHIRT normalizes the ground-truth quaternion in float64.
GROUND_TRUTH_NORMALIZATION: Normalization = "float64"
# The stage that writes the records each stage reads, for the messages.
PRODUCERS = {"framewise": "framewise", "candidate": "candidates"}


def packaged_recipe() -> ShirtRecipe:
    """The packaged ``shirt`` recipe.

    Raises:
        ValueError: When the packaged recipe is not a SHIRT recipe.
    """
    recipe = load_any_recipe("shirt")
    if not isinstance(recipe, ShirtRecipe):
        raise ValueError("the packaged shirt recipe is not a SHIRT recipe")
    return recipe


def check_request(trajectories: Sequence[str], domain: str, recipe: ShirtRecipe) -> None:
    """Refuse trajectories or a domain the recipe's SHIRT profile does not release.

    Raises:
        ValueError: As ``doper.shirt.check_request``.
    """
    profile = data.load_profile(recipe.profile)
    check_names(trajectories, domain, profile.trajectories, profile.domains)


def check_checkpoint_pin(checkpoint_sha256: str, pin: str | None, allow_unpinned: bool) -> bool:
    """Whether the checkpoint is the pinned one; refuse another unless ``allow_unpinned``.

    Raises:
        ValueError: For an unpinned checkpoint without ``allow_unpinned``.
    """
    pinned = pin is not None and checkpoint_sha256 == pin
    if pin is not None and not pinned and not allow_unpinned:
        raise ValueError(
            f"checkpoint {checkpoint_sha256} is not the recipe's pinned checkpoint {pin}; "
            "pass --allow-unpinned-checkpoint to evaluate it (the summary records that)"
        )
    return pinned


def sequence_dir(out: Path, roe: str, domain: str) -> Path:
    """``<out>/<roe>/<domain>``, where a stage writes one trajectory's outputs."""
    return Path(out) / roe / domain


def load_records(
    root: Path, roe: str, domain: str, limit: int = 0, kind: str = "candidate"
) -> list[dict[str, Any]]:
    """The ``kind`` (``framewise`` or ``candidate``) records of one trajectory, by frame.

    Raises:
        FileNotFoundError: Without records.
        ValueError: Without records in the file.
    """
    path = sequence_dir(root, roe, domain) / RECORDS_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{kind} records {path} not found; run `doper evaluate shirt --method "
            f"{PRODUCERS.get(kind, kind)}`"
        )
    records = sorted(
        (json.loads(line) for line in path.read_text().splitlines() if line.strip()),
        key=lambda record: record["index"],
    )
    if limit > 0:
        records = records[:limit]
    if not records:
        raise ValueError(f"{kind} records {path} are empty")
    return records


def recorded_summary(root: Path) -> dict[str, Any] | None:
    """A stage's top-level ``summary.json``, or None when it has none."""
    path = Path(root) / SUMMARY_NAME
    return json.loads(path.read_text()) if path.is_file() else None


def recorded_checkpoint(
    root: Path, kind: str = "candidate", checkpoint_sha256: str | None = None
) -> str | None:
    """The checkpoint a stage's records came from; refuse another than ``checkpoint_sha256``.

    Raises:
        ValueError: When the stage's summary names a different checkpoint.
    """
    summary = recorded_summary(root) or {}
    recorded = summary.get("checkpoint_sha256")
    if checkpoint_sha256 is not None and recorded is not None and recorded != checkpoint_sha256:
        raise ValueError(
            f"the {kind} records in {root} come from checkpoint {recorded}, not "
            f"{checkpoint_sha256}; re-run `doper evaluate shirt --method "
            f"{PRODUCERS.get(kind, kind)}` with this checkpoint"
        )
    return recorded


def read_image(shirt_root: Path, roe: str, domain: str, filename: str) -> np.ndarray:
    """One SHIRT frame as BGR.

    Raises:
        FileNotFoundError: When the image cannot be read.
    """
    path = data.image_path(shirt_root, roe, domain, filename)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def load_model(checkpoint: Path, num_keypoints: int, device: str) -> tuple[Any, Any]:  # noqa: ANN401
    """The model of a SPEED+ checkpoint and the config it was loaded with."""
    from .. import infer
    from ..config import DataConfig, DoperConfig, ModelConfig

    cfg = DoperConfig(
        data=DataConfig(root="."), model=ModelConfig(pretrained=False, num_keypoints=num_keypoints)
    )
    return infer.load_model(str(checkpoint), cfg, device), cfg


def heatmap_beta(payload: Mapping[str, Any], cfg: Any) -> float:  # noqa: ANN401
    """The checkpoint's ``cfg.loss.keypoint_heatmap_beta``; the loss default when not recorded.

    SPEED+ checkpoints record only ``cfg.model``, so they decode with the default (that of a
    default ``LossConfig``). ``cfg`` is updated in place.
    """
    from ..checkpoint_io import checkpoint_configs

    _, checkpoint_cfg, _ = checkpoint_configs(dict(payload), "checkpoint")
    loss = checkpoint_cfg.get("loss") or {}
    beta = float(loss.get("keypoint_heatmap_beta", cfg.loss.keypoint_heatmap_beta))
    cfg.loss.keypoint_heatmap_beta = beta
    return beta


@dataclass(frozen=True)
class ModelInputs:
    """A checked checkpoint and the layout it was trained on.

    Attributes:
        checkpoint: The checkpoint file.
        checkpoint_sha256: Its SHA-256.
        pin: The recipe's pinned checkpoint SHA-256.
        pinned: Whether the checkpoint is the pinned one.
        allow_unpinned: Whether an unpinned checkpoint was allowed.
        layout: The layout file, its points and identity.
        match: How the layout was accepted.
        payload: The loaded checkpoint payload.
    """

    checkpoint: Path
    checkpoint_sha256: str
    pin: str | None
    pinned: bool
    allow_unpinned: bool
    layout: LayoutFile
    match: LayoutMatch
    payload: Mapping[str, Any]

    def summary_fields(self) -> dict[str, Any]:
        """The checkpoint and layout identity every model stage's summary records."""
        return {
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_pin": self.pin,
            "checkpoint_pinned": self.pinned,
            "allow_unpinned_checkpoint": self.allow_unpinned,
            "layout_sha256": self.layout.sha256,
            "layout_points_sha256": self.layout.points_sha256,
            **self.match.summary_fields(),
            "num_keypoints": len(self.layout.points),
        }


def check_model_inputs(
    checkpoint: str | Path,
    layout: str | Path,
    recipe: ShirtRecipe,
    allow_unpinned_checkpoint: bool,
) -> ModelInputs:
    """Check a checkpoint against the recipe pin and the layout, before loading any model.

    Raises:
        ValueError: For an unpinned checkpoint without ``allow_unpinned_checkpoint``, a layout
            that is not ``(N, 3)`` finite points, or a checkpoint not trained on ``layout``
            (naming both SHA-256s).
    """
    checkpoint = Path(checkpoint)
    checkpoint_sha = sha256_file(checkpoint)
    pinned = check_checkpoint_pin(
        checkpoint_sha, recipe.checkpoint_sha256, allow_unpinned_checkpoint
    )
    layout_file = load_layout(layout)
    payload = load_checkpoint(checkpoint, "cpu")
    match = check_checkpoint_layout(payload, checkpoint_sha, layout_file, recipe.legacy_checkpoints)
    return ModelInputs(
        checkpoint, checkpoint_sha, recipe.checkpoint_sha256, pinned, allow_unpinned_checkpoint,
        layout_file, match, payload,
    )  # fmt: skip
