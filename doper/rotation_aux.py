# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coarse-orientation auxiliary head, SO(3) targets and pose-label utilities.

The head classifies a pooled visual descriptor against a fixed quaternion codebook (the rotation
bins). Quaternions are scalar-first, and their sign is removed with an absolute dot product, so
``q`` and ``-q`` describe the same rotation. The codebook is deliberately external to
:class:`CosineRotationHead`: checkpoints store the exact ordered bins alongside the learned weight
matrix (:class:`RotationAuxPayload`) so teacher scoring uses the same classes.
Integrated DOPER training stores the bins as a model buffer and the classifier
as a model submodule; the payload class also supports standalone SPEED+ heads.

Like ``doper.model``, this module imports torch; ``import doper`` does not load it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional

from .config import ModelConfig

# Pool size of the auxiliary descriptor when a payload does not record one.
DEFAULT_POOL_SIZE = 4
# Number of ranked bins the oracle-best metric looks at.
_TOP_K = 5


def normalize_quaternions(values: np.ndarray) -> np.ndarray:
    """Normalize an ``(N, 4)`` scalar-first quaternion array.

    The result is float32, as in the source: SPEED+ ground-truth matrices built from it
    (``doper.speedplus.pose.quaternion_to_object_camera``) carry float32 rounding.

    Args:
        values: ``(N, 4)`` quaternions ``(w, x, y, z)``.

    Returns:
        ``(N, 4)`` float32 unit quaternions.

    Raises:
        ValueError: If the shape is not ``(N, 4)`` or a quaternion is zero or non-finite.
    """
    quaternions = np.asarray(values, dtype=np.float32)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError(
            f"expected scalar-first quaternions with shape (N, 4), got {quaternions.shape}"
        )
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise ValueError("quaternions must be finite and non-zero")
    return quaternions / norms


def farthest_rotation_bins(quaternions: np.ndarray, num_bins: int, seed: int) -> np.ndarray:
    """Select deterministic approximate k-centers in SO(3).

    Distance is sign-invariant because unit quaternions ``q`` and ``-q`` represent the same
    rotation.

    Args:
        quaternions: ``(N, 4)`` scalar-first candidate poses.
        num_bins: Number of bins to select.
        seed: Seed of the random first center.

    Returns:
        ``(num_bins,)`` int64 indices into ``quaternions``.

    Raises:
        ValueError: If ``num_bins`` is not positive or exceeds the number of poses.
    """
    quaternions = normalize_quaternions(quaternions)
    if num_bins <= 0:
        raise ValueError("rotation-bin count must be positive")
    if num_bins > len(quaternions):
        raise ValueError(f"requested {num_bins} bins from {len(quaternions)} poses")
    rng = np.random.default_rng(seed)
    selected = np.empty(num_bins, dtype=np.int64)
    selected[0] = int(rng.integers(len(quaternions)))
    nearest_dot = np.abs(quaternions @ quaternions[selected[0]])
    for index in range(1, num_bins):
        selected[index] = int(np.argmin(nearest_dot))
        nearest_dot = np.maximum(nearest_dot, np.abs(quaternions @ quaternions[selected[index]]))
    return selected


def rotation_errors_deg(
    query_quaternions: np.ndarray, candidate_quaternions: np.ndarray
) -> np.ndarray:
    """Pairwise geodesic rotation errors of two quaternion arrays.

    Args:
        query_quaternions: ``(N, 4)`` scalar-first quaternions.
        candidate_quaternions: ``(M, 4)`` scalar-first quaternions.

    Returns:
        ``(N, M)`` float32 errors in degrees.
    """
    query = normalize_quaternions(query_quaternions)
    candidates = normalize_quaternions(candidate_quaternions)
    dots = np.abs(query @ candidates.T)
    return np.degrees(2.0 * np.arccos(np.clip(dots, 0.0, 1.0)))


def pooled_rotation_features(
    features: torch.Tensor, pool_size: int = DEFAULT_POOL_SIZE
) -> torch.Tensor:
    """Pool a fused keypoint-FPN map into the auxiliary descriptor.

    Args:
        features: ``(B, C, H, W)`` feature map.
        pool_size: Side of the adaptive average pool.

    Returns:
        ``(B, C * pool_size**2)`` descriptors.

    Raises:
        ValueError: For a non-4D map or a non-positive pool size.
    """
    if features.ndim != 4:
        raise ValueError("fused_features must have shape (B, C, H, W)")
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    return functional.adaptive_avg_pool2d(features, (pool_size, pool_size)).flatten(1)


class CosineRotationHead(nn.Module):
    """Cosine classifier of the coarse-orientation objective.

    The logit of bin ``k`` is ``logit_scale * cos(feature, weight[k])``.
    """

    def __init__(
        self,
        feature_dim: int,
        num_bins: int,
        logit_scale: float = 30.0,
        pool_size: int = DEFAULT_POOL_SIZE,
    ) -> None:
        """Build the head with normally initialized weights.

        Args:
            feature_dim: Descriptor size (``C * pool_size**2``).
            num_bins: Number of rotation bins.
            logit_scale: Cosine logit scale.
            pool_size: Pool size of the descriptors this head classifies.

        Raises:
            ValueError: For a non-positive size or a non-finite or non-positive scale.
        """
        super().__init__()
        if feature_dim <= 0 or num_bins <= 0:
            raise ValueError("feature_dim and num_bins must be positive")
        if not math.isfinite(logit_scale) or logit_scale <= 0.0:
            raise ValueError("logit_scale must be finite and positive")
        if pool_size <= 0:
            raise ValueError("pool_size must be positive")
        self.weight = nn.Parameter(torch.empty(num_bins, feature_dim))
        nn.init.normal_(self.weight, std=feature_dim**-0.5)
        self.logit_scale = float(logit_scale)
        self.pool_size = int(pool_size)

    @property
    def scale(self) -> float:
        """The logit scale under the name the source scripts use."""
        return self.logit_scale

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return ``(B, num_bins)`` cosine logits of ``(B, feature_dim)`` descriptors."""
        return self.logit_scale * functional.linear(
            functional.normalize(features, dim=1), functional.normalize(self.weight, dim=1)
        )


# The name the source checkpoints and scripts use.
CosineClassifier = CosineRotationHead


def soft_rotation_targets(
    quaternions: torch.Tensor, bin_quaternions: torch.Tensor, sigma_rad: float
) -> torch.Tensor:
    """Gaussian soft classes over the quaternion geodesic distance.

    Args:
        quaternions: ``(B, 4)`` scalar-first quaternions.
        bin_quaternions: ``(K, 4)`` scalar-first bin quaternions.
        sigma_rad: Gaussian width in radians (callers pass ``math.radians(15.0)``).

    Returns:
        ``(B, K)`` float32 class distributions.

    Raises:
        ValueError: For a non-finite or non-positive width or a bad shape.
    """
    if sigma_rad <= 0.0 or not math.isfinite(sigma_rad):
        raise ValueError("sigma_rad must be finite and positive")
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError("quaternions must have shape (B, 4)")
    if bin_quaternions.ndim != 2 or bin_quaternions.shape[1] != 4:
        raise ValueError("bin_quaternions must have shape (K, 4)")
    query = functional.normalize(quaternions.float(), dim=1)
    bins = functional.normalize(bin_quaternions.float(), dim=1)
    dots = torch.abs(query @ bins.T).clamp(0.0, 1.0)
    angles = 2.0 * torch.acos(dots)
    return torch.softmax(-0.5 * (angles / sigma_rad) ** 2, dim=1)


def matrix_soft_targets(
    rotations: torch.Tensor, bin_rotations: torch.Tensor, sigma_rad: float
) -> torch.Tensor:
    """Gaussian soft classes over the rotation-matrix geodesic distance (trace form).

    Equal to :func:`soft_rotation_targets` for the same rotations, up to rounding.

    Args:
        rotations: ``(B, 3, 3)`` rotation matrices.
        bin_rotations: ``(K, 3, 3)`` bin rotation matrices.
        sigma_rad: Gaussian width in radians.

    Returns:
        ``(B, K)`` class distributions in the input dtype.
    """
    traces = torch.einsum("bij,kij->bk", rotations, bin_rotations)
    angles = torch.acos(((traces - 1.0) / 2.0).clamp(-1.0, 1.0))
    return torch.softmax(-0.5 * (angles / sigma_rad) ** 2, dim=1)


def rotation_aux_loss(
    logits: torch.Tensor,
    quaternions: torch.Tensor,
    bin_quaternions: torch.Tensor,
    *,
    sigma_deg: float = 15.0,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross entropy against soft SO(3) bins, optionally weighted per sample.

    Args:
        logits: ``(B, K)`` head logits.
        quaternions: ``(B, 4)`` target quaternions.
        bin_quaternions: ``(K, 4)`` bin quaternions.
        sigma_deg: Soft-target width in degrees.
        sample_weight: Optional ``(B,)`` weights; the sum is divided by ``max(sum(w), 1)``.

    Returns:
        The scalar loss.

    Raises:
        ValueError: If ``sample_weight`` is not ``(B,)``.
    """
    targets = soft_rotation_targets(quaternions, bin_quaternions, math.radians(float(sigma_deg)))
    per_sample = -(targets * functional.log_softmax(logits.float(), dim=1)).sum(dim=1)
    if sample_weight is None:
        return per_sample.mean()
    weight = sample_weight.float().reshape(-1)
    if weight.shape != per_sample.shape:
        raise ValueError("sample_weight must have shape (B,)")
    return (per_sample * weight).sum() / weight.sum().clamp(min=1.0)


def rotation_metrics(
    logits: torch.Tensor, quaternions: torch.Tensor, bin_quaternions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-1 and oracle-best-of-top-5 angular errors in degrees.

    Args:
        logits: ``(B, K)`` head logits.
        quaternions: ``(B, 4)`` target quaternions.
        bin_quaternions: ``(K, 4)`` bin quaternions.

    Returns:
        ``(top1, top5)``, each ``(B,)``.
    """
    order = logits.argsort(dim=1, descending=True)
    query = functional.normalize(quaternions.float(), dim=1)
    bins = functional.normalize(bin_quaternions.float(), dim=1)
    dots = torch.abs(query @ bins.T).clamp(0.0, 1.0)
    errors = torch.rad2deg(2.0 * torch.acos(dots))
    top1 = errors.gather(1, order[:, :1]).squeeze(1)
    top5 = errors.gather(1, order[:, : min(_TOP_K, order.shape[1])]).amin(dim=1)
    return top1, top5


@dataclass(frozen=True, eq=False)
class RotationAuxPayload:
    """The ``rotation_aux`` section of a SPEED+ checkpoint: the head and its ordered bins.

    Attributes:
        weight: ``(num_bins, feature_dim)`` head weight (CPU).
        feature_dim: Descriptor size.
        num_bins: Number of bins.
        pool_size: Descriptor pool size.
        logit_scale: Cosine logit scale.
        bin_quaternions: ``(num_bins, 4)`` float32 unit scalar-first bin quaternions.
    """

    weight: torch.Tensor
    feature_dim: int
    num_bins: int
    pool_size: int
    logit_scale: float
    bin_quaternions: torch.Tensor

    @classmethod
    def from_head(
        cls, head: CosineRotationHead, bin_quaternions: np.ndarray | torch.Tensor
    ) -> RotationAuxPayload:
        """Capture a trained head and its bins.

        Args:
            head: The rotation head.
            bin_quaternions: ``(num_bins, 4)`` bins, in the head's class order.

        Returns:
            The payload.
        """
        weight = head.weight.detach().cpu()
        return cls._validated(
            weight=weight,
            feature_dim=int(weight.shape[1]),
            num_bins=int(weight.shape[0]),
            pool_size=head.pool_size,
            logit_scale=head.logit_scale,
            bin_quaternions=bin_quaternions,
        )

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> RotationAuxPayload:
        """Read the section from a checkpoint (or the section itself, as a probe stores it).

        Source checkpoints store the bins as a float32 ndarray, new ones as a tensor; both load.
        The bins are re-normalized, as every source consumer does on load.

        Args:
            payload: A checkpoint with a ``rotation_aux`` mapping, or that mapping.

        Returns:
            The payload.

        Raises:
            ValueError: If there is no section or its sizes disagree with its arrays.
        """
        section = payload.get("rotation_aux", payload)
        if not isinstance(section, Mapping) or "weight" not in section:
            raise ValueError("checkpoint has no rotation_aux payload")
        return cls._validated(
            weight=section["weight"].detach().cpu(),
            feature_dim=int(section["feature_dim"]),
            num_bins=int(section["num_bins"]),
            pool_size=int(section.get("pool_size", DEFAULT_POOL_SIZE)),
            logit_scale=float(section["logit_scale"]),
            bin_quaternions=section["bin_quaternions"],
        )

    @classmethod
    def _validated(
        cls,
        *,
        weight: torch.Tensor,
        feature_dim: int,
        num_bins: int,
        pool_size: int,
        logit_scale: float,
        bin_quaternions: np.ndarray | torch.Tensor,
    ) -> RotationAuxPayload:
        if isinstance(bin_quaternions, torch.Tensor):
            bin_quaternions = bin_quaternions.detach().cpu().numpy()
        bins = torch.from_numpy(normalize_quaternions(bin_quaternions))
        if tuple(weight.shape) != (num_bins, feature_dim) or len(bins) != num_bins:
            raise ValueError(
                f"rotation_aux sizes disagree: num_bins={num_bins}, feature_dim={feature_dim}, "
                f"weight {tuple(weight.shape)}, {len(bins)} bin quaternions"
            )
        return cls(weight, feature_dim, num_bins, pool_size, logit_scale, bins)

    def to_checkpoint(self) -> dict[str, Any]:
        """Return the checkpoint section, with the bins as a float32 tensor."""
        return {
            "weight": self.weight,
            "feature_dim": self.feature_dim,
            "num_bins": self.num_bins,
            "pool_size": self.pool_size,
            "logit_scale": self.logit_scale,
            "bin_quaternions": self.bin_quaternions,
        }

    def build_head(self) -> CosineRotationHead:
        """Return a CPU head carrying the stored weight."""
        head = CosineRotationHead(self.feature_dim, self.num_bins, self.logit_scale, self.pool_size)
        with torch.no_grad():
            head.weight.copy_(self.weight)
        return head


def uniform_rotation_bins(num_bins: int, seed: int) -> torch.Tensor:
    """Return a seeded SO(3) codebook, independent of training or validation poses.

    Farthest-point sampling spreads bins over uniformly sampled unit quaternions.
    The exact ordered result is persisted as a model buffer in checkpoints.
    """
    if num_bins < 2 or seed < 0:
        raise ValueError("rotation bins must be >= 2 and seed must be nonnegative")
    candidates = normalize_quaternions(
        np.random.default_rng(seed).normal(size=(max(4096, num_bins * 8), 4))
    )
    indices = farthest_rotation_bins(candidates, num_bins, seed)
    return torch.from_numpy(candidates[indices])


def annotation_quaternion(annotation: Mapping[str, Any]) -> np.ndarray | None:
    """Read an object-to-OpenCV-camera rotation as a scalar-first quaternion.

    ``object_pose`` is a 4x4 object-to-camera matrix unless ``camera_extrinsics``
    is supplied; in that case it is object-to-world and the extrinsics are
    Blender camera-to-world. SPEED+ scalar-first labels are also supported.
    Missing poses return None; malformed or improper rotations are rejected.
    """
    from scipy.spatial.transform import Rotation

    speedplus = annotation.get("speedplus_q_vbs2tango_true")
    if speedplus is not None:
        return normalize_quaternions(np.asarray(speedplus)[None])[0]
    value = annotation.get("object_pose")
    if value is None:
        return None
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("rotation supervision requires a finite 4x4 object_pose")
    extrinsics = annotation.get("camera_extrinsics")
    if extrinsics is not None:
        camera = np.asarray(extrinsics, dtype=np.float64)
        if camera.shape != (4, 4) or not np.isfinite(camera).all():
            raise ValueError("rotation supervision requires finite 4x4 camera_extrinsics")
        pose = np.diag([1.0, -1.0, -1.0, 1.0]) @ np.linalg.inv(camera) @ pose
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-4
    ):
        raise ValueError("rotation supervision requires a proper orthonormal rotation")
    xyzw = Rotation.from_matrix(rotation).as_quat()
    return xyzw[[3, 0, 1, 2]].astype(np.float32)


def require_matching_rotation_metadata(metadata: Mapping[str, Any], cfg: ModelConfig) -> None:
    """Reject changed rotation-head semantics when restoring trained weights."""
    enabled = bool(metadata.get("rotation_aux_enabled", False))
    if enabled != cfg.rotation_aux_enabled:
        raise ValueError("rotation auxiliary setting differs from checkpoint")
    if enabled:
        for name in (
            "rotation_num_bins",
            "rotation_pool_size",
            "rotation_bin_seed",
            "rotation_logit_scale",
        ):
            if metadata.get(name) != getattr(cfg, name):
                raise ValueError(f"{name} differs from checkpoint")
