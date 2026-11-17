# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared SPEED+ and SHIRT pose helpers: USAC candidates, their records and projected crops.

SHIRT shows the same spacecraft through the same USAC solve, so both recipes use these
helpers. The USAC parameters are ``doper.infer``'s (sequential USAC, 4,000 iterations, 3 px,
seed 42); unlike ``doper.infer.solve_pnp`` the solve here drops non-finite keypoints and scores
first and returns the inlier count. Torch-backed modules load only inside the functions that
need them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np

from ..rotation_aux import normalize_quaternions
from .metrics import ROTATION_DEADBAND_DEG, TRANSLATION_DEADBAND_RELATIVE, pose_score

DEFAULT_SCORE_THRESHOLD = 0.30
USAC_SOURCE = "argmax-usac4k"
PRACTICAL_ROTATION_DEG = 5.0
PRACTICAL_TRANSLATION_RELATIVE = 0.02
LOOSE_ROTATION_DEG = 10.0
LOOSE_TRANSLATION_RELATIVE = 0.05
CROP_BORDER_GREY = (114, 114, 114)
# How a ground-truth quaternion is normalized before it becomes a rotation matrix. SPEED+ normalizes
# in float32 and SHIRT in float64. The float32 floor is about 0.01 to 0.07 degrees of reported
# rotation error on a perfect pose.
Normalization = Literal["float32", "float64"]
NORMALIZATIONS: tuple[str, ...] = ("float32", "float64")


@dataclass(frozen=True)
class Candidate:
    """One pose candidate.

    Attributes:
        source: How the pose was found, for example ``argmax-usac4k``.
        rvec: ``(3, 1)`` object-to-camera Rodrigues vector.
        tvec: ``(3, 1)`` object-to-camera translation in meters.
        ransac_inliers: USAC inlier count.
        ransac_correspondences: Correspondences handed to USAC.
    """

    source: str
    rvec: np.ndarray
    tvec: np.ndarray
    ransac_inliers: int = 0
    ransac_correspondences: int = 0


def _normalized(q: np.ndarray, normalization: Normalization) -> np.ndarray:
    """``(N, 4)`` unit quaternions, normalized in float32 or row by row in float64.

    Raises:
        ValueError: For an unknown normalization.
    """
    if normalization == "float32":
        return normalize_quaternions(np.atleast_2d(q))
    if normalization == "float64":
        rows = np.atleast_2d(np.asarray(q, dtype=np.float64))
        return np.stack([row / np.linalg.norm(row) for row in rows])
    raise ValueError(f"unknown normalization {normalization!r}; known: {NORMALIZATIONS}")


def quaternion_to_object_camera(q: np.ndarray, *, normalization: Normalization) -> np.ndarray:
    """Object-to-camera rotation of scalar-first SPEED+ quaternions (the transposed DCM).

    The normalization is required, so no caller inherits one silently: SPEED+ normalizes in
    float32 (``doper.rotation_aux.normalize_quaternions``), so the matrices carry float32
    rounding, as in the SPEED+ source; SHIRT normalizes in float64, as its source does.

    Args:
        q: One ``(4,)`` quaternion or ``(N, 4)`` quaternions.
        normalization: ``float32`` (SPEED+) or ``float64`` (SHIRT).

    Returns:
        ``(3, 3)`` for one quaternion, ``(N, 3, 3)`` for ``N``.

    Raises:
        ValueError: For an unknown normalization.
    """
    single = np.ndim(q) == 1
    quaternions = _normalized(q, normalization)
    q0, q1, q2, q3 = quaternions.T
    matrices = np.empty((len(quaternions), 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = 2 * q0**2 - 1 + 2 * q1**2
    matrices[:, 0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    matrices[:, 0, 2] = 2 * q1 * q3 - 2 * q0 * q2
    matrices[:, 1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    matrices[:, 1, 1] = 2 * q0**2 - 1 + 2 * q2**2
    matrices[:, 1, 2] = 2 * q2 * q3 + 2 * q0 * q1
    matrices[:, 2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    matrices[:, 2, 1] = 2 * q2 * q3 - 2 * q0 * q1
    matrices[:, 2, 2] = 2 * q0**2 - 1 + 2 * q3**2
    rotations = np.transpose(matrices, (0, 2, 1))
    return rotations[0] if single else rotations


def argmax_usac_candidate(
    keypoints: np.ndarray,
    scores: np.ndarray,
    points_3d: np.ndarray,
    K: np.ndarray,  # noqa: N803
    dist: np.ndarray | None,
    *,
    score_thr: float = DEFAULT_SCORE_THRESHOLD,
) -> Candidate | None:
    """Solve the confident, finite heatmap argmaxes with sequential USAC.

    Args:
        keypoints: ``(K, 2)`` keypoints in image pixels.
        scores: ``(K,)`` keypoint scores.
        points_3d: ``(K, 3)`` object keypoints, ordered like ``keypoints``.
        K: ``(3, 3)`` intrinsics.
        dist: Distortion coefficients, or None for none.
        score_thr: Minimum score of a correspondence.

    Returns:
        The candidate, or None with fewer than ``doper.infer.MIN_PNP_POINTS`` correspondences
        or when the solver fails.

    Raises:
        RuntimeError: If OpenCV lacks ``cv2.UsacParams``.
    """
    from ..infer import MIN_PNP_POINTS, usac_pnp_ransac

    selected = np.isfinite(keypoints).all(axis=1) & np.isfinite(scores) & (scores >= score_thr)
    if int(selected.sum()) < MIN_PNP_POINTS:
        return None
    ok, rvec, tvec, inliers = usac_pnp_ransac(points_3d[selected], keypoints[selected], K, dist)
    if not ok:
        return None
    return Candidate(
        source=USAC_SOURCE,
        rvec=np.asarray(rvec).reshape(3, 1),
        tvec=np.asarray(tvec).reshape(3, 1),
        ransac_inliers=0 if inliers is None else len(inliers),
        ransac_correspondences=int(selected.sum()),
    )


def candidate_record(
    candidate: Candidate,
    q_gt: np.ndarray,
    t_gt: np.ndarray,
    *,
    normalization: Normalization,
) -> dict[str, Any]:
    """Serialize a candidate with its SPEED+ errors against the released ground truth.

    The errors are reporting fields only: no admission or selection decision may read them.
    ``pose_score`` is the paper SPEED score; ``pose_score_hil`` zeroes it when both errors are
    inside the official deadbands.

    Args:
        candidate: The pose candidate.
        q_gt: Scalar-first ground-truth quaternion (``q_vbs2tango_true``).
        t_gt: Ground-truth translation in meters (``r_Vo2To_vbs_true``).
        normalization: Ground-truth quaternion normalization, ``float32`` (SPEED+) or
            ``float64`` (SHIRT); see :func:`quaternion_to_object_camera`.
    """
    predicted_rotation, _ = cv2.Rodrigues(candidate.rvec)
    target_rotation = quaternion_to_object_camera(
        np.asarray(q_gt, dtype=np.float64), normalization=normalization
    )
    relative = predicted_rotation @ target_rotation.T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    rotation_deg = math.degrees(math.acos(cosine))
    target_translation = np.asarray(t_gt, dtype=np.float64).reshape(3)
    translation_m = float(np.linalg.norm(candidate.tvec.reshape(3) - target_translation))
    translation_relative = translation_m / float(np.linalg.norm(target_translation))
    score = pose_score(math.radians(rotation_deg), translation_relative, definition="paper")
    within_deadband = (
        rotation_deg < ROTATION_DEADBAND_DEG
        and translation_relative < TRANSLATION_DEADBAND_RELATIVE
    )
    correspondences = candidate.ransac_correspondences
    return {
        "source": candidate.source,
        "rvec": candidate.rvec.reshape(3).astype(float).tolist(),
        "tvec": candidate.tvec.reshape(3).astype(float).tolist(),
        "ransac_inliers": candidate.ransac_inliers,
        "ransac_correspondences": correspondences,
        "ransac_inlier_fraction": (
            candidate.ransac_inliers / correspondences if correspondences else 0.0
        ),
        "rotation_deg": rotation_deg,
        "translation_m": translation_m,
        "translation_relative": translation_relative,
        "pose_score": score,
        "pose_score_hil": 0.0 if within_deadband else score,
        "within_hil_deadband": within_deadband,
        "official_success": within_deadband,
        "practical_success": rotation_deg < PRACTICAL_ROTATION_DEG
        and translation_relative < PRACTICAL_TRANSLATION_RELATIVE,
        "loose_success": rotation_deg < LOOSE_ROTATION_DEG
        and translation_relative < LOOSE_TRANSLATION_RELATIVE,
    }


def projected_pseudo_box(
    rvec: np.ndarray,
    tvec: np.ndarray,
    points_3d: np.ndarray,
    K: np.ndarray,  # noqa: N803
    dist: np.ndarray,
    *,
    image_size: tuple[int, int],
    pad_frac: float = 0.0,
) -> np.ndarray:
    """The ``[x1, y1, x2, y2]`` box of the projected keypoints, padded and clipped.

    Each side grows by ``pad_frac`` times the box extent (at least 1 px), then the box is
    clipped to the image.

    Args:
        rvec: Object-to-camera Rodrigues vector.
        tvec: Object-to-camera translation.
        points_3d: ``(K, 3)`` object keypoints.
        K: ``(3, 3)`` intrinsics.
        dist: Distortion coefficients.
        image_size: ``(width, height)`` in pixels.
        pad_frac: Padding as a fraction of the extent.

    Returns:
        ``(4,)`` float32 box.
    """
    projected, _ = cv2.projectPoints(
        points_3d,
        np.asarray(rvec, dtype=np.float64),
        np.asarray(tvec, dtype=np.float64),
        K,
        dist,
    )
    xy = projected.reshape(-1, 2)
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    extent = np.maximum(hi - lo, 1.0)
    width, height = image_size
    lo = np.maximum(lo - pad_frac * extent, 0.0)
    hi = np.minimum(hi + pad_frac * extent, np.asarray([width - 1.0, height - 1.0]))
    return np.asarray([lo[0], lo[1], hi[0], hi[1]], dtype=np.float32)


def pose_projected_crop(
    image: np.ndarray,
    box: np.ndarray,
    *,
    crop_size: int,
    crop_pad_factor: float,
    angle_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop a BGR image around a pose-projected box, rolled by ``angle_deg``.

    The square crop is ``doper.targets.bbox_to_center_size`` of the box with the model's
    ``crop_pad_factor``, warped by ``topdown_warp_matrix`` with a gray-114 border, converted to
    RGB and normalized like the training crops.

    Args:
        image: ``(H, W, 3)`` uint8 BGR image.
        box: ``[x1, y1, x2, y2]`` box, for example ``projected_pseudo_box``.
        crop_size: Output side in pixels.
        crop_pad_factor: Square padding factor of the box.
        angle_deg: In-plane crop roll in degrees.

    Returns:
        The ``(3, crop_size, crop_size)`` float32 crop and the ``(2, 3)`` image-to-crop warp.
    """
    from ..dataset import normalize_image
    from ..targets import bbox_to_center_size, topdown_warp_matrix

    x1, y1, x2, y2 = box
    center, src_size = bbox_to_center_size(
        (float(x1), float(y1), float(x2 - x1), float(y2 - y1)), crop_pad_factor
    )
    warp = topdown_warp_matrix(center, src_size, crop_size, float(angle_deg))
    crop = cv2.warpAffine(
        image,
        warp,
        (crop_size, crop_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=CROP_BORDER_GREY,
    )
    return normalize_image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), warp
