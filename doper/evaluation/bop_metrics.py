# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BOP pose-error math: MSSD, MSPD and average recall for single-instance evaluation.

The two official BOP pose errors are computed against a ``models_eval`` mesh:

* MSSD - maximum symmetry-aware surface distance (mm), recall at thresholds of 5..50 % of the
  object diameter;
* MSPD - maximum symmetry-aware projection distance (px, normalized to a 640 px wide image),
  recall at thresholds of 5..50 px.

AR is the mean recall over the ten thresholds; a frame without an accepted detection counts as a
failure at every threshold. This mirrors ``bop_toolkit``'s ``eval_calc_errors`` and
``eval_calc_scores`` for the single-instance case without depending on it. A single continuous
symmetry axis is sampled at 36 rotations; combinations with discrete symmetries or multiple
continuous axes are rejected.

Two symmetry generators coexist on purpose. ``symmetry_transforms`` with ``mssd`` / ``mspd`` is
DOPER's own scorer (36 samples per continuous axis). The ``toolkit_*`` functions reproduce
``bop_toolkit_lib`` (``misc.get_symmetry_transformations`` with ``max_sym_disc_step=0.01`` and
``pose_error.mssd`` / ``mspd``), including its arithmetic order, offsets and
discrete-plus-continuous compositions. The BOP validation AR protocols in ``doper.bop.ar`` were
scored with the toolkit, so they use those.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence

import numpy as np

MSSD_FRACTIONS = tuple(0.05 * i for i in range(1, 11))  # of the object diameter
MSPD_THRESHOLDS_PX = tuple(5.0 * i for i in range(1, 11))  # at 640 px image width
TOOLKIT_MAX_SYM_DISC_STEP = 0.01  # fraction of the diameter, as the BOP validation scorers used

_PLY_SCALAR_FORMATS = {
    "float": "f",
    "float32": "f",
    "double": "d",
    "uchar": "B",
    "uint8": "B",
    "int": "i",
    "uint": "I",
    "short": "h",
    "ushort": "H",
    "char": "b",
}


def read_ply_vertices(path: str) -> np.ndarray:
    """Read vertex positions from an ASCII or binary little-endian PLY.

    Args:
        path: PLY file containing scalar vertex properties including x, y and z.

    Returns:
        An (N, 3) float64 array in source mesh units (millimeters for BOP).

    Raises:
        ValueError: The header is unterminated or the PLY format is unsupported.
        OSError: The file cannot be opened or read.
    """
    with open(path, "rb") as f:
        header: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: no end_header")
            header.append(line.decode("ascii", errors="replace").strip())
            if header[-1] == "end_header":
                break
        fmt = next(h.split()[1] for h in header if h.startswith("format"))
        n_vertices = int(next(h.split()[2] for h in header if h.startswith("element vertex")))
        # Vertex properties come between "element vertex" and the next "element".
        start = next(i for i, h in enumerate(header) if h.startswith("element vertex"))
        props = []
        for h in header[start + 1 :]:
            if h.startswith("element"):
                break
            if h.startswith("property"):
                _, ptype, name = h.split()[:3]
                props.append((ptype, name))
        names = [name for _, name in props]
        if fmt == "ascii":
            rows = [f.readline().split() for _ in range(n_vertices)]
            data = np.array(rows, dtype=np.float64)
        else:
            if fmt != "binary_little_endian":
                raise ValueError(f"{path}: unsupported PLY format {fmt}")
            record = "<" + "".join(_PLY_SCALAR_FORMATS[ptype] for ptype, _ in props)
            size = struct.calcsize(record)
            raw = f.read(size * n_vertices)
            data = np.array(
                [struct.unpack_from(record, raw, i * size) for i in range(n_vertices)],
                dtype=np.float64,
            )
    return data[:, [names.index("x"), names.index("y"), names.index("z")]]


def symmetry_transforms(model_info: dict, *, continuous_steps: int = 36) -> list[np.ndarray]:
    """Build identity and discrete or sampled continuous model symmetries.

    Args:
        model_info: One object entry from BOP models_info.json. Translations and
            continuous-axis offsets are in millimeters.
        continuous_steps: Positive number of evenly spaced rotations for a single
            continuous axis, including identity; 36 gives 10-degree spacing.

    Returns:
        A list of (4, 4) homogeneous model-frame transforms with translations
        in millimeters, beginning with identity.

    Raises:
        ValueError: Discrete and continuous symmetries are combined, or more
            than one continuous axis is specified.
    """
    discrete = model_info.get("symmetries_discrete", [])
    continuous = model_info.get("symmetries_continuous", [])
    if (discrete and continuous) or len(continuous) > 1:
        raise ValueError(
            "unsupported symmetry composition: use the BOP toolkit for models combining "
            "discrete and continuous symmetries or multiple continuous axes"
        )
    transforms = [np.eye(4)]
    for sym in discrete:
        transforms.append(np.asarray(sym, dtype=float).reshape(4, 4))
    for sym in continuous:
        axis = np.asarray(sym["axis"], dtype=float)
        axis /= np.linalg.norm(axis)
        offset = np.asarray(sym.get("offset", [0, 0, 0]), dtype=float)
        for k in range(1, continuous_steps):
            angle = 2 * np.pi * k / continuous_steps
            cross = np.array(
                [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
            )
            rot = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)
            transform = np.eye(4)
            transform[:3, :3] = rot
            transform[:3, 3] = offset - rot @ offset
            transforms.append(transform)
    return transforms


def _apply(rot: np.ndarray, trans: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Transform row-vector points with a rigid pose.

    Args:
        rot: (3, 3) rotation from the input frame to the output frame.
        trans: (3,) translation in the same units as pts.
        pts: (N, 3) points in the input frame.

    Returns:
        An (N, 3) array in the output frame, retaining the input length units.
    """
    return pts @ rot.T + trans


def mssd(
    R_est: np.ndarray,  # noqa: N803 - BOP toolkit argument name
    t_est: np.ndarray,
    R_gt: np.ndarray,  # noqa: N803 - BOP toolkit argument name
    t_gt: np.ndarray,
    pts: np.ndarray,
    syms: Sequence[np.ndarray],
) -> float:
    """Compute symmetry-aware maximum surface distance.

    Args:
        R_est: Estimated (3, 3) model-to-camera rotation.
        t_est: Estimated (3,) model-to-camera translation in millimeters.
        R_gt: Ground-truth (3, 3) model-to-camera rotation.
        t_gt: Ground-truth (3,) model-to-camera translation in millimeters.
        pts: Nonempty (N, 3) model-frame surface points in millimeters.
        syms: Nonempty sequence of (4, 4) model-frame symmetry transforms,
            including identity, with translations in millimeters.

    Returns:
        The minimum over symmetries of the maximum point distance, in millimeters.
    """
    est = _apply(R_est, t_est, pts)
    best = np.inf
    for sym in syms:
        gt = _apply(R_gt @ sym[:3, :3], R_gt @ sym[:3, 3] + t_gt, pts)
        best = min(best, float(np.linalg.norm(est - gt, axis=1).max()))
    return best


def _project(camera_matrix: np.ndarray, pts_cam: np.ndarray) -> np.ndarray:
    """Project camera-space points through camera intrinsics.

    Args:
        camera_matrix: (3, 3) camera intrinsic matrix in native image pixels.
        pts_cam: (N, 3) camera-frame points with nonzero depth; all coordinates
            must share the same length units.

    Returns:
        An (N, 2) array of image coordinates in native pixels.
    """
    projected = pts_cam @ camera_matrix.T
    return projected[:, :2] / projected[:, 2:3]


def mspd(
    R_est: np.ndarray,  # noqa: N803 - BOP toolkit argument name
    t_est: np.ndarray,
    R_gt: np.ndarray,  # noqa: N803 - BOP toolkit argument name
    t_gt: np.ndarray,
    K: np.ndarray,  # noqa: N803 - BOP toolkit argument name
    pts: np.ndarray,
    syms: Sequence[np.ndarray],
) -> float:
    """Compute symmetry-aware maximum projection distance.

    Args:
        R_est: Estimated (3, 3) model-to-camera rotation.
        t_est: Estimated (3,) model-to-camera translation in millimeters.
        R_gt: Ground-truth (3, 3) model-to-camera rotation.
        t_gt: Ground-truth (3,) model-to-camera translation in millimeters.
        K: (3, 3) camera intrinsic matrix in native image pixels.
        pts: Nonempty (N, 3) model-frame surface points in millimeters.
        syms: Nonempty sequence of (4, 4) model-frame symmetry transforms,
            including identity, with translations in millimeters.

    Returns:
        The minimum over symmetries of the maximum projected point distance
        in native pixels. The caller applies any image-width normalization.
    """
    est = _project(K, _apply(R_est, t_est, pts))
    best = np.inf
    for sym in syms:
        gt = _project(K, _apply(R_gt @ sym[:3, :3], R_gt @ sym[:3, 3] + t_gt, pts))
        best = min(best, float(np.linalg.norm(est - gt, axis=1).max()))
    return best


def _toolkit_axis_rotation(angle: float, axis: np.ndarray) -> np.ndarray:
    """Rotation about ``axis`` with ``bop_toolkit_lib.transform.rotation_matrix`` arithmetic."""
    sin_angle = math.sin(angle)
    cos_angle = math.cos(angle)
    direction = np.array(axis[:3], dtype=np.float64, copy=True)
    direction /= math.sqrt(np.dot(direction, direction))
    rotation = np.diag([cos_angle, cos_angle, cos_angle])
    rotation += np.outer(direction, direction) * (1.0 - cos_angle)
    direction *= sin_angle
    rotation += np.array(
        [
            [0.0, -direction[2], direction[1]],
            [direction[2], 0.0, -direction[0]],
            [-direction[1], direction[0], 0.0],
        ]
    )
    return rotation


def _homogeneous(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(translation).reshape(3)
    return transform


def toolkit_symmetry_transforms(
    model_info: dict, *, max_sym_disc_step: float = TOOLKIT_MAX_SYM_DISC_STEP
) -> list[np.ndarray]:
    """Build model symmetries exactly as ``bop_toolkit_lib.misc.get_symmetry_transformations``.

    Each continuous axis is discretized into ``ceil(pi / max_sym_disc_step)`` rotations about
    its ``offset`` point; every discrete symmetry (identity first) is composed with every
    continuous one, continuous applied last. Without continuous symmetries the discrete set is
    returned as is.

    Args:
        model_info: One object entry from BOP models_info.json; translations and offsets in
            millimeters. Continuous entries need ``axis`` and ``offset``, as in the toolkit.
        max_sym_disc_step: Largest travel between consecutive discretized rotations, as a
            fraction of the object diameter.

    Returns:
        (4, 4) homogeneous model-frame transforms in the toolkit's order.
    """
    discrete: list[tuple[np.ndarray, np.ndarray]] = [(np.eye(3), np.zeros((3, 1)))]
    for sym in model_info.get("symmetries_discrete", []):
        matrix = np.reshape(sym, (4, 4))
        discrete.append((matrix[:3, :3], matrix[:3, 3].reshape((3, 1))))
    continuous = []
    for sym in model_info.get("symmetries_continuous", []):
        axis = np.array(sym["axis"])
        offset = np.array(sym["offset"]).reshape((3, 1))
        steps = int(np.ceil(np.pi / max_sym_disc_step))
        step = 2.0 * np.pi / steps
        for i in range(steps):
            rotation = _toolkit_axis_rotation(i * step, axis)
            continuous.append((rotation, -rotation.dot(offset) + offset))
    if not continuous:
        return [_homogeneous(rotation, translation) for rotation, translation in discrete]
    return [
        _homogeneous(rot_c.dot(rot_d), rot_c.dot(trans_d) + trans_c)
        for rot_d, trans_d in discrete
        for rot_c, trans_c in continuous
    ]


def _toolkit_transform(
    pts: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """``bop_toolkit_lib.misc.transform_pts_Rt``: column-vector rigid transform."""
    return (rotation.dot(pts.T) + translation.reshape((3, 1))).T


def _toolkit_project(
    pts: np.ndarray, camera_matrix: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """``bop_toolkit_lib.misc.project_pts``: pinhole projection with a 3x4 matrix."""
    projection = camera_matrix.dot(np.hstack((rotation, translation)))
    homogeneous = np.hstack((pts, np.ones((pts.shape[0], 1))))
    image = projection.dot(homogeneous.T)
    image /= image[2, :]
    return image[:2, :].T


def _toolkit_symmetric_gt(
    R_gt: np.ndarray,  # noqa: N803 - BOP toolkit argument name
    t_gt: np.ndarray,
    sym: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    translation = np.ascontiguousarray(sym[:3, 3:4])
    return R_gt.dot(np.ascontiguousarray(sym[:3, :3])), R_gt.dot(translation) + t_gt


def toolkit_mssd(
    R_est: np.ndarray,  # noqa: N803 - BOP toolkit argument name
    t_est: np.ndarray,
    R_gt: np.ndarray,  # noqa: N803 - BOP toolkit argument name
    t_gt: np.ndarray,
    pts: np.ndarray,
    syms: Sequence[np.ndarray],
) -> float:
    """MSSD with ``bop_toolkit_lib.pose_error.mssd`` arithmetic.

    Args:
        R_est: Estimated (3, 3) model-to-camera rotation.
        t_est: Estimated model-to-camera translation (3 values) in millimeters.
        R_gt: Ground-truth (3, 3) model-to-camera rotation.
        t_gt: Ground-truth translation (3 values) in millimeters.
        pts: Nonempty (N, 3) model points in millimeters.
        syms: Transforms from ``toolkit_symmetry_transforms``.

    Returns:
        The minimum over symmetries of the maximum point distance, in millimeters.
    """
    t_gt = np.asarray(t_gt).reshape((3, 1))
    estimate = _toolkit_transform(pts, R_est, np.asarray(t_est).reshape((3, 1)))
    errors = []
    for sym in syms:
        rotation, translation = _toolkit_symmetric_gt(R_gt, t_gt, sym)
        truth = _toolkit_transform(pts, rotation, translation)
        errors.append(np.linalg.norm(estimate - truth, axis=1).max())
    return float(min(errors))


def toolkit_mspd(
    R_est: np.ndarray,  # noqa: N803 - BOP toolkit argument name
    t_est: np.ndarray,
    R_gt: np.ndarray,  # noqa: N803 - BOP toolkit argument name
    t_gt: np.ndarray,
    K: np.ndarray,  # noqa: N803 - BOP toolkit argument name
    pts: np.ndarray,
    syms: Sequence[np.ndarray],
) -> float:
    """MSPD with ``bop_toolkit_lib.pose_error.mspd`` arithmetic.

    Args:
        R_est: Estimated (3, 3) model-to-camera rotation.
        t_est: Estimated model-to-camera translation (3 values) in millimeters.
        R_gt: Ground-truth (3, 3) model-to-camera rotation.
        t_gt: Ground-truth translation (3 values) in millimeters.
        K: (3, 3) camera intrinsic matrix in native image pixels.
        pts: Nonempty (N, 3) model points in millimeters.
        syms: Transforms from ``toolkit_symmetry_transforms``.

    Returns:
        The minimum over symmetries of the maximum projected distance in native pixels; the
        caller applies any image-width normalization.
    """
    t_gt = np.asarray(t_gt).reshape((3, 1))
    estimate = _toolkit_project(pts, K, R_est, np.asarray(t_est).reshape((3, 1)))
    errors = []
    for sym in syms:
        rotation, translation = _toolkit_symmetric_gt(R_gt, t_gt, sym)
        truth = _toolkit_project(pts, K, rotation, translation)
        errors.append(np.linalg.norm(estimate - truth, axis=1).max())
    return float(min(errors))


def average_recall(errors: Sequence[float | None], thresholds: Sequence[float]) -> float:
    """Average the fraction of successful frames over error thresholds.

    Args:
        errors: Per-frame errors; None represents a missed detection and counts
            as a failure at every finite threshold.
        thresholds: Nonempty sequence of acceptance thresholds in the same units
            as errors. An error must be strictly below a threshold to pass.

    Returns:
        Mean recall in [0, 1], or NaN when errors is empty.
    """
    values = np.array([np.inf if e is None else e for e in errors], dtype=float)
    if len(values) == 0:
        return float("nan")
    return float(np.mean([(values < th).mean() for th in thresholds]))


def summarize(frames: list[dict], *, diameter_mm: float) -> dict[str, float]:
    """Aggregate recall, detection counts and median pose errors.

    Args:
        frames: Per-frame dictionaries containing mssd_mm and mspd_px640;
            use None for both errors when no pose was detected.
        diameter_mm: Object diameter in millimeters, used for MSSD thresholds
            from 5 percent through 50 percent of the diameter.

    Returns:
        A dictionary of frame/detection counts, detection rate, AR_MSSD,
        AR_MSPD, their mean, median errors (mm and pixels at 640-pixel width),
        and diameter_mm. Medians are NaN without detections; recalls are NaN
        for an empty population.
    """
    mssd_errors = [f["mssd_mm"] for f in frames]
    mspd_errors = [f["mspd_px640"] for f in frames]
    found = [f for f in frames if f["mssd_mm"] is not None]
    ar_mssd = average_recall(mssd_errors, [frac * diameter_mm for frac in MSSD_FRACTIONS])
    ar_mspd = average_recall(mspd_errors, MSPD_THRESHOLDS_PX)
    return {
        "frames": len(frames),
        "detections": len(found),
        "det_rate": len(found) / max(1, len(frames)),
        "ar_mssd": ar_mssd,
        "ar_mspd": ar_mspd,
        "ar_mean": 0.5 * (ar_mssd + ar_mspd),
        "mssd_median_mm": float(np.median([f["mssd_mm"] for f in found]))
        if found
        else float("nan"),
        "mspd_median_px640": (
            float(np.median([f["mspd_px640"] for f in found])) if found else float("nan")
        ),
        "diameter_mm": diameter_mm,
    }
