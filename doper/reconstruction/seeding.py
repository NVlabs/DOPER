# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Camera seeding strategies of a NuRec reconstruction and the shared COLMAP path.

A ``SeedStrategy`` says how the capture cameras enter COLMAP. ``RobotHandEyeSeed``
(``seed_mode: robot_handeye``, mode (a), the reference) seeds every image with its robot camera
pose, ``cam_from_world = inv(T_base_mount @ T_mount_camera)`` in the robot base frame
(meters), and the capture intrinsics scaled to the normalized image size. ``ColmapSfmSeed``
(``seed_mode: colmap_sfm``, mode (b)) gives COLMAP no poses: incremental SfM picks the world
frame and the scale, so the reconstruction is not metric. ``DoperPnpSeed`` (``seed_mode:
doper_pnp``, mode (c), experimental) seeds the images
whose DOPER PnP pose passes its gates with that pose: the world is the CAD object frame, in
meters; ``image_registrator`` then adds the images it left out.

``reconstruct`` runs either: features with the capture's intrinsics fixed, exhaustive
matching, then the mapper (unseeded) or ``refine_seeded`` (seeded: triangulate the seed poses,
bundle-adjust, and retriangulate and bundle-adjust twice more), the text model, the COLMAP
quality gate and ``reconstruction.json``. A seeded model is not transformed: ``anchor_scale``
records the similarity from the seed camera centers (meters) to the refined ones (scene units),
whose scale is ``scene_units_per_metre``. Mode (c) also checks that scale against the CAD keypoints
(``cad_scale_check``).

Images are bound by name, never by index: COLMAP names are the normalized frame names, and the
frame map binds each to its capture image.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from ..bop.sources import canonical_sha256, sha256_file
from ..nurec.capture import (
    FrameMap,
    ImageSet,
    PinholeCamera,
    RobotReference,
    reference_camera_poses,
)
from ..recipes.nurec_schema import DoperSeedGates
from . import colmap_io
from .colmap_runner import ColmapCli, select_largest_model
from .geometry import (
    CameraPose,
    Observation,
    camera_centre,
    invert,
    project_pinhole,
    triangulate_points,
    umeyama_sim3,
)  # fmt: skip
from .metadata import (
    METADATA_NAME,
    CameraRecord,
    ColmapRecord,
    ImageRecord,
    ReconstructionMetadata,
    ScaleAnchor,
    ScaleCheck,
)

IMAGES_DIR = "images"
DATABASE_NAME = "database.db"
SPARSE_DIR = "sparse"
TEXT_DIR = "sparse-text"
SEED_DIR = "seed"
SEED_BINDINGS_NAME = "seed-bindings.json"
REFINE_DIR = "refine"
REFINED_MODEL = "0"
REFINE_PASSES = 3
NOT_REGISTERED = "not registered"
INTRINSICS_RTOL = 1e-9
ANCHOR_METHOD = "sim3_seed_to_refined_centres"
MM_PER_METRE = 1000.0
REGISTERED_DIR = "registered"
# Scene evidence (mode (c)): an image with fewer triangulated observations than this fraction of
# the median over the seeded images is un-seeded (seeding alone cannot see a symmetry-flipped pose).
SCENE_MIN_OBSERVATION_FRACTION = 0.25
SCALE_CHECK_METHOD = "sim3_cad_keypoints_to_retriangulated"
# Largest relative difference between the CAD-keypoint scale and the anchor (mode (c)).
SCALE_CHECK_TOLERANCE = 0.01
# Fewest seeded images: the scale anchor fits a similarity to at least three camera centers.
MIN_SEEDED_IMAGES = 3


@dataclass(frozen=True)
class QualityGate:
    """The COLMAP quality gate of the final model.

    ``check`` refuses a model with fewer than ``min_registered_images`` registered images, a
    largest-model fraction (registered / total) below ``min_model_fraction`` or no 3D point,
    and warns for a fraction below ``warn_model_fraction`` or a median reprojection error above
    ``warn_median_reprojection_px``.

    Attributes:
        min_registered_images: Fewest registered images.
        min_model_fraction: Smallest registered fraction of the images.
        warn_model_fraction: Registered fraction below which a warning is recorded.
        warn_median_reprojection_px: Median reprojection error (pixels) above which a warning
            is recorded.
    """

    min_registered_images: int = 20
    min_model_fraction: float = 0.40
    warn_model_fraction: float = 0.70
    warn_median_reprojection_px: float = 2.0

    def check(self, metrics: Mapping[str, Any]) -> tuple[str, ...]:
        """Apply the gate to ``colmap_io.model_metrics`` of the final model.

        Returns:
            The warning codes (``COLMAP_LOW_REGISTRATION``, ``COLMAP_HIGH_REPROJECTION``).

        Raises:
            ValueError: ``COLMAP_TOO_FEW_IMAGES``, ``COLMAP_MODEL_TOO_SMALL`` or
                ``COLMAP_NO_POINTS``.
        """
        registered, total = metrics["registered_images"], metrics["total_images"]
        fraction = metrics["largest_model_fraction"]
        if registered < self.min_registered_images:
            raise ValueError(f"COLMAP_TOO_FEW_IMAGES: the model registers {registered} of "
                             f"{total} images, fewer than "
                             f"{self.min_registered_images}")  # fmt: skip
        if fraction < self.min_model_fraction:
            raise ValueError(f"COLMAP_MODEL_TOO_SMALL: the model registers {registered} of "
                             f"{total} images, a fraction below "
                             f"{self.min_model_fraction}")  # fmt: skip
        if not metrics["points"]:
            raise ValueError("COLMAP_NO_POINTS: the model has no 3D points")
        warnings = []
        if fraction < self.warn_model_fraction:
            warnings.append("COLMAP_LOW_REGISTRATION")
        if metrics["median_reprojection_px"] > self.warn_median_reprojection_px:
            warnings.append("COLMAP_HIGH_REPROJECTION")
        return tuple(warnings)


DEFAULT_GATE = QualityGate()


@dataclass(frozen=True)
class KeypointObservations:
    """The DOPER keypoints of one image.

    Attributes:
        uv: ``(K, 2)`` keypoints in normalized-frame pixels (the seed camera's).
        confident: ``(K,)`` mask of the keypoints scored at least the score threshold.
    """

    uv: NDArray[np.float64]
    confident: NDArray[np.bool_]


@dataclass(frozen=True)
class SeedResult:
    """Seed poses of the normalized images.

    Attributes:
        camera: The normalized images' camera (fixed in COLMAP).
        cam_from_world: Normalized image name to its 4x4 ``cam_from_world`` (OpenCV camera
            axes, seed world frame, meters), in capture order: the seeded images only.
        sources: Normalized image name to its capture image name.
        rejected: Normalized image name to why the strategy left it out of the seed.
        observations: Normalized image name to its keypoints, for the seeded images of a
            keypoint strategy (mode (c)).
    """

    camera: PinholeCamera
    cam_from_world: Mapping[str, NDArray[np.float64]]
    sources: Mapping[str, str]
    rejected: Mapping[str, str] = field(default_factory=dict)
    observations: Mapping[str, KeypointObservations] = field(default_factory=dict)

    def T_world_camera(self, name: str) -> NDArray[np.float64]:  # noqa: N802 - frame-pair name
        """The camera pose (camera into the seed world) of one normalized image."""
        return invert(self.cam_from_world[name])


class SeedStrategy(ABC):
    """How the capture cameras enter COLMAP (the recipe's ``seed_mode``).

    Attributes:
        name: The ``seed_mode``.
        world_frame: The reconstruction's world frame (``reconstruction.json``).
        reference_path: Whether this is the recommended reconstruction path (``robot_handeye``).
        register_leftovers: Whether ``image_registrator`` adds the images the seed left out
            (``refine_checked``).
        scene_evidence_px: The scene-evidence check's median reprojection limit (pixels) after
            the first triangulation (``scene_evidence``), or None for no check.
    """

    name: ClassVar[str]
    world_frame: ClassVar[str]
    reference_path: ClassVar[bool] = False
    register_leftovers: bool = False
    scene_evidence_px: float | None = None

    @abstractmethod
    def seed(
        self, image_set: ImageSet | None, frames: FrameMap, robot: RobotReference | None
    ) -> SeedResult | None:
        """The seed poses of the normalized images, or None for an unseeded reconstruction.

        Raises:
            ValueError: When the capture lacks what the strategy needs.
        """

    def check_scale(
        self, seed: SeedResult, model: colmap_io.ColmapModel, anchor: ScaleAnchor
    ) -> ScaleCheck | None:
        """An independent check of the anchor's scale, or None (the default: no check).

        Raises:
            ValueError: When the check fails.
        """
        return None


def _normalized_camera(image_set: ImageSet, frames: FrameMap) -> PinholeCamera:
    """The capture camera at the normalized size, which must be the frame map's camera.

    Raises:
        ValueError: When the frame map's camera is not the capture camera at its size.
    """
    camera = image_set.camera.resized(frames.camera.width, frames.camera.height)
    given = [frames.camera.fx, frames.camera.fy, frames.camera.cx, frames.camera.cy]
    if not np.allclose([camera.fx, camera.fy, camera.cx, camera.cy], given,
                       rtol=INTRINSICS_RTOL, atol=0.0):  # fmt: skip
        raise ValueError(f"the frame map's camera {frames.camera} is not the capture camera "
                         f"{image_set.camera} at its size")  # fmt: skip
    return camera


class ColmapSfmSeed(SeedStrategy):
    """Mode (b): no seed; COLMAP's incremental SfM chooses the frame and scale."""

    name: ClassVar[str] = "colmap_sfm"
    world_frame: ClassVar[str] = "arbitrary"
    length_unit: ClassVar[str] = "arbitrary"
    scene_units_per_metre: ClassVar[None] = None

    def seed(
        self, image_set: ImageSet | None, frames: FrameMap, robot: RobotReference | None
    ) -> None:
        """No seed poses: the mapper registers every image."""
        return None


class RobotHandEyeSeed(SeedStrategy):
    """Mode (a): robot kinematics and the hand-eye calibration seed every image.

    ``cam_from_world_i = inv(T_base_mount_i @ T_mount_camera)``: the world is the robot base
    frame, in meters. The intrinsics are the capture session's scaled to the normalized image
    size.
    """

    name: ClassVar[str] = "robot_handeye"
    world_frame: ClassVar[str] = "robot_base"
    reference_path: ClassVar[bool] = True

    def seed(
        self, image_set: ImageSet | None, frames: FrameMap, robot: RobotReference | None
    ) -> SeedResult:
        """Seed each normalized frame with its robot camera pose.

        Raises:
            ValueError: Without an image set or a robot reference, when the frame map's camera
                is not the capture camera at the normalized size, or for a frame without a
                robot pose.
        """
        if robot is None or image_set is None:
            raise ValueError(f"seed_mode {self.name} needs the capture bundle's robot reference "
                             "(robot.jsonl and handeye.json)")  # fmt: skip
        camera = _normalized_camera(image_set, frames)
        poses = reference_camera_poses(robot)
        missing = [frame.source for frame in frames.frames if frame.source not in poses]
        if missing:
            raise ValueError(f"the robot reference has no pose for capture images {missing}")
        return SeedResult(
            camera=camera,
            cam_from_world={f.name: invert(poses[f.source]) for f in frames.frames},
            sources={frame.name: frame.source for frame in frames.frames},
        )


Infer = Callable[..., Sequence[Any]]


def _pose_matrix(rvec: Any, tvec: Any) -> NDArray[np.float64]:  # noqa: ANN401
    import cv2

    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return pose


def _triangulate_keypoints(
    poses: Mapping[str, NDArray[np.float64]],
    observations: Mapping[str, KeypointObservations],
    k: NDArray[np.float64],
) -> tuple[list[int], NDArray[np.float64]]:
    """Triangulate every keypoint two or more images saw confidently, the cameras fixed.

    Returns:
        The keypoint indices and their points ``(n, 3)`` in the cameras' world.
    """
    names = list(poses)
    count = len(next(iter(observations.values())).confident) if observations else 0
    indices, items = [], []
    for j in range(count):
        seen = [i for i, name in enumerate(names) if observations[name].confident[j]]
        if len(seen) < 2:
            continue
        for i in seen:
            uv = observations[names[i]].uv[j]
            items.append(Observation(len(indices), i, (float(uv[0]), float(uv[1]))))
        indices.append(j)
    if not indices:
        return [], np.empty((0, 3))
    cameras = [CameraPose(poses[name]) for name in names]
    return indices, triangulate_points(items, cameras, k)


def _view_medians(
    poses: Mapping[str, NDArray[np.float64]],
    observations: Mapping[str, KeypointObservations],
    k: NDArray[np.float64],
) -> dict[str, float]:
    """Per image, the median residual of its confident keypoints against the triangulation."""
    indices, points = _triangulate_keypoints(poses, observations, k)
    medians = {}
    for name, pose in poses.items():
        seen = [i for i, j in enumerate(indices) if observations[name].confident[j]]
        if not seen:
            medians[name] = 0.0
            continue
        uv = observations[name].uv[[indices[i] for i in seen]]
        residuals = np.linalg.norm(project_pinhole(points[seen], pose, k) - uv, axis=1)
        medians[name] = float(np.median(residuals))
    return medians


def consistency_gate(
    poses: Mapping[str, NDArray[np.float64]],
    observations: Mapping[str, KeypointObservations],
    k: NDArray[np.float64],
    max_px: float,
) -> dict[str, str]:
    """The cross-view consistency gate of the DOPER poses of a static object.

    Triangulates every confident keypoint from all kept images with the cameras fixed at their
    DOPER poses and drops the image with the largest median residual until every kept image's
    median is at most ``max_px``: a pose whose rays miss the keypoints the other views agree on
    (for example one rotated about a symmetry axis while its keypoints are right) is dropped.
    Limitation: keypoints relabeled by the object's symmetry that one rigid pose explains
    (DOPER confusing the two sides) are self-consistent in the CAD frame and pass; only the
    scene's feature geometry contradicts such a flip (``scene_evidence``).

    Args:
        poses: Image name to its DOPER ``cam_from_world``.
        observations: Image name to its keypoints, in the pixels of ``k``.
        k: The 3x3 intrinsics of the keypoints.
        max_px: Largest median residual, pixels.

    Returns:
        The dropped images, name to reason, in drop order.
    """
    kept = dict(poses)
    rejected: dict[str, str] = {}
    while len(kept) >= 2:
        medians = _view_medians(kept, observations, k)
        worst = max(medians, key=lambda name: medians[name])
        if medians[worst] <= max_px:
            break
        rejected[worst] = (f"cross-view consistency: median keypoint residual "
                           f"{medians[worst]:.3g} px > {max_px} px")  # fmt: skip
        del kept[worst]
    return rejected


def cad_scale_check(
    keypoints_3d: NDArray[np.float64],
    seed: SeedResult,
    model: colmap_io.ColmapModel,
    anchor_scene_units_per_metre: float,
    tolerance: float = SCALE_CHECK_TOLERANCE,
) -> ScaleCheck:
    """Check the anchor's scale against the CAD keypoints (mode (c)).

    The seeded images' DOPER keypoints are triangulated with the refined cameras (scene units)
    and Umeyama-fitted from the CAD keypoints (meters), the anchor's direction: the fit's
    scale is a second ``scene_units_per_metre``.

    Args:
        keypoints_3d: ``(K, 3)`` CAD keypoints, meters.
        seed: The seed, with the keypoint observations of its images.
        model: The refined model.
        anchor_scene_units_per_metre: The anchor's scale.
        tolerance: Largest relative difference accepted.

    Returns:
        The check.

    Raises:
        ValueError: For fewer than three keypoints seen by two refined seeded images, or a
            relative difference above ``tolerance`` (the message gives both scales).
    """
    refined = {image.name: colmap_io.cam_from_world(image) for image in model.images.values()}
    poses = {name: refined[name] for name in seed.cam_from_world
             if name in refined and name in seed.observations}  # fmt: skip
    indices, points = _triangulate_keypoints(poses, seed.observations, seed.camera.matrix)
    if len(indices) < 3:
        raise ValueError(f"the CAD-keypoint scale check needs at least 3 keypoints seen by two "
                         f"refined seeded images, got {len(indices)}")  # fmt: skip
    cad = np.asarray(keypoints_3d, dtype=np.float64)[indices]
    fit = umeyama_sim3(cad, points)
    residuals = np.linalg.norm(fit.apply(cad) - points, axis=1)
    rms = float(np.sqrt(np.mean(residuals**2)))
    difference = abs(fit.scale - anchor_scene_units_per_metre) / anchor_scene_units_per_metre
    if difference > tolerance:
        raise ValueError(
            f"the CAD-keypoint scale {fit.scale:.6g} and the seed-center anchor "
            f"{anchor_scene_units_per_metre:.6g} scene units per meter differ by "
            f"{difference:.3%}, more than {tolerance:.3%}: the DOPER seed and the refined "
            "reconstruction disagree"
        )
    return ScaleCheck(
        method=SCALE_CHECK_METHOD,
        scene_units_per_metre=fit.scale,
        anchor_scene_units_per_metre=float(anchor_scene_units_per_metre),
        relative_difference=float(difference),
        tolerance=float(tolerance),
        rms_scene=rms,
        rms_mm=rms / fit.scale * MM_PER_METRE,
        n=len(indices),
    )


class DoperPnpSeed(SeedStrategy):
    """Mode (c), experimental: DOPER poses seed COLMAP.

    For a static object and a moving camera the world is the CAD object frame (meters), and
    DOPER's object pose ``P_i = T_camera_object`` is the image's ``cam_from_world``: no
    inverse. Steps: predict each capture image (``infer_image`` at the capture resolution with
    the session K, one detection), gate it (``DoperSeedGates``: detection score, confident
    keypoints, ``solve_pnp`` with a positive depth and a small median reprojection error), drop
    the poses that disagree across views (``consistency_gate``), and seed the rest.
    ``reconstruct`` then refines with ``refine_checked``: after the first triangulation,
    ``scene_evidence`` (limit ``max_consistency_px``) un-seeds the images COLMAP's features
    contradict (a symmetry flip the keypoints cannot see), and ``image_registrator`` adds every
    image left out before the bundle adjustment passes. The seed
    camera is the session K at the normalized size COLMAP runs on (a pose does not depend on
    the image size; only K scales), and the keypoints are kept in normalized pixels for
    ``cad_scale_check``.

    Args:
        model: The DOPER model (``doper.infer.load_model``).
        keypoints_3d: ``(K, 3)`` CAD keypoints, meters, in the model's order.
        gates: The gates.
        cfg: The config ``load_model`` filled.
        device: Torch device.
        infer: ``infer_image`` or a stand-in with its signature (default
            ``doper.infer.infer_image``).
    """

    name: ClassVar[str] = "doper_pnp"
    world_frame: ClassVar[str] = "object_cad"
    reference_path: ClassVar[bool] = False

    def __init__(
        self,
        model: Any,  # noqa: ANN401 - a DoperNet; torch stays out of this module's imports
        keypoints_3d: Any,  # noqa: ANN401
        gates: DoperSeedGates | None = None,
        *,
        cfg: Any = None,  # noqa: ANN401
        device: str = "cpu",
        infer: Infer | None = None,
    ) -> None:
        """Keep the model, the keypoints and the gates.

        Raises:
            ValueError: For keypoints that are not a ``(K, 3)`` array of at least
                ``min_keypoints`` finite points.
        """
        self.gates = DoperSeedGates() if gates is None else gates
        points = np.asarray(keypoints_3d, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError(f"keypoints_3d must be a finite (K, 3) array, got {points.shape}")
        if len(points) < self.gates.min_keypoints:
            raise ValueError(f"keypoints_3d has {len(points)} points, fewer than min_keypoints "
                             f"{self.gates.min_keypoints}")  # fmt: skip
        self.keypoints_3d = points
        self.model, self.cfg, self.device, self.infer = model, cfg, device, infer
        self.register_leftovers = self.gates.register_leftovers
        self.scene_evidence_px = self.gates.max_consistency_px
        self._seeds: dict[tuple[ImageSet, FrameMap], SeedResult] = {}

    def _predict(self, image: NDArray[Any]) -> Sequence[Any]:
        infer = self.infer
        if infer is None:
            from ..infer import infer_image as infer
        return infer(self.model, image, self.cfg, self.device, score_thr=self.gates.score_thr,
                     max_det=1, keypoints_3d=self.keypoints_3d)  # fmt: skip

    def _gate(
        self, detections: Sequence[Any], k: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64] | None, str | None, KeypointObservations | None]:
        """One image's pose, or the reason it is left out, and its keypoints."""
        from ..infer import solve_pnp

        gates = self.gates
        if not detections:
            return None, "no detection", None
        detection = max(detections, key=lambda d: float(d.score))
        if float(detection.score) < gates.score_thr:
            return None, f"detection score {float(detection.score):.3g} < {gates.score_thr}", None
        uv = np.asarray(detection.keypoints, dtype=np.float64)
        confident = np.asarray(detection.keypoint_scores) >= gates.score_thr
        observed = KeypointObservations(uv, confident)
        count = int(confident.sum())
        if count < gates.min_keypoints:
            return None, (f"{count} keypoints with score >= {gates.score_thr}, fewer than "
                          f"{gates.min_keypoints}"), observed  # fmt: skip
        solve_pnp(detection, self.keypoints_3d, k, None, score_thr=gates.score_thr)
        if detection.rvec is None or detection.tvec is None:
            return None, "PnP failed", observed
        pose = _pose_matrix(detection.rvec, detection.tvec)
        points = self.keypoints_3d[confident]
        depth = (points @ pose[:3, :3].T + pose[:3, 3])[:, 2]
        if not np.isfinite(pose).all() or not (depth > 0.0).all():
            return None, "PnP pose puts keypoints behind the camera", observed
        error = float(
            np.median(np.linalg.norm(project_pinhole(points, pose, k) - uv[confident], axis=1))
        )
        if error > gates.max_reproj_px:
            return None, (f"PnP median reprojection {error:.3g} px > {gates.max_reproj_px} "
                          "px"), observed  # fmt: skip
        return pose, None, observed

    def seed(
        self, image_set: ImageSet | None, frames: FrameMap, robot: RobotReference | None
    ) -> SeedResult:
        """Seed the normalized frames whose DOPER pose passes the gates (computed once).

        Raises:
            ValueError: Without the capture images, for a frame map whose camera is not the
                capture camera at its size, an unreadable image, or fewer than three seeded
                images.
        """
        if image_set is None:
            raise ValueError(f"seed_mode {self.name} needs the capture images (DOPER runs on "
                             "them)")  # fmt: skip
        key = (image_set, frames)
        if key not in self._seeds:
            self._seeds[key] = self._seed(image_set, frames)
        return self._seeds[key]

    def _seed(self, image_set: ImageSet, frames: FrameMap) -> SeedResult:
        import cv2

        camera, k = _normalized_camera(image_set, frames), image_set.camera.matrix
        poses: dict[str, NDArray[np.float64]] = {}
        observed: dict[str, KeypointObservations] = {}
        rejected: dict[str, str] = {}
        for frame in frames.frames:
            image = cv2.imread(str(image_set.image_path(frame.source)), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"cannot read capture image {image_set.image_path(frame.source)}")
            pose, reason, keypoints = self._gate(self._predict(image), k)
            if pose is None or keypoints is None:
                rejected[frame.name] = str(reason)
                continue
            poses[frame.name], observed[frame.name] = pose, keypoints
        dropped = consistency_gate(poses, observed, k, self.gates.max_consistency_px)
        rejected.update(dropped)
        kept = {name: pose for name, pose in poses.items() if name not in dropped}
        if len(kept) < MIN_SEEDED_IMAGES:
            raise ValueError(f"seed_mode {self.name} seeded {len(kept)} of {len(frames.frames)} "
                             f"images, fewer than {MIN_SEEDED_IMAGES}; left out: "
                             f"{rejected}")  # fmt: skip
        scale = np.array([camera.width / image_set.camera.width,
                          camera.height / image_set.camera.height])  # fmt: skip
        return SeedResult(
            camera=camera,
            cam_from_world=kept,
            sources={frame.name: frame.source for frame in frames.frames},
            rejected={f.name: rejected[f.name] for f in frames.frames if f.name in rejected},
            observations={
                name: KeypointObservations(observed[name].uv * scale, observed[name].confident)
                for name in kept
            },  # fmt: skip
        )

    def check_scale(
        self, seed: SeedResult, model: colmap_io.ColmapModel, anchor: ScaleAnchor
    ) -> ScaleCheck:
        """``cad_scale_check`` of the anchor (1% tolerance).

        Raises:
            ValueError: When the CAD-keypoint scale and the anchor disagree.
        """
        return cad_scale_check(self.keypoints_3d, seed, model, anchor.scene_units_per_metre)


STRATEGIES: dict[str, type[SeedStrategy]] = {
    ColmapSfmSeed.name: ColmapSfmSeed,
    RobotHandEyeSeed.name: RobotHandEyeSeed,
    DoperPnpSeed.name: DoperPnpSeed,
}


def database_image_ids(db: Path) -> dict[str, int]:
    """Image name to image id of a COLMAP database (its ``images`` table)."""
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = connection.execute("SELECT image_id, name FROM images").fetchall()
    finally:
        connection.close()
    return {str(name): int(image_id) for image_id, name in rows}


def link_images(frames: FrameMap, frames_dir: Path, out: Path) -> None:
    """Put exactly the frame map's normalized images into ``out`` (hard links, else copies).

    COLMAP reads every file of its image directory, so the frame map's ``frames.json`` and
    any other file stay out.
    """
    out.mkdir(parents=True)
    for frame in frames.frames:
        try:
            os.link(frames_dir / frame.name, out / frame.name)
        except OSError:
            shutil.copyfile(frames_dir / frame.name, out / frame.name)


def text_model_id(directory: Path) -> str:
    """The identity of a text model: canonical SHA-256 of its file name to SHA-256."""
    files = sorted(p for p in directory.iterdir() if p.is_file())
    return canonical_sha256({p.name: sha256_file(p) for p in files})


def _camera_record(model: colmap_io.ColmapModel, given: PinholeCamera) -> CameraRecord:
    """The model's one camera; ``refined`` when its pinhole part or distortion moved."""
    if len(model.cameras) != 1:
        raise ValueError(f"expected one shared camera, the model has {len(model.cameras)}")
    camera = next(iter(model.cameras.values()))
    expected = [given.fx, given.fy, given.cx, given.cy] + [0.0] * (len(camera.params) - 4)
    same = len(camera.params) >= 4 and np.allclose(camera.params, expected,
                                                   rtol=INTRINSICS_RTOL, atol=0.0)  # fmt: skip
    refined = not same
    return CameraRecord(camera.model, camera.width, camera.height, camera.params,
                        given.rectified, "capture session", bool(refined))  # fmt: skip


def _image_records(
    frames: FrameMap,
    model: colmap_io.ColmapModel,
    ids: Mapping[str, int],
    seed: SeedResult | None,
) -> tuple[ImageRecord, ...]:
    registered = {image.name for image in model.images.values()}
    seeded = {} if seed is None else seed.cam_from_world
    rejected = {} if seed is None else seed.rejected

    def registered_by(name: str) -> str | None:
        if name not in registered:
            return None
        if seed is None:
            return "mapper"
        return "seed" if name in seeded else "image_registrator"

    return tuple(
        ImageRecord(
            name=frame.name,
            source_name=frame.source,
            sha256=frame.sha256,
            source_sha256=frame.source_sha256,
            colmap_image_id=ids.get(frame.name),
            seeded=frame.name in seeded,
            seed_cam_from_world=seeded[frame.name].tolist() if frame.name in seeded else None,
            registered=frame.name in registered,
            excluded_reason=None if frame.name in registered else NOT_REGISTERED,
            seed_rejected_reason=rejected.get(frame.name),
            registered_by=registered_by(frame.name),
        )
        for frame in frames.frames
    )


def seed_images(seed: SeedResult, ids: Mapping[str, int]) -> list[colmap_io.SeedImage]:
    """The seed's images bound to their database ids by name, in database id order.

    Raises:
        ValueError: For a seeded image the database does not hold.
    """
    missing = [name for name in seed.cam_from_world if name not in ids]
    if missing:
        raise ValueError(f"the COLMAP database has no image {missing}")
    images = [colmap_io.SeedImage(ids[name], name, pose) for name, pose in
              seed.cam_from_world.items()]  # fmt: skip
    return sorted(images, key=lambda image: image.image_id)


def seed_bindings(seed: SeedResult, ids: Mapping[str, int]) -> list[dict[str, Any]]:
    """Per seeded image: database id, normalized and capture names, ``T_world_camera``."""
    return [
        {
            "image_id": image.image_id,
            "normalized_image": image.name,
            "source_image": seed.sources[image.name],
            "T_world_camera": seed.T_world_camera(image.name).tolist(),
        }
        for image in seed_images(seed, ids)
    ]


def write_seed_model(seed: SeedResult, ids: Mapping[str, int], out: Path) -> None:
    """Write the seed model (``out/seed``) and its bindings (``out/seed-bindings.json``)."""
    colmap_io.write_seed(out / SEED_DIR, seed.camera, seed_images(seed, ids))
    bindings = json.dumps(seed_bindings(seed, ids), indent=2, allow_nan=False)
    (out / SEED_BINDINGS_NAME).write_text(bindings + "\n")


def refine_seeded(
    runner: ColmapCli, db: Path, images: Path, seed: Path, out: Path, passes: int = REFINE_PASSES
) -> Path:
    """Refine a seed model: triangulate, bundle-adjust, then retriangulate and bundle-adjust.

    ``point_triangulator`` with the intrinsics fixed on the seed poses, ``bundle_adjuster`` (poses
    free, intrinsics fixed) into ``sparse/0``, then ``passes - 1`` rounds of retriangulating
    ``sparse/0`` and bundle-adjusting it again (initial pose errors can reject valid matches). The
    flags are the runner's.

    Args:
        runner: The COLMAP runner.
        db: Database with the features and matches.
        images: Image directory.
        seed: The seed model directory.
        out: Reconstruction directory: the triangulations go to ``out/refine/triangulated-N``,
            the refined model to ``out/sparse/0``.
        passes: Triangulate and bundle-adjust rounds (at least one).

    Returns:
        The refined model directory, ``out/sparse/0``.

    Raises:
        ValueError: For fewer than one pass.
    """
    if passes < 1:
        raise ValueError(f"refine_seeded needs at least one pass, got {passes}")
    model = out / SPARSE_DIR / REFINED_MODEL
    source = seed
    for index in range(1, passes + 1):
        triangulated = out / REFINE_DIR / f"triangulated-{index}"
        runner.triangulate(db, images, source, triangulated)
        runner.bundle_adjust(triangulated, model)
        source = model
    return model


def scene_evidence(model: colmap_io.ColmapModel, seed: SeedResult, max_px: float) -> dict[str, str]:
    """The seeded images COLMAP's own feature evidence contradicts (mode (c)).

    Keypoints cannot see a pose flipped by the object's symmetry (the flipped pose explains
    its relabeled keypoints exactly); the scene's features can. On the first triangulation of
    the seed, an image is rejected when it has fewer triangulated observations (2D points with
    a 3D point) than ``SCENE_MIN_OBSERVATION_FRACTION`` of the median over the seeded images,
    or a median reprojection error of those observations above ``max_px``.

    Args:
        model: The triangulated seed model (poses as seeded, intrinsics fixed).
        seed: The seed.
        max_px: Largest median reprojection error, pixels.

    Returns:
        Rejected image name to reason, in seed order.
    """
    camera = next(iter(model.cameras.values()))
    fx, fy, cx, cy = camera.params[:4]
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    by_name = {image.name: image for image in model.images.values()}
    evidence: dict[str, tuple[int, float]] = {}
    for name in seed.cam_from_world:
        image = by_name.get(name)
        if image is None:
            evidence[name] = (0, math.inf)
            continue
        seen = [(x, y, model.points3d[i].xyz) for x, y, i in image.points2d
                if i != -1 and i in model.points3d]  # fmt: skip
        if not seen:
            evidence[name] = (0, math.inf)
            continue
        points = np.array([xyz for _, _, xyz in seen], dtype=np.float64)
        uv = np.array([(x, y) for x, y, _ in seen], dtype=np.float64)
        residuals = np.linalg.norm(project_pinhole(points, colmap_io.cam_from_world(image), k) - uv,
                                   axis=1)  # fmt: skip
        evidence[name] = (len(seen), float(np.median(residuals)))
    median_count = float(np.median([count for count, _ in evidence.values()]))
    least = SCENE_MIN_OBSERVATION_FRACTION * median_count
    rejected = {}
    for name, (count, error) in evidence.items():
        if count < least:
            rejected[name] = (f"scene_evidence: {count} triangulated observations < "
                              f"{SCENE_MIN_OBSERVATION_FRACTION} x median "
                              f"{median_count:g}")  # fmt: skip
        elif error > max_px:
            rejected[name] = (f"scene_evidence: median reprojection {error:.3g} px > {max_px} "
                              "px")  # fmt: skip
    return rejected


def _unseed(seed: SeedResult, rejected: Mapping[str, str], order: Sequence[str]) -> SeedResult:
    """The seed without the rejected images; their reasons join the seed's, in frame order."""
    reasons = {**seed.rejected, **rejected}
    kept = {name: pose for name, pose in seed.cam_from_world.items() if name not in rejected}
    if len(kept) < MIN_SEEDED_IMAGES:
        raise ValueError(f"the scene-evidence check left {len(kept)} seeded images, fewer than "
                         f"{MIN_SEEDED_IMAGES}; left out: {reasons}")  # fmt: skip
    return replace(
        seed,
        cam_from_world=kept,
        rejected={name: reasons[name] for name in order if name in reasons},
        observations={n: o for n, o in seed.observations.items() if n in kept},
    )


def refine_checked(
    runner: ColmapCli,
    db: Path,
    images: Path,
    seed: SeedResult,
    ids: Mapping[str, int],
    out: Path,
    *,
    total_images: int,
    scene_evidence_px: float | None,
    register: bool,
    passes: int = REFINE_PASSES,
) -> SeedResult:
    """Mode (c)'s refinement: scene evidence, completion, then the passes of ``refine_seeded``.

    Writes the seed and triangulates it (``refine/triangulated-1``). With
    ``scene_evidence_px``, the triangulation goes to text (``refine/triangulated-1-text``) and
    ``scene_evidence`` un-seeds the images the scene contradicts: the seed is rewritten and
    triangulated again (``refine/triangulated-1-checked``). With ``register``, when the seed
    lacks images, ``image_registrator`` (intrinsics fixed) adds them (``refine/registered``).
    Then bundle adjustment into ``sparse/0`` and ``passes - 1`` rounds of retriangulating and
    bundle-adjusting it, as ``refine_seeded``.

    Returns:
        The seed as refined (without the images the scene evidence rejected).

    Raises:
        ValueError: For fewer than one pass or fewer than three images left seeded.
    """
    if passes < 1:
        raise ValueError(f"refine_checked needs at least one pass, got {passes}")
    write_seed_model(seed, ids, out)
    source = out / REFINE_DIR / "triangulated-1"
    runner.triangulate(db, images, out / SEED_DIR, source)
    if scene_evidence_px is not None:
        text = out / REFINE_DIR / "triangulated-1-text"
        runner.to_text(source, text)
        rejected = scene_evidence(colmap_io.read_text_model(text), seed, scene_evidence_px)
        if rejected:
            seed = _unseed(seed, rejected, list(seed.sources))
            write_seed_model(seed, ids, out)
            source = out / REFINE_DIR / "triangulated-1-checked"
            runner.triangulate(db, images, out / SEED_DIR, source)
    if register and len(seed.cam_from_world) < total_images:
        registered = out / REFINE_DIR / REGISTERED_DIR
        runner.register(db, source, registered, fixed_intrinsics=True)
        source = registered
    model = out / SPARSE_DIR / REFINED_MODEL
    runner.bundle_adjust(source, model)
    for index in range(2, passes + 1):
        triangulated = out / REFINE_DIR / f"triangulated-{index}"
        runner.triangulate(db, images, model, triangulated)
        runner.bundle_adjust(triangulated, model)
    return seed


def _centres(model: colmap_io.ColmapModel) -> dict[str, NDArray[np.float64]]:
    return {i.name: camera_centre(colmap_io.cam_from_world(i)) for i in model.images.values()}


def anchor_scale(seed: colmap_io.ColmapModel, refined: colmap_io.ColmapModel) -> ScaleAnchor:
    """The similarity from the seed camera centers (meters) to the refined ones (scene units).

    Umeyama over the images both models hold (by name, in the seed's order): ``refined ~
    scale * R @ seed + t``, so the scale is ``scene_units_per_metre``. Residuals are in scene
    units and, divided by the scale, in millimeters. Neither model is transformed.

    Raises:
        ValueError: For fewer than three shared images or a degenerate seed.
    """
    seed_centres, refined_centres = _centres(seed), _centres(refined)
    names = [name for name in seed_centres if name in refined_centres]
    if len(names) < 3:
        raise ValueError(f"a scale anchor needs at least 3 cameras in both models, got "
                         f"{len(names)}")  # fmt: skip
    source = np.array([seed_centres[name] for name in names])
    target = np.array([refined_centres[name] for name in names])
    fit = umeyama_sim3(source, target)
    residuals = np.linalg.norm(fit.apply(source) - target, axis=1)
    rms, largest = float(np.sqrt(np.mean(residuals**2))), float(residuals.max())
    tx, ty, tz = (float(v) for v in fit.translation)
    return ScaleAnchor(
        method=ANCHOR_METHOD,
        scene_units_per_metre=fit.scale,
        rotation=fit.rotation.tolist(),
        translation=(tx, ty, tz),
        rms_scene=rms,
        max_scene=largest,
        rms_mm=rms / fit.scale * MM_PER_METRE,
        max_mm=largest / fit.scale * MM_PER_METRE,
        n=len(names),
    )


def reconstruct(
    image_set: ImageSet | None,
    frames: FrameMap,
    strategy: SeedStrategy,
    runner: ColmapCli,
    out: Path,
    *,
    frames_dir: Path,
    robot: RobotReference | None = None,
    capture_flags: Mapping[str, Any],
    inputs: Mapping[str, str],
    gate: QualityGate = DEFAULT_GATE,
) -> ReconstructionMetadata:
    """Reconstruct the normalized capture with COLMAP, seeded or not, with its metadata.

    Writes ``out/images`` (the normalized frames), ``out/database.db``, the model in
    ``out/sparse/<n>``, ``out/sparse-text`` (the final model as text) and
    ``out/reconstruction.json``; a seeded run also writes ``out/seed``,
    ``out/seed-bindings.json`` and ``out/refine``. A strategy with a scene-evidence check or
    leftover registration (mode (c)) refines with ``refine_checked`` instead of
    ``refine_seeded``; its ``check_scale`` runs after the anchor.

    Args:
        image_set: The capture images (a seeded strategy needs them; None otherwise).
        frames: The frame map of the normalized capture (its camera is fixed in COLMAP).
        strategy: The seeding strategy.
        runner: The COLMAP runner (its calls are recorded in the metadata).
        out: Reconstruction directory; it must not exist.
        frames_dir: Directory of the normalized frames.
        robot: The capture's robot reference, when it has one.
        capture_flags: The capture's ``capture_flags``.
        inputs: Input name to SHA-256, recorded as given.
        gate: The quality gate of the final model.

    Returns:
        The metadata (also written to ``out/reconstruction.json``).

    Raises:
        FileExistsError: When ``out`` exists.
        ColmapError: For a failed COLMAP call or a COLMAP other than 3.13.
        ValueError: When the strategy cannot seed the capture, the gate refuses the model, the
            strategy's scale check fails, or the model has several cameras.
    """
    if out.exists():
        raise FileExistsError(f"{out} exists")
    seed = strategy.seed(image_set, frames, robot)
    version = runner.version()
    images, db = out / IMAGES_DIR, out / DATABASE_NAME
    link_images(frames, frames_dir, images)
    runner.extract(db, images, frames.camera)
    runner.match(db)
    ids = database_image_ids(db)
    if seed is None:
        runner.map(db, images, out / SPARSE_DIR, fixed_intrinsics=True)
    elif strategy.scene_evidence_px is None and not strategy.register_leftovers:
        write_seed_model(seed, ids, out)
        refine_seeded(runner, db, images, out / SEED_DIR, out)
    else:
        seed = refine_checked(runner, db, images, seed, ids, out,
                              total_images=len(frames.frames),
                              scene_evidence_px=strategy.scene_evidence_px,
                              register=strategy.register_leftovers)  # fmt: skip
    selected = select_largest_model(out / SPARSE_DIR)
    runner.to_text(selected, out / TEXT_DIR)
    model = colmap_io.read_text_model(out / TEXT_DIR)
    metrics = colmap_io.model_metrics(model, len(frames.frames))
    warnings = gate.check(metrics)
    anchor = check = None
    if seed is not None:
        anchor = anchor_scale(colmap_io.read_text_model(out / SEED_DIR), model)
        check = strategy.check_scale(seed, model, anchor)
    metadata = ReconstructionMetadata(
        strategy=strategy.name,
        world_frame=strategy.world_frame,
        length_unit="arbitrary" if anchor is None else "colmap",
        scene_units_per_metre=None if anchor is None else anchor.scene_units_per_metre,
        scale_anchor=anchor,
        camera=_camera_record(model, frames.camera),
        images=_image_records(frames, model, ids, seed),
        colmap=ColmapRecord(
            version=version,
            argv=[list(call.argv) for call in runner.calls],
            metrics=metrics,
            selected_model=selected.relative_to(out).as_posix(),
            warnings=list(warnings),
        ),
        inputs=dict(inputs),
        capture_flags=dict(capture_flags),
        reference_path=strategy.reference_path,
        reconstruction_id=text_model_id(out / TEXT_DIR),
        model_text=TEXT_DIR,
        scale_check=check,
    )
    metadata.write(out / METADATA_NAME)
    return metadata


def reconstruct_sfm(
    frames: FrameMap,
    frames_dir: Path,
    runner: ColmapCli,
    out: Path,
    *,
    capture_flags: Mapping[str, Any],
    inputs: Mapping[str, str],
    gate: QualityGate = DEFAULT_GATE,
) -> ReconstructionMetadata:
    """Mode (b): ``reconstruct`` with ``ColmapSfmSeed`` (unseeded SfM, not metric)."""
    return reconstruct(None, frames, ColmapSfmSeed(), runner, out, frames_dir=frames_dir,
                       capture_flags=capture_flags, inputs=inputs, gate=gate)  # fmt: skip


__all__ = [
    "DEFAULT_GATE",
    "STRATEGIES",
    "ColmapSfmSeed",
    "DoperPnpSeed",
    "DoperSeedGates",
    "KeypointObservations",
    "QualityGate",
    "RobotHandEyeSeed",
    "SeedResult",
    "SeedStrategy",
    "anchor_scale",
    "cad_scale_check",
    "consistency_gate",
    "database_image_ids",
    "link_images",
    "reconstruct",
    "reconstruct_sfm",
    "refine_seeded",
    "refine_checked",
    "scene_evidence",
    "seed_bindings",
    "seed_images",
    "text_model_id",
    "write_seed_model",
]
