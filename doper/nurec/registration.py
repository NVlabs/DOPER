# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CAD-to-scene registration of a NuRec scene: keypoint seed, fixed-scale edge refinement, review.

The registration places the object's CAD model (meters) in the NuRec scene frame:
``scene = s * R @ p_cad + t`` with ``s = scene_units_per_metre``. ``T_nre_object`` stores
``R`` and ``t`` as a 4x4 ``[[R, t], [0, 0, 0, 1]]`` (the rotation unscaled) and the scale is
recorded beside it. The steps, in order:

1. ``fit_keypoint_similarity``: the DOPER keypoints of the capture views are triangulated with
   the scene cameras and Umeyama-fitted from the CAD keypoints (a similarity, with the
   reflection fix): the seed's rotation and translation.
2. ``robot_scale_audit``: the scale is fixed from the robot camera centers against the scene
   camera centers (a capture with a robot reference), else from the reconstruction's anchor.
3. ``refine_edges``: the six rigid parameters are refined against Canny distance fields of
   capture views at that fixed scale. CAD vertices near the seed silhouette's boundary are
   chosen once, at the seed, and their projections' distances to the nearest edge pixel
   (bilinear) are the residuals; alternate views are held out and reported.
4. ``restart_audit``: perturbed restarts of the refinement, a diagnostic of how well the edge
   cost pins the pose.

The result is a *candidate*: ``accept_registration`` with a reviewer's note turns it into the
``registration.json`` that labels read (``resolve_registration``); a capture bundle may also
carry a reviewed ``registration.json`` of its scene. Gaussian-to-CAD ICP is not used: the
Gaussian centers are not surfaces.

Frames and units: ``T_nre_cam`` maps OpenCV camera coordinates (x right, y down, z forward)
into the scene; pixels are capture-image pixels; scene lengths are scene units, CAD lengths
meters.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..reconstruction.geometry import (
    CameraPose,
    Observation,
    as_rigid,
    invert,
    project_pinhole,
    triangulate_points,
    umeyama_sim3,
)  # fmt: skip
from ..reconstruction.metadata import ScaleAnchor
from .capture import ImageSet, RobotReference, reference_camera_poses
from .scene import NuRecScene

if TYPE_CHECKING:
    from ..evaluation.relative_pose import FramePose

CANDIDATE_NAME = "candidate.json"
REGISTRATION_NAME = "registration.json"
CANDIDATE_STATUS = "candidate_requires_review"
ACCEPTED_STATUS = "accepted"
# A capture bundle's registration was reviewed by its authors.
REVIEWED_STATUS = "reviewed"
CONVENTION = "scene = scene_units_per_metre * R @ p_cad_m + t; T_nre_object = [[R, t], [0, 1]]"
SEED_METHOD = "sim3_cad_keypoints_to_triangulated_keypoints"
AUDIT_METHOD = "sim3_robot_camera_centres_to_scene_camera_centres"
EDGE_METHOD = (
    "keypoint-triangulated similarity seed; scale fixed; six rigid parameters refined against "
    "Canny distance fields of the fit views (boundary CAD vertices chosen at the seed)"
)
MIN_TRIANGULATION_VIEWS = 2
MIN_AUDIT_CAMERAS = 3
MM_PER_UNIT = 1000.0
P90 = 90
# The silhouette boundary is the mask minus its erosion by this square structuring element.
BOUNDARY_KERNEL = 3
DISTANCE_MASK = 5
REGISTRATION_KEYS = frozenset({"status", "method", "convention", "T_nre_object",
                               "scene_units_per_metre", "artifact_sha256", "reconstruction_id",
                               "reviewer_note", "provenance"})  # fmt: skip
REQUIRED_REGISTRATION_KEYS = ("status", "T_nre_object", "scene_units_per_metre", "artifact_sha256")
_SHA256_LENGTH = 64


class RegistrationNotAcceptedError(FileNotFoundError):
    """No accepted (or reviewed) registration exists for a stage that needs one."""


@dataclass(frozen=True)
class TriangleMesh:
    """A triangle mesh.

    Attributes:
        vertices: ``(v, 3)`` vertices, meters (object frame).
        faces: ``(f, 3)`` vertex indices.
    """

    vertices: NDArray[np.float64]
    faces: NDArray[np.int64]

    def __post_init__(self) -> None:
        """Check the shapes and indices.

        Raises:
            ValueError: For malformed arrays or a face naming a missing vertex.
        """
        vertices = np.array(self.vertices, dtype=np.float64)
        faces = np.array(self.faces, dtype=np.int64).reshape(-1, 3)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            raise ValueError("mesh vertices must be a finite (v, 3) array")
        if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("a mesh face names a vertex the mesh does not have")
        vertices.setflags(write=False)
        faces.setflags(write=False)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)


def read_obj_mesh(path: str | Path) -> TriangleMesh:
    """Read a Wavefront OBJ: its ``v`` vertices and ``f`` polygons fanned into triangles.

    Only positions and faces are read (``f a/b/c`` takes ``a``); materials, normals and texture
    coordinates are ignored.

    Raises:
        ValueError: For a face that names a missing vertex.
    """
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("v "):
            vertices.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            ids = [int(x.split("/")[0]) - 1 for x in line.split()[1:]]
            faces.extend([ids[0], ids[k], ids[k + 1]] for k in range(1, len(ids) - 1))
    try:
        return TriangleMesh(np.array(vertices).reshape(-1, 3), np.array(faces, dtype=np.int64))
    except ValueError as err:
        raise ValueError(f"{path}: {err}") from err


@dataclass(frozen=True)
class SimilarityFit:
    """The keypoint seed: ``scene ~ scale * rotation @ p_cad + translation``.

    Attributes:
        scale: Scene units per CAD meter.
        rotation: 3x3 rotation.
        translation: Translation, scene units.
        rms_scene: RMS residual of the fit, scene units.
        points: The triangulated keypoints, scene units.
        views: Capture names whose keypoints were triangulated.
    """

    scale: float
    rotation: NDArray[np.float64]
    translation: NDArray[np.float64]
    rms_scene: float
    points: NDArray[np.float64]
    views: tuple[str, ...]

    def __post_init__(self) -> None:
        """Freeze the arrays."""
        for name in ("rotation", "translation", "points"):
            array = np.array(getattr(self, name), dtype=np.float64)
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        object.__setattr__(self, "views", tuple(self.views))

    def as_matrix(self) -> NDArray[np.float64]:
        """``[[rotation, translation], [0, 0, 0, 1]]`` (unscaled rotation, as recorded)."""
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.translation
        return matrix

    def to_record(self) -> dict[str, Any]:
        """The fit as a JSON object (``triangulation.json``)."""
        return {"method": SEED_METHOD, "scale": self.scale, "T": self.as_matrix().tolist(),
                "rms_scene": self.rms_scene, "views": list(self.views),
                "triangulated_keypoints": self.points.tolist()}  # fmt: skip


def _usable(frame: FramePose, scene: NuRecScene, excluded: set[str], max_reproj_px: float) -> bool:
    error = frame.reprojection_error_px
    return (
        frame.image in scene.cameras
        and frame.image not in excluded
        and frame.keypoints is not None
        and error is not None
        and error < max_reproj_px
    )


def fit_keypoint_similarity(
    frames: Sequence[FramePose],
    scene: NuRecScene,
    keypoints_3d: ArrayLike,
    k: ArrayLike,
    *,
    exclude_views: Iterable[str] = (),
    max_reproj_px: float = 5.0,
    init: ArrayLike | None = None,
) -> SimilarityFit:
    """Triangulate the DOPER keypoints with the scene cameras; fit the CAD keypoints to them.

    The views are the frames (in their order) that are scene cameras, not excluded, and whose
    PnP reprojection error is below ``max_reproj_px``. Every keypoint is triangulated from all
    of them (soft-L1, 3 px) and ``umeyama_sim3(keypoints_3d, points)`` gives the seed.

    Args:
        frames: DOPER results per capture (``doper evaluate``'s ``frames.json`` rows).
        scene: The scene whose cameras observe the frames.
        keypoints_3d: ``(K, 3)`` CAD keypoints, meters.
        k: 3x3 intrinsics of the frames' keypoints (the capture camera).
        exclude_views: Capture names left out.
        max_reproj_px: PnP reprojection error a view must stay below, pixels.
        init: Triangulation start (``(3,)`` or ``(K, 3)``); default the linear solution.

    Returns:
        The seed similarity, its RMS and the triangulated keypoints.

    Raises:
        ValueError: With fewer than two usable views or a keypoint count that differs from
            the layout.
    """
    keypoints_3d = np.asarray(keypoints_3d, dtype=np.float64)
    excluded = set(exclude_views)
    used = [f for f in frames if _usable(f, scene, excluded, max_reproj_px)]
    if len(used) < MIN_TRIANGULATION_VIEWS:
        raise ValueError(f"the keypoint seed needs at least {MIN_TRIANGULATION_VIEWS} views "
                         f"(scene cameras, not excluded, PnP error < {max_reproj_px} px); "
                         f"{len(used)} qualify")  # fmt: skip
    uv = [np.asarray(f.keypoints, dtype=np.float64) for f in used]
    if any(view.shape != (len(keypoints_3d), 2) for view in uv):
        raise ValueError(f"every frame must hold {len(keypoints_3d)} keypoints, as the layout")
    cameras = [CameraPose(invert(scene.cameras[f.image])) for f in used]
    observations = [
        Observation(point=j, camera=c, uv=(float(uv[c][j, 0]), float(uv[c][j, 1])))
        for j in range(len(keypoints_3d)) for c in range(len(used))
    ]  # fmt: skip
    points = triangulate_points(observations, cameras, k, init=init)
    fit = umeyama_sim3(keypoints_3d, points)
    rms = float(np.sqrt(np.mean(np.sum((fit.apply(keypoints_3d) - points) ** 2, axis=1))))
    return SimilarityFit(fit.scale, fit.rotation, fit.translation, rms, points,
                         tuple(f.image for f in used))  # fmt: skip


def robot_scale_audit(robot: RobotReference, scene: NuRecScene) -> ScaleAnchor:
    """The similarity from the robot camera centers (meters) to the scene's (scene units).

    Umeyama over the captures that are scene cameras, in capture order: ``scene ~ s * R @
    c_robot + t`` with ``c_robot`` the translation of ``T_base_mount @ T_mount_camera``. Its
    scale is the registration's ``scene_units_per_metre`` when the capture has a robot
    reference; residuals are in scene units and, divided by the scale, millimeters.

    Raises:
        ValueError: For fewer than three shared captures or a degenerate set of centers.
    """
    robot_poses = reference_camera_poses(robot)
    names = [name for name in robot_poses if name in scene.cameras]
    if len(names) < MIN_AUDIT_CAMERAS:
        raise ValueError(f"the robot scale audit needs at least {MIN_AUDIT_CAMERAS} captures "
                         f"that are scene cameras, got {len(names)}")  # fmt: skip
    source = np.array([robot_poses[name][:3, 3] for name in names])
    target = np.array([scene.cameras[name][:3, 3] for name in names])
    fit = umeyama_sim3(source, target)
    residuals = np.linalg.norm(fit.apply(source) - target, axis=1)
    rms, largest = float(np.sqrt(np.mean(residuals**2))), float(residuals.max())
    tx, ty, tz = (float(v) for v in fit.translation)
    return ScaleAnchor(
        method=AUDIT_METHOD, scene_units_per_metre=fit.scale, rotation=fit.rotation.tolist(),
        translation=(tx, ty, tz), rms_scene=rms, max_scene=largest,
        rms_mm=rms / fit.scale * MM_PER_UNIT, max_mm=largest / fit.scale * MM_PER_UNIT,
        n=len(names),
    )  # fmt: skip


def robot_poses_in_scene(
    robot: RobotReference, anchor: ScaleAnchor
) -> dict[str, NDArray[np.float64]]:
    """Every capture's robot camera pose mapped into the scene by the robot audit.

    Rotation ``R @ R_robot``, center ``s * R @ c_robot + t``: the ``T_nre_cam`` the novel-view
    sampler rejects against (the 35 reference robot poses, captures outside the scene included).

    Returns:
        Capture name to the 4x4 ``T_nre_cam``, in capture order.
    """
    rotation = np.asarray(anchor.rotation, dtype=np.float64)
    translation = np.asarray(anchor.translation, dtype=np.float64)
    mapped = {}
    for name, pose in reference_camera_poses(robot).items():
        out = pose.copy()
        out[:3, :3] = rotation @ pose[:3, :3]
        out[:3, 3] = anchor.scene_units_per_metre * pose[:3, 3] @ rotation.T + translation
        mapped[name] = out
    return mapped


def _project(
    points: NDArray[np.float64], x: NDArray[np.float64], log_scale: float,
    cam_from_world: NDArray[np.float64], k: NDArray[np.float64],
) -> NDArray[np.float64]:  # fmt: skip
    """Pixels of CAD points under ``x = [rotvec, t]``: ``K (C (e^log_scale R p + t))``."""
    from scipy.spatial.transform import Rotation

    q = (points * np.exp(log_scale)) @ Rotation.from_rotvec(x[:3]).as_matrix().T + x[3:6]
    return project_pinhole(q, cam_from_world, k)


@dataclass(frozen=True)
class EdgeView:
    """One capture view of the edge refinement.

    Attributes:
        name: Capture name.
        cam_from_world: 4x4 scene-to-camera transform.
        distance: Distance (pixels) of each pixel to the nearest Canny edge pixel (float32).
        points: CAD vertices (meters) near the seed silhouette's boundary.
    """

    name: str
    cam_from_world: NDArray[np.float64]
    distance: NDArray[np.float32]
    points: NDArray[np.float64]


@dataclass(frozen=True)
class EdgeProblem:
    """The fixed data of an edge refinement: views, intrinsics, the fixed scale and solver.

    Attributes:
        views: Every view, in capture order.
        fit: Indices into ``views`` that the fit uses.
        holdout: Indices into ``views`` held out (reported only).
        k: 3x3 capture intrinsics.
        log_scale: Natural log of the fixed scale.
        f_scale: Soft-L1 inlier scale, pixels.
        diff_step: Relative finite-difference step.
    """

    views: tuple[EdgeView, ...]
    fit: tuple[int, ...]
    holdout: tuple[int, ...]
    k: NDArray[np.float64]
    log_scale: float
    f_scale: float
    diff_step: float

    def project(self, x: NDArray[np.float64], view: EdgeView) -> NDArray[np.float64]:
        """Pixels of ``view.points`` under ``x = [rotvec, t]`` and the fixed scale."""
        return _project(view.points, x, self.log_scale, view.cam_from_world, self.k)

    def residuals(self, x: NDArray[np.float64], indices: Sequence[int]) -> NDArray[np.float32]:
        """Edge distances (pixels, bilinear) of the chosen views' points, concatenated."""
        from scipy.ndimage import map_coordinates

        out = []
        for i in indices:
            view = self.views[i]
            uv = self.project(x, view)
            out.append(map_coordinates(view.distance, [uv[:, 1], uv[:, 0]], order=1,
                                       mode="nearest"))  # fmt: skip
        return np.concatenate(out)

    def solve(self, start: NDArray[np.float64], max_nfev: int) -> Any:  # noqa: ANN401
        """``least_squares`` (soft-L1) of the fit views' residuals from ``start``."""
        from scipy.optimize import least_squares

        return least_squares(lambda x: self.residuals(x, self.fit), start, loss="soft_l1",
                             f_scale=self.f_scale, diff_step=self.diff_step,
                             max_nfev=max_nfev)  # fmt: skip


@dataclass(frozen=True)
class RegistrationCandidate:
    """An edge-refined registration awaiting review.

    Attributes:
        T_nre_object: ``[[R, t], [0, 0, 0, 1]]`` with ``scene = s * R @ p_cad + t``.
        scene_units_per_metre: The fixed scale ``s``.
        scale_source: ``robot_audit`` or ``reconstruction_anchor``.
        seed: The seed's ``T`` (unscaled rotation) and its own scale.
        per_view_edges: Per view: ``source``, ``fit_view``, ``seed_median_px``,
            ``refined_median_px`` and ``refined_p90_px`` (edge distances, pixels).
        fit: Solver summary (``cost``, ``nfev``, ``status``, ``x``).
        settings: The refinement's settings.
        reconstruction_id: The scene's reconstruction.
        artifact_sha256: The scene's artifact.
        status: ``candidate_requires_review``.
        provenance: Inputs and capture flags recorded with the candidate.
        problem: The views and fields (in memory only, for ``restart_audit``).
    """

    T_nre_object: NDArray[np.float64]  # noqa: N815 - frame-named transform
    scene_units_per_metre: float
    scale_source: str
    seed: Mapping[str, Any]
    per_view_edges: tuple[Mapping[str, Any], ...]
    fit: Mapping[str, Any]
    settings: Mapping[str, Any]
    reconstruction_id: str
    artifact_sha256: str
    status: str = CANDIDATE_STATUS
    provenance: Mapping[str, Any] = field(default_factory=dict)
    problem: EdgeProblem | None = field(default=None, repr=False, compare=False)

    def to_record(self) -> dict[str, Any]:
        """The candidate as a JSON object (``candidate.json``)."""
        return {
            "status": self.status, "method": EDGE_METHOD, "convention": CONVENTION,
            "T_nre_object": np.asarray(self.T_nre_object).tolist(),
            "scene_units_per_metre": self.scene_units_per_metre, "scale_source": self.scale_source,
            "seed": dict(self.seed), "per_view_edges": [dict(v) for v in self.per_view_edges],
            "fit": dict(self.fit), "settings": dict(self.settings),
            "reconstruction_id": self.reconstruction_id, "artifact_sha256": self.artifact_sha256,
            **dict(self.provenance),
        }  # fmt: skip

    def write(self, path: str | Path) -> Path:
        """Write ``to_record`` as JSON (atomically)."""
        from ..bop.sources import publish_json

        publish_json(Path(path), self.to_record())
        return Path(path)

    @classmethod
    def read(cls, path: str | Path) -> RegistrationCandidate:
        """Read a ``candidate.json`` (without its in-memory problem)."""
        record = json.loads(Path(path).read_text())
        known = {"status", "method", "convention", "T_nre_object", "scene_units_per_metre",
                 "scale_source", "seed", "per_view_edges", "fit", "settings", "reconstruction_id",
                 "artifact_sha256"}  # fmt: skip
        return cls(
            T_nre_object=_rigid_transform(record["T_nre_object"], f"{path} T_nre_object"),
            scene_units_per_metre=_scale(record["scene_units_per_metre"], str(path)),
            scale_source=record["scale_source"], seed=record["seed"],
            per_view_edges=tuple(record["per_view_edges"]), fit=record["fit"],
            settings=record["settings"], reconstruction_id=record["reconstruction_id"],
            artifact_sha256=record["artifact_sha256"], status=record["status"],
            provenance={k: v for k, v in record.items() if k not in known},
        )  # fmt: skip


def _edge_field(path: Path, canny: tuple[float, float], size: tuple[int, int]) -> NDArray[Any]:
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"cannot read capture image {path}")
    if (image.shape[1], image.shape[0]) != size:
        raise ValueError(f"{path} is {image.shape[1]}x{image.shape[0]}, the capture camera "
                         f"{size[0]}x{size[1]}")  # fmt: skip
    edges = cv2.Canny(image, canny[0], canny[1])
    return cv2.distanceTransform(255 - edges, cv2.DIST_L2, DISTANCE_MASK)


def _boundary_points(
    mesh: TriangleMesh, pixels: NDArray[np.float64], size: tuple[int, int], *,
    boundary_px: float, points: int,
) -> NDArray[np.float64]:  # fmt: skip
    """CAD vertices whose seed projection is within ``boundary_px`` of the silhouette boundary.

    The silhouette is every face filled at its rounded vertex pixels; vertices are deduplicated
    (sorted) and every ``max(1, n // points)``-th one kept.
    """
    import cv2
    from scipy.ndimage import map_coordinates

    mask = np.zeros((size[1], size[0]), np.uint8)
    rounded = np.round(pixels).astype(np.int32)
    for face in mesh.faces:
        cv2.fillConvexPoly(mask, rounded[face], 255)
    kernel = np.ones((BOUNDARY_KERNEL, BOUNDARY_KERNEL), np.uint8)
    boundary = mask - cv2.erode(mask, kernel)
    distance = cv2.distanceTransform(255 - boundary, cv2.DIST_L2, DISTANCE_MASK)  # pyright: ignore[reportCallIssue, reportArgumentType]
    near = map_coordinates(distance, [pixels[:, 1], pixels[:, 0]], order=1, mode="nearest")
    chosen = np.unique(mesh.vertices[near < boundary_px], axis=0)
    return chosen[:: max(1, len(chosen) // points)]


def _check_views(
    fit_views: Sequence[str], holdout_views: Sequence[str], images: ImageSet, scene: NuRecScene
) -> list[str]:
    """The views in capture order; refuses unknown, duplicated or overlapping ones."""
    if not fit_views:
        raise ValueError("the edge refinement needs at least one fit view")
    overlap = sorted(set(fit_views) & set(holdout_views))
    if overlap or len(set(fit_views)) != len(fit_views):
        raise ValueError(f"views {overlap or sorted(fit_views)} are listed twice or as both fit "
                         "and holdout views")  # fmt: skip
    names = set(images.names)
    unknown = [v for v in (*fit_views, *holdout_views) if v not in scene.cameras or v not in names]
    if unknown:
        raise ValueError(f"views {unknown} are not a camera of the scene and an image of the "
                         "capture")  # fmt: skip
    chosen = {*fit_views, *holdout_views}
    return [name for name in images.names if name in chosen]


def _edge_problem(
    seed_x: NDArray[np.float64], images: ImageSet, scene: NuRecScene, mesh: TriangleMesh,
    names: Sequence[str], fit_views: Sequence[str], settings: Mapping[str, Any],
) -> EdgeProblem:  # fmt: skip
    size = (images.camera.width, images.camera.height)
    k = images.camera.matrix
    views = []
    for name in names:
        cam_from_world = np.linalg.inv(scene.cameras[name])
        distance = _edge_field(images.image_path(name), settings["canny"], size)
        pixels = _project(mesh.vertices, seed_x[:6], float(seed_x[6]), cam_from_world, k)  # pyright: ignore[reportArgumentType]
        chosen = _boundary_points(mesh, pixels, size, boundary_px=settings["boundary_px"],
                                  points=settings["points"])  # fmt: skip
        if not len(chosen):
            raise ValueError(f"the seed silhouette of view {name} has no CAD vertex near its "
                             "boundary (is the object in the image?)")  # fmt: skip
        views.append(EdgeView(name, cam_from_world, distance, chosen))  # pyright: ignore[reportArgumentType]
    fit = tuple(i for i, name in enumerate(names) if name in set(fit_views))
    holdout = tuple(i for i in range(len(names)) if i not in fit)
    return EdgeProblem(tuple(views), fit, holdout, k, float(seed_x[6]),
                       settings["f_scale"], settings["diff_step"])  # fmt: skip


def refine_edges(
    seed: SimilarityFit,
    images: ImageSet,
    scene: NuRecScene,
    mesh: TriangleMesh,
    *,
    fixed_scale: float,
    fit_views: Sequence[str],
    holdout_views: Sequence[str],
    canny: tuple[float, float] = (40, 100),
    boundary_px: float = 1.5,
    points: int = 250,
    f_scale: float = 2.0,
    diff_step: float = 1e-4,
    max_nfev: int = 200,
    scale_source: str = "robot_audit",
) -> RegistrationCandidate:
    """Refine the seed's rotation and translation against image edges at a fixed scale.

    Per view: ``Canny(image, *canny)``, then the L2 distance transform (5x5 mask) of the
    non-edge pixels. The CAD silhouette is filled at the seed pose (``scene = fixed_scale *
    R_seed @ p + t_seed``) and the CAD vertices within ``boundary_px`` of its boundary are kept,
    deduplicated and thinned to about ``points``; they stay fixed. The residuals are the
    distance fields (bilinear) at those vertices' projections. ``least_squares`` (soft-L1,
    ``f_scale``, ``diff_step``, ``max_nfev``) fits ``[rotvec, t]`` on the fit views only.

    Args:
        seed: The keypoint seed (its scale is replaced by ``fixed_scale``).
        images: The capture images (their camera is the projection's intrinsics).
        scene: The scene of the cameras.
        mesh: The CAD mesh, meters.
        fixed_scale: Scene units per meter.
        fit_views: Capture names the fit uses.
        holdout_views: Capture names held out and reported.
        canny: Canny thresholds.
        boundary_px: Distance to the seed silhouette boundary within which a vertex is kept.
        points: Approximate kept vertices per view.
        f_scale: Soft-L1 inlier scale, pixels.
        diff_step: Relative finite-difference step.
        max_nfev: Function evaluations of the fit.
        scale_source: Where ``fixed_scale`` came from (recorded).

    Returns:
        The candidate; per view the median edge distance at the seed and refined poses and
        the refined 90th percentile.

    Raises:
        ValueError: For unknown, duplicated or overlapping views, a non-positive scale, an
            unreadable or mis-sized image, or a view without boundary vertices.
    """
    from scipy.spatial.transform import Rotation

    if not math.isfinite(fixed_scale) or fixed_scale <= 0.0:
        raise ValueError(f"fixed_scale must be positive, got {fixed_scale}")
    names = _check_views(fit_views, holdout_views, images, scene)
    settings = {"canny": tuple(canny), "boundary_px": boundary_px, "points": points,
                "f_scale": f_scale, "diff_step": diff_step, "max_nfev": max_nfev,
                "fit_views": list(fit_views), "holdout_views": list(holdout_views)}  # fmt: skip
    seed_x = np.r_[Rotation.from_matrix(seed.rotation).as_rotvec(), seed.translation,
                   np.log(fixed_scale)]  # fmt: skip
    problem = _edge_problem(seed_x, images, scene, mesh, names, fit_views, settings)
    result = problem.solve(seed_x[:6], max_nfev)
    x = np.asarray(result.x, dtype=np.float64)
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix()
    transform[:3, 3] = x[3:6]
    per_view = []
    for i, view in enumerate(problem.views):
        before, after = problem.residuals(seed_x[:6], [i]), problem.residuals(x, [i])
        per_view.append({"source": view.name, "fit_view": i in problem.fit,
                         "seed_median_px": float(np.median(before)),
                         "refined_median_px": float(np.median(after)),
                         "refined_p90_px": float(np.percentile(after, P90)),
                         "points": len(view.points)})  # fmt: skip
    return RegistrationCandidate(
        T_nre_object=transform, scene_units_per_metre=float(np.exp(problem.log_scale)),
        scale_source=scale_source,
        seed={"T": seed.as_matrix().tolist(), "scale": seed.scale, "views": list(seed.views)},
        per_view_edges=tuple(per_view),
        fit={"cost": float(result.cost), "nfev": int(result.nfev), "status": int(result.status),
             "x": x.tolist()},
        settings={**settings, "canny": list(canny)}, reconstruction_id=scene.reconstruction_id,
        artifact_sha256=scene.artifact_sha256, problem=problem,
    )  # fmt: skip


def restart_audit(
    candidate: RegistrationCandidate,
    *,
    seed: int = 62,
    starts: int = 6,
    rot_deg: float = 5.0,
    trans: float = 0.002,
    max_nfev: int = 150,
) -> list[dict[str, Any]]:
    """Restart the edge fit from perturbations of the candidate (a diagnostic).

    Each start adds ``U(-rot_deg, rot_deg)`` degrees to each rotation-vector component, then
    ``U(-trans, trans)`` scene units to each translation component (``default_rng(seed)``, in
    that order), and refits with the candidate's boundary vertices, fields and solver settings.

    Returns:
        Per start: ``start``, ``rotation_from_candidate_deg``,
        ``translation_from_candidate_scene_mm`` (scene units x 1000), the fit and holdout median
        edge distances, the solver ``cost`` and its ``x``.

    Raises:
        ValueError: For a candidate read back from JSON (no in-memory views).
    """
    from scipy.spatial.transform import Rotation

    problem = candidate.problem
    if problem is None:
        raise ValueError("restart_audit needs the candidate refine_edges returned (with views)")
    best = np.asarray(candidate.T_nre_object)
    start_x = np.r_[Rotation.from_matrix(best[:3, :3]).as_rotvec(), best[:3, 3]]
    rng = np.random.default_rng(seed)
    results = []
    for j in range(starts):
        start = start_x.copy()
        start[:3] += rng.uniform(-np.deg2rad(rot_deg), np.deg2rad(rot_deg), 3)
        start[3:] += rng.uniform(-trans, trans, 3)
        fit = problem.solve(start, max_nfev)
        relative = Rotation.from_rotvec(fit.x[:3]).as_matrix() @ best[:3, :3].T
        results.append({
            "start": j,
            "rotation_from_candidate_deg": float(np.rad2deg(Rotation.from_matrix(relative)
                                                            .magnitude())),
            "translation_from_candidate_scene_mm": float(np.linalg.norm(fit.x[3:] - best[:3, 3])
                                                         * MM_PER_UNIT),
            "median_fit_edge_px": float(np.median(problem.residuals(fit.x, problem.fit))),
            "median_holdout_edge_px": float(np.median(problem.residuals(fit.x, problem.holdout)))
            if problem.holdout else None,
            "cost": float(fit.cost), "x": fit.x.tolist(),
        })  # fmt: skip
    return results


def _rigid_transform(matrix: Any, where: str) -> NDArray[np.float64]:  # noqa: ANN401
    return as_rigid(
        matrix, where, not_rigid="must hold a proper rotation (orthonormal, det +1) and the "
        "last row 0 0 0 1; the scale is recorded separately",
    )  # fmt: skip


def _scale(value: Any, where: str) -> float:  # noqa: ANN401
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} scene_units_per_metre must be a positive number, got {value!r}")
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{where} scene_units_per_metre must be positive and finite, got "
                         f"{value!r}")  # fmt: skip
    return float(value)


@dataclass(frozen=True)
class Registration:
    """A reviewed CAD-to-scene registration: ``scene = s * R @ p_cad + t``.

    Attributes:
        T_nre_object: ``[[R, t], [0, 0, 0, 1]]`` (proper rotation, scene-unit translation).
        scene_units_per_metre: ``s``.
        status: ``accepted`` (a candidate a reviewer accepted) or ``reviewed`` (a capture
            bundle's).
        artifact_sha256: SHA-256 of the NuRec artifact (``last.usdz``) whose scene frame the
            transform is in. It binds the registration to its scene: a registration never
            labels another artifact's scene.
        reconstruction_id: The reconstruction that scene was trained on, when known.
        method: How the transform was made.
        reviewer_note: The reviewer's note (an accepted candidate).
        provenance: The candidate's file hash, scale source and capture flags (an accepted
            candidate).
    """

    T_nre_object: NDArray[np.float64]  # noqa: N815 - frame-named transform
    scene_units_per_metre: float
    status: str
    artifact_sha256: str
    reconstruction_id: str | None = None
    method: str = ""
    reviewer_note: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the transform, scale and status.

        Raises:
            ValueError: For a non-rigid transform, a non-positive or non-finite scale, a
                status that is not ``accepted`` or ``reviewed``, an ``accepted`` registration
                without a reviewer note or a malformed artifact SHA-256.
        """
        sha = self.artifact_sha256
        if (
            not isinstance(sha, str)
            or len(sha) != _SHA256_LENGTH
            or not all(c in "0123456789abcdef" for c in sha)
        ):
            raise ValueError(f"registration artifact_sha256 must be the SHA-256 (64 lowercase hex) "
                             f"of the NuRec artifact it registers, got {sha!r}")  # fmt: skip
        object.__setattr__(
            self, "T_nre_object", _rigid_transform(self.T_nre_object, "registration T_nre_object")
        )
        object.__setattr__(self, "scene_units_per_metre",
                           _scale(self.scene_units_per_metre, "registration"))  # fmt: skip
        if self.status not in (ACCEPTED_STATUS, REVIEWED_STATUS):
            raise ValueError(f"registration status must be {ACCEPTED_STATUS!r} or "
                             f"{REVIEWED_STATUS!r}, got {self.status!r}: a candidate needs "
                             "review (--accept-registration)")  # fmt: skip
        if self.status == ACCEPTED_STATUS and not str(self.reviewer_note or "").strip():
            raise ValueError("an accepted registration needs the reviewer's note (what was "
                             "checked); accept the candidate with --accept-registration "
                             "--reviewer-note TEXT")  # fmt: skip

    def apply(self, points_cad_m: ArrayLike) -> NDArray[np.float64]:
        """CAD points (meters) in the scene: ``s * R @ p + t``."""
        points = np.asarray(points_cad_m, dtype=np.float64)
        rotation, translation = self.T_nre_object[:3, :3], self.T_nre_object[:3, 3]
        return self.scene_units_per_metre * points @ rotation.T + translation

    def to_record(self) -> dict[str, Any]:
        """The registration as its ``registration.json`` object."""
        return {
            "status": self.status,
            "method": self.method,
            "convention": CONVENTION,
            "T_nre_object": self.T_nre_object.tolist(),
            "scene_units_per_metre": self.scene_units_per_metre,
            "artifact_sha256": self.artifact_sha256,
            "reconstruction_id": self.reconstruction_id,
            "reviewer_note": self.reviewer_note,
            "provenance": dict(self.provenance),
        }

    def write(self, path: str | Path) -> Path:
        """Write ``to_record`` as JSON (atomically)."""
        from ..bop.sources import publish_json

        publish_json(Path(path), self.to_record())
        return Path(path)

    @classmethod
    def read(cls, path: str | Path) -> Registration:
        """Read and validate a ``registration.json``.

        Raises:
            ValueError: For unknown or missing keys, or an invalid transform, scale or status.
        """
        record = json.loads(Path(path).read_text())
        if not isinstance(record, Mapping):
            raise ValueError(f"{path} must hold a JSON object")
        unknown = sorted(set(record) - REGISTRATION_KEYS)
        missing = [key for key in REQUIRED_REGISTRATION_KEYS if key not in record]
        if unknown or missing:
            raise ValueError(f"{path} has unknown keys {unknown} / missing keys {missing}")
        try:
            return cls(
                T_nre_object=record["T_nre_object"],
                scene_units_per_metre=record["scene_units_per_metre"],
                status=record["status"], artifact_sha256=record["artifact_sha256"],
                reconstruction_id=record.get("reconstruction_id"),
                method=str(record.get("method", "")),
                reviewer_note=record.get("reviewer_note"),
                provenance=dict(record.get("provenance") or {}),
            )  # fmt: skip
        except ValueError as err:
            raise ValueError(f"{path}: {err}") from err


def accept_registration(
    candidate: RegistrationCandidate,
    reviewer_note: str,
    *,
    candidate_sha256: str | None = None,
    out: str | Path | None = None,
) -> Registration:
    """Accept a reviewed candidate as the registration labels use.

    Args:
        candidate: The reviewed candidate.
        reviewer_note: What the reviewer checked (required, non-blank).
        candidate_sha256: SHA-256 of the reviewed ``candidate.json`` (recorded).
        out: Where to write ``registration.json`` (not written when None).

    Returns:
        The accepted registration.

    Raises:
        ValueError: For a blank note or a candidate that is not awaiting review.
    """
    if not str(reviewer_note).strip():
        raise ValueError("accepting a registration needs a reviewer note (what was checked)")
    if candidate.status != CANDIDATE_STATUS:
        raise ValueError(f"only a {CANDIDATE_STATUS!r} candidate is accepted, got "
                         f"{candidate.status!r}")  # fmt: skip
    provenance = {"candidate_sha256": candidate_sha256,
                  "scale_source": candidate.scale_source,
                  "capture_flags": candidate.provenance.get("capture_flags")}  # fmt: skip
    registration = Registration(
        T_nre_object=candidate.T_nre_object,
        scene_units_per_metre=candidate.scene_units_per_metre, status=ACCEPTED_STATUS,
        artifact_sha256=candidate.artifact_sha256, reconstruction_id=candidate.reconstruction_id,
        method=EDGE_METHOD, reviewer_note=str(reviewer_note).strip(), provenance=provenance,
    )  # fmt: skip
    if out is not None:
        registration.write(out)
    return registration


def check_registration_scene(registration: Registration, scene: NuRecScene, where: str) -> None:
    """Check that ``registration`` is in ``scene``'s frame (it records the scene it was fitted on).

    The artifact SHA-256 must be the scene's: the transform is in that artifact's NRE frame. An
    accepted registration (fitted on the user's scene) must also name the scene's
    reconstruction when it records one. A bundle's reviewed registration is bound by its
    artifact only: its recorded reconstruction is the reference one, whose text model a re-run
    need not reproduce byte for byte, and the artifact already fixes the frame.

    Raises:
        ValueError: For a registration of another artifact or reconstruction.
    """
    if registration.artifact_sha256 != scene.artifact_sha256:
        raise ValueError(f"{where} registers the NuRec artifact {registration.artifact_sha256}, "
                         f"not this scene's {scene.artifact_sha256} ({scene.artifact}): a "
                         "registration never labels another artifact's scene; rerun --stage "
                         "register on this artifact")  # fmt: skip
    recorded = registration.reconstruction_id
    if (registration.status == ACCEPTED_STATUS and recorded is not None
            and recorded != scene.reconstruction_id):  # fmt: skip
        raise ValueError(f"{where} registers reconstruction {recorded}, not the scene's "
                         f"{scene.reconstruction_id}; rerun --stage register")  # fmt: skip


def resolve_registration(
    stage_dir: str | Path, bundle: str | Path | None, *, scene: NuRecScene | None = None
) -> Registration:
    """The registration later stages use: the accepted one, else the bundle's reviewed one.

    Args:
        stage_dir: The register stage's directory (``<root>/registration``).
        bundle: The capture bundle (its ``registration.json``, when it has one), or None.
        scene: When given, the registration must belong to it (``check_registration_scene``).

    Returns:
        The validated registration.

    Raises:
        RegistrationNotAcceptedError: When there is neither (a candidate alone does not count).
        ValueError: For an invalid file or a registration of another artifact or (accepted)
            reconstruction.
    """
    stage_dir = Path(stage_dir)
    accepted = stage_dir / REGISTRATION_NAME
    path = accepted if accepted.is_file() else None
    if path is None and bundle is not None and (Path(bundle) / REGISTRATION_NAME).is_file():
        path = Path(bundle) / REGISTRATION_NAME
    if path is not None:
        registration = Registration.read(path)
        if scene is not None:
            check_registration_scene(registration, scene, str(path))
        return registration
    pending = stage_dir / CANDIDATE_NAME
    hint = (f"review {pending} (its per-view edge distances and restart audit) and accept it"
            if pending.is_file() else "run `doper prepare <recipe> --stage register` and accept "
            "its candidate")  # fmt: skip
    raise RegistrationNotAcceptedError(
        f"no accepted registration: {hint} with `doper prepare <recipe> --stage register "
        "--accept-registration --reviewer-note TEXT`, or use a capture bundle with a reviewed "
        f"{REGISTRATION_NAME}"
    )


__all__ = [
    "CANDIDATE_NAME",
    "CONVENTION",
    "REGISTRATION_NAME",
    "EdgeProblem",
    "EdgeView",
    "Registration",
    "RegistrationCandidate",
    "RegistrationNotAcceptedError",
    "SimilarityFit",
    "TriangleMesh",
    "accept_registration",
    "check_registration_scene",
    "fit_keypoint_similarity",
    "read_obj_mesh",
    "refine_edges",
    "resolve_registration",
    "restart_audit",
    "robot_poses_in_scene",
    "robot_scale_audit",
]
