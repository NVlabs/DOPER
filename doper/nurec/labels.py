# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry labels of NuRec novel views: metric pose, keypoints, visibility and CAD silhouette.

Every rendered novel view gets its label from geometry alone: the view's camera pose in the
NuRec scene and the reviewed CAD-to-scene registration (``doper.nurec.registration``,
``scene = s * R @ p_cad + t``). The object pose in the camera is

    ``T_camera_object_m = T_cam_nre @ T_nre_object``, its translation divided by ``s``

(``T_cam_nre = inv(T_nre_cam)``; ``s = scene_units_per_metre`` of the registration). The
rotation is unchanged by the scale, and the translation becomes meters; the CAD model and its
keypoints stay in meters, never scaled. From that pose:

* ``keypoints_xy``: the CAD keypoints projected with the render camera's ``K`` (pixels);
* ``ray_visible``: whether each keypoint sees the camera past the mesh
  (``doper.reconstruction.visibility.ray_mesh_visible``, 0.5 mm tolerance by default) from the
  camera center in the object frame, ``-R^T t``;
* ``cad_mask``: the CAD silhouette, every triangle filled with ``cv2.fillConvexPoly`` at
  ``mask_scale`` of the image size (half resolution by default, its vertices rounded there),
  then resized to the image size by nearest neighbor;
* ``cad_bbox``: ``[x, y, w, h]`` of the projected CAD vertices, unclipped (a view whose object
  leaves the image keeps its full extent).

The silhouette is amodal: it ignores anything in front of the object. These labels are the
geometry step; masks from the renders, qualification and packaging follow in later steps of the
labels stage.

Frames and units: OpenCV camera axes (x right, y down, z forward); ``T_a_b`` maps
b-coordinates into a; scene lengths are NuRec scene units, labels are meters and pixels.
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..reconstruction.geometry import project_pinhole
from ..reconstruction.visibility import MeshEdges, mesh_edges, ray_mesh_visible_edges
from .capture import PinholeCamera
from .registration import Registration, TriangleMesh, check_registration_scene
from .render import NovelView
from .scene import NuRecScene

GEOMETRY_LABELS_NAME = "geometry_labels.json"
CAD_MASK_DIR = "cad_masks"
CAD_MASK_PATTERN = "{index:06d}.png"
DEFAULT_TOLERANCE_M = 0.0005
DEFAULT_MASK_SCALE = 0.5
DEFAULT_WORKERS = 8
# Keypoints tested per vectorized visibility batch: memory is about 50 bytes x points x faces,
# so 16 keypoints of a 65k-face mesh stay near 50 MB per worker.
VISIBILITY_CHUNK = 16
MASK_VALUE = 255
CONVENTION = ("T_camera_object_m = inv(T_nre_cam) @ T_nre_object, translation divided by the "
              "registration's scene_units_per_metre; OpenCV camera axes")  # fmt: skip


@dataclass(frozen=True)
class ViewLabel:
    """The geometry label of one novel view.

    Attributes:
        index: The view's index (its render frame).
        seed_view: The capture image the view perturbs.
        reconstruction_id: The scene's reconstruction (carried to packaging).
        lineage: The capture lineage of ``seed_view`` (carried to packaging).
        T_camera_nre: 4x4 NuRec scene to OpenCV camera (scene units).
        T_camera_object_m: 4x4 object (CAD, meters) to OpenCV camera, meters.
        keypoints_xy: ``(n, 2)`` projected CAD keypoints, pixels.
        ray_visible: ``(n,)`` whether each keypoint is ray-visible past the mesh.
        cad_bbox: ``(x, y, w, h)`` of the projected CAD vertices, pixels, unclipped.
        K: 3x3 intrinsic matrix of the render camera.
        image_size: ``(width, height)`` of the render, pixels.
        cad_mask: ``(height, width)`` uint8 silhouette (0 or 255), or None once written.
        mask_file: The written silhouette's file name (in the mask directory), or None.
    """

    index: int
    seed_view: str
    reconstruction_id: str
    lineage: str
    T_camera_nre: NDArray[np.float64]  # noqa: N815 - frame-pair name
    T_camera_object_m: NDArray[np.float64]  # noqa: N815 - frame-pair name
    keypoints_xy: NDArray[np.float64]
    ray_visible: NDArray[np.bool_]
    cad_bbox: tuple[float, float, float, float]
    K: NDArray[np.float64]  # noqa: N815 - the intrinsic matrix's usual name
    image_size: tuple[int, int]
    cad_mask: NDArray[np.uint8] | None = None
    mask_file: str | None = None

    def to_record(self) -> dict[str, Any]:
        """The label as its ``geometry_labels.json`` row (without the mask array)."""
        return {
            "index": self.index, "seed_view": self.seed_view,
            "reconstruction_id": self.reconstruction_id, "lineage": self.lineage,
            "T_camera_nre": self.T_camera_nre.tolist(),
            "T_camera_object_m": self.T_camera_object_m.tolist(),
            "keypoints_xy": self.keypoints_xy.tolist(),
            "ray_visible": [bool(v) for v in self.ray_visible],
            "cad_bbox": [float(v) for v in self.cad_bbox], "K": self.K.tolist(),
            "image_size": list(self.image_size), "mask_file": self.mask_file,
        }  # fmt: skip

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ViewLabel:
        """Build a label from its ``geometry_labels.json`` row."""
        return cls(
            index=int(record["index"]), seed_view=str(record["seed_view"]),
            reconstruction_id=str(record["reconstruction_id"]), lineage=str(record["lineage"]),
            T_camera_nre=np.array(record["T_camera_nre"], dtype=np.float64),
            T_camera_object_m=np.array(record["T_camera_object_m"], dtype=np.float64),
            keypoints_xy=np.array(record["keypoints_xy"], dtype=np.float64).reshape(-1, 2),
            ray_visible=np.array(record["ray_visible"], dtype=bool),
            cad_bbox=tuple(float(v) for v in record["cad_bbox"]),  # type: ignore[arg-type]
            K=np.array(record["K"], dtype=np.float64),
            image_size=(int(record["image_size"][0]), int(record["image_size"][1])),
            mask_file=record.get("mask_file"),
        )  # fmt: skip


def _check_scene(view: NovelView, scene: NuRecScene) -> None:
    if view.reconstruction_id != scene.reconstruction_id:
        raise ValueError(f"view {view.index} was sampled from reconstruction "
                         f"{view.reconstruction_id}, not the scene's {scene.reconstruction_id}; "
                         "rerun the render stage on this scene")  # fmt: skip


def rasterize_silhouette(
    vertices_xy: NDArray[np.float64], faces: NDArray[np.int64], camera: PinholeCamera,
    mask_scale: float,
) -> NDArray[np.uint8]:  # fmt: skip
    """The CAD silhouette: triangles filled at ``mask_scale``, resized (nearest) to the image.

    Args:
        vertices_xy: ``(v, 2)`` projected mesh vertices, image pixels.
        faces: ``(f, 3)`` triangle vertex indices.
        camera: The image camera (its size).
        mask_scale: Raster size relative to the image (0.5 = half resolution).

    Returns:
        ``(height, width)`` uint8 mask of 0 and 255.

    Raises:
        ValueError: For a non-positive or above-one ``mask_scale``.
    """
    if not 0.0 < mask_scale <= 1.0:
        raise ValueError(f"mask_scale must be in (0, 1], got {mask_scale}")
    size = (round(camera.height * mask_scale), round(camera.width * mask_scale))
    raster = np.zeros(size, np.uint8)
    corners = np.round(vertices_xy * mask_scale).astype(np.int32)[faces]
    for triangle in corners:
        cv2.fillConvexPoly(raster, triangle, MASK_VALUE)
    resized = cv2.resize(raster, (camera.width, camera.height), interpolation=cv2.INTER_NEAREST)
    return np.asarray(resized, dtype=np.uint8)


def geometry_labels(
    view: NovelView,
    scene: NuRecScene,
    registration: Registration,
    mesh: TriangleMesh,
    keypoints_3d: ArrayLike,
    camera: PinholeCamera,
    *,
    tolerance_m: float = DEFAULT_TOLERANCE_M,
    mask_scale: float = DEFAULT_MASK_SCALE,
) -> ViewLabel:
    """The geometry label of one novel view.

    Args:
        view: The novel view (``T_nre_cam`` in scene units).
        scene: The NuRec scene the view was sampled from.
        registration: The reviewed registration of the CAD model in that scene.
        mesh: The CAD mesh (meters, object frame).
        keypoints_3d: ``(n, 3)`` CAD keypoints, meters.
        camera: The render camera (the scene's calibration).
        tolerance_m: Ray visibility tolerance, meters.
        mask_scale: Silhouette raster size relative to the image.

    Returns:
        The label, with its full-size ``cad_mask``.

    Raises:
        ValueError: For a registration of another artifact or a view of another reconstruction.
    """
    check_registration_scene(registration, scene, "the registration")
    _check_scene(view, scene)
    return _label(view, registration, mesh, mesh_edges(mesh.vertices, mesh.faces),
                  np.asarray(keypoints_3d, dtype=np.float64), camera, tolerance_m,
                  mask_scale)  # fmt: skip


def _label(
    view: NovelView, registration: Registration, mesh: TriangleMesh, edges: MeshEdges,
    keypoints: NDArray[np.float64], camera: PinholeCamera, tolerance_m: float, mask_scale: float,
) -> ViewLabel:  # fmt: skip
    T_cam_nre = view.T_cam_nre  # noqa: N806 - frame-pair name
    pose = T_cam_nre @ registration.T_nre_object
    pose[:3, 3] /= registration.scene_units_per_metre
    k = camera.matrix
    centre = -pose[:3, :3].T @ pose[:3, 3]
    visible = ray_mesh_visible_edges(keypoints, centre, edges, tolerance_m=tolerance_m,
                                     chunk=VISIBILITY_CHUNK)  # fmt: skip
    vertices_xy = project_pinhole(mesh.vertices, pose, k)
    bbox = (float(vertices_xy[:, 0].min()), float(vertices_xy[:, 1].min()),
            float(np.ptp(vertices_xy[:, 0])), float(np.ptp(vertices_xy[:, 1])))  # fmt: skip
    return ViewLabel(
        index=view.index, seed_view=view.seed_view, reconstruction_id=view.reconstruction_id,
        lineage=view.lineage, T_camera_nre=T_cam_nre, T_camera_object_m=pose,
        keypoints_xy=project_pinhole(keypoints, pose, k), ray_visible=visible, cad_bbox=bbox, K=k,
        image_size=(camera.width, camera.height),
        cad_mask=rasterize_silhouette(vertices_xy, mesh.faces, camera, mask_scale),
    )  # fmt: skip


@dataclass(frozen=True)
class _BatchContext:
    registration: Registration
    mesh: TriangleMesh
    edges: MeshEdges
    keypoints: NDArray[np.float64]
    camera: PinholeCamera
    tolerance_m: float
    mask_scale: float
    mask_dir: Path | None


_CONTEXT: _BatchContext | None = None


def _init_worker(context: _BatchContext) -> None:
    global _CONTEXT  # noqa: PLW0603 - one batch context per worker process
    _CONTEXT = context


def _label_one(view: NovelView, context: _BatchContext | None = None) -> ViewLabel:
    context = context or _CONTEXT
    if context is None:
        raise RuntimeError("geometry label worker used before its batch context was set")
    label = _label(view, context.registration, context.mesh, context.edges, context.keypoints,
                   context.camera, context.tolerance_m, context.mask_scale)  # fmt: skip
    if context.mask_dir is None:
        return label
    name = CAD_MASK_PATTERN.format(index=view.index)
    if label.cad_mask is None:
        raise RuntimeError(f"view {view.index} was labeled without its CAD silhouette")
    if not cv2.imwrite(str(context.mask_dir / name), np.ascontiguousarray(label.cad_mask)):
        raise OSError(f"could not write {context.mask_dir / name}")
    return replace(label, cad_mask=None, mask_file=name)


def geometry_labels_batch(
    views: Sequence[NovelView],
    scene: NuRecScene,
    registration: Registration,
    mesh: TriangleMesh,
    keypoints_3d: ArrayLike,
    camera: PinholeCamera,
    *,
    tolerance_m: float = DEFAULT_TOLERANCE_M,
    mask_scale: float = DEFAULT_MASK_SCALE,
    workers: int = DEFAULT_WORKERS,
    mask_dir: str | Path | None = None,
) -> list[ViewLabel]:
    """``geometry_labels`` of many views, in view order, on ``workers`` processes.

    Args:
        views: The novel views.
        scene: The NuRec scene they were sampled from.
        registration: The reviewed registration in that scene.
        mesh: The CAD mesh (meters).
        keypoints_3d: ``(n, 3)`` CAD keypoints, meters.
        camera: The render camera.
        tolerance_m: Ray visibility tolerance, meters.
        mask_scale: Silhouette raster size relative to the image.
        workers: Worker processes (1 runs in this process). The labels do not depend on it.
        mask_dir: When given, each silhouette is written there as ``CAD_MASK_PATTERN`` PNG
            (the directory is created) and the labels carry ``mask_file`` instead of the array.

    Returns:
        The labels, in the order of ``views``.

    Raises:
        ValueError: For ``workers < 1``, a registration of another artifact or a view of
            another reconstruction.
    """
    if workers < 1:
        raise ValueError(f"workers must be positive, got {workers}")
    check_registration_scene(registration, scene, "the registration")
    for view in views:
        _check_scene(view, scene)
    context = _BatchContext(registration, mesh, mesh_edges(mesh.vertices, mesh.faces),
                            np.asarray(keypoints_3d, dtype=np.float64), camera, tolerance_m,
                            mask_scale,
                            None if mask_dir is None else Path(mask_dir))  # fmt: skip
    if mask_dir is not None:
        Path(mask_dir).mkdir(parents=True, exist_ok=True)
    if workers == 1 or len(views) <= 1:
        return [_label_one(view, context) for view in views]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(workers, len(views)),
        initializer=_init_worker,
        initargs=(context,),
        # Spawned, not forked: the caller may be multi-threaded (BLAS, OpenCV).
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        return list(pool.map(_label_one, views))


__all__ = [
    "CAD_MASK_DIR",
    "CAD_MASK_PATTERN",
    "CONVENTION",
    "DEFAULT_MASK_SCALE",
    "DEFAULT_TOLERANCE_M",
    "DEFAULT_WORKERS",
    "GEOMETRY_LABELS_NAME",
    "ViewLabel",
    "geometry_labels",
    "geometry_labels_batch",
    "rasterize_silhouette",
]
