# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The NuRec scene contract: a trained artifact's cameras, frames, units and capabilities.

A NuRec artifact (``last.usdz``, an uncompressed zip) carries ``rig_trajectories.json``:
``world_to_nre`` (``{"matrix": 4x4}``), the training rig poses ``T_rig_worlds`` and the
camera calibration ``camera_calibrations[<camera>]`` (``T_sensor_rig`` and an
``opencv-pinhole`` model). ``import_scene`` reads it and binds every training frame to its
capture image, so later stages never need a separate camera list.

Frames and units:

* Camera axes are OpenCV (x right, y down, z forward); ``T_a_b`` maps b-coordinates into a.
* NuRec's ``T_rig_world`` is the rig's pose (rig to world) and its ``T_sensor_rig`` maps sensor
  coordinates into the rig, so a training camera's pose in the NuRec scene frame is
  ``T_nre_cam = world_to_nre @ T_rig_world @ T_sensor_rig`` (``geometry.camera_from_rig``).
* The NuRec world is the COLMAP model's world (NuRec trains on the model's poses), and
  ``world_to_nre`` is the shift NuRec applies to it. Scene units are the model's units, which
  are **not** meters: ``scene_units_per_metre`` is the reconstruction's scale anchor
  (``reconstruction.json``). A scene without one is ``visual_only`` and every metric request on
  it raises ``MetricAlignmentError`` (it never silently becomes meters).

Binding: the artifact does not name its training images. Each training camera is matched to the
registered COLMAP image whose pose it reproduces (NuRec stores float32 poses, so the match
tolerance is ``BINDING_ATOL`` relative to the scene extent); the match must be one-to-one. A
frame that matches no image means the artifact was trained on another reconstruction, or with
pose optimization, and is refused. Frames NuRec left out of training are simply absent.

Capabilities: a NuRec scene renders RGB only. Metric depth and object masks are not available
(the renderer has no z-depth output and no object segmentation), and asking for them raises
``CapabilityError``.
"""

from __future__ import annotations

import json
import math
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..bop.sources import sha256_file
from ..reconstruction.geometry import T_SENSOR_RIG_FLU, as_rigid, camera_from_rig
from .capture import FrameMap, PinholeCamera

RIG_TRAJECTORIES_NAME = "rig_trajectories.json"
# The one camera a NuRec reconstruction of a single capture camera trains (NRE ``camera1``).
CAMERA_ID = "camera1@input"
CAMERA_MODEL = "opencv-pinhole"
SHUTTER = "GLOBAL"
ALIGNMENTS = ("metric_anchor", "visual_only")
# A training rig pose and its start/end pair agree to float32 storage precision.
POSE_PAIR_ATOL = 1e-6
# A training camera matches a model image within this tolerance (float32 storage), relative to
# the larger of 1 and the largest translation magnitude.
BINDING_ATOL = 1e-5
# The last row 0 0 0 1 is checked at numpy's default ``allclose`` tolerance.
LAST_ROW_ATOL = 1e-8
DISTORTION_KEYS = ("radial_coeffs", "tangential_coeffs", "thin_prism_coeffs")


class CapabilityError(ValueError):
    """A request for an output the NuRec scene cannot render (depth, object masks)."""


class MetricAlignmentError(ValueError):
    """A metric request (meters) on a ``visual_only`` scene, which has no metric scale."""


@dataclass(frozen=True)
class SceneCapabilities:
    """What a NuRec scene can render.

    Attributes:
        rgb: Color images (always).
        depth: Metric depth (never: the renderer has no z-depth output).
        object_mask: Object masks (never: the renderer has no object segmentation).
    """

    rgb: bool = True
    depth: bool = False
    object_mask: bool = False


def _rigid(matrix: ArrayLike, where: str) -> NDArray[np.float64]:
    return as_rigid(
        matrix, where, not_finite="must be a finite 4x4 matrix",
        not_rigid="must be a rigid transform", last_row_atol=LAST_ROW_ATOL,
    )  # fmt: skip


@dataclass(frozen=True)
class NuRecScene:
    """A trained NuRec scene bound to its reconstruction and capture.

    Attributes:
        artifact: The artifact (``last.usdz``).
        artifact_sha256: Its SHA-256.
        world_to_nre: 4x4 shift from the reconstruction world into the NuRec scene frame.
        cameras: Capture image (source) name to its training camera's ``T_nre_cam`` (OpenCV
            camera to NuRec scene), in NuRec's training order.
        camera: The training calibration (rectified pinhole).
        scene_units_per_metre: The reconstruction's metric scale, or None (``visual_only``).
        alignment: ``metric_anchor`` (scale known) or ``visual_only``.
        reconstruction_id: The reconstruction's identity (``reconstruction.json``).
        lineages: Source name to its capture lineage.
        capabilities: What the scene renders (RGB only).
    """

    artifact: Path
    artifact_sha256: str
    world_to_nre: NDArray[np.float64]
    cameras: Mapping[str, NDArray[np.float64]]
    camera: PinholeCamera
    scene_units_per_metre: float | None
    alignment: Literal["metric_anchor", "visual_only"]
    reconstruction_id: str
    lineages: Mapping[str, str]
    capabilities: SceneCapabilities = field(default_factory=SceneCapabilities)

    def __post_init__(self) -> None:
        """Check the alignment against the scale; freeze the poses.

        Raises:
            ValueError: For an unknown alignment, a ``metric_anchor`` scene without a positive
                scale or a ``visual_only`` one with a scale, no cameras, a non-rigid pose, a
                camera without a lineage, or capabilities beyond RGB.
        """
        if self.alignment not in ALIGNMENTS:
            raise ValueError(f"scene alignment must be one of {ALIGNMENTS}, got "
                             f"{self.alignment!r}")  # fmt: skip
        scale = self.scene_units_per_metre
        if self.alignment == "metric_anchor":
            if scale is None or not math.isfinite(scale) or scale <= 0.0:
                raise ValueError("a metric_anchor scene needs a positive scene_units_per_metre, "
                                 f"got {scale!r}")  # fmt: skip
            object.__setattr__(self, "scene_units_per_metre", float(scale))
        elif scale is not None:
            raise ValueError("a visual_only scene has no scene_units_per_metre")
        if self.capabilities != SceneCapabilities():
            raise ValueError("a NuRec scene renders RGB only: no depth and no object_mask")
        if not self.cameras:
            raise ValueError("a NuRec scene needs at least one training camera")
        object.__setattr__(self, "artifact", Path(self.artifact))
        object.__setattr__(self, "world_to_nre", _rigid(self.world_to_nre, "world_to_nre"))
        cameras = {str(k): _rigid(v, f"camera {k}") for k, v in self.cameras.items()}
        object.__setattr__(self, "cameras", cameras)
        missing = sorted(set(cameras) - set(self.lineages))
        if missing:
            raise ValueError(f"cameras {missing} have no lineage")
        object.__setattr__(self, "lineages", {k: str(self.lineages[k]) for k in cameras})

    @property
    def is_metric(self) -> bool:
        """Whether scene lengths convert to meters."""
        return self.alignment == "metric_anchor"

    def require(self, capability: str) -> None:
        """Check that the scene renders ``capability`` (``rgb``, ``depth``, ``object_mask``).

        Raises:
            CapabilityError: For depth, object masks or an unknown output.
        """
        names = {f.name for f in fields(SceneCapabilities)}
        if capability not in names or not getattr(self.capabilities, capability):
            raise CapabilityError(f"a NuRec scene cannot render {capability!r}: it renders RGB "
                                  "only (no metric depth, no object masks)")  # fmt: skip

    def _scale(self) -> float:
        if self.scene_units_per_metre is None:
            raise MetricAlignmentError(
                "this scene is visual_only (its reconstruction has no metric scale anchor): it "
                "has no metric lengths"
            )
        return self.scene_units_per_metre

    def metres_from_scene(self, length: float) -> float:
        """A scene-unit length in meters.

        Raises:
            MetricAlignmentError: On a ``visual_only`` scene.
        """
        return float(length) / self._scale()

    def scene_from_metres(self, length: float) -> float:
        """A length in meters in scene units.

        Raises:
            MetricAlignmentError: On a ``visual_only`` scene.
        """
        return float(length) * self._scale()


def read_rig_trajectories(artifact: str | Path) -> dict[str, Any]:
    """Read ``rig_trajectories.json`` from a NuRec artifact (a USDZ, which is a zip).

    Raises:
        ValueError: When the artifact is not a zip or has no ``rig_trajectories.json``.
    """
    try:
        with zipfile.ZipFile(artifact) as archive:
            data = archive.read(RIG_TRAJECTORIES_NAME)
    except (zipfile.BadZipFile, KeyError) as err:
        raise ValueError(f"{artifact} is not a NuRec artifact with {RIG_TRAJECTORIES_NAME}: "
                         f"{err}") from err  # fmt: skip
    return json.loads(data)


def _pair(values: Any, where: str) -> tuple[float, float]:  # noqa: ANN401
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError(f"{where} must be a pair of numbers")
    return float(values[0]), float(values[1])


def _calibration_camera(calibration: Mapping[str, Any]) -> PinholeCamera:
    """The rectified pinhole camera of a calibration (``opencv-pinhole``, GLOBAL, no distortion).

    Raises:
        ValueError: For another model or shutter, distortion, malformed intrinsics or a
            ``T_sensor_rig`` that is not the OpenCV-to-forward-left-up rotation.
    """
    model = calibration.get("camera_model", {})
    parameters = model.get("parameters", {})
    if model.get("type") != CAMERA_MODEL:
        raise ValueError(
            f"{CAMERA_ID} must use the {CAMERA_MODEL} model, got {model.get('type')!r}"
        )
    if parameters.get("shutter_type") != SHUTTER:
        raise ValueError(f"{CAMERA_ID} must have a {SHUTTER} shutter, got "
                         f"{parameters.get('shutter_type')!r}")  # fmt: skip
    distortion = [v for key in DISTORTION_KEYS for v in parameters.get(key) or ()]
    if any(float(v) != 0.0 for v in distortion) or parameters.get("external_distortion_parameters"):
        raise ValueError(f"{CAMERA_ID} has lens distortion; the scene contract needs a rectified "
                         "pinhole")  # fmt: skip
    sensor_rig = np.asarray(calibration.get("T_sensor_rig"), dtype=np.float64)
    expected = np.eye(4)
    expected[:3, :3] = T_SENSOR_RIG_FLU
    if sensor_rig.shape != (4, 4) or not np.allclose(sensor_rig, expected, atol=1e-12, rtol=0.0):
        raise ValueError(f"{CAMERA_ID} T_sensor_rig must be the OpenCV-to-forward-left-up "
                         "rotation")  # fmt: skip
    width, height = _pair(parameters.get("resolution"), f"{CAMERA_ID} resolution")
    fx, fy = _pair(parameters.get("focal_length"), f"{CAMERA_ID} focal_length")
    cx, cy = _pair(parameters.get("principal_point"), f"{CAMERA_ID} principal_point")
    return PinholeCamera(int(width), int(height), fx, fy, cx, cy, True)


def _training_rig_poses(record: Mapping[str, Any]) -> list[NDArray[np.float64]]:
    """The training rig poses (``T_rig_worlds``), checked to be global-shutter frames.

    Raises:
        ValueError: For several trajectories, a start/end pair that differs or disagrees with
            ``T_rig_worlds``, or start/end timestamps that differ.
    """
    trajectories = record.get("rig_trajectories")
    if not isinstance(trajectories, list) or len(trajectories) != 1:
        raise ValueError("the artifact must hold exactly one rig trajectory")
    trajectory = trajectories[0]
    poses = [_rigid(p, f"T_rig_worlds[{i}]") for i, p in enumerate(trajectory["T_rig_worlds"])]
    pairs = trajectory.get("cameras_frame_T_rig_worlds", {}).get(CAMERA_ID)
    stamps = trajectory.get("cameras_frame_timestamps_us", {}).get(CAMERA_ID)
    if pairs is None or stamps is None or len(pairs) != len(poses) or len(stamps) != len(poses):
        raise ValueError(f"{CAMERA_ID} frame poses and timestamps must cover every training frame")
    for i, (pose, pair, stamp) in enumerate(zip(poses, pairs, stamps, strict=True)):
        start, end = (np.asarray(p, dtype=np.float64) for p in pair)
        same = np.allclose(start, end, atol=POSE_PAIR_ATOL, rtol=0.0) and np.allclose(
            start, pose, atol=POSE_PAIR_ATOL, rtol=0.0
        )
        if not same or stamp[0] != stamp[1]:
            raise ValueError(f"training frame {i} is not a global-shutter frame (start and end "
                             "poses or timestamps differ)")  # fmt: skip
    return poses


def _bind_frames(
    T_world_cameras: Sequence[NDArray[np.float64]],  # noqa: N803 - frame-pair name
    model_poses: Mapping[str, NDArray[np.float64]],
) -> list[str]:
    """The model image name of each training camera, matched by pose (one to one).

    Raises:
        ValueError: For a frame that matches no image or several, or two frames on one image.
    """
    names = list(model_poses)
    stack = np.stack([model_poses[n][:3] for n in names])
    extent = max(1.0, float(np.abs(stack[:, :, 3]).max()))
    bound: list[str] = []
    for i, pose in enumerate(T_world_cameras):
        error = np.abs(stack - pose[:3]).max(axis=(1, 2))
        matches = np.flatnonzero(error <= BINDING_ATOL * extent)
        if len(matches) != 1:
            what = "matches no registered image" if not len(matches) else "matches several images"
            raise ValueError(f"training frame {i} {what} of the reconstruction (closest "
                             f"{error.min():.3g} scene units): the artifact was trained on "
                             "another reconstruction or with pose optimization")  # fmt: skip
        name = names[int(matches[0])]
        if name in bound:
            raise ValueError(f"training frames {bound.index(name)} and {i} both match {name}")
        bound.append(name)
    return bound


def import_scene(
    artifact: str | Path, reconstruction_json: str | Path, frames: FrameMap
) -> NuRecScene:
    """Import a trained NuRec artifact as a scene bound to its reconstruction and capture.

    Reads the artifact's ``rig_trajectories.json``, forms each training camera's
    ``T_nre_cam = world_to_nre @ T_rig_world @ T_sensor_rig``, binds it to the registered model
    image whose pose it reproduces and, through ``reconstruction.json`` and the frame map, to its
    capture image and lineage. ``scene_units_per_metre`` is the reconstruction's scale anchor;
    a reconstruction without one gives a ``visual_only`` scene.

    Args:
        artifact: The NuRec artifact (``last.usdz``).
        reconstruction_json: The reconstruction's ``reconstruction.json``.
        frames: The capture's frame map (normalized name to source name and lineage).

    Returns:
        The scene; ``cameras`` is keyed by source name in NuRec's training order.

    Raises:
        ValueError: When the text model changed since the reconstruction (its id differs), the
            calibration is not a rectified global-shutter pinhole on ``CAMERA_ID``,
            ``T_world_base`` is not the identity, a training frame does not bind, or the frame
            map does not hold a bound image.
    """
    from ..reconstruction import colmap_io
    from ..reconstruction.metadata import ReconstructionMetadata
    from ..reconstruction.seeding import text_model_id

    artifact, reconstruction_json = Path(artifact), Path(reconstruction_json)
    metadata = ReconstructionMetadata.read(reconstruction_json)
    model_dir = reconstruction_json.parent / metadata.model_text
    if text_model_id(model_dir) != metadata.reconstruction_id:
        raise ValueError(f"{model_dir} changed since the reconstruction: its identity is not "
                         f"reconstruction_id {metadata.reconstruction_id}")  # fmt: skip
    record = read_rig_trajectories(artifact)
    if not np.array_equal(np.asarray(record.get("T_world_base", np.eye(4))), np.eye(4)):
        raise ValueError("the artifact's T_world_base must be the identity")
    calibrations = record.get("camera_calibrations", {})
    if list(calibrations) != [CAMERA_ID]:
        raise ValueError(f"the artifact must calibrate exactly one camera, {CAMERA_ID}; it has "
                         f"{sorted(calibrations)}")  # fmt: skip
    camera = _calibration_camera(calibrations[CAMERA_ID])
    world_to_nre = _rigid(record["world_to_nre"]["matrix"], "world_to_nre")
    rig_poses = _training_rig_poses(record)
    model = colmap_io.read_text_model(model_dir)
    registered = {image.name for image in metadata.images if image.registered}
    model_poses = {
        image.name: np.linalg.inv(colmap_io.cam_from_world(image))
        for image in model.images.values() if image.name in registered
    }  # fmt: skip
    world_cameras = [camera_from_rig(pose, T_SENSOR_RIG_FLU) for pose in rig_poses]
    names = _bind_frames(world_cameras, model_poses)  # pyright: ignore[reportArgumentType]
    by_name = {image.name: image for image in metadata.images}
    cameras: dict[str, NDArray[np.float64]] = {}
    lineages: dict[str, str] = {}
    for name, pose in zip(names, rig_poses, strict=True):
        source = by_name[name].source_name
        frame = frames.by_source(source)
        if frame.name != name:
            raise ValueError(f"the frame map names source {source} {frame.name}, the "
                             f"reconstruction {name}")  # fmt: skip
        cameras[source] = camera_from_rig(world_to_nre @ pose, T_SENSOR_RIG_FLU)
        lineages[source] = frame.lineage
    return NuRecScene(
        artifact=artifact,
        artifact_sha256=sha256_file(artifact),
        world_to_nre=world_to_nre,
        cameras=cameras,
        camera=camera,
        scene_units_per_metre=metadata.scene_units_per_metre,
        alignment="metric_anchor" if metadata.is_metric else "visual_only",
        reconstruction_id=metadata.reconstruction_id,
        lineages=lineages,
    )


def import_metric_scene(
    artifact: str | Path, reconstruction_json: str | Path, frames: FrameMap
) -> NuRecScene:
    """``import_scene`` for the stages that need metric lengths (register, render, labels).

    Raises:
        MetricAlignmentError: On a ``visual_only`` scene (no metric scale anchor).
        ValueError: As ``import_scene``.
    """
    scene = import_scene(artifact, reconstruction_json, frames)
    scene._scale()  # noqa: SLF001 - raises on a visual_only scene
    return scene


__all__ = [
    "ALIGNMENTS",
    "CAMERA_ID",
    "RIG_TRAJECTORIES_NAME",
    "CapabilityError",
    "MetricAlignmentError",
    "NuRecScene",
    "SceneCapabilities",
    "import_metric_scene",
    "import_scene",
    "read_rig_trajectories",
]
