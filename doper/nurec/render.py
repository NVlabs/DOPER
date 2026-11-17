# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Novel views of a NuRec scene: the pose sampler, the NRE rig trajectory and the render backend.

``sample_novel_views`` perturbs source cameras of the scene and rejects views too close to any
rejection source. ``write_rig_trajectory`` writes those views as an NRE custom rig trajectory,
and ``NreCliBackend`` renders it with the NRE container's ``render`` command.

Frames and units: poses are ``T_nre_cam`` (OpenCV camera, x right, y down, z forward, to the
NuRec scene frame). NuRec scene units are not meters; the sampler's translation perturbation
``perturb_m`` is applied in scene units while its rejection distance ``reject_mm`` is metric and
converted with the scene's ``scene_units_per_metre``, so the sampler refuses a ``visual_only``
scene.

The render backend is the NRE CLI with ``--custom-rig-trajectory``: the path that rendered the
reference data. That flag is listed by the container's ``--help`` but is not in the public render
documentation, so the backend pins the image by digest (``NAME@sha256:...``) and nothing else.
Rendering needs a host with docker, an NVIDIA GPU and NGC access to the image; the backend
refuses to start without docker or a GPU.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import struct
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..bop.sources import publish_text
from ..reconstruction.geometry import T_SENSOR_RIG_FLU, invert, rig_from_camera
from .capture import PinholeCamera
from .scene import CAMERA_ID, CAMERA_MODEL, SHUTTER, NuRecScene

# Consecutive rejected draws after which the sampler gives up (thresholds wider than the range).
MAX_CONSECUTIVE_REJECTIONS = 10_000
# NRE trajectory frames are 0.1 s apart; each frame's start and end time are equal (global
# shutter).
FRAME_INTERVAL_US = 100_000
SEQUENCE_ID = "input"
SENSOR_NAME = "camera1"
IMAGE_DIGEST = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
CONTAINER_ARTIFACT = "/artifact"
CONTAINER_RENDER = "/render"
RGB_DIR = "rgb"
FRAME_PATTERN = "{index:06d}.png"
# NRE writes this sidecar (file_name and frame timestamps per frame) next to the frames.
TIMESTAMPS_NAME = "timestamps.json"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
EXCLUSION = ("a view is rejected when it is within reject_deg AND reject_mm of any rejection "
             "source (the scene's training cameras, then the extra sources)")  # fmt: skip


class RenderHostError(OSError):
    """The host cannot run the NRE container (no docker or no NVIDIA GPU)."""


class RenderFailedError(OSError):
    """The NRE container exited with an error."""


@dataclass(frozen=True)
class NovelView:
    """One sampled novel camera.

    Attributes:
        index: Position in the trajectory.
        seed_view: Source camera (capture image name) it perturbs.
        T_nre_cam: 4x4 pose, OpenCV camera to NuRec scene.
        min_source_rotation_deg: Smallest rotation to any rejection source, degrees.
        min_source_translation_mm: Smallest camera-center distance to any source, millimeters.
        closest_source: Index of the source with the smallest rotation (deg) + translation (mm).
        closest_rotation_deg: That source's rotation, degrees.
        closest_translation_mm: That source's distance, millimeters.
        rejected_before: Draws rejected between the previous view and this one.
        reconstruction_id: The scene's reconstruction.
        lineage: Capture lineage of ``seed_view``.
    """

    index: int
    seed_view: str
    T_nre_cam: NDArray[np.float64]
    min_source_rotation_deg: float
    min_source_translation_mm: float
    closest_source: int
    closest_rotation_deg: float
    closest_translation_mm: float
    rejected_before: int
    reconstruction_id: str
    lineage: str

    @property
    def T_cam_nre(self) -> NDArray[np.float64]:  # noqa: N802 - frame-pair name
        """The inverse pose, NuRec scene to OpenCV camera."""
        return invert(self.T_nre_cam)

    def to_record(self) -> dict[str, Any]:
        """The view as a JSON object (``views.json`` row)."""
        return {
            "index": self.index, "seed_view": self.seed_view,
            "T_nre_cam": np.asarray(self.T_nre_cam).tolist(),
            "min_source_rotation_deg": self.min_source_rotation_deg,
            "min_source_translation_mm": self.min_source_translation_mm,
            "closest_source": self.closest_source,
            "closest_rotation_deg": self.closest_rotation_deg,
            "closest_translation_mm": self.closest_translation_mm,
            "rejected_before": self.rejected_before,
            "reconstruction_id": self.reconstruction_id, "lineage": self.lineage,
        }  # fmt: skip

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> NovelView:
        """Build a view from its ``views.json`` row."""
        return cls(**{**record, "T_nre_cam": np.array(record["T_nre_cam"], dtype=np.float64)})


def _sources(scene: NuRecScene, extra: Sequence[ArrayLike]) -> NDArray[np.float64]:
    stacked = [np.asarray(pose, dtype=np.float64) for pose in (*scene.cameras.values(), *extra)]
    if any(pose.shape != (4, 4) for pose in stacked):
        raise ValueError("every rejection source must be a 4x4 T_nre_cam pose")
    return np.stack(stacked)


def _check_sampling(count: int, seed_views: Sequence[str], scene: NuRecScene,
                    limits: Mapping[str, float]) -> None:  # fmt: skip
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    if isinstance(seed_views, str) or not seed_views:
        raise ValueError("seed_views must be a non-empty list of camera names")
    unknown = [view for view in seed_views if view not in scene.cameras]
    if unknown:
        raise ValueError(f"seed views {unknown} are not training cameras of the scene")
    for name, value in limits.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative, got {value}")


def sample_novel_views(
    scene: NuRecScene,
    extra_sources: Sequence[ArrayLike] = (),
    *,
    seed: int,
    count: int,
    seed_views: Sequence[str],
    perturb_deg: float = 4.0,
    perturb_m: float = 0.005,
    reject_deg: float = 1.0,
    reject_mm: float = 2.0,
) -> list[NovelView]:
    """Sample novel cameras by perturbing source cameras, rejecting near-duplicates of sources.

    Round-robin over ``seed_views`` (a rejected draw retries the same seed view): each draw is
    ``novel = T_nre_cam[seed] @ D`` with ``D`` a camera-local rotation
    ``Rotation.from_euler("xyz", U(-perturb_deg, perturb_deg)^3, degrees=True)`` and a
    camera-local translation ``U(-perturb_m, perturb_m)^3`` in scene units, both from
    ``numpy.random.default_rng(seed)`` in that order. A draw within ``reject_deg`` (geodesic
    rotation) AND ``reject_mm`` (camera-center distance, scene units converted to millimeters
    with ``scene_units_per_metre``) of any rejection source is rejected.

    Args:
        scene: The scene; its training cameras are rejection sources (first, in order).
        extra_sources: Further rejection sources, already mapped into the scene
            (``T_nre_cam``), such as the capture's robot camera poses mapped by the robot to
            scene similarity.
        seed: RNG seed.
        count: Views to sample.
        seed_views: Camera names (keys of ``scene.cameras``) to perturb, round-robin.
        perturb_deg: Half-range of each local Euler angle, degrees.
        perturb_m: Half-range of each local translation component, **scene units** (despite
            the recipe key's name).
        reject_deg: Rotation half of the rejection test, degrees.
        reject_mm: Translation half of the rejection test, millimeters.

    Returns:
        The views in sampling order; ``rejected_before`` sums to the rejected draws.

    Raises:
        MetricAlignmentError: On a ``visual_only`` scene (``reject_mm`` needs meters).
        ValueError: For an unknown seed view, a non-positive count, a negative or non-finite
            range or threshold, a source that is not 4x4, or after
            ``MAX_CONSECUTIVE_REJECTIONS`` rejected draws in a row.
    """
    from scipy.spatial.transform import Rotation

    scale = scene.scene_from_metres(1.0)
    limits = {"perturb_deg": perturb_deg, "perturb_m": perturb_m, "reject_deg": reject_deg,
              "reject_mm": reject_mm}  # fmt: skip
    _check_sampling(count, seed_views, scene, limits)
    sources = _sources(scene, extra_sources)
    source_rotations_t = sources[:, :3, :3].transpose(0, 2, 1)
    rng = np.random.default_rng(seed)
    views: list[NovelView] = []
    rejected = 0
    while len(views) < count:
        name = seed_views[len(views) % len(seed_views)]
        perturbation = np.eye(4)
        angles = rng.uniform(-perturb_deg, perturb_deg, 3)
        perturbation[:3, :3] = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
        perturbation[:3, 3] = rng.uniform(-perturb_m, perturb_m, 3)
        novel = scene.cameras[name] @ perturbation
        relative = np.einsum("nij,jk->nik", source_rotations_t, novel[:3, :3])
        rotation = np.rad2deg(Rotation.from_matrix(relative).magnitude())
        translation = np.linalg.norm(sources[:, :3, 3] - novel[:3, 3], axis=1) / scale * 1000
        if np.any((rotation < reject_deg) & (translation < reject_mm)):
            rejected += 1
            if rejected >= MAX_CONSECUTIVE_REJECTIONS:
                raise ValueError(f"{rejected} consecutive draws were rejected: the rejection "
                                 "thresholds cover the whole perturbation range")  # fmt: skip
            continue
        closest = int(np.argmin(rotation + translation))
        views.append(NovelView(
            index=len(views), seed_view=name, T_nre_cam=novel,
            min_source_rotation_deg=float(rotation.min()),
            min_source_translation_mm=float(translation.min()), closest_source=closest,
            closest_rotation_deg=float(rotation[closest]),
            closest_translation_mm=float(translation[closest]), rejected_before=rejected,
            reconstruction_id=scene.reconstruction_id, lineage=scene.lineages[name],
        ))  # fmt: skip
        rejected = 0
    return views


def _calibration(camera: PinholeCamera) -> dict[str, Any]:
    sensor_rig = np.eye(4)
    sensor_rig[:3, :3] = T_SENSOR_RIG_FLU
    parameters = {
        "resolution": [camera.width, camera.height], "shutter_type": SHUTTER,
        "external_distortion_parameters": None, "principal_point": [camera.cx, camera.cy],
        "focal_length": [camera.fx, camera.fy], "radial_coeffs": [0.0] * 6,
        "tangential_coeffs": [0.0] * 2, "thin_prism_coeffs": [0.0] * 4,
    }  # fmt: skip
    return {"sequence_id": SEQUENCE_ID, "logical_sensor_name": SENSOR_NAME,
            "unique_sensor_idx": 0, "T_sensor_rig": sensor_rig.tolist(),
            "camera_model": {"type": CAMERA_MODEL, "parameters": parameters}}  # fmt: skip


def write_rig_trajectory(
    views: Sequence[NovelView], scene: NuRecScene, camera: PinholeCamera, out: str | Path
) -> Path:
    """Write views as an NRE custom rig trajectory (``--custom-rig-trajectory`` JSON).

    Each view becomes the rig pose ``T_rig_world = inv(world_to_nre) @ T_nre_cam @
    inv(T_sensor_rig)`` (the inverse of ``import_scene``'s composition) at timestamp
    ``index * 100000`` us, with equal start and end poses and times (global shutter). The
    calibration is ``camera`` as an ``opencv-pinhole`` with zero distortion.

    Args:
        views: The novel views, in render order.
        scene: Their scene.
        camera: Render camera (rectified pinhole).
        out: Output JSON path (written atomically).

    Returns:
        ``out``.

    Raises:
        ValueError: For no views, views of another reconstruction or an unrectified camera.
    """
    out = Path(out)
    if not views:
        raise ValueError("write_rig_trajectory needs at least one view")
    other = {v.reconstruction_id for v in views} - {scene.reconstruction_id}
    if other:
        raise ValueError(f"views of reconstruction {sorted(other)} are not of the scene's "
                         f"reconstruction {scene.reconstruction_id}")  # fmt: skip
    if not camera.rectified:
        raise ValueError("the render camera must be rectified (zero distortion)")
    nre_to_world = np.linalg.inv(scene.world_to_nre)
    rigs = [rig_from_camera(nre_to_world @ v.T_nre_cam, T_SENSOR_RIG_FLU).tolist() for v in views]
    stamps = [i * FRAME_INTERVAL_US for i in range(len(views))]
    record = {
        "T_world_base": np.eye(4).tolist(),
        "world_to_nre": {"matrix": scene.world_to_nre.tolist()},
        "rig_trajectories": [{
            "sequence_id": SEQUENCE_ID, "rig_bbox": None,
            "cameras_linear_start_frame_indices": {CAMERA_ID: 0},
            "lidars_linear_start_frame_indices": {},
            "cameras_frame_timestamps_us": {CAMERA_ID: [[t, t] for t in stamps]},
            "lidars_frame_timestamps_us": {}, "T_rig_worlds": rigs,
            "T_rig_world_timestamps_us": stamps,
            "cameras_frame_T_rig_worlds": {CAMERA_ID: [[r, r] for r in rigs]},
        }],
        "camera_calibrations": {CAMERA_ID: _calibration(camera)},
        "lidar_calibrations": {},
    }  # fmt: skip
    publish_text(out, json.dumps(record))
    return out


def _check_timestamps(path: Path, expected: Sequence[str]) -> None:
    """Check the NRE ``timestamps.json`` sidecar names exactly the expected frames, in order.

    Raises:
        ValueError: For an unreadable or malformed sidecar (not a JSON list of objects with a
            ``file_name``) or one naming other frames.
    """
    try:
        rows = json.loads(path.read_text())
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise TypeError("expected a JSON list of objects")
        names = [row["file_name"] for row in rows]
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ValueError(f"{path} is not an NRE frame timestamps list: {error}") from error
    if names != list(expected):
        raise ValueError(f"{path} names {len(names)} frames that are not the expected "
                         f"{len(expected)} frames {expected[:1]}..{expected[-1:]}")  # fmt: skip


def _png_size(path: Path) -> tuple[int, int]:
    with open(path, "rb") as handle:
        head = handle.read(24)
    if len(head) < 24 or head[:8] != PNG_SIGNATURE or head[12:16] != b"IHDR":
        raise ValueError(f"{path} is not a PNG")
    width, height = struct.unpack(">II", head[16:24])
    return int(width), int(height)


@dataclass(frozen=True)
class RenderResult:
    """A complete render: one PNG per trajectory frame, all at the camera size.

    Attributes:
        frames: The frames in trajectory order.
        width: Frame width, pixels.
        height: Frame height, pixels.
    """

    frames: tuple[Path, ...]
    width: int
    height: int

    @classmethod
    def collect(cls, directory: str | Path, *, count: int, width: int, height: int) -> RenderResult:
        """Verify a render directory holds exactly frames ``000000.png`` .. ``count - 1``.

        NRE's ``timestamps.json`` sidecar may sit next to the frames; when present it must name
        exactly those frames.

        Raises:
            ValueError: For a missing or extra frame (partial or mixed output), a frame of
                another size or a sidecar naming other frames.
        """
        directory = Path(directory)
        expected = [FRAME_PATTERN.format(index=i) for i in range(count)]
        present = sorted(p.name for p in directory.iterdir()) if directory.is_dir() else []
        if TIMESTAMPS_NAME in present:
            present.remove(TIMESTAMPS_NAME)
            _check_timestamps(directory / TIMESTAMPS_NAME, expected)
        missing = sorted(set(expected) - set(present))
        extra = sorted(set(present) - set(expected))
        if missing or extra:
            raise ValueError(f"render output {directory} is incomplete or mixed: missing "
                             f"{missing[:5]} ({len(missing)}), unexpected {extra[:5]} "
                             f"({len(extra)})")  # fmt: skip
        frames = tuple(directory / name for name in expected)
        for frame in frames:
            size = _png_size(frame)
            if size != (width, height):
                raise ValueError(f"{frame} is {size[0]}x{size[1]}, expected {width}x{height}")
        return cls(frames, width, height)


class RenderBackend(Protocol):
    """Renders a trajectory of a NuRec scene."""

    def render(self, scene: NuRecScene, trajectory: Path, out: Path) -> RenderResult:
        """Render ``trajectory`` (written by ``write_rig_trajectory``) into ``out``."""
        ...


@dataclass(frozen=True)
class Mount:
    """A host directory mounted into the render container.

    Attributes:
        source: Host path.
        target: Absolute container path (not ``/artifact`` or ``/render``, which the backend
            mounts itself).
        read_only: Mount read-only.
    """

    source: Path
    target: str
    read_only: bool = False

    def __post_init__(self) -> None:
        """Check the target.

        Raises:
            ValueError: For a relative target or one the backend uses.
        """
        object.__setattr__(self, "source", Path(self.source))
        if not self.target.startswith("/") or self.target in (CONTAINER_ARTIFACT, CONTAINER_RENDER):
            raise ValueError(f"mount target must be absolute and not {CONTAINER_ARTIFACT} or "
                             f"{CONTAINER_RENDER}, got {self.target!r}")  # fmt: skip

    def argv(self) -> list[str]:
        """The ``-v`` arguments."""
        suffix = ":ro" if self.read_only else ""
        return ["-v", f"{self.source.absolute()}:{self.target}{suffix}"]


def _run(argv: list[str]) -> int:
    return subprocess.run(argv, check=False).returncode


class NreCliBackend:
    """Render with the NRE container's ``render`` command and ``--custom-rig-trajectory``.

    The ``docker run`` command runs as the current user (``--user uid:gid``), mounts the
    artifact's directory read-only at ``/artifact`` and the output directory at ``/render``, plus
    the caller's mounts and environment. ``--custom-rig-trajectory`` is undocumented in the
    public render guide, so the image must be pinned by digest.
    """

    def __init__(
        self,
        image: str,
        *,
        docker: str = "docker",
        mounts: Sequence[Mount] = (),
        env: Mapping[str, str] | None = None,
        shm_size: str = "64g",
        gpus: str = "all",
        runner: Callable[[list[str]], int] | None = None,
    ) -> None:
        """Configure the backend.

        Args:
            image: The NRE image, ``NAME@sha256:<64 hex>``.
            docker: The docker executable.
            mounts: Extra mounts (for example a model cache).
            env: Extra container environment.
            shm_size: ``--shm-size``.
            gpus: ``--gpus``.
            runner: Runs the argv and returns its exit status (default: ``subprocess.run``).

        Raises:
            ValueError: For an image not pinned by digest.
        """
        if not IMAGE_DIGEST.fullmatch(image):
            raise ValueError(f"the NRE image must be pinned by digest (NAME@sha256:<64 hex>), got "
                             f"{image!r}")  # fmt: skip
        self.image = image
        self.docker = docker
        self.mounts = tuple(mounts)
        self.env = dict(env or {})
        self.shm_size = shm_size
        self.gpus = gpus
        self.runner = runner or _run

    def command(self, scene: NuRecScene, trajectory: str | Path, out: str | Path) -> list[str]:
        """The ``docker run`` argv that renders ``trajectory`` of ``scene`` into ``out/rgb``.

        Raises:
            ValueError: When ``trajectory`` is not inside ``out``.
        """
        out = Path(out).absolute()
        trajectory = Path(trajectory).absolute()
        if not trajectory.is_relative_to(out):
            raise ValueError(f"the trajectory {trajectory} must be inside the output directory "
                             f"{out} (mounted at {CONTAINER_RENDER})")  # fmt: skip
        artifact = scene.artifact.absolute()
        argv = [self.docker, "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
                "--shm-size", self.shm_size, "--gpus", self.gpus,
                "-v", f"{artifact.parent}:{CONTAINER_ARTIFACT}:ro",
                "-v", f"{out}:{CONTAINER_RENDER}"]  # fmt: skip
        for mount in self.mounts:
            argv += mount.argv()
        for key, value in self.env.items():
            argv += ["-e", f"{key}={value}"]
        container_trajectory = f"{CONTAINER_RENDER}/{trajectory.relative_to(out).as_posix()}"
        return [*argv, self.image, "render",
                "--artifact-path", f"{CONTAINER_ARTIFACT}/{artifact.name}",
                "--output-dir", f"{CONTAINER_RENDER}/{RGB_DIR}",
                "--custom-rig-trajectory", container_trajectory, "--no-replicate-training-views",
                "--camera-id", SENSOR_NAME, "--image-scale", "1.0", "--image-format", "png",
                "--frame-naming", "contiguous-output-index"]  # fmt: skip

    def check_host(self) -> None:
        """Check that this host can run the container.

        Raises:
            RenderHostError: Without the docker executable or an NVIDIA GPU (``nvidia-smi``).
        """
        needs = (f"rendering runs {self.image} with docker on an NVIDIA GPU host (NVIDIA "
                 "Container Toolkit, NGC access to the image)")  # fmt: skip
        if shutil.which(self.docker) is None:
            raise RenderHostError(f"docker executable {self.docker!r} not found: {needs}")
        if shutil.which("nvidia-smi") is None:
            raise RenderHostError(f"no NVIDIA GPU found (nvidia-smi is missing): {needs}")

    def render(
        self,
        scene: NuRecScene,
        trajectory: Path,
        out: Path,
        *,
        outputs: Sequence[str] = ("rgb",),
    ) -> RenderResult:
        """Render ``trajectory`` into ``out/rgb/camera1`` and verify the frames.

        Raises:
            CapabilityError: For an output other than RGB.
            ValueError: For a trajectory outside ``out``, or partial or missized output.
            RenderHostError: Without docker or a GPU.
            FileExistsError: When ``out/rgb`` already holds files.
            RenderFailedError: When the container exits with an error.
        """
        for output in outputs:
            scene.require(output)
        argv = self.command(scene, trajectory, out)
        self.check_host()
        rgb = Path(out) / RGB_DIR
        if rgb.exists() and any(rgb.iterdir()):
            raise FileExistsError(f"{rgb} already holds a render (unfinished or of other inputs); "
                                  "rerun `doper prepare --stage render` with --force to replace "
                                  "it")  # fmt: skip
        record = json.loads(Path(trajectory).read_text())
        count = len(record["rig_trajectories"][0]["T_rig_worlds"])
        width, height = record["camera_calibrations"][CAMERA_ID]["camera_model"]["parameters"][
            "resolution"]  # fmt: skip
        status = self.runner(argv)
        if status != 0:
            raise RenderFailedError(f"the NRE render exited with exit status {status}: "
                                    f"{' '.join(argv)}")  # fmt: skip
        return RenderResult.collect(rgb / SENSOR_NAME, count=count, width=width, height=height)


__all__ = [
    "EXCLUSION",
    "FRAME_INTERVAL_US",
    "MAX_CONSECUTIVE_REJECTIONS",
    "Mount",
    "NovelView",
    "NreCliBackend",
    "RenderBackend",
    "RenderFailedError",
    "RenderHostError",
    "RenderResult",
    "sample_novel_views",
    "write_rig_trajectory",
]
