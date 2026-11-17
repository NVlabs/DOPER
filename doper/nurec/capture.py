# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture bundles of a NuRec adaptation: images, camera, robot reference and normalization.

A capture bundle is a directory:

* ``session.json``: ``{"camera": {...}}`` (and an optional ``description``). The camera is a
  pinhole at the image resolution: ``width``, ``height``, ``fx``, ``fy``, ``cx``, ``cy`` in
  pixels and ``rectified: true``; an optional ``distortion`` list must be all zeros. Images
  with lens distortion are undistorted before they enter a bundle.
* ``images.jsonl``: one ``{"name", "sha256", "lineage"}`` row per image in capture order. The
  file is ``images/<name>``; ``lineage`` names the capture session the image belongs to (the
  adaptation split groups views by it).
* ``exclude.json`` (optional): a JSON list of image names the relative-pose metric leaves out.
* ``robot.jsonl`` and ``handeye.json`` (optional, together the robot reference): per image
  ``{"name", "T_base_mount"}`` (``T_base_link`` is accepted as an alias) and
  ``{"T_mount_camera", "camera", "validation"}`` (``camera`` and ``validation`` optional).
  Seeding mode ``robot_handeye`` and the relative-pose evaluation need it; the other modes do
  not.
* ``keypoints_3d.json`` and ``registration.json`` (optional): the object's keypoint layout and
  a reviewed CAD-to-reconstruction registration, which later stages read. Like the other
  bundle files they enter the bundle identity (``bundle_sha256``).

Frames and units: ``T_a_b`` maps b-coordinates into a. ``T_base_mount`` is the pose of the
camera mount (the robot link the camera is fixed to) in the robot base frame;
``T_mount_camera`` maps OpenCV camera coordinates (x right, y down, z forward) into the mount
frame. Translations are meters. The camera pose in the base frame is
``T_base_camera = T_base_mount @ T_mount_camera``.

The hand-eye ``camera`` block is the camera the calibration was estimated with. When it is
not the capture camera (another resolution or intrinsics, for example another sensor mode),
``capture_flags`` records ``handeye_camera_mismatch`` with both matrices, and the hand-eye
transform is used as is.

``normalize_images`` writes the resized copies every later stage reads, with ``frames.json``
binding each normalized name to its source name and SHA-256. Later stages bind by source name
through that map, never by index or file time.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..bop.sources import canonical_sha256, check_keys, sha256_file
from ..keypoints import KEYPOINTS_NAME
from ..reconstruction.geometry import as_rigid

SESSION_NAME = "session.json"
IMAGES_NAME = "images.jsonl"
EXCLUDE_NAME = "exclude.json"
ROBOT_NAME = "robot.jsonl"
HANDEYE_NAME = "handeye.json"
IMAGES_DIR = "images"
FRAMES_NAME = "frames.json"
REGISTRATION_NAME = "registration.json"
BUNDLE_FILES = (
    SESSION_NAME, IMAGES_NAME, EXCLUDE_NAME, ROBOT_NAME, HANDEYE_NAME, KEYPOINTS_NAME,
    REGISTRATION_NAME,
)  # fmt: skip
# The long edge of the normalized frames the reconstruction runs on.
NORMALIZED_LONG_EDGE = 1280
# Normalized copies are resized with OpenCV area interpolation and written as PNG.
INTERPOLATION = "INTER_AREA"
FRAME_PATTERN = "frame-{index:06d}.png"
SESSION_KEYS = frozenset({"camera", "description"})
CAMERA_KEYS = frozenset({"width", "height", "fx", "fy", "cx", "cy", "rectified"})
IMAGE_KEYS = frozenset({"name", "sha256", "lineage"})
ROBOT_POSE_KEYS = ("T_base_mount", "T_base_link")
HANDEYE_KEYS = frozenset({"T_mount_camera", "camera", "validation"})


@dataclass(frozen=True)
class PinholeCamera:
    """A pinhole camera at its image resolution, without distortion.

    Attributes:
        width: Image width, pixels.
        height: Image height, pixels.
        fx: Horizontal focal length, pixels.
        fy: Vertical focal length, pixels.
        cx: Principal point x, pixels.
        cy: Principal point y, pixels.
        rectified: Whether the images are rectified (zero distortion).
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    rectified: bool

    def __post_init__(self) -> None:
        """Check and normalize the values.

        Raises:
            ValueError: For a non-positive size or focal length or a non-finite value.
        """
        for name in ("width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value <= 0:
                raise ValueError(f"camera {name} must be a positive integer, got {value!r}")
            object.__setattr__(self, name, int(value))
        for name in ("fx", "fy", "cx", "cy"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"camera {name} must be finite, got {value!r}")
            object.__setattr__(self, name, value)
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError(f"camera focal lengths must be positive: {self.fx}, {self.fy}")
        if not isinstance(self.rectified, bool):
            raise ValueError(f"camera rectified must be true or false, got {self.rectified!r}")

    @property
    def matrix(self) -> NDArray[np.float64]:
        """The 3x3 intrinsic matrix K."""
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def as_record(self) -> dict[str, Any]:
        """The camera as its ``session.json`` block."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def describe(self) -> dict[str, Any]:
        """The camera block with its intrinsic matrix ``K`` (for receipts and flags)."""
        return {**self.as_record(), "K": self.matrix.tolist()}

    def resized(self, width: int, height: int) -> PinholeCamera:
        """The camera of the images resized to ``width`` x ``height``.

        Focal lengths and the principal point scale by the per-axis size ratio
        (``cx' = cx * width' / width``).
        """
        sx, sy = width / self.width, height / self.height
        return PinholeCamera(
            width, height, self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy, self.rectified
        )


@dataclass(frozen=True)
class CaptureImage:
    """One capture image.

    Attributes:
        name: File name under ``images/``.
        sha256: SHA-256 of the file.
        lineage: Capture session the image belongs to.
    """

    name: str
    sha256: str
    lineage: str


@dataclass(frozen=True)
class ImageSet:
    """The images of a capture bundle and their camera.

    Attributes:
        root: Bundle directory.
        camera: Capture camera (rectified, zero distortion).
        images: Images in capture order.
        exclude: Names the relative-pose metric leaves out.
    """

    root: Path
    camera: PinholeCamera
    images: tuple[CaptureImage, ...]
    exclude: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        """Image names in capture order."""
        return tuple(image.name for image in self.images)

    def image_path(self, name: str) -> Path:
        """The file of one image."""
        return self.root / IMAGES_DIR / name


@dataclass(frozen=True)
class RobotReference:
    """Robot poses and the hand-eye calibration of a capture bundle.

    Attributes:
        T_base_mount: Image name to the mount pose in the robot base frame (4x4, meters), in
            capture order.
        T_mount_camera: Hand-eye transform, OpenCV camera into mount (4x4, meters).
        handeye_camera: Camera the hand-eye was calibrated with, when the bundle records it.
        validation: The calibration's validation fields, as recorded.
    """

    T_base_mount: Mapping[str, NDArray[np.float64]]
    T_mount_camera: NDArray[np.float64]
    handeye_camera: PinholeCamera | None = None
    validation: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizedFrame:
    """One normalized image bound to its source.

    Attributes:
        name: Normalized file name.
        source: Source image name.
        source_sha256: SHA-256 of the source image.
        sha256: SHA-256 of the normalized file.
        lineage: Capture lineage of the source.
    """

    name: str
    source: str
    source_sha256: str
    sha256: str
    lineage: str


@dataclass(frozen=True)
class FrameMap:
    """Normalized images and their binding to the sources.

    Attributes:
        frames: Normalized frames in capture order.
        camera: Camera of the normalized images.
        source_camera: Camera of the source images.
        max_long_edge: Long-edge limit of the normalization, pixels.
    """

    frames: tuple[NormalizedFrame, ...]
    camera: PinholeCamera
    source_camera: PinholeCamera
    max_long_edge: int

    def by_source(self, name: str) -> NormalizedFrame:
        """The normalized frame of a source image.

        Raises:
            KeyError: For a name that is not a source of the map.
        """
        for frame in self.frames:
            if frame.source == name:
                return frame
        raise KeyError(f"no normalized frame of source image {name!r}")


def _read_json(path: Path) -> Any:  # noqa: ANN401
    with path.open() as stream:
        return json.load(stream)


def _read_jsonl(path: Path) -> list[Any]:
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number} is not JSON: {error}") from error
    return rows


def _camera(record: Any, where: str) -> PinholeCamera:  # noqa: ANN401
    """Parse a camera block; any recorded distortion must be zero."""
    check_keys(record, CAMERA_KEYS | {"distortion"}, sorted(CAMERA_KEYS), where)
    distortion = record.get("distortion", [])
    if not isinstance(distortion, list) or any(float(d) != 0.0 for d in distortion):
        raise ValueError(
            f"{where} distortion must be zero, got {distortion!r}: undistort the images first"
        )
    return PinholeCamera(**{key: record[key] for key in CAMERA_KEYS})


def _image_name(name: Any, where: str) -> str:  # noqa: ANN401
    if (
        not isinstance(name, str)
        or name in ("", ".", "..")
        or Path(name).name != name
        or "\\" in name
    ):
        raise ValueError(f"{where}: image name must be a plain file name, got {name!r}")
    return name


def _images(path: Path) -> tuple[CaptureImage, ...]:
    images, seen = [], set()
    for number, row in enumerate(_read_jsonl(path), 1):
        where = f"{path}:{number}"
        check_keys(row, IMAGE_KEYS, sorted(IMAGE_KEYS), where)
        name = _image_name(row["name"], where)
        if name in seen:
            raise ValueError(f"{where}: image {name} is listed twice")
        seen.add(name)
        sha = row["sha256"]
        if not isinstance(sha, str) or len(sha) != 64 or set(sha) - set("0123456789abcdef"):
            raise ValueError(f"{where}: sha256 must be a lowercase hex SHA-256, got {sha!r}")
        if not isinstance(row["lineage"], str) or not row["lineage"]:
            raise ValueError(f"{where}: lineage must be a non-empty string")
        images.append(CaptureImage(name, sha, row["lineage"]))
    if not images:
        raise ValueError(f"{path} lists no images")
    return tuple(images)


def _exclude(root: Path, names: Sequence[str]) -> tuple[str, ...]:
    path = root / EXCLUDE_NAME
    if not path.exists():
        return ()
    exclude = _read_json(path)
    if not isinstance(exclude, list) or not all(isinstance(e, str) for e in exclude):
        raise ValueError(f"{path} must be a JSON list of image names")
    unknown = sorted(set(exclude) - set(names))
    if unknown:
        raise ValueError(f"{path} names images that are not in the bundle: {unknown}")
    return tuple(exclude)


def verify_images(image_set: ImageSet) -> None:
    """Check every image's SHA-256 and size against the bundle.

    Raises:
        FileNotFoundError: For a missing image.
        ValueError: For a SHA-256 or image size that differs from the bundle's.
    """
    from PIL import Image

    camera = image_set.camera
    for image in image_set.images:
        path = image_set.image_path(image.name)
        if not path.is_file():
            raise FileNotFoundError(f"capture image {path} is missing")
        if sha256_file(path) != image.sha256:
            raise ValueError(f"{path} SHA-256 differs from {IMAGES_NAME}")
        with Image.open(path) as opened:
            size = opened.size
        if size != (camera.width, camera.height):
            raise ValueError(
                f"{path} size {size[0]}x{size[1]} differs from the session camera "
                f"{camera.width}x{camera.height}"
            )


def load_image_set(root: str | Path, *, verify: bool = True) -> ImageSet:
    """Load the images of a capture bundle (``session.json``, ``images.jsonl``).

    Args:
        root: Bundle directory.
        verify: Check every image's SHA-256 and size (``verify_images``).

    Returns:
        The image set.

    Raises:
        FileNotFoundError: For a missing bundle file or image.
        ValueError: For unknown or missing keys, a camera that is not rectified or has
            distortion, a malformed or duplicate image row, an exclusion of an unknown image,
            or (``verify``) an image whose SHA-256 or size differs.
    """
    root = Path(root)
    session = _read_json(root / SESSION_NAME)
    check_keys(session, SESSION_KEYS, ["camera"], str(root / SESSION_NAME))
    camera = _camera(session["camera"], f"{root / SESSION_NAME} camera")
    if not camera.rectified:
        raise ValueError(
            f"{root / SESSION_NAME}: the capture camera must be rectified (zero distortion); "
            "undistort the images and record the rectified intrinsics"
        )
    images = _images(root / IMAGES_NAME)
    image_set = ImageSet(root, camera, images, _exclude(root, [i.name for i in images]))
    if verify:
        verify_images(image_set)
    return image_set


def has_robot_reference(root: str | Path) -> bool:
    """Whether a bundle carries a robot reference (``robot.jsonl`` or ``handeye.json``)."""
    return any((Path(root) / name).exists() for name in (ROBOT_NAME, HANDEYE_NAME))


def _rigid(matrix: Any, where: str) -> NDArray[np.float64]:  # noqa: ANN401
    return as_rigid(
        matrix, where, not_finite="must be a finite rigid 4x4 transform",
        not_rigid="must be a rigid 4x4 transform (proper rotation, last row 0001)",
    )  # fmt: skip


def _robot_poses(path: Path, image_set: ImageSet) -> dict[str, NDArray[np.float64]]:
    poses: dict[str, NDArray[np.float64]] = {}
    names = set(image_set.names)
    for number, row in enumerate(_read_jsonl(path), 1):
        where = f"{path}:{number}"
        check_keys(row, frozenset({"name", *ROBOT_POSE_KEYS}), ["name"], where)
        given = [key for key in ROBOT_POSE_KEYS if key in row]
        if len(given) != 1:
            raise ValueError(f"{where} needs exactly one of {ROBOT_POSE_KEYS}, got {given}")
        name = row["name"]
        if name not in names:
            raise ValueError(f"{where}: {name!r} is not an image of the bundle")
        if name in poses:
            raise ValueError(f"{where}: {name} is listed twice")
        poses[name] = _rigid(row[given[0]], f"{where} {given[0]}")
    missing = [name for name in image_set.names if name not in poses]
    if missing:
        raise ValueError(f"{path} has no robot pose for images {missing}")
    return {name: poses[name] for name in image_set.names}


def load_robot_reference(root: str | Path, image_set: ImageSet) -> RobotReference:
    """Load the robot reference of a capture bundle (``robot.jsonl``, ``handeye.json``).

    Args:
        root: Bundle directory.
        image_set: The bundle's images; the robot poses must cover exactly these.

    Returns:
        The robot reference.

    Raises:
        FileNotFoundError: When the bundle has no robot reference.
        ValueError: For a pose that is missing, unknown, duplicated or not rigid, or unknown
            keys.
    """
    root = Path(root)
    for name in (ROBOT_NAME, HANDEYE_NAME):
        if not (root / name).is_file():
            raise FileNotFoundError(
                f"{root / name} is missing: the robot reference ({ROBOT_NAME} and {HANDEYE_NAME})"
                " is needed by seed mode robot_handeye and the relative-pose evaluation"
            )
    poses = _robot_poses(root / ROBOT_NAME, image_set)
    handeye = _read_json(root / HANDEYE_NAME)
    where = str(root / HANDEYE_NAME)
    check_keys(handeye, HANDEYE_KEYS, ["T_mount_camera"], where)
    camera = handeye.get("camera")
    validation = handeye.get("validation", {})
    if not isinstance(validation, Mapping):
        raise ValueError(f"{where} validation must be a JSON object")
    return RobotReference(
        T_base_mount=poses,
        T_mount_camera=_rigid(handeye["T_mount_camera"], f"{where} T_mount_camera"),
        handeye_camera=None if camera is None else _camera(camera, f"{where} camera"),
        validation=dict(validation),
    )


def reference_camera_poses(reference: RobotReference) -> dict[str, NDArray[np.float64]]:
    """Camera poses in the robot base frame, ``T_base_camera = T_base_mount @ T_mount_camera``.

    Returns:
        Image name to the 4x4 ``T_base_camera`` (meters), in capture order.
    """
    return {name: pose @ reference.T_mount_camera for name, pose in reference.T_base_mount.items()}


def capture_flags(image_set: ImageSet, reference: RobotReference | None) -> dict[str, Any]:
    """The capture's camera conditions that later stages and summaries record.

    Returns:
        ``distortion`` (always ``zero``: the bundle refuses others), ``robot_reference``,
        ``handeye_camera_mismatch`` (None without a robot reference or a hand-eye camera
        block, else whether that camera differs from the capture camera), ``capture_camera``
        and ``handeye_camera`` (each with its ``K``; the latter None when unknown).
    """
    handeye = reference.handeye_camera if reference is not None else None
    return {
        "distortion": "zero",
        "robot_reference": reference is not None,
        "handeye_camera_mismatch": None if handeye is None else handeye != image_set.camera,
        "capture_camera": image_set.camera.describe(),
        "handeye_camera": None if handeye is None else handeye.describe(),
    }


def bundle_files_sha256(root: str | Path) -> dict[str, str]:
    """SHA-256 of each bundle file present (the images are pinned by ``images.jsonl``)."""
    root = Path(root)
    return {name: sha256_file(root / name) for name in BUNDLE_FILES if (root / name).is_file()}


def bundle_sha256(root: str | Path) -> str:
    """The bundle's identity: the canonical SHA-256 of ``bundle_files_sha256``."""
    return canonical_sha256(bundle_files_sha256(root))


def profile_bundle_pin(profile: str) -> str | None:
    """The bundle identity a packaged NuRec profile pins (``bundle.sha256``).

    Returns:
        The pinned identity, or None for a profile that is not packaged (a user's own capture)
        or pins none.
    """
    from ..recipes import profile_path

    path = profile_path(profile)
    if not path.is_file():
        return None
    return json.loads(path.read_text())["bundle"]["sha256"]


def verify_bundle_pin(root: str | Path, pin: str | None, profile: str) -> str:
    """The bundle's identity, checked against the pin its profile resolved to.

    Args:
        root: Capture bundle.
        pin: The identity to require (``profile_bundle_pin``), or None for no pin.
        profile: The profile's name, for the error.

    Returns:
        The bundle's identity (``bundle_sha256``).

    Raises:
        ValueError: For a bundle whose identity is not the pinned one.
    """
    identity = bundle_sha256(root)
    if pin is not None and identity != pin:
        raise ValueError(f"{root} is not the {profile} bundle: identity {identity}, pinned {pin}")
    return identity


def _normalized_size(camera: PinholeCamera, max_long_edge: int) -> tuple[int, int]:
    scale = min(1.0, max_long_edge / max(camera.width, camera.height))
    return round(camera.width * scale), round(camera.height * scale)


def _write_frame(source: Path, sha256: str, camera: PinholeCamera, size: tuple[int, int],
                 out: Path) -> str:  # fmt: skip
    """Check a source image as ``verify_images`` does, from one read, and write its copy."""
    import cv2

    if not source.is_file():
        raise FileNotFoundError(f"capture image {source} is missing")
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f"{source} SHA-256 differs from {IMAGES_NAME}")
    flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    pixels = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flags)
    if pixels is None:
        raise ValueError(f"{source} is not a readable image")
    if (pixels.shape[1], pixels.shape[0]) != (camera.width, camera.height):
        raise ValueError(
            f"{source} size {pixels.shape[1]}x{pixels.shape[0]} differs from the session camera "
            f"{camera.width}x{camera.height}"
        )
    if (pixels.shape[1], pixels.shape[0]) != size:
        pixels = cv2.resize(pixels, size, interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(str(out), pixels):
        raise OSError(f"could not write {out}")
    return sha256_file(out)


def _frame_map_record(frames: FrameMap) -> dict[str, Any]:
    return {
        "max_long_edge": frames.max_long_edge,
        "interpolation": INTERPOLATION,
        "source_camera": frames.source_camera.as_record(),
        "camera": frames.camera.as_record(),
        "frames": [
            {"name": f.name, "source": f.source, "source_sha256": f.source_sha256,
             "sha256": f.sha256, "lineage": f.lineage}
            for f in frames.frames
        ],
    }  # fmt: skip


def normalize_images(
    image_set: ImageSet,
    out: str | Path,
    *,
    max_long_edge: int = NORMALIZED_LONG_EDGE,
    force: bool = False,
) -> FrameMap:
    """Write the normalized copies of a capture and ``frames.json`` binding them to sources.

    Each image is resized with OpenCV ``INTER_AREA`` so its long edge is at most
    ``max_long_edge`` (never upscaled; 1280 by default) and
    written as ``frame-NNNNNN.png``, numbered in the image set's capture order (not by file
    time). Images are decoded ignoring EXIF orientation, so pixels stay in the camera's frame.
    Each source is checked as ``verify_images`` checks it (present, SHA-256, size) from the
    same read that decodes it, so an image set loaded with ``verify=False`` is still verified.

    Args:
        image_set: The capture images.
        out: Output directory; it must not exist unless ``force``.
        max_long_edge: Long-edge limit, pixels.
        force: Replace an existing ``out``.

    Returns:
        The frame map (also written to ``out/frames.json``).

    Raises:
        FileExistsError: When ``out`` exists and ``force`` is false.
        FileNotFoundError: For a missing source.
        ValueError: For a source whose SHA-256 or size differs or that cannot be decoded.
    """
    out = Path(out)
    if out.exists() and not force:
        raise FileExistsError(f"{out} exists; pass --force to replace it")
    size = _normalized_size(image_set.camera, max_long_edge)
    staging = out.parent / f".{out.name}.partial-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    frames = []
    for index, image in enumerate(image_set.images):
        name = FRAME_PATTERN.format(index=index)
        source = image_set.image_path(image.name)
        sha = _write_frame(source, image.sha256, image_set.camera, size, staging / name)
        frames.append(NormalizedFrame(name, image.name, image.sha256, sha, image.lineage))
    frame_map = FrameMap(
        tuple(frames), image_set.camera.resized(*size), image_set.camera, max_long_edge
    )
    (staging / FRAMES_NAME).write_text(json.dumps(_frame_map_record(frame_map), indent=2) + "\n")
    if out.exists():
        shutil.rmtree(out)
    staging.rename(out)
    return frame_map


def load_frame_map(out: str | Path, *, verify: bool = True) -> FrameMap:
    """Load ``frames.json`` of a normalized capture.

    Args:
        out: The normalized directory.
        verify: Check each normalized file's SHA-256.

    Raises:
        ValueError: For a normalized file whose SHA-256 differs from the map.
    """
    out = Path(out)
    record = _read_json(out / FRAMES_NAME)
    frames = tuple(
        NormalizedFrame(f["name"], f["source"], f["source_sha256"], f["sha256"], f["lineage"])
        for f in record["frames"]
    )
    if verify:
        for frame in frames:
            path = out / frame.name
            if not path.is_file() or sha256_file(path) != frame.sha256:
                raise ValueError(f"normalized frame {path} is missing or differs from the map")
    return FrameMap(
        frames,
        PinholeCamera(**record["camera"]),
        PinholeCamera(**record["source_camera"]),
        int(record["max_long_edge"]),
    )


__all__ = [
    "INTERPOLATION",
    "NORMALIZED_LONG_EDGE",
    "CaptureImage",
    "FrameMap",
    "ImageSet",
    "NormalizedFrame",
    "PinholeCamera",
    "RobotReference",
    "bundle_files_sha256",
    "bundle_sha256",
    "capture_flags",
    "has_robot_reference",
    "load_frame_map",
    "load_image_set",
    "load_robot_reference",
    "normalize_images",
    "profile_bundle_pin",
    "reference_camera_poses",
    "verify_bundle_pin",
    "verify_images",
]
