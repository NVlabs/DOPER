# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""COLMAP 3.13 text models: ``cameras.txt``, ``images.txt``, ``points3D.txt`` and rigs/frames.

The formats (COLMAP's own comment headers say the same):

* ``cameras.txt``: ``CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]``.
* ``images.txt``: two lines per image, ``IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME`` and then
  the 2D points as ``X Y POINT3D_ID`` triples (``-1`` for none). The second line may be empty.
* ``points3D.txt``: ``POINT3D_ID X Y Z R G B ERROR TRACK[]``, the track as
  ``IMAGE_ID POINT2D_IDX`` pairs.
* ``rigs.txt`` (3.12+): ``RIG_ID NUM_SENSORS REF_SENSOR_TYPE REF_SENSOR_ID``, then per
  non-reference sensor ``SENSOR_TYPE SENSOR_ID HAS_POSE [QW QX QY QZ TX TY TZ]``.
* ``frames.txt`` (3.12+): ``FRAME_ID RIG_ID QW QX QY QZ TX TY TZ NUM_DATA_IDS`` and the data ids
  as ``SENSOR_TYPE SENSOR_ID DATA_ID`` triples.

A model without ``rigs.txt`` and ``frames.txt`` (a seed, or a pre-3.12 model) is a legacy
model; COLMAP 3.13 reads it with one trivial rig and frame per camera and image.

Frames and units: an image pose is ``cam_from_world`` (COLMAP's world-to-camera), OpenCV camera
axes (x right, y down, z forward); quaternions are ``wxyz`` in the text files (pycolmap's
``Rotation3d.quat`` is ``xyzw``). Quaternions keep the sign they were read with. Lengths are in
the model's own units (see ``reconstruction.json``).

Floats are written with Python's shortest round-trip ``repr``, so write-then-read returns the
same values; the bytes differ from COLMAP's 17-digit output.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..nurec.capture import PinholeCamera

CAMERAS_NAME = "cameras.txt"
IMAGES_NAME = "images.txt"
POINTS_NAME = "points3D.txt"
RIGS_NAME = "rigs.txt"
FRAMES_NAME = "frames.txt"
SEED_CAMERA_ID = 1
SEED_CAMERA_MODEL = "OPENCV"
ROTATION_ATOL = 1e-6
CAMERAS_HEADER = (
    "# Camera list with one line of data per camera:\n"
    "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
)
IMAGES_HEADER = (
    "# Image list with two lines of data per image:\n"
    "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
    "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
)
POINTS_HEADER = (
    "# 3D point list with one line of data per point:\n"
    "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
)
RIGS_HEADER = (
    "# Rig calib list with one line of data per calib:\n"
    "#   RIG_ID, NUM_SENSORS, REF_SENSOR_TYPE, REF_SENSOR_ID, SENSORS[] as (SENSOR_TYPE, "
    "SENSOR_ID, HAS_POSE, [QW, QX, QY, QZ, TX, TY, TZ])\n"
)
FRAMES_HEADER = (
    "# Frame list with one line of data per frame:\n"
    "#   FRAME_ID, RIG_ID, RIG_FROM_WORLD[QW, QX, QY, QZ, TX, TY, TZ], NUM_DATA_IDS, "
    "DATA_IDS[] as (SENSOR_TYPE, SENSOR_ID, DATA_ID)\n"
)

Quaternion = tuple[float, float, float, float]
Vector3 = tuple[float, float, float]
Sensor = tuple[str, int]
SensorPose = tuple[Quaternion, Vector3]


@dataclass(frozen=True)
class ColmapCamera:
    """One camera of a model.

    Attributes:
        camera_id: Camera id.
        model: COLMAP camera model name (``OPENCV``, ``PINHOLE``, ...).
        width: Image width, pixels.
        height: Image height, pixels.
        params: Model parameters in COLMAP's order (``OPENCV``: fx fy cx cy k1 k2 p1 p2).
    """

    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True)
class ColmapImage:
    """One registered image of a model.

    Attributes:
        image_id: Image id (the database's).
        qvec_wxyz: Rotation of ``cam_from_world`` as a ``wxyz`` quaternion, sign as read.
        tvec: Translation of ``cam_from_world``, model units.
        camera_id: Camera id.
        name: Image name (relative to the image directory).
        points2d: The image's keypoints as ``(x, y, point3D id)``; the id is ``-1`` for none.
    """

    image_id: int
    qvec_wxyz: Quaternion
    tvec: Vector3
    camera_id: int
    name: str
    points2d: tuple[tuple[float, float, int], ...] = ()


@dataclass(frozen=True)
class ColmapPoint3D:
    """One triangulated point.

    Attributes:
        point3d_id: Point id.
        xyz: Position in the world frame, model units.
        rgb: Color, 8-bit.
        error: Mean reprojection error, pixels.
        track: Observations as ``(image id, point2D index)``.
    """

    point3d_id: int
    xyz: Vector3
    rgb: tuple[int, int, int]
    error: float
    track: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ColmapRig:
    """One rig (COLMAP 3.12+).

    Attributes:
        rig_id: Rig id.
        ref_sensor: The reference sensor, ``(type, id)``.
        sensors: The other sensors as ``(type, id, sensor_from_rig)``; the pose (``wxyz``
            quaternion, translation) is None when the rig does not know it.
    """

    rig_id: int
    ref_sensor: Sensor
    sensors: tuple[tuple[str, int, SensorPose | None], ...] = ()


@dataclass(frozen=True)
class ColmapFrame:
    """One frame (COLMAP 3.12+): a rig pose and the data captured at it.

    Attributes:
        frame_id: Frame id.
        rig_id: Rig id.
        qvec_wxyz: Rotation of ``rig_from_world``, ``wxyz``, sign as read.
        tvec: Translation of ``rig_from_world``, model units.
        data_ids: ``(sensor type, sensor id, data id)``; for a camera the data id is the image id.
    """

    frame_id: int
    rig_id: int
    qvec_wxyz: Quaternion
    tvec: Vector3
    data_ids: tuple[tuple[str, int, int], ...]


@dataclass(frozen=True)
class ColmapModel:
    """A COLMAP model, keyed by id in file order.

    Attributes:
        cameras: Camera id to camera.
        images: Image id to registered image.
        points3d: Point id to point.
        rigs: Rig id to rig; None for a legacy model (no ``rigs.txt``).
        frames: Frame id to frame; None for a legacy model (no ``frames.txt``).
    """

    cameras: Mapping[int, ColmapCamera]
    images: Mapping[int, ColmapImage]
    points3d: Mapping[int, ColmapPoint3D]
    rigs: Mapping[int, ColmapRig] | None = None
    frames: Mapping[int, ColmapFrame] | None = None


@dataclass(frozen=True)
class SeedImage:
    """One image of a seed model.

    Attributes:
        image_id: The image's id in the COLMAP database the seed is used with.
        name: The image's name in that database.
        cam_from_world: 4x4 world-to-camera transform (OpenCV camera axes), seed world units.
    """

    image_id: int
    name: str
    cam_from_world: NDArray[np.float64] = field(repr=False)


def quaternion_matrix(qvec_wxyz: Sequence[float]) -> NDArray[np.float64]:
    """The 3x3 rotation of a ``wxyz`` quaternion (normalized first; ``q`` and ``-q`` agree).

    Raises:
        ValueError: For a zero or non-finite quaternion.
    """
    q = np.asarray(qvec_wxyz, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if q.shape != (4,) or not math.isfinite(norm) or norm == 0.0:
        raise ValueError(f"quaternion must be four finite values, not all zero: {qvec_wxyz!r}")
    w, x, y, z = q / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip


def quaternion_wxyz(rotation: ArrayLike) -> Quaternion:
    """The ``wxyz`` unit quaternion of a 3x3 rotation.

    Raises:
        ValueError: For a matrix that is not a proper rotation.
    """
    from scipy.spatial.transform import Rotation

    matrix = np.asarray(rotation, dtype=np.float64)
    is_rotation = (
        matrix.shape == (3, 3)
        and np.allclose(matrix.T @ matrix, np.eye(3), atol=ROTATION_ATOL)
        and np.linalg.det(matrix) > 0.0
    )
    if not is_rotation:
        raise ValueError("expected a proper 3x3 rotation matrix")
    x, y, z, w = Rotation.from_matrix(matrix).as_quat()
    return float(w), float(x), float(y), float(z)


def cam_from_world(image: ColmapImage) -> NDArray[np.float64]:
    """The 4x4 ``cam_from_world`` of an image (OpenCV camera axes, model units)."""
    pose = np.eye(4)
    pose[:3, :3] = quaternion_matrix(image.qvec_wxyz)
    pose[:3, 3] = image.tvec
    return pose


# --- reading -----------------------------------------------------------------------------------


def _data_lines(path: Path) -> list[tuple[int, str]]:
    """Numbered lines of a text file without comment lines (blank lines are kept)."""
    lines = path.read_text().split("\n")
    return [(n, line.strip()) for n, line in enumerate(lines, 1) if not line.startswith("#")]


def _parse(where: str, convert: Any, tokens: Sequence[str]) -> Any:  # noqa: ANN401
    try:
        return convert(tokens)
    except (ValueError, IndexError) as error:
        raise ValueError(f"{where}: malformed line ({error})") from error


def _floats(tokens: Sequence[str]) -> tuple[float, ...]:
    return tuple(float(token) for token in tokens)


def _pose(tokens: Sequence[str]) -> SensorPose:
    values = _floats(tokens)
    if len(values) != 7:
        raise ValueError("a pose needs QW QX QY QZ TX TY TZ")
    return (values[0], values[1], values[2], values[3]), (values[4], values[5], values[6])


def _put(items: dict[int, Any], key: int, value: Any, where: str) -> None:  # noqa: ANN401
    if key in items:
        raise ValueError(f"{where}: id {key} is listed twice")
    items[key] = value


def _camera(tokens: Sequence[str]) -> ColmapCamera:
    if len(tokens) < 4:
        raise ValueError("expected CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]")
    return ColmapCamera(int(tokens[0]), tokens[1], int(tokens[2]), int(tokens[3]),
                        _floats(tokens[4:]))  # fmt: skip


def _read_cameras(path: Path) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    for number, line in _data_lines(path):
        if line:
            where = f"{path.name}:{number}"
            camera = _parse(where, _camera, line.split())
            _put(cameras, camera.camera_id, camera, where)
    return cameras


def _image_head(tokens: Sequence[str]) -> tuple[int, Quaternion, Vector3, int, str]:
    if len(tokens) != 10:
        raise ValueError("expected IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME")
    q, t = _pose(tokens[1:8])
    return int(tokens[0]), q, t, int(tokens[8]), tokens[9]


def _points2d(tokens: Sequence[str]) -> tuple[tuple[float, float, int], ...]:
    if len(tokens) % 3:
        raise ValueError("2D points come as X Y POINT3D_ID triples")
    return tuple(
        (float(tokens[i]), float(tokens[i + 1]), int(tokens[i + 2]))
        for i in range(0, len(tokens), 3)
    )


def _read_images(path: Path) -> dict[int, ColmapImage]:
    lines = _data_lines(path)
    images: dict[int, ColmapImage] = {}
    index = 0
    while index < len(lines):
        number, line = lines[index]
        index += 1
        if not line:
            continue
        where = f"{path.name}:{number}"
        image_id, q, t, camera_id, name = _parse(where, _image_head, line.split())
        points_number, points_line = lines[index] if index < len(lines) else (number + 1, "")
        index += 1
        points = _parse(f"{path.name}:{points_number}", _points2d, points_line.split())
        _put(images, image_id, ColmapImage(image_id, q, t, camera_id, name, points), where)
    return images


def _point3d(tokens: Sequence[str]) -> ColmapPoint3D:
    if len(tokens) < 8 or (len(tokens) - 8) % 2:
        raise ValueError("expected POINT3D_ID X Y Z R G B ERROR and (IMAGE_ID, POINT2D_IDX) pairs")
    x, y, z = _floats(tokens[1:4])
    r, g, b = (int(token) for token in tokens[4:7])
    track = tuple((int(tokens[i]), int(tokens[i + 1])) for i in range(8, len(tokens), 2))
    return ColmapPoint3D(int(tokens[0]), (x, y, z), (r, g, b), float(tokens[7]), track)


def _read_points(path: Path) -> dict[int, ColmapPoint3D]:
    points: dict[int, ColmapPoint3D] = {}
    for number, line in _data_lines(path):
        if line:
            where = f"{path.name}:{number}"
            point = _parse(where, _point3d, line.split())
            _put(points, point.point3d_id, point, where)
    return points


def _rig(tokens: Sequence[str]) -> ColmapRig:
    rig_id, count = int(tokens[0]), int(tokens[1])
    ref = (tokens[2], int(tokens[3]))
    sensors, index = [], 4
    for _ in range(count - 1):
        kind, sensor_id, has_pose = tokens[index], int(tokens[index + 1]), int(tokens[index + 2])
        index += 3
        pose = None
        if has_pose:
            pose = _pose(tokens[index : index + 7])
            index += 7
        sensors.append((kind, sensor_id, pose))
    if count < 1 or index != len(tokens):
        raise ValueError("sensor count does not match the sensors listed")
    return ColmapRig(rig_id, ref, tuple(sensors))


def _frame(tokens: Sequence[str]) -> ColmapFrame:
    q, t = _pose(tokens[2:9])
    count = int(tokens[9])
    rest = tokens[10:]
    if len(rest) != 3 * count:
        raise ValueError("data id count does not match the data ids listed")
    data = tuple((rest[i], int(rest[i + 1]), int(rest[i + 2])) for i in range(0, len(rest), 3))
    return ColmapFrame(int(tokens[0]), int(tokens[1]), q, t, data)


def _read_records(path: Path, parse: Any, key: str) -> dict[int, Any]:  # noqa: ANN401
    records: dict[int, Any] = {}
    for number, line in _data_lines(path):
        if line:
            where = f"{path.name}:{number}"
            record = _parse(where, parse, line.split())
            _put(records, getattr(record, key), record, where)
    return records


def read_text_model(directory: str | Path) -> ColmapModel:
    """Read a COLMAP text model (``rigs.txt`` and ``frames.txt`` when present).

    Raises:
        FileNotFoundError: For a missing ``cameras.txt``, ``images.txt`` or ``points3D.txt``,
            or only one of ``rigs.txt`` and ``frames.txt``.
        ValueError: For a malformed line (named by file and line number) or a repeated id.
    """
    directory = Path(directory)
    has_rigs = [(directory / name).is_file() for name in (RIGS_NAME, FRAMES_NAME)]
    if any(has_rigs) and not all(has_rigs):
        raise FileNotFoundError(f"{directory} has only one of {RIGS_NAME} and {FRAMES_NAME}")
    rigs = frames = None
    if all(has_rigs):
        rigs = _read_records(directory / RIGS_NAME, _rig, "rig_id")
        frames = _read_records(directory / FRAMES_NAME, _frame, "frame_id")
    return ColmapModel(
        _read_cameras(directory / CAMERAS_NAME),
        _read_images(directory / IMAGES_NAME),
        _read_points(directory / POINTS_NAME),
        rigs,
        frames,
    )


# --- writing -----------------------------------------------------------------------------------


def _number(value: float) -> str:
    return repr(float(value))


def _join(values: Iterable[Any]) -> str:  # noqa: ANN401
    return " ".join(_number(v) if isinstance(v, float) else str(v) for v in values)


def _observations(model: ColmapModel) -> int:
    return sum(len(point.track) for point in model.points3d.values())


def _camera_lines(cameras: Iterable[ColmapCamera]) -> list[str]:
    return [_join([c.camera_id, c.model, c.width, c.height, *c.params]) for c in cameras]


def _image_lines(images: Iterable[ColmapImage]) -> list[str]:
    lines = []
    for image in images:
        lines.append(_join([image.image_id, *image.qvec_wxyz, *image.tvec, image.camera_id,
                            image.name]))  # fmt: skip
        lines.append(_join(value for point in image.points2d for value in point))
    return lines


def _point_lines(points: Iterable[ColmapPoint3D]) -> list[str]:
    return [
        _join([p.point3d_id, *p.xyz, *p.rgb, p.error, *(v for pair in p.track for v in pair)])
        for p in points
    ]


def _rig_lines(rigs: Iterable[ColmapRig]) -> list[str]:
    lines = []
    for rig in rigs:
        values: list[Any] = [rig.rig_id, len(rig.sensors) + 1, *rig.ref_sensor]
        for kind, sensor_id, pose in rig.sensors:
            values += [kind, sensor_id, 0 if pose is None else 1]
            values += [] if pose is None else [*pose[0], *pose[1]]
        lines.append(_join(values))
    return lines


def _frame_lines(frames: Iterable[ColmapFrame]) -> list[str]:
    return [
        _join([f.frame_id, f.rig_id, *f.qvec_wxyz, *f.tvec, len(f.data_ids),
               *(v for data in f.data_ids for v in data)])
        for f in frames
    ]  # fmt: skip


def _write(path: Path, header: str, lines: Sequence[str]) -> None:
    path.write_text(header + "".join(line + "\n" for line in lines))


def write_text_model(model: ColmapModel, directory: str | Path) -> None:
    """Write a model as COLMAP text files, with COLMAP's comment headers and counts.

    ``rigs.txt`` and ``frames.txt`` are written when the model has them (not for a legacy
    model). The directory is created when missing.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cameras, images, points = model.cameras, model.images, model.points3d
    observations = _observations(model)
    _write(directory / CAMERAS_NAME, CAMERAS_HEADER + f"# Number of cameras: {len(cameras)}\n",
           _camera_lines(cameras.values()))  # fmt: skip
    mean_observations = observations / len(images) if images else 0.0
    _write(directory / IMAGES_NAME,
           IMAGES_HEADER + f"# Number of images: {len(images)}, mean observations per image: "
           f"{_number(mean_observations)}\n", _image_lines(images.values()))  # fmt: skip
    mean_track = observations / len(points) if points else 0.0
    _write(directory / POINTS_NAME,
           POINTS_HEADER + f"# Number of points: {len(points)}, mean track length: "
           f"{_number(mean_track)}\n", _point_lines(points.values()))  # fmt: skip
    if model.rigs is not None and model.frames is not None:
        _write(directory / RIGS_NAME, RIGS_HEADER + f"# Number of rigs: {len(model.rigs)}\n",
               _rig_lines(model.rigs.values()))  # fmt: skip
        _write(directory / FRAMES_NAME,
               FRAMES_HEADER + f"# Number of frames: {len(model.frames)}\n",
               _frame_lines(model.frames.values()))  # fmt: skip


def _seed_params(camera: PinholeCamera) -> str:
    """``OPENCV`` params of a rectified pinhole: ``fx fy cx cy 0 0 0 0``."""
    return _join([camera.fx, camera.fy, camera.cx, camera.cy, *("0",) * 4])


def write_seed(directory: str | Path, camera: PinholeCamera, images: Sequence[SeedImage]) -> None:
    """Write a seed model for ``point_triangulator``: known poses, no points.

    ``cameras.txt`` holds one ``OPENCV`` camera (id 1, zero distortion) at the images'
    resolution; ``images.txt`` one line per image (its database id and name, the
    ``cam_from_world`` as a ``wxyz`` quaternion and translation) followed by an empty 2D-points
    line; ``points3D.txt`` is empty. No rigs or frames: COLMAP 3.13 reads a legacy model.

    Raises:
        ValueError: For a repeated image id or name, or a pose that is not rigid.
    """
    ids, names = [i.image_id for i in images], [i.name for i in images]
    if len(set(ids)) != len(ids) or len(set(names)) != len(names):
        raise ValueError("seed image ids and names must be unique")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    line = f"{SEED_CAMERA_ID} {SEED_CAMERA_MODEL} {camera.width} {camera.height} "
    (directory / CAMERAS_NAME).write_text(line + _seed_params(camera) + "\n")
    lines = []
    for image in images:
        pose = np.asarray(image.cam_from_world, dtype=np.float64)
        if pose.shape != (4, 4) or not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0]):
            raise ValueError(f"seed pose of {image.name} must be a rigid 4x4 transform")
        q = quaternion_wxyz(pose[:3, :3])
        lines += [_join([image.image_id, *q, *(float(v) for v in pose[:3, 3]),
                         SEED_CAMERA_ID, image.name]), ""]  # fmt: skip
    (directory / IMAGES_NAME).write_text("".join(line + "\n" for line in lines))
    (directory / POINTS_NAME).write_text("")


def model_metrics(model: ColmapModel, total_images: int) -> dict[str, Any]:
    """Reconstruction metrics of a COLMAP model.

    Returns:
        ``total_images``, ``registered_images``, ``largest_model_fraction`` (registered /
        total), ``observations`` (track entries), ``points``, ``mean_track_length`` and
        ``median_reprojection_px`` (median of the points' errors); the means and median are
        None without points.

    Raises:
        ValueError: For a non-positive ``total_images`` or fewer than the registered images.
    """
    registered = len(model.images)
    if total_images <= 0 or registered > total_images:
        raise ValueError(f"total_images {total_images} is inconsistent with {registered} "
                         "registered images")  # fmt: skip
    points = len(model.points3d)
    observations = _observations(model)
    errors = [point.error for point in model.points3d.values()]
    return {
        "total_images": total_images,
        "registered_images": registered,
        "largest_model_fraction": registered / total_images,
        "observations": observations,
        "points": points,
        "mean_track_length": observations / points if points else None,
        "median_reprojection_px": float(statistics.median(errors)) if errors else None,
    }


__all__ = [
    "ColmapCamera",
    "ColmapFrame",
    "ColmapImage",
    "ColmapModel",
    "ColmapPoint3D",
    "ColmapRig",
    "SeedImage",
    "cam_from_world",
    "model_metrics",
    "quaternion_matrix",
    "quaternion_wxyz",
    "read_text_model",
    "write_seed",
    "write_text_model",
]
