# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``reconstruction.json`` (schema 1): what a reconstruction is, in which frame and units.

Every seeding mode (``robot_handeye``, ``colmap_sfm``, ``doper_pnp``) writes the same record
next to its COLMAP model. Later stages (scene import, rendering, labels) read it to bind each
model image to its source capture and to learn whether lengths are metric.

Frames and units: image poses are ``cam_from_world`` (COLMAP world-to-camera) with OpenCV
camera axes; quaternions in the text model are ``wxyz``; ``T_a_b`` maps b-coordinates into a.
The model stays in COLMAP units (the units NuRec trains in). ``length_unit`` is ``arbitrary``
for an unseeded reconstruction (gauge and scale are whatever SfM chose, so
``scene_units_per_metre`` is null) and ``colmap`` for a seeded one, whose ``scale_anchor`` is
the similarity that maps the metric seed camera centers onto the refined centers: its scale is
``scene_units_per_metre`` and its residuals are stored in scene units and in millimeters
(scene units / scale). The model itself is not transformed.
``colmap.warnings`` lists the quality warnings of the final model.

Per image, ``seed_rejected_reason`` says why a seeding strategy left a capture image out of
the seed (mode (c): a DOPER gate), and ``registered_by`` says what put it in the final model:
``seed`` (a seed pose), ``mapper`` (unseeded SfM) or ``image_registrator`` (mode (c)'s
completion of the images the seed left out). ``scale_check`` (mode (c)) is the independent
CAD-keypoint scale the anchor was checked against.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from ..bop.sources import check_keys, publish_json
from ..recipes.nurec_schema import SEED_MODES

SCHEMA_VERSION = 1
METADATA_NAME = "reconstruction.json"
WORLD_FRAMES = ("robot_base", "arbitrary", "object_cad")
LENGTH_UNITS = ("arbitrary", "colmap")
POSE_CONVENTION = (
    "cam_from_world (COLMAP world_to_camera), OpenCV camera axes x right, y down, z forward"
)
QUATERNION_ORDER = "wxyz"
TRANSFORM_CONVENTION = "T_parent_child maps child coordinates into parent"
CAMERA_SOURCES = ("capture session", "estimated")
REGISTERED_BY = ("seed", "mapper", "image_registrator")

Matrix = tuple[tuple[float, ...], ...]


def _matrix(value: Any, shape: tuple[int, int], where: str) -> Matrix:  # noqa: ANN401
    rows = tuple(tuple(float(v) for v in row) for row in value)
    if len(rows) != shape[0] or any(len(row) != shape[1] for row in rows):
        raise ValueError(f"{where} must be a {shape[0]}x{shape[1]} matrix")
    if not all(math.isfinite(v) for row in rows for v in row):
        raise ValueError(f"{where} must be finite")
    return rows


def _check_keys(cls: type, record: Any, where: str) -> None:  # noqa: ANN401
    names = [f.name for f in fields(cls)]
    check_keys(record, names, names, where)


def _record(value: Any) -> Any:  # noqa: ANN401
    """A dataclass tree as JSON values (tuples become lists)."""
    if hasattr(value, "__dataclass_fields__"):
        return {f.name: _record(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (list, tuple)):
        return [_record(v) for v in value]
    if isinstance(value, Mapping):
        return {k: _record(v) for k, v in value.items()}
    return value


@dataclass(frozen=True)
class ScaleAnchor:
    """The similarity from the seed camera centers (meters) to the refined ones (scene units).

    Attributes:
        method: How it was fitted (for example ``sim3_seed_to_refined_centres``).
        scene_units_per_metre: Its scale.
        rotation: Its 3x3 rotation.
        translation: Its translation, scene units.
        rms_scene: RMS center residual, scene units.
        max_scene: Largest center residual, scene units.
        rms_mm: RMS center residual, millimeters (``rms_scene / scale * 1000``).
        max_mm: Largest center residual, millimeters (``max_scene / scale * 1000``).
        n: Cameras in the fit.
    """

    method: str
    scene_units_per_metre: float
    rotation: Matrix
    translation: tuple[float, float, float]
    rms_scene: float
    max_scene: float
    rms_mm: float
    max_mm: float
    n: int

    def __post_init__(self) -> None:
        """Normalize the arrays and check the values.

        Raises:
            ValueError: For a non-positive scale or fewer than three cameras.
        """
        object.__setattr__(self, "rotation", _matrix(self.rotation, (3, 3), "rotation"))
        translation = _matrix([self.translation], (1, 3), "translation")[0]
        object.__setattr__(self, "translation", translation)
        if not math.isfinite(self.scene_units_per_metre) or self.scene_units_per_metre <= 0:
            raise ValueError(f"scale_anchor.scene_units_per_metre must be positive, got "
                             f"{self.scene_units_per_metre}")  # fmt: skip
        if self.n < 3:
            raise ValueError(f"a scale anchor needs at least 3 cameras, got {self.n}")

    def to_record(self) -> dict[str, Any]:
        """The anchor as its JSON block."""
        return _record(self)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ScaleAnchor:
        """Build the anchor from its JSON block."""
        _check_keys(cls, record, "scale_anchor")
        return cls(**record)


@dataclass(frozen=True)
class ScaleCheck:
    """An independent check of the scale anchor: CAD keypoints onto re-triangulated keypoints.

    The DOPER keypoints of the seeded images are triangulated with the refined cameras (scene
    units) and Umeyama-fitted from the CAD keypoints (meters), the anchor's direction, so its
    scale is a second ``scene_units_per_metre``.

    Attributes:
        method: How it was fitted (``sim3_cad_keypoints_to_retriangulated``).
        scene_units_per_metre: The check's scale.
        anchor_scene_units_per_metre: The anchor's scale it was compared with.
        relative_difference: ``|check - anchor| / anchor``.
        tolerance: Largest relative difference accepted.
        rms_scene: RMS keypoint residual, scene units.
        rms_mm: RMS keypoint residual, millimeters (``rms_scene / scale * 1000``).
        n: Keypoints in the fit.
    """

    method: str
    scene_units_per_metre: float
    anchor_scene_units_per_metre: float
    relative_difference: float
    tolerance: float
    rms_scene: float
    rms_mm: float
    n: int

    def __post_init__(self) -> None:
        """Check the values.

        Raises:
            ValueError: For a non-positive scale or fewer than three keypoints.
        """
        for name in ("scene_units_per_metre", "anchor_scene_units_per_metre"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"scale_check.{name} must be positive, got {value}")
        if self.n < 3:
            raise ValueError(f"a scale check needs at least 3 keypoints, got {self.n}")

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ScaleCheck:
        """Build the check from its JSON block."""
        _check_keys(cls, record, "scale_check")
        return cls(**record)


@dataclass(frozen=True)
class CameraRecord:
    """The reconstruction's camera (one shared camera).

    Attributes:
        model: COLMAP camera model.
        width: Image width, pixels.
        height: Image height, pixels.
        params: Final model parameters in COLMAP's order.
        rectified: Whether the images are rectified (zero distortion).
        source: ``capture session`` (the bundle's intrinsics) or ``estimated`` (by COLMAP).
        refined: Whether COLMAP changed the given intrinsics.
    """

    model: str
    width: int
    height: int
    params: tuple[float, ...]
    rectified: bool
    source: str
    refined: bool

    def __post_init__(self) -> None:
        """Normalize ``params`` and check ``source``.

        Raises:
            ValueError: For an unknown source.
        """
        object.__setattr__(self, "params", tuple(float(p) for p in self.params))
        if self.source not in CAMERA_SOURCES:
            raise ValueError(f"camera.source must be one of {CAMERA_SOURCES}, got {self.source!r}")


@dataclass(frozen=True)
class ImageRecord:
    """One capture image's provenance in the reconstruction.

    Attributes:
        name: Normalized image name (the COLMAP image name).
        source_name: The capture image it was made from.
        sha256: SHA-256 of the normalized image.
        source_sha256: SHA-256 of the capture image.
        colmap_image_id: Its COLMAP database id (None when not in the database).
        seeded: Whether the seed gave it a pose.
        seed_cam_from_world: The seed pose (4x4, seed units), or None.
        registered: Whether the final model registered it.
        excluded_reason: Why it is not in the final model, or None.
        seed_rejected_reason: Why the seeding strategy left it out of the seed, or None.
        registered_by: What registered it (``REGISTERED_BY``), or None when not registered.
    """

    name: str
    source_name: str
    sha256: str
    source_sha256: str
    colmap_image_id: int | None
    seeded: bool
    seed_cam_from_world: Matrix | None
    registered: bool
    excluded_reason: str | None
    seed_rejected_reason: str | None
    registered_by: str | None

    def __post_init__(self) -> None:
        """Normalize the seed pose; check that only seeded images carry one.

        Raises:
            ValueError: For a seed pose on an unseeded image or the reverse, a rejection
                reason on a seeded image, or a ``registered_by`` that does not match
                ``registered``.
        """
        pose = self.seed_cam_from_world
        if (pose is not None) != self.seeded:
            raise ValueError(f"image {self.name}: seed_cam_from_world must be given exactly when "
                             "seeded")  # fmt: skip
        if self.seeded and self.seed_rejected_reason is not None:
            raise ValueError(f"image {self.name}: a seeded image has no seed_rejected_reason")
        allowed = REGISTERED_BY if self.registered else (None,)
        if self.registered_by not in allowed:
            raise ValueError(f"image {self.name}: registered_by must be one of {allowed}, got "
                             f"{self.registered_by!r}")  # fmt: skip
        if pose is not None:
            object.__setattr__(self, "seed_cam_from_world", _matrix(pose, (4, 4), self.name))


@dataclass(frozen=True)
class ColmapRecord:
    """The COLMAP runs of a reconstruction.

    Attributes:
        version: COLMAP release.
        argv: Every call's exact argv, in order.
        metrics: ``colmap_io.model_metrics`` of the final model.
        selected_model: The final binary model, relative to the reconstruction directory.
        warnings: Quality warnings of the final model (``COLMAP_LOW_REGISTRATION``,
            ``COLMAP_HIGH_REPROJECTION``); the refusals stop the stage instead.
    """

    version: str
    argv: list[list[str]]
    metrics: dict[str, Any]
    selected_model: str
    warnings: list[str]


@dataclass(frozen=True)
class ReconstructionMetadata:
    """``reconstruction.json``, schema 1.

    Attributes:
        strategy: Seeding mode (``robot_handeye``, ``colmap_sfm``, ``doper_pnp``).
        world_frame: ``robot_base``, ``arbitrary`` or ``object_cad``.
        length_unit: ``arbitrary`` (unseeded) or ``colmap`` (seeded, anchored).
        scene_units_per_metre: The anchor's scale, or None when not metric.
        scale_anchor: Seed-to-refined similarity, or None (unseeded).
        camera: The shared camera.
        images: Every capture image, in capture order.
        colmap: Version, argv and metrics.
        inputs: Input name to SHA-256 (bundle, capture receipt, frames, flags, ...).
        capture_flags: The capture's ``capture_flags`` (distortion, hand-eye camera mismatch).
        reference_path: Whether this is the recommended reconstruction path (``robot_handeye``).
        reconstruction_id: SHA-256 of the final text model's files (canonical JSON of name to
            SHA-256); renders and labels carry it.
        model_text: The final model as text, relative to the reconstruction directory.
        scale_check: The CAD-keypoint check of the anchor's scale (mode (c)), or None.
        pose_convention: Fixed (``POSE_CONVENTION``).
        quaternion_order: Fixed (``wxyz``).
        transform_convention: Fixed (``TRANSFORM_CONVENTION``).
        schema_version: Fixed (1).
    """

    strategy: str
    world_frame: str
    length_unit: str
    scene_units_per_metre: float | None
    scale_anchor: ScaleAnchor | None
    camera: CameraRecord
    images: tuple[ImageRecord, ...]
    colmap: ColmapRecord
    inputs: dict[str, str]
    capture_flags: dict[str, Any]
    reference_path: bool
    reconstruction_id: str
    model_text: str
    scale_check: ScaleCheck | None = None
    pose_convention: str = POSE_CONVENTION
    quaternion_order: str = QUATERNION_ORDER
    transform_convention: str = TRANSFORM_CONVENTION
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Check the enumerations, the conventions and the units.

        Raises:
            ValueError: For an unknown strategy, frame or unit, a changed convention, an
                ``arbitrary`` length unit with a scale, or a ``colmap`` unit without an anchor
                or with a ``scene_units_per_metre`` other than the anchor's.
        """
        object.__setattr__(self, "images", tuple(self.images))
        checks = (("strategy", SEED_MODES), ("world_frame", WORLD_FRAMES),
                  ("length_unit", LENGTH_UNITS), ("schema_version", (SCHEMA_VERSION,)),
                  ("pose_convention", (POSE_CONVENTION,)),
                  ("quaternion_order", (QUATERNION_ORDER,)),
                  ("transform_convention", (TRANSFORM_CONVENTION,)))  # fmt: skip
        for name, allowed in checks:
            if getattr(self, name) not in allowed:
                raise ValueError(f"reconstruction {name} must be one of {allowed}, got "
                                 f"{getattr(self, name)!r}")  # fmt: skip
        if self.length_unit == "arbitrary":
            if self.scale_anchor is not None or self.scene_units_per_metre is not None:
                raise ValueError("an arbitrary length unit has no scale_anchor and no "
                                 "scene_units_per_metre")  # fmt: skip
        elif (
            self.scale_anchor is None
            or self.scene_units_per_metre != self.scale_anchor.scene_units_per_metre
        ):
            raise ValueError("a colmap length unit needs a scale_anchor, and "
                             "scene_units_per_metre must be its scale")  # fmt: skip

    @property
    def is_metric(self) -> bool:
        """Whether scene lengths convert to meters (``scene_units_per_metre`` is known)."""
        return self.scene_units_per_metre is not None

    def image_by_source(self, source_name: str) -> ImageRecord:
        """The record of one capture image.

        Raises:
            KeyError: For a name that is not a capture image of the reconstruction.
        """
        for image in self.images:
            if image.source_name == source_name:
                return image
        raise KeyError(f"no image of capture {source_name!r} in the reconstruction")

    def to_record(self) -> dict[str, Any]:
        """The metadata as its JSON object."""
        return _record(self)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ReconstructionMetadata:
        """Build the metadata from its JSON object.

        Raises:
            ValueError: For another schema version, unknown or missing keys, or invalid values.
        """
        if not isinstance(record, Mapping) or record.get("schema_version") != SCHEMA_VERSION:
            got = record.get("schema_version") if isinstance(record, Mapping) else None
            raise ValueError(f"reconstruction.json schema_version must be {SCHEMA_VERSION}, "
                             f"got {got!r}")  # fmt: skip
        _check_keys(cls, record, "reconstruction.json")
        anchor, check = record["scale_anchor"], record["scale_check"]
        _check_keys(CameraRecord, record["camera"], "reconstruction.json camera")
        _check_keys(ColmapRecord, record["colmap"], "reconstruction.json colmap")
        images: Sequence[Any] = record["images"]
        for image in images:
            _check_keys(ImageRecord, image, "reconstruction.json image")
        return cls(**{
            **record,
            "scale_anchor": None if anchor is None else ScaleAnchor.from_record(anchor),
            "scale_check": None if check is None else ScaleCheck.from_record(check),
            "camera": CameraRecord(**record["camera"]),
            "images": tuple(ImageRecord(**image) for image in images),
            "colmap": ColmapRecord(**record["colmap"]),
        })  # fmt: skip

    def write(self, path: str | Path) -> None:
        """Write the JSON (atomically: a temporary file renamed into place)."""
        publish_json(path, self.to_record())

    @classmethod
    def read(cls, path: str | Path) -> ReconstructionMetadata:
        """Read and check a ``reconstruction.json``."""
        return cls.from_record(json.loads(Path(path).read_text()))


__all__ = [
    "LENGTH_UNITS",
    "METADATA_NAME",
    "REGISTERED_BY",
    "SCHEMA_VERSION",
    "WORLD_FRAMES",
    "CameraRecord",
    "ColmapRecord",
    "ImageRecord",
    "ReconstructionMetadata",
    "ScaleAnchor",
    "ScaleCheck",
]
