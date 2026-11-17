# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BOP test export: every model of a bank scans every test image; valid poses go to a BOP CSV.

The engine combines the single-pass test export (inference at score 0.15 with at most five
detections per object and image, meter keypoints converted to millimeters before PnP) with its
direct-timing variant: all object models stay resident and scan each image in turn, and one
wall-clock span per image runs from the decoded input to the final valid pose list (CUDA
synchronized at both ends; ``doper.bop.timing`` documents the protocol). For the HOT3D native camera
the span also covers rectification of the native fisheye image and the JPEG round trip that
reproduces the staged test pixels.

Only test RGB and intrinsics are read: from a BOP split directory (``<scene>/rgb`` and
``scene_camera.json``; ground truth is never opened) or from a frozen test manifest (``frames``
with ``scene_id``, ``im_id``, ``path``, ``rgb_sha256``, ``cam_K`` and, for the native camera,
``native_path``, ``native_rgb_sha256`` and ``native_calibration``).

Each image's result is appended to a ``frames.jsonl`` journal, so an interrupted export resumes
where it stopped; only a torn final line is discarded. An untimed export may instead scan model by
model through a ``JournalCache``: one journal per (checkpoint, keypoints, test population, scan
settings), which every later bank holding the same model reuses (timed banks always rescan every
model). The cached journal is the model's raw pre-filter detection journal: every kept detection
with its box, keypoints, scores and PnP status (``doper.bop.variants`` fuses from it; its raw
keypoint heatmaps are stored separately, only for the heatmap variants); the valid-pose journal is
derived from it. A pose is written only if it passes ``valid_pose``, the BOP server's rules; dropped
rows are counted, never written. The CSV is ``scene_id,im_id,obj_id,score,R,t,time`` with ``R``
row-major and ``t`` in millimeters.
"""

from __future__ import annotations

import csv
import dataclasses
import functools
import hashlib
import io
import json
import math
import os
import re
import shutil
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, NamedTuple, TypeVar

import cv2
import numpy as np

from ..console import log
from . import hot3d
from .layout import CHECKPOINT_PATH, KEYPOINTS_PATH
from .sources import (
    IMAGE_SUFFIXES,
    DatasetProfile,
    append_jsonl,
    canonical_sha256,
    file_lock,
    json_roundtrip,
    load_profile,
    load_resource,
    publish_bytes,
    publish_json,
    publish_text,
    sha256_file,
)

CameraName = Literal["pinhole", "native"]
Detect = Callable[[np.ndarray, np.ndarray], list[Any]]

SCORE_THRESHOLD = 0.15
MAX_DETECTIONS = 5
DATASET_OBJECTS: dict[str, int] = {"handal": 40, "hopev2": 28, "hot3d": 33, "tless": 30}
LEGACY_DATASET_NAMES: dict[str, str] = {"hope": "hopev2"}
CSV_HEADER = ("scene_id", "im_id", "obj_id", "score", "R", "t", "time")
NO_RUNTIME = -1
MM_PER_M = 1000.0
JPEG_QUALITY = 95
SEED = 42
ORTHONORMAL_ATOL = 1e-5
DETERMINANT_TOLERANCE = 1e-5
SCORE_DIGITS = ".10g"
VALUE_DIGITS = ".12g"
JOURNAL_NAME = "frames.jsonl"
IDENTITY_NAME = "identity.json"
# The cached payload format: version 1 was the per-model journal of valid, filtered poses;
# version 2 was the raw pre-filter detection journal with heatmap archives inline; version 3
# is the raw journal of small per-detection fields (``RawJournal``), with heatmaps in a separate,
# lazily filled store (``JournalCache.heatmap_entry``).
# Bump it with ANY change that can alter a model's predictions (detector, keypoints, PnP,
# rectification, pose filtering or the record format): the key holds only the model,
# population and settings hashes, and ``--force`` keeps the cache.
CACHE_SCHEMA_VERSION = 3
CACHE_RECEIPT_NAME = "COMPLETE.json"
CACHE_PAYLOAD_DIR = "journal"
MODEL_NAME = "model.json"
HEATMAP_ENTRY_SUFFIX = ".heatmaps"
HEATMAP_INDEX_NAME = "index.jsonl"
HEATMAP_ARCHIVE_DIR = "archives"
REBUILD_SUFFIX = ".rebuild"
LOCK_SUFFIX = ".lock"
GENERATION_NAME = "GENERATION"
BINDING_NAME = "BINDING.json"
STATUS_VALID, STATUS_PNP_FAILED, STATUS_INVALID = "valid", "pnp_failed", "invalid_pose"
DETECTION_STATUSES = (STATUS_VALID, STATUS_PNP_FAILED, STATUS_INVALID)
_METHOD = re.compile(r"[A-Za-z0-9]+")
_CSV_NAME = re.compile(r"(?P<method>[A-Za-z0-9]+)_(?P<dataset>[a-z0-9]+)-test\.csv")
_clock = time.perf_counter
_T = TypeVar("_T")


@dataclass(frozen=True)
class TestFrame:
    """One test image and its intrinsics.

    Attributes:
        scene_id: BOP scene id.
        im_id: BOP image id.
        path: Network input image (the staged rectified JPEG for the native camera).
        rgb_sha256: SHA-256 of the input image bytes.
        camera_matrix: Row-major 3x3 intrinsics of the input image.
        native_path: Native fisheye image (native camera only).
        native_rgb_sha256: SHA-256 of the native image bytes.
        native_calibration: Native fisheye calibration (``doper.bop.hot3d``).
    """

    scene_id: int
    im_id: int
    path: Path
    rgb_sha256: str
    camera_matrix: tuple[float, ...]
    native_path: Path | None = None
    native_rgb_sha256: str | None = None
    native_calibration: Mapping[str, Any] | None = None

    @property
    def key(self) -> tuple[int, int]:
        """``(scene_id, im_id)``."""
        return self.scene_id, self.im_id


@dataclass(frozen=True)
class BankModel:
    """One resident object model of a bank.

    Attributes:
        obj_id: BOP object id.
        detect: ``detect(image_bgr, K)`` returning ``doper.infer.Detection`` objects with poses
            (translations in millimeters).
        checkpoint_sha256: SHA-256 of the checkpoint file.
        keypoints_sha256: SHA-256 of the ordered keypoint file.
        config: The loaded model's configuration (``model_config``): its hash and the heatmap
            geometry (``crop_size``, ``keypoint_heatmap_beta``); None for a stand-in model.
    """

    obj_id: int
    detect: Detect
    checkpoint_sha256: str
    keypoints_sha256: str
    config: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ExportSummary:
    """Coverage and filter counts of one test export.

    Attributes:
        csv_path: Written CSV.
        csv_sha256: SHA-256 of the CSV.
        dataset: Dataset name (``handal``, ``hopev2``, ``hot3d`` or ``tless``).
        method: Method name of the CSV.
        population_sha256: Identity of the test image population.
        images: Images processed (every image, including those without poses).
        objects: Objects in the bank.
        missing_objects: Dataset objects absent from a partial bank.
        detections: Detections the bank models returned, counted before each object's top
            ``max_detections`` are kept for PnP.
        pose_rows: Rows written.
        invalid_poses: Poses dropped by ``valid_pose``.
        pnp_failures: Detections without a PnP pose.
        images_without_poses: Images that produced no row.
        resumed_images: Images taken from an existing journal.
        cached_models: Models whose journal came from the journal cache.
        journal_generations: Object id to the generation of the cached journal (the forward
            pass) its rows came from, as a one-element list; empty without the cache.
    """

    csv_path: Path
    csv_sha256: str
    dataset: str
    method: str
    population_sha256: str
    images: int
    objects: tuple[int, ...]
    missing_objects: tuple[int, ...]
    detections: int
    pose_rows: int
    invalid_poses: int
    pnp_failures: int
    images_without_poses: int
    resumed_images: int
    cached_models: int = 0
    journal_generations: Mapping[str, list[str]] = field(default_factory=dict)


def valid_pose(pose: Mapping[str, Any]) -> bool:
    """Return whether a pose passes the BOP server's rules.

    Finite ``R``, ``t`` and score; 0.15 <= score <= 1; ``t_z > 0``; ``R^T R`` close to the
    identity (``np.allclose`` with ``atol=1e-5`` and NumPy's default ``rtol``); and
    ``|det(R) - 1| < 1e-5``.

    Args:
        pose: Mapping with ``score``, row-major ``R`` (9 values) and ``t`` (3 values, mm).

    Returns:
        True for a valid pose; False for any malformed or out-of-range pose.
    """
    try:
        rotation = np.asarray(pose["R"], dtype=float).reshape(3, 3)
        translation = np.asarray(pose["t"], dtype=float).reshape(3)
        score = float(pose["score"])
    except (ValueError, KeyError, TypeError):
        return False
    return bool(
        np.isfinite(rotation).all()
        and np.isfinite(translation).all()
        and math.isfinite(score)
        and SCORE_THRESHOLD <= score <= 1
        and translation[2] > 0
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=ORTHONORMAL_ATOL)
        and abs(np.linalg.det(rotation) - 1) < DETERMINANT_TOLERANCE
    )


def pose_from_detection(detection: Any) -> dict[str, Any] | None:  # noqa: ANN401
    """Convert a detection's PnP result to a pose record (no validity check).

    Args:
        detection: ``doper.infer.Detection``; ``tvec`` is already in millimeters because PnP
            consumed millimeter keypoints, so it is never rescaled here.

    Returns:
        ``{"score", "R", "t"}``, or None when PnP produced no pose.
    """
    if detection.rvec is None or detection.tvec is None:
        return None
    rotation, _ = cv2.Rodrigues(np.asarray(detection.rvec, dtype=np.float64))
    return {
        "score": float(detection.score),
        "R": rotation.ravel().tolist(),
        "t": np.asarray(detection.tvec, dtype=np.float64).reshape(3).tolist(),
    }


def normalize_dataset(name: str) -> str:
    """Return the release dataset name (the legacy ``hope`` becomes ``hopev2``).

    Raises:
        ValueError: For a dataset without a test export.
    """
    dataset = LEGACY_DATASET_NAMES.get(name, name)
    if dataset not in DATASET_OBJECTS:
        raise ValueError(
            f"unknown BOP test dataset {name!r}; expected one of {sorted(DATASET_OBJECTS)}"
        )
    return dataset


def csv_name(method: str, dataset: str) -> str:
    """Return the BOP submission file name ``<method>_<dataset>-test.csv``.

    Args:
        method: Alphanumeric method name (the BOP server's rule).
        dataset: Dataset name; ``hope`` is written as ``hopev2``.

    Raises:
        ValueError: For a non-alphanumeric method or an unknown dataset.
    """
    if not _METHOD.fullmatch(method):
        raise ValueError(f"BOP method names are alphanumeric: {method!r}")
    return f"{method}_{normalize_dataset(dataset)}-test.csv"


def keypoints_mm(path: str | Path) -> np.ndarray:
    """Read an ordered keypoint file (meters) and return the keypoints in millimeters.

    Raises:
        ValueError: If the keypoints are not a finite ``(K, 3)`` array.
    """
    values = json.loads(Path(path).read_text())["keypoints_3d"]
    points = np.asarray(values, dtype=np.float64) * MM_PER_M
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f"{path}: expected finite (K, 3) meter keypoints")
    return points


def encode_jpeg(image: np.ndarray) -> bytes:
    """Encode a rectified image as the staged test JPEG (quality 95).

    Raises:
        ValueError: If OpenCV cannot encode the image.
    """
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("JPEG encode failed")
    return data.tobytes()


def _camera_matrix(scene_camera: Mapping[str, Any], im_id: int) -> tuple[float, ...]:
    """Frame intrinsics; a missing frame falls back to constant scene intrinsics (source rule)."""
    first = next(iter(scene_camera.values()))
    row = scene_camera.get(str(im_id))
    if row is None:
        if not all(
            np.allclose(r["cam_K"], first["cam_K"], rtol=1e-8, atol=1e-6)
            for r in scene_camera.values()
        ):
            raise ValueError("missing frame calibration with nonconstant intrinsics")
        row = first
    matrix = np.asarray(row["cam_K"], dtype=float).reshape(3, 3)
    if (
        not np.isfinite(matrix).all()
        or matrix[0, 0] <= 0
        or matrix[1, 1] <= 0
        or not np.allclose(matrix[2], [0, 0, 1])
    ):
        raise ValueError("invalid camera intrinsics")
    return tuple(matrix.ravel().tolist())


def _scene_images(scene: Path) -> list[Path]:
    """RGB images of a scene; anything else in ``rgb/`` makes the scene malformed."""
    rgb = scene / "rgb"
    if not (scene / "scene_camera.json").is_file() or not rgb.is_dir():
        raise ValueError(f"{scene}: malformed BOP test scene (needs scene_camera.json and rgb/)")
    images = sorted(rgb.iterdir())
    bad = [p.name for p in images if p.suffix.lower() not in IMAGE_SUFFIXES or not p.stem.isdigit()]
    if bad or not images:
        raise ValueError(
            f"{scene}: malformed rgb/ entries {bad[:5]}" if bad else f"{scene}: no RGB"
        )
    return images


def _split_frames(root: Path) -> tuple[list[TestFrame], str]:
    """Frames of a BOP split directory and the hash of their relative-path population.

    Every entry of the split must be a numeric scene directory with ``scene_camera.json`` and
    an ``rgb/`` of numerically named PNG/JPEG images; anything else is refused, not skipped.
    """
    entries = sorted(root.iterdir(), key=lambda p: p.name)
    malformed = [p.name for p in entries if not (p.is_dir() and p.name.isdigit())]
    if malformed:
        raise ValueError(f"{root}: malformed BOP test split entries {malformed[:5]}")
    if not entries:
        raise ValueError(f"{root}: no BOP test scenes")
    frames = []
    for scene in entries:
        images = _scene_images(scene)
        camera = json.loads((scene / "scene_camera.json").read_text())
        for image in images:
            im_id = int(image.stem)
            frames.append(
                TestFrame(
                    int(scene.name), im_id, image, sha256_file(image), _camera_matrix(camera, im_id)
                )
            )
    records = [
        {
            "scene_id": f.scene_id,
            "im_id": f.im_id,
            "rgb_sha256": f.rgb_sha256,
            "cam_K": list(f.camera_matrix),
            "path": PurePosixPath(f.path.relative_to(root)).as_posix(),
        }
        for f in frames
    ]
    return frames, canonical_sha256(records)


def _manifest_frame(row: Mapping[str, Any], base: Path, camera: CameraName) -> TestFrame:
    frame = TestFrame(
        scene_id=int(row["scene_id"]),
        im_id=int(row["im_id"]),
        path=base / row["path"],
        rgb_sha256=row["rgb_sha256"],
        camera_matrix=tuple(float(v) for v in np.asarray(row["cam_K"], dtype=float).ravel()),
    )
    if camera == "pinhole":
        return frame
    missing = [
        k for k in ("native_path", "native_rgb_sha256", "native_calibration") if k not in row
    ]
    if missing:
        raise ValueError(f"native test frames need {missing}")
    return dataclasses.replace(
        frame,
        native_path=base / row["native_path"],
        native_rgb_sha256=row["native_rgb_sha256"],
        native_calibration=row["native_calibration"],
    )


def load_test_frames(test_root: str | Path, camera: CameraName) -> tuple[list[TestFrame], str]:
    """Load the test population (RGB and intrinsics only) and its identity hash.

    Args:
        test_root: BOP split directory, or a test manifest JSON (relative paths resolve against
            its directory).
        camera: ``pinhole``, or ``native`` (needs a manifest with native frames).

    Returns:
        Frames sorted by ``(scene_id, im_id)`` and the population SHA-256 (of the manifest
        bytes, or of the canonical frame list with paths relative to the split directory).

    Raises:
        ValueError: For an empty or duplicated population, or a split directory with ``native``.
    """
    root = Path(test_root)
    if root.is_file():
        data = root.read_bytes()
        rows = json.loads(data)["frames"]
        frames = [_manifest_frame(row, root.parent, camera) for row in rows]
        population = hashlib.sha256(data).hexdigest()
    elif camera == "native":
        raise ValueError("the native camera needs a test manifest with native frames")
    else:
        frames, population = _split_frames(root)
    frames.sort(key=lambda f: f.key)
    keys = [f.key for f in frames]
    if not frames or len(set(keys)) != len(keys):
        raise ValueError(f"{root}: empty or duplicated test population")
    return frames, population


@dataclass(frozen=True)
class TestPopulation:
    """The expected test population of a dataset.

    Attributes:
        images: Test images.
        scenes: Test scenes (HOT3D: clips).
        keys: Frozen ``(scene_id, im_id)`` index, when the dataset has one (HOT3D).
    """

    images: int
    scenes: int
    keys: frozenset[tuple[int, int]] | None = None


def frozen_population(dataset: str) -> TestPopulation:
    """Return a packaged dataset's frozen test population (profile ``meta.test_population``).

    Raises:
        ValueError: For an unknown dataset or a frozen index that disagrees with the counts.
    """
    return profile_population(load_profile(normalize_dataset(dataset)))


def profile_population(profile: DatasetProfile) -> TestPopulation:
    """Return a profile's frozen test population (``meta.test_population``).

    Raises:
        ValueError: If the frozen index disagrees with the counts.
    """
    spec = profile.meta["test_population"]
    keys = None
    if "index_resource" in spec:
        index = load_resource(profile, spec["index_resource"])
        keys = frozenset((int(r["scene_id"]), int(r["im_id"])) for r in index["images"])
        if len(keys) != spec["images"] or len({k[0] for k in keys}) != spec["scenes"]:
            raise ValueError(f"{profile.name}: frozen test index disagrees with its counts")
    return TestPopulation(int(spec["images"]), int(spec["scenes"]), keys)


def require_population(frames: Sequence[TestFrame], expected: TestPopulation) -> None:
    """Require the loaded test frames to be exactly the expected population.

    Raises:
        ValueError: If the image or scene count, or the frozen index, differs.
    """
    keys = {f.key for f in frames}
    scenes = {f.scene_id for f in frames}
    if len(frames) != expected.images or len(scenes) != expected.scenes:
        raise ValueError(
            f"test population has {len(frames)} images in {len(scenes)} scenes; expected "
            f"{expected.images} images in {expected.scenes} scenes"
        )
    if expected.keys is not None and keys != expected.keys:
        raise ValueError("test population differs from the frozen test image index")


def _timing_hardware(device: str) -> dict[str, Any]:
    from .timing import hardware

    return hardware(device)


def _detect_with_model(
    model: Any,  # noqa: ANN401
    cfg: Any,  # noqa: ANN401
    device: str,
    keypoints: np.ndarray,
    score_threshold: float,
    max_detections: int,
    raw_outputs: bool,
    image: np.ndarray,
    camera_matrix: np.ndarray,
) -> list[Any]:
    from ..infer import infer_image

    return infer_image(
        model,
        image,
        cfg,
        device,
        score_thr=score_threshold,
        max_det=max_detections,
        keypoints_3d=keypoints,
        camera_matrix=camera_matrix,
        return_raw_keypoint_outputs=raw_outputs,
    )


def model_config(cfg: Any) -> dict[str, Any]:  # noqa: ANN401
    """The configuration record of a loaded model (``BankModel.config``).

    Args:
        cfg: The ``DoperConfig`` after ``load_model`` filled it from the checkpoint.

    Returns:
        ``config_sha256`` (canonical hash of the whole configuration), ``keypoint_head``,
        ``crop_size`` and ``keypoint_heatmap_beta`` (the heatmap refinement geometry).
    """
    values = json.loads(json.dumps(dataclasses.asdict(cfg), default=str))
    return {
        "config_sha256": canonical_sha256(values),
        "keypoint_head": cfg.model.keypoint_head,
        "crop_size": int(cfg.model.crop_size),
        "keypoint_heatmap_beta": float(cfg.loss.keypoint_heatmap_beta),
    }


def load_bank(
    bank: Mapping[int, Path],
    *,
    device: str,
    score_threshold: float = SCORE_THRESHOLD,
    max_detections: int = MAX_DETECTIONS,
    raw_outputs: bool = False,
) -> list[BankModel]:
    """Load every model of a bank onto ``device`` (all models stay resident).

    Each bank entry is a run directory holding ``work/last.pth`` and the ordered meter keypoints
    ``geometry/keypoints_3d.json``; the keypoint layout must match the checkpoint.

    Args:
        bank: Object id to run directory.
        device: Torch device string.
        score_threshold: Detection score threshold.
        max_detections: Detections per object and image.
        raw_outputs: Keep each detection's raw keypoint heatmaps and crop affine (the same
            forward pass; detections and poses are unchanged).

    Returns:
        Resident models sorted by object id.
    """
    models = []
    for obj in sorted(bank):
        run = load_run(bank[obj], device)
        detect = functools.partial(
            _detect_with_model,
            run.model,
            run.cfg,
            device,
            run.keypoints,
            score_threshold,
            max_detections,
            raw_outputs,
        )
        models.append(
            BankModel(
                obj,
                detect,
                run.checkpoint_sha256,
                run.keypoints_sha256,
                model_config(run.cfg),
            )
        )
    return models


class LoadedRun(NamedTuple):
    """One run directory's model, loaded in evaluation mode (``load_run``).

    Attributes:
        model: The ``DoperNet`` on its device.
        cfg: The config it was built with (``DoperConfig``).
        keypoints: Its ordered keypoints in millimeters (``keypoints_mm``).
        checkpoint_sha256: SHA-256 of ``work/last.pth``.
        keypoints_sha256: SHA-256 of ``geometry/keypoints_3d.json``.
    """

    model: Any
    cfg: Any
    keypoints: np.ndarray
    checkpoint_sha256: str
    keypoints_sha256: str


def load_run(run: str | Path, device: str) -> LoadedRun:
    """Load one run's model onto ``device`` under the export seeds, checking its keypoints.

    Args:
        run: Run directory holding ``work/last.pth`` and ``geometry/keypoints_3d.json`` (meters);
            the keypoint layout must match the checkpoint.
        device: Torch device string.

    Returns:
        The loaded run.
    """
    import torch

    from ..config import DataConfig, DoperConfig, ModelConfig
    from ..infer import assert_keypoints_match, load_model

    run = Path(run)
    checkpoint, keypoint_file = run / CHECKPOINT_PATH, run / KEYPOINTS_PATH
    cfg = DoperConfig(data=DataConfig(root="."), model=ModelConfig(pretrained=False))
    torch.manual_seed(SEED)
    cv2.setRNGSeed(SEED)
    model = load_model(checkpoint, cfg, device).eval()
    keypoints = keypoints_mm(keypoint_file)
    assert_keypoints_match(model, keypoints)
    return LoadedRun(model, cfg, keypoints, sha256_file(checkpoint), sha256_file(keypoint_file))


def synchronizer(device: str) -> Callable[[], None]:
    """Return ``torch.cuda.synchronize`` for a CUDA device, else a no-op."""
    if device.startswith("cuda"):
        import torch

        return torch.cuda.synchronize
    return lambda: None


def _read_checked(path: Path, digest: str | None, what: str) -> bytes:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"{path}: {what} bytes changed")
    return raw


def read_checked_image(path: Path, digest: str | None, flags: int, what: str) -> np.ndarray:
    """Read an image, check its SHA-256 against ``digest`` and decode it (``cv2.imdecode``).

    Raises:
        ValueError: If the bytes changed (naming ``what``) or do not decode.
    """
    return _decode(_read_checked(path, digest, what), flags)


def _decode(raw: bytes, flags: int) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), flags)
    if image is None:
        raise ValueError("undecodable test image")
    return image


def _load_input(frame: TestFrame, camera: CameraName) -> np.ndarray:
    """Read, verify and decode the timer's input (outside the timed span)."""
    if camera == "native":
        assert frame.native_path is not None
        return read_checked_image(
            frame.native_path, frame.native_rgb_sha256, cv2.IMREAD_UNCHANGED, "native test RGB"
        )
    return read_checked_image(frame.path, frame.rgb_sha256, cv2.IMREAD_COLOR, "test RGB")


def _rectified_input(native: np.ndarray, frame: TestFrame) -> tuple[np.ndarray, np.ndarray]:
    """Rectify and JPEG round-trip a native image; it must reproduce the staged pixels and K."""
    assert frame.native_calibration is not None
    rectified, camera_matrix = hot3d.rectify(native, dict(frame.native_calibration))
    encoded = encode_jpeg(rectified)
    expected = np.asarray(frame.camera_matrix, dtype=np.float64).reshape(3, 3)
    if hashlib.sha256(encoded).hexdigest() != frame.rgb_sha256 or not np.array_equal(
        camera_matrix, expected
    ):
        raise ValueError("on-the-fly rectification differs from the frozen staged pixels/K")
    return _decode(encoded, cv2.IMREAD_COLOR), camera_matrix


def _scan_models(
    models: Sequence[BankModel], image: np.ndarray, camera_matrix: np.ndarray, max_detections: int
) -> dict[str, Any]:
    """Run every model on one image; keep the valid poses and count the rest."""
    poses, counts = [], Counter()
    for model in models:
        detections = model.detect(image, camera_matrix)
        counts["detections"] += len(detections)
        for det in sorted(detections, key=lambda d: d.score, reverse=True)[:max_detections]:
            pose = pose_from_detection(det)
            if pose is None:
                counts["pnp_failures"] += 1
            elif not valid_pose(pose):
                counts["invalid_poses"] += 1
            else:
                poses.append({**pose, "object_id": model.obj_id})
    return {
        "predictions": poses,
        **{k: counts[k] for k in ("detections", "invalid_poses", "pnp_failures")},
    }


def _timed_span(
    frame: TestFrame,
    camera: CameraName,
    synchronize: Callable[[], None],
    detect: Callable[[np.ndarray, np.ndarray], _T],
) -> tuple[_T, float]:
    """Run ``detect`` on one image inside the direct-timing span.

    The input is read, hash-checked and decoded before the timer; the span, synchronized at both
    ends, covers (native camera) rectification and ``detect(image, camera_matrix)``.

    Returns:
        What ``detect`` returned and the span in seconds.

    Raises:
        ValueError: For changed image bytes, a rectification mismatch or an invalid timer.
    """
    image = _load_input(frame, camera)
    synchronize()
    start = _clock()
    if camera == "native":
        image, camera_matrix = _rectified_input(image, frame)
    else:
        camera_matrix = np.asarray(frame.camera_matrix, dtype=np.float64).reshape(3, 3)
    result = detect(image, camera_matrix)
    synchronize()
    seconds = _clock() - start
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("invalid timer")
    return result, seconds


def scan_frame(
    models: Sequence[BankModel],
    frame: TestFrame,
    *,
    camera: CameraName,
    max_detections: int,
    synchronize: Callable[[], None],
) -> dict[str, Any]:
    """Scan one image with every model under the direct-timing protocol.

    The image is read, hash-checked and decoded before the timer; one wall-clock span, CUDA
    synchronized at both ends, then covers (native camera) rectification and the JPEG round
    trip, and every model's detection, keypoints, PnP and pose validation.

    Returns:
        The journal record: ``scene_id``, ``im_id``, ``rgb_sha256``, ``objects_scanned``,
        ``seconds``, ``predictions`` (valid poses with ``object_id``) and the filter counts.

    Raises:
        ValueError: For changed image bytes, a rectification mismatch or an invalid timer.
    """
    scanned, seconds = _timed_span(
        frame,
        camera,
        synchronize,
        lambda image, camera_matrix: _scan_models(models, image, camera_matrix, max_detections),
    )
    return {
        "scene_id": frame.scene_id,
        "im_id": frame.im_id,
        "rgb_sha256": frame.rgb_sha256,
        "objects_scanned": len(models),
        "seconds": seconds,
        **scanned,
    }


def _validate_record(
    record: Mapping[str, Any],
    expected: Mapping[tuple[int, int], str],
    objects: Sequence[int],
    max_detections: int,
    *,
    poses: bool = True,
) -> tuple[int, int]:
    key = record["scene_id"], record["im_id"]
    if key not in expected or record["rgb_sha256"] != expected[key]:
        raise ValueError("journal population/identity differs")
    seconds = record["seconds"]
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not seconds > 0:
        raise ValueError("invalid journal image duration")
    if record["objects_scanned"] != len(objects):
        raise ValueError("journal record did not scan every object")
    counts = Counter(p["object_id"] for p in record["predictions"])
    if not set(counts) <= set(objects) or any(n > max_detections for n in counts.values()):
        raise ValueError("journal pose object or cap invalid")
    if poses and not all(valid_pose(p) for p in record["predictions"]):
        raise ValueError("invalid cached pose")
    return key


def recover_journal(
    path: Path, expected: Mapping[tuple[int, int], str], objects: Sequence[int], max_detections: int
) -> dict[tuple[int, int], dict[str, Any]]:
    """Read a journal; only a torn final line may be dropped (the file is then rewritten clean).

    Raises:
        ValueError: For a corrupt or duplicated record, or one that does not validate.
    """
    return _recover_records(
        path, lambda record: _validate_record(record, expected, objects, max_detections)
    )


def _recover_records(
    path: Path, validate: Callable[[Mapping[str, Any]], tuple[int, int]]
) -> dict[tuple[int, int], dict[str, Any]]:
    """``recover_journal`` with a record validator returning each record's image key."""
    if not path.exists():
        return {}
    lines = path.read_bytes().splitlines(keepends=True)
    records: dict[tuple[int, int], dict[str, Any]] = {}
    for i, line in enumerate(lines):
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if i == len(lines) - 1 and not line.endswith(b"\n"):
                break
            raise ValueError(f"{path}: corrupt journal record") from None
        key = validate(record)
        if key in records:
            raise ValueError(f"{path}: duplicate journal image")
        records[key] = record
    else:
        if not lines or lines[-1].endswith(b"\n"):
            return records  # nothing torn: the file is already clean
    clean = path.with_suffix(".clean")
    clean.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in records.values()))
    os.replace(clean, path)
    return records


@contextmanager
def _locked(
    directory: Path, lock_path: Path | None = None, *, wait: bool = False
) -> Iterator[None]:
    """Hold an exclusive lock on ``directory/run.lock``, or on a sibling ``lock_path``.

    A sibling lock stays in place while ``directory`` itself is renamed. An export's own journal
    refuses a second writer (``wait`` false); the shared journal cache entries and heatmap
    stores wait for one (``wait``), always taking a heatmap lock before an entry lock.
    """
    if lock_path is None:
        directory.mkdir(parents=True, exist_ok=True)
        lock_path = directory / "run.lock"
    busy = f"another export holds the journal lock of {directory}"
    with file_lock(lock_path, wait=wait, busy=busy):
        yield


def _check_identity(directory: Path, identity: Mapping[str, Any]) -> None:
    path = directory / IDENTITY_NAME
    if path.exists() and json.loads(path.read_text()) != json_roundtrip(identity):
        raise ValueError(
            f"{directory}: journal identity changed (bank, population or settings); remove the "
            "journal to start the export over"
        )
    publish_json(path, identity)


def scan_bank(
    models: Sequence[BankModel],
    frames: Sequence[TestFrame],
    *,
    camera: CameraName,
    max_detections: int,
    synchronize: Callable[[], None],
    journal_dir: Path | None,
    identity: Mapping[str, Any],
) -> tuple[dict[tuple[int, int], dict[str, Any]], int]:
    """Scan every frame with every model, resuming from and appending to a journal.

    Args:
        models: Resident bank models.
        frames: Test population.
        camera: ``pinhole`` or ``native``.
        max_detections: Detections kept per object and image.
        synchronize: Device synchronization around the timed span.
        journal_dir: Journal directory; None keeps records in memory only.
        identity: Bank, population and settings; a journal written under another identity is
            refused.

    Returns:
        Records in population order and the number taken from the journal.
    """
    objects = [m.obj_id for m in models]
    expected = {f.key: f.rgb_sha256 for f in frames}

    def scan(frame: TestFrame) -> dict[str, Any]:
        record = scan_frame(
            models, frame, camera=camera, max_detections=max_detections, synchronize=synchronize
        )
        _validate_record(record, expected, objects, max_detections)
        return record

    if journal_dir is None:
        return {f.key: scan(f) for f in frames}, 0
    with _locked(journal_dir):
        _check_identity(journal_dir, identity)
        journal = journal_dir / JOURNAL_NAME
        records = recover_journal(journal, expected, objects, max_detections)
        resumed = len(records)
        with journal.open("a") as stream:
            for frame in frames:
                if frame.key in records:
                    continue
                record = scan(frame)
                append_jsonl(stream, record)
                records[frame.key] = record
    return {f.key: records[f.key] for f in frames}, resumed


def write_csv(
    path: Path, records: Mapping[tuple[int, int], Mapping[str, Any]], *, record_time: bool
) -> None:
    """Atomically write the BOP CSV, sorted by image, object and descending score.

    Args:
        path: Destination.
        records: Journal records keyed by ``(scene_id, im_id)``.
        record_time: Write each image's measured seconds; otherwise ``-1`` (no runtime claim).
    """
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(CSV_HEADER)
    for key, record in sorted(records.items()):
        duration = format(record["seconds"], VALUE_DIGITS) if record_time else NO_RUNTIME
        for pose in sorted(record["predictions"], key=lambda p: (p["object_id"], -p["score"])):
            writer.writerow(
                [
                    key[0],
                    key[1],
                    pose["object_id"],
                    format(pose["score"], SCORE_DIGITS),
                    " ".join(format(v, VALUE_DIGITS) for v in pose["R"]),
                    " ".join(format(v, VALUE_DIGITS) for v in pose["t"]),
                    duration,
                ]
            )
    publish_text(path, stream.getvalue())


def _resolve_csv(out_csv: Path) -> tuple[Path, str, str]:
    match = _CSV_NAME.fullmatch(out_csv.name)
    if match is None:
        raise ValueError(f"{out_csv.name}: the CSV name must be <method>_<dataset>-test.csv")
    method, dataset = match["method"], normalize_dataset(match["dataset"])
    return out_csv.with_name(csv_name(method, dataset)), method, dataset


def _bank_objects(
    bank: Mapping[int, Path], dataset: str, objects: Sequence[int] | None, allow_partial_bank: bool
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    declared = (
        list(objects) if objects is not None else list(range(1, DATASET_OBJECTS[dataset] + 1))
    )
    foreign = sorted(set(bank) - set(declared))
    if foreign:
        raise ValueError(f"bank objects {foreign} are not in the dataset objects")
    missing = tuple(o for o in declared if o not in bank)
    if missing and not allow_partial_bank:
        raise ValueError(
            f"partial bank: objects {list(missing)} have no model; pass allow_partial_bank=True "
            "to export the objects that have one"
        )
    return tuple(sorted(bank)), missing


def bank_identity(
    models: Sequence[BankModel],
    population_sha256: str,
    **settings: Any,  # noqa: ANN401
) -> dict[str, Any]:
    """Journal identity: model hashes, the test population and the scan settings."""
    return {
        "population_sha256": population_sha256,
        "models": [
            {
                "object_id": m.obj_id,
                "checkpoint_sha256": m.checkpoint_sha256,
                "keypoints_sha256": m.keypoints_sha256,
            }
            for m in models
        ],
        **settings,
    }


def default_journal_dir(csv_path: Path) -> Path:
    """The journal directory ``run_bank`` uses for a CSV: ``<csv stem>.journal`` beside it."""
    csv_path = Path(csv_path)
    return csv_path.with_name(f"{csv_path.stem}.journal")


@dataclass(frozen=True)
class JournalKey:
    """Identity of one model's cached journal.

    Attributes:
        checkpoint_sha256: SHA-256 of the checkpoint.
        keypoints_sha256: SHA-256 of the ordered keypoint file.
        population_sha256: Identity of the test population (``load_test_frames``).
        settings_sha256: ``cache_settings_sha256`` of the scan settings.
        schema_version: Version of the cached payload format (``CACHE_SCHEMA_VERSION``).
    """

    checkpoint_sha256: str
    keypoints_sha256: str
    population_sha256: str
    settings_sha256: str
    schema_version: int = CACHE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """The key as a JSON mapping."""
        return dataclasses.asdict(self)

    def digest(self) -> str:
        """SHA-256 of the canonical key."""
        return canonical_sha256(self.to_dict())


def cache_settings_sha256(camera: CameraName, score_threshold: float, max_detections: int) -> str:
    """The settings hash of a cache key: camera, score threshold and detection cap."""
    return canonical_sha256(
        {"camera": camera, "score_threshold": score_threshold, "max_detections": max_detections}
    )


def _read_payload(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    return _parse_payload(path.read_text(), path)


def _parse_payload(text: str, path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    records: dict[tuple[int, int], dict[str, Any]] = {}
    for line in text.splitlines():
        record = json.loads(line)
        key = int(record["scene_id"]), int(record["im_id"])
        if key in records:
            raise ValueError(f"{path}: duplicate cached journal image")
        records[key] = record
    return records


def _optional_sha256(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


class CachedJournal(NamedTuple):
    """A complete cache entry's journal as one locked read saw it (``JournalCache.open``).

    Attributes:
        records: The journal records keyed by ``(scene_id, im_id)``.
        sha256: SHA-256 of the journal bytes those records were parsed from.
        generation: The journal's generation id (``ensure_generation``).
    """

    records: dict[tuple[int, int], dict[str, Any]]
    sha256: str
    generation: str


class JournalCache:
    """Per-model journals keyed by ``JournalKey``, under ``<root>/v<schema>/<key digest>``.

    An entry holds the payload directory (``journal/``: the raw detection journal
    ``frames.jsonl`` and the model record ``model.json``) and, once complete, ``COMPLETE.json``
    with the key and the SHA-256 of the journal and model record. Raw keypoint heatmaps live in
    a sibling heatmap store (``heatmap_entry``: ``archives/*.npz``, ``index.jsonl`` with each
    archive's SHA-256, its own lock and receipt), filled only for the heatmap variants
    (``cached_raw_journal``). An incomplete entry resumes: the
    producer journals into the payload directory. A complete entry whose receipt differs from
    its key, journal or model record is refused, never silently refilled.
    """

    def __init__(self, root: Path) -> None:
        """Use ``root`` as the cache directory (created on first write)."""
        self.root = Path(root)

    def entry(self, key: JournalKey) -> Path:
        """The entry directory of ``key``."""
        return self.root / f"v{key.schema_version}" / key.digest()

    def payload(self, key: JournalKey) -> Path:
        """The payload directory of ``key`` (journal, model record and heatmaps)."""
        return self.entry(key) / CACHE_PAYLOAD_DIR

    def contains(self, key: JournalKey) -> bool:
        """Whether ``key`` has a complete entry."""
        return (self.entry(key) / CACHE_RECEIPT_NAME).is_file()

    def heatmap_entry(self, key: JournalKey) -> Path:
        """The heatmap store of ``key``: a sibling of its entry."""
        return self.entry(key).with_name(f"{key.digest()}{HEATMAP_ENTRY_SUFFIX}")

    def entry_lock(self, key: JournalKey) -> Path:
        """The entry's lock file, a sibling that stays while the entry is replaced."""
        return self.entry(key).with_name(f"{key.digest()}{LOCK_SUFFIX}")

    def heatmap_lock(self, key: JournalKey) -> Path:
        """The heatmap store's lock file, a sibling that stays while the store is replaced."""
        return self.entry(key).with_name(f"{key.digest()}{HEATMAP_ENTRY_SUFFIX}{LOCK_SUFFIX}")

    def generation(self, key: JournalKey) -> str:
        """The generation id of ``key``'s journal, created with its payload (``GENERATION``).

        A journal written by another forward pass (a regenerated or replaced entry) has another
        generation; a heatmap store records the generation it was filled against.
        """
        with _locked(self.entry(key), self.entry_lock(key), wait=True):
            return ensure_generation(self.payload(key))

    def get_or_run(
        self, key: JournalKey, fn: Callable[[Path], object]
    ) -> dict[tuple[int, int], dict[str, Any]]:
        """Return the cached journal of ``key``, producing it with ``fn`` on a miss.

        Args:
            key: The entry's identity.
            fn: ``fn(directory)`` writes the payload journal ``directory/frames.jsonl`` (and,
                for a loaded model, ``directory/model.json``); it is called only when the entry
                is not complete.

        Returns:
            The journal records keyed by ``(scene_id, im_id)``.

        Raises:
            ValueError: If ``fn`` wrote no journal, or a complete entry's receipt differs from
                its key, journal or model record bytes.
        """
        with _locked(self.entry(key), self.entry_lock(key), wait=True):
            return self._complete(key, fn)[0]

    def open(self, key: JournalKey, fn: Callable[[Path], object]) -> CachedJournal:
        """``get_or_run``, with the journal's SHA-256 and generation read under the same lock."""
        with _locked(self.entry(key), self.entry_lock(key), wait=True):
            records, digest = self._complete(key, fn)
            return CachedJournal(records, digest, ensure_generation(self.payload(key)))

    def _complete(
        self, key: JournalKey, fn: Callable[[Path], object]
    ) -> tuple[dict[tuple[int, int], dict[str, Any]], str]:
        """The checked journal records and SHA-256 of ``key`` (the caller holds the entry lock).

        The journal is read once: its bytes are hashed and parsed.
        """
        entry = self.entry(key)
        receipt_path = entry / CACHE_RECEIPT_NAME
        journal = self.payload(key) / JOURNAL_NAME
        if not receipt_path.is_file():
            fn(journal.parent)
            _publish_entry_receipt(entry, key)
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("key") != key.to_dict() or not journal.is_file():
            raise ValueError(f"{entry}: cache entry differs from its key; remove it")
        data = journal.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if receipt.get("journal_sha256") != digest or receipt.get(
            "model_sha256"
        ) != _optional_sha256(self.payload(key) / MODEL_NAME):
            raise ValueError(f"{entry}: cached journal bytes changed; remove the cache entry")
        return _parse_payload(data.decode(), journal), digest


def _publish_entry_receipt(entry: Path, key: JournalKey) -> None:
    """Complete an entry whose payload is written: its key and journal/model hashes."""
    journal = entry / CACHE_PAYLOAD_DIR / JOURNAL_NAME
    if not journal.is_file():
        raise ValueError(f"{entry}: the producer wrote no cached journal")
    receipt = {
        "key": key.to_dict(),
        "journal_sha256": sha256_file(journal),
        "model_sha256": _optional_sha256(entry / CACHE_PAYLOAD_DIR / MODEL_NAME),
    }
    publish_json(entry / CACHE_RECEIPT_NAME, receipt)


def ensure_generation(payload: Path) -> str:
    """Return the payload's journal generation id, creating a new one if it has none."""
    path = payload / GENERATION_NAME
    if not path.is_file():
        publish_text(path, uuid.uuid4().hex)
    return path.read_text()


def heatmap_archive_name(key: tuple[int, int]) -> str:
    """The heatmap archive of one image in a heatmap store: ``archives/<scene>_<im>.npz``."""
    return f"{HEATMAP_ARCHIVE_DIR}/{key[0]:06d}_{key[1]:06d}.npz"


def _finite_vector(value: Any) -> list[Any] | None:  # noqa: ANN401
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    return array.tolist() if np.isfinite(array).all() else None


def raw_detection(detection: Any, index: int) -> dict[str, Any]:  # noqa: ANN401
    """One kept detection of a raw journal record (``doper.infer.Detection``).

    ``rvec``/``tvec`` are null when PnP produced no pose or a non-finite one; ``R``/``t`` is
    the ``pose_from_detection`` pose of the kept (finite) ``rvec``/``tvec``. ``status`` is
    derived from them: ``pnp_failed`` (no PnP pose), ``invalid_pose`` (non-finite, or failing
    ``valid_pose``) or ``valid``, the export's filter classes.
    """
    rvec, tvec = _finite_vector(detection.rvec), _finite_vector(detection.tvec)
    if rvec is None or tvec is None:
        rvec = tvec = None
    pose = None if rvec is None else pose_from_detection(detection)
    if detection.rvec is None or detection.tvec is None:
        status = STATUS_PNP_FAILED
    elif pose is None or not valid_pose(pose):
        status = STATUS_INVALID
    else:
        status = STATUS_VALID
    return {
        "index": index,
        "score": float(detection.score),
        "bbox_xyxy": np.asarray(detection.bbox_xyxy).tolist(),
        "keypoints": np.asarray(detection.keypoints).tolist(),
        "keypoint_scores": np.asarray(detection.keypoint_scores).tolist(),
        "rvec": rvec,
        "tvec": tvec,
        "R": None if pose is None else pose["R"],
        "t": None if pose is None else pose["t"],
        "status": status,
    }


def _heatmap_arrays(detections: Sequence[Any]) -> dict[str, np.ndarray]:
    """``h<j>`` (raw keypoint heatmap, dtype kept) and ``w<j>`` (crop affine) per detection."""
    arrays = {}
    for j, det in enumerate(detections):
        heatmap = getattr(det, "keypoint_heatmap", None)
        affine = getattr(det, "crop_to_image", None)
        if heatmap is not None and affine is not None:
            arrays[f"h{j}"] = np.asarray(heatmap)
            arrays[f"w{j}"] = np.asarray(affine)
    return arrays


def scan_raw_frame(
    model: BankModel,
    frame: TestFrame,
    *,
    camera: CameraName,
    max_detections: int,
    synchronize: Callable[[], None],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Scan one image with one model and keep its raw, pre-filter detections.

    The same span as ``scan_frame`` (input decoded before the timer; native rectification
    inside it). The model's detections are sorted by descending score (stable) and the first
    ``max_detections`` kept, indexed in that order.

    Returns:
        The raw journal record (``scene_id``, ``im_id``, ``rgb_sha256``, ``objects_scanned``,
        ``seconds``, ``detections_returned`` and the kept ``detections``) and the heatmap
        arrays of the same pass (``h<j>``, ``w<j>``; empty unless the model keeps raw outputs).

    Raises:
        ValueError: For changed image bytes, a rectification mismatch or an invalid timer.
    """
    detections, seconds = _timed_span(frame, camera, synchronize, model.detect)
    kept = sorted(detections, key=lambda d: d.score, reverse=True)[:max_detections]
    record = {
        "scene_id": frame.scene_id,
        "im_id": frame.im_id,
        "rgb_sha256": frame.rgb_sha256,
        "objects_scanned": 1,
        "seconds": seconds,
        "detections_returned": len(detections),
        "detections": [raw_detection(det, j) for j, det in enumerate(kept)],
    }
    return record, _heatmap_arrays(kept)


def _validate_raw_record(
    record: Mapping[str, Any], expected: Mapping[tuple[int, int], str], max_detections: int
) -> tuple[int, int]:
    key = record["scene_id"], record["im_id"]
    if key not in expected or record["rgb_sha256"] != expected[key]:
        raise ValueError("raw journal population/identity differs")
    seconds = record["seconds"]
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not seconds > 0:
        raise ValueError("invalid raw journal image duration")
    detections = record["detections"]
    if record["objects_scanned"] != 1 or len(detections) > max_detections:
        raise ValueError("raw journal record scanned other models or exceeds the cap")
    if record["detections_returned"] < len(detections):
        raise ValueError("raw journal keeps more detections than were returned")
    for j, det in enumerate(detections):
        if det["index"] != j or det["status"] not in DETECTION_STATUSES:
            raise ValueError("raw journal detection index or status invalid")
        if det["status"] == STATUS_VALID and not valid_pose(det):
            raise ValueError("raw journal detection marked valid has an invalid pose")
        if det["status"] == STATUS_PNP_FAILED and det["rvec"] is not None:
            raise ValueError("raw journal PnP failure carries a pose")
    return key


def derive_valid_record(raw: Mapping[str, Any], obj_id: int) -> dict[str, Any]:
    """The valid-pose journal record (``scan_frame`` form) of one model's raw record.

    Valid poses keep their score, ``R`` and ``t`` in detection order; PnP failures and invalid
    poses are counted.
    """
    counts = Counter(det["status"] for det in raw["detections"])
    return {
        "scene_id": raw["scene_id"],
        "im_id": raw["im_id"],
        "rgb_sha256": raw["rgb_sha256"],
        "objects_scanned": 1,
        "seconds": raw["seconds"],
        "predictions": [
            {"score": det["score"], "R": det["R"], "t": det["t"], "object_id": obj_id}
            for det in raw["detections"]
            if det["status"] == STATUS_VALID
        ],
        "detections": raw["detections_returned"],
        "invalid_poses": counts[STATUS_INVALID],
        "pnp_failures": counts[STATUS_PNP_FAILED],
    }


class HeatmapSink:
    """Appends one image's heatmap archive and its index line to a heatmap store.

    The archive (compressed npz, every array's dtype kept) is published before its index line,
    whose ``sha256`` pins its bytes; an image without heatmaps gets a null archive.
    """

    def __init__(
        self,
        directory: Path,
        stream: Any,  # noqa: ANN401
        indexed: set[tuple[int, int]],
    ) -> None:
        """Write into ``directory`` and its open index ``stream``; ``indexed`` is updated."""
        self.directory, self.stream, self.indexed = directory, stream, indexed

    def write(self, key: tuple[int, int], arrays: Mapping[str, Any]) -> None:
        """Archive one image's arrays and append its index line."""
        name = sha = None
        if arrays:
            name = heatmap_archive_name(key)
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **arrays)
            data = buffer.getvalue()
            publish_bytes(self.directory / name, data)
            sha = hashlib.sha256(data).hexdigest()
        line = {"scene_id": key[0], "im_id": key[1], "archive": name, "sha256": sha}
        append_jsonl(self.stream, line)
        self.indexed.add(key)


def scan_raw_journal(
    model: BankModel,
    frames: Sequence[TestFrame],
    *,
    camera: CameraName,
    max_detections: int,
    synchronize: Callable[[], None],
    directory: Path,
    identity: Mapping[str, Any],
    heatmaps: HeatmapSink | None = None,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Scan every frame with one model into a raw journal, resuming from it.

    ``model.json`` records the model configuration. With ``heatmaps``, each newly scanned
    image's heatmaps of the same forward pass go to that store right after its journal line.

    Returns:
        Raw records keyed by ``(scene_id, im_id)`` in population order.
    """
    expected = {f.key: f.rgb_sha256 for f in frames}
    with _locked(directory):
        _check_identity(directory, identity)
        ensure_generation(directory)
        model_path = directory / MODEL_NAME
        model_record = {"config": None if model.config is None else dict(model.config)}
        if model_path.is_file() and json.loads(model_path.read_text()) != model_record:
            raise ValueError(f"{directory}: the model configuration changed while caching")
        publish_json(model_path, model_record)
        journal = directory / JOURNAL_NAME
        records = _recover_records(
            journal, lambda r: _validate_raw_record(r, expected, max_detections)
        )
        with journal.open("a") as stream:
            for frame in frames:
                if frame.key in records:
                    continue
                record, arrays = scan_raw_frame(
                    model,
                    frame,
                    camera=camera,
                    max_detections=max_detections,
                    synchronize=synchronize,
                )
                _validate_raw_record(record, expected, max_detections)
                append_jsonl(stream, record)
                records[frame.key] = record
                if heatmaps is not None:
                    heatmaps.write(frame.key, arrays)
    return {f.key: records[f.key] for f in frames}


HeatmapIndex = Mapping[tuple[int, int], tuple[Path | None, str | None]]


@dataclass(frozen=True)
class HeatmapArchive:
    """Loads one detection's raw heatmap logits and crop affine from its image's archive.

    The archive is read whole; its bytes are checked against the index SHA-256 the first time
    any loader of the same journal reads it (``verified`` is shared per journal).
    """

    key: tuple[int, int]
    index: int
    path: Path
    sha256: str
    verified: set[Path] = field(compare=False, repr=False)

    def __call__(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(logits, crop_to_image)`` of the detection.

        Raises:
            ValueError: If the archive changed or the detection has no heatmap.
        """
        raw = self.path.read_bytes()
        if self.path not in self.verified:
            if hashlib.sha256(raw).hexdigest() != self.sha256:
                raise ValueError(
                    f"{self.path}: heatmap archive bytes changed; remove the heatmap store"
                )
            self.verified.add(self.path)
        with np.load(io.BytesIO(raw)) as archive:
            if f"h{self.index}" not in archive:
                raise ValueError(
                    f"image {self.key}: detection {self.index} has no keypoint heatmap"
                )
            return archive[f"h{self.index}"], archive[f"w{self.index}"]


@dataclass(frozen=True)
class RawJournal:
    """One model's cached raw detection journal over one image population.

    Attributes:
        obj_id: Object id of the model.
        records: Raw records keyed by ``(scene_id, im_id)`` in population order.
        model: The model configuration record (``model_config``), or None for a stand-in.
        payload: The cache payload directory.
        generation: The journal's generation id (``JournalCache.generation``), read under the
            same entry lock as the records.
        heatmaps: Per image, its heatmap archive and SHA-256 (``None, None`` without
            heatmaps); None when the journal was opened without its heatmap store.
    """

    obj_id: int
    records: Mapping[tuple[int, int], Mapping[str, Any]]
    model: Mapping[str, Any] | None
    payload: Path
    generation: str
    heatmaps: HeatmapIndex | None = None
    _verified: set[Path] = field(default_factory=set, compare=False, repr=False)

    def has_heatmaps(self, key: tuple[int, int]) -> bool:
        """Whether the journal was opened with its heatmap store and image ``key`` has heatmaps."""
        return self.heatmaps is not None and self.heatmaps[key][0] is not None

    def heatmap_loader(self, key: tuple[int, int], index: int) -> HeatmapArchive:
        """The loader of one detection's raw heatmap logits and crop affine (hash-checked).

        Raises:
            ValueError: If the journal was opened without heatmaps or the image has no archive.
        """
        if self.heatmaps is None:
            raise ValueError("the raw journal was opened without its heatmap store")
        path, digest = self.heatmaps[key]
        if path is None or digest is None:
            raise ValueError(f"image {key}: the heatmap store holds no heatmaps")
        return HeatmapArchive(key, index, path, digest, self._verified)

    def heatmap(self, key: tuple[int, int], index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the raw heatmap logits and crop affine of one detection (hash-checked).

        Raises:
            ValueError: If the journal was opened without heatmaps, the image has no archive,
                the archive changed, or the detection has no heatmap.
        """
        return self.heatmap_loader(key, index)()


def journal_key(
    checkpoint_sha256: str,
    keypoints_sha256: str,
    population: str,
    *,
    camera: CameraName,
    score_threshold: float = SCORE_THRESHOLD,
    max_detections: int = MAX_DETECTIONS,
) -> JournalKey:
    """The cache key of a model (by its hashes) over a population."""
    settings = cache_settings_sha256(camera, score_threshold, max_detections)
    return JournalKey(checkpoint_sha256, keypoints_sha256, population, settings)


def raw_journal_key(
    run: Path,
    population: str,
    *,
    camera: CameraName,
    score_threshold: float = SCORE_THRESHOLD,
    max_detections: int = MAX_DETECTIONS,
) -> JournalKey:
    """The cache key of one run directory's journal over a population (hashes the run)."""
    run = Path(run)
    return journal_key(
        sha256_file(run / CHECKPOINT_PATH), sha256_file(run / KEYPOINTS_PATH), population,
        camera=camera, score_threshold=score_threshold, max_detections=max_detections,
    )  # fmt: skip


def _read_index(path: Path, expected: Mapping[tuple[int, int], str]) -> dict[Any, Any]:
    def validate(line: Mapping[str, Any]) -> tuple[int, int]:
        key = line["scene_id"], line["im_id"]
        if key not in expected or (line["archive"] is None) != (line["sha256"] is None):
            raise ValueError(f"{path}: heatmap index line invalid")
        return key

    return _recover_records(path, validate)


def _same_detections(cached: Mapping[str, Any], scanned: Mapping[str, Any]) -> bool:
    """Whether a rescan reproduced a cached record's detections bit for bit."""
    fields = ("detections_returned", "detections")
    fresh = json_roundtrip({k: scanned[k] for k in fields}, allow_nan=False)
    return fresh == {k: cached[k] for k in fields}


def cached_raw_journal(
    run: Path,
    frames: Sequence[TestFrame],
    population: str,
    cache: JournalCache,
    *,
    camera: CameraName,
    device: str,
    obj_id: int | None = None,
    score_threshold: float = SCORE_THRESHOLD,
    max_detections: int = MAX_DETECTIONS,
    key: JournalKey | None = None,
    heatmaps: bool = False,
) -> RawJournal:
    """Return one model's raw detection journal over ``frames``, scanning it on a cache miss.

    The journal holds only the small per-detection fields. Raw keypoint heatmaps are kept only
    when asked (``heatmaps``, the heatmap variants), in a separate store beside the entry
    (``JournalCache.heatmap_entry``), filled lazily: on a journal miss one forward pass writes
    both; for images whose journal already exists the model rescans them with raw outputs. When
    every kept detection (index, score, box, keypoints, scores, pose and status) equals the
    cached one bit for bit, the journal stays and only the heatmaps are added. Otherwise (GPU
    inference is not bit-reproducible) the whole population is rescanned once and both the
    journal and the store are replaced by that pass (``_rebuild_pair``), with a warning naming
    the model and the number of differing detections. The heatmaps therefore always come from
    their journal's own forward pass; the store's receipt binds it to that journal's hash.

    Args:
        run: Run directory (``work/last.pth`` and ``geometry/keypoints_3d.json``).
        frames: The image population (test images, or one support stream's images).
        population: Its identity hash (``load_test_frames``).
        cache: The per-model journal cache.
        camera: ``pinhole``, or ``native`` (rectified inside the span).
        device: Torch device string (used only on a miss).
        obj_id: Object id recorded on the journal (0 when omitted).
        score_threshold: Detection score threshold.
        max_detections: Detections kept per image.
        key: The precomputed cache key (``journal_key``); default: hashed from ``run``.
        heatmaps: Also open (and fill) the heatmap store.

    Returns:
        The validated raw journal, with its heatmap index when ``heatmaps``.

    Raises:
        ValueError: If the cached journal does not cover ``frames`` or does not validate, the
            checkpoint changed while caching, or the heatmap store differs from its receipt.
    """
    run = Path(run)
    if key is None:
        key = raw_journal_key(
            run, population, camera=camera, score_threshold=score_threshold,
            max_detections=max_detections,
        )  # fmt: skip
    if key.population_sha256 != population:
        raise ValueError("the journal key belongs to another population")
    scan = _RawScan(
        run, frames, {f.key: f.rgb_sha256 for f in frames}, cache, key,
        0 if obj_id is None else obj_id, camera, device, score_threshold, max_detections,
    )  # fmt: skip
    if not heatmaps:
        return scan.journal(cache.open(key, scan.fill(None)), None)
    store = cache.heatmap_entry(key)
    index_path = store / HEATMAP_INDEX_NAME
    receipt_path = store / CACHE_RECEIPT_NAME
    binding_path = store / BINDING_NAME
    with _locked(store, cache.heatmap_lock(key), wait=True):
        shutil.rmtree(_rebuild_dir(store, key), ignore_errors=True)
        # The store is bound to its journal's generation from its first line on; a store bound
        # to another (regenerated or replaced) journal, or to none, is discarded whole.
        binding = _binding(key, cache.generation(key))
        if not binding_path.is_file() or json.loads(binding_path.read_text()) != binding:
            _clear_store(store)
            publish_json(binding_path, binding)
        opened = None
        if not receipt_path.is_file():
            opened = _fill_store(scan, store)
        if opened is None:
            opened = cache.open(key, scan.fill(None))
        receipt = json.loads(receipt_path.read_text())
        expected_receipt = _store_receipt(key, sha256_file(index_path), opened.sha256)
        if any(
            receipt.get(name) != value for name, value in expected_receipt.items()
        ) or json.loads(binding_path.read_text()) != _binding(key, opened.generation):
            raise ValueError(f"{store}: heatmap store differs from its receipt; remove it")
        lines = _read_index(index_path, scan.expected)
        if set(lines) != set(scan.expected):
            raise ValueError(f"{store}: heatmap store does not cover the image population")
        index = {
            k: (None if v["archive"] is None else store / v["archive"], v["sha256"])
            for k, v in lines.items()
        }
        # Opened under the store lock, so the journal read is the one the receipt binds.
        return scan.journal(opened, index)


@dataclass(frozen=True)
class _RawScan:
    """One model's raw journal over one image population, as ``cached_raw_journal`` scans it."""

    run: Path
    frames: Sequence[TestFrame]
    expected: Mapping[tuple[int, int], str]
    cache: JournalCache
    key: JournalKey
    obj: int
    camera: CameraName
    device: str
    score_threshold: float
    max_detections: int

    def load(self, raw_outputs: bool) -> BankModel:
        """Load the run's model, checking it is the one the key hashed."""
        model = load_bank(
            {self.obj: self.run},
            device=self.device,
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
            raw_outputs=raw_outputs,
        )[0]
        if (model.checkpoint_sha256, model.keypoints_sha256) != (
            self.key.checkpoint_sha256,
            self.key.keypoints_sha256,
        ):
            raise ValueError(f"{self.run}: checkpoint or keypoints changed while caching")
        return model

    def produce(self, directory: Path, model: BankModel, sink: HeatmapSink | None) -> None:
        """Scan every frame with ``model`` into the payload ``directory`` (and ``sink``)."""
        identity = {
            "key": self.key.to_dict(),
            "camera": self.camera,
            "score_threshold": self.score_threshold,
            "max_detections": self.max_detections,
        }
        scan_raw_journal(
            model,
            self.frames,
            camera=self.camera,
            max_detections=self.max_detections,
            synchronize=synchronizer(self.device),
            directory=directory,
            identity=identity,
            heatmaps=sink,
        )

    def fill(self, sink: HeatmapSink | None) -> Callable[[Path], None]:
        """The cache producer of a miss: loads the model (with raw outputs for ``sink``)."""
        return lambda directory: self.produce(directory, self.load(sink is not None), sink)

    def journal(self, opened: CachedJournal, index: HeatmapIndex | None) -> RawJournal:
        """The validated raw journal of an opened cache entry, in population order."""
        records = opened.records
        if set(records) != set(self.expected):
            raise ValueError(f"{self.run}: cached journal does not cover the image population")
        for record in records.values():
            _validate_raw_record(record, self.expected, self.max_detections)
        payload = self.cache.payload(self.key)
        model_path = payload / MODEL_NAME
        model = json.loads(model_path.read_text())["config"] if model_path.is_file() else None
        ordered = {f.key: records[f.key] for f in self.frames}
        return RawJournal(self.obj, ordered, model, payload, opened.generation, index)


def _binding(key: JournalKey, generation: str) -> dict[str, Any]:
    """The binding of a heatmap store to its journal's generation."""
    return {"key": key.to_dict(), "generation": generation}


def _store_receipt(key: JournalKey, index_sha256: str, journal_sha256: str) -> dict[str, Any]:
    """The receipt of a filled heatmap store: its key and its index and journal hashes."""
    return {"key": key.to_dict(), "index_sha256": index_sha256, "journal_sha256": journal_sha256}


def _fill_store(scan: _RawScan, store: Path) -> CachedJournal | None:
    """Fill an unreceipted heatmap store (the caller holds its lock) and receipt it.

    A journal miss writes the journal and the heatmaps in one pass. Otherwise the images the
    store lacks are rescanned; if a rescan differs from the journal, both are replaced
    (``_rebuild_pair``) and None is returned, so the caller reopens the new journal.

    Returns:
        The journal the store was filled against, or None after a rebuild.
    """
    index_path = store / HEATMAP_INDEX_NAME
    indexed = set(_read_index(index_path, scan.expected))
    with index_path.open("a") as stream:
        sink = HeatmapSink(store, stream, indexed)
        opened = scan.cache.open(scan.key, scan.fill(sink))
        rescan = _rescan_missing(scan, sink, opened.records)
    if rescan is None:
        publish_json(
            store / CACHE_RECEIPT_NAME,
            _store_receipt(scan.key, sha256_file(index_path), opened.sha256),
        )
        return opened
    model, mismatch = rescan
    changed = _rebuild_pair(scan, store, model, opened.records)
    log(
        f"warning: {scan.run} (object {scan.obj}): the heatmap rescan differs from the cached "
        f"journal ({changed} detections differ, first at image {mismatch}); the "
        "journal and its heatmaps were both replaced by the new forward pass"
    )
    return None


def _rescan_missing(
    scan: _RawScan, sink: HeatmapSink, records: Mapping[tuple[int, int], Mapping[str, Any]]
) -> tuple[BankModel, tuple[int, int]] | None:
    """Rescan the images ``sink`` lacks, adding their heatmaps while they match ``records``.

    Returns:
        None when every rescan matched; else the loaded model and the first differing image.
    """
    missing = [f for f in scan.frames if f.key not in sink.indexed]
    if not missing:
        return None
    model = scan.load(True)
    for frame in missing:
        scanned, arrays = scan_raw_frame(
            model, frame, camera=scan.camera, max_detections=scan.max_detections,
            synchronize=synchronizer(scan.device),
        )  # fmt: skip
        if not _same_detections(records[frame.key], scanned):
            return model, frame.key
        sink.write(frame.key, arrays)
    return None


def _rebuild_dir(store: Path, key: JournalKey) -> Path:
    return store.with_name(f"{key.digest()}{REBUILD_SUFFIX}")


def _clear_store(store: Path) -> None:
    """Empty a heatmap store: receipt, binding, index and archives."""
    for name in (CACHE_RECEIPT_NAME, BINDING_NAME, HEATMAP_INDEX_NAME):
        (store / name).unlink(missing_ok=True)
    shutil.rmtree(store / HEATMAP_ARCHIVE_DIR, ignore_errors=True)


def _detection_mismatches(old: Mapping[str, Any], new: Mapping[str, Any]) -> int:
    """Kept detections of one image that differ between two raw records (or are unpaired)."""
    before, after = old["detections"], new["detections"]
    return sum(a != b for a, b in zip(before, after, strict=False)) + abs(len(before) - len(after))


def _rebuild_pair(
    scan: _RawScan,
    store: Path,
    model: BankModel,
    old: Mapping[tuple[int, int], Mapping[str, Any]],
) -> int:
    """Rescan the whole population once and replace both the journal and its heatmap store.

    The new pair (a journal with a new generation, a store bound to it, both receipted) is
    written in a staging directory by one pass of the already loaded ``model``
    (``_RawScan.produce``). Then, holding the entry lock (a sibling file,
    so no other process can recreate the entry meanwhile) and the store lock of the caller,
    the old store is emptied first and the journal entry and the store are swapped in by
    rename. Whatever rename a crash interrupts, the next open finds either the old journal
    with an empty store, no journal (a cache miss: one pass writes both), or the new journal
    with a store bound to another generation or none, which it discards whole and refills.

    Returns:
        The number of kept detections that differ from the old journal.
    """
    cache, key = scan.cache, scan.key
    staging = _rebuild_dir(store, key)
    entry, new_entry, new_store = cache.entry(key), staging / "entry", staging / "heatmaps"
    new_store.mkdir(parents=True)
    payload = new_entry / CACHE_PAYLOAD_DIR
    publish_json(new_store / BINDING_NAME, _binding(key, ensure_generation(payload)))
    index = new_store / HEATMAP_INDEX_NAME
    with index.open("a") as stream:
        scan.produce(payload, model, HeatmapSink(new_store, stream, set()))
    _publish_entry_receipt(new_entry, key)
    journal = payload / JOURNAL_NAME
    new = _read_payload(journal)
    receipt = _store_receipt(key, sha256_file(index), sha256_file(journal))
    publish_json(new_store / CACHE_RECEIPT_NAME, receipt)
    trash = staging / "replaced"
    trash.mkdir()
    with _locked(entry, cache.entry_lock(key), wait=True):
        _clear_store(store)
        os.rename(entry, trash / "entry")
        os.rename(new_entry, entry)
        os.rename(store, trash / "heatmaps")
        os.rename(new_store, store)
    shutil.rmtree(staging, ignore_errors=True)
    return sum(_detection_mismatches(old[k], new[k]) for k in old)


def _merge_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One image's bank record from its per-model records (in object order).

    ``seconds`` is the sum of the separate model scans: an untimed bank makes no runtime claim.
    """
    first = records[0]
    return {
        "scene_id": first["scene_id"],
        "im_id": first["im_id"],
        "rgb_sha256": first["rgb_sha256"],
        "objects_scanned": len(records),
        "seconds": sum(r["seconds"] for r in records),
        "predictions": [p for r in records for p in r["predictions"]],
        **{k: sum(r[k] for r in records) for k in ("detections", "invalid_poses", "pnp_failures")},
    }


def _scan_cached(
    bank: Mapping[int, Path],
    frames: Sequence[TestFrame],
    population: str,
    cache: JournalCache,
    *,
    camera: CameraName,
    score_threshold: float,
    max_detections: int,
    device: str,
) -> tuple[dict[tuple[int, int], dict[str, Any]], int, dict[str, list[str]]]:
    """Scan a bank model by model through the cache.

    Each model's valid-pose journal is derived from its cached raw journal.

    Returns:
        The merged records, the cache hits and each object's journal generation.
    """
    expected = {f.key: f.rgb_sha256 for f in frames}
    per_model, hits, generations = {}, 0, {}
    for obj in sorted(bank):
        key = raw_journal_key(
            bank[obj], population, camera=camera, score_threshold=score_threshold,
            max_detections=max_detections,
        )  # fmt: skip
        hits += cache.contains(key)
        journal = cached_raw_journal(
            bank[obj], frames, population, cache, camera=camera, device=device, obj_id=obj,
            score_threshold=score_threshold, max_detections=max_detections, key=key,
        )  # fmt: skip
        generations[str(obj)] = [journal.generation]
        records = {k: derive_valid_record(r, obj) for k, r in journal.records.items()}
        for record in records.values():
            _validate_record(record, expected, [obj], max_detections)
        per_model[obj] = records
    objects = sorted(bank)
    merged = {}
    for frame in frames:
        merged[frame.key] = _merge_records([per_model[o][frame.key] for o in objects])
        # Every pose was checked in its model's record; only the bank-level object and cap.
        _validate_record(merged[frame.key], expected, objects, max_detections, poses=False)
    return merged, hits, generations


def run_bank(
    bank: Mapping[int, Path],
    test_root: Path,
    out_csv: Path,
    *,
    score_threshold: float = SCORE_THRESHOLD,
    max_detections: int = MAX_DETECTIONS,
    camera: CameraName = "pinhole",
    journal_dir: Path | None = None,
    objects: Sequence[int] | None = None,
    allow_partial_bank: bool = False,
    record_time: bool = False,
    expected_population: TestPopulation | None = None,
    device: str = "cuda:0",
    cache: JournalCache | None = None,
) -> ExportSummary:
    """Export a model bank's test predictions to a BOP CSV.

    Every model scans every test image; there is no GT, object-presence or instance-count
    prior. Results are journaled per image under ``journal_dir`` (default: ``<csv stem>.journal``
    beside the CSV), so a rerun resumes. Poses failing ``valid_pose`` are counted and dropped.

    Args:
        bank: Object id to run directory (``work/last.pth`` and ``geometry/keypoints_3d.json``).
        test_root: BOP test split directory, or a test manifest JSON (see the module doc).
        out_csv: Output path named ``<method>_<dataset>-test.csv``; a legacy ``hope`` name is
            written as ``hopev2``.
        score_threshold: Detection score threshold.
        max_detections: Detections per object and image.
        camera: ``pinhole``, or ``native`` (HOT3D: rectify the native image inside the span).
        journal_dir: Journal directory.
        objects: The dataset's object ids (default: all objects of the dataset).
        allow_partial_bank: Export a bank that lacks some of ``objects``.
        record_time: Write each image's measured seconds to ``time`` instead of ``-1``; the
            journal identity then includes the hardware, so a resume never mixes timings.
        expected_population: Required test population; default: the dataset's frozen
            population from its packaged profile (``frozen_population``).
        device: Torch device string.
        cache: Scan model by model through this per-model journal cache (untimed only); a
            model whose journal is cached is not loaded, and ``journal_dir`` is unused.

    Returns:
        The export summary.

    Raises:
        ValueError: For a bad CSV name, a partial bank without ``allow_partial_bank``, objects
            outside the dataset, a test population other than the expected one, a changed
            input or journal, or a cache with ``record_time``.
    """
    if cache is not None and record_time:
        raise ValueError("the journal cache is untimed only; a timed bank rescans every model")
    csv_path, method, dataset = _resolve_csv(Path(out_csv))
    present, missing = _bank_objects(bank, dataset, objects, allow_partial_bank)
    frames, population = load_test_frames(test_root, camera)
    require_population(
        frames,
        expected_population if expected_population is not None else frozen_population(dataset),
    )
    if cache is not None:
        records, hits, generations = _scan_cached(
            bank,
            frames,
            population,
            cache,
            camera=camera,
            score_threshold=score_threshold,
            max_detections=max_detections,
            device=device,
        )
        summary = _finish(csv_path, records, method, dataset, population, present, missing, 0, hits)
        return dataclasses.replace(summary, journal_generations=generations)
    models = load_bank(
        bank, device=device, score_threshold=score_threshold, max_detections=max_detections
    )
    identity = bank_identity(
        models,
        population,
        camera=camera,
        score_threshold=score_threshold,
        max_detections=max_detections,
        record_time=record_time,
        hardware=_timing_hardware(device) if record_time else None,
    )
    journal = journal_dir if journal_dir is not None else default_journal_dir(csv_path)
    records, resumed = scan_bank(
        models,
        frames,
        camera=camera,
        max_detections=max_detections,
        synchronize=synchronizer(device),
        journal_dir=Path(journal),
        identity=identity,
    )
    return _finish(
        csv_path, records, method, dataset, population, present, missing, resumed, 0, record_time
    )


def _finish(
    csv_path: Path,
    records: Mapping[tuple[int, int], Mapping[str, Any]],
    method: str,
    dataset: str,
    population: str,
    present: tuple[int, ...],
    missing: tuple[int, ...],
    resumed: int,
    cached: int,
    record_time: bool = False,
) -> ExportSummary:
    """Write the CSV and summarize the export."""
    write_csv(csv_path, records, record_time=record_time)
    rows = list(records.values())
    return ExportSummary(
        csv_path=csv_path,
        csv_sha256=sha256_file(csv_path),
        dataset=dataset,
        method=method,
        population_sha256=population,
        images=len(rows),
        objects=present,
        missing_objects=missing,
        detections=sum(r["detections"] for r in rows),
        pose_rows=sum(len(r["predictions"]) for r in rows),
        invalid_poses=sum(r["invalid_poses"] for r in rows),
        pnp_failures=sum(r["pnp_failures"] for r in rows),
        images_without_poses=sum(not r["predictions"] for r in rows),
        resumed_images=resumed,
        cached_models=cached,
    )
