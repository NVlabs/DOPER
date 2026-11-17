# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate worker inputs without importing Kit or modifying output artifacts."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from doper.scene.geometry import Intrinsics, intrinsics_from_lens
from rendering.config import RenderConfig, load_render_config
from rendering.dataset_contract import is_integer
from rendering.environment_bundle import ReplayInputs, load_replay
from rendering.isaac.settings import IsaacSettings
from rendering.robustness import require_backend
from rendering.scene_spec import SCENE_SPEC_VERSION, AssetCatalog, MeshInfo
from rendering.usd_mesh import read_mesh_usda


@dataclass(frozen=True)
class WorkerInputs:
    """Everything a worker renders from, validated before Kit starts."""

    args: argparse.Namespace
    cfg: RenderConfig
    settings: IsaacSettings
    usd_path: str
    mesh_data: dict[str, Any]
    kp_names: list[str]
    kp_local: np.ndarray
    intr: Intrinsics
    mesh: MeshInfo
    catalog: AssetCatalog
    scenes: dict[int, dict[str, Any]]
    environment: ReplayInputs | None = None


def _keypoints(
    payload: Any,  # noqa: ANN401 - JSON value
    source: str,
    *,
    allow_empty: bool = False,
) -> tuple[list[str], np.ndarray]:
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: keypoints payload must be an object")
    names = payload.get("keypoint_names")
    if (
        not isinstance(names, list)
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
        or (not names and not allow_empty)
    ):
        raise ValueError(f"{source}: keypoint names must be nonempty and unique")
    points = np.asarray(payload.get("keypoints_3d"), dtype=float)
    if not names and allow_empty and points.size == 0:
        return names, points.reshape(0, 3)
    if points.shape != (len(names), 3) or not np.isfinite(points).all():
        raise ValueError(f"{source}: keypoints must be a finite K x 3 array")
    return names, points


def _finite_json(value: Any) -> bool:  # noqa: ANN401 - JSON value
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_json(child) for child in value.values())
    if isinstance(value, list):
        return all(_finite_json(child) for child in value)
    return True


def _validate_scene(
    record: Any,  # noqa: ANN401 - JSON value
    *,
    seed: int,
    catalog: AssetCatalog,
    line: int,
) -> None:
    label = f"scene line {line}"
    if not isinstance(record, dict):
        raise ValueError(f"{label}: record must be an object")
    if not is_integer(record.get("index")) or record["index"] < 0:
        raise ValueError(f"{label}: index must be a nonnegative integer")
    if not is_integer(record.get("seed")) or record["seed"] != seed:
        raise ValueError(f"{label}: seed does not match worker")
    if not is_integer(record.get("version")) or record["version"] != SCENE_SPEC_VERSION:
        raise ValueError(f"{label}: unsupported scene version")
    if "catalog_fingerprint" in record and record["catalog_fingerprint"] != catalog.fingerprint:
        raise ValueError(f"{label}: asset catalog fingerprint differs")
    if not _finite_json(record):
        raise ValueError(f"{label}: nonfinite scene values")
    for key in ("object", "camera", "material", "background", "ground"):
        if not isinstance(record.get(key), dict):
            raise ValueError(f"{label}: missing {key} object")
    for key in ("distractors", "lights"):
        if not isinstance(record.get(key), list):
            raise ValueError(f"{label}: missing {key} list")
    for section, field, shape in (
        ("object", "pose", (4, 4)),
        ("object", "center", (3,)),
        ("camera", "cam_to_world", (4, 4)),
    ):
        values = np.asarray(record[section].get(field), dtype=float)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"{label}: invalid {section}.{field}")
    z = record["object"].get("bbox_min_z")
    if isinstance(z, bool) or not isinstance(z, (int, float)) or not math.isfinite(z):
        raise ValueError(f"{label}: invalid object.bbox_min_z")


def _load_scenes(
    path: str | Path, *, start: int, end: int, seed: int, catalog: AssetCatalog
) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    with Path(path).open() as stream:
        for line, text in enumerate(stream, start=1):
            if not text.strip():
                continue
            record = json.loads(text)
            _validate_scene(record, seed=seed, catalog=catalog, line=line)
            index = record["index"]
            if index in records:
                raise ValueError(f"duplicate scene index {index}")
            records[index] = record
    missing = set(range(start, end)) - records.keys()
    if missing:
        raise ValueError(f"scene file is missing requested indices: {sorted(missing)[:10]}")
    return {index: records[index] for index in range(start, end)}


def load_object_keypoints(
    mesh_data: dict[str, Any], usd_path: str, keypoints_path: str | None
) -> tuple[list[str], np.ndarray]:
    """Validate the named keypoint schema shared by launcher and worker.

    External labels must name every point and agree with embedded USD labels; inventing names
    could silently change the trained model's keypoint order.

    Args:
        mesh_data: ``read_mesh_usda`` result of the object.
        usd_path: Object USD (for messages).
        keypoints_path: Optional keypoints JSON overriding the embedded keypoints.

    Returns:
        Keypoint names and their ``(K, 3)`` object-frame coordinates.

    Raises:
        ValueError: The keypoints are missing, malformed, or differ from the USD.
    """
    kp_names, kp_local = _keypoints(mesh_data, usd_path, allow_empty=True)
    if keypoints_path:
        supplied = json.loads(Path(keypoints_path).read_text())
        names, coordinates = _keypoints(supplied, keypoints_path)
        if kp_names and (
            names != kp_names
            or coordinates.shape != kp_local.shape
            or not np.allclose(coordinates, kp_local, rtol=0, atol=1e-6)
        ):
            raise ValueError("supplied keypoints differ from embedded USD keypoints")
        kp_names, kp_local = names, coordinates
    if not kp_names:
        raise ValueError("no keypoints found in USD or keypoints_json")
    return kp_names, kp_local


def _check_worker_args(args: argparse.Namespace) -> None:
    for key in ("start_idx", "end_idx", "worker_id", "gpu_device"):
        value = getattr(args, key)
        if not is_integer(value) or value < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if args.end_idx <= args.start_idx:
        raise ValueError("end_idx must be greater than start_idx")
    if not is_integer(args.seed):
        raise ValueError("seed must be an integer")
    if not args.scenes:
        raise ValueError("a launcher-generated --scenes file is required")


def _object_points(mesh_data: dict[str, Any]) -> np.ndarray:
    points = np.asarray(mesh_data["points"], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("object points must be a nonempty finite N x 3 array")
    return points


def load_inputs(args: argparse.Namespace) -> WorkerInputs:
    """Load authored geometry and the mandatory launcher scenes before GPU startup.

    Args:
        args: Worker arguments (``config``, ``usd``, ``keypoints_json``, ``scenes``, the sample
            range, ``worker_id``, ``gpu_device``, ``seed`` and the replay options).

    Returns:
        The validated inputs; nothing is written.

    Raises:
        ValueError: An argument, the config, the object or the scene file is invalid.
    """
    _check_worker_args(args)
    cfg = load_render_config(args.config)
    require_backend(cfg.robustness, "isaac", appearance=cfg.robust_appearance)
    settings = IsaacSettings.from_config(cfg)
    settings.validate_environment(getattr(args, "environment_root", None))
    usd_path = cfg.resolve(args.usd or cfg.usd_path)
    if not usd_path or not Path(usd_path).is_file():
        raise ValueError(f"Object USD not found: {usd_path!r}")
    usd_path = str(Path(usd_path).resolve())
    mesh_data = read_mesh_usda(usd_path)
    points = _object_points(mesh_data)
    keypoints_path = cfg.resolve(args.keypoints_json or cfg.keypoints_json)
    kp_names, kp_local = load_object_keypoints(mesh_data, usd_path, keypoints_path)
    intr = intrinsics_from_lens(
        cfg.resolution, cfg.resolution, cfg.focal_length_mm, cfg.sensor_width_mm
    )
    catalog = AssetCatalog.scan(cfg)
    scenes = _load_scenes(
        args.scenes, start=args.start_idx, end=args.end_idx, seed=args.seed, catalog=catalog
    )
    environment = load_replay(args, usd_path, records=scenes.values())
    return WorkerInputs(
        args,
        cfg,
        settings,
        usd_path,
        mesh_data,
        kp_names,
        kp_local,
        intr,
        MeshInfo.from_points(points, kp_local),
        catalog,
        scenes,
        environment,
    )
