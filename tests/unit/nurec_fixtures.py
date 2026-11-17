# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic NuRec capture bundles for the unit tests (``doper.nurec.capture`` format)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

CAMERA = {"width": 96, "height": 54, "fx": 80.0, "fy": 80.0, "cx": 47.5, "cy": 26.5,
          "rectified": True}  # fmt: skip
HANDEYE_CAMERA = {"width": 64, "height": 36, "fx": 40.0, "fy": 40.0, "cx": 31.5, "cy": 17.5,
                  "rectified": True}  # fmt: skip
NAMES = ("000002.jpg", "000000.png", "000001.png")
LINEAGE = "fixture-capture"
# A profile that is not packaged, so it pins no bundle: fixture bundles stand in for a user's own
# capture under it.
OWN_PROFILE = "own-capture"
# The NuRec recipe template users copy (examples/nurec/recipe.yaml).
NUREC_TEMPLATE = Path(__file__).resolve().parents[2] / "examples" / "nurec" / "recipe.yaml"


def rigid(seed: int) -> np.ndarray:
    """A random rigid 4x4 transform (meters)."""
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    matrix = np.eye(4)
    matrix[:3, :3] = q
    matrix[:3, 3] = rng.uniform(-0.5, 0.5, 3)
    return matrix


def write_image(path: Path, size: tuple[int, int], seed: int) -> str:
    """Write a random BGR image of ``size`` (width, height); return its SHA-256."""
    pixels = np.random.default_rng(seed).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), pixels)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write one JSON object per line."""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def write_bundle(
    root: Path,
    *,
    names: Sequence[str] = NAMES,
    camera: Mapping[str, Any] = CAMERA,
    robot: bool = True,
    handeye_camera: Mapping[str, Any] | None = CAMERA,
    robot_key: str = "T_base_mount",
    exclude: Sequence[str] | None = None,
    image_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Write a capture bundle and return what the tests compare against.

    Returns:
        ``poses`` (name to ``T_base_mount``), ``T_mount_camera`` and ``sha256`` (name to
        image SHA-256).
    """
    root.mkdir(parents=True, exist_ok=True)
    size = image_size or (int(camera["width"]), int(camera["height"]))
    (root / "session.json").write_text(json.dumps({"camera": dict(camera)}))
    shas = {name: write_image(root / "images" / name, size, i) for i, name in enumerate(names)}
    rows = [{"name": n, "sha256": shas[n], "lineage": LINEAGE} for n in names]
    write_jsonl(root / "images.jsonl", rows)
    poses = {name: rigid(10 + i) for i, name in enumerate(names)}
    mount_camera = rigid(99)
    if robot:
        write_jsonl(root / "robot.jsonl",
                    [{"name": n, robot_key: poses[n].tolist()} for n in names])  # fmt: skip
        handeye: dict[str, Any] = {"T_mount_camera": mount_camera.tolist()}
        if handeye_camera is not None:
            handeye["camera"] = dict(handeye_camera)
        handeye["validation"] = {"max_translation_error_m": 0.004, "max_rotation_error_deg": 0.5}
        (root / "handeye.json").write_text(json.dumps(handeye))
    if exclude is not None:
        (root / "exclude.json").write_text(json.dumps(list(exclude)))
    return {"poses": poses, "T_mount_camera": mount_camera, "sha256": shas}


# A NuRec rig calibration: OpenCV sensor into a forward-left-up rig (NuRec's T_sensor_rig).
SENSOR_RIG = np.array([[0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0]])  # fmt: skip
SCENE_SOURCES = ("000000.jpg", "000001.jpg", "000002.jpg", "000003.jpg")
SCENE_LINEAGES = ("session-a", "session-a", "session-b", "session-b")
WORLD_TO_NRE = np.array([[1.0, 0.0, 0.0, -0.4], [0.0, 1.0, 0.0, 0.05], [0.0, 0.0, 1.0, -0.5],
                         [0.0, 0.0, 0.0, 1.0]])  # fmt: skip


def calibration(camera: Mapping[str, Any] = CAMERA, **overrides: Any) -> dict[str, Any]:  # noqa: ANN401
    """A ``camera_calibrations`` entry of ``rig_trajectories.json`` for ``camera``."""
    parameters = {
        "resolution": [camera["width"], camera["height"]], "shutter_type": "GLOBAL",
        "external_distortion_parameters": None, "principal_point": [camera["cx"], camera["cy"]],
        "focal_length": [camera["fx"], camera["fy"]], "radial_coeffs": [0.0] * 6,
        "tangential_coeffs": [0.0] * 2, "thin_prism_coeffs": [0.0] * 4,
    }  # fmt: skip
    parameters.update(overrides)
    return {"sequence_id": "input", "logical_sensor_name": "camera1", "unique_sensor_idx": 0,
            "T_sensor_rig": SENSOR_RIG.tolist(),
            "camera_model": {"type": "opencv-pinhole", "parameters": parameters}}  # fmt: skip


def rig_trajectories(T_world_cameras: Sequence[np.ndarray], **calibration_overrides: Any) -> dict:  # noqa: ANN401, N803
    """A NuRec ``rig_trajectories.json`` record whose training cameras are ``T_world_cameras``."""
    rigs = [(pose @ np.linalg.inv(SENSOR_RIG)).tolist() for pose in T_world_cameras]
    stamps = [i * 1_000_000 for i in range(len(rigs))]
    return {
        "T_world_base": np.eye(4).tolist(),
        "world_to_nre": {"matrix": WORLD_TO_NRE.tolist()},
        "rig_trajectories": [{
            "sequence_id": "input", "rig_bbox": None,
            "cameras_linear_start_frame_indices": {"camera1@input": 0},
            "lidars_linear_start_frame_indices": {},
            "cameras_frame_timestamps_us": {"camera1@input": [[t, t] for t in stamps]},
            "lidars_frame_timestamps_us": {}, "T_rig_worlds": rigs,
            "T_rig_world_timestamps_us": stamps,
            "cameras_frame_T_rig_worlds": {"camera1@input": [[r, r] for r in rigs]},
        }],
        "camera_calibrations": {"camera1@input": calibration(**calibration_overrides)},
        "lidar_calibrations": {},
    }  # fmt: skip


def write_usdz(path: Path, record: Mapping[str, Any]) -> Path:
    """Write a stored (uncompressed) zip holding ``rig_trajectories.json``, as a USDZ does."""
    import zipfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("checkpoint.ckpt", b"not a checkpoint")
        archive.writestr("rig_trajectories.json", json.dumps(record))
    return path


def write_reconstruction(
    root: Path,
    *,
    metric: bool = True,
    scale: float = 1.085,
    poses: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Write a reconstruction (COLMAP text model + ``reconstruction.json``) and its frame map.

    Every source in ``SCENE_SOURCES`` is registered; image ``frame-00000i.png`` is source i.
    ``poses`` (source name to ``cam_from_world``) replaces the random default poses.

    Returns:
        ``metadata_path``, ``frames`` (the ``FrameMap``), ``cam_from_world`` (source name to the
        model's 4x4 pose) and ``metadata``.
    """
    from doper.nurec.capture import FrameMap, NormalizedFrame, PinholeCamera
    from doper.reconstruction import colmap_io
    from doper.reconstruction.metadata import (
        CameraRecord,
        ColmapRecord,
        ImageRecord,
        ReconstructionMetadata,
        ScaleAnchor,
    )
    from doper.reconstruction.seeding import text_model_id

    camera = PinholeCamera(**CAMERA)
    if poses is None:
        poses = {name: np.linalg.inv(rigid(40 + i)) for i, name in enumerate(SCENE_SOURCES)}
    images = {
        i + 1: colmap_io.ColmapImage(
            i + 1, colmap_io.quaternion_wxyz(poses[name][:3, :3]),
            tuple(float(v) for v in poses[name][:3, 3]), 1, f"frame-{i:06d}.png")
        for i, name in enumerate(SCENE_SOURCES)
    }  # fmt: skip
    # The text model rounds the poses; read them back so the fixture poses are the model's.
    params = (camera.fx, camera.fy, camera.cx, camera.cy)
    model = colmap_io.ColmapModel(
        {1: colmap_io.ColmapCamera(1, "PINHOLE", camera.width, camera.height, params)}, images, {}
    )
    colmap_io.write_text_model(model, root / "sparse-text")
    model = colmap_io.read_text_model(root / "sparse-text")
    cam_from_world = {
        name: colmap_io.cam_from_world(model.images[i + 1]) for i, name in enumerate(SCENE_SOURCES)
    }
    anchor = ScaleAnchor("sim3_seed_to_refined_centres", scale, np.eye(3).tolist(), (0.0, 0.0, 0.0),
                         0.001, 0.002, 0.001 / scale * 1000, 0.002 / scale * 1000, 4)  # fmt: skip
    records = tuple(
        ImageRecord(f"frame-{i:06d}.png", name, "0" * 64, "1" * 64, i + 1, False, None, True, None,
                    None, "mapper")
        for i, name in enumerate(SCENE_SOURCES)
    )  # fmt: skip
    metadata = ReconstructionMetadata(
        strategy="robot_handeye" if metric else "colmap_sfm",
        world_frame="robot_base" if metric else "arbitrary",
        length_unit="colmap" if metric else "arbitrary",
        scene_units_per_metre=scale if metric else None,
        scale_anchor=anchor if metric else None,
        camera=CameraRecord("PINHOLE", camera.width, camera.height, params, True,
                            "capture session", False),
        images=records,
        colmap=ColmapRecord("3.13.0", [], {}, "sparse/0", []),
        inputs={}, capture_flags={}, reference_path=False,
        reconstruction_id=text_model_id(root / "sparse-text"), model_text="sparse-text",
    )  # fmt: skip
    metadata.write(root / "reconstruction.json")
    frames = FrameMap(
        tuple(NormalizedFrame(f"frame-{i:06d}.png", name, "1" * 64, "0" * 64, lineage)
              for i, (name, lineage) in enumerate(zip(SCENE_SOURCES, SCENE_LINEAGES, strict=True))),
        camera, camera, 1280,
    )  # fmt: skip
    return {"metadata_path": root / "reconstruction.json", "frames": frames,
            "cam_from_world": cam_from_world, "metadata": metadata}  # fmt: skip


def nurec_recipe_data() -> dict[str, Any]:
    """The template NuRec recipe as YAML data, under ``OWN_PROFILE`` (no bundle pin)."""
    import yaml

    data = yaml.safe_load(NUREC_TEMPLATE.read_text())
    data["profile"] = OWN_PROFILE
    return data


def own_nurec_recipe(directory: Path) -> str:
    """Write ``nurec_recipe_data`` to ``directory/nurec-own.yaml``; return its path."""
    import yaml

    path = Path(directory) / "nurec-own.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(nurec_recipe_data()))
    return str(path)
