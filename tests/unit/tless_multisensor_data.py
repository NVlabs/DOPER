# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic T-LESS TRAIN data of three sensors for the multisensor tests.

``make_dataset`` writes, under one dataset root, what ``doper.bop.multisensor.prepare`` reads:

* ``data/train_primesense/000001``: BOP JSON (``scene_gt.json``, ``scene_camera.json``,
  ``scene_gt_info.json``), PNG RGB and the released ``mask_visib``;
* ``sensors/train_{kinect,canon}/01``: the original v2 layout (``gt.yml``, ``info.yml``,
  ``rgb/NNNN.{png,jpg}``, ``depth/``), without masks;
* ``data/test_primesense/000001``: two reserved test frames;
* ``objects/obj_000001/geometry``: the CAD model (a 4x4x4 vertex grid spanning a cube, its 12
  corner triangles), the 64 grid vertices as keypoints in meters and the ``audit.json`` keys.

Every sensor has its own image size and intrinsics. ``v2_archive`` packs a v2 object directory as
the official TRAIN ZIP (top directory = the two-digit object id).
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import yaml

OBJ = 1
FRAMES = (0, 1, 2, 3)
SIDE_MM = 40.0
GRID = 4
DEPTH_MM = 260.0
SIZES = {"primesense": (64, 48), "kinect": (80, 60), "canon": (96, 72)}
FOCAL = {"primesense": 80.0, "kinect": 96.0, "canon": 120.0}
SUFFIX = {"primesense": ".png", "kinect": ".png", "canon": ".jpg"}


def grid_vertices() -> np.ndarray:
    """The 4x4x4 vertex grid (millimeters) spanning the cube."""
    axis = np.linspace(-SIDE_MM / 2, SIDE_MM / 2, GRID)
    return np.array([[x, y, z] for x in axis for y in axis for z in axis])


def cube_faces() -> np.ndarray:
    """The 12 cube triangles over the grid's corner vertices."""
    last = GRID - 1
    corner = {
        (sx, sy, sz): (sx * last) * GRID**2 + (sy * last) * GRID + sz * last
        for sx in (0, 1)
        for sy in (0, 1)
        for sz in (0, 1)
    }
    cube = [(sx, sy, sz) for sx in (0, 1) for sy in (0, 1) for sz in (0, 1)]
    faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
             (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]  # fmt: skip
    return np.array([[corner[cube[a]], corner[cube[b]], corner[cube[c]]] for a, b, c in faces])


def write_geometry(directory: Path) -> tuple[Path, Path, Path]:
    """Write the CAD PLY, ``keypoints_3d.json`` and ``audit.json``; return their paths."""
    directory.mkdir(parents=True, exist_ok=True)
    vertices, faces = grid_vertices(), cube_faces()
    lines = [
        "ply", "format ascii 1.0", f"element vertex {len(vertices)}",
        "property float x", "property float y", "property float z",
        f"element face {len(faces)}", "property list uchar int vertex_indices", "end_header",
    ]  # fmt: skip
    lines += [f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    lines += [f"3 {a} {b} {c}" for a, b, c in faces]
    mesh = directory / f"obj_{OBJ:06d}.ply"
    mesh.write_text("\n".join(lines) + "\n")
    names = [f"KP_{i:03d}" for i in range(len(vertices))]
    keypoints = directory / "keypoints_3d.json"
    keypoints.write_text(
        json.dumps(
            {
                "object": f"obj_{OBJ:06d}",
                "num_keypoints": len(names),
                "keypoint_names": names,
                "keypoints_3d": (vertices / 1000.0).tolist(),
            }
        )
    )
    audit = directory / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "mesh_units": "mm",
                "keypoint_units": "m",
                "mesh_sha256": sha(mesh),
                "keypoints_sha256": sha(keypoints),
                "problems": [],
            }
        )
    )
    return mesh, keypoints, audit


def sha(path: Path) -> str:
    """SHA-256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def camera(sensor: str) -> np.ndarray:
    """The sensor's camera matrix (principal point at the image center)."""
    width, height = SIZES[sensor]
    focal = FOCAL[sensor]
    return np.array([[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]])


def pose(frame: int) -> tuple[np.ndarray, np.ndarray]:
    """The object pose (rotation, millimeter translation) of a capture, shared by all sensors."""
    rng = np.random.default_rng(100 + frame)
    rotation = cv2.Rodrigues(rng.uniform(-np.pi, np.pi, 3))[0]
    return rotation, np.array([rng.uniform(-8, 8), rng.uniform(-6, 6), DEPTH_MM])


def hull_mask(sensor: str, frame: int) -> np.ndarray:
    """Filled projected hull of the cube (the released PrimeSense mask)."""
    rotation, translation = pose(frame)
    xyz = grid_vertices() @ rotation.T + translation
    uv = (xyz @ camera(sensor).T)[:, :2] / xyz[:, 2:]
    width, height = SIZES[sensor]
    mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(np.round(uv).astype(np.int32)), 255)
    return mask


def _image(sensor: str, frame: int) -> np.ndarray:
    width, height = SIZES[sensor]
    seed = {"primesense": 1, "kinect": 2, "canon": 3}[sensor] * 1000 + frame
    return np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)


def _gt(frame: int, *, bop: bool) -> dict:
    rotation, translation = pose(frame)
    row = {
        "cam_R_m2c": rotation.reshape(-1).tolist(),
        "cam_t_m2c": translation.tolist(),
        "obj_id": OBJ,
    }
    if not bop:
        row["obj_bb"] = [1, 1, 10, 10]
    return row


def write_primesense(directory: Path, frames: tuple[int, ...] = FRAMES) -> Path:
    """Write a BOP ``train_primesense`` object directory."""
    for sub in ("rgb", "mask_visib", "depth"):
        (directory / sub).mkdir(parents=True, exist_ok=True)
    gt, cam, info = {}, {}, {}
    for frame in frames:
        assert cv2.imwrite(str(directory / "rgb" / f"{frame:06d}.png"), _image("primesense", frame))
        mask = hull_mask("primesense", frame)
        assert cv2.imwrite(str(directory / "mask_visib" / f"{frame:06d}_000000.png"), mask)
        gt[str(frame)] = [_gt(frame, bop=True)]
        cam[str(frame)] = {"cam_K": camera("primesense").reshape(-1).tolist(), "depth_scale": 0.1}
        info[str(frame)] = [{"bbox_obj": [2, 2, 20, 20], "bbox_visib": [2, 2, 20, 20]}]
    (directory / "scene_gt.json").write_text(json.dumps(gt))
    (directory / "scene_camera.json").write_text(json.dumps(cam))
    (directory / "scene_gt_info.json").write_text(json.dumps(info))
    return directory


def write_v2(directory: Path, sensor: str, frames: tuple[int, ...] = FRAMES) -> Path:
    """Write an original v2 TRAIN object directory (``gt.yml``, ``info.yml``, RGB, depth)."""
    for sub in ("rgb", "depth"):
        (directory / sub).mkdir(parents=True, exist_ok=True)
    gt, info = {}, {}
    width, height = SIZES[sensor]
    for frame in frames:
        rgb = directory / "rgb" / f"{frame:04d}{SUFFIX[sensor]}"
        assert cv2.imwrite(str(rgb), _image(sensor, frame))
        depth = np.full((height, width), 2600, np.uint16)
        assert cv2.imwrite(str(directory / "depth" / f"{frame:04d}.png"), depth)
        gt[frame] = [_gt(frame, bop=False)]
        info[frame] = {
            "cam_K": camera(sensor).reshape(-1).tolist(),
            "depth_scale": 0.1,
            "elev": 85,
            "mode": 0,
        }
    (directory / "gt.yml").write_text(yaml.safe_dump(gt))
    (directory / "info.yml").write_text(yaml.safe_dump(info))
    return directory


def write_test(directory: Path, images: list[np.ndarray] | None = None) -> Path:
    """Write a reserved ``test_primesense`` scene (default: two images unseen in training)."""
    scene = directory / "000001"
    (scene / "rgb").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    width, height = SIZES["primesense"]
    images = images or [rng.integers(0, 256, (height, width, 3), dtype=np.uint8) for _ in "ab"]
    gt, cam, info = {}, {}, {}
    for frame, image in enumerate(images):
        assert cv2.imwrite(str(scene / "rgb" / f"{frame:06d}.png"), image)
        gt[str(frame)] = [_gt(0, bop=True)]
        cam[str(frame)] = {"cam_K": camera("primesense").reshape(-1).tolist(), "depth_scale": 0.1}
        info[str(frame)] = [{"bbox_obj": [2, 2, 20, 20], "bbox_visib": [2, 2, 20, 20]}]
    (scene / "scene_gt.json").write_text(json.dumps(gt))
    (scene / "scene_camera.json").write_text(json.dumps(cam))
    (scene / "scene_gt_info.json").write_text(json.dumps(info))
    return directory


def make_dataset(root: Path) -> dict[str, Path]:
    """Write the whole fixture under ``root``; return its notable paths."""
    primesense = write_primesense(root / "data" / "train_primesense" / f"{OBJ:06d}")
    kinect = write_v2(root / "sensors" / "train_kinect" / f"{OBJ:02d}", "kinect")
    canon = write_v2(root / "sensors" / "train_canon" / f"{OBJ:02d}", "canon")
    test = write_test(root / "data" / "test_primesense")
    mesh, keypoints, audit = write_geometry(root / "objects" / f"obj_{OBJ:06d}" / "geometry")
    return {
        "primesense": primesense,
        "kinect": kinect,
        "canon": canon,
        "test": test,
        "mesh": mesh,
        "keypoints": keypoints,
        "audit": audit,
    }


def v2_archive(path: Path, obj: int, frames: int, *, extra: dict[str, bytes] | None = None) -> Path:
    """Write an official-form v2 TRAIN ZIP of ``frames`` captures of ``obj`` (tiny members)."""
    top = f"{obj:02d}"
    gt = {i: [{"cam_R_m2c": np.eye(3).reshape(-1).tolist(), "cam_t_m2c": [0, 0, 300],
               "obj_bb": [1, 1, 4, 4], "obj_id": obj}] for i in range(frames)}  # fmt: skip
    info = {
        i: {"cam_K": [100, 0, 2, 0, 100, 2, 0, 0, 1], "depth_scale": 0.1} for i in range(frames)
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{top}/gt.yml", yaml.safe_dump(gt))
        archive.writestr(f"{top}/info.yml", yaml.safe_dump(info))
        for i in range(frames):
            archive.writestr(f"{top}/rgb/{i:04d}.png", b"rgb")
            archive.writestr(f"{top}/depth/{i:04d}.png", b"depth")
        for name, payload in (extra or {}).items():
            archive.writestr(name, payload)
    return path


SENSORS = ("primesense", "kinect", "canon")
# Byte hashes of files OpenCV encoded (fixture RGB, written masks, overlays) depend on the
# encoder build, so the comparable summary keeps decoded mask pixels instead.
FILE_HASH_KEYS = frozenset(
    {"rgb_sha256", "prepared_mask_sha256", "source_mask_sha256", "camera_sha256"}
)


def _relative(value: str | None, root: Path) -> str | None:
    if value is None or not Path(value).is_absolute():
        return value
    return Path(value).resolve().relative_to(root.resolve()).as_posix()


def summarize(out: Path, root: Path) -> dict:
    """The path-normalized, encoder-independent content of a prepared multisensor root."""
    ready = json.loads((out / "READY.json").read_text())
    summary: dict = {
        "object": ready["object"],
        "supervision": ready["supervision"],
        "mask_policy": ready["mask_policy"],
        "review_overlays": sorted(ready["review_overlays"]),
        "sensors": {},
    }
    for sensor in SENSORS:
        folder = out / sensor
        source = json.loads((folder / "SOURCE.json").read_text())
        records = []
        for record in source["records"]:
            row = {k: v for k, v in record.items() if k not in FILE_HASH_KEYS}
            row["source_rgb"] = _relative(row["source_rgb"], root)
            if record["mask_file"]:
                mask = cv2.imread(str(folder / record["mask_file"]), cv2.IMREAD_UNCHANGED)
                row["mask_pixels_sha256"] = hashlib.sha256(mask.tobytes()).hexdigest()
            records.append(row)
        geometry = source["mask_geometry"]
        summary["sensors"][sensor] = {
            "annotations": json.loads((folder / "train" / "annotations.json").read_text()),
            "records": records,
            "source": {
                k: v
                for k, v in source.items()
                if k not in {"records", "mask_geometry", "source_gt_sha256", "source_camera_sha256"}
            },
            "mask_geometry": {k: v for k, v in geometry.items() if k not in {"path"}},
            "ready": {
                k: ready["sensors"][sensor][k] for k in ("frames", "masks", "projected_masks")
            },
            "images": sorted(p.name for p in (folder / "train" / "images").iterdir()),
        }
    return summary


def summary_sha256(summary: dict) -> str:
    """Canonical SHA-256 of a summary."""
    text = json.dumps(summary, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def ring_mesh() -> tuple[np.ndarray, np.ndarray]:
    """A flat square annulus (millimeters, eight triangles): its projection has a hole."""
    outer, inner = 30.0, 12.0
    square = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    vertices = [(outer * x, outer * y, 0.0) for x, y in square]
    vertices += [(inner * x, inner * y, 0.0) for x, y in square]
    faces = []
    for i in range(4):
        j = (i + 1) % 4
        faces += [(i, j, 4 + j), (i, 4 + j, 4 + i)]
    return np.array(vertices), np.array(faces)


RING_POSE = (cv2.Rodrigues(np.array([0.3, -0.2, 0.1]))[0], np.array([2.0, -3.0, 250.0]))
