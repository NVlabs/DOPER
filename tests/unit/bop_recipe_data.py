# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned fixture datasets for the recipe and CLI tests.

``make_bop_download`` writes a small BOP dataset as ZIP parts under ``<root>/source`` with a
profile pinning their SHA-256, so ``doper prepare`` runs the real fetch/verify, extraction,
keypoint selection, conversion, audits and split without a network. Two sphere-like objects
appear in every PBR, validation and test frame; masks are the projected mesh hull and the depth
plane lies behind the objects, so the conversion and projection audits pass.

``make_hot3d_download`` writes HOT3D-form clip archives (native FISHEYE624 images, calibration
and amodal-mask annotations) with a profile, a validation panel and a test image index, for the
validation and test staging.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import yaml

import doper.recipes

WIDTH, HEIGHT = 128, 96
FOCAL = 160.0
CAMERA = np.array([[FOCAL, 0.0, WIDTH / 2], [0.0, FOCAL, HEIGHT / 2], [0.0, 0.0, 1.0]])
OBJECTS = (1, 2)
RADIUS_MM = 40.0
DEPTH_MM = 500.0
PLANE_MM = 2000
PBR_SCENES = ("000000", "000001", "000002")
FRAMES = (0, 1, 2)
RECIPES = Path(doper.recipes.__file__).parent


def sphere() -> tuple[np.ndarray, np.ndarray]:
    """Vertices (mm) and faces of a subdivided icosahedron, rounded to six decimals."""
    import trimesh

    mesh = trimesh.creation.icosphere(subdivisions=2, radius=RADIUS_MM)
    return np.round(np.asarray(mesh.vertices), 6), np.asarray(mesh.faces)


def write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> Path:
    """Write an ASCII PLY."""
    lines = [
        "ply", "format ascii 1.0", f"element vertex {len(vertices)}",
        "property float x", "property float y", "property float z",
        f"element face {len(faces)}", "property list uchar int vertex_indices", "end_header",
    ]  # fmt: skip
    lines += [f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    lines += [f"3 {a} {b} {c}" for a, b, c in faces]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def write_models(directory: Path) -> None:
    """Write both objects' PLY models and ``models_info.json`` (mm)."""
    vertices, faces = sphere()
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    info = {
        "diameter": 2 * RADIUS_MM,
        **dict(zip(("min_x", "min_y", "min_z"), low.tolist(), strict=True)),
        **dict(zip(("size_x", "size_y", "size_z"), (high - low).tolist(), strict=True)),
    }
    for obj in OBJECTS:
        write_ply(directory / f"obj_{obj:06d}.ply", vertices, faces)
    (directory / "models_info.json").write_text(json.dumps({str(o): info for o in OBJECTS}))


def _rotation(rng: np.random.Generator) -> np.ndarray:
    return cv2.Rodrigues(rng.uniform(-np.pi, np.pi, 3))[0]


def _hull_mask(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    vertices, _ = sphere()
    camera = vertices @ rotation.T + translation
    uv = (camera @ CAMERA.T)[:, :2] / camera[:, 2:3]
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(np.round(uv).astype(np.int32)), 255)
    return mask


def write_scene(scene: Path, seed: int, *, frames: tuple[int, ...] = FRAMES) -> None:
    """One BOP scene: RGB, depth, visible masks, GT, camera and GT info for both objects."""
    rng = np.random.default_rng(seed)
    for name in ("rgb", "depth", "mask_visib"):
        (scene / name).mkdir(parents=True, exist_ok=True)
    gt, cameras, infos = {}, {}, {}
    for frame in frames:
        image = rng.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)
        assert cv2.imwrite(str(scene / "rgb" / f"{frame:06d}.jpg"), image)
        depth = np.full((HEIGHT, WIDTH), PLANE_MM, np.uint16)
        assert cv2.imwrite(str(scene / "depth" / f"{frame:06d}.png"), depth)
        gt[str(frame)], infos[str(frame)] = [], []
        for index, obj in enumerate(OBJECTS):
            rotation = _rotation(rng)
            x = (-1 if index == 0 else 1) * 45.0 + rng.uniform(-10, 10)
            translation = np.array([x, rng.uniform(-15, 15), DEPTH_MM])
            mask = _hull_mask(rotation, translation)
            assert cv2.imwrite(str(scene / "mask_visib" / f"{frame:06d}_{index:06d}.png"), mask)
            ys, xs = np.nonzero(mask)
            box = [int(xs.min()), int(ys.min()), int(np.ptp(xs) + 1), int(np.ptp(ys) + 1)]
            gt[str(frame)].append(
                {
                    "obj_id": obj,
                    "cam_R_m2c": rotation.ravel().tolist(),
                    "cam_t_m2c": translation.tolist(),
                }
            )
            pixels = int(np.count_nonzero(mask))
            infos[str(frame)].append(
                {
                    "bbox_obj": box,
                    "bbox_visib": box,
                    "px_count_all": pixels,
                    "px_count_valid": pixels,
                    "px_count_visib": pixels,
                    "visib_fract": 1.0,
                }
            )
        cameras[str(frame)] = {"cam_K": CAMERA.ravel().tolist(), "depth_scale": 1.0}
    (scene / "scene_gt.json").write_text(json.dumps(gt))
    (scene / "scene_camera.json").write_text(json.dumps(cameras))
    (scene / "scene_gt_info.json").write_text(json.dumps(infos))


def _zip(path: Path, tree: Path, roots: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for root in roots:
            for file in sorted((tree / root).rglob("*")):
                if file.is_file():
                    archive.write(file, file.relative_to(tree).as_posix())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profile(directory: Path, name: str, files: dict[str, Path], parts: dict, meta: dict) -> Path:
    profile = {
        "name": name,
        "hf_repo": "bop-benchmark/fixture",
        "revision": "0" * 40,
        "files": {n: _sha(p) for n, p in files.items()},
        "parts": parts,
        "meta": meta,
    }
    path = directory / f"{name}.json"
    path.write_text(json.dumps(profile, indent=1))
    return path


def make_bop_download(root: Path, profile_dir: Path) -> Path:
    """Write the BOP fixture's ZIP parts under ``root/source``; return its profile path."""
    tree = profile_dir / "release"
    write_models(tree / "models")
    write_models(tree / "models_eval")
    for seed, scene in enumerate(PBR_SCENES):
        write_scene(tree / "train_pbr" / scene, seed)
    write_scene(tree / "val" / "000001", 10)
    write_scene(tree / "test" / "000001", 20)
    archives = {
        "fixture_models.zip": ("models", "models_eval"),
        "fixture_train_pbr.zip": ("train_pbr",),
        "fixture_val.zip": ("val",),
        "fixture_test.zip": ("test",),
    }
    files = {}
    for name, roots in archives.items():
        _zip(root / "source" / name, tree, roots)
        files[name] = root / "source" / name
    parts = {
        "models": ["fixture_models.zip"],
        "pbr": ["fixture_train_pbr.zip"],
        "val": ["fixture_val.zip"],
        "test": ["fixture_test.zip"],
    }
    meta = {"objects": list(OBJECTS), "test_population": {"images": len(FRAMES), "scenes": 1}}
    return _profile(profile_dir, "fixture", files, parts, meta)


def write_recipe(path: Path, base: str, profile: Path, **changes: object) -> Path:
    """Write a fixture recipe: a packaged recipe with fixture objects and profile."""
    recipe = yaml.safe_load((RECIPES / f"{base}.yaml").read_text())
    recipe.update(name=f"fixture_{base}", profile=str(profile), objects=list(OBJECTS))
    recipe["generated_objects"] = []
    recipe.pop("generated", None)  # the packaged campaigns name non-fixture objects
    for key, value in changes.items():
        section, _, field = key.partition("__")
        if field:
            recipe[section][field] = value
        else:
            recipe[section] = value
    path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    return path


# --------------------------------------------------------------------------------------
# HOT3D clips
# --------------------------------------------------------------------------------------
NATIVE_WIDTH, NATIVE_HEIGHT = 64, 48
STREAMS = {"Aria": "214-1", "Quest3": "1201-1"}
TEST_FRAMES = (29, 59, 89, 119, 149)
IDENTITY_POSE = {"quaternion_wxyz": [1.0, 0.0, 0.0, 0.0], "translation_xyz": [0.0, 0.0, 0.0]}


def native_calibration() -> dict:
    """A distortion-free FISHEYE624 calibration of the native fixture images."""
    return {
        "image_width": NATIVE_WIDTH,
        "image_height": NATIVE_HEIGHT,
        "projection_model_type": "CameraModelType.FISHEYE624",
        "projection_params": [40.0, NATIVE_WIDTH / 2, NATIVE_HEIGHT / 2] + [0.0] * 12,
        "label": "fixture",
        "serial_number": "fixture",
    }


def encode_rle(mask: np.ndarray) -> list[int]:
    """The official one-based absolute start / run length codec."""
    pixels = np.concatenate([[0], (mask > 0).flatten().astype(np.int64), [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return runs.tolist()


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _objects_json(stream: str, frame: int) -> bytes:
    rows = {}
    for index, obj in enumerate(OBJECTS):
        mask = np.zeros((NATIVE_HEIGHT, NATIVE_WIDTH), np.uint8)
        center = (16 + 32 * index, 24)
        cv2.circle(mask, center, 8, 255, -1)
        x = (center[0] - NATIVE_WIDTH / 2) / 40.0 * 0.4
        pose = {"quaternion_wxyz": [1.0, 0.0, 0.0, 0.0], "translation_xyz": [x, 0.0, 0.4]}
        encoded = {"height": NATIVE_HEIGHT, "width": NATIVE_WIDTH, "rle": encode_rle(mask)}
        rows[str(obj)] = [{"T_world_from_object": pose, "masks_amodal": {stream: encoded}}]
    return json.dumps(rows).encode()


def write_clip(
    path: Path,
    stream: str,
    frames: tuple[int, ...],
    *,
    annotated: bool,
    supports: tuple[str, ...] = (),
) -> None:
    """A HOT3D clip archive: per-frame native image, cameras and (annotated) objects.

    ``supports`` adds synchronized mono streams; support ``i`` sits ``0.06 * (i + 1)`` m along
    the world x axis from the reference camera.
    """
    rng = np.random.default_rng(len(frames) + int(annotated))
    support_rng = np.random.default_rng(7)
    path.parent.mkdir(parents=True, exist_ok=True)
    camera = {stream: {"calibration": native_calibration(), "T_world_from_camera": IDENTITY_POSE}}
    for index, name in enumerate(supports):
        pose = {
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "translation_xyz": [0.06 * (index + 1), 0, 0],
        }
        camera[name] = {"calibration": native_calibration(), "T_world_from_camera": pose}
    with tarfile.open(path, "w") as tar:
        for frame in frames:
            image = rng.integers(0, 256, (NATIVE_HEIGHT, NATIVE_WIDTH, 3), dtype=np.uint8)
            ok, data = cv2.imencode(".jpg", image)
            assert ok
            _add(tar, f"{frame:06d}.image_{stream}.jpg", data.tobytes())
            for name in supports:
                mono = support_rng.integers(0, 256, (NATIVE_HEIGHT, NATIVE_WIDTH), dtype=np.uint8)
                ok, data = cv2.imencode(".jpg", mono)
                assert ok
                _add(tar, f"{frame:06d}.image_{name}.jpg", data.tobytes())
            _add(tar, f"{frame:06d}.cameras.json", json.dumps(camera).encode())
            if annotated:
                _add(tar, f"{frame:06d}.objects.json", _objects_json(stream, frame))


def make_hot3d_download(root: Path, profile_dir: Path) -> Path:
    """Write HOT3D validation and test clips under ``root/source``; return the profile path."""
    clips = [
        ("Aria", "P0001_a", 1856, (29, 59)),
        ("Quest3", "P0002_b", 1900, (29,)),
    ]
    rows, files = [], {}
    for device, sequence, clip_id, frames in clips:
        name = f"train_{device.lower()}/clip-{clip_id:06d}.tar"
        write_clip(root / "source" / name, STREAMS[device], frames, annotated=True)
        files[name] = root / "source" / name
        rows.append(
            {
                "device": device,
                "sequence_id": sequence,
                "clip_id": clip_id,
                "source_path": name,
                "frames": list(frames),
            }
        )
    test_name = "test_aria/clip-003365.tar"
    write_clip(
        root / "source" / test_name,
        STREAMS["Aria"],
        TEST_FRAMES,
        annotated=False,
        supports=("1201-1", "1201-2"),
    )
    files[test_name] = root / "source" / test_name
    resources = profile_dir / "resources"
    resources.mkdir(parents=True, exist_ok=True)
    panel = resources / "panel.json"
    panel.write_text(json.dumps({"clip_rows": rows}))
    index = resources / "index.json"
    images = [{"scene_id": 3365, "im_id": f} for f in TEST_FRAMES]
    index.write_text(json.dumps({"images": images}))
    meta = {
        "objects": list(OBJECTS),
        "evaluation_streams": STREAMS,
        "resources": {
            "validation_panel": {"file": "resources/panel.json", "sha256": _sha(panel)},
            "test_image_index": {"file": "resources/index.json", "sha256": _sha(index)},
        },
        "test_population": {
            "images": len(TEST_FRAMES),
            "scenes": 1,
            "index_resource": "test_image_index",
        },
    }
    parts = {"val": [r["source_path"] for r in rows], "test": [test_name]}
    return _profile(profile_dir, "hot3d", files, parts, meta)
