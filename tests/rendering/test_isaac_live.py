# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live Isaac Sim renders; run only inside the Isaac Sim 6.0.1 container on a GPU host.

Example (from the checkout, with pytest importable by Isaac's Python)::

    /isaac-sim/python.sh -m pytest -m isaac tests/rendering/test_isaac_live.py
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from rendering.config import load_render_config
from rendering.dataset_contract import validate_dataset_records
from rendering.isaac.isaac_parallel import resolve_isaac_python
from rendering.synth_parallel import generate_blender_scenes
from rendering.usd_mesh import write_mesh_usda

pytestmark = pytest.mark.isaac

REPO_ROOT = Path(__file__).resolve().parents[2]
NUM_IMAGES = 4
RESOLUTION = 128
NUM_KEYPOINTS = 64
SEED = 7
RENDER_TIMEOUT_SEC = 3600


def write_object(path: Path) -> None:
    """A 10 cm lat-long sphere with 64 surface keypoints, as a writer-generated USDA."""
    rings, segments, radius = 12, 16, 0.05
    points = [(0.0, 0.0, radius)]
    for ring in range(1, rings):
        theta = np.pi * ring / rings
        for segment in range(segments):
            phi = 2 * np.pi * segment / segments
            points.append(
                (
                    radius * np.sin(theta) * np.cos(phi),
                    radius * np.sin(theta) * np.sin(phi),
                    radius * np.cos(theta),
                )
            )
    points.append((0.0, 0.0, -radius))
    bottom = len(points) - 1
    faces = []

    def ring_vertex(ring, segment):
        return 1 + (ring - 1) * segments + segment % segments

    for segment in range(segments):
        faces.append([0, ring_vertex(1, segment), ring_vertex(1, segment + 1)])
        faces.append([bottom, ring_vertex(rings - 1, segment + 1), ring_vertex(rings - 1, segment)])
    for ring in range(1, rings - 1):
        for segment in range(segments):
            faces.append(
                [
                    ring_vertex(ring, segment),
                    ring_vertex(ring + 1, segment),
                    ring_vertex(ring + 1, segment + 1),
                    ring_vertex(ring, segment + 1),
                ]
            )
    chosen = np.linspace(0, len(points) - 1, NUM_KEYPOINTS).astype(int)
    write_mesh_usda(
        str(path),
        object_name="object",
        points=[tuple(float(v) for v in point) for point in points],
        face_vertex_counts=[len(face) for face in faces],
        face_vertex_indices=[index for face in faces for index in face],
        keypoint_names=[f"KP_object_{i:03d}" for i in range(NUM_KEYPOINTS)],
        keypoints_3d=[tuple(float(v) for v in points[i]) for i in chosen],
    )


def write_config(path: Path, usd: Path) -> None:
    """A small asset-free recipe: color ground and dome, no distractors."""
    recipe = {
        "output": {"resolution": RESOLUTION, "render_masks": True},
        "object": {"usd_path": str(usd), "num_keypoints": NUM_KEYPOINTS},
        "scene": {
            "num_distractors_min": 0,
            "num_distractors_max": 0,
            "ground_style": "procedural",
            "world_background_modes": "studio,random",
        },
        "visibility": {"min_keypoints": 8},
    }
    path.write_text(yaml.safe_dump(recipe))


@pytest.fixture(scope="module")
def isaac_dataset(tmp_path_factory):
    """Render one 4-image dataset with the Isaac launcher; return ``(root, outdir)``."""
    if resolve_isaac_python(None) is None:
        pytest.skip("Isaac Sim Python not found (set DOPER_ISAAC_PYTHON)")
    root = tmp_path_factory.mktemp("isaac")
    write_object(root / "object.usda")
    write_config(root / "render.yaml", root / "object.usda")
    out = root / "out"
    cmd = [sys.executable, str(REPO_ROOT / "rendering/isaac/isaac_parallel.py")]
    cmd += ["--config", str(root / "render.yaml"), "--outdir", str(out)]
    cmd += ["--num-images", str(NUM_IMAGES), "--seed", str(SEED)]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=RENDER_TIMEOUT_SEC)
    logs = "".join(path.read_text()[-3000:] for path in sorted(out.glob("worker_*.log")))
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:] + logs
    return root, out


def test_isaac_render_writes_valid_dataset(isaac_dataset):
    _, out = isaac_dataset
    payload = json.loads((out / "annotations.json").read_text())
    errors = validate_dataset_records(
        payload,
        out,
        start_idx=0,
        end_idx=NUM_IMAGES,
        images_per_sample=1,
        render_masks=True,
        num_keypoints=NUM_KEYPOINTS,
    )
    assert errors == []
    assert len(payload["images"]) == len(payload["annotations"]) == NUM_IMAGES
    assert payload["categories"][0]["keypoints"] == [
        f"KP_object_{i:03d}" for i in range(NUM_KEYPOINTS)
    ]
    for image, annotation in zip(payload["images"], payload["annotations"], strict=True):
        assert (image["width"], image["height"]) == (RESOLUTION, RESOLUTION)
        assert len(annotation["keypoints"]) == 3 * NUM_KEYPOINTS
        assert annotation["num_keypoints"] > 0
        rgb = np.asarray(Image.open(out / "images" / image["file_name"]))
        mask = np.asarray(Image.open(out / annotation["mask_file"]))
        assert rgb.shape == (RESOLUTION, RESOLUTION, 3) and rgb.std() > 0
        assert mask.shape == (RESOLUTION, RESOLUTION) and mask.any()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    settings = json.loads((out / "settings_worker_000.json").read_text())
    assert settings["runtime"]["isaacsim"].startswith("6.0.1")


def test_isaac_and_blender_share_scenes(isaac_dataset, tmp_path):
    root, out = isaac_dataset
    cfg = load_render_config(root / "render.yaml")
    blender = generate_blender_scenes(
        cfg, str(root / "object.usda"), None, SEED, 0, NUM_IMAGES, tmp_path
    )
    isaac = out / "scenes" / f"scenes_{0:08d}_{NUM_IMAGES:08d}.jsonl"
    assert Path(blender["path"]).read_bytes() == isaac.read_bytes()
