# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end check of ``examples/minimal``: CAD setup, selector, render and dataset loading.

The example downloads 3DBenchy; this test runs the same steps on a generated bracket STL (mm).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from conftest import REPO_ROOT, run_blender, run_launcher

from doper.config import DataConfig, DoperConfig, ModelConfig
from doper.dataset import DoperDataset
from tests.stl_helpers import bracket_triangles_mm, write_binary_stl

pytestmark = pytest.mark.blender

EXAMPLE = REPO_ROOT / "examples" / "minimal"
MESH_SETUP = REPO_ROOT / "rendering" / "mesh_setup.py"
NUM_KEYPOINTS = 64
NUM_IMAGES = 4
RESOLUTION = 64


def run_mesh_setup(blender: str, mesh: Path, *args: str) -> None:
    """Run ``rendering/mesh_setup.py`` in background Blender on ``mesh`` (an STL in mm)."""
    result = run_blender(
        blender,
        "--",
        "--input",
        mesh,
        "--source-units",
        "mm",
        "--center-geometry",
        *args,
        script=MESH_SETUP,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def write_fast_config(path: Path) -> Path:
    """Copy ``render.yaml`` with CPU Cycles at four samples, one thread and 64 px."""
    config = yaml.safe_load((EXAMPLE / "render.yaml").read_text())
    config["output"]["resolution"] = RESOLUTION
    config["render"].update({"engine": "CYCLES", "cycles_samples": 4, "threads": 1})

    class FlowListDumper(yaml.SafeDumper):
        """Dump lists in flow style; Blender's fallback YAML parser has no block lists."""

    FlowListDumper.add_representer(
        list, lambda dumper, data: dumper.represent_sequence("tag:yaml.org,2002:seq", data, True)
    )
    path.write_text(yaml.dump(config, Dumper=FlowListDumper, sort_keys=False))
    return path


def test_minimal_example_end_to_end(blender, tmp_path):
    mesh = write_binary_stl(tmp_path / "bracket.stl", bracket_triangles_mm(), b"bracket")
    centered = tmp_path / "centered.stl"
    run_mesh_setup(blender, mesh, "--output", str(tmp_path / "draft.blend"),
                   "--centered-mesh-output", str(centered))  # fmt: skip
    selected = tmp_path / "keypoints.json"
    selector = [
        sys.executable, "-m", "doper.keypoint_selection", "--mesh", str(centered),
        "--source-units", "m", "--object-frame", "aabb_centered", "--out-json", str(selected),
        "--keypoint-prefix", "KP_object_", "--num-keypoints", str(NUM_KEYPOINTS),
        "--object-name", "object",
    ]  # fmt: skip
    done = subprocess.run(selector, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr
    run_mesh_setup(blender, mesh, "--output", str(tmp_path / "object.blend"),
                   "--keypoints-input", str(selected),
                   "--usd-output", str(tmp_path / "object.usda"),
                   "--keypoints-output", str(tmp_path / "keypoints_3d.json"))  # fmt: skip
    split = tmp_path / "data" / "train"
    config = write_fast_config(tmp_path / "render.yaml")
    result = run_launcher(
        "--blend-file", str(tmp_path / "object.blend"), "--usd", str(tmp_path / "object.usda"),
        "--config", str(config), "--outdir", str(split), "--num-images", str(NUM_IMAGES),
        "--num-workers", "1", "--seed", "0",
    )  # fmt: skip
    assert result.returncode == 0, result.stdout + result.stderr
    scene_files = sorted((split / "scenes").glob("*.jsonl"))
    assert scene_files and all(path.stat().st_size > 0 for path in scene_files)
    cfg = DoperConfig(
        data=DataConfig(root=str(tmp_path / "data")),
        model=ModelConfig(num_keypoints=NUM_KEYPOINTS),
    )
    dataset = DoperDataset(cfg, "train")
    assert len(dataset) == NUM_IMAGES
    for index in range(len(dataset)):
        sample = dataset[index]
        assert sample["kp_weight"].numel() == NUM_KEYPOINTS
        assert float(sample["mask_weight"]) > 0
        assert float(sample["mask_target"].sum()) > 0
    expected = json.loads(selected.read_text())
    coco = json.loads((split / "annotations.json").read_text())
    assert coco["categories"][0]["keypoints"] == expected["keypoint_names"]
    rendered = json.loads((split / "keypoints_3d.json").read_text())
    assert rendered["keypoint_names"] == expected["keypoint_names"]
    np.testing.assert_allclose(
        rendered["keypoints_3d"], expected["keypoints_3d"], rtol=0, atol=1e-6
    )
