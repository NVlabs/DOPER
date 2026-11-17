# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small real render/resume smoke through the launcher, mono and stereo."""

import hashlib
import json

import numpy as np
import pytest
import yaml
from conftest import part_recipe, run_launcher, snapshot

pytestmark = pytest.mark.blender


@pytest.mark.parametrize("stereo", [False, True], ids=["mono", "stereo"])
def test_blender_render_resume_and_identity_rejection(blender, part_inputs, tmp_path, stereo):
    recipe = part_recipe(part_inputs, stereo=stereo)
    first_name = "000000_left.png" if stereo else "000000.png"
    missing_name = "000001_left.png" if stereo else "000001.png"
    config = tmp_path / "recipe.yaml"
    config.write_text(yaml.safe_dump(recipe))
    out = tmp_path / "out"
    keypoints = part_inputs / "correct_keypoints.json"
    command = [
        "--num-images",
        "2",
        "--blend-file",
        str(part_inputs / "part.blend"),
        "--config",
        str(config),
        "--outdir",
        str(out),
        "--blender-bin",
        blender,
        "--keypoints-json",
        str(keypoints),
    ]

    def invoke(*extra):
        return run_launcher(*command, *extra)

    def log():
        return (out / "worker_000.log").read_text()

    first = invoke("--num-workers", "1")
    assert first.returncode == 0, first.stdout + first.stderr + "\n" + log()
    assert np.allclose(
        json.loads((out / "keypoints_3d.json").read_text())["keypoints_3d"],
        json.loads(keypoints.read_text())["keypoints_3d"],
        rtol=0,
        atol=1e-6,
    )

    def hashes():
        return {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (out / "images").glob("*.png")
        }

    images = hashes()
    assert len(images) == (4 if stereo else 2)
    resumed = invoke("--num-workers", "2", "--resume")
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert images == hashes()
    manifest = json.loads((out / "manifests/shard_00000000_00000002.json").read_text())
    assert manifest["status"] == "complete" and len(manifest["attempts"]) == 1
    assert manifest["reused_sample_indices"] == [0, 1]
    assert json.loads((out / "manifest.json").read_text())["status"] == "complete"
    # A lost image after a checkpoint must be rendered again; complete samples stay untouched.
    labels_before = json.loads((out / "annotations.json").read_text())
    (out / "images" / missing_name).unlink()
    (out / "annotations_worker_001.json").rename(out / "annotations_worker_001.partial.json")
    recovered = invoke("--num-workers", "1", "--resume")
    assert recovered.returncode == 0, recovered.stdout + recovered.stderr
    assert (
        hashlib.sha256((out / "images" / first_name).read_bytes()).hexdigest() == images[first_name]
    )
    assert (out / "images" / missing_name).is_file()
    assert json.loads((out / "annotations.json").read_text()) == labels_before
    recovery = json.loads((out / "manifests/shard_00000000_00000002.json").read_text())
    assert recovery["reused_sample_indices"] == [0]
    assert len(recovery["attempts"]) == 2
    legacy = invoke(
        "--num-workers",
        "1",
        "--num-images",
        "1",
        "--no-scenes",
        "--outdir",
        str(tmp_path / "legacy"),
    )
    assert legacy.returncode == 0, legacy.stdout + legacy.stderr
    assert (tmp_path / "legacy/annotations.json").is_file()

    before = snapshot(out)
    rejected = invoke("--num-workers", "2", "--resume", "--seed", "99")
    assert rejected.returncode != 0 and "identity mismatch" in rejected.stderr
    assert snapshot(out) == before
    recipe["render"]["exposure"] = 1.1
    config.write_text(yaml.safe_dump(recipe))
    rejected = invoke("--num-workers", "2", "--resume")
    assert rejected.returncode != 0 and "identity mismatch" in rejected.stderr
    assert snapshot(out) == before
