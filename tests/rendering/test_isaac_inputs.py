# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker preflight rejects invalid input before Kit or output creation."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from conftest import snapshot

from rendering.config import load_render_config
from rendering.isaac.inputs import load_inputs
from rendering.scene_spec import generate_scenes
from rendering.usd_mesh import write_mesh_usda


@pytest.fixture
def inputs(tmp_path):
    usd = tmp_path / "object.usda"
    write_mesh_usda(
        str(usd),
        object_name="object",
        points=[(0, 0, 0), (0.1, 0, 0), (0, 0.1, 0)],
        face_vertex_counts=[3],
        face_vertex_indices=[0, 1, 2],
        keypoint_names=["KP_000", "KP_001"],
        keypoints_3d=[(0, 0, 0), (0.1, 0, 0)],
    )
    config = tmp_path / "recipe.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "object": {"usd_path": str(usd), "num_keypoints": 2},
                "scene": {"num_distractors_min": 0, "num_distractors_max": 0},
            }
        )
    )
    scenes = tmp_path / "scenes.jsonl"
    cfg = load_render_config(config)
    generate_scenes(
        cfg,
        usd_path=str(usd),
        keypoints_json=None,
        seed=42,
        start_idx=0,
        end_idx=3,
        out_path=str(scenes),
    )
    return SimpleNamespace(
        config=str(config),
        usd=None,
        keypoints_json=None,
        start_idx=1,
        end_idx=3,
        seed=42,
        worker_id=0,
        gpu_device=0,
        scenes=str(scenes),
        outdir=str(tmp_path / "out"),
    )


def config_snapshot(args):
    return snapshot(Path(args.config).parent)


def test_valid_input_subrange_is_loaded_without_mutation(inputs):
    before = config_snapshot(inputs)
    loaded = load_inputs(inputs)
    assert loaded.kp_names == ["KP_000", "KP_001"]
    assert loaded.kp_local.shape == (2, 3)
    assert set(loaded.scenes) == {1, 2}
    assert np.isfinite(loaded.intr.fx)
    assert config_snapshot(inputs) == before
    assert not Path(inputs.outdir).exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("worker_id", -1),
        ("gpu_device", -1),
        ("start_idx", -1),
        ("end_idx", 1),
        ("seed", 4.5),
        ("gpu_device", True),
        ("scenes", None),
    ],
)
def test_invalid_worker_parameters_do_not_write(inputs, field, value):
    setattr(inputs, field, value)
    before = config_snapshot(inputs)
    with pytest.raises(ValueError):
        load_inputs(inputs)
    assert config_snapshot(inputs) == before


def rewrite_scenes(args, change):
    path = Path(args.scenes)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    change(records)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda records: records.append(records[-1]), "duplicate"),
        (lambda records: records.pop(), "missing requested"),
        (lambda records: records[1].update(seed=10), "seed"),
        (lambda records: records[1].update(version=-1), "version"),
        (lambda records: records[1].update(index=True), "index"),
        (lambda records: records[1].update(catalog_fingerprint="stale"), "catalog"),
        (lambda records: records[1]["object"].update(pose=[[0]]), "object.pose"),
        (
            lambda records: records[1]["object"].update(center=[float("nan"), 0, 0]),
            "nonfinite",
        ),
        (lambda records: records[1].pop("lights"), "lights"),
    ],
)
def test_invalid_scene_file_rejected_without_mutation(inputs, change, message):
    rewrite_scenes(inputs, change)
    before = config_snapshot(inputs)
    with pytest.raises(ValueError, match=message):
        load_inputs(inputs)
    assert config_snapshot(inputs) == before


def test_matching_optional_catalog_fingerprint(inputs):
    catalog = load_inputs(inputs).catalog
    rewrite_scenes(
        inputs,
        lambda records: records[1].update(catalog_fingerprint=catalog.fingerprint),
    )
    assert load_inputs(inputs).scenes[1]["catalog_fingerprint"] == catalog.fingerprint


@pytest.mark.parametrize(
    ("names", "points", "message"),
    [
        (["KP_001", "KP_000"], [[0, 0, 0], [0.1, 0, 0]], "differ"),
        (["KP_000"], [[0, 0, 0]], "differ"),
        (["KP_000", "KP_001"], [[0, 0, 0]], "K x 3"),
        (["KP_000", "KP_001"], [[0, 0, 0], [0.2, 0, 0]], "differ"),
        (["KP_000", "KP_001"], [[0, 0, 0], [float("inf"), 0, 0]], "finite"),
        (["KP_000", "KP_000"], [[0, 0, 0], [0.1, 0, 0]], "unique"),
    ],
)
def test_supplied_keypoints_must_match_embedded_schema(inputs, names, points, message):
    path = Path(inputs.config).parent / "keypoints.json"
    path.write_text(json.dumps({"keypoint_names": names, "keypoints_3d": points}))
    inputs.keypoints_json = str(path)
    before = config_snapshot(inputs)
    with pytest.raises(ValueError, match=message):
        load_inputs(inputs)
    assert config_snapshot(inputs) == before


def test_explicit_keypoints_override_wins_over_stale_config_path(inputs):
    config = Path(inputs.config)
    raw = yaml.safe_load(config.read_text())
    raw["object"]["keypoints_json"] = "missing-stale.json"
    config.write_text(yaml.safe_dump(raw))
    supplied = Path(inputs.config).parent / "override.json"
    supplied.write_text(
        json.dumps(
            {
                "keypoint_names": ["KP_000", "KP_001"],
                "keypoints_3d": [[0, 0, 0], [0.1, 0, 0]],
            }
        )
    )
    inputs.keypoints_json = str(supplied)
    assert load_inputs(inputs).kp_names == ["KP_000", "KP_001"]


def test_unknown_backend_setting_rejected_before_render(inputs):
    config = Path(inputs.config)
    raw = yaml.safe_load(config.read_text())
    raw["isaac"] = {"spps": 8}
    config.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="unknown isaac"):
        load_inputs(inputs)
    assert not Path(inputs.outdir).exists()
