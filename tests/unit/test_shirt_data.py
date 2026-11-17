# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SHIRT labels, camera, metadata and the packaged profile (``doper.shirt.data``)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import shirt_frames as sf

from doper.shirt import data


@pytest.fixture
def shirt_root(tmp_path: Path) -> Path:
    root = tmp_path / "shirt"
    sf.write_shirt_tree(root)
    return root


def test_label_field_aliases(shirt_root, tmp_path):
    released = data.load_labels(shirt_root / "roe1")
    aliased = data.load_labels(shirt_root / "roe2")
    for labels, slot in ((released, 0), (aliased, 1)):
        frames = sf.trajectory_frames(slot)
        assert [label.index for label in labels] == list(range(sf.FRAMES))
        assert [label.filename for label in labels] == [f["filename"] for f in frames]
        for label, frame in zip(labels, frames, strict=True):
            assert label.quaternion.dtype == np.float64 and label.quaternion.shape == (4,)
            np.testing.assert_array_equal(label.quaternion, frame["quaternion"])
            np.testing.assert_array_equal(label.translation, frame["translation"])
    # The other wrappers of the released schema, and a label without its pose fields.
    for key in ("labels", "poses"):
        roe = tmp_path / key / "roe9"
        roe.mkdir(parents=True)
        (roe / "roe9.json").write_text(json.dumps({key: [{"filename": "a.jpg",
            "q_vbs2tango_true": [1, 0, 0, 0], "translation": [0, 0, 5]}]}))  # fmt: skip
        assert [label.filename for label in data.load_labels(roe)] == ["a.jpg"]
    (roe / "roe9.json").write_text(json.dumps([{"filename": "a.jpg", "quaternion": [1, 0, 0, 0]}]))
    with pytest.raises(ValueError, match="missing pose fields"):
        data.load_labels(roe)
    (roe / "roe9.json").write_text(json.dumps({"frames": []}))
    with pytest.raises(ValueError, match="unsupported SHIRT label schema"):
        data.load_labels(roe)


def test_camera_schemas_and_image_size(shirt_root, tmp_path):
    camera = data.load_camera(shirt_root)
    np.testing.assert_array_equal(camera.K, sf.synthetic_shirt().CAMERA_MATRIX)
    assert (camera.width, camera.height) == (sf.CAMERA_SIZE, sf.CAMERA_SIZE)
    assert camera.dist.shape == (5,)
    other = tmp_path / "other"
    other.mkdir()
    payload = {"fx": 800.0, "fy": 810.0, "cx": 500.0, "cy": 400.0, "distortion": [0.1, 0, 0, 0, 0],
               "Nu": 1920, "Nv": 1200}  # fmt: skip
    (other / "camera.json").write_text(json.dumps(payload))
    focal = data.load_camera(other)
    np.testing.assert_array_equal(focal.K, [[800.0, 0, 500.0], [0, 810.0, 400.0], [0, 0, 1]])
    assert focal.dist[0] == 0.1 and (focal.width, focal.height) == (1920, 1200)
    (other / "camera.json").write_text(json.dumps({"K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}))
    with pytest.raises(ValueError, match="unsupported SHIRT camera schema"):
        data.load_camera(other)
    (other / "camera.json").write_text(json.dumps({**payload, "Nu": None}))
    with pytest.raises(ValueError, match="Nu"):
        data.load_camera(other)


def test_metadata_and_profile(shirt_root):
    metadata = data.load_metadata(shirt_root / "roe1")
    assert metadata["pSim"]["cam_step"] == sf.synthetic_shirt().CAMERA_STEP_S
    profile = data.load_profile()
    assert profile.name == "shirt" and profile.trajectories == ("roe1", "roe2")
    assert profile.domains == ("synthetic", "lightbox") and profile.paper_domain == "lightbox"
    assert profile.image_size == (1920, 1200)
    assert data.image_path(shirt_root, "roe1", "lightbox", "img000001.png") == (
        shirt_root / "roe1" / "lightbox" / "images" / "img000001.png"
    )


def test_shirt_recipe_trajectories_domain_and_pin(tmp_path):
    from doper.recipes import load_any_recipe

    recipe = load_any_recipe("shirt")
    assert recipe.trajectories == ("roe1", "roe2") and recipe.domain == "lightbox"
    assert recipe.checkpoint_sha256 == next(iter(recipe.legacy_checkpoints))
    base = {"name": "shirt", "kind": "shirt", "profile": "shirt"}
    for change, message in (
        ({"trajectories": []}, "trajectories"),
        ({"trajectories": "roe1"}, "trajectories"),
        ({"domain": "sunlamp"}, "domain"),
        ({"checkpoint_sha256": "A" * 64}, "checkpoint_sha256"),
    ):
        path = tmp_path / "bad.yaml"
        path.write_text(json.dumps({**base, **change}))
        with pytest.raises(ValueError, match=message):
            load_any_recipe(path)
    path.write_text(json.dumps({**base, "trajectories": ["roe2"]}))
    loaded = load_any_recipe(path)
    assert loaded.trajectories == ("roe2",) and loaded.checkpoint_sha256 is None
