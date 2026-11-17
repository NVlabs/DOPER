# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The NuRec scene contract: importing a trained artifact, its units and its capabilities."""

import hashlib
import json

import numpy as np
import pytest
from nurec_fixtures import (
    SCENE_LINEAGES,
    SCENE_SOURCES,
    WORLD_TO_NRE,
    rig_trajectories,
    write_reconstruction,
    write_usdz,
)

from doper.nurec.capture import PinholeCamera
from doper.nurec.scene import (
    CapabilityError,
    MetricAlignmentError,
    NuRecScene,
    SceneCapabilities,
    import_scene,
    read_rig_trajectories,
)

# The training subset in NuRec's own order (source 1 was left out of training).
TRAINED = ("000003.jpg", "000000.jpg", "000002.jpg")


def _artifact(tmp_path, recon, *, order=TRAINED, **overrides):
    poses = [np.linalg.inv(recon["cam_from_world"][name]) for name in order]
    return write_usdz(tmp_path / "artifact" / "last.usdz", rig_trajectories(poses, **overrides))


def _scene(**overrides):
    values = {
        "artifact": "last.usdz", "artifact_sha256": "a" * 64, "world_to_nre": np.eye(4),
        "cameras": {"x.jpg": np.eye(4)}, "camera": PinholeCamera(8, 6, 5.0, 5.0, 3.5, 2.5, True),
        "scene_units_per_metre": 1.085, "alignment": "metric_anchor", "reconstruction_id": "r",
        "lineages": {"x.jpg": "session"},
    }  # fmt: skip
    values.update(overrides)
    return NuRecScene(**values)


def test_import_scene_binds_training_frames_by_pose(tmp_path):
    recon = write_reconstruction(tmp_path / "reconstruction")
    artifact = _artifact(tmp_path, recon)
    scene = import_scene(artifact, recon["metadata_path"], recon["frames"])
    assert tuple(scene.cameras) == TRAINED
    for name in TRAINED:
        expected = WORLD_TO_NRE @ np.linalg.inv(recon["cam_from_world"][name])
        np.testing.assert_allclose(scene.cameras[name], expected, atol=1e-12)
        assert scene.lineages[name] == SCENE_LINEAGES[SCENE_SOURCES.index(name)]
    assert scene.artifact_sha256 == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert scene.reconstruction_id == recon["metadata"].reconstruction_id
    assert scene.alignment == "metric_anchor"
    assert scene.scene_units_per_metre == 1.085
    np.testing.assert_array_equal(scene.world_to_nre, WORLD_TO_NRE)
    assert (scene.camera.width, scene.camera.fx) == (96, 80.0)
    assert scene.capabilities == SceneCapabilities(rgb=True, depth=False, object_mask=False)


def test_scene_contract_requires_scale(tmp_path):
    with pytest.raises(ValueError, match="scene_units_per_metre"):
        _scene(scene_units_per_metre=None)
    with pytest.raises(ValueError, match="scene_units_per_metre"):
        _scene(scene_units_per_metre=-1.0)
    with pytest.raises(ValueError, match="visual_only"):
        _scene(alignment="visual_only")
    with pytest.raises(ValueError, match="alignment"):
        _scene(alignment="metric")
    # An unanchored (arbitrary-unit) reconstruction imports as visual_only, with no scale.
    recon = write_reconstruction(tmp_path / "reconstruction", metric=False)
    scene = import_scene(_artifact(tmp_path, recon), recon["metadata_path"], recon["frames"])
    assert scene.alignment == "visual_only"
    assert scene.scene_units_per_metre is None


def test_depth_and_mask_capabilities_raise():
    scene = _scene()
    scene.require("rgb")
    for capability in ("depth", "object_mask", "normals"):
        with pytest.raises(CapabilityError, match=capability):
            scene.require(capability)
    with pytest.raises(ValueError, match="depth"):
        _scene(capabilities=SceneCapabilities(depth=True))


def test_visual_only_scene_refuses_metric_labels():
    metric = _scene()
    assert metric.metres_from_scene(1.085) == pytest.approx(1.0)
    assert metric.scene_from_metres(1.0) == pytest.approx(1.085)
    visual = _scene(alignment="visual_only", scene_units_per_metre=None)
    with pytest.raises(MetricAlignmentError, match="visual_only"):
        visual.metres_from_scene(1.0)
    with pytest.raises(MetricAlignmentError, match="visual_only"):
        visual.scene_from_metres(1.0)


def test_import_metric_scene_refuses_a_visual_only_scene(monkeypatch):
    from doper.nurec import scene as scene_module

    metric, visual = _scene(), _scene(alignment="visual_only", scene_units_per_metre=None)
    monkeypatch.setattr(scene_module, "import_scene", lambda *args: metric)
    assert scene_module.import_metric_scene("a.usdz", "r.json", None) is metric
    monkeypatch.setattr(scene_module, "import_scene", lambda *args: visual)
    with pytest.raises(MetricAlignmentError, match="visual_only"):
        scene_module.import_metric_scene("a.usdz", "r.json", None)


def test_scene_arrays_are_frozen():
    scene = _scene()
    with pytest.raises(ValueError):
        scene.cameras["x.jpg"][0, 0] = 2.0
    with pytest.raises(ValueError, match="lineage"):
        _scene(lineages={})


def test_import_refuses_a_frame_that_matches_no_image(tmp_path):
    recon = write_reconstruction(tmp_path / "reconstruction")
    poses = [np.linalg.inv(recon["cam_from_world"][name]) for name in TRAINED]
    poses[1] = poses[1].copy()
    poses[1][:3, 3] += 0.01
    artifact = write_usdz(tmp_path / "last.usdz", rig_trajectories(poses))
    with pytest.raises(ValueError, match="training frame 1 matches no registered image"):
        import_scene(artifact, recon["metadata_path"], recon["frames"])


def test_import_refuses_a_changed_model(tmp_path):
    recon = write_reconstruction(tmp_path / "reconstruction")
    artifact = _artifact(tmp_path, recon)
    images = tmp_path / "reconstruction" / "sparse-text" / "images.txt"
    images.write_text(images.read_text() + "\n")
    with pytest.raises(ValueError, match="reconstruction_id"):
        import_scene(artifact, recon["metadata_path"], recon["frames"])


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"shutter_type": "ROLLING_TOP_TO_BOTTOM"}, "GLOBAL"),
        ({"radial_coeffs": [0.1, 0, 0, 0, 0, 0]}, "distortion"),
        ({"focal_length": [80.0]}, "focal_length"),
    ],
)
def test_import_refuses_unsupported_calibrations(tmp_path, overrides, match):
    recon = write_reconstruction(tmp_path / "reconstruction")
    artifact = _artifact(tmp_path, recon, **overrides)
    with pytest.raises(ValueError, match=match):
        import_scene(artifact, recon["metadata_path"], recon["frames"])


def test_import_refuses_rolling_shutter_pose_pairs(tmp_path):
    recon = write_reconstruction(tmp_path / "reconstruction")
    poses = [np.linalg.inv(recon["cam_from_world"][name]) for name in TRAINED]
    record = rig_trajectories(poses)
    pairs = record["rig_trajectories"][0]["cameras_frame_T_rig_worlds"]["camera1@input"]
    pairs[0][1] = np.eye(4).tolist()
    artifact = write_usdz(tmp_path / "last.usdz", record)
    with pytest.raises(ValueError, match="global-shutter"):
        import_scene(artifact, recon["metadata_path"], recon["frames"])


def test_read_rig_trajectories_needs_the_member(tmp_path):
    import zipfile

    path = tmp_path / "empty.usdz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("checkpoint.ckpt", b"")
    with pytest.raises(ValueError, match="rig_trajectories.json"):
        read_rig_trajectories(path)
    record = rig_trajectories([np.eye(4)])
    assert read_rig_trajectories(write_usdz(tmp_path / "ok.usdz", record)) == json.loads(
        json.dumps(record)
    )
