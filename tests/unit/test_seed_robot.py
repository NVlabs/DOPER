# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mode (a): the robot + hand-eye seed, the shared seeded refinement, the scale anchor and the
COLMAP quality gate."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from colmap_fake import install_fake_colmap, recorded_calls
from nurec_fixtures import NAMES, NUREC_TEMPLATE, nurec_recipe_data, rigid, write_bundle

from doper import cli
from doper.nurec.capture import (
    CaptureImage,
    FrameMap,
    ImageSet,
    NormalizedFrame,
    PinholeCamera,
    RobotReference,
    load_image_set,
    load_robot_reference,
    normalize_images,
)
from doper.recipes import load_any_recipe
from doper.reconstruction import colmap_io
from doper.reconstruction.colmap_runner import ColmapCli
from doper.reconstruction.metadata import ReconstructionMetadata
from doper.reconstruction.seeding import (
    ColmapSfmSeed,
    QualityGate,
    RobotHandEyeSeed,
    SeedStrategy,
    anchor_scale,
    reconstruct,
)

SESSION_CAMERA = PinholeCamera(1920, 1080, 1371.2464599609375, 1371.2464599609375,
                               941.239013671875, 591.705322265625, True)  # fmt: skip
REFERENCE_CAMERA = PinholeCamera(1280, 720, 914.164306640625, 914.164306640625, 627.49267578125,
                             394.47021484375, True)  # fmt: skip
DRIFT = 1.0847
RELAXED = QualityGate(min_registered_images=1)
FIXED_TRIANGULATOR = ["--refine_intrinsics", "0"]
FIXED_BUNDLE_ADJUSTER = [
    "--BundleAdjustment.refine_focal_length", "0", "--BundleAdjustment.refine_principal_point", "0",
    "--BundleAdjustment.refine_extra_params", "0", "--BundleAdjustment.refine_rig_from_world", "1",
    "--BundleAdjustment.refine_sensor_from_rig", "0",
]  # fmt: skip
MANY_NAMES = tuple(f"{i:06d}.png" for i in range(24))


@pytest.fixture
def fake(tmp_path):
    return install_fake_colmap(tmp_path / "bin")


def _robot_capture(tmp_path: Path, names=NAMES):
    bundle = tmp_path / "bundle"
    written = write_bundle(bundle, names=names)
    image_set = load_image_set(bundle)
    robot = load_robot_reference(bundle, image_set)
    frames = normalize_images(image_set, tmp_path / "frames")
    return written, image_set, robot, frames


def test_strategies_share_the_abc():
    assert isinstance(RobotHandEyeSeed(), SeedStrategy)
    assert isinstance(ColmapSfmSeed(), SeedStrategy)
    assert RobotHandEyeSeed.name == "robot_handeye" and RobotHandEyeSeed.world_frame == "robot_base"
    assert ColmapSfmSeed.world_frame == "arbitrary"


def test_robot_seed_equals_inverse_base_camera(tmp_path):
    written, image_set, robot, frames = _robot_capture(tmp_path)
    seed = RobotHandEyeSeed().seed(image_set, frames, robot)
    assert seed is not None
    assert list(seed.cam_from_world) == [frame.name for frame in frames.frames]
    for frame in frames.frames:
        T_base_camera = written["poses"][frame.source] @ written["T_mount_camera"]  # noqa: N806
        np.testing.assert_allclose(seed.cam_from_world[frame.name],
                                   np.linalg.inv(T_base_camera), atol=1e-12)  # fmt: skip
        np.testing.assert_allclose(seed.T_world_camera(frame.name), T_base_camera, atol=1e-12)
    assert seed.camera == frames.camera
    assert seed.sources == {frame.name: frame.source for frame in frames.frames}


def test_robot_seed_needs_the_robot_reference(tmp_path):
    _, image_set, _, frames = _robot_capture(tmp_path)
    with pytest.raises(ValueError, match=r"robot reference \(robot.jsonl and handeye.json\)"):
        RobotHandEyeSeed().seed(image_set, frames, None)
    assert ColmapSfmSeed().seed(image_set, frames, None) is None


def _image_set_and_frames(normalized: PinholeCamera) -> tuple[ImageSet, FrameMap]:
    images = (CaptureImage("000002.jpg", "a" * 64, "capture"),)
    frame = NormalizedFrame("frame-000000.png", "000002.jpg", "a" * 64, "b" * 64, "capture")
    return (ImageSet(Path("bundle"), SESSION_CAMERA, images),
            FrameMap((frame,), normalized, SESSION_CAMERA, 1280))  # fmt: skip


def test_intrinsics_scaled_to_normalized_size():
    image_set, frames = _image_set_and_frames(REFERENCE_CAMERA)
    robot_pose = {"000002.jpg": rigid(1)}
    robot = RobotReference(robot_pose, rigid(2))
    seed = RobotHandEyeSeed().seed(image_set, frames, robot)
    assert seed is not None
    # 1920x1080 -> 1280x720: the fixed, scaled intrinsics.
    assert (seed.camera.width, seed.camera.height) == (1280, 720)
    np.testing.assert_allclose([seed.camera.fx, seed.camera.fy, seed.camera.cx, seed.camera.cy],
                               [914.164306640625, 914.164306640625, 627.49267578125,
                                394.47021484375], rtol=0, atol=1e-9)  # fmt: skip
    # A frame map whose camera is not the capture camera at its size is refused.
    wrong = PinholeCamera(1280, 720, 900.0, 900.0, 627.49267578125, 394.47021484375, True)
    image_set, frames = _image_set_and_frames(wrong)
    with pytest.raises(ValueError, match="capture camera"):
        RobotHandEyeSeed().seed(image_set, frames, robot)


def _model(cam_from_world: list[np.ndarray]) -> colmap_io.ColmapModel:
    camera = colmap_io.ColmapCamera(1, "OPENCV", 96, 54, (80.0, 80.0, 47.5, 26.5, 0, 0, 0, 0))
    images = {
        i + 1: colmap_io.ColmapImage(i + 1, colmap_io.quaternion_wxyz(pose[:3, :3]),
                                     tuple(float(v) for v in pose[:3, 3]), 1, f"{i:06d}.png", ())
        for i, pose in enumerate(cam_from_world)
    }  # fmt: skip
    return colmap_io.ColmapModel({1: camera}, images, {})


def test_anchor_scale_recovers_known_drift():
    rng = np.random.default_rng(7)
    seed_poses = [np.linalg.inv(rigid(30 + i)) for i in range(12)]
    rotation = rigid(5)[:3, :3]
    translation = np.array([-0.031, -0.0005, -0.046])
    refined = []
    noise = rng.normal(scale=1e-3, size=(12, 3))
    for pose, jitter in zip(seed_poses, noise, strict=True):
        centre = -pose[:3, :3].T @ pose[:3, 3]
        moved = DRIFT * rotation @ centre + translation + jitter
        R_cw = pose[:3, :3] @ rotation.T  # noqa: N806
        new = np.eye(4)
        new[:3, :3] = R_cw
        new[:3, 3] = -R_cw @ moved
        refined.append(new)
    anchor = anchor_scale(_model(seed_poses), _model(refined))
    # Seed centers (meters) -> refined centers (scene units): the scale is scene units per
    # meter, not its inverse (0.9219).
    assert anchor.scene_units_per_metre == pytest.approx(DRIFT, abs=2e-3)
    assert abs(anchor.scene_units_per_metre - 1 / DRIFT) > 0.1
    assert anchor.method == "sim3_seed_to_refined_centres" and anchor.n == 12
    np.testing.assert_allclose(anchor.rotation, rotation, atol=5e-3)
    assert anchor.rms_mm == pytest.approx(anchor.rms_scene / anchor.scene_units_per_metre * 1000)
    assert anchor.max_mm == pytest.approx(anchor.max_scene / anchor.scene_units_per_metre * 1000)
    assert 0.5 < anchor.rms_mm < 3.0 and anchor.max_scene >= anchor.rms_scene
    # An exact drift has no residual.
    exact = anchor_scale(_model(seed_poses), _model(seed_poses))
    assert exact.scene_units_per_metre == pytest.approx(1.0) and exact.rms_scene < 1e-12


def test_anchor_scale_matches_images_by_name():
    poses = [np.linalg.inv(rigid(30 + i)) for i in range(4)]
    seed, refined = _model(poses), _model(poses[:3])
    assert anchor_scale(seed, refined).n == 3
    with pytest.raises(ValueError, match="at least 3"):
        anchor_scale(seed, _model(poses[:2]))


def test_quality_gate():
    gate = QualityGate()
    good = {"total_images": 35, "registered_images": 35, "largest_model_fraction": 1.0,
            "points": 9765, "median_reprojection_px": 0.7}  # fmt: skip
    assert gate.check(good) == ()
    warn = {**good, "total_images": 50, "largest_model_fraction": 0.69,
            "median_reprojection_px": 2.5}  # fmt: skip
    assert gate.check(warn) == ("COLMAP_LOW_REGISTRATION", "COLMAP_HIGH_REPROJECTION")
    with pytest.raises(ValueError, match="COLMAP_TOO_FEW_IMAGES"):
        gate.check({**good, "registered_images": 19, "largest_model_fraction": 19 / 35})
    with pytest.raises(ValueError, match="COLMAP_MODEL_TOO_SMALL"):
        gate.check({**good, "total_images": 60, "largest_model_fraction": 35 / 60 - 0.2})
    with pytest.raises(ValueError, match="COLMAP_NO_POINTS"):
        gate.check({**good, "points": 0, "median_reprojection_px": None})


def _commands(argv: list[list[str]]) -> list[str]:
    return [call[1] for call in argv]


def test_reconstruct_orchestration_with_fake_runner(fake, tmp_path):
    written, image_set, robot, frames = _robot_capture(tmp_path)
    (fake.parent / "scale.txt").write_text(repr(DRIFT))
    out = tmp_path / "reconstruction"
    runner = ColmapCli(str(fake), load_any_recipe(NUREC_TEMPLATE).colmap, out / "logs")
    flags = {"distortion": "zero", "robot_reference": True}
    metadata = reconstruct(image_set, frames, RobotHandEyeSeed(), runner, out,
                           frames_dir=tmp_path / "frames", robot=robot, capture_flags=flags,
                           inputs={"bundle_sha256": "cd" * 32}, gate=RELAXED)  # fmt: skip
    argv = metadata.colmap.argv
    assert _commands(argv) == [
        "help", "feature_extractor", "exhaustive_matcher",
        "point_triangulator", "bundle_adjuster",
        "point_triangulator", "bundle_adjuster",
        "point_triangulator", "bundle_adjuster",
        "model_converter",
    ]  # fmt: skip
    assert [call[1:] for call in argv] == recorded_calls(fake)
    seed, sparse0 = str(out / "seed"), str(out / "sparse" / "0")
    triangulate_inputs = [call[call.index("--input_path") + 1] for call in argv
                          if call[1] == "point_triangulator"]  # fmt: skip
    assert triangulate_inputs == [seed, sparse0, sparse0]
    assert [call[call.index("--output_path") + 1] for call in argv
            if call[1] == "bundle_adjuster"] == [sparse0] * 3  # fmt: skip
    for call in argv:
        if call[1] == "point_triangulator":
            assert call[-2:] == FIXED_TRIANGULATOR
        if call[1] == "bundle_adjuster":
            assert call[-10:] == FIXED_BUNDLE_ADJUSTER
    assert "--ImageReader.camera_params" in argv[1]
    # The seed: every frame, bound to its database id by name, intrinsics fixed.
    seed_model = colmap_io.read_text_model(out / "seed")
    ids = {r.name: r.colmap_image_id for r in metadata.images}
    assert {i.name: i.image_id for i in seed_model.images.values()} == ids
    assert list(seed_model.images) == sorted(ids.values())
    assert seed_model.points3d == {}
    bindings = json.loads((out / "seed-bindings.json").read_text())
    assert [b["image_id"] for b in bindings] == sorted(ids.values())
    for binding in bindings:
        expected = written["poses"][binding["source_image"]] @ written["T_mount_camera"]
        np.testing.assert_allclose(binding["T_world_camera"], expected, atol=1e-12)
    # Metric metadata: scale = scene units per meter of the drifted refinement.
    assert metadata.strategy == "robot_handeye" and metadata.world_frame == "robot_base"
    assert metadata.length_unit == "colmap" and metadata.is_metric and metadata.reference_path
    assert metadata.scene_units_per_metre == pytest.approx(DRIFT, rel=1e-12)
    anchor = metadata.scale_anchor
    assert anchor is not None and anchor.n == len(NAMES) and anchor.rms_scene < 1e-9
    assert metadata.colmap.selected_model == "sparse/0"
    assert metadata.colmap.warnings == []
    for record in metadata.images:
        assert record.seeded and record.registered and record.excluded_reason is None
        np.testing.assert_allclose(
            record.seed_cam_from_world, seed_model_pose(seed_model, record.name), atol=1e-12
        )
    assert ReconstructionMetadata.read(out / "reconstruction.json") == metadata


def seed_model_pose(model: colmap_io.ColmapModel, name: str) -> np.ndarray:
    return next(colmap_io.cam_from_world(i) for i in model.images.values() if i.name == name)


def test_reconstruct_gate_refuses_a_small_model(fake, tmp_path):
    _, image_set, robot, frames = _robot_capture(tmp_path)
    out = tmp_path / "reconstruction"
    with pytest.raises(ValueError, match="COLMAP_TOO_FEW_IMAGES"):
        reconstruct(image_set, frames, RobotHandEyeSeed(), ColmapCli(str(fake)), out,
                    frames_dir=tmp_path / "frames", robot=robot, capture_flags={},
                    inputs={})  # fmt: skip
    assert not (out / "reconstruction.json").exists()


def test_reconstruct_records_gate_warnings(fake, tmp_path):
    _, image_set, robot, frames = _robot_capture(tmp_path)
    gate = QualityGate(min_registered_images=1, warn_median_reprojection_px=0.1)
    metadata = reconstruct(image_set, frames, RobotHandEyeSeed(), ColmapCli(str(fake)),
                           tmp_path / "out", frames_dir=tmp_path / "frames", robot=robot,
                           capture_flags={}, inputs={}, gate=gate)  # fmt: skip
    assert metadata.colmap.warnings == ["COLMAP_HIGH_REPROJECTION"]


# --- doper prepare RECIPE --stage reconstruct, seed_mode robot_handeye ----------------------


def _recipe(tmp_path: Path, seed_mode: str) -> str:
    data = nurec_recipe_data()
    data["seed_mode"] = seed_mode
    path = tmp_path / f"nurec-{seed_mode}.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


def _prepare(recipe: str, root: Path, stage: str, *extra: str) -> list[str]:
    return ["prepare", recipe, "--root", str(root), "--stage", stage, *extra]


def test_prepare_reconstruct_mode_a(fake, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("DOPER_COLMAP", str(fake))
    bundle, root = tmp_path / "bundle", tmp_path / "root"
    write_bundle(bundle, names=MANY_NAMES)
    recipe = _recipe(tmp_path, "robot_handeye")
    assert cli.main(_prepare(recipe, root, "capture", "--bundle", str(bundle))) == 0
    (fake.parent / "scale.txt").write_text(repr(DRIFT))
    assert cli.main(_prepare(recipe, root, "reconstruct")) == 0
    out = capsys.readouterr().out
    assert "robot_handeye" in out and "scene units per meter 1.0847" in out
    metadata = ReconstructionMetadata.read(root / "reconstruction" / "reconstruction.json")
    assert metadata.strategy == "robot_handeye" and metadata.world_frame == "robot_base"
    assert metadata.scene_units_per_metre == pytest.approx(DRIFT, rel=1e-12)
    assert [r.source_name for r in metadata.images] == list(MANY_NAMES)
    assert all(r.seeded and r.registered for r in metadata.images)


def test_prepare_reconstruct_mode_a_needs_a_robot_reference(fake, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("DOPER_COLMAP", str(fake))
    bundle, root = tmp_path / "bundle", tmp_path / "root"
    write_bundle(bundle, names=MANY_NAMES, robot=False)
    recipe = _recipe(tmp_path, "robot_handeye")
    assert cli.main(_prepare(recipe, root, "capture", "--bundle", str(bundle))) == 0
    capsys.readouterr()
    assert cli.main(_prepare(recipe, root, "reconstruct")) == 1
    assert "robot reference" in capsys.readouterr().out
    assert recorded_calls(fake) == []
    assert not (root / "reconstruction" / "reconstruction.json").exists()


def test_prepare_reconstruct_mode_a_refuses_a_changed_bundle(fake, tmp_path, capsys,
                                                             monkeypatch):  # fmt: skip
    monkeypatch.setenv("DOPER_COLMAP", str(fake))
    bundle, root = tmp_path / "bundle", tmp_path / "root"
    write_bundle(bundle, names=MANY_NAMES)
    recipe = _recipe(tmp_path, "robot_handeye")
    assert cli.main(_prepare(recipe, root, "capture", "--bundle", str(bundle))) == 0
    handeye = json.loads((bundle / "handeye.json").read_text())
    handeye["T_mount_camera"] = rigid(3).tolist()
    (bundle / "handeye.json").write_text(json.dumps(handeye))
    capsys.readouterr()
    assert cli.main(_prepare(recipe, root, "reconstruct")) == 1
    assert "--stage capture" in capsys.readouterr().out


def test_prepare_reconstruct_mode_b_is_gated(fake, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("DOPER_COLMAP", str(fake))
    bundle, root = tmp_path / "bundle", tmp_path / "root"
    write_bundle(bundle, robot=False)
    recipe = _recipe(tmp_path, "colmap_sfm")
    assert cli.main(_prepare(recipe, root, "capture", "--bundle", str(bundle))) == 0
    capsys.readouterr()
    assert cli.main(_prepare(recipe, root, "reconstruct")) == 1
    assert "COLMAP_TOO_FEW_IMAGES" in capsys.readouterr().out
