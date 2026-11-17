# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The COLMAP CLI runner, model selection, mode (b) and ``reconstruction.json``."""

import json
import struct
from pathlib import Path

import numpy as np
import pytest
import yaml
from colmap_fake import install_fake_colmap, recorded_calls
from nurec_fixtures import NAMES, NUREC_TEMPLATE, nurec_recipe_data, write_bundle

from doper import cli
from doper.nurec.capture import PinholeCamera, load_image_set, normalize_images
from doper.recipes import load_any_recipe
from doper.reconstruction.colmap_runner import (
    DEFAULT_FLAGS,
    ColmapCli,
    ColmapError,
    count_registered_images,
    select_largest_model,
)
from doper.reconstruction.metadata import ReconstructionMetadata, ScaleAnchor
from doper.reconstruction.seeding import ColmapSfmSeed, QualityGate, reconstruct_sfm

REFERENCE_CAMERA = PinholeCamera(1280, 720, 914.164306640625, 914.164306640625, 627.49267578125,
                             394.47021484375, True)  # fmt: skip
# The template recipe's COLMAP commands and flags, plus the
# release's one deviation, single-threaded feature extraction (reproducible reruns).
REFERENCE_EXTRACT = [
    "feature_extractor", "--database_path", "database.db", "--image_path", "images",
    "--ImageReader.camera_model", "OPENCV", "--ImageReader.single_camera", "1",
    "--ImageReader.camera_params",
    "914.164306640625,914.164306640625,627.49267578125,394.47021484375,0,0,0,0",
    "--FeatureExtraction.use_gpu", "0", "--FeatureExtraction.num_threads", "1",
    "--SiftExtraction.max_num_features", "16384",
    "--SiftExtraction.estimate_affine_shape", "1", "--SiftExtraction.domain_size_pooling", "1",
]  # fmt: skip
REFERENCE_MATCH = [
    "exhaustive_matcher", "--database_path", "database.db", "--FeatureMatching.use_gpu", "1",
    "--FeatureMatching.guided_matching", "1", "--FeatureMatching.max_num_matches", "65536",
]  # fmt: skip
REFERENCE_TRIANGULATE = [
    "point_triangulator", "--database_path", "database.db", "--image_path", "images",
    "--input_path", "robot-seed", "--output_path", "robot-triangulated", "--refine_intrinsics", "0",
]  # fmt: skip
REFERENCE_BUNDLE_ADJUST = [
    "bundle_adjuster", "--input_path", "robot-triangulated", "--output_path", "sparse/0",
    "--BundleAdjustment.refine_focal_length", "0", "--BundleAdjustment.refine_principal_point", "0",
    "--BundleAdjustment.refine_extra_params", "0", "--BundleAdjustment.refine_rig_from_world", "1",
    "--BundleAdjustment.refine_sensor_from_rig", "0",
]  # fmt: skip
REFERENCE_MAP = [
    "mapper", "--database_path", "database.db", "--image_path", "images", "--output_path", "sparse",
]  # fmt: skip
# The fixture captures have three images; the release gate needs 20 registered.
RELAXED = QualityGate(min_registered_images=1)
MANY_NAMES = tuple(f"{i:06d}.png" for i in range(24))
FIXED_MAPPER = [
    "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_principal_point", "0",
    "--Mapper.ba_refine_extra_params", "0",
]  # fmt: skip


@pytest.fixture
def fake(tmp_path):
    return install_fake_colmap(tmp_path / "bin")


def test_default_flags_are_the_recipe_flags():
    assert load_any_recipe(NUREC_TEMPLATE).colmap == DEFAULT_FLAGS


def test_feature_extraction_is_single_threaded_by_default():
    # Threaded extraction reorders the database images and changes the reconstruction's scale
    # from run to run; the packaged recipe pins one thread so reruns are identical.
    actual_flags = load_any_recipe(NUREC_TEMPLATE).colmap["feature_extractor"]
    assert actual_flags["FeatureExtraction.num_threads"] == 1
    assert DEFAULT_FLAGS["feature_extractor"]["FeatureExtraction.num_threads"] == 1


def test_version_gate_refuses_3_11(tmp_path):
    old = install_fake_colmap(tmp_path / "old", version="3.11.1")
    with pytest.raises(ColmapError, match=r"3\.13.*3\.11\.1"):
        ColmapCli(str(old)).version()
    dev = install_fake_colmap(tmp_path / "dev", version="3.14.0.dev0")
    with pytest.raises(ColmapError, match="3.14.0.dev0"):
        ColmapCli(str(dev)).version()
    good = install_fake_colmap(tmp_path / "good", version="3.13.0")
    assert ColmapCli(str(good)).version() == "3.13.0"


def test_missing_binary_is_a_clear_error(tmp_path):
    with pytest.raises(ColmapError, match="COLMAP 3.13"):
        ColmapCli(str(tmp_path / "no-colmap")).version()


def test_runner_argv_matches_reference_flags(fake, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "images").mkdir()
    runner = ColmapCli(str(fake), load_any_recipe(NUREC_TEMPLATE).colmap)
    runner.extract(Path("database.db"), Path("images"), REFERENCE_CAMERA)
    runner.match(Path("database.db"))
    Path("robot-seed").mkdir()
    runner.triangulate(Path("database.db"), Path("images"), Path("robot-seed"),
                       Path("robot-triangulated"))  # fmt: skip
    runner.bundle_adjust(Path("robot-triangulated"), Path("sparse/0"))
    runner.map(Path("database.db"), Path("images"), Path("sparse"), fixed_intrinsics=True)
    runner.map(Path("database.db"), Path("images"), Path("other"))
    runner.to_text(Path("sparse/0"), Path("sparse-text"))
    runner.register(Path("database.db"), Path("sparse/0"), Path("registered"))
    calls = recorded_calls(fake)
    assert calls[0] == REFERENCE_EXTRACT
    assert calls[1] == REFERENCE_MATCH
    assert calls[2] == REFERENCE_TRIANGULATE
    assert calls[3] == REFERENCE_BUNDLE_ADJUST
    assert calls[4] == REFERENCE_MAP + FIXED_MAPPER
    assert calls[5][:-1] == [*REFERENCE_MAP[:-1]] and calls[5][-1] == "other"
    assert calls[6] == ["model_converter", "--input_path", "sparse/0",
                        "--output_path", "sparse-text", "--output_type", "TXT"]  # fmt: skip
    assert calls[7] == ["image_registrator", "--database_path", "database.db",
                        "--input_path", "sparse/0", "--output_path", "registered"]  # fmt: skip
    # Without a known camera the extractor estimates the intrinsics (COLMAP's default).
    runner.extract(Path("database2.db"), Path("images"), None)
    assert "--ImageReader.camera_params" not in recorded_calls(fake)[-1]
    # Every call's exact argv is recorded, the binary first.
    assert [list(call.argv[1:]) for call in runner.calls] == recorded_calls(fake)
    assert all(call.argv[0] == str(fake) for call in runner.calls)


def test_pinhole_camera_model_takes_four_params(fake, tmp_path):
    flags = {"feature_extractor": {"ImageReader.camera_model": "PINHOLE"}}
    (tmp_path / "images").mkdir()
    ColmapCli(str(fake), flags).extract(tmp_path / "db", tmp_path / "images", REFERENCE_CAMERA)
    argv = recorded_calls(fake)[-1]
    params = argv[argv.index("--ImageReader.camera_params") + 1]
    assert params == "914.164306640625,914.164306640625,627.49267578125,394.47021484375"
    flags = {"feature_extractor": {"ImageReader.camera_model": "SIMPLE_RADIAL"}}
    with pytest.raises(ValueError, match="SIMPLE_RADIAL"):
        ColmapCli(str(fake), flags).extract(tmp_path / "db", tmp_path / "images", REFERENCE_CAMERA)


def test_failed_call_raises_with_command_and_log_tail(fake, tmp_path):
    (fake.parent / "fail.txt").write_text("exhaustive_matcher")
    runner = ColmapCli(str(fake), log_dir=tmp_path / "logs")
    with pytest.raises(ColmapError) as raised:
        runner.match(tmp_path / "db")
    message = str(raised.value)
    assert "exhaustive_matcher --database_path" in message and "exit code 3" in message
    assert "E fake failure of exhaustive_matcher" in message
    assert "fake log line 29" in message and "fake log line 0\n" not in message
    log = (tmp_path / "logs" / "01-exhaustive_matcher.log").read_text()
    assert "fake log line 0" in log
    assert runner.calls[-1].returncode == 3


def _model(directory: Path, registered: int) -> None:
    directory.mkdir(parents=True)
    (directory / "images.bin").write_bytes(struct.pack("<Q", registered))


def test_select_largest_model_ties(tmp_path):
    sparse = tmp_path / "sparse"
    for name, registered in (("0", 3), ("2", 5), ("10", 5), ("1", 4)):
        _model(sparse / name, registered)
    assert select_largest_model(sparse) == sparse / "2"
    assert count_registered_images(sparse / "10") == 5
    _model(tmp_path / "zero" / "0", 0)
    with pytest.raises(ValueError, match="no registered images"):
        select_largest_model(tmp_path / "zero")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no sparse models"):
        select_largest_model(tmp_path / "empty")
    (sparse / "notes").mkdir()
    with pytest.raises(ValueError, match="notes"):
        select_largest_model(sparse)


def _frames(tmp_path: Path):
    bundle = tmp_path / "bundle"
    write_bundle(bundle, robot=False)
    image_set = load_image_set(bundle)
    return image_set, normalize_images(image_set, tmp_path / "frames")


def test_mode_b_metadata_is_not_metric(fake, tmp_path):
    image_set, frames = _frames(tmp_path)
    out = tmp_path / "reconstruction"
    runner = ColmapCli(str(fake), log_dir=out / "logs")
    flags = {"distortion": "zero", "robot_reference": False}
    inputs = {"bundle_sha256": "ab" * 32}
    metadata = reconstruct_sfm(frames, tmp_path / "frames", runner, out, capture_flags=flags,
                               inputs=inputs, gate=RELAXED)  # fmt: skip
    assert metadata.strategy == ColmapSfmSeed.name == "colmap_sfm"
    assert metadata.world_frame == "arbitrary" and metadata.length_unit == "arbitrary"
    assert metadata.scene_units_per_metre is None and metadata.scale_anchor is None
    assert metadata.is_metric is False and metadata.reference_path is False
    assert metadata.capture_flags == flags
    assert metadata.colmap.version == "3.13.0"
    assert metadata.colmap.selected_model == "sparse/1"
    commands = [argv[1] for argv in metadata.colmap.argv]
    assert commands == ["help", "feature_extractor", "exhaustive_matcher", "mapper",
                        "model_converter"]  # fmt: skip
    assert metadata.colmap.argv[3][-6:] == FIXED_MAPPER
    assert metadata.colmap.metrics["registered_images"] == 2
    assert metadata.colmap.metrics["total_images"] == 3
    # Per-image provenance, bound by name: the fake's database ids run opposite to file order.
    records = {record.source_name: record for record in metadata.images}
    assert [record.source_name for record in metadata.images] == list(NAMES)
    for frame in frames.frames:
        record = records[frame.source]
        assert record.name == frame.name and record.sha256 == frame.sha256
        assert record.seeded is False and record.seed_cam_from_world is None
    assert [records[n].colmap_image_id for n in NAMES] == [3, 2, 1]
    assert [records[n].registered for n in NAMES] == [True, True, False]
    assert records[NAMES[2]].excluded_reason == "not registered"
    assert metadata.camera.source == "capture session" and metadata.camera.refined is False
    assert metadata.camera.params == (80.0, 80.0, 47.5, 26.5, 0.0, 0.0, 0.0, 0.0)
    assert (out / "sparse-text" / "images.txt").exists()
    # reconstruction.json round-trips.
    assert ReconstructionMetadata.read(out / "reconstruction.json") == metadata
    record = json.loads((out / "reconstruction.json").read_text())
    assert record["schema_version"] == 1 and record["scene_units_per_metre"] is None
    assert record["quaternion_order"] == "wxyz"
    assert record["colmap"]["warnings"] == ["COLMAP_LOW_REGISTRATION"]


def test_metadata_refuses_an_arbitrary_unit_with_a_scale(fake, tmp_path):
    image_set, frames = _frames(tmp_path)
    metadata = reconstruct_sfm(frames, tmp_path / "frames", ColmapCli(str(fake)),
                               tmp_path / "out", capture_flags={}, inputs={},
                               gate=RELAXED)  # fmt: skip
    anchor = ScaleAnchor("sim3_seed_to_refined_centres", 1.0847, np.eye(3).tolist(),
                         [0.0, 0.0, 0.0], 0.0022, 0.0034, 2.03, 3.13, 35)  # fmt: skip
    record = metadata.to_record()
    with pytest.raises(ValueError, match="arbitrary"):
        ReconstructionMetadata.from_record({**record, "scale_anchor": anchor.to_record(),
                                            "scene_units_per_metre": 1.0847})  # fmt: skip
    with pytest.raises(ValueError, match="scene_units_per_metre"):
        ReconstructionMetadata.from_record({**record, "length_unit": "colmap",
                                            "scale_anchor": anchor.to_record(),
                                            "scene_units_per_metre": 2.0})  # fmt: skip
    anchored = ReconstructionMetadata.from_record(
        {**record, "length_unit": "colmap", "world_frame": "robot_base",
         "strategy": "robot_handeye", "scale_anchor": anchor.to_record(),
         "scene_units_per_metre": 1.0847})  # fmt: skip
    assert anchored.is_metric and anchored.scale_anchor == anchor
    with pytest.raises(ValueError, match="schema_version"):
        ReconstructionMetadata.from_record({**record, "schema_version": 2})
    with pytest.raises(ValueError, match="unknown keys"):
        ReconstructionMetadata.from_record({**record, "extra": 1})


# --- doper prepare RECIPE --stage reconstruct ------------------------------------------------


def _recipe(tmp_path: Path, seed_mode: str) -> str:
    data = nurec_recipe_data()
    data["seed_mode"] = seed_mode
    path = tmp_path / f"nurec-{seed_mode}.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


def _reconstruct(recipe: str, root: Path, *extra: str) -> list[str]:
    return ["prepare", recipe, "--root", str(root), "--stage", "reconstruct", *extra]


@pytest.mark.parametrize("seed_mode", ["robot_handeye", "colmap_sfm", "doper_pnp"])
def test_prepare_reconstruct_dry_run_for_every_mode(tmp_path, capsys, seed_mode):
    root = tmp_path / "root"
    extra = ["--checkpoint", str(tmp_path / "model.pth")] if seed_mode == "doper_pnp" else []
    assert cli.main(_reconstruct(_recipe(tmp_path, seed_mode), root, "--dry-run", *extra)) == 0
    out = capsys.readouterr().out
    assert "reconstruct" in out and "reconstruction.json" in out and seed_mode in out
    assert not root.exists()


def test_prepare_reconstruct_needs_the_capture_stage(tmp_path, capsys):
    assert cli.main(_reconstruct(_recipe(tmp_path, "colmap_sfm"), tmp_path / "root")) == 1
    assert "--stage capture" in capsys.readouterr().out


def test_prepare_reconstruct_mode_b(fake, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("DOPER_COLMAP", str(fake))
    bundle, root = tmp_path / "bundle", tmp_path / "root"
    write_bundle(bundle, names=MANY_NAMES, robot=False)
    recipe = _recipe(tmp_path, "colmap_sfm")
    capture = ["prepare", recipe, "--root", str(root), "--stage", "capture", "--bundle",
               str(bundle)]  # fmt: skip
    assert cli.main(capture) == 0
    assert cli.main(_reconstruct(recipe, root)) == 0
    metadata = ReconstructionMetadata.read(root / "reconstruction" / "reconstruction.json")
    ready = json.loads((root / "capture" / "READY.json").read_text())
    assert metadata.capture_flags == ready["flags"]
    assert metadata.inputs["bundle_sha256"] == ready["bundle_sha256"]
    assert [r.source_name for r in metadata.images] == list(MANY_NAMES)
    assert metadata.colmap.warnings == []
    calls = len(recorded_calls(fake))
    # A rerun keeps the verified reconstruction; --force rebuilds it.
    assert cli.main(_reconstruct(recipe, root)) == 0
    assert "already reconstructed" in capsys.readouterr().out
    assert len(recorded_calls(fake)) == calls
    assert cli.main(_reconstruct(recipe, root, "--force")) == 0
    assert len(recorded_calls(fake)) == 2 * calls
    # A corrupt reconstruction.json is refused, and --force always recovers from it.
    (root / "reconstruction" / "reconstruction.json").write_text("{not json")
    assert cli.main(_reconstruct(recipe, root)) == 1
    assert cli.main(_reconstruct(recipe, root, "--force")) == 0
    ReconstructionMetadata.read(root / "reconstruction" / "reconstruction.json")


def test_prepare_reconstruct_reports_a_failed_colmap_call(fake, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("DOPER_COLMAP", str(fake))
    bundle, root = tmp_path / "bundle", tmp_path / "root"
    write_bundle(bundle, names=MANY_NAMES, robot=False)
    recipe = _recipe(tmp_path, "colmap_sfm")
    capture = ["prepare", recipe, "--root", str(root), "--stage", "capture", "--bundle",
               str(bundle)]  # fmt: skip
    assert cli.main(capture) == 0
    (fake.parent / "fail.txt").write_text("mapper")
    assert cli.main(_reconstruct(recipe, root)) == 1
    out = capsys.readouterr().out
    assert "mapper --database_path" in out and "E fake failure of mapper" in out
    assert not (root / "reconstruction" / "reconstruction.json").exists()
    # The unfinished run is rebuilt without --force once COLMAP works.
    (fake.parent / "fail.txt").unlink()
    assert cli.main(_reconstruct(recipe, root)) == 0
