# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HOT3D validation-manifest and test-manifest staging from the real-image clips."""

import json
import tarfile

import pytest
from bop_recipe_data import STREAMS, TEST_FRAMES, make_hot3d_download, write_clip

from doper.bop import hot3d_staging as staging
from doper.bop import sources


@pytest.fixture
def downloaded(tmp_path):
    root = tmp_path / "root"
    profile = sources.load_profile(make_hot3d_download(root, tmp_path / "profile"))
    sources.fetch(profile, root, ["val", "test"], allow_transductive=True)
    sources.write_ready(root, profile, ["val", "test"])
    return root, profile


def _stage_validation(root, profile, shards=2):
    for shard in range(shards):
        staging.stage_validation_shard(root, shard, profile, shards=shards)
    return staging.merge_validation(root, profile, shards=shards)


def test_validation_manifest_holds_native_gt_of_the_panel(downloaded):
    root, profile = downloaded
    path = _stage_validation(root, profile)
    manifest = json.loads(path.read_text())
    assert [(f["scene_id"], f["frame"]) for f in manifest["frames"]] == [
        (1856, 29),
        (1856, 59),
        (1900, 29),
    ]
    frame = manifest["frames"][0]
    assert frame["objects"] == [1, 2] and frame["stream"] == STREAMS["Aria"]
    assert (path.parent / frame["path"]).is_file() and not frame["path"].startswith("/")
    assert frame["GT"][0]["rectified_amodal_bbox"] is not None
    assert frame["GT"][0]["cam_t_m2c"][2] == pytest.approx(400.0)
    assert "val_root" not in manifest
    # Re-running reuses the receipted shards and leaves the manifest unchanged.
    assert _stage_validation(root, profile).read_bytes() == path.read_bytes()


def test_validation_coverage_gate_refuses_an_uncovered_object(tmp_path):
    root = tmp_path / "root"
    path = make_hot3d_download(root, tmp_path / "profile")
    data = json.loads(path.read_text())
    data["meta"]["objects"] = [1, 2, 3]
    path.write_text(json.dumps(data))
    profile = sources.load_profile(path)
    sources.write_ready(root, profile, ["val"])
    for shard in range(2):
        staging.stage_validation_shard(root, shard, profile, shards=2)
    with pytest.raises(ValueError, match="coverage gate"):
        staging.merge_validation(root, profile, shards=2)


def test_validation_shard_identity_is_frozen(downloaded):
    root, profile = downloaded
    staging.stage_validation_shard(root, 0, profile, shards=2)
    receipt = root / staging.VALIDATION_DIR / "shard_00" / "data" / "READY.json"
    data = json.loads(receipt.read_text())
    data["identity"]["camera_code_sha256"] = "0" * 64
    receipt.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="identity changed"):
        staging.stage_validation_shard(root, 0, profile, shards=2)


def test_test_manifest_holds_only_rgb_and_sanitized_calibration(downloaded):
    root, profile = downloaded
    for shard in range(2):
        staging.stage_test_shard(root, shard, profile, shards=2)
    path = staging.merge_test(root, profile, shards=2)
    manifest = json.loads(path.read_text())
    assert [f["im_id"] for f in manifest["frames"]] == list(TEST_FRAMES)
    assert all(set(f) == staging.TEST_FRAME_FIELDS for f in manifest["frames"])
    assert all((path.parent / f["native_path"]).is_file() for f in manifest["frames"])


def test_test_clip_without_a_scheduled_image_is_refused(downloaded, tmp_path):
    root, profile = downloaded
    archive = tmp_path / "short.tar"
    write_clip(archive, STREAMS["Aria"], TEST_FRAMES[:-1], annotated=False)
    (clip,) = staging.held_out_test_clips(profile)
    identity = staging.held_out_test_identity(profile)
    with pytest.raises(ValueError, match="RGB/calibration-only"):
        staging._extract_clip(archive, clip, tmp_path / "clip", identity, profile, "clip")


def test_test_staging_never_reads_annotation_members(downloaded):
    root, profile = downloaded
    clip = sources.source_dir(root) / "test_aria/clip-003365.tar"
    write_clip(clip, STREAMS["Aria"], TEST_FRAMES, annotated=True)
    with tarfile.open(clip) as tar:
        assert any(m.name.endswith("objects.json") for m in tar.getmembers())
        members = staging.native_members(tar, STREAMS["Aria"])
    assert all(set(entry) == {"camera", "image"} for entry in members.values())


def test_test_frames_refuse_leaked_fields(downloaded):
    root, profile = downloaded
    staging.stage_test_shard(root, 0, profile, shards=1)
    receipt = root / staging.TEST_DIR / "shards" / "00" / "READY.json"
    frame = json.loads(receipt.read_text())["frames"][0]
    staging.validate_test_frame(frame, profile)
    with pytest.raises(ValueError, match="foreign"):
        staging.validate_test_frame({**frame, "GT": []}, profile)
    calibration = {**frame["native_calibration"], "T_world_from_camera": {}}
    with pytest.raises(ValueError, match="leaked"):
        staging.validate_test_frame({**frame, "native_calibration": calibration}, profile)


def test_reference_device_follows_the_bop_scene_mapping():
    assert staging.reference_device(1288) == "Quest3"
    assert staging.reference_device(3365) == "Aria"
    with pytest.raises(ValueError):
        staging.reference_device(5000)


def test_packaged_test_index_maps_every_clip_to_a_pinned_archive():
    clips = staging.held_out_test_clips()
    assert len(clips) == 1028 and sum(len(c["frames"]) for c in clips) == 5140
