# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture bundle loading, robot reference, flags and normalization (``doper.nurec.capture``)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from nurec_fixtures import (
    CAMERA,
    HANDEYE_CAMERA,
    LINEAGE,
    NAMES,
    NUREC_TEMPLATE,
    OWN_PROFILE,
    own_nurec_recipe,
    write_bundle,
    write_jsonl,
)

from doper import cli
from doper.nurec.capture import (
    PinholeCamera,
    capture_flags,
    has_robot_reference,
    load_frame_map,
    load_image_set,
    load_robot_reference,
    normalize_images,
    reference_camera_poses,
)


def test_image_set_loads_and_verifies_sha(tmp_path):
    written = write_bundle(tmp_path, robot=False)
    image_set = load_image_set(tmp_path)
    assert [image.name for image in image_set.images] == list(NAMES)
    assert {image.name: image.sha256 for image in image_set.images} == written["sha256"]
    assert {image.lineage for image in image_set.images} == {LINEAGE}
    assert image_set.camera == PinholeCamera(**CAMERA)
    assert image_set.exclude == ()
    np.testing.assert_allclose(image_set.camera.matrix, [[80, 0, 47.5], [0, 80, 26.5], [0, 0, 1]])


def test_bad_sha_refused(tmp_path):
    write_bundle(tmp_path, robot=False)
    (tmp_path / "images" / NAMES[1]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        load_image_set(tmp_path)
    assert len(load_image_set(tmp_path, verify=False).images) == len(NAMES)


@pytest.mark.parametrize(
    "camera",
    [{**CAMERA, "rectified": False}, {**CAMERA, "distortion": [0.1, 0.0, 0.0, 0.0]}],
)
def test_unrectified_camera_refused(tmp_path, camera):
    write_bundle(tmp_path, camera=camera, robot=False)
    with pytest.raises(ValueError, match="rectified|distortion"):
        load_image_set(tmp_path)


def test_zero_distortion_is_accepted(tmp_path):
    write_bundle(tmp_path, camera={**CAMERA, "distortion": [0.0, 0.0, 0.0, 0.0]}, robot=False)
    assert load_image_set(tmp_path).camera.rectified


def test_unknown_session_keys_refused(tmp_path):
    write_bundle(tmp_path, camera={**CAMERA, "serial": "123"}, robot=False)
    with pytest.raises(ValueError, match="serial"):
        load_image_set(tmp_path)


def test_robot_reference_optional_for_image_set(tmp_path):
    write_bundle(tmp_path, robot=False)
    image_set = load_image_set(tmp_path)
    assert not has_robot_reference(tmp_path)
    with pytest.raises(FileNotFoundError, match="robot"):
        load_robot_reference(tmp_path, image_set)
    flags = capture_flags(image_set, None)
    assert flags["distortion"] == "zero"
    assert flags["handeye_camera_mismatch"] is None


@pytest.mark.parametrize("robot_key", ["T_base_mount", "T_base_link"])
def test_reference_pose_composition(tmp_path, robot_key):
    written = write_bundle(tmp_path, robot_key=robot_key)
    image_set = load_image_set(tmp_path)
    assert has_robot_reference(tmp_path)
    reference = load_robot_reference(tmp_path, image_set)
    poses = reference_camera_poses(reference)
    assert list(poses) == list(NAMES)
    for name, pose in poses.items():
        np.testing.assert_allclose(pose, written["poses"][name] @ written["T_mount_camera"])
    assert reference.validation["max_rotation_error_deg"] == 0.5


def test_handeye_camera_mismatch_is_recorded(tmp_path):
    write_bundle(tmp_path, handeye_camera=HANDEYE_CAMERA)
    image_set = load_image_set(tmp_path)
    reference = load_robot_reference(tmp_path, image_set)
    assert reference.handeye_camera == PinholeCamera(**HANDEYE_CAMERA)
    flags = capture_flags(image_set, reference)
    assert flags["handeye_camera_mismatch"] is True
    assert flags["distortion"] == "zero"
    assert flags["capture_camera"]["K"] == [[80.0, 0.0, 47.5], [0.0, 80.0, 26.5], [0, 0, 1]]
    assert flags["handeye_camera"]["K"] == [[40.0, 0.0, 31.5], [0.0, 40.0, 17.5], [0, 0, 1]]
    assert flags["handeye_camera"]["width"] == HANDEYE_CAMERA["width"]
    same = _same_camera(tmp_path)
    same_set = load_image_set(same)
    matching = capture_flags(same_set, load_robot_reference(same, same_set))
    assert matching["handeye_camera_mismatch"] is False


def _same_camera(tmp_path: Path) -> Path:
    root = tmp_path / "same"
    write_bundle(root, handeye_camera=CAMERA)
    return root


def test_handeye_without_camera_is_unknown(tmp_path):
    write_bundle(tmp_path, handeye_camera=None)
    image_set = load_image_set(tmp_path)
    reference = load_robot_reference(tmp_path, image_set)
    assert reference.handeye_camera is None
    assert capture_flags(image_set, reference)["handeye_camera_mismatch"] is None


def test_image_size_must_match_session(tmp_path):
    write_bundle(tmp_path, robot=False, image_size=(CAMERA["width"] + 2, CAMERA["height"]))
    with pytest.raises(ValueError, match="size"):
        load_image_set(tmp_path)


def test_robot_reference_must_cover_every_image(tmp_path):
    write_bundle(tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "robot.jsonl").read_text().splitlines()]
    image_set = load_image_set(tmp_path)
    write_jsonl(tmp_path / "robot.jsonl", rows[:-1])
    with pytest.raises(ValueError, match=NAMES[-1]):
        load_robot_reference(tmp_path, image_set)
    write_jsonl(tmp_path / "robot.jsonl", [*rows, {**rows[0], "name": "stranger.png"}])
    with pytest.raises(ValueError, match="stranger"):
        load_robot_reference(tmp_path, image_set)
    write_jsonl(tmp_path / "robot.jsonl", [*rows, rows[0]])
    with pytest.raises(ValueError, match="twice"):
        load_robot_reference(tmp_path, image_set)


def test_robot_pose_must_be_rigid(tmp_path):
    write_bundle(tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "robot.jsonl").read_text().splitlines()]
    rows[0]["T_base_mount"][0][0] *= 2.0
    write_jsonl(tmp_path / "robot.jsonl", rows)
    with pytest.raises(ValueError, match="rigid"):
        load_robot_reference(tmp_path, load_image_set(tmp_path))


def test_exclude_names_must_be_captures(tmp_path):
    write_bundle(tmp_path, robot=False, exclude=[NAMES[0]])
    assert load_image_set(tmp_path).exclude == (NAMES[0],)
    (tmp_path / "exclude.json").write_text(json.dumps(["nope.png"]))
    with pytest.raises(ValueError, match="nope"):
        load_image_set(tmp_path)


def test_normalize_binds_by_source_name(tmp_path):
    bundle = tmp_path / "bundle"
    written = write_bundle(bundle, robot=False)
    # Reverse the file times: frames follow the image list, not the file times.
    for age, name in enumerate(NAMES):
        os.utime(bundle / "images" / name, (1000 + 10 * age, 1000 - 10 * age))
    image_set = load_image_set(bundle)
    frames = normalize_images(image_set, tmp_path / "frames", max_long_edge=48)
    assert [frame.source for frame in frames.frames] == list(NAMES)
    assert [frame.name for frame in frames.frames] == [f"frame-{i:06d}.png" for i in range(3)]
    assert frames.camera == PinholeCamera(48, 27, 40.0, 40.0, 23.75, 13.25, True)
    for name in NAMES:
        frame = frames.by_source(name)
        assert frame.source_sha256 == written["sha256"][name]
        source = cv2.imread(str(bundle / "images" / name))
        expected = cv2.resize(source, (48, 27), interpolation=cv2.INTER_AREA)
        actual = cv2.imread(str(tmp_path / "frames" / frame.name))
        np.testing.assert_array_equal(actual, expected)
        digest = hashlib.sha256((tmp_path / "frames" / frame.name).read_bytes()).hexdigest()
        assert frame.sha256 == digest
        assert frame.lineage == LINEAGE
    assert load_frame_map(tmp_path / "frames") == frames
    recorded = json.loads((tmp_path / "frames" / "frames.json").read_text())
    assert recorded["interpolation"] == "INTER_AREA"
    assert recorded["frames"][0]["source"] == NAMES[0]
    with pytest.raises(FileExistsError):
        normalize_images(image_set, tmp_path / "frames", max_long_edge=48)


def test_normalize_checks_an_unverified_image_set(tmp_path):
    write_bundle(tmp_path / "wide", robot=False, image_size=(CAMERA["width"] + 2, CAMERA["height"]))
    with pytest.raises(ValueError, match="differs from the session camera"):
        normalize_images(load_image_set(tmp_path / "wide", verify=False), tmp_path / "a")
    write_bundle(tmp_path / "bundle", robot=False)
    image_set = load_image_set(tmp_path / "bundle", verify=False)
    (tmp_path / "bundle" / "images" / NAMES[1]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        normalize_images(image_set, tmp_path / "b")
    (tmp_path / "bundle" / "images" / NAMES[1]).unlink()
    with pytest.raises(FileNotFoundError, match="is missing"):
        normalize_images(image_set, tmp_path / "c")
    assert not any((tmp_path / name).exists() for name in "abc")


def test_normalize_never_upscales(tmp_path):
    write_bundle(tmp_path / "bundle", robot=False)
    frames = normalize_images(load_image_set(tmp_path / "bundle"), tmp_path / "frames")
    assert frames.camera == PinholeCamera(**CAMERA)


def test_load_frame_map_detects_tampering(tmp_path):
    write_bundle(tmp_path / "bundle", robot=False)
    normalize_images(load_image_set(tmp_path / "bundle"), tmp_path / "frames", max_long_edge=48)
    (tmp_path / "frames" / "frame-000001.png").write_bytes(b"x")
    with pytest.raises(ValueError, match="frame-000001.png"):
        load_frame_map(tmp_path / "frames")


def _prepare(bundle: Path, root: Path, *extra: str, recipe: str | None = None) -> list[str]:
    """``prepare --stage capture`` under ``recipe`` (default: the template, ``OWN_PROFILE``)."""
    recipe = recipe or own_nurec_recipe(root.parent)
    return ["prepare", recipe, "--stage", "capture", "--bundle", str(bundle),
            "--root", str(root), *extra]  # fmt: skip


def test_prepare_capture_dry_run_touches_nothing(tmp_path, capsys):
    assert cli.main([*_prepare(tmp_path / "missing", tmp_path / "root"), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "verify and normalize the capture bundle" in out
    assert "READY.json" in out
    assert not (tmp_path / "root").exists()


def test_prepare_capture_writes_receipt(tmp_path, capsys):
    bundle, root = tmp_path / "bundle", tmp_path / "root"
    write_bundle(bundle, handeye_camera=HANDEYE_CAMERA)
    assert cli.main(_prepare(bundle, root)) == 0
    ready = json.loads((root / "capture" / "READY.json").read_text())
    assert ready["image_count"] == len(NAMES)
    assert ready["robot_reference"] is True
    assert ready["flags"]["handeye_camera_mismatch"] is True
    assert ready["lineages"] == [LINEAGE]
    assert ready["pinned"] is False
    frames = load_frame_map(root / "capture" / "frames")
    assert [frame.source for frame in frames.frames] == list(NAMES)
    # A second run verifies and keeps the receipt; a changed bundle needs --force.
    assert cli.main(_prepare(bundle, root)) == 0
    assert "already prepared" in capsys.readouterr().out
    (bundle / "exclude.json").write_text(json.dumps([NAMES[0]]))
    assert cli.main(_prepare(bundle, root)) == 1
    assert "--force" in capsys.readouterr().out
    assert cli.main(_prepare(bundle, root, "--force")) == 0
    ready = json.loads((root / "capture" / "READY.json").read_text())
    assert ready["exclude"] == [NAMES[0]]
    # The images are checked on a rerun too, though the receipt matches.
    (bundle / "images" / NAMES[1]).write_bytes(b"changed")
    assert cli.main(_prepare(bundle, root)) == 1
    assert "SHA-256 differs" in capsys.readouterr().out


def test_prepare_capture_without_robot_reference(tmp_path):
    bundle, root = tmp_path / "bundle", tmp_path / "root"
    write_bundle(bundle, robot=False)
    assert cli.main(_prepare(bundle, root)) == 0
    ready = json.loads((root / "capture" / "READY.json").read_text())
    assert ready["robot_reference"] is False
    assert ready["flags"]["handeye_camera_mismatch"] is None


@pytest.mark.parametrize(
    ("extra", "flag"),
    [
        (["--obj-id", "1"], "--obj-id"),
        (["--accept-license"], "--accept-license"),
        (["--splits", "train"], "--splits"),
    ],
)
def test_prepare_capture_refuses_other_kinds_flags(tmp_path, capsys, extra, flag):
    assert cli.main([*_prepare(tmp_path, tmp_path), *extra, "--dry-run"]) == 1
    assert flag in capsys.readouterr().out


def test_prepare_capture_needs_bundle(tmp_path, capsys):
    argv = [
        "prepare",
        str(NUREC_TEMPLATE),
        "--stage",
        "capture",
        "--root",
        str(tmp_path),
        "--dry-run",
    ]
    assert cli.main(argv) == 1
    assert "--bundle" in capsys.readouterr().out


def test_bundle_flag_refused_for_bop_recipes(tmp_path, capsys):
    argv = ["prepare", "tless", "--root", str(tmp_path), "--bundle", str(tmp_path), "--dry-run"]
    assert cli.main(argv) == 1
    assert "--bundle" in capsys.readouterr().out


def test_verify_bundle_pin_refuses_another_bundle(tmp_path):
    from doper.nurec import capture

    assert capture.profile_bundle_pin(OWN_PROFILE) is None
    bundle = tmp_path / "bundle"
    write_bundle(bundle)
    with pytest.raises(ValueError, match="is not the pinned-capture bundle"):
        capture.verify_bundle_pin(bundle, "f" * 64, "pinned-capture")
    identity = capture.bundle_sha256(bundle)
    assert capture.verify_bundle_pin(bundle, None, OWN_PROFILE) == identity
    assert capture.verify_bundle_pin(bundle, identity, "pinned-capture") == identity
