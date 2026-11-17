# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T-LESS Kinect/Canon TRAIN fetch and the three-sensor supervision roots.

The golden file ``golden/source_tless_multisensor.json`` holds what the research
``prepare_tless_multisensor.prepare`` (``--supervision gt-keypoints --mask-policy projected``)
and its ``projected_cad_mask`` produce on the fixture of ``tless_multisensor_data``, with paths
normalized and OpenCV-encoded file bytes replaced by decoded pixels.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import tless_multisensor_data as fixture

from doper.bop import multisensor, review, sources, tless_sensors

GOLDEN = json.loads(
    (Path(__file__).parent / "golden" / "source_tless_multisensor.json").read_text()
)


def _prepare(
    root: Path, paths: dict[str, Path], **kwargs: object
) -> multisensor.MultisensorSummary:
    return multisensor.prepare(
        root,
        fixture.OBJ,
        keypoints=paths["keypoints"],
        cad_model=paths["mesh"],
        cad_audit=paths["audit"],
        reserved=multisensor.reserved_population(paths["test"]),
        expected_frames=len(fixture.FRAMES),
        **kwargs,
    )


@pytest.fixture
def dataset(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    root = tmp_path / "root"
    return root, fixture.make_dataset(root)


def test_prepare_matches_source(dataset):
    root, paths = dataset
    summary = _prepare(root, paths)
    assert summary.root == multisensor.output_dir(root, fixture.OBJ)
    actual = fixture.summarize(summary.root, root)
    for sensor in fixture.SENSORS:
        first = actual["sensors"][sensor]["annotations"]["annotations"][0]
        expected = GOLDEN["first_annotation"][sensor]
        assert {**first, "keypoints": first["keypoints"][:6]} == expected
        assert actual["sensors"][sensor]["records"][0] == GOLDEN["records"][sensor]
        assert actual["sensors"][sensor]["ready"] == GOLDEN["ready"][sensor]
    assert actual["review_overlays"] == GOLDEN["review_overlays"]
    assert fixture.summary_sha256(actual) == GOLDEN["summary_sha256"]
    assert len(summary.overlays) == 9
    assert summary.frames == {sensor: 4 for sensor in fixture.SENSORS}


def test_prepared_root_is_relocatable(dataset, tmp_path):
    root, paths = dataset
    out = _prepare(root, paths).root
    for link in out.rglob("*"):
        if link.is_symlink():
            assert not os.path.isabs(os.readlink(link))
    for path in out.rglob("*.json"):
        assert str(root) not in path.read_text(), path
    moved = tmp_path / "moved"
    shutil.move(root, moved)
    multisensor.verify_prepared(multisensor.output_dir(moved, fixture.OBJ))


def test_projected_mask_matches_source():
    vertices, faces = fixture.ring_mesh()
    rotation, translation = fixture.RING_POSE
    mask = multisensor.projected_cad_mask(
        multisensor.CadMesh(vertices, faces), fixture.camera("canon"), rotation, translation,
        (72, 96),
    )  # fmt: skip
    assert mask.dtype == np.uint8 and mask.shape == (72, 96)
    assert hashlib.sha256(mask.tobytes()).hexdigest() == GOLDEN["ring_mask"]["pixels_sha256"]
    assert int((mask > 0).sum()) == GOLDEN["ring_mask"]["foreground"]
    assert mask[36, 48] == 0  # the hole stays open


def test_prepare_refuses_reserved_test_image(dataset):
    root, paths = dataset
    leaked = paths["primesense"] / "rgb" / "000002.png"
    shutil.copyfile(leaked, paths["test"] / "000001" / "rgb" / "000001.png")
    with pytest.raises(ValueError, match=r"reserved test_primesense.*primesense frame 2"):
        _prepare(root, paths)
    assert not multisensor.output_dir(root, fixture.OBJ).exists()


def test_reserved_population_must_be_flagged(dataset):
    root, paths = dataset
    reserved = {**multisensor.reserved_population(paths["test"]), "adaptation_allowed": True}
    with pytest.raises(ValueError, match="held-out"):
        multisensor.prepare(
            root, 1, keypoints=paths["keypoints"], cad_model=paths["mesh"],
            cad_audit=paths["audit"], reserved=reserved, expected_frames=4,
        )  # fmt: skip


def test_prepare_outputs_are_new_only(dataset):
    root, paths = dataset
    _prepare(root, paths)
    with pytest.raises(FileExistsError):
        _prepare(root, paths)


def test_prepare_refuses_skewed_camera(dataset):
    root, paths = dataset
    info = paths["kinect"] / "info.yml"
    info.write_text(info.read_text().replace("- 0.0\n", "- 0.5\n", 1))
    with pytest.raises(ValueError, match="skew"):
        _prepare(root, paths)


def test_prepare_refuses_rotated_exif(dataset):
    from PIL import Image

    root, paths = dataset
    path = paths["canon"] / "rgb" / "0001.jpg"
    with Image.open(path) as image:
        exif = image.getexif()
        exif[274] = 6
        image.save(path, exif=exif)
    with pytest.raises(ValueError, match="EXIF"):
        _prepare(root, paths)


def test_prepare_refuses_one_camera_relabeled(dataset):
    root, paths = dataset
    for frame in fixture.FRAMES:
        shutil.copyfile(
            paths["kinect"] / "rgb" / f"{frame:04d}.png",
            paths["canon"] / "rgb" / f"{frame:04d}.jpg",
        )
    info = paths["canon"] / "info.yml"
    info.write_text((paths["kinect"] / "info.yml").read_text())
    with pytest.raises(ValueError, match="byte-identical"):
        _prepare(root, paths)


def test_prepare_requires_the_audited_cad(dataset):
    root, paths = dataset
    paths["mesh"].write_text(paths["mesh"].read_text() + "\n")
    with pytest.raises(ValueError, match="audited"):
        _prepare(root, paths)


def test_prepare_requires_full_frame_count(dataset):
    root, paths = dataset
    with pytest.raises(ValueError, match="1296"):
        multisensor.prepare(
            root, 1, keypoints=paths["keypoints"], cad_model=paths["mesh"],
            cad_audit=paths["audit"], reserved=multisensor.reserved_population(paths["test"]),
        )  # fmt: skip


def test_review_binds_ready_and_all_overlays(dataset):
    root, paths = dataset
    out = _prepare(root, paths).root
    with pytest.raises(review.ReviewError):
        multisensor.verify(out, allow_partial=True)
    multisensor.approve(out, "reviewer")
    with pytest.raises(ValueError, match="partial"):
        multisensor.verify(out)
    ready = multisensor.verify(out, allow_partial=True)
    assert ready["object"] == 1 and ready["partial_smoke_only"] is True
    reviewed = review.read(out)["reviewed_overlays"]
    assert sorted(reviewed) == sorted(ready["review_overlays"]) and len(reviewed) == 9
    overlay = out / "kinect" / "review" / "000002.jpg"
    overlay.write_bytes(overlay.read_bytes() + b"x")
    with pytest.raises(review.ReviewError, match="changed"):
        multisensor.verify(out, allow_partial=True)


def test_verify_detects_changed_labels(dataset):
    root, paths = dataset
    out = _prepare(root, paths).root
    multisensor.approve(out, "reviewer")
    mask = out / "canon" / "train" / "masks" / "canon_000001.png"
    cv2.imwrite(str(mask), np.zeros((72, 96), np.uint8))
    with pytest.raises(ValueError, match="mask changed"):
        multisensor.verify(out, allow_partial=True)


# --- Kinect/Canon fetch ---------------------------------------------------------------------


def _downloader(archive: Path) -> tless_sensors.HttpGet:
    def download(url: str, destination: Path) -> None:
        assert url.startswith("http://ptak.felk.cvut.cz/darwin/t-less/v2/t-less_v2_train_")
        shutil.copyfile(archive, destination)

    return download


def test_fetch_records_an_observed_checksum(tmp_path):
    archive = fixture.v2_archive(tmp_path / "in.zip", 2, 1296)
    receipt = tless_sensors.fetch(tmp_path / "root", 2, "kinect", downloader=_downloader(archive))
    assert receipt.checksum_origin == "observed"
    assert receipt.frames == 1296
    assert receipt.archive_sha256 == sources.sha256_file(archive)
    assert (receipt.directory / "gt.yml").is_file() and (receipt.directory / "rgb").is_dir()
    record = json.loads(receipt.receipt.read_text())
    assert record["checksum_origin"] == "observed"
    assert record["url"].endswith("t-less_v2_train_kinect_02.zip")
    assert str(tmp_path) not in receipt.receipt.read_text()
    again = tless_sensors.fetch(tmp_path / "root", 2, "kinect", downloader=_downloader(archive))
    assert again == receipt


def test_observed_checksum_mismatch_fails(tmp_path):
    root = tmp_path / "root"
    archive = fixture.v2_archive(tmp_path / "in.zip", 2, 1296)
    receipt = tless_sensors.fetch(root, 2, "canon", downloader=_downloader(archive))
    other = fixture.v2_archive(tmp_path / "other.zip", 2, 1296, extra={"02/rgb/9999.png": b"x"})
    shutil.copyfile(other, tless_sensors.archive_path(root, 2, "canon"))
    with pytest.raises(ValueError, match="observed"):
        tless_sensors.fetch(root, 2, "canon", downloader=_downloader(other))
    assert json.loads(receipt.receipt.read_text())["archive_sha256"] == receipt.archive_sha256


def test_pinned_checksum_mismatch_fails(tmp_path):
    archive = fixture.v2_archive(tmp_path / "in.zip", 1, 1296)
    with pytest.raises(ValueError, match="pinned"):
        tless_sensors.fetch(tmp_path / "root", 1, "kinect", downloader=_downloader(archive))
    assert not tless_sensors.archive_path(tmp_path / "root", 1, "kinect").exists()


def test_pinned_checksum_is_recorded_as_pinned(tmp_path):
    archive = fixture.v2_archive(tmp_path / "in.zip", 3, 1296)
    profile = sources.load_profile("tless")
    pins = profile.meta["sensors"]["sha256"]
    pins["canon"] = {**pins["canon"], "3": sources.sha256_file(archive)}
    receipt = tless_sensors.fetch(
        tmp_path / "root", 3, "canon", downloader=_downloader(archive), profile=profile
    )
    assert receipt.checksum_origin == "pinned"


def test_profile_pins_object_one_observed_shas():
    pins = sources.load_profile("tless").meta["sensors"]["sha256"]
    assert pins["kinect"]["1"].startswith("74105f09")
    assert pins["canon"]["1"].startswith("6d6476a2")


def _link_member(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name)
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    return info


@pytest.mark.parametrize(
    "member",
    [
        "/02/rgb/0000.png",
        "02/../escape.png",
        "02\\rgb\\5000.png",
        "03/rgb/0000.png",
        "02/mask/0000.png",
        "02/rgb/0000.txt",
        "02/notes.yml",
        "link",
    ],
)
def test_zip_member_safety(tmp_path, member):
    archive = fixture.v2_archive(tmp_path / "in.zip", 2, 1296)
    with zipfile.ZipFile(archive, "a") as handle:
        if member == "link":
            handle.writestr(_link_member("02/rgb/7000.png"), "../../etc/passwd")
        else:
            handle.writestr(member, b"x")
    with pytest.raises(ValueError, match="member"):
        tless_sensors.fetch(tmp_path / "root", 2, "kinect", downloader=_downloader(archive))
    assert not tless_sensors.object_dir(tmp_path / "root", 2, "kinect").exists()
    assert not tless_sensors.receipt_path(tmp_path / "root", 2, "kinect").exists()


def test_obj19_frame_count_648(tmp_path):
    assert tless_sensors.expected_frames(19) == tless_sensors.expected_frames(20) == 648
    assert tless_sensors.expected_frames(1) == 1296
    full = fixture.v2_archive(tmp_path / "full.zip", 19, 1296)
    with pytest.raises(ValueError, match="648"):
        tless_sensors.fetch(tmp_path / "a", 19, "kinect", downloader=_downloader(full))
    half = fixture.v2_archive(tmp_path / "half.zip", 19, 648)
    receipt = tless_sensors.fetch(tmp_path / "b", 19, "kinect", downloader=_downloader(half))
    assert receipt.frames == 648


def test_fetch_refuses_multi_instance_frames(tmp_path):
    archive = tmp_path / "in.zip"
    fixture.v2_archive(archive, 2, 1296)
    with zipfile.ZipFile(archive) as handle:
        members = {n: handle.read(n) for n in handle.namelist()}
    import yaml

    gt = yaml.safe_load(members["02/gt.yml"])
    gt[5] = gt[5] * 2
    members["02/gt.yml"] = yaml.safe_dump(gt).encode()
    with zipfile.ZipFile(archive, "w") as handle:
        for name, payload in members.items():
            handle.writestr(name, payload)
    with pytest.raises(ValueError, match="one instance"):
        tless_sensors.fetch(tmp_path / "root", 2, "kinect", downloader=_downloader(archive))


def test_read_metadata_reads_v2_and_bop(dataset):
    _, paths = dataset
    gt, path = tless_sensors.read_metadata(paths["kinect"], "scene_gt.json", "gt.yml")
    assert path.name == "gt.yml" and sorted(gt) == list(fixture.FRAMES)
    gt, path = tless_sensors.read_metadata(paths["primesense"], "scene_gt.json", "gt.yml")
    assert path.name == "scene_gt.json" and sorted(gt) == list(fixture.FRAMES)


# --- doper prepare --multisensor / doper review tless ----------------------------------------


def test_prepare_multisensor_dry_run_lists_the_steps(tmp_path, capsys):
    from doper import cli

    argv = ["prepare", "tless", "--root", str(tmp_path / "r"), "--multisensor", "--obj-id", "1"]
    assert cli.main([*argv, "19", "--dry-run"]) == 0
    out = capsys.readouterr().out
    for step in ("fetch and verify", "extract", "fetch kinect obj 1", "fetch canon obj 19",
                 "prepare multisensor obj 19"):  # fmt: skip
        assert step in out
    assert "t-less_v2_train_canon_19.zip" in out and "tless_train_primesense.zip" in out
    assert "nothing was read" in out and not (tmp_path / "r").exists()


def test_prepare_multisensor_needs_tless(tmp_path, capsys):
    from doper import cli

    argv = ["prepare", "hopev2", "--root", str(tmp_path), "--multisensor", "--dry-run"]
    assert cli.main(argv) == 1
    assert "no additional TRAIN sensors" in capsys.readouterr().out


def test_prepare_multisensor_step_verifies_a_prepared_root(dataset, capsys):
    from doper.cli import build_parser
    from doper.cli import prepare as prepare_cli

    root, paths = dataset
    args = build_parser().parse_args(
        ["prepare", "tless", "--root", str(root), "--multisensor", "--obj-id", "1"]
    )
    step = next(s for s in prepare_cli.plan(args) if s.name == "prepare multisensor obj 1")
    with pytest.raises(ValueError, match="1296"):
        step.action()
    _prepare(root, paths)
    step.action()
    assert "already prepared" in capsys.readouterr().out
    mask = multisensor.output_dir(root, 1) / "kinect" / "train" / "masks" / "kinect_000000.png"
    mask.write_bytes(mask.read_bytes() + b"x")
    with pytest.raises(ValueError, match="changed"):
        step.action()


def test_prepare_multisensor_step_needs_the_test_split(dataset):
    from doper.cli import build_parser
    from doper.cli import prepare as prepare_cli

    root, _ = dataset
    shutil.rmtree(root / "data" / "test_primesense")
    args = build_parser().parse_args(
        ["prepare", "tless", "--root", str(root), "--multisensor", "--obj-id", "1"]
    )
    step = next(s for s in prepare_cli.plan(args) if s.name == "prepare multisensor obj 1")
    with pytest.raises(ValueError, match="--parts test"):
        step.action()


def test_review_tless_writes_the_multisensor_review(dataset, tmp_path):
    from doper import cli

    root, paths = dataset
    out = _prepare(root, paths).root
    argv = ["review", "tless", "--root", str(root), "--runs", str(tmp_path / "runs")]
    assert cli.main([*argv, "--obj-id", "2", "--reviewer", "A. Reviewer"]) == 1
    assert cli.main([*argv, "--obj-id", "1", "--reviewer", "A. Reviewer", "--dry-run"]) == 0
    assert not (out / review.REVIEWED_NAME).exists()
    assert cli.main([*argv, "--obj-id", "1", "--reviewer", "A. Reviewer"]) == 0
    record = review.read(out)
    assert record["reviewer"] == "A. Reviewer" and len(record["reviewed_overlays"]) == 9
    assert record["ready"] == "READY.json"
    multisensor.verify(out, allow_partial=True)


def test_prepare_multisensor_refuses_force(tmp_path, capsys):
    from doper import cli

    argv = ["prepare", "tless", "--root", str(tmp_path), "--multisensor", "--obj-id", "1"]
    assert cli.main([*argv, "--force", "--dry-run"]) == 1
    out = capsys.readouterr().out
    assert "--force" in out and "multisensor/obj_XXXXXX" in out and "REVIEWED.json" in out


def test_population_sha256_equals_the_directory_population_hash(dataset):
    from doper.bop import population

    _, paths = dataset
    reserved = multisensor.reserved_population(paths["test"])
    expected = population.load_population(paths["test"])[1]
    assert multisensor.population_sha256(reserved) == expected


def test_failed_staging_of_an_observed_archive_says_to_delete_it(tmp_path):
    archive = fixture.v2_archive(tmp_path / "in.zip", 2, 1295)
    root = tmp_path / "root"
    with pytest.raises(ValueError, match="t-less_v2_train_kinect_02.zip has no receipt.*delete it"):
        tless_sensors.fetch(root, 2, "kinect", downloader=_downloader(archive))
    assert tless_sensors.archive_path(root, 2, "kinect").exists()
    assert not tless_sensors.receipt_path(root, 2, "kinect").exists()


def test_failed_staging_of_a_truncated_archive_says_to_delete_it(tmp_path):
    archive = fixture.v2_archive(tmp_path / "in.zip", 2, 1296)
    archive.write_bytes(archive.read_bytes()[:-500])
    with pytest.raises(ValueError, match="re-fetched"):
        tless_sensors.fetch(tmp_path / "root", 2, "kinect", downloader=_downloader(archive))
