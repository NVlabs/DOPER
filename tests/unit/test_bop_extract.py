# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extraction of verified BOP ZIP parts and the PBR coverage receipt."""

import json
import zipfile

import pytest
from bop_recipe_data import make_bop_download

from doper.bop import extract, sources


@pytest.fixture
def downloaded(tmp_path):
    root = tmp_path / "root"
    profile = sources.load_profile(make_bop_download(root, tmp_path / "profile"))
    sources.write_ready(root, profile, ["models", "pbr"])
    return root, profile


def test_zip_archives_group_split_volumes():
    profile = sources.load_profile("hopev2")
    assert extract.zip_archives(profile, "pbr") == [
        ["hope_train_pbr.z01", "hope_train_pbr.z02", "hope_train_pbr.zip"]
    ]
    assert extract.zip_archives(sources.load_profile("hot3d"), "val") == []


def test_extract_and_receipt_coverage(downloaded):
    root, profile = downloaded
    data = extract.extract_parts(root, profile, ["models", "pbr"])
    assert (data / "models" / "obj_000001.ply").is_file()
    receipt = extract.write_pbr_ready(root, profile)
    assert receipt["coverage"]["frames"] == 9
    assert receipt["coverage"]["object_frame_counts"] == {"1": 9, "2": 9}
    marker = root / extract.RECEIPT_DIR / "fixture_train_pbr.zip.json"
    assert json.loads(marker.read_text())["crc_checked"]
    (data / "models" / "obj_000001.ply").unlink()
    extract.extract_parts(root, profile, ["models", "pbr"])  # receipted: not re-extracted
    assert not (data / "models" / "obj_000001.ply").exists()


def test_extract_requires_the_ready_receipt(tmp_path, downloaded):
    root, profile = downloaded
    with pytest.raises(sources.SourceNotReadyError):
        extract.extract_parts(root, profile, ["val"])


def test_extract_refuses_unsafe_members(downloaded):
    root, profile = downloaded
    archive = sources.source_dir(root) / "fixture_models.zip"
    with zipfile.ZipFile(archive, "a") as handle:
        handle.writestr("../escape.txt", b"x")
    profile.files["fixture_models.zip"] = sources.sha256_file(archive)
    with pytest.raises(ValueError, match="unsafe archive path"):
        extract._extract_archive(root, profile, ["fixture_models.zip"])
    assert not (root / "escape.txt").exists()


def _nest(archive, prefix, extra=None):
    with zipfile.ZipFile(archive) as handle:
        members = {name: handle.read(name) for name in handle.namelist() if not name.endswith("/")}
    with zipfile.ZipFile(archive, "w") as handle:
        for name, payload in members.items():
            handle.writestr(f"{prefix}/{name}", payload)
        for name, payload in (extra or {}).items():
            handle.writestr(name, payload)


def test_extract_strips_a_nesting_directory(downloaded):
    """T-LESS ``tless_base.zip`` nests its files under ``tless/`` (profile ``archive_prefix``)."""
    root, profile = downloaded
    archive = sources.source_dir(root) / "fixture_models.zip"
    _nest(archive, "tless")
    profile.files["fixture_models.zip"] = sources.sha256_file(archive)
    profile.meta["archive_prefix"] = {"fixture_models.zip": "tless"}
    assert extract._extract_archive(root, profile, ["fixture_models.zip"]) > 0
    assert (extract.data_dir(root) / "models" / "obj_000001.ply").is_file()
    assert not (extract.data_dir(root) / "tless").exists()
    marker = json.loads((root / extract.RECEIPT_DIR / "fixture_models.zip.json").read_text())
    assert marker["identity"]["strip_prefix"] == "tless"


def test_extract_refuses_a_member_outside_the_nesting_directory(downloaded):
    root, profile = downloaded
    archive = sources.source_dir(root) / "fixture_models.zip"
    _nest(archive, "tless", extra={"other/readme.txt": b"x"})
    profile.files["fixture_models.zip"] = sources.sha256_file(archive)
    profile.meta["archive_prefix"] = {"fixture_models.zip": "tless"}
    with pytest.raises(ValueError, match="not under tless/"):
        extract._extract_archive(root, profile, ["fixture_models.zip"])


def test_tless_base_archive_nests_under_tless():
    profile = sources.load_profile("tless")
    assert extract.archive_prefix(profile, "tless_base.zip") == "tless"
    assert extract.archive_prefix(profile, "tless_models.zip") is None


def test_pbr_receipt_refuses_unacknowledged_missing_objects(downloaded):
    root, profile = downloaded
    extract.extract_parts(root, profile, ["models", "pbr"])
    profile.meta["objects"] = [1, 2, 3]
    with pytest.raises(ValueError, match=r"objects \[3\]"):
        extract.write_pbr_ready(root, profile)
    receipt = extract.write_pbr_ready(root, profile, allowed_missing=[3])
    assert receipt["coverage"]["missing_objects"] == [3]


def test_pbr_receipt_checks_the_pinned_frame_count(downloaded):
    root, profile = downloaded
    extract.extract_parts(root, profile, ["models", "pbr"])
    profile.meta["expected_frames"] = {"train_pbr": 10}
    with pytest.raises(ValueError, match="pins 10"):
        extract.write_pbr_ready(root, profile)
