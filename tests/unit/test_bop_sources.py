# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for pinned BOP dataset sources, receipts and defect policies."""

import copy
import hashlib
import json
import os
import re
from pathlib import Path

import pytest

from doper.bop import sources
from doper.bop.sources import DatasetProfile

PACKAGED = ("handal", "hopev2", "hot3d", "tless")
REVISION = "a" * 40


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


PAYLOADS = {"models.zip": b"models", "pbr.z01": b"pbr-one", "pbr.zip": b"pbr-two", "test.zip": b"t"}


def make_profile(exclusions: list[str] | None = None) -> DatasetProfile:
    return DatasetProfile(
        name="fixture",
        hf_repo="bop-benchmark/fixture",
        revision=REVISION,
        files={name: _digest(data) for name, data in PAYLOADS.items()},
        parts={
            "models": ["models.zip"],
            "pbr": ["pbr.z01", "pbr.zip"],
            "test": ["test.zip"],
        },
        exclusions=exclusions or [],
        meta={"bytes": {name: len(data) for name, data in PAYLOADS.items()}},
    )


class FakeDownloader:
    def __init__(self, corrupt: str | None = None) -> None:
        self.calls: list[str] = []
        self.corrupt = corrupt

    def __call__(self, hf_repo: str, filename: str, revision: str, destination: Path) -> Path:
        assert (hf_repo, revision) == ("bop-benchmark/fixture", REVISION)
        self.calls.append(filename)
        data = PAYLOADS[filename]
        if filename == self.corrupt:
            data = bytes(len(data))
        target = destination / filename
        target.write_bytes(data)
        return target


def test_fetch_rejects_sha_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="SHA-256 mismatch: pbr.zip"):
        sources.fetch(make_profile(), tmp_path, ["pbr"], downloader=FakeDownloader("pbr.zip"))
    assert not (tmp_path / "source" / "pbr.zip").exists()


def test_fetch_verify_and_resume(tmp_path: Path) -> None:
    profile, downloader = make_profile(), FakeDownloader()
    directory = sources.fetch(profile, tmp_path, ["models", "pbr"], downloader=downloader)
    assert directory == tmp_path / "source"
    assert sources.verify(tmp_path, profile, ["models", "pbr"]) == []
    sources.fetch(profile, tmp_path, ["models", "pbr"], downloader=downloader)
    assert downloader.calls == ["models.zip", "pbr.z01", "pbr.zip"]
    (directory / "pbr.zip").write_bytes(b"pbr-twX")
    assert "SHA-256 mismatch: pbr.zip" in sources.verify(tmp_path, profile, ["pbr"])[0]
    (directory / "models.zip").unlink()
    assert sources.verify(tmp_path, profile, ["models"]) == ["missing: models.zip"]


def test_prepare_refuses_root_without_ready_receipt(tmp_path: Path) -> None:
    with pytest.raises(sources.SourceNotReadyError, match="READY.json"):
        sources.require_ready(tmp_path, ["pbr"])
    profile = make_profile()
    sources.fetch(profile, tmp_path, ["models"], downloader=FakeDownloader())
    sources.write_ready(tmp_path, profile, ["models"])
    assert sources.require_ready(tmp_path, ["models"])["revision"] == REVISION
    with pytest.raises(sources.SourceNotReadyError, match="part 'pbr'"):
        sources.require_ready(tmp_path, ["pbr"])
    (tmp_path / "source" / "models.zip").write_bytes(b"x")
    with pytest.raises(sources.SourceNotReadyError, match="changed"):
        sources.require_ready(tmp_path, ["models"])


def test_write_ready_refuses_unverified_and_foreign_roots(tmp_path: Path) -> None:
    profile = make_profile()
    with pytest.raises(ValueError, match="missing: models.zip"):
        sources.write_ready(tmp_path, profile, ["models"])
    sources.fetch(profile, tmp_path, ["models"], downloader=FakeDownloader())
    sources.write_ready(tmp_path, profile, ["models"])
    other = DatasetProfile(**{**profile.__dict__, "revision": "b" * 40})
    with pytest.raises(ValueError, match="identity changed"):
        sources.write_ready(tmp_path, other, ["models"])


def test_exclusions_are_skipped_and_recorded(tmp_path: Path) -> None:
    profile = make_profile(exclusions=["pbr.z01", "train_pbr/000001/depth/000002.png"])
    downloader = FakeDownloader()
    sources.fetch(profile, tmp_path, ["pbr"], downloader=downloader)
    assert downloader.calls == ["pbr.zip"]
    assert sources.verify(tmp_path, profile, ["pbr"]) == []
    receipt = sources.write_ready(tmp_path, profile, ["pbr"])
    assert receipt["parts"]["pbr"]["skipped"] == ["pbr.z01"]
    assert list(receipt["parts"]["pbr"]["files"]) == ["pbr.zip"]
    assert receipt["exclusions"] == ["pbr.z01", "train_pbr/000001/depth/000002.png"]
    assert json.loads((tmp_path / "READY.json").read_text()) == receipt


def test_transductive_parts_require_consent(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="explicit consent"):
        sources.require_transductive_consent(False)
    sources.require_transductive_consent(True)
    profile, downloader = make_profile(), FakeDownloader()
    with pytest.raises(ValueError, match="explicit"):
        sources.fetch(profile, tmp_path, ["test"], downloader=downloader)
    assert downloader.calls == []
    sources.fetch(profile, tmp_path, ["test"], downloader=downloader, allow_transductive=True)
    receipt = sources.write_ready(tmp_path, profile, ["test"])
    assert receipt["transductive_parts"] == ["test"]


def test_unknown_part_and_malformed_profile() -> None:
    with pytest.raises(ValueError, match="no part 'nope'"):
        sources.part_files(make_profile(), ["nope"])
    with pytest.raises(ValueError, match="40-hex"):
        DatasetProfile("x", "r", "abc", {}, {})
    with pytest.raises(ValueError, match="unsafe"):
        DatasetProfile("x", "r", REVISION, {"../a": "0" * 64}, {})
    with pytest.raises(ValueError, match="unpinned"):
        DatasetProfile("x", "r", REVISION, {}, {"p": ["a"]})


def test_default_downloader_names_the_extra_when_hub_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", None)
    with pytest.raises(ImportError, match=re.escape("doper[bop]")):
        sources.default_downloader("r", "f", REVISION, tmp_path)


@pytest.mark.parametrize("name", PACKAGED)
def test_profiles_pin_full_revisions(name: str) -> None:
    profile = sources.load_profile(name)
    assert re.fullmatch(r"[0-9a-f]{40}", profile.revision)
    assert profile.hf_repo == f"bop-benchmark/{'hope' if name == 'hopev2' else name}"
    text = (Path(sources.__file__).parents[1] / "recipes/profiles" / f"{name}.json").read_text()
    # Internal names are checked by the internal release scan, which covers these profiles.
    assert not re.search(r"/mnt|/home", text, re.IGNORECASE)
    assert set(profile.meta["bytes"]) == set(profile.files)
    assert "test" in profile.parts


def test_profile_revisions_match_the_paper_pins() -> None:
    expected = {
        "handal": "627203580f2e639764ae981daf1f4a6d521b5d62",
        "hopev2": "ddd0a26ca3460085e93b648748160fb25e4b5566",
        "hot3d": "30fe9674782f32e1e5edba98476b6ff4300132c5",
        "tless": "5fd309a04476a842d93abfb584fba9ee7caecdf1",
    }
    assert {n: sources.load_profile(n).revision for n in PACKAGED} == expected


def test_load_profile_by_path(tmp_path: Path) -> None:
    packaged = Path(sources.__file__).parents[1] / "recipes/profiles/tless.json"
    copy_path = tmp_path / "mine.json"
    copy_path.write_bytes(packaged.read_bytes())
    assert sources.load_profile(copy_path) == sources.load_profile("tless")
    with pytest.raises(FileNotFoundError, match="packaged profiles"):
        sources.load_profile("nonexistent")


def test_frozen_staging_resources_match_their_pins() -> None:
    hope = sources.load_profile("hopev2")
    inventory = sources.load_resource(hope, "pbr_inventory")
    assert inventory["revision"] == hope.revision and len(inventory["scenes"]) == 50
    hot3d = sources.load_profile("hot3d")
    panel = sources.load_resource(hot3d, "validation_panel")
    index = sources.load_resource(hot3d, "test_image_index")
    assert len(panel["clip_rows"]) == 257 and index["image_count"] == len(index["images"]) == 5140
    protocol = hot3d.meta["camera_protocol"]
    assert (
        protocol["validation_panel_sha256"] == hot3d.meta["resources"]["validation_panel"]["sha256"]
    )
    assert (
        protocol["test_image_index_sha256"] == hot3d.meta["resources"]["test_image_index"]["sha256"]
    )
    assert len(hot3d.parts["test"]) == 1028 and len(hot3d.parts["val"]) == 257


def test_handal_zero_byte_exclusions_are_profile_data() -> None:
    handal = sources.load_profile("handal")
    assert len(handal.exclusions) == 11
    assert len(handal.defects["excluded_frames"]) == 11
    assert len(handal.defects["missing_required_files"]) == 7
    assert "train_pbr/000023/depth/000103.png" in handal.exclusions


def _handal_tree(tmp_path: Path, profile: DatasetProfile) -> Path:
    source = tmp_path / "train_pbr"
    for scene in ("000001", "000002"):
        for name in ("rgb", "depth", "mask_visib"):
            (source / scene / name).mkdir(parents=True)
        frames = {str(i): [{"obj_id": 1}] for i in (487, 233, 5)}
        (source / scene / "scene_gt.json").write_text(json.dumps(frames))
        (source / scene / "scene_camera.json").write_text(json.dumps(frames))
    for name in profile.exclusions:
        path = source / name.split("/", 1)[1]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    return source


def test_clean_source_view_drops_whole_frames_and_keeps_originals(tmp_path: Path) -> None:
    profile = sources.load_profile("handal")
    source = _handal_tree(tmp_path, profile)
    view = sources.clean_source_view(source, tmp_path / "campaign", profile)
    assert set(json.loads((view / "000001/scene_gt.json").read_text())) == {"233", "5"}
    assert set(json.loads((view / "000002/scene_gt.json").read_text())) == {"487", "5"}
    assert (view / "000001/rgb").is_symlink()
    assert set(json.loads((source / "000001/scene_gt.json").read_text())) == {"487", "233", "5"}
    assert sources.clean_source_view(source, tmp_path / "campaign", profile) == view
    (source / "000014/mask_visib/000019_000019.png").write_bytes(b"x")
    with pytest.raises(ValueError, match="source view identity|documented zero-byte"):
        sources.clean_source_view(source, tmp_path / "other", profile)


def test_clean_source_view_survives_moving_the_root(tmp_path: Path) -> None:
    profile = sources.load_profile("handal")
    old = tmp_path / "old"
    source = _handal_tree(old, profile)
    sources.clean_source_view(source, old / "views", profile)
    new = tmp_path / "new"
    old.rename(new)
    view = sources.clean_source_view(new / "train_pbr", new / "views", profile)
    link = view / "000001" / "rgb"
    assert link.is_symlink() and not Path(os.readlink(link)).is_absolute()
    assert link.resolve() == (new / "train_pbr" / "000001" / "rgb").resolve()
    identity = json.loads((view / "identity.json").read_text())
    identity["source"] = "/elsewhere/train_pbr"
    (view / "identity.json").write_text(json.dumps(identity))
    with pytest.raises(ValueError, match=f"{view}.*delete it"):
        sources.clean_source_view(new / "train_pbr", new / "views", profile)


def test_hot3d_defect_policy_fails_closed() -> None:
    profile = sources.load_profile("hot3d")
    affected = sources.hot3d_affected_objects(profile)
    assert affected == {2, 3, 4, 5, 7, 8, 9, 12, 13, 15, 17, 18, 19, 21, 23, 26, 27, 30, 31, 32}
    empty = {"dataset": "hot3d", "excluded_frames": [], "missing_required_files": []}
    sources.validate_hot3d_exclusions(profile, empty, {}, 1)
    with pytest.raises(ValueError, match="requires the filtered"):
        sources.validate_hot3d_exclusions(profile, empty, {}, 8)
    ready = {"identity": {"defects_manifest_sha256": sources.hot3d_defects_sha256(profile)}}
    approved = sources.hot3d_exclusion_record(profile)
    sources.validate_hot3d_exclusions(profile, approved, ready, 8)
    changed = copy.deepcopy(approved)
    changed["excluded_frames"][0]["frame"] = 9
    with pytest.raises(ValueError, match="unapproved"):
        sources.validate_hot3d_exclusions(profile, changed, ready, 8)


def test_hot3d_mask_policy_scopes_records_to_one_object() -> None:
    profile = sources.load_profile("hot3d")
    assert sources.hot3d_mask_records(profile, 1) == []
    assert len(sources.hot3d_mask_records(profile, 4)) == 1
    assert sources.hot3d_mask_generation(profile, 19) == 2
    assert sources.hot3d_mask_policy_sha256(profile, 4) != sources.hot3d_mask_policy_sha256(
        profile, 12
    )
    record = sources.hot3d_mask_exclusion_record(profile, 4)
    ready = {
        "identity": {"mask_policy_sha256": record["mask_policy_sha256"]},
        "permitted_training_objects": [4],
    }
    sources.validate_hot3d_exclusions(profile, record, ready, 4)
    with pytest.raises(ValueError, match="object-scoped"):
        sources.validate_hot3d_exclusions(profile, record, {**ready, "identity": {}}, 4)
    with pytest.raises(ValueError, match="no evidenced"):
        sources.hot3d_mask_exclusion_record(profile, 1)
    bad = copy.deepcopy(profile.defects)
    bad["masks"]["objects"]["4"][0]["file"] = "train_pbr/x.png"
    broken = DatasetProfile(**{**profile.__dict__, "defects": bad})
    with pytest.raises(ValueError, match="unevidenced"):
        sources.hot3d_mask_records(broken, 4)


def test_every_image_filter_uses_the_shared_suffixes():
    """One list of source image suffixes (the research sources' ``.jpg`` and ``.png``)."""
    from doper.bop import export, hot3d_source, multisensor, onboarding, tless_sensors
    from doper.pseudo_labels import monocular

    assert sources.IMAGE_SUFFIXES == (".jpg", ".png")
    for module in (export, multisensor, onboarding, tless_sensors, monocular, hot3d_source):
        assert module.IMAGE_SUFFIXES is sources.IMAGE_SUFFIXES, module.__name__


def test_publish_text_is_atomic_and_synced(tmp_path, monkeypatch):
    synced = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), real_fsync(fd)))
    path = tmp_path / "nested" / "out.csv"
    sources.publish_text(path, "a,b\r\nc\n")
    assert path.read_bytes() == b"a,b\r\nc\n" and len(synced) == 1
    sources.publish_text(path, "replaced\n")
    assert path.read_text() == "replaced\n" and sorted(p.name for p in path.parent.iterdir()) == [
        "out.csv"
    ]


def test_atomic_text_writers_share_publish_text(tmp_path, monkeypatch):
    """The BOP CSV, the model matrix, the journal generation and the adaptation configs go
    through ``publish_text`` (fsync, rename)."""
    from doper.bop import banks, export
    from doper.training import adaptation

    written = []

    def spy(path, text):
        written.append(Path(path).name)
        sources.publish_text(path, text)

    for module in (banks, export, adaptation):
        monkeypatch.setattr(module, "publish_text", spy)
    export.write_csv(tmp_path / "x_tless-test.csv", {}, record_time=False)
    export.ensure_generation(tmp_path / "payload")
    assert written == ["x_tless-test.csv", export.GENERATION_NAME]
    assert not hasattr(adaptation, "_publish_text")


def test_file_lock_names_the_lock_it_cannot_take(tmp_path, monkeypatch):
    import errno
    import fcntl

    lock = tmp_path / ".obj_000001.lock"
    errors = [OSError(errno.ENOLCK, "No locks available")]
    monkeypatch.setattr(fcntl, "flock", lambda *args: (_ for _ in ()).throw(errors[0]))
    with (
        pytest.raises(OSError, match=f"cannot lock {re.escape(str(lock))}"),
        sources.file_lock(lock),
    ):
        pass
    calls = iter([BlockingIOError(errno.EAGAIN, "held"), OSError(errno.ENOLCK, "no locks")])
    monkeypatch.setattr(fcntl, "flock", lambda *args: (_ for _ in ()).throw(next(calls)))
    with pytest.raises(OSError, match="cannot lock .*no locks"), sources.file_lock(lock):
        pass
    monkeypatch.setattr(
        fcntl, "flock", lambda *args: (_ for _ in ()).throw(BlockingIOError(11, ""))
    )
    with (
        pytest.raises(BlockingIOError, match="busy here"),
        sources.file_lock(lock, wait=False, busy="busy here"),
    ):
        pass


def test_sha256_files_keeps_the_order_of_its_paths(tmp_path):
    paths = [tmp_path / f"{i}.bin" for i in range(5)]
    for i, path in enumerate(reversed(paths)):
        path.write_bytes(bytes([i]) * (i + 1))
    assert sources.sha256_files(paths) == [sources.sha256_file(p) for p in paths]
    assert sources.sha256_files([]) == []


def test_check_keys_names_unknown_and_missing_keys():
    sources.check_keys({"a": 1}, {"a", "b"}, ["a"], "x")
    with pytest.raises(ValueError, match=r"^x must be a JSON object$"):
        sources.check_keys([], {"a"}, [], "x")
    with pytest.raises(ValueError, match=r"^y$"):
        sources.check_keys([], {"a"}, [], "x", not_mapping="y")
    with pytest.raises(
        ValueError, match=re.escape("x has unknown keys ['c'] / missing keys ['b']")
    ):
        sources.check_keys({"c": 1}, {"a", "b"}, ["b"], "x")
