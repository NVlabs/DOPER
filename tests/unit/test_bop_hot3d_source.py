# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for HOT3D source extraction, unit canonicalization and the defect filters."""

import dataclasses
import io
import json
import random
import shutil
import struct
import subprocess
import zipfile
import zlib
from pathlib import Path

import pytest

from doper.bop import hot3d_source, sources

PROFILE = sources.load_profile("hot3d")
OBJECTS = list(range(1, 34))
AFFECTED = sorted(sources.hot3d_affected_objects(PROFILE))
MASK_DEPTH_OBJECT = 4
MASK_ONLY_OBJECT = 16


def _dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value))


def _frames() -> dict[str, dict[int, list[int]]]:
    """Scene -> frame -> object ids holding every pinned defect plus ordinary frames."""
    frames: dict[str, dict[int, list[int]]] = {"000001": {0: [1, 2, 3], 1: [2, 2]}}
    for row in PROFILE.defects["depth"]["defective_frames"]:
        frames.setdefault(row["scene"], {})[row["frame"]] = list(row["objects"])
    for obj in (MASK_DEPTH_OBJECT, MASK_ONLY_OBJECT):
        for row in PROFILE.defects["masks"]["objects"][str(obj)]:
            ids = [1, 2, 3, 5, 6, 7, 8]
            ids[row["instance"]] = obj
            frames.setdefault(row["scene"], {})[row["frame"]] = ids
    return frames


@pytest.fixture
def prepared(tmp_path: Path) -> Path:
    """A prepared (meters) HOT3D source root that holds every pinned defect."""
    raw = tmp_path / "raw" / "data"
    counts = {str(obj): 0 for obj in OBJECTS}
    hashes: dict[str, str] = {}
    total = 0
    for scene, by_frame in _frames().items():
        scene_dir = raw / "train_pbr" / scene
        for name in ("rgb", "depth", "mask_visib"):
            (scene_dir / name).mkdir(parents=True)
            (scene_dir / name / "keep.png").write_bytes(b"pixels")
        gt = {
            str(frame): [
                {
                    "cam_R_m2c": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    "cam_t_m2c": [0.1, 0.2, 0.8],
                    "obj_id": o,
                }
                for o in objects
            ]
            for frame, objects in sorted(by_frame.items())
        }
        cams = {
            str(frame): {"cam_K": [600.0, 0, 320.0, 0, 600.0, 240.0, 0, 0, 1.0],
                         "cam_t_w2c": [0.01, 0.02, 0.03], "depth_scale": 0.1}
            for frame in by_frame
        }  # fmt: skip
        _dump(scene_dir / "scene_gt.json", gt)
        _dump(scene_dir / "scene_camera.json", cams)
        _dump(scene_dir / "scene_gt_info.json", {str(f): [] for f in by_frame})
        hashes[scene] = sources.sha256_file(scene_dir / "scene_gt.json")
        for objects in by_frame.values():
            for obj in set(objects):
                counts[str(obj)] += 1
        total += len(by_frame)
    for row in PROFILE.defects["depth"]["defective_frames"]:
        for defect in row["defects"]:
            (raw / defect["file"]).write_bytes(b"")
    for rows in PROFILE.defects["masks"]["objects"].values():
        for row in rows:
            if row["reason"].startswith("zero_byte") and (raw / row["file"]).parent.is_dir():
                (raw / row["file"]).write_bytes(b"")
    for kind in ("models", "models_eval"):
        (raw / kind).mkdir()
    _dump(raw / "train_pbr" / "exclusions.json",
          {"dataset": "hot3d", "excluded_frames": [], "missing_required_files": []})  # fmt: skip
    _dump(raw / "READY.json", {
        "dataset": "hot3d", "revision": PROFILE.revision,
        "identity": {"dataset": "hot3d", "profile_sha256": sources.profile_sha256(PROFILE)},
        "coverage": {"dataset": "hot3d", "frames": total, "scenes": len(_frames()),
                     "object_frame_counts": counts, "scene_gt_sha256": hashes},
        "model_units": {"canonical_metadata_length_units": "millimetres"},
    })  # fmt: skip
    return raw.parent


def _gt(root: Path, scene: str) -> dict:
    return json.loads((root / "data" / "train_pbr" / scene / "scene_gt.json").read_text())


def test_canonicalize_scales_only_translations_and_keeps_pixels(prepared, tmp_path: Path) -> None:
    view = hot3d_source.canonicalize_units(prepared, tmp_path / "canonical")
    assert view == tmp_path / "canonical" / "data"
    pose = _gt(tmp_path / "canonical", "000001")["0"][0]
    assert pose["cam_t_m2c"] == pytest.approx([100.0, 200.0, 800.0])
    assert pose["cam_R_m2c"] == [1, 0, 0, 0, 1, 0, 0, 0, 1]
    camera = json.loads((view / "train_pbr/000001/scene_camera.json").read_text())["0"]
    assert camera["cam_t_w2c"] == pytest.approx([10.0, 20.0, 30.0])
    assert camera["depth_scale"] == 0.1
    assert (view / "train_pbr/000001/rgb").is_symlink()
    assert (view / "train_pbr/000001/rgb/keep.png").read_bytes() == b"pixels"
    assert _gt(prepared, "000001")["0"][0]["cam_t_m2c"] == [0.1, 0.2, 0.8]
    ready = json.loads((view / "READY.json").read_text())
    assert ready["PBR_units"]["canonical_pose_translation_units"] == "millimetres"
    assert ready["PBR_units"]["depth_scale_multiplier"] == 1
    assert hot3d_source.canonicalize_units(prepared, tmp_path / "canonical") == view


@pytest.mark.parametrize("change", ["revision", "profile", "units", "gt", "population"])
def test_canonicalize_rejects_a_source_that_is_not_the_pinned_one(
    prepared, tmp_path, change
) -> None:
    ready_path = prepared / "data" / "READY.json"
    ready = json.loads(ready_path.read_text())
    if change == "revision":
        ready["revision"] = "0" * 40
    elif change == "profile":
        ready["identity"]["profile_sha256"] = "0" * 64
    elif change == "units":
        ready["model_units"]["canonical_metadata_length_units"] = "metres"
    elif change == "population":
        ready["coverage"]["frames"] += 1
    else:
        scene = prepared / "data" / "train_pbr" / "000001" / "scene_gt.json"
        scene.write_text(scene.read_text() + " ")
    _dump(ready_path, ready)
    with pytest.raises(ValueError):
        hot3d_source.canonicalize_units(prepared, tmp_path / "canonical")


@pytest.fixture
def canonical(prepared, tmp_path: Path) -> Path:
    hot3d_source.canonicalize_units(prepared, tmp_path / "canonical")
    return tmp_path / "canonical"


def test_defect_filter_drops_frame_and_keeps_pixels_and_parent(canonical, tmp_path: Path) -> None:
    before = sources.sha256_file(canonical / "data" / "READY.json")
    parent_gt = _gt(canonical, "000016")
    view = hot3d_source.filter_source_defects(canonical, tmp_path / "depthfix")
    scene = view / "train_pbr" / "000016"
    assert "8" not in json.loads((scene / "scene_gt.json").read_text())
    assert "8" not in json.loads((scene / "scene_camera.json").read_text())
    assert "8" not in json.loads((scene / "scene_gt_info.json").read_text())
    assert _gt(canonical, "000016") == parent_gt
    assert sources.sha256_file(canonical / "data" / "READY.json") == before
    assert (scene / "depth").is_symlink()
    assert (scene / "depth" / "000008.png").read_bytes() == b""
    ready = json.loads((view / "READY.json").read_text())
    parent = json.loads((canonical / "data" / "READY.json").read_text())
    assert ready["coverage"]["frames"] == parent["coverage"]["frames"] - 1
    for obj in AFFECTED:
        counts = ready["coverage"]["object_frame_counts"][str(obj)]
        assert counts == parent["coverage"]["object_frame_counts"][str(obj)] - 1
    exclusions = json.loads((view / "train_pbr" / "exclusions.json").read_text())
    sources.validate_hot3d_exclusions(PROFILE, exclusions, ready, AFFECTED[0])
    assert hot3d_source.filter_source_defects(canonical, tmp_path / "depthfix") == view


def test_defect_filter_rejects_changed_evidence(canonical, tmp_path: Path) -> None:
    (canonical / "data" / "train_pbr" / "000016" / "depth" / "000008.png").write_bytes(b"data")
    with pytest.raises(ValueError, match="empty depth no longer"):
        hot3d_source.filter_source_defects(canonical, tmp_path / "view")


def test_defect_filter_requires_the_canonical_millimeter_parent(prepared, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="canonical-mm"):
        hot3d_source.filter_source_defects(prepared, tmp_path / "view")


def test_mask_view_omits_the_frame_for_one_object_and_keeps_instance_indices(
    canonical, tmp_path: Path
) -> None:
    depth_view = tmp_path / "depthfix"
    hot3d_source.filter_source_defects(canonical, depth_view)
    view = hot3d_source.filter_object_masks(depth_view, tmp_path / "mask4", MASK_DEPTH_OBJECT)
    gt = json.loads((view / "train_pbr/000016/scene_gt.json").read_text())
    assert "112" not in gt and "8" not in gt
    ready = json.loads((view / "READY.json").read_text())
    assert ready["permitted_training_objects"] == [MASK_DEPTH_OBJECT]
    exclusions = json.loads((view / "train_pbr/exclusions.json").read_text())
    sources.validate_hot3d_exclusions(PROFILE, exclusions, ready, MASK_DEPTH_OBJECT)
    other = hot3d_source.filter_object_masks(canonical, tmp_path / "mask16", MASK_ONLY_OBJECT)
    assert "49" not in json.loads((other / "train_pbr/000004/scene_gt.json").read_text())
    assert "8" in json.loads((other / "train_pbr/000016/scene_gt.json").read_text())
    untouched = json.loads((canonical / "data/train_pbr/000004/scene_gt.json").read_text())
    assert untouched["49"][3]["obj_id"] == MASK_ONLY_OBJECT
    assert (other / "train_pbr/000004/mask_visib").is_symlink()
    assert (
        hot3d_source.filter_object_masks(canonical, tmp_path / "mask16", MASK_ONLY_OBJECT) == other
    )


def test_mask_view_refuses_wrong_parents_and_objects_without_defects(
    canonical, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="depth-corrected parent"):
        hot3d_source.filter_object_masks(canonical, tmp_path / "a", MASK_DEPTH_OBJECT)
    with pytest.raises(ValueError, match="no evidenced mask defects"):
        hot3d_source.filter_object_masks(canonical, tmp_path / "b", 5)
    scoped = hot3d_source.filter_object_masks(canonical, tmp_path / "c", MASK_ONLY_OBJECT)
    with pytest.raises(ValueError, match="unscoped"):
        hot3d_source.filter_object_masks(scoped.parent, tmp_path / "d", MASK_DEPTH_OBJECT)


def test_mask_view_rejects_changed_evidence(canonical, tmp_path: Path) -> None:
    mask = canonical / "data/train_pbr/000004/mask_visib/000049_000003.png"
    mask.write_bytes(b"not empty")
    with pytest.raises(ValueError, match="empty mask"):
        hot3d_source.filter_object_masks(canonical, tmp_path / "x", MASK_ONLY_OBJECT)


def test_mask_view_rejects_instance_of_another_object(canonical, tmp_path: Path) -> None:
    path = canonical / "data/train_pbr/000004/scene_gt.json"
    gt = json.loads(path.read_text())
    gt["49"][3]["obj_id"] = 1
    _dump(path, gt)
    ready_path = canonical / "data" / "READY.json"
    ready = json.loads(ready_path.read_text())
    ready["coverage"]["scene_gt_sha256"]["000004"] = sources.sha256_file(path)
    _dump(ready_path, ready)
    with pytest.raises(ValueError, match="does not belong"):
        hot3d_source.filter_object_masks(canonical, tmp_path / "x", MASK_ONLY_OBJECT)


# --------------------------------------------------------------------------------------
# Extraction from the pinned archives
# --------------------------------------------------------------------------------------
CUBE_FACES = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
              (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]  # fmt: skip
CUBE_PLY = (
    "ply\nformat ascii 1.0\nelement vertex 8\nproperty float x\nproperty float y\n"
    "property float z\nelement face 12\nproperty list uchar int vertex_indices\nend_header\n"
    + "".join(f"{x} {y} {z}\n" for x in (0, 100) for y in (0, 100) for z in (0, 100))
    + "".join(f"3 {a} {b} {c}\n" for a, b, c in CUBE_FACES)
)
PIXELS = b"x" * 30000
NATIVE_INFO = {"diameter": 0.17, "size_x": 0.1, "size_y": 0.1, "size_z": 0.1, "min_x": 0.0}


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def _pbr_members(scenes: dict[str, list[int]]) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    for scene, objects in scenes.items():
        base = f"train_pbr/{scene}"
        gt = {"0": [{"obj_id": o, "cam_t_m2c": [0.0, 0.0, 0.5]} for o in objects]}
        members[f"{base}/scene_gt.json"] = json.dumps(gt).encode()
        members[f"{base}/scene_camera.json"] = json.dumps(
            {"0": {"cam_K": [500.0, 0, 320.0, 0, 500.0, 240.0, 0, 0, 1.0], "depth_scale": 0.1}}
        ).encode()
        for kind, suffix in (("rgb", "jpg"), ("depth", "png"), ("mask_visib", "png")):
            members[f"{base}/{kind}/000000{'_000000' if kind == 'mask_visib' else ''}.{suffix}"] = (
                PIXELS
            )
    return members


def _model_members() -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    for bank in ("models", "models_eval"):
        members[f"{bank}/models_info.json"] = json.dumps(
            {str(o): NATIVE_INFO for o in (1, 2, 3)}
        ).encode()
        for obj in (1, 2, 3):
            members[f"{bank}/obj_{obj:06d}.ply"] = CUBE_PLY.encode()
    return members


@pytest.fixture
def downloaded(tmp_path: Path) -> tuple[Path, sources.DatasetProfile]:
    """A fetched fixture root: models zip plus a PBR archive split into two volumes."""
    root = tmp_path / "hot3d"
    files = sources.source_dir(root)
    files.mkdir(parents=True)
    _write_zip(files / "hot3d_models.zip", _model_members())
    pbr = tmp_path / "pbr.zip"
    _write_zip(pbr, _pbr_members({"000000": [1, 2], "000003": [1]}))
    if shutil.which("zip") is None:
        pytest.skip("the zip tool is needed to build a split archive")
    subprocess.run(
        ["zip", "-q", "-s", "64k", str(pbr), "--out", str(files / "hot3d_train_pbr.zip")],
        check=True,
    )
    volumes = sorted(p.name for p in files.glob("hot3d_train_pbr.z*") if p.suffix != ".zip")
    assert volumes, "fixture archive must be split"
    names = ["hot3d_models.zip", *volumes, "hot3d_train_pbr.zip"]
    profile = dataclasses.replace(
        PROFILE,
        files={n: sources.sha256_file(files / n) for n in names},
        parts={"models": ["hot3d_models.zip"], "pbr": names[1:]},
        exclusions=[],
        meta={**PROFILE.meta, "objects": [1, 2, 3], "bytes": {}},
    )
    sources.write_ready(root, profile, ["models", "pbr"])
    return root, profile


def test_split_archive_members_read_back_across_volumes(downloaded) -> None:
    root, profile = downloaded
    volumes = [sources.source_dir(root) / name for name in profile.parts["pbr"]]
    expected = _pbr_members({"000000": [1, 2], "000003": [1]})
    with hot3d_source.open_split_zip(volumes) as archive:
        assert len({i.volume for i in archive.infolist()}) >= 2
        for name, content in expected.items():
            assert archive.read(name) == content


def test_extract_hot3d_source_publishes_a_receipted_source(downloaded) -> None:
    root, profile = downloaded
    data = hot3d_source.extract_hot3d_source(root, profile, extract_workers=2, allowed_missing=[3])
    ready = json.loads((data / "READY.json").read_text())
    assert ready["coverage"]["frames"] == 2
    assert ready["coverage"]["object_frame_counts"] == {"1": 2, "2": 1, "3": 0}
    assert ready["coverage"]["missing_objects"] == [3]
    assert ready["identity"]["profile_sha256"] == sources.profile_sha256(profile)
    assert ready["model_units"]["canonical_metadata_length_units"] == "millimetres"
    info = json.loads((data / "models" / "models_info.json").read_text())
    assert info["1"]["size_x"] == pytest.approx(100.0)
    assert (data / "models" / "obj_000001.ply").is_file()
    scene = data / "train_pbr" / "000000"
    assert (scene / "rgb" / "000000.jpg").read_bytes() == PIXELS
    assert json.loads((data / "train_pbr" / "exclusions.json").read_text())["excluded_frames"] == []
    assert hot3d_source.extract_hot3d_source(root, profile, allowed_missing=[3]) == data


def test_extract_requires_acknowledging_uncovered_objects(downloaded) -> None:
    root, profile = downloaded
    with pytest.raises(ValueError, match=r"missing object coverage \[3\]"):
        hot3d_source.extract_hot3d_source(root, profile)
    with pytest.raises(ValueError, match="invalid acknowledged"):
        hot3d_source.extract_hot3d_source(root, profile, allowed_missing=[9])
    assert not (root / "data").exists()


def test_extract_rejects_unsafe_archive_members(tmp_path: Path) -> None:
    for name in ("/etc/passwd", "train_pbr/../x", "other/000000/x.png"):
        info = zipfile.ZipInfo(name)
        with pytest.raises(ValueError, match="unsafe"):
            hot3d_source._safe_member(info, ("train_pbr",))
    link = zipfile.ZipInfo("train_pbr/000000/rgb/a.png")
    link.external_attr = 0o120777 << 16
    with pytest.raises(ValueError, match="symlink"):
        hot3d_source._safe_member(link, ("train_pbr",))
    with pytest.raises(ValueError, match="hierarchy"):
        hot3d_source._required_pbr_member(zipfile.ZipInfo("train_pbr/abc/scene_gt.json"))
    assert not hot3d_source._required_pbr_member(zipfile.ZipInfo("train_pbr/000000/mask/x.png"))


def test_extract_refuses_a_changed_download_identity(downloaded) -> None:
    root, profile = downloaded
    other = dataclasses.replace(profile, meta={**profile.meta, "note": "changed"})
    with pytest.raises(ValueError, match="download identity"):
        hot3d_source.extract_hot3d_source(root, other)


@pytest.mark.parametrize("zip64", [False, True])
def test_split_archive_with_entry_spanning_volumes(tmp_path: Path, zip64: bool) -> None:
    if shutil.which("zip") is None:
        pytest.skip("the zip tool is needed to build a split archive")
    rng = random.Random(3)
    members = {
        name: bytes(rng.getrandbits(8) for _ in range(size))
        for name, size in (("a.bin", 50_000), ("b.bin", 150_000), ("c.bin", 30_000))
    }
    whole = tmp_path / "whole.zip"
    _write_zip(whole, members)
    out = tmp_path / "split" / "x.zip"
    out.parent.mkdir()
    flags = ["-fz"] if zip64 else []
    subprocess.run(["zip", "-q", *flags, "-s", "64k", str(whole), "--out", str(out)], check=True)
    volumes = sorted(out.parent.glob("x.z*"), key=lambda p: (p.suffix == ".zip", p.name))
    assert len(volumes) >= 3
    with hot3d_source.open_split_zip(volumes) as archive:
        assert len({i.volume for i in archive.infolist()}) >= 2
        for name, content in members.items():
            assert archive.read(name) == content
        # b.bin starts in one volume and ends in a later one.
        info = archive.getinfo("b.bin")
        stream_starts = archive.fp.starts  # type: ignore[union-attr]
        start = info.header_offset
        assert any(start < boundary < start + info.compress_size for boundary in stream_starts)


def _write_zip64_split(directory: Path, members: dict[str, bytes], volume_size: int) -> list[Path]:
    """Hand-build a multi-volume zip64 archive (disk and offset in the zip64 extra field)."""
    stream = bytearray()
    entries = []
    for name, content in members.items():
        raw_name = name.encode()
        entries.append((raw_name, len(stream), len(content), zlib.crc32(content)))
        stream += struct.pack("<4s5H3L2H", b"PK\x03\x04", 45, 0, 0, 0, 0x21, entries[-1][3],
                              len(content), len(content), len(raw_name), 0)  # fmt: skip
        stream += raw_name + content

    def locate(position: int) -> tuple[int, int]:
        return position // volume_size, position % volume_size

    central = bytearray()
    for raw_name, offset, size, crc in entries:
        disk, relative = locate(offset)
        extra = struct.pack("<HHQL", 1, 12, relative, disk)
        central += struct.pack(
            "<4s4B4HL2L5H2L", b"PK\x01\x02", 45, 3, 45, 0, 0, 0, 0, 0x21, crc,
            size, size, len(raw_name), len(extra), 0, 0xFFFF, 0, 0, 0xFFFFFFFF,
        )  # fmt: skip
        central += raw_name + extra
    cd_start = len(stream)
    stream += central
    record_start = len(stream)
    cd_disk, cd_offset = locate(cd_start)
    stream += struct.pack("<4sQ2H2L4Q", b"PK\x06\x06", 44, 45, 45, 0, cd_disk, len(entries),
                          len(entries), len(central), cd_offset)  # fmt: skip
    disks = (len(stream) + 20 + 22 - 1) // volume_size + 1
    record_disk, record_offset = locate(record_start)
    stream += struct.pack("<4sLQL", b"PK\x06\x07", record_disk, record_offset, disks)
    stream += struct.pack("<4s4H2LH", b"PK\x05\x06", disks - 1, 0xFFFF, 0xFFFF, 0xFFFF,
                          0xFFFFFFFF, 0xFFFFFFFF, 0)  # fmt: skip
    assert len(stream) <= disks * volume_size
    paths = []
    for index in range(disks):
        suffix = ".zip" if index == disks - 1 else f".z{index + 1:02d}"
        paths.append(directory / f"hand{suffix}")
        paths[-1].write_bytes(bytes(stream[index * volume_size : (index + 1) * volume_size]))
    return paths


def test_zip64_split_archive_with_disk_start_in_extra_field(tmp_path: Path) -> None:
    rng = random.Random(11)
    members = {
        name: bytes(rng.getrandbits(8) for _ in range(size))
        for name, size in (("a.bin", 30_000), ("b.bin", 90_000), ("c.bin", 20_000))
    }
    volumes = _write_zip64_split(tmp_path, members, 64 * 1024)
    assert len(volumes) >= 3
    with hot3d_source.open_split_zip(volumes) as archive:
        assert {i.volume for i in archive.infolist()} >= {0, 1}
        for name, content in members.items():
            assert archive.read(name) == content
    # The stdlib cannot read this archive: the zip64 locator reports several disks.
    with pytest.raises(zipfile.BadZipFile, match="multiple disks"):
        zipfile.ZipFile(io.BytesIO(b"".join(p.read_bytes() for p in volumes)))


def _build_views(raw: Path, views: Path) -> list[Path]:
    canonical = hot3d_source.canonicalize_units(raw, views / "canonical")
    depth = hot3d_source.filter_source_defects(canonical.parent, views / "depthfix")
    mask = hot3d_source.filter_object_masks(depth.parent, views / "mask4", MASK_DEPTH_OBJECT)
    return [canonical, depth, mask]


def test_views_survive_moving_the_root_and_record_builder_versions(
    prepared, tmp_path: Path
) -> None:
    built = _build_views(prepared, tmp_path / "views")
    identity = json.loads((built[-1] / "READY.json").read_text())["identity"]
    assert identity["canonicalizer_sha256"] == sources.canonical_sha256(
        {"builder": hot3d_source.CANONICALIZER_VERSION}
    )
    assert identity["filter_sha256"] == sources.canonical_sha256(
        {"builder": hot3d_source.MASK_FILTER_VERSION}
    )
    moved = tmp_path / "moved"
    moved.mkdir()
    prepared.rename(moved / "raw")
    (tmp_path / "views").rename(moved / "views")
    again = _build_views(moved / "raw", moved / "views")
    assert again == [moved / "views" / name / "data" for name in ("canonical", "depthfix", "mask4")]
    for view in again:
        links = [p for p in view.rglob("*") if p.is_symlink()]
        assert links and all(not Path(p.readlink()).is_absolute() and p.exists() for p in links)
    assert (again[-1] / "train_pbr/000001/rgb/keep.png").read_bytes() == b"pixels"


def test_view_of_another_builder_version_names_the_directory_to_delete(
    prepared, tmp_path: Path, monkeypatch
) -> None:
    hot3d_source.canonicalize_units(prepared, tmp_path / "canonical")
    monkeypatch.setattr(hot3d_source, "CANONICALIZER_VERSION", "hot3d-canonicalize-units-v2")
    with pytest.raises(ValueError, match=f"{tmp_path / 'canonical'}.*delete it"):
        hot3d_source.canonicalize_units(prepared, tmp_path / "canonical")
