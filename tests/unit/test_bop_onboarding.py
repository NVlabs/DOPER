# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Onboarding, test and GT views of the BOP real-image archives, in one canonical layout."""

import io
import json
import shutil
import tarfile
import zipfile
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from bop_recipe_data import IDENTITY_POSE, NATIVE_HEIGHT, NATIVE_WIDTH, STREAMS, make_hot3d_download

from doper.bop import hot3d_staging, onboarding, onboarding_hot3d, sources

WIDTH, HEIGHT = 16, 12
INTRINSICS = [20.0, 0.0, 8.0, 0.0, 20.0, 6.0, 0.0, 0.0, 1.0]
STATIC_SEQUENCES = ("obj_000001_up", "obj_000001_down", "obj_000002_up", "obj_000003_up")
FRAMES = (0, 1, 2)
GT_FRAMES = (0, 1)
TEST_FRAMES = (1, 2)


def png(seed: int, *, gray: bool = False) -> bytes:
    rng = np.random.default_rng(seed)
    shape = (HEIGHT, WIDTH) if gray else (HEIGHT, WIDTH, 3)
    ok, data = cv2.imencode(".png", rng.integers(1, 255, shape, dtype=np.uint8))
    assert ok
    return data.tobytes()


def camera(frames, *, width: int = WIDTH, height: int = HEIGHT) -> bytes:
    rows = {
        str(frame): {
            "cam_K": INTRINSICS,
            "width": width,
            "height": height,
            "cam_R_w2c": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            "cam_t_w2c": [1.0, 2.0, 3.0],
            "depth_scale": 1.0,
        }
        for frame in frames
    }
    return json.dumps(rows).encode()


def scene_gt(obj: int, frames, *, instances: dict[int, int] | None = None) -> bytes:
    pose = {"cam_R_m2c": [1, 0, 0, 0, 1, 0, 0, 0, 1], "cam_t_m2c": [0.0, 0.0, 500.0]}
    counts = instances or {}
    return json.dumps(
        {str(f): [{"obj_id": obj, **pose} for _ in range(counts.get(f, 1))] for f in frames}
    ).encode()


def write_zip(path: Path, members: dict[str, bytes]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path


def static_members(prefix: str, *, instances: dict[int, int] | None = None) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    for index, sequence in enumerate(STATIC_SEQUENCES):
        obj = int(sequence.split("_")[1])
        for frame in FRAMES:
            members[f"{prefix}/{sequence}/rgb/{frame:06d}.png"] = png(10 * index + frame)
        members[f"{prefix}/{sequence}/scene_camera.json"] = camera(FRAMES)
        members[f"{prefix}/{sequence}/scene_gt.json"] = scene_gt(
            obj, GT_FRAMES, instances=instances
        )
        for frame in GT_FRAMES:
            members[f"{prefix}/{sequence}/mask_visib/{frame:06d}_000000.png"] = png(
                100 + frame, gray=True
            )
    return members


def dynamic_members(prefix: str) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    for obj in (1, 2):
        members.update(
            {f"{prefix}/obj_{obj:06d}/rgb/{f:06d}.png": png(200 + f + obj) for f in FRAMES}
        )
        members[f"{prefix}/obj_{obj:06d}/scene_camera.json"] = camera(FRAMES)
        members[f"{prefix}/obj_{obj:06d}/scene_gt.json"] = scene_gt(obj, FRAMES)
    return members


def archive_test_members(*, width: int = WIDTH) -> dict[str, bytes]:
    members = {f"test/000001/rgb/{f:06d}.png": png(300 + f) for f in TEST_FRAMES}
    members["test/000001/scene_camera.json"] = camera(TEST_FRAMES, width=width)
    members["test/000001/scene_gt.json"] = scene_gt(1, TEST_FRAMES)
    return members


class Dataset:
    def __init__(self, root: Path, profile: sources.DatasetProfile) -> None:
        self.root = root
        self.profile = profile


def make_dataset(
    tmp_path: Path,
    *,
    name: str = "handal",
    val_image: bytes | None = None,
    instances: dict[int, int] | None = None,
    test_policy: str | None = None,
    test_images: int = len(TEST_FRAMES),
    test_width: int = WIDTH,
) -> Dataset:
    """A BOP-form dataset with static, dynamic and test archives and an extracted val split."""
    root = tmp_path / "root"
    prefixes = {
        "handal": ("static", "dynamic"),
        "hopev2": ("onboarding_static", "onboarding_dynamic"),
    }
    static_prefix, dynamic_prefix = prefixes[name]
    archives = {
        "fx_static.zip": static_members(static_prefix, instances=instances),
        "fx_dynamic.zip": dynamic_members(dynamic_prefix),
        "fx_test.zip": archive_test_members(width=test_width),
    }
    files = {}
    for archive, members in archives.items():
        path = write_zip(root / "source" / archive, members)
        files[archive] = sources.sha256_file(path)
    val = root / "data" / "val" / "000001"
    (val / "rgb").mkdir(parents=True)
    (val / "rgb" / "000000.png").write_bytes(val_image or png(999))
    (val / "scene_gt.json").write_bytes(scene_gt(1, (0,)))
    (val / "scene_camera.json").write_bytes(camera((0,)))
    (val / "scene_gt_info.json").write_text("{}")
    meta = {
        "objects": [1, 2],
        "onboarding_prefix": {"static": static_prefix, "dynamic": dynamic_prefix, "test": "test"},
        "counts": {"static_rgb": 9, "dynamic_rgb": 6},
        "test_population": {"images": test_images, "scenes": 1},
    }
    if test_policy or name == "handal":
        meta["overlap_policy"] = test_policy or "record"
    profile = sources.DatasetProfile(
        name=name,
        hf_repo="bop-benchmark/fixture",
        revision="0" * 40,
        files=files,
        parts={"static": ["fx_static.zip"], "dynamic": ["fx_dynamic.zip"], "test": ["fx_test.zip"]},
        meta=meta,
    )
    return Dataset(root, profile)


def stage(dataset: Dataset, kind: str, **kwargs):
    return onboarding.stage_view(dataset.root, dataset.profile, kind, **kwargs)


def read(path: Path):
    return json.loads(path.read_text())


def test_hope_prefix_maps_to_canonical_layout(tmp_path):
    dataset = make_dataset(tmp_path, name="hopev2")
    summary = stage(dataset, "static")
    view = dataset.root / "onboarding" / "static"
    sequence = view / "obj_000001_up"
    assert (sequence / "rgb" / "000000.png").is_file()
    assert (sequence / "scene_camera.json").is_file()
    assert not (view / "onboarding_static").exists() and not (view / "static").exists()
    assert summary.kind == "static"
    assert summary.sequences == ("obj_000001_down", "obj_000001_up", "obj_000002_up")
    assert summary.rgb_counts == dict.fromkeys(summary.sequences, len(FRAMES))
    assert summary.excluded_validation == () and summary.ready == view / "READY.json"
    assert sorted(p.name for p in view.iterdir() if p.is_dir()) == list(summary.sequences)
    assert not list(dataset.root.rglob(".staging-*"))


def test_handal_prefix_and_dynamic_view(tmp_path):
    dataset = make_dataset(tmp_path)
    stage(dataset, "static")
    summary = stage(dataset, "dynamic")
    assert summary.sequences == ("obj_000001", "obj_000002")
    assert (dataset.root / "onboarding/dynamic/obj_000001/rgb/000002.png").is_file()
    assert not (dataset.root / "onboarding/dynamic/obj_000001/scene_gt.json").exists()
    ready = read(summary.ready)
    assert ready["source_kind"] == "onboarding_dynamic"
    assert ready["gt_supervision_staged"] is False and ready["camera_extrinsics_staged"] is False


def test_sanitize_camera_keeps_only_intrinsics():
    source = json.loads(camera(FRAMES))
    cleaned = onboarding.sanitize_camera(source)
    assert set(cleaned) == {"0", "1", "2"}
    assert all(set(row) == {"cam_K", "width", "height"} for row in cleaned.values())
    assert cleaned["0"]["cam_K"] == INTRINSICS and "cam_t_w2c" in source["0"]
    for bad in (
        {"0": {"cam_K": INTRINSICS[:8]}},
        {"0": {"cam_K": [0.0, *INTRINSICS[1:]]}},
        {"0": {"cam_K": [*INTRINSICS[:8], 2.0]}},
        {"0": {"cam_K": [float("nan"), *INTRINSICS[1:]]}},
        {"0": {"width": 3}},
        {},
    ):
        with pytest.raises(ValueError, match="intrinsics|empty"):
            onboarding.sanitize_camera(bad)


def test_staged_camera_files_hold_no_extrinsics(tmp_path):
    dataset = make_dataset(tmp_path)
    stage(dataset, "static")
    rows = read(dataset.root / "onboarding/static/obj_000001_up/scene_camera.json")
    assert all(set(row) == {"cam_K", "width", "height"} for row in rows.values())


def test_view_selects_requested_objects_and_resumes(tmp_path):
    dataset = make_dataset(tmp_path)
    first = stage(dataset, "static", objects=[2])
    assert first.sequences == ("obj_000002_up",)
    second = stage(dataset, "static", objects=[1, 2])
    assert second.sequences == ("obj_000001_down", "obj_000001_up", "obj_000002_up")
    assert sorted(read(second.ready)["sequences"]) == list(second.sequences)
    with pytest.raises(ValueError, match="object 3"):
        stage(dataset, "static", objects=[3])


def test_view_refuses_unsafe_archive_members(tmp_path):
    dataset = make_dataset(tmp_path)
    members = static_members("static")
    members["static/obj_000001_up/../../escape.png"] = b"x"
    path = write_zip(dataset.root / "source" / "fx_static.zip", members)
    files = {**dataset.profile.files, "fx_static.zip": sources.sha256_file(path)}
    profile = replace(dataset.profile, files=files)
    with pytest.raises(ValueError, match="unsafe"):
        onboarding.stage_view(dataset.root, profile, "static")


def test_test_view_requires_consent(tmp_path):
    dataset = make_dataset(tmp_path)
    with pytest.raises(ValueError, match="transductive"):
        stage(dataset, "test")
    assert not (dataset.root / "onboarding" / "test").exists()
    summary = stage(dataset, "test", consent=True)
    ready = read(summary.ready)
    assert summary.sequences == ("000001",) and ready["transductive_test_adaptation"] is True
    assert ready["test_targets_staged"] is False and ready["gt_supervision_staged"] is False
    assert not (dataset.root / "onboarding/test/000001/scene_gt.json").exists()


def test_test_view_checks_the_pinned_population(tmp_path):
    with pytest.raises(ValueError, match="inventory"):
        stage(make_dataset(tmp_path / "a", test_images=3), "test", consent=True)
    with pytest.raises(ValueError, match="size"):
        stage(make_dataset(tmp_path / "b", test_width=WIDTH + 1), "test", consent=True)


def test_handal_test_records_validation_overlap(tmp_path):
    dataset = make_dataset(tmp_path, val_image=png(301), test_policy="record")
    summary = stage(dataset, "test", consent=True)
    assert summary.excluded_validation == ("000001/rgb/000001.png",)
    ready = read(summary.ready)
    assert ready["validation_overlap"] == [
        {
            "path": "000001/rgb/000001.png",
            "sha256": sources.sha256_file(dataset.root / "onboarding/test/000001/rgb/000001.png"),
        }
    ]
    assert (dataset.root / "onboarding/test/000001/rgb/000001.png").is_file()


def test_handal_static_and_dynamic_record_validation_overlap(tmp_path):
    static = make_dataset(tmp_path / "s", val_image=png(0))
    summary = stage(static, "static")
    assert summary.excluded_validation == ("obj_000001_up/rgb/000000.png",)
    assert read(summary.ready)["validation_overlap"][0]["path"] == summary.excluded_validation[0]
    dynamic = make_dataset(tmp_path / "d", val_image=png(201))
    summary = stage(dynamic, "dynamic")
    assert summary.excluded_validation == ("obj_000001/rgb/000000.png",)
    assert (dynamic.root / "onboarding/dynamic/obj_000001/rgb/000000.png").is_file()


def test_hope_raises_on_validation_overlap(tmp_path):
    dataset = make_dataset(tmp_path, name="hopev2", val_image=png(0))
    with pytest.raises(ValueError, match="overlap"):
        stage(dataset, "static")
    assert not (dataset.root / "onboarding" / "static" / "READY.json").exists()
    assert not list(dataset.root.rglob(".staging-*"))
    test = make_dataset(tmp_path / "t", name="hopev2", val_image=png(301))
    with pytest.raises(ValueError, match="overlap"):
        stage(test, "test", consent=True)


def test_view_records_validation_hashes(tmp_path):
    dataset = make_dataset(tmp_path)
    summary = stage(dataset, "static")
    ready = read(summary.ready)
    population = onboarding.load_validation(dataset.root, dataset.profile)
    assert ready["evaluation_manifest_sha256"] == population.sha256
    assert ready["validation_frames"] == 1 and len(population.hashes) == 1
    assert population.hashes == {
        sources.sha256_file(dataset.root / "data/val/000001/rgb/000000.png")
    }
    assert ready["identity"]["archive_sha256"] == dataset.profile.files["fx_static.zip"]
    sequence = ready["sequences"]["obj_000001_up"]
    assert sequence["rgb_count"] == 3
    camera_file = dataset.root / "onboarding/static/obj_000001_up/scene_camera.json"
    assert sequence["camera_sha256"] == sources.sha256_file(camera_file)


def test_view_requires_the_extracted_validation_split(tmp_path):
    dataset = make_dataset(tmp_path)
    (dataset.root / "data/val/000001/scene_gt.json").unlink()
    with pytest.raises(ValueError, match="validation"):
        stage(dataset, "static")


def test_gt_view_layout_and_receipt(tmp_path):
    dataset = make_dataset(tmp_path)
    stage(dataset, "static")
    summary = onboarding.stage_gt_view(dataset.root, dataset.profile, objects=[1])
    gt = dataset.root / "onboarding" / "gt"
    assert summary.kind == "gt" and summary.sequences == ("obj_000001_down", "obj_000001_up")
    scene = gt / "obj_000001_up"
    assert (scene / "rgb").is_symlink() and (scene / "rgb").readlink() == Path(
        "../../static/obj_000001_up/rgb"
    )
    assert (scene / "rgb" / "000001.png").is_file()
    assert sorted(p.name for p in (scene / "mask_visib").iterdir()) == [
        "000000_000000.png",
        "000001_000000.png",
    ]
    assert set(read(scene / "scene_gt.json")) == {"0", "1"}
    assert set(read(scene / "scene_camera.json")) == {"0", "1"}
    ready = read(summary.ready)
    assert ready["object_frame_counts"] == {"1": 4}
    assert {f["mask_dir"] for f in ready["frames"]} == {"mask_visib"}
    first = ready["frames"][0]
    assert first["rgb"] == "obj_000001_down/rgb/000000.png"
    assert first["mask"] == "obj_000001_down/mask_visib/000000_000000.png"
    assert first["mask_sha256"] == sources.sha256_file(gt / first["mask"])
    assert first["rgb_sha256"] == sources.sha256_file(gt / first["rgb"])
    assert ready["gt_supervision_staged"] is True and ready["no_pose_interpolation"] is True


def test_gt_view_requires_one_instance_per_frame(tmp_path):
    dataset = make_dataset(tmp_path, instances={1: 2})
    stage(dataset, "static")
    with pytest.raises(ValueError, match=r"frame 1\b"):
        onboarding.stage_gt_view(dataset.root, dataset.profile, objects=[1])
    assert not (dataset.root / "onboarding/gt/READY.json").exists()
    assert not list(dataset.root.rglob(".staging-*"))


def test_gt_view_requires_the_static_view(tmp_path):
    dataset = make_dataset(tmp_path)
    with pytest.raises(ValueError, match="static view"):
        onboarding.stage_gt_view(dataset.root, dataset.profile)


def test_gt_view_refuses_validation_overlap(tmp_path):
    dataset = make_dataset(tmp_path, val_image=png(0))
    shutil_val = dataset.root / "data/val/000001/rgb/000000.png"
    shutil_val.write_bytes(png(999))
    stage(dataset, "static")
    shutil_val.write_bytes(png(0))
    with pytest.raises(ValueError, match="overlap"):
        onboarding.stage_gt_view(dataset.root, dataset.profile, objects=[1])


# ---------------------------------------------------------------------------------------
# HOT3D per-object onboarding tars
# ---------------------------------------------------------------------------------------
def add_member(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def write_reference(path: Path, stream: str, frames, obj: int, *, static: bool, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    camera_row = {
        stream: {
            "calibration": {
                "image_width": NATIVE_WIDTH,
                "image_height": NATIVE_HEIGHT,
                "projection_model_type": "CameraModelType.FISHEYE624",
                "projection_params": [40.0, NATIVE_WIDTH / 2, NATIVE_HEIGHT / 2] + [0.0] * 12,
                "label": "fixture",
                "serial_number": "fixture",
            },
            "T_world_from_camera": IDENTITY_POSE,
        }
    }
    pose = {"quaternion_wxyz": [1.0, 0.0, 0.0, 0.0], "translation_xyz": [0.0, 0.0, 0.4]}
    rng = np.random.default_rng(seed)
    with tarfile.open(path, "w") as tar:
        for frame in frames:
            image = rng.integers(0, 256, (NATIVE_HEIGHT, NATIVE_WIDTH, 3), dtype=np.uint8)
            add_member(
                tar, f"{frame:06d}.image_{stream}.jpg", cv2.imencode(".jpg", image)[1].tobytes()
            )
            add_member(tar, f"{frame:06d}.cameras.json", json.dumps(camera_row).encode())
            if static:
                row = {str(obj): [{"T_world_from_object": pose}]}
                add_member(tar, f"{frame:06d}.objects.json", json.dumps(row).encode())
                mask = np.zeros((NATIVE_HEIGHT, NATIVE_WIDTH), np.uint8)
                mask[10:30, 20:40] = 1
                add_member(
                    tar, f"{frame:06d}.mask_{stream}.png", cv2.imencode(".png", mask)[1].tobytes()
                )


def hot3d_dataset(tmp_path: Path, *, frames: dict[str, tuple[int, ...]] | None = None):
    """The HOT3D fixture plus object 1's reference tars and a frozen validation manifest."""
    frames = frames or {"aria": (3, 4), "quest3": (7,)}
    root = tmp_path / "root"
    profile_path = make_hot3d_download(root, tmp_path / "profile")
    data = json.loads(profile_path.read_text())
    names = []
    for kind in ("static", "dynamic"):
        for device in ("aria", "quest3"):
            for orientation in ("_up", "_down") if kind == "static" else ("",):
                names.append(f"object_ref_{device}_{kind}/obj_000001{orientation}.tar")
    for seed, name in enumerate(names):
        device = "Aria" if "_aria_" in name else "Quest3"
        write_reference(
            root / "source" / name,
            STREAMS[device],
            frames[device.lower()],
            1,
            static="_static/" in name,
            seed=seed,
        )
        data["files"][name] = sources.sha256_file(root / "source" / name)
    data["parts"]["static"] = [n for n in names if "_static/" in n]
    data["parts"]["dynamic"] = [n for n in names if "_dynamic/" in n]
    data["meta"]["onboarding_tars"] = {"1": names}
    profile_path.write_text(json.dumps(data))
    profile = sources.load_profile(profile_path)
    manifest = {
        "identity": hot3d_staging.validation_identity(profile),
        "frames": [{"sha256": "a" * 64, "native_rgb_sha256": "b" * 64}],
    }
    (root / hot3d_staging.VALIDATION_MANIFEST).write_text(json.dumps(manifest))
    return root, profile


def test_hot3d_object_logical_ids_and_mask_dir(tmp_path):
    root, profile = hot3d_dataset(tmp_path)
    summary = onboarding_hot3d.stage_hot3d_object(root, profile, 1)
    base = root / "onboarding"
    assert summary.sequences == ("obj_000001", "obj_000001_down", "obj_000001_up")
    up = base / "static" / "obj_000001_up"
    assert sorted(p.name for p in (up / "rgb").iterdir()) == [
        "000003.jpg",
        "000004.jpg",
        "1000007.jpg",
    ]
    assert set(read(up / "scene_camera.json")) == {"3", "4", "1000007"}
    assert summary.rgb_counts["obj_000001_up"] == 3 and summary.rgb_counts["obj_000001"] == 3
    scene = base / "gt" / "obj_000001_up"
    assert set(read(scene / "scene_gt.json")) == {"3", "4", "1000007"}
    assert (scene / "rgb").readlink() == Path("../../static/obj_000001_up/rgb")
    assert sorted(p.name for p in (scene / "mask_supplied").iterdir()) == [
        "000003_000000.png",
        "000004_000000.png",
        "1000007_000000.png",
    ]
    ready = read(base / "gt" / "READY.json")
    assert {f["mask_dir"] for f in ready["frames"]} == {"mask_supplied"}
    assert {f["frame"] for f in ready["frames"]} == {3, 4, 1000007}
    mask = cv2.imread(str(scene / "mask_supplied" / "000003_000000.png"), cv2.IMREAD_UNCHANGED)
    assert set(np.unique(mask)) <= {0, 255}
    image = cv2.imread(str(up / "rgb" / "000003.jpg"))
    assert image.shape[:2] == (NATIVE_HEIGHT, NATIVE_WIDTH)
    assert summary.ready.is_file() and not list(root.rglob(".staging-*"))
    again = onboarding_hot3d.stage_hot3d_object(root, profile, 1)
    assert again.rgb_counts == summary.rgb_counts


def test_hot3d_object_restages_when_a_receipted_view_is_gone(tmp_path):
    root, profile = hot3d_dataset(tmp_path)
    summary = onboarding_hot3d.stage_hot3d_object(root, profile, 1)
    ready = root / "onboarding/static/READY.json"
    ready.unlink()
    shutil.rmtree(root / "onboarding/static/obj_000001_up")
    again = onboarding_hot3d.stage_hot3d_object(root, profile, 1)
    assert again.rgb_counts == summary.rgb_counts and ready.is_file()
    assert (root / "onboarding/static/obj_000001_up/rgb/000003.jpg").is_file()


def test_hot3d_object_refuses_a_frame_outside_the_device_namespace(tmp_path, monkeypatch):
    root, profile = hot3d_dataset(tmp_path)
    real = hot3d_staging.native_members

    def shifted(tar, stream, **kwargs):
        return {fid + 999_997: row for fid, row in real(tar, stream, **kwargs).items()}

    monkeypatch.setattr(hot3d_staging, "native_members", shifted)
    with pytest.raises(ValueError, match="namespace"):
        onboarding_hot3d.stage_hot3d_object(root, profile, 1)


def test_hot3d_object_raises_on_native_or_rectified_overlap(tmp_path):
    root, profile = hot3d_dataset(tmp_path)
    onboarding_hot3d.stage_hot3d_object(root, profile, 1)
    staged = root / "onboarding/static/obj_000001_up/rgb/000003.jpg"
    rectified = sources.sha256_file(staged)
    with tarfile.open(root / "source/object_ref_aria_static/obj_000001_up.tar") as tar:
        member = tar.extractfile("000003.image_214-1.jpg")
        assert member is not None
        import hashlib

        native = hashlib.sha256(member.read()).hexdigest()
    for key, value in (("sha256", rectified), ("native_rgb_sha256", native)):
        fresh_root, fresh = hot3d_dataset(tmp_path / key)
        path = fresh_root / hot3d_staging.VALIDATION_MANIFEST
        manifest = json.loads(path.read_text())
        manifest["frames"][0][key] = value
        path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="overlap"):
            onboarding_hot3d.stage_hot3d_object(fresh_root, fresh, 1)
        assert not (fresh_root / "onboarding/static/READY.json").exists()


def test_hot3d_object_needs_the_validation_manifest_of_this_profile(tmp_path):
    root, profile = hot3d_dataset(tmp_path)
    path = root / hot3d_staging.VALIDATION_MANIFEST
    manifest = json.loads(path.read_text())
    manifest["identity"]["revision"] = "f" * 40
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity"):
        onboarding_hot3d.stage_hot3d_object(root, profile, 1)
    with pytest.raises(ValueError, match="object 99"):
        onboarding_hot3d.stage_hot3d_object(root, profile, 99)


def _plan(root: Path, *argv: str) -> list[str]:
    from doper import cli
    from doper.cli import prepare

    args = cli.build_parser().parse_args(["prepare", *argv, "--root", str(root), "--dry-run"])
    return prepare.plan(args)


def test_prepare_plans_onboarding_views_without_extracting_them(tmp_path):
    steps = _plan(tmp_path, "handal", "--parts", "val", "static", "dynamic")
    names = [s.name for s in steps]
    extract = next(s for s in steps if s.name == "extract")
    assert extract.inputs["parts"] == ["val"]
    assert "stage static onboarding" in names and "stage dynamic onboarding" in names
    assert "stage static GT onboarding" in names and "stage test onboarding" not in names
    names = [
        s.name for s in _plan(tmp_path, "handal", "--parts", "val", "test", "--allow-test-inputs")
    ]
    assert "stage test onboarding" in names
    names = [
        s.name for s in _plan(tmp_path, "hot3d", "--parts", "static", "dynamic", "--obj-id", "3")
    ]
    assert names == ["stage HOT3D onboarding obj 3"]


def test_handal_dynamic_overlap_is_recorded_then_excluded_by_the_exporter(tmp_path):
    """The HANDAL dynamic view keeps a frame that equals a validation image (recorded in
    READY.json); the pseudo-label exporter never shows it to the teacher."""
    from doper.pseudo_labels import monocular
    from doper.pseudo_labels.gate import GateSpec

    dataset = make_dataset(tmp_path, val_image=png(201))
    summary = stage(dataset, "dynamic")
    overlap = read(summary.ready)["validation_overlap"]
    assert [o["path"] for o in overlap] == ["obj_000001/rgb/000000.png"]
    view = dataset.root / "onboarding" / "dynamic"
    seen = []

    def teacher(image, camera_matrix, keypoints_mm, score_threshold):
        seen.append(cv2.imencode(".png", image)[1].tobytes())
        return [], np.zeros(image.shape[:2], np.float32)

    keypoints = tmp_path / "keypoints_3d.json"
    keypoints.write_text(json.dumps({"keypoints_3d": [[0.0, 0.0, 0.0]] * 4}))
    pool = monocular.export_pool(
        teacher, keypoints, view, "dynamic", frozenset({"0" * 64}), GateSpec(),
        tmp_path / "pool", teacher_sha256="t" * 64, obj_id=1,
    )  # fmt: skip
    assert pool.reject_reasons["reserved_eval"] == 1 and pool.accepted == 0
    assert pool.frames_seen == len(seen) + 1
    manifest = read(tmp_path / "pool" / "manifest.json")
    assert overlap[0]["sha256"] not in {r["source_image_sha256"] for r in manifest["records"]}
