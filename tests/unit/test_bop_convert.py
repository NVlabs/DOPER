# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the BOP train_pbr to DOPER COCO conversion."""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from bop_pbr_data import (
    CAMERA_MATRIX,
    HEIGHT,
    WIDTH,
    make_pbr,
    write_keypoints,
    write_scene,
)
from tiny_coco import tiny_model_config

from doper.bop import convert, hot3d_source
from doper.config import AugConfig, DataConfig, DoperConfig
from doper.dataset import DoperDataset
from doper.scene import geometry


def _run(pbr: Path, keypoints: Path, out: Path, **kwargs) -> convert.ConvertSummary:
    kwargs.setdefault("min_visible_keypoints", 1)
    return convert.convert_pbr(pbr, 2, keypoints, out, **kwargs)


def _single(tmp_path: Path, frames: dict) -> tuple[Path, Path, Path]:
    keypoints = write_keypoints(tmp_path / "kp.json")
    pbr = tmp_path / "train_pbr"
    scene = write_scene(pbr, "000000", frames)
    return pbr, keypoints, scene


ONE_FRAME = {0: [(2, np.eye(3), (0.0, 0.0, 300.0))]}


@pytest.fixture
def bop_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """A three-scene, three-frame, two-object ``train_pbr`` and a keypoint file."""
    return make_pbr(tmp_path), write_keypoints(tmp_path / "kp.json")


def test_convert_round_trip_projects_keypoints_on_gt(bop_fixture, tmp_path: Path) -> None:
    pbr, keypoints = bop_fixture
    summary = convert.convert_pbr(pbr, 2, keypoints, tmp_path / "out", min_visible_keypoints=1)
    data = json.loads((summary.out / "annotations.json").read_text())
    points_mm = np.asarray(json.loads(keypoints.read_text())["keypoints_3d"]) * 1000
    assert summary.images == 9 and summary.annotations == len(data["annotations"]) == 9
    for annotation in data["annotations"]:
        bop = annotation["bop"]
        gt = json.loads((pbr / bop["scene"] / "scene_gt.json").read_text())[str(bop["frame"])]
        pose = gt[bop["instance"]]
        rotation = np.asarray(pose["cam_R_m2c"]).reshape(3, 3)
        camera = points_mm @ rotation.T + pose["cam_t_m2c"]
        projected = camera @ CAMERA_MATRIX.T
        expected = projected[:, :2] / projected[:, 2:3]
        stored = np.asarray(annotation["keypoints"]).reshape(-1, 3)
        assert np.abs(stored[:, :2] - expected).max() < 0.5


def test_convert_output_loads_in_doper_dataset(bop_fixture, tmp_path: Path) -> None:
    pbr, keypoints = bop_fixture
    summary = convert.convert_pbr(pbr, 2, keypoints, tmp_path / "out", min_visible_keypoints=1)
    config = DoperConfig(
        data=DataConfig(root=str(summary.out), train_split="."),
        model=tiny_model_config(num_keypoints=4, det_input_size=64, crop_size=32),
        aug=AugConfig(enabled=False),
    )
    dataset = DoperDataset(config, "train")
    assert len(dataset) == summary.images
    sample = dataset[0]
    assert sample is not None


def test_projection_units_visibility_masks_instances_and_provenance(tmp_path: Path) -> None:
    keypoints = write_keypoints(tmp_path / "kp.json")
    pbr = tmp_path / "train_pbr"
    scene = write_scene(
        pbr,
        "000003",
        {
            17: [
                (2, np.eye(3), (0.0, 0.0, 300.0)),
                (7, np.eye(3), (50.0, 0.0, 300.0)),
                (2, np.eye(3), (-70.0, 0.0, 300.0)),
            ]
        },
    )
    source_bytes = (scene / "rgb" / "000017.jpg").read_bytes()
    out = tmp_path / "out"
    _run(pbr, keypoints, out, copy_images=True)
    payload = json.loads((out / "annotations.json").read_text())
    assert [image["file_name"] for image in payload["images"]] == ["000003_000017.jpg"]
    assert (out / "images" / "000003_000017.jpg").read_bytes() == source_bytes
    assert [a["id"] for a in payload["annotations"]] == [0, 1]
    assert [a["bop"]["instance"] for a in payload["annotations"]] == [0, 2]
    assert [a["mask_file"] for a in payload["annotations"]] == [
        "masks/000003_000017_000000.png",
        "masks/000003_000017_000002.png",
    ]
    assert (out / payload["annotations"][0]["mask_file"]).read_bytes() == (
        scene / "mask_visib" / "000017_000000.png"
    ).read_bytes()
    first, second = payload["annotations"]
    projected = np.asarray(first["keypoints"]).reshape(-1, 3)
    assert np.allclose(projected[0, :2], [32.0, 24.0], atol=0.01)
    assert np.isclose(projected[1, 0], 32 + 80 * 20 / 300, atol=0.01)
    assert projected[:, 2].tolist() == [2, 2, 2, 2]
    assert np.asarray(second["keypoints"]).reshape(-1, 3)[:, 2].tolist() == [1, 2, 1, 1]
    assert np.allclose(np.asarray(first["object_pose"])[:3, 3], [0.0, 0.0, 0.3])
    intr = geometry.Intrinsics(WIDTH, HEIGHT, 80.0, 80.0, 32.0, 24.0)
    meters = np.asarray(json.loads((out / "keypoints_3d.json").read_text())["keypoints_3d"])
    uv, depth = geometry.project_points(
        geometry.transform_points(np.asarray(first["object_pose"]), meters),
        np.asarray(first["camera_extrinsics"]),
        intr,
    )
    assert np.allclose(uv, projected[:, :2], atol=0.01)
    assert np.allclose(depth, [0.3, 0.3, 0.3, 0.28])
    provenance = json.loads((out / "conversion_provenance.json").read_text())
    assert provenance["selection"] == {
        "requested_scenes": None,
        "selected_scenes": ["000003"],
        "max_frames": 0,
    }
    assert provenance["emitted_frames"][0]["target_instances"] == [0, 2]
    assert provenance["emitted_frames"][0]["rgb"] == "000003/rgb/000017.jpg"
    assert set(provenance["source_metadata"]["000003"]) == {"scene_gt.json", "scene_camera.json"}


def test_scene_filter_frame_limit_symlink_and_model_offset(tmp_path: Path) -> None:
    keypoints = write_keypoints(tmp_path / "kp.json", points=[[0, 0, 0]])
    pbr = tmp_path / "train_pbr"
    for name in ("000000", "000001"):
        write_scene(pbr, name, {f: [(2, np.eye(3), (0.0, 0.0, 300.0))] for f in (4, 9, 12)})
    out = tmp_path / "out"
    _run(pbr, keypoints, out, scenes=["000001"], max_frames=2, model_offset_mm=(10, 0, 0))
    payload = json.loads((out / "annotations.json").read_text())
    assert [i["file_name"] for i in payload["images"]] == ["000001_000004.jpg", "000001_000009.jpg"]
    assert (out / "images" / "000001_000004.jpg").is_symlink()
    assert json.loads((out / "keypoints_3d.json").read_text())["keypoints_3d"] == [[0.01, 0.0, 0.0]]
    assert np.asarray(payload["annotations"][0]["keypoints"])[0] == pytest.approx(34.67)


@pytest.mark.parametrize("missing", ["rgb", "depth", "mask"])
def test_missing_required_target_input_aborts_without_publication(tmp_path, missing) -> None:
    pbr, keypoints, scene = _single(tmp_path, ONE_FRAME)
    target = {
        "rgb": scene / "rgb" / "000000.jpg",
        "depth": scene / "depth" / "000000.png",
        "mask": scene / "mask_visib" / "000000_000000.png",
    }[missing]
    target.unlink()
    with pytest.raises((FileNotFoundError, ValueError), match=missing):
        _run(pbr, keypoints, tmp_path / "out")
    assert not os.path.lexists(tmp_path / "out")
    assert not list(tmp_path.glob(".out.tmp-*"))


def test_failure_after_a_complete_frame_still_publishes_nothing(tmp_path: Path) -> None:
    frames = {0: ONE_FRAME[0], 1: ONE_FRAME[0]}
    pbr, keypoints, scene = _single(tmp_path, frames)
    (scene / "mask_visib" / "000001_000000.png").unlink()
    with pytest.raises(FileNotFoundError, match="mask"):
        _run(pbr, keypoints, tmp_path / "out")
    assert not os.path.lexists(tmp_path / "out")
    assert not list(tmp_path.glob(".out.tmp-*"))


@pytest.mark.parametrize("existing_kind", ["file", "directory", "symlink"])
def test_existing_output_is_never_modified(tmp_path: Path, existing_kind: str) -> None:
    pbr, keypoints, _ = _single(tmp_path, ONE_FRAME)
    out = tmp_path / "out"
    if existing_kind == "file":
        out.write_text("keep")
    elif existing_kind == "directory":
        out.mkdir()
        (out / "keep").write_text("keep")
    else:
        out.symlink_to(tmp_path / "missing-target")
    with pytest.raises(FileExistsError, match="output already exists"):
        _run(pbr, keypoints, out)
    if existing_kind == "file":
        assert out.read_text() == "keep"
    elif existing_kind == "directory":
        assert (out / "keep").read_text() == "keep"
    else:
        assert os.readlink(out) == str(tmp_path / "missing-target")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("keypoint_count", "num_keypoints"),
        ("calibration", "cam_K"),
        ("pose", "cam_R_m2c"),
        ("selection", "requested scene"),
        ("units", "units"),
        ("offset", "model-offset-mm"),
        ("min_visible", "min-visible-keypoints"),
    ],
)
def test_invalid_metadata_or_selection_is_rejected(
    tmp_path: Path, mutation: str, match: str
) -> None:
    pbr, keypoints, scene = _single(tmp_path, ONE_FRAME)
    kwargs: dict = {}
    if mutation == "keypoint_count":
        write_keypoints(keypoints, num_keypoints=99)
    elif mutation == "units":
        write_keypoints(keypoints, units="mm")
    elif mutation == "calibration":
        payload = json.loads((scene / "scene_camera.json").read_text())
        payload["0"]["cam_K"][0] = 0.0
        (scene / "scene_camera.json").write_text(json.dumps(payload))
    elif mutation == "pose":
        payload = json.loads((scene / "scene_gt.json").read_text())
        payload["0"][0]["cam_R_m2c"] = [1.0] * 9
        (scene / "scene_gt.json").write_text(json.dumps(payload))
    elif mutation == "offset":
        kwargs["model_offset_mm"] = (float("nan"), 0, 0)
    elif mutation == "min_visible":
        kwargs["min_visible_keypoints"] = 99
    else:
        kwargs["scenes"] = ["000099"]
    with pytest.raises(ValueError, match=match):
        _run(pbr, keypoints, tmp_path / "out", **kwargs)
    assert not os.path.lexists(tmp_path / "out")


def test_skewed_calibration_is_rejected_instead_of_ignored(tmp_path: Path) -> None:
    pbr, keypoints, scene = _single(tmp_path, ONE_FRAME)
    payload = json.loads((scene / "scene_camera.json").read_text())
    payload["0"]["cam_K"][1] = 0.25
    (scene / "scene_camera.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="zero skew"):
        _run(pbr, keypoints, tmp_path / "out")
    assert not os.path.lexists(tmp_path / "out")


def test_atomic_publication_does_not_replace_racing_target(tmp_path: Path) -> None:
    staged, target = tmp_path / "staged", tmp_path / "target"
    staged.mkdir()
    (staged / "new").write_text("new")
    target.mkdir()
    (target / "keep").write_text("keep")
    with pytest.raises(FileExistsError, match="output already exists"):
        convert._rename_noreplace(staged, target)
    assert (target / "keep").read_text() == "keep"
    assert (staged / "new").read_text() == "new"


def test_atomic_publication_fails_closed_without_renameat2(tmp_path, monkeypatch) -> None:
    class LibcWithoutRenameat2:
        pass

    monkeypatch.setattr(convert.ctypes, "CDLL", lambda *args, **kwargs: LibcWithoutRenameat2())
    staged = tmp_path / "staged"
    staged.mkdir()
    with pytest.raises(RuntimeError, match="atomic no-replace publication is unsupported"):
        convert._rename_noreplace(staged, tmp_path / "target")
    assert staged.is_dir()


def test_zero_emitted_frames_is_rejected(tmp_path: Path) -> None:
    pbr, keypoints, _ = _single(tmp_path, {0: [(7, np.eye(3), (0.0, 0.0, 300.0))]})
    with pytest.raises(ValueError, match="no frames"):
        _run(pbr, keypoints, tmp_path / "out")
    assert not os.path.lexists(tmp_path / "out")


def test_malformed_object_id_cannot_silently_remove_a_source_frame(tmp_path: Path) -> None:
    pbr, keypoints, scene = _single(tmp_path, {0: ONE_FRAME[0], 1: ONE_FRAME[0]})
    payload = json.loads((scene / "scene_gt.json").read_text())
    payload["1"][0]["obj_id"] = "2"
    (scene / "scene_gt.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="obj_id"):
        _run(pbr, keypoints, tmp_path / "out")
    assert not os.path.lexists(tmp_path / "out")


def test_default_discovery_rejects_incomplete_numeric_scene(tmp_path: Path) -> None:
    pbr, keypoints, _ = _single(tmp_path, ONE_FRAME)
    (pbr / "000001").mkdir()
    (pbr / "notes").mkdir()
    with pytest.raises(ValueError, match="numeric BOP scene.*scene_gt"):
        _run(pbr, keypoints, tmp_path / "out")
    assert not os.path.lexists(tmp_path / "out")


def test_zero_depth_keypoint_rejects_nonfinite_projection(tmp_path: Path) -> None:
    keypoints = write_keypoints(
        tmp_path / "kp.json", points=[[0, 0, 0], [0.02, 0, 0], [0, 0.02, 0], [0, 0, -0.3]]
    )
    pbr = tmp_path / "train_pbr"
    write_scene(pbr, "000004", {12: ONE_FRAME[0]})
    with pytest.raises(ValueError, match="non-finite projection for 000004 frame 12 instance 0"):
        _run(pbr, keypoints, tmp_path / "out")
    assert not os.path.lexists(tmp_path / "out")
    assert not list(tmp_path.glob(".out.tmp-*"))


@pytest.mark.parametrize(("input_name", "dtype"), [("depth", np.uint8), ("mask", np.uint16)])
def test_wrong_bop_image_dtype_aborts_without_publication(tmp_path, input_name, dtype) -> None:
    pbr, keypoints, scene = _single(tmp_path, ONE_FRAME)
    target = {
        "depth": scene / "depth" / "000000.png",
        "mask": scene / "mask_visib" / "000000_000000.png",
    }[input_name]
    assert cv2.imwrite(str(target), np.full((HEIGHT, WIDTH), 1, dtype=dtype))
    with pytest.raises(ValueError, match=rf"invalid {input_name}.*dtype"):
        _run(pbr, keypoints, tmp_path / "out")
    assert not os.path.lexists(tmp_path / "out")
    assert not list(tmp_path.glob(".out.tmp-*"))


def test_selector_release_json_with_units_is_accepted(tmp_path: Path) -> None:
    pbr, keypoints, _ = _single(tmp_path, ONE_FRAME)
    write_keypoints(keypoints, units="m", source_units="mm", method="hull_anchored")
    summary = _run(pbr, keypoints, tmp_path / "out")
    assert summary.images == 1
    out_payload = json.loads((summary.out / "keypoints_3d.json").read_text())
    assert set(out_payload) == {"object", "num_keypoints", "keypoint_names", "keypoints_3d"}


def test_convert_metadata_scales_translations_only() -> None:
    gt = {"0": [{"obj_id": 1, "cam_t_m2c": [0.1, 0.2, 0.3], "cam_R_m2c": [1] * 9}]}
    cams = {"0": {"cam_t_w2c": [1.0, 0.0, 0.0], "depth_scale": 0.1, "cam_K": [1.0]}}
    gt_mm, cams_mm = hot3d_source.convert_metadata(gt, cams)
    assert gt_mm["0"][0]["cam_t_m2c"] == pytest.approx([100.0, 200.0, 300.0])
    assert gt_mm["0"][0]["cam_R_m2c"] == [1] * 9
    assert cams_mm["0"]["cam_t_w2c"] == [1000.0, 0.0, 0.0]
    assert cams_mm["0"]["depth_scale"] == 0.1 and cams_mm["0"]["cam_K"] == [1.0]
    assert gt["0"][0]["cam_t_m2c"] == [0.1, 0.2, 0.3]
    with pytest.raises(ValueError, match="depth scale"):
        hot3d_source.convert_metadata(gt, {"0": {"depth_scale": 0}})
    with pytest.raises(ValueError, match="keys differ"):
        hot3d_source.convert_metadata(gt, {})
