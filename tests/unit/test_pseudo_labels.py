# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pseudo-label exporter tests: output-directory safety, mix roots, multiview labels, masks."""

import json
import os
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from doper.infer import Detection
from doper.pseudo_labels import hdf5_episodes, masks, mix_root, recorded_boxes
from doper.pseudo_labels import multiview as multiview_export
from doper.pseudo_labels._outdir import check_output_dir, replace_output_dir

IMAGE_SIZE = (160, 120)  # (width, height)
FOCAL = {"A": 200.0, "B": 210.0}
OBJECT_POSE_A = (np.array([0.1, -0.2, 0.05]), np.array([0.0, 0.0, 0.5]))
RIG_YAW_RAD = np.deg2rad(5.0)
RIG_TRANSLATION = np.array([0.02, 0.0, 0.0])
N_KEYPOINTS = 12


def _keypoints_3d() -> np.ndarray:
    return np.random.default_rng(0).uniform(-0.03, 0.03, size=(N_KEYPOINTS, 3))


def _camera(view: str) -> np.ndarray:
    width, height = IMAGE_SIZE
    focal = FOCAL[view]
    return np.array([[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]])


def _pose(view: str) -> tuple[np.ndarray, np.ndarray]:
    """Object pose in ``view``; the rig maps camera B points into camera A (R_AB, t_AB)."""
    rvec, tvec = OBJECT_POSE_A
    if view == "A":
        return rvec, tvec
    rotation_ab = cv2.Rodrigues(np.array([0.0, RIG_YAW_RAD, 0.0]))[0]
    rotation = rotation_ab.T @ cv2.Rodrigues(rvec)[0]
    translation = rotation_ab.T @ (tvec - RIG_TRANSLATION)
    return cv2.Rodrigues(rotation)[0].ravel(), translation


def _write_camera(directory: Path, view: str, n_frames: int, with_ir: bool) -> None:
    directory.mkdir(parents=True)
    width, height = IMAGE_SIZE
    for i in range(n_frames):
        cv2.imwrite(
            str(directory / f"{i:06d}_color.png"), np.full((height, width, 3), 60, np.uint8)
        )
        if with_ir:
            cv2.imwrite(
                str(directory / f"{i:06d}_ir_left.png"), np.full((height, width), 90, np.uint8)
            )
    camera = _camera(view)
    block = {"fx": camera[0, 0], "fy": camera[1, 1], "ppx": camera[0, 2], "ppy": camera[1, 2]}
    payload = {"color_intrinsics": block}
    if with_ir:
        payload["ir_left_intrinsics"] = block
        payload["T_color_irleft"] = np.eye(4).tolist()
    (directory / "intrinsics.json").write_text(json.dumps(payload))


def _detection(view: str, keypoints_3d: np.ndarray) -> Detection:
    rvec, tvec = _pose(view)
    points = cv2.projectPoints(keypoints_3d, rvec, tvec, _camera(view), np.zeros(5))[0]
    points = points.reshape(-1, 2).astype(np.float32)
    return Detection(
        bbox_xyxy=np.array([*points.min(axis=0), *points.max(axis=0)], dtype=np.float32),
        score=0.9,
        keypoints=points,
        keypoint_scores=np.full(len(points), 0.95, dtype=np.float32),
        rvec=np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        tvec=np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        reproj_error_px=0.1,
    )


def _multiview_inputs(root: Path) -> list[str]:
    """Write a two-camera capture (camera A with left-IR) and return the exporter arguments."""
    keypoints_3d = _keypoints_3d()
    _write_camera(root / "cam_a", "A", 2, with_ir=True)
    _write_camera(root / "cam_b", "B", 2, with_ir=False)
    rotation_ab = cv2.Rodrigues(np.array([0.0, RIG_YAW_RAD, 0.0]))[0]
    rig = {"R_AB": rotation_ab.tolist(), "t_AB": RIG_TRANSLATION.tolist()}
    (root / "rig.json").write_text(json.dumps(rig))
    payload = {"keypoints_3d": keypoints_3d.tolist(), "object": "widget"}
    (root / "keypoints_3d.json").write_text(json.dumps(payload))
    return [
        "--cam-a-dir", str(root / "cam_a"), "--cam-b-dir", str(root / "cam_b"),
        "--extrinsic", str(root / "rig.json"), "--checkpoint", "unused.pth",
        "--keypoints-3d", str(root / "keypoints_3d.json"), "--out-dir", str(root / "out"),
        "--device", "cpu",
    ]  # fmt: skip


def _stub_multiview(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the model with a detector that reports the true keypoints of each camera."""
    keypoints_3d = _keypoints_3d()

    def fake_infer_all(model, image, cfg, device, settings, view, points):
        label = "A" if view.camera_matrix[0, 0] == FOCAL["A"] else "B"
        return [_detection(label, keypoints_3d)]

    monkeypatch.setattr(multiview_export, "load_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(multiview_export, "infer_all", fake_infer_all)


@pytest.fixture
def stubbed_multiview(monkeypatch):
    """Multiview exporter with ``_stub_multiview`` applied."""
    _stub_multiview(monkeypatch)


def test_multiview_export_marks_pseudo_label(tmp_path, stubbed_multiview):
    argv = _multiview_inputs(tmp_path)
    code = multiview_export.main([*argv, "--filled-projected-box-mask"])
    out = tmp_path / "out"
    coco = json.loads((out / "annotations.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    assert code == 0
    assert summary["frames_accepted"] == 2
    assert (summary["color_images"], summary["ir_left_images"]) == (4, 2)
    assert [(i["camera_label"], i["modality"]) for i in coco["images"][:3]] == [
        ("A", "color"),
        ("A", "ir_left"),
        ("B", "color"),
    ]
    assert all(a["pseudo_label"] for a in coco["annotations"])
    assert all(a["num_keypoints"] == N_KEYPOINTS for a in coco["annotations"])
    assert all((out / a["mask_file"]).is_file() for a in coco["annotations"])
    assert coco["categories"][0]["name"] == "widget"
    color_a = coco["annotations"][0]
    expected = _detection("A", _keypoints_3d()).keypoints
    actual = np.asarray(color_a["keypoints"]).reshape(-1, 3)
    np.testing.assert_allclose(actual[:, :2], expected, atol=1e-2)
    assert sorted(p.name for p in (out / "overlays").iterdir()) == ["000000.jpg", "000001.jpg"]


def test_multiview_export_skips_pairs_failing_the_reprojection_gate(tmp_path, monkeypatch):
    keypoints_3d = _keypoints_3d()

    def disagreeing_infer_all(model, image, cfg, device, settings, view, points):
        detection = _detection("A", keypoints_3d)  # camera A keypoints reported in both views
        return [detection]

    monkeypatch.setattr(multiview_export, "load_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(multiview_export, "infer_all", disagreeing_infer_all)
    multiview_export.main(_multiview_inputs(tmp_path))
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert (summary["frames_total"], summary["frames_accepted"]) == (2, 0)
    assert json.loads((tmp_path / "out" / "annotations.json").read_text())["annotations"] == []


def _flat_export(root: Path) -> Path:
    """Write a flat export with one color image, one IR image and one mask."""
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    for name in ("A_000000.png", "A_000000_ir_left.png"):
        cv2.imwrite(str(root / "images" / name), np.zeros((4, 4, 3), np.uint8))
    cv2.imwrite(str(root / "masks" / "A_000000_mask.png"), np.zeros((4, 4), np.uint8))
    coco = {
        "info": {"description": "flat"},
        "images": [
            {"id": 0, "file_name": "A_000000.png", "modality": "color"},
            {"id": 1, "file_name": "A_000000_ir_left.png", "modality": "ir_left"},
        ],
        "annotations": [
            {"id": 0, "image_id": 0, "bbox": [0, 0, 2, 2], "keypoints": [1, 1, 2],
             "rvec": [0, 0, 0], "mask_file": "masks/A_000000_mask.png"},
            {"id": 1, "image_id": 1, "bbox": [0, 0, 2, 2], "keypoints": [1, 1, 2]},
        ],
        "categories": [{"id": 1, "name": "object"}],
    }  # fmt: skip
    (root / "annotations.json").write_text(json.dumps(coco))
    return root


def test_mix_root_symlinks_images_without_copying(tmp_path):
    src = _flat_export(tmp_path / "flat")
    code = mix_root.main(["--src", str(src), "--dst", str(tmp_path / "mix"), "--mask-only"])
    split = tmp_path / "mix" / "train"
    linked = sorted((split / "images").iterdir())
    mask = split / "masks" / "A_000000_mask.png"
    coco = json.loads((split / "annotations.json").read_text())
    assert code == 0
    assert [p.name for p in linked] == ["A_000000.png", "A_000000_ir_left.png"]
    assert all(p.is_symlink() for p in [*linked, mask])
    assert all(p.resolve() == (src / "images" / p.name).resolve() for p in linked)
    assert mask.resolve() == (src / "masks" / mask.name).resolve()
    assert coco["info"]["mask_only"] is True
    assert all("keypoints" not in a and "rvec" not in a for a in coco["annotations"])
    assert coco["annotations"][0]["bbox"] == [0, 0, 2, 2]


def test_mix_root_force_replaces_only_the_split(tmp_path):
    src = _flat_export(tmp_path / "flat")
    split = tmp_path / "mix" / "train"
    split.mkdir(parents=True)
    (split / "stale.txt").write_text("old")
    (tmp_path / "mix" / "other.txt").write_text("keep")
    mix_root.main(["--src", str(src), "--dst", str(tmp_path / "mix"), "--force"])
    assert not (split / "stale.txt").exists()
    assert (tmp_path / "mix" / "other.txt").exists()
    assert "keypoints" in json.loads((split / "annotations.json").read_text())["annotations"][0]


HDF5_INTRINSICS = ["--fx", "200", "--fy", "200", "--cx", "80", "--cy", "60"]


def _refusal_argv(name: str, tmp_path: Path) -> tuple[list[str], Path]:
    """Arguments that point an exporter at an existing output directory, and that directory."""
    if name == "multiview":
        out = tmp_path / "out"
        argv = [*_multiview_inputs(tmp_path)]
        return argv, out
    if name == "recorded_boxes":
        out = tmp_path / "out"
        argv = ["--detections-json", "d.json", "--cropped-color-dir", "c", "--source-ir-dir",
                "i", "--out-root", str(out)]  # fmt: skip
        return argv, out
    if name == "hdf5_episodes":
        out = tmp_path / "out"
        (tmp_path / "split.json").write_text(json.dumps({"train": [0]}))
        argv = ["--dataset-root", str(tmp_path / "episodes"),
                "--split-json", str(tmp_path / "split.json"),
                "--checkpoint", "unused.pth", "--keypoints-3d", "k.json",
                "--out-root", str(out), *HDF5_INTRINSICS]  # fmt: skip
        return argv, out
    src = _flat_export(tmp_path / "flat")
    return ["--src", str(src), "--dst", str(tmp_path / "mix")], tmp_path / "mix" / "train"


EXPORTERS = {
    "multiview": multiview_export,
    "recorded_boxes": recorded_boxes,
    "hdf5_episodes": hdf5_episodes,
    "mix_root": mix_root,
}


@pytest.mark.parametrize("name", list(EXPORTERS))
def test_exporter_refuses_existing_out_dir_without_force(tmp_path, monkeypatch, name):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("an exporter must refuse before loading a model")

    for module in (multiview_export, hdf5_episodes):
        monkeypatch.setattr(module, "load_model", fail_if_called)
    argv, out = _refusal_argv(name, tmp_path)
    out.mkdir(parents=True)
    (out / "earlier_label.json").write_text("keep")
    with pytest.raises(SystemExit, match="exists") as info:
        EXPORTERS[name].main(argv)
    assert info.value.code not in (0, None)
    assert (out / "earlier_label.json").read_text() == "keep"
    assert sorted(os.listdir(out)) == ["earlier_label.json"]


def test_hdf5_missing_extra_names_install_hint(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "h5py", None)
    argv, _ = _refusal_argv("hdf5_episodes", tmp_path)
    with pytest.raises(ImportError, match=r"doper\[hdf5\]"):
        hdf5_episodes.main(argv)
    assert not (tmp_path / "out").exists()


def _recorded_inputs(tmp_path: Path) -> tuple[list[str], np.ndarray]:
    """Write a cropped color frame, its full-size IR frame and reviewed boxes; return argv, IR."""
    color_dir = tmp_path / "cropped"
    ir_dir = tmp_path / "ir"
    color_dir.mkdir()
    ir_dir.mkdir()
    cv2.imwrite(str(color_dir / "000003_color.png"), np.zeros((6, 8, 3), np.uint8))
    ir = np.arange(10 * 12, dtype=np.uint8).reshape(10, 12)
    cv2.imwrite(str(ir_dir / "000003_ir_left.png"), ir)
    (tmp_path / "crop.json").write_text(
        json.dumps({"crop_xyxy": [2, 1, 10, 7], "output_size": [8, 6]})
    )
    records = [
        {"idx": 3, "frame": "000003_color.png", "detected": True, "bbox_xyxy": [-2, 1, 5, 9],
         "score": 0.75},
        {"idx": 4, "frame": "000004_color.png", "detected": False, "bbox_xyxy": None},
    ]  # fmt: skip
    (tmp_path / "detections.json").write_text(json.dumps(records))
    argv = [
        "--detections-json", str(tmp_path / "detections.json"), "--cropped-color-dir",
        str(color_dir), "--source-ir-dir", str(ir_dir), "--crop-json",
        str(tmp_path / "crop.json"), "--out-root", str(tmp_path / "out"),
    ]  # fmt: skip
    return argv, ir


def test_recorded_boxes_transfer_clipped_box_to_ir_crop(tmp_path):
    argv, ir = _recorded_inputs(tmp_path)
    out = tmp_path / "out"
    code = recorded_boxes.main(argv)
    images_dir = out / "train" / "images"
    coco = json.loads((out / "train" / "annotations.json").read_text())
    assert code == 0
    assert (images_dir / "000003_color.png").is_symlink()
    np.testing.assert_array_equal(
        cv2.imread(str(images_dir / "000003_ir_left.png"), cv2.IMREAD_GRAYSCALE), ir[1:7, 2:10]
    )
    assert [a["bbox"] for a in coco["annotations"]] == [[0.0, 1.0, 5.0, 5.0]] * 2
    assert [a["modality"] for a in coco["annotations"]] == ["color", "ir_left"]
    assert all(a["pseudo_label"] and a["detector_score"] == 0.75 for a in coco["annotations"])


def test_color_pose_is_transformed_before_ir_projection():
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = [0.1, -0.2, 0.0]
    rvec_ir, tvec_ir = multiview_export.color_pose_to_ir_left(
        np.zeros((3, 1)), np.asarray([[0.0], [0.0], [1.0]]), transform
    )
    np.testing.assert_allclose(rvec_ir, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(tvec_ir.ravel(), [0.1, -0.2, 1.0])
    projected, _ = cv2.projectPoints(
        np.zeros((1, 3)), rvec_ir, tvec_ir, np.diag([100.0, 100.0, 1.0]), np.zeros(5)
    )
    np.testing.assert_allclose(projected.reshape(2), [10.0, -20.0])


def test_projected_to_coco_marks_outside_and_uncertain_keypoints():
    projected = np.array([[1.0, 2.0], [50.0, 2.0], [np.nan, 1.0], [3.0, 4.0]])
    flat, visible = multiview_export.projected_to_coco(
        projected, np.array([0.9, 0.9, 0.9, 0.1]), 10, 10, 0.3
    )
    assert flat == [1.0, 2.0, 2, 0.0, 0.0, 0, 0.0, 0.0, 0, 3.0, 4.0, 1]
    assert visible == 2


def test_bbox_from_points_pads_and_clips_to_the_image():
    points = np.array([[2.0, 3.0], [6.0, 5.0], [100.0, 1.0], [np.inf, 2.0]])
    assert masks.bbox_from_points(points, 8, 8, 0.5) == [0.0, 2.0, 8.0, 4.0]
    assert masks.bbox_from_points(np.array([[-1.0, -1.0]]), 8, 8, 0.0) == [0.0, 0.0, 0.0, 0.0]


def test_projected_cuboid_mask_fills_the_convex_hull():
    corners = np.array([[2.0, 2.0], [7.0, 2.0], [7.0, 6.0], [2.0, 6.0], [np.nan, 0.0]])
    mask = masks.projected_cuboid_mask(corners, 10, 8)
    assert mask[4, 4] == masks.MASK_FOREGROUND
    assert mask[0, 0] == 0
    assert masks.bbox_from_mask(mask, 0.0) == [2.0, 2.0, 6.0, 5.0]


def _write_stl(path: Path, triangles: np.ndarray) -> None:
    with open(path, "wb") as handle:
        handle.write(b"\0" * masks.STL_HEADER_BYTES)
        handle.write(struct.pack("<I", len(triangles)))
        for triangle in triangles:
            handle.write(struct.pack("<12fH", 0, 0, 0, *triangle.ravel(), 0))


def test_mesh_mask_rasterizes_a_binary_stl(tmp_path):
    square = np.array(
        [[[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.1, 0.1, 0.0]],
         [[-0.1, -0.1, 0.0], [0.1, 0.1, 0.0], [-0.1, 0.1, 0.0]]]
    )  # fmt: skip
    _write_stl(tmp_path / "square.stl", square)
    triangles = masks.load_binary_stl_triangles(tmp_path / "square.stl")
    camera = np.array([[50.0, 0.0, 20.0], [0.0, 50.0, 20.0], [0.0, 0.0, 1.0]])
    mask = masks.projected_mesh_mask(
        triangles, np.zeros(3), np.array([0.0, 0.0, 1.0]), camera, np.zeros(5), 40, 40
    )
    np.testing.assert_allclose(triangles, square, atol=1e-7)
    assert masks.bbox_from_mask(mask, 0.0) == [15.0, 15.0, 11.0, 11.0]
    behind = masks.projected_mesh_mask(
        triangles, np.zeros(3), np.array([0.0, 0.0, -1.0]), camera, np.zeros(5), 40, 40
    )
    assert not behind.any()


def test_truncated_stl_is_rejected(tmp_path):
    path = tmp_path / "short.stl"
    path.write_bytes(b"\0" * masks.STL_HEADER_BYTES + struct.pack("<I", 2))
    with pytest.raises(ValueError, match="expected 2"):
        masks.load_binary_stl_triangles(path)


def _write_episode(path: Path, blank_right_frame: int) -> None:
    import h5py

    width, height = IMAGE_SIZE
    frames = np.full((2, height, width, 3), 80, np.uint8)
    right = frames.copy()
    right[blank_right_frame] = 0
    with h5py.File(path, "w") as hdf5:
        hdf5.create_dataset("obs/images/wrist_left", data=frames)
        hdf5.create_dataset("obs/images/wrist_right", data=right)


def _hdf5_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stub the model, write two episodes (train 0, validation 1) and return the arguments."""
    keypoints_3d = _keypoints_3d()
    camera = _camera("A")

    def fake_infer_image(model, image, cfg, device, **kwargs):
        return [_detection("A", keypoints_3d)] if image.any() else []

    monkeypatch.setattr(hdf5_episodes, "load_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(hdf5_episodes, "infer_image", fake_infer_image)
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    for episode in (0, 1):
        _write_episode(episodes / f"episode_{episode}.hdf5", blank_right_frame=episode)
    (tmp_path / "split.json").write_text(json.dumps({"train": [0], "validation": [1]}))
    (tmp_path / "kp.json").write_text(json.dumps({"keypoints_3d": keypoints_3d.tolist()}))
    return [
        "--dataset-root", str(episodes), "--split-json", str(tmp_path / "split.json"),
        "--checkpoint", "unused.pth", "--keypoints-3d", str(tmp_path / "kp.json"),
        "--out-root", str(tmp_path / "out"), "--eval-splits", "train,validation",
        "--min-inliers", "3", "--fx", str(camera[0, 0]), "--fy", str(camera[1, 1]),
        "--cx", str(camera[0, 2]), "--cy", str(camera[1, 2]), "--device", "cpu",
    ]  # fmt: skip


def test_hdf5_episodes_label_only_the_label_splits(tmp_path, monkeypatch):
    out = tmp_path / "out"
    code = hdf5_episodes.main(_hdf5_inputs(tmp_path, monkeypatch))
    metrics = json.loads((out / "metrics.json").read_text())
    coco = json.loads((out / "train" / "annotations.json").read_text())
    assert code == 0
    assert metrics["splits"]["validation"]["episodes"]["1"]["wrist_right"]["accepted"] == 1
    assert metrics["splits"]["train"]["episodes"]["0"]["wrist_right"]["accepted_frame_ids"] == [1]
    assert [(i["camera_label"], i["frame_id"]) for i in coco["images"]] == [
        ("wrist_left", 0),
        ("wrist_left", 1),
        ("wrist_right", 1),
    ]
    assert all(
        a["episode"] == 0 and a["ransac_inliers"] == N_KEYPOINTS for a in coco["annotations"]
    )
    assert all((out / "train" / a["mask_file"]).is_file() for a in coco["annotations"])


def test_hdf5_episodes_reject_label_splits_outside_eval_splits(tmp_path):
    argv, out = _refusal_argv("hdf5_episodes", tmp_path)
    with pytest.raises(SystemExit, match="subset"):
        hdf5_episodes.main([*argv, "--eval-splits", "validation"])
    assert not out.exists()


def test_hdf5_episodes_require_intrinsics(tmp_path, capsys):
    argv, _ = _refusal_argv("hdf5_episodes", tmp_path)
    without_fx = [a for a in argv if a not in ("--fx", "200")]
    with pytest.raises(SystemExit) as info:
        hdf5_episodes.main(without_fx)
    assert info.value.code == 2
    assert "--fx" in capsys.readouterr().err


def test_hdf5_episodes_read_the_named_cameras(tmp_path, monkeypatch):
    argv = _hdf5_inputs(tmp_path, monkeypatch)
    hdf5_episodes.main([*argv, "--cameras", "wrist_right"])
    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert list(metrics["splits"]["train"]["episodes"]["0"]) == ["wrist_right"]


def _force_inputs(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Valid arguments of an exporter whose output directory is ``tmp_path / "out"``."""
    if name == "multiview":
        _stub_multiview(monkeypatch)
        return _multiview_inputs(tmp_path)
    if name == "recorded_boxes":
        return _recorded_inputs(tmp_path)[0]
    return _hdf5_inputs(tmp_path, monkeypatch)


@pytest.mark.parametrize("name", ["multiview", "recorded_boxes", "hdf5_episodes"])
def test_exporter_force_replaces_existing_out_dir(tmp_path, monkeypatch, name):
    argv = _force_inputs(name, tmp_path, monkeypatch)
    out = tmp_path / "out"
    out.mkdir()
    (out / "earlier_label.json").write_text("stale")
    code = EXPORTERS[name].main([*argv, "--force"])
    assert code == 0
    assert not (out / "earlier_label.json").exists()
    assert list(out.rglob("annotations.json"))


@pytest.mark.parametrize("relation", ["equal", "contains", "inside"])
def test_output_dir_overlapping_an_input_is_refused(tmp_path, relation):
    data = tmp_path / "data"
    (data / "sub").mkdir(parents=True)
    out = {"equal": data, "contains": tmp_path, "inside": data / "sub"}[relation]
    with pytest.raises(SystemExit, match="overlaps input"):
        replace_output_dir(out, True, [None, data])
    assert (data / "sub").is_dir()


def test_output_dir_refuses_existing_without_force_and_replaces_with_it(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "stale").write_text("x")
    with pytest.raises(SystemExit, match="exists; pass --force"):
        check_output_dir(out, False, [tmp_path / "input.json"])
    replace_output_dir(out, True, [tmp_path / "input.json"])
    assert not out.exists()


def test_mix_root_force_refuses_to_replace_its_own_source(tmp_path):
    src = _flat_export(tmp_path / "mix" / "train")
    with pytest.raises(SystemExit, match="overlaps input"):
        mix_root.main(["--src", str(src), "--dst", str(tmp_path / "mix"), "--force"])
    assert (src / "annotations.json").is_file()
    assert (src / "images" / "A_000000.png").is_file()


def test_recorded_boxes_force_refuses_out_root_holding_inputs(tmp_path):
    argv, _ = _recorded_inputs(tmp_path)
    with pytest.raises(SystemExit, match="overlaps input"):
        recorded_boxes.main(
            [*argv[: argv.index("--out-root")], "--out-root", str(tmp_path), "--force"]
        )
    assert any(tmp_path.iterdir())
