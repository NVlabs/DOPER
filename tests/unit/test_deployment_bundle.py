# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployment bundle checksums and pose-parity gate."""

import json
import sys

import cv2
import numpy as np
import pytest
from tiny_coco import write_deployment_inputs

from doper.deployment import bundle, parity
from doper.deployment.onnx_export import export_bundle, main


def _pose(box, tvec=(0.0, 0.0, 0.5), rvec=(0.1, 0.2, 0.3)):
    return {
        "box": list(box),
        "score": 0.9,
        "keypoints": [[1.0, 2.0], [3.0, 4.0]],
        "keypoint_scores": [0.5, 0.5],
        "rvec": list(rvec),
        "tvec": list(tvec),
    }


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    return write_deployment_inputs(tmp_path_factory.mktemp("inputs"))


def _export(inputs, output):
    return export_bundle(
        inputs["checkpoint"],
        inputs["keypoints_3d"],
        inputs["camera"],
        inputs["frames"],
        output,
        score_thr=0.0,
    )


def test_bundle_rejects_modified_file(tmp_path):
    (tmp_path / "frame000.png").write_bytes(b"frame")
    (tmp_path / "det.onnx").write_bytes(b"graph")
    bundle.write_manifest(tmp_path, {"name": "tiny"})
    assert bundle.verify(tmp_path)["name"] == "tiny"
    (tmp_path / "det.onnx").write_bytes(b"edited")
    with pytest.raises(ValueError, match="Bundle checksum mismatch: det.onnx"):
        bundle.verify(tmp_path)


def test_bundle_rejects_missing_and_unrecorded_files(tmp_path):
    (tmp_path / "frame000.png").write_bytes(b"frame")
    bundle.write_manifest(tmp_path, {})
    (tmp_path / "frame001.png").write_bytes(b"extra")
    with pytest.raises(ValueError, match="unrecorded file: frame001.png"):
        bundle.verify(tmp_path)
    (tmp_path / "frame001.png").unlink()
    (tmp_path / "frame000.png").unlink()
    with pytest.raises(ValueError, match="Bundle checksum mismatch: frame000.png"):
        bundle.verify(tmp_path)


def test_pose_parity_matches_identical():
    reference = [[_pose([0, 0, 10, 10]), _pose([20, 20, 30, 30], tvec=(0.1, 0, 1))], []]
    result = parity.pose_parity(reference, json.loads(json.dumps(reference)))
    assert result["frames_with_detection_count_difference"] == 0
    assert result["matched_detections"] == 2
    assert result["pnp_validity_differences"] == 0
    assert result["translation_difference_m_max"] == 0.0
    assert result["rotation_difference_deg_max"] == pytest.approx(0.0, abs=1e-4)
    assert parity.pose_gate_passed(result)


def test_pose_gate_rejects_count_mismatch():
    reference = [[_pose([0, 0, 10, 10])], [_pose([0, 0, 10, 10])]]
    actual = [[_pose([0, 0, 10, 10])], []]
    result = parity.pose_parity(actual, reference)
    assert result["frames_with_detection_count_difference"] == 1
    assert not parity.pose_gate_passed(result)


def test_pose_gate_rejects_large_pose_difference():
    reference = [[_pose([0, 0, 10, 10])]]
    moved = [[_pose([0, 0, 10, 10], tvec=(0.0, 0.0, 0.502))]]
    assert not parity.pose_gate_passed(parity.pose_parity(moved, reference))
    rotated = [[_pose([0, 0, 10, 10], rvec=(0.1, 0.2, 0.35))]]
    assert not parity.pose_gate_passed(parity.pose_parity(rotated, reference))
    assert not parity.pose_gate_passed({})


def test_compare_outputs_and_summarize():
    reference = np.ones((2, 3))
    result = parity.compare_outputs(reference + 0.5, reference)
    assert result["max_abs"] == pytest.approx(0.5)
    assert result["relative_l2"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="equal, nonempty, finite"):
        parity.compare_outputs(np.ones(2), np.ones(3))
    stats = parity.summarize([1.0, 2.0, 3.0])
    assert stats["n"] == 3 and stats["median_ms"] == 2.0
    with pytest.raises(ValueError, match="finite positive"):
        parity.summarize([1.0, 0.0])


def test_export_bundle_round_trip(inputs, tmp_path):
    output = _export(inputs, tmp_path / "bundle")
    manifest = bundle.verify(output)
    names = set(manifest["files"])
    expected = {"det.onnx", "crop.onnx", "reference_poses.json", "config.json", "camera.json"}
    assert expected | {"keypoints.json", "frame000.png", "frame001.png"} <= names
    assert {"ref000_det.npz", "ref001_det.npz"} <= names
    assert any(name.endswith("_crop.npz") for name in names)
    poses = json.loads((output / "reference_poses.json").read_text())
    assert len(poses) == 2 and all(len(frame) >= 1 for frame in poses)
    assert manifest["detector_size"] == 96 and manifest["crop_size"] == 32
    assert "/" not in manifest["checkpoint"]


def test_export_bundle_reuses_matching_cache_and_refuses_other(inputs, tmp_path):
    output = _export(inputs, tmp_path / "bundle")
    before = (output / "manifest.json").read_text()
    assert _export(inputs, output) == output
    assert (output / "manifest.json").read_text() == before
    (output / "frame000.png").write_bytes(b"edited")
    with pytest.raises(ValueError, match="Incompatible export cache"):
        _export(inputs, output)
    stray = tmp_path / "stray"
    stray.mkdir()
    (stray / "notes.txt").write_text("x")
    with pytest.raises(ValueError, match="not empty"):
        _export(inputs, stray)


def test_export_bundle_names_extra_without_onnx(inputs, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "onnx", None)
    with pytest.raises(ImportError, match=r"doper\[tensorrt\]"):
        _export(inputs, tmp_path / "bundle")


def test_cli_exports_from_frames_dir_and_video(inputs, tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i, frame in enumerate(inputs["frames"]):
        assert cv2.imwrite(str(frames_dir / f"{i:04d}.png"), frame)
    video = tmp_path / "take.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (128, 96))
    for _ in range(4):
        writer.write(inputs["frames"][0])
    writer.release()
    common = [
        "--checkpoint",
        str(inputs["checkpoint"]),
        "--keypoints-3d",
        str(inputs["keypoints_3d"]),
        "--camera",
        str(inputs["camera"]),
        "--score-thr",
        "0",
    ]
    assert main([*common, "--frames-dir", str(frames_dir), "--output", str(tmp_path / "a")]) == 0
    assert len(bundle.verify(tmp_path / "a")["frames_sha256"]) == 2
    args = [*common, "--video", str(video), "--frames", "3", "--output", str(tmp_path / "b")]
    assert main(args) == 0
    assert len(bundle.verify(tmp_path / "b")["frames_sha256"]) == 3
    with pytest.raises(ValueError, match="Not enough source frames"):
        main([*common, "--video", str(video), "--frames", "9", "--output", str(tmp_path / "c")])
