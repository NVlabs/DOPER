# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from doper.evaluation import real
from doper.infer import Detection
from doper.recorded import list_color_frames

NUM_KEYPOINTS = 4
FRAME_SIZE = (32, 24)  # (width, height)


def write_recording(root: Path, n_frames: int) -> Path:
    """Write tiny frames plus intrinsics and keypoints files under ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        cv2.imwrite(
            str(root / f"{i:06d}_color.png"), np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), np.uint8)
        )
    block = {"fx": 30.0, "fy": 30.0, "ppx": 16.0, "ppy": 12.0}
    (root / "intrinsics.json").write_text(json.dumps({"color_intrinsics": block}))
    (root / "keypoints_3d.json").write_text(
        json.dumps({"keypoints_3d": np.zeros((NUM_KEYPOINTS, 3)).tolist()})
    )
    return root


def make_detection(
    angle_deg: float, depth_m: float, reproj: float = 1.5, kp_score: float = 0.9
) -> Detection:
    """Build a detection whose pose rotates ``angle_deg`` about z at ``depth_m``."""
    return Detection(
        bbox_xyxy=np.array([0.0, 0.0, 8.0, 8.0]),
        score=0.8,
        keypoints=np.zeros((NUM_KEYPOINTS, 2)),
        keypoint_scores=np.full(NUM_KEYPOINTS, kp_score),
        rvec=np.array([[0.0], [0.0], [math.radians(angle_deg)]]),
        tvec=np.array([[0.0], [0.0], [depth_m]]),
        reproj_error_px=reproj,
    )


def evaluate(frames, detections_per_call, monkeypatch, options=None):
    """Run ``evaluate_recording`` with ``infer_image`` replaying ``detections_per_call``."""
    replay = iter(detections_per_call)
    monkeypatch.setattr(real, "infer_image", lambda *args, **kwargs: next(replay))
    cfg = real.DoperConfig(
        data=real.DataConfig(root="."), model=real.ModelConfig(num_keypoints=NUM_KEYPOINTS)
    )
    return real.evaluate_recording(
        object(),
        cfg,
        "cpu",
        frames,
        np.zeros((NUM_KEYPOINTS, 3)),
        np.eye(3),
        np.zeros((5, 1)),
        options,
    )


def test_relative_rotation_deg_matches_known_angles():
    """The geodesic angle is symmetric, zero for equal rotations and clipped at 180 degrees."""
    z = lambda deg: np.array([0.0, 0.0, math.radians(deg)])  # noqa: E731
    assert real.relative_rotation_deg(z(10), z(10)) == pytest.approx(0.0, abs=1e-6)
    assert real.relative_rotation_deg(z(10), z(55)) == pytest.approx(45.0)
    assert real.relative_rotation_deg(z(55), z(10)) == pytest.approx(45.0)
    assert real.relative_rotation_deg(z(0), z(180)) == pytest.approx(180.0)


def test_real_scorecard_counts_frames_without_pose(tmp_path, monkeypatch):
    """A frame with no detection stays in every denominator instead of being dropped."""
    frames = list_color_frames(write_recording(tmp_path, 3))
    result = evaluate(frames, [[make_detection(0, 0.5)], [], [make_detection(5, 0.5)]], monkeypatch)
    card = result["scorecard"]
    assert card["frames"] == 3 and len(result["per_frame"]) == 3
    assert card["det_rate_pct"] == pytest.approx(200 / 3)
    assert card["pnp_rate_pct"] == pytest.approx(200 / 3)
    assert card["depth_plausible_pct_of_frames"] == pytest.approx(200 / 3)
    assert card["depth_plausible_pct_of_pnp"] == pytest.approx(100.0)
    assert card["temporal_pairs"] == 0  # the empty frame breaks adjacency
    assert result["per_frame"][1]["detected"] is False
    assert result["per_frame"][1]["n_kp_conf"] == 0


def test_real_scorecard_flags_rotation_and_depth_jumps(tmp_path, monkeypatch):
    """Adjacent poses beyond either threshold count as flips; steady poses do not."""
    frames = list_color_frames(write_recording(tmp_path, 4))
    dets = [
        [make_detection(0, 0.50)],
        [make_detection(5, 0.51)],
        [make_detection(120, 0.51)],
        [make_detection(120, 0.90)],
    ]
    card = evaluate(frames, dets, monkeypatch)["scorecard"]
    assert card["temporal_pairs"] == 3
    assert card["flip_jump_rate_pct"] == pytest.approx(200 / 3)
    assert card["rot_step_deg_median"] == pytest.approx(5.0)
    assert card["mean_kp_above_conf"] == NUM_KEYPOINTS
    assert card["reproj_px_median"] == pytest.approx(1.5)


def test_real_scorecard_without_detections_is_all_zero_and_null(tmp_path, monkeypatch):
    """No detections give zero rates and null medians, never an exception."""
    frames = list_color_frames(write_recording(tmp_path, 2))
    card = evaluate(frames, [[], []], monkeypatch)["scorecard"]
    assert card["det_rate_pct"] == card["pnp_rate_pct"] == card["flip_jump_rate_pct"] == 0.0
    assert card["reproj_px_median"] is None and card["kp_conf_median"] is None


def test_main_writes_strict_json_scorecard(tmp_path, monkeypatch, capsys):
    """The CLI writes a parseable scorecard (default path under the data dir) with null medians."""
    data_dir = write_recording(tmp_path / "rec", 2)
    monkeypatch.setattr(real, "load_model", lambda *args: object())
    monkeypatch.setattr(real, "infer_image", lambda *args, **kwargs: [])
    argv = [
        "--checkpoint",
        "ckpt.pth",
        "--data-dir",
        str(data_dir),
        "--keypoints-3d",
        str(data_dir / "keypoints_3d.json"),
        "--device",
        "cpu",
    ]
    assert real.main(argv) == 0

    def reject_constant(value):
        raise ValueError(f"non-standard JSON constant: {value}")

    payload = json.loads((data_dir / "real_eval.json").read_text(), parse_constant=reject_constant)
    assert payload["scorecard"]["frames"] == 2
    assert payload["scorecard"]["checkpoint"] == "ckpt.pth"
    assert payload["scorecard"]["reproj_px_median"] is None
    assert "real scorecard" in capsys.readouterr().out


def test_main_exits_when_recording_has_no_frames(tmp_path, monkeypatch):
    """An empty data directory is a usage error, not an empty scorecard."""
    data_dir = write_recording(tmp_path / "rec", 0)
    monkeypatch.setattr(real, "load_model", lambda *args: object())
    argv = [
        "--checkpoint",
        "ckpt.pth",
        "--data-dir",
        str(data_dir),
        "--keypoints-3d",
        str(data_dir / "keypoints_3d.json"),
        "--device",
        "cpu",
    ]
    with pytest.raises(SystemExit, match="no \\*_color.png frames"):
        real.main(argv)
