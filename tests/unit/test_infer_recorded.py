# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from doper import infer_recorded
from doper.infer import Detection

NUM_KEYPOINTS = 4
FRAME_SIZE = (32, 24)  # (width, height)
N_FRAMES = 3


def write_recording(root: Path) -> Path:
    """Write tiny frames plus an intrinsics file under ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    for i in range(N_FRAMES):
        cv2.imwrite(
            str(root / f"{i:06d}_color.png"), np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), np.uint8)
        )
    block = {"fx": 30.0, "fy": 30.0, "ppx": 16.0, "ppy": 12.0}
    (root / "intrinsics.json").write_text(json.dumps({"color_intrinsics": block}))
    return root


def fake_detection() -> Detection:
    """Return a posed detection."""
    return Detection(
        bbox_xyxy=np.array([2.0, 2.0, 20.0, 18.0]),
        score=0.8,
        keypoints=np.full((NUM_KEYPOINTS, 2), 8.0),
        keypoint_scores=np.full(NUM_KEYPOINTS, 0.9),
        rvec=np.zeros((3, 1)),
        tvec=np.array([[0.0], [0.0], [0.5]]),
        reproj_error_px=1.5,
    )


@pytest.fixture
def stubbed(monkeypatch):
    """Replace the model loader and detector with stubs."""
    monkeypatch.setattr(infer_recorded, "load_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        infer_recorded, "infer_image", lambda *args, **kwargs: ([fake_detection()], None)
    )


def run(tmp_path: Path, *extra: str) -> tuple[Path, int]:
    """Run ``main`` on a fresh recording and return the data dir and exit code."""
    data_dir = write_recording(tmp_path / "rec")
    code = infer_recorded.main(
        ["--checkpoint", "unused.pth", "--data-dir", str(data_dir), "--device", "cpu", *extra]
    )
    return data_dir, code


def test_requires_checkpoint(tmp_path):
    with pytest.raises(SystemExit) as info:
        infer_recorded.main(["--data-dir", str(tmp_path)])
    assert info.value.code == 2


def test_writes_one_record_per_frame(tmp_path, stubbed):
    data_dir, code = run(tmp_path, "--no-video")
    out_dir = data_dir / "doper_out"
    records = json.loads((out_dir / "per_frame.json").read_text())
    assert code == 0
    assert [r["idx"] for r in records] == [0, 1, 2]
    assert all(r["detected"] and r["pnp_ok"] for r in records)
    assert records[0]["depth_cm"] == pytest.approx(50.0)
    assert sorted(p.name for p in (out_dir / "overlays").iterdir()) == [
        "000000.png",
        "000001.png",
        "000002.png",
    ]


def test_no_video_skips_mp4(tmp_path, stubbed):
    data_dir, _ = run(tmp_path, "--no-video")
    assert not (data_dir / "doper_out" / "doper_overlay.mp4").exists()


def test_video_written_by_default(tmp_path, stubbed):
    data_dir, _ = run(tmp_path)
    assert (data_dir / "doper_out" / "doper_overlay.mp4").exists()


def test_run_recorded_returns_summary(tmp_path, stubbed):
    data_dir = write_recording(tmp_path / "rec")
    args = infer_recorded._build_parser().parse_args(
        ["--checkpoint", "x", "--data-dir", str(data_dir), "--device", "cpu", "--no-video"]
    )
    result = infer_recorded.run_recorded(args)
    assert result["summary"]["frames"] == N_FRAMES
    assert result["summary"]["poses"] == N_FRAMES


def test_missing_detection_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(infer_recorded, "load_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(infer_recorded, "infer_image", lambda *args, **kwargs: ([], None))
    data_dir, _ = run(tmp_path, "--no-video")
    records = json.loads((data_dir / "doper_out" / "per_frame.json").read_text())
    assert not any(r["detected"] for r in records)
    assert records[0]["score"] is None


def test_fixed_bbox_rejects_rot_tta(tmp_path):
    with pytest.raises(SystemExit):
        infer_recorded.main(
            ["--checkpoint", "x", "--data-dir", str(tmp_path), "--fixed-bbox", "0", "0", "5", "5",
             "--rot-tta"]
        )  # fmt: skip


def test_empty_data_dir_exits(tmp_path, stubbed):
    (tmp_path / "intrinsics.json").write_text(
        json.dumps({"color_intrinsics": {"fx": 1.0, "fy": 1.0, "ppx": 0.0, "ppy": 0.0}})
    )
    with pytest.raises(SystemExit, match="no .*frames"):
        infer_recorded.main(["--checkpoint", "x", "--data-dir", str(tmp_path), "--device", "cpu"])


def test_keypoint_count_mismatch_with_checkpoint_exits(tmp_path, stubbed):
    keypoints = tmp_path / "kp.json"
    keypoints.write_text(json.dumps({"keypoints_3d": np.zeros((3, 3)).tolist()}))
    with pytest.raises(SystemExit, match="3 keypoints but the checkpoint predicts 4"):
        run(tmp_path, "--keypoints-3d", str(keypoints), "--num-keypoints", str(NUM_KEYPOINTS))


def test_unopenable_video_writer_exits(tmp_path, stubbed, monkeypatch):
    class ClosedWriter:
        fourcc = staticmethod(cv2.VideoWriter.fourcc)

        def __init__(self, *args):
            pass

        def isOpened(self):  # noqa: N802 - OpenCV method name
            return False

    monkeypatch.setattr(infer_recorded.cv2, "VideoWriter", ClosedWriter)
    with pytest.raises(SystemExit, match="--no-video"):
        run(tmp_path)


def test_unreadable_frames_are_logged(tmp_path, stubbed, capsys):
    data_dir = write_recording(tmp_path / "rec")
    (data_dir / "000001_color.png").write_bytes(b"not a png")
    infer_recorded.main(
        ["--checkpoint", "unused.pth", "--data-dir", str(data_dir), "--device", "cpu", "--no-video"]
    )
    captured = capsys.readouterr()
    assert "skipped 1 unreadable frame(s)" in captured.out + captured.err
