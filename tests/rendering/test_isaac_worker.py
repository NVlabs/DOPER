# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests of sample-atomic worker checkpoints and resume."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from doper.scene.geometry import intrinsics_from_lens
from rendering.config import render_config_from_mapping
from rendering.isaac.worker import RenderWorker

REPO = Path(__file__).resolve().parents[2]


class FakeScene:
    def __init__(self):
        self.indices = []
        self.cameras = []

    def realize(self, record):
        self.indices.append(record["index"])

    def set_camera(self, camera):
        self.cameras.append(camera.copy())


class FakeCapture:
    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at
        self.primes = 0

    def prime(self):
        self.primes += 1

    def capture(self):
        assert self.primes == 1
        self.calls += 1
        if self.calls == self.fail_at:
            raise TimeoutError("right view unavailable")
        return (
            np.full((8, 8, 3), self.calls, dtype=np.uint8),
            np.ones((8, 8), dtype=np.float32),
            np.ones((8, 8), dtype=bool),
        )


def _inputs(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    cfg = render_config_from_mapping(
        {
            "output": {"resolution": 8, "render_masks": True},
            "camera": {"stereo": {"enabled": True}},
        },
        str(tmp_path),
    )
    args = SimpleNamespace(
        outdir=str(tmp_path / "out"),
        config=str(config),
        worker_id=0,
        start_idx=0,
        end_idx=2,
        resume=False,
        flush_every=1,
        flush_every_sec=0,
    )
    camera = np.eye(4)
    camera[2, 3] = 1
    scenes = {
        i: {
            "index": i,
            "camera": {"cam_to_world": camera.tolist()},
            "object": {"pose": np.eye(4).tolist()},
        }
        for i in range(2)
    }
    return SimpleNamespace(
        args=args,
        cfg=cfg,
        scenes=scenes,
        kp_names=["KP_0"],
        kp_local=np.zeros((1, 3)),
        intr=intrinsics_from_lens(8, 8, 50, 36),
    )


def test_failed_stereo_sample_never_enters_checkpoint_and_resume_replays_whole_pair(
    tmp_path,
):
    inputs = _inputs(tmp_path)
    scene = FakeScene()
    worker = RenderWorker(inputs, scene, FakeCapture(fail_at=4))
    with pytest.raises(TimeoutError, match="right view"):
        worker.run()
    out = Path(inputs.args.outdir)
    checkpoint = out / "annotations_worker_000.partial.json"
    payload = json.loads(checkpoint.read_text())
    assert [image["id"] for image in payload["images"]] == [0, 1]
    assert [ann["image_id"] for ann in payload["annotations"]] == [0, 1]
    assert (out / "images/000001_left.png").is_file()  # saved, but not committed
    assert not (out / "annotations_worker_000.json").exists()
    preserved = {path: path.read_bytes() for path in (out / "images").glob("000000*")}

    inputs.args.resume = True
    resumed_scene, capture = FakeScene(), FakeCapture()
    final = RenderWorker(inputs, resumed_scene, capture).run()
    assert resumed_scene.indices == [1] and capture.calls == 2
    assert [path.read_bytes() for path in preserved] == list(preserved.values())
    complete = json.loads(final.read_text())
    assert [image["id"] for image in complete["images"]] == [0, 1, 2, 3]
    assert [ann["image_id"] for ann in complete["annotations"]] == [0, 1, 2, 3]
    assert not checkpoint.exists()
    assert len(resumed_scene.cameras) == 2
    assert resumed_scene.cameras[1][0, 3] - resumed_scene.cameras[0][0, 3] == pytest.approx(
        inputs.cfg.stereo_baseline_m
    )


def test_worker_cli_help_does_not_import_kit():
    code = """
import importlib.abc, runpy, sys
class NoKit(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'isaacsim', 'omni', 'carb', 'pxr'}:
            raise AssertionError('help imported Kit: ' + fullname)
sys.meta_path.insert(0, NoKit())
sys.argv = ['isaac_worker.py', '--help']
runpy.run_module('rendering.isaac.isaac_worker', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--scenes" in result.stdout
    assert "Traceback" not in result.stderr


def test_fully_reused_worker_does_not_prime_or_capture(tmp_path):
    inputs = _inputs(tmp_path)
    final = RenderWorker(inputs, FakeScene(), FakeCapture()).run()
    final.rename(final.with_suffix(".partial.json"))
    inputs.args.resume = True
    capture, scene = FakeCapture(), FakeScene()
    resumed = RenderWorker(inputs, scene, capture).run()
    assert capture.primes == capture.calls == 0
    assert scene.indices == []
    assert len(json.loads(resumed.read_text())["images"]) == 4


def test_prime_failure_cannot_publish_final_annotations(tmp_path):
    inputs = _inputs(tmp_path)
    capture = FakeCapture()

    def fail_prime():
        raise TimeoutError("capture initialization")

    capture.prime = fail_prime
    with pytest.raises(TimeoutError, match="initialization"):
        RenderWorker(inputs, FakeScene(), capture).run()
    out = Path(inputs.args.outdir)
    assert not (out / "annotations_worker_000.json").exists()
    assert json.loads((out / "annotations_worker_000.partial.json").read_text())["images"] == []


def test_native_environment_settles_each_view_before_saving(tmp_path):
    inputs = _inputs(tmp_path)
    inputs.environment = object()
    capture = FakeCapture()
    scene = FakeScene()
    scene.environment = SimpleNamespace(verify=lambda record: {"scene": record["index"]})
    worker = RenderWorker(inputs, scene, capture)
    worker.run()
    assert capture.calls == 8  # 2 samples x 2 views x (discard + saved)

    assert np.asarray(Image.open(Path(inputs.args.outdir) / "images/000000_left.png"))[0, 0, 0] == 2


def test_portable_environment_settles_without_native_readbacks(tmp_path):
    inputs = _inputs(tmp_path)
    capture = FakeCapture()
    scene = FakeScene()
    scene.procedural_environment = object()
    RenderWorker(inputs, scene, capture).run()
    assert capture.calls == 8
    assert not (Path(inputs.args.outdir) / "environment_applied").exists()
