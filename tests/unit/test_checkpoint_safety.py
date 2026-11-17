# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoints load with ``weights_only=True`` and never touch the network."""

import hashlib
import json
import pickle
import socket
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from tiny_coco import make_mini_coco

import doper.train
from doper.checkpoint_io import load_checkpoint
from doper.config import DataConfig, DoperConfig, ModelConfig
from doper.infer import load_model
from doper.training.checkpoints import resumed_best_kp

TINY_MODEL = [
    "--backbone",
    "resnet18",
    "--allow-nonstandard-backbone",
    "--no-pretrained",
    "--det-size",
    "128",
    "--crop-size",
    "128",
    "--keypoint-head",
    "heatmap",
    "--epochs",
    "1",
    "--batch-size",
    "2",
    "--max-iters-per-epoch",
    "2",
    "--num-workers",
    "0",
    "--device",
    "cpu",
    "--no-amp",
    "--warmup-iters",
    "1",
    "--log-interval",
    "1",
    "--val-interval",
    "1",
    "--save-interval",
    "1000",
]


def make_config() -> DoperConfig:
    """Return a default config; ``load_model`` fills the model section."""
    return DoperConfig(data=DataConfig(root="."), model=ModelConfig())


def tiny_args(root: Path, work_dir: Path) -> list[str]:
    """Return CLI arguments for a two-step CPU training run on ``root``."""
    return [
        "--data-root",
        str(root),
        "--work-dir",
        str(work_dir),
        "--keypoints-3d",
        str(root / "keypoints_3d.json"),
        *TINY_MODEL,
    ]


@pytest.fixture(scope="session")
def tiny_run(tmp_path_factory) -> dict[str, Path]:
    """Train two steps once and return the dataset root and the work directory."""
    base = tmp_path_factory.mktemp("tiny_run")
    root = make_mini_coco(base / "coco")
    work_dir = base / "work"
    doper.train.main(tiny_args(root, work_dir))
    return {"root": root, "work_dir": work_dir}


@pytest.fixture
def tiny_checkpoint(tiny_run) -> Path:
    """Return the ``last.pth`` written by the real training loop."""
    return tiny_run["work_dir"] / "last.pth"


def _refuse(*args, **kwargs):
    raise AssertionError("network access attempted")


def spy_on_torch_load(monkeypatch) -> list[dict]:
    """Record keyword arguments of every ``torch.load`` call."""
    calls: list[dict] = []
    real_load = torch.load

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    return calls


def test_load_model_offline(tiny_checkpoint, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(socket, "socket", _refuse)
    cfg = make_config()
    model = load_model(tiny_checkpoint, cfg, "cpu")
    assert not model.training
    assert cfg.model.pretrained is False


def test_load_model_uses_weights_only(tiny_checkpoint, monkeypatch):
    calls = spy_on_torch_load(monkeypatch)
    load_model(tiny_checkpoint, make_config(), "cpu")
    assert calls
    assert all(call.get("weights_only") is True for call in calls)


def test_resume_and_init_use_weights_only(tiny_run, tmp_path, monkeypatch):
    calls = spy_on_torch_load(monkeypatch)
    work_dir = tmp_path / "resumed"
    resume = tiny_run["work_dir"] / "last.pth"
    doper.train.main(
        [*tiny_args(tiny_run["root"], work_dir), "--resume", str(resume), "--epochs", "2"]
    )
    assert calls
    assert all(call.get("weights_only") is True for call in calls)


def test_real_checkpoint_rng_state_round_trips(tiny_checkpoint):
    payload = load_checkpoint(tiny_checkpoint, "cpu")
    assert payload["rng_state"]["numpy"][0] == "MT19937"
    assert resumed_best_kp(payload, tiny_checkpoint) < float("inf")


def test_arbitrary_pickle_is_rejected(tmp_path):
    class Payload:
        def __reduce__(self):
            return (sys.exit, (1,))

    path = tmp_path / "evil.pth"
    torch.save({"model": Payload()}, path)
    with pytest.raises(pickle.UnpicklingError):
        load_model(path, make_config(), "cpu")


def test_resume_missing_file_raises_before_any_state(tmp_path):
    work_dir = tmp_path / "work"
    with pytest.raises(FileNotFoundError):
        doper.train.main(
            [*tiny_args(tmp_path / "coco", work_dir), "--resume", str(tmp_path / "missing.pth")]
        )
    assert not work_dir.exists()


# --- SPEED+ checkpoints (rotation-aux payload and layout identity) ------------------------------

GEN7_CHECKPOINT_SHA256 = "0824734e2d596adc0a5c45d8de034b67a11c2669926919e7105916974e4c4bba"
PAPER_LAYOUT_SHA256 = "a6697cafc62bdf5286fb8af453e44bae881b4ad481b8928edf288c85f77fbfa6"


def _rotation_aux_head():
    from doper.rotation_aux import CosineClassifier

    torch.manual_seed(0)
    return CosineClassifier(8, 3, 30.0)


def _bins() -> np.ndarray:
    return np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]], dtype=np.float32
    )


def _write_layout(path: Path, points: list[list[float]]) -> Path:
    path.write_text(json.dumps({"keypoints_3d": points}))
    return path


def _source_format_payload() -> dict:
    """A checkpoint shaped like the source pseudo trainer's ``save_checkpoint`` output."""
    head = _rotation_aux_head()
    return {
        "model": {"layer.weight": torch.ones(2, 2)},
        "epoch": 10,
        "cfg": {"model": {"num_keypoints": 75, "keypoint_head": "heatmap"}},
        "metrics": {"holdout_top1_deg": np.float64(12.5), "epoch": 10},
        "rotation_aux": {
            "weight": head.weight.detach().cpu(),
            "feature_dim": 8,
            "num_bins": 3,
            "pool_size": 4,
            "logit_scale": 30.0,
            "bin_quaternions": _bins(),
        },
        "training_provenance": {
            "experiment": "speedplus_rotation_aux_real_pseudo_backbone",
            "source_checkpoint": "teacher/epoch_10.pth",
            "inlier_gate": ">21",
            "arguments": {"min_inliers_exclusive": 21, "output_dir": "runs/gen07"},
        },
    }


def test_source_format_checkpoint_loads_weights_only(tmp_path, monkeypatch):
    from doper.rotation_aux import RotationAuxPayload, normalize_quaternions

    path = tmp_path / "epoch_10.pth"
    torch.save(_source_format_payload(), path)
    calls = spy_on_torch_load(monkeypatch)
    payload = load_checkpoint(path, "cpu")
    assert calls and all(call.get("weights_only") is True for call in calls)
    assert isinstance(payload["rotation_aux"]["bin_quaternions"], np.ndarray)
    assert payload["training_provenance"]["inlier_gate"] == ">21"
    auxiliary = RotationAuxPayload.from_checkpoint(payload)
    assert isinstance(auxiliary.bin_quaternions, torch.Tensor)
    assert torch.equal(auxiliary.bin_quaternions, torch.from_numpy(normalize_quaternions(_bins())))
    assert torch.equal(auxiliary.build_head().weight, _rotation_aux_head().weight)


PAPER_POINTS_SHA256 = "4c78a38de43fbe96ed2639749d95eafaa6a5db93f1de68f8ebdb3fae97c8ef0e"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_new_checkpoint_stores_tensor_bins_fingerprint_and_layout_sha(tmp_path):
    from doper.checkpoint_io import atomic_torch_save
    from doper.keypoints import keypoints_3d_fingerprint
    from doper.rotation_aux import RotationAuxPayload
    from doper.speedplus.checkpoints import (
        LAYOUT_POINTS_SHA_KEY,
        LAYOUT_SHA_KEY,
        LayoutMatch,
        check_checkpoint_layout,
        load_layout,
        load_speedplus_checkpoint,
        speedplus_checkpoint_payload,
    )

    points = [[0.1, 0.2, 0.3], [-0.4, 0.5, 0.6], [0.7, -0.8, 0.9], [1.0, 1.1, -1.2]]
    layout = _write_layout(tmp_path / "keypoints_3d.json", points)
    other = _write_layout(tmp_path / "other.json", points[::-1])
    payload = speedplus_checkpoint_payload(
        model_state={"layer.weight": torch.ones(2, 2)},
        model_cfg={"num_keypoints": 4},
        rotation_aux=RotationAuxPayload.from_head(_rotation_aux_head(), _bins()),
        keypoints_3d=layout,
        epoch=10,
        metrics={"holdout_top1_deg": 12.5},
    )
    path = tmp_path / "epoch_10.pth"
    atomic_torch_save(payload, path)
    loaded = load_checkpoint(path, "cpu")
    bins = loaded["rotation_aux"]["bin_quaternions"]
    assert isinstance(bins, torch.Tensor) and bins.dtype == torch.float32
    assert loaded["cfg"]["keypoints_3d_fingerprint"] == keypoints_3d_fingerprint(np.asarray(points))
    trained, given_other = load_layout(layout), load_layout(other)
    assert loaded[LAYOUT_SHA_KEY] == _sha256(layout) == trained.sha256
    assert loaded[LAYOUT_POINTS_SHA_KEY] == trained.points_sha256
    assert loaded["epoch"] == 10 and loaded["cfg"]["model"] == {"num_keypoints": 4}
    checkpoint_sha = _sha256(path)
    assert check_checkpoint_layout(loaded, checkpoint_sha, trained, {}) == LayoutMatch("recorded")
    assert load_speedplus_checkpoint(path, layout, {})[LAYOUT_SHA_KEY] == trained.sha256
    # Other points are refused, naming both points hashes.
    with pytest.raises(ValueError, match=f"{trained.points_sha256}.*{given_other.points_sha256}"):
        check_checkpoint_layout(loaded, checkpoint_sha, given_other, {})
    with pytest.raises(ValueError, match=given_other.points_sha256):
        load_speedplus_checkpoint(path, other, {})
    # The same points in another file format: accepted, recorded as a formatting deviation.
    reformatted = tmp_path / "reformatted.json"
    reformatted.write_text(json.dumps({"keypoints_3d": points}, indent=2))
    match = check_checkpoint_layout(loaded, checkpoint_sha, load_layout(reformatted), {})
    assert match == LayoutMatch("recorded", formatting_deviation=True)
    assert match.summary_fields() == {
        "layout_check": "recorded", "layout_formatting_deviation": True
    }  # fmt: skip


def test_checkpoint_recording_only_the_file_sha(tmp_path):
    from doper.speedplus.checkpoints import LAYOUT_SHA_KEY, LayoutMatch, check_checkpoint_layout
    from doper.speedplus.definitions import LayoutPins

    # An unknown layout file: the file SHA-256 must match.
    checkpoint = {LAYOUT_SHA_KEY: "6" * 64}
    assert check_checkpoint_layout(
        checkpoint, "c" * 64, LayoutPins("6" * 64, "7" * 64), {}
    ) == LayoutMatch("recorded")  # fmt: skip
    with pytest.raises(ValueError, match=f"{'6' * 64}.*{'8' * 64}"):
        check_checkpoint_layout(checkpoint, "c" * 64, LayoutPins("8" * 64, "7" * 64), {})
    # The paper file: its points hash is known, so a reformatted paper layout is a deviation.
    paper = {LAYOUT_SHA_KEY: PAPER_LAYOUT_SHA256}
    match = check_checkpoint_layout(paper, "c" * 64, LayoutPins("9" * 64, PAPER_POINTS_SHA256), {})
    assert match == LayoutMatch("recorded", formatting_deviation=True)
    with pytest.raises(ValueError, match=f"{PAPER_POINTS_SHA256}.*{'5' * 64}"):
        check_checkpoint_layout(paper, "c" * 64, LayoutPins(PAPER_LAYOUT_SHA256, "5" * 64), {})


def test_legacy_checkpoint_needs_registered_layout(tmp_path):
    from doper.recipes import load_any_recipe
    from doper.speedplus.checkpoints import (
        LayoutMatch,
        check_checkpoint_layout,
        load_speedplus_checkpoint,
    )
    from doper.speedplus.definitions import PAPER_LAYOUT, LayoutPins

    registry = load_any_recipe("speedplus").legacy_checkpoints
    assert registry == {GEN7_CHECKPOINT_SHA256: PAPER_LAYOUT_SHA256}
    assert LayoutPins(PAPER_LAYOUT_SHA256, PAPER_POINTS_SHA256) == PAPER_LAYOUT
    legacy = _source_format_payload()
    # Accepted through the registered file SHA-256.
    assert check_checkpoint_layout(
        legacy, GEN7_CHECKPOINT_SHA256, PAPER_LAYOUT, registry
    ) == LayoutMatch("legacy")  # fmt: skip
    # Accepted through the registered layout's points hash: a recorded formatting deviation.
    reformatted = LayoutPins("3" * 64, PAPER_POINTS_SHA256)
    match = check_checkpoint_layout(legacy, GEN7_CHECKPOINT_SHA256, reformatted, registry)
    assert match == LayoutMatch("legacy", formatting_deviation=True)
    # Refused on another points hash, naming both.
    other_points = LayoutPins("4" * 64, "5" * 64)
    with pytest.raises(ValueError, match=f"legacy.*{PAPER_POINTS_SHA256}.*{'5' * 64}"):
        check_checkpoint_layout(legacy, GEN7_CHECKPOINT_SHA256, other_points, registry)
    # The paper file with other points cannot exist, but the points hash still decides.
    with pytest.raises(ValueError, match=f"{PAPER_POINTS_SHA256}.*{'5' * 64}"):
        check_checkpoint_layout(
            legacy, GEN7_CHECKPOINT_SHA256, LayoutPins(PAPER_LAYOUT_SHA256, "5" * 64), registry
        )
    unregistered = "1" * 64
    with pytest.raises(ValueError, match=f"{unregistered}.*{PAPER_LAYOUT_SHA256}"):
        check_checkpoint_layout(legacy, unregistered, PAPER_LAYOUT, registry)
    # A registered non-paper layout has no known points hash: its file SHA-256 must match.
    path = tmp_path / "epoch_10.pth"
    torch.save(legacy, path)
    layout = _write_layout(tmp_path / "keypoints_3d.json", [[0.0, 0.0, 1.0]] * 4)
    file_sha = _sha256(path)
    with pytest.raises(ValueError, match=f"legacy.*{file_sha}"):
        load_speedplus_checkpoint(path, layout, registry)
    loaded = load_speedplus_checkpoint(path, layout, {file_sha: _sha256(layout)})
    assert isinstance(loaded["rotation_aux"]["bin_quaternions"], np.ndarray)
    with pytest.raises(ValueError, match=f"{'2' * 64}.*{_sha256(layout)}"):
        load_speedplus_checkpoint(path, layout, {file_sha: "2" * 64})


@pytest.mark.parametrize(
    ("points", "message"),
    [([[0.0, 0.0]], r"\(N, 3\)"), ([], r"\(N, 3\)"), ([[0.0, float("nan"), 1.0]], "non-finite")],
)
def test_layout_points_must_be_finite_n_by_3(tmp_path, points, message):
    from doper.speedplus.checkpoints import load_layout_points

    layout = tmp_path / "keypoints_3d.json"
    layout.write_text(json.dumps({"keypoints_3d": points}))
    with pytest.raises(ValueError, match=message):
        load_layout_points(layout)
