# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fresh-run RNG isolation and explicit deterministic-backend contracts."""

import random
import sys

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from doper.config import TrainConfig
from doper.performance import configure_runtime, loader_options, seed_training


@pytest.fixture(autouse=True)
def restore_runtime():
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    yield
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic


def draw():
    values = np.empty((3,), dtype=np.float32)
    cv2.randu(values, 0.0, 1.0)
    return random.random(), np.random.rand(), torch.rand(1).item(), values.copy()


@pytest.mark.parametrize("reproducible", [False, True])
def test_complete_seeding(reproducible):
    cfg = TrainConfig(reproducible=reproducible, seed=43)
    seed_training(cfg)
    first = draw()
    seed_training(cfg)
    second = draw()
    assert first[:3] == second[:3]
    np.testing.assert_array_equal(first[3], second[3])
    seed_training(TrainConfig(seed=43))
    expected = torch.Generator().manual_seed(43)
    assert torch.equal(torch.rand(3), torch.rand(3, generator=expected))


class WorkerRandomness(Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, index):
        a, b, c, d = draw()
        return torch.tensor([index, a, b, c, *d], dtype=torch.float64)


def sequence(noise, validation_between_epochs):
    cfg = TrainConfig(reproducible=True, seed=43, num_workers=2)
    loader = DataLoader(
        WorkerRandomness(),
        batch_size=4,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        multiprocessing_context="spawn",
        **loader_options(cfg),
    )
    val = DataLoader(
        torch.arange(16),
        batch_size=4,
        **loader_options(TrainConfig(reproducible=True, seed=43, num_workers=0), stream="val"),
    )
    torch.rand(noise)  # Model initialization or unrelated caller consumption.
    epochs = [torch.cat(list(loader))]
    if validation_between_epochs:
        list(val)
        torch.rand(137)
    epochs.append(torch.cat(list(loader)))
    return epochs


def test_worker_draws_and_shuffle_ignore_model_and_validation_rng():
    left = sequence(3, False)
    right = sequence(701, True)
    assert all(torch.equal(a, b) for a, b in zip(left, right, strict=True))
    assert not torch.equal(left[0], left[1])
    assert "generator" not in loader_options(TrainConfig())


def test_mode_enforces_strict_backend_without_changing_default(monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    cfg = TrainConfig(reproducible=True, cudnn_benchmark=True, amp=False)
    configure_runtime(cfg, "cpu")
    assert cfg.cudnn_benchmark is False  # Effective config must describe execution.
    assert not torch.backends.cudnn.benchmark
    assert torch.backends.cudnn.deterministic
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    configure_runtime(TrainConfig(amp=False), "cpu")
    assert torch.are_deterministic_algorithms_enabled()  # Do not reset caller policy.


def test_cuda_workspace_is_set_before_context_use(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    configure_runtime(TrainConfig(reproducible=True, amp=False), "cuda")
    import os

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


@pytest.mark.parametrize("value,initialized", [(":bad", False), (None, True)])
def test_unsafe_cuda_configuration_fails_early(monkeypatch, value, initialized):
    if value is None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", value)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: initialized)
    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        configure_runtime(TrainConfig(reproducible=True, amp=False), "cuda")


def test_cli_records_effective_mode_and_supports_yaml_override(tmp_path, monkeypatch):
    from doper import train

    config = tmp_path / "config.yaml"
    config.write_text("reproducible: true\ncudnn_benchmark: true\n")
    build = train.build_config
    captured = []

    class StopBeforeDataError(Exception):
        pass

    def checked(args):
        captured.append(build(args))
        raise StopBeforeDataError

    monkeypatch.setattr(train, "build_config", checked)
    for flags, enabled in [([], True), (["--no-reproducible"], False)]:
        monkeypatch.setattr(
            sys, "argv", ["train", "--config", str(config), "--data-root", str(tmp_path), *flags]
        )
        with pytest.raises(StopBeforeDataError):
            train.main()
        assert captured[-1].train.reproducible is enabled
        assert captured[-1].train.cudnn_benchmark is (not enabled)


def test_reproducible_resume_is_rejected_before_training(tmp_path, monkeypatch):
    from doper import train

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--data-root",
            str(tmp_path),
            "--reproducible",
            "--resume",
            str(tmp_path / "missing.pth"),
            "--device",
            "cpu",
        ],
    )
    with pytest.raises(ValueError, match="fresh"):
        train.main()


def test_default_worker_opencv_streams_are_seeded_and_distinct(monkeypatch):
    cfg = TrainConfig(num_workers=2, opencv_threads=None)
    initialize = loader_options(cfg)["worker_init_fn"]

    def values(seed):
        monkeypatch.setattr(torch, "initial_seed", lambda: seed)
        initialize(0)
        result = np.empty(8, dtype=np.float32)
        cv2.randu(result, 0.0, 1.0)
        return result

    np.testing.assert_array_equal(values(43), values(43))
    assert not np.array_equal(values(43), values(44))


def test_training_help_formats_all_flags(monkeypatch, capsys):
    from doper import train

    monkeypatch.setattr(sys, "argv", ["train", "--help"])
    with pytest.raises(SystemExit) as exc:
        train.main()
    assert exc.value.code == 0
    assert "--reproducible" in capsys.readouterr().out
