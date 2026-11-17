# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random

import numpy as np
import pytest
import torch

from doper.checkpoint_io import (
    atomic_torch_save,
    capture_rng_state,
    checkpoint_configs,
    restore_rng_state,
)


def test_atomic_checkpoint_keeps_previous_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "last.pth"
    atomic_torch_save({"epoch": 3, "weights": torch.ones(2)}, path)
    original = path.read_bytes()

    def fail(payload, stream):
        stream.write(b"partial")
        raise OSError("preempted write")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError):
        atomic_torch_save({"epoch": 4}, path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_checkpoint_replaces_complete_file(tmp_path):
    path = tmp_path / "last.pth"
    atomic_torch_save({"epoch": 0}, path)
    atomic_torch_save({"epoch": 1}, path)
    assert torch.load(path, weights_only=True)["epoch"] == 1


def test_rng_roundtrip():
    state = capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(3))
    restore_rng_state(state)
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])


@pytest.mark.parametrize(
    ("payload", "expected_cfg", "expected_model"),
    [
        ({}, {}, {}),
        ({"cfg": None}, {}, {}),
        ({"cfg": {"model": None}}, {"model": None}, {}),
        ({"cfg": {"model": {"crop_size": 512}}}, {"model": {"crop_size": 512}}, {"crop_size": 512}),
    ],
)
def test_checkpoint_configs_reads_legacy_none_as_empty(payload, expected_cfg, expected_model):
    _, cfg, model_cfg = checkpoint_configs(payload, "ckpt")
    assert cfg == expected_cfg
    assert model_cfg == expected_model


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "ckpt must contain a state dictionary"),
        ({"cfg": []}, "ckpt has a non-mapping cfg"),
        ({"cfg": {"model": 3}}, "ckpt has a non-mapping cfg.model"),
    ],
)
def test_checkpoint_configs_rejects_non_mappings(payload, message):
    with pytest.raises(ValueError, match=message):
        checkpoint_configs(payload, "ckpt")
