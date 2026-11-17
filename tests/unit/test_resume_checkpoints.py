# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation checkpoints keep the true best score across resumes."""

import json

import pytest
import torch

from doper.train import resumed_best_kp, save_validation_checkpoints, trim_epoch_timing


def test_last_checkpoint_records_new_best(tmp_path):
    best = save_validation_checkpoints({"epoch": 0}, 5.0, float("inf"), tmp_path)
    best = save_validation_checkpoints({"epoch": 1}, 3.0, best, tmp_path)
    assert best == 3.0
    assert torch.load(tmp_path / "last.pth", weights_only=False)["best_kp"] == 3.0
    assert torch.load(tmp_path / "best.pth", weights_only=False)["epoch"] == 1


def test_worse_epoch_keeps_best_checkpoint(tmp_path):
    best = save_validation_checkpoints({"epoch": 0}, 3.0, float("inf"), tmp_path)
    best = save_validation_checkpoints({"epoch": 1}, 4.0, best, tmp_path)
    assert best == 3.0
    assert torch.load(tmp_path / "best.pth", weights_only=False)["epoch"] == 0
    assert torch.load(tmp_path / "last.pth", weights_only=False)["best_kp"] == 3.0


def test_resume_repairs_stale_best_from_older_last_checkpoint(tmp_path):
    # Older runs saved last.pth before updating best_kp.
    torch.save({"epoch": 1, "best_kp": 3.0}, tmp_path / "best.pth")
    stale_last = {"epoch": 1, "best_kp": 5.0}
    assert resumed_best_kp(stale_last, tmp_path / "last.pth") == 3.0
    assert resumed_best_kp(stale_last, tmp_path / "missing" / "last.pth") == 5.0


def test_resume_reads_best_beside_checkpoint_outside_work_dir(tmp_path, monkeypatch):
    old_run = tmp_path / "old_run"
    old_run.mkdir()
    torch.save({"epoch": 1, "best_kp": 3.0}, old_run / "best.pth")
    work_dir = tmp_path / "new_run"
    work_dir.mkdir()
    torch.save({"epoch": 1, "best_kp": 9.0}, work_dir / "best.pth")
    monkeypatch.chdir(tmp_path)
    assert resumed_best_kp({"epoch": 1, "best_kp": 5.0}, "old_run/last.pth") == 3.0


def test_resume_ignores_unreadable_best_checkpoint(tmp_path):
    (tmp_path / "best.pth").write_bytes(b"truncated")
    with pytest.warns(UserWarning, match="unreadable"):
        assert resumed_best_kp({"epoch": 1, "best_kp": 5.0}, tmp_path / "last.pth") == 5.0


def test_resume_drops_timing_rows_for_repeated_epochs(tmp_path):
    path = tmp_path / "epoch_timing.jsonl"
    path.write_text("".join(json.dumps({"epoch": e}) + "\n" for e in range(4)))
    trim_epoch_timing(path, 2)
    assert [json.loads(line)["epoch"] for line in path.read_text().splitlines()] == [0, 1]
    trim_epoch_timing(tmp_path / "absent.jsonl", 2)


def test_resume_drops_truncated_timing_row(tmp_path):
    path = tmp_path / "epoch_timing.jsonl"
    path.write_text(json.dumps({"epoch": 0}) + "\n" + '{"epoch": 1, "sec')
    trim_epoch_timing(path, 2)
    assert [json.loads(line)["epoch"] for line in path.read_text().splitlines()] == [0]
