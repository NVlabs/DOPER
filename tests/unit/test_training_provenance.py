# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training provenance is sanitized by default and full only on request."""

from __future__ import annotations

import json

import pytest
import torch

from doper import train
from doper.training import loop


class _Samples(torch.utils.data.Dataset):
    ids = list(range(4))
    anns_by_img = {i: [dict(keypoints=[0.0] * 192)] for i in range(4)}

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return dict(det_image=torch.ones(3, 4, 4), crop_image=torch.ones(3, 4, 4))


class _TinyModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = torch.nn.Identity()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, det, crop):
        return self.weight * det.mean()


def _run_training(tmp_path, monkeypatch, extra_args):
    data_root = tmp_path / "renders" / "train_root"
    data_root.mkdir(parents=True)
    (data_root / "summary.json").write_text(json.dumps({"images": 4}))
    config_path = tmp_path / "configs" / "recipe.yaml"
    config_path.parent.mkdir()
    config_path.write_text("seed: 7\n")
    monkeypatch.setattr(train, "build_train_set", lambda *args: _Samples())
    monkeypatch.setattr(train, "DoperDataset", lambda *args: _Samples())
    monkeypatch.setattr(train, "DoperNet", _TinyModel)
    monkeypatch.setattr(loop, "compute_losses", lambda out, *a, **kw: (out.square(), {}))
    monkeypatch.setattr(train, "evaluate", lambda *args, **kw: {"kp_err_px": 1.0})
    train.main(
        [
            "--config",
            str(config_path),
            "--data-root",
            str(data_root),
            "--work-dir",
            str(tmp_path / "work"),
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--no-pretrained",
            "--no-amp",
            *extra_args,
        ]
    )
    checkpoint = torch.load(tmp_path / "work" / "best.pth", weights_only=False)
    on_disk = json.loads((tmp_path / "work" / "training_provenance.json").read_text())
    return checkpoint["training_provenance"], on_disk, data_root, config_path


def _strings(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


@pytest.mark.parametrize("source", ["checkpoint", "json"])
def test_default_provenance_has_no_paths(tmp_path, monkeypatch, source):
    checkpoint_record, json_record, _, _ = _run_training(tmp_path, monkeypatch, [])
    record = checkpoint_record if source == "checkpoint" else json_record

    assert not any(str(tmp_path) in text for text in _strings(record))
    assert not any(text.startswith("/") for text in _strings(record))
    path_derived = [
        record["arguments"]["data_root"],
        record["arguments"]["work_dir"],
        record["arguments"]["config"],
        record["resolved_config"]["data"]["root"],
        record["resolved_config"]["train"]["work_dir"],
        record["resolved_config"]["background"]["splits_json"],
        record["train_sources"][0]["root"],
        record["config_file"]["path"],
        *record["dataset_summaries"],
    ]
    assert not any("/" in str(value) for value in path_derived)
    assert record["arguments"]["data_root"] == "train_root"
    assert record["dataset_summaries"] == {"train_root": {"images": 4}}
    assert record["config_file"]["path"] == "recipe.yaml"
    assert record["config_file"]["sha256"] is not None
    assert record["arguments"]["seed"] == 7


def test_private_provenance_keeps_full_paths(tmp_path, monkeypatch):
    record, json_record, data_root, config_path = _run_training(
        tmp_path, monkeypatch, ["--private-provenance"]
    )

    assert json_record["arguments"]["data_root"] == str(data_root)
    assert json_record["config_file"]["path"] == str(config_path)
    assert record["arguments"]["data_root"] == str(data_root)
    assert record["arguments"]["private_provenance"] is True
    assert record["resolved_config"]["data"]["root"] == str(data_root)
    assert record["train_sources"][0]["root"] == str(data_root)
    assert record["dataset_summaries"] == {str(data_root): {"images": 4}}
    assert record["config_file"]["path"] == str(config_path)
