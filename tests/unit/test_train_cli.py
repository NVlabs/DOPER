# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training entry point accepts an argv list and records anonymous provenance."""

import json
import shutil

import pytest
import timm

import doper.train
from doper.training.checkpoints import provenance


def test_main_accepts_argv(tmp_path):
    with pytest.raises(SystemExit) as exc:
        doper.train.main(["--help"])
    assert exc.value.code == 0


def test_provenance_has_no_paths_or_remotes():
    prov = provenance()
    assert set(prov) <= {"doper_version", "git_commit", "git_dirty"}
    assert not any("/" in str(v) for v in prov.values())


def make_split(root, name):
    split = root / name
    (split / "images").mkdir(parents=True)
    (split / "annotations.json").write_text("{}")


def dry_run_args(tmp_path, num_keypoints=3, pretrained=False):
    """A valid ``--dry-run`` over empty splits and a keypoint file of three points."""
    data = tmp_path / "data"
    make_split(data, "train")
    make_split(data, "val")
    keypoints = tmp_path / "keypoints_3d.json"
    keypoints.write_text(json.dumps({"keypoints_3d": [[0.0, 0.0, 0.0]] * 3}))
    return [
        "--data-root", str(data), "--keypoints-3d", str(keypoints),
        "--num-keypoints", str(num_keypoints), "--work-dir", str(tmp_path / "work"),
        *([] if pretrained else ["--no-pretrained"]), "--dry-run",
    ]  # fmt: skip


def test_train_dry_run_does_not_build_the_model(tmp_path, monkeypatch, capsys):
    def unexpected(*args, **kwargs):
        pytest.fail("a dry run built a model")

    monkeypatch.setattr(timm, "create_model", unexpected)
    monkeypatch.setattr(doper.train, "DoperNet", unexpected)
    doper.train.main(dry_run_args(tmp_path, pretrained=True))
    out = capsys.readouterr().out
    assert "dry run" in out and "3 keypoints" in out and "work_dir" in out
    assert not (tmp_path / "work").exists()


def test_train_dry_run_refuses_a_keypoint_count_mismatch(tmp_path):
    with pytest.raises(ValueError, match="num-keypoints is 64 but .* has 3"):
        doper.train.main(dry_run_args(tmp_path, num_keypoints=64))
    assert not (tmp_path / "work").exists()


def test_train_dry_run_refuses_a_missing_split(tmp_path):
    args = dry_run_args(tmp_path)
    shutil.rmtree(tmp_path / "data" / "val")
    with pytest.raises(FileNotFoundError, match="val"):
        doper.train.main(args)


def test_train_dry_run_refuses_a_missing_keypoint_file(tmp_path):
    args = dry_run_args(tmp_path)
    (tmp_path / "keypoints_3d.json").unlink()
    with pytest.raises(FileNotFoundError, match="keypoints_3d"):
        doper.train.main(args)
