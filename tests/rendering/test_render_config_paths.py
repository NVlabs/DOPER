# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render config path semantics: relative to the config file, missing asset dirs are errors."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from rendering.config import load_render_config


def write_config(path: Path, **sections: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(sections))
    return path


def test_config_paths_resolve_against_config_dir(tmp_path):
    (tmp_path / "cfg").mkdir()
    (tmp_path / "cfg" / "hdri").mkdir()
    path = write_config(tmp_path / "cfg" / "r.yaml", scene={"hdri_dir": "hdri"})
    cfg = load_render_config(path)
    assert cfg.base_dir == str(tmp_path / "cfg")
    assert cfg.resolve("hdri") == str(tmp_path / "cfg" / "hdri")


def test_paths_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    (tmp_path / "cfg").mkdir()
    write_config(tmp_path / "cfg" / "r.yaml")
    monkeypatch.chdir(tmp_path)
    cfg = load_render_config(Path("cfg") / "r.yaml")
    assert cfg.resolve("x") == str(tmp_path / "cfg" / "x")


def test_absolute_paths_are_kept(tmp_path):
    cfg = load_render_config(write_config(tmp_path / "r.yaml"))
    assert cfg.resolve(str(tmp_path / "elsewhere")) == str(tmp_path / "elsewhere")
    assert cfg.resolve("") is None
    assert cfg.resolve(None) is None


@pytest.mark.parametrize("key", ["hdri_dir", "texture_dir", "distractor_dir"])
def test_missing_asset_dir_raises(tmp_path, key):
    path = write_config(tmp_path / "r.yaml", scene={key: "nope"})
    with pytest.raises(FileNotFoundError, match=key):
        load_render_config(path)


def test_asset_dir_that_is_a_file_raises(tmp_path):
    (tmp_path / "hdri").write_text("not a directory")
    path = write_config(tmp_path / "r.yaml", scene={"hdri_dir": "hdri"})
    with pytest.raises(FileNotFoundError, match="hdri_dir"):
        load_render_config(path)


@pytest.mark.parametrize("scene", [{}, {"hdri_dir": ""}, {"hdri_dir": None}])
def test_absent_or_empty_asset_dir_means_none(tmp_path, scene):
    cfg = load_render_config(write_config(tmp_path / "r.yaml", scene=scene))
    assert cfg.hdri_dir == ""
    assert cfg.resolve(cfg.hdri_dir) is None
    assert cfg.texture_dir == "" and cfg.distractor_dir == ""
