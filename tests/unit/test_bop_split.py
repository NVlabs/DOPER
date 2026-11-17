# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the whole-scene train/validation split of a converted BOP root."""

import json
from pathlib import Path

import pytest
from bop_pbr_data import make_pbr, write_keypoints
from tiny_coco import tiny_model_config

from doper.bop import convert, split
from doper.config import AugConfig, DataConfig, DoperConfig
from doper.dataset import DoperDataset


@pytest.fixture
def bop_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """A converted three-scene root (nine images) and its keypoint file."""
    keypoints = write_keypoints(tmp_path / "kp.json")
    summary = convert.convert_pbr(
        make_pbr(tmp_path), 2, keypoints, tmp_path / "converted", min_visible_keypoints=1
    )
    return summary.out, keypoints


def _scenes(root: Path, name: str) -> set[str]:
    data = json.loads((root / name / "annotations.json").read_text())
    return {a["bop"]["scene"] for a in data["annotations"]}


def test_split_holds_out_whole_scenes(bop_fixture, tmp_path: Path) -> None:
    converted, keypoints = bop_fixture
    summary = split.split_scenes(converted, tmp_path / "split", 2, keypoints, val_scenes=1)
    train, val = _scenes(summary.out, "train"), _scenes(summary.out, "val")
    assert len(val) == 1 and len(train) == 2 and not train & val
    assert tuple(sorted(val)) == summary.val_scenes
    assert (summary.train_images, summary.val_images) == (6, 3)
    assert (summary.out / "keypoints_3d.json").read_bytes() == keypoints.read_bytes()


def test_split_is_seeded_and_default_seed_matches_source(bop_fixture, tmp_path: Path) -> None:
    converted, keypoints = bop_fixture
    assert split.DEFAULT_SEED == 42 and split.DEFAULT_VAL_SCENES == 5
    first = split.split_scenes(converted, tmp_path / "a", 2, keypoints, val_scenes=1, seed=7)
    again = split.split_scenes(converted, tmp_path / "b", 2, keypoints, val_scenes=1, seed=7)
    assert first.val_scenes == again.val_scenes
    everything = split.split_scenes(converted, tmp_path / "c", 2, keypoints, val_scenes=99)
    assert everything.train_images == 0 and everything.val_images == 9


def test_split_relinks_pixels_and_renumbers(bop_fixture, tmp_path: Path) -> None:
    converted, keypoints = bop_fixture
    out = split.split_scenes(converted, tmp_path / "s", 2, keypoints, val_scenes=1).out
    data = json.loads((out / "val" / "annotations.json").read_text())
    assert [image["id"] for image in data["images"]] == [0, 1, 2]
    assert [a["id"] for a in data["annotations"]] == [0, 1, 2]
    for image in data["images"]:
        link = out / "val" / "images" / image["file_name"]
        assert link.is_symlink() and not link.readlink().is_absolute()
    for annotation in data["annotations"]:
        assert (out / "val" / annotation["mask_file"]).exists()


def test_split_links_survive_moving_the_tree(bop_fixture, tmp_path: Path) -> None:
    converted, keypoints = bop_fixture
    out = split.split_scenes(converted, tmp_path / "s", 2, keypoints, val_scenes=1).out
    moved = tmp_path.parent / f"{tmp_path.name}-moved"
    tmp_path.rename(moved)
    links = sorted((moved / "s").rglob("*.png"))
    assert links and all(p.is_symlink() and p.exists() for p in links)
    assert not out.exists()


def test_split_output_loads_in_doper_dataset(bop_fixture, tmp_path: Path) -> None:
    converted, keypoints = bop_fixture
    out = split.split_scenes(converted, tmp_path / "s", 2, keypoints, val_scenes=1).out
    config = DoperConfig(
        data=DataConfig(root=str(out)),
        model=tiny_model_config(num_keypoints=4, det_input_size=64, crop_size=32),
        aug=AugConfig(enabled=False),
    )
    assert len(DoperDataset(config, "train")) == 6
    assert len(DoperDataset(config, "val")) == 3


def test_split_keeps_every_instance_of_a_multi_instance_frame(tmp_path: Path) -> None:
    import numpy as np
    from bop_pbr_data import write_scene

    keypoints = write_keypoints(tmp_path / "kp.json")
    pbr = tmp_path / "train_pbr"
    write_scene(
        pbr, "000000", {0: [(2, np.eye(3), (0.0, 0.0, 300.0)), (2, np.eye(3), (-70.0, 0, 300.0))]}
    )
    write_scene(pbr, "000001", {0: [(2, np.eye(3), (0.0, 0.0, 300.0))]})
    converted = convert.convert_pbr(pbr, 2, keypoints, tmp_path / "c", min_visible_keypoints=1).out
    summary = split.split_scenes(converted, tmp_path / "s", 2, keypoints, val_scenes=1, seed=0)
    assert summary.multi_instance_images == 1
    assert summary.train_annotations + summary.val_annotations == 3


def test_resolve_root_accepts_nested_train_layout(bop_fixture, tmp_path: Path) -> None:
    converted, keypoints = bop_fixture
    nested = tmp_path / "nested"
    (nested / "train").mkdir(parents=True)
    for name in ("annotations.json", "images", "masks"):
        (nested / "train" / name).symlink_to(converted / name)
    assert split.resolve_root(nested) == nested / "train"
    assert split.split_scenes(nested, tmp_path / "s", 2, keypoints, val_scenes=1).val_images == 3
    with pytest.raises(FileNotFoundError, match="annotations.json"):
        split.resolve_root(tmp_path / "missing")


def test_scene_falls_back_to_file_name_only_without_recorded_scene() -> None:
    image = {"file_name": "000012_000003.jpg"}
    assert split.scene_of(image, [{"bop": {"scene": "000099"}}]) == "000099"
    assert split.scene_of(image, [{}]) == "000012"


def test_existing_destination_needs_force_and_bad_arguments_fail(bop_fixture, tmp_path) -> None:
    converted, keypoints = bop_fixture
    destination = tmp_path / "s"
    split.split_scenes(converted, destination, 2, keypoints, val_scenes=1)
    with pytest.raises(FileExistsError, match="force"):
        split.split_scenes(converted, destination, 2, keypoints, val_scenes=1)
    split.split_scenes(converted, destination, 2, keypoints, val_scenes=2, force=True)
    assert len(_scenes(destination, "val")) == 2
    with pytest.raises(ValueError, match="obj_id"):
        split.split_scenes(converted, tmp_path / "x", 0, keypoints)
    with pytest.raises(ValueError, match="val_scenes"):
        split.split_scenes(converted, tmp_path / "x", 2, keypoints, val_scenes=-1)
    with pytest.raises(FileNotFoundError, match="keypoints"):
        split.split_scenes(converted, tmp_path / "x", 2, tmp_path / "none.json")


def test_force_keeps_existing_output_when_the_source_is_invalid(bop_fixture, tmp_path) -> None:
    converted, keypoints = bop_fixture
    destination = tmp_path / "s"
    split.split_scenes(converted, destination, 2, keypoints, val_scenes=1)
    with pytest.raises(FileNotFoundError):
        split.split_scenes(tmp_path / "no_such_root", destination, 2, keypoints, force=True)
    with pytest.raises(FileNotFoundError):
        split.split_scenes(converted, destination, 2, tmp_path / "none.json", force=True)
    assert (destination / "train" / "annotations.json").is_file()
