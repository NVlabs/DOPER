# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import numpy as np
import pytest

from doper.recorded import (
    frame_index,
    intrinsics_block_to_arrays,
    list_color_frames,
    load_intrinsics,
)


def test_list_color_frames_sorts_by_index_and_caps(tmp_path: Path) -> None:
    for index in (10, 2, 1, 100):
        (tmp_path / f"{index:06d}_color.png").write_bytes(b"")
    (tmp_path / "000003_depth.png").write_bytes(b"")
    names = [path.name for path in list_color_frames(tmp_path)]
    assert names == [
        "000001_color.png",
        "000002_color.png",
        "000010_color.png",
        "000100_color.png",
    ]
    assert [p.name for p in list_color_frames(tmp_path, max_frames=2)] == names[:2]
    assert len(list_color_frames(tmp_path, max_frames=0)) == 4


def test_frame_index_parses_prefix_and_defaults_to_zero() -> None:
    assert frame_index("/data/000042_color.png") == 42
    assert frame_index(Path("000007_color.png")) == 7
    assert frame_index("notanumber_color.png") == 0


def test_load_intrinsics_builds_matrix_and_distortion(tmp_path: Path) -> None:
    block = {"fx": 600.0, "fy": 610.0, "ppx": 320.5, "ppy": 240.5}
    path = tmp_path / "intrinsics.json"
    path.write_text(json.dumps({"color_intrinsics": block}))
    camera, dist = load_intrinsics(path)
    np.testing.assert_array_equal(
        camera, [[600.0, 0.0, 320.5], [0.0, 610.0, 240.5], [0.0, 0.0, 1.0]]
    )
    assert dist.shape == (5, 1)
    assert not dist.any()
    block["distortion_coeffs"] = [0.1, 0.2, 0.0, 0.0, 0.3]
    path.write_text(json.dumps({"color_intrinsics": block}))
    assert load_intrinsics(str(path))[1].ravel().tolist() == [0.1, 0.2, 0.0, 0.0, 0.3]


def test_load_intrinsics_rejects_missing_block(tmp_path: Path) -> None:
    path = tmp_path / "intrinsics.json"
    path.write_text("{}")
    with pytest.raises(KeyError):
        load_intrinsics(path)


def test_load_intrinsics_reads_realsense_coeffs(tmp_path: Path) -> None:
    block = {"fx": 600.0, "fy": 610.0, "ppx": 320.5, "ppy": 240.5, "coeffs": [0.1, -0.2, 0, 0, 0]}
    path = tmp_path / "intrinsics.json"
    path.write_text(json.dumps({"color_intrinsics": block}))
    assert load_intrinsics(path)[1].ravel().tolist() == [0.1, -0.2, 0.0, 0.0, 0.0]


def test_intrinsics_block_precedence_and_cx_cy() -> None:
    block = {"fx": 1.0, "fy": 2.0, "cx": 3.0, "cy": 4.0}
    block |= {"distortion_coeffs": [0.5, 0, 0, 0, 0], "coeffs": [9.0, 0, 0, 0, 0]}
    camera, dist = intrinsics_block_to_arrays(block)
    np.testing.assert_array_equal(camera, [[1.0, 0.0, 3.0], [0.0, 2.0, 4.0], [0.0, 0.0, 1.0]])
    assert dist.ravel()[0] == 0.5
    with pytest.raises(ValueError, match="ppx/ppy or cx/cy"):
        intrinsics_block_to_arrays({"fx": 1.0, "fy": 1.0})
