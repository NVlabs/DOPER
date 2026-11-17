# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from doper.dataset import DoperDataset


@pytest.mark.parametrize(
    "bop,x,should_fail",
    [(True, 4.0, False), (False, 4.0, True), (True, 2.0, True), (True, -0.1, True)],
)
def test_boundary_tolerance_matches_bop_visibility_only(tmp_path, bop, x, should_fail):
    mask = np.zeros((10, 10), np.uint8)
    mask[5, 5] = 255
    cv2.imwrite(str(tmp_path / "mask.png"), mask)
    ann = {"keypoints": [x, 5, 2], "mask_file": "mask.png"}
    if bop:
        ann["bop"] = {"scene": "000000", "frame": 0, "instance": 0}
    dataset = SimpleNamespace(base=str(tmp_path), anns_by_img={0: [ann]}, _data_faults=[])
    DoperDataset._report_keypoint_mask_agreement(dataset, "val")
    assert bool(dataset._data_faults) == should_fail
