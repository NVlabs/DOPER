# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cv2
import numpy as np
import pytest
from PIL import Image

from doper.image_reader import read_image_size


def _decoded_size(path):
    image = cv2.imread(str(path))
    return image.shape[1], image.shape[0]


@pytest.mark.parametrize("suffix", [".png", ".jpg"])
def test_read_image_size_matches_opencv_decode(tmp_path, suffix):
    path = tmp_path / f"frame{suffix}"
    cv2.imwrite(str(path), np.zeros((30, 50, 3), dtype=np.uint8))
    assert read_image_size(path) == _decoded_size(path) == (50, 30)


def test_read_image_size_applies_exif_rotation_like_opencv(tmp_path):
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise on display
    Image.new("RGB", (50, 30)).save(path, exif=exif)
    assert read_image_size(path) == _decoded_size(path) == (30, 50)


def test_read_image_size_returns_none_for_unreadable_file(tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"not an image")
    assert read_image_size(path) is None
