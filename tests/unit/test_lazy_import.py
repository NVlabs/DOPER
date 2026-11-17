# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``import doper`` stays cheap; public names resolve lazily."""

import subprocess
import sys

import pytest

import doper


def _run(code: str) -> str:
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_import_doper_does_not_import_torch() -> None:
    assert _run("import doper, sys; print('torch' in sys.modules)") == "False"


def test_public_names_still_resolve() -> None:
    from doper import DoperConfig, DoperDataset, DoperNet, compute_losses

    assert all(obj is not None for obj in (DoperConfig, DoperDataset, DoperNet, compute_losses))
    assert set(doper.__all__) >= {"DoperNet", "DoperConfig", "DoperDataset", "compute_losses"}


def test_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="no_such_name"):
        _ = doper.no_such_name


def test_scene_modules_import_without_torch() -> None:
    code = (
        "import sys; import doper.scene.geometry, doper.scene.labels; "
        "print([m for m in ('torch', 'torchvision', 'timm', 'cv2') if m in sys.modules])"
    )
    assert _run(code) == "[]"


def test_keypoints_payload_rounds_to_six_decimals() -> None:
    from doper.scene.labels import keypoints_payload

    payload = keypoints_payload("clip", ["a"], [(0.12345678, 1.0, 2.0)])
    assert payload == {
        "object": "clip",
        "num_keypoints": 1,
        "keypoint_names": ["a"],
        "keypoints_3d": [[0.123457, 1.0, 2.0]],
    }
