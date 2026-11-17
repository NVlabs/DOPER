# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pose-basis Kalman filter and its optional RTS smoother."""

from __future__ import annotations

import importlib
import sys

import cv2
import numpy as np
import pytest

from doper.filtering.pose_basis import PoseBasisKalmanFilter


def _geodesic_deg(rvec_a: np.ndarray, rvec_b: np.ndarray) -> float:
    ra, _ = cv2.Rodrigues(np.asarray(rvec_a).reshape(3, 1))
    rb, _ = cv2.Rodrigues(np.asarray(rvec_b).reshape(3, 1))
    rel = ra.T @ rb
    cos = (np.trace(rel) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def test_rts_returns_one_pose_per_forward_step() -> None:
    filt = PoseBasisKalmanFilter(rot_process_std=0.1)
    filt.enable_recording()
    rvec = np.array([0.0, 0.0, 0.0], dtype=np.float64).reshape(3, 1)
    tvec = np.array([0.0, 0.0, 0.4], dtype=np.float64).reshape(3, 1)
    for i in range(12):
        filt.update(rvec, tvec, now_s=i / 15.0)
    smoothed = filt.apply_rts()
    assert len(smoothed) == 12
    assert all(p is not None for p in smoothed)


def test_rts_middle_frame_is_closer_to_neighbors_than_raw_spike() -> None:
    filt = PoseBasisKalmanFilter(rot_process_std=0.05, rot_measurement_std=0.05)
    filt.enable_recording()
    rvec = np.array([0.0, 0.0, 0.0], dtype=np.float64).reshape(3, 1)
    tvec = np.array([0.0, 0.0, 0.4], dtype=np.float64).reshape(3, 1)
    for i in range(8):
        filt.update(rvec, tvec, now_s=i / 15.0)
    spike = np.array([1.2, 0.0, 0.0], dtype=np.float64).reshape(3, 1)
    filt.update(spike, tvec, now_s=8 / 15.0)
    for i in range(9, 16):
        filt.update(rvec, tvec, now_s=i / 15.0)

    smoothed = filt.apply_rts()
    mid = smoothed[8]
    assert mid is not None
    r_mid, _ = mid
    assert _geodesic_deg(rvec, r_mid) < _geodesic_deg(rvec, spike)


def test_recorded_repeated_dropouts_reset_since_last_measurement() -> None:
    filt = PoseBasisKalmanFilter(reset_after_s=1.0)
    filt.enable_recording()
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.array([0.0, 0.0, 0.4], dtype=np.float64).reshape(3, 1)

    assert filt.update(rvec, tvec, now_s=0.0) is not None
    assert filt.update(None, None, now_s=0.4) is not None
    assert filt.update(None, None, now_s=0.8) is not None
    assert filt.update(None, None, now_s=1.2) is None
    assert not filt.is_active

    smoothed = filt.apply_rts()
    assert len(smoothed) == 4
    assert all(pose is not None for pose in smoothed[:3])
    assert smoothed[3] is None


def test_valid_measurement_after_recorded_dropout_starts_new_segment() -> None:
    filt = PoseBasisKalmanFilter(reset_after_s=1.0)
    filt.enable_recording()
    rvec = np.zeros((3, 1), dtype=np.float64)
    first_tvec = np.array([0.0, 0.0, 0.4], dtype=np.float64).reshape(3, 1)
    next_tvec = np.array([0.2, -0.1, 0.8], dtype=np.float64).reshape(3, 1)

    filt.update(rvec, first_tvec, now_s=0.0)
    filt.update(None, None, now_s=0.4)
    filt.update(None, None, now_s=0.8)
    assert filt.update(None, None, now_s=1.2) is None
    assert filt.update(rvec, next_tvec, now_s=1.3) is not None

    smoothed = filt.apply_rts()
    assert len(smoothed) == 5
    assert smoothed[3] is None
    assert smoothed[4] is not None
    assert len(filt._segments) == 2
    assert filt._segments[0].step_indices == [0, 1, 2]
    assert filt._segments[1].step_indices == [4]


def test_filter_missing_extra_names_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    # Evict the filter modules and every cached bayesfilter submodule so the import is
    # re-executed regardless of which tests ran earlier.
    prefixes = ("doper.filtering", "bayesfilter.")
    for name in [m for m in sys.modules if m.startswith(prefixes) or m == "doper.filtering"]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "bayesfilter", None)
    with pytest.raises(ImportError, match=r"doper\[filter\]"):
        importlib.import_module("doper.filtering.pose_basis")
