# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frame capture retries malformed buffers and never returns a late frame."""

import numpy as np
import pytest

from rendering.isaac.capture import FrameCapture


class Annotators:
    def __init__(self):
        self.now = 0.0
        self.steps = []
        self.rgb = np.full((2, 2, 4), 123, dtype=np.uint8)
        self.depth = np.ones((2, 2), dtype=np.float32)
        self.seg = {
            "data": np.array([[1, 2], [3, 0]], dtype=np.uint32),
            "info": {
                "idToLabels": {
                    "1": "/World/object",
                    "2": "/World/object/mesh",
                    "3": "/World/object2",
                    "0": "BACKGROUND",
                }
            },
        }

    def step(self, **kwargs):
        self.steps.append(kwargs)
        self.now += 0.1

    def capture(self, **overrides):
        kwargs = dict(
            step=self.step,
            rgb=lambda: self.rgb,
            depth=lambda: self.depth,
            segmentation=lambda: self.seg,
            resolution=2,
            object_path="/World/object",
            rt_subframes=8,
            timeout_sec=1,
            clock=lambda: self.now,
        )
        kwargs.update(overrides)
        return FrameCapture(**kwargs).capture()


def test_capture_copies_arrays_and_matches_only_object_subtree():
    source = Annotators()
    rgb, depth, mask = source.capture()
    assert rgb.shape == (2, 2, 3) and rgb.dtype == np.uint8
    assert depth.dtype == np.float32 and mask.dtype == np.bool_
    np.testing.assert_array_equal(mask, [[True, True], [False, False]])
    source.rgb[:] = 0
    source.depth[:] = 9
    source.seg["data"][:] = 0
    assert (rgb == 123).all() and (depth == 1).all()
    np.testing.assert_array_equal(mask, [[True, True], [False, False]])
    assert source.steps == [{"rt_subframes": 8, "delta_time": 0.0, "pause_timeline": True}]


@pytest.mark.parametrize(
    "field,empty",
    [
        ("rgb", np.array([])),
        ("depth", np.array([])),
        ("seg", {}),
        ("seg", {"data": np.array([]), "info": {}}),
    ],
)
def test_empty_annotators_retry_until_all_buffers_are_valid(field, empty):
    source = Annotators()
    good = getattr(source, field)

    def step(**kwargs):
        source.step(**kwargs)
        setattr(source, field, empty if len(source.steps) == 1 else good)

    rgb, depth, mask = source.capture(step=step)
    assert len(source.steps) == 2 and mask.sum() == 2


@pytest.mark.parametrize(
    "seg",
    [
        None,
        {"data": np.zeros((2, 2))},
        {"data": np.zeros((2, 2), dtype=np.uint32), "info": {}},
        {"data": np.zeros((1, 2), dtype=np.uint32), "info": {"idToLabels": {}}},
    ],
)
def test_invalid_segmentation_times_out_explicitly(seg):
    source = Annotators()
    source.seg = seg
    with pytest.raises(TimeoutError, match="frame capture"):
        source.capture(timeout_sec=0.25)
    assert len(source.steps) == 3


def test_foreground_depth_must_be_positive_finite_but_background_can_be_infinite():
    source = Annotators()
    source.depth[1] = [float("inf"), 0]
    _, depth, _ = source.capture()
    assert np.isposinf(depth[1, 0]) and depth[1, 1] == 0
    source.depth[0, 0] = float("inf")
    with pytest.raises(TimeoutError):
        source.capture(timeout_sec=0.25)


def test_expired_step_cannot_return_late_frame():
    source = Annotators()
    with pytest.raises(TimeoutError):
        source.capture(timeout_sec=0.05)


def test_prime_discards_initial_scene_without_reading_buffers_then_capture_uses_one_step():
    source = Annotators()

    def getter(value):
        assert len(source.steps) > 1, "prime must not read initial-scene annotators"
        return value

    capture = FrameCapture(
        step=source.step,
        rgb=lambda: getter(source.rgb),
        depth=lambda: getter(source.depth),
        segmentation=lambda: getter(source.seg),
        resolution=2,
        object_path="/World/object",
        rt_subframes=1,
        timeout_sec=1,
        clock=lambda: source.now,
    )
    source.rgb[:] = 0  # Uninitialized initial scene is intentionally discarded.
    capture.prime()
    assert len(source.steps) == 1
    # Caller applies its first saved scene/camera now; opaque black is valid.
    source.rgb[..., 3] = 255
    for index in range(2):
        rgb, _, _ = capture.capture()
        assert (rgb == 0).all()
        assert len(source.steps) == index + 2
    assert all(
        call == {"rt_subframes": 1, "delta_time": 0.0, "pause_timeline": True}
        for call in source.steps
    )


def test_prime_has_its_own_monotonic_deadline_without_reading_buffers():
    source = Annotators()

    def no_read():
        pytest.fail("prime must discard annotator buffers")

    capture = FrameCapture(
        step=source.step,
        rgb=no_read,
        depth=no_read,
        segmentation=no_read,
        resolution=2,
        object_path="/World/object",
        rt_subframes=1,
        timeout_sec=0.05,
        clock=lambda: source.now,
    )
    with pytest.raises(TimeoutError, match="prime"):
        capture.prime()
    assert len(source.steps) == 1


def test_all_zero_rgba_with_valid_foreground_buffers_never_completes():
    source = Annotators()
    source.rgb[:] = 0
    with pytest.raises(TimeoutError, match="foreground alpha"):
        source.capture(timeout_sec=0.35)
    assert len(source.steps) == 4


def test_zero_foreground_alpha_retries_until_renderer_populates_alpha():
    source = Annotators()
    source.rgb[..., :3] = 0  # A genuinely black image is valid once opaque.
    source.rgb[..., 3] = 0
    source.rgb[1, :, 3] = 255  # Background alpha cannot make foreground ready.

    def step(**kwargs):
        source.step(**kwargs)
        if len(source.steps) == 3:
            source.rgb[0, :, 3] = 255

    rgb, _, mask = source.capture(step=step)
    assert len(source.steps) == 3 and mask.sum() == 2
    assert (rgb == 0).all()


def test_transparent_empty_background_is_valid_without_segmented_object():
    source = Annotators()
    source.rgb[:] = 0
    source.seg["data"][:] = 0
    source.depth[:] = float("inf")
    rgb, depth, mask = source.capture()
    assert (rgb == 0).all() and not mask.any() and np.isposinf(depth).all()


def test_prime_can_initialize_controls_without_bypassing_capture_verification():
    source = Annotators()
    events = []
    exposure = {"value": 0.02}
    drift_during_step = {"enabled": False}

    def initialize(**kwargs):
        source.step(**kwargs)
        exposure["value"] = 0.01
        events.append("initialize")

    def checked(**kwargs):
        source.step(**kwargs)
        if drift_during_step["enabled"]:
            exposure["value"] = 0.02
        if exposure["value"] != 0.01:
            raise RuntimeError("exposure drift")
        events.append("capture")

    capture = FrameCapture(
        step=checked,
        prime_step=initialize,
        rgb=lambda: source.rgb,
        depth=lambda: source.depth,
        segmentation=lambda: source.seg,
        resolution=2,
        object_path="/World/object",
        rt_subframes=8,
        timeout_sec=1,
        clock=lambda: source.now,
    )
    capture.prime()
    capture.capture()
    assert events == ["initialize", "capture"]
    drift_during_step["enabled"] = True
    with pytest.raises(RuntimeError, match="exposure drift"):
        capture.capture()
    assert events == ["initialize", "capture"]
