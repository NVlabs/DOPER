# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAM2 prompts, the segmenter's optional import and the mask batch (``doper.nurec.masks``)."""

import json
import subprocess
import sys

import cv2
import numpy as np
import pytest

from doper.nurec.labels import ViewLabel
from doper.nurec.masks import (
    Prompts,
    Sam2Segmenter,
    excluded_pixels,
    sam2_prompts,
    segment_views,
)
from doper.recipes.nurec_schema import HsvExclusion

WIDTH, HEIGHT = 80, 60
# A blue fixture color: OpenCV HSV, strict bounds.
BLUE = HsvExclusion(h_above=85, h_below=120, s_above=90, v_above=100)
BLUE_BGR = (200, 60, 20)  # HSV about (108, 230, 200): excluded
GREY_BGR = (90, 90, 90)  # V 90: a dark, eligible keypoint color
WHITE_BGR = (240, 240, 240)  # V 240: too bright for a positive prompt


def _label(keypoints, visible, mask_box=(20, 5, 40, 30), cad_bbox=(1.5, 2.5, 70.0, 50.0)):
    """A label with a rectangular CAD silhouette ``mask_box`` (x0, y0, x1, y1 inclusive)."""
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    x0, y0, x1, y1 = mask_box
    mask[y0 : y1 + 1, x0 : x1 + 1] = 255
    return ViewLabel(
        index=7, seed_view="000000.png", reconstruction_id="recon", lineage="session",
        T_camera_nre=np.eye(4), T_camera_object_m=np.eye(4),
        keypoints_xy=np.asarray(keypoints, float), ray_visible=np.asarray(visible, bool),
        cad_bbox=cad_bbox, K=np.eye(3), image_size=(WIDTH, HEIGHT), cad_mask=mask,
    )  # fmt: skip


def _image(colours=()):
    image = np.full((HEIGHT, WIDTH, 3), GREY_BGR, np.uint8)
    for (x0, y0, x1, y1), colour in colours:
        image[y0:y1, x0:x1] = colour
    return image


def test_sam_box_from_raster_mask():
    keypoints = [[25.2, 10.0], [26.0, 12.0], [27.0, 14.0]]
    label = _label(keypoints, [True] * 3)
    prompts = sam2_prompts(label, _image(), exclude_hsv=BLUE)
    # The nonzero CAD pixels (x 20..40, y 5..30) widened by 10 px, not the float cad_bbox; it
    # reaches past the top edge, as SAM2 receives it.
    np.testing.assert_array_equal(prompts.box, [10, -5, 50, 40])
    np.testing.assert_array_equal(prompts.points, keypoints)
    np.testing.assert_array_equal(prompts.labels, [1, 1, 1])


def test_positive_points_are_eligible_and_evenly_spaced():
    # Eight eligible keypoints, one hidden, one blue, one bright and one outside the image.
    eligible = [[22.0 + 2 * i, 20.0] for i in range(8)]
    keypoints = [*eligible, [30.0, 10.0], [31.0, 26.0], [35.0, 26.0], [90.0, 10.0]]
    visible = [True] * 8 + [False, True, True, True]
    image = _image([((30, 25, 33, 28), BLUE_BGR), ((34, 25, 37, 28), WHITE_BGR)])
    prompts = sam2_prompts(_label(keypoints, visible), image, exclude_hsv=BLUE)
    picks = np.linspace(0, 7, 5).round().astype(int)
    np.testing.assert_array_equal(prompts.points[:5], np.array(eligible)[picks])
    # Nine blue pixels in the box: no negative point (more than ten are needed).
    assert len(prompts.points) == 5 and set(prompts.labels) == {1}


def test_negative_point_at_blue_median_and_mean_fallback():
    keypoints = [[30.0, 10.0], [31.0, 12.0]]
    image = _image([((30, 8, 34, 14), BLUE_BGR)])  # covers both keypoints: none eligible
    prompts = sam2_prompts(_label(keypoints, [True, True]), image, exclude_hsv=BLUE)
    np.testing.assert_array_equal(prompts.points[0], np.mean(keypoints, axis=0))
    blue_y, blue_x = np.nonzero(excluded_pixels(image, BLUE))
    np.testing.assert_array_equal(prompts.points[1], [np.median(blue_x), np.median(blue_y)])
    np.testing.assert_array_equal(prompts.labels, [1, 0])
    # Without an exclusion range the blue pixels are ordinary: dark enough to prompt.
    plain = sam2_prompts(_label(keypoints, [True, True]), image, exclude_hsv=None)
    np.testing.assert_array_equal(plain.labels, [1])


def test_excluded_pixels_bounds_are_strict():
    hsv = np.array([[[85, 200, 200], [86, 91, 101], [119, 255, 255], [120, 255, 255],
                     [100, 90, 255], [100, 255, 100]]], np.uint8)  # fmt: skip
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    hsv_back = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    expected = ((hsv_back[..., 0] > 85) & (hsv_back[..., 0] < 120) & (hsv_back[..., 1] > 90)
                & (hsv_back[..., 2] > 100))  # fmt: skip
    np.testing.assert_array_equal(excluded_pixels(image, BLUE), expected)
    assert not excluded_pixels(image, None).any()


def test_prompts_need_a_cad_mask():
    from dataclasses import replace

    label = replace(_label([[30.0, 10.0]], [True]), cad_mask=None)
    with pytest.raises(ValueError, match="cad_mask"):
        sam2_prompts(label, _image(), exclude_hsv=BLUE)
    empty = replace(label, cad_mask=np.zeros((HEIGHT, WIDTH), np.uint8))
    with pytest.raises(ValueError, match="empty"):
        sam2_prompts(empty, _image(), exclude_hsv=BLUE)


def test_sam2_not_imported_without_extra(monkeypatch, tmp_path):
    code = ("import sys, doper.nurec.masks, doper.nurec.package; "
            "print([m for m in ('sam2', 'torch') if m in sys.modules])")  # fmt: skip
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
    monkeypatch.setitem(sys.modules, "sam2", None)
    with pytest.raises(ImportError, match="nurec-sam"):
        Sam2Segmenter(weights=tmp_path / "sam2.1_hiera_small.pt")


class _BoxSegmenter:
    """Fills the prompt box (clipped): a stand-in for SAM2."""

    def __init__(self):
        self.calls = []

    def segment(self, image, prompts: Prompts):
        self.calls.append(prompts)
        mask = np.zeros(image.shape[:2], np.uint8)
        x0, y0, x1, y1 = np.clip(prompts.box, 0, None).astype(int)
        mask[y0:y1, x0:x1] = 255
        return mask, 0.5


def test_segment_views_writes_masks(tmp_path):
    from dataclasses import replace

    label = _label([[25.0, 10.0]], [True])
    cv2.imwrite(str(tmp_path / "cad.png"), label.cad_mask)
    cv2.imwrite(str(tmp_path / "rgb.png"), _image())
    segmenter = _BoxSegmenter()
    rows = segment_views([replace(label, cad_mask=None, mask_file="cad.png")], segmenter,
                         image_path=lambda _: tmp_path / "rgb.png",
                         cad_mask_path=lambda _: tmp_path / "cad.png", out=tmp_path / "sam",
                         exclude_hsv=BLUE)  # fmt: skip
    assert rows[0]["index"] == 7 and rows[0]["score"] == 0.5
    assert rows[0]["prompts"] == json.loads(json.dumps(segmenter.calls[0].to_record()))
    written = cv2.imread(str(tmp_path / "sam" / rows[0]["mask_file"]), cv2.IMREAD_GRAYSCALE)
    assert written.shape == (HEIGHT, WIDTH) and set(np.unique(written)) == {0, 255}


def test_sam2_segmenter_calls_the_predictor(monkeypatch, tmp_path):
    """The predictor gets the RGB image and the prompts once, single-mask (stub sam2)."""
    import types

    calls = {}

    class Predictor:
        def __init__(self, model):
            calls["model"] = model

        def set_image(self, image):
            calls["image"] = image

        def predict(self, point_coords, point_labels, box, multimask_output):
            calls["predict"] = (point_coords, point_labels, box, multimask_output)
            mask = np.zeros((1, *calls["image"].shape[:2]), bool)
            mask[0, 5:10, 5:10] = True
            return mask, np.array([0.9]), None

    build = types.ModuleType("sam2.build_sam")
    build.build_sam2 = lambda config, weights, device: (config, weights, device)
    predictor = types.ModuleType("sam2.sam2_image_predictor")
    predictor.SAM2ImagePredictor = Predictor
    monkeypatch.setitem(sys.modules, "sam2", types.ModuleType("sam2"))
    monkeypatch.setitem(sys.modules, "sam2.build_sam", build)
    monkeypatch.setitem(sys.modules, "sam2.sam2_image_predictor", predictor)
    weights = tmp_path / "w.pt"
    with pytest.raises(FileNotFoundError):
        Sam2Segmenter(weights=weights, device="cpu")
    weights.write_bytes(b"w")
    segmenter = Sam2Segmenter(weights=weights, device="cpu")
    assert calls["model"] == ("configs/sam2.1/sam2.1_hiera_s.yaml", str(weights), "cpu")
    image = _image([((0, 0, 1, 1), (10, 20, 30))])
    prompts = sam2_prompts(_label([[25.0, 10.0]], [True]), image, exclude_hsv=BLUE)
    mask, score = segmenter.segment(image, prompts)
    np.testing.assert_array_equal(calls["image"][0, 0], [30, 20, 10])  # BGR to RGB
    assert calls["predict"][3] is False and calls["predict"][2] is prompts.box
    assert score == pytest.approx(0.9) and mask.dtype == np.uint8 and mask.sum() == 25 * 255
