# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The GB300 deployment (warm) timing protocol: sample, order, statistics and manifest."""

import dataclasses
import json
import random

import cv2
import numpy as np
import pytest
from test_bop_variants import (
    CAMERAS,
    CROP_SIZE,
    REF,
    SUP,
    TLESS,
    _detection,
    _object_keypoints,
    _view,
)
from test_bop_variants import _settings as _variant_settings

from doper.bop import deployment_timing as timing
from doper.bop import variants
from doper.bop.multiview import FusedPose, ImageKey
from doper.refinement import RefineSettings


def _keys(scene, frames, stream=""):
    return [ImageKey(scene, f, stream) for f in frames]


# --- the sample ---------------------------------------------------------------------------------


def test_groups_and_weights_match_source():
    # HOPEv2: 188 uncalibrated single-view images and 269 calibrated four-view groups, the real
    # population sizes, so the stratum sizes are those of the GB300 sample (10 and 14).
    single = [k for s in range(1, 41) for k in _keys(s, range(5))][:188]
    calibrated = [k for s in range(41, 48) for k in _keys(s, range(40))][:269]
    groups = {k: (k,) for k in single}
    groups.update({k: (k, *_keys(k.scene_id, [k.im_id + 100 + j for j in range(3)]))
                   for k in calibrated})  # fmt: skip
    sample = timing.sample_groups("hopev2", sorted(groups), groups)
    picks = [int(x) for x in np.linspace(0, 187, 10)] + [int(x) for x in np.linspace(0, 268, 14)]
    expected = [single[i] for i in picks[:10]] + [calibrated[i] for i in picks[10:]]
    assert [g.reference for g in sample] == expected
    np.testing.assert_allclose([g.weight for g in sample[:10]], 188 / 457 / 10, rtol=0, atol=0)
    np.testing.assert_allclose([g.weight for g in sample[10:]], 269 / 457 / 14, rtol=0, atol=0)
    assert sample[0].weight == pytest.approx(0.04114, abs=1e-5)  # the GB300 groups.json value
    assert sample[-1].weight == pytest.approx(0.04204, abs=1e-5)
    assert all(g.views == groups[g.reference] for g in sample)
    # T-LESS: the middle scheduled image of each scene, weighted by the scene's share.
    tless = {k: (k, ImageKey(k.scene_id, 99)) for s, n in ((1, 5), (2, 4), (3, 6))
             for k in _keys(s, range(10, 10 + n))}  # fmt: skip
    sample = timing.sample_groups("tless", sorted(tless), tless)
    assert [g.reference for g in sample] == [ImageKey(1, 12), ImageKey(2, 12), ImageKey(3, 13)]
    assert [g.weight for g in sample] == [5 / 15, 4 / 15, 6 / 15]
    assert sample[0].views == (ImageKey(1, 12), ImageKey(1, 99))
    # HOT3D: strata by view count (2 then 3), 24 groups in all.
    two = _keys(1, range(30), "214-1")
    three = _keys(2, range(70), "214-1")
    hot3d = {k: (k, ImageKey(1, k.im_id, "1201-2")) for k in two}
    hot3d.update({k: (k, ImageKey(2, k.im_id, "1201-1"), ImageKey(2, k.im_id, "1201-2"))
                  for k in three})  # fmt: skip
    sample = timing.sample_groups("hot3d", sorted(hot3d), hot3d)
    n_two, n_three = round(24 * 30 / 100), round(24 * 70 / 100)
    assert [len(g.views) for g in sample] == [2] * n_two + [3] * n_three
    assert [g.reference for g in sample[:n_two]] == [two[int(x)] for x in np.linspace(0, 29, n_two)]
    assert sample[0].weight == 30 / 100 / n_two and sample[-1].weight == 70 / 100 / n_three
    # HANDAL: 24 evenly spaced images, weight 1/24.
    handal = {k: (k,) for s in (3, 7) for k in _keys(s, range(0, 40, 2))}
    sample = timing.sample_groups("handal", sorted(handal), handal)
    assert [g.reference for g in sample] == [sorted(handal)[int(x)] for x in np.linspace(0, 39, 24)]
    assert {g.weight for g in sample} == {1 / 24}
    with pytest.raises(ValueError, match="unknown"):
        timing.sample_groups("ycbv", sorted(handal), handal)
    with pytest.raises(ValueError, match="group"):
        timing.sample_groups("handal", [ImageKey(9, 9)], handal)


def test_weighted_stats_match_source():
    stats = timing.weighted_stats([(3.0, 1.0), (1.0, 1.0), (2.0, 1.0), (4.0, 1.0)])
    assert stats == {"mean": 2.5, "median": 2.0, "p95": 4.0}
    # Unnormalized weights: cumulative weights of the sorted values 1, 2, 5 are .2, .3, 1.
    stats = timing.weighted_stats([(5.0, 7.0), (1.0, 2.0), (2.0, 1.0)])
    assert stats["mean"] == pytest.approx((35 + 2 + 2) / 10)
    assert stats["median"] == 5.0 and stats["p95"] == 5.0
    # Equal weights accumulate rounding: 20 x 0.05 sums to just under 0.5 at the 10th value, so
    # the median is the 11th (as in the research benchmark, whose expression follows).
    stats = timing.weighted_stats([(float(v), 1.0) for v in range(1, 21)])
    assert stats["median"] == 11.0
    rng = np.random.default_rng(5)
    for size in (1, 2, 7, 48):
        vals, w = rng.uniform(0.5, 2.0, size), rng.uniform(0.01, 0.1, size)
        stats = timing.weighted_stats(list(zip(vals.tolist(), w.tolist(), strict=True)))
        w = w / w.sum()
        o = np.argsort(vals)
        source = {
            "mean": float(w @ vals),
            "median": float(vals[o][np.searchsorted(np.cumsum(w[o]), 0.5)]),
            "p95": float(vals[o][min(len(vals) - 1, np.searchsorted(np.cumsum(w[o]), 0.95))]),
        }
        assert stats == source
    with pytest.raises(ValueError):
        timing.weighted_stats([])


def test_variant_order_shuffle_seeded():
    for group in range(4):
        for repeat in range(2):
            expected = list(variants.VARIANTS)
            random.Random(42 + 2 * group + repeat).shuffle(expected)
            assert timing.variant_order(variants.VARIANTS, group, repeat) == expected
    orders = {tuple(timing.variant_order(variants.VARIANTS, g, r)) for g in range(4)
              for r in range(2)}  # fmt: skip
    assert len(orders) > 1
    handal = ["singleview", "singleheatmap"]
    random.Random(42 + 2 * 5 + 1).shuffle(handal)
    assert timing.variant_order(("singleview", "singleheatmap"), 5, 1) == handal


def _record(group, repeat, variant, predictions, wall=1.0, weight=0.5):
    return {"group": group, "repeat": repeat, "variant": variant, "predictions": predictions,
            "wall_s": wall, "weight": weight}  # fmt: skip


def test_variants_of_a_repeat_must_agree_on_prediction_counts():
    agree = [_record(g, r, v, 3 + g, wall=1.0 + g + r)
             for g in range(2) for r in range(2) for v in ("singleview", "multiview")]  # fmt: skip
    timing.check_prediction_counts(agree)
    summary = timing.summarize(agree)
    assert set(summary) == {"singleview", "multiview"}
    assert summary["singleview"]["requests"] == 4 and summary["singleview"]["reference_groups"] == 2
    assert summary["singleview"]["mean_predictions"] == pytest.approx(3.5)
    assert summary["singleview"]["mean_s"] == pytest.approx(2.0)
    disagree = [*agree[:-1], _record(1, 1, "multiview", 9)]
    with pytest.raises(ValueError, match="prediction counts"):
        timing.check_prediction_counts(disagree)
    with pytest.raises(ValueError, match="prediction counts"):
        timing.summarize(disagree)


def test_prediction_counts_precede_the_depth_filter(monkeypatch):
    """A PnP pose with t_z <= 0 whose fused pose is valid: rows differ, the counts do not."""
    behind = _detection(REF)
    behind.tvec = behind.tvec * np.array([[1.0], [1.0], [-1.0]])
    journal = {REF: _view(REF, [behind, _detection(REF, score=0.5)]),
               SUP: _view(SUP, [_detection(SUP)])}  # fmt: skip
    good = _detection(REF)
    fused = [
        FusedPose(reference_index=j, score=s, R=cv2.Rodrigues(good.rvec)[0],
                  t=np.asarray(good.tvec, float).ravel(), matched=((SUP, 0),), accepted=True,
                  reprojection_px=(), observations=0)
        for j, s in ((0, 0.9), (1, 0.5))
    ]  # fmt: skip
    monkeypatch.setattr(variants, "fuse_reference", lambda *args, **kwargs: fused)
    records = []
    for name in ("singleview", "multiview"):
        views = (REF, SUP) if name == "multiview" else (REF,)
        task = timing._Task(name, REF, views, journal, _object_keypoints(),
                            _variant_settings(name), TLESS)  # fmt: skip
        composed = timing._compose(task)
        records.append({"group": 0, "repeat": 0, "variant": name, "weight": 1.0, "wall_s": 1.0,
                        "predictions": composed.predictions, "rows": composed.rows})  # fmt: skip
    assert [r["rows"] for r in records] == [1, 2]
    assert [r["predictions"] for r in records] == [2, 2]
    timing.check_prediction_counts(records)
    assert timing.summarize(records)["singleview"]["mean_predictions"] == 2.0


def test_native_views_are_rectified_inside_the_span():
    from test_bop_hot3d import distorted_calibration

    from doper.bop import hot3d

    raw = distorted_calibration()
    native = np.arange(16 * 20, dtype=np.uint8).reshape(16, 20)
    map_x, map_y, intrinsics = hot3d.rectification_maps(raw)
    expected, _ = hot3d.rectify(native, raw)
    np.testing.assert_array_equal(timing.rectify_native(native, (map_x, map_y)), expected)
    key = ImageKey(7, 3, "214-1")
    seen = []

    def detect(image, camera_matrix, raw_outputs):
        seen.append((image.shape, image.dtype, image.copy()))
        return []

    model = timing.TimedModel(obj_id=1, detect=detect, keypoints_mm=_object_keypoints(),
                              crop_size=CROP_SIZE, beta=1.0, checkpoint_sha256="c" * 64,
                              keypoints_sha256="k" * 64, keypoint_head="heatmap")  # fmt: skip
    settings = timing.DeploymentSettings(
        dataset="handal", refine=RefineSettings(optimizer="powell"), spec=None,
        cameras={key: {"cam_K": intrinsics.ravel().tolist()}}, images={key: native},
        maps={key: (map_x, map_y)}, device="cpu", variants=("singleview",),
    )  # fmt: skip
    runner = timing._Runner([model], settings, pool=None, synchronize=lambda: None)  # type: ignore[arg-type]
    # The settings hold only the native image: the network sees the rectified one, so the
    # remap ran inside ``request``, between the span's start and the inference.
    record = runner.request("singleview", 0, timing.WeightedGroup(key, (key,), 1.0), 0)
    assert seen[0][0] == (16, 20, 3) and seen[0][1] == np.uint8
    np.testing.assert_array_equal(seen[0][2], expected)
    assert 0 < record["preprocess_s"] < record["wall_s"]


def test_requests_are_journaled_as_they_complete(tmp_path):
    models = [_stand_in(1, [(REF, _detection(REF))])]
    groups = [timing.WeightedGroup(REF, (REF, SUP), 1.0)]
    path = tmp_path / "timing" / "requests.jsonl"
    path.parent.mkdir()
    path.write_text("stale\n")
    settings = dataclasses.replace(_settings(), variants=("singleview", "multiview"))
    result = timing.run(models, groups, settings, repetitions=1, workers=1, torch_threads=1,
                        requests_path=path)  # fmt: skip
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines == list(result.records)
    with pytest.raises(ValueError, match="workers"):
        timing.run(models, groups, settings, repetitions=1, workers=0, torch_threads=1)


# --- a run on CPU stand-in models ---------------------------------------------------------------


IMAGE = np.zeros((480, 640, 3), dtype=np.uint8)


def _stand_in(obj_id, detections):
    """A resident-model stand-in returning fixed detections per camera (heatmaps when raw)."""

    def detect(image, camera_matrix, raw):
        key = REF if np.allclose(camera_matrix, np.reshape(CAMERAS[REF]["cam_K"], (3, 3))) else SUP
        dets = [det for owner, det in detections if owner == key]
        if raw:
            return dets
        return [dataclasses.replace(d, keypoint_heatmap=None, crop_to_image=None) for d in dets]

    return timing.TimedModel(
        obj_id=obj_id,
        detect=detect,
        keypoints_mm=_object_keypoints(),
        crop_size=CROP_SIZE,
        beta=1.0,
        checkpoint_sha256=f"{obj_id:064d}",
        keypoints_sha256="k" * 64,
        keypoint_head="heatmap",
    )


def _settings(optimizer="powell"):
    return timing.DeploymentSettings(
        dataset="tless",
        refine=RefineSettings(optimizer=optimizer),
        spec=TLESS,
        cameras=CAMERAS,
        images={REF: IMAGE, SUP: IMAGE},
        device="cpu",
    )


def test_timing_manifest_records_optimizer_and_preparation():
    models = [_stand_in(1, [(REF, _detection(REF)), (SUP, _detection(SUP))]),
              _stand_in(2, [(REF, _detection(REF, score=0.7))])]  # fmt: skip
    groups = [timing.WeightedGroup(REF, (REF, SUP), 1.0)]
    result = timing.run(models, groups, _settings(), repetitions=2, workers=2, torch_threads=1)
    assert len(result.records) == 2 * len(variants.VARIANTS)
    assert [r["variant"] for r in result.records[:4]] == timing.variant_order(
        variants.VARIANTS, 0, 0
    )
    for record in result.records:
        assert record["wall_s"] > 0 and record["weight"] == 1.0
        assert record["predictions"] == 2 and record["rows"] == 2
        assert record["views"] == (2 if record["variant"] in variants.MULTIVIEW_VARIANTS else 1)
    manifest = result.manifest
    assert manifest["optimizer"] == "powell" and manifest["preparation"] == "numpy"
    assert manifest["optimizers_run"] == ["powell"]
    assert manifest["threads"] == RefineSettings().threads
    assert manifest["protocol"] == timing.PROTOCOL
    assert manifest["repetitions"] == 2 and manifest["workers"] == 2
    assert manifest["torch_threads"] == 1 and manifest["worker_threads"] == 1
    assert manifest["worker_thread_env"] == dict.fromkeys(timing.WORKER_THREAD_ENV, "1")
    assert manifest["worker_numba_threads"] in (1, None)
    assert manifest["hardware"]["device"] == "cpu"
    assert manifest["versions"]["opencv"] == cv2.__version__ and "torch" in manifest["versions"]
    assert [m["object_id"] for m in manifest["models"]] == [1, 2]
    assert manifest["complete"] is True and manifest["variants"] == list(variants.VARIANTS)
    assert set(result.summary) == set(variants.VARIANTS)
    # The declared optimizer must be the one that ran (the export runner pitfall).
    with pytest.raises(ValueError, match="declares"):
        timing.timing_manifest(
            _settings(), groups, models, records=result.records, optimizers_run=["lbfgsb"],
            repetitions=2, workers=2, torch_threads=1, hardware={"device": "cpu"},
        )  # fmt: skip


def test_run_refuses_heatmap_variants_without_a_heatmap_head():
    model = _stand_in(1, [(REF, _detection(REF))])
    plain = timing.TimedModel(**{**vars(model), "keypoint_head": "simcc"})
    groups = [timing.WeightedGroup(REF, (REF, SUP), 1.0)]
    with pytest.raises(ValueError, match="heatmap keypoint head"):
        timing.run([plain], groups, _settings(), repetitions=1, workers=1, torch_threads=1)


@pytest.mark.gpu
def test_deployment_timing_gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from tiny_coco import tiny_model_config

    from doper.config import DataConfig, DoperConfig
    from doper.model import DoperNet

    config = tiny_model_config(num_classes=1, num_keypoints=6, det_input_size=96, letterbox=True,
                               neck_channels=8, head_channels=8, crop_size=32,
                               keypoint_head="heatmap")  # fmt: skip
    cfg = DoperConfig(data=DataConfig(root="."), model=config)
    torch.manual_seed(0)
    net = DoperNet(config)
    with torch.no_grad():
        net.wh_head.bias.add_(6.0)
    net = net.to("cuda:0").eval()
    keypoints = np.random.default_rng(0).uniform(-50.0, 50.0, (6, 3))
    model = timing.timed_model(1, net, cfg, keypoints, device="cuda:0", checkpoint_sha256="c" * 64,
                               keypoints_sha256="k" * 64, score_threshold=0.0)  # fmt: skip
    key = ImageKey(1, 0)
    camera = {"cam_K": [115.2, 0.0, 48.0, 0.0, 115.2, 48.0, 0.0, 0.0, 1.0]}
    image = np.random.default_rng(1).integers(0, 256, (96, 128, 3), dtype=np.uint8)
    settings = timing.DeploymentSettings(
        dataset="handal", refine=RefineSettings(optimizer="powell"), spec=None,
        cameras={key: camera}, images={key: image}, device="cuda:0",
    )  # fmt: skip
    result = timing.run([model], [timing.WeightedGroup(key, (key,), 1.0)], settings,
                        repetitions=2, workers=2, torch_threads=4)  # fmt: skip
    assert len(result.records) == 4 and all(r["wall_s"] > 0 for r in result.records)
    assert result.manifest["hardware"]["gpu_name"]
    assert set(result.summary) == {"singleview", "singleheatmap"}


def test_handal_prediction_counts_follow_the_validity_skip():
    """HANDAL (no multi-view policy) never composes a detection whose PnP pose has t_z <= 0, as
    the source HANDAL follow-up rejected nonpositive initial depth in both arms; so its count
    is taken after that skip, unlike the multi-view datasets'."""
    behind = _detection(REF)
    behind.tvec = behind.tvec * np.array([[1.0], [1.0], [-1.0]])
    journal = {REF: _view(REF, [behind, _detection(REF, score=0.5)])}
    for name in ("singleview", "singleheatmap"):
        task = timing._Task(name, REF, (REF,), journal, _object_keypoints(),
                            _variant_settings(name), None)  # fmt: skip
        composed = timing._compose(task)
        assert (composed.predictions, composed.rows) == (1, 1)
    assert "HANDAL" in timing.PROTOCOL["prediction_count"]
    assert "after" in timing.PROTOCOL["prediction_count"]
