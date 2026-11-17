# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``doper.reports.bop``: appendix A tables, fixed-policy rows, Table 3 variants and AP.

The expected macro values restate the research reports' aggregation: a comparison's population
is the objects whose every method has a single-pass score; skipped and missing students are
N/A (never zero); paired deltas count improvements, regressions and ties at 1e-10. The parity
harness case ``report_appendix`` runs the research report functions themselves on the same
kind of tree.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from doper.recipes.schema import load_recipe
from doper.reports import bop

HOPE_METHODS = (
    "static_pseudo",
    "dynamic_pseudo",
    "combined_pseudo",
    "gt_box_mask",
    "gt_box_mask_keypoints",
)


def _arm(value: float) -> dict:
    return {
        "frames": 4,
        "gt_instances": 4,
        "ar_mssd": value,
        "ar_mspd": value,
        "ar_mean": value,
        "mssd_recalls": [],
        "mspd_recalls": [],
        "pose_rate": None,
    }


def _arms(single: float) -> dict:
    return {"single": _arm(single), "jitter": _arm(single + 1), "gtbox": _arm(single + 2)}


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _scored(runs: Path, method: str, obj: int, single: float, devices: dict | None = None):
    run = runs / method / f"obj_{obj:06d}"
    _write(
        run / "validation" / "ar.json",
        {
            "obj_id": obj,
            "protocol": "p",
            "camera": "pinhole",
            "status": "scored",
            "population_sha256": "x",
            "frames": 4,
            "gt_instances": 4,
            "arms": _arms(single),
            "by_device": {d: _arms(v) for d, v in (devices or {}).items()},
        },
    )
    return run


def _adapted(runs: Path, method: str, obj: int, labels: int, single: float, **kwargs) -> None:
    run = _scored(runs, method, obj, single, **kwargs)
    _write(run / "labels_ready.json", {"labels": labels})
    updates = 3 * math.ceil(labels / 4)
    _write(run / "completed.json", {"labels": labels, "global_step": updates})


def _skipped(runs: Path, method: str, obj: int, labels: int) -> None:
    run = runs / method / f"obj_{obj:06d}"
    _write(run / "labels_ready.json", {"labels": labels})
    _write(run / "adaptation_skipped.json", {"reason": "below", "labels": labels})


@pytest.fixture
def hope_runs(tmp_path: Path) -> Path:
    """Three objects: all arms, a skipped static arm, and missing GT arms."""
    runs = tmp_path / "runs"
    for obj, base in ((1, 10.0), (2, 12.0), (3, 14.0)):
        _write(runs / "base" / f"obj_{obj:06d}" / "completed.json", {"completed_epochs": 25})
        _scored(runs, "base", obj, base)
    for method, value in zip(HOPE_METHODS, (20.0, 30.0, 40.0, 50.0, 60.0), strict=True):
        _adapted(runs, method, 1, 21 if method.endswith("pseudo") else 24, value)
    _skipped(runs, "static_pseudo", 2, 5)
    for method, value in (("dynamic_pseudo", 32.0), ("combined_pseudo", 42.0)):
        _adapted(runs, method, 2, 30, value)
    _adapted(runs, "gt_box_mask", 2, 8, 52.0)
    _adapted(runs, "gt_box_mask_keypoints", 2, 8, 66.0)
    for method, value in (("static_pseudo", 24), ("dynamic_pseudo", 34), ("combined_pseudo", 44)):
        _adapted(runs, method, 3, 40, float(value))
    return runs


def _means(single: float) -> dict:
    return {"single": single, "jitter": single + 1, "gtbox": single + 2}


def test_appendix_rows_match_source_reports(hope_runs):
    result = bop.appendix(hope_runs, HOPE_METHODS)
    rows = {(r["object"], r["method"]): r for r in result["rows"]}
    assert len(rows) == 18
    assert rows[(1, "base")]["status"] == "base" and rows[(1, "base")]["labels"] is None
    assert rows[(1, "static_pseudo")] | {} == {
        "object": 1,
        "method": "static_pseudo",
        "status": "completed",
        "labels": 21,
        "updates": 18,
        "single_AR": 20.0,
        "jitter_AR": 21.0,
        "gtbox_AR": 22.0,
        "by_device": {},
    }
    skipped = rows[(2, "static_pseudo")]
    assert (skipped["status"], skipped["labels"], skipped["updates"]) == ("skipped", 5, None)
    assert skipped["single_AR"] is None and skipped["gtbox_AR"] is None
    missing = rows[(3, "gt_box_mask")]
    assert (missing["status"], missing["labels"], missing["single_AR"]) == ("missing", None, None)
    macro = result["macro"]
    assert list(macro) == [
        "all_common",
        "GT_paired",
        "four_way_paired",
        *(f"{m}_paired" for m in HOPE_METHODS),
    ]
    assert macro["all_common"]["objects"] == [1] and macro["all_common"]["count"] == 1
    assert macro["all_common"]["methods"]["combined_pseudo"] == _means(40.0)
    assert macro["GT_paired"]["objects"] == [1, 2]
    assert macro["GT_paired"]["methods"] == {
        "base": _means(11.0),
        "gt_box_mask": _means(51.0),
        "gt_box_mask_keypoints": _means(63.0),
    }
    assert macro["four_way_paired"]["objects"] == [1]
    assert macro["static_pseudo_paired"]["objects"] == [1, 3]
    assert macro["static_pseudo_paired"]["methods"]["static_pseudo"] == _means(22.0)
    assert macro["dynamic_pseudo_paired"]["methods"]["base"] == _means(12.0)
    assert result["paired_deltas"]["gt_box_mask_keypoints"] == {
        "objects": [1, 2],
        "count": 2,
        "mean_delta": 52.0,
        "median_delta": 52.0,
        "improve": 2,
        "regress": 0,
        "ties": 0,
    }
    assert result["added_GT_keypoints"]["mean_delta"] == 12.0
    assert result["by_device_macro"] == {}


def test_appendix_by_device_and_empty_population(tmp_path):
    runs = tmp_path / "runs"
    _scored(runs, "base", 1, 10.0, devices={"Aria": 5.0, "Quest3": 7.0})
    _scored(runs, "base", 2, 20.0, devices={"Aria": 9.0})
    _adapted(runs, "gt_box_mask", 1, 4, 30.0, devices={"Aria": 15.0, "Quest3": 17.0})
    _adapted(runs, "gt_box_mask", 2, 4, 40.0, devices={"Aria": 19.0})
    result = bop.appendix(runs, ("gt_box_mask", "gt_box_mask_keypoints"))
    aria = result["by_device_macro"]["Aria"]["gt_box_mask_paired"]
    assert aria["objects"] == [1, 2] and aria["methods"]["gt_box_mask"] == _means(17.0)
    quest = result["by_device_macro"]["Quest3"]["gt_box_mask_paired"]
    assert quest["objects"] == [1] and quest["methods"]["base"] == _means(7.0)
    empty = result["macro"]["GT_paired"]
    assert empty["count"] == 0 and empty["methods"]["gt_box_mask"]["single"] is None
    assert result["added_GT_keypoints"]["mean_delta"] is None


def test_appendix_marks_a_partial_diagnostic_unavailable(hope_runs, tmp_path):
    path = hope_runs / "base" / "obj_000001" / "validation" / "ar.json"
    result = json.loads(path.read_text())
    del result["arms"]["jitter"]
    path.write_text(json.dumps(result))
    macro = bop.appendix(hope_runs, HOPE_METHODS)["macro"]
    assert "paired diagnostic" in macro["GT_paired"]["unavailable"]
    assert macro["GT_paired"]["methods"] == {} and macro["GT_paired"]["objects"] == [1, 2]
    assert "paired diagnostic" in macro["static_pseudo_paired"]["unavailable"]  # object 1
    _, md_path = bop.write_report(load_recipe("hopev2"), hope_runs, tmp_path / "out")
    assert "Unavailable: paired diagnostic" in md_path.read_text()


def _csv(path: Path, time: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scene_id", "im_id", "obj_id", "score", "R", "t", "time"])
        writer.writerow([1, 1, 1, 0.9, "1 0 0 0 1 0 0 0 1", "0 0 1", time])


def _export(runs: Path, policy: str, name: str, time: float, sub: str = "") -> Path:
    folder = runs / "test" / policy / sub
    _csv(folder / name, time)
    _write(
        folder / "summary.json",
        {
            "csv_path": str(Path("/elsewhere") / name),
            "images": 3,
            "objects": [1, 2],
            "missing_objects": [],
            "pose_rows": 5,
            "policy": policy,
        },
    )
    return folder


def test_policy_variant_ap_and_timing_rows(tmp_path):
    recipe = load_recipe("hopev2")
    runs = tmp_path / "runs"
    static = _export(runs, "static_pseudo", "a_hopev2-test.csv", -1)
    _write(static / "SKIPS.json", {"method": "static_pseudo", "cases": [{"object_id": 2}]})
    _export(runs, "gt_box_mask", "b_hopev2-test.csv", 0.25)
    variant = _export(runs, "base", "c_hopev2-test.csv", -1, sub="multiheatmap")
    _write(variant / "variants_manifest.json", {"optimizer": "powell", "preparation": "numpy"})
    _write(
        runs / "timing" / "base" / "deployment" / "summary.json",
        {"optimizer": "powell", "variants": {"multiheatmap": {"mean_s": 0.5, "median_s": 0.4,
                                                              "p95_s": 0.9, "requests": 7}}},
    )  # fmt: skip
    _write(runs / "timing" / "gt_box_mask" / "direct" / "summary.json", {"mean_seconds": 0.3})
    _write(
        runs / "ap_hopev2_static_pseudo.json",
        {"ap": 0.5, "ap_mssd": 0.4, "ap_mspd": 0.6, "targets_origin": "derived_from_bop19"},
    )
    _write(
        runs / "ap_hopev2_base_multiheatmap.json",
        {"ap": 0.7, "ap_mssd": 0.6, "ap_mspd": 0.8, "targets_origin": "bop24"},
    )
    summary = bop.summarize(recipe, runs)
    policies = {r["policy"]: r for r in summary["policies"]}
    assert policies["static_pseudo"]["base_fallbacks"] == [2]
    assert policies["static_pseudo"]["timed"] is False
    assert policies["static_pseudo"]["ap"]["targets_origin"] == "derived_from_bop19"
    assert policies["gt_box_mask"]["timed"] is True
    assert policies["gt_box_mask"]["direct_timing"]["mean_seconds"] == 0.3
    assert "mixed timing" in summary["timing_note"]
    assert "gt_box_mask" in summary["timing_note"]
    (row,) = summary["variants"]
    assert (row["policy"], row["variant"], row["optimizer"]) == ("base", "multiheatmap", "powell")
    assert row["deployment_timing"]["median_s"] == 0.4
    assert row["ap"]["ap"] == 0.7
    text = bop.markdown(summary)
    assert "| 0.7000 (bop24) |" in text
    for heading in ("## Appendix A", "## Fixed policies", "## Table 3 variants"):
        assert heading in text
    assert "mixed timing" in text and "| static_pseudo |" in text


def test_timed_csv_needs_a_uniform_time_column(tmp_path):
    path = tmp_path / "a.csv"
    _csv(path, -1)
    with path.open("a", newline="") as stream:
        csv.writer(stream).writerow([1, 2, 1, 0.8, "1 0 0 0 1 0 0 0 1", "0 0 1", 0.5])
    with pytest.raises(ValueError, match="mixes untimed"):
        bop._is_timed(path)
    path.write_text("scene_id,im_id\n1,1\n")
    with pytest.raises(ValueError, match="no time column"):
        bop._is_timed(path)
    path.write_text("scene_id,im_id,time\n")
    assert bop._is_timed(path) is None and bop._is_timed(tmp_path / "absent.csv") is None


def test_followup_rows(tmp_path):
    runs = tmp_path / "runs"
    _write(
        runs / "obj8_loop" / "progress.json",
        {
            "followup": "obj8_loop",
            "obj_id": 8,
            "finished": True,
            "stop_reason": "target_reached",
            "final_student_yield": 25,
            "rounds": [
                {"round": 1, "status": "completed", "labels": 9},
                {"round": 2, "status": "completed", "labels": 25},
            ],
        },
    )
    rows = bop.followup_rows(runs, load_recipe("handal"))
    assert rows == [
        {"followup": "obj8_loop", "object": 8, "round": 1, "status": "completed", "labels": 9,
         "stop_reason": "target_reached", "final_student_yield": 25},
        {"followup": "obj8_loop", "object": 8, "round": 2, "status": "completed", "labels": 25,
         "stop_reason": "target_reached", "final_student_yield": 25},
    ]  # fmt: skip


def test_write_report_keeps_the_validation_table(hope_runs, tmp_path):
    json_path, md_path = bop.write_report(load_recipe("hopev2"), hope_runs, tmp_path / "out")
    summary = json.loads(json_path.read_text())
    assert summary["appendix"]["macro"]["GT_paired"]["count"] == 2
    assert summary["pooled_cross_dataset_score"] is None
    text = md_path.read_text()
    assert "| 2 | static_pseudo | skipped | 5 | - | - | - | - |" in text
    assert "GT_paired: 2 objects" in text


def _audit_pool(run: Path, gt_size: int = 4) -> None:
    import cv2
    import numpy as np

    pool = run / "labels"
    (pool / "train").mkdir(parents=True)
    record = {"source_relative_path": "s/rgb/000001.jpg", "scene": "s", "frame": 1,
              "image": "1.jpg", "score": 0.7, "reproj_px": 2.0, "inliers": 60}  # fmt: skip
    _write(pool / "manifest.json", {"records": [record]})
    coco = {"images": [{"id": 1, "file_name": "1.jpg"}],
            "annotations": [{"image_id": 1, "mask_file": "mask.png"}]}  # fmt: skip
    _write(pool / "train" / "annotations.json", coco)
    cv2.imwrite(str(pool / "train" / "mask.png"), np.full((gt_size, gt_size), 255, np.uint8))
    _write(run / "labels_ready.json", {"labels": 1, "labels_root": "labels"})


def test_report_mask_audit(tmp_path, monkeypatch, capsys):
    import zipfile

    import cv2
    import numpy as np

    from doper import cli
    from doper.cli import report

    runs, out = tmp_path / "runs", tmp_path / "out"
    _audit_pool(runs / "obj8_loop" / "round01" / "obj_000008")
    _write(runs / "obj8_loop" / "progress.json",
           {"rounds": [{"round": 1, "run": "obj8_loop/round01/obj_000008"},
                       {"round": 2, "run": "obj8_loop/round02/obj_000008"}]})  # fmt: skip
    _write(runs / "obj7_bootstrap" / "progress.json", {"rounds": []})
    archive = tmp_path / "static.zip"
    with zipfile.ZipFile(archive, "w") as z:
        mask = cv2.imencode(".png", np.full((4, 4), 255, np.uint8))[1].tobytes()
        z.writestr("static/s/mask_visib/000001_000000.png", mask)
    audit = ["report", "handal", "--runs", str(runs), "--out", str(out), "--mask-audit"]
    explicit = [*audit, "--onboarding-zip", str(archive)]
    assert cli.main(audit) == 1  # neither --root nor --onboarding-zip
    assert cli.main(["report", "hopev2", "--runs", str(runs), *explicit[4:]]) == 1  # no follow-ups
    assert cli.main([*explicit, "--ap"]) == 1
    assert cli.main(explicit) == 1 and "SHA-256" in capsys.readouterr().out  # not the pinned zip
    hashed = []
    pin = report._static_archive_name(load_recipe("handal"))[1]
    monkeypatch.setattr(report, "sha256_file", lambda path: hashed.append(path) or pin)
    assert cli.main(explicit) == 0
    assert hashed == [archive]  # hashed once for both follow-ups
    result = json.loads((out / "mask_audit" / "obj8_loop.json").read_text())
    round1, round2 = result["rounds"]
    assert round1["round"] == 1 and round1["status"] == "audited"
    assert round1["mean_mask_iou"] == 1.0 and "diagnostic only" in result["protocol"]
    assert round2 == {"round": 2, "run": "obj8_loop/round02/obj_000008", "status": "missing"}
    bootstrap = json.loads((out / "mask_audit" / "obj7_bootstrap.json").read_text())
    assert bootstrap["rounds"] == []


def test_report_warns_when_a_policy_mixes_forward_passes(tmp_path):
    recipe = load_recipe("hopev2")
    runs = tmp_path / "runs"

    def generations(folder, table):
        summary = json.loads((folder / "summary.json").read_text())
        _write(folder / "summary.json", {**summary, "journal_generations": table})

    generations(_export(runs, "base", "a_hopev2-test.csv", -1), {"1": ["A"], "2": ["A2"]})
    single = _export(runs, "base", "b_hopev2-test.csv", -1, sub="singleview")
    generations(single, {"1": ["A"], "2": ["A2"]})
    heatmap = _export(runs, "base", "c_hopev2-test.csv", -1, sub="singleheatmap")
    generations(heatmap, {"1": ["A"], "2": ["A2"]})
    summary = bop.summarize(recipe, runs)
    assert summary["generation_warnings"] == []
    generations(heatmap, {"1": ["B"], "2": ["A2"]})
    multi = _export(runs, "base", "d_hopev2-test.csv", -1, sub="multiview")
    generations(multi, {"1": ["A", "S"], "2": ["A2", "S2"]})
    multiheatmap = _export(runs, "base", "e_hopev2-test.csv", -1, sub="multiheatmap")
    generations(multiheatmap, {"1": ["B", "S"], "2": ["A2", "T2"]})
    (warning,) = bop.summarize(recipe, runs)["generation_warnings"]
    assert warning.startswith("base:") and "object 1" in warning and "object 2" in warning
    assert "singleheatmap" in warning and "multiview" in warning
    assert "Warning: base:" in bop.markdown(bop.summarize(recipe, runs))
