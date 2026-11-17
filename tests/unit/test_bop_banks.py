# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-policy banks: recipe policy tables, base fallbacks, manifests and the journal cache."""

import csv
import json
import subprocess
import sys

import cv2
import numpy as np
import pytest
from bank_runs import (
    raw_record,
    sha256,
    write_base,
    write_skip,
    write_student,
    write_test_split,
    write_tree,
)

from doper.bop import banks, export, layout
from doper.infer import Detection
from doper.recipes.schema import Policy, load_recipe, parse_recipe

HANDAL_METHODS = ("static_pseudo", "gt_box_mask", "gt_box_mask_keypoints", "test_tta")
HOPE_METHODS = (
    "static_pseudo",
    "dynamic_pseudo",
    "combined_pseudo",
    "gt_box_mask",
    "gt_box_mask_keypoints",
)
# The source plans' skips: HANDAL static {7, 8, 40} and test TTA {26}; HOPE static on ten
# objects; HOT3D static 3, dynamic 8 and combined 3 objects (pseudo methods below 20 labels).
SOURCE_PLANS = {
    "handal": (
        range(1, 41),
        HANDAL_METHODS,
        {"static_pseudo": {7: 3, 8: 6, 40: 17}, "test_tta": {26: 12}},
    ),
    "hopev2": (
        range(1, 29),
        HOPE_METHODS,
        {"static_pseudo": {o: o % 20 for o in (2, 5, 9, 11, 14, 17, 20, 23, 26, 28)}},
    ),
    "hot3d": (
        range(1, 34),
        HOPE_METHODS,
        {
            "static_pseudo": {4: 11, 19: 0, 30: 19},
            "dynamic_pseudo": {o: 2 for o in (3, 4, 9, 12, 19, 25, 30, 31)},
            "combined_pseudo": {4: 13, 19: 5, 30: 19},
        },
    ),
}
# Fallback mode and BOP CSV method token of each recipe policy (source build_plans).
POLICY_TABLES = {
    "handal": {
        "static_pseudo": ("skipped", "doperstaticfallback"),
        "gt_box_mask": ("none", "dopergtboxmask"),
        "gt_box_mask_keypoints": ("none", "dopergtboxmaskkeypoints"),
        "test_tta": ("skipped", "dopertestttafallback"),
    },
    "hopev2": {
        "static_pseudo": ("skipped", "doperstaticfallback"),
        "dynamic_pseudo": ("none", "doperdynamic"),
        "combined_pseudo": ("none", "dopercombined"),
        "gt_box_mask": ("none", "dopergtboxmask"),
        "gt_box_mask_keypoints": ("none", "dopergtboxmaskkeypoints"),
    },
    "hot3d": {
        "static_pseudo": ("skipped", "doperstaticpseudofallback"),
        "dynamic_pseudo": ("skipped", "doperdynamicpseudofallback"),
        "combined_pseudo": ("skipped", "dopercombinedpseudofallback"),
        "gt_box_mask": ("none", "dopergtboxmask"),
        "gt_box_mask_keypoints": ("none", "dopergtboxmaskkeypoints"),
    },
}


@pytest.mark.parametrize("name", sorted(POLICY_TABLES))
def test_recipe_policy_tables(name):
    recipe = load_recipe(name)
    table = {
        p.name: (p.fallback, p.csv_method)
        for p in recipe.test.policies.values()
        if p.objects is None
    }
    assert table == POLICY_TABLES[name]
    assert all(p.method == p.name for p in recipe.test.policies.values() if p.objects is None)
    assert recipe.test.record_time is (name == "hot3d")
    assert banks.base_policy(recipe) == Policy("base", "base", "doperbase", "none", None)


def test_policy_table_is_validated():
    recipe = load_recipe("hopev2")
    data = {
        "name": "x",
        "profile": "hopev2",
        "objects": [1, 2],
        "base": {"config": dict(recipe.base.config)},
    }
    adapt = {
        "config": dict(recipe.adapt.config),
        "methods": {"static_pseudo": {"source": "static", "minimum_labels": 20}},
    }
    good = {"method": "static_pseudo", "csv_method": "doperstatic", "fallback": "skipped"}
    parse_recipe({**data, "adapt": adapt, "test": {"policies": {"static_pseudo": good}}})
    bad = [
        ({"selected": good}, "built-in"),
        ({"p": {**good, "csv_method": "doper_static"}}, "alphanumeric"),
        ({"p": {**good, "method": "dynamic_pseudo"}}, "not a"),
        ({"p": {**good, "fallback": "maybe"}}, "fallback"),
        ({"p": {**good, "fallback": "out_of_scope"}}, "objects"),
        ({"p": {**good, "objects": [1]}}, "objects"),
        ({"p": good, "q": good}, "unique"),
        ({"p": {**good, "csv_method": "doperbase"}}, "unique"),
        ({"Static-Pseudo": good}, "name"),
        ({"../x": good}, "name"),
        ({"cache": good}, "reserved"),
    ]
    for policies, message in bad:
        with pytest.raises(ValueError, match=message):
            parse_recipe({**data, "adapt": adapt, "test": {"policies": policies}})
    alias = {"run": "static_pseudo", "objects": [1]}
    parse_recipe({**data, "adapt": adapt, "test": {"aliases": {"renamed": alias}}})
    bad_aliases = [
        ({"Renamed": alias}, "name"),
        ({"cache": alias}, "reserved"),
        ({"base": alias}, "shadows"),
        ({"renamed": {**alias, "run": "base"}}, "base"),
        ({"renamed": {**alias, "run": "../base"}}, "relative"),
    ]
    for aliases, message in bad_aliases:
        with pytest.raises(ValueError, match=message):
            parse_recipe({**data, "adapt": adapt, "test": {"aliases": aliases}})


def test_banks_do_not_import_the_inference_stack():
    code = "import sys, doper.bop.banks; assert not {'cv2', 'torch'} & set(sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("name", sorted(SOURCE_PLANS))
def test_fixed_bank_matches_source_plans(name, tmp_path):
    objects, methods, skips = SOURCE_PLANS[name]
    write_tree(tmp_path, objects, methods, skips)
    recipe = load_recipe(name)
    primary = [p for p in recipe.test.policies.values() if p.objects is None]
    for policy in [banks.base_policy(recipe), *primary]:
        bank, fallbacks = banks.fixed_bank(policy, tmp_path, list(objects))
        skipped = skips.get(policy.method, {})
        assert sorted(bank) == list(objects)
        assert [(f.obj_id, f.method, f.reason, f.labels) for f in fallbacks] == [
            (o, policy.method, "skipped", n) for o, n in sorted(skipped.items())
        ]
        for obj, run in bank.items():
            arm = "base" if obj in skipped else policy.method
            assert run == layout.object_run(tmp_path, arm, obj)
        for fallback in fallbacks:
            base = layout.object_run(tmp_path, "base", fallback.obj_id)
            assert fallback.checkpoint_sha256 == sha256(base / layout.CHECKPOINT_PATH)
    expected = {"handal": {7, 8, 40}, "hopev2": {2, 5, 9, 11, 14, 17, 20, 23, 26, 28}}
    if name in expected:
        static = recipe.test.policies["static_pseudo"]
        _, fallbacks = banks.fixed_bank(static, tmp_path, list(objects))
        assert {f.obj_id for f in fallbacks} == expected[name]


def test_handal_followup_policies_change_only_their_object(tmp_path):
    # Source handal_all_adapt_test: ten single-object follow-up banks over 39 original bases.
    objects, methods, skips = SOURCE_PLANS["handal"]
    write_tree(tmp_path, objects, methods, skips)
    write_student(tmp_path, "obj7_bootstrap/round01", 7, labels=35)
    for number in range(1, 9):
        write_student(tmp_path, f"obj8_loop/round{number:02d}", 8, labels=6)
    write_student(tmp_path, "combined_pseudo", 8, labels=6)
    recipe = load_recipe("handal")
    scoped = [p for p in recipe.test.policies.values() if p.fallback == "out_of_scope"]
    assert len(scoped) == 10
    for policy in scoped:
        (obj,) = policy.objects
        bank, fallbacks = banks.fixed_bank(policy, tmp_path, list(objects))
        assert bank[obj] == layout.object_run(tmp_path, policy.method, obj)
        assert [f.obj_id for f in fallbacks] == [o for o in objects if o != obj]
        assert {f.reason for f in fallbacks} == {"out_of_scope"}
        assert all(bank[o] == layout.object_run(tmp_path, "base", o) for o in objects if o != obj)


def test_fallback_must_equal_base(tmp_path):
    for obj in (1, 2):
        write_base(tmp_path, obj)
    write_student(tmp_path, "static_pseudo", 1)
    write_skip(tmp_path, "static_pseudo", 2, labels=4)
    skipped = Policy("static_pseudo", "static_pseudo", "doperstatic", "skipped", None)
    strict = Policy("static_pseudo", "static_pseudo", "doperstatic", "none", None)
    bank, fallbacks = banks.fixed_bank(skipped, tmp_path, [1, 2])
    assert bank[2] == layout.object_run(tmp_path, "base", 2)
    assert fallbacks[0].checkpoint_sha256 == sha256(bank[2] / layout.CHECKPOINT_PATH)
    with pytest.raises(ValueError, match="allows no fallback"):
        banks.fixed_bank(strict, tmp_path, [1, 2])
    base_weights = layout.object_run(tmp_path, "base", 2) / layout.CHECKPOINT_PATH
    original = base_weights.read_bytes()
    base_weights.write_bytes(b"retrained")
    with pytest.raises(ValueError, match="base"):
        banks.fixed_bank(skipped, tmp_path, [1, 2])
    base_weights.write_bytes(original)
    masked = layout.object_run(tmp_path, "static_pseudo", 2) / layout.CHECKPOINT_PATH
    masked.parent.mkdir()
    masked.write_bytes(b"a student behind a skip marker")
    with pytest.raises(ValueError, match="masks"):
        banks.fixed_bank(skipped, tmp_path, [1, 2])
    masked.unlink()
    marker = layout.object_run(tmp_path, "static_pseudo", 2) / layout.SKIPPED_NAME
    marker.write_text(json.dumps({"labels": 25, "minimum_labels": 20}))
    with pytest.raises(ValueError, match="minimum"):
        banks.fixed_bank(skipped, tmp_path, [1, 2])


def test_student_must_descend_from_the_base_and_be_finished(tmp_path):
    write_base(tmp_path, 1)
    run = write_student(tmp_path, "gt_box_mask", 1)
    policy = Policy("gt_box_mask", "gt_box_mask", "dopergtboxmask", "none", None)
    assert banks.fixed_bank(policy, tmp_path, [1])[0] == {1: run}
    done = json.loads((run / layout.COMPLETED_NAME).read_text())
    (run / layout.COMPLETED_NAME).write_text(json.dumps({**done, "base_teacher_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="another base"):
        banks.fixed_bank(policy, tmp_path, [1])
    (run / layout.COMPLETED_NAME).unlink()
    with pytest.raises(ValueError, match="unfinished"):
        banks.fixed_bank(policy, tmp_path, [1])
    write_base(tmp_path, 2)
    with pytest.raises(ValueError, match="no gt_box_mask endpoint"):
        banks.fixed_bank(policy, tmp_path, [2])
    with pytest.raises(ValueError, match="no completed base"):
        banks.fixed_bank(policy, tmp_path, [3])


def test_out_of_scope_policy_changes_only_its_object(tmp_path):
    for obj in (7, 8, 9):
        write_base(tmp_path, obj)
    loop = write_student(tmp_path, "obj8_loop/round03", 8, labels=6)
    policy = Policy(
        "obj8_static_round03", "obj8_loop/round03", "doperobj8r03", "out_of_scope", (8,)
    )
    bank, fallbacks = banks.fixed_bank(policy, tmp_path, [7, 8, 9])
    assert bank[8] == loop and bank[7] == layout.object_run(tmp_path, "base", 7)
    assert [(f.obj_id, f.reason, f.labels) for f in fallbacks] == [
        (7, "out_of_scope", None),
        (9, "out_of_scope", None),
    ]
    skipped = Policy("p", "static_pseudo", "doperp", "skipped", None)
    write_skip(tmp_path, "static_pseudo", 7, labels=0)
    with pytest.raises(ValueError, match="allows only out_of_scope"):
        banks.fixed_bank(
            Policy("p", "static_pseudo", "doperp", "out_of_scope", (7,)), tmp_path, [7]
        )
    assert banks.fixed_bank(skipped, tmp_path, [7])[1][0].reason == "skipped"


def test_bank_manifest_writes_matrix_and_skips(tmp_path):
    runs = tmp_path / "runs"
    for obj in (1, 2):
        write_base(runs, obj)
    write_student(runs, "static_pseudo", 1, labels=22)
    write_skip(runs, "static_pseudo", 2, labels=7)
    policy = Policy("static_pseudo", "static_pseudo", "doperstaticfallback", "skipped", None)
    bank, fallbacks = banks.fixed_bank(policy, runs, [1, 2])
    matrix, skips = banks.bank_manifest(tmp_path / "out", policy, bank, fallbacks, runs)
    with matrix.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [(r["object"], r["actual_method"], r["status"], r["labels"]) for r in rows] == [
        ("1", "static_pseudo", "adapted_student", "22"),
        ("2", "base", "base_fallback_skipped", "7"),
    ]
    assert rows[1]["checkpoint_sha256"] == rows[1]["base_checkpoint_sha256"]
    assert rows[0]["run"] == "static_pseudo/obj_000001" and rows[1]["run"] == "base/obj_000002"
    record = json.loads(skips.read_text())
    assert record["policy"] == "static_pseudo" and record["fallback"] == "skipped"
    assert [(c["object_id"], c["reason"], c["labels"]) for c in record["cases"]] == [
        (2, "skipped", 7)
    ]


def _key(**changes):
    values = {
        "checkpoint_sha256": "c" * 64,
        "keypoints_sha256": "k" * 64,
        "population_sha256": "p" * 64,
        "settings_sha256": export.cache_settings_sha256("pinhole", 0.15, 5),
    }
    return export.JournalKey(**{**values, **changes})


def test_journal_cache_key_changes_with_checkpoint_or_settings(tmp_path):
    cache = export.JournalCache(tmp_path / "cache")
    calls = []

    def fill(directory):
        calls.append(directory)
        directory.mkdir(parents=True, exist_ok=True)
        record = {"scene_id": 1, "im_id": len(calls), "predictions": []}
        (directory / export.JOURNAL_NAME).write_text(json.dumps(record) + "\n")

    first = cache.get_or_run(_key(), fill)
    assert cache.get_or_run(_key(), fill) == first and len(calls) == 1
    assert list(first) == [(1, 1)]
    changed = [
        _key(checkpoint_sha256="d" * 64),
        _key(keypoints_sha256="j" * 64),
        _key(population_sha256="q" * 64),
        _key(settings_sha256=export.cache_settings_sha256("pinhole", 0.2, 5)),
        _key(schema_version=export.CACHE_SCHEMA_VERSION + 1),
    ]
    for n, key in enumerate(changed, start=2):
        assert list(cache.get_or_run(key, fill)) == [(1, n)]
    assert len(calls) == 6 and len({c.parent for c in calls}) == 6
    entry = cache.entry(_key())
    assert entry.parent.name == f"v{export.CACHE_SCHEMA_VERSION}"
    receipt = json.loads((entry / export.CACHE_RECEIPT_NAME).read_text())
    assert receipt["key"]["schema_version"] == export.CACHE_SCHEMA_VERSION
    journal = entry / export.CACHE_PAYLOAD_DIR / export.JOURNAL_NAME
    journal.write_text(journal.read_text().replace('"im_id": 1', '"im_id": 9'))
    with pytest.raises(ValueError, match="cache"):
        cache.get_or_run(_key(), fill)


WIDTH, HEIGHT = 32, 24
CAMERA_K = [40.0, 0.0, 16.0, 0.0, 40.0, 12.0, 0.0, 0.0, 1.0]


def _test_split(root, frames=3):
    return write_test_split(root, frames, CAMERA_K, (WIDTH, HEIGHT))


def _pose(seed, score):
    rotation = cv2.Rodrigues(np.array([0.1 * seed, -0.2, 0.3]))[0]
    return {"score": score, "R": rotation.ravel().tolist(), "t": [seed, -seed, 500.0 + seed]}


def _oracle_journal(obj, frames):
    """Per-model journal records: ties, several poses and an empty frame."""
    records = []
    for frame in frames:
        poses = [] if (obj + frame.im_id) % 3 == 0 else [_pose(obj, 0.9), _pose(obj + 1, 0.5)]
        if obj == 2 and frame.im_id == 1:
            poses.append(_pose(7, 0.5))  # a score tie within one object keeps journal order
        records.append(
            {
                "scene_id": frame.scene_id,
                "im_id": frame.im_id,
                "rgb_sha256": frame.rgb_sha256,
                "objects_scanned": 1,
                "seconds": 0.25,
                "predictions": [{**p, "object_id": obj} for p in poses],
                "detections": len(poses),
                "invalid_poses": 0,
                "pnp_failures": 0,
            }
        )
    return records


def _source_assemble_rows(journals):
    """The source ``all_adapt_test_engine.assemble`` CSV rows of per-model journals."""
    totals = []
    for obj, records in journals.items():
        for row in records:
            for rank, p in enumerate(row["predictions"]):
                totals.append((row["scene_id"], row["im_id"], obj, rank, p))
    rows = [["scene_id", "im_id", "obj_id", "score", "R", "t", "time"]]
    for scene, im, obj, _, pred in sorted(totals, key=lambda x: x[:4]):
        rows.append(
            [
                str(scene),
                str(im),
                str(obj),
                format(pred["score"], ".10g"),
                " ".join(format(v, ".12g") for v in pred["R"]),
                " ".join(format(v, ".12g") for v in pred["t"]),
                "-1",
            ]
        )
    return rows


def test_bank_csv_bytes_match_source_assemble(tmp_path, monkeypatch):
    test_root = _test_split(tmp_path / "test")
    frames, population = export.load_test_frames(test_root, "pinhole")
    runs = tmp_path / "runs"
    bank = {obj: write_base(runs, obj) for obj in (1, 2)}
    cache = export.JournalCache(tmp_path / "cache")
    journals = {obj: _oracle_journal(obj, frames) for obj in bank}
    settings = export.cache_settings_sha256(
        "pinhole", export.SCORE_THRESHOLD, export.MAX_DETECTIONS
    )
    for obj, run in bank.items():
        key = export.JournalKey(
            sha256(run / layout.CHECKPOINT_PATH),
            sha256(run / layout.KEYPOINTS_PATH),
            population,
            settings,
        )

        def seed(directory, records=journals[obj]):
            directory.mkdir(parents=True, exist_ok=True)
            lines = "".join(json.dumps(raw_record(r)) + "\n" for r in records)
            (directory / export.JOURNAL_NAME).write_text(lines)

        cache.get_or_run(key, seed)

    def no_model(*args, **kwargs):
        raise AssertionError("a cached model must not be loaded")

    monkeypatch.setattr(export, "load_bank", no_model)
    out = tmp_path / "out" / "doperstaticfallback_handal-test.csv"
    population_spec = export.TestPopulation(len(frames), 1)
    summary = export.run_bank(
        bank, test_root, out, objects=[1, 2], expected_population=population_spec, cache=cache,
        device="cpu",
    )  # fmt: skip
    with out.open(newline="") as stream:
        assert list(csv.reader(stream)) == _source_assemble_rows(journals)
    assert summary.images == 3 and summary.pose_rows == sum(
        len(r["predictions"]) for records in journals.values() for r in records
    )
    with pytest.raises(ValueError, match="untimed"):
        export.run_bank(
            bank, test_root, out, objects=[1, 2], expected_population=population_spec,
            cache=cache, record_time=True, device="cpu",
        )  # fmt: skip


def test_cache_miss_scans_one_model_and_is_reused_by_another_bank(tmp_path, monkeypatch):
    test_root = _test_split(tmp_path / "test")
    frames, _ = export.load_test_frames(test_root, "pinhole")
    runs = tmp_path / "runs"
    base = write_base(runs, 1)
    student = write_student(runs, "static_pseudo", 1)
    loads = []

    def detect(image, camera_matrix):
        det = Detection(np.zeros(4), 0.8, np.zeros((4, 2)), np.ones(4))
        det.rvec, det.tvec = np.array([0.1, 0.2, 0.3]), np.array([1.0, 2.0, 400.0])
        return [det]

    def load_bank(bank, **kwargs):
        loads.append(dict(bank))
        return [
            export.BankModel(
                obj,
                detect,
                sha256(run / layout.CHECKPOINT_PATH),
                sha256(run / layout.KEYPOINTS_PATH),
            )
            for obj, run in bank.items()
        ]

    monkeypatch.setattr(export, "load_bank", load_bank)
    cache = export.JournalCache(tmp_path / "cache")
    spec = export.TestPopulation(len(frames), 1)
    first = tmp_path / "a" / "doperbase_handal-test.csv"
    second = tmp_path / "b" / "doperstaticfallback_handal-test.csv"
    options = {"objects": [1], "expected_population": spec, "cache": cache, "device": "cpu"}
    export.run_bank({1: base}, test_root, first, **options)
    export.run_bank({1: base}, test_root, second, **options)
    assert loads == [{1: base}] and first.read_bytes() == second.read_bytes()
    export.run_bank({1: student}, test_root, second, **options)
    assert loads == [{1: base}, {1: student}]


def test_bank_matrix_is_published_atomically(tmp_path, monkeypatch):
    written = []
    real = banks.publish_text
    monkeypatch.setattr(banks, "publish_text", lambda p, t: (written.append(p.name), real(p, t)))
    runs = tmp_path / "runs"
    write_base(runs, 1)
    policy = Policy("base", "base", "doperbase", "skipped", None)
    bank, fallbacks = banks.fixed_bank(policy, runs, [1])
    matrix, _ = banks.bank_manifest(tmp_path / "out", policy, bank, fallbacks, runs)
    assert written == [banks.MATRIX_NAME] and matrix.read_text().count("\n") == 2
