# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation-selected model banks: ranking, base fallback and eligibility checks."""

import dataclasses
import hashlib
import json
import subprocess
import sys

import pytest

from doper.bop import selection
from doper.bop.ar import PROTOCOL_HOT3D_NATIVE, PROTOCOL_TOP1, ArmResult, ArResult
from doper.bop.selection import (
    FALLBACK_REASON,
    SELECTED_REASON,
    Candidate,
    build_bank,
    choose,
    require_shared_population,
)

POPULATION = "a" * 64
GEOMETRY = "g" * 64


def _result(obj, ar_mean, *, frames=10, protocol=PROTOCOL_TOP1, population=POPULATION, **lineage):
    arm = ArmResult(frames, frames, ar_mean, ar_mean, ar_mean, (), ())
    return ArResult(
        obj, protocol, "pinhole", "scored", population, frames, frames, {"single": arm}, **lineage
    )


def _candidate(obj, arm, ar_mean=None, **changes):
    """Candidate whose result (unless given) is scored on the candidate's own fields."""
    candidate = Candidate(
        obj_id=obj,
        arm=arm,
        endpoint_complete=True,
        checkpoint_sha256=f"{obj}-{arm}".ljust(64, "0"),
        geometry_sha256=GEOMETRY,
        validation_population_sha256=POPULATION,
        protocol=PROTOCOL_TOP1,
    )
    candidate = dataclasses.replace(candidate, **changes)
    result = candidate.result
    if result is None and ar_mean is not None:
        result = _result(
            obj, ar_mean, protocol=candidate.protocol,
            population=candidate.validation_population_sha256,
        )  # fmt: skip
    if result is not None and result.checkpoint_sha256 is None:
        result = dataclasses.replace(
            result,
            checkpoint_sha256=candidate.checkpoint_sha256,
            geometry_sha256=result.geometry_sha256 or candidate.geometry_sha256,
        )
    return dataclasses.replace(candidate, result=result)


def test_selection_falls_back_to_base_and_reports_every_object():
    candidates = [
        _candidate(1, "base"),
        _candidate(1, "static_pseudo"),
        _candidate(2, "base", 50.0),
        _candidate(2, "static_pseudo", 60.0),
        _candidate(2, "test_tta", 55.0),
        _candidate(3, "base", 60.0),
        _candidate(3, "static_pseudo", 60.0),
        _candidate(5, "base", 40.0),
        _candidate(5, "gt_box_mask", 70.0),
        _candidate(5, "dynamic_pseudo", 70.0),
    ]
    selection = choose(candidates, [1, 2, 3, 4, 5])
    assert sorted(selection) == [1, 2, 3, 4, 5]
    assert selection[1].status == "base_fallback" and selection[1].reason == FALLBACK_REASON
    assert selection[1].chosen.arm == "base"
    assert selection[2].status == "selected" and selection[2].reason == SELECTED_REASON
    assert selection[2].chosen.arm == "static_pseudo"
    assert selection[3].chosen.arm == "base"  # exact tie keeps the base
    assert selection[5].chosen.arm == "dynamic_pseudo"  # then method-name order
    assert selection[4].status == "error" and selection[4].chosen is None
    assert "missing" in selection[4].reason
    assert selection[2].base.arm == "base" and len(selection[2].candidates) == 3


def test_selection_ignores_unscored_candidates_but_needs_scored_base_fields():
    unscored = _result(1, 0.0)
    not_evaluable = dataclasses.replace(unscored, status="not_evaluable", arms={})
    candidates = [
        _candidate(1, "base", 30.0),
        _candidate(1, "bootstrap", result=not_evaluable),
    ]
    assert choose(candidates, [1])[1].chosen.arm == "base"
    candidates = [_candidate(1, "base"), _candidate(1, "static_pseudo", 70.0)]
    selection = choose(candidates, [1])[1]
    assert selection.status == "error" and "differs" in selection.reason


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"geometry_sha256": "h" * 64}, "differs"),
        ({"validation_population_sha256": "b" * 64}, "differs"),
        ({"protocol": "HOPE-multi-instance-top5-score-greedy-v1"}, "differs"),
        ({"result": _result(1, 70.0, frames=9)}, "differs"),
        ({"result": _result(1, 70.0, population="b" * 64)}, "lineage"),
        ({"result": _result(2, 70.0)}, "lineage"),
        ({"result": _result(1, 101.0)}, "invalid validation score"),
        ({"result": _result(1, float("nan"))}, "invalid validation score"),
        ({"endpoint_complete": False}, "unfinished"),
        ({"result": _result(1, 70.0, checkpoint_sha256="z" * 64)}, "lineage"),
        ({"result": _result(1, 70.0, geometry_sha256="h" * 64)}, "lineage"),
    ],
)
def test_selection_rejects_mismatched_hashes_and_unfinished_endpoints(change, message, tmp_path):
    candidates = [_candidate(1, "base", 50.0), _candidate(1, "static_pseudo", 70.0, **change)]
    selection = choose(candidates, [1])
    assert selection[1].status == "error"
    assert message in selection[1].reason
    with pytest.raises(ValueError, match="object 1"):
        build_bank(selection, tmp_path)


def test_selection_requires_exactly_one_base():
    assert "exactly one" in choose([_candidate(1, "static_pseudo", 50.0)], [1])[1].reason
    twice = [_candidate(1, "base", 50.0), _candidate(1, "base", 60.0)]
    assert choose(twice, [1])[1].status == "error"


def test_selection_does_not_import_the_inference_stack():
    code = "import sys, doper.bop.selection; assert not {'cv2', 'torch'} & set(sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_selection_rejects_foreign_objects():
    with pytest.raises(ValueError, match="not in"):
        choose([_candidate(9, "base", 1.0)], [1])


def _write_run(root, arm, obj, checkpoint=b"weights", geometry=b"{}"):
    run = root / arm / f"obj_{obj:06d}"
    (run / "work").mkdir(parents=True)
    (run / "geometry").mkdir()
    (run / "work" / "last.pth").write_bytes(checkpoint)
    (run / "geometry" / "keypoints_3d.json").write_bytes(geometry)
    return run


def test_build_bank_resolves_runs_and_verifies_hashes(tmp_path):
    run = _write_run(tmp_path, "static_pseudo", 2)
    digest = hashlib.sha256(b"weights").hexdigest()
    geometry = hashlib.sha256(b"{}").hexdigest()
    candidates = [
        _candidate(2, "base", 50.0, geometry_sha256=geometry),
        _candidate(2, "static_pseudo", 60.0, checkpoint_sha256=digest, geometry_sha256=geometry),
    ]
    assert build_bank(choose(candidates, [2]), tmp_path) == {2: run}
    (run / "work" / "last.pth").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint"):
        build_bank(choose(candidates, [2]), tmp_path)


def test_shared_population_across_objects():
    selection = choose([_candidate(1, "base", 1.0), _candidate(2, "base", 2.0)], [1, 2])
    require_shared_population(selection)
    other = _candidate(3, "base", validation_population_sha256="c" * 64)
    with pytest.raises(ValueError, match="share"):
        require_shared_population({**selection, 3: choose([other], [3])[3]})
    with pytest.raises(ValueError, match="needs a base"):
        require_shared_population(choose([], [4]))


def _native(obj, arm, ar_mean, *, status="scored", camera_code="c" * 64, **changes):
    result = dataclasses.replace(
        _result(obj, ar_mean, protocol=PROTOCOL_HOT3D_NATIVE),
        camera="native", status=status, camera_code_sha256=camera_code,
    )  # fmt: skip
    return _candidate(obj, arm, protocol=PROTOCOL_HOT3D_NATIVE, result=result, **changes)


HOT3D_RULES = {"protocol": PROTOCOL_HOT3D_NATIVE, "require_scored": True}


def test_hot3d_rules_require_scored_candidates_protocol_and_camera_code():
    good = [_native(1, "base", 30.0), _native(1, "static_pseudo", 40.0)]
    assert choose(good, [1], **HOT3D_RULES)[1].chosen.arm == "static_pseudo"
    unscored = [_native(1, "base", 30.0), _native(1, "static_pseudo", 0.0, status="not_evaluable")]
    assert choose(unscored, [1])[1].chosen.arm == "base"  # BOP rule: simply not ranked
    rejected = choose(unscored, [1], **HOT3D_RULES)[1]
    assert rejected.status == "error" and "unrankable" in rejected.reason
    other_code = [_native(1, "base", 30.0), _native(1, "static_pseudo", 40.0, camera_code="d" * 64)]
    assert "camera_code" in choose(other_code, [1], **HOT3D_RULES)[1].reason
    no_code = [_native(1, "base", 30.0, camera_code=None)]
    assert "camera code" in choose(no_code, [1], **HOT3D_RULES)[1].reason
    pinhole = [_candidate(1, "base", 30.0)]
    assert "not scored under" in choose(pinhole, [1], **HOT3D_RULES)[1].reason


def _score(run, obj, ar_mean):
    """Write ``validation/ar.json`` scoring the run's own checkpoint and geometry."""
    from bank_runs import sha256

    from doper.bop import layout

    result = _result(
        obj, ar_mean,
        checkpoint_sha256=sha256(run / layout.CHECKPOINT_PATH),
        geometry_sha256=sha256(run / layout.KEYPOINTS_PATH),
    )  # fmt: skip
    path = run / layout.VALIDATION_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict()))


def test_selection_omits_skipped_and_breaks_ties_by_source_name(tmp_path, monkeypatch):
    from bank_runs import write_base, write_skip, write_student

    from doper.bop import layout
    from doper.cli import evaluate
    from doper.recipes.schema import load_recipe

    monkeypatch.setattr(
        evaluate, "expected_validation", lambda *a, **k: (PROTOCOL_TOP1, POPULATION)
    )
    runs = tmp_path / "runs"
    handal = load_recipe("handal")
    # HANDAL: base, static, both GT arms and test TTA; object 8 adds its two follow-up aliases
    # (the combined student is selected under its source name); object 7 is unscored.
    scores = {
        1: {"base": 50, "static_pseudo": 60, "gt_box_mask": 70, "gt_box_mask_keypoints": 70,
            "test_tta": 65},
        8: {"base": 40, "gt_box_mask": 41, "gt_box_mask_keypoints": 42, "test_tta": 43,
            "combined_pseudo": 80, "obj8_loop/round08": 80},
    }  # fmt: skip
    for obj in (1, 7, 8):
        write_base(runs, obj)
        for method in ("static_pseudo", "gt_box_mask", "gt_box_mask_keypoints", "test_tta"):
            write_student(runs, method, obj)
    write_skip(runs, "static_pseudo", 7, labels=3)
    write_skip(runs, "static_pseudo", 8, labels=6)
    write_skip(runs, "dynamic_pseudo", 8, labels=0)
    write_student(runs, "combined_pseudo", 8, labels=6)
    write_student(runs, "obj8_loop/round08", 8, labels=12)
    write_student(runs, "obj7_bootstrap/round01", 7, labels=3)  # never a selection candidate
    for obj, table in scores.items():
        for arm, value in table.items():
            _score(layout.object_run(runs, arm, obj), obj, float(value))
    recipe = dataclasses.replace(handal, objects=(1, 7, 8), generated_objects=())
    bank, choices = evaluate.select_bank(recipe, tmp_path, runs, "selected", False)
    names = {obj: sorted(c.arm for c in s.candidates) for obj, s in choices.items()}
    assert names[7] == ["base", "gt_box_mask", "gt_box_mask_keypoints", "test_tta"]
    assert names[8] == [
        "base", "gt_box_mask", "gt_box_mask_keypoints", "static_dynamic_low_label",
        "static_iterative_round8", "test_tta",
    ]  # fmt: skip
    assert choices[1].chosen.arm == "gt_box_mask"
    assert choices[7].status == "base_fallback" and choices[7].reason == FALLBACK_REASON
    assert choices[8].chosen.arm == "static_dynamic_low_label"
    assert bank[8] == layout.object_run(runs, "combined_pseudo", 8)
    # HOPEv2: base and five methods; a tie between two methods goes to the first name.
    runs = tmp_path / "hope"
    write_base(runs, 3)
    hope = {"base": 10, "static_pseudo": 20, "dynamic_pseudo": 30, "combined_pseudo": 30,
            "gt_box_mask": 25, "gt_box_mask_keypoints": 5}  # fmt: skip
    for arm, value in hope.items():
        run = layout.object_run(runs, arm, 3) if arm == "base" else write_student(runs, arm, 3)
        _score(run, 3, float(value))
    recipe = dataclasses.replace(load_recipe("hopev2"), objects=(3,))
    _, choices = evaluate.select_bank(recipe, tmp_path, runs, "selected", False)
    assert len(choices[3].candidates) == 6 and choices[3].chosen.arm == "combined_pseudo"
    # HOT3D: base and five methods as candidates (skipped pseudo methods omitted).
    runs = tmp_path / "hot3d"
    write_base(runs, 2)
    for method in ("static_pseudo", "dynamic_pseudo", "combined_pseudo", "gt_box_mask",
                   "gt_box_mask_keypoints"):  # fmt: skip
        write_student(runs, method, 2)
    endpoints = selection.recipe_endpoints(load_recipe("hot3d"), runs, [2])
    assert [e.name for e in endpoints] == [
        "base", "static_pseudo", "dynamic_pseudo", "combined_pseudo", "gt_box_mask",
        "gt_box_mask_keypoints",
    ]  # fmt: skip
    write_skip(runs, "dynamic_pseudo", 2, labels=4)
    assert "dynamic_pseudo" not in [e.name for e in selection.recipe_endpoints(
        load_recipe("hot3d"), runs, [2])]  # fmt: skip


def test_handal_object40_skips_match_the_source(tmp_path, monkeypatch):
    """Object 40 runs every source arm (``handal40_learning.ARMS``). The source skipped static
    (17 strict labels < 20), dynamic (0) and combined (the union, 17 < 20: object 8's minimum
    of 1 is not object 40's), so its candidates are the base, both GT arms and test TTA."""
    from bank_runs import write_base, write_skip, write_student

    from doper.bop import layout
    from doper.cli import evaluate
    from doper.recipes.schema import load_recipe

    handal = load_recipe("handal")
    assert handal.adapt is not None
    methods = handal.adapt.methods
    assert [m for m in methods if methods[m].applies_to(40)] == [
        "static_pseudo", "dynamic_pseudo", "combined_pseudo", "gt_box_mask",
        "gt_box_mask_keypoints", "test_tta",
    ]  # fmt: skip
    assert methods["combined_pseudo"].minimum_for(40) == 20
    assert methods["combined_pseudo"].minimum_for(8) == 1
    monkeypatch.setattr(
        evaluate, "expected_validation", lambda *a, **k: (PROTOCOL_TOP1, POPULATION)
    )
    runs = tmp_path / "runs"
    write_base(runs, 40)
    write_skip(runs, "static_pseudo", 40, labels=17)
    write_skip(runs, "dynamic_pseudo", 40, labels=0)
    write_skip(runs, "combined_pseudo", 40, labels=17)
    for method in ("gt_box_mask", "gt_box_mask_keypoints", "test_tta"):
        write_student(runs, method, 40)
    endpoints = selection.recipe_endpoints(handal, runs, [40])
    assert [e.name for e in endpoints] == [
        "base", "gt_box_mask", "gt_box_mask_keypoints", "test_tta",
    ]  # fmt: skip
    for name, value in (("base", 80.0), ("gt_box_mask", 83.2), ("test_tta", 82.7)):
        _score(layout.object_run(runs, name, 40), 40, value)
    recipe = dataclasses.replace(handal, objects=(40,), generated_objects=())
    bank, choices = evaluate.select_bank(recipe, tmp_path, runs, "selected", False)
    assert choices[40].chosen.arm == "gt_box_mask"
    assert bank[40] == layout.object_run(runs, "gt_box_mask", 40)
