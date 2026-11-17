# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HANDAL follow-ups (Table 89): the obj8 self-training loop, the obj7 bootstrap and its review."""

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from bop_recipe_data import RECIPES, make_bop_download, write_recipe
from test_adaptation import (
    BAD,
    GOOD,
    SMOKE_BASE,
    SMOKE_LAUNCH,
    UNRELATED,
    Factory,
    FakeTrainer,
    ScriptedTeacher,
    make_base,
    prepare_object,
    sha256,
    write_view,
)

from doper import cli
from doper.bop import layout
from doper.recipes.schema import FollowupSpec, load_recipe, parse_recipe
from doper.training import adaptation, followups

LOOP, BOOTSTRAP = "obj8_loop", "obj7_bootstrap"
MIDDLE = 140  # accepted by a student teacher only (ChainFactory)


class ThresholdTeacher(ScriptedTeacher):
    """The scripted teacher with another gray-level acceptance threshold."""

    def __init__(self, threshold: int) -> None:
        self.threshold = threshold

    def __call__(self, image_bgr, camera_matrix, keypoints_mm, score_threshold):
        level = 255 if int(image_bgr[0, 0, 0]) > self.threshold else 0
        flat = np.full_like(image_bgr, level)
        return super().__call__(flat, camera_matrix, keypoints_mm, score_threshold)


class ChainFactory(Factory):
    """Base and student teachers that accept different frames."""

    def __init__(self, base: int = 150, student: int = 128) -> None:
        super().__init__()
        self.thresholds = {"base": base, "student": student}

    def __call__(self, checkpoint: Path):
        self.loaded.append(Path(checkpoint))
        kind = "base" if "base" in Path(checkpoint).parts else "student"
        return ThresholdTeacher(self.thresholds[kind])


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def handal_with(**changes: dict) -> object:
    """The packaged HANDAL recipe with replaced follow-up fields."""
    data = yaml.safe_load((RECIPES / "handal.yaml").read_text())
    for name, values in changes.items():
        data["followups"][name].update(values)
    rounds = {f"{n}/round{i:02d}" for n, f in data["followups"].items() for i in range(1, 100)}
    kept = {
        f"{n}/round{i:02d}" for n, f in data["followups"].items() for i in range(1, f["rounds"] + 1)
    }
    policies = data["test"]["policies"]
    for policy, values in list(policies.items()):
        if values["method"] in rounds - kept:  # rounds the shortened series no longer has
            del policies[policy]
    return parse_recipe(data)


def run_loop(recipe, name, root, runs, factory=None, **kwargs):
    return followups.run_self_training(
        recipe, name, root, runs, predictor_factory=factory or Factory(), **kwargs
    )


# --------------------------------------------------------------------------------------
# Stop policy and label churn (source handal_obj8_loop)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "labels,round_number,expected",
    [
        (7, 3, None),
        (19, 7, None),
        (20, 4, "target_training_pool_reached"),
        (20, 8, "target_training_pool_reached"),
        (7, 8, "round_cap_reached"),
        (0, 3, "no_positive_label_pool"),
        (0, 8, "no_positive_label_pool"),
    ],
)
def test_stop_reason_table(labels, round_number, expected):
    assert followups.stop_reason(labels, round_number, 20, 8) == expected


def test_label_churn_matches_source():
    def manifest(names):
        return {"records": [{"source_relative_path": n, "scene": n.split("/")[0]} for n in names]}

    churn = followups.label_churn(manifest(["down/1", "down/2"]), manifest(["down/1", "up/3"]))
    assert churn == {
        "retained": 1,
        "added": ["up/3"],
        "dropped": ["down/2"],
        "per_sequence": {"down": 1, "up": 1},
    }
    assert followups.label_churn({"records": []}, manifest(["a/1"]))["added"] == ["a/1"]


def test_post_training_mask_audit_does_not_change_labels(tmp_path):
    pool = tmp_path / "labels"
    train = pool / "train"
    train.mkdir(parents=True)
    manifest = {
        "records": [
            {
                "source_relative_path": "s/rgb/000001.jpg",
                "scene": "s",
                "frame": 1,
                "image": "1.jpg",
                "score": 0.7,
                "reproj_px": 2.0,
                "inliers": 60,
            }
        ]
    }
    (pool / "manifest.json").write_text(json.dumps(manifest))
    coco = {
        "images": [{"id": 1, "file_name": "1.jpg"}],
        "annotations": [{"image_id": 1, "mask_file": "mask.png"}],
    }
    (train / "annotations.json").write_text(json.dumps(coco))
    gt = np.ones((4, 4), np.uint8) * 255
    pred = gt.copy()
    pred[:2] = 0
    cv2.imwrite(str(train / "mask.png"), pred)
    frozen = {p: p.read_bytes() for p in pool.rglob("*") if p.is_file()}
    archive = tmp_path / "static.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("static/s/mask_visib/000001_000000.png", cv2.imencode(".png", gt)[1].tobytes())
    report = followups.mask_quality_audit(pool, archive)
    assert report["mean_mask_iou"] == 0.5 and report["mask_iou_ge_08"] == 0
    assert report["gt_masks"] == 1 and "diagnostic only" in report["protocol"]
    assert all(p.read_bytes() == data for p, data in frozen.items())


# --------------------------------------------------------------------------------------
# Recipe table
# --------------------------------------------------------------------------------------
def test_handal_followups_and_policies():
    recipe = load_recipe("handal")
    assert recipe.followups[LOOP] == FollowupSpec(
        obj_id=8,
        teacher="base",
        gate_overrides={},
        minimum_labels=1,
        rounds=8,
        target_labels=20,
        reuse_labels_from="static_pseudo",
        require_review=False,
    )
    assert recipe.followups[BOOTSTRAP] == FollowupSpec(
        obj_id=7,
        teacher="base",
        gate_overrides={"max_reprojection_px": 5.0},
        minimum_labels=1,
        rounds=8,
        target_labels=20,
        reuse_labels_from=None,
        require_review=True,
    )
    scoped = {
        p.name: (p.method, p.csv_method, p.objects)
        for p in recipe.test.policies.values()
        if p.fallback == "out_of_scope"
    }
    rounds = {
        f"obj8_static_round{i:02d}": (f"obj8_loop/round{i:02d}", f"doperobj8r{i:02d}fallback", (8,))
        for i in range(1, 9)
    }
    assert scoped == {
        "obj7_bootstrap": ("obj7_bootstrap/round01", "doperobj7bootstrapfallback", (7,)),
        **rounds,
        "obj8_combined": ("combined_pseudo", "doperobj8combinedfallback", (8,)),
    }
    # The source export order (handal_all_adapt_test.FOLLOWUPS).
    assert list(scoped) == ["obj7_bootstrap", *rounds, "obj8_combined"]
    assert recipe.test.aliases["static_iterative_round8"].run == "obj8_loop/round08"
    assert BOOTSTRAP not in {a.run.split("/")[0] for a in recipe.test.aliases.values()}


def test_followup_table_is_validated():
    data = yaml.safe_load((RECIPES / "handal.yaml").read_text())
    good = dict(data["followups"][BOOTSTRAP])
    parse_recipe(data)
    bad = [
        ({"obj_id": 99}, "recipe object"),
        ({"teacher": "nobody"}, "teacher"),
        ({"teacher": "dynamic_pseudo", "obj_id": 7}, "teacher"),
        ({"gate_overrides": {"max_reprojection": 5.0}}, "gate"),
        ({"minimum_labels": 0}, "at least 1"),
        ({"rounds": 0}, "rounds"),
        ({"rounds": 100}, "rounds"),
        ({"target_labels": 0}, "target"),
        ({"reuse_labels_from": "static_pseudo"}, "reuse"),
        ({"reuse_labels_from": "gt_box_mask", "gate_overrides": {}}, "reuse"),
        ({"surprise": 1}, "unknown"),
    ]
    for change, message in bad:
        broken = json.loads(json.dumps(data))
        broken["followups"][BOOTSTRAP] = {**good, **change}
        with pytest.raises(ValueError, match=message):
            parse_recipe(broken)
    for name in ("static_pseudo", "base", "gt_pools", "test", "Bad-Name"):
        broken = json.loads(json.dumps(data))
        broken["followups"][name] = good
        with pytest.raises(ValueError, match="followups"):
            parse_recipe(broken)
    policy = data["test"]["policies"]["obj8_static_round08"]
    for method, objects in (("obj8_loop/round09", [8]), ("obj8_loop/round08", [7])):
        broken = json.loads(json.dumps(data))
        broken["test"]["policies"]["obj8_static_round08"] = {
            **policy,
            "method": method,
            "objects": objects,
        }
        with pytest.raises(ValueError, match="follow-up"):
            parse_recipe(broken)


# --------------------------------------------------------------------------------------
# Self-training runs (stand-in trainer)
# --------------------------------------------------------------------------------------
@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root, runs = tmp_path / "root", tmp_path / "runs"
    monkeypatch.setattr(adaptation, "validation_exclusions", lambda recipe, root: UNRELATED)
    trainer = FakeTrainer()
    monkeypatch.setattr(adaptation, "launch_training", trainer)
    return root, runs, trainer


def test_obj8_loop_reuses_the_static_pool_and_runs_to_the_cap(workspace):
    root, runs, trainer = workspace
    prepare_object(root, runs, 8)
    write_view(root, "static", [GOOD] * 6 + [BAD], "obj_000008_up")
    recipe = load_recipe("handal")
    result = run_loop(recipe, LOOP, root, runs)
    assert result.stop_reason == "round_cap_reached" and len(result.rounds) == 8
    assert len(trainer.configs) == 8
    static = layout.object_run(runs, "static_pseudo", 8)
    assert read(static / "adaptation_skipped.json")["labels"] == 6
    first = followups.round_dir(runs, LOOP, 1, 8)
    assert first == runs / "obj8_loop" / "round01" / "obj_000008"
    copied = first / "labels" / "train" / "annotations.json"
    assert copied.read_bytes() == (static / "labels" / "train" / "annotations.json").read_bytes()
    identity = read(first / "identity.json")
    assert identity["teacher_kind"] == "base" and identity["minimum_labels"] == 1
    assert identity["reused_labels_from"] == "static_pseudo" and identity["round"] == 1
    previous = read(first / "completed.json")
    for number in range(2, 9):
        done = read(followups.round_dir(runs, LOOP, number, 8) / "completed.json")
        assert done["teacher_kind"] == "student"
        assert done["teacher_sha256"] == done["init_from_sha256"] == previous["student_sha256"]
        previous = done
    progress = read(runs / LOOP / "progress.json")
    assert progress["finished"] and progress["stop_reason"] == "round_cap_reached"
    assert progress["final_student"] == "obj8_loop/round08/obj_000008/work/last.pth"
    assert progress["final_student_sha256"] == previous["student_sha256"]
    assert progress["final_student_yield"] == 6 == result.final_student_yield
    rows = progress["rounds"]
    assert [r["round"] for r in rows] == list(range(1, 9))
    assert rows[0]["churn"]["retained"] == 0 and len(rows[0]["churn"]["added"]) == 6
    assert all(r["churn"]["retained"] == 6 and not r["churn"]["added"] for r in rows[1:])
    probe = runs / LOOP / "final_probe" / "obj_000008"
    assert read(probe / "labels_ready.json")["labels"] == 6
    assert read(probe / "identity.json")["teacher_sha256"] == previous["student_sha256"]
    assert not (probe / "work").exists()
    again = run_loop(recipe, LOOP, root, runs)
    assert len(trainer.configs) == 8 and again.final_student_sha256 == result.final_student_sha256


def test_loop_stops_on_target_and_only_probes_the_final_student(workspace):
    root, runs, trainer = workspace
    prepare_object(root, runs, 8)
    write_view(root, "static", [GOOD] * 2 + [MIDDLE] * 2 + [BAD], "obj_000008_up")
    recipe = handal_with(obj8_loop={"rounds": 3, "target_labels": 4})
    factory = ChainFactory()
    result = run_loop(recipe, LOOP, root, runs, factory)
    assert result.stop_reason == "target_training_pool_reached"
    assert [r["labels"] for r in result.rounds] == [2, 4] and len(trainer.configs) == 2
    assert not (runs / LOOP / "round03").exists()
    second = followups.round_dir(runs, LOOP, 2, 8)
    assert result.final_student == second / layout.CHECKPOINT_PATH
    assert result.final_student_yield == 4
    churn = result.rounds[1]["churn"]
    assert churn["retained"] == 2 and len(churn["added"]) == 2
    loaded = list(factory.loaded)
    assert run_loop(recipe, LOOP, root, runs, factory).stop_reason == result.stop_reason
    assert len(trainer.configs) == 2 and factory.loaded == loaded


def test_followup_identity_change_needs_force(workspace):
    root, runs, _ = workspace
    prepare_object(root, runs, 8)
    write_view(root, "static", [GOOD] * 2 + [MIDDLE] * 2, "obj_000008_up")
    run_loop(handal_with(obj8_loop={"rounds": 3, "target_labels": 4}), LOOP, root, runs)
    changed = handal_with(obj8_loop={"rounds": 3, "target_labels": 3})
    with pytest.raises(ValueError, match="another identity"):
        run_loop(changed, LOOP, root, runs)
    assert run_loop(changed, LOOP, root, runs, force=True).stop_reason


def bootstrap_workspace(workspace) -> tuple[Path, Path, FakeTrainer, ChainFactory, object]:
    root, runs, trainer = workspace
    prepare_object(root, runs, 7)
    write_view(root, "static", [GOOD] * 3 + [BAD], "obj_000007_up")
    # The bootstrap student accepts nothing at the strict gate (the source outcome).
    return root, runs, trainer, ChainFactory(base=128, student=255), load_recipe("handal")


def test_review_required_before_bootstrap_train(workspace, capsys):
    root, runs, trainer, factory, recipe = bootstrap_workspace(workspace)
    with pytest.raises(ValueError, match="review"):
        run_loop(recipe, BOOTSTRAP, root, runs, factory)
    first = followups.round_dir(runs, BOOTSTRAP, 1, 7)
    assert read(first / "labels_ready.json")["labels"] == 3 and not trainer.configs
    assert read(first / "identity.json")["gate"]["reprojection_lt_px"] == 5.0
    overlays = sorted((first / "review").glob("*.jpg"))
    assert len(overlays) == 3
    review = ["review", "handal", "--root", str(root), "--runs", str(runs), "--obj-id", "7"]
    assert cli.main([*review, "--reviewer", "A. Reviewer"]) == 0
    assert read(first / "REVIEWED.json")["reviewer"] == "A. Reviewer"
    result = run_loop(recipe, BOOTSTRAP, root, runs, factory)
    assert len(trainer.configs) == 1
    done = read(first / "completed.json")
    assert done["review"]["status"] == "reviewed"
    assert done["review"]["reviewer"] == "A. Reviewer"
    second = followups.round_dir(runs, BOOTSTRAP, 2, 7)
    identity = read(second / "identity.json")
    assert identity["gate"]["reprojection_lt_px"] == 3.0 and identity["teacher_kind"] == "student"
    assert identity["teacher_sha256"] == done["student_sha256"]
    assert read(second / "adaptation_skipped.json")["labels"] == 0
    assert result.stop_reason == "no_positive_label_pool"
    assert result.final_student == first / layout.CHECKPOINT_PATH
    assert result.final_student_yield == 0
    assert not (runs / BOOTSTRAP / "final_probe").exists()
    progress = read(runs / BOOTSTRAP / "progress.json")
    assert progress["rounds"][0]["review"]["status"] == "reviewed"


def test_bootstrap_labels_do_not_count_toward_the_target(workspace):
    root, runs, trainer, factory, _ = bootstrap_workspace(workspace)
    recipe = handal_with(obj7_bootstrap={"target_labels": 2, "rounds": 2})
    result = run_loop(recipe, BOOTSTRAP, root, runs, factory, accept_unreviewed=True)
    # Three relaxed-gate labels exceed the target, yet the loop continues at the strict gate.
    assert [r["labels"] for r in result.rounds] == [3, 0]
    assert result.stop_reason == "no_positive_label_pool"


def test_accept_unreviewed_is_recorded(workspace):
    root, runs, trainer, factory, recipe = bootstrap_workspace(workspace)
    run_loop(recipe, BOOTSTRAP, root, runs, factory, accept_unreviewed=True)
    first = followups.round_dir(runs, BOOTSTRAP, 1, 7)
    done = read(first / "completed.json")
    assert done["review"]["status"] == "accepted_unreviewed"
    assert "no human review" in done["review"]["reason"]
    progress = read(runs / BOOTSTRAP / "progress.json")
    assert progress["rounds"][0]["review"]["status"] == "accepted_unreviewed"


def test_stale_review_rejected(workspace):
    root, runs, trainer, factory, recipe = bootstrap_workspace(workspace)
    with pytest.raises(ValueError, match="review"):
        run_loop(recipe, BOOTSTRAP, root, runs, factory)
    first = followups.round_dir(runs, BOOTSTRAP, 1, 7)
    review = ["review", "handal", "--root", str(root), "--runs", str(runs), "--obj-id", "7"]
    assert cli.main([*review, "--reviewer", "reviewer"]) == 0
    ready = first / "labels_ready.json"
    ready.write_text(json.dumps(read(ready), indent=2))  # same pool, another READY SHA
    with pytest.raises(ValueError, match="stale"):
        run_loop(recipe, BOOTSTRAP, root, runs, factory)
    assert not trainer.configs


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def test_loop_dry_run_lists_rounds(tmp_path, capsys):
    root, runs = tmp_path / "root", tmp_path / "runs"
    common = ["--root", str(root), "--runs", str(runs), "--dry-run"]
    argv = ["train", "handal", "--stage", "followup", "--name", LOOP, "--obj-id", "8", *common]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    for number in range(1, 9):
        assert f"obj8_loop/round{number:02d}/obj_000008" in out
    assert "obj8_loop/final_probe/obj_000008" in out and "progress.json" in out
    assert "static_pseudo" in out and "nothing was read or written" in out
    argv = ["train", "handal", "--stage", "followup", "--name", BOOTSTRAP, "--obj-id", "7"]
    assert cli.main([*argv, *common]) == 0
    out = capsys.readouterr().out
    assert "obj7_bootstrap/round01/obj_000007" in out and "REVIEWED.json" in out
    assert "5.0" in out
    review = ["review", "handal", "--obj-id", "7", "--reviewer", "r", *common]
    assert cli.main(review) == 0
    assert "obj7_bootstrap/round01/obj_000007" in capsys.readouterr().out
    assert not root.exists() and not runs.exists()
    refused = [
        ["train", "handal", "--stage", "followup", "--name", LOOP, "--obj-id", "7", *common],
        ["train", "handal", "--stage", "followup", "--name", "nope", "--obj-id", "8", *common],
        ["train", "handal", "--stage", "followup", "--obj-id", "8", *common],
        ["train", "handal", "--stage", "adapt", "--name", LOOP, "--obj-id", "8", *common],
        [
            "train",
            "handal",
            "--stage",
            "adapt",
            "--method",
            "static_pseudo",
            "--obj-id",
            "8",
            "--accept-unreviewed",
            *common,
        ],
        ["review", "handal", "--obj-id", "8", "--reviewer", "r", *common],
    ]
    for argv in refused:
        assert cli.main(argv) == 1, argv


def test_review_cli_needs_rendered_overlays(workspace):
    root, runs, _ = workspace
    review = ["review", "handal", "--root", str(root), "--runs", str(runs), "--obj-id", "7"]
    assert cli.main([*review, "--reviewer", "r"]) == 1
    make_base(runs, 7)
    assert cli.main([*review, "--reviewer", "r"]) == 1


# --------------------------------------------------------------------------------------
# CPU smoke run
# --------------------------------------------------------------------------------------
@pytest.mark.slow
def test_two_round_cpu_smoke_stops_on_target(tmp_path):
    root, runs = tmp_path / "root", tmp_path / "runs"
    profile = make_bop_download(root, tmp_path / "profile")
    recipe_path = write_recipe(
        tmp_path / "fixture.yaml",
        "hopev2",
        profile,
        convert__val_scenes=1,
        base__launch=SMOKE_LAUNCH,
        base__config=SMOKE_BASE,
    )
    data = yaml.safe_load(recipe_path.read_text())
    data["adapt"]["config"].update(batch_size=6, num_workers=0)
    data["followups"] = {
        "loop": {
            "obj_id": 1,
            "teacher": "base",
            "reuse_labels_from": "static_pseudo",
            "rounds": 3,
            "target_labels": 4,
        }
    }
    recipe_path.write_text(yaml.safe_dump(data, sort_keys=False))
    assert cli.main(["prepare", str(recipe_path), "--root", str(root), "--obj-id", "1"]) == 0
    train = ["train", str(recipe_path), "--root", str(root), "--runs", str(runs), "--obj-id", "1"]
    assert cli.main([*train, "--stage", "base"]) == 0
    base = layout.object_run(runs, "base", 1)
    completed = read(base / "completed.json")
    completed["epochs"] = 25  # stand in for the epoch-25 base the engine requires
    (base / "completed.json").write_text(json.dumps(completed))
    write_view(root, "static", [GOOD] * 2 + [MIDDLE] * 2 + [BAD], "obj_000001")
    recipe = load_recipe(recipe_path)
    result = followups.run_self_training(
        recipe,
        "loop",
        root,
        runs,
        predictor_factory=ChainFactory(),
        overrides={"device": "cpu"},
    )
    assert result.stop_reason == "target_training_pool_reached"
    assert [r["labels"] for r in result.rounds] == [2, 4]
    first, second = (followups.round_dir(runs, "loop", n, 1) for n in (1, 2))
    one, two = read(first / "completed.json"), read(second / "completed.json")
    assert one["teacher_sha256"] == sha256(base / layout.CHECKPOINT_PATH)
    assert two["teacher_sha256"] == two["init_from_sha256"] == one["student_sha256"]
    assert one["global_step"] == two["global_step"] == 3
    assert result.final_student_sha256 == two["student_sha256"]
    assert result.final_student_yield == 4 and not (runs / "loop" / "round03").exists()
