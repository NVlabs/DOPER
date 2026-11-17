# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``doper`` command: dry runs, refusals, CAD keypoint parity and CPU smoke runs."""

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml
from bank_runs import sha256
from bop_recipe_data import (
    make_bop_download,
    make_hot3d_download,
    sphere,
    write_models,
    write_ply,
    write_recipe,
    write_scene,
)
from generated_render import render_config, write_render_root

import doper.keypoint_selection as keypoint_selection
from doper import cli
from doper.bop import layout
from doper.bop.ar import ArResult
from doper.cli.common import object_dir, prepared_receipt
from doper.cli.prepare import select_cad_keypoints
from doper.recipes.schema import load_recipe, training_namespace

GOLDEN_KEYPOINTS = Path(__file__).parent / "golden" / "source_cad_keypoints.json"
# The smoke backbone has no pretrained weights; the fixture recipe drops that launch check,
# since ``--override`` refuses launch keys for the base stage.
SMOKE_LAUNCH = {
    "init_from": None,
    "resume": None,
    "require_no_init_from": True,
    "require_pretrained_backbone": False,
}
SMOKE_OVERRIDES = (
    "backbone=resnet18",
    "allow_nonstandard_backbone=true",
    "no_pretrained=true",
    "det_size=128",
    "crop_size=128",
    "epochs=1",
    "batch_size=2",
    "max_iters_per_epoch=2",
    "gradient_accumulation_steps=1",
    "warmup_iters=1",
    "num_workers=0",
    "device=cpu",
    "no_amp=true",
    "save_interval=1000",
)


@pytest.fixture
def bop_fixture_root(tmp_path):
    """An empty dataset root: dry runs must plan without reading or writing it."""
    root = tmp_path / "root"
    root.mkdir()
    return root


def _tree(path: Path) -> list[str]:
    return sorted(str(p.relative_to(path)) for p in path.rglob("*"))


@pytest.mark.parametrize("name", ["handal", "hopev2", "hot3d", "tless"])
def test_recipe_dry_runs_on_fixture(name, bop_fixture_root, tmp_path, capsys):
    runs = tmp_path / "runs"
    root = str(bop_fixture_root)
    commands = [
        ["prepare", name, "--root", root],
        ["prepare", name, "--root", root, "--parts", "test", "--obj-id", "2"],
        ["train", name, "--stage", "base", "--root", root, "--runs", str(runs), "--obj-id", "2"],
        ["evaluate", name, "--split", "val", "--root", root, "--runs", str(runs)],
        ["evaluate", name, "--split", "test", "--root", root, "--runs", str(runs)],
        ["report", name, "--runs", str(runs)],
    ]
    for command in commands:
        assert cli.main([*command, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "nothing was read or written" in out
    assert "keypoints obj 2" in out and "obj_000002/work/last.pth" in out
    assert _tree(bop_fixture_root) == [] and not runs.exists()
    if name in ("handal", "hot3d"):
        assert "without official PBR" in out


def test_import_cli_is_fast(tmp_path):
    code = (
        "import sys\n"
        "from doper.cli import main\n"
        "try:\n"
        "    main(['--help'])\n"
        "except SystemExit:\n"
        "    pass\n"
        f"assert main(['prepare', 'handal', '--root', {str(tmp_path)!r}, '--dry-run']) == 0\n"
        "heavy = {'torch', 'huggingface_hub', 'trimesh'} & set(sys.modules)\n"
        "assert not heavy, heavy\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "prepare" in result.stdout and "evaluate" in result.stdout


@pytest.mark.source_golden
def test_cad_to_ordered_keypoints_matches_source(tmp_path, monkeypatch):
    golden = json.loads(GOLDEN_KEYPOINTS.read_text())
    monkeypatch.setattr(keypoint_selection, "load_numba_kernel", lambda: None)
    mesh = write_ply(tmp_path / "obj_000001.ply", *sphere())
    assert sha256(mesh) == golden["ply_sha256"]
    out = select_cad_keypoints(mesh, tmp_path / "geometry", 1, 64, "mm")
    stl = tmp_path / "geometry" / "object_mm.stl"
    assert sha256(stl) == golden["stl_sha256"]
    payload = json.loads(out.read_text())
    assert payload["indices"] == golden["indices"]
    assert payload["keypoints_3d"] == golden["keypoints_3d"]
    assert payload["keypoint_names"] == golden["keypoint_names"]
    assert payload["source_units"] == "mm" and payload["object"] == "obj_000001"


def test_train_refuses_existing_run_without_force(tmp_path):
    run = layout.object_run(tmp_path / "runs", "base", 2)
    (run / "work").mkdir(parents=True)
    (run / "work" / "last.pth").write_bytes(b"finished")
    argv = ["train", "hopev2", "--stage", "base", "--root", str(tmp_path / "root")]
    argv += ["--runs", str(tmp_path / "runs"), "--obj-id", "2"]
    assert cli.main(argv) == 1  # not prepared
    root = tmp_path / "root" / "objects" / "obj_000002"
    root.mkdir(parents=True)
    (root / "prepared.json").write_text(json.dumps({"hashes": {}}))
    with pytest.raises(SystemExit, match="--force"):
        cli.main(argv)
    assert (run / "work" / "last.pth").read_bytes() == b"finished"


def _smoke_train(recipe, root, runs, obj):
    argv = ["train", str(recipe), "--stage", "base", "--root", str(root), "--runs", str(runs)]
    argv += ["--obj-id", str(obj)]
    for item in SMOKE_OVERRIDES:
        argv += ["--override", item]
    assert cli.main(argv) == 0


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A prepared fixture dataset with a two-iteration CPU base model of object 1."""
    base = tmp_path_factory.mktemp("smoke")
    root, runs = base / "root", base / "runs"
    profile = make_bop_download(root, base / "profile")
    recipe = write_recipe(
        base / "fixture.yaml", "hopev2", profile, convert__val_scenes=1, base__launch=SMOKE_LAUNCH
    )
    assert cli.main(["prepare", str(recipe), "--root", str(root)]) == 0
    _smoke_train(recipe, root, runs, 1)
    return recipe, root, runs


def test_cpu_smoke_train_and_evaluate(trained, capsys):
    recipe, root, runs = trained
    paths = ["--root", str(root), "--runs", str(runs), "--device", "cpu"]
    assert cli.main(["prepare", str(recipe), "--root", str(root)]) == 0
    assert "already prepared and verified" in capsys.readouterr().out
    run = layout.object_run(runs, "base", 1)
    config = (run / "recipe.yaml").read_text()
    assert "backbone: resnet18" in config and "data_root:" in config
    completed = json.loads((run / "completed.json").read_text())
    assert completed["obj_id"] == 1 and completed["epochs"] == 1
    assert cli.main(["evaluate", str(recipe), "--split", "val", *paths]) == 0
    result = json.loads((run / layout.VALIDATION_PATH).read_text())
    assert result["status"] == "scored" and set(result["arms"]) == {"single", "jitter", "gtbox"}
    assert result["protocol"] == "HOPE-multi-instance-top5-score-greedy-v1"
    assert json.loads(json.dumps(ArResult.from_dict(result).to_dict())) == result
    capsys.readouterr()
    assert cli.main(["evaluate", str(recipe), "--split", "val", *paths, "--obj-id", "1"]) == 0
    assert "already scored" in capsys.readouterr().out
    test = ["evaluate", str(recipe), "--split", "test", *paths]
    assert cli.main(test) == 1
    assert "allow-partial-bank" in capsys.readouterr().out
    assert cli.main(["prepare", str(recipe), "--root", str(root), "--parts", "test"]) == 1
    argv = ["prepare", str(recipe), "--root", str(root), "--parts", "test", "--allow-test-inputs"]
    assert cli.main(argv) == 0
    assert cli.main([*test, "--allow-partial-bank"]) == 0
    out = runs / "test" / "selected"
    selection = json.loads((out / "selection.json").read_text())
    assert selection["1"]["arm"] == "base" and selection["2"]["status"] == "error"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["missing_objects"] == [2] and summary["images"] == 3
    with (out / "doperselected_hopev2-test.csv").open() as stream:
        assert next(csv.reader(stream)) == [
            "scene_id",
            "im_id",
            "obj_id",
            "score",
            "R",
            "t",
            "time",
        ]
    assert cli.main(["report", str(recipe), "--runs", str(runs)]) == 0
    report = (runs / "report_fixture_hopev2.md").read_text()
    assert "| base | 1 | 3 |" in report and "doperselected_hopev2-test.csv" in report


def test_hot3d_prepare_and_evaluate_on_fixture(trained, tmp_path):
    _, _, smoke_runs = trained
    root, runs = tmp_path / "root", tmp_path / "runs"
    profile = make_hot3d_download(root, tmp_path / "profile")
    recipe = write_recipe(tmp_path / "hot3d.yaml", "hot3d", profile, generated_objects=[2])
    argv = ["prepare", str(recipe), "--root", str(root), "--parts", "val", "test"]
    assert cli.main([*argv, "--allow-test-inputs"]) == 0
    manifest = json.loads((root / "validation_manifest.json").read_text())
    assert manifest["images"] == 3 and manifest["object_frame_counts"] == {"1": 3, "2": 3}
    assert {f["device"] for f in manifest["frames"]} == {"Aria", "Quest3"}
    test_manifest = json.loads((root / "hot3d_test" / "manifest.json").read_text())
    assert test_manifest["RGB_count"] == 5 and not test_manifest["test_GT_read"]
    support = json.loads((root / "hot3d_test" / "support_manifest.json").read_text())
    assert len(support["groups"]) == 5 and all(g["supports"] for g in support["groups"])
    write_models(root / "data" / "models_eval")
    shutil.copytree(
        layout.object_run(smoke_runs, "base", 1),
        layout.object_run(runs, "base", 1),
        ignore=shutil.ignore_patterns("validation"),
    )
    paths = ["--root", str(root), "--runs", str(runs), "--device", "cpu"]
    assert cli.main(["evaluate", str(recipe), "--split", "val", *paths]) == 0
    result = json.loads((layout.object_run(runs, "base", 1) / layout.VALIDATION_PATH).read_text())
    assert result["camera"] == "native" and result["camera_code_sha256"]
    assert result["frames"] == 3 and set(result["by_device"]) <= {"Aria", "Quest3"}
    test = ["evaluate", str(recipe), "--split", "test", *paths]
    assert cli.main(test) == 1
    assert cli.main([*test, "--allow-partial-bank", "--record-time"]) == 0
    summary = json.loads((runs / "test" / "selected" / "summary.json").read_text())
    assert summary["images"] == 5 and summary["missing_objects"] == [2]


def _train_argv(recipe, root, runs, *overrides):
    argv = ["train", str(recipe), "--stage", "base", "--root", str(root), "--runs", str(runs)]
    argv += ["--obj-id", "1", "--dry-run"]
    for item in overrides:
        argv += ["--override", item]
    return argv


@pytest.mark.parametrize(
    "item",
    ["work_dir=/tmp/elsewhere", "data_root=x", "keypoints_3d=k.json", "data-root=x"],
)
def test_train_refuses_path_overrides(item, tmp_path, capsys):
    assert cli.main(_train_argv("hopev2", tmp_path / "root", tmp_path / "runs", item)) == 1
    assert "--root, --runs and --obj-id" in capsys.readouterr().out


@pytest.mark.parametrize(
    "item",
    [
        "init_from=/tmp/other.pth",
        "resume=/tmp/last.pth",
        "require_no_init_from=false",
        "require_pretrained_backbone=false",
    ],
)
def test_train_refuses_launch_overrides_for_the_base(item, tmp_path, capsys):
    assert cli.main(_train_argv("hopev2", tmp_path / "root", tmp_path / "runs", item)) == 1
    assert "launch keys" in capsys.readouterr().out
    assert cli.main(_train_argv("hopev2", tmp_path / "root", tmp_path / "runs", "epochs=3")) == 0


def test_training_namespace_refuses_path_overrides():
    recipe = load_recipe("hopev2")
    paths = {"data_root": "d", "keypoints_3d": "k", "work_dir": "w"}
    with pytest.raises(ValueError, match="dataset paths"):
        training_namespace(recipe.base, paths, {"work_dir": "elsewhere"})


def _copy_runs(runs: Path, destination: Path) -> Path:
    shutil.copytree(runs, destination, ignore=shutil.ignore_patterns("validation", "test"))
    return destination


def test_validation_rerun_scores_only_missing_objects(trained, tmp_path, capsys):
    recipe, root, smoke_runs = trained
    runs = _copy_runs(smoke_runs, tmp_path / "runs")
    shutil.copytree(layout.object_run(runs, "base", 1), layout.object_run(runs, "base", 2))
    paths = ["--root", str(root), "--runs", str(runs), "--device", "cpu"]
    val = ["evaluate", str(recipe), "--split", "val", *paths]
    assert cli.main([*val, "--obj-id", "1"]) == 0
    first = layout.object_run(runs, "base", 1) / layout.VALIDATION_PATH
    second = layout.object_run(runs, "base", 2) / layout.VALIDATION_PATH
    scored = first.read_bytes()
    assert not second.exists()
    capsys.readouterr()
    assert cli.main(val) == 0
    out = capsys.readouterr().out
    assert "object 1: already scored" in out and "object 2: validation" in out
    assert first.read_bytes() == scored and second.is_file()
    stale = json.loads(first.read_text())
    stale["checkpoint_sha256"] = "0" * 64
    first.write_text(json.dumps(stale))
    with pytest.raises(SystemExit, match="--force"):
        cli.main(val)
    assert cli.main([*val, "--force"]) == 0
    assert json.loads(first.read_text())["checkpoint_sha256"] != "0" * 64


def _retrain(run: Path) -> None:
    """Stand in for retraining: a new ``last.pth`` with the same weights but other bytes."""
    path = run / layout.CHECKPOINT_PATH
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["best_kp"] = 123.0
    torch.save(checkpoint, path)


def test_forced_test_export_restarts_after_retraining(trained, tmp_path, capsys):
    recipe, root, smoke_runs = trained
    fetch = ["prepare", str(recipe), "--root", str(root), "--parts", "test"]
    assert cli.main([*fetch, "--allow-test-inputs"]) == 0
    runs = _copy_runs(smoke_runs, tmp_path / "runs")
    paths = ["--root", str(root), "--runs", str(runs), "--device", "cpu"]
    test = ["evaluate", str(recipe), "--split", "test", *paths, "--policy", "base"]
    test += ["--allow-partial-bank", "--record-time"]  # timed: the bank journal, no cache
    assert cli.main(test) == 0
    out = runs / "test" / "base"
    csv_path = out / "doperbase_hopev2-test.csv"
    _retrain(layout.object_run(runs, "base", 1))
    with pytest.raises(SystemExit, match="--force"):
        cli.main(test)
    csv_path.unlink()
    capsys.readouterr()
    assert cli.main(test) == 1
    assert "journal identity changed" in capsys.readouterr().out
    assert cli.main([*test, "--force"]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["resumed_images"] == 0 and summary["images"] == 3
    assert csv_path.is_file() and (out / "selection.json").is_file()


def test_untimed_base_policy_is_a_cached_fixed_bank(trained, tmp_path, monkeypatch):
    from doper.bop import export

    recipe, root, smoke_runs = trained
    fetch = ["prepare", str(recipe), "--root", str(root), "--parts", "test"]
    assert cli.main([*fetch, "--allow-test-inputs"]) == 0
    runs = _copy_runs(smoke_runs, tmp_path / "runs")
    paths = ["--root", str(root), "--runs", str(runs), "--device", "cpu"]
    test = ["evaluate", str(recipe), "--split", "test", *paths, "--allow-partial-bank"]
    skipped = layout.object_run(runs, "static_pseudo", 1)
    skipped.mkdir(parents=True)
    (skipped / layout.SKIPPED_NAME).write_text(json.dumps({"labels": 3, "minimum_labels": 20}))
    assert cli.main([*test, "--policy", "static_pseudo"]) == 0  # caches the base model

    def no_model(*args, **kwargs):
        raise AssertionError("a cached model must not be loaded")

    monkeypatch.setattr(export, "load_bank", no_model)
    assert cli.main([*test, "--policy", "base"]) == 0
    out = runs / "test" / "base"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["cached_models"] == 1 and summary["policy"] == "base"
    with (out / "MODEL_MATRIX.csv").open() as stream:
        assert [(r["actual_method"], r["status"]) for r in csv.DictReader(stream)] == [
            ("base", "base")
        ]
    assert json.loads((out / "SKIPS.json").read_text())["cases"] == []
    fixed = (out / "doperbase_hopev2-test.csv").read_bytes()
    assert (
        fixed
        == (runs / "test" / "static_pseudo" / "doperstaticfallback_hopev2-test.csv").read_bytes()
    )


def test_failed_test_export_publishes_no_selection(trained, tmp_path, monkeypatch):
    from doper.bop import export

    recipe, root, smoke_runs = trained
    runs = _copy_runs(smoke_runs, tmp_path / "runs")
    out = runs / "test" / "base"
    out.mkdir(parents=True)
    (out / "selection.json").write_text("{}")
    (out / "summary.json").write_text("{}")

    def fail(*args, **kwargs):
        raise ValueError("export failed")

    monkeypatch.setattr(export, "run_bank", fail)
    paths = ["--root", str(root), "--runs", str(runs), "--device", "cpu"]
    test = ["evaluate", str(recipe), "--split", "test", *paths, "--policy", "base"]
    test.append("--allow-partial-bank")
    assert cli.main(test) == 1
    assert (out / "selection.json").read_text() == "{}"
    assert cli.main([*test, "--force"]) == 1
    assert not (out / "selection.json").exists() and not (out / "summary.json").exists()


def test_moved_dataset_root_keeps_working_and_prepare_force_repairs_links(
    trained, tmp_path, capsys
):
    recipe, smoke_root, _ = trained
    root = tmp_path / "moved"
    shutil.copytree(smoke_root, tmp_path / "original", symlinks=True)
    (tmp_path / "original").rename(root)
    assert cli.main(["prepare", str(recipe), "--root", str(root), "--obj-id", "1"]) == 0
    assert "already prepared and verified" in capsys.readouterr().out
    assert prepared_receipt(root, 1)["membership"]["train"]["images"] > 0
    images = object_dir(root, 1) / "dataset" / "train" / "images"
    legacy = sorted(images.iterdir())[0]
    legacy.unlink()
    legacy.symlink_to(tmp_path / "original" / "gone" / legacy.name)  # an old absolute link
    runs = tmp_path / "runs"
    argv = ["train", str(recipe), "--stage", "base", "--root", str(root), "--runs", str(runs)]
    capsys.readouterr()
    assert cli.main([*argv, "--obj-id", "1"]) == 1
    message = capsys.readouterr().out
    assert str(legacy) in message and "doper prepare --force --obj-id 1" in message
    assert not runs.exists()
    prepare = ["prepare", str(recipe), "--root", str(root), "--obj-id", "1"]
    assert cli.main(prepare) == 1
    assert cli.main([*prepare, "--force"]) == 0
    assert prepared_receipt(root, 1)["membership"]["train"]["images"] > 0
    assert all(p.exists() and not p.readlink().is_absolute() for p in images.iterdir())


def test_train_adapt_refuses_method_and_override_misuse(tmp_path, capsys):
    common = ["--root", str(tmp_path / "root"), "--runs", str(tmp_path / "runs"), "--dry-run"]
    adapt = ["train", "hopev2", "--stage", "adapt", "--obj-id", "2", *common]
    assert cli.main(adapt) == 1
    assert "--method" in capsys.readouterr().out
    assert cli.main([*adapt, "--method", "test_tta"]) == 1
    for item in ("init_from=x.pth", "backbone=resnet18", "epochs=5", "mix_per_batch=[2]"):
        assert cli.main([*adapt, "--method", "static_pseudo", "--override", item]) == 1, item
    assert cli.main([*adapt, "--method", "static_pseudo", "--override", "num_workers=0"]) == 0
    base = ["train", "hopev2", "--stage", "base", "--obj-id", "2", *common]
    assert cli.main([*base, "--method", "static_pseudo"]) == 1
    assert cli.main([*base, "--labels-only"]) == 1


def test_evaluate_arms_come_from_the_recipe(tmp_path, capsys):
    paths = ["--root", str(tmp_path / "root"), "--runs", str(tmp_path / "runs"), "--dry-run"]
    val = ["evaluate", "handal", "--split", "val", *paths]
    assert cli.main([*val, "--arm", "test_tta"]) == 0
    assert "test_tta/obj_000001" in capsys.readouterr().out
    assert cli.main([*val, "--arm", "nope"]) == 1
    assert cli.main(["evaluate", "tless", "--split", "val", *paths, "--arm", "gt_box_mask"]) == 1
    # Follow-up rounds and selection aliases are scored in their run directory, for their
    # object only.
    for arm in ("obj8_loop/round08", "static_iterative_round8"):
        capsys.readouterr()
        assert cli.main([*val, "--arm", arm]) == 0
        out = capsys.readouterr().out
        assert "obj8_loop/round08/obj_000008" in out and "obj_000001" not in out
    assert cli.main([*val, "--arm", "obj8_loop/round08", "--obj-id", "1"]) == 1
    assert cli.main([*val, "--arm", "obj8_loop/round09"]) == 1


def test_followup_round_is_scored_and_ranked_through_the_cli(trained, tmp_path, capsys):
    """A follow-up round scored by ``doper evaluate`` (by path or alias) is a ranked selection
    candidate; a present but unscored candidate is reported, not dropped silently."""
    from doper.cli.evaluate import _selection_record, select_bank

    recipe, root, smoke_runs = trained
    data = yaml.safe_load(Path(recipe).read_text())
    data["followups"] = {"loop": {"obj_id": 1, "teacher": "base", "rounds": 2}}
    data["test"]["aliases"] = {"static_iterative_round2": {"run": "loop/round02", "objects": [1]}}
    fixture = tmp_path / "followup.yaml"
    fixture.write_text(yaml.safe_dump(data, sort_keys=False))
    runs = _copy_runs(smoke_runs, tmp_path / "runs")
    round2 = _student_from_base(runs, "loop/round02", 1)
    _student_from_base(runs, "gt_box_mask", 1)
    paths = ["--root", str(root), "--runs", str(runs), "--device", "cpu"]
    val = ["evaluate", str(fixture), "--split", "val", *paths, "--obj-id", "1"]
    assert cli.main(val) == 0
    assert cli.main([*val, "--arm", "loop/round02"]) == 0
    assert (round2 / layout.VALIDATION_PATH).is_file()
    capsys.readouterr()
    assert cli.main([*val, "--arm", "static_iterative_round2"]) == 0
    assert "already scored" in capsys.readouterr().out
    _, choices = select_bank(load_recipe(fixture), root, runs, "selected", True)
    assert choices[1].status == "selected"
    ranked = {c.arm: c.single_ar for c in choices[1].candidates}
    assert ranked["static_iterative_round2"] is not None and ranked["base"] is not None
    assert choices[1].unscored == ("gt_box_mask",)
    assert _selection_record(choices)["1"]["unscored"] == ["gt_box_mask"]
    assert "gt_box_mask" in capsys.readouterr().out


def test_selected_bank_omits_skipped_methods(trained, tmp_path):
    from doper.cli.evaluate import select_bank

    recipe, root, smoke_runs = trained
    runs = _copy_runs(smoke_runs, tmp_path / "runs")
    skipped = layout.object_run(runs, "static_pseudo", 1)
    skipped.mkdir(parents=True)
    (skipped / layout.SKIPPED_NAME).write_text("{}")
    bank, choices = select_bank(load_recipe(recipe), root, runs, "selected", True)
    assert choices[1].chosen.arm == "base" and bank[1] == layout.object_run(runs, "base", 1)
    (skipped / layout.SKIPPED_NAME).unlink()
    with pytest.raises(ValueError, match="unfinished"):
        select_bank(load_recipe(recipe), root, runs, "selected", True)


def _student_from_base(runs: Path, method: str, obj: int) -> Path:
    """A completed student run: the base's weights re-saved (other bytes) with its receipt."""
    base = layout.object_run(runs, "base", obj)
    run = layout.object_run(runs, method, obj)
    shutil.copytree(base, run, ignore=shutil.ignore_patterns("validation", "completed.json"))
    _retrain(run)
    digest = sha256(run / layout.CHECKPOINT_PATH)
    done = json.loads((base / "completed.json").read_text())
    receipt = {
        "stage": "adapt",
        "method": method,
        "obj_id": obj,
        "completed_epochs": 3,
        "labels": 25,
        "checkpoint_sha256": digest,
        "student_sha256": digest,
        "keypoints_sha256": done["keypoints_sha256"],
        "base_teacher_sha256": done["checkpoint_sha256"],
    }
    (run / "completed.json").write_text(json.dumps(receipt))
    return run


def test_fixed_policy_exports_reuse_the_model_cache(trained, tmp_path, capsys):
    recipe, root, smoke_runs = trained
    fetch = ["prepare", str(recipe), "--root", str(root), "--parts", "test"]
    assert cli.main([*fetch, "--allow-test-inputs"]) == 0
    runs = _copy_runs(smoke_runs, tmp_path / "runs")
    skipped = layout.object_run(runs, "static_pseudo", 1)
    skipped.mkdir(parents=True)
    (skipped / layout.SKIPPED_NAME).write_text(json.dumps({"labels": 3, "minimum_labels": 20}))
    _student_from_base(runs, "gt_box_mask", 1)
    paths = ["--root", str(root), "--runs", str(runs), "--device", "cpu"]
    test = ["evaluate", str(recipe), "--split", "test", *paths, "--allow-partial-bank"]
    assert cli.main([*test, "--policy", "nope"]) == 1
    assert "static_pseudo" in capsys.readouterr().out
    assert cli.main([*test, "--policy", "static_pseudo"]) == 0
    out = runs / "test" / "static_pseudo"
    assert (out / "doperstaticfallback_hopev2-test.csv").is_file()
    skips = json.loads((out / "SKIPS.json").read_text())
    assert [(c["object_id"], c["reason"]) for c in skips["cases"]] == [(1, "skipped")]
    with (out / "MODEL_MATRIX.csv").open() as stream:
        assert [r["actual_method"] for r in csv.DictReader(stream)] == ["base"]
    cache = runs / "test" / "cache"
    assert len(list(cache.glob("v*/*/COMPLETE.json"))) == 1
    assert cli.main([*test, "--policy", "dynamic_pseudo"]) == 1
    assert "no dynamic_pseudo endpoint" in capsys.readouterr().out
    assert cli.main([*test, "--policy", "gt_box_mask"]) == 0
    assert len(list(cache.glob("v*/*/COMPLETE.json"))) == 2
    with pytest.raises(SystemExit, match="--force"):
        cli.main([*test, "--policy", "gt_box_mask"])
    assert cli.main([*test, "--policy", "gt_box_mask", "--force"]) == 0
    summary = json.loads((runs / "test" / "gt_box_mask" / "summary.json").read_text())
    assert summary["policy"] == "gt_box_mask" and summary["images"] == 3
    assert len(list(cache.glob("v*/*/COMPLETE.json"))) == 2


def _posing_bank(bank, **kwargs):
    """``export.load_bank`` stand-in: every object finds one posed detection on every image."""
    import cv2
    import numpy as np

    from doper.bop import export
    from doper.infer import Detection

    def detect(image, camera_matrix):
        det = Detection(np.zeros(4), 0.9, np.zeros((4, 2)), np.ones(4))
        det.rvec = cv2.Rodrigues(np.eye(3))[0]
        det.tvec = np.array([0.0, 0.0, 500.0])
        return [det]

    return [
        export.BankModel(
            obj,
            detect,
            sha256(Path(bank[obj]) / layout.CHECKPOINT_PATH),
            sha256(Path(bank[obj]) / layout.KEYPOINTS_PATH),
        )
        for obj in sorted(bank)
    ]


def test_a_timed_policy_rescans_and_records_time(trained, tmp_path, capsys, monkeypatch):
    from doper.bop import export

    monkeypatch.setattr(export, "load_bank", _posing_bank)
    recipe, root, smoke_runs = trained
    fetch = ["prepare", str(recipe), "--root", str(root), "--parts", "test"]
    assert cli.main([*fetch, "--allow-test-inputs"]) == 0
    data = yaml.safe_load(Path(recipe).read_text())
    timed = {"method": "gt_box_mask", "csv_method": "dopergttimed", "record_time": True}
    data["test"]["policies"]["gt_timed"] = timed
    timed_recipe = recipe.parent / "fixture_timed.yaml"
    timed_recipe.write_text(yaml.safe_dump(data, sort_keys=False))
    runs = _copy_runs(smoke_runs, tmp_path / "runs")
    _student_from_base(runs, "gt_box_mask", 1)
    paths = ["--root", str(root), "--runs", str(runs), "--device", "cpu"]
    test = ["evaluate", str(timed_recipe), "--split", "test", *paths, "--allow-partial-bank"]
    assert cli.main([*test, "--policy", "gt_timed", "--dry-run"]) == 0
    assert "timed, every model rescanned" in capsys.readouterr().out
    assert cli.main([*test, "--policy", "gt_timed"]) == 0
    with (runs / "test" / "gt_timed" / "dopergttimed_hopev2-test.csv").open() as stream:
        times = [row["time"] for row in csv.DictReader(stream)]
    assert len(times) == 3 and all(float(t) > 0 for t in times)
    assert not (runs / "test" / "cache").exists()
    assert cli.main([*test, "--policy", "gt_box_mask"]) == 0  # untimed: cached, no runtime
    with (runs / "test" / "gt_box_mask" / "dopergtboxmask_hopev2-test.csv").open() as stream:
        assert [row["time"] for row in csv.DictReader(stream)] == ["-1"] * 3
    assert (runs / "test" / "cache").is_dir()


GENERATED = {
    "render_config": "hot3d_obj33",
    "layouts": 4,
    "frames_per_layout": 3,
    "base_seed": 100,
    "val_modulo": 4,
    "val_residue": 3,
    "empty_masks": "exclude",
}


def _generated_recipe(fixture: Path, path: Path, **changes: object) -> Path:
    """The fixture recipe with object 2 generated (a 4-layout render campaign)."""
    data = yaml.safe_load(Path(fixture).read_text())
    data.pop("generated_objects", None)
    spec = {**GENERATED, **changes}
    path.write_text(yaml.safe_dump({**data, "generated": {2: spec}}, sort_keys=False))
    return path


def _render_root(path: Path, models: Path, keypoints: Path, scene=None) -> Path:
    split = {"modulo": GENERATED["val_modulo"], "val_residue": GENERATED["val_residue"]}
    config = render_config(
        "hot3d_obj33",
        dataset="fixture",
        target_object=2,
        layouts=GENERATED["layouts"],
        frames_per_layout=3,
        base_seed=GENERATED["base_seed"],
        split=split,
    )
    return write_render_root(
        path,
        config=config,
        keypoints=keypoints,
        models=models,
        scene=scene or (lambda d, layout: write_scene(d, 30 + layout)),
    )


def _blank_target(scene: Path, layout: int) -> None:
    """``write_scene``; in layout 0, object 2 of frame 0 is fully occluded: an empty visible
    mask and a zero visible count (its keypoints still pass the depth test, so it is converted;
    one such annotation keeps the projection audit's off-mask mean under its bound)."""
    write_scene(scene, 30 + layout)
    if layout:
        return
    mask = scene / "mask_visib" / "000000_000001.png"
    cv2.imwrite(str(mask), np.zeros_like(cv2.imread(str(mask), cv2.IMREAD_UNCHANGED)))
    info = json.loads((scene / "scene_gt_info.json").read_text())
    info["0"][1].update(px_count_visib=0, visib_fract=0.0)
    (scene / "scene_gt_info.json").write_text(json.dumps(info))


@pytest.mark.parametrize("policy", ["keep", "exclude", "fail"])
def test_generated_empty_masks_policy(tmp_path, policy, capsys):
    """keep passes empty-mask annotations through (HANDAL obj40, as the source merge did),
    exclude drops them (HOT3D obj33), fail refuses the preparation."""
    root = tmp_path / "root"
    profile = make_bop_download(root, tmp_path / "profile")
    fixture = write_recipe(tmp_path / "fixture.yaml", "hopev2", profile)
    recipe = _generated_recipe(fixture, tmp_path / "generated.yaml", empty_masks=policy)
    models = tmp_path / "models"
    write_models(models)
    frozen = select_cad_keypoints(models / "obj_000002.ply", tmp_path / "frozen", 2, 64, "mm")
    render = _render_root(tmp_path / "render", models, frozen, scene=_blank_target)
    argv = ["prepare", str(recipe), "--root", str(root), "--parts", "val", "--obj-id", "2"]
    code = cli.main([*argv, "--generated-root", str(render)])
    if policy == "fail":
        assert code == 1 and "empty visible mask" in capsys.readouterr().out
        return
    assert code == 0
    receipt = prepared_receipt(root, 2)
    counts = {s: receipt["membership"][s]["annotations"] for s in ("train", "val")}
    assert sum(counts.values()) == 12 - (policy == "exclude")


def test_prepare_keypoints_only_writes_the_pinned_selection(tmp_path, capsys):
    """The frozen keypoints a campaign is staged with: the pinned official mesh only."""
    root = tmp_path / "root"
    profile = make_bop_download(root, tmp_path / "profile")
    fixture = write_recipe(tmp_path / "fixture.yaml", "hopev2", profile)
    models = tmp_path / "models"
    write_models(models)
    expected = select_cad_keypoints(models / "obj_000002.ply", tmp_path / "ref", 2, 64, "mm")
    pin = sha256(expected)
    recipe = _generated_recipe(fixture, tmp_path / "generated.yaml", keypoints_sha256=pin)
    argv = ["prepare", str(recipe), "--root", str(root), "--obj-id", "2", "--keypoints-only"]
    assert cli.main([*argv, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "fixture_models.zip" in out and pin in out and "nothing was read" in out
    assert cli.main([*argv, "--obj-id", "1"]) == 1  # generated objects only, one at a time
    assert cli.main(["prepare", str(recipe), "--root", str(root), "--keypoints-only"]) == 1
    capsys.readouterr()
    assert cli.main(argv) == 0
    written = object_dir(root, 2) / "geometry" / "keypoints_3d.json"
    assert written.read_bytes() == expected.read_bytes()
    assert f"keypoints_sha256 {pin}" in capsys.readouterr().out
    assert not (root / "data").exists()  # nothing extracted: the mesh is read from the archive
    written.unlink()
    wrong = _generated_recipe(fixture, tmp_path / "wrong.yaml", keypoints_sha256="0" * 64)
    assert (
        cli.main(["prepare", str(wrong), "--root", str(root), "--obj-id", "2", "--keypoints-only"])
        == 1
    )
    assert "pinned" in capsys.readouterr().out and not written.exists()


def test_full_bank_still_refuses_missing_generated_objects(trained, tmp_path, capsys):
    """Without a prepared render root, a generated object keeps every bank partial."""
    fixture, root, smoke_runs = trained
    recipe = _generated_recipe(fixture, tmp_path / "generated.yaml")
    runs = _copy_runs(smoke_runs, tmp_path / "runs")
    _student_from_base(runs, "static_pseudo", 1)
    paths = ["--root", str(root), "--runs", str(runs), "--device", "cpu"]
    for policy in ("static_pseudo", "selected", "base"):
        test = ["evaluate", str(recipe), "--split", "test", *paths, "--policy", policy]
        capsys.readouterr()
        assert cli.main(test) == 1
        out = capsys.readouterr().out
        assert "allow-partial-bank" in out and "--generated-root" in out


def test_full_bank_without_partial_flag(trained, tmp_path, capsys):
    """A generated object prepared from its render root joins the bank like any object."""
    fixture, smoke_root, smoke_runs = trained
    recipe = _generated_recipe(fixture, tmp_path / "generated.yaml")
    root = tmp_path / "root"  # the smoke root without the official preparation of object 2
    shutil.copytree(
        smoke_root,
        root,
        symlinks=True,
        ignore=lambda d, names: ["obj_000002"] if Path(d).name == "objects" else [],
    )
    models = tmp_path / "models"
    write_models(models)
    frozen = select_cad_keypoints(models / "obj_000002.ply", tmp_path / "frozen", 2, 64, "mm")
    render = _render_root(tmp_path / "render", models, frozen)
    refused = ["prepare", str(recipe), "--root", str(root), "--obj-id", "2", "--dry-run"]
    assert cli.main(refused) == 1  # its training data needs the render root
    assert "--generated-root" in capsys.readouterr().out
    prepare = [
        "prepare",
        str(recipe),
        "--root",
        str(root),
        "--parts",
        "test",
        "--allow-test-inputs",
        "--obj-id",
        "2",
    ]
    assert cli.main([*prepare, "--generated-root", str(render), "--dry-run"]) == 0
    assert "generated-obj000002" in capsys.readouterr().out
    assert cli.main([*prepare, "--generated-root", str(render)]) == 0
    receipt = prepared_receipt(root, 2)
    identity = json.loads((object_dir(root, 2) / "identity.json").read_text())
    assert identity["generated"]["render_config"] == "hot3d_obj33"
    assert identity["convert"]["split"] == "layout_modulo"
    assert receipt["membership"]["val"]["scenes"] == ["000003"]
    assert receipt["membership"]["train"]["scenes"] == ["000000", "000001", "000002"]
    runs = _copy_runs(smoke_runs, tmp_path / "runs")
    train = [
        "train",
        str(recipe),
        "--stage",
        "base",
        "--root",
        str(root),
        "--runs",
        str(runs),
        "--obj-id",
        "2",
        "--dry-run",
    ]
    assert cli.main(train) == 0  # the same base stage (and completed.json) as official objects
    assert "obj_000002/completed.json" in capsys.readouterr().out
    shutil.copytree(layout.object_run(runs, "base", 1), layout.object_run(runs, "base", 2))
    paths = ["--root", str(root), "--runs", str(runs), "--device", "cpu"]
    assert cli.main(["evaluate", str(recipe), "--split", "test", *paths]) == 0
    summary = json.loads((runs / "test" / "selected" / "summary.json").read_text())
    assert summary["missing_objects"] == [] and summary["images"] == 3
    selection = json.loads((runs / "test" / "selected" / "selection.json").read_text())
    assert sorted(selection) == ["1", "2"]


def _heatmap_bank(bank, **kwargs):
    """``export.load_bank`` stand-in: posed detections; raw heatmaps with ``raw_outputs``."""
    import cv2
    import numpy as np

    from doper.bop import export
    from doper.infer import Detection

    raw = kwargs.get("raw_outputs", False)
    models = []
    for obj in sorted(bank):
        run = Path(bank[obj])
        points = export.keypoints_mm(run / layout.KEYPOINTS_PATH)

        def detect(image, camera_matrix, points=points, obj=obj):
            rvec, tvec = np.array([[0.3], [-0.2], [0.1 * obj]]), np.array([[5.0], [-4.0], [600.0]])
            pixels = cv2.projectPoints(points, rvec, tvec, camera_matrix, None)[0].reshape(-1, 2)
            side = 1.3 * float(np.ptp(pixels, axis=0).max()) + 1.0
            center = pixels.mean(axis=0)
            crop = np.array([[side / 255, 0.0, center[0] - side / 2],
                             [0.0, side / 255, center[1] - side / 2]])  # fmt: skip
            inverse = cv2.invertAffineTransform(crop)
            bins = (pixels @ inverse[:, :2].T + inverse[:, 2]) * (63 / 255)
            grid = np.arange(64.0)
            logits = -((grid[None, None, :] - bins[:, 0, None, None]) ** 2
                       + (grid[None, :, None] - bins[:, 1, None, None]) ** 2) / 4.5  # fmt: skip
            scores = (0.9, 0.6)
            dets = []
            for score, offset in zip(scores, (8.0, -6.0), strict=True):
                det = Detection(
                    np.array([*pixels.min(0), *pixels.max(0)], np.float32), score,
                    pixels.astype(np.float32), np.full(len(points), 0.9, np.float32),
                    keypoint_heatmap=logits.astype(np.float32) if raw else None,
                    crop_to_image=crop if raw else None,
                )  # fmt: skip
                det.rvec, det.tvec = rvec + 0.03, tvec + offset
                dets.append(det)
            return dets

        models.append(
            export.BankModel(
                obj,
                detect,
                sha256(run / layout.CHECKPOINT_PATH),
                sha256(run / layout.KEYPOINTS_PATH),
                {
                    "config_sha256": "f" * 64,
                    "keypoint_head": "heatmap",
                    "crop_size": 256,
                    "keypoint_heatmap_beta": 1.0,
                },
            )  # fmt: skip
        )
    return models


def test_variant_exports_share_the_raw_journal(trained, tmp_path, monkeypatch):
    from doper.bop import export, variants

    recipe, root, smoke_runs = trained
    fetch = ["prepare", str(recipe), "--root", str(root), "--parts", "test"]
    assert cli.main([*fetch, "--allow-test-inputs"]) == 0
    paths = ["--root", str(root), "--device", "cpu", "--allow-partial-bank", "--policy", "base"]
    test = ["evaluate", str(recipe), "--split", "test", *paths]
    # The real smoke model goes through the raw-output inference path.
    real = _copy_runs(smoke_runs, tmp_path / "real")
    assert cli.main([*test, "--runs", str(real), "--variant", "singleheatmap"]) == 0
    monkeypatch.setattr(export, "load_bank", _heatmap_bank)
    runs = _copy_runs(smoke_runs, tmp_path / "runs")
    test += ["--runs", str(runs)]
    assert cli.main(test) == 0
    out = runs / "test" / "base"

    def rows(path):
        with path.open() as stream:
            return list(csv.DictReader(stream))

    def identity(table):
        return [(r["scene_id"], r["im_id"], r["obj_id"], r["score"]) for r in table]

    base = rows(out / "doperbase_hopev2-test.csv")
    assert len(base) == 6
    assert cli.main([*test, "--variant", "singleview"]) == 0
    assert rows(out / "singleview" / "doperbasesingleview_hopev2-test.csv") == base
    cache = runs / "test" / "cache"
    # Every export records the forward pass (journal generation) it used, per object; a resumed
    # variant export recomputes an object whose journal was replaced since.
    (generation,) = cache.glob(f"v*/*/{export.CACHE_PAYLOAD_DIR}/{export.GENERATION_NAME}")
    first = generation.read_text()
    policy_summary = json.loads((out / "summary.json").read_text())
    assert policy_summary["journal_generations"] == {"1": [first]}
    single = out / "singleview"
    manifest = json.loads((single / variants.MANIFEST_NAME).read_text())
    assert manifest["objects"]["1"]["journal_generations"] == [first]
    assert json.loads((single / "summary.json").read_text())["journal_generations"] == {
        "1": [first]
    }
    single_csv = single / "doperbasesingleview_hopev2-test.csv"
    single_csv.unlink()  # an interrupted export: the per-object results stay
    assert cli.main([*test, "--variant", "singleview"]) == 0
    assert json.loads((single / "summary.json").read_text())["resumed_objects"] == 1
    generation.write_text("replaced-by-another-forward-pass")
    single_csv.unlink()
    assert cli.main([*test, "--variant", "singleview"]) == 0
    resumed = json.loads((single / "summary.json").read_text())
    assert resumed["resumed_objects"] == 0
    assert resumed["journal_generations"] == {"1": ["replaced-by-another-forward-pass"]}
    generation.write_text(first)
    single_csv.unlink()
    assert cli.main([*test, "--variant", "singleview"]) == 0
    stores = f"v*/*{export.HEATMAP_ENTRY_SUFFIX}"
    assert not list(cache.glob(stores)) and not list(cache.rglob("*.npz"))
    assert cli.main([*test, "--variant", "singleheatmap", "--optimizer", "powell"]) == 0
    folder = out / "singleheatmap"
    heatmap = rows(folder / "doperbasesingleheatmap_hopev2-test.csv")
    assert identity(heatmap) == identity(base)
    assert all(h["t"] != b["t"] for h, b in zip(heatmap, base, strict=True))
    manifest = json.loads((folder / variants.MANIFEST_NAME).read_text())
    assert manifest["optimizer"] == "powell" and manifest["preparation"] == "numpy"
    assert manifest["crop_size"] == [256] and manifest["policy"] == "base"
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["variant"] == "singleheatmap" and summary["single_heatmap_accepted"] == 6
    # HOPEv2 scene 1 has no extrinsics: every multiheatmap row keeps the single-view pose.
    assert cli.main([*test, "--variant", "multiheatmap"]) == 0
    multi = rows(out / "multiheatmap" / "doperbasemultiheatmap_hopev2-test.csv")
    assert identity(multi) == identity(base)
    for got, want in zip(multi, base, strict=True):
        np.testing.assert_allclose(
            np.array(got["R"].split(), float), np.array(want["R"].split(), float), atol=1e-9
        )
        assert got["t"] == want["t"]
    journals = [p for p in cache.glob("v*/*/COMPLETE.json") if p.parent.suffix == ""]
    assert len(journals) == 1 and len(list(cache.glob(stores))) == 1
    assert len(list(cache.rglob("*.npz"))) == 3  # one archive per test image, filled once
    with pytest.raises(SystemExit, match="--force"):
        cli.main([*test, "--variant", "singleheatmap"])


def test_report_timing_protocols(trained, tmp_path, capsys):
    """``report --timing`` runs the direct and the deployment protocol on a bank."""
    recipe, root, smoke_runs = trained
    fetch = ["prepare", str(recipe), "--root", str(root), "--parts", "test"]
    assert cli.main([*fetch, "--allow-test-inputs"]) == 0
    runs = _copy_runs(smoke_runs, tmp_path / "runs")
    report = ["report", str(recipe), "--runs", str(runs), "--root", str(root), "--policy",
              "base", "--device", "cpu", "--allow-partial-bank"]  # fmt: skip
    capsys.readouterr()
    assert cli.main([*report, "--timing", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "timing base (deployment)" in out and "lbfgsb, numpy preparation" in out
    assert cli.main([*report, "--protocol", "direct", "--dry-run"]) == 1
    assert "need --timing" in capsys.readouterr().out
    assert cli.main([*report, "--timing", "--protocol", "direct", "--optimizer", "powell",
                     "--dry-run"]) == 1  # fmt: skip
    assert cli.main([*report, "--timing", "--workers", "0", "--dry-run"]) == 1
    assert "--workers must be >= 1" in capsys.readouterr().out
    assert cli.main([*report, "--timing", "--optimizer", "powell", "--workers", "2"]) == 0
    folder = runs / "timing" / "base" / "deployment"
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["optimizer"] == "powell" and manifest["preparation"] == "numpy"
    assert set(manifest["optimizers_run"]) <= {"powell"} and manifest["workers"] == 2
    assert manifest["policy"] == "base" and manifest["complete"] is True
    assert manifest["hardware"]["device"] == "cpu" and "torch" in manifest["versions"]
    summary = json.loads((folder / "summary.json").read_text())
    names = set(manifest["variants"])
    assert set(summary["variants"]) == names and summary["optimizer"] == "powell"
    missing = sorted(set(load_recipe(str(recipe)).objects) - {1})
    assert summary["missing_objects"] == missing and summary["partial_bank"] == bool(missing)
    requests = [json.loads(line) for line in (folder / "requests.jsonl").read_text().splitlines()]
    assert len(requests) == manifest["groups"] * 2 * len(names)
    assert {r["variant"] for r in requests} == names
    with pytest.raises(SystemExit, match="--force"):
        cli.main([*report, "--timing"])
    assert cli.main([*report, "--timing", "--protocol", "direct"]) == 0
    direct = json.loads((runs / "timing" / "base" / "direct" / "summary.json").read_text())
    assert direct["images"] == 3 and direct["protocol"]["explicit_warmup_passes"] == 0
    assert direct["partial_bank"] == bool(missing)


def test_force_from_refused_for_bop(bop_fixture_root, capsys):
    argv = ["prepare", "handal", "--root", str(bop_fixture_root), "--force-from", "geometry"]
    assert cli.main(argv) == 1
    assert "--force-from" in capsys.readouterr().out
