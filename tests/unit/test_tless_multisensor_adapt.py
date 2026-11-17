# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T-LESS multisensor adaptation: recipe namespace, uncapped budget, review gate, fixed bank."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
import tless_multisensor_data as fixture
import torch
import yaml
from bank_runs import sha256, write_base, write_student

from doper import cli
from doper.bop import banks, layout, multisensor, review
from doper.keypoints import keypoints_3d_fingerprint
from doper.recipes.schema import MultisensorSpec, load_recipe, parse_recipe
from doper.training import adaptation

SMALL = "convnext_small.dinov3_lvd1689m"
ARCHITECTURE = {
    "backbone": SMALL,
    "num_keypoints": 64,
    "det_input_size": 640,
    "crop_size": 512,
    "keypoint_head": "heatmap",
}
PATHS = {
    "data_root": "<data_root>",
    "keypoints_3d": "<keypoints_3d>",
    "work_dir": "<work_dir>",
    "teacher": "<init_from>",
    "sensors": ("<primesense>", "<kinect>", "<canon>"),
}
# configs/pbronly/tless_small_multisensor_adapt3.yaml after tless_multisensor_run.configs():
# the gt-keypoints min-mix update, paths replaced by placeholders.
SOURCE_ADAPT = {
    "data_root": "<data_root>",
    "keypoints_3d": "<keypoints_3d>",
    "work_dir": "<work_dir>",
    "mix_data_root": ["<primesense>", "<kinect>", "<canon>"],
    "mix_per_batch": [2, 2, 2],
    "backbone": SMALL,
    "backbone_lr_multiplier": 0.1,
    "no_pretrained": True,
    "require_pretrained_backbone": False,
    "require_no_init_from": False,
    "require_init_from": True,
    "init_from": "<init_from>",
    "resume": None,
    "num_classes": 1,
    "num_keypoints": 64,
    "det_size": 640,
    "crop_size": 512,
    "letterbox": True,
    "keypoint_head": "heatmap",
    "keypoint_heatmap_sigma": 0.75,
    "instance_sampling": "all",
    "mask_target_mode": "union",
    "epochs": 3,
    "batch_size": 16,
    "gradient_accumulation_steps": 1,
    "num_workers": 8,
    "lr": 1.0e-5,
    "warmup_iters": 20,
    "grad_clip": 5.0,
    "val_interval": 1,
    "save_interval": 1,
    "seed": 42,
    "no_amp": True,
    "max_iters_per_epoch": 0,
    "min_mix_keypoint_samples": 1,
    "min_mix_mask_only_samples": 0,
}
# Object 1 of the paper run: 19,294 all-instance PBR samples, 1,296 captures per sensor.
OBJ1_PBR, OBJ1_SENSOR = 19294, 1296
FAKE_PBR = 100


# --------------------------------------------------------------------------------------
# Recipe
# --------------------------------------------------------------------------------------
def test_adapt_namespace_matches_source():
    recipe = load_recipe("tless")
    assert recipe.adapt is not None and isinstance(recipe.adapt.multisensor, MultisensorSpec)
    assert recipe.adapt.methods == {}
    resolved, launch = adaptation.multisensor_config(recipe, PATHS)
    assert resolved == SOURCE_ADAPT
    assert launch.init_from == "<init_from>" and launch.require_init_from
    assert not launch.require_no_init_from and launch.resume is None


def test_multisensor_spec_is_validated():
    data = yaml.safe_load(Path(load_recipe("tless").source).read_text())
    for change, message in (
        (lambda s: s.update(mix_per_batch=[2, 2]), "three"),
        (lambda s: s.update(mix_per_batch=[8, 4, 4]), "PBR"),
        (lambda s: s["config"].update(epochs=4), "epochs"),
        (lambda s: s["config"].update(max_iters_per_epoch=5), "computes"),
        (lambda s: s["config"].update(backbone="resnet18"), "inherits"),
        (lambda s: s.update(min_mix_keypoint_samples=-1), "negative"),
    ):
        copy = json.loads(json.dumps(data))
        change(copy["adapt"]["multisensor"])
        with pytest.raises(ValueError, match=message):
            parse_recipe(copy)
    followup = json.loads(json.dumps(data))
    followup["followups"] = {"loop": {"obj_id": 1, "teacher": "base", "rounds": 2}}
    with pytest.raises(ValueError, match="needs adapt.methods"):
        parse_recipe(followup)
    no_stage = json.loads(json.dumps(data))
    del no_stage["adapt"]["multisensor"]
    with pytest.raises(ValueError, match="at least one method"):
        parse_recipe(no_stage)


def test_uncapped_epoch_batches_obj1():
    recipe = load_recipe("tless")
    batches = adaptation.multisensor_batches(recipe, OBJ1_PBR, (OBJ1_SENSOR,) * 3)
    assert batches == 1930 == math.ceil(OBJ1_PBR / 10)
    assert batches * adaptation.STUDENT_EPOCHS == 5790
    # A larger real stream would set the epoch: every retained sample is covered.
    assert adaptation.multisensor_batches(recipe, 10, (OBJ1_SENSOR, 1, 1)) == 648


# --------------------------------------------------------------------------------------
# Student audit
# --------------------------------------------------------------------------------------
def write_ms_student(
    run_dir: Path,
    keypoints: Path,
    sensors: tuple[int, ...],
    *,
    pbr: int = FAKE_PBR,
    global_step: int | None = None,
    init_from: str | None = None,
    resume: str | None = None,
    keypoint_samples: tuple[int, ...] | None = None,
    roots: tuple[str, ...] | None = None,
    drop_samples: bool = False,
) -> None:
    """The endpoint ``doper.train`` writes for a 10 + 2 + 2 + 2 student."""
    points = np.asarray(json.loads(keypoints.read_text())["keypoints_3d"])
    batches = max(math.ceil(n / b) for n, b in zip((pbr, *sensors), (10, 2, 2, 2), strict=False))
    keyed = keypoint_samples or sensors
    real = [
        {"samples": n, "keypoint_samples": k, "mask_only_samples": 0}
        for n, k in zip(sensors, keyed, strict=False)
    ]
    for index, stream in enumerate(real):
        if roots is not None:
            stream["root"] = roots[index]
        if drop_samples:
            del stream["samples"]
    checkpoint = {
        "model": {"weight": torch.tensor([1.0, 2.0])},
        "optimizer": {"state": {0: {"exp_avg": torch.ones(2)}}, "param_groups": []},
        "epoch": 2,
        "global_step": 3 * batches if global_step is None else global_step,
        "cfg": {
            "model": ARCHITECTURE,
            "keypoints_3d_fingerprint": keypoints_3d_fingerprint(points),
        },
        "training_provenance": {
            "init_checkpoint": {
                "path": init_from,
                "sha256": sha256(Path(init_from)) if init_from else None,
            },
            "resume_checkpoint": {"path": resume, "sha256": None},
            "train_sources": [
                {"samples": pbr, "keypoint_samples": pbr, "mask_only_samples": 0},
                *real,
            ],
        },
    }
    work = run_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, work / "last.pth")
    rows = [
        {"epoch": e, "nonfinite_microbatches": 0, "scaler_skipped_updates": 0} for e in range(3)
    ]
    (work / "epoch_timing.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_verify_multisensor_student_checks_sources_and_budget(tmp_path):
    spec = load_recipe("tless").adapt.multisensor
    keypoints = tmp_path / "kp.json"
    keypoints.write_text(json.dumps({"keypoints_3d": np.zeros((64, 3)).tolist()}))
    clean = tmp_path / "clean"
    write_ms_student(clean, keypoints, (OBJ1_SENSOR,) * 3, pbr=OBJ1_PBR)
    sensors = (OBJ1_SENSOR,) * 3
    assert adaptation.verify_multisensor_student(clean, sensors, spec, batch_size=16) == []
    short = tmp_path / "short"
    write_ms_student(short, keypoints, sensors, pbr=OBJ1_PBR, global_step=5789)
    faults = adaptation.verify_multisensor_student(short, sensors, spec, batch_size=16)
    assert any("5790" in f for f in faults)
    assert adaptation.verify_multisensor_student(clean, (OBJ1_SENSOR, 1295, OBJ1_SENSOR), spec,
                                                 batch_size=16)  # fmt: skip
    two = tmp_path / "two"
    write_ms_student(two, keypoints, (OBJ1_SENSOR,) * 2, pbr=OBJ1_PBR)
    faults = adaptation.verify_multisensor_student(two, sensors, spec, batch_size=16)
    assert any("four" in f for f in faults)
    roots = ["/data/multisensor/obj_000001/" + name for name in ("primesense", "kinect", "canon")]
    named = tmp_path / "named"
    write_ms_student(
        named, keypoints, sensors, pbr=OBJ1_PBR, roots=("primesense", "kinect", "canon")
    )
    assert not adaptation.verify_multisensor_student(
        named, sensors, spec, batch_size=16, mix_roots=roots
    )
    swapped = tmp_path / "swapped"
    write_ms_student(
        swapped, keypoints, sensors, pbr=OBJ1_PBR, roots=("kinect", "primesense", "canon")
    )
    faults = adaptation.verify_multisensor_student(
        swapped, sensors, spec, batch_size=16, mix_roots=roots
    )
    assert any("records root kinect" in f for f in faults)
    uncounted = tmp_path / "uncounted"
    write_ms_student(uncounted, keypoints, sensors, pbr=OBJ1_PBR, drop_samples=True)
    faults = adaptation.verify_multisensor_student(uncounted, sensors, spec, batch_size=16)
    assert any("recorded sample counts" in f for f in faults)
    masked = tmp_path / "masked"
    write_ms_student(masked, keypoints, sensors, keypoint_samples=(OBJ1_SENSOR, 0, OBJ1_SENSOR))
    assert adaptation.verify_multisensor_student(masked, sensors, spec, batch_size=16)


# --------------------------------------------------------------------------------------
# The stage: review gate, leakage guard, receipts
# --------------------------------------------------------------------------------------
class FakeTrainer:
    """``launch_training`` stand-in: writes the endpoint of the resolved multisensor config."""

    def __init__(self) -> None:
        self.configs: list[dict] = []

    def __call__(self, config: Path) -> None:
        resolved = yaml.safe_load(Path(config).read_text())
        self.configs.append(resolved)
        sizes = tuple(
            len(json.loads((Path(r) / "train" / "annotations.json").read_text())["annotations"])
            for r in resolved["mix_data_root"]
        )
        write_ms_student(
            Path(resolved["work_dir"]).parent,
            Path(resolved["keypoints_3d"]),
            sizes,
            init_from=resolved["init_from"],
            resume=resolved["resume"],
        )


@pytest.fixture
def ms_workspace(tmp_path, monkeypatch):
    """A prepared object-1 multisensor root (4 captures per sensor) and an epoch-25 base."""
    root, runs = tmp_path / "root", tmp_path / "runs"
    paths = fixture.make_dataset(root)
    multisensor.prepare(
        root,
        fixture.OBJ,
        keypoints=paths["keypoints"],
        cad_model=paths["mesh"],
        cad_audit=paths["audit"],
        reserved=multisensor.reserved_population(paths["test"]),
        expected_frames=len(fixture.FRAMES),
    )
    base = layout.object_run(runs, "base", 1)
    (base / "work").mkdir(parents=True)
    (base / layout.CHECKPOINT_PATH).write_bytes(b"epoch-25 base")
    (base / "geometry").mkdir()
    shutil.copyfile(paths["keypoints"], base / layout.KEYPOINTS_PATH)
    done = {
        "stage": "base",
        "obj_id": 1,
        "epochs": 25,
        "checkpoint_sha256": sha256(base / layout.CHECKPOINT_PATH),
        "keypoints_sha256": sha256(base / layout.KEYPOINTS_PATH),
    }
    (base / layout.COMPLETED_NAME).write_text(json.dumps(done))
    (root / "objects" / "obj_000001" / "prepared.json").write_text(json.dumps({"hashes": {}}))
    trainer = FakeTrainer()
    monkeypatch.setattr(adaptation, "launch_training", trainer)
    return root, runs, paths, trainer


def run_ms(root: Path, runs: Path, **kwargs: object) -> adaptation.MethodResult:
    return adaptation.run_multisensor(
        load_recipe("tless"), 1, root, runs, allow_partial=True, **kwargs
    )


def test_train_refuses_stale_review(ms_workspace):
    root, runs, _, trainer = ms_workspace
    out = multisensor.output_dir(root, 1)
    run_dir = layout.object_run(runs, "multisensor", 1)
    with pytest.raises(review.ReviewError, match="accept-unreviewed"):
        run_ms(root, runs)
    multisensor.approve(out, "reviewer")
    ready = out / multisensor.READY_NAME
    ready.write_text(ready.read_text() + "\n")  # same labels, another receipt: stale review
    with pytest.raises(review.ReviewError, match="stale"):
        run_ms(root, runs)
    assert not trainer.configs and not (run_dir / layout.COMPLETED_NAME).exists()
    result = run_ms(root, runs, accept_unreviewed=True)
    assert result.status == "completed" and result.labels == 12
    done = json.loads((run_dir / layout.COMPLETED_NAME).read_text())
    assert done["review"]["accepted_unreviewed"] is True and done["review"]["reviewer"] is None
    assert done["stage"] == "multisensor" and done["completed_epochs"] == 3
    assert done["batches_per_epoch"] == 10 and done["global_step"] == 30
    assert len(done["source_supervision"]) == 4


def test_reviewed_root_trains_with_the_source_recipe(ms_workspace):
    root, runs, _, trainer = ms_workspace
    out = multisensor.output_dir(root, 1)
    multisensor.approve(out, "reviewer")
    reserved = adaptation.reserved_test(root)
    assert (
        reserved.sha256
        == json.loads((out / "READY.json").read_text())["reserved_test_population_sha256"]
    )
    result = run_ms(root, runs, reserved=reserved)
    run_dir = layout.object_run(runs, "multisensor", 1)
    assert result.status == "completed" and result.run_dir == run_dir
    (config,) = trainer.configs
    assert config["mix_data_root"] == [str(out / s) for s in multisensor.SENSORS]
    assert config["mix_per_batch"] == [2, 2, 2] and config["max_iters_per_epoch"] == 0
    assert config["init_from"] == str(layout.object_run(runs, "base", 1) / layout.CHECKPOINT_PATH)
    done = json.loads((run_dir / layout.COMPLETED_NAME).read_text())
    assert done["review"] == {
        "reviewer": "reviewer",
        "accepted_unreviewed": False,
        "reviewed_sha256": sha256(out / review.REVIEWED_NAME),
    }
    assert done["ready_sha256"] == sha256(out / multisensor.READY_NAME)
    identity = json.loads((run_dir / layout.IDENTITY_NAME).read_text())
    assert identity["sensors"] == ["primesense", "kinect", "canon"]
    assert identity["test_gt_used"] is False and not identity["transductive_test_adaptation"]
    assert identity["teacher"] == "../../base/obj_000001/work/last.pth"
    assert run_ms(root, runs).student_sha256 == result.student_sha256
    assert len(trainer.configs) == 1  # idempotent: no second launch
    # The completed student belongs to the fixed multisensor bank.
    bank, fallbacks = banks.fixed_bank(
        banks.recipe_policy(load_recipe("tless"), "multisensor"), runs, [1]
    )
    assert bank == {1: run_dir} and fallbacks == []


def test_train_refuses_a_reserved_test_image(ms_workspace):
    root, runs, paths, trainer = ms_workspace
    multisensor.approve(multisensor.output_dir(root, 1), "reviewer")
    test_image = paths["test"] / "000001" / "rgb" / "000001.png"
    shutil.copyfile(paths["primesense"] / "rgb" / "000002.png", test_image)
    with pytest.raises(ValueError, match="primesense frame 2"):
        run_ms(root, runs)
    assert not trainer.configs


def test_train_refuses_another_test_population_or_keypoints(ms_workspace):
    root, runs, paths, trainer = ms_workspace
    multisensor.approve(multisensor.output_dir(root, 1), "reviewer")
    rgb = paths["test"] / "000001" / "rgb" / "000000.png"
    rgb.write_bytes(rgb.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="reserved test_primesense population changed"):
        run_ms(root, runs)
    shutil.rmtree(paths["test"])
    with pytest.raises(ValueError, match="test_primesense"):
        run_ms(root, runs)
    fixture.write_test(paths["test"])
    keypoints = layout.object_run(runs, "base", 1) / layout.KEYPOINTS_PATH
    keypoints.write_text(keypoints.read_text() + " ")
    with pytest.raises(ValueError, match="keypoints"):
        run_ms(root, runs)
    assert not trainer.configs


def test_train_refuses_a_partial_root_by_default(ms_workspace):
    root, runs, *_ = ms_workspace
    multisensor.approve(multisensor.output_dir(root, 1), "reviewer")
    with pytest.raises(ValueError, match="partial"):
        adaptation.run_multisensor(load_recipe("tless"), 1, root, runs)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def test_train_stage_multisensor_dry_run(tmp_path, capsys):
    root, runs = tmp_path / "root", tmp_path / "runs"
    argv = ["train", "tless", "--stage", "multisensor", "--root", str(root), "--runs", str(runs)]
    assert cli.main([*argv, "--obj-id", "3", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "multisensor/obj_000003" in out and "REVIEWED.json" in out
    assert "[2, 2, 2]" in out and "nothing was read or written" in out
    assert cli.main([*argv, "--obj-id", "3", "--method", "static_pseudo", "--dry-run"]) == 1
    assert cli.main([*argv, "--obj-id", "31", "--dry-run"]) == 1
    handal = ["train", "handal", "--stage", "multisensor", "--root", str(root)]
    assert cli.main([*handal, "--runs", str(runs), "--obj-id", "1", "--dry-run"]) == 1
    assert not root.exists() and not runs.exists()


def test_train_stage_multisensor_passes_accept_unreviewed(tmp_path, monkeypatch):
    calls = []

    def fake(recipe, obj, root, runs, **kwargs):
        calls.append((obj, kwargs))
        return adaptation.MethodResult("multisensor", obj, "completed", 3, Path(runs))

    monkeypatch.setattr(adaptation, "run_multisensor", fake)
    argv = ["train", "tless", "--stage", "multisensor", "--root", str(tmp_path / "r")]
    argv += ["--runs", str(tmp_path / "o"), "--obj-id", "2", "--accept-unreviewed"]
    assert cli.main(argv) == 0
    assert calls == [(2, {"accept_unreviewed": True, "force": False, "overrides": {}})]


# --------------------------------------------------------------------------------------
# Fixed bank
# --------------------------------------------------------------------------------------
def test_multisensor_policy_dry_run_30_models(tmp_path, capsys):
    recipe = load_recipe("tless")
    policy = banks.recipe_policy(recipe, "multisensor")
    assert (policy.method, policy.csv_method, policy.fallback) == (
        "multisensor",
        "dopermultisensor",
        "none",
    )
    runs = tmp_path / "runs"
    for obj in recipe.objects:
        write_base(runs, obj)
        write_student(runs, "multisensor", obj)
    bank, fallbacks = banks.fixed_bank(policy, runs, recipe.objects)
    assert len(bank) == 30 and fallbacks == []
    assert all(run == layout.object_run(runs, "multisensor", o) for o, run in bank.items())
    shutil.rmtree(layout.object_run(runs, "multisensor", 17))
    with pytest.raises(ValueError, match="object 17"):
        banks.fixed_bank(policy, runs, recipe.objects)
    argv = ["evaluate", "tless", "--split", "test", "--policy", "multisensor"]
    argv += ["--root", str(tmp_path / "root"), "--runs", str(runs), "--dry-run"]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "test/multisensor/dopermultisensor_tless-test.csv" in out
    assert "multisensor (fallback: none)" in out and "MODEL_MATRIX.csv" in out
    assert "timed, every model rescanned" in out and "model cache" not in out
    base = ["evaluate", "tless", "--split", "test", "--policy", "base"]
    base += ["--root", str(tmp_path / "root"), "--runs", str(runs), "--dry-run"]
    assert cli.main(base) == 0
    out = capsys.readouterr().out
    assert "untimed" in out and "test/cache" in out
    assert recipe.test.record_time is False and policy.record_time is True


def test_dry_run_paths_are_the_run_paths(tmp_path, capsys):
    root, runs = tmp_path / "root", tmp_path / "runs"
    paths = adaptation.multisensor_paths(root, runs, 3)
    assert paths["teacher"] == layout.object_run(runs, "base", 3) / layout.CHECKPOINT_PATH
    assert paths["sensors"] == [multisensor.output_dir(root, 3) / s for s in multisensor.SENSORS]
    assert paths["work_dir"] == layout.object_run(runs, "multisensor", 3) / "work"
    argv = ["train", "tless", "--stage", "multisensor", "--root", str(root), "--runs", str(runs)]
    assert cli.main([*argv, "--obj-id", "3", "--dry-run"]) == 0
    assert str(paths["teacher"]) in capsys.readouterr().out
