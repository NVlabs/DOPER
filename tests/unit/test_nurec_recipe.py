# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The NuRec recipe template (``kind: nurec``): schema, training stages and CLI routes."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from nurec_fixtures import NUREC_TEMPLATE

from doper import cli
from doper.recipes import load_any_recipe, recipe_kind
from doper.recipes.nurec_schema import NurecRecipe, expected_updates, stage_arguments
from doper.training.recipe import build_config

BASE_PATHS = {
    "data_root": "/data/synthetic/dataset",
    "keypoints_3d": "/data/synthetic/dataset/keypoints_3d.json",
    "work_dir": "/runs/base/work",
}
ADAPT_PATHS = {
    "data_root": "/data/adaptation/dataset",
    "keypoints_3d": "/data/adaptation/dataset/keypoints_3d.json",
    "work_dir": "/runs/adapt/work",
    "init_from": "/runs/base/work/best.pth",
}


def _recipe_data() -> dict:
    return yaml.safe_load(load_any_recipe(NUREC_TEMPLATE).source.read_text())


def _write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "nurec.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_template_recipe_loads_with_nurec_kind():
    recipe = load_any_recipe(NUREC_TEMPLATE)
    assert isinstance(recipe, NurecRecipe)
    assert recipe_kind(recipe) == "nurec" and recipe.kind == "nurec"
    assert recipe.name == "my-object" and recipe.profile == "my-object"
    assert recipe.object.category == "object"
    assert recipe.seed_mode == "robot_handeye"
    assert recipe.labels.mask_mode == "sam2"
    assert (recipe.split.seed, recipe.split.train, recipe.split.val) == (817, 1024, 128)
    assert recipe.split.single_reconstruction_split is True
    assert recipe.evaluation.score_thr == 0.3 and recipe.evaluation.exclude == ()
    assert (recipe.render.seed, recipe.render.count) == (891, 1600)
    assert "@sha256:" in recipe.render.image_digest


def test_base_stage_arguments():
    recipe = load_any_recipe(NUREC_TEMPLATE)
    args = stage_arguments(recipe, "base", BASE_PATHS)
    expected = {
        "backbone": "convnext_small.dinov3_lvd1689m",
        "det_size": 640,
        "crop_size": 256,
        "keypoint_head": "heatmap",
        "keypoint_heatmap_sigma": 0.75,
        "kp_dark": True,
        "epochs": 40,
        "batch_size": 16,
        "gradient_accumulation_steps": 4,
        "lr": 5.0e-4,
        "backbone_lr_multiplier": 0.1,
        "warmup_iters": 500,
        "lr_schedule": "cosine",
        "lr_cooldown_fraction": 0.2,
        "grad_clip": 5.0,
        "seed": 42,
        "no_amp": True,
        "background": False,
        "mix_data_root": [],
        "keypoint_only_mix_data_root": [],
        "save_interval": 10,
        "val_interval": 1,
        "require_no_init_from": True,
        "require_pretrained_backbone": True,
        "require_init_from": False,
        "init_from": None,
        "resume": None,
        "no_pretrained": False,
        **BASE_PATHS,
    }
    assert {key: args[key] for key in expected} == expected
    cfg = build_config(SimpleNamespace(**args))
    # Weight decay is not a training argument; the release default is the recipe's 0.05.
    assert cfg.train.weight_decay == 0.05
    assert cfg.model.pretrained and not cfg.train.amp and not cfg.background.enabled
    assert cfg.background.composite_source_root == BASE_PATHS["data_root"]


def test_adapt_stage_arguments():
    recipe = load_any_recipe(NUREC_TEMPLATE)
    base = stage_arguments(recipe, "base", BASE_PATHS)
    adapt = stage_arguments(recipe, "adapt", ADAPT_PATHS)
    changed = {key: adapt[key] for key in adapt if adapt[key] != base[key]}
    assert changed == {
        **ADAPT_PATHS,
        "epochs": 5,
        "lr": 1.0e-5,
        "warmup_iters": 2,
        "save_interval": 1,
        "no_pretrained": True,
        "require_init_from": True,
        "require_no_init_from": False,
        "require_pretrained_backbone": False,
    }
    assert adapt["val_interval"] == 1
    assert not build_config(SimpleNamespace(**adapt)).model.pretrained


def test_stage_arguments_refuse_bad_stage_and_paths():
    recipe = load_any_recipe(NUREC_TEMPLATE)
    with pytest.raises(ValueError, match="stage"):
        stage_arguments(recipe, "followup", BASE_PATHS)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="init_from"):
        stage_arguments(recipe, "adapt", BASE_PATHS)
    with pytest.raises(ValueError, match="init_from"):
        stage_arguments(recipe, "base", ADAPT_PATHS)


def test_expected_updates():
    assert expected_updates(9750, 16, 4, 40) == 6120
    assert expected_updates(1024, 16, 4, 5) == 80
    # The trainer drops the last partial batch and steps on a partial accumulation.
    assert expected_updates(33, 16, 4, 1) == 1
    with pytest.raises(ValueError, match="positive"):
        expected_updates(10, 0, 4, 1)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r.update(stages={}), "unknown keys"),
        (lambda r: r["render"].update(fps=30), "unknown keys"),
        (lambda r: r["labels"].update(exclude_hsv={"h_above": 85, "hue": 1}), "unknown keys"),
        (lambda r: r["base"].update(not_a_flag=1), "doper.train does not accept"),
        (lambda r: r["base"].update(data_root="/x"), "must not set"),
        (lambda r: r["adapt"].update(init_from="/x"), "must not set"),
        (lambda r: r.pop("labels"), "missing keys"),
        (lambda r: r.update(kind="nurec2"), "kind"),
        (lambda r: r.update(seed_mode="gps"), "seed_mode"),
        (lambda r: r["labels"].update(mask_mode="gaussian"), "mask_mode"),
        (lambda r: r["object"].update(keypoints_points_sha256="ABC"), "SHA-256"),
        (lambda r: r["render"].update(image_digest="nre:latest"), "@sha256"),
        (lambda r: r["colmap"].update(mapper=["--x"]), "colmap"),
        (
            lambda r: r["colmap"].update(mapper={"Mapper.ba_refine_focal_length": 1}),
            "fixes",
        ),
        (
            lambda r: r["colmap"].update(
                image_registrator={"Mapper.ba_refine_principal_point": True}
            ),
            "fixes",
        ),
    ],
)
def test_recipe_rejects_unknown_keys(tmp_path, change, message):
    data = _recipe_data()
    change(data)
    with pytest.raises(ValueError, match=message):
        load_any_recipe(_write(tmp_path, data))


def test_single_reconstruction_split_required(tmp_path):
    data = _recipe_data()
    data["split"]["single_reconstruction_split"] = False
    with pytest.raises(ValueError, match="single_reconstruction_split"):
        load_any_recipe(_write(tmp_path, data))
    del data["split"]["single_reconstruction_split"]
    with pytest.raises(ValueError, match="single_reconstruction_split"):
        load_any_recipe(_write(tmp_path, data))


def _run_cli(argv: list[str]) -> subprocess.CompletedProcess:
    code = (
        "import sys; from doper.cli import main; "
        f"rc = main({argv!r}); print('RC', rc, 'torch' in sys.modules)"
    )
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


@pytest.mark.parametrize("stage", ["base", "adapt"])
def test_train_dry_runs(tmp_path, stage):
    argv = ["train", str(NUREC_TEMPLATE), "--stage", stage, "--root", str(tmp_path / "data")]
    result = _run_cli([*argv, "--runs", str(tmp_path / "runs"), "--dry-run"])
    assert result.returncode == 0, result.stderr
    out = result.stdout + result.stderr
    assert "RC 0 False" in out
    assert "expected updates" in out and "nothing was read or written" in out
    if stage == "adapt":
        assert str(tmp_path / "runs" / "base" / "work" / "best.pth") in out
    assert not (tmp_path / "data").exists() and not (tmp_path / "runs").exists()


def test_train_refuses_bop_flags_for_nurec(tmp_path, capsys):
    argv = ["train", str(NUREC_TEMPLATE), "--stage", "base", "--root", str(tmp_path)]
    argv += ["--runs", str(tmp_path), "--dry-run", "--obj-id", "1"]
    assert cli.main(argv) == 1
    assert "--obj-id" in capsys.readouterr().out
    argv = ["train", str(NUREC_TEMPLATE), "--stage", "followup", "--root", str(tmp_path)]
    assert cli.main([*argv, "--runs", str(tmp_path), "--dry-run"]) == 1
    assert "base or adapt" in capsys.readouterr().out


def test_bop_train_still_needs_obj_id(tmp_path, capsys):
    argv = ["train", "tless", "--stage", "base", "--root", str(tmp_path)]
    assert cli.main([*argv, "--runs", str(tmp_path), "--dry-run"]) == 1
    assert "--obj-id" in capsys.readouterr().out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepared(root: Path, name: str, images: int, points: list) -> None:
    """A prepared dataset with the receipt its prepare step writes last (and, adapting, QC)."""
    dataset = root / name / "dataset"
    (dataset / "train").mkdir(parents=True)
    coco = {"images": [{"id": i} for i in range(images)], "annotations": []}
    (dataset / "train" / "annotations.json").write_text(json.dumps(coco))
    (dataset / "keypoints_3d.json").write_text(json.dumps({"keypoints_3d": points}))
    (dataset / "package.json").write_text(json.dumps({"train": images}))
    receipt = {"step": "split", "package_sha256": _sha256(dataset / "package.json")}
    if name == "adaptation":
        qc = root / name / "dataset_qc.json"
        qc.write_text(json.dumps({"status": "passed"}))
        receipt.update(step="package", qc="passed", qc_sha256=_sha256(qc))
    (root / name / "receipt.json").write_text(json.dumps(receipt))


def test_train_stages_check_the_split_and_record_the_run(tmp_path, monkeypatch, capsys):
    points = [[0.01 * i, 0.0, 0.002 * i] for i in range(64)]
    data = _recipe_data()
    digest = hashlib.sha256(np.asarray(points, dtype="<f8").tobytes()).hexdigest()
    data["object"]["keypoints_points_sha256"] = digest
    recipe = str(_write(tmp_path, data))
    root, runs = tmp_path / "data", tmp_path / "runs"
    configs = []

    def fake_main(argv):
        config = yaml.safe_load(Path(argv[1]).read_text())
        configs.append(config)
        work = Path(config["work_dir"])
        work.mkdir(parents=True)
        for name in ("best.pth", "last.pth"):
            (work / name).write_bytes(name.encode() + config["data_root"].encode())

    monkeypatch.setattr("doper.train.main", fake_main)
    argv = ["train", recipe, "--root", str(root), "--runs", str(runs)]
    _prepared(root, "adaptation", 1024, points)
    assert cli.main([*argv, "--stage", "adapt"]) == 1
    assert "train --stage base first" in capsys.readouterr().out
    _prepared(root, "synthetic", 9749, points)
    assert cli.main([*argv, "--stage", "base"]) == 1
    assert "holds 9749 training images; the base stage plans 9750" in capsys.readouterr().out
    (root / "synthetic").rename(tmp_path / "short")
    _prepared(root, "synthetic", 9750, points[::-1])
    assert cli.main([*argv, "--stage", "base"]) == 1
    assert "not the recipe's keypoint layout" in capsys.readouterr().out
    (root / "synthetic").rename(tmp_path / "reversed")
    _prepared(root, "synthetic", 9750, points)
    assert cli.main([*argv, "--stage", "base"]) == 0
    base = json.loads((runs / "base" / "completed.json").read_text())
    assert (base["stage"], base["train_images"], base["expected_updates"]) == ("base", 9750, 6120)
    assert base["dataset_receipt_sha256"] == _sha256(root / "synthetic" / "receipt.json")
    assert configs[-1]["epochs"] == 40 and configs[-1]["init_from"] is None
    with pytest.raises(SystemExit, match="--force"):
        cli.main([*argv, "--stage", "base"])
    assert cli.main([*argv, "--stage", "adapt"]) == 0
    adapt = json.loads((runs / "adapt" / "completed.json").read_text())
    assert (adapt["train_images"], adapt["expected_updates"]) == (1024, 80)
    assert adapt["init_from_sha256"] == base["best_sha256"]
    assert adapt["dataset_receipt_sha256"] == _sha256(root / "adaptation" / "receipt.json")
    assert configs[-1]["init_from"] == str(runs / "base" / "work" / "best.pth")
    assert configs[-1]["require_init_from"] and configs[-1]["no_pretrained"]
    (runs / "base" / "work" / "best.pth").write_bytes(b"changed")
    assert cli.main([*argv, "--stage", "adapt", "--force"]) == 1
    assert "changed since the base completed" in capsys.readouterr().out


def test_every_seed_mode_has_a_strategy():
    from doper.recipes.nurec_schema import SEED_MODES
    from doper.reconstruction.seeding import STRATEGIES

    assert set(SEED_MODES) == set(STRATEGIES)


def test_load_layout_checks_the_points_hash(tmp_path):
    from doper.keypoints import KEYPOINTS_NAME, load_layout, points_sha256

    points = np.arange(12, dtype=np.float64).reshape(4, 3)
    path = tmp_path / KEYPOINTS_NAME
    path.write_text(json.dumps({"keypoints_3d": points.tolist()}))
    np.testing.assert_array_equal(load_layout(path, points_sha256(points)), points)
    with pytest.raises(ValueError, match="not the recipe's keypoint layout"):
        load_layout(path, points_sha256(points[::-1]))


def _train_world(tmp_path, monkeypatch):
    """A recipe whose layout is ``points``, prepared datasets and a completed base run."""
    points = [[0.01 * i, 0.0, 0.002 * i] for i in range(64)]
    data = _recipe_data()
    data["object"]["keypoints_points_sha256"] = hashlib.sha256(
        np.asarray(points, dtype="<f8").tobytes()
    ).hexdigest()
    recipe = str(_write(tmp_path, data))
    root, runs = tmp_path / "data", tmp_path / "runs"

    def fake_main(argv):
        work = Path(yaml.safe_load(Path(argv[1]).read_text())["work_dir"])
        work.mkdir(parents=True)
        for name in ("best.pth", "last.pth"):
            (work / name).write_bytes(name.encode())

    monkeypatch.setattr("doper.train.main", fake_main)
    argv = ["train", recipe, "--root", str(root), "--runs", str(runs)]
    _prepared(root, "synthetic", 9750, points)
    _prepared(root, "adaptation", 1024, points)
    assert cli.main([*argv, "--stage", "base"]) == 0
    return root, [*argv, "--stage", "adapt"]


def test_train_refuses_a_dataset_whose_prepare_did_not_finish(tmp_path, monkeypatch, capsys):
    root, adapt = _train_world(tmp_path, monkeypatch)
    stage = root / "adaptation"
    receipt = json.loads((stage / "receipt.json").read_text())
    # QC failed: prepare writes no receipt, so the receipt is missing.
    (stage / "receipt.json").unlink()
    assert cli.main(adapt) == 1
    assert "receipt.json is missing" in capsys.readouterr().out
    # A receipt that records a failed QC.
    (stage / "receipt.json").write_text(json.dumps({**receipt, "qc": "failed"}))
    assert cli.main(adapt) == 1
    assert "did not pass its QC" in capsys.readouterr().out
    # A stale passing QC summary beside a dataset its receipt does not record.
    (stage / "receipt.json").write_text(json.dumps(receipt))
    (stage / "dataset" / "package.json").write_text(json.dumps({"train": 1024, "other": 1}))
    assert cli.main(adapt) == 1
    assert "is not the dataset its receipt" in capsys.readouterr().out
    (stage / "dataset" / "package.json").write_text(json.dumps({"train": 1024}))
    (stage / "dataset_qc.json").write_text(json.dumps({"status": "passed", "stale": True}))
    assert cli.main(adapt) == 1
    assert "is not the passed QC" in capsys.readouterr().out
    (stage / "dataset_qc.json").write_text(json.dumps({"status": "passed"}))
    assert cli.main(adapt) == 0


def test_train_base_needs_the_split_receipt(tmp_path, monkeypatch, capsys):
    root, adapt = _train_world(tmp_path, monkeypatch)
    argv = [*adapt[:-1], "base", "--force"]
    receipt = root / "synthetic" / "receipt.json"
    receipt.write_text(json.dumps({"step": "audit"}))
    assert cli.main(argv) == 1
    assert "not split" in capsys.readouterr().out
    receipt.unlink()
    assert cli.main(argv) == 1
    assert "the split step of `doper prepare" in capsys.readouterr().out


def test_recipe_accepts_mapper_flags_equal_to_the_fixed_intrinsics(tmp_path):
    data = _recipe_data()
    data["colmap"]["mapper"] = {"Mapper.ba_refine_focal_length": 0, "Mapper.min_num_matches": 15}
    recipe = load_any_recipe(str(_write(tmp_path, data)))
    assert recipe.colmap["mapper"]["Mapper.min_num_matches"] == 15
