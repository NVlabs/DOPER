# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adaptation engine: adapt configs, teachers, pools, skips, student audits and a smoke run."""

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml
from bop_recipe_data import make_bop_download, write_recipe

from doper import cli
from doper.bop import layout
from doper.infer import Detection
from doper.keypoints import keypoints_3d_fingerprint
from doper.recipes.schema import load_recipe
from doper.training import adaptation

GOLDEN = Path(__file__).parent / "golden" / "source_adapt_templates.json"
SMALL = "convnext_small.dinov3_lvd1689m"
ARCHITECTURE = {
    "backbone": SMALL,
    "num_keypoints": 64,
    "det_input_size": 640,
    "crop_size": 512,
    "keypoint_head": "heatmap",
}
KEYPOINT_COUNT = 64
WIDTH, HEIGHT = 100, 50
FOCAL, DEPTH_MM = 200.0, 500.0
BOX = (30.0, 12.0, 70.0, 38.0)
INTRINSICS = [FOCAL, 0.0, WIDTH / 2, 0.0, FOCAL, HEIGHT / 2, 0.0, 0.0, 1.0]
PATHS = {
    "data_root": "<data_root>",
    "keypoints_3d": "<keypoints_3d>",
    "work_dir": "<work_dir>",
    "labels": "<labels>",
    "teacher": "<init_from>",
}
GOOD, BAD = 200, 10  # gray levels: the scripted teacher accepts GOOD frames only
UNRELATED = frozenset({"0" * 64})


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------
# Fixtures: base runs, students, onboarding views and a scripted teacher
# --------------------------------------------------------------------------------------
def keypoints_m() -> np.ndarray:
    return np.random.default_rng(3).uniform(-0.03, 0.03, (KEYPOINT_COUNT, 3))


def write_keypoints(path: Path, points: np.ndarray | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = keypoints_m() if points is None else points
    payload = {
        "keypoint_names": [f"k{i}" for i in range(len(values))],
        "num_keypoints": len(values),
        "keypoints_3d": values.tolist(),
    }
    path.write_text(json.dumps(payload))
    return path


def fake_checkpoint(path: Path, epoch: int, keypoints: Path, **architecture: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(json.loads(keypoints.read_text())["keypoints_3d"])
    cfg = {
        "model": {**ARCHITECTURE, **architecture},
        "keypoints_3d_fingerprint": keypoints_3d_fingerprint(points),
    }
    torch.save({"model": {"weight": torch.zeros(2)}, "epoch": epoch, "cfg": cfg}, path)
    return path


def make_base(runs: Path, obj: int = 1, *, epochs: int = 25) -> Path:
    run = layout.object_run(runs, "base", obj)
    keypoints = write_keypoints(run / layout.KEYPOINTS_PATH)
    checkpoint = fake_checkpoint(run / layout.CHECKPOINT_PATH, epochs - 1, keypoints)
    receipt = {
        "recipe": "handal",
        "stage": "base",
        "obj_id": obj,
        "epochs": epochs,
        "checkpoint_sha256": sha256(checkpoint),
        "keypoints_sha256": sha256(keypoints),
    }
    (run / "completed.json").write_text(json.dumps(receipt))
    return run


def make_student(
    runs: Path,
    base: Path,
    *,
    obj: int = 1,
    epoch: int = 2,
    completed_epochs: int = 3,
    receipt: bool = True,
    keypoints: Path | None = None,
    **architecture: object,
) -> Path:
    run = layout.object_run(runs, "round1", obj)
    geometry = keypoints or base / layout.KEYPOINTS_PATH
    checkpoint = fake_checkpoint(run / layout.CHECKPOINT_PATH, epoch, geometry, **architecture)
    if receipt:
        done = {
            "obj_id": obj,
            "completed_epochs": completed_epochs,
            "student_sha256": sha256(checkpoint),
            "base_teacher_sha256": sha256(base / layout.CHECKPOINT_PATH),
        }
        (run / "completed.json").write_text(json.dumps(done))
    return run


def write_view(root: Path, kind: str, codes: list[int], sequence: str) -> Path:
    """``<root>/onboarding/<kind>/<sequence>`` with one flat PNG per gray level."""
    folder = root / "onboarding" / kind / sequence / "rgb"
    folder.mkdir(parents=True, exist_ok=True)
    cameras = {}
    for frame, code in enumerate(codes):
        ok, data = cv2.imencode(".png", np.full((HEIGHT, WIDTH, 3), code, np.uint8))
        assert ok
        (folder / f"{frame:06d}.png").write_bytes(data.tobytes())
        cameras[str(frame)] = {"cam_K": INTRINSICS, "width": WIDTH, "height": HEIGHT}
    (folder.parent / "scene_camera.json").write_text(json.dumps(cameras))
    return folder.parent.parent


class ScriptedTeacher:
    """A ``Predictor`` that accepts a frame whose gray level is above 128 and rejects others."""

    def __call__(self, image_bgr, camera_matrix, keypoints_mm, score_threshold):
        accept = int(image_bgr[0, 0, 0]) > 128
        projected = cv2.projectPoints(
            keypoints_mm, np.zeros(3), np.array([0.0, 0.0, DEPTH_MM]), camera_matrix, np.zeros(5)
        )[0].reshape(-1, 2)
        detection = Detection(
            bbox_xyxy=np.array(BOX),
            score=0.9 if accept else 0.2,
            keypoints=projected,
            keypoint_scores=np.full(len(keypoints_mm), 0.9),
            rvec=np.zeros(3),
            tvec=np.array([0.0, 0.0, DEPTH_MM]),
            reproj_error_px=1.0,
        )
        prob = np.zeros(image_bgr.shape[:2], np.float32)
        prob[20:30, 40:60] = 0.9
        return [detection], prob


class Factory:
    """``predictor_factory`` that records the teacher checkpoints it was asked to load."""

    def __init__(self) -> None:
        self.loaded: list[Path] = []

    def __call__(self, checkpoint: Path):
        self.loaded.append(Path(checkpoint))
        return ScriptedTeacher()


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A dataset root, a runs root with an epoch-25 base of object 1, and no validation."""
    root, runs = tmp_path / "root", tmp_path / "runs"
    make_base(runs)
    prepared = root / "objects" / "obj_000001"
    prepared.mkdir(parents=True)
    (prepared / "prepared.json").write_text(json.dumps({"hashes": {}}))
    monkeypatch.setattr(adaptation, "validation_exclusions", lambda recipe, root: UNRELATED)
    return root, runs


def run(recipe, method, root, runs, *, factory=None, **kwargs):
    options = {"consent": False, "labels_only": False, "force": False, **kwargs}
    return adaptation.run_method(
        recipe, method, 1, root, runs, predictor_factory=factory or Factory(), **options
    )


# --------------------------------------------------------------------------------------
# Adapt configs
# --------------------------------------------------------------------------------------
def _source_adaptation(name: str, method: str, n: int) -> dict:
    """``adaptation_recipe`` of the source over its template, plus the GT-arm min-mix keys."""
    golden = json.loads(GOLDEN.read_text())[name]["template"]
    expected = {
        **golden,
        "mix_per_batch": [4],
        "init_from": "<init_from>",
        "resume": None,
        "require_init_from": True,
        "require_no_init_from": False,
        "epochs": 3,
        "max_iters_per_epoch": math.ceil(n / 4),
    }
    if method.startswith("gt_"):
        keypoints = method == "gt_box_mask_keypoints"
        expected["min_mix_mask_only_samples"] = 0 if keypoints else 1
        expected["min_mix_keypoint_samples"] = n if keypoints else 0
    return expected


@pytest.mark.parametrize("name", ["handal", "hopev2", "hot3d"])
@pytest.mark.parametrize("n", [1, 4, 5, 21])
@pytest.mark.parametrize("method", ["static_pseudo", "gt_box_mask", "gt_box_mask_keypoints"])
def test_adapt_config_matches_source(name, n, method):
    recipe = load_recipe(name)
    resolved, launch = adaptation.adapt_config(recipe, method, n, PATHS)
    expected = _source_adaptation(name, method, n)
    if name == "hot3d":
        # User decision: HOT3D adapts the Small base; the source template trained Tiny.
        assert expected["backbone"] == "convnext_tiny.dinov3_lvd1689m"
        expected["backbone"] = SMALL
    assert resolved == expected
    assert launch.init_from == "<init_from>" and launch.require_init_from
    assert not launch.require_no_init_from and launch.resume is None


def test_adapt_config_refuses_zero_labels_and_below_object_minimum():
    recipe = load_recipe("handal")
    with pytest.raises(ValueError, match="never zero"):
        adaptation.adapt_config(recipe, "gt_box_mask", 0, PATHS)
    with pytest.raises(ValueError, match="at least 20"):
        adaptation.adapt_config(recipe, "static_pseudo", 19, PATHS, obj_id=1)
    adaptation.adapt_config(recipe, "combined_pseudo", 1, PATHS, obj_id=8)


def test_adapt_backbone_follows_base():
    recipe = load_recipe("hot3d")
    assert recipe.base.config["backbone"] == SMALL
    resolved, _ = adaptation.adapt_config(recipe, "static_pseudo", 21, PATHS)
    assert resolved["backbone"] == SMALL
    for key, value in (("backbone", "convnext_tiny.dinov3_lvd1689m"), ("det_size", 320)):
        with pytest.raises(ValueError, match="architecture"):
            adaptation.adapt_config(recipe, "static_pseudo", 21, PATHS, overrides={key: value})
    with pytest.raises(ValueError, match="budget"):
        adaptation.adapt_config(recipe, "static_pseudo", 21, PATHS, overrides={"epochs": 5})


def test_minimum_override_per_object():
    methods = load_recipe("handal").adapt.methods
    assert methods["combined_pseudo"].minimum_for(8) == 1
    assert methods["combined_pseudo"].minimum_for(40) == 20
    assert methods["static_pseudo"].minimum_for(8) == 20
    assert methods["gt_box_mask"].minimum_for(3) == 1
    assert methods["dynamic_pseudo"].applies_to(8) and not methods["dynamic_pseudo"].applies_to(3)
    assert methods["static_pseudo"].applies_to(3)


# --------------------------------------------------------------------------------------
# Teachers
# --------------------------------------------------------------------------------------
def test_default_teacher_is_the_completed_base(workspace):
    _, runs = workspace
    base = layout.object_run(runs, "base", 1)
    identity = adaptation.require_teacher(base)
    assert identity.kind == "base" and identity.path == base / layout.CHECKPOINT_PATH
    assert identity.sha256 == identity.base_sha256 == sha256(base / layout.CHECKPOINT_PATH)
    early = make_base(runs, 2, epochs=3)
    with pytest.raises(ValueError, match="25-epoch"):
        adaptation.require_teacher(early)
    (base / layout.CHECKPOINT_PATH).write_bytes(b"changed")
    with pytest.raises(ValueError, match="25-epoch"):
        adaptation.require_teacher(base)


def test_teacher_override_requires_completed_student(workspace, monkeypatch):
    root, runs = workspace
    base = layout.object_run(runs, "base", 1)
    other_object = make_base(runs, 2)
    refusals = {
        "same-object three-epoch student": make_student(runs, other_object, obj=2),
        "epoch-index-2": make_student(runs / "epoch", base, epoch=1),
        "completed_epochs 3": make_student(runs / "short", base, completed_epochs=2),
        "must be a completed": make_student(runs / "bare", base, receipt=False),
        "differs from the base in backbone": make_student(
            runs / "arch", base, backbone="convnext_tiny.dinov3_lvd1689m"
        ),
        "another keypoint layout": make_student(
            runs / "kp",
            base,
            keypoints=write_keypoints(runs / "other_kp.json", keypoints_m()[::-1]),
        ),
    }
    for message, student in refusals.items():
        with pytest.raises(ValueError, match=message):
            adaptation.require_teacher(base, student)
    tampered = make_student(runs / "tampered", base)
    torch.save({"model": {}}, tampered / layout.CHECKPOINT_PATH)
    with pytest.raises(ValueError, match="student"):
        adaptation.require_teacher(base, tampered)
    student = make_student(runs, base)
    checkpoint = student / layout.CHECKPOINT_PATH
    for override in (student, checkpoint):
        identity = adaptation.require_teacher(base, override)
        assert identity.kind == "student" and identity.path == checkpoint
        assert identity.sha256 == sha256(checkpoint)
        assert identity.base_sha256 == sha256(base / layout.CHECKPOINT_PATH)
    assert other_object.is_dir()
    # The accepted override labels the pool and initializes the student.
    write_view(root, "static", [GOOD] * 21, "obj_000001")
    launched = []

    def capture(config: Path) -> None:
        launched.append(yaml.safe_load(config.read_text()))
        raise RuntimeError("stop after the launch")

    monkeypatch.setattr(adaptation, "launch_training", capture)
    factory = Factory()
    with pytest.raises(RuntimeError, match="stop after the launch"):
        run(load_recipe("handal"), "static_pseudo", root, runs, factory=factory, teacher=student)
    assert factory.loaded == [checkpoint]
    assert launched[0]["init_from"] == str(checkpoint) and launched[0]["require_init_from"]
    run_dir = layout.object_run(runs, "static_pseudo", 1)
    identity = json.loads((run_dir / "identity.json").read_text())
    assert identity["teacher_sha256"] == sha256(checkpoint)
    assert identity["base_teacher_sha256"] == sha256(base / layout.CHECKPOINT_PATH)
    manifest = json.loads((run_dir / "labels" / "manifest.json").read_text())
    assert manifest["checkpoint_sha256"] == sha256(checkpoint)
    with pytest.raises(ValueError, match="test inputs adapt from the epoch-25 base"):
        run(load_recipe("handal"), "test_tta", root, runs, teacher=student, consent=True)
    with pytest.raises(ValueError, match="GT methods adapt from the epoch-25 base"):
        run(load_recipe("handal"), "gt_box_mask", root, runs, teacher=student)
    assert not layout.object_run(runs, "gt_box_mask", 1).exists()


# --------------------------------------------------------------------------------------
# Labels, skips and pool audits
# --------------------------------------------------------------------------------------
def test_below_minimum_writes_skip_and_no_student(workspace, monkeypatch):
    root, runs = workspace
    write_view(root, "static", [GOOD] * 3 + [BAD] * 2, "obj_000001_up")
    monkeypatch.setattr(adaptation, "launch_training", pytest.fail)
    result = run(load_recipe("handal"), "static_pseudo", root, runs)
    run_dir = layout.object_run(runs, "static_pseudo", 1)
    assert result.status == "skipped" and result.labels == 3
    skipped = json.loads((run_dir / "adaptation_skipped.json").read_text())
    assert skipped == {"reason": "fewer than 20 accepted labels", "labels": 3, "minimum_labels": 20}
    assert json.loads((run_dir / "labels_ready.json").read_text())["labels"] == 3
    assert not (run_dir / "work").exists() and not (run_dir / "completed.json").exists()
    again = run(load_recipe("handal"), "static_pseudo", root, runs)
    assert again.status == "skipped" and again.labels == 3


def test_a_second_run_waits_for_the_run_lock(workspace, monkeypatch, capsys):
    from lock_helpers import assert_waits_for

    from doper.bop import sources

    root, runs = workspace
    write_view(root, "static", [GOOD] * 3, "obj_000001")
    monkeypatch.setattr(adaptation, "launch_training", pytest.fail)
    run_dir = layout.object_run(runs, "static_pseudo", 1)
    lock = sources.sibling_lock(run_dir)
    assert lock == run_dir.parent / ".obj_000001.lock"

    def waiting() -> None:
        assert not (run_dir / "labels_ready.json").exists()

    result = assert_waits_for(lock, lambda: run(load_recipe("handal"), "static_pseudo", root, runs,
                                                force=True), waiting)  # fmt: skip
    assert result.status == "skipped" and "waiting" in capsys.readouterr().out


def test_labels_only_still_skips_a_pool_below_the_minimum(workspace, monkeypatch):
    root, runs = workspace
    write_view(root, "static", [GOOD] * 3, "obj_000001")
    monkeypatch.setattr(adaptation, "launch_training", pytest.fail)
    result = run(load_recipe("handal"), "static_pseudo", root, runs, labels_only=True)
    run_dir = layout.object_run(runs, "static_pseudo", 1)
    assert result.status == "skipped" and (run_dir / "adaptation_skipped.json").is_file()


def test_labels_only_stops_after_verify(workspace, monkeypatch):
    root, runs = workspace
    write_view(root, "static", [GOOD] * 21, "obj_000001")
    monkeypatch.setattr(adaptation, "launch_training", pytest.fail)
    result = run(load_recipe("handal"), "static_pseudo", root, runs, labels_only=True)
    run_dir = layout.object_run(runs, "static_pseudo", 1)
    assert result.status == "labels_ready" and result.labels == 21
    assert (run_dir / "labels_ready.json").is_file() and not (run_dir / "recipe.yaml").exists()


def test_adapt_refuses_unverified_pool(workspace, monkeypatch):
    root, runs = workspace
    write_view(root, "static", [GOOD] * 4, "obj_000001")
    recipe = load_recipe("handal")
    from doper.pseudo_labels import monocular

    original = monocular.verify_pool
    monkeypatch.setattr(monocular, "verify_pool", lambda *args: ["a planted fault"])
    with pytest.raises(ValueError, match="a planted fault"):
        run(recipe, "static_pseudo", root, runs, labels_only=True)
    run_dir = layout.object_run(runs, "static_pseudo", 1)
    assert not (run_dir / "labels_ready.json").exists()
    monkeypatch.setattr(monocular, "verify_pool", original)
    assert run(recipe, "static_pseudo", root, runs, labels_only=True).labels == 4
    mask = next((run_dir / "labels" / "train" / "masks").glob("*.png"))
    mask.write_bytes(mask.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="label pool changed"):
        run(recipe, "static_pseudo", root, runs, labels_only=True)
    assert run(recipe, "static_pseudo", root, runs, labels_only=True, force=True).labels == 4


def test_test_tta_requires_consent(workspace, monkeypatch):
    root, runs = workspace
    write_view(root, "test", [GOOD] * 2, "000001")
    recipe = load_recipe("handal")
    assert recipe.adapt.methods["test_tta"].transductive
    with pytest.raises(ValueError, match="consent"):
        run(recipe, "test_tta", root, runs, labels_only=True)
    assert not layout.object_run(runs, "test_tta", 1).exists()
    monkeypatch.setattr(adaptation, "launch_training", pytest.fail)
    result = run(recipe, "test_tta", root, runs, consent=True)
    assert result.status == "skipped"
    run_dir = layout.object_run(runs, "test_tta", 1)
    identity = json.loads((run_dir / "identity.json").read_text())
    assert identity["transductive_test_adaptation"] is True
    assert identity["test_gt_used"] is False and identity["test_targets_used"] is False
    manifest = json.loads((run_dir / "labels" / "manifest.json").read_text())
    assert manifest["transductive_test_adaptation"] is True
    with pytest.raises(ValueError, match="not a hopev2 method"):
        run(load_recipe("hopev2"), "test_tta", root, runs, consent=True)


def test_method_objects_are_enforced(workspace):
    root, runs = workspace
    with pytest.raises(ValueError, match="objects"):
        run(load_recipe("handal"), "dynamic_pseudo", root, runs)


# --------------------------------------------------------------------------------------
# Student audit
# --------------------------------------------------------------------------------------
def write_trained_student(
    run_dir: Path,
    n: int,
    *,
    keypoint_samples: int = 0,
    epochs: tuple[int, ...] = (0, 1, 2),
    global_step: int | None = None,
    poison: bool = False,
    skipped_updates: int = 0,
    keypoints: Path | None = None,
    init_from: str | None = None,
    resume: str | None = None,
) -> None:
    keypoints = keypoints or write_keypoints(run_dir / layout.KEYPOINTS_PATH)
    points = np.asarray(json.loads(keypoints.read_text())["keypoints_3d"])
    weight = torch.tensor([1.0, float("nan") if poison else 2.0])
    checkpoint = {
        "model": {"weight": weight},
        "optimizer": {"state": {0: {"exp_avg": torch.ones(2)}}, "param_groups": []},
        "epoch": epochs[-1],
        "global_step": 3 * math.ceil(n / 4) if global_step is None else global_step,
        "cfg": {
            "model": ARCHITECTURE,
            "keypoints_3d_fingerprint": keypoints_3d_fingerprint(points),
        },
        "training_provenance": {
            "init_checkpoint": {
                "path": init_from,
                "sha256": sha256(Path(init_from))
                if init_from and Path(init_from).is_file()
                else None,
            },
            "resume_checkpoint": {"path": resume, "sha256": None},
            "train_sources": [
                {"samples": 100, "keypoint_samples": 100, "mask_only_samples": 0},
                {
                    "samples": n,
                    "keypoint_samples": keypoint_samples,
                    "mask_only_samples": n - keypoint_samples,
                },
            ],
        },
    }
    (run_dir / "work").mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, run_dir / layout.CHECKPOINT_PATH)
    rows = [
        {"epoch": e, "nonfinite_microbatches": 0, "scaler_skipped_updates": skipped_updates}
        for e in epochs
    ]
    (run_dir / "work" / "epoch_timing.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


def test_verify_student_budget(tmp_path):
    methods = load_recipe("handal").adapt.methods
    pseudo, keypoint_arm = methods["static_pseudo"], methods["gt_box_mask_keypoints"]
    clean = tmp_path / "clean"
    write_trained_student(clean, 21)
    assert adaptation.verify_student(clean, 21, pseudo) == []
    keyed = tmp_path / "keyed"
    write_trained_student(keyed, 5, keypoint_samples=5)
    assert adaptation.verify_student(keyed, 5, keypoint_arm) == []
    assert adaptation.verify_student(keyed, 5, pseudo)
    cases = {
        "budget": dict(global_step=17),
        "epoch": dict(epochs=(0, 1)),
        "skip": dict(epochs=(0, 2)),
        "finite": dict(poison=True),
        "updates": dict(skipped_updates=1),
        "supervision": dict(keypoint_samples=2),
    }
    for name, change in cases.items():
        folder = tmp_path / name
        write_trained_student(folder, 21, **change)
        assert adaptation.verify_student(folder, 21, pseudo), name
    assert adaptation.verify_student(clean, 22, pseudo)


# --------------------------------------------------------------------------------------
# End-to-end engine runs with a stand-in trainer
# --------------------------------------------------------------------------------------
class FakeTrainer:
    """``launch_training`` stand-in: writes the endpoint ``doper.train`` would, from the config."""

    def __init__(self) -> None:
        self.configs: list[dict] = []

    def __call__(self, config: Path) -> None:
        resolved = yaml.safe_load(Path(config).read_text())
        self.configs.append(resolved)
        pool = Path(resolved["mix_data_root"][0])
        n = len(json.loads((pool / "train" / "annotations.json").read_text())["images"])
        keypoint_samples = resolved["min_mix_keypoint_samples"]
        work = Path(resolved["work_dir"])
        epochs = resolved["epochs"]
        write_trained_student(
            work.parent,
            n,
            keypoint_samples=keypoint_samples,
            epochs=tuple(range(epochs)),
            global_step=epochs * resolved["max_iters_per_epoch"],
            keypoints=Path(resolved["keypoints_3d"]),
            init_from=resolved["init_from"],
            resume=resolved["resume"],
        )


def prepare_object(root: Path, runs: Path, obj: int) -> None:
    if not layout.object_run(runs, "base", obj).exists():
        make_base(runs, obj)
    prepared = root / "objects" / f"obj_{obj:06d}"
    prepared.mkdir(parents=True, exist_ok=True)
    (prepared / "prepared.json").write_text(json.dumps({"hashes": {}}))


def test_combined_obj8_trains_and_leaves_components_skipped(workspace, monkeypatch):
    root, runs = workspace
    prepare_object(root, runs, 8)
    write_view(root, "static", [GOOD], "obj_000008_up")
    write_view(root, "dynamic", [GOOD + 1], "obj_000008")
    trainer = FakeTrainer()
    monkeypatch.setattr(adaptation, "launch_training", trainer)
    recipe = load_recipe("handal")
    options = {"consent": False, "labels_only": False, "force": False}
    result = adaptation.run_method(
        recipe, "combined_pseudo", 8, root, runs, predictor_factory=Factory(), **options
    )
    assert result.status == "completed" and result.labels == 2
    teacher = sha256(layout.object_run(runs, "base", 8) / layout.CHECKPOINT_PATH)
    for component in ("static_pseudo", "dynamic_pseudo"):
        folder = layout.object_run(runs, component, 8)
        assert json.loads((folder / "adaptation_skipped.json").read_text())["labels"] == 1
        assert json.loads((folder / "identity.json").read_text())["teacher_sha256"] == teacher
        assert not (folder / "work").exists()
    run_dir = layout.object_run(runs, "combined_pseudo", 8)
    manifest = json.loads((run_dir / "labels" / "manifest.json").read_text())
    assert set(manifest["components"]) == {"static", "dynamic"} and manifest["accepted"] == 2
    done = json.loads((run_dir / "completed.json").read_text())
    assert done["labels"] == 2 and done["teacher_sha256"] == done["init_from_sha256"] == teacher
    assert trainer.configs[0]["max_iters_per_epoch"] == 1
    from doper.cli.evaluate import select_bank

    bank, choices = select_bank(recipe, root, runs, "selected", True)
    assert choices[8].chosen.arm == "base" and 8 in bank
    again = adaptation.run_method(
        recipe, "combined_pseudo", 8, root, runs, predictor_factory=Factory(), **options
    )
    assert again.student_sha256 == result.student_sha256 and len(trainer.configs) == 1


def test_gt_methods_share_one_pool(workspace, monkeypatch):
    from test_bop_gt_labels import DEFAULT_FRAMES, build_gt_view

    root, runs = workspace
    build_gt_view(root / "onboarding", DEFAULT_FRAMES)
    trainer = FakeTrainer()
    monkeypatch.setattr(adaptation, "launch_training", trainer)
    recipe = load_recipe("handal")
    pool = layout.gt_pool(runs, 1)
    results = {}
    for method in ("gt_box_mask", "gt_box_mask_keypoints"):
        results[method] = run(recipe, method, root, runs)
        assert results[method].status == "completed" and results[method].labels == 3
        run_dir = layout.object_run(runs, method, 1)
        ready = json.loads((run_dir / "labels_ready.json").read_text())
        arm = method.removeprefix("gt_")
        assert ready["labels_root"] == f"../../gt_pools/obj_000001/{arm}"
        assert (run_dir / ready["labels_root"]).resolve() == (pool / arm).resolve()
        assert not (run_dir / "labels").exists()
    receipt = sha256(pool / "POOL_READY.json")
    keyed, plain = trainer.configs[1], trainer.configs[0]
    assert (plain["min_mix_keypoint_samples"], plain["min_mix_mask_only_samples"]) == (0, 1)
    assert (keyed["min_mix_keypoint_samples"], keyed["min_mix_mask_only_samples"]) == (3, 0)
    done = json.loads(
        (layout.object_run(runs, "gt_box_mask_keypoints", 1) / "completed.json").read_text()
    )
    assert done["source_supervision"][1]["keypoint_samples"] == 3
    for method, first in results.items():
        again = run(recipe, method, root, runs)
        assert again.status == "completed" and again.student_sha256 == first.student_sha256
    assert len(trainer.configs) == 2 and sha256(pool / "POOL_READY.json") == receipt


def test_resume_requires_the_teacher_lineage(workspace, monkeypatch):
    root, runs = workspace
    write_view(root, "static", [GOOD] * 21, "obj_000001")
    trainer = FakeTrainer()
    monkeypatch.setattr(adaptation, "launch_training", trainer)
    recipe = load_recipe("handal")
    run(recipe, "static_pseudo", root, runs, labels_only=True)
    run_dir = layout.object_run(runs, "static_pseudo", 1)
    keypoints = write_keypoints(run_dir / layout.KEYPOINTS_PATH)
    stranger = run_dir / "stranger.pth"
    stranger.write_bytes(b"another teacher")
    write_trained_student(run_dir, 21, epochs=(0,), keypoints=keypoints, init_from=str(stranger))
    with pytest.raises(ValueError, match="not initialized from the static_pseudo teacher"):
        run(recipe, "static_pseudo", root, runs)
    assert not trainer.configs
    teacher = layout.object_run(runs, "base", 1) / layout.CHECKPOINT_PATH
    write_trained_student(run_dir, 21, epochs=(0,), keypoints=keypoints, init_from=str(teacher))
    result = run(recipe, "static_pseudo", root, runs)
    assert result.status == "completed"
    assert trainer.configs[0]["resume"] == str(run_dir / layout.CHECKPOINT_PATH)
    assert trainer.configs[0]["init_from"] is None
    assert yaml.safe_load((run_dir / "recipe_resume.yaml").read_text())["resume"]


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
METHOD_OBJECT = {"dynamic_pseudo": 8, "combined_pseudo": 8}


@pytest.mark.parametrize("name", ["handal", "hopev2", "hot3d"])
def test_every_method_dry_runs(name, tmp_path, capsys):
    recipe = load_recipe(name)
    root, runs = tmp_path / "root", tmp_path / "runs"
    assert recipe.adapt is not None
    for method, spec in recipe.adapt.methods.items():
        obj = METHOD_OBJECT.get(method, 2) if name == "handal" else 2
        argv = ["train", name, "--stage", "adapt", "--method", method, "--root", str(root)]
        argv += ["--runs", str(runs), "--obj-id", str(obj), "--dry-run"]
        if spec.transductive:
            assert cli.main(argv) == 1
            argv.append("--allow-test-inputs")
        assert cli.main(argv) == 0, method
        out = capsys.readouterr().out
        assert f"{method}/obj_{obj:06d}" in out and "nothing was read or written" in out
    assert not root.exists() and not runs.exists()


# --------------------------------------------------------------------------------------
# CPU smoke run
# --------------------------------------------------------------------------------------
SMOKE_BASE = {
    "backbone": "resnet18",
    "allow_nonstandard_backbone": True,
    "no_pretrained": True,
    "num_classes": 1,
    "num_keypoints": 64,
    "det_size": 128,
    "crop_size": 128,
    "keypoint_head": "heatmap",
    "keypoint_heatmap_sigma": 0.75,
    "epochs": 1,
    "batch_size": 2,
    "max_iters_per_epoch": 2,
    "gradient_accumulation_steps": 1,
    "warmup_iters": 1,
    "num_workers": 0,
    "device": "cpu",
    "no_amp": True,
    "save_interval": 1000,
}
SMOKE_LAUNCH = {
    "init_from": None,
    "resume": None,
    "require_no_init_from": True,
    "require_pretrained_backbone": False,
}


@pytest.mark.slow
def test_cpu_smoke_adapt_two_iterations(tmp_path):
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
    data["adapt"]["methods"]["static_pseudo"]["minimum_labels"] = 2
    recipe_path.write_text(yaml.safe_dump(data, sort_keys=False))
    assert cli.main(["prepare", str(recipe_path), "--root", str(root), "--obj-id", "1"]) == 0
    train = ["train", str(recipe_path), "--root", str(root), "--runs", str(runs), "--obj-id", "1"]
    assert cli.main([*train, "--stage", "base"]) == 0
    base = layout.object_run(runs, "base", 1)
    completed = json.loads((base / "completed.json").read_text())
    completed["epochs"] = 25  # stand in for the epoch-25 base the engine requires
    (base / "completed.json").write_text(json.dumps(completed))
    write_view(root, "static", [GOOD] * 5 + [BAD], "obj_000001")
    recipe = load_recipe(recipe_path)
    result = adaptation.run_method(
        recipe,
        "static_pseudo",
        1,
        root,
        runs,
        predictor_factory=Factory(),
        consent=False,
        labels_only=False,
        force=False,
        overrides={"device": "cpu"},
    )
    assert result.status == "completed" and result.labels == 5
    run_dir = layout.object_run(runs, "static_pseudo", 1)
    done = json.loads((run_dir / "completed.json").read_text())
    teacher = sha256(base / layout.CHECKPOINT_PATH)
    assert done["completed_epochs"] == 3 and done["global_step"] == 3 * 2
    assert done["teacher_sha256"] == done["base_teacher_sha256"] == teacher
    assert done["init_from_sha256"] == teacher
    assert done["student_sha256"] == sha256(run_dir / layout.CHECKPOINT_PATH)
    real = done["source_supervision"][1]
    assert (real["samples"], real["keypoint_samples"], real["mask_only_samples"]) == (5, 0, 5)
    checkpoint = torch.load(run_dir / layout.CHECKPOINT_PATH, weights_only=False)
    provenance = checkpoint["training_provenance"]
    assert provenance["init_checkpoint"]["sha256"] == teacher
    assert provenance["train_sources"][1]["mask_only_samples"] == 5
    config = yaml.safe_load((run_dir / "recipe.yaml").read_text())
    assert config["backbone"] == "resnet18" and config["max_iters_per_epoch"] == 2
    again = adaptation.run_method(
        recipe,
        "static_pseudo",
        1,
        root,
        runs,
        predictor_factory=Factory(),
        consent=False,
        labels_only=False,
        force=False,
        overrides={"device": "cpu"},
    )
    assert again.status == "completed" and again.student_sha256 == done["student_sha256"]
