# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local BOP24 AP (``doper.bop.ap``) against hand-computed cases and the pinned toolkit."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from doper import cli
from doper.bop import ap
from doper.bop.sources import sha256_file

K = [1000.0, 0.0, 360.0, 0.0, 1000.0, 270.0, 0.0, 0.0, 1.0]
TOOLKIT_SHORT = "cea62d6"
# Closed form for two visible GTs scored TP, FP (duplicate), TP: precision 1 up to recall 0.5
# (51 of the 101 COCO recall points), then 2/3.
DUPLICATE_AP = (51 * 1.0 + 50 * (2.0 / 3.0)) / 101


def _rotation(axis: str, degrees: float) -> np.ndarray:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _write_ply(path: Path, points: np.ndarray) -> None:
    header = (
        f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
    )
    rows = "".join(f"{x!r} {y!r} {z!r}\n" for x, y, z in points.tolist())
    path.write_text(header + rows)


def _diameter(points: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(points[:, None] - points[None], axis=2)))


def _symmetries(obj: int) -> dict:
    half_turn_z = np.eye(4)
    half_turn_z[:3, :3] = _rotation("z", 180.0)
    half_turn_x = np.eye(4)
    half_turn_x[:3, :3] = _rotation("x", 180.0)
    if obj == 1:
        return {"symmetries_continuous": [{"axis": [0, 0, 1], "offset": [0, 0, 0]}]}
    if obj == 2:
        return {"symmetries_discrete": [half_turn_z.ravel().tolist()]}
    if obj == 3:
        return {
            "symmetries_discrete": [half_turn_x.ravel().tolist()],
            "symmetries_continuous": [{"axis": [0, 0, 1], "offset": [0, 0, 5.0]}],
        }
    return {}


def write_models(models: Path, objects: int = 30, seed: int = 3) -> dict[int, np.ndarray]:
    """Small random meshes (vertices only) and models_info for objects 1..objects."""
    rng = np.random.default_rng(seed)
    models.mkdir(parents=True, exist_ok=True)
    info, points = {}, {}
    for obj in range(1, objects + 1):
        extent = np.array([20.0 + obj, 15.0 + obj / 2, 10.0 + obj / 3])
        pts = np.round(rng.uniform(-1.0, 1.0, (24, 3)) * extent, 3)
        _write_ply(models / f"obj_{obj:06d}.ply", pts)
        info[str(obj)] = {"diameter": _diameter(pts), **_symmetries(obj)}
        points[obj] = pts
    (models / "models_info.json").write_text(json.dumps(info))
    return points


def _pose(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    rotation = _rotation("x", rng.uniform(0, 360)) @ _rotation("y", rng.uniform(0, 360))
    rotation = rotation @ _rotation("z", rng.uniform(0, 360))
    translation = np.array([rng.uniform(-100, 100), rng.uniform(-80, 80), rng.uniform(600, 900)])
    return rotation, translation


def _noisy(rng: np.random.Generator, rotation: np.ndarray, translation: np.ndarray, scale: float):
    noise = _rotation("x", rng.normal(0, 4 * scale)) @ _rotation("y", rng.normal(0, 4 * scale))
    return rotation @ noise, translation + rng.normal(0, 6 * scale, 3)


def _row(scene: int, im: int, obj: int, score: float, rotation, translation, time=-1.0) -> str:
    r = " ".join(repr(float(v)) for v in np.asarray(rotation).ravel())
    t = " ".join(repr(float(v)) for v in np.asarray(translation).ravel())
    return f"{scene},{im},{obj},{score!r},{r},{t},{time!r}\n"


def write_random_dataset(base: Path, seed: int = 7) -> dict[str, Path]:
    """A T-LESS-shaped fixture with symmetric objects, ties, duplicates and low visibility.

    Returns:
        Paths ``gt_root`` (``test_primesense``), ``targets``, ``models`` and ``csv``.
    """
    rng = np.random.default_rng(seed)
    write_models(base / "models_eval")
    split = base / "test_primesense"
    targets, rows = [], ["scene_id,im_id,obj_id,score,R,t,time\n"]
    for scene in (2, 1):
        gt, info, camera = {}, {}, {}
        for im in (5, 0, 3):
            objects = [int(o) for o in rng.choice([1, 2, 3, 4, 5, 6], size=4)]
            gt[str(im)], info[str(im)] = [], []
            camera[str(im)] = {"cam_K": K, "depth_scale": 0.1}
            for gt_id, obj in enumerate(objects):
                rotation, translation = _pose(rng)
                gt[str(im)].append(
                    {
                        "obj_id": obj,
                        "cam_R_m2c": rotation.ravel().tolist(),
                        "cam_t_m2c": translation.tolist(),
                    }
                )
                visib = 0.05 if gt_id == 3 else float(np.round(rng.uniform(0.1, 1.0), 2))
                info[str(im)].append({"visib_fract": visib, "px_count_visib": 100})
                for _ in range(int(rng.integers(0, 3))):
                    score = float(np.round(rng.uniform(0.1, 1.0), 1))  # coarse: score ties
                    est_r, est_t = _noisy(rng, rotation, translation, rng.uniform(0.2, 2.0))
                    if obj in (1, 2) and rng.uniform() < 0.5:
                        est_r = est_r @ _rotation("z", 180.0)  # symmetric flip stays correct
                    rows.append(_row(scene, im, obj, score, est_r, est_t))
            for obj in (int(rng.choice(objects)), 7):  # a wrong pose and an absent object
                rotation, translation = _pose(rng)
                rows.append(_row(scene, im, obj, float(rng.uniform(0, 1)), rotation, translation))
            if (scene, im) != (1, 3):  # one image is outside the targets
                targets.append({"im_id": im, "scene_id": scene})
        directory = split / f"{scene:06d}"
        directory.mkdir(parents=True)
        (directory / "scene_gt.json").write_text(json.dumps(gt))
        (directory / "scene_gt_info.json").write_text(json.dumps(info))
        (directory / "scene_camera.json").write_text(json.dumps(camera))
    # More than 100 estimates in one image: only the 100 best-scoring are scored.
    first = json.loads((split / "000002" / "scene_gt.json").read_text())["5"][0]
    rotation = np.reshape(first["cam_R_m2c"], (3, 3))
    for i in range(104):
        est_r, est_t = _noisy(rng, rotation, np.array(first["cam_t_m2c"]), 1.0)
        rows.append(_row(2, 5, first["obj_id"], 0.01 * (i % 50), est_r, est_t))
    (base / "test_targets_bop24.json").write_text(json.dumps(targets))
    csv = base / "doper_tless-test.csv"
    csv.write_text("".join(rows))
    return {
        "gt_root": split,
        "targets": base / "test_targets_bop24.json",
        "models": base / "models_eval",
        "csv": csv,
    }


def write_scene(
    base: Path, instances: list[dict], estimates: list[tuple], objects: int = 4
) -> dict[str, Path]:
    """One scene and image; ``instances`` hold obj, R, t, visib; estimates (obj, score, R, t)."""
    write_models(base / "models_eval", objects)
    directory = base / "test_primesense" / "000001"
    directory.mkdir(parents=True)
    gt = [
        {"obj_id": i["obj"], "cam_R_m2c": np.ravel(i["R"]).tolist(), "cam_t_m2c": list(i["t"])}
        for i in instances
    ]
    info = [{"visib_fract": i["visib"]} for i in instances]
    (directory / "scene_gt.json").write_text(json.dumps({"0": gt}))
    (directory / "scene_gt_info.json").write_text(json.dumps({"0": info}))
    camera = {"0": {"cam_K": K, "depth_scale": 0.1}}
    (directory / "scene_camera.json").write_text(json.dumps(camera))
    (base / "test_targets_bop24.json").write_text(json.dumps([{"im_id": 0, "scene_id": 1}]))
    rows = ["scene_id,im_id,obj_id,score,R,t,time\n"]
    rows += [_row(1, 0, obj, score, r, t) for obj, score, r, t in estimates]
    (base / "doper_tless-test.csv").write_text("".join(rows))
    return {
        "gt_root": base / "test_primesense",
        "targets": base / "test_targets_bop24.json",
        "models": base / "models_eval",
        "csv": base / "doper_tless-test.csv",
    }


def _instance(obj: int, x: float, visib: float = 0.9) -> dict:
    return {"obj": obj, "R": np.eye(3), "t": [x, 0.0, 800.0], "visib": visib}


def _toolkit_root() -> Path | None:
    """The bop_toolkit checkout with ``scripts/`` (``BOP_TOOLKIT_ROOT`` or the uv git cache)."""
    if os.environ.get("BOP_TOOLKIT_ROOT"):
        root = Path(os.environ["BOP_TOOLKIT_ROOT"])
        return root if (root / "scripts" / "eval_bop24_pose.py").is_file() else None
    cache = Path(os.environ.get("UV_CACHE_DIR", Path.home() / ".cache" / "uv"))
    found = sorted(cache.glob(f"git-v0/checkouts/*/{TOOLKIT_SHORT}*/scripts/eval_bop24_pose.py"))
    return found[0].parents[1] if found else None


def run_toolkit(root: Path, bop: Path, csv: Path, eval_dir: Path) -> dict:
    """Run ``scripts/eval_bop24_pose.py`` end to end; return its scores and per-object APs."""
    env = {
        **os.environ,
        "BOP_PATH": str(bop),
        "PYTHONPATH": str(root),
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}",
    }
    command = [
        sys.executable,
        str(root / "scripts" / "eval_bop24_pose.py"),
        f"--result_filenames={csv.name}",
        f"--results_path={csv.parent}",
        f"--eval_path={eval_dir}",
        "--targets_filename=test_targets_bop24.json",
        "--num_workers=1",
    ]
    subprocess.run(command, check=True, env=env, cwd=root, capture_output=True)
    result = eval_dir / csv.name.split(".")[0]
    scores = json.loads((result / "scores_bop24.json").read_text())
    per_object = {}
    for error, ths in (("mssd", np.arange(0.05, 0.51, 0.05)), ("mspd", np.arange(5, 51, 5))):
        for th in ths:
            name = f"scores_th={th:.3f}_min-visib=0.100.json"
            data = json.loads((result / f"error={error}_ntop=0" / name).read_text())
            for obj, value in data["scores"].items():
                if data["num_targets_per_object"][obj] > 0:
                    per_object.setdefault((error, int(obj)), []).append(value)
    return {"scores": scores, "per_object": per_object}


def test_ap_matches_toolkit_script(tmp_path):
    pytest.importorskip("bop_toolkit_lib")
    root = _toolkit_root()
    if root is None:
        pytest.skip("bop_toolkit checkout with scripts/ not found (set BOP_TOOLKIT_ROOT)")
    paths = write_random_dataset(tmp_path / "bop" / "tless")
    expected = run_toolkit(root, tmp_path / "bop", paths["csv"], tmp_path / "eval")
    actual = ap.bop24_ap(paths["csv"], paths["gt_root"], paths["targets"], paths["models"])
    scores = expected["scores"]
    assert abs(actual.ap_mssd - scores["bop24_mAP_mssd"]) <= 1e-9
    assert abs(actual.ap_mspd - scores["bop24_mAP_mspd"]) <= 1e-9
    assert abs(actual.ap - scores["bop24_mAP"]) <= 1e-9
    assert 0.0 < actual.ap < 1.0  # the fixture exercises both hits and misses
    for (error, obj), values in expected["per_object"].items():
        assert abs(actual.per_object[obj][f"ap_{error}"] - float(np.mean(values))) <= 1e-9
    assert set(actual.per_object) == {obj for _, obj in expected["per_object"]}


def test_ap_duplicates_and_false_positives(tmp_path):
    instances = [_instance(4, -50.0), _instance(4, 50.0)]
    estimates = [
        (4, 0.9, np.eye(3), [-50.0, 0.0, 800.0]),
        (4, 0.8, np.eye(3), [-50.0, 0.0, 800.0]),  # duplicate of a matched GT: FP
        (4, 0.7, np.eye(3), [50.0, 0.0, 800.0]),
    ]
    paths = write_scene(tmp_path, instances, estimates)
    result = ap.bop24_ap(paths["csv"], paths["gt_root"], paths["targets"], paths["models"])
    assert result.ap_mssd == pytest.approx(DUPLICATE_AP, abs=1e-12)
    assert result.ap_mspd == pytest.approx(DUPLICATE_AP, abs=1e-12)
    assert result.per_object[4]["targets"] == 2


def test_ap_ignores_low_visibility_gt(tmp_path):
    instances = [_instance(4, -50.0), _instance(4, 50.0, visib=0.05), _instance(2, 0.0, 0.02)]
    estimates = [
        (4, 0.9, np.eye(3), [-50.0, 0.0, 800.0]),
        (4, 0.95, np.eye(3), [50.0, 0.0, 800.0]),  # on the barely visible GT: ignored
        (2, 0.9, np.eye(3), [0.0, 0.0, 800.0]),  # object 2 has no valid target
    ]
    paths = write_scene(tmp_path, instances, estimates)
    result = ap.bop24_ap(paths["csv"], paths["gt_root"], paths["targets"], paths["models"])
    assert result.ap == 1.0
    assert set(result.per_object) == {4}
    assert result.per_object[4]["targets"] == 1


def test_ap_mspd_normalizes_non_640_width(tmp_path):
    # A 6 mm shift parallel to the image plane at 800 mm moves every point of a flat model by
    # 7.5 px at fx = 1000: it fails the 5 px MSPD threshold at width 640 but not at 1280.
    paths = write_scene(tmp_path, [_instance(4, 0.0)], [(4, 0.9, np.eye(3), [6.0, 0.0, 800.0])])
    points = np.array([[x, y, 0.0] for x in (-40.0, 40.0) for y in (-30.0, 30.0)])
    _write_ply(paths["models"] / "obj_000004.ply", points)
    info = json.loads((paths["models"] / "models_info.json").read_text())
    info["4"]["diameter"] = 200.0  # MSSD 6 mm < 5 % of 200 mm: passes every threshold
    (paths["models"] / "models_info.json").write_text(json.dumps(info))
    args = (paths["csv"], paths["gt_root"], paths["targets"], paths["models"])
    native = ap.bop24_ap(*args, image_width=640)
    assert native.ap_mssd == 1.0
    assert native.ap_mspd == pytest.approx(0.9, abs=1e-12)
    assert ap.bop24_ap(*args, image_width=1280).ap_mspd == 1.0
    # T-LESS PrimeSense default width 720 px: 6.67 px still fails 5 px.
    assert ap.bop24_ap(*args).ap_mspd == pytest.approx(0.9, abs=1e-12)


def test_ap_requires_targets_file(tmp_path):
    paths = write_scene(tmp_path, [_instance(4, 0.0)], [])
    assert ap.resolve_targets(tmp_path) == (paths["targets"], "bop24")
    bop19 = tmp_path / "test_targets_bop19.json"
    bop19.write_text(paths["targets"].read_text())
    paths["targets"].unlink()
    assert ap.resolve_targets(tmp_path) == (bop19, "derived_from_bop19")
    assert ap.resolve_targets(tmp_path, bop19) == (bop19, "user")
    with pytest.raises(FileNotFoundError, match="test_targets_bop24.json"):
        ap.resolve_targets(tmp_path, tmp_path / "missing.json")
    bop19.unlink()
    with pytest.raises(FileNotFoundError, match="test_targets_bop19.json"):
        ap.resolve_targets(tmp_path)
    with pytest.raises(FileNotFoundError, match="targets file not found"):
        ap.bop24_ap(paths["csv"], paths["gt_root"], paths["targets"], paths["models"])


def test_ap_bop19_targets_give_the_bop24_ap(tmp_path):
    paths = write_random_dataset(tmp_path)
    bop24 = json.loads(paths["targets"].read_text())
    # BOP19 lists one row per (image, object) with inst_count; only the image ids are read.
    bop19 = [
        {"scene_id": t["scene_id"], "im_id": t["im_id"], "obj_id": obj, "inst_count": 1}
        for t in bop24
        for obj in (4, 1)
    ]
    bop19_path = tmp_path / "test_targets_bop19.json"
    bop19_path.write_text(json.dumps(bop19))
    args = (paths["csv"], paths["gt_root"])
    expected = ap.bop24_ap(*args, paths["targets"], paths["models"])
    assert ap.bop24_ap(*args, bop19_path, paths["models"]) == expected


def test_ap_empty_csv_scores_zero(tmp_path):
    paths = write_scene(tmp_path, [_instance(4, -50.0), _instance(2, 50.0)], [])
    result = ap.bop24_ap(paths["csv"], paths["gt_root"], paths["targets"], paths["models"])
    assert result.ap == 0.0 and result.estimates == 0
    assert {obj: v["ap"] for obj, v in result.per_object.items()} == {2: 0.0, 4: 0.0}
    assert result.average_time_per_image == -1.0  # no estimate: no runtime claim


def test_ap_without_valid_gt_raises(tmp_path):
    instances = [_instance(4, 0.0, visib=0.05)]
    paths = write_scene(tmp_path, instances, [(4, 0.9, np.eye(3), [0.0, 0.0, 800.0])])
    with pytest.raises(ValueError, match="no GT instance with visibility >= 0.1"):
        ap.bop24_ap(paths["csv"], paths["gt_root"], paths["targets"], paths["models"])


def test_ap_refuses_datasets_without_public_gt(tmp_path):
    paths = write_scene(tmp_path, [_instance(4, 0.0)], [])
    (paths["gt_root"] / "000001" / "scene_gt.json").unlink()
    with pytest.raises(ValueError, match="public"):
        ap.bop24_ap(paths["csv"], paths["gt_root"], paths["targets"], paths["models"])
    with pytest.raises(ValueError, match="public"):
        ap.require_public_gt("hopev2")
    ap.require_public_gt("tless")
    runs, root = tmp_path / "runs", tmp_path / "root"
    assert cli.main(["report", "hopev2", "--runs", str(runs), "--ap", "--root", str(root)]) == 1


def test_report_ap_dry_run_and_run(tmp_path, capsys):
    root, runs = tmp_path / "root", tmp_path / "runs"
    argv = ["report", "tless", "--runs", str(runs), "--ap", "--root", str(root), "--policy", "base"]
    assert cli.main([*argv, "--dry-run"]) == 0
    assert "test_targets_bop24.json" in capsys.readouterr().out
    assert not root.exists() and not runs.exists()
    instances = [_instance(4, -50.0), _instance(4, 50.0)]
    estimates = [(4, 0.9, np.eye(3), [-50.0, 0.0, 800.0])]
    paths = write_scene(root / "data", instances, estimates)
    csv = runs / "test" / "base" / "doperbase_tless-test.csv"
    csv.parent.mkdir(parents=True)
    csv.write_text(paths["csv"].read_text())
    assert cli.main(argv) == 0
    payload = json.loads((runs / "ap_tless_base.json").read_text())
    assert payload["ap_mssd"] == pytest.approx(51 / 101, abs=1e-12)  # recall 0.5 at precision 1
    assert payload["toolkit_revision"] == ap.TOOLKIT_REVISION
    assert payload["targets_origin"] == "bop24"
    assert payload["targets_sha256"] == sha256_file(paths["targets"])
    assert payload["csv_sha256"] == sha256_file(csv)
    bop19 = root / "data" / "test_targets_bop19.json"
    paths["targets"].rename(bop19)
    assert cli.main(argv) == 0
    payload = json.loads((runs / "ap_tless_base.json").read_text())
    assert payload["targets_origin"] == "derived_from_bop19"
    assert payload["targets"] == str(bop19)
    assert payload["ap_mssd"] == pytest.approx(51 / 101, abs=1e-12)
    assert cli.main([*argv, "--targets", str(bop19)]) == 0
    assert json.loads((runs / "ap_tless_base.json").read_text())["targets_origin"] == "user"
    bop19.unlink()
    assert cli.main(argv) == 1  # no targets file at all


def test_report_ap_scores_variant_csvs(tmp_path, capsys):
    root, runs = tmp_path / "root", tmp_path / "runs"
    argv = ["report", "tless", "--runs", str(runs), "--ap", "--root", str(root), "--policy", "base"]
    argv += ["--variant", "singleheatmap", "--variant", "multiview"]
    assert cli.main([*argv, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "ap_tless_base_singleheatmap.json" in out and "ap_tless_base_multiview.json" in out
    assert "doperbasesingleheatmap_tless-test.csv" in out and "ap_tless_base.json" not in out
    instances = [_instance(4, -50.0), _instance(4, 50.0)]
    estimates = [(4, 0.9, np.eye(3), [-50.0, 0.0, 800.0])]
    paths = write_scene(root / "data", instances, estimates)
    csv = runs / "test" / "base" / "singleheatmap" / "doperbasesingleheatmap_tless-test.csv"
    csv.parent.mkdir(parents=True)
    csv.write_text(paths["csv"].read_text())
    assert cli.main([*argv[:-2]]) == 0
    payload = json.loads((runs / "ap_tless_base_singleheatmap.json").read_text())
    assert payload["csv_sha256"] == sha256_file(csv) and payload["variant"] == "singleheatmap"
    assert payload["ap_mssd"] == pytest.approx(51 / 101, abs=1e-12)
    assert cli.main(["report", "tless", "--runs", str(runs), "--variant", "multiview"]) == 1
