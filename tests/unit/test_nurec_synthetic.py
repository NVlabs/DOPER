# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The NuRec synthetic baseline: render config, split, audit and the Blender assets."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from nurec_fixtures import NUREC_TEMPLATE

from doper import cli
from doper.nurec import synthetic
from doper.nurec.synthetic import audit_render, package_synthetic, render_config, split_synthetic
from tests.stl_helpers import bracket_triangles_mm

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "rendering" / "nurec_render.yaml"
ASSETS_SCRIPT = REPO / "rendering" / "nurec_assets.py"
OBJECT = "object"
NUM_KEYPOINTS = 64
SIZE = 32


def _asset_dirs(tmp_path: Path) -> dict[str, Path]:
    dirs = {}
    for key in synthetic.ASSET_KEYS:
        directory = tmp_path / key
        directory.mkdir()
        (directory / "asset.bin").write_bytes(b"x")
        dirs[key] = directory
    return dirs


def test_render_config_requires_asset_dirs(tmp_path):
    from rendering.config import load_render_config

    out, object_paths = tmp_path / "render.yaml", {"usd_path": "m.usda", "keypoints_json": "k.json"}
    dirs = _asset_dirs(tmp_path)
    with pytest.raises(ValueError, match="--hdri-dir, --distractor-dir"):
        render_config(TEMPLATE, out, asset_dirs={"texture_dir": dirs["texture_dir"]},
                      **object_paths)  # fmt: skip
    empty = tmp_path / "empty"
    empty.mkdir()
    for bad in (tmp_path / "missing", empty):
        with pytest.raises(ValueError, match="--texture-dir .* nonempty directory"):
            render_config(TEMPLATE, out, asset_dirs={**dirs, "texture_dir": bad}, **object_paths)
    unsafe = tmp_path / "hdri#1"
    shutil.copytree(dirs["hdri_dir"], unsafe)
    with pytest.raises(ValueError, match="plain path"):
        render_config(TEMPLATE, out, asset_dirs={**dirs, "hdri_dir": unsafe}, **object_paths)
    preset = yaml.safe_load(TEMPLATE.read_text())
    preset["scene"]["hdri_dir"] = "assets/hdri"
    (tmp_path / "preset.yaml").write_text(yaml.safe_dump(preset))
    with pytest.raises(ValueError, match="user inputs"):
        render_config(tmp_path / "preset.yaml", out, asset_dirs=dirs, **object_paths)
    assert not out.exists()
    render_config(TEMPLATE, out, asset_dirs=dirs, **object_paths)
    cfg = load_render_config(out)
    assert (cfg.hdri_dir, cfg.texture_dir, cfg.distractor_dir) == tuple(
        str(dirs[key].resolve()) for key in synthetic.ASSET_KEYS)  # fmt: skip
    assert Path(str(cfg.usd_path)).name == "m.usda" and Path(str(cfg.usd_path)).is_absolute()
    # The render config's settings.
    assert (cfg.resolution, cfg.num_keypoints, cfg.render_masks) == (640, 64, True)
    assert (cfg.mesh_name, cfg.keypoint_prefix) == (OBJECT, "KP_object_")
    assert (cfg.focal_length_mm, cfg.ground_style, cfg.world_background_modes) == (
        50.0, "texture", ("hdri",))  # fmt: skip
    assert cfg.num_distractors == (0, 8) and cfg.change_distractors_every == 1
    assert cfg.object_material_mode == "preserve"
    # Flow-style lists only (Blender's fallback YAML reader).
    assert "\n  - " not in out.read_text()


def test_split_counts_and_determinism():
    records = [{"id": i, "file_name": f"{i:06d}.png"} for i in range(10000)]
    train, val = split_synthetic(records)
    assert (len(train), len(val)) == (9750, 250)
    ids = [r["id"] for r in train + val]
    assert sorted(ids) == list(range(10000))
    shuffled = records[:]
    random.Random(7).shuffle(shuffled)
    assert split_synthetic(shuffled) == (train, val)
    digests = [hashlib.sha256(str(r["id"]).encode()).hexdigest() for r in train + val]
    assert digests == sorted(digests)
    small_val = split_synthetic(records[:4], train=3, val=1)[1]
    assert small_val[0]["id"] == max(
        range(4), key=lambda i: hashlib.sha256(str(i).encode()).digest()
    )
    with pytest.raises(ValueError, match="cannot be split"):
        split_synthetic(records[:9999])
    with pytest.raises(ValueError, match="repeat"):
        split_synthetic([*records[:9999], {"id": 0}])


def _write_render(
    root: Path, *, count: int = 3, faint_mask: int | None = None, mask_peak: int = 100
) -> dict:
    """A merged render of ``count`` samples at SIZE px, as synth_parallel writes it."""
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    rng = np.random.default_rng(0)
    images, annotations = [], []
    for i in range(count):
        name = f"{i:06d}.png"
        cv2.imwrite(str(root / "images" / name), rng.integers(0, 255, (SIZE, SIZE, 3), np.uint8))
        mask = np.zeros((SIZE, SIZE), np.uint8)
        mask[8:20, 8:20] = mask_peak if i == faint_mask else 255
        cv2.imwrite(str(root / "masks" / name), mask)
        keypoints = []
        for k in range(NUM_KEYPOINTS):
            keypoints += [float(k % SIZE), float(k // 4), 2 if k % 3 else 1]
        images.append({"id": i, "file_name": name, "width": SIZE, "height": SIZE})
        annotations.append({"id": i, "image_id": i, "category_id": 1, "bbox": [8, 8, 12, 12],
                            "area": 144, "keypoints": keypoints, "num_keypoints": NUM_KEYPOINTS,
                            "mask_file": f"masks/{name}"})  # fmt: skip
    category = {"id": 1, "name": OBJECT, "keypoints": [f"k{k}" for k in range(NUM_KEYPOINTS)]}
    coco = {"images": images, "annotations": annotations, "categories": [category]}
    (root / "annotations.json").write_text(json.dumps(coco))
    layout = {"keypoint_names": [f"k{k}" for k in range(NUM_KEYPOINTS)],
              "keypoints_3d": [[0.001 * k, 0.0, 0.0] for k in range(NUM_KEYPOINTS)]}  # fmt: skip
    (root / "keypoints_3d.json").write_text(json.dumps(layout))
    objects = {"blend_sha256": "b" * 64, "usd_sha256": "u" * 64, "keypoints_sha256": "k" * 64}
    manifest = {"status": "complete", "problems": [], "num_images": count, "num_samples": count,
                "seed": 142, "config_sha256": "c" * 64,
                "render_identity": {"object": objects}}  # fmt: skip
    (root / "manifest.json").write_text(json.dumps(manifest))
    return {"config_sha256": "c" * 64, "object_hashes": objects,
            "keypoints": root / "keypoints_3d.json"}  # fmt: skip


def test_audit_passes_fixture(tmp_path):
    expected = _write_render(tmp_path / "render")
    audit = audit_render(tmp_path / "render", count=3, resolution=SIZE, **expected)
    assert audit["status"] == "passed", audit["problems"]
    assert (audit["images"], audit["decoded_media"], audit["problem_count"]) == (3, 6, 0)
    assert audit["visible_keypoints_min"] == sum(1 for k in range(NUM_KEYPOINTS) if k % 3)
    assert len(audit["data_sha256"]) == 3 * 2 + 3
    out = tmp_path / "dataset"
    receipt = package_synthetic(tmp_path / "render", out, train=2, val=1, audit_sha256="a" * 64)
    assert (receipt["train"], receipt["val"]) == (2, 1)
    for split, size in (("train", 2), ("val", 1)):
        coco = json.loads((out / split / "annotations.json").read_text())
        assert len(coco["images"]) == len(coco["annotations"]) == size
        for image, ann in zip(coco["images"], coco["annotations"], strict=True):
            assert (out / split / "images" / image["file_name"]).is_file()
            assert (out / split / ann["mask_file"]).is_file() and ann["image_id"] == image["id"]
    assert (out / "keypoints_3d.json").is_file() and (out / "sha256.json").is_file()
    with pytest.raises(ValueError, match="not empty"):
        package_synthetic(tmp_path / "render", out, train=2, val=1)


def test_audit_flags_small_mask(tmp_path):
    expected = _write_render(tmp_path / "render", faint_mask=1)
    audit = audit_render(tmp_path / "render", count=3, resolution=SIZE, **expected)
    assert audit["status"] == "failed"
    assert audit["problems"] == ["image 1: mask has no foreground pixel (>= 128)"]


def test_audit_mask_foreground_boundary(tmp_path):
    for peak, status in ((127, "failed"), (128, "passed")):
        expected = _write_render(tmp_path / str(peak), faint_mask=0, mask_peak=peak)
        assert audit_render(tmp_path / str(peak), count=3, resolution=SIZE,
                            **expected)["status"] == status  # fmt: skip


def test_audit_counts_decoded_media(tmp_path):
    expected = _write_render(tmp_path / "render")
    (tmp_path / "render" / "images" / "000002.png").write_bytes(b"not a png")
    audit = audit_render(tmp_path / "render", count=3, resolution=SIZE, **expected)
    assert audit["decoded_media"] == 5 and audit["status"] == "failed"


def test_audit_flags_identity_count_and_size(tmp_path):
    expected = _write_render(tmp_path / "render")
    other = {**expected, "config_sha256": "d" * 64,
             "object_hashes": {**expected["object_hashes"], "usd_sha256": "v" * 64}}  # fmt: skip
    audit = audit_render(tmp_path / "render", count=4, seed=1, resolution=64, **other)
    text = "\n".join(audit["problems"])
    for needle in ("not 4", "seed 142 is not 1", "render config differs",
                   "usd_sha256 differs", "not a decodable 64x64"):  # fmt: skip
        assert needle in text
    assert audit_render(tmp_path / "missing")["problems"][0].startswith("manifest.json or")


def test_blender_points_hash_is_the_recipe_points_hash():
    from doper.keypoints import points_sha256
    from rendering import nurec_assets

    rng = np.random.default_rng(5)
    points = rng.normal(size=(64, 3))
    assert nurec_assets.points_sha256(points.tolist()) == points_sha256(points)
    assert nurec_assets.points_sha256(points[:, ::-1]) == points_sha256(points[:, ::-1])
    assert nurec_assets.points_sha256(points.astype(">f8")) == points_sha256(points)


def test_synthetic_stage_plan_and_flags(tmp_path, capsys):
    argv = [
        "prepare",
        str(NUREC_TEMPLATE),
        "--root",
        str(tmp_path),
        "--stage",
        "synthetic",
        "--dry-run",
    ]
    assert cli.main(argv) == 1
    assert "--hdri-dir, --texture-dir, --distractor-dir" in capsys.readouterr().out
    dirs = ["--hdri-dir", "h", "--texture-dir", "t", "--distractor-dir", "d"]
    assert cli.main([*argv, *dirs]) == 0
    out = capsys.readouterr().out
    for step in ("author the render object", "render the synthetic images", "audit every",
                 "split the render", "9750 train / 250 val"):  # fmt: skip
        assert step in out
    assert not any(tmp_path.iterdir())
    labels = [
        "prepare",
        str(NUREC_TEMPLATE),
        "--root",
        str(tmp_path),
        "--stage",
        "labels",
        "--dry-run",
    ]
    assert cli.main([*labels, "--hdri-dir", "h"]) == 1
    assert "only for --stage synthetic" in capsys.readouterr().out
    assert cli.main([*argv, *dirs, "--render-workers", "0"]) == 1
    assert "positive" in capsys.readouterr().out


def test_synthetic_stage_render_audit_and_split(tmp_path, monkeypatch):
    """The render, audit and split steps around a faked ``synth_parallel.py`` run."""
    from dataclasses import replace
    from types import SimpleNamespace

    from doper.bop.sources import sha256_file
    from doper.cli.prepare import nurec_synthetic
    from doper.recipes import load_any_recipe

    root = tmp_path / "root"
    assets = root / "synthetic" / "assets"
    assets.mkdir(parents=True)
    fixture = _write_render(tmp_path / "fixture")
    shutil.copyfile(fixture["keypoints"], assets / "keypoints_3d.json")
    for name in ("object.blend", "model.usda"):
        (assets / name).write_bytes(name.encode())
    files = {name: sha256_file(assets / name)
             for name in ("object.blend", "model.usda", "keypoints_3d.json")}  # fmt: skip
    (assets / "receipt.json").write_text(json.dumps({"files": files}))
    points = json.loads(fixture["keypoints"].read_text())["keypoints_3d"]
    recipe = load_any_recipe(NUREC_TEMPLATE)
    pin = hashlib.sha256(np.asarray(points, dtype="<f8").tobytes()).hexdigest()
    recipe = replace(recipe, object=replace(recipe.object, keypoints_points_sha256=pin))
    calls = []

    def fake_render(argv, what, *, capture):
        calls.append(argv)
        outdir = Path(argv[argv.index("--outdir") + 1])
        shutil.copytree(tmp_path / "fixture", outdir)
        manifest = json.loads((outdir / "manifest.json").read_text())
        object_hashes = {"blend_sha256": files["object.blend"], "usd_sha256": files["model.usda"],
                         "keypoints_sha256": files["keypoints_3d.json"]}  # fmt: skip
        manifest.update(config_sha256=sha256_file(argv[argv.index("--config") + 1]),
                        render_identity={"object": object_hashes})  # fmt: skip
        (outdir / "manifest.json").write_text(json.dumps(manifest))

    monkeypatch.setattr(nurec_synthetic, "_run_checked", fake_render)
    monkeypatch.setattr(nurec_synthetic, "SYNTHETIC_SPLIT", (2, 1))
    monkeypatch.setattr(synthetic, "RENDER_COUNT", 3)
    monkeypatch.setattr(synthetic, "RENDER_RESOLUTION", SIZE)
    monkeypatch.setenv("DOPER_BLENDER", "blender")
    dirs = _asset_dirs(tmp_path)
    args = SimpleNamespace(**dirs, render_workers=2, force=False)
    nurec_synthetic._nurec_synthetic_render(root, args)
    argv = calls[0]
    assert argv[argv.index("--num-images") + 1] == "3" and "--resume" not in argv
    assert argv[argv.index("--seed") + 1] == "142" and argv[argv.index("--num-workers") + 1] == "2"
    command = json.loads((root / "synthetic" / "command.json").read_text())
    assert command["argv"] == argv
    nurec_synthetic._nurec_synthetic_render(root, args)  # complete: kept
    assert len(calls) == 1
    other = SimpleNamespace(**{**dirs, "hdri_dir": dirs["texture_dir"]}, render_workers=2,
                            force=False)  # fmt: skip
    with pytest.raises(ValueError, match="another config"):
        nurec_synthetic._nurec_synthetic_render(root, other)
    nurec_synthetic._nurec_synthetic_audit(root, False)
    audit = json.loads((root / "synthetic" / "audit.json").read_text())
    assert audit["status"] == "passed" and audit["images"] == 3
    nurec_synthetic._nurec_synthetic_split(root, recipe, False)
    dataset = root / "synthetic" / "dataset"
    receipt = json.loads((root / "synthetic" / "receipt.json").read_text())
    assert (receipt["train"], receipt["val"]) == (2, 1) and "sha256" in receipt["split_rule"]
    assert len(json.loads((dataset / "train" / "annotations.json").read_text())["images"]) == 2
    nurec_synthetic._nurec_synthetic_split(root, recipe, False)  # same audit: kept
    # A recorded audit is reused only against the same config and authored object.
    receipt = json.loads((assets / "receipt.json").read_text())
    receipt["files"]["object.blend"] = "0" * 64
    (assets / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="blend_sha256 differs"):
        nurec_synthetic._nurec_synthetic_audit(root, False)
    (assets / "receipt.json").write_text(json.dumps({"files": files}))
    # A failed audit is recorded and refuses the split.
    (root / "synthetic" / "render" / "masks" / "000001.png").unlink()
    with pytest.raises(ValueError, match="failed its audit"):
        nurec_synthetic._nurec_synthetic_audit(root, True)
    with pytest.raises(ValueError, match="records a failed audit"):
        nurec_synthetic._nurec_synthetic_split(root, recipe, True)
    (root / "synthetic" / "audit.json").unlink()
    with pytest.raises(FileNotFoundError, match="audit step did not finish"):
        nurec_synthetic._nurec_synthetic_split(root, recipe, True)


# Blender: a generated bracket mesh's assets, their identity and material, and a 4-image render.


def _blender() -> str:
    binary = os.environ.get("DOPER_BLENDER") or shutil.which("blender")
    if not binary:
        pytest.skip("Blender >= 4.2 not found (set DOPER_BLENDER or put blender on PATH)")
    return binary


def _write_mesh(path: Path) -> np.ndarray:
    """Write the test bracket as an indexed OBJ in meters; return its float64 vertices."""
    corners = bracket_triangles_mm().reshape(-1, 3) / 1000.0
    vertices, inverse = np.unique(np.round(corners, 9), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3) + 1
    lines = [f"v {x:.9f} {y:.9f} {z:.9f}" for x, y, z in vertices]
    lines += [f"f {a} {b} {c}" for a, b, c in faces]
    path.write_text("\n".join(lines) + "\n")
    return np.array(
        [line.split()[1:4] for line in lines if line.startswith("v ")], dtype=np.float64
    )


def _test_layout(path: Path, vertices: np.ndarray) -> str:
    """64 mesh vertices as a layout; return its points pin."""
    points = vertices[np.linspace(0, len(vertices) - 1, 64, dtype=int)].tolist()
    names = [f"KP_{OBJECT}_{i:03d}" for i in range(NUM_KEYPOINTS)]
    path.write_text(json.dumps({"object": OBJECT, "keypoint_names": names, "keypoints_3d": points}))
    return hashlib.sha256(np.asarray(points, dtype="<f8").tobytes()).hexdigest()


def _run_assets(blender: str, out: Path, keypoints: Path, pin: str, mesh: Path,
                mesh_pin: str) -> subprocess.CompletedProcess:  # fmt: skip
    argv = [blender, "--background", "--factory-startup", "--python-exit-code", "1", "--python",
            str(ASSETS_SCRIPT), "--", "--mesh", str(mesh), "--mesh-sha256", mesh_pin,
            "--keypoints", str(keypoints), "--points-sha256", pin, "--out", str(out)]  # fmt: skip
    return subprocess.run(argv, capture_output=True, text=True, timeout=600, check=False)


@pytest.fixture(scope="module")
def assets(tmp_path_factory) -> dict:
    blender = _blender()
    root = tmp_path_factory.mktemp("nurec_assets")
    mesh = root / "object.obj"
    vertices = _write_mesh(mesh)
    mesh_pin = synthetic.sha256_file(mesh)
    pin = _test_layout(root / "layout.json", vertices)
    result = _run_assets(blender, root / "assets", root / "layout.json", pin, mesh, mesh_pin)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return {"blender": blender, "root": root, "out": root / "assets", "pin": pin, "mesh": mesh,
            "mesh_pin": mesh_pin, "vertices": vertices}  # fmt: skip


@pytest.mark.blender
def test_nurec_assets_identity(assets):
    from rendering.usd_mesh import read_mesh_usda

    out = assets["out"]
    receipt = json.loads((out / "assets_receipt.json").read_text())
    identity = receipt["identity"]
    faces = sum(line.startswith("f ") for line in assets["mesh"].read_text().splitlines())
    assert (identity["vertices"], identity["faces"]) == (len(assets["vertices"]), faces)
    assert identity["faces_equal"] and identity["usd_faces_equal"]
    assert identity["identity_transform"] and identity["usd_keypoint_names_equal"]
    assert identity["max_vertex_error_m"] <= 1e-8 and identity["usd_max_point_difference_m"] <= 1e-8
    assert identity["max_keypoint_error_mm"] <= 0.001 and identity["max_nearest_vertex_mm"] == 0
    assert receipt["inputs"]["mesh"]["sha256"] == assets["mesh_pin"]
    assert receipt["inputs"]["keypoints"]["points_sha256"] == assets["pin"]
    for name, digest in receipt["files"].items():
        assert synthetic.sha256_file(out / name) == digest
    usd = read_mesh_usda(str(out / "model.usda"))
    np.testing.assert_allclose(usd["points"], assets["vertices"], rtol=0, atol=1e-8)
    layout = json.loads((assets["root"] / "layout.json").read_text())
    written = json.loads((out / "keypoints_3d.json").read_text())
    assert written["keypoint_names"] == layout["keypoint_names"] == usd["keypoint_names"]
    assert written["keypoints_3d"] == layout["keypoints_3d"]
    # Another layout pin, another mesh pin, or an output that is not empty, is refused.
    mesh, layout_path, other = (
        assets["mesh"],
        assets["root"] / "layout.json",
        assets["root"] / "other",
    )
    refused = _run_assets(assets["blender"], other, layout_path, "0" * 64, mesh, assets["mesh_pin"])
    assert refused.returncode != 0 and "is not the pin" in refused.stdout + refused.stderr
    refused = _run_assets(assets["blender"], other, layout_path, assets["pin"], mesh, "0" * 64)
    assert refused.returncode != 0 and "is not the mesh pin" in refused.stdout + refused.stderr
    refused = _run_assets(
        assets["blender"], out, layout_path, assets["pin"], mesh, assets["mesh_pin"]
    )
    assert refused.returncode != 0 and "is not empty" in refused.stdout + refused.stderr


MATERIAL_PROBE = """
import bpy, json, sys
mesh = bpy.data.objects["object"].data
slots = [p.material_index for p in mesh.polygons]
out = []
for i, m in enumerate(mesh.materials):
    bsdf = m.node_tree.nodes["Principled BSDF"]
    out.append({"name": m.name, "faces": [f for f, s in enumerate(slots) if s == i][:40],
                "count": slots.count(i), "rgba": list(bsdf.inputs["Base Color"].default_value),
                "metallic": bsdf.inputs["Metallic"].default_value,
                "roughness": bsdf.inputs["Roughness"].default_value})
print("PROBE" + json.dumps(out))
print("TYPES" + json.dumps({"objects": sorted({o.type for o in bpy.data.objects}),
                            "lights": len(bpy.data.lights), "cameras": len(bpy.data.cameras)}))
"""


@pytest.mark.blender
def test_nurec_assets_material(assets):
    argv = [assets["blender"], "--background", "--factory-startup", str(assets["out"] /
            "object.blend"), "--python-expr", MATERIAL_PROBE]  # fmt: skip
    result = subprocess.run(argv, capture_output=True, text=True, timeout=300, check=False)
    line = next(x for x in result.stdout.splitlines() if x.startswith("PROBE"))
    (body,) = json.loads(line[len("PROBE") :])
    faces = sum(line.startswith("f ") for line in assets["mesh"].read_text().splitlines())
    assert (body["name"], body["count"]) == ("object_body", faces)
    np.testing.assert_allclose(body["rgba"], [0.02, 0.02, 0.02, 1.0], atol=1e-6)
    assert body["roughness"] == pytest.approx(0.45)
    # Only the mesh and its empties: no camera or light enters the renders.
    line = next(x for x in result.stdout.splitlines() if x.startswith("TYPES"))
    assert json.loads(line[len("TYPES") :]) == {"objects": ["EMPTY", "MESH"], "lights": 0,
                                                "cameras": 0}  # fmt: skip


@pytest.mark.blender
def test_four_image_render_passes_the_audit(assets, tmp_path):
    from doper.bop.sources import sha256_file

    out = assets["out"]
    config = yaml.safe_load(TEMPLATE.read_text())
    config["output"]["resolution"] = SIZE * 2
    config["object"].update(usd_path=str(out / "model.usda"),
                            keypoints_json=str(out / "keypoints_3d.json"))  # fmt: skip
    # Procedural backgrounds: no asset directories in a test.
    config["scene"].update(ground_style="procedural", procedural_world_background=True,
                           world_background_modes="studio,random,dark",
                           num_distractors_max=0)  # fmt: skip
    for key in synthetic.ASSET_KEYS:
        del config["scene"][key]
    config["render"].update({"engine": "CYCLES", "cycles_samples": 4, "threads": 1})
    path = tmp_path / "render.yaml"
    path.write_text(yaml.dump(config, Dumper=synthetic._flow_lists_dumper(), sort_keys=False))
    render = tmp_path / "render"
    argv = [sys.executable, str(REPO / "rendering" / "synth_parallel.py"), "--blend-file",
            str(out / "object.blend"), "--usd", str(out / "model.usda"), "--keypoints-json",
            str(out / "keypoints_3d.json"), "--config", str(path), "--outdir", str(render),
            "--num-images", "4", "--num-workers", "1", "--seed", "142", "--blender-bin",
            assets["blender"]]  # fmt: skip
    result = subprocess.run(
        argv, capture_output=True, text=True, timeout=900, cwd=REPO, check=False
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    files = json.loads((out / "assets_receipt.json").read_text())["files"]
    audit = audit_render(
        render, count=4, seed=142, resolution=SIZE * 2, config_sha256=sha256_file(path),
        object_hashes={"blend_sha256": files["object.blend"], "usd_sha256": files["model.usda"],
                       "keypoints_sha256": files["keypoints_3d.json"]},
        keypoints=out / "keypoints_3d.json",
    )  # fmt: skip
    assert audit["status"] == "passed", audit["problems"]
    assert audit["images"] == 4 and audit["visible_keypoints_min"] >= 0
    receipt = package_synthetic(render, tmp_path / "dataset", train=3, val=1)
    assert (receipt["train"], receipt["val"]) == (3, 1)


def test_force_from_render_keeps_synthetic_assets(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from doper.cli.prepare import nurec_synthetic
    from doper.cli.prepare.nurec_capture import CaptureContext
    from doper.recipes import load_any_recipe

    seen = {}
    for name in ("assets", "audit", "split"):
        monkeypatch.setattr(nurec_synthetic, f"_nurec_synthetic_{name}",
                            lambda *args, name=name: seen.setdefault(name, args[-1]))  # fmt: skip
    monkeypatch.setattr(nurec_synthetic, "_nurec_synthetic_render",
                        lambda root, args: seen.setdefault("render", args.force))  # fmt: skip
    dirs = {key: tmp_path / key for key in synthetic.ASSET_KEYS}
    recipe = load_any_recipe(NUREC_TEMPLATE)
    for start, expected in (("render", [False, True, True, True]), (None, [False] * 4),
                            ("assets", [True] * 4)):  # fmt: skip
        seen.clear()
        args = SimpleNamespace(**dirs, render_workers=None, force=False, force_from=start)
        for step in nurec_synthetic._nurec_synthetic_steps(CaptureContext(tmp_path, recipe), args):
            step.action()
        assert [seen[k] for k in ("assets", "render", "audit", "split")] == expected
