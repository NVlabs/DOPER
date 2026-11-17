# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BlenderProc campaign for the generated BOP objects (HANDAL obj40, HOT3D obj33).

Everything except the BlenderProc scene itself runs without BlenderProc: the pure-NumPy scene plan
(against draws recorded from the source renderers), the layout partition, the audit and
visible-count canonicalization on a synthetic layout, the pinned asset manifest and the campaign
command line. ``test_one_layout_two_views_smoke`` needs BlenderProc 2.8.0 and a GPU.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import bproc_fakes
import pytest

from rendering.blenderproc import assets, audit, campaign, install, plan

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN = json.loads((Path(__file__).parent / "golden" / "bproc_plan_source.json").read_text())
CONFIGS = ("handal_obj40", "hot3d_obj33")
HOST_MODULES = (
    "rendering.blenderproc",
    "rendering.blenderproc.plan",
    "rendering.blenderproc.campaign",
    "rendering.blenderproc.assets",
    "rendering.blenderproc.audit",
    "rendering.blenderproc.install",
)


def golden_pools(config: plan.RenderConfig) -> plan.ScenePools:
    low, high = config.control_objects
    return plan.ScenePools(
        target_candidates=tuple(i for i in range(low, high + 1) if i != config.target_object),
        distractors={k: tuple(v) for k, v in GOLDEN["pools"]["distractors"].items()},
        room_materials=tuple(GOLDEN["pools"]["room_materials"]),
    )


@pytest.mark.parametrize(
    ("name", "seed"),
    [(n, int(s)) for n in CONFIGS for s in GOLDEN["draws"][n]],
)
def test_plan_scene_matches_source_draws(name, seed):
    config = plan.load_config(name)
    expected = GOLDEN["draws"][name][str(seed)]
    actual = plan.plan_scene(seed, golden_pools(config), config).as_record()
    assert sorted(expected["room_materials"]) == list(golden_pools(config).room_materials)
    for key in (
        "target_ids",
        "distractors",
        "objects",
        "material_samples",
        "room_material",
        "emission",
        "emission_color",
        "point_light_energy",
        "point_light_color",
        "point_light_position_m",
        "rng_state_sha256",
    ):
        assert actual[key] == expected[key], key


@pytest.mark.parametrize(("name", "seed"), [("handal_obj40", 60003), ("hot3d_obj33", 70008)])
def test_scene_realizes_the_plan_until_physics(tmp_path, name, seed):
    """``scene.py`` under recording fakes loads, shades and lights exactly what the plan drew."""
    assets_root = bproc_fakes.write_fixture_assets(tmp_path / "assets")
    config = plan.load_config(name)
    script = REPO_ROOT / "rendering" / "blenderproc" / "scene.py"
    args = [
        "--assets",
        str(assets_root),
        "--config",
        str(plan.CONFIG_DIR / f"{name}.json"),
        "--out",
        str(tmp_path / "attempt"),
        "--seed",
        str(seed),
        "--gpu-index",
        "0",
    ]
    recorded = bproc_fakes.record_draws(script, args, tmp_path, "Fake OptiX GPU")
    expected = plan.plan_scene(seed, plan.load_pools(assets_root, config), config).as_record()
    assert recorded["dataset"] == config.dataset
    for key in (
        "target_ids",
        "distractors",
        "objects",
        "material_samples",
        "room_material",
        "emission",
        "emission_color",
        "point_light_energy",
        "point_light_color",
        "point_light_position_m",
        "rng_state_sha256",
    ):
        assert recorded[key] == expected[key], key
    assert recorded == GOLDEN["draws"][name][str(seed)]


def test_plan_is_independent_of_material_listing_order():
    config = plan.load_config("hot3d_obj33")
    pools = golden_pools(config)
    shuffled = dataclasses.replace(pools, room_materials=tuple(reversed(pools.room_materials)))
    with pytest.raises(ValueError, match="sorted"):
        plan.plan_scene(70000, shuffled, config)


def test_partition_and_seeds_match_source():
    # Source: ``assigned_layouts`` = range(worker, layouts, workers); seed = base + layout;
    # layout % 10 == 9 -> val_pbr, whole layouts only.
    assert campaign.assigned_layouts(3, 32, 1000) == list(range(3, 1000, 32))
    assert campaign.assigned_layouts(7, 8, 1000) == list(range(7, 1000, 8))
    covered = sorted(i for w in range(32) for i in campaign.assigned_layouts(w, 32, 1000))
    assert covered == list(range(1000))
    for bad in [(-1, 8, 10), (8, 8, 10), (0, 0, 10), (0, 1, 0)]:
        with pytest.raises(ValueError, match="partition"):
            campaign.assigned_layouts(*bad)
    config = plan.load_config("handal_obj40")
    splits = [campaign.split_for_layout(i, config) for i in range(1000)]
    assert splits.count("val_pbr") == 100
    assert all((s == "val_pbr") == (i % 10 == 9) for i, s in enumerate(splits))
    assert campaign.layout_seed(config, 17) == 60017
    assert campaign.layout_seed(plan.load_config("hot3d_obj33"), 999) == 70999
    assert (config.layouts, config.frames_per_layout, config.workers) == (1000, 25, 32)
    assert plan.load_config("hot3d_obj33").workers == 8


# --- synthetic layout -------------------------------------------------------------------------


def small_config(name: str = "hot3d_obj33", frames: int = 3) -> plan.RenderConfig:
    config = plan.load_config(name)
    camera = dataclasses.replace(config.camera, K=bproc_fakes.SMALL_K, width=32, height=24)
    return dataclasses.replace(config, camera=camera, frames_per_layout=frames)


def make_attempt(
    root: Path, config: plan.RenderConfig, *, count_offset: int = 0, target_visible: bool = True
) -> Path:
    last = (config.frames_per_layout - 1, 0)
    return bproc_fakes.make_attempt(
        root,
        dataset=config.dataset,
        target=config.target_object,
        frames=config.frames_per_layout,
        visible_offsets={last: count_offset},
        target_visible=target_visible,
    )


write_keypoints = bproc_fakes.write_keypoints


def assembled(tmp_path: Path, config: plan.RenderConfig, **kwargs) -> Path:
    attempt = make_attempt(tmp_path / "attempt", config, **kwargs)
    layout = tmp_path / "layout"
    campaign.assemble_layout(attempt, layout, config)
    return layout


def test_audit_canonicalizes_counts_within_bound(tmp_path):
    config = small_config()
    layout = assembled(tmp_path, config, count_offset=config.count_bound_px)
    result = audit.audit_layout(layout, config, write_keypoints(tmp_path / "kp.json"))
    assert result.ok and result.max_count_correction == config.count_bound_px
    info = json.loads((layout / "scene_gt_info.json").read_text())
    raw = json.loads((layout / "records" / "writer_raw" / "scene_gt_info.json").read_text())
    last = str(config.frames_per_layout - 1)
    assert raw[last][0]["px_count_visib"] == info[last][0]["px_count_visib"] + 8
    receipt = json.loads((layout / "records" / "visible_count_canonicalization.json").read_text())
    assert receipt["max_allowed_count_correction"] == 8 and len(receipt["changes"]) == 1
    assert result.target_positive_frames == config.frames_per_layout
    assert json.loads((layout / "records" / "AUDITED.json").read_text())["ok"] is True


def test_audit_rejects_count_beyond_bound(tmp_path):
    config = small_config()
    layout = assembled(tmp_path, config, count_offset=config.count_bound_px + 1)
    with pytest.raises(audit.CountBoundError, match="8"):
        audit.audit_layout(layout, config, write_keypoints(tmp_path / "kp.json"))
    assert not (layout / "records" / "AUDITED.json").exists()
    tight = dataclasses.replace(config, count_bound_px=0)
    layout2 = assembled(tmp_path / "b", tight, count_offset=1)
    with pytest.raises(audit.CountBoundError):
        audit.audit_layout(layout2, tight, write_keypoints(tmp_path / "kp.json"))


def test_audit_retains_zero_visible_target(tmp_path):
    config = small_config("handal_obj40")
    layout = assembled(tmp_path, config, target_visible=False)
    result = audit.audit_layout(layout, config, write_keypoints(tmp_path / "kp.json"))
    assert result.ok and result.zero_visible_layout_retained
    assert result.target_positive_frames == 0
    assert result.target_visibility == [0.0] * config.frames_per_layout


def test_audit_rejects_wrong_pose_and_population(tmp_path):
    config = small_config()
    layout = assembled(tmp_path, config)
    gt = json.loads((layout / "scene_gt.json").read_text())
    gt["0"][3]["cam_t_m2c"][0] += 1.0
    (layout / "scene_gt.json").write_text(json.dumps(gt))
    with pytest.raises(ValueError, match="projection"):
        audit.audit_layout(layout, config, write_keypoints(tmp_path / "kp.json"))
    layout2 = assembled(tmp_path / "b", config)
    gt = json.loads((layout2 / "scene_gt.json").read_text())
    gt["1"] = gt["1"][:-1]
    (layout2 / "scene_gt.json").write_text(json.dumps(gt))
    with pytest.raises(ValueError, match="instance"):
        audit.audit_layout(layout2, config, write_keypoints(tmp_path / "kp.json"))


# --- assets and install -------------------------------------------------------------------------

SOURCE_ASSET_SHAS = {
    "tless": "6a29c59766b8d2af05e62e71739f0ca7243ad81bc3c7f9a24925504a8cb37928",
    "hb": "026a9417387e7225463e63fd0f4ac9b1b15a961d85f3095f9dbf416f0b8bd9b8",
    "tyol": "a200a97b26adecd9dc38da2fa0e9841048d898def7d25741ec7fd67bfd864390",
    "Metal001": "0b22df0a8fc4706b2e91f26d5ae6e475dd09bfaf74ad27d1883e5d074fef088e",
    "Metal004": "d517565b0f1694c7309500e604a77ca497d2fc57f61f85bff464a40c90e1b376",
    "Metal032": "4b8884843c490963d5734be036c35639d064e7a817eb61145322f69c6c19895a",
    "Plastic001": "b5794733ceb475d3955d69607e7f8e5bf6305ca4395005367f5e7784dae14e35",
    "WoodFloor007": "8d7065ec1281113d68c1e9dd7593df833a9727b4c9bc46e98ac68a65c9374063",
    "WoodFloor018": "ef3056464cafe4896f9a2187bd6505015d0260dff5dad0babdb713e072e0aad6",
    "Bricks002": "d71530de46996ef7d44bfdca74abc1d90936157f54dae9bb986749e81b38dbd6",
    "Bricks020": "39f1ab1199ff7678517ab9c878acd99634dd021bba95add9514f3ab99279df11",
    "Tiles011": "a29a0f415424a37534103dc4e017cc7d2a5479d7f0b6ccc149cd23c9e4009bbb",
    "Tiles046": "9c690ae51380ad00d2318c8587fc20e052de0aba5b2d4150eb5123e706faac79",
    "Concrete013": "8c5f350480abd73b4a27a1b2ea4b02ec08d1444598c0cf51ac2a17e1fb297b76",
    "Ground024": "f4da7152a8c8cd299010e98b341e27185cdce5eb065febac47d08e41516d4692",
    "Carpet004": "3dea5a23316442f03c836608acfabee8010fb055b95f3a19a85d774a71c60233",
    "Fabric030": "7fc77fde6203911713a2fa41fa54f41f400c2670f7f7a3f4d57567dfdbf733e8",
    "Rocks006": "50337c03a369a8a4622a2f88281d543e3d65b13801b9e9eb9e294a6514bc090b",
    "Marble001": "38c22282d25a8b65be383f4fbedea54927461cd4dc7616a14065d8ce2ed4e4f5",
    "blender": "be0fbaa0c1e52d4552023220b4c67351efbb707cac49bb381fcbee2182447005",
}
SOURCE_REVISIONS = {
    "tless": "5fd309a04476a842d93abfb584fba9ee7caecdf1",
    "hb": "985ebaef7b1bd4bffb2039f18dd81faa647fd060",
    "tyol": "b8226af3cb5fb143980b6bd44b01ea3134746327",
}


def fake_zip(path: Path, members: dict[str, bytes]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path.read_bytes()


def test_asset_manifest_shas_match_source():
    pinned = assets.load_manifest()
    actual = {k: v["sha256"] for k, v in pinned["bop_models"].items()}
    actual |= {k: v["sha256"] for k, v in pinned["materials"].items()}
    actual["blender"] = pinned["blender"]["sha256"]
    assert actual == SOURCE_ASSET_SHAS
    assert {k: v["revision"] for k, v in pinned["bop_models"].items()} == SOURCE_REVISIONS
    assert all(v["license"] == "CC0" for v in pinned["materials"].values())
    assert pinned["bop_toolkit_revision"] == install.BOP_TOOLKIT_REVISION


def test_fetch_verifies_shas_and_writes_manifest(tmp_path):
    staging = tmp_path / "blobs"
    blobs: dict[str, bytes] = {}
    for name in ("tless", "hb", "tyol"):
        folder = "models_cad" if name == "tless" else "models"
        blobs[name] = fake_zip(
            staging / f"{name}.zip", {f"{folder}/models_info.json": b"{}", f"{folder}/a.ply": b"p"}
        )
    for name in assets.load_manifest()["materials"]:
        blobs[name] = fake_zip(staging / f"{name}.zip", {f"{name}_2K-JPG_Color.jpg": b"c"})
    pinned = assets.load_manifest()
    for group in ("bop_models", "materials"):
        for name in pinned[group]:
            pinned[group][name]["sha256"] = hashlib.sha256(blobs[name]).hexdigest()

    def downloader(hf_repo, filename, revision, destination):
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / filename
        target.write_bytes(blobs[hf_repo.split("/")[1]])
        return target

    def http_get(url, path):
        name = url.split("file=")[1].split("_2K")[0]
        path.write_bytes(blobs[name])

    dest = assets.fetch(tmp_path / "assets", downloader=downloader, http_get=http_get, pins=pinned)
    manifest = json.loads((dest / assets.MANIFEST_NAME).read_text())
    assert "bop/tless/models_cad/a.ply" in manifest["files"]
    assert "cctextures/Marble001/Marble001_2K-JPG_Color.jpg" in manifest["files"]
    assert json.loads((dest / "READY.json").read_text())["manifest_sha256"]
    assets.verify(dest)
    (dest / "bop/hb/models/a.ply").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="a.ply"):
        assets.verify(dest)
    pinned["materials"]["Metal001"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        assets.fetch(tmp_path / "other", downloader=downloader, http_get=http_get, pins=pinned)


def test_pin_bop_toolkit_refuses_unexpected_files(tmp_path):
    site = tmp_path / "site-packages"
    package = site / "blenderproc" / "python"
    for relative in install.PIN_FILES:
        (package / relative).parent.mkdir(parents=True, exist_ok=True)
        (package / relative).write_text('setup_pip(["git+https://github.com/thodan/bop_toolkit"])')
    meta = site / "blenderproc-2.8.0.dist-info" / "METADATA"
    meta.parent.mkdir()
    meta.write_text("Name: blenderproc\nVersion: 2.8.0\n")
    with pytest.raises(ValueError, match="unexpected"):
        install.pin_bop_toolkit(site)
    meta.write_text("Name: blenderproc\nVersion: 2.9.0\n")
    with pytest.raises(ValueError, match="2.8.0"):
        install.pin_bop_toolkit(site)


def test_host_modules_import_without_bpy():
    banned = ("bpy", "blenderproc", "mathutils", "doper.training", "torch")
    for module in HOST_MODULES:
        code = f"import sys, {module}; print([m for m in {banned!r} if m in sys.modules])"
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO_ROOT
        )
        assert out.stdout.strip() == "[]", module
    package = REPO_ROOT / "rendering" / "blenderproc"
    importers = sorted(
        p.name
        for p in package.glob("*.py")
        if any(
            line.startswith(("import bpy", "import blenderproc", "from blenderproc"))
            for line in p.read_text().splitlines()
        )
    )
    assert importers == ["scene.py"]
    first = (package / "scene.py").read_text().splitlines()
    assert next(line for line in first if line.startswith(("import", "from"))) == (
        "import blenderproc as bproc"
    )
    doper_sources = (REPO_ROOT / "doper").rglob("*.py")
    assert not [p for p in doper_sources if "rendering.blenderproc" in p.read_text()]


# --- campaign -----------------------------------------------------------------------------------


def write_models(root: Path, config: plan.RenderConfig) -> Path:
    low, high = config.control_objects
    ids = sorted({*range(low, high + 1), config.target_object})
    root.mkdir(parents=True)
    info = {str(i): {"size_x": 1.0, "size_y": 1.0, "size_z": 1.0} for i in ids}
    (root / "models_info.json").write_text(json.dumps(info))
    for i in ids:
        (root / f"obj_{i:06d}.ply").write_bytes(b"ply")
    return root


def write_assets(root: Path) -> Path:
    for name, folder in [("tless", "models_cad"), ("hb", "models"), ("tyol", "models")]:
        target = root / "bop" / name / folder
        target.mkdir(parents=True)
        (target / "models_info.json").write_text(json.dumps({"1": {}, "2": {}, "3": {}}))
    for name in assets.load_manifest()["materials"]:
        (root / "cctextures" / name).mkdir(parents=True)
        (root / "cctextures" / name / f"{name}_2K-JPG_Color.jpg").write_bytes(b"c")
    assets.write_manifest(root)
    return root


def run_campaign(*args: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "rendering.blenderproc.campaign", *map(str, args)]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)


def test_campaign_dry_run_command_line(tmp_path):
    config = plan.load_config("hot3d_obj33")
    keypoints = write_keypoints(tmp_path / "keypoints_3d.json")
    out = tmp_path / "campaign"
    common = ["--config", "hot3d_obj33", "--assets", write_assets(tmp_path / "assets")]
    common += ["--models", write_models(tmp_path / "models", config), "--keypoints", keypoints]
    common += [
        "--out",
        out,
        "--keypoints-sha256",
        hashlib.sha256(keypoints.read_bytes()).hexdigest(),
    ]
    dry = run_campaign("stage", *common, "--dry-run")
    assert dry.returncode == 0, dry.stderr
    assert not out.exists()
    staged = run_campaign("stage", *common)
    assert staged.returncode == 0, staged.stderr
    identity = json.loads((out / "identity.json").read_text())
    assert (identity["layouts"], identity["workers"], identity["base_seed"]) == (1000, 8, 70000)
    blender = tmp_path / "blender-4.2.1-linux-x64"
    run = run_campaign(
        "run", "--out", out, "--worker", 2, "--blender", blender, "--gpu-index", 1,
        "--scratch", tmp_path / "scratch", "--max-layouts", 2, "--dry-run",
    )  # fmt: skip
    assert run.returncode == 0, run.stderr
    lines = [json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")]
    assert [line["layout"] for line in lines] == [2, 10]
    assert [line["seed"] for line in lines] == [70002, 70010]
    assert [line["split"] for line in lines] == ["train_pbr", "train_pbr"]
    command = lines[0]["command"]
    assert command[:5] == [sys.executable, "-m", "blenderproc", "run", "--custom-blender-path"]
    assert command[5] == str(blender)
    assert command[6] == "--temp-dir"
    assert command[8] == str(REPO_ROOT / "rendering" / "blenderproc" / "scene.py")
    assert command[9:] == [
        "--assets", str(out / "assets"), "--config", str(out / "config.json"),
        "--out", command[14], "--seed", "70002", "--gpu-index", "1",
    ]  # fmt: skip
    assert lines[0]["publish"] == str(out / "train_pbr" / "000002")
    no_blender = run_campaign("run", "--out", out, "--worker", 2, "--max-layouts", 1, "--dry-run")
    assert no_blender.returncode == 0, no_blender.stderr
    assert json.loads(no_blender.stdout)["command"][5] == "<blender>"
    assert not (out / "train_pbr").exists()
    status = run_campaign("status", "--out", out)
    assert status.returncode == 0, status.stderr
    report = json.loads(status.stdout)
    assert report["completed"] == 0 and report["missing"] == 1000
    restage = run_campaign("stage", *common[:-2], "--keypoints-sha256", "0" * 64)
    assert restage.returncode != 0 and "keypoints" in restage.stderr


def test_publish_and_resume_verify_receipts(tmp_path, monkeypatch):
    config = small_config(frames=2)
    out = tmp_path / "campaign"
    out.mkdir()
    keypoints = write_keypoints(tmp_path / "kp.json")
    identity_sha = "f" * 64
    attempt = make_attempt(tmp_path / "attempt", config)
    final = campaign.publish_attempt(
        attempt, out, config, keypoints, layout=0, seed=70000, identity_sha256=identity_sha
    )
    assert final == out / "train_pbr" / "000000"
    receipt = campaign.verify_layout(final, identity_sha, 0, 70000, config)
    assert receipt["split"] == "train_pbr" and receipt["frames"] == 2
    assert (final / "rgb" / "000000.jpg").is_file()
    assert (final / "records" / "scene_record.json").is_file()
    with pytest.raises(FileExistsError):
        campaign.publish_attempt(
            make_attempt(tmp_path / "again", config), out, config, keypoints,
            layout=0, seed=70000, identity_sha256=identity_sha,
        )  # fmt: skip
    (final / "scene_gt.json").write_text("{}")
    with pytest.raises(ValueError, match="scene_gt.json"):
        campaign.verify_layout(final, identity_sha, 0, 70000, config)


FAKE_RENDERER = """
import json, pathlib, sys
sys.path.insert(0, {tests!r})
import bproc_fakes
args = sys.argv[1:]
assert args[:2] == ["-m", "blenderproc"] and args[2] == "run", args
value = lambda name: args[args.index(name) + 1]
config = json.loads(pathlib.Path(value("--config")).read_text())
log = pathlib.Path({log!r})
previous = log.read_text() if log.exists() else ""
log.write_text(previous + value("--seed") + "\\n")
if pathlib.Path({crash!r}).exists():
    pathlib.Path(value("--out")).mkdir(parents=True)
    (pathlib.Path(value("--out")) / "partial.txt").write_text("partial")
    sys.exit(3)
# Seed 70001 stays over the count bound until the "GPU" behaves differently (the fix file).
broken = value("--seed") == "70001" and not pathlib.Path({fixed!r}).exists()
offsets = {{(config["frames_per_layout"] - 1, 0): 9}} if broken else {{}}
bproc_fakes.make_attempt(pathlib.Path(value("--out")), dataset=config["dataset"],
    target=config["target_object"], frames=config["frames_per_layout"],
    visible_offsets=offsets, seed=int(value("--seed")))
"""


def fake_campaign(tmp_path: Path) -> tuple[Path, list, dict[str, Path]]:
    """Stage a 3-layout, 1-worker campaign and a fake BlenderProc interpreter."""
    config = dataclasses.replace(small_config(frames=2), layouts=3, workers=1)
    config_path = tmp_path / "small.json"
    config_path.write_text(json.dumps(config.as_json()))
    keypoints = write_keypoints(tmp_path / "keypoints_3d.json")
    out = tmp_path / "campaign"
    staged = run_campaign(
        "stage", "--config", config_path, "--assets", write_assets(tmp_path / "assets"),
        "--models", write_models(tmp_path / "models", config), "--keypoints", keypoints,
        "--keypoints-sha256", hashlib.sha256(keypoints.read_bytes()).hexdigest(), "--out", out,
    )  # fmt: skip
    assert staged.returncode == 0, staged.stderr
    files = {name: tmp_path / name for name in ("seeds.txt", "fixed", "crash")}
    script = tmp_path / "fake_render.py"
    script.write_text(FAKE_RENDERER.format(tests=str(Path(__file__).parent),
                                           log=str(files["seeds.txt"]),
                                           fixed=str(files["fixed"]),
                                           crash=str(files["crash"])))  # fmt: skip
    python = tmp_path / "fake-python"
    python.write_text(f'#!/bin/sh\nexec {sys.executable} {script} "$@"\n')
    python.chmod(0o755)
    blender = tmp_path / "blender-4.2.1-linux-x64"
    blender.mkdir()
    (blender / "blender").write_text("")
    command = ["run", "--out", out, "--worker", 0, "--blender", blender, "--gpu-index", 0,
               "--blenderproc-python", python, "--scratch", tmp_path / "scratch",
               "--min-free-gib", 0]  # fmt: skip
    return out, command, files


def seeds(files: dict[str, Path]) -> list[str]:
    return files["seeds.txt"].read_text().split() if files["seeds.txt"].exists() else []


def test_over_bound_layout_is_recorded_failed_and_the_rest_publish(tmp_path):
    out, command, files = fake_campaign(tmp_path)
    run = run_campaign(*command)
    assert run.returncode == 1, run.stdout + run.stderr
    assert seeds(files) == ["70000", "70001", "70002"]  # one attempt per layout, no retry
    assert (out / "train_pbr" / "000000").is_dir() and (out / "train_pbr" / "000002").is_dir()
    assert not (out / "train_pbr" / "000001").exists()
    archived = [p.name for p in (out / "failed").iterdir()]
    assert len(archived) == 1 and archived[0].startswith("000001-a1-")
    worker = json.loads((out / "workers" / "000.json").read_text())
    assert worker["status"] == "failed_layouts" and worker["completed"] == [0, 2]
    assert worker["failed"]["1"]["error"] == "CountBoundError"
    assert worker["failed"]["1"]["attempts"] == 1 and worker["failed"]["1"]["seed"] == 70001
    status = json.loads(run_campaign("status", "--out", out, "--deep").stdout)
    assert (status["completed"], status["missing"], status["failed_layouts"]) == (2, 1, [1])
    again = run_campaign(*command)  # a plain re-run skips the recorded failure
    assert again.returncode == 1 and seeds(files) == ["70000", "70001", "70002"]


def test_retry_failed_reattempts_only_recorded_failures(tmp_path):
    out, command, files = fake_campaign(tmp_path)
    assert run_campaign(*command).returncode == 1
    dry = run_campaign(*command, "--retry-failed", "--dry-run")
    assert [json.loads(line)["layout"] for line in dry.stdout.splitlines()] == [1]
    files["fixed"].write_text("")  # emulate GPU nondeterminism changing the outcome
    retry = run_campaign(*command, "--retry-failed")
    assert retry.returncode == 0, retry.stdout + retry.stderr
    assert seeds(files) == ["70000", "70001", "70002", "70001"]
    receipt = json.loads((out / "train_pbr" / "000001" / "COMPLETE.json").read_text())
    assert (receipt["attempt"], receipt["seed"]) == (2, 70001)
    worker = json.loads((out / "workers" / "000.json").read_text())
    assert worker["status"] == "complete" and worker["failed"] == {}
    status = json.loads(run_campaign("status", "--out", out).stdout)
    assert (status["completed"], status["failed_layouts"]) == (3, [])
    assert run_campaign(*command, "--retry-failed").returncode == 0
    assert len(seeds(files)) == 4


def test_low_free_space_refuses_to_render(tmp_path):
    out, command, files = fake_campaign(tmp_path)
    command[command.index("--min-free-gib") + 1] = 10**9
    run = run_campaign(*command)
    assert run.returncode == 2 and run.stderr.startswith("error: less than 1000000000.0 GiB free")
    assert seeds(files) == []


def test_blenderproc_exit_is_a_clean_worker_failure(tmp_path):
    out, command, files = fake_campaign(tmp_path)
    files["crash"].write_text("")
    run = run_campaign(*command)
    assert run.returncode == 2 and run.stderr.startswith("error: layout 0: BlenderProc exited 3")
    assert "Traceback" not in run.stderr
    worker = json.loads((out / "workers" / "000.json").read_text())
    assert (worker["status"], worker["layout"], worker["error"]) == ("failed", 0, "RuntimeError")
    archived = list((out / "failed").iterdir())
    assert len(archived) == 1 and archived[0].name.startswith("000000-a1-")
    assert (archived[0] / "partial.txt").is_file()


def blenderproc_python() -> str | None:
    candidate = os.environ.get("DOPER_BLENDERPROC_PYTHON")
    if candidate:
        return candidate
    return sys.executable if importlib.util.find_spec("blenderproc") else None


@pytest.mark.blenderproc
def test_one_layout_two_views_smoke(tmp_path):
    python = blenderproc_python()
    blender = os.environ.get("DOPER_BLENDER")
    fetched = os.environ.get("DOPER_BLENDERPROC_ASSETS")
    models = os.environ.get("DOPER_BLENDERPROC_MODELS")
    keypoints = os.environ.get("DOPER_BLENDERPROC_KEYPOINTS")
    if not (python and blender and fetched and models and keypoints):
        pytest.skip(
            "needs BlenderProc 2.8.0 (DOPER_BLENDERPROC_PYTHON), DOPER_BLENDER, a GPU and "
            "DOPER_BLENDERPROC_{ASSETS,MODELS,KEYPOINTS}"
        )
    config = dataclasses.replace(
        plan.load_config("handal_obj40"), frames_per_layout=2, samples=8, layouts=1, workers=1
    )
    config_path = tmp_path / "smoke.json"
    config_path.write_text(json.dumps(config.as_json()))
    out = tmp_path / "campaign"
    staged = run_campaign(
        "stage", "--config", config_path, "--assets", fetched, "--models", models,
        "--keypoints", keypoints, "--out", out,
    )  # fmt: skip
    assert staged.returncode == 0, staged.stderr
    run = run_campaign(
        "run", "--out", out, "--worker", 0, "--blender", Path(blender).parent,
        "--gpu-index", 0, "--blenderproc-python", python, "--scratch", tmp_path / "scratch",
        "--min-free-gib", 0,
    )  # fmt: skip
    assert run.returncode == 0, run.stdout + run.stderr
    receipt = json.loads((out / "train_pbr" / "000000" / "COMPLETE.json").read_text())
    assert receipt["frames"] == 2 and receipt["seed"] == 60000


# --- the receipt contract doper reads (doper.bop.generated) ----------------------------------


def _generated_render():
    """``tests/unit/generated_render.py``, the doper tests' campaign fixture, loaded by path."""
    path = REPO_ROOT / "tests" / "unit" / "generated_render.py"
    spec = importlib.util.spec_from_file_location("generated_render", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _doper_spec(config: plan.RenderConfig):
    from doper.recipes.schema import GeneratedSpec

    return GeneratedSpec(
        render_config=config.name,
        layouts=config.layouts,
        frames_per_layout=config.frames_per_layout,
        base_seed=config.base_seed,
        val_modulo=config.val_modulo,
        val_residue=config.val_residue,
        empty_masks="exclude",
    )


def test_published_campaign_passes_the_doper_receipt_check(tmp_path):
    """A campaign this module published is what ``doper prepare --generated-root`` accepts."""
    from doper.bop import generated

    out, command, files = fake_campaign(tmp_path)
    config = plan.config_from_json(json.loads((out / "config.json").read_text()))
    spec = _doper_spec(config)
    kwargs = {"dataset": config.dataset, "obj": config.target_object}
    assert run_campaign(*command).returncode == 1  # layout 1 over the bound: recorded failed
    with pytest.raises(ValueError, match=r"recorded failed layouts \[1\]"):
        generated.source_view(out, spec, tmp_path / "view", **kwargs)
    files["fixed"].write_text("")
    assert run_campaign(*command, "--retry-failed").returncode == 0
    view = generated.source_view(out, spec, tmp_path / "view", **kwargs)
    assert sorted(p.name for p in view.iterdir()) == ["000000", "000001", "000002"]
    verified, receipts = generated.verify_campaign(out, spec, **kwargs)
    for layout, receipt in receipts.items():
        expected = campaign.verify_layout(
            campaign.layout_dir(out, layout),
            verified.identity_sha256,
            layout,
            campaign.layout_seed(config, layout),
            config,
        )
        assert receipt == expected
    assert generated.METADATA_FILES == campaign.METADATA_FILES
    (out / "train_pbr" / "000002" / "records" / "AUDITED.json").write_text("{}")
    with pytest.raises(ValueError, match="000002"):
        generated.verify_campaign(out, spec, **kwargs)


def test_doper_campaign_fixture_passes_the_campaign_status(tmp_path):
    """The fixture the doper tests build is a campaign this module verifies (deep)."""
    fixture = _generated_render()
    config = small_config(frames=2)
    keypoints = write_keypoints(tmp_path / "keypoints_3d.json")
    models = write_models(tmp_path / "models", config)
    split = {"modulo": 2, "val_residue": 1}
    root = fixture.write_render_root(
        tmp_path / "render",
        keypoints=keypoints,
        models=models,
        config=fixture.render_config("hot3d_obj33", layouts=3, frames_per_layout=2, split=split),
        failed=[2],
    )
    status = run_campaign("status", "--out", root, "--deep")
    assert status.returncode == 0, status.stdout + status.stderr
    report = json.loads(status.stdout)
    assert (report["completed"], report["failed_layouts"], report["problems"]) == (2, [2], [])
    assert report["splits"] == {"train_pbr": 1, "val_pbr": 1}
