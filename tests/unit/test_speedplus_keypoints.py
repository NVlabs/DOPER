# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The 75-point SPEED+ layout and the tracked-landmark extractor.

``golden/source_speedplus_layout.json`` and ``golden/source_speedplus_tracked_landmarks.json``
hold what the research scripts (``prepare_speedplus_custom_keypoints.py --splits train val``
and ``extract_speedplus_trackable_keypoints.py``) write for the fixtures of
``speedplus_keypoint_data``, in its canonical forms.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import speedplus_keypoint_data as kd
import speedplus_raw_data as fx

from doper.cli import main
from doper.recipes import load_any_recipe
from doper.speedplus import data, keypoints
from doper.speedplus.checkpoints import (
    LAYOUT_POINTS_SHA_KEY,
    LAYOUT_SHA_KEY,
    check_checkpoint_layout,
)
from doper.speedplus.definitions import (
    PAPER_LAYOUT,
    PAPER_LAYOUT_POINTS_SHA256,
    PAPER_LAYOUT_SHA256,
    LayoutPins,
)

GOLDEN = Path(__file__).parent / "golden"
SOURCE_KEY_ORDER = [
    "object_name",
    "keypoint_layout",
    "keypoint_layout_sha256",
    "official_keypoint_count",
    "tracked_keypoint_count",
    "keypoints_3d",
]


@pytest.fixture
def converted(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A fetched and converted fixture root, its 64-point layout and a tracked-64 file."""
    root, _ = fx.staged_root(tmp_path)
    tango_64 = root / data.DERIVED_64
    data.convert(root, tango_64)
    tracked = kd.write_tracked_keypoints(tmp_path / "tracked" / "keypoints_3d.json")
    return root, tango_64, tracked


def _fixture_pins(tango_64: Path, tracked: Path) -> LayoutPins:
    """Pins of the fixture's layout, as the paper's are pinned for the real one."""
    official = np.asarray(json.loads((tango_64 / "keypoints_3d.json").read_text())["keypoints_3d"])
    points = np.vstack([official[:11], kd.tracked_points()])
    return LayoutPins(sha256="0" * 64, points_sha256=keypoints.layout_points_sha256(points))


def test_layout_matches_source(converted):
    root, tango_64, tracked = converted
    out = root / keypoints.DERIVED_75
    keypoints.build_layout(tango_64, out, tracked, accept_new_layout=True)
    golden = json.loads((GOLDEN / "source_speedplus_layout.json").read_text())
    assert kd.canonical_layout(out, root) == golden
    # Images and masks are relative links; no absolute path is written.
    assert not Path((out / "train" / "images").readlink()).is_absolute()
    for name in ("conversion_manifest.json", "train/annotations.json"):
        assert str(root) not in (out / name).read_text()


def test_layout_sha_written_and_checked(converted, tmp_path):
    root, tango_64, tracked = converted
    out = root / keypoints.DERIVED_75
    manifest = keypoints.build_layout(tango_64, out, tracked, accept_new_layout=True)
    layout_file = out / "keypoints_3d.json"
    digest = hashlib.sha256(layout_file.read_bytes()).hexdigest()
    assert manifest.layout_sha256 == keypoints.layout_sha256(layout_file) == digest
    recorded = json.loads(manifest.manifest.read_text())
    assert recorded["layout_sha256"] == digest
    # A checkpoint trained on this layout is refused with another one, naming both SHA-256s.
    moved = kd.tracked_points() + 0.01
    other_tracked = kd.write_tracked_keypoints(tmp_path / "other" / "keypoints_3d.json", moved)
    other = keypoints.build_layout(
        tango_64, root / "other75", other_tracked, accept_new_layout=True
    )
    checkpoint = {
        LAYOUT_SHA_KEY: manifest.layout_sha256, LAYOUT_POINTS_SHA_KEY: manifest.points_sha256
    }  # fmt: skip
    built, moved_layout = (LayoutPins(m.layout_sha256, m.points_sha256) for m in (manifest, other))
    assert check_checkpoint_layout(checkpoint, "c" * 64, built, {}).check == "recorded"
    with pytest.raises(ValueError) as refused:
        check_checkpoint_layout(checkpoint, "c" * 64, moved_layout, {})
    assert manifest.points_sha256 in str(refused.value)
    assert other.points_sha256 in str(refused.value)
    # The finished layout is returned as it is, and re-checked: an edited file is refused.
    assert keypoints.build_layout(tango_64, out, tracked, accept_new_layout=True) == manifest
    layout_file.write_text(layout_file.read_text().replace("\n", "\n ", 1))
    with pytest.raises(ValueError) as edited:
        keypoints.build_layout(tango_64, out, tracked, accept_new_layout=True)
    assert digest in str(edited.value)
    assert keypoints.layout_sha256(layout_file) in str(edited.value)


def test_points_hash_matches_source_formula():
    points = np.vstack([kd.tracked_points()[:11], kd.tracked_points()])
    expected = hashlib.sha256(np.asarray(points, dtype="<f8").tobytes()).hexdigest()
    assert keypoints.layout_points_sha256(points) == expected
    # The hash is of the little-endian float64 values, whatever the input's dtype or nesting.
    assert keypoints.layout_points_sha256(points.astype(">f8")) == expected
    assert keypoints.layout_points_sha256(points.tolist()) == expected
    assert keypoints.layout_points_sha256(points.astype(np.float32)) != expected
    # The research builder's hash of the fixture layout.
    golden = json.loads((GOLDEN / "source_speedplus_layout.json").read_text())
    source_points = np.asarray(json.loads(golden["keypoints_3d"])["keypoints_3d"])
    assert (
        keypoints.layout_points_sha256(source_points)
        == (golden["manifest"]["keypoint_layout_sha256"])
    )


def test_written_layout_embeds_points_hash(converted):
    root, tango_64, tracked = converted
    out = root / keypoints.DERIVED_75
    manifest = keypoints.build_layout(tango_64, out, tracked, accept_new_layout=True)
    text = (out / "keypoints_3d.json").read_text()
    payload = json.loads(text)
    assert list(payload) == SOURCE_KEY_ORDER
    assert payload["keypoint_layout"] == "speedplus_official_11_plus_tracked_64"
    assert (payload["official_keypoint_count"], payload["tracked_keypoint_count"]) == (11, 64)
    assert text == json.dumps(payload, indent=2) + "\n"
    points_hash = keypoints.layout_points_sha256(np.asarray(payload["keypoints_3d"]))
    assert payload["keypoint_layout_sha256"] == points_hash == manifest.points_sha256
    recorded = json.loads(manifest.manifest.read_text())
    assert recorded["keypoint_layout_sha256"] == points_hash
    coco = json.loads((out / "train" / "annotations.json").read_text())
    assert coco["info"]["keypoint_layout_sha256"] == points_hash
    names = coco["categories"][0]["keypoints"]
    assert names[:2] == ["official_00", "official_01"] and names[11] == "tracked_00"
    assert names[-1] == "tracked_63" and len(names) == 75


def test_layout_defaults_to_train_val(converted):
    root, tango_64, tracked = converted
    out = root / keypoints.DERIVED_75
    manifest = keypoints.build_layout(tango_64, out, tracked, accept_new_layout=True)
    assert manifest.splits_requested == ("train", "val")
    assert [result["split"] for result in manifest.splits] == ["train", "val"]
    assert sorted(p.name for p in out.iterdir() if p.is_dir()) == ["train", "val"]
    assert json.loads(manifest.manifest.read_text())["splits_requested"] == ["train", "val"]
    # Another split set is another layout build: it needs force.
    with pytest.raises(FileExistsError, match="--force"):
        keypoints.build_layout(tango_64, out, tracked, splits=("train",), accept_new_layout=True)
    rebuilt = keypoints.build_layout(
        tango_64, out, tracked, splits=("sunlamp",), accept_new_layout=True, force=True
    )
    assert rebuilt.splits_requested == ("sunlamp",)
    with pytest.raises(ValueError, match="splits"):
        keypoints.build_layout(tango_64, root / "x", tracked, splits=("test",))


def test_non_paper_layout_needs_flag_and_is_marked(converted):
    root, tango_64, tracked = converted
    out = root / keypoints.DERIVED_75
    points_hash = _fixture_pins(tango_64, tracked).points_sha256
    with pytest.raises(ValueError, match="--accept-new-layout") as refused:
        keypoints.build_layout(tango_64, out, tracked)
    assert points_hash in str(refused.value) and PAPER_LAYOUT_POINTS_SHA256 in str(refused.value)
    assert not out.exists()
    new = keypoints.build_layout(tango_64, out, tracked, accept_new_layout=True)
    assert new.paper_layout is False
    recorded = json.loads(new.manifest.read_text())
    assert recorded["paper_layout"] is False
    assert recorded["paper_points_sha256"] == PAPER_LAYOUT_POINTS_SHA256
    assert recorded["paper_layout_sha256"] == PAPER_LAYOUT_SHA256
    # A tracked file that gives the pinned points is the paper layout, without the flag. A file
    # SHA-256 that differs from the pin is recorded as a formatting deviation, not refused.
    pins = _fixture_pins(tango_64, tracked)
    paper = keypoints.build_layout(tango_64, root / "paper75", tracked, pins=pins)
    assert paper.paper_layout is True and paper.layout_file_matches_paper is False
    recorded = json.loads(paper.manifest.read_text())
    assert recorded["paper_layout"] is True and recorded["layout_file_matches_paper"] is False
    assert pins.sha256 in recorded["formatting_deviation"]
    exact = LayoutPins(sha256=paper.layout_sha256, points_sha256=pins.points_sha256)
    # Other pins are another build: they need force, which recomputes the decision.
    with pytest.raises(FileExistsError, match="--force"):
        keypoints.build_layout(tango_64, root / "paper75", tracked, pins=exact)
    rebuilt = keypoints.build_layout(tango_64, root / "paper75", tracked, pins=exact, force=True)
    assert rebuilt.paper_layout and rebuilt.layout_file_matches_paper
    same = keypoints.build_layout(tango_64, root / "exact75", tracked, pins=exact)
    assert same.paper_layout and same.layout_file_matches_paper
    assert "formatting_deviation" not in json.loads(same.manifest.read_text())


def test_recipe_pins_the_paper_layout():
    recipe = load_any_recipe("speedplus")
    assert recipe.layout == PAPER_LAYOUT
    assert (
        recipe.layout.sha256
        == PAPER_LAYOUT_SHA256
        == recipe.legacy_checkpoints[
            "0824734e2d596adc0a5c45d8de034b67a11c2669926919e7105916974e4c4bba"
        ]
    )
    assert recipe.layout.points_sha256 == (
        "4c78a38de43fbe96ed2639749d95eafaa6a5db93f1de68f8ebdb3fae97c8ef0e"
    )
    with pytest.raises(ValueError, match="points_sha256"):
        LayoutPins(sha256=PAPER_LAYOUT_SHA256, points_sha256="nope")


def _extract(scene: Path, out: Path, **settings) -> Path:
    return keypoints.extract_tracked_landmarks(
        scene / "speedplusv2", scene / "tango_64" / "train" / "masks", out, **settings
    )


def _source_golden_or_skip() -> dict:
    """The extractor golden, without its versions; skip unless this env has the same builds.

    SIFT, the JPEG encoder and DBSCAN make the golden specific to the OpenCV and scikit-learn
    builds it was produced with. The parity case ``speedplus_tracked_landmarks`` compares the
    research extractor and the port bit-exact in any one environment.
    """
    golden = json.loads((GOLDEN / "source_speedplus_tracked_landmarks.json").read_text())
    recorded, current = golden.pop("versions"), kd.library_versions()
    if recorded != current:
        pytest.skip(
            f"the extractor golden was produced with {recorded}, this env has {current}; "
            f"{kd.REGENERATE}"
        )
    return golden


def test_extractor_golden_skips_on_other_builds(monkeypatch):
    monkeypatch.setattr(kd, "library_versions", lambda: {"opencv": "0", "scikit-learn": "0"})
    with pytest.raises(pytest.skip.Exception, match="research scripts") as skipped:
        _source_golden_or_skip()
    assert "'opencv': '0'" in str(skipped.value) and "speedplus_keypoint_data.py" in str(
        skipped.value
    )


@pytest.mark.source_golden
def test_extractor_matches_source_on_fixture(tmp_path):
    pytest.importorskip("sklearn")
    scene = tmp_path / "scene"
    kd.write_landmark_scene(scene / "speedplusv2", scene / "tango_64")
    out = tmp_path / "landmarks"
    written = _extract(scene, out)
    # Build-independent: the outputs, 64 finite tracked points, and determinism.
    assert written == out / "keypoints_3d.json"
    actual = kd.canonical_landmarks(out)
    payload = actual["keypoints_3d"]
    points = np.asarray(payload["keypoints_3d"])
    assert points.shape == (64, 3) and np.isfinite(points).all()
    assert payload["keypoint_layout"] == "mesh_free_sift_multiview_tracks_v1"
    assert payload["keypoint_layout_sha256"] == keypoints.layout_points_sha256(points)
    assert payload["extraction"]["seed"] == 20260826
    assert actual["report"]["selected_keypoints"] == 64
    assert len(actual["overlays"]) == 12
    assert (out / "keypoints_3d_scatter.png").is_file()
    assert str(tmp_path) not in (out / "report.json").read_text()
    _extract(scene, tmp_path / "again")
    assert kd.canonical_landmarks(tmp_path / "again") == actual
    with pytest.raises(FileExistsError, match="force"):
        _extract(scene, out)
    with pytest.raises(TypeError, match="unknown"):
        _extract(scene, tmp_path / "x", anchor_count=3)
    # Bit-exact against the research extractor, on the builds the golden was produced with.
    assert actual == _source_golden_or_skip()


def test_extractor_needs_speedplus_extra(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "sklearn", None)
    monkeypatch.setitem(sys.modules, "sklearn.cluster", None)
    with pytest.raises(ImportError, match=r"doper\[speedplus\]"):
        keypoints.extract_tracked_landmarks(tmp_path, tmp_path / "masks", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_extractor_skips_the_scatter_without_matplotlib(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.figure", None)
    scene = tmp_path / "scene"
    kd.write_landmark_scene(scene / "speedplusv2", scene / "tango_64")
    out = tmp_path / "landmarks"
    keypoints.extract_tracked_landmarks(
        scene / "speedplusv2", scene / "tango_64" / "train" / "masks", out
    )
    assert not (out / "keypoints_3d_scatter.png").exists()
    assert (out / "projection_contact_sheet.jpg").is_file()


def _tree(path: Path) -> list[str]:
    return sorted(str(p.relative_to(path)) for p in path.rglob("*"))


def test_prepare_layout_and_landmark_stages(converted, tmp_path, capsys):
    root, tango_64, tracked = converted
    base = ["prepare", "speedplus", "--root", str(root)]
    before = _tree(root)
    for stage in (["--stage", "layout", "--tracked-keypoints", str(tracked)],
                  ["--stage", "extract-landmarks"]):  # fmt: skip
        assert main([*base, *stage, "--dry-run"]) == 0
    assert _tree(root) == before
    # The dry runs stay light: no torch and no scikit-learn.
    code = (
        "import sys\n"
        "from doper.cli import main\n"
        f"base = ['prepare', 'speedplus', '--root', {str(root)!r}, '--dry-run']\n"
        f"assert main([*base, '--stage', 'layout', '--tracked-keypoints', {str(tracked)!r}]) == 0\n"
        "assert main([*base, '--stage', 'extract-landmarks']) == 0\n"
        "assert 'torch' not in sys.modules and 'sklearn' not in sys.modules\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    capsys.readouterr()
    assert main([*base, "--stage", "layout"]) == 1
    assert "--tracked-keypoints" in capsys.readouterr().out
    layout = [*base, "--stage", "layout", "--tracked-keypoints", str(tracked)]
    assert main(layout) == 1
    assert "--accept-new-layout" in capsys.readouterr().out
    assert main([*layout, "--accept-new-layout"]) == 0
    manifest = json.loads((root / keypoints.DERIVED_75 / "conversion_manifest.json").read_text())
    assert manifest["paper_layout"] is False and manifest["splits_requested"] == ["train", "val"]
    # The landmark stage runs on the fetched SPEED+ tree and the conversion's masks.
    pytest.importorskip("sklearn")
    speedplus_dir = root / data.SPEEDPLUS_DIR
    shutil.rmtree(speedplus_dir / "synthetic")
    shutil.rmtree(tango_64 / "train" / "masks")
    shutil.rmtree(tango_64 / "val" / "masks")
    kd.write_landmark_scene(speedplus_dir, tango_64)
    assert main([*base, "--stage", "extract-landmarks"]) == 0
    extracted = root / keypoints.TRACKED_DIR / "keypoints_3d.json"
    assert len(json.loads(extracted.read_text())["keypoints_3d"]) == 64
    assert main([*base, "--stage", "extract-landmarks"]) == 1
    assert "--force" in capsys.readouterr().out
    # Extracted landmarks are not the paper's layout: the layout stage needs the flag for them.
    capsys.readouterr()
    regenerated = [*base, "--stage", "layout", "--tracked-keypoints", str(extracted), "--force"]
    assert main(regenerated) == 1
    assert "--accept-new-layout" in capsys.readouterr().out
    assert main([*regenerated, "--accept-new-layout"]) == 0
    manifest = json.loads((root / keypoints.DERIVED_75 / "conversion_manifest.json").read_text())
    assert manifest["paper_layout"] is False
    assert manifest["additional_keypoints_sha256"] == keypoints.layout_sha256(extracted)


def test_tracked_keypoints_are_checked(converted, tmp_path):
    root, tango_64, _ = converted
    bad = {
        "expected 64": kd.tracked_points()[:63],
        "duplicates": np.vstack([kd.tracked_points()[:63], kd.tracked_points()[:1]]),
        "non-finite": np.where(np.arange(192).reshape(64, 3) == 5, np.nan, kd.tracked_points()),
    }
    for message, points in bad.items():
        path = tmp_path / "bad" / f"{len(message)}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"keypoints_3d": points.tolist()}))
        with pytest.raises(ValueError, match=message):
            keypoints.build_layout(tango_64, root / "bad75", path, accept_new_layout=True)
    flat = tmp_path / "flat.json"
    flat.write_text(json.dumps([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="Nx3"):
        keypoints.load_tracked_keypoints(flat)
    # A split the 64-point conversion lacks is named with the stage that converts it.
    shutil.rmtree(tango_64 / "val")
    with pytest.raises(FileNotFoundError, match="--stage convert"):
        keypoints.build_layout(
            tango_64,
            root / "bad75",
            kd.write_tracked_keypoints(tmp_path / "ok.json"),
            accept_new_layout=True,
        )
