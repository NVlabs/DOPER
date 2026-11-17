# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SPEED+ fetch, verification and conversion to the 64-point hull layout.

``golden/source_speedplus_convert.json`` holds what the research converter
(its hull mask-mode conversion script) writes for the fixture tree of
``speedplus_raw_data``, in the canonical form of ``speedplus_raw_data.canonical_conversion``:
paths made root-relative and the masks as digests of their decoded pixels.

The last six tests are ported from the research SPEED+ tests; the dataset-load test now loads
the converted split with ``doper.dataset.DoperDataset``, and the keypoint visibility test runs
against ``keypoint_payload``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import speedplus_raw_data as fx
from speedplus_helpers import OFFICIAL_POINTS
from tiny_coco import tiny_model_config

from doper.bop import tless_sensors
from doper.cli import main
from doper.config import DataConfig, DoperConfig
from doper.dataset import DoperDataset
from doper.recipes import load_any_recipe
from doper.speedplus import data, geometry, metrics

GOLDEN = Path(__file__).parent / "golden" / "source_speedplus_convert.json"


@pytest.fixture
def staged(tmp_path: Path) -> tuple[Path, Path]:
    return fx.staged_root(tmp_path)


def _upstream(tmp_path: Path) -> tuple[data.SpeedplusProfile, dict[str, Path]]:
    work = tmp_path / "upstream"
    counts = fx.write_speedplus_tree(work / "speedplusv2")
    archive = fx.zip_tree(work / "speedplusv2", work / "speedplusv2.zip")
    mat = fx.write_tango_points(work / "tangoPoints.mat")
    profile = data.load_profile(fx.fixture_profile(archive, mat, counts, work / "profile.json"))
    return profile, {profile.archive["url"]: archive, profile.tango_points["url"]: mat}


def test_convert_matches_source(staged):
    root, _ = staged
    out = root / data.DERIVED_64
    manifest = data.convert(root, out)
    assert fx.canonical_conversion(out, root) == json.loads(GOLDEN.read_text())
    assert manifest.paper_masks and manifest.mask_sources == {"projected_hull": 6}
    # Images are linked, not copied; no absolute path is written.
    assert (out / "train" / "images").is_symlink()
    assert not Path((out / "train" / "images").readlink()).is_absolute()
    assert str(root) not in (out / "conversion_manifest.json").read_text()
    camera = geometry.load_camera(out / "camera_intrinsics.json")
    assert (camera.width, camera.height) == (fx.WIDTH, fx.HEIGHT)


def test_convert_is_idempotent_and_needs_force_to_change(staged):
    root, _ = staged
    out = root / data.DERIVED_64
    first = data.convert(root, out, splits=("train",))
    assert data.convert(root, out, splits=("train",)) == first
    with pytest.raises(FileExistsError, match="--force"):
        data.convert(root, out, splits=("train", "val"))
    rebuilt = data.convert(root, out, splits=("train", "val"), force=True)
    assert [s["split"] for s in rebuilt.splits] == ["train", "val"]


def test_convert_needs_the_fetch_receipt(tmp_path):
    with pytest.raises(FileNotFoundError, match="READY.json"):
        data.convert(tmp_path, tmp_path / "out")


def test_fetch_extracts_and_records_the_license(tmp_path):
    profile, sources = _upstream(tmp_path)
    calls: list[str] = []
    root = tmp_path / "root"
    receipt = data.fetch(
        root, consent=True, downloader=fx.copying_downloader(sources, calls), profile=profile
    )
    ready = json.loads((root / "READY.json").read_text())
    assert ready["license"]["name"] == "CC BY-NC-SA 4.0"
    assert ready["image_counts"] == {"train": 2, "val": 1, "lightbox": 2, "sunlamp": 1}
    assert (root / data.SPEEDPLUS_DIR / "camera.json").is_file()
    assert receipt.speedplus_dir == root / data.SPEEDPLUS_DIR
    assert len(calls) == 2
    # A second fetch re-verifies the kept files and downloads nothing.
    again = data.fetch(
        root, consent=True, downloader=fx.copying_downloader(sources, calls), profile=profile
    )
    assert again == receipt and len(calls) == 2


def test_fetch_needs_consent(tmp_path):
    profile, sources = _upstream(tmp_path)
    with pytest.raises(PermissionError, match="CC BY-NC-SA 4.0"):
        data.fetch(tmp_path / "root", consent=False, downloader=fx.copying_downloader(sources))
    assert not (tmp_path / "root").exists()


def test_fetch_refuses_md5_mismatch(tmp_path):
    profile, sources = _upstream(tmp_path)
    archive = sources[profile.archive["url"]]
    data_bytes = bytearray(archive.read_bytes())
    data_bytes[-30] ^= 0xFF  # same size, other bytes
    archive.write_bytes(bytes(data_bytes))
    root = tmp_path / "root"
    with pytest.raises(ValueError, match="MD5"):
        data.fetch(root, consent=True, downloader=fx.copying_downloader(sources), profile=profile)
    assert not (root / "READY.json").exists()
    assert not (root / data.SPEEDPLUS_DIR).exists()


def test_fetch_refuses_wrong_size_and_points_checksum(tmp_path):
    profile, sources = _upstream(tmp_path)
    archive = sources[profile.archive["url"]]
    archive.write_bytes(archive.read_bytes() + b"x")
    with pytest.raises(ValueError, match="bytes"):
        data.fetch(
            tmp_path / "a", consent=True, downloader=fx.copying_downloader(sources),
            profile=profile,
        )  # fmt: skip
    profile, sources = _upstream(tmp_path / "b")
    sources[profile.tango_points["url"]].write_bytes(b"not the points")
    with pytest.raises(ValueError, match="SHA-256"):
        data.fetch(
            tmp_path / "b" / "root", consent=True, downloader=fx.copying_downloader(sources),
            profile=profile,
        )  # fmt: skip


def test_fetch_records_observed_sha256(tmp_path):
    profile, sources = _upstream(tmp_path)
    archive = sources[profile.archive["url"]]
    assert profile.archive["sha256"] is None
    root = tmp_path / "root"
    receipt = data.fetch(
        root, consent=True, downloader=fx.copying_downloader(sources), profile=profile
    )
    expected = hashlib.sha256(archive.read_bytes()).hexdigest()
    ready = json.loads((root / "READY.json").read_text())
    assert receipt.archive_sha256 == expected and receipt.checksum_origin == "observed"
    assert ready["archive"]["sha256"] == expected
    assert ready["archive"]["checksum_origin"] == "observed"
    assert ready["archive"]["md5"] == profile.archive["md5"]
    # A later fetch fails when the kept archive no longer has the recorded SHA-256 (its MD5
    # pin is lifted here to reach the trust-on-first-use check).
    kept = root / data.ARCHIVE_PATH
    changed = bytearray(kept.read_bytes())
    changed[-30] ^= 0xFF
    kept.write_bytes(bytes(changed))
    lifted = data.SpeedplusProfile(
        **{**profile.__dict__, "archive": {**profile.archive, "md5": None}}
    )
    with pytest.raises(ValueError, match="recorded"):
        data.fetch(root, consent=True, downloader=fx.copying_downloader(sources), profile=lifted)


def test_fetch_refuses_unsafe_archive_members(tmp_path):
    profile, sources = _upstream(tmp_path)
    archive = sources[profile.archive["url"]]
    with zipfile.ZipFile(archive, "a") as bundle:
        bundle.writestr("speedplusv2/../escape.txt", "x")
    profile = data.load_profile(
        fx.fixture_profile(
            archive, sources[profile.tango_points["url"]], dict(profile.image_counts),
            tmp_path / "unsafe.json",
        )
    )  # fmt: skip
    with pytest.raises(ValueError, match="unsafe"):
        data.fetch(
            tmp_path / "root", consent=True, downloader=fx.copying_downloader(sources),
            profile=profile,
        )  # fmt: skip
    assert not (tmp_path / "escape.txt").exists()


def test_validate_image_counts(tmp_path):
    packaged = data.load_profile()
    assert dict(packaged.image_counts) == {
        "train": 47966, "val": 11994, "lightbox": 6740, "sunlamp": 2791,
    }  # fmt: skip
    assert packaged.image_size == (1920, 1200)
    speedplus_dir = tmp_path / "speedplusv2"
    counts = fx.write_speedplus_tree(speedplus_dir)
    # The fixture profile lowers the pinned counts to the fixture's.
    assert data.validate_image_counts(speedplus_dir, counts) == counts
    with pytest.raises(ValueError, match="train has 2 images; expected 47966"):
        data.validate_image_counts(speedplus_dir, packaged.image_counts)
    (speedplus_dir / "sunlamp" / "images" / "img000006.jpg").unlink()
    with pytest.raises(ValueError, match="sunlamp"):
        data.validate_image_counts(speedplus_dir, counts)


def _tree(path: Path) -> list[str]:
    return sorted(str(p.relative_to(path)) for p in path.rglob("*"))


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a dry run must not download")

    monkeypatch.setattr(tless_sensors, "http_download", refuse)
    root = tmp_path / "root"
    root.mkdir()
    for stage in ("fetch", "convert"):
        argv = ["prepare", "speedplus", "--root", str(root), "--stage", stage, "--dry-run"]
        assert main([*argv, "--accept-license"] if stage == "fetch" else argv) == 0
    assert _tree(root) == []
    # The SPEED+ dry run stays light: no torch.
    code = (
        "import sys\n"
        "from doper.cli import main\n"
        f"argv = ['prepare', 'speedplus', '--root', {str(root)!r}, '--stage', 'convert']\n"
        "assert main([*argv, '--dry-run']) == 0\n"
        "assert 'torch' not in sys.modules\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert _tree(root) == []


def test_prepare_speedplus_cli_converts_and_refuses_bop_flags(staged, capsys):
    root, _ = staged
    assert main(["prepare", "speedplus", "--root", str(root), "--stage", "convert"]) == 0
    manifest = json.loads((root / data.DERIVED_64 / "conversion_manifest.json").read_text())
    assert manifest["mask_mode"] == "hull" and manifest["paper_masks"] is True
    capsys.readouterr()
    refused = {
        "takes ['--multisensor'] only for BOP recipes": ["--stage", "convert", "--multisensor"],
        "needs --stage": [],
        "--accept-license": ["--stage", "fetch"],
    }
    for message, extra in refused.items():
        assert main(["prepare", "speedplus", "--root", str(root), *extra]) == 1
        assert message in capsys.readouterr().out
    assert main(["prepare", "tless", "--root", str(root), "--stage", "fetch", "--dry-run"]) == 1
    assert "takes ['--stage'] only for SPEED+ recipes" in capsys.readouterr().out


def _install_official_masks(root: Path) -> None:
    speedplus_dir = root / data.SPEEDPLUS_DIR
    masks = speedplus_dir / "synthetic" / "masks"
    masks.mkdir()
    for name in ("img000001", "img000002"):
        mask = np.zeros((fx.HEIGHT, fx.WIDTH), dtype=np.uint8)
        mask[40:80, 60:120] = 255
        assert cv2.imwrite(str(masks / f"{name}.png"), mask)


def test_paper_recipe_uses_projected_hull_masks(staged):
    root, _ = staged
    recipe = load_any_recipe("speedplus")
    assert recipe.mask_mode == "hull"
    _install_official_masks(root)
    # The paper mode ignores installed official masks.
    hull = data.convert(root, root / "hull", mask_mode=recipe.mask_mode, splits=("train",))
    assert hull.paper_masks and hull.mask_sources == {"projected_hull": 2}
    # ``auto`` picks them up silently, so it is a non-paper conversion that must be accepted.
    with pytest.raises(ValueError, match="--accept-non-paper-masks"):
        data.convert(root, root / "auto", mask_mode="auto", splits=("train",))
    assert not (root / "auto").exists()
    auto = data.convert(
        root, root / "auto", mask_mode="auto", splits=("train",), accept_non_paper_masks=True
    )
    assert not auto.paper_masks and auto.mask_sources == {"official": 2}
    recorded = json.loads((root / "auto" / "conversion_manifest.json").read_text())
    assert recorded["paper_masks"] is False and recorded["mask_sources"] == {"official": 2}
    # Without official masks ``auto`` falls back to projected hulls: a paper conversion.
    assert data.convert(root, root / "auto2", mask_mode="auto", splits=("val",)).paper_masks
    with pytest.raises(ValueError, match="--accept-non-paper-masks"):
        data.convert(root, root / "none", mask_mode="none")
    boxes = data.convert(
        root, root / "none", mask_mode="none", splits=("val",), accept_non_paper_masks=True
    )
    assert boxes.mask_sources == {"none": 1} and not boxes.paper_masks
    coco = json.loads((root / "none" / "val" / "annotations.json").read_text())
    assert "mask_file" not in coco["annotations"][0] and coco["annotations"][0]["area"] > 0
    argv = ["prepare", "speedplus", "--root", str(root), "--stage", "convert", "--dry-run"]
    assert main([*argv, "--mask-mode", "official"]) == 1
    assert main([*argv, "--mask-mode", "official", "--accept-non-paper-masks"]) == 0


# --- Ported research SPEED+ tests -------------------------------------------------------------


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = geometry.quaternion_to_dcm_scalar_first(first).T @ (
        geometry.quaternion_to_dcm_scalar_first(second)
    )
    return float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))))


def test_expand_tango_keypoints_retains_official_points():
    expanded = geometry.expand_tango_keypoints_64(OFFICIAL_POINTS)
    assert expanded.shape == (64, 3)
    np.testing.assert_array_equal(expanded[:11], OFFICIAL_POINTS)
    assert np.unique(np.round(expanded, 10), axis=0).shape[0] == 64


def test_speedplus_opencv_pose_round_trip():
    quaternion = np.asarray([0.72, -0.21, 0.51, 0.42], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    translation = np.asarray([0.13, -0.08, 5.4])
    rvec, tvec = geometry.speedplus_pose_to_opencv(quaternion, translation)
    recovered_q, recovered_t = geometry.opencv_pose_to_speedplus(rvec, tvec)
    assert _rotation_error_deg(recovered_q, quaternion) < 1e-6
    np.testing.assert_allclose(recovered_t, translation, atol=1e-12)


def test_perfect_projected_keypoints_recover_pose_with_pnp():
    points = geometry.expand_tango_keypoints_64(OFFICIAL_POINTS)
    quaternion = np.asarray([0.82, 0.21, -0.31, 0.43], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    translation = np.asarray([0.18, -0.11, 6.25])
    camera = geometry.Camera(
        K=np.asarray([[1200.0, 0.0, 960.0], [0.0, 1200.0, 600.0], [0.0, 0.0, 1.0]]),
        dist=np.zeros(5), width=1920, height=1200,
    )  # fmt: skip
    pixels, depth = geometry.project_keypoints(points, quaternion, translation, camera)
    assert np.all(depth > 0)
    ok, rvec, tvec = cv2.solvePnP(points, pixels, camera.K, camera.dist, flags=cv2.SOLVEPNP_EPNP)
    assert ok
    recovered_q, recovered_t = geometry.opencv_pose_to_speedplus(rvec, tvec)
    assert _rotation_error_deg(recovered_q, quaternion) < 1e-4
    assert np.linalg.norm(recovered_t - translation) < 1e-6


def test_speedplus_metrics_are_sign_invariant_and_thresholded():
    target_q = np.asarray([1.0, 0.0, 0.0, 0.0])
    target_t = np.asarray([0.0, 0.0, 10.0])
    assert _rotation_error_deg(-target_q, target_q) == 0.0
    within_rotation = np.radians(metrics.ROTATION_DEADBAND_DEG * 0.5)
    predicted_q = np.asarray([np.cos(within_rotation / 2), np.sin(within_rotation / 2), 0.0, 0.0])
    relative = metrics.TRANSLATION_DEADBAND_RELATIVE * 0.5
    predicted_t = target_t + np.asarray([relative * np.linalg.norm(target_t), 0, 0])
    e_r = np.radians(_rotation_error_deg(predicted_q, target_q))
    e_t = float(np.linalg.norm(predicted_t - target_t) / np.linalg.norm(target_t))
    assert metrics.pose_score(e_r, e_t, definition="paper") > 0
    assert metrics.pose_score(e_r, e_t, definition="official") == 0.0


def test_keypoint_payload_flags_invalid_and_outside_points():
    pixels = np.asarray(
        [[10.5, 20.25], [-1.0, 5.0], [30.0, 40.0], [np.nan, 12.0], [99.0, 49.0]],
        dtype=np.float64,
    )
    depth = np.asarray([4.0, 4.0, -1.0, 4.0, 2.0], dtype=np.float64)
    encoded, count = data.keypoint_payload(pixels, depth, width=100, height=50)
    keypoints = np.asarray(encoded, dtype=np.float64).reshape(-1, 3)
    assert count == 2
    np.testing.assert_array_equal(keypoints[:, 2], [2, 0, 0, 0, 2])
    # Coordinates are kept as projected, also for the points flagged invisible.
    np.testing.assert_array_equal(keypoints[:, :2], pixels)


def test_converter_writes_a_doper_loadable_split(staged):
    root, _ = staged
    out = root / data.DERIVED_64
    manifest = data.convert(root, out, splits=("train",))
    assert manifest.splits[0]["count"] == 2
    payload = json.loads((out / "train" / "annotations.json").read_text())
    assert len(payload["annotations"][0]["keypoints"]) == 64 * 3
    assert payload["annotations"][0]["mask_file"] == "masks/img000001.png"
    cfg = DoperConfig(
        data=DataConfig(root=str(out)),
        model=tiny_model_config(
            num_keypoints=64, det_input_size=64, crop_size=32, keypoint_head="heatmap"
        ),
    )
    cfg.aug.enabled = False
    sample = DoperDataset(cfg, "train")[0]
    assert sample["det_image"].shape == (3, 64, 64)
    assert sample["crop_image"].shape == (3, 32, 32)
    assert float(sample["mask_weight"]) == 1.0
    assert float(sample["has_kp"]) == 1.0
