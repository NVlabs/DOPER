# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The four BOP test variants: raw-detection journal, composition, fallbacks and manifest."""

import dataclasses
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from bank_runs import write_base, write_test_split

from doper import cli
from doper.bop import export, variants
from doper.bop.multiview import FusedPose, ImageKey, MultiviewSpec
from doper.infer import Detection
from doper.recipes.schema import load_recipe
from doper.refinement import PowellAudit, RefineSettings

K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
CROP_SIZE = 256
BINS = 64
KEYPOINTS = 32
TLESS = MultiviewSpec(convention="w2c", support="nearest", scope="test_targets")
REF, SUP = ImageKey(1, 0), ImageKey(1, 1)
WIDTH, HEIGHT = 32, 24


def _object_keypoints():
    return np.random.default_rng(7).uniform(-40.0, 40.0, size=(KEYPOINTS, 3))


def _camera(angle_deg):
    """World-to-camera record of a camera circling the world point (0, 0, 600) about y."""
    rotation = cv2.Rodrigues(np.array([0.0, np.radians(angle_deg), 0.0]))[0]
    center = np.array([0.0, 0.0, 600.0])
    return {
        "cam_K": K.ravel().tolist(),
        "cam_R_w2c": rotation.ravel().tolist(),
        "cam_t_w2c": (center - rotation @ center).tolist(),
    }


def _world_to_camera(camera):
    matrix = np.eye(4)
    matrix[:3, :3] = np.reshape(camera["cam_R_w2c"], (3, 3))
    matrix[:3, 3] = camera["cam_t_w2c"]
    return matrix


OBJECT_POSE = np.eye(4)
OBJECT_POSE[:3, :3] = cv2.Rodrigues(np.array([0.3, -0.2, 0.1]))[0]
OBJECT_POSE[:3, 3] = [5.0, -4.0, 600.0]
CAMERAS = {REF: _camera(0.0), SUP: _camera(20.0)}


def _heatmap(pixels):
    """Crop affine and log-Gaussian logits peaked at ``pixels`` (float32, as inference)."""
    center = pixels.mean(axis=0)
    side = 1.3 * float(np.ptp(pixels, axis=0).max())
    crop_to_image = np.array(
        [[side / (CROP_SIZE - 1), 0.0, center[0] - side / 2],
         [0.0, side / (CROP_SIZE - 1), center[1] - side / 2]]
    )  # fmt: skip
    image_to_crop = cv2.invertAffineTransform(crop_to_image)
    bins = (pixels @ image_to_crop[:, :2].T + image_to_crop[:, 2]) * ((BINS - 1) / (CROP_SIZE - 1))
    grid = np.arange(BINS, dtype=np.float64)
    logits = -((grid[None, None, :] - bins[:, 0, None, None]) ** 2
               + (grid[None, :, None] - bins[:, 1, None, None]) ** 2) / (2 * 1.5**2)  # fmt: skip
    return logits.astype(np.float32), crop_to_image


def _detection(key, *, score=0.9, posed=True, perturb=(0.03, -0.02, 0.02), offset=(4, -3, 8),
               shift=0.0):  # fmt: skip
    """A detection of the object seen from ``key``: true keypoints, perturbed PnP pose."""
    camera_pose = _world_to_camera(CAMERAS[key]) @ OBJECT_POSE
    rvec = cv2.Rodrigues(camera_pose[:3, :3])[0]
    tvec = camera_pose[:3, 3:4].copy()
    points = _object_keypoints()
    pixels = cv2.projectPoints(points, rvec, tvec, K, None)[0].reshape(-1, 2) + shift
    logits, crop_to_image = _heatmap(pixels)
    det = Detection(
        bbox_xyxy=np.array([*pixels.min(axis=0), *pixels.max(axis=0)], dtype=np.float32),
        score=score,
        keypoints=pixels.astype(np.float32),
        keypoint_scores=np.full(KEYPOINTS, 0.9, dtype=np.float32),
        keypoint_heatmap=logits,
        crop_to_image=crop_to_image,
    )
    if posed:
        det.rvec = rvec + np.reshape(perturb, (3, 1))
        det.tvec = tvec - np.reshape(offset, (3, 1))
    return det


def _journal_detection(det, index):
    pose = det.rvec is not None
    return variants.JournalDetection(
        index=index,
        score=float(det.score),
        keypoints=det.keypoints,
        keypoint_scores=det.keypoint_scores,
        rvec=np.asarray(det.rvec, dtype=np.float64) if pose else None,
        tvec=np.asarray(det.tvec, dtype=np.float64) if pose else None,
        heatmap=lambda det=det: (det.keypoint_heatmap, det.crop_to_image),
    )


def _view(key, detections):
    return variants.View(
        camera=CAMERAS[key],
        detections=tuple(_journal_detection(d, j) for j, d in enumerate(detections)),
    )


def _settings(*names, optimizer="powell"):
    return variants.VariantSettings(
        refine=RefineSettings(optimizer=optimizer),
        crop_size=CROP_SIZE,
        beta=1.0,
        variants=names or variants.VARIANTS,
    )


def _row_pose(row):
    return np.asarray(row.R).reshape(3, 3), np.asarray(row.t)


# --- the raw-detection journal ------------------------------------------------------------------


def _test_split(root, frames=2):
    return write_test_split(root, frames, K.ravel().tolist(), (WIDTH, HEIGHT))


def _raw_frames():
    """Frame 0 sees the reference view, frame 1 the support view (a PnP failure at index 1)."""
    behind = _detection(REF, score=0.5)
    behind.tvec = -behind.tvec  # posed, but behind the camera: an invalid pose
    distractor = _detection(SUP, score=0.3, shift=120.0)
    return {
        0: [_detection(REF), behind],
        1: [distractor, _detection(SUP, score=0.2, posed=False)],
    }


TAG = np.float32(0.25)


def _install_bank(monkeypatch, run, by_frame, loads, passes=None):
    """Patch ``export.load_bank``: heatmaps only with ``raw_outputs``; record each load.

    With ``passes`` (load number to ``{frame: score}`` of each frame's first detection), the
    detector of load ``n`` uses those scores and tags its heatmaps with ``n`` (``_tag``), so a
    test can tell which forward pass produced a heatmap.
    """

    def load_bank(bank, **kwargs):
        raw = kwargs.get("raw_outputs", False)
        number = len(loads)
        loads.append(raw)

        def detect(image, camera_matrix):
            frame = (int(image[0, 0, 0]) - 1) // 20
            dets = list(by_frame[frame])
            if passes is not None:
                score = passes.get(number, passes[max(passes)]).get(frame)
                tagged = dets[0].keypoint_heatmap + TAG * number
                dets[0] = dataclasses.replace(
                    dets[0], keypoint_heatmap=tagged, score=score or dets[0].score
                )
            if raw:
                return dets
            return [dataclasses.replace(d, keypoint_heatmap=None, crop_to_image=None) for d in dets]

        return [
            export.BankModel(
                obj,
                detect,
                export.sha256_file(run / "work" / "last.pth"),
                export.sha256_file(run / "geometry" / "keypoints_3d.json"),
                {"config_sha256": "c" * 64, "crop_size": CROP_SIZE, "keypoint_heatmap_beta": 1.0},
            )
            for obj in bank
        ]

    monkeypatch.setattr(export, "load_bank", load_bank)


def _heatmap_files(root):
    return sorted(root.rglob("*.npz")) + sorted(root.rglob(f"*{export.HEATMAP_ENTRY_SUFFIX}"))


@pytest.fixture
def raw_setup(tmp_path, monkeypatch):
    test_root = _test_split(tmp_path / "test")
    run = write_base(tmp_path / "runs", 1)
    by_frame, loads = _raw_frames(), []
    _install_bank(monkeypatch, run, by_frame, loads)
    frames, population = export.load_test_frames(test_root, "pinhole")
    cache = export.JournalCache(tmp_path / "cache")

    def journal(heatmaps):
        return export.cached_raw_journal(
            run, frames, population, cache, camera="pinhole", device="cpu", heatmaps=heatmaps
        )

    return SimpleNamespace(
        test_root=test_root, run=run, by_frame=by_frame, loads=loads, cache=cache,
        journal=journal, tmp_path=tmp_path,
    )  # fmt: skip


def test_raw_journal_keeps_pnp_failures_and_indices(raw_setup):
    out = raw_setup.tmp_path / "out" / "doperbase_tless-test.csv"
    summary = export.run_bank(
        {1: raw_setup.run}, raw_setup.test_root, out, objects=[1],
        expected_population=export.TestPopulation(2, 1), cache=raw_setup.cache, device="cpu",
    )  # fmt: skip
    assert summary.pose_rows == 2 and summary.pnp_failures == 1 and summary.invalid_poses == 1
    assert raw_setup.loads == [False] and _heatmap_files(raw_setup.cache.root) == []
    plain = raw_setup.journal(False)
    assert plain.heatmaps is None and plain.model["crop_size"] == CROP_SIZE
    record = plain.records[1, 1]
    assert [d["index"] for d in record["detections"]] == [0, 1]
    failed = record["detections"][1]
    support_failed = raw_setup.by_frame[1][1]
    assert failed["status"] == "pnp_failed" and failed["rvec"] is None and failed["tvec"] is None
    np.testing.assert_array_equal(
        np.asarray(failed["keypoints"], np.float32), support_failed.keypoints
    )
    assert plain.records[1, 0]["detections"][1]["status"] == "invalid_pose"
    assert "heatmaps" not in record and "heatmaps_sha256" not in record
    assert all(d.heatmap is None for d in variants.journal_views(plain, CAMERAS)[SUP].detections)
    # The heatmap store is filled from a rescan that reproduces the journal exactly.
    journal = raw_setup.journal(True)
    assert raw_setup.loads == [False, True]
    logits, affine = journal.heatmap((1, 1), 1)
    assert logits.dtype == np.float32
    np.testing.assert_array_equal(logits, support_failed.keypoint_heatmap)
    np.testing.assert_array_equal(affine, support_failed.crop_to_image)
    views = variants.journal_views(journal, {REF: CAMERAS[REF], SUP: CAMERAS[SUP]})
    assert views[SUP].detections[1].rvec is None
    audit = []
    variants.compose_variants(
        views, {REF: (REF, SUP)}, _object_keypoints(), _settings("multiview"), TLESS, audit=audit
    )
    assert audit[0]["detection_index"] == 0 and audit[0]["matched"] == [[1, 1, "", 1]]


def test_heatmap_store_is_filled_once_in_the_journal_pass(raw_setup):
    journal = raw_setup.journal(True)
    assert raw_setup.loads == [True]  # one forward pass wrote the journal and the heatmaps
    stores = list(raw_setup.cache.root.rglob(f"*{export.HEATMAP_ENTRY_SUFFIX}"))
    assert len(stores) == 1 and len(list(stores[0].rglob("*.npz"))) == 2
    again = raw_setup.journal(True)
    assert raw_setup.loads == [True] and again.heatmaps == journal.heatmaps
    assert raw_setup.journal(False).records == journal.records
    archive = journal.heatmaps[1, 0][0]
    with np.load(archive) as data:
        assert {data[k].dtype for k in data.files if k.startswith("h")} == {np.dtype(np.float32)}
    archive.write_bytes(archive.read_bytes() + b"x")
    with pytest.raises(ValueError, match="changed"):
        journal.heatmap((1, 0), 0)
    index = stores[0] / export.HEATMAP_INDEX_NAME
    index.write_text(index.read_text().replace('"im_id": 0', '"im_id": 0 '))
    with pytest.raises(ValueError, match="receipt"):
        raw_setup.journal(True)


def _retrain_frame0(raw_setup):
    """The detector now differs on frame 0 (score and heatmap): a non-reproducible rescan."""
    det = raw_setup.by_frame[0][0]
    logits = det.keypoint_heatmap + np.float32(0.5)
    raw_setup.by_frame[0][0] = dataclasses.replace(det, score=0.91, keypoint_heatmap=logits)
    return logits


def _assert_consistent_pair(raw_setup, logits):
    journal = raw_setup.journal(True)
    assert journal.records[1, 0]["detections"][0]["score"] == 0.91
    np.testing.assert_array_equal(journal.heatmap((1, 0), 0)[0], logits)
    assert raw_setup.journal(False).records == journal.records
    assert not list(raw_setup.cache.root.rglob(f"*{export.REBUILD_SUFFIX}"))


def test_heatmap_rescan_that_differs_replaces_journal_and_heatmaps(raw_setup, capsys):
    old = raw_setup.journal(False)
    assert old.records[1, 0]["detections"][0]["score"] == 0.9
    logits = _retrain_frame0(raw_setup)
    capsys.readouterr()
    journal = raw_setup.journal(True)
    out = capsys.readouterr().out
    assert "warning" in out and "object 0" in out and "1 detections differ" in out
    assert raw_setup.loads == [False, True]  # the rescan's model also runs the full pass
    assert journal.records[1, 0]["detections"][0]["score"] == 0.91
    unchanged = journal.records[1, 1]["detections"]
    assert unchanged == old.records[1, 1]["detections"]  # the other image rescanned identically
    np.testing.assert_array_equal(journal.heatmap((1, 0), 0)[0], logits)
    _assert_consistent_pair(raw_setup, logits)
    assert raw_setup.loads == [False, True]  # reopened without any scan
    key = _key(raw_setup)
    for lock in (raw_setup.cache.entry_lock(key), raw_setup.cache.heatmap_lock(key)):
        assert lock.is_file() and lock.parent == raw_setup.cache.entry(key).parent
    assert not (raw_setup.cache.entry(key) / "run.lock").exists()


def test_heatmap_store_matching_rescan_keeps_the_journal(raw_setup, capsys):
    plain = raw_setup.journal(False)
    receipt = raw_setup.cache.entry(_key(raw_setup))
    before = (receipt / export.CACHE_RECEIPT_NAME).read_bytes()
    journal = raw_setup.journal(True)
    assert journal.records == plain.records and "warning" not in capsys.readouterr().out
    assert (receipt / export.CACHE_RECEIPT_NAME).read_bytes() == before


def _key(raw_setup):
    frames, population = export.load_test_frames(raw_setup.test_root, "pinhole")
    return export.raw_journal_key(raw_setup.run, population, camera="pinhole")


@pytest.mark.parametrize("crash_at", [1, 2, 3, 4], ids=lambda n: f"rename{n}")
def test_heatmap_rebuild_crash_never_mixes_passes(raw_setup, monkeypatch, crash_at):
    raw_setup.journal(False)
    logits = _retrain_frame0(raw_setup)
    rename, calls = os.rename, []

    def crashing_rename(source, destination):
        calls.append(source)
        if len(calls) == crash_at:
            raise OSError("simulated crash")
        return rename(source, destination)

    monkeypatch.setattr(export.os, "rename", crashing_rename)
    with pytest.raises(OSError, match="simulated crash"):
        raw_setup.journal(True)
    monkeypatch.setattr(export.os, "rename", rename)
    journal = raw_setup.journal(False) if crash_at == 1 else None
    if journal is not None:  # crashed before any swap: the old journal is intact
        assert journal.records[1, 0]["detections"][0]["score"] == 0.9
    _assert_consistent_pair(raw_setup, logits)


def _tag(journal, key, base):
    """The load number whose detector produced the first detection's heatmap of ``key``."""
    return int(round(float((journal.heatmap(key, 0)[0] - base)[0, 0, 0] / TAG)))


def _assert_never_mixed(raw_setup, passes, base):
    """Every image's heatmap comes from a pass whose detection matches the journal's."""
    journal = raw_setup.journal(True)
    for frame in (0, 1):
        key = (1, frame)
        score = journal.records[key]["detections"][0]["score"]
        tag = _tag(journal, key, base[frame])
        produced = passes.get(tag, passes[max(passes)]).get(
            frame, raw_setup.by_frame[frame][0].score
        )
        assert produced == score, (frame, tag, score)


def _tagged_setup(tmp_path, monkeypatch, passes):
    test_root = _test_split(tmp_path / "test")
    run = write_base(tmp_path / "runs", 1)
    by_frame, loads = _raw_frames(), []
    _install_bank(monkeypatch, run, by_frame, loads, passes)
    frames, population = export.load_test_frames(test_root, "pinhole")
    cache = export.JournalCache(tmp_path / "cache")

    def journal(heatmaps):
        return export.cached_raw_journal(
            run, frames, population, cache, camera="pinhole", device="cpu", heatmaps=heatmaps
        )

    base = {f: by_frame[f][0].keypoint_heatmap for f in (0, 1)}
    setup = SimpleNamespace(run=run, by_frame=by_frame, loads=loads, cache=cache, journal=journal)
    return setup, base


@pytest.mark.parametrize("crash_at", [1, 2, 3, 4], ids=lambda n: f"rename{n}")
def test_rebuild_crash_with_a_later_mismatch_never_mixes(tmp_path, monkeypatch, crash_at):
    """Frame 0 matches the old journal (its rescan heatmap enters the old store) and frame 1
    does not; the rebuild pass then differs on frame 0 too. No crash may leave frame 0's
    old-pass heatmap under the new journal."""
    passes = {0: {}, 1: {1: 0.35}, 2: {0: 0.92, 1: 0.35}}
    setup, base = _tagged_setup(tmp_path, monkeypatch, passes)
    setup.journal(False)
    rename, calls = os.rename, []

    def crashing_rename(source, destination):
        calls.append(source)
        if len(calls) == crash_at:
            raise OSError("simulated crash")
        return rename(source, destination)

    monkeypatch.setattr(export.os, "rename", crashing_rename)
    with pytest.raises(OSError, match="simulated crash"):
        setup.journal(True)
    monkeypatch.setattr(export.os, "rename", rename)
    _assert_never_mixed(setup, passes, base)


def test_interrupted_fill_then_regenerated_journal_discards_the_store(tmp_path, monkeypatch):
    passes = {0: {}, 1: {}, 2: {0: 0.92}}
    setup, base = _tagged_setup(tmp_path, monkeypatch, passes)
    setup.journal(False)
    scan = export.scan_raw_frame

    def interrupted(model, frame, **kwargs):
        if frame.im_id == 1:
            raise KeyboardInterrupt  # the fill stops after frame 0's heatmap is indexed
        return scan(model, frame, **kwargs)

    monkeypatch.setattr(export, "scan_raw_frame", interrupted)
    with pytest.raises(KeyboardInterrupt):
        setup.journal(True)
    monkeypatch.setattr(export, "scan_raw_frame", scan)
    key = next(iter(setup.cache.root.glob("v*/*.heatmaps")))
    assert (key / export.HEATMAP_INDEX_NAME).read_text().count("\n") == 1
    entry = key.with_name(key.name.removesuffix(export.HEATMAP_ENTRY_SUFFIX))
    shutil.rmtree(entry)  # the journal is regenerated by a plain run (another pass)
    assert setup.journal(False).records[1, 0]["detections"][0]["score"] == 0.92
    _assert_never_mixed(setup, passes, base)


def test_heatmap_store_bound_to_a_replaced_journal_is_discarded(raw_setup):
    raw_setup.journal(True)
    logits = _retrain_frame0(raw_setup)
    shutil.rmtree(raw_setup.cache.entry(_key(raw_setup)))  # a journal from another pass
    raw_setup.journal(False)
    _assert_consistent_pair(raw_setup, logits)


def test_raw_journal_derives_the_valid_pose_journal(tmp_path, monkeypatch):
    """The cached export's CSV equals the uncached one (the uncached bytes unchanged)."""
    test_root = _test_split(tmp_path / "test")
    run = write_base(tmp_path / "runs", 1)
    dets = [_detection(REF), _detection(REF, score=0.4, posed=False)]
    model = export.BankModel(
        1,
        lambda image, k: list(dets),
        export.sha256_file(run / "work" / "last.pth"),
        export.sha256_file(run / "geometry" / "keypoints_3d.json"),
    )
    monkeypatch.setattr(export, "load_bank", lambda bank, **kwargs: [model])
    options = {"objects": [1], "expected_population": export.TestPopulation(2, 1), "device": "cpu"}
    plain = tmp_path / "plain" / "doperbase_tless-test.csv"
    export.run_bank({1: run}, test_root, plain, **options)
    cached = tmp_path / "cached" / "doperbase_tless-test.csv"
    export.run_bank(
        {1: run}, test_root, cached, cache=export.JournalCache(tmp_path / "c"), **options
    )
    assert plain.read_bytes() == cached.read_bytes()


# --- composition --------------------------------------------------------------------------------


def test_multiheatmap_unmatched_keeps_unrefined_pose():
    journal = {REF: _view(REF, [_detection(REF)]), SUP: _view(SUP, [])}
    audit = []
    rows = variants.compose_variants(
        journal, {REF: (REF, SUP)}, _object_keypoints(), _settings(), TLESS, audit=audit
    )
    assert audit[0]["matched"] == [] and audit[0]["single_heatmap"]["accepted"]
    assert audit[0]["multi_heatmap"] is None
    multiview, multiheatmap = rows["multiview"][0], rows["multiheatmap"][0]
    assert multiheatmap == multiview
    single_r, single_t = _row_pose(rows["singleview"][0])
    np.testing.assert_allclose(_row_pose(multiheatmap)[0], single_r, atol=1e-12)
    np.testing.assert_array_equal(_row_pose(multiheatmap)[1], single_t)
    refined_t = _row_pose(rows["singleheatmap"][0])[1]
    assert np.linalg.norm(refined_t - single_t) > 1.0


def test_multiheatmap_rejected_fit_refines_from_original(monkeypatch):
    det = _detection(REF)
    journal = {REF: _view(REF, [det]), SUP: _view(SUP, [_detection(SUP)])}
    original_r = cv2.Rodrigues(np.asarray(det.rvec, np.float64))[0]
    rejected = FusedPose(
        reference_index=0,
        score=0.9,
        R=original_r,
        t=np.asarray(det.tvec, np.float64).ravel(),
        matched=((SUP, 0),),
        accepted=False,
        reprojection_px=(),
        observations=0,
    )
    monkeypatch.setattr(variants, "fuse_reference", lambda *args, **kwargs: [rejected])
    calls = []

    def refine(rvec, tvec, keypoints_mm, observations, crop_size, beta, settings):
        calls.append((rvec.copy(), tvec.copy(), observations))
        audit = PowellAudit(initial_nll=1.0, final_nll=0.0, accepted=True, success=True,
                            iterations=1, evaluations=1, views=len(observations))  # fmt: skip
        return rvec, tvec + 1.0, audit

    monkeypatch.setattr(variants, "refine_pose", refine)
    rows = variants.compose_variants(
        journal, {REF: (REF, SUP)}, _object_keypoints(), _settings("multiheatmap"), TLESS
    )
    ((rvec, tvec, observations),) = calls
    np.testing.assert_array_equal(rvec, cv2.Rodrigues(original_r)[0])
    np.testing.assert_array_equal(tvec.ravel(), np.asarray(det.tvec).ravel())
    assert len(observations) == 2
    relative = _world_to_camera(CAMERAS[SUP]) @ np.linalg.inv(_world_to_camera(CAMERAS[REF]))
    np.testing.assert_array_equal(observations[1].relative_mm, relative)
    np.testing.assert_array_equal(rows["multiheatmap"][0].t, np.asarray(det.tvec).ravel() + 1.0)


@pytest.mark.parametrize("spec", [TLESS, None], ids=["multiview", "singleview_dataset"])
def test_variants_drop_nonpositive_z_and_keep_scores(spec):
    good, behind = _detection(REF, score=0.8), _detection(REF, score=0.6)
    behind.tvec = behind.tvec * np.array([[1.0], [1.0], [-1.0]])
    failed = _detection(REF, score=0.4, posed=False)
    journal = {REF: _view(REF, [good, behind, failed]), SUP: _view(SUP, [])}
    groups = {REF: (REF, SUP) if spec else (REF,)}
    names = variants.VARIANTS if spec else ("singleview", "singleheatmap")
    rows = variants.compose_variants(journal, groups, _object_keypoints(), _settings(*names), spec)
    assert set(rows) == set(names)
    for name in names:
        assert [(r.detection_index, r.score) for r in rows[name]] == [(0, 0.8)], name
        assert all(r.t[2] > 0 for r in rows[name])
    with pytest.raises(ValueError, match="multi-view"):
        variants.compose_variants(journal, groups, _object_keypoints(), _settings(), None)


def test_variant_manifest_matches_settings():
    settings = _settings("singleheatmap", optimizer="powell")
    groups = {REF: (REF, SUP), SUP: (SUP, REF)}
    manifest = variants.variant_manifest(settings, groups, variant="singleheatmap")
    assert manifest["optimizer"] == "powell" and manifest["preparation"] == "numpy"
    assert manifest["variant"] == "singleheatmap" and manifest["crop_size"] == CROP_SIZE
    assert manifest["threads"] == settings.refine.threads
    assert {"numba", "scipy", "opencv", "numpy"} <= set(manifest["versions"])
    assert manifest["versions"]["opencv"] == cv2.__version__
    assert manifest["groups_sha256"] == variants.groups_sha256(groups)
    assert variants.groups_sha256({REF: (REF,), SUP: (SUP,)}) != manifest["groups_sha256"]
    default = variants.variant_manifest(_settings(optimizer="lbfgsb"), groups, variant="singleview")
    assert default["optimizer"] == "lbfgsb"


def test_variant_csv_tokens_are_alphanumeric():
    recipe = load_recipe("hopev2")
    assert variants.csv_method(recipe, "base", "multiheatmap") == "doperbasemultiheatmap"
    assert (
        variants.csv_method(recipe, "static_pseudo", "singleheatmap")
        == "doperstaticfallbacksingleheatmap"
    )
    with pytest.raises(ValueError, match="variant"):
        variants.csv_method(recipe, "base", "multi-view")


# --- the CLI -------------------------------------------------------------------------------------


def test_handal_refuses_multiview(tmp_path, capsys):
    paths = ["--root", str(tmp_path / "root"), "--runs", str(tmp_path / "runs"), "--dry-run"]
    test = ["evaluate", "handal", "--split", "test", *paths]
    for variant in ("multiview", "multiheatmap"):
        assert cli.main([*test, "--variant", variant]) == 1
        assert "multi-view" in capsys.readouterr().out
    assert cli.main([*test, "--variant", "singleheatmap"]) == 0
    out = capsys.readouterr().out
    assert "singleheatmap" in out and "lbfgsb" in out
    assert cli.main([*test, "--variant", "singleheatmap", "--optimizer", "powell",
                     "--preparation", "parallel"]) == 1  # fmt: skip
    assert "Powell" in capsys.readouterr().out
    assert cli.main([*test, "--variant", "singleheatmap", "--record-time"]) == 1
    assert "untimed" in capsys.readouterr().out


def test_heatmap_variants_need_numba_for_lbfgsb(tmp_path, capsys, monkeypatch):
    import importlib.util

    find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: None if name == "numba" else find_spec(name, *a),
    )
    paths = ["--root", str(tmp_path / "root"), "--runs", str(tmp_path / "runs"), "--dry-run"]
    test = ["evaluate", "tless", "--split", "test", *paths]
    assert cli.main([*test, "--variant", "singleheatmap"]) == 1
    assert "doper[refine]" in capsys.readouterr().out
    assert cli.main([*test, "--variant", "multiheatmap", "--optimizer", "powell"]) == 0
    assert cli.main([*test, "--variant", "multiview"]) == 0  # no refinement, no numba


def test_hot3d_multiview_refuses_unstaged_support_streams(tmp_path):
    from doper.cli import evaluate

    recipe = load_recipe("hot3d")
    for variant in variants.MULTIVIEW_VARIANTS:
        with pytest.raises(ValueError, match="support streams are not staged"):
            evaluate.variant_inputs(recipe, tmp_path / "root", variant)


def test_recipe_refine_defaults_and_overrides():
    import yaml

    import doper.recipes
    from doper.recipes.schema import RefineSpec, parse_recipe

    assert load_recipe("tless").refine == RefineSpec(optimizer="lbfgsb", preparation="numpy")
    data = yaml.safe_load((Path(doper.recipes.__file__).parent / "tless.yaml").read_text())
    data["refine"] = {"optimizer": "powell"}
    assert parse_recipe(data).refine.optimizer == "powell"
    for bad in ({"optimizer": "adam"}, {"optimizer": "powell", "preparation": "parallel"},
                {"threads": 0}, {"unknown": 1}):  # fmt: skip
        data["refine"] = bad
        with pytest.raises(ValueError):
            parse_recipe(data)
