# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BOP test export: validity filter, journals, partial banks, CSV format and timing protocol."""

import csv
import json

import cv2
import numpy as np
import pytest

from doper.bop import export, timing
from doper.bop.hot3d import rectify
from doper.bop.sources import sha256_file
from doper.infer import Detection

WIDTH, HEIGHT = 64, 48
CAMERA_K = [80.0, 0.0, 32.0, 0.0, 80.0, 24.0, 0.0, 0.0, 1.0]
TVEC_MM = np.array([10.0, -5.0, 600.0])
RVEC = np.array([0.3, -0.2, 0.4])
ROTATION = cv2.Rodrigues(RVEC)[0]
FISHEYE_DISTORTION = [0.12, -0.05, 0.01, 0.002, 0.0005, -0.0004, 0.0002, 0.0001]
FISHEYE_DISTORTION += [0.001, -0.001, 0.0002, 0.0001]
CALIBRATION = {
    "calibration": {
        "image_width": WIDTH,
        "image_height": HEIGHT,
        "projection_model_type": "CameraModelType.FISHEYE624",
        "projection_params": [80.0, WIDTH / 2, HEIGHT / 2, *FISHEYE_DISTORTION],
        "label": "fixture",
        "serial_number": "fixture",
    },
}


def _write_test_split(root, frames=3, scenes=(1,)):
    """BOP test split: RGB and ``scene_camera.json`` only (no GT is ever read)."""
    for scene in scenes:
        folder = root / f"{scene:06d}"
        (folder / "rgb").mkdir(parents=True)
        camera = {}
        for frame in range(frames):
            image = np.full((HEIGHT, WIDTH, 3), 10 * frame + scene, np.uint8)
            cv2.imwrite(str(folder / "rgb" / f"{frame:06d}.png"), image)
            camera[str(frame)] = {"cam_K": CAMERA_K, "depth_scale": 1.0}
        (folder / "scene_camera.json").write_text(json.dumps(camera))
    return root


def _fixture_population(test_root, camera="pinhole"):
    """The fixture's own population (packaged datasets use their frozen one)."""
    frames, _ = export.load_test_frames(test_root, camera)
    return export.TestPopulation(len(frames), len({f.scene_id for f in frames}))


def _run_bank(bank, test_root, out, **kwargs):
    camera = kwargs.get("camera", "pinhole")
    kwargs.setdefault("expected_population", _fixture_population(test_root, camera))
    return export.run_bank(bank, test_root, out, **kwargs)


def _measure(bank, test_root, **kwargs):
    kwargs.setdefault("expected_population", _fixture_population(test_root, kwargs["camera"]))
    return timing.measure(bank, test_root, **kwargs)


def _detection(score=0.9, rotation=ROTATION, tvec=TVEC_MM, solved=True):
    rvec = cv2.Rodrigues(np.asarray(rotation, dtype=np.float64))[0] if solved else None
    det = Detection(np.zeros(4), score, np.zeros((4, 2)), np.ones(4))
    det.rvec, det.tvec = rvec, (np.asarray(tvec, dtype=np.float64) if solved else None)
    return det


def _raw(rotation, score=0.9):
    """Detection whose pose is a raw rotation matrix (bypasses Rodrigues orthonormalization)."""
    det = _detection(score)
    det.raw_rotation = np.asarray(rotation, dtype=np.float64)
    return det


_POSE_FROM_DETECTION = export.pose_from_detection


def _raw_aware_pose(det):
    pose = _POSE_FROM_DETECTION(det)
    if pose is not None and hasattr(det, "raw_rotation"):
        pose["R"] = det.raw_rotation.ravel().tolist()
    return pose


def _bank_models(detections_by_obj, calls=None):
    """Fake resident bank: each object returns its fixed detections on every image."""

    def make(obj, dets):
        def detect(image, camera_matrix):
            if calls is not None:
                calls.append((obj, image.shape, tuple(np.asarray(camera_matrix).ravel())))
            return list(dets)

        return export.BankModel(obj, detect, f"ckpt{obj}", f"geo{obj}")

    return [make(obj, dets) for obj, dets in sorted(detections_by_obj.items())]


@pytest.fixture
def fake_bank(monkeypatch):
    """Patch ``export.load_bank`` to serve fake models; returns a setter."""
    state = {}

    def install(detections_by_obj, calls=None, poses=None):
        models = _bank_models(detections_by_obj, calls)
        state["models"] = models
        monkeypatch.setattr(export, "load_bank", lambda bank, **kwargs: models)
        if poses is not None:
            monkeypatch.setattr(export, "pose_from_detection", poses)

    return install


def _read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.reader(stream))


def _rotation_pose(rotation, score=0.9, t=TVEC_MM):
    return {"score": score, "R": np.asarray(rotation).ravel().tolist(), "t": list(map(float, t))}


def test_valid_pose_matches_source_predicate():
    shear = np.eye(3)
    assert export.valid_pose(_rotation_pose(ROTATION))
    assert not export.valid_pose(_rotation_pose(np.diag([1.0, 1.0, -1.0])))
    # Scaling: det drift just inside / outside 1e-5 (orthonormality within atol + rtol).
    inside, outside = (1 + 0.9e-5) ** (1 / 3), (1 + 1.1e-5) ** (1 / 3)
    assert export.valid_pose(_rotation_pose(ROTATION * inside))
    assert not export.valid_pose(_rotation_pose(ROTATION * outside))
    # Shear with det exactly 1: off-diagonal R^T R drift just inside / outside atol 1e-5.
    shear[0, 1] = 0.9e-5
    assert export.valid_pose(_rotation_pose(ROTATION @ shear))
    shear[0, 1] = 1.1e-5
    assert not export.valid_pose(_rotation_pose(ROTATION @ shear))
    # Diagonal drift above atol but within atol + NumPy's default rtol (det stays 1).
    stretch = np.diag([1 + 0.75e-5, 1 / (1 + 0.75e-5), 1.0])
    assert export.valid_pose(_rotation_pose(stretch))
    assert export.valid_pose(_rotation_pose(ROTATION, score=0.15))
    assert export.valid_pose(_rotation_pose(ROTATION, score=1.0))
    assert not export.valid_pose(_rotation_pose(ROTATION, score=0.1499))
    assert not export.valid_pose(_rotation_pose(ROTATION, score=1.0001))
    assert not export.valid_pose(_rotation_pose(ROTATION, t=[0.0, 0.0, 0.0]))
    assert not export.valid_pose(_rotation_pose(ROTATION, t=[0.0, 0.0, float("nan")]))
    assert not export.valid_pose(_rotation_pose(ROTATION, score=float("nan")))
    assert not export.valid_pose({"score": 0.9, "R": [1.0], "t": [0, 0, 1]})
    assert not export.valid_pose({"R": ROTATION.ravel().tolist(), "t": [0, 0, 1]})


def test_export_drops_invalid_rows_and_counts_them(tmp_path, fake_bank):
    test_root = _write_test_split(tmp_path / "test", frames=2)
    detections = {
        1: [
            _detection(0.95),
            _detection(0.9, tvec=[0.0, 0.0, -5.0]),
            _detection(0.8, tvec=[0.0, 0.0, float("nan")]),
            _detection(0.7, solved=False),
            _detection(1.2),
        ],
        2: [
            _raw(ROTATION, 0.9),
            _raw(np.diag([1.0, 1.0, -1.0]), 0.8),  # reflection, det -1
            _raw(ROTATION * (1 + 0.9e-5) ** (1 / 3), 0.7),  # det drift inside 1e-5
            _raw(ROTATION * (1 + 1.1e-5) ** (1 / 3), 0.6),  # det drift outside 1e-5
            _raw(ROTATION, 0.1),  # below the 0.15 score floor
        ],
    }
    fake_bank(detections, poses=_raw_aware_pose)
    out = tmp_path / "out" / "doperbase_hopev2-test.csv"
    summary = _run_bank(
        {1: tmp_path / "a", 2: tmp_path / "b"}, test_root, out, objects=[1, 2], device="cpu"
    )
    rows = _read_csv(out)[1:]
    # Per image: object 1 keeps only 0.95; object 2 keeps the valid and the inside drift.
    assert [(r[2], r[3]) for r in rows] == [("1", "0.95"), ("2", "0.9"), ("2", "0.7")] * 2
    assert summary.pose_rows == 6
    # Per image: obj1 negative z, NaN, score > 1; obj2 reflection, outside drift, score < 0.15.
    assert summary.invalid_poses == 2 * 6
    assert summary.pnp_failures == 2
    assert summary.detections == 2 * 10
    for row in rows:
        pose = {
            "score": float(row[3]),
            "R": list(map(float, row[4].split())),
            "t": list(map(float, row[5].split())),
        }
        assert export.valid_pose(pose)


def test_export_resumes_from_journal(tmp_path, fake_bank):
    test_root = _write_test_split(tmp_path / "test", frames=3)
    calls = []
    fake_bank({1: [_detection(0.9)]}, calls)
    out = tmp_path / "doperbase_hopev2-test.csv"
    journal = tmp_path / "journal"
    kwargs = {"objects": [1], "device": "cpu", "journal_dir": journal}
    first = _run_bank({1: tmp_path / "run"}, test_root, out, **kwargs)
    reference = out.read_bytes()
    assert len(calls) == 3 and first.resumed_images == 0
    lines = (journal / "frames.jsonl").read_text().splitlines(keepends=True)
    (journal / "frames.jsonl").write_text(lines[0] + lines[1][:20])  # torn final line
    calls.clear()
    second = _run_bank({1: tmp_path / "run"}, test_root, out, **kwargs)
    assert len(calls) == 2 and second.resumed_images == 1
    assert out.read_bytes() == reference
    assert second.images == 3
    with pytest.raises(ValueError, match="identity"):
        _run_bank({1: tmp_path / "run"}, test_root, out, score_threshold=0.3, **kwargs)
    (journal / "frames.jsonl").write_text("not json\n" + lines[1])
    with pytest.raises(ValueError, match="corrupt"):
        _run_bank({1: tmp_path / "run"}, test_root, out, **kwargs)


def test_partial_bank_refused_without_flag(tmp_path, fake_bank):
    test_root = _write_test_split(tmp_path / "test", frames=1)
    fake_bank({1: [_detection()]})
    out = tmp_path / "doperbase_hot3d-test.csv"
    with pytest.raises(ValueError, match="partial bank"):
        _run_bank({1: tmp_path / "run"}, test_root, out, device="cpu")
    summary = _run_bank(
        {1: tmp_path / "run"}, test_root, out, device="cpu", allow_partial_bank=True
    )
    assert summary.objects == (1,)
    assert summary.missing_objects == tuple(range(2, 34))
    with pytest.raises(ValueError, match="not in the dataset"):
        _run_bank({99: tmp_path / "run"}, test_root, out, device="cpu", allow_partial_bank=True)


def test_csv_name_rules(tmp_path, fake_bank):
    assert export.csv_name("doperbase", "hope") == "doperbase_hopev2-test.csv"
    assert export.csv_name("doperbase", "tless") == "doperbase_tless-test.csv"
    with pytest.raises(ValueError):
        export.csv_name("doper-base", "handal")
    with pytest.raises(ValueError):
        export.csv_name("doperbase", "ycbv")
    test_root = _write_test_split(tmp_path / "test", frames=1)
    fake_bank({1: [_detection()]})
    summary = _run_bank(
        {1: tmp_path / "run"},
        test_root,
        tmp_path / "doperbase_hope-test.csv",
        objects=[1],
        device="cpu",
    )
    assert summary.dataset == "hopev2"
    assert summary.csv_path == tmp_path / "doperbase_hopev2-test.csv"
    assert summary.csv_path.is_file()
    with pytest.raises(ValueError, match="CSV name"):
        _run_bank({1: tmp_path / "run"}, test_root, tmp_path / "out.csv", device="cpu")


def test_csv_header_and_units(tmp_path, fake_bank):
    test_root = _write_test_split(tmp_path / "test", frames=2, scenes=(2, 1))
    fake_bank({3: [_detection(0.5), _detection(0.9)], 1: [_detection(0.4)]})
    out = tmp_path / "doperbase_tless-test.csv"
    _run_bank({1: tmp_path / "a", 3: tmp_path / "b"}, test_root, out, objects=[1, 3], device="cpu")
    table = _read_csv(out)
    assert table[0] == ["scene_id", "im_id", "obj_id", "score", "R", "t", "time"]
    keys = [(int(r[0]), int(r[1]), int(r[2]), -float(r[3])) for r in table[1:]]
    assert keys == sorted(keys) and len(keys) == 2 * 2 * 3
    row = table[1]
    assert row[:4] == ["1", "0", "1", "0.4"]
    np.testing.assert_allclose(np.array(row[4].split(), float).reshape(3, 3), ROTATION, atol=1e-10)
    assert [float(v) for v in row[5].split()] == [10.0, -5.0, 600.0]  # millimeters, unscaled
    assert row[6] == "-1"
    _run_bank(
        {1: tmp_path / "a", 3: tmp_path / "b"},
        test_root,
        out,
        objects=[1, 3],
        device="cpu",
        record_time=True,
        journal_dir=tmp_path / "timed",
    )
    times = {(r[0], r[1]): float(r[6]) for r in _read_csv(out)[1:]}
    assert all(t > 0 for t in times.values()) and len(times) == 4


def test_test_frames_from_split_never_read_gt(tmp_path):
    root = _write_test_split(tmp_path / "test", frames=2)
    (root / "000001" / "scene_gt.json").write_text("not json")
    frames, population = export.load_test_frames(root, "pinhole")
    assert [(f.scene_id, f.im_id) for f in frames] == [(1, 0), (1, 1)]
    assert frames[0].rgb_sha256 == sha256_file(frames[0].path)
    assert len(population) == 64
    with pytest.raises(ValueError, match="native"):
        export.load_test_frames(root, "native")


def test_keypoints_are_converted_to_millimetres(tmp_path):
    path = tmp_path / "keypoints_3d.json"
    path.write_text(json.dumps({"keypoints_3d": [[0.001, 0.002, -0.5]] * 4}))
    np.testing.assert_allclose(export.keypoints_mm(path), [[1.0, 2.0, -500.0]] * 4)
    path.write_text(json.dumps({"keypoints_3d": [[0.0, float("nan"), 0.0]] * 4}))
    with pytest.raises(ValueError):
        export.keypoints_mm(path)


def _native_manifest(tmp_path, *, corrupt=False):
    rng = np.random.default_rng(0)
    native = rng.integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    native_path = tmp_path / "native" / "000000.png"
    native_path.parent.mkdir(parents=True)
    cv2.imwrite(str(native_path), native)
    rectified, camera_matrix = rectify(native, CALIBRATION)
    payload = export.encode_jpeg(rectified)
    staged = tmp_path / "staged.jpg"
    staged.write_bytes(payload)
    frame = {
        "scene_id": 7,
        "im_id": 3,
        "path": "staged.jpg",
        "rgb_sha256": sha256_file(staged) if not corrupt else "0" * 64,
        "cam_K": camera_matrix.ravel().tolist(),
        "native_path": "native/000000.png",
        "native_rgb_sha256": sha256_file(native_path),
        "native_calibration": CALIBRATION,
    }
    manifest = tmp_path / "test_manifest.json"
    manifest.write_text(json.dumps({"dataset": "hot3d", "frames": [frame]}))
    return manifest, camera_matrix


def test_native_export_rectifies_inside_the_timed_span(tmp_path, fake_bank, monkeypatch):
    manifest, camera_matrix = _native_manifest(tmp_path)
    calls = []
    fake_bank({5: [_detection()]}, calls)
    events = []
    clock = iter(range(100, 200))

    def fake_clock():
        events.append("clock")
        return float(next(clock))

    original_rectify = export.hot3d.rectify

    def logged_rectify(*args, **kwargs):
        events.append("rectify")
        return original_rectify(*args, **kwargs)

    monkeypatch.setattr(export, "_clock", fake_clock)
    monkeypatch.setattr(export.hot3d, "rectify", logged_rectify)
    out = tmp_path / "doperbase_hot3d-test.csv"
    summary = _run_bank(
        {5: tmp_path / "run"},
        manifest,
        out,
        objects=[5],
        camera="native",
        device="cpu",
        record_time=True,
    )
    assert events == ["clock", "rectify", "clock"]
    assert calls[0][1] == (HEIGHT, WIDTH, 3)
    np.testing.assert_allclose(calls[0][2], camera_matrix.ravel())
    assert summary.pose_rows == 1
    assert _read_csv(out)[1][6] == "1"


def test_native_export_rejects_drifted_rectification(tmp_path, fake_bank):
    manifest, _ = _native_manifest(tmp_path, corrupt=True)
    fake_bank({5: [_detection()]})
    with pytest.raises(ValueError, match="rectification"):
        _run_bank(
            {5: tmp_path / "run"},
            manifest,
            tmp_path / "doperbase_hot3d-test.csv",
            objects=[5],
            camera="native",
            device="cpu",
        )


def test_timing_counts_cold_first_image_without_warmup(tmp_path, fake_bank, monkeypatch):
    test_root = _write_test_split(tmp_path / "test", frames=3)
    calls = []
    fake_bank({1: [_detection()], 2: []}, calls)
    events = []
    clock = iter(float(v) for v in (0, 2, 10, 11, 20, 24))

    def fake_clock():
        events.append("clock")
        return next(clock)

    original_decode = export.cv2.imdecode

    def logged_decode(*args, **kwargs):
        events.append("decode")
        return original_decode(*args, **kwargs)

    monkeypatch.setattr(export, "_clock", fake_clock)
    monkeypatch.setattr(export.cv2, "imdecode", logged_decode)
    summary = _measure(
        {1: tmp_path / "a", 2: tmp_path / "b"}, test_root, camera="pinhole", device="cpu"
    )
    assert events == ["decode", "clock", "clock"] * 3  # timer starts after decode
    assert len(calls) == 6  # every model on every image, no warmup pass
    assert summary.seconds == (2.0, 1.0, 4.0)  # cold first image counted
    assert summary.images == 3 and summary.objects_per_image == 2
    assert summary.mean_seconds == pytest.approx(7 / 3)
    assert summary.median_seconds == 2.0 and summary.max_seconds == 4.0
    assert summary.p95_seconds == pytest.approx(2.0 + 2.0 * 0.9)
    assert summary.protocol["explicit_warmup_passes"] == 0
    assert summary.protocol["cold_first_measured_image_included"] is True
    assert summary.pose_rows == 3 and summary.images_without_poses == 0
    assert "torch" in summary.hardware


def test_timing_native_protocol_names_rectification(tmp_path, fake_bank):
    manifest, _ = _native_manifest(tmp_path)
    fake_bank({5: [_detection()]})
    summary = _measure({5: tmp_path / "run"}, manifest, camera="native", device="cpu")
    assert summary.images == 1
    assert "native-camera rectification" in summary.protocol["includes"]
    assert summary.protocol["warmup_frames_discarded"] == 0


def test_truncated_split_refused(tmp_path, fake_bank):
    test_root = _write_test_split(tmp_path / "test", frames=3)
    fake_bank({1: [_detection()]})
    population = _fixture_population(test_root)
    (test_root / "000001" / "rgb" / "000002.png").unlink()
    out = tmp_path / "doperbase_hopev2-test.csv"
    with pytest.raises(ValueError, match="expected 3 images in 1 scenes"):
        export.run_bank({1: tmp_path / "run"}, test_root, out, objects=[1], device="cpu",
                        expected_population=population)  # fmt: skip
    # Packaged datasets default to their frozen population (HOPEv2: 457 images, 47 scenes).
    with pytest.raises(ValueError, match="expected 457 images in 47 scenes"):
        export.run_bank({1: tmp_path / "run"}, test_root, out, objects=[1], device="cpu")
    with pytest.raises(ValueError, match="expected 457"):
        timing.measure({1: tmp_path / "run"}, test_root, camera="pinhole", dataset="hope",
                       device="cpu")  # fmt: skip
    with pytest.raises(ValueError, match="dataset or an expected"):
        timing.measure({1: tmp_path / "run"}, test_root, camera="pinhole", device="cpu")


def test_frozen_populations_match_the_research_counts():
    counts = {"handal": (1684, 61), "hope": (457, 47), "tless": (1000, 20), "hot3d": (5140, 1028)}
    for dataset, (images, scenes) in counts.items():
        population = export.frozen_population(dataset)
        assert (population.images, population.scenes) == (images, scenes)
    hot3d = export.frozen_population("hot3d")
    assert hot3d.keys is not None and len(hot3d.keys) == 5140
    assert export.frozen_population("tless").keys is None


@pytest.mark.parametrize("defect", ["no_camera", "stray_rgb", "named_scene", "stray_file"])
def test_malformed_scene_dir_refused(tmp_path, defect):
    test_root = _write_test_split(tmp_path / "test", frames=2)
    scene = test_root / "000001"
    if defect == "no_camera":
        (scene / "scene_camera.json").unlink()
    elif defect == "stray_rgb":
        (scene / "rgb" / "notes.txt").write_text("x")
    elif defect == "named_scene":
        (test_root / "extra").mkdir()
    else:
        (test_root / "README.md").write_text("x")
    with pytest.raises(ValueError, match="malformed"):
        export.load_test_frames(test_root, "pinhole")


def test_native_population_must_match_the_frozen_index(tmp_path, fake_bank):
    manifest, _ = _native_manifest(tmp_path)
    fake_bank({5: [_detection()]})
    frozen = export.TestPopulation(1, 1, frozenset({(7, 4)}))
    with pytest.raises(ValueError, match="frozen test image index"):
        export.run_bank({5: tmp_path / "run"}, manifest, tmp_path / "doperbase_hot3d-test.csv",
                        objects=[5], camera="native", device="cpu",
                        expected_population=frozen)  # fmt: skip


def test_timed_resume_refuses_other_hardware(tmp_path, fake_bank, monkeypatch):
    test_root = _write_test_split(tmp_path / "test", frames=2)
    fake_bank({1: [_detection()]})
    out = tmp_path / "doperbase_hopev2-test.csv"
    kwargs = {"objects": [1], "device": "cpu", "journal_dir": tmp_path / "journal",
              "record_time": True}  # fmt: skip
    _run_bank({1: tmp_path / "run"}, test_root, out, **kwargs)
    _run_bank({1: tmp_path / "run"}, test_root, out, **kwargs)  # same hardware resumes
    with pytest.raises(ValueError, match="identity"):
        _run_bank({1: tmp_path / "run"}, test_root, out, **{**kwargs, "record_time": False})
    other = {**timing.hardware("cpu"), "cpu": "another CPU"}
    monkeypatch.setattr(timing, "hardware", lambda device: other)
    with pytest.raises(ValueError, match="identity"):
        _run_bank({1: tmp_path / "run"}, test_root, out, **kwargs)


def test_cpu_model_reads_proc_cpuinfo(tmp_path, monkeypatch):
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor\t: 0\nmodel name\t: Fixture CPU 9000\n")
    monkeypatch.setattr(timing, "CPUINFO", cpuinfo)
    assert timing.cpu_model() == "Fixture CPU 9000"
    monkeypatch.setattr(timing, "CPUINFO", tmp_path / "missing")
    assert timing.cpu_model()


def test_a_held_journal_lock_is_a_clear_error(tmp_path) -> None:
    import fcntl

    tmp_path.joinpath("run.lock").touch()
    with tmp_path.joinpath("run.lock").open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        error = pytest.raises(BlockingIOError, match=f"another export holds .*{tmp_path.name}")
        with error, export._locked(tmp_path):
            pass


def test_cache_entry_and_heatmap_locks_wait(tmp_path, capsys) -> None:
    from lock_helpers import assert_waits_for

    cache = export.JournalCache(tmp_path / "cache")
    key = export.JournalKey("a" * 64, "b" * 64, "c" * 64, "d" * 64)

    def produce(directory):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / export.JOURNAL_NAME).write_text("")

    def waiting() -> None:
        assert not cache.contains(key)

    assert assert_waits_for(cache.entry_lock(key), lambda: cache.get_or_run(key, produce),
                            waiting) == {}  # fmt: skip
    assert "waiting" in capsys.readouterr().out
    store = cache.heatmap_entry(key)

    def fill() -> str:
        with export._locked(store, cache.heatmap_lock(key), wait=True):
            return cache.generation(key)  # the heatmap lock, then the entry lock

    assert assert_waits_for(cache.heatmap_lock(key), fill, lambda: None) == cache.generation(key)
