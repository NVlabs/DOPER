# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isaac launch contracts exercised without Kit or a GPU."""

import json
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from rendering.config import render_config_from_mapping
from rendering.environment_bundle import ReplayInputs
from rendering.isaac import isaac_parallel as launch
from rendering.isaac.settings import IsaacSettings
from rendering.usd_mesh import write_mesh_usda

RUNTIME = "6.0.1"
MINIMAL = ["--num-images", "1", "--config", "render.yaml", "--outdir", "out"]


def launch_inputs(outdir, scenes):
    return launch.LaunchInputs(
        outdir=outdir,
        scenes_path=scenes,
        config_path=Path("config"),
        usd_path="usd",
        keypoints_path=None,
    )


@pytest.fixture
def recipe(tmp_path, monkeypatch):
    usd = tmp_path / "object.usda"
    write_mesh_usda(
        str(usd),
        object_name="object",
        points=[(0, 0, 0), (0.1, 0, 0), (0, 0.1, 0)],
        face_vertex_counts=[3],
        face_vertex_indices=[0, 1, 2],
        keypoint_names=["KP_000"],
        keypoints_3d=[(0, 0, 0)],
    )
    config = tmp_path / "recipe.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "object": {"usd_path": str(usd), "num_keypoints": 1},
                "scene": {"num_distractors_min": 0, "num_distractors_max": 0},
            }
        )
    )
    monkeypatch.setattr(launch, "probe_runtime", lambda _: {"isaacsim": RUNTIME})
    monkeypatch.setattr("rendering.run_manifest.code_revision", lambda _: "test-revision")
    return [
        "--num-images",
        "2",
        "--config",
        str(config),
        "--outdir",
        str(tmp_path / "out"),
        "--isaac-python",
        "isaac-python",
        "--max-restarts",
        "0",
        "--progress-interval-sec",
        "0.001",
        "--no-merge",
    ]


def values(command):
    return dict(zip(command[2::2], command[3::2], strict=False))  # --resume is a trailing flag


def mock_worker(monkeypatch, *, returncode=0):
    calls = []

    def popen(command, **kwargs):
        calls.append(command)
        flags = values(command)
        out = Path(flags["--outdir"])
        manifests = list((out / "manifests").glob("shard_*.json"))
        assert manifests, "attempt must be durable before process startup"
        manifest = next(
            json.loads(path.read_text())
            for path in manifests
            if json.loads(path.read_text())["status"] == "running"
        )
        assert kwargs["start_new_session"] is True
        worker = int(flags["--worker-id"])
        images, annotations = [], []
        for index in range(int(flags["--start-idx"]), int(flags["--end-idx"])):
            name = f"{index:06d}.png"
            (out / "images" / name).write_bytes(b"png")
            images.append({"id": index, "file_name": name, "width": 8, "height": 8})
            annotations.append(
                {
                    "id": index,
                    "image_id": index,
                    "keypoints": [1, 1, 2],
                    "num_keypoints": 1,
                    "bbox": [0, 0, 2, 2],
                }
            )
        (out / f"annotations_worker_{worker:03d}.json").write_text(
            json.dumps(
                {
                    "start_idx": int(flags["--start-idx"]),
                    "end_idx": int(flags["--end-idx"]),
                    "images": images,
                    "annotations": annotations,
                }
            )
        )
        (out / f"settings_worker_{worker:03d}.json").write_text(
            json.dumps(manifest["renderer_settings"])
        )
        return SimpleNamespace(returncode=returncode, poll=lambda: returncode)

    monkeypatch.setattr(launch.subprocess, "Popen", popen)
    return calls


def manifest(tmp_path):
    return json.loads((tmp_path / "out/manifests/shard_00000000_00000002.json").read_text())


@pytest.mark.parametrize(
    "extra",
    [
        ["--num-images", "0"],
        ["--workers-per-gpu", "0"],
        ["--start-idx", "-1"],
        ["--worker-id-offset", "-1"],
        ["--max-restarts", "-1"],
        ["--gpus", "0,0"],
        ["--gpus", "-1"],
        ["--gpus", "0,"],
        ["--gpus", "hello"],
        ["--gpus", ""],
        ["--progress-interval-sec", "nan"],
        ["--progress-interval-sec", "0"],
        ["--stall-timeout-sec", "-1"],
        ["--startup-grace-sec", "inf"],
    ],
)
def test_bad_cli_never_creates_output(recipe, tmp_path, extra):
    with pytest.raises(SystemExit):
        launch.main(recipe + extra)
    assert not (tmp_path / "out").exists()


def test_success_and_worker_layout_resume_preserve_attempts(recipe, tmp_path, monkeypatch):
    calls = mock_worker(monkeypatch)
    assert launch.main(recipe) == 0
    assert manifest(tmp_path)["status"] == "complete"
    assert launch.main(recipe + ["--resume", "--gpus", "0,1"]) == 0
    current = manifest(tmp_path)
    assert len(current["attempts"]) == 1
    assert current["attempts"][0]["status"] == "complete"
    assert current["reused_sample_indices"] == [0, 1]
    assert len(current["shard"]["workers"]) == 2
    assert len(calls) == 3


@pytest.mark.parametrize("extra", [["--seed", "9"], ["--num-images", "1"]])
def test_resume_rejection_does_not_mutate_output(recipe, tmp_path, monkeypatch, extra):
    mock_worker(monkeypatch)
    launch.main(recipe)
    out = tmp_path / "out"
    before = {str(path): path.read_bytes() for path in out.rglob("*") if path.is_file()}
    with pytest.raises(ValueError):
        launch.main(recipe + ["--resume", *extra])
    assert before == {str(path): path.read_bytes() for path in out.rglob("*") if path.is_file()}


def test_legacy_resume_does_not_mutate_output(recipe, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "old.json").write_text("{}")
    with pytest.raises(ValueError):
        launch.main(recipe + ["--resume"])
    assert list(out.iterdir()) == [out / "old.json"]


@pytest.mark.parametrize("error", [OSError("spawn failed"), KeyboardInterrupt()])
def test_partial_startup_cleans_started_workers_and_marks_failed(
    recipe, tmp_path, monkeypatch, error
):
    live = SimpleNamespace(returncode=None, poll=lambda: None)
    started, stopped = [], []

    def popen(*args, **kwargs):
        if started:
            raise error
        started.append(live)
        return live

    monkeypatch.setattr(launch.subprocess, "Popen", popen)
    monkeypatch.setattr(launch, "terminate_process_group", stopped.append)
    with pytest.raises(type(error)):
        launch.main(recipe + ["--gpus", "0,1"])
    assert stopped == [live]
    assert manifest(tmp_path)["status"] == "failed"
    assert manifest(tmp_path)["failed_workers"] == [0, 1]


def test_nonzero_exit_cannot_succeed_with_valid_artifacts(recipe, tmp_path, monkeypatch):
    mock_worker(monkeypatch, returncode=9)
    monkeypatch.setattr("rendering.watchdog.terminate_process_group", lambda _: None)
    monkeypatch.setattr(launch, "terminate_process_group", lambda _: None)
    with pytest.raises(RuntimeError, match="failure"):
        launch.main(recipe)
    assert manifest(tmp_path)["status"] == "failed"


def test_explicit_validation_enforces_stereo_mask_and_keypoints(tmp_path):
    worker = launch.WorkerRange(0, 0, 1, 0)
    (tmp_path / "images").mkdir()
    (tmp_path / "images/000000_left.png").touch()
    payload = {
        "images": [
            {
                "id": 0,
                "file_name": "000000_left.png",
                "width": 8,
                "height": 8,
                "pair_id": 0,
                "camera_index": 0,
                "view": "left",
            }
        ],
        "annotations": [
            {"id": 0, "image_id": 0, "keypoints": [], "num_keypoints": 0, "bbox": [0, 0, 1, 1]}
        ],
    }
    (tmp_path / "annotations_worker_000.json").write_text(json.dumps(payload))
    errors = launch.worker_output_errors(
        tmp_path, worker, SimpleNamespace(images_per_sample=2, render_masks=True), 1
    )
    assert any("sample range" in error for error in errors)
    assert any("keypoints shape" in error for error in errors)
    assert any("mask" in error for error in errors)


def test_effective_settings_mismatch_rejected(tmp_path):
    worker = launch.WorkerRange(0, 0, 1, 0)
    (tmp_path / "settings_worker_000.json").write_text(json.dumps({"effective": {"spp": 1}}))
    assert launch.worker_settings_errors(tmp_path, worker, {"effective": {"spp": 64}})


def test_retry_of_zero_exit_missing_outputs_is_bounded_and_resumes(recipe, tmp_path, monkeypatch):
    successful_calls = mock_worker(monkeypatch)
    success = launch.subprocess.Popen
    calls, stopped = [], []

    def popen(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            return SimpleNamespace(returncode=0, poll=lambda: 0)
        return success(command, **kwargs)

    monkeypatch.setattr(launch.subprocess, "Popen", popen)
    monkeypatch.setattr(launch, "terminate_process_group", stopped.append)
    assert launch.main(recipe + ["--max-restarts", "1"]) == 0
    assert len(calls) == 2 and len(stopped) == 1
    assert "--resume" in calls[1]
    assert len(successful_calls) == 1
    assert manifest(tmp_path)["status"] == "complete"


def test_exhausted_zero_exit_missing_outputs_marks_failure(recipe, tmp_path, monkeypatch):
    calls = []

    def popen(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, poll=lambda: 0)

    monkeypatch.setattr(launch.subprocess, "Popen", popen)
    monkeypatch.setattr(launch, "terminate_process_group", lambda _: None)
    with pytest.raises(RuntimeError):
        launch.main(recipe)
    assert len(calls) == 1
    assert manifest(tmp_path)["status"] == "failed"


@pytest.mark.parametrize("extra", [[], ["--start-idx", "2"]])
def test_duplicate_shard_or_worker_ids_rejected_without_mutation(
    recipe, tmp_path, monkeypatch, extra
):
    calls = mock_worker(monkeypatch)
    launch.main(recipe)
    out = tmp_path / "out"
    before = {str(path): path.read_bytes() for path in out.rglob("*") if path.is_file()}
    with pytest.raises(ValueError):
        launch.main(recipe + extra)
    assert len(calls) == 1
    assert before == {str(path): path.read_bytes() for path in out.rglob("*") if path.is_file()}


def test_successful_merge_creates_complete_dataset_manifest(recipe, tmp_path, monkeypatch):
    mock_worker(monkeypatch)
    assert launch.main([arg for arg in recipe if arg != "--no-merge"]) == 0
    assert (tmp_path / "out/annotations.json").is_file()
    assert json.loads((tmp_path / "out/manifest.json").read_text())["status"] == "complete"


def test_runtime_probe_reads_only_marked_json(monkeypatch):
    def run(command, **kwargs):
        assert command[:2] == ["/worker/python.sh", "-c"]
        assert "/isaac-sim/VERSION" in command[2]
        assert kwargs["timeout"] == 60 and kwargs["check"]
        return SimpleNamespace(stdout='wrapper banner\nISAAC_RUNTIME={"isaacsim":"6.0"}\n')

    monkeypatch.setattr(launch.subprocess, "run", run)
    assert launch.probe_runtime("/worker/python.sh") == {"isaacsim": "6.0"}


@pytest.mark.parametrize("offset", [0, 7])
def test_resume_shrinks_layout_without_stale_merge_records(recipe, tmp_path, monkeypatch, offset):
    mock_worker(monkeypatch)
    launch.main(recipe + ["--gpus", "0,1"])
    # An adjacent shard must remain intact when the first shard changes layout.
    launch.main(recipe + ["--start-idx", "2", "--worker-id-offset", "10"])
    out = tmp_path / "out"
    adjacent = (out / "annotations_worker_010.json").read_bytes()
    old = {p.name: p.read_bytes() for p in out.glob("annotations_worker_00*.json")}
    assert (
        launch.main(
            [arg for arg in recipe if arg != "--no-merge"]
            + ["--resume", "--worker-id-offset", str(offset)]
        )
        == 0
    )
    merged = json.loads((out / "annotations.json").read_text())
    assert [image["id"] for image in merged["images"]] == [0, 1, 2, 3]
    assert [ann["image_id"] for ann in merged["annotations"]] == [0, 1, 2, 3]
    assert (out / "annotations_worker_010.json").read_bytes() == adjacent
    archived = {p.name: p.read_bytes() for p in (out / "resume/consumed").rglob("*.json")}
    assert archived == old
    assert {p.name for p in out.glob("annotations_worker_*.json")} == {
        f"annotations_worker_{offset:03d}.json",
        "annotations_worker_010.json",
    }


@pytest.mark.parametrize("offset", [0, 7])
def test_launcher_publishes_shared_metadata_before_any_worker(
    recipe, tmp_path, monkeypatch, offset
):
    mock_worker(monkeypatch)
    spawn = launch.subprocess.Popen

    def check_metadata(command, **kwargs):
        out = tmp_path / "out"
        for name in ("keypoints_3d.json", "camera_intrinsics.json", "render_config.yaml"):
            assert (out / name).stat().st_mode & 0o777 == 0o644
        kp = json.loads((out / "keypoints_3d.json").read_text())
        assert kp["keypoint_names"] == ["KP_000"]
        assert kp["keypoints_3d"] == [[0, 0, 0]]
        intr = json.loads((out / "camera_intrinsics.json").read_text())
        assert intr["camera_matrix"][0][0] > 0
        assert (out / "render_config.yaml").read_bytes() == (tmp_path / "recipe.yaml").read_bytes()
        return spawn(command, **kwargs)

    monkeypatch.setattr(launch.subprocess, "Popen", check_metadata)
    assert (
        launch.main(
            [arg for arg in recipe if arg != "--no-merge"] + ["--worker-id-offset", str(offset)]
        )
        == 0
    )
    merged = json.loads((tmp_path / "out/annotations.json").read_text())
    assert merged["categories"][0]["keypoints"] == ["KP_000"]


@pytest.mark.parametrize(
    "payload",
    [
        {"keypoints_3d": [[0, 0, 0]]},
        {"keypoint_names": None, "keypoints_3d": [[0, 0, 0]]},
        {"keypoint_names": [], "keypoints_3d": [[0, 0, 0]]},
        {"keypoint_names": ["KP_000", "KP_001"], "keypoints_3d": [[0, 0, 0]]},
        {"keypoint_names": ["renamed"], "keypoints_3d": [[0, 0, 0]]},
        {"keypoint_names": ["KP_000"], "keypoints_3d": [[1, 0, 0]]},
        {"keypoint_names": ["KP_000"], "keypoints_3d": [[float("nan"), 0, 0]]},
        [],
    ],
)
def test_invalid_keypoint_schema_fails_before_output_or_workers(
    recipe, tmp_path, monkeypatch, payload
):
    supplied = tmp_path / "keypoints.json"
    supplied.write_text(json.dumps(payload))
    monkeypatch.setattr(launch.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch"))
    with pytest.raises(ValueError, match="keypoint"):
        launch.main(recipe + ["--keypoints-json", str(supplied)])
    assert not (tmp_path / "out").exists()


def test_explicit_named_keypoints_are_published_unchanged(recipe, tmp_path, monkeypatch):
    payload = {"keypoint_names": ["KP_000"], "keypoints_3d": [[0, 0, 0]]}
    supplied = tmp_path / "keypoints.json"
    supplied.write_text(json.dumps(payload))
    mock_worker(monkeypatch)
    assert launch.main(recipe + ["--keypoints-json", str(supplied), "--worker-id-offset", "7"]) == 0
    published = json.loads((tmp_path / "out/keypoints_3d.json").read_text())
    assert {key: published[key] for key in payload} == payload


def test_completion_accepts_float_readback_but_not_changed_request_or_runtime(tmp_path):
    settings = IsaacSettings.from_config(
        render_config_from_mapping({"isaac": {"exposure_time": 0.01}}, str(tmp_path))
    )
    expected = {
        "requested": settings.to_dict(),
        "effective": settings.carb_settings(),
        "runtime": {"isaacsim": RUNTIME},
    }
    actual = json.loads(json.dumps(expected))
    actual["effective"]["/rtx/post/tonemap/exposureTime"] = 0.009999999776482582
    path = tmp_path / "settings_worker_000.json"
    path.write_text(json.dumps(actual))
    assert not launch.worker_settings_errors(tmp_path, SimpleNamespace(worker_id=0), expected)
    for section, key, value in [
        ("requested", "exposure_time", 0.02),
        ("runtime", "isaacsim", "other"),
        ("effective", "/rtx/post/tonemap/exposureTime", 0.02),
    ]:
        changed = json.loads(json.dumps(actual))
        changed[section][key] = value
        path.write_text(json.dumps(changed))
        assert launch.worker_settings_errors(tmp_path, SimpleNamespace(worker_id=0), expected)


def test_replay_asset_mismatch_rejected_before_output(recipe, tmp_path, monkeypatch):
    replay = ReplayInputs(SimpleNamespace(sha256="a" * 64, source={"assets_sha256": "b" * 64}), {})
    monkeypatch.setattr(launch, "load_replay", lambda *a: replay)
    with pytest.raises(ValueError, match="asset content"):
        launch.main(recipe)
    assert not (tmp_path / "out").exists()


def test_replay_scene_mismatch_leaves_destination_retryable(recipe, tmp_path, monkeypatch):
    def reject(record):
        raise ValueError("environment scene mismatch")

    replay = SimpleNamespace(
        identity={},
        validate_assets=lambda digest: None,
        bundle=SimpleNamespace(verify_record=reject),
    )
    monkeypatch.setattr(launch, "load_replay", lambda *a: replay)
    with pytest.raises(ValueError, match="scene mismatch"):
        launch.main(recipe)
    assert not (tmp_path / "out").exists()
    monkeypatch.setattr(launch, "load_replay", lambda *a: None)
    mock_worker(monkeypatch)
    assert launch.main(recipe) == 0


def test_replay_setup_failure_marks_shard_failed(recipe, tmp_path, monkeypatch):
    mock_worker(monkeypatch)
    launch.main(recipe)
    calls = []
    monkeypatch.setattr(launch, "finish_shard", lambda *a, **kw: calls.append(kw))

    def fail(*args):
        raise ValueError("invalid replay setup")

    monkeypatch.setattr(launch, "load_replay", fail)
    args = launch.parse_args(recipe)
    with pytest.raises(ValueError, match="invalid replay setup"):
        launch.run_workers(
            args,
            [launch.WorkerRange(0, 0, 2, 0)],
            launch_inputs(tmp_path / "out", tmp_path / "missing-scenes"),
            SimpleNamespace(),
            1,
            tmp_path / "manifest",
            {"environment": {}},
        )
    assert calls and calls[0]["failed_workers"] == [0]
    assert "invalid replay setup" in calls[0]["problems"][0]


def test_sigterm_stops_workers_records_failure_and_restores_handler(recipe, tmp_path, monkeypatch):
    previous = signal.getsignal(signal.SIGTERM)
    stopped = []
    monkeypatch.setattr(
        launch.subprocess,
        "Popen",
        lambda *a, **kw: SimpleNamespace(returncode=None, poll=lambda: None),
    )
    monkeypatch.setattr(launch, "terminate_process_group", stopped.append)

    def stop(*args, **kwargs):
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler), "launcher must intercept SIGTERM for cleanup"
        handler(signal.SIGTERM, None)

    monkeypatch.setattr(launch, "supervise", stop)
    with pytest.raises(SystemExit) as exc:
        launch.main(recipe)
    assert exc.value.code == 128 + signal.SIGTERM
    assert len(stopped) == 1
    assert manifest(tmp_path)["status"] == "failed"
    assert signal.getsignal(signal.SIGTERM) == previous


def test_sigterm_is_ignored_during_cleanup_after_another_failure(recipe, tmp_path, monkeypatch):
    previous = signal.getsignal(signal.SIGTERM)
    started = []

    def popen(*args, **kwargs):
        if started:
            raise OSError("second worker spawn failed")
        proc = SimpleNamespace(returncode=None, poll=lambda: None)
        started.append(proc)
        return proc

    def cleanup(proc):
        assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN

    monkeypatch.setattr(launch.subprocess, "Popen", popen)
    monkeypatch.setattr(launch, "terminate_process_group", cleanup)
    with pytest.raises(OSError, match="second worker spawn failed"):
        launch.main(recipe + ["--gpus", "0,1"])
    assert not any("cleanup:" in problem for problem in manifest(tmp_path)["problems"])
    assert signal.getsignal(signal.SIGTERM) == previous


@pytest.mark.parametrize(
    "options",
    [
        ["--environment-gain", "1"],
        ["--environment-root", "bundle"],
        [
            "--environment-root",
            "bundle",
            "--environment-gain",
            "0",
            "--environment-adapter",
            "blender42-isaac601",
        ],
        [
            "--environment-root",
            "bundle",
            "--environment-gain",
            "nan",
            "--environment-adapter",
            "blender42-isaac601",
        ],
        [
            "--environment-root",
            "bundle",
            "--environment-gain",
            "1",
            "--environment-adapter",
            "unsupported",
        ],
    ],
)
def test_invalid_replay_options_report_cli_error(options, capsys):
    with pytest.raises(SystemExit) as exc:
        launch.parse_args([*MINIMAL, *options])
    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert "error:" in stderr
    assert "environment" in stderr
    assert "Traceback" not in stderr


def test_valid_replay_options_parse():
    args = launch.parse_args(
        [
            *MINIMAL,
            "--environment-root",
            "bundle",
            "--environment-gain",
            "1",
            "--environment-adapter",
            "blender42-isaac601",
        ]
    )
    assert args.environment_gain == 1.0
    assert args.environment_root == "bundle"


def test_isaac_python_comes_from_flag_then_environment_then_container(monkeypatch):
    monkeypatch.setenv(launch.ISAAC_PYTHON_ENV, "/env/python.sh")
    assert launch.resolve_isaac_python("/flag/python.sh") == "/flag/python.sh"
    assert launch.resolve_isaac_python(None) == "/env/python.sh"
    monkeypatch.delenv(launch.ISAAC_PYTHON_ENV)
    monkeypatch.setattr(launch, "DEFAULT_ISAAC_PYTHON", "/missing/python.sh")
    assert launch.resolve_isaac_python(None) is None


def test_missing_isaac_python_fails_before_output(recipe, tmp_path, monkeypatch):
    monkeypatch.delenv(launch.ISAAC_PYTHON_ENV, raising=False)
    monkeypatch.setattr(launch, "DEFAULT_ISAAC_PYTHON", str(tmp_path / "missing.sh"))
    argv = [arg for arg in recipe if arg not in ("--isaac-python", "isaac-python")]
    with pytest.raises(SystemExit, match="DOPER_ISAAC_PYTHON"):
        launch.main(argv)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("version", ["5.1.0", "4.5.0-rc.1", "unknown"])
def test_old_or_unknown_isaac_rejected_before_output(recipe, tmp_path, monkeypatch, version):
    monkeypatch.setattr(launch, "probe_runtime", lambda _: {"isaacsim": version})
    with pytest.raises(SystemExit, match=f"6.0 or newer.*{version}"):
        launch.main(recipe)
    assert not (tmp_path / "out").exists()


def test_unprobeable_isaac_python_names_the_flag(recipe, tmp_path, monkeypatch):
    def fail(_):
        raise OSError("no such file")

    monkeypatch.setattr(launch, "probe_runtime", fail)
    with pytest.raises(SystemExit, match="--isaac-python"):
        launch.main(recipe)
    assert not (tmp_path / "out").exists()


def test_python_bin_alias_and_relative_paths_resolve_against_cwd(recipe, tmp_path, monkeypatch):
    calls = mock_worker(monkeypatch)
    monkeypatch.chdir(tmp_path)
    argv = [arg for arg in recipe if arg not in ("--isaac-python", "isaac-python")]
    index = argv.index("--outdir")
    argv[index + 1] = "out"
    index = argv.index("--config")
    argv[index + 1] = "recipe.yaml"
    assert launch.main([*argv, "--python-bin", "/opt/isaac/python.sh"]) == 0
    flags = values(calls[0])
    assert calls[0][0] == "/opt/isaac/python.sh"
    assert flags["--outdir"] == str(tmp_path / "out")
    assert flags["--config"] == str(tmp_path / "recipe.yaml")
    assert manifest(tmp_path)["config_path"] == "recipe.yaml"


def test_missing_object_usd_fails_before_output(recipe, tmp_path):
    with pytest.raises(SystemExit, match="Object USD not found"):
        launch.main([*recipe, "--usd", str(tmp_path / "missing.usda")])
    assert not (tmp_path / "out").exists()


def test_configured_object_usd_that_is_missing_fails_before_output(recipe, tmp_path):
    config = tmp_path / "recipe.yaml"
    payload = yaml.safe_load(config.read_text())
    payload["object"]["usd_path"] = str(tmp_path / "missing.usda")
    config.write_text(yaml.safe_dump(payload))
    with pytest.raises(SystemExit, match="Object USD not found"):
        launch.main(recipe)
    assert not (tmp_path / "out").exists()
