# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Novel-view sampling, the NRE rig trajectory and the NRE CLI render backend."""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from nurec_fixtures import (
    NUREC_TEMPLATE,
    SCENE_SOURCES,
    SENSOR_RIG,
    nurec_recipe_data,
    rig_trajectories,
    rigid,
    write_reconstruction,
    write_usdz,
)

from doper import cli
from doper.nurec import render
from doper.nurec.capture import PinholeCamera
from doper.nurec.render import (
    Mount,
    NovelView,
    NreCliBackend,
    RenderHostError,
    RenderResult,
    sample_novel_views,
    write_rig_trajectory,
)
from doper.nurec.scene import CapabilityError, MetricAlignmentError, NuRecScene

DIGEST = "nvcr.io/nvidia/nre/nre-ga@sha256:" + "1c" * 32
CAMERA = PinholeCamera(8, 6, 5.0, 5.0, 3.5, 2.5, True)


def _scene(*, alignment="metric_anchor", scale=1.0, world_to_nre=None):
    names = ("a.jpg", "b.jpg", "c.jpg")
    world_to_nre = np.eye(4) if world_to_nre is None else world_to_nre
    return NuRecScene(
        artifact=Path("last.usdz"), artifact_sha256="a" * 64, world_to_nre=world_to_nre,
        cameras={name: rigid(70 + i) for i, name in enumerate(names)}, camera=CAMERA,
        scene_units_per_metre=scale if alignment == "metric_anchor" else None,
        alignment=alignment, reconstruction_id="recon-1",
        lineages={"a.jpg": "s1", "b.jpg": "s1", "c.jpg": "s2"},
    )  # fmt: skip


def test_sampler_rejects_near_source():
    scene = _scene()
    # The seed camera itself is a source: a draw within 1 degree of it (the translation is at
    # most ~0.9 mm here) is rejected and redrawn.
    kwargs = {"seed": 5, "count": 40, "seed_views": ["a.jpg", "c.jpg"], "perturb_deg": 1.2,
              "perturb_m": 0.0005}  # fmt: skip
    extra = [rigid(90)]
    views = sample_novel_views(scene, extra, **kwargs)
    assert len(views) == 40
    assert sum(view.rejected_before for view in views) > 0
    sources = np.array([*scene.cameras.values(), *extra])
    for i, view in enumerate(views):
        assert view.index == i
        assert view.seed_view == ("a.jpg", "c.jpg")[i % 2]
        assert view.lineage == ("s1", "s2")[i % 2]
        assert view.reconstruction_id == "recon-1"
        cosine = (np.trace(sources[:, :3, :3].transpose(0, 2, 1) @ view.T_nre_cam[:3, :3],
                           axis1=1, axis2=2) - 1) / 2  # fmt: skip
        rotation = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        translation = np.linalg.norm(sources[:, :3, 3] - view.T_nre_cam[:3, 3], axis=1) * 1000
        assert not np.any((rotation < 1.0) & (translation < 2.0))
        assert view.min_source_rotation_deg == pytest.approx(rotation.min(), abs=1e-6)
        assert view.min_source_translation_mm == pytest.approx(translation.min(), abs=1e-9)
        np.testing.assert_allclose(view.T_cam_nre @ view.T_nre_cam, np.eye(4), atol=1e-12)
    # Deterministic for a seed.
    again = sample_novel_views(scene, extra, **kwargs)
    assert all(np.array_equal(a.T_nre_cam, b.T_nre_cam) for a, b in zip(views, again, strict=True))


def test_sampler_translation_threshold_is_metric():
    # At 1000 scene units per meter a 0.5-unit offset is 0.5 mm: inside the 2 mm threshold.
    scene = _scene(scale=1000.0)
    views = sample_novel_views(scene, seed=1, count=5, seed_views=["a.jpg"], perturb_deg=0.0,
                               perturb_m=0.5, reject_deg=1.0, reject_mm=0.4)  # fmt: skip
    assert all(view.min_source_translation_mm >= 0.4 for view in views)
    with pytest.raises(ValueError, match="consecutive"):
        sample_novel_views(scene, seed=1, count=1, seed_views=["a.jpg"], perturb_deg=0.0,
                           perturb_m=0.5, reject_deg=1.0, reject_mm=1000.0)  # fmt: skip


def test_sampler_needs_a_metric_scene_and_known_views():
    with pytest.raises(MetricAlignmentError, match="visual_only"):
        sample_novel_views(_scene(alignment="visual_only"), seed=1, count=1, seed_views=["a.jpg"])
    with pytest.raises(ValueError, match="z.jpg"):
        sample_novel_views(_scene(), seed=1, count=1, seed_views=["z.jpg"])
    with pytest.raises(ValueError, match="4x4"):
        sample_novel_views(_scene(), [np.eye(3)], seed=1, count=1, seed_views=["a.jpg"])


def test_trajectory_rig_round_trip(tmp_path):
    world_to_nre = np.eye(4)
    world_to_nre[:3, 3] = [-0.4, 0.05, -0.5]
    scene = _scene(world_to_nre=world_to_nre)
    views = sample_novel_views(scene, seed=3, count=4, seed_views=["a.jpg", "b.jpg"])
    path = write_rig_trajectory(views, scene, CAMERA, tmp_path / "trajectory.json")
    record = json.loads(path.read_text())
    assert list(record) == ["T_world_base", "world_to_nre", "rig_trajectories",
                            "camera_calibrations", "lidar_calibrations"]  # fmt: skip
    rig = record["rig_trajectories"][0]
    np.testing.assert_array_equal(record["world_to_nre"]["matrix"], world_to_nre)
    assert rig["T_rig_world_timestamps_us"] == [0, 100000, 200000, 300000]
    assert rig["cameras_frame_timestamps_us"]["camera1@input"] == [
        [t, t] for t in (0, 100000, 200000, 300000)
    ]
    pairs = rig["cameras_frame_T_rig_worlds"]["camera1@input"]
    for view, rig_pose, pair in zip(views, rig["T_rig_worlds"], pairs, strict=True):
        assert pair == [rig_pose, rig_pose]
        back = world_to_nre @ np.array(rig_pose) @ SENSOR_RIG
        np.testing.assert_allclose(back, view.T_nre_cam, atol=1e-12)
    calibration = record["camera_calibrations"]["camera1@input"]
    assert calibration["T_sensor_rig"] == SENSOR_RIG.tolist()
    parameters = calibration["camera_model"]["parameters"]
    assert calibration["camera_model"]["type"] == "opencv-pinhole"
    assert parameters["shutter_type"] == "GLOBAL"
    assert parameters["resolution"] == [8, 6]
    assert parameters["focal_length"] == [5.0, 5.0]
    assert parameters["principal_point"] == [3.5, 2.5]
    assert parameters["radial_coeffs"] == [0.0] * 6
    other = _scene()
    object.__setattr__(other, "reconstruction_id", "recon-2")
    with pytest.raises(ValueError, match="reconstruction"):
        write_rig_trajectory(views, other, CAMERA, tmp_path / "other.json")
    with pytest.raises(ValueError, match="rectified"):
        write_rig_trajectory(views, scene, PinholeCamera(8, 6, 5.0, 5.0, 3.5, 2.5, False),
                             tmp_path / "unrectified.json")  # fmt: skip


def test_backend_requires_digest():
    for image in ("nvcr.io/nvidia/nre/nre-ga:26.04", "nre-ga", "nre@sha256:abc"):
        with pytest.raises(ValueError, match="@sha256:"):
            NreCliBackend(image)
    NreCliBackend(DIGEST)


def test_backend_command_matches_the_reference_render(tmp_path):
    out = tmp_path / "render"
    out.mkdir()
    cache = Mount(tmp_path / "cache", "/workdir/cache")
    backend = NreCliBackend(DIGEST, mounts=[cache], env={"TORCH_HOME": "/workdir/cache"})
    scene = _scene()
    object.__setattr__(scene, "artifact", tmp_path / "artifacts" / "last.usdz")
    argv = backend.command(scene, out / "trajectory.json", out)
    assert argv == [
        "docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}", "--shm-size", "64g",
        "--gpus", "all", "-v", f"{tmp_path / 'artifacts'}:/artifact:ro", "-v", f"{out}:/render",
        "-v", f"{tmp_path / 'cache'}:/workdir/cache", "-e", "TORCH_HOME=/workdir/cache", DIGEST,
        "render", "--artifact-path", "/artifact/last.usdz", "--output-dir", "/render/rgb",
        "--custom-rig-trajectory", "/render/trajectory.json", "--no-replicate-training-views",
        "--camera-id", "camera1", "--image-scale", "1.0", "--image-format", "png",
        "--frame-naming", "contiguous-output-index",
    ]  # fmt: skip
    with pytest.raises(ValueError, match="inside"):
        backend.command(scene, tmp_path / "trajectory.json", out)


def _write_trajectory(tmp_path, count=3):
    scene = _scene()
    views = sample_novel_views(scene, seed=3, count=count, seed_views=["a.jpg"])
    out = tmp_path / "render"
    out.mkdir(exist_ok=True)
    return scene, write_rig_trajectory(views, scene, CAMERA, out / "trajectory.json"), out


def _png(path, width=8, height=6):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.zeros((height, width, 3), np.uint8))


def test_render_output_refuses_missing_frames(tmp_path):
    frames = tmp_path / "rgb" / "camera1"
    for i in (0, 1):
        _png(frames / f"{i:06d}.png")
    with pytest.raises(ValueError, match="000002.png"):
        RenderResult.collect(frames, count=3, width=8, height=6)
    _png(frames / "000002.png", width=16)
    with pytest.raises(ValueError, match="16x6"):
        RenderResult.collect(frames, count=3, width=8, height=6)
    _png(frames / "000002.png")
    _png(frames / "000003.png")
    with pytest.raises(ValueError, match="000003.png"):
        RenderResult.collect(frames, count=3, width=8, height=6)
    (frames / "000003.png").unlink()
    result = RenderResult.collect(frames, count=3, width=8, height=6)
    assert [p.name for p in result.frames] == ["000000.png", "000001.png", "000002.png"]
    assert (result.width, result.height) == (8, 6)


def test_render_output_accepts_the_nre_timestamps_sidecar(tmp_path):
    # The rows mirror the sidecar NRE wrote next to the frames on the qualification host and in
    # the bulk render (file_name, render_frame_idx, frame_start/end_timestamp_us).
    frames = tmp_path / "rgb" / "camera1"
    for i in range(2):
        _png(frames / f"{i:06d}.png")
    rows = [
        {
            "file_name": f"{i:06d}.png",
            "render_frame_idx": i,
            "frame_start_timestamp_us": i * 100000,
            "frame_end_timestamp_us": i * 100000,
        }
        for i in range(2)
    ]
    (frames / render.TIMESTAMPS_NAME).write_text(json.dumps(rows))
    result = RenderResult.collect(frames, count=2, width=8, height=6)
    assert [p.name for p in result.frames] == ["000000.png", "000001.png"]
    (frames / render.TIMESTAMPS_NAME).write_text(json.dumps(rows[:1]))
    with pytest.raises(ValueError, match="timestamps.json"):
        RenderResult.collect(frames, count=2, width=8, height=6)
    (frames / render.TIMESTAMPS_NAME).write_text(json.dumps(rows[::-1]))
    with pytest.raises(ValueError, match="timestamps.json"):
        RenderResult.collect(frames, count=2, width=8, height=6)
    extra = {**rows[1], "file_name": "000002.png", "render_frame_idx": 2}
    (frames / render.TIMESTAMPS_NAME).write_text(json.dumps([*rows, extra]))
    with pytest.raises(ValueError, match="timestamps.json"):
        RenderResult.collect(frames, count=2, width=8, height=6)
    for malformed in (
        "{}",
        '{"file_name": "000000.png"}',
        '["000000.png", "000001.png"]',
        "[1, 2]",
        '[{"name": "000000.png"}]',
        "not json",
    ):
        (frames / render.TIMESTAMPS_NAME).write_text(malformed)
        with pytest.raises(ValueError, match="not an NRE frame timestamps list"):
            RenderResult.collect(frames, count=2, width=8, height=6)
    (frames / render.TIMESTAMPS_NAME).unlink()
    (frames / render.TIMESTAMPS_NAME).mkdir()
    with pytest.raises(ValueError, match="not an NRE frame timestamps list"):
        RenderResult.collect(frames, count=2, width=8, height=6)
    (frames / render.TIMESTAMPS_NAME).rmdir()
    (frames / render.TIMESTAMPS_NAME).write_text(json.dumps(rows))
    (frames / "notes.txt").write_text("x")
    with pytest.raises(ValueError, match="notes.txt"):
        RenderResult.collect(frames, count=2, width=8, height=6)


def test_backend_refuses_a_host_without_docker_or_gpu(tmp_path, monkeypatch):
    scene, trajectory, out = _write_trajectory(tmp_path)
    monkeypatch.setattr(render.shutil, "which", lambda name: None)
    with pytest.raises(RenderHostError, match="docker"):
        NreCliBackend(DIGEST).render(scene, trajectory, out)
    no_gpu = lambda name: None if name == "nvidia-smi" else f"/usr/bin/{name}"  # noqa: E731
    monkeypatch.setattr(render.shutil, "which", no_gpu)
    with pytest.raises(RenderHostError, match="GPU"):
        NreCliBackend(DIGEST).render(scene, trajectory, out)


def test_backend_render_runs_and_verifies(tmp_path, monkeypatch):
    scene, trajectory, out = _write_trajectory(tmp_path)
    monkeypatch.setattr(render.shutil, "which", lambda name: f"/usr/bin/{name}")
    calls = []

    def run(argv, *, written):
        calls.append(argv)
        for i in range(written):
            _png(out / "rgb" / "camera1" / f"{i:06d}.png")
        return 0

    backend = NreCliBackend(DIGEST, runner=lambda argv: run(argv, written=3))
    result = backend.render(scene, trajectory, out)
    assert len(result.frames) == 3
    assert calls[0][:2] == ["docker", "run"]
    # A rerun refuses existing output rather than mixing renders.
    with pytest.raises(FileExistsError, match="rgb.*--force"):
        backend.render(scene, trajectory, out)
    import shutil

    shutil.rmtree(out / "rgb")
    partial = NreCliBackend(DIGEST, runner=lambda argv: run(argv, written=2))
    with pytest.raises(ValueError, match="000002.png"):
        partial.render(scene, trajectory, out)
    shutil.rmtree(out / "rgb")
    failing = NreCliBackend(DIGEST, runner=lambda argv: 3)
    with pytest.raises(render.RenderFailedError, match="exit status 3"):
        failing.render(scene, trajectory, out)
    with pytest.raises(CapabilityError, match="depth"):
        backend.render(scene, trajectory, out, outputs=("rgb", "depth"))


def test_novel_view_record_round_trip():
    view = sample_novel_views(_scene(), seed=1, count=1, seed_views=["b.jpg"])[0]
    record = view.to_record()
    assert record["seed_view"] == "b.jpg"
    assert record["lineage"] == "s1"
    assert record["reconstruction_id"] == "recon-1"
    assert NovelView.from_record(json.loads(json.dumps(record))).to_record() == record


def _prepare_render(root: Path, *extra: str) -> list[str]:
    return ["prepare", str(NUREC_TEMPLATE), "--stage", "render", "--root", str(root), *extra]


def test_prepare_render_dry_run_touches_nothing(tmp_path, capsys):
    argv = _prepare_render(tmp_path / "root", "--artifact", str(tmp_path / "last.usdz"))
    assert cli.main([*argv, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "sample novel views" in out
    assert "trajectory.json" in out
    assert "nre-ga@sha256:97f43e71" in out
    assert not (tmp_path / "root").exists()


def test_prepare_render_needs_artifact(tmp_path, capsys):
    assert cli.main([*_prepare_render(tmp_path), "--dry-run"]) == 1
    assert "--artifact" in capsys.readouterr().out


def test_artifact_flag_refused_elsewhere(tmp_path, capsys):
    argv = ["prepare", str(NUREC_TEMPLATE), "--stage", "capture", "--bundle", str(tmp_path),
            "--root", str(tmp_path), "--artifact", str(tmp_path), "--dry-run"]  # fmt: skip
    assert cli.main(argv) == 1
    assert "--artifact" in capsys.readouterr().out


def _prepared_root(tmp_path):
    """A root with a capture frame map and a reconstruction, plus a matching artifact."""
    root = tmp_path / "root"
    recon = write_reconstruction(root / "reconstruction")
    frames_dir = root / "capture" / "frames"
    frames_dir.mkdir(parents=True)
    frames = recon["frames"]
    record = {"max_long_edge": 1280, "interpolation": "INTER_AREA",
              "source_camera": frames.source_camera.as_record(),
              "camera": frames.camera.as_record(),
              "frames": [{"name": f.name, "source": f.source,
                          "source_sha256": f.source_sha256, "sha256": f.sha256,
                          "lineage": f.lineage} for f in frames.frames]}  # fmt: skip
    (frames_dir / "frames.json").write_text(json.dumps(record))
    (root / "capture" / "READY.json").write_text(
        json.dumps(
            {
                "stage": "capture",
                "robot_reference": False,
                "flags": {"distortion": "zero", "robot_reference": False},
            }
        )
    )
    poses = [np.linalg.inv(recon["cam_from_world"][name]) for name in SCENE_SOURCES]
    artifact = write_usdz(tmp_path / "artifact" / "last.usdz", rig_trajectories(poses))
    return root, artifact


def _small_recipe(tmp_path: Path) -> str:
    import yaml

    data = nurec_recipe_data()
    data["render"].update(count=6, seed_views=[0, 2, 3])
    path = tmp_path / "nurec-few-views.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


def test_prepare_render_plans_then_refuses_without_docker(tmp_path, capsys, monkeypatch):
    root, artifact = _prepared_root(tmp_path)
    monkeypatch.setattr(render.shutil, "which", lambda name: None)
    argv = ["prepare", _small_recipe(tmp_path), "--stage", "render", "--root", str(root),
            "--artifact", str(artifact)]  # fmt: skip
    assert cli.main(argv) == 1
    assert "docker" in capsys.readouterr().out
    plan = root / "render"
    views = json.loads((plan / "views.json").read_text())
    assert [v["seed_view"] for v in views] == [SCENE_SOURCES[i] for i in (0, 2, 3, 0, 2, 3)]
    sampling = json.loads((plan / "sampling.json").read_text())
    assert (sampling["count"], sampling["seed"], sampling["source_count"]) == (6, 891, 4)
    command = json.loads((plan / "command.json").read_text())
    assert command[command.index("--custom-rig-trajectory") + 1] == "/render/trajectory.json"
    rig = json.loads((plan / "trajectory.json").read_text())["rig_trajectories"][0]
    assert rig["T_rig_world_timestamps_us"][-1] == 500000
    # A rerun with the same inputs keeps the plan; it still refuses to render without docker.
    assert cli.main(argv) == 1
    assert "already planned" in capsys.readouterr().out
