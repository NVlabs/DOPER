# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native replay must not expose the large shared bundle to Kit's folder watcher."""

import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from rendering.environment_bundle import (
    BundleReader,
    ReplayInputs,
    publish_bytes,
    publish_record,
    seal_bundle,
)
from rendering.isaac import isaac_parallel as launch
from rendering.isaac.asset_cache import local_replay


def test_staging_preserves_bytes_identity_and_cleans_up_without_listing_source(
    tmp_path, monkeypatch, native_bundle
):
    root, record, source = native_bundle
    other_record = {**record, "index": 1}
    other_bytes = b"#usda 1.0\n# another scene\n"
    other_layer = f"layers/{hashlib.sha256(other_bytes).hexdigest()}.usda"
    publish_bytes(root / other_layer, other_bytes)
    publish_record(
        root,
        other_record,
        source=source,
        layer=other_layer,
        dependencies=["textures/asset.hdr"],
        light_count=2,
        ground=False,
    )
    (root / "manifest.json").unlink()
    seal_bundle(root, [record, other_record], source=source)
    replay = ReplayInputs(
        BundleReader(root, usd_sha256=source["usd_sha256"], seed=42), {"gain": 2000}
    )
    metadata = replay.bundle.verify_record(record)
    original_scandir = os.scandir

    def guarded_scandir(path):
        if not isinstance(path, int):
            assert not Path(path).is_relative_to(root), "must not enumerate source directories"
        return original_scandir(path)

    monkeypatch.setattr(os, "scandir", guarded_scandir)
    with local_replay(replay, [record], directory=tmp_path) as staged:
        assert staged.identity == replay.identity
        assert staged.bundle is not replay.bundle
        cached = staged.bundle.root
        assert cached != root
        assert (cached / "manifest.json").read_bytes() == (root / "manifest.json").read_bytes()
        assert not (cached / other_layer).exists()
        assert not (cached / "records/000001.json").exists()
        assert set(staged.bundle.manifest["records"]) == {"0", "1"}
        for name in ["records/000000.json", *metadata["files"]]:
            assert (cached / name).read_bytes() == (root / name).read_bytes()
            assert not (cached / name).is_symlink()
        assert staged.bundle.verify_record(record) == metadata
    assert not cached.exists()
    assert (root / other_layer).exists()


def test_corrupt_copy_rejected_even_after_source_verified(tmp_path, monkeypatch, native_bundle):
    root, record, source = native_bundle
    replay = ReplayInputs(BundleReader(root, usd_sha256=source["usd_sha256"], seed=42), {})
    replay.bundle.verify_record(record)
    real_copy = shutil.copyfile
    copied = []

    def corrupt(source_path, destination):
        real_copy(source_path, destination)
        copied.append(Path(destination))
        if str(destination).endswith(".hdr"):
            Path(destination).write_bytes(b"corrupted")

    monkeypatch.setattr(shutil, "copyfile", corrupt)
    with (
        pytest.raises(ValueError, match="hash"),
        local_replay(replay, [record], directory=tmp_path),
    ):
        pytest.fail("corrupt data reached renderer")
    assert copied and all(not path.exists() for path in copied)


def test_legacy_does_not_create_cache(tmp_path):
    with local_replay(None, [], directory=tmp_path) as staged:
        assert staged is None
        assert list(tmp_path.iterdir()) == []


def test_insufficient_disk_fails_before_copy(tmp_path, monkeypatch, native_bundle):
    root, record, source = native_bundle
    replay = ReplayInputs(BundleReader(root, usd_sha256=source["usd_sha256"], seed=42), {})
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with (
        pytest.raises(OSError, match="staging needs"),
        local_replay(replay, [record], directory=tmp_path),
    ):
        pytest.fail("must fail before renderer starts")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bundle"]


@pytest.mark.parametrize("startup_fails", [False, True])
def test_launcher_owns_cache_until_worker_exit(tmp_path, monkeypatch, native_bundle, startup_fails):
    root, record, source = native_bundle
    replay = ReplayInputs(BundleReader(root, usd_sha256=source["usd_sha256"], seed=42), {})
    monkeypatch.setattr(launch, "load_replay", lambda *a: replay)
    monkeypatch.setattr(launch, "validate_applied_readbacks", lambda *a, **kw: None)
    monkeypatch.setattr(
        launch, "local_replay", lambda env, records: local_replay(env, records, directory=tmp_path)
    )
    for name in ("worker_output_errors", "worker_settings_errors"):
        monkeypatch.setattr(launch, name, lambda *a: [])
    monkeypatch.setattr(launch, "finish_shard", lambda *a, **kw: None)
    monkeypatch.setattr(launch, "count_shard_images", lambda *a: 1)
    scenes = tmp_path / "scenes.jsonl"
    scenes.write_text(json.dumps(record) + "\n")
    args = SimpleNamespace(
        isaac_python="python",
        environment_root=str(root),
        environment_gain=2000,
        environment_adapter="blender42-isaac601",
        seed=42,
        stall_timeout_sec=900,
        startup_grace_sec=1200,
        resume=False,
        max_restarts=0,
        start_idx=0,
        num_images=1,
        progress_interval_sec=0,
    )
    cached = []

    def start(command, **kwargs):
        path = Path(command[command.index("--environment-root") + 1])
        cached.append(path)
        assert path != root and path.is_dir()
        assert BundleReader(path, usd_sha256=source["usd_sha256"], seed=42).verify_record(record)
        if startup_fails:
            raise RuntimeError("worker start failed")

        def poll():
            assert path.is_dir(), "cache removed while worker is alive"
            return 0

        return SimpleNamespace(poll=poll, returncode=0)

    monkeypatch.setattr(launch.subprocess, "Popen", start)

    def run():
        paths = launch.LaunchInputs(
            outdir=tmp_path,
            scenes_path=scenes,
            config_path=Path("config"),
            usd_path="object",
            keypoints_path=None,
        )
        launch.run_workers(
            args,
            [launch.WorkerRange(0, 0, 1, 0)],
            paths,
            SimpleNamespace(images_per_sample=1),
            1,
            tmp_path / "manifest",
            {"environment": replay.identity},
        )

    if startup_fails:
        with pytest.raises(RuntimeError, match="worker start failed"):
            run()
    else:
        run()
    assert cached and all(not path.exists() for path in cached)
    assert args.environment_root == str(root)
