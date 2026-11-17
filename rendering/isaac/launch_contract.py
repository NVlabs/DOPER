# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only launch authorization shared by the Isaac launcher and worker bootstrap."""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
from pathlib import Path
from typing import Any

from rendering.environment_bundle import include_replay_identity
from rendering.isaac.inputs import WorkerInputs
from rendering.run_manifest import build_render_identity, validate_resume_identity
from rendering.shard_io import load_shards, sha256_file

RENDERER_VERSION = "isaac-scenes-v2"
ISAAC_VERSION_FILE = "/isaac-sim/VERSION"
PROBE_PREFIX = "ISAAC_RUNTIME="
PROBE_TIMEOUT_SEC = 60
# Runs under the worker interpreter; must not import Kit.
PROBE_CODE = f"""import importlib.metadata, json
from pathlib import Path
try:
    version = importlib.metadata.version('isaacsim')
except importlib.metadata.PackageNotFoundError:
    version = Path({ISAAC_VERSION_FILE!r}).read_text().strip()
print({PROBE_PREFIX!r} + json.dumps({{'isaacsim': version}}))
"""


def runtime_version() -> dict[str, str]:
    """Identify Isaac Sim without starting Kit.

    Raises:
        ValueError: The version is empty.
    """
    try:
        version = importlib.metadata.version("isaacsim")
    except importlib.metadata.PackageNotFoundError:
        version = Path(ISAAC_VERSION_FILE).read_text().strip()
    if not version:
        raise ValueError("Isaac runtime version is empty")
    return {"isaacsim": version}


def probe_runtime(python_bin: str) -> dict[str, str]:
    """Identify the worker interpreter's Isaac Sim without importing Kit or modifying output.

    Raises:
        ValueError: The probe did not print exactly one valid version record.
        subprocess.CalledProcessError: The interpreter failed.
    """
    result = subprocess.run(
        [python_bin, "-c", PROBE_CODE],
        capture_output=True,
        text=True,
        timeout=PROBE_TIMEOUT_SEC,
        check=True,
    )
    records = [
        line[len(PROBE_PREFIX) :]
        for line in result.stdout.splitlines()
        if line.startswith(PROBE_PREFIX)
    ]
    if len(records) != 1:
        raise ValueError("Isaac runtime probe did not return exactly one version record")
    runtime = json.loads(records[0])
    if (
        not isinstance(runtime, dict)
        or not isinstance(runtime.get("isaacsim"), str)
        or not runtime["isaacsim"]
    ):
        raise ValueError("Isaac runtime probe returned invalid version metadata")
    return runtime


def _declared_attempt(inputs: WorkerInputs) -> dict[str, Any]:
    """Return the one running Isaac shard manifest that declares this worker as launched."""
    args = inputs.args
    matches = [
        (manifest, worker)
        for manifest in load_shards(Path(args.outdir))
        for worker in manifest.get("shard", {}).get("workers", [])
        if worker.get("worker_id") == args.worker_id
    ]
    if len(matches) != 1:
        raise ValueError("worker requires one unambiguous launcher-declared shard attempt")
    manifest, worker = matches[0]
    expected_worker = {
        "worker_id": args.worker_id,
        "start_idx": args.start_idx,
        "end_idx": args.end_idx,
        "gpu": args.gpu_device,
    }
    if any(worker.get(key) != value for key, value in expected_worker.items()):
        raise ValueError("worker range or GPU differs from launcher declaration")
    if manifest.get("status") != "running" or manifest.get("renderer") != "isaac":
        raise ValueError("worker requires an active Isaac shard attempt")
    shard = manifest["shard"]
    if not shard["start_idx"] <= args.start_idx < args.end_idx <= shard["end_idx"]:
        raise ValueError("worker range lies outside its declared shard")
    scene_file = Path(args.scenes)
    scenes = manifest.get("scenes", {})
    if scenes.get("file") != scene_file.name or scenes.get("sha256") != sha256_file(scene_file):
        raise ValueError("scene file differs from launcher provenance")
    return manifest


def validate_worker_launch(
    inputs: WorkerInputs, *, runtime: dict[str, str] | None = None
) -> dict[str, Any]:
    """Require an active launcher attempt matching this worker and its immutable inputs.

    Call after ``load_inputs`` and before ``SimulationApp`` or any output writes. Standalone
    workers cannot authorize their own initial launch or resume. Local object and
    configuration inputs and catalog membership are rechecked; the launcher's content manifest
    is reused for asset packs, which must stay immutable throughout the attempt.

    Args:
        inputs: Validated worker inputs.
        runtime: Isaac runtime identity; probed from this interpreter when ``None``.

    Returns:
        The renderer settings the launcher recorded (requested, effective and runtime).

    Raises:
        ValueError: The attempt, settings, runtime or inputs differ from the launch record.
    """
    args, cfg = inputs.args, inputs.cfg
    manifest = _declared_attempt(inputs)
    runtime = runtime_version() if runtime is None else runtime
    expected_settings = {
        "requested": inputs.settings.to_dict(),
        "effective": inputs.settings.carb_settings(),
        "runtime": runtime,
    }
    include_replay_identity(expected_settings, inputs.environment)
    if manifest.get("renderer_settings") != expected_settings:
        raise ValueError("worker settings or runtime differ from launcher provenance")
    asset_manifest = manifest.get("render_identity", {}).get("assets")
    if not isinstance(asset_manifest, dict):
        raise ValueError("launcher identity is missing its persisted asset manifest")
    identity = build_render_identity(
        cfg=cfg,
        catalog=inputs.catalog,
        seed=args.seed,
        renderer="isaac",
        renderer_version=RENDERER_VERSION,
        usd_path=inputs.usd_path,
        keypoints_path=cfg.resolve(args.keypoints_json or cfg.keypoints_json),
        renderer_settings=expected_settings,
        asset_manifest=asset_manifest,
    )
    if inputs.environment is not None:
        inputs.environment.validate_assets(identity.to_dict()["assets"]["content_sha256"])
    validate_resume_identity(Path(args.outdir), identity, require_existing=True)
    return expected_settings
