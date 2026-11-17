#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isaac Sim worker entry point; ``isaac_parallel.py`` creates its validated launch inputs.

Runs under Isaac Sim's Python. Kit (``isaacsim``, ``omni``, ``carb``) is imported only after
the inputs and the launch record have been validated.
"""

# pyright: reportMissingImports=false, reportMissingModuleSource=false

from __future__ import annotations

import argparse
import inspect
import math
import os
import sys
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

# Direct script execution must resolve shared renderer modules from this checkout.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rendering.environment_bundle import (  # noqa: E402
    add_replay_arguments,
    include_replay_identity,
)
from rendering.isaac.capture import FrameCapture  # noqa: E402
from rendering.isaac.inputs import WorkerInputs, load_inputs  # noqa: E402
from rendering.isaac.launch_contract import (  # noqa: E402
    runtime_version,
    validate_worker_launch,
)
from rendering.isaac.worker import RenderWorker  # noqa: E402
from rendering.shard_io import write_json  # noqa: E402

FAILURE_EXIT_CODE = 3


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the worker command line written by ``isaac_parallel.py``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Render config (YAML)")
    parser.add_argument("--outdir", required=True, help="Dataset directory")
    parser.add_argument("--usd", help="Object .usda (else object.usd_path from the config)")
    parser.add_argument("--keypoints-json", help="Override object.keypoints_json")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument(
        "--scenes", required=True, help="Authoritative scene records produced by the launcher"
    )
    parser.add_argument("--flush-every", type=int, default=100)
    parser.add_argument("--flush-every-sec", type=float, default=60.0)
    parser.add_argument("--resume", action="store_true")
    add_replay_arguments(parser)
    args = parser.parse_args(argv)
    if args.flush_every < 0 or not math.isfinite(args.flush_every_sec) or args.flush_every_sec < 0:
        parser.error("Checkpoint intervals must be finite and nonnegative")
    return args


def _settings_report(inputs: WorkerInputs, getter: Callable[[str], Any]) -> dict[str, Any]:
    """Requested, effective and runtime settings, written for the launcher to verify."""
    report = {
        "requested": inputs.settings.to_dict(),
        "effective": inputs.settings.verify_effective(getter),
        "runtime": runtime_version(),
    }
    return include_replay_identity(report, inputs.environment)


def render(inputs: WorkerInputs) -> None:
    """Start Kit, build the scene and render the worker's sample range.

    Any failure exits the process with status 3, because Kit's fast shutdown can skip
    surrounding handlers.
    """
    # Isaac requires SimulationApp before importing any Kit-dependent modules.
    from isaacsim import SimulationApp

    app = SimulationApp(inputs.settings.app_config(inputs.cfg.resolution, inputs.args.gpu_device))
    try:
        import carb
        import omni.replicator.core as rep

        from rendering.isaac.scene import KitScene

        settings = carb.settings.get_settings()
        rep.orchestrator.set_capture_on_play(False)
        scene = KitScene(inputs)

        def apply_settings() -> None:
            inputs.settings.apply_to_stage(
                scene.cam_prim, scene.stage.GetPrimAtPath(scene.render_product.path)
            )
            for key, value in inputs.settings.carb_settings().items():
                settings.set(key, value)

        # New-stage initialization restores Kit defaults. Apply the requested
        # renderer only after the scene and render product exist.
        for key, value in inputs.settings.carb_settings().items():
            settings.set(key, value)
        inputs.settings.apply_to_stage(
            scene.cam_prim, scene.stage.GetPrimAtPath(scene.render_product.path)
        )

        def checked_step(**kwargs: Any) -> None:  # noqa: ANN401 - orchestrator options
            rep.orchestrator.step(**kwargs)
            inputs.settings.verify_effective(settings.get)

        def prime_with_settings(**kwargs: Any) -> None:  # noqa: ANN401 - orchestrator options
            # The discarded first frame initializes Hydra's global aliases.
            # It can reset Carb even when the camera USD attributes are correct.
            rep.orchestrator.step(**kwargs)
            apply_settings()
            inputs.settings.verify_effective(settings.get)

        capture = FrameCapture(
            step=checked_step,
            prime_step=prime_with_settings if inputs.settings.has_photometric_controls else None,
            rgb=scene.rgb_annot.get_data,
            depth=scene.depth_annot.get_data,
            segmentation=scene.seg_annot.get_data,
            resolution=inputs.cfg.resolution,
            object_path=scene.obj_path,
            rt_subframes=inputs.settings.carb_settings()["/omni/replicator/RTSubframes"],
            timeout_sec=inputs.settings.capture_timeout_sec,
        )
        outdir = Path(inputs.args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        write_json(
            outdir / f"settings_worker_{inputs.args.worker_id:03d}.json",
            _settings_report(inputs, settings.get),
        )
        RenderWorker(inputs, scene, capture).run()
        inputs.settings.verify_effective(settings.get)
    except BaseException:
        # Kit fast shutdown can terminate before a surrounding handler runs.
        # Log first and preserve failure status rather than hiding it in teardown.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        if "exit_code" in inspect.signature(app.close).parameters:
            app.close(exit_code=FAILURE_EXIT_CODE)
        os._exit(FAILURE_EXIT_CODE)
    else:
        app.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the inputs and the launch record, then render."""
    inputs = load_inputs(parse_args(argv))
    validate_worker_launch(inputs)
    render(inputs)
    return 0


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code in (None, 0):
            raise
        # Some Kit python.sh versions swallow uncaught Python exceptions. Preserve
        # failure for the supervisor, even after Kit has handled interpreter teardown.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(FAILURE_EXIT_CODE)
