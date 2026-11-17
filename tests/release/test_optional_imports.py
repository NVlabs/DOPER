# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core modules import without any optional runtime (bayesfilter, h5py, onnx, TensorRT, ...)."""

import pkgutil
import subprocess
import sys

import pytest

import doper

OPTIONAL = (
    "bayesfilter",
    "h5py",
    "onnx",
    "onnxruntime",
    "tensorrt",
    "pyrealsense2",
    "bpy",
    "omni",
    "isaacsim",
    "pxr",
    "numba",
    "nurec_toolbox",
    "robotics_clients",
    # SAM2 masks of the NuRec labels stage (`doper[nurec-sam]`), imported when a segmenter is
    # built.
    "sam2",
)
# The only modules allowed to import an optional runtime at module level.
ALLOWED_EXCEPTIONS = {
    # The single guarded bayesfilter import and the filters built on it (`doper[filter]`).
    "doper.filtering._bayesfilter",
    "doper.filtering.pose_basis",
    "doper.filtering.shirt_pose",
    "doper.filtering.shirt_ephemeris",
    # Wraps the platform TensorRT runtime; importing it without TensorRT raises by design.
    "doper.deployment.tensorrt_runtime",
    # The Numba heatmap kernels (`doper[refine]`), imported only for L-BFGS-B or parallel prep.
    "doper.refinement._numba_kernels",
}
CORE = tuple(
    sorted(
        info.name
        for info in pkgutil.walk_packages(doper.__path__, "doper.")
        if info.name not in ALLOWED_EXCEPTIONS
    )
)


def test_core_list_covers_every_module() -> None:
    assert "doper.evaluation.bop" in CORE
    assert "doper.scene.geometry" in CORE
    assert "doper.scene.labels" in CORE
    assert "doper.pseudo_labels.hdf5_episodes" in CORE
    assert "doper.refinement.heatmap" in CORE
    assert "doper.reconstruction.geometry" in CORE
    assert "doper.reconstruction.visibility" in CORE
    assert "doper.nurec.masks" in CORE
    assert "doper.nurec.package" in CORE
    assert len(CORE) + len(ALLOWED_EXCEPTIONS) == len(
        list(pkgutil.walk_packages(doper.__path__, "doper."))
    )


@pytest.mark.parametrize("module", ("doper", *CORE))
def test_optional_runtimes_not_imported(module: str) -> None:
    code = f"import sys, {module}; print([m for m in {OPTIONAL!r} if m in sys.modules])"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


RENDER_OPTIONAL = ("pxr", "isaacsim", "omni", "carb", "bpy", "blenderproc")
# Host-side modules, including the Isaac modules that run before Kit starts.
HOST_MODULES = (
    "rendering.config",
    "rendering.scene_spec",
    "rendering.run_manifest",
    "rendering.shard_io",
    "rendering.dataset_contract",
    "rendering.output_lock",
    "rendering.resume",
    "rendering.watchdog",
    "rendering.synth_parallel",
    "rendering.usd_mesh",
    "rendering.environment_bundle",
    "rendering.isaac.asset_cache",
    "rendering.isaac.background_mapping",
    "rendering.isaac.capture",
    "rendering.isaac.environment_profile",
    "rendering.isaac.inputs",
    "rendering.isaac.isaac_parallel",
    "rendering.isaac.isaac_worker",
    "rendering.isaac.launch_contract",
    "rendering.isaac.settings",
    "rendering.isaac.worker",
    "rendering.blenderproc",
    "rendering.blenderproc.plan",
    "rendering.blenderproc.assets",
    "rendering.blenderproc.audit",
    "rendering.blenderproc.campaign",
    "rendering.blenderproc.install",
)


@pytest.mark.parametrize("module", HOST_MODULES)
def test_render_host_modules_need_no_engine(module: str) -> None:
    code = f"import sys, {module}; print([m for m in {RENDER_OPTIONAL!r} if m in sys.modules])"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
