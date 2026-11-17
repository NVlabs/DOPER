# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT runtime loading, benchmark report assembly with a fake engine, and GPU parity."""

import importlib
import json
import math
import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from tiny_coco import write_deployment_inputs

from doper.deployment import benchmark
from doper.deployment.onnx_export import export_bundle
from doper.deployment.parity import compare_outputs, pose_gate_passed

RUNTIME = "doper.deployment.tensorrt_runtime"
FAKE_TENSORRT_VERSION = "0.0-fake"


def _strict_json(path: Path) -> dict:
    def reject(token):
        raise AssertionError(f"non-finite JSON token {token}")

    return json.loads(path.read_text(), parse_constant=reject)


def _import_runtime(monkeypatch, tensorrt_module):
    """Import the runtime against ``tensorrt_module``; monkeypatch undoes it at teardown."""
    import doper.deployment

    monkeypatch.setitem(sys.modules, "tensorrt", tensorrt_module)
    # setitem records the prior state (absent or a real module) so teardown restores it;
    # a plain delitem records nothing when the key is absent and the stub build would leak.
    monkeypatch.setitem(sys.modules, RUNTIME, None)
    del sys.modules[RUNTIME]
    monkeypatch.setattr(doper.deployment, "tensorrt_runtime", None, raising=False)
    return importlib.import_module(RUNTIME)


def _fake_tensorrt() -> types.ModuleType:
    module = types.ModuleType("tensorrt")
    module.__version__ = FAKE_TENSORRT_VERSION
    module.Logger = mock.MagicMock()
    return module


@pytest.fixture
def fake_runtime(monkeypatch):
    """The real runtime module imported against a stub ``tensorrt``."""
    return _import_runtime(monkeypatch, _fake_tensorrt())


def test_runtime_without_tensorrt_names_extra(monkeypatch):
    with pytest.raises(ImportError, match=r"doper\[tensorrt\]") as raised:
        _import_runtime(monkeypatch, None)
    assert "JetPack" in str(raised.value)


def test_cudart_missing_raises_clear_error(fake_runtime, monkeypatch):
    monkeypatch.setattr(fake_runtime.ctypes, "CDLL", mock.Mock(side_effect=OSError("not found")))
    with pytest.raises(OSError, match=r"libcudart\.so\.13.*libcudart\.so\.12"):
        fake_runtime.load_cudart()


def test_cudart_falls_back_to_cuda_12(fake_runtime, monkeypatch):
    def load(soname):
        if soname == "libcudart.so.13":
            raise OSError("not found")
        return mock.MagicMock()

    monkeypatch.setattr(fake_runtime.ctypes, "CDLL", load)
    assert fake_runtime.load_cudart().soname == "libcudart.so.12"


def test_reduced_precision_needs_weakly_typed_tensorrt(fake_runtime, monkeypatch, tmp_path):
    monkeypatch.setattr(fake_runtime.trt, "BuilderFlag", types.SimpleNamespace(), raising=False)
    with pytest.raises(RuntimeError, match="use TensorRT 10"):
        fake_runtime.build_engine(tmp_path / "det.onnx", tmp_path / "det.plan", "bf16", 96, 3)


def _builder_tensorrt() -> types.ModuleType:
    module = _fake_tensorrt()
    for name in ("Builder", "BuilderFlag", "MemoryPoolType", "ProfilingVerbosity"):
        setattr(module, name, mock.MagicMock())
    builder = module.Builder.return_value
    builder.build_serialized_network.return_value = b"plan"
    config = builder.create_builder_config.return_value
    config.get_timing_cache.return_value.serialize.return_value = b""
    return module


@pytest.mark.parametrize("max_det", [1, 5])
def test_crop_profile_and_build_key_follow_max_det(monkeypatch, tmp_path, max_det):
    runtime = _import_runtime(monkeypatch, _builder_tensorrt())
    monkeypatch.setattr(runtime, "load_cudart", lambda: types.SimpleNamespace(device_name=str))
    onnx_path = tmp_path / "crop.onnx"
    onnx_path.write_bytes(b"onnx")
    monkeypatch.setattr(runtime, "_parse", lambda network, path: None)
    runtime.build_engine(onnx_path, tmp_path / "crop.plan", "fp32", 64, max_det)
    builder = runtime.trt.Builder.return_value
    shapes = builder.create_optimization_profile.return_value.set_shape.call_args.args[1:]
    assert [shape[0] for shape in shapes] == [1, 1, max_det]
    key = json.loads((tmp_path / "crop.build.json").read_text())
    assert key["crop_batch_profile"] == [1, 1, max_det]


def test_crop_profile_rejects_non_positive_max_det(fake_runtime):
    with pytest.raises(ValueError, match="at least 1"):
        fake_runtime.crop_batch_profile(0)


def test_crop_timing_batches_stop_at_max_det():
    assert benchmark.crop_timing_batches(1) == (1,)
    assert benchmark.crop_timing_batches(3) == (1, 2, 3)
    assert benchmark.crop_timing_batches(5) == (1, 2, 3, 5)


def test_stub_runtime_does_not_leak():
    # Runs after the stub tests above; a leaked stub build would break the GPU test.
    assert getattr(sys.modules.get("tensorrt"), "__version__", None) != FAKE_TENSORRT_VERSION
    runtime = sys.modules.get(RUNTIME)
    assert runtime is None or runtime.TENSORRT_VERSION != FAKE_TENSORRT_VERSION


# --- Benchmark report assembly on CPU: onnxruntime stands in for TensorRT engines. ---


def _export(inputs, output):
    return export_bundle(
        inputs["checkpoint"],
        inputs["keypoints_3d"],
        inputs["camera"],
        inputs["frames"],
        output,
        score_thr=0.0,
        max_det=1,
    )


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory):
    """A one-frame, one-detection bundle whose PnP problem is well posed.

    A random-init model predicts arbitrary 2D keypoints, so with arbitrary 3D keypoints
    RANSAC PnP flips between solutions on sub-pixel changes. Back-projecting the predicted
    keypoints to distinct depths makes them an exact projection of the 3D layout.
    """
    pytest.importorskip("onnx")
    inputs = write_deployment_inputs(tmp_path_factory.mktemp("inputs"), n_frames=1)
    probe = _export(inputs, tmp_path_factory.mktemp("probe") / "bundle")
    pixels = np.asarray(json.loads((probe / "reference_poses.json").read_text())[0][0]["keypoints"])
    camera = np.asarray(json.loads(inputs["camera"].read_text())["camera_matrix"])
    rays = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(camera).T
    depths = 0.4 + 0.03 * np.arange(len(pixels))
    inputs["keypoints_3d"].write_text(
        json.dumps({"keypoints_3d": (rays * depths[:, None]).tolist()})
    )
    output = _export(inputs, tmp_path_factory.mktemp("export") / "bundle")
    assert json.loads((output / "reference_poses.json").read_text())[0][0]["tvec"] is not None
    return output


class _OrtEngine:
    """Engine stand-in: ``build`` writes the ONNX path into the plan, ``run`` uses ORT."""

    perturb = 0.0

    def __init__(self, path, text=True):
        ort = pytest.importorskip("onnxruntime")
        onnx_path = Path(path).read_text() if text else str(path)
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.layers = json.dumps({"Layers": [{"Inputs": [{"Format/Datatype": "Row major FP32"}]}]})
        self.device_memory_bytes = 0
        self.last_ms = None

    def run(self, x):
        names = [output.name for output in self.session.get_outputs()]
        values = self.session.run(None, {"input": np.ascontiguousarray(x, np.float32)})
        self.last_ms = 0.5
        return {name: value + self.perturb for name, value in zip(names, values, strict=True)}

    def close(self):
        self.session = None


class _FakeCuda:
    soname = "libcudart.so.12"

    def device_name(self):
        return "Fake GPU"

    def memory(self):
        return {"free_bytes": 2, "total_bytes": 3, "used_bytes": 1}


def _patch_runtime(monkeypatch, runtime, fail_mode=None, failure="build"):
    class Engine(_OrtEngine):
        perturb = 0.0

    def build_engine(onnx_path, destination, precision, spatial, max_det):
        assert max_det == 1  # the bundle's recorded max_det
        if precision == fail_mode and failure == "build":
            raise RuntimeError("TensorRT build failed")
        Path(destination).write_text(str(onnx_path))

    def make_engine(path):
        engine = Engine(path)
        if failure == "parity" and Path(path).stem.endswith(str(fail_mode)):
            engine.perturb = 1.0
        return engine

    monkeypatch.setattr(runtime, "build_engine", build_engine)
    monkeypatch.setattr(runtime, "Engine", make_engine)
    monkeypatch.setattr(runtime, "load_cudart", _FakeCuda)
    monkeypatch.setattr(benchmark, "_load_runtime", lambda: runtime)


def test_report_passes_with_matching_engine(bundle_dir, fake_runtime, monkeypatch, tmp_path):
    _patch_runtime(monkeypatch, fake_runtime)
    report = benchmark.run(bundle_dir, tmp_path / "out", ["fp32"], warmup=1, iterations=2)
    assert report["passed"] is True
    assert report["gpu"] == "Fake GPU"
    assert report["cudart"] == "libcudart.so.12"
    assert "doper/deployment/benchmark.py" in report["source_sha256"]
    result = report["results"]["fp32"]
    assert result["status"] == "measured"
    assert result["raw_output_relative_l2_gate"] == {"limit": 0.001, "pass": True}
    # The bundle records max_det=1, so its crop engine only accepts batch 1.
    assert set(result["network_cuda_events"]) == {"det_batch1", "crop_batch1"}
    assert result["pose_gate_pass"] is True, result["pipeline_parity"]
    assert result["pipeline_parity"]["frames_with_detection_count_difference"] == 0
    assert _strict_json(tmp_path / "out" / "results.json") == report
    assert "numpy==" in (tmp_path / "out" / "environment.txt").read_text()


@pytest.mark.parametrize("failure", ["build", "parity"])
def test_report_fails_when_any_mode_fails(bundle_dir, fake_runtime, monkeypatch, tmp_path, failure):
    _patch_runtime(monkeypatch, fake_runtime, fail_mode="fp16", failure=failure)
    out = tmp_path / "out"
    argv = ["--bundle", str(bundle_dir), "--output", str(out), "--modes", "fp32", "fp16"]
    assert benchmark.main([*argv, "--warmup", "1", "--iterations", "2"]) == 1
    report = _strict_json(out / "results.json")
    assert report["passed"] is False
    assert report["results"]["fp32"]["status"] == "measured"
    failed = report["results"]["fp16"]
    if failure == "build":
        assert failed["status"] == "failed"
        assert "TensorRT build failed" in failed["error"]
    else:
        assert failed["status"] == "parity_failed"
        assert failed["raw_output_relative_l2_gate"] == {"limit": 0.02, "pass": False}
        assert "status" not in failed["pipeline_parity_vs_device_fp32"]


def test_benchmark_rejects_modified_bundle(bundle_dir, fake_runtime, monkeypatch, tmp_path):
    _patch_runtime(monkeypatch, fake_runtime)
    copy = tmp_path / "bundle"
    copy.mkdir()
    for path in bundle_dir.iterdir():
        (copy / path.name).write_bytes(path.read_bytes())
    (copy / "camera.json").write_text("{}")
    with pytest.raises(ValueError, match="camera.json"):
        benchmark.run(copy, tmp_path / "out", ["fp32"])


def test_benchmark_refuses_output_inside_bundle(bundle_dir, fake_runtime, monkeypatch):
    _patch_runtime(monkeypatch, fake_runtime)
    with pytest.raises(ValueError, match="outside the bundle"):
        benchmark.run(bundle_dir, bundle_dir / "results", ["fp32"])


def test_benchmark_without_modes_does_not_pass(bundle_dir, fake_runtime, monkeypatch, tmp_path):
    _patch_runtime(monkeypatch, fake_runtime)
    report = benchmark.run(bundle_dir, tmp_path / "out", [])
    assert report["passed"] is False
    assert (tmp_path / "out" / "environment.txt").is_file()


def test_benchmark_rejects_unknown_mode(bundle_dir, tmp_path):
    with pytest.raises(ValueError, match="int8"):
        benchmark.run(bundle_dir, tmp_path / "out", ["int8"])


# --- GPU: a real TensorRT FP32 engine against the exported bundle. ---


@pytest.mark.gpu
def test_fp32_engine_matches_onnx_bundle(bundle_dir, tmp_path):
    pytest.importorskip("tensorrt")
    runtime = importlib.import_module(RUNTIME)
    try:
        runtime.load_cudart().device_name()
    except (OSError, RuntimeError) as exc:
        pytest.skip(f"No usable CUDA runtime: {exc}")
    report = benchmark.run(bundle_dir, tmp_path / "out", ["fp32"], warmup=2, iterations=5)
    result = report["results"]["fp32"]
    assert result["status"] == "measured", result.get("error")
    assert all(p["relative_l2"] <= 0.001 for p in result["raw_output_parity"])
    assert {p["sample"] for p in result["raw_output_parity"]} == {
        path.name for path in bundle_dir.glob("ref*_*.npz")
    }
    assert pose_gate_passed(result["pipeline_parity"]), result["pipeline_parity"]
    assert report["passed"] is True
    assert math.isfinite(result["network_cuda_events"]["crop_batch1"]["mean_ms"])
    # The bundle references only hold crop batch 1; build a max_det=3 crop engine and
    # check batch 3 against onnxruntime.
    with np.load(bundle_dir / "ref000_crop.npz") as reference:
        size = reference["input"].shape[-1]
    x = np.random.default_rng(3).standard_normal((3, 3, size, size)).astype(np.float32)
    expected = _OrtEngine(bundle_dir / "crop.onnx", text=False).run(x)
    plan = tmp_path / "batch3" / "crop_fp32.plan"
    runtime.build_engine(bundle_dir / "crop.onnx", plan, "fp32", size, 3)
    with runtime.Engine(plan) as engine:
        actual = engine.run(x)
    for name, value in expected.items():
        assert compare_outputs(actual[name], value)["relative_l2"] <= 0.001
