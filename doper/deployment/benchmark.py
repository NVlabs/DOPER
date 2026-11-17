# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Benchmark a deployment bundle with TensorRT: build, raw and pose parity, and latency.

For each precision mode the detector and crop engines are built from the bundle's ONNX
files, their raw outputs are compared with the frozen PyTorch references (relative-L2
gates: fp32 0.001, fp16 0.02, bf16 0.05), the network is timed with CUDA events, and the
full ``infer_image`` pipeline (preprocessing, copies, decoding, default PnP) is timed and
compared with the reference poses. Engines and results go to ``--output``, which must lie
outside the bundle. Exit status 1 means at least one mode failed or missed its gate.

Usage::

    python -m doper.deployment.benchmark --bundle bundle/ --output trt_results/ \
        --modes fp32 fp16 bf16

Needs TensorRT (see :mod:`doper.deployment.tensorrt_runtime`); this module imports the
runtime only when a benchmark runs.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.metadata
import json
import platform
import time
import traceback
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np
import torch

import doper
from doper.config import DataConfig, DoperConfig, ModelConfig
from doper.console import log
from doper.deployment.bundle import MANIFEST_NAME, sha256, verify
from doper.deployment.parity import (
    compare_outputs,
    detections_json,
    pose_gate_passed,
    pose_parity,
    summarize,
)
from doper.infer import infer_image
from doper.keypoints import load_keypoints_3d

MODES = ("fp32", "fp16", "bf16")
RELATIVE_L2_LIMITS = {"fp32": 0.001, "fp16": 0.02, "bf16": 0.05}
BRANCHES = ("det", "crop")
CROP_TIMING_BATCHES = (1, 2, 3)
PIPELINE_PASSES = 5
DEFAULT_WARMUP = 20
DEFAULT_ITERATIONS = 100
CPU_THREADS = 1
_MS_PER_SECOND = 1000.0
_DATATYPE_KEY = "Format/Datatype"
RESULTS_NAME = "results.json"
ENVIRONMENT_NAME = "environment.txt"


def _load_runtime() -> ModuleType:
    """Import :mod:`doper.deployment.tensorrt_runtime` (and so TensorRT) on first use."""
    return importlib.import_module("doper.deployment.tensorrt_runtime")


def source_hashes() -> dict[str, str]:
    """Return the SHA-256 of every ``doper/**/*.py`` file, keyed ``doper/<path>``."""
    package = Path(doper.__file__).resolve().parent
    return {
        str(path.relative_to(package.parent)): sha256(path)
        for path in sorted(package.rglob("*.py"))
    }


def distributions() -> list[str]:
    """Return sorted ``name==version`` lines of the installed Python distributions."""
    found = {
        f"{dist.metadata['Name']}=={dist.version}"
        for dist in importlib.metadata.distributions()
        if dist.metadata["Name"]
    }
    return sorted(found, key=str.lower)


def layer_io_datatypes(layers_json: str) -> dict[str, int]:
    """Count every ``Format/Datatype`` entry in TensorRT engine-inspector JSON."""
    counts: Counter[str] = Counter()

    def walk(node: Any) -> None:  # noqa: ANN401 - parsed JSON
        if isinstance(node, dict):
            for key, value in node.items():
                if key == _DATATYPE_KEY:
                    counts[str(value)] += 1
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(json.loads(layers_json))
    return dict(counts)


def _bundle_config(folder: Path) -> DoperConfig:
    saved = json.loads((folder / "config.json").read_text())["model"]
    saved["curve_names"] = tuple(saved.get("curve_names", ()))
    return DoperConfig(data=DataConfig(root=""), model=ModelConfig(**saved))


def _check_output(folder: Path, output: Path) -> None:
    folder, output = folder.resolve(), output.resolve()
    if output == folder or folder in output.parents:
        raise ValueError(f"Write benchmark output outside the bundle, not to {output}")


def _raw_parity(folder: Path, engines: dict[str, Any]) -> list[dict[str, Any]]:
    parity = []
    for branch, engine in engines.items():
        references = sorted(folder.glob(f"ref*_{branch}.npz"))
        if not references:
            raise ValueError(f"No reference for the {branch} branch in {folder}")
        for file in references:
            with np.load(file) as reference:
                actual = engine.run(reference["input"])
                for name, value in actual.items():
                    parity.append(
                        {
                            "sample": file.name,
                            "tensor": name,
                            **compare_outputs(value, reference[name]),
                        }
                    )
    return parity


def crop_timing_batches(max_det: int) -> tuple[int, ...]:
    """Crop batches to time: ``CROP_TIMING_BATCHES`` up to ``max_det``, plus ``max_det``."""
    return tuple(sorted({*(b for b in CROP_TIMING_BATCHES if b <= max_det), max_det}))


def _network_timing(
    folder: Path, engines: dict[str, Any], warmup: int, iterations: int, max_det: int
) -> dict[str, Any]:
    timing = {}
    for branch, engine in engines.items():
        with np.load(sorted(folder.glob(f"ref*_{branch}.npz"))[0]) as reference:
            seed = reference["input"][:1].copy()
        for batch in (1,) if branch == "det" else crop_timing_batches(max_det):
            x = np.repeat(seed, batch, axis=0)
            for _ in range(warmup):
                engine.run(x)
            samples = []
            for _ in range(iterations):
                engine.run(x)
                samples.append(engine.last_ms)
            timing[f"{branch}_batch{batch}"] = {**summarize(samples), "samples_ms": samples}
    return timing


def _read_frame(path: Path) -> np.ndarray:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Cannot read frame {path}")
    return np.asarray(frame)


def _pipeline(
    folder: Path,
    manifest: dict[str, Any],
    model: Any,  # noqa: ANN401 - DoperNet or tensorrt_runtime.EngineModel
    cfg: DoperConfig,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    """Run ``infer_image`` on every bundle frame; return predictions and wall-clock timing."""
    camera = json.loads((folder / "camera.json").read_text())
    kwargs = {
        "score_thr": manifest["score_thr"],
        "max_det": manifest["max_det"],
        "keypoints_3d": load_keypoints_3d(folder / "keypoints.json").astype(np.float32),
        "camera_matrix": np.asarray(camera["camera_matrix"], np.float64),
        "dist_coeffs": np.asarray(camera["distortion"], np.float64),
    }
    frames = [_read_frame(path) for path in sorted(folder.glob("frame*.png"))]
    with torch.inference_mode():
        for frame in frames:
            infer_image(model, frame, cfg, "cpu", **kwargs)
        predicted = [detections_json(infer_image(model, f, cfg, "cpu", **kwargs)) for f in frames]
        samples: list[float] = []
        passes = []
        for _ in range(PIPELINE_PASSES):
            pass_samples = []
            for frame in frames:
                start = time.perf_counter()
                infer_image(model, frame, cfg, "cpu", **kwargs)
                pass_samples.append((time.perf_counter() - start) * _MS_PER_SECOND)
            samples.extend(pass_samples)
            passes.append(summarize(pass_samples))
    wall = {
        **summarize(samples),
        "samples_ms": samples,
        "passes": passes,
        "scope": "predecoded image through preprocessing, H2D/D2H, decode and default PnP; "
        "no tracking/render",
    }
    return predicted, wall


def _parity_vs_device_fp32(
    output: Path, mode: str, predicted: list, identity: dict[str, Any]
) -> dict[str, Any]:
    baseline = output / "fp32.predictions.json"
    baseline_identity = output / "fp32.predictions.identity.json"
    if (
        baseline.exists()
        and baseline_identity.exists()
        and json.loads(baseline_identity.read_text()) == identity
    ):
        return pose_parity(predicted, json.loads(baseline.read_text()))
    return {"status": "unavailable_or_stale_reference"}


def _benchmark_mode(
    folder: Path,
    output: Path,
    mode: str,
    context: dict[str, Any],
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    """Build, check and time one precision mode of a verified bundle."""
    runtime, manifest, cfg = context["runtime"], context["manifest"], context["cfg"]
    sizes = {"det": cfg.model.det_input_size, "crop": cfg.model.crop_size}
    result: dict[str, Any] = {
        "precision_requested": mode,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
    }
    engines: dict[str, Any] = {}
    try:
        for branch in BRANCHES:
            plan = output / f"{branch}_{mode}.plan"
            runtime.build_engine(
                folder / f"{branch}.onnx", plan, mode, sizes[branch], manifest["max_det"]
            )
            engines[branch] = runtime.Engine(plan)
            (output / f"{branch}_{mode}.layers.json").write_text(engines[branch].layers)
        result["layer_io_datatypes"] = {b: layer_io_datatypes(e.layers) for b, e in engines.items()}
        result["engine_device_memory_bytes"] = {
            b: int(e.device_memory_bytes) for b, e in engines.items()
        }
        parity = _raw_parity(folder, engines)
        limit = RELATIVE_L2_LIMITS[mode]
        result["raw_output_parity"] = parity
        result["raw_output_relative_l2_gate"] = {
            "limit": limit,
            "pass": all(p["relative_l2"] <= limit for p in parity),
        }
        result["network_cuda_events"] = _network_timing(
            folder, engines, warmup, iterations, manifest["max_det"]
        )
        predicted, result["pipeline_wall"] = _pipeline(
            folder, manifest, runtime.EngineModel(engines), cfg
        )
        reference = json.loads((folder / "reference_poses.json").read_text())
        result["pipeline_parity"] = pose_parity(predicted, reference)
        result["pose_gate_pass"] = pose_gate_passed(result["pipeline_parity"])
        if mode != "fp32":
            result["pipeline_parity_vs_device_fp32"] = _parity_vs_device_fp32(
                output, mode, predicted, context["identity"]
            )
        result["memory_after_warmup"] = context["cuda"].memory()
        (output / f"{mode}.predictions.json").write_text(json.dumps(predicted))
        identity_text = json.dumps(context["identity"], indent=2)
        (output / f"{mode}.predictions.identity.json").write_text(identity_text)
        result["status"] = (
            "measured" if result["raw_output_relative_l2_gate"]["pass"] else ("parity_failed")
        )
        return result
    finally:
        for engine in engines.values():
            try:
                engine.close()
            except RuntimeError as exc:  # keep the mode's own error and close the others
                log(f"{mode}: closing an engine failed: {exc}")
        gc.collect()


def report_passed(results: dict[str, dict[str, Any]]) -> bool:
    """Return whether every mode was measured and passed its relative-L2 gate."""
    return bool(results) and all(
        r.get("status") == "measured"
        and r.get("raw_output_relative_l2_gate", {}).get("pass") is True
        for r in results.values()
    )


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False))


def run(
    bundle: Path,
    output: Path,
    modes: Sequence[str],
    warmup: int = DEFAULT_WARMUP,
    iterations: int = DEFAULT_ITERATIONS,
) -> dict[str, Any]:
    """Benchmark one deployment bundle in each precision mode and write ``results.json``.

    A mode that raises is recorded as ``{"status": "failed", "error": ...}`` and the
    remaining modes still run. ``results.json`` is rewritten after every mode.

    Args:
        bundle: Bundle folder written by :func:`doper.deployment.onnx_export.export_bundle`.
        output: Folder for engines, layer dumps, predictions and the report; must lie
            outside the bundle.
        modes: Subset of ``fp32``, ``fp16`` and ``bf16``.
        warmup: Untimed engine runs before each network timing.
        iterations: Timed engine runs per network timing.

    Returns:
        The report: environment, source hashes, per-mode ``results`` and ``passed``.

    Raises:
        ValueError: For an unknown mode, an output inside the bundle, or a bundle whose
            files do not match its manifest.
    """
    bundle_dir, output = Path(bundle), Path(output)
    unknown = [mode for mode in modes if mode not in MODES]
    if unknown:
        raise ValueError(f"Unknown precision modes {unknown}; expected some of {MODES}")
    _check_output(bundle_dir, output)
    manifest = verify(bundle_dir)
    runtime = _load_runtime()
    cuda = runtime.load_cudart()
    gpu = cuda.device_name()
    output.mkdir(parents=True, exist_ok=True)
    sources = source_hashes()
    report: dict[str, Any] = {
        "gpu": gpu,
        "cudart": cuda.soname,
        "platform": platform.platform(),
        "tensorrt": runtime.TENSORRT_VERSION,
        "torch_cpu": torch.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "cpu_threads": torch.get_num_threads(),
        "memory_start": cuda.memory(),
        "precision_policy": "FP32 I/O; FP16 or BF16 tactics enabled, FP32 fallback permitted; "
        "TF32 disabled",
        "network_scope": "CUDA events around engine enqueue only; allocation, copies and "
        "output decoding excluded",
        "relative_l2_limits": {mode: RELATIVE_L2_LIMITS[mode] for mode in modes},
        "bundle_manifest_sha256": sha256(bundle_dir / MANIFEST_NAME),
        "source_sha256": sources,
        "results": {},
        "passed": False,
    }
    identity = {
        "model_manifest": report["bundle_manifest_sha256"],
        "gpu": gpu,
        "tensorrt": report["tensorrt"],
        "torch": torch.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "source_sha256": sources,
    }
    context = {
        "runtime": runtime,
        "manifest": manifest,
        "cfg": _bundle_config(bundle_dir),
        "cuda": cuda,
        "identity": identity,
    }
    (output / ENVIRONMENT_NAME).write_text("\n".join(distributions()) + "\n")
    for mode in modes:
        try:
            result = _benchmark_mode(bundle_dir, output, mode, context, warmup, iterations)
        except Exception as exc:  # noqa: BLE001 - recorded per mode; other modes still run
            traceback.print_exc()
            result = {"status": "failed", "error": repr(exc)}
        report["results"][mode] = result
        report["passed"] = report_passed(report["results"])
        _write_json(output / RESULTS_NAME, report)
        log(f"{mode}: {result.get('status')} {result.get('raw_output_relative_l2_gate')}")
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark a DOPER deployment bundle")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="outside the bundle")
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark CLI; return 0 when every mode passed, else 1."""
    args = _build_parser().parse_args(argv)
    threads = torch.get_num_threads(), cv2.getNumThreads()
    torch.set_num_threads(CPU_THREADS)
    cv2.setNumThreads(CPU_THREADS)
    try:
        report = run(args.bundle, args.output, args.modes, args.warmup, args.iterations)
    finally:
        torch.set_num_threads(threads[0])
        cv2.setNumThreads(threads[1])
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
