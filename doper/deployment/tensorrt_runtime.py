# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and run TensorRT engines for the two DOPER branches without CUDA PyTorch.

CUDA memory, streams and events go through a ``ctypes`` binding of the CUDA runtime
(``libcudart.so.13``, falling back to ``libcudart.so.12``), so a CPU PyTorch build is
enough; this is what runs on Jetson Thor. Importing this module needs TensorRT, which
comes from the platform (pip ``tensorrt`` on x86, JetPack on Thor); install the rest with
``doper[tensorrt]``. The ``fp16`` and ``bf16`` modes need TensorRT 10 (e.g. pip
``tensorrt-cu12<11`` on x86, JetPack 7 on Thor): TensorRT 11 dropped weakly typed builds.
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import json
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

import numpy as np
import torch

from doper.deployment.bundle import sha256

TENSORRT_HINT = (
    "tensorrt is required for doper.deployment.tensorrt_runtime; install doper[tensorrt] "
    "plus TensorRT from the platform (pip `tensorrt` on x86, JetPack on Thor; fp16/bf16 "
    "need TensorRT 10, e.g. pip `tensorrt-cu12<11`)"
)

try:
    import tensorrt as trt  # pyright: ignore[reportMissingImports]
except ImportError as _error:
    raise ImportError(TENSORRT_HINT) from _error

Precision = Literal["fp32", "fp16", "bf16"]

CUDART_SONAMES = ("libcudart.so.13", "libcudart.so.12")
WORKSPACE_BYTES = 2 << 30
INPUT_NAME = "input"
TENSORRT_VERSION: str = trt.__version__
LOGGER = trt.Logger(trt.Logger.WARNING)

_MEMCPY_HOST_TO_DEVICE = 1
_MEMCPY_DEVICE_TO_HOST = 2
# cudaDeviceProp starts with ``char name[256]`` in every CUDA version; the buffer only has
# to be at least as large as the whole struct.
_DEVICE_NAME_BYTES = 256
_DEVICE_PROP_BUFFER_BYTES = 1 << 13
_PROPERTY_SYMBOLS = ("cudaGetDeviceProperties_v2", "cudaGetDeviceProperties")
_P = ctypes.c_void_p
_SIZE_P = ctypes.POINTER(ctypes.c_size_t)
_SIGNATURES: dict[str, list[Any]] = {
    "cudaMalloc": [ctypes.POINTER(_P), ctypes.c_size_t],
    "cudaFree": [_P],
    "cudaMemcpy": [_P, _P, ctypes.c_size_t, ctypes.c_int],
    "cudaStreamCreate": [ctypes.POINTER(_P)],
    "cudaStreamDestroy": [_P],
    "cudaStreamSynchronize": [_P],
    "cudaEventCreate": [ctypes.POINTER(_P)],
    "cudaEventRecord": [_P, _P],
    "cudaEventSynchronize": [_P],
    "cudaEventElapsedTime": [ctypes.POINTER(ctypes.c_float), _P, _P],
    "cudaEventDestroy": [_P],
    "cudaMemGetInfo": [_SIZE_P, _SIZE_P],
    "cudaGetDevice": [ctypes.POINTER(ctypes.c_int)],
}


class CudaRuntime:
    """Minimal ``ctypes`` binding of the CUDA runtime API.

    Attributes:
        soname: The shared library that was loaded, e.g. ``libcudart.so.12``.
    """

    def __init__(self, lib: ctypes.CDLL, soname: str) -> None:
        """Declare the signatures of the functions used from ``lib``."""
        self._lib = lib
        self.soname = soname
        for name, argtypes in _SIGNATURES.items():
            function = getattr(lib, name)
            function.argtypes = argtypes
            function.restype = ctypes.c_int
        lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        lib.cudaGetErrorString.restype = ctypes.c_char_p

    def call(self, name: str, *args: Any) -> None:  # noqa: ANN401 - ctypes arguments
        """Call a CUDA runtime function and raise on a nonzero status.

        Raises:
            RuntimeError: ``<name>: CUDA error <code> (<description>)``.
        """
        code = getattr(self._lib, name)(*args)
        if code:
            message = self._lib.cudaGetErrorString(code)
            text = message.decode() if isinstance(message, bytes) else "unknown"
            raise RuntimeError(f"{name}: CUDA error {code} ({text})")

    def memory(self) -> dict[str, int]:
        """Return free, total and used device memory in bytes."""
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        self.call("cudaMemGetInfo", ctypes.byref(free), ctypes.byref(total))
        return {
            "free_bytes": free.value,
            "total_bytes": total.value,
            "used_bytes": total.value - free.value,
        }

    def device_name(self) -> str:
        """Return the name of the current CUDA device."""
        device = ctypes.c_int()
        self.call("cudaGetDevice", ctypes.byref(device))
        symbol = next(s for s in _PROPERTY_SYMBOLS if hasattr(self._lib, s))
        function = getattr(self._lib, symbol)
        function.argtypes = [_P, ctypes.c_int]
        function.restype = ctypes.c_int
        buffer = ctypes.create_string_buffer(_DEVICE_PROP_BUFFER_BYTES)
        self.call(symbol, ctypes.cast(buffer, _P), device.value)
        return buffer.raw[:_DEVICE_NAME_BYTES].split(b"\0", 1)[0].decode()


@functools.cache
def load_cudart() -> CudaRuntime:
    """Load the CUDA runtime, trying ``libcudart.so.13`` and then ``libcudart.so.12``.

    Raises:
        OSError: If neither library can be loaded; the message lists every attempt.
    """
    errors = []
    for soname in CUDART_SONAMES:
        try:
            return CudaRuntime(ctypes.CDLL(soname), soname)
        except OSError as exc:
            errors.append(f"{soname}: {exc}")
    raise OSError(
        "Cannot load the CUDA runtime; tried " + "; ".join(errors) + ". Install the CUDA 13 or "
        "12 runtime (JetPack on Thor) or add its directory to LD_LIBRARY_PATH."
    )


def _parse(network: Any, onnx_path: Path) -> None:  # noqa: ANN401 - trt.INetworkDefinition
    parser = trt.OnnxParser(network, LOGGER)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))


def _precision_flag(precision: str) -> Any:  # noqa: ANN401 - trt.BuilderFlag
    """Return the weakly typed builder flag enabling ``fp16`` or ``bf16`` tactics.

    Raises:
        RuntimeError: On TensorRT 11 and later, which only build strongly typed networks.
    """
    flag = getattr(trt.BuilderFlag, precision.upper(), None)
    if flag is None:
        raise RuntimeError(
            f"TensorRT {TENSORRT_VERSION} has no weakly typed {precision} builds "
            f"(BuilderFlag.{precision.upper()} was removed); use TensorRT 10 for this mode"
        )
    return flag


def crop_batch_profile(max_det: int) -> tuple[int, int, int]:
    """Return the min / opt / max crop batch of the crop engine for a bundle's ``max_det``.

    Raises:
        ValueError: If ``max_det`` is below 1.
    """
    if max_det < 1:
        raise ValueError(f"max_det must be at least 1, got {max_det}")
    return (1, 1, max_det)


def build_engine(
    onnx_path: Path, destination: Path, precision: Precision, spatial: int, max_det: int
) -> None:
    """Build a serialized TensorRT engine, reusing ``destination`` when its build key matches.

    TF32 is disabled and the workspace is capped at 2 GiB. ``fp16``/``bf16`` enable those
    tactics with FP32 fallback; I/O stays FP32. A ``crop.onnx`` (dynamic batch) gets an
    optimization profile for batch 1/1/``max_det`` at ``spatial`` x ``spatial``. The build
    key (ONNX hash, TensorRT version, GPU, precision and settings, including the crop batch
    profile) is written next to the plan as ``<plan>.build.json``; the timing cache is
    ``timing_<precision>.cache`` in the same folder.

    Args:
        onnx_path: ``det.onnx`` or ``crop.onnx`` of a deployment bundle.
        destination: Engine plan file to write.
        precision: ``fp32``, ``fp16`` or ``bf16``.
        spatial: Square input size of the branch.
        max_det: The bundle's recorded ``max_det``; the largest crop batch the engine accepts.

    Raises:
        RuntimeError: If parsing or building fails, or for ``fp16``/``bf16`` on a TensorRT
            without weakly typed reduced-precision builds (11 and later).
        ValueError: If ``max_det`` is below 1.
    """
    onnx_path, destination = Path(onnx_path), Path(destination)
    batch_profile = crop_batch_profile(max_det)
    if precision != "fp32":
        _precision_flag(precision)  # fail before hashing and parsing
    is_crop = onnx_path.stem == "crop"
    key = {
        "onnx_sha256": sha256(onnx_path),
        "tensorrt": TENSORRT_VERSION,
        "gpu": load_cudart().device_name(),
        "precision": precision,
        "workspace_bytes": WORKSPACE_BYTES,
        "tf32": False,
        "crop_batch_profile": list(batch_profile) if is_crop else None,
        "spatial": spatial,
    }
    metadata = destination.with_suffix(".build.json")
    if destination.exists() and metadata.exists() and json.loads(metadata.read_text()) == key:
        return
    builder = trt.Builder(LOGGER)
    network = builder.create_network(0)
    _parse(network, onnx_path)
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_BYTES)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    cache_file = destination.parent / f"timing_{precision}.cache"
    cache = config.create_timing_cache(cache_file.read_bytes() if cache_file.exists() else b"")
    config.set_timing_cache(cache, False)
    if precision != "fp32":
        config.set_flag(_precision_flag(precision))
    if is_crop:
        profile = builder.create_optimization_profile()
        shapes = [(batch, 3, spatial, spatial) for batch in batch_profile]
        profile.set_shape(INPUT_NAME, *shapes)
        config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"TensorRT build failed for {onnx_path.name} ({precision})")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial")
    temporary.write_bytes(bytes(plan))
    temporary.replace(destination)
    metadata.write_text(json.dumps(key, indent=2))
    cache_file.write_bytes(bytes(config.get_timing_cache().serialize()))


class Engine:
    """A deserialized engine with device buffers, a stream and CUDA-event timing.

    Attributes:
        last_ms: Enqueue-to-completion time of the last :meth:`run`, from CUDA events.
        layers: Engine inspector layer information as a JSON string.
        device_memory_bytes: Activation memory the engine needs.
    """

    def __init__(self, path: Path) -> None:
        """Deserialize the engine at ``path`` and create its execution context.

        Raises:
            RuntimeError: If the plan cannot be deserialized.
            ValueError: If an input or output is not FP32.
        """
        self._cuda = load_cudart()
        self._runtime: Any = trt.Runtime(LOGGER)
        self._engine: Any = self._runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self._engine is None:
            raise RuntimeError(f"Engine deserialize failed: {path}")
        self._names = [self._engine.get_tensor_name(i) for i in range(self._engine.num_io_tensors)]
        for name in self._names:
            if self._engine.get_tensor_dtype(name) != trt.float32:
                raise ValueError(
                    f"Expected FP32 I/O, {name} is {self._engine.get_tensor_dtype(name)}"
                )
        self._context: Any = self._engine.create_execution_context()
        inspector = self._engine.create_engine_inspector()
        self.layers: str = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
        self.device_memory_bytes = int(self._engine.device_memory_size_v2)
        self.last_ms: float | None = None
        self._buffers: dict[str, tuple[ctypes.c_void_p, int]] = {}
        self._stream, self._start, self._end = _P(), _P(), _P()
        try:
            self._cuda.call("cudaStreamCreate", ctypes.byref(self._stream))
            self._cuda.call("cudaEventCreate", ctypes.byref(self._start))
            self._cuda.call("cudaEventCreate", ctypes.byref(self._end))
        except RuntimeError:
            with contextlib.suppress(RuntimeError):
                self.close()
            raise

    def _bind(self, name: str, nbytes: int) -> ctypes.c_void_p:
        """Return a device buffer of at least ``nbytes`` bound to tensor ``name``."""
        if name not in self._buffers or self._buffers[name][1] < nbytes:
            if name in self._buffers:
                self._cuda.call("cudaFree", self._buffers[name][0])
                del self._buffers[name]
            pointer = _P()
            self._cuda.call("cudaMalloc", ctypes.byref(pointer), nbytes)
            self._buffers[name] = (pointer, nbytes)
        pointer = self._buffers[name][0]
        if not self._context.set_tensor_address(name, pointer.value):
            raise RuntimeError(f"Binding failed: {name}")
        return pointer

    def run(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Run the engine on host input ``x`` and return its outputs by name.

        Raises:
            RuntimeError: If the engine is closed or execution fails.
            ValueError: If the input shape is outside the engine's profile.
        """
        if self._context is None:
            raise RuntimeError("Engine is closed")
        x = np.ascontiguousarray(x, dtype=np.float32)
        if not self._context.set_input_shape(INPUT_NAME, x.shape):
            raise ValueError(f"Unsupported input shape {x.shape}")
        outputs = {}
        for name in self._names:
            if name == INPUT_NAME:
                continue
            shape = tuple(self._context.get_tensor_shape(name))
            if any(v < 0 for v in shape):
                raise ValueError(f"Unresolved output shape for {name}: {shape}")
            outputs[name] = np.empty(shape, np.float32)
        source = self._bind(INPUT_NAME, x.nbytes)
        targets = {name: self._bind(name, array.nbytes) for name, array in outputs.items()}
        self._cuda.call("cudaMemcpy", source, _P(x.ctypes.data), x.nbytes, _MEMCPY_HOST_TO_DEVICE)
        self._cuda.call("cudaEventRecord", self._start, self._stream)
        if not self._context.execute_async_v3(self._stream.value):
            raise RuntimeError("TensorRT execution failed")
        self._cuda.call("cudaEventRecord", self._end, self._stream)
        self._cuda.call("cudaEventSynchronize", self._end)
        elapsed = ctypes.c_float()
        self._cuda.call("cudaEventElapsedTime", ctypes.byref(elapsed), self._start, self._end)
        self.last_ms = elapsed.value
        for name, array in outputs.items():
            target = _P(array.ctypes.data)
            self._cuda.call(
                "cudaMemcpy", target, targets[name], array.nbytes, _MEMCPY_DEVICE_TO_HOST
            )
        return outputs

    def close(self) -> None:
        """Free device buffers, events and the stream; safe to call more than once.

        Every release is attempted even if an earlier one fails; the first CUDA error is
        raised afterwards.
        """
        if self._context is None:
            return
        calls = [
            ("cudaStreamSynchronize", self._stream),
            *(("cudaFree", pointer) for pointer, _ in self._buffers.values()),
            ("cudaEventDestroy", self._start),
            ("cudaEventDestroy", self._end),
            ("cudaStreamDestroy", self._stream),
        ]
        self._buffers.clear()
        errors = []
        for name, handle in calls:
            if not handle.value:
                continue
            try:
                self._cuda.call(name, handle)
            except RuntimeError as exc:
                errors.append(exc)
        self._context = self._engine = self._runtime = None
        if errors:
            raise errors[0]

    def __enter__(self) -> Engine:
        """Return the engine itself."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the engine."""
        self.close()


class EngineModel:
    """Expose ``det`` and ``crop`` engines as the model interface ``infer_image`` expects.

    The crop engine accepts at most the ``max_det`` it was built for, so call ``infer_image``
    with ``max_det`` no larger than the bundle's recorded ``max_det``.

    Attributes:
        keypoints_3d_fingerprint: Always ``None``; engines carry no keypoint layout.
    """

    keypoints_3d_fingerprint = None

    def __init__(self, engines: Mapping[str, Any]) -> None:
        """Wrap a mapping with ``det`` and ``crop`` engines (anything with ``run``)."""
        self.engines = engines

    def _call(self, branch: str, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.engines[branch].run(x.detach().cpu().numpy())
        return {name: torch.from_numpy(value) for name, value in outputs.items()}

    def forward_det(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the detector engine on a ``(1, 3, S, S)`` tensor."""
        return self._call("det", x)

    def forward_crop(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the crop engine on an ``(N, 3, S, S)`` tensor."""
        return self._call("crop", x)
