# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ONNX deployment bundles and the parity checks used to validate deployed engines.

* ``onnx_export``: ``python -m doper.deployment.onnx_export`` freezes PyTorch references on
  real frames and exports the detector and crop branches to ONNX (needs the ``tensorrt``
  extra for ``onnx``).
* ``bundle``: SHA-256 manifest of a bundle folder; ``verify`` rejects edited files.
* ``parity``: latency summaries, raw-tensor and pose parity, and the pose acceptance gate.
* ``precision``: FP32/BF16 layer-precision policies and the explicit BF16 MatMul cast.
* ``tensorrt_runtime``: engine build and execution through a ``ctypes`` CUDA runtime
  binding (needs TensorRT from the platform).
* ``benchmark``: ``python -m doper.deployment.benchmark`` builds FP32/FP16/BF16 engines for
  one bundle, checks raw and pose parity and records latency; imports TensorRT lazily.

Nothing is imported here, so ``import doper.deployment`` never pulls in onnx, onnxruntime
or TensorRT.
"""
