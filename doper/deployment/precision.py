# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accuracy-first reduced-precision policies for TensorRT layers and ONNX MatMuls.

A policy names which layers must stay in FP32 when BF16 is enabled:

* ``fp32``: every layer.
* ``norm_heads``: normalization layers and every layer of a head.
* ``matmul_only``: everything except matrix multiplies (only MatMuls run in BF16).
* ``heads_only``: everything outside the heads, plus normalizations inside them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import onnx

POLICIES = ("fp32", "norm_heads", "matmul_only", "heads_only")
MATRIX_MULTIPLY = "MATRIX_MULTIPLY"


def _is_norm(name: str, kind: str) -> bool:
    return "NORM" in kind or "norm" in name.lower()


def fp32_layer(name: str, kind: str, policy: str) -> bool:
    """Return whether a layer must stay in FP32 under ``policy``.

    Args:
        name: Layer (or ONNX node) name.
        kind: TensorRT layer type name, e.g. ``MATRIX_MULTIPLY`` or ``NORMALIZATION``.
        policy: One of :data:`POLICIES`.

    Raises:
        ValueError: For an unknown policy.
    """
    if policy == "fp32":
        return True
    if policy == "norm_heads":
        return _is_norm(name, kind) or "head" in name.lower()
    if policy == "matmul_only":
        return kind != MATRIX_MULTIPLY
    if policy == "heads_only":
        return "head" not in name.lower() or _is_norm(name, kind)
    raise ValueError(f"Unknown precision policy {policy!r}; expected one of {POLICIES}")


def cast_matmuls_bf16(model: onnx.ModelProto, policy: str) -> list[str]:
    """Run the MatMuls ``policy`` releases in BF16 by wrapping them in Casts, in place.

    Each selected MatMul gets its inputs cast to BF16 and its outputs cast back to FP32,
    so the graph stays explicitly typed and every other node stays FP32.

    Args:
        model: ONNX model, modified in place.
        policy: One of :data:`POLICIES`.

    Returns:
        Names of the MatMul nodes moved to BF16.
    """
    from onnx import TensorProto, helper

    nodes = []
    changed = []
    for i, node in enumerate(model.graph.node):
        if node.op_type != "MatMul" or fp32_layer(node.name, MATRIX_MULTIPLY, policy):
            nodes.append(node)
            continue
        changed.append(node.name)
        for j, source in enumerate(list(node.input)):
            target = f"precision_{i}_input_{j}_bf16"
            nodes.append(
                helper.make_node("Cast", [source], [target], name=target, to=TensorProto.BFLOAT16)
            )
            node.input[j] = target
        outputs = list(node.output)
        for j in range(len(outputs)):
            node.output[j] = f"precision_{i}_output_{j}_bf16"
        nodes.append(node)
        for j, target in enumerate(outputs):
            nodes.append(
                helper.make_node(
                    "Cast",
                    [node.output[j]],
                    [target],
                    name=f"precision_{i}_restore_{j}",
                    to=TensorProto.FLOAT,
                )
            )
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return changed
