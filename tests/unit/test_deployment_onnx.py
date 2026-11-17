# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exported ONNX branches match PyTorch, and the BF16 MatMul cast policy."""

import numpy as np
import pytest
import torch
from tiny_coco import write_deployment_inputs

from doper.config import DataConfig, DoperConfig
from doper.deployment import precision
from doper.deployment.onnx_export import BranchModule, export_bundle
from doper.infer import load_model

ort = pytest.importorskip("onnxruntime")
onnx = pytest.importorskip("onnx")


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    inputs = write_deployment_inputs(tmp_path_factory.mktemp("inputs"))
    output = export_bundle(
        inputs["checkpoint"],
        inputs["keypoints_3d"],
        inputs["camera"],
        inputs["frames"],
        tmp_path_factory.mktemp("export") / "bundle",
        score_thr=0.0,
    )
    model = load_model(inputs["checkpoint"], DoperConfig(data=DataConfig(root="")), "cpu")
    return output, model, inputs["model_cfg"]


def _run(path, x):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [output.name for output in session.get_outputs()]
    return names, session.run(None, {"input": x})


@pytest.mark.parametrize(("branch", "batch"), [("det", 1), ("crop", 1), ("crop", 3)])
def test_onnx_matches_torch_det_and_crop(exported, branch, batch):
    output, model, model_cfg = exported
    size = model_cfg.det_input_size if branch == "det" else model_cfg.crop_size
    x = np.random.default_rng(batch).standard_normal((batch, 3, size, size)).astype(np.float32)
    names, actual = _run(output / f"{branch}.onnx", x)
    wrapper = BranchModule(model, branch)
    assert names == wrapper.output_names
    with torch.inference_mode():
        expected = wrapper(torch.from_numpy(x))
    for got, want in zip(actual, expected, strict=True):
        assert got.shape[0] == batch
        np.testing.assert_allclose(got, want.numpy(), atol=1e-4, rtol=0)


def test_onnx_matches_recorded_references(exported):
    output = exported[0]
    for reference in sorted(output.glob("ref*_*.npz")):
        branch = reference.stem.split("_")[1]
        with np.load(reference) as recorded:
            names, actual = _run(output / f"{branch}.onnx", recorded["input"])
            for name, value in zip(names, actual, strict=True):
                np.testing.assert_allclose(value, recorded[name], atol=1e-4, rtol=0)


def test_branch_module_rejects_unknown_branch(exported):
    with pytest.raises(ValueError, match="branch"):
        BranchModule(exported[1], "kpt")


@pytest.mark.parametrize(
    ("name", "kind", "policy", "fp32"),
    [
        ("/backbone/MatMul", "MATRIX_MULTIPLY", "fp32", True),
        ("/backbone/MatMul", "MATRIX_MULTIPLY", "matmul_only", False),
        ("/backbone/Conv", "CONVOLUTION", "matmul_only", True),
        ("/backbone/norm/Mul", "ELEMENTWISE", "norm_heads", True),
        ("/kpt_head/out/Conv", "CONVOLUTION", "norm_heads", True),
        ("/backbone/Conv", "CONVOLUTION", "norm_heads", False),
        ("/kpt_head/out/Conv", "CONVOLUTION", "heads_only", False),
        ("/kpt_head/norm/Mul", "ELEMENTWISE", "heads_only", True),
        ("/backbone/Conv", "CONVOLUTION", "heads_only", True),
    ],
)
def test_fp32_layer_policies(name, kind, policy, fp32):
    assert precision.fp32_layer(name, kind, policy) is fp32


def test_fp32_layer_rejects_unknown_policy():
    with pytest.raises(ValueError, match="last_matmul"):
        precision.fp32_layer("x", "MATRIX_MULTIPLY", "last_matmul")


def test_cast_matmuls_bf16_wraps_selected_matmuls():
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("MatMul", ["a", "b"], ["y"], name="/head/MatMul")],
        "g",
        [
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [3, 4]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    assert precision.cast_matmuls_bf16(model, "fp32") == []
    assert precision.cast_matmuls_bf16(model, "matmul_only") == ["/head/MatMul"]
    onnx.checker.check_model(model)
    ops = [node.op_type for node in model.graph.node]
    assert ops == ["Cast", "Cast", "MatMul", "Cast"]
    assert model.graph.node[-1].output[0] == "y"
