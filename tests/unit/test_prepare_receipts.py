# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shared stage-receipt check of the NuRec prepare steps."""

import json
from pathlib import Path

import pytest

from doper.cli.prepare.receipts import StageOutput, clear, keep_step, reuse_or_refuse


def _spec(tmp_path: Path, **kwargs: object) -> StageOutput:
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    (out / "frame.png").write_text("x")
    return StageOutput(out=out, receipt=out / "receipt.json", **kwargs)  # type: ignore[arg-type]


def test_matching_subset_receipt_is_reused(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    spec.receipt.write_text(json.dumps({"a": 1, "extra": 2}))
    assert reuse_or_refuse(spec, {"a": 1}, force=False) == {"a": 1, "extra": 2}


def test_exact_spec_rejects_extra_recorded_keys(tmp_path: Path) -> None:
    spec = _spec(tmp_path, exact=True)
    spec.receipt.write_text(json.dumps({"a": 1, "extra": 2}))
    with pytest.raises(ValueError, match="rerun with --force"):
        reuse_or_refuse(spec, {"a": 1}, force=False)
    spec.receipt.write_text(json.dumps({"a": 1}))
    assert reuse_or_refuse(spec, {"a": 1}, force=False) == {"a": 1}


def test_mismatch_without_force_refuses_with_path_and_verb(tmp_path: Path) -> None:
    spec = _spec(tmp_path, verb="rebuild it", refusal="was made elsewhere")
    spec.receipt.write_text(json.dumps({"a": 2}))
    with pytest.raises(ValueError) as caught:
        reuse_or_refuse(spec, {"a": 1}, force=False)
    assert str(caught.value) == f"{spec.out} was made elsewhere; rerun with --force to rebuild it"


def test_mismatch_with_force_rebuilds(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    spec.receipt.write_text(json.dumps({"a": 2}))
    assert reuse_or_refuse(spec, {"a": 1}, force=True) is None


def test_missing_receipt_clears_unfinished_output(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    assert reuse_or_refuse(spec, {"a": 1}, force=False) is None
    clear(spec)
    assert not spec.out.exists()
    assert not spec.receipt.exists()


def test_clear_out_false_keeps_output(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}")
    spec = _spec(tmp_path, clear_out=False)
    spec = StageOutput(out=spec.out, receipt=receipt, clear_out=False)
    clear(spec)
    assert (spec.out / "frame.png").exists()
    assert not receipt.exists()


def test_force_rebuilds_over_a_corrupt_receipt(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    spec.receipt.write_text("{")
    assert reuse_or_refuse(spec, {"a": 1}, force=True) is None
    assert keep_step(spec.out, spec.receipt, {"a": 1}, True, "made") is False
    assert not spec.out.exists()
    assert not spec.receipt.exists()
