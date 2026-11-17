# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import math
from pathlib import Path

import pytest

from doper.evaluation.io import json_safe, parse_named_inputs, write_json


@pytest.mark.parametrize("option,prefix", [("model", None), ("checkpoint", None), ("run", "model")])
@pytest.mark.parametrize(
    "path",
    [
        "/checkpoints/epoch=10.pth",
        "./epoch=10.pth",
        "../runs/epoch=10.pth",
        r"C:\runs\epoch=10.pth",
        "/tmp/escape=weights.pth",
        r"..\escape=weights.pth",
    ],
)
def test_bare_paths_preserve_equals_without_becoming_output_names(option, prefix, path):
    """Path-like prefixes stay intact and receive safe positional labels."""
    label = f"{prefix or option}0"
    assert parse_named_inputs([path], option, prefix=prefix) == [(label, path)]


@pytest.mark.parametrize("value", [".=weights.pth", "..=weights.pth", "bad\x00name=weights.pth"])
def test_named_inputs_reject_invalid_filename_components(value):
    """Dot segments and NUL labels cannot become result filenames."""
    with pytest.raises(ValueError, match="invalid"):
        parse_named_inputs([value], "data-root")


def test_explicit_name_preserves_equals_in_its_path():
    """Only the first equals sign separates an explicit label from its path."""
    assert parse_named_inputs(["epoch-10.v1=./run=baseline/epoch=10.pth"], "data-root") == [
        ("epoch-10.v1", "./run=baseline/epoch=10.pth")
    ]


def test_named_inputs_accept_names_or_positions_and_reject_bad_labels():
    """Named and positional inputs retain order while invalid labels fail."""
    assert parse_named_inputs(["a=/x", "b=/y"], "checkpoint") == [("a", "/x"), ("b", "/y")]
    assert parse_named_inputs(["/x", "/y"], "checkpoint") == [
        ("checkpoint0", "/x"),
        ("checkpoint1", "/y"),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        parse_named_inputs(["a=/x", "a=/y"], "checkpoint")
    with pytest.raises(ValueError, match="name=path"):
        parse_named_inputs(["=/x"], "checkpoint")


def test_json_safe_nulls_nonfinite_floats_recursively():
    """Non-finite floats become None at any depth; tuples become lists."""
    payload = {"a": math.nan, "b": (1.0, math.inf), "c": {"d": [-math.inf, 2.5]}, "e": "x"}
    assert json_safe(payload) == {"a": None, "b": [1.0, None], "c": {"d": [None, 2.5]}, "e": "x"}


def test_write_json_creates_parents_and_writes_strict_sorted_json(tmp_path: Path):
    """The writer creates directories and never emits a bare NaN or Infinity token."""
    out = tmp_path / "new" / "nested" / "out.json"
    write_json(out, {"b": math.nan, "a": [math.inf, 1.0]})
    text = out.read_text()
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == {"a": [None, 1.0], "b": None}
    assert text.index('"a"') < text.index('"b"')
