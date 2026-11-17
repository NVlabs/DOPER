# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging

import pytest

from doper.evaluation import synthetic


def test_format_matrix_models_down_splits_across():
    """Matrix output aligns checkpoints and splits while marking missing cells."""
    results = {
        "blender": {"blender": {"kp_err_px": 1.109}, "isaac": {"kp_err_px": 2.5}},
        "isaac": {"blender": {"kp_err_px": 3.0}},  # missing cell renders as '-'
    }
    table = synthetic.format_matrix(results, "kp_err_px").splitlines()
    assert table[0].split() == ["kp_err_px", "blender", "isaac"]
    assert table[1].split() == ["blender", "1.109", "2.500"]
    assert table[2].split() == ["isaac", "3.000", "-"]


def test_main_rejects_bad_names(capsys):
    """Duplicate checkpoint names are reported as CLI usage errors."""
    with pytest.raises(SystemExit):
        synthetic.main(["--checkpoint", "a=/x", "--checkpoint", "a=/y", "--data-root", "r=/d"])
    assert "duplicate" in capsys.readouterr().err


def test_main_writes_strict_json_for_nonfinite_metrics(tmp_path, monkeypatch, caplog):
    """Missing curve metrics become null without altering finite metrics or arrays."""
    monkeypatch.setattr(
        synthetic,
        "evaluate_matrix",
        lambda *args, **kwargs: {
            "model": {
                "data": {
                    "kp_err_px": 2.5,
                    "num_samples": 1,
                    "curve_iou": float("nan"),
                    "curve_distance": float("inf"),
                    "extra": [float("-inf"), 3.0],
                }
            }
        },
    )
    caplog.set_level(logging.INFO, logger=synthetic.__name__)
    out = tmp_path / "new" / "nested" / "metrics.json"
    assert (
        synthetic.main(
            [
                "--checkpoint",
                "model=checkpoint.pth",
                "--data-root",
                "data=dataset",
                "--out-json",
                str(out),
            ]
        )
        == 0
    )

    def reject_constant(value):
        """Reject non-standard JSON constants that permissive decoders otherwise accept."""
        raise ValueError(f"non-standard JSON constant: {value}")

    assert any(
        record.name == synthetic.__name__ and record.getMessage() == f"wrote {out}"
        for record in caplog.records
    )
    payload = json.loads(out.read_text(), parse_constant=reject_constant)
    assert payload["results"]["model"]["data"] == {
        "kp_err_px": 2.5,
        "num_samples": 1,
        "curve_iou": None,
        "curve_distance": None,
        "extra": [None, 3.0],
    }


@pytest.mark.parametrize("annotated", [False, True])
def test_evaluate_matrix_rejects_empty_validation(tmp_path, monkeypatch, annotated):
    """Reject empty annotated populations while preserving nonempty metrics."""
    split = tmp_path / "holdout"
    split.mkdir()
    (split / "annotations.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "unused.png"}],
                "categories": [{"id": 1}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1}] if annotated else [],
            }
        )
    )
    monkeypatch.setattr(synthetic, "load_model", lambda *args: object())
    calls = []

    def score(model, loader, cfg, device, **kwargs):
        """Record the scoring population and return a known metric."""
        calls.append(len(loader.dataset))
        return {"kp_err_px": 2.5}

    monkeypatch.setattr(synthetic, "evaluate", score)
    options = {
        "split": "holdout",
        "batch_size": 1,
        "num_workers": 0,
        "device": "cpu",
        "kp_dark": True,
    }
    checkpoints, roots = [("model", "unused.pth")], [("dataset", str(tmp_path))]
    if annotated:
        result = synthetic.evaluate_matrix(checkpoints, roots, **options)
        assert result == {"model": {"dataset": {"kp_err_px": 2.5, "num_samples": 1}}}
        assert calls == [1]
    else:
        message = "dataset/holdout contains no annotated validation samples"
        with pytest.raises(ValueError, match=message):
            synthetic.evaluate_matrix(checkpoints, roots, **options)
        assert calls == []
