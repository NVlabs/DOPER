# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shared human review gate: REVIEWED.json binds the overlays and the READY receipt."""

import json
from pathlib import Path

import pytest

from doper.bop import review


def make_target(root: Path, count: int = 2) -> tuple[Path, list[Path]]:
    root.mkdir(parents=True, exist_ok=True)
    ready = root / "READY.json"
    ready.write_text(json.dumps({"labels": count}))
    overlays = []
    for index in range(count):
        overlay = root / "review" / f"{index:03d}.jpg"
        overlay.parent.mkdir(exist_ok=True)
        overlay.write_bytes(f"overlay {index}".encode())
        overlays.append(overlay)
    return ready, overlays


def test_write_binds_ready_and_overlays(tmp_path):
    ready, overlays = make_target(tmp_path)
    path = review.write(tmp_path, ready, overlays, "A. Reviewer")
    assert path == tmp_path / review.REVIEWED_NAME
    record = json.loads(path.read_text())
    assert record["reviewer"] == "A. Reviewer" and record["operator_confirmed"] is True
    assert record["ready"] == "READY.json"
    assert record["ready_sha256"] == review.sha256_file(ready)
    assert sorted(record["reviewed_overlays"]) == ["review/000.jpg", "review/001.jpg"]
    review.verify(tmp_path, ready)
    review.verify(tmp_path, ready, overlays=overlays)


def test_missing_review_rejected(tmp_path):
    ready, _ = make_target(tmp_path)
    with pytest.raises(review.ReviewError, match="no human review"):
        review.verify(tmp_path, ready)


def test_stale_review_rejected(tmp_path):
    ready, overlays = make_target(tmp_path)
    review.write(tmp_path, ready, overlays, "reviewer")
    ready.write_text(json.dumps({"labels": 3}))
    with pytest.raises(review.ReviewError, match="stale"):
        review.verify(tmp_path, ready)


def test_changed_overlay_rejected(tmp_path):
    ready, overlays = make_target(tmp_path)
    review.write(tmp_path, ready, overlays, "reviewer")
    overlays[1].write_bytes(b"repainted")
    with pytest.raises(review.ReviewError, match="changed since"):
        review.verify(tmp_path, ready)


def test_review_must_cover_every_overlay(tmp_path):
    ready, overlays = make_target(tmp_path, 3)
    review.write(tmp_path, ready, overlays[:2], "reviewer")
    with pytest.raises(review.ReviewError, match="cover"):
        review.verify(tmp_path, ready, overlays=overlays)


def test_unconfirmed_review_rejected(tmp_path):
    ready, overlays = make_target(tmp_path)
    path = review.write(tmp_path, ready, overlays, "reviewer")
    record = json.loads(path.read_text())
    path.write_text(json.dumps({**record, "operator_confirmed": False}))
    with pytest.raises(review.ReviewError, match="confirm"):
        review.verify(tmp_path, ready)


def test_write_refuses_bad_input(tmp_path):
    ready, overlays = make_target(tmp_path / "target")
    with pytest.raises(ValueError, match="reviewer"):
        review.write(tmp_path / "target", ready, overlays, "  ")
    with pytest.raises(ValueError, match="no overlays"):
        review.write(tmp_path / "target", ready, [], "reviewer")
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"x")
    with pytest.raises(ValueError, match="inside"):
        review.write(tmp_path / "target", ready, [outside], "reviewer")
    with pytest.raises(ValueError, match="missing"):
        review.write(tmp_path / "target", ready, [tmp_path / "target" / "gone.jpg"], "reviewer")
    with pytest.raises(ValueError, match="missing"):
        review.write(tmp_path / "target", tmp_path / "target" / "NONE.json", overlays, "r")
