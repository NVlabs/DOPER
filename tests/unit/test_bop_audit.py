# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the independent geometry, projection and checkpoint audits."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import trimesh
from bop_pbr_data import HEIGHT, WIDTH, make_pbr, write_keypoints

from doper.bop import audit, convert

OBJECT_ID = 2
KEYPOINT_COUNT = 64


@pytest.fixture
def bop_fixture(tmp_path: Path) -> dict:
    """An off-origin sphere CAD with 64 keypoints on its vertices (meters)."""
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=100)
    mesh.apply_translation([30, -20, 10])
    path = tmp_path / "object.ply"
    mesh.export(path)
    vertices = np.asarray(mesh.vertices)
    chosen = np.random.default_rng(42).choice(len(vertices), KEYPOINT_COUNT, replace=False)
    points = vertices[chosen] / 1000
    keypoints = tmp_path / "keypoints.json"
    keypoints.write_text(json.dumps({"keypoints_3d": points.tolist()}))
    info = {str(OBJECT_ID): dict(zip(("size_x", "size_y", "size_z"), mesh.extents, strict=True))}
    return {"mesh": path, "keypoints": keypoints, "points": points, "info": info}


def test_audit_accepts_original_bop_frame_and_meters(bop_fixture: dict) -> None:
    f = bop_fixture
    assert audit.geometry_audit(f["mesh"], f["keypoints"], f["info"], OBJECT_ID) == []
    report = audit.geometry_report(f["mesh"], f["keypoints"], f["info"], OBJECT_ID)
    assert report["max_vertex_distance_mm"] < 0.05


@pytest.mark.parametrize("change", ["mm_instead_of_m", "centered"])
def test_audit_flags_keypoints_off_mesh(bop_fixture: dict, change: str) -> None:
    f = bop_fixture
    bad = f["points"] * 1000 if change == "mm_instead_of_m" else f["points"] - [0.03, -0.02, 0.01]
    f["keypoints"].write_text(json.dumps({"keypoints_3d": bad.tolist()}))
    problems = audit.geometry_audit(f["mesh"], f["keypoints"], f["info"], OBJECT_ID)
    assert problems and "scale/frame" in problems[0]


def test_audit_flags_cad_extent_that_disagrees_with_models_info(bop_fixture: dict) -> None:
    f = bop_fixture
    f["info"][str(OBJECT_ID)]["size_x"] += 50
    problems = audit.geometry_audit(f["mesh"], f["keypoints"], f["info"], OBJECT_ID)
    assert any("models_info" in problem for problem in problems)


def test_audit_rejects_nonfinite_keypoints(bop_fixture: dict) -> None:
    f = bop_fixture
    f["keypoints"].write_text('{"keypoints_3d": [[NaN, 0, 0]]}')
    with pytest.raises(ValueError, match="finite"):
        audit.geometry_audit(f["mesh"], f["keypoints"], f["info"], OBJECT_ID)


def test_ascii_edges_do_not_change_vertices_or_faces(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "edge.ply"
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\n"
        "property float z\nelement face 1\nproperty list uchar uint vertex_indices\n"
        "element edge 1\nproperty int vertex1\nproperty int vertex2\nend_header\n"
        "0 0 0\n10 0 0\n0 10 0\n3 0 1 2\n0 1\n"
    )
    original = trimesh.load

    def loader(value, **kwargs):
        if value == path:
            raise AttributeError("'dict' object has no attribute 'dtype'")
        return original(value, **kwargs)

    monkeypatch.setattr(trimesh, "load", loader)
    mesh = audit.load_bop_mesh(path)
    assert np.array_equal(mesh.vertices, [[0, 0, 0], [10, 0, 0], [0, 10, 0]])
    assert np.array_equal(mesh.faces, [[0, 1, 2]])


def test_ascii_ply_row_count_mismatch_refuses_repair(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "bad.ply"
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\n"
        "property float z\nelement edge 1\nproperty int vertex1\nproperty int vertex2\n"
        "end_header\n0 0 0\n"
    )
    monkeypatch.setattr(
        trimesh, "load", lambda *a, **k: (_ for _ in ()).throw(AttributeError("dtype"))
    )
    with pytest.raises(ValueError, match="row count"):
        audit.load_bop_mesh(path)


@pytest.fixture
def converted(tmp_path: Path) -> tuple[Path, Path]:
    """A converted root and the ``train_pbr`` it came from."""
    pbr = make_pbr(tmp_path)
    keypoints = write_keypoints(tmp_path / "kp.json")
    out = convert.convert_pbr(pbr, OBJECT_ID, keypoints, tmp_path / "out", min_visible_keypoints=1)
    return out.out, pbr


def test_projection_audit_accepts_consistent_conversion(converted, tmp_path: Path) -> None:
    out, pbr = converted
    overlays = tmp_path / "overlays"
    assert audit.projection_audit(out, pbr, OBJECT_ID, overlays) == []
    report = audit.projection_report(out, pbr, OBJECT_ID)
    assert report["max_reprojection_error_px"] < 0.1
    assert report["annotation_count"] == 9
    assert sorted(p.name for p in overlays.iterdir())


def test_projection_audit_flags_shifted_labels_and_wrong_object(converted, tmp_path: Path) -> None:
    out, pbr = converted
    data = json.loads((out / "annotations.json").read_text())
    for annotation in data["annotations"]:
        annotation["keypoints"][0] += 3.0
    (out / "annotations.json").write_text(json.dumps(data))
    assert any("reprojection" in p for p in audit.projection_audit(out, pbr, OBJECT_ID))
    assert any("object" in p for p in audit.projection_audit(out, pbr, 7))


def _blank_first_mask(out: Path, pbr: Path) -> dict:
    annotation = json.loads((out / "annotations.json").read_text())["annotations"][0]
    bop = annotation["bop"]
    zero = np.zeros((HEIGHT, WIDTH), np.uint8)
    source_mask = (
        pbr / bop["scene"] / "mask_visib" / f"{bop['frame']:06d}_{bop['instance']:06d}.png"
    )
    for path in (source_mask, out / annotation["mask_file"]):
        path.unlink()
        assert cv2.imwrite(str(path), zero)
    return annotation


def test_empty_mask_needs_confirmation_from_the_source_gt_info(converted) -> None:
    out, pbr = converted
    annotation = _blank_first_mask(out, pbr)
    assert any("empty" in p for p in audit.projection_audit(out, pbr, OBJECT_ID))
    bop = annotation["bop"]
    info_path = pbr / bop["scene"] / "scene_gt_info.json"
    info = {str(bop["frame"]): [{"px_count_visib": 0, "visib_fract": 0}] * (bop["instance"] + 1)}
    info_path.write_text(json.dumps(info))
    report = audit.projection_report(out, pbr, OBJECT_ID, allow_source_empty_masks=True)
    assert [row["annotation_id"] for row in report["source_empty_visible_masks"]] == [
        annotation["id"]
    ]
    info[str(bop["frame"])][bop["instance"]] = {"px_count_visib": 5, "visib_fract": 0.1}
    info_path.write_text(json.dumps(info))
    problems = audit.projection_audit(out, pbr, OBJECT_ID, allow_source_empty_masks=True)
    assert any("GT-info" in p for p in problems)


def _checkpoint(path: Path, *, nan: bool = False, state: bool = True) -> Path:
    payload = {
        "model": {"weight": torch.tensor([float("nan") if nan else 1.0])},
        "optimizer": {"state": {0: {"step": torch.tensor(3.0)}} if state else {}},
    }
    torch.save(payload, path)
    return path


def test_checkpoint_is_finite_rejects_nan_inf_and_untrained(tmp_path: Path) -> None:
    assert audit.checkpoint_is_finite(_checkpoint(tmp_path / "ok.pth"))
    assert not audit.checkpoint_is_finite(_checkpoint(tmp_path / "nan.pth", nan=True))
    assert not audit.checkpoint_is_finite(_checkpoint(tmp_path / "fresh.pth", state=False))


def test_is_finite_tree_walks_containers() -> None:
    assert audit.is_finite_tree({"a": [torch.ones(2), (1.0, "x", 3)]})
    assert not audit.is_finite_tree({"a": [{"b": float("inf")}]})
    assert not audit.is_finite_tree([torch.tensor([float("nan")])])


def test_audit_accepts_a_keypoint_count_other_than_64(bop_fixture: dict) -> None:
    f = bop_fixture
    f["keypoints"].write_text(json.dumps({"keypoints_3d": f["points"][:25].tolist()}))
    assert audit.geometry_audit(f["mesh"], f["keypoints"], f["info"], OBJECT_ID) == []


def test_audit_rejects_empty_keypoints(bop_fixture: dict) -> None:
    f = bop_fixture
    f["keypoints"].write_text(json.dumps({"keypoints_3d": []}))
    with pytest.raises(ValueError, match="finite ordered"):
        audit.geometry_audit(f["mesh"], f["keypoints"], f["info"], OBJECT_ID)
