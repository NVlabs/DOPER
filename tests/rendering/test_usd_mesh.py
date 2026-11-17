# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import numpy as np
import pytest

try:
    from pxr import Usd, UsdGeom
except ModuleNotFoundError as exc:
    if exc.name != "pxr":
        raise
    Usd = UsdGeom = None

from rendering.usd_mesh import (
    keypoints_payload,
    read_layer_data,
    read_mesh_usda,
    sanitize_prim_name,
    usd_texture_file,
    write_mesh_usda,
)


def _box_mesh(scale: float = 0.05):
    # Axis-aligned box centered on the origin, quads, like a centered CAD part.
    """Build a deterministic box with quad faces.

    Args:
        scale: Half-width scale used to construct a deterministic box.
    """
    pts = (
        np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
        * scale
        * np.array([1.0, 0.5, 0.25])
    )
    faces = [
        [0, 1, 3, 2],
        [4, 6, 7, 5],
        [0, 4, 5, 1],
        [2, 3, 7, 6],
        [0, 2, 6, 4],
        [1, 5, 7, 3],
    ]
    counts = [len(f) for f in faces]
    indices = [i for f in faces for i in f]
    return pts, counts, indices


def _keypoints(pts: np.ndarray, n: int = 4):
    """Select the first vertices as numbered keypoints.

    Args:
        pts: Box vertices used as deterministic keypoint positions.
        n: Number of ordered vertex keypoints to return.
    """
    names = [f"KP_clip_{i:03d}" for i in range(n)]
    return names, pts[:n].tolist()


def test_usda_round_trip_preserves_mesh_and_keypoints(tmp_path):
    """Preserve mesh topology, coordinate frame and metadata through round-trip.

    Args:
        tmp_path: Temporary directory for generated USDA and JSON artifacts.
    """
    pts, counts, indices = _box_mesh()
    names, kps = _keypoints(pts)
    path = tmp_path / "clip.usda"
    prim = write_mesh_usda(
        str(path),
        object_name="clip",
        points=pts,
        face_vertex_counts=counts,
        face_vertex_indices=indices,
        keypoint_names=names,
        keypoints_3d=kps,
        metadata={"keypoint_prefix": "KP_clip_", "source_scale": 10.0, "aabb_centered": True},
    )
    assert prim == "/clip/mesh"

    data = read_mesh_usda(str(path))
    assert data["default_prim"] == "clip"
    assert data["up_axis"] == "Z"
    assert data["meters_per_unit"] == 1.0
    np.testing.assert_allclose(np.array(data["points"]), pts, atol=1e-6)
    assert data["face_vertex_counts"] == counts
    assert data["face_vertex_indices"] == indices
    assert data["keypoint_names"] == names
    np.testing.assert_allclose(np.array(data["keypoints_3d"]), np.array(kps), atol=1e-6)
    np.testing.assert_allclose(np.array(data["extent"]), [pts.min(0), pts.max(0)], atol=1e-6)

    layer = data["custom_layer_data"]
    assert layer["object"] == "clip"
    assert layer["num_keypoints"] == len(names)
    assert layer["keypoint_prefix"] == "KP_clip_"
    assert layer["source_scale"] == 10.0
    assert layer["aabb_centered"] is True
    assert layer["mesh_prim"] == "/clip/mesh"
    assert layer["keypoint_scope"] == "/clip/keypoints"


def test_usd_keypoints_match_json_payload_frame(tmp_path):
    """The USD keypoints and keypoints_3d.json must be the same numbers.

    Args:
        tmp_path: Temporary directory for generated USDA and JSON artifacts.
    """
    pts, counts, indices = _box_mesh()
    names, kps = _keypoints(pts)
    payload = keypoints_payload("clip", names, kps)
    json_path = tmp_path / "clip_keypoints_3d.json"
    json_path.write_text(json.dumps(payload))
    usd_path = tmp_path / "clip.usda"
    write_mesh_usda(
        str(usd_path),
        object_name="clip",
        points=pts,
        face_vertex_counts=counts,
        face_vertex_indices=indices,
        keypoint_names=names,
        keypoints_3d=kps,
    )

    loaded = json.loads(json_path.read_text())
    data = read_mesh_usda(str(usd_path))
    assert loaded["keypoint_names"] == data["keypoint_names"]
    np.testing.assert_allclose(
        np.array(loaded["keypoints_3d"]), np.array(data["keypoints_3d"]), atol=1e-6
    )
    # Keypoints were sampled on mesh vertices, so each must coincide with one.
    usd_pts = np.array(data["points"])
    for kp in data["keypoints_3d"]:
        assert np.min(np.linalg.norm(usd_pts - np.array(kp), axis=1)) < 1e-6


def test_prim_names_are_sanitized_but_keypoint_names_preserved(tmp_path):
    """Sanitize USD identifiers without changing model keypoint labels.

    Args:
        tmp_path: Temporary directory for generated USDA and JSON artifacts.
    """
    pts, counts, indices = _box_mesh()
    names = ["KP_Odd-Part_000", "KP_Odd-Part_001"]
    path = tmp_path / "odd_part.usda"
    prim = write_mesh_usda(
        str(path),
        object_name="Odd-Part",
        points=pts,
        face_vertex_counts=counts,
        face_vertex_indices=indices,
        keypoint_names=names,
        keypoints_3d=pts[:2].tolist(),
    )
    assert prim == "/Odd_Part/mesh"
    assert sanitize_prim_name("9abc") == "_9abc"
    text = path.read_text()
    assert 'def Xform "KP_Odd_Part_000"' in text
    assert read_mesh_usda(str(path))["keypoint_names"] == names


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"keypoint_names": ["a"], "keypoints_3d": []}, "length mismatch"),
        ({"face_vertex_counts": [3]}, "do not sum"),
        ({"face_vertex_indices": [0, 1, 99, 2] + [0] * 20}, "outside points"),
    ],
)
def test_writer_rejects_inconsistent_input(tmp_path, kwargs, message):
    """Reject inconsistent geometry and keypoint arrays.

    Args:
        tmp_path: Temporary directory for generated USDA and JSON artifacts.
        kwargs: Input override containing the invalid case.
        message: Expected validation error fragment.
    """
    pts, counts, indices = _box_mesh()
    base = dict(
        object_name="clip", points=pts, face_vertex_counts=counts, face_vertex_indices=indices
    )
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        write_mesh_usda(str(tmp_path / "bad.usda"), **base)


@pytest.mark.skipif(Usd is None, reason="OpenUSD Python bindings are optional")
def test_usda_parses_with_pxr_if_available(tmp_path):
    """Read exported points and keypoint transforms with OpenUSD.

    Args:
        tmp_path: Temporary directory for generated USDA and JSON artifacts.
    """
    pts, counts, indices = _box_mesh()
    names, kps = _keypoints(pts)
    path = tmp_path / "clip.usda"
    write_mesh_usda(
        str(path),
        object_name="clip",
        points=pts,
        face_vertex_counts=counts,
        face_vertex_indices=indices,
        keypoint_names=names,
        keypoints_3d=kps,
    )
    stage = Usd.Stage.Open(str(path))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/clip/mesh"))
    np.testing.assert_allclose(np.array(mesh.GetPointsAttr().Get()), pts, atol=1e-6)
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    kp0 = UsdGeom.Xformable(stage.GetPrimAtPath("/clip/keypoints/KP_clip_000"))
    np.testing.assert_allclose(
        np.array(kp0.ComputeLocalToWorldTransform(0).ExtractTranslation()), kps[0], atol=1e-6
    )


def test_textured_usda_carries_uvs_and_a_bound_preview_surface(tmp_path):
    """Preserve face-corner UVs and bind a portable texture material.

    Args:
        tmp_path: Temporary directory for generated USDA and JSON artifacts.
    """
    pts, counts, indices = _box_mesh()
    names, kps = _keypoints(pts)
    uvs = [[(i % 7) / 7.0, (i % 5) / 5.0] for i in range(len(indices))]  # one per face corner
    path = tmp_path / "obj_000002.usda"
    write_mesh_usda(
        str(path),
        object_name="object",
        points=pts.tolist(),
        face_vertex_counts=counts,
        face_vertex_indices=indices,
        keypoint_names=names,
        keypoints_3d=kps,
        uvs=uvs,
        texture_file="obj_000002.png",
    )
    text = path.read_text()
    assert "texCoord2f[] primvars:st = [" in text and 'interpolation = "faceVarying"' in text
    assert 'prepend apiSchemas = ["MaterialBindingAPI"]' in text
    assert "rel material:binding = </object/Looks/Texture>" in text
    assert 'uniform token info:id = "UsdPreviewSurface"' in text
    assert "asset inputs:file = @./obj_000002.png@" in text and 'inputs:varname = "st"' in text
    back = read_mesh_usda(str(path))
    assert back["texture_file"] == "obj_000002.png"
    assert back["custom_layer_data"]["texture_file"] == "obj_000002.png"
    assert np.allclose(back["uvs"], uvs) and len(back["uvs"]) == len(indices)
    assert np.allclose(back["points"], pts) and np.allclose(back["keypoints_3d"], kps)
    # a layer without a texture stays exactly as before
    plain = tmp_path / "plain.usda"
    write_mesh_usda(
        str(plain),
        object_name="object",
        points=pts.tolist(),
        face_vertex_counts=counts,
        face_vertex_indices=indices,
    )
    assert "material:binding" not in plain.read_text() and "primvars:st" not in plain.read_text()
    assert (
        read_mesh_usda(str(plain))["texture_file"] is None
        and read_mesh_usda(str(plain))["uvs"] is None
    )
    if Usd is not None:
        stage = Usd.Stage.Open(str(path))
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/object/mesh"))
        assert len(UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st").Get()) == len(indices)


def test_textured_usda_validates_inputs(tmp_path):
    """Reject texture references without valid face-corner UVs.

    Args:
        tmp_path: Temporary directory for generated USDA and JSON artifacts.
    """
    pts, counts, indices = _box_mesh()
    kw = dict(
        object_name="object",
        points=pts.tolist(),
        face_vertex_counts=counts,
        face_vertex_indices=indices,
    )
    with pytest.raises(ValueError, match="one entry per face corner"):
        write_mesh_usda(str(tmp_path / "a.usda"), uvs=[[0, 0]], **kw)
    with pytest.raises(ValueError, match="requires uvs"):
        write_mesh_usda(str(tmp_path / "b.usda"), texture_file="t.png", **kw)
    with pytest.raises(ValueError, match="relative"):
        write_mesh_usda(
            str(tmp_path / "c.usda"), uvs=[[0, 0]] * len(indices), texture_file="/abs/t.png", **kw
        )


def test_read_layer_data_and_texture_path_only_read_the_header(tmp_path):
    """Read texture metadata without requiring full mesh parsing.

    Args:
        tmp_path: Temporary directory for generated USDA and JSON artifacts.
    """
    pts, counts, indices = _box_mesh()
    textured = tmp_path / "t.usda"
    write_mesh_usda(
        str(textured),
        object_name="object",
        points=pts.tolist(),
        face_vertex_counts=counts,
        face_vertex_indices=indices,
        uvs=[[0.5, 0.5]] * len(indices),
        texture_file="t.png",
        metadata={"source_scale": 0.001, "aabb_centered": True},
    )
    data = read_layer_data(str(textured))
    assert (
        data["texture_file"] == "t.png"
        and data["source_scale"] == 0.001
        and data["aabb_centered"] is True
    )
    assert usd_texture_file(str(textured)) == str(tmp_path / "t.png")
    plain = tmp_path / "p.usda"
    write_mesh_usda(
        str(plain),
        object_name="object",
        points=pts.tolist(),
        face_vertex_counts=counts,
        face_vertex_indices=indices,
    )
    assert usd_texture_file(str(plain)) is None
    (tmp_path / "bad.usda").write_text("not a usda\n")
    with pytest.raises(ValueError, match="usda header"):
        read_layer_data(str(tmp_path / "bad.usda"))


def test_keypoint_names_with_escapes_round_trip(tmp_path):
    points, counts, indices = _box_mesh()
    names = ['KP_"quoted"', "KP_back\\slash", "KP_unicode_é"]
    path = tmp_path / "escaped.usda"
    write_mesh_usda(
        str(path),
        object_name="part",
        points=points,
        face_vertex_counts=counts,
        face_vertex_indices=indices,
        keypoint_names=names,
        keypoints_3d=points[:3].tolist(),
    )
    result = read_mesh_usda(str(path))
    assert result["keypoint_names"] == names
    np.testing.assert_allclose(result["keypoints_3d"], points[:3])


@pytest.mark.parametrize(
    "field,arity",
    [
        ("points", 2),
        ("points", 4),
        ("keypoints_3d", 2),
        ("keypoints_3d", 4),
        ("uvs", 1),
        ("uvs", 3),
    ],
)
def test_writer_rejects_invalid_coordinate_arity_before_writing(tmp_path, field, arity):
    points, counts, indices = _box_mesh()
    values = dict(
        points=points.tolist(),
        face_vertex_counts=counts,
        face_vertex_indices=indices,
        keypoint_names=["KP_0"],
        keypoints_3d=[[0, 0, 0]],
        uvs=[[0, 0] for _ in indices],
    )
    values[field][0] = [0] * arity
    output = tmp_path / "invalid.usda"
    with pytest.raises(ValueError, match=field):
        write_mesh_usda(str(output), object_name="part", **values)
    assert not output.exists()


def test_keypoints_payload_matches_the_host_side_copy() -> None:
    """``doper.scene.labels`` duplicates this payload for host code; both must stay identical."""
    from doper.scene.labels import keypoints_payload as host_keypoints_payload

    names = ["a", "b", "c"]
    # Values on and next to the sixth-decimal rounding boundary, in float32 and float64.
    kps = [
        (0.1234565, -0.0000005, 2.5e-7),
        np.array([1.0000004999, -1.2345675, 0.3333335], dtype=np.float32),
        np.array([0.0000015, 1e-12, -7.0000005]),
    ]
    payload = keypoints_payload("obj", names, kps)
    assert payload == host_keypoints_payload("obj", names, kps)
    assert json.dumps(payload) == json.dumps(host_keypoints_payload("obj", names, kps))
