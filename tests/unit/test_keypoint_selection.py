# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hull-anchored keypoint selector: closure tests ported from the source, units, and the CLI."""

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from doper import keypoint_selection as ks
from doper.keypoint_selection import KeypointLayout, load_mesh, main, select_keypoints
from tests.stl_helpers import write_binary_stl

BRACKET_PROFILE_MM = ((0, 0), (60, 0), (60, 8), (8, 8), (8, 20), (0, 20))
BRACKET_DEPTH_MM = 40.0
GRID = 4
CUBE_VERTICES = np.array(
    [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)] + [[0, 0, 0]], dtype=np.float64
)
CUBE_FACES = np.array(
    [
        [0, 2, 3], [0, 3, 1],  # x = -1
        [4, 5, 7], [4, 7, 6],  # x = 1
        [0, 1, 5], [0, 5, 4],  # y = -1
        [2, 6, 7], [2, 7, 3],  # y = 1
        [0, 4, 6], [0, 6, 2],  # z = -1
        [1, 3, 7], [1, 7, 5],  # z = 1
    ],
    dtype=np.int64,
)  # fmt: skip


def grid_triangles(origin: np.ndarray, u: np.ndarray, v: np.ndarray) -> list[np.ndarray]:
    """Triangulate ``origin + s * u + t * v`` on a regular grid."""
    steps = np.linspace(0.0, 1.0, GRID + 1)
    triangles = []
    for i in range(GRID):
        for j in range(GRID):
            p00, p10, p01, p11 = (
                origin + s * u + t * v
                for s, t in (
                    (steps[i], steps[j]), (steps[i + 1], steps[j]),
                    (steps[i], steps[j + 1]), (steps[i + 1], steps[j + 1]),
                )
            )  # fmt: skip
            triangles += [np.stack([p00, p10, p11]), np.stack([p00, p11, p01])]
    return triangles


def bracket_triangles_mm() -> np.ndarray:
    """Return an asymmetric L-bracket (60 x 40 x 20 mm) as ``(T, 3, 3)`` triangles."""
    profile = [np.array([x, 0.0, z]) for x, z in BRACKET_PROFILE_MM]
    along_y = np.array([0.0, BRACKET_DEPTH_MM, 0.0])
    triangles: list[np.ndarray] = []
    for start, end in zip(profile, profile[1:] + profile[:1], strict=True):
        triangles += grid_triangles(start + along_y, end - start, -along_y)
    for origin, u, v in (
        (np.array([0.0, 0.0, 0.0]), np.array([60.0, 0.0, 0.0]), np.array([0.0, 0.0, 8.0])),
        (np.array([0.0, 0.0, 8.0]), np.array([8.0, 0.0, 0.0]), np.array([0.0, 0.0, 12.0])),
    ):
        triangles += grid_triangles(origin, u, v)
        triangles += grid_triangles(origin + along_y, v, u)
    vertices = np.stack(triangles)
    vertices[..., 0] *= 1.0 - 0.2 * vertices[..., 1] / BRACKET_DEPTH_MM
    return vertices


def write_ascii_stl(path: Path, triangles: np.ndarray) -> Path:
    """Write ``(T, 3, 3)`` triangles as an ASCII STL at float64 precision."""
    lines = ["solid t"]
    for tri in triangles:
        lines += ["facet normal 0 0 0", "outer loop"]
        lines += [f"vertex {float(x)!r} {float(y)!r} {float(z)!r}" for x, y, z in tri]
        lines += ["endloop", "endfacet"]
    path.write_text("\n".join(lines + ["endsolid t"]) + "\n")
    return path


def bracket(tmp_path: Path, units: str) -> Path:
    """Write the bracket as an ASCII STL in ``units`` (``m`` or ``mm``).

    ASCII keeps float64 values, so the two files agree after the loader's nine-decimal rounding;
    float32 binary files would quantize millimeters and meters differently.
    """
    factor = 0.001 if units == "m" else 1.0
    return write_ascii_stl(tmp_path / f"bracket_{units}.stl", bracket_triangles_mm() * factor)


def notched_plate_triangles_mm() -> np.ndarray:
    """Return a 100 x 60 x 2 mm plate with a 6 mm wide, 40 mm deep slot as ``(T, 3, 3)``."""
    thick = 2.0
    outline = [(0, 0), (100, 0), (100, 27), (60, 27), (60, 33), (100, 33), (100, 60), (0, 60)]
    triangles: list[np.ndarray] = []
    for x0, x1, y0, y1 in ((0, 60, 0, 60), (60, 100, 0, 27), (60, 100, 33, 60)):
        for z, flip in ((thick, False), (0.0, True)):
            quad = [
                np.array([x, y, z], dtype=float)
                for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
            ]
            quad = quad[::-1] if flip else quad
            triangles += [np.stack(quad[:3]), np.stack([quad[0], quad[2], quad[3]])]
    rim = [np.array([x, y, 0.0]) for x, y in outline]
    lift = np.array([0.0, 0.0, thick])
    for a, b in zip(rim, rim[1:] + rim[:1], strict=True):
        triangles += [np.stack([a, b, b + lift]), np.stack([a, b + lift, a + lift])]
    return np.stack(triangles)


def cube_mesh() -> ks.SurfaceMesh:
    """Return the unit cube with a vertex at its center."""
    return ks.SurfaceMesh(vertices=CUBE_VERTICES, faces=CUBE_FACES)


def test_stl_loader_deduplicates_vertices_and_faces(tmp_path):
    triangles = bracket_triangles_mm() * 0.001
    vertices, faces = load_mesh(write_binary_stl(tmp_path / "b.stl", triangles), "m")
    flat = np.round(triangles.astype(np.float32).astype(np.float64).reshape(-1, 3), 9)
    assert len(vertices) == len(np.unique(flat, axis=0))
    assert faces.shape == (len(triangles), 3)
    np.testing.assert_allclose(vertices[faces], np.round(flat.reshape(-1, 3, 3), 9), atol=1e-12)


def test_ascii_stl_matches_binary(tmp_path):
    triangles = bracket_triangles_mm()[:20].astype(np.float32) * 0.001
    text = "solid t\n"
    for tri in triangles:
        text += "facet normal 0 0 0\nouter loop\n"
        text += "".join(f"vertex {float(x)!r} {float(y)!r} {float(z)!r}\n" for x, y, z in tri)
        text += "endloop\nendfacet\n"
    (tmp_path / "a.stl").write_text(text + "endsolid t\n")
    from_ascii = load_mesh(tmp_path / "a.stl", "m")
    from_binary = load_mesh(write_binary_stl(tmp_path / "b.stl", triangles), "m")
    np.testing.assert_array_equal(from_ascii[0], from_binary[0])
    np.testing.assert_array_equal(from_ascii[1], from_binary[1])


def test_obj_and_ply_load_the_same_mesh(tmp_path):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    quad = "f 1/1/1 2/2/2 3/3/3 4/4/4\nf -5 -4 -1\n"
    (tmp_path / "m.obj").write_text(
        "# comment\n" + "".join(f"v {x} {y} {z}\n" for x, y, z in vertices) + quad
    )
    obj_vertices, obj_faces = load_mesh(tmp_path / "m.obj", "mm")
    expected_faces = [[0, 1, 2], [0, 2, 3], [0, 1, 4]]
    np.testing.assert_allclose(obj_vertices, vertices * 0.001)
    assert obj_faces.tolist() == expected_faces
    header = (
        "ply\nformat {fmt} 1.0\nelement vertex 5\nproperty float x\nproperty float y\n"
        "property float z\nelement face 3\nproperty list uchar int vertex_indices\nend_header\n"
    )
    faces = [[0, 1, 2], [0, 2, 3], [0, 1, 4]]
    ascii_body = "".join(f"{x} {y} {z}\n" for x, y, z in vertices)
    ascii_body += "".join(f"3 {a} {b} {c}\n" for a, b, c in faces)
    (tmp_path / "a.ply").write_text(header.format(fmt="ascii") + ascii_body)
    binary = header.format(fmt="binary_little_endian").encode() + vertices.astype("<f4").tobytes()
    for face in faces:
        binary += struct.pack("<B3i", 3, *face)
    (tmp_path / "b.ply").write_bytes(binary)
    for name in ("a.ply", "b.ply"):
        ply_vertices, ply_faces = load_mesh(tmp_path / name, "mm")
        np.testing.assert_allclose(ply_vertices, vertices * 0.001)
        assert ply_faces.tolist() == expected_faces


def test_load_mesh_rejects_unknown_units_and_formats(tmp_path):
    path = bracket(tmp_path, "m")
    with pytest.raises(ValueError, match="source_units"):
        load_mesh(path, "ft")
    (tmp_path / "m.step").write_text("x")
    with pytest.raises(ValueError, match="unsupported mesh format"):
        load_mesh(tmp_path / "m.step", "m")


def test_surface_saliency_marks_crease_vertices():
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1]], dtype=np.float64
    )
    faces = np.array([[0, 1, 2], [0, 2, 3], [1, 4, 5], [1, 5, 2]], dtype=np.int64)
    saliency = ks.surface_saliency_scores(
        ks.SurfaceMesh(vertices, faces),
        sharpness_weight=1.0,
        normal_variation_weight=0.0,
        boundary_weight=0.0,
    )
    assert saliency[[1, 2]].min() > saliency[[0, 3, 4, 5]].max()


def mesh_visibility_mask(
    mesh: ks.SurfaceMesh, points: np.ndarray, *, eye: np.ndarray, tolerance: float = 1e-6
) -> np.ndarray:
    """Return True for points visible from ``eye`` under mesh self-occlusion."""
    tri_v0, tri_e1, tri_e2 = ks.triangle_edge_arrays(mesh.vertices, mesh.faces)
    return ks.visibility_mask_from_edges(
        points, eye=eye, tri_v0=tri_v0, tri_e1=tri_e1, tri_e2=tri_e2, tolerance=tolerance
    )


def test_mesh_visibility_masks_occluded_cube_points():
    mesh = cube_mesh()
    visible = mesh_visibility_mask(mesh, mesh.vertices, eye=np.array([3.0, 0.0, 0.0]))
    assert np.all(visible[[4, 5, 6, 7]])
    assert not np.any(visible[[0, 1, 2, 3]])


def test_numba_and_numpy_visibility_agree(monkeypatch):
    pytest.importorskip("numba")
    rng = np.random.default_rng(0)
    mesh = ks.SurfaceMesh(
        vertices=np.vstack([CUBE_VERTICES, rng.uniform(-1.5, 1.5, size=(40, 3))]),
        faces=CUBE_FACES,
    )
    eyes = [np.array([3.0, 0.0, 0.0]), np.array([-2.0, 2.5, 1.0]), np.array([0.1, 0.2, -4.0])]
    with_numba = [mesh_visibility_mask(mesh, mesh.vertices, eye=eye) for eye in eyes]
    assert ks.load_numba_kernel() is not None
    monkeypatch.setattr(ks, "load_numba_kernel", lambda: None)
    with_numpy = [mesh_visibility_mask(mesh, mesh.vertices, eye=eye) for eye in eyes]
    for fast, slow in zip(with_numba, with_numpy, strict=True):
        np.testing.assert_array_equal(fast, slow)


def test_farthest_point_indices_from_seed_points_respects_fixed_points():
    points = np.array([[0.0, 0, 0], [0.1, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    selected = ks.farthest_point_indices_from_seed_points(
        points, 2, seed_points=np.array([[0.0, 0.0, 0.0]])
    )
    assert selected.tolist() == [3, 2]


def test_visibility_refinement_replaces_hidden_points():
    mesh = cube_mesh()
    refined, swaps, after = ks.visibility_refined_indices(
        mesh,
        np.array([8, 0, 1, 2]),
        np.linspace(0.0, 1.0, len(mesh.vertices)),
        min_visibility_rate=0.45,
        num_views=6,
        exact_candidate_limit=5,
    )
    assert refined.shape == (4,)
    assert len(swaps) >= 1
    assert after.shape == (4,)
    assert all(swap.new_visibility_rate >= 0.45 for swap in swaps)
    assert all(swap.old_index == 8 or swap.old_index in (0, 1, 2) for swap in swaps)


def test_convex_hull_anchor_selection_returns_visible_anchors():
    mesh = cube_mesh()
    saliency = np.ones(len(mesh.vertices))
    anchors = ks.convex_hull_anchor_indices(mesh, saliency, 4, min_visibility_rate=0.1, num_views=6)
    selected = ks.hull_anchored_saliency_fps_indices(mesh, saliency, 6, anchors, pool_fraction=1.0)
    assert anchors.shape == (4,)
    assert selected.shape == (6,)
    assert len(set(selected.tolist())) == 6


def test_selector_mm_and_m_inputs_agree(tmp_path):
    m = select_keypoints(*load_mesh(bracket(tmp_path, "m"), "m"), num_keypoints=16)
    mm = select_keypoints(*load_mesh(bracket(tmp_path, "mm"), "mm"), num_keypoints=16)
    np.testing.assert_array_equal(m.indices, mm.indices)
    np.testing.assert_allclose(m.points, mm.points, atol=1e-12)


def test_selector_layout_is_consistent(tmp_path):
    vertices, faces = load_mesh(bracket(tmp_path, "m"), "m")
    layout = select_keypoints(vertices, faces, num_keypoints=16, num_hull_anchors=4)
    assert isinstance(layout, KeypointLayout)
    assert layout.points.shape == (16, 3)
    np.testing.assert_array_equal(layout.points, vertices[layout.indices])
    assert len(set(layout.indices.tolist())) == 16
    assert layout.saliency.shape == layout.visibility_rate.shape == (16,)
    assert layout.method == "surface_saliency_fps_hull_anchors"
    with pytest.raises(AttributeError):
        layout.method = "other"  # type: ignore[misc]


def test_initial_anchors_are_hull_vertices(tmp_path):
    from scipy.spatial import ConvexHull

    vertices, faces = load_mesh(bracket(tmp_path, "m"), "m")
    layout = select_keypoints(vertices, faces, num_keypoints=16, num_hull_anchors=6)
    hull = ConvexHull(vertices, qhull_options="QJ")
    hull_indices = set(np.unique(hull.simplices.reshape(-1)).tolist())
    assert len(layout.initial_anchor_indices) == 6
    assert set(layout.initial_anchor_indices.tolist()) <= hull_indices
    replaced = {swap.position for swap in layout.replaced_anchors}
    for position, anchor in enumerate(layout.initial_anchor_indices):
        if position not in replaced:
            assert layout.indices[position] == anchor


def test_selector_validates_arguments(tmp_path):
    vertices, faces = load_mesh(bracket(tmp_path, "m"), "m")
    with pytest.raises(ValueError, match="num_keypoints"):
        select_keypoints(vertices, faces, num_keypoints=0)
    with pytest.raises(ValueError, match="num_hull_anchors"):
        select_keypoints(vertices, faces, num_hull_anchors=0)
    with pytest.raises(ValueError, match="shape"):
        select_keypoints(vertices[:, :2], faces)


def test_cli_writes_json_with_unit_frame_and_hash(tmp_path, capsys):
    mesh = bracket(tmp_path, "mm")
    out_json, out_md = tmp_path / "kp.json", tmp_path / "kp.md"
    argv = ["--mesh", str(mesh), "--source-units", "mm", "--out-json", str(out_json)]
    argv += ["--num-keypoints", "16", "--num-hull-anchors", "4", "--keypoint-prefix", "KP_object_"]
    argv += ["--object-frame", "aabb_centered", "--out-report-md", str(out_md)]
    assert main(argv) == 0
    payload = json.loads(out_json.read_text())
    assert payload["units"] == "m"
    assert payload["source_units"] == "mm"
    assert payload["scale"] == 0.001
    assert payload["object_frame"] == "aabb_centered"
    assert payload["object"] == "bracket_mm"
    assert payload["keypoint_names"] == [f"KP_object_{i:03d}" for i in range(16)]
    points = np.asarray(payload["keypoints_3d"])
    assert points.shape == (16, 3)
    assert payload["num_keypoints"] == 16
    assert payload["coordinates_sha256"] == ks.coordinates_sha256(points)
    assert len(payload["indices"]) == len(payload["initial_anchor_indices"]) + 12
    assert "Hull-Anchored Keypoints" in out_md.read_text()
    assert "wrote" in capsys.readouterr().out.lower()


def test_cli_hash_matches_documented_formula(tmp_path):
    import hashlib

    out_json = tmp_path / "kp.json"
    main(
        ["--mesh", str(bracket(tmp_path, "m")), "--source-units", "m", "--out-json", str(out_json)]
    )
    payload = json.loads(out_json.read_text())
    raw = np.ascontiguousarray(payload["keypoints_3d"], dtype="<f8").tobytes()
    assert payload["coordinates_sha256"] == hashlib.sha256(raw).hexdigest()
    assert payload["num_keypoints"] == 64


def test_cli_requires_source_units(tmp_path):
    with pytest.raises(SystemExit):
        main(["--mesh", str(bracket(tmp_path, "m")), "--out-json", str(tmp_path / "k.json")])


def test_cli_help_documents_keypoint_prefix(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert "object.keypoint_prefix" in help_text


@pytest.mark.filterwarnings("ignore:only .* hull vertices met visibility:RuntimeWarning")
def test_replaced_anchors_are_reported_not_silently_dropped(tmp_path):
    path = write_binary_stl(tmp_path / "plate.stl", notched_plate_triangles_mm() * 0.001)
    vertices, faces = load_mesh(path, "m")
    layout = select_keypoints(
        vertices,
        faces,
        12,
        6,
        anchor_visibility_views=3,
        repair_visibility_views=48,
        min_visibility_rate=0.8,
    )
    assert len(layout.replaced_anchors) >= 1
    for swap in layout.replaced_anchors:
        assert swap.position < len(layout.initial_anchor_indices)
        assert layout.initial_anchor_indices[swap.position] == swap.old_index
        assert layout.indices[swap.position] == swap.new_index
        assert swap.old_index not in layout.indices
        assert swap.new_visibility_rate >= 0.8 > swap.old_visibility_rate
    assert not set(layout.initial_anchor_indices.tolist()) <= set(layout.indices.tolist())
    out_json = tmp_path / "plate.json"
    argv = ["--mesh", str(path), "--source-units", "m", "--out-json", str(out_json)]
    argv += ["--num-keypoints", "12", "--num-hull-anchors", "6", "--anchor-visibility-views", "3"]
    argv += ["--min-visibility-rate", "0.8", "--out-report-md", str(tmp_path / "plate.md")]
    main(argv)
    recorded = json.loads(out_json.read_text())["replaced_anchors"]
    assert [r["new_index"] for r in recorded] == [s.new_index for s in layout.replaced_anchors]
    assert "swapped" in (tmp_path / "plate.md").read_text()


def test_low_visibility_without_candidates_warns_and_keeps_points():
    mesh = cube_mesh()
    with pytest.warns(RuntimeWarning, match="no replacement candidate"):
        refined, swaps, after = ks.visibility_refined_indices(
            mesh, [8], np.ones(len(mesh.vertices)), min_visibility_rate=1.1, num_views=6
        )
    assert refined.tolist() == [8]
    assert swaps == []
    assert after.shape == (1,)


def test_too_few_visible_hull_vertices_warns():
    mesh = cube_mesh()
    with pytest.warns(RuntimeWarning, match="hull vertices met visibility"):
        anchors = ks.convex_hull_anchor_indices(
            mesh, np.ones(len(mesh.vertices)), 4, min_visibility_rate=1.1, num_views=6
        )
    assert len(anchors) == 4


def test_more_anchors_than_keypoints_truncates_and_reports(tmp_path):
    mesh = bracket(tmp_path, "m")
    out_json, out_md = tmp_path / "kp.json", tmp_path / "kp.md"
    argv = ["--mesh", str(mesh), "--source-units", "m", "--out-json", str(out_json)]
    argv += ["--num-keypoints", "4", "--num-hull-anchors", "8", "--out-report-md", str(out_md)]
    assert main(argv) == 0
    payload = json.loads(out_json.read_text())
    assert len(payload["indices"]) == 4
    assert len(payload["initial_anchor_indices"]) == 4
    assert out_md.read_text().count("\n| ") >= 4
