# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CAD-to-scene registration: keypoint similarity, robot scale audit, edge refinement, review."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from nurec_fixtures import (
    NUREC_TEMPLATE,
    SCENE_SOURCES,
    WORLD_TO_NRE,
    nurec_recipe_data,
    rig_trajectories,
    write_jsonl,
    write_reconstruction,
    write_usdz,
)
from scipy.spatial.transform import Rotation

from doper import cli
from doper.bop.sources import sha256_file
from doper.evaluation.relative_pose import FramePose
from doper.nurec.capture import CaptureImage, ImageSet, PinholeCamera, RobotReference
from doper.nurec.registration import (
    Registration,
    RegistrationNotAcceptedError,
    SimilarityFit,
    TriangleMesh,
    accept_registration,
    fit_keypoint_similarity,
    read_obj_mesh,
    refine_edges,
    resolve_registration,
    restart_audit,
    robot_poses_in_scene,
    robot_scale_audit,
)
from doper.nurec.scene import NuRecScene

SCALE = 1.085
IMAGE_CAMERA = PinholeCamera(1920, 1440, 2000.0, 2000.0, 959.5, 719.5, True)


def _look_at(centre: np.ndarray, target: np.ndarray) -> np.ndarray:
    """An OpenCV camera pose (camera to world) at ``center`` looking at ``target``."""
    forward = (target - centre) / np.linalg.norm(target - centre)
    down = np.cross(forward, [0.0, 0.0, 1.0])
    down = np.cross(forward, down / np.linalg.norm(down))
    right = np.cross(down, forward)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([right, down, forward], axis=1)
    pose[:3, 3] = centre
    return pose


def _egg_mesh(subdivisions: int = 12) -> TriangleMesh:
    """An irregular hexahedron (meters) with sharp edges, its faces finely subdivided.

    Sharp edges keep the silhouette's generating vertices fixed under small pose changes, as
    on a machined CAD part.
    """
    from scipy.spatial import ConvexHull

    corners = np.array([[-0.032, -0.02, -0.015], [0.05, -0.024, -0.015], [0.044, 0.026, -0.015],
                        [-0.03, 0.018, -0.015], [-0.026, -0.016, 0.012], [0.036, -0.02, 0.02],
                        [0.03, 0.016, 0.017], [-0.022, 0.012, 0.01]])  # fmt: skip
    vertices, faces = [], []
    n = subdivisions
    for a, b, c in ConvexHull(corners).simplices:
        index = {}
        for i in range(n + 1):
            for j in range(n + 1 - i):
                index[i, j] = len(vertices)
                vertices.append(corners[a] + (corners[b] - corners[a]) * i / n
                                + (corners[c] - corners[a]) * j / n)  # fmt: skip
        for i in range(n):
            for j in range(n - i):
                faces.append([index[i, j], index[i + 1, j], index[i, j + 1]])
                if i + j < n - 1:
                    faces.append([index[i + 1, j], index[i + 1, j + 1], index[i, j + 1]])
    return TriangleMesh(np.array(vertices), np.array(faces))


def _true_pose() -> tuple[np.ndarray, np.ndarray]:
    rotation = Rotation.from_rotvec([0.3, -0.5, 0.2]).as_matrix()
    return rotation, np.array([0.01, -0.02, 0.03])


def _render(mesh: TriangleMesh, scene_from_cad, T_cam_scene, camera) -> np.ndarray:  # noqa: N803
    """A gray silhouette of the mesh on a dark background, rasterized with 4 subpixel bits."""
    points = scene_from_cad(mesh.vertices) @ T_cam_scene[:3, :3].T + T_cam_scene[:3, 3]
    uv = points @ camera.matrix.T
    uv = np.round(uv[:, :2] / uv[:, 2:] * 16).astype(np.int32)
    image = np.full((camera.height, camera.width, 3), 30, np.uint8)
    for face in mesh.faces:
        cv2.fillConvexPoly(image, uv[face], (200, 200, 200), lineType=cv2.LINE_8, shift=4)
    return image


def _camera_poses_m(count: int = 4) -> list[np.ndarray]:
    """Camera poses (camera to robot base, meters) around the object."""
    target = np.array([0.01, -0.02, 0.03])
    poses = []
    for i in range(count):
        azimuth = 2 * np.pi * i / count + 0.3
        elevation = 0.5 + 0.25 * (i % 2)
        offset = 0.3 * np.array(
            [
                np.cos(azimuth) * np.cos(elevation),
                np.sin(azimuth) * np.cos(elevation),
                np.sin(elevation),
            ]
        )
        poses.append(_look_at(target + offset, target))
    return poses


def _scene_world(names, poses_m, lineage="s"):
    """The scene of camera poses given in meters: scene units are ``SCALE`` per meter."""
    cameras = {}
    for name, pose in zip(names, poses_m, strict=True):
        scaled = pose.copy()
        scaled[:3, 3] *= SCALE
        cameras[name] = WORLD_TO_NRE @ scaled
    return NuRecScene(
        artifact=Path("last.usdz"), artifact_sha256="a" * 64, world_to_nre=WORLD_TO_NRE,
        cameras=cameras, camera=IMAGE_CAMERA, scene_units_per_metre=SCALE,
        alignment="metric_anchor", reconstruction_id="recon-1",
        lineages=dict.fromkeys(names, lineage),
    )  # fmt: skip


def _truth() -> tuple[np.ndarray, np.ndarray]:
    """The true registration ``scene = SCALE * R @ p + t`` in the NRE frame."""
    rotation, translation = _true_pose()
    return rotation, SCALE * translation + WORLD_TO_NRE[:3, 3]


def _apply(rotation, translation, scale=SCALE):
    return lambda p: scale * np.asarray(p) @ rotation.T + translation


def _project(points_scene: np.ndarray, T_nre_cam: np.ndarray, camera) -> np.ndarray:  # noqa: N803
    cam = np.linalg.inv(T_nre_cam)
    q = (points_scene @ cam[:3, :3].T + cam[:3, 3]) @ camera.matrix.T
    return q[:, :2] / q[:, 2:]


def _frames(scene, keypoints, names, *, reproj=0.5, noise=0.0, seed=0):
    rotation, translation = _truth()
    rng = np.random.default_rng(seed)
    out = []
    for name in names:
        uv = _project(_apply(rotation, translation)(keypoints), scene.cameras[name], IMAGE_CAMERA)
        uv = uv + rng.normal(scale=noise, size=uv.shape) if noise else uv
        out.append(FramePose(image=name, sha256="0" * 64, detections=1, score=0.9, keypoints=uv,
                             keypoint_scores=np.ones(len(uv)), bbox=np.zeros(4),
                             T_camera_object=np.eye(4), reprojection_error_px=reproj))  # fmt: skip
    return out


def _keypoints() -> np.ndarray:
    mesh = _egg_mesh()
    return mesh.vertices[np.random.default_rng(3).choice(len(mesh.vertices), 16, replace=False)]


def test_keypoint_similarity_on_synthetic_scene():
    names = [f"{i:06d}.jpg" for i in range(6)]
    scene = _scene_world(names, _camera_poses_m(6))
    keypoints = _keypoints()
    frames = _frames(scene, keypoints, names, noise=0.2)
    # A wrong frame is excluded by name; a frame with a large PnP error is left out.
    frames[1] = FramePose(**{**frames[1].__dict__, "keypoints": frames[1].keypoints + 40.0})
    frames[4] = FramePose(**{**frames[4].__dict__, "reprojection_error_px": 9.0})
    fit = fit_keypoint_similarity(frames, scene, keypoints, IMAGE_CAMERA.matrix,
                                  exclude_views=[names[1]])  # fmt: skip
    rotation, translation = _truth()
    assert isinstance(fit, SimilarityFit)
    assert fit.views == (names[0], names[2], names[3], names[5])
    assert fit.scale == pytest.approx(SCALE, rel=2e-3)
    angle = np.degrees(Rotation.from_matrix(fit.rotation @ rotation.T).magnitude())
    assert angle < 0.2
    assert np.linalg.norm(fit.translation - translation) < 2e-4
    assert fit.rms_scene < 5e-4
    assert fit.points.shape == (16, 3)
    matrix = fit.as_matrix()
    np.testing.assert_allclose(matrix[:3, :3], fit.rotation)  # unscaled rotation, as recorded
    with pytest.raises(ValueError, match="views"):
        fit_keypoint_similarity(frames[:1], scene, keypoints, IMAGE_CAMERA.matrix)


def test_robot_scale_audit_maps_robot_poses_into_the_scene():
    names = [f"{i:06d}.jpg" for i in range(5)]
    poses_m = _camera_poses_m(5)
    scene = _scene_world(names[:4], poses_m[:4])  # the fifth capture is not in the scene
    mount = np.eye(4)
    mount[:3, 3] = [0.0, 0.0, 0.05]
    robot = RobotReference(T_base_mount={n: p @ np.linalg.inv(mount)
                                         for n, p in zip(names, poses_m, strict=True)},
                           T_mount_camera=mount)  # fmt: skip
    anchor = robot_scale_audit(robot, scene)
    assert anchor.scene_units_per_metre == pytest.approx(SCALE, rel=1e-12)
    assert anchor.n == 4
    assert anchor.rms_scene < 1e-12
    mapped = robot_poses_in_scene(robot, anchor)
    assert list(mapped) == names
    for name in names[:4]:
        np.testing.assert_allclose(mapped[name], scene.cameras[name], atol=1e-12)
    expected = poses_m[4].copy()
    expected[:3, 3] *= SCALE
    np.testing.assert_allclose(mapped[names[4]], WORLD_TO_NRE @ expected, atol=1e-12)


def _edge_world(tmp_path, names, poses_m):
    mesh = _egg_mesh()
    scene = _scene_world(names, poses_m)
    rotation, translation = _truth()
    images = []
    for name in names:
        image = _render(mesh, _apply(rotation, translation), np.linalg.inv(scene.cameras[name]),
                        IMAGE_CAMERA)  # fmt: skip
        path = tmp_path / "images" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image)
        images.append(CaptureImage(name, "0" * 64, "s"))
    return mesh, scene, ImageSet(tmp_path, IMAGE_CAMERA, tuple(images))


def _offset_seed(degrees: float, metres: float) -> SimilarityFit:
    rotation, translation = _truth()
    turn = Rotation.from_rotvec(np.radians(degrees) * np.array([0.6, -0.8, 0.0])).as_matrix()
    shift = metres * np.array([0.0, 0.6, 0.8])
    return SimilarityFit(scale=1.09, rotation=turn @ rotation, translation=translation + shift,
                         rms_scene=0.0, points=np.zeros((0, 3)), views=())  # fmt: skip


def test_edge_refinement_recovers_offset(tmp_path):
    names = [f"{i:06d}.png" for i in range(8)]
    mesh, scene, images = _edge_world(tmp_path, names, _camera_poses_m(8))
    seed = _offset_seed(2.0, 0.002)
    candidate = refine_edges(seed, images, scene, mesh, fixed_scale=SCALE,
                             fit_views=names[0::2], holdout_views=names[1::2])  # fmt: skip
    rotation, translation = _truth()
    refined = candidate.T_nre_object
    angle = np.degrees(Rotation.from_matrix(refined[:3, :3] @ rotation.T).magnitude())
    assert angle < 0.2
    assert np.linalg.norm(refined[:3, 3] - translation) < 0.0002
    assert candidate.scene_units_per_metre == pytest.approx(SCALE, rel=1e-15)
    assert candidate.status == "candidate_requires_review"
    views = candidate.per_view_edges
    assert [v["source"] for v in views] == names
    assert [v["fit_view"] for v in views] == [True, False] * 4
    for view in views:
        assert view["refined_median_px"] < view["seed_median_px"]
        assert view["refined_median_px"] < 1.5
    restarts = restart_audit(candidate, starts=2, rot_deg=1.0, trans=0.001)
    assert [r["start"] for r in restarts] == [0, 1]
    for row in restarts:
        assert row["rotation_from_candidate_deg"] < 0.5
        assert set(row) >= {"translation_from_candidate_scene_mm", "median_fit_edge_px",
                            "median_holdout_edge_px", "cost", "x"}  # fmt: skip


def test_edge_refinement_refuses_bad_views(tmp_path):
    names = [f"{i:06d}.png" for i in range(2)]
    mesh, scene, images = _edge_world(tmp_path, names, _camera_poses_m(2))
    seed = _offset_seed(0.0, 0.0)
    with pytest.raises(ValueError, match="both"):
        refine_edges(seed, images, scene, mesh, fixed_scale=SCALE, fit_views=names,
                     holdout_views=names[:1])  # fmt: skip
    with pytest.raises(ValueError, match="not a camera"):
        refine_edges(seed, images, scene, mesh, fixed_scale=SCALE, fit_views=["x.png"],
                     holdout_views=[])  # fmt: skip


def test_read_obj_mesh_fans_polygons(tmp_path):
    path = tmp_path / "quad.obj"
    path.write_text(
        "# quad\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvn 0 0 1\nf 1/1/1 2/2/1 3/3/1 4/4/1\n"
    )
    mesh = read_obj_mesh(path)
    assert mesh.vertices.shape == (4, 3)
    np.testing.assert_array_equal(mesh.faces, [[0, 1, 2], [0, 2, 3]])
    path.write_text("v 0 0 0\nf 1 2 3\n")
    with pytest.raises(ValueError, match="vertex"):
        read_obj_mesh(path)


def _candidate_for_review(tmp_path):
    names = [f"{i:06d}.png" for i in range(2)]
    mesh, scene, images = _edge_world(tmp_path / "edge", names, _camera_poses_m(2))
    return refine_edges(_offset_seed(0.5, 0.0005), images, scene, mesh, fixed_scale=SCALE,
                        fit_views=names[:1], holdout_views=names[1:], max_nfev=5)  # fmt: skip


def test_labels_need_accepted_registration(tmp_path):
    stage = tmp_path / "registration"
    stage.mkdir()
    candidate = _candidate_for_review(tmp_path)
    candidate.write(stage / "candidate.json")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    # A candidate alone does not label: it needs review.
    with pytest.raises(RegistrationNotAcceptedError, match="--accept-registration"):
        resolve_registration(stage, bundle)
    with pytest.raises(ValueError, match="reviewer note"):
        accept_registration(candidate, "  ")
    accepted = accept_registration(candidate, "overlays checked on 10 views",
                                   out=stage / "registration.json")  # fmt: skip
    assert accepted.status == "accepted"
    resolved = resolve_registration(stage, bundle)
    np.testing.assert_array_equal(resolved.T_nre_object, candidate.T_nre_object)
    assert resolved.reviewer_note == "overlays checked on 10 views"
    assert resolved.scene_units_per_metre == candidate.scene_units_per_metre
    # The registration records the scene it was fitted on.
    assert (resolved.artifact_sha256, resolved.reconstruction_id) == ("a" * 64, "recon-1")
    points = np.array([[0.01, 0.0, 0.0]])
    expected = SCALE * points @ resolved.T_nre_object[:3, :3].T + resolved.T_nre_object[:3, 3]
    np.testing.assert_allclose(resolved.apply(points), expected)
    other = _scene_world(["000000.png", "000001.png"], _camera_poses_m(2))
    other = NuRecScene(**{**other.__dict__, "reconstruction_id": "another"})
    with pytest.raises(ValueError, match="reconstruction"):
        resolve_registration(stage, bundle, scene=other)
    other = NuRecScene(**{**other.__dict__, "reconstruction_id": "recon-1",
                          "artifact_sha256": "b" * 64})  # fmt: skip
    with pytest.raises(ValueError, match="artifact"):
        resolve_registration(stage, bundle, scene=other)


def test_bundle_registration_is_validated(tmp_path):
    stage = tmp_path / "registration"
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    rotation, translation = _truth()
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, translation
    record = {"status": "reviewed", "method": "reference", "T_nre_object": transform.tolist(),
              "scene_units_per_metre": SCALE, "artifact_sha256": "a" * 64,
              "reconstruction_id": "reference"}  # fmt: skip
    (bundle / "registration.json").write_text(json.dumps(record))
    resolved = resolve_registration(stage, bundle)
    assert resolved.status == "reviewed"
    # A reviewed bundle registration is bound by its artifact, not its recorded reconstruction.
    scene = _scene_world(["000000.png"], _camera_poses_m(1))
    assert resolve_registration(stage, bundle, scene=scene).reconstruction_id == "reference"
    with pytest.raises(ValueError, match="artifact"):
        resolve_registration(
            stage, bundle, scene=NuRecScene(**{**scene.__dict__, "artifact_sha256": "c" * 64})
        )
    np.testing.assert_array_equal(resolved.T_nre_object, transform)
    bad = [({"T_nre_object": (transform @ np.diag([1.0, 1.0, -1.0, 1.0])).tolist()}, "rotation"),
           ({"T_nre_object": (transform * 2.0).tolist()}, "rotation"),
           ({"scene_units_per_metre": -1.0}, "positive"),
           ({"scene_units_per_metre": float("nan")}, "positive"),
           ({"status": "candidate_requires_review"}, "status"),
           ({"artifact_sha256": "A" * 64}, "artifact_sha256"),
           ({"extra": 1}, "unknown")]  # fmt: skip
    missing = {k: v for k, v in record.items() if k != "artifact_sha256"}
    (bundle / "registration.json").write_text(json.dumps(missing))
    with pytest.raises(ValueError, match="missing keys"):
        Registration.read(bundle / "registration.json")
    for change, match in bad:
        (bundle / "registration.json").write_text(json.dumps({**record, **change}))
        with pytest.raises(ValueError, match=match):
            Registration.read(bundle / "registration.json")


# --- the register stage -------------------------------------------------------------------


def _write_obj(path: Path, mesh: TriangleMesh) -> None:
    lines = [f"v {float(x)!r} {float(y)!r} {float(z)!r}" for x, y, z in mesh.vertices]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in mesh.faces]
    path.write_text("\n".join(lines) + "\n")


def _register_world(tmp_path):
    """A capture bundle, its capture stage, a reconstruction, an artifact and keypoint frames
    of the egg seen by the four scene sources, with robot poses in meters."""
    import hashlib

    import yaml

    mesh, names = _egg_mesh(), list(SCENE_SOURCES)
    poses_m = _camera_poses_m(4)
    scene = _scene_world(names, poses_m)
    rotation, translation = _truth()
    bundle = tmp_path / "bundle"
    (bundle / "images").mkdir(parents=True)
    rows = []
    for name in names:
        image = _render(mesh, _apply(rotation, translation), np.linalg.inv(scene.cameras[name]),
                        IMAGE_CAMERA)  # fmt: skip
        cv2.imwrite(str(bundle / "images" / name), image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        sha = hashlib.sha256((bundle / "images" / name).read_bytes()).hexdigest()
        rows.append({"name": name, "sha256": sha, "lineage": "fixture"})
    (bundle / "session.json").write_text(json.dumps({"camera": IMAGE_CAMERA.as_record()}))
    write_jsonl(bundle / "images.jsonl", rows)
    write_jsonl(bundle / "robot.jsonl", [{"name": n, "T_base_mount": p.tolist()}
                                         for n, p in zip(names, poses_m, strict=True)])  # fmt: skip
    (bundle / "handeye.json").write_text(json.dumps({"T_mount_camera": np.eye(4).tolist()}))
    keypoints = _keypoints()
    (bundle / "keypoints_3d.json").write_text(json.dumps({"keypoints_3d": keypoints.tolist()}))
    data = nurec_recipe_data()
    _write_obj(bundle / data["object"]["mesh"], mesh)
    data["object"]["mesh_sha256"] = sha256_file(bundle / data["object"]["mesh"])
    data["object"]["keypoints_points_sha256"] = hashlib.sha256(
        np.asarray(keypoints, dtype="<f8").tobytes()).hexdigest()  # fmt: skip
    data["render"].update(count=4, seed_views=[0, 1])
    data["registration"].update(exclude_views=[], fit_views=[0, 2], holdout_views=[1, 3])
    data["registration"]["restarts"].update(starts=2)
    recipe = tmp_path / "nurec-test.yaml"
    recipe.write_text(yaml.safe_dump(data))
    root = tmp_path / "root"
    assert cli.main(["prepare", str(recipe), "--stage", "capture", "--bundle", str(bundle),
                     "--root", str(root)]) == 0  # fmt: skip
    cam_from_world = {}
    for name, pose in zip(names, poses_m, strict=True):
        scaled = pose.copy()
        scaled[:3, 3] *= SCALE
        cam_from_world[name] = np.linalg.inv(scaled)
    recon = write_reconstruction(root / "reconstruction", poses=cam_from_world)
    world = [np.linalg.inv(recon["cam_from_world"][name]) for name in names]
    artifact = write_usdz(tmp_path / "artifact" / "last.usdz", rig_trajectories(world))
    frames = [FramePose(**{**f.__dict__, "sha256": r["sha256"]})
              for f, r in zip(_frames(scene, keypoints, names), rows, strict=True)]  # fmt: skip
    frames_path = tmp_path / "evaluation" / "frames.json"
    frames_path.parent.mkdir()
    frames_path.write_text(json.dumps([f.as_record(np.eye(4)) for f in frames]))
    return str(recipe), root, artifact, frames_path, bundle


def test_prepare_register_writes_a_candidate_then_accepts_it(tmp_path, capsys, monkeypatch):
    from doper.nurec import render as render_module

    monkeypatch.setattr(render_module.shutil, "which", lambda name: None)
    recipe, root, artifact, frames, bundle = _register_world(tmp_path)
    argv = ["prepare", recipe, "--stage", "register", "--root", str(root)]
    assert cli.main([*argv, "--artifact", str(artifact)]) == 1
    assert "--keypoint-frames" in capsys.readouterr().out
    run = [*argv, "--artifact", str(artifact), "--keypoint-frames", str(frames)]
    assert cli.main(run) == 0
    out = capsys.readouterr().out
    assert "candidate" in out and "--accept-registration" in out
    stage = root / "registration"
    candidate = json.loads((stage / "candidate.json").read_text())
    assert candidate["status"] == "candidate_requires_review"
    assert candidate["scale_source"] == "robot_audit"
    assert candidate["capture_flags"]["robot_reference"] is True
    assert candidate["scene_units_per_metre"] == pytest.approx(SCALE, rel=1e-6)
    rotation, translation = _truth()
    refined = np.array(candidate["T_nre_object"])
    assert np.degrees(Rotation.from_matrix(refined[:3, :3] @ rotation.T).magnitude()) < 0.5
    assert np.linalg.norm(refined[:3, 3] - translation) < 0.001
    for name in ("triangulation.json", "robot_scale_audit.json", "restart_audit.json"):
        assert (stage / name).is_file()
    assert not (stage / "registration.json").exists()
    with pytest.raises(RegistrationNotAcceptedError):
        resolve_registration(stage, bundle)
    # A rerun keeps the candidate; acceptance needs a reviewer note and no recomputation flags.
    assert cli.main(run) == 0
    assert "already registered" in capsys.readouterr().out
    assert cli.main([*argv, "--accept-registration"]) == 1
    assert "--reviewer-note" in capsys.readouterr().out
    assert cli.main([*run, "--accept-registration", "--reviewer-note", "ok"]) == 1
    assert "--artifact" in capsys.readouterr().out
    assert cli.main([*argv, "--accept-registration", "--reviewer-note", "overlays ok"]) == 0
    registration = resolve_registration(stage, bundle)
    assert registration.status == "accepted"
    assert registration.reviewer_note == "overlays ok"
    np.testing.assert_array_equal(registration.T_nre_object, refined)
    record = json.loads((stage / "registration.json").read_text())
    assert record["provenance"]["capture_flags"]["robot_reference"] is True
    # The render stage now rejects against the robot poses mapped by the robot audit.
    render = ["prepare", recipe, "--stage", "render", "--root", str(root), "--artifact",
              str(artifact)]  # fmt: skip
    assert cli.main(render) == 1  # planned; no docker to render
    sampling = json.loads((root / "render" / "sampling.json").read_text())
    assert sampling["source_count"] == 8
    assert sampling["scale_source"] == "robot_audit"
    audit = json.loads((stage / "robot_scale_audit.json").read_text())
    assert sampling["scene_units_per_metre"] == audit["scene_units_per_metre"]
    plan = json.loads((root / "render" / "plan.json").read_text())
    assert plan["extra_sources"] == 4


def test_prepare_register_refuses_a_swapped_mesh(tmp_path, capsys, monkeypatch):
    from doper.nurec import render as render_module

    monkeypatch.setattr(render_module.shutil, "which", lambda name: None)
    recipe, root, artifact, frames, bundle = _register_world(tmp_path)
    mesh = bundle / "object.obj"
    mesh.write_text(mesh.read_text() + "# another revision\n")
    run = ["prepare", recipe, "--stage", "register", "--root", str(root), "--artifact",
           str(artifact), "--keypoint-frames", str(frames)]  # fmt: skip
    assert cli.main(run) == 1
    assert "is not the recipe's mesh" in capsys.readouterr().out
    assert not (root / "registration").exists()


def test_register_flags_are_refused_elsewhere(tmp_path, capsys):
    base = ["prepare", str(NUREC_TEMPLATE), "--root", str(tmp_path), "--dry-run"]
    for stage, flag in [("render", ["--keypoint-frames", "f.json"]),
                        ("capture", ["--accept-registration"]),
                        ("reconstruct", ["--reviewer-note", "x"]),
                        ("register", ["--render-cache", "c"])]:  # fmt: skip
        assert cli.main([*base, "--stage", stage, *flag]) == 1
        assert flag[0] in capsys.readouterr().out


def test_prepare_register_dry_run_touches_nothing(tmp_path, capsys):
    argv = ["prepare", str(NUREC_TEMPLATE), "--stage", "register", "--root", str(tmp_path / "root"),
            "--artifact", str(tmp_path / "a.usdz"), "--keypoint-frames", str(tmp_path / "f.json"),
            "--dry-run"]  # fmt: skip
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "candidate.json" in out
    assert not (tmp_path / "root").exists()


def test_recipe_registration_section():
    import yaml

    from doper.recipes import load_any_recipe
    from doper.recipes.nurec_schema import RestartSpec, parse_nurec_recipe

    recipe = load_any_recipe(NUREC_TEMPLATE)
    spec = recipe.registration
    assert (spec.exclude_views, spec.fit_views, spec.holdout_views) == (
        (), (0, 7, 14, 21, 28), (2, 10, 18, 25, 32))  # fmt: skip
    assert (spec.canny, spec.boundary_px, spec.points, spec.f_scale, spec.diff_step,
            spec.max_nfev) == ((40, 100), 1.5, 250, 2.0, 1e-4, 200)  # fmt: skip
    assert spec.restarts == RestartSpec(62, 6, 5.0, 0.002, 150)
    data = yaml.safe_load(recipe.source.read_text())
    data["registration"] = {"fit_views": [1, 2]}
    parsed = parse_nurec_recipe(data).registration
    assert parsed.holdout_views == () and parsed.restarts == RestartSpec()
    for change, match in [
        ({"fit_views": []}, "at least one"),
        ({"fit_views": [1], "holdout_views": [1]}, "overlap"),
        ({"fit_views": [1], "canny": [100, 40]}, "canny"),
        ({"fit_views": [-1]}, "capture indices"),
        ({"fit_views": [1], "restarts": {"seeds": 1}}, "unknown keys"),
    ]:
        data["registration"] = change
        with pytest.raises(ValueError, match=match):
            parse_nurec_recipe(data)
    del data["registration"]
    with pytest.raises(ValueError, match="registration"):
        parse_nurec_recipe(data)


def test_prepare_register_refuses_a_visual_only_scene(tmp_path, capsys, monkeypatch):
    from dataclasses import replace

    from doper.nurec import scene as scene_module

    recipe, root, artifact, frames, _ = _register_world(tmp_path)
    real = scene_module.import_scene
    monkeypatch.setattr(
        scene_module,
        "import_scene",
        lambda *a, **k: replace(real(*a, **k), alignment="visual_only"),
    )
    run = ["prepare", recipe, "--stage", "register", "--root", str(root), "--artifact",
           str(artifact), "--keypoint-frames", str(frames)]  # fmt: skip
    assert cli.main(run) == 1
    assert "visual_only" in capsys.readouterr().out
    assert not (root / "registration").exists()


def test_an_accepted_registration_needs_a_reviewer_note(tmp_path):
    fields = {"T_nre_object": np.eye(4), "scene_units_per_metre": 1.0, "status": "accepted",
              "artifact_sha256": "a" * 64}  # fmt: skip
    for note in (None, "  "):
        with pytest.raises(ValueError, match="reviewer's note"):
            Registration(**fields, reviewer_note=note)
    path = tmp_path / "registration.json"
    Registration(**fields, reviewer_note="checked").write(path)
    record = json.loads(path.read_text())
    record["reviewer_note"] = ""
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="reviewer's note"):
        Registration.read(path)
    reviewed = Registration(**{**fields, "status": "reviewed"})
    assert reviewed.reviewer_note is None
