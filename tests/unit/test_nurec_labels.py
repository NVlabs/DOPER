# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry labels of NuRec novel views (``doper.nurec.labels``) and the labels stage."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from nurec_fixtures import NUREC_TEMPLATE

from doper import cli
from doper.bop.sources import sha256_file
from doper.nurec.capture import PinholeCamera
from doper.nurec.labels import ViewLabel, geometry_labels, geometry_labels_batch
from doper.nurec.registration import Registration, TriangleMesh
from doper.nurec.render import NovelView
from doper.nurec.scene import NuRecScene

CAMERA = PinholeCamera(160, 120, 150.0, 150.0, 79.5, 59.5, True)
ARTIFACT_SHA = "a" * 64
OFFSET = np.array([-0.4, 0.05, -0.5])
HALF = 0.02


def _look_at(centre, target) -> np.ndarray:
    """An OpenCV camera pose (camera to world) at ``center`` looking at ``target``."""
    centre, target = np.asarray(centre, float), np.asarray(target, float)
    forward = (target - centre) / np.linalg.norm(target - centre)
    down = np.cross(forward, [0.0, 0.0, 1.0])
    down = np.cross(forward, down / np.linalg.norm(down))
    right = np.cross(down, forward)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([right, down, forward], axis=1)
    pose[:3, 3] = centre
    return pose


def _cube() -> TriangleMesh:
    """A closed cube of side ``2 * HALF`` meters centered on the object origin."""
    corners = np.array([[x, y, z] for x in (-HALF, HALF) for y in (-HALF, HALF)
                        for z in (-HALF, HALF)])  # fmt: skip
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    faces = [tri for a, b, c, d in quads for tri in ((a, b, c), (a, c, d))]
    return TriangleMesh(corners, np.array(faces))


def _keypoints() -> np.ndarray:
    """The cube's corners and face centers (surface points)."""
    corners = _cube().vertices
    centres = np.array([[s * HALF if i == axis else 0.0 for i in range(3)]
                        for axis in range(3) for s in (-1.0, 1.0)])  # fmt: skip
    return np.vstack([corners, centres])


def _world() -> tuple[np.ndarray, np.ndarray]:
    """``T_world_camera`` and ``T_world_object`` in meters."""
    rotation = cv2.Rodrigues(np.array([0.3, -0.5, 0.2]))[0]
    T_world_object = np.eye(4)  # noqa: N806
    T_world_object[:3, :3], T_world_object[:3, 3] = rotation, [0.01, 0.0, 0.02]
    return _look_at([0.12, -0.25, 0.15], [0.01, 0.0, 0.02]), T_world_object


def _in_scene(pose_m: np.ndarray, scale: float) -> np.ndarray:
    """A metric pose in a scene of ``scale`` units per meter (shifted by ``OFFSET``)."""
    pose = pose_m.copy()
    pose[:3, 3] = scale * pose_m[:3, 3] + OFFSET
    return pose


def _setup(scale: float, *, artifact_sha: str = ARTIFACT_SHA):
    """The same physical camera and object in a scene of ``scale`` units per meter."""
    T_world_camera, T_world_object = _world()  # noqa: N806
    T_nre_cam = _in_scene(T_world_camera, scale)  # noqa: N806
    scene = NuRecScene(
        artifact=Path("last.usdz"), artifact_sha256=ARTIFACT_SHA, world_to_nre=np.eye(4),
        cameras={"000000.png": T_nre_cam}, camera=CAMERA, scene_units_per_metre=scale,
        alignment="metric_anchor", reconstruction_id="recon-1",
        lineages={"000000.png": "session-a"},
    )  # fmt: skip
    view = NovelView(index=3, seed_view="000000.png", T_nre_cam=T_nre_cam,
                     min_source_rotation_deg=2.0, min_source_translation_mm=3.0,
                     closest_source=0, closest_rotation_deg=2.0, closest_translation_mm=3.0,
                     rejected_before=0, reconstruction_id="recon-1",
                     lineage="session-a")  # fmt: skip
    registration = Registration(T_nre_object=_in_scene(T_world_object, scale),
                                scene_units_per_metre=scale, status="reviewed",
                                artifact_sha256=artifact_sha)  # fmt: skip
    return view, scene, registration


def _label(scale: float, **kwargs) -> ViewLabel:
    view, scene, registration = _setup(scale)
    return geometry_labels(view, scene, registration, _cube(), _keypoints(), CAMERA, **kwargs)


def test_label_translation_invariant_to_scene_scale():
    unit, scaled = _label(1.0), _label(1.085)
    T_world_camera, T_world_object = _world()  # noqa: N806
    expected = np.linalg.inv(T_world_camera) @ T_world_object
    for label in (unit, scaled):
        np.testing.assert_allclose(label.T_camera_object_m, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(scaled.keypoints_xy, unit.keypoints_xy, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(scaled.ray_visible, unit.ray_visible)
    np.testing.assert_array_equal(scaled.cad_mask, unit.cad_mask)
    # Meters, not scene units: the camera is about 0.3 m from the object.
    assert 0.25 < np.linalg.norm(scaled.T_camera_object_m[:3, 3]) < 0.4


def test_keypoints_reproject_exactly():
    view, scene, registration = _setup(1.085)
    label = geometry_labels(view, scene, registration, _cube(), _keypoints(), CAMERA)
    # Through the label pose (meters) ...
    rotation, translation = label.T_camera_object_m[:3, :3], label.T_camera_object_m[:3, 3]
    camera_points = _keypoints() @ rotation.T + translation
    pixels = camera_points @ CAMERA.matrix.T
    np.testing.assert_allclose(label.keypoints_xy, pixels[:, :2] / pixels[:, 2:], rtol=0,
                               atol=1e-12)  # fmt: skip
    # ... and through the scene: the CAD keypoints placed by the registration, seen by the view.
    scene_points = registration.apply(_keypoints())
    in_camera = scene_points @ view.T_cam_nre[:3, :3].T + view.T_cam_nre[:3, 3]
    projected = in_camera @ CAMERA.matrix.T
    np.testing.assert_allclose(label.keypoints_xy, projected[:, :2] / projected[:, 2:], rtol=0,
                               atol=1e-9)  # fmt: skip
    # Visibility: the face facing away hides its center; at least one corner is hidden.
    assert label.ray_visible.dtype == bool and label.ray_visible.shape == (14,)
    assert 0 < label.ray_visible.sum() < 14
    x, y, w, h = label.cad_bbox
    corners = label.keypoints_xy[:8]
    assert (x, y) == pytest.approx(tuple(corners.min(axis=0)), abs=1e-12)
    assert (x + w, y + h) == pytest.approx(tuple(corners.max(axis=0)), abs=1e-12)
    np.testing.assert_array_equal(label.K, CAMERA.matrix)


def test_half_resolution_mask_shape():
    label = _label(1.085)
    mask = label.cad_mask
    assert mask.shape == (CAMERA.height, CAMERA.width) and mask.dtype == np.uint8
    assert set(np.unique(mask)) == {0, 255}
    # Rasterized at 80x60, then resized by nearest neighbor: every 2x2 block is one value.
    for dy, dx in ((0, 1), (1, 0), (1, 1)):
        np.testing.assert_array_equal(mask[dy::2, dx::2], mask[::2, ::2])
    # The silhouette lies within the unclipped CAD bbox (half-pixel raster rounding aside).
    ys, xs = np.nonzero(mask)
    x, y, w, h = label.cad_bbox
    assert xs.min() >= x - 2 and xs.max() <= x + w + 2
    assert ys.min() >= y - 2 and ys.max() <= y + h + 2
    full = _label(1.085, mask_scale=1.0).cad_mask
    assert abs(int((full > 0).sum()) - int((mask > 0).sum())) < 0.1 * (full > 0).sum()


def test_labels_refuse_another_scene():
    view, scene, registration = _setup(1.085, artifact_sha="b" * 64)
    with pytest.raises(ValueError, match="artifact"):
        geometry_labels(view, scene, registration, _cube(), _keypoints(), CAMERA)
    view, scene, registration = _setup(1.085)
    other = NovelView(**{**view.__dict__, "reconstruction_id": "recon-2"})
    with pytest.raises(ValueError, match="reconstruction"):
        geometry_labels(other, scene, registration, _cube(), _keypoints(), CAMERA)


def test_batch_matches_single_views_and_writes_masks(tmp_path):
    view, scene, registration = _setup(1.085)

    def nudged(i: int) -> np.ndarray:
        step = np.eye(4)
        step[:3, :3] = cv2.Rodrigues(np.array([0.0, 0.02 * i, 0.01 * i]))[0]
        step[:3, 3] = [0.002 * i, 0.0, 0.0]
        return view.T_nre_cam @ step

    views = [NovelView(**{**view.__dict__, "index": i, "T_nre_cam": nudged(i)})
             for i in range(3)]  # fmt: skip
    single = [geometry_labels(v, scene, registration, _cube(), _keypoints(), CAMERA)
              for v in views]  # fmt: skip
    batch = geometry_labels_batch(views, scene, registration, _cube(), _keypoints(), CAMERA,
                                  workers=2, mask_dir=tmp_path / "masks")  # fmt: skip
    assert [label.index for label in batch] == [0, 1, 2]
    for one, many in zip(single, batch, strict=True):
        np.testing.assert_array_equal(one.T_camera_object_m, many.T_camera_object_m)
        np.testing.assert_array_equal(one.keypoints_xy, many.keypoints_xy)
        np.testing.assert_array_equal(one.ray_visible, many.ray_visible)
        assert many.cad_mask is None and many.mask_file == f"{many.index:06d}.png"
        written = cv2.imread(str(tmp_path / "masks" / many.mask_file), cv2.IMREAD_GRAYSCALE)
        np.testing.assert_array_equal(written, one.cad_mask)
        record = json.loads(json.dumps(many.to_record()))
        again = ViewLabel.from_record(record)
        np.testing.assert_array_equal(again.T_camera_object_m, many.T_camera_object_m)
        assert (again.reconstruction_id, again.lineage) == ("recon-1", "session-a")
        assert record["seed_view"] == "000000.png" and len(record["ray_visible"]) == 14
    in_process = geometry_labels_batch(views, scene, registration, _cube(), _keypoints(),
                                       CAMERA, workers=1)  # fmt: skip
    np.testing.assert_array_equal(in_process[2].cad_mask, single[2].cad_mask)


# --- the labels stage ---------------------------------------------------------------------


def _labels_world(tmp_path, monkeypatch, mask_mode="sam2"):
    """A captured, reconstructed, registered and render-planned root (no docker to render)."""
    from test_nurec_registration import _register_world, _truth

    from doper.nurec import render as render_module

    monkeypatch.setattr(render_module.shutil, "which", lambda name: None)
    recipe, root, artifact, _, bundle = _register_world(tmp_path)
    rotation, translation = _truth()
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, translation
    recon = json.loads((root / "reconstruction" / "reconstruction.json").read_text())
    stage = root / "registration"
    stage.mkdir()
    Registration(T_nre_object=transform, scene_units_per_metre=1.085, status="accepted",
                 artifact_sha256=sha256_file(artifact),
                 reconstruction_id=recon["reconstruction_id"],
                 reviewer_note="fixture").write(stage / "registration.json")  # fmt: skip
    render = ["prepare", recipe, "--stage", "render", "--root", str(root), "--artifact",
              str(artifact)]  # fmt: skip
    assert cli.main(render) == 1  # planned; no docker to render
    _small_split(Path(recipe), mask_mode)
    frames = root / "render" / "rgb" / "camera1"
    frames.mkdir(parents=True)
    for view in json.loads((root / "render" / "views.json").read_text()):
        image = np.full((54, 96, 3), 90, np.uint8)
        image[0, 0] = view["index"]  # every frame differs
        cv2.imwrite(str(frames / f"{view['index']:06d}.png"), image)
    assert cli.main(render) == 0  # complete frames: the render step writes its receipt
    return recipe, root, artifact, bundle


def _small_split(recipe: Path, mask_mode: str) -> None:
    """Qualification and split settings for the four fixture views."""
    import yaml

    data = yaml.safe_load(recipe.read_text())
    data["labels"].update(mask_mode=mask_mode, min_visible=1, min_coverage=0.1)
    data["split"].update(train=2, val=1)
    recipe.write_text(yaml.safe_dump(data))


class _BoxSegmenter:
    """Fills the prompt box (clipped to the image): a CPU stand-in for SAM2."""

    def segment(self, image, prompts):
        mask = np.zeros(image.shape[:2], np.uint8)
        x0, y0, x1, y1 = np.clip(prompts.box, 0, None).astype(int)
        mask[y0:y1, x0:x1] = 255
        return mask, 0.75


def _fake_sam2(monkeypatch):
    from doper.cli.prepare import nurec_labels

    monkeypatch.setattr(nurec_labels, "_sam2_version", lambda: "1.1.0")
    monkeypatch.setattr(nurec_labels, "_sam2_segmenter", lambda weights: _BoxSegmenter())


def test_prepare_labels_writes_geometry_labels(tmp_path, capsys, monkeypatch):
    recipe, root, artifact, _ = _labels_world(tmp_path, monkeypatch)
    _fake_sam2(monkeypatch)
    weights = tmp_path / "sam2.1_hiera_small.pt"
    weights.write_bytes(b"weights")
    for name in ("sampling.json", "plan.json"):
        flags = json.loads((root / "render" / name).read_text())["capture_flags"]
        assert flags["robot_reference"] is True
    argv = ["prepare", recipe, "--stage", "labels", "--root", str(root)]
    assert cli.main([*argv, "--sam2-weights", str(weights)]) == 1
    assert "--artifact" in capsys.readouterr().out
    assert cli.main([*argv, "--artifact", str(artifact)]) == 1
    assert "--sam2-weights" in capsys.readouterr().out
    run = [*argv, "--artifact", str(artifact), "--sam2-weights", str(weights)]
    assert cli.main(run) == 0
    out = root / "labels" / "geometry"
    rows = json.loads((out / "geometry_labels.json").read_text())
    views = json.loads((root / "render" / "views.json").read_text())
    assert [r["index"] for r in rows] == [v["index"] for v in views]
    for row in rows:
        mask = cv2.imread(str(out / "cad_masks" / row["mask_file"]), cv2.IMREAD_GRAYSCALE)
        assert mask.shape == (54, 96)
        assert row["reconstruction_id"] == views[0]["reconstruction_id"]
        assert row["lineage"] == "fixture"
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["capture_flags"]["robot_reference"] is True
    assert receipt["count"] == len(views)
    assert receipt["artifact_sha256"] == sha256_file(artifact)
    assert receipt["registration_status"] == "accepted"
    # SAM2 masks, qualification, the adaptation dataset and its QC follow.
    masks = root / "labels" / "masks"
    scores = json.loads((masks / "sam_scores.json").read_text())
    assert [s["index"] for s in scores] == [v["index"] for v in views]
    masks_receipt = json.loads((masks / "receipt.json").read_text())
    assert masks_receipt["sam2_config"] == "configs/sam2.1/sam2.1_hiera_s.yaml"
    assert masks_receipt["sam2_version"] == "1.1.0" and masks_receipt["reference_path"] is True
    qualified = json.loads((root / "labels" / "qualified" / "qualified_labels.json").read_text())
    assert len(qualified) == len(views) and sum(q["accepted"] for q in qualified) >= 3
    dataset = root / "adaptation" / "dataset"
    train = json.loads((dataset / "train" / "annotations.json").read_text())
    assert len(train["images"]) == 2 and train["categories"][0]["name"] == "object"
    assert all("mask_file" in ann for ann in train["annotations"])
    qc = json.loads((root / "adaptation" / "dataset_qc.json").read_text())
    assert qc["status"] == "passed" and qc["splits"] == {"train": 2, "val": 1}
    package = json.loads((root / "adaptation" / "receipt.json").read_text())
    assert package["mask_mode"] == "sam2" and package["reference_path"] is True
    assert package["lineage_split"] is False and package["lineage_caveat"]
    assert package["capture_flags"]["robot_reference"] is True
    # A rerun keeps the labels; another registration needs --force.
    assert cli.main(run) == 0
    out_text = capsys.readouterr().out
    assert "already labeled" in out_text and "already packaged" in out_text
    registration = root / "registration" / "registration.json"
    record = json.loads(registration.read_text())
    record["T_nre_object"][0][3] += 0.001
    registration.write_text(json.dumps(record))
    assert cli.main(run) == 1
    assert "--force" in capsys.readouterr().out
    assert cli.main([*run, "--force"]) == 0
    moved = json.loads((out / "geometry_labels.json").read_text())
    assert moved[0]["T_camera_object_m"] != rows[0]["T_camera_object_m"]
    # A registration of another artifact never labels this scene.
    record["artifact_sha256"] = "f" * 64
    registration.write_text(json.dumps(record))
    assert cli.main([*run, "--force"]) == 1
    assert "artifact" in capsys.readouterr().out


def test_prepare_labels_flags_and_dry_run(tmp_path, capsys):
    base = ["prepare", str(NUREC_TEMPLATE), "--stage", "labels", "--root", str(tmp_path / "root")]
    weights = ["--sam2-weights", str(tmp_path / "sam2.pt")]
    assert cli.main([*base, "--artifact", str(tmp_path / "a.usdz"), "--dry-run", *weights]) == 0
    out = capsys.readouterr().out
    assert "geometry_labels.json" in out and "cad_masks" in out
    assert "sam_scores.json" in out and "qualified_labels.json" in out
    assert "adaptation/dataset" in out and "dataset_qc.json" in out
    assert not (tmp_path / "root").exists()
    for flag in (["--keypoint-frames", "f.json"], ["--render-cache", "c"],
                 ["--reviewer-note", "x"], ["--bundle", "b"]):  # fmt: skip
        assert cli.main([*base, "--artifact", "a.usdz", "--dry-run", *weights, *flag]) == 1
        assert flag[0] in capsys.readouterr().out
    render = ["prepare", str(NUREC_TEMPLATE), "--stage", "render", "--root", str(tmp_path / "root"),
              "--artifact", "a.usdz", "--dry-run", *weights]  # fmt: skip
    assert cli.main(render) == 1
    assert "--sam2-weights" in capsys.readouterr().out


def test_prepare_labels_cad_only_has_no_mask_supervision(tmp_path, capsys, monkeypatch):
    recipe, root, artifact, _ = _labels_world(tmp_path, monkeypatch, mask_mode="cad_only")
    run = ["prepare", recipe, "--stage", "labels", "--root", str(root), "--artifact",
           str(artifact)]  # fmt: skip
    assert cli.main([*run, "--sam2-weights", str(tmp_path / "w.pt")]) == 1
    assert "cad_only" in capsys.readouterr().out
    assert cli.main(run) == 0
    assert not (root / "labels" / "masks").exists()
    dataset = root / "adaptation" / "dataset"
    train = json.loads((dataset / "train" / "annotations.json").read_text())
    assert all("mask_file" not in ann for ann in train["annotations"])
    assert not (dataset / "train" / "masks").exists()
    for receipt in (root / "labels" / "qualified" / "receipt.json",
                    root / "adaptation" / "receipt.json"):  # fmt: skip
        record = json.loads(receipt.read_text())
        assert (record["mask_mode"] if "mask_mode" in record else record["labels"]["mask_mode"],
                record["reference_path"]) == ("cad_only", False)  # fmt: skip


def test_prepare_labels_refuses_a_swapped_mesh(tmp_path, capsys, monkeypatch):
    recipe, root, artifact, bundle = _labels_world(tmp_path, monkeypatch, mask_mode="cad_only")
    mesh = bundle / "object.obj"
    mesh.write_text(mesh.read_text() + "# another revision\n")
    run = ["prepare", recipe, "--stage", "labels", "--root", str(root), "--artifact",
           str(artifact)]  # fmt: skip
    assert cli.main(run) == 1
    assert "is not the recipe's mesh" in capsys.readouterr().out
    assert not (root / "labels").exists()


def test_prepare_labels_refuses_a_visual_only_scene(tmp_path, capsys, monkeypatch):
    from dataclasses import replace

    from doper.nurec import scene as scene_module

    recipe, root, artifact, _ = _labels_world(tmp_path, monkeypatch, mask_mode="cad_only")
    real = scene_module.import_scene
    monkeypatch.setattr(
        scene_module,
        "import_scene",
        lambda *a, **k: replace(real(*a, **k), alignment="visual_only"),
    )
    run = ["prepare", recipe, "--stage", "labels", "--root", str(root), "--artifact",
           str(artifact)]  # fmt: skip
    assert cli.main(run) == 1
    assert "visual_only" in capsys.readouterr().out
    assert not (root / "labels").exists()


def test_a_failed_qc_leaves_no_passing_evidence(tmp_path, capsys, monkeypatch):
    from doper.nurec import package as package_module

    recipe, root, artifact, _ = _labels_world(tmp_path, monkeypatch, mask_mode="cad_only")
    run = ["prepare", recipe, "--stage", "labels", "--root", str(root), "--artifact",
           str(artifact)]  # fmt: skip
    assert cli.main(run) == 0
    stage = root / "adaptation"
    assert json.loads((stage / "dataset_qc.json").read_text())["status"] == "passed"
    receipt = json.loads((stage / "receipt.json").read_text())
    assert receipt["qc_sha256"] == sha256_file(stage / "dataset_qc.json")
    assert receipt["package_sha256"] == sha256_file(stage / "dataset" / "package.json")

    def failing(*args, **kwargs):
        raise package_module.DatasetQcError("reprojection", "fixture")

    monkeypatch.setattr(package_module, "qc_dataset", failing)
    assert cli.main([*run, "--force"]) == 1
    assert "reprojection" in capsys.readouterr().out
    assert not (stage / "dataset_qc.json").exists() and not (stage / "receipt.json").exists()
    train = ["train", recipe, "--stage", "adapt", "--root", str(root), "--runs",
             str(tmp_path / "runs")]  # fmt: skip
    assert cli.main(train) == 1
    assert "did not finish (or its QC failed)" in capsys.readouterr().out


class _RerenderBackend:
    """Writes every planned frame with another pixel value: a re-render that differs."""

    def command(self, scene, trajectory, out):
        return ["fake-render", str(trajectory)]

    def render(self, scene, trajectory, out):
        from doper.nurec.render import RenderResult

        record = json.loads(Path(trajectory).read_text())
        count = len(record["rig_trajectories"][0]["T_rig_worlds"])
        frames = Path(out) / "rgb" / "camera1"
        frames.mkdir(parents=True)
        for index in range(count):
            image = np.full((54, 96, 3), 120, np.uint8)
            image[0, 0] = index
            cv2.imwrite(str(frames / f"{index:06d}.png"), image)
        return RenderResult.collect(frames, count=count, width=96, height=54)


def test_a_forced_rerender_invalidates_the_masks_and_qualification(tmp_path, capsys, monkeypatch):
    from doper.cli.prepare import nurec_render

    recipe, root, artifact, _ = _labels_world(tmp_path, monkeypatch)
    _fake_sam2(monkeypatch)
    weights = tmp_path / "sam2.1_hiera_small.pt"
    weights.write_bytes(b"weights")
    receipt = json.loads((root / "render" / "receipt.json").read_text())
    assert receipt["plan_sha256"] == sha256_file(root / "render" / "plan.json")
    frame = root / "render" / "rgb" / "camera1" / "000000.png"
    assert receipt["frames_sha256"]["000000.png"] == sha256_file(frame)
    labels = ["prepare", recipe, "--stage", "labels", "--root", str(root), "--artifact",
              str(artifact), "--sam2-weights", str(weights)]  # fmt: skip
    assert cli.main(labels) == 0
    masks = json.loads((root / "labels" / "masks" / "receipt.json").read_text())
    assert masks["render_receipt_sha256"] == sha256_file(root / "render" / "receipt.json")
    # Re-rendering with --force (same plan, other frames) rewrites the render receipt ...
    monkeypatch.setattr(nurec_render, "_render_backend", lambda recipe, cache: _RerenderBackend())
    render = ["prepare", recipe, "--stage", "render", "--root", str(root), "--artifact",
              str(artifact)]  # fmt: skip
    assert cli.main([*render, "--force"]) == 0
    assert sha256_file(frame) != receipt["frames_sha256"]["000000.png"]
    capsys.readouterr()
    # ... so the SAM2 masks and the qualification of the old frames are no longer current.
    assert cli.main(labels) == 1
    out = capsys.readouterr().out
    assert "already labeled" in out and "labels/masks was made from other inputs" in out
    assert cli.main([*labels, "--force"]) == 0
    qualified = json.loads((root / "labels" / "qualified" / "receipt.json").read_text())
    assert qualified["render_receipt_sha256"] == sha256_file(root / "render" / "receipt.json")


def test_labels_need_the_render_receipt(tmp_path, capsys, monkeypatch):
    recipe, root, artifact, _ = _labels_world(tmp_path, monkeypatch, mask_mode="cad_only")
    (root / "render" / "receipt.json").unlink()
    run = ["prepare", recipe, "--stage", "labels", "--root", str(root), "--artifact",
           str(artifact)]  # fmt: skip
    assert cli.main(run) == 1
    assert "the render step did not finish" in capsys.readouterr().out


def test_a_label_worker_without_its_batch_context_raises():
    from doper.nurec import labels as labels_module

    view, _, _ = _setup(1.085)
    with pytest.raises(RuntimeError, match="batch context"):
        labels_module._label_one(view)


def _stamp(path):
    return path.read_bytes(), path.stat().st_mtime_ns


def test_force_from_masks_rebuilds_masks_and_keeps_geometry(tmp_path, capsys, monkeypatch):
    from doper.cli.prepare import nurec_render

    recipe, root, artifact, _ = _labels_world(tmp_path, monkeypatch)
    _fake_sam2(monkeypatch)
    weights = tmp_path / "sam2.1_hiera_small.pt"
    weights.write_bytes(b"weights")
    labels = ["prepare", recipe, "--stage", "labels", "--root", str(root), "--artifact",
              str(artifact), "--sam2-weights", str(weights)]  # fmt: skip
    assert cli.main(labels) == 0
    geometry = root / "labels" / "geometry" / "receipt.json"
    masks = root / "labels" / "masks" / "receipt.json"
    before = _stamp(geometry), _stamp(masks)
    monkeypatch.setattr(nurec_render, "_render_backend", lambda recipe, cache: _RerenderBackend())
    render = ["prepare", recipe, "--stage", "render", "--root", str(root), "--artifact",
              str(artifact)]  # fmt: skip
    assert cli.main([*render, "--force-from", "frames"]) == 0
    capsys.readouterr()
    assert cli.main(labels) == 1
    assert cli.main([*labels, "--force-from", "masks"]) == 0
    assert _stamp(geometry) == before[0]
    assert _stamp(masks) != before[1]
    qualified = json.loads((root / "labels" / "qualified" / "receipt.json").read_text())
    assert qualified["render_receipt_sha256"] == sha256_file(root / "render" / "receipt.json")


def test_force_from_rejects_a_step_of_another_stage(tmp_path, capsys, monkeypatch):
    recipe, root, artifact, _ = _labels_world(tmp_path, monkeypatch)
    argv = ["prepare", recipe, "--stage", "labels", "--root", str(root), "--artifact",
            str(artifact), "--force-from", "split"]  # fmt: skip
    before = _tree_of(root)
    assert cli.main(argv) == 1
    out = capsys.readouterr().out
    assert "--force-from split is not a step of --stage labels" in out
    assert "geometry, masks, qualify, package" in out
    assert _tree_of(root) == before
    assert cli.main(["prepare", recipe, "--stage", "register", "--root", str(root),
                     "--force-from", "plan"]) == 1  # fmt: skip
    assert "--force-from only for --stage render, labels, synthetic" in capsys.readouterr().out


def test_force_from_masks_refused_for_cad_only(tmp_path, capsys, monkeypatch):
    recipe, root, artifact, _ = _labels_world(tmp_path, monkeypatch, mask_mode="cad_only")
    argv = ["prepare", recipe, "--stage", "labels", "--root", str(root), "--artifact",
            str(artifact), "--force-from", "masks"]  # fmt: skip
    assert cli.main(argv) == 1
    out = capsys.readouterr().out
    assert "not a step of --stage labels" in out and "geometry, qualify, package" in out


def test_force_from_is_exclusive_with_force(tmp_path, monkeypatch):
    recipe, root, artifact, _ = _labels_world(tmp_path, monkeypatch, mask_mode="cad_only")
    argv = ["prepare", recipe, "--stage", "labels", "--root", str(root), "--force", "--force-from",
            "qualify"]  # fmt: skip
    with pytest.raises(SystemExit):
        cli.main(argv)


def _tree_of(path):
    return sorted(str(p.relative_to(path)) for p in path.rglob("*"))


def test_labels_stage_verifies_the_bundle_once(tmp_path, capsys, monkeypatch):
    from doper.nurec import capture

    recipe, root, artifact, _ = _labels_world(tmp_path, monkeypatch, mask_mode="cad_only")
    loads = []
    real = capture.load_image_set
    monkeypatch.setattr(
        capture,
        "load_image_set",
        lambda *args, **kwargs: loads.append(args) or real(*args, **kwargs),
    )
    run = ["prepare", recipe, "--stage", "labels", "--root", str(root), "--artifact",
           str(artifact)]  # fmt: skip
    assert cli.main(run) == 0
    assert len(loads) == 1
    assert cli.main([*run, "--dry-run"]) == 0
    assert len(loads) == 1


def test_capture_context_reads_nothing_until_a_member_is_used(tmp_path):
    from doper.cli.prepare.nurec_capture import CaptureContext
    from doper.recipes import load_any_recipe

    ctx = CaptureContext(tmp_path / "missing", load_any_recipe(NUREC_TEMPLATE))
    assert not (tmp_path / "missing").exists()
    with pytest.raises(FileNotFoundError, match="--stage capture --bundle DIR"):
        ctx.receipt  # noqa: B018
