# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mode (c), experimental: DOPER PnP poses seed COLMAP.

Synthetic contract: OpenCV cameras around a CAD keypoint cloud (symmetric under a 180 degree
rotation about the object z axis), projected with the capture K plus 0.5 px noise; a stub
``infer_image`` returns those keypoints, and the release ``solve_pnp`` solves each pose.
"""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from colmap_fake import install_fake_colmap, recorded_calls
from nurec_fixtures import NUREC_TEMPLATE, nurec_recipe_data, write_bundle

from doper import cli
from doper.infer import Detection
from doper.nurec.capture import load_frame_map, load_image_set, normalize_images
from doper.recipes import load_any_recipe
from doper.recipes.nurec_schema import parse_nurec_recipe
from doper.reconstruction import colmap_io
from doper.reconstruction.colmap_runner import ColmapCli
from doper.reconstruction.geometry import camera_centre, rotation_geodesic_deg
from doper.reconstruction.metadata import ReconstructionMetadata
from doper.reconstruction.seeding import (
    STRATEGIES,
    DoperPnpSeed,
    DoperSeedGates,
    KeypointObservations,
    QualityGate,
    SeedStrategy,
    cad_scale_check,
    consistency_gate,
    reconstruct,
    scene_evidence,
)

CAMERA = {"width": 320, "height": 240, "fx": 300.0, "fy": 300.0, "cx": 159.5, "cy": 119.5,
          "rectified": True}  # fmt: skip
NORMALIZED_EDGE = 160
VIEWS = 8
NAMES = tuple(f"{i:06d}.png" for i in range(VIEWS))
FLIPPED = 3
LEFT_OUT = 5
NOISE_PX = 0.5
DRIFT = 1.0847
RELAXED = QualityGate(min_registered_images=1)
RZ180 = np.diag([-1.0, -1.0, 1.0, 1.0])


def _cad_keypoints() -> np.ndarray:
    """16 CAD keypoints (meters), symmetric under a 180 degree rotation about z."""
    half = np.random.default_rng(3).uniform(-0.08, 0.08, size=(8, 3))
    return np.concatenate([half, half @ RZ180[:3, :3].T])


def _look_at(centre: np.ndarray, target: np.ndarray) -> np.ndarray:
    """OpenCV ``cam_from_world`` (x right, y down, z forward) of a camera at ``center``."""
    z = (target - centre) / np.linalg.norm(target - centre)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    rotation = np.stack([x, np.cross(z, x), z])
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = -rotation @ centre
    return pose


def _true_poses() -> list[np.ndarray]:
    poses = []
    for i in range(VIEWS):
        azimuth, elevation = 2 * np.pi * i / VIEWS, np.radians(25 + 10 * (i % 3))
        ground = np.cos(elevation)
        centre = 0.5 * np.array([np.cos(azimuth) * ground, np.sin(azimuth) * ground,
                                 np.sin(elevation)])  # fmt: skip
        poses.append(_look_at(centre, np.array([0.003, -0.002, 0.004])))
    return poses


def _project(pose: np.ndarray, points: np.ndarray, k: np.ndarray) -> np.ndarray:
    camera = points @ pose[:3, :3].T + pose[:3, 3]
    pixels = camera @ k.T
    return pixels[:, :2] / pixels[:, 2:]


class StubInfer:
    """Stands in for ``doper.infer.infer_image``: one detection per image, by image bytes."""

    def __init__(self, image_set, detections):
        self.by_image = {}
        for name, found in detections.items():
            pixels = cv2.imread(str(image_set.image_path(name)), cv2.IMREAD_COLOR)
            self.by_image[hashlib.sha256(pixels.tobytes()).hexdigest()] = found
        self.calls = []

    def __call__(self, model, image_bgr, cfg, device, **kwargs):
        self.calls.append(kwargs)
        assert kwargs["max_det"] == 1 and "camera_matrix" not in kwargs
        return list(self.by_image[hashlib.sha256(image_bgr.tobytes()).hexdigest()])


def _detection(keypoints: np.ndarray, score: float = 0.9, scores=None) -> Detection:
    scores = np.full(len(keypoints), 0.9) if scores is None else np.asarray(scores)
    box = np.array([*keypoints.min(axis=0), *keypoints.max(axis=0)])
    return Detection(box, score, keypoints.astype(np.float32), scores.astype(np.float32))


def _capture(tmp_path: Path, names=NAMES, edge=NORMALIZED_EDGE):
    bundle = tmp_path / "bundle"
    write_bundle(bundle, names=names, camera=CAMERA, robot=False)
    image_set = load_image_set(bundle)
    frames = normalize_images(image_set, tmp_path / "frames", max_long_edge=edge)
    return image_set, frames


def _observed(names=NAMES, *, flipped=None, noise=NOISE_PX, seed=11):
    """Name to (true pose, noisy keypoints in capture pixels)."""
    rng = np.random.default_rng(seed)
    k = np.array([[CAMERA["fx"], 0, CAMERA["cx"]], [0, CAMERA["fy"], CAMERA["cy"]], [0, 0, 1]])
    points, observed = _cad_keypoints(), {}
    for i, (name, pose) in enumerate(zip(names, _true_poses() * 4, strict=False)):
        seen = pose @ RZ180 if i == flipped else pose
        observed[name] = (pose, _project(seen, points, k) + rng.normal(0, noise, (len(points), 2)))
    return observed


def _seed(tmp_path, detections, gates=None):
    image_set, frames = _capture(tmp_path)
    stub = StubInfer(image_set, detections)
    strategy = DoperPnpSeed(None, _cad_keypoints(), gates or DoperSeedGates(), infer=stub)
    return strategy, strategy.seed(image_set, frames, None), image_set, frames, stub


def test_doper_pnp_strategy_and_r19_defaults():
    r19 = DoperSeedGates(score_thr=0.3, min_keypoints=8, max_reproj_px=3.0,
                         max_consistency_px=5.0, register_leftovers=True)  # fmt: skip
    assert DoperSeedGates() == r19
    strategy = DoperPnpSeed(None, _cad_keypoints())
    assert isinstance(strategy, SeedStrategy) and STRATEGIES["doper_pnp"] is DoperPnpSeed
    assert strategy.name == "doper_pnp" and strategy.world_frame == "object_cad"
    assert strategy.reference_path is False
    assert "experimental" in (DoperPnpSeed.__doc__ or "")
    with pytest.raises(ValueError, match="min_keypoints"):
        DoperSeedGates(min_keypoints=3)


def test_recipe_doper_pnp_section():
    recipe = load_any_recipe(NUREC_TEMPLATE)
    assert recipe.seed_mode == "robot_handeye" and recipe.doper_pnp == DoperSeedGates()
    data = yaml.safe_load(recipe.source.read_text())
    del data["doper_pnp"]
    assert parse_nurec_recipe(data).doper_pnp == DoperSeedGates()
    data["doper_pnp"] = {"score_thr": 0.5}
    assert parse_nurec_recipe(data).doper_pnp == DoperSeedGates(score_thr=0.5)
    data["doper_pnp"] = {"score": 0.5}
    with pytest.raises(ValueError, match="doper_pnp has unknown keys"):
        parse_nurec_recipe(data)
    data["doper_pnp"] = {"register_leftovers": "yes"}
    with pytest.raises(ValueError, match="register_leftovers"):
        parse_nurec_recipe(data)


def test_seed_identity_recovers_cameras(tmp_path):
    observed = _observed()
    _, seed, image_set, frames, stub = _seed(
        tmp_path, {name: [_detection(uv)] for name, (_, uv) in observed.items()}
    )
    assert seed is not None and seed.rejected == {}
    assert list(seed.cam_from_world) == [frame.name for frame in frames.frames]
    # COLMAP runs on the normalized frames: the seed camera is the session K at their size,
    # while DOPER ran on the capture images with the session K (poses do not depend on size).
    assert (seed.camera.width, seed.camera.height) == (160, 120)
    assert seed.camera == image_set.camera.resized(160, 120) == frames.camera
    assert len(stub.calls) == VIEWS and all(c["score_thr"] == 0.3 for c in stub.calls)
    for frame in frames.frames:
        truth, pose = observed[frame.source][0], seed.cam_from_world[frame.name]
        # cam_from_world = P_i (world = CAD frame, meters): no inverse.
        assert rotation_geodesic_deg(pose[:3, :3], truth[:3, :3]) < 1.0
        np.testing.assert_allclose(pose[:3, 3], truth[:3, 3], atol=1e-2)
        np.testing.assert_allclose(camera_centre(pose), -pose[:3, :3].T @ pose[:3, 3])
        np.testing.assert_allclose(camera_centre(pose), camera_centre(truth), atol=1e-2)
        # Keypoint observations are kept in normalized-frame pixels for the scale cross-check.
        uv = seed.observations[frame.name].uv
        np.testing.assert_allclose(uv, observed[frame.source][1] * 0.5, atol=1e-4)


def test_consistency_gate_drops_keypoint_inconsistent_pose():
    # One pose rotated 180 degrees about the object's symmetry axis while its keypoints are the
    # true ones: its rays miss the keypoints the other views triangulate.
    observed = _observed()
    poses = {name: pose for name, (pose, _) in observed.items()}
    poses[NAMES[FLIPPED]] = poses[NAMES[FLIPPED]] @ RZ180
    keypoints = {name: KeypointObservations(uv, np.ones(16, dtype=bool))
                 for name, (_, uv) in observed.items()}  # fmt: skip
    k = np.array([[CAMERA["fx"], 0, CAMERA["cx"]], [0, CAMERA["fy"], CAMERA["cy"]], [0, 0, 1]])
    rejected = consistency_gate(poses, keypoints, k, 5.0)
    assert list(rejected) == [NAMES[FLIPPED]]
    assert rejected[NAMES[FLIPPED]].startswith("cross-view consistency: median keypoint residual")
    assert consistency_gate({n: observed[n][0] for n in NAMES}, keypoints, k, 5.0) == {}


def test_consistency_gate_cannot_see_a_symmetric_relabelling(tmp_path):
    """Known limitation: keypoints relabeled by the object's symmetry fit one rigid pose.

    The flipped pose then explains its keypoints exactly in the CAD frame, so no keypoint gate
    can reject it; only scene evidence (COLMAP's feature geometry) disagrees.
    """
    observed = _observed(flipped=FLIPPED)
    _, seed, _, frames, _ = _seed(
        tmp_path, {name: [_detection(uv)] for name, (_, uv) in observed.items()}
    )
    assert seed.rejected == {} and frames.by_source(NAMES[FLIPPED]).name in seed.cam_from_world


def test_gates_reject_low_score_and_few_keypoints(tmp_path):
    observed = _observed()
    detections = {name: [_detection(uv)] for name, (_, uv) in observed.items()}
    few = np.where(np.arange(16) < 6, 0.9, 0.1)
    detections[NAMES[0]] = [_detection(observed[NAMES[0]][1], score=0.2)]
    detections[NAMES[1]] = [_detection(observed[NAMES[1]][1], scores=few)]
    detections[NAMES[2]] = []
    noisy = _observed(noise=12.0, seed=5)[NAMES[5]][1]
    detections[NAMES[5]] = [_detection(noisy)]
    _, seed, _, frames, _ = _seed(tmp_path, detections)
    assert seed is not None
    reasons = {frames.frames[i].source: seed.rejected.get(frames.frames[i].name)
               for i in range(VIEWS)}  # fmt: skip
    assert "detection score 0.2" in reasons[NAMES[0]]
    assert "6 keypoints with score >= 0.3" in reasons[NAMES[1]]
    assert reasons[NAMES[2]] == "no detection"
    assert "PnP" in reasons[NAMES[5]]
    assert [n for n, r in reasons.items() if r is None] == [NAMES[3], NAMES[4], NAMES[6], NAMES[7]]


def test_seed_refuses_fewer_than_three_seeded_images(tmp_path):
    observed = _observed()
    detections = {name: [] for name in NAMES}
    detections[NAMES[0]] = [_detection(observed[NAMES[0]][1])]
    image_set, frames = _capture(tmp_path)
    strategy = DoperPnpSeed(None, _cad_keypoints(), infer=StubInfer(image_set, detections))
    with pytest.raises(ValueError, match="seeded 1 of 8 images"):
        strategy.seed(image_set, frames, None)
    with pytest.raises(ValueError, match="capture images"):
        strategy.seed(None, frames, None)


def _scaled_model(seed, scale: float) -> colmap_io.ColmapModel:
    """The seed cameras with the world scaled by ``scale`` (a refinement's gauge)."""
    camera = colmap_io.ColmapCamera(1, "OPENCV", seed.camera.width, seed.camera.height,
                                    (seed.camera.fx, seed.camera.fy, seed.camera.cx,
                                     seed.camera.cy, 0, 0, 0, 0))  # fmt: skip
    images = {
        i + 1: colmap_io.ColmapImage(i + 1, colmap_io.quaternion_wxyz(pose[:3, :3]),
                                     tuple(float(v) * scale for v in pose[:3, 3]), 1, name, ())
        for i, (name, pose) in enumerate(seed.cam_from_world.items())
    }  # fmt: skip
    return colmap_io.ColmapModel({1: camera}, images, {})


def test_scale_cross_check_fails_on_disagreement(tmp_path):
    observed = _observed()
    _, seed, _, _, _ = _seed(
        tmp_path, {name: [_detection(uv)] for name, (_, uv) in observed.items()}
    )
    model = _scaled_model(seed, DRIFT)
    check = cad_scale_check(_cad_keypoints(), seed, model, DRIFT)
    # CAD meters -> scene units, the anchor's direction (not 1 / DRIFT).
    assert check.scene_units_per_metre == pytest.approx(DRIFT, rel=2e-3)
    assert check.anchor_scene_units_per_metre == DRIFT and check.n == 16
    assert check.relative_difference < 2e-3 and check.tolerance == 0.01
    assert check.rms_mm == pytest.approx(check.rms_scene / check.scene_units_per_metre * 1000)
    with pytest.raises(ValueError, match=r"CAD-keypoint scale 1\.08.* anchor 1\.12"):
        cad_scale_check(_cad_keypoints(), seed, model, 1.12)


def _write_scene(fake: Path, frames, observed) -> None:
    """``scene.json`` of the fake COLMAP: static scene points and every frame's true camera."""
    rng = np.random.default_rng(17)
    points = np.column_stack([rng.uniform(-0.25, 0.25, (30, 2)), rng.uniform(-0.15, 0.05, 30)])
    truth = {f.name: observed[f.source][0].tolist() for f in frames.frames}
    (fake.parent / "scene.json").write_text(json.dumps({"points": points.tolist(),
                                                        "cam_from_world": truth}))  # fmt: skip


def _reconstruct(tmp_path, fake, gates=None, drift=True, flipped=None):
    observed = _observed(flipped=flipped)
    image_set, frames = _capture(tmp_path)
    _write_scene(fake, frames, observed)
    detections = {name: [_detection(uv)] for name, (_, uv) in observed.items()}
    detections[NAMES[LEFT_OUT]] = [_detection(observed[NAMES[LEFT_OUT]][1], score=0.1)]
    strategy = DoperPnpSeed(None, _cad_keypoints(), gates or DoperSeedGates(),
                            infer=StubInfer(image_set, detections))  # fmt: skip
    if drift:
        (fake.parent / "scale.txt").write_text(repr(DRIFT))
    out = tmp_path / "reconstruction"
    runner = ColmapCli(str(fake), load_any_recipe(NUREC_TEMPLATE).colmap, out / "logs")
    metadata = reconstruct(image_set, frames, strategy, runner, out,
                           frames_dir=tmp_path / "frames", capture_flags={},
                           inputs={"bundle_sha256": "cd" * 32}, gate=RELAXED)  # fmt: skip
    return metadata, frames, out


def test_reconstruct_doper_pnp_registers_leftovers(tmp_path):
    fake = install_fake_colmap(tmp_path / "bin")
    metadata, frames, out = _reconstruct(tmp_path, fake)
    commands = [call[1] for call in metadata.colmap.argv]
    # Triangulate the seed, check its scene evidence (text model), register the leftovers,
    # then the bundle adjustment passes.
    assert commands == [
        "help", "feature_extractor", "exhaustive_matcher",
        "point_triangulator", "model_converter", "image_registrator",
        *["bundle_adjuster", "point_triangulator"] * 2, "bundle_adjuster", "model_converter",
    ]  # fmt: skip
    assert [call[1:] for call in metadata.colmap.argv] == recorded_calls(fake)
    register = metadata.colmap.argv[5]
    sparse0, registered = str(out / "sparse" / "0"), str(out / "refine" / "registered")
    first = str(out / "refine" / "triangulated-1")
    assert register[register.index("--input_path") + 1] == first
    assert register[register.index("--output_path") + 1] == registered
    assert "--Mapper.ba_refine_focal_length" in register
    adjust = metadata.colmap.argv[6]
    assert adjust[adjust.index("--input_path") + 1] == registered
    assert adjust[adjust.index("--output_path") + 1] == sparse0
    # Mode (c) metadata: CAD frame, metric through the seed, not the recommended path.
    assert metadata.strategy == "doper_pnp" and metadata.world_frame == "object_cad"
    assert metadata.reference_path is False and metadata.length_unit == "colmap"
    assert metadata.scene_units_per_metre == pytest.approx(DRIFT, rel=1e-9)
    check = metadata.scale_check
    assert check is not None and check.relative_difference < 0.01
    assert check.scene_units_per_metre == pytest.approx(DRIFT, rel=5e-3)
    left_out = frames.by_source(NAMES[LEFT_OUT]).name
    for record in metadata.images:
        assert record.registered and record.excluded_reason is None
        if record.name == left_out:
            assert not record.seeded and record.registered_by == "image_registrator"
            assert record.seed_rejected_reason == "detection score 0.1 < 0.3"
        else:
            assert record.seeded and record.registered_by == "seed"
            assert record.seed_rejected_reason is None
    seed_model = colmap_io.read_text_model(out / "seed")
    assert left_out not in {image.name for image in seed_model.images.values()}
    assert ReconstructionMetadata.read(out / "reconstruction.json") == metadata


def test_scene_evidence_drops_flipped_view(tmp_path):
    # A symmetry flip WITH relabeled keypoints passes every keypoint gate (see
    # test_consistency_gate_cannot_see_a_symmetric_relabelling); the scene's features, seen
    # through the true camera, contradict the flipped seed pose after the first triangulation.
    fake = install_fake_colmap(tmp_path / "bin")
    metadata, frames, out = _reconstruct(tmp_path, fake, flipped=FLIPPED)
    commands = [call[1] for call in metadata.colmap.argv]
    assert commands[3:7] == ["point_triangulator", "model_converter", "point_triangulator",
                             "image_registrator"]  # fmt: skip
    rechecked = metadata.colmap.argv[5]
    assert rechecked[rechecked.index("--output_path") + 1] == str(
        out / "refine" / "triangulated-1-checked"
    )
    flipped = metadata.image_by_source(NAMES[FLIPPED])
    assert not flipped.seeded and flipped.registered_by == "image_registrator"
    assert (flipped.seed_rejected_reason or "").startswith("scene_evidence: median reprojection")
    left_out = metadata.image_by_source(NAMES[LEFT_OUT])
    assert left_out.seed_rejected_reason == "detection score 0.1 < 0.3"
    assert left_out.registered_by == "image_registrator"
    others = [r for r in metadata.images if r.source_name not in (NAMES[FLIPPED], NAMES[LEFT_OUT])]
    assert all(r.seeded and r.registered_by == "seed" for r in others)
    # The rewritten seed and its bindings no longer hold the flipped image.
    seed_names = {i.name for i in colmap_io.read_text_model(out / "seed").images.values()}
    assert flipped.name not in seed_names and len(seed_names) == VIEWS - 2
    bindings = json.loads((out / "seed-bindings.json").read_text())
    assert flipped.name not in {b["normalized_image"] for b in bindings}
    assert metadata.scale_check is not None and metadata.scale_check.relative_difference < 0.01


def _evidence_model(counts: list[int]) -> tuple[colmap_io.ColmapModel, object]:
    """Images seen as their seed poses: ``counts[i]`` exactly reprojected observations."""
    from doper.reconstruction.seeding import SeedResult

    k = np.array([[150.0, 0, 79.75], [0, 150.0, 59.75], [0, 0, 1]])
    camera = colmap_io.ColmapCamera(1, "OPENCV", 160, 120, (150.0, 150.0, 79.75, 59.75, 0, 0, 0, 0))
    points = np.random.default_rng(2).uniform(-0.1, 0.1, (max(counts), 3))
    images, poses = {}, {}
    for i, (count, pose) in enumerate(zip(counts, _true_poses(), strict=False)):
        uv = _project(pose, points[:count], k)
        name = f"{i:06d}.png"
        poses[name] = pose
        images[i + 1] = colmap_io.ColmapImage(
            i + 1, colmap_io.quaternion_wxyz(pose[:3, :3]), tuple(pose[:3, 3]), 1, name,
            tuple((float(u), float(v), j + 1) for j, (u, v) in enumerate(uv)),
        )  # fmt: skip
    model_points = {j + 1: colmap_io.ColmapPoint3D(j + 1, tuple(p), (0, 0, 0), 0.1, ())
                    for j, p in enumerate(points)}  # fmt: skip
    seed = SeedResult(None, poses, {n: n for n in poses})  # type: ignore[arg-type]
    return colmap_io.ColmapModel({1: camera}, images, model_points), seed


def test_scene_evidence_drops_images_with_few_observations():
    model, seed = _evidence_model([20, 18, 22, 4, 20])
    rejected = scene_evidence(model, seed, 5.0)
    assert rejected == {
        "000003.png": "scene_evidence: 4 triangulated observations < 0.25 x median 20"
    }
    model, seed = _evidence_model([20, 18, 22, 6, 20])
    assert scene_evidence(model, seed, 5.0) == {}


def test_reconstruct_doper_pnp_without_completion(tmp_path):
    fake = install_fake_colmap(tmp_path / "bin")
    gates = DoperSeedGates(register_leftovers=False)
    metadata, frames, _ = _reconstruct(tmp_path, fake, gates=gates, drift=False)
    assert "image_registrator" not in [call[1] for call in metadata.colmap.argv]
    left_out = metadata.image_by_source(NAMES[LEFT_OUT])
    assert not left_out.registered and left_out.registered_by is None
    assert left_out.excluded_reason == "not registered"
    assert left_out.seed_rejected_reason == "detection score 0.1 < 0.3"
    assert metadata.scene_units_per_metre == pytest.approx(1.0, rel=1e-9)


# --- doper prepare RECIPE --stage reconstruct, seed_mode doper_pnp ------------------------

MANY = tuple(f"{i:06d}.png" for i in range(24))


def _recipe(tmp_path: Path, seed_mode: str) -> str:
    data = nurec_recipe_data()
    data["seed_mode"] = seed_mode
    points = np.asarray(_cad_keypoints(), dtype="<f8")
    data["object"]["keypoints_points_sha256"] = hashlib.sha256(points.tobytes()).hexdigest()
    path = tmp_path / f"nurec-{seed_mode}.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


def _prepare(recipe: str, root: Path, stage: str, *extra: str) -> list[str]:
    return ["prepare", recipe, "--root", str(root), "--stage", stage, *extra]


def _cli_capture(tmp_path, seed_mode):
    bundle, root = tmp_path / "bundle", tmp_path / "root"
    write_bundle(bundle, names=MANY, camera=CAMERA, robot=False)
    keypoints = {"keypoints_3d": _cad_keypoints().tolist()}
    (bundle / "keypoints_3d.json").write_text(json.dumps(keypoints))
    recipe = _recipe(tmp_path, seed_mode)
    assert cli.main(_prepare(recipe, root, "capture", "--bundle", str(bundle))) == 0
    return bundle, root, recipe


def test_prepare_reconstruct_mode_c(tmp_path, capsys, monkeypatch):
    import doper.evaluation.relative_pose as relative_pose
    import doper.infer

    fake = install_fake_colmap(tmp_path / "bin")
    monkeypatch.setenv("DOPER_COLMAP", str(fake))
    bundle, root, recipe = _cli_capture(tmp_path, "doper_pnp")
    image_set = load_image_set(bundle)
    observed = _observed(MANY)
    _write_scene(fake, load_frame_map(root / "capture" / "frames"), observed)
    stub = StubInfer(image_set, {name: [_detection(uv)] for name, (_, uv) in observed.items()})
    loaded = []

    def load(checkpoint, device):
        loaded.append(device)
        return "m", "cfg", {}

    monkeypatch.setattr(relative_pose, "load_evaluation_model", load)
    monkeypatch.setattr(doper.infer, "infer_image", stub)
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"weights")
    capsys.readouterr()
    args = _prepare(recipe, root, "reconstruct", "--checkpoint", str(checkpoint), "--device",
                    "cpu")  # fmt: skip
    assert cli.main(args) == 0, capsys.readouterr().out
    out = capsys.readouterr().out
    assert "doper_pnp" in out and "CAD-keypoint scale" in out
    assert loaded == ["cpu"] and len(stub.calls) == len(MANY)
    metadata = ReconstructionMetadata.read(root / "reconstruction" / "reconstruction.json")
    assert metadata.world_frame == "object_cad" and metadata.reference_path is False
    assert metadata.inputs["checkpoint"] == hashlib.sha256(b"weights").hexdigest()
    assert "doper_pnp_gates" in metadata.inputs
    assert all(r.seeded and r.registered_by == "seed" for r in metadata.images)
    # A rerun with the same inputs keeps the reconstruction without running DOPER again.
    assert cli.main(args) == 0
    assert "already reconstructed" in capsys.readouterr().out
    assert len(stub.calls) == len(MANY)


def test_prepare_reconstruct_checkpoint_only_for_doper_pnp(tmp_path, capsys, monkeypatch):
    fake = install_fake_colmap(tmp_path / "bin")
    monkeypatch.setenv("DOPER_COLMAP", str(fake))
    _, root, recipe = _cli_capture(tmp_path, "colmap_sfm")
    capsys.readouterr()
    args = _prepare(recipe, root, "reconstruct", "--checkpoint", str(tmp_path / "m.pth"))
    assert cli.main(args) == 1
    assert "--checkpoint" in capsys.readouterr().out
    doper_recipe = _recipe(tmp_path, "doper_pnp")
    assert cli.main(_prepare(doper_recipe, root, "reconstruct", "--dry-run")) == 1
    assert "needs --checkpoint" in capsys.readouterr().out
    assert recorded_calls(fake) == []
