# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BOP multi-view fusion: support selection per dataset and fusion over calibrated views."""

import dataclasses
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from bop_recipe_data import TEST_FRAMES, make_hot3d_download

from doper.bop import hot3d_staging as staging
from doper.bop import multiview as bop_mv
from doper.bop import sources
from doper.bop.multiview import ImageKey, MultiviewSpec, fuse_reference, support_groups
from doper.infer import Detection
from doper.recipes.schema import load_recipe

GOLDEN = Path(__file__).parent / "golden" / "hope_multiview_groups.json"
K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
CAMERA_CENTERS = ((0.0, 0.0, -600.0), (180.0, 0.0, -570.0), (-160.0, 60.0, -580.0), (0, 170, -560))
TLESS = MultiviewSpec(convention="w2c", support="nearest", scope="test_targets")
HOPE = MultiviewSpec(
    convention="c2w_named_w2c",
    support="nearest",
    scope="scene",
    scenes=tuple(range(41, 48)),
    skip_nonpositive_support_depth=True,
)


def _look_at(center):
    """World-to-camera (R, t) of a camera at ``center`` looking at the world origin."""
    center = np.asarray(center, dtype=np.float64)
    forward = -center / np.linalg.norm(center)
    right = np.cross([0.0, -1.0, 0.0], forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    return rotation, -rotation @ center


def _w2c_camera(center):
    rotation, translation = _look_at(center)
    return {
        "cam_K": K.ravel().tolist(),
        "cam_R_w2c": rotation.ravel().tolist(),
        "cam_t_w2c": translation.tolist(),
    }


def _c2w_named_w2c(camera):
    world_to_camera = np.eye(4)
    world_to_camera[:3, :3] = np.reshape(camera["cam_R_w2c"], (3, 3))
    world_to_camera[:3, 3] = camera["cam_t_w2c"]
    camera_to_world = np.linalg.inv(world_to_camera)
    return {
        "cam_K": camera["cam_K"],
        "cam_R_w2c": camera_to_world[:3, :3].ravel().tolist(),
        "cam_t_w2c": camera_to_world[:3, 3].tolist(),
    }


def _keypoints_mm():
    rng = np.random.default_rng(3)
    return rng.uniform(-40.0, 40.0, (24, 3))


def _object_in_camera(camera, object_rotation, object_translation):
    rotation = np.reshape(camera["cam_R_w2c"], (3, 3))
    translation = np.asarray(camera["cam_t_w2c"], dtype=np.float64)
    pose_rotation = rotation @ object_rotation
    pose_translation = rotation @ object_translation + translation
    return cv2.Rodrigues(pose_rotation)[0], pose_translation.reshape(3, 1)


def _detection(camera, pose, *, noise=0.3, seed=0, posed=True, perturb=0.0):
    rvec, tvec = pose
    points = _keypoints_mm()
    pixels = cv2.projectPoints(points, rvec, tvec, K, None)[0].reshape(-1, 2)
    rng = np.random.default_rng(seed)
    pixels = (pixels + rng.normal(0.0, noise, pixels.shape)).astype(np.float32)
    x1, y1 = pixels.min(axis=0)
    x2, y2 = pixels.max(axis=0)
    return Detection(
        bbox_xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
        score=np.float32(0.9 - 0.01 * seed),
        keypoints=pixels,
        keypoint_scores=np.full(len(points), 0.9, dtype=np.float32),
        rvec=(rvec + perturb).astype(np.float32) if posed else None,
        tvec=(tvec + 8.0 * perturb).astype(np.float32) if posed else None,
    )


def _rig(convention="w2c"):
    """Reference camera 0 and three supports seeing one object at the world origin."""
    cameras = [_w2c_camera(c) for c in CAMERA_CENTERS]
    rotation = cv2.Rodrigues(np.array([0.3, -0.2, 0.5]))[0]
    poses = [_object_in_camera(c, rotation, np.zeros(3)) for c in cameras]
    if convention == "c2w_named_w2c":
        cameras = [_c2w_named_w2c(c) for c in cameras]
    keys = [ImageKey(48, i) for i in range(len(cameras))]
    return keys, dict(zip(keys, cameras, strict=True)), poses


def _fuse(keys, cameras, detections, spec):
    supports = {k: detections[k] for k in keys[1:]}
    return fuse_reference(detections[keys[0]], supports, cameras, _keypoints_mm(), spec)


def _rig_detections(keys, poses):
    detections = {
        k: [_detection(None, p, seed=i)] for i, (k, p) in enumerate(zip(keys, poses, strict=True))
    }
    detections[keys[0]] = [_detection(None, poses[0], seed=0, perturb=0.02)]
    return detections


def _rotation_error(rotation, rvec):
    delta = np.asarray(rotation) @ cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0].T
    return float(np.degrees(np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1))))


def test_fuse_matches_source_on_synthetic_rig():
    keys, cameras, poses = _rig()
    detections = _rig_detections(keys, poses)
    (fused,) = _fuse(keys, cameras, detections, TLESS)
    assert fused.accepted and fused.reference_index == 0
    assert fused.matched == tuple((k, 0) for k in keys[1:])
    assert len(fused.reprojection_px) == 4 and max(fused.reprojection_px) <= 6.0
    original = detections[keys[0]][0]
    before = np.linalg.norm(np.asarray(original.tvec, dtype=np.float64) - poses[0][1])
    assert np.linalg.norm(fused.t.reshape(3, 1) - poses[0][1]) < before
    assert _rotation_error(fused.R, poses[0][0]) < _rotation_error(
        cv2.Rodrigues(np.asarray(original.rvec, dtype=np.float64))[0], poses[0][0]
    )
    assert fused.R.dtype == np.float64 and fused.t.shape == (3,)


def test_hope_convention_inverts_and_skips_negative_depth(monkeypatch):
    keys, cameras, poses = _rig("c2w_named_w2c")
    detections = _rig_detections(keys, poses)
    (inverted,) = _fuse(keys, cameras, detections, HOPE)
    keys_w2c, cameras_w2c, _ = _rig()
    (direct,) = _fuse(
        keys_w2c, cameras_w2c, dict(zip(keys_w2c, detections.values(), strict=True)), TLESS
    )
    assert inverted.accepted and [k.im_id for k, _ in inverted.matched] == [1, 2, 3]
    np.testing.assert_allclose(inverted.R, direct.R, atol=1e-9)
    np.testing.assert_allclose(inverted.t, direct.t, atol=1e-6)
    # A support camera that sees the original pose behind it is skipped before association.
    behind = ImageKey(48, 9)
    center = np.array([0.0, 0.0, 600.0])
    facing_away = np.diag([-1.0, 1.0, -1.0]) @ _look_at(center)[0]
    camera = {
        "cam_K": K.ravel().tolist(),
        "cam_R_w2c": facing_away.ravel().tolist(),
        "cam_t_w2c": (-facing_away @ center).tolist(),
    }
    rotation = cv2.Rodrigues(np.array([0.3, -0.2, 0.5]))[0]
    behind_pose = _object_in_camera(camera, rotation, np.zeros(3))
    assert behind_pose[1][2, 0] < 0
    calls = []
    real = bop_mv.associate_detection_pairs
    monkeypatch.setattr(
        bop_mv, "associate_detection_pairs", lambda *a, **kw: calls.append(a) or real(*a, **kw)
    )
    group = [keys[0], behind]
    rig = {keys[0]: cameras[keys[0]], behind: _c2w_named_w2c(camera)}
    dets = {keys[0]: detections[keys[0]], behind: [_detection(None, behind_pose, noise=0.0)]}
    (skipped,) = _fuse(group, rig, dets, HOPE)
    assert calls == [] and skipped.matched == () and not skipped.accepted
    unskipped = MultiviewSpec(
        convention="c2w_named_w2c", support="nearest", scope="scene", scenes=(48,)
    )
    _fuse(group, rig, dets, unskipped)
    assert len(calls) == 1


def test_support_groups_match_hope_protocol():
    golden = json.loads(GOLDEN.read_text())
    cameras = {
        ImageKey(int(scene), int(im)): camera
        for scene, frames in golden["cameras"].items()
        for im, camera in frames.items()
    }
    population = [ImageKey(scene, im) for scene, im in golden["targets"]]
    groups = support_groups(population, cameras, load_recipe("hopev2").multiview)
    actual = {f"{k.scene_id}/{k.im_id}": [v.im_id for v in g] for k, g in groups.items()}
    assert actual == golden["groups"]
    assert all(len(g) == 4 for k, g in groups.items() if k.scene_id >= 41)


def test_tless_groups_match_bop19_schedule():
    rng = np.random.default_rng(5)
    centers = {i: rng.uniform(-500, 500, 3) + [0, 0, -700] for i in range(12)}
    centers[7] = centers[3] * 1.0  # a distance tie with frame 3 is broken by image id
    cameras = {ImageKey(s, i): _w2c_camera(c) for s in (1, 2) for i, c in centers.items()}
    targets = [
        {"scene_id": s, "im_id": i, "obj_id": obj, "inst_count": 2}
        for s, ids in ((1, (0, 3, 5, 7, 9, 11)), (2, (1, 2, 3, 4)))
        for i in ids
        for obj in (4, 9)
    ]
    population = bop_mv.target_images(targets)
    assert population == sorted({ImageKey(t["scene_id"], t["im_id"]) for t in targets})
    groups = support_groups(population, cameras, load_recipe("tless").multiview)
    for key, group in groups.items():
        same_scene = [k for k in population if k.scene_id == key.scene_id and k != key]
        expected = sorted(
            same_scene,
            key=lambda k: (np.linalg.norm(centers[k.im_id] - centers[key.im_id]), k.im_id),
        )[:3]
        assert group == (key, *expected)
    # Frames 3 and 7 share a camera center: frame 3 always precedes frame 7.
    assert (
        groups[ImageKey(1, 9)].index(ImageKey(1, 3)) + 1
        == groups[ImageKey(1, 9)].index(ImageKey(1, 7))
        or ImageKey(1, 7) not in groups[ImageKey(1, 9)]
    )
    assert set(groups) == set(population)


def test_fused_pose_records_detection_indices():
    keys, cameras, poses = _rig()
    detections = _rig_detections(keys, poses)
    shifted = (poses[1][0], poses[1][1] + np.array([[120.0], [0.0], [0.0]]))
    detections[keys[1]] = [_detection(None, shifted, seed=4), detections[keys[1]][0]]
    no_pose = _detection(None, poses[0], posed=False)
    detections[keys[0]] = [no_pose, detections[keys[0]][0]]
    (fused,) = _fuse(keys, cameras, detections, TLESS)
    assert fused.reference_index == 1
    assert fused.matched[0] == (keys[1], 1)
    assert fused.matched == ((keys[1], 1), (keys[2], 0), (keys[3], 0))


def test_hope_scenes_1_40_single_view():
    golden = json.loads(GOLDEN.read_text())
    cameras = {ImageKey(1, int(im)): c for im, c in golden["cameras"]["1"].items()}
    # Even with extrinsics, a scene outside 41-47 stays single-view.
    keys, rig, poses = _rig("c2w_named_w2c")
    cameras.update({ImageKey(40, k.im_id): c for k, c in rig.items()})
    population = sorted(cameras)
    groups = support_groups(population, cameras, HOPE)
    assert all(group == (key,) for key, group in groups.items())
    detections = _rig_detections(keys, poses)[keys[0]]
    (fused,) = fuse_reference(detections, {}, {keys[0]: rig[keys[0]]}, _keypoints_mm(), HOPE)
    assert not fused.accepted and fused.matched == ()


def test_hot3d_groups_synchronized_streams(tmp_path):
    root = tmp_path / "root"
    profile = sources.load_profile(make_hot3d_download(root, tmp_path / "profile"))
    sources.fetch(profile, root, ["test"], allow_transductive=True)
    sources.write_ready(root, profile, ["test"])
    path = staging.stage_hot3d_support(root, profile)
    manifest = json.loads(path.read_text())
    assert manifest["identity"]["camera_extrinsics_used"] is True
    assert "pending author confirmation" in manifest["identity"]["grouping"]
    assert [(g["scene_id"], g["im_id"]) for g in manifest["groups"]] == [
        (3365, f) for f in TEST_FRAMES
    ]
    group = manifest["groups"][0]
    assert [v["stream"] for v in group["supports"]] == ["1201-1", "1201-2"]
    support = group["supports"][0]
    image = cv2.imread(str(path.parent / support["path"]), cv2.IMREAD_UNCHANGED)
    assert image.ndim == 3 and image.shape[2] == 3
    cameras = staging.support_cameras(manifest)
    reference = ImageKey(3365, 29, "214-1")
    spec = load_recipe("hot3d").multiview
    groups = support_groups([reference], cameras, spec)
    assert groups[reference] == (
        reference,
        ImageKey(3365, 29, "1201-1"),
        ImageKey(3365, 29, "1201-2"),
    )
    # The world-from-camera translation is staged in millimeters under the named-w2c fields.
    assert cameras[ImageKey(3365, 29, "1201-1")]["cam_t_w2c"] == pytest.approx([60.0, 0.0, 0.0])
    # Quest3: reference 1201-1 and its single synchronized support 1201-2.
    quest = {ImageKey(1300, 29, s): cameras[ImageKey(3365, 29, s)] for s in ("1201-1", "1201-2")}
    assert support_groups([ImageKey(1300, 29, "1201-1")], quest, spec) == {
        ImageKey(1300, 29, "1201-1"): (ImageKey(1300, 29, "1201-1"), ImageKey(1300, 29, "1201-2"))
    }
    # Re-running reuses the clip receipts and leaves the manifest unchanged.
    assert staging.stage_hot3d_support(root, profile).read_bytes() == path.read_bytes()
    # Deployment timing decodes the native support image and rectifies it with precomputed maps
    # (hash-checked, K equal to the staged cam_K) to the staged pixels before JPEG coding.
    from doper.bop import hot3d
    from doper.bop.deployment_timing import rectify_native
    from doper.bop.export import TestFrame
    from doper.cli.report import decode_views

    key = ImageKey(3365, 29, "1201-1")
    frame = TestFrame(3365, 29, path.parent / support["path"], support["rgb_sha256"],
                      tuple(support["cam_K"]), native_path=path.parent / support["native_path"],
                      native_rgb_sha256=support["native_rgb_sha256"],
                      native_calibration=support["native_calibration"])  # fmt: skip
    images, maps = decode_views({key: frame}, [key])
    native = cv2.imread(str(frame.native_path), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(images[key], native)
    expected, _ = hot3d.rectify(native, support["native_calibration"])
    np.testing.assert_array_equal(rectify_native(images[key], maps[key]), expected)
    bad = dataclasses.replace(frame, camera_matrix=(1.0,) * 9)
    with pytest.raises(ValueError, match="cam_K"):
        decode_views({key: bad}, [key])


def test_unmatched_reference_keeps_original_pose():
    keys, cameras, poses = _rig()
    detections = _rig_detections(keys, poses)
    far = (poses[1][0], poses[1][1] + np.array([[200.0], [0.0], [0.0]]))
    supports = {keys[1]: [_detection(None, far)]}
    reference = detections[keys[0]]
    group = {k: cameras[k] for k in keys[:2]}
    (fused,) = fuse_reference(reference, supports, group, _keypoints_mm(), TLESS)
    assert not fused.accepted and fused.matched == () and fused.reprojection_px == ()
    original = reference[0]
    np.testing.assert_array_equal(
        fused.R, cv2.Rodrigues(np.asarray(original.rvec, dtype=np.float64))[0]
    )
    np.testing.assert_array_equal(fused.t, np.asarray(original.tvec, dtype=np.float64).ravel())


def test_recipe_multiview_blocks():
    tless = load_recipe("tless").multiview
    assert (tless.convention, tless.support, tless.nearest_k, tless.scope) == (
        "w2c",
        "nearest",
        3,
        "test_targets",
    )
    assert not tless.skip_nonpositive_support_depth
    hope = load_recipe("hopev2").multiview
    assert hope.convention == "c2w_named_w2c" and hope.scope == "scene"
    assert hope.scenes == tuple(range(41, 48)) and hope.skip_nonpositive_support_depth
    hot3d = load_recipe("hot3d").multiview
    assert hot3d.support == "streams" and hot3d.skip_nonpositive_support_depth
    assert hot3d.streams == ("214-1", "1201-1", "1201-2")
    assert load_recipe("handal").multiview is None
    for spec in (tless, hope, hot3d):
        assert spec.prefilter_px == 30.0
        gate = spec.gate
        assert (gate.kp_score_thr, gate.min_keypoints, gate.ransac_px) == (0.3, 6, 3.0)
        assert (gate.max_reprojection_px, gate.min_depth, gate.max_depth) == (6.0, 50.0, 3000.0)


def test_multiview_spec_rejects_inconsistent_values():
    with pytest.raises(ValueError, match="convention"):
        MultiviewSpec(convention="c2w", support="nearest")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="streams"):
        MultiviewSpec(convention="c2w_named_w2c", support="streams")
    with pytest.raises(ValueError, match="nearest_k"):
        MultiviewSpec(convention="w2c", support="nearest", nearest_k=0)
