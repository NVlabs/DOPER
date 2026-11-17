# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BOP validation average recall: protocols, scorer and ground-truth box diagnostics."""

import copy
import json
from dataclasses import dataclass

import cv2
import numpy as np
import pytest
from test_box_dither import BOX, NUM_KEYPOINTS, _tiny_setup

from doper.bop import hot3d
from doper.bop.ar import (
    PROTOCOL_HOT3D_NATIVE,
    PROTOCOL_TOP1,
    PROTOCOL_TOP5,
    JitterSpec,
    aggregate,
    gt_box_keypoints,
    independent_scores,
    matching_hits,
    score_model,
)
from doper.bop.hot3d import project_native
from doper.bop.sources import sha256_file
from doper.box_dither import keypoints_from_boxes
from doper.evaluation.bop_metrics import toolkit_mspd, toolkit_symmetry_transforms
from doper.infer import Detection


def test_gt_box_keypoints_returns_one_detection_per_box():
    model, cfg, _, _, image = _tiny_setup()
    boxes = [tuple(BOX), (10.0, 10.0, 50.0, 60.0)]
    detections = gt_box_keypoints(model, image, boxes, cfg, "cpu", batch_size=1)
    assert len(detections) == 2
    for detection, box in zip(detections, boxes, strict=True):
        np.testing.assert_allclose(detection.bbox_xyxy, box)
        assert detection.score == 1.0
        assert detection.keypoints.shape == (NUM_KEYPOINTS, 2)
        assert detection.keypoint_scores.dtype == np.float32
        assert detection.rvec is None
    reference = keypoints_from_boxes(model, image, cfg, "cpu", boxes, batch_size=2)
    np.testing.assert_allclose(detections[1].keypoints, reference[1].keypoints, atol=1e-4)


def test_gt_box_keypoints_rejects_empty_boxes():
    model, cfg, _, _, image = _tiny_setup()
    with pytest.raises(ValueError, match="at least one box"):
        gt_box_keypoints(model, image, [], cfg, "cpu")


# --- Validation AR protocols -------------------------------------------------------------------

OBJ = 1
WIDTH, HEIGHT = 640, 480
FOCAL = 768.0
DIAMETER = 173.2
MESH = np.array(
    [[sx * 50.0, sy * 30.0, sz * 20.0] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    + [[10.0, -5.0, 20.0], [-25.0, 12.0, -20.0]]
)
GT_POSE = (np.array([0.3, -0.2, 0.4]), np.array([10.0, -5.0, 600.0]))
SYMMETRY_INFOS = {
    "asymmetric": {},
    "discrete": {"symmetries_discrete": [[-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]]},
    "continuous": {"symmetries_continuous": [{"axis": [0, 0, 1], "offset": [0, 0, 0]}]},
    "offset": {"symmetries_continuous": [{"axis": [0.0, 1.0, 2.0], "offset": [3.0, -2.0, 1.5]}]},
    "composed": {
        "symmetries_discrete": [[1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 4, 0, 0, 0, 1]],
        "symmetries_continuous": [{"axis": [0, 0, 1], "offset": [1, 2, 0]}],
    },
}
FISHEYE_DISTORTION = [0.12, -0.05, 0.01, 0.002, 0.0005, -0.0004, 0.0002, 0.0001]
FISHEYE_DISTORTION += [0.001, -0.001, 0.0002, 0.0001]
CALIBRATION = {
    "calibration": {
        "image_width": WIDTH,
        "image_height": HEIGHT,
        "projection_model_type": "CameraModelType.FISHEYE624",
        "projection_params": [FOCAL, WIDTH / 2, HEIGHT / 2, *FISHEYE_DISTORTION],
        "label": "fixture",
        "serial_number": "fixture",
    },
    "T_world_from_camera": {"quaternion_wxyz": [1, 0, 0, 0], "translation_xyz": [0, 0, 0]},
}
CAMERA_K = [FOCAL, 0.0, WIDTH / 2, 0.0, FOCAL, HEIGHT / 2, 0.0, 0.0, 1.0]


def _write_models(root, info=None):
    models = root / "models_eval"
    models.mkdir(parents=True)
    header = [
        "ply", "format ascii 1.0", f"element vertex {len(MESH)}", "property float x",
        "property float y", "property float z", "end_header",
    ]  # fmt: skip
    rows = [" ".join(f"{v:.6f}" for v in point) for point in MESH]
    (models / f"obj_{OBJ:06d}.ply").write_text("\n".join(header + rows) + "\n")
    entry = {"diameter": DIAMETER, **(info or {})}
    (models / "models_info.json").write_text(json.dumps({str(OBJ): entry}))
    return models


def _gt(rvec, tvec, obj=OBJ):
    rotation = cv2.Rodrigues(np.asarray(rvec, float))[0]
    return {
        "cam_R_m2c": rotation.ravel().tolist(),
        "cam_t_m2c": list(map(float, tvec)),
        "obj_id": obj,
    }


def _write_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((HEIGHT, WIDTH, 3), np.uint8))
    return path


def _write_split(root, gts_per_frame):
    """BOP split directory: one scene, one frame per entry of ``gts_per_frame``."""
    scene = root / "val" / "000001"
    scene_gt, scene_camera, scene_info = {}, {}, {}
    for frame, gts in enumerate(gts_per_frame):
        _write_image(scene / "rgb" / f"{frame:06d}.png")
        scene_gt[str(frame)] = gts
        scene_camera[str(frame)] = {"cam_K": CAMERA_K, "depth_scale": 1.0}
        scene_info[str(frame)] = [{"bbox_visib": [200, 150, 120, 100]} for _ in gts]
    for name, value in (("scene_gt", scene_gt), ("scene_camera", scene_camera),
                        ("scene_gt_info", scene_info)):  # fmt: skip
        (scene / f"{name}.json").write_text(json.dumps(value))
    return root / "val"


def _write_native_manifest(root, gts):
    image = _write_image(root / "val" / "000001" / "000000.png")
    frame = {
        "scene": "000001", "frame": 0, "path": str(image), "sha256": sha256_file(image),
        "width": WIDTH, "height": HEIGHT, "cam_K": CAMERA_K, "native_calibration": CALIBRATION,
        "objects": [g["obj_id"] for g in gts], "device": "Aria", "stream": "214-1",
        "GT": [{**g, "rectified_amodal_bbox": [200, 150, 120, 100]} for g in gts],
    }  # fmt: skip
    path = root / "validation_manifest.json"
    path.write_text(json.dumps({"dataset": "hot3d", "frames": [frame]}))
    return path


def _detection(rvec, tvec, score):
    return Detection(
        bbox_xyxy=np.array([200.0, 150.0, 320.0, 250.0]),
        score=score,
        keypoints=np.zeros((4, 2)),
        keypoint_scores=np.ones(4, np.float32),
        rvec=np.asarray(rvec, float).reshape(3, 1),
        tvec=np.asarray(tvec, float).reshape(3, 1),
        reproj_error_px=1.0,
    )


@dataclass
class StubPredictor:
    """Predictor returning fixed detections, in frame order."""

    frames: list[list[Detection]]
    calls: int = 0

    def detect(self, image, camera_matrix):
        self.calls += 1
        return [copy.deepcopy(d) for d in self.frames[self.calls - 1]]

    def predict_in_box(self, image, box_xyxy, camera_matrix):
        return copy.deepcopy(self.frames[self.calls - 1][0])

    def refine(self, image, detection, camera_matrix, spec):
        return detection, {"attempted": False, "adopted": False, "proposal_count": 0}


KEYPOINTS_MM = MESH[:4]
OFF_AXIS_T = np.array([180.0, -130.0, 500.0])


def _near_gt(delta_r=0.05, delta_t=(4.0, -3.0, 8.0)):
    return GT_POSE[0] + delta_r, GT_POSE[1] + np.asarray(delta_t)


def test_hot3d_ar_uses_native_camera(tmp_path):
    models = _write_models(tmp_path, SYMMETRY_INFOS["discrete"])
    gt_pose = (GT_POSE[0], OFF_AXIS_T)  # far from the principal point, where distortion matters
    manifest = _write_native_manifest(tmp_path, [_gt(*gt_pose)])
    rvec, tvec = gt_pose[0] + 0.05, OFF_AXIS_T + np.array([4.0, -3.0, 8.0])
    results = {
        camera: score_model(
            StubPredictor([[_detection(rvec, tvec, 0.9)]]),
            manifest,
            obj_id=OBJ,
            keypoints_3d_mm=KEYPOINTS_MM,
            models_dir=models,
            protocol="top5-greedy",
            camera=camera,
        )  # fmt: skip
        for camera in ("native", "pinhole")
    }
    native_mspd = results["native"].rows[0]["arms"]["single"]["mspd_px640"][0][0]
    pinhole_mspd = results["pinhole"].rows[0]["arms"]["single"]["mspd_px640"][0][0]
    assert results["native"].protocol == PROTOCOL_HOT3D_NATIVE
    assert abs(native_mspd - pinhole_mspd) > 0.05 * pinhole_mspd
    rotation = cv2.Rodrigues(rvec)[0]
    gt_rotation = cv2.Rodrigues(GT_POSE[0])[0]
    estimate = project_native(MESH, rotation, tvec, CALIBRATION)
    expected = min(
        np.linalg.norm(
            estimate
            - project_native(
                MESH, gt_rotation @ sym[:3, :3], gt_rotation @ sym[:3, 3] + OFF_AXIS_T, CALIBRATION
            ),  # fmt: skip
            axis=1,
        ).max()
        for sym in toolkit_symmetry_transforms(SYMMETRY_INFOS["discrete"])
    )
    assert native_mspd == pytest.approx(expected * 640 / WIDTH, rel=1e-12)
    pinhole_expected = toolkit_mspd(
        rotation, tvec, gt_rotation, OFF_AXIS_T, np.reshape(CAMERA_K, (3, 3)), MESH,
        toolkit_symmetry_transforms(SYMMETRY_INFOS["discrete"]),
    )  # fmt: skip
    assert pinhole_mspd == pytest.approx(pinhole_expected * 640 / WIDTH, rel=1e-12)


def test_score_records_checkpoint_geometry_and_camera_lineage(tmp_path):
    models = _write_models(tmp_path)
    manifest = _write_native_manifest(tmp_path, [_gt(*GT_POSE)])
    lineage = {"checkpoint_sha256": "c" * 64, "geometry_sha256": "g" * 64}
    results = {
        camera: score_model(
            StubPredictor([[_detection(*GT_POSE, 0.9)]]),
            manifest,
            obj_id=OBJ,
            keypoints_3d_mm=KEYPOINTS_MM,
            models_dir=models,
            protocol="top5-greedy",
            camera=camera,
            **lineage,
        )  # fmt: skip
        for camera in ("native", "pinhole")
    }
    for result in results.values():
        assert (result.checkpoint_sha256, result.geometry_sha256) == ("c" * 64, "g" * 64)
    assert results["native"].camera_code_sha256 == sha256_file(hot3d.__file__)
    assert results["pinhole"].camera_code_sha256 is None


def test_top1_native_has_no_source_protocol(tmp_path):
    models = _write_models(tmp_path)
    manifest = _write_native_manifest(tmp_path, [_gt(*GT_POSE)])
    with pytest.raises(ValueError, match="no validation protocol"):
        score_model(
            StubPredictor([[]]), manifest, obj_id=OBJ, keypoints_3d_mm=KEYPOINTS_MM,
            models_dir=models, protocol="top1", camera="native",
        )  # fmt: skip


def test_top5_greedy_matches_one_to_one(tmp_path):
    assert matching_hits([[1.0, 2.0], [0.5, 9.0]], 3.0, 2) == 1
    assert matching_hits([[0.5, 9.0], [1.0, 2.0]], 3.0, 2) == 2
    assert matching_hits([[None, float("nan")], [2.9, 3.0]], 3.0, 2) == 1
    with pytest.raises(ValueError, match="shape"):
        matching_hits([[1.0]], 3.0, 2)
    models = _write_models(tmp_path)
    second = (np.array([-0.4, 0.1, 0.2]), np.array([-60.0, 40.0, 700.0]))
    split = _write_split(tmp_path, [[_gt(*GT_POSE), _gt(*second)]])
    duplicates = [_detection(*GT_POSE, 0.9), _detection(*GT_POSE, 0.8)]
    result = score_model(
        StubPredictor([duplicates]), split, obj_id=OBJ, keypoints_3d_mm=KEYPOINTS_MM,
        models_dir=models, protocol="top5-greedy", gt_box=True,
    )  # fmt: skip
    assert result.protocol == PROTOCOL_TOP5
    assert (result.frames, result.gt_instances) == (1, 2)
    single = result.arms["single"]
    assert single.mssd_recalls == (0.5,) * 10 and single.mspd_recalls == (0.5,) * 10
    assert single.ar_mean == pytest.approx(50.0)
    assert set(result.arms) == {"single", "gtbox"}
    assert result.gtbox_keypoints is not None and result.gtbox_keypoints["points"] > 0
    assert independent_scores(result, DIAMETER)["single"] == pytest.approx(single.ar_mean)


def test_top1_scores_highest_detection_and_counts_misses(tmp_path):
    models = _write_models(tmp_path)
    split = _write_split(tmp_path, [[_gt(*GT_POSE)], [_gt(*GT_POSE)]])
    predictor = StubPredictor([[_detection(*GT_POSE, 0.9), _detection(*_near_gt(0.5), 0.5)], []])
    result = score_model(
        predictor, split, obj_id=OBJ, keypoints_3d_mm=KEYPOINTS_MM, models_dir=models,
        protocol="top1", jitter=JitterSpec(),
    )  # fmt: skip
    assert result.protocol == PROTOCOL_TOP1 and result.status == "scored"
    single = result.arms["single"]
    assert single.pose_rate == 0.5 and single.mssd_recalls == (0.5,) * 10
    assert result.rows[1]["arms"]["single"]["prediction"] is None
    assert set(result.arms) == {"single", "jitter"}
    assert len(result.population_sha256) == 64


def test_top1_requires_one_instance_and_reports_absent_object(tmp_path):
    models = _write_models(tmp_path)
    split = _write_split(tmp_path, [[_gt(*GT_POSE), _gt(*GT_POSE)]])
    with pytest.raises(ValueError, match="one target instance"):
        score_model(
            StubPredictor([[]]), split, obj_id=OBJ, keypoints_3d_mm=KEYPOINTS_MM,
            models_dir=models, protocol="top1",
        )  # fmt: skip
    absent = score_model(
        StubPredictor([]), split, obj_id=OBJ + 1, keypoints_3d_mm=KEYPOINTS_MM,
        models_dir=models, protocol="top1",
    )  # fmt: skip
    assert absent.status == "not_evaluable" and absent.frames == 0 and not absent.arms


def test_scoring_matches_bop_toolkit(tmp_path):
    pytest.importorskip("bop_toolkit_lib")
    from bop_toolkit_lib import misc, pose_error

    gt_poses = [GT_POSE, (np.array([-0.4, 0.1, 0.2]), np.array([-60.0, 40.0, 700.0]))]
    detections = [_detection(*_near_gt(0.02 * k, (1.5 * k, -k, 3.0 * k)), 0.9 - 0.1 * k)
                  for k in range(3)]  # fmt: skip
    for name, info in SYMMETRY_INFOS.items():
        root = tmp_path / name
        models = _write_models(root, info)
        split = _write_split(root, [[_gt(*pose) for pose in gt_poses]])
        result = score_model(
            StubPredictor([detections]), split, obj_id=OBJ, keypoints_3d_mm=KEYPOINTS_MM,
            models_dir=models, protocol="top5-greedy",
        )  # fmt: skip
        reference_syms = misc.get_symmetry_transformations(
            {"diameter": DIAMETER, **info}, max_sym_disc_step=0.01
        )
        ours = toolkit_symmetry_transforms(info)
        assert len(ours) == len(reference_syms)
        for mine, theirs in zip(ours, reference_syms, strict=True):
            np.testing.assert_allclose(mine[:3, :3], theirs["R"], rtol=0, atol=1e-9)
            np.testing.assert_allclose(mine[:3, 3:], theirs["t"], rtol=0, atol=1e-9)
        camera_matrix = np.reshape(CAMERA_K, (3, 3))
        mssd, mspd = [], []
        for det in detections:
            rotation, tvec = cv2.Rodrigues(det.rvec)[0], det.tvec.reshape(3, 1)
            mssd.append([]), mspd.append([])
            for rvec, tvec_gt in gt_poses:
                gt_rotation, gt_t = cv2.Rodrigues(rvec)[0], tvec_gt.reshape(3, 1)
                mssd[-1].append(
                    float(pose_error.mssd(rotation, tvec, gt_rotation, gt_t, MESH, reference_syms))
                )
                mspd[-1].append(float(pose_error.mspd(
                    rotation, tvec, gt_rotation, gt_t, camera_matrix, MESH, reference_syms
                )) * 640 / WIDTH)  # fmt: skip
        single = result.rows[0]["arms"]["single"]
        np.testing.assert_allclose(single["mssd_mm"], mssd, rtol=0, atol=1e-9)
        np.testing.assert_allclose(single["mspd_px640"], mspd, rtol=0, atol=1e-9)
        expected = aggregate(
            [{"gt_indices": [0, 1], "arms": {"single": {"mssd_mm": mssd, "mspd_px640": mspd}}}],
            DIAMETER, arms=("single",),
        )["single"]  # fmt: skip
        arm = result.arms["single"]
        np.testing.assert_allclose(arm.mssd_recalls, expected["mssd_recalls"], rtol=0, atol=1e-9)
        np.testing.assert_allclose(arm.mspd_recalls, expected["mspd_recalls"], rtol=0, atol=1e-9)
        assert arm.ar_mean == pytest.approx(expected["AR_mean"], abs=1e-9)
