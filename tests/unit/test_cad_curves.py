# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import numpy as np
import pytest

from doper.cad_curves import (
    CADCurve,
    ProjectedCurves,
    curve_heatmap_metrics,
    geometry_sha256,
    load_projected_curves,
    overlay_named_curve_heatmaps,
    parse_curve_schema,
    project_curve,
    project_schema,
    rasterize_curve_heatmaps,
    refine_pose_with_named_curves,
    sample_named_curve_scores,
    save_projected_curves,
    transform_projected_curves,
)
from doper.infer import Detection, refine_detection_pose_with_curves


def _schema_payload():
    return {
        "version": 1,
        "object_id": "part",
        "curves": [
            {
                "id": 0,
                "name": "top",
                "closed": False,
                "points_xyz": [[-1.0, 0.0, 2.0], [1.0, 0.0, 2.0]],
            },
            {
                "id": 1,
                "name": "loop",
                "closed": True,
                "points_xyz": [
                    [-1.0, -1.0, 3.0],
                    [1.0, -1.0, 3.0],
                    [1.0, 1.0, 3.0],
                ],
            },
        ],
    }


def _projected(*curves):
    arrays = []
    offsets = [0]
    valid = []
    for curve in curves:
        array = np.asarray(curve, dtype=np.float32).reshape(-1, 2)
        arrays.append(array)
        offsets.append(offsets[-1] + len(array))
        valid.append(float(len(array) >= 2))
    return ProjectedCurves(
        points_xy=np.concatenate(arrays, axis=0),
        offsets=np.asarray(offsets, dtype=np.int64),
        valid=np.asarray(valid, dtype=np.float32),
        schema_sha256="abc",
    )


def test_schema_is_ordered_versioned_and_hashed():
    schema = parse_curve_schema(_schema_payload())

    assert schema.version == 1
    assert schema.object_id == "part"
    assert schema.names == ("top", "loop")
    assert schema.num_curves == 2
    assert schema.geometry_sha256 == geometry_sha256(schema.object_id, schema.curves)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda payload: payload.update(version=9), "version"),
        (lambda payload: payload["curves"][1].update(id=3), "contiguous"),
        (lambda payload: payload["curves"][1].update(name="top"), "duplicate"),
        (
            lambda payload: payload["curves"][0].update(points_xyz=[[0, 0, 0]]),
            "shape",
        ),
    ],
)
def test_schema_rejects_incompatible_definitions(mutation, match):
    payload = json.loads(json.dumps(_schema_payload()))
    mutation(payload)
    with pytest.raises(ValueError, match=match):
        parse_curve_schema(payload)


def test_projection_clips_segment_crossing_near_plane():
    curve = CADCurve(
        id=0,
        name="crossing",
        points_xyz=np.asarray([[-1.0, 0.0, -1.0], [1.0, 0.0, 1.0]]),
    )
    camera = np.eye(4)
    intrinsics = np.eye(3)

    projected = project_curve(curve, camera, intrinsics, near=0.5)

    assert projected.shape == (2, 2)
    assert np.isfinite(projected).all()
    np.testing.assert_allclose(projected[-1], [1.0, 0.0])


def test_closed_curve_adds_last_to_first_segment():
    curve = CADCurve(
        id=0,
        name="closed",
        points_xyz=np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]),
        closed=True,
    )

    projected = project_curve(curve, np.eye(4), np.eye(3))

    # Three disconnected segment pairs with two NaN separators.
    assert projected.shape == (8, 2)
    assert np.isnan(projected).all(axis=1).sum() == 2


def test_sidecar_round_trip_and_schema_check(tmp_path):
    projected = _projected([[1, 2], [3, 4]], [[5, 6], [7, 8]])
    path = tmp_path / "curves.npz"

    save_projected_curves(str(path), projected)
    loaded = load_projected_curves(str(path), expected_num_curves=2, expected_schema_sha256="abc")

    np.testing.assert_array_equal(loaded.points_xy, projected.points_xy)
    np.testing.assert_array_equal(loaded.offsets, projected.offsets)
    np.testing.assert_array_equal(loaded.valid, projected.valid)
    with pytest.raises(ValueError, match="schema hash"):
        load_projected_curves(str(path), expected_schema_sha256="wrong")


def test_affine_transform_preserves_nan_piece_separators():
    projected = ProjectedCurves(
        points_xy=np.asarray([[1, 2], [3, 4], [np.nan, np.nan], [5, 6]]),
        offsets=np.asarray([0, 4]),
        valid=np.asarray([1.0]),
    )
    affine = np.asarray([[2, 0, 10], [0, 3, -2]], dtype=np.float32)

    transformed = transform_projected_curves(projected, affine)

    np.testing.assert_allclose(transformed.points_xy[:2], [[12, 4], [16, 10]])
    assert np.isnan(transformed.points_xy[2]).all()
    np.testing.assert_allclose(transformed.points_xy[3], [20, 16])


def test_rasterizer_has_gaussian_distance_profile_and_channel_isolation():
    projected = _projected(
        [[0, 16], [32, 16]],
        [[0, 48], [32, 48]],
    )

    target, weight = rasterize_curve_heatmaps(projected, crop_size=65, heatmap_size=65, sigma=2.0)

    np.testing.assert_array_equal(weight, [1.0, 1.0])
    assert target[0, 16, 10] == pytest.approx(1.0)
    assert target[0, 18, 10] == pytest.approx(np.exp(-0.5), rel=1e-5)
    assert target[0, 48, 10] == 0.0
    assert target[1, 48, 10] == pytest.approx(1.0)


def test_rasterizer_clips_segments_and_ignores_wholly_outside_curves():
    projected = _projected(
        [[-10, 20], [20, 20]],
        [[-10, -10], [-5, -5]],
    )

    target, weight = rasterize_curve_heatmaps(projected, crop_size=65, heatmap_size=65, sigma=1.5)

    np.testing.assert_array_equal(weight, [1.0, 0.0])
    assert target[0, 20, 0] == pytest.approx(1.0)
    assert target[1].sum() == 0.0


def test_named_curve_sampling_prefers_aligned_channel():
    projected = _projected([[0, 16], [32, 16]], [[0, 48], [32, 48]])
    target, _ = rasterize_curve_heatmaps(projected, crop_size=65, heatmap_size=65, sigma=1.5)

    aligned = sample_named_curve_scores(target, projected, crop_size=65)
    swapped = sample_named_curve_scores(target[::-1], projected, crop_size=65)

    assert np.nanmean(aligned) > 0.99
    assert np.nanmean(swapped) < 0.01


def test_curve_metrics_report_per_channel_alignment():
    projected = _projected([[0, 16], [32, 16]], [[0, 48], [32, 48]])
    target, weight = rasterize_curve_heatmaps(projected, crop_size=65, heatmap_size=65, sigma=1.5)

    aligned = curve_heatmap_metrics(target, target, weight)
    swapped = curve_heatmap_metrics(target[::-1], target, weight)

    np.testing.assert_allclose(aligned["iou_per_curve"], [1.0, 1.0])
    np.testing.assert_allclose(aligned["distance_per_curve"], [0.0, 0.0])
    assert swapped["mean_iou"] == 0.0
    assert swapped["mean_distance"] > 10.0


def test_curve_overlay_maps_crop_heatmaps_to_image():
    image = np.zeros((65, 65, 3), dtype=np.uint8)
    probabilities = np.zeros((1, 65, 65), dtype=np.float32)
    probabilities[0, 32, :] = 1.0
    affine = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    overlay = overlay_named_curve_heatmaps(image, probabilities, affine, crop_size=65)

    assert overlay.shape == image.shape
    assert overlay[32].sum() > 0
    assert overlay[0].sum() == 0
    assert image.sum() == 0


def test_pose_refinement_improves_named_curve_alignment():
    schema = parse_curve_schema(
        {
            "version": 1,
            "object_id": "line",
            "curves": [
                {
                    "id": 0,
                    "name": "horizontal",
                    "closed": False,
                    "points_xyz": [[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0]],
                }
            ],
        }
    )
    camera_matrix = np.asarray([[50.0, 0.0, 32.0], [0.0, 50.0, 32.0], [0.0, 0.0, 1.0]])
    camera_from_object = np.eye(4)
    camera_from_object[2, 3] = 1.0
    projected = project_schema(schema, camera_from_object, camera_matrix)
    target, _ = rasterize_curve_heatmaps(projected, crop_size=65, heatmap_size=65, sigma=2.0)
    affine = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    result = refine_pose_with_named_curves(
        target,
        schema,
        initial_rvec=np.zeros(3),
        initial_tvec=np.asarray([0.08, 0.04, 1.0]),
        camera_matrix=camera_matrix,
        image_to_crop=affine,
        crop_size=65,
        steps=60,
        rotation_lr=1e-3,
        translation_lr=5e-3,
        samples_per_segment=24,
        evaluation_points_3d=schema.curves[0].points_xyz,
        evaluation_points_2d=np.asarray([[22.0, 32.0], [42.0, 32.0]]),
    )

    assert result.final_score > result.initial_score
    assert abs(result.tvec[0]) < 0.08
    assert abs(result.tvec[1]) < 0.04
    assert result.final_reprojection_error_px < result.initial_reprojection_error_px


def test_detection_refinement_rejects_schema_mismatch():
    schema = parse_curve_schema(
        {
            "version": 1,
            "object_id": "line",
            "curves": [
                {
                    "id": 0,
                    "name": "line",
                    "points_xyz": [[0, 0, 0], [1, 0, 0]],
                }
            ],
        }
    )
    detection = Detection(
        bbox_xyxy=np.zeros(4),
        score=1.0,
        keypoints=np.zeros((0, 2)),
        keypoint_scores=np.zeros((0,)),
        rvec=np.zeros(3),
        tvec=np.asarray([0, 0, 1.0]),
        curve_heatmap=np.zeros((1, 8, 8), dtype=np.float32),
        curve_names=("line",),
        curve_schema_sha256="wrong",
        crop_to_image=np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="schema hash"):
        refine_detection_pose_with_curves(detection, schema, np.eye(3), crop_size=32)
