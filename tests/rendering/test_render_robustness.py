# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Robustness and robust-appearance validation and backend guards.

Sampling tests run through ``rendering.scene_spec``; identity tests through ``run_manifest``.
"""

import numpy as np
import pytest

from doper.scene import geometry as G  # noqa: N812
from rendering import scene_spec as S  # noqa: N812
from rendering.config import render_config_from_mapping
from rendering.robust_appearance import validate_robust_appearance
from rendering.robustness import require_backend, validate_robustness
from rendering.run_manifest import config_values, normalized_scene_config

BASE_DIR = "/nonexistent"
BOX = np.array([[-0.05, -0.04, -0.03], [0.05, 0.04, 0.03]])
THIN_BOX = np.array([[-0.025, -0.06, -0.025], [0.025, 0.06, 0.025]])


@pytest.mark.parametrize(
    "options",
    [
        {"mode_weights": {"clipping": -0.1, "easy": 1.1}},
        {"mode_weights": {"banana": 1}},
        {"mode_weights": {"easy": 0.5}},
        {"clipping_fraction": [0, 1]},
        {"clipping_edges": ["center"]},
        {"proxy_count": [0, 3]},
        {"proxy_radius_fraction": [0.1, float("nan")]},
    ],
)
def test_reject_invalid_robustness(options):
    with pytest.raises(ValueError, match="robustness"):
        validate_robustness(options)


def test_robustness_defaults_are_filled_in():
    robust = validate_robustness({})
    assert robust["mode_weights"]["easy"] == 0.4
    assert robust["proxy_count"] == [1, 3]
    assert validate_robustness(None) is None


def test_robustness_requires_masks_for_realized_coverage():
    with pytest.raises(ValueError, match="render_masks"):
        render_config_from_mapping({"robustness": {}}, "/tmp")


@pytest.mark.parametrize(
    "backend,scenes,export",
    [("isaac", True, False), ("blender", False, False), ("blender", True, True)],
)
def test_unsupported_render_modes_fail_instead_of_silently_losing_augmentation(
    backend, scenes, export
):
    with pytest.raises(ValueError, match="robustness"):
        require_backend({}, backend, scenes=scenes, export_environment=export)


def test_appearance_alone_cannot_silently_render_in_unsupported_backend():
    with pytest.raises(ValueError, match="robustness"):
        require_backend(None, "isaac", appearance={"enabled": True})
    require_backend(None, "isaac", appearance=None)


@pytest.mark.parametrize(
    "section", ["yes", {"enabled": True}, {"enabled": True, "axis": "y", "typo": 1}]
)
def test_appearance_section_is_validated_at_config_load(section):
    with pytest.raises(ValueError, match="robust_appearance"):
        render_config_from_mapping({"robust_appearance": section}, "/tmp")
    assert render_config_from_mapping({"robust_appearance": None}, "/tmp").robust_appearance is None


@pytest.mark.parametrize(
    "settings",
    [
        {"end_view_probability": 1.5},
        {"end_view_probability": float("nan")},
        {"preset": "bad"},
        {"axis": None},
        {"axis": "w"},
        {"end_view_prob": 0.9},
    ],
)
def test_invalid_appearance_configuration_rejected(settings):
    with pytest.raises(ValueError, match="robust_appearance"):
        validate_robust_appearance({"enabled": True, "axis": "y", **settings})


@pytest.mark.parametrize(
    "raw", [None, {}, {"enabled": False}, {"enabled": False, "preset": "glare"}]
)
def test_absent_or_disabled_appearance_is_none(raw):
    assert validate_robust_appearance(raw) is None


@pytest.mark.parametrize("raw", ["yes", [1], {"enabled": False, "typo": 1}])
def test_malformed_appearance_section_rejected_even_when_disabled(raw):
    with pytest.raises(ValueError, match="robust_appearance"):
        validate_robust_appearance(raw)


def test_implicit_appearance_defaults_normalize_like_explicit_defaults():
    implicit = validate_robust_appearance({"enabled": True, "axis": "y"})
    explicit = validate_robust_appearance(
        {"enabled": True, "axis": "y", "preset": "mixed", "end_view_probability": 0.45}
    )
    assert implicit == explicit


def mesh_and_intrinsics(points=BOX):
    mesh = S.MeshInfo.from_points(points, np.zeros((1, 3)))
    return mesh, G.intrinsics_from_lens(640, 640, 50, 36)


def sample(options=None, index=0):
    raw = {"output": {"render_masks": True}}
    if options is not None:
        raw["robustness"] = options
    cfg = render_config_from_mapping(raw, BASE_DIR)
    mesh, intr = mesh_and_intrinsics()
    record = S.sample_scene(
        cfg, index, 71, S.AssetCatalog(BASE_DIR), mesh, intr, S.camera_distance_range(cfg, mesh)
    )
    return record, mesh, intr


def test_disabled_robustness_keeps_baseline_records_exact():
    base, _, _ = sample()
    easy, _, _ = sample({"mode_weights": {"easy": 1}})
    assert easy.pop("robustness")["mode"] == "easy"
    assert easy == base


@pytest.mark.parametrize("edge", ["left", "right", "top", "bottom", "top_left", "bottom_right"])
def test_requested_clipping_is_realized_in_camera_geometry(edge):
    options = {
        "mode_weights": {"clipping": 1},
        "clipping_edges": [edge],
        "clipping_fraction": [0.3, 0.3],
    }
    record, mesh, intr = sample(options)
    pose = np.array(record["object"]["pose"])
    camera = np.array(record["camera"]["cam_to_world"])
    uv, depths = G.project_points(G.transform_points(pose, mesh.local_corners), camera, intr)
    lo, hi = uv.min(0), uv.max(0)
    assert (depths > 0).all()
    for direction, axis, boundary in [
        ("left", 0, 0),
        ("right", 0, 640),
        ("top", 1, 0),
        ("bottom", 1, 640),
    ]:
        if direction in edge:
            span = hi[axis] - lo[axis]
            outside = (
                (boundary - lo[axis]) / span if boundary == 0 else (hi[axis] - boundary) / span
            )
            assert outside == pytest.approx(0.3, abs=1e-7)
    expected = 1 - 0.7 ** (2 if "_" in edge else 1)
    assert record["robustness"]["clipping"]["projected_bbox_outside_fraction"] == pytest.approx(
        expected
    )


def test_proxies_are_in_front_and_deterministic_without_perturbing_other_fields():
    options = {"mode_weights": {"occlusion": 1}, "proxy_count": [2, 2]}
    record, mesh, intr = sample(options)
    again, _, _ = sample(options)
    assert record == again
    base, _, _ = sample()
    robust = record.pop("robustness")
    assert record == base
    assert robust["mode"] == "occlusion"
    assert robust["occluder_kind"] == "capsule_grasp_proxy"
    assert len(robust["proxies"]) == 2
    camera = np.array(record["camera"]["cam_to_world"])
    pose = np.array(record["object"]["pose"])
    _, target_depths = G.project_points(G.transform_points(pose, mesh.local_corners), camera, intr)
    for proxy in robust["proxies"]:
        _, depths = G.project_points(np.array([proxy["start"], proxy["end"]]), camera, intr)
        assert depths.max() + proxy["radius"] < target_depths.min()
        assert proxy["radius"] > 0


def test_sample_scene_rejects_invalid_robustness_options():
    with pytest.raises(ValueError, match="robustness"):
        sample({"mode_weights": {"banana": 1}})


def test_opt_in_identity_does_not_change_legacy_identity():
    default = config_values(render_config_from_mapping({}, BASE_DIR))
    assert "robustness" not in default
    raw = {"output": {"render_masks": True}, "robustness": {}}
    robust = config_values(render_config_from_mapping(raw, BASE_DIR))
    assert robust["robustness"]["mode_weights"]["easy"] == 0.4


def test_config_values_drop_the_machine_specific_base_dir():
    assert "base_dir" not in config_values(render_config_from_mapping({}, BASE_DIR))
    assert BASE_DIR not in str(normalized_scene_config(render_config_from_mapping({}, BASE_DIR)))


def test_appearance_identity_tracks_enabled_options_only():
    def identity(section):
        return normalized_scene_config(
            render_config_from_mapping({"robust_appearance": section}, BASE_DIR)
        )

    baseline = normalized_scene_config(render_config_from_mapping({}, BASE_DIR))
    assert baseline == identity({"enabled": False})
    mixed = identity({"enabled": True, "axis": "y", "preset": "mixed"})
    glare = identity({"enabled": True, "axis": "y", "preset": "glare"})
    assert mixed != glare and mixed != baseline
    assert identity({"enabled": True, "axis": "y"}) == mixed, "written-out defaults change nothing"
    assert "robust_appearance" not in config_values(render_config_from_mapping({}, BASE_DIR))


def end_face_scene(edge, index):
    raw = {
        "output": {"render_masks": True},
        "robust_appearance": {
            "enabled": True,
            "axis": "y",
            "end_view_probability": 1,
            "preset": "glare",
        },
        "robustness": {
            "mode_weights": {"clipping": 1},
            "clipping_edges": [edge],
            "clipping_fraction": [0.35, 0.35],
        },
    }
    cfg = render_config_from_mapping(raw, BASE_DIR)
    mesh, intr = mesh_and_intrinsics(THIN_BOX)
    distance = S.camera_distance_range(cfg, mesh)
    return S.sample_scene(cfg, index, 71, S.AssetCatalog(BASE_DIR), mesh, intr, distance)


def test_top_clipping_keeps_camera_above_enabled_ground():
    record = end_face_scene("top", 4)
    mesh, _ = mesh_and_intrinsics(THIN_BOX)
    camera = np.asarray(record["camera"]["cam_to_world"])
    assert record["ground"]["enabled"]
    assert camera[2, 3] > record["object"]["bbox_min_z"] - 0.001 * mesh.scene_scale
    clipping = record["robustness"]["clipping"]
    assert clipping["camera_roll_fallback_deg"] == 180
    assert clipping["edge"] == "top"
    lo, hi = np.asarray(clipping["projected_bbox_xyxy"]).reshape(2, 2)
    assert -lo[1] / (hi[1] - lo[1]) == pytest.approx(0.35, abs=1e-7)


@pytest.mark.parametrize("edge", ["left", "top_right"])
def test_glare_light_is_aimed_from_the_final_clipped_camera(edge):
    record = end_face_scene(edge, 3)
    appearance = record["appearance"]
    camera = np.asarray(record["camera"]["cam_to_world"])
    view = camera[:3, 3] - np.asarray(record["object"]["center"])
    view /= np.linalg.norm(view)
    # Glare jitter |j| <= sqrt(2*.25**2 + .2**2) bounds alignment by sqrt(1-|j|**2) ~= .914.
    # Aiming from the pre-clip camera gives ~.80 for top_right.
    assert np.dot(appearance["sun_direction"], view) > 0.91
    axis = np.asarray(record["object"]["pose"])[:3, 1]
    axis = axis if axis[2] >= 0 else -axis
    measured = np.degrees(np.arccos(min(1, abs(view @ axis) / np.linalg.norm(axis))))
    assert appearance["final_axis_angle_deg"] == pytest.approx(measured)
    assert appearance["end_face_view"]
