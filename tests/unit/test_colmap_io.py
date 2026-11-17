# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""COLMAP 3.13 text models: read, write, seeds and metrics."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from doper.nurec.capture import PinholeCamera
from doper.reconstruction.colmap_io import (
    ColmapCamera,
    ColmapFrame,
    ColmapImage,
    ColmapModel,
    ColmapPoint3D,
    ColmapRig,
    SeedImage,
    cam_from_world,
    model_metrics,
    quaternion_wxyz,
    read_text_model,
    write_seed,
    write_text_model,
)

CAMERA = ColmapCamera(1, "OPENCV", 1280, 720, (914.164306640625, 914.164306640625, 627.5, 394.5,
                                              0.0, 0.0, 0.0, 0.0))  # fmt: skip
# A negative-w quaternion, as COLMAP writes them (the sign is kept on read).
QVEC = (-0.154357329547223, -0.30790042538274182, 0.85876569373267919, -0.37933181534377886)


def _model() -> ColmapModel:
    images = {
        1: ColmapImage(1, QVEC, (0.44585113124581799, 0.39677714161461786, 0.2187211089027342),
                       1, "frame-000001.png",
                       ((525.92401123046875, 23.234872817993164, -1), (633.8, 172.06, 7))),
        4: ColmapImage(4, (0.5, 0.5, 0.5, 0.5), (0.1, -0.2, 0.3), 1, "frame-000000.png",
                       ((10.5, 20.25, 7),)),
    }  # fmt: skip
    points = {7: ColmapPoint3D(7, (0.56878006544283388, -1.3917871989758468, -0.2492476654),
                               (96, 121, 144), 1.1821045777918642, ((1, 1), (4, 0)))}  # fmt: skip
    rigs = {1: ColmapRig(1, ("CAMERA", 1), ())}
    frames = {
        1: ColmapFrame(1, 1, QVEC, images[1].tvec, (("CAMERA", 1, 1),)),
        2: ColmapFrame(2, 1, images[4].qvec_wxyz, images[4].tvec, (("CAMERA", 1, 4),)),
    }
    return ColmapModel({1: CAMERA}, images, points, rigs, frames)


def test_images_txt_round_trip_wxyz(tmp_path):
    model = _model()
    write_text_model(model, tmp_path)
    lines = [line for line in (tmp_path / "images.txt").read_text().splitlines() if line[:1] != "#"]
    first = lines[0].split()
    # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME: w first, sign as given.
    assert first[0] == "1" and float(first[1]) == QVEC[0] and first[9] == "frame-000001.png"
    assert [float(v) for v in lines[1].split()] == [525.92401123046875, 23.234872817993164, -1,
                                                    633.8, 172.06, 7]  # fmt: skip
    again = read_text_model(tmp_path)
    assert again == model
    assert again.images[1].qvec_wxyz == QVEC
    # cam_from_world: R from wxyz (scipy takes xyzw), t as stored.
    w, x, y, z = QVEC
    expected = np.eye(4)
    expected[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    expected[:3, 3] = model.images[1].tvec
    np.testing.assert_allclose(cam_from_world(again.images[1]), expected, atol=1e-15)
    # q and -q are the same rotation.
    flipped = ColmapImage(1, tuple(-q for q in QVEC), model.images[1].tvec, 1, "x.png", ())
    np.testing.assert_allclose(cam_from_world(flipped), expected, atol=1e-15)


def test_header_counts_and_legacy_model(tmp_path):
    model = _model()
    write_text_model(model, tmp_path)
    images = (tmp_path / "images.txt").read_text().splitlines()
    assert images[3] == "# Number of images: 2, mean observations per image: 1.0"
    points = (tmp_path / "points3D.txt").read_text().splitlines()
    assert points[2] == "# Number of points: 1, mean track length: 2.0"
    assert (tmp_path / "rigs.txt").exists() and (tmp_path / "frames.txt").exists()
    legacy = ColmapModel(model.cameras, model.images, model.points3d)
    out = tmp_path / "legacy"
    write_text_model(legacy, out)
    assert not (out / "rigs.txt").exists() and not (out / "frames.txt").exists()
    assert read_text_model(out) == legacy


def test_rig_with_posed_sensor_round_trips(tmp_path):
    pose = ((1.0, 0.0, 0.0, 0.0), (0.1, 0.0, 0.0))
    rig = ColmapRig(1, ("CAMERA", 1), (("CAMERA", 2, pose), ("CAMERA", 3, None)))
    model = ColmapModel({1: CAMERA}, {}, {}, {1: rig}, {})
    write_text_model(model, tmp_path)
    assert (tmp_path / "rigs.txt").read_text().splitlines()[-1] == (
        "1 3 CAMERA 1 CAMERA 2 1 1.0 0.0 0.0 0.0 0.1 0.0 0.0 CAMERA 3 0"
    )
    assert read_text_model(tmp_path) == model


def test_seed_has_empty_points(tmp_path):
    camera = PinholeCamera(1280, 720, 914.164306640625, 914.164306640625, 627.49, 394.47, True)
    rng = np.random.default_rng(3)
    poses = []
    for _ in range(3):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.random(random_state=rng).as_matrix()
        pose[:3, 3] = rng.normal(size=3)
        poses.append(pose)
    seeds = [SeedImage(image_id, f"frame-{i:06d}.png", pose)
             for i, (image_id, pose) in enumerate(zip((5, 1, 3), poses, strict=True))]  # fmt: skip
    write_seed(tmp_path, camera, seeds)
    assert (tmp_path / "points3D.txt").read_bytes() == b""
    assert not (tmp_path / "rigs.txt").exists()
    data = [line for line in (tmp_path / "images.txt").read_text().split("\n") if line[:1] != "#"]
    # An image line, then an empty 2D-points line, per image.
    assert [bool(line) for line in data] == [True, False] * 3 + [False]
    assert (tmp_path / "cameras.txt").read_text().splitlines()[-1] == (
        "1 OPENCV 1280 720 914.164306640625 914.164306640625 627.49 394.47 0 0 0 0"
    )
    model = read_text_model(tmp_path)
    assert list(model.images) == [5, 1, 3] and model.points3d == {}
    for seed in seeds:
        image = model.images[seed.image_id]
        assert image.name == seed.name and image.points2d == () and image.camera_id == 1
        np.testing.assert_allclose(cam_from_world(image), seed.cam_from_world, atol=1e-12)


def test_quaternion_wxyz_matches_rotation():
    rotation = Rotation.random(random_state=np.random.default_rng(5)).as_matrix()
    w, x, y, z = quaternion_wxyz(rotation)
    np.testing.assert_allclose(Rotation.from_quat([x, y, z, w]).as_matrix(), rotation, atol=1e-12)
    with pytest.raises(ValueError, match="rotation"):
        quaternion_wxyz(np.diag([1.0, 1.0, -1.0]))


def test_model_metrics_match_the_paper_pipeline_definitions():
    metrics = model_metrics(_model(), total_images=4)
    assert metrics == {
        "total_images": 4, "registered_images": 2, "largest_model_fraction": 0.5,
        "observations": 2, "points": 1, "mean_track_length": 2.0,
        "median_reprojection_px": 1.1821045777918642,
    }  # fmt: skip


@pytest.mark.parametrize(
    ("name", "text", "match"),
    [
        ("cameras.txt", "1 OPENCV 1280\n", "cameras.txt:1"),
        ("images.txt", "1 1 0 0 0 0 0 0 1\n\n", "images.txt:1"),
        ("images.txt", "1 1 0 0 0 0 0 0 1 a.png\n1 2\n", "images.txt:2"),
        ("points3D.txt", "1 0 0 0 1 2 3 0.5 1\n", "points3D.txt:1"),
    ],
)
def test_malformed_files_are_refused(tmp_path, name, text, match):
    write_text_model(ColmapModel({1: CAMERA}, {}, {}), tmp_path)
    (tmp_path / name).write_text(text)
    with pytest.raises(ValueError, match=match):
        read_text_model(tmp_path)


def test_duplicate_ids_are_refused(tmp_path):
    write_text_model(ColmapModel({1: CAMERA}, {}, {}), tmp_path)
    line = "1 OPENCV 1280 720 1 1 1 1 0 0 0 0\n"
    (tmp_path / "cameras.txt").write_text(line + line)
    with pytest.raises(ValueError, match="twice"):
        read_text_model(tmp_path)
