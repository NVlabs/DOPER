# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paired box/mask and box/mask/keypoint GT pools from a staged GT view."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

from doper.bop import gt_labels, onboarding_hot3d
from doper.bop.onboarding import GT_SOURCE_KIND

WIDTH, HEIGHT = 64, 48
K_MATRIX = [60.0, 0.0, 32.0, 0.0, 60.0, 24.0, 0.0, 0.0, 1.0]
SEQUENCE = "obj_000001_up"
OBJECT = 1
KEYPOINT_COUNT = 64
EVALUATION = "e" * 64
BASE_EPOCHS = 25
CENTRAL = (0.0, 0.0, 500.0)
FAR_SIDE = (4000.0, 0.0, 500.0)
GOOD_MASK = (20, 14, 24, 18)
SLIVER_MASK = (20, 14, 24, 1)


@dataclass
class Fixture:
    gt_view: Path
    keypoints: Path
    base: Path
    out: Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def keypoints_m() -> np.ndarray:
    return np.random.default_rng(5).uniform(-0.1, 0.1, (KEYPOINT_COUNT, 3))


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def mask_image(rectangle: tuple[int, int, int, int] | None) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    if rectangle is not None:
        x, y, w, h = rectangle
        mask[y : y + h, x : x + w] = 255
    return mask


def build_gt_view(
    root: Path,
    frames: dict[int, tuple[tuple[float, float, float], tuple[int, int, int, int] | None]],
    *,
    mask_dir: str = "mask_visib",
    extra_ready: dict | None = None,
) -> Path:
    """GT view as ``stage_gt_view`` stages it: ``rgb`` links to the static view, masks in
    ``mask_dir``."""
    static = root / "static" / SEQUENCE / "rgb"
    scene = root / "gt" / SEQUENCE
    rows, gt, cameras = [], {}, {}
    rng = np.random.default_rng(11)
    for frame, (translation, rectangle) in sorted(frames.items()):
        write_png(static / f"{frame:06d}.png", rng.integers(1, 255, (HEIGHT, WIDTH, 3), np.uint8))
        name = f"{frame:06d}_000000.png"
        write_png(scene / mask_dir / name, mask_image(rectangle))
        gt[str(frame)] = [
            {"obj_id": OBJECT, "cam_R_m2c": [1, 0, 0, 0, 1, 0, 0, 0, 1], "cam_t_m2c": translation}
        ]
        cameras[str(frame)] = {"cam_K": K_MATRIX, "width": WIDTH, "height": HEIGHT}
        rows.append(
            {
                "object_id": OBJECT,
                "sequence": SEQUENCE,
                "frame": frame,
                "rgb": f"{SEQUENCE}/rgb/{frame:06d}.png",
                "rgb_sha256": sha256(static / f"{frame:06d}.png"),
                "mask_dir": mask_dir,
                "mask": f"{SEQUENCE}/{mask_dir}/{name}",
                "mask_sha256": sha256(scene / mask_dir / name),
            }
        )
    (scene / "scene_gt.json").write_text(json.dumps(gt))
    (scene / "scene_camera.json").write_text(json.dumps(cameras))
    (scene / "rgb").symlink_to(Path("..") / ".." / "static" / SEQUENCE / "rgb")
    ready = {
        "identity": {"dataset": "handal"},
        "source_kind": GT_SOURCE_KIND,
        "evaluation_manifest_sha256": EVALUATION,
        "gt_supervision_staged": True,
        "pose_annotated_only": True,
        "metadata": {
            SEQUENCE: {
                "staged_gt_sha256": sha256(scene / "scene_gt.json"),
                "staged_camera_sha256": sha256(scene / "scene_camera.json"),
            }
        },
        "frames": rows,
        **(extra_ready or {}),
    }
    (root / "gt" / "READY.json").write_text(json.dumps(ready))
    return root / "gt"


def build_base(root: Path, *, epochs: int = BASE_EPOCHS, checkpoint_sha: str | None = None) -> Path:
    base = root / "runs" / "base" / "obj_000001"
    (base / "work").mkdir(parents=True)
    (base / "geometry").mkdir()
    (base / "work" / "last.pth").write_bytes(b"epoch 25 checkpoint")
    keypoints = base / "geometry" / "keypoints_3d.json"
    keypoints.write_text(
        json.dumps(
            {
                "keypoint_names": [f"k{i}" for i in range(KEYPOINT_COUNT)],
                "num_keypoints": KEYPOINT_COUNT,
                "keypoints_3d": keypoints_m().tolist(),
            }
        )
    )
    (base / "completed.json").write_text(
        json.dumps(
            {
                "stage": "base",
                "obj_id": OBJECT,
                "epochs": epochs,
                "checkpoint_sha256": checkpoint_sha or sha256(base / "work" / "last.pth"),
                "keypoints_sha256": sha256(keypoints),
            }
        )
    )
    return base


DEFAULT_FRAMES = {
    0: (CENTRAL, GOOD_MASK),
    1: ((10.0, 5.0, 520.0), GOOD_MASK),
    2: ((-8.0, 3.0, 480.0), (18, 12, 26, 20)),
}


def make_fixture(root: Path, frames=None, **kwargs) -> Fixture:
    base = build_base(root)
    view = build_gt_view(root, frames or DEFAULT_FRAMES, **kwargs)
    return Fixture(view, base / "geometry" / "keypoints_3d.json", base, root / "pool")


def export(fx: Fixture):
    return gt_labels.export_gt_pool(fx.gt_view, OBJECT, fx.keypoints, fx.base, fx.out)


def read(path: Path):
    return json.loads(path.read_text())


def annotations(fx: Fixture, arm: str) -> dict:
    return read(fx.out / arm / "train" / "annotations.json")


def test_arms_share_images_boxes_and_masks(tmp_path: Path) -> None:
    fx = make_fixture(tmp_path)
    summary = export(fx)
    assert summary.labels == 3 and summary.skipped == 0
    box, full = annotations(fx, "box_mask"), annotations(fx, "box_mask_keypoints")
    assert box["images"] == full["images"] and len(box["images"]) == 3
    for plain, rich in zip(box["annotations"], full["annotations"], strict=True):
        assert {k: rich[k] for k in plain} == plain
        assert not {"keypoints", "num_keypoints", "object_pose"} & set(plain)
        assert {"keypoints", "object_pose"} <= set(rich)
    for image in box["images"]:
        names = [Path("train/images") / image["file_name"]]
        names += [Path("train") / a["mask_file"] for a in box["annotations"]]
        for name in names:
            assert sha256(fx.out / "box_mask" / name) == sha256(
                fx.out / "box_mask_keypoints" / name
            )
    for arm in gt_labels.GT_ARMS:
        assert (fx.out / arm / "keypoints_3d.json").is_file()
    assert gt_labels.GT_ARMS == ("box_mask", "box_mask_keypoints")
    assert gt_labels.verify_gt_pool(fx.out) == []
    first = box["annotations"][0]
    assert first["bbox"] == [20.0, 14.0, 24.0, 18.0] and first["area"] == 24 * 18
    assert first["bop"] == {
        "scene": SEQUENCE,
        "frame": 0,
        "instance": 0,
        "source": "onboarding_static_gt",
    }


def test_keypoint_projection_round_trip_below_0_01_px(tmp_path: Path) -> None:
    fx = make_fixture(tmp_path)
    export(fx)
    receipt = read(fx.out / "POOL_READY.json")
    assert receipt["records"] and all(
        row["projection_roundtrip_max_px"] <= 0.01 for row in receipt["records"]
    )
    k = np.asarray(K_MATRIX).reshape(3, 3)
    for ann, row in zip(
        annotations(fx, "box_mask_keypoints")["annotations"], DEFAULT_FRAMES.items(), strict=True
    ):
        xyz = keypoints_m() + np.asarray(row[1][0]) / 1000.0
        uvh = xyz @ k.T
        uv = uvh[:, :2] / uvh[:, 2:]
        flat = np.asarray(ann["keypoints"]).reshape(-1, 3)
        assert np.abs(flat[:, :2] - uv).max() <= 0.005 + 1e-9
        in_frame = (uv[:, 0] >= 0) & (uv[:, 0] < WIDTH) & (uv[:, 1] >= 0) & (uv[:, 1] < HEIGHT)
        assert flat[:, 2].tolist() == in_frame.astype(int).tolist()
        assert ann["num_keypoints"] == int(in_frame.sum())


def test_out_of_frame_keypoints_get_visibility_zero(tmp_path: Path) -> None:
    frames = {0: ((250.0, 0.0, 500.0), GOOD_MASK)}
    fx = make_fixture(tmp_path, frames)
    export(fx)
    flat = np.asarray(annotations(fx, "box_mask_keypoints")["annotations"][0]["keypoints"])
    visibility = flat.reshape(-1, 3)[:, 2]
    assert 0 < visibility.sum() < KEYPOINT_COUNT and set(visibility) == {0.0, 1.0}


def test_degenerate_mask_skipped_with_reason(tmp_path: Path) -> None:
    frames = {
        0: (CENTRAL, GOOD_MASK),
        1: (CENTRAL, None),
        2: (CENTRAL, SLIVER_MASK),
        3: (FAR_SIDE, GOOD_MASK),
    }
    fx = make_fixture(tmp_path, frames)
    summary = export(fx)
    assert summary.labels == 1 and summary.skipped == 3
    assert dict(summary.skip_reasons) == {
        "empty_or_degenerate_mask": 2,
        "no_in_frame_gt_keypoints": 1,
    }
    skipped = read(fx.out / "POOL_READY.json")["skipped"]
    assert [(s["frame"], s["reason"]) for s in skipped] == [
        (1, "empty_or_degenerate_mask"),
        (2, "empty_or_degenerate_mask"),
        (3, "no_in_frame_gt_keypoints"),
    ]
    assert len(annotations(fx, "box_mask")["images"]) == 1


def test_all_frames_skipped_is_refused(tmp_path: Path) -> None:
    fx = make_fixture(tmp_path, {0: (CENTRAL, None)})
    with pytest.raises(ValueError, match="zero-label"):
        export(fx)
    assert not fx.out.exists()
    assert not list(tmp_path.glob("*.partial"))


def test_hot3d_mask_supplied_read(tmp_path: Path) -> None:
    ready = {
        "identity": {"dataset": "hot3d"},
        "mask_semantics": onboarding_hot3d.HOT3D_MASK_SEMANTICS,
    }
    fx = make_fixture(tmp_path, mask_dir="mask_supplied", extra_ready=ready)
    summary = export(fx)
    assert summary.labels == 3
    identity = read(fx.out / "identity.json")
    assert identity["mask_semantics"] == onboarding_hot3d.HOT3D_MASK_SEMANTICS
    assert "released static" in identity["gt_annotation_source"]
    rows = read(fx.out / "POOL_READY.json")["records"]
    assert {row["mask_dir"] for row in rows} == {"mask_supplied"}
    assert gt_labels.verify_gt_pool(fx.out) == []


def test_hot3d_requires_released_mask_semantics(tmp_path: Path) -> None:
    ready = {"identity": {"dataset": "hot3d"}, "mask_semantics": "something_else"}
    fx = make_fixture(tmp_path, mask_dir="mask_supplied", extra_ready=ready)
    with pytest.raises(ValueError, match="mask semantics"):
        export(fx)


def test_requires_epoch_25_base(tmp_path: Path) -> None:
    fx = make_fixture(tmp_path)
    base = fx.base / "completed.json"
    done = read(base)
    for patch in ({"epochs": 3}, {"obj_id": 2}, {"checkpoint_sha256": "0" * 64}):
        base.write_text(json.dumps({**done, **patch}))
        with pytest.raises(ValueError, match="epoch-25 base"):
            export(fx)
    base.unlink()
    with pytest.raises(ValueError, match="epoch-25 base"):
        export(fx)
    assert not fx.out.exists()


def test_keypoints_must_be_the_base_geometry(tmp_path: Path) -> None:
    fx = make_fixture(tmp_path)
    other = tmp_path / "other.json"
    payload = read(fx.keypoints)
    payload["keypoints_3d"][0][0] += 0.001
    other.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="base"):
        gt_labels.export_gt_pool(fx.gt_view, OBJECT, other, fx.base, fx.out)


def test_requires_supervised_gt_view(tmp_path: Path) -> None:
    fx = make_fixture(tmp_path)
    ready = read(fx.gt_view / "READY.json")
    ready["gt_supervision_staged"] = False
    (fx.gt_view / "READY.json").write_text(json.dumps(ready))
    with pytest.raises(ValueError, match="supervised"):
        export(fx)


def test_changed_source_bytes_are_refused(tmp_path: Path) -> None:
    fx = make_fixture(tmp_path)
    mask = fx.gt_view / SEQUENCE / "mask_visib" / "000001_000000.png"
    write_png(mask, mask_image((1, 1, 5, 5)))
    with pytest.raises(ValueError, match="changed"):
        export(fx)


def test_rerun_with_same_identity_returns_the_pool(tmp_path: Path) -> None:
    fx = make_fixture(tmp_path)
    first = export(fx)
    marker = (fx.out / "POOL_READY.json").stat().st_mtime_ns
    second = export(fx)
    assert second.labels == first.labels
    assert (fx.out / "POOL_READY.json").stat().st_mtime_ns == marker
    receipt = read(fx.gt_view / "READY.json")
    receipt["evaluation_manifest_sha256"] = "f" * 64
    (fx.gt_view / "READY.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="identity changed"):
        export(fx)


def test_pool_is_relocatable(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    fx = make_fixture(tree)
    export(fx)
    moved = tmp_path / "moved"
    tree.rename(moved)
    assert gt_labels.verify_gt_pool(moved / "pool") == []
    receipt_path = moved / "pool" / "POOL_READY.json"
    receipt = read(receipt_path)
    assert all(not Path(row["rgb_source"]).is_absolute() for row in receipt["records"])
    assert str(tmp_path) not in receipt_path.read_text()
    rerun = gt_labels.export_gt_pool(
        moved / "gt",
        OBJECT,
        moved / "runs/base/obj_000001/geometry/keypoints_3d.json",
        moved / "runs/base/obj_000001",
        moved / "pool",
    )
    assert rerun.labels == 3


def test_verify_reports_tampering(tmp_path: Path) -> None:
    fx = make_fixture(tmp_path)
    export(fx)
    mask = sorted((fx.out / "box_mask_keypoints" / "train" / "masks").glob("*.png"))[0]
    write_png(mask, mask_image((0, 0, 6, 6)))
    faults = gt_labels.verify_gt_pool(fx.out)
    assert any("mask" in fault for fault in faults)
    path = fx.out / "box_mask" / "train" / "annotations.json"
    coco = read(path)
    coco["annotations"][0]["keypoints"] = [0.0] * 3
    path.write_text(json.dumps(coco))
    assert any("changed" in fault for fault in gt_labels.verify_gt_pool(fx.out))


def test_a_second_export_waits_for_the_pool_lock(tmp_path: Path) -> None:
    from lock_helpers import assert_waits_for

    from doper.bop import sources

    fx = make_fixture(tmp_path)
    stage = fx.out.with_name(fx.out.name + gt_labels.PARTIAL_SUFFIX)

    def waiting() -> None:
        assert not stage.exists() and not fx.out.exists()

    summary = assert_waits_for(sources.sibling_lock(fx.out), lambda: export(fx), waiting)
    assert summary.labels == 3 and export(fx).labels == 3
