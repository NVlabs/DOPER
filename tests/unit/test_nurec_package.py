# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualification, packaging and QC of the NuRec adaptation dataset (``doper.nurec.package``)."""

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from doper.nurec.labels import ViewLabel
from doper.nurec.package import (
    DatasetQcError,
    assess,
    assess_files,
    package_dataset,
    qc_dataset,
    qualify,
)
from doper.recipes.nurec_schema import HsvExclusion

WIDTH, HEIGHT = 80, 60
K = np.array([[60.0, 0.0, 40.0], [0.0, 60.0, 30.0], [0.0, 0.0, 1.0]])
BLUE = HsvExclusion(h_above=85, h_below=120, s_above=90, v_above=100)
BLUE_BGR = (200, 60, 20)
POINTS = np.random.default_rng(5).uniform(-0.05, 0.05, (16, 3))
MIN_VISIBLE = 8
PAD = 3


def _project(pose: np.ndarray) -> np.ndarray:
    p = (POINTS @ pose[:3, :3].T + pose[:3, 3]) @ K.T
    return p[:, :2] / p[:, 2:]


def _pose(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(rng.uniform(-0.3, 0.3, 3))[0]
    pose[:3, 3] = [rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02), 0.35]
    return pose


def _rect(xy: np.ndarray) -> tuple[int, int, int, int]:
    x0, y0 = np.floor(xy.min(axis=0)).astype(int) - PAD
    x1, y1 = np.ceil(xy.max(axis=0)).astype(int) + PAD
    return x0, y0, x1, y1


def _view(index: int, lineage: str = "session-a", hidden: int = 2, seed_views: int = 2):
    """A label whose CAD silhouette is the padded keypoint rectangle (as an array)."""
    pose = _pose(index)
    xy = _project(pose)
    visible = np.ones(len(POINTS), bool)
    visible[:hidden] = False
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    x0, y0, x1, y1 = _rect(xy)
    mask[max(0, y0) : y1 + 1, max(0, x0) : x1 + 1] = 255
    T_camera_nre = np.eye(4)  # noqa: N806
    T_camera_nre[0, 3] = index  # unique per view
    return ViewLabel(
        index=index, seed_view=f"{index % seed_views:06d}.png", reconstruction_id="recon",
        lineage=lineage, T_camera_nre=T_camera_nre, T_camera_object_m=pose, keypoints_xy=xy,
        ray_visible=visible,
        cad_bbox=(float(xy[:, 0].min()), float(xy[:, 1].min()), float(np.ptp(xy[:, 0])),
                  float(np.ptp(xy[:, 1]))),
        K=K.copy(), image_size=(WIDTH, HEIGHT), cad_mask=mask,
    )  # fmt: skip


def _grey(index: int = 0) -> np.ndarray:
    image = np.full((HEIGHT, WIDTH, 3), 90, np.uint8)
    image[0, 0] = index  # every render differs
    return image


def _write_views(tmp_path: Path, labels, *, sam: bool = True):
    """Write each view's render, CAD mask and SAM mask (= the CAD mask); assess from files."""
    out = []
    for label in labels:
        stem = f"{label.index:06d}.png"
        paths = {kind: tmp_path / kind / stem for kind in ("rgb", "cad", "sam")}
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(paths["rgb"]), _grey(label.index))
        cv2.imwrite(str(paths["cad"]), label.cad_mask)
        cv2.imwrite(str(paths["sam"]), label.cad_mask)
        stored = replace(label, cad_mask=None, mask_file=stem)
        out.append(assess_files(stored, image=paths["rgb"], cad_mask=paths["cad"],
                                mask=paths["sam"] if sam else None, exclude_hsv=BLUE,
                                min_visible=MIN_VISIBLE))  # fmt: skip
    return out


def _keypoints_file(tmp_path: Path) -> Path:
    path = tmp_path / "keypoints_3d.json"
    names = [f"KP_object_{i:03d}" for i in range(len(POINTS))]
    path.write_text(json.dumps({"object": "object", "keypoint_names": names,
                                "keypoints_3d": POINTS.tolist()}))  # fmt: skip
    return path


def _package(tmp_path: Path, qualified, **kwargs) -> Path:
    options = {
        "keypoints_file": _keypoints_file(tmp_path),
        "seed": 817,
        "train": 8,
        "val": 2,
        "single_reconstruction_split": True,
        "min_visible": MIN_VISIBLE,
        **kwargs,
    }
    return package_dataset(qualified, tmp_path / "dataset", **options)


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- qualification -------------------------------------------------------------------------


def test_qualify_rules():
    label = _view(3, hidden=0)
    full, image = label.cad_mask, _grey()
    rules = {"min_visible": 12, "min_coverage": 0.5, "exclude_hsv": BLUE}
    # 12 visible keypoints qualify, 11 do not.
    twelve = replace(label, ray_visible=np.arange(16) < 12)
    assert qualify(twelve, full, image, **rules) is not None
    eleven = replace(label, ray_visible=np.arange(16) < 11)
    assert qualify(eleven, full, image, **rules) is None
    assert "visible" in assess(eleven, full, image, **rules).reason
    # Coverage |mask and CAD| / |CAD|: exactly half qualifies, less does not.
    big = replace(label, ray_visible=np.ones(16, bool))
    square = np.zeros_like(full)
    square[10:30, 10:30] = 255
    half = np.zeros_like(full)
    half[10:30, 10:20] = 255
    half[0:5, 0:5] = 255  # outside the CAD silhouette: not coverage
    at_half = assess(replace(big, cad_mask=square), half, image, **{**rules, "min_visible": 0})
    assert at_half.coverage == 0.5 and at_half.accepted
    less = half.copy()
    less[10, 10] = 0
    low = assess(replace(big, cad_mask=square), less, image, **{**rules, "min_visible": 0})
    assert low.coverage < 0.5 and not low.accepted and "coverage" in low.reason
    # A keypoint on the excluded color, or outside the eroded mask, is not visible.
    x, y = np.round(label.keypoints_xy[5]).astype(int)
    blue = image.copy()
    blue[y, x] = BLUE_BGR
    assert not assess(big, full, blue, **rules).visibility[5]
    assert assess(big, full, blue, **{**rules, "exclude_hsv": None}).visibility[5]
    edge = np.zeros_like(full)
    edge[: y + 1, :] = 255  # keypoint 5 on the mask's last row: eroded away (5x5 kernel)
    assert not assess(big, edge, image, **rules).visibility[5]
    assert assess(big, edge, image, **{**rules, "erode_px": 0}).visibility[5]
    # cad_only: no mask, so no coverage check; ray visibility and color only.
    cad_only = assess(big, None, blue, **rules)
    assert cad_only.coverage is None and cad_only.accepted
    assert not cad_only.visibility[5] and cad_only.visible_count == 15


# --- packaging -----------------------------------------------------------------------------


ANNOTATION_KEYS = {"id", "image_id", "category_id", "iscrowd", "bbox", "area", "keypoints",
                   "num_keypoints", "mask_file", "camera_intrinsics", "camera_extrinsics",
                   "T_camera_object_m", "source_render_index"}  # fmt: skip


def test_package_split_and_keys(tmp_path):
    labels = [_view(i) for i in range(12)]
    labels[4] = replace(labels[4], ray_visible=np.arange(16) < MIN_VISIBLE - 1)  # rejected
    qualified = _write_views(tmp_path, labels)
    out = _package(tmp_path, qualified)
    accepted = [q for q in qualified if q.accepted]
    np.random.default_rng(817).shuffle(accepted)
    expected = {"train": accepted[:8], "val": accepted[8:10]}
    for split, chosen in expected.items():
        data = json.loads((out / split / "annotations.json").read_text())
        names = [f"nurec_{q.label.index:06d}.png" for q in chosen]
        assert [im["file_name"] for im in data["images"]] == names
        assert data["categories"] == [
            {
                "id": 1,
                "name": "object",
                "keypoints": [f"KP_object_{i:03d}" for i in range(16)],
                "skeleton": [],
            }
        ]
        for q, image, ann in zip(chosen, data["images"], data["annotations"], strict=True):
            assert set(ann) == ANNOTATION_KEYS and ann["image_id"] == image["id"]
            assert (image["width"], image["height"]) == (WIDTH, HEIGHT)
            assert ann["source_render_index"] == q.label.index
            assert ann["camera_extrinsics"] == ann["T_camera_object_m"]
            flat = np.array(ann["keypoints"]).reshape(16, 3)
            assert set(flat[:, 2]) <= {0, 2} and ann["num_keypoints"] == (flat[:, 2] > 0).sum()
            np.testing.assert_array_equal(flat[~q.visibility, :2], 0)
            np.testing.assert_array_equal(
                flat[q.visibility, :2], q.label.keypoints_xy[q.visibility]
            )
            x, y, w, h = ann["bbox"]
            assert x >= 0 and y >= 0 and x + w <= WIDTH and y + h <= HEIGHT
            mask = cv2.imread(str(out / split / ann["mask_file"]), cv2.IMREAD_GRAYSCALE)
            assert ann["area"] == int((mask > 0).sum())
            image_file = out / split / "images" / image["file_name"]
            assert image_file.read_bytes() == q.image.read_bytes()
            assert os.stat(image_file).st_ino == os.stat(q.image).st_ino  # hard link
    selection = json.loads((out / "selection_manifest.json").read_text())
    assert selection["total_candidates"] == 12 and selection["accepted_candidates"] == 11
    assert [r["index"] for r in selection["rejected"]] == [4]
    assert selection["unselected_valid"] == [q.label.index for q in accepted[10:]]
    receipt = json.loads((out / "package.json").read_text())
    assert receipt["lineage_split"] is False and "share" in receipt["lineage_caveat"]
    assert receipt["mask_mode"] == "sam2" and receipt["reference_path"] is True
    manifest = json.loads((out / "sha256.json").read_text())
    files = {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()}
    assert set(manifest) == files - {"sha256.json", "package.json"}
    assert receipt["keypoints_sha256"] == manifest["keypoints_3d.json"]
    qc = qc_dataset(out, receipt["keypoints_sha256"])
    assert qc["status"] == "passed" and qc["splits"] == {"train": 8, "val": 2}
    assert qc["max_projection_consistency_error_px"] < 1e-6


def test_package_requires_enough_views(tmp_path):
    qualified = _write_views(tmp_path, [_view(i) for i in range(6)])
    with pytest.raises(ValueError, match="accepted"):
        _package(tmp_path, qualified)


def test_lineage_groups_never_straddle_split(tmp_path):
    labels = [_view(i, lineage=f"session-{i % 4}") for i in range(20)]
    out = _package(tmp_path, _write_views(tmp_path, labels), val=4)
    manifest = json.loads((out / "pose_manifest.json").read_text())
    lineages = {split: {r["lineage"] for r in manifest if r["split"] == split}
                for split in ("train", "val")}  # fmt: skip
    assert lineages["train"] and lineages["val"]
    assert not lineages["train"] & lineages["val"]
    assert [sum(r["split"] == s for r in manifest) for s in ("train", "val")] == [8, 4]
    assert json.loads((out / "package.json").read_text())["lineage_split"] is True


def test_single_lineage_needs_flag(tmp_path):
    qualified = _write_views(tmp_path, [_view(i) for i in range(12)])
    with pytest.raises(ValueError, match="single_reconstruction_split"):
        _package(tmp_path, qualified, single_reconstruction_split=False)


def test_cad_only_dataset_has_no_mask_supervision(tmp_path):
    from test_speedplus_data import tiny_model_config

    from doper.config import DataConfig, DoperConfig
    from doper.dataset import DoperDataset

    qualified = _write_views(tmp_path, [_view(i) for i in range(12)], sam=False)
    out = _package(tmp_path, qualified, mask_mode="cad_only")
    assert not (out / "train" / "masks").exists()
    data = json.loads((out / "train" / "annotations.json").read_text())
    assert all("mask_file" not in ann for ann in data["annotations"])
    receipt = json.loads((out / "package.json").read_text())
    assert (receipt["mask_mode"], receipt["reference_path"]) == ("cad_only", False)
    assert qc_dataset(out, receipt["keypoints_sha256"])["status"] == "passed"
    # A cad_only dataset must carry no mask supervision, even a stray mask_file.
    stray = out.parent / "stray"
    shutil.copytree(out, stray)
    _first_ann("train", lambda ann, _: ann.update(mask_file="masks/x.png"))(stray)
    _restamp(stray)
    with pytest.raises(DatasetQcError, match="mask_file"):
        qc_dataset(stray, receipt["keypoints_sha256"])
    cfg = DoperConfig(data=DataConfig(root=str(out)),
                      model=tiny_model_config(num_keypoints=16, det_input_size=64, crop_size=32,
                                              keypoint_head="heatmap"))  # fmt: skip
    cfg.aug.enabled = False
    sample = DoperDataset(cfg, "train")[0]
    assert float(sample["mask_weight"]) == 0.0 and float(sample["has_kp"]) == 1.0


def test_packaged_dataset_loads(tmp_path):
    from test_speedplus_data import tiny_model_config

    from doper.config import DataConfig, DoperConfig
    from doper.dataset import DoperDataset

    out = _package(tmp_path, _write_views(tmp_path, [_view(i) for i in range(12)]))
    cfg = DoperConfig(
        data=DataConfig(root=str(out), keypoints_3d_json=str(out / "keypoints_3d.json")),
        model=tiny_model_config(
            num_keypoints=16, det_input_size=64, crop_size=32, keypoint_head="heatmap"
        ),
    )
    cfg.aug.enabled = False
    for split, count in (("train", 8), ("val", 2)):
        dataset = DoperDataset(cfg, split)
        assert len(dataset) == count
        sample = dataset[0]
        assert float(sample["mask_weight"]) == 1.0 and float(sample["has_kp"]) == 1.0


# --- QC ------------------------------------------------------------------------------------


def _edit(path: Path, change) -> None:
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def _first_ann(split: str, change):
    def edit(root: Path) -> None:
        _edit(root / split / "annotations.json", lambda d: change(d["annotations"][0], d))

    return edit


def _keypoints(change):
    def edit(ann, _):
        flat = np.array(ann["keypoints"], dtype=float).reshape(-1, 3)
        change(flat, ann)
        ann["keypoints"] = flat.ravel().tolist()

    return edit


def _set(flat, ann, row, values):
    flat[row] = values


def _fewer_visible(flat, ann):
    visible = np.nonzero(flat[:, 2] > 0)[0]
    flat[visible[MIN_VISIBLE - 1 :]] = 0
    ann["num_keypoints"] = int((flat[:, 2] > 0).sum())


def _visible_row(flat):
    return int(np.nonzero(flat[:, 2] > 0)[0][0])


def _hidden_row(flat):
    return int(np.nonzero(flat[:, 2] == 0)[0][0])


def _same_hash(root: Path) -> None:
    def change(manifest):
        names = sorted(k for k in manifest if k.startswith("train/images/"))
        manifest[names[1]] = manifest[names[0]]

    _edit(root / "sha256.json", change)


def _small_mask(root: Path) -> None:
    ann = json.loads((root / "train" / "annotations.json").read_text())["annotations"][0]
    cv2.imwrite(str(root / "train" / ann["mask_file"]), np.zeros((HEIGHT - 1, WIDTH), np.uint8))


def _changed_file(root: Path) -> None:
    image = sorted((root / "train" / "images").iterdir())[0]
    image.write_bytes(image.read_bytes() + b"\0")


VIOLATIONS = {
    "keypoints_sha256": None,
    "split_counts": lambda root: _edit(
        root / "val" / "annotations.json", lambda d: d["images"].pop()
    ),
    "keypoint_count": lambda root: _edit(
        root / "train" / "annotations.json", lambda d: d["categories"][0]["keypoints"].pop()
    ),
    "image_id": _first_ann("train", lambda ann, _: ann.update(image_id=99)),
    "finite": _first_ann(
        "train", _keypoints(lambda f, a: f.__setitem__((_visible_row(f), 0), np.nan))
    ),
    "visibility_values": _first_ann(
        "train", _keypoints(lambda f, a: f.__setitem__((_visible_row(f), 2), 1))
    ),
    "min_visible": _first_ann("train", _keypoints(_fewer_visible)),
    "invisible_zeroed": _first_ann(
        "train", _keypoints(lambda f, a: f.__setitem__((_hidden_row(f), 0), 3.0))
    ),
    "num_keypoints": _first_ann(
        "train", lambda ann, _: ann.update(num_keypoints=ann["num_keypoints"] + 1)
    ),
    "rotation": _first_ann(
        "train",
        lambda ann, _: ann.update(
            T_camera_object_m=(
                np.array(ann["T_camera_object_m"]) * [[1.01], [1.01], [1.01], [1.0]]
            ).tolist()
        ),
    ),
    "reprojection": _first_ann(
        "train",
        _keypoints(lambda f, a: f.__setitem__((_visible_row(f), 0), f[_visible_row(f), 0] + 0.01)),
    ),
    "bbox_positive": _first_ann("train", lambda ann, _: ann["bbox"].__setitem__(2, 0.0)),
    "mask_area": _first_ann("train", lambda ann, _: ann.update(area=ann["area"] + 1)),
    "mask_file": _first_ann("train", lambda ann, _: ann.pop("mask_file")),
    "mask_shape": _small_mask,
    "unique_names": lambda root: _edit(
        root / "val" / "annotations.json",
        lambda d: d["images"][1].update(file_name=d["images"][0]["file_name"]),
    ),
    "unique_image_hashes": _same_hash,
    "seed_neighbourhoods": lambda root: _edit(
        root / "package.json", lambda d: d["candidate_seed_views"].append("999999.png")
    ),
    "source_exclusion": lambda root: _edit(
        root / "pose_manifest.json", lambda rows: rows[0].update(source_exclusion_pass=False)
    ),
    "file_sha256": _changed_file,
}


@pytest.fixture(scope="module")
def packaged(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("qc")
    out = _package(tmp_path, _write_views(tmp_path, [_view(i) for i in range(12)]))
    return out, json.loads((out / "package.json").read_text())["keypoints_sha256"]


@pytest.mark.parametrize("check", sorted(VIOLATIONS))
def test_qc_catches_each_violation(packaged, tmp_path, check):
    source, sha = packaged
    root = tmp_path / "dataset"
    shutil.copytree(source, root)
    assert qc_dataset(root, sha)["status"] == "passed"
    if VIOLATIONS[check] is None:
        sha = "0" * 64
    else:
        VIOLATIONS[check](root)
    if check not in ("file_sha256", "unique_image_hashes"):
        # Edits other than the manifest's own target keep the file hashes consistent.
        _restamp(root)
    with pytest.raises(DatasetQcError, match=check) as error:
        qc_dataset(root, sha)
    assert error.value.check == check


def _restamp(root: Path) -> None:
    """Rewrite ``sha256.json`` for the edited files (a QC fixture of one violation at a time)."""
    manifest = json.loads((root / "sha256.json").read_text())
    for name in manifest:
        if (root / name).is_file():
            manifest[name] = _sha(root / name)
    (root / "sha256.json").write_text(json.dumps(manifest))


def test_qc_summary_says_how_the_source_exclusion_was_enforced(packaged, tmp_path):
    source, sha = packaged
    assert qc_dataset(source, sha)["source_exclusion"].startswith("enforced at sampling")
    root = tmp_path / "dataset"
    shutil.copytree(source, root)
    _edit(root / "pose_manifest.json", lambda rows: rows[0].update(source_exclusion_pass=True))
    _restamp(root)
    assert qc_dataset(root, sha)["source_exclusion"] == "checked per view"
