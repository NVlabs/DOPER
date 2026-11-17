# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral completeness rejects records with plausible aggregate counts."""

import copy
import json

import pytest

from rendering.dataset_contract import merge_annotations, validate_dataset_records


def dataset(tmp_path, *, stereo=False, masks=True):
    images, annotations = [], []
    (tmp_path / "images").mkdir(exist_ok=True)
    (tmp_path / "masks").mkdir(exist_ok=True)
    for sample in range(3, 5):
        for camera in range(2 if stereo else 1):
            image_id = sample * 2 + camera if stereo else sample
            filename = f"{image_id}.png"
            (tmp_path / "images" / filename).touch()
            record = dict(id=image_id, file_name=filename, width=64, height=64)
            ann = dict(
                id=image_id,
                image_id=image_id,
                keypoints=[1, 2, 2],
                num_keypoints=1,
                bbox=[1, 2, 3, 4],
                area=12,
            )
            if stereo:
                fields = dict(pair_id=sample, camera_index=camera, view=("L", "R")[camera])
                record.update(fields)
                ann.update(fields)
            if masks:
                ann["mask_file"] = f"masks/{filename}"
                (tmp_path / ann["mask_file"]).touch()
            images.append(record)
            annotations.append(ann)
    return dict(images=images, annotations=annotations)


def check(payload, tmp_path, *, stereo=False, masks=True):
    return validate_dataset_records(
        payload,
        tmp_path,
        start_idx=3,
        end_idx=5,
        images_per_sample=2 if stereo else 1,
        render_masks=masks,
        num_keypoints=1,
    )


@pytest.mark.parametrize("stereo", [False, True])
def test_valid_records(tmp_path, stereo):
    assert check(dataset(tmp_path, stereo=stereo), tmp_path, stereo=stereo) == []


def test_duplicate_ids_with_expected_counts(tmp_path):
    payload = dataset(tmp_path)
    payload["images"][1] = copy.deepcopy(payload["images"][0])
    payload["annotations"][1] = copy.deepcopy(payload["annotations"][0])
    errors = check(payload, tmp_path)
    assert any("duplicate image IDs" in error for error in errors)
    assert any("duplicate annotation IDs" in error for error in errors)
    assert any("expected sample range" in error for error in errors)


@pytest.mark.parametrize("damage", ["annotation", "rgb", "mask", "mask_field", "reference"])
def test_missing_artifacts_and_references(tmp_path, damage):
    payload = dataset(tmp_path)
    if damage == "annotation":
        payload["annotations"].pop()
    elif damage == "rgb":
        (tmp_path / "images" / "3.png").unlink()
    elif damage == "mask":
        (tmp_path / "masks" / "3.png").unlink()
    elif damage == "mask_field":
        del payload["annotations"][0]["mask_file"]
    else:
        payload["annotations"][0]["image_id"] = 99
    assert check(payload, tmp_path)


@pytest.mark.parametrize(
    "damage", ["missing", "excess", "pair", "camera", "view", "annotation_view"]
)
def test_stereo_integrity(tmp_path, damage):
    payload = dataset(tmp_path, stereo=True)
    if damage == "missing":
        payload["images"].pop()
        payload["annotations"].pop()
    elif damage == "excess":
        payload["images"].append(copy.deepcopy(payload["images"][0]))
    elif damage == "pair":
        payload["images"][0]["pair_id"] = 0
    elif damage == "camera":
        payload["images"][0]["camera_index"] = 1
    elif damage == "view":
        payload["images"][1]["view"] = "L"
    else:
        payload["annotations"][0]["view"] = "R"
    assert check(payload, tmp_path, stereo=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("keypoints", [1, 2]),
        ("keypoints", [1, float("nan"), 2]),
        ("keypoints", [1, 2, 3]),
        ("keypoints", [1, 2, 2, 3, 4, 2]),
        ("num_keypoints", 0),
        ("bbox", [1, 2, -3, 4]),
        ("bbox", None),
        ("area", float("inf")),
        ("object_pose", [[1]]),
        ("camera_extrinsics", [[float("nan")] * 4] * 4),
        ("camera_intrinsics", {"fx": 0, "fy": 1, "cx": 0, "cy": 0}),
    ],
)
def test_corrupt_labels(tmp_path, field, value):
    payload = dataset(tmp_path)
    payload["annotations"][0][field] = value
    assert check(payload, tmp_path)


@pytest.mark.parametrize("name", ["../escape.png", "/tmp/escape.png", "..\\escape.png"])
def test_unsafe_paths(tmp_path, name):
    payload = dataset(tmp_path)
    payload["images"][0]["file_name"] = name
    payload["annotations"][0]["mask_file"] = name
    assert any("unsafe" in error for error in check(payload, tmp_path))


def test_symlink_escape(tmp_path):
    payload = dataset(tmp_path)
    (tmp_path / "images" / "escape").symlink_to(tmp_path.parent, target_is_directory=True)
    payload["images"][0]["file_name"] = "escape/file.png"
    assert any("escapes" in error for error in check(payload, tmp_path))


def test_worker_metadata_cannot_disable_expectations(tmp_path):
    payload = dataset(tmp_path)
    payload.update(images_per_sample=0, render_masks=False, num_keypoints=0)
    del payload["annotations"][0]["mask_file"]
    assert check(payload, tmp_path)


@pytest.mark.parametrize("damage", ["duplicate", "reference"])
def test_merge_rejects_ambiguous_records_before_write(tmp_path, damage):
    payload = dataset(tmp_path)
    if damage == "duplicate":
        payload["images"][1]["id"] = payload["images"][0]["id"]
    else:
        payload["annotations"][0]["image_id"] = 99
    (tmp_path / "annotations_worker_000.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        merge_annotations(tmp_path)
    assert not (tmp_path / "annotations.json").exists()


def test_merge_preserves_api_and_ignores_partial_files(tmp_path):
    payload = dataset(tmp_path)
    (tmp_path / "annotations_worker_000.json").write_text(json.dumps(payload))
    (tmp_path / "annotations_worker_001.partial.json").write_text("invalid")
    assert merge_annotations(tmp_path) == dict(images=2, annotations=2, masks=False)
    assert json.loads((tmp_path / "annotations.json").read_text())["images"] == payload["images"]
