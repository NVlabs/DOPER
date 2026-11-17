# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generated objects: render-root verification, source view, layout split and visible filter."""

import hashlib
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from bop_recipe_data import write_models, write_scene
from generated_render import render_config, write_render_root

from doper.bop import generated
from doper.cli.prepare import select_cad_keypoints
from doper.recipes import schema

GOLDEN = json.loads((Path(__file__).parent / "golden" / "generated_keypoints.json").read_text())
CONFIGS = Path(__file__).resolve().parents[2] / "rendering" / "blenderproc" / "configs"
# Directory holding the pinned official meshes as extracted from the dataset model archives:
# ``<dir>/handal/models/obj_000040.ply`` and ``<dir>/hot3d/models/obj_000033.ply``.
MODELS_ENV = "DOPER_BOP_MODELS"
SPEC = schema.GeneratedSpec(
    render_config="hot3d_obj33",
    layouts=6,
    frames_per_layout=3,
    base_seed=100,
    val_modulo=3,
    val_residue=2,
    empty_masks="exclude",
)


def _config(spec: schema.GeneratedSpec = SPEC, **changes) -> dict:
    split = {"modulo": spec.val_modulo, "val_residue": spec.val_residue}
    return render_config(
        spec.render_config,
        dataset="fixture",
        target_object=2,
        layouts=spec.layouts,
        frames_per_layout=spec.frames_per_layout,
        base_seed=spec.base_seed,
        split=split,
        **changes,
    )


@pytest.fixture
def campaign(tmp_path):
    """A complete 6-layout campaign of object 2 with frozen release keypoints."""
    models = tmp_path / "models"
    write_models(models)
    keypoints = select_cad_keypoints(models / "obj_000002.ply", tmp_path / "frozen", 2, 64, "mm")

    def build(name: str = "render", **kwargs) -> Path:
        return write_render_root(
            tmp_path / name,
            config=_config(),
            keypoints=keypoints,
            models=models,
            scene=lambda d, layout: write_scene(d, layout),
            **kwargs,
        )

    return build


def _view(render: Path, tmp_path: Path, spec: schema.GeneratedSpec = SPEC) -> Path:
    return generated.source_view(render, spec, tmp_path / "view", dataset="fixture", obj=2)


# --- keypoints ------------------------------------------------------------------------------


@pytest.mark.parametrize(("obj", "config"), [(40, "handal_obj40"), (33, "hot3d_obj33")])
def test_generated_keypoint_hashes(obj, config, tmp_path):
    """Release selection on the pinned mesh: the pinned file SHA and the research layout."""
    golden = GOLDEN["objects"][str(obj)]
    root = os.environ.get(MODELS_ENV)
    mesh = Path(root or "/nonexistent") / golden["dataset"] / "models" / f"obj_{obj:06d}.ply"
    if not mesh.is_file():
        pytest.skip(f"{MODELS_ENV} does not hold the pinned {golden['dataset']} mesh")
    assert hashlib.sha256(mesh.read_bytes()).hexdigest() == golden["mesh_sha256"]
    out = select_cad_keypoints(mesh, tmp_path, obj, 64, "mm")
    pinned = json.loads((CONFIGS / f"{config}.json").read_text())
    assert hashlib.sha256(out.read_bytes()).hexdigest() == pinned["keypoints_sha256"]
    assert pinned["source_keypoints_sha256"] == golden["source_keypoints_sha256"]
    payload = json.loads(out.read_text())
    assert payload["object"] == golden["object"]
    assert payload["keypoint_names"] == golden["keypoint_names"]
    assert payload["indices"] == golden["indices"]
    assert payload["coordinates_sha256"] == golden["coordinates_sha256"]


def test_frozen_keypoints_must_have_the_selected_layout(tmp_path):
    selected = tmp_path / "selected.json"
    names = [f"KP_{i:03d}" for i in range(3)]
    points = [[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.0, 0.002, 0.0]]
    selected.write_text(
        json.dumps(
            {
                "object": "obj_000002",
                "keypoint_names": names,
                "indices": [3, 1, 2],
                "keypoints_3d": points,
                "units": "m",
            }
        )
    )
    frozen = tmp_path / "frozen.json"
    frozen.write_text(
        json.dumps(
            {
                "object": "obj_000002",
                "keypoint_names": names,
                "indices": [3, 1, 2],
                "keypoints_3d": points,
                "hull_anchor_selection": {"anchor_saliency": [0.1]},
            }
        )
    )
    generated.require_same_layout(selected, frozen)  # other report fields may differ
    frozen.write_text(
        json.dumps(
            {
                "object": "obj_000002",
                "keypoint_names": names,
                "indices": [1, 3, 2],
                "keypoints_3d": points,
            }
        )
    )
    with pytest.raises(ValueError, match="indices"):
        generated.require_same_layout(selected, frozen)


# --- render root ----------------------------------------------------------------------------


def test_source_view_links_every_verified_layout(campaign, tmp_path):
    render = campaign()
    view = _view(render, tmp_path)
    assert view == tmp_path / "view" / "train_pbr"
    assert sorted(p.name for p in view.iterdir()) == [f"{i:06d}" for i in range(6)]
    assert (view / "000004").resolve() == (render / "train_pbr" / "000004").resolve()
    ready = json.loads((tmp_path / "view" / "READY.json").read_text())
    assert ready["layouts"] == 6 and ready["target_object"] == 2
    assert ready["render_config"] == "hot3d_obj33" and ready["base_seed"] == 100
    assert ready["validation_layouts"] == [2, 5]
    assert _view(render, tmp_path) == view  # idempotent


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"missing": [1, 4]}, r"missing layouts \[1, 4\]"),
        ({"failed": [3]}, r"recorded failed layouts \[3\]"),
    ],
)
def test_prepare_generated_refuses_unreceipted_layout(campaign, tmp_path, kwargs, message):
    render = campaign(**kwargs)
    with pytest.raises(ValueError, match=message):
        _view(render, tmp_path)
    assert not (tmp_path / "view").exists()


def test_source_view_refuses_changed_layouts_and_identity(campaign, tmp_path):
    render = campaign()
    info = render / "train_pbr" / "000002" / "scene_gt_info.json"
    original = info.read_bytes()
    info.write_text("{}")
    with pytest.raises(ValueError, match=r"000002.*scene_gt_info.json"):
        _view(render, tmp_path)
    info.write_bytes(original)
    receipt = render / "train_pbr" / "000003" / "COMPLETE.json"
    record = json.loads(receipt.read_text())
    receipt.write_text(json.dumps({**record, "seed": 7}))
    with pytest.raises(ValueError, match="000003"):
        _view(render, tmp_path)
    receipt.write_text(json.dumps(record))
    for change, match in [
        ({"base_seed": 70000}, "base_seed"),
        ({"layouts": 7}, "layouts"),
        ({"render_config": "handal_obj40"}, "render_config"),
        ({"val_residue": 1}, "split"),
    ]:
        spec = schema.GeneratedSpec(**{**SPEC.as_record(), **change})
        with pytest.raises(ValueError, match=match):
            _view(render, tmp_path, spec)
    with pytest.raises(ValueError, match="target_object"):
        generated.source_view(render, SPEC, tmp_path / "view", dataset="fixture", obj=1)
    with pytest.raises(ValueError, match="dataset"):
        generated.source_view(render, SPEC, tmp_path / "view", dataset="hot3d", obj=2)
    (render / "STAGED.json").unlink()
    with pytest.raises(ValueError, match="STAGED"):
        _view(render, tmp_path)


def test_source_view_refuses_a_different_render_root(campaign, tmp_path):
    first = campaign("first")
    _view(first, tmp_path)
    other = campaign("second")
    with pytest.raises(ValueError, match="render root changed"):
        _view(other, tmp_path)


# --- split ----------------------------------------------------------------------------------


def _converted(root: Path, scenes: int, frames: int = 2) -> Path:
    """A converted root: one annotation per frame, two on frame 0 of every scene."""
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    images, annotations = [], []
    for scene in range(scenes):
        for frame in range(frames):
            name = f"{scene:06d}_{frame:06d}.jpg"
            (root / "images" / name).write_bytes(b"x")
            images.append({"id": len(images), "file_name": name, "width": 4, "height": 3})
            for instance in range(2 if frame == 0 else 1):
                mask = f"masks/{scene:06d}_{frame:06d}_{instance:06d}.png"
                (root / mask).write_bytes(b"m")
                annotations.append(
                    {
                        "id": len(annotations),
                        "image_id": images[-1]["id"],
                        "mask_file": mask,
                        "bop": {"scene": f"{scene:06d}", "frame": frame, "instance": instance},
                    }
                )
    data = {"images": images, "annotations": annotations, "categories": [{"id": 1}]}
    (root / "annotations.json").write_text(json.dumps(data))
    return root


def test_layout_modulo_split_matches_source(tmp_path):
    """Source ``merge``: layout ``l`` is validation iff ``l % 10 == 9``; no layout is split;
    images follow the 16 conversion shards (``range(shard, layouts, 16)``) in shard order."""
    converted = _converted(tmp_path / "converted", 20)
    keypoints = tmp_path / "keypoints_3d.json"
    keypoints.write_text("{}")
    spec = schema.GeneratedSpec(
        render_config="handal_obj40", base_seed=60000, conversion_shards=16, empty_masks="keep"
    )
    summary = generated.split_layouts(converted, tmp_path / "dataset", 40, keypoints, spec)
    shards = [layout for shard in range(16) for layout in range(shard, 20, 16)]
    for split, val in (("train", False), ("val", True)):
        data = json.loads((tmp_path / "dataset" / split / "annotations.json").read_text())
        order = [im["file_name"] for im in data["images"]]
        expected = [
            f"{layout:06d}_{frame:06d}.jpg"
            for layout in shards
            for frame in range(2)
            if (layout % 10 == 9) == val
        ]
        assert order == expected
    assert summary.val_scenes == ("000009", "000019")
    assert summary.train_scenes == tuple(f"{i:06d}" for i in range(20) if i % 10 != 9)
    source = json.loads((converted / "annotations.json").read_text())
    by_key = {
        (a["bop"]["scene"], a["bop"]["frame"], a["bop"]["instance"]): a
        for a in source["annotations"]
    }
    for split, scenes in (("train", summary.train_scenes), ("val", summary.val_scenes)):
        data = json.loads((tmp_path / "dataset" / split / "annotations.json").read_text())
        assert [im["id"] for im in data["images"]] == list(range(len(data["images"])))
        assert [a["id"] for a in data["annotations"]] == list(range(len(data["annotations"])))
        assert {a["bop"]["scene"] for a in data["annotations"]} == set(scenes)
        files = {im["id"]: im["file_name"] for im in data["images"]}
        for annotation in data["annotations"]:
            key = (
                annotation["bop"]["scene"],
                annotation["bop"]["frame"],
                annotation["bop"]["instance"],
            )
            original = by_key[key]
            assert files[annotation["image_id"]].startswith(key[0])
            assert {k: v for k, v in annotation.items() if k not in ("id", "image_id")} == {
                k: v for k, v in original.items() if k not in ("id", "image_id")
            }
    assert (summary.train_annotations, summary.val_annotations) == (54, 6)
    assert (tmp_path / "dataset" / "keypoints_3d.json").read_text() == "{}"


# --- empty visible masks --------------------------------------------------------------------


def _visible_inputs(tmp_path: Path) -> tuple[Path, Path, dict]:
    """Converted data of one scene: frame 0 has a visible and an invisible target, frame 1 only
    an invisible one; ``scene_gt_info`` holds the authoritative visible counts."""
    source = tmp_path / "train_pbr" / "000000"
    converted = tmp_path / "converted"
    for directory in (source / "mask_visib", converted / "masks", converted / "images"):
        directory.mkdir(parents=True)
    rows = {"0": [(33, 12), (33, 0)], "1": [(5, 9), (33, 0)]}
    images, annotations, gt, info = [], [], {}, {}
    for frame, instances in rows.items():
        images.append(
            {"id": int(frame), "file_name": f"000000_{int(frame):06d}.jpg", "width": 4, "height": 3}
        )
        gt[frame] = [{"obj_id": obj} for obj, _ in instances]
        info[frame] = [{"px_count_visib": count} for _, count in instances]
        for instance, (obj, count) in enumerate(instances):
            mask = np.zeros((3, 4), np.uint8)
            mask[0, : min(count, 4)] = 255
            name = f"{int(frame):06d}_{instance:06d}.png"
            cv2.imwrite(str(source / "mask_visib" / name), mask)
            if obj == 33:
                shutil.copyfile(
                    source / "mask_visib" / name, converted / "masks" / f"000000_{name}"
                )
                annotations.append(
                    {
                        "id": len(annotations),
                        "image_id": int(frame),
                        "mask_file": f"masks/000000_{name}",
                        "keypoints": [1, 1, 2, 2, 2, 1],
                        "bop": {"scene": "000000", "frame": int(frame), "instance": instance},
                    }
                )
    (source / "scene_gt.json").write_text(json.dumps(gt))
    (source / "scene_gt_info.json").write_text(json.dumps(info))
    data = {
        "info": {"description": "parent"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1}],
    }
    (converted / "annotations.json").write_text(json.dumps(data))
    (converted / "keypoints_3d.json").write_text("{}")
    (converted / "conversion_provenance.json").write_text("{}")
    return converted, tmp_path / "train_pbr", data


def test_obj33_excludes_zero_visible_annotations(tmp_path):
    """Source ``prepare_hot3d33_visible.select_visible``: drop only proven-empty targets."""
    converted, train_pbr, data = _visible_inputs(tmp_path)
    kept, excluded = generated.select_visible(
        data,
        generated.scene_lookup(train_pbr),
        lambda ann, image: generated.zero_mask_proof(converted, train_pbr, ann, image),
        33,
    )
    assert [a["id"] for a in kept["annotations"]] == [0]
    assert kept["annotations"][0] == data["annotations"][0]  # retained records verbatim
    assert [im["id"] for im in kept["images"]] == [0]  # frame 1 lost its only target
    assert [(e["annotation_id"], e["reason"]) for e in excluded] == [
        (1, "original_visible_count_zero"),
        (2, "original_visible_count_zero"),
    ]
    assert excluded[0]["mask_proof"]["nonzero_pixels"] == 0
    assert kept["info"]["num_samples"] == 1
    assert kept["info"]["selection_policy"] == generated.VISIBLE_POLICY
    assert kept["info"]["keypoint_visibility"] == {"visible": 1, "occluded": 1}
    summary = generated.exclude_invisible(converted, train_pbr, tmp_path / "visible", 33)
    assert (summary["excluded_annotations"], summary["excluded_images"]) == (2, 1)
    visible = json.loads((tmp_path / "visible" / "annotations.json").read_text())
    assert visible["annotations"] == kept["annotations"]
    selection = json.loads((tmp_path / "visible" / "VISIBLE_SELECTION.json").read_text())
    assert selection["output_annotations"] == 1 and selection["input_images"] == 2
    assert (tmp_path / "visible" / "masks" / "000000_000000_000000.png").exists()


def test_visible_filter_requires_a_proven_empty_mask(tmp_path):
    converted, train_pbr, data = _visible_inputs(tmp_path)
    lookup = generated.scene_lookup(train_pbr)

    def proof(ann, image):
        return generated.zero_mask_proof(converted, train_pbr, ann, image)

    (converted / "masks" / "000000_000001_000001.png").unlink()
    with pytest.raises(ValueError, match="missing mask"):
        generated.select_visible(data, lookup, proof, 33)
    mask = np.zeros((3, 4), np.uint8)
    mask[1, 1] = 255
    for path in (
        converted / "masks" / "000000_000001_000001.png",
        train_pbr / "000000" / "mask_visib" / "000001_000001.png",
    ):
        cv2.imwrite(str(path), mask)
    with pytest.raises(ValueError, match="zero mask"):
        generated.select_visible(data, lookup, proof, 33)
    with pytest.raises(ValueError, match="visibility"):
        generated.select_visible(data, lookup, proof, 34)


def test_empty_masks_fail_names_the_annotations(tmp_path):
    converted, train_pbr, _ = _visible_inputs(tmp_path)
    with pytest.raises(ValueError, match=r"2 converted annotations .* empty visible mask"):
        generated.require_visible(converted, train_pbr, 33)


# --- recipes --------------------------------------------------------------------------------


def test_recipe_generated_campaigns():
    handal, hot3d = schema.load_recipe("handal"), schema.load_recipe("hot3d")
    assert handal.generated_spec(40) == schema.GeneratedSpec(
        render_config="handal_obj40",
        base_seed=60000,
        conversion_shards=16,
        keypoints_sha256="91c62d706e63afc3c2dcd8422bcc402c324b19cd86fa232ed93521839b50de9c",
        empty_masks="keep",
    )
    assert hot3d.generated_spec(33).conversion_shards == 4
    spec = hot3d.generated_spec(33)
    assert (spec.render_config, spec.base_seed, spec.empty_masks) == (
        "hot3d_obj33",
        70000,
        "exclude",
    )
    assert (spec.layouts, spec.frames_per_layout, spec.val_modulo, spec.val_residue) == (
        1000,
        25,
        10,
        9,
    )
    assert hot3d.convert_for(33).split == "layout_modulo"
    assert hot3d.convert_for(2).split == "scenes"
    for recipe, obj in ((handal, 40), (hot3d, 33)):
        config = json.loads(
            (CONFIGS / f"{recipe.generated_spec(obj).render_config}.json").read_text()
        )
        assert (config["dataset"], config["target_object"]) == (recipe.name, obj)
        assert config["base_seed"] == recipe.generated_spec(obj).base_seed
        # One pin: the recipe checks the selection, the campaign stages it.
        assert config["keypoints_sha256"] == recipe.generated_spec(obj).keypoints_sha256
        # The generated object's base is the dataset's Small base.
        assert recipe.base.config["backbone"] == "convnext_small.dinov3_lvd1689m"


def test_recipe_objects_accept_generated_objects_with_a_render_root():
    recipe = schema.load_recipe("handal")
    assert schema.recipe_objects(recipe, None) == tuple(range(1, 40))
    assert schema.recipe_objects(recipe, None, generated=True) == tuple(range(1, 41))
    assert schema.recipe_objects(recipe, [40], generated=True) == (40,)
    with pytest.raises(ValueError, match="--generated-root"):
        schema.recipe_objects(recipe, [40])
