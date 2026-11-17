# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recipe schema, packaged recipes and the per-dataset training inputs."""

import dataclasses
import json
from pathlib import Path

import pytest
import yaml
from bop_recipe_data import RECIPES, make_bop_download, write_recipe

from doper.bop import sources
from doper.recipes import inputs, schema

GOLDEN = Path(__file__).parent / "golden" / "source_base_recipes.json"
NAMES = ("handal", "hopev2", "hot3d", "tless")
PLACEHOLDERS = {key: f"<{key}>" for key in schema.PATH_KEYS}
SMALL = "convnext_small.dinov3_lvd1689m"


def _source_resolution(config):
    """Resolve a source campaign config the way ``doper.train`` reads a config file."""
    from doper.training.recipe import build_config, build_parser

    args = build_parser(dict(config)).parse_args([])
    launch = schema.LaunchSpec(
        **{f.name: getattr(args, f.name) for f in dataclasses.fields(schema.LaunchSpec)}
    )
    return build_config(args), launch


@pytest.mark.parametrize("name", NAMES)
def test_recipes_match_source_configs(name):
    source = json.loads(GOLDEN.read_text())[name]["config"]
    expected_cfg, expected_launch = _source_resolution(source)
    actual_cfg, actual_launch = schema.resolve_config(schema.load_recipe(name).base, PLACEHOLDERS)
    assert actual_launch == expected_launch
    if name != "hot3d":
        assert actual_cfg == expected_cfg
        return
    # User decision: the HOT3D recipe trains ConvNeXt-Small; the source campaign trained Tiny.
    assert expected_cfg.model.backbone == "convnext_tiny.dinov3_lvd1689m"
    assert actual_cfg.model.backbone == SMALL
    expected_cfg.model.backbone = SMALL
    assert actual_cfg == expected_cfg


@pytest.mark.parametrize("name", NAMES)
def test_recipe_shared_settings(name):
    cfg, launch = schema.resolve_config(schema.load_recipe(name).base, PLACEHOLDERS)
    model, train = cfg.model, cfg.train
    assert (model.backbone, model.det_input_size, model.crop_size, model.num_keypoints) == (
        SMALL,
        640,
        512,
        64,
    )
    assert model.keypoint_heatmap_sigma == 0.75 and model.pretrained
    assert (train.epochs, train.batch_size, train.gradient_accumulation_steps) == (25, 16, 4)
    assert (train.lr, train.backbone_lr_multiplier, train.warmup_iters) == (5e-4, 0.1, 200)
    assert train.amp and train.seed == 42
    assert launch.require_no_init_from and launch.require_pretrained_backbone
    tless = name == "tless"
    if tless:
        assert model.letterbox
    assert (cfg.data.instance_sampling, cfg.data.mask_target_mode) == (
        ("all", "union") if tless else ("primary", "primary")
    )


def test_packaged_recipe_objects():
    recipes = {name: schema.load_recipe(name) for name in NAMES}
    assert {n: len(r.objects) for n, r in recipes.items()} == {
        "handal": 40,
        "hopev2": 28,
        "hot3d": 33,
        "tless": 30,
    }
    assert recipes["handal"].generated_objects == (40,)
    assert recipes["hot3d"].generated_objects == (33,)
    assert 33 not in recipes["hot3d"].trainable_objects
    assert recipes["handal"].convert.source_view == "exclusions"
    assert recipes["hot3d"].validation.camera == "native"
    assert schema.packaged_recipes() == sorted((*NAMES, "shirt", "speedplus"))


def test_hot3d_recipe_records_the_backbone_decision():
    text = (RECIPES / "hot3d.yaml").read_text()
    assert "ConvNeXt-Small" in text and "Tiny" in text


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r.update(unknown=1), "unknown keys"),
        (lambda r: r["convert"].update(val_scene=5), "unknown keys"),
        (lambda r: r["base"]["config"].update(det_sise=640), "doper.train does not accept"),
        (lambda r: r["base"]["config"].update(init_from="x.pth"), "must not set"),
        (lambda r: r["base"]["config"].update(data_root="/data"), "must not set"),
        (lambda r: r["base"]["launch"].update(freeze=True), "unknown keys"),
        (lambda r: r.update(generated_objects=[99]), "generated_objects"),
        (lambda r: r["keypoints"].update(count=32), "num_keypoints"),
    ],
)
def test_load_recipe_rejects_unknown_and_inconsistent_keys(tmp_path, change, message):
    recipe = yaml.safe_load((RECIPES / "hopev2.yaml").read_text())
    change(recipe)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(recipe))
    with pytest.raises(ValueError, match=message):
        schema.load_recipe(path)


def test_load_recipe_rejects_an_unsupported_backbone(tmp_path):
    recipe = yaml.safe_load((RECIPES / "hopev2.yaml").read_text())
    recipe["base"]["config"]["backbone"] = "resnet18"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(recipe))
    with pytest.raises(ValueError, match="backbone"):
        schema.load_recipe(path)


def test_unknown_recipe_lists_packaged_ones():
    with pytest.raises(FileNotFoundError, match="hopev2"):
        schema.load_recipe("nope")
    with pytest.raises(FileNotFoundError):
        schema.load_recipe("train_example")


def test_train_example_is_a_valid_training_config():
    from doper.training.recipe import build_config, build_parser, load_config_file

    config = load_config_file(str(RECIPES / "train_example.yaml"))
    cfg = build_config(build_parser(config).parse_args([]))
    assert cfg.model.backbone == SMALL and cfg.data.root == "data/my_object"
    assert (cfg.model.det_input_size, cfg.model.crop_size) == (640, 512)


def test_recipe_objects_refuse_generated_and_foreign_objects():
    recipe = schema.load_recipe("handal")
    assert schema.recipe_objects(recipe, None) == tuple(range(1, 40))
    assert schema.recipe_objects(recipe, [3, 1]) == (1, 3)
    with pytest.raises(ValueError, match="no official PBR"):
        schema.recipe_objects(recipe, [40])
    with pytest.raises(ValueError, match="not handal objects"):
        schema.recipe_objects(recipe, [41])


def test_hot3d_view_chain_follows_the_pinned_defects():
    profile = sources.load_profile("hot3d")
    affected = min(sources.hot3d_affected_objects(profile))
    masked = next(o for o in range(1, 33) if sources.hot3d_mask_records(profile, o))
    plain = next(
        o
        for o in range(1, 33)
        if o not in sources.hot3d_affected_objects(profile)
        and not sources.hot3d_mask_records(profile, o)
    )
    assert inputs.hot3d_view_names(profile, plain) == ["canonical"]
    assert inputs.hot3d_view_names(profile, affected)[:2] == ["canonical", "canonical-depthfix"]
    assert inputs.hot3d_view_names(profile, masked)[-1].endswith(f"obj{masked:06d}")


def test_dataset_inputs_resolve_the_extracted_source(tmp_path):
    root = tmp_path / "root"
    profile_path = make_bop_download(root, tmp_path / "profile")
    recipe_path = write_recipe(tmp_path / "fixture.yaml", "hopev2", profile_path)
    recipe, profile = schema.load_recipe(recipe_path), sources.load_profile(profile_path)
    from doper.bop import extract

    sources.write_ready(root, profile, ["models", "pbr"])
    extract.extract_parts(root, profile, ["models", "pbr"])
    with pytest.raises(FileNotFoundError):
        inputs.dataset_inputs(recipe, profile, root, 1)
    extract.write_pbr_ready(root, profile)
    resolved = inputs.dataset_inputs(recipe, profile, root, 1)
    assert resolved.train_pbr == root / "data" / "train_pbr"
    assert resolved.models == root / "data" / "models"
    assert resolved.source_ready_sha256 == sources.sha256_file(root / "data" / "READY.json")
    assert inputs.planned_source(recipe, profile, root, 1) == resolved.train_pbr
    exclusions = root / "data" / "train_pbr" / "exclusions.json"
    record = json.loads(exclusions.read_text())
    exclusions.write_text(json.dumps({**record, "excluded_frames": [{"scene": "0", "frame": 1}]}))
    with pytest.raises(ValueError, match="separate policy"):
        inputs.dataset_inputs(recipe, profile, root, 1)
    with pytest.raises(ValueError, match="not a trainable"):
        inputs.dataset_inputs(recipe, profile, root, 9)
    ready_path = root / "data" / "READY.json"
    ready = json.loads(ready_path.read_text())
    del ready["identity"]
    ready_path.write_text(json.dumps(ready))
    with pytest.raises(ValueError, match="lacks 'identity'"):
        inputs.dataset_inputs(recipe, profile, root, 1)


ADAPT_METHODS = {
    "handal": [
        "static_pseudo",
        "dynamic_pseudo",
        "combined_pseudo",
        "gt_box_mask",
        "gt_box_mask_keypoints",
        "test_tta",
    ],
    "hopev2": [
        "static_pseudo",
        "dynamic_pseudo",
        "combined_pseudo",
        "gt_box_mask",
        "gt_box_mask_keypoints",
    ],
}
ADAPT_METHODS["hot3d"] = ADAPT_METHODS["hopev2"]


@pytest.mark.parametrize("name", ["handal", "hopev2", "hot3d"])
def test_adapt_methods_table(name):
    recipe = schema.load_recipe(name)
    adapt = recipe.adapt
    assert list(adapt.methods) == ADAPT_METHODS[name]
    assert schema.recipe_arms(recipe) == ("base", *ADAPT_METHODS[name])
    assert adapt.real_per_batch == 4 and adapt.gate == schema.GateSpec(pnp_method="epnp")
    minimums = {method: spec.minimum_labels for method, spec in adapt.methods.items()}
    assert minimums == {m: 1 if m.startswith("gt_") else 20 for m in ADAPT_METHODS[name]}
    methods = adapt.methods
    assert methods["gt_box_mask_keypoints"].supervision == "box_mask_keypoints"
    assert methods["gt_box_mask"].supervision == "box_mask"
    assert methods["combined_pseudo"].components == ("static_pseudo", "dynamic_pseudo")
    if name == "handal":
        assert methods["dynamic_pseudo"].objects == (8, 40)
        assert methods["combined_pseudo"].objects == (8, 40)
        assert dict(methods["combined_pseudo"].minimum_overrides) == {8: 1}
        assert methods["test_tta"].transductive and methods["test_tta"].source == "test"
    else:
        assert all(spec.objects is None for spec in methods.values())
    assert schema.recipe_arms(schema.load_recipe("tless")) == ("base",)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda a: a["config"].update(backbone="resnet18"), "inherits"),
        (lambda a: a["config"].update(max_iters_per_epoch=3), "computes"),
        (lambda a: a["config"].update(init_from="x.pth"), "must not set"),
        (lambda a: a["config"].update(det_sise=640), "doper.train does not accept"),
        (lambda a: a["gate"].update(score_gt=0.5), "unknown keys"),
        (lambda a: a["methods"]["static_pseudo"].update(supervision="box_mask_keypoints"), "GT"),
        (lambda a: a["methods"]["static_pseudo"].update(minimum_labels=0), "minimum"),
        (lambda a: a["methods"]["static_pseudo"].update(transductive=True), "transductive"),
        (lambda a: a["methods"]["combined_pseudo"].update(components=[]), "components"),
        (
            lambda a: a["methods"]["combined_pseudo"].update(components=["gt_box_mask"]),
            "components",
        ),
        (lambda a: a["methods"]["static_pseudo"].update(objects=[99]), "objects"),
        (lambda a: a["methods"]["static_pseudo"].update(minimum_overrides={99: 1}), "objects"),
        (lambda a: a["methods"].update(base={"source": "static", "minimum_labels": 1}), "base"),
        (lambda a: a["methods"]["static_pseudo"].update(source="pbr"), "source"),
    ],
)
def test_load_recipe_rejects_invalid_adapt_sections(tmp_path, change, message):
    recipe = yaml.safe_load((RECIPES / "hopev2.yaml").read_text())
    change(recipe["adapt"])
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(recipe))
    with pytest.raises(ValueError, match=message):
        schema.load_recipe(path)


def test_adapt_epochs_are_the_fixed_budget(tmp_path):
    recipe = yaml.safe_load((RECIPES / "hopev2.yaml").read_text())
    for epochs in (5, None):
        if epochs is None:
            del recipe["adapt"]["config"]["epochs"]
        else:
            recipe["adapt"]["config"]["epochs"] = epochs
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump(recipe))
        with pytest.raises(ValueError, match="epochs must be 3"):
            schema.load_recipe(path)
