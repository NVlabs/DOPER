# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recipe kinds: BOP, SPEED+ and SHIRT recipes load through one entry point and route by kind."""

import argparse
import subprocess
import sys

import pytest
import yaml

from doper import cli
from doper.cli import common
from doper.recipes import load_any_recipe, recipe_kind
from doper.recipes.schema import Recipe, load_recipe, packaged_recipes
from doper.recipes.shirt_schema import ShirtRecipe
from doper.recipes.speedplus_schema import SpeedplusRecipe


def test_packaged_recipes_carry_their_kind():
    speedplus = load_any_recipe("speedplus")
    shirt = load_any_recipe("shirt")
    tless = load_any_recipe("tless")
    assert isinstance(speedplus, SpeedplusRecipe) and speedplus.kind == "speedplus"
    assert isinstance(shirt, ShirtRecipe) and shirt.kind == "shirt"
    assert isinstance(tless, Recipe)
    assert [recipe_kind(r) for r in (speedplus, shirt, tless)] == ["speedplus", "shirt", "bop"]
    assert speedplus.score == shirt.score == "paper"
    assert speedplus.profile == "speedplus" and speedplus.source is not None
    assert {"speedplus", "shirt", "tless"} <= set(packaged_recipes())


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r.update(kind="spec"), "kind"),
        (lambda r: r.update(score="spec"), "score"),
        (lambda r: r.update(stages={}), "unknown keys"),
        (lambda r: r.pop("profile"), "missing keys"),
    ],
)
def test_kind_recipes_reject_bad_keys(tmp_path, change, message):
    data = {"name": "speedplus", "kind": "speedplus", "profile": "speedplus", "score": "paper"}
    change(data)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match=message):
        load_any_recipe(path)


def test_explicit_bop_kind_loads_a_bop_recipe(tmp_path):
    data = yaml.safe_load(load_any_recipe("hopev2").source.read_text())
    path = tmp_path / "hopev2.yaml"
    path.write_text(yaml.safe_dump({**data, "kind": "bop"}))
    assert isinstance(load_any_recipe(path), Recipe)
    assert isinstance(load_recipe(path), Recipe)
    with pytest.raises(ValueError, match="load_any_recipe"):
        load_recipe("speedplus")
    with pytest.raises(FileNotFoundError, match="speedplus"):
        load_any_recipe("nope")


def test_recipe_kind_routes_and_review_refuses_non_bop(tmp_path, capsys):
    planners = {
        "bop": lambda args: [common.Step("bop plan")],
        "speedplus": lambda args: [common.Step("speedplus plan")],
    }
    args = argparse.Namespace(recipe="speedplus")
    assert [s.name for s in common.route_kind(args, planners, "train")] == ["speedplus plan"]
    args.recipe = "tless"
    assert [s.name for s in common.route_kind(args, planners, "train")] == ["bop plan"]
    args.recipe = "shirt"
    with pytest.raises(ValueError, match=r"`doper train` does not take shirt recipes"):
        common.route_kind(args, planners, "train")
    review = ["review", "speedplus", "--root", str(tmp_path), "--runs", str(tmp_path)]
    review += ["--obj-id", "1", "--reviewer", "someone", "--dry-run"]
    assert cli.main(review) == 1
    assert "`doper review` does not take speedplus recipes" in capsys.readouterr().out
    report = ["report", "speedplus", "--runs", str(tmp_path), "--dry-run"]
    assert cli.main(report) == 1
    assert "`doper report` does not take speedplus recipes" in capsys.readouterr().out
    # Commands that do not route SPEED+ or SHIRT yet refuse them instead of misparsing them.
    with pytest.raises(ValueError, match="is a shirt recipe"):
        common.load("shirt")


def test_recipe_parsing_imports_no_numpy():
    code = (
        "import sys; from doper.recipes import load_any_recipe; "
        "load_any_recipe('speedplus'); load_any_recipe('shirt'); print('numpy' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


@pytest.mark.parametrize("kind", ["speedplus", "shirt"])
@pytest.mark.parametrize(
    ("registry", "message"),
    [
        (["0" * 64], "mapping"),
        ({"0" * 63: "a" * 64}, "checkpoint SHA-256"),
        ({"0" * 64: "A" * 64}, "layout SHA-256"),
        ({"0" * 64: 7}, "layout SHA-256"),
    ],
)
def test_legacy_checkpoint_registry_holds_sha256_pairs(tmp_path, kind, registry, message):
    data = {"name": kind, "kind": kind, "profile": kind, "legacy_checkpoints": registry}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match=message):
        load_any_recipe(path)
    data["legacy_checkpoints"] = {"0" * 64: "a" * 64}
    path.write_text(yaml.safe_dump(data))
    assert load_any_recipe(path).legacy_checkpoints == {"0" * 64: "a" * 64}


def test_legacy_checkpoint_registries_agree():
    """SPEED+ and SHIRT consume the same checkpoints, so their registries must be equal."""
    registries = [load_any_recipe(name).legacy_checkpoints for name in ("speedplus", "shirt")]
    assert registries[0] == registries[1]
    assert registries[0]  # seeded with the gen-7 checkpoint
