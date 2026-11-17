# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every recipe's whole command chain plans under ``--dry-run`` without touching any file.

prepare (every part, generated objects, T-LESS sensors) -> train base -> adapt every method ->
follow-ups -> review -> evaluate val -> select -> every fixed policy -> every variant ->
timing -> report (tables, local AP, mask audit).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from doper import cli
from doper.bop import sources
from doper.bop.variants import MULTIVIEW_VARIANTS, VARIANTS
from doper.recipes.schema import Recipe, load_recipe, recipe_arms

RECIPES = ("handal", "hopev2", "hot3d", "tless")
TIMING_PROTOCOLS = ("deployment", "direct")


def _prepare(recipe: Recipe, root: str, render: str) -> list[list[str]]:
    parts = list(sources.load_profile(recipe.profile_ref()).parts)
    commands = [["prepare", recipe.name, "--root", root, "--parts", *parts, "--allow-test-inputs"]]
    for obj in recipe.generated_objects:
        generated = ["prepare", recipe.name, "--root", root, "--obj-id", str(obj)]
        commands += [[*generated, "--keypoints-only"], [*generated, "--generated-root", render]]
    if recipe.adapt is not None and recipe.adapt.multisensor is not None:
        commands.append(["prepare", recipe.name, "--root", root, "--multisensor", "--obj-id", "1"])
    return commands


def _train(recipe: Recipe, paths: list[str], obj: int) -> list[list[str]]:
    train = ["train", recipe.name, *paths]
    commands = [[*train, "--stage", "base", "--obj-id", str(obj)]]
    for method, spec in (recipe.adapt.methods if recipe.adapt is not None else {}).items():
        target = spec.objects[0] if spec.objects else obj
        command = [*train, "--stage", "adapt", "--method", method, "--obj-id", str(target)]
        commands.append([*command, "--allow-test-inputs"] if spec.transductive else command)
    for name, spec in recipe.followups.items():
        command = [*train, "--stage", "followup", "--name", name, "--obj-id", str(spec.obj_id)]
        commands.append(command)
        if spec.require_review:
            commands.append(["review", recipe.name, *paths, "--obj-id", str(spec.obj_id),
                             "--reviewer", "R", "--name", name])  # fmt: skip
    if recipe.adapt is not None and recipe.adapt.multisensor is not None:
        commands.append(["review", recipe.name, *paths, "--obj-id", "1", "--reviewer", "R"])
        commands.append([*train, "--stage", "multisensor", "--obj-id", "1"])
    return commands


def _evaluate(recipe: Recipe, paths: list[str]) -> list[list[str]]:
    evaluate = ["evaluate", recipe.name, *paths]
    commands = [[*evaluate, "--split", "val", "--arm", arm] for arm in recipe_arms(recipe)]
    test = [*evaluate, "--split", "test"]
    commands += [[*test, "--policy", p] for p in ("base", "selected", *recipe.test.policies)]
    variants = [v for v in VARIANTS if recipe.multiview is not None or v not in MULTIVIEW_VARIANTS]
    for variant in variants:
        commands.append([*test, "--policy", "base", "--variant", variant])
    commands.append([*test, "--policy", "base", "--variant", variants[-1], "--optimizer", "powell"])
    return commands


def _report(recipe: Recipe, root: str, runs: str) -> list[list[str]]:
    report = ["report", recipe.name, "--runs", runs]
    commands = [report]
    for protocol in TIMING_PROTOCOLS:
        commands.append([*report, "--root", root, "--timing", "--protocol", protocol])
    if recipe.name == "tless":
        commands.append([*report, "--root", root, "--ap"])
    if recipe.followups:
        commands.append([*report, "--root", root, "--mask-audit"])
    return commands


def _tree(path: Path) -> list[str]:
    return sorted(str(p.relative_to(path)) for p in path.rglob("*"))


@pytest.mark.parametrize("name", RECIPES)
def test_dry_run_chain_every_recipe(name, tmp_path, capsys):
    recipe = load_recipe(name)
    root, runs, render = tmp_path / "root", tmp_path / "runs", tmp_path / "render"
    root.mkdir()
    paths = ["--root", str(root), "--runs", str(runs)]
    obj = recipe.trainable_objects[0]
    commands = [
        *_prepare(recipe, str(root), str(render)),
        *_train(recipe, paths, obj),
        *_evaluate(recipe, paths),
        *_report(recipe, str(root), str(runs)),
    ]
    for command in commands:
        assert cli.main([*command, "--dry-run"]) == 0, command
    out = capsys.readouterr().out
    assert out.count("nothing was read or written") == len(commands)
    assert _tree(tmp_path) == ["root"]
