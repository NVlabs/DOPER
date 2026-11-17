# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install contract: the core package imports without optional runtimes."""

import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import doper

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTICE_ENTRIES = [
    "hand_tracking_toolkit",
    "Copyright (c) Meta Platforms, Inc. and affiliates",
    "bop_toolkit",
    "Copyright (c) 2019 Tomas Hodan",
    "MMPose",
    "Copyright 2018-2020 Open-MMLab",
    "CenterNet",
    "Copyright (c) 2019 Xingyi Zhou",
    "CornerNet",
    "Copyright (c) 2018, University of Michigan",
]

PAPER_RECIPES = ["handal", "hopev2", "hot3d", "tless", "speedplus", "shirt"]

OPTIONAL = [
    "bpy",
    "omni",
    "isaacsim",
    "pxr",
    "pyrealsense2",
    "tensorrt",
    "onnx",
    "bayesfilter",
    "h5py",
    "nurec_toolbox",
    "robotics_clients",
]


def test_core_import_has_no_optional_runtimes():
    code = (
        "import sys, doper, doper.train, doper.infer; "
        "print(sorted(m for m in sys.modules if m.split('.')[0] in %r))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code % OPTIONAL], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "[]"


def test_version():
    assert doper.__version__ == "0.1.0.dev0"


def test_third_party_notices_cover_vendored_and_ported_code():
    text = (REPO_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    missing = [entry for entry in NOTICE_ENTRIES if entry not in text]
    assert missing == []


def planner_kinds(module: str, verb: str) -> set[str]:
    """Reads the recipe kinds a ``doper`` verb routes from the planner dict in its source."""
    source = (REPO_ROOT / "doper" / "cli" / f"{module}").read_text(encoding="utf-8")
    pattern = r"(?:planners = |route_kind\(args, )\{([^}]*)\}"
    tables = re.findall(pattern, source)
    assert len(tables) == 1, verb
    names = {"BOP_KIND": "bop", "NurecRecipe.kind": "nurec"}
    keys = [item.split(":")[0].strip() for item in tables[0].split(",") if item.strip()]
    return {names.get(key, key.strip('"')) for key in keys}


VERB_MODULES = {
    "prepare": "prepare/__init__.py",
    "train": "train.py",
    "review": "review.py",
    "evaluate": "evaluate.py",
    "report": "report.py",
}


def table_verbs(readme: str) -> dict[str, set[str]]:
    """Maps each recipe of the README's recipe table to the verbs its row lists."""
    rows = {}
    for line in readme.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not line.startswith("| `") or len(cells) < 2:
            continue
        names = re.findall(r"`([a-z0-9_]+)`", cells[0])
        listed = set(re.findall(r"`([a-z]+)`", re.sub(r"\([^)]*\)", "", cells[1])))
        if listed and listed <= set(VERB_MODULES):
            rows.update({name: listed for name in names})
    return rows


def table_mismatches(readme: str, kinds: dict[str, set[str]]) -> list[str]:
    from doper.recipes import load_any_recipe, recipe_kind

    mismatches = []
    for name, listed in table_verbs(readme).items():
        kind = recipe_kind(load_any_recipe(name))
        expected = {verb for verb in VERB_MODULES if kind in kinds[verb]}
        if listed != expected:
            mismatches.append(f"{name}: lists {sorted(listed)}, routes {sorted(expected)}")
    return mismatches


def test_table_check_rejects_a_verb_the_recipe_does_not_route():
    kinds = {verb: planner_kinds(module, verb) for verb, module in VERB_MODULES.items()}
    bad = "| `speedplus` | `prepare`, `train` | x |"
    good = "| `speedplus` | `prepare` (`--stage fetch`) | x |"
    assert table_mismatches(bad, kinds) != []
    assert table_mismatches(good, kinds) == []


def test_readme_lists_every_pyproject_extra():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    section = pyproject.split("[project.optional-dependencies]")[1].split("\n[")[0]
    extras = re.findall(r"^([a-z][a-z0-9-]*) = \[", section, re.M)
    assert len(extras) >= 11
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert [name for name in extras if f"`{name}`" not in readme] == []


def test_readme_verbs_match_the_planner_tables():
    from doper.recipes import load_any_recipe, recipe_kind

    recipes = sorted(path.stem for path in (REPO_ROOT / "doper" / "recipes").glob("*.yaml"))
    recipes = [name for name in recipes if name != "train_example"]
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    verbs = "|".join(VERB_MODULES)
    uses = re.findall(rf"\bdoper\s+({verbs})\s+({'|'.join(recipes)})\b", readme)
    assert uses
    kinds = {verb: planner_kinds(module, verb) for verb, module in VERB_MODULES.items()}
    unrouted = [
        f"doper {verb} {name}"
        for verb, name in uses
        if recipe_kind(load_any_recipe(name)) not in kinds[verb]
    ]
    assert unrouted == []
    assert set(table_verbs(readme)) >= {"handal", "speedplus", "shirt"}
    assert table_mismatches(readme, kinds) == []
    for name in PAPER_RECIPES:
        assert name in readme, name


@pytest.mark.slow
def test_wheel_ships_the_notices_and_vendored_licenses(tmp_path):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not available to build the wheel")
    subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    wheels = list(tmp_path.glob("doper-*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = wheel.namelist()
    assert any(n.endswith(".dist-info/licenses/THIRD_PARTY_NOTICES.md") for n in names)
    assert "doper/_vendor/hot3d_camera/LICENSE" in names
    assert "doper/_vendor/bop_toolkit_ap/LICENSE" in names


CI_FILES = [".gitlab-ci.yml", ".github/workflows/ci.yml"]


@pytest.mark.parametrize("name", CI_FILES)
def test_ci_runs_make_through_uv(name):
    text = (REPO_ROOT / name).read_text(encoding="utf-8")
    assert "uv run --no-sync make check test" in text
    assert "uv sync --extra filter --extra hdf5 --extra tensorrt --group dev" in text
    cpu = "uv pip install --no-deps --reinstall"
    assert cpu in text
    assert text.index("uv sync") < text.index(cpu) < text.index("uv run --no-sync")
    assert "3.11" in text and "3.12" in text


def locked_version(name: str) -> str:
    """Reads a package's pinned version from uv.lock."""
    text = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")
    match = re.search(rf'^name = "{name}"\nversion = "([^"]+)"', text, re.M)
    assert match, name
    return match.group(1)


@pytest.mark.parametrize("name", CI_FILES)
def test_ci_cpu_torch_matches_lock(name):
    text = (REPO_ROOT / name).read_text(encoding="utf-8")
    expected = f"torch=={locked_version('torch')} torchvision=={locked_version('torchvision')} "
    assert f"uv pip install --no-deps --reinstall {expected}--index-url" in text


def test_github_ci_has_read_only_permissions():
    text = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "permissions:\n  contents: read\n" in text
