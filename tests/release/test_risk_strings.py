# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard: tracked release files must not contain local paths or secret-like literals.

Only generic, portable patterns live here. Organization-specific patterns
(internal hosts, infrastructure and customer names) are checked by an internal
audit test outside the release tree.
"""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SELF = Path(__file__).resolve()
RELEASE_PATHS = [
    "doper/",
    "rendering/",
    "scripts/",
    "configs/",
    "tests/",
    "README.md",
    "docs/nurec.md",
    "examples/",
    "release/public-root/",
    "skills/",
    "pyproject.toml",
    "Makefile",
    ".gitlab-ci.yml",
    ".github/",
    "uv.lock",
]
SKIPPED_PARTS = {"__pycache__"}
SKIPPED_SUFFIXES = {".pyc"}

RISK_PATTERNS = {
    "home_path": r"/home/[a-z]|~/repos",
    "cluster_storage": r"(?i)/lustre|slurm|/mnt/",
    "secret_like": r"(?i)(api[_-]?key|password|secret|token)\s*[:=]\s*['\"][^'\"]{8,}",
}


def is_git_checkout_root(root: Path = ROOT) -> bool:
    """Returns whether root is itself the top level of a git work tree."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0 and Path(result.stdout.strip()).resolve() == root.resolve()


def list_release_files(root: Path = ROOT) -> list[Path]:
    """Lists release files under root by walking RELEASE_PATHS on disk."""
    found: list[Path] = []
    for name in RELEASE_PATHS:
        target = root / name
        candidates = sorted(target.rglob("*")) if target.is_dir() else [target]
        found.extend(candidates)
    return [
        p
        for p in found
        if p.is_file()
        and not SKIPPED_PARTS.intersection(p.parts)
        and p.suffix not in SKIPPED_SUFFIXES
    ]


def tracked_files(root: Path = ROOT) -> list[Path]:
    """Lists release files: tracked ones in a git checkout root, else those on disk."""
    if not is_git_checkout_root(root):
        paths = list_release_files(root)
    else:
        out = subprocess.run(
            ["git", "ls-files", "--", *RELEASE_PATHS],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        paths = [root / name for name in out]
    return [p for p in paths if p.is_file() and p.resolve() != SELF]


def test_release_tree_has_no_risk_strings():
    files = tracked_files()
    assert files
    compiled = {name: re.compile(rx) for name, rx in RISK_PATTERNS.items()}
    hits = []
    for path in files:
        text = path.read_text(errors="replace")
        hits.extend((path, name) for name, rx in compiled.items() if rx.search(text))
    assert hits == []


def copy_release_files(root: Path) -> None:
    for name in ("pyproject.toml", "tests/release/test_risk_strings.py"):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((ROOT / name).read_text())


def test_release_guard_runs_outside_a_git_checkout(tmp_path):
    copy_release_files(tmp_path)
    listed = list_release_files(tmp_path)
    assert {p.relative_to(tmp_path).as_posix() for p in listed} == {
        "pyproject.toml",
        "tests/release/test_risk_strings.py",
    }


def test_release_guard_walks_disk_when_nested_in_another_git_repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    exported = tmp_path / "exported"
    copy_release_files(exported)
    assert not is_git_checkout_root(exported)
    listed = tracked_files(exported)
    assert {p.relative_to(exported).as_posix() for p in listed} == {
        "pyproject.toml",
        "tests/release/test_risk_strings.py",
    }
