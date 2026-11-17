# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every NVIDIA source file carries the Apache-2.0 SPDX header; vendored code keeps its own."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRS = ["doper", "rendering", "tests", "examples", "skills", "configs"]
VENDORED = ROOT / "doper" / "_vendor"
SPDX = "SPDX-License-Identifier: Apache-2.0"


def test_source_files_have_the_spdx_header() -> None:
    """Checks the first five lines of every Python and shell file outside the vendored code."""
    missing = []
    for top in SOURCE_DIRS:
        for path in sorted((ROOT / top).rglob("*")):
            if path.suffix not in {".py", ".sh"} or VENDORED in path.parents:
                continue
            if SPDX not in "".join(path.read_text().splitlines(True)[:5]):
                missing.append(path.relative_to(ROOT).as_posix())
    assert missing == []


def test_license_is_apache_2() -> None:
    """The root LICENSE is the Apache License, Version 2.0."""
    assert (
        "Apache License\n                           Version 2.0" in (ROOT / "LICENSE").read_text()
    )


def test_readme_states_contributions_are_not_accepted() -> None:
    """NVIDIA's open-source policy requires this exact README sentence when a project takes no
    external contributions."""
    assert (
        "This project is currently not accepting contributions." in (ROOT / "README.md").read_text()
    )
