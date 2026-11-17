# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin the vendored HOT3D camera files to the hashes recorded in their NOTICE."""

import hashlib
import re
from pathlib import Path

import pytest

import doper._vendor.hot3d_camera as vendored

VENDOR_DIR = Path(vendored.__file__).parent
MODIFIED_FILES = ("camera.py", "camera_distortion.py")
MODIFIED_MARKER = "# Modified from upstream; see NOTICE."


def _notice_hashes() -> dict[str, str]:
    notice = (VENDOR_DIR / "NOTICE").read_text()
    shipped = notice.split("SHA-256 of the files as shipped", 1)[1]
    return dict(re.findall(r"^  (\S+\.py)\s+([0-9a-f]{64})$", shipped, flags=re.MULTILINE))


def test_notice_records_every_vendored_python_file() -> None:
    names = {path.name for path in VENDOR_DIR.glob("*.py")}
    assert set(_notice_hashes()) == names


@pytest.mark.parametrize("name", sorted(_notice_hashes()))
def test_vendored_file_matches_noticed_sha256(name: str) -> None:
    digest = hashlib.sha256((VENDOR_DIR / name).read_bytes()).hexdigest()
    assert digest == _notice_hashes()[name]


@pytest.mark.parametrize("name", MODIFIED_FILES)
def test_modified_files_keep_header_then_modification_marker(name: str) -> None:
    lines = (VENDOR_DIR / name).read_text().splitlines()
    assert lines[0].startswith("# Copyright (c) Meta Platforms")
    assert lines[13] == MODIFIED_MARKER


def test_unmodified_file_has_no_marker() -> None:
    assert MODIFIED_MARKER not in (VENDOR_DIR / "math_utils.py").read_text()
