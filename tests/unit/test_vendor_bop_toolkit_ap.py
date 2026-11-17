# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin the vendored bop_toolkit scoring files to their NOTICE hashes and to upstream code."""

import hashlib
import inspect
import re
from pathlib import Path

import pytest

import doper._vendor.bop_toolkit_ap as vendored
from doper._vendor.bop_toolkit_ap import misc, pose_matching, score

VENDOR_DIR = Path(vendored.__file__).parent
MODIFIED_FILES = ("score.py", "misc.py")
MODIFIED_MARKER = "# Modified from upstream; see NOTICE."


def _notice_hashes(section: str) -> dict[str, str]:
    notice = (VENDOR_DIR / "NOTICE").read_text()
    block = notice.split(section, 1)[1].split("\n\n", 1)[0]
    return dict(re.findall(r"^  (\S+\.py)\s+([0-9a-f]{64})$", block, flags=re.MULTILINE))


def test_notice_records_every_vendored_python_file() -> None:
    names = {path.name for path in VENDOR_DIR.glob("*.py")}
    assert set(_notice_hashes("SHA-256 of the files as shipped")) == names


@pytest.mark.parametrize("name", sorted(_notice_hashes("SHA-256 of the files as shipped")))
def test_vendored_file_matches_noticed_sha256(name: str) -> None:
    digest = hashlib.sha256((VENDOR_DIR / name).read_bytes()).hexdigest()
    assert digest == _notice_hashes("SHA-256 of the files as shipped")[name]


def test_unmodified_file_is_upstream_bytes() -> None:
    upstream = _notice_hashes("SHA-256 of the upstream files")["pose_matching.py"]
    assert hashlib.sha256((VENDOR_DIR / "pose_matching.py").read_bytes()).hexdigest() == upstream
    assert MODIFIED_MARKER not in (VENDOR_DIR / "pose_matching.py").read_text()


@pytest.mark.parametrize("name", MODIFIED_FILES)
def test_modified_files_keep_header_then_modification_marker(name: str) -> None:
    lines = (VENDOR_DIR / name).read_text().splitlines()
    assert lines[0].startswith("# Author: Tomas Hodan")
    assert lines[2] == MODIFIED_MARKER


def test_score_edit_is_only_the_misc_import() -> None:
    shipped = (VENDOR_DIR / "score.py").read_text().split("\n")
    shipped.remove(MODIFIED_MARKER)
    upstream = "\n".join(shipped).replace("from . import misc", "from bop_toolkit_lib import misc")
    expected = _notice_hashes("SHA-256 of the upstream files")["score.py"]
    assert hashlib.sha256(upstream.encode()).hexdigest() == expected


def test_vendored_functions_equal_installed_toolkit() -> None:
    toolkit_score = pytest.importorskip("bop_toolkit_lib.score")
    toolkit_matching = pytest.importorskip("bop_toolkit_lib.pose_matching")
    toolkit_misc = pytest.importorskip("bop_toolkit_lib.misc")
    pairs = [
        (score.calc_ap, toolkit_score.calc_ap),
        (score.calc_pose_detection_scores, toolkit_score.calc_pose_detection_scores),
        (pose_matching.match_poses, toolkit_matching.match_poses),
        (pose_matching.match_poses_scene, toolkit_matching.match_poses_scene),
        (misc.log, toolkit_misc.log),
    ]
    for ours, theirs in pairs:
        assert inspect.getsource(ours) == inspect.getsource(theirs)
