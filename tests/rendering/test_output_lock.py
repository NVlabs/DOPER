# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concurrent launchers must share one stable setup-lock inode."""

import fcntl
import subprocess
import sys
from pathlib import Path

from rendering.output_lock import output_lock


def test_setup_lock_blocks_another_process_without_creating_dataset(tmp_path):
    out = tmp_path / "data"
    with output_lock(out):
        lock = tmp_path / ".data.setup.lock"
        code = """import fcntl,sys
with open(sys.argv[1], 'a') as f:
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(7)
"""
        result = subprocess.run([sys.executable, "-c", code, str(lock)], timeout=5)
        assert result.returncode == 7
        assert not out.exists()
        inode = lock.stat().st_ino
    with output_lock(out):
        assert lock.stat().st_ino == inode
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_shard_ownership_is_separate_from_setup_and_other_shards(tmp_path):
    out = Path(tmp_path / "data")
    with output_lock(out, shard=(0, 2)), output_lock(out, shard=(2, 4)), output_lock(out):
        assert len(list(tmp_path.glob("*.lock"))) == 3
