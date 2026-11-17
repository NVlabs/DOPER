# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hold a file lock from the test and check that a second user waits for it."""

import fcntl
import threading
import time
from collections.abc import Callable
from pathlib import Path

WAIT_SECONDS = 0.5
JOIN_SECONDS = 60.0


def assert_waits_for(lock: Path, action: Callable[[], object], held: Callable[[], None]) -> object:
    """Run ``action`` in a thread while ``lock`` is held, then release it.

    Args:
        lock: The lock file the action must wait for.
        action: The second user (a run, an export).
        held: Checks made while the lock is held and the action is waiting.

    Returns:
        What ``action`` returned once the lock was released.
    """
    lock.parent.mkdir(parents=True, exist_ok=True)
    outcome: dict[str, object] = {}

    def target() -> None:
        try:
            outcome["result"] = action()
        except BaseException as err:  # noqa: BLE001 - re-raised in the test thread
            outcome["error"] = err

    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        thread = threading.Thread(target=target)
        thread.start()
        time.sleep(WAIT_SECONDS)
        assert thread.is_alive(), f"the second user did not wait for {lock}: {outcome}"
        held()
    thread.join(JOIN_SECONDS)
    assert not thread.is_alive()
    if "error" in outcome:
        raise outcome["error"]  # type: ignore[misc]
    return outcome.get("result")
