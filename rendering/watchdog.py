# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stall detection for long-lived render workers.

Kit occasionally hangs inside a render step without returning or exiting; from inside the process
there is nothing to time out. The launcher instead watches each worker's footprint on disk (its
log and ``.partial.json`` checkpoint, which the worker touches at least every ``--flush_every_sec``)
and treats a worker whose footprint has not changed for ``stall_timeout_sec`` as hung: kill it and
relaunch the same range with ``--resume``, a bounded number of times. Resume reconstructs the
finished samples from the checkpoint, so a restart costs at most the un-flushed tail.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

_POPEN_TYPE = subprocess.Popen


def poll_without_reaping(proc: Any) -> int | None:  # noqa: ANN401 - Popen or a test double
    """Observe a child exit while retaining its PID until group cleanup finishes.

    Only this supervisor may reap its workers. WNOWAIT keeps an exited leader
    as our child, preventing PID reuse between group verification and signaling.
    """
    if not isinstance(proc, _POPEN_TYPE):
        return proc.poll()  # Test doubles have no operating-system child.
    if proc.returncode is not None:
        return proc.returncode
    info = os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    if info is None:
        return None
    return info.si_status if info.si_code == os.CLD_EXITED else -info.si_status


def latest_mtime(paths: Sequence[str]) -> float | None:
    """Newest mtime among the paths that exist, or None when none does."""
    times = []
    for path in paths:
        try:
            times.append(os.stat(path).st_mtime)
        except OSError:
            continue
    return max(times) if times else None


@dataclass
class StallMonitor:
    """Per-worker liveness from file activity.

    ``observe(mtime, now)`` records a changed mtime as a sign of life; ``stalled(now)`` is true once
    ``timeout_sec`` passed without one. Startup (Kit boot, shader compile, first frame) prints a
    banner and then goes quiet for many minutes, so no worker is declared stalled within
    ``startup_grace_sec + timeout_sec`` of its (re)launch regardless of file activity.
    """

    timeout_sec: float
    startup_grace_sec: float = 0.0
    last_activity: float = field(default_factory=time.monotonic)
    last_mtime: float | None = None
    launched_at: float = field(default_factory=time.monotonic)
    restarts: int = 0

    def launched(self, now: float) -> None:
        """Record a (re)launch: reopens the startup window."""
        self.launched_at = now
        self.last_activity = now

    def observe(self, mtime: float | None, now: float) -> None:
        """Count a changed ``mtime`` as a sign of life at ``now``."""
        if mtime is not None and mtime != self.last_mtime:
            self.last_mtime = mtime
            self.last_activity = now

    def stalled(self, now: float) -> bool:
        """True once the worker was quiet past the timeout and the startup window."""
        return (
            self.timeout_sec > 0
            and now - self.last_activity > self.timeout_sec
            and now - self.launched_at > self.timeout_sec + self.startup_grace_sec
        )


@dataclass
class WorkerSlot:
    """One worker's launch parameters and its current process."""

    worker_id: int
    start_idx: int
    end_idx: int
    gpu: int
    log_path: str
    proc: Any  # subprocess.Popen, or a test double with ``poll()``
    monitor: StallMonitor

    gave_up: bool = False
    completed: bool = False
    failure_reason: str | None = None

    def final_path(self, outdir: str) -> str:
        """The worker's completed annotation file."""
        return os.path.join(outdir, f"annotations_worker_{self.worker_id:03d}.json")

    def watched_paths(self, outdir: str) -> list[str]:
        """Files whose mtime shows the worker is alive."""
        return [
            self.log_path,
            os.path.join(outdir, f"annotations_worker_{self.worker_id:03d}.partial.json"),
            self.final_path(outdir),
        ]


def terminate_process_group(
    proc: subprocess.Popen, *, term_timeout_sec: float = 5.0, kill_timeout_sec: float = 5.0
) -> None:
    """Stop a POSIX worker launched with ``start_new_session=True`` and reap it.

    Signal only while the launcher still owns the unreaped leader PID. An
    exited but unreaped leader can still have live render subprocesses. Never
    infer group ownership from a missing or already-reaped leader.
    Descendants receive KILL after the grace period; waiting for the direct
    child is bounded even if the OS cannot immediately complete its exit.
    """
    if term_timeout_sec < 0 or kill_timeout_sec < 0:
        raise ValueError("process termination timeouts must be nonnegative")
    if getattr(proc, "returncode", None) is not None:
        return  # Already reaped: the numeric PID no longer proves ownership.
    pgid = proc.pid
    if pgid <= 1 or pgid == os.getpgrp():
        raise ValueError("refusing to terminate the launcher's process group")
    try:
        actual_group = os.getpgid(proc.pid)
    except ProcessLookupError:
        proc.wait(timeout=kill_timeout_sec)
        return  # No verifiable leader: never signal a possibly reused PGID.
    if actual_group != pgid:
        raise ValueError("worker must be launched with start_new_session=True")

    def signal_group(sig: int) -> bool:
        try:
            os.killpg(pgid, sig)
            return True
        except ProcessLookupError:
            return False

    if signal_group(signal.SIGTERM):
        deadline = time.monotonic() + term_timeout_sec
        while True:
            # Do not poll/wait here: reaping would release the PID before KILL.
            if not signal_group(0):
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                signal_group(signal.SIGKILL)
                break
            time.sleep(min(0.05, remaining))
    proc.wait(timeout=kill_timeout_sec)


def supervise(
    slots: Sequence[WorkerSlot],
    *,
    outdir: str,
    relaunch: Callable[[WorkerSlot], Any],
    validate_complete: Callable[[WorkerSlot], bool],
    max_restarts: int,
    now: float,
    log: Callable[[str], None] = print,
    stop: Callable[[Any], None] = terminate_process_group,
) -> list[int]:
    """Run one pass and return newly terminal failed worker IDs.

    Completion requires exit zero AND launcher validation of all artifacts.
    ``timeout_sec <= 0`` disables running-worker stall detection only. Before
    retry or terminal failure this function stops/reaps the old process group;
    ``relaunch`` must only launch a new worker with ``start_new_session=True``.
    Cleanup errors propagate so the launcher can abort instead of retrying over
    a worker that might still be alive.
    """
    gave_up = []
    for slot in slots:
        if slot.gave_up or slot.completed:
            continue
        code = poll_without_reaping(slot.proc)
        if code is None:
            slot.monitor.observe(latest_mtime(slot.watched_paths(outdir)), now)
            if not slot.monitor.stalled(now):
                continue
            reason = f"no progress for {(now - slot.monitor.last_activity) / 60.0:.0f} min"
        elif code != 0:
            reason = f"exited with code {code}"
        else:
            try:
                valid = validate_complete(slot)
            except Exception as exc:
                valid = False
                log(f"  Worker {slot.worker_id}: artifact validation failed: {exc}")
            if valid:
                if isinstance(slot.proc, _POPEN_TYPE):
                    slot.proc.wait(timeout=0)
                slot.completed = True
                continue
            reason = "exited without valid artifacts"
        slot.failure_reason = reason
        stop(slot.proc)
        if slot.monitor.restarts >= max_restarts:
            log(
                f"  Worker {slot.worker_id}: {reason} and {max_restarts} restart(s) used; giving up"
            )
            slot.gave_up = True
            gave_up.append(slot.worker_id)
            continue
        slot.monitor.restarts += 1
        log(
            f"  Worker {slot.worker_id}: {reason}; relaunching [{slot.start_idx}, {slot.end_idx}) "
            f"with --resume (restart {slot.monitor.restarts}/{max_restarts})"
        )
        slot.proc = relaunch(slot)
        slot.monitor.launched(now)
        slot.failure_reason = None
    return gave_up
