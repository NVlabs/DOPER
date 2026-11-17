# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from rendering import watchdog as W  # noqa: N812


class FakeProc:
    def __init__(self, running: bool = True):
        self.running = running
        self.killed = False
        self.returncode = None if running else 0

    def poll(self):
        return None if self.running else self.returncode


def _stop(proc):
    proc.running = False
    proc.killed = True
    proc.returncode = -15


def _valid_artifacts(slot, outdir):
    path = Path(slot.final_path(str(outdir)))
    return path.exists() and path.read_text() == "valid"


def _slot(
    tmp_path: Path, worker_id: int = 0, *, timeout=100.0, grace=0.0, proc=None
) -> W.WorkerSlot:
    log = tmp_path / f"worker_{worker_id:03d}.log"
    log.write_text("start\n")
    monitor = W.StallMonitor(timeout_sec=timeout, startup_grace_sec=grace)
    monitor.launched(0.0)
    return W.WorkerSlot(
        worker_id=worker_id,
        start_idx=0,
        end_idx=10,
        gpu=0,
        log_path=str(log),
        proc=proc or FakeProc(),
        monitor=monitor,
    )


def _touch(path: Path, mtime: float) -> None:
    path.touch()
    os.utime(path, (mtime, mtime))


def test_latest_mtime_ignores_missing(tmp_path: Path):
    a = tmp_path / "a"
    _touch(a, 10.0)
    assert W.latest_mtime([str(a), str(tmp_path / "missing")]) == 10.0
    assert W.latest_mtime([str(tmp_path / "missing")]) is None


def test_stall_monitor_uses_activity_and_startup_grace():
    m = W.StallMonitor(timeout_sec=100.0, startup_grace_sec=500.0)
    m.launched(0.0)
    m.observe(1000.0, now=5.0)  # Kit banner right after launch must not consume the grace
    assert not m.stalled(550.0)  # quiet since 5 s but still inside timeout + grace = 600
    assert m.stalled(601.0)
    m.observe(1001.0, now=620.0)  # first frame
    assert not m.stalled(700.0)
    assert m.stalled(721.0)  # 101 s idle, startup window long over
    m.observe(1001.0, now=800.0)  # unchanged mtime is not activity
    assert m.stalled(800.0)
    m.observe(1002.0, now=800.0)
    assert not m.stalled(850.0)
    m.launched(900.0)  # relaunch reopens the startup window
    assert not m.stalled(1400.0) and m.stalled(1501.0)


def test_supervise_relaunches_stalled_worker_and_respects_max_restarts(tmp_path: Path):
    slot = _slot(tmp_path, timeout=100.0)
    log = Path(slot.log_path)
    _touch(log, 1000.0)
    relaunched = []
    messages = []

    def relaunch(s):
        s.proc.killed = True
        relaunched.append(s.worker_id)
        return FakeProc()

    kw = dict(
        outdir=str(tmp_path),
        validate_complete=lambda s: _valid_artifacts(s, tmp_path),
        stop=_stop,
        relaunch=relaunch,
        max_restarts=1,
        log=messages.append,
    )
    assert (
        W.supervise([slot], now=50.0, **kw) == [] and relaunched == []
    )  # first observation, mtime recorded
    _touch(log, 1001.0)
    assert (
        W.supervise([slot], now=140.0, **kw) == [] and relaunched == []
    )  # activity at 140 resets the clock
    assert W.supervise([slot], now=230.0, **kw) == [] and relaunched == []  # 90 s idle < 100
    old_proc = slot.proc
    assert W.supervise([slot], now=241.0, **kw) == [] and relaunched == [0]  # stalled -> relaunched
    assert old_proc.killed and slot.proc is not old_proc and slot.monitor.restarts == 1
    assert "no progress for 2 min; relaunching [0, 10) with --resume (restart 1/1)" in messages[-1]
    # after the relaunch the startup window starts over; no restarts once the budget is spent
    assert W.supervise([slot], now=300.0, **kw) == []
    assert W.supervise([slot], now=400.0, **kw) == [0] and relaunched == [0]
    assert "giving up" in messages[-1] and slot.gave_up
    assert slot.proc.killed and slot.proc.poll() is not None
    assert (
        W.supervise([slot], now=420.0, **kw) == [] and len(messages) == 2
    )  # not repeated every pass


def test_supervise_relaunches_exited_worker_without_output(tmp_path: Path):
    """A worker can exit 0 after an exception: a missing annotations file is the failure signal."""
    slot = _slot(tmp_path, proc=FakeProc(running=False))
    slot.proc.returncode = 0
    relaunched = []
    messages = []
    kw = dict(
        outdir=str(tmp_path),
        validate_complete=lambda s: _valid_artifacts(s, tmp_path),
        stop=_stop,
        relaunch=lambda s: relaunched.append(s.worker_id) or FakeProc(),
        max_restarts=2,
        log=messages.append,
    )
    assert W.supervise([slot], now=10.0, **kw) == [] and relaunched == [0]
    assert "exited without valid artifacts" in messages[-1]
    # the relaunched process runs and finishes properly -> untouched
    slot.proc.running = False
    slot.proc.returncode = 0
    Path(slot.final_path(str(tmp_path))).write_text("valid")
    assert W.supervise([slot], now=20.0, **kw) == [] and relaunched == [0] and len(messages) == 1


def test_supervise_skips_finished_workers(tmp_path: Path):
    slot = _slot(tmp_path, proc=FakeProc(running=False))
    slot.proc.returncode = 0
    Path(slot.final_path(str(tmp_path))).write_text("valid")
    called = []
    assert (
        W.supervise(
            [slot],
            outdir=str(tmp_path),
            validate_complete=lambda s: _valid_artifacts(s, tmp_path),
            stop=_stop,
            relaunch=lambda s: called.append(s),
            max_restarts=3,
            now=1e9,
            log=lambda m: None,
        )
        == []
    )
    assert called == []


def test_watched_paths_cover_log_and_checkpoints(tmp_path: Path):
    slot = _slot(tmp_path, worker_id=7)
    names = [os.path.basename(p) for p in slot.watched_paths(str(tmp_path))]
    assert names == [
        "worker_007.log",
        "annotations_worker_007.partial.json",
        "annotations_worker_007.json",
    ]


@pytest.mark.parametrize(
    "code,valid,failed", [(0, True, False), (0, False, True), (9, True, True), (9, False, True)]
)
def test_exit_status_and_validated_artifacts_define_completion(tmp_path, code, valid, failed):
    slot = _slot(tmp_path, timeout=0, proc=FakeProc(running=False))
    slot.proc.returncode = code
    # Presence alone must never count as completion.
    Path(slot.final_path(str(tmp_path))).write_text("{}")
    result = W.supervise(
        [slot],
        outdir=str(tmp_path),
        relaunch=lambda s: pytest.fail("retry disallowed"),
        validate_complete=lambda s: valid,
        stop=_stop,
        max_restarts=0,
        now=100,
    )
    assert result == ([0] if failed else [])
    assert slot.gave_up is failed
    assert slot.completed is (not failed)
    if failed:
        assert (
            "exited with code 9" if code else "exited without valid artifacts"
        ) in slot.failure_reason


def test_zero_timeout_disables_only_running_stall_detection(tmp_path):
    slot = _slot(tmp_path, timeout=0)
    kwargs = dict(
        outdir=str(tmp_path),
        relaunch=lambda s: pytest.fail("retry disallowed"),
        validate_complete=lambda s: False,
        stop=_stop,
        max_restarts=0,
    )
    assert W.supervise([slot], now=1, **kwargs) == []
    assert W.supervise([slot], now=1e9, **kwargs) == []
    slot.proc.running = False
    slot.proc.returncode = 0
    assert W.supervise([slot], now=1e9 + 1, **kwargs) == [0]


def test_retry_stops_previous_process_before_relaunch(tmp_path):
    slot = _slot(tmp_path, timeout=1)
    events = []

    def stop(proc):
        events.append("stop")
        _stop(proc)

    def relaunch(s):
        assert s.proc.poll() is not None
        events.append("launch")
        return FakeProc()

    kwargs = dict(
        outdir=str(tmp_path),
        relaunch=relaunch,
        validate_complete=lambda s: False,
        stop=stop,
        max_restarts=1,
    )
    W.supervise([slot], now=0, **kwargs)
    W.supervise([slot], now=2, **kwargs)
    assert events == ["stop", "launch"]


def test_artifact_validation_error_is_a_terminal_failure(tmp_path):
    slot = _slot(tmp_path, proc=FakeProc(running=False))

    def invalid(s):
        raise ValueError("truncated JSON")

    assert W.supervise(
        [slot],
        outdir=str(tmp_path),
        relaunch=lambda s: None,
        validate_complete=invalid,
        stop=_stop,
        max_restarts=0,
        now=1,
    ) == [0]
    assert slot.gave_up


def test_terminate_process_group_reaps_worker_and_child(tmp_path):
    ready = tmp_path / "ready"
    # The worker handles TERM by reaping its child. Both receive the group signal.
    child_code = "import time; time.sleep(60)"
    worker_code = """
import signal, subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', sys.argv[2]])
def stop(signum, frame):
    child.wait(timeout=2)
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
Path(sys.argv[1]).write_text(str(child.pid))
time.sleep(60)
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", worker_code, str(ready), child_code], start_new_session=True
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        child_pid = int(ready.read_text())
        started = time.monotonic()
        slot = _slot(tmp_path, timeout=1, proc=proc)
        options = dict(
            outdir=str(tmp_path),
            validate_complete=lambda s: False,
            relaunch=lambda s: pytest.fail("retry budget exhausted"),
            max_restarts=0,
            stop=lambda p: W.terminate_process_group(p, term_timeout_sec=1, kill_timeout_sec=1),
        )
        assert W.supervise([slot], now=0, **options) == []
        assert W.supervise([slot], now=2, **options) == [0]
        assert slot.gave_up
        assert time.monotonic() - started < 3
        assert proc.poll() is not None
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        with pytest.raises(ProcessLookupError):
            os.killpg(proc.pid, 0)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=2)


def test_terminate_escalates_when_worker_ignores_term(tmp_path):
    ready = tmp_path / "ready"
    code = (
        "import signal,time,sys; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); Path(sys.argv[1]).touch(); time.sleep(60)"
    )
    proc = subprocess.Popen([sys.executable, "-c", code, str(ready)], start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        W.terminate_process_group(proc, term_timeout_sec=0.05, kill_timeout_sec=1)
        assert proc.returncode == -signal.SIGKILL
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=2)


def test_missing_leader_never_signals_a_possibly_reused_group(monkeypatch):
    from types import SimpleNamespace

    calls = []

    def gone(pid):
        raise ProcessLookupError

    monkeypatch.setattr(W.os, "getpgid", gone)
    monkeypatch.setattr(W.os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))
    proc = SimpleNamespace(
        pid=424242, poll=lambda: 0, wait=lambda timeout: calls.append(("wait", timeout))
    )
    W.terminate_process_group(proc, term_timeout_sec=0, kill_timeout_sec=1)
    assert calls == [("wait", 1)]


def test_cleanup_rejects_worker_without_own_process_group(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(W.os, "getpgid", lambda pid: pid + 1)
    monkeypatch.setattr(W.os, "killpg", lambda *args: pytest.fail("must not signal shared group"))
    with pytest.raises(ValueError, match="start_new_session"):
        W.terminate_process_group(SimpleNamespace(pid=424242))


def test_reaped_leader_never_signals_even_if_pid_has_been_reused(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(W.os, "getpgid", lambda pid: pytest.fail("reaped PID is not owned"))
    monkeypatch.setattr(W.os, "killpg", lambda *args: pytest.fail("must not signal"))
    W.terminate_process_group(SimpleNamespace(pid=424242, returncode=0))


def test_exited_leader_stays_unreaped_through_last_group_signal(tmp_path, monkeypatch):
    proc = subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"], start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while W.poll_without_reaping(proc) is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert W.poll_without_reaping(proc) == 7
        assert proc.returncode is None
        original = os.killpg
        signals = []

        def owned_signal(pgid, sig):
            # The exited child still exists as a zombie and pins this PID.
            assert (
                os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT).si_pid
                == proc.pid
            )
            assert os.getpgid(proc.pid) == pgid
            signals.append(sig)
            return original(pgid, sig)

        monkeypatch.setattr(W.os, "killpg", owned_signal)
        W.terminate_process_group(proc, term_timeout_sec=0, kill_timeout_sec=1)
        assert signal.SIGTERM in signals and signal.SIGKILL in signals
        assert proc.returncode == 7
    finally:
        proc.wait(timeout=2)
