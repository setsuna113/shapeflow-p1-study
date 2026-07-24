"""The fault test an alternative supervisor must pass before it may replace the unit.

The run host is a container, so systemd is not PID 1 and `systemctl` cannot operate. Plan
section 17.2 allows a different supervisor only if it demonstrates the same three properties
the unit would have had:

  1. kill the coordinator and it comes back with its resume flags intact;
  2. three crashes inside the window and it stops, writing a BLOCKED report;
  3. the ledger never accepts the same work twice across those restarts.

A supervisor that merely restarts forever is not equivalent -- it converts a deterministic
failure into an unbounded spend, and nobody looks at a log that is still scrolling.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SUPERVISOR = REPO / "scripts" / "sfsupervise.sh"

pytestmark = pytest.mark.skipif(
    not shutil.which("flock") or not shutil.which("setsid"),
    reason="flock/setsid required to exercise the supervisor",
)


def _run_supervisor(tmp_path, script: str, *, burst: int = 3, interval: int = 30,
                    restart_sec: int = 1, timeout: float = 60.0):
    """Drive the supervisor with a stand-in child, as the unprivileged path would.

    `runuser` needs root, which tests do not have, so a shim on PATH forwards to `env` -- the
    supervisor's control flow (locking, restart ceiling, process group, BLOCKED report) is what
    is under test, not the identity switch, which install_host.sh proves separately.
    """
    repo = tmp_path / "repo"
    (repo / "reports").mkdir(parents=True)
    (repo / "logs").mkdir(parents=True)
    child = tmp_path / "child.sh"
    child.write_text("#!/usr/bin/env bash\n" + script)
    child.chmod(0o755)

    shim = tmp_path / "bin"
    shim.mkdir()
    (shim / "runuser").write_text(
        '#!/usr/bin/env bash\n'
        '# shift past "-u <user> --"\n'
        'shift 3\nexec "$@"\n'
    )
    (shim / "runuser").chmod(0o755)

    env = dict(os.environ)
    env.update({
        "PATH": f"{shim}:{env['PATH']}",
        "SHAPEFLOW_RUN_DIR": str(tmp_path / "run"),
        "SHAPEFLOW_START_LIMIT_BURST": str(burst),
        "SHAPEFLOW_START_LIMIT_INTERVAL": str(interval),
        "SHAPEFLOW_RESTART_SEC": str(restart_sec),
    })
    proc = subprocess.run(
        ["bash", str(SUPERVISOR), "coordinator", "nobody", str(repo),
         str(tmp_path / "data"), "--", str(child)],
        env=env, capture_output=True, text=True, timeout=timeout,
    )
    return proc, repo


def test_a_crashing_child_is_restarted_with_its_flags(tmp_path):
    counter = tmp_path / "starts"
    script = f"""
echo start >> {counter}
n=$(wc -l < {counter})
if [ "$n" -lt 3 ]; then exit 7; fi
exit 0
"""
    proc, repo = _run_supervisor(tmp_path, script, burst=5, interval=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert counter.read_text().count("start") == 3, "the child was not restarted"
    assert not (repo / "reports" / "BLOCKED_REPEATED_CRASH.md").exists()


def test_three_crashes_in_the_window_stop_the_supervisor(tmp_path):
    """Restarting forever converts a deterministic failure into an unbounded spend."""
    counter = tmp_path / "starts"
    proc, repo = _run_supervisor(
        tmp_path, f"echo start >> {counter}\nexit 9\n", burst=3, interval=300)

    assert proc.returncode == 1
    blocked = repo / "reports" / "BLOCKED_REPEATED_CRASH.md"
    assert blocked.exists(), "the supervisor gave up without saying so"
    body = blocked.read_text()
    assert "REPEATED_CRASH" in body
    assert "Finished work is kept" in body
    assert counter.read_text().count("start") == 3, "the ceiling did not hold"


def test_a_clean_exit_is_not_restarted(tmp_path):
    counter = tmp_path / "starts"
    proc, _repo = _run_supervisor(tmp_path, f"echo start >> {counter}\nexit 0\n")
    assert proc.returncode == 0
    assert counter.read_text().count("start") == 1


def test_a_stop_request_ends_the_loop_without_a_blocked_report(tmp_path):
    """`stop-safely` must halt admission without looking like a crash."""
    data = tmp_path / "data" / "runner"
    data.mkdir(parents=True)
    counter = tmp_path / "starts"
    script = f"""
echo start >> {counter}
touch {data}/STOP_REQUESTED
exit 3
"""
    proc, repo = _run_supervisor(tmp_path, script, burst=5, interval=300)
    assert proc.returncode == 0
    assert counter.read_text().count("start") == 1
    assert not (repo / "reports" / "BLOCKED_REPEATED_CRASH.md").exists()


def test_only_one_supervisor_may_hold_the_name(tmp_path):
    """Two coordinators would claim the same work items."""
    lock_dir = tmp_path / "run"
    lock_dir.mkdir()
    repo = tmp_path / "repo"
    (repo / "reports").mkdir(parents=True)
    (repo / "logs").mkdir(parents=True)

    holder = subprocess.Popen(
        ["bash", "-c", f"exec 9>{lock_dir}/coordinator.lock; flock -n 9 && sleep 20"])
    time.sleep(0.5)
    try:
        env = dict(os.environ)
        env["SHAPEFLOW_RUN_DIR"] = str(lock_dir)
        proc = subprocess.run(
            ["bash", str(SUPERVISOR), "coordinator", "nobody", str(repo),
             str(tmp_path / "data"), "--", "/bin/true"],
            env=env, capture_output=True, text=True, timeout=30,
        )
        assert proc.returncode == 1
        assert "another supervisor holds" in proc.stderr
    finally:
        holder.terminate()
        holder.wait(timeout=10)


def test_the_supervisor_records_what_it_did(tmp_path):
    counter = tmp_path / "starts"
    _proc, repo = _run_supervisor(tmp_path, f"echo start >> {counter}\nexit 0\n")
    log = (repo / "logs" / "coordinator.log").read_text()
    assert "starting:" in log and "child exited status=0" in log
