"""Phase FSM and the coordinator loop (execution, retry, resume, idempotency)."""

from __future__ import annotations

import pytest

from shapeflow_p1.experiment.coordinator import (
    Coordinator,
    FatalProtocolError,
    RetryableError,
)
from shapeflow_p1.experiment.ledger import Ledger
from shapeflow_p1.experiment.state_machine import (
    IllegalPhaseTransition,
    Phase,
    PhaseMachine,
    can_transition,
)
from shapeflow_p1.object_store import ObjectStore


# --- phase FSM ------------------------------------------------------------------------


def test_phase_happy_path_to_holdout():
    m = PhaseMachine()
    for nxt in [
        Phase.DOCTOR_PASSED, Phase.ACQUISITION_COMPLETE, Phase.SNAPSHOTS_FROZEN,
        Phase.P0_PARITY_PASSED, Phase.GPU_SMOKE_PASSED, Phase.SCREEN_RUNNING,
        Phase.SCREEN_COMPLETE, Phase.MINI_ITT_RUNNING, Phase.MINI_ITT_COMPLETE,
        Phase.POLICY_FROZEN, Phase.HOLDOUT_RUNNING, Phase.HOLDOUT_COMPLETE,
        Phase.REPORT_COMPLETE,
    ]:
        m.transition(nxt)
    assert m.phase is Phase.REPORT_COMPLETE


def test_cannot_skip_parity_gate_to_screen():
    assert not can_transition(Phase.SNAPSHOTS_FROZEN, Phase.SCREEN_RUNNING)
    m = PhaseMachine(Phase.SNAPSHOTS_FROZEN)
    with pytest.raises(IllegalPhaseTransition):
        m.transition(Phase.SCREEN_RUNNING)


def test_cannot_open_holdout_before_policy_freeze():
    assert not can_transition(Phase.MINI_ITT_COMPLETE, Phase.HOLDOUT_RUNNING)


def test_screen_can_branch_to_characterize_no_go():
    assert can_transition(Phase.SCREEN_COMPLETE, Phase.CHARACTERIZE_NO_GO_RUNNING)
    assert can_transition(Phase.SCREEN_COMPLETE, Phase.MINI_ITT_RUNNING)


def test_blocked_reachable_from_any_nonterminal():
    m = PhaseMachine(Phase.SCREEN_RUNNING)
    m.transition(Phase.BLOCKED)
    assert m.is_terminal


# --- coordinator ----------------------------------------------------------------------


def _setup(tmp_path, clock=None):
    ledger = Ledger(str(tmp_path / "l.sqlite"), clock=clock or (lambda: 1.0))
    store = ObjectStore(tmp_path / "obj")
    return ledger, store


def _make_items(ledger, n, **over):
    keys = []
    for i in range(n):
        keys.append(ledger.ensure_work_item(
            protocol_sha="p", split="SCREEN", phase_id="screen", task_id=f"t{i}",
            arm_id="H", variant_id="H02", **over,
        ))
    return keys


def test_happy_path_commits_all(tmp_path):
    ledger, store = _setup(tmp_path)
    _make_items(ledger, 3)

    def executor(item):
        return f"output for {item.task_id}".encode()

    coord = Coordinator(ledger, store, executor=executor)
    summary = coord.run()
    assert summary.committed == 3
    assert ledger.state_counts().get("COMMITTED") == 3
    # every committed artifact verifies
    for i in range(3):
        key = ledger.work_key(protocol_sha="p", split="SCREEN", phase_id="screen",
                              task_id=f"t{i}", arm_id="H", variant_id="H02",
                              replicate_id="0", checkpoint_hash="-", stage_version="v1")
        assert coord.committed_artifact_valid(key)


def test_running_twice_does_not_recommit(tmp_path):
    ledger, store = _setup(tmp_path)
    _make_items(ledger, 2)
    coord = Coordinator(ledger, store, executor=lambda it: b"x")
    coord.run()
    again = coord.run()  # resume: nothing left to do
    assert again.committed == 0
    assert ledger.state_counts().get("COMMITTED") == 2


def test_retryable_failure_then_success(tmp_path):
    ledger, store = _setup(tmp_path)
    _make_items(ledger, 1)
    calls = {"n": 0}

    def flaky(item):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RetryableError("transient")
        return b"ok"

    coord = Coordinator(ledger, store, executor=flaky)
    coord.run(max_items=5)
    assert ledger.state_counts().get("COMMITTED") == 1
    assert calls["n"] == 2  # failed once, retried, succeeded


def test_fatal_error_fails_final_and_records_incident(tmp_path):
    ledger, store = _setup(tmp_path)
    _make_items(ledger, 1)

    def boom(item):
        raise FatalProtocolError("schema not closed")

    coord = Coordinator(ledger, store, executor=boom)
    summary = coord.run()
    assert summary.failed_final == 1
    assert ledger.state_counts().get("FAILED_FINAL") == 1
    # incident recorded
    row = ledger.raw_connection.execute("SELECT COUNT(*) c FROM incidents").fetchone()
    assert row["c"] == 1


def test_resume_reopens_a_crashed_in_progress_item(tmp_path):
    # Simulate a crash: claim + materialize but never commit, lease expires, then resume.
    clock = type("C", (), {"t": 100.0})()
    ledger = Ledger(str(tmp_path / "l.sqlite"), clock=lambda: clock.t)
    store = ObjectStore(tmp_path / "obj")
    key = _make_items(ledger, 1)[0]

    att = ledger.claim(key, "dead-worker", lease_seconds=30)
    ledger.advance(att.attempt_id, "MATERIALIZED")  # crashed here, before commit
    clock.t += 31  # lease expires

    coord = Coordinator(ledger, store, executor=lambda it: b"recovered")
    summary = coord.run()
    assert summary.committed == 1  # reopened and completed
    assert ledger.state_counts().get("COMMITTED") == 1
