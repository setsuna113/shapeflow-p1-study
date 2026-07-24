"""Work-item state machine: idempotency, legal transitions, lease reclaim, resume."""

from __future__ import annotations

import pytest

from shapeflow_p1.experiment.ledger import IllegalTransition, Ledger


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def tick(self, dt):
        self.t += dt


def _ledger(tmp_path, clock=None):
    return Ledger(str(tmp_path / "ledger.sqlite"), clock=clock or (lambda: 1000.0))


def _item(ledger, **over):
    coords = dict(
        protocol_sha="p", split="SCREEN", phase_id="screen", task_id="t1",
        arm_id="H", variant_id="H02",
    )
    coords.update(over)
    return ledger.ensure_work_item(**coords)


def test_ensure_work_item_is_idempotent(tmp_path):
    lg = _ledger(tmp_path)
    a = _item(lg)
    b = _item(lg)
    assert a == b
    assert lg.state_counts() == {"PENDING": 1}


def test_happy_path_to_committed(tmp_path):
    lg = _ledger(tmp_path)
    key = _item(lg)
    att = lg.claim(key, "w1", lease_seconds=60)
    assert att is not None
    lg.advance(att.attempt_id, "MATERIALIZED")
    lg.advance(att.attempt_id, "VALIDATED")
    lg.commit(att.attempt_id, result_object_ref="a" * 64)
    assert lg.is_committed(key)
    assert lg.committed_ref(key) == "a" * 64


def test_second_claim_while_claimed_returns_none(tmp_path):
    lg = _ledger(tmp_path)
    key = _item(lg)
    first = lg.claim(key, "w1", lease_seconds=60)
    assert first is not None
    assert lg.claim(key, "w2", lease_seconds=60) is None  # not claimable


def test_illegal_transition_rejected(tmp_path):
    lg = _ledger(tmp_path)
    key = _item(lg)
    att = lg.claim(key, "w1", lease_seconds=60)
    # CLAIMED -> VALIDATED skips MATERIALIZED and must be refused.
    with pytest.raises(IllegalTransition):
        lg.advance(att.attempt_id, "VALIDATED")


def test_cannot_commit_before_validated(tmp_path):
    lg = _ledger(tmp_path)
    key = _item(lg)
    att = lg.claim(key, "w1", lease_seconds=60)
    lg.advance(att.attempt_id, "MATERIALIZED")
    with pytest.raises(IllegalTransition):
        lg.commit(att.attempt_id, result_object_ref="a" * 64)


def test_retryable_failure_reopens_then_exhausts(tmp_path):
    lg = Ledger(str(tmp_path / "l.sqlite"), clock=lambda: 1.0, default_max_retries=2)
    key = _item(lg)
    for expected_retry in range(2):
        att = lg.claim(key, "w", lease_seconds=60)
        assert att is not None, f"should be claimable on retry {expected_retry}"
        state = lg.fail(att.attempt_id, disposition="FAILED_RETRYABLE", error_class="net")
        assert state == "PENDING"
    # third failure exhausts the 2 retries -> FAILED_FINAL
    att = lg.claim(key, "w", lease_seconds=60)
    state = lg.fail(att.attempt_id, disposition="FAILED_RETRYABLE", error_class="net")
    assert state == "FAILED_FINAL"
    assert lg.claim(key, "w", lease_seconds=60) is None


def test_side_effecting_unknown_failure_freezes(tmp_path):
    lg = _ledger(tmp_path)
    key = _item(lg, task_id="ext", side_effecting=True)
    att = lg.claim(key, "w", lease_seconds=60)
    state = lg.fail(att.attempt_id, disposition="FAILED_UNKNOWN", error_class="timeout")
    assert state == "FAILED_UNKNOWN"
    # frozen: not claimable again, because a possible external effect must not be repeated
    assert lg.claim(key, "w", lease_seconds=60) is None


def test_stale_lease_reclaims_pure_item_but_freezes_side_effecting(tmp_path):
    clock = FakeClock()
    lg = Ledger(str(tmp_path / "l.sqlite"), clock=clock)
    pure = _item(lg, task_id="pure", side_effecting=False)
    ext = _item(lg, task_id="ext", side_effecting=True)
    lg.claim(pure, "w", lease_seconds=30)
    lg.claim(ext, "w", lease_seconds=30)

    clock.tick(31)  # both leases now expired
    reopened = lg.reclaim_stale()
    assert pure in reopened and ext not in reopened
    # pure is claimable again; ext is frozen FAILED_UNKNOWN
    assert lg.claim(pure, "w2", lease_seconds=30) is not None
    assert lg.get_work_item(ext).state == "FAILED_UNKNOWN"


def test_committed_item_survives_reopen_as_terminal(tmp_path):
    """Resume relies on this: a COMMITTED item is never reclaimed or re-run."""
    clock = FakeClock()
    lg = Ledger(str(tmp_path / "l.sqlite"), clock=clock)
    key = _item(lg)
    att = lg.claim(key, "w", lease_seconds=30)
    lg.advance(att.attempt_id, "MATERIALIZED")
    lg.advance(att.attempt_id, "VALIDATED")
    lg.commit(att.attempt_id, result_object_ref="c" * 64)
    clock.tick(1000)
    assert lg.reclaim_stale() == []
    assert lg.is_committed(key)


def test_resume_across_reopen_sees_committed(tmp_path):
    path = str(tmp_path / "l.sqlite")
    key_val = None
    lg = Ledger(path)
    key_val = _item(lg)
    att = lg.claim(key_val, "w", lease_seconds=30)
    lg.advance(att.attempt_id, "MATERIALIZED")
    lg.advance(att.attempt_id, "VALIDATED")
    lg.commit(att.attempt_id, result_object_ref="d" * 64)
    lg.close()

    # Reopen the database (simulating a coordinator restart).
    lg2 = Ledger(path)
    assert lg2.is_committed(key_val)
    assert lg2.committed_ref(key_val) == "d" * 64
    assert lg2.integrity_check()
    lg2.close()


def test_blocked_budget_is_terminal(tmp_path):
    lg = _ledger(tmp_path)
    key = _item(lg)
    lg.block_budget(key, reason="tavily_credits")
    assert lg.get_work_item(key).state == "BLOCKED_BUDGET"
    assert lg.claim(key, "w", lease_seconds=30) is None
