"""Budget admission control and the external-call state machine."""

from __future__ import annotations

import pytest

from shapeflow_p1.experiment.budget import Budget, BudgetCapRaised, BudgetExceeded
from shapeflow_p1.experiment.ledger import Ledger
from shapeflow_p1.object_store import ObjectStore
from shapeflow_p1.providers.external_call_ledger import (
    CallAlreadyCommitted,
    CallNotReplayable,
    ExternalCallLedger,
    IllegalCallTransition,
)
from shapeflow_p1.secrets import SecretRedactor


def _budget(tmp_path, caps):
    lg = Ledger(str(tmp_path / "l.sqlite"), clock=lambda: 5.0)
    b = Budget(lg)
    for resource, cap in caps.items():
        b.ensure_account(resource, cap)
    return lg, b


def test_reserve_then_settle_releases_surplus(tmp_path):
    _, b = _budget(tmp_path, {"tavily_requests": 10})
    grp = b.reserve({"tavily_requests": 3})
    assert b.available("tavily_requests") == 7  # 3 reserved
    b.settle(grp, {"tavily_requests": 1})       # actual was 1
    assert b.available("tavily_requests") == 9  # 2 surplus released, 1 committed


def test_reservation_is_admission_control(tmp_path):
    _, b = _budget(tmp_path, {"usd": 10.0})
    b.reserve({"usd": 8.0})
    # Only 2.0 headroom; a 3.0 reservation must be refused BEFORE any dispatch.
    with pytest.raises(BudgetExceeded) as ei:
        b.reserve({"usd": 3.0})
    assert ei.value.resource == "usd"
    assert ei.value.available == 2.0


def test_multi_resource_reserve_is_all_or_nothing(tmp_path):
    _, b = _budget(tmp_path, {"requests": 10, "credits": 1})
    # credits cap is 1; asking for 2 must roll back the requests leg too.
    with pytest.raises(BudgetExceeded) as ei:
        b.reserve({"requests": 5, "credits": 2})
    assert ei.value.resource == "credits"
    assert b.available("requests") == 10  # requests leg was rolled back
    assert b.available("credits") == 1


def test_keep_worst_case_charges_full_reservation(tmp_path):
    _, b = _budget(tmp_path, {"usd": 10.0})
    grp = b.reserve({"usd": 4.0})
    # Timeout after send: we don't know if it was billed, so keep the worst case.
    b.keep_worst_case(grp)
    assert b.available("usd") == 6.0  # 4.0 permanently committed, not released


def test_release_returns_everything(tmp_path):
    _, b = _budget(tmp_path, {"usd": 10.0})
    grp = b.reserve({"usd": 4.0})
    b.release(grp)
    assert b.available("usd") == 10.0


def test_accounting_invariant_never_exceeds_cap(tmp_path):
    _, b = _budget(tmp_path, {"tok": 100})
    grps = [b.reserve({"tok": 20}) for _ in range(5)]  # exactly fills cap
    with pytest.raises(BudgetExceeded):
        b.reserve({"tok": 1})
    # settle some low, some worst-case; availability must stay in [0, cap]
    b.settle(grps[0], {"tok": 5})
    b.keep_worst_case(grps[1])
    avail = b.available("tok")
    assert 0 <= avail <= 100


def test_external_call_fsm_happy_path_redacts_request(tmp_path):
    lg = Ledger(str(tmp_path / "l.sqlite"), clock=lambda: 5.0)
    b = Budget(lg)
    b.ensure_account("requests", 10)
    store = ObjectStore(tmp_path / "obj")
    red = SecretRedactor()
    red.register("tvly-FAKEFAKEFAKEFAKEFAKE", label="tavily")
    ecl = ExternalCallLedger(lg, b, store, red)

    call = ecl.open_call(provider="tavily", op_class="search", call_key="q1")
    assert ecl.get_state(call) == "OPEN"
    attempt = ecl.begin_attempt(call)
    assert ecl.attempt_state(attempt.attempt_id) == "INTENT"
    grp = ecl.reserve(attempt, {"requests": 1})
    assert ecl.attempt_state(attempt.attempt_id) == "BUDGET_RESERVED"
    # The request text carries the key; only the redacted form may be stored.
    ecl.mark_sent(attempt, request_text="POST search key=tvly-FAKEFAKEFAKEFAKEFAKE")
    ecl.store_response(attempt, response_text="{\"results\": []}", provider_request_id="rq1")
    ecl.validate(attempt)
    ecl.commit(attempt, grp, {"requests": 1})
    assert ecl.get_state(call) == "COMMITTED"
    assert ecl.attempt_state(attempt.attempt_id) == "COMMITTED"

    # Prove no stored blob contains the secret.
    for blob in (tmp_path / "obj").rglob("*.zst"):
        raw = store.get_bytes(blob.stem)
        assert b"tvly-FAKEFAKEFAKEFAKEFAKE" not in raw


def test_external_call_budget_refusal_marks_failed(tmp_path):
    lg = Ledger(str(tmp_path / "l.sqlite"), clock=lambda: 5.0)
    b = Budget(lg)
    b.ensure_account("requests", 0)  # no headroom at all
    ecl = ExternalCallLedger(lg, b, ObjectStore(tmp_path / "obj"), SecretRedactor())
    call = ecl.open_call(provider="tavily", op_class="search", call_key="q1")
    attempt = ecl.begin_attempt(call)
    with pytest.raises(BudgetExceeded):
        ecl.reserve(attempt, {"requests": 1})
    assert ecl.attempt_state(attempt.attempt_id) == "FAILED_FINAL"


def test_external_call_timeout_after_send_is_unknown_and_keeps_cost(tmp_path):
    lg = Ledger(str(tmp_path / "l.sqlite"), clock=lambda: 5.0)
    b = Budget(lg)
    b.ensure_account("usd", 10.0)
    ecl = ExternalCallLedger(lg, b, ObjectStore(tmp_path / "obj"), SecretRedactor())
    call = ecl.open_call(provider="deepseek", op_class="judge", call_key="j1")
    attempt = ecl.begin_attempt(call)
    grp = ecl.reserve(attempt, {"usd": 2.0})
    ecl.mark_sent(attempt, request_text="judge request")
    # ...silence...
    ecl.fail_unknown(attempt, grp, error_class="timeout")
    assert ecl.attempt_state(attempt.attempt_id) == "FAILED_UNKNOWN"
    assert b.available("usd") == 8.0  # worst-case kept, not released


# --- logical call vs physical attempt -------------------------------------------------------


def _ecl(tmp_path, **caps):
    lg = Ledger(str(tmp_path / "l.sqlite"), clock=lambda: 5.0)
    b = Budget(lg)
    for resource, cap in caps.items():
        b.ensure_account(resource, cap)
    return lg, b, ExternalCallLedger(lg, b, ObjectStore(tmp_path / "obj"), SecretRedactor())


def test_a_retry_is_a_new_attempt_and_never_edits_the_first(tmp_path):
    """Two dispatches leave two rows. The first attempt's request ref, error class and
    settlement survive the second, because a retry is a second possible charge."""
    lg, b, ecl = _ecl(tmp_path, usd=10.0)
    call = ecl.open_call(provider="deepseek", op_class="judge", call_key="j1")

    first = ecl.begin_attempt(call)
    grp = ecl.reserve(first, {"usd": 1.0})
    ecl.mark_sent(first, request_text="attempt one")
    ecl.fail_after_response(first, grp, {"usd": 1.0}, error_class="http_500")

    second = ecl.begin_attempt(call)
    grp2 = ecl.reserve(second, {"usd": 1.0})
    ecl.mark_sent(second, request_text="attempt two")
    ecl.store_response(second, response_text="{}", returned_model="deepseek-v4-flash")
    ecl.validate(second)
    ecl.commit(second, grp2, {"usd": 0.5})

    rows = lg.raw_connection.execute(
        "SELECT attempt_ordinal, state, error_class, request_object_ref FROM"
        " external_call_attempts WHERE call_id=? ORDER BY attempt_ordinal", (call,)
    ).fetchall()
    assert [r["attempt_ordinal"] for r in rows] == [0, 1]
    assert rows[0]["state"] == "FAILED_FINAL" and rows[0]["error_class"] == "http_500"
    assert rows[1]["state"] == "COMMITTED"
    assert rows[0]["request_object_ref"] != rows[1]["request_object_ref"]
    # Both dispatches are on the bill: 1.0 for the failure, 0.5 for the success.
    assert b.available("usd") == pytest.approx(8.5)


def test_a_committed_call_refuses_a_second_attempt_and_hands_back_its_response(tmp_path):
    lg, b, ecl = _ecl(tmp_path, usd=10.0)
    call = ecl.open_call(provider="deepseek", op_class="judge", call_key="j1")
    attempt = ecl.begin_attempt(call)
    grp = ecl.reserve(attempt, {"usd": 1.0})
    ecl.mark_sent(attempt, request_text="q")
    ecl.store_response(attempt, response_text='{"answer": 42}')
    ecl.validate(attempt)
    ecl.commit(attempt, grp, {"usd": 0.25})

    with pytest.raises(CallAlreadyCommitted) as excinfo:
        ecl.begin_attempt(call)
    assert excinfo.value.response_object_ref == ecl.committed_response_ref(call)
    assert b.available("usd") == pytest.approx(9.75), "a replay must not cost anything"


def test_a_failed_attempt_may_be_retried_but_an_abandoned_call_may_not(tmp_path):
    """A timeout keeps its worst-case charge and still allows a retry (§3.5); giving up on
    the call is a separate, recorded decision that closes it for good."""
    lg, b, ecl = _ecl(tmp_path, usd=10.0)
    call = ecl.open_call(provider="deepseek", op_class="judge", call_key="j1")
    attempt = ecl.begin_attempt(call)
    grp = ecl.reserve(attempt, {"usd": 1.0})
    ecl.mark_sent(attempt, request_text="q")
    ecl.fail_unknown(attempt, grp, error_class="timeout")
    assert ecl.get_state(call) == "OPEN"
    ecl.begin_attempt(call)  # a retry is allowed, as its own attempt

    ecl.close_call(call, reason="retries_exhausted")
    with pytest.raises(CallNotReplayable):
        ecl.begin_attempt(call)


def test_an_attempt_cannot_move_backwards_or_be_committed_twice(tmp_path):
    lg, b, ecl = _ecl(tmp_path, usd=10.0)
    call = ecl.open_call(provider="deepseek", op_class="judge", call_key="j1")
    attempt = ecl.begin_attempt(call)
    grp = ecl.reserve(attempt, {"usd": 1.0})
    ecl.mark_sent(attempt, request_text="q")
    ecl.store_response(attempt, response_text="{}")
    ecl.validate(attempt)
    ecl.commit(attempt, grp, {"usd": 0.5})
    with pytest.raises(IllegalCallTransition):
        ecl.commit(attempt, grp, {"usd": 0.5})
    with pytest.raises(IllegalCallTransition):
        ecl.mark_sent(attempt, request_text="again")


def test_two_dispatches_hold_two_reservations(tmp_path):
    """Keying the reservation on the attempt is what stops the second reserve from
    overwriting the first one's settled record."""
    lg, b, ecl = _ecl(tmp_path, usd=10.0)
    call = ecl.open_call(provider="deepseek", op_class="judge", call_key="j1")
    first = ecl.begin_attempt(call)
    grp = ecl.reserve(first, {"usd": 1.0})
    ecl.mark_sent(first, request_text="one")
    ecl.fail_after_response(first, grp, {"usd": 1.0}, error_class="http_500")
    second = ecl.begin_attempt(call)
    ecl.reserve(second, {"usd": 1.0})

    rows = lg.raw_connection.execute(
        "SELECT state, settled_amount FROM budget_reservations ORDER BY created_at"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["state"] == "SETTLED" and rows[0]["settled_amount"] == pytest.approx(1.0)
    assert rows[1]["state"] == "RESERVED"


def test_actual_usage_above_the_reservation_is_charged_in_full(tmp_path):
    """Clamping to the reservation made an over-budget call look exactly affordable."""
    lg, b, ecl = _ecl(tmp_path, usd=10.0)
    call = ecl.open_call(provider="deepseek", op_class="judge", call_key="j1")
    attempt = ecl.begin_attempt(call)
    grp = ecl.reserve(attempt, {"usd": 1.0})
    ecl.mark_sent(attempt, request_text="q")
    ecl.store_response(attempt, response_text="{}")
    ecl.validate(attempt)
    ecl.commit(attempt, grp, {"usd": 3.0})

    assert b.available("usd") == pytest.approx(7.0), "the real 3.0 must be charged, not 1.0"
    incident = lg.raw_connection.execute(
        "SELECT kind FROM incidents WHERE kind='budget_under_reserved'").fetchone()
    assert incident is not None, "an under-reserved call must raise an incident"


def test_a_cap_may_be_tightened_but_never_raised(tmp_path):
    lg = Ledger(str(tmp_path / "l.sqlite"), clock=lambda: 5.0)
    b = Budget(lg)
    b.ensure_account("usd", 10.0)
    b.ensure_account("usd", 4.0)
    assert b.available("usd") == 4.0
    with pytest.raises(BudgetCapRaised):
        b.ensure_account("usd", 20.0)


def test_a_v1_database_keeps_every_row_it_already_paid_for(tmp_path):
    """The migration moves history forward. Nothing recorded is dropped or rewritten."""
    import sqlite3

    path = str(tmp_path / "legacy.sqlite")
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE external_calls (
            call_id TEXT PRIMARY KEY, provider TEXT NOT NULL, op_class TEXT NOT NULL,
            state TEXT NOT NULL, work_key TEXT, attempt_ordinal INTEGER NOT NULL DEFAULT 0,
            request_object_ref TEXT, response_object_ref TEXT, provider_request_id TEXT,
            model TEXT, usage_json TEXT, error_class TEXT,
            created_at REAL NOT NULL, updated_at REAL NOT NULL);
        CREATE TABLE budget_accounts (
            resource TEXT PRIMARY KEY, cap REAL NOT NULL, reserved_total REAL NOT NULL DEFAULT 0,
            settled_total REAL NOT NULL DEFAULT 0, updated_at REAL NOT NULL);
        CREATE TABLE budget_reservations (
            reservation_id TEXT PRIMARY KEY, resource TEXT NOT NULL, amount REAL NOT NULL,
            state TEXT NOT NULL, settled_amount REAL, work_key TEXT, external_call_id TEXT,
            created_at REAL NOT NULL, updated_at REAL NOT NULL);
        """
    )
    conn.execute(
        "INSERT INTO external_calls VALUES ('c1','tavily','search','FAILED_FINAL','T1',0,"
        "'req','resp','rq','m','{}','http_401',1.0,2.0)")
    conn.execute("INSERT INTO budget_accounts VALUES ('tavily_requests',500,0,262,1.0)")
    conn.execute(
        "INSERT INTO budget_reservations VALUES ('r1','tavily_requests',1.0,'SETTLED',"
        "1.0,'T1','c1',1.0,2.0)")
    conn.commit()
    conn.close()

    lg = Ledger(path, clock=lambda: 9.0)
    call = lg.raw_connection.execute(
        "SELECT * FROM external_calls WHERE call_id='c1'").fetchone()
    # It never committed, so it is closed rather than reopened: these rows belong to a
    # round being frozen, not to work a later run should continue.
    assert call["state"] == "ABANDONED"
    attempt = lg.raw_connection.execute(
        "SELECT * FROM external_call_attempts WHERE call_id='c1'").fetchone()
    assert attempt["error_class"] == "http_401"
    assert attempt["request_object_ref"] == "req"
    assert attempt["returned_model"] == "m"
    # Spend is untouched, and the reservation now names the dispatch it paid for.
    account = lg.raw_connection.execute(
        "SELECT settled_total FROM budget_accounts WHERE resource='tavily_requests'").fetchone()
    assert account["settled_total"] == 262
    reservation = lg.raw_connection.execute(
        "SELECT attempt_id FROM budget_reservations WHERE reservation_id='r1'").fetchone()
    assert reservation["attempt_id"] == attempt["attempt_id"]


# --- an authorized cap raise leaves evidence -------------------------------------------------


def _capped_budget(tmp_path):
    from shapeflow_p1.experiment.budget import Budget
    from shapeflow_p1.experiment.ledger import Ledger

    ledger = Ledger(str(tmp_path / "l.sqlite"))
    budget = Budget(ledger)
    budget.ensure_account("deepseek_usd", 10.0)
    return ledger, budget


def test_a_config_edit_alone_cannot_widen_a_ceiling(tmp_path):
    """The guard that matters: loading a bigger number must not spend against it."""
    from shapeflow_p1.experiment.budget import BudgetCapRaised

    _ledger, budget = _capped_budget(tmp_path)
    with pytest.raises(BudgetCapRaised):
        budget.ensure_account("deepseek_usd", 200.0)
    assert budget.available("deepseek_usd") == 10.0


def test_an_authorized_raise_applies_and_records_what_changed(tmp_path):
    ledger, budget = _capped_budget(tmp_path)
    result = budget.authorize_cap_raise(
        "deepseek_usd", 200.0,
        authorization="approval binding abc123", reason="measured cost exceeds the ceiling")

    assert result["previous_cap"] == 10.0 and result["cap"] == 200.0 and result["changed"]
    assert budget.available("deepseek_usd") == 200.0

    row = ledger.raw_connection.execute(
        "SELECT severity, kind, detail FROM incidents WHERE kind='budget_cap_raised'").fetchone()
    assert row is not None, "the ceiling moved with no incident recording it"
    # Old value, new value and the authorization must all be recoverable from the record.
    assert "10.0 -> 200.0" in row["detail"]
    assert "abc123" in row["detail"]
    assert "measured cost exceeds the ceiling" in row["detail"]


def test_a_raise_will_not_quietly_lower_a_cap(tmp_path):
    """Accepting both directions here would make an audited raise an ordinary write."""
    _ledger, budget = _capped_budget(tmp_path)
    with pytest.raises(ValueError, match="tightening goes"):
        budget.authorize_cap_raise(
            "deepseek_usd", 1.0, authorization="a", reason="b")
    assert budget.available("deepseek_usd") == 10.0


def test_a_raise_must_name_its_authorization_and_reason(tmp_path):
    _ledger, budget = _capped_budget(tmp_path)
    for authorization, reason in (("", "r"), ("a", "")):
        with pytest.raises(ValueError, match="authorization and its reason"):
            budget.authorize_cap_raise(
                "deepseek_usd", 200.0, authorization=authorization, reason=reason)


def test_raising_to_the_same_value_records_nothing(tmp_path):
    ledger, budget = _capped_budget(tmp_path)
    result = budget.authorize_cap_raise(
        "deepseek_usd", 10.0, authorization="a", reason="b")
    assert result["changed"] is False
    assert ledger.raw_connection.execute(
        "SELECT COUNT(*) c FROM incidents WHERE kind='budget_cap_raised'").fetchone()["c"] == 0
