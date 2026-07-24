"""Budget admission control and the external-call state machine."""

from __future__ import annotations

import pytest

from shapeflow_p1.experiment.budget import Budget, BudgetExceeded
from shapeflow_p1.experiment.ledger import Ledger
from shapeflow_p1.object_store import ObjectStore
from shapeflow_p1.providers.external_call_ledger import ExternalCallLedger
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
    assert ecl.get_state(call) == "INTENT"
    grp = ecl.reserve(call, {"requests": 1})
    assert ecl.get_state(call) == "BUDGET_RESERVED"
    # The request text carries the key; only the redacted form may be stored.
    ecl.mark_sent(call, request_text="POST search key=tvly-FAKEFAKEFAKEFAKEFAKE")
    ecl.store_response(call, response_text="{\"results\": []}", provider_request_id="rq1")
    ecl.validate(call)
    ecl.commit(call, grp, {"requests": 1})
    assert ecl.get_state(call) == "COMMITTED"

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
    with pytest.raises(BudgetExceeded):
        ecl.reserve(call, {"requests": 1})
    assert ecl.get_state(call) == "FAILED_FINAL"


def test_external_call_timeout_after_send_is_unknown_and_keeps_cost(tmp_path):
    lg = Ledger(str(tmp_path / "l.sqlite"), clock=lambda: 5.0)
    b = Budget(lg)
    b.ensure_account("usd", 10.0)
    ecl = ExternalCallLedger(lg, b, ObjectStore(tmp_path / "obj"), SecretRedactor())
    call = ecl.open_call(provider="deepseek", op_class="judge", call_key="j1")
    grp = ecl.reserve(call, {"usd": 2.0})
    ecl.mark_sent(call, request_text="judge request")
    # ...silence...
    ecl.fail_unknown(call, grp, error_class="timeout")
    assert ecl.get_state(call) == "FAILED_UNKNOWN"
    assert b.available("usd") == 8.0  # worst-case kept, not released
