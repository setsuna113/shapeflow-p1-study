"""The provider boundary: who may call what, what it costs, and what it never writes down.

These tests drive the service directly with a fake upstream, so the whole reserve -> dispatch ->
settle path is exercised without a network and without a credential. The fake credential planted
here is checked for absence everywhere the service could plausibly have written it.
"""

from __future__ import annotations

import json

import pytest

from shapeflow_p1.experiment.budget import Budget
from shapeflow_p1.experiment.ledger import Ledger
from shapeflow_p1.object_store import ObjectStore
from shapeflow_p1.runtime.provider_server import (
    PROVIDER_KEY_PLACEHOLDER,
    ProviderConfig,
    ProviderError,
    ProviderService,
    RoleTokens,
    resolve_route,
    serve_forever,
)
from shapeflow_p1.secrets import SecretRedactor

FAKE_TAVILY = "tvly-FAKEKEYFORTESTS0000000000"
FAKE_DEEPSEEK = "sk-FAKEDEEPSEEKKEY000000000000"

TOKENS = {
    "runner": "runner-token-0000000000000000",
    "steward": "steward-token-000000000000000",
    "evaluator": "evaluator-token-00000000000000",
}


class FakeUpstream:
    """Records every outbound call and replays scripted replies."""

    def __init__(self, replies=None) -> None:
        self.calls: list[dict] = []
        self._replies = list(replies or [])
        self.raise_with: Exception | None = None

    def __call__(self, url, headers, body, timeout):
        self.calls.append({"url": url, "headers": dict(headers), "body": json.loads(
            json.dumps(body, default=str)), "timeout": timeout})
        if self.raise_with is not None:
            raise self.raise_with
        if self._replies:
            status, payload = self._replies.pop(0)
            return status, payload, 0.01
        return 200, {}, 0.01


def _service(tmp_path, *, upstream=None, caps=None, keys=True):
    ledger = Ledger(str(tmp_path / "ledger.sqlite"))
    budget = Budget(ledger)
    defaults = {
        "tavily_requests": 100.0, "tavily_credits": 200.0, "remote_calls": 500.0,
        "deepseek_requests": 100.0, "deepseek_input_tokens": 1_000_000.0,
        "deepseek_output_tokens": 200_000.0, "deepseek_usd": 5.0, "gpu_seconds": 100_000.0,
    }
    defaults.update(caps or {})
    for resource, cap in defaults.items():
        budget.ensure_account(resource, cap)
    redactor = SecretRedactor()
    service = ProviderService(
        ProviderConfig(served_model="Qwen3-14B-AWQ"),
        ledger=ledger, budget=budget, store=ObjectStore(tmp_path / "objects"),
        redactor=redactor, tokens=RoleTokens(TOKENS),
        upstream=upstream or FakeUpstream(),
        tavily_key=FAKE_TAVILY if keys else None,
        deepseek_key=FAKE_DEEPSEEK if keys else None,
        allowed_uids={},
    )
    service.reconcile_on_start()
    return service, ledger, budget, redactor


def _tavily_body(query="q1"):
    return {
        "api_key": PROVIDER_KEY_PLACEHOLDER, "query": query, "search_depth": "advanced",
        "include_raw_content": "markdown", "include_answer": False, "include_usage": True,
        "max_results": 8, "_task_id": "T1", "_call_key": f"qs-{query}",
    }


def _deepseek_body(op="JUDGE_REPORT"):
    return {
        "api_key": PROVIDER_KEY_PLACEHOLDER, "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {"type": "json_object"}, "temperature": 0.0,
        "_op_class": op, "_work_key": "W1",
    }


# --- who may call what -------------------------------------------------------------------


def test_runner_cannot_reach_a_credential_bearing_route(tmp_path):
    service, *_ = _service(tmp_path)
    for route in ("tavily.search", "deepseek.chat"):
        with pytest.raises(ProviderError) as excinfo:
            service.authorize(route, token=TOKENS["runner"], peer_uid=None)
        assert excinfo.value.status == 403


def test_steward_and_evaluator_have_the_routes_they_need(tmp_path):
    service, *_ = _service(tmp_path)
    assert service.authorize("tavily.search", token=TOKENS["steward"], peer_uid=None) == "steward"
    assert service.authorize("deepseek.chat", token=TOKENS["evaluator"], peer_uid=None) == "evaluator"
    assert service.authorize("chat.completions", token=TOKENS["runner"], peer_uid=None) == "runner"


def test_an_unknown_token_is_refused(tmp_path):
    service, *_ = _service(tmp_path)
    with pytest.raises(ProviderError) as excinfo:
        service.authorize("healthz", token="not-a-real-token-0000000000", peer_uid=None)
    assert excinfo.value.status == 401


def test_a_token_presented_by_the_wrong_uid_is_refused(tmp_path):
    service, ledger, budget, redactor = _service(tmp_path)
    service._allowed_uids["steward"] = 996
    with pytest.raises(ProviderError) as excinfo:
        service.authorize("tavily.search", token=TOKENS["steward"], peer_uid=998)
    assert excinfo.value.status == 403


def test_two_roles_may_not_share_a_token():
    with pytest.raises(ProviderError):
        RoleTokens({"runner": TOKENS["runner"], "steward": TOKENS["runner"]})


# --- the client never holds a credential ---------------------------------------------------


def test_a_client_supplied_key_is_refused(tmp_path):
    """A client that could send its own key would have had one in memory."""
    service, *_ = _service(tmp_path)
    body = _tavily_body()
    body["api_key"] = "tvly-SOMEONE-ELSES-KEY-000000"
    with pytest.raises(ProviderError) as excinfo:
        service.tavily_search(body)
    assert excinfo.value.status == 400


def test_the_real_key_reaches_the_upstream_and_nothing_else(tmp_path):
    upstream = FakeUpstream([(200, {"request_id": "r1", "results": [], "usage": {"credits": 1.0}})])
    service, ledger, _budget, redactor = _service(tmp_path, upstream=upstream)

    status, _payload = service.tavily_search(_tavily_body())
    assert status == 200
    assert upstream.calls[0]["body"]["api_key"] == FAKE_TAVILY

    # Nothing the service persisted or emitted may contain it.
    dumped = json.dumps(service.events)
    assert FAKE_TAVILY not in dumped
    rows = ledger.raw_connection.execute(
        "SELECT request_object_ref, response_object_ref FROM external_calls").fetchall()
    store = ObjectStore(tmp_path / "objects")
    for row in rows:
        for ref in (row["request_object_ref"], row["response_object_ref"]):
            if ref:
                text = store.get_bytes(ref).decode("utf-8")
                assert FAKE_TAVILY not in text
                assert redactor.is_clean(text)


def test_deepseek_key_travels_only_in_the_outbound_authorization_header(tmp_path):
    upstream = FakeUpstream([(200, {"id": "d1", "model": "deepseek-chat",
                                    "choices": [{"message": {"content": "{}"},
                                                 "finish_reason": "stop"}],
                                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}})])
    service, ledger, _b, _r = _service(tmp_path, upstream=upstream)
    status, _ = service.deepseek_chat(_deepseek_body(), role="evaluator")
    assert status == 200
    assert upstream.calls[0]["headers"]["Authorization"] == f"Bearer {FAKE_DEEPSEEK}"
    assert "api_key" not in upstream.calls[0]["body"]
    assert FAKE_DEEPSEEK not in json.dumps(service.events)


# --- admission control ---------------------------------------------------------------------


def test_reservation_is_won_before_dispatch_and_settled_after(tmp_path):
    upstream = FakeUpstream([(200, {"request_id": "r1", "results": [], "usage": {"credits": 1.0}})])
    service, _ledger, budget, _r = _service(tmp_path, upstream=upstream)
    before = budget.available("tavily_credits")
    service.tavily_search(_tavily_body())
    # Worst case is 2.0, actual 1.0: exactly the surplus comes back.
    assert budget.available("tavily_credits") == pytest.approx(before - 1.0)


def test_a_refused_reservation_never_dispatches(tmp_path):
    upstream = FakeUpstream()
    service, _ledger, _budget, _r = _service(
        tmp_path, upstream=upstream, caps={"tavily_credits": 0.5})
    with pytest.raises(ProviderError) as excinfo:
        service.tavily_search(_tavily_body())
    assert excinfo.value.status == 429
    assert upstream.calls == [], "the call went out after admission was refused"


def test_timeout_after_send_keeps_the_worst_case_and_is_not_a_free_retry(tmp_path):
    upstream = FakeUpstream()
    upstream.raise_with = TimeoutError("no reply")
    service, ledger, budget, _r = _service(tmp_path, upstream=upstream)
    before = budget.available("tavily_credits")

    with pytest.raises(ProviderError) as excinfo:
        service.tavily_search(_tavily_body())
    assert excinfo.value.status == 504
    # The call DID go out, so the worst case stays spent.
    assert budget.available("tavily_credits") == pytest.approx(before - 2.0)
    state = ledger.raw_connection.execute(
        "SELECT state FROM external_calls").fetchone()["state"]
    assert state == "FAILED_UNKNOWN"


def test_an_error_response_settles_at_actual_rather_than_releasing(tmp_path):
    upstream = FakeUpstream([(500, {"error": "boom"})])
    service, ledger, budget, _r = _service(tmp_path, upstream=upstream)
    before = budget.available("tavily_requests")
    status, payload = service.tavily_search(_tavily_body())
    assert status == 500
    assert payload["retry"] == "BACKOFF"
    assert budget.available("tavily_requests") == pytest.approx(before - 1.0)


def test_repeating_a_committed_acquisition_query_is_refused(tmp_path):
    upstream = FakeUpstream([
        (200, {"request_id": "r1", "results": [], "usage": {"credits": 1.0}}),
        (200, {"request_id": "r2", "results": [], "usage": {"credits": 1.0}}),
    ])
    service, *_ = _service(tmp_path, upstream=upstream)
    service.tavily_search(_tavily_body())
    with pytest.raises(ProviderError) as excinfo:
        service.tavily_search(_tavily_body())
    assert excinfo.value.status == 409
    assert len(upstream.calls) == 1, "the frozen world was re-fetched and charged twice"


def test_deepseek_usd_is_derived_from_reported_usage(tmp_path):
    upstream = FakeUpstream([(200, {
        "id": "d1", "model": "deepseek-chat",
        "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
    })])
    service, _ledger, budget, _r = _service(tmp_path, upstream=upstream)
    before = budget.available("deepseek_usd")
    service.deepseek_chat(_deepseek_body(), role="evaluator")
    # 0.27 + 1.10 per million in and out, capped by the worst-case reservation.
    assert budget.available("deepseek_usd") == pytest.approx(before - 0.05)


# --- schemas and op classes -----------------------------------------------------------------


def test_a_deepseek_call_without_json_mode_is_refused(tmp_path):
    service, *_ = _service(tmp_path)
    body = _deepseek_body()
    body["response_format"] = {"type": "text"}
    with pytest.raises(ProviderError) as excinfo:
        service.deepseek_chat(body, role="evaluator")
    assert excinfo.value.status == 400


def test_deepseek_may_not_serve_a_treatment_op_class(tmp_path):
    service, *_ = _service(tmp_path)
    with pytest.raises(ProviderError) as excinfo:
        service.deepseek_chat(_deepseek_body(op="RESEARCHER_REACT"), role="evaluator")
    assert excinfo.value.status == 403


def test_an_unknown_body_field_is_refused_by_name(tmp_path):
    service, *_ = _service(tmp_path)
    body = _tavily_body()
    body["auto_parameters"] = True
    with pytest.raises(ProviderError) as excinfo:
        service.tavily_search(body)
    assert "auto_parameters" in str(excinfo.value)


# --- inference tagging -----------------------------------------------------------------------


def _register_cell(service, token="cell-0001"):
    service.register_cell({
        "cell_token": token, "run_id": "RUN1", "task_id": "T1", "arm_id": "H",
        "variant_id": "H02", "replicate_id": "0", "work_key": "WK1", "layer": "causal",
    })
    return token


def test_an_unregistered_model_alias_is_refused(tmp_path):
    service, *_ = _service(tmp_path)
    token = _register_cell(service)
    with pytest.raises(ProviderError) as excinfo:
        service.chat_completions(
            {"model": "some-other-model", "messages": [{"role": "user", "content": "x"}]},
            cell_token=token)
    assert excinfo.value.status == 400


def test_the_alias_is_rewritten_and_the_op_class_recorded(tmp_path):
    upstream = FakeUpstream([(200, {
        "model": "Qwen3-14B-AWQ", "choices": [{"message": {"content": "ok"},
                                               "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    })])
    service, *_ = _service(tmp_path, upstream=upstream)
    token = _register_cell(service)
    status, _ = service.chat_completions(
        {"model": "openai:qwen-selector-page", "messages": [{"role": "user", "content": "x"}]},
        cell_token=token)
    assert status == 200
    # The engine sees the one served model, so P0 and P1 issue identical upstream requests.
    assert upstream.calls[0]["body"]["model"] == "Qwen3-14B-AWQ"
    event = [e for e in service.events if e["kind"] == "INFERENCE_COMMITTED"][-1]
    assert event["op_class"] == "PAGE_P1_SELECTOR_LOCAL"
    assert (event["task_id"], event["arm_id"], event["variant_id"]) == ("T1", "H", "H02")
    assert event["cached_prompt_tokens"] is None, "cached tokens must not be invented"


def test_cached_tokens_are_recorded_only_when_the_engine_reports_them(tmp_path):
    upstream = FakeUpstream([(200, {
        "model": "Qwen3-14B-AWQ", "choices": [{"message": {"content": "ok"},
                                               "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3,
                  "prompt_tokens_details": {"cached_tokens": 4}},
    })])
    service, *_ = _service(tmp_path, upstream=upstream)
    token = _register_cell(service)
    service.chat_completions(
        {"model": "qwen-research", "messages": [{"role": "user", "content": "x"}]},
        cell_token=token)
    event = [e for e in service.events if e["kind"] == "INFERENCE_COMMITTED"][-1]
    assert event["cached_prompt_tokens"] == 4


def test_rebinding_a_cell_token_to_new_coordinates_is_refused(tmp_path):
    service, *_ = _service(tmp_path)
    _register_cell(service)
    with pytest.raises(ProviderError) as excinfo:
        service.register_cell({
            "cell_token": "cell-0001", "run_id": "RUN1", "task_id": "T2", "arm_id": "P0",
            "variant_id": "P0", "replicate_id": "0", "work_key": "WK2",
        })
    assert excinfo.value.status == 409


def test_an_unknown_cell_token_is_refused(tmp_path):
    service, *_ = _service(tmp_path)
    with pytest.raises(ProviderError) as excinfo:
        service.chat_completions(
            {"model": "qwen-research", "messages": [{"role": "user", "content": "x"}]},
            cell_token="cell-9999")
    assert excinfo.value.status == 404


# --- restart -----------------------------------------------------------------------------------


def test_restart_closes_out_a_call_that_was_in_flight(tmp_path):
    """A provider that died mid-call cannot know whether it was billed, so it keeps the charge."""
    upstream = FakeUpstream()
    upstream.raise_with = None
    service, ledger, budget, redactor = _service(tmp_path, upstream=upstream)

    # Reach SENT and stop there, exactly as a SIGKILL would.
    call_id = service._calls.open_call(provider="tavily", op_class="search", call_key="k1")
    service._calls.reserve(call_id, {"tavily_requests": 1.0, "tavily_credits": 2.0})
    service._calls.mark_sent(call_id, request_text="{}")
    before = budget.available("tavily_credits")

    fresh = ProviderService(
        ProviderConfig(), ledger=ledger, budget=budget,
        store=ObjectStore(tmp_path / "objects"), redactor=redactor,
        tokens=RoleTokens(TOKENS), upstream=upstream,
        tavily_key=FAKE_TAVILY, deepseek_key=FAKE_DEEPSEEK,
    )
    counts = fresh.reconcile_on_start()
    assert counts["failed_unknown"] == 1
    assert ledger.raw_connection.execute(
        "SELECT state FROM external_calls WHERE call_id=?", (call_id,)
    ).fetchone()["state"] == "FAILED_UNKNOWN"
    assert budget.available("tavily_credits") == pytest.approx(before)


def test_restart_releases_a_call_that_never_went_out(tmp_path):
    upstream = FakeUpstream()
    service, ledger, budget, redactor = _service(tmp_path, upstream=upstream)
    call_id = service._calls.open_call(provider="tavily", op_class="search", call_key="k2")
    service._calls.reserve(call_id, {"tavily_requests": 1.0, "tavily_credits": 2.0})
    reserved = budget.available("tavily_credits")

    fresh = ProviderService(
        ProviderConfig(), ledger=ledger, budget=budget,
        store=ObjectStore(tmp_path / "objects"), redactor=redactor,
        tokens=RoleTokens(TOKENS), upstream=upstream,
        tavily_key=FAKE_TAVILY, deepseek_key=FAKE_DEEPSEEK,
    )
    counts = fresh.reconcile_on_start()
    assert counts["released"] == 1
    assert budget.available("tavily_credits") == pytest.approx(reserved + 2.0)


# --- binding and routing -----------------------------------------------------------------------


def test_the_provider_refuses_to_bind_a_non_loopback_address(tmp_path):
    service, _l, _b, redactor = _service(tmp_path)
    with pytest.raises(ProviderError) as excinfo:
        serve_forever(service, ProviderConfig(bind_host="0.0.0.0"), redactor)
    assert "loopback-only" in str(excinfo.value)


def test_route_resolution_maps_the_cell_path(tmp_path):
    assert resolve_route("/v1/tavily/search") == ("tavily.search", None)
    assert resolve_route("/v1/cell/cell-0001/chat/completions") == ("chat.completions", "cell-0001")
    assert resolve_route("/v1/chat/completions") == ("chat.completions", None)
    with pytest.raises(ProviderError):
        resolve_route("/v1/anything/else")


def test_no_credential_route_answers_without_a_loaded_key(tmp_path):
    service, *_ = _service(tmp_path, keys=False)
    with pytest.raises(ProviderError) as excinfo:
        service.tavily_search(_tavily_body())
    assert excinfo.value.status == 503
