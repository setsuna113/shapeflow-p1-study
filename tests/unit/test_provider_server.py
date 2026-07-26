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

FAKE_TAVILY = "tvly-FAKE-KEY-FOR-TESTS-00000"
FAKE_DEEPSEEK = "sk-FAKE-DEEPSEEK-KEY-00000000"

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


def _service(tmp_path, *, upstream=None, caps=None, keys=True, config=None):
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
        config or ProviderConfig(served_model="Qwen3-14B-AWQ"),
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


def test_canary_audit_is_runner_only_and_uid_bound(tmp_path):
    service, *_ = _service(tmp_path)
    assert (
        service.authorize(
            "canary.audit", token=TOKENS["runner"], peer_uid=None)
        == "runner"
    )
    for role in ("steward", "evaluator"):
        with pytest.raises(ProviderError) as excinfo:
            service.authorize(
                "canary.audit", token=TOKENS[role], peer_uid=None)
        assert excinfo.value.status == 403

    service._allowed_uids["runner"] = 1001
    with pytest.raises(ProviderError) as excinfo:
        service.authorize(
            "canary.audit", token=TOKENS["runner"], peer_uid=1002)
    assert excinfo.value.status == 403


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
    body["api_key"] = "tvly-FAKE-SOMEONE-ELSES-KEY0"
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
        "SELECT request_object_ref, response_object_ref FROM external_call_attempts").fetchall()
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
        "SELECT state FROM external_call_attempts").fetchone()["state"]
    assert state == "FAILED_UNKNOWN"


def test_an_error_response_settles_at_actual_rather_than_releasing(tmp_path):
    upstream = FakeUpstream([(500, {"error": "boom"})])
    service, ledger, budget, _r = _service(tmp_path, upstream=upstream)
    before = budget.available("tavily_requests")
    status, payload = service.tavily_search(_tavily_body())
    assert status == 500
    assert payload["retry"] == "BACKOFF"
    assert budget.available("tavily_requests") == pytest.approx(before - 1.0)


def test_repeating_a_committed_acquisition_query_is_served_from_the_ledger(tmp_path):
    """The second ask is answered with the frozen response, not with a second purchase."""
    upstream = FakeUpstream([
        (200, {"request_id": "r1", "results": [], "usage": {"credits": 1.0}}),
        (200, {"request_id": "r2", "results": [], "usage": {"credits": 1.0}}),
    ])
    service, _ledger, budget, _r = _service(tmp_path, upstream=upstream)
    first_status, first = service.tavily_search(_tavily_body())
    spent = budget.available("tavily_credits")

    second_status, second = service.tavily_search(_tavily_body())
    assert (second_status, second) == (first_status, first)
    assert len(upstream.calls) == 1, "the frozen world was re-fetched and charged twice"
    assert budget.available("tavily_credits") == pytest.approx(spent), "a replay cost money"


def test_a_rejected_credential_stops_the_loop_instead_of_burning_the_cap(tmp_path):
    """262 consecutive 401s is what happens when nothing fails closed here."""
    upstream = FakeUpstream([(401, {"error": "unauthorized"})] * 5)
    service, ledger, budget, _r = _service(tmp_path, upstream=upstream)
    with pytest.raises(ProviderError) as excinfo:
        service.tavily_search(_tavily_body())
    assert excinfo.value.status == 503
    incident = ledger.raw_connection.execute(
        "SELECT kind FROM incidents WHERE kind='provider_unauthorized'").fetchone()
    assert incident is not None


def test_a_run_of_failures_opens_the_breaker(tmp_path):
    upstream = FakeUpstream([(500, {"error": "boom"})] * 10)
    service, ledger, _b, _r = _service(tmp_path, upstream=upstream)
    statuses = []
    for i in range(5):
        body = _tavily_body()
        body["query"] = f"q{i}"          # distinct logical calls, all failing
        try:
            statuses.append(service.tavily_search(body)[0])
        except ProviderError as e:
            statuses.append(e.status)
    assert statuses[-1] == 503, f"the breaker never opened: {statuses}"


# --- the breaker refuses BEFORE dispatch, and stays open across a restart -----------------
#
# Status codes alone never caught the real defect: _note_upstream_failure runs after a
# response, so every refusal above was still dispatched and still billed. These assert the
# only thing that costs money -- how many times the upstream was actually called.


def test_an_open_breaker_dispatches_nothing_further(tmp_path):
    """The 262-call lesson: a dead credential must cost one call, not one per query."""
    upstream = FakeUpstream([(401, {"error": "unauthorized"})] * 20)
    service, _ledger, budget, _r = _service(tmp_path, upstream=upstream)

    with pytest.raises(ProviderError):
        service.tavily_search(_tavily_body("q0"))
    assert len(upstream.calls) == 1
    spent_after_first = budget.available("tavily_requests")

    for i in range(1, 10):
        with pytest.raises(ProviderError) as excinfo:
            service.tavily_search(_tavily_body(f"q{i}"))
        assert excinfo.value.status == 503

    assert len(upstream.calls) == 1, (
        f"the breaker let {len(upstream.calls)} calls reach the upstream; each one is billed"
    )
    assert budget.available("tavily_requests") == pytest.approx(spent_after_first), (
        "a refused call still consumed request cap"
    )


def test_the_breaker_survives_a_provider_restart(tmp_path):
    """The counter lived in memory, so restarting the provider re-armed the same mistake.

    ``sfsupervise`` restarts the provider on failure, so an in-memory-only breaker meant a
    rejected credential was re-tried in every new process -- which is how one bad key spent
    hundreds of requests across a single acquisition run.
    """
    upstream = FakeUpstream([(401, {"error": "unauthorized"})] * 20)
    service, _ledger, _b, _r = _service(tmp_path, upstream=upstream)
    with pytest.raises(ProviderError):
        service.tavily_search(_tavily_body("q0"))
    assert len(upstream.calls) == 1

    # A brand-new service over the same ledger, exactly as a supervised restart would build.
    restarted = FakeUpstream([(401, {"error": "unauthorized"})] * 20)
    service2, _l2, _b2, _r2 = _service(tmp_path, upstream=restarted)

    with pytest.raises(ProviderError) as excinfo:
        service2.tavily_search(_tavily_body("q1"))
    assert excinfo.value.status == 503
    assert "provider_unauthorized" in str(excinfo.value)
    assert restarted.calls == [], "the restarted provider re-dispatched against a dead credential"


def test_a_healthy_provider_is_not_fenced_by_another_providers_incident(tmp_path):
    """Fencing is per provider: a dead Tavily key must not stop DeepSeek judging."""
    upstream = FakeUpstream([(401, {"error": "unauthorized"})])
    service, _ledger, _b, _r = _service(tmp_path, upstream=upstream)
    with pytest.raises(ProviderError):
        service.tavily_search(_tavily_body("q0"))

    ok = FakeUpstream([(200, {
        "id": "d1", "model": "deepseek-chat",
        "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    })])
    service2, _l2, _b2, _r2 = _service(tmp_path, upstream=ok)
    status, _payload = service2.deepseek_chat(_deepseek_body(), role="evaluator")
    assert status == 200
    assert len(ok.calls) == 1


# --- credentials are probed before anything is bought ------------------------------------


def test_the_credential_probe_buys_nothing(tmp_path):
    """A good key answers 400 on a deliberately invalid body: auth passed, nothing searched."""
    upstream = FakeUpstream([(400, {"error": "missing query"})] * 3)
    service, ledger, budget, _r = _service(tmp_path, upstream=upstream)
    before = budget.available("tavily_requests")

    status, body = service.credentials_probe()

    assert status == 200 and body["ok"] is True
    assert budget.available("tavily_requests") == pytest.approx(before), "the probe cost budget"
    assert ledger.raw_connection.execute(
        "SELECT COUNT(*) FROM external_calls").fetchone()[0] == 0, "the probe opened a call"


def test_the_credential_probe_reports_a_rejected_key_and_fences_it(tmp_path):
    """This is the check whose absence turned one dead key into 262 billed rejections."""
    upstream = FakeUpstream([(401, {"error": "unauthorized"})] * 3)
    service, ledger, _b, _r = _service(tmp_path, upstream=upstream)

    status, body = service.credentials_probe()

    assert status == 503 and body["ok"] is False
    assert body["providers"]["tavily"]["ok"] is False
    assert "401" in body["providers"]["tavily"]["detail"]
    # And the rejection is durable, so the real path refuses before dispatch from now on.
    assert ledger.raw_connection.execute(
        "SELECT COUNT(*) FROM incidents WHERE kind='provider_unauthorized'").fetchone()[0] >= 1
    with pytest.raises(ProviderError) as excinfo:
        service.tavily_search(_tavily_body("q0"))
    assert excinfo.value.status == 503


def test_a_missing_credential_is_reported_as_unconfigured_not_as_working(tmp_path):
    service, _ledger, _b, _r = _service(tmp_path, keys=False)
    status, body = service.credentials_probe()
    assert status == 503
    assert body["providers"]["tavily"]["configured"] is False
    assert body["providers"]["tavily"]["ok"] is None


# --- an unfreezable response must not become a COMMITTED call ----------------------------


def test_an_oversize_response_is_not_committed_with_a_truncated_body(tmp_path):
    """Truncating to 200 KB stored invalid JSON under a COMMITTED call.

    The call was then bought and permanently unreplayable: every later read raised 409, so it
    could neither be recovered nor legitimately re-fetched. Failing the attempt keeps the
    logical call re-fetchable instead.
    """
    from shapeflow_p1.runtime.provider_server import ProviderConfig

    huge = {"results": [{"content": "x" * 5000} for _ in range(50)], "usage": {"credits": 1}}
    config = ProviderConfig(served_model="Qwen3-14B-AWQ", max_frozen_response_bytes=1024)
    upstream = FakeUpstream([(200, huge)])
    service, ledger, _b, _r = _service(tmp_path, upstream=upstream, config=config)

    with pytest.raises(ProviderError) as excinfo:
        service.tavily_search(_tavily_body("q0"))
    assert excinfo.value.status == 502

    committed = ledger.raw_connection.execute(
        "SELECT COUNT(*) FROM external_calls WHERE state='COMMITTED'").fetchone()[0]
    assert committed == 0, "an unfreezable response was committed anyway"
    incident = ledger.raw_connection.execute(
        "SELECT detail FROM incidents WHERE kind='provider_response_too_large'").fetchone()
    assert incident is not None


def test_a_normal_response_is_frozen_whole(tmp_path):
    """The guard is a corruption check, not a routine limit: real bodies store verbatim."""
    payload = {"results": [{"content": "y" * 300_000}], "usage": {"credits": 1}}
    upstream = FakeUpstream([(200, payload)])
    service, ledger, _b, _r = _service(tmp_path, upstream=upstream)

    status, _body = service.tavily_search(_tavily_body("q0"))
    assert status == 200

    call_id = ledger.raw_connection.execute(
        "SELECT call_id FROM external_calls WHERE state='COMMITTED'").fetchone()[0]
    ref = service._calls.committed_response_ref(call_id)
    stored = json.loads(ObjectStore(tmp_path / "objects").get_bytes(ref).decode("utf-8"))
    assert stored == payload, "the frozen body is not byte-faithful to what the vendor returned"
    # 300 KB: the old [:200000] truncation would have cut this into invalid JSON.
    assert len(json.dumps(stored)) > 200_000


def test_deepseek_usd_is_derived_from_reported_usage(tmp_path):
    upstream = FakeUpstream([(200, {
        "id": "d1", "model": "deepseek-chat",
        "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
    })])
    service, ledger, budget, _r = _service(tmp_path, upstream=upstream)
    before = budget.available("deepseek_usd")
    service.deepseek_chat(_deepseek_body(), role="evaluator")
    # A million tokens each way at the configured prices costs far more than the 0.05
    # worst case. The real amount is charged; clamping it to the reservation would make an
    # under-reserved call look exactly affordable.
    charged = before - budget.available("deepseek_usd")
    expected = ProviderConfig().deepseek_usd_per_1m_input + \
        ProviderConfig().deepseek_usd_per_1m_output
    assert charged == pytest.approx(expected)
    assert ledger.raw_connection.execute(
        "SELECT kind FROM incidents WHERE kind='budget_under_reserved'").fetchone() is not None


def test_cache_hit_prompt_tokens_are_charged_at_the_cache_rate(tmp_path):
    """DeepSeek bills cache hits ~120x below misses and reports the split.

    A judging workload re-sends near-identical prompts constantly, so charging every prompt
    token at the miss rate materially over-counts -- which is how a ledger reading $31.40 was
    produced against a real bill near $1.60.
    """
    upstream = FakeUpstream([(200, {
        "id": "d1", "model": "deepseek-v4-pro",
        "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 1_000_000, "completion_tokens": 0,
            "prompt_cache_hit_tokens": 750_000, "prompt_cache_miss_tokens": 250_000,
        },
    })])
    service, _ledger, budget, _r = _service(tmp_path, upstream=upstream)
    before = budget.available("deepseek_usd")
    service.deepseek_chat(_deepseek_body(), role="evaluator")
    charged = before - budget.available("deepseek_usd")

    cfg = ProviderConfig()
    expected = 0.25 * cfg.deepseek_usd_per_1m_input + 0.75 * cfg.deepseek_usd_per_1m_input_cached
    assert charged == pytest.approx(expected)
    # And it must be strictly cheaper than pricing the whole prompt as a miss, or the split
    # is being read but not applied.
    assert charged < cfg.deepseek_usd_per_1m_input


def test_an_unreported_cache_split_is_charged_as_all_miss(tmp_path):
    """Absent the split, over-count rather than under-count: a bound must stay a bound."""
    upstream = FakeUpstream([(200, {
        "id": "d1", "model": "deepseek-v4-pro",
        "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0},
    })])
    service, _ledger, budget, _r = _service(tmp_path, upstream=upstream)
    before = budget.available("deepseek_usd")
    service.deepseek_chat(_deepseek_body(), role="evaluator")
    charged = before - budget.available("deepseek_usd")
    assert charged == pytest.approx(ProviderConfig().deepseek_usd_per_1m_input)


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


def test_canary_audit_request_is_closed(tmp_path):
    service, *_ = _service(tmp_path)
    with pytest.raises(ProviderError) as excinfo:
        service.canary_audit({
            "work_keys": ["WK1"],
            "include_prompt": True,
        })
    assert excinfo.value.status == 400
    assert "include_prompt" in str(excinfo.value)


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


def test_canary_audit_returns_only_sanitized_exact_work_aggregates(tmp_path):
    secret_prompt = "PRIVATE-PROMPT-THAT-MUST-NOT-CROSS-THE-BOUNDARY"
    upstream = FakeUpstream([(200, {
        "model": "Qwen3-14B-AWQ",
        "choices": [{"message": {"content": "PRIVATE-RESPONSE"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "prompt_tokens_details": {"cached_tokens": 2},
        },
    })])
    service, ledger, _budget, _redactor = _service(tmp_path, upstream=upstream)
    token = _register_cell(service)
    service.chat_completions(
        {
            "model": "qwen-selector-page",
            "messages": [{"role": "user", "content": secret_prompt}],
            "max_tokens": 123,
        },
        cell_token=token,
    )
    # A historical cell exists in the same provider ledger but was not requested.
    service.register_cell({
        "cell_token": "cell-0002", "run_id": "OLD", "task_id": "OLD",
        "arm_id": "P0", "variant_id": "P0", "replicate_id": "0",
        "work_key": "WK-HISTORICAL", "layer": "causal",
    })
    service.chat_completions(
        {
            "model": "qwen-research",
            "messages": [{"role": "user", "content": "HISTORICAL-PROMPT"}],
            "max_tokens": 456,
        },
        cell_token="cell-0002",
    )

    status, attestation = service.canary_audit(
        {"work_keys": ["WK-NO-CALLS", "WK1"]})
    assert status == 200
    assert attestation["work_keys"] == ["WK-NO-CALLS", "WK1"]
    assert [row["work_key"] for row in attestation["work"]] == [
        "WK-NO-CALLS", "WK1",
    ]
    by_work = {row["work_key"]: row for row in attestation["work"]}
    assert by_work["WK-NO-CALLS"]["ops"] == []
    assert by_work["WK1"]["open_attempts"] == 0
    op = by_work["WK1"]["ops"][0]
    settled = op["settled_gpu_seconds"]
    assert settled >= 0
    authoritative = ledger.raw_connection.execute(
        "SELECT settled_amount FROM budget_reservations"
        " WHERE work_key='WK1' AND resource='gpu_seconds'"
    ).fetchone()["settled_amount"]
    assert settled == pytest.approx(authoritative)
    assert by_work["WK1"]["settled_gpu_seconds"] == pytest.approx(settled)
    assert op == {
        "op_class": "PAGE_P1_SELECTOR_LOCAL",
        "attempt_count": 1,
        "committed_attempt_count": 1,
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "cached_prompt_tokens": 2,
        "max_completion_tokens_observed": 3,
        "settled_gpu_seconds": settled,
        "selector_request_max_tokens": [123],
    }
    assert attestation["gpu_budget"]["resource"] == "gpu_seconds"
    assert (
        attestation["gpu_budget"]["remaining"]
        == pytest.approx(
            attestation["gpu_budget"]["cap"]
            - attestation["gpu_budget"]["reserved"]
            - attestation["gpu_budget"]["settled"]
        )
    )

    dumped = json.dumps(attestation, sort_keys=True)
    for forbidden in (
        secret_prompt,
        "PRIVATE-RESPONSE",
        "HISTORICAL-PROMPT",
        FAKE_TAVILY,
        FAKE_DEEPSEEK,
        "request_object_ref",
        "response_object_ref",
        "prompt_sha256",
        "messages",
        "headers",
        "credential",
    ):
        assert forbidden not in dumped
    assert "WK-HISTORICAL" not in dumped


def test_canary_audit_reports_exact_work_open_attempts(tmp_path):
    service, _ledger, _budget, _redactor = _service(tmp_path)
    call_id = service._calls.open_call(
        provider="vllm",
        op_class="PAGE_P1_SELECTOR_LOCAL",
        call_key="open-call",
        work_key="WK1",
    )
    service._calls.begin_attempt(call_id)
    _status, attestation = service.canary_audit({"work_keys": ["WK1"]})
    assert attestation["open_attempts"] == 1
    assert attestation["work"][0]["open_attempts"] == 1


def test_canary_audit_fails_closed_without_the_gpu_budget_account(tmp_path):
    service, ledger, *_ = _service(tmp_path)
    with ledger.transaction() as cur:
        cur.execute("DELETE FROM budget_accounts WHERE resource='gpu_seconds'")
    with pytest.raises(ProviderError, match="budget account is missing") as excinfo:
        service.canary_audit({"work_keys": ["WK1"]})
    assert excinfo.value.status == 409


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


def test_causal_gateway_serializes_upstream_and_persists_work_intervals(tmp_path):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    class ConcurrentProbe:
        def __init__(self):
            self.lock = threading.Lock()
            self.active = 0
            self.max_active = 0

        def __call__(self, _url, _headers, _body, _timeout):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.03)
            with self.lock:
                self.active -= 1
            return 200, {
                "model": "Qwen3-14B-AWQ",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                    "prompt_tokens_details": {"cached_tokens": 2},
                },
            }, 0.03

    upstream = ConcurrentProbe()
    service, ledger, *_ = _service(
        tmp_path,
        upstream=upstream,
        config=ProviderConfig(
            served_model="Qwen3-14B-AWQ",
            max_upstream_inflight=1,
        ),
    )
    _register_cell(service, "cell-0001")
    service.register_cell({
        "cell_token": "cell-0002", "run_id": "RUN1", "task_id": "T1", "arm_id": "P0",
        "variant_id": "P0", "replicate_id": "0", "work_key": "WK2", "layer": "causal",
    })

    def invoke(token):
        return service.chat_completions(
            {"model": "qwen-research", "messages": [{"role": "user", "content": token}]},
            cell_token=token,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(invoke, ("cell-0001", "cell-0002")))

    assert [status for status, _ in results] == [200, 200]
    assert upstream.max_active == 1
    rows = ledger.raw_connection.execute(
        "SELECT proxy_ingress_at, dispatched_at, response_end_at, telemetry_json"
        " FROM external_call_attempts ORDER BY dispatched_at"
    ).fetchall()
    assert len(rows) == 2
    assert all(r["proxy_ingress_at"] is not None and r["response_end_at"] is not None for r in rows)
    assert rows[1]["dispatched_at"] >= rows[0]["response_end_at"]
    assert all(json.loads(r["telemetry_json"])["work_key"] in {"WK1", "WK2"} for r in rows)

    summary = service.cell_work({"work_key": "WK1", "require_isolated": True})[1]
    assert summary["telemetry_complete"] is True
    assert summary["overlap_valid"] is True
    assert summary["tokens"] == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "cached_prompt_tokens": 2,
    }


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
    attempt = service._calls.begin_attempt(call_id)
    service._calls.reserve(attempt, {"tavily_requests": 1.0, "tavily_credits": 2.0})
    service._calls.mark_sent(attempt, request_text="{}")
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
        "SELECT state FROM external_call_attempts WHERE call_id=?", (call_id,)
    ).fetchone()["state"] == "FAILED_UNKNOWN"
    assert budget.available("tavily_credits") == pytest.approx(before)


def test_restart_releases_a_call_that_never_went_out(tmp_path):
    upstream = FakeUpstream()
    service, ledger, budget, redactor = _service(tmp_path, upstream=upstream)
    call_id = service._calls.open_call(provider="tavily", op_class="search", call_key="k2")
    attempt = service._calls.begin_attempt(call_id)
    service._calls.reserve(attempt, {"tavily_requests": 1.0, "tavily_credits": 2.0})
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
    assert resolve_route("/v1/canary/audit") == ("canary.audit", None)
    with pytest.raises(ProviderError):
        resolve_route("/v1/anything/else")


def test_no_credential_route_answers_without_a_loaded_key(tmp_path):
    service, *_ = _service(tmp_path, keys=False)
    with pytest.raises(ProviderError) as excinfo:
        service.tavily_search(_tavily_body())
    assert excinfo.value.status == 503
