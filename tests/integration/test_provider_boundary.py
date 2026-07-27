"""The provider over real HTTP, driving the real Tavily and DeepSeek clients.

The unit tests exercise the service object; this one starts the listener, sends real requests
through :class:`ProviderClient`, and drives the *existing*
:class:`~shapeflow_p1.acquire.tavily_client.TavilyCaptureClient` and
:class:`~shapeflow_p1.bench.grading.judge_client.DeepSeekJudge` through it unchanged. That is the
point: there is one implementation of "call Tavily", and it now reaches the network only through
the boundary.

The fake credential is planted in the provider and additionally *echoed back by the upstream* in
an error body -- the way a real provider leaks one -- and asserted absent from every artifact.
"""

from __future__ import annotations

import json

import pytest

from shapeflow_p1.acquire.tavily_client import TavilyCaptureClient, TavilyParams
from shapeflow_p1.bench.grading.judge_client import DeepSeekJudge, JudgeUnavailable
from shapeflow_p1.experiment.budget import Budget
from shapeflow_p1.experiment.ledger import Ledger
from shapeflow_p1.object_store import ObjectStore
from shapeflow_p1.providers.provider_client import (
    PROVIDER_KEY_PLACEHOLDER,
    ProviderCallError,
    ProviderClient,
)
from shapeflow_p1.runtime.provider_server import (
    ProviderConfig,
    ProviderService,
    RoleTokens,
    serve_forever,
)
from shapeflow_p1.secrets import SecretRedactor

FAKE_TAVILY = "tvly-FAKE-INTEGRATION-000000"
FAKE_DEEPSEEK = "sk-FAKE-INTEGRATION-DEEPSEEK"

TOKENS = {
    "runner": "itest-runner-token-000000000",
    "steward": "itest-steward-token-00000000",
    "evaluator": "itest-evaluator-token-000000",
}


class EchoingUpstream:
    """A fake upstream that, on demand, echoes the credential back the way a real one can."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.mode = "ok"

    def __call__(self, url, headers, body, timeout):
        self.calls.append({"url": url, "headers": dict(headers), "body": dict(body)})
        if "tavily" in url:
            if self.mode == "leak":
                # A provider that repeats the key in its error body. This is the realistic leak.
                return 401, {"detail": f"invalid api key {body.get('api_key')}"}, 0.01
            return 200, {
                "request_id": "req-1", "response_time": 0.4,
                "usage": {"credits": 1.0}, "failed_results": [],
                "results": [
                    {"url": "https://a.example", "title": "A", "content": "snippet",
                     "raw_content": "RAW A", "score": 0.9, "published_date": "2026-01-01"},
                ],
            }, 0.01
        if "deepseek" in url:
            if self.mode == "leak":
                return 500, {"error": {"message": f"auth {headers.get('Authorization')}"}}, 0.01
            return 200, {
                "id": "chat-1", "model": "deepseek-chat", "system_fingerprint": "fp_1",
                "choices": [{"message": {"content": json.dumps({"verdict": "ok"})},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4},
            }, 0.01
        return 200, {"model": "Qwen3-14B-AWQ",
                     "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 3, "completion_tokens": 1}}, 0.01


@pytest.fixture()
def provider(tmp_path):
    ledger = Ledger(str(tmp_path / "ledger.sqlite"))
    budget = Budget(ledger)
    for resource, cap in {
        "tavily_requests": 50.0, "tavily_credits": 100.0, "remote_calls": 200.0,
        "deepseek_requests": 50.0, "deepseek_input_tokens": 500_000.0,
        "deepseek_output_tokens": 100_000.0, "deepseek_usd": 5.0, "gpu_seconds": 10_000.0,
    }.items():
        budget.ensure_account(resource, cap)

    redactor = SecretRedactor()
    store = ObjectStore(tmp_path / "objects")
    upstream = EchoingUpstream()
    service = ProviderService(
        ProviderConfig(served_model="Qwen3-14B-AWQ"),
        ledger=ledger, budget=budget, store=store, redactor=redactor,
        tokens=RoleTokens(TOKENS), upstream=upstream,
        tavily_key=FAKE_TAVILY, deepseek_key=FAKE_DEEPSEEK,
    )
    service.reconcile_on_start()
    # Port 0: the OS picks a free port, so parallel test runs cannot collide.
    tcp, _uds = serve_forever(service, ProviderConfig(bind_port=0), redactor)
    base = f"http://127.0.0.1:{tcp.server_address[1]}"
    try:
        yield base, service, upstream, store, ledger, redactor
    finally:
        tcp.shutdown()
        ledger.close()


def _artifact_texts(store: ObjectStore, ledger: Ledger) -> list[str]:
    rows = ledger.raw_connection.execute(
        "SELECT request_object_ref, response_object_ref FROM external_call_attempts"
    ).fetchall()
    texts = []
    for row in rows:
        for ref in (row["request_object_ref"], row["response_object_ref"]):
            if ref:
                texts.append(store.get_bytes(ref).decode("utf-8"))
    return texts


async def test_the_real_tavily_client_runs_through_the_boundary(provider):
    base, service, upstream, store, ledger, redactor = provider
    client = ProviderClient(base_url=base, token=TOKENS["steward"])

    # The existing capture client, unchanged, with the provider as its transport.
    capture = TavilyCaptureClient(
        client.tavily_transport(task_id="T1", call_key="qs-1"),
        TavilyParams(), PROVIDER_KEY_PLACEHOLDER,
    )
    captured = await capture.search("T1", "what is a cat")

    assert captured.request_id == "req-1"
    assert captured.usage == {"credits": 1.0}
    assert len(captured.response.results) == 1
    assert captured.response.results[0].raw_content == "RAW A"
    # The real key went out exactly once, to the upstream, and to nowhere else.
    assert upstream.calls[0]["body"]["api_key"] == FAKE_TAVILY
    for text in _artifact_texts(store, ledger):
        assert FAKE_TAVILY not in text and redactor.is_clean(text)


async def test_the_real_judge_client_runs_through_the_boundary(provider):
    base, service, upstream, store, ledger, redactor = provider
    client = ProviderClient(base_url=base, token=TOKENS["evaluator"])
    judge = DeepSeekJudge(
        client.deepseek_transport(op_class="JUDGE_REPORT", work_key="W1"),
        "deepseek-chat", PROVIDER_KEY_PLACEHOLDER,
    )
    response = await judge.judge("system", "user")

    assert response.data == {"verdict": "ok"}
    assert response.returned_model == "deepseek-chat"
    assert response.system_fingerprint == "fp_1"
    assert upstream.calls[0]["headers"]["Authorization"] == f"Bearer {FAKE_DEEPSEEK}"
    for text in _artifact_texts(store, ledger):
        assert FAKE_DEEPSEEK not in text and redactor.is_clean(text)


async def test_a_credential_echoed_back_by_the_upstream_never_lands_in_an_artifact(provider):
    base, service, upstream, store, ledger, redactor = provider
    upstream.mode = "leak"

    steward = ProviderClient(base_url=base, token=TOKENS["steward"])
    capture = TavilyCaptureClient(
        steward.tavily_transport(task_id="T2", call_key="qs-2"),
        TavilyParams(), PROVIDER_KEY_PLACEHOLDER,
    )
    with pytest.raises(Exception):
        await capture.search("T2", "leaky")

    evaluator = ProviderClient(base_url=base, token=TOKENS["evaluator"])
    judge = DeepSeekJudge(
        evaluator.deepseek_transport(op_class="JUDGE_TRUTH"), "deepseek-chat",
        PROVIDER_KEY_PLACEHOLDER, max_retries=0,
    )
    with pytest.raises(JudgeUnavailable):
        await judge.judge("system", "user")

    for text in _artifact_texts(store, ledger):
        assert FAKE_TAVILY not in text
        assert FAKE_DEEPSEEK not in text
        assert redactor.is_clean(text)
    dumped = json.dumps(service.events)
    assert FAKE_TAVILY not in dumped and FAKE_DEEPSEEK not in dumped


async def test_the_runner_token_is_refused_on_a_credential_route(provider):
    base, *_ = provider
    runner = ProviderClient(base_url=base, token=TOKENS["runner"])
    status, payload = await runner._post("/v1/tavily/search", {
        "api_key": PROVIDER_KEY_PLACEHOLDER, "query": "x", "search_depth": "advanced",
        "include_raw_content": "markdown", "include_answer": False, "include_usage": True,
        "max_results": 8,
    })
    assert status == 403
    assert "may not call" in payload["error"]


async def test_inference_is_tagged_by_the_cell_path(provider):
    base, service, upstream, *_ = provider
    runner = ProviderClient(base_url=base, token=TOKENS["runner"])
    await runner.register_cell(
        cell_token="cell-abc123", run_id="RUN1", task_id="T1", arm_id="H",
        variant_id="H02", replicate_id="0", work_key="WK1",
    )
    assert runner.cell_base_url("cell-abc123").endswith("/v1/cell/cell-abc123")

    payload = await runner.chat_completions(
        {"model": "qwen-research", "messages": [{"role": "user", "content": "hi"}]},
        cell_token="cell-abc123",
    )
    assert payload["choices"][0]["message"]["content"] == "hi"
    event = [e for e in service.events if e["kind"] == "INFERENCE_COMMITTED"][-1]
    assert event["op_class"] == "RESEARCHER_REACT"
    assert event["task_id"] == "T1" and event["variant_id"] == "H02"

    attestation = await runner.canary_audit(work_keys=["WK1"])
    assert attestation["work_keys"] == ["WK1"]
    assert attestation["work"][0]["ops"][0]["op_class"] == "RESEARCHER_REACT"
    assert attestation["work"][0]["ops"][0]["prompt_tokens"] == 3
    assert (
        attestation["work"][0]["settled_gpu_seconds"]
        == pytest.approx(
            attestation["work"][0]["ops"][0]["settled_gpu_seconds"])
    )
    assert attestation["open_attempts"] == 0
    assert "prompt_sha256" not in json.dumps(attestation)


async def test_empty_canary_work_set_fails_locally_without_an_http_request():
    client = ProviderClient(
        base_url="http://127.0.0.1:1", token=TOKENS["runner"])
    with pytest.raises(
        ProviderCallError, match="requires at least one work_key"
    ):
        await client.canary_audit(work_keys=[])


async def test_non_runner_role_cannot_call_the_canary_audit(provider):
    base, *_ = provider
    evaluator = ProviderClient(base_url=base, token=TOKENS["evaluator"])
    status, payload = await evaluator._post(
        "/v1/canary/audit", {"work_keys": ["WK1"]})
    assert status == 403
    assert "may not call" in payload["error"]


async def test_an_unknown_route_is_a_404_not_a_guess(provider):
    base, *_ = provider
    runner = ProviderClient(base_url=base, token=TOKENS["runner"])
    status, _ = await runner._post("/v1/anything", {})
    assert status == 404


async def test_health_and_readiness_answer(provider):
    base, *_ = provider
    runner = ProviderClient(base_url=base, token=TOKENS["runner"])
    assert await runner.healthy() is True
    assert await runner.ready() is True
