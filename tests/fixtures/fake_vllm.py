"""A stand-in for vLLM's ``/metrics`` and ``/v1/completions``, for driving ``ProxyHost``.

The metrics text has the shape vLLM 0.24.0 actually serves -- labelled gauges, ``_total``
counters beside them, a ``process_start_time_seconds`` line -- so the parser is exercised on the
real format rather than on a tidied one. Thread-safe: the host sends from worker threads.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from shapeflow.broker.hosts.agent import AgentHost
from shapeflow.broker.hosts.proxy import ProxyHost, RequestBody
from shapeflow.contracts.offer import Form, Offer


class FakeVllm:
    def __init__(self, *, start_time: float = 1791563561.79, running: float = 0.0,
                 waiting: float = 0.0, kv: float = 0.0,
                 reply: Callable[[dict], str] | None = None) -> None:
        self.start_time = start_time
        self.running, self.waiting, self.kv = running, waiting, kv
        self.reply = reply or (lambda body: "ok")
        #: Called with each completion body before it is answered; tests use it to block.
        self.gate: Callable[[dict], None] | None = None
        self.fail: Callable[[dict], bool] | None = None
        self.requests: list[dict] = []
        self.scrapes = 0
        self._lock = threading.Lock()

    def metrics(self) -> str:
        with self._lock:
            self.scrapes += 1
        labels = 'engine="0",model_name="fake"'
        return (
            "# HELP process_start_time_seconds Start time of the process.\n"
            "# TYPE process_start_time_seconds gauge\n"
            f"process_start_time_seconds {self.start_time!r}\n"
            "# TYPE vllm:num_requests_running gauge\n"
            f"vllm:num_requests_running{{{labels}}} {self.running}\n"
            f"vllm:num_requests_waiting{{{labels}}} {self.waiting}\n"
            f'vllm:num_requests_waiting_by_reason{{{labels},reason="capacity"}} {self.waiting}\n'
            f"vllm:kv_cache_usage_perc{{{labels}}} {self.kv}\n"
            f"vllm:prefix_cache_queries_total{{{labels}}} 123.0\n"
            f"vllm:prefix_cache_queries_created{{{labels}}} 1.79e+09\n"
        )

    def completions(self, body: dict) -> dict:
        with self._lock:
            self.requests.append(body)
            n = len(self.requests)
        if self.gate is not None:
            self.gate(body)
        if self.fail is not None and self.fail(body):
            raise ConnectionError("fake engine refused the connection")
        return {
            "id": f"cmpl-fake-{n}",
            "choices": [{"text": self.reply(body), "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(body["prompt"].split()), "completion_tokens": 1},
        }

    def host(self, **kwargs) -> ProxyHost:
        return ProxyHost(scrape=self.metrics, send=self.completions, model="fake", **kwargs)


def bodies_for(offer: Offer) -> dict[Form, tuple[RequestBody, ...]]:
    """One distinguishable body per descriptor, so a test can tell which plan was sent."""
    out: dict[Form, tuple[RequestBody, ...]] = {}
    for form in offer.forms():
        plan = offer.plan_for(form)
        descriptors = plan.requests if form is Form.P0 else (plan.selector,)
        out[form] = tuple(
            RequestBody(prompt=f"{offer.item_id} {form.value} {i}", max_tokens=d.max_tokens)
            for i, d in enumerate(descriptors))
    return out


def proxy_host_with(offers, **kwargs) -> ProxyHost:
    host = FakeVllm().host(**kwargs)
    for offer in offers:
        host.submit(offer, bodies_for(offer))
    return host


def agent_host_with(offers, engine: FakeVllm | None = None, **kwargs) -> AgentHost:
    host = AgentHost(scrape=(engine or FakeVllm()).metrics, **kwargs)
    for offer in offers:
        host.submit(offer)
    return host
