"""The closed set of features observable at broker decision time.

`OBSERVABILITY_v1.md` is the contract; this is its executable form. Nothing outside
:data:`OBSERVABLE` may reach a cost model, a predictor or a broker decision.

Two enforcement points, because either alone leaks. :class:`ClosedFeatureMap` raises at runtime on
an unregistered key, which catches paths that execute; ``tools/ci/check_observability.py`` walks
the AST of every broker/predictor/cost module, which catches paths that do not.

The features are grouped by *how they can be wrong*, not by where they come from, because that is
what decides how they may be used:

- ``STATE`` gauges describe the engine now.
- ``COUNTER`` totals are monotone; usable as a rate over an interval, never as a level.
- ``COMPLETED`` histograms summarize requests that already finished. They are load evidence and
  are **not** properties of the offer being decided -- reading time-to-first-token as a
  prediction about *this* offer is the unobservable-quantity error wearing a real metric's name.
- ``LOCAL`` is broker-owned and exact.
- ``OFFER`` is intrinsic to the work and needs no engine call.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator, Mapping

from .versions import assert_contract

CONTRACT = "OBSERVABILITY_v1.md"
CONTRACT_SHA256 = "2633ef1f2dee07a4c3d307481013bc19429d897be7609b8b287ba93ce057e5ce"

__all__ = ["Kind", "ObservableFeature", "OBSERVABLE", "ClosedFeatureMap", "UnobservableFeature"]


class Kind(Enum):
    STATE = "STATE"
    COUNTER = "COUNTER"
    COMPLETED = "COMPLETED"
    LOCAL = "LOCAL"
    OFFER = "OFFER"


@dataclass(frozen=True)
class ObservableFeature:
    name: str
    kind: Kind
    source: str


def _f(name: str, kind: Kind, source: str) -> ObservableFeature:
    return ObservableFeature(name=name, kind=kind, source=source)


_FEATURES = (
    # 1. engine state gauges
    _f("engine.num_requests_running", Kind.STATE, "vllm:num_requests_running"),
    _f("engine.num_requests_waiting", Kind.STATE, "vllm:num_requests_waiting"),
    _f("engine.num_requests_waiting_by_reason", Kind.STATE, "vllm:num_requests_waiting_by_reason"),
    _f("engine.kv_cache_usage_perc", Kind.STATE, "vllm:kv_cache_usage_perc"),
    # 2. engine cumulative counters
    _f("engine.prefix_cache_queries", Kind.COUNTER, "vllm:prefix_cache_queries"),
    _f("engine.prefix_cache_hits", Kind.COUNTER, "vllm:prefix_cache_hits"),
    _f("engine.prompt_tokens_cached", Kind.COUNTER, "vllm:prompt_tokens_cached"),
    _f("engine.num_preemptions", Kind.COUNTER, "vllm:num_preemptions"),
    _f("engine.iteration_tokens_total", Kind.COUNTER, "vllm:iteration_tokens_total"),
    # 3. completed-request histograms -- load only
    _f("engine.ttft_seconds", Kind.COMPLETED, "vllm:time_to_first_token_seconds"),
    _f("engine.inter_token_latency_seconds", Kind.COMPLETED, "vllm:inter_token_latency_seconds"),
    _f("engine.request_queue_time_seconds", Kind.COMPLETED, "vllm:request_queue_time_seconds"),
    # 5. broker-local, exact
    _f("local.inflight_total", Kind.LOCAL, "broker"),
    _f("local.inflight_by_op_class", Kind.LOCAL, "broker"),
    _f("local.queue_depth_by_op_class", Kind.LOCAL, "broker"),
    _f("local.offer_age_ns", Kind.LOCAL, "broker"),
    _f("local.deadline_slack_ns", Kind.LOCAL, "broker"),
    # 6. offer-intrinsic
    _f("offer.boundary", Kind.OFFER, "offer"),
    _f("offer.batch_size", Kind.OFFER, "offer"),
    _f("offer.page_tokens_total", Kind.OFFER, "offer"),
    _f("offer.page_tokens_max", Kind.OFFER, "offer"),
    _f("offer.page_tokens_p90", Kind.OFFER, "offer"),
    _f("offer.chunk_count", Kind.OFFER, "offer"),
    _f("offer.chunk_len_mean", Kind.OFFER, "offer"),
    _f("offer.chunk_len_p90", Kind.OFFER, "offer"),
    _f("offer.retrieval_score_mean", Kind.OFFER, "offer"),
    _f("offer.retrieval_rank_mean", Kind.OFFER, "offer"),
    _f("offer.query_page_overlap", Kind.OFFER, "offer"),
    _f("offer.table_list_density", Kind.OFFER, "offer"),
    _f("offer.language", Kind.OFFER, "offer"),
    _f("offer.researcher_iteration", Kind.OFFER, "offer"),
    _f("offer.supervisor_iteration", Kind.OFFER, "offer"),
    _f("offer.selector_prefill_tokens", Kind.OFFER, "offer"),
    _f("offer.est_uncached_prefill", Kind.OFFER, "offer"),
    _f("offer.est_cached_prefill", Kind.OFFER, "offer"),
    _f("offer.est_decode", Kind.OFFER, "offer"),
)

#: name -> feature. The closed set.
OBSERVABLE: Mapping[str, ObservableFeature] = {f.name: f for f in _FEATURES}


class UnobservableFeature(KeyError):
    """A decision tried to read something the broker could not have known at the tick."""


class ClosedFeatureMap(Mapping[str, Any]):
    """A feature mapping that refuses unregistered keys.

    Refuses on *read* rather than only on construction, so a decision path that reaches for an
    unobservable quantity fails there, naming the feature, instead of silently receiving a
    default. ``.get`` is deliberately not softened: a missing observable is a real condition the
    caller must handle, but an unregistered one is a contract violation either way.
    """

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, Any]) -> None:
        unknown = sorted(set(values) - set(OBSERVABLE))
        if unknown:
            raise UnobservableFeature(
                f"not in the observability contract: {unknown}. Add it to "
                f"{CONTRACT} and this registry first, or the broker is deciding on something it "
                "could not have known at the tick."
            )
        self._values = dict(values)

    def __getitem__(self, key: str) -> Any:
        if key not in OBSERVABLE:
            raise UnobservableFeature(
                f"{key!r} is not in the observability contract; broker decisions may only read "
                f"registered features ({len(OBSERVABLE)} of them)"
            )
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


assert_contract(CONTRACT, CONTRACT_SHA256)
