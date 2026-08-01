# Observability contract — v1

Freeze-1 §5.1(c). The closed list of what is observable at broker decision time. **A quantity not
named here may not enter a cost model, a predictor feature, or any broker decision.**

The rule exists because the tempting features are the unobservable ones. How long this request
will actually take, whether this page turns out to matter, what the trajectory does next — all
are knowable in an offline analysis and none are knowable at the tick. A broker built on them
would report headroom that no deployable system can reach.

Enforced twice, deliberately: the feature map raises on an unregistered key at runtime, and a
static check rejects any broker, cost-model or predictor module that names a feature outside this
registry. Runtime alone would only catch paths that execute; static alone would miss dynamic
lookups.

Names below were read from the pinned engine (vLLM 0.24.0), not from documentation.

---

## 1. Engine state — gauges

Describe the engine **now**, at the tick. These are the decision-time state features.

| Feature | Source |
|---|---|
| `engine.num_requests_running` | `vllm:num_requests_running` |
| `engine.num_requests_waiting` | `vllm:num_requests_waiting` |
| `engine.num_requests_waiting_by_reason` | `vllm:num_requests_waiting_by_reason` |
| `engine.kv_cache_usage_perc` | `vllm:kv_cache_usage_perc` |

## 2. Engine cumulative counters — usable as deltas or rates

Monotone totals. Usable as a rate over an interval, never as a level.

| Feature | Source |
|---|---|
| `engine.prefix_cache_queries` | `vllm:prefix_cache_queries` |
| `engine.prefix_cache_hits` | `vllm:prefix_cache_hits` |
| `engine.prompt_tokens_cached` | `vllm:prompt_tokens_cached` |
| `engine.num_preemptions` | `vllm:num_preemptions` |
| `engine.iteration_tokens_total` | `vllm:iteration_tokens_total` |

## 3. Completed-request histograms — load signals only

These summarize requests that have **already finished**. They are legitimate evidence about
current load, and they are **not** properties of the item being decided. Reading
`time_to_first_token` as a prediction about *this* offer is exactly the unobservable-quantity
error this contract exists to prevent, dressed as a real metric.

| Feature | Source |
|---|---|
| `engine.ttft_seconds` | `vllm:time_to_first_token_seconds` |
| `engine.inter_token_latency_seconds` | `vllm:inter_token_latency_seconds` |
| `engine.request_queue_time_seconds` | `vllm:request_queue_time_seconds` |

## 4. Per-request accounting — after the fact

Available on a response, so usable for cost calibration and never for deciding the request that
produced it.

| Feature | Source |
|---|---|
| `usage.prompt_tokens` | response usage |
| `usage.completion_tokens` | response usage |
| `usage.cached_prompt_tokens` | `usage.prompt_tokens_details.cached_tokens` |

## 5. Broker-local state

Exactly known at the tick, because the broker owns it. No staleness.

| Feature | Meaning |
|---|---|
| `local.inflight_total` | requests dispatched and not yet returned |
| `local.inflight_by_op_class` | the same, split by op class |
| `local.queue_depth_by_op_class` | offers pending, by op class |
| `local.offer_age_ns` | now minus submission time |
| `local.deadline_slack_ns` | deadline minus now, where a deadline exists |

## 6. Offer-intrinsic features

Properties of the work itself, computable from the offer without asking the engine anything.
These are the features the B1 study fills in; each must appear here before a predictor may use it.

| Feature | Meaning |
|---|---|
| `offer.boundary` | H or C |
| `offer.batch_size` | pages in the gather batch (H) |
| `offer.page_tokens_total` / `offer.page_tokens_max` / `offer.page_tokens_p90` | candidate size |
| `offer.chunk_count` / `offer.chunk_len_mean` / `offer.chunk_len_p90` | fragmentation |
| `offer.retrieval_score_mean` / `offer.retrieval_rank_mean` | retrieval signal |
| `offer.query_page_overlap` | lexical overlap, query against candidate text |
| `offer.table_list_density` | structural density |
| `offer.language` | detected language |
| `offer.researcher_iteration` / `offer.supervisor_iteration` | trajectory position |
| `offer.selector_prefill_tokens` | P1-plan prefill size |
| `offer.est_uncached_prefill` / `offer.est_cached_prefill` / `offer.est_decode` | per-plan demand |

## Staleness

Sections 1–3 come from an engine scrape and are therefore **stale by construction** on any host
that does not sit inside the scheduler. Section 5 is exact. A host must report the age of its
snapshot, and the pre-registered ceiling on that age is a **validity criterion for the broker
itself**, checked at G0′ — because a broker deciding on features uncorrelated with actual engine
state is decorative, and would still produce a full ledger and a plausible paper.

## Explicitly not observable

Named so they cannot be reintroduced by good intentions:

- the realized service time, TTFT or decode length **of the offer being decided**;
- whether this decision will turn out to have lost evidence — that is the label, not a feature;
- anything derived from gold answers, qrels, evidence sets or negative document sets, which are
  evaluator-only and physically separated;
- any property of a *later* step in the same trajectory;
- wall-clock time of day, and any host-identifying value that would let a model learn the lane
  rather than the workload.
