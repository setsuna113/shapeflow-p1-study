# ProxyHost against a live vLLM — run record, 2026-10-09

Branch `feat/proxy-host-live-vllm`, based on tag `archive/2026-08-06` (`69e7f22`).

## What ran

The broker ran **outside an unmodified vLLM**, as a separate process talking HTTP to it:

- `ProxyHost` (`src/shapeflow/broker/hosts/proxy.py`) — snapshot = scrape of vLLM `/metrics`
  (`num_requests_running`, `num_requests_waiting`, `kv_cache_usage_perc`) plus exact local
  in-flight count; epoch = API server `process_start_time_seconds`; materialize gates on
  snapshot version and age, then sends each chosen sub-request as its own `/v1/completions`.
- Policy: `EngineGatedBroker(GreedyJointBroker)` (`reference.py`) — defers the whole tick while
  `engine.num_requests_waiting > 4`, otherwise the unchanged greedy joint choice.
- Driver: `scripts/proxy_live.py` loops `run_tick`, releasing replayed page batches as offers
  (`src/shapeflow/broker/replay.py`): P0 = one summary request per page, P1 = one selector
  request (production `render_selector_prompt`, contract P1_ID) + CPU render.

Stack: RunPod community 1× RTX 3090 24 GB, image `vllm/vllm-openai:v0.24.0-cu129`
(vLLM 0.24.0, torch 2.11.0+cu129), `Qwen/Qwen3-14B-AWQ` @ `31c69efc29464b6bb0aee1398b5a7b50a99340c3`.
Serve flags mirror the repo's operational unit: `--max-model-len 16384
--gpu-memory-utilization 0.90 --max-num-seqs 256 --enable-prefix-caching
--enable-chunked-prefill --seed 0` (`_engine/serve.sh`). KV cache: 66,848 tokens.
Corpus: `Salesforce/wikitext` wikitext-2-raw-v1 test @ `b08601e04326c79dfdd32d625aee71d232d685c3`,
61 articles, each cut to ≤ 1200 tokens. Requests at temperature 0, seed 0.

## Runs

| run | items | capacity | P0 cap | ticks | gate shut | admitted (P0/P1) | requests | prompt-token est. = vLLM usage | wall |
|---|---|---|---|---|---|---|---|---|---|
| `smoke` (pre-final code) | 4 | 8192 | 256 | 40 | 0 | 4 (4/0) | 11 | 11/11 | 9.8 s |
| `live_20261009T164524Z` | 48 | 8192 | 1024 | 3600 | 206 | 46 (5/41) | 46 | 46/46 | 900 s (timeout) |
| `live_20261009T170053Z_cap16k` | 48 | 16384 | 1024 | 395 | 217 | 48 (5/43) | 48 | 48/48 | 98.5 s |
| `live_20261009T170301Z_p0multi` | 8 | 16384 | 256 | 67 | 0 | 8 (8/0) | 21 | 21/21 | 16.5 s |

All runs: 0 failed requests, 0 stale retries, 0 fail-closed ticks, 0 solver budget overruns.
Snapshot age at commit 4.3–34.2 ms (median 7.3–9.7 ms per run); scrape 6.9–9.3 ms median.
`smoke` predates the switch to the repo's chunker and selector cap; the other three ran the
branch code.

## What the evidence shows, and where

- **Decisions read live engine state.** `ticks.jsonl` `reason` alternates between
  `engine gate open: N waiting; greedy joint: …` and `engine gate shut: N waiting > 4`;
  `snapshots.jsonl` holds the scraped gauges behind each.
- **Only the chosen form is sent.** Per item, `requests.jsonl` contains P0 or P1 requests, never
  both; counts match `ticks.jsonl` `result.dispatched`.
- **P0 siblings enter vLLM independently.** `live_…_p0multi/requests.jsonl`: each multi-page P0
  sends N separate requests within 0.3–11 ms of commit, and they finish out of send order
  (e.g. H005, H006, H007).
- **Token estimates are exact.** `est_prompt_tokens == prompt_tokens` for every request; the
  chat wrapper is byte-identical to the model's template with `enable_thinking=False`.
- **Snapshot age is measured, not assumed.** `gate.jsonl` records the age of every committed
  snapshot against the 100 ms ceiling.

## Findings and limits

- **Starvation under the reference policy.** With `capacity_tokens=8192`, the two 5-page offers
  (H003, H019) had both plans above capacity and were deferred every tick until the 900 s
  timeout. Deferral keeps them in the ITT set, but `GreedyJointBroker` has no aging or
  oversize path, so they never run.
- **Selector JSON.** 5/41 and 5/43 P1 outputs were invalid JSON (ids unquoted); the render
  rejected them. The production stack used guided decoding (`guided_decoding: true`); this
  replay sent unconstrained completions. The C2 republish-as-P0 path is not implemented here.
- **Staleness was not exercised live.** No live snapshot exceeded the age ceiling and the
  engine did not restart; the stale, restart and superseded paths are covered by unit tests
  and the conformance suite against a fake engine.
- **Placeholders, not calibrated values:** `capacity_tokens`, `max_waiting=4`, the 100 ms age
  ceiling, and `est_decode = max_tokens`. The P0 cap was 1024 (vendor default 8192, under
  which every offer prices P1 cheaper). The P0 prompt is a stand-in for vendor's
  summarization prompt (not in this tree).
- **Not an evaluation.** No baseline arm, no quality or cost comparison.

## Files

Raw evidence is not in git; it stays on the experiment host under `runs/proxy-live-2026-10-09/`:
`<run>/run.json` (config, versions, digests, summary) · `offers.jsonl` · `ticks.jsonl` ·
`snapshots.jsonl` · `gate.jsonl` · `requests.jsonl` · `renders.jsonl`; `_engine/` holds `serve.sh`, the vLLM log and a pre-run `/metrics` capture.
Archive sha256 `0440c8099d59561475e51717ca4798ba2d910aa2019ec21b4f40fb3c492091a9`.
