# Real ODR agents offer both forms; a broker outside vLLM chooses — run record, 2026-10-09

Branch `feat/proxy-host-live-vllm`, based on tag `archive/2026-08-06` (`69e7f22`). The
replay driver against the same engine is recorded in `BROKER_PROXYHOST_LIVE.md`.

## What changed in the agent

At every H handoff (a researcher turn's page batch, `vendor_hooks.reduce_published_batch`) and C
handoff (`run_close_strategy`), the pinned, patched ODR agent now:

1. prices both forms without sending anything (`strategies/offer_pricing.py`: vendor's own
   summarization/compression prompts for P0; the production selector strategy run with a
   recording probe for P1);
2. submits an `Offer` to the broker service (`broker/service.py`, `broker/client.py`);
3. suspends on the decision — the hook is awaited inline, so the researcher does not proceed;
4. runs only the chosen form (`strategies/brokered.py`): vendor's P0, or the arm's P1, whose
   failures still fall back as before (C2 at H, vendor compression at C).

The broker (`AgentHost` + `DeadlineBroker(EngineGatedBroker(GreedyJointBroker))`) runs in its own
process, reads vLLM's `/metrics` each tick, and releases the waiting agent on commit.

## Setup

RunPod community RTX 3090 24 GB; `vllm/vllm-openai:v0.24.0-cu129`; `Qwen/Qwen3-14B-AWQ` @
`31c69ef`; flags as `scripts/start_engine.sh` with `stack.yaml` `isolation.operational` (32k
window, 0.90 memory, 256 seqs, APC + chunked prefill on, hermes tool calls) — `_engine/serve.sh`.
World: BC+ corpus @ `b27b02b` with the benchmark's prebuilt BM25 index @ `b3f37f7`
(`scripts/bm25_service.py`, Anserini 2.3.0) — **not** the study's dense world. Tasks: the first
12 of the `pilot_competence` layer (`tasks.provenance.json`; split files match the prereg pins).
Arms on offer: H P1 = `LLM-PROMPTVIEW` (whole_batch), C P1 = `C01`. Broker placeholders:
capacity 262,144 tokens, gate at 4 waiting, snapshot age ceiling 100 ms, deadline 60 s.

## Runs

| run | agents × tasks | cells ok | offers (H/C) | decided | wall |
|---|---|---|---|---|---|
| `smoke_1x1` | 1 × 1 | 1/1 | 4 (3/1) | H/P1 3, C/P1 1 | — |
| `live_4x12` | 4 × 12 | 12/12 | 46 (34/12) | H/P1 34, C/P1 3, C/P0 9 | 996 s |

`live_4x12` audit (`run.json` → `audit`, per item in `items.json`):

- **Offers came from the agents.** 46/46 offers name a checkpoint the agent's own cell record
  captured, ordered checkpoint → `BROKER_OFFER` → `BROKER_DECISION`; 46/46 broker-received offers
  match the agent's priced offer by digest. P1 was offered 37 times; 9 C offers were P0-only
  (`VISIBLE_SOURCE_REGISTRY`).
- **Decisions on live engine state.** 45 ticks, one engine epoch, commits 3.2–26.5 ms after the
  scrape (median 3.8), 0 stale retries, 0 fail-closed, 0 budget overruns. Re-running the policy
  on each logged (offers, snapshot) reproduces 45/45 decision digests; with the waiting gauge set
  above the gate, all 45 decisions change.
- **Only the chosen form was sent.** All 48 treatment requests the providers committed join to
  an offer, none before its ticket. H: 34/34 P1 items sent their selector and no summaries. C: 9
  P0 items sent compression requests whose prompt hash equals the priced one (9/9); of 3 P1
  items, 2 failed preflight (rendered 605 and 521 > 512 tokens) and fell back to vendor
  compression, both attributed.
- **Pricing.** Selector prompt-token estimates equal vLLM's `prompt_tokens` 37/37; compression
  (multi-turn template rendered by hand) within 8 tokens, 0/11 exact.

## Findings and limits

- **Almost no joint decisions.** 44 of 45 ticks had a single pending offer; one tick decided
  offers from two agents. Offers resolve in ~150 ms (median agent wait) against a 250 ms tick, and
  handoffs are sparse within ~5-minute tasks, so four agents rarely overlap.
- **The engine gate never shut.** Load stayed light (waiting ≤ 1, KV ≤ 60%), so engine state
  bounded *when* admission could happen but changed no decision in this run; the counterfactual
  replay shows it would have deferred all 45 under a long queue.
- **Form choice follows plan weight, not load.** With `est_decode = max_tokens` P0 is priced at
  8192 per page, so whenever P1 was offered it was chosen. No load-aware form policy exists.
- **Pricing costs latency.** Median 0.5 s (max 1.8 s) per handoff, mostly the probe rebuilding the
  view; median wait for the decision 152 ms.
- BM25 world and placeholders as above; no baseline arm; nothing here measures answer quality or
  whether brokering helps.

## Files

Raw evidence is not in git; it stays on the experiment host under
`runs/brokered-agents-2026-10-09/`: `live_4x12/{run.json, items.json, broker/*.jsonl,
worker-*/{cells,agent_offers,requests}.jsonl}`, `smoke_1x1/…`, `_engine/{serve.sh, vllm.log, bm25.log}`, `tasks*.json*`. Archive sha256
`96953ca1c91bbf1a6aebe7484a719e80458e9ce10a94d7c8dd76bef6e9f05355`. The decrypted query file
(contains answers) was not copied and was destroyed with the pod.
