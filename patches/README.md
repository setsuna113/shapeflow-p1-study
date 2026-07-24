# patches/ — the only sanctioned modification to vendor ODR

`vendor/open_deep_research` is pinned at `408da442a661ea5e40a6163329f82e3f22628949` and is
read-only. `odr_p1_hooks.patch` is applied to a throwaway materialization under `.build/` by
bootstrap (`git archive` → apply → `uv sync` resolves ODR from there). The submodule tree is
never edited in place, and bootstrap asserts `open_deep_research.__file__` lives under the
patched materialization so an unpatched global copy can never be imported by accident.

This file specifies exactly what the patch does. It is authored here rather than generated
blind because the patch must be validated against the **frozen** stack (the pinned
langgraph/langchain resolved by `uv sync --frozen` on the run host), not against whatever
langchain a dev machine happens to have.

> **Status: specification only.** The patch does not exist yet. Block 2A froze the interfaces
> it will use; Block 2B writes it. `bootstrap_and_run.sh` hard-fails on the missing file.

## Where the boundary actually is

This is the part the earlier version of this document got wrong, so it is worth stating
precisely.

`utils.tavily_search` receives a *list of queries* and deduplicates their results into
`unique_results` (vendor step 2, `utils.py:47`). That set is one tool call's results. It is
**not** the assistant turn's batch: a single assistant turn can emit several sibling tool
calls, and they are only brought together in `deep_researcher.researcher_tools`, by
`asyncio.gather` (`deep_researcher.py:473`). The `ToolMessage` list is built immediately after,
in the pinned `zip(observations, tool_calls)` order, and published by one `Command`.

So there are three candidate capture points and only one is correct:

| Point | Sees | Verdict |
|---|---|---|
| inside `tavily_search` | its own call's results only | too early — cannot see siblings |
| top of `researcher_tools` | the tool calls, no results | too early — searches have not run |
| **after `gather`, before `ToolMessage`** | the whole batch, unpublished | **correct** |

`AGENTS.md §3` and plan §11.1 both say the publish unit is the whole assistant turn's sibling
batch. The middle column is why.

## What the patch may change (and nothing else)

Per plan §11.2, confined to:

1. **Defer, don't transform, inside `tavily_search`.** When a strategy is bound
   (`shapeflow_p1.odr.hooks.current_strategies()`), skip vendor steps 3–6 — the
   `summarize_webpage` calls and the string formatting — and return a tagged
   `DeferredPageBatch` carrying `unique_results` with each `raw_content[:max_content_length]`
   written to the object store. No transform, no publish, and the raw registry H needs is still
   intact. With no strategy bound, vendor's path runs **unchanged**, including its own
   `asyncio.gather` over the summarization tasks — the hooks-off fast path must remain
   byte-identical *and* keep vendor's concurrency, because an explicit-P0 arm that ran the
   summaries sequentially would have different timing than the vendor baseline it is compared
   to.

2. **Capture, transform and publish the whole batch in `researcher_tools`.** After `gather`
   returns, if any observation is a `DeferredPageBatch`:

   ```
   build HCheckpoint      assistant AIMessage + ordered sibling tool calls + each search
                          call's complete vendor-visible result set + non-search siblings'
                          outputs verbatim + failure states + sampling envelope
   await strategy.page.transform_tool_batch(...)      one call, whole batch
   preflight the whole batch, stage it                nothing published yet
   publish                one Command update, observations refilled by tool_call_id in the
                          original zip order
   ```

   On **any** P1 failure — selector error, preflight failure, timeout, budget refusal — discard
   every staged P1 output, keep the P1 cost already spent on the ledger, and re-run vendor's
   `summarize_webpage` path for the entire batch. Never `[P1(A), P0(B)]`: a hybrid batch is not
   an arm, and it would be scored as one.

   A component trial and an end-to-end run diverge here: the trial records the P1 failure and
   marks the sample failed (falling back would hide the failure rate being measured); the
   end-to-end run falls back, because the arm under test is "P1 with its fallback".

3. **`RESEARCHER_CLOSE` as its own boundary.** In `compress_research`, **before** the in-place
   `researcher_messages.append(compress_research_simple_human_message)` (`deep_researcher.py:538`),
   clone `researcher_messages` losslessly and build the `CCheckpoint`. This hook is reached
   independently of the page batch — a researcher can close having published none at all
   (`ResearchComplete` on the first turn, or a no-tool exit) — so it must not be implemented as
   a continuation of the H path, or the cheapest runs would silently skip it.

4. **Classify and record the close reason** with `odr.close_reason.classify_close`, on the same
   signals vendor uses. All three exit paths enter the same treatment policy.

5. **Propagate identity via `ContextVar`**, bound with `hooks.strategies_bound` so the reset is
   in a `finally`. Researchers run concurrently; a binding that survived an exception would
   execute one arm's strategy while another arm's id was recorded — a mislabelled observation,
   not a crash. `CancelledError` propagates and the incomplete work is recorded; it is never
   converted into "P1 failed, use P0", which would fabricate a P0 result for abandoned work.

6. **Point the model base URL at the local telemetry proxy.** Done via config/env
   (`OPENAI_BASE_URL`); the patch only threads the per-call config through.

## What the patch must NOT change under P0

Prompt bytes, tool schemas, join/publish order, retry counts, token limits, supervisor and
researcher control flow, result formatting, fallback behaviour, and the concurrency structure
(both the outer sibling `gather` and vendor's inner summarization `gather`). With no strategy
bound, the graph must be byte-for-byte vendor.

## The fused variant is not here

`C05-FUSED-EXT` changes the stopping policy, not just the reducer, and `ResearchComplete` has
an empty schema. It therefore cannot be a close-after hook. It is a separate P1-only graph
adapter (a new tool such as `ResearchCompleteWithSelection`) with its own arm, checkpoint schema
and P0 comparator, and it keeps the dedicated-selector fallback for the two exit paths that
never call the fused tool. It is never attributed as compressor-only.

## The parity gate

`tests/integration/test_p0_parity.py` drives both graphs with a deterministic **mock**
OpenAI-compatible model and a mock search backend, records a `RunTrace` from each
(`odr.p0_parity`), and asserts equality at two levels:

- **vendor vs patched-hooks-off** — the patch is inert when nothing is bound.
- **patched-hooks-off vs patched-with-explicit-P0-strategy** — the deferred/refill path itself
  introduces no difference. This one catches what the first cannot: a bug that lives in the new
  code path rather than in its absence.

`RunTrace` records `publish_batches`, not a flat message list, because `[[A,B]]` and
`[[A],[B]]` are the distinction the design turns on and a flat list cannot express it. Each
batch also carries its `goto`, its state-update digest, its checkpoint digest and any fallback,
so a patched graph that produced identical bytes while routing differently — or while quietly
falling back — does not compare equal.

Both a CPU fixture and a small-model fixture are exercised. **P0 parity failure blocks all GPU
screening** — the launch gate refuses to proceed.

Coverage the gate must include: mixed search/non-search siblings; a tool raising; cancellation
mid-batch; all three close paths; zero partial publish when P1 fails after producing its first
sibling output; and `publish_batches: [[A,B]] != [[A],[B]]`.
