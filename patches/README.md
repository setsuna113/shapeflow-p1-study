# patches/ — the only sanctioned modification to vendor ODR

`vendor/open_deep_research` is pinned at `408da442a661ea5e40a6163329f82e3f22628949` and is
read-only. `odr_p1_hooks.patch` is applied to a throwaway materialization under `.build/`
by bootstrap (`git archive` → apply → `uv sync` resolves ODR from there). The submodule
tree is never edited in place, and bootstrap asserts `open_deep_research.__file__` lives
under the patched materialization so an unpatched global copy can never be imported by
accident.

This file specifies exactly what the patch does. It is authored here rather than generated
blind because the patch must be validated against the **frozen** stack (the pinned
langgraph/langchain resolved by `uv sync --frozen` on the run host), not against whatever
langchain a dev machine happens to have. The P0-parity integration test
(`tests/integration/test_p0_parity.py`) is what proves it landed correctly, and it runs in
that frozen environment during bootstrap, before any GPU work.

## What the patch may change (and nothing else)

Per plan §11.2, the patch is confined to:

1. **Inject a `PageTransformStrategy` at the WEBPAGE boundary.** In `utils.tavily_search`,
   after `unique_results` is built (the URL-deduped, first-occurrence view — vendor step
   2, around line 71), branch: if a strategy is bound in the current context
   (`shapeflow_p1.odr.hooks.current_strategies()`), build the `HCheckpoint` from
   `unique_results` + the assistant turn and dispatch to `strategy.page.transform_tool_batch`;
   otherwise run vendor's `summarize_webpage` path unchanged. Both paths start from the
   **same** `result['raw_content'][:max_content_length]` bytes and the same truncation.

2. **Inject a `ResearchCloseStrategy` at the RESEARCHER_CLOSE boundary.** In
   `deep_researcher.compress_research`, **before** the in-place
   `researcher_messages.append(compress_research_simple_human_message)` at line 538,
   losslessly clone `researcher_messages` and build the `CCheckpoint`. If a strategy is
   bound, dispatch to `strategy.close.close_researcher` and return its
   `{compressed_research, raw_notes}`; otherwise run vendor compression unchanged.

3. **Capture, without altering, the whole assistant-turn batch.** In `researcher_tools`,
   record the assistant `AIMessage` and its ordered sibling tool calls for the checkpoint.
   The publish order of `tool_outputs` (line 482-489) is **not** changed.

4. **Classify and record the close reason** using `odr.close_reason.classify_close` on the
   same signals vendor uses.

5. **Propagate identity via `ContextVar`** (task/attempt/node/variant/boundary ids) — never
   a mutable module global — so concurrent researchers don't cross-contaminate.

6. **Point the model base URL at the local telemetry proxy.** Done via config/env
   (`OPENAI_BASE_URL`), so this is largely outside the patch; the patch only ensures the
   per-call config is threaded through.

## What the patch must NOT change under P0

Prompt bytes, tool schemas, join/publish order, retry counts, token limits, supervisor and
researcher control flow, result formatting, and fallback behavior. With no strategy bound,
the graph must be byte-for-byte vendor.

## The fused variant is not here

`C05-FUSED-EXT` changes the stopping policy, not just the reducer, and `ResearchComplete`
has an empty schema. It therefore cannot be a close-after hook. It is a separate P1-only
graph adapter (a new tool such as `ResearchCompleteWithSelection`) with its own arm,
checkpoint schema and P0 comparator, and it keeps the dedicated-selector fallback for the
two exit paths that never call the fused tool. It is never attributed as compressor-only.

## The parity gate

`tests/integration/test_p0_parity.py` drives both the vendor graph and the patched graph
(hooks off) with a deterministic **mock** OpenAI-compatible model and a mock search
backend, records a `RunTrace` from each (`odr.p0_parity`), and asserts
`compare_traces(vendor, patched, require_output_equality=True).ok`. A CPU fixture and a
small-model fixture are both exercised. **P0 parity failure blocks all GPU screening** —
the launch gate refuses to proceed.
