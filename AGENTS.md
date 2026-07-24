# AGENTS.md — invariants for anyone (human or model) touching this repo

This is a **kill/keep study** for two ShapeFlow P1 nodes. Its output is a verdict that
must survive an adversarial reader. Most rules below exist because violating them
produces a *result that looks fine and is wrong*. Treat them as load-bearing.

Protocol of record: `SHAPEFLOW_P1_WEEK1_CODING_PLAN_v0.1_2026-07-24.md` (parent dir).

---

## 1. Naming

- The page node is `WEBPAGE_P1` / `PAGE_RAW_CONTENT`. **Never** `HTML_P1` in code,
  schemas, figures or verdicts. Tavily `raw_content` is cleaned markdown/text, not
  browser HTML bytes. A real `DOM_HTML_P1` is a *secondary* study with its own fetch.
- The close node is `RESEARCHER_CLOSE`.
- `C_VISIBLE` and `C_REGISTRY` are different treatments and may never be merged or
  reported under one label. `C_VISIBLE` is the compressor-only experiment.

## 2. The fairness invariants (most likely to be broken silently)

- **Same bytes in.** P0 and P1 must both start from `result['raw_content'][:max_content_length]`
  using the pinned ODR truncation. If a variant reads the full page while P0 reads a
  prefix, it is `FULL_PAGE_P1` and is barred from the primary causal comparison.
- **Same visible world.** Primary selectors read `VendorVisibleOccurrenceView` — the
  URL-deduped, first-occurrence-ordered view that vendor `tavily_search` actually builds
  (`utils.py` step 2). The full `AuditOccurrenceGraph` is evaluator-only; handing its
  rank/diversity metadata to P1 is a leak.
- **No oracle in the treatment path.** `AcquisitionSpec`, `TruthPacket`,
  `VisibleTruthProjection`, gold facets, critical items and contradiction pairs are
  **evaluator-only**, different UID, different directory, different ID namespace. They
  must never reach a selector, aggregator, or preflight. Preflight checks *structure*
  (IDs in namespace, offsets, lineage closure, budgets) — never truth coverage.
- **`C_VISIBLE` sees only what P0's compressor saw.** Its span namespace is
  `VisibleMessageSpan` over the exact `researcher_messages` bytes. Reverse-mapping a
  model-written page summary back to raw page text is smuggling; that is `C_REGISTRY`.
- **P0 is not gold.** Every arm is scored against the same frozen `TruthPacket`.

## 3. Vendor and forking

- `vendor/open_deep_research` is pinned at `408da442a661ea5e40a6163329f82e3f22628949`
  and is **read-only**. Changes go through `patches/odr_p1_hooks.patch` applied to a
  materialization under `.build/`. `uv sync` resolves ODR from that patched path, and
  bootstrap asserts `open_deep_research.__file__` lives under it.
- `compress_research` **mutates `researcher_messages` in place**
  (`deep_researcher.py:538` appends a `HumanMessage`). Checkpoints must be losslessly
  cloned *before* the reducer runs or the first fork poisons every later fork.
- Publish unit is the **whole assistant turn's sibling tool-call batch**, in pinned join
  order. Do not fork or publish a single page or a single search call early.
- All three close paths (`ResearchComplete`, `max_react_tool_calls`, no-tool exit) enter
  the same treatment policy. `ResearchComplete` has an **empty schema**, so the fused
  variant (`C05-FUSED-EXT`) cannot be a close-after hook — it needs its own P1-only tool
  and its own arm, and it is never attributed as compressor-only.

## 4. Secrets

- Never persist a real key: not in code, configs, `.env`, argv, logs, SQLite, exception
  text, fixtures, or reports. `SecretRedactor` runs **before** every logging handler and
  exception renderer. Tests inject fake secrets and assert absence everywhere.
- Log via a field allowlist. Never dump a full request/header/environment then filter.
- No `set -x`, no HTTP debug tracing, no credentials in query strings, never disable TLS
  verification, bind local services to `127.0.0.1` only.
- Web page content is untrusted data. Selectors get no shell, no tools, no network.

## 5. Budgets and accounting

- Budget checks are **admission control**: reserve worst-case in a SQLite transaction
  *before* dispatch. Never call first and check after.
- A timeout after send is `FAILED_UNKNOWN` and keeps its worst-case reservation. It is
  not "no call happened".
- Every failure, retry, fallback and possible double-charge enters the cost ledger and
  the all-offered work total. A P1 that fell back to P0 still cost what it spent.

## 6. Statistics

- Thresholds in `configs/decision.yaml` are protocol v0.1 and are hash-locked by
  `protocol/launch_approval.json`. They are **not** tunable defaults. Changing any of
  them mints a new protocol SHA and invalidates the existing approval.
- Quality guards are co-primary via intersection-union — all must pass; a good average
  never compensates a failed guard.
- Holdout opens **once**, after an outcome-free tracked freeze. Repeated seeds are
  within-cluster replicates and never increase independent n.
- Recovery unit is the whole pre-registered pair/block. Never splice a pre-crash P0 with
  a post-crash P1 into one paired observation.
- `NOT_ESTABLISHED` is NO-GO for the proposal but must never be written up as
  "P1 proven ineffective".

## 7. Fail closed

Missing secret, stack mismatch, unfrozen data, P0 parity failure, schema not closed,
smoke failure, or holdout read-before-release ⇒ stop and write `reports/BLOCKED*.md`.
Never silently substitute a different model, dataset, or default parameter to keep going.
