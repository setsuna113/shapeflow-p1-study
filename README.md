# shapeflow-p1-study

Week-1 **kill/keep study** for two ShapeFlow P1 nodes. It does not implement a broker,
a scheduler, or any part of the ShapeFlow proposal — it decides whether the P1 idea
survives at two specific points in Open Deep Research.

## The two nodes

| Node | Hook | Question |
|---|---|---|
| `WEBPAGE_P1` | Tavily `raw_content` → vendor `summarize_webpage` | Does short evidence-ID selection beat prose summarization? |
| `RESEARCHER_CLOSE` | researcher exit → vendor `compress_research` | Does evidence-ID selection beat long prose compression? |

`RESEARCHER_CLOSE` splits into two treatments that are **never merged**:
`C_VISIBLE` (selector sees exactly what P0's compressor saw — the compressor-only
experiment) and `C_REGISTRY` (selector may also re-read raw spans — a registry-assisted
extension, reported separately).

Primary design is a 2×2: `P0`, `H`, `C_VISIBLE`, `H+C_VISIBLE`.

## What "answer" means here

Each node gets one of: `KEEP`, `CONDITIONAL`, `MECHANISM_ONLY`, `KILL_STRUCTURAL`,
`KILL_HARM`, `KILL_NO_HEADROOM`, `NOT_ESTABLISHED` — plus effect size, eligibility
envelope, coverage, and failure boundary. `MECHANISM_ONLY` means the effect exists in
the isolated causal setting but not under real batching, and is a proposal NO-GO.
`NOT_ESTABLISHED` is a NO-GO for the proposal but is *not* evidence of no effect.

## Design commitments

- **Frozen web.** Tavily is called only during acquisition. Treatment runs query a
  task-local frozen source pool, so arms cannot get different worlds from ranking drift.
- **Two measurement layers.** An isolated causal layer (`max_num_seqs=1`, one upstream
  request in flight) for quality and mechanism; an operational layer (real batching,
  fixed arrival traces, `TraceBlock` as the unit) for deployment claims. A `KEEP`
  requires both.
- **Honest holdout.** Opened once, after an outcome-free tracked freeze, behind a
  UID/ACL gate — not a convention.
- **All-offered accounting.** Fallbacks, retries and timeouts keep their cost. A P1 that
  fell back to P0 is not free.

## Status

Under construction. `reports/BLOCKED*.md` is authoritative when present — the launch
gate is fail-closed and refuses to start on a missing secret, stack mismatch, unfrozen
data, P0 parity failure, or smoke failure.

## Layout

```
protocol/   hash-locked design locks and launch approval (tracked, outcome-free)
vendor/     pinned ODR submodule, read-only
patches/    the only sanctioned modification to vendor
configs/    stack, acquisition, variants, decision thresholds
schemas/    JSON Schemas; every artifact validates against one
src/        the study implementation
data/ runs/ object_store/ logs/   gitignored experiment state
reports/    generated verdicts and audits
```

See `AGENTS.md` before changing anything — it lists the invariants whose violation
produces a result that looks fine and is wrong.
