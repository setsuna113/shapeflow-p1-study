# Gate: P1 liveness smoke — **FAIL** at the H boundary, PASS at C

`run_id smoke5` · layer `pilot_competence` · 5 arms × 4 tasks × 2 lanes = 20 cells, all committed
· binding `f02f7f22a3d1529906f210002e36b6727ed15631d74671fcedef6b026611eaea`
· retriever `a289e6f65eb18633a0613843952c5f47924fce2505ef5fe71aa971a366242fc8`
· evidence: `STATUS_smoke5_lane0.json`, `STATUS_smoke5_lane1.json` beside this file.

The smoke exists to answer one question before 400 cells are spent: does assigning P1 actually
put P1 output into the graph? For two of the four treatment arms the answer is no, and the two
lanes agree cell for cell.

## Measured versus threshold

Criterion `treatment_live` (`campaign/bcplus.py::_the_treatment_was_actually_live`), and the same
condition as the pre-registered canary check `p1_published_output`
(`campaign/canary.py:355`). Threshold: **every treatment arm publishes ≥ 1 P1 span.** Structural,
not numeric — there is no fraction to tune.

| arm | boundary | published spans | page fallbacks | close failures | verdict |
|---|---|---|---|---|---|
| `C_ID` (C01) | C | 6 / 6 | 0 / 0 | 2 / 2 | **PASS** |
| `H_PROSE_CONTROL` (H00-PROSE) | H | 3 / 6 | 0 / 0 | 0 / 0 | **PASS** |
| `H_MARKDOWN_ID` (H02) | H | **0 / 0** | 5 / 6 | 0 / 0 | **FAIL** |
| `H_PLUS_C` (H02+C01) | H then C | **0 / 0** | 6 / 5 | 2 / 2 | **FAIL** |

(lane 0 / lane 1. Two cells per arm per lane.)

`world_searched` PASS on both lanes: 0/10 committed cells issued no search query, against a
`MAX_SILENT_CELL_FRACTION` of 0.20. Retrieval cache hit rate 0.48 over 50 searches. The graph is
live, the frozen world is being searched, and the installed-tree refusal did not fire — this is
not the stale-vendor failure of the previous campaign.

## Diagnosis

**Root cause, one sentence: at the H boundary the selector chooses more evidence than the
512-token rendered budget allows, `stable_union_v1` does not trim to the budget, preflight
rejects, and the whole page batch falls back to P0.**

Every failing record carries the same shape:

```json
{"contract":"P1_ID","node":"H","fell_back":true,
 "failure":{"reason":"PREFLIGHT",
            "detail":"rendered 4815 tokens exceeds selected_token_budget 512"}}
```

with siblings in the same batch recording `STRATEGY_ERROR` carrying that detail — the whole-batch
fallback rule (`odr/adapter.py`, `DeferredPageBatch`) working as specified. Observed overruns:
1174, 1178, 1180, 1716, 1722, 1730, 1744, 2109, 2114, 2333, 2362, 4815 tokens against 512.

Four measurements, over 85 preflight overruns with resolvable per-span costs, decide what kind of
failure this is:

- **The renderer is not the problem.** rendered ÷ raw selected tokens is 1.07 min, **1.34 median**,
  3.42 max. Framing overhead is a third, not a multiple.
- **The budget is reachable.** Taking candidates cheapest-first, the number that fits in 512
  rendered tokens is 5 min, **35 median**, 259 max, out of 60–170 offered. 512 is not a budget
  only an empty selection can meet.
- **The selector overshoots on raw material alone.** It emitted 12–64 ids totalling 562–3070 raw
  tokens. Even at zero rendering cost the smallest of those, 562, exceeds 512.
- **The overshoot is often modest.** Median selection ≈ 16 spans ≈ 665 rendered tokens — 30 % over.
  A 30 % overshoot destroys the entire batch, because `stable_union_v1` has no budget step.

So the arithmetic of the failure is: *the model is asked to satisfy a rendered-token budget it is
told (`prompts.py` `TOKEN BUDGET for your selected evidence: {budget}`) but is never shown the
cost of any candidate, and the aggregator behind it drops nothing.* `coverage_budget_v1`
(`p1/aggregators.py:157`) greedily includes under exactly this budget and would publish a trimmed
selection; `stable_union_v1` (`:131`) deduplicates and returns. **Both H02 and C01 use
`stable_union_v1`.**

That last fact is also why C passing is informative rather than lucky: same contract, same
aggregator, same budget, same renderer, same preflight — the only difference is the candidate
view. C sees one researcher's compressor input (median 86 spans, 84 tokens each). H sees a raw
BrowseComp-Plus page: median 94 spans but up to **1673 spans / 21 539 tokens**. Handed a
21 k-token page and told to keep 512, the selector keeps far too much.

## What this is and is not

It is **not** a plumbing defect. The world is searched, the selector decodes, ids parse against
the candidate set, aggregation runs, and the failure is a budget assertion that fires at the
designed place. `C_ID` publishing 6 spans on both lanes proves the whole P1 path end to end.

It is **not yet** the finding "span-ID selection does not work". The instrument has not delivered
the treatment at H even once, so no P0-vs-H comparison measures span selection; it measures the
cost of attempting it. `H_MARKDOWN_ID` spends 176 k prompt tokens where P0 spends 123 k — a 44 %
surcharge for output that is byte-identical to P0's.

It **is** a real, reportable property of the configuration: *under Freeze-1's frozen 512-token
budget, `P1_ID` + `stable_union_v1` at the WEBPAGE_P1 boundary never publishes on
BrowseComp-Plus.* Nothing was tuned to produce that; it is what the frozen design does.

## Disposition

No pre-registered parameter was changed. `selected_token_budget` stays 512, contracts stay,
thresholds stay, and the H arms stay in the campaign exactly as frozen — a 100 % fallback rate is
a legitimate intention-to-treat outcome and the primary endpoints stay well defined under it.

One change was made, and only to the arm roster: `H_CPU_CONTROL` (H00-CPU) and `C_CPU_CONTROL`
(C00-CPU) join `bcplus_arms`. Both were already pre-registered — `H_LLM_VS_CPU` and `C_LLM_VS_CPU`
are frozen entries in `matched_contrasts`, authored before any BrowseComp-Plus data existed — and
the first roster omitted them only for budget. Their selector is a greedy loop that charges the
exact shared renderer and stops at the same budget, so they publish or fail for structural reasons
alone. Without them, 100 % fallback stays ambiguous between "this publication path cannot emit a
span on this corpus" and "this selector will not keep to a budget it cannot measure". A control
can only weaken a P1 claim, never manufacture one.

Options that were **not** taken, and what each would have cost the pre-registration:

| option | what changes | cost | consequence |
|---|---|---|---|
| raise `selected_token_budget` above 512 | a frozen treatment parameter, after seeing it fail | 0 GPU-h, re-mint | **declared deviation**; every H conclusion becomes exploratory |
| swap H02 → `H_TYPED_COVERAGE` (coverage_budget_v1) | the treatment arm itself | ~8 GPU-h | changes what is being tested from the champion to a different variant |
| show per-candidate token costs in the selector prompt | `prompt_bundle_sha256`, i.e. the contract | 0 GPU-h, re-mint | amendment to a frozen contract after seeing data |
| **run as frozen, add the two pre-registered controls** | roster only | +2 arms ≈ +40 % cells | **taken** — no threshold, contract or budget touched |

The competence pilot is P0-only and untouched by any of this; it proceeds.
