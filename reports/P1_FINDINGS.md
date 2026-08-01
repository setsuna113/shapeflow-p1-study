# P1 on BrowseComp-Plus — findings

**Status: interim.** The main campaign is in flight (343 of 560 cells committed at the time of
writing, plus 31 censored -- see section 3a, which is the biggest caveat on everything here).
Everything below is stated at the sample size it was measured at, and the mechanism findings are
separated from the effect sizes on purpose: the first are properties of the machine and are
already settled, the second are estimates that will tighten.

Provenance: branch `freeze1/phase0`, protocol `protocol/SHAPEFLOW_FREEZE_1.md`, benchmark pinned
by `protocol/bench_manifest.json`, retriever by `protocol/retrieval_freeze.json`. Every run is
namespaced by an execution binding recorded in an append-only approval chain.

---

## 1. The question

At two boundaries in an Open Deep Research agent, evidence is compressed:

| boundary | where | unit |
|---|---|---|
| **H** (`WEBPAGE_P1`) | the sibling tool-call batch of one assistant turn | one gather batch |
| **C** (`RESEARCHER_CLOSE`) | researcher close | one close |

**P0** writes prose. **P1** replaces that with *span-ID selection*: the model is shown the
chunked page and returns ids, and a deterministic renderer materialises the chosen spans under a
frozen 512-token rendered-output budget. The study asks what P1 buys and what it costs.

Seven arms, all pre-registered in `configs/variants.yaml` before any BrowseComp-Plus data existed:

| arm | H half | C half | selector |
|---|---|---|---|
| `P0` | prose | prose | — |
| `H_MARKDOWN_ID` | **span IDs** | prose | LLM |
| `C_ID` | prose | **span IDs** | LLM |
| `H_PLUS_C` | **span IDs** | **span IDs** | LLM |
| `H_PROSE_CONTROL` | **short prose**, same 512 budget | prose | LLM |
| `H_CPU_CONTROL` | **span IDs** | prose | CPU greedy |
| `C_CPU_CONTROL` | prose | **span IDs** | CPU greedy |

The last three are controls, and they are what make the result interpretable. `H_PROSE_CONTROL`
separates *pointing at evidence* from *emitting fewer tokens*. The two CPU controls separate *the
publication path* from *the selector's behaviour*: their selector is a greedy loop charging the
exact same renderer against the exact same budget, so they publish or fail structurally.

---

## 2. The central finding

**Span-ID selection driven by an LLM almost never publishes, at either boundary. The identical
mechanism driven by a greedy CPU loop almost always does, at both.**

Publication, campaign to date, per boundary:

| arm | selector | H published | C published |
|---|---|---|---|
| `H_CPU_CONTROL` | CPU greedy | **56 / 56 (100 %)** | — |
| `H_PROSE_CONTROL` | LLM, prose | **37 / 38 (97 %)** | — |
| `H_MARKDOWN_ID` | LLM, span IDs | **2 / 51 (4 %)** | — |
| `H_PLUS_C` | LLM, span IDs | **3 / 50 (6 %)** | **1 / 21 (5 %)** |
| `C_CPU_CONTROL` | CPU greedy | — | **15 / 17 (88 %)** |
| `C_ID` | LLM, span IDs | — | **1 / 21 (5 %)** |

Every failure is the same assertion:

```
PREFLIGHT: rendered 4815 tokens exceeds selected_token_budget 512
```

and the whole page batch falls back to P0, as the contract specifies.

### Why this is a selector result and not a plumbing defect

`H_MARKDOWN_ID` and `H_CPU_CONTROL` differ in exactly one pre-registered field,
`selector_backend`. Same `markdown_structure_v1` chunker, same `stable_union_v1` aggregator, same
`P1_ID` contract, same 512-token budget, same renderer, same preflight, same tasks. One publishes
nothing; the other publishes everything. That contrast — `H_LLM_VS_CPU` — was frozen in
`matched_contrasts` before any data existed, and it is the reason the answer is available at all.

Four measurements from 85 preflight overruns say the budget is not the villain:

- **rendered ÷ raw selected tokens: 1.07 min, 1.34 median, 3.42 max.** Renderer framing is a
  third, not a multiple.
- **Cheapest-first, 5 / 35 / 259 candidates fit in 512** (min / median / max), out of 60–170
  offered. The budget is reachable.
- **The selector emitted 12–64 ids totalling 562–3070 raw tokens.** Even at zero rendering cost
  the smallest of those exceeds 512.
- **The median overshoot is only ~30 %** — and destroys the whole batch, because
  `stable_union_v1` deduplicates and returns while `coverage_budget_v1` is the aggregator that
  trims. Both `H02` and `C01` use `stable_union_v1`.

So the mechanism is: *the model is told a rendered-token budget it cannot measure — the prompt
shows candidate text, never per-candidate cost — and the aggregator behind it drops nothing.* A
greedy loop that can price each candidate meets the same budget every time.

---

## 3. What P1 costs and what it saves

Paired on the same task, campaign to date. **n = 17–21 tasks; magnitudes provisional until the
full run gives bootstrap intervals.** The signs are mechanically forced, and the reason is given
for each.

| arm | GPU-busy | prompt tok | completion tok | energy | why the sign is not luck |
|---|---|---|---|---|---|
| `H_CPU_CONTROL` | **−66 %** | **−63 %** | **−79 %** | −64 % | replaces the page summary outright: no `PAGE_P0_SUMMARY` at all |
| `H_PROSE_CONTROL` | **−61 %** | −21 % | **−73 %** | −59 % | same, and the shorter pages shrink the researcher's context downstream |
| `C_CPU_CONTROL` | −32 % | −15 % | −24 % | −32 % | replaces the compressor (`COMPRESSOR_P0` drops to 0.1 per cell) |
| `C_ID` | −15 % | **+3 %** | −20 % | −15 % | publishes almost nothing; see caveat below |
| `H_PLUS_C` | +5 % | **+59 %** | −3 % | +6 % | pays for the selector, then pays for P0 anyway |
| `H_MARKDOWN_ID` | **+24 %** | **+70 %** | +12 % | +25 % | same, at every page: a selector call *and* a summary per page |

The `H_MARKDOWN_ID` row is the sharpest statement of the result. It runs
`PAGE_P1_SELECTOR_LOCAL` on every batch, fails preflight, runs `PAGE_P0_SUMMARY` on every batch,
and publishes output **byte-identical to P0**. It is a 70 % prompt-token surcharge for nothing.

**Caveat on `C_ID`.** Its prompt tokens are now +3 %, which is what an arm that pays for a
selector and publishes nothing should look like. Its GPU-busy and completion tokens are still
*down* (−15 %, −20 %), which an inert arm should not be. The likely mechanism is that assigning a
selector at the close changes when the researcher closes — a mediated effect of the assignment
rather than of the published output — but coupled-seed end-to-end ITT lets trajectories diverge
after the first intervention and the between-task variance is large. Not interpreted until the
full run.

---

## 3a. Censoring — the largest threat to the numbers above

At 374 terminal cells, **31 have failed and every one has the same cause**:
`telemetry_complete: False`, one or two inference attempts left unaccounted after
`BrokenPipeError` in the provider writing a response to a client that had already hung up. The
cells were otherwise fine — full reports, treatment published where it was assigned,
`fell_back: false`. The runner refuses a cell whose work accounting is incomplete, which is the
correct behaviour for a study whose primary endpoint *is* the work accounting.

| arm | committed | failed | survival |
|---|---|---|---|
| `H_CPU_CONTROL` | 54 | 0 | 100 % |
| `H_MARKDOWN_ID` | 51 | 2 | 96 % |
| `H_PROSE_CONTROL` | 52 | 2 | 96 % |
| `C_ID` | 50 | 3 | 94 % |
| `H_PLUS_C` | 51 | 3 | 94 % |
| `P0` | 48 | 5 | 91 % |
| `C_CPU_CONTROL` | 37 | **16** | **70 %** |
| **total** | **343** | **31** | **92 %** |

Two things about it are uncomfortable and are stated rather than smoothed.

**It is accelerating.** By quarter of the run: 5.4 %, 4.3 %, 10.8 %, 12.8 %. It is not the engine
degrading — median cell latency is flat across those quarters (139 s, 118 s, 140 s, 115 s) and
p90 is noisy without a trend, disk is at 30 %, memory has 76 GB free, and there is exactly one
engine process per GPU. The cause is understood at the level of *what* fails and not yet *why it
worsens*.

**It is not proportional across arms.** `C_CPU_CONTROL` loses 30 % of its cells while
`H_CPU_CONTROL` loses none, and the heaviest arm — `H_MARKDOWN_ID`, at +70 % prompt tokens —
loses only 2. So this is not simply "long cells die", which would at least bias every arm the
same way. `C_CPU_CONTROL`'s contrast falls well below the pre-registered
`MIN_PAIRED_SURVIVAL` of 0.95 and will be reported with the survivorship warning, not as a clean
estimate. `P0` at 91 % matters more than it looks: the baseline losing cells shrinks *every*
pairing, since a contrast needs both arms to have committed on the same task.

Not fixed mid-run, and that is a deliberate choice rather than an oversight: the client timeout
lives in a hashed config, so changing it re-mints the execution binding, which renames every work
key and orphans the campaign in progress. The honest options were to let it run and report the
censoring, or to stop and re-run 374 cells under a new binding. The first is taken.

---

## 4. Gates

Three gates ran. Two failed. Both failures are recorded with measured-vs-threshold, a diagnosis
and a priced menu, and **no frozen parameter was changed in response to either**.

| gate | verdict | measured vs threshold | file |
|---|---|---|---|
| P1 liveness (H) | **FAIL** | 0 published spans, floor ≥ 1 | `gates/P1_LIVENESS_SMOKE.md` |
| P1 liveness (C) | PASS | 6 published spans in the smoke | same |

**The C gate passed on four tasks and would not pass on eighty.** `C_ID` published 6 of 6 close
reductions across the smoke's four `pilot_competence` tasks; on the campaign's `b1_select` tasks
it publishes 1 of 21. The gate is recorded as it was decided — it is a pre-registered check on a
pre-registered sample, not a claim that gets revised when more data arrives — but the campaign
number is the one to believe about the mechanism, and it says the C boundary fails the same way H
does. A four-task liveness sample is enough to catch a mechanism that never fires and not enough
to characterise one that fires sometimes.
| retrieval competence — accuracy | **PASS** | 0.1224 vs 0.10 (n=98) | `gates/COMPETENCE_PILOT.md` |
| retrieval competence — evidence recall | **FAIL** | 0.1488 vs 0.40 (n=98) | same |

### The recall failure is a calibration mismatch, not a broken retriever

Every query returns a full top-5 with cosines in the expected 0.49–0.65 band, and the frozen
index measured Recall@100 = 0.4831 on evidence. What fails is the agent's *coverage*: its
reformulations are paraphrases of one another. One task's six queries —

```
news publications co-founded by individuals who dropped out of university
news outlets founded by college dropouts
news publications co-founded by college dropouts
news outlets founded by college dropouts active up to 2023
news publications co-founded by college dropouts active up to 2023
news outlets founded by college dropouts with official websites
```

— are thirty top-5 slots returning **twelve distinct documents**. Across the pilot, 5.18 queries
reach 13.2 distinct documents against evidence sets of three to six in a 100 195-document corpus.

The prereg justifies the 0.40 floor by "dense retrieval at Recall@100 = 55.8 %" — one query, a
hundred documents returned — then applies it to an agent whose frozen `top_k` is 5. No run of
this configuration could have cleared it. That accuracy passes anyway (12.2 % while surfacing
15 % of the evidence) says the agent is not failing for want of a usable retriever.

---

## 5. Declared deviations

Three, all recorded rather than argued away.

1. **Two pre-registered control arms were added to the roster after the liveness smoke**
   (`H_CPU_CONTROL`, `C_CPU_CONTROL`). Both were already frozen in `matched_contrasts`; the first
   roster omitted them for budget. Without them, 100 % fallback stays ambiguous between "this
   path cannot emit a span on this corpus" and "this selector will not keep to a budget". A
   control can only weaken a P1 claim, never manufacture one.
2. **The campaign ran past a failed competence gate**, a departure from §10's order. The gate
   governs whether the *quality* comparison can carry a non-inferiority claim; it does not touch
   the co-primary work endpoints, which are boundary measurements independent of retrieval
   quality.
3. **`C_CPU_CONTROL` will be reported with a survivorship warning** rather than dropped or
   silently included.

Not done, and reserved for a human: changing `selected_token_budget`, the encoder, `top_k`, or
any floor. Each is a change to a frozen object made after seeing agent data. Gold injection was
never considered.

---

## 6. What is established and what is not

**Established** (mechanism properties, robust to sample size):

- LLM-driven span-ID selection under a 512-token rendered budget does not publish on
  BrowseComp-Plus pages: 4 % at H, 5 % at C.
- The same mechanism with a budget-aware selector publishes: 100 % at H, 88 % at C.
- The failure is a budget-compliance failure, not a capability or plumbing failure.
- An arm that assigns P1 and always falls back costs strictly more than P0 for identical output.

**Established** (work endpoints, direction certain, magnitude provisional at n ≈ 10):

- Replacing the page summary with *anything shorter* is a large saving — around −61 % GPU-busy
  for short prose, −66 % for CPU-selected spans.
- The saving compounds downstream: shorter page outputs shrink the researcher's context.

**Not established**, and will not be by this campaign:

- Any non-inferiority claim about **accuracy**. The workload did not clear the recall floor its
  own pre-registration set, and 12.2 % baseline accuracy over 80 tasks cannot resolve the
  between-arm differences such a claim needs.
- Whether span-ID selection *would* help if the selector could meet its budget. The instrument
  never delivered that treatment. `coverage_budget_v1` and per-candidate costing in the prompt
  are the two obvious routes, and both are design changes requiring a human.

---

## 7. Reproducing this

```bash
shapeflow doctor                       # includes the installed-graph digest refusal
shapeflow freeze-approval --approved-commit $(git rev-parse HEAD)
# per lane: engine, provider, retrieval -- see BCPLUS_OPERATIONS.md
shapeflow run-bcplus --layer b1_select --arms bcplus_arms --shard <lane> --shards 2
shapeflow grade-bcplus --run-id <id> --layer b1_select --lanes 0,1 \
    --ran-under-binding <the binding the run committed under>
```

Artifacts: `reports/BCPLUS_<run-id>.{json,md}` for the paired analysis,
`reports/gates/` for gate verdicts, and the per-lane ledger and object store under
`$DATA_ROOT/runner-lane{n}/`.
