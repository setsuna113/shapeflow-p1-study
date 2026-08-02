# P1 on BrowseComp-Plus — findings

**Status: final.** The campaign is complete: 80 tasks × 7 arms = 560 cells, of which **520
committed and 40 were censored** (section 5). Every number below is measured on the full run.

Provenance: branch `freeze1/phase0`, protocol `protocol/SHAPEFLOW_FREEZE_1.md`, benchmark pinned
by `protocol/bench_manifest.json`, retriever by `protocol/retrieval_freeze.json`. The cells were
committed under execution binding `c091c41f1a0e`, recorded in an append-only approval chain; the
analysis names that binding explicitly rather than re-running the campaign under a later one.

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

| arm | selector | H published | C published |
|---|---|---|---|
| `H_CPU_CONTROL` | CPU greedy | **206 / 206 (100 %)** | — |
| `H_PROSE_CONTROL` | LLM, prose | **129 / 141 (91 %)** | — |
| `H_MARKDOWN_ID` | LLM, span IDs | **2 / 188 (1 %)** | — |
| `H_PLUS_C` | LLM, span IDs | **4 / 181 (2 %)** | **3 / 73 (4 %)** |
| `C_CPU_CONTROL` | CPU greedy | — | **50 / 62 (81 %)** |
| `C_ID` | LLM, span IDs | — | **5 / 76 (7 %)** |

Pooled across four arms and both boundaries: the LLM span-ID selector published **14 times out of
518 opportunities (2.7 %)**. The CPU selector published **256 out of 268 (95.5 %)**.

### Why this is a selector result and not a plumbing defect

`H_MARKDOWN_ID` and `H_CPU_CONTROL` differ in exactly one pre-registered field,
`selector_backend`. Same `markdown_structure_v1` chunker, same `stable_union_v1` aggregator, same
`P1_ID` contract, same 512-token budget, same renderer, same preflight, same tasks. One publishes
nothing; the other publishes everything. That contrast — `H_LLM_VS_CPU` — was frozen in
`matched_contrasts` before any data existed, and it is the reason the answer is available at all.

### How far over budget, measured on all 472 overruns

Every rejection is the same assertion, and the whole page batch then falls back to P0 as the
contract specifies:

```
PREFLIGHT: rendered <n> tokens exceeds selected_token_budget 512
```

| arm | overruns | min | median | p90 | max | median ÷ budget | within 2× budget |
|---|---|---|---|---|---|---|---|
| `H_MARKDOWN_ID` | 186 | 525 | 1 605 | 3 642 | 9 938 | **3.1×** | 27 % |
| `H_PLUS_C` | 231 | 525 | 2 142 | 7 063 | 9 938 | **4.2×** | 20 % |
| `C_ID` | 55 | 652 | 5 728 | 8 113 | 9 622 | **11.2×** | 4 % |
| **all** | **472** | 525 | **2 095** | 7 038 | 9 938 | **4.1×** | 21 % |

**This corrects the interim report**, which put the median overshoot at about 30 % from an early
sample of 85 overruns. At full scale the median selection renders to **four times** its budget,
and at the C boundary to **eleven times**. The consequence matters for section 7: an aggregator
that trimmed to budget would not be recovering a near-miss, it would be discarding 75–91 % of
what the model asked for.

The three regimes are cleanest in what each selector renders *when it does publish*:

| selector | median rendered tokens | as a fraction of the 512 budget |
|---|---|---|
| CPU greedy (`H_CPU_CONTROL`, n = 79 cells) | **512** | packs to the limit, exactly |
| LLM prose (`H_PROSE_CONTROL`, n = 76 cells) | **226** | 44 % — comfortably under |
| LLM span IDs | — | rejected before publication |

So the model complies with a budget when the task is *write briefly*, and does not when the task
is *choose items whose combined rendered size you cannot see*. The prompt shows candidate text and
never per-candidate cost, and `stable_union_v1` deduplicates and returns without dropping
anything. A greedy loop that can price each candidate meets the same budget every time.

---

## 3. What P1 costs and what it saves

Paired within task, on the tasks where both arms committed. 10 000-resample bootstrap, seed
pinned. **Bold** where the 95 % interval excludes zero.

| arm | n | GPU-busy (primary) | prompt tok (co-primary) | completion tok (co-primary) | e2e |
|---|---|---|---|---|---|
| `H_CPU_CONTROL` | 74 | **−67.1 %** | **−66.0 %** | **−78.5 %** | **−64.9 %** |
| `H_PROSE_CONTROL` | 71 | **−57.0 %** | **−17.6 %** | **−66.9 %** | **−56.4 %** |
| `C_CPU_CONTROL` | 61 | **−31.7 %** | **−9.8 %** | **−25.7 %** | **−31.2 %** |
| `C_ID` | 74 | −13.4 % | +4.7 % | −13.8 % | −13.3 % |
| `H_PLUS_C` | 72 | +3.6 % | **+67.6 %** | −4.6 % | +3.8 % |
| `H_MARKDOWN_ID` | 73 | +6.8 % | **+59.0 %** | −0.6 % | +6.9 % |

Two rows carry the result.

**`H_CPU_CONTROL` is the largest saving in the study, and it is real.** It issues *no*
`PAGE_P0_SUMMARY` calls at all — the selector replaces every page summary rather than preceding
it — so the arm runs about 12 inference requests per cell against P0's 34. 67 of 74 tasks moved
in the same direction.

**`H_MARKDOWN_ID` is the sharpest statement of the failure.** It runs `PAGE_P1_SELECTOR_LOCAL` on
every batch (19.8 per cell), fails preflight, runs `PAGE_P0_SUMMARY` on every batch (20.8 per
cell), and publishes output byte-identical to P0. It is a **59 % prompt-token surcharge for
nothing**, and 69 of 73 tasks paid it.

`C_ID` and `H_PLUS_C` are the arms that assign a treatment which then almost never fires; their
work differences are not distinguishable from zero except for the prompt-token surcharge they pay
for the attempt.

### Two behavioural changes, weighted honestly

The precheck flagged an evidence-recall regression for `H_CPU_CONTROL`: −20.1 %, interval
[−0.054, −0.004], nominally excluding zero. **It is not being reported as established harm.**
The interval is on the mean, and the mean rests on 14 discordant pairs out of 74 — 4 tasks better,
10 worse, sign test *p* = 0.18. Across 6 arms × 9 metrics at nominal 95 % coverage, one or two
such exclusions are expected by chance alone, and no multiplicity correction is pre-registered.
It is a signal worth a follow-up, not a finding.

The one behavioural change that does survive that scrutiny is `H_PROSE_CONTROL` issuing **18.9 %
fewer search queries** ([−1.63, −0.39], 16 tasks better / 38 worse, sign *p* = 0.004). Shorter
page summaries change what the agent goes looking for next. `search_queries` is *not* a
pre-registered endpoint, so this is exploratory — but it is the more robust of the two, and the
asymmetry is worth recording: the registered endpoint gave the weaker evidence.

---

## 4. Accuracy — reported, and not established

All 520 committed cells were graded by the benchmark's official grader. None failed to grade.

| arm | graded | correct | accuracy |
|---|---|---|---|
| `H_PLUS_C` | 74 | 8 | 0.108 |
| `H_MARKDOWN_ID` | 77 | 8 | 0.104 |
| `H_PROSE_CONTROL` | 77 | 8 | 0.104 |
| `H_CPU_CONTROL` | 80 | 8 | 0.100 |
| `P0` | 74 | 7 | **0.095** |
| `C_ID` | 76 | 7 | 0.092 |
| `C_CPU_CONTROL` | 62 | 2 | 0.032 |

**The whole study contains 48 correct answers across 520 cells.** Every paired contrast against
P0 is "no detectable difference", and the discordant-pair counts show why there was never a
prospect of anything else:

| arm | n paired | paired diff | 95 % CI | tasks where the arms disagreed |
|---|---|---|---|---|
| `H_MARKDOWN_ID` | 73 | 0 (0 %) | [0, 0] | **0** |
| `H_PLUS_C` | 72 | +0.014 | [0, +0.042] | 1 |
| `H_CPU_CONTROL` | 74 | −0.014 | [−0.041, 0] | 1 |
| `C_ID` | 74 | −0.014 | [−0.041, 0] | 1 |
| `C_CPU_CONTROL` | 61 | −0.016 | [−0.049, 0] | 1 |
| `H_PROSE_CONTROL` | 71 | −0.014 | [−0.070, +0.042] | 5 |

Across all six arms, **nine tasks in total** were graded differently from P0. At a 9.5 % baseline
almost every task is wrong in every arm, so almost every pair is concordant-and-wrong and carries
no information. This is the outcome committed to in writing in `reports/gates/COMPETENCE_PILOT.md`
*before* any cell was graded, and it is reported here rather than mined.

`C_CPU_CONTROL`'s 0.032 is the one number that looks like a finding and is not: it is **2 correct
answers out of 62**, on the arm that also lost 22 % of its cells to censoring (section 5), and its
paired contrast rests on a single discordant task.

**`H_MARKDOWN_ID` at 0 discordant tasks out of 73 is a positive control, and it passed.** That arm
falls back on every batch and therefore publishes output byte-identical to P0, so a coherent
pipeline must grade it identically on every task — and it did, 73 times out of 73. The grading
path is behaving; the resolution simply is not there.

---

## 5. Censoring

**40 of 560 cells failed, and every one has the same cause**: `telemetry_complete: False` — one or
two inference attempts left unaccounted after `BrokenPipeError` in the provider writing a response
to a client that had already hung up. The cells were otherwise fine: full reports, treatment
published where assigned. The runner refuses a cell whose work accounting is incomplete, which is
correct behaviour for a study whose primary endpoint *is* the work accounting.

The cause is uniform and verifiable: `unavailable_attempt_ids` is **0.00 per cell in every
committed cell of every arm**, and **1.0–1.8 in every failed one**.

| arm | committed | failed | survival |
|---|---|---|---|
| `H_CPU_CONTROL` | 80 | 0 | **100 %** |
| `H_MARKDOWN_ID` | 77 | 3 | 96 % |
| `H_PROSE_CONTROL` | 77 | 3 | 96 % |
| `C_ID` | 76 | 4 | 95 % |
| `H_PLUS_C` | 74 | 6 | 93 % |
| `P0` | 74 | 6 | 93 % |
| `C_CPU_CONTROL` | 62 | **18** | **78 %** |
| **total** | **520** | **40** | **93 %** |

`C_CPU_CONTROL` is a genuine outlier at 22.5 %, against 0 % for the other CPU-selector arm. Two
candidate explanations were tested and **both are refuted**:

- *Publication at C causes the failure* — i.e. the CPU selector publishes, the in-flight
  `COMPRESSOR_P0` request is abandoned, and the provider hits a broken pipe. Refuted: committed
  cells publish 0.81 closes per cell, failed cells 0.78. Publication does not distinguish them.
- *It is positional* — the arm sits at an unlucky position in the block order. Refuted: within-block
  arm order is randomised across 14 distinct orders, and failure by position is flat at
  3.8 %–10.0 %.

So the arm association is real, is not explained by the treatment firing, and is **not explained
at all**. `C_CPU_CONTROL` falls below the pre-registered `MIN_PAIRED_SURVIVAL` of 0.95 and is
reported with a survivorship warning, not as a clean estimate. `H_PLUS_C`, `H_MARKDOWN_ID`,
`H_PROSE_CONTROL` and `H_CPU_CONTROL` also carry the warning, because `P0` at 93 % shrinks every
pairing: a contrast needs both arms to have committed on the same task.

Not fixed mid-run, deliberately: the client timeout lives in a hashed config, so changing it
re-mints the execution binding, renames every work key and orphans the campaign in progress. The
honest options were to let it run and report the censoring, or to stop and re-run under a new
binding. The first was taken.

---

## 6. Gates

Four gate criteria ran. Two failed. Both failures are recorded with measured-vs-threshold, a
diagnosis and a priced menu, and **no frozen parameter was changed in response to either**.

| gate | verdict | measured vs threshold | file |
|---|---|---|---|
| P1 liveness (H) | **FAIL** | 0 published spans, floor ≥ 1 | `gates/P1_LIVENESS_SMOKE.md` |
| P1 liveness (C) | PASS | 6 published spans in the smoke | same |
| retrieval competence — accuracy | **PASS** | 0.1224 vs 0.10 (n = 98) | `gates/COMPETENCE_PILOT.md` |
| retrieval competence — evidence recall | **FAIL** | 0.1488 vs 0.40 (n = 98) | same |

**The C liveness gate passed on four tasks and would not pass on eighty.** `C_ID` published 6 of 6
close reductions across the smoke's four `pilot_competence` tasks; on the campaign's 80
`b1_select` tasks it published 5 of 76. The gate is recorded as it was decided — a pre-registered
check on a pre-registered sample, not a claim revised when more data arrives — but the campaign
number is the one to believe about the mechanism, and it says the C boundary fails the same way H
does. A four-task liveness sample catches a mechanism that never fires; it cannot characterise one
that fires sometimes.

### The recall failure is a calibration mismatch, not a broken retriever

Every query returns a full top-5 with cosines in the expected 0.49–0.65 band, and the frozen index
measured Recall@100 = 0.4831 on evidence. What fails is the agent's *coverage*: its reformulations
are paraphrases of one another. One task's six queries —

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
hundred documents returned — then applies it to an agent whose frozen `top_k` is 5. No run of this
configuration could have cleared it. That accuracy passes anyway, while surfacing 15 % of the
evidence, says the agent is not failing for want of a usable retriever.

---

## 7. The verdict

**P1 as specified — LLM-driven span-ID selection under a fixed rendered-token budget — does not
work, and the study can say precisely which part fails.**

**What works.** The *boundary* is real and the saving at it is large. Replacing an LLM-written
page summary with a short, budget-respecting artefact cuts **67 % of GPU-busy seconds, 66 % of
prompt tokens and 79 % of completion tokens**, paired on 74 tasks, and cuts **32 %** at the
researcher-close boundary. Both are the study's pre-registered primary and co-primary endpoints,
both intervals exclude zero, and the direction holds on 67 of 74 and 52 of 61 tasks respectively.
The H boundary is worth roughly twice the C boundary. Nothing about the agent's measured accuracy
degraded when this was done — though see the limit on that claim below.

**What does not work.** The *selector* is the failure, not the boundary and not the plumbing.
Asked to choose spans under a 512-token rendered budget, the LLM published **14 times out of 518
opportunities**. A greedy CPU loop, on the identical chunker, aggregator, contract, budget,
renderer and preflight — differing in exactly one pre-registered field — published **256 of 268**.
The failure is specifically budget compliance: the median rejected selection renders to **4.1× its
budget**, and at the C boundary **11.2×**. The same model on the same pages keeps to the same
budget when asked for short prose (median 226 of 512 tokens). It cannot keep to it when asked to
choose items whose combined rendered cost the prompt never shows it.

**What it costs to get this wrong.** An arm that assigns span-ID selection and always falls back
pays for the selector *and* the prose it was meant to replace: `H_MARKDOWN_ID` costs **+59 %
prompt tokens on 69 of 73 tasks** and publishes output byte-identical to P0.

**So the mechanism claim is settled and the quality claim is not.** The publication result is a
property of the machine at n = 518 and will not move. The accuracy comparison has no resolution at
all — 48 correct answers in 520 cells, nine discordant tasks across six arms — and that was
committed to in writing before grading, not concluded after seeing it.

---

## 8. What is established and what is not

**Established — mechanism.** These are properties of the machine, measured at n = 518 selector
opportunities, and they will not move with more data:

- LLM-driven span-ID selection under a 512-token rendered budget does not publish on
  BrowseComp-Plus pages: **2.7 % across both boundaries** (1–2 % at H, 4–7 % at C).
- The identical mechanism with a budget-aware selector publishes **95.5 %** of the time.
- The failure is budget compliance, not capability or plumbing: the median rejected selection
  renders to **4.1× its budget**, and the same model keeps to the same budget when asked for
  short prose instead of ids.
- An arm that assigns P1 and always falls back costs strictly more than P0 for byte-identical
  output — **+59 % prompt tokens** for `H_MARKDOWN_ID`.

**Established — work.** Paired, bootstrap intervals excluding zero, on 61–74 tasks:

- Replacing the page summary with anything shorter is a large saving: **−67 % GPU-busy** for
  CPU-selected spans, **−57 %** for short prose.
- The C boundary is worth about half of that: **−32 % GPU-busy** for `C_CPU_CONTROL`.
- The saving is not a token artefact — it holds on GPU-busy seconds, on both token components
  separately, and on end-to-end latency.

**Not established, and not resolvable by this campaign:**

- **Any non-inferiority claim about accuracy.** The workload never cleared the evidence-recall
  floor its own pre-registration set, and 12.2 % baseline accuracy over 80 tasks cannot resolve
  the between-arm differences such a claim needs. This was committed to in writing *before* the
  campaign was graded.
- **Whether span-ID selection would help if the selector met its budget.** The instrument never
  delivered that treatment. Section 2's overrun distribution makes this worse than the interim
  report implied: switching to `coverage_budget_v1` would trim a median selection by 75 %, so it
  would test a substantially different treatment rather than rescue this one.
- **Why `C_CPU_CONTROL` alone lost 22 % of its cells.** Two explanations tested, both refuted.

**Reserved for a human, and not done:** changing `selected_token_budget`, the encoder, `top_k`, or
any floor. Each is a change to a frozen object made after seeing agent data. Gold injection was
never considered.

---

## 9. Declared deviations

1. **Two pre-registered control arms were added to the roster after the liveness smoke**
   (`H_CPU_CONTROL`, `C_CPU_CONTROL`). Both were already frozen in `matched_contrasts`; the first
   roster omitted them for budget. Without them, 100 % fallback stays ambiguous between "this path
   cannot emit a span on this corpus" and "this selector will not keep to a budget". A control can
   only weaken a P1 claim, never manufacture one.
2. **The campaign ran past a failed competence gate**, a departure from §10's order. The gate
   governs whether the *quality* comparison can carry a non-inferiority claim; it does not touch
   the co-primary work endpoints, which are boundary measurements independent of retrieval
   quality.
3. **Five arms are reported with a survivorship warning** rather than dropped or silently
   included (section 5).
4. **The analysis ran under a later execution binding than the campaign** (`--ran-under-binding`).
   Grading is read-only; requiring a matching binding would mean a bug in the *analysis* could
   only ever be fixed by re-running the campaign.

---

## 10. Reproducing this

```bash
shapeflow doctor                       # includes the installed-graph digest refusal
shapeflow freeze-approval --approved-commit $(git rev-parse HEAD)
# per lane: engine, provider, retrieval -- see BCPLUS_OPERATIONS.md
shapeflow run-bcplus --layer b1_select --arms bcplus_arms --shard <lane> --shards 2
shapeflow grade-bcplus --run-id <id> --layer b1_select --lanes 0,1 \
    --ran-under-binding <the binding the run committed under>
```

The paired analysis behind every number above is `reports/BCPLUS_campaign1.json` and its rendering
`reports/BCPLUS_campaign1.md` — the arm summaries, the per-boundary publication counts and the
bootstrap intervals, keyed by `content_sha256` and by the object-store digest of every cell that
fed them. Gate verdicts are in `reports/gates/`; the per-lane ledger and object store live under
`$DATA_ROOT/runner-lane{n}/` and are not in this repository.

A run other than this one writes `reports/BCPLUS_<run-id>.{json,md}`.

**One amendment to that artifact is on the record.** It was written without
`cells_committed_under_sha256`, so it named only the binding of the tree that *graded* the run —
`3e73c63194e7`, under which not one of its cells was produced — and could not say what to pass to
reproduce itself. The field was added in place and `content_sha256` recomputed with the same
canonical function `build_report` uses, after verifying the artifact still hashed to its recorded
digest (`63cbda85b9fa`, now `6979963ca2f9`). It was amended rather than regenerated because the
value is known exactly while re-deriving it would mean 520 fresh judge calls against an engine
stack that has since been shut down. No number in the file changed; the code that writes it now
records the field, so no later artifact needs this note.
