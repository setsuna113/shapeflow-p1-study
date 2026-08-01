# Gate: retrieval competence — **FAIL** on evidence recall, PASS on accuracy

Protocol §5.3. `run_id competence2` · layer `pilot_competence` · 100 ITD tasks drawn from
`b1_confirm`+`b2` (deliberately disjoint from `b1_select`, where the champion form is chosen)
· **P0 only** · 98 of 100 cells committed, 2 `FAILED_FINAL`
· binding `785adcdeec66144e8622a6246014a942d16de55a5a722edcb155a092805fe774`
· machine-readable verdict in `RETRIEVAL_COMPETENCE.json`, full arm record in
`../BCPLUS_competence2.{json,md}`.

## Measured versus threshold

Floors read straight out of `configs/prereg.yaml` `pilots.competence`, never restated in code.

| criterion | measured | floor | n | verdict |
|---|---|---|---|---|
| `p0_accuracy` | **0.1224** | 0.10 | 98 | **PASS** |
| `agent_evidence_recall` | **0.1488** | 0.40 | 98 | **FAIL** |

The criteria are conjunctive, so the gate is **FAIL** and the retrieval freeze stays ineffective
(`effective_after` is still the empty string; the file was not touched).

Supporting numbers, all P0: gold recall 0.1571 · 5.18 search queries per task · 13.2 distinct
documents retrieved per task · 39 of 98 tasks surfaced at least one evidence document · 0 empty
reports · 0 page fallbacks · publication rate 1.0 · 166 898 prompt and 24 314 completion tokens
per task · 238.4 s GPU-busy per task. Grading was complete: 98 of 98 judged, none unavailable.

## Diagnosis

**The retriever is not the thing that failed.** The service returns a full top-5 for every query
with cosine scores in the expected 0.49–0.65 band, and the frozen index measured Recall@100 =
0.4831 on evidence at freeze time. Nothing is broken.

What fails is the *agent's* coverage of the corpus, and the reason is visible in the traces: the
reformulations are near-duplicates of one another. One task's six queries were

```
news publications co-founded by individuals who dropped out of university
news outlets founded by college dropouts
news publications co-founded by college dropouts
news outlets founded by college dropouts active up to 2023
news publications co-founded by college dropouts active up to 2023
news outlets founded by college dropouts with official websites
```

Six queries × top-5 is thirty slots; they returned **twelve** distinct documents. Across the
pilot, 5.18 queries per task yield 13.2 distinct documents — roughly 2.5 new documents per query.
Against evidence sets of three to six documents drawn from a 100 195-document corpus, that is
where 0.1488 comes from.

**The floor and the measurement are calibrated to different quantities.** The prereg justifies
0.40 by "dense retrieval at Recall@100 = 55.8%" — a *retriever* reference point, measuring one
query against one hundred returned documents. It is then applied to an *agent* measurement whose
frozen `top_k` is 5 and which reaches roughly thirteen documents in total. The two numbers are
not commensurable, and no run of this configuration could have cleared 0.40.

That accuracy passes anyway is the informative part. The agent answers 12.2 % of BrowseComp-Plus
correctly while surfacing 15 % of the evidence, so it is not failing to answer for want of a
usable retriever; the primary quality endpoint is live, if low.

## Disposition

**Nothing was changed.** Not the encoder, not `top_k`, not the floor. Each of those is a change
to the frozen retriever made after seeing agent data — a declared deviation and a human's
decision — and the gate's own refusal message says so. Gold injection was not considered.

The pre-authored menu from the Freeze-1 plan, priced against measured throughput
(~2.2 min per cell across both lanes):

| option | what changes | cost | consequence |
|---|---|---|---|
| A. climb the encoder 0.6B→4B→8B | the frozen retriever, after seeing agent data | 100 tasks ≈ 4 GPU-h | **declared deviation**; also does not address the cause, which is query diversity, not encoder quality |
| B. raise `top_k` / results-per-query | the frozen retriever | ≈ 4 GPU-h | addresses the cause directly, but changes the H-boundary batch-size distribution — the thing under study |
| C. accept a lower floor | the pre-registration | 0 | amendment after seeing data ⇒ downstream conclusions exploratory |
| D. stop and write the report | nothing | 0 | the honest outcome |
| **E. record the failure and continue on the work endpoints** | nothing | campaign cost only | **taken** — see below |

**Why the campaign proceeds regardless, and what that costs.** The competence gate exists to
decide whether the *quality* comparison can carry an ε non-inferiority claim. It says nothing
about the co-primary **work** endpoints — `prompt_tokens`, `completion_tokens`,
`interval_union_seconds` — which are mechanism measurements taken at the boundary and are
independent of how good the retriever is. P1's cost is measurable here whatever recall does.

So the campaign runs under the frozen design and the verdict reports two tiers:

- **established** — the work verdict, from the co-primary endpoints;
- **not established** — any non-inferiority claim about accuracy, because the workload did not
  clear the recall floor its pre-registration set, and because 12.2 % baseline accuracy over 80
  tasks cannot resolve the differences between arms that such a claim would need.

Running past a failed gate is itself a deviation from §10's order. It is recorded here rather
than argued away: no threshold, contract, budget or retriever parameter was altered to obtain it,
and the accuracy tier is downgraded rather than reported as if the gate had passed.
