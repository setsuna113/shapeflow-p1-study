# Freeze-2 Phase 0 — census

Decided 2026-08-02. Record: `reports/gates/FREEZE2_CENSUS.json`. Zero GPU, zero task budget:
every input is an artifact the completed Freeze-1 campaign already wrote.

Three questions, none of which could be estimated: can the stored gather batches be replayed,
does a whole-batch prompt fit the engine, and does the selector schema's 64-id cap bind.

---

## 1. The offline corpus is real

| | |
|---|---:|
| H checkpoints walked | 1,808 |
| Batches fully reconstructed | **1,632 (90.3%)** |
| Pages verified | 14,367 |
| Pages with no docid | 2,032 |
| **Pages whose rebuilt bytes hashed wrong** | **0** |
| **Docids the corpus did not hold** | **0** |

A page is `VERIFIED` only when `sha256(apply_shared_budget(CorpusStore[docid].text))` equals the
`raw_content_id` the checkpoint recorded. **Not one page failed that check.** Every failure is
`NO_DOCID` — the occurrence appears in no persisted retrieval trace — and those are concentrated
in the 249 smoke-run checkpoints under `runner/`, which ran before cell outputs were being kept.
Excluding them the marginal rate is ~99–100%: each 100 batches after the first 200 added ~100
reconstructions.

So the plan's largest risk (R1) is retired. The selector shootout replays 1,632 real gather
batches at one request each, instead of re-running the agent at roughly three times the GPU cost
of everything else in the programme combined.

Provenance: corpus 100,195 documents over 7 shards (`7c07f9e23b1c`, `e92d8202e0f6`, …),
tokenizer `aeb13307a71a`, shared content budget 50,000 chars / 23,552 tokens, 7,990 occurrences
indexed from 17,505 trace rows.

## 2. Batch geometry

Every one of the 1,808 batches has exactly **one sibling tool call**, so "whole batch" is "all
pages of the one search call". Content pages per batch:

| min | p50 | mean | p90 | p95 | p99 | max |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 8 | 9.07 | 15 | 19 | 23 | 33 |

## 3. The decisive finding: the contract's unit does not fit the engine

Whole-batch selector prompt, in tokens, against a ceiling of **32,000**
(`max_model_len` 32,768 − `selector_max_completion_tokens` 768):

| min | p50 | mean | p90 | p95 | p99 | max |
|---:|---:|---:|---:|---:|---:|---:|
| 4,835 | **37,094** | 46,367 | 89,198 | 107,143 | 147,680 | 261,004 |

**1,011 of 1,632 batches (61.9%) cannot carry a whole-batch LLM selector prompt at all.** The
median batch overflows the window by 16%; the p95 batch by 235%.

This is not a tuning problem and not a defect. It follows from three frozen commitments that
cannot all hold at once on a 32k engine:

- `HC_MECHANISM_v1` requires **one selector request covering every page in the batch**;
- it also requires **the same bytes in** — P1 may not read less of a page than P0 did;
- `SharedContentBudget.derive` sizes the page budget so that **one** page plus its prompt fits
  the window (50,000 chars / 23,552 tokens per page).

P0 satisfies all three because it issues one request per page. A whole-batch P1 must carry ~9
pages of full document text in one prompt, and nine pages sized to fit one window do not fit one
window. The per-page implementation Freeze-1 shipped was, accidentally, the only form of the H
treatment this stack can execute for most batches.

**Why the pre-run estimate was wrong.** The planning estimate was ~18.6% overflow, from
`per-page prompt p50 (2,977) × pages`. That understates badly: page sizes are heavy-tailed, so a
batch total is driven by the *mean* page, not the median one. Measured, a whole-batch view runs
~4,100 tokens per page against the 2,977 median. This is precisely why the census was run
instead of the estimate being trusted.

**The CPU selector family is unaffected.** It has no context window. Whole-batch packing is
feasible for CPU packers on 100% of batches and for LLM selectors on 38%. That asymmetry is a
first-class finding rather than an inconvenience.

## 4. The 64-id cap

Candidates offered per whole-batch view:

| min | p50 | mean | p90 | p95 | p99 | max |
|---:|---:|---:|---:|---:|---:|---:|
| 87 | 913 | 1,185 | 2,278 | 2,848 | 5,422 | 14,168 |

Every view (1,632/1,632) offers more than 64 candidates, so `selected_ids`' `maxItems: 64` is
always a binding constraint on what a selector may *name*. It is **not** the limiting factor on
what gets published: a 512-token budget admits roughly 12–14 spans, well inside 64. The cap
matters only as a failure mode for a selector that emits a long list, since it is a post-hoc
validator and the tokens are charged before it runs.

## 5. What this does not yet measure

The ineligible stratum is the *large* batches by construction, so it holds disproportionately
more pages than its 62% share of batches suggests. The share of offered pages — and of offered
evidence — sitting in the ineligible stratum is not computed here and is required before any
`ρ_mech` figure is quoted. It is a zero-cost addition to the next census pass.

## 6. Status

The pre-authored rule in the plan was: window-eligible below 60% ⇒ fall back to a frozen uniform
prefilter. Eligible is **38.1%**, so that branch fires. But the rule was authored against an
estimated ~18% overflow, where a prefilter trims a tail. At 62% a prefilter is not a fallback —
it would be a new, untested selection stage doing most of the selection work on exactly the
batches that matter, and confounding every arm. Applying it automatically would be an auto-pivot
on a design decision, so this stops here for a human.

**Proceeding regardless, because it is valid under every option:** the CPU whole-batch shootout
over all 1,632 reconstructed batches. CPU packers have no window limit, so their mechanical
legality and local quality are measurable now at zero GPU cost.
