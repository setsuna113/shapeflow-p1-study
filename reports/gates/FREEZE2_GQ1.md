# Freeze-2 GQ1 — local quality, CPU arms

Decided 2026-08-03. Record: `reports/gates/FREEZE2_GQ1.json`. Thresholds pre-registered in
`configs/prereg2.yaml`, written and committed **before** any of these numbers was computed.

3,264 trials over 1,632 gather batches × 2 arms, scored against 830 judged queries. Zero GPU.

---

## The numbers

| | CPU-FULL | CPU-PROMPTVIEW |
|---|---:|---:|
| Batches | 1,632 | 1,632 |
| Scored / none-offered / unjudged | 482 / 1,150 / 0 | 482 / 1,150 / 0 |
| **Source coverage** | **0.261** | 0.262 |
| **Evidence-doc retention** | **0.162** | 0.165 |
| Hard-negative interference | 0.292 | 0.293 |
| **Negative displacement** | **0.469** | 0.471 |
| Answer-string retention | 0.011 | 0.012 |

Mechanically, from the same trials: publication 99.88% / 100%, **zero** over-budget
publications, rendered tokens p50 506 and p90 512 — it packs to the frozen limit exactly, and
`prompt_pack_v1` held one-span-per-page on 1,630/1,630 batches.

## What this says

**The whole-batch pack collapses onto a fraction of the batch.** Source coverage 0.261 over a
mean 9.07-page batch is ~2.4 sources published. This was the primary risk named in the plan for
a 512-token whole-batch budget, and it is realised.

**Most of the evidence the reducer was handed does not survive.** Retention 0.162 means 84% of
the relevant documents *present in that very batch* are dropped. The denominator is per batch,
so this is not the retriever's 0.1488 recall reappearing — it is a second, independent loss
stacked on top of it, at the boundary under study.

**Negative displacement 0.469 is the sharp result.** In 47% of scored batches the selector
published a hard negative *while dropping a relevant document offered in the same batch*. Both
documents were in front of it at the same moment, so this needs no cross-arm normalisation and
cannot be explained by one arm having seen a different world. A lexical ranker at this
compression ratio is not merely losing evidence; it is preferring known-misleading material to
known-relevant material roughly half the time.

**Pruning to fit the LLM's context window costs nothing measurable.** Every metric moves by less
than 0.003 between the two arms, and 95.5% of batches published an identical span set. Combined
with the per-page floor holding everywhere, `prompt_pack_v1` is not the thing degrading quality.

## What this does *not* say

**This is not a verdict on P1.** It is a verdict on one selector — a deterministic lexical
ranker — at one budget. `LLM-PROMPTVIEW` is the arm that tests whether a semantic ranker
prefers evidence to hard negatives, and it has not run: every GPU on the host is held by
processes outside this container.

**There is no measured P0 reference for these endpoints.** P0 emits one summary per page with
non-empty content, so by construction it represents every offered source: coverage 1.0,
retention 1.0. That is an analytic statement, not a measurement, and it is the correct baseline
to read the table against — P1 at 512 keeps roughly a sixth of what P0 keeps, using roughly a
seventh of the tokens (3,790 → 512 decode tokens per batch).

**Form and amount are not separated, by design.** Operator decision D1 froze the budget at 512
for the whole batch, accepting a 7.4× compression against P0 and accepting that a quality drop
could not then be attributed to span-ID form rather than to compression. This table is that
drop. It must not be read as "pointer selection is lossy"; it is "this pointer selector at 7.4×
compression is lossy", and the two are different claims.

**Answer-string retention (0.011) is uninterpretable without a P0 reference.** The gold answer
is a short fact that a page summary need not restate verbatim, so a low rate here may be normal
for the workload rather than a property of the treatment. Reported for completeness, not used.

**Only 482 of 1,632 batches are scorable.** The other 1,150 were offered no relevant document at
all and are reported as `NONE_OFFERED` rather than scored 0.0 — scoring them would assert the
reducer dropped something it was never given. That 70% figure is itself a statement about the
retriever, not about the reducer.

## Status against the pre-registered gates

GQ0 (mechanical legality) **passes** for both CPU arms: zero budget, ID, lineage, atomicity or
partial-publication violations, publication rate 0.9988 / 1.0000 against a 0.95 floor.

GQ1 has no absolute quality floor pre-registered — the champion rule is comparative, and its
2-percentage-point indifference band is applied *between arms*, not against a constant. On the
two arms that exist, `CPU-PROMPTVIEW` and `CPU-FULL` are inside that band on every endpoint, so
the tie-break falls to all-offered work, where `CPU-PROMPTVIEW` is the cheaper shape because it
is the one an LLM arm can also run.

**No champion may be frozen yet.** `configs/prereg2.yaml` freezes exactly one selector and
forbids a second entering Phase 2; choosing between CPU and LLM before the LLM arm has run would
be selecting on the arms that happened to be runnable.
