# Retrieval freeze — decision record

**Date** 2026-07-28 · **Split** dev-530 only (the sealed 300 was never touched) ·
**Freeze digest** `6c248f9dc25f7777fb980f4dbc0cb1302ebdd7658133a5882f2af6a3913fe51a`

**Outcome: Qwen3-Embedding-8B, bfloat16, index `qwen3-embedding-8b`, top-k 5, full documents.**
Written **not effective** — it takes force only when the P0 competence pilot records its digest.

---

## Measurements

All three shipped index sizes, over the 530 dev queries, against the benchmark's own qrels. The
serving dtype is bfloat16 (see §3), so those rows are the operative ones; the float32 rows are
kept because the recall arm was decided before the dtype question arose and they show the answer
does not depend on it.

| size | dtype | dim | gold R@5 | **gold R@100** | gold R@1000 | evid R@100 | encode p50 | encode p95 |
|---|---|---|---|---|---|---|---|
| 0.6b | fp32 | 1024 | 0.0799 | 0.3153 | 0.6769 | 0.2692 | 307 ms | 368 ms |
| 4b | fp32 | 2560 | 0.1435 | 0.5155 | 0.8009 | 0.4225 | 1282 ms | 1720 ms |
| 4b | **bf16** | 2560 | 0.1434 | 0.5135 | 0.7993 | 0.4215 | **394 ms** | **759 ms** |
| 8b | fp32 | 4096 | 0.1864 | 0.5635 | 0.8423 | 0.4816 | 2422 ms | 3534 ms |
| **8b** | **bf16** | 4096 | 0.1855 | **0.5654** | 0.8423 | 0.4831 | **~600 ms** | **1049 ms** |

**The pipeline is validated independently at both ends.** Our dev-530 numbers land on the
benchmark's published all-830 figures: 0.6b measured 0.3153 against a published 0.3023, and 8b
measured 0.5635 (fp32) against a published 0.558. Nothing about the encoder, index, qrels or
recall computation is doing something private.

**BM25 remains on the record as the completed-and-failed candidate**: gold Recall@100 = 0.061,
matching the paper exactly, and verified as a genuine property of the benchmark rather than a
pipeline defect (all 558 sampled gold docids resolve; gold documents self-retrieve at rank 1 from
a non-boilerplate span). Dense retrieval is worth roughly **9x** BM25 here.

## Applying the selection rule

Rule, written before any number existed: *the smallest encoder whose dev gold Recall@100 is at
least 0.90 x the best, subject to a p95 encode-latency ceiling of 100 ms.*

- Recall bar = 0.90 x 0.5654 = **0.5089**. 0.6b (0.3153) fails; 4b (0.5135) passes; 8b passes.
- Latency arm at 100 ms: **all three fail**, 0.6b included. The ceiling was set from an estimate
  that queries would be ~50 tokens. BrowseComp-Plus queries are long multi-constraint prompts,
  and the estimate was wrong by an order of magnitude.

Operator pre-authorisation (2026-07-28): *if 4B passes both arms choose 4B; otherwise choose 8B,
relax the latency arm to the measured value and record it.* 4B does not pass both arms, so **8B**,
with the latency arm relaxed as directed.

**Relaxed latency arm: p95 = 1049 ms**, the measured value in the serving dtype. Recording the
float32 figure (3534 ms) would have frozen a ceiling roughly 3x above what the service actually
does -- conservative, but false, and later indistinguishable from a deliberate choice.

## Why bfloat16, and what it cost to establish

8B in float32 needs 32 GB per encoder instance; at one instance per lane that is 128 GB against
125 GB of host RAM. **8B is viable only in bfloat16** (16 GB per instance, 64 GB across four
lanes). The dtype is therefore not a tuning preference but a feasibility constraint, and it is
recorded in the freeze because it changes the vectors.

Conformance was re-run in the serving dtype, because a check passed in one dtype says nothing
about a service running another:

| dtype | min cosine over 40 docs | rank-1 self-retrieval | 40-doc encode |
|---|---|---|---|
| float32 | 0.999982 | 40/40 | 1050 s |
| **bfloat16** | **0.999781** | **40/40** | **326 s** |

Both clear the 0.999 floor. bfloat16 is additionally **3.2x faster** on this host, which has AMX
bf16 -- which is why the served p95 is a third of the float32 measurement.

The first bfloat16 attempt failed, and the cause was ours: `normalize` ran *in* bfloat16, whose
8 mantissa bits cannot land on unit norm (the result measured 1.0031), and the index's search
guard refused a vector it could not interpret as a cosine. Normalising in float32 after the cast
fixes it at the source instead of loosening the guard. The error could not have changed a ranking
-- scaling a query vector preserves inner-product order -- but it would have made every recorded
score subtly wrong, which is exactly the class of fault the guard exists for.

## What is frozen

Encoder repo and revision `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`, dtype, pooling, normalise,
both prefix digests (query and passage differ, and conflating them is the failure the conformance
check exists to catch), the four index shard digests, the seven corpus shard digests, top-k, and
the full-document rule.

**top-k = 5** is the benchmark's own search-agent default, not a value chosen here.

**Full documents rather than snippets** is the declared workload deviation: the H boundary exists
to compress long pages, and a snippet workload would delete the phenomenon under study. Results
are never placed beside official leaderboard numbers.

## Standing caveat

`effective_after` is empty, so the freeze is **not in force**. `require_effective()` raises until
the competence pilot records its digest. Selecting a retriever and validating one are separate
acts, and only the second licenses a campaign.

Recall at k=5 -- what a single search actually shows the agent -- is **0.1855** for 8B. Coverage
depends on the agent reformulating across many searches, and whether that clears the 40%
agent-level gold evidence recall floor is precisely what the competence pilot measures. It is not
predictable from these numbers.
