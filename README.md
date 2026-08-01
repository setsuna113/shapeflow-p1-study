# shapeflow

**ShapeFlow Freeze-1**: a broker that decides, at the vLLM scheduler tick, *which compression
form* each job uses and *which jobs are admitted* this round — jointly, not in sequence.

Protocol of record: [`protocol/SHAPEFLOW_FREEZE_1.md`](protocol/SHAPEFLOW_FREEZE_1.md). It is
frozen; changes append as numbered amendments.

## The setting

An Open Deep Research-style agent on vLLM. A supervisor dispatches researchers; each searches in
a ReAct loop. Evidence compression happens at two boundaries:

| Boundary | Where | Decision unit |
|---|---|---|
| **H** | the sibling tool-call batch of one assistant turn, joined by `asyncio.gather` | one gather batch |
| **C** | researcher close (`ResearchComplete` / max tool calls / no tool call) | one close |

Two forms compete at each boundary:

- **P0 (prose)** — one summarization request per page at H; a long free decode at C. Decode is
  serial per token and is the GPU bottleneck.
- **P1 (span-ID selection)** — pages are snapshotted by content hash and cut into located
  fragments; the model emits only fragment IDs, and CPU validates, merges, stably orders and
  renders them. At H this is one whole-batch selector call published atomically; at C it is a
  separate selector request after close.

P1 shortens the free decode. It does not remove the language problem: the selector still
prefills the candidate evidence, and relevance, contradiction and gap judgements remain
semantic.

## The contribution

The broker, not span selection. Each pending job arrives as an **offer** — a set of alternative
execution plans (P0: N summary requests; P1: one selector request plus a CPU render) — and the
broker chooses form *and* admission together, per tick, under a 2 ms solve budget. Quality is a
hard qualification gate rather than a tradable weight: a job whose P1 plan does not qualify
keeps P0. Anything that times out or loses its valuation degrades fail-closed.

## What "answer" means here

Five claims, in a chain where each link gates the next: safe heterogeneity **exists** → it is
**predictable** → gating beats any static policy → joint beats sequential → it holds up under
real serving. A link that fails stops the one after it, and the claims shrink to what survived
rather than being restated more softly.

## Design commitments

- **Frozen retrieval.** Treatment runs never reach the live web; every arm of a task retrieves
  the same corpus with the same ranking, so arms cannot be compared across two different worlds.
- **Pre-registration with provenance.** Every frozen threshold records the procedure that
  produced it and the digest of the artifact it read. Pilot data is P0-only, and the fill
  asserts it.
- **Gates as code.** Every gate reads its thresholds from the frozen pre-registration by dotted
  path, is three-valued (`PASS` / `FAIL` / `INPUTS_UNAVAILABLE`), and records the digest of every
  artifact behind its verdict.
- **All-offered accounting.** Fallbacks, retries and timeouts keep their cost. A P1 that fell
  back to P0 is not free.

## Layout

```
protocol/   the frozen plan, the stack manifest, the frozen pre-registration (tracked)
configs/    prereg template, campaign, retrieval, variants, budget, stack, gates
vendor/     pinned ODR submodule, read-only
patches/    the only sanctioned modification to vendor, plus its tree digest
schemas/    JSON Schemas; every artifact validates against one
src/        the implementation
data/ runs/ object_store/ logs/   gitignored experiment state
reports/    generated gate reports and verdicts
```

See [`AGENTS.md`](AGENTS.md) before changing anything — it lists the invariants whose violation
produces a result that looks fine and is wrong.

The Week-1 kill/keep study this repo grew out of is sealed at the `week1-archive` tag.
