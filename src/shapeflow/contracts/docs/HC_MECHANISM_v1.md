# H/C mechanism contract — v1

Freeze-1 §5.1(a). Pins the mechanism as implemented. A change here is an amendment, not a
refactor, because both boundaries define the decision units every result is counted in.

---

## H — the page boundary

**Where.** Inside `researcher_tools`, *after* `asyncio.gather` returns the sibling tool results
and *before* the `ToolMessage` list is built. That position is what makes the whole batch
observable at once; any earlier and the siblings are not all present, any later and the outputs
are already committed.

**Unit.** One **gather batch** = the sibling tool-call batch of a single assistant turn, in pinned
join order. This is the decision unit. Page-level numbers exist only as sub-metrics inside it.

**P0.** One summarization request per page with non-empty content.

**P1.** **One whole-batch selector request** covering every page in the batch, published
atomically.

**Atomic publication.** The batch publishes as one form or the other, never both. If any part of
the P1 path fails — an out-of-set label, a preflight rejection, a timeout, a budget refusal — the
**entire batch** falls back to P0. Half a batch is never observable.

**Why not per page.** Per-page choice would make the batch a mixture, and a mixture belongs to
neither arm. It would also let a favourable page and an unfavourable one be assigned differently
by the same trajectory, which is selection on the outcome.

## C — the close boundary

**Where.** At the top of `compress_research`, injected *before* the vendor appends its compression
instruction to `researcher_messages`.

**Why before.** The vendor mutates `researcher_messages` **in place**. A checkpoint captured after
that mutation carries the instruction, so the first fork would poison every later fork from the
same close. Checkpoints are cloned losslessly before the reducer runs.

**Unit.** One researcher close.

**P0.** The vendor compressor prefills the full history and free-decodes a long prose note.

**P1.** A **separate selector request issued after close** — the "scheme A" shape. Not a
close-after hook on the vendor's own call.

**All three exits are one policy.** `ResearchComplete`, `max_react_tool_calls` and the no-tool
exit enter the same treatment policy. Treating them differently would make the close arm's
composition depend on how a researcher happened to terminate, which correlates with difficulty.

## What both boundaries share

- **Same bytes in.** P0 and P1 start from the identical frozen input view, under the same
  token-aware shared truncation and overflow policy. A form that reads more than the other read
  is measuring a different system.
- **Fallback is not free.** A P1 attempt that fell back to P0 keeps every token it spent. Both
  costs enter the all-offered work total.
- **Publication is structural, not advisory.** Preflight checks structure — identifiers within
  namespace, offsets reconstructing exactly, lineage closure, budgets — and never consults truth.
  It returns a fail-closed result rather than raising, so a rejection is a recorded outcome rather
  than an exception path.
