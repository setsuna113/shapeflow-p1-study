# Offer interface contract — v1

Freeze-1 §5.1(b). This document is the contract. The simulation, the conformance tests and the
real vLLM extension all reference *this text*, and each host module asserts its digest at import,
so a host that drifts from it cannot be loaded.

Changing this file changes `contracts_sha` and therefore invalidates the approval. That is the
intent: a broker operating under different rules is a different experiment.

---

## What an offer is

A unit of pending work arrives as an **offer**: a set of *alternative execution plans* for the
same job, none of them started.

| Plan | Content | Boundary |
|---|---|---|
| **P0-plan** | N summarization request descriptors (one per page in the batch) | H |
| **P0-plan** | one compression request descriptor over the whole history | C |
| **P1-plan** | one selector request descriptor, plus a CPU render | H and C |

A plan is a **description of what would be sent**, never something sent. The distinction is the
whole point of the interface: the broker chooses between futures, not between things already in
flight.

The unit of an H offer is the **whole sibling tool-call batch of one assistant turn**, in pinned
join order. Never a single page.

## The six steps

**1. Submit.** Upstream submits the alternative-plan descriptors. **No form is started.** After
this step, zero engine requests exist for the item.

**2. Snapshot.** At a scheduler tick, engine state (queue depth, KV occupancy, prefix-cache
state) is captured as a **versioned snapshot**. The version identifies both the engine epoch and
the tick, so a snapshot from a restarted engine can never be mistaken for a current one.

**3. Decide.** The broker atomically selects, for this tick, which jobs are admitted **and** which
form each admitted job uses. One decision covers both questions. A decision that answered them in
sequence would be a different algorithm, and distinguishing the two is claim C2b.

**4. Materialize.** Only the chosen form is materialized. The rejected plan leaves no trace: no
request, no reservation, no ledger entry beyond the decision itself.

**5. Dispatch.** When P0 is chosen, its N sub-requests enter the **native** vLLM queue and are
scheduled independently by native continuous batching. **Gang scheduling is forbidden.** The
driver must not block one sub-request on a sibling, must not set any co-scheduling flag, and must
not require them to start together. Forcing them to run as a unit would change the throughput
being measured into an artifact of our own harness.

**6. Retry or fail closed.** If the snapshot version is stale at materialization time, the
**whole item** is retried and **nothing is partially materialized**. Once retries are exhausted,
the item degrades **fail-closed to P0-only**, and the degradation is recorded in the ITT ledger
with its reason.

## What atomicity does and does not mean

Atomicity covers **form choice + admission + materialization**. Either an item is admitted with
exactly one form and fully materialized, or nothing about it is observable.

Atomicity does **not** mean the sub-requests execute together. Step 5 says the opposite: once
materialized, P0's N sub-requests are ordinary independent members of the native queue. Conflating
these two is the most likely way to implement this contract wrongly, because "atomic" suggests
co-execution and here it means only "no partial commitment".

## Whole-batch form purity

Within one H item, every published sub-output carries the same form. A `[P1(A), P0(B)]` hybrid is
not merely discouraged — it must be unconstructible. If any part of a P1 plan fails, the **entire**
batch republishes as P0.

A hybrid batch would make the H boundary's unit ambiguous: the decision unit is the batch, and a
batch that was half one form and half the other belongs to neither arm.

## The 2 ms budget

`decide` returns within its solve budget (2 ms by default). On overrun the decision is **discarded**
and the tick fails closed; a late decision is not applied, because it was computed against a
snapshot the engine has already moved past.

The 2 ms is the **solver** budget. The total critical path — snapshot, features, predict, solve,
atomic commit — carries its own separate budget and is measured separately.

## Observability

`decide` may read only features named in the observability contract. Enforced twice: the feature
map raises on an unregistered key at runtime, and a static check rejects any broker, cost-model or
predictor module that names a feature outside the registry. A quantity the broker could not have
known at decision time may not enter the decision.

## Determinism

The same `(offers, snapshot, seed)` yields a byte-identical decision. Without this, a disagreement
between the simulation and the live system could not be attributed to anything.

---

## Conformance obligations

A host implementation is conformant only if all of the following hold. Each is an assertion in
`tests/conformance/test_offer_contract.py`, and the suite runs against every host.

| # | Obligation |
|---|---|
| **C1** | No form mixing within an item: every published sub-output carries one form. |
| **C2** | Any P1 sub-failure republishes the whole batch as P0. |
| **C3** | No form is started before a decision: zero engine requests for an item until `materialize`. |
| **C4** | Materialization is all-or-nothing over {form choice, admission, materialization}. |
| **C5** | No gang scheduling: a P0 item's N sub-requests are independently schedulable, and are not forced to start together even by a host that would happily serialize them. |
| **C6** | A stale snapshot retries the whole item with nothing partial observable, and the item reappears as pending. |
| **C7** | Exhausted retries degrade fail-closed to P0-only, recorded with a reason. |
| **C8** | `decide` respects the solve budget at p99, and an overrun is discarded rather than applied. |
| **C9** | `decide` reads only registered observable features. |
| **C10** | The same `(offers, snapshot, seed)` produces a byte-identical decision. |

A conformance suite that no implementation can fail proves nothing. The suite is therefore also
run against a **deliberately broken** broker, and at least one obligation must fail — otherwise
the suite is vacuous and is itself failing.
