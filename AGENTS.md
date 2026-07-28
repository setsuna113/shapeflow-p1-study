# AGENTS.md — invariants for anyone (human or model) touching this repo

This repo implements **ShapeFlow Freeze-1**: a broker that decides, at the vLLM scheduler tick,
which *form* each compression job uses (P0 prose vs P1 span-ID selection) and which jobs are
admitted this round. Its output is a systems result that must survive an adversarial reader.
Most rules below exist because violating them produces a *result that looks fine and is wrong*.
Treat them as load-bearing.

Protocol of record: `protocol/SHAPEFLOW_FREEZE_1.md`. It is frozen; changes append as numbered
amendments at its end (§12), and an amendment made after the relevant data was seen downgrades
the affected conclusions to exploratory.

---

## 1. Naming

- The page node is `WEBPAGE_P1` / `PAGE_RAW_CONTENT`. **Never** `HTML_P1` in code, schemas,
  figures or verdicts. Retrieval returns cleaned page text, not browser HTML bytes, so the node
  keeps its name. A real `DOM_HTML_P1` would be a separate study with its own fetch.
- The close node is `RESEARCHER_CLOSE`.
- The two boundaries are **H** (a gather batch: the sibling tool-call batch of one assistant
  turn) and **C** (one researcher close). They are the decision units. Page-level numbers are
  sub-metrics inside an H decision and are never the unit of analysis.
- The retrieval provider name and the field-mapping version are inside every snapshot id, so
  two corpora can never be confused for the same query.

## 2. The fairness invariants (most likely to be broken silently)

- **Same bytes in.** P0 and P1 both start from the same frozen input view under the same
  token-aware shared truncation and overflow policy. A variant that reads more than P0 read is
  measuring a different system.
- **Same visible world.** Every arm of a task retrieves from the same corpus with the same
  ranking. Query *text* may differ between arms — P1 changes what a researcher asks, and that
  divergence is part of the effect — but if the corpus or ranking also moved, the arms would be
  compared across two different worlds.
- **No oracle in the treatment path.** Gold answers, qrels, evidence and negative document sets
  are **evaluator-only**: different identity, different directory, different ID namespace. They
  must never reach a selector, aggregator, preflight or broker feature. Preflight checks
  *structure* (ids in namespace, offsets, lineage closure, budgets) — never truth coverage.
- **P0 is not gold.** Every arm is scored against the same frozen ground truth.
- **Nothing unobservable may be used.** Only features listed in the observability contract are
  admissible in a cost model or predictor. If the broker could not have known it at decision
  time, it may not be in the decision.

## 3. Vendor and forking

- `vendor/open_deep_research` is pinned at `408da442a661ea5e40a6163329f82e3f22628949` and is
  **read-only**. Changes go through `patches/odr_p1_hooks.patch`, applied to a materialization
  under `.build/`. The patch injects our imports into vendor bytes, so it is part of the package
  boundary: rename a module and the patch, `patches/patched_tree.sha256` and the installed copy
  all move together, and `test-p0-parity` is what proves the graph did not change.
- `compress_research` **mutates `researcher_messages` in place**, so checkpoints must be cloned
  *before* the reducer runs or the first fork poisons every later fork.
- Publish unit is the **whole assistant turn's sibling tool-call batch**, in pinned join order.
  Never fork or publish a single page early. A P1 failure takes the *entire* batch back to P0
  rather than leaving a `[P1(A), P0(B)]` hybrid.
- All three close paths (`ResearchComplete`, `max_react_tool_calls`, no-tool exit) enter the
  same treatment policy.

## 4. The offer contract

- Six-step semantics, in `contracts/docs/OFFER_INTERFACE_v1.md`: submit alternative plans
  without starting any form → versioned snapshot → atomic choice of admitted jobs *and* their
  forms → materialize only the chosen form → P0's sub-requests enter the native queue and are
  scheduled **independently** (gang scheduling is forbidden) → a stale snapshot retries the
  whole item, never materializing partially, and exhausting retries fails closed to P0-only.
- Atomicity covers form choice, admission and materialization. It does **not** mean sub-requests
  execute together.
- One contract text serves the simulation, the conformance tests and the real extension. Each
  host module asserts the contract digest at import, so a host that drifts cannot be imported.

## 5. Secrets

- Never persist a real key: not in code, configs, `.env`, argv, logs, SQLite, exception text,
  fixtures or reports. `SecretRedactor` runs **before** every logging handler and exception
  renderer. Tests inject fake secrets and assert absence everywhere.
- Log via a field allowlist. Never dump a full request/header/environment then filter.
- No `set -x`, no HTTP debug tracing, no credentials in query strings, never disable TLS
  verification, bind local services to `127.0.0.1` only.
- Web page content is untrusted data. Selectors get no shell, no tools, no network.

## 6. Budgets, accounting and measurement

- Budget checks are **admission control**: reserve worst-case in a SQLite transaction *before*
  dispatch. Never call first and check after.
- A timeout after send is `FAILED_UNKNOWN` and keeps its worst-case reservation. It is not "no
  call happened".
- Every failure, retry, fallback and possible double-charge enters the cost ledger and the
  all-offered work total. A P1 that fell back to P0 still cost what it spent.
- Work is demand vectors plus trace-level interval-union busy time. **Summing request latency
  and calling it GPU work is banned** and CI checks for it.
- W's components are always reported in full, so a reader can refit. W is additive only in the
  regime its weights were fitted in; a saving measured in one regime is never reported as
  absolute in another.

## 7. Pre-registration and statistics

- `configs/prereg.yaml` is the template; `protocol/prereg.lock.json` is the frozen document,
  written once at Phase 0 exit with per-slot provenance. Both are hash-locked by the external
  append-only approval store named by `SHAPEFLOW_APPROVAL_FILE`. They are **not** tunable
  defaults; changing one mints a new execution binding and invalidates the approval.
- A slot's **procedure** is pre-registered; its **value** is data. Fill procedures read
  P0-only artifacts and assert it, which is what makes using pilot data for the freeze
  legitimate rather than circular.
- A threshold the plan does not state and no P0-only data can derive must be decided by a
  person. Never invent one: afterwards it is indistinguishable from a pre-registered value.
- Quality guards are co-primary via intersection-union — all must pass; a good average never
  compensates a failed guard. Quality is always reported as mean difference **and** incident
  rate.
- Sealed splits open **once**. FV-B is the only post-iteration retest, and each gate allows at
  most one repair iteration.
- Recovery unit is the whole pre-registered pair/block. Never splice a pre-crash P0 with a
  post-crash P1 into one paired observation.

## 8. Gates

- Every gate is a declarative list of criteria bound to frozen prereg fields by dotted path.
  A literal threshold in a gate file is a second copy of a pre-registered number and CI rejects
  it for anything but a structural constant.
- Gate status is three-valued. `INPUTS_UNAVAILABLE` is not a pass and not a failure; a producer
  that claims completion while emitting nothing is a **failure**, and so is an artifact that
  does not parse.
- Options on failure are authored in the gate file *before* the run. A menu written after
  seeing the failure is a post-hoc design decision.
- A gate that could be satisfied by asserting it had been satisfied is not a gate: every
  criterion records the digest of the artifact it read, so a result is re-derivable rather than
  believed. **A SKIP is not a pass.**

## 9. Fail closed

Missing secret, stack mismatch, unfrozen data, P0 parity failure, schema not closed, smoke
failure, or a sealed split read before release ⇒ stop and write `reports/BLOCKED*.md`. Never
silently substitute a different model, dataset, retriever or default parameter to keep going.

A failing gate stops the program and waits for a person. Automatic iteration is forbidden: it
spends the single post-iteration retest allowance, and only a person may spend it.
