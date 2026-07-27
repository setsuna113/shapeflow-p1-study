# Judge calibration — outcome and what it revealed

**Decision: `JUDGE_CALIBRATION_INCONCLUSIVE` → keep the incumbent `deepseek-v4-flash @ max_tokens 8000`.**

The rule in `configs/judge_calibration.yaml` was frozen (commit `d096249`) and its span-binding
metric narrowed to what the output schema can show (`fb1c2ac`) — both **before** any calibration
data existed, verifiable from the history. It was then applied mechanically. No candidate cleared
admissibility, so the pre-registered fallback fired and the incumbent stands.

Material: 3 held-out `RESERVE` tasks (`decision.yaml` excludes RESERVE from every analysis), 2
excerpt batches each, identical prompts across candidates. 16 Exa calls to acquire the worlds;
real DeepSeek spend for the whole exercise ≈ $0.45.

## Measurements

| candidate | non-truncated | stability | binding validity | atoms/task | projected $ |
|---|---|---|---|---|---|
| **flash-none-8k** (incumbent) | 1.000 | 0.000 | **0.895** | 11.33 | 12.69 |
| flash-none-32k | 1.000 | 0.000 | 1.000 | 11.00 | 12.51 |
| pro-high-32k | 1.000 | 0.013 | **0.609** | 13.00 | 19.23 |
| pro-max-32k | 1.000 | 0.066 | 1.000 | 24.33 | 26.02 |

Rejections: all four on `repeat_decision_stability < 0.95`; `flash-none-8k` and `pro-high-32k`
additionally on `span_binding_validity < 0.98`.

## What the numbers mean, and what they do not

**Binding validity is about wasted yield, not a corrupted answer key.** `truth.py` already drops
span ids outside the offered batch (`s for s in atom["supporting_span_ids"] if s in batch_ids`),
and an atom left with no verified span never enters the packet. So the incumbent's 10.5% invalid
citations are discarded by the pipeline rather than written into truth. That is
`require_exact_span_binding` doing its job. It does mean roughly a tenth of the incumbent's
proposals are thrown away.

**`pro-high` is much worse than both `pro-max` and plain `flash`** at citing real spans (39%
invalid). Nothing predicted that ordering, which is precisely why the rule was frozen first.

**Truncation did not reproduce.** Every candidate scored 1.000, including at `max_tokens: 8000`.
The earlier `JudgeUnavailable: every one of 4 attempts hit the 8000 token output cap` happened on a
span-dense task outside this set. So this shows 8000 was *not exercised* here — **not** that it is
safe. The adaptive batch splitting added earlier remains the mitigation.

**Stability is the substantive finding, and it is a property of the pipeline, not of any judge.**
Re-asking an identical prompt with only the seed advanced produces near-disjoint atom sets under
the pipeline's own atom identity. Inspecting the stored responses shows two distinct causes:

- *Genuine content instability.* One pair returned 6 atoms vs 3 with zero overlap, differing in
  which facts they proposed at all — one run asserting "Arabs used gunpowder at the siege of Mecca
  in 690 AD", which the other never mentions.
- *Pure paraphrase.* Another pair returned the same two facts — "painted between 1509 and 1511"
  vs "completed between 1509 and 1511", with the second atom character-identical — and still
  scored zero overlap.

Exact-text Jaccard cannot separate those. **Neither can `truth.py`**, which keys atom identity on
`(facet_id, text)` and derives `_stable_atom_id` from `sha256({facet, text})`. Two paraphrases of
one fact are therefore two atoms in the answer key, which inflates the recall denominator.

This also independently confirms the earlier diagnosis of the write-once conflict on
`T114aaa1d55b0cb`: truth-building is not reproducible, so rebuilding a task yields a different
packet and the write-once guard correctly refuses it.

## Consequences accepted

- The gate cannot discriminate between judges — all four fail it — so it contributed no selection
  information. Its value here was diagnostic, not selective.
- Keeping the incumbent means the 9 packets already built under it stay valid; no
  `freeze-corpus-attempt` seal and no rebuild are needed, and truth-building resumes from 9/48.
- The richer candidate (`pro-max-32k`, 2.2× the atoms/task at perfect validity) is **not** adopted,
  because adopting it would mean overriding a rule after seeing it reject everything. That is the
  forking path the pre-registration exists to prevent. A thinner answer key costs statistical
  power; it does not bias the arm comparison, since the same judge scores every arm.
- Near-duplicate atoms and the instability above are recorded as limitations on the answer key.
  They are exactly what `authoring_method: MACHINE_CANDIDATE_PENDING_HUMAN_AUDIT` and
  `verifier_status: PENDING` exist to qualify, and the 20% human audit is what settles them.

## Recorded, not gated

Per the frozen rule: style-differential bias (P0 and P1 differ systematically in length and
citation density, and an arm-blind judge can still be differentially biased by style),
shared-model preference (the same family authors truth and judges reports), and binding *recall*,
which needs human labels that do not exist yet.

## Ledger note

`deepseek_usd` settled reads ≈ $40.57, of which ≈ $40.12 predates the pricing correction and is
denominated in the invented 2.00/8.00 snapshot that over-stated spend ~19×. That history stays on
the books rather than being rewritten; only calls after the correction settle at verified rates.
