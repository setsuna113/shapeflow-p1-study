"""The machine verdict, and the multiple-comparison discipline behind it.

The verdict function encodes plan §15.3 as an explicit decision tree over pre-computed
evidence, so "why this verdict" is always inspectable. Two principles are baked in:

- Quality and work are **co-primary**: KEEP requires the quality NI guards to pass AND a work
  saving, in BOTH the isolated and operational layers. An isolated-only pass is MECHANISM_ONLY
  (a deployment NO-GO), never KEEP.
- ``NOT_ESTABLISHED`` is distinct from ``KILL_NO_HEADROOM``: the former means we proved nothing
  either way; only the latter claims "quality is fine but there is no work to save". Neither may
  be written up as "P1 proven ineffective".

The guard combiners implement the two corrections the plan requires: intersection-union for the
co-primary quality family (all must pass -- so it needs no multiplicity correction), and
Holm-FWER for the secondary factorial family.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

__all__ = ["Verdict", "NodeEvidence", "decide", "intersection_union_pass", "holm_reject"]


class Verdict(enum.Enum):
    KEEP = "KEEP"
    THESIS_GRADE = "THESIS_GRADE"
    CONDITIONAL = "CONDITIONAL"
    MECHANISM_ONLY = "MECHANISM_ONLY"
    KILL_STRUCTURAL = "KILL_STRUCTURAL"
    KILL_HARM = "KILL_HARM"
    KILL_NO_HEADROOM = "KILL_NO_HEADROOM"
    NOT_ESTABLISHED = "NOT_ESTABLISHED"


@dataclass(frozen=True)
class NodeEvidence:
    structural_pass: bool
    deterministic_harm: bool
    quality_guards_pass: bool          # intersection-union of isolated NI guards
    isolated_saving_lcb: float         # one-sided LCB95 of work saving
    saving_ucb: float                  # UCB95 of work saving (for no-headroom)
    operational_available: bool        # operational block data exists and is powered
    operational_saving_lcb: float
    operational_quality_pass: bool
    e2e_speedup_lcb: float = 0.0
    # eligibility-conditional deployment (subset passes FULL keep on holdout)
    conditional_rule_keeps: bool = False
    coverage_lcb: float = 0.0
    # thresholds (from the frozen decision config)
    min_saving: float = 0.10
    thesis_speedup: float = 1.5
    coverage_min: float = 0.30


def decide(ev: NodeEvidence) -> Verdict:
    # Deterministic structural failure and harm dominate everything.
    if not ev.structural_pass:
        return Verdict.KILL_STRUCTURAL
    if ev.deterministic_harm:
        return Verdict.KILL_HARM

    isolated_ok = ev.quality_guards_pass and ev.isolated_saving_lcb >= ev.min_saving
    operational_ok = (
        ev.operational_available
        and ev.operational_saving_lcb >= ev.min_saving
        and ev.operational_quality_pass
    )

    if isolated_ok and operational_ok:
        if ev.e2e_speedup_lcb >= ev.thesis_speedup:
            return Verdict.THESIS_GRADE
        return Verdict.KEEP

    # Full KEEP failed. A restricted, pre-frozen eligibility envelope that fully KEEPs on the
    # untouched holdout, with adequate coverage, is CONDITIONAL.
    if ev.conditional_rule_keeps and ev.coverage_lcb >= ev.coverage_min:
        return Verdict.CONDITIONAL

    # The mechanism exists in isolation but not under real batching.
    if isolated_ok and not operational_ok:
        return Verdict.MECHANISM_ONLY

    # Quality is feasible but the work-saving upper bound is below the minimum meaningful effect.
    if ev.quality_guards_pass and ev.saving_ucb < ev.min_saving:
        return Verdict.KILL_NO_HEADROOM

    # Neither a benefit nor an absence-of-headroom was established.
    return Verdict.NOT_ESTABLISHED


def intersection_union_pass(guard_lcbs: dict[str, float], margins: dict[str, float]) -> bool:
    """Co-primary quality family: every one-sided guard must clear its margin. Because a PASS
    requires ALL to pass, the family needs no multiplicity correction (intersection-union)."""
    if set(guard_lcbs) != set(margins):
        raise ValueError("guard/margin keys must match exactly")
    return all(guard_lcbs[k] >= margins[k] for k in margins)


def holm_reject(pvalues: dict[str, float], *, alpha: float = 0.05) -> dict[str, bool]:
    """Holm-Bonferroni step-down for the secondary factorial family. Returns which hypotheses
    are rejected at FWER ``alpha``. Once a hypothesis fails to reject, all larger p-values are
    retained too."""
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    reject: dict[str, bool] = {}
    still_rejecting = True
    for i, (name, p) in enumerate(ordered):
        threshold = alpha / (m - i)
        if still_rejecting and p <= threshold:
            reject[name] = True
        else:
            still_rejecting = False
            reject[name] = False
    return reject
