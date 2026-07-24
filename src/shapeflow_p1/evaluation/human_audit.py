"""The human audit queue and the gate that upgrades provisional verdicts (plan §13.2/§13.6).

A machine-only run can only produce PROVISIONAL_* verdicts. This module selects what a human must
review and defines when a provisional verdict may become final. The queue is not just a random
sample: it always includes every critical miss, every judge disagreement, and every near-margin
case, because those are exactly where a machine judgment is most likely wrong and most consequential.

The upgrade gate is intentionally strict: the stratified sample must reach the pre-frozen fraction,
critical atoms and source edges must have zero audited error, non-critical accuracy must clear its
bar, and no critical harm may have been missed. Until then, KEEP/CONDITIONAL/KILL_HARM stay
provisional -- the study never launders a machine judgment into a final claim.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Sequence

__all__ = ["AuditItem", "select_audit_queue", "AuditGate", "audit_gate_ready"]


@dataclass(frozen=True)
class AuditItem:
    task_id: str
    kind: str  # RANDOM_SAMPLE | CRITICAL_MISS | DISAGREEMENT | NEAR_MARGIN
    detail: str = ""


def _stable_unit(seed: int, key: str) -> float:
    h = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def select_audit_queue(
    *,
    tasks_by_stratum: dict[str, list[str]],
    sample_fraction: float,
    critical_misses: Sequence[str] = (),
    disagreements: Sequence[str] = (),
    near_margin: Sequence[str] = (),
    seed: int = 0,
) -> list[AuditItem]:
    """Build the audit queue: a deterministic stratified random sample plus every mandatory case.

    Sampling is deterministic (a stable hash per task), so re-running selects the same tasks; the
    fraction is applied within each stratum so no stratum is skipped.
    """
    items: list[AuditItem] = []
    seen: set[tuple[str, str]] = set()

    def add(task_id: str, kind: str, detail: str = "") -> None:
        key = (task_id, kind)
        if key not in seen:
            seen.add(key)
            items.append(AuditItem(task_id, kind, detail))

    # Stratified random sample.
    for stratum, tasks in sorted(tasks_by_stratum.items()):
        ranked = sorted(tasks, key=lambda t: _stable_unit(seed, t))
        n = max(1, round(sample_fraction * len(tasks))) if tasks else 0
        for t in ranked[:n]:
            add(t, "RANDOM_SAMPLE", stratum)

    # Mandatory inclusions -- always audited regardless of the sample.
    for t in critical_misses:
        add(t, "CRITICAL_MISS")
    for t in disagreements:
        add(t, "DISAGREEMENT")
    for t in near_margin:
        add(t, "NEAR_MARGIN")
    return items


@dataclass(frozen=True)
class AuditGate:
    ready: bool
    reasons: tuple[str, ...] = ()


def audit_gate_ready(
    *,
    sampled_fraction: float,
    required_fraction: float,
    critical_errors: int,
    noncritical_accuracy: float,
    noncritical_accuracy_min: float,
    critical_harm_misses: int,
) -> AuditGate:
    """Whether provisional verdicts may be upgraded to final. All conditions must hold."""
    reasons = []
    if sampled_fraction < required_fraction:
        reasons.append(f"sample {sampled_fraction:.2f} < required {required_fraction:.2f}")
    if critical_errors > 0:
        reasons.append(f"{critical_errors} critical audited error(s) (must be 0)")
    if noncritical_accuracy < noncritical_accuracy_min:
        reasons.append(
            f"non-critical accuracy {noncritical_accuracy:.3f} < {noncritical_accuracy_min}"
        )
    if critical_harm_misses > 0:
        reasons.append(f"{critical_harm_misses} missed critical harm(s) (must be 0)")
    return AuditGate(ready=not reasons, reasons=tuple(reasons))
