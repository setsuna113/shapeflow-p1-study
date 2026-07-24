"""The campaign phase state machine (plan §14.1).

The campaign advances through explicit phases, and only along legal edges, so a run can never
skip a gate -- you cannot reach SCREEN_RUNNING without P0_PARITY_PASSED, and you cannot open the
holdout without POLICY_FROZEN. The machine also encodes the two ways a campaign can legitimately
diverge: the mini-ITT path toward a holdout, and the CHARACTERIZE_NO_GO path when screening shows
every design failing. The terminal states (BLOCKED, COMPLETE_NO_GO, BUDGET_EXHAUSTED,
ABORTED_SAFELY) are dead ends -- once there, the only artifact produced is a report.

Keeping this as a small, pure FSM means the legal-edge set is auditable in one place, and the
coordinator just asks it "may I go here?" rather than scattering phase logic across the loop.
"""

from __future__ import annotations

import enum
from typing import Optional

__all__ = ["Phase", "TERMINAL_PHASES", "can_transition", "next_phases", "PhaseMachine",
           "IllegalPhaseTransition"]


class Phase(enum.Enum):
    NEW = "NEW"
    DOCTOR_PASSED = "DOCTOR_PASSED"
    ACQUISITION_COMPLETE = "ACQUISITION_COMPLETE"
    SNAPSHOTS_FROZEN = "SNAPSHOTS_FROZEN"
    P0_PARITY_PASSED = "P0_PARITY_PASSED"
    GPU_SMOKE_PASSED = "GPU_SMOKE_PASSED"
    SCREEN_RUNNING = "SCREEN_RUNNING"
    SCREEN_COMPLETE = "SCREEN_COMPLETE"
    MINI_ITT_RUNNING = "MINI_ITT_RUNNING"
    MINI_ITT_COMPLETE = "MINI_ITT_COMPLETE"
    POLICY_FROZEN = "POLICY_FROZEN"
    HOLDOUT_RUNNING = "HOLDOUT_RUNNING"
    HOLDOUT_COMPLETE = "HOLDOUT_COMPLETE"
    LIVE_VALIDITY_OPTIONAL = "LIVE_VALIDITY_OPTIONAL"
    CHARACTERIZE_NO_GO_RUNNING = "CHARACTERIZE_NO_GO_RUNNING"
    CHARACTERIZE_NO_GO_COMPLETE = "CHARACTERIZE_NO_GO_COMPLETE"
    REPORT_COMPLETE = "REPORT_COMPLETE"
    # terminals
    BLOCKED = "BLOCKED"
    COMPLETE_NO_GO = "COMPLETE_NO_GO"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    ABORTED_SAFELY = "ABORTED_SAFELY"


TERMINAL_PHASES = frozenset({
    Phase.REPORT_COMPLETE, Phase.BLOCKED, Phase.COMPLETE_NO_GO,
    Phase.BUDGET_EXHAUSTED, Phase.ABORTED_SAFELY,
})

# Failure edges reachable from any non-terminal phase.
_ANYTIME_FAILURES = frozenset({Phase.BLOCKED, Phase.BUDGET_EXHAUSTED, Phase.ABORTED_SAFELY})

_LEGAL: dict[Phase, frozenset[Phase]] = {
    Phase.NEW: frozenset({Phase.DOCTOR_PASSED}),
    Phase.DOCTOR_PASSED: frozenset({Phase.ACQUISITION_COMPLETE}),
    Phase.ACQUISITION_COMPLETE: frozenset({Phase.SNAPSHOTS_FROZEN}),
    Phase.SNAPSHOTS_FROZEN: frozenset({Phase.P0_PARITY_PASSED}),
    Phase.P0_PARITY_PASSED: frozenset({Phase.GPU_SMOKE_PASSED}),
    Phase.GPU_SMOKE_PASSED: frozenset({Phase.SCREEN_RUNNING}),
    Phase.SCREEN_RUNNING: frozenset({Phase.SCREEN_COMPLETE}),
    # After screening, either advance toward the holdout or characterize a no-go.
    Phase.SCREEN_COMPLETE: frozenset({Phase.MINI_ITT_RUNNING, Phase.CHARACTERIZE_NO_GO_RUNNING}),
    Phase.MINI_ITT_RUNNING: frozenset({Phase.MINI_ITT_COMPLETE}),
    # Mini-ITT may still find no promotable policy and fall to the no-go characterization.
    Phase.MINI_ITT_COMPLETE: frozenset({Phase.POLICY_FROZEN, Phase.CHARACTERIZE_NO_GO_RUNNING}),
    Phase.POLICY_FROZEN: frozenset({Phase.HOLDOUT_RUNNING}),
    Phase.HOLDOUT_RUNNING: frozenset({Phase.HOLDOUT_COMPLETE}),
    Phase.HOLDOUT_COMPLETE: frozenset({Phase.LIVE_VALIDITY_OPTIONAL, Phase.REPORT_COMPLETE}),
    Phase.LIVE_VALIDITY_OPTIONAL: frozenset({Phase.REPORT_COMPLETE}),
    Phase.CHARACTERIZE_NO_GO_RUNNING: frozenset({Phase.CHARACTERIZE_NO_GO_COMPLETE}),
    Phase.CHARACTERIZE_NO_GO_COMPLETE: frozenset({Phase.REPORT_COMPLETE}),
    Phase.REPORT_COMPLETE: frozenset({Phase.COMPLETE_NO_GO}),  # report may conclude a no-go
}


class IllegalPhaseTransition(RuntimeError):
    pass


def next_phases(phase: Phase) -> frozenset[Phase]:
    if phase in TERMINAL_PHASES and phase is not Phase.REPORT_COMPLETE:
        return frozenset()
    forward = _LEGAL.get(phase, frozenset())
    if phase in TERMINAL_PHASES:
        return forward
    return forward | _ANYTIME_FAILURES


def can_transition(frm: Phase, to: Phase) -> bool:
    return to in next_phases(frm)


class PhaseMachine:
    """Tracks the current phase and records the ordered history of transitions."""

    def __init__(self, phase: Phase = Phase.NEW) -> None:
        self._phase = phase
        self._history: list[Phase] = [phase]

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def history(self) -> list[Phase]:
        return list(self._history)

    @property
    def is_terminal(self) -> bool:
        return self._phase in TERMINAL_PHASES and self._phase is not Phase.REPORT_COMPLETE

    def transition(self, to: Phase) -> None:
        if not can_transition(self._phase, to):
            raise IllegalPhaseTransition(f"{self._phase.value} -> {to.value} is not a legal edge")
        self._phase = to
        self._history.append(to)
