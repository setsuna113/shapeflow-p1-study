"""The Week-1 decision, rendered to Markdown and JSON from one typed object.

Both outputs derive from the same :class:`DecisionObject`, so the human-readable verdict and the
machine verdict cannot disagree -- a property the plan (§20.3) requires and a test enforces. The
renderer also enforces the provisional-verdict discipline: while the run is machine-only
(``verdict_status == PROVISIONAL_MACHINE``), the quality-dependent verdicts are shown as
``PROVISIONAL_*`` and may not be read as final. Work-only verdicts (KILL_STRUCTURAL,
KILL_NO_HEADROOM) are exempt because they do not rest on the un-audited quality judgments.

The Markdown follows the fixed §20.3 section order and never reduces to "P1 was faster on
average": every node reports whether it helped, how much, when, over how many tasks, at what
quality cost, the worst case, and whether the conclusion has been human-audited.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..canonical import canonical_str
from .decision import Verdict

__all__ = [
    "EffectWithCI",
    "NodeDecision",
    "DecisionObject",
    "render_json",
    "render_markdown",
    "PROVISIONAL_STATUS",
    "AUDITED_STATUS",
]

PROVISIONAL_STATUS = "PROVISIONAL_MACHINE"
AUDITED_STATUS = "HUMAN_AUDIT_COMPLETE"

# Verdicts that depend on the (un-audited) quality judgments and must carry PROVISIONAL_ until audit.
_QUALITY_DEPENDENT = {
    Verdict.KEEP, Verdict.THESIS_GRADE, Verdict.CONDITIONAL,
    Verdict.MECHANISM_ONLY, Verdict.KILL_HARM,
}


@dataclass(frozen=True)
class EffectWithCI:
    point: float
    lower: float
    upper: float
    unit: str = ""

    def obj(self) -> dict:
        return {"point": self.point, "lower": self.lower, "upper": self.upper, "unit": self.unit}

    def fmt(self) -> str:
        return f"{self.point:.3g} [{self.lower:.3g}, {self.upper:.3g}]{(' ' + self.unit) if self.unit else ''}"


@dataclass(frozen=True)
class NodeDecision:
    node: str
    verdict: Verdict
    champion_variant: Optional[str]
    work_saving: Optional[EffectWithCI]
    coverage: Optional[EffectWithCI] = None
    quality_effects: dict[str, EffectWithCI] = field(default_factory=dict)
    critical_harm_rate: Optional[float] = None
    where_it_works: tuple[str, ...] = ()
    where_it_fails: tuple[str, ...] = ()

    def display_verdict(self, status: str) -> str:
        if status == PROVISIONAL_STATUS and self.verdict in _QUALITY_DEPENDENT:
            return f"PROVISIONAL_{self.verdict.value}"
        return self.verdict.value

    def obj(self, status: str) -> dict:
        return {
            "node": self.node,
            "verdict": self.verdict.value,
            "display_verdict": self.display_verdict(status),
            "champion_variant": self.champion_variant,
            "work_saving": self.work_saving.obj() if self.work_saving else None,
            "coverage": self.coverage.obj() if self.coverage else None,
            "quality_effects": {k: v.obj() for k, v in self.quality_effects.items()},
            "critical_harm_rate": self.critical_harm_rate,
            "where_it_works": list(self.where_it_works),
            "where_it_fails": list(self.where_it_fails),
        }


@dataclass(frozen=True)
class DecisionObject:
    webpage_p1: NodeDecision
    c_visible: NodeDecision
    c_registry: NodeDecision
    h_plus_c_visible: NodeDecision
    verdict_status: str
    confirmatory_power_shortfall: bool
    human_audit_status: str
    protocol_sha: str
    freeze_sha: str
    generated_at_utc: str

    @property
    def nodes(self) -> list[NodeDecision]:
        return [self.webpage_p1, self.c_visible, self.c_registry, self.h_plus_c_visible]

    def to_json_obj(self) -> dict:
        return {
            "WEBPAGE_P1": self.webpage_p1.obj(self.verdict_status),
            "C_VISIBLE": self.c_visible.obj(self.verdict_status),
            "C_REGISTRY": self.c_registry.obj(self.verdict_status),
            "H_PLUS_C_VISIBLE": self.h_plus_c_visible.obj(self.verdict_status),
            "verdict_status": self.verdict_status,
            "confirmatory_power_shortfall": self.confirmatory_power_shortfall,
            "human_audit_status": self.human_audit_status,
            "protocol_sha": self.protocol_sha,
            "freeze_sha": self.freeze_sha,
            "generated_at_utc": self.generated_at_utc,
        }


def render_json(decision: DecisionObject) -> str:
    return canonical_str(decision.to_json_obj())


def _node_block(n: NodeDecision, status: str) -> str:
    lines = [
        f"### {n.node}: **{n.display_verdict(status)}**",
        f"- Champion variant: `{n.champion_variant or '(none advanced)'}`",
        f"- Complete work saving: {n.work_saving.fmt() if n.work_saving else 'n/a'}",
    ]
    if n.coverage:
        lines.append(f"- Eligibility coverage: {n.coverage.fmt()}")
    if n.critical_harm_rate is not None:
        lines.append(f"- Critical-harm rate: {n.critical_harm_rate:.3g}")
    if n.quality_effects:
        lines.append("- Quality effects (NI):")
        for name, eff in n.quality_effects.items():
            lines.append(f"    - {name}: {eff.fmt()}")
    if n.where_it_works:
        lines.append("- Where it works: " + "; ".join(n.where_it_works))
    if n.where_it_fails:
        lines.append("- Where it fails: " + "; ".join(n.where_it_fails))
    return "\n".join(lines)


def render_markdown(decision: DecisionObject) -> str:
    status = decision.verdict_status
    provisional_banner = ""
    if status == PROVISIONAL_STATUS:
        provisional_banner = (
            "> **PROVISIONAL — machine verdicts only.** Quality-dependent verdicts carry a "
            "`PROVISIONAL_` prefix and are NOT final until the human audit gate is discharged.\n\n"
        )

    sections = [
        f"# Week-1 P1 Decision\n\n{provisional_banner}"
        f"- verdict_status: `{status}`\n"
        f"- human_audit_status: `{decision.human_audit_status}`\n"
        f"- confirmatory_power_shortfall: `{decision.confirmatory_power_shortfall}`\n"
        f"- protocol_sha: `{decision.protocol_sha}`  freeze_sha: `{decision.freeze_sha}`\n"
        f"- generated_at_utc: `{decision.generated_at_utc}`",

        "## 1. Executive verdict\n"
        + "\n".join(
            f"- **{n.node}**: {n.display_verdict(status)}"
            f"{' — champion `' + n.champion_variant + '`' if n.champion_variant else ''}"
            for n in decision.nodes
        ),

        "## 5. WEBPAGE-P1 verdict\n" + _node_block(decision.webpage_p1, status),
        "## 6. C_VISIBLE verdict\n" + _node_block(decision.c_visible, status),
        "## 7. C_REGISTRY extension verdict\n" + _node_block(decision.c_registry, status),
        "## 8. H×C interaction\n" + _node_block(decision.h_plus_c_visible, status),

        "## 15. Human-audit status\n"
        + (f"`{decision.human_audit_status}`. "
           + ("Verdicts are provisional until this completes."
              if status == PROVISIONAL_STATUS else "Audit complete; verdicts are final.")),

        "## 16. Decision for the ShapeFlow proposal\n"
        + _proposal_line(decision),
    ]
    return "\n\n".join(sections) + "\n"


def _proposal_line(decision: DecisionObject) -> str:
    # A NOT_ESTABLISHED or MECHANISM_ONLY node is a proposal NO-GO; report it as such without
    # claiming a proven absence of effect.
    go_blockers = [
        n.node for n in decision.nodes
        if n.verdict in {Verdict.NOT_ESTABLISHED, Verdict.MECHANISM_ONLY,
                         Verdict.KILL_STRUCTURAL, Verdict.KILL_HARM, Verdict.KILL_NO_HEADROOM}
    ]
    if any(n.verdict in {Verdict.KEEP, Verdict.THESIS_GRADE, Verdict.CONDITIONAL}
           for n in decision.nodes):
        return "At least one node reached KEEP/CONDITIONAL; see per-node envelopes for scope."
    return (
        "No node reached KEEP/CONDITIONAL — proposal NO-GO for now. "
        f"Nodes not established / mechanism-only / killed: {', '.join(go_blockers)}. "
        "NOT_ESTABLISHED is a NO-GO but is not a demonstrated absence of effect."
    )
