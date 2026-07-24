"""The final-report quality metrics, exactly as specified in plan §13.5.

These are the co-primary quality endpoints. They are pure functions of a frozen TruthPacket
and an arm-blind ReportClaim ledger, so the same truth version scores every arm identically
and a truth correction re-scores all arms, not just P1. Nothing here is a preference score:
each metric is a defined ratio with a defined denominator, and the qualified-report endpoint is
their intersection (all guards must pass -- a good average never compensates a failed guard).

Definitions kept faithful to the plan, including the denominator edge cases: grounded-claim
precision is 0 (not 1) when a nontrivial task emits no factual claims; contradiction handling
is NA when there are no truth contradiction pairs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "TruthAtom",
    "ContradictionPair",
    "TruthPacket",
    "ReportClaim",
    "ReportAssessment",
    "QualityScores",
    "score_report",
]


@dataclass(frozen=True)
class TruthAtom:
    atom_id: str
    facet_id: str
    weight: float
    critical: bool
    # facets the TruthPacket itself records as unresolved (a critical atom on such a facet is
    # "safe" if the report explicitly reports it unresolved).
    known_unresolved: bool = False


@dataclass(frozen=True)
class ContradictionPair:
    atom_id_a: str
    atom_id_b: str


@dataclass(frozen=True)
class TruthPacket:
    required_facets: tuple[str, ...]
    atoms: tuple[TruthAtom, ...]
    contradiction_pairs: tuple[ContradictionPair, ...] = ()

    def atoms_for_facet(self, facet_id: str) -> list[TruthAtom]:
        return [a for a in self.atoms if a.facet_id == facet_id]


@dataclass(frozen=True)
class ReportClaim:
    """One atomic claim from a report, already matched by the (arm-blind) claim/citation eval.

    ``supporting_citation`` means at least one linked citation occurrence's accepted source span
    actually supports this claim -- the gate that turns a bare match into 'covered'.
    """

    claim_id: str
    factual: bool
    support_status: str  # SUPPORTED | CONTRADICTED | UNSUPPORTED | UNVERIFIABLE
    matched_atom_ids: tuple[str, ...] = ()
    has_citation: bool = False
    supporting_citation: bool = False
    critical_harm: bool = False


@dataclass(frozen=True)
class ReportAssessment:
    """The report-level facts the metrics need beyond per-claim data."""

    claims: tuple[ReportClaim, ...]
    reported_unresolved_facets: frozenset[str] = frozenset()
    # citation edges: (occurrence supports an adjacent claim?) booleans, one per emitted citation
    emitted_citations_support: tuple[bool, ...] = ()
    # claim-citation edges: one bool per edge (does the edge's occurrence support that claim?)
    claim_citation_edges_support: tuple[bool, ...] = ()
    terminal_failure: bool = False
    empty_or_truncated: bool = False


@dataclass(frozen=True)
class QualityScores:
    weighted_required_atom_recall: float
    critical_atom_safety: float
    grounded_claim_precision: float
    citation_correctness: float
    citation_association: float
    citation_completeness: float
    required_facet_coverage: float
    contradiction_handling: Optional[float]  # None == NA
    critical_harm: int
    qualified_report: int


def _covered_atom_ids(assessment: ReportAssessment) -> set[str]:
    """An atom is covered iff some SUPPORTED claim with a supporting citation matched it."""
    covered: set[str] = set()
    for c in assessment.claims:
        if c.support_status == "SUPPORTED" and c.supporting_citation:
            covered.update(c.matched_atom_ids)
    return covered


def score_report(truth: TruthPacket, assessment: ReportAssessment) -> QualityScores:
    covered = _covered_atom_ids(assessment)
    required_atoms = [a for a in truth.atoms if a.facet_id in truth.required_facets]

    # weighted_required_atom_recall
    total_w = sum(a.weight for a in required_atoms)
    covered_w = sum(a.weight for a in required_atoms if a.atom_id in covered)
    recall = (covered_w / total_w) if total_w else 0.0

    # critical_atom_safety: covered, OR known-unresolved and explicitly reported unresolved.
    critical_atoms = [a for a in truth.atoms if a.critical]
    if critical_atoms:
        safe = 0
        for a in critical_atoms:
            if a.atom_id in covered:
                safe += 1
            elif a.known_unresolved and a.facet_id in assessment.reported_unresolved_facets:
                safe += 1
        critical_safety = safe / len(critical_atoms)
    else:
        critical_safety = 1.0

    # grounded_claim_precision: SUPPORTED factual / all factual (0 if none, for a nontrivial task)
    factual = [c for c in assessment.claims if c.factual]
    supported_factual = [c for c in factual if c.support_status == "SUPPORTED"]
    grounded = (len(supported_factual) / len(factual)) if factual else 0.0

    # citation_correctness: citations whose occurrence supports >=1 adjacent claim / all citations
    cites = assessment.emitted_citations_support
    citation_correctness = (sum(cites) / len(cites)) if cites else 0.0

    # citation_association: supported claim-citation edges / all edges
    edges = assessment.claim_citation_edges_support
    citation_association = (sum(edges) / len(edges)) if edges else 0.0

    # citation_completeness: supported factual claims with a citation / all supported factual
    if supported_factual:
        with_cite = sum(1 for c in supported_factual if c.has_citation)
        citation_completeness = with_cite / len(supported_factual)
    else:
        citation_completeness = 0.0

    # required_facet_coverage: facets with a covered required atom OR a correct unresolved marker
    covered_facets = 0
    for facet in truth.required_facets:
        facet_atoms = [a for a in required_atoms if a.facet_id == facet]
        if any(a.atom_id in covered for a in facet_atoms):
            covered_facets += 1
        elif all(a.known_unresolved for a in facet_atoms) and facet in assessment.reported_unresolved_facets:
            covered_facets += 1
    facet_coverage = (covered_facets / len(truth.required_facets)) if truth.required_facets else 0.0

    # contradiction_handling: pairs with both sides covered / pairs (NA if none)
    if truth.contradiction_pairs:
        handled = sum(
            1 for p in truth.contradiction_pairs
            if p.atom_id_a in covered and p.atom_id_b in covered
        )
        contradiction_handling: Optional[float] = handled / len(truth.contradiction_pairs)
    else:
        contradiction_handling = None

    # critical_harm: any per-claim harm flag, or a contradicted/unsupported critical factual claim
    critical_harm = 0
    if any(c.critical_harm for c in assessment.claims):
        critical_harm = 1

    scores = QualityScores(
        weighted_required_atom_recall=recall,
        critical_atom_safety=critical_safety,
        grounded_claim_precision=grounded,
        citation_correctness=citation_correctness,
        citation_association=citation_association,
        citation_completeness=citation_completeness,
        required_facet_coverage=facet_coverage,
        contradiction_handling=contradiction_handling,
        critical_harm=critical_harm,
        qualified_report=0,
    )
    return _with_qualified(scores, assessment)


# Absolute thresholds for the qualified-report endpoint (plan §13.5). These are the
# report-quality bars; the NI margins in configs/decision.yaml are a separate, arm-relative
# gate applied by the analysis stage.
_Q = dict(
    recall=0.85, grounded=0.95, citation_correctness=0.95, citation_association=0.90,
    facet_coverage=0.90,
)


def _with_qualified(s: QualityScores, assessment: ReportAssessment) -> QualityScores:
    ok = (
        not assessment.empty_or_truncated
        and not assessment.terminal_failure
        and s.weighted_required_atom_recall >= _Q["recall"]
        and s.critical_atom_safety >= 1.0
        and s.grounded_claim_precision >= _Q["grounded"]
        and s.citation_correctness >= _Q["citation_correctness"]
        and s.citation_association >= _Q["citation_association"]
        and s.required_facet_coverage >= _Q["facet_coverage"]
        and s.critical_harm == 0
    )
    return QualityScores(**{**s.__dict__, "qualified_report": 1 if ok else 0})
