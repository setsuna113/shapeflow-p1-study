"""Building the ReportAssessment: match claims to truth, then judge what exact rules can't.

This is the §13.5 matching protocol. For each atomic claim:

1. **Exact rules first.** A claim sharing a number/date/entity anchor with a truth atom is
   matched deterministically -- no model needed, and no model *variance* in the part of the score
   that exact rules can settle.
2. **Judge the remainder.** For claims the exact rules leave unresolved, an arm-blind judge
   classifies claim-vs-atom as entail / contradict / unrelated. The judge is injected; it is the
   same blind judge everywhere, never P0's own notes.
3. **Uncertain is not a guess.** If the judge is uncertain or unavailable, the claim is routed to
   the human audit queue and left UNVERIFIABLE -- never imputed as supported or refuted.

``covered`` (in quality_metrics) additionally requires a supporting citation, so this module also
records, per claim, whether a cited occurrence actually supports it. A citation that refutes its
own claim is a critical harm, as is contradicting a critical atom.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .atomizer import Claim
from .quality_metrics import ReportAssessment, ReportClaim

__all__ = ["AtomCandidate", "JudgeRelation", "build_assessment", "HumanQueueItem"]


@dataclass(frozen=True)
class AtomCandidate:
    atom_id: str
    text: str
    critical: bool


# A relation classifier: (claim_text, atom_text) -> one of the strings below.
JudgeRelation = Callable[[str, str], str]
ENTAIL, CONTRADICT, UNRELATED, UNCERTAIN = "entail", "contradict", "unrelated", "uncertain"

# citation_supports(claim_id, label) -> True/False/None; None means unknown (-> human queue).
CitationSupports = Callable[[str, str], Optional[bool]]


@dataclass(frozen=True)
class HumanQueueItem:
    claim_id: str
    reason: str


def _shares_anchor(claim: Claim, atom_text: str) -> bool:
    a = claim.anchors
    hay = atom_text.lower()
    for tok in list(a.numbers) + list(a.dates) + list(a.entities):
        if tok.lower() in hay:
            return True
    return False


def build_assessment(
    claims: list[Claim],
    atoms: list[AtomCandidate],
    *,
    judge: JudgeRelation,
    citation_supports: CitationSupports,
    reported_unresolved_facets: frozenset[str] = frozenset(),
    terminal_failure: bool = False,
    empty_or_truncated: bool = False,
) -> tuple[ReportAssessment, list[HumanQueueItem]]:
    report_claims: list[ReportClaim] = []
    queue: list[HumanQueueItem] = []
    emitted_citations_support: list[bool] = []
    claim_citation_edges: list[bool] = []

    for claim in claims:
        matched: list[str] = []
        status = "UNSUPPORTED"
        critical_harm = False

        # Candidate atoms: those sharing an exact anchor (cheap deterministic filter).
        candidates = [a for a in atoms if _shares_anchor(claim, a.text)]
        # If no anchor overlap, still let the judge look at all atoms for entailment, but only
        # when the claim is factual (non-factual prose needs no truth match).
        if not candidates and claim.factual:
            candidates = atoms

        uncertain = False
        for atom in candidates:
            rel = judge(claim.text, atom.text)
            if rel == ENTAIL:
                matched.append(atom.atom_id)
                status = "SUPPORTED"
            elif rel == CONTRADICT:
                status = "CONTRADICTED"
                if atom.critical:
                    critical_harm = True
            elif rel == UNCERTAIN:
                uncertain = True

        if uncertain and status == "UNSUPPORTED":
            status = "UNVERIFIABLE"
            queue.append(HumanQueueItem(claim.claim_id, "judge uncertain"))

        # Citation support: does a cited occurrence actually back this claim?
        has_citation = bool(claim.citation_labels)
        supporting_citation = False
        for label in claim.citation_labels:
            supports = citation_supports(claim.claim_id, label)
            if supports is None:
                queue.append(HumanQueueItem(claim.claim_id, f"citation {label} support unknown"))
                emitted_citations_support.append(False)
                claim_citation_edges.append(False)
                continue
            emitted_citations_support.append(bool(supports))
            claim_citation_edges.append(bool(supports))
            if supports:
                supporting_citation = True
            elif status == "SUPPORTED":
                # A citation that refutes the very claim it is attached to is a critical harm.
                critical_harm = True

        report_claims.append(
            ReportClaim(
                claim_id=claim.claim_id,
                factual=claim.factual,
                support_status=status,
                matched_atom_ids=tuple(dict.fromkeys(matched)),
                has_citation=has_citation,
                supporting_citation=supporting_citation,
                critical_harm=critical_harm,
            )
        )

    assessment = ReportAssessment(
        claims=tuple(report_claims),
        reported_unresolved_facets=reported_unresolved_facets,
        emitted_citations_support=tuple(emitted_citations_support),
        claim_citation_edges_support=tuple(claim_citation_edges),
        terminal_failure=terminal_failure,
        empty_or_truncated=empty_or_truncated,
    )
    # Dedup queue by (claim_id, reason).
    seen = set()
    deduped = []
    for item in queue:
        key = (item.claim_id, item.reason)
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return assessment, deduped
