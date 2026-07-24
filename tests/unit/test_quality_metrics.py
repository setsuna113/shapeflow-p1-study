"""Quality metrics: each formula, including the denominator edge cases from plan §13.5."""

from __future__ import annotations

import pytest

from shapeflow_p1.evaluation.quality_metrics import (
    ContradictionPair,
    ReportAssessment,
    ReportClaim,
    TruthAtom,
    TruthPacket,
    score_report,
)


def _atom(aid, facet, w=1.0, critical=False, unresolved=False):
    return TruthAtom(atom_id=aid, facet_id=facet, weight=w, critical=critical, known_unresolved=unresolved)


def _claim(cid, *, factual=True, status="SUPPORTED", atoms=(), cite=True, supporting=True, harm=False):
    return ReportClaim(
        claim_id=cid, factual=factual, support_status=status, matched_atom_ids=tuple(atoms),
        has_citation=cite, supporting_citation=supporting, critical_harm=harm,
    )


def test_weighted_recall_uses_weights():
    truth = TruthPacket(
        required_facets=("f1", "f2"),
        atoms=(_atom("a1", "f1", w=3.0), _atom("a2", "f2", w=1.0)),
    )
    # Only the heavy atom a1 is covered -> 3/4.
    assess = ReportAssessment(claims=(_claim("c1", atoms=("a1",)),))
    s = score_report(truth, assess)
    assert s.weighted_required_atom_recall == pytest.approx(0.75)


def test_covered_requires_supporting_citation():
    truth = TruthPacket(required_facets=("f1",), atoms=(_atom("a1", "f1"),))
    # SUPPORTED but no supporting citation -> not covered.
    assess = ReportAssessment(claims=(_claim("c1", atoms=("a1",), supporting=False),))
    assert score_report(truth, assess).weighted_required_atom_recall == 0.0


def test_critical_atom_safety_via_reported_unresolved():
    truth = TruthPacket(
        required_facets=("f1",),
        atoms=(_atom("crit", "f1", critical=True, unresolved=True),),
    )
    # Not covered, but the facet is known-unresolved AND the report says so -> safe.
    assess = ReportAssessment(claims=(), reported_unresolved_facets=frozenset({"f1"}))
    assert score_report(truth, assess).critical_atom_safety == 1.0
    # If the report does NOT mark it unresolved, it is unsafe.
    assess2 = ReportAssessment(claims=())
    assert score_report(truth, assess2).critical_atom_safety == 0.0


def test_grounded_precision_zero_when_no_factual_claims():
    truth = TruthPacket(required_facets=("f1",), atoms=(_atom("a1", "f1"),))
    # A nontrivial task that emits no factual claims scores 0, not 1.
    assess = ReportAssessment(claims=(_claim("c1", factual=False),))
    assert score_report(truth, assess).grounded_claim_precision == 0.0


def test_grounded_precision_ratio():
    truth = TruthPacket(required_facets=("f1",), atoms=(_atom("a1", "f1"),))
    assess = ReportAssessment(claims=(
        _claim("c1", status="SUPPORTED"),
        _claim("c2", status="UNSUPPORTED"),
    ))
    assert score_report(truth, assess).grounded_claim_precision == pytest.approx(0.5)


def test_citation_correctness_and_association():
    truth = TruthPacket(required_facets=("f1",), atoms=(_atom("a1", "f1"),))
    assess = ReportAssessment(
        claims=(_claim("c1", atoms=("a1",)),),
        emitted_citations_support=(True, True, False),   # 2/3
        claim_citation_edges_support=(True, False),       # 1/2
    )
    s = score_report(truth, assess)
    assert s.citation_correctness == pytest.approx(2 / 3)
    assert s.citation_association == pytest.approx(0.5)


def test_contradiction_handling_na_when_no_pairs():
    truth = TruthPacket(required_facets=("f1",), atoms=(_atom("a1", "f1"),))
    assert score_report(truth, ReportAssessment(claims=())).contradiction_handling is None


def test_contradiction_handling_needs_both_sides_covered():
    truth = TruthPacket(
        required_facets=("f1",),
        atoms=(_atom("a", "f1"), _atom("b", "f1")),
        contradiction_pairs=(ContradictionPair("a", "b"),),
    )
    # Only one side covered -> 0.
    one = ReportAssessment(claims=(_claim("c1", atoms=("a",)),))
    assert score_report(truth, one).contradiction_handling == 0.0
    # Both sides covered -> 1.
    both = ReportAssessment(claims=(_claim("c1", atoms=("a", "b")),))
    assert score_report(truth, both).contradiction_handling == 1.0


def test_qualified_report_is_intersection_of_guards():
    truth = TruthPacket(required_facets=("f1",), atoms=(_atom("a1", "f1"),))
    # A report that passes everything.
    good = ReportAssessment(
        claims=(_claim("c1", atoms=("a1",)),),
        emitted_citations_support=(True,),
        claim_citation_edges_support=(True,),
    )
    assert score_report(truth, good).qualified_report == 1


def test_qualified_report_fails_on_single_harm_even_if_average_high():
    truth = TruthPacket(required_facets=("f1",), atoms=(_atom("a1", "f1"),))
    harmed = ReportAssessment(
        claims=(_claim("c1", atoms=("a1",)), _claim("c2", harm=True)),
        emitted_citations_support=(True,),
        claim_citation_edges_support=(True,),
    )
    s = score_report(truth, harmed)
    assert s.critical_harm == 1
    assert s.qualified_report == 0  # a good average cannot compensate a harm


def test_qualified_report_fails_on_truncation():
    truth = TruthPacket(required_facets=("f1",), atoms=(_atom("a1", "f1"),))
    assess = ReportAssessment(
        claims=(_claim("c1", atoms=("a1",)),),
        emitted_citations_support=(True,), claim_citation_edges_support=(True,),
        empty_or_truncated=True,
    )
    assert score_report(truth, assess).qualified_report == 0
