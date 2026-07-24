"""Atomizer and the exact-then-judge matching protocol into a ReportAssessment."""

from __future__ import annotations

from shapeflow_p1.evaluation.atomizer import atomize_report, extract_anchors
from shapeflow_p1.evaluation.citation_eval import (
    CONTRADICT,
    ENTAIL,
    UNCERTAIN,
    UNRELATED,
    AtomCandidate,
    build_assessment,
)
from shapeflow_p1.evaluation.quality_metrics import score_report
from shapeflow_p1.evaluation.quality_metrics import TruthAtom, TruthPacket


# --- atomizer -------------------------------------------------------------------------


def test_atomize_splits_and_flags_factual():
    text = "The tower is 330 meters tall [E1]. It is pretty. Paris hosted the 1900 Olympics [E2]."
    claims = atomize_report(text)
    assert len(claims) == 3
    # sentences with a number or citation are factual; the bare opinion is not
    assert claims[0].factual and claims[2].factual
    assert claims[1].factual is False
    assert claims[0].citation_labels == ("[E1]",)


def test_atomizer_offsets_reconstruct():
    text = "First claim here. Second claim there."
    for c in atomize_report(text):
        assert text[c.char_start:c.char_end] == c.text


def test_extract_anchors_numbers_dates_entities():
    a = extract_anchors("Eiffel Tower opened 1889-03-31 at 300 meters.")
    assert "300" in a.numbers
    assert any("1889" in d for d in a.dates)
    assert "Eiffel Tower" in a.entities
    # the year inside the date is not double-counted as a bare number
    assert "1889" not in a.numbers


# --- matching protocol ----------------------------------------------------------------


def _judge(mapping):
    def judge(claim_text, atom_text):
        for (c_sub, a_sub), rel in mapping.items():
            if c_sub in claim_text and a_sub in atom_text:
                return rel
        return UNRELATED
    return judge


def test_supported_claim_with_supporting_citation_is_covered():
    claims = atomize_report("The tower is 330 meters tall [E1].")
    atoms = [AtomCandidate("a1", "The tower stands 330 meters.", critical=False)]
    judge = _judge({("330 meters", "330 meters"): ENTAIL})
    assessment, queue = build_assessment(
        claims, atoms, judge=judge,
        citation_supports=lambda cid, label: True,  # the citation backs the claim
    )
    truth = TruthPacket(required_facets=("f",), atoms=(TruthAtom("a1", "f", 1.0, False),))
    scores = score_report(truth, assessment)
    assert scores.weighted_required_atom_recall == 1.0
    assert not queue


def test_contradicting_critical_atom_is_harm():
    claims = atomize_report("The tower is 500 meters tall [E1].")
    atoms = [AtomCandidate("a1", "The tower is 330 meters tall.", critical=True)]
    judge = _judge({("500 meters", "330 meters"): CONTRADICT})
    assessment, _ = build_assessment(
        claims, atoms, judge=judge, citation_supports=lambda cid, label: True,
    )
    assert any(c.critical_harm for c in assessment.claims)


def test_uncertain_judge_routes_to_human_queue_not_imputed():
    claims = atomize_report("The tower height is disputed among sources [E1].")
    atoms = [AtomCandidate("a1", "The tower is 330 meters.", critical=False)]
    judge = _judge({("disputed", "330 meters"): UNCERTAIN})
    assessment, queue = build_assessment(
        claims, atoms, judge=judge, citation_supports=lambda cid, label: True,
    )
    # not marked supported or refuted; queued for a human
    assert assessment.claims[0].support_status == "UNVERIFIABLE"
    assert any(item.claim_id == "claim_0" for item in queue)


def test_citation_refuting_its_own_claim_is_harm():
    claims = atomize_report("The tower is 330 meters tall [E1].")
    atoms = [AtomCandidate("a1", "The tower is 330 meters.", critical=False)]
    judge = _judge({("330 meters", "330 meters"): ENTAIL})
    # The claim is entailed by truth, but its attached citation does NOT support it.
    assessment, _ = build_assessment(
        claims, atoms, judge=judge, citation_supports=lambda cid, label: False,
    )
    assert any(c.critical_harm for c in assessment.claims)


def test_unknown_citation_support_is_queued():
    claims = atomize_report("The tower is 330 meters tall [E1].")
    atoms = [AtomCandidate("a1", "The tower is 330 meters.", critical=False)]
    judge = _judge({("330", "330"): ENTAIL})
    _, queue = build_assessment(
        claims, atoms, judge=judge, citation_supports=lambda cid, label: None,
    )
    assert any("support unknown" in item.reason for item in queue)
