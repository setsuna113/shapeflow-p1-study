"""The Week-1 decision renders consistently to Markdown and JSON, with provisional gating."""

from __future__ import annotations

import json
from dataclasses import replace

from shapeflow_p1.analysis.decision import Verdict
from shapeflow_p1.analysis.report import (
    AUDITED_STATUS,
    PROVISIONAL_STATUS,
    DecisionObject,
    EffectWithCI,
    NodeDecision,
    render_json,
    render_markdown,
)


def _node(node, verdict, champion="H03"):
    return NodeDecision(
        node=node,
        verdict=verdict,
        champion_variant=champion,
        work_saving=EffectWithCI(0.22, 0.12, 0.31, unit="frac"),
        coverage=EffectWithCI(0.4, 0.32, 0.48),
        quality_effects={"weighted_recall": EffectWithCI(-0.01, -0.03, 0.01, "pp")},
        critical_harm_rate=0.0,
        where_it_works=("high evidence volume", "table-heavy"),
        where_it_fails=("single-source facts",),
    )


def _decision(status=PROVISIONAL_STATUS, webpage_verdict=Verdict.KEEP):
    return DecisionObject(
        webpage_p1=_node("WEBPAGE_P1", webpage_verdict),
        c_visible=_node("C_VISIBLE", Verdict.MECHANISM_ONLY),
        c_registry=_node("C_REGISTRY", Verdict.NOT_ESTABLISHED, champion=None),
        h_plus_c_visible=_node("H_PLUS_C_VISIBLE", Verdict.NOT_ESTABLISHED, champion=None),
        verdict_status=status,
        confirmatory_power_shortfall=False,
        human_audit_status="0/48 audited",
        protocol_sha="abc",
        freeze_sha="def",
        generated_at_utc="2026-07-30T00:00:00Z",
    )


def test_json_and_markdown_agree_on_verdicts():
    d = _decision()
    j = json.loads(render_json(d))
    md = render_markdown(d)
    for key in ("WEBPAGE_P1", "C_VISIBLE", "C_REGISTRY", "H_PLUS_C_VISIBLE"):
        disp = j[key]["display_verdict"]
        assert disp in md  # the MD shows exactly the JSON's display verdict


def test_provisional_prefixes_quality_dependent_verdicts():
    d = _decision(status=PROVISIONAL_STATUS)
    j = json.loads(render_json(d))
    # KEEP is quality-dependent -> shown provisional.
    assert j["WEBPAGE_P1"]["display_verdict"] == "PROVISIONAL_KEEP"
    assert j["C_VISIBLE"]["display_verdict"] == "PROVISIONAL_MECHANISM_ONLY"
    # NOT_ESTABLISHED is not in the quality-dependent set -> no prefix.
    assert j["C_REGISTRY"]["display_verdict"] == "NOT_ESTABLISHED"
    proposal = render_markdown(d).split(
        "## 16. Decision for the ShapeFlow proposal", 1
    )[1].split("## 17.", 1)[0]
    assert "NO FINAL GO" in proposal
    assert "reached KEEP/CONDITIONAL" not in proposal


def test_no_headroom_is_provisional_until_quality_audit():
    d = _decision(
        status=PROVISIONAL_STATUS,
        webpage_verdict=Verdict.KILL_NO_HEADROOM,
    )
    assert json.loads(render_json(d))["WEBPAGE_P1"]["display_verdict"] == (
        "PROVISIONAL_KILL_NO_HEADROOM"
    )


def test_audited_status_drops_provisional_prefix():
    d = _decision(status=AUDITED_STATUS)
    j = json.loads(render_json(d))
    assert j["WEBPAGE_P1"]["display_verdict"] == "KEEP"


def test_json_is_canonical_and_reparses():
    d = _decision()
    a = render_json(d)
    b = render_json(d)
    assert a == b
    assert json.loads(a)["verdict_status"] == PROVISIONAL_STATUS


def test_markdown_carries_provisional_banner():
    assert "PROVISIONAL" in render_markdown(_decision(PROVISIONAL_STATUS))
    md_final = render_markdown(_decision(AUDITED_STATUS))
    assert "Audit complete" in md_final


def test_proposal_line_is_no_go_when_nothing_keeps():
    d = _decision(webpage_verdict=Verdict.NOT_ESTABLISHED)
    md = render_markdown(d)
    assert "NO-GO" in md
    assert "not a demonstrated absence of effect" in md


def test_matched_not_estimable_status_is_not_rendered_as_recorded():
    d = _decision(webpage_verdict=Verdict.NOT_ESTABLISHED)
    d = replace(
        d,
        matched_variant_outcomes={
            "decision_use": "SECONDARY_EXPLORATORY",
            "contrasts": [
                {
                    "contrast_id": "H_ID_VS_TYPED",
                    "pairing_status": "NOT_ESTIMABLE",
                    "endpoint_status_counts": {"NOT_ESTIMABLE": 12},
                }
            ],
        },
    )

    md = render_markdown(d)

    assert "pairing_status=`NOT_ESTIMABLE`" in md
    assert "status=`RECORDED`" not in md


def test_report_never_reduces_to_faster_on_average():
    md = render_markdown(_decision())
    # It must report coverage, where-it-works, where-it-fails, and quality -- not just speed.
    assert "Where it works" in md and "Where it fails" in md
    assert "Eligibility coverage" in md and "Quality effects" in md


def test_markdown_has_every_fixed_protocol_section_and_json_source_fields():
    decision = _decision()
    md = render_markdown(decision)
    parsed = json.loads(render_json(decision))

    expected = {
        1: "Executive verdict",
        2: "Frozen scope and stack",
        3: "What actually ran",
        4: "Variant and control comparison",
        5: "WEBPAGE-P1 verdict",
        6: "C_VISIBLE verdict",
        7: "C_REGISTRY extension verdict",
        8: "H×C interaction",
        9: "Average, median, and tail behavior",
        10: "Quality and harm",
        11: "Complete work and cost balance",
        12: "Eligibility, when useful, and coverage",
        13: "Fallback, restart, and terminal failure",
        14: "Sensitivity and mediated trajectory outcomes",
        15: "Human-audit status",
        16: "Decision for the ShapeFlow proposal",
        17: "Bound inputs and limitations",
        18: "Reproduction commands",
        19: "Artifact and configuration hashes",
    }
    for number, title in expected.items():
        assert f"## {number}. {title}" in md
    for field in (
        "executed_design",
        "matched_variant_outcomes",
        "distribution_outcomes",
        "work_outcomes",
        "sensitivity_outcomes",
        "trajectory_outcomes",
        "reproduction_commands",
        "artifact_locations",
    ):
        assert field in parsed
