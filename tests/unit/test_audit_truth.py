"""Human audit queue selection, the upgrade gate, and truth-packet assembly."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from shapeflow_p1.evaluation.human_audit import (
    audit_gate_ready,
    select_audit_queue,
)
from shapeflow_p1.evaluation.truth_builder import CandidateAtom, assemble_truth_packet

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[2] / "schemas" / "truth_packet.schema.json").read_text()
)


# --- audit queue ----------------------------------------------------------------------


def test_queue_includes_sample_and_all_mandatory():
    tasks = {"lowvol": [f"l{i}" for i in range(10)], "highvol": [f"h{i}" for i in range(10)]}
    q = select_audit_queue(
        tasks_by_stratum=tasks, sample_fraction=0.2,
        critical_misses=["l0", "special_crit"], disagreements=["h3"], near_margin=["l9"],
        seed=1,
    )
    kinds = {(i.task_id, i.kind) for i in q}
    # every mandatory case present regardless of the sample
    assert ("special_crit", "CRITICAL_MISS") in kinds
    assert ("h3", "DISAGREEMENT") in kinds
    assert ("l9", "NEAR_MARGIN") in kinds
    # sample drawn from both strata
    sampled = [i for i in q if i.kind == "RANDOM_SAMPLE"]
    assert any(i.detail == "lowvol" for i in sampled)
    assert any(i.detail == "highvol" for i in sampled)


def test_queue_is_deterministic():
    tasks = {"s": [f"t{i}" for i in range(20)]}
    a = select_audit_queue(tasks_by_stratum=tasks, sample_fraction=0.25, seed=7)
    b = select_audit_queue(tasks_by_stratum=tasks, sample_fraction=0.25, seed=7)
    assert [i.task_id for i in a] == [i.task_id for i in b]


# --- upgrade gate ---------------------------------------------------------------------


def test_gate_ready_when_all_conditions_met():
    gate = audit_gate_ready(
        sampled_fraction=0.20, required_fraction=0.15, critical_errors=0,
        noncritical_accuracy=0.97, noncritical_accuracy_min=0.95, critical_harm_misses=0,
    )
    assert gate.ready and not gate.reasons


def test_gate_blocked_by_a_single_critical_error():
    gate = audit_gate_ready(
        sampled_fraction=0.20, required_fraction=0.15, critical_errors=1,
        noncritical_accuracy=0.99, noncritical_accuracy_min=0.95, critical_harm_misses=0,
    )
    assert not gate.ready
    assert any("critical audited error" in r for r in gate.reasons)


def test_gate_blocked_by_insufficient_sample():
    gate = audit_gate_ready(
        sampled_fraction=0.05, required_fraction=0.15, critical_errors=0,
        noncritical_accuracy=0.99, noncritical_accuracy_min=0.95, critical_harm_misses=0,
    )
    assert not gate.ready


# --- truth assembly -------------------------------------------------------------------


def test_spanless_atom_is_rejected():
    cands = [
        CandidateAtom("a1", "f1", 1.0, False, supporting_span_ids=("s" * 1,)),
        CandidateAtom("a2", "f1", 1.0, True, supporting_span_ids=()),  # no span -> rejected
    ]
    packet, rejected = assemble_truth_packet("t1", cands, required_facets=["f1"])
    assert [a["atom_id"] for a in packet["atomic_evidence"]] == ["a1"]
    assert rejected and rejected[0].atom_id == "a2"


def test_contradiction_survives_only_if_both_sides_accepted():
    cands = [
        CandidateAtom("a", "f", 1.0, False, ("x",)),
        CandidateAtom("b", "f", 1.0, False, ()),  # dropped
    ]
    packet, _ = assemble_truth_packet("t", cands, required_facets=["f"],
                                     contradiction_pairs=[("a", "b")])
    assert packet["contradiction_pairs"] == []  # half-grounded pair not presented as truth


def test_assembled_packet_validates_against_schema_and_is_provisional():
    cands = [CandidateAtom("a1", "f1", 2.0, True, ("a" * 64,))]  # span id must be 64-hex
    packet, _ = assemble_truth_packet("t1", cands, required_facets=["f1"])
    Draft202012Validator(SCHEMA).validate(packet)
    assert packet["authoring_method"] == "MACHINE_CANDIDATE_PENDING_HUMAN_AUDIT"
    assert packet["verifier_status"] == "PENDING"
    assert packet["critical_items"] == ["a1"]
