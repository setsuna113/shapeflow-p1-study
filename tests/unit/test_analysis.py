"""Analysis: cluster bootstrap reproducibility, work-saving, verdict tree, corrections."""

from __future__ import annotations

import numpy as np
import pytest

from shapeflow_p1.analysis.bootstrap import (
    cluster_bootstrap_ci,
    paired_log_ratio_saving,
    seed_from,
)
from shapeflow_p1.analysis.decision import (
    NodeEvidence,
    Verdict,
    decide,
    holm_reject,
    intersection_union_pass,
)
from shapeflow_p1.analysis.estimands import (
    _difference_ci,
    _paired_risk_difference,
    _work,
    build_verdict_inputs,
)
from shapeflow_p1.canonical import canonical_json
from shapeflow_p1.hashing import sha256_hex

# --- bootstrap ------------------------------------------------------------------------


def test_seed_is_deterministic():
    assert seed_from("protocol", "holdout", "H02") == seed_from("protocol", "holdout", "H02")
    assert seed_from("a") != seed_from("b")


def test_cluster_bootstrap_is_reproducible():
    vals = list(range(20))
    clusters = [f"c{i % 5}" for i in range(20)]
    a = cluster_bootstrap_ci(vals, clusters, np.mean, n_boot=500, seed=42, side="two")
    b = cluster_bootstrap_ci(vals, clusters, np.mean, n_boot=500, seed=42, side="two")
    assert (a.point, a.lower, a.upper) == (b.point, b.lower, b.upper)


def test_cluster_bootstrap_requires_two_clusters():
    with pytest.raises(ValueError, match="2 clusters"):
        cluster_bootstrap_ci([1.0, 2.0], ["c", "c"], np.mean)


def test_cluster_bootstrap_weights_clusters_not_rows():
    """One one-row cluster and one nine-row cluster each own half the estimand."""
    values = [0.0] + [1.0] * 9
    clusters = ["small"] + ["large"] * 9
    ci = cluster_bootstrap_ci(values, clusters, np.mean, n_boot=200, seed=3)
    assert ci.point == pytest.approx(0.5)


def test_work_saving_positive_when_p1_cheaper():
    # P1 uses ~70% of P0 work across clusters -> ~30% saving, LCB should be well above 0.
    rng = np.random.default_rng(0)
    p0 = rng.uniform(8, 12, size=40)
    p1 = p0 * rng.uniform(0.65, 0.75, size=40)
    clusters = [f"c{i % 8}" for i in range(40)]
    ci = paired_log_ratio_saving(p1, p0, clusters, n_boot=1000, seed=7)
    assert ci.point > 0.2
    assert ci.lower > 0.0  # 95% lower bound clears zero


def test_work_saving_rejects_nonpositive():
    with pytest.raises(ValueError, match="strictly positive"):
        paired_log_ratio_saving([1.0, 0.0], [2.0, 2.0], ["a", "b"])


def test_near_zero_work_effect_has_finite_one_sided_lcb_and_ucb():
    p0 = [10.0] * 8
    p1 = [8.5, 9.0, 9.5, 9.8, 10.2, 10.5, 11.0, 11.5]
    clusters = [f"c{i}" for i in range(8)]
    lower = paired_log_ratio_saving(p1, p0, clusters, n_boot=2000, seed=19, side="lower")
    upper = paired_log_ratio_saving(p1, p0, clusters, n_boot=2000, seed=19, side="upper")

    assert np.isfinite(lower.lower)
    assert np.isfinite(upper.upper)
    assert lower.lower < 0.0 < upper.upper
    assert lower.upper == float("inf")
    assert upper.lower == float("-inf")


# --- verdict tree ---------------------------------------------------------------------


def _ev(**over):
    base = dict(
        structural_pass=True,
        deterministic_harm=False,
        quality_guards_pass=True,
        isolated_saving_lcb=0.2,
        saving_ucb=0.3,
        operational_available=True,
        operational_saving_lcb=0.15,
        operational_quality_pass=True,
        e2e_speedup_lcb=1.0,
    )
    base.update(over)
    return NodeEvidence(**base)


def test_keep_when_both_layers_pass():
    assert decide(_ev()) is Verdict.KEEP


def test_thesis_grade_needs_speedup():
    assert decide(_ev(e2e_speedup_lcb=1.6)) is Verdict.THESIS_GRADE


def test_structural_failure_dominates():
    assert decide(_ev(structural_pass=False, quality_guards_pass=True)) is Verdict.KILL_STRUCTURAL


def test_measurement_failure_is_not_misreported_as_a_killed_mechanism():
    assert (
        decide(_ev(study_valid=False, structural_pass=False, quality_guards_pass=True))
        is Verdict.NOT_ESTABLISHED
    )


def test_harm_dominates_even_with_savings():
    assert decide(_ev(deterministic_harm=True)) is Verdict.KILL_HARM


def test_isolated_pass_operational_fail_is_mechanism_only():
    v = decide(_ev(operational_saving_lcb=0.0, operational_quality_pass=False))
    assert v is Verdict.MECHANISM_ONLY


def test_underpowered_operational_is_mechanism_only():
    v = decide(_ev(operational_available=False))
    assert v is Verdict.MECHANISM_ONLY


def test_no_headroom_when_quality_fine_but_saving_bounded_low():
    v = decide(
        _ev(
            isolated_saving_lcb=-0.05,
            saving_ucb=0.05,
            operational_saving_lcb=0.0,
            operational_quality_pass=False,
        )
    )
    assert v is Verdict.KILL_NO_HEADROOM


def test_not_established_when_nothing_proven():
    # Quality guards fail (so not no-headroom), savings inconclusive, no conditional rule.
    v = decide(
        _ev(
            quality_guards_pass=False,
            isolated_saving_lcb=0.0,
            saving_ucb=0.5,
            operational_saving_lcb=0.0,
            operational_quality_pass=False,
        )
    )
    assert v is Verdict.NOT_ESTABLISHED


def test_conditional_when_subset_keeps_with_coverage():
    v = decide(
        _ev(
            operational_saving_lcb=0.0,
            operational_quality_pass=False,
            conditional_rule_keeps=True,
            coverage_lcb=0.35,
        )
    )
    assert v is Verdict.CONDITIONAL


def test_conditional_needs_coverage():
    v = decide(
        _ev(
            operational_saving_lcb=0.0,
            operational_quality_pass=False,
            conditional_rule_keeps=True,
            coverage_lcb=0.25,
        )
    )
    assert v is not Verdict.CONDITIONAL  # coverage below 0.30


# --- guard combiners ------------------------------------------------------------------


def test_intersection_union_all_must_pass():
    margins = {"recall": -0.05, "citation": -0.03}
    assert intersection_union_pass({"recall": -0.02, "citation": -0.01}, margins)
    # one guard below margin -> whole family fails
    assert not intersection_union_pass({"recall": -0.06, "citation": -0.01}, margins)


def test_holm_step_down():
    # Classic Holm behavior: smallest p compared to alpha/m, and a failure stops the chain.
    pvals = {"a": 0.001, "b": 0.04, "c": 0.20}
    rej = holm_reject(pvals, alpha=0.05)
    assert rej["a"] is True  # 0.001 <= 0.05/3
    assert rej["b"] is False  # 0.04 > 0.05/2 -> retained
    assert rej["c"] is False  # chain stopped


def test_holm_rejects_all_when_tiny():
    rej = holm_reject({"a": 0.001, "b": 0.002, "c": 0.003}, alpha=0.05)
    assert all(rej.values())


def _quality_view(value):
    return {
        "weighted_required_atom_recall": value,
        "grounded_claim_precision": value,
        "citation_correctness": value,
        "citation_association": value,
        "required_facet_coverage": value,
        "qualified_report": int(value >= 0.5),
        "critical_harm": 0,
    }


def _itt_row(value, *, fallback=False, unavailable=False):
    failure = _quality_view(0.0)
    best = _quality_view(1.0)
    return {
        "assignment_state": "COMMITTED",
        "fell_back": fallback,
        "evaluation_status": "JUDGE_UNAVAILABLE" if unavailable else "SCORED",
        "direct_node_metrics": {
            "status": "OK",
            "selector_guards_complete": True,
            "by_node": {
                "H": {
                    "applicable": True,
                    "status": "OK",
                    "selector_guards_complete": True,
                    "selector_normalization": {
                        "status": "OK",
                        "selector_attempt_count": 1,
                        "strict_valid_count": 1,
                        "repaired_attempt_count": 0,
                        "rejected_attempt_count": 0,
                        "invalid_id_count": 0,
                        "schema_rejection_count": 0,
                        "semantic_conflict_attempt_count": 0,
                        "semantic_conflict_event_count": 0,
                        "failure_before_parse_count": 0,
                        "normalization_missing_count": 0,
                        "normalization_invalid_count": 0,
                        "normalization_trace_error_count": 0,
                        "component_failure_count": 0,
                        "no_repair_adverse": False,
                    },
                    "candidate_coverage": 0.9,
                    "total_published_prechunk_recall": 0.9,
                    "selected_token_precision": 0.75,
                    "weighted_truth_per_100_rendered_tokens": 2.5,
                    "materialization_ratio": 1.2,
                    "token_trace_complete": True,
                    "published_rendered_tokens": 120,
                    "offered_evidence_tokens": 100,
                },
                "C": {"applicable": False},
            },
        },
        "work_summary": {
            "telemetry_complete": True,
            "overlap_valid": True,
            "service_seconds": 7.0 if fallback else 6.0,
        },
        "quality_views": {
            "strict": None if unavailable else (failure if fallback else _quality_view(value)),
            "fallback_assisted": None if unavailable else _quality_view(value),
            "worst_case": failure if unavailable else _quality_view(value),
            "best_case": best if unavailable else _quality_view(value),
        },
    }


def _scope_receipt(records, *, invalid_blocks=frozenset()):
    """Seal synthetic scores the same way campaign.evaluate seals production scores."""
    schedule_sha = "a" * 64
    root_sha = "b" * 64
    execution_binding_sha = "1" * 64
    protocol_document_sha = "2" * 64
    expected = []
    for record in records:
        record.setdefault("judge_policy_sha256", "f" * 64)
        record["execution_binding_sha256"] = execution_binding_sha
        record["protocol_document_sha256"] = protocol_document_sha
        block_id = record["block_id"]
        block_freeze = sha256_hex(f"freeze:{block_id}".encode())
        block_digest = sha256_hex(f"block:{block_id}".encode())
        validity = block_id not in invalid_blocks
        invalid_reason = "" if validity else "SPANS_ENGINE_EPOCHS"
        epoch_by_arm = {
            key: ("epoch-a" if validity or key.startswith("P0:") else "epoch-b")
            for key in record["per_arm"]
        }
        epochs = sorted(set(epoch_by_arm.values()))
        frozen_scope = {
            "freeze_root_sha256": root_sha,
            "schedule_sha256": schedule_sha,
            "block_freeze_sha256": block_freeze,
            "block_digest": block_digest,
            "valid_for_paired_estimate": validity,
            "invalid_reason": invalid_reason,
            "engine_epochs": epochs,
            "engine_epoch_by_arm": dict(sorted(epoch_by_arm.items())),
        }
        for row in record["per_arm"].values():
            row["frozen_scope"] = frozen_scope
        record["content_sha256"] = sha256_hex(canonical_json(record))
        expected.append(
            {
                "block_id": block_id,
                "task_id": record["task_id"],
                "replicate_id": "",
                "score_content_sha256": record["content_sha256"],
                "execution_binding_sha256": execution_binding_sha,
                "protocol_document_sha256": protocol_document_sha,
                "block_freeze_sha256": block_freeze,
                "block_digest": block_digest,
                "valid_for_paired_estimate": validity,
                "invalid_reason": invalid_reason,
                "engine_epochs": epochs,
                "engine_epoch_by_arm": dict(sorted(epoch_by_arm.items())),
            }
        )
    receipt = {
        "schema_version": "evaluated_itt_scope_v1",
        "run_id": "r",
        "phase_id": "p",
        "execution_binding_sha256": execution_binding_sha,
        "protocol_document_sha256": protocol_document_sha,
        "schedule_sha256": schedule_sha,
        "freeze_root_sha256": root_sha,
        "analysis_design_receipt_sha256": "c" * 64,
        "task_feature_registry_sha256": "d" * 64,
        "eligibility_spec_content_sha256": "e" * 64,
        "all_offered_blocks": len(expected),
        "scores": expected,
    }
    receipt["evaluation_scope_sha256"] = sha256_hex(canonical_json(receipt))
    return receipt


def _human_audit_receipt(records, scope):
    truth_by_task = {record["task_id"]: record["truth_packet_sha256"] for record in records}
    score_by_block = {record["block_id"]: record["content_sha256"] for record in records}
    receipt = {
        "schema_version": "human_audit_receipt_v2",
        "status": "AUDITED",
        "run_id": "r",
        "phase_id": "p",
        "execution_binding_sha256": scope["execution_binding_sha256"],
        "protocol_document_sha256": scope["protocol_document_sha256"],
        "evaluation_scope_sha256": scope["evaluation_scope_sha256"],
        "queue_content_sha256": "4" * 64,
        "results_content_sha256": "5" * 64,
        "reviewer_id": "reviewer",
        "reviewed_at_utc": "2026-07-25T00:00:00Z",
        "reviewer_attestation_sha256": "6" * 64,
        "truth_packet_sha256_by_task": dict(sorted(truth_by_task.items())),
        "score_content_sha256_by_block": dict(sorted(score_by_block.items())),
        "sampled_fraction": 1.0,
        "required_fraction": 0.1,
        "audit_basis": "ORIGINAL_ARTIFACTS_CORRECTION_FREE",
        "correction_count": 0,
        "items_requiring_correction": 0,
        "error_counts_by_check": {
            "critical_truth_atoms_correct": 0,
            "critical_source_edges_correct": 0,
            "noncritical_truth_atoms_correct": 0,
            "noncritical_source_edges_correct": 0,
            "relation_judgments_correct": 0,
            "critical_harm_complete": 0,
        },
        "correction_resolution": "NOT_REQUIRED",
        "corrected_truth_packet_sha256_by_task": None,
        "all_arm_rescore_manifest_sha256": None,
        "critical_errors": 0,
        "critical_harm_misses": 0,
        "noncritical_accuracy": 1.0,
        "gate_reasons": [],
    }
    receipt["content_sha256"] = sha256_hex(canonical_json(receipt))
    return receipt


def test_verdict_inputs_keep_all_offered_and_separate_fallback_policy():
    records = []
    for i, task in enumerate(("t1", "t2")):
        p0 = _itt_row(0.8)
        p0["work_summary"] = {
            "telemetry_complete": True,
            "overlap_valid": True,
            "service_seconds": 10.0,
        }
        h = _itt_row(0.9, fallback=(i == 0), unavailable=(i == 1))
        records.append(
            {
                "run_id": "r",
                "phase_id": "p",
                "block_id": f"B{i}",
                "task_id": task,
                "per_arm": {"P0:0": p0, "H02:0": h},
            }
        )
    body = build_verdict_inputs(
        records,
        cluster_by_task={"t1": "c1", "t2": "c2"},
        n_boot=100,
        scope_receipt=_scope_receipt(records),
    )
    h = body["arms"]["H02"]
    assert h["all_offered_pairs"] == 2
    assert h["fallback_count"] == 1
    assert h["judge_unavailable_count"] == 1
    # Strict observed has the fallback as a failure; judge-unavailable stays explicitly missing.
    assert h["quality_effects"]["weighted_required_atom_recall"]["strict_n_pairs"] == 1
    # Worst-case sensitivity restores the unavailable assignment to the denominator.
    assert h["quality_effects"]["weighted_required_atom_recall"]["worst_case_n_pairs"] == 2
    assert h["work_pairs_observed"] == 2
    assert np.isfinite(h["work_saving"]["lower"])
    assert np.isfinite(h["work_saving"]["upper"])
    assert h["work_saving"]["lower_bound"]["interval_type"] == (
        "one_sided_95_percent_lower_confidence_bound"
    )
    assert h["work_saving"]["upper_bound"]["interval_type"] == (
        "one_sided_95_percent_upper_confidence_bound"
    )
    assert h["required_direct_estimands"] == [
        "h_candidate_coverage",
        "h_total_published_prechunk_recall",
    ]
    assert (
        h["selector_estimands"]["h_total_published_prechunk_recall"]["all_offered_worst_case_n"]
        == 2
    )


def test_invalid_selector_id_hard_gate_survives_high_quality_p0_fallback():
    records = []
    for index, task in enumerate(("t1", "t2")):
        treatment = _itt_row(1.0 if index == 0 else 0.9, fallback=(index == 0))
        if index == 0:
            summary = treatment["direct_node_metrics"]["by_node"]["H"]["selector_normalization"]
            summary.update(
                {
                    "strict_valid_count": 0,
                    "rejected_attempt_count": 1,
                    "invalid_id_count": 1,
                    "component_failure_count": 1,
                    "no_repair_adverse": True,
                }
            )
        records.append(
            {
                "run_id": "r",
                "phase_id": "p",
                "block_id": f"B{index}",
                "task_id": task,
                "per_arm": {"P0:0": _itt_row(0.8), "H02:0": treatment},
            }
        )
    body = build_verdict_inputs(
        records,
        cluster_by_task={"t1": "c1", "t2": "c2"},
        n_boot=100,
        scope_receipt=_scope_receipt(records),
    )
    arm = body["arms"]["H02"]
    assert arm["selector_output_validity"]["invalid_id_hard_gate"]["status"] == "FAIL"
    assert arm["selector_output_validity"]["invalid_id_count"] == 1
    no_repair = arm["quality_effects"]["weighted_required_atom_recall"]["no_repair_adverse"]
    assert no_repair["point"] == pytest.approx(-0.35)


def test_duplicate_changes_repair_sensitivity_but_not_invalid_id_gate():
    records = []
    for index, task in enumerate(("t1", "t2")):
        treatment = _itt_row(0.9)
        if index == 0:
            summary = treatment["direct_node_metrics"]["by_node"]["H"]["selector_normalization"]
            summary.update(
                {
                    "strict_valid_count": 0,
                    "repaired_attempt_count": 1,
                    "no_repair_adverse": True,
                }
            )
        records.append(
            {
                "run_id": "r",
                "phase_id": "p",
                "block_id": f"B{index}",
                "task_id": task,
                "per_arm": {"P0:0": _itt_row(0.8), "H02:0": treatment},
            }
        )
    body = build_verdict_inputs(
        records,
        cluster_by_task={"t1": "c1", "t2": "c2"},
        n_boot=100,
        scope_receipt=_scope_receipt(records),
    )
    validity = body["arms"]["H02"]["selector_output_validity"]
    assert validity["invalid_id_hard_gate"]["status"] == "PASS"
    assert validity["invalid_id_count"] == 0
    assert validity["repair_rate_ucb"]["ci"]["point"] == pytest.approx(0.5)
    assert validity["strict_valid_rate_lcb"]["ci"]["point"] == pytest.approx(0.5)


def test_missing_normalization_summary_is_not_a_zero_invalid_id_pass():
    records = []
    for index, task in enumerate(("t1", "t2")):
        treatment = _itt_row(0.9)
        if index == 0:
            del treatment["direct_node_metrics"]["by_node"]["H"]["selector_normalization"]
        records.append(
            {
                "run_id": "r",
                "phase_id": "p",
                "block_id": f"B{index}",
                "task_id": task,
                "per_arm": {"P0:0": _itt_row(0.8), "H02:0": treatment},
            }
        )
    body = build_verdict_inputs(
        records,
        cluster_by_task={"t1": "c1", "t2": "c2"},
        n_boot=100,
        scope_receipt=_scope_receipt(records),
    )
    validity = body["arms"]["H02"]["selector_output_validity"]
    assert validity["invalid_id_count"] == 0
    assert validity["invalid_id_hard_gate"]["status"] == (
        "NOT_ESTABLISHED_INCOMPLETE_NORMALIZATION_TRACE"
    )
    assert validity["no_repair_adverse_row_count"] == 1


def test_known_direct_denominator_with_missing_value_is_adverse_not_deleted():
    records = []
    for index, task in enumerate(("t1", "t2")):
        treatment = _itt_row(0.9)
        h_metrics = treatment["direct_node_metrics"]["by_node"]["H"]
        h_metrics["eligible_denominators"] = {"candidate_coverage": 1}
        if index == 0:
            h_metrics["candidate_coverage"] = None
        records.append(
            {
                "run_id": "r",
                "phase_id": "p",
                "block_id": f"B{index}",
                "task_id": task,
                "per_arm": {"P0:0": _itt_row(0.8), "H02:0": treatment},
            }
        )
    body = build_verdict_inputs(
        records,
        cluster_by_task={"t1": "c1", "t2": "c2"},
        n_boot=100,
        scope_receipt=_scope_receipt(records),
    )
    direct = body["arms"]["H02"]["selector_estimands"]["h_candidate_coverage"]
    assert direct["eligible_denominator_rows"] == 2
    assert direct["all_offered_worst_case_n"] == 2
    assert direct["adverse_missing_value_rows"] == 1
    assert direct["all_offered_eligible_rows_covered"] is True
    assert direct["all_offered_worst_case"]["point"] == pytest.approx(0.45)


def test_joint_h_c_estimands_read_only_their_node_scoped_values():
    records = []
    for index, task in enumerate(("t1", "t2")):
        treatment = _itt_row(0.9)
        treatment["direct_node_metrics"].update(
            {
                # Deliberately conflicting legacy aliases: neither H nor C may read these.
                "candidate_coverage": 0.99,
                "total_published_prechunk_recall": 0.99,
            }
        )
        treatment["direct_node_metrics"]["by_node"]["H"].update(
            {
                "candidate_coverage": 0.8,
                "total_published_prechunk_recall": 0.7,
            }
        )
        treatment["direct_node_metrics"]["by_node"]["C"] = {
            "applicable": True,
            "status": "OK",
            "selector_guards_complete": True,
            "candidate_coverage": 0.2,
            "total_published_prechunk_recall": 0.1,
        }
        records.append(
            {
                "run_id": "r",
                "phase_id": "p",
                "block_id": f"B{index}",
                "task_id": task,
                "per_arm": {"P0:0": _itt_row(0.8), "H02+C01:0": treatment},
            }
        )
    body = build_verdict_inputs(
        records,
        cluster_by_task={"t1": "c1", "t2": "c2"},
        n_boot=100,
        scope_receipt=_scope_receipt(records),
    )
    arm = body["arms"]["H02+C01"]
    assert arm["required_direct_estimands"] == [
        "h_candidate_coverage",
        "h_total_published_prechunk_recall",
        "c_candidate_coverage",
        "c_total_published_prechunk_recall",
    ]
    assert arm["selector_estimands"]["h_candidate_coverage"]["strict_observed"][
        "point"
    ] == pytest.approx(0.8)
    assert arm["selector_estimands"]["c_candidate_coverage"]["strict_observed"][
        "point"
    ] == pytest.approx(0.2)
    assert arm["selector_estimands"]["h_total_published_prechunk_recall"]["strict_observed"][
        "point"
    ] == pytest.approx(0.7)
    assert arm["selector_estimands"]["c_total_published_prechunk_recall"]["strict_observed"][
        "point"
    ] == pytest.approx(0.1)


def test_missing_c_trace_cannot_hide_behind_a_healthy_h_flat_status():
    records = []
    for index, task in enumerate(("t1", "t2")):
        treatment = _itt_row(0.9)
        treatment["direct_node_metrics"].update(
            {
                "status": "OK",
                "selector_guards_complete": True,
            }
        )
        treatment["direct_node_metrics"]["by_node"]["C"] = {
            "applicable": True,
            "status": "DIRECT_TRACE_UNAVAILABLE",
            "selector_guards_complete": False,
        }
        records.append(
            {
                "run_id": "r",
                "phase_id": "p",
                "block_id": f"B{index}",
                "task_id": task,
                "per_arm": {"P0:0": _itt_row(0.8), "H02+C01:0": treatment},
            }
        )
    body = build_verdict_inputs(
        records,
        cluster_by_task={"t1": "c1", "t2": "c2"},
        n_boot=100,
        scope_receipt=_scope_receipt(records),
    )
    arm = body["arms"]["H02+C01"]
    assert arm["direct_trace_unavailable_count"] == 2
    assert arm["selector_guard_unavailable_count"] == 2
    assert arm["structural_pass"] is False
    assert arm["machine_estimands_ready"] is False


def test_terminal_failure_and_fallback_risks_are_separate_all_offered_estimands():
    records = []
    for index, task in enumerate(("t1", "t2")):
        treatment = _itt_row(0.9, fallback=(index == 1))
        if index == 0:
            treatment["assignment_state"] = "FAILED_FINAL"
        records.append(
            {
                "run_id": "r",
                "phase_id": "p",
                "block_id": f"B{index}",
                "task_id": task,
                "per_arm": {"P0:0": _itt_row(0.8), "H02:0": treatment},
            }
        )
    body = build_verdict_inputs(
        records,
        cluster_by_task={"t1": "c1", "t2": "c2"},
        n_boot=100,
        scope_receipt=_scope_receipt(records),
    )
    arm = body["arms"]["H02"]
    terminal = arm["terminal_failure_risk_difference"]
    fallback = arm["fallback_risk_difference"]
    assert terminal["status"] == "OK"
    assert terminal["ci"]["point"] == pytest.approx(50.0)
    assert terminal["ci"]["unit"] == "percentage_points"
    assert fallback["status"] == "OK"
    assert fallback["ci"]["point"] == pytest.approx(50.0)


def test_unknown_assignment_state_cannot_be_read_as_a_successful_terminal_outcome():
    records = []
    for index, task in enumerate(("t1", "t2")):
        treatment = _itt_row(0.9)
        if index == 0:
            treatment["assignment_state"] = "MYSTERY"
        records.append(
            {
                "run_id": "r",
                "phase_id": "p",
                "block_id": f"B{index}",
                "task_id": task,
                "per_arm": {"P0:0": _itt_row(0.8), "H02:0": treatment},
            }
        )
    body = build_verdict_inputs(
        records,
        cluster_by_task={"t1": "c1", "t2": "c2"},
        n_boot=100,
        scope_receipt=_scope_receipt(records),
    )
    arm = body["arms"]["H02"]
    assert arm["terminal_failure_risk_difference"]["status"] == "NOT_ESTIMABLE"
    assert arm["terminal_failure_risk_difference"]["missing_blocks"] == ["B0"]
    assert arm["machine_estimands_ready"] is False


def test_quality_effect_reduces_replicates_within_task_before_cluster_bootstrap():
    records = []
    assignments = [
        ("t1", "c1", 0.8, 1.0),
        ("t1", "c1", 0.8, 1.0),
        ("t1", "c1", 0.8, 1.0),
        ("t2", "c2", 0.8, 0.6),
    ]
    for index, (task, _cluster, p0_value, treatment_value) in enumerate(assignments):
        records.append(
            {
                "run_id": "r",
                "phase_id": "p",
                "block_id": f"B{index}",
                "task_id": task,
                "per_arm": {
                    "P0:0": _itt_row(p0_value),
                    "H02:0": _itt_row(treatment_value),
                },
            }
        )
    body = build_verdict_inputs(
        records,
        cluster_by_task={"t1": "c1", "t2": "c2"},
        n_boot=100,
        scope_receipt=_scope_receipt(records),
    )
    effect = body["arms"]["H02"]["quality_effects"]["weighted_required_atom_recall"][
        "fallback_assisted"
    ]
    # Task t1 contributes +0.2 once and t2 contributes -0.2 once.  Treating the three t1
    # replicates as independent rows would incorrectly report +0.1.
    assert effect["point"] == pytest.approx(0.0)


def test_c_direct_metrics_report_distinct_macro_and_pooled_micro_estimands():
    records = []
    for index, (task, numerator, denominator) in enumerate((("t1", 1, 1), ("t2", 0, 9))):
        treatment = _itt_row(0.9)
        ratio = numerator / denominator
        treatment["direct_node_metrics"]["by_node"]["H"] = {"applicable": False}
        treatment["direct_node_metrics"]["by_node"]["C"] = {
            "applicable": True,
            "status": "OK",
            "selector_guards_complete": True,
            "candidate_coverage": ratio,
            "total_published_prechunk_recall": ratio,
            "micro_counts": {
                "candidate_coverage": {
                    "numerator": numerator,
                    "denominator": denominator,
                },
                "total_published_prechunk_recall": {
                    "numerator": numerator,
                    "denominator": denominator,
                },
            },
        }
        records.append(
            {
                "run_id": "r",
                "phase_id": "p",
                "block_id": f"B{index}",
                "task_id": task,
                "per_arm": {"P0:0": _itt_row(0.8), "C01:0": treatment},
            }
        )
    body = build_verdict_inputs(
        records,
        cluster_by_task={"t1": "c1", "t2": "c2"},
        n_boot=100,
        scope_receipt=_scope_receipt(records),
    )
    direct = body["arms"]["C01"]["selector_estimands"]["c_candidate_coverage"]
    assert direct["strict_observed"]["point"] == pytest.approx(0.5)
    assert direct["strict_observed_micro"]["point"] == pytest.approx(0.1)
    assert direct["missing_micro_denominator_count"] == 0


def test_verdict_effect_bounds_do_not_cancel_when_both_judges_are_missing():
    records = []
    for i, task in enumerate(("t1", "t2")):
        records.append(
            {
                "run_id": "r",
                "phase_id": "p",
                "block_id": f"B{i}",
                "task_id": task,
                "per_arm": {
                    "P0:0": _itt_row(0.0, unavailable=True),
                    "H02:0": _itt_row(0.0, unavailable=True),
                },
            }
        )
    body = build_verdict_inputs(
        records,
        cluster_by_task={"t1": "c1", "t2": "c2"},
        n_boot=100,
        scope_receipt=_scope_receipt(records),
    )
    effect = body["arms"]["H02"]["quality_effects"]["weighted_required_atom_recall"]
    assert effect["worst_case"]["point"] == -1.0
    assert effect["best_case"]["point"] == 1.0


def test_verdict_refuses_whichever_score_files_happen_to_exist():
    records = [
        {
            "run_id": "r",
            "phase_id": "p",
            "block_id": "B0",
            "task_id": "t1",
            "per_arm": {"P0:0": _itt_row(0.8), "H02:0": _itt_row(0.9)},
        }
    ]
    with pytest.raises(ValueError, match="all-offered evaluation scope"):
        build_verdict_inputs(records, cluster_by_task={"t1": "c1"})


def test_cross_epoch_block_stays_in_itt_but_fails_structural_readiness():
    records = []
    for i, task in enumerate(("t1", "t2")):
        p0 = _itt_row(0.8)
        p0["work_summary"] = {
            "telemetry_complete": True,
            "overlap_valid": True,
            "service_seconds": 10.0,
        }
        records.append(
            {
                "run_id": "r",
                "phase_id": "p",
                "block_id": f"B{i}",
                "task_id": task,
                "per_arm": {"P0:0": p0, "H02:0": _itt_row(0.9)},
            }
        )
    receipt = _scope_receipt(records, invalid_blocks={"B0"})
    body = build_verdict_inputs(
        records, cluster_by_task={"t1": "c1", "t2": "c2"}, n_boot=100, scope_receipt=receipt
    )
    arm = body["arms"]["H02"]
    assert arm["all_offered_pairs"] == 2
    assert arm["paired_invalid_block_count"] == 1
    assert arm["paired_invalid_reasons"] == {"SPANS_ENGINE_EPOCHS": 1}
    assert arm["structural_pass"] is False
    assert arm["machine_estimands_ready"] is False
    assert "PAIRED_BLOCK_INVALID" in arm["verdict_blockers"]


@pytest.mark.parametrize(
    "summary",
    [
        {"service_seconds": 1.0},
        {
            "telemetry_complete": False,
            "overlap_valid": True,
            "service_seconds": 1.0,
        },
        {
            "telemetry_complete": True,
            "overlap_valid": False,
            "service_seconds": 1.0,
        },
        {
            "telemetry_complete": True,
            "overlap_valid": True,
            "service_seconds": float("nan"),
        },
        {
            "telemetry_complete": True,
            "overlap_valid": True,
            "service_seconds": float("inf"),
        },
        {
            "telemetry_complete": True,
            "overlap_valid": True,
            "service_seconds": 0.0,
        },
    ],
)
def test_itt_work_rejects_incomplete_overlapping_or_nonfinite_telemetry(summary):
    assert _work({"work_summary": summary}) is None


def test_bad_work_telemetry_cannot_produce_a_machine_ready_saving():
    records = []
    for index, task in enumerate(("t1", "t2")):
        treatment = _itt_row(0.9)
        if index == 0:
            treatment["work_summary"]["overlap_valid"] = False
        records.append(
            {
                "run_id": "r",
                "phase_id": "p",
                "block_id": f"B{index}",
                "task_id": task,
                "per_arm": {"P0:0": _itt_row(0.8), "H02:0": treatment},
            }
        )
    body = build_verdict_inputs(
        records,
        cluster_by_task={"t1": "c1", "t2": "c2"},
        n_boot=100,
        scope_receipt=_scope_receipt(records),
    )
    arm = body["arms"]["H02"]
    assert arm["work_pairs_observed"] == 1
    assert arm["work_pairs_missing"] == 1
    assert arm["work_saving"] is None
    assert arm["machine_estimands_ready"] is False


@pytest.mark.parametrize(
    "policies, error",
    [
        (("a" * 64, "b" * 64), "uniform judge_policy_sha256"),
        (("", ""), "non-empty lowercase SHA-256 judge_policy_sha256"),
        (("not-a-sha", "not-a-sha"), "non-empty lowercase SHA-256 judge_policy_sha256"),
    ],
)
def test_itt_rejects_missing_invalid_or_mixed_judge_policy_hashes(policies, error):
    records = []
    for index, (task, policy) in enumerate(zip(("t1", "t2"), policies, strict=True)):
        records.append(
            {
                "run_id": "r",
                "phase_id": "p",
                "block_id": f"B{index}",
                "task_id": task,
                "judge_policy_sha256": policy,
                "per_arm": {"P0:0": _itt_row(0.8), "H02:0": _itt_row(0.9)},
            }
        )
    receipt = _scope_receipt(records)
    with pytest.raises(ValueError, match=error):
        build_verdict_inputs(
            records,
            cluster_by_task={"t1": "c1", "t2": "c2"},
            n_boot=100,
            scope_receipt=receipt,
        )


def test_quality_decision_interval_is_exact_one_sided_lower_bound():
    values = [-0.30, -0.10, 0.00, 0.20, 0.40, 0.55]
    clusters = [f"c{i}" for i in range(len(values))]
    seed = 913
    n_boot = 2000
    result = _difference_ci(values, clusters, seed=seed, n_boot=n_boot)
    expected = cluster_bootstrap_ci(
        values, clusters, np.mean, seed=seed, n_boot=n_boot, side="lower"
    )
    descriptive_two_sided = cluster_bootstrap_ci(
        values, clusters, np.mean, seed=seed, n_boot=n_boot, side="two"
    )

    assert result["lower"] == pytest.approx(expected.lower)
    assert result["upper"] is None
    assert result["lower"] != pytest.approx(descriptive_two_sided.lower)
    assert result["interval_type"] == ("one_sided_95_percent_lower_confidence_bound")
    assert result["bound_direction"] == "lower"
    assert result["effect_orientation"] == "positive_favors_treatment"


def test_critical_harm_is_reoriented_then_uses_the_quality_lower_bound():
    records = []
    for index, task in enumerate(("t1", "t2")):
        p0 = _itt_row(0.8)
        treatment = _itt_row(0.9)
        for mode in ("strict", "fallback_assisted", "worst_case", "best_case"):
            p0["quality_views"][mode]["critical_harm"] = 0.20
            treatment["quality_views"][mode]["critical_harm"] = 0.10
        records.append(
            {
                "run_id": "r",
                "phase_id": "p",
                "block_id": f"B{index}",
                "task_id": task,
                "per_arm": {"P0:0": p0, "H02:0": treatment},
            }
        )
    body = build_verdict_inputs(
        records,
        cluster_by_task={"t1": "c1", "t2": "c2"},
        n_boot=200,
        scope_receipt=_scope_receipt(records),
    )
    harm = body["arms"]["H02"]["quality_effects"]["critical_harm"]["strict"]
    assert harm["point"] == pytest.approx(0.10)
    assert harm["upper"] is None
    assert harm["interval_type"] == ("one_sided_95_percent_lower_confidence_bound")


def test_adverse_risk_decision_interval_is_exact_one_sided_upper_bound():
    observations = []
    diffs = []
    clusters = []
    for index in range(7):
        failed = index in {0, 1, 2}
        observations.append(
            {
                "block_id": f"B{index}",
                "task_id": f"t{index}",
                "cluster_id": f"c{index}",
                "p0": {"assignment_state": "COMMITTED"},
                "p1": {
                    "assignment_state": "FAILED_FINAL" if failed else "COMMITTED",
                },
            }
        )
        diffs.append(1.0 if failed else 0.0)
        clusters.append(f"c{index}")
    seed = 417
    n_boot = 2000
    result = _paired_risk_difference(
        observations,
        field="terminal_failure",
        seed=seed,
        n_boot=n_boot,
    )
    expected = cluster_bootstrap_ci(
        diffs, clusters, np.mean, seed=seed, n_boot=n_boot, side="upper"
    )
    descriptive_two_sided = cluster_bootstrap_ci(
        diffs, clusters, np.mean, seed=seed, n_boot=n_boot, side="two"
    )

    ci = result["ci"]
    assert ci["lower"] is None
    assert ci["upper"] == pytest.approx(expected.upper * 100.0)
    assert ci["upper"] != pytest.approx(descriptive_two_sided.upper * 100.0)
    assert ci["interval_type"] == ("one_sided_95_percent_upper_confidence_bound")
    assert ci["bound_direction"] == "upper"
    assert ci["effect_orientation"] == "positive_is_increased_adverse_risk"


def test_score_declared_audited_status_cannot_make_a_verdict_ready():
    records = []
    for index, task in enumerate(("t1", "t2")):
        records.append(
            {
                "run_id": "r",
                "phase_id": "p",
                "block_id": f"B{index}",
                "task_id": task,
                "truth_packet_sha256": f"{index + 1}" * 64,
                "truth_verifier_status": "AUDITED",
                "per_arm": {"P0:0": _itt_row(0.8), "H02:0": _itt_row(0.9)},
            }
        )
    scope = _scope_receipt(records)
    body = build_verdict_inputs(
        records,
        cluster_by_task={"t1": "c1", "t2": "c2"},
        n_boot=100,
        scope_receipt=scope,
    )
    assert body["truth_audited"] is False
    assert body["human_audit_receipt_sha256"] is None
    assert body["score_declared_truth_verifier_statuses"] == ["AUDITED"]
    assert body["score_truth_verifier_status_is_decision_input"] is False


def test_only_exact_truth_and_score_bound_audit_receipt_marks_truth_audited():
    records = []
    for index, task in enumerate(("t1", "t2")):
        records.append(
            {
                "run_id": "r",
                "phase_id": "p",
                "block_id": f"B{index}",
                "task_id": task,
                "truth_packet_sha256": f"{index + 1}" * 64,
                "truth_verifier_status": "PENDING",
                "per_arm": {"P0:0": _itt_row(0.8), "H02:0": _itt_row(0.9)},
            }
        )
    scope = _scope_receipt(records)
    receipt = _human_audit_receipt(records, scope)
    body = build_verdict_inputs(
        records,
        cluster_by_task={"t1": "c1", "t2": "c2"},
        n_boot=100,
        scope_receipt=scope,
        human_audit_receipt=receipt,
    )
    assert body["truth_audited"] is True
    assert body["human_audit_receipt_sha256"] == receipt["content_sha256"]

    swapped = dict(receipt)
    swapped["score_content_sha256_by_block"] = {
        **receipt["score_content_sha256_by_block"],
        "B0": "0" * 64,
    }
    swapped["content_sha256"] = sha256_hex(
        canonical_json({key: value for key, value in swapped.items() if key != "content_sha256"})
    )
    with pytest.raises(ValueError, match="different score bytes"):
        build_verdict_inputs(
            records,
            cluster_by_task={"t1": "c1", "t2": "c2"},
            n_boot=100,
            scope_receipt=scope,
            human_audit_receipt=swapped,
        )

    corrected = dict(receipt)
    corrected["correction_count"] = 1
    corrected["items_requiring_correction"] = 1
    corrected["content_sha256"] = sha256_hex(
        canonical_json({key: value for key, value in corrected.items() if key != "content_sha256"})
    )
    with pytest.raises(ValueError, match="correction-free v2"):
        build_verdict_inputs(
            records,
            cluster_by_task={"t1": "c1", "t2": "c2"},
            n_boot=100,
            scope_receipt=scope,
            human_audit_receipt=corrected,
        )


def test_selector_efficiency_uses_complete_task_mean_token_traces_but_is_secondary():
    records = []
    for index, task in enumerate(("t1", "t2")):
        records.append(
            {
                "run_id": "r",
                "phase_id": "p",
                "block_id": f"B{index}",
                "task_id": task,
                "truth_packet_sha256": f"{index + 1}" * 64,
                "per_arm": {"P0:0": _itt_row(0.8), "H02:0": _itt_row(0.9)},
            }
        )
    scope = _scope_receipt(records)
    body = build_verdict_inputs(
        records,
        cluster_by_task={"t1": "c1", "t2": "c2"},
        n_boot=100,
        scope_receipt=scope,
    )
    efficiency = body["arms"]["H02"]["selector_efficiency"]["H"]
    assert efficiency["status"] == "ESTIMABLE"
    assert efficiency["metrics"]["selected_token_precision"]["ci"]["point"] == pytest.approx(0.75)
    assert efficiency["published_rendered_tokens"]["total"] == 240
    assert efficiency["decision_use"] == "SECONDARY_DESCRIPTIVE_NOT_PRIMARY_CAUSAL_UTILITY"
    harm = body["arms"]["H02"]["critical_harm_absolute"]["all_offered_adverse"]
    assert harm["status"] == "ESTIMABLE"
    assert harm["ci"]["point"] == pytest.approx(0.0)
    assert harm["n_pairs"] == 2

    records[0]["per_arm"]["H02:0"]["direct_node_metrics"]["by_node"]["H"][
        "token_trace_complete"
    ] = False
    records[0]["per_arm"]["H02:0"]["direct_node_metrics"]["by_node"]["H"][
        "selected_token_precision"
    ] = None
    for record in records:
        record.pop("content_sha256", None)
    scope = _scope_receipt(records)
    incomplete = build_verdict_inputs(
        records,
        cluster_by_task={"t1": "c1", "t2": "c2"},
        n_boot=100,
        scope_receipt=scope,
    )
    assert (
        incomplete["arms"]["H02"]["selector_efficiency"]["H"]["status"]
        == "NOT_ESTABLISHED_INCOMPLETE_TOKEN_TRACE"
    )
    assert incomplete["arms"]["H02"]["all_offered_pairs"] == 2


def test_absolute_selector_gates_expose_all_offered_lcb_and_true_denominators():
    records = []
    for index, task in enumerate(("t1", "t2")):
        treatment = _itt_row(0.9)
        h = treatment["direct_node_metrics"]["by_node"]["H"]
        h.update(
            {
                "selector_conditional_recall": 0.95 - index * 0.05,
                "weighted_evidence_recall": 0.92 - index * 0.02,
                "critical_truth_recall": 1.0 if index == 0 else 0.0,
                "contradiction_pair_recall": None,
                "grounded_negative_atom_recall": None,
                "negative_query_trace_recall": None,
                "unresolved_gap_recall": None,
                "negative_gap_recall": None,
                "eligible_denominators": {
                    "candidate_coverage": 4,
                    "selector_conditional_recall": 3,
                    "weighted_evidence_recall": 3.5,
                    "total_published_prechunk_recall": 4,
                    "critical_truth_recall": 1,
                    "contradiction_pair_recall": 0,
                    "grounded_negative_atom_recall": 0,
                    "negative_query_trace_recall": 0,
                    "unresolved_gap_recall": 0,
                    "negative_gap_recall": 0,
                },
            }
        )
        records.append(
            {
                "run_id": "r",
                "phase_id": "p",
                "block_id": f"B{index}",
                "task_id": task,
                "per_arm": {"P0:0": _itt_row(0.8), "H02:0": treatment},
            }
        )
    body = build_verdict_inputs(
        records,
        cluster_by_task={"t1": "c1", "t2": "c2"},
        n_boot=200,
        scope_receipt=_scope_receipt(records),
    )
    arm = body["arms"]["H02"]
    assert "h_weighted_evidence_recall" in (arm["required_selector_gate_estimands"])
    assert "h_selector_conditional_recall" not in (arm["required_selector_gate_estimands"])
    assert "h_negative_gap_recall" in arm["required_selector_gate_estimands"]
    assert "h_grounded_negative_atom_recall" not in (arm["required_selector_gate_estimands"])
    selector = arm["selector_estimands"]["h_selector_conditional_recall"]
    assert selector["all_offered_gate_status"] == "ESTIMABLE"
    assert selector["eligible_denominator_rows"] == 2
    assert selector["eligible_denominator_opportunities"] == 6
    assert selector["all_offered_lcb"] == (selector["all_offered_worst_case"]["lower"])
    weighted = arm["selector_estimands"]["h_weighted_evidence_recall"]
    assert weighted["eligible_denominator_rows"] == 2
    assert weighted["eligible_denominator_opportunities"] == pytest.approx(7.0)
    assert weighted["all_offered_gate_status"] == "ESTIMABLE"
    critical = arm["selector_estimands"]["h_critical_truth_recall"]
    assert critical["eligible_denominator_opportunities"] == 2
    assert critical["all_offered_lcb"] == pytest.approx(0.0)
    contradiction = arm["selector_estimands"]["h_contradiction_pair_recall"]
    assert contradiction["all_offered_gate_status"] == ("NOT_APPLICABLE_NO_ELIGIBLE_DENOMINATOR")
    assert contradiction["missing_direct_denominator_count"] == 0


def test_fallback_is_zero_in_all_offered_weighted_and_negative_gap_gates():
    records = []
    for index, task in enumerate(("t1", "t2")):
        treatment = _itt_row(0.9, fallback=True)
        h = treatment["direct_node_metrics"]["by_node"]["H"]
        h.update(
            {
                "weighted_evidence_recall": 1.0,
                "critical_truth_recall": 1.0,
                "contradiction_pair_recall": None,
                "negative_gap_recall": 1.0,
                "eligible_denominators": {
                    "candidate_coverage": 1,
                    "total_published_prechunk_recall": 1,
                    "weighted_evidence_recall": 7.5,
                    "critical_truth_recall": 1,
                    "contradiction_pair_recall": 0,
                    "negative_gap_recall": 2,
                },
            }
        )
        records.append(
            {
                "run_id": "r",
                "phase_id": "p",
                "block_id": f"B{index}",
                "task_id": task,
                "per_arm": {"P0:0": _itt_row(0.8), "H02:0": treatment},
            }
        )

    body = build_verdict_inputs(
        records,
        cluster_by_task={"t1": "c1", "t2": "c2"},
        n_boot=100,
        scope_receipt=_scope_receipt(records),
    )
    gates = body["arms"]["H02"]["selector_estimands"]
    for metric in ("h_weighted_evidence_recall", "h_negative_gap_recall"):
        assert gates[metric]["strict_observed"] is None
        assert gates[metric]["all_offered_worst_case"]["point"] == pytest.approx(0.0)
        assert gates[metric]["all_offered_lcb"] == pytest.approx(0.0)
