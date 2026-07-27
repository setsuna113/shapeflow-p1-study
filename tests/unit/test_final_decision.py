"""The final decision answers benefit/size/envelope without exceeding the evidence."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from shapeflow_p1.analysis.decision import Verdict
from shapeflow_p1.analysis.finalize import (
    build_final_decision,
    write_final_decision,
)
from shapeflow_p1.analysis.matched import resolve_arm_semantics
from shapeflow_p1.analysis.report import (
    AUDITED_STATUS,
    PROVISIONAL_STATUS,
    render_json,
    render_markdown,
)
from shapeflow_p1.canonical import canonical_json
from shapeflow_p1.config import config_sha, load_config
from shapeflow_p1.hashing import sha256_hex

REPO = Path(__file__).resolve().parents[2]


def _seal(body: dict) -> dict:
    body["content_sha256"] = sha256_hex(canonical_json(body))
    return body


def _reseal(body: dict) -> dict:
    body.pop("content_sha256", None)
    return _seal(body)


def _configs() -> tuple[dict, dict, dict]:
    decision, _ = load_config(REPO / "configs" / "decision.yaml")
    week1, _ = load_config(REPO / "configs" / "week1.yaml")
    variants, _ = load_config(REPO / "configs" / "variants.yaml")
    return decision, week1, variants


def _selector_estimate() -> dict:
    return {
        "all_offered_worst_case": {
            "point": 0.98,
            "lower": 0.96,
            "upper": None,
        },
        "all_offered_lcb": 0.96,
        "missing_direct_denominator_count": 0,
        "eligible_denominator_rows": 16,
        "eligible_denominator_opportunities": 32,
        "not_applicable_rows": 0,
        "all_offered_gate_status": "ESTIMABLE",
    }


def _not_applicable_selector() -> dict:
    return {
        "all_offered_worst_case": None,
        "all_offered_lcb": None,
        "missing_direct_denominator_count": 0,
        "eligible_denominator_rows": 0,
        "eligible_denominator_opportunities": 0,
        "not_applicable_rows": 16,
        "all_offered_gate_status": "NOT_APPLICABLE_NO_ELIGIBLE_DENOMINATOR",
    }


def _arm_row(
    *prefixes: str,
    work_lower: float = 0.20,
    work_upper: float = 0.30,
    machine_ready: bool = True,
) -> dict:
    quality = {}
    for metric in (
        "weighted_required_atom_recall",
        "grounded_claim_precision",
        "citation_correctness",
        "citation_association",
        "required_facet_coverage",
        "qualified_report",
        "critical_harm",
    ):
        quality[metric] = {
            "strict": {"point": 0.0, "lower": -0.01, "upper": None},
            "strict_n_pairs": 16,
            "strict_n_tasks": 16,
            "strict_missing_pairs": 0,
            "no_repair_adverse": {
                "point": 0.0,
                "lower": -0.01,
                "upper": None,
                "interval_type": "one_sided_95_percent_lower_confidence_bound",
            },
            "no_repair_adverse_n_pairs": 16,
            "no_repair_adverse_n_tasks": 16,
            "no_repair_adverse_dirty_pairs": 0,
            "no_repair_adverse_judge_missing_pairs": 0,
            "no_repair_adverse_not_applicable_pairs": 0,
        }
    selectors = {}
    selector_efficiency = {}
    for prefix in prefixes:
        selectors[f"{prefix}_selector_conditional_recall"] = _selector_estimate()
        selectors[f"{prefix}_weighted_evidence_recall"] = _selector_estimate()
        for suffix in (
            "critical_truth_recall",
            "contradiction_pair_recall",
            "negative_gap_recall",
        ):
            selectors[f"{prefix}_{suffix}"] = _not_applicable_selector()
        selector_efficiency[prefix.upper()] = {
            "status": "ESTIMABLE",
            "node": prefix.upper(),
            "all_offered_applicable_rows": 16,
            "token_trace_complete_rows": 16,
            "token_trace_incomplete_rows": 0,
            "malformed_metric_rows": 0,
            "decision_use": "SECONDARY_DESCRIPTIVE_NOT_PRIMARY_CAUSAL_UTILITY",
            "metrics": {
                field: {
                    "status": "ESTIMABLE",
                    "ci": {
                        "point": point,
                        "lower": point - 0.05,
                        "upper": point + 0.05,
                        "interval_type": "two_sided_95_percent_cluster_bootstrap",
                    },
                    "tasks": 16,
                }
                for field, point in (
                    ("selected_token_precision", 0.70),
                    ("weighted_truth_per_100_rendered_tokens", 2.5),
                    ("materialization_ratio", 1.2),
                )
            },
            "published_rendered_tokens": {
                "status": "OBSERVED_ROW_TOTAL",
                "total": 800,
            },
            "offered_evidence_tokens": {
                "status": "OBSERVED_ROW_TOTAL",
                "total": 640,
            },
        }
    return {
        "all_offered_pairs": 16,
        "independent_clusters": 8,
        "work_saving": {
            "point": 0.25,
            "lower": work_lower,
            "upper": work_upper,
        },
        "work_pairs_observed": 16,
        "work_pairs_missing": 0,
        "quality_effects": quality,
        "selector_estimands": selectors,
        "selector_efficiency": selector_efficiency,
        "selector_output_validity": {
            "selector_attempt_count": 32,
            "strict_valid_count": 32,
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
            "missing_node_summary_count": 0,
            "no_repair_adverse_row_count": 0,
            "all_offered_rows": 16,
            "strict_valid_rate_lcb": {
                "status": "ESTIMABLE",
                "ci": {
                    "point": 1.0,
                    "lower": 1.0,
                    "upper": None,
                    "interval_type": "one_sided_95_percent_lower_confidence_bound",
                },
                "tasks": 16,
                "attempts": 32,
                "incomplete_rows": 0,
                "no_attempt_rows": 0,
            },
            "repair_rate_ucb": {
                "status": "ESTIMABLE",
                "ci": {
                    "point": 0.0,
                    "lower": None,
                    "upper": 0.0,
                    "interval_type": "one_sided_95_percent_upper_confidence_bound",
                },
                "tasks": 16,
                "attempts": 32,
                "incomplete_rows": 0,
                "no_attempt_rows": 0,
            },
            "invalid_id_hard_gate": {
                "status": "PASS",
                "invalid_id_count": 0,
                "required_maximum": 0,
                "duplicates_are_invalid_ids": False,
            },
        },
        "critical_harm_absolute": {
            "strict_observed": {
                "status": "ESTIMABLE",
                "n_pairs": 16,
                "n_tasks": 16,
                "missing_pairs": 0,
                "ci": {
                    "point": 0.04,
                    "lower": 0.0,
                    "upper": 0.10,
                    "interval_type": ("two_sided_95_percent_source_topic_cluster_bootstrap"),
                },
            },
            "all_offered_adverse": {
                "status": "ESTIMABLE",
                "n_pairs": 16,
                "n_tasks": 16,
                "missing_pairs": 0,
                "missing_strict_pairs_assigned_adverse": 0,
                "definition": (
                    "P1 worst_case critical_harm; unavailable worst_case is assigned 1.0"
                ),
                "ci": {
                    "point": 0.05,
                    "lower": 0.01,
                    "upper": 0.12,
                    "interval_type": ("two_sided_95_percent_source_topic_cluster_bootstrap"),
                },
            },
        },
        "terminal_failure_risk_difference": {
            "status": "OK",
            "ci": {"point": 0.0, "lower": None, "upper": 1.0},
        },
        "fallback_risk_difference": {
            "status": "OK",
            "ci": {"point": 0.0, "lower": None, "upper": 1.0},
        },
        "paired_invalid_block_count": 0,
        "direct_trace_unavailable_count": 0,
        "direct_trace_invalid_count": 0,
        "selector_guard_unavailable_count": 0,
        "machine_estimands_ready": machine_ready,
    }


def _inputs(
    *,
    h_row: dict | None = None,
    c_row: dict | None = None,
    hc_row: dict | None = None,
    audited: bool = False,
) -> tuple[dict, dict, dict, dict, dict, dict | None]:
    decision, week1, variants = _configs()
    config_hashes = {
        "decision": config_sha(decision),
        "week1": config_sha(week1),
        "variants": config_sha(variants),
    }
    run_id = "run-1"
    phase_id = "screen"
    scope_sha = "e" * 64
    execution_binding_sha = "b" * 64
    analysis_execution_binding_sha = "8" * 64
    analysis_approved_commit = "1" * 40
    protocol_document_sha = "c" * 64
    schedule_sha = "s" * 64
    freeze_sha = "f" * 64
    design_sha = "d" * 64
    registry_sha = "r" * 64
    spec_sha = "a" * 64
    source_cluster_map_sha = "9" * 64

    receipt = None
    if audited:
        receipt = _seal(
            {
                "schema_version": "human_audit_receipt_v2",
                "status": "AUDITED",
                "run_id": run_id,
                "phase_id": phase_id,
                "execution_binding_sha256": execution_binding_sha,
                "protocol_document_sha256": protocol_document_sha,
                "evaluation_scope_sha256": scope_sha,
                "queue_content_sha256": "4" * 64,
                "results_content_sha256": "5" * 64,
                "reviewer_id": "reviewer",
                "reviewed_at_utc": "2026-07-25T00:00:00Z",
                "reviewer_attestation_sha256": "6" * 64,
                "truth_packet_sha256_by_task": {},
                "score_content_sha256_by_block": {},
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
        )
    itt = _seal(
        {
            "schema_version": "itt_verdict_input_v1",
            "run_id": run_id,
            "phase_id": phase_id,
            "evaluation_scope_sha256": scope_sha,
            "execution_binding_sha256": execution_binding_sha,
            "analysis_execution_binding_sha256": analysis_execution_binding_sha,
            "analysis_approved_commit": analysis_approved_commit,
            "protocol_document_sha256": protocol_document_sha,
            "schedule_sha256": schedule_sha,
            "freeze_root_sha256": freeze_sha,
            "analysis_design_receipt_sha256": design_sha,
            "task_feature_registry_sha256": registry_sha,
            "eligibility_spec_content_sha256": spec_sha,
            "source_cluster_map_sha256": source_cluster_map_sha,
            "claim_scope": "FORMATIVE_ONLY",
            "truth_audited": audited,
            "human_audit_receipt_sha256": (receipt["content_sha256"] if receipt else None),
            "arms": {
                "H_MARKDOWN_ID": h_row or _arm_row("h"),
                "C_ID": c_row or _arm_row("c"),
                "H_PLUS_C": hc_row or _arm_row("h", "c"),
            },
        }
    )
    effects = _seal(
        {
            "schema_version": "e2e_effects_v2",
            "run_id": run_id,
            "phase_id": phase_id,
            "evaluation_scope_sha256": scope_sha,
            "task_feature_registry_sha256": registry_sha,
            "provenance": {
                "execution_binding_sha256": execution_binding_sha,
                "protocol_document_sha256": protocol_document_sha,
                "schedule_sha256": schedule_sha,
                "freeze_root_sha256": freeze_sha,
            },
            "eligibility": {
                "status": "OK",
                "conditional_rule_stability_pass": True,
                "heldout_evaluation": {
                    "fit_used_heldout": False,
                    "rows": 0,
                    "status": "NOT_PRESENT_IN_THIS_RUN_PHASE",
                },
            },
            "factorial_endpoints": {
                # The primary work endpoint under the native-concurrent layer. The serialized
                # `service_work_seconds` below is the mechanism layer's metric and is kept so a
                # test cannot pass by reading whichever one happens to be present.
                "interval_union_seconds": {
                    "status": "OK",
                    "h_simple": {"point": 0.20, "lower": 0.12, "upper": None},
                    "c_simple": {"point": 0.15, "lower": 0.08, "upper": None},
                    "joint": {"point": 0.28, "lower": 0.18, "upper": None},
                    "interaction": {"point": 0.01, "lower": -0.02, "upper": None},
                    "task_effect_distribution": {
                        name: {
                            "n_tasks": 16,
                            "mean": 0.2,
                            "median": median,
                            "p05": p05,
                            "p10": p05 + 0.01,
                            "p90": median + 0.08,
                            "p95": median + 0.1,
                            "minimum": p05 - 0.02,
                            "maximum": median + 0.12,
                            "scale": "per_task_fraction_saved",
                            "inference": ("DESCRIPTIVE_ONLY_CLUSTER_BOOTSTRAP_REMAINS_PRIMARY"),
                        }
                        for name, median, p05 in (
                            ("h_simple", 0.20, -0.04),
                            ("c_simple", 0.15, -0.02),
                            ("joint", 0.28, 0.03),
                            ("interaction", 0.01, -0.05),
                        )
                    },
                },
                "service_work_seconds": {
                    "status": "OK",
                    "h_simple": {"point": 0.20, "lower": 0.12, "upper": None},
                    "c_simple": {"point": 0.15, "lower": 0.08, "upper": None},
                    "joint": {"point": 0.28, "lower": 0.18, "upper": None},
                    "interaction": {"point": 0.01, "lower": -0.02, "upper": None},
                    "task_effect_distribution": {
                        name: {
                            "n_tasks": 16,
                            "mean": 0.2,
                            "median": median,
                            "p05": p05,
                            "p10": p05 + 0.01,
                            "p90": median + 0.08,
                            "p95": median + 0.1,
                            "minimum": p05 - 0.02,
                            "maximum": median + 0.12,
                            "scale": "per_task_fraction_saved",
                            "inference": ("DESCRIPTIVE_ONLY_CLUSTER_BOOTSTRAP_REMAINS_PRIMARY"),
                        }
                        for name, median, p05 in (
                            ("h_simple", 0.20, -0.04),
                            ("c_simple", 0.15, -0.02),
                            ("joint", 0.28, 0.03),
                            ("interaction", 0.01, -0.05),
                        )
                    },
                },
                "e2e_latency_seconds": {
                    "status": "OK",
                    "h_simple": {"point": -1.0, "lower": -2.0, "upper": 0.2},
                    "c_simple": {"point": -0.5, "lower": -1.5, "upper": 0.5},
                    "joint": {"point": -1.8, "lower": -3.0, "upper": -0.1},
                    "interaction": {"point": -0.3, "lower": -1.0, "upper": 0.4},
                    "task_effect_distribution": {
                        name: {
                            "n_tasks": 16,
                            "mean": median,
                            "median": median,
                            "p05": median - 2.0,
                            "p10": median - 1.5,
                            "p90": p95 - 0.2,
                            "p95": p95,
                            "minimum": median - 3.0,
                            "maximum": p95 + 0.5,
                            "scale": "per_task_treatment_minus_baseline",
                            "inference": ("DESCRIPTIVE_ONLY_CLUSTER_BOOTSTRAP_REMAINS_PRIMARY"),
                        }
                        for name, median, p95 in (
                            ("h_simple", -1.0, 1.2),
                            ("c_simple", -0.5, 1.8),
                            ("joint", -1.8, 0.4),
                            ("interaction", -0.3, 1.0),
                        )
                    },
                },
            },
            "task_level_joint_outcomes": {
                "status": "OK",
                "scope": "TASK_LEVEL_PRIMARY_H_C_HC_JOINT_OUTCOMES",
                "policy": copy.deepcopy(decision["task_level_joint_outcomes"]),
                "policy_sha256": sha256_hex(canonical_json(decision["task_level_joint_outcomes"])),
                "tasks": 16,
                "clusters": 8,
                "replicate_reduction": ("task_mean_before_source_topic_cluster_bootstrap"),
                "arms": {
                    arm_name: {
                        "saving_threshold_proportions": {
                            threshold: {
                                "status": "ESTIMABLE",
                                "point": point,
                                "lower": max(0.0, point - 0.1),
                                "upper": min(1.0, point + 0.1),
                            }
                            for threshold, point in (
                                ("saving_ge_10pct", 0.75),
                                ("saving_ge_25pct", 0.50),
                                ("saving_ge_33pct", 0.25),
                            )
                        },
                        "quality_qualified_pareto_win_proportion": {
                            "status": "ESTIMABLE",
                            "point": 0.70,
                            "lower": 0.60,
                            "upper": 0.80,
                        },
                        "slower_and_quality_harmed_proportion": {
                            "status": "ESTIMABLE",
                            "point": 0.05,
                            "lower": 0.0,
                            "upper": 0.15,
                        },
                    }
                    for arm_name in ("h_simple", "c_simple", "joint")
                },
            },
        }
    )
    factorial = decision["e2e_analysis"]["core_factorial"]
    variant_by_id = {row["variant_id"]: row for row in variants["variants"]}
    screen_by_id = {row["arm_id"]: row for row in week1["screen_arms"]["arms"]}
    matched_design = week1["matched_contrasts"]
    control_contrast_ids = {
        "H_LLM_VS_CPU",
        "H_ID_VS_PROSE",
        "C_LLM_VS_CPU",
        "C_ID_VS_PROSE",
        "HC_LLM_VS_CPU",
        "HC_ID_VS_PROSE",
    }
    matched_rows = [
        {
            **copy.deepcopy(row),
            "pairing_status": "OK",
            "semantic_validation": {"status": "OK"},
        }
        for row in matched_design["pairs"]
        if row["contrast_id"] in control_contrast_ids
    ]
    requested_arms = {
        row[side] for row in matched_design["pairs"] for side in ("left_arm_id", "right_arm_id")
    }
    arm_variants = {}
    for arm_id in sorted(requested_arms):
        arm = screen_by_id[arm_id]
        arm_variants[arm_id] = resolve_arm_semantics(
            arm_id,
            arm,
            variant_by_id,
            matched_design["executable_variant_fields"],
        )
    structured_policy = decision["structured_increment"]
    control_policy = structured_policy["control_estimands"]

    def component_gate(control_type: str, contrast_id: str) -> dict:
        return {
            "status": "ESTABLISHED",
            "contrast_id": contrast_id,
            "holm_family_id": structured_policy["multiplicity_family"]["family_id"],
            "estimand_contract_sha256": sha256_hex(canonical_json(control_policy[control_type])),
            "adjusted_primary_gate_pass": True,
            "all_strict_quality_ni_guards_pass": True,
            "first_boundary_input_comparability_pass": True,
            **(
                {"service_work_cost_guard_pass": True}
                if control_type == "LLM_vs_CPU"
                else {
                    "prose_control_integrity_gate_pass": True,
                    "main_estimand_scope":
                        "STRUCTURED_ID_POLICY_VS_BOUNDED_SHORT_PROSE_POLICY",
                    "pointer_only_attribution_status": "ESTABLISHED",
                    "raw_contract_quality_ni_guards": {},
                    "raw_contract_work_sensitivity": {"status": "OK"},
                }
            ),
        }

    matched = _seal(
        {
            "schema_version": "matched_contrast_analysis_v2",
            "run_id": run_id,
            "phase_id": phase_id,
            "evaluation_scope_sha256": scope_sha,
            "execution_binding_sha256": execution_binding_sha,
            "protocol_document_sha256": protocol_document_sha,
            "input_provenance": {
                "source_cluster_map_sha256": source_cluster_map_sha,
                "matched_contrasts_sha256": sha256_hex(canonical_json(matched_design)),
                "arm_variants_sha256": sha256_hex(canonical_json(arm_variants)),
            },
            "contrasts": matched_rows,
            "structured_increment_gates": {
                "schema_version": structured_policy["schema_version"],
                "policy_sha256": sha256_hex(canonical_json(structured_policy)),
                "multiplicity_family": copy.deepcopy(structured_policy["multiplicity_family"]),
                "by_node": {
                    "WEBPAGE_P1": {
                        "status": "ESTABLISHED",
                        "primary_arm_id": "H_MARKDOWN_ID",
                        "required_contrast_ids": [
                            "H_LLM_VS_CPU",
                            "H_ID_VS_PROSE",
                        ],
                        "compound_status": "ESTABLISHED",
                        "pointer_only_attribution_status": "ESTABLISHED",
                        "attribution_scope":
                            "LLM_PLUS_STRUCTURED_POINTER_REPRESENTATION",
                        "component_gates": {
                            "LLM_vs_CPU": component_gate("LLM_vs_CPU", "H_LLM_VS_CPU"),
                            "structured_selection_vs_prose": component_gate(
                                "structured_selection_vs_prose", "H_ID_VS_PROSE"
                            ),
                        },
                    },
                    "C_VISIBLE": {
                        "status": "ESTABLISHED",
                        "primary_arm_id": "C_ID",
                        "required_contrast_ids": [
                            "C_LLM_VS_CPU",
                            "C_ID_VS_PROSE",
                        ],
                        "compound_status": "ESTABLISHED",
                        "pointer_only_attribution_status": "ESTABLISHED",
                        "attribution_scope":
                            "LLM_PLUS_STRUCTURED_POINTER_REPRESENTATION",
                        "component_gates": {
                            "LLM_vs_CPU": component_gate("LLM_vs_CPU", "C_LLM_VS_CPU"),
                            "structured_selection_vs_prose": component_gate(
                                "structured_selection_vs_prose", "C_ID_VS_PROSE"
                            ),
                        },
                    },
                    "H_PLUS_C_VISIBLE": {
                        "status": "ESTABLISHED",
                        "primary_arm_id": "H_PLUS_C",
                        "required_contrast_ids": [
                            "HC_LLM_VS_CPU",
                            "HC_ID_VS_PROSE",
                        ],
                        "compound_status": "ESTABLISHED",
                        "pointer_only_attribution_status": "ESTABLISHED",
                        "attribution_scope":
                            "LLM_PLUS_STRUCTURED_POINTER_REPRESENTATION",
                        "component_gates": {
                            "LLM_vs_CPU": component_gate(
                                "LLM_vs_CPU", "HC_LLM_VS_CPU"
                            ),
                            "structured_selection_vs_prose": component_gate(
                                "structured_selection_vs_prose", "HC_ID_VS_PROSE"
                            ),
                        },
                    },
                },
            },
        }
    )
    e2e = _seal(
        {
            "schema_version": "e2e_analysis_bundle_v1",
            "run_id": run_id,
            "phase_id": phase_id,
            "evaluation_scope_sha256": scope_sha,
            "execution_binding_sha256": execution_binding_sha,
            "protocol_document_sha256": protocol_document_sha,
            "design_bindings": {
                "analysis_design_receipt_sha256": design_sha,
                "task_feature_registry_sha256": registry_sha,
                "eligibility_spec_content_sha256": spec_sha,
            },
            "config_sha256": config_hashes,
            "semantic_mapping": {
                "core_arm_map": {key: factorial[key]["arm_id"] for key in ("p0", "h", "c", "hc")}
            },
            "analysis_policy": {
                "analysis_execution_binding_sha256": analysis_execution_binding_sha,
                "analysis_approved_commit": analysis_approved_commit,
                "trajectory_checkpoint_divergence": "mediated_end_to_end_outcome_not_pairing_error"
            },
            "matched_contrasts": matched,
            "e2e_effects": effects,
        }
    )
    return itt, e2e, decision, week1, variants, receipt


def _build(inputs, **kwargs):
    itt, e2e, decision, week1, variants, receipt = inputs
    return build_final_decision(
        itt,
        e2e,
        decision_config=decision,
        week1_config=week1,
        variants_config=variants,
        human_audit_receipt=kwargs.pop("human_audit_receipt", None),
        protocol_sha=kwargs.pop(
            "protocol_sha", str(itt["protocol_document_sha256"])
        ),
        generated_at_utc=kwargs.pop("generated_at_utc", "2026-07-25T00:00:00Z"),
        **kwargs,
    )


def _mutate_matched(inputs: tuple, mutate) -> tuple:
    values = list(inputs)
    e2e = copy.deepcopy(values[1])
    matched = e2e["matched_contrasts"]
    mutate(matched)
    e2e["matched_contrasts"] = _reseal(matched)
    values[1] = _reseal(e2e)
    return tuple(values)


def _operational(inputs, *, arm_id: str, variant_id: str) -> dict:
    itt, e2e, decision, week1, variants, _ = inputs
    return _seal(
        {
            "schema_version": "operational_decision_evidence_v1",
            "measurement_layer": "operational",
            "run_id": "operational-run-1",
            "phase_id": "operational-screen",
            "evaluation_scope_sha256": "9" * 64,
            "parent_causal_evidence_sha256": {
                "ITT_VERDICT_INPUTS": itt["content_sha256"],
                "E2E_ANALYSIS": e2e["content_sha256"],
            },
            "policy_sha256": "8" * 64,
            "taskset_sha256": "7" * 64,
            "config_sha256": {
                "decision": config_sha(decision),
                "week1": config_sha(week1),
                "variants": config_sha(variants),
            },
            "arms": {
                arm_id: {
                    "status": "OK",
                    "arm_id": arm_id,
                    "variant_id": variant_id,
                    "block_makespan_saving_lcb": 0.20,
                    "throughput_improvement_lcb": 0.18,
                    "p95_latency_ratio_ucb": 1.02,
                    "gpu_joule_ratio_ucb": 0.90,
                    "terminal_failure_increase_ucb_pp": 1.0,
                    "critical_harm_increase_ucb_pp": 1.0,
                    "quality_guards_pass": True,
                    "e2e_speedup_lcb": 1.2,
                }
            },
        }
    )


def test_causal_pass_is_mechanism_only_never_keep() -> None:
    result = _build(_inputs())

    assert result.webpage_p1.verdict is Verdict.MECHANISM_ONLY
    assert result.c_visible.verdict is Verdict.MECHANISM_ONLY
    assert result.h_plus_c_visible.verdict is Verdict.MECHANISM_ONLY
    assert (
        result.h_plus_c_visible.arm_results["H_PLUS_C"]["structured_increment"][
            "decision_use"
        ]
        == "REQUIRED_FOR_POSITIVE_P1_MECHANISM_ATTRIBUTION"
    )
    assert all(node.verdict is not Verdict.KEEP for node in result.nodes)
    assert result.operational_status == "UNAVAILABLE_PRODUCER_NOT_IMPLEMENTED"
    assert result.trajectory_outcomes
    assert all(
        value["interpretation"] == "DESCRIPTIVE_MEDIATED_OUTCOME_NOT_TRAJECTORY_EQUALITY_FILTER"
        for value in result.trajectory_outcomes.values()
    )


def test_consumer_shaped_operational_json_cannot_impersonate_missing_producer() -> None:
    inputs = _inputs()
    correct = _operational(inputs, arm_id="H_MARKDOWN_ID", variant_id="H02")
    with pytest.raises(ValueError, match="producer is NOT_IMPLEMENTED"):
        _build(inputs, operational_evidence=correct)

    result = _build(inputs)
    assert result.webpage_p1.verdict is Verdict.MECHANISM_ONLY
    assert result.operational_status == "UNAVAILABLE_PRODUCER_NOT_IMPLEMENTED"
    assert "Full concurrent batching/APC" in render_markdown(result)


def test_secondary_high_lcb_cannot_replace_exact_primary_arm() -> None:
    inputs = list(_inputs())
    itt = copy.deepcopy(inputs[0])
    secondary = _arm_row("h", work_lower=0.90, work_upper=0.99)
    itt["arms"]["H_FIXED_ID"] = secondary
    itt["content_sha256"] = sha256_hex(
        canonical_json({key: value for key, value in itt.items() if key != "content_sha256"})
    )
    inputs[0] = itt

    result = _build(tuple(inputs))
    assert result.webpage_p1.champion_variant == "H02"
    assert result.webpage_p1.work_saving.lower == pytest.approx(0.20)
    assert result.champion_selection_rule == "exact_preregistered_core_factorial_primary_arm_only"
    assert (
        result.executed_design["H_FIXED_ID"]["analysis_role"]
        == "SECONDARY_EXPLORATORY_NO_MULTIPLICITY_ADJUSTED_CLAIM"
    )


def test_work_ucb_below_meaningful_effect_kills_no_headroom() -> None:
    result = _build(_inputs(hc_row=_arm_row("h", "c", work_lower=0.01, work_upper=0.08)))

    assert result.h_plus_c_visible.verdict is Verdict.KILL_NO_HEADROOM
    arm = result.h_plus_c_visible.arm_results["H_PLUS_C"]
    assert arm["work_saving"]["upper"] == pytest.approx(0.08)


def test_invalid_measurement_is_not_established_not_killed() -> None:
    invalid = _arm_row("h", machine_ready=False)
    invalid["work_saving"] = None
    invalid["work_pairs_observed"] = 0
    invalid["work_pairs_missing"] = 16
    result = _build(_inputs(h_row=invalid))

    assert result.webpage_p1.verdict is Verdict.NOT_ESTABLISHED
    assert result.webpage_p1.verdict not in {
        Verdict.KILL_STRUCTURAL,
        Verdict.KILL_HARM,
        Verdict.KILL_NO_HEADROOM,
    }


def test_unweighted_selector_ratio_cannot_satisfy_weighted_recall_gate() -> None:
    row = _arm_row("h")
    del row["selector_estimands"]["h_weighted_evidence_recall"]
    result = _build(_inputs(h_row=row))

    assert result.webpage_p1.verdict is Verdict.NOT_ESTABLISHED
    arm = result.webpage_p1.arm_results["H_MARKDOWN_ID"]
    assert (
        arm["selector_gates"]["h_selector_conditional_recall"]["decision_use"]
        == "DESCRIPTIVE_ONLY_NOT_A_SUBSTITUTE_FOR_WEIGHTED_SELECTOR_RECALL"
    )
    assert arm["selector_gates"]["h_weighted_evidence_recall"]["status"] == ("MISSING_REQUIRED")


def test_missing_invalid_id_gate_is_not_established_never_a_structural_kill() -> None:
    row = _arm_row("h", "c")
    del row["selector_output_validity"]
    row["machine_estimands_ready"] = False

    result = _build(_inputs(hc_row=row))

    assert result.h_plus_c_visible.verdict is Verdict.NOT_ESTABLISHED
    assert result.h_plus_c_visible.verdict is not Verdict.KILL_STRUCTURAL
    arm = result.h_plus_c_visible.arm_results["H_PLUS_C"]
    assert "INVALID_ID_HARD_GATE_NOT_ESTABLISHED" in arm["blockers"]


def test_verified_out_of_set_id_is_a_structural_failure() -> None:
    row = _arm_row("h", "c")
    validity = row["selector_output_validity"]
    validity["invalid_id_count"] = 1
    validity["invalid_id_hard_gate"]["invalid_id_count"] = 1
    validity["invalid_id_hard_gate"]["status"] = "FAIL"
    # The real estimator also drops machine_estimands_ready when the hard gate fails.  A
    # complete trace still establishes the structural violation rather than turning it into
    # mere missingness.
    row["machine_estimands_ready"] = False

    result = _build(_inputs(hc_row=row))

    assert result.h_plus_c_visible.verdict is Verdict.KILL_STRUCTURAL
    arm = result.h_plus_c_visible.arm_results["H_PLUS_C"]
    assert arm["selector_output_validity"]["decision_status"] == "FAIL"


def test_duplicate_repairs_are_sensitivity_events_not_invalid_ids() -> None:
    row = _arm_row("h", "c")
    validity = row["selector_output_validity"]
    validity["strict_valid_count"] = 27
    validity["repaired_attempt_count"] = 5
    validity["no_repair_adverse_row_count"] = 3
    validity["strict_valid_rate_lcb"]["ci"]["point"] = 27 / 32
    validity["strict_valid_rate_lcb"]["ci"]["lower"] = 0.70
    validity["repair_rate_ucb"]["ci"]["point"] = 5 / 32
    validity["repair_rate_ucb"]["ci"]["upper"] = 0.30
    for metric in row["quality_effects"].values():
        metric["no_repair_adverse_dirty_pairs"] = 3

    result = _build(_inputs(hc_row=row))

    assert result.h_plus_c_visible.verdict is Verdict.MECHANISM_ONLY
    arm = result.h_plus_c_visible.arm_results["H_PLUS_C"]
    assert arm["selector_output_validity"]["invalid_id_hard_gate"]["status"] == "PASS"
    assert arm["selector_output_validity"]["repaired_attempt_count"] == 5
    assert arm["no_repair_quality"]["weighted_required_atom_recall"]["dirty_pairs"] == 3


def test_control_composite_missing_or_failed_cannot_be_called_p1_mechanism() -> None:
    missing = _mutate_matched(
        _inputs(),
        lambda matched: matched.pop("structured_increment_gates"),
    )
    missing_result = _build(missing)
    assert missing_result.webpage_p1.verdict is Verdict.NOT_ESTABLISHED
    assert missing_result.c_visible.verdict is Verdict.NOT_ESTABLISHED
    assert "P1_VS_P0_BENEFIT_OBSERVED" in " ".join(missing_result.webpage_p1.reasons)

    def fail_cpu(matched: dict) -> None:
        gate = matched["structured_increment_gates"]["by_node"]["WEBPAGE_P1"]
        gate["component_gates"]["LLM_vs_CPU"]["adjusted_primary_gate_pass"] = False

    failed = _mutate_matched(_inputs(), fail_cpu)
    failed_result = _build(failed)
    assert failed_result.webpage_p1.verdict is Verdict.NOT_ESTABLISHED
    assert failed_result.webpage_p1.verdict is not Verdict.KILL_NO_HEADROOM
    arm = failed_result.webpage_p1.arm_results["H_MARKDOWN_ID"]
    assert arm["structured_increment"]["status"] == "NOT_ESTABLISHED"


def test_no_repair_and_task_distribution_are_rendered_without_becoming_primary() -> None:
    result = _build(_inputs())
    markdown = render_markdown(result)
    body = json.loads(render_json(result))

    assert (
        body["H_PLUS_C_VISIBLE"]["arm_results"]["H_PLUS_C"]["no_repair_quality"][
            "weighted_required_atom_recall"
        ]["status"]
        == "ESTIMABLE"
    )
    assert "no-repair adverse weighted-required-atom" in markdown
    assert "h_simple: median=0.2, bad lower tail p05=-0.04" in markdown
    assert "joint: median=-1.8s, bad upper tail p95=0.4s" in markdown
    assert "Raw operational p95 latency *ratio*" in markdown
    assert "cluster bootstrap remains primary inference" in markdown
    assert "quality_qualified_pareto_win_proportion: 0.7" in markdown
    assert "slower_and_quality_harmed_proportion: 0.05" in markdown
    assert "selected_token_precision=0.7" in markdown
    assert "exact token totals: published=800, offered=640" in markdown
    assert "Absolute all-offered adverse critical-harm rate: 0.05 [0.01, 0.12]" in markdown
    assert body["WEBPAGE_P1"]["critical_harm_rate"] == pytest.approx(0.05)


def test_incomplete_token_trace_only_blocks_selector_efficiency_subconclusion() -> None:
    row = _arm_row("c")
    summary = row["selector_efficiency"]["C"]
    summary["status"] = "NOT_ESTABLISHED_INCOMPLETE_TOKEN_TRACE"
    summary["token_trace_complete_rows"] = 15
    summary["token_trace_incomplete_rows"] = 1
    summary["metrics"] = {
        metric: {"status": "NOT_ESTABLISHED_INCOMPLETE_TOKEN_TRACE"}
        for metric in (
            "selected_token_precision",
            "weighted_truth_per_100_rendered_tokens",
            "materialization_ratio",
        )
    }

    result = _build(_inputs(c_row=row))

    assert result.c_visible.verdict is Verdict.MECHANISM_ONLY
    arm = result.c_visible.arm_results["C_ID"]
    assert arm["selector_efficiency"]["C"]["decision_status"] == "NOT_ESTABLISHED"
    assert "selector-efficiency conclusion is NOT_ESTABLISHED" in render_markdown(result)


def test_no_heldout_eligibility_can_never_be_conditional() -> None:
    inconclusive = _arm_row("h", work_lower=0.01, work_upper=0.25)
    result = _build(_inputs(h_row=inconclusive))

    assert result.webpage_p1.verdict is Verdict.NOT_ESTABLISHED
    assert result.webpage_p1.verdict is not Verdict.CONDITIONAL
    assert "WHEN_USEFUL_NOT_ESTABLISHED_TASK_LEVEL_ONLY" in (result.webpage_p1.limitations)
    assert result.eligibility_status == "WHEN_USEFUL_NOT_ESTABLISHED_TASK_LEVEL_ONLY"
    assert "not invocation coverage" in render_markdown(result)


def test_c_registry_is_not_substituted_by_c_visible() -> None:
    result = _build(_inputs())

    assert result.c_visible.verdict is Verdict.MECHANISM_ONLY
    assert result.c_registry.verdict is Verdict.NOT_ESTABLISHED
    assert result.c_registry.candidate_variants == ("C06-REG",)
    assert "C_VISIBLE_IS_NOT_A_SUBSTITUTE" in " ".join(result.c_registry.reasons)


def test_audit_only_changes_provisional_state_not_formative_scope() -> None:
    inputs = _inputs(audited=True)
    receipt = inputs[-1]
    provisional = _build(inputs)
    audited = _build(inputs, human_audit_receipt=receipt)

    assert provisional.verdict_status == PROVISIONAL_STATUS
    assert audited.verdict_status == AUDITED_STATUS
    assert provisional.claim_scope == audited.claim_scope == "FORMATIVE_ONLY"
    assert provisional.corpus_tier == audited.corpus_tier
    assert audited.confirmatory_power_shortfall is None
    assert audited.confirmatory_status == "NOT_APPLICABLE_FORMATIVE_ONLY"
    assert audited.webpage_p1.claim_level == "FORMATIVE_MECHANISM_CANDIDATE"
    assert "NOT_ASSESSED_FORMATIVE_ONLY" in render_markdown(audited)


def test_audited_label_with_any_correction_cannot_finalize_quality_verdict() -> None:
    inputs = _inputs(audited=True)
    receipt = copy.deepcopy(inputs[-1])
    receipt["correction_count"] = 1
    receipt["items_requiring_correction"] = 1
    receipt["audit_basis"] = "CORRECTED_BUT_NOT_RESCORED"
    receipt["content_sha256"] = sha256_hex(
        canonical_json({key: value for key, value in receipt.items() if key != "content_sha256"})
    )

    with pytest.raises(ValueError, match="correction-free v2"):
        _build(inputs, human_audit_receipt=receipt)


def test_provenance_mismatch_fails_before_deciding() -> None:
    inputs = list(_inputs())
    e2e = copy.deepcopy(inputs[1])
    e2e["evaluation_scope_sha256"] = "x" * 64
    e2e["content_sha256"] = sha256_hex(
        canonical_json({key: value for key, value in e2e.items() if key != "content_sha256"})
    )
    inputs[1] = e2e

    with pytest.raises(ValueError, match="bindings disagree"):
        _build(tuple(inputs))


def test_execution_binding_mismatch_fails_before_deciding() -> None:
    inputs = list(_inputs())
    e2e = copy.deepcopy(inputs[1])
    e2e["execution_binding_sha256"] = "9" * 64
    inputs[1] = _reseal(e2e)

    with pytest.raises(ValueError, match="execution identity execution_binding_sha256"):
        _build(tuple(inputs))


def test_protocol_and_generation_identity_cannot_default_to_caller_ambiguity() -> None:
    with pytest.raises(ValueError, match="protocol_sha"):
        _build(_inputs(), protocol_sha="")
    with pytest.raises(ValueError, match="differs from the frozen run protocol"):
        _build(_inputs(), protocol_sha="d" * 64)
    with pytest.raises(ValueError, match="generated_at_utc"):
        _build(_inputs(), generated_at_utc="")


def test_matched_controls_cannot_be_spliced_from_another_scope_or_cluster_map() -> None:
    inputs = _mutate_matched(
        _inputs(),
        lambda matched: matched.update(evaluation_scope_sha256="7" * 64),
    )
    with pytest.raises(ValueError, match="evaluation_scope_sha256 bindings disagree"):
        _build(inputs)

    inputs = _mutate_matched(
        _inputs(),
        lambda matched: matched["input_provenance"].update(
            source_cluster_map_sha256="8" * 64
        ),
    )
    with pytest.raises(ValueError, match="source/topic cluster map"):
        _build(inputs)


def test_json_markdown_are_write_once_derivations_of_same_typed_object(
    tmp_path: Path,
) -> None:
    decision = _build(_inputs())
    json_path = tmp_path / "WEEK1_P1_DECISION.json"
    md_path = tmp_path / "WEEK1_P1_DECISION.md"

    assert write_final_decision(decision, json_path=json_path, markdown_path=md_path) == {
        "json": "CREATED",
        "markdown": "CREATED",
    }
    assert write_final_decision(decision, json_path=json_path, markdown_path=md_path) == {
        "json": "EXISTING_IDENTICAL",
        "markdown": "EXISTING_IDENTICAL",
    }
    parsed = json.loads(render_json(decision))
    assert parsed["content_sha256"] == sha256_hex(
        canonical_json({key: value for key, value in parsed.items() if key != "content_sha256"})
    )
    assert parsed["input_sha256"]["ITT_VERDICT_INPUTS"]
    assert "Post-treatment search" in md_path.read_text(encoding="utf-8")


def test_finalize_cli_uses_only_fixed_scoped_paths_and_reuses_generation_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typer.testing import CliRunner

    from shapeflow_p1 import cli, protocol
    from shapeflow_p1.campaign.settings import Settings

    inputs = _inputs()
    itt, e2e = inputs[:2]
    settings = Settings.load(REPO, data_root=tmp_path)
    analysis_dir = settings.path("judgments") / itt["run_id"] / itt["phase_id"] / "analysis"
    analysis_dir.mkdir(parents=True)
    (analysis_dir / "ITT_VERDICT_INPUTS.json").write_text(json.dumps(itt), encoding="utf-8")
    (analysis_dir / "E2E_ANALYSIS.json").write_text(json.dumps(e2e), encoding="utf-8")
    monkeypatch.setattr(cli, "_require_role", lambda role: None)
    monkeypatch.setattr(cli, "_settings", lambda: settings)
    monkeypatch.setattr(
        cli,
        "_verified_analysis_binding",
        lambda **_kwargs: SimpleNamespace(
            digest=itt["analysis_execution_binding_sha256"],
            protocol_sha=itt["protocol_document_sha256"],
            approved_commit=itt["analysis_approved_commit"],
        ),
    )
    monkeypatch.setattr(
        protocol,
        "protocol_sha",
        lambda _repo: itt["protocol_document_sha256"],
    )

    args = [
        "finalize-decision",
        "--run-id",
        itt["run_id"],
        "--phase-id",
        itt["phase_id"],
    ]
    first = CliRunner().invoke(cli.app, args)
    assert first.exit_code == 0, first.output
    output_path = analysis_dir / "WEEK1_P1_DECISION.json"
    first_body = json.loads(output_path.read_text(encoding="utf-8"))
    second = CliRunner().invoke(cli.app, args)
    assert second.exit_code == 0, second.output
    second_body = json.loads(output_path.read_text(encoding="utf-8"))
    assert first_body["generated_at_utc"] == second_body["generated_at_utc"]
    assert '"json": "EXISTING_IDENTICAL"' in second.output

    rejected = CliRunner().invoke(cli.app, [*args, "--output", str(tmp_path / "elsewhere")])
    assert rejected.exit_code != 0
