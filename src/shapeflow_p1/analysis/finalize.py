"""Conservative, provenance-bound final decision for the P1 formative study.

This module answers three different questions without collapsing them into one:

* **does an executable P1 mechanism save complete service work without violating the
  all-offered strict quality guards?**
* **how large is the established saving?**  The lower confidence bound establishes benefit;
  the upper confidence bound is separately used to establish absence of meaningful headroom.
* **where might it work?**  A frozen task-level learner may produce formative hypotheses from
  pre-treatment features.  Those findings are reported in full, including null and unstable
  targets, but never alter a primary verdict or define an invocation/deployment policy.

The primary campaign is coupled-seed end-to-end ITT.  Search results, reasoning events and
checkpoint identities are therefore free to diverge after treatment; the E2E artifact records
those differences as mediated outcomes.  They are never used as a trajectory-equality filter.

The builder consumes only content-addressed ITT and E2E artifacts and verifies their exact
run/phase/evaluation/config bindings.  A missing or invalid arm measurement becomes
``NOT_ESTABLISHED``.  It cannot be converted into a structural or no-headroom kill.  Likewise,
causal-only evidence can establish ``MECHANISM_ONLY`` but never ``KEEP`` without a separately
bound operational artifact.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..canonical import canonical_json
from ..config import config_sha
from ..hashing import sha256_hex
from .decision import NodeEvidence, Verdict, decide
from .matched import resolve_arm_semantics
from .report import (
    AUDITED_STATUS,
    PROVISIONAL_STATUS,
    DecisionObject,
    EffectWithCI,
    NodeDecision,
    render_json,
    render_markdown,
)

__all__ = [
    "build_final_decision",
    "write_final_decision",
]


_QUALITY_CONFIG_KEYS = {
    "weighted_required_atom_recall": "weighted_required_atom_recall_pp",
    "grounded_claim_precision": "grounded_claim_precision_pp",
    "citation_correctness": "citation_correctness_pp",
    "citation_association": "citation_association_pp",
    "required_facet_coverage": "required_facet_coverage_pp",
    "qualified_report": "qualified_report_rate_pp",
}
_DIRECT_CONDITIONAL_GATES = (
    ("contradiction_pair_recall", "contradiction_pair_recall_min"),
    ("negative_gap_recall", "negative_gap_recall_min"),
)

_POSITIVE_VERDICTS = {
    Verdict.MECHANISM_ONLY,
    Verdict.CONDITIONAL,
    Verdict.KEEP,
    Verdict.THESIS_GRADE,
}


@dataclass(frozen=True)
class _ArmAssessment:
    arm_id: str
    variant_id: str
    evidence: NodeEvidence
    verdict: Verdict
    work: EffectWithCI | None
    quality: dict[str, EffectWithCI]
    selector_gates: dict[str, dict[str, Any]]
    selector_output_validity: dict[str, Any]
    selector_efficiency: dict[str, Any]
    no_repair_quality: dict[str, Any]
    critical_harm_absolute: dict[str, Any]
    structured_increment: dict[str, Any]
    reasons: tuple[str, ...]
    blockers: tuple[str, ...]

    def obj(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "variant_id": self.variant_id,
            "verdict": self.verdict.value,
            "study_valid": self.evidence.study_valid,
            "structural_pass": self.evidence.structural_pass,
            "deterministic_harm": self.evidence.deterministic_harm,
            "quality_guards_pass": self.evidence.quality_guards_pass,
            "work_saving": self.work.obj() if self.work else None,
            "selector_gates": self.selector_gates,
            "selector_output_validity": self.selector_output_validity,
            "selector_efficiency": self.selector_efficiency,
            "no_repair_quality": self.no_repair_quality,
            "critical_harm_absolute": self.critical_harm_absolute,
            "structured_increment": self.structured_increment,
            "quality_effects": {name: value.obj() for name, value in sorted(self.quality.items())},
            "operational_available": self.evidence.operational_available,
            "conditional_rule_keeps": self.evidence.conditional_rule_keeps,
            "coverage_lcb": self.evidence.coverage_lcb,
            "reasons": list(self.reasons),
            "blockers": list(self.blockers),
        }


def _unsigned_sha(body: Mapping[str, Any]) -> str:
    return sha256_hex(
        canonical_json({key: value for key, value in body.items() if key != "content_sha256"})
    )


def _verified(
    raw: Mapping[str, Any],
    *,
    label: str,
    schemas: str | Sequence[str],
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} is not an object")
    body = dict(raw)
    accepted = {schemas} if isinstance(schemas, str) else set(schemas)
    if body.get("schema_version") not in accepted:
        raise ValueError(f"{label} has unsupported schema {body.get('schema_version')!r}")
    recorded = str(body.get("content_sha256") or "")
    actual = _unsigned_sha(body)
    if recorded != actual:
        raise ValueError(
            f"{label} content hash does not verify: records {recorded!r}, hashes to {actual}"
        )
    return body


def _finite(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite")
    return number


def _maybe_finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _require_sha256(value: Any, *, label: str) -> str:
    digest = str(value or "")
    try:
        valid = len(digest) == 64 and digest == digest.lower() and int(digest, 16) >= 0
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(f"{label} must be a non-empty lowercase SHA-256")
    return digest


def _require_generated_at(value: str) -> str:
    timestamp = str(value or "")
    if not timestamp.endswith("Z"):
        raise ValueError("generated_at_utc must be an explicit UTC timestamp ending in Z")
    try:
        datetime.fromisoformat(timestamp[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("generated_at_utc is not a valid ISO-8601 timestamp") from exc
    return timestamp


def _effect(ci: Mapping[str, Any] | None, *, unit: str) -> EffectWithCI | None:
    if not isinstance(ci, Mapping):
        return None
    point = _maybe_finite(ci.get("point"))
    lower = _maybe_finite(ci.get("lower"))
    upper = _maybe_finite(ci.get("upper"))
    if point is None:
        return None
    # One-sided artifacts intentionally omit the irrelevant side. Preserve that open side in
    # the typed report rather than fabricating a two-sided interval around the point estimate.
    return EffectWithCI(
        point=point,
        lower=lower,
        upper=upper,
        unit=unit,
        interval_type=str(ci.get("interval_type") or ci.get("interval") or ""),
        direction=str(
            ci.get("decision_direction")
            or ci.get("effect_orientation")
            or ci.get("bound_direction")
            or ""
        ),
    )


def _config_hashes(
    decision_config: Mapping[str, Any],
    week1_config: Mapping[str, Any],
    variants_config: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "decision": config_sha(dict(decision_config)),
        "variants": config_sha(dict(variants_config)),
        "week1": config_sha(dict(week1_config)),
    }


def _verify_input_bindings(
    itt_raw: Mapping[str, Any],
    e2e_raw: Mapping[str, Any],
    *,
    config_sha256: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    itt = _verified(itt_raw, label="ITT_VERDICT_INPUTS", schemas="itt_verdict_input_v1")
    e2e = _verified(e2e_raw, label="E2E_ANALYSIS", schemas="e2e_analysis_bundle_v1")
    effects = _verified(
        e2e.get("e2e_effects") or {},
        label="E2E_ANALYSIS.e2e_effects",
        schemas="e2e_effects_v2",
    )
    matched = e2e.get("matched_contrasts")
    if matched is not None:
        matched = _verified(
            matched,
            label="E2E_ANALYSIS.matched_contrasts",
            schemas="matched_contrast_analysis_v2",
        )

    for field in ("run_id", "phase_id", "evaluation_scope_sha256"):
        values = [
            str(itt.get(field) or ""),
            str(e2e.get(field) or ""),
            str(effects.get(field) or ""),
        ]
        if isinstance(matched, Mapping):
            values.append(str(matched.get(field) or ""))
        if len(set(values)) != 1 or "" in values:
            raise ValueError(f"ITT/E2E {field} bindings disagree or are empty")

    e2e_config = e2e.get("config_sha256")
    if not isinstance(e2e_config, Mapping) or {
        key: str(e2e_config.get(key) or "") for key in ("decision", "variants", "week1")
    } != dict(config_sha256):
        raise ValueError("E2E analysis does not bind the supplied frozen configs")

    bindings = e2e.get("design_bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("E2E analysis lacks pre-treatment design bindings")
    for field in (
        "analysis_design_receipt_sha256",
        "task_feature_registry_sha256",
        "eligibility_spec_content_sha256",
    ):
        if not str(itt.get(field) or "") or str(itt.get(field)) != str(bindings.get(field) or ""):
            raise ValueError(f"ITT/E2E pre-treatment binding {field} disagrees")
    if str(effects.get("task_feature_registry_sha256") or "") != str(
        itt["task_feature_registry_sha256"]
    ):
        raise ValueError("E2E effects use another task-feature registry")
    provenance = effects.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or str(provenance.get("freeze_root_sha256") or "")
        != str(itt.get("freeze_root_sha256") or "")
        or str(provenance.get("schedule_sha256") or "") != str(itt.get("schedule_sha256") or "")
    ):
        raise ValueError("ITT/E2E schedule or frozen-root provenance disagrees")
    for field in ("execution_binding_sha256", "protocol_document_sha256"):
        values = [
            str(itt.get(field) or ""),
            str(e2e.get(field) or ""),
            str((provenance or {}).get(field) or "")
            if isinstance(provenance, Mapping)
            else "",
        ]
        if isinstance(matched, Mapping):
            values.append(str(matched.get(field) or ""))
        if len(set(values)) != 1 or "" in values:
            raise ValueError(f"ITT/E2E execution identity {field} disagrees or is empty")
        _require_sha256(values[0], label=field)
    if isinstance(matched, Mapping):
        matched_inputs = matched.get("input_provenance")
        if (
            not isinstance(matched_inputs, Mapping)
            or not str(itt.get("source_cluster_map_sha256") or "")
            or str(matched_inputs.get("source_cluster_map_sha256") or "")
            != str(itt.get("source_cluster_map_sha256") or "")
        ):
            raise ValueError(
                "matched controls use another or unbound source/topic cluster map"
            )
    policy = e2e.get("analysis_policy")
    if (
        not isinstance(policy, Mapping)
        or policy.get("trajectory_checkpoint_divergence")
        != "mediated_end_to_end_outcome_not_pairing_error"
    ):
        raise ValueError(
            "E2E analysis does not freeze post-treatment trajectory divergence as an outcome"
        )
    if (
        not str(itt.get("analysis_execution_binding_sha256") or "")
        or str(policy.get("analysis_execution_binding_sha256") or "")
        != str(itt.get("analysis_execution_binding_sha256") or "")
        or not str(itt.get("analysis_approved_commit") or "")
        or str(policy.get("analysis_approved_commit") or "")
        != str(itt.get("analysis_approved_commit") or "")
    ):
        raise ValueError(
            "ITT/E2E analysis implementations are not bound to the same approved clean commit"
        )
    _require_sha256(
        str(itt["analysis_execution_binding_sha256"]),
        label="analysis_execution_binding_sha256",
    )
    if len(str(itt["analysis_approved_commit"])) != 40:
        raise ValueError("analysis_approved_commit is not a full Git commit identity")
    return itt, e2e, effects


def _trajectory_outcomes(
    effects: Mapping[str, Any],
    decision_config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    expected = (
        (decision_config.get("e2e_analysis") or {}).get("trajectory_endpoints")
        if isinstance(decision_config.get("e2e_analysis"), Mapping)
        else None
    )
    if not isinstance(expected, list) or not expected or len(expected) != len(set(expected)):
        raise ValueError("decision config has no unique frozen trajectory endpoint family")
    observed = effects.get("trajectory_endpoints")
    observed = observed if isinstance(observed, Mapping) else {}
    output: dict[str, dict[str, Any]] = {}
    for raw_name in expected:
        name = str(raw_name)
        raw = observed.get(name)
        if not isinstance(raw, Mapping):
            output[name] = {
                "status": "NOT_ESTIMABLE",
                "reason": "PREREGISTERED_TRAJECTORY_ENDPOINT_MISSING",
                "interpretation": "DESCRIPTIVE_MEDIATED_OUTCOME_NOT_TRAJECTORY_EQUALITY_FILTER",
            }
            continue
        # Canonical round-trip makes the final typed object own a plain JSON snapshot, not a
        # caller-controlled Mapping subclass.
        result = json.loads(canonical_json(dict(raw)))
        result["interpretation"] = "DESCRIPTIVE_MEDIATED_OUTCOME_NOT_TRAJECTORY_EQUALITY_FILTER"
        output[name] = result
    return output


def _variant_registry(
    week1_config: Mapping[str, Any],
    variants_config: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    variants_raw = variants_config.get("variants")
    arms_raw = (
        (week1_config.get("screen_arms") or {}).get("arms")
        if isinstance(week1_config.get("screen_arms"), Mapping)
        else None
    )
    if not isinstance(variants_raw, list) or not isinstance(arms_raw, list):
        raise ValueError("frozen variant or screen-arm registry is absent")
    variants: dict[str, dict[str, Any]] = {}
    for raw in variants_raw:
        if not isinstance(raw, Mapping):
            raise ValueError("variant registry contains a non-object")
        variant_id = str(raw.get("variant_id") or "")
        if not variant_id or variant_id in variants:
            raise ValueError(f"variant registry has duplicate/unnamed entry {variant_id!r}")
        variants[variant_id] = dict(raw)
    arms: dict[str, dict[str, Any]] = {}
    for raw in arms_raw:
        if not isinstance(raw, Mapping):
            raise ValueError("screen-arm registry contains a non-object")
        arm_id = str(raw.get("arm_id") or "")
        if not arm_id or arm_id in arms:
            raise ValueError(f"screen-arm registry has duplicate/unnamed entry {arm_id!r}")
        page = str(raw.get("page_variant") or "")
        close = str(raw.get("close_variant") or "")
        if page not in variants or close not in variants:
            raise ValueError(f"screen arm {arm_id!r} names an unknown variant")
        arms[arm_id] = dict(raw)
    return variants, arms


def _arm_groups(
    decision_config: Mapping[str, Any],
    variants: Mapping[str, Mapping[str, Any]],
    arms: Mapping[str, Mapping[str, Any]],
    e2e: Mapping[str, Any],
) -> tuple[dict[str, list[tuple[str, str]]], str]:
    factorial = (
        (decision_config.get("e2e_analysis") or {}).get("core_factorial")
        if isinstance(decision_config.get("e2e_analysis"), Mapping)
        else None
    )
    if not isinstance(factorial, Mapping) or set(factorial) != {"p0", "h", "c", "hc"}:
        raise ValueError("decision config must freeze exactly p0/h/c/hc factorial corners")
    semantic = e2e.get("semantic_mapping")
    e2e_map = (semantic or {}).get("core_arm_map") if isinstance(semantic, Mapping) else None
    configured_map = {
        key: str((factorial[key] or {}).get("arm_id") or "") for key in ("p0", "h", "c", "hc")
    }
    if (
        not isinstance(e2e_map, Mapping)
        or {key: str(e2e_map.get(key) or "") for key in configured_map} != configured_map
    ):
        raise ValueError("E2E core arm mapping disagrees with decision config")

    # Primary verdicts are intentionally *not* selected from all seven H / four C candidates
    # after observing their estimates.  The exact core-factorial arms are the preregistered
    # primary tests; all other arms remain secondary exploratory comparisons.
    primary_policy = decision_config.get("champion_selection")
    primary_by_node = (
        primary_policy.get("primary_arm_by_node") if isinstance(primary_policy, Mapping) else None
    )
    expected_primary = {
        "WEBPAGE_P1": configured_map["h"],
        "C_VISIBLE": configured_map["c"],
        "H_PLUS_C_VISIBLE": configured_map["hc"],
    }
    if (
        not isinstance(primary_by_node, Mapping)
        or {node: str(primary_by_node.get(node) or "") for node in expected_primary}
        != expected_primary
    ):
        raise ValueError(
            "primary decision arms must exactly equal the preregistered core-factorial arms"
        )

    semantic_keys = {
        "WEBPAGE_P1": "h",
        "C_VISIBLE": "c",
        "H_PLUS_C_VISIBLE": "hc",
    }
    groups: dict[str, list[tuple[str, str]]] = {}
    for node, semantic_key in semantic_keys.items():
        arm_id = configured_map[semantic_key]
        arm = arms.get(arm_id)
        expected = factorial[semantic_key]
        if not isinstance(arm, Mapping):
            raise ValueError(f"configured primary arm {arm_id!r} is absent from the screen")
        if str(arm.get("page_variant") or "") != str(expected.get("page_variant") or "") or str(
            arm.get("close_variant") or ""
        ) != str(expected.get("close_variant") or ""):
            raise ValueError(f"configured primary arm {arm_id!r} executes different variants")
        page = str(arm["page_variant"])
        close = str(arm["close_variant"])
        active = [item for item in (page, close) if item != "P0"]
        if not active or any(
            bool(variants[item].get("is_control")) or variants[item].get("runnable", True) is False
            for item in active
        ):
            raise ValueError(f"configured primary arm {arm_id!r} is not executable P1")
        variant_id = "+".join(active)
        groups[node] = [(arm_id, variant_id)]

    registry_extensions = [
        variant_id
        for variant_id, variant in variants.items()
        if str(variant.get("node") or "") == "C_REGISTRY"
    ]
    if len(registry_extensions) != 1:
        raise ValueError(
            "variant registry must declare exactly one separately identified C_REGISTRY "
            "extension"
        )
    return groups, registry_extensions[0]


def _ci_lower(ci: Any) -> float | None:
    return _maybe_finite(ci.get("lower")) if isinstance(ci, Mapping) else None


def _selector_gate(
    row: Mapping[str, Any],
    *,
    metric: str,
    threshold: float,
    required: bool,
) -> tuple[dict[str, Any], bool, bool]:
    raw = (row.get("selector_estimands") or {}).get(metric)
    if not isinstance(raw, Mapping):
        result = {
            "status": "MISSING_REQUIRED" if required else "NOT_APPLICABLE",
            "threshold": threshold,
        }
        return result, not required, required
    gate_status = str(raw.get("all_offered_gate_status") or "")
    if gate_status == "NOT_APPLICABLE_NO_ELIGIBLE_DENOMINATOR":
        return (
            {
                "status": gate_status,
                "threshold": threshold,
                "eligible_denominator_rows": 0,
                "eligible_denominator_opportunities": 0,
            },
            True,
            False,
        )
    if gate_status and gate_status != "ESTIMABLE":
        return (
            {
                "status": gate_status,
                "threshold": threshold,
                "eligible_denominator_rows": raw.get("eligible_denominator_rows"),
                "eligible_denominator_opportunities": raw.get("eligible_denominator_opportunities"),
            },
            False,
            True,
        )
    ci = raw.get("all_offered_worst_case")
    lower = _maybe_finite(raw.get("all_offered_lcb"))
    if lower is None:
        lower = _ci_lower(ci)
    if lower is None:
        result = {
            "status": "MISSING_REQUIRED" if required else "NOT_APPLICABLE",
            "threshold": threshold,
            "missing_direct_denominator_count": raw.get("missing_direct_denominator_count"),
        }
        return result, not required, required
    complete = int(raw.get("missing_direct_denominator_count") or 0) == 0
    passed = complete and lower >= threshold
    result = {
        "status": "PASS" if passed else "FAIL",
        "lower_confidence_bound": lower,
        "threshold": threshold,
        "all_offered_complete": complete,
        "eligible_denominator_rows": raw.get("eligible_denominator_rows"),
        "eligible_denominator_opportunities": raw.get("eligible_denominator_opportunities"),
    }
    return result, passed, not complete


def _selector_output_validity(
    row: Mapping[str, Any],
    *,
    invalid_id_max: int,
) -> tuple[dict[str, Any], bool, bool, tuple[str, ...]]:
    """Validate the all-offered normalization summary without conflating repairs with bad IDs.

    Repeated IDs are a post-hoc repair/sensitivity event.  Only the explicit
    ``out_of_set_label`` counter feeds the invalid-ID structural gate.  Conversely, a missing
    or internally inconsistent normalization trace cannot be read as zero invalid IDs.
    """
    raw = row.get("selector_output_validity")
    if not isinstance(raw, Mapping):
        return (
            {
                "status": "NOT_ESTABLISHED",
                "reason": "SELECTOR_OUTPUT_VALIDITY_MISSING",
                "invalid_id_hard_gate": {
                    "status": "NOT_ESTABLISHED",
                    "required_maximum": invalid_id_max,
                },
            },
            False,
            True,
            ("INVALID_ID_HARD_GATE_NOT_ESTABLISHED",),
        )

    snapshot = json.loads(canonical_json(dict(raw)))
    hard = raw.get("invalid_id_hard_gate")
    integer_fields = (
        "selector_attempt_count",
        "strict_valid_count",
        "repaired_attempt_count",
        "rejected_attempt_count",
        "invalid_id_count",
        "normalization_trace_error_count",
    )
    malformed = not isinstance(hard, Mapping)
    counts: dict[str, int] = {}
    for field in integer_fields:
        value = raw.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            malformed = True
        else:
            counts[field] = value

    if isinstance(hard, Mapping):
        hard_count = hard.get("invalid_id_count")
        required_maximum = hard.get("required_maximum")
        duplicates_are_invalid = hard.get("duplicates_are_invalid_ids")
        malformed |= (
            isinstance(hard_count, bool)
            or not isinstance(hard_count, int)
            or hard_count < 0
            or hard_count != counts.get("invalid_id_count")
            or isinstance(required_maximum, bool)
            or not isinstance(required_maximum, int)
            or required_maximum != invalid_id_max
            or duplicates_are_invalid is not False
        )

    if malformed:
        snapshot["decision_status"] = "NOT_ESTABLISHED"
        snapshot["decision_reason"] = "SELECTOR_OUTPUT_VALIDITY_MALFORMED_OR_INCONSISTENT"
        return (
            snapshot,
            False,
            True,
            ("INVALID_ID_HARD_GATE_NOT_ESTABLISHED",),
        )

    invalid_count = counts["invalid_id_count"]
    trace_errors = counts["normalization_trace_error_count"]
    expected_status = (
        "NOT_ESTABLISHED_INCOMPLETE_NORMALIZATION_TRACE"
        if trace_errors
        else "FAIL"
        if invalid_count > invalid_id_max
        else "PASS"
    )
    observed_status = str(hard.get("status") or "")
    if observed_status != expected_status:
        snapshot["decision_status"] = "NOT_ESTABLISHED"
        snapshot["decision_reason"] = "INVALID_ID_HARD_GATE_STATUS_INCONSISTENT_WITH_COUNTS"
        return (
            snapshot,
            False,
            True,
            ("INVALID_ID_HARD_GATE_NOT_ESTABLISHED",),
        )

    strict_rate = raw.get("strict_valid_rate_lcb")
    repair_rate = raw.get("repair_rate_ucb")
    rates_estimable = all(
        isinstance(value, Mapping)
        and str(value.get("status") or "") == "ESTIMABLE"
        and isinstance(value.get("ci"), Mapping)
        for value in (strict_rate, repair_rate)
    )
    snapshot["decision_status"] = expected_status
    snapshot["rate_sensitivity_status"] = "ESTIMABLE" if rates_estimable else "NOT_ESTABLISHED"
    if trace_errors:
        return (
            snapshot,
            False,
            True,
            ("NORMALIZATION_TRACE_INCOMPLETE", "INVALID_ID_HARD_GATE_NOT_ESTABLISHED"),
        )
    if invalid_count > invalid_id_max:
        return (
            snapshot,
            True,
            False,
            (f"INVALID_ID_COUNT_EXCEEDS_{invalid_id_max}",),
        )
    return snapshot, False, False, ()


def _no_repair_quality(row: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the preregistered adverse no-repair sensitivity without making it primary."""
    quality = row.get("quality_effects")
    if not isinstance(quality, Mapping):
        return {}
    output: dict[str, Any] = {}
    for metric, raw in sorted(quality.items()):
        if not isinstance(raw, Mapping):
            continue
        ci = raw.get("no_repair_adverse")
        n_pairs = raw.get("no_repair_adverse_n_pairs")
        dirty_pairs = raw.get("no_repair_adverse_dirty_pairs")
        judge_missing = raw.get("no_repair_adverse_judge_missing_pairs")
        if not isinstance(ci, Mapping):
            output[str(metric)] = {
                "status": "NOT_ESTABLISHED",
                "n_pairs": n_pairs,
                "dirty_pairs": dirty_pairs,
                "judge_missing_pairs": judge_missing,
            }
            continue
        output[str(metric)] = {
            "status": "ESTIMABLE",
            "effect": json.loads(canonical_json(dict(ci))),
            "n_pairs": n_pairs,
            "dirty_pairs": dirty_pairs,
            "judge_missing_pairs": judge_missing,
            "decision_use": "SENSITIVITY_ONLY_STRICT_ALL_OFFERED_REMAINS_PRIMARY",
        }
    return output


def _selector_efficiency_snapshot(
    row: Mapping[str, Any],
    *,
    node_prefixes: Sequence[str],
) -> dict[str, Any]:
    raw_family = row.get("selector_efficiency")
    output: dict[str, Any] = {}
    for prefix in node_prefixes:
        node = prefix.upper()
        raw = raw_family.get(node) if isinstance(raw_family, Mapping) else None
        if not isinstance(raw, Mapping):
            output[node] = {
                "status": "NOT_ESTABLISHED",
                "reason": "SELECTOR_EFFICIENCY_SUMMARY_MISSING",
                "decision_use": "SECONDARY_DESCRIPTIVE_NOT_PRIMARY_CAUSAL_UTILITY",
            }
            continue
        snapshot = json.loads(canonical_json(dict(raw)))
        applicable = raw.get("all_offered_applicable_rows")
        complete = raw.get("token_trace_complete_rows")
        malformed = raw.get("malformed_metric_rows")
        trace_complete = (
            isinstance(applicable, int)
            and not isinstance(applicable, bool)
            and applicable > 0
            and isinstance(complete, int)
            and not isinstance(complete, bool)
            and complete == applicable
            and isinstance(malformed, int)
            and not isinstance(malformed, bool)
            and malformed == 0
        )
        metrics = raw.get("metrics")
        metrics_complete = (
            isinstance(metrics, Mapping)
            and set(metrics)
            == {
                "selected_token_precision",
                "weighted_truth_per_100_rendered_tokens",
                "materialization_ratio",
            }
            and all(
                isinstance(value, Mapping)
                and value.get("status") == "ESTIMABLE"
                and isinstance(value.get("ci"), Mapping)
                for value in metrics.values()
            )
        )
        if raw.get("status") == "ESTIMABLE" and trace_complete and metrics_complete:
            snapshot["decision_status"] = "ESTIMABLE_DESCRIPTIVE"
        else:
            snapshot["decision_status"] = "NOT_ESTABLISHED"
            snapshot["reason"] = "TOKEN_TRACE_INCOMPLETE_OR_SELECTOR_EFFICIENCY_NOT_ESTIMABLE"
        output[node] = snapshot
    return output


def _critical_harm_absolute_snapshot(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    raw = row.get("critical_harm_absolute")
    if not isinstance(raw, Mapping):
        return {
            "decision_status": "NOT_ESTABLISHED",
            "reason": "ABSOLUTE_ALL_OFFERED_CRITICAL_HARM_RATE_MISSING",
        }
    snapshot = json.loads(canonical_json(dict(raw)))
    adverse = raw.get("all_offered_adverse")
    ci = adverse.get("ci") if isinstance(adverse, Mapping) else None
    all_pairs = row.get("all_offered_pairs")
    valid = (
        isinstance(adverse, Mapping)
        and adverse.get("status") == "ESTIMABLE"
        and isinstance(ci, Mapping)
        and all(_maybe_finite(ci.get(field)) is not None for field in ("point", "lower", "upper"))
        and isinstance(all_pairs, int)
        and not isinstance(all_pairs, bool)
        and adverse.get("n_pairs") == all_pairs
        and adverse.get("missing_pairs") == 0
    )
    snapshot["decision_status"] = "ESTIMABLE" if valid else "NOT_ESTABLISHED"
    if not valid:
        snapshot["reason"] = "ABSOLUTE_ALL_OFFERED_CRITICAL_HARM_RATE_INCOMPLETE"
    return snapshot


def _quality_guards(
    row: Mapping[str, Any],
    *,
    margins: Mapping[str, Any],
) -> tuple[dict[str, EffectWithCI], bool, list[str]]:
    effects: dict[str, EffectWithCI] = {}
    blockers: list[str] = []
    passed = True
    all_pairs = int(row.get("all_offered_pairs") or 0)
    quality = row.get("quality_effects")
    if not isinstance(quality, Mapping) or all_pairs <= 0:
        return effects, False, ["STRICT_QUALITY_FAMILY_MISSING"]
    for metric, config_key in _QUALITY_CONFIG_KEYS.items():
        raw = quality.get(metric)
        strict = raw.get("strict") if isinstance(raw, Mapping) else None
        effect = _effect(strict, unit="fraction_treatment_minus_p0")
        n_pairs = int((raw or {}).get("strict_n_pairs") or 0) if isinstance(raw, Mapping) else 0
        missing = (
            int((raw or {}).get("strict_missing_pairs") or 0)
            if isinstance(raw, Mapping)
            else all_pairs
        )
        if effect is None or effect.lower is None or n_pairs != all_pairs or missing != 0:
            passed = False
            blockers.append(f"STRICT_ALL_OFFERED_{metric.upper()}_NOT_ESTIMABLE")
            continue
        margin = _finite(margins.get(config_key), label=config_key) / 100.0
        effects[metric] = effect
        if effect.lower < margin:
            passed = False
            blockers.append(f"{metric.upper()}_NI_LCB_BELOW_{margin:g}")

    critical = quality.get("critical_harm")
    strict_critical = critical.get("strict") if isinstance(critical, Mapping) else None
    critical_effect = _effect(strict_critical, unit="fraction_p0_harm_minus_treatment_harm")
    critical_n = (
        int((critical or {}).get("strict_n_pairs") or 0) if isinstance(critical, Mapping) else 0
    )
    critical_missing = (
        int((critical or {}).get("strict_missing_pairs") or 0)
        if isinstance(critical, Mapping)
        else all_pairs
    )
    if (
        critical_effect is None
        or critical_effect.lower is None
        or critical_n != all_pairs
        or critical_missing != 0
    ):
        passed = False
        blockers.append("STRICT_ALL_OFFERED_CRITICAL_HARM_NOT_ESTIMABLE")
    else:
        effects["critical_harm"] = critical_effect
        maximum = (
            _finite(
                margins.get("critical_harm_risk_pp_max"),
                label="critical_harm_risk_pp_max",
            )
            / 100.0
        )
        if critical_effect.lower < -maximum:
            passed = False
            blockers.append("CRITICAL_HARM_INCREASE_UCB_EXCEEDS_MARGIN")

    for field, config_key in (
        ("terminal_failure_risk_difference", "terminal_failure_risk_pp_max"),
        ("fallback_risk_difference", "fallback_risk_pp_max"),
    ):
        raw = row.get(field)
        ci = raw.get("ci") if isinstance(raw, Mapping) and raw.get("status") == "OK" else None
        upper = _maybe_finite(ci.get("upper")) if isinstance(ci, Mapping) else None
        if upper is None:
            passed = False
            blockers.append(f"{field.upper()}_NOT_ESTIMABLE")
            continue
        maximum = _finite(margins.get(config_key), label=config_key)
        effects[field] = EffectWithCI(
            point=_finite(ci.get("point"), label=f"{field}.point"),
            lower=None,
            upper=upper,
            unit="percentage_points_treatment_minus_p0",
            interval_type=str(ci.get("interval_type") or ""),
            direction=str(ci.get("decision_direction") or ""),
        )
        if upper > maximum:
            passed = False
            blockers.append(f"{field.upper()}_UCB_EXCEEDS_{maximum:g}PP")
    return effects, passed, blockers


def _operational_evidence_for(
    operational: Mapping[str, Any] | None,
    arm_id: str,
    variant_id: str,
    *,
    guards: Mapping[str, Any],
) -> tuple[bool, float, bool, float, list[str]]:
    if operational is None:
        return False, 0.0, False, 0.0, ["OPERATIONAL_CAMPAIGN_NOT_RUN"]
    # Operational evidence is arm-scoped. A batched H02 result cannot be transferred to H05
    # merely because both belong to WEBPAGE_P1.
    raw = (operational.get("arms") or {}).get(arm_id)
    if not isinstance(raw, Mapping) or raw.get("status") != "OK":
        return False, 0.0, False, 0.0, ["OPERATIONAL_ARM_NOT_ESTIMABLE"]
    if str(raw.get("arm_id") or "") != arm_id or str(raw.get("variant_id") or "") != variant_id:
        return False, 0.0, False, 0.0, ["OPERATIONAL_ARM_VARIANT_BINDING_MISMATCH"]
    required = (
        "block_makespan_saving_lcb",
        "throughput_improvement_lcb",
        "p95_latency_ratio_ucb",
        "gpu_joule_ratio_ucb",
        "terminal_failure_increase_ucb_pp",
        "critical_harm_increase_ucb_pp",
        "e2e_speedup_lcb",
    )
    values: dict[str, float] = {}
    try:
        values = {
            key: _finite(raw.get(key), label=f"operational.{arm_id}.{key}") for key in required
        }
    except ValueError:
        return False, 0.0, False, 0.0, ["OPERATIONAL_ARM_MEASUREMENT_INCOMPLETE"]
    quality_pass = (
        values["p95_latency_ratio_ucb"]
        <= _finite(guards.get("p95_latency_ratio_ucb_max"), label="p95_latency_ratio_ucb_max")
        and values["gpu_joule_ratio_ucb"]
        <= _finite(guards.get("gpu_joule_ratio_ucb_max"), label="gpu_joule_ratio_ucb_max")
        and values["terminal_failure_increase_ucb_pp"]
        <= _finite(
            guards.get("terminal_failure_increase_ucb_max_pp"),
            label="terminal_failure_increase_ucb_max_pp",
        )
        and values["critical_harm_increase_ucb_pp"]
        <= _finite(
            guards.get("critical_harm_increase_ucb_max_pp"),
            label="critical_harm_increase_ucb_max_pp",
        )
        and raw.get("quality_guards_pass") is True
    )
    saving_lcb = min(
        values["block_makespan_saving_lcb"],
        values["throughput_improvement_lcb"],
    )
    required_saving = max(
        _finite(guards.get("block_makespan_saving_lcb_min"), label="block_makespan_saving_lcb_min"),
        _finite(
            guards.get("throughput_improvement_lcb_min"), label="throughput_improvement_lcb_min"
        ),
    )
    reasons: list[str] = []
    if saving_lcb < required_saving:
        reasons.append("OPERATIONAL_SAVING_LCB_BELOW_FROZEN_GUARD")
    if not quality_pass:
        reasons.append("OPERATIONAL_QUALITY_OR_RELIABILITY_GUARD_FAILED")
    return True, saving_lcb, quality_pass, values["e2e_speedup_lcb"], reasons


_EXPLORATORY_ELIGIBILITY_SCHEMA = "e2e_eligibility_result_v3"
_EXPLORATORY_ELIGIBILITY_SCOPE = "EXPLORATORY_TASK_LEVEL_PRETREATMENT_FORMATIVE_ONLY"


def _exploratory_eligibility_snapshot(
    effects: Mapping[str, Any],
    *,
    decision_config: Mapping[str, Any],
) -> tuple[str, dict[str, Any], tuple[str, ...]]:
    """Normalize the frozen target family for reporting, never for verdict construction.

    The producer owns target estimation.  The finalizer only verifies that a v3 result is a
    complete, uniquely named family and snapshots its canonical bytes.  In particular, this
    function deliberately returns no keep flag, coverage threshold, or conditional evidence
    that could leak an exploratory rule into :func:`decide`.
    """

    eligibility = effects.get("eligibility")
    limitations = (
        "ELIGIBILITY_IS_EXPLORATORY_TASK_LEVEL_PRETREATMENT_FORMATIVE_ONLY",
        "ELIGIBILITY_DOES_NOT_CHANGE_PRIMARY_VERDICTS_CHAMPIONS_OR_MECHANISM_ATTRIBUTION",
        "TASK_LEVEL_RULES_DO_NOT_DEFINE_INVOCATION_COVERAGE_OR_A_DEPLOYMENT_ENVELOPE",
        "NULL_UNSTABLE_AND_NOT_ESTIMABLE_TARGETS_ARE_REPORTED_WITHOUT_BEST_TARGET_SELECTION",
    )
    if not isinstance(eligibility, Mapping):
        return (
            "WHEN_USEFUL_NOT_ESTABLISHED_TASK_LEVEL_ONLY",
            {},
            limitations
            + (
                "WHEN_USEFUL_NOT_ESTABLISHED_TASK_LEVEL_ONLY",
                "TASK_LEVEL_ELIGIBILITY_RESULT_NOT_PRESENT",
            ),
        )
    schema = str(eligibility.get("schema_version") or "")
    if schema != _EXPLORATORY_ELIGIBILITY_SCHEMA:
        return (
            "WHEN_USEFUL_NOT_ESTABLISHED_TASK_LEVEL_ONLY",
            {},
            limitations
            + (
                "WHEN_USEFUL_NOT_ESTABLISHED_TASK_LEVEL_ONLY",
                "VALID_V3_EXPLORATORY_ELIGIBILITY_FAMILY_NOT_BOUND",
                f"OBSERVED_ELIGIBILITY_SCHEMA={schema or 'ABSENT'}",
            ),
        )
    scope = str(eligibility.get("analysis_scope") or "")
    if scope != _EXPLORATORY_ELIGIBILITY_SCOPE:
        raise ValueError("v3 eligibility result has a non-formative or non-task-level scope")

    raw_targets = eligibility.get("target_results")
    targets: list[dict[str, Any]] = []
    if isinstance(raw_targets, Mapping):
        for raw_id, raw_target in raw_targets.items():
            if not isinstance(raw_target, Mapping):
                raise ValueError("eligibility target result must be an object")
            target = json.loads(canonical_json(dict(raw_target)))
            embedded_id = str(target.get("target_id") or raw_id)
            if str(raw_id) != embedded_id and target.get("target_id") is not None:
                raise ValueError("eligibility target key and embedded target_id disagree")
            target["target_id"] = embedded_id
            targets.append(target)
    elif isinstance(raw_targets, Sequence) and not isinstance(raw_targets, str | bytes):
        for raw_target in raw_targets:
            if not isinstance(raw_target, Mapping):
                raise ValueError("eligibility target result must be an object")
            targets.append(json.loads(canonical_json(dict(raw_target))))
    else:
        raise ValueError("v3 eligibility result has no target_results family")
    target_ids = [str(target.get("target_id") or "") for target in targets]
    if not targets or "" in target_ids or len(target_ids) != len(set(target_ids)):
        raise ValueError("v3 eligibility target_results must be non-empty and uniquely named")
    if any(
        not str(target.get("status") or target.get("finding_status") or "")
        for target in targets
    ):
        raise ValueError("every eligibility target must report a status, including null targets")

    learner = decision_config.get("eligibility_learner")
    configured_targets = learner.get("targets") if isinstance(learner, Mapping) else None
    expected_ids: list[str] = []
    if isinstance(configured_targets, Mapping):
        for raw_id, configured in configured_targets.items():
            if not isinstance(configured, Mapping):
                raise ValueError("frozen eligibility target is malformed")
            embedded_id = str(configured.get("target_id") or raw_id)
            if configured.get("target_id") is not None and embedded_id != str(raw_id):
                raise ValueError("frozen eligibility target key and target_id disagree")
            expected_ids.append(embedded_id)
    elif isinstance(configured_targets, Sequence) and not isinstance(
        configured_targets, str | bytes
    ):
        for configured in configured_targets:
            if not isinstance(configured, Mapping) or not str(configured.get("target_id") or ""):
                raise ValueError("frozen eligibility target is malformed")
            expected_ids.append(str(configured["target_id"]))
    else:
        raise ValueError("frozen eligibility target family is absent")
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("frozen eligibility target ids are not unique")
    if set(target_ids) != set(expected_ids):
        raise ValueError("eligibility result does not report every frozen target exactly once")

    snapshot = json.loads(canonical_json(dict(eligibility)))
    snapshot["target_results"] = sorted(targets, key=lambda target: str(target["target_id"]))
    snapshot["decision_use"] = (
        "EXPLORATORY_REPORTING_ONLY_NO_PRIMARY_VERDICT_OR_DEPLOYMENT_ENVELOPE"
    )
    return (
        "EXPLORATORY_TASK_LEVEL_PRETREATMENT_FORMATIVE_ONLY",
        snapshot,
        limitations,
    )


def _structured_increment_gates(
    matched: Mapping[str, Any] | None,
    *,
    decision_config: Mapping[str, Any],
    primary_arm_by_node: Mapping[str, str],
    matched_design: Mapping[str, Any],
    variants: Mapping[str, Mapping[str, Any]],
    arms: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Read an explicit multiplicity-controlled control composite, never infer one ad hoc.

    The matched producer must emit the preregistered six-member Holm composite from its paired
    task/source-cluster resamples. Standalone H, standalone C, and joint H+C each require their
    own CPU and prose controls; one node's gate cannot stand in for another. Inspecting ordinary
    contrast CIs after the run is not a substitute.
    """
    config = decision_config.get("structured_increment")
    if not isinstance(config, Mapping) or config.get("required_for_positive_p1_claim") is not True:
        raise ValueError("structured-increment attribution must be required for positive P1")
    schema = str(config.get("schema_version") or "")
    family_policy = config.get("multiplicity_family")
    control_estimands = config.get("control_estimands")
    compound_policy = config.get("compound_claim")
    if (
        schema != "structured_increment_gate_v3"
        or not isinstance(family_policy, Mapping)
        or not isinstance(control_estimands, Mapping)
        or set(control_estimands) != {"LLM_vs_CPU", "structured_selection_vs_prose"}
        or not isinstance(compound_policy, Mapping)
        or compound_policy.get("requires_both_control_types") is not True
        or compound_policy.get("component_claims_reported_separately") is not True
        or compound_policy.get("joint_attribution_status")
        != "IDENTIFIED_BY_PREREGISTERED_JOINT_MATCHED_CONTROLS"
    ):
        raise ValueError("structured-increment policy is incomplete")
    family_id = str(family_policy.get("family_id") or "")
    method = str(family_policy.get("method") or "")
    alpha = _finite(
        family_policy.get("familywise_alpha"),
        label="structured_increment.multiplicity_family.familywise_alpha",
    )
    family_members = tuple(map(str, family_policy.get("member_contrast_ids") or ()))
    if (
        not family_id
        or method != "holm_one_sided_across_six_primary_tests"
        or not (0.0 < alpha < 1.0)
        or len(family_members) != 6
        or len(set(family_members)) != 6
    ):
        raise ValueError("structured-increment six-contrast Holm family is incomplete")
    llm_policy = control_estimands["LLM_vs_CPU"]
    prose_policy = control_estimands["structured_selection_vs_prose"]
    if (
        not isinstance(llm_policy, Mapping)
        or llm_policy.get("primary_hypothesis")
        != "strict_weighted_required_atom_recall_superiority"
        or _finite(llm_policy.get("adjusted_lcb_min_pp"), label="LLM adjusted_lcb_min_pp") <= 0
        or _finite(
            llm_policy.get("service_work_increase_ucb_max"),
            label="LLM service_work_increase_ucb_max",
        )
        < 0
        or llm_policy.get("all_strict_quality_ni_guards_required") is not True
        or not isinstance(prose_policy, Mapping)
        or prose_policy.get("primary_hypothesis") != "complete_service_work_saving_superiority"
        or _finite(prose_policy.get("adjusted_lcb_min"), label="prose adjusted_lcb_min") <= 0
        or prose_policy.get("all_strict_quality_ni_guards_required") is not True
        or prose_policy.get("prose_control_integrity_required") is not True
        or prose_policy.get(
            "raw_contract_adherence_sensitivity_required_for_pointer_only_attribution"
        )
        is not True
    ):
        raise ValueError("structured-increment control estimands are incomplete")
    by_node = config.get("by_node")
    if not isinstance(by_node, Mapping):
        raise ValueError("structured-increment node policy is absent")
    nodes = ("WEBPAGE_P1", "C_VISIBLE", "H_PLUS_C_VISIBLE")
    declared_component_members = {
        str(contrast_id)
        for node in nodes
        for contrast_id in ((by_node.get(node) or {}).get("required_contrast_ids") or ())
        if isinstance(by_node.get(node), Mapping)
    }
    if declared_component_members != set(family_members):
        raise ValueError(
            "structured-increment Holm family must cover exactly both controls for H, C, and H+C"
        )

    def unavailable(node: str, reason: str) -> dict[str, Any]:
        return {
            "status": "NOT_ESTABLISHED",
            "node": node,
            "reason": reason,
            "decision_use": "REQUIRED_FOR_POSITIVE_P1_MECHANISM_ATTRIBUTION",
        }

    if str(matched_design.get("version") or "") != "matched_contrasts_v2":
        raise ValueError("frozen matched-contrast design is absent or unsupported")
    executable_fields = tuple(map(str, matched_design.get("executable_variant_fields") or ()))
    design_rows = {
        str(item.get("contrast_id") or ""): dict(item)
        for item in (matched_design.get("pairs") or ())
        if isinstance(item, Mapping) and str(item.get("contrast_id") or "")
    }
    requested_arms = {
        str(row.get(side) or "")
        for row in design_rows.values()
        for side in ("left_arm_id", "right_arm_id")
    }
    expected_arm_variants: dict[str, dict[str, Any]] = {}
    for arm_id in sorted(requested_arms):
        arm = arms.get(arm_id)
        if not isinstance(arm, Mapping):
            raise ValueError(f"matched-control arm {arm_id!r} is absent from the frozen screen")
        expected_arm_variants[arm_id] = resolve_arm_semantics(
            arm_id,
            arm,
            variants,
            executable_fields,
        )
    expected_design_sha = sha256_hex(canonical_json(dict(matched_design)))
    expected_variants_sha = sha256_hex(canonical_json(expected_arm_variants))

    gates: dict[str, dict[str, Any]] = {}
    raw_composite = (
        matched.get("structured_increment_gates") if isinstance(matched, Mapping) else None
    )
    raw_by_node = raw_composite.get("by_node") if isinstance(raw_composite, Mapping) else None
    input_provenance = matched.get("input_provenance") if isinstance(matched, Mapping) else None
    semantic_inputs_bound = (
        isinstance(input_provenance, Mapping)
        and str(input_provenance.get("matched_contrasts_sha256") or "") == expected_design_sha
        and str(input_provenance.get("arm_variants_sha256") or "") == expected_variants_sha
    )
    top_level_valid = (
        isinstance(raw_composite, Mapping)
        and str(raw_composite.get("schema_version") or "") == schema
        and str(raw_composite.get("policy_sha256") or "")
        == sha256_hex(canonical_json(dict(config)))
        and raw_composite.get("multiplicity_family") == dict(family_policy)
        and isinstance(raw_by_node, Mapping)
        and semantic_inputs_bound
    )
    contrasts = {
        str(item.get("contrast_id") or ""): item
        for item in ((matched or {}).get("contrasts") or ())
        if isinstance(item, Mapping)
    }

    for node in nodes:
        policy = by_node.get(node)
        if not isinstance(policy, Mapping):
            raise ValueError(f"structured-increment policy is absent for {node}")
        if (
            policy.get("decision_use")
            != "REQUIRED_FOR_POSITIVE_P1_MECHANISM_ATTRIBUTION"
        ):
            raise ValueError(f"structured-increment policy for {node} is not decision gating")
        if node == "H_PLUS_C_VISIBLE" and policy.get(
            "joint_matched_control_status"
        ) != "IMPLEMENTED":
            raise ValueError("joint H+C matched controls are not declared implemented")
        primary_arm = str(policy.get("primary_arm_id") or "")
        if primary_arm != str(primary_arm_by_node.get(node) or ""):
            raise ValueError(f"structured-increment policy targets another primary arm for {node}")
        required_ids = tuple(map(str, policy.get("required_contrast_ids") or ()))
        if len(required_ids) != 2 or len(set(required_ids)) != 2:
            raise ValueError(f"structured-increment policy for {node} needs two unique controls")
        if not set(required_ids) <= set(family_members):
            raise ValueError(f"structured-increment policy for {node} is outside the Holm family")
        if not top_level_valid:
            gates[node] = unavailable(
                node,
                "MULTIPLICITY_CONTROLLED_STRUCTURED_INCREMENT_COMPOSITE_NOT_EMITTED",
            )
            continue
        raw = raw_by_node.get(node)
        if not isinstance(raw, Mapping):
            gates[node] = unavailable(node, "STRUCTURED_INCREMENT_NODE_GATE_MISSING")
            continue
        raw_ids = tuple(map(str, raw.get("required_contrast_ids") or ()))
        component_gates = raw.get("component_gates")
        contrast_rows_valid = True
        for contrast_id in required_ids:
            observed = contrasts.get(contrast_id)
            declared = design_rows.get(contrast_id)
            if not isinstance(observed, Mapping) or not isinstance(declared, Mapping):
                contrast_rows_valid = False
                break
            fields = (
                "contrast_id",
                "kind",
                "left_arm_id",
                "right_arm_id",
                "target_factor",
                "left_level",
                "right_level",
                "factor_scope",
                "affected_nodes",
                "first_treatment_boundary",
            )
            if (
                any(observed.get(field) != declared.get(field) for field in fields)
                or set(map(str, observed.get("factor_fields") or ()))
                != set(map(str, declared.get("factor_fields") or ()))
                or str(observed.get("left_arm_id") or "") != primary_arm
                or str(observed.get("pairing_status") or "") != "OK"
                or not isinstance(observed.get("semantic_validation"), Mapping)
                or str(observed["semantic_validation"].get("status") or "") != "OK"
            ):
                contrast_rows_valid = False
                break
        by_kind = {
            str(design_rows[contrast_id].get("kind") or ""): contrast_id
            for contrast_id in required_ids
            if contrast_id in design_rows
        }
        component_rows_pass = (
            set(by_kind) == {"LLM_vs_CPU", "structured_selection_vs_prose"}
            and isinstance(component_gates, Mapping)
            and set(map(str, component_gates)) == {"LLM_vs_CPU", "structured_selection_vs_prose"}
        )
        if component_rows_pass:
            for control_type, contrast_id in by_kind.items():
                component = component_gates.get(control_type)
                estimand_policy = control_estimands[control_type]
                expected_policy_sha = sha256_hex(canonical_json(dict(estimand_policy)))
                common_pass = (
                    isinstance(component, Mapping)
                    and component.get("status") == "ESTABLISHED"
                    and component.get("contrast_id") == contrast_id
                    and component.get("holm_family_id") == family_id
                    and component.get("estimand_contract_sha256") == expected_policy_sha
                    and component.get("adjusted_primary_gate_pass") is True
                    and component.get("all_strict_quality_ni_guards_pass") is True
                    and component.get("first_boundary_input_comparability_pass") is True
                )
                if control_type == "LLM_vs_CPU":
                    common_pass = (
                        common_pass and component.get("service_work_cost_guard_pass") is True
                    )
                else:
                    common_pass = (
                        common_pass
                        and component.get("prose_control_integrity_gate_pass") is True
                        and component.get("main_estimand_scope")
                        == "STRUCTURED_ID_POLICY_VS_BOUNDED_SHORT_PROSE_POLICY"
                        and component.get("pointer_only_attribution_status")
                        in {"ESTABLISHED", "NOT_ESTABLISHED"}
                        and isinstance(
                            component.get("raw_contract_quality_ni_guards"),
                            Mapping,
                        )
                        and isinstance(
                            component.get("raw_contract_work_sensitivity"),
                            Mapping,
                        )
                    )
                component_rows_pass = component_rows_pass and common_pass
        established = (
            str(raw.get("status") or "") == "ESTABLISHED"
            and str(raw.get("primary_arm_id") or "") == primary_arm
            and set(raw_ids) == set(required_ids)
            and raw.get("compound_status") == "ESTABLISHED"
            and contrast_rows_valid
            and component_rows_pass
        )
        snapshot = json.loads(canonical_json(dict(raw)))
        snapshot["node"] = node
        snapshot["decision_use"] = "REQUIRED_FOR_POSITIVE_P1_MECHANISM_ATTRIBUTION"
        prose_component = (
            component_gates.get("structured_selection_vs_prose")
            if isinstance(component_gates, Mapping)
            else None
        )
        snapshot["pointer_only_attribution_status"] = (
            str(
                (prose_component or {}).get(
                    "pointer_only_attribution_status"
                )
                or "NOT_ESTABLISHED"
            )
            if isinstance(prose_component, Mapping)
            else "NOT_ESTABLISHED"
        )
        snapshot["attribution_scope"] = (
            "LLM_PLUS_STRUCTURED_POINTER_REPRESENTATION"
            if snapshot["pointer_only_attribution_status"] == "ESTABLISHED"
            else "LLM_PLUS_STRUCTURED_BOUNDED_POLICY_ONLY"
        )
        if established:
            snapshot["status"] = "ESTABLISHED"
        else:
            snapshot["status"] = "NOT_ESTABLISHED"
            snapshot["reason"] = (
                "CONTROL_COMPOSITE_MISSING_FAILED_OR_INCONSISTENT; "
                "LLM_INCREMENT_AND_STRUCTURED_POINTER_INCREMENT_REMAIN_SEPARATE; "
                "P1_VS_P0_BENEFIT_CANNOT_BE_ATTRIBUTED_TO_COMPOUND_LLM_PLUS_STRUCTURED_PATH"
            )
        gates[node] = snapshot
    return gates


def _assess_arm(
    arm_id: str,
    variant_id: str,
    row: Mapping[str, Any] | None,
    *,
    node_prefixes: Sequence[str],
    decision_config: Mapping[str, Any],
    operational: Mapping[str, Any] | None,
    conditional: tuple[bool, float],
    structured_increment: Mapping[str, Any],
) -> _ArmAssessment:
    utility = decision_config["utility"]
    selector = decision_config["selector"]
    structural = decision_config["structural"]
    quality_margins = decision_config["quality_ni_margin"]
    coverage = decision_config["coverage"]
    thesis = decision_config["thesis"]
    op_guards = decision_config["operational_guards"]
    minimum_saving = _finite(
        utility.get("minimum_meaningful_work_reduction"),
        label="minimum_meaningful_work_reduction",
    )
    reasons: list[str] = []
    blockers: list[str] = []
    if not isinstance(row, Mapping):
        evidence = NodeEvidence(
            structural_pass=True,
            deterministic_harm=False,
            quality_guards_pass=False,
            isolated_saving_lcb=0.0,
            saving_ucb=1.0,
            operational_available=False,
            operational_saving_lcb=0.0,
            operational_quality_pass=False,
            min_saving=minimum_saving,
            thesis_speedup=_finite(thesis.get("e2e_speedup_lcb_min"), label="e2e_speedup_lcb_min"),
            coverage_min=_finite(
                coverage.get("conditional_task_exposure_coverage_lcb_min"),
                label="conditional_task_exposure_coverage_lcb_min",
            ),
            study_valid=False,
        )
        return _ArmAssessment(
            arm_id=arm_id,
            variant_id=variant_id,
            evidence=evidence,
            verdict=Verdict.NOT_ESTABLISHED,
            work=None,
            quality={},
            selector_gates={},
            selector_output_validity={},
            selector_efficiency={},
            no_repair_quality={},
            critical_harm_absolute={},
            structured_increment=json.loads(canonical_json(dict(structured_increment))),
            reasons=(),
            blockers=("ASSIGNED_ARM_MEASUREMENT_MISSING",),
        )

    work_raw = row.get("work_saving")
    work = _effect(work_raw, unit="fraction_complete_service_work_saved")
    work_lcb = (
        _maybe_finite((work_raw or {}).get("lower")) if isinstance(work_raw, Mapping) else None
    )
    work_ucb = (
        _maybe_finite((work_raw or {}).get("upper")) if isinstance(work_raw, Mapping) else None
    )
    if (
        work is None
        or work_lcb is None
        or work_ucb is None
        or int(row.get("work_pairs_missing") or 0) != 0
        or int(row.get("work_pairs_observed") or 0) != int(row.get("all_offered_pairs") or -1)
    ):
        blockers.append("ALL_OFFERED_COMPLETE_WORK_NOT_ESTIMABLE")

    quality_effects, quality_pass, quality_blockers = _quality_guards(row, margins=quality_margins)
    blockers.extend(quality_blockers)

    selector_results: dict[str, dict[str, Any]] = {}
    selector_pass = True
    selector_incomplete = False
    critical_miss = False
    for prefix in node_prefixes:
        # selector_conditional_recall is a set-count ratio.  The frozen threshold explicitly
        # says *weighted* recall, so treating the former as the latter would silently weaken
        # the gate.  The evaluator must emit the genuine weight-aware direct estimand.
        metric = f"{prefix}_weighted_evidence_recall"
        result, passed, incomplete = _selector_gate(
            row,
            metric=metric,
            threshold=_finite(selector.get("weighted_recall_min"), label="weighted_recall_min"),
            required=True,
        )
        selector_results[metric] = result
        selector_pass &= passed
        selector_incomplete |= incomplete

        unweighted_metric = f"{prefix}_selector_conditional_recall"
        unweighted = (row.get("selector_estimands") or {}).get(unweighted_metric)
        selector_results[unweighted_metric] = {
            "status": (
                str(unweighted.get("all_offered_gate_status") or "")
                if isinstance(unweighted, Mapping)
                else "MISSING_DESCRIPTIVE_ESTIMAND"
            ),
            "decision_use": ("DESCRIPTIVE_ONLY_NOT_A_SUBSTITUTE_FOR_WEIGHTED_SELECTOR_RECALL"),
            "all_offered_lcb": (
                unweighted.get("all_offered_lcb") if isinstance(unweighted, Mapping) else None
            ),
        }

        critical_metric = f"{prefix}_critical_truth_recall"
        result, passed, incomplete = _selector_gate(
            row, metric=critical_metric, threshold=1.0, required=True
        )
        selector_results[critical_metric] = result
        selector_pass &= passed
        selector_incomplete |= incomplete
        if result.get("status") == "FAIL":
            raw = (row.get("selector_estimands") or {}).get(critical_metric) or {}
            point = _maybe_finite((raw.get("all_offered_worst_case") or {}).get("point"))
            critical_miss |= point is not None and point < (
                1.0
                - _finite(structural.get("critical_item_miss_max"), label="critical_item_miss_max")
            )

        for suffix, threshold_key in _DIRECT_CONDITIONAL_GATES:
            conditional_metric = f"{prefix}_{suffix}"
            result, passed, incomplete = _selector_gate(
                row,
                metric=conditional_metric,
                threshold=_finite(selector.get(threshold_key), label=threshold_key),
                required=True,
            )
            selector_results[conditional_metric] = result
            selector_pass &= passed
            selector_incomplete |= incomplete

    if not selector_pass:
        blockers.append("SELECTOR_ABSOLUTE_GATE_FAILED")
    if selector_incomplete:
        blockers.append("SELECTOR_REQUIRED_MEASUREMENT_INCOMPLETE")

    invalid_id_max_float = _finite(
        structural.get("invalid_id_max"),
        label="invalid_id_max",
    )
    if not invalid_id_max_float.is_integer() or invalid_id_max_float < 0:
        raise ValueError("invalid_id_max must be a nonnegative integer")
    selector_validity, invalid_id_failure, validity_incomplete, validity_blockers = (
        _selector_output_validity(
            row,
            invalid_id_max=int(invalid_id_max_float),
        )
    )
    blockers.extend(validity_blockers)
    no_repair_quality = _no_repair_quality(row)
    critical_harm_absolute = _critical_harm_absolute_snapshot(row)
    selector_efficiency = _selector_efficiency_snapshot(
        row,
        node_prefixes=node_prefixes,
    )

    trace_invalid = int(row.get("direct_trace_invalid_count") or 0)
    known_structural_failure = (
        trace_invalid > _finite(structural.get("lineage_error_max"), label="lineage_error_max")
        or invalid_id_failure
    )
    measurement_invalid = (
        int(row.get("paired_invalid_block_count") or 0) > 0
        or int(row.get("direct_trace_unavailable_count") or 0) > 0
        or int(row.get("selector_guard_unavailable_count") or 0) > 0
        or validity_incomplete
        or bool(
            blockers
            and any(
                "NOT_ESTIMABLE" in blocker or "INCOMPLETE" in blocker or "MISSING" in blocker
                for blocker in blockers
            )
        )
    )
    machine_ready = row.get("machine_estimands_ready") is True
    study_valid = machine_ready and not measurement_invalid
    if known_structural_failure and not measurement_invalid:
        # A verified trace integrity violation is itself decisive structural evidence even
        # though downstream quality/work estimands may be unavailable as a consequence.
        study_valid = True
    if not study_valid:
        blockers.append("MACHINE_ESTIMANDS_OR_PROVENANCE_INCOMPLETE")

    op_available, op_saving, op_quality, speedup, op_reasons = _operational_evidence_for(
        operational, arm_id, variant_id, guards=op_guards
    )
    reasons.extend(op_reasons)
    conditional_keep, coverage_lcb = conditional
    evidence = NodeEvidence(
        structural_pass=not known_structural_failure,
        deterministic_harm=critical_miss,
        quality_guards_pass=quality_pass and selector_pass,
        isolated_saving_lcb=work_lcb if work_lcb is not None else 0.0,
        saving_ucb=work_ucb if work_ucb is not None else 1.0,
        operational_available=op_available,
        operational_saving_lcb=op_saving,
        operational_quality_pass=op_quality,
        e2e_speedup_lcb=speedup,
        conditional_rule_keeps=conditional_keep,
        coverage_lcb=coverage_lcb,
        min_saving=minimum_saving,
        thesis_speedup=_finite(thesis.get("e2e_speedup_lcb_min"), label="e2e_speedup_lcb_min"),
        coverage_min=_finite(
            coverage.get("conditional_task_exposure_coverage_lcb_min"),
            label="conditional_task_exposure_coverage_lcb_min",
        ),
        study_valid=study_valid,
    )
    verdict = decide(evidence)
    attribution_downgraded = False
    structured_snapshot = json.loads(canonical_json(dict(structured_increment)))
    if verdict in _POSITIVE_VERDICTS and structured_snapshot.get("status") != "ESTABLISHED":
        # Preserve the observed P1-vs-P0 work/quality result in the arm evidence, but do not
        # relabel generic compression as the structured ShapeFlow/P1 contribution.
        reasons.append(
            "P1_VS_P0_BENEFIT_OBSERVED_BUT_STRUCTURED_MECHANISM_ATTRIBUTION_NOT_ESTABLISHED"
        )
        blockers.append("STRUCTURED_INCREMENT_CONTROL_COMPOSITE_NOT_ESTABLISHED")
        verdict = Verdict.NOT_ESTABLISHED
        attribution_downgraded = True
    if verdict is Verdict.MECHANISM_ONLY:
        reasons.append("CAUSAL_STRICT_QUALITY_AND_WORK_PASS_BUT_OPERATIONAL_KEEP_IS_UNAVAILABLE")
    elif verdict is Verdict.KILL_NO_HEADROOM:
        reasons.append("QUALITY_PASSES_AND_WORK_SAVING_UCB_IS_BELOW_MINIMUM_MEANINGFUL_EFFECT")
    elif verdict is Verdict.NOT_ESTABLISHED and not attribution_downgraded:
        reasons.append("NO_PREREGISTERED_POSITIVE_OR_NEGATIVE_BOUND_WAS_ESTABLISHED")
    return _ArmAssessment(
        arm_id=arm_id,
        variant_id=variant_id,
        evidence=evidence,
        verdict=verdict,
        work=work,
        quality=quality_effects,
        selector_gates=selector_results,
        selector_output_validity=selector_validity,
        selector_efficiency=selector_efficiency,
        no_repair_quality=no_repair_quality,
        critical_harm_absolute=critical_harm_absolute,
        structured_increment=structured_snapshot,
        reasons=tuple(dict.fromkeys(reasons)),
        blockers=tuple(dict.fromkeys(blockers)),
    )


def _node_from_assessments(
    node: str,
    assessments: Sequence[_ArmAssessment],
    *,
    where_it_works: tuple[str, ...] = (),
    eligibility_limitations: tuple[str, ...] = (),
    formative: bool,
    simplicity_order: Sequence[str],
) -> NodeDecision:
    if not assessments:
        return NodeDecision(
            node=node,
            verdict=Verdict.NOT_ESTABLISHED,
            champion_variant=None,
            work_saving=None,
            reasons=("NO_EXECUTED_NON_CONTROL_VARIANT_FOR_THIS_NODE",),
            limitations=eligibility_limitations,
            claim_level="NOT_ESTABLISHED",
        )
    candidates = [item.variant_id for item in assessments]
    order = list(map(str, simplicity_order))
    if len(order) != len(set(order)) or set(order) != set(candidates):
        raise ValueError(
            f"frozen champion simplicity order for {node} must name exactly its "
            "executable non-control candidates"
        )
    simplicity_rank = {variant_id: index for index, variant_id in enumerate(order)}
    eligible = [
        assessment
        for assessment in assessments
        if assessment.evidence.study_valid
        and assessment.evidence.structural_pass
        and not assessment.evidence.deterministic_harm
        and assessment.evidence.quality_guards_pass
        and assessment.work is not None
    ]
    if eligible:
        claim_rank = {
            Verdict.THESIS_GRADE: 6,
            Verdict.KEEP: 5,
            Verdict.CONDITIONAL: 4,
            Verdict.MECHANISM_ONLY: 3,
            Verdict.KILL_NO_HEADROOM: 2,
            Verdict.NOT_ESTABLISHED: 1,
            Verdict.KILL_HARM: 0,
            Verdict.KILL_STRUCTURAL: 0,
        }
        champion = max(
            eligible,
            key=lambda item: (
                claim_rank[item.verdict],
                item.evidence.isolated_saving_lcb,
                -simplicity_rank[item.variant_id],
            ),
        )
        verdict = champion.verdict
    else:
        champion = None
        verdicts = {item.verdict for item in assessments}
        # A family-level kill is claimed only when every executable non-control variant reaches
        # that same negative bound.  One bad variant does not kill the node's other designs.
        if verdicts == {Verdict.KILL_STRUCTURAL}:
            verdict = Verdict.KILL_STRUCTURAL
        elif verdicts == {Verdict.KILL_HARM}:
            verdict = Verdict.KILL_HARM
        elif verdicts == {Verdict.KILL_NO_HEADROOM}:
            verdict = Verdict.KILL_NO_HEADROOM
        else:
            verdict = Verdict.NOT_ESTABLISHED
    reasons = (
        champion.reasons
        if champion is not None
        else ("NO_STRUCTURALLY_AND_QUALITY_VALID_VARIANT_ESTABLISHED_A_CHAMPION",)
    )
    limitations = list(eligibility_limitations)
    if formative:
        limitations.append(
            "FORMATIVE_MACHINE_AUTHORED_CORPUS_DOES_NOT_SUPPORT_A_CONFIRMATORY_CLAIM"
        )
    if champion is None:
        limitations.extend(blocker for item in assessments for blocker in item.blockers)
    claim_level = {
        Verdict.MECHANISM_ONLY: "FORMATIVE_MECHANISM_CANDIDATE",
        Verdict.KEEP: "FORMATIVE_OPERATIONAL_CANDIDATE" if formative else "OPERATIONAL_KEEP",
        Verdict.THESIS_GRADE: "FORMATIVE_OPERATIONAL_CANDIDATE" if formative else "THESIS_GRADE",
        Verdict.CONDITIONAL: "FORMATIVE_CONDITIONAL_CANDIDATE" if formative else "CONDITIONAL_KEEP",
        Verdict.KILL_NO_HEADROOM: "NEGATIVE_BOUND",
        Verdict.KILL_HARM: "HARM_BOUND",
        Verdict.KILL_STRUCTURAL: "STRUCTURAL_BOUND",
        Verdict.NOT_ESTABLISHED: "NOT_ESTABLISHED",
    }[verdict]
    return NodeDecision(
        node=node,
        verdict=verdict,
        champion_variant=champion.variant_id if champion else None,
        work_saving=champion.work if champion else None,
        coverage=(
            EffectWithCI(
                champion.evidence.coverage_lcb,
                champion.evidence.coverage_lcb,
                champion.evidence.coverage_lcb,
                "fraction_task_exposure",
            )
            if champion and champion.evidence.conditional_rule_keeps
            else None
        ),
        quality_effects=champion.quality if champion else {},
        critical_harm_rate=(
            _maybe_finite(
                (
                    (champion.critical_harm_absolute.get("all_offered_adverse") or {}).get("ci")
                    or {}
                ).get("point")
            )
            if champion and champion.critical_harm_absolute.get("decision_status") == "ESTIMABLE"
            else None
        ),
        critical_harm_absolute=(champion.critical_harm_absolute if champion else {}),
        where_it_works=where_it_works,
        where_it_fails=tuple(
            dict.fromkeys(blocker for item in assessments for blocker in item.blockers)
        ),
        candidate_variants=tuple(item.variant_id for item in assessments),
        reasons=tuple(dict.fromkeys(reasons)),
        limitations=tuple(dict.fromkeys(limitations)),
        arm_results={
            item.arm_id: item.obj() for item in sorted(assessments, key=lambda value: value.arm_id)
        },
        claim_level=claim_level,
    )


def _verify_operational(
    raw: Mapping[str, Any] | None,
    *,
    itt: Mapping[str, Any],
    e2e: Mapping[str, Any],
    config_sha256: Mapping[str, str],
    producer_policy: Mapping[str, Any],
) -> dict[str, Any] | None:
    if raw is None:
        return None
    if producer_policy.get("implementation_status") != "IMPLEMENTED_VERIFIED":
        raise ValueError(
            "operational evidence producer is NOT_IMPLEMENTED; a consumer-shaped JSON "
            "cannot upgrade the causal result"
        )
    body = _verified(
        raw,
        label="operational evidence",
        schemas="operational_decision_evidence_v1",
    )
    for field in ("run_id", "phase_id", "evaluation_scope_sha256"):
        if not str(body.get(field) or ""):
            raise ValueError(f"operational evidence lacks its own {field}")
    _require_sha256(
        body.get("evaluation_scope_sha256"),
        label="operational evaluation_scope_sha256",
    )
    parents = body.get("parent_causal_evidence_sha256")
    if not isinstance(parents, Mapping) or {
        "ITT_VERDICT_INPUTS": str(parents.get("ITT_VERDICT_INPUTS") or ""),
        "E2E_ANALYSIS": str(parents.get("E2E_ANALYSIS") or ""),
    } != {
        "ITT_VERDICT_INPUTS": str(itt["content_sha256"]),
        "E2E_ANALYSIS": str(e2e["content_sha256"]),
    }:
        raise ValueError(
            "operational evidence does not bind the exact parent causal evidence hashes"
        )
    _require_sha256(body.get("policy_sha256"), label="operational policy_sha256")
    _require_sha256(body.get("taskset_sha256"), label="operational taskset_sha256")
    observed = body.get("config_sha256")
    if not isinstance(observed, Mapping) or {
        key: str(observed.get(key) or "") for key in config_sha256
    } != dict(config_sha256):
        raise ValueError("operational evidence does not bind the frozen configs")
    if body.get("measurement_layer") != "operational":
        raise ValueError("operational evidence is not labeled as the operational layer")
    producer = body.get("producer_receipt")
    required_schema = str(producer_policy.get("required_receipt_schema") or "")
    if (
        not isinstance(producer, Mapping)
        or producer.get("schema_version") != required_schema
        or producer.get("status") != "VERIFIED_PRODUCER_OUTPUT"
        or producer.get("measurement_mode") != producer_policy.get("required_measurement_mode")
        or any(
            not isinstance(producer.get(field), str) or len(str(producer.get(field))) != 64
            for field in (
                "raw_block_index_sha256",
                "raw_measurements_sha256",
                "implementation_sha256",
            )
        )
    ):
        raise ValueError("operational evidence lacks a verified live producer receipt")
    return body


def _verify_audit_receipt(
    raw: Mapping[str, Any] | None,
    *,
    itt: Mapping[str, Any],
) -> dict[str, Any] | None:
    if raw is None:
        return None
    from ..evaluation.human_audit_workflow import validate_human_audit_receipt

    body = validate_human_audit_receipt(
        raw,
        run_id=str(itt["run_id"]),
        phase_id=str(itt["phase_id"]),
        evaluation_scope_sha256=str(itt["evaluation_scope_sha256"]),
        execution_binding_sha256=str(itt["execution_binding_sha256"]),
        protocol_document_sha256=str(itt["protocol_document_sha256"]),
    )
    if (
        str(body.get("content_sha256") or "") != str(itt.get("human_audit_receipt_sha256") or "")
        or itt.get("truth_audited") is not True
    ):
        raise ValueError("human-audit receipt does not bind the exact ITT analysis")
    return body


def build_final_decision(
    itt_verdict_inputs: Mapping[str, Any],
    e2e_analysis: Mapping[str, Any],
    *,
    decision_config: Mapping[str, Any],
    week1_config: Mapping[str, Any],
    variants_config: Mapping[str, Any],
    operational_evidence: Mapping[str, Any] | None = None,
    human_audit_receipt: Mapping[str, Any] | None = None,
    protocol_sha: str = "",
    generated_at_utc: str = "",
) -> DecisionObject:
    """Build one typed decision from the exact frozen evidence and threshold configs."""
    protocol_sha = _require_sha256(protocol_sha, label="protocol_sha")
    generated_at_utc = _require_generated_at(generated_at_utc)
    hashes = _config_hashes(decision_config, week1_config, variants_config)
    itt, e2e, effects = _verify_input_bindings(
        itt_verdict_inputs, e2e_analysis, config_sha256=hashes
    )
    if protocol_sha != str(itt.get("protocol_document_sha256") or ""):
        raise ValueError(
            "report protocol_sha differs from the frozen run protocol document"
        )
    operational = _verify_operational(
        operational_evidence,
        itt=itt,
        e2e=e2e,
        config_sha256=hashes,
        producer_policy=(
            decision_config.get("operational_evidence_producer")
            if isinstance(decision_config.get("operational_evidence_producer"), Mapping)
            else {}
        ),
    )
    audit = _verify_audit_receipt(human_audit_receipt, itt=itt)
    variants, arms = _variant_registry(week1_config, variants_config)
    groups, registry_variant = _arm_groups(decision_config, variants, arms, e2e)

    campaign = week1_config.get("campaign")
    if not isinstance(campaign, Mapping):
        raise ValueError("week1 campaign claim scope is absent")
    claim_scope = str(campaign.get("claim_scope") or "")
    corpus_tier = str(campaign.get("corpus_tier") or "")
    if not claim_scope or not corpus_tier or str(itt.get("claim_scope") or "") != claim_scope:
        raise ValueError("ITT claim scope disagrees with the frozen campaign")
    formative = claim_scope == "FORMATIVE_ONLY" or corpus_tier == "FORMATIVE_MACHINE_AUTHORED"

    (
        eligibility_status,
        exploratory_eligibility,
        eligibility_limitations,
    ) = _exploratory_eligibility_snapshot(
        effects,
        decision_config=decision_config,
    )

    itt_arms = itt.get("arms")
    if not isinstance(itt_arms, Mapping):
        raise ValueError("ITT verdict input has no arm estimands")
    node_decisions: dict[str, NodeDecision] = {}
    champion_selection = decision_config.get("champion_selection")
    if not isinstance(champion_selection, Mapping):
        raise ValueError("decision config lacks frozen champion-selection policy")
    champion_rule = str(champion_selection.get("rule") or "")
    simplicity_by_node = champion_selection.get("simplicity_order_by_node")
    primary_by_node = champion_selection.get("primary_arm_by_node")
    secondary_policy = str(champion_selection.get("secondary_policy") or "")
    if (
        champion_rule != "exact_preregistered_core_factorial_primary_arm_only"
        or not isinstance(simplicity_by_node, Mapping)
        or not isinstance(primary_by_node, Mapping)
        or secondary_policy != "SECONDARY_EXPLORATORY_NO_MULTIPLICITY_ADJUSTED_CLAIM"
    ):
        raise ValueError("primary-arm or secondary-exploratory decision policy is absent")
    structured_gates = _structured_increment_gates(
        e2e.get("matched_contrasts") if isinstance(e2e.get("matched_contrasts"), Mapping) else None,
        decision_config=decision_config,
        primary_arm_by_node={
            node: str(primary_by_node.get(node) or "")
            for node in ("WEBPAGE_P1", "C_VISIBLE", "H_PLUS_C_VISIBLE")
        },
        matched_design=(
            week1_config.get("matched_contrasts")
            if isinstance(week1_config.get("matched_contrasts"), Mapping)
            else {}
        ),
        variants=variants,
        arms=arms,
    )
    for node, candidates in groups.items():
        prefixes = {
            "WEBPAGE_P1": ("h",),
            "C_VISIBLE": ("c",),
            "H_PLUS_C_VISIBLE": ("h", "c"),
        }[node]
        assessments = [
            _assess_arm(
                arm_id,
                variant_id,
                itt_arms.get(arm_id),
                node_prefixes=prefixes,
                decision_config=decision_config,
                operational=operational,
                # Eligibility is a separate formative task-level analysis.  It must never
                # manufacture conditional evidence or alter the primary decision tree.
                conditional=(False, 0.0),
                structured_increment=structured_gates[node],
            )
            for arm_id, variant_id in candidates
        ]
        node_decisions[node] = _node_from_assessments(
            node,
            assessments,
            where_it_works=(),
            eligibility_limitations=eligibility_limitations,
            formative=formative,
            simplicity_order=simplicity_by_node.get(node) or (),
        )

    # C_REGISTRY has its own visibility contract.  A C_VISIBLE result can neither execute nor
    # stand in for it, even when both happen to use the same selector implementation.
    c_registry = NodeDecision(
        node="C_REGISTRY",
        verdict=Verdict.NOT_ESTABLISHED,
        champion_variant=None,
        work_saving=None,
        candidate_variants=(registry_variant,),
        reasons=(
            "C_REGISTRY_EXTENSION_NOT_EXECUTED",
            "C_VISIBLE_IS_NOT_A_SUBSTITUTE_FOR_RAW_REGISTRY_VISIBILITY",
        ),
        limitations=("NO_RUNNABLE_RAW_REGISTRY_SNAPSHOT_AND_PROVENANCE_ADAPTER",),
        claim_level="NOT_ESTABLISHED",
    )

    global_limitations: list[str] = []
    global_blockers: list[str] = []
    if formative:
        global_limitations.append(
            "FORMATIVE_ONLY: machine-authored corpus cannot support confirmatory claims"
        )
    if operational is None:
        if (decision_config.get("operational_evidence_producer") or {}).get(
            "implementation_status"
        ) != "IMPLEMENTED_VERIFIED":
            global_blockers.append(
                "Operational evidence producer is NOT_IMPLEMENTED; live concurrent "
                "batching/APC cost is unanswered and causal passes cannot exceed MECHANISM_ONLY"
            )
        else:
            global_blockers.append(
                "Operational APC/batching campaign unavailable; causal passes are MECHANISM_ONLY"
            )
    if audit is None:
        global_blockers.append(
            "Human-audit receipt absent; quality-dependent displays remain PROVISIONAL"
        )
    if eligibility_limitations:
        global_limitations.extend(eligibility_limitations)
    if any(gate.get("status") != "ESTABLISHED" for gate in structured_gates.values()):
        global_blockers.append(
            "Structured ID-path increment over frozen CPU/prose controls is not established; "
            "P1-vs-P0 benefit cannot become a positive P1 mechanism claim"
        )
    global_limitations.append(
        "Post-treatment search, reasoning, and checkpoint divergence is an E2E outcome, "
        "not a pairing exclusion"
    )
    global_limitations.append(
        "Primary verdicts use only the exact preregistered core-factorial arms; other H/C "
        "variants are SECONDARY_EXPLORATORY_NO_MULTIPLICITY_ADJUSTED_CLAIM and cannot replace "
        "or upgrade the primary"
    )

    status = AUDITED_STATUS if audit is not None else PROVISIONAL_STATUS
    inputs = {
        "ITT_VERDICT_INPUTS": str(itt["content_sha256"]),
        "E2E_ANALYSIS": str(e2e["content_sha256"]),
    }
    if operational is not None:
        inputs["OPERATIONAL_EVIDENCE"] = str(operational["content_sha256"])
    if audit is not None:
        inputs["HUMAN_AUDIT_RECEIPT"] = str(audit["content_sha256"])
    trajectory_outcomes = _trajectory_outcomes(effects, decision_config)
    factorial_endpoints = effects.get("factorial_endpoints")
    factorial_endpoints = (
        json.loads(canonical_json(dict(factorial_endpoints)))
        if isinstance(factorial_endpoints, Mapping)
        else {}
    )
    quality_sensitivity = effects.get("quality_guards")
    quality_sensitivity = (
        json.loads(canonical_json(dict(quality_sensitivity)))
        if isinstance(quality_sensitivity, Mapping)
        else {}
    )
    matched_outcomes = e2e.get("matched_contrasts")
    matched_outcomes = (
        json.loads(canonical_json(dict(matched_outcomes)))
        if isinstance(matched_outcomes, Mapping)
        else {}
    )
    if matched_outcomes:
        matched_outcomes["decision_use"] = (
            "SECONDARY_EXPLORATORY_NO_MULTIPLICITY_ADJUSTED_CLAIM; "
            "STRUCTURED_INCREMENT_ONLY_VIA_EXPLICIT_VERSIONED_COMPOSITE"
        )
    task_joint_outcomes = effects.get("task_level_joint_outcomes")
    if isinstance(task_joint_outcomes, Mapping):
        task_joint_outcomes = json.loads(canonical_json(dict(task_joint_outcomes)))
        if task_joint_outcomes.get("status") == "OK":
            expected_joint_policy = decision_config.get("task_level_joint_outcomes")
            if (
                not isinstance(expected_joint_policy, Mapping)
                or str(task_joint_outcomes.get("policy_sha256") or "")
                != sha256_hex(canonical_json(dict(expected_joint_policy)))
                or task_joint_outcomes.get("scope") != "TASK_LEVEL_PRIMARY_H_C_HC_JOINT_OUTCOMES"
            ):
                raise ValueError("task-level joint outcomes do not bind the frozen H/C/H+C policy")
    else:
        task_joint_outcomes = {
            "status": "NOT_ESTABLISHED",
            "reason": "TASK_LEVEL_JOINT_OUTCOME_ARTIFACT_MISSING",
        }
    executed_design: dict[str, dict[str, Any]] = {}
    primary_arm_ids = set(map(str, primary_by_node.values()))
    for arm_id, arm in sorted(arms.items()):
        page = str(arm["page_variant"])
        close = str(arm["close_variant"])
        active = [item for item in (page, close) if item != "P0"]
        is_control = not active or any(bool(variants[item].get("is_control")) for item in active)
        executed_design[arm_id] = {
            "page_variant": page,
            "close_variant": close,
            "is_control": is_control,
            "analysis_role": (
                "PRIMARY_EXACT_CORE_FACTORIAL_ARM"
                if arm_id in primary_arm_ids
                else "FROZEN_MATCHED_CONTROL"
                if is_control
                else "SECONDARY_EXPLORATORY_NO_MULTIPLICITY_ADJUSTED_CLAIM"
            ),
            "measurement_status": (
                "COMPARATOR_EMBEDDED_IN_EACH_ITT_PAIR"
                if arm_id
                == str(
                    (
                        (
                            (decision_config.get("e2e_analysis") or {}).get("core_factorial") or {}
                        ).get("p0")
                        or {}
                    ).get("arm_id")
                    or ""
                )
                else "PRESENT_IN_ITT_VERDICT_INPUTS"
                if arm_id in itt_arms
                else "NOT_PRESENT_IN_THIS_RUN_PHASE"
            ),
        }
    artifact_base = f"judgments/{itt['run_id']}/{itt['phase_id']}/analysis"
    return DecisionObject(
        webpage_p1=node_decisions["WEBPAGE_P1"],
        c_visible=node_decisions["C_VISIBLE"],
        c_registry=c_registry,
        h_plus_c_visible=node_decisions["H_PLUS_C_VISIBLE"],
        verdict_status=status,
        confirmatory_power_shortfall=None,
        human_audit_status=(
            "AUDITED_RECEIPT_VERIFIED" if audit is not None else "AUDITED_RECEIPT_NOT_PRESENT"
        ),
        protocol_sha=protocol_sha,
        freeze_sha=str(itt["freeze_root_sha256"]),
        generated_at_utc=generated_at_utc,
        run_id=str(itt["run_id"]),
        phase_id=str(itt["phase_id"]),
        evaluation_scope_sha256=str(itt["evaluation_scope_sha256"]),
        claim_scope=claim_scope,
        corpus_tier=corpus_tier,
        operational_status=(
            "BOUND_OPERATIONAL_EVIDENCE"
            if operational is not None
            else "UNAVAILABLE_PRODUCER_NOT_IMPLEMENTED"
            if (
                (decision_config.get("operational_evidence_producer") or {}).get(
                    "implementation_status"
                )
                != "IMPLEMENTED_VERIFIED"
            )
            else "UNAVAILABLE_NOT_RUN"
        ),
        eligibility_status=eligibility_status,
        exploratory_task_level_eligibility=exploratory_eligibility,
        input_sha256=inputs,
        config_sha256=hashes,
        limitations=tuple(dict.fromkeys(global_limitations)),
        blockers=tuple(dict.fromkeys(global_blockers)),
        confirmatory_status=(
            "NOT_APPLICABLE_FORMATIVE_ONLY"
            if formative
            else "NOT_ASSESSED_BY_THIS_DECISION_BUILDER"
        ),
        champion_selection_rule=champion_rule,
        trajectory_outcomes=trajectory_outcomes,
        blocks_offered=int(itt.get("blocks_offered") or 0),
        stack_status=(
            "FROZEN_ROOT_AND_CONFIG_HASHES_BOUND; live stack details are not duplicated "
            "inside the final decision inputs"
        ),
        executed_design=executed_design,
        matched_variant_outcomes=matched_outcomes,
        distribution_outcomes={
            **{
                key: factorial_endpoints[key]
                for key in ("e2e_latency_seconds",)
                if key in factorial_endpoints
            },
            "task_level_joint_outcomes": task_joint_outcomes,
        },
        work_outcomes={
            key: factorial_endpoints[key]
            for key in (
                "interval_union_seconds",
                "service_work_seconds",
                "energy_joules",
                "prompt_tokens",
                "completion_tokens",
                "cached_prompt_tokens",
            )
            if key in factorial_endpoints
        },
        sensitivity_outcomes=quality_sensitivity,
        reproduction_commands=(
            f"shapeflow-p1 analyze-itt --run-id {itt['run_id']} " f"--phase-id {itt['phase_id']}",
            f"shapeflow-p1 analyze-e2e --run-id {itt['run_id']} " f"--phase-id {itt['phase_id']}",
            f"shapeflow-p1 finalize-decision --run-id {itt['run_id']} "
            f"--phase-id {itt['phase_id']}",
        ),
        artifact_locations={
            "ITT_VERDICT_INPUTS": f"{artifact_base}/ITT_VERDICT_INPUTS.json",
            "E2E_ANALYSIS": f"{artifact_base}/E2E_ANALYSIS.json",
            "DECISION_JSON": f"{artifact_base}/WEEK1_P1_DECISION.json",
            "DECISION_MARKDOWN": f"{artifact_base}/WEEK1_P1_DECISION.md",
        },
    )


def _write_once(path: Path, payload: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o440)
        return "CREATED"
    except FileExistsError:
        if path.read_text(encoding="utf-8") != payload:
            raise ValueError(f"{path} already contains a different final decision") from None
        return "EXISTING_IDENTICAL"


def write_final_decision(
    decision: DecisionObject,
    *,
    json_path: Path,
    markdown_path: Path,
) -> dict[str, str]:
    """Write JSON and Markdown derived from the same typed object, exclusively and idempotently."""
    json_payload = render_json(decision) + "\n"
    # Recheck the content address immediately before persistence.
    parsed = json.loads(json_payload)
    if str(parsed.get("content_sha256") or "") != _unsigned_sha(parsed):
        raise ValueError("rendered decision JSON is not correctly content addressed")
    markdown_payload = render_markdown(decision)
    return {
        "json": _write_once(Path(json_path), json_payload),
        "markdown": _write_once(Path(markdown_path), markdown_payload),
    }
