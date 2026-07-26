"""Scoring, under the evaluator identity, after the treatment output is frozen.

The ordering is the whole design. Evaluation reads outputs that already exist and cannot change;
it never runs while a treatment run is in flight, and nothing it computes is ever handed back to
a selector, an aggregator or a preflight. Truth in the treatment path would be an oracle, and an
arm scored against an oracle it could consult is not measuring what it appears to measure.

Three refusals encoded here:

**No scoring before freeze.** :func:`score_task` requires each arm's output to be marked frozen.
Scoring a live run would let a result depend on when it happened to be read.

**One truth packet and one judge policy per task.** Every arm of a task -- P0, P1 and the
controls -- is scored against the same packet and the same frozen prompts. Re-deriving truth per
arm would let each arm be graded against a slightly different answer key, which is
indistinguishable from grading them differently.

**A missing judgment stays missing.** :class:`JudgeUnavailable` produces ``JUDGE_UNAVAILABLE``
and a human-queue entry. It is never imputed as a zero, a mean or a pass, and the sample is
never dropped: a silently-imputed score corrupts exactly the quality endpoints the study exists
to report, and a silently-dropped one biases whatever is left.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from ..canonical import canonical_json
from ..hashing import sha256_hex
from ..p1.contracts import P1_CONTRACTS, canonical_normalization_document
from .atomizer import atomize_report
from .citation_eval import AtomCandidate, HumanQueueItem, build_assessment
from .judge_client import JudgeUnavailable
from .quality_metrics import ContradictionPair, TruthAtom, TruthPacket, score_report

__all__ = [
    "ArmOutput",
    "TaskScore",
    "EvaluationError",
    "TreatmentReadTruth",
    "score_task",
    "score_direct_node_records",
    "selector_normalization_metrics",
    "prose_control_normalization_metrics",
    "write_scores",
    "assert_evaluator_only",
]


class EvaluationError(RuntimeError):
    """Evaluation cannot proceed as asked."""


class TreatmentReadTruth(EvaluationError):
    """Something on the treatment path reached for evaluator-only material. Fatal."""


@dataclass(frozen=True)
class ArmOutput:
    """One arm's frozen output for one task."""

    task_id: str
    arm_id: str
    variant_id: str
    replicate_id: str
    final_report: str
    frozen: bool = False
    terminal_failure: bool = False
    reported_unresolved_facets: frozenset = frozenset()
    block_id: str = ""
    assignment_state: str = "COMMITTED"
    fell_back: bool = False
    direct_node_records: tuple[dict, ...] = ()
    prose_control_records: tuple[dict, ...] = ()
    prose_control_expected_nodes: tuple[str, ...] = ()
    # Evaluator-derived receipt for the first eligible H/C input. Matched controls use this
    # pre-treatment boundary only; later candidate views may legitimately diverge through the
    # end-to-end treatment trajectory.
    first_boundary_input: dict = field(default_factory=dict)
    work_summary: dict = field(default_factory=dict)
    retrieved_source_occurrence_ids: tuple[str, ...] = ()
    trajectory_metrics: dict = field(default_factory=dict)


@dataclass
class TaskScore:
    task_id: str
    truth_packet_sha256: str
    judge_policy_sha256: str
    claim_scope: str
    run_id: str = ""
    phase_id: str = ""
    block_id: str = ""
    replicate_id: str = ""
    execution_binding_sha256: str = ""
    protocol_document_sha256: str = ""
    truth_authoring_method: str = ""
    truth_verifier_status: str = ""
    judge_provenance: dict = field(default_factory=dict)
    per_arm: dict = field(default_factory=dict)
    unavailable: list = field(default_factory=list)
    human_queue: list = field(default_factory=list)
    #: arm key -> the atom ids that arm's claims matched. The C_VISIBLE projection needs to
    #: know what the reducer retained, not just how the full-truth score came out.
    matched_atoms_by_arm: dict = field(default_factory=dict)

    def content(self) -> dict:
        return {
            "task_id": self.task_id,
            "truth_packet_sha256": self.truth_packet_sha256,
            "judge_policy_sha256": self.judge_policy_sha256,
            "claim_scope": self.claim_scope,
            "run_id": self.run_id,
            "phase_id": self.phase_id,
            "block_id": self.block_id,
            "replicate_id": self.replicate_id,
            "execution_binding_sha256": self.execution_binding_sha256,
            "protocol_document_sha256": self.protocol_document_sha256,
            "truth_authoring_method": self.truth_authoring_method,
            "truth_verifier_status": self.truth_verifier_status,
            "judge_provenance": dict(sorted(self.judge_provenance.items())),
            "per_arm": {k: dict(sorted(v.items())) for k, v in sorted(self.per_arm.items())},
            "unavailable": sorted(self.unavailable),
            "human_queue": [dict(sorted(item.items())) for item in self.human_queue],
        }

    @property
    def content_sha256(self) -> str:
        return sha256_hex(canonical_json(self.content()))


def assert_evaluator_only(role: str | None) -> None:
    """Refuse to run as anything but the evaluator.

    The runner identity holds treatment outputs and must never hold the answer key. Checking the
    effective role here means a mis-wired CLI invocation fails instead of quietly producing
    scores computed by the process that also produced the outputs.
    """
    user = os.environ.get("USER") or os.environ.get("USERNAME") or ""
    if role == "evaluator" and user and user != "sfevaluator":
        raise TreatmentReadTruth(
            f"evaluation must run as sfevaluator, not {user!r}: the identity that produced the "
            "outputs must not also hold the truth they are scored against"
        )


def _truth_packet_from(body: dict) -> TruthPacket:
    atoms = tuple(
        TruthAtom(
            atom_id=a["atom_id"], facet_id=a["facet_id"], weight=float(a["weight"]),
            critical=bool(a["critical"]), known_unresolved=bool(a.get("known_unresolved", False)),
        )
        for a in body["atomic_evidence"]
    )
    pairs = tuple(
        ContradictionPair(atom_id_a=p["atom_id_a"], atom_id_b=p["atom_id_b"])
        for p in body.get("contradiction_pairs", ())
    )
    return TruthPacket(
        required_facets=tuple(body["required_facets"]), atoms=atoms, contradiction_pairs=pairs,
    )


def score_task(
    *,
    truth_body: dict,
    outputs: Sequence[ArmOutput],
    atom_texts: dict,
    judge_relation: Callable[[str, str], str],
    citation_supports: Callable[[str, str], bool | None],
    judge_policy_sha256: str,
    claim_scope: str,
    run_id: str = "",
    phase_id: str = "",
    block_id: str = "",
    replicate_id: str = "",
    execution_binding_sha256: str = "",
    protocol_document_sha256: str = "",
    truth_chunker: str = "markdown_structure_v1",
    atom_support_index: dict | None = None,
) -> TaskScore:
    """Score every arm of one task against one truth packet and one judge policy.

    Arms are scored in a fixed order and with identical inputs, so the only thing that differs
    between them is the report each produced.
    """
    frozen_missing = [o.arm_id for o in outputs if not o.frozen]
    if frozen_missing:
        raise EvaluationError(
            f"arms {frozen_missing} are not frozen; scoring a live run would let the result "
            "depend on when it happened to be read"
        )
    packet = _truth_packet_from(truth_body)
    candidates = [
        AtomCandidate(atom_id=a.atom_id, text=atom_texts.get(a.atom_id, ""), critical=a.critical)
        for a in packet.atoms
    ]
    score = TaskScore(
        task_id=outputs[0].task_id if outputs else "",
        truth_packet_sha256=str(truth_body.get("content_sha256", "")),
        judge_policy_sha256=judge_policy_sha256,
        claim_scope=claim_scope,
        run_id=run_id,
        phase_id=phase_id,
        block_id=block_id or (outputs[0].block_id if outputs else ""),
        replicate_id=replicate_id or (outputs[0].replicate_id if outputs else ""),
        execution_binding_sha256=execution_binding_sha256,
        protocol_document_sha256=protocol_document_sha256,
        truth_authoring_method=str(truth_body.get("authoring_method") or ""),
        truth_verifier_status=str(truth_body.get("verifier_status") or ""),
    )

    for output in sorted(outputs, key=lambda o: (o.arm_id, o.replicate_id)):
        arm_key = f"{output.arm_id}:{output.replicate_id}"
        base = {
            "variant_id": output.variant_id,
            "assignment_state": output.assignment_state,
            "fell_back": output.fell_back,
            "claim_scope": claim_scope,
            "work_summary": dict(output.work_summary),
            "trajectory_metrics": dict(output.trajectory_metrics),
            "first_boundary_input": dict(output.first_boundary_input),
            "direct_node_metrics": score_direct_node_records(
                truth_body, output.direct_node_records, truth_chunker=truth_chunker,
                atom_support_index=atom_support_index),
            "prose_control_normalization": prose_control_normalization_metrics(
                output.prose_control_records,
                expected_nodes=output.prose_control_expected_nodes,
            ),
        }
        if output.terminal_failure or output.assignment_state != "COMMITTED":
            failed = _quality_failure_values(packet, truth_body=truth_body)
            score.per_arm[arm_key] = {
                **base,
                **failed,
                "negative_gap_report": None,
                "evaluation_status": "TERMINAL_FAILURE",
                "quality_views": {
                    mode: dict(failed)
                    for mode in ("strict", "fallback_assisted", "worst_case", "best_case")
                },
            }
            score.matched_atoms_by_arm[arm_key] = []
            continue
        try:
            claims = atomize_report(output.final_report)
            if hasattr(citation_supports, "bind"):
                # A citation label only means something inside the report that defined it,
                # so the resolver is bound per arm rather than per task.
                citation_supports.bind(
                    output.final_report,
                    {c.claim_id: c.text for c in claims},
                    allowed_occurrence_ids=output.retrieved_source_occurrence_ids,
                )
            assessment, queue = build_assessment(
                claims, candidates, judge=judge_relation,
                citation_supports=citation_supports,
                reported_unresolved_facets=frozenset(output.reported_unresolved_facets),
                terminal_failure=output.terminal_failure,
                empty_or_truncated=not output.final_report.strip(),
            )
            # A negative/gap truth atom is credited only when a supported report claim
            # semantically matches that exact, grounded unresolved atom. This derives the
            # report-level unresolved facets from the same blind assessment used for positive
            # claims; leaving the field permanently empty penalized every arm even when it
            # explicitly and correctly communicated a known gap.
            unresolved_atom_facets = {
                atom.atom_id: atom.facet_id for atom in packet.atoms
                if atom.known_unresolved
            }
            derived_unresolved = {
                unresolved_atom_facets[atom_id]
                for claim in assessment.claims
                if claim.support_status == "SUPPORTED"
                for atom_id in claim.matched_atom_ids
                if atom_id in unresolved_atom_facets
            }
            assessment = replace(
                assessment,
                reported_unresolved_facets=frozenset(
                    set(assessment.reported_unresolved_facets) | derived_unresolved
                ),
            )
            negative_gap_values, negative_gap_report, gap_queue = (
                _score_negative_and_gap_reporting(
                    truth_body=truth_body,
                    report_claims=claims,
                    assessment=assessment,
                    judge_relation=judge_relation,
                )
            )
            queue.extend(gap_queue)
        except JudgeUnavailable as e:
            # Never imputed, never dropped. The sample is recorded as unavailable and routed to
            # the human queue; a zero here would be a fabricated failure for that arm.
            score.unavailable.append(arm_key)
            score.human_queue.append({
                "task_id": output.task_id, "arm_id": output.arm_id,
                "replicate_id": output.replicate_id, "reason": f"JUDGE_UNAVAILABLE: {e}",
            })
            failed = _quality_failure_values(packet, truth_body=truth_body)
            best = _quality_best_values(packet, truth_body=truth_body)
            score.per_arm[arm_key] = {
                **base,
                **{name: None for name in failed},
                "negative_gap_report": None,
                "evaluation_status": "JUDGE_UNAVAILABLE",
                "quality_views": {
                    "strict": dict(failed) if output.fell_back else None,
                    "fallback_assisted": None,
                    "worst_case": dict(failed),
                    "best_case": dict(best),
                },
            }
            score.matched_atoms_by_arm[arm_key] = []
            continue

        scores = score_report(packet, assessment)
        actual = _quality_values(scores)
        actual.update(negative_gap_values)
        failed = _quality_failure_values(packet, truth_body=truth_body)
        score.matched_atoms_by_arm[arm_key] = sorted({
            atom_id for c in assessment.claims
            if c.support_status == "SUPPORTED"
            for atom_id in c.matched_atom_ids
        })
        score.per_arm[arm_key] = {
            **base,
            **actual,
            "negative_gap_report": negative_gap_report,
            "evaluation_status": "SCORED",
            # Strict asks whether P1 itself produced the output. Assisted asks whether the
            # deployable P1+fallback policy did. They coincide for a strict-valid P1 result.
            "quality_views": {
                "strict": dict(failed if output.fell_back else actual),
                "fallback_assisted": dict(actual),
                "worst_case": dict(actual),
                "best_case": dict(actual),
            },
        }
        score.human_queue.extend({
            "task_id": output.task_id, "arm_id": output.arm_id,
            "replicate_id": output.replicate_id, "claim_id": item.claim_id,
            "reason": item.reason,
        } for item in queue)
    return score


def _score_negative_and_gap_reporting(
    *,
    truth_body: dict,
    report_claims: Sequence,
    assessment,
    judge_relation: Callable[[str, str], str],
) -> tuple[dict, dict, list[HumanQueueItem]]:
    """Score proven absence and operationally unresolved search separately.

    A grounded negative is an accepted, source-backed truth atom and therefore requires a
    supported report claim with a supporting citation. A known gap is not source truth at all:
    it is a frozen failed/timeout/budget-blocked query attempt, so the report only has to state
    the unresolved limitation accurately; inventing a citation requirement would be misleading.
    """
    negative_atom_ids = {
        str(item.get("atom_id") or "")
        for item in (truth_body.get("negative_evidence") or ())
    }
    negative_atom_ids.discard("")
    reported_negative_ids = {
        atom_id
        for claim in assessment.claims
        if claim.support_status == "SUPPORTED" and claim.supporting_citation
        for atom_id in claim.matched_atom_ids
        if atom_id in negative_atom_ids
    }

    gap_records = [
        item for item in (truth_body.get("known_gaps") or ())
        if isinstance(item, dict)
        and str(item.get("status") or "") in {
            "FAILED", "TIMEOUT", "BLOCKED_BUDGET",
        }
        and str(item.get("query_attempt_id") or "")
    ]
    reported_gap_ids: set[str] = set()
    queue: list[HumanQueueItem] = []
    for gap in gap_records:
        query_id = str(gap["query_attempt_id"])
        query_text = str(gap.get("query_text") or "")
        status = str(gap["status"])
        target = (
            f"The evidence search for {query_text!r} remained unresolved because its frozen "
            f"query attempt ended with status {status}."
        )
        uncertain_claims: list[str] = []
        for claim in report_claims:
            relation = judge_relation(str(claim.text), target)
            if relation == "entail":
                reported_gap_ids.add(query_id)
                uncertain_claims = []
                break
            if relation == "uncertain":
                uncertain_claims.append(str(claim.claim_id))
        queue.extend(
            HumanQueueItem(
                claim_id=claim_id,
                reason=f"known gap {query_id} communication uncertain",
            )
            for claim_id in uncertain_claims
        )

    values = {
        "grounded_negative_recall": (
            len(reported_negative_ids) / len(negative_atom_ids)
            if negative_atom_ids else None
        ),
        "unresolved_gap_reporting_recall": (
            len(reported_gap_ids) / len(gap_records)
            if gap_records else None
        ),
    }
    report = {
        "grounded_negative_atom_ids": sorted(negative_atom_ids),
        "reported_grounded_negative_atom_ids": sorted(reported_negative_ids),
        "known_gap_query_attempt_ids": sorted(
            str(item["query_attempt_id"]) for item in gap_records),
        "reported_unresolved_query_attempt_ids": sorted(reported_gap_ids),
    }
    return values, report, queue


_QUALITY_FIELDS = (
    "weighted_required_atom_recall",
    "critical_atom_safety",
    "grounded_claim_precision",
    "citation_correctness",
    "citation_association",
    "citation_completeness",
    "required_facet_coverage",
    "contradiction_handling",
    "critical_harm",
    "qualified_report",
)


def _quality_values(scores) -> dict:
    return {name: getattr(scores, name) for name in _QUALITY_FIELDS}


def _quality_failure_values(packet: TruthPacket, *, truth_body: dict | None = None) -> dict:
    """Pre-registered adverse outcome for an assigned arm that produced no judgeable report."""
    truth_body = truth_body or {}
    return {
        "weighted_required_atom_recall": 0.0,
        "critical_atom_safety": 0.0 if any(a.critical for a in packet.atoms) else 1.0,
        "grounded_claim_precision": 0.0,
        "citation_correctness": 0.0,
        "citation_association": 0.0,
        "citation_completeness": 0.0,
        "required_facet_coverage": 0.0,
        "contradiction_handling": 0.0 if packet.contradiction_pairs else None,
        "critical_harm": 1,
        "qualified_report": 0,
        "grounded_negative_recall": (
            0.0 if truth_body.get("negative_evidence") else None),
        "unresolved_gap_reporting_recall": (
            0.0 if truth_body.get("known_gaps") else None),
    }


def _quality_best_values(packet: TruthPacket, *, truth_body: dict | None = None) -> dict:
    """Upper endpoint used only for explicit judge-missing sensitivity bounds."""
    truth_body = truth_body or {}
    return {
        "weighted_required_atom_recall": 1.0,
        "critical_atom_safety": 1.0,
        "grounded_claim_precision": 1.0,
        "citation_correctness": 1.0,
        "citation_association": 1.0,
        "citation_completeness": 1.0,
        "required_facet_coverage": 1.0,
        "contradiction_handling": 1.0 if packet.contradiction_pairs else None,
        "critical_harm": 0,
        "qualified_report": 1,
        "grounded_negative_recall": (
            1.0 if truth_body.get("negative_evidence") else None),
        "unresolved_gap_reporting_recall": (
            1.0 if truth_body.get("known_gaps") else None),
    }


_NORMALIZATION_BASE_FIELDS = frozenset({
    "raw_count",
    "unique_count",
    "duplicate_count",
    "semantic_conflict_count",
    "rejected_reason",
})
_NORMALIZATION_DERIVED_FIELDS = frozenset({
    "was_repaired",
    "was_rejected",
    "strict_valid",
})


def _validated_normalization(value: object) -> dict:
    """Validate either the treatment primitives or the recorder's canonical document."""
    if not isinstance(value, Mapping):
        raise ValueError("normalization must be an object")
    keys = frozenset(map(str, value))
    if keys == _NORMALIZATION_BASE_FIELDS:
        primitives = dict(value)
    elif keys == _NORMALIZATION_BASE_FIELDS | _NORMALIZATION_DERIVED_FIELDS:
        primitives = {key: value.get(key) for key in _NORMALIZATION_BASE_FIELDS}
    else:
        raise ValueError(
            "normalization fields are not closed "
            f"(missing={sorted(_NORMALIZATION_BASE_FIELDS - keys)}, "
            f"extra={sorted(keys - _NORMALIZATION_BASE_FIELDS
                            - _NORMALIZATION_DERIVED_FIELDS)})"
        )
    canonical = canonical_normalization_document(primitives)
    if keys != _NORMALIZATION_BASE_FIELDS and dict(value) != canonical:
        raise ValueError("normalization derived flags do not match primitive counters")
    return canonical


def _selector_attempted(raw: Mapping) -> tuple[bool, str]:
    if "selector_attempted" in raw:
        value = raw.get("selector_attempted")
        if type(value) is not bool:
            return True, "selector_attempted must be a boolean"
        return value, ""
    # Pre-schema records need a fail-closed migration path. An offered P1 view or a present
    # normalization is evidence that a call occurred; absence of the new trace can then never
    # be counted as a strict-valid completion.
    return bool(
        raw.get("normalization") is not None
        or raw.get("candidate_view_sha256")
        or (
            str(raw.get("contract") or "") in P1_CONTRACTS
            and raw.get("offered_span_ids")
        )
    ), ""


def selector_normalization_metrics(
    records: Sequence[dict], *, node: str | None = None
) -> dict:
    """Aggregate selector validity by node and stage over every offered attempt.

    Publication/fallback quality is intentionally irrelevant here. An out-of-set ID remains
    an invalid selector completion even when whole-batch P0 fallback later writes an excellent
    report. Conversely, duplicate IDs are repairs, never invalid IDs.
    """
    node_prefix = str(node or "").upper()
    selected_records = [
        raw for raw in records
        if not node_prefix
        or str(raw.get("node") or "").upper().startswith(node_prefix)
    ]

    def summarize(rows: Sequence[dict]) -> dict:
        attempts = 0
        strict_valid = 0
        repaired = 0
        rejected = 0
        invalid_ids = 0
        schema_rejections = 0
        semantic_attempts = 0
        semantic_events = 0
        failure_before_parse = 0
        normalization_missing = 0
        normalization_invalid = 0
        component_failures = 0
        errors: list[str] = []
        raw_items = 0
        unique_items = 0
        duplicate_items = 0

        for index, raw in enumerate(rows):
            attempted, attempt_error = _selector_attempted(raw)
            failure = raw.get("failure")
            component_failures += int(bool(failure))
            if attempt_error:
                normalization_invalid += 1
                errors.append(f"record {index}: {attempt_error}")
            if not attempted:
                if raw.get("normalization") is not None:
                    normalization_invalid += 1
                    errors.append(
                        f"record {index}: normalization present for non-attempted selector"
                    )
                continue
            attempts += 1
            value = raw.get("normalization")
            if value is None:
                normalization_missing += 1
                errors.append(
                    f"record {index}: selector attempt has no normalization trace"
                )
                continue
            try:
                normalized = _validated_normalization(value)
            except ValueError as exc:
                normalization_invalid += 1
                errors.append(f"record {index}: {exc}")
                continue
            raw_items += normalized["raw_count"]
            unique_items += normalized["unique_count"]
            duplicate_items += normalized["duplicate_count"]
            semantic_events += normalized["semantic_conflict_count"]
            strict_valid += int(normalized["strict_valid"])
            repaired += int(normalized["was_repaired"])
            rejected += int(normalized["was_rejected"])
            reason = normalized["rejected_reason"]
            invalid_ids += int(reason == "out_of_set_label")
            schema_rejections += int(reason == "schema")
            semantic_attempts += int(
                normalized["semantic_conflict_count"] > 0
                or reason in {"semantic_conflict", "semantic_invalid"}
            )
            failure_before_parse += int(reason == "failure_before_parse")

        invalid_traces = normalization_missing + normalization_invalid
        denominator = attempts
        no_repair_adverse = bool(
            component_failures
            or repaired
            or rejected
            or invalid_traces
        )
        return {
            "status": (
                "INVALID_NORMALIZATION_TRACE" if invalid_traces
                else "NOT_APPLICABLE" if denominator == 0
                else "OK"
            ),
            "selector_attempt_count": denominator,
            "strict_valid_count": strict_valid,
            "strict_valid_rate": (
                strict_valid / denominator if denominator else None
            ),
            "repaired_attempt_count": repaired,
            "repair_rate": repaired / denominator if denominator else None,
            "rejected_attempt_count": rejected,
            "rejection_rate": rejected / denominator if denominator else None,
            "invalid_id_count": invalid_ids,
            "invalid_id_rate": invalid_ids / denominator if denominator else None,
            "schema_rejection_count": schema_rejections,
            "schema_rejection_rate": (
                schema_rejections / denominator if denominator else None
            ),
            "semantic_conflict_attempt_count": semantic_attempts,
            "semantic_conflict_attempt_rate": (
                semantic_attempts / denominator if denominator else None
            ),
            "semantic_conflict_event_count": semantic_events,
            "failure_before_parse_count": failure_before_parse,
            "normalization_missing_count": normalization_missing,
            "normalization_invalid_count": normalization_invalid,
            "normalization_trace_error_count": invalid_traces,
            "component_failure_count": component_failures,
            "raw_item_count": raw_items,
            "unique_item_count": unique_items,
            "duplicate_item_count": duplicate_items,
            "no_repair_adverse": no_repair_adverse,
            "errors": errors,
        }

    by_stage: dict[str, dict] = {}
    for stage in sorted({
        str(raw.get("stage") or "single") for raw in selected_records
    }):
        by_stage[stage] = summarize([
            raw for raw in selected_records
            if str(raw.get("stage") or "single") == stage
        ])
    return {
        **summarize(selected_records),
        "node": node_prefix or "ALL",
        "record_count": len(selected_records),
        "by_stage": by_stage,
    }


_PROSE_NORMALIZATION_FIELDS = frozenset({
    "schema_version",
    "raw_semantically_valid",
    "raw_fits_rendered_token_budget",
    "raw_contract_adherent",
    "published_semantically_valid",
    "published_fits_rendered_token_budget",
    "published_policy_valid",
    "raw_validation",
    "published_validation",
    "raw_body_tokens",
    "published_body_tokens",
    "raw_rendered_tokens",
    "published_rendered_tokens",
    "budget_truncated",
    "invalid_raw_suffix_outside_policy_output",
    "semantic_repair_applied",
    "normalization_reasons",
    "raw_contract_adherence_sensitivity_excludes",
})
_PROSE_VALIDATION_FIELDS = frozenset({
    "valid",
    "unknown_citation_labels",
    "inline_citation_labels",
    "body_url_count",
    "source_definition_injection_count",
    "dangling_numeric_citation",
    "missing_required_inline_citation",
})


def _validated_prose_validation(value: object) -> dict:
    if not isinstance(value, Mapping) or frozenset(map(str, value)) != _PROSE_VALIDATION_FIELDS:
        raise ValueError("SHORT_PROSE citation validation fields are not closed")
    body = dict(value)
    for field in (
        "valid",
        "dangling_numeric_citation",
        "missing_required_inline_citation",
    ):
        if type(body[field]) is not bool:
            raise ValueError(f"SHORT_PROSE citation validation {field} must be boolean")
    for field in ("unknown_citation_labels", "inline_citation_labels"):
        raw = body[field]
        if (
            not isinstance(raw, list)
            or any(not isinstance(item, str) or not item for item in raw)
            or len(raw) != len(set(raw))
        ):
            raise ValueError(f"SHORT_PROSE citation validation {field} is malformed")
    for field in ("body_url_count", "source_definition_injection_count"):
        if (
            isinstance(body[field], bool)
            or not isinstance(body[field], int)
            or body[field] < 0
        ):
            raise ValueError(f"SHORT_PROSE citation validation {field} is malformed")
    expected_valid = not (
        body["unknown_citation_labels"]
        or body["body_url_count"]
        or body["source_definition_injection_count"]
        or body["dangling_numeric_citation"]
        or body["missing_required_inline_citation"]
    )
    if body["valid"] is not expected_valid:
        raise ValueError("SHORT_PROSE citation validation derived validity is inconsistent")
    return body


def _validated_prose_normalization(value: object) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("SHORT_PROSE normalization must be an object")
    body = dict(value)
    schema = str(body.get("schema_version") or "")
    if schema not in {
        "short_prose_page_policy_normalization_v2",
        "short_prose_close_policy_normalization_v2",
    }:
        raise ValueError(f"unknown SHORT_PROSE normalization schema {schema!r}")
    expected_fields = _PROSE_NORMALIZATION_FIELDS
    keys = frozenset(map(str, body))
    if keys != expected_fields:
        raise ValueError(
            "SHORT_PROSE normalization fields are not closed "
            f"(missing={sorted(expected_fields - keys)}, "
            f"extra={sorted(keys - expected_fields)})"
        )
    boolean_fields = (
        "raw_semantically_valid",
        "raw_fits_rendered_token_budget",
        "raw_contract_adherent",
        "published_semantically_valid",
        "published_fits_rendered_token_budget",
        "published_policy_valid",
        "budget_truncated",
        "invalid_raw_suffix_outside_policy_output",
        "semantic_repair_applied",
        "raw_contract_adherence_sensitivity_excludes",
    )
    for field in boolean_fields:
        if type(body[field]) is not bool:
            raise ValueError(f"SHORT_PROSE normalization {field} must be boolean")
    for field in (
        "raw_body_tokens",
        "published_body_tokens",
        "raw_rendered_tokens",
        "published_rendered_tokens",
    ):
        raw = body[field]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ValueError(f"SHORT_PROSE normalization {field} is malformed")
    reasons = body["normalization_reasons"]
    if (
        not isinstance(reasons, list)
        or any(not isinstance(reason, str) or not reason for reason in reasons)
        or len(reasons) != len(set(reasons))
        or not set(reasons) <= {
            "TOKEN_BUDGET_PREFIX",
            "INVALID_RAW_SUFFIX_OUTSIDE_POLICY_OUTPUT",
        }
    ):
        raise ValueError("SHORT_PROSE normalization_reasons is malformed")
    if body["published_rendered_tokens"] > body["raw_rendered_tokens"] and not body[
        "budget_truncated"
    ]:
        raise ValueError("SHORT_PROSE untruncated publication grew beyond its raw rendering")

    raw_validation = _validated_prose_validation(body["raw_validation"])
    published_validation = _validated_prose_validation(body["published_validation"])
    if body["raw_semantically_valid"] is not raw_validation["valid"]:
        raise ValueError("SHORT_PROSE raw semantic validity is inconsistent")
    if body["published_semantically_valid"] is not published_validation["valid"]:
        raise ValueError("SHORT_PROSE published semantic validity is inconsistent")
    if body["raw_contract_adherent"] is not (
        body["raw_semantically_valid"]
        and body["raw_fits_rendered_token_budget"]
    ):
        raise ValueError("SHORT_PROSE raw contract adherence is inconsistent")
    if body["published_policy_valid"] is not (
        body["published_semantically_valid"]
        and body["published_fits_rendered_token_budget"]
    ):
        raise ValueError("SHORT_PROSE published policy validity is inconsistent")
    expected_invalid_suffix = bool(
        body["budget_truncated"]
        and not body["raw_semantically_valid"]
        and body["published_semantically_valid"]
    )
    if body["invalid_raw_suffix_outside_policy_output"] is not expected_invalid_suffix:
        raise ValueError("SHORT_PROSE invalid raw suffix flag is inconsistent")
    if body["semantic_repair_applied"] is not False:
        raise ValueError("SHORT_PROSE semantic repair is forbidden")
    expected_exclusion = bool(
        body["budget_truncated"]
        or not body["raw_semantically_valid"]
        or not body["raw_fits_rendered_token_budget"]
    )
    expected_reasons = []
    if body["budget_truncated"]:
        expected_reasons.append("TOKEN_BUDGET_PREFIX")
    if expected_invalid_suffix:
        expected_reasons.append("INVALID_RAW_SUFFIX_OUTSIDE_POLICY_OUTPUT")
    if body["raw_contract_adherence_sensitivity_excludes"] is not expected_exclusion:
        raise ValueError("SHORT_PROSE raw-contract exclusion flag is inconsistent")
    if reasons != expected_reasons:
        raise ValueError("SHORT_PROSE normalization reasons are inconsistent")
    return body


def prose_control_normalization_metrics(
    records: Sequence[dict],
    *,
    expected_nodes: Sequence[str] = (),
) -> dict:
    """Audit every direct-prose publication without pretending it used the ID contract.

    The primary matched comparison remains all-offered ITT over the bytes actually published.
    This trace exists for the separate adverse no-repair sensitivity: a dirty structured
    output is made worst-case and a dirty prose control best-case, never silently excluded.
    """

    expected = tuple(dict.fromkeys(str(node).upper() for node in expected_nodes))
    if any(node not in {"H", "C"} for node in expected):
        raise ValueError("SHORT_PROSE expected nodes must be H and/or C")

    def summarize(rows: Sequence[dict], *, node: str) -> dict:
        attempts = 0
        published = 0
        rejected = 0
        discarded = 0
        call_failed = 0
        cancelled = 0
        raw_contract_adherent = 0
        truncated = 0
        semantic_repairs = 0
        raw_nonadherent = 0
        errors: list[str] = []
        for index, raw in enumerate(rows):
            if not isinstance(raw, Mapping):
                errors.append(f"record {index}: control record is not an object")
                continue
            attempts += 1
            digest = str(raw.get("candidate_view_sha256") or "")
            offered = raw.get("offered_span_ids")
            cap = raw.get("completion_token_cap")
            if (
                len(digest) != 64
                or digest != digest.lower()
                or not isinstance(offered, list)
                or not offered
                or any(not isinstance(value, str) or not value for value in offered)
                or len(offered) != len(set(offered))
                or isinstance(cap, bool)
                or not isinstance(cap, int)
                or cap <= 0
            ):
                errors.append(f"record {index}: control provenance is malformed")
            status = str(raw.get("publication_status") or "")
            if status == "PUBLISHED":
                published += 1
            elif status == "REJECTED":
                rejected += 1
            elif status == "FALLBACK_DISCARDED":
                discarded += 1
            else:
                errors.append(f"record {index}: publication_status is absent or invalid")
            attempt_status = str(raw.get("attempt_status") or "")
            trace_status = str(raw.get("normalization_trace_status") or "")
            selector_attempted = raw.get("selector_attempted")
            batch_accepted = raw.get("batch_accepted")
            work_incomplete = raw.get("work_incomplete")
            if (
                attempt_status not in {
                    "OUTPUT_OBSERVED",
                    "CALL_FAILED",
                    "CANCELLED",
                }
                or trace_status not in {"OK", "EXPLICIT_NO_OUTPUT", "MISSING"}
                or selector_attempted is not True
                or type(batch_accepted) is not bool
                or type(work_incomplete) is not bool
            ):
                errors.append(f"record {index}: attempt state is malformed")
            call_failed += int(attempt_status == "CALL_FAILED")
            cancelled += int(attempt_status == "CANCELLED")
            if attempt_status == "OUTPUT_OBSERVED" and trace_status != "OK":
                errors.append(
                    f"record {index}: observed output lacks a valid normalization trace"
                )
            if attempt_status != "OUTPUT_OBSERVED" and trace_status != "EXPLICIT_NO_OUTPUT":
                errors.append(
                    f"record {index}: no-output attempt lacks an explicit trace state"
                )
            if status == "PUBLISHED" and batch_accepted is not True:
                errors.append(f"record {index}: published attempt is not batch accepted")
            if status != "PUBLISHED" and batch_accepted is not False:
                errors.append(f"record {index}: discarded/rejected attempt is batch accepted")
            if trace_status == "EXPLICIT_NO_OUTPUT":
                if raw.get("normalization") is not None:
                    errors.append(
                        f"record {index}: no-output attempt carries a normalization object"
                    )
                raw_nonadherent += 1
                continue
            try:
                normalization = _validated_prose_normalization(raw.get("normalization"))
            except ValueError as exc:
                errors.append(f"record {index}: {exc}")
                continue
            if status == "PUBLISHED" and normalization["published_policy_valid"] is not True:
                errors.append(f"record {index}: invalid prefix is labelled PUBLISHED")
            if status == "REJECTED" and normalization["published_policy_valid"] is not False:
                errors.append(f"record {index}: valid prefix is labelled REJECTED")
            raw_contract_adherent += int(normalization["raw_contract_adherent"])
            truncated += int(normalization["budget_truncated"])
            semantic_repairs += int(normalization["semantic_repair_applied"])
            raw_nonadherent += int(
                normalization["raw_contract_adherence_sensitivity_excludes"]
            )
        policy_adverse = bool(
            errors or rejected or discarded or call_failed or cancelled or semantic_repairs
        )
        raw_adverse = bool(policy_adverse or raw_nonadherent)
        return {
            "status": (
                "INVALID_NORMALIZATION_TRACE" if errors
                else "NOT_APPLICABLE_NO_PROSE_ATTEMPT" if attempts == 0
                else "OK"
            ),
            "node": node,
            "attempt_count": attempts,
            "published_count": published,
            "rejected_count": rejected,
            "fallback_discarded_count": discarded,
            "call_failed_count": call_failed,
            "cancelled_count": cancelled,
            "raw_contract_adherent_count": raw_contract_adherent,
            "raw_contract_adherence_rate": (
                raw_contract_adherent / attempts if attempts else None
            ),
            "truncated_attempt_count": truncated,
            "semantic_repair_count": semantic_repairs,
            "raw_contract_nonadherent_count": raw_nonadherent,
            "normalization_trace_error_count": len(errors),
            "trace_complete": not errors,
            "published_policy_adverse": policy_adverse,
            "raw_contract_adverse": raw_adverse,
            "errors": errors,
        }

    observed_nodes = sorted({
        str(raw.get("node") or "").upper()
        for raw in records
        if isinstance(raw, Mapping) and str(raw.get("node") or "")
    })
    unexpected = sorted(set(observed_nodes) - set(expected))
    by_node = {
        node: summarize(
            [
                raw for raw in records
                if isinstance(raw, Mapping)
                and str(raw.get("node") or "").upper() == node
            ],
            node=node,
        )
        for node in expected
    }
    total_attempts = sum(item["attempt_count"] for item in by_node.values())
    trace_errors = sum(
        item["normalization_trace_error_count"] for item in by_node.values()
    )
    missing_expected_nodes = sorted(
        node for node, item in by_node.items() if item["attempt_count"] == 0
    )
    # Absent everywhere and absent in part are different facts, and conflating them made the
    # NOT_APPLICABLE_NO_PROSE_ATTEMPT status below unreachable: any zero-attempt node counted
    # as a trace error, so a structurally inapplicable arm was reported as a *corrupt* trace.
    # That is the wrong reason for the right refusal, and it biases HC mechanism attribution
    # conservatively for a reason the artifact then misdescribes.
    #
    # Partial coverage stays an error: an arm that ran prose at H but not at C really is
    # inconsistent with its own variant definition. Total absence is structural -- the graph
    # never presented the node -- and is reported as such. Either way the downstream integrity
    # gate still refuses to establish the contrast (matched.py records
    # PROSE_CONTROL_HAS_NO_OBSERVED_ATTEMPT), so this changes the stated reason, not the
    # fail-closed outcome.
    partially_missing = 0 < len(missing_expected_nodes) < len(expected)
    if unexpected:
        trace_errors += len(unexpected)
    if partially_missing:
        trace_errors += len(missing_expected_nodes)
    return {
        "schema_version": "prose_control_normalization_summary_v1",
        "status": (
            "INVALID_NORMALIZATION_TRACE" if trace_errors
            else "NOT_APPLICABLE" if not expected
            else "NOT_APPLICABLE_NO_PROSE_ATTEMPT" if total_attempts == 0
            else "OK"
        ),
        "expected_nodes": list(expected),
        "observed_nodes": observed_nodes,
        "attempt_count": total_attempts,
        "published_count": sum(item["published_count"] for item in by_node.values()),
        "rejected_count": sum(item["rejected_count"] for item in by_node.values()),
        "fallback_discarded_count": sum(
            item["fallback_discarded_count"] for item in by_node.values()
        ),
        "call_failed_count": sum(
            item["call_failed_count"] for item in by_node.values()
        ),
        "cancelled_count": sum(
            item["cancelled_count"] for item in by_node.values()
        ),
        "raw_contract_adherent_count": sum(
            item["raw_contract_adherent_count"] for item in by_node.values()
        ),
        "truncated_attempt_count": sum(
            item["truncated_attempt_count"] for item in by_node.values()
        ),
        "semantic_repair_count": sum(
            item["semantic_repair_count"] for item in by_node.values()
        ),
        "raw_contract_nonadherent_count": sum(
            item["raw_contract_nonadherent_count"] for item in by_node.values()
        ),
        "normalization_trace_error_count": trace_errors,
        "trace_complete": trace_errors == 0,
        "published_policy_adverse": bool(
            unexpected
            or any(item["published_policy_adverse"] for item in by_node.values())
        ),
        "raw_contract_adverse": bool(
            unexpected
            or any(item["raw_contract_adverse"] for item in by_node.values())
        ),
        "unexpected_nodes": unexpected,
        "missing_expected_nodes": missing_expected_nodes,
        "by_node": by_node,
    }


def score_direct_node_records(
    truth_body: dict,
    records: Sequence[dict],
    *,
    truth_chunker: str = "markdown_structure_v1",
    atom_support_index: dict | None = None,
) -> dict:
    """Score the reducer trace itself, never the final report it later influenced.

    Exact identity is intentionally conservative. An atom is in the direct denominator only
    when one of its exact supporting span ids was offered at that checkpoint. C_VISIBLE may
    replace this exact-span denominator with its checkpoint-derived visible projection in the
    campaign evaluator, but even there retention comes from ``published_span_ids``.
    """
    if not records:
        return {
            "status": "DIRECT_TRACE_UNAVAILABLE",
            "reason": "no direct_node_records/NODE_SELECTION event was frozen",
            "checkpoints": [],
        }
    del truth_chunker  # support identity now comes from the explicit cross-chunker index
    atomic_evidence = list(truth_body.get("atomic_evidence") or [])
    truth_atoms: dict[str, set[str]] = {}
    atom_weights: dict[str, float] = {}
    truth_weight_errors: list[str] = []
    for index, atom in enumerate(atomic_evidence):
        atom_id = str(atom.get("atom_id") or "")
        if not atom_id:
            truth_weight_errors.append(f"truth atom {index} has no atom_id")
            continue
        if atom_id in truth_atoms:
            truth_weight_errors.append(f"truth atom id {atom_id!r} is duplicated")
            continue
        truth_atoms[atom_id] = set(map(
            str, atom.get("supporting_span_ids") or ()))
        if "weight" not in atom:
            truth_weight_errors.append(
                f"truth atom {atom_id!r} has no frozen weight")
            continue
        raw_weight = atom.get("weight")
        if isinstance(raw_weight, bool):
            truth_weight_errors.append(
                f"truth atom {atom_id!r} has a boolean weight")
            continue
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError):
            truth_weight_errors.append(
                f"truth atom {atom_id!r} has a non-numeric weight")
            continue
        if not math.isfinite(weight) or weight < 0:
            truth_weight_errors.append(
                f"truth atom {atom_id!r} has an invalid weight {raw_weight!r}")
            continue
        atom_weights[atom_id] = weight
    atom_facets = {
        str(a["atom_id"]): str(a.get("facet_id") or "")
        for a in (truth_body.get("atomic_evidence") or [])
    }
    support_by_chunker = (
        (atom_support_index or {}).get("chunkers") or {}
    )
    support_tokenizer_sha256 = str(
        (atom_support_index or {}).get("tokenizer_sha256") or ""
    )
    prechunk_occurrences_by_atom = {
        str(atom_id): set(map(str, occurrence_ids or ()))
        for atom_id, occurrence_ids in (
            (atom_support_index or {}).get("prechunk_atom_occurrence_ids") or {}
        ).items()
    }
    prechunk_index_complete = (
        bool(truth_atoms)
        and set(truth_atoms).issubset(prechunk_occurrences_by_atom)
    )
    candidate_occurrences_by_chunker = (
        (atom_support_index or {}).get("candidate_span_occurrence_ids") or {}
    )
    critical = {
        str(a["atom_id"]) for a in (truth_body.get("atomic_evidence") or [])
        if a.get("critical")
    }
    contradiction_pairs = {
        (str(p["atom_id_a"]), str(p["atom_id_b"]))
        for p in (truth_body.get("contradiction_pairs") or ())
    }
    negative_atom_ids = {
        str(item.get("atom_id") or "")
        for item in (truth_body.get("negative_evidence") or ())
    }
    negative_atom_ids.discard("")
    negative_query_ids = {
        str(item.get("query_attempt_id") or "")
        for item in (truth_body.get("negative_evidence") or ())
    }
    known_gap_query_ids = {
        str(item.get("query_attempt_id") or "")
        for item in (truth_body.get("known_gaps") or ())
        if isinstance(item, dict)
        and str(item.get("status") or "") in {
            "FAILED", "TIMEOUT", "BLOCKED_BUDGET",
        }
    }
    negative_query_ids.discard("")
    known_gap_query_ids.discard("")
    unknown_negative_atoms = negative_atom_ids - set(truth_atoms)
    if unknown_negative_atoms:
        truth_weight_errors.append(
            "negative evidence references unknown truth atoms: "
            f"{sorted(unknown_negative_atoms)}"
        )
    guarded_query_ids = negative_query_ids | known_gap_query_ids
    checkpoint_rows: list[dict] = []
    invalid: list[str] = list(truth_weight_errors)

    def one_row(index: int, raw: dict) -> dict:
        row_errors: list[str] = []

        def reject(message: str) -> None:
            error = f"record {index}: {message}"
            row_errors.append(error)
            invalid.append(error)

        def gaps_query_ids(value) -> set[str]:
            ids: set[str] = set()
            for gap in value or ():
                if isinstance(gap, dict):
                    query_ids = gap.get("query_attempt_ids") or ()
                elif isinstance(gap, list | tuple) and len(gap) >= 2:
                    query_ids = gap[1] or ()
                else:
                    continue
                ids.update(map(str, query_ids))
            return ids

        def relations(value) -> list[tuple[str, str, str]]:
            out: list[tuple[str, str, str]] = []
            for relation in value or ():
                if isinstance(relation, dict):
                    triple = (
                        relation.get("span_id"), relation.get("facet_id"),
                        relation.get("role"),
                    )
                elif isinstance(relation, list | tuple) and len(relation) >= 3:
                    triple = relation[:3]
                else:
                    continue
                out.append(tuple(map(str, triple)))
            return out

        stage = str(raw.get("stage") or "single")
        chunker = str(raw.get("chunker") or "")
        node = str(raw.get("node") or "").upper()
        trace_tokenizer_sha256 = str(raw.get("tokenizer_sha256") or "")
        if support_tokenizer_sha256:
            if trace_tokenizer_sha256 != support_tokenizer_sha256:
                reject("trace tokenizer does not match the frozen support-index tokenizer")
        elif trace_tokenizer_sha256:
            reject("support index has no tokenizer identity for a tokenized trace")
        indexed = support_by_chunker.get(chunker)
        indexed_occurrences = candidate_occurrences_by_chunker.get(chunker)
        chunk_atom_index_complete = (
            isinstance(indexed, dict)
            and set(truth_atoms).issubset(map(str, indexed))
        )
        # C_VISIBLE uses a checkpoint-specific visible projection below; H uses this frozen
        # cross-chunker support index. Raw truth ids are never compared directly.
        comparable = bool(
            node.startswith("H")
            and isinstance(indexed, dict)
            and isinstance(indexed_occurrences, dict)
            and prechunk_index_complete
            and chunk_atom_index_complete
            and "offered_source_occurrence_ids" in raw
        )
        atom_spans = {
            atom_id: set(map(str, (indexed or {}).get(atom_id) or ()))
            for atom_id in truth_atoms
        }
        offered = set(map(str, raw.get("offered_span_ids") or ()))
        selected = set(map(str, raw.get("selected_span_ids") or ()))
        published = set(map(str, raw.get("published_span_ids") or ()))
        token_fields = (
            "offered_span_token_counts",
            "offered_evidence_tokens",
            "staged_rendered_tokens",
            "published_rendered_tokens",
        )
        semantic_token_fields = (
            "offered_material_tokens",
            "offered_context_tokens",
        )
        token_trace_present = all(field in raw for field in token_fields)
        if any(field in raw for field in token_fields) and not token_trace_present:
            reject("materialization token trace is partial")
        span_token_counts: dict[str, int] = {}
        if token_trace_present:
            raw_counts = raw.get("offered_span_token_counts")
            if not isinstance(raw_counts, list | tuple):
                reject("offered_span_token_counts is not a sequence")
                raw_counts = ()
            for item in raw_counts:
                if (
                    not isinstance(item, list | tuple)
                    or len(item) != 2
                    or not isinstance(item[0], str)
                    or not item[0]
                    or isinstance(item[1], bool)
                    or not isinstance(item[1], int)
                    or item[1] < 0
                    or item[0] in span_token_counts
                ):
                    reject("offered_span_token_counts has an invalid or duplicate entry")
                    continue
                span_token_counts[item[0]] = item[1]
            if set(span_token_counts) != offered:
                reject("span token counts do not cover exactly the offered set")

            def traced_nonnegative_int(field: str) -> int:
                value = raw.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    reject(f"{field} is not a non-negative integer")
                    return 0
                return value

            offered_evidence_tokens = traced_nonnegative_int(
                "offered_evidence_tokens"
            )
            if all(field in raw for field in semantic_token_fields):
                offered_material_tokens = traced_nonnegative_int(
                    "offered_material_tokens")
                offered_context_tokens = traced_nonnegative_int(
                    "offered_context_tokens")
            else:
                offered_material_tokens = offered_evidence_tokens
                offered_context_tokens = 0
            staged_rendered_tokens = traced_nonnegative_int(
                "staged_rendered_tokens"
            )
            published_rendered_tokens = traced_nonnegative_int(
                "published_rendered_tokens"
            )
            if offered_material_tokens != sum(span_token_counts.values()):
                reject("offered material tokens do not equal the frozen per-span sum")
            if (
                all(field in raw for field in semantic_token_fields)
                and offered_evidence_tokens + offered_context_tokens
                != offered_material_tokens
            ):
                reject("offered evidence plus context tokens do not equal material")
            if raw.get("fell_back") or raw.get("failure"):
                if published_rendered_tokens != 0:
                    reject("failed/fallback selection claims published rendered tokens")
            elif published and published_rendered_tokens <= 0:
                reject("published evidence has no rendered-token count")
            if published_rendered_tokens > staged_rendered_tokens:
                reject("published rendered tokens exceed staged rendered tokens")
        else:
            offered_evidence_tokens = None
            offered_material_tokens = None
            offered_context_tokens = None
            staged_rendered_tokens = None
            published_rendered_tokens = None
        span_derived_occurrences = {
            str(occurrence_id)
            for span_id in offered
            for occurrence_id in (
                (indexed_occurrences or {}).get(span_id) or ()
            )
        }
        offered_occurrences = set(map(
            str, raw.get("offered_source_occurrence_ids") or ()))
        if node.startswith("H") and isinstance(indexed_occurrences, dict):
            indexed_span_ids = set(map(str, indexed_occurrences))
            unknown_offered_spans = offered - indexed_span_ids
            if unknown_offered_spans:
                reject(
                    "offered H spans are absent from the frozen candidate-occurrence index: "
                    f"{sorted(unknown_offered_spans)}"
                )
            spans_without_occurrences = {
                span_id for span_id in offered
                if not (indexed_occurrences.get(span_id) or ())
            }
            if spans_without_occurrences:
                reject(
                    "offered H spans lack frozen source-occurrence bindings: "
                    f"{sorted(spans_without_occurrences)}"
                )
        if (
            node.startswith("H")
            and "offered_source_occurrence_ids" in raw
            and not span_derived_occurrences.issubset(offered_occurrences)
        ):
            reject("candidate spans reference source occurrences outside the H checkpoint")
        offered_queries = set(map(str, raw.get("offered_query_attempt_ids") or ()))
        published_queries = set(map(
            str, raw.get("published_query_attempt_ids")
            or raw.get("selected_query_attempt_ids") or ())) | gaps_query_ids(
                raw.get("published_gaps"))
        query_trace_complete = (
            "offered_query_attempt_ids" in raw
            and any(key in raw for key in (
                "published_query_attempt_ids", "published_gaps"))
        )
        published_relations = relations(raw.get("published_relations"))
        relation_trace_complete = "published_relations" in raw
        if not selected.issubset(offered):
            reject("selected ids outside offered set")
        if not published.issubset(selected):
            reject("published ids outside selected set")
        if query_trace_complete and not published_queries.issubset(offered_queries):
            reject("published query ids outside offered set")
        for span_id, _facet_id, role in published_relations:
            if span_id not in published:
                reject(f"published relation references unpublished span {span_id}")
            if role not in {"support", "contradict", "background"}:
                reject(f"published relation has invalid role {role!r}")
        handle_trace_status = str(
            raw.get("publication_handle_trace_status") or "")
        if node.startswith("H") and handle_trace_status:
            if handle_trace_status != "OK":
                for error in raw.get("publication_handle_trace_errors") or (
                    "publication handle trace is invalid",
                ):
                    reject(str(error))
        # Evidence span ids include chunker identity. An id produced by fixed-token chunks is
        # not comparable to a markdown-structure truth id even if the bytes overlap. Until the
        # evaluator has a frozen semantic cross-chunker binding map, that denominator is
        # unavailable -- never zero and never a pass.
        offered_atoms = (
            {atom_id for atom_id, spans in atom_spans.items() if spans & offered}
            if comparable else set()
        )
        published_atoms = (
            {atom_id for atom_id, spans in atom_spans.items() if spans & published}
            if comparable else set()
        )
        relevant_published_spans = (
            {
                span_id
                for atom_id in published_atoms
                for span_id in atom_spans.get(atom_id, set())
                if span_id in published
            }
            if comparable else set()
        )
        published_span_tokens = (
            sum(span_token_counts.get(span_id, 0) for span_id in published)
            if token_trace_present else None
        )
        relevant_published_span_tokens = (
            sum(
                span_token_counts.get(span_id, 0)
                for span_id in relevant_published_spans
            )
            if token_trace_present and comparable else None
        )
        return {
            "node": str(raw.get("node") or ""),
            "checkpoint_hash": str(raw.get("checkpoint_hash") or ""),
            "contract": str(raw.get("contract") or ""),
            "aggregation": str(raw.get("aggregation") or ""),
            "chunker": chunker,
            "tokenizer_sha256": trace_tokenizer_sha256,
            "stage": stage,
            "fell_back": bool(raw.get("fell_back")),
            "failure": raw.get("failure"),
            "offered_count": len(offered),
            "selected_count": len(selected),
            "published_count": len(published),
            "token_trace_status": "OK" if token_trace_present else "UNAVAILABLE",
            "publication_handle_trace_status":
                handle_trace_status or "UNAVAILABLE",
            "offered_material_tokens": offered_material_tokens,
            "offered_evidence_tokens": offered_evidence_tokens,
            "offered_context_tokens": offered_context_tokens,
            "staged_rendered_tokens": staged_rendered_tokens,
            "published_rendered_tokens": published_rendered_tokens,
            "published_span_tokens": published_span_tokens,
            "relevant_published_span_tokens": relevant_published_span_tokens,
            "selected_token_precision": (
                relevant_published_span_tokens / published_span_tokens
                if (
                    relevant_published_span_tokens is not None
                    and published_span_tokens is not None
                    and published_span_tokens > 0
                ) else None
            ),
            "materialization_ratio": (
                published_rendered_tokens / offered_material_tokens
                if (
                    published_rendered_tokens is not None
                    and offered_material_tokens is not None
                    and offered_material_tokens > 0
                ) else None
            ),
            "offered_query_attempt_count": len(offered_queries),
            "offered_source_occurrence_count": len(offered_occurrences),
            "published_query_attempt_count": len(published_queries),
            "offered_truth_atom_ids": sorted(offered_atoms),
            "published_truth_atom_ids": sorted(published_atoms),
            "direct_denominator_status": (
                "OK" if comparable else "DIRECT_DENOMINATOR_UNAVAILABLE_SUPPORT_INDEX"
            ),
            "exact_truth_recall": (
                len(published_atoms & offered_atoms) / len(offered_atoms)
                if comparable and offered_atoms else None
            ),
            "_offered_atoms": offered_atoms,
            "_published_atoms": published_atoms,
            "_offered_queries": offered_queries,
            "_published_queries": published_queries,
            "_published_relations": published_relations,
            "_span_token_counts": span_token_counts,
            "_published_span_tokens": published_span_tokens,
            "_relevant_published_span_tokens": relevant_published_span_tokens,
            "_published_rendered_tokens": published_rendered_tokens,
            "_offered_evidence_tokens": offered_evidence_tokens,
            "_atom_spans": atom_spans,
            "_offered_occurrences": offered_occurrences,
            "_comparable": comparable,
            "_query_trace_complete": query_trace_complete,
            "_relation_trace_complete": relation_trace_complete,
            "_errors": tuple(row_errors),
        }

    # Materialize every stage separately. For a hierarchical arm, local/map publication is an
    # intermediate shortlist; only the global/reduce publication is the reducer's final output.
    internal_rows = [one_row(index, raw) for index, raw in enumerate(records)]
    # H and C are distinct interventions and can both occur in the same arm.  A C single-stage
    # record must not make an otherwise comparable H denominator unavailable, nor may an H
    # global stage cause the C record to disappear.  Top-level exact metrics below are H-only;
    # the campaign evaluator adds C_VISIBLE projection metrics alongside them.
    h_rows = [row for row in internal_rows if row["node"].upper().startswith("H")]
    has_global = any(row["stage"] == "global" for row in h_rows)
    premap_rows = (
        [row for row in h_rows if row["stage"] == "local"]
        if has_global else
        [row for row in h_rows if row["stage"] != "local"]
    )
    final_rows = [
        row for row in h_rows
        if row["stage"] == "global" or (not has_global and row["stage"] != "local")
    ]
    h_structural_errors: list[str] = []
    if not final_rows and h_rows:
        message = "hierarchical trace has local/map stages but no global/final stage"
        h_structural_errors.append(message)
        invalid.append(message)
    comparable_final = (
        bool(final_rows)
        and bool(premap_rows)
        and not truth_weight_errors
        and all(row["_comparable"] for row in (*premap_rows, *final_rows))
    )
    final_offered_atoms = set().union(
        *(row["_offered_atoms"] for row in final_rows)) if final_rows else set()
    premap_candidate_atoms = set().union(
        *(row["_offered_atoms"] for row in premap_rows)) if premap_rows else set()
    published_atoms_all = set().union(
        *(row["_published_atoms"] for row in final_rows)) if final_rows else set()
    premap_occurrences = set().union(
        *(row["_offered_occurrences"] for row in premap_rows)) if premap_rows else set()
    prechunk_atoms_all = {
        atom_id for atom_id, occurrence_ids in prechunk_occurrences_by_atom.items()
        if occurrence_ids & premap_occurrences
    }
    offered_queries_all = set().union(
        *(row["_offered_queries"] for row in final_rows)) if final_rows else set()
    published_queries_all = set().union(
        *(row["_published_queries"] for row in final_rows)) if final_rows else set()
    published_atom_roles: dict[str, set[str]] = {}
    for row in final_rows:
        for span_id, facet_id, role in row["_published_relations"]:
            for atom_id, support_ids in row["_atom_spans"].items():
                if span_id in support_ids and (
                    not facet_id or facet_id == atom_facets.get(atom_id)
                ):
                    published_atom_roles.setdefault(atom_id, set()).add(role)

    def stage_summary(stage: str) -> dict | None:
        rows = [row for row in h_rows if row["stage"] == stage]
        if not rows:
            return None
        comparable = all(row["_comparable"] for row in rows)
        offered_atoms = set().union(*(row["_offered_atoms"] for row in rows))
        published_atoms = set().union(*(row["_published_atoms"] for row in rows))
        return {
            "stage": stage,
            "checkpoints": len(rows),
            "direct_denominator_status": (
                "OK" if comparable else "DIRECT_DENOMINATOR_UNAVAILABLE_SUPPORT_INDEX"
            ),
            "offered_truth_atom_ids": sorted(offered_atoms),
            "published_truth_atom_ids": sorted(published_atoms),
            "truth_recall": (
                len(published_atoms & offered_atoms) / len(offered_atoms)
                if comparable and offered_atoms else None
            ),
        }

    # Remove internal set-valued fields before serialization.
    checkpoint_rows = [
        {k: v for k, v in row.items() if not k.startswith("_")}
        for row in internal_rows
    ]
    offered_critical = prechunk_atoms_all & critical
    published_critical = published_atoms_all & critical
    offered_pairs = {
        pair for pair in contradiction_pairs
        if pair[0] in prechunk_atoms_all and pair[1] in prechunk_atoms_all
    }
    retained_pairs = {
        pair for pair in offered_pairs
        if pair[0] in published_atoms_all and pair[1] in published_atoms_all
    }
    offered_negative_queries = negative_query_ids & offered_queries_all
    retained_negative_queries = offered_negative_queries & published_queries_all
    offered_known_gaps = known_gap_query_ids & offered_queries_all
    retained_known_gaps = offered_known_gaps & published_queries_all
    offered_negative_atoms = negative_atom_ids & prechunk_atoms_all
    retained_negative_atoms = offered_negative_atoms & published_atoms_all
    prechunk_truth_weight = sum(
        atom_weights[atom_id] for atom_id in prechunk_atoms_all
        if atom_id in atom_weights
    )
    published_prechunk_truth_weight = sum(
        atom_weights[atom_id]
        for atom_id in published_atoms_all & prechunk_atoms_all
        if atom_id in atom_weights
    )
    token_trace_complete = bool(final_rows) and all(
        row["_published_span_tokens"] is not None
        and row["_relevant_published_span_tokens"] is not None
        and row["_published_rendered_tokens"] is not None
        and row["_offered_evidence_tokens"] is not None
        for row in final_rows
    )
    published_span_tokens = (
        sum(int(row["_published_span_tokens"]) for row in final_rows)
        if token_trace_complete else None
    )
    relevant_published_span_tokens = (
        sum(int(row["_relevant_published_span_tokens"]) for row in final_rows)
        if token_trace_complete else None
    )
    published_rendered_tokens = (
        sum(int(row["_published_rendered_tokens"]) for row in final_rows)
        if token_trace_complete else None
    )
    offered_evidence_tokens = (
        sum(int(row["_offered_evidence_tokens"]) for row in final_rows)
        if token_trace_complete else None
    )
    negative_atom_weight = sum(
        atom_weights[atom_id] for atom_id in offered_negative_atoms
        if atom_id in atom_weights
    )
    retained_negative_atom_weight = sum(
        atom_weights[atom_id] for atom_id in retained_negative_atoms
        if atom_id in atom_weights
    )
    negative_gap_weight = (
        negative_atom_weight
        + float(len(offered_negative_queries))
        + float(len(offered_known_gaps))
    )
    retained_negative_gap_weight = (
        retained_negative_atom_weight
        + float(len(retained_negative_queries))
        + float(len(retained_known_gaps))
    )
    relation_covered_atoms = {
        atom_id for atom_id in published_atoms_all
        if published_atom_roles.get(atom_id)
    }
    typed_pair_roles_ok = {
        pair for pair in retained_pairs
        if (
            (
                "support" in published_atom_roles.get(pair[0], set())
                and "contradict" in published_atom_roles.get(pair[1], set())
            )
            or (
                "contradict" in published_atom_roles.get(pair[0], set())
                and "support" in published_atom_roles.get(pair[1], set())
            )
        )
    }
    final_contracts = {
        str(row.get("contract") or "") for row in final_rows
        if row.get("contract")
    }
    typed_contract = bool(final_contracts & {"P1_TYPED", "P1_BRIDGE"})
    query_guard_complete = (
        not guarded_query_ids
        or not h_rows
        or (
            bool(final_rows)
            and all(row["_query_trace_complete"] for row in final_rows)
        )
    )
    relation_guard_complete = (
        not typed_contract
        or not offered_pairs
        or (
            bool(final_rows)
            and all(row["_relation_trace_complete"] for row in final_rows)
        )
    )
    selector_guards_complete = (
        comparable_final and query_guard_complete and relation_guard_complete
    )
    normalization_by_node = {
        node_name: selector_normalization_metrics(records, node=node_name)
        for node_name in ("H", "C")
    }
    h_errors = [
        error
        for row in h_rows
        for error in row.get("_errors", ())
    ] + h_structural_errors + truth_weight_errors
    h_metrics = {
        "applicable": bool(h_rows),
        "status": (
            "NOT_APPLICABLE" if not h_rows
            else "INVALID_DIRECT_TRACE" if h_errors
            else "DIRECT_DENOMINATOR_UNAVAILABLE" if not comparable_final
            else "OK"
        ),
        "errors": h_errors,
        "checkpoint_count": len(h_rows),
        "selector_normalization": normalization_by_node["H"],
        "final_stage": "global" if has_global else "single",
        "stage_metrics": {
            key: value for key, value in (
                ("map", stage_summary("local")),
                ("reduce", stage_summary("global")),
                ("single", stage_summary("single")),
            ) if value is not None
        },
        "direct_denominator_status": (
            "OK" if comparable_final
            else "DIRECT_DENOMINATOR_UNAVAILABLE_SUPPORT_INDEX"
        ),
        "prechunk_truth_atom_ids": sorted(prechunk_atoms_all),
        "candidate_covered_truth_atom_ids": sorted(premap_candidate_atoms),
        "selector_offered_truth_atom_ids": sorted(final_offered_atoms),
        "published_truth_atom_ids": sorted(published_atoms_all),
        "eligible_denominators": {
            "candidate_coverage":
                len(prechunk_atoms_all) if comparable_final else None,
            "selector_conditional_recall":
                len(final_offered_atoms) if comparable_final else None,
            "weighted_evidence_recall":
                prechunk_truth_weight if comparable_final else None,
            "total_published_prechunk_recall":
                len(prechunk_atoms_all) if comparable_final else None,
            "critical_truth_recall":
                len(offered_critical) if comparable_final else None,
            "contradiction_pair_recall":
                len(offered_pairs) if comparable_final else None,
            "grounded_negative_atom_recall":
                len(offered_negative_atoms) if comparable_final else None,
            "negative_query_trace_recall":
                len(offered_negative_queries) if comparable_final else None,
            "unresolved_gap_recall":
                len(offered_known_gaps) if comparable_final else None,
            "negative_gap_recall":
                negative_gap_weight if comparable_final else None,
            "typed_relation_coverage": (
                len(published_atoms_all) if typed_contract else 0
            ) if comparable_final else None,
            "contradiction_role_pair_recall": (
                len(offered_pairs) if typed_contract else 0
            ) if comparable_final else None,
        },
        "candidate_coverage": (
            len(premap_candidate_atoms & prechunk_atoms_all) / len(prechunk_atoms_all)
            if comparable_final and prechunk_atoms_all else None
        ),
        "selector_conditional_recall": (
            len(published_atoms_all & final_offered_atoms) / len(final_offered_atoms)
            if comparable_final and final_offered_atoms else None
        ),
        # Primary selector guard (§13.3): weight-aware recall over the raw-evidence
        # universe that existed before the arm's chunker/selector could discard anything.
        # The unweighted conditional recall above is a diagnostic, not a substitute.
        "weighted_evidence_recall": (
            published_prechunk_truth_weight / prechunk_truth_weight
            if comparable_final and prechunk_truth_weight > 0 else None
        ),
        # Token precision counts the selected evidence-body tokens that support at least one
        # frozen atom. Efficiency counts unique frozen truth weight retained per 100 exact
        # renderer tokens, so duplicate selections and header overhead cannot improve it.
        "selected_token_precision": (
            relevant_published_span_tokens / published_span_tokens
            if (
                comparable_final
                and token_trace_complete
                and published_span_tokens
            ) else None
        ),
        "weighted_truth_per_100_rendered_tokens": (
            100.0 * published_prechunk_truth_weight / published_rendered_tokens
            if (
                comparable_final
                and token_trace_complete
                and published_rendered_tokens
            ) else None
        ),
        "materialization_ratio": (
            published_rendered_tokens / offered_evidence_tokens
            if (
                comparable_final
                and token_trace_complete
                and offered_evidence_tokens
            ) else None
        ),
        "token_trace_complete": token_trace_complete,
        "published_rendered_tokens": published_rendered_tokens,
        "offered_evidence_tokens": offered_evidence_tokens,
        "pipeline_conditional_recall": (
            len(published_atoms_all & premap_candidate_atoms)
            / len(premap_candidate_atoms)
            if comparable_final and premap_candidate_atoms else None
        ),
        "total_published_prechunk_recall": (
            len(published_atoms_all & prechunk_atoms_all) / len(prechunk_atoms_all)
            if comparable_final and prechunk_atoms_all else None
        ),
        "critical_truth_recall": (
            len(published_critical) / len(offered_critical)
            if comparable_final and offered_critical else None
        ),
        "contradiction_pair_recall": (
            len(retained_pairs) / len(offered_pairs) if offered_pairs else None
        ),
        "grounded_negative_atom_recall": (
            len(retained_negative_atoms) / len(offered_negative_atoms)
            if offered_negative_atoms else None
        ),
        "negative_query_trace_recall": (
            len(retained_negative_queries) / len(offered_negative_queries)
            if offered_negative_queries else None
        ),
        "unresolved_gap_recall": (
            len(retained_known_gaps) / len(offered_known_gaps)
            if offered_known_gaps else None
        ),
        # Pool the three pre-registered opportunity types. Atom opportunities carry their
        # frozen truth weights; query/gap opportunities carry unit weight because the frozen
        # truth packet defines no query-specific weight.
        "negative_gap_recall": (
            retained_negative_gap_weight / negative_gap_weight
            if comparable_final and negative_gap_weight > 0 else None
        ),
        "typed_relation_coverage": (
            len(relation_covered_atoms) / len(published_atoms_all)
            if typed_contract and published_atoms_all else None
        ),
        "contradiction_role_pair_recall": (
            len(typed_pair_roles_ok) / len(offered_pairs)
            if typed_contract and offered_pairs else None
        ),
        "selector_guards_complete": selector_guards_complete,
        "query_guard_complete": query_guard_complete,
        "relation_guard_complete": relation_guard_complete,
    }
    return {
        "status": "INVALID_DIRECT_TRACE" if invalid else "OK",
        "errors": invalid,
        "checkpoints": checkpoint_rows,
        "final_stage": "global" if has_global else "single",
        "h_checkpoint_count": len(h_rows),
        "c_checkpoint_count": sum(
            row["node"].upper().startswith("C") for row in internal_rows),
        "stage_metrics": {
            key: value for key, value in (
                ("map", stage_summary("local")),
                ("reduce", stage_summary("global")),
                ("single", stage_summary("single")),
            ) if value is not None
        },
        "direct_denominator_status": (
            "OK" if comparable_final
            else "DIRECT_DENOMINATOR_UNAVAILABLE_SUPPORT_INDEX"
        ),
        "prechunk_truth_atom_ids": sorted(prechunk_atoms_all),
        "candidate_covered_truth_atom_ids": sorted(premap_candidate_atoms),
        "final_selector_offered_truth_atom_ids": sorted(final_offered_atoms),
        # Backward-readable name, now explicitly the treatment-independent prechunk
        # denominator rather than the selector's own candidate set.
        "offered_truth_atom_ids": sorted(prechunk_atoms_all),
        "published_truth_atom_ids": sorted(published_atoms_all),
        "candidate_coverage": (
            len(premap_candidate_atoms & prechunk_atoms_all) / len(prechunk_atoms_all)
            if comparable_final and prechunk_atoms_all else None
        ),
        "selector_conditional_recall": (
            len(published_atoms_all & final_offered_atoms) / len(final_offered_atoms)
            if comparable_final and final_offered_atoms else None
        ),
        "weighted_evidence_recall": h_metrics["weighted_evidence_recall"],
        "selected_token_precision": h_metrics["selected_token_precision"],
        "weighted_truth_per_100_rendered_tokens":
            h_metrics["weighted_truth_per_100_rendered_tokens"],
        "materialization_ratio": h_metrics["materialization_ratio"],
        "token_trace_complete": h_metrics["token_trace_complete"],
        "published_rendered_tokens": h_metrics["published_rendered_tokens"],
        "offered_evidence_tokens": h_metrics["offered_evidence_tokens"],
        "pipeline_conditional_recall": (
            len(published_atoms_all & premap_candidate_atoms)
            / len(premap_candidate_atoms)
            if comparable_final and premap_candidate_atoms else None
        ),
        "total_published_prechunk_recall": (
            len(published_atoms_all & prechunk_atoms_all) / len(prechunk_atoms_all)
            if comparable_final and prechunk_atoms_all else None
        ),
        "h_total_published_recall": (
            len(published_atoms_all & prechunk_atoms_all) / len(prechunk_atoms_all)
            if comparable_final and prechunk_atoms_all else None
        ),
        "exact_truth_recall": (
            len(published_atoms_all & final_offered_atoms) / len(final_offered_atoms)
            if comparable_final and final_offered_atoms else None
        ),
        "critical_truth_recall": (
            len(published_critical) / len(offered_critical)
            if comparable_final and offered_critical else None
        ),
        "contradiction_pair_recall": (
            len(retained_pairs) / len(offered_pairs) if offered_pairs else None
        ),
        "grounded_negative_atom_recall": h_metrics["grounded_negative_atom_recall"],
        "negative_query_trace_recall": h_metrics["negative_query_trace_recall"],
        "unresolved_gap_recall": h_metrics["unresolved_gap_recall"],
        "negative_gap_recall": h_metrics["negative_gap_recall"],
        "typed_relation_coverage": (
            len(relation_covered_atoms) / len(published_atoms_all)
            if typed_contract and published_atoms_all else None
        ),
        "contradiction_role_pair_recall": (
            len(typed_pair_roles_ok) / len(offered_pairs)
            if typed_contract and offered_pairs else None
        ),
        "selector_guards_complete": selector_guards_complete,
        "query_guard_complete": query_guard_complete,
        "relation_guard_complete": relation_guard_complete,
        "selector_guard_note": (
            "" if selector_guards_complete else
            "truth contains contradiction/negative-gap guards not addressable by the "
            "frozen direct-node trace"
        ),
        "selector_normalization_by_node": normalization_by_node,
        "by_node": {"H": h_metrics},
    }


def write_scores(score: TaskScore, path: Path) -> str:
    """Write one task's scores under the evaluator tree. Write-once.

    Re-scoring under a corrected truth packet is a *new version*, recomputed for every arm --
    never an edit in place, which would leave some arms scored against one answer key and some
    against another.
    """
    path = Path(path)
    body = score.content()
    body["content_sha256"] = score.content_sha256
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(body, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o440)
        return body["content_sha256"]
    except FileExistsError:
        pass
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"existing score artifact {path} is unreadable: {exc}") from exc
    recorded = str(existing.get("content_sha256") or "")
    existing_actual = sha256_hex(canonical_json({
        key: value for key, value in existing.items() if key != "content_sha256"
    }))
    if recorded != existing_actual:
        raise EvaluationError(
            f"{path} existing score artifact was edited: records {recorded!r}, "
            f"hashes to {existing_actual!r}"
        )
    if (
        recorded != body["content_sha256"]
        or canonical_json(existing) != canonical_json(body)
    ):
        raise EvaluationError(
            f"{path} already records different scores; a corrected truth packet is a new "
            "version recomputed for every arm, not an edit in place"
        )
    return body["content_sha256"]


def judge_policy_digest(judge_config: dict, *, requested_model: str, returned_model: str,
                        system_fingerprint: str = "") -> str:
    """The identity of the scoring policy: model, prompts, schema and blinding, together.

    Recorded per task so a mid-campaign model or prompt drift is detectable rather than mixed
    into one pooled number.
    """
    return sha256_hex(canonical_json({
        "requested_model": requested_model,
        "returned_model": returned_model,
        "system_fingerprint": system_fingerprint,
        "relation_prompt_sha256": str(
            judge_config.get("relation_prompt_sha256") or ""),
        "prompts": dict(sorted(judge_config.get("prompts", {}).items())),
        "blinding": dict(sorted(judge_config.get("blinding", {}).items())),
        "failure_policy": dict(sorted(judge_config.get("failure_policy", {}).items())),
        "model_sampling": dict(sorted(judge_config.get("model", {}).items())),
        "retry": dict(sorted(judge_config.get("retry", {}).items())),
    }))


def treatment_import_violations(modules: Iterable[str]) -> list[str]:
    """Names in ``modules`` that a treatment process must never import.

    Used by a test that walks the treatment path's import graph. The separation is enforced by
    filesystem ownership at runtime; this catches the mistake earlier, at the point someone adds
    the import.
    """
    forbidden = ("shapeflow_p1.evaluation.runner",
                 "shapeflow_p1.evaluation.visible_truth_projection",
                 "shapeflow_p1.evaluation.truth_builder",
                 "shapeflow_p1.evaluation.quality_metrics",
                 "shapeflow_p1.evaluation.citation_eval")
    return sorted({m for m in modules if m in forbidden})
