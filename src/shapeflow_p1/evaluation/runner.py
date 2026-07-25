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
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from ..canonical import canonical_json
from ..hashing import sha256_hex
from .atomizer import atomize_report
from .citation_eval import AtomCandidate, build_assessment
from .judge_client import JudgeUnavailable
from .quality_metrics import ContradictionPair, TruthAtom, TruthPacket, score_report

__all__ = [
    "ArmOutput",
    "TaskScore",
    "EvaluationError",
    "TreatmentReadTruth",
    "score_task",
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


@dataclass
class TaskScore:
    task_id: str
    truth_packet_sha256: str
    judge_policy_sha256: str
    claim_scope: str
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
            "per_arm": {k: dict(sorted(v.items())) for k, v in sorted(self.per_arm.items())},
            "unavailable": sorted(self.unavailable),
            "human_queue": [dict(sorted(item.items())) for item in self.human_queue],
        }

    @property
    def content_sha256(self) -> str:
        return sha256_hex(canonical_json(self.content()))


def assert_evaluator_only(role: Optional[str]) -> None:
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
    citation_supports: Callable[[str, str], Optional[bool]],
    judge_policy_sha256: str,
    claim_scope: str,
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
    )

    for output in sorted(outputs, key=lambda o: (o.arm_id, o.replicate_id)):
        try:
            claims = atomize_report(output.final_report)
            if hasattr(citation_supports, "bind"):
                # A citation label only means something inside the report that defined it,
                # so the resolver is bound per arm rather than per task.
                citation_supports.bind(
                    output.final_report, {c.claim_id: c.text for c in claims})
            assessment, queue = build_assessment(
                claims, candidates, judge=judge_relation,
                citation_supports=citation_supports,
                reported_unresolved_facets=frozenset(output.reported_unresolved_facets),
                terminal_failure=output.terminal_failure,
                empty_or_truncated=not output.final_report.strip(),
            )
        except JudgeUnavailable as e:
            # Never imputed, never dropped. The sample is recorded as unavailable and routed to
            # the human queue; a zero here would be a fabricated failure for that arm.
            score.unavailable.append(f"{output.arm_id}:{output.replicate_id}")
            score.human_queue.append({
                "task_id": output.task_id, "arm_id": output.arm_id,
                "replicate_id": output.replicate_id, "reason": f"JUDGE_UNAVAILABLE: {e}",
            })
            continue

        scores = score_report(packet, assessment)
        arm_key = f"{output.arm_id}:{output.replicate_id}"
        score.matched_atoms_by_arm[arm_key] = sorted({
            atom_id for c in assessment.claims
            if c.support_status == "SUPPORTED"
            for atom_id in c.matched_atom_ids
        })
        score.per_arm[arm_key] = {
            "variant_id": output.variant_id,
            "weighted_required_atom_recall": scores.weighted_required_atom_recall,
            "critical_atom_safety": scores.critical_atom_safety,
            "grounded_claim_precision": scores.grounded_claim_precision,
            "citation_correctness": scores.citation_correctness,
            "citation_association": scores.citation_association,
            "citation_completeness": scores.citation_completeness,
            "required_facet_coverage": scores.required_facet_coverage,
            "contradiction_handling": scores.contradiction_handling,
            "critical_harm": scores.critical_harm,
            "qualified_report": scores.qualified_report,
            "claim_scope": claim_scope,
        }
        score.human_queue.extend({
            "task_id": output.task_id, "arm_id": output.arm_id,
            "replicate_id": output.replicate_id, "claim_id": item.claim_id,
            "reason": item.reason,
        } for item in queue)
    return score


def write_scores(score: TaskScore, path: Path) -> str:
    """Write one task's scores under the evaluator tree. Write-once.

    Re-scoring under a corrected truth packet is a *new version*, recomputed for every arm --
    never an edit in place, which would leave some arms scored against one answer key and some
    against another.
    """
    path = Path(path)
    body = score.content()
    body["content_sha256"] = score.content_sha256
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("content_sha256") == body["content_sha256"]:
            return body["content_sha256"]
        raise EvaluationError(
            f"{path} already records different scores; a corrected truth packet is a new "
            "version recomputed for every arm, not an edit in place"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
        "prompts": dict(sorted(judge_config.get("prompts", {}).items())),
        "blinding": dict(sorted(judge_config.get("blinding", {}).items())),
        "failure_policy": dict(sorted(judge_config.get("failure_policy", {}).items())),
        "model": dict(sorted(judge_config.get("model", {}).items())),
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
