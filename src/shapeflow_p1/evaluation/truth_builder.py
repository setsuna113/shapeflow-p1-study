"""Assembling machine-candidate TruthPackets (plan §13.2).

For an unattended run, DeepSeek may propose candidate atoms, but the assembly rule is strict: an
atom enters accepted truth ONLY if it binds to at least one exact source span. A fact the model
asserted without a span is dropped, a contradiction needs a span on each side, and a gap binds to a
real query attempt rather than the model's opinion. The result is always marked
MACHINE_CANDIDATE_PENDING_HUMAN_AUDIT with verifier_status PENDING, so downstream verdicts stay
provisional until the audit gate.

The building itself is deterministic given candidates; the judge/atomizer that produce candidates
are injected elsewhere, so this assembly is validated against the truth_packet schema without any
model call.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from ..canonical import canonical_json
from ..hashing import sha256_hex

__all__ = ["CandidateAtom", "assemble_truth_packet", "RejectedAtom"]


@dataclass(frozen=True)
class CandidateAtom:
    atom_id: str
    facet_id: str
    weight: float
    critical: bool
    supporting_span_ids: tuple[str, ...]
    known_unresolved: bool = False
    # Kept outside the schema-constrained packet (the packet has a parallel atom_texts
    # document), but carried through assembly so a binding verifier can judge the claim
    # against the exact bytes of each alleged supporting span.
    text: str = ""


@dataclass(frozen=True)
class RejectedAtom:
    atom_id: str
    reason: str


def assemble_truth_packet(
    task_id: str,
    candidates: Sequence[CandidateAtom],
    *,
    required_facets: Sequence[str],
    contradiction_pairs: Sequence[tuple[str, str]] = (),
    span_texts: Mapping[str, str] | None = None,
    semantic_verifier: Callable[[str, str, str], object] | None = None,
    negative_evidence: Sequence[dict] = (),
    known_gaps: Sequence[dict] = (),
    known_query_attempt_ids: set[str] | None = None,
) -> tuple[dict, list[RejectedAtom]]:
    """Return (truth_packet_dict, rejected). The dict validates against truth_packet.schema.json.

    Atoms with no supporting span are rejected. When exact ``span_texts`` are supplied, an
    invented id is not a binding. When a semantic verifier is supplied, at least one exact
    span must also entail the atom text. A contradiction pair survives only if both of its
    atoms were accepted, so a half-grounded contradiction is not presented as truth.

    The verifier is deliberately injected. Production uses the frozen evaluator judge; unit
    tests and human-audited rebuilds can supply a deterministic or human-backed verifier. It
    may return True, ``"entail"`` or ``"supported"``. An async verifier belongs in the
    campaign builder, which awaits it before calling this deterministic assembly function.
    """
    accepted: list[dict] = []
    rejected: list[RejectedAtom] = []
    accepted_ids: set[str] = set()
    accepted_facets: dict[str, str] = {}

    for atom in candidates:
        bound = list(dict.fromkeys(atom.supporting_span_ids))
        if span_texts is not None:
            bound = [span_id for span_id in bound if span_id in span_texts]
        if semantic_verifier is not None:
            verified: list[str] = []
            for span_id in bound:
                result = semantic_verifier(atom.text, span_texts[span_id], span_id)
                if hasattr(result, "__await__"):
                    raise TypeError(
                        "assemble_truth_packet is deterministic; await an async semantic "
                        "verifier in build_truth_for_task before assembly"
                    )
                if result is True or str(result).lower() in {
                    "entail", "entailed", "support", "supported",
                }:
                    verified.append(span_id)
            bound = verified
        if not bound:
            reason = "no semantically verified supporting span" if semantic_verifier else \
                "no supporting span"
            rejected.append(RejectedAtom(atom.atom_id, reason))
            continue
        accepted.append({
            "atom_id": atom.atom_id,
            "facet_id": atom.facet_id,
            "weight": atom.weight,
            "critical": atom.critical,
            "known_unresolved": atom.known_unresolved,
            "supporting_span_ids": bound,
        })
        accepted_ids.add(atom.atom_id)
        accepted_facets[atom.atom_id] = atom.facet_id

    pairs = [
        {"atom_id_a": a, "atom_id_b": b}
        for a, b in contradiction_pairs
        if a in accepted_ids and b in accepted_ids
    ]

    # An unresolved/absence claim is meaningful only when it names an attempt in the frozen
    # acquisition ledger.  Treat an omitted or empty registry as "no attempts are known", not
    # as permission to accept arbitrary model-proposed ids.
    known_query_attempt_ids = set(known_query_attempt_ids or ())
    negatives: list[dict] = []
    seen_negative: set[tuple[str, str, str]] = set()
    for item in negative_evidence:
        atom_id = str(item.get("atom_id") or "")
        facet_id = str(item.get("facet_id") or "")
        query_attempt_id = str(item.get("query_attempt_id") or "")
        # Absence is a factual claim, not a search-status inference. It enters only through a
        # span-grounded accepted atom in the same facet, plus a real frozen query attempt.
        if (
            not atom_id
            or atom_id not in accepted_ids
            or accepted_facets.get(atom_id) != facet_id
            or not query_attempt_id
        ):
            continue
        if query_attempt_id not in known_query_attempt_ids:
            continue
        key = (atom_id, facet_id, query_attempt_id)
        if key not in seen_negative:
            negatives.append({
                "atom_id": atom_id,
                "facet_id": facet_id,
                "query_attempt_id": query_attempt_id,
            })
            seen_negative.add(key)

    gaps: list[dict] = []
    seen_gaps: set[str] = set()
    for item in known_gaps:
        if not isinstance(item, dict):
            continue
        query_attempt_id = str(item.get("query_attempt_id") or "")
        status = str(item.get("status") or "")
        if (
            not query_attempt_id
            or query_attempt_id in seen_gaps
            or status not in {"FAILED", "TIMEOUT", "BLOCKED_BUDGET"}
            or query_attempt_id not in known_query_attempt_ids
        ):
            continue
        gaps.append({
            "query_attempt_id": query_attempt_id,
            "query_text": str(item.get("query_text") or ""),
            "status": status,
        })
        seen_gaps.add(query_attempt_id)

    packet = {
        "task_id": task_id,
        "required_facets": list(required_facets),
        "atomic_evidence": accepted,
        "contradiction_pairs": pairs,
        "negative_evidence": negatives,
        "known_gaps": gaps,
        "critical_items": [a["atom_id"] for a in accepted if a["critical"]],
        "authoring_method": "MACHINE_CANDIDATE_PENDING_HUMAN_AUDIT",
        "verifier_status": "PENDING",
        "content_sha256": "",
    }
    packet["content_sha256"] = sha256_hex(canonical_json({**packet, "content_sha256": ""}))
    return packet, rejected
