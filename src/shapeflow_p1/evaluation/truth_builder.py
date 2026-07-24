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

from dataclasses import dataclass, field
from typing import Sequence

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
) -> tuple[dict, list[RejectedAtom]]:
    """Return (truth_packet_dict, rejected). The dict validates against truth_packet.schema.json.

    Atoms with no supporting span are rejected. A contradiction pair survives only if both of its
    atoms were accepted, so a half-grounded contradiction is not presented as truth.
    """
    accepted: list[dict] = []
    rejected: list[RejectedAtom] = []
    accepted_ids: set[str] = set()

    for atom in candidates:
        if not atom.supporting_span_ids:
            rejected.append(RejectedAtom(atom.atom_id, "no supporting span"))
            continue
        accepted.append({
            "atom_id": atom.atom_id,
            "facet_id": atom.facet_id,
            "weight": atom.weight,
            "critical": atom.critical,
            "known_unresolved": atom.known_unresolved,
            "supporting_span_ids": list(atom.supporting_span_ids),
        })
        accepted_ids.add(atom.atom_id)

    pairs = [
        {"atom_id_a": a, "atom_id_b": b}
        for a, b in contradiction_pairs
        if a in accepted_ids and b in accepted_ids
    ]

    packet = {
        "task_id": task_id,
        "required_facets": list(required_facets),
        "atomic_evidence": accepted,
        "contradiction_pairs": pairs,
        "negative_evidence": [],
        "known_gaps": [],
        "critical_items": [a["atom_id"] for a in accepted if a["critical"]],
        "authoring_method": "MACHINE_CANDIDATE_PENDING_HUMAN_AUDIT",
        "verifier_status": "PENDING",
        "content_sha256": "",
    }
    packet["content_sha256"] = sha256_hex(canonical_json({**packet, "content_sha256": ""}))
    return packet, rejected
