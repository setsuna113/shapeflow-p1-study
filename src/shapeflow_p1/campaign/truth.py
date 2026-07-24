"""Building TruthPackets from the frozen world, under the evaluator identity.

DeepSeek proposes candidate atoms; the assembly rule decides what becomes truth, and it is
strict: an atom enters accepted truth only if it binds to at least one exact span of a frozen
page. A fact the model asserted without a span is dropped, a contradiction needs a span on each
side, and a gap binds to a real query attempt rather than to the model's opinion.

Two things this is careful about.

**Truth is derived from sources, never from an arm's output.** The candidate generator reads the
frozen source pool only. Deriving truth from what P0 produced would make P0 the gold standard,
and every arm would then be scored on how closely it reproduced P0 rather than on whether it was
right (plan §13.1).

**Nothing here is presented as human-verified.** Every packet is
``MACHINE_CANDIDATE_PENDING_HUMAN_AUDIT`` with ``verifier_status: PENDING``, and that label
travels into every score computed against it. An unaudited machine packet supporting a
paper-grade claim is exactly the failure this study is built to avoid.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from ..canonical import canonical_json
from ..evaluation.judge_client import DeepSeekJudge, JudgeUnavailable
from ..evaluation.truth_builder import CandidateAtom, assemble_truth_packet
from ..evidence.chunkers import WhitespaceTokenizer, markdown_structure_v1
from ..evidence.identity import build_evidence_span
from ..hashing import sha256_hex
from .acquire import load_frozen_pool
from .settings import Settings

__all__ = ["TRUTH_PROMPT_VERSION", "TruthResult", "build_truth_for_task", "truth_prompt_sha256"]

TRUTH_PROMPT_VERSION = "judge_truth_v1"

_TRUTH_SYSTEM = """\
You extract verifiable atomic facts from source excerpts. You never add anything the excerpts do
not state, and you never answer from your own knowledge.

Every fact you return must be traceable to at least one excerpt id you were given. A fact you
cannot bind to an excerpt must be omitted, not guessed.

Return only JSON matching the requested schema.
"""

_TRUTH_PROMPT = """\
Research question:
{question}

Required facets:
{facets}

Source excerpts (id: text):
{excerpts}

For each atomic fact the excerpts establish, return:
- "atom_id": a short unique slug
- "facet_id": the required facet it answers, or "other"
- "text": the fact, stated in one sentence, using the excerpts' own numbers and names
- "critical": true only if a report that got this wrong would mislead a reader materially
- "supporting_span_ids": the excerpt ids that state it (at least one, never invented)

Also return "contradictions": pairs of atom_ids that the excerpts genuinely disagree about.

Return JSON:
{{"atoms": [...], "contradictions": [["atom_id_a", "atom_id_b"]]}}
"""

_TRUTH_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["atoms"],
    "properties": {
        "atoms": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["atom_id", "facet_id", "text", "critical", "supporting_span_ids"],
            "properties": {
                "atom_id": {"type": "string", "minLength": 1},
                "facet_id": {"type": "string"},
                "text": {"type": "string", "minLength": 5},
                "critical": {"type": "boolean"},
                "supporting_span_ids": {"type": "array", "items": {"type": "string"}},
            },
        }},
        "contradictions": {"type": "array", "items": {
            "type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "string"},
        }},
    },
}


def truth_prompt_sha256() -> str:
    return sha256_hex(canonical_json({
        "version": TRUTH_PROMPT_VERSION, "system": _TRUTH_SYSTEM, "user": _TRUTH_PROMPT,
    }))


@dataclass
class TruthResult:
    task_id: str
    packet: dict
    atom_texts: dict
    rejected: list = field(default_factory=list)
    span_count: int = 0
    path: Optional[Path] = None


def _checker():
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(_TRUTH_SCHEMA)
    return validator.validate


def _excerpt_spans(settings: Settings, task_id: str, *, max_spans: int) -> list[dict]:
    """Chunk the task's frozen pages into addressable excerpts.

    Truth is bound to these ids, so an atom's provenance is a span of a page that was actually
    frozen -- not a page the model remembers.
    """
    pool, store = load_frozen_pool(settings, task_id)
    tokenizer = WhitespaceTokenizer()
    spans: list[dict] = []
    for occurrence in pool.vendor_visible:
        if not occurrence.content_hash:
            continue
        snapshot = pool.snapshots.get(occurrence.content_hash)
        if snapshot is None:
            continue
        text = store.read_text(snapshot)
        for chunk in markdown_structure_v1(text, tokenizer=tokenizer, max_tokens=320):
            spans.append(build_evidence_span(
                chunk, text, content_hash=occurrence.content_hash,
                source_occurrence_ids=[occurrence.occurrence_id],
                chunker_version="markdown_structure_v1",
            ))
            if len(spans) >= max_spans:
                return spans
    return spans


async def build_truth_for_task(
    settings: Settings,
    *,
    judge: DeepSeekJudge,
    task_id: str,
    question: str,
    required_facets: Sequence[str],
    max_spans: int = 60,
) -> TruthResult:
    """Propose, bind and assemble one task's TruthPacket. Write-once under the evaluator tree."""
    spans = _excerpt_spans(settings, task_id, max_spans=max_spans)
    span_text = {s["span_id"]: s["text"] for s in spans}
    excerpts = "\n".join(f"{sid}: {text[:400]}" for sid, text in span_text.items())

    try:
        response = await judge.judge(
            _TRUTH_SYSTEM,
            _TRUTH_PROMPT.format(
                question=question,
                facets="\n".join(f"- {f}" for f in required_facets) or "- (none stated)",
                excerpts=excerpts or "(no excerpts)",
            ),
            validate=_checker(),
        )
    except JudgeUnavailable as e:
        # No packet rather than an empty one: an empty packet would score every arm as perfect.
        raise JudgeUnavailable(f"truth for {task_id} unavailable: {e}") from e

    known = set(span_text)
    candidates: list[CandidateAtom] = []
    atom_texts: dict = {}
    for atom in response.data["atoms"]:
        bound = tuple(s for s in atom["supporting_span_ids"] if s in known)
        candidates.append(CandidateAtom(
            atom_id=str(atom["atom_id"]),
            facet_id=str(atom["facet_id"]),
            weight=1.0,
            critical=bool(atom["critical"]),
            supporting_span_ids=bound,
        ))
        atom_texts[str(atom["atom_id"])] = str(atom["text"])

    pairs = [tuple(p) for p in response.data.get("contradictions", []) if len(p) == 2]
    packet, rejected = assemble_truth_packet(
        task_id, candidates, required_facets=list(required_facets),
        contradiction_pairs=pairs,
    )
    packet["provenance"] = {
        "prompt_version": TRUTH_PROMPT_VERSION,
        "prompt_sha256": truth_prompt_sha256(),
        "requested_model": response.requested_model,
        "returned_model": response.returned_model,
        "system_fingerprint": response.system_fingerprint,
        "span_count": len(spans),
        "claim_scope": settings.claim_scope,
    }
    packet["content_sha256"] = sha256_hex(canonical_json(
        {**packet, "content_sha256": ""}))

    directory = settings.path("truth_packets")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{task_id}.json"
    if not path.exists():
        path.write_text(json.dumps(
            {"packet": packet, "atom_texts": atom_texts,
             "rejected": [{"atom_id": r.atom_id, "reason": r.reason} for r in rejected]},
            indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return TruthResult(task_id=task_id, packet=packet, atom_texts=atom_texts,
                       rejected=list(rejected), span_count=len(spans), path=path)
