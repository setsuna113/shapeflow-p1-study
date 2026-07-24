"""Span identity and the candidate set the selector actually reads.

Two namespaces, never mixed (mirroring ``schemas/evidence_span.schema.json``):

- ``RAW_SOURCE`` spans (``EvidenceSpan``) address bytes of a frozen snapshot. Visible to
  WEBPAGE_P1 and C_REGISTRY.
- ``VISIBLE_MESSAGE`` spans (``VisibleMessageSpan``) address bytes of the exact
  ``researcher_messages`` a compressor saw. The only namespace C_VISIBLE may select from.

A span id is a digest of the fields that define the span, so an equal span always gets an
equal id and a byte-drift changes it. But the id the *model* sees is a short label (``E7``),
allocated per candidate set, because span ids are rendered into the prompt and prompt tokens
are exactly what the study measures -- a 64-hex id per candidate would inflate the P1 arm's
cost with an artifact of our ID scheme. :class:`CandidateSet` allocates the labels
deterministically and resolves them back, rejecting any label outside the offered set, so the
cheap form loses no validation strength.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..hashing import derive_id, sha256_hex
from .chunkers import Chunk

__all__ = [
    "build_evidence_span",
    "build_visible_message_span",
    "CandidateSet",
    "OutOfSetLabel",
]


def _evidence_span_id(
    *, content_hash: str, char_start: int, char_end: int, text_sha256: str,
    kind: str, chunker_version: str, source_occurrence_ids: tuple[str, ...]
) -> str:
    return derive_id(
        "evidence_span",
        {
            "content_hash": content_hash,
            "char_start": char_start,
            "char_end": char_end,
            "text_sha256": text_sha256,
            "kind": kind,
            "chunker_version": chunker_version,
            "source_occurrence_ids": sorted(source_occurrence_ids),
        },
    )


def build_evidence_span(
    chunk: Chunk,
    source_text: str,
    *,
    content_hash: str,
    source_occurrence_ids: list[str],
    chunker_version: str,
) -> dict:
    """Build a RAW_SOURCE span dict (validates against evidence_span.schema.json).

    ``text_sha256`` is computed from the exact snapshot slice, so preflight can later re-derive
    it and catch any offset drift. A span must have at least one occurrence or it is a dangling
    citation.
    """
    if not source_occurrence_ids:
        raise ValueError("evidence span needs at least one source occurrence (no dangling citations)")
    span_text = chunk.text(source_text)
    text_hash = sha256_hex(span_text.encode("utf-8"))
    span_id = _evidence_span_id(
        content_hash=content_hash,
        char_start=chunk.char_start,
        char_end=chunk.char_end,
        text_sha256=text_hash,
        kind=chunk.kind,
        chunker_version=chunker_version,
        source_occurrence_ids=tuple(source_occurrence_ids),
    )
    return {
        "namespace": "RAW_SOURCE",
        "span_id": span_id,
        "content_hash": content_hash,
        "source_occurrence_ids": list(source_occurrence_ids),
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "token_start": None,
        "token_end": None,
        "text_sha256": text_hash,
        "kind": chunk.kind,
        "heading_path": list(chunk.heading_path),
        "token_len": chunk.token_len,
        "chunker_version": chunker_version,
    }


def build_visible_message_span(
    *,
    message_id: str,
    message_role: str,
    byte_start: int,
    byte_end: int,
    message_bytes: bytes,
    kind: str,
    visible_compressor_view_hash: str,
    source_occurrence_ids: Optional[list[str]] = None,
    token_len: int = 0,
) -> dict:
    """Build a VISIBLE_MESSAGE span dict. Only TOOL_EVIDENCE may carry occurrences (a citation
    binding); MODEL_DERIVED_CONTEXT and USER_CONTEXT may not."""
    if kind not in {"TOOL_EVIDENCE", "MODEL_DERIVED_CONTEXT", "USER_CONTEXT"}:
        raise ValueError(f"bad visible-message kind {kind!r}")
    if kind != "TOOL_EVIDENCE" and source_occurrence_ids:
        raise ValueError("only TOOL_EVIDENCE spans may carry source occurrences")
    exact = message_bytes[byte_start:byte_end]
    exact_hash = sha256_hex(exact)
    visible_span_id = derive_id(
        "visible_message_span",
        {
            "message_id": message_id,
            "byte_start": byte_start,
            "byte_end": byte_end,
            "exact_text_sha256": exact_hash,
            "kind": kind,
            "visible_compressor_view_hash": visible_compressor_view_hash,
        },
    )
    span: dict = {
        "namespace": "VISIBLE_MESSAGE",
        "visible_span_id": visible_span_id,
        "message_id": message_id,
        "message_role": message_role,
        "byte_start": byte_start,
        "byte_end": byte_end,
        "exact_text_sha256": exact_hash,
        "kind": kind,
        "visible_compressor_view_hash": visible_compressor_view_hash,
        "token_len": token_len,
    }
    if source_occurrence_ids:
        span["source_occurrence_ids"] = list(source_occurrence_ids)
    return span


class OutOfSetLabel(KeyError):
    """A selector emitted a label outside the offered candidate set. Never resolved
    leniently -- an out-of-set id is a hard rejection, which is what keeps the short-label
    scheme as strict as full ids."""


@dataclass
class CandidateSet:
    """Maps spans and query attempts to short prompt labels and back.

    Labels are allocated in the given (deterministic) order: evidence spans as ``E1..En`` and
    query attempts as ``Q1..Qm``. The span dicts may be either namespace; the id field is read
    accordingly.
    """

    _label_to_id: dict[str, str]
    _id_to_label: dict[str, str]

    @classmethod
    def build(
        cls, span_ids: list[str], query_attempt_ids: Optional[list[str]] = None
    ) -> "CandidateSet":
        label_to_id: dict[str, str] = {}
        id_to_label: dict[str, str] = {}
        for i, sid in enumerate(span_ids, start=1):
            label = f"E{i}"
            label_to_id[label] = sid
            id_to_label[sid] = label
        for j, qid in enumerate(query_attempt_ids or [], start=1):
            label = f"Q{j}"
            label_to_id[label] = qid
            id_to_label[qid] = label
        return cls(_label_to_id=label_to_id, _id_to_label=id_to_label)

    def label_for(self, span_id: str) -> str:
        return self._id_to_label[span_id]

    def resolve(self, label: str) -> str:
        """Resolve a label the model emitted to its full id, or reject it as out-of-set."""
        if label not in self._label_to_id:
            raise OutOfSetLabel(label)
        return self._label_to_id[label]

    def resolve_all(self, labels: list[str]) -> list[str]:
        return [self.resolve(x) for x in labels]

    def __contains__(self, label: str) -> bool:
        return label in self._label_to_id

    def __len__(self) -> int:
        return len(self._label_to_id)
