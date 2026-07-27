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
    "EVIDENCE",
    "QUERY_ATTEMPT",
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
        # Metadata the renderer surfaces so a fragment can be read without widening the byte
        # range it addresses. context_refs are addressed and hashed like the span itself, so
        # preflight can re-derive them; a free-text context field would be an unbound channel
        # into the prompt that every integrity check would pass over.
        "heading_path": list(chunk.heading_path),
        "heading_refs": [
            {
                "content_hash": content_hash,
                "char_start": hs,
                "char_end": he,
                "text_sha256": sha256_hex(source_text[hs:he].encode("utf-8")),
            }
            for hs, he in chunk.heading_ranges
        ],
        "context_refs": [
            {
                "content_hash": content_hash,
                "char_start": cs,
                "char_end": ce,
                "text_sha256": sha256_hex(source_text[cs:ce].encode("utf-8")),
            }
            for cs, ce in chunk.context_ranges
        ],
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
    """Build a VISIBLE_MESSAGE span dict. Only TOOL_EVIDENCE may carry occurrences."""
    if kind not in {
        "TOOL_EVIDENCE",
        "TOOL_UNATTRIBUTED_CONTEXT",
        "MODEL_DERIVED_CONTEXT",
        "USER_CONTEXT",
    }:
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


EVIDENCE = "evidence"
QUERY_ATTEMPT = "query_attempt"
_KINDS = (EVIDENCE, QUERY_ATTEMPT)


@dataclass
class CandidateSet:
    """Maps spans and query attempts to short prompt labels and back.

    Labels are allocated in the given (deterministic) order: evidence spans as ``E1..En`` and
    query attempts as ``Q1..Qm``.

    **Two tables, not one.** ``E`` and ``Q`` are different *kinds* and each resolves only
    against its own table, so ``resolve`` takes the kind the caller expects. A single shared
    table would let a selector put ``Q1`` where a span id belongs -- pointing "evidence" at a
    search query -- or ``E1`` where a query attempt belongs, claiming a gap was probed by a
    piece of evidence. Both parse cleanly under one table and both corrupt the very selection
    semantics the study measures.

    **The set knows its span namespace.** ``namespace`` is ``RAW_SOURCE`` or
    ``VISIBLE_MESSAGE``. Building C_VISIBLE's set over ``VISIBLE_MESSAGE`` spans is what makes
    "the selector may not read raw page bytes" structurally unreachable rather than a
    convention someone has to remember.
    """

    _evidence_label_to_id: dict[str, str]
    _evidence_id_to_label: dict[str, str]
    _query_label_to_id: dict[str, str]
    _query_id_to_label: dict[str, str]
    namespace: str = "RAW_SOURCE"

    @classmethod
    def build(
        cls,
        span_ids: list[str],
        query_attempt_ids: Optional[list[str]] = None,
        *,
        namespace: str = "RAW_SOURCE",
    ) -> "CandidateSet":
        if namespace not in {"RAW_SOURCE", "VISIBLE_MESSAGE"}:
            raise ValueError(f"unknown span namespace {namespace!r}")
        e_l2i: dict[str, str] = {}
        e_i2l: dict[str, str] = {}
        for i, sid in enumerate(span_ids, start=1):
            label = f"E{i}"
            e_l2i[label] = sid
            e_i2l[sid] = label
        q_l2i: dict[str, str] = {}
        q_i2l: dict[str, str] = {}
        for j, qid in enumerate(query_attempt_ids or [], start=1):
            label = f"Q{j}"
            q_l2i[label] = qid
            q_i2l[qid] = label
        return cls(
            _evidence_label_to_id=e_l2i, _evidence_id_to_label=e_i2l,
            _query_label_to_id=q_l2i, _query_id_to_label=q_i2l,
            namespace=namespace,
        )

    def _table(self, kind: str) -> dict[str, str]:
        if kind == EVIDENCE:
            return self._evidence_label_to_id
        if kind == QUERY_ATTEMPT:
            return self._query_label_to_id
        raise ValueError(f"unknown candidate kind {kind!r}; expected one of {_KINDS}")

    def label_for(self, span_id: str) -> str:
        """The evidence label for a span id. Used by the renderer, which only renders spans."""
        return self._evidence_id_to_label[span_id]

    def label_for_query(self, query_attempt_id: str) -> str:
        return self._query_id_to_label[query_attempt_id]

    def resolve(self, label: str, *, kind: str) -> str:
        """Resolve a label the model emitted, or reject it as out-of-set.

        ``kind`` is required: the caller always knows whether it is reading a span position or
        a query-attempt position, and making it explicit is what keeps the two apart.
        """
        table = self._table(kind)
        if label not in table:
            raise OutOfSetLabel(f"{label} (expected a {kind} label)")
        return table[label]

    def resolve_all(self, labels: list[str], *, kind: str) -> list[str]:
        return [self.resolve(x, kind=kind) for x in labels]

    def contains(self, label: str, *, kind: str) -> bool:
        return label in self._table(kind)

    def __len__(self) -> int:
        return len(self._evidence_label_to_id) + len(self._query_label_to_id)
