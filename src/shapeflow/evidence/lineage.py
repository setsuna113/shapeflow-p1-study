"""Integrity checks over spans: reconstruction and citation-lineage closure.

These are the structural guarantees preflight relies on. They check *form*, never truth: a
span must reconstruct from the bytes it claims to address and its citations must resolve to
real occurrences, but whether the span is the *right* evidence is an evaluator question and is
deliberately not touched here.

Both span namespaces are reconstructed, and that symmetry is load-bearing. ``RAW_SOURCE``
spans address a frozen snapshot's text by character offset; ``VISIBLE_MESSAGE`` spans address
the exact bytes a compressor saw, by byte offset. Skipping the visible namespace -- as this
module once did -- leaves C_VISIBLE's central claim ("the selector saw only what P0's
compressor saw") entirely unchecked: a span could declare ``byte_end=999`` against a 12-byte
message and still pass preflight. The compressor-only boundary is only real if it is
recomputed from bytes.
"""

from __future__ import annotations

from typing import Mapping, Optional

from ..hashing import sha256_hex

__all__ = ["reconstruction_errors", "lineage_closure_errors"]


def _short(span: dict) -> str:
    sid = span.get("span_id") or span.get("visible_span_id") or "?"
    return sid[:12]


def _raw_source_error(span: dict, snapshot_texts: Mapping[str, str]) -> Optional[str]:
    ch = span["content_hash"]
    text = snapshot_texts.get(ch)
    if text is None:
        return f"span {_short(span)}: snapshot {ch[:12]} not available"
    start, end = span["char_start"], span["char_end"]
    if not (0 <= start <= end <= len(text)):
        return f"span {_short(span)}: offsets [{start},{end}] out of bounds (len {len(text)})"
    if sha256_hex(text[start:end].encode("utf-8")) != span["text_sha256"]:
        return (
            f"span {_short(span)}: text hash mismatch (offsets no longer address the "
            "recorded bytes)"
        )
    return None


def _visible_message_error(span: dict, visible_views: Mapping[str, bytes]) -> Optional[str]:
    view_hash = span["visible_compressor_view_hash"]
    view = visible_views.get(view_hash)
    if view is None:
        # Not verifiable is not the same as verified. A visible span whose compressor view is
        # not on hand cannot be waved through -- that is precisely how an unbounded offset
        # would slip past.
        return f"span {_short(span)}: compressor view {view_hash[:12]} not available"
    start, end = span["byte_start"], span["byte_end"]
    if not (0 <= start <= end <= len(view)):
        return (
            f"span {_short(span)}: byte offsets [{start},{end}] out of bounds "
            f"(compressor view is {len(view)} bytes)"
        )
    if sha256_hex(view[start:end]) != span["exact_text_sha256"]:
        return (
            f"span {_short(span)}: exact-text hash mismatch (byte offsets no longer address "
            "the bytes the compressor saw)"
        )
    return None


def reconstruction_errors(
    spans: list[dict],
    snapshot_texts: Mapping[str, str],
    *,
    visible_views: Optional[Mapping[str, bytes]] = None,
) -> list[str]:
    """Return one error per span that does not reproduce the bytes it claims to address.

    ``snapshot_texts`` maps content_hash -> the snapshot's normalized text (``RAW_SOURCE``).
    ``visible_views`` maps visible_compressor_view_hash -> the exact compressor-visible bytes
    (``VISIBLE_MESSAGE``). A missing source, an out-of-bounds offset, or a hash mismatch each
    yields an error string; an empty list means every span reconstructs exactly.

    A span in an unrecognized namespace is an error, not a skip. Silently ignoring namespaces
    this function does not understand is how the visible-message hole stayed open.
    """
    views = visible_views or {}
    errors: list[str] = []
    for span in spans:
        namespace = span.get("namespace")
        if namespace == "RAW_SOURCE":
            err = _raw_source_error(span, snapshot_texts)
        elif namespace == "VISIBLE_MESSAGE":
            err = _visible_message_error(span, views)
        else:
            err = f"span {_short(span)}: unknown namespace {namespace!r}; cannot reconstruct"
        if err:
            errors.append(err)
    return errors


def lineage_closure_errors(spans: list[dict], known_occurrence_ids: set[str]) -> list[str]:
    """Return an error per span citing an occurrence that does not exist in the pool.

    A dangling citation is a fatal integrity error: it means a selected span points at a source
    the frozen world does not contain.
    """
    errors: list[str] = []
    for span in spans:
        for oid in span.get("source_occurrence_ids", []) or []:
            if oid not in known_occurrence_ids:
                errors.append(f"span {_short(span)}: cites unknown occurrence {oid[:12]}")
    return errors
