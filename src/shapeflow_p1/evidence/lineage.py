"""Integrity checks over spans: reconstruction and citation-lineage closure.

These are the structural guarantees preflight relies on. They check *form*, never truth: a
span must reconstruct from the snapshot and its citations must resolve to real occurrences,
but whether the span is the *right* evidence is an evaluator question and is deliberately
not touched here.
"""

from __future__ import annotations

from ..hashing import sha256_hex

__all__ = ["reconstruction_errors", "lineage_closure_errors"]


def reconstruction_errors(spans: list[dict], snapshot_texts: dict[str, str]) -> list[str]:
    """Return an error per RAW_SOURCE span whose offsets do not reproduce its recorded hash.

    ``snapshot_texts`` maps content_hash -> the snapshot's normalized text. A missing snapshot,
    an out-of-bounds offset, or a text-hash mismatch each yields an error string; an empty list
    means every span reconstructs exactly.
    """
    errors: list[str] = []
    for span in spans:
        if span.get("namespace") != "RAW_SOURCE":
            continue
        ch = span["content_hash"]
        text = snapshot_texts.get(ch)
        if text is None:
            errors.append(f"span {span['span_id'][:12]}: snapshot {ch[:12]} not available")
            continue
        start, end = span["char_start"], span["char_end"]
        if not (0 <= start <= end <= len(text)):
            errors.append(
                f"span {span['span_id'][:12]}: offsets [{start},{end}] out of bounds (len {len(text)})"
            )
            continue
        actual = sha256_hex(text[start:end].encode("utf-8"))
        if actual != span["text_sha256"]:
            errors.append(
                f"span {span['span_id'][:12]}: text hash mismatch (offsets no longer address "
                "the recorded bytes)"
            )
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
                sid = span.get("span_id") or span.get("visible_span_id") or "?"
                errors.append(f"span {sid[:12]}: cites unknown occurrence {oid[:12]}")
    return errors
