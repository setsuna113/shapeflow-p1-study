"""The evidence manifest handed to a selector: candidate spans in a stable order.

The manifest is the deterministic, hashable description of what a selector was offered. Its
order is fixed (so two runs offer candidates identically) and its digest lets a checkpoint or
ledger record commit to exactly which candidate set produced a selection.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..hashing import derive_id

__all__ = ["EvidenceManifest", "build_manifest"]


@dataclass(frozen=True)
class EvidenceManifest:
    span_ids: tuple[str, ...]
    query_attempt_ids: tuple[str, ...]
    chunker_version: str

    def content(self) -> dict:
        return {
            "span_ids": list(self.span_ids),
            "query_attempt_ids": list(self.query_attempt_ids),
            "chunker_version": self.chunker_version,
        }

    @property
    def digest(self) -> str:
        return derive_id("raw_evidence_manifest", self.content())


def build_manifest(
    spans: list[dict], query_attempt_ids: list[str], *, chunker_version: str
) -> EvidenceManifest:
    """Build a manifest from span dicts, reading whichever id field each namespace uses. The
    span order is preserved as given -- callers pass spans in the deterministic presentation
    order (e.g. vendor-visible order, then char offset)."""
    ids: list[str] = []
    for span in spans:
        ids.append(span.get("span_id") or span["visible_span_id"])
    return EvidenceManifest(
        span_ids=tuple(ids),
        query_attempt_ids=tuple(query_attempt_ids),
        chunker_version=chunker_version,
    )
