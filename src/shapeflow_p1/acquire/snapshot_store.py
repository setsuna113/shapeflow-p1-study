"""Freezing retrieved content into canonical, content-addressed snapshots.

A snapshot is the immutable record of one piece of retrieved content. Its identity is a
digest over its *normalized* bytes, so the same page fetched twice, or reached via two
URLs, resolves to one snapshot -- while the distinct citation lineage lives in the
occurrences that point at it.

Normalization is deliberately minimal (``normalize_v1``): decode as UTF-8, convert CRLF/CR
to LF, and apply Unicode NFC. That is enough to make trivially-different encodings of the
same text share a hash, without rewriting content in a way that could drop information P0
and P1 must both see. The full normalized bytes are stored; the ``max_content_length``
truncation both arms share happens later, at transform time, not here -- so the snapshot is
the complete page and neither arm is silently handed more of it than the other.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Optional

from ..hashing import content_id
from ..object_store import ObjectStore

__all__ = ["NORMALIZATION_VERSION", "CanonicalSnapshot", "normalize_v1", "SnapshotStore"]

NORMALIZATION_VERSION = "normalize_v1"


def normalize_v1(raw: str) -> bytes:
    """Return the canonical bytes of retrieved text content.

    Minimal on purpose: newline unification + NFC. Anything more aggressive risks changing
    what a selector or summarizer actually reads.
    """
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    return text.encode("utf-8")


@dataclass(frozen=True)
class CanonicalSnapshot:
    content_hash: str  # content_id over normalized bytes
    raw_content_format: str  # markdown | text | dom_html
    byte_len: int
    object_ref: str  # object-store key (plain sha256 of stored bytes)
    normalization_version: str
    fetched_at_utc: str


class SnapshotStore:
    """Creates and retrieves CanonicalSnapshots, backed by an ObjectStore."""

    def __init__(self, store: ObjectStore) -> None:
        self._store = store

    def freeze(
        self, raw_content: str, *, raw_content_format: str, fetched_at_utc: str
    ) -> CanonicalSnapshot:
        """Normalize, hash, and store ``raw_content``. Idempotent: identical content yields
        the same snapshot and one stored blob."""
        if raw_content_format not in {"markdown", "text", "dom_html"}:
            raise ValueError(f"unexpected raw_content_format {raw_content_format!r}")
        normalized = normalize_v1(raw_content)
        ref = self._store.put_bytes(normalized)
        return CanonicalSnapshot(
            content_hash=content_id(normalized),
            raw_content_format=raw_content_format,
            byte_len=len(normalized),
            object_ref=ref.key,
            normalization_version=NORMALIZATION_VERSION,
            fetched_at_utc=fetched_at_utc,
        )

    def read_text(self, snapshot: CanonicalSnapshot) -> str:
        """Return the snapshot's normalized text, verifying integrity via the object store."""
        return self._store.get_bytes(snapshot.object_ref).decode("utf-8")
