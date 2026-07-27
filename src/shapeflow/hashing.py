"""Content addressing and domain-separated identity derivation.

Two rules hold everywhere in this study:

**Domain separation.** Every derived ID commits to a domain tag, so two different kinds
of object can never share an ID even when their field values coincide. Without this, a
``SourceOccurrence`` whose fields happened to match an ``EvidenceSpan`` would produce the
same digest, and the two would alias in a content-addressed store.

**Digest IDs are not prompt IDs.** A full digest is 64 hex characters. Span identifiers
are rendered *into the selector prompt*, and selector prompt/decode tokens are the
quantity this study is measuring — paying ~16-32 tokens per candidate ID would inflate
the P1 arm's measured cost with an artifact of our own ID scheme and bias the very
comparison the study exists to make. So each span carries both:

- ``span_id`` — the full domain-separated digest. Global, content-addressed, used in the
  ledger, object store, artifacts and all evaluator-side joins.
- a short **local label** (``E7``) — unique only within one selector call's candidate
  set, cheap in tokens, and what the model actually reads and emits.

The adapter maps labels back to ``span_id`` deterministically and rejects any label
outside the offered candidate set, so the short form never weakens validation. Label
allocation lives in :mod:`shapeflow.evidence.identity` because it needs the candidate
set; this module only provides the digests.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

from .canonical import canonical_json

__all__ = [
    "ID_SCHEME_VERSION",
    "sha256_hex",
    "derive_id",
    "content_id",
    "query_snapshot_id",
    "occurrence_id",
    "merkle_root",
]

#: Bump only with a protocol version bump: changing it changes every derived ID.
ID_SCHEME_VERSION = "v1"

#: Deliberately still the old name, and deliberately not swept up in the package rename. This
#: byte string is domain separation for every derived ID, so editing it is not a rename -- it
#: silently re-derives every content id, query snapshot id and occurrence id in the system. It
#: changes once, together with ``ID_SCHEME_VERSION``, when the protocol document it belongs to
#: changes; a cosmetic edit here would be the same act without the version bump that makes it
#: visible.
_PREFIX = b"shapeflow-p1"
_SEP = b"\x1f"  # ASCII unit separator; forbidden inside domain tags below.


def sha256_hex(data: bytes) -> str:
    """SHA-256 of raw bytes as lowercase hex."""
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(f"sha256_hex expects bytes, got {type(data).__name__}")
    return hashlib.sha256(data).hexdigest()


def merkle_root(leaves: Sequence[str]) -> str:
    """A binary Merkle root over sorted leaf digests.

    A single hash over a concatenation would also detect change; a Merkle root additionally lets
    one leaf's membership be proved without republishing the whole set.
    """
    if not leaves:
        return sha256_hex(b"")
    level = [bytes.fromhex(h) for h in sorted(leaves)]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else left
            nxt.append(bytes.fromhex(sha256_hex(left + right)))
        level = nxt
    return level[0].hex()


def _preimage(domain: str, payload: bytes) -> bytes:
    if not domain or _SEP.decode("latin-1") in domain or "\x00" in domain:
        raise ValueError(f"invalid domain tag {domain!r}")
    return _SEP.join((_PREFIX, ID_SCHEME_VERSION.encode(), domain.encode("utf-8"), payload))


def derive_id(domain: str, fields: dict[str, Any]) -> str:
    """Derive a stable ID for ``fields`` within ``domain``.

    ``fields`` must contain exactly the values that define identity. Adding a field later
    changes every ID in that domain, which is a protocol change, not a refactor.
    """
    return sha256_hex(_preimage(domain, canonical_json(fields)))


def content_id(raw: bytes) -> str:
    """Identity of retrieved content, over its exact bytes.

    Two URLs serving byte-identical content share a ``content_id``. That is intended —
    it deduplicates storage — but it must never collapse citation lineage: the distinct
    ``SourceOccurrence`` records still point at this one blob. See
    :func:`occurrence_id`.
    """
    return sha256_hex(_preimage("content", raw))


def query_snapshot_id(
    *, query: str, params: dict[str, Any], api_version: str, schema_version: str
) -> str:
    """Identity of one exact retrieval request.

    Commits to the full parameter set, not just the query text: the same words sent with
    a different ``search_depth`` or ``max_results`` address a different slice of the web
    and must not replay as a cache hit for one another.
    """
    return derive_id(
        "query_snapshot",
        {
            "query": query,
            "params": params,
            "api_version": api_version,
            "schema_version": schema_version,
        },
    )


def occurrence_id(*, query_snapshot_id: str, rank: int, url: str) -> str:
    """Identity of "this URL, at this rank, in this specific query's results".

    Deliberately *not* a function of content. One page reachable from three queries is
    three occurrences over one ``content_id``; collapsing them to one would erase the
    citation lineage the quality metrics are computed against.
    """
    return derive_id(
        "occurrence",
        {"query_snapshot_id": query_snapshot_id, "rank": rank, "url": url},
    )
