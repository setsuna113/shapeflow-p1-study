"""Three retrieval backends that must never be confused, plus a deterministic index.

The study's validity rests on treatment runs seeing a fixed world. That is enforced by
having three backends with distinct identities and no shared fallback path:

- ``TavilyCaptureBackend`` -- the only one that touches the live network, used solely during
  acquisition to build the frozen pool.
- ``ExactSnapshotReplayBackend`` -- replays a captured query by its exact snapshot id (query
  text + every parameter). A miss is an immediate, fatal error; it must never fall back to a
  live search, because that would let a checkpoint fork diverge from the world it was frozen
  in.
- ``FrozenTaskCorpusBackend`` -- answers a researcher's (possibly novel) query during an
  end-to-end run by deterministic BM25 over the task-local pool. Query text may legitimately
  differ between arms -- that divergence is part of the end-to-end effect being measured --
  but the *corpus* and the ranking are fixed, so the same query always yields byte-identical
  results.

BM25 is implemented here rather than pulled in, so ranking is fully specified and stable:
lowercase alnum tokenization, fixed k1/b, and ties broken by occurrence id so equal scores
never reorder run to run.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Protocol

from .snapshot_store import SnapshotStore
from .source_pool import SourcePool

__all__ = [
    "SearchRecord",
    "SearchBackend",
    "Bm25Index",
    "FrozenTaskCorpusBackend",
    "ExactSnapshotReplayBackend",
    "ReplayMiss",
]

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class SearchRecord:
    """A Tavily-shaped result, so downstream P0/P1 code is agnostic to the backend."""

    url: str
    title: str
    content: str  # the snippet
    raw_content: Optional[str]
    score: float
    occurrence_id: str


class SearchBackend(Protocol):
    def search(self, query: str, *, max_results: int) -> list[SearchRecord]: ...


class ReplayMiss(RuntimeError):
    """A replay query was not in the captured set. Fatal -- never fall back to live."""


class Bm25Index:
    """A small, deterministic BM25 index over (doc_id, text) pairs."""

    def __init__(
        self, documents: list[tuple[str, str]], *, k1: float = 1.5, b: float = 0.75
    ) -> None:
        self._k1 = k1
        self._b = b
        self._doc_ids: list[str] = []
        self._tokens: list[list[str]] = []
        self._tf: list[Counter] = []
        df: Counter = Counter()
        for doc_id, text in documents:
            toks = _tokenize(text)
            self._doc_ids.append(doc_id)
            self._tokens.append(toks)
            tf = Counter(toks)
            self._tf.append(tf)
            for term in tf:
                df[term] += 1
        self._n = len(documents)
        self._avgdl = (sum(len(t) for t in self._tokens) / self._n) if self._n else 0.0
        # BM25 idf with the +1 form, kept non-negative.
        self._idf = {
            term: max(0.0, math.log(1 + (self._n - d + 0.5) / (d + 0.5)))
            for term, d in df.items()
        }

    def search(self, query: str, *, top_k: int) -> list[tuple[str, float]]:
        q_terms = _tokenize(query)
        scored: list[tuple[str, float]] = []
        for i, doc_id in enumerate(self._doc_ids):
            tf = self._tf[i]
            dl = len(self._tokens[i])
            score = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                idf = self._idf.get(term, 0.0)
                freq = tf[term]
                denom = freq + self._k1 * (1 - self._b + self._b * dl / (self._avgdl or 1))
                score += idf * (freq * (self._k1 + 1)) / (denom or 1)
            if score > 0.0:
                scored.append((doc_id, score))
        # Deterministic order: score desc, then doc_id asc so equal scores never reorder.
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:top_k]


class FrozenTaskCorpusBackend:
    """Deterministic retrieval over one task's frozen, vendor-visible source pool."""

    def __init__(self, pool: SourcePool, snapshot_store: SnapshotStore) -> None:
        self._pool = pool
        self._store = snapshot_store
        self._by_occurrence = {o.occurrence_id: o for o in pool.vendor_visible}
        documents = []
        self._doc_text: dict[str, str] = {}
        for occ in pool.vendor_visible:
            raw = ""
            if occ.content_hash and occ.content_hash in pool.snapshots:
                raw = snapshot_store.read_text(pool.snapshots[occ.content_hash])
            # Index title + snippet + full text; snippet ensures no-raw-content pages rank.
            doc = "\n".join(filter(None, [occ.title or "", occ.snippet_content or "", raw]))
            self._doc_text[occ.occurrence_id] = raw
            documents.append((occ.occurrence_id, doc))
        self._index = Bm25Index(documents)

    def search(self, query: str, *, max_results: int) -> list[SearchRecord]:
        records = []
        for occ_id, score in self._index.search(query, top_k=max_results):
            occ = self._by_occurrence[occ_id]
            raw = self._doc_text.get(occ_id) or None
            records.append(
                SearchRecord(
                    url=occ.url,
                    title=occ.title or "",
                    content=occ.snippet_content or "",
                    raw_content=raw,
                    score=score,
                    occurrence_id=occ_id,
                )
            )
        return records


class ExactSnapshotReplayBackend:
    """Replays captured queries by exact snapshot id. A miss is fatal, by design."""

    def __init__(self, captured: dict[str, list[SearchRecord]]) -> None:
        # keyed by query_snapshot_id (hash of query text + every parameter)
        self._captured = captured

    def search_by_snapshot(self, query_snapshot_id: str) -> list[SearchRecord]:
        if query_snapshot_id not in self._captured:
            raise ReplayMiss(
                f"query_snapshot {query_snapshot_id} not captured; refusing to fall back to live"
            )
        return list(self._captured[query_snapshot_id])
