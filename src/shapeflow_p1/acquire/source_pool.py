"""Building a task-local source pool with two strictly separated views.

This is where the study's central fairness invariant is enforced. Vendor
``tavily_search`` deduplicates results by URL, keeping the first occurrence, and presents
them in first-occurrence order (``utils.py`` step 2). A primary selector -- P0 or P1 -- may
see **only** that view (``VENDOR_VISIBLE``). The complete set of (query, rank, URL)
occurrences (``AUDIT_ONLY`` for the later duplicates) is retained for provenance and
evaluation, but handing its extra rank/diversity signal to P1 would give the treatment arm
information P0 never had, turning any measured advantage into an artifact of the harness.

So the dedup here reproduces vendor's exactly: iterate responses in query order, results in
rank order, first URL wins and takes the next ``vendor_visible_order``; every later hit on a
seen URL becomes an audit-only occurrence pointing at the winner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..hashing import occurrence_id
from .snapshot_store import CanonicalSnapshot, SnapshotStore

__all__ = ["RawResult", "QueryResponse", "SourceOccurrence", "SourcePool", "build_source_pool"]


@dataclass(frozen=True)
class RawResult:
    """One Tavily result row, before any dedup."""

    url: str
    title: str
    rank: int  # 1-based within its query
    snippet: str  # the provider 'content' field
    raw_content: Optional[str]  # full page text/markdown, or None
    score: Optional[float] = None
    published_date: Optional[str] = None


@dataclass(frozen=True)
class QueryResponse:
    """One acquisition query's frozen results, in rank order."""

    query_snapshot_id: str
    query_text: str
    results: tuple[RawResult, ...]


@dataclass(frozen=True)
class SourceOccurrence:
    occurrence_id: str
    task_id: str
    query_snapshot_id: str
    url: str
    title: Optional[str]
    rank: int
    score: Optional[float]
    published_date: Optional[str]
    content_hash: Optional[str]
    snippet_content: Optional[str]
    visibility: str  # VENDOR_VISIBLE | AUDIT_ONLY
    duplicate_of_occurrence_id: Optional[str]
    vendor_visible_order: Optional[int]


@dataclass
class SourcePool:
    task_id: str
    occurrences: list[SourceOccurrence] = field(default_factory=list)
    snapshots: dict[str, CanonicalSnapshot] = field(default_factory=dict)  # content_hash -> snapshot

    @property
    def vendor_visible(self) -> list[SourceOccurrence]:
        """The occurrences a primary selector may see, in vendor first-occurrence order."""
        visible = [o for o in self.occurrences if o.visibility == "VENDOR_VISIBLE"]
        return sorted(visible, key=lambda o: o.vendor_visible_order)

    @property
    def audit_graph(self) -> list[SourceOccurrence]:
        """Every occurrence, for provenance/evaluation only."""
        return list(self.occurrences)


def build_source_pool(
    task_id: str,
    responses: list[QueryResponse],
    snapshot_store: SnapshotStore,
    *,
    fetched_at_utc: str,
) -> SourcePool:
    """Freeze content and build both occurrence views for one task.

    ``responses`` must be in the frozen query order, and each response's ``results`` in rank
    order, because vendor's dedup is order-sensitive and we reproduce it byte-for-byte.
    """
    pool = SourcePool(task_id=task_id)
    seen_url_to_winner: dict[str, str] = {}
    next_visible_order = 0

    for response in responses:
        for result in response.results:
            oid = occurrence_id(
                query_snapshot_id=response.query_snapshot_id,
                rank=result.rank,
                url=result.url,
            )

            # Freeze raw content into a snapshot (deduped by content hash).
            content_hash: Optional[str] = None
            if result.raw_content:
                snap = snapshot_store.freeze(
                    result.raw_content,
                    raw_content_format="markdown",
                    fetched_at_utc=fetched_at_utc,
                )
                content_hash = snap.content_hash
                pool.snapshots.setdefault(content_hash, snap)

            if result.url not in seen_url_to_winner:
                # First occurrence of this URL: it is the vendor-visible one.
                seen_url_to_winner[result.url] = oid
                occ = SourceOccurrence(
                    occurrence_id=oid,
                    task_id=task_id,
                    query_snapshot_id=response.query_snapshot_id,
                    url=result.url,
                    title=result.title,
                    rank=result.rank,
                    score=result.score,
                    published_date=result.published_date,
                    content_hash=content_hash,
                    snippet_content=result.snippet,
                    visibility="VENDOR_VISIBLE",
                    duplicate_of_occurrence_id=None,
                    vendor_visible_order=next_visible_order,
                )
                next_visible_order += 1
            else:
                # A later hit on a seen URL: audit-only, pointing at the winner.
                occ = SourceOccurrence(
                    occurrence_id=oid,
                    task_id=task_id,
                    query_snapshot_id=response.query_snapshot_id,
                    url=result.url,
                    title=result.title,
                    rank=result.rank,
                    score=result.score,
                    published_date=result.published_date,
                    content_hash=content_hash,
                    snippet_content=result.snippet,
                    visibility="AUDIT_ONLY",
                    duplicate_of_occurrence_id=seen_url_to_winner[result.url],
                    vendor_visible_order=None,
                )
            pool.occurrences.append(occ)

    return pool
