"""The runner's view of a frozen world: where it lives, and how it is read back.

A task's world is published in two views to two different owners. The steward's manifest holds
every query including the ones that failed, both occurrence views, and the raw response hashes.
The runner gets only this one -- the vendor-visible occurrences, in vendor's first-occurrence
order.

That asymmetry is the point. The audit occurrence graph (later hits on an already-seen URL, with
their ranks and query provenance) is evaluator material: handing its rank and diversity metadata
to a P1 selector would give the treatment arm information vendor's own dedup never showed P0, and
any measured advantage would then be an artifact of the harness. The runner cannot read the file
that contains it, so this is a property of the filesystem layout rather than of anyone's care.

The pool file is self-verifying: it records the digest of its own body, and :func:`load_frozen_pool`
refuses a pool whose bytes no longer hash to what it claims. A silently edited world would make
every arm of that task incomparable to every other task's, and nothing downstream would show it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from ..canonical import canonical_json
from ..hashing import sha256_hex
from ..object_store import ObjectStore
from .snapshot_store import CanonicalSnapshot, SnapshotStore
from .source_pool import SourceOccurrence, SourcePool

if TYPE_CHECKING:  # a settings object is passed in; the world layer does not depend on campaign
    from ..campaign.settings import Settings

__all__ = ["runner_pool_path", "load_frozen_pool", "acquired_task_ids"]


def runner_pool_path(settings: "Settings", task_id: str) -> Path:
    return settings.path("frozen_corpus_for_runner") / "pools" / f"{task_id}.json"


def load_frozen_pool(settings: "Settings", task_id: str) -> tuple[SourcePool, SnapshotStore]:
    """Rebuild one task's vendor-visible pool for a treatment run.

    Returns the pool plus the snapshot store its content lives in. Nothing here can reach the
    audit occurrence graph: the file this reads does not contain it.
    """
    path = runner_pool_path(settings, task_id)
    body = json.loads(path.read_text(encoding="utf-8"))
    recorded = body.pop("pool_sha256", "")
    actual = sha256_hex(canonical_json(body))
    if recorded != actual:
        raise ValueError(f"{path} has been edited: records {recorded!r}, hashes to {actual!r}")

    pool = SourcePool(task_id=task_id)
    for record in body["occurrences"]:
        pool.occurrences.append(SourceOccurrence(
            occurrence_id=record["occurrence_id"],
            task_id=task_id,
            query_snapshot_id="",
            url=record["url"],
            title=record.get("title"),
            rank=0,
            score=None,
            published_date=None,
            content_hash=record.get("content_hash"),
            snippet_content=record.get("snippet_content"),
            visibility="VENDOR_VISIBLE",
            duplicate_of_occurrence_id=None,
            vendor_visible_order=record["vendor_visible_order"],
        ))
    for content_hash, snap in body["snapshots"].items():
        pool.snapshots[content_hash] = CanonicalSnapshot(
            content_hash=content_hash,
            raw_content_format=snap["raw_content_format"],
            byte_len=snap["byte_len"],
            object_ref=snap["object_ref"],
            normalization_version=snap["normalization_version"],
            fetched_at_utc=snap["fetched_at_utc"],
        )
    store = SnapshotStore(ObjectStore(settings.path("frozen_corpus_for_runner") / "objects"))
    return pool, store


def acquired_task_ids(settings: "Settings") -> Sequence[str]:
    """Task ids whose world is frozen and readable by the runner."""
    pools = settings.path("frozen_corpus_for_runner") / "pools"
    if not pools.exists():
        return []
    return sorted(p.stem for p in pools.glob("*.json"))
