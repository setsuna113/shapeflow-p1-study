"""The acquisition manifest: proof that every arm searched the same world.

One file per task, plus a campaign root, each a Merkle root over what was actually frozen. It
exists so "the arms saw the same sources" is a hash comparison rather than an assurance.

The acquisition digest is deliberately **separate** from the task digest. A task's wording and
the world its queries returned change for different reasons and at different times: re-reading
the question must not look like a changed corpus, and a re-acquisition must not look like a
re-worded task. Collapsing them into one hash would make both changes indistinguishable, and
the one that matters -- a world that moved under a frozen experiment -- is the one that would
be hidden.

The manifest also records what *failed*: a query that returned nothing, or errored, is part of
the frozen world. Recording only successes would let a task with three dead queries look
identical to one with three rich ones, and the difference is exactly what makes a task hard.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from ..canonical import canonical_json
from ..hashing import sha256_hex
from ..world.source_pool import SourcePool
from .task_registry import merkle_root

__all__ = [
    "QueryRecord",
    "TaskAcquisition",
    "build_task_manifest",
    "write_task_manifest",
    "campaign_manifest",
    "load_task_manifest",
    "AcquisitionIntegrityError",
]


class AcquisitionIntegrityError(RuntimeError):
    """A frozen world no longer matches the manifest that describes it."""


@dataclass(frozen=True)
class QueryRecord:
    """Everything about one acquisition query, successful or not."""

    query_snapshot_id: str
    query_text: str
    status: str                      # SUCCESS | EMPTY | FAILED | TIMEOUT | BLOCKED_BUDGET
    request_id: str = ""
    response_time: Optional[float] = None
    usage: dict = field(default_factory=dict)
    failed_results: list = field(default_factory=list)
    raw_response_sha256: str = ""
    result_count: int = 0

    def content(self) -> dict:
        return {
            "query_snapshot_id": self.query_snapshot_id,
            "query_text": self.query_text,
            "status": self.status,
            "request_id": self.request_id,
            "response_time": self.response_time,
            "usage": dict(sorted(self.usage.items())),
            "failed_results": list(self.failed_results),
            "raw_response_sha256": self.raw_response_sha256,
            "result_count": self.result_count,
        }

    @property
    def digest(self) -> str:
        return sha256_hex(canonical_json(self.content()))


@dataclass(frozen=True)
class TaskAcquisition:
    """One task's frozen world and the evidence that it is the world that was fetched."""

    task_id: str
    acquisition_spec_sha256: str
    queries: tuple[QueryRecord, ...]
    occurrences: tuple[dict, ...]
    snapshots: tuple[dict, ...]
    fetched_at_utc: str
    tavily_params: dict

    def content(self) -> dict:
        return {
            "task_id": self.task_id,
            "acquisition_spec_sha256": self.acquisition_spec_sha256,
            "fetched_at_utc": self.fetched_at_utc,
            "tavily_params": dict(sorted(self.tavily_params.items())),
            "queries": [q.content() for q in self.queries],
            "occurrences": list(self.occurrences),
            "snapshots": list(self.snapshots),
        }

    @property
    def merkle_root(self) -> str:
        """A root over the query, occurrence and snapshot leaves.

        A Merkle root rather than one flat hash so a single source's membership in the frozen
        world can be proved without republishing every page the task retrieved.
        """
        leaves = [q.digest for q in self.queries]
        leaves += [sha256_hex(canonical_json(o)) for o in self.occurrences]
        leaves += [sha256_hex(canonical_json(s)) for s in self.snapshots]
        return merkle_root(leaves)

    @property
    def acquisition_digest(self) -> str:
        return sha256_hex(canonical_json(self.content()))


def build_task_manifest(
    *,
    task_id: str,
    acquisition_spec_sha256: str,
    pool: SourcePool,
    queries: Sequence[QueryRecord],
    fetched_at_utc: str,
    tavily_params: dict,
) -> TaskAcquisition:
    """Freeze one task's acquisition into a manifest record.

    Both occurrence views are recorded -- the vendor-visible ones and the audit-only duplicates.
    The distinction is preserved in the record's ``visibility`` field rather than by omitting
    the duplicates: a later hit on a seen URL still carries citation lineage, and dropping it
    would leave two genuinely different sources sharing one citation.
    """
    occurrences = tuple(
        {
            "occurrence_id": o.occurrence_id,
            "query_snapshot_id": o.query_snapshot_id,
            "url": o.url,
            "title": o.title,
            "rank": o.rank,
            "score": o.score,
            "published_date": o.published_date,
            "content_hash": o.content_hash,
            "visibility": o.visibility,
            "duplicate_of_occurrence_id": o.duplicate_of_occurrence_id,
            "vendor_visible_order": o.vendor_visible_order,
        }
        for o in pool.occurrences
    )
    snapshots = tuple(
        {
            "content_hash": snap.content_hash,
            "raw_content_format": snap.raw_content_format,
            "byte_len": snap.byte_len,
            "object_ref": snap.object_ref,
            "normalization_version": snap.normalization_version,
            "fetched_at_utc": snap.fetched_at_utc,
        }
        for _, snap in sorted(pool.snapshots.items())
    )
    return TaskAcquisition(
        task_id=task_id,
        acquisition_spec_sha256=acquisition_spec_sha256,
        queries=tuple(queries),
        occurrences=occurrences,
        snapshots=snapshots,
        fetched_at_utc=fetched_at_utc,
        tavily_params=dict(tavily_params),
    )


def write_task_manifest(manifest: TaskAcquisition, path: Path) -> str:
    """Write one task's manifest. Write-once; returns the acquisition digest.

    Refusing to overwrite is what makes "acquired once" checkable. A re-acquisition produces a
    different world, and a world that could be replaced in place under a running experiment
    would make every earlier result incomparable to every later one without any trace.
    """
    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("acquisition_digest") == manifest.acquisition_digest:
            return manifest.acquisition_digest
        raise AcquisitionIntegrityError(
            f"{path} already records a different frozen world "
            f"({existing.get('acquisition_digest')!r} vs {manifest.acquisition_digest!r}); "
            "Tavily is called once per task and the world it returned is immutable"
        )
    body = manifest.content()
    body["merkle_root"] = manifest.merkle_root
    body["acquisition_digest"] = manifest.acquisition_digest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest.acquisition_digest


def load_task_manifest(path: Path) -> dict:
    """Read one task's manifest and re-derive its digest rather than trusting the recorded one."""
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    recorded_root = body.pop("merkle_root", "")
    recorded_digest = body.pop("acquisition_digest", "")
    actual = sha256_hex(canonical_json(body))
    if recorded_digest != actual:
        raise AcquisitionIntegrityError(
            f"{path} records acquisition_digest {recorded_digest!r} but hashes to {actual!r}; "
            "the frozen world has been edited"
        )
    body["merkle_root"] = recorded_root
    body["acquisition_digest"] = recorded_digest
    return body


def campaign_manifest(task_digests: dict[str, str], *, registry_sha256: str,
                      claim_scope: str, corpus_tier: str) -> dict:
    """One root over every task's frozen world, carried with the corpus labels.

    The labels travel with the manifest because a frozen world is only interpretable together
    with what the corpus it belongs to is allowed to support.
    """
    root = merkle_root(list(task_digests.values()))
    body = {
        "registry_sha256": registry_sha256,
        "corpus_tier": corpus_tier,
        "claim_scope": claim_scope,
        "task_acquisition_digests": dict(sorted(task_digests.items())),
        "acquisition_merkle_root": root,
    }
    body["campaign_acquisition_sha256"] = sha256_hex(canonical_json(body))
    return body
