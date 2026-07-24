"""`acquire`: call Tavily once per task, then never again.

Every arm of a task must search the same world. If each arm searched live, P1 -- which changes
the queries a researcher issues -- would also change which pages exist, and the two arms would
be compared across two different webs. So Tavily is called exactly once per task, during this
phase, and every treatment run afterwards queries the frozen result.

The world is written in two views, to two different owners:

``steward/acquisition/<task>.json``      the full manifest: both occurrence views, every query
                                          including the ones that failed, the raw response
                                          hashes and the Merkle root.
``runner/frozen_corpus/pools/<task>.json`` the vendor-visible occurrences only.

The audit occurrence graph -- the later hits on an already-seen URL, with their ranks and query
provenance -- is evaluator material. Handing its rank and diversity metadata to a P1 selector
would give the treatment arm information vendor's own dedup never showed P0, and any measured
advantage would then be an artifact of the harness. The runner cannot read the file that
contains it.

A query that returned nothing, or failed, is recorded as part of the frozen world. Keeping only
the successes would make a task with three dead queries indistinguishable from one with three
rich ones, and that difference is exactly what makes a task hard.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from ..acquire.manifest import (
    QueryRecord,
    build_task_manifest,
    campaign_manifest,
    load_task_manifest,
    write_task_manifest,
)
from ..acquire.snapshot_store import CanonicalSnapshot, SnapshotStore
from ..acquire.source_pool import QueryResponse, SourceOccurrence, SourcePool, build_source_pool
from ..acquire.tavily_client import TavilyHTTPError, TavilyParams
from ..canonical import canonical_json
from ..hashing import sha256_hex
from ..object_store import ObjectStore
from .prepare import load_sealed_registry
from .settings import Settings

__all__ = [
    "AcquisitionOutcome",
    "queries_for_task",
    "tavily_params_from",
    "acquire_all",
    "load_frozen_pool",
    "runner_pool_path",
]


@dataclass
class AcquisitionOutcome:
    tasks_acquired: int = 0
    tasks_skipped: int = 0
    queries_ok: int = 0
    queries_failed: int = 0
    queries_empty: int = 0
    blocked_budget: bool = False
    campaign_manifest_path: Optional[Path] = None
    campaign_sha256: str = ""
    per_task: dict = field(default_factory=dict)


def tavily_params_from(settings: Settings) -> TavilyParams:
    """The pinned retrieval parameters. ``auto_parameters`` is absent by construction."""
    block = settings.get("acquisition", "tavily")
    return TavilyParams(
        search_depth=str(block["search_depth"]),
        include_raw_content=str(block["include_raw_content"]),
        include_answer=bool(block["include_answer"]),
        include_usage=bool(block["include_usage"]),
        max_results=int(block["max_results_per_query"]),
        topic=str(block["topic"]),
    )


def queries_for_task(task: dict, *, max_total: int) -> list[str]:
    """The frozen query set: the question, one per facet, then the two probes.

    Order is fixed and the cap is applied from the end, so the whole-question query and the
    facet queries are never the ones dropped. Vendor's dedup is order-sensitive, so a stable
    order is also what makes the vendor-visible view reproducible.
    """
    spec = task["acquisition_spec"] if "acquisition_spec" in task else task
    queries: list[str] = list(spec.get("fixed_queries") or [])
    for probe_key in ("conflict_probe", "negative_or_gap_probe", "negative_probe",
                      "table_list_numeric_probe"):
        probe = spec.get(probe_key)
        if probe and probe not in queries:
            queries.append(str(probe))
    seen: set[str] = set()
    ordered: list[str] = []
    for query in queries:
        if query not in seen:
            seen.add(query)
            ordered.append(query)
    return ordered[:max_total]


def runner_pool_path(settings: Settings, task_id: str) -> Path:
    return settings.path("frozen_corpus_for_runner") / "pools" / f"{task_id}.json"


def _write_runner_pool(settings: Settings, task_id: str, pool: SourcePool,
                       snapshots: dict) -> None:
    """The runner's view: vendor-visible occurrences only, in vendor's first-occurrence order."""
    path = runner_pool_path(settings, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "task_id": task_id,
        "occurrences": [
            {
                "occurrence_id": o.occurrence_id,
                "url": o.url,
                "title": o.title,
                "snippet_content": o.snippet_content,
                "content_hash": o.content_hash,
                "vendor_visible_order": o.vendor_visible_order,
            }
            for o in pool.vendor_visible
        ],
        "snapshots": {
            content_hash: {
                "object_ref": snap.object_ref,
                "byte_len": snap.byte_len,
                "raw_content_format": snap.raw_content_format,
                "normalization_version": snap.normalization_version,
                "fetched_at_utc": snap.fetched_at_utc,
            }
            for content_hash, snap in sorted(snapshots.items())
        },
    }
    body["pool_sha256"] = sha256_hex(canonical_json(body))
    if path.exists():
        return
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_frozen_pool(settings: Settings, task_id: str) -> tuple[SourcePool, SnapshotStore]:
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


async def acquire_all(
    settings: Settings,
    *,
    client_factory,
    fetched_at_utc: str,
) -> AcquisitionOutcome:
    """Acquire and freeze every task in the configured splits. Idempotent per task.

    ``client_factory(task_id, call_key) -> TavilyCaptureClient`` is injected so the transport
    (which goes through the provider) is supplied by the caller and tests need no network.

    A task whose manifest already exists is skipped without a single request: the world was
    frozen once and re-fetching it would both spend the cap twice and produce a second,
    different world under the same task id.
    """
    registry, _ = load_sealed_registry(settings)
    wanted = set(settings.get("acquisition", "acquire_splits"))
    max_total = int(settings.get("acquisition", "queries_per_task", "max_total"))
    params = tavily_params_from(settings)

    acquisition_dir = settings.path("acquisition")
    acquisition_dir.mkdir(parents=True, exist_ok=True)
    store = SnapshotStore(ObjectStore(settings.path("frozen_corpus_for_runner") / "objects"))

    outcome = AcquisitionOutcome()
    digests: dict[str, str] = {}

    for task in registry["tasks"]:
        if task["split"] not in wanted:
            continue
        task_id = task["task_id"]
        manifest_path = acquisition_dir / f"{task_id}.json"
        if manifest_path.exists():
            digests[task_id] = load_task_manifest(manifest_path)["acquisition_digest"]
            outcome.tasks_skipped += 1
            continue

        queries = queries_for_task(task, max_total=max_total)
        client = client_factory(task_id)
        records: list[QueryRecord] = []
        responses: list[QueryResponse] = []

        for query in queries:
            qsid = client.query_snapshot_id(query)
            try:
                captured = await client.search(task_id, query)
            except TavilyHTTPError as e:
                records.append(QueryRecord(query_snapshot_id=qsid, query_text=query,
                                           status="FAILED", raw_response_sha256="",
                                           usage={"http_status": e.status}))
                outcome.queries_failed += 1
                if e.status == 429:
                    # The provider refuses on an exhausted budget with 429. Stopping here keeps
                    # already-frozen tasks intact instead of leaving half a world behind.
                    outcome.blocked_budget = True
                    break
                continue
            except Exception as e:  # noqa: BLE001 - sent, outcome unknown
                records.append(QueryRecord(query_snapshot_id=qsid, query_text=query,
                                           status="TIMEOUT",
                                           usage={"error": type(e).__name__}))
                outcome.queries_failed += 1
                continue

            status = "SUCCESS" if captured.response.results else "EMPTY"
            records.append(QueryRecord(
                query_snapshot_id=captured.query_snapshot_id,
                query_text=query,
                status=status,
                request_id=captured.request_id,
                response_time=captured.response_time,
                usage=dict(captured.usage),
                failed_results=list(captured.failed_results),
                raw_response_sha256=captured.raw_response_sha256,
                result_count=len(captured.response.results),
            ))
            responses.append(captured.response)
            if status == "SUCCESS":
                outcome.queries_ok += 1
            else:
                outcome.queries_empty += 1

        pool = build_source_pool(task_id, responses, store, fetched_at_utc=fetched_at_utc)
        manifest = build_task_manifest(
            task_id=task_id,
            acquisition_spec_sha256=task.get("acquisition_spec_sha256", ""),
            pool=pool, queries=records, fetched_at_utc=fetched_at_utc,
            tavily_params=params.snapshot_params(),
        )
        digests[task_id] = write_task_manifest(manifest, manifest_path)
        _write_runner_pool(settings, task_id, pool, pool.snapshots)
        outcome.tasks_acquired += 1
        outcome.per_task[task_id] = {
            "acquisition_digest": digests[task_id],
            "merkle_root": manifest.merkle_root,
            "vendor_visible": len(pool.vendor_visible),
            "audit_only": len(pool.occurrences) - len(pool.vendor_visible),
        }
        if outcome.blocked_budget:
            break

    body = campaign_manifest(
        digests,
        registry_sha256=registry.get("registry_sha256", ""),
        claim_scope=settings.claim_scope,
        corpus_tier=settings.corpus_tier,
    )
    campaign_path = acquisition_dir / "campaign_acquisition.json"
    campaign_path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outcome.campaign_manifest_path = campaign_path
    outcome.campaign_sha256 = body["campaign_acquisition_sha256"]
    return outcome


def acquired_task_ids(settings: Settings) -> Sequence[str]:
    """Task ids whose world is frozen and readable by the runner."""
    pools = settings.path("frozen_corpus_for_runner") / "pools"
    if not pools.exists():
        return []
    return sorted(p.stem for p in pools.glob("*.json"))
