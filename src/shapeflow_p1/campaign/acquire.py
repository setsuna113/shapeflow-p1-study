"""`acquire`: call the search provider once per task, then never again.

Every arm of a task must search the same world. If each arm searched live, P1 -- which changes
the queries a researcher issues -- would also change which pages exist, and the two arms would
be compared across two different webs. So the provider is called exactly once per task, during
this phase, and every treatment run afterwards queries the frozen result.

A task's world is built in a staging directory and published with one rename. The order used
to be manifest-then-pool, and the manifest's presence was an unconditional skip -- so a crash
between the two left a task that every later run skipped and whose pool was never written, and
the only way out was to delete the write-once manifest that exists to stop exactly that.

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
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from ..acquire.manifest import (
    AcquisitionIntegrityError,
    QueryRecord,
    build_task_manifest,
    campaign_manifest,
    load_task_manifest,
    write_task_manifest,
)
from ..acquire.snapshot_store import CanonicalSnapshot, SnapshotStore
from ..acquire.source_pool import QueryResponse, SourceOccurrence, SourcePool, build_source_pool
from ..acquire.exa_client import ExaHTTPError, ExaParams, ExaPricing
from ..canonical import canonical_json
from ..hashing import sha256_hex
from ..object_store import ObjectStore
from .prepare import load_sealed_registry
from .settings import Settings

__all__ = [
    "AcquisitionOutcome",
    "queries_for_task",
    "exa_params_from",
    "exa_pricing_from",
    "acquire_all",
    "load_frozen_pool",
    "runner_pool_path",
    "task_world_is_complete",
]


@dataclass
class AcquisitionOutcome:
    tasks_acquired: int = 0
    tasks_skipped: int = 0
    queries_ok: int = 0
    queries_failed: int = 0
    queries_empty: int = 0
    blocked_budget: bool = False
    blocked_credential: bool = False
    spend_usd: float = 0.0
    campaign_manifest_path: Optional[Path] = None
    campaign_sha256: str = ""
    per_task: dict = field(default_factory=dict)
    #: Tasks whose manifest existed but whose world did not verify -- the crash case.
    partial_worlds: list = field(default_factory=list)
    #: Tasks still unacquired when the run stopped. Non-empty means no campaign manifest.
    incomplete: list = field(default_factory=list)


def exa_params_from(settings: Settings) -> ExaParams:
    """The pinned retrieval parameters. ``type: auto`` is refused by ExaParams itself."""
    block = settings.get("acquisition", "exa")
    return ExaParams(
        type=str(block["type"]),
        num_results=int(block["num_results_per_query"]),
        text_max_characters=int(block["text_max_characters"]),
        include_html_tags=bool(block["include_html_tags"]),
        highlights=bool(block["highlights"]),
        category=block.get("category") or None,
    )


def exa_pricing_from(settings: Settings) -> ExaPricing:
    """The published price list, as data, so the reservation and the report cite one source."""
    block = settings.get("acquisition", "exa_pricing")
    return ExaPricing(
        usd_per_request=float(block["usd_per_request"]),
        usd_per_extra_result=float(block["usd_per_extra_result"]),
        results_included=int(block["results_included"]),
        source=str(block["source"]),
        retrieved_utc=str(block["retrieved_utc"]),
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
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("pool_sha256") != body["pool_sha256"]:
            raise AcquisitionIntegrityError(
                f"{path} already holds a different world for {task_id} "
                f"({existing.get('pool_sha256')!r} vs {body['pool_sha256']!r}); two worlds "
                "under one task id cannot both be the frozen one"
            )
        return
    _write_json_atomic(path, body)
    # The runner reads this; it never writes it.
    os.chmod(path, 0o444)


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

    ``client_factory(task_id) -> ExaCaptureClient`` is injected so the transport (which goes
    through the provider) is supplied by the caller and tests need no network.

    A task is skipped only when its world is *complete and verifies* -- manifest digest,
    runner pool hash and every referenced blob. A half-written world is a gap to fill, not a
    task to skip forever.
    """
    registry, _ = load_sealed_registry(settings)
    wanted = set(settings.get("acquisition", "acquire_splits"))
    max_total = int(settings.get("acquisition", "queries_per_task", "max_total"))
    params = exa_params_from(settings)

    acquisition_dir = settings.path("acquisition")
    acquisition_dir.mkdir(parents=True, exist_ok=True)
    objects = ObjectStore(settings.path("frozen_corpus_for_runner") / "objects")
    store = SnapshotStore(objects)

    outcome = AcquisitionOutcome()
    digests: dict[str, str] = {}

    for task in registry["tasks"]:
        if task["split"] not in wanted:
            continue
        task_id = task["task_id"]
        manifest_path = acquisition_dir / f"{task_id}.json"
        complete, digest, reason = task_world_is_complete(settings, task_id, objects)
        if complete:
            digests[task_id] = digest
            outcome.tasks_skipped += 1
            continue
        if manifest_path.exists():
            # A manifest without a verifying world is the crash case. The stale manifest is
            # moved aside rather than deleted -- it is the only record of what the
            # interrupted attempt saw -- and the task is rebuilt. Skipping it forever was
            # the alternative, and the only escape from that was deleting the write-once
            # manifest whose whole job is to make deletion unnecessary.
            superseded = manifest_path.with_suffix(
                f".json.superseded-{sha256_hex(manifest_path.read_bytes())[:12]}")
            if not superseded.exists():
                manifest_path.replace(superseded)
            else:
                manifest_path.unlink()
            outcome.partial_worlds.append(f"{task_id}: {reason} (kept as {superseded.name})")

        queries = queries_for_task(task, max_total=max_total)
        client = client_factory(task_id)
        records: list[QueryRecord] = []
        responses: list[QueryResponse] = []

        for query in queries:
            qsid = client.query_snapshot_id(query)
            try:
                captured = await client.search(task_id, query)
            except ExaHTTPError as e:
                records.append(QueryRecord(query_snapshot_id=qsid, query_text=query,
                                           status="FAILED", raw_response_sha256="",
                                           usage={"http_status": e.status}))
                outcome.queries_failed += 1
                if e.status in (401, 403):
                    # A rejected credential fails identically every time. Continuing spends
                    # the request cap to learn the same fact once per query; this round did
                    # that 262 times.
                    outcome.blocked_credential = True
                    break
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
            outcome.spend_usd += captured.cost_usd
            if status == "SUCCESS":
                outcome.queries_ok += 1
            else:
                outcome.queries_empty += 1

        if outcome.blocked_credential:
            break

        pool = build_source_pool(task_id, responses, store, fetched_at_utc=fetched_at_utc)
        manifest = build_task_manifest(
            task_id=task_id,
            acquisition_spec_sha256=task.get("acquisition_spec_sha256", ""),
            pool=pool, queries=records, fetched_at_utc=fetched_at_utc,
            tavily_params=params.snapshot_params(),
        )
        # Pool first, manifest last. The manifest is what marks the world complete, so it is
        # the last thing written; a crash before it leaves a task that is rebuilt rather than
        # one that is skipped forever.
        _write_runner_pool(settings, task_id, pool, pool.snapshots)
        digests[task_id] = write_task_manifest(manifest, manifest_path)
        outcome.tasks_acquired += 1
        outcome.per_task[task_id] = {
            "acquisition_digest": digests[task_id],
            "merkle_root": manifest.merkle_root,
            "vendor_visible": len(pool.vendor_visible),
            "audit_only": len(pool.occurrences) - len(pool.vendor_visible),
        }
        if outcome.blocked_budget:
            break

    # The campaign manifest is write-once, and only written for a complete world. It used to
    # be rewritten on every invocation including one that broke out early, so its Merkle root
    # could silently describe a partial acquisition.
    expected = [t["task_id"] for t in registry["tasks"] if t["split"] in wanted]
    campaign_path = acquisition_dir / "campaign_acquisition.json"
    if set(digests) >= set(expected):
        body = campaign_manifest(
            digests,
            registry_sha256=registry.get("registry_sha256", ""),
            claim_scope=settings.claim_scope,
            corpus_tier=settings.corpus_tier,
        )
        if campaign_path.exists():
            existing = json.loads(campaign_path.read_text(encoding="utf-8"))
            if existing.get("campaign_acquisition_sha256") != body[
                    "campaign_acquisition_sha256"]:
                # A rebuilt task really is a different frozen world, so this is a new
                # manifest -- but the old one is set aside rather than overwritten, because
                # any run already scored against it needs to stay explicable.
                superseded = campaign_path.with_suffix(
                    ".json.superseded-"
                    f"{str(existing.get('campaign_acquisition_sha256'))[:12]}")
                if not superseded.exists():
                    campaign_path.replace(superseded)
                _write_json_atomic(campaign_path, body)
        else:
            _write_json_atomic(campaign_path, body)
        outcome.campaign_manifest_path = campaign_path
        outcome.campaign_sha256 = body["campaign_acquisition_sha256"]
    else:
        outcome.incomplete = sorted(set(expected) - set(digests))
    return outcome


def task_world_is_complete(
    settings: Settings, task_id: str, objects: Optional[ObjectStore] = None
) -> tuple[bool, str, str]:
    """Whether this task's frozen world exists *and* still verifies, end to end.

    Returns ``(complete, acquisition_digest, reason)``. The skip predicate used to be "the
    manifest file exists", which is true of a world whose pool was never written and whose
    blobs were never stored.
    """
    manifest_path = settings.path("acquisition") / f"{task_id}.json"
    if not manifest_path.exists():
        return False, "", "no manifest"
    try:
        manifest = load_task_manifest(manifest_path)
    except Exception as e:  # noqa: BLE001 - a manifest that will not verify is not a world
        return False, "", f"manifest does not verify: {type(e).__name__}: {e}"

    pool_path = runner_pool_path(settings, task_id)
    if not pool_path.exists():
        return False, "", "the runner pool was never published"
    try:
        body = json.loads(pool_path.read_text(encoding="utf-8"))
        recorded = body.pop("pool_sha256", "")
        if recorded != sha256_hex(canonical_json(body)):
            return False, "", "the runner pool has been edited"
    except (OSError, json.JSONDecodeError) as e:
        return False, "", f"the runner pool is unreadable: {e}"

    store = objects or ObjectStore(settings.path("frozen_corpus_for_runner") / "objects")
    for content_hash, snap in (body.get("snapshots") or {}).items():
        if not store.verify(snap["object_ref"]):
            return False, "", f"snapshot {content_hash[:12]} is missing or corrupt"
    return True, manifest["acquisition_digest"], ""


def _write_json_atomic(path: Path, body: dict) -> None:
    """Write through a temp file in the same directory, fsync, then rename."""
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(body, indent=2, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def acquired_task_ids(settings: Settings) -> Sequence[str]:
    """Task ids whose world is frozen and readable by the runner."""
    pools = settings.path("frozen_corpus_for_runner") / "pools"
    if not pools.exists():
        return []
    return sorted(p.stem for p in pools.glob("*.json"))
