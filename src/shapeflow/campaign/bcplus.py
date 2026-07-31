"""The BrowseComp-Plus campaign: one layer of the frozen split, every arm, on one lane.

This is the driver Week-1's ``screen.py`` was, rebuilt around a world that is not task-local.
The difference is the whole reason it is a separate module rather than a flag:

- **The world is loaded once, not per task.** A 100,195-document dense corpus with a 4096-dim
  index cannot be rebuilt per cell, and there is no per-task page set to enumerate. Every cell
  in a lane searches the same service, over loopback, and the service is the thing that pins
  which corpus that is -- by digest, not by path.
- **The tasks are the benchmark's own queries.** No authoring step, no acquisition step. The
  split was frozen before any of this code existed and is carved into design layers here,
  write-once.
- **The retrieval trace is part of the cell record.** Evidence recall is a docid-set
  intersection, and the treatment path deliberately never sees a docid, so the docids come off
  the client and are sliced per cell.

Everything else -- the ledger, the schedule, the paired blocks, the ITT accounting, the
publication canary's counters -- is the Week-1 apparatus unchanged, because none of it was ever
about where the pages came from.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from ..bench.bcplus.splits import (
    LAYER_SIZES,
    SEALED_LAYER,
    carve,
    load_manifest,
    write_manifest,
)
from ..bench.bcplus.tasks import load_questions, questions_digest, read_split_ids
from ..canonical import canonical_json
from ..hashing import sha256_hex
from ..protocol import verified_execution_binding
from .runner import CampaignRunner, RunnerConfig, write_status
from .schedule import cell_key
from .selector_client import SelectorModelCall
from .session import gpu_lease, open_run_ledger, provider_client_for
from .settings import Settings

__all__ = [
    "run_bcplus",
    "resolve_layer_tasks",
    "benchdata_root",
    "DEFAULT_BENCHDATA",
    "BCPlusCampaignError",
]

#: Where Phase 0 put the benchmark. Overridable, because a path is not an identity: what pins
#: the corpus is the shard digests the retrieval service verifies at startup, not this string.
DEFAULT_BENCHDATA = "/storage/sata/shapeflow/benchdata"


class BCPlusCampaignError(RuntimeError):
    """The campaign cannot be assembled as specified."""


def benchdata_root(override: Optional[Path] = None) -> Path:
    if override is not None:
        return Path(override)
    return Path(os.environ.get("SHAPEFLOW_BENCHDATA", DEFAULT_BENCHDATA))


def _split_manifest_path(settings: Settings) -> Path:
    return settings.path("runs") / "bcplus_split_plan.json"


def resolve_layer_tasks(
    settings: Settings, *, layer: str, benchdata: Optional[Path] = None,
) -> tuple[list[str], dict[str, str], dict]:
    """The task ids of one design layer, their questions, and the provenance of both.

    The carve is done once and written once. A second call either reproduces the same assignment
    byte for byte or refuses -- which is the only way "task-disjoint layers" survives a resumed
    campaign, since a layer silently recarved between two halves of a run would put the same task
    in two layers and nothing downstream would notice.
    """
    root = benchdata_root(benchdata)
    dev_path = root / "splits" / "bcplus_dev.txt"
    test_path = root / "splits" / "bcplus_test.txt"
    queries_path = root / "browsecomp-plus" / "repo" / "topics-qrels" / "queries.tsv"
    source_manifest = root / "splits" / "splits_manifest.json"

    if layer not in LAYER_SIZES and layer != SEALED_LAYER:
        raise BCPlusCampaignError(
            f"{layer!r} is not a design layer; known layers are "
            f"{sorted(LAYER_SIZES) + [SEALED_LAYER]}")

    body = json.loads(source_manifest.read_text(encoding="utf-8"))
    seed = int(body["seed"])
    plan = carve(read_split_ids(dev_path), read_split_ids(test_path), seed=seed)

    manifest_path = _split_manifest_path(settings)
    written = write_manifest(plan, manifest_path)
    # Read it back rather than trusting the object in hand: on the second and every later run
    # this is the only path that executes, and the plan a resumed campaign uses must be the one
    # on disk, not the one this process happened to recompute.
    plan = load_manifest(manifest_path)

    task_ids = list(plan.layers[layer])
    questions = load_questions(queries_path)
    missing = [t for t in task_ids if t not in questions]
    if missing:
        raise BCPlusCampaignError(
            f"{len(missing)} tasks in layer {layer!r} have no question in {queries_path}, e.g. "
            f"{missing[:5]}; the split and the topic file are different vintages")

    provenance = {
        "layer": layer,
        "split_plan_digest": written["digest"],
        "split_plan_seed": seed,
        "dev_sha256": plan.dev_sha256,
        "test_sha256": plan.test_sha256,
        "layer_size": len(task_ids),
        "questions_sha256": questions_digest(questions, task_ids),
        "queries_path": str(queries_path),
    }
    return task_ids, {t: questions[t] for t in task_ids}, provenance


async def run_bcplus(
    settings: Settings,
    *,
    repo: Path,
    layer: str,
    arms_block: str,
    retrieval_base_url: str,
    task_limit: Optional[int] = None,
    max_cells: Optional[int] = None,
    execution_binding_sha256: Optional[str] = None,
    phase_id: str = "bcplus",
    require_effective_freeze: bool = True,
    benchdata: Optional[Path] = None,
    stop_sentinel: Optional[Path] = None,
    run_id: Optional[str] = None,
) -> dict:
    """Run one layer x one arm block, under the same GPU lease the campaign holds."""
    binding = verified_execution_binding(repo, expected_digest=execution_binding_sha256)
    with gpu_lease(settings):
        return await _run_leased(
            settings,
            repo=repo,
            layer=layer,
            arms_block=arms_block,
            retrieval_base_url=retrieval_base_url,
            task_limit=task_limit,
            max_cells=max_cells,
            execution_binding_sha256=binding.digest,
            protocol_document_sha256=binding.protocol_sha,
            phase_id=phase_id,
            require_effective_freeze=require_effective_freeze,
            benchdata=benchdata,
            stop_sentinel=stop_sentinel,
            run_id=run_id,
        )


async def _run_leased(
    settings: Settings,
    *,
    repo: Path,
    layer: str,
    arms_block: str,
    retrieval_base_url: str,
    task_limit: Optional[int],
    max_cells: Optional[int],
    execution_binding_sha256: str,
    protocol_document_sha256: str,
    phase_id: str,
    require_effective_freeze: bool,
    benchdata: Optional[Path],
    stop_sentinel: Optional[Path],
    run_id: Optional[str],
) -> dict:
    from ..retrieval.client import RetrievalClient

    task_ids, questions, split_provenance = resolve_layer_tasks(
        settings, layer=layer, benchdata=benchdata)
    if task_limit is not None:
        # Prefix of the layer's own sorted order, not a sample: a "first N" that depended on a
        # shuffle nobody recorded would make a partial run un-resumable into a full one.
        task_ids = task_ids[:task_limit]
        questions = {t: questions[t] for t in task_ids}
    if not task_ids:
        raise BCPlusCampaignError(f"layer {layer!r} resolved to no tasks")

    world = RetrievalClient(
        retrieval_base_url, require_effective_freeze=require_effective_freeze)
    # ``verify`` returns the retriever block; the id that summarises it is a sibling of that
    # block in the health document, so both are read here rather than one being reconstructed.
    identity = world.verify()
    retriever_id = str(world.health().get("retriever_id", ""))

    ledger, store = open_run_ledger(settings)
    client = provider_client_for(settings, "runner")

    async def register(spec):
        await client.register_cell(
            cell_token=spec.cell_token, run_id=spec.run_id, task_id=spec.task_id,
            arm_id=spec.arm_id, variant_id=spec.variant_id, replicate_id=spec.replicate_id,
            work_key=spec.work_key, layer=str(settings.get("week1", "measurement", "layer")))

    async def fetch_work_summary(work_key: str):
        return await client.work_summary(
            work_key=work_key, require_isolated=settings.layer_is_serialized)

    def model_call_factory(cell_token: str, *, seed: int) -> SelectorModelCall:
        return SelectorModelCall(
            client, cell_token=cell_token, repo=repo,
            temperature=float(settings.get("stack", "sampling", "temperature")),
            top_p=float(settings.get("stack", "sampling", "top_p")),
            max_completion_tokens=int(settings.get("week1", "measurement",
                                                   "selector_max_completion_tokens")),
            seed=seed,
            guided_decoding=bool(settings.get("week1", "measurement", "guided_decoding")),
        )

    resolved_run_id = run_id or f"bcplus-{layer}-{arms_block}-{execution_binding_sha256[:12]}"
    config = RunnerConfig(
        run_id=resolved_run_id,
        provider_base_url=client.base_url,
        runner_token=client.token,
        lease_seconds=float(settings.get("week1", "runtime", "lease_seconds")),
        max_cells=max_cells,
        stop_sentinel=stop_sentinel,
    )
    runner = CampaignRunner(
        settings, ledger=ledger, store=store, config=config,
        execution_binding_sha256=execution_binding_sha256,
        protocol_document_sha256=protocol_document_sha256,
        model_call_factory=model_call_factory, register_cell=register,
        fetch_work_summary=fetch_work_summary, world_backend=world)

    arms = runner.arms_from_config(arms_block)
    manifest = runner.build_schedule(task_ids=task_ids, arms=arms, split=layer)
    manifest.notes["bcplus"] = {
        **split_provenance,
        "arms_block": arms_block,
        "arm_variant_ids": {a.arm_id: f"{a.page_variant}+{a.close_variant}" for a in arms},
        "retriever_id": retriever_id,
        "retrieval_freeze_digest": (identity.get("freeze") or {}).get("digest", ""),
        "retrieval_freeze_effective": bool((identity.get("freeze") or {}).get("effective")),
        "require_effective_freeze": require_effective_freeze,
        "index_num_docs": (identity.get("index") or {}).get("num_docs"),
        "top_k": (identity.get("freeze") or {}).get("top_k"),
        "task_ids_sha256": sha256_hex(canonical_json(sorted(task_ids))),
    }
    runner.freeze_schedule(
        manifest, settings.path("runs") / "schedules" / resolved_run_id / f"{layer}.json")
    ledger.create_run(resolved_run_id, execution_binding_sha256, json.dumps({
        "split": layer,
        "schedule_sha256": manifest.digest,
        "execution_binding_sha256": execution_binding_sha256,
        "protocol_document_sha256": protocol_document_sha256,
        "claim_scope": settings.claim_scope,
        "execution_semantics": "BCPLUS_E2E_ITT",
        "arms_block": arms_block,
    }, sort_keys=True))

    await runner.run_cells(manifest, phase_id=phase_id, split=layer, questions=questions)

    states = runner.cell_states(manifest, phase_id=phase_id, split=layer)
    body = runner.status(manifest, phase_id=phase_id, split=layer)
    body.update({
        "layer": layer,
        "arms_block": arms_block,
        "arms": [a.arm_id for a in arms],
        "tasks": len(task_ids),
        "schedule_sha256": manifest.digest,
        "retriever_id": retriever_id,
        "retrieval_cache": _retrieval_cache_stats(world),
        "cell_states": _state_counts(states),
        "ok": all(state == "COMMITTED" for state in states.values()),
        **split_provenance,
    })
    write_status(body, settings.path("runs") / f"STATUS_{resolved_run_id}.json")
    ledger.close()
    return body


def _retrieval_cache_stats(world) -> dict:
    """The query-embedding cache's hit rate, reported rather than assumed.

    R9: an embedding is a pure function of its text and the pinned encoder, so the cache cannot
    change *what* an arm retrieves -- but whichever arm runs second finds it warm, and that is a
    latency asymmetry with nothing to do with compression form. Recording it is what lets the
    write-up say whether the arms were warmed equally instead of hoping they were.
    """
    try:
        return dict(world.stats())
    except Exception as exc:  # noqa: BLE001 - an unavailable stat is reported, never invented
        return {"unavailable_reason": f"{type(exc).__name__}: {exc}"[:200]}


def _state_counts(states) -> dict:
    counts: dict = {}
    for state in states.values():
        counts[state] = counts.get(state, 0) + 1
    return dict(sorted(counts.items()))
