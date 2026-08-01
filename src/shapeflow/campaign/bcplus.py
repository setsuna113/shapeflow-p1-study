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
from ..scoped_paths import safe_scope_component
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

#: The competence pilot's task set (protocol §5.3: "100 ITD tasks"). Not a layer of the frozen
#: carve -- adding one would change ``SplitPlan.digest`` and break the write-once manifest
#: against the prereg-pinned split -- but a *derived* set, ranked by the same digest discipline
#: ``splits._ranked`` uses so it is reproducible without depending on the interpreter's PRNG.
#:
#: Drawn from b1_confirm and b2, deliberately **not** b1_select. The pilot is a P0 measurement of
#: whether the frozen retriever supports the task at all; b1_select is where the champion form is
#: chosen. Keeping them disjoint means the tasks that select a form were never used to validate
#: the retriever that serves it, and it costs nothing -- ITD holds 360 tasks outside b1_select.
PILOT_LAYER = "pilot_competence"
PILOT_SOURCE_LAYERS = ("b1_confirm", "b2")
PILOT_TASKS = 100


class BCPlusCampaignError(RuntimeError):
    """The campaign cannot be assembled as specified."""


def benchdata_root(override: Optional[Path] = None) -> Path:
    if override is not None:
        return Path(override)
    return Path(os.environ.get("SHAPEFLOW_BENCHDATA", DEFAULT_BENCHDATA))


def _split_manifest_path(settings: Settings) -> Path:
    return settings.path("runs") / "bcplus_split_plan.json"


def _pilot_task_ids(plan, seed: int) -> list[str]:
    """The competence pilot's 100 tasks, derived from the frozen carve and nothing else."""
    pool = [q for name in PILOT_SOURCE_LAYERS for q in plan.layers[name]]
    ranked = sorted(pool, key=lambda q: sha256_hex(f"{seed}:competence-pilot:{q}".encode()))
    if len(ranked) < PILOT_TASKS:
        raise BCPlusCampaignError(
            f"the competence pilot needs {PILOT_TASKS} ITD tasks and the pool holds "
            f"{len(ranked)}; shrinking the pilot is a design change, not an accommodation")
    return sorted(ranked[:PILOT_TASKS])


def resolve_layer_tasks(
    settings: Settings, *, layer: str, repo: Path, benchdata: Optional[Path] = None,
) -> tuple[list[str], dict[str, str], dict]:
    """The task ids of one design layer, their questions, and the provenance of both.

    The carve is done once and written once. A second call either reproduces the same assignment
    byte for byte or refuses -- which is the only way "task-disjoint layers" survives a resumed
    campaign, since a layer silently recarved between two halves of a run would put the same task
    in two layers and nothing downstream would notice.
    """
    from ..config import load_config

    root = benchdata_root(benchdata)
    dev_path = root / "splits" / "bcplus_dev.txt"
    test_path = root / "splits" / "bcplus_test.txt"
    queries_path = root / "browsecomp-plus" / "repo" / "topics-qrels" / "queries.tsv"

    if layer not in LAYER_SIZES and layer not in (SEALED_LAYER, PILOT_LAYER):
        raise BCPlusCampaignError(
            f"{layer!r} is not a design layer; known layers are "
            f"{sorted(LAYER_SIZES) + [SEALED_LAYER, PILOT_LAYER]}")

    # The seed and both split digests come from the frozen prereg, which is hashed into the
    # execution binding -- not from a manifest sitting next to the data. The loud failure here is
    # a missing file; the quiet one is a *regenerated* split file whose ids are in a different
    # order, because ``carve`` hashes the ids in the order given, so a reordered file carves
    # different layers under the same name and every recall number afterwards is computed against
    # a different task set. Checking the digests makes that impossible rather than unlikely.
    prereg, _ = load_config(Path(repo) / "configs" / "prereg.yaml")
    pinned = prereg["design"]["splits"]
    seed = int(pinned["seed"])
    dev_ids, test_ids = read_split_ids(dev_path), read_split_ids(test_path)
    plan = carve(dev_ids, test_ids, seed=seed)
    # The prereg pins the digest of the split *files as written* -- which is what Phase 0
    # recorded and what a reader can reproduce with sha256sum. ``SplitPlan.dev_sha256`` is a
    # different quantity on purpose: it hashes the id sequence in the order carve consumed it,
    # with no trailing newline, and it identifies the carve rather than the file. Both are kept;
    # only the first is the pinned one, and confusing them is a check that fails on a correct
    # file, which is worse than no check because the obvious fix is to delete it.
    for name, expected, path in (
        ("dev", str(pinned["dev_sha256"]), dev_path),
        ("test", str(pinned["test_sha256"]), test_path),
    ):
        measured = sha256_hex(Path(path).read_bytes())
        if measured != expected:
            raise BCPlusCampaignError(
                f"{path} hashes to {measured[:12]}, the frozen prereg pins the {name} split at "
                f"{expected[:12]}. These are different task sets under one name; every recall "
                "and accuracy number computed from this run would be against the wrong one.")

    manifest_path = _split_manifest_path(settings)
    written = write_manifest(plan, manifest_path)
    # Read it back rather than trusting the object in hand: on the second and every later run
    # this is the only path that executes, and the plan a resumed campaign uses must be the one
    # on disk, not the one this process happened to recompute.
    plan = load_manifest(manifest_path)

    task_ids = (_pilot_task_ids(plan, seed) if layer == PILOT_LAYER
                else list(plan.layers[layer]))
    questions = load_questions(queries_path)
    missing = [t for t in task_ids if t not in questions]
    if missing:
        raise BCPlusCampaignError(
            f"{len(missing)} tasks in layer {layer!r} have no question in {queries_path}, e.g. "
            f"{missing[:5]}; the split and the topic file are different vintages")

    # 530 dev queries, 510 layer slots. The 20 that land in no layer are a deliberate reserve,
    # but a reserve nobody can name is indistinguishable from a carve that silently dropped
    # tasks, so they are written down.
    assigned = {q for ids in plan.layers.values() for q in ids}
    unassigned = sorted(q for q in dev_ids if q not in assigned)

    provenance = {
        "layer": layer,
        "split_plan_digest": written["digest"],
        "split_plan_seed": seed,
        "dev_file_sha256": sha256_hex(dev_path.read_bytes()),
        "test_file_sha256": sha256_hex(test_path.read_bytes()),
        "carve_dev_sha256": plan.dev_sha256,
        "carve_test_sha256": plan.test_sha256,
        "split_files_match_prereg": True,
        "unassigned_dev_ids": unassigned,
        "layer_size": len(task_ids),
        "questions_sha256": questions_digest(questions, task_ids),
        "queries_path": str(queries_path),
        "benchdata_root": str(root),
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
    shard: int = 0,
    shards: int = 1,
) -> dict:
    """Run one layer x one arm block, under the same GPU lease the campaign holds."""
    if shards < 1 or not 0 <= shard < shards:
        raise BCPlusCampaignError(
            f"shard {shard} of {shards} is not a partition; a lane that owned nothing would "
            "report a complete run having executed no cells")
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
            shard=shard,
            shards=shards,
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
    shard: int,
    shards: int,
) -> dict:
    from ..doctor import check_installed_graph
    from ..retrieval.client import RetrievalClient

    # Before anything is claimed. A stale installed graph does not crash: vendor's supervisor
    # catches every exception out of the hook block and returns an empty note set, so the cells
    # commit, the reports are written from nothing, and the campaign looks like it ran.
    graph_check = check_installed_graph(Path(repo))
    if graph_check.status != "PASS":
        raise BCPlusCampaignError(
            f"refusing to run: {graph_check.name} is {graph_check.status} -- {graph_check.detail}")

    task_ids, questions, split_provenance = resolve_layer_tasks(
        settings, layer=layer, repo=repo, benchdata=benchdata)
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

    # Three independent copies of the number 5 -- the campaign config, the retrieval freeze, and
    # vendor's own tool default -- and only the last two are compared, mid-cell, by the seam. The
    # freeze is the one the write-up cites, so it is checked here, before anything is claimed.
    configured_top_k = int(settings.get("retrieval", "frozen_corpus", "top_k"))
    frozen_top_k = (identity.get("freeze") or {}).get("top_k")
    if frozen_top_k is not None and int(frozen_top_k) != configured_top_k:
        raise BCPlusCampaignError(
            f"configs/retrieval.yaml asks for top-{configured_top_k} but the retrieval freeze "
            f"records top-{frozen_top_k}; the world served and the world recorded would be "
            "different ones")

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

    # run_id and layer arrive from the command line and are then joined into filesystem paths --
    # the schedule directory, the frozen block directory, the STATUS file. That is a trust
    # boundary, and the grammar that guards it already existed in shapeflow.scoped_paths with no
    # caller. A run_id of "../.." would otherwise write a campaign's schedule outside the data
    # root that the isolation between roles depends on.
    resolved_run_id = safe_scope_component(
        run_id or f"bcplus-{layer}-{arms_block}-{execution_binding_sha256[:12]}", name="run_id")
    safe_scope_component(layer, name="layer")
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

    # Task-atomic sharding across lanes. Every arm and replicate of a task stays on one lane,
    # because a block is paired-valid only under a single engine epoch and the paired contrast is
    # P1 against P0 on the same task -- P0 on one card and P1 on another folds that card's clocks
    # and thermals into the treatment effect with nothing afterwards able to separate them. The
    # assignment is a digest of the frozen binding and the task id, so both lanes compute the
    # same partition without talking to each other.
    if shards > 1:
        owned = frozenset(
            block.block_id for block in manifest.blocks
            if int(sha256_hex(f"{execution_binding_sha256}:{layer}:{block.task_id}".encode()),
                   16) % shards == shard)
        runner.config.shard_id = shard
        runner.config.owned_block_ids = owned
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
        "shard": shard,
        "shards": shards,
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

    # Before the status snapshot: freezing is what records each cell's engine epoch and raises an
    # incident when a block spans two of them. A vLLM restart mid-run leaves every cell looking
    # fine on its own and the paired contrast quietly comparing arms served by different engines.
    frozen_blocks = runner.freeze_blocks(
        manifest, phase_id=phase_id, split=layer,
        directory=settings.path("runs") / "bcplus_blocks" / resolved_run_id)

    states = runner.cell_states(manifest, phase_id=phase_id, split=layer)
    body = runner.status(manifest, phase_id=phase_id, split=layer)
    cells = _committed_cells(runner, manifest, ledger, store, phase_id=phase_id, layer=layer)
    searched = _the_world_was_actually_searched(cells)
    live = _the_treatment_was_actually_live(cells, arms=arms)
    body.update({
        "blocks_frozen": len(frozen_blocks),
        "blocks_valid_for_paired_estimate": sum(
            1 for b in frozen_blocks if b.get("valid_for_paired_estimate", False)),
        "treatment_live": live,
        "layer": layer,
        "arms_block": arms_block,
        "arms": [a.arm_id for a in arms],
        "tasks": len(task_ids),
        "shard": shard,
        "shards": shards,
        "owned_blocks": (len(runner.config.owned_block_ids)
                         if runner.config.owned_block_ids is not None else len(manifest.blocks)),
        "schedule_sha256": manifest.digest,
        "retriever_id": retriever_id,
        "retrieval_cache": _retrieval_cache_stats(world),
        "cell_states": _state_counts(states),
        "world_searched": searched,
        "ok": (all(state == "COMMITTED" for state in states.values())
               and searched["status"] == "PASS" and live["status"] == "PASS"),
        **split_provenance,
    })
    write_status(body, settings.path("runs") / f"STATUS_{resolved_run_id}.json")
    ledger.close()
    return body


#: A committed cell that searched nothing is the signature of a graph whose research arm is
#: inert, not of an agent that thought hard and declined. Vendor's supervisor swallows every
#: exception out of the research block and returns an empty note set, so the failure arrives
#: looking exactly like a cheap successful cell. One or two are plausible; a fifth of the
#: campaign is not.
MAX_SILENT_CELL_FRACTION = 0.20


def _committed_cells(runner, manifest, ledger, store, *, phase_id: str, layer: str) -> list[dict]:
    """Every committed cell's stored record, read once for all the post-run checks."""
    records: list[dict] = []
    for cell in manifest.cells:
        ref = ledger.committed_ref(
            runner.work_key_for(cell, phase_id=phase_id, split=layer))
        if ref is None:
            continue
        body = json.loads(store.get_bytes(ref).decode("utf-8"))
        body["_arm_id"] = cell.arm.arm_id
        records.append(body)
    return records


def _the_treatment_was_actually_live(cells: list[dict], *, arms) -> dict:
    """Refuse to call a run successful if a P1 arm published nothing.

    This is the failure the whole apparatus is built against, and it has happened here before:
    nineteen arms, 146 canary cells, every check green, and not one P1 span published. Under a
    dense corpus it gains a second trigger -- if the page registry never fills, the selector is
    offered candidates whose bytes it cannot read, publishes nothing for each, spends nothing,
    and reports a large work saving. An arm that fell back on every batch has P0's numbers and
    P1's label, and no metric downstream can tell the difference.

    Reported per arm, because a single campaign-wide count is satisfied by one live arm.
    """
    treatment_arms = [a.arm_id for a in arms
                      if not (a.page_variant == "P0" and a.close_variant == "P0")]
    by_arm: dict[str, dict] = {}
    for arm_id in treatment_arms:
        arm_cells = [c for c in cells if c["_arm_id"] == arm_id]
        counts = [c.get("counts") or {} for c in arm_cells]
        published = sum(
            max(0, int(c.get("page_batches_reduced", 0) or 0)
                - int(c.get("page_fallbacks", 0) or 0))
            + int(c.get("close_reduced", 0) or 0)
            for c in counts)
        by_arm[arm_id] = {
            "cells": len(arm_cells),
            "published": published,
            "page_fallbacks": sum(int(c.get("page_fallbacks", 0) or 0) for c in counts),
            "close_failures": sum(int(c.get("close_failed", 0) or 0) for c in counts),
            "pages_registered": sum(int(c.get("pages_registered", 0) or 0) for c in arm_cells),
            "status": "PASS" if (arm_cells and published > 0) else "FAIL",
        }
    inert = sorted(a for a, r in by_arm.items() if r["status"] != "PASS" and r["cells"])
    return {
        "status": "FAIL" if inert else "PASS",
        "treatment_arms": treatment_arms,
        "inert_arms": inert,
        "by_arm": by_arm,
        "detail": ("every treatment arm published at least one P1 output" if not inert
                   else f"{inert} committed cells but published nothing as P1"),
    }


def _the_world_was_actually_searched(cells: list[dict]) -> dict:
    """Refuse to call a run successful if its cells never touched the corpus.

    The measured quantity is deliberately the *search* count and not the note count: a cell can
    legitimately produce a thin note, but a research agent on a 100k-document benchmark that
    issued no query at all did not do the task, whatever it wrote afterwards.
    """
    committed = len(cells)
    silent = [f"{c['cell']['task_id']}:{c['_arm_id']}" for c in cells
              if not int((c.get("counts") or {}).get("search_queries", 0) or 0)]
    fraction = (len(silent) / committed) if committed else 0.0
    status = "PASS" if committed and fraction <= MAX_SILENT_CELL_FRACTION else "FAIL"
    return {
        "status": status,
        "committed": committed,
        "cells_with_no_search": len(silent),
        "fraction": fraction,
        "max_fraction": MAX_SILENT_CELL_FRACTION,
        "examples": sorted(silent)[:10],
        "detail": ("no committed cells to check" if not committed else
                   f"{len(silent)}/{committed} committed cells issued no search query"),
    }


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
