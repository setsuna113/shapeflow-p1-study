"""The campaign runner: phases, blocks, cells, and a restart that fills gaps.

Everything below is arranged around one question -- what happens when this process dies halfway
through -- because it will, and the answers decide whether the results survive it.

**A cell is a ledger work item.** Its key is derived from the protocol SHA and its coordinates,
never from a clock or a counter, so a restart recomputes the same key and asks the ledger whether
that cell is already done. "Done" means COMMITTED *and* the output blob still verifies; anything
weaker would let a commit written before its bytes were flushed be trusted.

**The terminal record is written last.** Output to the object store, then MATERIALIZED, then
VALIDATED, then COMMITTED. A kill between the blob and the commit leaves the cell non-terminal
and it re-runs -- wasteful, and correct.

**Blocks freeze whole or not at all.** A block is frozen only when every cell in it is COMMITTED.
An incomplete block is kept, reported and excluded from the primary comparison, rather than being
completed later with cells that ran under a different engine epoch.

**Fallback is not free.** A P1 that fell back to P0 keeps everything it spent on the ledger; the
run records the fallback and the failure separately from the cell's outcome, because the arm
under test end to end is "P1 with its fallback" and its cost includes the attempt that failed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from ..canonical import canonical_json
from ..experiment.ledger import Ledger
from ..experiment.state_machine import Phase
from ..hashing import sha256_hex
from ..object_store import ObjectStore
from ..strategies.factory import StrategyFactory, load_registry
from .acquire import acquired_task_ids, load_frozen_pool
from .graph_driver import CellSpec, run_cell, summarize_events
from .phases import PhaseStore
from .schedule import ArmSpec, ScheduleManifest, build_blocks, cell_key, freeze_record
from .settings import Settings

__all__ = ["CampaignRunner", "RunnerConfig", "CellOutcome", "STAGE_VERSION"]

#: Bumped when the meaning of a cell's execution changes. It is part of the work key, so a
#: changed stage produces new work items instead of silently reusing results from the old one.
STAGE_VERSION = "screen_v1"


@dataclass
class RunnerConfig:
    run_id: str
    provider_base_url: str
    runner_token: str
    worker_id: str = "w0"
    lease_seconds: float = 1800.0
    max_cells: Optional[int] = None
    stop_sentinel: Optional[Path] = None


@dataclass
class CellOutcome:
    cell_key: str
    state: str
    output_ref: str = ""
    fell_back: bool = False
    error: str = ""
    counts: dict = field(default_factory=dict)


class CampaignRunner:
    """Drives one phase's worth of cells to terminal states, resumably."""

    def __init__(
        self,
        settings: Settings,
        *,
        ledger: Ledger,
        store: ObjectStore,
        config: RunnerConfig,
        model_call: Optional[Callable] = None,
        register_cell: Optional[Callable] = None,
        graph: object = None,
    ) -> None:
        self.settings = settings
        self.ledger = ledger
        self.store = store
        self.config = config
        self.phases = PhaseStore(ledger, protocol_sha=settings.shas["week1"])
        self.registry = load_registry(settings.repo / "configs")
        self._model_call = model_call
        self._register_cell = register_cell
        self._graph = graph
        self.outcomes: list[CellOutcome] = []

    # --- schedule ---------------------------------------------------------------------

    def arms_from_config(self, block_name: str = "canary") -> list[ArmSpec]:
        specs = [
            ArmSpec(arm_id=str(a["arm_id"]), page_variant=str(a["page_variant"]),
                    close_variant=str(a["close_variant"]))
            for a in self.settings.get("week1", block_name, "arms")
        ]
        unknown = sorted(
            v for a in specs for v in (a.page_variant, a.close_variant)
            if v != "P0" and v not in self.registry
        )
        if unknown:
            raise ValueError(
                f"arms name unregistered variants {unknown}; an arm that was not pre-registered "
                "must not run, or the design's balance no longer describes what executed"
            )
        return specs

    def build_schedule(self, *, task_ids: Sequence[str], arms: Sequence[ArmSpec],
                       split: str, second_seed_fraction: float = 0.0) -> ScheduleManifest:
        return build_blocks(
            protocol_sha=self.settings.shas["week1"],
            split=split,
            task_ids=list(task_ids),
            arms=list(arms),
            seeds=[int(s) for s in self.settings.get("week1", "screen", "seeds")],
            layer=str(self.settings.get("week1", "measurement", "layer")),
            claim_scope=self.settings.claim_scope,
            second_seed_fraction=second_seed_fraction,
        )

    def freeze_schedule(self, manifest: ScheduleManifest, path: Path) -> str:
        """Write the schedule before anything runs. Write-once.

        A schedule that could be rewritten mid-campaign would let the plan follow the results,
        which is the definition of not having pre-registered one.
        """
        body = manifest.to_json()
        path = Path(path)
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("schedule_sha256") != body["schedule_sha256"]:
                raise RuntimeError(
                    f"{path} already froze a different schedule "
                    f"({existing.get('schedule_sha256')!r} vs {body['schedule_sha256']!r})"
                )
            return body["schedule_sha256"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return body["schedule_sha256"]

    # --- work items -------------------------------------------------------------------

    def work_key_for(self, cell, *, phase_id: str, split: str) -> str:
        return self.ledger.ensure_work_item(
            protocol_sha=self.settings.shas["week1"],
            split=split,
            phase_id=phase_id,
            task_id=cell.task_id,
            arm_id=cell.arm.arm_id,
            variant_id=f"{cell.arm.page_variant}+{cell.arm.close_variant}",
            replicate_id=cell.replicate_id,
            checkpoint_hash=cell.block_id,
            stage_version=STAGE_VERSION,
            # A cell dispatches model calls, so a lease that expires mid-flight must not be
            # blindly re-run: the ledger freezes a side-effecting item at FAILED_UNKNOWN.
            side_effecting=True,
        )

    def cell_states(self, manifest: ScheduleManifest, *, phase_id: str, split: str) -> dict:
        """Terminal state per cell, with the artifact re-verified for committed ones."""
        states: dict = {}
        for cell in manifest.cells:
            key = self.work_key_for(cell, phase_id=phase_id, split=split)
            item = self.ledger.get_work_item(key)
            state = item.state if item else "PENDING"
            if state == "COMMITTED":
                ref = self.ledger.committed_ref(key)
                if not (ref and self.store.verify(ref)):
                    # The database says done and the bytes disagree. Re-running is the only
                    # honest option; trusting the flag would carry a hole into the analysis.
                    state = "PENDING"
            states[cell_key(cell)] = state
        return states

    # --- execution ---------------------------------------------------------------------

    def _bundle_for(self, arm: ArmSpec):
        from ..odr.hooks import StrategyBundle
        from ..strategies.p0 import VendorCloseStrategy, VendorPageStrategy

        if arm.page_variant == "P0" and arm.close_variant == "P0":
            return StrategyBundle(variant_id="P0", page=VendorPageStrategy({}),
                                  close=VendorCloseStrategy())
        factory = StrategyFactory(
            registry=self.registry,
            model_call=self._model_call,
            token_budget=int(self.settings.get("week1", "measurement",
                                               "selected_token_budget")),
        )
        page = factory.build(arm.page_variant).page if arm.page_variant != "P0" \
            else VendorPageStrategy({})
        close = factory.build(arm.close_variant).close if arm.close_variant != "P0" \
            else VendorCloseStrategy()
        return StrategyBundle(variant_id=f"{arm.page_variant}+{arm.close_variant}",
                              page=page, close=close)

    def stop_requested(self) -> bool:
        sentinel = self.config.stop_sentinel
        return bool(sentinel and Path(sentinel).exists())

    async def run_cells(
        self,
        manifest: ScheduleManifest,
        *,
        phase_id: str,
        split: str,
        questions: dict,
    ) -> list[CellOutcome]:
        """Execute every cell that is not already COMMITTED, in the frozen block order."""
        states = self.cell_states(manifest, phase_id=phase_id, split=split)
        executed = 0

        for block in manifest.blocks:
            for cell in sorted(block.cells, key=lambda c: c.order_index):
                key = cell_key(cell)
                if states.get(key) == "COMMITTED":
                    continue
                if self.stop_requested():
                    return self.outcomes
                if self.config.max_cells is not None and executed >= self.config.max_cells:
                    return self.outcomes
                outcome = await self._execute_cell(cell, phase_id=phase_id, split=split,
                                                   question=questions[cell.task_id])
                self.outcomes.append(outcome)
                states[key] = outcome.state
                executed += 1
        return self.outcomes

    async def _execute_cell(self, cell, *, phase_id: str, split: str, question: str) -> CellOutcome:
        key = cell_key(cell)
        work_key = self.work_key_for(cell, phase_id=phase_id, split=split)
        item = self.ledger.get_work_item(work_key)
        if item is None or item.state != "PENDING":
            return CellOutcome(cell_key=key, state=item.state if item else "MISSING")

        attempt = self.ledger.claim(work_key, self.config.worker_id,
                                    lease_seconds=self.config.lease_seconds,
                                    run_id=self.config.run_id)
        if attempt is None:
            return CellOutcome(cell_key=key, state="CLAIMED_ELSEWHERE")

        token = "cell-" + sha256_hex(canonical_json({
            "run": self.config.run_id, "block": cell.block_id, "arm": cell.arm.arm_id,
            "replicate": cell.replicate_id, "attempt": attempt.attempt_ordinal,
        }))[:24]
        spec = CellSpec(
            run_id=self.config.run_id, task_id=cell.task_id, arm_id=cell.arm.arm_id,
            page_variant=cell.arm.page_variant, close_variant=cell.arm.close_variant,
            replicate_id=cell.replicate_id, seed=cell.seed, work_key=work_key,
            question=question, cell_token=token,
        )

        try:
            if self._register_cell is not None:
                await self._register_cell(spec)
            pool, snapshots = load_frozen_pool(self.settings, cell.task_id)
            result = await run_cell(
                self.settings, spec, pool=pool, snapshots=snapshots,
                bundle=self._bundle_for(cell.arm),
                provider_base_url=self.config.provider_base_url,
                runner_token=self.config.runner_token,
                store_checkpoint=self._store_checkpoint,
                graph=self._graph,
            )
        except Exception as e:  # noqa: BLE001 - a cell that died may have already spent tokens
            self.ledger.fail(attempt.attempt_id, disposition="FAILED_UNKNOWN",
                             error_class=type(e).__name__, reason=str(e)[:400])
            return CellOutcome(cell_key=key, state="FAILED_UNKNOWN", error=f"{type(e).__name__}")

        counts = summarize_events(result.events)
        payload = {
            "cell": cell.content(),
            "run_id": self.config.run_id,
            "variant_id": spec.variant_id,
            "final_report": result.final_report,
            "notes": list(result.notes),
            "raw_notes": list(result.raw_notes),
            "events": result.events,
            "checkpoints": result.checkpoints,
            "counts": counts,
            "error": result.error,
            "claim_scope": self.settings.claim_scope,
        }
        raw = canonical_json(payload)
        ref = self.store.put_bytes(raw)
        self.ledger.register_artifact(ref.key, kind="cell_output", raw_size=ref.raw_size,
                                      stored_size=ref.stored_size, work_key=work_key)
        self.ledger.advance(attempt.attempt_id, "MATERIALIZED")

        if result.error:
            # The graph itself failed. The work it spent is already on the provider's ledger;
            # the cell is a final failure and its block will not freeze.
            self.ledger.fail(attempt.attempt_id, disposition="FAILED_FINAL",
                             error_class="graph_error", reason=result.error[:400])
            return CellOutcome(cell_key=key, state="FAILED_FINAL", output_ref=ref.key,
                               error=result.error, counts=counts)

        self.ledger.advance(attempt.attempt_id, "VALIDATED")
        self.ledger.commit(attempt.attempt_id, result_object_ref=ref.key)
        return CellOutcome(cell_key=key, state="COMMITTED", output_ref=ref.key,
                           fell_back=bool(counts.get("page_fallbacks")), counts=counts)

    def _store_checkpoint(self, checkpoint) -> str:
        """Content-address every boundary checkpoint the run produced.

        Kept for every arm, not just P1: the forked-state component analysis needs each arm's
        boundary states, and a checkpoint that only exists for the arm that happened to use it
        cannot anchor a comparison.
        """
        body = canonical_json({"kind": type(checkpoint).__name__,
                               "digest": checkpoint.digest})
        ref = self.store.put_bytes(body)
        self.ledger.register_artifact(ref.key, kind="checkpoint", raw_size=ref.raw_size,
                                      stored_size=ref.stored_size)
        return checkpoint.digest

    # --- freezing ---------------------------------------------------------------------

    def freeze_blocks(self, manifest: ScheduleManifest, *, phase_id: str, split: str,
                      directory: Path) -> list[dict]:
        """Freeze every block whose cells are all COMMITTED. Incomplete blocks are left alone."""
        states = self.cell_states(manifest, phase_id=phase_id, split=split)
        outputs = {
            cell_key(cell): (self.ledger.committed_ref(
                self.work_key_for(cell, phase_id=phase_id, split=split)) or "")
            for cell in manifest.cells
        }
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        frozen: list[dict] = []
        for block in manifest.blocks:
            record = freeze_record(block, states=states, outputs=outputs)
            if not record["complete"]:
                continue
            path = directory / f"{block.block_id}.json"
            if not path.exists():
                path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8")
            frozen.append(record)
        return frozen

    def status(self, manifest: Optional[ScheduleManifest] = None, *, phase_id: str = "",
               split: str = "") -> dict:
        """A snapshot fit for reports/STATUS.json."""
        counts = self.ledger.state_counts()
        body = {
            "run_id": self.config.run_id,
            "phase": self.phases.current().value,
            "work_items": counts,
            "claim_scope": self.settings.claim_scope,
            "corpus_tier": self.settings.corpus_tier,
        }
        if manifest is not None:
            states = self.cell_states(manifest, phase_id=phase_id, split=split)
            body["cells_total"] = len(manifest.cells)
            body["cells_committed"] = sum(1 for s in states.values() if s == "COMMITTED")
            body["blocks_total"] = len(manifest.blocks)
            body["blocks_complete"] = sum(
                1 for b in manifest.blocks
                if all(states.get(cell_key(c)) == "COMMITTED" for c in b.cells)
            )
        return body


def write_status(body: dict, path: Path) -> None:
    """Write STATUS.json atomically, so a reader never sees half a document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def questions_for(settings: Settings, task_ids: Sequence[str]) -> dict:
    """Read each task's question from the runner-readable view.

    The runner's copy carries the question and nothing else, so this cannot accidentally hand a
    selector the authored facets even if someone later passes the wrong dictionary along.
    """
    directory = settings.path("frozen_corpus_for_runner") / "tasks"
    questions: dict = {}
    for task_id in task_ids:
        body = json.loads((directory / f"{task_id}.json").read_text(encoding="utf-8"))
        questions[task_id] = body["original_question"]
    return questions


def available_tasks(settings: Settings, split: str) -> list[str]:
    """Tasks in ``split`` whose world is frozen and readable by the runner."""
    directory = settings.path("frozen_corpus_for_runner") / "tasks"
    frozen = set(acquired_task_ids(settings))
    ids: list[str] = []
    for path in sorted(directory.glob("*.json")):
        body = json.loads(path.read_text(encoding="utf-8"))
        if body.get("split") == split and body["task_id"] in frozen:
            ids.append(body["task_id"])
    return ids


def phase_for(name: str) -> Phase:
    return Phase(name)
