"""Independent full-graph cells plus durable artifacts and work telemetry.

Everything below is arranged around one question -- what happens when this process dies halfway
through -- because it will, and the answers decide whether the results survive it.

**A cell is a ledger work item.** Its key is derived from the complete approved execution
binding and its coordinates, never from a clock or a counter, so a restart recomputes the same
key and asks the ledger whether that cell is already done. "Done" means COMMITTED *and* the
output blob still verifies; anything weaker would let a commit written before its bytes were
flushed be trusted.

**The terminal record is written last.** Output to the object store, then MATERIALIZED, then
VALIDATED, then COMMITTED. A kill between the blob and the commit leaves the cell non-terminal
and it re-runs -- wasteful, and correct.

This runner starts every cell from the task input. It is therefore the executor for the primary
coupled-seed full-run E2E ITT and for engineering diagnostics. It is **not** the component-only
executor: that separate mode requires a boundary backend and never calls ``run_cells``. Calling
task-level cells a same-checkpoint fork was the confounding bug this split prevents.

**Fallback is not free.** A P1 that fell back to P0 keeps everything it spent on the ledger; the
run records the fallback and the failure separately from the cell's outcome, because the arm
under test end to end is "P1 with its fallback" and its cost includes the attempt that failed.
"""

from __future__ import annotations

import inspect
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from ..canonical import canonical_json
from ..evidence.model_tokenizer import load_frozen_tokenizer
from ..evidence.shared_view import apply_shared_budget
from ..experiment.ledger import TERMINAL_STATES, Ledger
from ..hashing import sha256_hex
from ..object_store import ObjectStore
from ..strategies.factory import StrategyFactory, load_registry
from ..world.pools import acquired_task_ids, load_frozen_pool
from .graph_driver import CellSpec, run_cell, summarize_events
from .phases import PhaseStore
from .schedule import (
    FROZEN_ROOT_FILENAME,
    ArmSpec,
    ScheduleManifest,
    block_is_complete,
    block_is_terminal,
    build_blocks,
    cell_key,
    freeze_record,
    freeze_root_record,
)
from .settings import Settings

__all__ = ["CampaignRunner", "RunnerConfig", "CellOutcome", "STAGE_VERSION"]

#: Bumped when the meaning of a cell's execution changes. It is part of the work key, so a
#: changed stage produces new work items instead of silently reusing results from the old one.
STAGE_VERSION = "independent_full_run_v3"


@dataclass
class RunnerConfig:
    run_id: str
    provider_base_url: str
    runner_token: str
    worker_id: str = "w0"
    lease_seconds: float = 1800.0
    max_cells: Optional[int] = None
    stop_sentinel: Optional[Path] = None
    #: This lane's identity and its frozen share of the schedule, when the campaign is sharded
    #: across GPUs. None means one lane owning everything, which is the unsharded campaign.
    shard_id: Optional[int] = None
    owned_block_ids: Optional[frozenset[str]] = None


@dataclass
class CellOutcome:
    cell_key: str
    state: str
    output_ref: str = ""
    fell_back: bool = False
    error: str = ""
    counts: dict = field(default_factory=dict)


def _vendor_today_str() -> str:
    """Vendor's own ``get_today_str``, so the recorded date is the one the prompts rendered.

    Read through the vendor module rather than reimplemented: a second date formatter would
    drift from the string actually interpolated into the compressor and final-report prompts,
    which is the whole thing this value exists to detect.
    """
    try:
        from open_deep_research.utils import get_today_str
    except ImportError:  # pragma: no cover - vendor tree absent in pure-unit environments
        return ""
    return str(get_today_str())


def _cell_energy_counter():
    """An energy counter for the GPU this runner leased, or an inert one off the run host.

    The UUID comes from the environment the unit sets, never from probing the machine: a lane
    reading "whichever GPU is visible" would attribute one lane's joules to another's cells the
    moment four lanes share a host.
    """
    from ..runtime.nvml_sampler import EnergyCounter

    uuid = (
        os.environ.get("SHAPEFLOW_GPU_UUID")
        or os.environ.get("CUDA_VISIBLE_DEVICES")
        or ""
    ).strip()
    return EnergyCounter(gpu_uuid=uuid) if uuid.startswith("GPU-") else EnergyCounter(gpu_uuid="")


def _engine_epoch(settings: Settings) -> str:
    """Read the identity of the concrete vLLM boot serving cells.

    The stack manifest identifies bytes/flags and remains constant across a service restart, so
    it cannot be an epoch.  The engine unit writes its per-invocation id atomically before
    serving.  Missing/malformed provenance stops before a cell is claimed.
    """
    del settings
    path_text = os.environ.get("SHAPEFLOW_ENGINE_EPOCH_FILE", "").strip()
    if not path_text:
        raise RuntimeError(
            "SHAPEFLOW_ENGINE_EPOCH_FILE is not set; a stack hash is not a vLLM boot epoch")
    path = Path(path_text)
    try:
        value = path.read_text(encoding="ascii").strip().lower()
    except OSError as exc:
        raise RuntimeError(f"cannot read vLLM engine epoch {path}: {exc}") from exc
    if len(value) != 32 or any(c not in "0123456789abcdef" for c in value):
        raise RuntimeError(f"vLLM engine epoch {path} is not a 128-bit invocation id")
    return value


def _write_once_content_addressed_json(
    path: Path, body: dict, *, digest_field: str, label: str
) -> None:
    """Create without replacement, or verify the identical existing artifact.

    ``exists()`` followed by ``write_text()`` has a truncation race when two resumptions reach
    freeze concurrently. Exclusive creation makes write-once an enforced property rather than a
    convention.
    """
    path = Path(path)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(body, indent=2, sort_keys=True) + "\n")
        return
    except FileExistsError:
        pass
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} {path} is unreadable: {exc}") from exc
    recorded = str(existing.get(digest_field) or "")
    actual = sha256_hex(canonical_json({
        key: value for key, value in existing.items() if key != digest_field
    }))
    if recorded != actual:
        raise RuntimeError(
            f"{label} {path} was edited: records {recorded!r}, hashes to {actual}")
    if recorded != str(body.get(digest_field) or ""):
        raise RuntimeError(f"{path} already contains a different {label}; it is write-once")


class CampaignRunner:
    """Drives one phase's worth of cells to terminal states, resumably."""

    def __init__(
        self,
        settings: Settings,
        *,
        ledger: Ledger,
        store: ObjectStore,
        config: RunnerConfig,
        execution_binding_sha256: str,
        protocol_document_sha256: str,
        model_call_factory: Optional[Callable] = None,
        register_cell: Optional[Callable] = None,
        fetch_work_summary: Optional[Callable] = None,
        graph: object = None,
        engine_epoch: Optional[str] = None,
        world_backend: object = None,
    ) -> None:
        self.settings = settings
        self.ledger = ledger
        self.store = store
        self.config = config
        for label, value in (
            ("execution_binding_sha256", execution_binding_sha256),
            ("protocol_document_sha256", protocol_document_sha256),
        ):
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")
        self.execution_binding_sha256 = execution_binding_sha256
        self.protocol_document_sha256 = protocol_document_sha256
        self.phases = PhaseStore(
            ledger, protocol_sha=self.execution_binding_sha256)
        self.registry = load_registry(settings.repo / "configs")
        # This is part of the treatment, not a reporting convenience: it defines chunk
        # boundaries, renderer budgets and the CPU/prose controls. Production refuses to use
        # the whitespace test double.
        self.tokenizer = load_frozen_tokenizer(settings)
        self._model_call_factory = model_call_factory
        self._register_cell = register_cell
        self._fetch_work_summary = fetch_work_summary
        self._graph = graph
        # One world for the whole lane, or None for the Week-1 per-task frozen pool. A
        # 100k-document dense corpus is not task-local and cannot be: it is loaded once, behind a
        # socket, and every cell searches the same one. Which of the two is in force is decided
        # here, once, rather than per cell -- a runner that could take either world per cell is a
        # runner whose corpus is not a property of the run.
        self._world_backend = world_backend
        # Tests/in-process harnesses may inject an explicit epoch. Production always reads the
        # engine unit's per-boot file, and re-reads it around every cell.
        self._engine_epoch_override = engine_epoch
        self.engine_epoch = engine_epoch or _engine_epoch(settings)
        self.outcomes: list[CellOutcome] = []

    async def _durable_work_summary(self, work_key: str) -> dict:
        """Fetch provider-owned timing/usage; never derive causal work from local wall time."""
        if self._fetch_work_summary is None:
            return {
                "telemetry_complete": False,
                "overlap_valid": False,
                "unavailable_reason": "NO_PROVIDER_WORK_SUMMARY_CALLBACK",
                "work_key": work_key,
            }
        try:
            value = self._fetch_work_summary(work_key)
            if inspect.isawaitable(value):
                value = await value
            if not isinstance(value, dict):
                raise TypeError("provider work summary is not an object")
            return dict(value)
        except Exception as exc:  # noqa: BLE001 - preserve graph failure plus telemetry failure
            return {
                "telemetry_complete": False,
                "overlap_valid": False,
                "unavailable_reason": f"{type(exc).__name__}: {exc}",
                "work_key": work_key,
            }

    # --- schedule ---------------------------------------------------------------------

    def arms_from_config(self, block_name: str = "canary") -> list[ArmSpec]:
        return self.arms_from_config_static(self.settings, block_name, registry=self.registry)

    @staticmethod
    def arms_from_config_static(
        settings: Settings, block_name: str = "canary", *, registry=None
    ) -> list[ArmSpec]:
        """Resolve an arm block without a runner.

        The steward needs the same arm set to derive the schedule that the shard partition is
        frozen against, and deriving it a second way is how two "identical" schedules come to
        have different digests.
        """
        if registry is None:
            registry = load_registry(settings.repo / "configs")
        specs = [
            ArmSpec(arm_id=str(a["arm_id"]), page_variant=str(a["page_variant"]),
                    close_variant=str(a["close_variant"]))
            for a in settings.get("week1", block_name, "arms")
        ]
        unknown = sorted(
            v for a in specs for v in (a.page_variant, a.close_variant)
            if v != "P0" and v not in registry
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
            execution_binding_sha256=self.execution_binding_sha256,
            protocol_sha=self.protocol_document_sha256,
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
        self._assert_manifest_identity(manifest)
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

    def _assert_manifest_identity(self, manifest: ScheduleManifest) -> None:
        if manifest.execution_binding_sha256 != self.execution_binding_sha256:
            raise ValueError(
                "schedule execution binding does not match this CampaignRunner")
        if manifest.protocol_sha != self.protocol_document_sha256:
            raise ValueError(
                "schedule protocol-document SHA does not match this CampaignRunner")

    # --- work items -------------------------------------------------------------------

    def _work_key_for(self, cell, *, phase_id: str, split: str) -> str:
        return self.ledger.ensure_work_item(
            protocol_sha=self.execution_binding_sha256,
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

    def work_key_for(self, cell, *, phase_id: str, split: str) -> str:
        return self._work_key_for(cell, phase_id=phase_id, split=split)

    def _current_engine_epoch(self) -> str:
        return self._engine_epoch_override or _engine_epoch(self.settings)

    def _execution_epoch(
        self, work_key: str, start_epoch: str
    ) -> tuple[str, str, bool]:
        """Freeze the actual boot interval after a cell finishes or fails.

        Reading only at coordinator construction misses a vLLM restart while the coordinator
        stays alive.  Reading only at the end can attribute old-engine calls to the new boot.
        A transition is encoded as its own epoch value, so the block validity gate cannot
        mistake it for a stable cell.
        """
        try:
            end_epoch = self._current_engine_epoch()
        except RuntimeError:
            end_epoch = "UNREADABLE"
        stable = end_epoch == start_epoch
        recorded = start_epoch if stable else f"SPANS:{start_epoch}:{end_epoch}"
        self.ledger.record_engine_epoch(work_key, recorded)
        return recorded, end_epoch, stable

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

    def shared_content_budget(self):
        """How much of a page either arm may see, derived from the frozen engine window.

        Derived rather than configured: the bound only stays correct if it moves when the
        window or the completion cap moves. A separately-chosen number is how a legal character
        count became an illegal token count and vLLM refused the request outright.
        """
        from ..evidence.shared_view import SharedContentBudget

        return SharedContentBudget.derive(
            max_chars=int(self.settings.get("week1", "odr", "max_content_length")),
            max_model_len=int(self.settings.get("stack", "engine", "max_model_len")),
            completion_cap=int(
                self.settings.get("week1", "odr", "summarization_model_max_tokens")),
            prompt_overhead_tokens=int(
                self.settings.get("week1", "odr", "summarization_prompt_overhead_tokens")),
        )

    def _page_bytes(self, pool, snapshots) -> tuple[dict, dict]:
        """Resolve the exact bytes vendor would have summarised, keyed as the checkpoint keys them.

        The H checkpoint carries only ``raw_content_id`` -- the SHA-256 of
        ``result['raw_content'][:max_content_length]`` -- because the batch is captured before
        anything is transformed. The selector needs the bytes themselves, and they must be
        *vendor's truncation of them*, not the whole page: an arm that read the full page while
        P0 read a prefix would be measuring page length rather than selection, and AGENTS.md
        bars it from the primary contrast.
        """
        budget = self.shared_content_budget()
        text_by_id: dict[str, str] = {}
        occurrence_by_id: dict[str, str] = {}
        for occurrence in pool.vendor_visible:
            if not occurrence.content_hash:
                continue
            snapshot = pool.snapshots.get(occurrence.content_hash)
            if snapshot is None:
                continue
            truncated = apply_shared_budget(
                snapshots.read_text(snapshot), budget, self.tokenizer).text
            key = sha256_hex(truncated.encode("utf-8"))
            text_by_id.setdefault(key, truncated)
            occurrence_by_id.setdefault(key, occurrence.occurrence_id)
        return text_by_id, occurrence_by_id

    def _model_call_for(self, cell_token: str, seed: int):
        """Construct a per-cell selector call, passing the real replicate seed when supported."""
        if self._model_call_factory is None:
            raise ValueError("no model call factory was provided")
        try:
            signature = inspect.signature(self._model_call_factory)
            accepts_seed = "seed" in signature.parameters or any(
                p.kind is inspect.Parameter.VAR_KEYWORD
                for p in signature.parameters.values()
            )
        except (TypeError, ValueError):
            accepts_seed = False
        if accepts_seed:
            return self._model_call_factory(cell_token, seed=seed)
        return self._model_call_factory(cell_token)

    def _bundle_for(self, arm: ArmSpec, cell_token: str, *, seed: int,
                    pool=None, snapshots=None, pages=None):
        """Build the arm's strategies, with a selector bound to *this* cell.

        The model call has to be per-cell: a selector wired to a shared, untagged endpoint would
        spend its tokens outside any cell, and the work it cost could not be attributed to the
        arm that spent it.
        """
        from ..odr.hooks import StrategyBundle
        from ..strategies.p0 import VendorCloseStrategy, VendorPageStrategy

        if arm.page_variant == "P0" and arm.close_variant == "P0":
            return StrategyBundle(variant_id="P0", page=VendorPageStrategy({}),
                                  close=VendorCloseStrategy())
        if self._model_call_factory is None:
            raise ValueError(
                f"arm {arm.arm_id} needs a selector but no model call factory was provided")
        if pages is None:
            from .pages import PageRegistry

            pages = PageRegistry()
        if pool is not None and snapshots is not None:
            # An enumerable world can be resolved before the cell starts, and is: the pool holds
            # every page the task could ever see. A query-reached world cannot be, so its pages
            # arrive through the seam as they are served. Both land in the same registry, so the
            # selector's lookup does not know which world it is reading.
            pages.prefill(*self._page_bytes(pool, snapshots))
        factory = StrategyFactory(
            registry=self.registry,
            model_call=self._model_call_for(cell_token, seed),
            tokenizer=self.tokenizer,
            token_budget=int(self.settings.get("week1", "measurement",
                                               "selected_token_budget")),
            # Without these the page selector is offered no candidates at all: it would publish
            # empty content, spend nothing, and look like a working P1 arm that happens to save
            # everything. That is exactly the inert-P1 failure the canary exists to catch.
            raw_text_for=pages.text_for,
            occurrence_for=pages.occurrence_for,
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
        self._assert_manifest_identity(manifest)
        states = self.cell_states(manifest, phase_id=phase_id, split=split)
        executed = 0

        for block in manifest.blocks:
            if not self._owns(block):
                # Another lane's task. Not skipped as "already done" -- never claimed at all, so
                # its work items stay PENDING in this lane's ledger and the merge can tell the
                # difference between a block another lane ran and a block nobody ran.
                continue
            for cell in sorted(block.cells, key=lambda c: c.order_index):
                key = cell_key(cell)
                if states.get(key) in TERMINAL_STATES:
                    # ITT keeps terminal failures as outcomes; resume must never re-run them
                    # until one happens to succeed.
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
        # Resolve before claim: missing boot provenance must not mutate the assignment or spend.
        start_epoch = self._current_engine_epoch()
        item = self.ledger.get_work_item(work_key)
        if item is None or item.state != "PENDING":
            return CellOutcome(cell_key=key, state=item.state if item else "MISSING")

        attempt = self.ledger.claim(work_key, self.config.worker_id,
                                    lease_seconds=self.config.lease_seconds,
                                    run_id=self.config.run_id)
        if attempt is None:
            return CellOutcome(cell_key=key, state="CLAIMED_ELSEWHERE")
        cell_started = time.monotonic()
        # The GPU's own energy counter, read across the cell. A work saving that only moved cost
        # onto the power draw is not a saving, and that is invisible in tokens and intervals.
        # An unavailable counter yields None -- reported as not measured, never as zero.
        energy = _cell_energy_counter()
        energy.start()

        token = "cell-" + sha256_hex(canonical_json({
            "run": self.config.run_id, "block": cell.block_id, "arm": cell.arm.arm_id,
            "replicate": cell.replicate_id, "attempt": attempt.attempt_ordinal,
        }))[:24]
        spec = CellSpec(
            run_id=self.config.run_id, task_id=cell.task_id, arm_id=cell.arm.arm_id,
            page_variant=cell.arm.page_variant, close_variant=cell.arm.close_variant,
            replicate_id=cell.replicate_id, seed=cell.seed, work_key=work_key,
            question=question, cell_token=token,
            execution_binding_sha256=self.execution_binding_sha256,
            protocol_document_sha256=self.protocol_document_sha256,
        )

        # Where this cell's retrieval starts in the world's own trace. Evidence recall is a
        # docid-set intersection, and a docid is exactly what the treatment-facing SearchRecord
        # deliberately does not carry -- it is shaped like a vendor search result and those have
        # none. So the docids are read off the world, not off the arm's view of it, and sliced by
        # position: cells run one at a time in a lane, so the slice is the cell's own retrieval
        # and nothing else's.
        trace = getattr(self._world_backend, "trace", None)
        trace_start = len(trace) if isinstance(trace, list) else 0
        from .pages import PageRegistry

        pages = PageRegistry()

        try:
            if self._register_cell is not None:
                await self._register_cell(spec)
            pool, snapshots = (None, None)
            if self._world_backend is None:
                pool, snapshots = load_frozen_pool(self.settings, cell.task_id)
            result = await run_cell(
                self.settings, spec, pool=pool, snapshots=snapshots,
                backend=self._world_backend, pages=pages,
                bundle=self._bundle_for(cell.arm, token, seed=cell.seed, pool=pool,
                                        snapshots=snapshots, pages=pages),
                provider_base_url=self.config.provider_base_url,
                runner_token=self.config.runner_token,
                store_checkpoint=self._store_checkpoint,
                store_continuation=self._store_continuation,
                graph=self._graph,
                content_budget=self.shared_content_budget(),
                tokenizer=self.tokenizer,
            )
        except Exception as e:  # noqa: BLE001 - a cell that died may have already spent tokens
            e2e_latency_seconds = max(0.0, time.monotonic() - cell_started)
            recorded_epoch, end_epoch, epoch_stable = self._execution_epoch(
                work_key, start_epoch)
            energy.stop()
            work_summary = await self._durable_work_summary(work_key)
            work_summary["e2e_latency_seconds"] = e2e_latency_seconds
            work_summary["energy_joules"] = energy.joules()
            failure_payload = canonical_json({
                "cell": cell.content(),
                "run_id": self.config.run_id,
                "phase_id": phase_id,
                "work_key": work_key,
                "execution_binding_sha256": self.execution_binding_sha256,
                "protocol_document_sha256": self.protocol_document_sha256,
                "variant_id": spec.variant_id,
                "error": f"{type(e).__name__}: {e}",
                "work_summary": work_summary,
                "e2e_latency_seconds": e2e_latency_seconds,
                "engine_epoch_start": start_epoch,
                "engine_epoch_end": end_epoch,
                "engine_epoch_recorded": recorded_epoch,
                "engine_epoch_stable": epoch_stable,
                "terminal_state": "FAILED_UNKNOWN",
                "claim_scope": self.settings.claim_scope,
                "retrieval_trace": (
                    list(trace[trace_start:]) if isinstance(trace, list) else []),
                "pages_registered": len(pages),
            })
            failure_ref = self.store.put_bytes(failure_payload)
            self.ledger.register_artifact(
                failure_ref.key, kind="cell_failure_output",
                raw_size=failure_ref.raw_size, stored_size=failure_ref.stored_size,
                work_key=work_key,
            )
            self.ledger.fail(attempt.attempt_id, disposition="FAILED_UNKNOWN",
                             error_class=type(e).__name__, reason=str(e)[:400],
                             result_object_ref=failure_ref.key)
            return CellOutcome(
                cell_key=key, state="FAILED_UNKNOWN", output_ref=failure_ref.key,
                error=f"{type(e).__name__}",
            )

        e2e_latency_seconds = max(0.0, time.monotonic() - cell_started)
        recorded_epoch, end_epoch, epoch_stable = self._execution_epoch(
            work_key, start_epoch)
        energy.stop()
        counts = summarize_events(result.events)
        work_summary = await self._durable_work_summary(work_key)
        work_summary["e2e_latency_seconds"] = e2e_latency_seconds
        work_summary["energy_joules"] = energy.joules()
        payload = {
            "cell": cell.content(),
            "run_id": self.config.run_id,
            "phase_id": phase_id,
            "execution_binding_sha256": self.execution_binding_sha256,
            "protocol_document_sha256": self.protocol_document_sha256,
            # The provider records its calls against this key, so it is what joins a cell's
            # output to the tokens it actually spent. Without it the two ledgers cannot be
            # reconciled and every token-based check would compare against nothing.
            "work_key": work_key,
            "variant_id": spec.variant_id,
            "final_report": result.final_report,
            "notes": list(result.notes),
            "raw_notes": list(result.raw_notes),
            "events": result.events,
            "checkpoints": result.checkpoints,
            "direct_node_records": result.direct_node_records,
            "trajectory_summary": result.trajectory_summary,
            "first_treatment_checkpoint_digest":
                result.first_treatment_checkpoint_digest,
            "seed_requested": cell.seed,
            "seed_applied_to_odr": result.seed_applied,
            "work_summary": work_summary,
            "e2e_latency_seconds": e2e_latency_seconds,
            "engine_epoch_start": start_epoch,
            "engine_epoch_end": end_epoch,
            "engine_epoch_recorded": recorded_epoch,
            "engine_epoch_stable": epoch_stable,
            "counts": counts,
            "error": result.error,
            "claim_scope": self.settings.claim_scope,
            "retrieval_trace": list(trace[trace_start:]) if isinstance(trace, list) else [],
            "pages_registered": len(pages),
        }
        raw = canonical_json(payload)
        ref = self.store.put_bytes(raw)
        self.ledger.register_artifact(ref.key, kind="cell_output", raw_size=ref.raw_size,
                                      stored_size=ref.stored_size, work_key=work_key)
        self.ledger.advance(attempt.attempt_id, "MATERIALIZED")

        telemetry_invalid = (
            not epoch_stable
            or
            not bool(work_summary.get("telemetry_complete"))
            or not bool(work_summary.get("by_op"))
            or (
                # Overlapping intervals invalidate the summed-service metric and nothing else,
                # so they are only a cell failure in the layer that reports it.
                self.settings.layer_is_serialized
                and not bool(work_summary.get("overlap_valid"))
            )
        )
        if result.error or telemetry_invalid:
            # The graph itself failed. The work it spent is already on the provider's ledger;
            # the cell is a final failure and its block will not freeze.
            reason = result.error or (
                "vLLM engine restarted while the cell was running"
                if not epoch_stable else
                "provider work telemetry incomplete or overlapping: "
                + str(work_summary.get("unavailable_reason")
                      or work_summary.get("overlap_error") or "unknown")
            )
            self.ledger.fail(attempt.attempt_id, disposition="FAILED_FINAL",
                             error_class="graph_error" if result.error
                             else "invalid_work_telemetry",
                             reason=reason[:400], result_object_ref=ref.key)
            return CellOutcome(cell_key=key, state="FAILED_FINAL", output_ref=ref.key,
                               error=reason, counts=counts)

        self.ledger.advance(attempt.attempt_id, "VALIDATED")
        self.ledger.commit(attempt.attempt_id, result_object_ref=ref.key)
        return CellOutcome(cell_key=key, state="COMMITTED", output_ref=ref.key,
                           fell_back=bool(counts.get("page_fallbacks")), counts=counts)

    def _store_checkpoint(self, checkpoint) -> str:
        """Persist every boundary checkpoint the run produced, in full.

        Kept for every arm, not just P1: the forked-state component analysis needs each arm's
        boundary states, and a checkpoint that only exists for the arm that happened to use it
        cannot anchor a comparison.

        In full, because this used to write `{"kind": ..., "digest": ...}` -- two fields, no
        state, and no reader anywhere -- so nothing could fork from a boundary and the
        component trial re-ran the entire graph per arm instead.
        """
        from ..odr.checkpoints import CheckpointStore, to_document

        digest = CheckpointStore(self.settings.path("checkpoints")).put(checkpoint)
        ref = self.store.put_bytes(canonical_json(to_document(checkpoint)))
        self.ledger.register_artifact(ref.key, kind="checkpoint", raw_size=ref.raw_size,
                                      stored_size=ref.stored_size)
        return digest

    def _store_continuation(self, continuation: dict) -> str:
        """Persist the root/supervisor state a C fork writes its report into.

        Stamped here rather than in the driver because the two fields that make a pair valid
        are the runner's to know: which engine boot produced the shared upstream work, and
        which UTC date the prompts were rendered under. ``get_today_str`` is ``datetime.now()``
        and is formatted into both the compressor and the final-report prompt, so a boundary
        whose arms straddle midnight differs by more than its treatment.
        """
        from ..odr.continuation import ContinuationEnvelope, ContinuationStore

        envelope = ContinuationEnvelope(
            task_id=str(continuation["task_id"]),
            seed=int(continuation["seed"]),
            research_brief=str(continuation["research_brief"]),
            root_messages=tuple(continuation["root_messages"]),
            notes=tuple(continuation["notes"]),
            anchor_today_str=_vendor_today_str(),
            anchor_engine_epoch=self._current_engine_epoch(),
            anchor_run_ref=self.config.run_id,
        )
        digest = ContinuationStore(self.settings.path("checkpoints") / "continuations").put(
            envelope)
        ref = self.store.put_bytes(canonical_json(envelope.content()))
        self.ledger.register_artifact(ref.key, kind="continuation", raw_size=ref.raw_size,
                                      stored_size=ref.stored_size)
        return digest

    # --- freezing ---------------------------------------------------------------------

    def _terminal_output_ref(self, cell, *, phase_id: str, split: str, state: str) -> str:
        """Return a verifiable artifact for every terminal ITT assignment.

        A process may die after dispatch and leave FAILED_UNKNOWN without materialized graph
        bytes. That is still an outcome. In that case freeze a tombstone naming the exact cell,
        state and work key; never erase the assignment or rerun it until it looks successful.
        """
        work_key = self.work_key_for(cell, phase_id=phase_id, split=split)
        existing = self.ledger.terminal_ref(work_key)
        if existing and self.store.verify(existing):
            return existing
        tombstone = canonical_json({
            "cell": cell.content(),
            "run_id": self.config.run_id,
            "phase_id": phase_id,
            "work_key": work_key,
            "execution_binding_sha256": self.execution_binding_sha256,
            "protocol_document_sha256": self.protocol_document_sha256,
            "terminal_state": state,
            "artifact_kind": "terminal_failure_tombstone",
            "missing_or_corrupt_prior_ref": existing or "",
            "claim_scope": self.settings.claim_scope,
        })
        ref = self.store.put_bytes(tombstone)
        self.ledger.register_artifact(
            ref.key, kind="terminal_failure_tombstone",
            raw_size=ref.raw_size, stored_size=ref.stored_size, work_key=work_key,
        )
        return ref.key

    def freeze_blocks(self, manifest: ScheduleManifest, *, phase_id: str, split: str,
                      directory: Path) -> list[dict]:
        """Freeze terminal blocks and seal the full all-offered root once all are terminal.

        Per-block hashes alone cannot prove completeness: removing one block leaves all
        remaining hashes valid. ``FREEZE_ROOT.json`` is therefore written only when every
        scheduled block is terminal, and commits to the schedule plus every block outcome.
        """
        self._assert_manifest_identity(manifest)
        states = self.cell_states(manifest, phase_id=phase_id, split=split)
        outputs = {
            cell_key(cell): (
                self._terminal_output_ref(
                    cell, phase_id=phase_id, split=split, state=states[cell_key(cell)])
                if states.get(cell_key(cell)) in TERMINAL_STATES else ""
            )
            for cell in manifest.cells
        }
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        frozen: list[dict] = []
        owned_blocks = [block for block in manifest.blocks if self._owns(block)]
        for block in owned_blocks:
            record = freeze_record(block, states=states, outputs=outputs)
            if not record["terminal_frozen"]:
                continue
            record["execution_binding_sha256"] = self.execution_binding_sha256
            record["protocol_document_sha256"] = self.protocol_document_sha256
            epoch_by_cell = self._block_cell_epochs(
                block, phase_id=phase_id, split=split)
            for cell_record in record["cells"]:
                key = (
                    f"{cell_record['block_id']}:{cell_record['arm']['arm_id']}:"
                    f"{cell_record['replicate_id']}"
                )
                cell_record["engine_epoch"] = str(epoch_by_cell.get(key) or "")
            epochs = {epoch for epoch in epoch_by_cell.values() if epoch}
            record["engine_epochs"] = sorted(epochs)
            if len(epoch_by_cell) != len(block.cells) or any(
                not epoch for epoch in epoch_by_cell.values()
            ):
                self.ledger.record_incident(
                    severity="ERROR", kind="block_missing_engine_epoch",
                    detail=f"{block.block_id} has incomplete engine epoch provenance")
                record["invalid_reason"] = "MISSING_ENGINE_EPOCH"
                record["valid_for_paired_estimate"] = False
            elif any(epoch.startswith("SPANS:") for epoch in epochs):
                self.ledger.record_incident(
                    severity="ERROR", kind="cell_spans_engine_epochs",
                    detail=f"{block.block_id} contains within-cell epoch transition(s)")
                record["invalid_reason"] = "CELL_SPANS_ENGINE_EPOCHS"
                record["valid_for_paired_estimate"] = False
            elif len(epochs) > 1:
                # Complete, and still not one observation: these cells were served by two
                # different engines. Plan §16.3 makes the whole pre-registered block the
                # recovery unit precisely so this is never spliced into a paired result.
                self.ledger.record_incident(
                    severity="ERROR", kind="block_spans_engine_epochs",
                    detail=f"{block.block_id} ran under {sorted(epochs)}")
                record["invalid_reason"] = "SPANS_ENGINE_EPOCHS"
                record["valid_for_paired_estimate"] = False
            else:
                record["valid_for_paired_estimate"] = True
            # invalid_reason is part of the immutable record and therefore must be included in
            # the final digest, not appended after freeze_record computed it.
            body_for_hash = {k: v for k, v in record.items() if k != "freeze_sha256"}
            record["freeze_sha256"] = sha256_hex(canonical_json(body_for_hash))
            path = directory / f"{block.block_id}.json"
            _write_once_content_addressed_json(
                path, record, digest_field="freeze_sha256", label="frozen block")
            frozen.append(record)

        root_path = directory / FROZEN_ROOT_FILENAME
        if len(frozen) == len(owned_blocks):
            root = freeze_root_record(
                manifest,
                run_id=self.config.run_id,
                phase_id=phase_id,
                split=split,
                block_records=frozen,
                shard_id=self.config.shard_id,
                owned_block_ids=(
                    None if self.config.shard_id is None
                    else [block.block_id for block in owned_blocks]
                ),
            )
            expected_entries = {
                FROZEN_ROOT_FILENAME,
                *(f"{block.block_id}.json" for block in owned_blocks),
            }
            observed_entries = {path.name for path in directory.iterdir()}
            extra = sorted(observed_entries - expected_entries)
            missing = sorted(
                (expected_entries - {FROZEN_ROOT_FILENAME}) - observed_entries)
            if extra or missing:
                raise RuntimeError(
                    "frozen block directory differs from the offered schedule: "
                    f"missing={missing}, extra={extra}"
                )
            _write_once_content_addressed_json(
                root_path,
                root,
                digest_field="freeze_root_sha256",
                label="frozen campaign root",
            )
        elif root_path.exists():
            raise RuntimeError(
                f"{root_path} claims a complete share but only "
                f"{len(frozen)}/{len(owned_blocks)} of this lane's blocks are terminal"
            )
        return frozen

    def _owns(self, block) -> bool:
        """Is this block part of this lane's frozen share?

        A lane that ran a block the partition gave to another lane would produce a task executed
        on two GPUs -- exactly the pairing violation task-atomic sharding exists to prevent --
        and each lane would still look internally consistent. The merge catches it too; catching
        it here means the GPU time is never spent.
        """
        owned = self.config.owned_block_ids
        return owned is None or block.block_id in owned

    def _block_epochs(self, block, *, phase_id: str, split: str) -> set:
        """Which engine epochs this block's committed cells were actually run under."""
        return {
            epoch for epoch in self._block_cell_epochs(
                block, phase_id=phase_id, split=split).values()
            if epoch
        }

    def _block_cell_epochs(self, block, *, phase_id: str, split: str) -> dict[str, str]:
        """Bind each offered cell to the engine epoch recorded for its work item."""
        epochs: dict[str, str] = {}
        for cell in block.cells:
            row = self.ledger.raw_connection.execute(
                "SELECT engine_epoch FROM cell_epochs WHERE work_key=?",
                (self.work_key_for(cell, phase_id=phase_id, split=split),),
            ).fetchone()
            epochs[cell_key(cell)] = (
                str(row["engine_epoch"]) if row is not None and row["engine_epoch"] else "")
        return epochs

    def status(self, manifest: Optional[ScheduleManifest] = None, *, phase_id: str = "",
               split: str = "") -> dict:
        """A snapshot fit for reports/STATUS.json."""
        counts = self.ledger.state_counts()
        body = {
            "run_id": self.config.run_id,
            "execution_binding_sha256": self.execution_binding_sha256,
            "protocol_document_sha256": self.protocol_document_sha256,
            "phase": self.phases.current().value,
            "work_items": counts,
            "claim_scope": self.settings.claim_scope,
            "corpus_tier": self.settings.corpus_tier,
        }
        if manifest is not None:
            states = self.cell_states(manifest, phase_id=phase_id, split=split)
            body["cells_total"] = len(manifest.cells)
            body["cells_committed"] = sum(1 for s in states.values() if s == "COMMITTED")
            body["cells_terminal"] = sum(1 for s in states.values() if s in TERMINAL_STATES)
            body["blocks_total"] = len(manifest.blocks)
            body["blocks_complete"] = sum(
                1 for b in manifest.blocks if block_is_complete(b, states))
            body["blocks_complete_success"] = body["blocks_complete"]
            body["blocks_terminal"] = sum(
                1 for b in manifest.blocks if block_is_terminal(b, states))
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
