"""Boundary forks for component, H total-effect, C-only, and nested HxC trials.

The primary screen deliberately re-runs the complete graph for every arm.  Different searches,
pages and assistant turns *after the first intervention* are mediated outcomes of P1, so that
coupled-seed comparison is the correct end-to-end total effect.  It is not, however, a
same-input reducer comparison: calling those independently reached boundaries a component
trial would mix reducer behavior with the downstream trajectory it caused.

A component or C-only fork starts every variant from **the same stored checkpoint digest**.
For H E2E, equality is required at the first H intervention and deliberately not afterwards:
different queries, sources, rounds and close states are possible mediated treatment effects.
Nested HxC therefore lets H0/H1 reach different close checkpoints, then shares one close
checkpoint between C0/C1 *inside* each H trajectory.

Forks also run with ``component_trial=True``, which switches the failure policy: a P1 that
fails here is recorded as a failure rather than falling back to P0. Falling back is right in
an end-to-end run -- the campaign should still produce a report -- but in a component trial
it would silently turn "this variant broke" into "this variant behaved exactly like P0",
which is the difference between a null result and a missing one. That policy existed and was
unreachable, because nothing ever set the flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence, runtime_checkable

from ..canonical import canonical_json
from ..hashing import sha256_hex
from ..odr.checkpoints import CheckpointStore, fork_key
from .settings import Settings

__all__ = [
    "BoundaryCapture",
    "BoundaryForkBackend",
    "ForkCapabilityError",
    "ForkExecution",
    "ForkOutcome",
    "ForkPlan",
    "ForkSpec",
    "NestedHXCPlan",
    "PROMPT_RENDERER_VERSION",
    "TrialKind",
    "assert_comparable_executions",
    "first_trajectory_divergence",
    "inert_variants",
    "plan_c_only_e2e",
    "plan_forks",
    "plan_h_e2e",
    "plan_nested_hxc",
    "run_forks",
    "run_production_forks",
    "write_fork_record",
]

#: Enters the fork key: a reworded prompt is a different execution of the same boundary,
#: not the same one resumed.
PROMPT_RENDERER_VERSION = "renderer_v1"


class TrialKind(str, Enum):
    """The four estimands whose state-sharing rules are intentionally different.

    ``COMPONENT`` freezes the whole world at one reducer and changes only that reducer.
    ``H_E2E`` forks immediately before the first H intervention and then lets the two
    trajectories evolve independently: downstream divergence is part of H's total effect.
    ``C_ONLY_E2E`` forks at close, after all research, so *no* researcher work may differ.
    ``HXC_NESTED`` first creates the natural H0/H1 trajectories and only then forks C0/C1
    inside each trajectory's own close checkpoint.
    """

    COMPONENT = "COMPONENT"
    H_E2E = "H_E2E"
    C_ONLY_E2E = "C_ONLY_E2E"
    HXC_NESTED = "HXC_NESTED"
    #: Fork at close, run only the reducer, substitute that arm's note into the anchor's
    #: frozen note vector at exactly its slot, and run only the final report. A distinct
    #: estimand rather than a flavour of ``C_ONLY_E2E``, because the supervisor's mediated
    #: response is deliberately *blocked*: re-entering the supervisor would re-run
    #: SUPERVISOR_CONTINUE, which can spawn a fresh researcher with real search -- upstream
    #: work after the fork, differing between arms. Naming it separately stops any artifact
    #: from claiming an end-to-end run it did not perform; the full-graph C arms remain as
    #: the secondary sensitivity that bounds the blocked path.
    C_FROZEN_CONTINUATION = "C_FROZEN_CONTINUATION"


class ForkCapabilityError(RuntimeError):
    """The pinned graph cannot honestly execute the requested fork semantics."""


@dataclass(frozen=True)
class ForkSpec:
    """One variant's execution from one boundary."""

    fork_id: str
    checkpoint_digest: str
    boundary_kind: str          # "H" or "C"
    task_id: str
    variant_id: str
    seed: int
    execution_binding_sha256: str
    trial_kind: TrialKind = TrialKind.COMPONENT
    parent_fork_id: str = ""

    def content(self) -> dict:
        return {
            "fork_id": self.fork_id,
            "checkpoint_digest": self.checkpoint_digest,
            "boundary_kind": self.boundary_kind,
            "task_id": self.task_id,
            "variant_id": self.variant_id,
            "seed": self.seed,
            "execution_binding_sha256": self.execution_binding_sha256,
            "trial_kind": self.trial_kind.value,
            "parent_fork_id": self.parent_fork_id,
        }


@dataclass(frozen=True)
class BoundaryCapture:
    """One real boundary captured during a single anchor trajectory.

    ``upstream_trace_sha256`` commits to the complete pre-boundary trajectory, rather than
    merely naming the task.  Two independently re-run graphs can reach equal-looking
    checkpoints by accident; a component trial must replay this stored digest itself.
    """

    checkpoint_digest: str
    boundary_kind: str
    task_id: str
    anchor_run_ref: str
    upstream_trace_sha256: str
    seed: int
    seed_applied: bool


@dataclass(frozen=True)
class ForkExecution:
    """Auditable result returned by a real boundary replay/resume backend.

    A bare string is deliberately insufficient in production.  The executor must prove which
    checkpoint it loaded, that the sampling seed reached the engine, and whether it performed
    any upstream researcher work.  ``trajectory_events`` are ordered, canonical event dicts;
    downstream differences are retained for H E2E instead of treated as an error.
    """

    output: Any
    start_checkpoint_digest: str
    first_treatment_checkpoint_digest: str
    seed_applied: bool
    upstream_research_calls: int = 0
    trajectory_events: tuple[Mapping[str, Any], ...] = ()
    terminal_close_checkpoint_digest: str = ""
    output_ref: str = ""


@dataclass
class ForkOutcome:
    fork_id: str
    variant_id: str
    checkpoint_digest: str
    state: str = "PENDING"
    output_ref: str = ""
    error: str = ""
    #: The bytes this variant published, for the differ. Two forks of one boundary that
    #: produce identical output are two names for the same behaviour.
    output_sha256: str = ""
    first_treatment_checkpoint_digest: str = ""
    seed_applied: bool = False
    upstream_research_calls: int = 0
    trajectory_sha256: str = ""
    terminal_close_checkpoint_digest: str = ""


@dataclass
class ForkPlan:
    checkpoint_digest: str
    boundary_kind: str
    task_id: str
    execution_binding_sha256: str
    trial_kind: TrialKind = TrialKind.COMPONENT
    parent_fork_id: str = ""
    forks: list[ForkSpec] = field(default_factory=list)

    @property
    def variant_ids(self) -> list[str]:
        return [f.variant_id for f in self.forks]

    @property
    def downstream_divergence_is_an_outcome(self) -> bool:
        return self.trial_kind in (TrialKind.H_E2E, TrialKind.HXC_NESTED)


@dataclass(frozen=True)
class NestedHXCPlan:
    """The correct 2x2 topology: one H fork, then one C fork inside each H trajectory."""

    h_plan: ForkPlan
    c_plans_by_h_variant: Mapping[str, ForkPlan]


@runtime_checkable
class BoundaryForkBackend(Protocol):
    """Capability the run host must implement before ``run-screen`` may spend anything.

    The current pinned ODR graph has no durable LangGraph checkpointer and cannot resume its
    compiled graph from an H/C state.  Defining this protocol is not claiming otherwise: the
    production caller fails closed until an adapter implements and validates these methods on
    the actual graph.
    """

    def supports(self, trial_kind: TrialKind, boundary_kind: str) -> bool: ...

    async def capture_boundaries(
        self, *, task_id: str, question: str, seed: int
    ) -> Sequence[BoundaryCapture]: ...

    async def execute(self, spec: ForkSpec, checkpoint: Any) -> ForkExecution: ...


def plan_forks(
    settings: Settings,
    *,
    checkpoint_digest: str,
    boundary_kind: str,
    task_id: str,
    variant_ids: Sequence[str],
    seed: int,
    execution_binding_sha256: str,
    trial_kind: TrialKind = TrialKind.COMPONENT,
    parent_fork_id: str = "",
) -> ForkPlan:
    """Plan one boundary's forks: P0 plus the variants assigned to it.

    P0 is prepended and de-duplicated rather than assumed present. Plan §14.4 requires every
    state to contain P0, and the freeze gate that was supposed to enforce it only checked
    that the assignment was non-empty -- which it says so in its own comment.
    """
    ordered: list[str] = ["P0"]
    for variant_id in variant_ids:
        if variant_id not in ordered:
            ordered.append(variant_id)
    if len(ordered) < 2:
        raise ValueError(
            f"boundary {checkpoint_digest[:12]} has no P1 variants assigned; a fork set of "
            "P0 alone measures nothing"
        )
    if boundary_kind not in {"H", "C"}:
        raise ValueError(f"boundary_kind must be H or C, got {boundary_kind!r}")
    if trial_kind is TrialKind.H_E2E and boundary_kind != "H":
        raise ValueError("an H E2E treatment must fork at an H checkpoint")
    if trial_kind is TrialKind.C_ONLY_E2E and boundary_kind != "C":
        raise ValueError("a C-only E2E treatment must fork at a C checkpoint")
    if trial_kind is TrialKind.C_FROZEN_CONTINUATION and boundary_kind != "C":
        raise ValueError("a frozen-continuation treatment must fork at a C checkpoint")
    if (
        len(execution_binding_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in execution_binding_sha256)
    ):
        raise ValueError("execution_binding_sha256 must be a lowercase SHA-256 digest")
    return ForkPlan(
        checkpoint_digest=checkpoint_digest,
        boundary_kind=boundary_kind,
        task_id=task_id,
        execution_binding_sha256=execution_binding_sha256,
        trial_kind=trial_kind,
        parent_fork_id=parent_fork_id,
        forks=[
            ForkSpec(
                fork_id=fork_key(
                    # The low-level helper retains its historical parameter name, but the
                    # namespace is the complete approved execution binding.
                    protocol_sha=execution_binding_sha256,
                    boundary_id=checkpoint_digest,
                    variant_id=variant_id,
                    seed=seed,
                    prompt_renderer_version=PROMPT_RENDERER_VERSION,
                ),
                checkpoint_digest=checkpoint_digest,
                boundary_kind=boundary_kind,
                task_id=task_id,
                variant_id=variant_id,
                seed=seed,
                execution_binding_sha256=execution_binding_sha256,
                trial_kind=trial_kind,
                parent_fork_id=parent_fork_id,
            )
            for variant_id in ordered
        ],
    )


def plan_h_e2e(
    settings: Settings,
    *,
    first_h_checkpoint_digest: str,
    task_id: str,
    page_variant_ids: Sequence[str],
    seed: int,
    execution_binding_sha256: str,
) -> ForkPlan:
    """Plan the total effect of H from the first eligible H checkpoint.

    The common checkpoint is required only at the first intervention.  A later query, H
    checkpoint, close reason or source set is allowed to differ and must be measured as a
    mediated outcome, not overwritten with the baseline trajectory.
    """
    return plan_forks(
        settings,
        checkpoint_digest=first_h_checkpoint_digest,
        boundary_kind="H",
        task_id=task_id,
        variant_ids=page_variant_ids,
        seed=seed,
        execution_binding_sha256=execution_binding_sha256,
        trial_kind=TrialKind.H_E2E,
    )


def plan_c_only_e2e(
    settings: Settings,
    *,
    close_checkpoint_digest: str,
    task_id: str,
    close_variant_ids: Sequence[str],
    seed: int,
    execution_binding_sha256: str,
    parent_h_fork_id: str = "",
) -> ForkPlan:
    """Plan C0/C1 from one close checkpoint.

    C happens after researcher search has ended, so unlike H E2E it has no licence to re-run
    or change upstream research.  ``parent_h_fork_id`` identifies which natural H trajectory
    supplied the close checkpoint in a nested HxC design.
    """
    kind = TrialKind.HXC_NESTED if parent_h_fork_id else TrialKind.C_ONLY_E2E
    return plan_forks(
        settings,
        checkpoint_digest=close_checkpoint_digest,
        boundary_kind="C",
        task_id=task_id,
        variant_ids=close_variant_ids,
        seed=seed,
        execution_binding_sha256=execution_binding_sha256,
        trial_kind=kind,
        parent_fork_id=parent_h_fork_id,
    )


def plan_nested_hxc(
    h_plan: ForkPlan,
    *,
    close_checkpoint_by_h_variant: Mapping[str, str],
    close_variant_ids: Sequence[str],
    settings: Settings,
    seed: int,
    execution_binding_sha256: str,
) -> NestedHXCPlan:
    """Build H0/H1 natural trajectories with a C0/C1 fork inside each.

    The two H branches are *not* required to reach the same close checkpoint -- that
    difference may be H's mediated effect.  Within one H branch, however, every C variant
    receives that branch's exact same C checkpoint.
    """
    if h_plan.trial_kind is not TrialKind.H_E2E or h_plan.boundary_kind != "H":
        raise ValueError("nested HxC requires an H_E2E parent plan")
    if h_plan.execution_binding_sha256 != execution_binding_sha256:
        raise ValueError(
            "nested HxC child plans must use the parent H plan's execution binding")
    missing = sorted(set(h_plan.variant_ids) - set(close_checkpoint_by_h_variant))
    if missing:
        raise ValueError(f"H branches have no terminal C checkpoint: {missing}")
    c_plans: dict[str, ForkPlan] = {}
    for h_spec in h_plan.forks:
        c_plans[h_spec.variant_id] = plan_c_only_e2e(
            settings,
            close_checkpoint_digest=close_checkpoint_by_h_variant[h_spec.variant_id],
            task_id=h_plan.task_id,
            close_variant_ids=close_variant_ids,
            seed=seed,
            execution_binding_sha256=execution_binding_sha256,
            parent_h_fork_id=h_spec.fork_id,
        )
    return NestedHXCPlan(h_plan=h_plan, c_plans_by_h_variant=c_plans)


def _execution_payload(result: ForkExecution) -> bytes:
    value = result.output
    return bytes(value) if isinstance(value, (bytes, bytearray)) else str(value).encode()


def _validate_execution(spec: ForkSpec, result: ForkExecution) -> None:
    if result.start_checkpoint_digest != spec.checkpoint_digest:
        raise ForkCapabilityError(
            f"{spec.variant_id} loaded {result.start_checkpoint_digest!r}, not the planned "
            f"checkpoint {spec.checkpoint_digest!r}"
        )
    if result.first_treatment_checkpoint_digest != spec.checkpoint_digest:
        raise ForkCapabilityError(
            f"{spec.variant_id} first intervened at "
            f"{result.first_treatment_checkpoint_digest!r}, not the shared boundary "
            f"{spec.checkpoint_digest!r}"
        )
    if not result.seed_applied:
        raise ForkCapabilityError(
            f"{spec.variant_id} recorded seed {spec.seed} but did not prove that it reached "
            "every model request; this execution is not a paired replicate"
        )
    if spec.trial_kind in (TrialKind.COMPONENT, TrialKind.C_ONLY_E2E,
                            TrialKind.C_FROZEN_CONTINUATION,
                            TrialKind.HXC_NESTED) and result.upstream_research_calls:
        raise ForkCapabilityError(
            f"{spec.trial_kind.value} replay performed {result.upstream_research_calls} "
            "upstream researcher calls; it re-ran the trajectory instead of forking it"
        )


def assert_comparable_executions(
    plan: ForkPlan, executions: Sequence[ForkExecution]
) -> None:
    """Validate the shared-state invariant without suppressing legitimate H divergence."""
    if len(executions) != len(plan.forks):
        raise ForkCapabilityError(
            f"plan has {len(plan.forks)} forks but backend returned {len(executions)} executions"
        )
    for spec, result in zip(plan.forks, executions, strict=True):
        _validate_execution(spec, result)
    if plan.trial_kind in (TrialKind.COMPONENT, TrialKind.C_ONLY_E2E,
                           TrialKind.C_FROZEN_CONTINUATION,
                           TrialKind.HXC_NESTED):
        starts = {r.start_checkpoint_digest for r in executions}
        if starts != {plan.checkpoint_digest}:
            raise ForkCapabilityError("component/C forks did not load one shared checkpoint")
    # Deliberately no equality check over post-treatment trajectories for H_E2E.  Those
    # differences -- searches, rounds, sources and close reason -- are outcomes.


def first_trajectory_divergence(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> Optional[dict]:
    """Return the first ordered event mismatch, retaining which side ended first."""
    width = max(len(left), len(right))
    for index in range(width):
        l_event = left[index] if index < len(left) else None
        r_event = right[index] if index < len(right) else None
        if canonical_json(l_event) != canonical_json(r_event):
            return {"event_index": index, "left": l_event, "right": r_event}
    return None


async def run_forks(
    settings: Settings,
    plan: ForkPlan,
    *,
    execute: Callable[[ForkSpec, Any], Any],
    checkpoint_store: Optional[CheckpointStore] = None,
    require_provenance: bool = False,
) -> list[ForkOutcome]:
    """Run every fork of one boundary from the one checkpoint, and report each separately.

    ``execute(spec, checkpoint) -> output`` is injected: this module owns the invariant that
    all forks share one boundary, not the details of running a variant.

    The checkpoint is loaded once and its digest re-verified on load, so "every variant
    started from byte-identical input" is a property of the run rather than of a docstring.
    """
    store = checkpoint_store or CheckpointStore(settings.path("checkpoints"))
    checkpoint = store.get(plan.checkpoint_digest)
    if checkpoint.digest != plan.checkpoint_digest:  # pragma: no cover - store re-verifies
        raise ValueError("the loaded checkpoint is not the one planned")
    if require_provenance:
        recorded_seed = getattr(getattr(checkpoint, "sampling", None), "seed", None)
        planned_seeds = {spec.seed for spec in plan.forks}
        if len(planned_seeds) != 1 or recorded_seed not in planned_seeds:
            raise ForkCapabilityError(
                f"checkpoint sampling seed {recorded_seed!r} does not equal planned fork seed "
                f"{sorted(planned_seeds)!r}"
            )

    outcomes: list[ForkOutcome] = []
    for spec in plan.forks:
        outcome = ForkOutcome(fork_id=spec.fork_id, variant_id=spec.variant_id,
                              checkpoint_digest=spec.checkpoint_digest)
        try:
            result = execute(spec, checkpoint)
            if hasattr(result, "__await__"):
                result = await result
            if require_provenance and not isinstance(result, ForkExecution):
                raise ForkCapabilityError(
                    "a production fork backend must return ForkExecution provenance; a bare "
                    "payload cannot prove shared input, zero upstream rerun, or seed delivery"
                )
            execution = result if isinstance(result, ForkExecution) else ForkExecution(
                output=result,
                start_checkpoint_digest=spec.checkpoint_digest,
                first_treatment_checkpoint_digest=spec.checkpoint_digest,
                # Legacy injected unit executors predate request provenance.  They remain useful
                # for pure unit tests but production always sets require_provenance=True.
                seed_applied=False,
            )
            if require_provenance:
                _validate_execution(spec, execution)
        except ForkCapabilityError:
            # Provenance/capability failure invalidates the campaign, not one treatment arm.
            # Converting it to FAILED_FINAL would let the other arms run and spend against an
            # estimand the backend cannot implement.
            raise
        except Exception as e:  # noqa: BLE001 - a component trial records the failure
            # Deliberately not a fallback to P0. In a component trial that would turn
            # "this variant broke" into "this variant behaved exactly like P0", which is a
            # null result standing in for a missing one.
            outcome.state = "FAILED_FINAL"
            outcome.error = f"{type(e).__name__}: {e}"
            outcomes.append(outcome)
            continue
        outcome.state = "COMMITTED"
        outcome.output_ref = execution.output_ref
        outcome.output_sha256 = sha256_hex(_execution_payload(execution))
        outcome.first_treatment_checkpoint_digest = \
            execution.first_treatment_checkpoint_digest
        outcome.seed_applied = execution.seed_applied
        outcome.upstream_research_calls = execution.upstream_research_calls
        outcome.trajectory_sha256 = sha256_hex(canonical_json(
            list(execution.trajectory_events)))
        outcome.terminal_close_checkpoint_digest = \
            execution.terminal_close_checkpoint_digest
        outcomes.append(outcome)
    return outcomes


async def run_production_forks(
    settings: Settings,
    plan: ForkPlan,
    *,
    backend: Optional[BoundaryForkBackend],
    checkpoint_store: Optional[CheckpointStore] = None,
) -> list[ForkOutcome]:
    """Production entry point: capability-check first, before any model call or spend."""
    if backend is None:
        raise ForkCapabilityError(
            "the pinned ODR graph has no validated boundary-resume backend; refusing to "
            "re-run one full graph per arm and label it a checkpoint fork"
        )
    if not backend.supports(plan.trial_kind, plan.boundary_kind):
        raise ForkCapabilityError(
            f"backend does not support {plan.trial_kind.value} at {plan.boundary_kind}; "
            "unsupported estimands must fail closed"
        )
    return await run_forks(
        settings,
        plan,
        execute=backend.execute,
        checkpoint_store=checkpoint_store,
        require_provenance=True,
    )


def inert_variants(outcomes: Sequence[ForkOutcome]) -> list[str]:
    """Variants whose output is byte-identical to P0's from the same boundary.

    From one shared boundary this is a real signal rather than a coincidence: same input,
    same output means the variant did nothing. Across separately-run graphs it would have
    been meaningless, which is part of why the screen could not detect an inert arm.
    """
    by_variant = {o.variant_id: o for o in outcomes if o.state == "COMMITTED"}
    baseline = by_variant.get("P0")
    if baseline is None or not baseline.output_sha256:
        return []
    return sorted(
        variant_id for variant_id, outcome in by_variant.items()
        if variant_id != "P0" and outcome.output_sha256 == baseline.output_sha256
    )


def write_fork_record(directory: Path, plan: ForkPlan, outcomes: Sequence[ForkOutcome]) -> Path:
    """One record per boundary: what forked from it, and what each fork produced."""
    import json

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    body = {
        "checkpoint_digest": plan.checkpoint_digest,
        "boundary_kind": plan.boundary_kind,
        "task_id": plan.task_id,
        "execution_binding_sha256": plan.execution_binding_sha256,
        "trial_kind": plan.trial_kind.value,
        "component_trial": plan.trial_kind is TrialKind.COMPONENT,
        "downstream_divergence_is_an_outcome":
            plan.downstream_divergence_is_an_outcome,
        "parent_fork_id": plan.parent_fork_id,
        "prompt_renderer_version": PROMPT_RENDERER_VERSION,
        "forks": [
            {**spec.content(),
             **{k: v for k, v in vars(outcome).items() if k != "fork_id"}}
            for spec, outcome in zip(plan.forks, outcomes, strict=True)
        ],
        "inert_variants": inert_variants(outcomes),
    }
    path = directory / f"{plan.checkpoint_digest}.json"
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
