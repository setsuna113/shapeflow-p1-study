"""Forked-state component trials: many variants, one boundary, one set of input bytes.

This is what the design has always said the component screen is, and what it was not. The
screen re-ran the whole seven-arm graph per task, so every arm reached its own boundary with
its own upstream history, and the comparison absorbed all of it: different searches,
different pages, different assistant turns. Any measured difference between two selectors
was confounded with the run each of them happened to get.

A fork starts every variant from **the same stored checkpoint digest**. The parent boundary
is immutable and shared; each fork writes only new content-addressed objects. What differs
between two forks is the variant, and nothing else.

Forks also run with ``component_trial=True``, which switches the failure policy: a P1 that
fails here is recorded as a failure rather than falling back to P0. Falling back is right in
an end-to-end run -- the campaign should still produce a report -- but in a component trial
it would silently turn "this variant broke" into "this variant behaved exactly like P0",
which is the difference between a null result and a missing one. That policy existed and was
unreachable, because nothing ever set the flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from ..odr.checkpoints import CheckpointStore, fork_key
from .settings import Settings

__all__ = ["ForkSpec", "ForkOutcome", "plan_forks", "run_forks", "PROMPT_RENDERER_VERSION"]

#: Enters the fork key: a reworded prompt is a different execution of the same boundary,
#: not the same one resumed.
PROMPT_RENDERER_VERSION = "renderer_v1"


@dataclass(frozen=True)
class ForkSpec:
    """One variant's execution from one boundary."""

    fork_id: str
    checkpoint_digest: str
    boundary_kind: str          # "H" or "C"
    task_id: str
    variant_id: str
    seed: int

    def content(self) -> dict:
        return {
            "fork_id": self.fork_id,
            "checkpoint_digest": self.checkpoint_digest,
            "boundary_kind": self.boundary_kind,
            "task_id": self.task_id,
            "variant_id": self.variant_id,
            "seed": self.seed,
        }


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


@dataclass
class ForkPlan:
    checkpoint_digest: str
    boundary_kind: str
    task_id: str
    forks: list[ForkSpec] = field(default_factory=list)

    @property
    def variant_ids(self) -> list[str]:
        return [f.variant_id for f in self.forks]


def plan_forks(
    settings: Settings,
    *,
    checkpoint_digest: str,
    boundary_kind: str,
    task_id: str,
    variant_ids: Sequence[str],
    seed: int,
    protocol_sha: str = "",
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
    sha = protocol_sha or settings.shas["week1"]
    return ForkPlan(
        checkpoint_digest=checkpoint_digest,
        boundary_kind=boundary_kind,
        task_id=task_id,
        forks=[
            ForkSpec(
                fork_id=fork_key(
                    protocol_sha=sha,
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
            )
            for variant_id in ordered
        ],
    )


async def run_forks(
    settings: Settings,
    plan: ForkPlan,
    *,
    execute: Callable[[ForkSpec, Any], Any],
    checkpoint_store: Optional[CheckpointStore] = None,
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

    from ..hashing import sha256_hex

    outcomes: list[ForkOutcome] = []
    for spec in plan.forks:
        outcome = ForkOutcome(fork_id=spec.fork_id, variant_id=spec.variant_id,
                              checkpoint_digest=spec.checkpoint_digest)
        try:
            result = execute(spec, checkpoint)
            if hasattr(result, "__await__"):
                result = await result
        except Exception as e:  # noqa: BLE001 - a component trial records the failure
            # Deliberately not a fallback to P0. In a component trial that would turn
            # "this variant broke" into "this variant behaved exactly like P0", which is a
            # null result standing in for a missing one.
            outcome.state = "FAILED_FINAL"
            outcome.error = f"{type(e).__name__}: {e}"
            outcomes.append(outcome)
            continue
        outcome.state = "COMMITTED"
        payload = result if isinstance(result, (bytes, bytearray)) else str(result).encode()
        outcome.output_sha256 = sha256_hex(bytes(payload))
        outcomes.append(outcome)
    return outcomes


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
        "component_trial": True,
        "prompt_renderer_version": PROMPT_RENDERER_VERSION,
        "forks": [
            {**spec.content(),
             **{k: v for k, v in vars(outcome).items() if k != "fork_id"}}
            for spec, outcome in zip(plan.forks, outcomes)
        ],
        "inert_variants": inert_variants(outcomes),
    }
    path = directory / f"{plan.checkpoint_digest}.json"
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
