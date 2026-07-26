"""The C-fork phase: one anchor per task, then every arm from each of its close boundaries.

Shape of a pass, per task:

1. **Anchor.** One full-graph P0 cell. It produces the research trajectory, every close
   checkpoint, the continuation envelope the arms write their reports into, and -- through the
   provider ledger -- the upstream work every arm will carry.
2. **Fork.** For each close boundary the anchor captured, run P0 and every C variant from that
   one stored checkpoint. Each arm re-enters ``compress_research``, proves by re-derived digest
   that it started at the planned boundary, substitutes its note into the anchor's frozen note
   vector at exactly its slot, and writes a report.

Arms of a boundary run **back to back, immediately after their anchor**. That is not a
scheduling convenience: ``get_today_str()`` is ``datetime.now()`` and is formatted into both
the compressor and the final-report prompt, and the shared upstream constant belongs to one
engine boot. Spreading a boundary's arms across a UTC midnight or an engine restart would make
them differ by more than their treatment, and both are checked rather than assumed.

Failures stay in the offered set. An arm that raises is recorded ``FAILED_FINAL`` and its pair
is still a pair -- dropping it would leave only the boundaries where P1 happened to work, which
is the survivorship bias this design exists to avoid. A *capability* failure is different and
deliberately aborts the phase: it means the backend cannot honestly perform the estimand at
all, and letting the remaining arms run would spend against a claim that cannot be made.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..canonical import canonical_json
from ..hashing import sha256_hex
from ..odr.checkpoints import CheckpointStore
from ..odr.continuation import ContinuationStore
from .c_fork_backend import CForkBackend
from .c_fork_work import (
    ForkAccountingError,
    assert_arms_share_one_upstream,
    attribution_for_arm,
    partition_anchor_work,
)
from .fork import (
    TrialKind,
    plan_forks,
    run_production_forks,
    write_fork_record,
)
from .settings import Settings

__all__ = ["CForkPhaseError", "c_fork_arms", "run_c_fork_boundary", "summarize_boundary"]

PHASE_ID = "c-fork"


class CForkPhaseError(RuntimeError):
    """The phase cannot proceed honestly."""


def c_fork_arms(settings: Settings) -> list[str]:
    """The variants this phase forks, from the hash-locked config.

    P0 is required and is not optional decoration: without a baseline from the *same* boundary
    there is nothing to compare a reducer against, and ``plan_forks`` refuses a fork set that
    is P0 alone.
    """
    block = settings.get("week1", "c_fork")
    arms = [str(v) for v in (block.get("arms") or [])]
    if "P0" not in arms:
        raise CForkPhaseError("the C-fork arm set must include P0 as the same-boundary baseline")
    if len(arms) < 2:
        raise CForkPhaseError("a fork set of P0 alone measures nothing")
    return arms


async def run_c_fork_boundary(
    settings: Settings,
    *,
    repo: Path,
    checkpoint_digest: str,
    boundary_ordinal: int,
    task_id: str,
    seed: int,
    execution_binding_sha256: str,
    backend: CForkBackend,
    shared_upstream,
    post_boundary_work,
    records_dir: Path,
) -> dict:
    """Fork one close boundary across every arm, and write its record.

    ``post_boundary_work(fork_id) -> mapping`` is injected rather than read here, because the
    work belongs to the provider ledger and this module's job is the attribution, not the query.
    """
    arms = c_fork_arms(settings)
    plan = plan_forks(
        settings,
        checkpoint_digest=checkpoint_digest,
        boundary_kind="C",
        task_id=task_id,
        variant_ids=arms,
        seed=seed,
        execution_binding_sha256=execution_binding_sha256,
        trial_kind=TrialKind.C_FROZEN_CONTINUATION,
    )

    # A capability failure is re-raised by run_production_forks and must stay fatal: it means
    # no arm of any boundary can be executed honestly, so continuing would spend against an
    # estimand that cannot be claimed.
    outcomes = await run_production_forks(
        settings, plan, backend=backend,
        checkpoint_store=CheckpointStore(settings.path("checkpoints")),
    )

    attributions: dict[str, dict] = {}
    for spec, outcome in zip(plan.forks, outcomes, strict=True):
        if outcome.state != "COMMITTED":
            # A failed arm has no honest work attribution, and inventing one would put a
            # fabricated saving next to a failure. It stays in the offered set regardless.
            continue
        attributions[spec.variant_id] = attribution_for_arm(
            shared_upstream, post_boundary_work(spec.fork_id)
        )
    if attributions:
        assert_arms_share_one_upstream(list(attributions.values()))

    record_path = write_fork_record(records_dir, plan, outcomes)
    body = json.loads(record_path.read_text(encoding="utf-8"))
    body["phase_id"] = PHASE_ID
    body["boundary_ordinal"] = boundary_ordinal
    body["seed"] = seed
    body["estimand"] = TrialKind.C_FROZEN_CONTINUATION.value
    body["terminal_mode"] = str(settings.get("week1", "c_fork").get("terminal_mode", "REPORT"))
    body["work_attribution"] = attributions
    body["anchor_work_partition"] = shared_upstream.content()
    # All-offered: every arm that was planned appears here, terminal or not. A denominator
    # built from `attributions` alone would silently be the successful-P1 subset.
    body["offered_variants"] = list(plan.variant_ids)
    body["terminal_variants"] = sorted(o.variant_id for o in outcomes if o.state == "COMMITTED")
    body["failed_variants"] = sorted(o.variant_id for o in outcomes if o.state != "COMMITTED")
    body["content_sha256"] = sha256_hex(canonical_json(
        {k: v for k, v in body.items() if k != "content_sha256"}))
    record_path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return body


def summarize_boundary(body: dict) -> dict:
    """The one-line view: what was offered, what survived, and whether it is comparable."""
    offered = list(body.get("offered_variants") or [])
    terminal = list(body.get("terminal_variants") or [])
    attributions = body.get("work_attribution") or {}
    return {
        "checkpoint_digest": body.get("checkpoint_digest", ""),
        "boundary_ordinal": body.get("boundary_ordinal", -1),
        "estimand": body.get("estimand", ""),
        "offered": len(offered),
        "terminal": len(terminal),
        "failed": sorted(set(offered) - set(terminal)),
        "inert_variants": list(body.get("inert_variants") or []),
        # A boundary is usable for the paired estimate only when the baseline and at least one
        # treatment both reached terminal *and* carry the same upstream constant.
        "valid_for_paired_estimate": bool(
            "P0" in terminal and len(terminal) >= 2 and len(attributions) >= 2
        ),
    }


def load_anchor_shared_upstream(
    ledger: Any, *, anchor_work_key: str, boundary_count: int
) -> list:
    """The per-boundary upstream constants, read from the anchor's own ledger rows."""
    from ..runtime.work_accounting import extract_request_events

    extraction = extract_request_events(ledger, work_keys=[anchor_work_key])
    if not extraction.complete:
        raise ForkAccountingError(
            f"anchor {anchor_work_key} has incomplete request telemetry; its upstream work "
            "cannot be attributed to the boundaries that follow it"
        )
    return partition_anchor_work(
        extraction.events, anchor_work_key=anchor_work_key, boundary_count=boundary_count
    )


def anchor_continuation(settings: Settings, digest: str):
    """Load the anchor's continuation envelope, verifying its digest on the way in."""
    return ContinuationStore(settings.path("checkpoints") / "continuations").get(digest)


def c_boundary_digests(checkpoints: Sequence[dict]) -> list[str]:
    """Close-boundary digests from an anchor's captured checkpoints, in capture order."""
    return [
        str(entry["digest"]) for entry in checkpoints
        if str(entry.get("kind", "")) == "CCheckpoint"
    ]


def terminal_mode(settings: Settings) -> str:
    """``REPORT`` or ``CLOSE_ONLY``, from the hash-locked config.

    A config-visible state rather than a runtime flag: it changes what the phase can claim, so
    it enters the execution binding and cannot be flipped after seeing results.
    """
    mode = str(settings.get("week1", "c_fork").get("terminal_mode", "REPORT"))
    if mode not in ("REPORT", "CLOSE_ONLY"):
        raise CForkPhaseError(f"unknown c_fork.terminal_mode {mode!r}")
    return mode
