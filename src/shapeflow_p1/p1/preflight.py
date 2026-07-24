"""Publish-time preflight: structural gates, and nothing about truth.

Preflight is the last check before P1 output is published downstream. It verifies *form*:
ids resolve, offsets reconstruct, citations close, budgets hold, bridges stay bounded, and a
declared contradiction keeps both sides. It must never consult a TruthPacket, gold facet, or
critical-item list -- those are evaluator-only, and letting any of them gate the treatment
would let the selector be tuned against the answer key it is later scored on. Truth-critical
recall is computed *after* publish, by the isolated evaluator.

A failure here is recorded, not hidden: in a component trial the sample is marked failed; in
an end-to-end run the policy falls back to P0 and the P1 cost already spent is still charged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..evidence.chunkers import Tokenizer
from ..evidence.lineage import lineage_closure_errors, reconstruction_errors
from .aggregators import AggregatedEvidence
from .contracts import ParsedSelection
from .renderer import RenderResult

__all__ = ["PreflightConfig", "PreflightResult", "preflight"]


@dataclass(frozen=True)
class PreflightConfig:
    selected_token_budget: int
    bridge_token_cap_total: int | None = None
    bridge_token_cap_each: int | None = None


@dataclass
class PreflightResult:
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def preflight(
    *,
    selection: ParsedSelection,
    aggregated: AggregatedEvidence,
    rendered: RenderResult,
    registry: dict,
    snapshot_texts: dict[str, str],
    known_occurrence_ids: set[str],
    tokenizer: Tokenizer,
    config: PreflightConfig,
) -> PreflightResult:
    result = PreflightResult()

    # 1. Every aggregated span must be a real candidate.
    spans = []
    for item in aggregated.items:
        span = registry.get(item.span_id)
        if span is None:
            result.errors.append(f"published span {item.span_id[:12]} not in candidate registry")
        else:
            spans.append(span)

    # 2. Offsets still reconstruct the recorded bytes (raw-source spans).
    result.errors.extend(reconstruction_errors(spans, snapshot_texts))

    # 3. Every citation resolves to a real occurrence in the frozen world.
    result.errors.extend(lineage_closure_errors(spans, known_occurrence_ids))

    # 4. A declared contradiction keeps both sides: if the selection marked both support and
    #    contradict for a facet, the aggregated output must retain both -- the aggregator may
    #    not one-side a disagreement to save budget.
    kept_ids = {it.span_id for it in aggregated.items}
    facet_roles: dict[str, set[str]] = {}
    facet_role_ids: dict[tuple[str, str], set[str]] = {}
    for item in selection.items:
        for facet in item.facet_ids:
            if item.role:
                facet_roles.setdefault(facet, set()).add(item.role)
                facet_role_ids.setdefault((facet, item.role), set()).add(item.span_id)
    for facet, roles in facet_roles.items():
        if {"support", "contradict"} <= roles:
            for role in ("support", "contradict"):
                if not (facet_role_ids[(facet, role)] & kept_ids):
                    result.errors.append(
                        f"contradiction on facet {facet!r} was one-sided: no {role} span survived"
                    )

    # 5. Rendered token budget -- the binding materialized-token gate.
    if rendered.token_count > config.selected_token_budget:
        result.errors.append(
            f"rendered {rendered.token_count} tokens exceeds selected_token_budget "
            f"{config.selected_token_budget}"
        )

    # 6. Bridge caps: an unbounded bridge is just P0 in disguise.
    if aggregated.bridges:
        total = 0
        for bridge in aggregated.bridges:
            each = tokenizer.count(bridge.text)
            total += each
            if config.bridge_token_cap_each is not None and each > config.bridge_token_cap_each:
                result.errors.append(
                    f"bridge exceeds per-bridge token cap ({each} > {config.bridge_token_cap_each})"
                )
        if config.bridge_token_cap_total is not None and total > config.bridge_token_cap_total:
            result.errors.append(
                f"bridges exceed total token cap ({total} > {config.bridge_token_cap_total})"
            )

    return result
