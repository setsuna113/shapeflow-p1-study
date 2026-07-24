"""Variant elimination and promotion (plan §14.5-§14.6).

Screening decides which variants survive to the mini end-to-end stage. Two rules matter most,
because both guard against over-claiming:

- **Elimination judges a variant, not a family.** A hard-gate failure kills that specific design.
  Only if *every* legal variant at a node shares a deterministic structural failure may the node
  be called KILL_STRUCTURAL; otherwise, if nothing advances, the node is NOT_ESTABLISHED (or a
  pre-frozen best-case sentinel goes to a confirmatory no-headroom block) -- never "the whole
  P1 idea is dead".

- **Ties break toward simplicity.** When two candidates' work saving is within the pre-frozen
  indifference band, the simpler, lower-freedom one wins: ID < TYPED < BRIDGE, union < coverage <
  rerank, separate < fused. So the study never promotes extra freedom it did not need, which is
  what keeps a later "BRIDGE helps" claim honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "ScreenResult",
    "ScreenGates",
    "screen_eliminate",
    "simplicity_key",
    "promote",
    "node_verdict_when_none_advance",
]

_CONTRACT_RANK = {"id": 0, "typed": 1, "bridge": 2}
_AGG_RANK = {"stable_union": 0, "coverage_budget": 1, "global_rerank": 2, "vendor": 0}
_CLOSE_RANK = {"separate": 0, "dedicated_selector": 0, "prefix_preserving": 1, "fused_with_fallback": 2}


@dataclass(frozen=True)
class ScreenResult:
    variant_id: str
    node: str
    structural_pass: bool          # no out-of-set id / namespace / reconstruction error
    selector_quality: float        # absolute selector quality (e.g. weighted recall)
    harm_screen_pass: bool         # end-to-end harm screen
    work_saving: float             # complete (all-offered) work saving fraction
    crash_rate: float
    # tie-break descriptors
    contract: str = "id"
    aggregation: str = "stable_union"
    close_mode: str = "separate"
    is_control: bool = False       # controls (P0, cpu-lexical, short-prose) never promote
    # if eliminated for a deterministic structural reason shared by the whole family
    deterministic_structural_failure: bool = False


@dataclass(frozen=True)
class ScreenGates:
    min_selector_quality: float = 0.90
    max_crash_rate: float = 0.10
    indifference_band: float = 0.02
    max_per_node: int = 2


def screen_eliminate(r: ScreenResult, gates: ScreenGates) -> Optional[str]:
    """Return an elimination reason for this variant, or None if it survives screening."""
    if not r.structural_pass:
        return "structural failure (out-of-set id / wrong namespace / reconstruction)"
    if not r.harm_screen_pass:
        return "end-to-end harm screen failed"
    if r.selector_quality < gates.min_selector_quality:
        return f"selector quality {r.selector_quality:.3f} < {gates.min_selector_quality}"
    if r.crash_rate > gates.max_crash_rate:
        return f"crash/timeout rate {r.crash_rate:.3f} > {gates.max_crash_rate}"
    # A variant whose work clearly increases with no quality headroom is eliminated too.
    if r.work_saving <= 0 and r.selector_quality < 1.0:
        return "work increased with no quality gain"
    return None


def simplicity_key(r: ScreenResult) -> tuple[int, int, int]:
    """Lower is simpler / lower-freedom. Used only to break an indifference-band tie."""
    return (
        _CONTRACT_RANK.get(r.contract, 1),
        _AGG_RANK.get(r.aggregation, 1),
        _CLOSE_RANK.get(r.close_mode, 0),
    )


def promote(results: list[ScreenResult], gates: ScreenGates) -> dict[str, list[str]]:
    """Promote up to ``max_per_node`` variants per node.

    Survivors are bucketed by work saving at the indifference-band granularity; within a bucket
    the simpler variant ranks first. So a marginally-higher-saving but more complex variant does
    not beat a simpler one whose saving is within the band.
    """
    by_node: dict[str, list[ScreenResult]] = {}
    for r in results:
        if r.is_control:
            continue
        if screen_eliminate(r, gates) is not None:
            continue
        by_node.setdefault(r.node, []).append(r)

    promoted: dict[str, list[str]] = {}
    band = gates.indifference_band if gates.indifference_band > 0 else 1e-9
    for node, survivors in by_node.items():
        ranked = sorted(
            survivors,
            key=lambda r: (-round(r.work_saving / band), simplicity_key(r), r.variant_id),
        )
        promoted[node] = [r.variant_id for r in ranked[: gates.max_per_node]]
    return promoted


def node_verdict_when_none_advance(
    node: str, node_results: list[ScreenResult]
) -> str:
    """When no candidate advances at ``node``, decide the node-level label.

    KILL_STRUCTURAL only if every legal (non-control) variant shares a deterministic structural
    failure -- one bad prompt is never enough to kill a family. Otherwise NOT_ESTABLISHED, which
    routes to a pre-frozen best-case sentinel for a confirmatory no-headroom/harm block.
    """
    legal = [r for r in node_results if not r.is_control]
    if legal and all(r.deterministic_structural_failure for r in legal):
        return "KILL_STRUCTURAL"
    return "NOT_ESTABLISHED"
