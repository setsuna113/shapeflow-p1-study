"""Parsing and resolving a selector's structured output.

The model emits short labels (E1, Q1) under one of three contracts. This module turns that
raw output into a resolved, validated selection: every label is looked up in the candidate
set (an out-of-set label is a hard rejection, never leniently dropped), and the contract's
structural rules are enforced. It does **not** check truth -- whether the selection is any
*good* is an evaluator question.

The three contracts exist to separate mechanisms the study must not confound:

- ``P1_ID`` -- pure pointers, no roles, no prose. Isolates the pointer mechanism.
- ``P1_TYPED`` -- pointers plus roles/facets/gaps, all mechanically checkable.
- ``P1_BRIDGE`` -- adds bounded connective text, every clause bound to evidence and charged
  to the selector's decode cost.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..evidence.identity import EVIDENCE, QUERY_ATTEMPT, CandidateSet, OutOfSetLabel

__all__ = [
    "ParsedItem",
    "ParsedGap",
    "ParsedBridge",
    "ParsedSelection",
    "canonical_normalization_document",
    "parse_selection",
    "OVERALL_FACET",
    "validate_selector_output",
    "SelectionContractError",
    "P1_CONTRACTS",
]

_SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schemas" / "selector_output.schema.json"
P1_CONTRACTS = frozenset({"P1_ID", "P1_TYPED", "P1_BRIDGE"})


# A role claimed without naming a facet is a claim about the question as a whole. Representing
# that as "no facet" made it invisible to any per-facet rule, so two contradictory overall
# claims about one span looked like two unrelated annotations. The sentinel makes the overall
# relation a first-class facet that conflict detection can see.
OVERALL_FACET = "__overall__"


class SelectionContractError(ValueError):
    """The selector output violates its contract (bad shape, out-of-set label, empty bridge).

    Carries the ``normalization`` record accumulated up to the failure. A rejection that took
    its counters with it never reached the strict-valid rate, which needs the denominator --
    how often output was malformed -- not only the repairs that happened to succeed.
    """

    def __init__(self, message: str, normalization: "NormalizationRecord | None" = None) -> None:
        super().__init__(message)
        self.normalization = normalization


@dataclass(frozen=True)
class ParsedItem:
    """One selected span and every (facet, role) relation the selector claimed for it.

    Roles are scoped to a facet. A source can support an efficacy claim and undercut a safety
    claim in the same breath; treating "same span, two roles" as a flat contradiction rejected
    the whole sample and so failed P1 hardest on exactly the conflict-rich tasks the study is
    for. The conflict rule is per ``(span_id, facet_id)``.
    """

    span_id: str
    relations: tuple[tuple[str, Optional[str]], ...] = ()  # (facet_id, role), ordered

    @property
    def role(self) -> Optional[str]:
        """The single role, when the span plays exactly one. None if it plays several."""
        roles = {r for _, r in self.relations if r}
        return roles.pop() if len(roles) == 1 else None

    @property
    def facet_ids(self) -> tuple[str, ...]:
        return tuple(f for f, _ in self.relations if f != OVERALL_FACET)


@dataclass(frozen=True)
class ParsedGap:
    facet_id: str
    query_attempt_ids: tuple[str, ...]


@dataclass(frozen=True)
class ParsedBridge:
    text: str
    evidence_span_ids: tuple[str, ...]


@dataclass(frozen=True)
class NormalizationRecord:
    """What repair, if any, the raw selector output needed.

    Normalizing silently would let a variant that emits malformed output look identical to one
    that does not, hiding a real quality difference between contracts. These counters feed the
    reported strict-valid rate and the "no repair at all" sensitivity analysis, and the raw
    completion tokens are charged in full regardless -- repair is free for us, not for the GPU.
    """

    raw_count: int
    unique_count: int
    duplicate_count: int
    semantic_conflict_count: int
    # Set on every rejection, so the strict-valid rate has its denominator: schema failures and
    # out-of-set labels used to raise with normalization=None, which counted them as if they
    # had never happened.
    rejected_reason: Optional[str] = None

    @property
    def was_repaired(self) -> bool:
        return self.duplicate_count > 0 or self.semantic_conflict_count > 0

    @property
    def was_rejected(self) -> bool:
        return self.rejected_reason is not None


_CLEAN = NormalizationRecord(0, 0, 0, 0)

_NORMALIZATION_BASE_FIELDS = frozenset({
    "raw_count",
    "unique_count",
    "duplicate_count",
    "semantic_conflict_count",
    "rejected_reason",
})


def canonical_normalization_document(value: object) -> dict:
    """Closed-validate one selector-normalization trace and recompute its flags.

    This document crosses the treatment/evaluator boundary, so callers must not accept an
    arbitrary mapping or trust serialized ``was_repaired``/``strict_valid`` booleans.  The
    five primitive fields are the complete wire contract; derived flags are produced here
    from those primitives and checked again by the evaluator.
    """
    if not isinstance(value, dict):
        raise ValueError("normalization must be an object")
    actual = frozenset(map(str, value))
    if actual != _NORMALIZATION_BASE_FIELDS:
        missing = sorted(_NORMALIZATION_BASE_FIELDS - actual)
        extra = sorted(actual - _NORMALIZATION_BASE_FIELDS)
        raise ValueError(
            f"normalization fields are not closed (missing={missing}, extra={extra})"
        )
    counts: dict[str, int] = {}
    for key in (
        "raw_count",
        "unique_count",
        "duplicate_count",
        "semantic_conflict_count",
    ):
        raw = value.get(key)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ValueError(f"normalization.{key} must be a non-negative integer")
        counts[key] = raw
    reason = value.get("rejected_reason")
    if reason is not None and (not isinstance(reason, str) or not reason.strip()):
        raise ValueError(
            "normalization.rejected_reason must be null or a non-empty string"
        )
    repaired = (
        counts["duplicate_count"] > 0
        or counts["semantic_conflict_count"] > 0
    )
    rejected = reason is not None
    return {
        **counts,
        "rejected_reason": reason,
        "was_repaired": repaired,
        "was_rejected": rejected,
        "strict_valid": not repaired and not rejected,
    }


@dataclass(frozen=True)
class ParsedSelection:
    contract: str
    items: tuple[ParsedItem, ...]
    gaps: tuple[ParsedGap, ...] = ()
    bridges: tuple[ParsedBridge, ...] = ()
    normalization: NormalizationRecord = _CLEAN

    @property
    def selected_span_ids(self) -> tuple[str, ...]:
        return tuple(item.span_id for item in self.items)


@functools.lru_cache(maxsize=1)
def _validator():
    import jsonschema

    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    return jsonschema.Draft202012Validator(schema)


def validate_selector_output(raw: dict) -> None:
    """Validate raw selector output against ``schemas/selector_output.schema.json``.

    This runs *here*, not "somewhere upstream". The previous contract said the caller had
    already validated -- and no production caller existed, so in practice nothing did. Model
    output is untrusted input; the closed schema is what bounds list lengths, keeps E and Q
    labels in their own positions, and rejects unknown fields before any of it is resolved.
    """
    errors = sorted(_validator().iter_errors(raw), key=lambda e: list(e.absolute_path))
    if errors:
        first = errors[0]
        where = "/".join(str(p) for p in first.absolute_path) or "<root>"
        tally = _Tally(raw=_raw_element_count(raw))
        raise SelectionContractError(
            f"selector output failed schema at {where}: {first.message}",
            tally.record(rejected="schema"),
        )


def _raw_element_count(raw: dict) -> int:
    return sum(len(raw.get(k, []) or []) for k in ("selected_ids", "selections", "gaps", "bridges"))


def _resolve(candidates: CandidateSet, label: str, *, kind: str, tally: "_Tally | None" = None) -> str:
    try:
        return candidates.resolve(label, kind=kind)
    except OutOfSetLabel as e:
        record = (tally or _Tally()).record(rejected="out_of_set_label")
        raise SelectionContractError(
            f"label {label!r} is not in the offered {kind} candidate set", record
        ) from e


def parse_selection(
    raw: dict,
    candidates: CandidateSet,
    *,
    expected_contract: str | None = None,
) -> ParsedSelection:
    """Validate and resolve a raw selector output against the candidate set.

    Three layers, in order: the closed JSON Schema (shape, bounds, label kind), candidate
    membership (an out-of-set label is a hard rejection, never leniently dropped), then the
    contract's own semantic rules. None of them consults truth -- whether the selection is any
    *good* is an evaluator question and deliberately not decidable here.
    """
    if expected_contract is not None:
        if expected_contract not in P1_CONTRACTS:
            raise ValueError(
                f"expected_contract must be one of {sorted(P1_CONTRACTS)}, "
                f"got {expected_contract!r}"
            )
        actual = raw.get("contract") if isinstance(raw, dict) else None
        if actual != expected_contract:
            tally = _Tally(raw=_raw_element_count(raw) if isinstance(raw, dict) else 0)
            raise SelectionContractError(
                f"selector returned contract {actual!r}, but this experimental arm is locked "
                f"to {expected_contract!r}; accepting it would cross treatment arms",
                tally.record(rejected="contract_mismatch"),
            )
    validate_selector_output(raw)
    contract = raw.get("contract")
    tally = _Tally()

    if contract == "P1_ID":
        tally = _Tally(raw=len(raw.get("selected_ids", [])))
        resolved = [
            _resolve(candidates, lbl, kind=EVIDENCE, tally=tally)
            for lbl in raw.get("selected_ids", [])
        ]
        ids = tally.dedup(resolved)
        return ParsedSelection(
            contract=contract,
            items=tuple(ParsedItem(span_id=sid) for sid in ids),
            normalization=tally.record(unique=len(ids)),
        )

    if contract in {"P1_TYPED", "P1_BRIDGE"}:
        raw_selections = raw.get("selections", [])
        raw_gaps_in = raw.get("gaps", [])
        raw_bridges_in = raw.get("bridges", [])
        tally.raw = len(raw_selections) + len(raw_gaps_in) + len(raw_bridges_in)

        pairs: list[tuple[str, str, Optional[str]]] = []   # (span_id, facet_id, role)
        for sel in raw_selections:
            span_id = _resolve(
                candidates, sel["span_id"], kind=EVIDENCE, tally=tally
            )
            facets = tally.dedup(list(sel.get("facet_ids", []))) or [OVERALL_FACET]
            for facet in facets:
                pairs.append((span_id, facet, sel["role"]))
        items = _normalize_relations(pairs, tally)

        gaps = _normalize_gaps(
            [
                ParsedGap(
                    facet_id=g["facet_id"],
                    query_attempt_ids=tuple(
                        tally.dedup(
                            [_resolve(candidates, q, kind=QUERY_ATTEMPT, tally=tally)
                             for q in g["query_attempt_ids"]]
                        )
                    ),
                )
                for g in raw_gaps_in
            ],
            tally,
        )

        bridges: tuple[ParsedBridge, ...] = ()
        if contract == "P1_BRIDGE":
            parsed: list[ParsedBridge] = []
            for b in raw_bridges_in:
                evidence = tally.dedup(
                    [
                        _resolve(candidates, e, kind=EVIDENCE, tally=tally)
                        for e in b["evidence_ids"]
                    ]
                )
                if not evidence:
                    raise SelectionContractError(
                        "a bridge must bind at least one evidence id",
                        tally.record(rejected="semantic_invalid"),
                    )
                if not b.get("text", "").strip():
                    raise SelectionContractError(
                        "a bridge must have non-empty text",
                        tally.record(rejected="semantic_invalid"),
                    )
                bridge = ParsedBridge(text=b["text"], evidence_span_ids=tuple(evidence))
                if bridge in parsed:
                    tally.duplicates += 1
                    continue
                parsed.append(bridge)
            bridges = tuple(parsed)

        selection = ParsedSelection(
            contract=contract, items=items, gaps=gaps, bridges=bridges,
            normalization=tally.record(unique=len(items) + len(gaps) + len(bridges)),
        )
        _check_facet_closure(selection, tally)
        return selection

    raise SelectionContractError(
        f"unknown contract {contract!r}", tally.record(rejected="unknown_contract")
    )


@dataclass
class _Tally:
    """Accumulates normalization counters so a rejection can report them too."""

    raw: int = 0
    duplicates: int = 0
    conflicts: int = 0

    def dedup(self, values: list[str]) -> list[str]:
        """Stable first-seen dedup, counting what it dropped.

        Used for every repeated-value field, including *inside* one field: `facet_ids=["f","f"]`
        and `evidence_ids=["E1","E1"]` are repairs too, and reporting duplicate_count=0 for them
        made the strict-valid rate wrong in the safe-looking direction.
        """
        seen: list[str] = []
        for v in values:
            if v not in seen:
                seen.append(v)
        self.duplicates += len(values) - len(seen)
        return seen

    def record(self, *, unique: int = 0, rejected: Optional[str] = None) -> NormalizationRecord:
        return NormalizationRecord(
            raw_count=self.raw, unique_count=unique,
            duplicate_count=self.duplicates, semantic_conflict_count=self.conflicts,
            rejected_reason=rejected,
        )


def _normalize_relations(
    pairs: list[tuple[str, str, Optional[str]]], tally: _Tally
) -> tuple[ParsedItem, ...]:
    """Fold (span, facet, role) triples into one item per span, rejecting per-facet conflicts.

    The conflict key is ``(span_id, facet_id)``. One span supporting `efficacy` while
    contradicting `safety` is a coherent, common claim and is kept; the same span given two
    roles on one facet is incoherent, and picking by output order would let the model's line
    order decide the result.
    """
    by_span: dict[str, dict[str, Optional[str]]] = {}
    order: list[str] = []
    for span_id, facet, role in pairs:
        if span_id not in by_span:
            by_span[span_id] = {}
            order.append(span_id)
        prior = by_span[span_id].get(facet, _MISSING)
        if prior is _MISSING:
            by_span[span_id][facet] = role
            continue
        tally.duplicates += 1
        if prior != role:
            tally.conflicts += 1
            raise SelectionContractError(
                f"span {span_id[:12]} was given conflicting roles on facet {facet!r} "
                f"({prior!r} and {role!r}); a span bears on one facet one way, and picking by "
                "output order would let the model's line order decide the result",
                tally.record(rejected="semantic_conflict"),
            )
    return tuple(
        ParsedItem(span_id=sid, relations=tuple(by_span[sid].items())) for sid in order
    )


_MISSING = object()


# --- semantic normalization -----------------------------------------------------------
#
# Duplicate labels are common in model output and are not, by themselves, an error. But
# "duplicates are harmless, the aggregator dedups them" is false: the aggregator deduped by
# span_id alone, first-wins, so two lines naming the same span with *different* roles resolved
# to whichever the model happened to emit first. Swapping two lines of output flipped the
# recorded role while preflight passed both ways -- and the pair also looked like a preserved
# contradiction to the guard that exists to detect exactly that.
#
# So each shape gets a stated, auditable rule rather than positional luck:
#
#   exact duplicate            -> drop, counted
#   same span, same role,
#     different facets         -> union facets in first-seen order, counted
#   same span, different roles -> hard reject; there is no defensible merge
#   repeated gap facet         -> union its query attempts in first-seen order, counted
#   identical bridge           -> drop, counted


def _normalize_gaps(gaps: list[ParsedGap], tally: "_Tally") -> tuple[ParsedGap, ...]:
    """One gap per facet; a repeated facet unions its query attempts in first-seen order."""
    by_facet: dict[str, list[str]] = {}
    order: list[str] = []
    for gap in gaps:
        if gap.facet_id not in by_facet:
            by_facet[gap.facet_id] = []
            order.append(gap.facet_id)
        else:
            tally.duplicates += 1
        for qid in gap.query_attempt_ids:
            if qid not in by_facet[gap.facet_id]:
                by_facet[gap.facet_id].append(qid)
    return tuple(ParsedGap(facet_id=f, query_attempt_ids=tuple(by_facet[f])) for f in order)


def _check_facet_closure(selection: ParsedSelection, tally: "_Tally") -> None:
    """Every facet the selector names must be either answered or explicitly declared a gap.

    A facet that appears on neither side is a silent omission: the report loses the facet
    without ever saying it looked and found nothing, which is exactly the "gap honesty" the
    quality guards score. Catching it here keeps the honesty requirement mechanical rather
    than something the evaluator has to infer after the fact.
    """
    answered = {f for item in selection.items for f in item.facet_ids}
    declared = {g.facet_id for g in selection.gaps}
    overlap = answered & declared
    if overlap:
        tally.conflicts += 1
        raise SelectionContractError(
            f"facet(s) {sorted(overlap)} are both selected-for and declared a gap; "
            "a facet is either answered or explicitly unanswered, never both",
            tally.record(rejected="semantic_conflict"),
        )
