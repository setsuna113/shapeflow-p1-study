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

from dataclasses import dataclass
from typing import Optional

from ..evidence.identity import CandidateSet, OutOfSetLabel

__all__ = [
    "ParsedItem",
    "ParsedGap",
    "ParsedBridge",
    "ParsedSelection",
    "parse_selection",
    "SelectionContractError",
]


class SelectionContractError(ValueError):
    """The selector output violates its contract (bad shape, out-of-set label, empty bridge)."""


@dataclass(frozen=True)
class ParsedItem:
    span_id: str
    role: Optional[str]  # None for P1_ID (no role claimed)
    facet_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParsedGap:
    facet_id: str
    query_attempt_ids: tuple[str, ...]


@dataclass(frozen=True)
class ParsedBridge:
    text: str
    evidence_span_ids: tuple[str, ...]


@dataclass(frozen=True)
class ParsedSelection:
    contract: str
    items: tuple[ParsedItem, ...]
    gaps: tuple[ParsedGap, ...] = ()
    bridges: tuple[ParsedBridge, ...] = ()

    @property
    def selected_span_ids(self) -> tuple[str, ...]:
        return tuple(item.span_id for item in self.items)


def _resolve(candidates: CandidateSet, label: str) -> str:
    try:
        return candidates.resolve(label)
    except OutOfSetLabel as e:
        raise SelectionContractError(f"label {label!r} is not in the offered candidate set") from e


def parse_selection(raw: dict, candidates: CandidateSet) -> ParsedSelection:
    """Resolve a raw selector output against the candidate set. Assumes the raw dict already
    passed JSON-Schema validation; this adds candidate-membership and semantic checks."""
    contract = raw.get("contract")
    if contract == "P1_ID":
        items = tuple(
            ParsedItem(span_id=_resolve(candidates, lbl), role=None)
            for lbl in raw.get("selected_ids", [])
        )
        return ParsedSelection(contract=contract, items=items)

    if contract in {"P1_TYPED", "P1_BRIDGE"}:
        items = tuple(
            ParsedItem(
                span_id=_resolve(candidates, sel["span_id"]),
                role=sel["role"],
                facet_ids=tuple(sel.get("facet_ids", [])),
            )
            for sel in raw.get("selections", [])
        )
        gaps = tuple(
            ParsedGap(
                facet_id=g["facet_id"],
                query_attempt_ids=tuple(_resolve(candidates, q) for q in g["query_attempt_ids"]),
            )
            for g in raw.get("gaps", [])
        )
        bridges: tuple[ParsedBridge, ...] = ()
        if contract == "P1_BRIDGE":
            parsed_bridges = []
            for b in raw.get("bridges", []):
                evidence = tuple(_resolve(candidates, e) for e in b["evidence_ids"])
                if not evidence:
                    raise SelectionContractError("a bridge must bind at least one evidence id")
                if not b.get("text", "").strip():
                    raise SelectionContractError("a bridge must have non-empty text")
                parsed_bridges.append(ParsedBridge(text=b["text"], evidence_span_ids=evidence))
            bridges = tuple(parsed_bridges)
        return ParsedSelection(contract=contract, items=items, gaps=gaps, bridges=bridges)

    raise SelectionContractError(f"unknown contract {contract!r}")
