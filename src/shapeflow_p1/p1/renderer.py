"""Deterministic serialization of aggregated evidence into downstream prompt bytes.

The renderer is pure, CPU-only string assembly -- no semantic rewriting, so the exact span
bytes reach the next stage unchanged and can still be re-hashed at preflight. Its token count
is the *binding* budget quantity: the study measures what the P1 path materializes downstream,
so the aggregator's budget is enforced against this renderer's real output (via ``make_coster``),
not against a count of ids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..evidence.chunkers import Tokenizer
from .aggregators import AggregatedEvidence, AggregatedItem, Coster

__all__ = ["RenderResult", "render", "make_coster"]

# Given a span dict, return its exact text (snapshot slice or message slice).
SourceTextFn = Callable[[dict], str]
# Given a span id, return the short display label (E3).
LabelFn = Callable[[str], str]


@dataclass(frozen=True)
class RenderResult:
    text: str
    token_count: int
    byte_len: int


def _title_url(span: dict, registry_meta: dict) -> str:
    meta = registry_meta.get(span.get("span_id") or span.get("visible_span_id"), {})
    title = meta.get("title", "")
    url = meta.get("url", "")
    if title or url:
        return f"{title} — {url}".strip(" —")
    return "(source)"


def render(
    evidence: AggregatedEvidence,
    registry: dict,
    *,
    source_text_for: SourceTextFn,
    label_for: LabelFn,
    tokenizer: Tokenizer,
    source_meta: dict | None = None,
) -> RenderResult:
    """Render aggregated evidence to a deterministic string with its token count.

    ``source_meta`` optionally maps span id -> {title, url} for the source header line; it is
    display-only and never affects selection.
    """
    source_meta = source_meta or {}
    lines: list[str] = []

    for item in evidence.items:
        span = registry[item.span_id]
        label = label_for(item.span_id)
        header_bits = [f"[{label}]"]
        if item.role:
            header_bits.append(f"({item.role})")
        if item.facet_ids:
            header_bits.append("facets: " + ",".join(item.facet_ids))
        header_bits.append("SOURCE: " + _title_url(span, {item.span_id: source_meta.get(item.span_id, {})}))
        lines.append(" ".join(header_bits))
        lines.append(source_text_for(span))
        lines.append("")

    for gap in evidence.gaps:
        attempts = ",".join(gap.query_attempt_ids)
        lines.append(f"[GAP facet {gap.facet_id}] no evidence found ({attempts})")

    for bridge in evidence.bridges:
        refs = ",".join(bridge.evidence_span_ids and
                        [label_for(sid) for sid in bridge.evidence_span_ids])
        lines.append(f"[BRIDGE] {bridge.text} ({refs})")

    text = "\n".join(lines)
    return RenderResult(
        text=text, token_count=tokenizer.count(text), byte_len=len(text.encode("utf-8"))
    )


def make_coster(
    *,
    source_text_for: SourceTextFn,
    label_for: LabelFn,
    tokenizer: Tokenizer,
) -> Coster:
    """Build a coster that estimates budget by actually rendering the items, so the aggregator
    and the renderer agree on what a 'token' costs."""

    def coster(items: list[AggregatedItem], registry: dict) -> int:
        ev = AggregatedEvidence(items=tuple(items))
        return render(
            ev, registry, source_text_for=source_text_for, label_for=label_for,
            tokenizer=tokenizer,
        ).token_count

    return coster
