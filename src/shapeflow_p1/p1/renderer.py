"""Deterministic serialization of aggregated evidence into downstream prompt bytes.

The renderer is pure, CPU-only string assembly -- no semantic rewriting, so the exact span
bytes reach the next stage unchanged and can still be re-hashed at preflight. Its token count
is the *binding* budget quantity: the study measures what the P1 path materializes downstream,
so the aggregator's budget is enforced against this renderer's real output (via ``make_coster``),
not against a count of ids.

Three things the renderer must never do, because each silently changes what the comparison
means:

- **Cost less than it renders.** The coster renders the *whole* evidence object -- items, gaps
  and bridges -- not just the items. Costing items alone understates every P1 arm's budget by
  exactly the output that distinguishes TYPED and BRIDGE from ID.
- **Import bytes the selector never saw.** ``source_meta`` (title/URL) is display-only, and on
  the C_VISIBLE path it must be *proved* to come from the compressor-visible bytes. Injecting a
  title from the acquisition record would hand C_VISIBLE provenance its compressor never had,
  which is the whole difference between C_VISIBLE and C_REGISTRY.
- **Report a failed lookup as an absence.** A gap whose backing queries all timed out is not
  "no evidence found"; saying so converts an infrastructure failure into a claim about the
  world, and the gap-honesty guards then score it as a correct one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from ..evidence.chunkers import Tokenizer
from .aggregators import AggregatedEvidence, Coster
from .contracts import OVERALL_FACET

__all__ = [
    "RenderResult", "render", "make_coster", "SourceMetaLeak",
    "RENDERER_GROUPING_VERSION", "context_text_for",
]

# Sharing one SOURCE header across a contiguous run changes how many tokens P1 materializes
# relative to P0, so it is a frozen, named behaviour rather than an implementation detail. It
# enters the protocol hash; changing it mints a new protocol SHA.
RENDERER_GROUPING_VERSION = "renderer_source_grouping_v1"

# Given a span dict, return its exact text (snapshot slice or message slice).
SourceTextFn = Callable[[dict], str]
# Given a span id, return the short display label (E3).
LabelFn = Callable[[str], str]
# Given a context_ref dict, return the exact frozen bytes it addresses.
ContextTextFn = Callable[[dict], str]

# Query outcomes that actually looked at the source universe. Anything else (TIMEOUT, FAILED,
# unknown) means we never got an answer.
_ANSWERED_STATUSES = frozenset({"OK", "EMPTY"})


class SourceMetaLeak(ValueError):
    """Display metadata was offered that is not present in the compressor-visible bytes."""


@dataclass(frozen=True)
class RenderResult:
    text: str
    token_count: int
    byte_len: int


def _span_id_of(span: dict) -> str:
    return span.get("span_id") or span.get("visible_span_id") or ""


def _no_context(ref: dict) -> str:
    raise SourceMetaLeak(
        "a span carries context_refs but no context_text_for was supplied; context is frozen "
        "evidence and may not be rendered without resolving it from the snapshot"
    )


def context_text_for(snapshot_texts: dict[str, str]) -> ContextTextFn:
    """Resolve a context_ref against the frozen snapshots it addresses.

    Context (currently a table's header row) used to be a free string carried on the span. That
    made it an unbound channel straight into the prompt: tampering with it left the span id
    unchanged, RAW_SOURCE reconstruction passing and preflight ok, while the injected text was
    rendered downstream. A context_ref is ``{content_hash, char_start, char_end, text_sha256}``
    -- addressed and re-hashed exactly like any other evidence.
    """

    def resolve(ref: dict) -> str:
        text = snapshot_texts[ref["content_hash"]]
        return text[ref["char_start"]:ref["char_end"]]

    return resolve


def _source_of(span: dict) -> str:
    return span.get("content_hash") or span.get("message_id") or _span_id_of(span)


def _start_end(span: dict) -> tuple[int, int]:
    if "char_start" in span:
        return span["char_start"], span["char_end"]
    return span["byte_start"], span["byte_end"]


def _canonical_order(items, registry: dict) -> list:
    """The one order the renderer and the coster both use.

    Without this the cost estimate depends on the order the aggregator happened to hand over,
    so the budget gate is off by whatever the difference between that order and the published
    order comes to.
    """
    return sorted(items, key=lambda it: _order_key(registry[it.span_id]))


def _order_key(span: dict) -> tuple:
    start, _ = _start_end(span)
    return (span.get("namespace", ""), _source_of(span), start, _span_id_of(span))


def _contiguous_runs(items, registry: dict) -> list[list]:
    """Group consecutive items that come from one source and abut in that source.

    Two selected sentences that are adjacent in the same page share one SOURCE header instead
    of two. That removes the duplicated header tokens without fusing the spans themselves --
    each keeps its own id, label, annotations and recorded hash, so preflight can still
    reconstruct every one of them and a bridge can still cite them individually.

    Runs are formed at render time rather than in an aggregator so all three aggregators share
    the behaviour and it cannot become a difference between them. Because it does change the
    P1-vs-P0 materialized-token difference, it is a named, frozen renderer behaviour --
    ``RENDERER_GROUPING_VERSION`` -- and enters the protocol hash.
    """
    runs: list[list] = []
    for item in items:
        span = registry[item.span_id]
        if runs:
            prev = registry[runs[-1][-1].span_id]
            same_source = _source_of(prev) == _source_of(span) and (
                prev.get("namespace") == span.get("namespace")
            )
            # Abutting, allowing the whitespace a chunker leaves between blocks.
            adjacent = same_source and 0 <= _start_end(span)[0] - _start_end(prev)[1] <= 2
            if adjacent:
                runs[-1].append(item)
                continue
        runs.append([item])
    return runs


def _title_url(span_id: str, source_meta: dict) -> str:
    meta = source_meta.get(span_id, {})
    title = meta.get("title", "")
    url = meta.get("url", "")
    if title or url:
        return f"{title} — {url}".strip(" —")
    return "(source)"


def _check_visible_provenance(
    span: dict, span_id: str, source_meta: dict, visible_views: dict[str, bytes]
) -> None:
    """For a VISIBLE_MESSAGE span, title/URL must appear in the bytes the compressor saw.

    Without this, C_VISIBLE can be handed a clean title and URL from the acquisition record --
    metadata that P0's compressor, reading a model-written page summary, may never have had.
    That is not cosmetic: source attribution is one of the things the citation metrics score.
    """
    meta = source_meta.get(span_id)
    if not meta:
        return
    view = visible_views.get(span["visible_compressor_view_hash"])
    if view is None:
        raise SourceMetaLeak(
            f"span {span_id[:12]}: cannot prove title/URL provenance -- the compressor view "
            "is not available to check against"
        )
    for field in ("title", "url"):
        value = meta.get(field)
        if value and value.encode("utf-8") not in view:
            raise SourceMetaLeak(
                f"span {span_id[:12]}: {field} {value!r} does not appear in the "
                "compressor-visible bytes; that is C_REGISTRY provenance, not C_VISIBLE"
            )


def render(
    evidence: AggregatedEvidence,
    registry: dict,
    *,
    source_text_for: SourceTextFn,
    label_for: LabelFn,
    tokenizer: Tokenizer,
    source_meta: Optional[dict] = None,
    visible_views: Optional[dict[str, bytes]] = None,
    query_status: Optional[dict[str, str]] = None,
    context_text_for: Optional[ContextTextFn] = None,
) -> RenderResult:
    """Render aggregated evidence to a deterministic string with its token count.

    ``source_meta`` optionally maps span id -> {title, url} for the source header line; it is
    display-only and never affects selection. On the C_VISIBLE path, pass ``visible_views``
    (view hash -> exact compressor bytes) so provenance is checked rather than trusted.

    ``query_status`` maps query_attempt_id -> terminal status, so a gap backed only by lookups
    that never returned is rendered as a search failure rather than as an absence.
    """
    source_meta = source_meta or {}
    visible_views = visible_views or {}
    query_status = query_status or {}
    context_text_for = context_text_for or _no_context
    lines: list[str] = []

    for run in _contiguous_runs(_canonical_order(evidence.items, registry), registry):
        first = registry[run[0].span_id]
        first_id = _span_id_of(first) or run[0].span_id
        if first.get("namespace") == "VISIBLE_MESSAGE":
            _check_visible_provenance(first, first_id, source_meta, visible_views)
        # ONLY the source header is shared across the run. Everything below stays bound to the
        # span it describes: collapsing a run's roles and facets into one line leaves no way to
        # tell which role belongs to which span, so a contradiction the selector correctly
        # marked becomes unreadable downstream.
        lines.append("SOURCE: " + _title_url(first_id, source_meta))
        for item in run:
            span = registry[item.span_id]
            bits = [f"[{label_for(item.span_id)}]"]
            # Every (facet, role) relation, each bound to THIS span. A span can support one
            # facet and contradict another; flattening to a single role would erase exactly the
            # disagreement the contradiction guard exists to preserve.
            relations = getattr(item, "relations", None)
            if relations:
                shown = [f"{role}:{facet}" if facet != OVERALL_FACET else str(role)
                         for facet, role in relations if role]
                if shown:
                    bits.append("(" + ", ".join(shown) + ")")
            elif item.role:
                bits.append(f"({item.role})")
                if item.facet_ids:
                    bits.append("facets: " + ",".join(item.facet_ids))
            # Breadcrumb and context both come from addressed, re-hashed ranges -- never from
            # free strings, which would be an unbound channel into the prompt.
            headings = [context_text_for(r) for r in span.get("heading_refs") or []]
            if headings:
                bits.append("under: " + " > ".join(headings))
            lines.append(" ".join(bits))
            for ref in span.get("context_refs") or []:
                lines.append(f"  ctx| {context_text_for(ref)}")
            lines.append(source_text_for(span))
        lines.append("")

    for gap in evidence.gaps:
        attempts = ",".join(gap.query_attempt_ids)
        statuses = {query_status.get(q, "UNKNOWN") for q in gap.query_attempt_ids}
        if statuses & _ANSWERED_STATUSES:
            note = "no evidence found"
        else:
            note = f"search did not complete ({'/'.join(sorted(statuses))}); coverage unknown"
        lines.append(f"[GAP facet {gap.facet_id}] {note} ({attempts})")

    for bridge in evidence.bridges:
        refs = ",".join(label_for(sid) for sid in bridge.evidence_span_ids)
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
    source_meta: Optional[dict] = None,
    query_status: Optional[dict[str, str]] = None,
    context_text_for: Optional[ContextTextFn] = None,
) -> Coster:
    """Build a coster that estimates budget by actually rendering, so the aggregator and the
    renderer agree on what a 'token' costs.

    The coster receives the full evidence object, gaps and bridges included, and renders it in
    the same canonical order the published output uses. Costing only the items -- as this once
    did -- systematically understates the P1 budget by exactly the lines that separate
    P1_TYPED and P1_BRIDGE from P1_ID; costing in the caller's order makes the gate off by the
    difference between that order and the published one.
    """

    def coster(evidence: AggregatedEvidence, registry: dict) -> int:
        return render(
            evidence, registry, source_text_for=source_text_for, label_for=label_for,
            tokenizer=tokenizer, source_meta=source_meta, query_status=query_status,
            context_text_for=context_text_for,
        ).token_count

    return coster
