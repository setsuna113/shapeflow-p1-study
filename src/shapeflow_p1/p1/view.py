"""The candidate view: one object owning everything the selector sees and everything published.

Block 1 closed three injection channels one at a time -- a caller-supplied body resolver, a
free-text context field, an unhashed heading breadcrumb -- and each fix left the next one open.
They were all the same defect in different clothes: *the caller hands a value to the
publication path*. As long as `render(source_text_for=...)`, `label_for=...` and `coster=...`
exist as parameters, closing the known ones is a treadmill, and each new one has to be found
by hand before it is closed.

So this changes the interface instead of adding checks. A ``CandidateViewRecord`` is built once
from frozen storage and owns:

- every candidate's exact bytes, resolved by namespace from ``snapshot_texts``/``visible_views``
- the short labels (E1..En) and the query-attempt labels (Q1..Qm)
- context and heading text, each addressed into the *same* source as its span
- the rendered prompt bytes and their digest
- the prompt-bundle and renderer-grouping versions that shape both

Callers pass the record. There is no parameter through which a body, a label, a cost or a piece
of context can be supplied, so there is nothing left to substitute.

Two further rules the record enforces at construction, before any model call:

**The whole offered set is the boundary.** A C_VISIBLE view containing one RAW_SOURCE candidate
already broke the compressor-only claim even if that candidate was never selected -- the model
read it. Checking only the published spans' namespace was too late by one model call.

**Context is same-source.** A context ref pointing into another snapshot is not context; it is
evidence imported from somewhere the span does not live.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Optional, Sequence

from ..canonical import canonical_json
from ..evidence.chunkers import Tokenizer
from ..evidence.identity import CandidateSet
from ..hashing import sha256_hex
from .prompts import PROMPT_BUNDLE_VERSION, render_selector_prompt

__all__ = [
    "CandidateViewRecord",
    "OfferedCandidate",
    "ViewConstructionError",
    "guard_publication",
]


class ViewConstructionError(ValueError):
    """The offered candidate set cannot be built as declared.

    Always raised *before* a selector call. Everything it catches -- a foreign namespace, a
    cross-source context ref, an unresolvable body -- would otherwise become a property of what
    the model already read, at which point no downstream check can undo it.
    """


@dataclass(frozen=True)
class OfferedCandidate:
    """One candidate exactly as the selector will see it, with its bytes already resolved."""

    label: str
    span_id: str
    namespace: str
    text: str
    heading_path: tuple[str, ...] = ()
    context: tuple[str, ...] = ()

    def identity(self) -> dict:
        """The part of this candidate the view digest commits to."""
        return {
            "label": self.label,
            "span_id": self.span_id,
            "namespace": self.namespace,
            "text_sha256": sha256_hex(self.text.encode("utf-8")),
            "heading_sha256": [sha256_hex(h.encode("utf-8")) for h in self.heading_path],
            "context_sha256": [sha256_hex(c.encode("utf-8")) for c in self.context],
        }


def _span_id_of(span: dict) -> str:
    return span.get("span_id") or span.get("visible_span_id") or ""


def _source_of(span: dict) -> str:
    return span.get("content_hash") or span.get("visible_compressor_view_hash") or ""


@dataclass(frozen=True)
class CandidateViewRecord:
    """Everything the selector was shown, and the only source the publication renders from."""

    namespace: str
    candidates: tuple[OfferedCandidate, ...]
    candidate_set: CandidateSet
    registry: dict
    tokenizer: Tokenizer
    prompt_bytes: bytes
    prompt_bundle_version: str
    renderer_grouping_version: str
    query_status: Mapping[str, str] = field(default_factory=dict)
    source_meta: Mapping[str, dict] = field(default_factory=dict)
    visible_views: Mapping[str, bytes] = field(default_factory=dict)
    snapshot_texts: Mapping[str, str] = field(default_factory=dict)

    # --- construction ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        spans: Sequence[dict],
        tokenizer: Tokenizer,
        namespace: str,
        topic: str,
        contract: str,
        token_budget: int,
        query_attempts: Sequence[tuple[str, str]],
        snapshot_texts: Optional[Mapping[str, str]] = None,
        visible_views: Optional[Mapping[str, bytes]] = None,
        source_meta: Optional[Mapping[str, dict]] = None,
        query_status: Optional[Mapping[str, str]] = None,
    ) -> "CandidateViewRecord":
        snapshot_texts = dict(snapshot_texts or {})
        visible_views = dict(visible_views or {})
        candidates: list[OfferedCandidate] = []

        for index, span in enumerate(spans, start=1):
            span_id = _span_id_of(span)
            got = span.get("namespace")
            if got != namespace:
                raise ViewConstructionError(
                    f"candidate {span_id[:12]} is in namespace {got!r} but this view offers "
                    f"{namespace!r}. The selector reads the whole offered set, so a foreign "
                    "candidate breaks the boundary even if it is never selected."
                )
            candidates.append(OfferedCandidate(
                label=f"E{index}",
                span_id=span_id,
                namespace=namespace,
                text=_resolve_body(span, snapshot_texts, visible_views),
                heading_path=_resolve_refs(span, "heading_refs", snapshot_texts),
                context=_resolve_refs(span, "context_refs", snapshot_texts),
            ))

        candidate_set = CandidateSet.build(
            [c.span_id for c in candidates],
            [qid for qid, _ in query_attempts],
            namespace=namespace,
        )
        prompt = render_selector_prompt(
            topic=topic,
            candidates=[(c.label, c.text, c.heading_path, c.context) for c in candidates],
            query_attempts=[(f"Q{j}", text) for j, (_, text) in enumerate(query_attempts, 1)],
            budget=token_budget,
            contract=contract,
        )
        return cls(
            namespace=namespace,
            candidates=tuple(candidates),
            candidate_set=candidate_set,
            registry={c.span_id: dict(s) for c, s in zip(candidates, spans)},
            tokenizer=tokenizer,
            prompt_bytes=prompt.encode("utf-8"),
            prompt_bundle_version=PROMPT_BUNDLE_VERSION,
            renderer_grouping_version=_renderer_version(),
            query_status=dict(query_status or {}),
            source_meta=dict(source_meta or {}),
            visible_views=visible_views,
            snapshot_texts=snapshot_texts,
        )

    # --- identity ----------------------------------------------------------------------

    @property
    def prompt_sha256(self) -> str:
        return sha256_hex(self.prompt_bytes)

    @property
    def view_sha256(self) -> str:
        """Digest over the bytes THIS record resolved, not over the mutable registry.

        Recomputing from the registry at publish time and comparing to a digest taken the same
        way proves nothing when both read the same mutable dicts: a ref repointed before the
        first digest was simply baked into both. The resolved candidates are immutable, so a
        later mutation cannot agree with them.
        """
        return sha256_hex(canonical_json({
            "namespace": self.namespace,
            "prompt_bundle_version": self.prompt_bundle_version,
            "renderer_grouping_version": self.renderer_grouping_version,
            "prompt_sha256": self.prompt_sha256,
            "candidates": [c.identity() for c in self.candidates],
        }))

    # --- rendering and costing ---------------------------------------------------------

    def text_for(self, span_id: str) -> str:
        return self._by_id()[span_id].text

    def label_for(self, span_id: str) -> str:
        return self._by_id()[span_id].label

    def _by_id(self) -> dict[str, OfferedCandidate]:
        return {c.span_id: c for c in self.candidates}

    def render(self, evidence) -> "object":
        """Render aggregated evidence from the record's own resolved bytes."""
        from .renderer import render

        by_id = self._by_id()
        return render(
            evidence,
            self.registry,
            source_text_for=lambda span: by_id[_span_id_of(span)].text,
            label_for=self.label_for,
            tokenizer=self.tokenizer,
            source_meta=self.source_meta,
            visible_views=self.visible_views,
            query_status=self.query_status,
            context_text_for=None,
            resolved_context_for=lambda span: by_id[_span_id_of(span)].context,
            resolved_headings_for=lambda span: by_id[_span_id_of(span)].heading_path,
        )

    def cost(self, evidence) -> int:
        return self.render(evidence).token_count

    def coster(self) -> Callable:
        """A Coster bound to this record, for the aggregators' budget search."""
        return lambda evidence, _registry: self.cost(evidence)

    def drift_errors(self, span_ids: Iterable[str]) -> list[str]:
        """Report any span whose registry entry no longer resolves to what the view resolved.

        The record renders from its own immutable candidates, so a post-construction mutation
        of the registry cannot change the published bytes. But it is still a bug -- something
        edited state the selector had already been shown -- and a silently ignored mutation is
        a defect that surfaces nowhere. Surfacing it here fails the batch closed instead.
        """
        by_id = self._by_id()
        errors: list[str] = []
        for span_id in span_ids:
            candidate = by_id.get(span_id)
            span = self.registry.get(span_id)
            if candidate is None or span is None:
                continue
            try:
                text = _resolve_body(span, self.snapshot_texts, self.visible_views)
                headings = _resolve_refs(span, "heading_refs", self.snapshot_texts)
                context = _resolve_refs(span, "context_refs", self.snapshot_texts)
            except ViewConstructionError as e:
                errors.append(f"span {span_id[:12]}: no longer resolvable ({e})")
                continue
            if (text, headings, context) != (candidate.text, candidate.heading_path,
                                             candidate.context):
                errors.append(
                    f"span {span_id[:12]}: the candidate view has drifted since the selector "
                    "saw it; the bytes offered to the model are not the bytes now recorded"
                )
        return errors


def _resolve_body(
    span: dict, snapshot_texts: Mapping[str, str], visible_views: Mapping[str, bytes]
) -> str:
    span_id = _span_id_of(span)[:12]
    namespace = span.get("namespace")
    try:
        if namespace == "RAW_SOURCE":
            return snapshot_texts[span["content_hash"]][span["char_start"]:span["char_end"]]
        if namespace == "VISIBLE_MESSAGE":
            view = visible_views[span["visible_compressor_view_hash"]]
            return view[span["byte_start"]:span["byte_end"]].decode("utf-8")
    except (KeyError, UnicodeDecodeError) as e:
        raise ViewConstructionError(
            f"candidate {span_id}: cannot resolve its bytes from frozen storage ({e!r}); a "
            "candidate whose body is not recoverable must never be offered"
        ) from e
    raise ViewConstructionError(f"candidate {span_id}: unknown namespace {namespace!r}")


def _resolve_refs(
    span: dict, field_name: str, snapshot_texts: Mapping[str, str]
) -> tuple[str, ...]:
    span_id = _span_id_of(span)[:12]
    kind = field_name.removesuffix("_refs")
    source = _source_of(span)
    out: list[str] = []
    for ref in span.get(field_name) or ():
        if ref["content_hash"] != source:
            raise ViewConstructionError(
                f"candidate {span_id}: {kind} must address the same source as its span, but "
                f"addresses {ref['content_hash'][:12]} while the span lives in {source[:12]}. "
                "Metadata from another source is not context; it is imported evidence."
            )
        text = snapshot_texts.get(ref["content_hash"])
        if text is None:
            raise ViewConstructionError(
                f"candidate {span_id}: {kind} snapshot {ref['content_hash'][:12]} unavailable"
            )
        start, end = ref["char_start"], ref["char_end"]
        if not (0 <= start <= end <= len(text)):
            raise ViewConstructionError(
                f"candidate {span_id}: {kind} offsets [{start},{end}] out of bounds"
            )
        exact = text[start:end]
        if sha256_hex(exact.encode("utf-8")) != ref["text_sha256"]:
            raise ViewConstructionError(
                f"candidate {span_id}: {kind} hash mismatch -- the text it addresses is not "
                "the text it recorded"
            )
        out.append(exact)
    return tuple(out)


def _renderer_version() -> str:
    from .renderer import RENDERER_GROUPING_VERSION

    return RENDERER_GROUPING_VERSION


def guard_publication(fn: Callable[[], object]) -> list[str]:
    """Run ``fn``, converting an ordinary failure into recorded errors -- but never a
    cancellation.

    The adapter's whole-batch P0 fallback needs a decision object rather than an exception, so
    structural and rendering failures come back as error strings. ``CancelledError`` is
    deliberately excluded: a cooperative cancellation is the run being torn down, not P1
    failing, and turning it into "P1 failed, fall back to P0" would fabricate a P0 result for
    work that was abandoned -- and charge the arm for it.
    """
    try:
        fn()
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001 - deliberately broad; it becomes a recorded failure
        return [f"{type(e).__name__}: {e}"]
    return []
