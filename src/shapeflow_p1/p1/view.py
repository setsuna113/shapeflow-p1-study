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
from .handle_codec import (
    MAX_PUBLICATION_HANDLE_TOKENS,
    HandleDomainError,
    encode as encode_publication_handle,
    structural_ordinal,
)
from .prompts import PROMPT_BUNDLE_VERSION, render_selector_prompt

__all__ = [
    "CandidateViewRecord",
    "OfferedCandidate",
    "ViewConstructionError",
    "guard_publication",
    "compact_publication_handle",
    "MAX_PUBLICATION_HANDLE_TOKENS",
]


class ViewConstructionError(ValueError):
    """The offered candidate set cannot be built as declared.

    Always raised *before* a selector call. Everything it catches -- a foreign namespace, a
    cross-source context ref, an unresolvable body -- would otherwise become a property of what
    the model already read, at which point no downstream check can undo it.
    """


def compact_publication_handle(
    assistant_turn_index: int,
    toolset_ordinal: int,
    evidence_ordinal: int,
    *,
    researcher_coordinate: tuple[int, int] | None = None,
) -> str:
    """A collision-free handle inside one cell's nested researcher trajectories.

    The handle is an encoding of structural ordinals, not a truncated hash.  Its scope is one
    ConductResearch child (when present) -> assistant turn -> ordered sibling search-result set
    -> evidence ordinal in that atomic publication batch.  The exact tool-call id remains in
    HCheckpoint.researcher_coordinate and detects structural-slot reuse.

    The encoding itself lives in :mod:`shapeflow_p1.p1.handle_codec`, which producer, validator,
    preflight and evaluator all share.  This wrapper only folds the optional researcher
    coordinate into the codec's four-coordinate form.
    """
    return encode_publication_handle(
        structural_ordinal(researcher_coordinate),
        assistant_turn_index,
        toolset_ordinal,
        evidence_ordinal,
    )


@dataclass(frozen=True)
class OfferedCandidate:
    """One candidate exactly as the selector will see it, with its bytes already resolved."""

    label: str
    publication_handle: str
    publication_handle_tokens: int
    span_id: str
    namespace: str
    text: str
    origin_kind: str = ""
    message_role: str = ""
    heading_path: tuple[str, ...] = ()
    context: tuple[str, ...] = ()

    def identity(self) -> dict:
        """The part of this candidate the view digest commits to."""
        return {
            "label": self.label,
            "span_id": self.span_id,
            "namespace": self.namespace,
            "origin_kind": self.origin_kind,
            "message_role": self.message_role,
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
    source_meta_sha256: str = ""
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
        publication_scope: tuple[int, ...] | None = None,
        publication_ordinals: Mapping[str, int] | None = None,
    ) -> "CandidateViewRecord":
        snapshot_texts = dict(snapshot_texts or {})
        visible_views = dict(visible_views or {})
        source_meta = {
            str(span_id): dict(meta)
            for span_id, meta in dict(source_meta or {}).items()
        }
        candidates: list[OfferedCandidate] = []
        spans = list(spans)

        if namespace == "RAW_SOURCE":
            scope = publication_scope or (0, 0)
            if len(scope) == 2:
                researcher_coordinate = None
                turn_index, toolset_ordinal = scope
            elif len(scope) == 4:
                researcher_iteration, child_ordinal, turn_index, toolset_ordinal = scope
                researcher_coordinate = (researcher_iteration, child_ordinal)
            else:
                raise ViewConstructionError(
                    "H publication scope must be (turn, toolset) or "
                    "(researcher iteration, child ordinal, turn, toolset)"
                )
            # Validate coordinates even for an empty view: an out-of-domain scope must fail
            # here, before a selector call, not on whichever span first happens to use it.
            #
            # HandleDomainError is deliberately NOT wrapped in ViewConstructionError. A view
            # that cannot be built is a treatment outcome and falls back to P0; a publication
            # domain too small to name this batch is a defect in frozen protocol, and laundering
            # it into a P0 fallback is precisely how an entirely inert P1 came to look like a
            # completed run in all 146 canary cells.
            structural = structural_ordinal(researcher_coordinate)
            encode_publication_handle(structural, turn_index, toolset_ordinal, 1)
            if publication_ordinals is None:
                publication_ordinals = {
                    _span_id_of(span): index for index, span in enumerate(spans, 1)
                }
            else:
                publication_ordinals = {
                    str(span_id): ordinal
                    for span_id, ordinal in publication_ordinals.items()
                }
            ordinal_values = list(publication_ordinals.values())
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in ordinal_values
            ):
                raise ViewConstructionError(
                    "publication ordinals must be positive integers"
                )
            if len(ordinal_values) != len(set(ordinal_values)):
                raise ViewConstructionError(
                    "publication ordinals collide inside one atomic H batch"
                )
        elif publication_scope is not None or publication_ordinals is not None:
            raise ViewConstructionError(
                "publication ordinals belong only to RAW_SOURCE/H views"
            )

        for index, span in enumerate(spans, start=1):
            span_id = _span_id_of(span)
            got = span.get("namespace")
            if got != namespace:
                raise ViewConstructionError(
                    f"candidate {span_id[:12]} is in namespace {got!r} but this view offers "
                    f"{namespace!r}. The selector reads the whole offered set, so a foreign "
                    "candidate breaks the boundary even if it is never selected."
                )
            origin_kind = str(span.get("kind") or "")
            message_role = str(span.get("message_role") or "")
            if namespace == "VISIBLE_MESSAGE":
                allowed_origins = {
                    "TOOL_EVIDENCE",
                    "TOOL_UNATTRIBUTED_CONTEXT",
                    "MODEL_DERIVED_CONTEXT",
                    "USER_CONTEXT",
                }
                if origin_kind not in allowed_origins:
                    raise ViewConstructionError(
                        f"candidate {span_id[:12]} has unknown visible-message origin "
                        f"{origin_kind!r}; without a frozen origin it cannot be classified as "
                        "citable evidence or non-citable context"
                    )
                allowed_roles = {
                    "TOOL_EVIDENCE": {"tool"},
                    "TOOL_UNATTRIBUTED_CONTEXT": {"tool"},
                    "MODEL_DERIVED_CONTEXT": {"ai"},
                    "USER_CONTEXT": {"human", "system"},
                }
                if message_role not in allowed_roles[origin_kind]:
                    raise ViewConstructionError(
                        f"candidate {span_id[:12]} declares {origin_kind} but has message role "
                        f"{message_role!r}; visible origin must be derived from the frozen role"
                    )
                if (
                    origin_kind != "TOOL_EVIDENCE"
                    and span.get("source_occurrence_ids")
                ):
                    raise ViewConstructionError(
                        f"candidate {span_id[:12]} is non-citable {origin_kind} but carries "
                        "source occurrences"
                    )
            if namespace == "RAW_SOURCE":
                try:
                    publication_ordinal = publication_ordinals[span_id]
                except KeyError as exc:
                    raise ViewConstructionError(
                        f"candidate {span_id[:12]} has no publication ordinal in its atomic "
                        "H batch"
                    ) from exc
                # Out-of-domain raises HandleDomainError, which is not a ViewConstructionError
                # and so does not fall back to P0. See the note at the scope check above.
                publication_handle = compact_publication_handle(
                    turn_index,
                    toolset_ordinal,
                    publication_ordinal,
                    researcher_coordinate=researcher_coordinate,
                )
            else:
                publication_handle = f"E{index}"
            publication_handle_tokens = tokenizer.count(
                f"[{publication_handle}]"
            )
            if (
                namespace == "RAW_SOURCE"
                and publication_handle_tokens > MAX_PUBLICATION_HANDLE_TOKENS
            ):
                # Unreachable if the codec's exhaustive proof holds: every handle the frozen
                # domain can produce was enumerated against this tokenizer. It stays because the
                # proof is over one tokenizer file, and the run host is the only place that
                # knows which file is actually loaded.
                raise ViewConstructionError(
                    f"publication handle {publication_handle!r} costs "
                    f"{publication_handle_tokens} exact model tokens, exceeding the frozen cap "
                    f"{MAX_PUBLICATION_HANDLE_TOKENS}. The frozen handle domain was proven to "
                    "fit this cap, so either the tokenizer or the codec radices moved."
                )
            candidates.append(OfferedCandidate(
                label=f"E{index}",
                # E-labels are prompt-local compression.  H publication uses a compact
                # trajectory-scoped ordinal whose exact tokenizer cost is gated above.
                publication_handle=publication_handle,
                publication_handle_tokens=publication_handle_tokens,
                span_id=span_id,
                namespace=namespace,
                text=_resolve_body(span, snapshot_texts, visible_views),
                origin_kind=origin_kind,
                message_role=message_role,
                heading_path=_resolve_refs(span, "heading_refs", snapshot_texts),
                context=_resolve_refs(span, "context_refs", snapshot_texts),
            ))

        _validate_source_meta(
            spans=spans,
            namespace=namespace,
            source_meta=source_meta,
            visible_views=visible_views,
        )
        source_meta_digest = _source_meta_sha256(source_meta)
        candidate_set = CandidateSet.build(
            [c.span_id for c in candidates],
            [qid for qid, _ in query_attempts],
            namespace=namespace,
        )
        prompt = render_selector_prompt(
            topic=topic,
            candidates=[
                (
                    c.label,
                    c.text,
                    c.heading_path,
                    c.context,
                    c.origin_kind,
                    c.message_role,
                )
                for c in candidates
            ],
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
            source_meta=source_meta,
            source_meta_sha256=source_meta_digest,
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
            "source_meta_sha256": self.source_meta_sha256,
            "candidates": [c.identity() for c in self.candidates],
        }))

    # --- rendering and costing ---------------------------------------------------------

    def text_for(self, span_id: str) -> str:
        return self._by_id()[span_id].text

    def label_for(self, span_id: str) -> str:
        return self._by_id()[span_id].label

    def publication_handle_for(self, span_id: str) -> str:
        return self._by_id()[span_id].publication_handle

    @property
    def publication_handle_map(self) -> tuple[tuple[str, str], ...]:
        """The auditable publication-handle -> stable-span mapping."""
        return tuple(
            (candidate.publication_handle, candidate.span_id)
            for candidate in self.candidates
        )

    @property
    def publication_handle_token_counts(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (candidate.publication_handle, candidate.publication_handle_tokens)
            for candidate in self.candidates
        )

    @property
    def publication_map_sha256(self) -> str:
        """Bind publication identity separately from the selector-visible candidate digest."""
        return sha256_hex(canonical_json({
            "handle_to_span": [list(item) for item in self.publication_handle_map],
            "handle_token_counts": [
                list(item) for item in self.publication_handle_token_counts
            ],
        }))

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
            label_for=self.publication_handle_for,
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
        if _source_meta_sha256(self.source_meta) != self.source_meta_sha256:
            errors.append(
                "source affordance metadata has drifted since the selector call"
            )
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
            origin = str(span.get("kind") or "")
            message_role = str(span.get("message_role") or "")
            if (text, headings, context, origin, message_role) != (
                candidate.text,
                candidate.heading_path,
                candidate.context,
                candidate.origin_kind,
                candidate.message_role,
            ):
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


_VISIBLE_SOURCE_META_FIELDS = frozenset({
    "binding_version",
    "title",
    "url",
    "message_id",
    "visible_compressor_view_hash",
    "byte_start",
    "byte_end",
    "title_byte_start",
    "title_byte_end",
    "url_byte_start",
    "url_byte_end",
    "source_occurrence_ids",
    "source_label",
})


def _source_meta_sha256(source_meta: Mapping[str, dict]) -> str:
    return sha256_hex(canonical_json({
        str(span_id): dict(meta)
        for span_id, meta in sorted(source_meta.items())
    }))


def _validate_source_meta(
    *,
    spans: Sequence[dict],
    namespace: str,
    source_meta: Mapping[str, dict],
    visible_views: Mapping[str, bytes],
) -> None:
    """For C_VISIBLE, accept only exact byte-range-bound title/URL metadata."""

    if namespace != "VISIBLE_MESSAGE":
        return
    by_id = {_span_id_of(span): span for span in spans}
    unknown = sorted(set(source_meta) - set(by_id))
    if unknown:
        raise ViewConstructionError(
            f"source metadata names candidate {unknown[0][:12]} outside the offered view"
        )
    for span_id, meta in source_meta.items():
        span = by_id[span_id]
        if span.get("kind") != "TOOL_EVIDENCE":
            raise ViewConstructionError(
                f"candidate {span_id[:12]} is non-citable context but carries source metadata"
            )
        if set(meta) != _VISIBLE_SOURCE_META_FIELDS:
            missing = sorted(_VISIBLE_SOURCE_META_FIELDS - set(meta))
            extra = sorted(set(meta) - _VISIBLE_SOURCE_META_FIELDS)
            raise ViewConstructionError(
                f"candidate {span_id[:12]} source metadata is not the closed "
                f"byte-range binding (missing={missing}, extra={extra})"
            )
        if meta.get("binding_version") != "visible_source_byte_range_v1":
            raise ViewConstructionError(
                f"candidate {span_id[:12]} source metadata has unknown binding version"
            )
        view_hash = str(meta.get("visible_compressor_view_hash") or "")
        if view_hash != str(span.get("visible_compressor_view_hash") or ""):
            raise ViewConstructionError(
                f"candidate {span_id[:12]} source metadata addresses another compressor view"
            )
        if str(meta.get("message_id") or "") != str(span.get("message_id") or ""):
            raise ViewConstructionError(
                f"candidate {span_id[:12]} source metadata addresses another message"
            )
        view_bytes = visible_views.get(view_hash)
        if view_bytes is None:
            raise ViewConstructionError(
                f"candidate {span_id[:12]} source metadata has no frozen compressor view"
            )
        coordinate_names = (
            "byte_start",
            "byte_end",
            "title_byte_start",
            "title_byte_end",
            "url_byte_start",
            "url_byte_end",
        )
        if any(
            isinstance(meta.get(name), bool) or not isinstance(meta.get(name), int)
            for name in coordinate_names
        ):
            raise ViewConstructionError(
                f"candidate {span_id[:12]} source metadata has non-integer byte coordinates"
            )
        block_start = int(meta["byte_start"])
        block_end = int(meta["byte_end"])
        span_start = int(span["byte_start"])
        span_end = int(span["byte_end"])
        if not (
            0 <= block_start <= span_start <= span_end <= block_end <= len(view_bytes)
        ):
            raise ViewConstructionError(
                f"candidate {span_id[:12]} is not wholly contained in its declared source range"
            )
        for meta_field in ("title", "url"):
            field_start = int(meta[f"{meta_field}_byte_start"])
            field_end = int(meta[f"{meta_field}_byte_end"])
            if not (
                block_start <= field_start <= field_end <= block_end
            ):
                raise ViewConstructionError(
                    f"candidate {span_id[:12]} {meta_field} byte range escapes its source block"
                )
            expected = str(meta.get(meta_field) or "")
            actual = view_bytes[field_start:field_end].decode("utf-8", errors="strict")
            if expected == "(untitled source)" and field_start == field_end:
                # The empty title is explicit; the URL remains capture-time bytes. The
                # display-only placeholder does not claim to have appeared in the prompt.
                continue
            if actual != expected:
                raise ViewConstructionError(
                    f"candidate {span_id[:12]} {meta_field} does not reconstruct from its "
                    "capture-time byte range"
                )
        occurrences = meta.get("source_occurrence_ids")
        if (
            not isinstance(occurrences, list)
            or not occurrences
            or any(not isinstance(value, str) or not value for value in occurrences)
        ):
            raise ViewConstructionError(
                f"candidate {span_id[:12]} source metadata has no occurrence lineage"
            )
        span_occurrences = {
            str(value) for value in (span.get("source_occurrence_ids") or ())
        }
        if not set(occurrences).issubset(span_occurrences):
            raise ViewConstructionError(
                f"candidate {span_id[:12]} source metadata occurrence lineage disagrees "
                "with the visible span"
            )


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
