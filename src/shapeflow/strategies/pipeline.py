"""The selection pipeline every P1 arm shares.

One implementation, because the arms must differ *only* in the dimensions the study varies --
chunker, scope, contract, aggregator, namespace. If each arm assembled its own pipeline, a
difference in how two of them budgeted or rendered would be indistinguishable from the
mechanism under test.

The order is fixed and each step's output is the next one's only input:

    frozen bytes -> chunker -> candidate view -> selector -> parse -> aggregate -> preflight

Preflight owns the render, so what is verified is what is published. A failure anywhere returns
a :class:`SelectionFailure` rather than raising, because the caller's job is to fall the whole
batch back to P0, and a decision object is what that needs.

Work accounting is not optional and not conditional on success. A selector call that produced
an unusable selection still spent its tokens; recording them only on the happy path would make
P1 look cheaper exactly when it went wrong.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from ..evidence.chunkers import Tokenizer, fixed_token_v1, markdown_structure_v1, paragraph_sentence_v1
from ..evidence.model_tokenizer import tokenizer_sha256
from ..evidence.identity import build_evidence_span, build_visible_message_span
from ..p1.aggregators import (
    budget_pack_v1,
    coverage_budget_v1,
    global_rerank_v1,
    stable_union_v1,
)
from ..p1.contracts import SelectionContractError, parse_selection
from ..p1.preflight import PreflightConfig, preflight
from ..p1.prompt_pack import PromptPackUnsatisfiable, prompt_pack_v1
from ..p1.view import CandidateViewRecord, ViewConstructionError

__all__ = [
    "SelectionOutcome",
    "SelectionFailure",
    "WorkRecord",
    "CHUNKERS",
    "AGGREGATORS",
    "run_selection",
]

CHUNKERS: dict[str, Callable] = {
    "fixed_token_v1": lambda text, tok, **kw: fixed_token_v1(text, tokenizer=tok, **kw),
    "paragraph_sentence_v1": lambda text, tok, **kw: paragraph_sentence_v1(text, tokenizer=tok, **kw),
    "markdown_structure_v1": lambda text, tok, **kw: markdown_structure_v1(text, tokenizer=tok, **kw),
}

#: Only names with an implementation behind them. `mmr_stable_union` was an alias for
#: stable_union_v1 with no MMR anywhere, and `token_matched` was in this set with no branch
#: in _aggregate, so it would have raised had anything reached it. A registry that lists a
#: variant it cannot run reports a null result for a thing it never tried.
AGGREGATORS = {"stable_union_v1", "coverage_budget_v1", "budget_pack_v1", "global_rerank_v1"}


@dataclass
class WorkRecord:
    """Everything an arm spent, whether or not it produced output."""

    selector_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cpu_seconds: float = 0.0
    retries: int = 0

    def add(self, other: "WorkRecord") -> None:
        self.selector_calls += other.selector_calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.cpu_seconds += other.cpu_seconds
        self.retries += other.retries


@dataclass(frozen=True)
class SelectionFailure:
    reason: str
    detail: str = ""


@dataclass
class SelectionOutcome:
    """What one selector call produced, plus what it cost either way."""

    text: Optional[str] = None
    work: WorkRecord = field(default_factory=WorkRecord)
    failure: Optional[SelectionFailure] = None
    view_sha256: str = ""
    selected: int = 0
    offered: int = 0
    dropped_for_budget: int = 0
    normalization: Optional[dict] = None
    # Do not infer this from ``work.selector_calls``.  A provider can fail before returning
    # usage, but that is still a selector attempt and must remain in the strict-valid
    # denominator (as an explicit failure-before-parse), not disappear as "no call".
    selector_attempted: bool = False
    # Direct-node estimands need identities, not a count inferred from rendered prose.  Keep the
    # three sets distinct: offered -> model-selected -> aggregator-published.
    offered_span_ids: tuple[str, ...] = ()
    # Token materialization metrics are measured on the exact immutable candidate bytes and
    # on the exact renderer output that preflight accepted.  Counts inferred later from prose
    # cannot distinguish selected evidence from headers/metadata and previously made the
    # canary's same-budget check read a permanently absent field as zero.
    offered_span_token_counts: tuple[tuple[str, int], ...] = ()
    publication_handle_map: tuple[tuple[str, str], ...] = ()
    publication_handle_token_counts: tuple[tuple[str, int], ...] = ()
    publication_map_sha256: str = ""
    # C_VISIBLE offers both source evidence and non-citable model/user context.  Keep all three
    # totals so work/materialization uses the complete input while evidence precision and recall
    # never promote context into their denominator.  For H, material == evidence and context=0.
    offered_material_tokens: int = 0
    offered_evidence_tokens: int = 0
    offered_context_tokens: int = 0
    staged_rendered_tokens: int = 0
    published_rendered_tokens: int = 0
    # Raw-source occurrence identity is the treatment-independent H denominator.  It must not
    # be reconstructed from candidate spans: a broken chunker that emits no span for a source
    # would otherwise make the source and every fact it contained disappear from its own score.
    offered_source_occurrence_ids: tuple[str, ...] = ()
    selected_span_ids: tuple[str, ...] = ()
    # ``staged`` is the deterministic aggregator output before publish-time preflight and
    # whole-batch acceptance.  It is not called published: a failed preflight or a sibling
    # failure causes the entire staged batch to be discarded.
    staged_span_ids: tuple[str, ...] = ()
    published_span_ids: tuple[str, ...] = ()
    # Preserve typed semantics for contradiction and gap estimands.  IDs alone can say whether
    # a span survived, but not whether its support/contradict role or explicit "looked but
    # unresolved" declaration survived with it.
    selected_relations: tuple[tuple[str, str, Optional[str]], ...] = ()
    staged_relations: tuple[tuple[str, str, Optional[str]], ...] = ()
    published_relations: tuple[tuple[str, str, Optional[str]], ...] = ()
    selected_gaps: tuple[tuple[str, tuple[str, ...]], ...] = ()
    staged_gaps: tuple[tuple[str, tuple[str, ...]], ...] = ()
    published_gaps: tuple[tuple[str, tuple[str, ...]], ...] = ()
    # Query-attempt provenance is the denominator for explicit negative/gap retention. Gaps
    # alone expose only what the model chose to mention, which would condition recall on its
    # own output and make a silent omission invisible.
    offered_query_attempt_ids: tuple[str, ...] = ()
    selected_query_attempt_ids: tuple[str, ...] = ()
    staged_query_attempt_ids: tuple[str, ...] = ()
    published_query_attempt_ids: tuple[str, ...] = ()
    contract: str = ""
    aggregation: str = ""
    # Span IDs are chunker-specific. Without this field the evaluator cannot select the
    # matching cross-chunker support index and must (correctly) mark direct recall unavailable.
    chunker: str = ""
    tokenizer_sha256: str = ""
    stage: str = "single"       # single | local | global
    checkpoint_hash: str = ""
    #: The source occurrences actually published, as distinct from those offered. Source
    #: coverage, evidence retention and hard-negative interference are all ratios over these two
    #: sets, and a span id cannot be decoded back to its source -- it is a digest. Without this
    #: the whole judge-free endpoint family is uncomputable from a trial record.
    published_source_occurrence_ids: tuple[str, ...] = ()
    #: What the prompt-admission stage dropped to fit the window, or None when no admission ran.
    #: Carried because CPU-FULL minus CPU-PROMPTVIEW is the price of that pruning, and pricing it
    #: needs the identities of the removed spans rather than a count.
    prompt_admission: Optional[dict] = None
    prompt_admission_dropped_span_ids: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.failure is None and self.text is not None


def _aggregate(name: str, selection, registry, *, token_budget: int, coster):
    if name == "stable_union_v1":
        return stable_union_v1(selection, registry)
    if name == "coverage_budget_v1":
        return coverage_budget_v1(selection, registry, token_budget=token_budget,
                                  coster=coster, min_sources=1)
    if name == "budget_pack_v1":
        # The selector's emitted order is its ranking, exactly as `global_rerank_v1` reads it.
        # Passing it here is what makes a ranking-only selector meaningful: without it every
        # candidate selector would be packed in document order and the shootout would compare
        # nothing but the packer.
        return budget_pack_v1(
            selection, registry, token_budget=token_budget, coster=coster, min_sources=1,
            _selection_rank={
                span_id: index
                for index, span_id in enumerate(dict.fromkeys(selection.selected_span_ids))
            },
        )
    if name == "global_rerank_v1":
        return global_rerank_v1(selection, registry, token_budget=token_budget, coster=coster)
    raise ValueError(f"unknown aggregator {name!r}")


async def run_selection(
    *,
    spans: Sequence[dict],
    namespace: str,
    snapshot_texts: dict[str, str],
    visible_views: dict[str, bytes],
    topic: str,
    contract: str,
    aggregation: str,
    chunker: str,
    token_budget: int,
    tokenizer: Tokenizer,
    query_attempts: Sequence[tuple[str, str]],
    selector,
    task_ctx,
    known_occurrence_ids: set[str],
    query_status: Optional[dict[str, str]] = None,
    source_meta: Optional[dict] = None,
    bridge_token_cap_each: Optional[int] = None,
    bridge_token_cap_total: Optional[int] = None,
    checkpoint_hash: str = "",
    stage: str = "single",
    publication_scope: tuple[int, ...] | None = None,
    publication_ordinals: dict[str, int] | None = None,
    prompt_admission: str = "none",
    prompt_budget: int = 0,
    prompt_window_ceiling: int = 0,
) -> SelectionOutcome:
    """Offer, select, publish -- or fail with everything it cost recorded."""
    work = WorkRecord()
    tokenizer_digest = tokenizer_sha256(tokenizer)
    offered_source_occurrence_ids = tuple(sorted(map(str, known_occurrence_ids)))
    offered_query_attempt_ids = tuple(dict.fromkeys(
        str(attempt_id) for attempt_id, _query in query_attempts
    ))
    if not spans:
        return SelectionOutcome(
            text="", work=work, offered=0, contract=contract, aggregation=aggregation,
            chunker=chunker, tokenizer_sha256=tokenizer_digest,
            stage=stage, checkpoint_hash=checkpoint_hash,
            offered_query_attempt_ids=offered_query_attempt_ids,
            offered_source_occurrence_ids=offered_source_occurrence_ids,
        )

    # Prompt admission, before the view exists. Two budgets bind at this boundary and they are
    # enforced by two different objects: this one decides what the selector may *see* (the
    # engine's context window), `budget_pack_v1` decides what survives *rendering* (512 tokens).
    # Fusing them would put the model's ranking inside the decision about what to show the model.
    prompt_pack: Optional[object] = None
    if prompt_admission and prompt_admission != "none":
        if prompt_admission != "prompt_pack_v1":
            raise ValueError(f"unknown prompt admission {prompt_admission!r}")
        texts = [
            snapshot_texts.get(str(s.get("content_hash", "")), "")[
                int(s.get("char_start", 0)):int(s.get("char_end", 0))]
            for s in spans
        ]
        try:
            prompt_pack = prompt_pack_v1(
                spans=list(spans), texts=texts,
                token_counts=[tokenizer.count(t) for t in texts],
                budget=prompt_budget, topic=topic,
            )
        except PromptPackUnsatisfiable as e:
            # A real outcome for a batch of very long pages, not an error: no prompt covering
            # this batch fits the window. Recorded so it lands in the all-offered denominator.
            return SelectionOutcome(
                work=work, failure=SelectionFailure("PROMPT_INFEASIBLE", str(e)),
                contract=contract, aggregation=aggregation, chunker=chunker, stage=stage,
                tokenizer_sha256=tokenizer_digest, checkpoint_hash=checkpoint_hash,
                offered_query_attempt_ids=offered_query_attempt_ids,
                offered_source_occurrence_ids=offered_source_occurrence_ids,
            )
        admitted = set(prompt_pack.span_ids)
        spans = [s for s in spans if str(s.get("span_id", "")) in admitted]

    try:
        view = CandidateViewRecord.build(
            spans=list(spans), tokenizer=tokenizer, namespace=namespace, topic=topic,
            contract=contract, token_budget=token_budget, query_attempts=list(query_attempts),
            snapshot_texts=snapshot_texts, visible_views=visible_views,
            source_meta=source_meta, query_status=query_status,
            publication_scope=publication_scope,
            publication_ordinals=publication_ordinals,
        )
    except ViewConstructionError as e:
        return SelectionOutcome(
            work=work, failure=SelectionFailure("VIEW_CONSTRUCTION", str(e)),
            contract=contract, aggregation=aggregation, chunker=chunker, stage=stage,
            tokenizer_sha256=tokenizer_digest,
            checkpoint_hash=checkpoint_hash,
            offered_query_attempt_ids=offered_query_attempt_ids,
            offered_source_occurrence_ids=offered_source_occurrence_ids,
        )

    # The diagnostic arm: no admission stage, so the prompt may simply not fit. Refusing here
    # rather than dispatching keeps the failure attributable -- the engine would reject the
    # request anyway, but as a provider error indistinguishable from an outage.
    if prompt_window_ceiling and prompt_admission in ("", "none"):
        prompt_tokens = tokenizer.count(view.prompt_bytes.decode("utf-8"))
        if prompt_tokens > prompt_window_ceiling:
            return SelectionOutcome(
                work=work, view_sha256=view.view_sha256,
                failure=SelectionFailure(
                    "PROMPT_INFEASIBLE",
                    f"whole-batch prompt is {prompt_tokens} tokens against a "
                    f"{prompt_window_ceiling}-token ceiling"),
                contract=contract, aggregation=aggregation, chunker=chunker, stage=stage,
                tokenizer_sha256=tokenizer_digest, checkpoint_hash=checkpoint_hash,
                offered=len(view.candidates),
                offered_span_ids=tuple(c.span_id for c in view.candidates),
                offered_query_attempt_ids=offered_query_attempt_ids,
                offered_source_occurrence_ids=offered_source_occurrence_ids,
            )

    # span id -> the source occurrence it came from. Built from the offered spans, because a
    # span id is a digest over content hash, offsets and text and cannot be decoded back.
    source_of_span = {
        str(s.get("span_id", "")): str((s.get("source_occurrence_ids") or [""])[0])
        for s in spans
    }

    offered_span_ids = tuple(c.span_id for c in view.candidates)
    publication_handle_map = view.publication_handle_map
    publication_handle_token_counts = view.publication_handle_token_counts
    publication_map_sha256 = view.publication_map_sha256
    offered_span_token_counts = tuple(
        (candidate.span_id, tokenizer.count(candidate.text))
        for candidate in view.candidates
    )
    offered_material_tokens = sum(
        count for _span_id, count in offered_span_token_counts
    )
    non_citable_ids = {
        candidate.span_id for candidate in view.candidates
        if candidate.origin_kind in {
            "TOOL_UNATTRIBUTED_CONTEXT",
            "MODEL_DERIVED_CONTEXT",
            "USER_CONTEXT",
        }
    }
    offered_context_tokens = sum(
        count for span_id, count in offered_span_token_counts
        if span_id in non_citable_ids
    )
    offered_evidence_tokens = offered_material_tokens - offered_context_tokens

    try:
        raw, call_work = await selector.select(task_ctx=task_ctx, view=view)
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001
        # A response can be unusable only *after* the engine decoded it. SelectorModelCall
        # attaches provider usage to those exceptions; keep it in the local direct-node record
        # even though the provider ledger remains authoritative.
        usage = getattr(e, "usage", None)
        usage = usage if isinstance(usage, dict) else {}
        work.add(WorkRecord(
            selector_calls=1,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            retries=int(usage.get("retries", 0) or 0),
        ))
        return SelectionOutcome(
            work=work, view_sha256=view.view_sha256,
            selector_attempted=True,
            normalization={
                "raw_count": 0,
                "unique_count": 0,
                "duplicate_count": 0,
                "semantic_conflict_count": 0,
                "rejected_reason": "failure_before_parse",
            },
            failure=SelectionFailure("SELECTOR_ERROR", f"{type(e).__name__}: {e}"),
            offered=len(view.candidates), offered_span_ids=offered_span_ids,
            offered_span_token_counts=offered_span_token_counts,
            publication_handle_map=publication_handle_map,
            publication_handle_token_counts=publication_handle_token_counts,
            publication_map_sha256=publication_map_sha256,
            offered_material_tokens=offered_material_tokens,
            offered_evidence_tokens=offered_evidence_tokens,
            offered_context_tokens=offered_context_tokens,
            contract=contract, aggregation=aggregation, stage=stage,
            chunker=chunker, tokenizer_sha256=tokenizer_digest,
            checkpoint_hash=checkpoint_hash,
            offered_query_attempt_ids=offered_query_attempt_ids,
            offered_source_occurrence_ids=offered_source_occurrence_ids,
        )
    work.add(call_work)

    try:
        selection = parse_selection(
            raw, view.candidate_set, expected_contract=contract
        )
    except SelectionContractError as e:
        # The tokens were still spent. Recording them only on success would make P1 look
        # cheaper exactly when it failed.
        return SelectionOutcome(
            work=work, view_sha256=view.view_sha256,
            selector_attempted=True,
            normalization=getattr(e.normalization, "__dict__", None),
            failure=SelectionFailure("CONTRACT", str(e)),
            offered=len(view.candidates), offered_span_ids=offered_span_ids,
            offered_span_token_counts=offered_span_token_counts,
            publication_handle_map=publication_handle_map,
            publication_handle_token_counts=publication_handle_token_counts,
            publication_map_sha256=publication_map_sha256,
            offered_material_tokens=offered_material_tokens,
            offered_evidence_tokens=offered_evidence_tokens,
            offered_context_tokens=offered_context_tokens,
            contract=contract, aggregation=aggregation, stage=stage,
            chunker=chunker, tokenizer_sha256=tokenizer_digest,
            checkpoint_hash=checkpoint_hash,
            offered_query_attempt_ids=offered_query_attempt_ids,
            offered_source_occurrence_ids=offered_source_occurrence_ids,
        )

    selected_relations = tuple(
        (item.span_id, facet_id, role)
        for item in selection.items
        for facet_id, role in item.relations
    )
    selected_gaps = tuple(
        (gap.facet_id, tuple(gap.query_attempt_ids)) for gap in selection.gaps
    )
    selected_query_attempt_ids = tuple(dict.fromkeys(
        query_id for _facet_id, query_ids in selected_gaps for query_id in query_ids
    ))

    try:
        aggregated = _aggregate(aggregation, selection, view.registry,
                                token_budget=token_budget, coster=view.coster())
    except Exception as e:  # noqa: BLE001
        return SelectionOutcome(work=work, view_sha256=view.view_sha256,
                                selector_attempted=True,
                                normalization=selection.normalization.__dict__,
                                failure=SelectionFailure("AGGREGATION", str(e)),
                                offered=len(view.candidates),
                                offered_span_ids=offered_span_ids,
                                offered_span_token_counts=offered_span_token_counts,
                                publication_handle_map=publication_handle_map,
                                publication_handle_token_counts=
                                    publication_handle_token_counts,
                                publication_map_sha256=publication_map_sha256,
                                offered_material_tokens=offered_material_tokens,
                                offered_evidence_tokens=offered_evidence_tokens,
                                offered_context_tokens=offered_context_tokens,
                                offered_source_occurrence_ids=offered_source_occurrence_ids,
                                selected_span_ids=selection.selected_span_ids,
                                selected_relations=selected_relations,
                                selected_gaps=selected_gaps,
                                offered_query_attempt_ids=offered_query_attempt_ids,
                                selected_query_attempt_ids=selected_query_attempt_ids,
                                contract=contract, aggregation=aggregation, chunker=chunker,
                                tokenizer_sha256=tokenizer_digest,
                                stage=stage,
                                checkpoint_hash=checkpoint_hash)

    staged_span_ids = tuple(item.span_id for item in aggregated.items)
    staged_relations = tuple(
        (item.span_id, facet_id, role)
        for item in aggregated.items
        for facet_id, role in item.relations
    )
    staged_gaps = tuple(
        (gap.facet_id, tuple(gap.query_attempt_ids)) for gap in aggregated.gaps
    )
    staged_query_attempt_ids = tuple(dict.fromkeys(
        query_id for _facet_id, query_ids in staged_gaps for query_id in query_ids
    ))
    result = preflight(
        selection=selection, aggregated=aggregated, view=view,
        known_occurrence_ids=known_occurrence_ids,
        config=PreflightConfig(selected_token_budget=token_budget,
                               bridge_token_cap_each=bridge_token_cap_each,
                               bridge_token_cap_total=bridge_token_cap_total,
                               expected_namespace=namespace,
                               expected_contract=contract),
    )
    if not result.ok:
        return SelectionOutcome(
            work=work, view_sha256=view.view_sha256,
            selector_attempted=True,
            normalization=selection.normalization.__dict__,
            failure=SelectionFailure("PREFLIGHT", "; ".join(result.errors[:3])),
            offered=len(view.candidates), offered_span_ids=offered_span_ids,
            offered_span_token_counts=offered_span_token_counts,
            publication_handle_map=publication_handle_map,
            publication_handle_token_counts=publication_handle_token_counts,
            publication_map_sha256=publication_map_sha256,
            offered_material_tokens=offered_material_tokens,
            offered_evidence_tokens=offered_evidence_tokens,
            offered_context_tokens=offered_context_tokens,
            offered_source_occurrence_ids=offered_source_occurrence_ids,
            selected_span_ids=selection.selected_span_ids,
            staged_span_ids=staged_span_ids,
            selected_relations=selected_relations,
            staged_relations=staged_relations,
            selected_gaps=selected_gaps,
            staged_gaps=staged_gaps,
            offered_query_attempt_ids=offered_query_attempt_ids,
            selected_query_attempt_ids=selected_query_attempt_ids,
            staged_query_attempt_ids=staged_query_attempt_ids,
            contract=contract, aggregation=aggregation, chunker=chunker, stage=stage,
            tokenizer_sha256=tokenizer_digest,
            checkpoint_hash=checkpoint_hash,
        )

    return SelectionOutcome(
        text=result.rendered.text, work=work, view_sha256=view.view_sha256,
        selector_attempted=True,
        published_source_occurrence_ids=tuple(dict.fromkeys(
            source_of_span[span_id] for span_id in staged_span_ids
            if source_of_span.get(span_id))),
        prompt_admission=(prompt_pack.accounting() if prompt_pack is not None else None),
        prompt_admission_dropped_span_ids=(
            prompt_pack.dropped_span_ids if prompt_pack is not None else ()),
        selected=len(aggregated.items), offered=len(view.candidates),
        dropped_for_budget=len(aggregated.dropped_for_budget),
        normalization=selection.normalization.__dict__,
        offered_span_ids=offered_span_ids,
        offered_span_token_counts=offered_span_token_counts,
        publication_handle_map=publication_handle_map,
        publication_handle_token_counts=publication_handle_token_counts,
        publication_map_sha256=publication_map_sha256,
        offered_material_tokens=offered_material_tokens,
        offered_evidence_tokens=offered_evidence_tokens,
        offered_context_tokens=offered_context_tokens,
        staged_rendered_tokens=result.rendered.token_count,
        published_rendered_tokens=result.rendered.token_count,
        offered_source_occurrence_ids=offered_source_occurrence_ids,
        selected_span_ids=selection.selected_span_ids,
        staged_span_ids=staged_span_ids,
        published_span_ids=staged_span_ids,
        selected_relations=selected_relations,
        staged_relations=staged_relations,
        published_relations=staged_relations,
        selected_gaps=selected_gaps,
        staged_gaps=staged_gaps,
        published_gaps=staged_gaps,
        offered_query_attempt_ids=offered_query_attempt_ids,
        selected_query_attempt_ids=selected_query_attempt_ids,
        staged_query_attempt_ids=staged_query_attempt_ids,
        published_query_attempt_ids=staged_query_attempt_ids,
        contract=contract, aggregation=aggregation, chunker=chunker, stage=stage,
        tokenizer_sha256=tokenizer_digest,
        checkpoint_hash=checkpoint_hash,
    )


def spans_from_page(text: str, *, content_hash: str, occurrence_id: str, chunker: str,
                    tokenizer: Tokenizer, max_tokens: int = 320) -> list[dict]:
    """Chunk one page's frozen bytes into RAW_SOURCE spans."""
    fn = CHUNKERS[chunker]
    kw = {"window": max_tokens, "overlap": 0} if chunker == "fixed_token_v1" \
        else {"max_tokens": max_tokens}
    return [
        build_evidence_span(c, text, content_hash=content_hash,
                            source_occurrence_ids=[occurrence_id], chunker_version=chunker)
        for c in fn(text, tokenizer, **kw)
    ]


#: Chunkers the close node can use over the compressor-visible view. `manifest` and
#: `visible_view` were declared as chunker names in configs/variants.yaml and read by
#: nothing: the close path never looked at cfg.chunker, so every C arm chunked identically
#: and the chunker axis did not exist at that node.
#:
#: Each entry takes one token budget, because the variant declares one. `fixed_token_v1`
#: expresses that as a sliding window with overlap, so the budget is translated here rather
#: than passed through under a name it does not have.
CLOSE_CHUNKERS: dict[str, Callable] = {
    "paragraph_sentence_v1": lambda text, tok, budget: paragraph_sentence_v1(
        text, tokenizer=tok, max_tokens=budget),
    "markdown_structure_v1": lambda text, tok, budget: markdown_structure_v1(
        text, tokenizer=tok, max_tokens=budget),
    "fixed_token_v1": lambda text, tok, budget: fixed_token_v1(
        text, tokenizer=tok, window=budget, overlap=max(1, budget // 8)),
}


def spans_from_visible_view(view_bytes: bytes, *, view_hash: str, messages: Sequence[dict],
                            tokenizer: Tokenizer, max_tokens: int = 320,
                            chunker: str = "paragraph_sentence_v1") -> list[dict]:
    """Chunk the exact compressor-visible bytes into VISIBLE_MESSAGE spans.

    ``messages`` gives each message's ``(message_id, role, byte_start, byte_end, kind)`` inside
    the view, so a span never straddles two messages and every span's kind is the message's own.
    AI reasoning is ``MODEL_DERIVED_CONTEXT`` and can never become TOOL_EVIDENCE -- that is the
    line between C_VISIBLE reading what the compressor read and C_VISIBLE inventing provenance.
    """
    spans: list[dict] = []
    for msg in messages:
        segment = view_bytes[msg["byte_start"]:msg["byte_end"]]
        try:
            text = segment.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if chunker not in CLOSE_CHUNKERS:
            raise ValueError(
                f"unknown close chunker {chunker!r} (have {sorted(CLOSE_CHUNKERS)})")
        for chunk in CLOSE_CHUNKERS[chunker](text, tokenizer, max_tokens):
            start = msg["byte_start"] + len(text[:chunk.char_start].encode("utf-8"))
            end = msg["byte_start"] + len(text[:chunk.char_end].encode("utf-8"))
            spans.append(build_visible_message_span(
                message_id=msg["message_id"], message_role=msg["role"],
                byte_start=start, byte_end=end, message_bytes=view_bytes,
                kind=msg["kind"], visible_compressor_view_hash=view_hash,
                source_occurrence_ids=list(msg.get("occurrence_ids") or [])
                if msg["kind"] == "TOOL_EVIDENCE" else None,
                token_len=chunk.token_len,
            ))
    return spans
