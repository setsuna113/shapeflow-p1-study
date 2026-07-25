"""RESEARCHER_CLOSE: C_VISIBLE, the compressor-only experiment, and C_REGISTRY beside it.

The two are different experiments and share nothing.

**C_VISIBLE** may read only the bytes P0's compressor read: the exact rendered
``researcher_messages``. Its candidate namespace is ``VISIBLE_MESSAGE`` and its offered view is
built from the ``VisibleCompressorView``, so raw page text is not merely discouraged but absent.
Its recall is scored against the ``VisibleTruthProjection`` -- truth restricted to what was
visible -- because scoring it against the full TruthPacket would penalise it for evidence it
could not have seen, and that is not the question C_VISIBLE answers.

**C_REGISTRY** may additionally re-read raw spans. That is a strictly larger visible world, so
it is a separate arm with its own report column and its own (full) recall denominator, and it is
never attributed as compressor-only. Merging the two would be the single most misleading thing
this study could do: the registry arm's numbers would flatter a mechanism it did not use.

``prefix_preserving`` and ``dedicated_selector`` are separate close modes rather than a flag,
because they differ in prompt prefix reuse and therefore in APC behaviour -- which the
operational layer measures directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from ..evidence.chunkers import Tokenizer
from ..odr.checkpoints import CCheckpoint
from ..odr.hooks import ResearcherHandoff
from .pipeline import (
    SelectionFailure,
    SelectionOutcome,
    WorkRecord,
    run_selection,
    spans_from_visible_view,
)
from .visible_view import VisibleCompressorView, build_visible_view

__all__ = ["CloseSelectionStrategy", "CloseStrategyConfig", "CloseSelectionError"]

VISIBLE = "C_VISIBLE"
REGISTRY = "C_REGISTRY"


@dataclass(frozen=True)
class CloseStrategyConfig:
    variant_id: str
    node: str                    # C_VISIBLE | C_REGISTRY | C_FUSED_EXT
    contract: str
    aggregation: str
    close_mode: str              # dedicated_selector | prefix_preserving | fused_with_fallback
    token_budget: int
    chunk_max_tokens: int = 320
    #: Which chunker runs over the compressor-visible view. The variant registry has always
    #: declared one per C arm; this node never had a field to put it in, so every C arm
    #: chunked the same way and the axis did not exist here.
    chunker: str = "paragraph_sentence_v1"

    def __post_init__(self) -> None:
        if self.node not in (VISIBLE, REGISTRY, "C_FUSED_EXT"):
            raise ValueError(f"unknown close node {self.node!r}")
        from .pipeline import CLOSE_CHUNKERS

        if self.chunker not in CLOSE_CHUNKERS:
            raise ValueError(
                f"close chunker {self.chunker!r} has no implementation "
                f"(have {sorted(CLOSE_CHUNKERS)})")


class CloseSelectionStrategy:
    """The C arm. ``raw_spans_for`` is supplied ONLY for C_REGISTRY."""

    def __init__(self, config: CloseStrategyConfig, *, selector, tokenizer: Tokenizer,
                 raw_spans_for=None, snapshot_texts_for=None, work_sink=None) -> None:
        self.config = config
        self._selector = selector
        self._tokenizer = tokenizer
        # Structural, not conventional: a C_VISIBLE strategy has no way to reach raw spans,
        # because it was never given the accessor. There is no flag to get wrong.
        if config.node == VISIBLE and raw_spans_for is not None:
            raise ValueError(
                "C_VISIBLE was constructed with a raw-span accessor. Its whole claim is that it "
                "sees only the compressor's bytes; holding the means to read more makes that "
                "claim a promise instead of a property."
            )
        self._raw_spans_for = raw_spans_for
        self._snapshot_texts_for = snapshot_texts_for
        self._work_sink = work_sink
        self.last_work = WorkRecord()
        self.last_view: Optional[VisibleCompressorView] = None

    async def close_researcher(
        self, *, task_ctx, checkpoint: CCheckpoint
    ) -> ResearcherHandoff:
        cfg = self.config
        view = build_visible_view(checkpoint.researcher_messages)
        self.last_view = view

        spans = spans_from_visible_view(
            view.view_bytes, view_hash=view.view_hash, messages=view.message_segments,
            tokenizer=self._tokenizer, max_tokens=cfg.chunk_max_tokens,
            # Read, not ignored. Every C arm used to chunk identically whatever its variant
            # declared, so the chunker axis did not exist at this node at all.
            chunker=cfg.chunker,
        )
        namespace = "VISIBLE_MESSAGE"
        snapshot_texts: dict[str, str] = {}
        visible_views = {view.view_hash: view.view_bytes}

        if cfg.node == REGISTRY and self._raw_spans_for is not None:
            # The extension's larger world. Kept in its own namespace and its own arm; the
            # offered view is built once per namespace, so the two never appear in one prompt
            # by accident.
            raw = self._raw_spans_for(checkpoint)
            if raw:
                spans = raw
                namespace = "RAW_SOURCE"
                snapshot_texts = self._snapshot_texts_for(checkpoint) if \
                    self._snapshot_texts_for else {}
                visible_views = {}

        outcome = await run_selection(
            spans=spans, namespace=namespace, snapshot_texts=snapshot_texts,
            visible_views=visible_views,
            topic=getattr(task_ctx, "research_topic", ""),
            contract=cfg.contract, aggregation=cfg.aggregation,
            token_budget=cfg.token_budget, tokenizer=self._tokenizer,
            query_attempts=[(qid, qid) for qid in checkpoint.query_attempt_ids],
            selector=self._selector, task_ctx=task_ctx,
            known_occurrence_ids={
                oid for s in spans for oid in (s.get("source_occurrence_ids") or [])
            },
        )
        self.last_work = outcome.work
        if self._work_sink is not None:
            self._work_sink(cfg.variant_id, outcome.work)
        if not outcome.ok:
            raise CloseSelectionError(outcome.failure)

        return ResearcherHandoff(
            compressed_research=_wrap(outcome.text, cfg.close_mode),
            # raw_notes keeps vendor's shape: the supervisor concatenates them, and an arm that
            # returned a different shape here would change downstream prompts for reasons that
            # have nothing to do with selection.
            raw_notes=(outcome.text,),
        )


def _wrap(text: str, close_mode: str) -> str:
    """Prefix handling, which is a measured dimension rather than cosmetics.

    ``prefix_preserving`` keeps a stable leading block so the operational layer's prefix cache
    can hit across arms; ``dedicated_selector`` does not. Comparing them is how the study
    separates a real saving from one the cache happened to provide.
    """
    if close_mode == "prefix_preserving":
        return "<research_findings>\n" + text + "\n</research_findings>"
    return text


class CloseSelectionError(RuntimeError):
    def __init__(self, failure: Optional[SelectionFailure]) -> None:
        super().__init__(f"{failure.reason}: {failure.detail}" if failure else "unknown")
        self.failure = failure
