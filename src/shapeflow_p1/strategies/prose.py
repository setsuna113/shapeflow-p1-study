"""SHORT_PROSE: the control that separates the pointer mechanism from decoding less text.

This arm writes a summary instead of selecting ids, held to the *same rendered-token budget* as
the P1 arm it controls for. That equality is the entire design: if P1 saves tokens and
SHORT_PROSE saves the same tokens, the saving came from asking for less output, not from
pointing at evidence. Given a different budget it would answer neither question.

It deliberately does not go through the selection contract -- there are no ids to validate -- so
it bypasses parse/aggregate/preflight and returns its prose. What it does share is the offered
candidate view, so both arms read exactly the same bytes.
"""

from __future__ import annotations

from typing import Sequence

from ..evidence.chunkers import Tokenizer
from ..hashing import sha256_hex
from ..odr.checkpoints import CCheckpoint, HCheckpoint
from ..odr.hooks import ResearcherHandoff, ToolObservation
from ..p1.view import CandidateViewRecord
from .close_visible import CloseSelectionError, CloseStrategyConfig
from .page_h import PageSelectionError, PageStrategyConfig
from .pipeline import SelectionFailure, WorkRecord, spans_from_page, spans_from_visible_view
from .visible_view import build_visible_view

__all__ = ["ProsePageStrategy", "ProseCloseStrategy"]


def _truncate_to_budget(text: str, tokenizer: Tokenizer, budget: int) -> str:
    """Enforce the budget the prompt asked for.

    The model is told the limit, but a control that quietly exceeded it would no longer be
    token-matched, and the comparison it exists to support would be void. Truncation is at a
    token boundary so the arm's cost is exactly the budget.
    """
    offsets = tokenizer.encode_offsets(text)
    if len(offsets) <= budget:
        return text
    return text[: offsets[budget - 1][1]]


class ProsePageStrategy:
    """SHORT_PROSE at the WEBPAGE boundary."""

    def __init__(self, *, config: PageStrategyConfig, selector, tokenizer: Tokenizer,
                 raw_text_for, occurrence_for, work_sink=None) -> None:
        self.config = config
        self._selector = selector
        self._tokenizer = tokenizer
        self._raw_text_for = raw_text_for
        self._occurrence_for = occurrence_for
        self._work_sink = work_sink
        self.last_work = WorkRecord()

    async def transform_tool_batch(self, *, task_ctx, checkpoint: HCheckpoint
                                   ) -> Sequence[ToolObservation]:
        work = WorkRecord()
        observations: list[ToolObservation] = []
        for tool_call_id, results in checkpoint.search_result_sets:
            spans, texts, meta = [], {}, {}
            for result in results:
                if result.raw_content_id is None:
                    continue
                text = self._raw_text_for(result.raw_content_id)
                ch = sha256_hex(text.encode("utf-8"))
                page_spans = spans_from_page(
                    text, content_hash=ch,
                    occurrence_id=self._occurrence_for(result.raw_content_id),
                    chunker=self.config.chunker, tokenizer=self._tokenizer,
                    max_tokens=self.config.chunk_max_tokens)
                spans.extend(page_spans)
                texts[ch] = text
                meta.update({s["span_id"]: {"title": result.title, "url": result.url}
                             for s in page_spans})
            if not spans:
                observations.append(ToolObservation(
                    tool_call_id=tool_call_id, name="tavily_search",
                    content="No valid search results found. Please try different search queries or use a different search API."))
                continue
            view = CandidateViewRecord.build(
                spans=spans, tokenizer=self._tokenizer, namespace="RAW_SOURCE",
                topic=getattr(task_ctx, "research_topic", ""), contract="P1_ID",
                token_budget=self.config.token_budget, query_attempts=[],
                snapshot_texts=texts, source_meta=meta)
            try:
                text, call_work = await self._selector.summarize(
                    task_ctx=task_ctx, view=view, token_budget=self.config.token_budget)
            except Exception as e:  # noqa: BLE001
                work.selector_calls += 1
                self._finish(work)
                raise PageSelectionError(
                    SelectionFailure("PROSE_CONTROL_ERROR", f"{type(e).__name__}: {e}")) from e
            work.add(call_work)
            observations.append(ToolObservation(
                tool_call_id=tool_call_id, name="tavily_search",
                content=_truncate_to_budget(text, self._tokenizer, self.config.token_budget)))
        self._finish(work)
        return observations

    def _finish(self, work: WorkRecord) -> None:
        self.last_work = work
        if self._work_sink is not None:
            self._work_sink(self.config.variant_id, work)


class ProseCloseStrategy:
    """SHORT_PROSE at the RESEARCHER_CLOSE boundary, over the compressor-visible bytes only."""

    def __init__(self, *, config: CloseStrategyConfig, selector, tokenizer: Tokenizer,
                 work_sink=None) -> None:
        self.config = config
        self._selector = selector
        self._tokenizer = tokenizer
        self._work_sink = work_sink
        self.last_work = WorkRecord()

    async def close_researcher(self, *, task_ctx, checkpoint: CCheckpoint) -> ResearcherHandoff:
        view_obj = build_visible_view(checkpoint.researcher_messages)
        spans = spans_from_visible_view(
            view_obj.view_bytes, view_hash=view_obj.view_hash,
            messages=view_obj.message_segments, tokenizer=self._tokenizer,
            max_tokens=self.config.chunk_max_tokens)
        if not spans:
            raise CloseSelectionError(SelectionFailure("NO_VISIBLE_SPANS", "empty compressor view"))
        view = CandidateViewRecord.build(
            spans=spans, tokenizer=self._tokenizer, namespace="VISIBLE_MESSAGE",
            topic=getattr(task_ctx, "research_topic", ""), contract="P1_ID",
            token_budget=self.config.token_budget, query_attempts=[],
            visible_views={view_obj.view_hash: view_obj.view_bytes})
        try:
            text, work = await self._selector.summarize(
                task_ctx=task_ctx, view=view, token_budget=self.config.token_budget)
        except Exception as e:  # noqa: BLE001
            raise CloseSelectionError(
                SelectionFailure("PROSE_CONTROL_ERROR", f"{type(e).__name__}: {e}")) from e
        self.last_work = work
        if self._work_sink is not None:
            self._work_sink(self.config.variant_id, work)
        bounded = _truncate_to_budget(text, self._tokenizer, self.config.token_budget)
        return ResearcherHandoff(compressed_research=bounded, raw_notes=(bounded,))
