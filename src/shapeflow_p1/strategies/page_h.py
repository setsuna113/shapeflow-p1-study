"""WEBPAGE_P1: evidence-ID selection where vendor summarised each page into prose.

The comparison is fair only if both arms start from the same bytes, and the bytes are
``result['raw_content'][:max_content_length]`` -- vendor's own truncation, taken from the
deferred batch rather than re-derived. A variant that read the full page while P0 read a prefix
would be measuring page length, not selection, and AGENTS.md bars it from the primary contrast.

Scope is a first-class dimension, not an implementation detail:

- ``per_page`` -- one selector call per source. Most calls, smallest prompt each.
- ``per_tool_call`` -- one call per search result set.
- ``hierarchical`` -- a per-page shortlist, then a genuine second model pass over the union.
  That second pass is a real call with real tokens; charging it to the aggregator (or skipping
  it and calling externally-supplied scores a rerank) would hide the arm's actual cost.

Publication stays whole-batch: this returns one observation per sibling tool call, and a failure
anywhere returns failures for the batch so the caller falls all of it back to P0.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional, Sequence

from ..evidence.chunkers import Tokenizer
from ..hashing import sha256_hex
from ..odr.checkpoints import HCheckpoint
from ..odr.hooks import ToolObservation
from .pipeline import SelectionFailure, SelectionOutcome, WorkRecord, run_selection, spans_from_page

__all__ = ["PageSelectionStrategy", "PageStrategyConfig"]


@dataclass(frozen=True)
class PageStrategyConfig:
    variant_id: str
    chunker: str
    scope: str                  # per_page | per_tool_call | hierarchical
    contract: str               # P1_ID | P1_TYPED | P1_BRIDGE
    aggregation: str
    token_budget: int
    chunk_max_tokens: int = 320


class PageSelectionStrategy:
    """The H arm. ``raw_text_for`` resolves a content id to the frozen truncated page bytes."""

    def __init__(self, config: PageStrategyConfig, *, selector, tokenizer: Tokenizer,
                 raw_text_for, occurrence_for, work_sink=None) -> None:
        self.config = config
        self._selector = selector
        self._tokenizer = tokenizer
        self._raw_text_for = raw_text_for
        self._occurrence_for = occurrence_for
        self._work_sink = work_sink
        self.last_work = WorkRecord()

    async def transform_tool_batch(
        self, *, task_ctx, checkpoint: HCheckpoint
    ) -> Sequence[ToolObservation]:
        cfg = self.config
        batch_work = WorkRecord()
        observations: list[ToolObservation] = []

        # Groups are formed by scope, so a single failure in any group still fails the batch:
        # publishing the groups that happened to succeed would be a partial batch.
        for tool_call_id, results in checkpoint.search_result_sets:
            groups = self._groups_for(tool_call_id, results)
            rendered: list[str] = []
            for group in groups:
                outcome = await self._select_group(task_ctx, group)
                batch_work.add(outcome.work)
                if not outcome.ok:
                    self._record(batch_work)
                    raise PageSelectionError(outcome.failure)
                rendered.append(outcome.text)

            if cfg.scope == "hierarchical" and len(groups) > 1:
                # A real second pass over the union, with its own tokens. Not a reordering of
                # scores someone else supplied.
                union = [span for group in groups for span in group["spans"]]
                reduced = await self._select_group(
                    task_ctx, {"spans": union, "snapshot_texts": self._snapshots(union),
                               "meta": {}},
                    op_note="global",
                )
                batch_work.add(reduced.work)
                if not reduced.ok:
                    self._record(batch_work)
                    raise PageSelectionError(reduced.failure)
                rendered = [reduced.text]

            observations.append(ToolObservation(
                tool_call_id=tool_call_id, name="tavily_search",
                content=_format(rendered),
            ))

        self._record(batch_work)
        return observations

    # --- internals ---------------------------------------------------------------------

    def _groups_for(self, tool_call_id: str, results) -> list[dict]:
        """Chunk the frozen page bytes and split into selector-call groups by scope."""
        per_source: list[dict] = []
        for result in results:
            if result.raw_content_id is None:
                # Vendor falls back to the short snippet when raw content is absent, and so
                # must we: inventing evidence for a page that had none would be a different
                # world from P0's.
                per_source.append({"snippet": result.snippet, "spans": [],
                                   "snapshot_texts": {}, "meta": {}})
                continue
            text = self._raw_text_for(result.raw_content_id)
            content_hash = sha256_hex(text.encode("utf-8"))
            spans = spans_from_page(
                text, content_hash=content_hash,
                occurrence_id=self._occurrence_for(result.raw_content_id),
                chunker=self.config.chunker, tokenizer=self._tokenizer,
                max_tokens=self.config.chunk_max_tokens,
            )
            per_source.append({
                "spans": spans, "snapshot_texts": {content_hash: text},
                "meta": {s["span_id"]: {"title": result.title, "url": result.url}
                         for s in spans},
            })

        if self.config.scope == "per_tool_call":
            merged_spans = [s for g in per_source for s in g["spans"]]
            merged_texts: dict[str, str] = {}
            merged_meta: dict = {}
            for g in per_source:
                merged_texts.update(g["snapshot_texts"])
                merged_meta.update(g["meta"])
            return [{"spans": merged_spans, "snapshot_texts": merged_texts,
                     "meta": merged_meta}]
        return [g for g in per_source if g["spans"]] or [{"spans": [], "snapshot_texts": {},
                                                          "meta": {}}]

    def _snapshots(self, spans) -> dict[str, str]:
        out: dict[str, str] = {}
        for span in spans:
            ch = span["content_hash"]
            if ch not in out:
                out[ch] = self._raw_text_for_by_hash(ch)
        return out

    def _raw_text_for_by_hash(self, content_hash: str) -> str:
        return self._raw_text_for(content_hash)

    async def _select_group(self, task_ctx, group: dict, *, op_note: str = "") -> SelectionOutcome:
        if not group["spans"]:
            return SelectionOutcome(text=group.get("snippet", ""), work=WorkRecord())
        return await run_selection(
            spans=group["spans"], namespace="RAW_SOURCE",
            snapshot_texts=group["snapshot_texts"], visible_views={},
            topic=getattr(task_ctx, "research_topic", ""),
            contract=self.config.contract, aggregation=self.config.aggregation,
            token_budget=self.config.token_budget, tokenizer=self._tokenizer,
            query_attempts=[], selector=self._selector, task_ctx=task_ctx,
            known_occurrence_ids={
                oid for s in group["spans"] for oid in s.get("source_occurrence_ids", [])
            },
            source_meta=group["meta"],
        )

    def _record(self, work: WorkRecord) -> None:
        self.last_work = work
        if self._work_sink is not None:
            self._work_sink(self.config.variant_id, work)


class PageSelectionError(RuntimeError):
    """A P1 page failure. Carries the reason so the adapter can record it before falling back."""

    def __init__(self, failure: Optional[SelectionFailure]) -> None:
        super().__init__(f"{failure.reason}: {failure.detail}" if failure else "unknown")
        self.failure = failure


def _format(rendered: Sequence[str]) -> str:
    """One tool message per search call, sources separated as vendor separates them."""
    body = "\n\n".join(r for r in rendered if r)
    return f"Selected evidence: \n\n{body}\n" if body else \
        "No valid search results found. Please try different search queries or use a different search API."
