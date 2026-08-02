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
from ..evidence.model_tokenizer import tokenizer_sha256
from ..hashing import sha256_hex
from ..odr.checkpoints import HCheckpoint
from ..odr.hooks import ToolObservation
from .pipeline import SelectionFailure, SelectionOutcome, WorkRecord, run_selection, spans_from_page

__all__ = ["PAGE_SCOPES", "PageSelectionStrategy", "PageStrategyConfig"]

#: How candidates are partitioned across selector calls.
#:
#: ``whole_batch`` is the one the H/C mechanism contract actually specifies: *"One whole-batch
#: selector request covering every page in the batch"* (HC_MECHANISM_v1 §H). The three that
#: preceded it all fan out below the gather batch, because the sibling loop in
#: :meth:`PageSelectionStrategy.transform_tool_batch` sits *above* this switch -- so before
#: ``whole_batch`` existed the contract's shape was unrepresentable, and every H arm ran N
#: selector requests per batch, each with its own ``selected_token_budget``.
#:
#: ``per_tool_call`` is deliberately kept rather than renamed. It means "one call per search
#: result set" and stays correct when a batch has several siblings; ``whole_batch`` means "one
#: call per gather batch". They coincide only at one sibling per batch, which is what the
#: recorded checkpoints happen to show. Renaming would mean a two-sibling batch silently issued
#: two requests under two budgets while claiming to be whole-batch -- the defect, restored under
#: the fixed name.
PAGE_SCOPES = frozenset({"per_page", "per_tool_call", "whole_batch", "hierarchical"})


@dataclass(frozen=True)
class PageStrategyConfig:
    variant_id: str
    chunker: str
    scope: str                  # per_page | per_tool_call | whole_batch | hierarchical
    contract: str               # P1_ID | P1_TYPED | P1_BRIDGE
    aggregation: str
    token_budget: int
    chunk_max_tokens: int = 320
    bridge_token_cap_each: int | None = None
    bridge_token_cap_total: int | None = None

    def __post_init__(self) -> None:
        if self.scope not in PAGE_SCOPES:
            raise ValueError(f"unknown page scope {self.scope!r}")
        if self.contract == "P1_BRIDGE":
            if self.bridge_token_cap_each is None or self.bridge_token_cap_total is None:
                raise ValueError("P1_BRIDGE page variant requires per-bridge and total token caps")
            if not (0 < self.bridge_token_cap_each <= self.bridge_token_cap_total):
                raise ValueError("bridge caps require 0 < each <= total")
        elif self.bridge_token_cap_each is not None or self.bridge_token_cap_total is not None:
            raise ValueError("bridge caps belong only to a P1_BRIDGE variant")


class PageSelectionStrategy:
    """The H arm. ``raw_text_for`` resolves a content id to the frozen truncated page bytes."""

    def __init__(self, config: PageStrategyConfig, *, selector, tokenizer: Tokenizer,
                 raw_text_for, occurrence_for, global_selector=None, work_sink=None) -> None:
        self.config = config
        self._selector = selector
        self._global_selector = global_selector
        if config.scope == "hierarchical" and global_selector is None:
            raise ValueError(
                "hierarchical page strategy requires a distinct global selector so local and "
                "global calls are attributed separately"
            )
        self._tokenizer = tokenizer
        self._raw_text_for = raw_text_for
        self._occurrence_for = occurrence_for
        self._work_sink = work_sink
        self.last_work = WorkRecord()
        self.last_outcomes: list[SelectionOutcome] = []
        self._publication_scope_digests: dict[tuple[object, ...], str] = {}

    async def transform_tool_batch(
        self, *, task_ctx, checkpoint: HCheckpoint
    ) -> Sequence[ToolObservation]:
        batch_work = WorkRecord()
        observations: list[ToolObservation] = []
        self.last_outcomes = []

        if self.config.scope == "whole_batch" and len(checkpoint.search_result_sets) != 1:
            # One selector request per gather batch is the treatment; the loop below is per
            # sibling, so with two siblings this arm would issue two requests under two separate
            # 512-token budgets while still calling itself whole-batch. Refusing sends the batch
            # down the ordinary P1-failure path -- component trials record it, end-to-end falls
            # the whole batch back to P0 -- so the cell stays comparable and the exclusion is
            # counted rather than silently absorbed.
            #
            # Every H checkpoint recorded so far carries exactly one sibling, so this is a
            # checked precondition rather than a code path anyone has seen taken. That is the
            # reason to check it: an assumption that holds by observation and not by
            # construction is one a future vendor bump can retire without telling anyone.
            self._record(batch_work)
            raise PageSelectionError(SelectionFailure(
                "WHOLE_BATCH_MULTI_SIBLING",
                f"gather batch has {len(checkpoint.search_result_sets)} sibling tool calls; "
                "a whole_batch arm publishes one selector request per batch and cannot span "
                "siblings without splitting the rendered-token budget",
            ))

        # Vendor executes sibling tool calls in one outer gather.  Running each sibling's P1
        # reducer sequentially would create a scheduling treatment unrelated to ID selection
        # and artificially hurt operational latency.  Each sibling is reduced concurrently,
        # then the ordered result list is inspected as one atomic batch before any publication.
        reduced_siblings = await asyncio.gather(*(
            self._transform_one_tool_call(
                task_ctx=task_ctx,
                checkpoint=checkpoint,
                checkpoint_hash=checkpoint.digest,
                toolset_ordinal=toolset_ordinal,
                tool_call_id=tool_call_id,
                results=results,
            )
            for toolset_ordinal, (tool_call_id, results) in enumerate(
                checkpoint.search_result_sets
            )
        ))
        first_failure: Optional[SelectionFailure] = None
        for observation, outcomes, work, failure in reduced_siblings:
            self.last_outcomes.extend(outcomes)
            batch_work.add(work)
            if failure is not None and first_failure is None:
                first_failure = failure
            if observation is not None:
                observations.append(observation)
        if first_failure is not None:
            self._record(batch_work)
            raise PageSelectionError(first_failure)

        self._record(batch_work)
        return observations

    # --- internals ---------------------------------------------------------------------

    async def _transform_one_tool_call(
        self,
        *,
        task_ctx,
        checkpoint: HCheckpoint,
        checkpoint_hash: str,
        toolset_ordinal: int,
        tool_call_id: str,
        results,
    ) -> tuple[
        Optional[ToolObservation],
        list[SelectionOutcome],
        WorkRecord,
        Optional[SelectionFailure],
    ]:
        """Reduce one sibling while keeping its map/reduce dependency local to that sibling."""
        cfg = self.config
        groups = self._groups_for(tool_call_id, results)
        self._bind_publication_scope(
            checkpoint=checkpoint,
            toolset_ordinal=toolset_ordinal,
            groups=groups,
        )
        work = WorkRecord()
        outcomes: list[SelectionOutcome] = []
        rendered: list[str] = []
        # Only these outcomes' material is placed in the final ToolMessage.  Hierarchical
        # local/map outputs merely form the shortlist; attributing their occurrences would
        # over-claim sources that the global reducer later dropped.
        published_outcomes: list[SelectionOutcome] = []
        if cfg.scope == "hierarchical":
            # Map calls see one raw page each. Reduce starts only after every map result in this
            # sibling exists, but unrelated siblings remain concurrent.
            selectable = [group for group in groups if group["spans"]]
            local = list(await asyncio.gather(*(
                self._select_group(
                    task_ctx,
                    group,
                    selector=self._selector,
                    stage="local",
                    checkpoint_hash=checkpoint_hash,
                )
                for group in selectable
            )))
            outcomes.extend(local)
            for outcome in local:
                work.add(outcome.work)
            failure = next(
                (outcome.failure for outcome in local if not outcome.ok), None
            )
            if failure is not None:
                return None, outcomes, work, failure

            shortlist = {
                span_id
                for outcome in local
                for span_id in outcome.published_span_ids
            }
            union = [
                span
                for group in selectable
                for span in group["spans"]
                if _span_id(span) in shortlist
            ]
            if union:
                reduced = await self._select_group(
                    task_ctx,
                    self._merged_group(selectable, union),
                    selector=self._global_selector,
                    stage="global",
                    checkpoint_hash=checkpoint_hash,
                )
                outcomes.append(reduced)
                work.add(reduced.work)
                if not reduced.ok:
                    return None, outcomes, work, reduced.failure
                rendered.append(reduced.text or "")
                published_outcomes.append(reduced)
            rendered.extend(
                passthrough
                for group in groups
                for passthrough in group.get("passthrough", ())
            )
        else:
            selected = list(await asyncio.gather(*(
                self._select_group(
                    task_ctx,
                    group,
                    selector=self._selector,
                    stage="single",
                    checkpoint_hash=checkpoint_hash,
                )
                for group in groups
            )))
            outcomes.extend(selected)
            for outcome in selected:
                work.add(outcome.work)
            failure = next(
                (outcome.failure for outcome in selected if not outcome.ok), None
            )
            if failure is not None:
                return None, outcomes, work, failure
            rendered.extend(outcome.text or "" for outcome in selected)
            published_outcomes.extend(selected)

        return (
            ToolObservation(
                tool_call_id=tool_call_id,
                name="tavily_search",
                content=_format(rendered),
                source_occurrence_ids=self._published_occurrences(
                    groups, published_outcomes
                ),
            ),
            outcomes,
            work,
            None,
        )

    def _bind_publication_scope(
        self,
        *,
        checkpoint: HCheckpoint,
        toolset_ordinal: int,
        groups: Sequence[dict],
    ) -> None:
        """Allocate compact handles once for the whole atomic sibling publication.

        ``assistant_turn_index`` can recur on a graph retry. Replaying the identical checkpoint
        is idempotent; different checkpoint bytes under the same structural scope are rejected
        before a selector call, so retry/resume cannot give one handle two meanings.
        """
        scope_key = (
            str(checkpoint.task_id),
            str(checkpoint.researcher_id),
            tuple(checkpoint.researcher_coordinate or ()),
            int(checkpoint.assistant_turn_index),
            int(toolset_ordinal),
        )
        checkpoint_digest = checkpoint.digest
        previous = self._publication_scope_digests.get(scope_key)
        if previous is not None and previous != checkpoint_digest:
            raise PageSelectionError(SelectionFailure(
                "PUBLICATION_SCOPE_REUSE",
                "a different H checkpoint reused the same researcher/assistant-turn/toolset "
                "publication namespace; retry/resume may replay identical bytes but may not "
                "alias a prior publication",
            ))
        self._publication_scope_digests[scope_key] = checkpoint_digest

        ordinals: dict[str, int] = {}
        for group in groups:
            for span in group.get("spans", ()):
                span_id = _span_id(span)
                if span_id not in ordinals:
                    ordinals[span_id] = len(ordinals) + 1
        coordinate = checkpoint.researcher_coordinate
        scope = (
            (
                int(coordinate[0]),
                int(coordinate[1]),
                int(checkpoint.assistant_turn_index),
                int(toolset_ordinal),
            )
            if coordinate is not None
            else (int(checkpoint.assistant_turn_index), int(toolset_ordinal))
        )
        for group in groups:
            group["publication_scope"] = scope
            group["publication_ordinals"] = ordinals

    def _groups_for(self, tool_call_id: str, results) -> list[dict]:
        """Chunk the frozen page bytes and split into selector-call groups by scope."""
        per_source: list[dict] = []
        for result in results:
            if result.raw_content_id is None:
                # Vendor falls back to the short snippet when raw content is absent, and so
                # must we: inventing evidence for a page that had none would be a different
                # world from P0's.
                per_source.append({"snippet": result.snippet, "spans": [],
                                   "snapshot_texts": {}, "meta": {},
                                   "occurrence_ids": (),
                                   "passthrough": (_vendor_snippet(result),),
                                   "passthrough_occurrence_ids": (
                                       (str(result.source_occurrence_id),)
                                       if result.source_occurrence_id else ()
                                   )})
                continue
            text = self._raw_text_for(result.raw_content_id)
            content_hash = sha256_hex(text.encode("utf-8"))
            occurrence_id = (
                result.source_occurrence_id
                or self._occurrence_for(result.raw_content_id)
            )
            spans = spans_from_page(
                text, content_hash=content_hash,
                # Content identity is not occurrence identity. Two URLs can serve identical
                # bytes; the checkpoint carries the per-result frozen occurrence so citations do
                # not collapse to whichever URL the runner inserted first.
                occurrence_id=occurrence_id,
                chunker=self.config.chunker, tokenizer=self._tokenizer,
                max_tokens=self.config.chunk_max_tokens,
            )
            per_source.append({
                "spans": spans, "snapshot_texts": {content_hash: text},
                "meta": {s["span_id"]: {"title": result.title, "url": result.url}
                         for s in spans},
                "occurrence_ids": (occurrence_id,),
                "passthrough": (),
                "passthrough_occurrence_ids": (),
            })

        # ``whole_batch`` merges by the same rule. It differs from ``per_tool_call`` only in
        # what it refuses, and that refusal lives in ``transform_tool_batch`` where the sibling
        # count is visible; here there is one sibling's results either way.
        if self.config.scope in {"per_tool_call", "whole_batch"}:
            merged_spans = [s for g in per_source for s in g["spans"]]
            merged_texts: dict[str, str] = {}
            merged_meta: dict = {}
            for g in per_source:
                merged_texts.update(g["snapshot_texts"])
                merged_meta.update(g["meta"])
            return [{"spans": merged_spans, "snapshot_texts": merged_texts,
                     "meta": merged_meta,
                     "occurrence_ids": tuple(dict.fromkeys(
                         oid for group in per_source
                         for oid in group.get("occurrence_ids", ())
                     )),
                     "passthrough": tuple(
                         p for g in per_source for p in g.get("passthrough", ())
                     ),
                     "passthrough_occurrence_ids": tuple(dict.fromkeys(
                         oid for g in per_source
                         for oid in g.get("passthrough_occurrence_ids", ())
                     ))}]
        return per_source or [{"spans": [], "snapshot_texts": {}, "meta": {},
                               "occurrence_ids": (),
                               "passthrough": (),
                               "passthrough_occurrence_ids": ()}]

    @staticmethod
    def _merged_group(groups: Sequence[dict], spans: Sequence[dict]) -> dict:
        """Build reduce input from map survivors and only their frozen backing data."""
        wanted = {_span_id(span) for span in spans}
        texts: dict[str, str] = {}
        meta: dict = {}
        for group in groups:
            if any(_span_id(span) in wanted for span in group["spans"]):
                texts.update(group["snapshot_texts"])
                meta.update({
                    sid: value for sid, value in group["meta"].items() if sid in wanted
                })
        return {
            "spans": list(spans), "snapshot_texts": texts, "meta": meta, "passthrough": (),
            "passthrough_occurrence_ids": (),
            "publication_scope": (
                groups[0].get("publication_scope") if groups else None
            ),
            "publication_ordinals": (
                groups[0].get("publication_ordinals", {}) if groups else {}
            ),
            "occurrence_ids": tuple(dict.fromkeys(
                oid for group in groups
                if any(_span_id(span) in wanted for span in group["spans"])
                for oid in group.get("occurrence_ids", ())
            )),
        }

    @staticmethod
    def _published_occurrences(
        groups: Sequence[dict],
        published_outcomes: Sequence[SelectionOutcome],
    ) -> tuple[str, ...]:
        """Lineage of bytes that actually enter this P1 ToolMessage.

        Iterate the frozen source/span order after forming the set selected by the outcome(s).
        This deliberately does not use ``offered_source_occurrence_ids``: a per-tool-call view
        can offer ten sources and publish one, and C_VISIBLE must not cite the other nine.
        Snippet passthrough is tracked separately because it has no selectable span.
        """
        published_span_ids = {
            str(span_id)
            for outcome in published_outcomes
            for span_id in outcome.published_span_ids
        }
        ordered: list[str] = []
        for group in groups:
            for span in group.get("spans", ()):
                if _span_id(span) not in published_span_ids:
                    continue
                for occurrence_id in span.get("source_occurrence_ids") or ():
                    value = str(occurrence_id)
                    if value and value not in ordered:
                        ordered.append(value)
            for occurrence_id in group.get("passthrough_occurrence_ids", ()):
                value = str(occurrence_id)
                if value and value not in ordered:
                    ordered.append(value)
        return tuple(ordered)

    async def _select_group(
        self,
        task_ctx,
        group: dict,
        *,
        selector,
        stage: str,
        checkpoint_hash: str,
    ) -> SelectionOutcome:
        if not group["spans"]:
            return SelectionOutcome(
                text="\n\n".join(group.get("passthrough", ())),
                work=WorkRecord(), contract=self.config.contract,
                aggregation=self.config.aggregation, stage=stage,
                checkpoint_hash=checkpoint_hash,
                # Empty/no-raw groups are still part of the all-offered H denominator.  Give
                # their zero-length trace the same frozen chunker/tokenizer identity as the
                # selectable groups; otherwise one snippet-only result makes the complete H
                # trace incomparable even though it never entered the selector.
                chunker=self.config.chunker,
                tokenizer_sha256=tokenizer_sha256(self._tokenizer),
                offered_source_occurrence_ids=tuple(
                    map(str, group.get("occurrence_ids", ()))),
            )
        outcome = await run_selection(
            spans=group["spans"], namespace="RAW_SOURCE",
            snapshot_texts=group["snapshot_texts"], visible_views={},
            topic=getattr(task_ctx, "research_topic", ""),
            contract=self.config.contract, aggregation=self.config.aggregation,
            chunker=self.config.chunker,
            token_budget=self.config.token_budget, tokenizer=self._tokenizer,
            query_attempts=[], selector=selector, task_ctx=task_ctx,
            known_occurrence_ids=set(map(
                str, group.get("occurrence_ids", ()))),
            source_meta=group["meta"],
            bridge_token_cap_each=self.config.bridge_token_cap_each,
            bridge_token_cap_total=self.config.bridge_token_cap_total,
            checkpoint_hash=checkpoint_hash,
            stage=stage,
            publication_scope=group.get("publication_scope"),
            publication_ordinals=group.get("publication_ordinals"),
        )
        if outcome.ok and group.get("passthrough"):
            outcome.text = "\n\n".join((outcome.text, *group["passthrough"]))
        return outcome

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


def _span_id(span: dict) -> str:
    return span.get("span_id") or span.get("visible_span_id") or ""


def _vendor_snippet(result) -> str:
    """Vendor's no-raw-content branch, kept byte-for-byte as a fixed passthrough."""
    return (
        f"--- SOURCE {result.vendor_visible_order + 1}: {result.title} ---\n"
        f"URL: {result.url}\n\n"
        f"SUMMARY:\n{result.snippet}\n\n"
        + "-" * 80
    )
