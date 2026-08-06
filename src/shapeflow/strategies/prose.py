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

import asyncio
import re
from collections.abc import Sequence

from ..evidence.chunkers import Tokenizer
from ..evidence.model_tokenizer import tokenizer_sha256
from ..hashing import sha256_hex
from ..odr.checkpoints import CCheckpoint, HCheckpoint
from ..odr.hooks import ResearcherHandoff, ToolObservation
from ..p1.view import CandidateViewRecord
from .close_visible import CloseSelectionError, CloseStrategyConfig
from .page_h import PageSelectionError, PageStrategyConfig, _format, _vendor_snippet
from .pipeline import SelectionFailure, WorkRecord, spans_from_page, spans_from_visible_view
from .visible_view import (
    VisibleSourceEntry,
    VisibleSourceRegistryError,
    build_visible_source_registry,
    build_visible_view,
    source_partitioned_message_segments,
)

__all__ = ["ProsePageStrategy", "ProseCloseStrategy"]


_ANY_URL = re.compile(r"https?://[^\s<>\]\)]+", re.IGNORECASE)
_CITATION = re.compile(r"\[(?P<label>[^\]\r\n]{1,32})\]")
_DANGLING_NUMERIC_CITATION = re.compile(r"\[\d*$")
_SOURCE_DEFINITION_INJECTION = re.compile(
    r"(?im)^\s*(?:---\s*SOURCE\b|URL\s*:|SOURCES?\s*:)"
)


class _ProseValidationError(ValueError):
    """A bounded SHORT_PROSE prefix is invalid, with its sensitivity record attached."""

    def __init__(self, message: str, *, normalization: dict) -> None:
        super().__init__(message)
        self.normalization = normalization


def _work_from_exception(error: BaseException) -> WorkRecord:
    """Recover already-spent work from either a wrapped attempt or provider usage."""

    attached = getattr(error, "work", None)
    if isinstance(attached, WorkRecord):
        return attached
    usage = getattr(error, "usage", {}) or {}
    if not isinstance(usage, dict):
        usage = {}
    return WorkRecord(
        # We are called only after entering the model-call await. A provider may fail before it
        # can return usage, but that is still an attempted selector call.
        selector_calls=1,
        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        retries=int(usage.get("retries", 0) or 0),
    )


class _ProseAttemptError(RuntimeError):
    """Post-model materialization failure that retains the successful call's WorkRecord."""

    def __init__(
        self,
        cause: Exception,
        *,
        work: WorkRecord,
        control_record: dict | None = None,
    ) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.work = work
        self.control_record = dict(control_record or {})


def _validation_of(
    text: str, *, by_label: dict[str, VisibleSourceEntry], sources_required: bool
) -> dict:
    labels = _citation_labels(text)
    inline_labels = _inline_citation_labels(text)
    unknown = tuple(label for label in labels if label not in by_label)
    url_count = len(_ANY_URL.findall(text))
    source_definition_injection_count = len(
        _SOURCE_DEFINITION_INJECTION.findall(text)
    )
    dangling = bool(_DANGLING_NUMERIC_CITATION.search(text))
    missing_inline = bool(sources_required and not inline_labels)
    return {
        "valid": not (
            unknown
            or url_count
            or source_definition_injection_count
            or dangling
            or missing_inline
        ),
        "unknown_citation_labels": list(unknown),
        "inline_citation_labels": list(inline_labels),
        "body_url_count": int(url_count),
        "source_definition_injection_count": int(
            source_definition_injection_count
        ),
        "dangling_numeric_citation": dangling,
        "missing_required_inline_citation": missing_inline,
    }


def _validation_error(validation: dict) -> str:
    unknown = validation["unknown_citation_labels"]
    if unknown:
        return "SHORT_PROSE cited unknown source label(s): " + ", ".join(unknown)
    if validation["body_url_count"]:
        return (
            "SHORT_PROSE body emitted a URL; source definitions are appended by the publisher"
        )
    if validation["source_definition_injection_count"]:
        return "SHORT_PROSE body attempted to inject publisher-owned source definitions"
    if validation["dangling_numeric_citation"]:
        return "SHORT_PROSE bounded prefix ends in an incomplete numeric citation"
    if validation["missing_required_inline_citation"]:
        return "SHORT_PROSE had citable sources but emitted no valid inline citation"
    return "SHORT_PROSE bounded prefix failed validation"


def _render_close_prose(
    text: str,
    *,
    source_entries: Sequence[VisibleSourceEntry],
    tokenizer: Tokenizer,
    budget: int,
) -> tuple[str, str, tuple[VisibleSourceEntry, ...], dict]:
    """Budget first, then validate exactly the longest publishable prefix.

    Text strictly beyond the shared rendered-token budget cannot affect downstream quality and
    therefore cannot turn a usable control output into a fallback. Conversely, an unknown
    citation or raw URL that remains inside the chosen prefix is never repaired by silently
    shortening again: that exact prefix fails closed.
    """

    by_label = {entry.label: entry for entry in source_entries}
    offsets = tokenizer.encode_offsets(text)
    bounded: tuple[str, str, tuple[VisibleSourceEntry, ...]] | None = None
    for kept in range(min(len(offsets), budget), -1, -1):
        body = text[: offsets[kept - 1][1]] if kept else ""
        # A token boundary can bisect "[12]". Backing off that mechanically incomplete suffix
        # does not forgive an unknown complete citation or URL inside the chosen prefix.
        if _DANGLING_NUMERIC_CITATION.search(body):
            continue
        labels = _inline_citation_labels(body)
        used = tuple(
            entry for entry in source_entries if entry.label in set(labels)
        )
        rendered = _render_close_document(body, used)
        if tokenizer.count(rendered) <= budget:
            bounded = (rendered, body, used)
            break
    if bounded is None:
        raise ValueError("SHORT_PROSE could not fit its summary wrapper")

    rendered, body, used = bounded
    raw_validation = _validation_of(
        text, by_label=by_label, sources_required=bool(source_entries)
    )
    published_validation = _validation_of(
        body, by_label=by_label, sources_required=bool(source_entries)
    )
    full_labels = set(_inline_citation_labels(text))
    full_used = tuple(
        entry for entry in source_entries if entry.label in full_labels
    )
    raw_rendered = _render_close_document(text, full_used)
    truncated = body != text
    raw_fits_budget = tokenizer.count(raw_rendered) <= budget
    invalid_raw_suffix = bool(
        truncated
        and not raw_validation["valid"]
        and published_validation["valid"]
    )
    normalization = {
        "schema_version": "short_prose_close_policy_normalization_v2",
        "raw_semantically_valid": bool(raw_validation["valid"]),
        "raw_fits_rendered_token_budget": bool(raw_fits_budget),
        "raw_contract_adherent": bool(raw_validation["valid"] and raw_fits_budget),
        "published_semantically_valid": bool(published_validation["valid"]),
        "published_fits_rendered_token_budget": bool(
            tokenizer.count(rendered) <= budget
        ),
        "published_policy_valid": bool(
            published_validation["valid"] and tokenizer.count(rendered) <= budget
        ),
        "raw_validation": raw_validation,
        "published_validation": published_validation,
        "raw_body_tokens": tokenizer.count(text),
        "published_body_tokens": tokenizer.count(body),
        "raw_rendered_tokens": tokenizer.count(raw_rendered),
        "published_rendered_tokens": tokenizer.count(rendered),
        "budget_truncated": truncated,
        "invalid_raw_suffix_outside_policy_output": invalid_raw_suffix,
        # The prefix is fixed by budget before semantic validation. Nothing searches for a
        # shorter valid answer, so invalid text beyond that independent boundary is not a
        # semantic repair.
        "semantic_repair_applied": False,
        "normalization_reasons": [
            reason
            for reason, condition in (
                ("TOKEN_BUDGET_PREFIX", truncated),
                (
                    "INVALID_RAW_SUFFIX_OUTSIDE_POLICY_OUTPUT",
                    invalid_raw_suffix,
                ),
            )
            if condition
        ],
        "raw_contract_adherence_sensitivity_excludes": bool(
            truncated or not raw_validation["valid"] or not raw_fits_budget
        ),
    }
    if not normalization["published_policy_valid"]:
        raise _ProseValidationError(
            _validation_error(published_validation),
            normalization=normalization,
        )
    return rendered, body, used, normalization


def _citation_labels(text: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(match.group("label").strip() for match in _CITATION.finditer(text))
    )


def _inline_citation_labels(text: str) -> tuple[str, ...]:
    """Labels that follow substantive text on their line, not model-written source entries."""

    labels: list[str] = []
    for match in _CITATION.finditer(text):
        line_start = text.rfind("\n", 0, match.start()) + 1
        if not text[line_start:match.start()].strip():
            continue
        label = match.group("label").strip()
        if label not in labels:
            labels.append(label)
    return tuple(labels)


def _render_close_document(
    body: str, source_entries: Sequence[VisibleSourceEntry]
) -> str:
    rendered = "SUMMARY:\n" + body
    if source_entries:
        rendered += "\n\nSOURCES:\n" + "\n".join(
            entry.map_line for entry in source_entries
        )
    return rendered


def _minimum_close_wrapper(
    source_entries: Sequence[VisibleSourceEntry], tokenizer: Tokenizer
) -> str:
    """Reserve fixed framing plus the cheapest one legal source entry, never the whole map."""

    if not source_entries:
        return _render_close_document("", ())
    return min(
        (_render_close_document("", (entry,)) for entry in source_entries),
        key=tokenizer.count,
    )


def _completion_cap(*, immutable_wrapper: str, tokenizer: Tokenizer, budget: int) -> int:
    """A competent control does not decode tokens that its fixed wrapper must discard."""

    remaining = budget - tokenizer.count(immutable_wrapper)
    if remaining <= 0:
        raise ValueError(
            "SHORT_PROSE token budget leaves no completion token after immutable provenance"
        )
    return remaining


def _batch_prose_wrapper(sources: Sequence[tuple[int, str, str]]) -> tuple[str, str]:
    """Fixed provenance for a whole-batch prose summary: one body, a map of every source.

    The per-page wrapper names one source in its header because a per-page summary describes one
    page. A whole-batch summary describes the batch, so every page it drew on has to remain
    citable -- otherwise the control is a straw man, since the downstream writer can cite both
    P0 and structured P1 but not this. Modelled on the C-side close document, which solved the
    same problem: a body followed by a SOURCES map.

    The whole map is immutable framing and is charged against the budget before a token is
    decoded, so the control never generates text its own wrapper must then discard.
    """
    prefix = "SUMMARY:\n"
    if not sources:
        return prefix, ""
    lines = "\n".join(f"[{number}] {title} — {url}" for number, title, url in sources)
    return prefix, "\n\nSOURCES:\n" + lines


def _page_prose_wrapper(
    *, title: str, url: str, source_number: int
) -> tuple[str, str]:
    return (
        (
            f"--- SOURCE {source_number}: {title} ---\n"
            f"URL: {url}\n\n"
            "SUMMARY:\n"
        ),
        "\n\n" + "-" * 80,
    )


def _render_page_prose(
    text: str,
    *,
    title: str,
    url: str,
    source_number: int,
    tokenizer: Tokenizer,
    budget: int,
) -> tuple[str, str, dict]:
    """Keep citation provenance while fitting the complete page document to the H budget.

    A bare prose summary is a straw-man control: the downstream writer cannot cite it while it
    can cite both P0 and structured P1.  Title, URL and the source delimiter are fixed
    provenance, so only the generated body may be shortened.  The count is taken over the
    complete page document, just as P1 preflight counts its complete structured renderer.
    """
    prefix, suffix = _page_prose_wrapper(
        title=title,
        url=url,
        source_number=source_number,
    )
    if tokenizer.count(prefix + suffix) > budget:
        raise ValueError(
            "SHORT_PROSE token budget cannot preserve the source title/URL wrapper"
        )
    offsets = tokenizer.encode_offsets(text)
    raw_rendered = prefix + text + suffix
    raw_rendered_tokens = tokenizer.count(raw_rendered)
    # Tokenization at the prefix/body boundary can merge tokens, so count the complete
    # candidate rather than subtracting independently-tokenized wrapper counts.
    for kept in range(min(len(offsets), budget), -1, -1):
        body = text[: offsets[kept - 1][1]] if kept else ""
        rendered = prefix + body + suffix
        rendered_tokens = tokenizer.count(rendered)
        if rendered_tokens <= budget:
            truncated = body != text
            raw_validation = _validation_of(
                text, by_label={}, sources_required=False
            )
            published_validation = _validation_of(
                body, by_label={}, sources_required=False
            )
            invalid_raw_suffix = bool(
                truncated
                and not raw_validation["valid"]
                and published_validation["valid"]
            )
            normalization = {
                "schema_version": "short_prose_page_policy_normalization_v2",
                "raw_semantically_valid": bool(raw_validation["valid"]),
                "raw_fits_rendered_token_budget": raw_rendered_tokens <= budget,
                "raw_contract_adherent": bool(
                    raw_validation["valid"] and raw_rendered_tokens <= budget
                ),
                "published_semantically_valid": bool(
                    published_validation["valid"]
                ),
                "published_fits_rendered_token_budget": rendered_tokens <= budget,
                "published_policy_valid": bool(
                    published_validation["valid"] and rendered_tokens <= budget
                ),
                "raw_validation": raw_validation,
                "published_validation": published_validation,
                "raw_body_tokens": tokenizer.count(text),
                "published_body_tokens": tokenizer.count(body),
                "raw_rendered_tokens": raw_rendered_tokens,
                "published_rendered_tokens": rendered_tokens,
                "budget_truncated": truncated,
                "invalid_raw_suffix_outside_policy_output": invalid_raw_suffix,
                "semantic_repair_applied": False,
                "normalization_reasons": [
                    reason
                    for reason, condition in (
                        ("TOKEN_BUDGET_PREFIX", truncated),
                        (
                            "INVALID_RAW_SUFFIX_OUTSIDE_POLICY_OUTPUT",
                            invalid_raw_suffix,
                        ),
                    )
                    if condition
                ],
                "raw_contract_adherence_sensitivity_excludes": bool(
                    truncated
                    or not raw_validation["valid"]
                    or raw_rendered_tokens > budget
                ),
            }
            if not normalization["published_policy_valid"]:
                raise _ProseValidationError(
                    _validation_error(published_validation),
                    normalization=normalization,
                )
            return rendered, body, normalization
    raise ValueError("SHORT_PROSE could not fit its immutable source wrapper")


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
        self.last_rendered_token_counts: tuple[int, ...] = ()
        self.last_published_token_counts: tuple[int, ...] = ()
        self.last_control_records: tuple[dict, ...] = ()
        self.last_work_incomplete = False

    async def transform_tool_batch(self, *, task_ctx, checkpoint: HCheckpoint
                                   ) -> Sequence[ToolObservation]:
        work = WorkRecord()
        observations: list[ToolObservation] = []
        rendered_token_counts: list[int] = []
        published_token_counts: list[int] = []
        self.last_rendered_token_counts = ()
        self.last_published_token_counts = ()
        self.last_control_records = ()
        self.last_work_incomplete = False
        control_records: list[dict] = []
        reduced_siblings = await asyncio.gather(*(
            self._transform_one_tool_call(
                task_ctx=task_ctx,
                tool_call_id=tool_call_id,
                results=results,
            )
            for tool_call_id, results in checkpoint.search_result_sets
        ))
        first_failure: SelectionFailure | None = None
        cancellation: asyncio.CancelledError | None = None
        for (
            observation,
            call_work,
            rendered_counts,
            published_count,
            records,
            failure,
            call_cancellation,
        ) in reduced_siblings:
            work.add(call_work)
            rendered_token_counts.extend(rendered_counts)
            if published_count is not None:
                published_token_counts.append(published_count)
            control_records.extend(records)
            if failure is not None and first_failure is None:
                first_failure = failure
            if call_cancellation is not None and cancellation is None:
                cancellation = call_cancellation
            if observation is not None:
                observations.append(observation)
        if cancellation is not None or first_failure is not None:
            control_records = [
                {
                    **record,
                    "batch_accepted": False,
                    "publication_status": (
                        "FALLBACK_DISCARDED"
                        if record.get("publication_status") == "PUBLISHED"
                        else record.get("publication_status")
                    ),
                }
                for record in control_records
            ]
        self.last_rendered_token_counts = tuple(rendered_token_counts)
        self.last_published_token_counts = tuple(published_token_counts)
        self.last_control_records = tuple(control_records)
        self._finish(work)
        if cancellation is not None:
            self.last_work_incomplete = True
            raise cancellation
        if first_failure is not None:
            self.last_work_incomplete = True
            raise PageSelectionError(first_failure)
        return observations

    async def _transform_one_tool_call(self, *, task_ctx, tool_call_id: str, results):
        if self.config.scope == "whole_batch":
            return await self._transform_whole_batch(
                task_ctx=task_ctx, tool_call_id=tool_call_id, results=results)
        # Keep one slot per vendor-visible result. Calls within and across sibling tool calls
        # execute concurrently, but publication order remains the frozen vendor order.
        rendered_slots: list[str | None] = [None] * len(results)
        occurrence_slots: list[str | None] = [None] * len(results)
        calls = []
        call_slots: list[int] = []
        call_occurrences: list[str] = []
        for result_index, result in enumerate(results):
            if result.raw_content_id is None:
                rendered_slots[result_index] = _vendor_snippet(result)
                if result.source_occurrence_id:
                    occurrence_slots[result_index] = str(result.source_occurrence_id)
                continue
            text = self._raw_text_for(result.raw_content_id)
            ch = sha256_hex(text.encode("utf-8"))
            occurrence_id = (
                result.source_occurrence_id
                or self._occurrence_for(result.raw_content_id)
            )
            spans = spans_from_page(
                text,
                content_hash=ch,
                occurrence_id=occurrence_id,
                chunker=self.config.chunker,
                tokenizer=self._tokenizer,
                max_tokens=self.config.chunk_max_tokens,
            )
            if not spans:
                continue
            view = CandidateViewRecord.build(
                spans=spans,
                tokenizer=self._tokenizer,
                namespace="RAW_SOURCE",
                topic=getattr(task_ctx, "research_topic", ""),
                # Candidate construction remains the exact H02 P1_ID path; only the output
                # instruction/language differs in ShortProseSelector.
                contract="P1_ID",
                token_budget=self.config.token_budget,
                query_attempts=[],
                snapshot_texts={ch: text},
                source_meta={
                    span["span_id"]: {
                        "title": result.title,
                        "url": result.url,
                    }
                    for span in spans
                },
            )
            calls.append(self._summarize_one(
                task_ctx=task_ctx,
                view=view,
                occurrence_id=occurrence_id,
                title=result.title,
                url=result.url,
                source_number=result.vendor_visible_order + 1,
            ))
            call_slots.append(result_index)
            call_occurrences.append(str(occurrence_id))

        completed = await asyncio.gather(*calls, return_exceptions=True)
        work = WorkRecord()
        rendered_counts: list[int] = []
        records: list[dict] = []
        first_exception: BaseException | None = None
        cancellation: asyncio.CancelledError | None = None
        for result_index, occurrence_id, completed_call in zip(
            call_slots, call_occurrences, completed, strict=True
        ):
            if isinstance(completed_call, asyncio.CancelledError):
                work.add(_work_from_exception(completed_call))
                control_record = getattr(
                    completed_call, "prose_control_record", None
                )
                if isinstance(control_record, dict):
                    records.append(dict(control_record))
                if cancellation is None:
                    cancellation = completed_call
                continue
            if isinstance(completed_call, BaseException):
                if not isinstance(completed_call, Exception):
                    raise completed_call
                if first_exception is None:
                    first_exception = completed_call
                work.add(_work_from_exception(completed_call))
                control_record = getattr(
                    completed_call, "control_record", None
                )
                if isinstance(control_record, dict) and control_record:
                    records.append(dict(control_record))
                continue
            bounded, call_work, record = completed_call
            work.add(call_work)
            rendered_slots[result_index] = bounded
            occurrence_slots[result_index] = occurrence_id
            rendered_counts.append(int(record["rendered_tokens"]))
            records.append(record)

        if cancellation is not None or first_exception is not None:
            records = [
                {
                    **record,
                    "batch_accepted": False,
                    "publication_status": (
                        "FALLBACK_DISCARDED"
                        if record.get("publication_status") == "PUBLISHED"
                        else record.get("publication_status")
                    ),
                }
                for record in records
            ]
        if cancellation is not None:
            return None, work, rendered_counts, None, records, None, cancellation
        if first_exception is not None:
            return (
                None,
                work,
                rendered_counts,
                None,
                records,
                SelectionFailure(
                    "PROSE_CONTROL_ERROR",
                    f"{type(first_exception).__name__}: {first_exception}",
                ),
                None,
            )
        rendered = [value for value in rendered_slots if value is not None]
        published = _format(rendered)
        published_tokens = self._tokenizer.count(published)
        return (
            ToolObservation(
                tool_call_id=tool_call_id,
                name="tavily_search",
                content=published,
                source_occurrence_ids=tuple(dict.fromkeys(
                    occurrence_id
                    for rendered_value, occurrence_id in zip(
                        rendered_slots, occurrence_slots, strict=True
                    )
                    if rendered_value is not None and occurrence_id
                )),
            ),
            work,
            rendered_counts,
            published_tokens,
            records,
            None,
            None,
        )

    async def _transform_whole_batch(self, *, task_ctx, tool_call_id: str, results):
        """One prose summary for the whole gather batch, under one completion cap.

        The per-page path builds a view and a request per result. This builds one view over
        every page's spans and issues one request, which is what makes it the matched control
        for a whole-batch selector: both forms then see the same evidence and are held to the
        same rendered-token budget, and the only difference is prose against span ids.

        Snippet-only results still pass through as vendor rendered them -- they carry no raw
        content for either form to compress, so summarising them would be inventing evidence P0
        never had.
        """
        spans: list[dict] = []
        snapshot_texts: dict[str, str] = {}
        source_meta: dict = {}
        sources: list[tuple[int, str, str]] = []
        occurrence_ids: list[str] = []
        passthrough: list[str] = []

        for result in results:
            if result.raw_content_id is None:
                passthrough.append(_vendor_snippet(result))
                if result.source_occurrence_id:
                    occurrence_ids.append(str(result.source_occurrence_id))
                continue
            text = self._raw_text_for(result.raw_content_id)
            content_hash = sha256_hex(text.encode("utf-8"))
            occurrence_id = (result.source_occurrence_id
                             or self._occurrence_for(result.raw_content_id))
            page_spans = spans_from_page(
                text, content_hash=content_hash, occurrence_id=occurrence_id,
                chunker=self.config.chunker, tokenizer=self._tokenizer,
                max_tokens=self.config.chunk_max_tokens,
            )
            if not page_spans:
                continue
            spans.extend(page_spans)
            snapshot_texts[content_hash] = text
            source_meta.update({
                span["span_id"]: {"title": result.title, "url": result.url}
                for span in page_spans
            })
            sources.append((result.vendor_visible_order + 1, result.title, result.url))
            occurrence_ids.append(str(occurrence_id))

        if not spans:
            published = _format(passthrough)
            return (
                ToolObservation(tool_call_id=tool_call_id, name="tavily_search",
                                content=published,
                                source_occurrence_ids=tuple(dict.fromkeys(occurrence_ids))),
                WorkRecord(), [], self._tokenizer.count(published), [], None, None,
            )

        view = CandidateViewRecord.build(
            spans=spans, tokenizer=self._tokenizer, namespace="RAW_SOURCE",
            topic=getattr(task_ctx, "research_topic", ""),
            # Candidate construction stays the exact structured path; only the output
            # instruction differs, so the contrast is the output form and nothing else.
            contract="P1_ID", token_budget=self.config.token_budget,
            query_attempts=[], snapshot_texts=snapshot_texts, source_meta=source_meta,
        )
        try:
            bounded, work, record = await self._summarize_batch(
                task_ctx=task_ctx, view=view, sources=sources,
                occurrence_ids=tuple(dict.fromkeys(occurrence_ids)))
        except asyncio.CancelledError as cancelled:
            return None, _work_from_exception(cancelled), [], None, [], None, cancelled
        except Exception as failure:  # noqa: BLE001
            control_record = getattr(failure, "control_record", None)
            records = [dict(control_record)] if isinstance(control_record, dict) else []
            return (
                None, _work_from_exception(failure), [], None, records,
                SelectionFailure("PROSE_CONTROL_ERROR",
                                 f"{type(failure).__name__}: {failure}"), None,
            )

        published = _format([bounded, *passthrough])
        return (
            ToolObservation(tool_call_id=tool_call_id, name="tavily_search", content=published,
                            source_occurrence_ids=tuple(dict.fromkeys(occurrence_ids))),
            work, [int(record["rendered_tokens"])], self._tokenizer.count(published),
            [record], None, None,
        )

    async def _summarize_batch(self, *, task_ctx, view, sources, occurrence_ids=()):
        """One model call for the batch, capped so the source map is never decoded away."""
        prose_prompt = self._selector.prompt_for(
            task_ctx=task_ctx, view=view, token_budget=self.config.token_budget)
        prefix, suffix = _batch_prose_wrapper(sources)
        completion_cap = _completion_cap(
            immutable_wrapper=prefix + suffix, tokenizer=self._tokenizer,
            budget=self.config.token_budget)
        base_record = {
            "candidate_view_sha256": view.view_sha256,
            "prose_prompt_sha256": sha256_hex(prose_prompt.encode("utf-8")),
            "offered_span_ids": [c.span_id for c in view.candidates],
            "completion_token_cap": completion_cap,
            "sources": len(sources),
            # A prose summary publishes no span ids, so source identity is the only thing the
            # judge-free endpoints can be computed from for this arm. Published equals offered
            # by construction: the source map cites every page the summary was built from, which
            # is exactly what makes this control citable rather than a straw man.
            "offered_source_occurrence_ids": list(occurrence_ids),
            "published_source_occurrence_ids": list(occurrence_ids),
            "selector_attempted": True,
            "chunker": self.config.chunker,
            "scope": self.config.scope,
            "tokenizer_sha256": tokenizer_sha256(self._tokenizer),
        }
        # The cap is what makes the rendered total fit: the wrapper is charged first, so
        # prefix + body + suffix is at or under the budget by construction and nothing has to be
        # truncated after the fact. The engine enforces it as max_tokens.
        body, work = await self._selector.summarize(
            task_ctx=task_ctx, view=view, token_budget=self.config.token_budget,
            max_completion_tokens=completion_cap)
        rendered = prefix + body + suffix
        record = {
            **base_record,
            "rendered_tokens": self._tokenizer.count(rendered),
            "publication_status": "PUBLISHED",
            "batch_accepted": True,
            "work": {
                "selector_calls": work.selector_calls,
                "prompt_tokens": work.prompt_tokens,
                "completion_tokens": work.completion_tokens,
                "cpu_seconds": round(work.cpu_seconds, 6),
                "retries": work.retries,
            },
        }
        return rendered, work, record

    async def _summarize_one(
        self,
        *,
        task_ctx,
        view,
        occurrence_id: str,
        title: str,
        url: str,
        source_number: int,
    ):
        prose_prompt = self._selector.prompt_for(
            task_ctx=task_ctx,
            view=view,
            token_budget=self.config.token_budget,
        )
        prefix, suffix = _page_prose_wrapper(
            title=title,
            url=url,
            source_number=source_number,
        )
        completion_cap = _completion_cap(
            immutable_wrapper=prefix + suffix,
            tokenizer=self._tokenizer,
            budget=self.config.token_budget,
        )
        base_record = {
            "candidate_view_sha256": view.view_sha256,
            "prose_prompt_sha256": sha256_hex(prose_prompt.encode("utf-8")),
            "offered_span_ids": [
                candidate.span_id for candidate in view.candidates
            ],
            "offered_source_occurrence_ids": [str(occurrence_id)],
            "completion_token_cap": completion_cap,
            "source_number": int(source_number),
            "title_sha256": sha256_hex(title.encode("utf-8")),
            "url_sha256": sha256_hex(url.encode("utf-8")),
            "selector_attempted": True,
            "chunker": self.config.chunker,
            "scope": self.config.scope,
            "tokenizer_sha256": tokenizer_sha256(self._tokenizer),
        }
        try:
            text, call_work = await self._selector.summarize(
                task_ctx=task_ctx,
                view=view,
                token_budget=self.config.token_budget,
                max_completion_tokens=completion_cap,
            )
        except asyncio.CancelledError as exc:
            exc.prose_control_record = {
                **base_record,
                "attempt_status": "CANCELLED",
                "normalization_trace_status": "EXPLICIT_NO_OUTPUT",
                "publication_status": "FALLBACK_DISCARDED",
                "batch_accepted": False,
                "work_incomplete": True,
                "normalization": None,
            }
            raise
        except Exception as exc:  # noqa: BLE001
            raise _ProseAttemptError(
                exc,
                work=_work_from_exception(exc),
                control_record={
                    **base_record,
                    "attempt_status": "CALL_FAILED",
                    "normalization_trace_status": "EXPLICIT_NO_OUTPUT",
                    "publication_status": "FALLBACK_DISCARDED",
                    "batch_accepted": False,
                    "work_incomplete": True,
                    "normalization": None,
                },
            ) from exc
        try:
            rendered, bounded_body, normalization = _render_page_prose(
                text,
                title=title,
                url=url,
                source_number=source_number,
                tokenizer=self._tokenizer,
                budget=self.config.token_budget,
            )
        except _ProseValidationError as exc:
            raise _ProseAttemptError(
                exc,
                work=call_work,
                control_record={
                    **base_record,
                    "attempt_status": "OUTPUT_OBSERVED",
                    "normalization_trace_status": "OK",
                    "publication_status": "REJECTED",
                    "batch_accepted": False,
                    "work_incomplete": False,
                    "normalization": exc.normalization,
                },
            ) from exc
        except Exception as exc:
            raise _ProseAttemptError(
                exc,
                work=call_work,
                control_record={
                    **base_record,
                    "attempt_status": "OUTPUT_OBSERVED",
                    "normalization_trace_status": "MISSING",
                    "publication_status": "REJECTED",
                    "batch_accepted": False,
                    "work_incomplete": False,
                    "normalization": None,
                },
            ) from exc
        rendered_tokens = self._tokenizer.count(rendered)
        return rendered, call_work, {
            **base_record,
            "rendered_tokens": rendered_tokens,
            "prose_body_tokens": self._tokenizer.count(bounded_body),
            "attempt_status": "OUTPUT_OBSERVED",
            "normalization_trace_status": "OK",
            "publication_status": "PUBLISHED",
            "batch_accepted": True,
            "work_incomplete": False,
            "normalization": normalization,
        }

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
        self.last_rendered_token_counts: tuple[int, ...] = ()
        self.last_published_token_counts: tuple[int, ...] = ()
        self.last_control_records: tuple[dict, ...] = ()
        self.last_work_incomplete = False

    async def close_researcher(self, *, task_ctx, checkpoint: CCheckpoint) -> ResearcherHandoff:
        self.last_work = WorkRecord()
        self.last_work_incomplete = False
        self.last_rendered_token_counts = ()
        self.last_published_token_counts = ()
        self.last_control_records = ()
        view_obj = build_visible_view(checkpoint.researcher_messages)
        try:
            source_registry = build_visible_source_registry(view_obj)
        except VisibleSourceRegistryError as exc:
            raise CloseSelectionError(
                SelectionFailure("VISIBLE_SOURCE_REGISTRY", str(exc))
            ) from exc
        spans = spans_from_visible_view(
            view_obj.view_bytes, view_hash=view_obj.view_hash,
            messages=source_partitioned_message_segments(
                view_obj, source_registry
            ),
            tokenizer=self._tokenizer,
            max_tokens=self.config.chunk_max_tokens,
            chunker=self.config.chunker)
        if not spans:
            raise CloseSelectionError(SelectionFailure("NO_VISIBLE_SPANS", "empty compressor view"))
        try:
            source_meta = source_registry.source_meta_for(spans)
        except VisibleSourceRegistryError as exc:
            raise CloseSelectionError(
                SelectionFailure("VISIBLE_SOURCE_REGISTRY", str(exc))
            ) from exc
        source_entries = source_registry.entries
        view = CandidateViewRecord.build(
            spans=spans, tokenizer=self._tokenizer, namespace="VISIBLE_MESSAGE",
            topic=getattr(task_ctx, "research_topic", ""), contract="P1_ID",
            token_budget=self.config.token_budget,
            query_attempts=[
                (query_attempt_id, query_attempt_id)
                for query_attempt_id in checkpoint.query_attempt_ids
            ],
            visible_views={view_obj.view_hash: view_obj.view_bytes},
            source_meta=source_meta,
        )
        prompt_source_entries = tuple(
            (entry.label, entry.title, entry.url) for entry in source_entries
        )
        try:
            completion_cap = _completion_cap(
                immutable_wrapper=_minimum_close_wrapper(
                    source_entries, self._tokenizer
                ),
                tokenizer=self._tokenizer,
                budget=self.config.token_budget,
            )
        except ValueError as exc:
            raise CloseSelectionError(
                SelectionFailure("PROSE_CONTROL_ERROR", str(exc))
            ) from exc
        prose_prompt = self._selector.prompt_for(
            task_ctx=task_ctx,
            view=view,
            token_budget=self.config.token_budget,
            source_entries=prompt_source_entries,
        )
        offered_source_occurrence_ids = sorted({
            str(occurrence_id)
            for span in spans
            for occurrence_id in (span.get("source_occurrence_ids") or ())
        })
        base_record = {
            "candidate_view_sha256": view.view_sha256,
            "source_affordance_binding_sha256": view.source_meta_sha256,
            "prose_prompt_sha256": sha256_hex(prose_prompt.encode("utf-8")),
            "offered_span_ids": [
                candidate.span_id for candidate in view.candidates
            ],
            "offered_source_occurrence_ids": offered_source_occurrence_ids,
            "completion_token_cap": completion_cap,
            "selector_attempted": True,
            "chunker": self.config.chunker,
            "scope": self.config.scope,
            "tokenizer_sha256": tokenizer_sha256(self._tokenizer),
        }
        try:
            text, work = await self._selector.summarize(
                task_ctx=task_ctx,
                view=view,
                token_budget=self.config.token_budget,
                max_completion_tokens=completion_cap,
                source_entries=prompt_source_entries,
            )
        except asyncio.CancelledError as error:
            work = _work_from_exception(error)
            self.last_work_incomplete = True
            self.last_control_records = ({
                **base_record,
                "attempt_status": "CANCELLED",
                "normalization_trace_status": "EXPLICIT_NO_OUTPUT",
                "publication_status": "FALLBACK_DISCARDED",
                "batch_accepted": False,
                "work_incomplete": True,
                "normalization": None,
            },)
            self._finish(work)
            raise
        except Exception as e:  # noqa: BLE001
            work = _work_from_exception(e)
            self.last_work_incomplete = True
            self.last_control_records = ({
                **base_record,
                "attempt_status": "CALL_FAILED",
                "normalization_trace_status": "EXPLICIT_NO_OUTPUT",
                "publication_status": "FALLBACK_DISCARDED",
                "batch_accepted": False,
                "work_incomplete": True,
                "normalization": None,
            },)
            self._finish(work)
            raise CloseSelectionError(
                SelectionFailure("PROSE_CONTROL_ERROR", f"{type(e).__name__}: {e}")) from e
        self._finish(work)
        try:
            (
                bounded,
                bounded_body,
                used_source_entries,
                normalization,
            ) = _render_close_prose(
                text,
                source_entries=source_entries,
                tokenizer=self._tokenizer,
                budget=self.config.token_budget,
            )
        except _ProseValidationError as exc:
            self.last_control_records = ({
                **base_record,
                "attempt_status": "OUTPUT_OBSERVED",
                "normalization_trace_status": "OK",
                "publication_status": "REJECTED",
                "batch_accepted": False,
                "work_incomplete": False,
                "normalization": exc.normalization,
            },)
            raise CloseSelectionError(
                SelectionFailure("PROSE_CONTROL_ERROR", str(exc))
            ) from exc
        except ValueError as exc:
            self.last_control_records = ({
                **base_record,
                "attempt_status": "OUTPUT_OBSERVED",
                "normalization_trace_status": "MISSING",
                "publication_status": "REJECTED",
                "batch_accepted": False,
                "work_incomplete": False,
                "normalization": None,
            },)
            raise CloseSelectionError(
                SelectionFailure("PROSE_CONTROL_ERROR", str(exc))
            ) from exc
        rendered_tokens = self._tokenizer.count(bounded)
        self.last_rendered_token_counts = (rendered_tokens,)
        # C publishes the bounded prose itself; unlike H it has no common outer tool-message
        # wrapper, so materialized and published token counts are identical.
        self.last_published_token_counts = (rendered_tokens,)
        used_source_occurrence_ids = tuple(dict.fromkeys(
            occurrence_id
            for entry in used_source_entries
            for occurrence_id in entry.occurrence_ids
        ))
        visible_source_map = "\n".join(
            entry.map_line for entry in source_entries
        )
        used_source_map = "\n".join(
            entry.map_line for entry in used_source_entries
        )
        self.last_control_records = ({
            **base_record,
            "used_source_occurrence_ids": list(used_source_occurrence_ids),
            "used_citation_labels": [
                entry.label for entry in used_source_entries
            ],
            "rendered_tokens": rendered_tokens,
            "prose_body_tokens": self._tokenizer.count(bounded_body),
            "visible_source_affordance_count": len(source_entries),
            "visible_source_affordances_sha256": sha256_hex(
                visible_source_map.encode("utf-8")
            ),
            "citable_source_map_sha256": sha256_hex(
                visible_source_map.encode("utf-8")
            ),
            "used_source_map_sha256": sha256_hex(
                used_source_map.encode("utf-8")
            ),
            "published_prose_body_sha256": sha256_hex(
                bounded_body.encode("utf-8")
            ),
            "published_output_sha256": sha256_hex(
                bounded.encode("utf-8")
            ),
            "attempt_status": "OUTPUT_OBSERVED",
            "normalization_trace_status": "OK",
            "publication_status": "PUBLISHED",
            "batch_accepted": True,
            "work_incomplete": False,
            "normalization": normalization,
            "citation_affordance_policy":
                "CAPTURE_TIME_BYTE_RANGE_SOURCE_REGISTRY_V3",
        },)
        return ResearcherHandoff(compressed_research=bounded, raw_notes=(bounded,))

    def _finish(self, work: WorkRecord) -> None:
        self.last_work = work
        if self._work_sink is not None:
            self._work_sink(self.config.variant_id, work)
