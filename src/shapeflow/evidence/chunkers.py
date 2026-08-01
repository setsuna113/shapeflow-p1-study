"""Three deterministic chunkers, each guaranteeing exact reconstruction.

A chunk is a half-open ``[char_start, char_end)`` window into the snapshot text plus some
metadata. The one invariant every chunker upholds: ``text[c.char_start:c.char_end]`` is the
chunk's exact bytes. Chunkers only cut; they never rewrite, strip, or re-encode, because a
selected span is later re-hashed against the snapshot and any drift is treated as corruption.

Tokenization is behind a small interface so the logic is deterministic and testable in the
dev environment with a whitespace tokenizer, while the run host injects the real model
tokenizer for accurate token budgets. Token counts drive budgets; char offsets drive
reconstruction, so the reconstruction guarantee never depends on which tokenizer is used.

The three chunkers (plan §8.3):

- ``fixed_token_v1`` -- fixed token window with configurable overlap. A control, not a
  default winner.
- ``paragraph_sentence_v1`` -- paragraph first, splitting an over-long paragraph by sentence
  then by token window.
- ``markdown_structure_v1`` -- structure-aware: headings, paragraphs, lists, block quotes,
  code fences and tables. A table row always carries its header row, and a list item its
  heading breadcrumb, so structure survives selection without large duplicating overlaps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Protocol

__all__ = [
    "Tokenizer",
    "WhitespaceTokenizer",
    "Chunk",
    "fixed_token_v1",
    "paragraph_sentence_v1",
    "markdown_structure_v1",
]


class Tokenizer(Protocol):
    def encode_offsets(self, text: str) -> list[tuple[int, int]]:
        """Return one ``(char_start, char_end)`` per token, in order."""

    def count(self, text: str) -> int: ...


class WhitespaceTokenizer:
    """A deterministic stand-in: one token per run of non-whitespace. Good enough for dev and
    tests; the run host swaps in the real subword tokenizer via the same interface."""

    _WORD = re.compile(r"\S+")

    def encode_offsets(self, text: str) -> list[tuple[int, int]]:
        return [(m.start(), m.end()) for m in self._WORD.finditer(text)]

    def count(self, text: str) -> int:
        return len(self.encode_offsets(text))


@dataclass(frozen=True)
class Chunk:
    """A half-open ``[char_start, char_end)`` window plus the metadata needed to read it.

    ``heading_path`` and ``context_ranges`` are *metadata*, never part of the span. Both exist
    so a selected fragment can be understood without widening the byte range it addresses:

    - ``heading_path`` is the enclosing heading breadcrumb.
    - ``context_ranges`` are ``(char_start, char_end)`` windows elsewhere in the same source
      that are needed to read this chunk -- currently a table's header row, which names the
      columns a data row's cells belong to.

    Keeping context out of the span is what stops table chunking from going quadratic: if row
    *i* were anchored on the header's start offset, the rows of an n-row table would together
    span O(n^2) bytes and the candidate set would repeat the whole table n times.

    Context is a *range*, not a string, and that matters. Carried as free text it would be an
    unbound channel into the prompt -- tampering with it left the span id unchanged and every
    reconstruction check passing while the injected text was rendered downstream. As a range it
    is addressed and re-hashed exactly like the span itself.
    """

    char_start: int
    char_end: int
    kind: str
    heading_path: tuple[str, ...] = ()
    token_len: int = 0
    context_ranges: tuple[tuple[int, int], ...] = ()
    # The heading lines themselves, as ranges into the same source. heading_path is the
    # convenience view; these are what preflight re-hashes, because a breadcrumb rendered into
    # the prompt is evidence and free strings are an injection channel.
    heading_ranges: tuple[tuple[int, int], ...] = ()

    def text(self, source: str) -> str:
        return source[self.char_start : self.char_end]

    def context_texts(self, source: str) -> tuple[str, ...]:
        return tuple(source[s:e] for s, e in self.context_ranges)


def _finalize(source: str, start: int, end: int, kind: str, tokenizer: Tokenizer,
              heading_path: tuple[str, ...] = (),
              context_ranges: tuple[tuple[int, int], ...] = (),
              heading_ranges: tuple[tuple[int, int], ...] = ()) -> Chunk:
    return Chunk(
        char_start=start,
        char_end=end,
        kind=kind,
        heading_path=heading_path,
        token_len=tokenizer.count(source[start:end]),
        context_ranges=context_ranges,
        heading_ranges=heading_ranges,
    )


# --- fixed_token_v1 -------------------------------------------------------------------


def fixed_token_v1(
    text: str, *, tokenizer: Tokenizer, window: int = 256, overlap: int = 32
) -> list[Chunk]:
    """Slide a fixed token window with overlap. Each chunk spans from its first token's start
    to its last token's end, so reconstruction stays exact."""
    if window <= 0:
        raise ValueError("window must be positive")
    if not 0 <= overlap < window:
        raise ValueError("overlap must be in [0, window)")
    offsets = tokenizer.encode_offsets(text)
    if not offsets:
        return []
    chunks: list[Chunk] = []
    step = window - overlap
    i = 0
    while i < len(offsets):
        window_offsets = offsets[i : i + window]
        start = window_offsets[0][0]
        end = window_offsets[-1][1]
        chunks.append(_finalize(text, start, end, "fixed_window", tokenizer))
        if i + window >= len(offsets):
            break
        i += step
    return chunks


# --- paragraph_sentence_v1 ------------------------------------------------------------

_PARA_SPLIT = re.compile(r"\n[ \t]*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _split_offsets(text: str, pattern: re.Pattern, lo: int, hi: int) -> list[tuple[int, int]]:
    """Split ``text[lo:hi]`` on ``pattern`` boundaries, returning content spans (excluding the
    matched separators) as absolute offsets."""
    spans: list[tuple[int, int]] = []
    cursor = lo
    for m in pattern.finditer(text, lo, hi):
        if m.start() > cursor:
            spans.append((cursor, m.start()))
        cursor = m.end()
    if cursor < hi:
        spans.append((cursor, hi))
    return spans


def paragraph_sentence_v1(
    text: str, *, tokenizer: Tokenizer, max_tokens: int = 256
) -> list[Chunk]:
    """Paragraphs first; an over-long paragraph is split by sentence, then any still-too-long
    sentence by fixed token window. Offsets are preserved throughout."""
    chunks: list[Chunk] = []
    for p_start, p_end in _split_offsets(text, _PARA_SPLIT, 0, len(text)):
        if tokenizer.count(text[p_start:p_end]) <= max_tokens:
            chunks.append(_finalize(text, p_start, p_end, "paragraph", tokenizer))
            continue
        for s_start, s_end in _split_offsets(text, _SENTENCE_END, p_start, p_end):
            if tokenizer.count(text[s_start:s_end]) <= max_tokens:
                chunks.append(_finalize(text, s_start, s_end, "sentence", tokenizer))
            else:
                # Still too long: fall back to fixed token windows over just this sentence.
                for sub in fixed_token_v1(
                    text[s_start:s_end], tokenizer=tokenizer, window=max_tokens, overlap=0
                ):
                    chunks.append(
                        _finalize(text, s_start + sub.char_start, s_start + sub.char_end,
                                  "sentence", tokenizer)
                    )
    return chunks


# --- markdown_structure_v1 ------------------------------------------------------------

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")
_LIST_ITEM = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+")
_FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass
class _Line:
    start: int
    end: int  # end of content, before the newline
    nl_end: int  # end including the newline
    text: str


def _iter_lines(text: str) -> list[_Line]:
    lines: list[_Line] = []
    pos = 0
    n = len(text)
    while pos < n:
        nl = text.find("\n", pos)
        if nl == -1:
            lines.append(_Line(pos, n, n, text[pos:n]))
            break
        lines.append(_Line(pos, nl, nl + 1, text[pos:nl]))
        pos = nl + 1
    return lines


def markdown_structure_v1(
    text: str, *, tokenizer: Tokenizer, max_tokens: int = 320
) -> list[Chunk]:
    """Structure-aware chunking over a line-based parse.

    Hand-rolled rather than delegated so char offsets are exact and table-header association
    is explicit. A table row chunk spans the row alone and carries the header row in its
    ``context``, so a selected row is never orphaned from the columns that give it meaning
    without the span itself growing with the row index.

    **Every** block kind goes through ``emit_block``, so ``max_tokens`` is a real ceiling.
    Headings, fenced code and table rows used to be finalized directly and could therefore
    exceed the cap by an unbounded amount -- and because the aggregator budgets in rendered
    tokens, an over-cap chunk smuggles text past the very budget that defines the treatment.
    """
    lines = _iter_lines(text)
    chunks: list[Chunk] = []
    heading_stack: list[tuple[int, str, int, int]] = []  # (level, title, start, end)
    i = 0

    def breadcrumb() -> tuple[str, ...]:
        return tuple(title for _, title, _, _ in heading_stack)

    def breadcrumb_ranges() -> tuple[tuple[int, int], ...]:
        return tuple((s_, e_) for _, _, s_, e_ in heading_stack)

    def emit_block(start: int, end: int, kind: str,
                   context_ranges: tuple[tuple[int, int], ...] = ()) -> None:
        if end <= start:
            return
        block = text[start:end]
        if tokenizer.count(block) <= max_tokens:
            chunks.append(
                _finalize(text, start, end, kind, tokenizer, breadcrumb(), context_ranges,
                          breadcrumb_ranges())
            )
            return
        subs = paragraph_sentence_v1(block, tokenizer=tokenizer, max_tokens=max_tokens)
        if not subs:
            # A single unbreakable run (no paragraph or sentence boundary, e.g. one long
            # heading line or a fence with no blank lines) still has to respect the cap.
            subs = fixed_token_v1(block, tokenizer=tokenizer, window=max_tokens, overlap=0)
        for sub in subs:
            chunks.append(
                _finalize(text, start + sub.char_start, start + sub.char_end,
                          kind, tokenizer, breadcrumb(), context_ranges, breadcrumb_ranges())
            )

    while i < len(lines):
        line = lines[i]
        stripped = line.text.strip()

        # blank line
        if not stripped:
            i += 1
            continue

        # heading
        m = _HEADING.match(line.text)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title, line.start, line.end))
            emit_block(line.start, line.end, "heading")
            i += 1
            continue

        # fenced code block
        fence = _FENCE.match(line.text)
        if fence:
            marker = fence.group(1)
            start = line.start
            j = i + 1
            while j < len(lines) and marker not in lines[j].text:
                j += 1
            end = lines[j].end if j < len(lines) else lines[-1].end
            emit_block(start, end, "code")
            i = j + 1
            continue

        # table: a header line followed by a separator row of dashes
        if "|" in line.text and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1].text):
            header = lines[i]
            sep = lines[i + 1]
            j = i + 2
            row_lines = []
            while j < len(lines) and "|" in lines[j].text and lines[j].text.strip():
                row_lines.append(lines[j])
                j += 1
            # The header is emitted once as its own chunk, and every row chunk spans only that
            # row while naming the header in `context`. Reconstruction stays exact (the span is
            # a contiguous slice of the source) and the columns still travel with the cells.
            emit_block(header.start, sep.end, "table_header")
            header_range = ((header.start, header.end),)
            for row in row_lines:
                emit_block(row.start, row.end, "table_row", header_range)
            i = j
            continue

        # blockquote
        if stripped.startswith(">"):
            start = line.start
            j = i
            while j < len(lines) and lines[j].text.strip().startswith(">"):
                j += 1
            emit_block(start, lines[j - 1].end, "blockquote")
            i = j
            continue

        # list: consecutive list items form a block; each item is its own chunk
        if _LIST_ITEM.match(line.text):
            j = i
            while j < len(lines):
                lt = lines[j].text
                if _LIST_ITEM.match(lt):
                    # find extent of this item (until next list item or blank line)
                    k = j + 1
                    while k < len(lines) and lines[k].text.strip() and not _LIST_ITEM.match(lines[k].text):
                        k += 1
                    emit_block(lines[j].start, lines[k - 1].end, "list_item")
                    j = k
                elif not lt.strip():
                    break
                else:
                    break
            i = j
            continue

        # paragraph: consecutive non-blank, non-structural lines
        j = i
        while j < len(lines):
            lt = lines[j].text
            if (not lt.strip() or _HEADING.match(lt) or _FENCE.match(lt)
                    or _LIST_ITEM.match(lt) or lt.strip().startswith(">")):
                break
            j += 1
        emit_block(line.start, lines[j - 1].end, "paragraph")
        i = j

    return chunks
