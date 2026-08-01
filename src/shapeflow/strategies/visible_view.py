"""The VisibleCompressorView: exactly the bytes P0's compressor read, and where each one came from.

This object is what makes "compressor-only" checkable instead of asserted. It is built by
rendering the frozen ``researcher_messages`` the same way the compressor's prompt renders them,
and it records each message's byte range inside the result -- so every C_VISIBLE span addresses
a real offset in a real byte string that a real compressor would have seen, and preflight can
re-derive it.

Two rules are enforced here rather than downstream:

**A message's kind is the message's own.** A ToolMessage is ``TOOL_EVIDENCE`` only when its
capture-time artifact names source occurrences; otherwise it is
``TOOL_UNATTRIBUTED_CONTEXT``. An AIMessage is ``MODEL_DERIVED_CONTEXT``; a HumanMessage is
``USER_CONTEXT``. Neither tool origin nor model selection can manufacture citation provenance.

**Occurrence ids come from the tool call, not from a reverse lookup.** Mapping a
model-written page summary back to the raw page it summarised is exactly the smuggling that
distinguishes C_REGISTRY from C_VISIBLE, so a tool message carries only the occurrences its own
call recorded.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from ..hashing import sha256_hex
from ..odr.checkpoints import FrozenMessage

__all__ = [
    "VisibleCompressorView",
    "VisibleSourceEntry",
    "VisibleSourceRegistry",
    "VisibleSourceRegistryError",
    "build_visible_source_registry",
    "build_visible_view",
    "source_partitioned_message_segments",
    "KIND_BY_ROLE",
]

KIND_BY_ROLE = {
    "ai": "MODEL_DERIVED_CONTEXT",
    "human": "USER_CONTEXT",
    "system": "USER_CONTEXT",
}

_SOURCE_DELIMITER = re.compile(
    r"^---\s*SOURCE\s+[^:\r\n]+:\s*(?P<title>.*?)\s*---$",
    re.IGNORECASE,
)
_RENDERED_SOURCE = re.compile(r"^SOURCE:\s*(?P<rest>\S.*)$", re.IGNORECASE)
_VISIBLE_SOURCE_URL = re.compile(
    r"^URL:\s*(?P<url>https?://[^\s<>\]\)]+)\s*$",
    re.IGNORECASE,
)
_ANY_URL = re.compile(r"https?://[^\s<>\]\)]+", re.IGNORECASE)


class VisibleSourceRegistryError(ValueError):
    """Capture-time bytes cannot identify one unambiguous source association."""


@dataclass(frozen=True)
class VisibleSourceEntry:
    """One title/URL affordance addressed into the exact compressor-visible bytes."""

    label: str
    title: str
    url: str
    occurrence_ids: tuple[str, ...]
    message_id: str
    visible_compressor_view_hash: str
    byte_start: int
    byte_end: int
    title_byte_start: int
    title_byte_end: int
    url_byte_start: int
    url_byte_end: int

    @property
    def map_line(self) -> str:
        return f"[{self.label}] {self.title} -- {self.url}"

    def source_meta(self) -> dict:
        """Renderer metadata plus the byte addresses that make it provenance, not a hint."""

        return {
            "binding_version": "visible_source_byte_range_v1",
            "title": self.title,
            "url": self.url,
            "message_id": self.message_id,
            "visible_compressor_view_hash": self.visible_compressor_view_hash,
            "byte_start": self.byte_start,
            "byte_end": self.byte_end,
            "title_byte_start": self.title_byte_start,
            "title_byte_end": self.title_byte_end,
            "url_byte_start": self.url_byte_start,
            "url_byte_end": self.url_byte_end,
            "source_occurrence_ids": list(self.occurrence_ids),
            "source_label": self.label,
        }


@dataclass(frozen=True)
class VisibleSourceRegistry:
    """The one source-affordance object shared by structured C and C SHORT_PROSE."""

    entries: tuple[VisibleSourceEntry, ...] = ()
    # Includes title-only/no-URL blocks. They remain TOOL_EVIDENCE with occurrence lineage
    # during chunking, but do not become citation affordances.
    partitions: tuple[VisibleSourceEntry, ...] = ()

    def source_meta_for(self, spans: Sequence[dict]) -> dict[str, dict]:
        """Bind a span only when exactly one frozen source range wholly contains it.

        A chunk that crosses two source blocks (or starts outside one and ends inside it) has
        no honest title/URL owner. Guessing by nearest header would make the structured arm
        look better than the prose control, so such a checkpoint is rejected before decode.
        """

        metadata: dict[str, dict] = {}
        for span in spans:
            if span.get("namespace") != "VISIBLE_MESSAGE":
                continue
            if span.get("kind") != "TOOL_EVIDENCE":
                continue
            span_id = str(span.get("visible_span_id") or span.get("span_id") or "")
            start = int(span["byte_start"])
            end = int(span["byte_end"])
            same_message = [
                entry
                for entry in self.entries
                if entry.message_id == str(span.get("message_id") or "")
            ]
            overlapping = [
                entry
                for entry in same_message
                if start < entry.byte_end and entry.byte_start < end
            ]
            containing = [
                entry
                for entry in overlapping
                if entry.byte_start <= start and end <= entry.byte_end
            ]
            if overlapping and len(containing) != 1:
                raise VisibleSourceRegistryError(
                    f"visible span {span_id[:12]} crosses or ambiguously overlaps "
                    f"{len(overlapping)} capture-time source ranges"
                )
            if len(containing) > 1:
                raise VisibleSourceRegistryError(
                    f"visible span {span_id[:12]} belongs to multiple capture-time sources"
                )
            if containing:
                entry = containing[0]
                span_occurrences = tuple(
                    str(value)
                    for value in (span.get("source_occurrence_ids") or ())
                )
                if (
                    not span_occurrences
                    or not set(entry.occurrence_ids).issubset(span_occurrences)
                ):
                    raise VisibleSourceRegistryError(
                        f"visible span {span_id[:12]} source range disagrees with its "
                        "capture-time occurrence lineage"
                    )
                metadata[span_id] = entry.source_meta()
        return metadata


@dataclass(frozen=True)
class _ParsedSourceBlock:
    title: str
    url: str
    byte_start: int
    byte_end: int
    title_byte_start: int
    title_byte_end: int
    url_byte_start: int
    url_byte_end: int


@dataclass(frozen=True)
class VisibleCompressorView:
    """The rendered bytes, their digest, and the per-message segments inside them."""

    view_bytes: bytes
    view_hash: str
    message_segments: tuple[dict, ...] = field(default_factory=tuple)

    @property
    def byte_len(self) -> int:
        return len(self.view_bytes)


def _render_content(content) -> str:
    """Flatten a message's content the way a chat template lays it out.

    Structured blocks are rendered block by block rather than repr'd: the compressor sees the
    text of each block, and hashing a Python repr would make the view depend on our own
    serialisation rather than on what the model was shown.
    """
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or ():
        if isinstance(block, dict):
            parts.append(str(block.get("text") or block.get("content") or ""))
        else:
            parts.append(str(block))
    return "".join(parts)


def build_visible_view(messages: Sequence[FrozenMessage]) -> VisibleCompressorView:
    """Render the frozen messages into the compressor's visible bytes.

    Ordering and separators are fixed here and enter the view hash, so two runs that saw the
    same conversation produce the same view -- and any change to how we render it changes the
    hash rather than silently changing what C_VISIBLE was allowed to address.
    """
    chunks: list[bytes] = []
    segments: list[dict] = []
    cursor = 0
    for index, message in enumerate(messages):
        header = f"<{message.role}>\n".encode()
        body = _render_content(message.content).encode("utf-8")
        footer = f"\n</{message.role}>\n".encode()
        chunks.extend((header, body, footer))
        body_start = cursor + len(header)
        occurrences = tuple(_occurrences_of(message)) if message.role == "tool" else ()
        kind = (
            "TOOL_EVIDENCE" if message.role == "tool" and occurrences
            else "TOOL_UNATTRIBUTED_CONTEXT" if message.role == "tool"
            else KIND_BY_ROLE.get(message.role, "USER_CONTEXT")
        )
        segments.append({
            "message_id": message.message_id or f"m{index}",
            "role": message.role,
            "kind": kind,
            "byte_start": body_start,
            "byte_end": body_start + len(body),
            # Only a tool message carries occurrences, and only the ones its own call recorded.
            # A reverse lookup from a model-written summary back to the page it described is the
            # smuggling that separates C_REGISTRY from C_VISIBLE.
            "occurrence_ids": list(occurrences),
        })
        cursor = body_start + len(body) + len(footer)

    view_bytes = b"".join(chunks)
    return VisibleCompressorView(
        view_bytes=view_bytes,
        view_hash=sha256_hex(view_bytes),
        message_segments=tuple(segments),
    )


def _trimmed_group_span(line: str, start: int, end: int) -> tuple[int, int]:
    while start < end and line[start].isspace():
        start += 1
    while end > start and line[end - 1].isspace():
        end -= 1
    return start, end


def _safe_title(value: str) -> str:
    """Keep title display one-line without allowing a second URL affordance."""

    value = str(value)
    if _ANY_URL.search(value):
        raise VisibleSourceRegistryError(
            "a capture-time source title contains a URL and cannot be separated "
            "unambiguously from its URL field"
        )
    # The parser already supplies one physical line. Preserve internal whitespace so the title
    # still reconstructs from one contiguous byte range.
    return value.strip(" \t—-") or "(untitled source)"


def _source_blocks(
    body: str, *, body_byte_start: int
) -> tuple[_ParsedSourceBlock, ...]:
    """Parse source-shaped blocks while retaining their exact byte coordinates."""

    lines = body.splitlines(keepends=True)
    line_char_starts: list[int] = []
    cursor = 0
    for line in lines:
        line_char_starts.append(cursor)
        cursor += len(line)

    starts: list[dict] = []
    index = 0
    while index < len(lines):
        raw_line = lines[index].rstrip("\r\n")
        line_char_start = line_char_starts[index]
        delimiter = _SOURCE_DELIMITER.fullmatch(raw_line)
        if delimiter:
            title_start, title_end = _trimmed_group_span(
                raw_line, *delimiter.span("title")
            )
            title_raw = raw_line[title_start:title_end]
            url = ""
            url_char_start = url_char_end = line_char_start + len(raw_line)
            if index + 1 < len(lines):
                next_raw = lines[index + 1].rstrip("\r\n")
                url_match = _VISIBLE_SOURCE_URL.fullmatch(next_raw)
                if url_match:
                    local_start, local_end = _trimmed_group_span(
                        next_raw, *url_match.span("url")
                    )
                    url = next_raw[local_start:local_end]
                    url_char_start = line_char_starts[index + 1] + local_start
                    url_char_end = line_char_starts[index + 1] + local_end
                    index += 1
            starts.append({
                "char_start": line_char_start,
                "title": _safe_title(title_raw),
                "title_char_start": line_char_start + title_start,
                "title_char_end": line_char_start + title_end,
                "url": url,
                "url_char_start": url_char_start,
                "url_char_end": url_char_end,
            })
            index += 1
            continue

        rendered = _RENDERED_SOURCE.fullmatch(raw_line)
        if rendered:
            rest_start, rest_end = _trimmed_group_span(
                raw_line, *rendered.span("rest")
            )
            rest = raw_line[rest_start:rest_end]
            url_match = _ANY_URL.search(rest)
            if url_match:
                url = url_match.group(0)
                title_fragment = rest[:url_match.start()]
                title = _safe_title(title_fragment)
                # `_safe_title` only strips the trailing separator on this branch. Locate the
                # exact retained title rather than claiming the separator as title bytes.
                title_local = rest.find(title) if title != "(untitled source)" else 0
                title_char_start = line_char_start + rest_start + title_local
                title_char_end = title_char_start + (
                    len(title) if title != "(untitled source)" else 0
                )
                starts.append({
                    "char_start": line_char_start,
                    "title": title,
                    "title_char_start": title_char_start,
                    "title_char_end": title_char_end,
                    "url": url,
                    "url_char_start": (
                        line_char_start + rest_start + url_match.start()
                    ),
                    "url_char_end": (
                        line_char_start + rest_start + url_match.end()
                    ),
                })
            else:
                starts.append({
                    "char_start": line_char_start,
                    "title": _safe_title(rest),
                    "title_char_start": line_char_start + rest_start,
                    "title_char_end": line_char_start + rest_end,
                    "url": "",
                    "url_char_start": line_char_start + rest_end,
                    "url_char_end": line_char_start + rest_end,
                })
            index += 1
            continue

        url_match = _VISIBLE_SOURCE_URL.fullmatch(raw_line)
        if url_match:
            local_start, local_end = _trimmed_group_span(
                raw_line, *url_match.span("url")
            )
            starts.append({
                "char_start": line_char_start,
                "title": "(untitled source)",
                "title_char_start": line_char_start,
                "title_char_end": line_char_start,
                "url": raw_line[local_start:local_end],
                "url_char_start": line_char_start + local_start,
                "url_char_end": line_char_start + local_end,
            })
        index += 1

    def absolute_byte(char_offset: int) -> int:
        return body_byte_start + len(body[:char_offset].encode("utf-8"))

    blocks: list[_ParsedSourceBlock] = []
    for ordinal, start in enumerate(starts):
        end_char = (
            starts[ordinal + 1]["char_start"]
            if ordinal + 1 < len(starts)
            else len(body)
        )
        blocks.append(_ParsedSourceBlock(
            title=str(start["title"]),
            url=str(start["url"]),
            byte_start=absolute_byte(int(start["char_start"])),
            byte_end=absolute_byte(end_char),
            title_byte_start=absolute_byte(int(start["title_char_start"])),
            title_byte_end=absolute_byte(int(start["title_char_end"])),
            url_byte_start=absolute_byte(int(start["url_char_start"])),
            url_byte_end=absolute_byte(int(start["url_char_end"])),
        ))
    return tuple(blocks)


def build_visible_source_registry(
    view: VisibleCompressorView,
) -> VisibleSourceRegistry:
    """Derive title/URL affordances only from byte-addressed TOOL_EVIDENCE messages.

    Occurrence ids are capture-time sidecar provenance. Source blocks are visible bytes. They
    may be paired only when their ordered cardinalities agree exactly; proximity or a later
    acquisition lookup is not a safe association.
    """

    entries: list[VisibleSourceEntry] = []
    partitions: list[VisibleSourceEntry] = []
    for segment in view.message_segments:
        if segment.get("kind") != "TOOL_EVIDENCE":
            continue
        start = int(segment["byte_start"])
        end = int(segment["byte_end"])
        body = view.view_bytes[start:end].decode("utf-8", errors="strict")
        occurrences = tuple(
            str(value) for value in (segment.get("occurrence_ids") or ())
        )
        blocks = _source_blocks(body, body_byte_start=start)
        # No title/URL bytes means there is no affordance to propagate. It is not ambiguous:
        # both arms simply operate without one. Once any source-shaped block exists, however,
        # an ordinal mismatch would make every guessed title/URL association unsafe.
        if not blocks:
            continue
        if (
            len(blocks) != len(occurrences)
            or len(occurrences) != len(set(occurrences))
        ):
            raise VisibleSourceRegistryError(
                f"tool message {str(segment.get('message_id') or '')!r} has "
                f"{len(blocks)} visible source blocks but {len(occurrences)} distinct "
                "capture-time occurrence ids"
            )
        for block, occurrence_id in zip(blocks, occurrences, strict=True):
            partition = VisibleSourceEntry(
                label="",
                title=block.title,
                url=block.url,
                occurrence_ids=(occurrence_id,),
                message_id=str(segment.get("message_id") or ""),
                visible_compressor_view_hash=view.view_hash,
                byte_start=block.byte_start,
                byte_end=block.byte_end,
                title_byte_start=block.title_byte_start,
                title_byte_end=block.title_byte_end,
                url_byte_start=block.url_byte_start,
                url_byte_end=block.url_byte_end,
            )
            partitions.append(partition)
            if block.url:
                entries.append(
                    replace(partition, label=str(len(entries) + 1))
                )
    return VisibleSourceRegistry(
        entries=tuple(entries),
        partitions=tuple(partitions),
    )


def source_partitioned_message_segments(
    view: VisibleCompressorView,
    registry: VisibleSourceRegistry,
) -> tuple[dict, ...]:
    """Split TOOL_EVIDENCE messages at source boundaries before visible-view chunking.

    Chunking a whole multi-source ToolMessage can create one candidate spanning two URLs. No
    post-hoc title choice can make that candidate single-source. Partitioning the already
    frozen byte ranges first preserves every byte while making the source ownership structural.
    Preamble bytes remain visible but become non-citable context.
    """

    out: list[dict] = []
    for segment in view.message_segments:
        message_entries = [
            entry
            for entry in registry.partitions
            if entry.message_id == str(segment.get("message_id") or "")
        ]
        if segment.get("kind") != "TOOL_EVIDENCE" or not message_entries:
            out.append(dict(segment))
            continue
        cursor = int(segment["byte_start"])
        segment_end = int(segment["byte_end"])
        for entry in message_entries:
            if cursor < entry.byte_start:
                out.append({
                    **segment,
                    "kind": "TOOL_UNATTRIBUTED_CONTEXT",
                    "byte_start": cursor,
                    "byte_end": entry.byte_start,
                    "occurrence_ids": [],
                })
            out.append({
                **segment,
                "kind": "TOOL_EVIDENCE",
                "byte_start": entry.byte_start,
                "byte_end": entry.byte_end,
                "occurrence_ids": list(entry.occurrence_ids),
                "source_binding_label": entry.label,
            })
            cursor = entry.byte_end
        if cursor < segment_end:
            out.append({
                **segment,
                "kind": "TOOL_UNATTRIBUTED_CONTEXT",
                "byte_start": cursor,
                "byte_end": segment_end,
                "occurrence_ids": [],
            })
    return tuple(out)


def _occurrences_of(message: FrozenMessage) -> tuple[str, ...]:
    """Occurrences recorded on the tool message itself, via its artifact.

    Absent an artifact there are none. A tool message whose provenance was not recorded at
    capture time does not acquire it later; a span over it simply cannot be cited, which is the
    honest outcome rather than a guessed one.
    """
    import json

    if not message.artifact_canonical:
        return ()
    try:
        artifact = json.loads(message.artifact_canonical)
    except (TypeError, ValueError):
        return ()
    if isinstance(artifact, dict):
        ids = artifact.get("source_occurrence_ids") or []
        return tuple(str(i) for i in ids)
    return ()
