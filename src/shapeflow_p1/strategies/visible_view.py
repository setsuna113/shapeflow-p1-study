"""The VisibleCompressorView: exactly the bytes P0's compressor read, and where each one came from.

This object is what makes "compressor-only" checkable instead of asserted. It is built by
rendering the frozen ``researcher_messages`` the same way the compressor's prompt renders them,
and it records each message's byte range inside the result -- so every C_VISIBLE span addresses
a real offset in a real byte string that a real compressor would have seen, and preflight can
re-derive it.

Two rules are enforced here rather than downstream:

**A message's kind is the message's own.** A ToolMessage is ``TOOL_EVIDENCE``; an AIMessage is
``MODEL_DERIVED_CONTEXT``; a HumanMessage is ``USER_CONTEXT``. Model reasoning cannot become
evidence by being selected, which is the line between C_VISIBLE reading what the compressor read
and C_VISIBLE manufacturing provenance.

**Occurrence ids come from the tool call, not from a reverse lookup.** Mapping a
model-written page summary back to the raw page it summarised is exactly the smuggling that
distinguishes C_REGISTRY from C_VISIBLE, so a tool message carries only the occurrences its own
call recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..hashing import sha256_hex
from ..odr.checkpoints import FrozenMessage

__all__ = ["VisibleCompressorView", "build_visible_view", "KIND_BY_ROLE"]

KIND_BY_ROLE = {
    "tool": "TOOL_EVIDENCE",
    "ai": "MODEL_DERIVED_CONTEXT",
    "human": "USER_CONTEXT",
    "system": "USER_CONTEXT",
}


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
        header = f"<{message.role}>\n".encode("utf-8")
        body = _render_content(message.content).encode("utf-8")
        footer = f"\n</{message.role}>\n".encode("utf-8")
        chunks.extend((header, body, footer))
        body_start = cursor + len(header)
        segments.append({
            "message_id": message.message_id or f"m{index}",
            "role": message.role,
            "kind": KIND_BY_ROLE.get(message.role, "USER_CONTEXT"),
            "byte_start": body_start,
            "byte_end": body_start + len(body),
            # Only a tool message carries occurrences, and only the ones its own call recorded.
            # A reverse lookup from a model-written summary back to the page it described is the
            # smuggling that separates C_REGISTRY from C_VISIBLE.
            "occurrence_ids": list(_occurrences_of(message))
            if message.role == "tool" else [],
        })
        cursor = body_start + len(body) + len(footer)

    view_bytes = b"".join(chunks)
    return VisibleCompressorView(
        view_bytes=view_bytes,
        view_hash=sha256_hex(view_bytes),
        message_segments=tuple(segments),
    )


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
