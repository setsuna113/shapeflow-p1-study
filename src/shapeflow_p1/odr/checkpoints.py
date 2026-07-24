"""Immutable, content-addressed checkpoints at the two P1 boundaries.

A checkpoint is the complete state *before* a reducer runs, frozen so that every variant
in a component trial forks from byte-identical input. Two properties are load-bearing:

- **Content addressing.** ``digest`` is a hash of the canonical checkpoint content. All
  variants of one boundary must fork from the same digest; the screening analysis groups
  by it. If two forks disagreed on their starting bytes, their paired comparison would be
  meaningless.

- **Reference, don't inline, the big bytes.** Raw page content is carried by ``content_id``
  reference into the object store, never inlined. That keeps a checkpoint small and lets
  the store dedup, while the digest still commits to *which* bytes via their hash.

The capture point matters and is encoded in the shapes here:

- ``HCheckpoint`` is taken at the top of ``researcher_tools``, holding the researcher's
  assistant turn and every sibling tool call, plus each search call's vendor-visible
  result set -- i.e. the whole batch, before any page is transformed or published. The
  publish unit is the batch, so the checkpoint is the batch.

- ``CCheckpoint`` is taken at the top of ``compress_research``, holding a lossless copy of
  ``researcher_messages`` captured *before* vendor appends its compression instruction
  (``deep_researcher.py`` line 538). Cloning after that append would poison the first fork
  and, through the shared list, every fork after it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..canonical import canonical_json
from ..hashing import derive_id, sha256_hex

__all__ = [
    "SamplingEnvelope",
    "FrozenToolCall",
    "FrozenMessage",
    "VendorVisibleResult",
    "EvidenceManifest",
    "HCheckpoint",
    "CCheckpoint",
    "fork_key",
]


@dataclass(frozen=True)
class SamplingEnvelope:
    """Everything needed to reproduce the model call that produced this state."""

    model: str
    temperature: float
    top_p: float
    max_tokens: int
    n: int = 1
    seed: Optional[int] = None

    def content(self) -> dict:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "n": self.n,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class FrozenToolCall:
    """One tool call from an assistant turn. ``args_canonical`` is the canonical JSON of the
    arguments, so equal argument objects hash equally regardless of key order."""

    id: str
    name: str
    args_canonical: str

    def content(self) -> dict:
        return {"id": self.id, "name": self.name, "args": self.args_canonical}


@dataclass(frozen=True)
class FrozenMessage:
    """A langchain message reduced to its identity-bearing fields. The adapter builds these
    from live messages; nothing here depends on langchain."""

    role: str  # ai | human | system | tool
    content: str
    tool_calls: tuple[FrozenToolCall, ...] = ()
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    message_id: Optional[str] = None

    def content_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": [tc.content() for tc in self.tool_calls],
            "name": self.name,
            "tool_call_id": self.tool_call_id,
            "message_id": self.message_id,
        }

    @property
    def text_sha256(self) -> str:
        return sha256_hex(self.content.encode("utf-8"))


@dataclass(frozen=True)
class VendorVisibleResult:
    """One entry of a search call's vendor-visible (URL-deduped, first-occurrence) result
    set. Raw content is referenced by ``raw_content_id``; ``snippet`` is the small vendor
    'content' field kept inline because P0 falls back to it when raw content is absent."""

    vendor_visible_order: int
    url: str
    title: str
    snippet: str
    raw_content_id: Optional[str]  # content_id into the object store, or None

    def content(self) -> dict:
        return {
            "order": self.vendor_visible_order,
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "raw_content_id": self.raw_content_id,
        }


@dataclass(frozen=True)
class EvidenceManifest:
    """The evidence addressable at a boundary, as a stable ordered set of span ids."""

    span_ids: tuple[str, ...]

    def content(self) -> dict:
        return {"span_ids": list(self.span_ids)}

    @property
    def digest(self) -> str:
        return derive_id("evidence_manifest", self.content())


@dataclass(frozen=True)
class HCheckpoint:
    """The whole assistant-turn batch at the WEBPAGE boundary, before publish."""

    task_id: str
    researcher_id: str
    assistant_turn_index: int
    assistant_message: FrozenMessage
    sibling_tool_calls: tuple[FrozenToolCall, ...]
    # tool_call_id -> that search call's vendor-visible result set
    search_result_sets: tuple[tuple[str, tuple[VendorVisibleResult, ...]], ...]
    # tool_call_id -> content_id of a non-search sibling's output (executed unchanged)
    non_search_outputs: tuple[tuple[str, str], ...]
    researcher_state_hash: str
    sampling: SamplingEnvelope

    def content(self) -> dict:
        return {
            "task_id": self.task_id,
            "researcher_id": self.researcher_id,
            "assistant_turn_index": self.assistant_turn_index,
            "assistant_message": self.assistant_message.content_dict(),
            "sibling_tool_calls": [tc.content() for tc in self.sibling_tool_calls],
            "search_result_sets": [
                [tcid, [r.content() for r in results]]
                for tcid, results in self.search_result_sets
            ],
            "non_search_outputs": [list(pair) for pair in self.non_search_outputs],
            "researcher_state_hash": self.researcher_state_hash,
            "sampling": self.sampling.content(),
        }

    @property
    def digest(self) -> str:
        return derive_id("h_checkpoint", self.content())


@dataclass(frozen=True)
class CCheckpoint:
    """The full researcher envelope at the RESEARCHER_CLOSE boundary."""

    task_id: str
    researcher_id: str
    # Lossless clone of researcher_messages captured BEFORE vendor's in-place append.
    researcher_messages: tuple[FrozenMessage, ...]
    evidence_manifest: EvidenceManifest
    query_attempt_ids: tuple[str, ...]
    close_reason: str
    sampling: SamplingEnvelope

    def content(self) -> dict:
        return {
            "task_id": self.task_id,
            "researcher_id": self.researcher_id,
            "researcher_messages": [m.content_dict() for m in self.researcher_messages],
            "evidence_manifest": self.evidence_manifest.content(),
            "query_attempt_ids": list(self.query_attempt_ids),
            "close_reason": self.close_reason,
            "sampling": self.sampling.content(),
        }

    @property
    def digest(self) -> str:
        return derive_id("c_checkpoint", self.content())


def fork_key(
    *,
    protocol_sha: str,
    boundary_id: str,
    variant_id: str,
    seed: int,
    prompt_renderer_version: str,
) -> str:
    """Identity of one variant's fork from one boundary.

    Every fork writes only new content-addressed objects; the parent boundary is
    immutable. This key names the (boundary, variant, seed) execution so the ledger can
    dedup it on resume.
    """
    return derive_id(
        "fork",
        {
            "protocol_sha": protocol_sha,
            "boundary_id": boundary_id,
            "variant_id": variant_id,
            "seed": seed,
            "prompt_renderer_version": prompt_renderer_version,
        },
    )
