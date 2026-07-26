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

Note the ``HCheckpoint`` capture point precisely: it is inside ``researcher_tools``, *after*
``asyncio.gather`` returns the sibling results and *before* the ``ToolMessage`` list is built.
Not at the top of the function -- the searches have not run there, so their result sets do not
exist yet -- and not inside ``tavily_search``, which sees only its own call's results and
cannot know what its siblings returned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..canonical import canonical_json
from ..fsmode import chmod_shared
from ..hashing import derive_id, sha256_hex

__all__ = [
    "CheckpointStore",
    "to_document",
    "from_document",
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
    """A langchain message frozen losslessly. The adapter builds these from live messages;
    nothing here depends on langchain.

    "Identity-bearing fields" is not enough, and C_VISIBLE is why. Its whole claim is that the
    selector saw *exactly* what P0's compressor saw -- and what the compressor sees is the
    rendered message list. Any field dropped here is a field the reconstructed clone lacks, so
    the two prompts differ by however that field renders, while every hash we compute agrees.

    ``content`` is ``str | list[dict]``: modern providers emit structured content blocks, and
    flattening them to text loses block boundaries that the chat template renders differently.
    ``additional_kwargs`` and ``response_metadata`` carry provider fields (refusals, tool-call
    deltas, logprobs) that some templates surface. ``usage_metadata`` is not rendered but is
    the arm's token accounting, and ``artifact`` is ToolMessage payload.

    The round-trip property test in the adapter is the real check: clone, rebuild, render, and
    require byte equality with the original rendering.
    """

    role: str  # ai | human | system | tool
    content: object  # str | list[dict]
    tool_calls: tuple[FrozenToolCall, ...] = ()
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    message_id: Optional[str] = None
    additional_kwargs_canonical: str = "{}"
    response_metadata_canonical: str = "{}"
    usage_metadata_canonical: str = "{}"
    artifact_canonical: Optional[str] = None
    invalid_tool_calls_canonical: str = "[]"
    status: Optional[str] = None

    def content_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": [tc.content() for tc in self.tool_calls],
            "name": self.name,
            "tool_call_id": self.tool_call_id,
            "message_id": self.message_id,
            "additional_kwargs": self.additional_kwargs_canonical,
            "response_metadata": self.response_metadata_canonical,
            "usage_metadata": self.usage_metadata_canonical,
            "artifact": self.artifact_canonical,
            "invalid_tool_calls": self.invalid_tool_calls_canonical,
            "status": self.status,
        }

    @property
    def text_sha256(self) -> str:
        """Digest of the content as it renders. Structured blocks hash over their canonical
        form, so a block-boundary change is visible rather than flattened away."""
        if isinstance(self.content, str):
            return sha256_hex(self.content.encode("utf-8"))
        return sha256_hex(canonical_json(self.content))


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
    # Exact frozen-pool occurrence. Content hashes are intentionally not identities: two distinct
    # URLs may serve identical bytes and both remain distinct citation/provenance occurrences.
    source_occurrence_id: Optional[str] = None

    def content(self) -> dict:
        return {
            "order": self.vendor_visible_order,
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "raw_content_id": self.raw_content_id,
            "source_occurrence_id": self.source_occurrence_id,
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
    # Structural coordinate of the ConductResearch child that owns this checkpoint:
    # (supervisor research iteration, allowed tool-call ordinal, tool-call id).  A cell-level
    # researcher id is not enough because every child resets its assistant-turn/toolset
    # counters.  ``None`` is reserved for direct researcher-subgraph probes.
    researcher_coordinate: Optional[tuple[int, int, str]] = None

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
            "researcher_coordinate": (
                list(self.researcher_coordinate)
                if self.researcher_coordinate is not None else None
            ),
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


# --- persistence ---------------------------------------------------------------------------
#
# A checkpoint that exists only in the process that produced it is not a checkpoint. The
# runner used to store `{"kind": ..., "digest": ...}` -- two fields, no state, no reader --
# so nothing could ever fork from a boundary, and the "forked-state component trial" ran the
# whole seven-arm graph again per arm instead.


def to_document(checkpoint) -> dict:
    """The on-disk form: the full state, plus its kind and its digest."""
    kind = "H" if isinstance(checkpoint, HCheckpoint) else "C"
    return {"kind": kind, "digest": checkpoint.digest, **checkpoint.content()}


def _message_from(body: dict) -> FrozenMessage:
    return FrozenMessage(
        role=body["role"],
        content=body["content"],
        tool_calls=tuple(
            FrozenToolCall(id=tc["id"], name=tc["name"], args_canonical=tc["args"])
            for tc in body.get("tool_calls") or ()
        ),
        name=body.get("name"),
        tool_call_id=body.get("tool_call_id"),
        message_id=body.get("message_id"),
        additional_kwargs_canonical=body.get("additional_kwargs", "{}"),
        response_metadata_canonical=body.get("response_metadata", "{}"),
        usage_metadata_canonical=body.get("usage_metadata", "{}"),
        artifact_canonical=body.get("artifact"),
        invalid_tool_calls_canonical=body.get("invalid_tool_calls", "[]"),
        status=body.get("status"),
    )


def _sampling_from(body: dict) -> SamplingEnvelope:
    return SamplingEnvelope(
        model=body["model"], temperature=body["temperature"], top_p=body["top_p"],
        max_tokens=body["max_tokens"], n=body.get("n", 1), seed=body.get("seed"),
    )


def from_document(body: dict):
    """Rebuild a checkpoint from its document, and refuse one whose digest disagrees.

    The digest check is the point: a boundary every variant forks from has to be the same
    boundary, and "the file was there" is not that.
    """
    kind = body.get("kind")
    if kind == "H":
        checkpoint = HCheckpoint(
            task_id=body["task_id"],
            researcher_id=body["researcher_id"],
            assistant_turn_index=body["assistant_turn_index"],
            assistant_message=_message_from(body["assistant_message"]),
            sibling_tool_calls=tuple(
                FrozenToolCall(id=tc["id"], name=tc["name"], args_canonical=tc["args"])
                for tc in body["sibling_tool_calls"]
            ),
            search_result_sets=tuple(
                (tcid, tuple(
                    VendorVisibleResult(
                        vendor_visible_order=r["order"], url=r["url"], title=r["title"],
                        snippet=r["snippet"], raw_content_id=r["raw_content_id"],
                        source_occurrence_id=r.get("source_occurrence_id"),
                    ) for r in results
                ))
                for tcid, results in body["search_result_sets"]
            ),
            non_search_outputs=tuple(
                (pair[0], pair[1]) for pair in body["non_search_outputs"]
            ),
            researcher_state_hash=body["researcher_state_hash"],
            sampling=_sampling_from(body["sampling"]),
            # Part of content() and therefore of the digest. Dropping it here rebuilt every
            # ConductResearch child's checkpoint with None, so the digest check below fired on
            # load and no child boundary could be read back at all -- the store was write-only
            # for exactly the checkpoints a fork needs. JSON has no tuples, so restore the
            # (iteration, ordinal, tool_call_id) shape rather than leaving a list, which would
            # canonicalize identically but compare unequal to a freshly built checkpoint.
            researcher_coordinate=(
                (int(coordinate[0]), int(coordinate[1]), str(coordinate[2]))
                if (coordinate := body.get("researcher_coordinate")) is not None
                else None
            ),
        )
    elif kind == "C":
        checkpoint = CCheckpoint(
            task_id=body["task_id"],
            researcher_id=body["researcher_id"],
            researcher_messages=tuple(
                _message_from(m) for m in body["researcher_messages"]),
            evidence_manifest=EvidenceManifest(
                span_ids=tuple(body["evidence_manifest"]["span_ids"])),
            query_attempt_ids=tuple(body["query_attempt_ids"]),
            close_reason=body["close_reason"],
            sampling=_sampling_from(body["sampling"]),
        )
    else:
        raise ValueError(f"unknown checkpoint kind {kind!r}")
    recorded = body.get("digest")
    if recorded and recorded != checkpoint.digest:
        raise ValueError(
            f"checkpoint document records digest {recorded!r} but rebuilds to "
            f"{checkpoint.digest!r}; the boundary is not the one it claims to be"
        )
    return checkpoint


class CheckpointStore:
    """Content-addressed checkpoints, keyed by digest.

    One file per boundary under ``paths.checkpoints``, so a fork can be run in a later
    process -- which is what makes a component trial a fork rather than a re-run.
    """

    def __init__(self, root) -> None:
        from pathlib import Path

        self.root = Path(root)

    def _path(self, digest: str):
        return self.root / digest[:2] / f"{digest}.json"

    def put(self, checkpoint) -> str:
        import json
        import os
        import tempfile

        body = to_document(checkpoint)
        path = self._path(body["digest"])
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return body["digest"]
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(canonical_json(body))
                fh.flush()
                os.fsync(fh.fileno())
            # The runner writes these; the evaluator scores against them and a later fork
            # process loads them. mkstemp's 0600 would zero the inherited ACL mask and make
            # the checkpoint unreadable to both. See shapeflow_p1.fsmode.
            chmod_shared(tmp)
            os.replace(tmp, path)
        except BaseException:
            from pathlib import Path as _P

            _P(tmp).unlink(missing_ok=True)
            raise
        return body["digest"]

    def get(self, digest: str):
        import json

        path = self._path(digest)
        if not path.exists():
            raise FileNotFoundError(f"no checkpoint {digest} under {self.root}")
        return from_document(json.loads(path.read_text(encoding="utf-8")))

    def digests(self) -> list[str]:
        return sorted(p.stem for p in self.root.rglob("*.json"))


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
