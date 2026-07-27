"""The graph state a C fork needs downstream of the boundary, which no checkpoint stores.

A ``CCheckpoint`` is a complete *researcher* envelope: enough to re-enter
``compress_research`` and produce that arm's ``compressed_research``. It is not enough to reach
a final report, because the report is written by the root graph from ``notes`` +
``research_brief`` + ``messages`` -- state that lives above the researcher and that the
checkpoint deliberately does not carry.

This module captures exactly that missing state, once, during the anchor run.

**Why the report re-entry is frozen rather than resumed.** Re-entering the *supervisor* would
re-run ``SUPERVISOR_CONTINUE``, which may emit a fresh ``ConductResearch`` and spawn a real
researcher with real search. That is researcher inference after the fork, it differs between
arms, and it makes ``upstream_research_calls == 0`` unenforceable. So the continuation holds
the supervisor's realized research plan fixed: the arm's own note is substituted into the
anchor's note vector at exactly its slot, every sibling note is byte-identical, and only
``final_report_generation`` runs. That estimand is a controlled direct effect of the C reducer
on the final report; the supervisor-mediated path is deliberately blocked and is bounded
separately by the full-graph C arms.

**The slot mapping is exact, not positional.** Children run concurrently under
``asyncio.gather``, so C boundaries are captured in completion order, while notes are appended
in ``allowed_conduct_research_calls`` order -- and ``get_notes_from_tool_calls`` also picks up
``think_tool`` results and overflow errors, so note index is not child index either. The
binding at ``vendor_hooks.invoke_child_researcher`` names each child
``.../child-<iteration>-<ordinal>-<sha256(tool_call_id)[:16]>``, and the note for that child is
the ``ConductResearch`` ToolMessage carrying that same ``tool_call_id``. Matching on that hash
is unambiguous under concurrency, interleaving and retries. A boundary whose slot cannot be
matched is excluded *before* any arm is offered, so the ITT denominator is unaffected.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..canonical import canonical_json
from ..fsmode import chmod_shared
from ..hashing import derive_id, sha256_hex
from .checkpoints import FrozenMessage, _message_from

__all__ = [
    "ContinuationEnvelope",
    "ContinuationStore",
    "NoteSlot",
    "child_slot_key",
    "note_slots_from_supervisor_messages",
]


def child_slot_key(tool_call_id: str) -> str:
    """The child-name fragment a ``ConductResearch`` tool-call id hashes to.

    Must stay identical to ``vendor_hooks.invoke_child_researcher``; the C fork's note
    substitution is only exact because both sides derive this the same way.
    """
    return sha256_hex(str(tool_call_id).encode("utf-8"))[:16]


@dataclass(frozen=True)
class NoteSlot:
    """One entry of the supervisor's note vector, and which child produced it (if any)."""

    ordinal: int
    content: str
    #: Set only for ConductResearch results. think_tool outputs and overflow errors are notes
    #: too, and they are held fixed rather than being forkable slots.
    tool_call_id: str = ""

    @property
    def slot_key(self) -> str:
        return child_slot_key(self.tool_call_id) if self.tool_call_id else ""

    def content_dict(self) -> dict:
        return {
            "ordinal": self.ordinal,
            "content_sha256": sha256_hex(self.content.encode("utf-8")),
            "tool_call_id": self.tool_call_id,
        }


def note_slots_from_supervisor_messages(
    supervisor_messages: Sequence[Any], notes: Sequence[str]
) -> tuple[NoteSlot, ...]:
    """Pair the note vector with the tool calls that produced it.

    ``get_notes_from_tool_calls`` is ``[m.content for m in filter_messages(include_types="tool")]``,
    so the notes are exactly the tool messages of ``supervisor_messages``, in order. Rebuilding
    that filter here rather than trusting index arithmetic means a mismatch is caught instead
    of silently shifting every slot by one.
    """
    tool_messages = [
        message for message in supervisor_messages
        if str(getattr(message, "type", "")) == "tool"
    ]
    if len(tool_messages) != len(notes):
        raise ValueError(
            f"supervisor note vector has {len(notes)} entries but {len(tool_messages)} tool "
            "messages; the note slots cannot be attributed to their producers"
        )
    slots: list[NoteSlot] = []
    for ordinal, (message, note) in enumerate(zip(tool_messages, notes, strict=False)):
        content = str(getattr(message, "content", "") or "")
        if content != str(note):
            raise ValueError(
                f"note {ordinal} does not match the tool message it should have come from"
            )
        slots.append(NoteSlot(
            ordinal=ordinal,
            content=str(note),
            tool_call_id=(
                str(getattr(message, "tool_call_id", "") or "")
                if str(getattr(message, "name", "")) == "ConductResearch" else ""
            ),
        ))
    return tuple(slots)


@dataclass(frozen=True)
class ContinuationEnvelope:
    """The frozen world a C fork writes its report into.

    Everything here is held byte-identical across the arms of one boundary. Only one note slot
    -- the forking child's -- is replaced, and that is asserted at execution time.
    """

    task_id: str
    seed: int
    research_brief: str
    root_messages: tuple[FrozenMessage, ...]
    notes: tuple[NoteSlot, ...]
    #: ``get_today_str()`` is ``datetime.now()`` and is formatted into both the compressor and
    #: the final-report prompt. A boundary whose arms straddle UTC midnight would differ by
    #: more than the treatment, so the anchor's value is recorded and re-checked per arm.
    anchor_today_str: str
    #: The anchor produced the shared upstream work constant. If the engine rebooted between
    #: the anchor and an arm, that constant came from a different boot and the pair is void.
    anchor_engine_epoch: str
    anchor_run_ref: str

    def content(self) -> dict:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "research_brief": self.research_brief,
            "root_messages": [m.content_dict() for m in self.root_messages],
            "notes": [slot.content_dict() for slot in self.notes],
            "anchor_today_str": self.anchor_today_str,
            "anchor_engine_epoch": self.anchor_engine_epoch,
            "anchor_run_ref": self.anchor_run_ref,
        }

    @property
    def digest(self) -> str:
        return derive_id("c_continuation_envelope", self.content())

    def slot_for(self, researcher_id: str) -> NoteSlot | None:
        """The note slot produced by ``researcher_id``, or None when it cannot be attributed.

        None is a refusal, not a default: the caller excludes the boundary rather than
        substituting into a slot it merely guessed.
        """
        key = researcher_id.rsplit("-", 1)[-1] if "-" in researcher_id else ""
        if not key:
            return None
        matches = [slot for slot in self.notes if slot.slot_key and slot.slot_key == key]
        if len(matches) != 1:
            return None
        return matches[0]

    def substitute(self, slot: NoteSlot, replacement: str) -> list[str]:
        """The note vector with exactly one slot replaced.

        Returns a plain list for ``{"type": "override", "value": ...}``; a bare list would be
        *added* to the channel by ``override_reducer`` rather than replacing it.
        """
        if slot.ordinal < 0 or slot.ordinal >= len(self.notes):
            raise ValueError(f"note slot {slot.ordinal} is outside the anchor note vector")
        values = [entry.content for entry in self.notes]
        values[slot.ordinal] = replacement
        return values


def to_document(envelope: ContinuationEnvelope) -> dict:
    return {"digest": envelope.digest, **envelope.content()}


class ContinuationStore:
    """One envelope per anchor boundary set, keyed by digest, verified on load."""

    def __init__(self, root) -> None:
        self.root = Path(root)

    def _path(self, digest: str) -> Path:
        return self.root / digest[:2] / f"{digest}.json"

    def put(self, envelope: ContinuationEnvelope) -> str:
        body = to_document(envelope)
        path = self._path(body["digest"])
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return body["digest"]
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(canonical_json(body))
                handle.flush()
                os.fsync(handle.fileno())
            chmod_shared(tmp)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return body["digest"]

    def get(self, digest: str) -> ContinuationEnvelope:
        path = self._path(digest)
        if not path.exists():
            raise FileNotFoundError(f"no continuation envelope {digest} under {self.root}")
        return from_document(json.loads(path.read_text(encoding="utf-8")))


def from_document(body: Mapping[str, Any]) -> ContinuationEnvelope:
    """Rebuild an envelope, refusing one whose digest disagrees.

    The note *contents* are hashed rather than stored, so a rebuilt envelope carries hashes
    where the original carried text. Reconstructing with the recorded hashes and re-deriving
    the digest is therefore the whole integrity check: a tampered note changes it.
    """
    envelope = ContinuationEnvelope(
        task_id=str(body["task_id"]),
        seed=int(body["seed"]),
        research_brief=str(body["research_brief"]),
        root_messages=tuple(_message_from(m) for m in body["root_messages"]),
        notes=tuple(
            _NoteSlotDigest(
                ordinal=int(entry["ordinal"]),
                content_sha256=str(entry["content_sha256"]),
                tool_call_id=str(entry.get("tool_call_id") or ""),
            )
            for entry in body["notes"]
        ),
        anchor_today_str=str(body["anchor_today_str"]),
        anchor_engine_epoch=str(body["anchor_engine_epoch"]),
        anchor_run_ref=str(body["anchor_run_ref"]),
    )
    recorded = str(body.get("digest") or "")
    if recorded and recorded != envelope.digest:
        raise ValueError(
            f"continuation envelope records digest {recorded!r} but rebuilds to "
            f"{envelope.digest!r}; the frozen world is not the one it claims to be"
        )
    return envelope


@dataclass(frozen=True)
class _NoteSlotDigest:
    """A note slot rebuilt from its document, which holds the hash rather than the text.

    Substitution needs the *other* slots' text, so a reloaded envelope can verify and attribute
    but cannot itself rebuild the report input. The fork keeps the in-memory envelope from the
    anchor pass for that, and uses the reloaded one to prove the two agree.
    """

    ordinal: int
    content_sha256: str
    tool_call_id: str = ""

    @property
    def slot_key(self) -> str:
        return child_slot_key(self.tool_call_id) if self.tool_call_id else ""

    @property
    def content(self) -> str:  # pragma: no cover - guarded by the raise below
        raise ValueError(
            "a reloaded continuation envelope stores note hashes, not note text; run the "
            "arms of a boundary in the same pass as its anchor"
        )

    def content_dict(self) -> dict:
        return {
            "ordinal": self.ordinal,
            "content_sha256": self.content_sha256,
            "tool_call_id": self.tool_call_id,
        }
