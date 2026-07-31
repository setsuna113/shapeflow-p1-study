"""The bytes the search seam actually served, addressed the way a checkpoint addresses them.

The H checkpoint names a page by ``raw_content_id`` -- ``sha256(raw_content[:max_content_length])``
-- and nothing else. A page selector is handed that id and has to produce the bytes behind it,
because the whole point of P1 at the H boundary is to publish spans *of the page P0 read*. Hand
it the wrong bytes and the arm still runs, still publishes, still looks like a working P1 arm;
it is just selecting from a different document than the one it is compared against.

Week-1 resolved the id by re-deriving it from the task-local frozen pool: every page that task
could ever see was on disk, so the map could be built before the cell started. That does not
survive the move to a 100k-document corpus reached by a live dense query -- there is no
enumerable per-task page set, and building one would mean encoding the whole corpus per task.

So the map is built where the bytes are: at the seam, from the records it just bounded. Those
are the exact bytes handed to vendor, after the shared budget, which is precisely what the
checkpoint hashes. Registering them there makes the correspondence structural rather than
something two code paths have to keep agreeing about.
"""

from __future__ import annotations

from ..hashing import sha256_hex

__all__ = ["PageRegistry"]


class PageRegistry:
    """content_id -> the visible bytes, and the occurrence they arrived as.

    First writer wins. Two occurrences of the same bytes are the same page as far as the
    checkpoint is concerned -- it addresses content, not position -- and the first occurrence id
    is the one vendor's URL dedup keeps, so a later duplicate must not overwrite the lineage the
    citation scorer will look for.
    """

    def __init__(self) -> None:
        self._text: dict[str, str] = {}
        self._occurrence: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self._text)

    def record(self, text: str, *, occurrence_id: str) -> str:
        """Register one page's visible bytes and return the id the checkpoint will use."""
        if not text:
            # A page with no body has no content id in the checkpoint either
            # (``visible_raw`` returns None), so registering one would invent a key nothing
            # will ever ask for.
            return ""
        content_id = sha256_hex(text.encode("utf-8"))
        self._text.setdefault(content_id, text)
        self._occurrence.setdefault(content_id, occurrence_id or content_id)
        return content_id

    def prefill(self, text_by_id: dict, occurrence_by_id: dict) -> None:
        """Seed from a pre-enumerated page set, for worlds that have one."""
        for content_id, text in text_by_id.items():
            self._text.setdefault(str(content_id), str(text))
        for content_id, occurrence_id in occurrence_by_id.items():
            self._occurrence.setdefault(str(content_id), str(occurrence_id))

    def text_for(self, content_id: str) -> str:
        """The bytes behind an id, or empty. Empty is a real answer, not a lookup failure.

        A selector offered a candidate it cannot read publishes nothing for it, which the
        publication canary sees as an inert arm. That is the correct outcome: better an arm that
        visibly selects nothing than one that silently selects from bytes P0 never saw.
        """
        return self._text.get(content_id, "")

    def occurrence_for(self, content_id: str) -> str:
        return self._occurrence.get(content_id, content_id)
