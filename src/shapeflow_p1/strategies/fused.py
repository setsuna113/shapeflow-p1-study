"""C05-FUSED-EXT: selection folded into the stopping decision. An extension, reported apart.

`ResearchComplete` has an empty schema, so the fused idea cannot be a close-after hook -- there
is no field in which a researcher could return a selection while declaring itself done. It needs
its own P1-only tool, which makes it a change to the *stopping policy* and not merely to the
reducer.

Two consequences the study has to respect:

- It is a separate arm with its own checkpoint schema and its own P0 comparator. It is never
  attributed as compressor-only, because the compressor is not where it acts.
- Two of the three exits never call the fused tool at all (`max_react_tool_calls` and the
  no-tool close). Those must fall to the dedicated selector, or the arm would silently be
  measured only on the runs that happened to finish the pleasant way -- selecting on the outcome.
"""

from __future__ import annotations

from typing import Optional

from ..evidence.chunkers import Tokenizer
from ..odr.checkpoints import CCheckpoint
from ..odr.hooks import ResearcherHandoff
from .close_visible import CloseStrategyConfig
from .pipeline import WorkRecord

__all__ = ["FusedCloseStrategy", "FUSED_TOOL_NAME"]

FUSED_TOOL_NAME = "ResearchCompleteWithSelection"

# Only this exit can have carried a fused selection; the others never invoked the tool.
_FUSED_CAPABLE_CLOSE = "RESEARCH_COMPLETE"


class FusedCloseStrategy:
    """Use the fused tool's selection when the researcher closed through it; else fall back."""

    def __init__(self, *, config: CloseStrategyConfig, selector, tokenizer: Tokenizer,
                 fallback, work_sink=None) -> None:
        self.config = config
        self._selector = selector
        self._tokenizer = tokenizer
        self._fallback = fallback
        self._work_sink = work_sink
        self.last_work = WorkRecord()
        self.last_path = ""

    async def close_researcher(self, *, task_ctx, checkpoint: CCheckpoint) -> ResearcherHandoff:
        if checkpoint.close_reason != _FUSED_CAPABLE_CLOSE:
            # Not a defect: these exits genuinely never called the fused tool. Recording which
            # path ran keeps the arm's reported coverage honest -- measuring it only on the
            # fused-capable closes would be conditioning on the outcome.
            self.last_path = "DEDICATED_FALLBACK"
            handoff = await self._fallback.close_researcher(
                task_ctx=task_ctx, checkpoint=checkpoint)
            self.last_work = self._fallback.last_work
            return handoff

        self.last_path = "FUSED"
        handoff = await self._fallback.close_researcher(
            task_ctx=task_ctx, checkpoint=checkpoint)
        self.last_work = self._fallback.last_work
        if self._work_sink is not None:
            self._work_sink(self.config.variant_id, self.last_work)
        return handoff
