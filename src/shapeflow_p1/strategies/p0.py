"""The P0 strategies: vendor's own behaviour, reached through the P1 code path.

These exist to be *compared against*, and they are the second parity gate. Hooks-off proves the
patch is inert when unbound -- but with hooks off the deferred/checkpoint/reduce/refill path
never executes at all, so that gate cannot see a bug living inside it. Running vendor's logic
*through* that path, and requiring the result to be identical, can.

The page strategy does not re-implement vendor's summarization. It calls the closure the patch
captured, which is vendor's steps 3-7 verbatim -- including their inner ``asyncio.gather`` over
the summarization tasks. A re-implementation that summarised sequentially would produce the
same bytes with a different critical path, and the operational layer measures exactly that
difference; it would be read as a treatment effect.
"""

from __future__ import annotations

import asyncio
from typing import Sequence

from ..odr.checkpoints import CCheckpoint, HCheckpoint
from ..odr.hooks import ResearcherHandoff, TaskContext, ToolObservation

__all__ = ["VendorPageStrategy", "VendorCloseStrategy"]


class VendorPageStrategy:
    """Publish exactly what vendor's ``tavily_search`` would have published.

    Takes the deferred batches the patch produced and renders each through vendor's own
    closure, concurrently -- one ``gather`` over the siblings, mirroring the outer gather that
    would have run had the tool not deferred.
    """

    def __init__(self, deferred_by_call: dict) -> None:
        # tool_call_id -> DeferredPageBatch, supplied by the adapter for this batch.
        self._deferred = deferred_by_call

    async def transform_tool_batch(
        self, *, task_ctx: TaskContext, checkpoint: HCheckpoint
    ) -> Sequence[ToolObservation]:
        call_ids = [call_id for call_id, _ in checkpoint.search_result_sets]
        rendered = await asyncio.gather(
            *(self._deferred[cid].render_vendor() for cid in call_ids)
        )
        return [
            ToolObservation(tool_call_id=cid, name="tavily_search", content=content)
            for cid, content in zip(call_ids, rendered)
        ]


class VendorCloseStrategy:
    """Return None-equivalent: let vendor's own compression run.

    Modelled as "no handoff" rather than as a re-implementation of ``compress_research``. That
    function has retry-on-token-limit behaviour, an in-place append and a specific raw_notes
    construction; reproducing them here would create a second implementation to keep in sync,
    and any drift between the two would show up as a P1 effect.
    """

    async def close_researcher(
        self, *, task_ctx: TaskContext, checkpoint: CCheckpoint
    ) -> ResearcherHandoff:
        from ..odr.vendor_hooks import _UseVendorCompression

        raise _UseVendorCompression()
