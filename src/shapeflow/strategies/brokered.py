"""Strategies that let a broker choose the form at each handoff, while the agent waits.

In the study the form was fixed per arm before the run started. Here the agent's control flow
changes at both boundaries. At a page batch (H) or a researcher close (C) the agent:

1. prices both forms without sending anything (:mod:`.offer_pricing`);
2. submits the offer to the broker;
3. **suspends** -- the hook is awaited inline on the graph's event loop, so the researcher does
   not proceed -- until the broker decides, or until its own deadline makes it withdraw;
4. runs only the chosen form: vendor's own P0, or the production P1 strategy.

Nothing about the forms themselves changes. P0 at H replays vendor's deferred closures; P0 at C
declines so vendor's ``compress_research`` runs; P1 is the strategy the factory builds for the
arm. A P1 failure after P1 was chosen re-raises, so the adapter's whole-batch fallback (C2) and
the close fallback still apply exactly as they do for a fixed P1 arm, and the broker is told.

Each offer is recorded in the cell's own trajectory (``BROKER_OFFER``, ``BROKER_DECISION``)
between the checkpoint and the reduction it names, so the order checkpoint -> offer -> decision
-> execution is visible in the cell record, not only in the broker's logs.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from typing import Any

from ..broker.client import BrokerClientProtocol, Resolution
from ..broker.codec import offer_digest
from ..contracts.offer import Boundary, Form, Offer, P0Plan
from ..evidence.chunkers import Tokenizer
from ..odr.hooks import ResearcherHandoff, ToolObservation
from .close_visible import CloseSelectionError
from .offer_pricing import (
    P1Price,
    PricingProbe,
    PricingProbeDeclined,
    VendorPrompts,
    price_p0_c,
    price_p0_h,
    price_p1,
)
from .p0 import VendorPageStrategy
from .page_h import PageSelectionError

__all__ = ["BrokeredPageStrategy", "BrokeredCloseStrategy", "BrokerSettings"]


class BrokerSettings:
    """What both brokered halves share for one cell."""

    def __init__(self, *, client: BrokerClientProtocol, cell_token: str, tokenizer: Tokenizer,
                 prompts: VendorPrompts, p0_page_max_tokens: int, p0_close_max_tokens: int,
                 max_model_len: int, deadline_ns: int,
                 log: Callable[[dict], None] | None = None,
                 clock_ns: Callable[[], int] = time.time_ns) -> None:
        self.client = client
        self.cell_token = cell_token
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.p0_page_max_tokens = p0_page_max_tokens
        self.p0_close_max_tokens = p0_close_max_tokens
        self.max_model_len = max_model_len
        self.deadline_ns = deadline_ns
        self.log = log
        self.clock_ns = clock_ns
        self._seen: dict[str, int] = {}

    def item_id(self, boundary: str, checkpoint_digest: str) -> str:
        base = f"{self.cell_token}:{boundary}:{checkpoint_digest[:16]}"
        # A graph retry can reach the same checkpoint twice in one cell; each arrival is its
        # own decision.
        n = self._seen.get(base, 0)
        self._seen[base] = n + 1
        return base if n == 0 else f"{base}:r{n}"


def _emit(kind: str, payload: dict) -> None:
    from ..odr.vendor_hooks import current_run

    run = current_run()
    if run is not None and run.on_event is not None:
        run.on_event(kind, payload)


class _Brokered:
    """The submit / wait / execute / acknowledge sequence both boundaries share."""

    def __init__(self, settings: BrokerSettings, boundary: Boundary) -> None:
        self._settings = settings
        self._boundary = boundary

    async def _offer_and_resolve(self, *, checkpoint, p0: tuple, p1: P1Price,
                                 priced_ns: int) -> tuple[Offer, Resolution]:
        s = self._settings
        now = s.clock_ns()
        offer = Offer(
            item_id=s.item_id(self._boundary.value, checkpoint.digest),
            boundary=self._boundary,
            unit_id=checkpoint.digest,
            p0=P0Plan(requests=p0),
            p1=p1.plan,
            submitted_at_ns=now,
            deadline_ns=now + s.deadline_ns,
        )
        _emit("BROKER_OFFER", {
            "checkpoint": checkpoint.digest, "item_id": offer.item_id,
            "offer_digest": offer_digest(offer),
            "forms": [f.value for f in offer.forms()],
            "p0_requests": len(p0),
            "p1_absent_reason": p1.absent_reason or None,
            "p1_view_sha256": p1.view_sha256 or None,
        })
        resolution = await s.client.resolve(offer)
        _emit("BROKER_DECISION", {
            "checkpoint": checkpoint.digest, "item_id": offer.item_id,
            "form": resolution.form.value, "source": resolution.source,
            "reason": resolution.reason,
        })
        if s.log is not None:
            s.log({"item_id": offer.item_id, "boundary": self._boundary.value,
                   "checkpoint": checkpoint.digest, "offer": offer.content(),
                   "p1_absent_reason": p1.absent_reason, "p1_view_sha256": p1.view_sha256,
                   "p1_probe_failure": p1.probe_failure,
                   "pricing_ns": priced_ns, "waited_ns": resolution.waited_ns,
                   "form": resolution.form.value, "source": resolution.source,
                   "reason": resolution.reason, "ticket": resolution.ticket})
        return offer, resolution

    async def _ack(self, offer: Offer, resolution: Resolution, outcome: str,
                   detail: str = "") -> None:
        s = self._settings
        record = {"form": resolution.form.value, "source": resolution.source,
                  "outcome": outcome, "detail": detail[:300]}
        if s.log is not None:
            s.log({"item_id": offer.item_id, "executed": record, "at_ns": s.clock_ns()})
        await s.client.ack(offer.item_id, record)


class BrokeredPageStrategy(_Brokered):
    """H: wraps the arm's executing page strategy and a factory of probe instances."""

    def __init__(self, *, settings: BrokerSettings, executor: Any,
                 probe_strategy_for: Callable[[PricingProbe], Any],
                 body_for: Callable[..., dict]) -> None:
        super().__init__(settings, Boundary.H)
        self._executor = executor
        self._probe_strategy_for = probe_strategy_for
        self._body_for = body_for
        #: Set by the hook immediately before ``transform_tool_batch`` is awaited.
        self._deferred: dict = {}
        self._ran: Any = None

    # _emit_selection_records reads these off the page strategy; they describe whichever form ran.
    @property
    def last_outcomes(self) -> Sequence[Any]:
        return getattr(self._ran, "last_outcomes", ()) or ()

    @property
    def last_rendered_token_counts(self) -> Sequence[Any]:
        return getattr(self._ran, "last_rendered_token_counts", ()) or ()

    @property
    def last_published_token_counts(self) -> Sequence[Any]:
        return getattr(self._ran, "last_published_token_counts", ()) or ()

    @property
    def last_control_records(self) -> Sequence[Any]:
        return getattr(self._ran, "last_control_records", ()) or ()

    async def transform_tool_batch(self, *, task_ctx, checkpoint) -> Sequence[ToolObservation]:
        # Copied before the first await: the hook rewrites ``_deferred`` for the next batch.
        deferred = dict(self._deferred)
        self._ran = None
        s = self._settings
        started = s.clock_ns()

        p0 = price_p0_h(checkpoint, deferred, prompts=s.prompts, tokenizer=s.tokenizer,
                        max_tokens=s.p0_page_max_tokens)
        if not p0:
            # Every result is snippet-only: neither form makes a model request, so there is no
            # choice to offer. Vendor's rendering runs, as it would under any arm.
            _emit("BROKER_NO_OFFER", {"checkpoint": checkpoint.digest,
                                      "reason": "NO_MODEL_WORK"})
            return await VendorPageStrategy(deferred).transform_tool_batch(
                task_ctx=task_ctx, checkpoint=checkpoint)

        probe = PricingProbe(self._body_for)
        probe_strategy = self._probe_strategy_for(probe)
        failure = ""
        try:
            await probe_strategy.transform_tool_batch(task_ctx=task_ctx, checkpoint=checkpoint)
        except PageSelectionError as exc:
            failure = exc.failure.reason if exc.failure else "UNKNOWN"
        except PricingProbeDeclined:
            failure = ""
        p1 = price_p1(probe, outcomes=getattr(probe_strategy, "last_outcomes", ()),
                      failure_reason=failure, tokenizer=s.tokenizer,
                      max_model_len=s.max_model_len)
        if p1.plan is not None and not p1.candidates:
            raise RuntimeError(
                f"{checkpoint.digest}: P0 has pages but the P1 probe was offered no candidates; "
                "the page registry is not reaching the strategy, which is the inert-arm failure")

        offer, resolution = await self._offer_and_resolve(
            checkpoint=checkpoint, p0=p0, p1=p1, priced_ns=s.clock_ns() - started)

        if resolution.form is Form.P1:
            self._ran = self._executor
            try:
                observations = await self._executor.transform_tool_batch(
                    task_ctx=task_ctx, checkpoint=checkpoint)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # The adapter publishes the whole batch as P0 (C2); the broker is told P1 failed.
                await self._ack(offer, resolution, "P1_FAILED", f"{type(exc).__name__}: {exc}")
                raise
            await self._ack(offer, resolution, "P1_OK")
            return observations

        try:
            observations = await VendorPageStrategy(deferred).transform_tool_batch(
                task_ctx=task_ctx, checkpoint=checkpoint)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._ack(offer, resolution, "P0_FAILED", f"{type(exc).__name__}: {exc}")
            raise
        await self._ack(offer, resolution, "P0_OK")
        return observations


class BrokeredCloseStrategy(_Brokered):
    """C: wraps the arm's executing close strategy and a factory of probe instances."""

    def __init__(self, *, settings: BrokerSettings, executor: Any,
                 probe_strategy_for: Callable[[PricingProbe], Any],
                 body_for: Callable[..., dict]) -> None:
        super().__init__(settings, Boundary.C)
        self._executor = executor
        self._probe_strategy_for = probe_strategy_for
        self._body_for = body_for
        self._ran: Any = None

    @property
    def last_outcome(self) -> Any:
        return getattr(self._ran, "last_outcome", None)

    @property
    def last_outcomes(self) -> Sequence[Any]:
        return getattr(self._ran, "last_outcomes", ()) or ()

    @property
    def last_control_records(self) -> Sequence[Any]:
        return getattr(self._ran, "last_control_records", ()) or ()

    async def close_researcher(self, *, task_ctx, checkpoint) -> ResearcherHandoff:
        self._ran = None
        s = self._settings
        started = s.clock_ns()
        p0 = price_p0_c(checkpoint, prompts=s.prompts, tokenizer=s.tokenizer,
                        max_tokens=s.p0_close_max_tokens)

        probe = PricingProbe(self._body_for)
        probe_strategy = self._probe_strategy_for(probe)
        failure = ""
        try:
            await probe_strategy.close_researcher(task_ctx=task_ctx, checkpoint=checkpoint)
        except CloseSelectionError as exc:
            failure = exc.failure.reason if exc.failure else "UNKNOWN"
        except PricingProbeDeclined:
            failure = ""
        outcome = getattr(probe_strategy, "last_outcome", None)
        p1 = price_p1(probe, outcomes=[outcome] if outcome is not None else [],
                      failure_reason=failure, tokenizer=s.tokenizer,
                      max_model_len=s.max_model_len)

        offer, resolution = await self._offer_and_resolve(
            checkpoint=checkpoint, p0=p0, p1=p1, priced_ns=s.clock_ns() - started)

        if resolution.form is Form.P1:
            self._ran = self._executor
            try:
                handoff = await self._executor.close_researcher(
                    task_ctx=task_ctx, checkpoint=checkpoint)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # run_close_strategy falls back to vendor compression; the broker is told.
                await self._ack(offer, resolution, "P1_FAILED", f"{type(exc).__name__}: {exc}")
                raise
            await self._ack(offer, resolution, "P1_OK")
            return handoff

        from ..odr.vendor_hooks import _UseVendorCompression

        await self._ack(offer, resolution, "P0_DECLINED_TO_VENDOR")
        raise _UseVendorCompression()
