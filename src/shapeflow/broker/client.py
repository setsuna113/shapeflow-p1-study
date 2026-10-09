"""The agent's side of the broker: submit an offer, wait for the decision, report what ran.

``resolve`` returns exactly one form for an item, and it is the form the agent must run:

- the broker decided -> that form;
- the agent stopped waiting (its deadline passed) -> it withdraws; if the item was still pending
  it is withdrawn and the agent runs **P0 fail-closed**, and if the broker had decided first the
  decision stands and the agent runs it;
- the broker cannot be reached to settle that race -> P0 fail-closed, recorded as such, so an
  audit can flag any ticket the agent did not honour;
- the broker does not know the item (it restarted) -> P0 fail-closed.

So every offered item is run exactly once, and never as a form nobody chose or recorded. A
cancelled wait withdraws, shielded from the cancellation, and re-raises: nothing is run and no
P0 result is fabricated for a researcher that is being torn down.
"""

from __future__ import annotations

import asyncio
import time
import urllib.parse
from dataclasses import dataclass
from typing import Protocol

from ..contracts.offer import Form, Offer
from ..contracts.versions import assert_contract

CONTRACT = "OFFER_INTERFACE_v1.md"
CONTRACT_SHA256 = "f9a400457f00ed7cf918fd202ac0307a9cf9c196c4ce7f22a94f3ce575c93084"

__all__ = ["Resolution", "BrokerClientProtocol", "BrokerClient", "BROKER", "FAIL_CLOSED"]

BROKER, FAIL_CLOSED = "broker", "fail_closed"


@dataclass(frozen=True)
class Resolution:
    form: Form
    source: str            # BROKER | FAIL_CLOSED
    reason: str = ""
    ticket: dict | None = None
    waited_ns: int = 0


class BrokerClientProtocol(Protocol):
    async def resolve(self, offer: Offer) -> Resolution: ...

    async def ack(self, item_id: str, outcome: dict) -> None: ...


class _Unreachable(Exception):
    pass


class _Unknown(Exception):
    pass


class BrokerClient:
    """Loopback HTTP client for :class:`~shapeflow.broker.service.BrokerService`.

    ``grace_ns`` is how long past the offer's own ``deadline_ns`` the agent keeps waiting before
    it withdraws: long enough for the broker to reach a deadline-forced decision on its next
    tick, so the fail-closed path is the exception it is meant to be.
    """

    def __init__(self, base_url: str, *, grace_ns: int = 5_000_000_000,
                 poll_ms: int = 2000, request_timeout_s: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.grace_ns = grace_ns
        self.poll_ms = poll_ms
        self.request_timeout_s = request_timeout_s

    async def _call(self, method: str, path: str, body: dict | None = None,
                    timeout_s: float | None = None) -> dict:
        import httpx

        try:
            async with httpx.AsyncClient(timeout=timeout_s or self.request_timeout_s) as client:
                response = await client.request(method, self.base_url + path, json=body)
        except httpx.HTTPError as exc:
            raise _Unreachable(f"{type(exc).__name__}: {exc}") from exc
        if response.status_code == 404:
            raise _Unknown(path)
        if response.status_code != 200:
            raise _Unreachable(f"HTTP {response.status_code}: {response.text[:200]}")
        return response.json()

    @staticmethod
    def _path(item_id: str, suffix: str = "") -> str:
        return f"/v1/offers/{urllib.parse.quote(item_id, safe='')}{suffix}"

    async def resolve(self, offer: Offer) -> Resolution:
        started = time.monotonic_ns()
        deadline_ns = (offer.deadline_ns or time.time_ns()) + self.grace_ns
        item_id = offer.item_id

        def waited() -> int:
            return time.monotonic_ns() - started

        def decided(status: dict, source: str = BROKER, reason: str = "") -> Resolution:
            ticket = status["ticket"]
            return Resolution(form=Form(ticket["form"]), source=source,
                              reason=reason or ticket.get("reason", ""), ticket=ticket,
                              waited_ns=waited())

        try:
            submitted = False
            while True:
                try:
                    if not submitted:
                        await self._call("POST", "/v1/offers", {"offer": offer.content()})
                        submitted = True
                    remaining_ms = (deadline_ns - time.time_ns()) // 1_000_000
                    if remaining_ms <= 0:
                        break
                    wait_ms = int(min(self.poll_ms, remaining_ms))
                    status = await self._call(
                        "GET", self._path(item_id, f"?wait_ms={wait_ms}"),
                        timeout_s=wait_ms / 1000 + self.request_timeout_s)
                    if status["state"] == "DECIDED":
                        return decided(status)
                except _Unreachable:
                    if time.time_ns() >= deadline_ns:
                        break
                    await asyncio.sleep(0.2)
                except _Unknown:
                    return Resolution(form=Form.P0, source=FAIL_CLOSED, reason="BROKER_LOST",
                                      waited_ns=waited())
            # Deadline passed: settle the race with the broker before running anything.
            try:
                status = await self._call(
                    "POST", self._path(item_id, "/withdraw"), {"reason": "AGENT_WAIT_TIMEOUT"})
            except _Unknown:
                return Resolution(form=Form.P0, source=FAIL_CLOSED, reason="BROKER_LOST",
                                  waited_ns=waited())
            except _Unreachable:
                return Resolution(form=Form.P0, source=FAIL_CLOSED,
                                  reason="BROKER_UNREACHABLE", waited_ns=waited())
            if status["state"] == "DECIDED":
                return decided(status, reason="DECIDED_BEFORE_WITHDRAW")
            return Resolution(form=Form.P0, source=FAIL_CLOSED, reason="AGENT_WAIT_TIMEOUT",
                              waited_ns=waited())
        except asyncio.CancelledError:
            try:
                await asyncio.shield(asyncio.wait_for(
                    self._call("POST", self._path(item_id, "/withdraw"),
                               {"reason": "CANCELLED"}), timeout=2.0))
            except Exception:  # noqa: BLE001, S110 - best effort; the cancellation is what matters
                pass
            raise

    async def ack(self, item_id: str, outcome: dict) -> None:
        """Best effort: the ack is evidence, and losing one must not fail the researcher."""
        try:
            await self._call("POST", self._path(item_id, "/executed"), {"outcome": outcome})
        except (_Unreachable, _Unknown):
            pass


assert_contract(CONTRACT, CONTRACT_SHA256)
