"""A host in front of an unmodified vLLM: the broker runs outside the engine.

**Snapshot, epoch and gate** come from :mod:`.engine_state`, shared with ``AgentHost``: a
scrape of the engine's ``/metrics`` with a measurable age, plus proxy-local exact state, and a
commit refused when its snapshot is superseded or older than ``max_snapshot_age_ns``.

**Materialize.** The staleness gate runs before anything is written. Past the gate the commit
is host-local and all-or-nothing: every chosen sub-request is recorded and handed to the
dispatcher under one lock, and the item leaves the pending set. Each sub-request then goes to
vLLM as its *own* HTTP request on its own worker, so P0's N siblings enter the engine's native
queue independently and none waits on another (C5). A transport failure after the commit is
recorded on that request's outcome; it does not un-commit the item, the same way an
engine-side error would not.

What this host cannot see from outside the engine: the engine can move between the scrape and
the dispatch, and nothing short of an in-scheduler host closes that window. The age gate bounds
it; it does not remove it.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass

from ...contracts.offer import (
    ContractViolation,
    Decision,
    Form,
    MaterializeResult,
    Offer,
    SchedulerSnapshot,
)
from ...contracts.versions import assert_contract
from .engine_state import (
    DEFAULT_MAX_SNAPSHOT_AGE_NS,
    ENGINE_GAUGES,
    EPOCH_METRIC,
    EngineSnapshotGate,
    EngineUnobservable,
    SnapshotRecord,
    parse_prometheus,
)
from .simulated import Dispatch

CONTRACT = "OFFER_INTERFACE_v1.md"
CONTRACT_SHA256 = "f9a400457f00ed7cf918fd202ac0307a9cf9c196c4ce7f22a94f3ce575c93084"

__all__ = [
    "ProxyHost", "RequestBody", "RequestRecord", "SnapshotRecord", "HttpEngine",
    "EngineUnobservable", "parse_prometheus", "ENGINE_GAUGES", "EPOCH_METRIC",
    "DEFAULT_MAX_SNAPSHOT_AGE_NS",
]

@dataclass(frozen=True)
class RequestBody:
    """What one descriptor becomes on the wire. Held by the host, never by the offer."""

    prompt: str
    max_tokens: int


@dataclass
class RequestRecord:
    """One materialized sub-request and what the engine did with it."""

    item_id: str
    form: str
    ordinal: int
    op_class: str
    est_prompt_tokens: int
    max_tokens: int
    snapshot_epoch: str
    snapshot_tick: int
    committed_ns: int
    sent_ns: int = 0
    done_ns: int = 0
    response_id: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    text: str | None = None
    error: str | None = None

    def to_json(self) -> dict:
        return asdict(self)


class HttpEngine:
    """Default transport: ``GET /metrics`` and ``POST /v1/completions`` on one vLLM server.

    One client per worker thread, so concurrent sends never share a connection pool.
    """

    def __init__(self, base_url: str, *, timeout_s: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._local = threading.local()

    def _client(self):
        client = getattr(self._local, "client", None)
        if client is None:
            import httpx

            client = httpx.Client(base_url=self.base_url, timeout=self.timeout_s)
            self._local.client = client
        return client

    def scrape(self) -> str:
        response = self._client().get("/metrics")
        response.raise_for_status()
        return response.text

    def send(self, body: dict) -> dict:
        response = self._client().post("/v1/completions", json=body)
        response.raise_for_status()
        return response.json()


class ProxyHost:
    """``BrokerHost`` for a broker that sits outside an unmodified vLLM."""

    def __init__(
        self,
        *,
        scrape: Callable[[], str],
        send: Callable[[dict], dict],
        model: str,
        sampling: Mapping | None = None,
        max_snapshot_age_ns: int = DEFAULT_MAX_SNAPSHOT_AGE_NS,
        max_workers: int = 64,
        clock_ns: Callable[[], int] = time.time_ns,
        on_complete: Callable[[RequestRecord], None] | None = None,
    ) -> None:
        self._scrape = scrape
        self._send = send
        self.model = model
        self.sampling = dict(sampling or {"temperature": 0.0})
        self._gate = EngineSnapshotGate(clock_ns=clock_ns,
                                        max_snapshot_age_ns=max_snapshot_age_ns)
        self._clock_ns = clock_ns
        self._on_complete = on_complete

        self._lock = threading.Lock()
        self._offers: list[Offer] = []
        self._bodies: dict[tuple[str, Form], tuple[RequestBody, ...]] = {}
        self._inflight = 0
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="proxyhost")
        self._futures: list[Future] = []

        self.dispatches: list[Dispatch] = []
        self.committed: list[Decision] = []
        self.records: list[RequestRecord] = []

    @property
    def max_snapshot_age_ns(self) -> int:
        return self._gate.max_snapshot_age_ns

    @property
    def snapshots(self) -> list[SnapshotRecord]:
        return self._gate.snapshots

    @property
    def gate_log(self) -> list[dict]:
        """One entry per materialize attempt: the snapshot's age at the gate and the verdict."""
        return self._gate.gate_log

    # --- step 1: offers -----------------------------------------------------------------------

    def submit(self, offer: Offer, bodies: Mapping[Form, Sequence[RequestBody]]) -> None:
        """Queue an offer together with the bodies its plans would send. Sends nothing."""
        if set(bodies) != set(offer.forms()):
            raise ContractViolation(
                f"{offer.item_id}: bodies given for {sorted(f.value for f in bodies)}, "
                f"plans offered for {sorted(f.value for f in offer.forms())}")
        frozen: dict[Form, tuple[RequestBody, ...]] = {}
        for form in offer.forms():
            plan = offer.plan_for(form)
            descriptors = plan.requests if form is Form.P0 else (plan.selector,)
            given = tuple(bodies[form])
            if len(given) != len(descriptors):
                raise ContractViolation(
                    f"{offer.item_id} {form.value}: {len(given)} bodies for "
                    f"{len(descriptors)} descriptors")
            for descriptor, body in zip(descriptors, given, strict=True):
                if descriptor.max_tokens != body.max_tokens:
                    raise ContractViolation(
                        f"{offer.item_id} {form.value}: body max_tokens {body.max_tokens} "
                        f"differs from the priced descriptor's {descriptor.max_tokens}")
            frozen[form] = given
        with self._lock:
            if any(o.item_id == offer.item_id for o in self._offers):
                raise ContractViolation(f"{offer.item_id} is already pending")
            self._offers.append(offer)
            for form, given in frozen.items():
                self._bodies[(offer.item_id, form)] = given

    def pending(self) -> Sequence[Offer]:
        with self._lock:
            return tuple(self._offers)

    # --- step 2: snapshot ---------------------------------------------------------------------

    def take_snapshot(self) -> SchedulerSnapshot:
        observation = self._gate.observe(self._scrape)
        with self._lock:
            return self._gate.stamp(observation, {"local.inflight_total": self._inflight})

    # --- steps 4-6: materialize ---------------------------------------------------------------

    def materialize(self, decision: Decision, snapshot: SchedulerSnapshot) -> MaterializeResult:
        with self._lock:
            # The gate. Nothing is materialized above the commit marker below, so a stale
            # snapshot leaves no trace beyond the gate log: no request, no dispatch record, and
            # the offers stay pending.
            gate = self._gate.check(snapshot, decision.admitted)

            by_id = {o.item_id: o for o in self._offers}
            work = []
            for item_id in decision.admitted:
                offer = by_id.get(item_id)
                if offer is None:
                    raise ContractViolation(f"{item_id} was admitted but is not pending")
                form = decision.form[item_id]
                plan = offer.plan_for(form)
                descriptors = plan.requests if form is Form.P0 else (plan.selector,)
                bodies = self._bodies[(item_id, form)]
                work.extend((item_id, form, i, d, b)
                            for i, (d, b) in enumerate(zip(descriptors, bodies, strict=True)))

            # --- commit: from here every admitted item is fully materialized ---
            self._gate.passed(gate)
            admitted = set(decision.admitted)
            self._offers = [o for o in self._offers if o.item_id not in admitted]
            for key in [k for k in self._bodies if k[0] in admitted]:
                del self._bodies[key]
            self.committed.append(decision)
            now = self._clock_ns()
            dispatched: Counter = Counter()
            for item_id, form, ordinal, descriptor, body in work:
                self.dispatches.append(Dispatch(
                    item_id=item_id, form=form, ordinal=ordinal,
                    op_class=descriptor.op_class.value))
                dispatched[item_id] += 1
                self._inflight += 1
                record = RequestRecord(
                    item_id=item_id, form=form.value, ordinal=ordinal,
                    op_class=descriptor.op_class.value,
                    est_prompt_tokens=descriptor.est_uncached_prefill
                    + descriptor.est_cached_prefill,
                    max_tokens=body.max_tokens, snapshot_epoch=snapshot.version.epoch,
                    snapshot_tick=snapshot.version.tick, committed_ns=now)
                # One future per sub-request: siblings share no barrier and no batch.
                self._futures.append(self._pool.submit(self._send_one, record, body))

        return MaterializeResult(
            materialized=tuple(decision.admitted),
            dispatched=dict(dispatched),
            degraded_to_p0=tuple(i for i in decision.admitted
                                 if decision.fail_closed and decision.form[i] is Form.P0),
        )

    def _send_one(self, record: RequestRecord, body: RequestBody) -> None:
        payload = {"model": self.model, "prompt": body.prompt, "max_tokens": body.max_tokens,
                   **self.sampling}
        record.sent_ns = self._clock_ns()
        try:
            response = self._send(payload)
            choice = response["choices"][0]
            usage = response.get("usage") or {}
            record.response_id = response.get("id")
            record.text = choice.get("text")
            record.finish_reason = choice.get("finish_reason")
            record.prompt_tokens = usage.get("prompt_tokens")
            record.completion_tokens = usage.get("completion_tokens")
        except Exception as exc:  # noqa: BLE001 - recorded on the request, never un-commits
            record.error = f"{type(exc).__name__}: {exc}"[:500]
        record.done_ns = self._clock_ns()
        with self._lock:
            self._inflight -= 1
            self.records.append(record)
        if self._on_complete is not None:
            self._on_complete(record)

    # --- lifecycle and inspection -------------------------------------------------------------

    def inflight(self) -> int:
        with self._lock:
            return self._inflight

    def drain(self, timeout_s: float | None = None) -> bool:
        """Wait for every dispatched request to finish. True when nothing is left in flight."""
        with self._lock:
            futures = list(self._futures)
        done, not_done = wait(futures, timeout=timeout_s)
        for future in done:
            future.result()
        return not not_done

    def close(self) -> None:
        self._pool.shutdown(wait=True)

    def forms_of(self, item_id: str) -> set[Form]:
        return {d.form for d in self.dispatches if d.item_id == item_id}

    def dispatch_count(self, item_id: str) -> int:
        return sum(1 for d in self.dispatches if d.item_id == item_id)

    def requests_for(self, item_id: str) -> list[Dispatch]:
        return [d for d in self.dispatches if d.item_id == item_id]


assert_contract(CONTRACT, CONTRACT_SHA256)
