"""The broker as a service: one tick loop over an ``AgentHost``, reached by agents over loopback.

Agents run one per process -- the graph driver refuses two cells in one interpreter -- so a joint
decision over several agents needs the broker in a process of its own. The tick loop calls the
shared :func:`~shapeflow.broker.driver.run_tick` unchanged; this module only schedules it and
gives agents a way in.

Routes (JSON, loopback only):

- ``POST /v1/offers`` ``{"offer": Offer.content()}`` -- submit; idempotent for identical content.
- ``GET /v1/offers/<id>?wait_ms=N`` -- long-poll until the item leaves PENDING (N <= 5000).
- ``POST /v1/offers/<id>/withdraw`` ``{"reason": ...}`` -- withdraw if still pending.
- ``POST /v1/offers/<id>/executed`` ``{"outcome": {...}}`` -- record what the agent ran.
- ``GET /healthz``.

A withdrawal takes the tick lock, so it lands between ticks and never inside one: a decision is
never computed over an item that its agent has already given up on.

Every submission, tick, withdrawal and acknowledgement is appended to a JSONL log as it happens,
so a crashed run still leaves the evidence of everything it decided.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..contracts.offer import ContractViolation
from ..contracts.versions import assert_contract
from .codec import offer_from_content
from .driver import DEFAULT_SOLVE_BUDGET_NS, TickOutcome, run_tick
from .hosts.agent import AgentHost, UnknownItem
from .hosts.engine_state import EngineUnobservable

CONTRACT = "OFFER_INTERFACE_v1.md"
CONTRACT_SHA256 = "f9a400457f00ed7cf918fd202ac0307a9cf9c196c4ce7f22a94f3ce575c93084"

__all__ = ["BrokerService", "MAX_WAIT_MS"]

LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})
MAX_WAIT_MS = 5000


class _JsonlLog:
    def __init__(self, path: Path | None) -> None:
        self._lock = threading.Lock()
        self._f = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._f = path.open("a", encoding="utf-8", buffering=1)

    def write(self, row: dict) -> None:
        if self._f is None:
            return
        line = json.dumps(row, ensure_ascii=False, sort_keys=True)
        with self._lock:
            self._f.write(line + "\n")

    def close(self) -> None:
        if self._f is not None:
            self._f.close()


class BrokerService:
    def __init__(
        self,
        host: AgentHost,
        broker,
        *,
        tick_ms: float = 250.0,
        budget_ns: int = DEFAULT_SOLVE_BUDGET_NS,
        max_retries: int = 2,
        log_dir: Path | None = None,
    ) -> None:
        self.host = host
        self.broker = broker
        self.tick_ms = tick_ms
        self.budget_ns = budget_ns
        self.max_retries = max_retries
        self._tick_lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._server: ThreadingHTTPServer | None = None
        self.ticks_run = 0
        self.ticks_with_offers = 0
        log_dir = Path(log_dir) if log_dir is not None else None
        self._offers_log = _JsonlLog(log_dir / "offers.jsonl" if log_dir else None)
        self._ticks_log = _JsonlLog(log_dir / "ticks.jsonl" if log_dir else None)
        self._agents_log = _JsonlLog(log_dir / "agent_events.jsonl" if log_dir else None)
        self._log_dir = log_dir

    # --- ticking ------------------------------------------------------------------------------

    def tick_once(self) -> TickOutcome | None:
        with self._tick_lock:
            pending = [o.item_id for o in self.host.pending()]
            self.ticks_run += 1
            if not pending:
                return None
            self.ticks_with_offers += 1
            row: dict = {"tick": self.ticks_run, "wall_ns": time.time_ns(), "pending": pending}
            try:
                outcome = run_tick(self.host, self.broker, budget_ns=self.budget_ns,
                                   max_retries=self.max_retries)
            except EngineUnobservable as exc:
                row["error"] = f"EngineUnobservable: {exc}"
                self._ticks_log.write(row)
                return None
            snapshot = self.host.snapshots[-1] if self.host.snapshots else None
            row.update({
                "attempts": outcome.attempts,
                "stale_retries": outcome.stale_retries,
                "budget_overrun": outcome.budget_overrun,
                "fail_closed": outcome.fail_closed,
                "committed": outcome.committed,
                "reason": outcome.reason,
                "snapshot": None if snapshot is None else {
                    "epoch": snapshot.epoch, "tick": snapshot.tick,
                    "taken_at_ns": snapshot.taken_at_ns, "scrape_ns": snapshot.scrape_ns,
                    "features": snapshot.features,
                },
                "decision": outcome.decision.content() if outcome.decision else None,
                "decision_digest": outcome.decision.digest if outcome.decision else None,
                "result": outcome.result.content() if outcome.result else None,
            })
            self._ticks_log.write(row)
            return outcome

    def _tick_loop(self) -> None:
        while not self._stop.is_set():
            started = time.perf_counter()
            try:
                self.tick_once()
            except Exception as exc:  # noqa: BLE001 - logged; a tick must not kill the service
                self._ticks_log.write({"tick": self.ticks_run, "wall_ns": time.time_ns(),
                                       "error": f"{type(exc).__name__}: {exc}"})
            remaining = self.tick_ms / 1000 - (time.perf_counter() - started)
            if remaining > 0:
                self._stop.wait(remaining)

    # --- agent operations (also used directly by in-process clients) --------------------------

    def submit(self, doc: dict) -> dict:
        offer = offer_from_content(doc)
        status = self.host.submit(offer)
        self._offers_log.write({"received_ns": status["received_ns"], "offer": offer.content(),
                                "digest": status["digest"]})
        return status

    def withdraw(self, item_id: str, reason: str) -> dict:
        with self._tick_lock:
            status = self.host.withdraw(item_id, reason)
        self._agents_log.write({"event": "withdraw", "item_id": item_id, "reason": reason,
                                "state": status["state"], "wall_ns": time.time_ns()})
        return status

    def ack(self, item_id: str, outcome: dict) -> dict:
        status = self.host.ack(item_id, outcome)
        self._agents_log.write({"event": "executed", "item_id": item_id, "outcome": outcome,
                                "wall_ns": time.time_ns()})
        return status

    # --- lifecycle ----------------------------------------------------------------------------

    def start(self, *, bind_host: str = "127.0.0.1", port: int = 0) -> int:
        if bind_host not in LOOPBACK:
            raise ValueError(f"refusing to bind the broker to {bind_host!r}; loopback only")
        self._server = ThreadingHTTPServer((bind_host, port), _handler_for(self))
        self._server.daemon_threads = True
        for target in (self._server.serve_forever, self._tick_loop):
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            self._threads.append(thread)
        return int(self._server.server_address[1])

    def stop(self) -> None:
        self._stop.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        for thread in self._threads:
            thread.join(timeout=5)
        if self._log_dir is not None:
            for name, rows in (("snapshots.jsonl", [s.__dict__ for s in self.host.snapshots]),
                               ("gate.jsonl", self.host.gate_log),
                               ("items.jsonl", self.host.items())):
                log = _JsonlLog(self._log_dir / name)
                for row in rows:
                    log.write(row)
                log.close()
        for log in (self._offers_log, self._ticks_log, self._agents_log):
            log.close()

    def health(self) -> dict:
        return {"ok": True, "pending": len(self.host.pending()), "ticks": self.ticks_run,
                "ticks_with_offers": self.ticks_with_offers}


def _handler_for(service: BrokerService):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002 - stdlib signature
            return

        def _reply(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw.decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("request body must be a JSON object")
            return body

        def _route(self, method: str) -> None:
            parsed = urllib.parse.urlsplit(self.path)
            parts = [urllib.parse.unquote(p) for p in parsed.path.split("/") if p]
            try:
                if method == "GET" and parts == ["healthz"]:
                    return self._reply(200, service.health())
                if parts[:2] == ["v1", "offers"]:
                    if method == "POST" and len(parts) == 2:
                        return self._reply(200, service.submit(self._body()["offer"]))
                    if method == "GET" and len(parts) == 3:
                        query = urllib.parse.parse_qs(parsed.query)
                        wait_ms = min(int(query.get("wait_ms", ["0"])[0]), MAX_WAIT_MS)
                        return self._reply(200, service.host.wait(parts[2], wait_ms / 1000))
                    if method == "POST" and len(parts) == 4 and parts[3] == "withdraw":
                        reason = str(self._body().get("reason", ""))
                        return self._reply(200, service.withdraw(parts[2], reason))
                    if method == "POST" and len(parts) == 4 and parts[3] == "executed":
                        outcome = self._body().get("outcome", {})
                        return self._reply(200, service.ack(parts[2], dict(outcome)))
                return self._reply(404, {"error": f"no route {method} {parsed.path}"})
            except UnknownItem as exc:
                return self._reply(404, {"error": f"unknown item {exc}"})
            except ContractViolation as exc:
                return self._reply(409, {"error": str(exc)})
            except (KeyError, ValueError, TypeError) as exc:
                return self._reply(400, {"error": f"{type(exc).__name__}: {exc}"})

        def do_GET(self):  # noqa: N802 - stdlib naming
            self._route("GET")

        def do_POST(self):  # noqa: N802 - stdlib naming
            self._route("POST")

    return Handler


assert_contract(CONTRACT, CONTRACT_SHA256)
