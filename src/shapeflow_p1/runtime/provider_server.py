"""The provider: the only process that holds a credential, and the only way out.

Every external call in the study -- Tavily acquisition, DeepSeek judging, and local vLLM
inference -- goes through this service. Runner, inference and evaluator processes never read
``/etc/shapeflow/*.key``; they hold a role token that opens a *route*, and the routes that can
reach a credential are not the routes the runner is allowed to call. That is a structural
property of the deployment, not a rule someone has to remember.

Four things this module is built around.

**Admission before dispatch.** Nothing is sent until a worst-case reservation has been won
inside a SQLite transaction (:mod:`shapeflow_p1.experiment.budget`). After the response we
settle at the reported usage and the surplus is released. A timeout *after* send keeps its
worst-case charge and ends ``FAILED_UNKNOWN`` -- "we lost contact" is never recorded as "no
call happened", because a call we cannot account for is the one most likely to have been billed.

**The client never supplies a key.** Request bodies carry :data:`PROVIDER_KEY_PLACEHOLDER`
where a credential would go. A body carrying anything else in that slot is rejected with 400,
so a compromised or careless client cannot smuggle its own credential out through this service,
and a real key can never appear in a client's memory, argv or logs.

**Every call is tagged, or it is an incident.** vLLM traffic originates inside vendor ODR, which
cannot be made to set headers. So the cell identity travels in the *URL path*
(``/v1/cell/<token>/chat/completions``) -- the runner registers a cell, sets ``OPENAI_BASE_URL``
to that path, and every model call made while that cell runs is attributed to it without any
shared mutable state that concurrent researchers could race on. The op class comes from a model
alias (``qwen-research``/``qwen-summarize``/``qwen-compress``/``qwen-final``/``qwen-selector``)
that the provider rewrites to the one served model, so P0 and P1 issue byte-identical upstream
requests while remaining separable in the work ledger.

**Redaction happens before storage, not after.** The request bytes are the ones carrying the
credential, so :class:`~shapeflow_p1.providers.external_call_ledger.ExternalCallLedger` stores
their redacted form and the raw bytes never reach the object store or a log handler.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import socket
import socketserver
import struct
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from ..experiment.budget import Budget, BudgetExceeded
from ..experiment.ledger import Ledger
from ..hashing import derive_id, sha256_hex
from ..object_store import ObjectStore
from ..providers.external_call_ledger import (
    CallAlreadyCommitted,
    CallNotReplayable,
    ExternalCallLedger,
)
from ..providers.retry import classify_status
from ..secrets import SecretRedactor
from .request_tags import REMOTE_ALLOWED_OPS, OpClass

__all__ = [
    "PROVIDER_KEY_PLACEHOLDER",
    "ROLE_ROUTES",
    "ProviderError",
    "ProviderConfig",
    "CellRegistration",
    "UpstreamReply",
    "ProviderService",
    "RoleTokens",
    "peer_uid_of_tcp",
    "make_handler",
    "serve_forever",
]

#: What a client puts where a credential would go. The provider substitutes the real value.
#: Deliberately not secret-shaped, so a placeholder left in a log is obviously not a leak.
PROVIDER_KEY_PLACEHOLDER = "@SHAPEFLOW_PROVIDER@"

#: Route -> the roles allowed to call it. The runner has no route that touches a credential:
#: acquisition is the steward's, judging is the evaluator's. This is the boundary; the token
#: check below only enforces it.
ROLE_ROUTES: dict[str, frozenset[str]] = {
    "exa.search": frozenset({"steward"}),
    "tavily.search": frozenset({"steward"}),
    "deepseek.chat": frozenset({"steward", "evaluator"}),
    "chat.completions": frozenset({"runner", "steward"}),
    "cells.register": frozenset({"runner"}),
    # Sanitized measurements only: no prompt or response bytes, credentials, or evaluator truth.
    "cells.work": frozenset({"runner", "evaluator"}),
    # Closed aggregate attestation only. The runner cannot read the provider's 0700 tree.
    "canary.audit": frozenset({"runner"}),
    "healthz": frozenset({"runner", "steward", "evaluator", "infer"}),
    "readyz": frozenset({"runner", "steward", "evaluator", "infer"}),
    "credentials.probe": frozenset({"runner", "steward", "evaluator"}),
}

#: Model alias -> op class. The alias is what ODR is configured with; the provider rewrites it
#: to the one served model, so the alias separates costs without changing the upstream request.
DEFAULT_MODEL_ALIASES: dict[str, OpClass] = {
    "qwen-research": OpClass.RESEARCHER_REACT,
    "qwen-summarize": OpClass.PAGE_P0_SUMMARY,
    "qwen-compress": OpClass.COMPRESSOR_P0,
    "qwen-final": OpClass.FINAL_WRITER,
    "qwen-supervisor": OpClass.SUPERVISOR_CONTINUE,
    "qwen-selector-page": OpClass.PAGE_P1_SELECTOR_LOCAL,
    "qwen-selector-page-global": OpClass.PAGE_P1_SELECTOR_GLOBAL,
    "qwen-selector-close": OpClass.COMPRESSOR_P1_SELECTOR,
    # The SHORT_PROSE controls are still the arm's selector work and belong in the same op
    # class; the arm id is what separates them, so the treatment-work total stays complete.
    "qwen-prose-page": OpClass.PAGE_P1_SELECTOR_LOCAL,
    "qwen-prose-close": OpClass.COMPRESSOR_P1_SELECTOR,
}

_TOKEN_RE = re.compile(r"^[A-Za-z0-9_\-]{24,128}$")
_CELL_RE = re.compile(r"^[A-Za-z0-9_\-]{8,128}$")

_VALIDATORS: dict[str, Any] = {}


def _schema_path() -> Path:
    """Locate schemas/ whether running from the source tree or an editable install."""
    candidate = Path(__file__).resolve().parents[3] / "schemas" / "provider_request.schema.json"
    if candidate.exists():
        return candidate
    root = os.environ.get("SHAPEFLOW_REPO")
    if root:
        override = Path(root) / "schemas" / "provider_request.schema.json"
        if override.exists():
            return override
    raise ProviderError(500, "provider_request.schema.json not found; set SHAPEFLOW_REPO")


def _validator_for(route: str):
    """A closed-schema validator per route, built once.

    The schema is a hard allowlist: a field nobody enumerated is refused by name instead of
    being forwarded to a provider or to the engine. That is what makes "no arbitrary body
    forwarding" a property of the code and not a promise in a document.
    """
    if route in _VALIDATORS:
        return _VALIDATORS[route]
    from jsonschema import Draft202012Validator

    document = json.loads(_schema_path().read_text(encoding="utf-8"))
    # Keep `$defs` alongside the `$ref` so the shared `message` definition resolves inside the
    # same schema resource. Lifting the sub-schema out on its own re-roots `#/$defs/...` and the
    # reference dangles.
    _VALIDATORS[route] = Draft202012Validator({
        "$schema": document["$schema"],
        "$defs": document["$defs"],
        "$ref": f"#/$defs/{route}",
    })
    return _VALIDATORS[route]


def validate_request(route: str, body: Mapping[str, Any]) -> None:
    """Reject anything the route's closed schema does not describe, naming the offending field."""
    from jsonschema import ValidationError

    try:
        _validator_for(route).validate(dict(body))
    except ValidationError as e:
        where = ".".join(str(p) for p in e.absolute_path) or "<root>"
        raise ProviderError(400, f"{route} body rejected at {where}: {e.message}") from e


class ProviderError(RuntimeError):
    """A request the provider refuses. Carries the HTTP status it should produce."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class ResponseTooLarge(RuntimeError):
    """An upstream body too large to freeze verbatim.

    Deliberately not a :class:`ProviderError`: it is raised *after* the bytes arrived, so the
    caller has to settle the attempt and the reservation before turning it into a response.
    """


# --- configuration ----------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderConfig:
    """Everything the provider needs, resolved once at start and then frozen.

    Paths, not values: the credential file paths are recorded, the credentials themselves are
    read into memory once and registered with the redactor. Nothing here is ever serialized.
    """

    bind_host: str = "127.0.0.1"
    bind_port: int = 8787
    unix_socket: Optional[str] = None
    token_dir: str = "/run/shapeflow/tokens"

    exa_endpoint: str = "https://api.exa.ai/search"
    tavily_endpoint: str = "https://api.tavily.com/search"
    deepseek_base_url: str = "https://api.deepseek.com"
    vllm_base_url: str = "http://127.0.0.1:8000"

    served_model: str = ""
    # Qwen3 reasoning is off in the frozen stack; the provider enforces it per request because
    # vLLM only honours it through chat_template_kwargs.
    disable_thinking: bool = True
    model_aliases: Mapping[str, OpClass] = field(
        default_factory=lambda: dict(DEFAULT_MODEL_ALIASES))

    request_timeout_seconds: float = 120.0
    inference_timeout_seconds: float = 900.0
    # Isolated causal mode serializes the actual upstream vLLM dispatches. Zero is deliberately
    # unbounded for the operational batching/APC study. This is enforced by ProviderService, not
    # merely declared in stack.yaml.
    max_upstream_inflight: int = 0
    # Protocol §3.5 retry_policy.max_consecutive_failures. Zero disables the breaker.
    max_consecutive_failures: int = 5
    # Ceiling on a frozen response body. A response above this is *not* truncated and stored:
    # truncation produced invalid JSON that was still marked COMMITTED, so the call was bought
    # and permanently unreplayable (every later read raised 409). Oversize now fails the
    # attempt instead, which leaves the call re-fetchable. Sized well above any real Exa,
    # Tavily or DeepSeek response so it is a corruption guard, not a routine limit.
    max_frozen_response_bytes: int = 8 * 1024 * 1024

    # Worst-case reservations. Admission control is only meaningful if the reserved amount is
    # an upper bound on what the call can cost, so these are ceilings, not estimates.
    exa_usd_worst_case: float = 0.007
    tavily_credits_worst_case: float = 2.0
    deepseek_input_tokens_worst_case: float = 32000.0
    deepseek_output_tokens_worst_case: float = 8000.0
    deepseek_usd_worst_case: float = 0.128
    gpu_seconds_worst_case: float = 900.0

    # Pricing snapshot (source + retrieval date live in configs/judge.yaml and are hashed
    # into the protocol SHA). USD is derived here so the ledger holds money, not just tokens.
    # These defaults mirror that snapshot rather than an older, cheaper one: when they
    # diverged, every test priced a call at a quarter of what production charged for it, and
    # the test named "usd is derived from reported usage" passed against the wrong rate.
    deepseek_usd_per_1m_input: float = 2.00
    deepseek_usd_per_1m_output: float = 8.00

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ProviderConfig":
        """Build from the ``provider:`` block of configs/week1.yaml, rejecting unknown keys.

        An unknown key is an error rather than an ignored typo: a misspelled timeout that
        silently kept the default is exactly the kind of drift the config hash exists to catch.
        """
        known = {f for f in cls.__dataclass_fields__ if f != "model_aliases"}
        unknown = sorted(set(data) - known - {"model_aliases"})
        if unknown:
            raise ProviderError(500, f"unknown provider config keys: {unknown}")
        kwargs: dict[str, Any] = {k: data[k] for k in known if k in data}
        aliases = data.get("model_aliases")
        if aliases:
            kwargs["model_aliases"] = {
                str(alias): OpClass(str(op)) for alias, op in aliases.items()
            }
        config = cls(**kwargs)
        if config.max_upstream_inflight < 0:
            raise ProviderError(500, "max_upstream_inflight must be zero or a positive integer")
        return config


@dataclass(frozen=True)
class CellRegistration:
    """One (task, arm, variant, replicate) cell the runner is about to execute."""

    cell_token: str
    run_id: str
    task_id: str
    arm_id: str
    variant_id: str
    replicate_id: str
    work_key: str
    layer: str = "causal"


@dataclass(frozen=True)
class UpstreamReply:
    """What an upstream returned, normalized so every provider path settles the same way."""

    status: int
    payload: dict
    elapsed_s: float
    request_id: str = ""
    returned_model: str = ""
    system_fingerprint: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: Optional[int] = None
    credits: Optional[float] = None


#: Injected so tests drive the provider with a fake upstream and no network. Returns the raw
#: status and decoded body; every provider-specific interpretation happens in the service.
Upstream = Callable[[str, dict, dict, float], tuple[int, dict, float]]


# --- role tokens and peer identity -------------------------------------------------------


class RoleTokens:
    """Bearer tokens, one per role, each file owned and readable only by that role's uid.

    The token proves which identity is calling. It is a capability, not a secret worth
    protecting from disclosure in the credential sense -- but it is still registered with the
    redactor so a token echoed in an error body never reaches a log.
    """

    def __init__(self, tokens: Mapping[str, str]) -> None:
        for role, token in tokens.items():
            if not _TOKEN_RE.match(token):
                raise ProviderError(500, f"role token for {role!r} is not a valid token shape")
        self._by_token = {token: role for role, token in tokens.items()}
        if len(self._by_token) != len(tokens):
            raise ProviderError(500, "two roles share one token; roles would be indistinguishable")

    @classmethod
    def load(cls, token_dir: str | Path, roles: tuple[str, ...] = (
        "runner", "steward", "evaluator", "infer",
    )) -> "RoleTokens":
        directory = Path(token_dir)
        tokens: dict[str, str] = {}
        for role in roles:
            path = directory / f"{role}.token"
            if not path.exists():
                continue
            tokens[role] = path.read_text(encoding="utf-8").strip()
        if not tokens:
            raise ProviderError(500, f"no role tokens found under {directory}")
        return cls(tokens)

    def role_of(self, token: Optional[str]) -> Optional[str]:
        return self._by_token.get(token or "")

    def register_with(self, redactor: SecretRedactor) -> None:
        for token in self._by_token:
            redactor.register(token, label="role-token")


def peer_uid_of_tcp(local: tuple[str, int], remote: tuple[str, int]) -> Optional[int]:
    """The uid owning the peer end of a loopback TCP connection, or None if not determinable.

    Defence in depth behind the bearer token: a stolen token used from a different account is
    still refused. ``local``/``remote`` are from the *server's* point of view, so the peer's
    socket is the mirror image. Returns None off Linux or when /proc is unreadable -- the caller
    treats that as "unverified", never as "verified".
    """
    try:
        with open("/proc/net/tcp", encoding="ascii") as fh:
            lines = fh.read().splitlines()[1:]
    except OSError:
        return None

    def hexaddr(ip: str, port: int) -> str:
        try:
            packed = socket.inet_aton(ip)
        except OSError:
            return ""
        # /proc/net/tcp renders the address little-endian as a 32-bit word.
        return f"{int.from_bytes(packed[::-1], 'big'):08X}:{port:04X}"

    want_local = hexaddr(remote[0], remote[1])   # peer's local end == our remote end
    want_remote = hexaddr(local[0], local[1])
    if not want_local or not want_remote:
        return None
    for line in lines:
        parts = line.split()
        if len(parts) < 8:
            continue
        if parts[1].upper() == want_local and parts[2].upper() == want_remote:
            try:
                return int(parts[7])
            except ValueError:
                return None
    return None


# --- the service ---------------------------------------------------------------------------


class ProviderService:
    """Route handling, admission control and settlement. No HTTP, so it is directly testable."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        ledger: Ledger,
        budget: Budget,
        store: ObjectStore,
        redactor: SecretRedactor,
        tokens: RoleTokens,
        upstream: Upstream,
        exa_key: Optional[str] = None,
        tavily_key: Optional[str] = None,
        deepseek_key: Optional[str] = None,
        clock: Callable[[], float] = time.time,
        allowed_uids: Optional[Mapping[str, int]] = None,
    ) -> None:
        self._cfg = config
        self._ledger = ledger
        self._budget = budget
        self._calls = ExternalCallLedger(ledger, budget, store, redactor)
        self._redactor = redactor
        self._tokens = tokens
        self._upstream = upstream
        self._exa_key = exa_key
        self._tavily_key = tavily_key
        self._deepseek_key = deepseek_key
        self._clock = clock
        self._allowed_uids = dict(allowed_uids or {})
        self._cells: dict[str, CellRegistration] = {}
        self._nonce_counter = 0
        self._store = store
        # Consecutive upstream failures per provider. Protocol §3.5 caps these; 262
        # consecutive 401s is what happens when nothing does.
        self._consecutive_failures: dict[str, int] = {}
        self._lock = threading.RLock()
        self._events: list[dict] = []
        self._inference_gate = (
            threading.BoundedSemaphore(config.max_upstream_inflight)
            if config.max_upstream_inflight > 0
            else None
        )
        self._ready = False
        tokens.register_with(redactor)
        for key in (exa_key, tavily_key, deepseek_key):
            if key:
                redactor.register(key, label="provider")

    # --- lifecycle ------------------------------------------------------------------

    def reconcile_on_start(self) -> dict[str, int]:
        """Close out calls a previous process left in flight.

        A row still in ``SENT``/``RESPONSE_STORED``/``VALIDATED`` means the provider died with a
        request outstanding. We cannot know whether it was billed, so it keeps its worst-case
        reservation and ends ``FAILED_UNKNOWN``. Rows that never got past ``INTENT``/
        ``BUDGET_RESERVED`` provably never went out, so their reservation is released in full.
        """
        counts = {"failed_unknown": 0, "released": 0}
        for attempt in self._calls.live_attempts():
            group = self._group_for(attempt.attempt_id)
            if self._calls.attempt_was_sent(attempt.attempt_id):
                self._calls.fail_unknown(attempt, group, error_class="provider_restart")
                counts["failed_unknown"] += 1
            else:
                self._calls.fail_before_send(attempt, group, error_class="provider_restart")
                counts["released"] += 1
        # Keep inference call ids monotonic across restarts. A counter that restarts at 1
        # would let a post-restart request collide with a pre-restart call id and silently
        # attach itself to that call's history.
        with self._ledger.lock:
            row = self._ledger.raw_connection.execute(
                "SELECT COUNT(*) c FROM external_calls WHERE provider='vllm'"
            ).fetchone()
        with self._lock:
            self._nonce_counter = int(row["c"] or 0)
        self._ready = True
        return counts

    def _group_for(self, attempt_id: str):
        """Rebuild a reservation handle for an attempt whose in-memory handle is gone."""
        from ..experiment.budget import ReservationGroup

        conn = self._ledger.raw_connection
        with self._ledger.lock:
            rows = conn.execute(
                "SELECT resource, amount FROM budget_reservations WHERE attempt_id=?"
                " AND state='RESERVED'",
                (attempt_id,),
            ).fetchall()
        if not rows:
            return None
        return ReservationGroup(
            group_id=attempt_id, amounts={r["resource"]: r["amount"] for r in rows})

    # --- replay and the failure breaker ------------------------------------------------

    def _replayed(self, call_id: str) -> Optional[tuple[int, dict]]:
        """Answer a committed call from its frozen response instead of buying it again.

        Returns ``None`` when the call has not been committed. When it has, the stored
        bytes are returned verbatim: this is the only honest way to serve a replay, since
        re-dispatching would produce a second charge and, for acquisition, a second and
        different frozen world under the same identity.
        """
        ref = self._calls.committed_response_ref(call_id)
        if ref is None:
            return None
        try:
            payload = json.loads(self._store.get_bytes(ref).decode("utf-8"))
        except Exception as e:  # noqa: BLE001 - a committed call with no readable body
            raise ProviderError(
                409,
                f"call {call_id} is committed but its frozen response is unreadable "
                f"({type(e).__name__}); refusing to pay for it a second time",
            ) from e
        self._emit("REPLAYED_FROM_LEDGER", {"call_id": call_id})
        return 200, payload

    #: Incident kinds that hold the breaker open across a provider restart.
    _BREAKER_INCIDENTS = ("provider_unauthorized", "provider_consecutive_failures")

    def require_breaker_closed(self, provider: str) -> None:
        """Refuse before dispatch when this provider's breaker is already open.

        The breaker used to be recorded and never consulted: ``_note_upstream_failure`` runs
        *after* a response, so each rejection was dispatched, billed, and only then counted.
        Four consecutive 401s were four real charges. Worse, the counter lived in memory, so a
        provider restart reset it -- which is how a single dead credential burned 262 requests
        across one acquisition run.

        Both limbs are checked here, before ``open_call``/``reserve``/``mark_sent``:

        * the in-memory consecutive-failure count for this process, and
        * a FATAL incident already durable in the ledger, so the refusal survives a restart.

        A credential the upstream rejects fails identically every time; retrying is not
        recovery, it is paying to relearn the same fact.
        """
        with self._lock:
            count = self._consecutive_failures.get(provider, 0)
            limit = self._cfg.max_consecutive_failures
        if limit and count >= limit:
            raise ProviderError(
                503,
                f"{provider} failed {count} times in a row; the circuit breaker is open and "
                "no further call will be dispatched",
            )
        for kind in self._BREAKER_INCIDENTS:
            incident = self._ledger.latest_incident(severity="FATAL", kind=kind)
            if incident is None:
                continue
            if provider not in str(incident.get("detail") or ""):
                continue
            raise ProviderError(
                503,
                f"{provider} is fenced by a durable {kind} incident "
                f"({incident.get('detail')}); refusing to dispatch. Replace the credential and "
                "clear the incident before resuming.",
            )

    def _note_upstream_success(self, provider: str) -> None:
        with self._lock:
            self._consecutive_failures[provider] = 0

    def _frozen_response_text(self, payload: Any) -> str:
        """Serialize a response for the ledger, refusing to freeze a body we cannot replay.

        Returning a truncated prefix here is what made a committed call unreadable forever:
        the stored bytes were invalid JSON, ``_replayed`` raised 409, and the call could
        neither be recovered nor re-bought. Raising instead keeps the attempt failable and the
        call re-fetchable.
        """
        text = json.dumps(payload, sort_keys=True)
        limit = self._cfg.max_frozen_response_bytes
        if limit and len(text.encode("utf-8")) > limit:
            raise ResponseTooLarge(
                f"response is {len(text.encode('utf-8'))} bytes, above the "
                f"{limit}-byte freeze ceiling; refusing to commit a body that cannot be "
                "replayed verbatim"
            )
        return text

    def _note_upstream_failure(self, provider: str, status: int) -> None:
        """Trip the breaker after a run of failures, before the cap is burned.

        A credential the upstream rejects fails identically every time, so retrying is not
        recovery -- it is spending the request cap to learn the same fact repeatedly.
        """
        with self._lock:
            count = self._consecutive_failures.get(provider, 0) + 1
            self._consecutive_failures[provider] = count
            limit = self._cfg.max_consecutive_failures
        if status in (401, 403):
            self._ledger.record_incident(
                severity="FATAL", kind="provider_unauthorized",
                detail=f"{provider} rejected the credential with HTTP {status}",
            )
            raise ProviderError(
                503,
                f"{provider} rejected the credential (HTTP {status}); refusing further "
                "calls until it is replaced",
            )
        if limit and count >= limit:
            self._ledger.record_incident(
                severity="FATAL", kind="provider_consecutive_failures",
                detail=f"{provider} failed {count} times in a row (limit {limit})",
            )
            raise ProviderError(
                503,
                f"{provider} failed {count} times in a row; the circuit breaker is open",
            )

    @property
    def events(self) -> list[dict]:
        return list(self._events)

    def _emit(self, kind: str, payload: dict) -> None:
        with self._lock:
            self._events.append({"kind": kind, "at": self._clock(), **payload})

    # --- authorization --------------------------------------------------------------

    def authorize(self, route: str, *, token: Optional[str], peer_uid: Optional[int]) -> str:
        """Return the caller's role, or raise. Token first, then the peer uid if we have one."""
        role = self._tokens.role_of(token)
        if role is None:
            raise ProviderError(401, "unknown or missing role token")
        allowed = ROLE_ROUTES.get(route)
        if allowed is None:
            raise ProviderError(404, f"no such route {route!r}")
        if role not in allowed:
            raise ProviderError(
                403,
                f"role {role!r} may not call {route!r}; that route is not authorized "
                "for this role",
            )
        expected_uid = self._allowed_uids.get(role)
        if expected_uid is not None and peer_uid is not None and peer_uid != expected_uid:
            raise ProviderError(
                403, f"role {role!r} token presented by uid {peer_uid}, expected {expected_uid}"
            )
        return role

    # --- routes ---------------------------------------------------------------------

    def handle(self, route: str, body: dict, *, role: str) -> tuple[int, dict]:
        if route == "healthz":
            return 200, {"status": "ok"}
        if route == "readyz":
            return (200, {"status": "ready"}) if self._ready else (503, {"status": "starting"})
        if route == "credentials.probe":
            return self.credentials_probe()
        if route == "cells.register":
            return self.register_cell(body)
        if route == "cells.work":
            return self.cell_work(body)
        if route == "canary.audit":
            return self.canary_audit(body)
        if route == "exa.search":
            return self.exa_search(body)
        if route == "tavily.search":
            return self.tavily_search(body)
        if route == "deepseek.chat":
            return self.deepseek_chat(body, role=role)
        raise ProviderError(404, f"no such route {route!r}")

    def credentials_probe(self) -> tuple[int, dict]:
        """Ask each upstream whether it accepts our credential, without buying anything.

        This is the gate that did not exist when a rejected Tavily key turned into 262 billed
        401s.  "A credential file is readable" and "the vendor accepts it" are different
        facts, and only the second one predicts whether acquisition can succeed.

        The probe is free by construction: it sends a deliberately **invalid body** to each
        endpoint. Authentication is checked before body validation, so a good credential comes
        back 400 (the request was understood and rejected on its contents) while a bad one
        comes back 401/403. Nothing is searched, so nothing is charged and no frozen world is
        created. Deliberately does not touch the ledger for the same reason -- there is no
        logical call here to record.
        """
        results: dict[str, dict] = {}
        probes = [
            ("exa", self._exa_key, self._cfg.exa_endpoint,
             {"x-api-key": self._exa_key or "", "Content-Type": "application/json"}, {}),
            ("tavily", self._tavily_key, self._cfg.tavily_endpoint,
             {"Content-Type": "application/json"}, {"api_key": self._tavily_key or ""}),
            ("deepseek", self._deepseek_key, f"{self._cfg.deepseek_base_url}/chat/completions",
             {"Authorization": f"Bearer {self._deepseek_key or ''}",
              "Content-Type": "application/json"}, {}),
        ]
        for name, key, url, headers, body in probes:
            if key is None:
                results[name] = {"configured": False, "ok": None, "detail": "no credential loaded"}
                continue
            # A transient connection error is not a rejected credential, and the two must not
            # be reported the same way: one is retried, the other stops the campaign. Observed
            # on the run host, where a single ConnectError to one upstream failed this gate and
            # would have cost a full re-run of the whole 18-gate chain. Retried here, bounded,
            # because the probe costs nothing.
            outcome: Optional[tuple[int, Any]] = None
            last_error = ""
            for attempt in range(3):
                try:
                    status, payload, _elapsed = self._upstream(
                        url, headers, body, self._cfg.request_timeout_seconds)
                    outcome = (status, payload)
                    break
                except Exception as e:  # noqa: BLE001 - an unreachable upstream is retried
                    last_error = type(e).__name__
                    self._emit("CREDENTIAL_PROBE_RETRY", {"provider": name, "attempt": attempt})
            if outcome is None:
                results[name] = {
                    "configured": True, "ok": False,
                    "reachable": False,
                    "detail": (
                        f"upstream unreachable after 3 attempts ({last_error}); this is a "
                        "connectivity failure, not a rejected credential"
                    ),
                }
                continue
            status, payload = outcome
            rejected = status in (401, 403)
            results[name] = {
                "configured": True,
                "ok": not rejected,
                "reachable": True,
                "status": status,
                "detail": (
                    f"upstream rejected the credential with HTTP {status}"
                    if rejected else
                    f"credential accepted (probe returned {status} on a deliberately "
                    "invalid body, so authentication passed)"
                ),
            }
            if rejected:
                # Durable, so the breaker also fences the real path across a restart rather
                # than rediscovering this one rejection at a time, with a charge each.
                self._ledger.record_incident(
                    severity="FATAL", kind="provider_unauthorized",
                    detail=f"{name} rejected the credential with HTTP {status} during probe",
                )
            self._emit("CREDENTIAL_PROBE", {"provider": name, "ok": not rejected})

        configured = {n: r for n, r in results.items() if r["configured"]}
        all_ok = bool(configured) and all(r["ok"] for r in configured.values())
        return (200 if all_ok else 503), {"ok": all_ok, "providers": results}

    def register_cell(self, body: dict) -> tuple[int, dict]:
        validate_request("cells.register", body)
        token = str(body["cell_token"])
        if not _CELL_RE.match(token):
            raise ProviderError(400, "cell_token has an invalid shape")
        registration = CellRegistration(
            cell_token=token, run_id=str(body["run_id"]), task_id=str(body["task_id"]),
            arm_id=str(body["arm_id"]), variant_id=str(body["variant_id"]),
            replicate_id=str(body["replicate_id"]), work_key=str(body["work_key"]),
            layer=str(body.get("layer", "causal")),
        )
        with self._lock:
            existing = self._cells.get(token)
            if existing is not None and existing != registration:
                raise ProviderError(
                    409,
                    "cell token already bound to different coordinates; re-using one token for "
                    "two cells would mis-attribute every request made under it",
                )
            self._cells[token] = registration
        return 200, {"cell_token": token, "registered": True}

    def cell(self, token: str) -> Optional[CellRegistration]:
        with self._lock:
            return self._cells.get(token)

    def cell_work(self, body: dict) -> tuple[int, dict]:
        """Return durable, sanitized work telemetry for exactly one work item."""

        from .work_accounting import extract_request_events, summarize_work_extraction

        validate_request("cells.work", body)
        work_key = str(body["work_key"])
        extraction = extract_request_events(self._ledger, work_keys=[work_key])
        summary = summarize_work_extraction(
            extraction, require_isolated=bool(body["require_isolated"])
        )
        summary["work_key"] = work_key
        return 200, summary

    def canary_audit(self, body: dict) -> tuple[int, dict]:
        """Return a closed, sanitized attestation for exactly the requested canary work.

        The provider owns the 0700 ledger and object store.  The runner therefore asks the
        provider to reduce them to the few numbers the smoke gate needs; it never receives a
        request/response body, prompt hash, object reference, header, URL, credential, or truth
        artifact.  Work keys are an exact allowlist, so historical calls cannot make a current
        inert canary look healthy.
        """

        validate_request("canary.audit", body)
        work_keys = sorted(str(value) for value in body["work_keys"])
        wanted = set(work_keys)
        terminal = {"COMMITTED", "FAILED_FINAL", "FAILED_UNKNOWN"}
        selector_ops = {
            OpClass.PAGE_P1_SELECTOR_LOCAL.value,
            OpClass.PAGE_P1_SELECTOR_GLOBAL.value,
            OpClass.COMPRESSOR_P1_SELECTOR.value,
        }

        with self._ledger.lock:
            # Read a consistent provider-owned snapshot. Filtering values in Python avoids
            # constructing SQL from caller input; only rows whose exact work key was requested
            # can cross the boundary.
            rows = self._ledger.raw_connection.execute(
                "SELECT c.work_key, c.op_class, a.state, a.request_object_ref, a.usage_json"
                " FROM external_calls c JOIN external_call_attempts a USING(call_id)"
                " WHERE c.work_key IS NOT NULL ORDER BY c.work_key, c.op_class,"
                " a.attempt_ordinal"
            ).fetchall()
            settlement_rows = self._ledger.raw_connection.execute(
                "SELECT c.work_key, c.op_class, a.attempt_id,"
                " b.state AS reservation_state, b.settled_amount"
                " FROM external_calls c"
                " JOIN external_call_attempts a USING(call_id)"
                " JOIN budget_reservations b ON b.attempt_id=a.attempt_id"
                " WHERE c.work_key IS NOT NULL AND b.resource='gpu_seconds'"
                " ORDER BY c.work_key, c.op_class, a.attempt_ordinal"
            ).fetchall()
            budget = self._ledger.raw_connection.execute(
                "SELECT cap, reserved_total, settled_total FROM budget_accounts"
                " WHERE resource='gpu_seconds'"
            ).fetchone()

        if budget is None:
            raise ProviderError(
                409, "gpu_seconds budget account is missing; canary audit cannot attest headroom"
            )

        by_work: dict[str, dict[str, dict[str, Any]]] = {
            key: {} for key in work_keys
        }
        open_by_work = {key: 0 for key in work_keys}
        for row in rows:
            work_key = str(row["work_key"] or "")
            if work_key not in wanted:
                continue
            state = str(row["state"] or "")
            if state not in terminal:
                open_by_work[work_key] += 1
            op_class = str(row["op_class"] or "")
            aggregate = by_work[work_key].setdefault(op_class, {
                "op_class": op_class,
                "attempt_count": 0,
                "committed_attempt_count": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_prompt_tokens": 0,
                "max_completion_tokens_observed": 0,
                "settled_gpu_seconds": 0.0,
                "selector_request_max_tokens": [],
                "_cached_complete": True,
            })
            aggregate["attempt_count"] += 1

            if state == "COMMITTED":
                try:
                    usage = json.loads(row["usage_json"] or "{}")
                except (json.JSONDecodeError, TypeError) as exc:
                    raise ProviderError(
                        409,
                        f"committed usage is unreadable for work {work_key!r}, "
                        f"op {op_class!r}",
                    ) from exc
                if not isinstance(usage, dict):
                    raise ProviderError(
                        409,
                        f"committed usage is not an object for work {work_key!r}, "
                        f"op {op_class!r}",
                    )
                prompt = _nonnegative_int_or_error(
                    usage.get("prompt_tokens", 0), label="prompt_tokens",
                    work_key=work_key, op_class=op_class,
                )
                completion = _nonnegative_int_or_error(
                    usage.get("completion_tokens", 0), label="completion_tokens",
                    work_key=work_key, op_class=op_class,
                )
                cached = _cached_tokens(usage)
                aggregate["committed_attempt_count"] += 1
                aggregate["prompt_tokens"] += prompt
                aggregate["completion_tokens"] += completion
                aggregate["max_completion_tokens_observed"] = max(
                    aggregate["max_completion_tokens_observed"], completion)
                if cached is None:
                    aggregate["_cached_complete"] = False
                else:
                    aggregate["cached_prompt_tokens"] += int(cached)

            if op_class in selector_ops and row["request_object_ref"]:
                try:
                    request = json.loads(
                        self._store.get_bytes(row["request_object_ref"]).decode("utf-8"))
                except Exception as exc:  # noqa: BLE001 - unreadable evidence fails closed
                    raise ProviderError(
                        409,
                        f"selector request metadata is unreadable for work {work_key!r}, "
                        f"op {op_class!r}",
                    ) from exc
                if not isinstance(request, dict):
                    raise ProviderError(
                        409,
                        f"selector request metadata is not an object for work {work_key!r}, "
                        f"op {op_class!r}",
                    )
                limit = request.get("max_tokens")
                aggregate["selector_request_max_tokens"].append(
                    limit if isinstance(limit, int) and not isinstance(limit, bool) else None
                )

        # Budget settlement is read from budget_reservations, not reconstructed from response
        # latency.  It is the same authority that increments budget_accounts.settled_total.
        # Keeping this pass separate from attempt telemetry prevents a corrupt duplicate
        # reservation from duplicating token/attempt counts through a SQL join.
        seen_gpu_reservations: set[str] = set()
        for row in settlement_rows:
            work_key = str(row["work_key"] or "")
            if work_key not in wanted:
                continue
            attempt_id = str(row["attempt_id"] or "")
            if not attempt_id or attempt_id in seen_gpu_reservations:
                raise ProviderError(
                    409,
                    f"gpu settlement identity is invalid for work {work_key!r}",
                )
            seen_gpu_reservations.add(attempt_id)
            state = str(row["reservation_state"] or "")
            if state == "SETTLED":
                settled = _nonnegative_finite_float_or_error(
                    row["settled_amount"],
                    label="settled_gpu_seconds",
                    work_key=work_key,
                    op_class=str(row["op_class"] or ""),
                )
                aggregate = by_work[work_key].get(str(row["op_class"] or ""))
                if aggregate is None:
                    raise ProviderError(
                        409,
                        f"gpu settlement has no matching attempt for work {work_key!r}",
                    )
                aggregate["settled_gpu_seconds"] += settled
            elif state not in {"RESERVED", "RELEASED"}:
                raise ProviderError(
                    409,
                    f"gpu settlement state is invalid for work {work_key!r}",
                )

        work_rows: list[dict] = []
        for work_key in work_keys:
            rendered_ops = []
            for op_class in sorted(by_work[work_key]):
                aggregate = by_work[work_key][op_class]
                cached = aggregate.pop("_cached_complete")
                if not cached:
                    aggregate["cached_prompt_tokens"] = None
                rendered_ops.append(aggregate)
            work_rows.append({
                "work_key": work_key,
                "open_attempts": open_by_work[work_key],
                "settled_gpu_seconds": sum(
                    float(op["settled_gpu_seconds"]) for op in rendered_ops
                ),
                "ops": rendered_ops,
            })

        cap = float(budget["cap"])
        reserved = float(budget["reserved_total"])
        settled = float(budget["settled_total"])
        response = {
            "version": "canary_audit_v1",
            "work_keys": work_keys,
            "work": work_rows,
            "open_attempts": sum(open_by_work.values()),
            "gpu_budget": {
                "resource": "gpu_seconds",
                "cap": cap,
                "reserved": reserved,
                "settled": settled,
                "remaining": cap - reserved - settled,
            },
        }
        # Response validation is part of the boundary: adding a field later cannot silently
        # expose it to the runner merely because the producer started returning it.
        validate_request("canary.audit.response", response)
        return 200, response

    # --- Tavily ----------------------------------------------------------------------

    def exa_search(self, body: dict) -> tuple[int, dict]:
        """One acquisition query. The only outbound search path in the whole study.

        Exa returns ``costDollars`` per response, so the ledger settles in the currency the
        vendor billed rather than converting a credit count by assumption.
        """
        if self._exa_key is None:
            raise ProviderError(503, "provider has no Exa credential loaded")
        # Before open_call/reserve/mark_sent, so an open breaker costs nothing at all.
        self.require_breaker_closed("exa")
        validate_request("exa.search", body)
        if str(body.get("type", "")) == "auto":
            raise ProviderError(
                400,
                "type 'auto' is forbidden: the same query must address the same slice of "
                "the web on every acquisition",
            )
        task_id = str(body.pop("_task_id", "") or "")
        call_key = str(body.pop("_call_key", "") or sha256_hex(
            json.dumps(body, sort_keys=True).encode("utf-8")))

        call_id = self._calls.open_call(
            provider="exa", op_class="search", call_key=call_key, work_key=task_id or None,
        )
        replay = self._replayed(call_id)
        if replay is not None:
            return replay
        try:
            attempt = self._calls.begin_attempt(call_id)
        except CallNotReplayable as e:
            raise ProviderError(409, str(e)) from e

        amounts = {
            "exa_requests": 1.0,
            "exa_usd": self._cfg.exa_usd_worst_case,
            "remote_calls": 1.0,
        }
        try:
            group = self._calls.reserve(attempt, amounts, work_key=task_id or None)
        except BudgetExceeded as e:
            raise ProviderError(429, f"budget refused: {e.resource} exhausted") from e

        headers = {"x-api-key": self._exa_key, "Content-Type": "application/json"}
        self._calls.mark_sent(attempt, request_text=json.dumps(
            {**body, "x-api-key": PROVIDER_KEY_PLACEHOLDER}, sort_keys=True))
        try:
            status, payload, elapsed = self._upstream(
                self._cfg.exa_endpoint, headers, body, self._cfg.request_timeout_seconds,
            )
        except Exception as e:  # noqa: BLE001 - sent, then silence
            self._calls.fail_unknown(attempt, group, error_class=type(e).__name__)
            self._emit("EXA_FAILED_UNKNOWN", {"call_id": call_id, "error": type(e).__name__})
            raise ProviderError(
                504, f"exa upstream failed after send: {type(e).__name__}") from e

        if status != 200:
            self._calls.fail_after_response(
                attempt, group, {"exa_requests": 1.0, "remote_calls": 1.0},
                error_class=f"http_{status}")
            self._emit("EXA_HTTP_ERROR", {"call_id": call_id, "status": status})
            self._note_upstream_failure("exa", status)
            return status, {"error": f"exa returned {status}",
                            "retry": classify_status(status).value}

        self._note_upstream_success("exa")
        cost = _float_or(payload.get("costDollars", {}) or {}, "total",
                         default=self._cfg.exa_usd_worst_case)
        try:
            frozen = self._frozen_response_text(payload)
        except ResponseTooLarge as e:
            # Settle the money before answering: the request really was sent, so the request
            # cap is spent, but the call must not become a COMMITTED row with an unreadable
            # body. Failing it here leaves the identity re-fetchable.
            self._calls.fail_after_response(
                attempt, group, {"exa_requests": 1.0, "remote_calls": 1.0},
                error_class="response_too_large")
            self._ledger.record_incident(
                severity="ERROR", kind="provider_response_too_large",
                detail=f"exa call {call_id}: {e}",
            )
            self._emit("EXA_RESPONSE_TOO_LARGE", {"call_id": call_id})
            raise ProviderError(502, f"exa response cannot be frozen: {e}") from e
        self._calls.store_response(
            attempt, response_text=frozen,
            provider_request_id=str(payload.get("requestId", "")),
        )
        self._calls.validate(attempt)
        self._calls.commit(attempt, group, {
            "exa_requests": 1.0,
            "exa_usd": cost,
            "remote_calls": 1.0,
        })
        self._emit("EXA_COMMITTED", {"call_id": call_id, "usd": cost, "elapsed_s": elapsed})
        return 200, payload

    def tavily_search(self, body: dict) -> tuple[int, dict]:
        """One acquisition query. The only outbound Tavily path in the whole study."""
        if self._tavily_key is None:
            raise ProviderError(503, "provider has no Tavily credential loaded")
        self.require_breaker_closed("tavily")
        validate_request("tavily.search", body)
        self._require_placeholder(body, "api_key")
        if body.get("auto_parameters"):
            raise ProviderError(
                400,
                "auto_parameters is forbidden: the same query must address the same slice of "
                "the web on every acquisition",
            )
        task_id = str(body.pop("_task_id", "") or "")
        call_key = str(body.pop("_call_key", "") or sha256_hex(
            json.dumps(body, sort_keys=True).encode("utf-8")))

        call_id = self._calls.open_call(
            provider="tavily", op_class="search", call_key=call_key, work_key=task_id or None,
        )
        replay = self._replayed(call_id)
        if replay is not None:
            return replay
        try:
            attempt = self._calls.begin_attempt(call_id)
        except CallNotReplayable as e:
            raise ProviderError(409, str(e)) from e

        amounts = {
            "tavily_requests": 1.0,
            "tavily_credits": self._cfg.tavily_credits_worst_case,
            "remote_calls": 1.0,
        }
        try:
            group = self._calls.reserve(attempt, amounts, work_key=task_id or None)
        except BudgetExceeded as e:
            raise ProviderError(429, f"budget refused: {e.resource} exhausted") from e

        outbound = dict(body)
        outbound["api_key"] = self._tavily_key
        self._calls.mark_sent(attempt, request_text=json.dumps(
            {**body, "api_key": PROVIDER_KEY_PLACEHOLDER}, sort_keys=True))
        try:
            status, payload, elapsed = self._upstream(
                self._cfg.tavily_endpoint, {}, outbound, self._cfg.request_timeout_seconds,
            )
        except Exception as e:  # noqa: BLE001 - sent, then silence
            self._calls.fail_unknown(attempt, group, error_class=type(e).__name__)
            self._emit("TAVILY_FAILED_UNKNOWN", {"call_id": call_id, "error": type(e).__name__})
            raise ProviderError(
                504, f"tavily upstream failed after send: {type(e).__name__}") from e

        if status != 200:
            self._calls.fail_after_response(
                attempt, group, {"tavily_requests": 1.0, "remote_calls": 1.0},
                error_class=f"http_{status}")
            self._emit("TAVILY_HTTP_ERROR", {"call_id": call_id, "status": status})
            self._note_upstream_failure("tavily", status)
            return status, {"error": f"tavily returned {status}",
                            "retry": classify_status(status).value}

        self._note_upstream_success("tavily")
        credits = _float_or(payload.get("usage", {}), "credits",
                            default=self._cfg.tavily_credits_worst_case)
        try:
            frozen = self._frozen_response_text(payload)
        except ResponseTooLarge as e:
            self._calls.fail_after_response(
                attempt, group, {"tavily_requests": 1.0, "remote_calls": 1.0},
                error_class="response_too_large")
            self._ledger.record_incident(
                severity="ERROR", kind="provider_response_too_large",
                detail=f"tavily call {call_id}: {e}",
            )
            self._emit("TAVILY_RESPONSE_TOO_LARGE", {"call_id": call_id})
            raise ProviderError(502, f"tavily response cannot be frozen: {e}") from e
        self._calls.store_response(
            attempt, response_text=frozen,
            provider_request_id=str(payload.get("request_id", "")),
        )
        self._calls.validate(attempt)
        self._calls.commit(attempt, group, {
            "tavily_requests": 1.0,
            "tavily_credits": credits,
            "remote_calls": 1.0,
        })
        self._emit("TAVILY_COMMITTED", {"call_id": call_id, "credits": credits,
                                        "elapsed_s": elapsed})
        return 200, payload

    # --- DeepSeek ---------------------------------------------------------------------

    def deepseek_chat(self, body: dict, *, role: str) -> tuple[int, dict]:
        """One DeepSeek chat completion, for task authoring or blind judging only.

        DeepSeek never appears on the treatment path; the route policy already bars the runner,
        and the op class recorded here keeps its cost out of treatment work in the ledger.
        """
        if self._deepseek_key is None:
            raise ProviderError(503, "provider has no DeepSeek credential loaded")
        self.require_breaker_closed("deepseek")
        validate_request("deepseek.chat", body)
        self._require_placeholder(body, "api_key")
        op_class = str(body.pop("_op_class", "JUDGE_REPORT"))
        if op_class not in {o.value for o in OpClass}:
            raise ProviderError(400, f"unknown op_class {op_class!r}")
        if op_class not in {o.value for o in REMOTE_ALLOWED_OPS}:
            # DeepSeek must never appear on the treatment path. Refusing a treatment op class
            # here means a mis-wired caller fails loudly instead of quietly putting a second
            # model inside the system under measurement.
            raise ProviderError(
                403,
                f"DeepSeek may not serve op class {op_class!r}: it authors and judges, and is "
                "never part of the treatment path",
            )
        work_key = str(body.pop("_work_key", "") or "")
        call_key = str(body.pop("_call_key", "") or sha256_hex(
            json.dumps(body, sort_keys=True).encode("utf-8")))

        call_id = self._calls.open_call(
            provider="deepseek", op_class=op_class, call_key=call_key,
            work_key=work_key or None,
        )
        # A judge retry re-sends a byte-identical body, so it lands on this same logical
        # call. Serving the frozen response is what stops the retry loop from buying the
        # same completion again -- and again -- while only the last one leaves a trace.
        replay = self._replayed(call_id)
        if replay is not None:
            return replay
        try:
            attempt = self._calls.begin_attempt(call_id)
        except CallNotReplayable as e:
            raise ProviderError(409, str(e)) from e

        amounts = {
            "deepseek_requests": 1.0,
            "deepseek_input_tokens": self._cfg.deepseek_input_tokens_worst_case,
            "deepseek_output_tokens": self._cfg.deepseek_output_tokens_worst_case,
            "deepseek_usd": self._cfg.deepseek_usd_worst_case,
            "remote_calls": 1.0,
        }
        try:
            group = self._calls.reserve(attempt, amounts, work_key=work_key or None)
        except BudgetExceeded as e:
            raise ProviderError(429, f"budget refused: {e.resource} exhausted") from e

        outbound = {k: v for k, v in body.items() if k != "api_key"}
        headers = {"Authorization": f"Bearer {self._deepseek_key}",
                   "Content-Type": "application/json"}
        self._calls.mark_sent(attempt, request_text=json.dumps(
            {**outbound, "authorization": PROVIDER_KEY_PLACEHOLDER}, sort_keys=True))
        url = self._cfg.deepseek_base_url.rstrip("/") + "/chat/completions"
        try:
            status, payload, elapsed = self._upstream(
                url, headers, outbound, self._cfg.request_timeout_seconds)
        except Exception as e:  # noqa: BLE001
            self._calls.fail_unknown(attempt, group, error_class=type(e).__name__)
            self._emit("DEEPSEEK_FAILED_UNKNOWN", {"call_id": call_id, "error": type(e).__name__})
            raise ProviderError(
                504, f"deepseek upstream failed after send: {type(e).__name__}") from e

        usage = payload.get("usage", {}) or {}
        actuals = self._deepseek_actuals(usage)
        if status != 200:
            self._calls.fail_after_response(
                attempt, group, actuals, error_class=f"http_{status}")
            self._emit("DEEPSEEK_HTTP_ERROR", {"call_id": call_id, "status": status})
            self._note_upstream_failure("deepseek", status)
            return status, {"error": f"deepseek returned {status}",
                            "retry": classify_status(status).value}

        self._note_upstream_success("deepseek")
        try:
            frozen = self._frozen_response_text(payload)
        except ResponseTooLarge as e:
            self._calls.fail_after_response(
                attempt, group, actuals, error_class="response_too_large")
            self._ledger.record_incident(
                severity="ERROR", kind="provider_response_too_large",
                detail=f"deepseek call {call_id}: {e}",
            )
            self._emit("DEEPSEEK_RESPONSE_TOO_LARGE", {"call_id": call_id})
            raise ProviderError(502, f"deepseek response cannot be frozen: {e}") from e
        self._calls.store_response(
            attempt, response_text=frozen,
            provider_request_id=str(payload.get("id", "")),
            requested_model=str(outbound.get("model", "")),
            returned_model=str(payload.get("model", "")),
            system_fingerprint=str(payload.get("system_fingerprint", "")),
            usage_json=json.dumps(usage, sort_keys=True),
        )
        self._calls.validate(attempt)
        self._calls.commit(attempt, group, actuals)
        self._emit("DEEPSEEK_COMMITTED", {
            "call_id": call_id, "requested_model": str(outbound.get("model", "")),
            "returned_model": str(payload.get("model", "")),
            "system_fingerprint": str(payload.get("system_fingerprint", "")),
            "usd": actuals["deepseek_usd"], "elapsed_s": elapsed,
        })
        return 200, payload

    def _deepseek_actuals(self, usage: Mapping[str, Any]) -> dict[str, float]:
        prompt = float(usage.get("prompt_tokens", 0) or 0)
        completion = float(usage.get("completion_tokens", 0) or 0)
        usd = (prompt / 1_000_000.0) * self._cfg.deepseek_usd_per_1m_input + \
              (completion / 1_000_000.0) * self._cfg.deepseek_usd_per_1m_output
        return {
            "deepseek_requests": 1.0,
            "deepseek_input_tokens": prompt,
            "deepseek_output_tokens": completion,
            "deepseek_usd": usd,
            "remote_calls": 1.0,
        }

    # --- local inference ---------------------------------------------------------------

    def chat_completions(self, body: dict, *, cell_token: Optional[str]) -> tuple[int, dict]:
        """Proxy one completion to the local vLLM, tagged, timed and accounted.

        The model alias is rewritten to the one served model here, so P0 and P1 issue an
        identical upstream request while remaining separable in the ledger. An unmapped alias is
        refused rather than passed through: an untagged treatment request would land in the work
        total with no op class, and a work number nobody can attribute is not a measurement.
        """
        proxy_ingress_ts = self._clock()
        requested_model = str(body.get("model", ""))
        alias = requested_model.split(":")[-1]
        op = self._cfg.model_aliases.get(alias)
        if op is None:
            raise ProviderError(
                400,
                f"model {requested_model!r} is not a registered alias; every treatment request "
                "must carry an op class or its work cannot be attributed",
            )
        cell = self.cell(cell_token) if cell_token else None
        if cell_token and cell is None:
            raise ProviderError(404, f"cell {cell_token!r} is not registered")

        call_key = derive_id("inference_call", {
            "cell": cell_token or "-",
            "op": op.value,
            "body": sha256_hex(json.dumps(body, sort_keys=True).encode("utf-8")),
            "nonce": self._next_nonce(),
        })
        call_id = self._calls.open_call(
            provider="vllm", op_class=op.value, call_key=call_key,
            work_key=(cell.work_key if cell else None),
        )
        attempt = self._calls.begin_attempt(call_id)
        try:
            group = self._calls.reserve(
                attempt, {"gpu_seconds": self._cfg.gpu_seconds_worst_case},
                work_key=(cell.work_key if cell else None),
            )
        except BudgetExceeded as e:
            raise ProviderError(429, f"budget refused: {e.resource} exhausted") from e

        outbound = dict(body)
        if self._cfg.served_model:
            outbound["model"] = self._cfg.served_model
        if self._cfg.disable_thinking:
            # Qwen3 emits <think>...</think> reasoning by default, which the frozen stack
            # declares off (configs/stack.yaml model.enable_thinking: false). vLLM only honours
            # that per request via chat_template_kwargs, so it is injected HERE, uniformly for
            # every arm -- P0 and P1 alike. Without it, react calls burn their budget thinking
            # and get truncated before they act, and selector calls never reach their JSON.
            kwargs = dict(outbound.get("chat_template_kwargs") or {})
            kwargs.setdefault("enable_thinking", False)
            outbound["chat_template_kwargs"] = kwargs
        prompt_sha = sha256_hex(json.dumps(outbound.get("messages", []), sort_keys=True)
                                .encode("utf-8"))
        url = self._cfg.vllm_base_url.rstrip("/") + "/chat/completions"
        gate = self._inference_gate if self._inference_gate is not None else contextlib.nullcontext()
        with gate:
            # Queue wait ends here. SENT is recorded immediately before the only upstream call, so
            # the durable dispatch timestamp denotes service rather than time waiting at ingress.
            dispatch_ts = self._clock()
            self._calls.mark_sent(
                attempt,
                request_text=json.dumps(
                    {
                        "model": outbound.get("model"),
                        "prompt_sha256": prompt_sha,
                        "max_tokens": outbound.get("max_tokens"),
                        "temperature": outbound.get("temperature"),
                        "seed": outbound.get("seed"),
                    },
                    sort_keys=True,
                ),
                dispatched_at=dispatch_ts,
                proxy_ingress_at=proxy_ingress_ts,
            )
            try:
                status, payload, elapsed = self._upstream(
                    url, {"Content-Type": "application/json"}, outbound,
                    self._cfg.inference_timeout_seconds)
            except Exception as e:  # noqa: BLE001
                self._calls.fail_unknown(attempt, group, error_class=type(e).__name__)
                self._emit("INFERENCE_FAILED_UNKNOWN", {
                    "call_id": call_id, "op_class": op.value, "error": type(e).__name__,
                    "cell": cell_token})
                raise ProviderError(504, f"vLLM failed after send: {type(e).__name__}") from e
            response_end_ts = self._clock()

        usage = payload.get("usage", {}) or {}
        actual_gpu = min(max(0.0, response_end_ts - dispatch_ts), self._cfg.gpu_seconds_worst_case)
        telemetry = {
            "call_id": call_id,
            "op_class": op.value,
            "cell": cell_token or "",
            "run_id": cell.run_id if cell else "",
            "task_id": cell.task_id if cell else "",
            "arm_id": cell.arm_id if cell else "",
            "variant_id": cell.variant_id if cell else "",
            "replicate_id": cell.replicate_id if cell else "",
            "work_key": cell.work_key if cell else "",
            "layer": cell.layer if cell else "",
            "requested_model": requested_model,
            "returned_model": str(payload.get("model", "")),
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "cached_prompt_tokens": _cached_tokens(usage),
            "prompt_sha256": prompt_sha,
            "upstream_dispatch_ts": dispatch_ts,
            "upstream_response_end_ts": response_end_ts,
            "proxy_ingress_ts": proxy_ingress_ts,
            "queue_wait_seconds": max(0.0, dispatch_ts - proxy_ingress_ts),
            "elapsed_s": elapsed,
            "finish_reason": _finish_reason(payload),
        }
        if status != 200:
            self._calls.fail_after_response(
                attempt,
                group,
                {"gpu_seconds": actual_gpu},
                error_class=f"http_{status}",
                response_end_at=response_end_ts,
                telemetry_json=json.dumps(telemetry, sort_keys=True),
            )
            self._emit("INFERENCE_HTTP_ERROR", {"call_id": call_id, "status": status})
            return status, {"error": f"vllm returned {status}"}

        self._calls.store_response(
            attempt, response_text=json.dumps({"usage": usage}, sort_keys=True),
            requested_model=requested_model,
            returned_model=str(payload.get("model", "")),
            usage_json=json.dumps(usage, sort_keys=True),
            telemetry_json=json.dumps(telemetry, sort_keys=True),
            response_end_at=response_end_ts,
        )
        self._calls.validate(attempt)
        self._calls.commit(attempt, group, {"gpu_seconds": actual_gpu})
        self._emit("INFERENCE_COMMITTED", telemetry)
        return 200, payload

    def _next_nonce(self) -> int:
        """A per-instance counter, so two identical prompts are two calls.

        Inference is not idempotent the way an acquisition query is: the same prompt may
        legitimately be issued twice within a cell, and collapsing them onto one call id would
        under-count the work actually spent. Recovery re-runs the whole cell, not a request.
        """
        with self._lock:
            self._nonce_counter += 1
            return self._nonce_counter

    # --- helpers ------------------------------------------------------------------------

    @staticmethod
    def _require_placeholder(body: dict, field_name: str) -> None:
        """A client may only ever send the placeholder where a credential goes.

        Rejecting anything else is what stops a client from routing its own credential out
        through this service, and guarantees no real key was ever in the client's address space.
        """
        value = body.get(field_name)
        if value != PROVIDER_KEY_PLACEHOLDER:
            raise ProviderError(
                400,
                f"{field_name} must be exactly the provider placeholder; clients never hold a "
                "credential and must not send one",
            )


def _float_or(mapping: Mapping[str, Any], key: str, *, default: float) -> float:
    try:
        value = mapping.get(key)
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _nonnegative_int_or_error(
    value: Any,
    *,
    label: str,
    work_key: str,
    op_class: str,
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise ProviderError(
            409,
            f"{label} is invalid for work {work_key!r}, op {op_class!r}",
        )
    return value


def _nonnegative_finite_float_or_error(
    value: Any,
    *,
    label: str,
    work_key: str,
    op_class: str,
) -> float:
    if isinstance(value, bool):
        valid = False
        rendered = 0.0
    else:
        try:
            rendered = float(value)
        except (TypeError, ValueError):
            valid = False
        else:
            valid = math.isfinite(rendered) and rendered >= 0
    if not valid:
        raise ProviderError(
            409,
            f"{label} is invalid for work {work_key!r}, op {op_class!r}",
        )
    return rendered


def _cached_tokens(usage: Mapping[str, Any]) -> Optional[int]:
    details = usage.get("prompt_tokens_details")
    if isinstance(details, Mapping) and details.get("cached_tokens") is not None:
        try:
            return int(details["cached_tokens"])
        except (TypeError, ValueError):
            return None
    return None


def _finish_reason(payload: Mapping[str, Any]) -> str:
    try:
        return str(payload["choices"][0].get("finish_reason", ""))
    except (KeyError, IndexError, TypeError):
        return ""


# --- HTTP ------------------------------------------------------------------------------


_ROUTE_BY_PATH = {
    "/v1/exa/search": "exa.search",
    "/v1/tavily/search": "tavily.search",
    "/v1/deepseek/chat": "deepseek.chat",
    "/v1/chat/completions": "chat.completions",
    "/v1/cells": "cells.register",
    "/v1/cells/work": "cells.work",
    "/v1/canary/audit": "canary.audit",
    "/healthz": "healthz",
    "/readyz": "readyz",
    "/v1/credentials/probe": "credentials.probe",
}

_CELL_PATH = re.compile(r"^/v1/cell/([A-Za-z0-9_\-]{8,128})/chat/completions$")

MAX_BODY_BYTES = 8 * 1024 * 1024


def resolve_route(path: str) -> tuple[str, Optional[str]]:
    """Map a request path to (route, cell_token). Unknown paths are 404, never guessed."""
    clean = path.split("?", 1)[0].rstrip("/") or "/"
    if clean in _ROUTE_BY_PATH:
        return _ROUTE_BY_PATH[clean], None
    match = _CELL_PATH.match(clean)
    if match:
        return "chat.completions", match.group(1)
    raise ProviderError(404, "no such route")


def make_handler(service: ProviderService, redactor: SecretRedactor):
    """Build the request handler class bound to one service instance."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "shapeflow-provider"
        sys_version = ""

        def log_message(self, fmt: str, *args) -> None:  # noqa: A003
            # Field allowlist: method and path only, redacted. Never a header or body dump.
            line = redactor.redact(fmt % args)
            print(f"provider {self.address_string()} {line}", flush=True)

        def _send(self, status: int, payload: dict) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _peer_uid(self) -> Optional[int]:
            sock = getattr(self, "connection", None)
            if sock is None:
                return None
            try:
                if sock.family == socket.AF_UNIX:
                    # struct ucred is {pid, uid, gid} as three native int32s.
                    creds = sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED,
                                            struct.calcsize("3i"))
                    _pid, uid, _gid = struct.unpack("3i", creds)
                    return uid
                return peer_uid_of_tcp(sock.getsockname(), sock.getpeername())
            except (OSError, AttributeError, ValueError, struct.error):
                return None

        def do_GET(self) -> None:  # noqa: N802
            self._dispatch(b"")

        def do_POST(self) -> None:  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                self._send(400, {"error": "bad Content-Length"})
                return
            if length > MAX_BODY_BYTES:
                self._send(413, {"error": "body too large"})
                return
            self._dispatch(self.rfile.read(length) if length else b"")

        def _dispatch(self, raw: bytes) -> None:
            try:
                route, cell_token = resolve_route(self.path)
                auth = self.headers.get("Authorization") or ""
                token = auth[7:].strip() if auth.lower().startswith("bearer ") else None
                role = service.authorize(route, token=token, peer_uid=self._peer_uid())
                body = json.loads(raw.decode("utf-8")) if raw else {}
                if not isinstance(body, dict):
                    raise ProviderError(400, "request body must be a JSON object")
                if route == "chat.completions":
                    validate_request("chat.completions", body)
                    status, payload = service.chat_completions(body, cell_token=cell_token)
                else:
                    status, payload = service.handle(route, body, role=role)
            except ProviderError as e:
                self._send(e.status, {"error": redactor.redact(str(e))})
                return
            except json.JSONDecodeError:
                self._send(400, {"error": "body is not valid JSON"})
                return
            except Exception as e:  # noqa: BLE001 - never leak an unredacted traceback
                self._send(500, {"error": redactor.redact(f"{type(e).__name__}: {e}")})
                return
            self._send(status, payload)

    return Handler


class _ThreadingUnixServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


def serve_forever(
    service: ProviderService, config: ProviderConfig, redactor: SecretRedactor
) -> tuple[ThreadingHTTPServer, Optional[_ThreadingUnixServer]]:
    """Start the loopback listener (and the Unix socket when configured). Never binds 0.0.0.0."""
    if config.bind_host not in ("127.0.0.1", "::1", "localhost"):
        raise ProviderError(
            500,
            f"refusing to bind {config.bind_host!r}: the provider holds every credential and is "
            "loopback-only by design",
        )
    handler = make_handler(service, redactor)
    tcp = ThreadingHTTPServer((config.bind_host, config.bind_port), handler)
    tcp.daemon_threads = True
    threading.Thread(target=tcp.serve_forever, name="provider-tcp", daemon=True).start()

    uds: Optional[_ThreadingUnixServer] = None
    if config.unix_socket:
        path = Path(config.unix_socket)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        uds = _ThreadingUnixServer(str(path), handler)
        os.chmod(path, 0o660)
        threading.Thread(target=uds.serve_forever, name="provider-uds", daemon=True).start()
    return tcp, uds
