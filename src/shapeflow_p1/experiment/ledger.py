"""The run ledger: durable, idempotent, crash-resumable bookkeeping.

This is the spine of the campaign. Every unit of work -- a checkpoint fork, a selector
call, a scored report -- is a **work item** with a logical identity derived from the
protocol and its coordinates, never from wall-clock time or a run counter. That identity
is what makes the campaign resumable: after a crash, the coordinator recomputes the same
work keys and asks the ledger which are already done. "Done" here is strict -- a work
item counts as finished only when it is ``COMMITTED`` in the database *and* its result
blob still verifies in the object store. Anything less is redone.

Design rules taken from the plan (§16), each defending against a specific way a research
result gets quietly corrupted:

- **Single writer.** All mutations go through one connection under one lock, using
  ``BEGIN IMMEDIATE``. WAL still lets external read-only connections (a status probe)
  read concurrently, but there is exactly one writer, so two workers can never both
  believe they claimed the same item.

- **One terminal accepted attempt.** Retries create *new* attempts; they never overwrite
  an old record. A partial unique index enforces that at most one attempt per work item
  is ``COMMITTED``. So a flaky item that eventually succeeds keeps its full failure
  history, and a double-commit is a database error rather than a silent overwrite.

- **Terminal record last.** A work item is marked ``COMMITTED`` only after its output is
  in the object store. A process killed between "wrote blob" and "wrote COMMITTED" leaves
  the item non-terminal, so resume re-runs it -- wasteful but correct -- instead of
  trusting a commit whose bytes might be absent.

- **Timeout after a side effect is not "nothing happened".** An attempt that may have
  already hit an external API and then lost contact ends ``FAILED_UNKNOWN`` and, for a
  side-effecting item, freezes the item there for investigation rather than cheerfully
  retrying and double-charging.

The state machine::

    PENDING --claim--> CLAIMED --> MATERIALIZED --> VALIDATED --commit--> COMMITTED*
       |                  \\------------\\-------------\\--> FAILED_RETRYABLE --> (PENDING | FAILED_FINAL*)
       |                   \\------------\\-------------\\--> FAILED_FINAL*
       |                    \\-----------\\-------------\\--> FAILED_UNKNOWN*
       \\--reservation refused--> BLOCKED_BUDGET*

(* terminal). ``FAILED_RETRYABLE`` reopens the item to ``PENDING`` while retries remain,
else it becomes ``FAILED_FINAL``.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Sequence

from ..hashing import derive_id

__all__ = [
    "Ledger",
    "WorkItem",
    "Attempt",
    "LedgerError",
    "IllegalTransition",
    "NON_TERMINAL_STATES",
    "TERMINAL_STATES",
]

# --- states ---------------------------------------------------------------------------

NON_TERMINAL_STATES = frozenset({"PENDING", "CLAIMED", "MATERIALIZED", "VALIDATED"})
TERMINAL_STATES = frozenset(
    {"COMMITTED", "FAILED_FINAL", "FAILED_UNKNOWN", "BLOCKED_BUDGET"}
)

# Legal forward transitions of a work item's state. Reopening RETRYABLE->PENDING and the
# routing of failures are handled in code (they depend on retry budget), not here.
_LEGAL: dict[str, frozenset[str]] = {
    "PENDING": frozenset({"CLAIMED", "BLOCKED_BUDGET"}),
    "CLAIMED": frozenset(
        {"MATERIALIZED", "FAILED_RETRYABLE", "FAILED_FINAL", "FAILED_UNKNOWN"}
    ),
    "MATERIALIZED": frozenset(
        {"VALIDATED", "FAILED_RETRYABLE", "FAILED_FINAL", "FAILED_UNKNOWN"}
    ),
    "VALIDATED": frozenset(
        {"COMMITTED", "FAILED_RETRYABLE", "FAILED_FINAL", "FAILED_UNKNOWN"}
    ),
}

_IN_PROGRESS = frozenset({"CLAIMED", "MATERIALIZED", "VALIDATED"})


class LedgerError(RuntimeError):
    pass


class IllegalTransition(LedgerError):
    pass


# --- rows -----------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkItem:
    work_key: str
    protocol_sha: str
    split: str
    phase_id: str
    task_id: str
    arm_id: str
    variant_id: str
    replicate_id: str
    checkpoint_hash: str
    stage_version: str
    side_effecting: bool
    max_retries: int
    retry_count: int
    state: str


@dataclass(frozen=True)
class Attempt:
    attempt_id: str
    work_key: str
    attempt_ordinal: int
    worker_id: str
    state: str
    lease_expires_at: Optional[float]
    result_object_ref: Optional[str]


_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    protocol_sha  TEXT NOT NULL,
    created_at    REAL NOT NULL,
    meta_json     TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS config_versions (
    config_sha    TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,
    canonical     TEXT NOT NULL,
    created_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS work_items (
    work_key        TEXT PRIMARY KEY,
    protocol_sha    TEXT NOT NULL,
    split           TEXT NOT NULL,
    phase_id        TEXT NOT NULL,
    task_id         TEXT NOT NULL,
    arm_id          TEXT NOT NULL,
    variant_id      TEXT NOT NULL,
    replicate_id    TEXT NOT NULL,
    checkpoint_hash TEXT NOT NULL,
    stage_version   TEXT NOT NULL,
    side_effecting  INTEGER NOT NULL DEFAULT 0,
    max_retries     INTEGER NOT NULL DEFAULT 3,
    retry_count     INTEGER NOT NULL DEFAULT 0,
    state           TEXT NOT NULL DEFAULT 'PENDING',
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_work_items_state ON work_items(state);

CREATE TABLE IF NOT EXISTS attempts (
    attempt_id        TEXT PRIMARY KEY,
    work_key          TEXT NOT NULL REFERENCES work_items(work_key),
    attempt_ordinal   INTEGER NOT NULL,
    run_id            TEXT,
    worker_id         TEXT NOT NULL,
    state             TEXT NOT NULL,
    lease_expires_at  REAL,
    started_at        REAL NOT NULL,
    ended_at          REAL,
    terminal_status   TEXT,
    result_object_ref TEXT,
    error_class       TEXT,
    UNIQUE(work_key, attempt_ordinal)
);
-- The core idempotency guarantee: at most one committed attempt per work item.
CREATE UNIQUE INDEX IF NOT EXISTS ux_one_commit
    ON attempts(work_key) WHERE state='COMMITTED';
CREATE INDEX IF NOT EXISTS ix_attempts_lease
    ON attempts(state, lease_expires_at);

CREATE TABLE IF NOT EXISTS transitions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    work_key    TEXT NOT NULL,
    attempt_id  TEXT,
    from_state  TEXT,
    to_state    TEXT NOT NULL,
    at          REAL NOT NULL,
    reason      TEXT
);

CREATE TABLE IF NOT EXISTS incidents (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    at        REAL NOT NULL,
    severity  TEXT NOT NULL,
    kind      TEXT NOT NULL,
    detail    TEXT,
    work_key  TEXT
);

CREATE TABLE IF NOT EXISTS artifacts (
    object_ref   TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,
    raw_size     INTEGER NOT NULL,
    stored_size  INTEGER NOT NULL,
    work_key     TEXT,
    created_at   REAL NOT NULL
);

-- Budget tables live here so the whole infrastructure schema is one coherent unit;
-- the reservation logic that uses them is in budget.py.
CREATE TABLE IF NOT EXISTS budget_accounts (
    resource       TEXT PRIMARY KEY,
    cap            REAL NOT NULL,
    reserved_total REAL NOT NULL DEFAULT 0,
    settled_total  REAL NOT NULL DEFAULT 0,
    updated_at     REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS budget_reservations (
    reservation_id   TEXT PRIMARY KEY,
    resource         TEXT NOT NULL REFERENCES budget_accounts(resource),
    amount           REAL NOT NULL,
    state            TEXT NOT NULL,            -- RESERVED | SETTLED | RELEASED
    settled_amount   REAL,
    work_key         TEXT,
    external_call_id TEXT,
    attempt_id       TEXT,
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_reservations_state ON budget_reservations(state);
CREATE INDEX IF NOT EXISTS ix_reservations_call ON budget_reservations(external_call_id);
CREATE INDEX IF NOT EXISTS ix_reservations_attempt ON budget_reservations(attempt_id);

-- A *logical* external call: one row per thing we meant to buy, keyed by its coordinates.
-- It carries no request or response of its own -- those belong to the physical attempts
-- below, because a retry is a second dispatch and a second possible charge, not an edit
-- of the first one.
CREATE TABLE IF NOT EXISTS external_calls (
    call_id              TEXT PRIMARY KEY,
    provider             TEXT NOT NULL,
    op_class             TEXT NOT NULL,
    call_key             TEXT NOT NULL,
    work_key             TEXT,
    state                TEXT NOT NULL,        -- OPEN | COMMITTED | FAILED_FINAL | FAILED_UNKNOWN
    attempt_count        INTEGER NOT NULL DEFAULT 0,
    committed_attempt_id TEXT,
    created_at           REAL NOT NULL,
    updated_at           REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_external_calls_state ON external_calls(state);

-- Append-only. Every real dispatch gets its own row and its own id; nothing here is ever
-- rewritten to describe a later attempt, so a call that was sent three times shows three
-- request refs, three outcomes and three settlements rather than one surviving guess.
CREATE TABLE IF NOT EXISTS external_call_attempts (
    attempt_id          TEXT PRIMARY KEY,
    call_id             TEXT NOT NULL REFERENCES external_calls(call_id),
    attempt_ordinal     INTEGER NOT NULL,
    state               TEXT NOT NULL,
    request_object_ref  TEXT,
    response_object_ref TEXT,
    provider_request_id TEXT,
    requested_model     TEXT,
    returned_model      TEXT,
    system_fingerprint  TEXT,
    usage_json          TEXT,
    error_class         TEXT,
    opened_at           REAL NOT NULL,
    dispatched_at       REAL,
    settled_at          REAL,
    UNIQUE(call_id, attempt_ordinal)
);
-- The money-side analogue of ux_one_commit: a logical call may be paid for once.
CREATE UNIQUE INDEX IF NOT EXISTS ux_one_external_commit
    ON external_call_attempts(call_id) WHERE state='COMMITTED';
CREATE INDEX IF NOT EXISTS ix_external_attempts_call ON external_call_attempts(call_id);
CREATE INDEX IF NOT EXISTS ix_external_attempts_state ON external_call_attempts(state);

CREATE TABLE IF NOT EXISTS external_call_transitions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id     TEXT NOT NULL,
    attempt_id  TEXT,
    from_state  TEXT,
    to_state    TEXT NOT NULL,
    at          REAL NOT NULL,
    reason      TEXT
);

-- One row per authoring/acquisition round. A round that failed keeps its spend on the
-- books and is marked here, so a later round is a new attempt rather than a silent
-- continuation of a corpus nobody can reconstruct.
CREATE TABLE IF NOT EXISTS corpus_attempts (
    attempt_id     TEXT PRIMARY KEY,
    corpus_version TEXT NOT NULL,
    state          TEXT NOT NULL,
    reason         TEXT,
    protocol_sha   TEXT,
    started_at     REAL NOT NULL,
    closed_at      REAL,
    note           TEXT
);

-- Which engine served each cell. It is in the work key too, but the key is a digest;
-- reading it back is what lets a freeze refuse a block that spans two engines.
CREATE TABLE IF NOT EXISTS cell_epochs (
    work_key     TEXT PRIMARY KEY,
    engine_epoch TEXT NOT NULL,
    recorded_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS schema_versions (
    component  TEXT PRIMARY KEY,
    version    INTEGER NOT NULL,
    applied_at REAL NOT NULL
);
"""

#: Bumped when the physical layout of the external-call tables changes. ``CREATE TABLE IF
#: NOT EXISTS`` silently keeps an old shape, so the migration below is what actually moves
#: a database forward -- and it moves rows, never deletes them.
EXTERNAL_CALL_SCHEMA_VERSION = 2


class Ledger:
    """A single-writer SQLite ledger. One instance owns one write connection."""

    def __init__(
        self,
        path: str,
        *,
        clock: Callable[[], float] = time.time,
        default_max_retries: int = 3,
    ) -> None:
        self._clock = clock
        self._default_max_retries = default_max_retries
        self._lock = threading.RLock()
        # isolation_level=None -> autocommit; we drive BEGIN IMMEDIATE explicitly so the
        # write lock is taken at statement start, not deferred to first write.
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout=30000;")
        with self._lock:
            _migrate_before_schema(self._conn)
            self._conn.executescript(_SCHEMA)
            _migrate_after_schema(self._conn, self._now())

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _now(self) -> float:
        return self._clock()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Cursor]:
        """One serialized write transaction."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("BEGIN IMMEDIATE;")
            try:
                yield cur
            except BaseException:
                cur.execute("ROLLBACK;")
                raise
            else:
                cur.execute("COMMIT;")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Cursor]:
        """A write transaction callers outside this module can join.

        Budget movement and the call-state change that explains it have to commit or roll
        back together; otherwise a crash between them leaves money spent against a call
        the ledger still calls unsent, and restart reconciliation then charges it again.
        """
        with self._tx() as cur:
            yield cur

    def _log_transition(
        self, cur, work_key, attempt_id, from_state, to_state, reason
    ) -> None:
        cur.execute(
            "INSERT INTO transitions(work_key, attempt_id, from_state, to_state, at, reason)"
            " VALUES (?,?,?,?,?,?)",
            (work_key, attempt_id, from_state, to_state, self._now(), reason),
        )

    # --- runs / configs ---------------------------------------------------------------

    def create_run(self, run_id: str, protocol_sha: str, meta_json: str = "{}") -> None:
        with self._tx() as cur:
            cur.execute(
                "INSERT OR IGNORE INTO runs(run_id, protocol_sha, created_at, meta_json)"
                " VALUES (?,?,?,?)",
                (run_id, protocol_sha, self._now(), meta_json),
            )

    def register_config(self, config_sha: str, kind: str, canonical: str) -> None:
        with self._tx() as cur:
            cur.execute(
                "INSERT OR IGNORE INTO config_versions(config_sha, kind, canonical, created_at)"
                " VALUES (?,?,?,?)",
                (config_sha, kind, canonical, self._now()),
            )

    # --- work items -------------------------------------------------------------------

    @staticmethod
    def work_key(
        *,
        protocol_sha: str,
        split: str,
        phase_id: str,
        task_id: str,
        arm_id: str,
        variant_id: str,
        replicate_id: str,
        checkpoint_hash: str,
        stage_version: str,
        engine_epoch: str = "",
    ) -> str:
        """Derive the logical identity. Deliberately excludes run_id and any timestamp so
        the same logical work resolves to the same key across restarts.

        ``engine_epoch`` is included: a cell run before an engine restart and one run after
        it are not the same observation, and a paired block completed across the boundary
        compares two arms served by two engines.
        """
        return derive_id(
            "work_item",
            {
                "protocol_sha": protocol_sha,
                "split": split,
                "phase_id": phase_id,
                "task_id": task_id,
                "arm_id": arm_id,
                "variant_id": variant_id,
                "replicate_id": replicate_id,
                "checkpoint_hash": checkpoint_hash,
                "engine_epoch": engine_epoch,
                "stage_version": stage_version,
            },
        )

    def ensure_work_item(
        self,
        *,
        protocol_sha: str,
        split: str,
        phase_id: str,
        task_id: str,
        arm_id: str,
        variant_id: str,
        replicate_id: str = "0",
        checkpoint_hash: str = "-",
        stage_version: str = "v1",
        engine_epoch: str = "",
        side_effecting: bool = False,
        max_retries: Optional[int] = None,
    ) -> str:
        """Create the work item if absent; return its key. Idempotent -- calling twice
        with the same coordinates never makes two items."""
        key = self.work_key(
            protocol_sha=protocol_sha,
            split=split,
            phase_id=phase_id,
            task_id=task_id,
            arm_id=arm_id,
            variant_id=variant_id,
            replicate_id=replicate_id,
            checkpoint_hash=checkpoint_hash,
            stage_version=stage_version,
            engine_epoch=engine_epoch,
        )
        with self._tx() as cur:
            now = self._now()
            cur.execute(
                "INSERT OR IGNORE INTO work_items("
                "work_key, protocol_sha, split, phase_id, task_id, arm_id, variant_id,"
                "replicate_id, checkpoint_hash, stage_version, side_effecting, max_retries,"
                "retry_count, state, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0,'PENDING',?,?)",
                (
                    key,
                    protocol_sha,
                    split,
                    phase_id,
                    task_id,
                    arm_id,
                    variant_id,
                    replicate_id,
                    checkpoint_hash,
                    stage_version,
                    1 if side_effecting else 0,
                    self._default_max_retries if max_retries is None else max_retries,
                    now,
                    now,
                ),
            )
        return key

    def get_work_item(self, work_key: str) -> Optional[WorkItem]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM work_items WHERE work_key=?", (work_key,)
            ).fetchone()
        return _row_to_work_item(row) if row else None

    def _set_state(self, cur, work_key, from_state, to_state, reason) -> None:
        if to_state not in _LEGAL.get(from_state, frozenset()):
            raise IllegalTransition(f"{from_state} -> {to_state} is not permitted")
        cur.execute(
            "UPDATE work_items SET state=?, updated_at=? WHERE work_key=? AND state=?",
            (to_state, self._now(), work_key, from_state),
        )
        if cur.rowcount != 1:
            raise IllegalTransition(
                f"work item {work_key} was not in {from_state} (concurrent change?)"
            )
        self._log_transition(cur, work_key, None, from_state, to_state, reason)

    # --- claiming / lease -------------------------------------------------------------

    def claim(
        self, work_key: str, worker_id: str, *, lease_seconds: float, run_id: str = ""
    ) -> Optional[Attempt]:
        """Claim a PENDING item. Returns the new CLAIMED attempt, or None if it was not
        claimable (already terminal, or held by a live lease)."""
        with self._tx() as cur:
            row = cur.execute(
                "SELECT state, retry_count FROM work_items WHERE work_key=?", (work_key,)
            ).fetchone()
            if row is None:
                raise LedgerError(f"unknown work item {work_key}")
            if row["state"] != "PENDING":
                return None

            ordinal = (
                cur.execute(
                    "SELECT COALESCE(MAX(attempt_ordinal),0)+1 FROM attempts WHERE work_key=?",
                    (work_key,),
                ).fetchone()[0]
            )
            attempt_id = derive_id("attempt", {"work_key": work_key, "ordinal": ordinal})
            now = self._now()
            expires = now + lease_seconds
            cur.execute(
                "INSERT INTO attempts(attempt_id, work_key, attempt_ordinal, run_id,"
                " worker_id, state, lease_expires_at, started_at)"
                " VALUES (?,?,?,?,?,'CLAIMED',?,?)",
                (attempt_id, work_key, ordinal, run_id, worker_id, expires, now),
            )
            self._set_state(cur, work_key, "PENDING", "CLAIMED", f"claim:{worker_id}")
            return Attempt(
                attempt_id=attempt_id,
                work_key=work_key,
                attempt_ordinal=ordinal,
                worker_id=worker_id,
                state="CLAIMED",
                lease_expires_at=expires,
                result_object_ref=None,
            )

    def heartbeat(self, attempt_id: str, *, lease_seconds: float) -> None:
        """Extend the lease on a live in-progress attempt."""
        with self._tx() as cur:
            cur.execute(
                "UPDATE attempts SET lease_expires_at=? WHERE attempt_id=? AND state IN"
                " ('CLAIMED','MATERIALIZED','VALIDATED')",
                (self._now() + lease_seconds, attempt_id),
            )

    def advance(self, attempt_id: str, to_state: str, *, reason: str = "") -> None:
        """CLAIMED -> MATERIALIZED -> VALIDATED."""
        if to_state not in {"MATERIALIZED", "VALIDATED"}:
            raise IllegalTransition(f"advance target must be MATERIALIZED/VALIDATED, got {to_state}")
        with self._tx() as cur:
            att = _require_attempt(cur, attempt_id)
            self._set_state(cur, att["work_key"], att["state"], to_state, reason)
            cur.execute(
                "UPDATE attempts SET state=? WHERE attempt_id=?", (to_state, attempt_id)
            )

    def commit(self, attempt_id: str, *, result_object_ref: str, reason: str = "") -> None:
        """VALIDATED -> COMMITTED, writing the terminal record last.

        The unique index makes a second commit of the same work item raise, so a buggy
        double-commit surfaces loudly instead of overwriting."""
        with self._tx() as cur:
            att = _require_attempt(cur, attempt_id)
            if att["state"] != "VALIDATED":
                raise IllegalTransition(
                    f"commit requires VALIDATED, attempt is {att['state']}"
                )
            self._set_state(cur, att["work_key"], "VALIDATED", "COMMITTED", reason or "commit")
            try:
                cur.execute(
                    "UPDATE attempts SET state='COMMITTED', ended_at=?, terminal_status="
                    "'COMMITTED', result_object_ref=? WHERE attempt_id=?",
                    (self._now(), result_object_ref, attempt_id),
                )
            except sqlite3.IntegrityError as e:  # pragma: no cover - guard
                raise LedgerError(
                    f"work item {att['work_key']} already has a committed attempt"
                ) from e

    def fail(
        self,
        attempt_id: str,
        *,
        disposition: str,
        error_class: str = "",
        reason: str = "",
    ) -> str:
        """End an attempt in failure. ``disposition`` is FAILED_RETRYABLE / FAILED_FINAL /
        FAILED_UNKNOWN. Returns the work item's resulting state.

        RETRYABLE reopens the item to PENDING while retries remain; a side-effecting item
        that failed UNKNOWN is frozen at FAILED_UNKNOWN because we cannot know whether its
        external effect already happened."""
        if disposition not in {"FAILED_RETRYABLE", "FAILED_FINAL", "FAILED_UNKNOWN"}:
            raise ValueError(f"bad disposition {disposition}")
        with self._tx() as cur:
            att = _require_attempt(cur, attempt_id)
            work_key = att["work_key"]
            wi = cur.execute(
                "SELECT state, retry_count, max_retries FROM work_items WHERE work_key=?",
                (work_key,),
            ).fetchone()
            from_state = wi["state"]
            cur.execute(
                "UPDATE attempts SET state=?, ended_at=?, terminal_status=?, error_class=?"
                " WHERE attempt_id=?",
                (disposition, self._now(), disposition, error_class, attempt_id),
            )

            if disposition == "FAILED_RETRYABLE":
                if wi["retry_count"] + 1 <= wi["max_retries"]:
                    # Reopen for another attempt.
                    cur.execute(
                        "UPDATE work_items SET state='PENDING', retry_count=retry_count+1,"
                        " updated_at=? WHERE work_key=?",
                        (self._now(), work_key),
                    )
                    self._log_transition(
                        cur, work_key, attempt_id, from_state, "PENDING",
                        reason or "retry",
                    )
                    return "PENDING"
                final = "FAILED_FINAL"
            elif disposition == "FAILED_UNKNOWN":
                final = "FAILED_UNKNOWN"
            else:
                final = "FAILED_FINAL"

            cur.execute(
                "UPDATE work_items SET state=?, updated_at=? WHERE work_key=?",
                (final, self._now(), work_key),
            )
            self._log_transition(cur, work_key, attempt_id, from_state, final, reason or error_class)
            return final

    def block_budget(self, work_key: str, *, reason: str = "") -> None:
        """Mark a PENDING item BLOCKED_BUDGET -- a reservation was refused, so no dispatch
        happened. This is admission control's terminal state, reached before any call."""
        with self._tx() as cur:
            wi = cur.execute(
                "SELECT state FROM work_items WHERE work_key=?", (work_key,)
            ).fetchone()
            if wi is None:
                raise LedgerError(f"unknown work item {work_key}")
            self._set_state(cur, work_key, wi["state"], "BLOCKED_BUDGET", reason or "budget")

    # --- stale lease reclamation ------------------------------------------------------

    def reclaim_stale(self, *, now: Optional[float] = None) -> list[str]:
        """Find in-progress attempts whose lease expired and dispose of them.

        A crashed worker leaves its attempt CLAIMED/MATERIALIZED/VALIDATED with a dead
        lease. We end that attempt FAILED_UNKNOWN (we don't know how far it got) and then:
        a non-side-effecting item reopens to PENDING for a fresh attempt if retries remain;
        a side-effecting item is frozen at FAILED_UNKNOWN so a possible external effect is
        never blindly repeated. Returns the reopened work keys."""
        now = self._now() if now is None else now
        reopened: list[str] = []
        with self._tx() as cur:
            stale = cur.execute(
                "SELECT attempt_id, work_key FROM attempts WHERE state IN"
                " ('CLAIMED','MATERIALIZED','VALIDATED') AND lease_expires_at IS NOT NULL"
                " AND lease_expires_at < ?",
                (now,),
            ).fetchall()
            for att in stale:
                work_key = att["work_key"]
                wi = cur.execute(
                    "SELECT state, retry_count, max_retries, side_effecting"
                    " FROM work_items WHERE work_key=?",
                    (work_key,),
                ).fetchone()
                from_state = wi["state"]
                cur.execute(
                    "UPDATE attempts SET state='FAILED_UNKNOWN', ended_at=?,"
                    " terminal_status='FAILED_UNKNOWN', error_class='lease_expired'"
                    " WHERE attempt_id=?",
                    (now, att["attempt_id"]),
                )
                if not wi["side_effecting"] and wi["retry_count"] + 1 <= wi["max_retries"]:
                    cur.execute(
                        "UPDATE work_items SET state='PENDING', retry_count=retry_count+1,"
                        " updated_at=? WHERE work_key=?",
                        (now, work_key),
                    )
                    self._log_transition(
                        cur, work_key, att["attempt_id"], from_state, "PENDING", "lease_reclaim"
                    )
                    reopened.append(work_key)
                else:
                    final = "FAILED_UNKNOWN" if wi["side_effecting"] else "FAILED_FINAL"
                    cur.execute(
                        "UPDATE work_items SET state=?, updated_at=? WHERE work_key=?",
                        (final, now, work_key),
                    )
                    self._log_transition(
                        cur, work_key, att["attempt_id"], from_state, final, "lease_reclaim"
                    )
        return reopened

    # --- queries used by planning / resume --------------------------------------------

    def is_committed(self, work_key: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM work_items WHERE work_key=? AND state='COMMITTED'", (work_key,)
            ).fetchone()
        return row is not None

    def committed_ref(self, work_key: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT result_object_ref FROM attempts WHERE work_key=? AND state='COMMITTED'",
                (work_key,),
            ).fetchone()
        return row["result_object_ref"] if row else None

    def pending_keys(self, *, limit: int = 1000) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT work_key FROM work_items WHERE state='PENDING'"
                " ORDER BY created_at LIMIT ?",
                (limit,),
            ).fetchall()
        return [r["work_key"] for r in rows]

    def state_counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT state, COUNT(*) c FROM work_items GROUP BY state"
            ).fetchall()
        return {r["state"]: r["c"] for r in rows}

    # --- artifacts / incidents --------------------------------------------------------

    def register_artifact(
        self, object_ref: str, *, kind: str, raw_size: int, stored_size: int,
        work_key: Optional[str] = None,
    ) -> None:
        with self._tx() as cur:
            cur.execute(
                "INSERT OR IGNORE INTO artifacts(object_ref, kind, raw_size, stored_size,"
                " work_key, created_at) VALUES (?,?,?,?,?,?)",
                (object_ref, kind, raw_size, stored_size, work_key, self._now()),
            )

    def record_incident(
        self, *, severity: str, kind: str, detail: str = "", work_key: Optional[str] = None
    ) -> None:
        with self._tx() as cur:
            cur.execute(
                "INSERT INTO incidents(at, severity, kind, detail, work_key) VALUES (?,?,?,?,?)",
                (self._now(), severity, kind, detail, work_key),
            )

    # --- corpus attempts --------------------------------------------------------------

    def record_corpus_attempt(
        self,
        *,
        attempt_id: str,
        corpus_version: str,
        state: str,
        reason: str = "",
        protocol_sha: str = "",
        note: str = "",
        closed: bool = True,
    ) -> None:
        """Append one round of corpus authoring/acquisition and how it ended.

        Append-only by construction: an attempt id that already exists is left alone
        rather than rewritten, so a failed round cannot be relabelled as a later success.
        """
        with self._tx() as cur:
            cur.execute(
                "INSERT OR IGNORE INTO corpus_attempts(attempt_id, corpus_version, state,"
                " reason, protocol_sha, started_at, closed_at, note) VALUES (?,?,?,?,?,?,?,?)",
                (
                    attempt_id, corpus_version, state, reason, protocol_sha,
                    self._now(), self._now() if closed else None, note,
                ),
            )

    def record_engine_epoch(self, work_key: str, engine_epoch: str) -> None:
        with self._tx() as cur:
            cur.execute(
                "INSERT OR IGNORE INTO cell_epochs(work_key, engine_epoch, recorded_at)"
                " VALUES (?,?,?)",
                (work_key, engine_epoch, self._now()),
            )

    def corpus_attempts(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM corpus_attempts ORDER BY started_at"
            ).fetchall()
        return [dict(r) for r in rows]

    def backup_to(self, path: str) -> None:
        """Take a consistent copy through SQLite's backup API.

        Never copy the file: a WAL database on disk is the main file plus a write-ahead log,
        and a plain ``cp`` of the three parts at three different moments produces an
        archive that can be silently missing the most recent transactions (§16.3).
        """
        with self._lock:
            target = sqlite3.connect(path)
            try:
                self._conn.backup(target)
            finally:
                target.close()

    def integrity_check(self) -> bool:
        with self._lock:
            row = self._conn.execute("PRAGMA integrity_check;").fetchone()
        return row[0] == "ok"

    # exposed for budget.py / external_call_ledger.py which share this connection
    @property
    def raw_connection(self) -> sqlite3.Connection:
        return self._conn

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def now(self) -> float:
        return self._now()


# --- schema migration -------------------------------------------------------------------
#
# There is no drop and no rewrite anywhere below. A database that already recorded real
# spend is carried forward row by row: the v1 external-call rows become one attempt each,
# and the budget tables are never touched except to add the link column that lets a
# reservation name the dispatch it paid for.


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


#: v1 kept the physical state on the call row. Splitting them, the physical state moves to
#: the attempt and the call gets a logical one. Anything that did not commit is ABANDONED
#: rather than OPEN: those rows belong to a round that is being frozen, and reopening them
#: would let a later run silently continue a corpus attempt that already failed.
_V1_TO_LOGICAL = {
    "COMMITTED": "COMMITTED",
}

_TERMINAL_V1 = frozenset({"COMMITTED", "FAILED_FINAL", "FAILED_UNKNOWN"})


def _migrate_before_schema(conn: sqlite3.Connection) -> None:
    """Reshape what the schema script cannot: ``CREATE TABLE IF NOT EXISTS`` leaves an
    existing table alone, and an index over a column that table does not have yet fails."""
    reservation_cols = _columns(conn, "budget_reservations")
    if reservation_cols and "attempt_id" not in reservation_cols:
        conn.execute("ALTER TABLE budget_reservations ADD COLUMN attempt_id TEXT")
    call_cols = _columns(conn, "external_calls")
    if call_cols and "call_key" not in call_cols:
        # v1: one mutable row per logical call. Renamed, never dropped.
        conn.execute("ALTER TABLE external_calls RENAME TO external_calls_v1")


def _migrate_after_schema(conn: sqlite3.Connection, now: float) -> None:
    row = conn.execute(
        "SELECT version FROM schema_versions WHERE component='external_calls'"
    ).fetchone()
    if row is not None and row["version"] >= EXTERNAL_CALL_SCHEMA_VERSION:
        return
    if _columns(conn, "external_calls_v1"):
        _copy_v1_external_calls(conn, now)
    conn.execute(
        "INSERT INTO schema_versions(component, version, applied_at) VALUES"
        " ('external_calls',?,?) ON CONFLICT(component) DO UPDATE SET"
        " version=excluded.version, applied_at=excluded.applied_at",
        (EXTERNAL_CALL_SCHEMA_VERSION, now),
    )


def _copy_v1_external_calls(conn: sqlite3.Connection, now: float) -> None:
    for old in conn.execute("SELECT * FROM external_calls_v1").fetchall():
        call_id = old["call_id"]
        physical = old["state"]
        logical = _V1_TO_LOGICAL.get(physical, "ABANDONED")
        ordinal = int(old["attempt_ordinal"] or 0)
        attempt_id = derive_id(
            "external_call_attempt", {"call_id": call_id, "ordinal": ordinal}
        )
        dispatched = old["updated_at"] if physical not in ("INTENT", "BUDGET_RESERVED") else None
        settled = old["updated_at"] if physical in _TERMINAL_V1 else None
        conn.execute(
            "INSERT OR IGNORE INTO external_calls(call_id, provider, op_class, call_key,"
            " work_key, state, attempt_count, committed_attempt_id, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                call_id, old["provider"], old["op_class"],
                # v1 never stored the call key; recording that plainly beats inventing one.
                "<v1-unrecorded>", old["work_key"], logical, 1,
                attempt_id if logical == "COMMITTED" else None,
                old["created_at"], old["updated_at"],
            ),
        )
        conn.execute(
            "INSERT OR IGNORE INTO external_call_attempts(attempt_id, call_id,"
            " attempt_ordinal, state, request_object_ref, response_object_ref,"
            " provider_request_id, requested_model, returned_model, system_fingerprint,"
            " usage_json, error_class, opened_at, dispatched_at, settled_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                attempt_id, call_id, ordinal, physical,
                old["request_object_ref"], old["response_object_ref"],
                old["provider_request_id"],
                None, old["model"], None,
                old["usage_json"], old["error_class"],
                old["created_at"], dispatched, settled,
            ),
        )
        conn.execute(
            "UPDATE budget_reservations SET attempt_id=? WHERE external_call_id=?"
            " AND attempt_id IS NULL",
            (attempt_id, call_id),
        )
    conn.execute(
        "INSERT INTO external_call_transitions(call_id, attempt_id, from_state, to_state,"
        " at, reason) SELECT call_id, attempt_id, NULL, state, ?, 'migrated_from_v1'"
        " FROM external_call_attempts",
        (now,),
    )


def _require_attempt(cur, attempt_id: str) -> sqlite3.Row:
    row = cur.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
    if row is None:
        raise LedgerError(f"unknown attempt {attempt_id}")
    return row


def _row_to_work_item(row: sqlite3.Row) -> WorkItem:
    return WorkItem(
        work_key=row["work_key"],
        protocol_sha=row["protocol_sha"],
        split=row["split"],
        phase_id=row["phase_id"],
        task_id=row["task_id"],
        arm_id=row["arm_id"],
        variant_id=row["variant_id"],
        replicate_id=row["replicate_id"],
        checkpoint_hash=row["checkpoint_hash"],
        stage_version=row["stage_version"],
        side_effecting=bool(row["side_effecting"]),
        max_retries=row["max_retries"],
        retry_count=row["retry_count"],
        state=row["state"],
    )
