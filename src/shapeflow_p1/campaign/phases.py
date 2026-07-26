"""The campaign's phase state, in the ledger, so a restart resumes rather than repeats.

The pure legality rules already live in :mod:`shapeflow_p1.experiment.state_machine`. What was
missing is durability: a phase machine held in memory forgets everything a crash interrupts, and
the campaign would either redo a phase that already spent Tavily credits or skip one that never
finished. So the current phase and the history of transitions are rows.

Two rules the table enforces:

**A phase completes once.** ``complete`` is idempotent for the same detail digest and refuses a
second, different completion. A phase that could complete twice with different evidence would
make "acquisition is done" a statement with two answers.

**Only legal edges.** Transitions go through the same pure FSM, so a campaign cannot reach
SCREEN_RUNNING without P0 parity, or open a holdout without a policy freeze -- and the rule
lives in exactly one place.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from ..canonical import canonical_json
from ..experiment.ledger import Ledger
from ..experiment.state_machine import (
    IllegalPhaseTransition,
    Phase,
    can_transition,
)
from ..hashing import sha256_hex

__all__ = ["PhaseStore", "PhaseRecord", "PhaseError"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS campaign_phases (
    phase         TEXT PRIMARY KEY,
    state         TEXT NOT NULL,          -- RUNNING | COMPLETE | FAILED
    protocol_sha  TEXT NOT NULL,
    detail_json   TEXT NOT NULL DEFAULT '{}',
    detail_sha    TEXT NOT NULL DEFAULT '',
    started_at    REAL NOT NULL,
    ended_at      REAL
);

CREATE TABLE IF NOT EXISTS campaign_phase_transitions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    from_phase  TEXT,
    to_phase    TEXT NOT NULL,
    at          REAL NOT NULL,
    reason      TEXT
);

-- v2 scopes state by the complete approved execution binding.  The v1 table used ``phase`` as
-- its only primary key, so a newly approved experiment silently inherited the old campaign's
-- completed gates.
CREATE TABLE IF NOT EXISTS campaign_phases_v2 (
    protocol_sha  TEXT NOT NULL,
    phase         TEXT NOT NULL,
    state         TEXT NOT NULL,
    detail_json   TEXT NOT NULL DEFAULT '{}',
    detail_sha    TEXT NOT NULL DEFAULT '',
    started_at    REAL NOT NULL,
    ended_at      REAL,
    PRIMARY KEY(protocol_sha, phase)
);

CREATE TABLE IF NOT EXISTS campaign_phase_transitions_v2 (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    protocol_sha  TEXT NOT NULL,
    from_phase    TEXT,
    to_phase      TEXT NOT NULL,
    at            REAL NOT NULL,
    reason        TEXT
);
"""


class PhaseError(RuntimeError):
    pass


@dataclass(frozen=True)
class PhaseRecord:
    phase: Phase
    state: str
    detail: dict
    detail_sha: str


class PhaseStore:
    """Durable phase state layered on the ledger's connection."""

    def __init__(self, ledger: Ledger, *, protocol_sha: str) -> None:
        self._ledger = ledger
        self._protocol_sha = protocol_sha
        with ledger.lock:
            ledger.raw_connection.executescript(_SCHEMA)
            # Preserve every v1 phase row under the identity it already recorded.  v1 could
            # hold at most one row per phase, so this migration is deterministic and
            # idempotent.  New writes go only to v2.
            ledger.raw_connection.execute(
                "INSERT OR IGNORE INTO campaign_phases_v2("
                " protocol_sha, phase, state, detail_json, detail_sha, started_at, ended_at)"
                " SELECT protocol_sha, phase, state, detail_json, detail_sha, started_at,"
                " ended_at FROM campaign_phases"
            )
            ledger.raw_connection.execute(
                "INSERT OR IGNORE INTO campaign_phase_transitions_v2("
                " id, protocol_sha, from_phase, to_phase, at, reason)"
                " SELECT t.id, p.protocol_sha, t.from_phase, t.to_phase, t.at, t.reason"
                " FROM campaign_phase_transitions AS t"
                " JOIN campaign_phases AS p ON p.phase=t.to_phase"
            )

    def _conn(self):
        return self._ledger.raw_connection

    def _write(self, sql: str, params: tuple) -> None:
        with self._ledger.lock:
            cur = self._conn().cursor()
            cur.execute("BEGIN IMMEDIATE;")
            try:
                cur.execute(sql, params)
                cur.execute("COMMIT;")
            except BaseException:
                cur.execute("ROLLBACK;")
                raise

    def record(self, phase: Phase) -> Optional[PhaseRecord]:
        with self._ledger.lock:
            row = self._conn().execute(
                "SELECT phase, state, detail_json, detail_sha FROM campaign_phases_v2"
                " WHERE protocol_sha=? AND phase=?",
                (self._protocol_sha, phase.value),
            ).fetchone()
        if row is None:
            return None
        return PhaseRecord(
            phase=Phase(row["phase"]), state=row["state"],
            detail=json.loads(row["detail_json"]), detail_sha=row["detail_sha"],
        )

    def is_complete(self, phase: Phase) -> bool:
        record = self.record(phase)
        return record is not None and record.state == "COMPLETE"

    def current(self) -> Phase:
        """The furthest phase that completed, or NEW."""
        with self._ledger.lock:
            rows = self._conn().execute(
                "SELECT phase FROM campaign_phases_v2"
                " WHERE protocol_sha=? AND state='COMPLETE'",
                (self._protocol_sha,),
            ).fetchall()
        completed = {Phase(r["phase"]) for r in rows}
        latest = Phase.NEW
        for phase in _ORDER:
            if phase in completed:
                latest = phase
        return latest

    def begin(self, phase: Phase, *, reason: str = "") -> None:
        """Mark a phase running, refusing an illegal edge from the current one."""
        if self.is_complete(phase):
            return
        current = self.current()
        if phase is not current and not can_transition(current, phase):
            raise IllegalPhaseTransition(
                f"{current.value} -> {phase.value} is not a legal edge; a campaign cannot skip "
                "a gate by starting the phase after it"
            )
        now = self._ledger.now()
        self._write(
            "INSERT INTO campaign_phases_v2(protocol_sha, phase, state, started_at)"
            " VALUES (?,?,'RUNNING',?)"
            " ON CONFLICT(protocol_sha, phase) DO UPDATE SET"
            " state='RUNNING', started_at=excluded.started_at",
            (self._protocol_sha, phase.value, now),
        )
        self._write(
            "INSERT INTO campaign_phase_transitions_v2("
            " protocol_sha, from_phase, to_phase, at, reason)"
            " VALUES (?,?,?,?,?)",
            (self._protocol_sha, current.value, phase.value, now, reason or "begin"),
        )

    def complete(self, phase: Phase, detail: dict) -> str:
        """Record a phase complete with its evidence. Idempotent; a different result is fatal."""
        digest = sha256_hex(canonical_json(detail))
        existing = self.record(phase)
        if existing is not None and existing.state == "COMPLETE":
            if existing.detail_sha != digest:
                raise PhaseError(
                    f"{phase.value} already completed with different evidence "
                    f"({existing.detail_sha[:12]} vs {digest[:12]}); a phase that can complete "
                    "twice makes its result a statement with two answers"
                )
            return digest
        self._write(
            "INSERT INTO campaign_phases_v2("
            " protocol_sha, phase, state, detail_json, detail_sha, started_at, ended_at)"
            " VALUES (?,?,'COMPLETE',?,?,?,?)"
            " ON CONFLICT(protocol_sha, phase) DO UPDATE SET state='COMPLETE',"
            " detail_json=excluded.detail_json, detail_sha=excluded.detail_sha,"
            " ended_at=excluded.ended_at",
            (self._protocol_sha, phase.value, json.dumps(detail, sort_keys=True), digest,
             self._ledger.now(), self._ledger.now()),
        )
        return digest

    def fail(self, phase: Phase, reason: str) -> None:
        self._write(
            "INSERT INTO campaign_phases_v2("
            " protocol_sha, phase, state, detail_json, started_at, ended_at)"
            " VALUES (?,?,'FAILED',?,?,?)"
            " ON CONFLICT(protocol_sha, phase) DO UPDATE SET state='FAILED',"
            " detail_json=excluded.detail_json, ended_at=excluded.ended_at",
            (self._protocol_sha, phase.value, json.dumps({"reason": reason}),
             self._ledger.now(), self._ledger.now()),
        )

    def history(self) -> list[dict]:
        with self._ledger.lock:
            rows = self._conn().execute(
                "SELECT from_phase, to_phase, at, reason"
                " FROM campaign_phase_transitions_v2 WHERE protocol_sha=? ORDER BY id",
                (self._protocol_sha,),
            ).fetchall()
        return [dict(r) for r in rows]


#: Phase order for "how far has the campaign got". Only the linear spine; the branches
#: (CHARACTERIZE_NO_GO, the terminals) are legality questions, answered by the pure FSM.
_ORDER = (
    Phase.NEW,
    Phase.DOCTOR_PASSED,
    Phase.ACQUISITION_COMPLETE,
    Phase.SNAPSHOTS_FROZEN,
    Phase.P0_PARITY_PASSED,
    Phase.GPU_SMOKE_PASSED,
    Phase.SCREEN_RUNNING,
    Phase.SCREEN_COMPLETE,
    Phase.MINI_ITT_RUNNING,
    Phase.MINI_ITT_COMPLETE,
    Phase.POLICY_FROZEN,
    Phase.HOLDOUT_RUNNING,
    Phase.HOLDOUT_COMPLETE,
    Phase.REPORT_COMPLETE,
)
