"""Outcome-free design locks and the launch approval.

A freeze record is a tracked, content-addressed lock written BEFORE any outcome is observed:
the champion variant, the thresholds, the sample-size plan, the manifest hashes. It exists so
that "what was pre-registered" is a hash, not a promise. Any change after freeze mints a new
protocol version rather than overwriting -- so the frozen record and the results computed under
it can never silently disagree.

The launch approval pins the protocol/budget/threshold hashes and records the approval MODE.
The protocol of record (`SHAPEFLOW_P1_WEEK1_CODING_PLAN_v0.1_2026-07-24.md` sections 0 and 4.1)
fixes that mode as ``USER_EXPLICIT_AUTO_LAUNCH``: the user's "build it and start running"
instruction of 2026-07-24 *is* the launch authorization, so once every hard gate is green the
runner starts the campaign without asking again. Auto-launch is not a bypass -- a missing
secret, stack mismatch, unfrozen data, P0 parity failure, unclosed schema or smoke failure
still fails closed into ``reports/BLOCKED*.md`` (plan section 0, AGENTS.md section 7).

There is deliberately no "gate-green-then-pause" mode. A mode that stops for a human go-ahead
would be a *different* user decision, and inventing one here would encode a decision the user
never made.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..canonical import canonical_json, canonical_str
from ..hashing import sha256_hex

__all__ = [
    "APPROVAL_MODE_AUTO",
    "APPROVAL_MODES",
    "FreezeRecord",
    "build_launch_approval",
    "verify_launch_approval",
    "ApprovalMismatch",
]

# The only mode protocol v0.1 authorizes. Adding a mode here is a protocol change: it mints a
# new protocol SHA and invalidates every existing approval.
APPROVAL_MODE_AUTO = "USER_EXPLICIT_AUTO_LAUNCH"
APPROVAL_MODES = frozenset({APPROVAL_MODE_AUTO})


class ApprovalMismatch(RuntimeError):
    """The approval file's pinned hashes do not match the loaded configuration."""


@dataclass(frozen=True)
class FreezeRecord:
    """An outcome-free design lock. ``content`` excludes the digest; ``digest`` is over it."""

    kind: str  # e.g. "p1_holdout_v1"
    protocol_sha: str
    champion_variant: dict
    thresholds_sha: str
    budget_sha: str
    holdout_task_ids: tuple[str, ...]
    randomization_schedule_sha: str
    software_manifest_sha: str
    sample_size_plan: dict

    def content(self) -> dict:
        return {
            "kind": self.kind,
            "protocol_sha": self.protocol_sha,
            "champion_variant": self.champion_variant,
            "thresholds_sha": self.thresholds_sha,
            "budget_sha": self.budget_sha,
            "holdout_task_ids": list(self.holdout_task_ids),
            "randomization_schedule_sha": self.randomization_schedule_sha,
            "software_manifest_sha": self.software_manifest_sha,
            "sample_size_plan": self.sample_size_plan,
        }

    @property
    def digest(self) -> str:
        return sha256_hex(canonical_json(self.content()))

    def to_json(self) -> str:
        body = self.content()
        body["freeze_sha256"] = self.digest
        return canonical_str(body)


def build_launch_approval(
    *,
    protocol_sha: str,
    budget_sha: str,
    decision_thresholds_sha: str,
    approved_at_utc: str,
    approval_source_date: str = "2026-07-24",
    mode: str = APPROVAL_MODE_AUTO,
) -> dict:
    """Materialize the launch_approval content.

    This only *hashes values the protocol already fixed*; it is not a fresh approval request and
    it may not invent a mode. An unrecognized ``mode`` is a hard error rather than a silently
    accepted new policy.
    """
    if mode not in APPROVAL_MODES:
        raise ApprovalMismatch(
            f"approval_mode {mode!r} is not authorized by protocol v0.1 "
            f"(authorized: {sorted(APPROVAL_MODES)}); a new mode requires a new protocol SHA"
        )
    return {
        "protocol_sha": protocol_sha,
        "budget_sha": budget_sha,
        "decision_thresholds_sha": decision_thresholds_sha,
        "approval_mode": mode,
        "approval_source_date": approval_source_date,
        "approved_at_utc": approved_at_utc,
        "note": (
            "Protocol v0.1 auto-launch: the user's 2026-07-24 instruction to build and start "
            "running is the launch authorization. Once every hard gate is green the runner "
            "starts the campaign without asking again. Any gate failure still fails closed "
            "into reports/BLOCKED*.md."
        ),
    }


def verify_launch_approval(approval: dict, *, protocol_sha: str, budget_sha: str,
                           decision_thresholds_sha: str) -> None:
    """Raise ApprovalMismatch unless the approval's pinned hashes match the live configuration.
    The launch gate calls this so an edited config can never run under a stale approval."""
    expected = {
        "protocol_sha": protocol_sha,
        "budget_sha": budget_sha,
        "decision_thresholds_sha": decision_thresholds_sha,
    }
    for key, want in expected.items():
        got = approval.get(key)
        if got != want:
            raise ApprovalMismatch(
                f"{key}: approval pins {got!r} but live config is {want!r}; re-approval required"
            )
    mode = approval.get("approval_mode")
    if mode not in APPROVAL_MODES:
        raise ApprovalMismatch(
            f"approval_mode {mode!r} is not authorized by protocol v0.1 "
            f"(authorized: {sorted(APPROVAL_MODES)})"
        )
