"""Outcome-free design locks and the launch approval.

A freeze record is a tracked, content-addressed lock written BEFORE any outcome is observed:
the champion variant, the thresholds, the sample-size plan, the manifest hashes. It exists so
that "what was pre-registered" is a hash, not a promise. Any change after freeze mints a new
protocol version rather than overwriting -- so the frozen record and the results computed under
it can never silently disagree.

The launch approval pins the protocol/budget/threshold hashes and records the approval MODE.
For this campaign the user explicitly chose gate-green-then-pause, which OVERRIDES the plan's
v0.1 auto-launch default: all hard safety/integrity gates must pass and then the runner stops
and waits for an explicit human go-ahead before spending any Tavily credits or GPU hours.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..canonical import canonical_json, canonical_str
from ..hashing import sha256_hex

__all__ = [
    "APPROVAL_MODE_PAUSE",
    "APPROVAL_MODE_AUTO",
    "FreezeRecord",
    "build_launch_approval",
    "verify_launch_approval",
    "ApprovalMismatch",
]

# Gate-green-then-pause: the mode selected for this campaign. It supersedes the plan's auto mode.
APPROVAL_MODE_PAUSE = "USER_EXPLICIT_GATE_GREEN_THEN_PAUSE"
# The plan's original v0.1 mode, kept only so the override is explicit and auditable.
APPROVAL_MODE_AUTO = "USER_EXPLICIT_AUTO_LAUNCH"


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
    mode: str = APPROVAL_MODE_PAUSE,
) -> dict:
    """Materialize the launch_approval content. Defaults to the gate-green-then-pause mode the
    user chose; ``requires_human_launch`` is True in that mode so the runner cannot self-start."""
    return {
        "protocol_sha": protocol_sha,
        "budget_sha": budget_sha,
        "decision_thresholds_sha": decision_thresholds_sha,
        "approval_mode": mode,
        "approval_source_date": approval_source_date,
        "approved_at_utc": approved_at_utc,
        "requires_human_launch": mode != APPROVAL_MODE_AUTO,
        "note": (
            "User chose gate-green-then-pause on 2026-07-24, overriding the plan's auto-launch. "
            "All hard gates must pass; the runner then stops for an explicit human go-ahead "
            "before spending Tavily credits or GPU hours."
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
