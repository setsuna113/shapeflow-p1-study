"""Drift detection and the auto-stop decision (plan §17.3-§17.4).

Two jobs, both pure decisions over sampled state so they can be tested without hardware:

- **Period drift.** A fixed P0 sentinel runs periodically; if its latency or throughput drifts past
  the config guard versus the frozen baseline, the affected pair/block is invalidated
  (PERIOD_INVALIDATED) and re-run whole -- never patched by topping up the worse arm.
- **Auto-stop.** A set of conditions that must halt the campaign immediately: a GPU ECC/Xid/driver
  error, an OOM streak, an engine/model drift, a secret-leak hit, a snapshot/freeze hash change, a
  P0 parity failure, budget exhaustion, a SQLite integrity failure, or a holdout read-before-release.
  The decision is centralized here so "when do we stop" is one auditable list.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["DriftGuard", "check_drift", "AutoStop", "evaluate_auto_stop"]


@dataclass(frozen=True)
class DriftGuard:
    max_latency_ratio: float = 1.25   # current/baseline latency
    min_throughput_ratio: float = 0.80  # current/baseline tokens-per-second


def check_drift(
    *, baseline_latency: float, current_latency: float,
    baseline_tps: float, current_tps: float, guard: DriftGuard,
) -> list[str]:
    """Return drift reasons (empty = within guard). A non-empty result invalidates the period."""
    reasons = []
    if baseline_latency > 0 and current_latency / baseline_latency > guard.max_latency_ratio:
        reasons.append(
            f"latency drift {current_latency / baseline_latency:.2f}x > {guard.max_latency_ratio}x"
        )
    if baseline_tps > 0 and current_tps / baseline_tps < guard.min_throughput_ratio:
        reasons.append(
            f"throughput drift {current_tps / baseline_tps:.2f}x < {guard.min_throughput_ratio}x"
        )
    return reasons


@dataclass(frozen=True)
class AutoStop:
    stop: bool
    reasons: tuple[str, ...] = ()


def evaluate_auto_stop(
    *,
    gpu_hardware_error: bool = False,
    consecutive_oom: int = 0,
    oom_threshold: int = 3,
    engine_or_model_drift: bool = False,
    secret_leak_detected: bool = False,
    freeze_hash_changed: bool = False,
    p0_parity_failed: bool = False,
    budget_exhausted: bool = False,
    sqlite_integrity_failed: bool = False,
    holdout_read_before_release: bool = False,
) -> AutoStop:
    """Centralized halt decision. Any true condition stops the campaign."""
    reasons: list[str] = []
    if gpu_hardware_error:
        reasons.append("GPU ECC/Xid/driver error")
    if consecutive_oom >= oom_threshold:
        reasons.append(f"OOM streak {consecutive_oom} >= {oom_threshold}")
    if engine_or_model_drift:
        reasons.append("engine/model revision drift")
    if secret_leak_detected:
        reasons.append("secret leakage detector hit")
    if freeze_hash_changed:
        reasons.append("snapshot/freeze hash changed")
    if p0_parity_failed:
        reasons.append("P0 parity failure")
    if budget_exhausted:
        reasons.append("budget exhausted")
    if sqlite_integrity_failed:
        reasons.append("SQLite integrity failure")
    if holdout_read_before_release:
        reasons.append("holdout read before gate release")
    return AutoStop(stop=bool(reasons), reasons=tuple(reasons))
