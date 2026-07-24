"""Retry classification and backoff, specified rather than improvised.

Different failures deserve different responses, and getting this wrong either wastes budget
hammering a dead endpoint or gives up on a transient blip. The policy (plan §12.3):

- **429** -- respect ``Retry-After``; the server told us when to come back.
- **5xx** -- capped exponential backoff with full jitter; a transient server problem.
- **400 / 401 / 402 / 422** -- fail fast; a bad request, bad key, or exhausted quota will
  not fix itself on retry.
- **empty / invalid JSON** -- a small bounded number of retries; occasionally a provider
  returns a truncated body.

The backoff's jitter and sleep are injected so the logic is deterministically testable
without real time passing.
"""

from __future__ import annotations

import enum
from typing import Callable, Optional

__all__ = ["RetryDecision", "classify_status", "capped_backoff", "FAIL_FAST_STATUSES"]

FAIL_FAST_STATUSES = frozenset({400, 401, 402, 403, 404, 422})


class RetryDecision(enum.Enum):
    RETRY_AFTER = "RETRY_AFTER"          # honor server-provided delay (429)
    BACKOFF = "BACKOFF"                  # exponential backoff + jitter (5xx)
    FAIL_FAST = "FAIL_FAST"              # do not retry
    RETRYABLE_LIMITED = "RETRYABLE_LIMITED"  # small bounded retries (bad body)


def classify_status(status: int) -> RetryDecision:
    if status == 429:
        return RetryDecision.RETRY_AFTER
    if status in FAIL_FAST_STATUSES:
        return RetryDecision.FAIL_FAST
    if 500 <= status < 600:
        return RetryDecision.BACKOFF
    return RetryDecision.FAIL_FAST


def capped_backoff(
    attempt: int,
    *,
    base: float = 0.5,
    cap: float = 30.0,
    jitter: Callable[[float], float] = lambda x: x,
) -> float:
    """Full-jitter capped exponential backoff for ``attempt`` (0-based).

    ``delay = jitter(min(cap, base * 2**attempt))``. With the default identity ``jitter`` the
    result is the deterministic ceiling (useful for tests); in production ``jitter`` samples
    uniformly in ``[0, ceiling]`` (full jitter), which decorrelates concurrent retries.
    """
    if attempt < 0:
        raise ValueError("attempt must be >= 0")
    ceiling = min(cap, base * (2 ** attempt))
    return jitter(ceiling)
