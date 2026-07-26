"""Cluster bootstrap confidence intervals.

The independent unit is the topic/source cluster, so resampling is over clusters (all tasks in
a drawn cluster come along together). That is what stops correlated tasks from masquerading as
independent evidence and inflating precision. Seeding is explicit and derived from the protocol,
so intervals are reproducible: the same data and seed give the same CI on every machine.

The work-saving estimand is the paired log-ratio one: ``saving = 1 - exp(mean(log(W_P1/W_P0)))``.
It supports the pre-registered one-sided 95% lower and upper decision bounds.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["BootstrapCI", "seed_from", "cluster_bootstrap_ci", "paired_log_ratio_saving"]


@dataclass(frozen=True)
class BootstrapCI:
    point: float
    lower: float  # one-sided or two-sided lower bound per `side`
    upper: float
    n_clusters: int
    n_boot: int


def seed_from(*parts: str) -> int:
    """Derive a deterministic 32-bit seed from protocol strings, so a re-run reproduces the CI
    without hardcoding a magic number."""
    h = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")


def _cluster_index(cluster_ids: Sequence[str]) -> dict[str, list[int]]:
    idx: dict[str, list[int]] = {}
    for i, c in enumerate(cluster_ids):
        idx.setdefault(c, []).append(i)
    return idx


def cluster_bootstrap_ci(
    values: Sequence[float],
    cluster_ids: Sequence[str],
    statistic: Callable[[np.ndarray], float],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    side: str = "two",
    within_cluster: Callable[[np.ndarray], float] = np.mean,
) -> BootstrapCI:
    """Bootstrap ``statistic`` over resampled clusters.

    ``side`` is 'lower' (one-sided lower bound at 1-alpha), 'upper', or 'two'.

    The input is first reduced to one summary per cluster using ``within_cluster``. The
    requested statistic is then applied to those equally weighted summaries. Merely drawing
    clusters and concatenating their rows, as the previous implementation did, still gives a
    nine-row cluster nine times the weight of a one-row cluster in every resample.
    """
    if len(values) != len(cluster_ids):
        raise ValueError("values and cluster_ids must align")
    arr = np.asarray(values, dtype=float)
    idx = _cluster_index(cluster_ids)
    clusters = list(idx.keys())
    if len(clusters) < 2:
        raise ValueError("need at least 2 clusters for a cluster bootstrap")

    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if side not in {"lower", "upper", "two"}:
        raise ValueError("side must be 'lower', 'upper' or 'two'")

    cluster_values = np.asarray([
        float(within_cluster(arr[idx[cluster]])) for cluster in clusters
    ], dtype=float)
    rng = np.random.default_rng(seed)
    point = float(statistic(cluster_values))
    stats = np.empty(n_boot, dtype=float)
    n_clusters = len(clusters)
    for b in range(n_boot):
        drawn = rng.integers(0, n_clusters, size=n_clusters)
        stats[b] = statistic(cluster_values[drawn])

    if side == "lower":
        lower = float(np.quantile(stats, alpha))
        upper = float("inf")
    elif side == "upper":
        lower = float("-inf")
        upper = float(np.quantile(stats, 1 - alpha))
    else:
        lower = float(np.quantile(stats, alpha / 2))
        upper = float(np.quantile(stats, 1 - alpha / 2))
    return BootstrapCI(point=point, lower=lower, upper=upper, n_clusters=n_clusters, n_boot=n_boot)


def paired_log_ratio_saving(
    w_p1: Sequence[float],
    w_p0: Sequence[float],
    cluster_ids: Sequence[str],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    side: str = "lower",
) -> BootstrapCI:
    """Work-saving = 1 - exp(mean(log(W_P1/W_P0))), with a one-sided 95% bound.

    ``side="lower"`` answers whether useful saving is established. ``side="upper"`` answers
    whether meaningful headroom can still be excluded. Both use the same paired log-ratio
    estimand; callers can request both with one fixed seed to obtain the pre-registered LCB/UCB.
    Both work vectors must be strictly positive (they are summed service times).
    """
    p1 = np.asarray(w_p1, dtype=float)
    p0 = np.asarray(w_p0, dtype=float)
    if np.any(p1 <= 0) or np.any(p0 <= 0):
        raise ValueError("work values must be strictly positive")
    log_ratio = np.log(p1 / p0)

    def saving(rows_log_ratio: np.ndarray) -> float:
        return 1.0 - float(np.exp(np.mean(rows_log_ratio)))

    return cluster_bootstrap_ci(
        log_ratio, cluster_ids, saving, n_boot=n_boot, seed=seed, alpha=0.05, side=side
    )
