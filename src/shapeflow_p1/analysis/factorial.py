"""The 2x2 H x C factorial contrasts (plan §15.3).

The four secondary contrasts of the WEBPAGE x C_VISIBLE design, estimated from per-task arm
outcomes. The interaction term is the whole reason for the 2x2: it tests whether H and C combine
additively, so the study never simply sums their separate benefits. All four contrasts form one
family and are corrected together (Holm) in the decision stage; here we only estimate them with
cluster-bootstrap CIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .bootstrap import BootstrapCI, cluster_bootstrap_ci

__all__ = ["ArmOutcomes", "FactorialContrasts", "estimate_contrasts"]


@dataclass(frozen=True)
class ArmOutcomes:
    """One task's outcome under each of the four arms (any scalar endpoint: quality or work)."""

    task_id: str
    cluster_id: str
    p0: float
    h: float
    c: float
    hc: float


@dataclass(frozen=True)
class FactorialContrasts:
    h_simple: BootstrapCI       # H - P0
    c_simple: BootstrapCI       # C - P0
    joint: BootstrapCI          # H+C - P0
    interaction: BootstrapCI    # (H+C - H) - (C - P0)


def estimate_contrasts(
    outcomes: Sequence[ArmOutcomes], *, n_boot: int = 2000, seed: int = 0
) -> FactorialContrasts:
    """Estimate the four paired contrasts with cluster-bootstrap 95% CIs."""
    import numpy as np

    clusters = [o.cluster_id for o in outcomes]
    h = np.array([o.h - o.p0 for o in outcomes])
    c = np.array([o.c - o.p0 for o in outcomes])
    joint = np.array([o.hc - o.p0 for o in outcomes])
    inter = np.array([(o.hc - o.h) - (o.c - o.p0) for o in outcomes])

    def ci(vals):
        return cluster_bootstrap_ci(vals, clusters, np.mean, n_boot=n_boot, seed=seed, side="two")

    return FactorialContrasts(
        h_simple=ci(h), c_simple=ci(c), joint=ci(joint), interaction=ci(inter)
    )
