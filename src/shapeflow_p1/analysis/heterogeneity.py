"""A small, constrained heterogeneity learner for "when does P1 help?" (plan §15.4).

The learner is deliberately generic about its sampling unit.  The current E2E producer supplies
exactly one pre-treatment row per task and reports task coverage only; that formative result is
not an online node-invocation policy.  Any future invocation-level use would require boundary
randomization and a separate frozen design.  The learner is constrained to prevent subgroup
fishing:

- **Depth <= 2 CART.** A shallow, readable rule, not a high-variance deep tree.
- **Leaves counted in independent tasks, not boundaries.** A leaf must rest on at least N distinct
  tasks; boundaries within one task are correlated and must not be counted as independent support.
- **Stability gated.** If the chosen split is not reproduced in >= 70% of source/topic-cluster
  bootstraps, no CONDITIONAL claim may be made -- an unstable rule is not a rule.

The legacy coverage helper can summarize repeated rows and distinct tasks.  Callers must label the
sampling unit honestly; the E2E eligibility producer does not use its invocation terminology.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from ..hashing import derive_id

__all__ = [
    "Sample",
    "EligibilityRule",
    "fit_eligibility_tree",
    "rule_stability",
    "coverage_estimands",
]


@dataclass(frozen=True)
class Sample:
    task_id: str
    features: dict[str, float]
    training_success: bool
    # The source/topic cluster is the independent resampling unit.  Keeping a default preserves
    # the small standalone CART API: callers that do not have a wider clustering scheme get one
    # cluster per task rather than silently falling back to a row bootstrap.
    cluster_id: str | None = None


@dataclass
class _Node:
    feature: str | None = None
    threshold: float | None = None
    left: _Node | None = None
    right: _Node | None = None
    eligible: bool | None = None
    success_rate: float = 0.0
    n_tasks: int = 0


def _gini(rows: list[Sample]) -> float:
    if not rows:
        return 0.0
    p = sum(1 for r in rows if r.training_success) / len(rows)
    return 1.0 - (p * p + (1 - p) * (1 - p))


def _distinct_tasks(rows: list[Sample]) -> int:
    return len({r.task_id for r in rows})


def _candidate_thresholds(rows: list[Sample], feature: str) -> list[float]:
    vals = sorted({r.features[feature] for r in rows if feature in r.features})
    return [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)]


def _best_split(rows: list[Sample], features: Sequence[str], min_tasks: int):
    best = None
    base = _gini(rows)
    for feature in features:
        for thr in _candidate_thresholds(rows, feature):
            left = [r for r in rows if r.features.get(feature, 0.0) <= thr]
            right = [r for r in rows if r.features.get(feature, 0.0) > thr]
            if _distinct_tasks(left) < min_tasks or _distinct_tasks(right) < min_tasks:
                continue
            w = (len(left) * _gini(left) + len(right) * _gini(right)) / len(rows)
            gain = base - w
            # A zero-gain split manufactures a readable-looking subgroup out of a constant
            # label.  Refuse it rather than letting feature order decide a spurious rule.
            if gain <= 1e-12:
                continue
            if best is None or gain > best[0] + 1e-12:
                best = (gain, feature, thr, left, right)
    return best


def _build(
    rows: list[Sample], features, *, depth, max_depth, min_tasks, eligible_threshold
) -> _Node:
    rate = (sum(1 for r in rows if r.training_success) / len(rows)) if rows else 0.0
    node = _Node(
        eligible=rate >= eligible_threshold, success_rate=rate, n_tasks=_distinct_tasks(rows)
    )
    if depth >= max_depth:
        return node
    split = _best_split(rows, features, min_tasks)
    if split is None:
        return node
    _, feature, thr, left, right = split
    node.feature, node.threshold = feature, thr
    node.left = _build(
        left,
        features,
        depth=depth + 1,
        max_depth=max_depth,
        min_tasks=min_tasks,
        eligible_threshold=eligible_threshold,
    )
    node.right = _build(
        right,
        features,
        depth=depth + 1,
        max_depth=max_depth,
        min_tasks=min_tasks,
        eligible_threshold=eligible_threshold,
    )
    return node


@dataclass
class EligibilityRule:
    root: _Node
    features: tuple[str, ...]
    eligible_threshold: float

    def predict(self, features: dict[str, float]) -> bool:
        node = self.root
        while node.feature is not None:
            go_left = features.get(node.feature, 0.0) <= node.threshold
            node = node.left if go_left else node.right
        return bool(node.eligible)

    @property
    def root_split_feature(self) -> str | None:
        return self.root.feature

    def to_readable(self) -> list[str]:
        lines: list[str] = []

        def walk(node: _Node, prefix: str) -> None:
            if node.feature is None:
                tag = "ELIGIBLE" if node.eligible else "not-eligible"
                lines.append(
                    f"{prefix}=> {tag} (success={node.success_rate:.2f}, tasks={node.n_tasks})"
                )
                return
            walk(node.left, f"{prefix}{node.feature}<={node.threshold:.3g} ")
            walk(node.right, f"{prefix}{node.feature}>{node.threshold:.3g} ")

        walk(self.root, "")
        return lines

    @property
    def rule_hash(self) -> str:
        return derive_id(
            "eligibility_rule",
            {"readable": self.to_readable(), "threshold": self.eligible_threshold},
        )


def fit_eligibility_tree(
    samples: Sequence[Sample],
    features: Sequence[str],
    *,
    max_depth: int = 2,
    min_tasks_per_leaf: int = 8,
    eligible_threshold: float = 0.5,
) -> EligibilityRule:
    root = _build(
        list(samples),
        features,
        depth=0,
        max_depth=max_depth,
        min_tasks=min_tasks_per_leaf,
        eligible_threshold=eligible_threshold,
    )
    return EligibilityRule(
        root=root, features=tuple(features), eligible_threshold=eligible_threshold
    )


def rule_stability(
    samples: Sequence[Sample],
    features: Sequence[str],
    *,
    n_boot: int = 200,
    seed: int = 0,
    min_tasks_per_leaf: int = 8,
    eligible_threshold: float = 0.5,
) -> float:
    """Fraction of *cluster*-bootstrap rules whose root split matches the full-data rule.

    Tasks from one source/topic cluster move together.  Each resampled copy receives fresh
    pseudo task IDs so a cluster drawn twice contributes two independent bootstrap units while
    repeated invocations of the same original task remain correlated.
    """
    import numpy as np

    full = fit_eligibility_tree(
        samples,
        features,
        min_tasks_per_leaf=min_tasks_per_leaf,
        eligible_threshold=eligible_threshold,
    )
    target = full.root_split_feature
    if target is None:
        return 0.0
    by_cluster: dict[str, list[Sample]] = {}
    for s in samples:
        by_cluster.setdefault(s.cluster_id or s.task_id, []).append(s)
    clusters = list(by_cluster)
    if len(clusters) < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    matches = 0
    for _ in range(n_boot):
        drawn = rng.integers(0, len(clusters), size=len(clusters))
        boot: list[Sample] = []
        for draw_index, j in enumerate(drawn):
            cluster = clusters[j]
            pseudo_tasks: dict[str, str] = {}
            for sample in by_cluster[cluster]:
                pseudo_task = pseudo_tasks.setdefault(
                    sample.task_id, f"boot-{draw_index}:{sample.task_id}"
                )
                boot.append(
                    replace(
                        sample,
                        task_id=pseudo_task,
                        cluster_id=f"boot-{draw_index}:{cluster}",
                    )
                )
        rule = fit_eligibility_tree(
            boot,
            features,
            min_tasks_per_leaf=min_tasks_per_leaf,
            eligible_threshold=eligible_threshold,
        )
        if rule.root_split_feature == target:
            matches += 1
    return matches / n_boot


def coverage_estimands(rule: EligibilityRule, samples: Sequence[Sample]) -> dict[str, float]:
    """Invocation coverage and task-exposure coverage over the offered samples."""
    if not samples:
        return {"invocation_coverage": 0.0, "task_exposure_coverage": 0.0}
    eligible = [rule.predict(s.features) for s in samples]
    invocation = sum(eligible) / len(samples)

    by_task: dict[str, bool] = {}
    for s, e in zip(samples, eligible, strict=False):
        by_task[s.task_id] = by_task.get(s.task_id, False) or e
    task_exposure = sum(by_task.values()) / len(by_task)
    return {"invocation_coverage": invocation, "task_exposure_coverage": task_exposure}
