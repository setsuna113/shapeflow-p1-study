"""Turning a finished BrowseComp-Plus campaign into the paired numbers the verdict rests on.

Evaluator-only. This module reads the gold answers, the evidence sets and the qrels; nothing on
the treatment path may import it, and the static firewall enforces that by keying on
:data:`EVALUATOR_ONLY` below.

What it computes, and why each one is here rather than a proxy for it:

- **Accuracy** -- the benchmark's own grader over the final report. BrowseComp-Plus answers are
  short factual strings and the grader is the one the published baselines are scored against, so
  a number produced here is comparable to a number in the paper.
- **Evidence recall** -- ``|retrieved ∩ evidence_docs| / |evidence_docs|`` over the union of
  every query the agent issued in a cell. A docid-set intersection, needing no judge, and the
  only measure here that is immune to a grader's opinion. It answers the question accuracy
  cannot: did the agent's *search behaviour* change, as opposed to its writing.
- **Work** -- ``interval_union_seconds``, the pre-registered primary endpoint: the wall-clock
  union of the cell's vLLM request intervals, which is GPU-busy time and not a token proxy. The
  token components are co-primary and reported *split*, never summed, because P1 buys a shorter
  decode with a longer prefill and one combined number would hide the entire trade.
- **Publication and fallback** -- how often the treatment actually fired. An arm that fell back
  to P0 on every batch has P0's numbers and P1's label, and the difference is invisible in every
  metric above.

The contrasts are paired within task and computed on the tasks where *both* arms reached a
terminal committed state, because a P1 arm that failed on the hard tasks would otherwise show a
quality gain that is entirely survivorship.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from ...canonical import canonical_json
from ...hashing import sha256_hex

__all__ = [
    "EVALUATOR_ONLY",
    "CellRecord",
    "ArmSummary",
    "PairedContrast",
    "AnalysisError",
    "load_cells",
    "attach_recall",
    "attach_grades",
    "summarize_arm",
    "paired_contrast",
    "build_report",
]

#: Read by ``tools/ci/check_leakage_firewall.py``. See the module docstring.
EVALUATOR_ONLY = True

#: Fixed so a re-run of the analysis reproduces the same intervals. A confidence interval that
#: moves between two runs of the same analysis over the same data is not evidence about the data.
BOOTSTRAP_SEED = 20260731
BOOTSTRAP_RESAMPLES = 10_000


class AnalysisError(RuntimeError):
    """The campaign's artifacts cannot be analysed as what they claim to be."""


@dataclass
class CellRecord:
    """One committed cell, reduced to the fields the verdict uses."""

    task_id: str
    arm_id: str
    variant_id: str
    replicate_id: str
    seed: int
    state: str
    final_report: str
    error: str
    counts: dict
    work: dict
    e2e_latency_seconds: float
    energy_joules: Optional[float]
    retrieval_trace: list
    pages_registered: int
    output_ref: str

    #: Filled by :func:`attach_recall` / :func:`attach_grades`.
    evidence_recall: Optional[float] = None
    gold_recall: Optional[float] = None
    retrieved_docids: frozenset = field(default_factory=frozenset)
    correct: Optional[bool] = None
    grade_outcome: str = ""

    @property
    def committed(self) -> bool:
        return self.state == "COMMITTED"

    @property
    def interval_union_seconds(self) -> Optional[float]:
        value = self.work.get("interval_union_seconds")
        return float(value) if isinstance(value, (int, float)) else None

    @property
    def prompt_tokens(self) -> Optional[int]:
        return _token(self.work, "prompt_tokens")

    @property
    def completion_tokens(self) -> Optional[int]:
        return _token(self.work, "completion_tokens")

    @property
    def cached_prompt_tokens(self) -> Optional[int]:
        return _token(self.work, "cached_prompt_tokens")

    @property
    def published_p1(self) -> int:
        """H batches published as P1, plus C closes reduced by the selector."""
        counts = self.counts or {}
        reduced = int(counts.get("page_batches_reduced", 0) or 0)
        fell_back = int(counts.get("page_fallbacks", 0) or 0)
        return max(0, reduced - fell_back) + int(counts.get("close_reduced", 0) or 0)

    @property
    def p1_opportunities(self) -> int:
        counts = self.counts or {}
        return (int(counts.get("page_batches_reduced", 0) or 0)
                + int(counts.get("close_reduced", 0) or 0)
                + int(counts.get("close_failed", 0) or 0))


def _token(work: Mapping, name: str) -> Optional[int]:
    tokens = work.get("tokens")
    if not isinstance(tokens, Mapping):
        return None
    value = tokens.get(name)
    return int(value) if isinstance(value, (int, float)) else None


# --- reading the campaign ---------------------------------------------------------------


def load_cells(
    *,
    schedule_path: Path,
    ledger,
    store,
    work_key_for: Callable[[Mapping], str],
) -> list[CellRecord]:
    """Every cell of a frozen schedule, with its committed output if it has one.

    Reads the schedule rather than scanning the object store, so a cell that never ran is a
    *missing* record with a state rather than an absence. An analysis that only sees what
    succeeded silently conditions on success.
    """
    schedule = json.loads(Path(schedule_path).read_text(encoding="utf-8"))
    records: list[CellRecord] = []
    for block in schedule["blocks"]:
        for cell in block["cells"]:
            work_key = work_key_for(cell)
            ref = ledger.terminal_ref(work_key) or ledger.committed_ref(work_key)
            item = ledger.get_work_item(work_key)
            state = item.state if item is not None else "MISSING"
            arm = cell["arm"]
            if ref is None:
                records.append(CellRecord(
                    task_id=str(cell["task_id"]), arm_id=str(arm["arm_id"]),
                    variant_id=f"{arm['page_variant']}+{arm['close_variant']}",
                    replicate_id=str(cell["replicate_id"]), seed=int(cell["seed"]),
                    state=state, final_report="", error="no artifact", counts={}, work={},
                    e2e_latency_seconds=float("nan"), energy_joules=None, retrieval_trace=[],
                    pages_registered=0, output_ref=""))
                continue
            body = json.loads(store.get_bytes(ref).decode("utf-8"))
            records.append(CellRecord(
                task_id=str(body["cell"]["task_id"]),
                arm_id=str(body["cell"]["arm"]["arm_id"]),
                variant_id=str(body.get("variant_id", "")),
                replicate_id=str(body["cell"]["replicate_id"]),
                seed=int(body["cell"]["seed"]),
                state=state,
                final_report=str(body.get("final_report", "") or ""),
                error=str(body.get("error") or ""),
                counts=dict(body.get("counts") or {}),
                work=dict(body.get("work_summary") or {}),
                e2e_latency_seconds=float(body.get("e2e_latency_seconds") or 0.0),
                energy_joules=body.get("work_summary", {}).get("energy_joules"),
                retrieval_trace=list(body.get("retrieval_trace") or []),
                pages_registered=int(body.get("pages_registered") or 0),
                output_ref=str(ref),
            ))
    return records


# --- the two quality measures -----------------------------------------------------------


def attach_recall(records: Sequence[CellRecord], evaluator_queries) -> None:
    """Set evidence and gold recall on every record that retrieved anything.

    Agent-level: the union of every docid the cell retrieved across all its queries, against the
    query's own evidence and gold sets. Not per-query, because the agent is free to issue as many
    queries as it likes and a per-query mean would reward issuing one good query and many empty
    ones. What the downstream reasoning can use is what the agent saw *at all*.
    """
    for record in records:
        docids: set[str] = set()
        for entry in record.retrieval_trace:
            docids.update(str(d) for d in (entry.get("docids") or ()))
        record.retrieved_docids = frozenset(docids)
        if record.task_id not in evaluator_queries:
            continue
        view = evaluator_queries.get(record.task_id)
        record.evidence_recall = _recall(docids, view.evidence_docids)
        record.gold_recall = _recall(docids, view.gold_docids)


def _recall(retrieved: set, relevant) -> Optional[float]:
    relevant = frozenset(str(d) for d in relevant)
    if not relevant:
        # A query with no labelled relevant document has no recall -- reporting 0.0 would drag
        # the mean down with a number that means "unjudged", and 1.0 would inflate it.
        return None
    return len(retrieved & relevant) / len(relevant)


def attach_grades(records: Sequence[CellRecord], evaluator_queries, grader,
                  *, on_progress: Optional[Callable[[int, int], None]] = None) -> dict:
    """Grade every committed cell's final report against the gold answer.

    An empty report goes through the grader too, which settles it as an ITT miss without a judge
    call. Dropping it instead would let a fragile arm buy accuracy by failing: the tasks it could
    not finish would leave the denominator along with the tasks it got wrong.

    A grade the judge could not produce is left as ``None`` -- reported as unavailable, never
    counted as wrong. A judge outage that scored as zero would present as a quality regression in
    whichever arm happened to be graded during it.
    """
    gradeable = [r for r in records if r.committed and r.task_id in evaluator_queries]
    errors: list[dict] = []
    for index, record in enumerate(gradeable, start=1):
        view = evaluator_queries.get(record.task_id)
        try:
            grade = grader.grade(question=view.query,
                                 prediction=record.final_report,
                                 gold=view.answer)
            record.grade_outcome = str(getattr(grade.outcome, "value", grade.outcome))
            record.correct = bool(grade.is_correct)
        except Exception as exc:  # noqa: BLE001 - an ungraded cell is reported, never guessed
            record.correct = None
            record.grade_outcome = record.grade_outcome or "JUDGE_UNAVAILABLE"
            errors.append({"task_id": record.task_id, "arm_id": record.arm_id,
                           "error": f"{type(exc).__name__}: {exc}"[:300]})
        if on_progress is not None:
            on_progress(index, len(gradeable))
    return {"graded": len(gradeable), "errors": errors,
            "ungraded": sum(1 for r in gradeable if r.correct is None)}


# --- summaries and contrasts ------------------------------------------------------------


@dataclass
class ArmSummary:
    arm_id: str
    variant_id: str
    cells: int
    committed: int
    failed: int
    accuracy: Optional[float]
    accuracy_n: int
    evidence_recall_mean: Optional[float]
    gold_recall_mean: Optional[float]
    recall_n: int
    interval_union_seconds_mean: Optional[float]
    prompt_tokens_mean: Optional[float]
    completion_tokens_mean: Optional[float]
    cached_prompt_tokens_mean: Optional[float]
    e2e_latency_seconds_mean: Optional[float]
    search_queries_mean: Optional[float]
    p1_publications: int
    p1_opportunities: int
    page_fallbacks: int
    close_failures: int

    @property
    def publication_rate(self) -> Optional[float]:
        if not self.p1_opportunities:
            return None
        return self.p1_publications / self.p1_opportunities

    def content(self) -> dict:
        body = {k: v for k, v in self.__dict__.items()}
        body["publication_rate"] = self.publication_rate
        return body


def summarize_arm(records: Sequence[CellRecord]) -> ArmSummary:
    if not records:
        raise AnalysisError("cannot summarize an arm with no cells")
    committed = [r for r in records if r.committed]
    graded = [r for r in committed if r.correct is not None]
    with_recall = [r for r in committed if r.evidence_recall is not None]
    return ArmSummary(
        arm_id=records[0].arm_id,
        variant_id=records[0].variant_id,
        cells=len(records),
        committed=len(committed),
        failed=sum(1 for r in records if not r.committed),
        accuracy=(sum(1 for r in graded if r.correct) / len(graded)) if graded else None,
        accuracy_n=len(graded),
        evidence_recall_mean=_mean([r.evidence_recall for r in with_recall]),
        gold_recall_mean=_mean([r.gold_recall for r in with_recall
                                if r.gold_recall is not None]),
        recall_n=len(with_recall),
        interval_union_seconds_mean=_mean([r.interval_union_seconds for r in committed]),
        prompt_tokens_mean=_mean([r.prompt_tokens for r in committed]),
        completion_tokens_mean=_mean([r.completion_tokens for r in committed]),
        cached_prompt_tokens_mean=_mean([r.cached_prompt_tokens for r in committed]),
        e2e_latency_seconds_mean=_mean([r.e2e_latency_seconds for r in committed]),
        search_queries_mean=_mean([r.counts.get("search_queries") for r in committed]),
        p1_publications=sum(r.published_p1 for r in committed),
        p1_opportunities=sum(r.p1_opportunities for r in committed),
        page_fallbacks=sum(int(r.counts.get("page_fallbacks", 0) or 0) for r in committed),
        close_failures=sum(int(r.counts.get("close_failed", 0) or 0) for r in committed),
    )


def _mean(values) -> Optional[float]:
    clean = [float(v) for v in values
             if isinstance(v, (int, float)) and not isinstance(v, bool)
             and not math.isnan(float(v))]
    return sum(clean) / len(clean) if clean else None


@dataclass
class PairedContrast:
    """One P1 arm against P0 on the tasks where both arms finished."""

    arm_id: str
    baseline_arm_id: str
    n_pairs: int
    metrics: dict

    def content(self) -> dict:
        return {"arm_id": self.arm_id, "baseline_arm_id": self.baseline_arm_id,
                "n_pairs": self.n_pairs, "metrics": self.metrics}


#: (name, extractor, "higher is better"?). The direction is recorded rather than assumed by the
#: reader, because two of these are costs and three are goods, and the sign of an improvement
#: differs between them.
_METRICS: tuple[tuple[str, Callable[[CellRecord], Optional[float]], bool], ...] = (
    ("accuracy", lambda r: None if r.correct is None else float(r.correct), True),
    ("evidence_recall", lambda r: r.evidence_recall, True),
    ("gold_recall", lambda r: r.gold_recall, True),
    ("interval_union_seconds", lambda r: r.interval_union_seconds, False),
    ("prompt_tokens", lambda r: r.prompt_tokens, False),
    ("completion_tokens", lambda r: r.completion_tokens, False),
    ("cached_prompt_tokens", lambda r: r.cached_prompt_tokens, False),
    ("e2e_latency_seconds", lambda r: r.e2e_latency_seconds, False),
    ("search_queries", lambda r: r.counts.get("search_queries"), True),
)


def paired_contrast(baseline: Sequence[CellRecord], treatment: Sequence[CellRecord],
                    *, resamples: int = BOOTSTRAP_RESAMPLES) -> PairedContrast:
    """Pair by task id, on tasks where both arms committed, and bootstrap each metric."""
    by_task_b = {r.task_id: r for r in baseline if r.committed}
    by_task_t = {r.task_id: r for r in treatment if r.committed}
    shared = sorted(set(by_task_b) & set(by_task_t))

    metrics: dict = {}
    for name, extract, higher_is_better in _METRICS:
        pairs = []
        for task_id in shared:
            b, t = extract(by_task_b[task_id]), extract(by_task_t[task_id])
            if b is None or t is None:
                continue
            b, t = float(b), float(t)
            if math.isnan(b) or math.isnan(t):
                continue
            pairs.append((b, t))
        metrics[name] = _paired_metric(pairs, resamples=resamples,
                                       higher_is_better=higher_is_better)
    return PairedContrast(
        arm_id=(treatment[0].arm_id if treatment else ""),
        baseline_arm_id=(baseline[0].arm_id if baseline else ""),
        n_pairs=len(shared), metrics=metrics)


def _paired_metric(pairs: Sequence[tuple[float, float]], *, resamples: int,
                   higher_is_better: bool) -> dict:
    if not pairs:
        return {"n": 0, "reportable": False,
                "reason": "no task had a value for this metric in both arms"}
    diffs = [t - b for b, t in pairs]
    baseline_mean = sum(b for b, _ in pairs) / len(pairs)
    treatment_mean = sum(t for _, t in pairs) / len(pairs)
    mean_diff = sum(diffs) / len(diffs)
    lo, hi = _bootstrap_ci(diffs, resamples=resamples)
    return {
        "n": len(pairs),
        "reportable": True,
        "higher_is_better": higher_is_better,
        "baseline_mean": baseline_mean,
        "treatment_mean": treatment_mean,
        "mean_paired_difference": mean_diff,
        "ci95_low": lo,
        "ci95_high": hi,
        "relative_change": (mean_diff / baseline_mean) if baseline_mean else None,
        # A confidence interval that excludes zero is the only claim made here. No p-value is
        # reported: the design's inferential machinery lives in the prereg, and a number computed
        # here would be mistaken for it.
        "excludes_zero": (lo > 0.0) or (hi < 0.0),
        "direction": ("better" if (mean_diff > 0) == higher_is_better else "worse")
                     if mean_diff != 0 else "unchanged",
        "wins": sum(1 for d in diffs if (d > 0) == higher_is_better and d != 0),
        "losses": sum(1 for d in diffs if (d < 0) == higher_is_better and d != 0),
        "ties": sum(1 for d in diffs if d == 0),
    }


def _bootstrap_ci(values: Sequence[float], *, resamples: int,
                  seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """Percentile bootstrap over the paired differences, with a deterministic PRNG.

    ``random.Random`` seeded explicitly rather than the module-level generator: an interval whose
    endpoints depend on whatever else in the process happened to draw a random number is not
    reproducible, and reproducibility is the only reason to prefer a bootstrap here over an
    asymptotic interval on a sample this small.
    """
    import random

    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    return (means[int(0.025 * resamples)], means[min(resamples - 1, int(0.975 * resamples))])


# --- the report -------------------------------------------------------------------------


def build_report(records: Sequence[CellRecord], *, baseline_arm: str = "P0",
                 context: Optional[Mapping] = None,
                 resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    """Arm summaries, every P1-vs-P0 contrast, and the provenance to re-derive both."""
    by_arm: dict[str, list[CellRecord]] = {}
    for record in records:
        by_arm.setdefault(record.arm_id, []).append(record)
    if baseline_arm not in by_arm:
        raise AnalysisError(
            f"no {baseline_arm!r} cells in this campaign; every contrast here is against it, and "
            "an arm compared with nothing is not a result")

    summaries = {arm: summarize_arm(cells).content() for arm, cells in sorted(by_arm.items())}
    contrasts = {
        arm: paired_contrast(by_arm[baseline_arm], cells, resamples=resamples).content()
        for arm, cells in sorted(by_arm.items()) if arm != baseline_arm
    }
    body = {
        "schema": "bcplus_analysis_v1",
        "baseline_arm": baseline_arm,
        "arms": summaries,
        "contrasts": contrasts,
        "cells_total": len(records),
        "cells_committed": sum(1 for r in records if r.committed),
        "tasks": sorted({r.task_id for r in records}),
        "bootstrap": {"seed": BOOTSTRAP_SEED, "resamples": resamples},
        "context": dict(context or {}),
        # Every cell's stored artifact, so a reader can re-derive the whole table rather than
        # believe it.
        "source_refs": sorted({r.output_ref for r in records if r.output_ref}),
    }
    body["content_sha256"] = sha256_hex(canonical_json(body))
    return body
