"""The mechanical audit a corpus must survive before it is sealed.

Every check here catches a defect that is invisible once results exist:

- A question no query could answer produces an empty frozen world, and every arm then fails on
  it. That reads as "P1 is unreliable" and is actually "the corpus was unretrievable".
- Two paraphrases of one question counted as two tasks overstate the independent n, which
  narrows every confidence interval the study reports.
- A topic cluster split across SCREEN and POWER_PILOT leaks: the pilot's variance estimate is
  computed on tasks that share sources with the screen it was meant to be independent of.
- A stratum with no tasks makes "where does P1 work" unanswerable for that condition, and the
  eligibility envelope is one of the five questions the study exists to settle.

The audit runs before the Tavily budget is spent and before any P1 output exists. It refuses to
seal rather than reporting a warning, because a warning at this point is a decision nobody will
revisit and every downstream number will silently depend on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .task_registry import STRATA, TaskRegistry, TaskSpec

__all__ = ["AuditFinding", "AuditReport", "audit_tasks", "audit_registry", "AuditFailed"]

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")

# Words that carry no retrieval signal. A query made only of these would match everything, which
# is the same as matching nothing.
_STOPWORDS = frozenset("""
a an and are as at be but by for from has have how in into is it its of on or that the their
this to was were what when where which who why will with about over under between
""".split())


class AuditFailed(RuntimeError):
    """The corpus cannot be sealed as authored."""


@dataclass(frozen=True)
class AuditFinding:
    check: str
    subject: str
    detail: str

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"[{self.check}] {self.subject}: {self.detail}"


@dataclass
class AuditReport:
    findings: list[AuditFinding] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.findings

    def add(self, check: str, subject: str, detail: str) -> None:
        self.findings.append(AuditFinding(check, subject, detail))

    def raise_if_failed(self) -> None:
        if self.findings:
            joined = "\n".join(f"  {f}" for f in self.findings[:40])
            more = "" if len(self.findings) <= 40 else f"\n  ... and {len(self.findings) - 40} more"
            raise AuditFailed(
                f"{len(self.findings)} audit finding(s); the corpus is not sealed:\n"
                f"{joined}{more}"
            )


def _content_words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text) if w.lower() not in _STOPWORDS]


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def audit_tasks(tasks: Sequence[TaskSpec], *, config: dict) -> AuditReport:
    """Audit the authored tasks against ``configs/task_source.yaml``."""
    audit = config["audit"]
    report = AuditReport()
    seen_ids: set[str] = set()

    for task in tasks:
        subject = task.task_id
        length = len(task.question)
        if not (audit["min_question_chars"] <= length <= audit["max_question_chars"]):
            report.add("question_length", subject,
                       f"{length} characters, outside "
                       f"[{audit['min_question_chars']}, {audit['max_question_chars']}]")
        if not (audit["min_required_facets"] <= len(task.required_facets)
                <= audit["max_required_facets"]):
            report.add("facet_count", subject, f"{len(task.required_facets)} facets")
        if len(task.fixed_queries) < audit["min_fixed_queries"]:
            report.add("query_count", subject,
                       f"{len(task.fixed_queries)} fixed queries, minimum "
                       f"{audit['min_fixed_queries']}")
        if task.task_id in seen_ids:
            report.add("duplicate_id", subject, "two tasks share one id")
        seen_ids.add(task.task_id)

        # Retrievability: a query with no content words cannot address a slice of the web.
        for query in task.fixed_queries:
            words = _content_words(query)
            if len(words) < 2:
                report.add("retrievability", subject,
                           f"query {query!r} carries fewer than two content words")
        if not task.required_facets:
            report.add("facets_present", subject, "no required facets")
        # A question whose words appear in none of its queries is not what the queries fetch.
        question_words = set(_content_words(task.question))
        query_words = set()
        for query in task.fixed_queries:
            query_words |= set(_content_words(query))
        if question_words and not (question_words & query_words):
            report.add("query_alignment", subject,
                       "no query shares a content word with the question")

        unknown = [s for s in task.strata if s not in STRATA]
        if unknown:
            report.add("unknown_stratum", subject, f"{unknown}")

    # Near-duplicate detection over the whole corpus. Two paraphrases are one observation.
    threshold = float(audit["max_duplicate_jaccard"])
    words = {t.task_id: _content_words(t.question) for t in tasks}
    for i, left in enumerate(tasks):
        for right in tasks[i + 1:]:
            score = _jaccard(words[left.task_id], words[right.task_id])
            if score >= threshold:
                report.add("near_duplicate", f"{left.task_id}~{right.task_id}",
                           f"question Jaccard {score:.2f} >= {threshold}")

    clusters = {t.topic_cluster for t in tasks}
    if len(clusters) < audit["require_distinct_topic_clusters"]:
        report.add("cluster_count", "<corpus>",
                   f"{len(clusters)} clusters, minimum "
                   f"{audit['require_distinct_topic_clusters']}")

    if audit.get("require_all_strata", True):
        counts = {s: 0 for s in STRATA}
        for task in tasks:
            for stratum in task.strata:
                if stratum in counts:
                    counts[stratum] += 1
        for stratum, minimum in config["strata_min_counts"].items():
            if counts.get(stratum, 0) < minimum:
                report.add("stratum_shortfall", stratum,
                           f"{counts.get(stratum, 0)} tasks, minimum {minimum}")

    report.stats = {
        "tasks": len(tasks),
        "clusters": len(clusters),
        "mean_facets": (sum(len(t.required_facets) for t in tasks) / len(tasks)) if tasks else 0.0,
        "mean_queries": (sum(len(t.fixed_queries) for t in tasks) / len(tasks)) if tasks else 0.0,
    }
    return report


def audit_registry(registry: TaskRegistry, *, config: dict) -> AuditReport:
    """Audit the sealed structure: split isolation, reserve order, and the corpus labels."""
    report = audit_tasks(registry.tasks, config=config)
    audit = config["audit"]

    if audit.get("forbid_cluster_split_across_splits", True):
        by_cluster: dict[str, set[str]] = {}
        for task in registry.tasks:
            by_cluster.setdefault(task.topic_cluster, set()).add(
                registry.split_of[task.task_id])
        for cluster, splits in sorted(by_cluster.items()):
            if len(splits) > 1:
                report.add(
                    "cluster_split", cluster,
                    f"appears in {sorted(splits)}; tasks sharing sources are one observation "
                    "and splitting them leaks between the screen and the pilot",
                )

    expected = config["splits"]
    for split, wanted in expected.items():
        got = len(registry.of_split(split))
        if got != wanted:
            report.add("split_size", split, f"{got} tasks, expected {wanted}")

    reserve_ids = [t.task_id for t in registry.of_split("RESERVE")]
    if sorted(registry.reserve_order) != sorted(reserve_ids):
        report.add("reserve_order", "<registry>",
                   "the frozen reserve order does not enumerate exactly the reserve tasks")

    if registry.corpus_tier != config["source"]["corpus_tier"]:
        report.add("corpus_tier", "<registry>",
                   f"{registry.corpus_tier!r} != {config['source']['corpus_tier']!r}")
    if registry.claim_scope != config["source"]["claim_scope"]:
        report.add("claim_scope", "<registry>",
                   f"{registry.claim_scope!r} != {config['source']['claim_scope']!r}")

    forbidden = str(config["authoring"].get("forbidden_author_model_substring", "")).lower()
    if forbidden:
        for model in (registry.fingerprint.requested_model, registry.fingerprint.returned_model):
            if forbidden and forbidden in (model or "").lower():
                report.add(
                    "author_is_subject", model,
                    "the corpus was authored by the model under test; every arm would inherit "
                    "its bias while the comparison silently absorbed it",
                )
    return report
