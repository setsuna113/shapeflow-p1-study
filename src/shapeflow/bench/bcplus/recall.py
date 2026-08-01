"""Evidence recall: a docid-set intersection over what a trajectory actually saw.

No judge, no model, no entailment. Recall here is exactly ``|retrieved ∩ relevant| / |relevant|``
over document identifiers, which makes it the one quality-adjacent number in the study that is
mechanically reproducible from a trace and an answer key. That is why it is the co-secondary
endpoint (Freeze-1 §4.1) and why it is computed here rather than by the probe: a judged
"the evidence was preserved" and a counted "the document was in the window" answer different
questions, and only the second one can be recomputed years later from artifacts.

Three states, never two. A query can be judged with relevant documents (recall is a number), be
judged with none (the ratio has no denominator), or be unjudged (there is no answer key entry at
all). Folding either of the last two into 0.0 or 1.0 puts a fabricated value into the mean; both
directions have been seen in published tables and neither is recoverable afterwards.

EVALUATOR-ONLY: this reads the answer key. Nothing on the treatment path may import it -- an
aggregator that could compute its own recall could also optimize for it, which is the oracle
AGENTS.md §2 forbids.

**What calls this, and what does not.** Retriever-level Recall@k over the TREC qrels files: one
query, its top-k window, per-query status, found and missed docids kept so a disputed number is
checkable. That is the quantity the index screen reported when it chose the encoder.

The campaign's ``evidence_recall`` endpoint is *not* this function. ``analysis.attach_recall``
computes it separately, because it is a different measurement: the union of every docid one cell
retrieved across all its queries, against the decrypted record's own evidence and gold sets, with
no k -- the agent decides how many queries to issue, so a per-query mean would reward issuing one
good query and many empty ones. Both are named here so that finding two recall implementations in
one package does not read as one of them being a stale copy of the other.
"""

from __future__ import annotations

import collections.abc
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Sequence

from .qrels import Qrels, assert_evaluator_process

__all__ = [
    "EVALUATOR_ONLY",
    "RecallError",
    "RecallStatus",
    "QueryRecall",
    "RecallReport",
    "PairedRecall",
    "recall_at_k",
    "recall_over_split",
    "paired_recall",
]

#: See :mod:`shapeflow.bench.bcplus.qrels`. Same marker, same firewall.
EVALUATOR_ONLY = True

assert_evaluator_process()


class RecallError(RuntimeError):
    """Recall cannot be computed as specified, and no substitute value exists."""


class RecallStatus(str, Enum):
    SCORED = "SCORED"
    #: The query has rows in the answer key but nothing graded relevant. ``recall`` is None.
    NO_RELEVANT = "NO_RELEVANT"
    #: The query is absent from the answer key entirely. ``recall`` is None.
    NOT_JUDGED = "NOT_JUDGED"


@dataclass(frozen=True)
class QueryRecall:
    """One query's recall, with the found/missed docids kept for audit.

    ``found`` and ``missed`` are what make a disputed number checkable: an aggregate recall that
    moved between two runs is unattributable, a named document that stopped being retrieved is a
    lead.
    """

    query_id: str
    status: RecallStatus
    recall: Optional[float]
    n_relevant: int
    n_found: int
    n_considered: int
    k: Optional[int]
    found: tuple[str, ...]
    missed: tuple[str, ...]

    def value(self) -> float:
        """The recall, or raise. There is no numeric stand-in for an unscorable query."""
        if self.recall is None:
            raise RecallError(
                f"{self.query_id}: status {self.status.value} has no recall value; it must be "
                "reported as such, not folded into the mean"
            )
        return self.recall


def _distinct_in_order(docids: Sequence[str], *, query_id: str) -> list[str]:
    """De-duplicate while preserving first appearance.

    A trajectory that fetched the same page on three turns has seen one document. Counting it
    three times would spend three slots of the top-k window on one document and make recall@k
    depend on how often the agent repeated itself rather than on what it found.
    """
    # A str is a Sequence[str], so passing one docid instead of a list of them type-checks, gets
    # iterated character by character, matches nothing and scores a confident 0.0. That is the
    # worst available outcome: it is in range, it is plausible, and it reads as the treatment
    # arm having retrieved none of the evidence. Bytes are rejected for the same reason.
    if isinstance(docids, (str, bytes, bytearray)):
        raise RecallError(
            f"{query_id}: retrieved is a {type(docids).__name__}, not a sequence of docids. "
            "A single string would be iterated character by character and score 0.0 rather "
            "than fail."
        )
    if not isinstance(docids, collections.abc.Sequence):
        raise RecallError(
            f"{query_id}: retrieved is a {type(docids).__name__}, not a sequence of docids. "
            "The order of what the trajectory saw is part of the measurement, so an unordered "
            "or one-shot iterable cannot stand in for it."
        )
    seen: set[str] = set()
    ordered: list[str] = []
    for i, docid in enumerate(docids):
        if not isinstance(docid, str) or not docid:
            raise RecallError(
                f"{query_id}: retrieved[{i}] is {docid!r}, not a docid. A malformed entry that "
                "matched nothing would read as a miss and lower recall for a parsing bug."
            )
        if docid not in seen:
            seen.add(docid)
            ordered.append(docid)
    return ordered


def recall_at_k(
    *,
    query_id: str,
    retrieved: Sequence[str],
    qrels: Qrels,
    k: Optional[int] = None,
) -> QueryRecall:
    """Recall of ``qrels``' relevant docids within the first ``k`` distinct retrieved docids.

    ``retrieved`` is the ordered list of docids the trajectory actually pulled or published, in
    the order it saw them. ``k=None`` scores the whole set, which is the right call for a
    published-evidence payload: the payload is not a ranking and truncating it would measure the
    renderer's ordering rather than the selection.
    """
    if k is not None and (not isinstance(k, int) or isinstance(k, bool) or k < 1):
        raise RecallError(f"k must be a positive integer or None, got {k!r}")

    ordered = _distinct_in_order(retrieved, query_id=query_id)
    window = ordered if k is None else ordered[:k]

    if not qrels.is_judged(query_id):
        return QueryRecall(
            query_id=query_id, status=RecallStatus.NOT_JUDGED, recall=None, n_relevant=0,
            n_found=0, n_considered=len(window), k=k, found=(), missed=(),
        )
    relevant = qrels.relevant(query_id)
    found = frozenset(window) & relevant
    if not relevant:
        return QueryRecall(
            query_id=query_id, status=RecallStatus.NO_RELEVANT, recall=None, n_relevant=0,
            n_found=0, n_considered=len(window), k=k, found=(), missed=(),
        )
    return QueryRecall(
        query_id=query_id,
        status=RecallStatus.SCORED,
        recall=len(found) / len(relevant),
        n_relevant=len(relevant),
        n_found=len(found),
        n_considered=len(window),
        k=k,
        found=tuple(sorted(found)),
        missed=tuple(sorted(relevant - found)),
    )


@dataclass(frozen=True)
class RecallReport:
    """A split's recall, with everything that was excluded from the mean counted in the open."""

    qrels_kind: str
    qrels_source: str
    qrels_sha256: str
    #: Digest of the judgements, not of the file. Two arms scored against the same key loaded
    #: from differently-ordered copies must pair; two arms scored against different judgements
    #: must not, however alike the filenames look.
    qrels_digest: str
    k: Optional[int]
    per_query: tuple[QueryRecall, ...]
    n_queries: int
    n_scored: int
    n_no_relevant: int
    n_not_judged: int
    macro_recall: Optional[float]
    micro_recall: Optional[float]

    @property
    def reportable(self) -> bool:
        """True when every query in the split contributed a value.

        Not a pass/fail: an unjudged query is a fact about the benchmark, not about the run. It
        just may not be invisible, because a mean over 340 of 380 tasks presented as the split's
        recall is a different quantity than the one that was pre-registered.
        """
        return self.n_queries > 0 and self.n_scored == self.n_queries

    def by_query(self) -> dict[str, QueryRecall]:
        return {q.query_id: q for q in self.per_query}

    def content(self) -> dict:
        return {
            "qrels_kind": self.qrels_kind,
            "qrels_source": self.qrels_source,
            "qrels_sha256": self.qrels_sha256,
            "qrels_digest": self.qrels_digest,
            "k": self.k,
            "n_queries": self.n_queries,
            "n_scored": self.n_scored,
            "n_no_relevant": self.n_no_relevant,
            "n_not_judged": self.n_not_judged,
            "macro_recall": self.macro_recall,
            "micro_recall": self.micro_recall,
            "reportable": self.reportable,
            "unscored_query_ids": sorted(
                q.query_id for q in self.per_query if q.status is not RecallStatus.SCORED
            ),
        }


def recall_over_split(
    *,
    query_ids: Sequence[str],
    retrieved_by_query: Mapping[str, Sequence[str]],
    qrels: Qrels,
    k: Optional[int] = None,
) -> RecallReport:
    """Aggregate recall over exactly ``query_ids``.

    Every query in the split must have an entry in ``retrieved_by_query``, even if the entry is
    an empty list. "The trajectory retrieved nothing" is a score of zero; "the run never produced
    a record for this task" is missing data, and a mapping lookup cannot tell them apart unless
    the caller is forced to say which one it means.
    """
    ids = list(query_ids)
    if len(set(ids)) != len(ids):
        raise RecallError("duplicate query ids in the split; one task would be counted twice")
    missing = [q for q in ids if q not in retrieved_by_query]
    if missing:
        raise RecallError(
            f"{len(missing)} split quer(y|ies) have no retrieval record, "
            f"e.g. {sorted(missing)[:5]}. "
            "Pass an explicit empty list for a task that retrieved nothing; a missing key is "
            "missing data and must not average as a zero."
        )

    per_query = tuple(
        recall_at_k(query_id=q, retrieved=retrieved_by_query[q], qrels=qrels, k=k)
        for q in sorted(ids)
    )
    scored = [r for r in per_query if r.status is RecallStatus.SCORED]
    macro = sum(r.value() for r in scored) / len(scored) if scored else None
    denominator = sum(r.n_relevant for r in scored)
    micro = (sum(r.n_found for r in scored) / denominator) if denominator else None
    return RecallReport(
        qrels_kind=qrels.kind.value,
        qrels_source=qrels.source_name,
        qrels_sha256=qrels.source_sha256,
        qrels_digest=qrels.digest,
        k=k,
        per_query=per_query,
        n_queries=len(per_query),
        n_scored=len(scored),
        n_no_relevant=sum(1 for r in per_query if r.status is RecallStatus.NO_RELEVANT),
        n_not_judged=sum(1 for r in per_query if r.status is RecallStatus.NOT_JUDGED),
        macro_recall=macro,
        micro_recall=micro,
    )


@dataclass(frozen=True)
class PairedRecall:
    """Recall reported the way §4.9 requires quality to be reported: mean difference *and*
    incident rate.

    The incident here is a strict per-task regression -- the treatment arm recalled less evidence
    than the baseline did on the same task. It needs no threshold, which is deliberate: an
    absolute recall floor is a pre-registered value (§4.4) and inventing one in a metrics module
    would make it indistinguishable afterwards from a number a person decided in advance.
    """

    n_pairs: int
    n_comparable: int
    n_excluded: int
    baseline_macro: Optional[float]
    treatment_macro: Optional[float]
    mean_difference: Optional[float]
    n_incidents: int
    incident_rate: Optional[float]
    n_improvements: int
    improvement_rate: Optional[float]
    incident_query_ids: tuple[str, ...]
    excluded_query_ids: tuple[str, ...]

    @property
    def reportable(self) -> bool:
        return self.n_pairs > 0 and self.n_excluded == 0

    def content(self) -> dict:
        return {
            "n_pairs": self.n_pairs,
            "n_comparable": self.n_comparable,
            "n_excluded": self.n_excluded,
            "baseline_macro_recall": self.baseline_macro,
            "treatment_macro_recall": self.treatment_macro,
            "mean_difference": self.mean_difference,
            "n_incidents": self.n_incidents,
            "incident_rate": self.incident_rate,
            "n_improvements": self.n_improvements,
            "improvement_rate": self.improvement_rate,
            "incident_query_ids": list(self.incident_query_ids),
            "excluded_query_ids": list(self.excluded_query_ids),
            "reportable": self.reportable,
        }


def paired_recall(baseline: RecallReport, treatment: RecallReport) -> PairedRecall:
    """Pair two arms' recall reports task by task.

    Refuses to intersect. Two arms that scored different task sets are not a paired comparison,
    and quietly taking the overlap drops exactly the tasks where one arm failed to produce a
    trajectory -- which is the population the ITT analysis exists to keep in.
    """
    if baseline.k != treatment.k:
        raise RecallError(
            f"paired arms used different windows (k={baseline.k} vs k={treatment.k}); "
            "recall@5 and recall@20 are different metrics"
        )
    if baseline.qrels_digest != treatment.qrels_digest:
        raise RecallError(
            "paired arms were scored against different answer keys "
            f"({baseline.qrels_digest[:12]} vs {treatment.qrels_digest[:12]})"
        )
    left, right = baseline.by_query(), treatment.by_query()
    if set(left) != set(right):
        only_left = sorted(set(left) - set(right))[:5]
        only_right = sorted(set(right) - set(left))[:5]
        raise RecallError(
            f"paired arms cover different tasks (baseline-only e.g. {only_left}, "
            f"treatment-only e.g. {only_right}); pairing the intersection would silently drop "
            "the tasks one arm failed on"
        )

    ids = sorted(left)
    comparable = [
        q for q in ids
        if left[q].status is RecallStatus.SCORED and right[q].status is RecallStatus.SCORED
    ]
    excluded = tuple(q for q in ids if q not in set(comparable))
    if not comparable:
        return PairedRecall(
            n_pairs=len(ids), n_comparable=0, n_excluded=len(excluded),
            baseline_macro=None, treatment_macro=None, mean_difference=None,
            n_incidents=0, incident_rate=None, n_improvements=0, improvement_rate=None,
            incident_query_ids=(), excluded_query_ids=excluded,
        )
    base_mean = sum(left[q].value() for q in comparable) / len(comparable)
    treat_mean = sum(right[q].value() for q in comparable) / len(comparable)
    incidents = tuple(q for q in comparable if right[q].value() < left[q].value())
    improvements = sum(1 for q in comparable if right[q].value() > left[q].value())
    return PairedRecall(
        n_pairs=len(ids),
        n_comparable=len(comparable),
        n_excluded=len(excluded),
        baseline_macro=base_mean,
        treatment_macro=treat_mean,
        mean_difference=treat_mean - base_mean,
        n_incidents=len(incidents),
        incident_rate=len(incidents) / len(comparable),
        n_improvements=improvements,
        improvement_rate=improvements / len(comparable),
        incident_query_ids=incidents,
        excluded_query_ids=excluded,
    )
