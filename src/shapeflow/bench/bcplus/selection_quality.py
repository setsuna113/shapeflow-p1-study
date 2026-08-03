"""What a reducer kept, scored against the benchmark's own labels. No judge, no model.

Freeze-2 dropped binary short-answer accuracy as a primary endpoint: P0 scored 12.2% over the
Freeze-1 campaign, and the workload never cleared the evidence-recall floor its own
pre-registration set. A binary outcome at that rate cannot resolve between-arm differences. So
the primary quality family is these four, all of which are counted rather than judged and all of
which have far more resolution than a 12% binary:

1. **evidence-doc retention** -- of the relevant documents *offered in this batch*, how many
   survive into the publication;
2. **source coverage** -- distinct sources published over distinct sources offered;
3. **hard-negative interference**, and **negative displacement**, its sharper within-batch form;
4. **answer-string retention** -- whether the gold answer survives the compression at all.

**The denominator is what makes these mean anything.** Retention is measured against what the
reducer *was handed*, not against what the benchmark holds. The retriever's own recall was
0.1488 against a 0.40 floor, so a denominator of "all relevant documents" would score the
retriever and report it as the reducer's quality. The whole point is to isolate the boundary
under study from the retrieval that fed it.

**Three states, never two.** A batch that was offered no relevant document has no retention --
it is not retention of zero. Folding that into 0.0 puts a fabricated value into the mean, and the
fabrication is not recoverable afterwards. This follows `recall.RecallStatus`.

EVALUATOR-ONLY: this reads the answer key. The trial records it consumes are produced on the
treatment side and are keyed by span id, occurrence id and docid -- all of which the agent itself
handled. The labels are joined here, on this side of the firewall, which is precisely what makes
it legitimate to choose a selector on these numbers: no arm could have seen them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Optional, Sequence

from .qrels import EvaluatorQuerySet, assert_evaluator_process

__all__ = [
    "EVALUATOR_ONLY",
    "SelectionQualityError",
    "RetentionStatus",
    "BatchQuality",
    "ArmQuality",
    "score_batch",
    "score_arm",
]

#: See :mod:`shapeflow.bench.bcplus.qrels`. Same marker, same firewall.
EVALUATOR_ONLY = True

assert_evaluator_process()


class SelectionQualityError(RuntimeError):
    """The metric cannot be computed as specified, and no substitute value exists."""


class RetentionStatus(str, Enum):
    SCORED = "SCORED"
    #: The batch offered no document from the relevant set. There is no denominator, so
    #: retention is None -- not zero, which would say the reducer dropped something.
    NONE_OFFERED = "NONE_OFFERED"
    #: The task has no entry in the answer key at all.
    NOT_JUDGED = "NOT_JUDGED"


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    return (numerator / denominator) if denominator else None


@dataclass(frozen=True)
class BatchQuality:
    """One gather batch, one arm: what was offered, what was published, and what that cost."""

    checkpoint_digest: str
    task_id: str
    variant_id: str
    status: RetentionStatus

    sources_offered: int = 0
    sources_published: int = 0

    evidence_offered: int = 0
    evidence_published: int = 0

    negatives_published: int = 0
    docids_published: int = 0
    #: Batches where a hard negative was published while a relevant document offered in the very
    #: same batch was dropped. A within-batch paired event, so it needs no cross-arm
    #: normalisation and cannot be explained by one arm having seen a different world.
    negative_displacement: bool = False

    answer_offered: bool = False
    answer_retained: bool = False

    @property
    def source_coverage(self) -> Optional[float]:
        """No answer key involved, so this one is reportable live and on the treatment side."""
        return _ratio(self.sources_published, self.sources_offered)

    @property
    def evidence_retention(self) -> Optional[float]:
        return _ratio(self.evidence_published, self.evidence_offered)

    @property
    def interference(self) -> Optional[float]:
        return _ratio(self.negatives_published, self.docids_published)


@dataclass(frozen=True)
class ArmQuality:
    """One arm's quality over the batches it was scored on."""

    variant_id: str
    batches: int
    scored: int
    not_judged: int
    none_offered: int
    source_coverage_mean: Optional[float]
    evidence_retention_mean: Optional[float]
    interference_mean: Optional[float]
    negative_displacement_rate: Optional[float]
    answer_retention: Optional[float]
    answer_opportunities: int

    def content(self) -> dict:
        return {
            "variant_id": self.variant_id,
            "batches": self.batches,
            "scored": self.scored,
            "not_judged": self.not_judged,
            "none_offered": self.none_offered,
            "source_coverage_mean": self.source_coverage_mean,
            "evidence_retention_mean": self.evidence_retention_mean,
            "interference_mean": self.interference_mean,
            "negative_displacement_rate": self.negative_displacement_rate,
            "answer_retention": self.answer_retention,
            "answer_opportunities": self.answer_opportunities,
        }


def _normalise(text: str) -> str:
    """Case- and space-insensitive containment, the weakest defensible answer-string test.

    Deliberately weak: a stricter match would start measuring formatting. This is a floor -- if
    the string is not present under this test, the answer did not survive the compression under
    any reading.
    """
    return " ".join(text.lower().split())


def score_batch(
    trial: Mapping,
    *,
    queries: EvaluatorQuerySet,
    published_text: Optional[str] = None,
) -> BatchQuality:
    """Score one trial record against the answer key.

    ``trial`` is the treatment-side record: `page_docids` maps every offered occurrence to its
    document, and each outcome carries the occurrences it published. The join is by identifier
    on both sides; nothing here re-reads a page.
    """
    task_id = str(trial.get("task_id", ""))
    variant_id = str(trial.get("variant_id", ""))
    digest = str(trial.get("checkpoint_digest", ""))

    if task_id not in queries:
        return BatchQuality(digest, task_id, variant_id, RetentionStatus.NOT_JUDGED)
    view = queries.get(task_id)

    page_docids: Mapping[str, str] = trial.get("page_docids") or {}
    offered_occurrences = set(page_docids)
    published_occurrences: set[str] = set()
    for outcome in trial.get("outcomes") or ():
        if outcome.get("ok"):
            published_occurrences.update(
                str(o) for o in outcome.get("published_source_occurrence_ids") or ())

    offered_docids = {page_docids[o] for o in offered_occurrences if page_docids.get(o)}
    published_docids = {page_docids[o] for o in published_occurrences if page_docids.get(o)}

    # Relevant means the benchmark's evidence set. Gold is the smaller hand-asserted subset and
    # is reported separately by the campaign endpoint; evidence is the stricter denominator here
    # because it is what the reducer plausibly had to keep.
    relevant = view.evidence_docids
    offered_relevant = offered_docids & relevant
    published_relevant = published_docids & relevant
    negatives = published_docids & view.negative_docids

    dropped_relevant = offered_relevant - published_relevant
    displacement = bool(negatives) and bool(dropped_relevant)

    answer = _normalise(view.answer)
    answer_offered = bool(answer) and bool(offered_docids)
    answer_retained = bool(
        answer and published_text and answer in _normalise(published_text))

    return BatchQuality(
        checkpoint_digest=digest, task_id=task_id, variant_id=variant_id,
        status=(RetentionStatus.SCORED if offered_relevant
                else RetentionStatus.NONE_OFFERED),
        sources_offered=len(offered_occurrences),
        sources_published=len(published_occurrences),
        evidence_offered=len(offered_relevant),
        evidence_published=len(published_relevant),
        negatives_published=len(negatives),
        docids_published=len(published_docids),
        negative_displacement=displacement,
        answer_offered=answer_offered,
        answer_retained=answer_retained,
    )


def _mean(values: Sequence[float]) -> Optional[float]:
    return (sum(values) / len(values)) if values else None


def score_arm(batches: Iterable[BatchQuality], *, variant_id: str = "") -> ArmQuality:
    """Aggregate one arm. Unscorable batches are counted, never imputed."""
    rows = list(batches)
    if not rows:
        raise SelectionQualityError(
            f"no batches to score for {variant_id!r}; an arm with no measurements has no "
            "quality, which is not the same as low quality")

    scored = [b for b in rows if b.status is RetentionStatus.SCORED]
    coverage = [b.source_coverage for b in rows if b.source_coverage is not None]
    retention = [b.evidence_retention for b in scored if b.evidence_retention is not None]
    interference = [b.interference for b in rows if b.interference is not None]
    answerable = [b for b in rows if b.answer_offered]

    return ArmQuality(
        variant_id=variant_id or (rows[0].variant_id if rows else ""),
        batches=len(rows),
        scored=len(scored),
        not_judged=sum(1 for b in rows if b.status is RetentionStatus.NOT_JUDGED),
        none_offered=sum(1 for b in rows if b.status is RetentionStatus.NONE_OFFERED),
        source_coverage_mean=_mean(coverage),
        evidence_retention_mean=_mean(retention),
        interference_mean=_mean(interference),
        negative_displacement_rate=_ratio(
            sum(1 for b in scored if b.negative_displacement), len(scored)),
        answer_retention=_ratio(
            sum(1 for b in answerable if b.answer_retained), len(answerable)),
        answer_opportunities=len(answerable),
    )
