"""Retention is measured against what the reducer was handed, not what the benchmark holds.

The retriever's own evidence recall was 0.1488 against a 0.40 floor. A retention denominator of
"all relevant documents" would therefore score the retriever and publish it as the reducer's
quality -- the boundary under study would barely show up. Isolating the two is the whole reason
these metrics exist, so the denominator is the property most of these tests are about.

The second property is three-valued reporting. A batch offered no relevant document has no
retention; scoring it 0.0 says the reducer dropped something, which is a fabricated value that
cannot be recovered from the mean afterwards.
"""

from __future__ import annotations

import pytest

from shapeflow.bench.bcplus.qrels import EvaluatorQuerySet, EvaluatorQueryView
from shapeflow.bench.bcplus.selection_quality import (
    RetentionStatus,
    SelectionQualityError,
    score_arm,
    score_batch,
)


def _queries(**overrides) -> EvaluatorQuerySet:
    view = EvaluatorQueryView(
        query_id="t1", query="q", answer=overrides.pop("answer", "Ada Lovelace"),
        gold_docids=frozenset(overrides.pop("gold", {"d1"})),
        evidence_docids=frozenset(overrides.pop("evidence", {"d1", "d2"})),
        negative_docids=frozenset(overrides.pop("negatives", {"d9"})),
    )
    return EvaluatorQuerySet({"t1": view}, source_name="test", source_sha256="0" * 64)


def _trial(*, offered: dict, published: list, task_id: str = "t1", ok: bool = True) -> dict:
    return {
        "checkpoint_digest": "c" * 64,
        "task_id": task_id,
        "variant_id": "CPU-FULL",
        "page_docids": offered,
        "outcomes": [{"ok": ok, "published_source_occurrence_ids": published}],
    }


def test_retention_is_over_what_the_batch_offered_not_over_the_benchmark():
    """The batch was handed one of the two relevant documents and kept it.

    Against the benchmark's full evidence set that would read as 50%; against what the reducer
    could actually have kept it is 100%. Only the second is a statement about this boundary.
    """
    q = _queries(evidence={"d1", "d2"})
    batch = score_batch(_trial(offered={"o1": "d1", "o2": "d9"}, published=["o1"]), queries=q)

    assert batch.status is RetentionStatus.SCORED
    assert batch.evidence_offered == 1
    assert batch.evidence_published == 1
    assert batch.evidence_retention == 1.0


def test_a_batch_offered_no_relevant_document_has_no_retention():
    """Not retention of zero. Zero would say the reducer dropped something it was given."""
    q = _queries(evidence={"d1"})
    batch = score_batch(_trial(offered={"o1": "d7", "o2": "d9"}, published=["o1"]), queries=q)

    assert batch.status is RetentionStatus.NONE_OFFERED
    assert batch.evidence_retention is None


def test_an_unjudged_task_is_reported_as_such():
    batch = score_batch(_trial(offered={"o1": "d1"}, published=["o1"], task_id="unknown"),
                        queries=_queries())

    assert batch.status is RetentionStatus.NOT_JUDGED
    assert batch.evidence_retention is None


def test_source_coverage_needs_no_answer_key():
    """The metric that says whether a 512-token whole-batch pack collapses onto one page.

    Computable entirely from treatment-side identifiers, so it can be reported live.
    """
    q = _queries()
    batch = score_batch(
        _trial(offered={f"o{i}": f"d{i}" for i in range(9)}, published=["o0", "o1"]), queries=q)

    assert batch.sources_offered == 9
    assert batch.sources_published == 2
    assert batch.source_coverage == pytest.approx(2 / 9)


def test_negative_displacement_is_a_within_batch_paired_event():
    """A hard negative published while a relevant document *from the same batch* was dropped.

    Sharper than an interference rate because it needs no cross-arm normalisation: both the
    negative and the dropped evidence were in front of this reducer at the same moment.
    """
    q = _queries(evidence={"d1"}, negatives={"d9"})

    displaced = score_batch(
        _trial(offered={"o1": "d1", "o9": "d9"}, published=["o9"]), queries=q)
    assert displaced.negative_displacement is True
    assert displaced.interference == 1.0

    # A negative published while nothing relevant was dropped is interference, not displacement.
    both = score_batch(
        _trial(offered={"o1": "d1", "o9": "d9"}, published=["o1", "o9"]), queries=q)
    assert both.negative_displacement is False
    assert both.interference == pytest.approx(0.5)


def test_a_failed_batch_publishes_nothing_and_retains_nothing():
    """A fallback to P0 is not a publication, so it must not be credited with retention."""
    q = _queries(evidence={"d1"})
    batch = score_batch(
        _trial(offered={"o1": "d1"}, published=["o1"], ok=False), queries=q)

    assert batch.sources_published == 0
    assert batch.evidence_published == 0
    assert batch.evidence_retention == 0.0, (
        "the batch was offered a relevant document and published none of it")


def test_answer_retention_is_measured_only_where_the_answer_could_have_survived():
    q = _queries(answer="Ada Lovelace")
    kept = score_batch(_trial(offered={"o1": "d1"}, published=["o1"]), queries=q,
                       published_text="...  ada   LOVELACE  wrote ...")
    lost = score_batch(_trial(offered={"o1": "d1"}, published=["o1"]), queries=q,
                       published_text="nothing relevant here")

    assert kept.answer_retained is True
    assert lost.answer_retained is False
    assert kept.answer_offered and lost.answer_offered


def test_the_arm_aggregate_counts_unscorable_batches_instead_of_imputing_them():
    q = _queries(evidence={"d1"})
    rows = [
        score_batch(_trial(offered={"o1": "d1"}, published=["o1"]), queries=q),
        score_batch(_trial(offered={"o7": "d7"}, published=["o7"]), queries=q),
        score_batch(_trial(offered={"o1": "d1"}, published=[], task_id="nope"), queries=q),
    ]

    arm = score_arm(rows, variant_id="CPU-FULL")

    assert (arm.batches, arm.scored, arm.none_offered, arm.not_judged) == (3, 1, 1, 1)
    assert arm.evidence_retention_mean == 1.0, "the mean must be over the scorable batch only"


def test_an_arm_with_no_batches_is_refused_rather_than_scored_zero():
    with pytest.raises(SelectionQualityError, match="no quality"):
        score_arm([], variant_id="EMPTY")
