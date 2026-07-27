"""Model-work accounting and the isolated-mode non-overlap invariant."""

from __future__ import annotations

import pytest

from shapeflow_p1.runtime.request_tags import OpClass, is_treatment_work
from shapeflow_p1.runtime.work_accounting import (
    OverlapError,
    RequestEvent,
    assert_non_overlapping,
    isolated_service_work,
    queue_wait_total,
    token_work,
    work_by_op,
)


def _ev(op, dispatch, end, *, ingress=None, pt=0, ct=0, cached=None):
    return RequestEvent(
        op_class=op, task_id="t", arm_id="H",
        upstream_dispatch_ts=dispatch, upstream_response_end_ts=end,
        proxy_ingress_ts=ingress, prompt_tokens=pt, completion_tokens=ct,
        cached_prompt_tokens=cached,
    )


def test_sequential_intervals_pass_and_sum():
    events = [
        _ev(OpClass.RESEARCHER_REACT, 0.0, 1.0),
        _ev(OpClass.COMPRESSOR_P1_SELECTOR, 1.0, 2.5),
    ]
    assert_non_overlapping(events)
    assert isolated_service_work(events) == pytest.approx(2.5)


def test_overlapping_intervals_are_refused():
    events = [
        _ev(OpClass.RESEARCHER_REACT, 0.0, 2.0),
        _ev(OpClass.RESEARCHER_REACT, 1.0, 3.0),  # overlaps the first
    ]
    with pytest.raises(OverlapError):
        assert_non_overlapping(events)
    with pytest.raises(OverlapError):
        isolated_service_work(events)  # verify=True by default


def test_queue_wait_is_separate_from_work():
    # ingress at 0, dispatch at 0.5 (0.5s queued), service 0.5->1.5 (1.0s work).
    events = [_ev(OpClass.RESEARCHER_REACT, 0.5, 1.5, ingress=0.0)]
    assert isolated_service_work(events) == pytest.approx(1.0)
    assert queue_wait_total(events) == pytest.approx(0.5)


def test_judge_ops_excluded_from_treatment_work():
    events = [
        _ev(OpClass.RESEARCHER_REACT, 0.0, 1.0),
        _ev(OpClass.JUDGE_REPORT, 2.0, 5.0),  # 3s of judge time
    ]
    # Judge time is not treatment work, and does not participate in the overlap check.
    assert isolated_service_work(events) == pytest.approx(1.0)
    assert not is_treatment_work(OpClass.JUDGE_REPORT)


def test_token_work_treatment_only():
    events = [
        _ev(OpClass.COMPRESSOR_P1_SELECTOR, 0.0, 1.0, pt=100, ct=20, cached=10),
        _ev(OpClass.JUDGE_TRUTH, 1.0, 2.0, pt=999, ct=999),  # excluded
    ]
    tw = token_work(events)
    assert tw == {"prompt_tokens": 100, "completion_tokens": 20, "cached_prompt_tokens": 10}


def test_work_by_op_breakdown():
    events = [
        _ev(OpClass.RESEARCHER_REACT, 0.0, 1.0, pt=50, ct=10),
        _ev(OpClass.RESEARCHER_REACT, 1.0, 2.0, pt=60, ct=12),
        _ev(OpClass.FINAL_WRITER, 2.0, 4.0, pt=200, ct=300),
    ]
    by = work_by_op(events)
    assert by[OpClass.RESEARCHER_REACT].count == 2
    assert by[OpClass.RESEARCHER_REACT].service_seconds == pytest.approx(2.0)
    assert by[OpClass.RESEARCHER_REACT].prompt_tokens == 110
    assert by[OpClass.FINAL_WRITER].completion_tokens == 300


def test_non_overlap_ignores_judge_intervals():
    # A judge call overlapping a treatment call must not trip the treatment invariant.
    events = [
        _ev(OpClass.RESEARCHER_REACT, 0.0, 1.0),
        _ev(OpClass.JUDGE_ATOMIZE, 0.5, 1.5),  # overlaps in time but is not treatment
    ]
    assert_non_overlapping(events)  # no raise


# --- the op-class registries must not drift apart -----------------------------------------


def test_every_dispatchable_alias_names_an_op_class_the_ledger_knows():
    """Three registries describe op classes; a gap between them loses work silently.

    ``selector_client.ALIAS_BY_OP`` (what the client dispatches), ``OpClass`` (what the ledger
    accepts) and ``configs/week1.yaml:model_aliases`` (what the provider maps an alias back to)
    must agree. They did not: the SHORT_PROSE op classes were dispatched by
    ``strategies/factory.py`` and listed in ALIAS_BY_OP, but were absent from OpClass -- so the
    config pointed their aliases at the *structured selector* classes instead. Nothing raised.
    Prose-control work was simply recorded under the selector's label, collapsing the exact
    distinction H_ID_VS_PROSE and C_ID_VS_PROSE exist to measure.

    ``work_accounting`` turns an unknown op_class into an ``unavailable`` row rather than an
    error, so a future gap would drop that work out of the totals with nothing to notice.
    """
    from pathlib import Path

    import yaml

    from shapeflow_p1.campaign.selector_client import ALIAS_BY_OP

    known = {o.value for o in OpClass}
    assert set(ALIAS_BY_OP) <= known, set(ALIAS_BY_OP) - known

    repo = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((repo / "configs" / "week1.yaml").read_text(encoding="utf-8"))
    aliases = config["model_aliases"]

    unknown = {op for op in aliases.values() if op not in known}
    assert not unknown, f"model_aliases names op classes absent from OpClass: {sorted(unknown)}"

    # The alias the client sends must map back to the op class it meant, or the ledger records
    # one kind of work as another.
    for op_class, alias in ALIAS_BY_OP.items():
        assert alias in aliases, f"{alias!r} is dispatchable but undeclared in model_aliases"
        assert aliases[alias] == op_class, (
            f"alias {alias!r} is dispatched as {op_class} but the provider records it as "
            f"{aliases[alias]}"
        )


def test_short_prose_controls_count_as_treatment_work():
    """Omitted from TREATMENT_OPS, the prose controls' GPU work vanishes from every total."""
    assert is_treatment_work(OpClass.PAGE_P1_SHORT_PROSE)
    assert is_treatment_work(OpClass.COMPRESSOR_SHORT_PROSE)


# --- the metric that does not require the system under test to be serialized ------------------


def test_interval_union_counts_concurrent_time_once():
    """The measure serialization was imposed to obtain, obtained without imposing it.

    Summing per-request intervals is a work figure only when nothing overlaps, and requiring
    that is why the gateway admitted one upstream request at a time -- against a graph that
    summarises a result set with ``asyncio.gather``. Three requests spanning the same two
    seconds are two seconds of engine time, not six.
    """
    from shapeflow_p1.runtime.work_accounting import interval_union_seconds

    events = [
        _ev(OpClass.PAGE_P0_SUMMARY, 0.0, 2.0),
        _ev(OpClass.PAGE_P0_SUMMARY, 0.5, 1.5),
        _ev(OpClass.PAGE_P0_SUMMARY, 1.0, 2.0),
    ]
    assert interval_union_seconds(events) == pytest.approx(2.0)
    assert isolated_service_work(events, verify=False) == pytest.approx(4.0)
    with pytest.raises(OverlapError):
        assert_non_overlapping(events)


def test_interval_union_equals_the_sum_when_nothing_overlaps():
    """The two agree exactly in the serialized layer, so the mechanism arm stays comparable."""
    from shapeflow_p1.runtime.work_accounting import interval_union_seconds

    events = [
        _ev(OpClass.RESEARCHER_REACT, 0.0, 1.0),
        _ev(OpClass.COMPRESSOR_P1_SELECTOR, 2.0, 3.5),
    ]
    assert interval_union_seconds(events) == pytest.approx(
        isolated_service_work(events)
    ) == pytest.approx(2.5)


def test_interval_union_excludes_idle_gaps_that_latency_would_include():
    """Union is engine-busy time, not wall clock: a form is not charged for time it did not use."""
    from shapeflow_p1.runtime.work_accounting import interval_union_seconds

    events = [
        _ev(OpClass.RESEARCHER_REACT, 0.0, 1.0),
        _ev(OpClass.RESEARCHER_REACT, 100.0, 101.0),
    ]
    assert interval_union_seconds(events) == pytest.approx(2.0)


def test_interval_union_ignores_judge_work():
    """Judge cost is API cost, not treatment work, in every metric."""
    from shapeflow_p1.runtime.work_accounting import interval_union_seconds

    judge_op = next(op for op in OpClass if not is_treatment_work(op))
    assert interval_union_seconds([_ev(judge_op, 0.0, 5.0)]) == pytest.approx(0.0)


def test_peak_concurrency_distinguishes_the_two_regimes():
    """A native layer that silently degraded to one-at-a-time reports the same union.

    Without this the two regimes are indistinguishable in the artifacts, and "we ran natively
    concurrent" would be a claim about configuration rather than an observation.
    """
    from shapeflow_p1.runtime.work_accounting import max_concurrent_treatment_requests

    serial = [
        _ev(OpClass.PAGE_P0_SUMMARY, 0.0, 1.0),
        _ev(OpClass.PAGE_P0_SUMMARY, 1.0, 2.0),
        _ev(OpClass.PAGE_P0_SUMMARY, 2.0, 3.0),
    ]
    concurrent = [
        _ev(OpClass.PAGE_P0_SUMMARY, 0.0, 3.0),
        _ev(OpClass.PAGE_P0_SUMMARY, 0.5, 3.0),
        _ev(OpClass.PAGE_P0_SUMMARY, 1.0, 3.0),
    ]
    # Requests that merely touch at a timestamp are consecutive, not simultaneous.
    assert max_concurrent_treatment_requests(serial) == 1
    assert max_concurrent_treatment_requests(concurrent) == 3


def test_summary_reports_the_union_even_when_the_summed_metric_is_void():
    """Overlap voids the sum. It must not void everything else measured about the cell."""
    from shapeflow_p1.runtime.work_accounting import (
        WorkExtraction,
        summarize_work_extraction,
    )

    events = (
        _ev(OpClass.PAGE_P0_SUMMARY, 0.0, 2.0, pt=100, ct=10),
        _ev(OpClass.PAGE_P0_SUMMARY, 0.5, 1.5, pt=200, ct=20),
    )
    summary = summarize_work_extraction(WorkExtraction(events=events), require_isolated=True)
    assert summary["overlap_valid"] is False
    assert summary["service_seconds"] is None
    assert summary["interval_union_seconds"] == pytest.approx(2.0)
    assert summary["max_concurrent_treatment_requests"] == 2
    # Tokens do not depend on scheduling and must survive.
    assert summary["tokens"]["prompt_tokens"] == 300
    assert summary["tokens"]["completion_tokens"] == 30
