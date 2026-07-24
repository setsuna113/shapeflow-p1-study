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
