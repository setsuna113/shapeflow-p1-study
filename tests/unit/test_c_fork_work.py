"""Fork work accounting, and the fabricated saving it exists to prevent.

A fork arm physically issues two model calls, because the upstream ran once in the anchor.
Recording that as the arm's work would say it produced a report having done almost nothing --
and a P1-vs-P0 comparison on that basis shows a spectacular saving that is entirely the
upstream nobody counted. These tests are mostly about refusing to produce that number.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from shapeflow_p1.campaign.c_fork_work import (
    ForkAccountingError,
    SharedUpstream,
    assert_arms_share_one_upstream,
    attribution_for_arm,
    partition_anchor_work,
    total_work_endpoint,
)


@dataclass(frozen=True)
class _Event:
    op_class: str
    call_id: str
    upstream_dispatch_ts: float
    upstream_response_end_ts: float
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def service_seconds(self) -> float:
        return self.upstream_response_end_ts - self.upstream_dispatch_ts


def _anchor_events():
    """One brief, two researcher turns, a page summary, then two children closing."""
    return [
        _Event("SUPERVISOR_CONTINUE", "c0", 0.0, 1.0, 100, 10),
        _Event("RESEARCHER_REACT", "c1", 1.0, 3.0, 200, 20),
        _Event("PAGE_P0_SUMMARY", "c2", 3.0, 6.0, 400, 40),
        _Event("COMPRESSOR_P0", "c3", 6.0, 7.0, 300, 30),      # boundary 0
        _Event("RESEARCHER_REACT", "c4", 7.0, 9.0, 250, 25),
        _Event("COMPRESSOR_P0", "c5", 9.0, 10.5, 320, 32),     # boundary 1
        _Event("FINAL_WRITER", "c6", 10.5, 13.0, 900, 90),
    ]


def _shared(index: int = 0) -> SharedUpstream:
    return partition_anchor_work(
        _anchor_events(), anchor_work_key="anchor-1", boundary_count=2
    )[index]


def _post(service=1.0, prompt=300, completion=30, by_op=None):
    return {
        "service_seconds": service, "prompt_tokens": prompt, "completion_tokens": completion,
        "by_op": by_op or {"COMPRESSOR_P1_SELECTOR": 1, "FINAL_WRITER": 1},
    }


# --- partitioning ---------------------------------------------------------------------------


def test_the_shared_prefix_is_everything_dispatched_before_that_boundary():
    first, second = partition_anchor_work(
        _anchor_events(), anchor_work_key="anchor-1", boundary_count=2)

    assert first.call_ids == ("c0", "c1", "c2")
    assert first.service_seconds == pytest.approx(6.0)
    # The second boundary's prefix includes the first child's own compressor call.
    assert second.call_ids == ("c0", "c1", "c2", "c3", "c4")
    assert second.service_seconds == pytest.approx(9.0)
    assert second.by_op["RESEARCHER_REACT"] == 2


def test_an_anchor_whose_compressors_do_not_match_its_boundaries_is_unusable():
    """Truncating to whichever is shorter would attribute one child's upstream to another."""
    with pytest.raises(ForkAccountingError, match="cannot be attributed"):
        partition_anchor_work(_anchor_events(), anchor_work_key="anchor-1", boundary_count=3)


def test_a_boundary_with_no_preceding_work_is_refused():
    events = [_Event("COMPRESSOR_P0", "c0", 0.0, 1.0, 10, 1)]
    with pytest.raises(ForkAccountingError, match="no upstream work before it"):
        partition_anchor_work(events, anchor_work_key="anchor-1", boundary_count=1)


def test_the_partition_receipt_is_stable_across_reads():
    a = partition_anchor_work(_anchor_events(), anchor_work_key="a", boundary_count=2)[0]
    b = partition_anchor_work(
        list(reversed(_anchor_events())), anchor_work_key="a", boundary_count=2)[0]
    assert a.digest == b.digest, "the partition depends on read order"


# --- the fabricated saving ------------------------------------------------------------------


def test_the_decision_facing_fraction_is_computed_on_total_not_post_boundary():
    """The whole point. Post-boundary alone turns a 5% saving into 90%."""
    shared = _shared()
    p0 = attribution_for_arm(shared, _post(service=10.0, by_op={"COMPRESSOR_P0": 1}))
    p1 = attribution_for_arm(shared, _post(service=1.0))

    post_saving = 1.0 - (
        p1["post_boundary_incremental"]["service_seconds"]
        / p0["post_boundary_incremental"]["service_seconds"]
    )
    total_saving = 1.0 - (
        p1["total_with_shared_upstream"]["service_seconds"]
        / p0["total_with_shared_upstream"]["service_seconds"]
    )

    assert post_saving == pytest.approx(0.90)
    assert total_saving == pytest.approx(0.5625)
    # Both are present, and each says which basis it is on, so neither can be quoted as the
    # other by accident.
    assert p1["post_boundary_incremental"]["basis"] == "POST_BOUNDARY_INCREMENTAL"
    assert p1["total_with_shared_upstream"]["basis"] == "TOTAL_WITH_SHARED_UPSTREAM"


def test_every_arm_carries_the_identical_upstream_constant():
    shared = _shared()
    records = [
        attribution_for_arm(shared, _post(service=10.0, by_op={"COMPRESSOR_P0": 1})),
        attribution_for_arm(shared, _post(service=1.0)),
    ]
    assert_arms_share_one_upstream(records)
    constants = {r["upstream_attribution"]["shared_service_seconds"] for r in records}
    assert constants == {6.0}


def test_arms_from_different_boundaries_are_not_comparable():
    records = [attribution_for_arm(_shared(0), _post()),
               attribution_for_arm(_shared(1), _post())]
    with pytest.raises(ForkAccountingError, match="not comparable"):
        assert_arms_share_one_upstream(records)


# --- zero upstream, refused at all three layers ---------------------------------------------


def test_zero_upstream_is_refused_at_write_time():
    empty = SharedUpstream(
        anchor_work_key="a", boundary_ordinal=0, call_ids=(), service_seconds=0.0,
        prompt_tokens=0, completion_tokens=0, by_op={},
    )
    with pytest.raises(ForkAccountingError, match="no shared upstream calls"):
        attribution_for_arm(empty, _post())


def test_a_positive_call_count_with_no_service_time_is_still_refused():
    hollow = SharedUpstream(
        anchor_work_key="a", boundary_ordinal=0, call_ids=("c0",), service_seconds=0.0,
        prompt_tokens=0, completion_tokens=0, by_op={"SUPERVISOR_CONTINUE": 1},
    )
    with pytest.raises(ForkAccountingError, match="non-positive"):
        attribution_for_arm(hollow, _post())


def test_a_record_without_upstream_reads_as_not_estimable_never_as_zero():
    assert total_work_endpoint(None)["status"] == "NOT_ESTIMABLE"
    assert total_work_endpoint({})["status"] == "NOT_ESTIMABLE"

    forged = attribution_for_arm(_shared(), _post())
    del forged["upstream_attribution"]["shared_service_seconds"]
    endpoint = total_work_endpoint(forged)
    assert endpoint["status"] == "NOT_ESTIMABLE"
    assert endpoint["reason"] == "SHARED_UPSTREAM_WORK_MISSING"
    assert "service_seconds" not in endpoint, "a missing upstream produced a number anyway"


def test_a_well_formed_record_reads_as_a_total():
    endpoint = total_work_endpoint(attribution_for_arm(_shared(), _post()))
    assert endpoint["status"] == "OK"
    assert endpoint["basis"] == "TOTAL_WITH_SHARED_UPSTREAM"
    assert endpoint["service_seconds"] == pytest.approx(7.0)


# --- the arm did only close work ------------------------------------------------------------


def test_an_arm_that_issued_upstream_ops_is_refused():
    with pytest.raises(ForkAccountingError, match="outside the close boundary"):
        attribution_for_arm(_shared(), _post(by_op={"RESEARCHER_REACT": 1, "FINAL_WRITER": 1}))
