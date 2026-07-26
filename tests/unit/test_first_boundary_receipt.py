"""Evaluator receipts compare the first candidate input, not post-treatment trajectories."""

from shapeflow_p1.campaign.evaluate import _first_boundary_input_receipt
from shapeflow_p1.canonical import canonical_json
from shapeflow_p1.hashing import sha256_hex


def _control_record(*, digest: str = "a" * 64, span: str = "s1") -> dict:
    return {
        "node": "H",
        "checkpoint_hash": "c" * 64,
        "candidate_view_sha256": digest,
        "offered_span_ids": [span],
        "offered_source_occurrence_ids": ["occ-1"],
    }


def test_structured_and_prose_events_freeze_the_same_first_boundary_identity() -> None:
    structured = _first_boundary_input_receipt(
        [
            {"kind": "SEARCH_QUERY", "event_index": 0},
            {
                "kind": "NODE_SELECTION",
                "event_index": 1,
                "direct_node_record": _control_record(),
            },
            {
                "kind": "NODE_SELECTION",
                "event_index": 2,
                "direct_node_record": _control_record(digest="b" * 64, span="s2"),
            },
        ]
    )
    prose = _first_boundary_input_receipt(
        [
            {
                "kind": "PROSE_CONTROL_OUTPUT",
                "event_index": 3,
                "node": "H",
                "checkpoint": "p" * 64,
                "control_records": [
                    _control_record(),
                    _control_record(digest="b" * 64, span="s2"),
                ],
            }
        ]
    )

    for field in ("node", "candidate_input_count", "candidate_inputs"):
        assert structured[field] == prose[field]
    assert structured["event_kind"] != prose["event_kind"]
    assert structured["candidate_input_count"] == 2
    assert structured["candidate_inputs"][1]["candidate_view_sha256"] == "b" * 64
    assert structured["content_sha256"] == sha256_hex(
        canonical_json(
            {
                key: value
                for key, value in structured.items()
                if key != "content_sha256"
            }
        )
    )


def test_malformed_first_eligible_view_does_not_fall_through_to_a_later_view() -> None:
    receipt = _first_boundary_input_receipt(
        [
            {
                "kind": "NODE_SELECTION",
                "event_index": 1,
                "direct_node_record": {
                    **_control_record(),
                    "candidate_view_sha256": "not-a-digest",
                },
            },
            {
                "kind": "NODE_SELECTION",
                "event_index": 2,
                "direct_node_record": _control_record(),
            },
        ]
    )

    assert receipt["status"] == "MALFORMED"
    assert receipt["event_index"] == 1


def test_leading_empty_or_snippet_group_is_skipped_before_complete_atomic_batch() -> None:
    receipt = _first_boundary_input_receipt(
        [
            {
                "kind": "NODE_SELECTION",
                "event_index": 1,
                "checkpoint": "0" * 64,
                "direct_node_record": {
                    "node": "H",
                    "checkpoint_hash": "0" * 64,
                    "candidate_view_sha256": "",
                    "offered_span_ids": [],
                    "offered_source_occurrence_ids": [],
                    "selector_attempted": False,
                },
            },
            {
                "kind": "NODE_SELECTION",
                "event_index": 2,
                "direct_node_record": _control_record(),
            },
            {
                "kind": "NODE_SELECTION",
                "event_index": 3,
                "direct_node_record": _control_record(digest="b" * 64, span="s2"),
            },
            {
                "kind": "NODE_SELECTION",
                "event_index": 4,
                "direct_node_record": {
                    **_control_record(digest="d" * 64, span="later"),
                    "checkpoint_hash": "d" * 64,
                },
            },
        ]
    )

    assert receipt["status"] == "OK"
    assert receipt["first_event_index"] == 2
    assert receipt["candidate_input_count"] == 2
    assert [item["offered_span_ids"] for item in receipt["candidate_inputs"]] == [
        ["s1"],
        ["s2"],
    ]
