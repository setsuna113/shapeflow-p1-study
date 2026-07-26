"""The candidate view: one object, built from frozen bytes, that nothing else may substitute.

Block 1 closed each injection channel it found individually -- the body supplier, the free-text
context, the unhashed breadcrumb. That was the wrong shape of fix. Every one of them was an
instance of the same thing: *the caller hands the publication path a value*. Close three and the
fourth (the label function, the offer body, the coster) is still open, and each new one has to
be discovered by hand.

So the interface changes rather than the checks. ``CandidateViewRecord`` is constructed once,
from frozen storage, and owns everything the selector sees and everything the publication is
rendered from: candidate bodies, labels, namespaces, context and heading text, query attempts,
the exact prompt bytes, and the versions that shape them. Callers pass the record. They cannot
pass a body, a label, or a coster, because there is no parameter for one.

These tests are the contract for that object, and every one of them is a reproduction of a hole
that survived Block 1.
"""

from __future__ import annotations

import pytest

from shapeflow_p1.evidence.chunkers import WhitespaceTokenizer, paragraph_sentence_v1
from shapeflow_p1.evidence.identity import build_evidence_span, build_visible_message_span
from shapeflow_p1.hashing import sha256_hex
from shapeflow_p1.p1.aggregators import AggregatedEvidence, AggregatedItem, stable_union_v1
from shapeflow_p1.p1.contracts import OVERALL_FACET, SelectionContractError, parse_selection
from shapeflow_p1.p1.preflight import PreflightConfig, preflight
from shapeflow_p1.p1.view import CandidateViewRecord, ViewConstructionError

TOK = WhitespaceTokenizer()
SOURCE = "Cats are feline animals here. Dogs are canine animals here. Birds can surely fly here."
CH = "c" * 64
OCC = "occ1"
VIEW_HASH = "v" * 64
VISIBLE = b"SOURCE 1: Cats are feline. SOURCE 2: Dogs are canine."


def _raw_spans():
    return [
        build_evidence_span(c, SOURCE, content_hash=CH, source_occurrence_ids=[OCC],
                            chunker_version="v1")
        for c in paragraph_sentence_v1(SOURCE, tokenizer=TOK, max_tokens=6)
    ]


def _view(spans=None, **kw):
    spans = spans if spans is not None else _raw_spans()
    kw.setdefault("namespace", "RAW_SOURCE")
    kw.setdefault("snapshot_texts", {CH: SOURCE})
    kw.setdefault("query_attempts", [("q_a", "cats diet")])
    kw.setdefault("token_budget", 10_000)
    kw.setdefault("contract", "P1_ID")
    kw.setdefault("topic", "feline animals")
    return CandidateViewRecord.build(spans=spans, tokenizer=TOK, **kw)


def _pf(view, sel, agg, **kw):
    return preflight(selection=sel, aggregated=agg, view=view,
                     known_occurrence_ids={OCC},
                     config=PreflightConfig(selected_token_budget=10_000), **kw)


# --- the record owns the bytes ---------------------------------------------------------


def test_bodies_come_from_frozen_storage_not_from_the_caller():
    """`Candidate.from_span(span, "INJECTED OFFER BODY")` put arbitrary text in front of the
    model while the candidate digest stayed identical. There is no longer a parameter for it."""
    import inspect

    sig = inspect.signature(CandidateViewRecord.build)
    assert "text" not in sig.parameters and "source_text_for" not in sig.parameters
    view = _view()
    assert view.candidates[0].text == "Cats are feline animals here."


def test_labels_are_allocated_by_the_record_not_supplied():
    """`label_for` was a caller callable. Returning "E1]\\nINJECTED\\n[" injected straight into
    the rendered publication and preflight passed."""
    import inspect

    assert "label_for" not in inspect.signature(preflight).parameters
    view = _view()
    assert [c.label for c in view.candidates] == ["E1", "E2", "E3"]


def test_the_record_renders_and_costs_so_no_caller_can_disagree_with_it():
    """A caller-supplied coster could price something other than what is published."""
    import inspect

    assert "coster" not in inspect.signature(preflight).parameters
    view = _view()
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1"]}, view.candidate_set)
    agg = stable_union_v1(sel, view.registry)
    res = _pf(view, sel, agg)
    assert res.ok
    assert view.cost(agg) == res.rendered.token_count


# --- the offered set, in full ----------------------------------------------------------


def test_a_foreign_namespace_candidate_cannot_be_offered_at_all():
    """C_VISIBLE with a RAW_SOURCE candidate in the offered set passed preflight.

    Only the *published* spans' namespace was checked. But the selector read the whole offered
    set, so a raw-page candidate sitting unselected beside the visible ones already broke the
    compressor-only boundary: the model saw bytes P0's compressor never had. The rejection has
    to happen when the view is built, before any model call.
    """
    raw = _raw_spans()[0]
    visible = build_visible_message_span(
        message_id="m1", message_role="tool", byte_start=10, byte_end=26,
        message_bytes=VISIBLE, kind="TOOL_EVIDENCE",
        visible_compressor_view_hash=VIEW_HASH, source_occurrence_ids=[OCC])
    with pytest.raises(ViewConstructionError, match="namespace"):
        CandidateViewRecord.build(
            spans=[visible, raw], tokenizer=TOK, namespace="VISIBLE_MESSAGE",
            visible_views={VIEW_HASH: VISIBLE}, snapshot_texts={CH: SOURCE},
            query_attempts=[], token_budget=1000, contract="P1_ID", topic="t")


def test_context_must_address_the_same_source_as_its_span():
    """A context_ref pointing at another snapshot is not context, it is imported evidence."""
    spans = _raw_spans()
    other = "d" * 64
    spans[0]["context_refs"] = [{
        "content_hash": other, "char_start": 0, "char_end": 4,
        "text_sha256": sha256_hex("Dogs".encode()),
    }]
    with pytest.raises(ViewConstructionError, match="same source"):
        _view(spans=spans, snapshot_texts={CH: SOURCE, other: "Dogs are canine."})


def test_the_digest_is_taken_from_the_bytes_the_record_resolved():
    """Repointing a ref *before* the digest was computed still published foreign text.

    Comparing a digest to itself proves nothing when both sides come from the same mutable
    registry. The record resolves every candidate's bytes once, and the digest is over what it
    resolved -- so a later mutation of the registry cannot agree with it.
    """
    spans = _raw_spans()
    spans[0]["context_refs"] = [{
        "content_hash": CH, "char_start": 30, "char_end": 58,
        "text_sha256": sha256_hex(SOURCE[30:58].encode()),
    }]
    view = _view(spans=spans)
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1"]}, view.candidate_set)
    agg = stable_union_v1(sel, view.registry)
    # Mutate the registry after the view was built, as a buggy or hostile caller would.
    view.registry[spans[0]["span_id"]]["context_refs"][0]["char_start"] = 0
    view.registry[spans[0]["span_id"]]["context_refs"][0]["char_end"] = 4
    view.registry[spans[0]["span_id"]]["context_refs"][0]["text_sha256"] = sha256_hex(
        SOURCE[0:4].encode())
    res = _pf(view, sel, agg)
    assert not res.ok
    assert any("view" in e for e in res.errors)


def test_the_digest_closes_over_the_versions_it_claims_to_cover():
    """`candidate_view_sha` needed the renderer version passed by hand or a clean publication
    failed. A digest whose inputs the caller must remember is not a protocol."""
    view = _view()
    assert view.prompt_bundle_version
    assert view.renderer_grouping_version
    assert view.prompt_sha256 == sha256_hex(view.prompt_bytes)
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1"]}, view.candidate_set)
    # A clean publication passes with no extra arguments at all.
    assert _pf(view, sel, stable_union_v1(sel, view.registry)).ok


def test_query_attempts_belong_to_the_view():
    view = _view()
    assert view.candidate_set.resolve("Q1", kind="query_attempt") == "q_a"


# --- C_VISIBLE order and evidence/context semantics ------------------------------------


def _visible_candidate_view(
    bodies: list[tuple[str, str, str]],
    *,
    contract: str = "P1_ID",
) -> CandidateViewRecord:
    """Build one exact visible byte stream from (message_id, kind, body) rows."""
    view_bytes = b"\n".join(body.encode("utf-8") for _, _, body in bodies)
    final_hash = sha256_hex(view_bytes)
    rebuilt = []
    cursor = 0
    for message_id, kind, body in bodies:
        if cursor:
            cursor += 1
        encoded = body.encode("utf-8")
        rebuilt.append(build_visible_message_span(
            message_id=message_id,
            message_role=(
                "tool" if kind in {
                    "TOOL_EVIDENCE",
                    "TOOL_UNATTRIBUTED_CONTEXT",
                }
                else "ai" if kind == "MODEL_DERIVED_CONTEXT"
                else "human"
            ),
            byte_start=cursor,
            byte_end=cursor + len(encoded),
            message_bytes=view_bytes,
            kind=kind,
            visible_compressor_view_hash=final_hash,
            source_occurrence_ids=[OCC] if kind == "TOOL_EVIDENCE" else None,
        ))
        cursor += len(encoded)
    return CandidateViewRecord.build(
        spans=rebuilt,
        tokenizer=TOK,
        namespace="VISIBLE_MESSAGE",
        visible_views={final_hash: view_bytes},
        query_attempts=[],
        token_budget=10_000,
        contract=contract,
        topic="feline evidence",
    )


def test_visible_publication_follows_view_offsets_not_message_ids():
    """Random message ids must not reorder the conversation the compressor actually read."""
    view = _visible_candidate_view([
        ("z-first-id", "TOOL_EVIDENCE", "First evidence in the visible stream."),
        ("a-second-id", "TOOL_EVIDENCE", "Second evidence in the visible stream."),
    ])
    sel = parse_selection(
        {"contract": "P1_ID", "selected_ids": ["E2", "E1"]},
        view.candidate_set,
    )
    rendered = view.render(stable_union_v1(sel, view.registry)).text
    assert rendered.index("First evidence") < rendered.index("Second evidence")


def test_visible_prompt_and_publication_keep_context_non_citable():
    view = _visible_candidate_view([
        ("z-tool", "TOOL_EVIDENCE", "A tool returned this source fact."),
        ("a-model", "MODEL_DERIVED_CONTEXT", "The model previously guessed X."),
        ("m-user", "USER_CONTEXT", "The user asked about X."),
    ])
    prompt = view.prompt_bytes.decode("utf-8")
    assert "<TOOL_EVIDENCE role=tool; CITABLE_SOURCE>" in prompt
    assert "<MODEL_DERIVED_CONTEXT role=ai; NON_CITABLE_CONTEXT>" in prompt
    assert "<USER_CONTEXT role=human; NON_CITABLE_CONTEXT>" in prompt

    sel = parse_selection(
        {"contract": "P1_ID", "selected_ids": ["E1", "E2", "E3"]},
        view.candidate_set,
    )
    result = preflight(
        selection=sel,
        aggregated=stable_union_v1(sel, view.registry),
        view=view,
        known_occurrence_ids={OCC},
        config=PreflightConfig(
            selected_token_budget=10_000,
            expected_namespace="VISIBLE_MESSAGE",
            expected_contract="P1_ID",
        ),
    )
    assert result.ok, result.errors
    assert result.rendered.text.count("SOURCE:") == 1
    assert "MODEL-DERIVED CONTEXT (non-citable; not source evidence):" in result.rendered.text
    assert "USER CONTEXT (non-citable; not source evidence):" in result.rendered.text
    assert "[E2]" not in result.rendered.text and "[E3]" not in result.rendered.text
    assert "CONTEXT ITEM (handle E2; non-citable)" in result.rendered.text


def test_typed_context_cannot_be_promoted_to_support():
    view = _visible_candidate_view([
        ("tool", "TOOL_EVIDENCE", "A tool returned a source fact."),
        ("model", "MODEL_DERIVED_CONTEXT", "The model asserted an unsupported conclusion."),
    ], contract="P1_TYPED")
    sel = parse_selection({
        "contract": "P1_TYPED",
        "selections": [
            {"span_id": "E2", "role": "support", "facet_ids": ["answer"]},
        ],
    }, view.candidate_set)
    result = preflight(
        selection=sel,
        aggregated=stable_union_v1(sel, view.registry),
        view=view,
        known_occurrence_ids={OCC},
        config=PreflightConfig(
            selected_token_budget=10_000,
            expected_namespace="VISIBLE_MESSAGE",
            expected_contract="P1_TYPED",
        ),
    )
    assert not result.ok
    assert any("non-citable MODEL_DERIVED_CONTEXT" in error for error in result.errors)


def test_bridge_cannot_cite_model_or_user_context():
    view = _visible_candidate_view([
        ("tool", "TOOL_EVIDENCE", "A tool returned a source fact."),
        ("model", "MODEL_DERIVED_CONTEXT", "The model previously inferred a conclusion."),
    ], contract="P1_BRIDGE")
    sel = parse_selection({
        "contract": "P1_BRIDGE",
        "selections": [
            {"span_id": "E1", "role": "support"},
            {"span_id": "E2", "role": "background"},
        ],
        "bridges": [{
            "text": "These statements connect.",
            "evidence_ids": ["E2"],
        }],
    }, view.candidate_set)
    result = preflight(
        selection=sel,
        aggregated=stable_union_v1(sel, view.registry),
        view=view,
        known_occurrence_ids={OCC},
        config=PreflightConfig(
            selected_token_budget=10_000,
            bridge_token_cap_each=20,
            bridge_token_cap_total=20,
            expected_namespace="VISIBLE_MESSAGE",
            expected_contract="P1_BRIDGE",
        ),
    )
    assert not result.ok
    assert any(
        "cites non-citable tool/model/user context" in error
        for error in result.errors
    )


# --- relations are the single source of truth ------------------------------------------


def test_an_aggregated_item_has_no_annotation_fields_to_disagree_with():
    """AggregatedItem carried relations AND role AND facet_ids.

    Preflight compared role/facet_ids; the renderer preferred relations. Setting them to
    disagree published `(contradict:f1)` for a span the selector marked `support:f1`, with
    preflight green. Two representations of one fact will eventually disagree, so there is one.
    """
    import dataclasses

    fields = {f.name for f in dataclasses.fields(AggregatedItem)}
    assert fields == {"span_id", "relations"}


def test_a_multi_role_span_is_not_reported_as_a_one_sided_contradiction():
    """`item.role` returned None whenever a span played more than one role.

    The contradiction guard read that single value, so a span legitimately supporting one facet
    and contradicting another was scored as a *missing* contradict side -- the check failed on
    exactly the evidence it exists to protect.
    """
    view = _view()
    sel = parse_selection({"contract": "P1_TYPED", "selections": [
        {"span_id": "E1", "role": "support", "facet_ids": ["eff"]},
        {"span_id": "E1", "role": "contradict", "facet_ids": ["saf"]},
        {"span_id": "E2", "role": "support", "facet_ids": ["saf"]},
    ]}, view.candidate_set)
    res = _pf(view, sel, stable_union_v1(sel, view.registry))
    assert res.ok, res.errors


def test_relations_still_catch_a_genuinely_one_sided_contradiction():
    view = _view()
    sel = parse_selection({"contract": "P1_TYPED", "selections": [
        {"span_id": "E1", "role": "support", "facet_ids": ["saf"]},
        {"span_id": "E2", "role": "contradict", "facet_ids": ["saf"]},
    ]}, view.candidate_set)
    ids = [c.span_id for c in view.candidates]
    one_sided = AggregatedEvidence(items=(AggregatedItem(ids[0], (("saf", "support"),)),))
    res = _pf(view, sel, one_sided)
    assert not res.ok
    assert any("one-sided" in e for e in res.errors)


def test_rewritten_relations_are_rejected():
    view = _view()
    sel = parse_selection({"contract": "P1_TYPED", "selections": [
        {"span_id": "E1", "role": "support", "facet_ids": ["f1"]},
    ]}, view.candidate_set)
    ids = [c.span_id for c in view.candidates]
    forged = AggregatedEvidence(items=(AggregatedItem(ids[0], (("f1", "contradict"),)),))
    res = _pf(view, sel, forged)
    assert not res.ok
    assert any("annotation" in e for e in res.errors)


# --- gaps are preserved exactly; bridges drop only with their evidence -------------------


def test_a_gap_may_not_silently_vanish():
    """A dropped gap is a coverage claim the selector made and the output no longer carries."""
    view = _view()
    sel = parse_selection({"contract": "P1_TYPED",
                           "selections": [{"span_id": "E1", "role": "support"}],
                           "gaps": [{"facet_id": "origin", "query_attempt_ids": ["Q1"]}]},
                          view.candidate_set)
    ids = [c.span_id for c in view.candidates]
    without = AggregatedEvidence(
        items=(AggregatedItem(ids[0], ((OVERALL_FACET, "support"),)),), gaps=())
    res = _pf(view, sel, without)
    assert not res.ok
    assert any("gap" in e for e in res.errors)


def test_a_bridge_drops_only_because_its_evidence_did_and_says_so():
    view = _view()
    sel = parse_selection({"contract": "P1_BRIDGE",
                           "selections": [{"span_id": "E1", "role": "support"},
                                          {"span_id": "E2", "role": "support"}],
                           "bridges": [{"text": "Both agree.",
                                        "evidence_ids": ["E1", "E2"]}]}, view.candidate_set)
    ids = [c.span_id for c in view.candidates]
    # Dropping the bridge while both its spans survive is unmotivated removal.
    unmotivated = AggregatedEvidence(
        items=tuple(AggregatedItem(s, ((OVERALL_FACET, "support"),)) for s in ids[:2]),
        bridges=(), dropped_bridges=("Both agree.",),
    )
    res = _pf(view, sel, unmotivated)
    assert not res.ok
    assert any("bridge" in e for e in res.errors)


# --- everything fails closed ------------------------------------------------------------


def test_unbound_visible_source_metadata_is_rejected_before_selector_dispatch():
    """C title/URL metadata must be byte-addressed, not checked by whole-view substring."""
    visible = build_visible_message_span(
        message_id="m1", message_role="tool", byte_start=10, byte_end=26,
        message_bytes=VISIBLE, kind="TOOL_EVIDENCE",
        visible_compressor_view_hash=VIEW_HASH, source_occurrence_ids=[OCC])
    with pytest.raises(ViewConstructionError, match="closed.*byte-range binding"):
        CandidateViewRecord.build(
            spans=[visible], tokenizer=TOK, namespace="VISIBLE_MESSAGE",
            visible_views={VIEW_HASH: VISIBLE}, snapshot_texts={},
            query_attempts=[], token_budget=1000, contract="P1_ID", topic="t",
            source_meta={
                visible["visible_span_id"]: {
                    "title": "NEVER IN THE VIEW",
                    "url": "u",
                }
            },
        )


def test_cancellation_is_never_swallowed():
    """A cooperative cancellation is not a P1 failure to fall back from -- it is the run being
    torn down, and converting it into "P1 failed, use P0" would fabricate a result."""
    import asyncio

    from shapeflow_p1.p1.view import guard_publication

    def boom():
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        guard_publication(boom)
    # An ordinary error becomes a recorded failure.
    def other():
        raise RuntimeError("kaboom")

    errors = guard_publication(other)
    assert errors and "kaboom" in errors[0]


# --- every rejection is measurable --------------------------------------------------------


@pytest.mark.parametrize("raw,why", [
    ({"contract": "P1_ID", "selected_ids": ["NOPE"]}, "schema"),
    ({"contract": "P1_ID", "selected_ids": ["E99"]}, "out-of-set"),
    ({"contract": "NOT_A_CONTRACT"}, "unknown contract"),
])
def test_every_rejection_carries_a_normalization_record(raw, why):
    """Schema and out-of-set rejections returned normalization=None.

    The strict-valid rate needs its denominator: how often output was malformed at all, not
    only the subset that got far enough to be counted.
    """
    view = _view()
    with pytest.raises(SelectionContractError) as excinfo:
        parse_selection(raw, view.candidate_set)
    assert excinfo.value.normalization is not None, why
    assert excinfo.value.normalization.rejected_reason
