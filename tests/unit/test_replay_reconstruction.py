"""Rebuilt page bytes are verified against the checkpoint, never substituted for it.

The replay corpus is only worth anything if the bytes it hands a selector are the bytes the
compared arm actually read. A checkpoint names a page solely by ``raw_content_id`` --
``sha256`` of vendor's already-truncated text -- so the reconstruction has a free, exact check
available, and the whole design rests on it being taken.

The failure it prevents is silent. Hand a selector a differently-truncated copy of the right
document and it still parses, still packs to budget, still publishes, and still produces a
publication rate and a quality score. Nothing downstream can tell that the spans it published
belong to a document the baseline never saw. So these tests are mostly about refusals.
"""

from __future__ import annotations

import pytest

from shapeflow.campaign.replay import (
    HASH_MISMATCH,
    NO_DOCID,
    NO_DOCUMENT,
    SNIPPET_ONLY,
    VERIFIED,
    BatchReconstruction,
    OccurrenceIndex,
    PageReconstructor,
    PageResolution,
    quantiles,
)
from shapeflow.evidence.model_tokenizer import WhitespaceTokenizer
from shapeflow.evidence.shared_view import SharedContentBudget, apply_shared_budget
from shapeflow.hashing import sha256_hex
from shapeflow.retrieval.corpus import CorpusStore, Document

BUDGET = SharedContentBudget(max_chars=200, max_tokens=50)
TOKENIZER = WhitespaceTokenizer()
DOC_TEXT = " ".join(f"word{n}" for n in range(400))


def _content_id(text: str) -> str:
    """The id the checkpoint would have recorded for this document."""
    return sha256_hex(apply_shared_budget(text, BUDGET, TOKENIZER).text.encode("utf-8"))


def _corpus(**docs: str) -> CorpusStore:
    return CorpusStore({
        docid: Document(docid=docid, text=text, url=f"https://{docid}.example")
        for docid, text in docs.items()
    })


def _index(**occurrence_to_docid: str) -> OccurrenceIndex:
    index = OccurrenceIndex()
    index.add_trace([{
        "query": "q",
        "docids": list(occurrence_to_docid.values()),
        "occurrence_ids": list(occurrence_to_docid.keys()),
    }])
    return index


def _reconstructor(corpus: CorpusStore, index: OccurrenceIndex) -> PageReconstructor:
    return PageReconstructor(
        corpus=corpus, budget=BUDGET, tokenizer=TOKENIZER, occurrences=index)


def test_bytes_that_hash_to_the_recorded_id_are_verified():
    corpus = _corpus(d1=DOC_TEXT)
    page = _reconstructor(corpus, _index(o1="d1")).resolve(
        raw_content_id=_content_id(DOC_TEXT), occurrence_id="o1")

    assert page.status == VERIFIED
    assert page.docid == "d1"
    assert page.text == apply_shared_budget(DOC_TEXT, BUDGET, TOKENIZER).text
    assert page.usable


def test_bytes_that_hash_to_something_else_are_refused_not_substituted():
    """The document resolves and the text is *plausible*; only the hash disagrees.

    This is the whole test file in one case. A reconstruction that returned these bytes would
    be handing the selector a real document with the wrong truncation, which is indistinguishable
    downstream from a correct replay.
    """
    corpus = _corpus(d1=DOC_TEXT)
    page = _reconstructor(corpus, _index(o1="d1")).resolve(
        raw_content_id=_content_id("a completely different document"), occurrence_id="o1")

    assert page.status == HASH_MISMATCH
    assert not page.usable
    assert page.text == "", "refused bytes must not be carried forward"
    assert page.docid == "d1", "the docid it tried is kept, so the failure is diagnosable"


def test_an_occurrence_absent_from_every_trace_has_no_docid_to_look_up():
    page = _reconstructor(_corpus(d1=DOC_TEXT), _index(other="d1")).resolve(
        raw_content_id=_content_id(DOC_TEXT), occurrence_id="o1")

    assert page.status == NO_DOCID
    assert not page.usable


def test_a_docid_the_corpus_does_not_hold_is_reported_rather_than_raised():
    """Index and corpus of different vintages. Loud in the census, not fatal mid-walk."""
    page = _reconstructor(_corpus(other=DOC_TEXT), _index(o1="d1")).resolve(
        raw_content_id=_content_id(DOC_TEXT), occurrence_id="o1")

    assert page.status == NO_DOCUMENT
    assert not page.usable


def test_a_page_with_no_raw_content_is_snippet_only_and_needs_no_reconstruction():
    """Vendor falls back to the snippet when a page has no body, and so does every arm."""
    page = _reconstructor(_corpus(), OccurrenceIndex()).resolve(
        raw_content_id=None, occurrence_id="o1")

    assert page.status == SNIPPET_ONLY
    assert page.usable, "a snippet-only page is not a reconstruction failure"


def test_a_batch_is_complete_only_when_every_page_is():
    """All-or-nothing: a batch missing one page is a different batch.

    Offering the survivors would shrink the candidate set relative to the arm this batch is
    compared against, and the shortfall would surface as a quality difference rather than as a
    missing page.
    """
    corpus = _corpus(d1=DOC_TEXT, d2=DOC_TEXT + " tail")
    reconstructor = _reconstructor(corpus, _index(o1="d1"))
    good = reconstructor.resolve(raw_content_id=_content_id(DOC_TEXT), occurrence_id="o1")
    bad = reconstructor.resolve(raw_content_id=_content_id(DOC_TEXT), occurrence_id="missing")

    assert BatchReconstruction("d", "t", 1, (good,)).complete
    assert not BatchReconstruction("d", "t", 1, (good, bad)).complete
    assert BatchReconstruction("d", "t", 1, (good, bad)).status_counts() == {
        VERIFIED: 1, NO_DOCID: 1}


def test_only_verified_pages_reach_the_registry_prefill():
    corpus = _corpus(d1=DOC_TEXT)
    reconstructor = _reconstructor(corpus, _index(o1="d1"))
    good = reconstructor.resolve(raw_content_id=_content_id(DOC_TEXT), occurrence_id="o1")
    snippet = reconstructor.resolve(raw_content_id=None, occurrence_id="o2")

    text_by_id, occurrence_by_id = BatchReconstruction(
        "d", "t", 1, (good, snippet)).registry_prefill()

    assert set(text_by_id) == {good.raw_content_id}
    assert occurrence_by_id == {good.raw_content_id: "o1"}


def test_identical_bytes_under_two_docids_keep_their_own_docids():
    """Caching a *resolution* by content id would report the first document for both.

    A corpus may hold the same text twice, and vendor addresses a page by content, so two
    occurrences can legitimately share a ``raw_content_id`` while naming different documents.
    The docid is what the evaluator joins relevance labels on, so returning the wrong one is a
    misattribution that nothing downstream could detect -- the bytes are right, the spans are
    right, and only the label lookup is wrong.
    """
    corpus = _corpus(d1=DOC_TEXT, d2=DOC_TEXT)
    reconstructor = _reconstructor(corpus, _index(o1="d1", o2="d2"))
    content_id = _content_id(DOC_TEXT)

    first = reconstructor.resolve(raw_content_id=content_id, occurrence_id="o1")
    second = reconstructor.resolve(raw_content_id=content_id, occurrence_id="o2")

    assert first.status == second.status == VERIFIED
    assert first.text == second.text
    assert (first.docid, second.docid) == ("d1", "d2")


def test_a_second_occurrence_of_verified_bytes_is_still_checked():
    """Verification is per page, not per document.

    Once ``d1``'s text is cached, a later occurrence pointing at nothing must still fail. If the
    cache short-circuited on content id, an unresolvable occurrence would inherit a VERIFIED
    status from an unrelated page that happened to share bytes.
    """
    reconstructor = _reconstructor(_corpus(d1=DOC_TEXT), _index(o1="d1"))
    content_id = _content_id(DOC_TEXT)

    assert reconstructor.resolve(
        raw_content_id=content_id, occurrence_id="o1").status == VERIFIED
    assert reconstructor.resolve(
        raw_content_id=content_id, occurrence_id="unknown").status == NO_DOCID


def test_the_occurrence_index_refuses_arrays_it_cannot_pair():
    index = OccurrenceIndex()
    with pytest.raises(ValueError, match="parallel arrays"):
        index.add_trace([{"query": "q", "docids": ["d1", "d2"], "occurrence_ids": ["o1"]}])


def test_the_occurrence_index_refuses_one_occurrence_naming_two_documents():
    """An occurrence id identifies one retrieved document; two means it identifies nothing."""
    index = OccurrenceIndex()
    index.add_trace([{"query": "q", "docids": ["d1"], "occurrence_ids": ["o1"]}])
    with pytest.raises(ValueError, match="resolves to both"):
        index.add_trace([{"query": "q2", "docids": ["d2"], "occurrence_ids": ["o1"]}])


def test_the_occurrence_index_accepts_the_same_pairing_twice():
    """The same page is retrieved by many queries across a campaign; that is not a conflict."""
    index = OccurrenceIndex()
    index.add_trace([{"query": "q", "docids": ["d1"], "occurrence_ids": ["o1"]}])
    index.add_trace([{"query": "q", "docids": ["d1"], "occurrence_ids": ["o1"]}])

    assert index.docid_for("o1") == "d1"
    assert len(index) == 1


def test_quantiles_of_nothing_report_no_observations_rather_than_zero():
    """A distribution over an empty sample is not a distribution centred on zero."""
    assert quantiles([]) == {"n": 0}
    assert quantiles([1, 2, 3, 4])["p50"] == 2


def test_a_checkpoint_present_in_two_lanes_is_walked_once(tmp_path):
    """Counting one gather batch twice inflates every rate the walk feeds, invisibly.

    Lanes partition tasks, so this does not happen in the campaign that has run; it is made
    structural because a resumed or migrated task could break the partition and both copies
    would be perfectly valid checkpoints.
    """
    from shapeflow.campaign.replay import iter_checkpoints

    digest = "a" * 64
    for lane in ("runner-lane0", "runner-lane1"):
        shard = tmp_path / lane / "checkpoints" / digest[:2]
        shard.mkdir(parents=True)
        (shard / f"{digest}.json").write_text(
            '{"kind": "H", "digest": "' + digest + '", "task_id": "t"}', encoding="utf-8")

    roots = sorted(tmp_path.glob("runner*/checkpoints"))
    assert len(roots) == 2
    assert len(list(iter_checkpoints(roots, kind="H"))) == 1


def test_the_trial_key_changes_when_the_prompt_or_renderer_does():
    """Resume is "the file exists", so the key must commit to everything that changes the answer.

    A key that ignored the prompt bundle or the renderer version would let a resumed shootout
    reuse results produced under a different prompt -- silently mixing two treatments inside one
    arm, which is invisible in the output because both halves look like valid trials.
    """
    from shapeflow.campaign.replay import trial_key

    base = dict(execution_binding="a" * 64, checkpoint_digest="b" * 64,
                variant_id="HW02", seed=1, tokenizer_sha256="c" * 64)
    assert trial_key(**base) == trial_key(**base)
    assert trial_key(**base) != trial_key(**{**base, "variant_id": "HW00-CPU"})
    assert trial_key(**base) != trial_key(**{**base, "tokenizer_sha256": "d" * 64})
    assert trial_key(**base) != trial_key(**{**base, "checkpoint_digest": "e" * 64})


@pytest.mark.asyncio
async def test_a_replayed_batch_records_its_failure_as_an_outcome_not_an_exception():
    """A whole-batch P1 failure is the measurement, not an error in taking it.

    `PageSelectionError` is what a rejected batch looks like in production, where it falls the
    batch back to P0. A harness that let it escape would drop exactly the samples whose failure
    rate the mechanical-legality gate is counting, and the surviving rate would look perfect.
    """
    from shapeflow.campaign.replay import run_trial
    from shapeflow.evidence.model_tokenizer import WhitespaceTokenizer
    from shapeflow.odr.checkpoints import (
        FrozenMessage,
        HCheckpoint,
        SamplingEnvelope,
        VendorVisibleResult,
    )
    from shapeflow.odr.checkpoints import to_document
    from shapeflow.strategies.page_h import PageSelectionStrategy, PageStrategyConfig

    page = "## h\n\n" + "alpha beta gamma " * 60
    result = VendorVisibleResult(
        vendor_visible_order=0, url="https://a.example", title="a",
        snippet="s", raw_content_id="cid-a", source_occurrence_id="o1",
    )
    checkpoint = HCheckpoint(
        task_id="t", researcher_id="r", assistant_turn_index=0,
        assistant_message=FrozenMessage(role="ai", content="search"),
        sibling_tool_calls=(), search_result_sets=(("call-1", (result,)),),
        non_search_outputs=(), researcher_state_hash="h",
        sampling=SamplingEnvelope(model="m", temperature=0.0, top_p=1.0, max_tokens=8, seed=1),
    )

    class Refusing:
        async def select(self, *, task_ctx, view):
            raise RuntimeError("selector unavailable")

    strategy = PageSelectionStrategy(
        PageStrategyConfig(variant_id="HW02", chunker="markdown_structure_v1",
                           scope="whole_batch", contract="P1_ID",
                           aggregation="stable_union_v1", token_budget=512),
        selector=Refusing(), tokenizer=WhitespaceTokenizer(),
        raw_text_for=lambda _cid: page, occurrence_for=lambda cid: "o1",
    )
    reconstruction = BatchReconstruction(
        checkpoint_digest=checkpoint.digest, task_id="t", siblings=1,
        pages=(PageResolution(raw_content_id="cid-a", occurrence_id="o1",
                              status=VERIFIED, docid="d1", text=page),),
    )

    record = await run_trial(
        document=to_document(checkpoint), reconstruction=reconstruction, strategy=strategy,
        variant_id="HW02", topic="a topic", token_budget=512)

    assert record["batch_failure"] == "SELECTOR_ERROR"
    assert record["published_text"] is None
    assert record["page_docids"] == {"o1": "d1"}
    assert record["outcomes"], "the failed attempt must still be recorded"
    assert record["outcomes"][0]["ok"] is False
    assert record["outcomes"][0]["selector_attempted"] is True
    assert record["outcomes"][0]["offered"] > 0, (
        "the offered candidate set is the denominator and survives the failure")
