"""Evidence IR: chunking exactness, table-header association, span identity, candidate
labels, lineage checks."""

from __future__ import annotations

import pytest

from shapeflow_p1.evidence.chunkers import (
    WhitespaceTokenizer,
    fixed_token_v1,
    markdown_structure_v1,
    paragraph_sentence_v1,
)
from shapeflow_p1.evidence.identity import (
    CandidateSet,
    OutOfSetLabel,
    build_evidence_span,
    build_visible_message_span,
)
from shapeflow_p1.evidence.lineage import lineage_closure_errors, reconstruction_errors

TOK = WhitespaceTokenizer()


def _spans(chunks, source, content_hash="c" * 64, occ=("o1",)):
    return [
        build_evidence_span(
            c, source, content_hash=content_hash, source_occurrence_ids=list(occ),
            chunker_version="test_v1",
        )
        for c in chunks
    ]


# --- exact reconstruction -------------------------------------------------------------


@pytest.mark.parametrize("chunker", [
    lambda t: fixed_token_v1(t, tokenizer=TOK, window=5, overlap=1),
    lambda t: paragraph_sentence_v1(t, tokenizer=TOK, max_tokens=6),
    lambda t: markdown_structure_v1(t, tokenizer=TOK, max_tokens=8),
])
def test_all_chunkers_reconstruct_exactly(chunker):
    source = (
        "# Title\n\nFirst paragraph has several words here. Second sentence follows it.\n\n"
        "- item one\n- item two\n\n> a quote line\n"
    )
    chunks = chunker(source)
    assert chunks
    spans = _spans(chunks, source)
    # The recorded text hash must reproduce from the offsets -- the core IR guarantee.
    assert reconstruction_errors(spans, {"c" * 64: source}) == []


def test_fixed_token_covers_every_token():
    source = "alpha beta gamma delta epsilon zeta eta theta"
    chunks = fixed_token_v1(source, tokenizer=TOK, window=3, overlap=0)
    covered = "".join(source[c.char_start:c.char_end] for c in chunks)
    # Every non-space token appears in the concatenated coverage.
    for word in source.split():
        assert word in covered


def test_fixed_token_overlap_repeats_boundary_tokens():
    source = "one two three four five six"
    no_ov = fixed_token_v1(source, tokenizer=TOK, window=3, overlap=0)
    ov = fixed_token_v1(source, tokenizer=TOK, window=3, overlap=1)
    assert len(ov) >= len(no_ov)


# --- markdown structure ---------------------------------------------------------------


def test_table_row_chunk_carries_header():
    source = (
        "## Data\n\n"
        "| Name | Value |\n"
        "|------|-------|\n"
        "| a | 1 |\n"
        "| b | 2 |\n"
    )
    chunks = markdown_structure_v1(source, tokenizer=TOK, max_tokens=100)
    row_chunks = [c for c in chunks if c.kind == "table_row"]
    assert len(row_chunks) == 2
    for rc in row_chunks:
        # The span itself is the row alone -- see the quadratic-growth test below.
        text = source[rc.char_start:rc.char_end]
        assert text.strip().startswith("|") and "Name" not in text
        # A selected row is never orphaned from the header that names its columns: the header
        # travels as an addressed RANGE into the same source, not as a re-sliced span and not
        # as a free string (which would be an unhashed channel into the prompt).
        assert any("Name" in c and "Value" in c for c in rc.context_texts(source))
        for cs_, ce_ in rc.context_ranges:
            assert 0 <= cs_ <= ce_ <= len(source)


def test_table_rows_do_not_grow_quadratically():
    """Each row chunk must span its own row, not header..row_i.

    Anchoring row i on `header.start` makes total chunk bytes O(n^2) in the number of rows, so
    a 200-row table duplicates the whole table ~200 times through the candidate set. That both
    blows the token budget the study is measuring and biases every chunker comparison.
    """
    rows = "".join(f"| r{i} | {i} |\n" for i in range(40))
    source = "| Name | Value |\n|------|-------|\n" + rows
    chunks = markdown_structure_v1(source, tokenizer=TOK, max_tokens=1000)
    row_chunks = [c for c in chunks if c.kind == "table_row"]
    assert len(row_chunks) == 40
    total = sum(c.char_end - c.char_start for c in row_chunks)
    # Linear in the table, not quadratic. The quadratic version totals ~20x the source.
    assert total <= 2 * len(source), f"row chunks total {total} bytes for a {len(source)}-byte table"
    # And no single row chunk swallows its predecessors.
    assert max(c.char_end - c.char_start for c in row_chunks) < 40


@pytest.mark.parametrize("source,label", [
    ("# " + " ".join(f"w{i}" for i in range(40)) + "\n", "heading"),
    ("```\n" + "\n".join(f"code line {i} here" for i in range(20)) + "\n```\n", "code fence"),
    ("| " + " | ".join(f"c{i}" for i in range(30)) + " |\n"
     + "|" + "---|" * 30 + "\n"
     + "| " + " | ".join(f"v{i}" for i in range(30)) + " |\n", "table"),
])
def test_markdown_chunker_never_exceeds_the_token_cap(source, label):
    """max_tokens is the budget contract; four block kinds bypassed it entirely.

    headings, fenced code, table headers and table rows called _finalize directly instead of
    going through the splitting path, so a single chunk could be many times the cap. Because
    the aggregator budgets in *rendered tokens*, an over-cap chunk silently smuggles text past
    the very budget that defines the P1 treatment.
    """
    cap = 5
    chunks = markdown_structure_v1(source, tokenizer=TOK, max_tokens=cap)
    assert chunks, f"{label}: produced no chunks"
    oversized = [(c.kind, c.token_len) for c in chunks if c.token_len > cap]
    assert not oversized, f"{label}: chunks over the {cap}-token cap: {oversized}"


def test_headings_become_breadcrumb():
    source = "# Top\n\n## Section\n\nbody text under section\n"
    chunks = markdown_structure_v1(source, tokenizer=TOK, max_tokens=100)
    para = next(c for c in chunks if c.kind == "paragraph")
    assert para.heading_path == ("Top", "Section")


def test_code_fence_kept_whole():
    source = "text\n\n```\nline1\nline2\n```\n\nmore\n"
    chunks = markdown_structure_v1(source, tokenizer=TOK, max_tokens=100)
    code = next(c for c in chunks if c.kind == "code")
    assert "line1" in source[code.char_start:code.char_end]
    assert "line2" in source[code.char_start:code.char_end]


# --- span identity --------------------------------------------------------------------


def test_evidence_span_requires_occurrence():
    chunks = fixed_token_v1("some text here", tokenizer=TOK, window=5, overlap=0)
    with pytest.raises(ValueError, match="dangling"):
        build_evidence_span(chunks[0], "some text here", content_hash="c" * 64,
                            source_occurrence_ids=[], chunker_version="v1")


def test_span_id_changes_with_offsets():
    source = "alpha beta gamma delta"
    chunks = fixed_token_v1(source, tokenizer=TOK, window=2, overlap=0)
    spans = _spans(chunks, source)
    assert len(spans) >= 2
    assert spans[0]["span_id"] != spans[1]["span_id"]


def test_visible_message_span_kind_rules():
    msg = b"tool observation bytes here"
    # TOOL_EVIDENCE may carry occurrences.
    ok = build_visible_message_span(
        message_id="m1", message_role="tool", byte_start=0, byte_end=4,
        message_bytes=msg, kind="TOOL_EVIDENCE", visible_compressor_view_hash="v" * 64,
        source_occurrence_ids=["o1"],
    )
    assert ok["namespace"] == "VISIBLE_MESSAGE"
    # MODEL_DERIVED_CONTEXT may NOT -- it can never become a citation.
    with pytest.raises(ValueError, match="only TOOL_EVIDENCE"):
        build_visible_message_span(
            message_id="m1", message_role="ai", byte_start=0, byte_end=4,
            message_bytes=msg, kind="MODEL_DERIVED_CONTEXT", visible_compressor_view_hash="v" * 64,
            source_occurrence_ids=["o1"],
        )


# --- candidate labels -----------------------------------------------------------------


def test_candidate_set_labels_and_rejects_out_of_set():
    cs = CandidateSet.build(["span_a", "span_b"], ["attempt_x"])
    assert cs.resolve("E1", kind="evidence") == "span_a"
    assert cs.resolve("E2", kind="evidence") == "span_b"
    assert cs.resolve("Q1", kind="query_attempt") == "attempt_x"
    assert cs.label_for("span_b") == "E2"
    assert cs.label_for_query("attempt_x") == "Q1"
    with pytest.raises(OutOfSetLabel):
        cs.resolve("E99", kind="evidence")  # never leniently resolved


def test_candidate_labels_are_short():
    cs = CandidateSet.build([f"span_{i}" for i in range(12)])
    # Labels stay compact so they don't inflate prompt tokens.
    assert all(len(lbl) <= 4 for lbl in ["E1", "E9", "E12"])
    assert cs.resolve("E12", kind="evidence") == "span_11"


def test_evidence_and_query_labels_are_separate_namespaces():
    """E and Q are different kinds and must not resolve through one table.

    With a single label->id map, a selector could put `Q1` where a span id belongs (pointing
    "evidence" at a search query) or `E1` where a query attempt belongs (claiming a gap was
    probed by a piece of evidence). Both are silently accepted today and both corrupt the
    selection semantics the study measures.
    """
    cs = CandidateSet.build(["span_a"], ["attempt_x"])
    with pytest.raises(OutOfSetLabel):
        cs.resolve("Q1", kind="evidence")
    with pytest.raises(OutOfSetLabel):
        cs.resolve("E1", kind="query_attempt")


def test_candidate_set_carries_and_enforces_its_span_namespace():
    """C_VISIBLE may only ever be offered VISIBLE_MESSAGE spans.

    Making the namespace a property of the offered set is what turns "the selector must not
    read raw page bytes" from a convention into something structurally unreachable.
    """
    cs = CandidateSet.build(["vspan_a"], namespace="VISIBLE_MESSAGE")
    assert cs.namespace == "VISIBLE_MESSAGE"
    assert cs.resolve("E1", kind="evidence") == "vspan_a"
    # Default stays RAW_SOURCE for the page node.
    assert CandidateSet.build(["span_a"]).namespace == "RAW_SOURCE"


# --- lineage --------------------------------------------------------------------------


def test_lineage_closure_flags_unknown_occurrence():
    chunks = fixed_token_v1("hello world here", tokenizer=TOK, window=5, overlap=0)
    spans = _spans(chunks, "hello world here", occ=("real_occ",))
    assert lineage_closure_errors(spans, {"real_occ"}) == []
    assert lineage_closure_errors(spans, {"other"})  # unknown -> flagged


def test_reconstruction_flags_tampered_offsets():
    source = "the quick brown fox"
    chunks = fixed_token_v1(source, tokenizer=TOK, window=10, overlap=0)
    spans = _spans(chunks, source)
    # Corrupt the recorded hash: reconstruction must catch it.
    spans[0]["text_sha256"] = "0" * 64
    assert reconstruction_errors(spans, {"c" * 64: source})


# --- visible-message reconstruction (the C_VISIBLE boundary) --------------------------
#
# These are the gates that make "the selector saw only what P0's compressor saw" checkable.
# Without them a VISIBLE_MESSAGE span is never reconstructed at all, so its offsets and its
# recorded bytes are pure assertion.


def _visible_span(message_bytes: bytes, start: int, end: int, view_hash: str = "v" * 64) -> dict:
    return build_visible_message_span(
        message_id="m1", message_role="tool", byte_start=start, byte_end=end,
        message_bytes=message_bytes, kind="TOOL_EVIDENCE",
        visible_compressor_view_hash=view_hash, source_occurrence_ids=["o1"],
    )


def test_reconstruction_catches_out_of_bounds_visible_span():
    """A 12-byte message with a span claiming byte_end=999 must fail.

    This is the concrete hole: reconstruction_errors skipped every non-RAW_SOURCE span, so a
    C_VISIBLE selection could address bytes the compressor never had and still preflight clean.
    """
    view = b"hello world!"          # 12 bytes
    span = _visible_span(view, 0, 12)
    span["byte_end"] = 999          # tamper after construction
    errors = reconstruction_errors([span], {}, visible_views={"v" * 64: view})
    assert errors, "an out-of-bounds visible-message span must be reported"
    assert "out of bounds" in errors[0]


def test_reconstruction_catches_tampered_visible_span_bytes():
    view = b"hello world!"
    span = _visible_span(view, 0, 5)
    span["exact_text_sha256"] = "0" * 64
    errors = reconstruction_errors([span], {}, visible_views={"v" * 64: view})
    assert errors


def test_reconstruction_accepts_a_faithful_visible_span():
    view = b"hello world!"
    span = _visible_span(view, 6, 11)   # "world"
    assert reconstruction_errors([span], {}, visible_views={"v" * 64: view}) == []


def test_reconstruction_reports_a_visible_span_with_no_available_view():
    """A span whose compressor view is not on hand cannot be waved through."""
    view = b"hello world!"
    span = _visible_span(view, 0, 5)
    errors = reconstruction_errors([span], {}, visible_views={})
    assert errors
    assert "not available" in errors[0]
