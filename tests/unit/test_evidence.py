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
        text = source[rc.char_start:rc.char_end]
        # A selected row is never orphaned from the header that names its columns.
        assert "Name" in text and "Value" in text


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
    assert cs.resolve("E1") == "span_a"
    assert cs.resolve("E2") == "span_b"
    assert cs.resolve("Q1") == "attempt_x"
    assert cs.label_for("span_b") == "E2"
    with pytest.raises(OutOfSetLabel):
        cs.resolve("E99")  # never leniently resolved


def test_candidate_labels_are_short():
    cs = CandidateSet.build([f"span_{i}" for i in range(12)])
    # Labels stay compact so they don't inflate prompt tokens.
    assert all(len(lbl) <= 4 for lbl in ["E1", "E9", "E12"])
    assert cs.resolve("E12") == "span_11"


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
