"""Property: whatever the text, every chunk reconstructs exactly from its offsets.

Exact reconstruction is the load-bearing guarantee of the evidence IR -- a selected span is
re-hashed against the snapshot at preflight, and any drift is corruption. So it must hold for
arbitrary input, not just the hand-picked cases in the unit tests.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from shapeflow.evidence.chunkers import (
    WhitespaceTokenizer,
    fixed_token_v1,
    markdown_structure_v1,
    paragraph_sentence_v1,
)
from shapeflow.evidence.identity import build_evidence_span
from shapeflow.evidence.lineage import reconstruction_errors

TOK = WhitespaceTokenizer()

# A mix of words, whitespace, newlines and markdown punctuation, so tables/lists/headings and
# odd spacing all get exercised. Built from multi-character fragments joined together
# (st.text needs single-char alphabets, so we assemble fragments via a list instead).
_FRAGMENTS = list("abcde ") + ["\n", "\n\n", "# ", "## ", "- ", "> ", "| ", "|", "---", "```", ". "]
_text = st.lists(st.sampled_from(_FRAGMENTS), min_size=0, max_size=120).map("".join)


def _reconstructs(source: str, chunks) -> bool:
    if not chunks:
        return True
    spans = [
        build_evidence_span(
            c, source, content_hash="c" * 64, source_occurrence_ids=["o1"],
            chunker_version="prop",
        )
        for c in chunks
    ]
    return reconstruction_errors(spans, {"c" * 64: source}) == []


@settings(max_examples=200)
@given(source=_text)
def test_fixed_token_reconstructs(source):
    assert _reconstructs(source, fixed_token_v1(source, tokenizer=TOK, window=4, overlap=1))


@settings(max_examples=200)
@given(source=_text)
def test_paragraph_sentence_reconstructs(source):
    assert _reconstructs(source, paragraph_sentence_v1(source, tokenizer=TOK, max_tokens=5))


@settings(max_examples=300)
@given(source=_text)
def test_markdown_structure_reconstructs(source):
    assert _reconstructs(source, markdown_structure_v1(source, tokenizer=TOK, max_tokens=6))


@settings(max_examples=200)
@given(source=_text)
def test_chunkers_are_deterministic(source):
    a = markdown_structure_v1(source, tokenizer=TOK, max_tokens=6)
    b = markdown_structure_v1(source, tokenizer=TOK, max_tokens=6)
    assert a == b
