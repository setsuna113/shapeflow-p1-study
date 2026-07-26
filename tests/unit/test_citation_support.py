"""Citation support resolved against the frozen world, instead of always-unknown."""

from __future__ import annotations

from shapeflow_p1.evaluation.citation_support import (
    CitationSupportResolver,
    parse_citation_map,
    resolver_for,
)

_REPORT = """\
The reactor reached 41% efficiency in 2025 [1]. A second source disagrees [2].

Sources:
[1] Efficiency review -- https://example.com/efficiency
[2] Other study -- https://Other.example.org/study/
[3] A page nobody froze -- https://ghost.example/missing
"""

_POOL = [
    {"url": "https://example.com/efficiency", "content_hash": "h1"},
    {"url": "https://other.example.org/study", "content_hash": "h2"},
]

_TEXTS = {"h1": "The reactor reached 41% efficiency in 2025.",
          "h2": "The reactor reached 12% efficiency."}


def _resolver(judge):
    return resolver_for(
        report_text=_REPORT, pool_occurrences=_POOL, content_texts=_TEXTS,
        claim_texts={"c1": "The reactor reached 41% efficiency in 2025."},
        judge_relation=judge,
    )


def test_labels_resolve_to_the_urls_the_report_declared():
    mapping = parse_citation_map(_REPORT)
    assert mapping["1"] == "https://example.com/efficiency"
    assert mapping["3"] == "https://ghost.example/missing"


def test_a_citation_whose_page_entails_the_claim_supports_it():
    """This is the value that was always None, which made `covered` empty for every arm and
    every quality metric 0.0 across the whole study."""
    resolver = _resolver(lambda claim, evidence: "entail")
    assert resolver("c1", "1") is True


def test_a_citation_whose_page_contradicts_the_claim_does_not_support_it():
    resolver = _resolver(lambda claim, evidence: "contradict")
    assert resolver("c1", "2") is False


def test_an_uncertain_judge_leaves_the_citation_unknown():
    resolver = _resolver(lambda claim, evidence: "uncertain")
    assert resolver("c1", "1") is None


def test_a_url_outside_the_report_source_lineage_is_a_definite_failure():
    """A URL outside the arm-local source lineage was not read by this report."""
    resolver = _resolver(lambda claim, evidence: "entail")
    assert resolver("c1", "3") is False
    assert any("not retrieved by this arm" in r["reason"] for r in resolver.record())


def test_a_label_the_report_never_defined_is_unknown():
    resolver = _resolver(lambda claim, evidence: "entail")
    assert resolver("c1", "9") is None


def test_url_matching_ignores_a_trailing_slash_and_a_www_prefix():
    resolver = CitationSupportResolver(
        url_to_content={"//example.com/page": "h1"},
        content_texts={"h1": "text"},
        judge_relation=lambda claim, evidence: "entail",
        claim_texts={"c1": "claim"},
        citation_map={"1": "https://WWW.example.com/page/"},
    )
    assert resolver("c1", "1") is True


def test_every_resolution_records_why():
    """A zero has to be explicable: which label, which URL, and what decided it."""
    resolver = _resolver(lambda claim, evidence: "entail")
    resolver("c1", "1")
    resolver("c1", "3")
    record = resolver.record()
    assert [r["label"] for r in record] == ["1", "3"]
    assert record[0]["content_hash"] == "h1"
    assert record[0]["reason"] == "judge said entail"
