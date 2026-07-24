"""Deterministic atomization of a report into atomic, matchable claims.

The atomizer is fixed and rule-based, not a model, so the same report always splits the same way
and the denominator of every claim-based metric is reproducible. It splits on sentence
boundaries (keeping char offsets so a claim traces back to the report), classifies each claim as
factual/verifiable, and extracts the exact-match anchors -- numbers, dates, and capitalized
entities -- that the matching protocol uses before it ever consults a judge.

Classifying a claim as factual is intentionally inclusive: a claim carrying a number, a date, a
citation marker, or a multi-word proper noun is factual. Grounded-claim precision then has a
non-empty denominator for any real report, and a report that states verifiable things but grounds
none of them scores low rather than being scored as having "no factual claims".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = ["Claim", "atomize_report", "extract_anchors", "Anchors"]

_SENTENCE = re.compile(r"[^.!?\n]+(?:[.!?]+|\n|$)")
_NUMBER = re.compile(r"(?<![A-Za-z])\d[\d,]*(?:\.\d+)?%?")
_DATE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s*\d{0,4})\b"
)
_ENTITY = re.compile(r"\b(?:[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)+)\b")
_CITATION = re.compile(r"\[[EQ]\d+\]|\[\d+\]")


@dataclass(frozen=True)
class Anchors:
    numbers: tuple[str, ...]
    dates: tuple[str, ...]
    entities: tuple[str, ...]

    def is_empty(self) -> bool:
        return not (self.numbers or self.dates or self.entities)


@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str
    char_start: int
    char_end: int
    factual: bool
    citation_labels: tuple[str, ...]
    anchors: Anchors


def extract_anchors(text: str) -> Anchors:
    dates = tuple(m.group(0) for m in _DATE.finditer(text))
    # Numbers that are part of a matched date are not counted separately.
    date_spans = [m.span() for m in _DATE.finditer(text)]

    def _in_date(span):
        return any(s <= span[0] and span[1] <= e for s, e in date_spans)

    numbers = tuple(m.group(0) for m in _NUMBER.finditer(text) if not _in_date(m.span()))
    entities = tuple(m.group(0) for m in _ENTITY.finditer(text))
    return Anchors(numbers=numbers, dates=dates, entities=entities)


def atomize_report(text: str) -> list[Claim]:
    """Split ``text`` into atomic claims with offsets, factual flags, and anchors."""
    claims: list[Claim] = []
    for i, m in enumerate(_SENTENCE.finditer(text)):
        raw = m.group(0)
        stripped = raw.strip()
        if not stripped:
            continue
        start = m.start() + (len(raw) - len(raw.lstrip()))
        end = start + len(stripped)
        citations = tuple(c.group(0) for c in _CITATION.finditer(stripped))
        anchors = extract_anchors(stripped)
        factual = bool(citations) or not anchors.is_empty()
        claims.append(
            Claim(
                claim_id=f"claim_{i}",
                text=stripped,
                char_start=start,
                char_end=end,
                factual=factual,
                citation_labels=citations,
                anchors=anchors,
            )
        )
    return claims
