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

__all__ = [
    "ATOMIZE_VERSION",
    "Claim",
    "atomize_protocol_sha256",
    "atomize_report",
    "extract_anchors",
    "Anchors",
]

ATOMIZE_VERSION = "rule_atomize_v1"

_SENTENCE = re.compile(r"[^.!?\n]+(?:[.!?]+|\n|$)")
_NUMBER = re.compile(r"(?<![A-Za-z])\d[\d,]*(?:\.\d+)?%?")
_DATE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s*\d{0,4})\b"
)
_ENTITY = re.compile(r"\b(?:[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)+)\b")
_CITATION = re.compile(r"\[[EQ]\d+\]|\[\d+\]")


def atomize_protocol_sha256() -> str:
    """Identity of the rule-based claim parser used by every quality score.

    It is not a model prompt, but it is just as much part of the measurement policy: changing
    a sentence boundary or the factual-claim rule changes both metric numerators and
    denominators.  Hash the executable rule inputs instead of writing the truth prompt's hash
    into this slot.
    """
    from ..canonical import canonical_json
    from ..hashing import sha256_hex

    return sha256_hex(canonical_json({
        "version": ATOMIZE_VERSION,
        "sentence_regex": _SENTENCE.pattern,
        "number_regex": _NUMBER.pattern,
        "date_regex": _DATE.pattern,
        "entity_regex": _ENTITY.pattern,
        "citation_regex": _CITATION.pattern,
        "factual_rule": "citation_or_number_or_date_or_multiword_proper_noun",
    }))


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
