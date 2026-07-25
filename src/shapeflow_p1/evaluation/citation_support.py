"""Does a claim's citation actually support it?

``covered`` in quality_metrics requires a SUPPORTED claim *with a supporting citation*, and
``citation_supports`` was ``lambda claim_id, label: None`` -- unknown, always. Unknown routes
every citation to the human queue and counts as unsupported, so ``covered`` was the empty
set for every arm, and weighted_required_atom_recall, required_facet_coverage,
contradiction_handling and qualified_report were 0.0 for every arm of every task. The whole
quality half of the study reported zeros and nothing in the pipeline could tell that apart
from an arm that genuinely cited nothing.

Resolving a citation is three steps, and each can fail honestly:

1. **Label to URL.** The report's own source list. A label with no entry resolves to nothing.
2. **URL to frozen content.** The task's published pool. A URL the frozen world does not
   contain is a fabricated citation, which is a definite ``False`` rather than an unknown.
3. **Content to support.** The same blind relation judge used everywhere else, over the
   claim and the cited page's text. Uncertain stays uncertain and goes to the human queue.

Step 2 is the one worth being careful about: "the runner cited a URL that does not exist in
its frozen world" is a finding, not a missing measurement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

__all__ = [
    "parse_citation_map",
    "CitationSupportResolver",
    "CitationResolution",
]

#: `[3] https://example.com/page` or `[3] Title -- https://example.com/page`, one per line,
#: which is the shape vendor's final report writes its source list in.
_SOURCE_LINE = re.compile(
    r"^\s*\[(?P<label>[^\]]{1,32})\]\s*(?P<rest>.+?)\s*$", re.MULTILINE)
_URL = re.compile(r"https?://[^\s)>\]]+")


def parse_citation_map(report_text: str) -> dict[str, str]:
    """Label -> URL, from the report's own source list.

    Only lines that carry a URL count. An inline ``[3]`` reference with no entry in the list
    is a label that resolves to nothing, which is the honest reading of a report that cited
    a number it never defined.
    """
    mapping: dict[str, str] = {}
    for match in _SOURCE_LINE.finditer(report_text or ""):
        url = _URL.search(match.group("rest"))
        if url:
            mapping.setdefault(match.group("label").strip(), url.group(0))
    return mapping


def _normalize(url: str) -> str:
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit(("", host, path, "", ""))


@dataclass(frozen=True)
class CitationResolution:
    """Why a citation resolved the way it did. Kept so a 0 is explicable."""

    label: str
    url: str = ""
    content_hash: str = ""
    supports: Optional[bool] = None
    reason: str = ""


@dataclass
class CitationSupportResolver:
    """Resolve citation labels against one task's frozen world."""

    #: url (normalized) -> content_hash, from the published runner pool.
    url_to_content: Mapping[str, str]
    #: content_hash -> the page text as frozen.
    content_texts: Mapping[str, str]
    #: (claim_text, evidence_text) -> entail | contradict | unrelated | uncertain.
    judge_relation: Callable[[str, str], str]
    #: claim_id -> claim text, so the resolver can be handed to build_assessment's
    #: (claim_id, label) contract.
    claim_texts: Mapping[str, str] = field(default_factory=dict)
    #: label -> url, from the report under evaluation.
    citation_map: Mapping[str, str] = field(default_factory=dict)
    #: Every resolution taken, for the score record.
    resolutions: list = field(default_factory=list)

    @staticmethod
    def _key(label: str) -> str:
        """The atomizer yields ``[1]``; the source list is keyed on ``1``."""
        return str(label).strip().strip("[]").strip()

    def __call__(self, claim_id: str, label: str) -> Optional[bool]:
        claim_text = self.claim_texts.get(claim_id, "")
        url = self.citation_map.get(self._key(label), "")
        if not url:
            self.resolutions.append(CitationResolution(
                label=str(label), reason="the report defines no source for this label"))
            return None
        content_hash = self.url_to_content.get(_normalize(url), "")
        if not content_hash:
            # Not unknown. The frozen world is the whole world this run could see, so a URL
            # that is not in it was not read.
            self.resolutions.append(CitationResolution(
                label=str(label), url=url, supports=False,
                reason="the cited URL is not in this task's frozen world"))
            return False
        text = self.content_texts.get(content_hash, "")
        if not text:
            self.resolutions.append(CitationResolution(
                label=str(label), url=url, content_hash=content_hash,
                reason="the frozen page has no readable content"))
            return None
        if not claim_text:
            self.resolutions.append(CitationResolution(
                label=str(label), url=url, content_hash=content_hash,
                reason="the claim text was not available to judge against"))
            return None
        relation = self.judge_relation(claim_text, text)
        if relation == "entail":
            supports: Optional[bool] = True
        elif relation in ("contradict", "unrelated"):
            supports = False
        else:
            supports = None
        self.resolutions.append(CitationResolution(
            label=str(label), url=url, content_hash=content_hash, supports=supports,
            reason=f"judge said {relation}"))
        return supports

    def record(self) -> list[dict]:
        return [
            {"label": r.label, "url": r.url, "content_hash": r.content_hash,
             "supports": r.supports, "reason": r.reason}
            for r in self.resolutions
        ]


def resolver_for(
    *,
    report_text: str,
    pool_occurrences: Sequence[Mapping],
    content_texts: Mapping[str, str],
    claim_texts: Mapping[str, str],
    judge_relation: Callable[[str, str], str],
) -> CitationSupportResolver:
    """Build a resolver for one arm's report against one task's frozen pool."""
    url_to_content = {}
    for occurrence in pool_occurrences:
        url = str(occurrence.get("url") or "")
        content_hash = str(occurrence.get("content_hash") or "")
        if url and content_hash:
            url_to_content.setdefault(_normalize(url), content_hash)
    return CitationSupportResolver(
        url_to_content=url_to_content,
        content_texts=content_texts,
        judge_relation=judge_relation,
        claim_texts=dict(claim_texts),
        citation_map=parse_citation_map(report_text),
    )
