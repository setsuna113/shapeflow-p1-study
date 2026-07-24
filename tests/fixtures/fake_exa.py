"""A fake Exa upstream that freezes a small, deterministic world.

Deterministic so the same query always returns the same pages: the whole point of acquisition
is that the world stops moving, and a double that returned different results on each call would
let a test pass against behaviour the frozen corpus never has.

Shaped like Exa's real response -- ``text`` for the page, ``highlights`` for the excerpts,
``costDollars`` for the bill, no relevance score -- so the field mapping is exercised rather
than assumed.
"""

from __future__ import annotations

from typing import Optional


class FakeExa:
    """Returns pages keyed by the first content word of the query."""

    def __init__(self, *, pages_per_query: int = 2, fail_queries: Optional[set] = None,
                 empty_queries: Optional[set] = None, missing_raw: Optional[set] = None,
                 status_for: Optional[dict] = None) -> None:
        self.pages_per_query = pages_per_query
        self.fail_queries = fail_queries or set()
        self.empty_queries = empty_queries or set()
        self.missing_raw = missing_raw or set()
        self.status_for = status_for or {}
        self.calls: list[str] = []

    async def __call__(self, endpoint: str, body: dict) -> tuple[int, dict]:
        query = str(body.get("query", ""))
        self.calls.append(query)
        if query in self.status_for:
            return self.status_for[query], {"error": "refused"}
        if query in self.fail_queries:
            return 500, {"error": "upstream unavailable"}
        if query in self.empty_queries:
            return 200, {"requestId": f"r-{len(self.calls)}", "results": [],
                         "costDollars": {"total": 0.007}}
        stem = "".join(ch for ch in query.split()[0] if ch.isalnum()) or "page"
        results = []
        for i in range(self.pages_per_query):
            url = f"https://{stem}{i}.example/doc"
            text = None if query in self.missing_raw else (
                f"# {stem} {i}\n\n"
                + "\n\n".join(
                    f"Paragraph {p} about {stem} number {i}. It records a measurement of "
                    f"{p * 7 + i} units observed in 2025 by the {stem} authority."
                    for p in range(6)
                )
            )
            results.append({
                "id": f"doc-{stem}-{i}",
                "url": url,
                "title": f"{stem.title()} {i}",
                "highlights": [f"Snippet for {stem} {i}."],
                "text": text,
                "publishedDate": "2025-06-01",
                "author": None,
            })
        return 200, {
            "requestId": f"r-{len(self.calls)}",
            "results": results,
            "costDollars": {"total": 0.007, "search": {"neural": 0.007}},
        }
