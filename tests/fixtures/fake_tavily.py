"""A fake Tavily upstream that freezes a small, deterministic world.

Deterministic so the same query always returns the same pages: the whole point of acquisition
is that the world stops moving, and a double that returned different results on each call would
let a test pass against behaviour the frozen corpus never has.
"""

from __future__ import annotations

from typing import Optional


class FakeTavily:
    """Returns pages keyed by the first content word of the query."""

    def __init__(self, *, pages_per_query: int = 2, fail_queries: Optional[set] = None,
                 empty_queries: Optional[set] = None, missing_raw: Optional[set] = None) -> None:
        self.pages_per_query = pages_per_query
        self.fail_queries = fail_queries or set()
        self.empty_queries = empty_queries or set()
        self.missing_raw = missing_raw or set()
        self.calls: list[str] = []

    async def __call__(self, endpoint: str, body: dict) -> tuple[int, dict]:
        query = str(body.get("query", ""))
        self.calls.append(query)
        if query in self.fail_queries:
            return 500, {"detail": "upstream unavailable"}
        if query in self.empty_queries:
            return 200, {"request_id": f"r-{len(self.calls)}", "response_time": 0.1,
                         "usage": {"credits": 1.0}, "failed_results": [], "results": []}
        stem = "".join(ch for ch in query.split()[0] if ch.isalnum()) or "page"
        results = []
        for i in range(self.pages_per_query):
            url = f"https://{stem}{i}.example/doc"
            raw = None if query in self.missing_raw else (
                f"# {stem} {i}\n\n"
                + "\n\n".join(
                    f"Paragraph {p} about {stem} number {i}. It records a measurement of "
                    f"{p * 7 + i} units observed in 2025 by the {stem} authority."
                    for p in range(6)
                )
            )
            results.append({
                "url": url, "title": f"{stem.title()} {i}",
                "content": f"Snippet for {stem} {i}.",
                "raw_content": raw, "score": 0.9 - 0.1 * i,
                "published_date": "2025-06-01",
            })
        return 200, {
            "request_id": f"r-{len(self.calls)}", "response_time": 0.2,
            "usage": {"credits": 1.0}, "failed_results": [], "results": results,
        }
