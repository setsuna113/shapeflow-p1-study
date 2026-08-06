#!/usr/bin/env python
"""Re-test the two self-retrieval misses with several different spans each.

A single 30-word span can land on boilerplate (nav bars, cookie notices) that is
shared across thousands of pages, which would fail to identify its own document
even on a perfectly good index. Trying multiple spans separates that from a real
indexing problem.
"""
import json
import os
import re

os.environ.setdefault("OPENAI_API_KEY", "not-used-bm25-only")
os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-21-openjdk-amd64")

B = "/storage/sata/shapeflow/benchdata"
from pyserini.search.lucene import LuceneSearcher  # noqa: E402

searcher = LuceneSearcher(f"{B}/browsecomp-plus/indexes/bm25")

for docid in ("68214", "72281"):
    stored = searcher.doc(docid)
    raw = stored.raw() or ""
    try:
        text = json.loads(raw).get("text", raw)
    except Exception:
        text = raw
    words = re.findall(r"\w+", text)
    print(f"\ndocid {docid}: {len(words)} words in stored text")
    for start in (0, 40, 200, 500, 1000):
        if start + 40 > len(words):
            continue
        span = " ".join(words[start:start + 40])
        hits = searcher.search(span, k=10)
        rank = next((i for i, h in enumerate(hits, 1) if h.docid == docid), None)
        print(f"   span@{start:<5} rank={rank}   top1={hits[0].docid if hits else None}")
