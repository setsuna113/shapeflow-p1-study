#!/usr/bin/env python
"""Sanity-check the BM25 index and docid mapping.

If BM25 recall on BrowseComp-Plus is genuinely low (rather than an artefact of a
broken index or a docid namespace mismatch), then *self-retrieval* must still
work: taking a distinctive span of text out of a known gold document and
searching for it should return that same document at rank 1.

Also verifies that qrel docids actually exist in the Lucene index.
"""
import json
import os
import random
import re

os.environ.setdefault("OPENAI_API_KEY", "not-used-bm25-only")
os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-21-openjdk-amd64")

B = "/storage/sata/shapeflow/benchdata"
from pyserini.search.lucene import LuceneSearcher  # noqa: E402

searcher = LuceneSearcher(f"{B}/browsecomp-plus/indexes/bm25")
print(f"index num_docs = {searcher.num_docs}\n")

rows = [json.loads(l) for l in
        open(f"{B}/browsecomp-plus/data/browsecomp_plus_decrypted.jsonl", encoding="utf-8")]

# ---- 1. do qrel docids exist in the index? ----
missing, checked = 0, 0
for r in rows[:200]:
    for d in (r.get("gold_docs") or []):
        checked += 1
        if searcher.doc(str(d["docid"])) is None:
            missing += 1
print(f"[1] gold docid presence: {checked - missing}/{checked} resolvable in index "
      f"({missing} missing)")

# ---- 2. self-retrieval from gold document text ----
rng = random.Random(20260727)
sample = rng.sample(rows, 10)
ranks = []
for r in sample:
    golds = r.get("gold_docs") or []
    if not golds:
        continue
    g = golds[0]
    docid = str(g["docid"])
    # pull a distinctive mid-document span straight from the index copy
    stored = searcher.doc(docid)
    if stored is None:
        ranks.append((docid, None, "docid not in index"))
        continue
    raw = stored.raw() or ""
    try:
        text = json.loads(raw).get("text", raw)
    except Exception:
        text = raw
    words = re.findall(r"\w+", text)
    if len(words) < 60:
        continue
    span = " ".join(words[40:70])          # 30-word span from inside the doc
    hits = searcher.search(span, k=10)
    rank = next((i for i, h in enumerate(hits, 1) if h.docid == docid), None)
    ranks.append((docid, rank, ""))

found_at_1 = sum(1 for _, r_, _ in ranks if r_ == 1)
found_top10 = sum(1 for _, r_, _ in ranks if r_ is not None)
print(f"\n[2] self-retrieval of gold docs by their own text "
      f"({len(ranks)} docs tested):")
for docid, rank, note in ranks:
    print(f"      docid {docid:>7}  rank={rank}  {note}")
print(f"    rank-1: {found_at_1}/{len(ranks)}   in top-10: {found_top10}/{len(ranks)}")

verdict = ("INDEX SOUND - low benchmark recall is a property of the queries, "
           "not a pipeline bug") if found_at_1 >= max(1, int(0.8 * len(ranks))) \
    else "INDEX SUSPECT - investigate analyzer / docid mapping"
print(f"\n[verdict] {verdict}")

json.dump({
    "index_num_docs": searcher.num_docs,
    "gold_docid_presence": {"checked": checked, "missing": missing},
    "self_retrieval": {"tested": len(ranks), "rank_1": found_at_1,
                       "in_top_10": found_top10,
                       "per_doc": [{"docid": d, "rank": r} for d, r, _ in ranks]},
    "verdict": verdict,
}, open(f"{B}/browsecomp-plus/index_sanity.json", "w"), indent=2)
print(f"wrote {B}/browsecomp-plus/index_sanity.json")
