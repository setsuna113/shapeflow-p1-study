#!/usr/bin/env python
"""Run BM25 over all BrowseComp-Plus queries and emit a TREC run file.

This exists to validate the smoke-test numbers against the benchmark's own
evaluation path (pyserini.eval.trec_eval over topics-qrels/*), so we can tell a
genuinely weak retriever apart from a broken indexing/docid pipeline.

Idempotent: skips the search if the run file already has the expected line count.
"""
import argparse
import json
import os
import time

os.environ.setdefault("OPENAI_API_KEY", "not-used-bm25-only")
os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-21-openjdk-amd64")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--queries-tsv", required=True)
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out-run", required=True)
    args = ap.parse_args()

    qs = []
    with open(args.queries_tsv, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            qid, _, text = line.rstrip("\n").partition("\t")
            qs.append((qid, text))
    print(f"  {len(qs)} queries, k={args.k}")

    if os.path.exists(args.out_run):
        have = sum(1 for _ in open(args.out_run))
        if have >= len(qs):          # every query contributed at least one line
            print(f"  [skip] run file already present with {have} lines")
            return

    from pyserini.search.lucene import LuceneSearcher
    searcher = LuceneSearcher(args.index)
    print(f"  index num_docs={searcher.num_docs}")

    # NOTE: searcher.batch_search() aborts the whole batch on this index
    # ("queryCount = 830 is not equal to completedTaskCount = 65" from
    # Anserini's SimpleSearcher thread pool). Sequential search is reliable and
    # fast enough (~220 ms/query), so we loop and isolate per-query failures.
    t0 = time.time()
    n, failures = 0, []
    with open(args.out_run + ".tmp", "w") as out:
        for i, (qid, text) in enumerate(qs, 1):
            try:
                hits = searcher.search(text, k=args.k)
            except Exception as e:
                failures.append((qid, f"{type(e).__name__}: {e}"))
                continue
            for rank, hit in enumerate(hits, start=1):
                out.write(f"{qid} Q0 {hit.docid} {rank} {hit.score:.6f} bm25\n")
                n += 1
            if i % 100 == 0:
                print(f"    {i}/{len(qs)} queries, {n} lines, {time.time()-t0:.0f}s", flush=True)
    os.replace(args.out_run + ".tmp", args.out_run)
    elapsed = time.time() - t0
    print(f"  searched in {elapsed:.1f}s ({elapsed/len(qs)*1000:.0f} ms/query)")
    print(f"  wrote {n} lines -> {args.out_run}")
    if failures:
        print(f"  WARNING: {len(failures)} queries failed:")
        for qid, err in failures[:10]:
            print(f"    {qid}: {err[:120]}")


if __name__ == "__main__":
    main()
