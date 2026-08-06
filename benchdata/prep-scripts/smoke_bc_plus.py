#!/usr/bin/env python
"""BM25 smoke test for BrowseComp-Plus.

Purpose: prove the Java 21 / pyserini / Lucene chain works end-to-end and give a
sanity number for retrieval quality. Retrieves top-k for a deterministic sample
of queries and reports recall@10 / recall@100 against both the `gold_docs` and
`evidence_docs` judgements.

Recall is macro-averaged: per query |retrieved@k ∩ relevant| / |relevant|,
then averaged over queries.

Query *text* is sent to the searcher but never printed; only ids and metrics are
emitted.
"""
import argparse
import json
import os
import random
import statistics
import time

# pyserini 1.6.0 builds an OpenAI client at import time for its LLM-based
# rerankers. We only touch Lucene, so a placeholder satisfies the constructor.
os.environ.setdefault("OPENAI_API_KEY", "not-used-bm25-only")
os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-21-openjdk-amd64")


def recall_at(retrieved, relevant, k):
    if not relevant:
        return None
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--decrypted", required=True)
    ap.add_argument("--n-queries", type=int, default=20)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260727)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    from pyserini.search.lucene import LuceneSearcher

    rows = [json.loads(l) for l in open(args.decrypted, encoding="utf-8") if l.strip()]
    rows.sort(key=lambda r: str(r["query_id"]))          # stable base order
    rng = random.Random(args.seed)
    sample = rng.sample(rows, args.n_queries)
    sample.sort(key=lambda r: str(r["query_id"]))

    searcher = LuceneSearcher(args.index)
    n_docs = searcher.num_docs
    print(f"  index: {args.index}")
    print(f"  index num_docs: {n_docs}")
    print(f"  sampling {args.n_queries} of {len(rows)} queries (seed={args.seed}), k={args.k}\n")

    per_query, latencies = [], []
    for r in sample:
        qid = str(r["query_id"])
        gold = {str(d["docid"]) for d in (r.get("gold_docs") or [])}
        evid = {str(d["docid"]) for d in (r.get("evidence_docs") or [])}

        t0 = time.perf_counter()
        hits = searcher.search(r["query"], k=args.k)
        dt = (time.perf_counter() - t0) * 1000
        latencies.append(dt)

        docids = [h.docid for h in hits]
        rec = {
            "query_id": qid,
            "n_gold": len(gold), "n_evidence": len(evid),
            "n_retrieved": len(docids),
            "latency_ms": round(dt, 1),
            "gold_recall@10":      recall_at(docids, gold, 10),
            "gold_recall@100":     recall_at(docids, gold, 100),
            "evidence_recall@10":  recall_at(docids, evid, 10),
            "evidence_recall@100": recall_at(docids, evid, 100),
        }
        per_query.append(rec)
        print(f"    {qid:>6}  gold {rec['gold_recall@10']:.2f}/{rec['gold_recall@100']:.2f}"
              f"   evidence {rec['evidence_recall@10']:.2f}/{rec['evidence_recall@100']:.2f}"
              f"   ({rec['n_gold']}g/{rec['n_evidence']}e, {rec['latency_ms']:.0f}ms)")

    def macro(key):
        vals = [r[key] for r in per_query if r[key] is not None]
        return round(statistics.mean(vals), 4) if vals else None

    summary = {
        "index": args.index,
        "index_num_docs": n_docs,
        "n_queries_sampled": args.n_queries,
        "k": args.k,
        "seed": args.seed,
        "macro_recall": {
            "gold@10": macro("gold_recall@10"),
            "gold@100": macro("gold_recall@100"),
            "evidence@10": macro("evidence_recall@10"),
            "evidence@100": macro("evidence_recall@100"),
        },
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 1),
            "median": round(statistics.median(latencies), 1),
            "min": round(min(latencies), 1),
            "max": round(max(latencies), 1),
        },
        "per_query": per_query,
    }

    print("\n  === macro-averaged recall ===")
    for k_, v in summary["macro_recall"].items():
        print(f"    {k_:<14} {v:.4f}")
    print(f"\n  latency: mean {summary['latency_ms']['mean']}ms  "
          f"median {summary['latency_ms']['median']}ms  max {summary['latency_ms']['max']}ms")

    json.dump(summary, open(args.out_json, "w"), indent=2)
    print(f"\n  wrote {args.out_json}")


if __name__ == "__main__":
    main()
