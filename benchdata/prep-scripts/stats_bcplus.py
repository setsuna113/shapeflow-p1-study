#!/usr/bin/env python
"""Corpus/query statistics for BrowseComp-Plus.

Deliberately emits only counts and distributions. Query text, answers and
document text are never printed (benchmark hygiene: decrypted answers must not
reach logs or terminals).
"""
import argparse
import glob
import json
import os
import statistics
from collections import Counter


def pct(sorted_vals, p):
    if not sorted_vals:
        return 0
    k = (len(sorted_vals) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] if lo == hi else sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def describe(name, vals, out):
    s = sorted(vals)
    out[name] = {
        "n_queries": len(s),
        "total": sum(s),
        "min": s[0], "max": s[-1],
        "mean": round(statistics.mean(s), 3),
        "median": statistics.median(s),
        "p25": pct(s, 25), "p75": pct(s, 75), "p90": pct(s, 90),
        "zero_count": sum(1 for v in s if v == 0),
    }
    hist = Counter()
    for v in s:
        if v <= 5:
            hist[str(v)] += 1
        elif v <= 10:
            hist["6-10"] += 1
        elif v <= 20:
            hist["11-20"] += 1
        elif v <= 50:
            hist["21-50"] += 1
        else:
            hist["50+"] += 1
    out[name]["histogram"] = dict(hist)
    print(f"\n  {name}:")
    print(f"    queries={len(s)} total={sum(s)} min={s[0]} max={s[-1]} "
          f"mean={statistics.mean(s):.2f} median={statistics.median(s)} "
          f"p25={pct(s,25):.0f} p75={pct(s,75):.0f} p90={pct(s,90):.0f} zero={out[name]['zero_count']}")
    print(f"    histogram: {dict(sorted(hist.items(), key=lambda kv: (len(kv[0]), kv[0])))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decrypted", required=True)
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--qrel-evidence", required=True)
    ap.add_argument("--qrel-golds", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    out = {}

    # ---- corpus doc count (parquet metadata only, no full read) ----
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(args.corpus_dir, "data", "*.parquet")))
    n_docs = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    out["corpus"] = {"parquet_files": len(files), "num_documents": n_docs}
    print(f"  corpus: {n_docs} documents across {len(files)} parquet files")

    # ---- queries ----
    rows = [json.loads(l) for l in open(args.decrypted, encoding="utf-8") if l.strip()]
    out["queries"] = {"num_queries": len(rows), "fields": sorted(rows[0].keys())}
    print(f"  queries: {len(rows)}   fields={sorted(rows[0].keys())}")

    qids = [r["query_id"] for r in rows]
    out["queries"]["unique_query_ids"] = len(set(qids))

    describe("evidence_docs_per_query", [len(r.get("evidence_docs") or []) for r in rows], out)
    describe("gold_docs_per_query",     [len(r.get("gold_docs")     or []) for r in rows], out)
    describe("negative_docs_per_query", [len(r.get("negative_docs") or []) for r in rows], out)

    # queries with at least one gold and at least one evidence doc
    out["queries"]["with_gold>=1"] = sum(1 for r in rows if len(r.get("gold_docs") or []) >= 1)
    out["queries"]["with_evidence>=1"] = sum(1 for r in rows if len(r.get("evidence_docs") or []) >= 1)

    # ---- cross-check against the repo's qrel files ----
    def read_qrel(p):
        per_q = Counter()
        with open(p, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 4 and int(parts[3]) > 0:
                    per_q[parts[0]] += 1
        return per_q

    ev, go = read_qrel(args.qrel_evidence), read_qrel(args.qrel_golds)
    out["qrel_files"] = {
        "qrel_evidence": {"queries": len(ev), "positive_judgements": sum(ev.values())},
        "qrel_golds":    {"queries": len(go), "positive_judgements": sum(go.values())},
    }
    print(f"\n  qrel_evidence.txt: {len(ev)} queries, {sum(ev.values())} positive judgements")
    print(f"  qrel_golds.txt   : {len(go)} queries, {sum(go.values())} positive judgements")

    jsonl_ev = {r["query_id"]: len(r.get("evidence_docs") or []) for r in rows}
    mismatch = [q for q in ev if ev[q] != jsonl_ev.get(q)]
    out["qrel_files"]["evidence_matches_jsonl"] = (len(mismatch) == 0)
    out["qrel_files"]["evidence_mismatch_count"] = len(mismatch)
    print(f"  evidence counts match decrypted jsonl: {len(mismatch) == 0} "
          f"({len(mismatch)} mismatched queries)")

    json.dump(out, open(args.out_json, "w"), indent=2)
    print(f"\n  wrote {args.out_json}")


if __name__ == "__main__":
    main()
