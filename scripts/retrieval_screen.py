#!/usr/bin/env python3
"""Measure one encoder size against the dev split. Idempotent, resumable, disk-backed.

Freeze-1 §3.1 reserves a slot for the frozen retriever; this fills it with a measurement rather
than a preference. It runs on the **dev split only** and never touches the sealed 300 -- choosing
a retriever is a design decision, and a design decision informed by the confirmatory split is not
a design decision, it is a result.

Two numbers per size, because the choice has two arms:

- **recall** against the benchmark's own qrels, gold and evidence, at k = 5/100/1000;
- **query encode latency** on the production configuration -- CPU, pinned cpuset, one query at a
  time -- because that cost lands on every search the agent makes.

Re-running is free: a size whose result file already exists is skipped. That matters because the
8B pass takes tens of minutes and a broken pipe should not cost it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from shapeflow.retrieval.encoder import EncoderSpec, QueryEncoder  # noqa: E402
from shapeflow.retrieval.index import load_index  # noqa: E402

RECALL_AT = (5, 100, 1000)


def read_qrels(path: pathlib.Path) -> dict[str, set[str]]:
    """TREC qrels: ``qid Q0 docid rel``. Only positive judgements count."""
    out: dict[str, set[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 4 and int(parts[3]) > 0:
            out.setdefault(parts[0], set()).add(parts[2])
    return out


def read_queries(path: pathlib.Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        qid, _, text = line.partition("\t")
        if text:
            out[qid.strip()] = text.strip()
    return out


def recall_at(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", required=True, type=pathlib.Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="")
    ap.add_argument("--tag", required=True, help="short name for the output file, e.g. 0.6b")
    ap.add_argument("--queries", required=True, type=pathlib.Path)
    ap.add_argument("--split", required=True, type=pathlib.Path, help="one query id per line")
    ap.add_argument("--qrel-golds", required=True, type=pathlib.Path)
    ap.add_argument("--qrel-evidence", required=True, type=pathlib.Path)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--outdir", type=pathlib.Path, default=REPO / "reports" / "retrieval_screen")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_path = args.outdir / f"{args.tag}.json"
    if out_path.exists():
        print(f"{out_path} exists; skipping {args.tag}", flush=True)
        return 0

    split_ids = [ln.strip() for ln in args.split.read_text().splitlines() if ln.strip()]
    queries = read_queries(args.queries)
    golds, evidence = read_qrels(args.qrel_golds), read_qrels(args.qrel_evidence)
    missing = [q for q in split_ids if q not in queries]
    if missing:
        print(f"FAILED: {len(missing)} split ids absent from the query file, e.g. {missing[:5]}",
              file=sys.stderr)
        return 1
    print(f"[{args.tag}] {len(split_ids)} dev queries, "
          f"{len(golds)} gold / {len(evidence)} evidence qrel rows", flush=True)

    index = load_index(args.index_dir)
    print(f"[{args.tag}] index {index.num_docs} docs, dim {index.dim}", flush=True)

    encoder = QueryEncoder(EncoderSpec(model=args.model, revision=args.revision),
                           threads=args.threads)
    latencies: list[float] = []
    per_query: dict[str, list[str]] = {}
    started = time.time()
    for n, qid in enumerate(split_ids, start=1):
        t0 = time.perf_counter()
        vector = encoder.encode_query(queries[qid])   # one at a time: the production shape
        latencies.append(time.perf_counter() - t0)
        per_query[qid] = [h.docid for h in index.search(vector, top_k=max(RECALL_AT))]
        if n % 50 == 0:
            print(f"[{args.tag}] {n}/{len(split_ids)} encoded, "
                  f"median {sorted(latencies)[len(latencies) // 2] * 1000:.0f}ms "
                  f"({time.time() - started:.0f}s)", flush=True)

    latencies.sort()
    body = {
        "tag": args.tag,
        "model": args.model,
        "revision": args.revision,
        "index_dir": str(args.index_dir),
        "index_dim": index.dim,
        "index_num_docs": index.num_docs,
        "index_shard_sha256": [s.sha256 for s in index.shards],
        "split": str(args.split),
        "num_queries": len(split_ids),
        "threads": args.threads,
        "encode_latency_ms": {
            "p50": round(latencies[len(latencies) // 2] * 1000, 2),
            "p95": round(latencies[int(0.95 * len(latencies)) - 1] * 1000, 2),
            "max": round(latencies[-1] * 1000, 2),
        },
        "total_seconds": round(time.time() - started, 1),
        "recall": {},
    }
    for label, qrels in (("gold", golds), ("evidence", evidence)):
        scored = {k: [] for k in RECALL_AT}
        for qid in split_ids:
            relevant = qrels.get(qid, set())
            if not relevant:
                continue
            for k in RECALL_AT:
                scored[k].append(recall_at(per_query[qid], relevant, k))
        body["recall"][label] = {
            f"recall@{k}": round(sum(v) / len(v), 4) if v else None for k, v in scored.items()
        }
        body["recall"][f"{label}_queries_scored"] = len(scored[RECALL_AT[0]])

    out_path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[{args.tag}] gold {body['recall']['gold']} | "
          f"evidence {body['recall']['evidence']} | "
          f"p95 {body['encode_latency_ms']['p95']}ms -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
