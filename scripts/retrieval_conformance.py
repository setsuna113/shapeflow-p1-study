#!/usr/bin/env python3
"""Run the encoder/index conformance check against the real shipped index.

Runs on the host, in an interpreter that has torch and transformers -- deliberately not the
study's own venv, whose dependency closure is pinned to the vendor's lock. Only
``shapeflow.retrieval`` and ``shapeflow.hashing`` are imported, and those need nothing but numpy.

    HF_ENDPOINT=https://hf-mirror.com \
    /storage/sata/shapeflow/benchdata/.venv/bin/python scripts/retrieval_conformance.py \
        --index-dir  /storage/sata/shapeflow/benchdata/browsecomp-plus/indexes/qwen3-embedding-0.6b \
        --corpus-dir /storage/sata/shapeflow/benchdata/browsecomp-plus/corpus/data \
        --model Qwen/Qwen3-Embedding-0.6B --docs 200

Exit 0 if the encoder reproduces the index, 1 otherwise. Nothing downstream should spend a GPU
hour until this passes: every wrong recipe yields an encoder that runs and retrieves badly, and
the resulting failure looks like "this benchmark is too hard" rather than like a bug.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from shapeflow.retrieval.conformance import check_encoder_matches_index  # noqa: E402
from shapeflow.retrieval.encoder import EncoderSpec, QueryEncoder  # noqa: E402
from shapeflow.retrieval.index import load_index  # noqa: E402


def sample_docs(corpus_dir: pathlib.Path, wanted: set[str], limit: int) -> dict[str, str]:
    """Read text for ``wanted`` docids out of the corpus parquet files."""
    import pyarrow.parquet as pq

    found: dict[str, str] = {}
    for path in sorted(corpus_dir.glob("*.parquet")):
        table = pq.read_table(path, columns=["docid", "text"])
        for docid, text in zip(table.column("docid").to_pylist(),
                               table.column("text").to_pylist()):
            if docid in wanted and docid not in found:
                found[docid] = text
                if len(found) >= limit:
                    return found
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", required=True, type=pathlib.Path)
    ap.add_argument("--corpus-dir", required=True, type=pathlib.Path)
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--revision", default="")
    ap.add_argument("--docs", type=int, default=200)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="passages pad to the longest in their batch; keep small")
    ap.add_argument("--output", type=pathlib.Path,
                    default=REPO / "reports" / "RETRIEVAL_CONFORMANCE.json")
    args = ap.parse_args()

    print(f"loading index from {args.index_dir} ...", flush=True)
    started = time.time()
    index = load_index(args.index_dir)
    print(f"  {index.num_docs} docs, dim {index.dim}, {len(index.shards)} shards "
          f"({time.time() - started:.1f}s)", flush=True)

    # Take documents spread across the whole index rather than the first N, so a shard that was
    # concatenated in the wrong order cannot pass by never being sampled.
    ids = _spread(index, args.docs)
    docs = sample_docs(args.corpus_dir, set(ids), args.docs)
    print(f"  resolved text for {len(docs)}/{len(ids)} sampled docids", flush=True)
    if len(docs) < args.docs // 2:
        print("FAILED: could not resolve enough document text from the corpus", file=sys.stderr)
        return 1

    print(f"loading encoder {args.model} on CPU ({args.threads} threads) ...", flush=True)
    encoder = QueryEncoder(
        EncoderSpec(model=args.model, revision=args.revision), threads=args.threads)

    started = time.time()
    def _tick(done, total, min_cos):
        print(f"  {done}/{total} checked, min cosine so far {min_cos:.5f} "
              f"({time.time() - started:.0f}s)", flush=True)

    result = check_encoder_matches_index(encoder, index, docs,
                                         batch_size=args.batch_size, progress=_tick)
    elapsed = time.time() - started

    body = result.content()
    body["model"] = args.model
    body["index_dir"] = str(args.index_dir)
    body["encode_seconds"] = round(elapsed, 2)
    body["batch_size"] = args.batch_size
    body["diagnosis"] = result.diagnosis()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\nchecked {body['checked']} docs in {elapsed:.1f}s")
    print(f"  min cosine      {body['min_cosine']}")
    print(f"  rank-1 self-hit {body['rank_1_count']}/{body['checked']}")
    print(f"  -> {body['diagnosis']}")
    print(f"written to {args.output}")
    return 0 if result.ok else 1


def _spread(index, count: int) -> list[str]:
    """``count`` docids spread evenly across the concatenated index."""
    total = index.num_docs
    step = max(1, total // max(1, count))
    ids = [index._docids[i] for i in range(0, total, step)][:count]  # noqa: SLF001
    return ids


if __name__ == "__main__":
    raise SystemExit(main())
