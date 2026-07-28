#!/usr/bin/env python3
"""Assemble the retrieval freeze from the artifacts that were actually measured.

Built from the screen and conformance reports on disk rather than from a config file, so the
frozen record names the index that was searched and the encoder that was checked against it --
not the ones someone believed were in use.

The freeze is written **not effective**: ``effective_after`` stays empty until the P0 competence
pilot records its digest. Selecting a retriever and validating one are different acts, and only
the second licenses a campaign.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from shapeflow.hashing import sha256_hex  # noqa: E402
from shapeflow.retrieval.freeze import RetrievalFreeze, write_freeze  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", required=True, type=pathlib.Path,
                    help="the screen result for the chosen size AND serving dtype")
    ap.add_argument("--conformance", required=True, type=pathlib.Path)
    ap.add_argument("--corpus-dir", required=True, type=pathlib.Path)
    ap.add_argument("--top-k", type=int, required=True)
    ap.add_argument("--bench-repo-commit", default="")
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "protocol" / "retrieval_freeze.json")
    args = ap.parse_args()

    screen = json.loads(args.screen.read_text())
    conf = json.loads(args.conformance.read_text())

    if not conf.get("ok"):
        print(f"REFUSED: {args.conformance} records a failed conformance check; a freeze whose "
              "encoder does not reproduce its index is a filename, not a fact", file=sys.stderr)
        return 1
    if conf.get("dtype") != screen.get("dtype"):
        print(f"REFUSED: conformance ran in {conf.get('dtype')!r} but the screen measured "
              f"{screen.get('dtype')!r}. The freeze must record one serving dtype, validated in "
              "that dtype.", file=sys.stderr)
        return 1
    if conf.get("index_shard_sha256") != screen.get("index_shard_sha256"):
        print("REFUSED: the conformance check and the screen used different index shards",
              file=sys.stderr)
        return 1

    spec = conf["encoder_spec"]
    corpus_digests = tuple(
        sha256_hex(p.read_bytes()) for p in sorted(args.corpus_dir.glob("*.parquet")))

    freeze = RetrievalFreeze(
        encoder_repo=spec["model"],
        encoder_revision=spec["revision"],
        encoder_dtype=spec["dtype"],
        pooling=spec["pooling"],
        normalize=spec["normalize"],
        query_prefix_sha256=spec["query_prefix_sha256"],
        passage_prefix_sha256=spec["passage_prefix_sha256"],
        query_max_len=spec["query_max_len"],
        passage_max_len=spec["passage_max_len"],
        index_subset=pathlib.Path(screen["index_dir"]).name,
        index_dim=screen["index_dim"],
        index_num_docs=screen["index_num_docs"],
        index_shard_sha256=tuple(screen["index_shard_sha256"]),
        top_k=args.top_k,
        full_document=True,
        corpus_shard_sha256=corpus_digests,
        bench_repo_commit=args.bench_repo_commit,
        conformance_sha256=sha256_hex(args.conformance.read_bytes()),
        effective_after="",     # the competence pilot fills this, and only it
        notes={
            "selection_rule": "smallest size with dev gold Recall@100 >= 0.90 x max, "
                              "subject to a p95 encode-latency ceiling",
            "dev_gold_recall_at_100": screen["recall"]["gold"]["recall@100"],
            "dev_evidence_recall_at_100": screen["recall"]["evidence"]["recall@100"],
            "encode_p50_ms": screen["encode_latency_ms"]["p50"],
            "encode_p95_ms": screen["encode_latency_ms"]["p95"],
            "latency_ceiling_relaxed_to_measured_p95": screen["encode_latency_ms"]["p95"],
            "latency_ceiling_original_ms": 100,
            "serving": "CPU only, pinned cpuset, one encoder instance per lane",
        },
    )
    body = write_freeze(freeze, args.out)
    print(f"retrieval freeze written: {args.out}")
    print(f"  encoder  {freeze.encoder_repo} @ {freeze.encoder_revision[:12]} ({freeze.encoder_dtype})")
    print(f"  index    {freeze.index_subset} dim {freeze.index_dim}, {freeze.index_num_docs} docs")
    print(f"  top_k    {freeze.top_k}, full documents")
    print(f"  digest   {body['digest']}")
    print(f"  effective: {freeze.effective} (competence pilot has not recorded a digest)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
