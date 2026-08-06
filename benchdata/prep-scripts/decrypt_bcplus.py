#!/usr/bin/env python
"""Decrypt the BrowseComp-Plus query/qrel dataset.

Why this wrapper exists (see MANIFEST "deviations"): the official
scripts_build_index/decrypt_dataset.py calls
``load_dataset("Tevatron/browsecomp-plus", split="test")``. On this host
huggingface.co is unreachable and, against the hf-mirror endpoint, the
`datasets` Hub-resolution path hangs indefinitely; with HF_HUB_OFFLINE=1 it
refuses to resolve from a warm cache at all. So we download the repo with
`hf download` (which works) and read the cached parquet directly.

The decryption itself is NOT reimplemented: `transform_decrypt` and the canary
constant are imported from the official script, so output is byte-identical to
what the official path would produce.

Idempotent: if both outputs already exist with the expected record count, exits
without rewriting.
"""
import argparse
import glob
import importlib.util
import json
import os
import sys

EXPECTED_RECORDS = 830


def load_official(repo_dir):
    """Import the official decrypt module so we reuse its exact logic."""
    path = os.path.join(repo_dir, "scripts_build_index", "decrypt_dataset.py")
    spec = importlib.util.spec_from_file_location("official_decrypt", path)
    mod = importlib.util.module_from_spec(spec)
    # The official module imports `datasets` at top level; that is fine, we just
    # never call its main().
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--snapshot-dir", required=True,
                    help="HF hub snapshot dir holding data/test-*.parquet")
    ap.add_argument("--output", required=True)
    ap.add_argument("--generate-tsv", required=True)
    args = ap.parse_args()

    if os.path.exists(args.output) and os.path.exists(args.generate_tsv):
        n = sum(1 for _ in open(args.output, encoding="utf-8"))
        if n == EXPECTED_RECORDS:
            print(f"[skip] {args.output} already has {n} records")
            return

    official = load_official(args.repo_dir)

    import pyarrow as pa
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(args.snapshot_dir, "data", "test-*.parquet")))
    if not files:
        sys.exit(f"FATAL: no test-*.parquet under {args.snapshot_dir}/data")
    table = pq.read_table(files[0]) if len(files) == 1 else pa.concat_tables(
        [pq.read_table(f) for f in files])
    records = table.to_pylist()
    print(f"Processing {len(records)} records from {len(files)} parquet file(s)...")

    skip_keys = {"query_id"}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as jsonl_out, \
         open(args.generate_tsv, "w", encoding="utf-8") as tsv_out:
        for rec in records:
            # mirror the official round-trip through json exactly
            row = json.loads(json.dumps(rec, ensure_ascii=False))
            decrypted_row = official.transform_decrypt(
                row, official.DEFAULT_CANARY, skip_keys)
            json.dump(decrypted_row, jsonl_out, ensure_ascii=False)
            jsonl_out.write("\n")
            qid = str(decrypted_row.get("query_id", ""))
            q = str(decrypted_row.get("query", "")).replace("\t", " ")
            tsv_out.write(f"{qid}\t{q}\n")

    n = sum(1 for _ in open(args.output, encoding="utf-8"))
    print(f"Wrote {n} records -> {args.output}")
    print(f"Wrote TSV -> {args.generate_tsv}")
    if n != EXPECTED_RECORDS:
        sys.exit(f"FATAL: expected {EXPECTED_RECORDS} records, got {n}")


if __name__ == "__main__":
    main()
