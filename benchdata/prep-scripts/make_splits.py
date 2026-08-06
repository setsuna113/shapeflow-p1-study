#!/usr/bin/env python
"""Freeze the BrowseComp-Plus dev/test split.

dev 530 / test 300 (830 total), seed 20260727.

Stratification: the decrypted records expose only
{query_id, query, answer, gold_docs, negative_docs, evidence_docs} — there is no
topic / answer-type / category field to stratify on, so the split is PURE
RANDOM. As a sanity check we report how the (incidental) gold- and
evidence-document counts landed across the two sides; they are not used to
drive the split.

Byte-reproducibility: ids are sorted into a canonical order before shuffling,
the RNG is explicitly seeded, and output is written with LF newlines in sorted
order. Re-running overwrites with identical bytes.
"""
import argparse
import hashlib
import json
import random
import statistics


def canonical_sort(ids):
    # numeric ids sort numerically; anything else sorts lexicographically after
    return sorted(ids, key=lambda q: (0, int(q), "") if q.isdigit() else (1, 0, q))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_split(path, ids):
    # explicit newline="\n" so the bytes never depend on platform
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for q in canonical_sort(ids):
            f.write(f"{q}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decrypted", required=True)
    ap.add_argument("--out-dev", required=True)
    ap.add_argument("--out-test", required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--n-dev", type=int, default=530)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260727)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.decrypted, encoding="utf-8") if l.strip()]
    by_id = {str(r["query_id"]): r for r in rows}
    ids = canonical_sort(by_id.keys())
    assert len(ids) == len(rows), "duplicate query_id in decrypted data"
    if len(ids) != args.n_dev + args.n_test:
        raise SystemExit(f"FATAL: {len(ids)} queries but n_dev+n_test="
                         f"{args.n_dev + args.n_test}")

    shuffled = list(ids)
    random.Random(args.seed).shuffle(shuffled)
    dev, test = shuffled[:args.n_dev], shuffled[args.n_dev:]
    assert not (set(dev) & set(test)), "dev/test overlap"
    assert set(dev) | set(test) == set(ids), "split does not cover all queries"

    write_split(args.out_dev, dev)
    write_split(args.out_test, test)

    def side_stats(split_ids):
        gold = [len(by_id[q].get("gold_docs") or []) for q in split_ids]
        evid = [len(by_id[q].get("evidence_docs") or []) for q in split_ids]
        return {
            "n": len(split_ids),
            "gold_docs_mean": round(statistics.mean(gold), 3),
            "gold_docs_median": statistics.median(gold),
            "evidence_docs_mean": round(statistics.mean(evid), 3),
            "evidence_docs_median": statistics.median(evid),
        }

    manifest = {
        "seed": args.seed,
        "method": "pure random (no topic/answer-type field exists to stratify on)",
        "source": args.decrypted,
        "total_queries": len(ids),
        "dev": {"path": args.out_dev, "count": len(dev),
                "sha256": sha256_file(args.out_dev), **side_stats(dev)},
        "test": {"path": args.out_test, "count": len(test),
                 "sha256": sha256_file(args.out_test), **side_stats(test)},
        "balance_note": "gold/evidence doc counts are reported only as a sanity "
                        "check that the random split did not skew difficulty; "
                        "they did not drive the split.",
    }
    json.dump(manifest, open(args.out_manifest, "w"), indent=2)

    print(f"  total {len(ids)}  ->  dev {len(dev)} / test {len(test)}  (seed={args.seed})")
    for side in ("dev", "test"):
        m = manifest[side]
        print(f"    {side:<5} n={m['count']:<4} sha256={m['sha256']}")
        print(f"          gold/query mean={m['gold_docs_mean']:<6} median={m['gold_docs_median']:<4} "
              f"evidence/query mean={m['evidence_docs_mean']:<6} median={m['evidence_docs_median']}")
    print(f"  wrote {args.out_manifest}")


if __name__ == "__main__":
    main()
