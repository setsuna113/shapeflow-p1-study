#!/usr/bin/env python
"""Count and sample the Researchy Questions dataset."""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    out = {"splits": {}}
    samples = []
    for name in ("researchy_questions.train.jsonl", "researchy_questions.test.jsonl"):
        p = os.path.join(args.dir, name)
        n = 0
        first = None
        with open(p, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                n += 1
                if first is None:
                    first = json.loads(line)
                if name.startswith("researchy_questions.train") and i < 5:
                    r = json.loads(line)
                    samples.append({
                        "id": r.get("id"),
                        "question": r.get("question"),
                        "n_doc_ids": len(r.get("DocStream") or r.get("docs") or []),
                        "intrinsic_score": r.get("intrinsic_score"),
                    })
        out["splits"][name] = {
            "path": p,
            "records": n,
            "bytes": os.path.getsize(p),
            "fields": sorted(first.keys()) if first else [],
        }
        print(f"  {name}: {n} records  ({os.path.getsize(p)/2**20:.1f} MB)")
        if first:
            print(f"    fields: {sorted(first.keys())}")

    total = sum(v["records"] for v in out["splits"].values())
    out["total_records"] = total
    print(f"\n  TOTAL: {total} records")

    print("\n  === 5 sample questions (train) ===")
    for s in samples[:5]:
        q = (s["question"] or "")[:150]
        print(f"    [{s['id']}] {q}")
    out["samples"] = samples[:5]

    json.dump(out, open(args.out_json, "w"), indent=2, ensure_ascii=False)
    print(f"\n  wrote {args.out_json}")


if __name__ == "__main__":
    main()
