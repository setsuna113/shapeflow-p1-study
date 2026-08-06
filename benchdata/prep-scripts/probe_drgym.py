#!/usr/bin/env python
"""Connectivity + latency probe for the DeepResearchGym search API (FineWeb only).

Spec (recovered from the deepresearchgym.ai app bundle, which embeds the
official "Example Usage" snippet):
    GET https://clueweb22.us/fineweb/search?query=<q>&k=<n>
    -> {"results": [<base64(JSON doc)>, ...]}
    FineWeb doc fields: url, dump, language, id, text
    FineWeb requires NO API key at present; ClueWeb22 endpoints need X-API-Key.

This script deliberately touches ONLY the FineWeb endpoint. ClueWeb22 is out of
scope for this project and is never queried.

Archives truncated response samples so the response shape is on record without
storing bulk corpus text.
"""
import argparse
import base64
import json
import os
import statistics
import time

import requests

FINEWEB_SEARCH = "https://clueweb22.us/fineweb/search"

QUERIES = [
    "what is the capital of portugal",
    "transformer architecture attention mechanism",
    "vLLM continuous batching throughput",
    "history of the Hanseatic League trade routes",
    "CRISPR gene editing off-target effects",
    "monetary policy quantitative tightening 2024",
    "norwegian forest cat origin",
    "photovoltaic perovskite stability degradation",
    "Kubernetes horizontal pod autoscaler metrics",
    "archaeological evidence minoan eruption thera",
]


def probe_search(q, k, timeout, archive_dir, idx):
    t0 = time.perf_counter()
    err = None
    n_results = 0
    status = None
    body_prefix = None
    try:
        r = requests.get(FINEWEB_SEARCH, params={"query": q, "k": k}, timeout=timeout)
        status = r.status_code
        dt = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            body_prefix = r.text[:300]
            err = f"HTTP {r.status_code}: {body_prefix}"
        if r.status_code == 200:
            payload = r.json()
            results = payload.get("results", [])
            n_results = len(results)
            if idx == 0 and results:
                # archive one decoded sample so the response schema is on record
                doc = json.loads(base64.b64decode(results[0]).decode("utf-8"))
                trunc = {k2: (v[:600] + " ...[TRUNCATED]" if isinstance(v, str) and len(v) > 600 else v)
                         for k2, v in doc.items()}
                with open(os.path.join(archive_dir, "sample_search_decoded_doc.json"), "w") as f:
                    json.dump({"query": q, "k": k, "doc_fields": sorted(doc.keys()),
                               "decoded_doc_truncated": trunc}, f, indent=2, ensure_ascii=False)
                with open(os.path.join(archive_dir, "sample_search_raw_envelope.json"), "w") as f:
                    json.dump({"query": q, "k": k,
                               "top_level_keys": sorted(payload.keys()),
                               "num_results": n_results,
                               "results_are_base64_encoded_json": True,
                               "first_result_b64_prefix": results[0][:120] + "...",
                               }, f, indent=2)
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000
        err = f"{type(e).__name__}: {e}"
    return {"query": q, "http_status": status, "latency_ms": round(dt, 1),
            "n_results": n_results, "error": err, "body_prefix": body_prefix}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=90)
    ap.add_argument("--archive-dir", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()
    os.makedirs(args.archive_dir, exist_ok=True)

    print(f"  endpoint: {FINEWEB_SEARCH}  (FineWeb only; ClueWeb22 never queried)")
    print(f"  {len(QUERIES)} queries, k={args.k}\n")

    recs = []
    for i, q in enumerate(QUERIES):
        rec = probe_search(q, args.k, args.timeout, args.archive_dir, i)
        recs.append(rec)
        flag = "ok " if rec["http_status"] == 200 and not rec["error"] else "ERR"
        print(f"    [{flag}] {rec['latency_ms']:8.1f} ms  n={rec['n_results']}  {q[:46]}")
        if rec["error"]:
            print(f"           {rec['error'][:160]}")

    ok = [r for r in recs if r["http_status"] == 200 and not r["error"]]
    summary = {
        "endpoint": FINEWEB_SEARCH,
        "index": "fineweb",
        "auth": "none required for FineWeb (ClueWeb22 would need X-API-Key; not used)",
        "method": "GET",
        "k": args.k,
        "n_queries": len(QUERIES),
        "n_success": len(ok),
        "response_shape": "{'results': [base64(JSON doc), ...]}; FineWeb doc fields: url, dump, language, id, text",
    }
    if ok:
        lat = sorted(r["latency_ms"] for r in ok)
        summary["latency_ms"] = {
            "mean": round(statistics.mean(lat), 1),
            "median": round(statistics.median(lat), 1),
            "min": lat[0], "max": lat[-1],
            "p90": lat[min(int(len(lat) * 0.9), len(lat) - 1)],
        }
        print(f"\n  success {len(ok)}/{len(QUERIES)}   latency mean={summary['latency_ms']['mean']}ms "
              f"median={summary['latency_ms']['median']}ms min={lat[0]}ms max={lat[-1]}ms")
    else:
        print(f"\n  *** ALL {len(QUERIES)} PROBES FAILED ***")
    summary["per_query"] = recs

    # ---- /fetch endpoint discovery: documented in the task brief, absent from
    # the official example; probe candidate paths and record what actually exists.
    fetch_probe = {}
    for path in ("/fineweb/fetch", "/fetch", "/fineweb/doc", "/docs"):
        url = "https://clueweb22.us" + path
        try:
            r = requests.get(url, timeout=30)
            fetch_probe[path] = {"http_status": r.status_code,
                                 "body_prefix": r.text[:200]}
        except Exception as e:
            fetch_probe[path] = {"error": f"{type(e).__name__}: {e}"}
    summary["fetch_endpoint_probe"] = fetch_probe
    print("\n  /fetch discovery:")
    for p, v in fetch_probe.items():
        print(f"    {p:<18} {v.get('http_status', v.get('error'))}  {str(v.get('body_prefix',''))[:80]}")

    json.dump(summary, open(args.out_json, "w"), indent=2, ensure_ascii=False)
    print(f"\n  wrote {args.out_json}")


if __name__ == "__main__":
    main()
