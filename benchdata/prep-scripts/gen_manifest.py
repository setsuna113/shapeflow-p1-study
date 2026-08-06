#!/usr/bin/env python
"""Build prep-scripts/MANIFEST.json describing every data asset.

One entry per asset: source URL, pinned revision (git commit / HF dataset sha),
file list with sha256, record counts, download date, licence, and the HF
endpoint actually used.

Idempotent: recomputing over unchanged files yields the same manifest except for
`generated_utc`. Hashing ~7 GB takes a few minutes.
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess

B = "/storage/sata/shapeflow/benchdata"
HF_ENDPOINT = "https://hf-mirror.com"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_entries(root, rel_paths):
    out = []
    for rel in sorted(rel_paths):
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            continue
        out.append({"path": rel, "bytes": os.path.getsize(p), "sha256": sha256_file(p)})
    return out


def list_files(root, skip_hidden=True, exts=None):
    rels = []
    for dirpath, dirnames, filenames in os.walk(root):
        if skip_hidden:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fn in filenames:
            if skip_hidden and fn.startswith("."):
                continue
            if exts and not any(fn.endswith(e) for e in exts):
                continue
            rels.append(os.path.relpath(os.path.join(dirpath, fn), root))
    return rels


def hf_revision(repo_id, repo_type="dataset"):
    """Read the pinned revision from the local HF cache ref."""
    slug = f"{repo_type}s--" + repo_id.replace("/", "--")
    ref = os.path.join("/root/.cache/huggingface/hub", slug, "refs", "main")
    if os.path.exists(ref):
        return open(ref).read().strip()
    return None


def load_json(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def mtime_date(path):
    if not os.path.exists(path):
        return None
    return datetime.datetime.utcfromtimestamp(
        os.path.getmtime(path)).strftime("%Y-%m-%d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(B, "prep-scripts", "MANIFEST.json"))
    args = ap.parse_args()

    stats = load_json(os.path.join(B, "browsecomp-plus", "bcplus_stats.json")) or {}
    smoke = load_json(os.path.join(B, "browsecomp-plus", "smoke_bm25.json")) or {}
    splits = load_json(os.path.join(B, "splits", "splits_manifest.json")) or {}
    rq = load_json(os.path.join(B, "drgym", "researchy_stats.json")) or {}
    probe = load_json(os.path.join(B, "drgym", "api_probe", "probe_summary.json")) or {}
    bc_fetch = load_json(os.path.join(B, "browsecomp-plus", "repo.fetch.json")) or {}
    drb_fetch = load_json(os.path.join(B, "drb", "repo.fetch.json")) or {}
    trec = load_json(os.path.join(B, "browsecomp-plus", "bm25_treceval.json")) or {}

    m = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
                                 .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base_dir": B,
        "documented_path_symlink": "/data/shapeflow -> /storage/sata/shapeflow",
        "hf_endpoint_used": HF_ENDPOINT,
        "hf_endpoint_note": ("huggingface.co is unreachable from this host "
                             "(TCP timeout); all HF traffic used the mirror above."),
        "assets": {},
    }

    print("hashing BrowseComp-Plus repo ...")
    m["assets"]["browsecomp_plus_repo"] = {
        "kind": "git repo (fetched as SHA-pinned tarball)",
        "source_url": "https://github.com/texttron/BrowseComp-Plus",
        "commit": bc_fetch.get("commit"),
        "tarball_url": bc_fetch.get("tarball_url"),
        "tarball_sha256": bc_fetch.get("tarball_sha256"),
        "fetched_utc": bc_fetch.get("fetched_utc"),
        "license": "MIT",
        "local_path": f"{B}/browsecomp-plus/repo",
        "note": bc_fetch.get("method"),
    }

    print("hashing BC+ corpus ...")
    corpus_root = os.path.join(B, "browsecomp-plus", "corpus")
    m["assets"]["browsecomp_plus_corpus"] = {
        "kind": "HF dataset",
        "source_url": "https://huggingface.co/datasets/Tevatron/browsecomp-plus-corpus",
        "hf_revision": hf_revision("Tevatron/browsecomp-plus-corpus"),
        "hf_endpoint": HF_ENDPOINT,
        "license": "MIT (per upstream repo)",
        "download_date": mtime_date(corpus_root),
        "num_documents": stats.get("corpus", {}).get("num_documents"),
        "local_path": corpus_root,
        "files": file_entries(corpus_root, list_files(corpus_root)),
    }

    print("hashing BC+ queries/qrels (decrypted) ...")
    data_root = os.path.join(B, "browsecomp-plus")
    dec = os.path.join(data_root, "data", "browsecomp_plus_decrypted.jsonl")
    qrel_root = os.path.join(data_root, "repo", "topics-qrels")
    m["assets"]["browsecomp_plus_queries"] = {
        "kind": "HF dataset (obfuscated upstream, decrypted locally)",
        "source_url": "https://huggingface.co/datasets/Tevatron/browsecomp-plus",
        "hf_revision": hf_revision("Tevatron/browsecomp-plus"),
        "hf_endpoint": HF_ENDPOINT,
        "license": "MIT",
        "download_date": mtime_date(dec),
        "num_queries": stats.get("queries", {}).get("num_queries"),
        "evidence_docs_per_query": stats.get("evidence_docs_per_query"),
        "gold_docs_per_query": stats.get("gold_docs_per_query"),
        "negative_docs_per_query": stats.get("negative_docs_per_query"),
        "qrel_files": stats.get("qrel_files"),
        "decryption": ("official transform_decrypt() imported verbatim from "
                       "scripts_build_index/decrypt_dataset.py; canary key unchanged"),
        "CONTAINS_ANSWERS": True,
        "handling": ("decrypted JSONL holds gold answers - kept out of git, "
                     "never printed to logs"),
        "files": file_entries(data_root, [
            os.path.relpath(dec, data_root),
            os.path.relpath(os.path.join(qrel_root, "qrel_evidence.txt"), data_root),
            os.path.relpath(os.path.join(qrel_root, "qrel_golds.txt"), data_root),
            os.path.relpath(os.path.join(qrel_root, "queries.tsv"), data_root),
        ]),
    }

    print("hashing BC+ indexes (this is the slow part) ...")
    idx_root = os.path.join(B, "browsecomp-plus", "indexes")
    idx_files = file_entries(idx_root, list_files(idx_root))
    per_index = {}
    for f in idx_files:
        top = f["path"].split("/")[0]
        d = per_index.setdefault(top, {"files": 0, "bytes": 0})
        d["files"] += 1
        d["bytes"] += f["bytes"]
    m["assets"]["browsecomp_plus_indexes"] = {
        "kind": "HF dataset (prebuilt indexes)",
        "source_url": "https://huggingface.co/datasets/Tevatron/browsecomp-plus-indexes",
        "hf_revision": hf_revision("Tevatron/browsecomp-plus-indexes"),
        "hf_endpoint": HF_ENDPOINT,
        "license": "MIT (per upstream repo)",
        "download_date": mtime_date(idx_root),
        "subsets": per_index,
        "dense_indexes_downloaded": True,
        "dense_note": ("all three Qwen3-Embedding indexes fetched; total index "
                       "footprint ~5 GB, far below the 50 GB threshold in the brief"),
        "lucene_num_docs": smoke.get("index_num_docs"),
        "local_path": idx_root,
        "files": idx_files,
    }

    m["assets"]["bm25_smoke_test"] = {
        "kind": "derived result",
        "script": "prep-scripts/smoke_bc_plus.py",
        "n_queries_sampled": smoke.get("n_queries_sampled"),
        "seed": smoke.get("seed"),
        "k": smoke.get("k"),
        "macro_recall": smoke.get("macro_recall"),
        "latency_ms": smoke.get("latency_ms"),
    }
    if trec:
        m["assets"]["bm25_full_treceval"] = trec

    print("hashing splits ...")
    split_root = os.path.join(B, "splits")
    m["assets"]["bcplus_splits"] = {
        "kind": "derived (ours, in git)",
        "script": "prep-scripts/make_splits.py",
        "seed": splits.get("seed"),
        "method": splits.get("method"),
        "dev_count": splits.get("dev", {}).get("count"),
        "test_count": splits.get("test", {}).get("count"),
        "dev_sha256": splits.get("dev", {}).get("sha256"),
        "test_sha256": splits.get("test", {}).get("sha256"),
        "balance_check": {"dev": {k: v for k, v in splits.get("dev", {}).items()
                                  if "docs_" in k},
                          "test": {k: v for k, v in splits.get("test", {}).items()
                                   if "docs_" in k}},
        "files": file_entries(split_root, list_files(split_root)),
    }

    print("hashing DeepResearch Bench ...")
    drb_root = os.path.join(B, "drb", "repo")
    qf = os.path.join(drb_root, "data", "prompt_data", "query.jsonl")
    n_q = zh = en = None
    if os.path.exists(qf):
        rows = [json.loads(l) for l in open(qf, encoding="utf-8") if l.strip()]
        n_q = len(rows)
        zh = sum(1 for r in rows if r.get("language") == "zh")
        en = sum(1 for r in rows if r.get("language") == "en")
    m["assets"]["deepresearch_bench"] = {
        "kind": "git repo (fetched as SHA-pinned tarball)",
        "source_url": "https://github.com/Ayanami0730/deep_research_bench",
        "commit": drb_fetch.get("commit"),
        "tarball_sha256": drb_fetch.get("tarball_sha256"),
        "fetched_utc": drb_fetch.get("fetched_utc"),
        "license": "Apache-2.0",
        "query_count": n_q,
        "query_language_split": {"zh": zh, "en": en},
        "judge_models_official": {"RACE": "gpt-5.5", "FACT": "gpt-5.4-mini",
                                  "note": "judge NOT run; needs external API key"},
        "entry_scripts": ["deepresearch_bench_race.py", "run_benchmark.sh",
                          "utils/extract.py", "utils/deduplicate.py",
                          "utils/validate.py", "utils/stat.py"],
        "usage_in_project": "guardrail only",
        "files": file_entries(drb_root, [os.path.relpath(qf, drb_root)]),
    }

    print("hashing Researchy Questions ...")
    rq_root = os.path.join(B, "drgym", "researchy_questions")
    m["assets"]["researchy_questions"] = {
        "kind": "HF dataset",
        "source_url": "https://huggingface.co/datasets/corbyrosset/researchy_questions",
        "hf_revision": hf_revision("corbyrosset/researchy_questions"),
        "hf_endpoint": HF_ENDPOINT,
        "license": "CDLA-Permissive-2.0",
        "download_date": mtime_date(rq_root),
        "total_records": rq.get("total_records"),
        "splits": {k: v.get("records") for k, v in (rq.get("splits") or {}).items()},
        "local_path": rq_root,
        "files": file_entries(rq_root, list_files(rq_root, exts=[".jsonl", ".md"])),
    }

    m["assets"]["deepresearchgym_api"] = {
        "kind": "remote API (no local data)",
        "project_page": "https://www.deepresearchgym.ai",
        "api_host": "https://clueweb22.us",
        "openapi": "https://clueweb22.us/openapi.json",
        "index_used": "fineweb",
        "clueweb22": "OUT OF SCOPE - not registered, not queried",
        "endpoints_present": ["GET /search", "GET /fineweb/search", "GET /health", "GET /docs"],
        "fetch_endpoint_exists": False,
        "auth": "x-api-key header",
        "status": "BLOCKED - 401 Invalid or missing API Key",
        "blocker_action": "email deepresearchgym@cmu.edu to request a free key",
        "network_reachable_from_sjtu": True,
        "probe": {k: probe.get(k) for k in
                  ("n_queries", "n_success", "latency_ms", "response_shape")},
        "docs": "drgym/README_api.md",
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(m, open(args.out, "w"), indent=2, ensure_ascii=False)

    total_files = sum(len(a.get("files", [])) for a in m["assets"].values())
    total_bytes = sum(f["bytes"] for a in m["assets"].values()
                      for f in a.get("files", []))
    print(f"\nwrote {args.out}")
    print(f"  {len(m['assets'])} assets, {total_files} files hashed, "
          f"{total_bytes/2**30:.2f} GB")


if __name__ == "__main__":
    main()
