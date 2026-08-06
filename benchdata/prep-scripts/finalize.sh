#!/usr/bin/env bash
# Wait for the BM25 full run, evaluate it with the benchmark's own trec_eval,
# then regenerate MANIFEST.json.
set -uo pipefail
B=/storage/sata/shapeflow/benchdata
export OPENAI_API_KEY=not-used-bm25-only
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
PY="$B/.venv/bin/python"
RUN="$B/browsecomp-plus/runs/bm25.full.trec"

echo "[$(date -u +%FT%TZ)] waiting for bm25 full run ..."
while pgrep -f "bm25_full_[r]un" >/dev/null 2>&1; do sleep 20; done
echo "[$(date -u +%FT%TZ)] bm25 run finished; $(wc -l < "$RUN") lines"

evaluate () {   # $1=qrel path  $2=label
  "$PY" -m pyserini.eval.trec_eval -c -m recall.5,10,100,1000 -m ndcg_cut.10 \
        "$1" "$RUN" 2>/dev/null | grep -E '^(recall|ndcg)' \
    | awk -v lbl="$2" '{printf "%s\t%s\t%s\n", lbl, $1, $3}'
}

echo "[$(date -u +%FT%TZ)] trec_eval ..."
{ evaluate "$B/browsecomp-plus/repo/topics-qrels/qrel_evidence.txt" evidence
  evaluate "$B/browsecomp-plus/repo/topics-qrels/qrel_golds.txt"    gold
} > /tmp/treceval.tsv
cat /tmp/treceval.tsv

"$PY" - <<'PY'
import json
res = {"evidence": {}, "gold": {}}
for line in open("/tmp/treceval.tsv"):
    parts = line.split()
    if len(parts) == 3:
        lbl, metric, val = parts
        res[lbl][metric] = float(val)
out = {
    "kind": "derived result",
    "script": "prep-scripts/bm25_full_run.py + pyserini.eval.trec_eval",
    "retriever": "BM25 (prebuilt Tevatron index)",
    "n_queries": 830, "k": 1000,
    "metrics": res,
    "note": "official evaluation path from the BrowseComp-Plus README; "
            "validates the 20-query smoke test against all 830 queries",
}
json.dump(out, open("/storage/sata/shapeflow/benchdata/browsecomp-plus/bm25_treceval.json", "w"), indent=2)
print("wrote bm25_treceval.json")
PY

echo "[$(date -u +%FT%TZ)] generating MANIFEST.json (hashing ~7 GB) ..."
"$PY" "$B/prep-scripts/gen_manifest.py"
echo "[$(date -u +%FT%TZ)] FINALIZE DONE"
