#!/usr/bin/env bash
# Download the three encoder sizes and screen each against the dev split. Idempotent.
#
# Designed to be launched detached and left alone:
#
#   nohup bash scripts/run_retrieval_screen.sh > logs/retrieval_screen.log 2>&1 &
#
# Every step is skip-if-done -- a cached model is not re-downloaded, a size whose result file
# exists is not re-run -- so re-launching after a disconnect resumes rather than restarts. The
# 8B pass takes tens of minutes and must survive a dropped connection.
#
# CPU only, on a pinned cpuset. The GPUs are reserved for the engine under measurement, and an
# encoder sharing one would spend SM time the serving measurement attributes to serving.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

BENCH="${SHAPEFLOW_BENCHDATA:-/storage/sata/shapeflow/benchdata}"
PY="${SHAPEFLOW_ENCODER_PYTHON:-$BENCH/.venv/bin/python}"
HF="${SHAPEFLOW_HF_CLI:-$BENCH/.venv/bin/hf}"
CPUSET="${SHAPEFLOW_ENCODER_CPUSET:-48-63}"
THREADS="${SHAPEFLOW_ENCODER_THREADS:-16}"

# hf-mirror proxies the Hub API but not HF's Xet storage, which 401s on cas-server.xethub.hf.co.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

QRELS="$BENCH/browsecomp-plus/repo/topics-qrels"
INDEXES="$BENCH/browsecomp-plus/indexes"
SPLIT="$BENCH/prep-scripts/splits/bcplus_dev.txt"
OUTDIR="$REPO/reports/retrieval_screen"
mkdir -p "$OUTDIR" logs

# tag : hf repo : index subdirectory
SIZES=(
  "0.6b:Qwen/Qwen3-Embedding-0.6B:qwen3-embedding-0.6b"
  "4b:Qwen/Qwen3-Embedding-4B:qwen3-embedding-4b"
  "8b:Qwen/Qwen3-Embedding-8B:qwen3-embedding-8b"
)

echo "=== retrieval screen starting $(date -u +%FT%TZ) ==="
echo "    dev split: $SPLIT ($(wc -l < "$SPLIT") queries)"
echo "    cpuset $CPUSET, $THREADS threads, CPU only"

for entry in "${SIZES[@]}"; do
  IFS=: read -r tag repo subdir <<< "$entry"

  if [ -f "$OUTDIR/$tag.json" ]; then
    echo "--- $tag: result exists, skipping"
    continue
  fi

  echo "--- $tag: ensuring $repo is cached $(date -u +%FT%TZ)"
  if ! "$HF" download "$repo" >/dev/null 2>&1; then
    echo "    download FAILED for $repo; skipping this size" >&2
    echo "{\"tag\":\"$tag\",\"error\":\"download failed\"}" > "$OUTDIR/$tag.error.json"
    continue
  fi

  echo "--- $tag: screening $(date -u +%FT%TZ)"
  taskset -c "$CPUSET" "$PY" scripts/retrieval_screen.py \
    --index-dir "$INDEXES/$subdir" \
    --model "$repo" \
    --tag "$tag" \
    --queries "$QRELS/queries.tsv" \
    --split "$SPLIT" \
    --qrel-golds "$QRELS/qrel_golds.txt" \
    --qrel-evidence "$QRELS/qrel_evidence.txt" \
    --threads "$THREADS" \
    --outdir "$OUTDIR" || echo "    screen FAILED for $tag (continuing)" >&2
done

echo "=== summarising $(date -u +%FT%TZ) ==="
"$PY" - "$OUTDIR" <<'PYEOF'
import json, pathlib, sys
outdir = pathlib.Path(sys.argv[1])
rows = []
for tag in ("0.6b", "4b", "8b"):
    path = outdir / f"{tag}.json"
    if not path.exists():
        print(f"  {tag}: MISSING"); continue
    b = json.loads(path.read_text())
    rows.append(b)
    g, e = b["recall"]["gold"], b["recall"]["evidence"]
    print(f"  {tag:5s} dim {b['index_dim']:5d} | gold R@5 {g['recall@5']}  R@100 {g['recall@100']}"
          f"  R@1000 {g['recall@1000']} | evid R@100 {e['recall@100']}"
          f" | encode p50 {b['encode_latency_ms']['p50']}ms p95 {b['encode_latency_ms']['p95']}ms")
if rows:
    (outdir / "SUMMARY.json").write_text(
        json.dumps({"sizes": rows}, indent=2, sort_keys=True) + "\n")
    print(f"  -> {outdir / 'SUMMARY.json'}")
PYEOF
echo "=== retrieval screen done $(date -u +%FT%TZ) ==="
