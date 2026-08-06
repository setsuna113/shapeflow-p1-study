#!/usr/bin/env bash
# Idempotent download of BrowseComp-Plus corpus + prebuilt indexes.
# huggingface-cli is natively resumable and skips files already present with
# matching size/etag, so re-running is cheap and safe.
set -euo pipefail
B=/storage/sata/shapeflow/benchdata
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"   # huggingface.co is unreachable from sjtu
HF="$B/.venv/bin/hf"

echo "[$(date -u +%FT%TZ)] endpoint=$HF_ENDPOINT"
echo "[$(date -u +%FT%TZ)] corpus ..."
"$HF" download Tevatron/browsecomp-plus-corpus --repo-type=dataset --local-dir "$B/browsecomp-plus/corpus"

for sub in bm25 qwen3-embedding-0.6b qwen3-embedding-4b qwen3-embedding-8b; do
  echo "[$(date -u +%FT%TZ)] index: $sub ..."
  "$HF" download Tevatron/browsecomp-plus-indexes --repo-type=dataset \
        --include="$sub/*" --local-dir "$B/browsecomp-plus/indexes"
done
echo "[$(date -u +%FT%TZ)] DONE"
