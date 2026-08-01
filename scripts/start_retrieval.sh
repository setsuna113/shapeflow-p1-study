#!/usr/bin/env bash
# Start the frozen BrowseComp-Plus retrieval service for one lane.
#
#   nohup bash scripts/start_retrieval.sh 0 > logs/retrieval-lane0.log 2>&1 &
#
# One service per lane, on disjoint cpusets. Not one shared service: campaign/sharding.py buys
# lane independence with separate engine epochs, and a single retrieval process behind all four
# lanes silently re-couples them through its request queue -- so a queueing effect would arrive
# later as a lane effect that the homogeneity gate has already blessed.
#
# CPU only, and the service refuses --device anything else. Each 4090 runs vLLM at
# gpu_memory_utilization 0.90 and S1's headline number is that engine's sustainable arrival
# rate; an encoder sharing the card spends SM time the measurement attributes to serving, and
# nothing downstream can subtract it afterwards.
#
# The encoder recipe is NOT passed here. It is read out of the conformance report, so the recipe
# that is served is by construction the recipe that was validated against the shipped index.
set -uo pipefail

LANE="${1:-0}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

BENCH="${SHAPEFLOW_BENCHDATA:-/storage/sata/shapeflow/benchdata}"
# The study's own venv deliberately has no torch: its dependency closure is pinned to the
# vendor's lock for everything that touches the graph. The encoder lives in the benchdata venv
# and reaches shapeflow.retrieval through PYTHONPATH, which needs nothing but numpy and pyarrow.
PY="${SHAPEFLOW_ENCODER_PYTHON:-$BENCH/.venv/bin/python}"
PORT="${SHAPEFLOW_RETRIEVAL_PORT:-$((8710 + LANE))}"
THREADS="${SHAPEFLOW_ENCODER_THREADS:-16}"
# Cores 0-47 are the four engines (12 each); 48-63 is retrieval, four cores per lane.
CPUSET="${SHAPEFLOW_ENCODER_CPUSET:-$((48 + LANE * 4))-$((51 + LANE * 4))}"

CONFORMANCE="${SHAPEFLOW_CONFORMANCE:-$REPO/reports/RETRIEVAL_CONFORMANCE_Qwen3-Embedding-8B_bfloat16.json}"
INDEX_DIR="${SHAPEFLOW_INDEX_DIR:-$BENCH/browsecomp-plus/indexes/qwen3-embedding-8b}"
CORPUS_DIR="${SHAPEFLOW_CORPUS_DIR:-$BENCH/browsecomp-plus/corpus/data}"
FREEZE="${SHAPEFLOW_RETRIEVAL_FREEZE:-$REPO/protocol/retrieval_freeze.json}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=0
export TZ=UTC

echo "=== retrieval lane $LANE on port $PORT, cpuset $CPUSET, $THREADS threads ==="
echo "    index      $INDEX_DIR"
echo "    corpus     $CORPUS_DIR"
echo "    conformance $CONFORMANCE"

exec "$PY" -m shapeflow.retrieval.service \
  --index-dir "$INDEX_DIR" \
  --corpus-dir "$CORPUS_DIR" \
  --conformance "$CONFORMANCE" \
  --freeze "$FREEZE" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --threads "$THREADS" \
  --cpuset "$CPUSET"
