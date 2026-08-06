#!/bin/bash
# Launch a BC+ campaign on one lane, detached, log on disk.
#   bash /storage/nvme/ops/bcplus.sh <lane> <logname> <args...>
set -u
REPO=/storage/nvme/shapeflow-p1-study
LANE="$1"; shift
LOG="$REPO/logs/$1.log"; shift
case "$LANE" in
  0) GPU=GPU-ef013951-e496-78da-da70-a5a289dcc634 ;;
  1) GPU=GPU-d1f5d8c4-2f0f-45f7-0574-ac90018440db ;;
  *) echo "unknown lane $LANE" >&2; exit 1 ;;
esac
BINDING=$(python3 -c "import json;print(json.load(open(\"/storage/nvme/shapeflow-data/approvals/launch_approval.json\"))[\"binding_sha256\"])")
cd "$REPO"
setsid runuser -u sfrunner -- env \
  USER=sfrunner LOGNAME=sfrunner HOME=/tmp SHAPEFLOW_REPO="$REPO" \
  SHAPEFLOW_DATA_ROOT=/storage/nvme/shapeflow-data \
  SHAPEFLOW_APPROVAL_FILE=/storage/nvme/shapeflow-data/approvals/launch_approval.json \
  SHAPEFLOW_BENCHDATA=/storage/sata/shapeflow/benchdata \
  SHAPEFLOW_LANE="$LANE" \
  SHAPEFLOW_GPU_UUID="$GPU" \
  SHAPEFLOW_ENGINE_EPOCH_FILE=/run/shapeflow-vllm-causal-native-lane$LANE/engine_epoch \
  PYTHONHASHSEED=0 TZ=UTC \
  "$REPO/.venv/bin/shapeflow" "$@" \
  --retrieval-url "http://127.0.0.1:$((8710 + LANE))" \
  --shard "$LANE" --shards 2 \
  --resume --protocol-sha "$BINDING" \
  > "$LOG" 2>&1 < /dev/null &
echo "lane $LANE launched, log=$LOG binding=${BINDING:0:12}"
