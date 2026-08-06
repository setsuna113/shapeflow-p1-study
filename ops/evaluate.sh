#!/bin/bash
# Run an evaluator-identity command (grade-bcplus, bcplus-competence). Foreground, logged.
#   bash /storage/nvme/ops/evaluate.sh <logname> <shapeflow args...>
set -u
REPO=/storage/nvme/shapeflow-p1-study
LOG="$REPO/logs/$1.log"; shift
BINDING=$(python3 -c "import json;print(json.load(open('/storage/nvme/shapeflow-data/approvals/launch_approval.json'))['binding_sha256'])")
cd "$REPO"
runuser -u sfevaluator -- env \
  USER=sfevaluator LOGNAME=sfevaluator HOME=/tmp SHAPEFLOW_REPO="$REPO" \
  SHAPEFLOW_DATA_ROOT=/storage/nvme/shapeflow-data \
  SHAPEFLOW_APPROVAL_FILE=/storage/nvme/shapeflow-data/approvals/launch_approval.json \
  SHAPEFLOW_BENCHDATA=/storage/sata/shapeflow/benchdata \
  SHAPEFLOW_LANE=0 \
  PYTHONHASHSEED=0 TZ=UTC \
  "$REPO/.venv/bin/shapeflow" "$@" 2>&1 | tee "$LOG"
echo "binding=${BINDING:0:12} log=$LOG"
