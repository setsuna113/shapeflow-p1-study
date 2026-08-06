#!/bin/bash
# Run a steward-identity command (bcplus-competence, freeze-approval). Foreground, logged.
#   bash /storage/nvme/ops/steward.sh <logname> <shapeflow args...>
set -u
REPO=/storage/nvme/shapeflow-p1-study
LOG="$REPO/logs/$1.log"; shift
cd "$REPO"
runuser -u sfsteward -- env \
  USER=sfsteward LOGNAME=sfsteward HOME=/tmp SHAPEFLOW_REPO="$REPO" \
  SHAPEFLOW_DATA_ROOT=/storage/nvme/shapeflow-data \
  SHAPEFLOW_APPROVAL_FILE=/storage/nvme/shapeflow-data/approvals/launch_approval.json \
  SHAPEFLOW_BENCHDATA=/storage/sata/shapeflow/benchdata \
  PYTHONHASHSEED=0 TZ=UTC \
  "$REPO/.venv/bin/shapeflow" "$@" 2>&1 | tee "$LOG"
