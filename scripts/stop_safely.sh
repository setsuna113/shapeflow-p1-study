#!/usr/bin/env bash
# Graceful stop (plan §17.4). Stops admission, lets the current attempt finish marking itself,
# flushes the ledger, and terminates only THIS run's process group -- never other processes on the
# shared host. Called by the systemd unit's ExecStop and usable by hand.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUN_PGID="${SHAPEFLOW_RUN_PGID:-}"

echo "stop_safely: requesting graceful shutdown at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# 1. Signal the coordinator to stop admitting new work (it watches for this sentinel).
touch "$REPO/runs/STOP_REQUESTED" 2>/dev/null || true

# 2. Give the current attempt a bounded window to reach a terminal ledger state and flush.
TIMEOUT="${STOP_TIMEOUT_SECONDS:-120}"
for _ in $(seq "$TIMEOUT"); do
  if [ -f "$REPO/runs/STOPPED_CLEAN" ]; then
    echo "stop_safely: coordinator confirmed clean stop"
    exit 0
  fi
  sleep 1
done

# 3. Timed out: terminate only this run's process group, if we were given it. Never pkill broadly.
if [ -n "$RUN_PGID" ]; then
  echo "stop_safely: timeout; sending TERM to process group $RUN_PGID (this run only)"
  kill -TERM "-$RUN_PGID" 2>/dev/null || true
else
  echo "stop_safely: timeout and no SHAPEFLOW_RUN_PGID set; NOT killing anything broadly" >&2
fi
