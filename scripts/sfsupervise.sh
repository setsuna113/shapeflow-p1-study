#!/usr/bin/env bash
# A supervisor for hosts where systemd is not PID 1 (plan §17.2).
#
# The run host is a container: `systemctl` cannot operate, so the system-level units this repo
# ships cannot be started. The protocol permits an alternative supervisor *only* if it passes the
# same fault test as the systemd unit -- restart the coordinator with `--resume` after a kill,
# give up after three crashes in thirty minutes, and never let the ledger accept the same work
# twice. tests/integration/test_supervisor_faults.py is that test; this script is what it tests.
#
# It is not a bare `nohup`. It holds a singleton lock, owns a process group, drops the workload
# to an unprivileged identity, bounds its own restarts, and writes a BLOCKED report when it gives
# up instead of looping forever.
#
# Usage: sfsupervise.sh <name> <role> <repo> <data_root> -- <command...>
set -euo pipefail

NAME="$1"; ROLE="$2"; REPO="$3"; DATA_ROOT="$4"; shift 4
[ "${1:-}" = "--" ] && shift

RUN_DIR="${SHAPEFLOW_RUN_DIR:-/run/shapeflow}"
LOG_DIR="$REPO/logs"
LOCK="$RUN_DIR/$NAME.lock"
PIDFILE="$RUN_DIR/$NAME.pid"
LOG="$LOG_DIR/$NAME.log"
RESTART_SEC="${SHAPEFLOW_RESTART_SEC:-30}"
BURST="${SHAPEFLOW_START_LIMIT_BURST:-3}"
INTERVAL="${SHAPEFLOW_START_LIMIT_INTERVAL:-1800}"

mkdir -p "$RUN_DIR" "$LOG_DIR"

# Singleton: exactly one supervisor per name, or two coordinators would claim the same work.
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "sfsupervise[$NAME]: another supervisor holds $LOCK" >&2
  exit 1
fi

echo $$ > "$PIDFILE"
say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] sfsupervise[$NAME] $*" >> "$LOG"; }

cleanup() {
  say "stopping; terminating this run's process group only"
  [ -n "${CHILD_PGID:-}" ] && kill -TERM "-$CHILD_PGID" 2>/dev/null || true
  rm -f "$PIDFILE"
}
# A signal handler that cleans up but does not exit leaves the supervisor waiting: bash resumes
# the interrupted `wait` and the loop carries on, so a stop request would look like it worked and
# the coordinator would still be running.
on_signal() { cleanup; trap - EXIT; exit 143; }
trap cleanup EXIT
trap on_signal INT TERM

starts=()
say "starting: role=$ROLE cmd=$*"

while true; do
  now=$(date +%s)
  # Keep only the starts inside the window, then apply the ceiling.
  kept=()
  for t in "${starts[@]:-}"; do
    [ -n "$t" ] && [ $((now - t)) -lt "$INTERVAL" ] && kept+=("$t")
  done
  starts=("${kept[@]:-}")
  live=0
  for t in "${starts[@]:-}"; do [ -n "$t" ] && live=$((live + 1)); done
  if [ "$live" -ge "$BURST" ]; then
    say "giving up: $live starts within ${INTERVAL}s"
    {
      echo "# BLOCKED: REPEATED_CRASH"
      echo
      echo "\`$NAME\` restarted $live times within ${INTERVAL}s and the supervisor stopped."
      echo "Finished work is kept; the ledger holds every attempt. Inspect before restarting:"
      echo
      echo '    shapeflow-p1 status'
      echo "    tail -100 $LOG"
    } > "$REPO/reports/BLOCKED_REPEATED_CRASH.md"
    exit 1
  fi
  starts+=("$now")

  # setsid gives the child its own process group, so stop_safely can signal exactly this run
  # and nothing else on a shared host.
  setsid runuser -u "$ROLE" -- env \
      USER="$ROLE" LOGNAME="$ROLE" HOME="/tmp" \
      SHAPEFLOW_REPO="$REPO" SHAPEFLOW_DATA_ROOT="$DATA_ROOT" \
      PYTHONHASHSEED=0 TZ=UTC PYTHONUNBUFFERED=1 \
      "$@" >> "$LOG" 2>&1 &
  CHILD=$!
  CHILD_PGID=$(ps -o pgid= -p "$CHILD" 2>/dev/null | tr -d ' ' || echo "")
  say "child pid=$CHILD pgid=${CHILD_PGID:-?}"

  set +e
  wait "$CHILD"
  status=$?
  set -e
  say "child exited status=$status"

  if [ "$status" -eq 0 ]; then
    say "clean exit; supervisor done"
    exit 0
  fi
  if [ -f "$DATA_ROOT/runner/STOP_REQUESTED" ]; then
    say "stop requested; not restarting"
    exit 0
  fi
  say "restarting in ${RESTART_SEC}s"
  sleep "$RESTART_SEC"
done
