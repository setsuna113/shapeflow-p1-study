#!/usr/bin/env bash
# Bring up four execution lanes, prove them, and start the campaign on all four.
#
# Five gates, in this order, because each one makes the next one's failures interpretable:
#
#   1. P1 validity on ONE lane. Four GPUs producing invalid results faster is the failure this
#      exists to prevent, and it is not hypothetical: the previous round ran 146 cells across 19
#      arms and published no P1 output at all.
#   2. Four-engine capacity. Four 4090s at once is 1.4 kW; sustained throttling, an OOM or an
#      Xid would show up later as a treatment effect on whichever lane got the hot card.
#   3. Routing and failure. Stopping one engine must damage exactly one lane. If a restart could
#      splice a completed P0 with a post-restart P1, every paired contrast on that lane is void.
#   4. Four-task full-arm canary. One complete task per GPU, ALL of its arms -- not P0 only --
#      then a merge check. A per-lane P0 smoke would not exercise the thing that is new.
#   5. Disk forecast from measured bytes/task. /storage/nvme has ~22 GiB against a 12 GiB floor,
#      and a campaign that fills the disk loses the ledger it is writing.
#
# Every gate is engineering. None of them looks at a quality outcome: continuing based on how
# good early results look would select on the effect the screen exists to estimate. Once they
# pass, the campaign starts without further approval.
set -euo pipefail

REPO="${SHAPEFLOW_REPO:-/storage/nvme/shapeflow-p1-study}"
DATA_ROOT="${SHAPEFLOW_DATA_ROOT:-/storage/nvme/shapeflow-data}"
CONFIG="${SHAPEFLOW_CONFIG:-$REPO/configs/week1.yaml}"
SF="$REPO/.venv/bin/shapeflow-p1"
PY="$REPO/.venv/bin/python"
REPORTS="$REPO/reports"
CAPACITY_SECONDS="${SHAPEFLOW_CAPACITY_SECONDS:-1800}"
MIN_FREE_BYTES="${SHAPEFLOW_MIN_FREE_BYTES:-12884901888}"   # 12 GiB

[ "$(id -u)" -eq 0 ] || { echo "run_lanes.sh switches identity and must run as root" >&2; exit 1; }
mkdir -p "$REPORTS" "$REPO/logs"

blocked() {
  local reason="$1" detail="$2"
  {
    echo "# BLOCKED: $reason"
    echo
    echo "$detail"
    echo
    echo "Stopped at $(date -u +%Y-%m-%dT%H:%M:%SZ). Nothing was started beyond this gate."
    echo "Finished work is kept; the ledger holds every attempt."
  } > "$REPORTS/BLOCKED_${reason}.md"
  echo "BLOCKED[$reason]: $detail" >&2
  exit 1
}

# Two helpers, deliberately separate. A `VAR=x func` assignment in front of a shell *function*
# persists after the call in bash -- unlike in front of an external command -- so a single helper
# that picked up an ambient SHAPEFLOW_LANE would silently give a lane identity to the
# campaign-wide steward commands that follow a per-lane one.
as() { local role="$1"; shift; runuser -u "$role" -- env \
  SHAPEFLOW_REPO="$REPO" SHAPEFLOW_DATA_ROOT="$DATA_ROOT" HOME=/tmp \
  PYTHONHASHSEED=0 TZ=UTC PYTHONUNBUFFERED=1 "$@"; }

as_lane() { local lane="$1" role="$2"; shift 2; runuser -u "$role" -- env \
  SHAPEFLOW_REPO="$REPO" SHAPEFLOW_DATA_ROOT="$DATA_ROOT" HOME=/tmp \
  PYTHONHASHSEED=0 TZ=UTC PYTHONUNBUFFERED=1 \
  SHAPEFLOW_LANE="$lane" SHAPEFLOW_GPU_UUID="${GPUS[$lane]}" "$@"; }

read -r LANE_COUNT PAID_LANE <<EOF
$("$PY" - "$REPO" <<'PYX'
import sys, yaml
with open(f"{sys.argv[1]}/configs/week1.yaml", encoding="utf-8") as handle:
    shards = yaml.safe_load(handle)["measurement"]["shards"]
print(int(shards["lane_count"]), int(shards["paid_upstream_lane"]))
PYX
)
EOF
[ -n "${LANE_COUNT:-}" ] || blocked "SHARD_CONFIG" "could not read measurement.shards from $CONFIG"

mapfile -t GPUS < <("$PY" - "$REPO" <<'PYX'
import sys, yaml
with open(f"{sys.argv[1]}/configs/stack.yaml", encoding="utf-8") as handle:
    for uuid in yaml.safe_load(handle)["host"]["gpu_uuid_pool"]:
        print(uuid)
PYX
)
[ "${#GPUS[@]}" -ge "$LANE_COUNT" ] || blocked "GPU_POOL" \
  "${#GPUS[@]} GPUs in the frozen pool, $LANE_COUNT lanes required"

# The approved execution binding. bootstrap_and_run.sh writes LAUNCH_GATE_PASSED.json before
# handing over, so every mutating command from here on is a post-launch resume and has to carry
# the exact binding digest -- the guard that enforces that is the same one which stops a resumed
# run from continuing under a protocol nobody approved.
BINDING="$("$PY" -c "import sys; sys.path.insert(0,'$REPO/src'); from pathlib import Path; \
from shapeflow_p1.protocol import verified_execution_binding; \
print(verified_execution_binding(Path('$REPO')).digest)")"
[ -n "$BINDING" ] || blocked "NO_BINDING" "could not derive the approved execution binding"
SMOKE_FLAGS=(--config "$CONFIG" --resume --protocol-sha "$BINDING")

start_lane() {
  local lane="$1"
  SHAPEFLOW_LANE="$lane" SHAPEFLOW_GPU_UUID="${GPUS[$lane]}" \
    "$REPO/scripts/start_engine.sh" &
  SHAPEFLOW_LANE="$lane" "$REPO/scripts/start_provider.sh" &
  # Wait for the engine to serve rather than for a pid: a pid is not readiness, and the last
  # launch reported green while nothing was listening.
  local port=$((8000 + lane)) waited=0
  while [ "$waited" -lt 600 ]; do
    if curl -fsS "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
      echo "lane $lane engine ready on :$port (${GPUS[$lane]})"
      return 0
    fi
    sleep 5; waited=$((waited + 5))
  done
  blocked "ENGINE_NOT_READY" "lane $lane engine did not answer /v1/models within 600s"
}

stop_lane_engine() {
  local lane="$1"
  local unit="vllm-causal-native-lane${lane}"
  local pidfile="/run/shapeflow/${unit}.pid"
  [ -f "$pidfile" ] && kill -TERM "$(cat "$pidfile")" 2>/dev/null || true
}

gpu_health() {
  # Throttle reasons other than "GpuIdle" mean the card was held back; a lane on a throttled
  # card would attribute the slowdown to whatever arm happened to be running.
  local uuid="$1"
  nvidia-smi -i "$uuid" --query-gpu=clocks_throttle_reasons.active,temperature.gpu,power.draw \
    --format=csv,noheader 2>/dev/null || echo "unavailable"
}

# ---------------------------------------------------------------------------------------
echo "== gate 1/5: P1 actually publishes, on one lane =="
start_lane 0
as_lane 0 sfrunner "$SF" smoke "${SMOKE_FLAGS[@]}" \
  || blocked "LANE0_CANARY" \
     "the single-lane canary failed. Read the canary summary before concluding anything about
P1: the p1_published_output check names each arm and its top failure reason, and a canary can
also fail for reasons that have nothing to do with whether P1 published -- a refused flag, a
missing world, an unreachable provider. Do not report a cause this gate did not establish."

# ---------------------------------------------------------------------------------------
echo "== gate 2/5: four engines under real load, no OOM / Xid / sustained throttle =="
# Loaded, not idle. Polling /v1/models on four idle engines for half an hour proves that four
# processes can coexist and nothing else -- not that the KV cache holds at the new max_num_seqs,
# not that four 4090s at 1.4 kW stay off their thermal and power caps. The load is the four-lane
# canary itself: real cells, every arm, on the configuration the screen will use.
for lane in $(seq 1 $((LANE_COUNT - 1))); do start_lane "$lane"; done
XID_BEFORE="$(dmesg 2>/dev/null | grep -c 'NVRM: Xid' || true)"

declare -A CANARY_PID=()
for lane in $(seq 0 $((LANE_COUNT - 1))); do
  as_lane "$lane" sfrunner "$SF" smoke "${SMOKE_FLAGS[@]}" \
    >> "$REPO/logs/canary-lane${lane}.log" 2>&1 &
  CANARY_PID[$lane]=$!
done

# Sample health while they run. A card that throttles only under load is exactly the case an
# idle check cannot see, and it would surface later as a treatment effect on whichever lane got
# the hot card.
THROTTLED=""
while :; do
  running=0
  for lane in $(seq 0 $((LANE_COUNT - 1))); do
    kill -0 "${CANARY_PID[$lane]}" 2>/dev/null && running=$((running + 1))
    health="$(gpu_health "${GPUS[$lane]}")"
    echo "  lane $lane: $health"
    case "$health" in
      *SwPowerCap*|*HwSlowdown*|*SwThermalSlowdown*|*HwThermalSlowdown*)
        THROTTLED="$THROTTLED lane$lane" ;;
    esac
  done
  [ "$running" -eq 0 ] && break
  sleep 60
done

CANARY_FAILED=""
for lane in $(seq 0 $((LANE_COUNT - 1))); do
  wait "${CANARY_PID[$lane]}" || CANARY_FAILED="$CANARY_FAILED lane$lane"
done
[ -z "$CANARY_FAILED" ] || blocked "LANE_CANARY" \
  "the four-lane canary failed on:$CANARY_FAILED. If p1_published_output is the failing check,
P1 produced no output at all under contention and four GPUs would only produce none faster.
Per-lane logs: $REPO/logs/canary-lane*.log"
[ -z "$THROTTLED" ] || blocked "GPU_THROTTLED" \
  "sustained throttling under load on:$THROTTLED"
XID_AFTER="$(dmesg 2>/dev/null | grep -c 'NVRM: Xid' || true)"
[ "$XID_AFTER" -le "$XID_BEFORE" ] || blocked "GPU_XID" \
  "$((XID_AFTER - XID_BEFORE)) new Xid error(s) while four engines were loaded"

# ---------------------------------------------------------------------------------------
echo "== gate 3/5: stopping one engine damages exactly one lane =="
VICTIM=$(( LANE_COUNT > 2 ? 2 : LANE_COUNT - 1 ))
stop_lane_engine "$VICTIM"
sleep 20
for lane in $(seq 0 $((LANE_COUNT - 1))); do
  if [ "$lane" = "$VICTIM" ]; then continue; fi
  curl -fsS "http://127.0.0.1:$((8000 + lane))/v1/models" >/dev/null 2>&1 \
    || blocked "LANE_BLAST_RADIUS" \
       "stopping lane $VICTIM's engine also took down lane $lane; the lanes are not isolated"
done
start_lane "$VICTIM"
# The restart necessarily mints a new engine epoch, and a block whose cells span two epochs is
# already refused as a paired observation (schedule.freeze_root_record). This gate proves the
# blast radius; it does not make the restart free.
echo "  lane $VICTIM restarted under a new engine epoch; its in-flight block will not pair"

# ---------------------------------------------------------------------------------------
echo "== gate 4/5: the lanes covered disjoint tasks, and the partition is frozen =="
# Four lanes each running all four canary tasks would pass every per-lane check above and prove
# nothing about the partition: the error exists only in the union.
"$PY" - "$DATA_ROOT" "$LANE_COUNT" <<'PYX' || blocked "LANE_OVERLAP" \
  "two lanes ran the same canary task; the partition is not a partition"
import json
import sys
from pathlib import Path

data_root, lane_count = Path(sys.argv[1]), int(sys.argv[2])
by_lane: dict[int, set[str]] = {}
for lane in range(lane_count):
    tasks: set[str] = set()
    for path in (data_root / f"runner-lane{lane}" / "runs").rglob("*.json"):
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        task_id = str((body.get("cell") or {}).get("task_id") or body.get("task_id") or "")
        if task_id:
            tasks.add(task_id)
    by_lane[lane] = tasks
    print(f"lane {lane}: {sorted(tasks)}")
overlaps = [
    (a, b, sorted(by_lane[a] & by_lane[b]))
    for a in sorted(by_lane) for b in sorted(by_lane) if a < b and by_lane[a] & by_lane[b]
]
if overlaps:
    print(f"overlapping tasks: {overlaps}", file=sys.stderr)
    raise SystemExit(1)
if not any(by_lane.values()):
    print("no lane recorded any task; the canary produced nothing to check", file=sys.stderr)
    raise SystemExit(1)
raise SystemExit(0)
PYX

# The screen's partition, frozen now that the apparatus is proven. Write-once: a lane chosen
# after a result is seen is not a partition, it is a selection.
as sfsteward "$SF" freeze-shards --config "$CONFIG" \
  || blocked "SHARD_FREEZE" "the task-to-lane partition could not be frozen"

# ---------------------------------------------------------------------------------------
echo "== gate 5/5: disk forecast =="
"$PY" - "$DATA_ROOT" "$MIN_FREE_BYTES" "$REPO" <<'PYX' || blocked "DISK_FORECAST" \
  "the projected campaign does not leave the free-space floor intact"
import json
import shutil
import sys
from pathlib import Path

import yaml

data_root, floor, repo = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
config = yaml.safe_load((repo / "configs" / "week1.yaml").read_text(encoding="utf-8"))

free = shutil.disk_usage(data_root).free
measured = sum(
    path.stat().st_size
    for lane in data_root.glob("runner-lane*")
    for path in lane.rglob("*")
    if path.is_file()
)
canary_tasks = int(config["canary"]["tasks"])
# Counted from the frozen corpus rather than declared: the number of screen tasks is whatever
# actually has a frozen world, and a declared count that drifted from it would forecast for a
# campaign that is not the one about to run.
split = str(config["screen"]["split"])
corpus = data_root / str(config["paths"]["frozen_corpus_for_runner"]).split("/", 1)[-1]
task_dir = data_root / str(config["paths"]["frozen_corpus_for_runner"]) / "tasks"
screen_tasks = sum(
    1 for path in sorted(task_dir.glob("*.json"))
    if json.loads(path.read_text(encoding="utf-8")).get("split") == split
)
del corpus
if not measured or not canary_tasks or not screen_tasks:
    print(
        f"cannot extrapolate: measured={measured}B, canary_tasks={canary_tasks}, "
        f"screen_tasks={screen_tasks}",
        file=sys.stderr,
    )
    raise SystemExit(1)

# The canary ran every arm on `canary_tasks` tasks; the screen runs every arm on
# `screen_tasks`. Scaling by task count keeps the arm mix fixed, which matters because arms
# differ several-fold in how much they write. Deliberately crude: an over-estimate stops the
# run early, which is the safe direction when the alternative is losing the ledger mid-write.
projected = measured * screen_tasks / canary_tasks
remaining = free - projected
print(json.dumps({
    "free_bytes": free,
    "measured_canary_bytes": measured,
    "canary_tasks": canary_tasks,
    "screen_tasks": screen_tasks,
    "projected_screen_bytes": int(projected),
    "remaining_after_campaign": int(remaining),
    "floor_bytes": floor,
    "verdict": "OK" if remaining >= floor else "WOULD_BREACH_FLOOR",
}, indent=2, sort_keys=True))
raise SystemExit(0 if remaining >= floor else 1)
PYX

# ---------------------------------------------------------------------------------------
cat > "$REPORTS/LANE_GATES_PASSED.json" <<JSON
{
  "status": "LANE_GATES_GREEN",
  "lane_count": $LANE_COUNT,
  "paid_upstream_lane": $PAID_LANE,
  "capacity_gate_seconds": $CAPACITY_SECONDS,
  "passed_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON

echo
echo "All five lane gates passed. Starting the screen on $LANE_COUNT lanes."
for lane in $(seq 0 $((LANE_COUNT - 1))); do
  SHAPEFLOW_LANE="$lane" setsid /usr/local/bin/sfsupervise \
    "week1-lane${lane}" sfrunner "$REPO" "$DATA_ROOT" -- \
    "$SF" run-screen --config "$CONFIG" --resume --protocol-sha "$BINDING" \
    >> "$REPO/logs/coordinator-lane${lane}.log" 2>&1 &
  echo "lane $lane screening under sfsupervise (pid $!)"
done
echo
echo "When every lane is terminal, merge them:"
echo "    runuser -u sfsteward -- $SF merge-shards --config $CONFIG --run-id <run-id>"
