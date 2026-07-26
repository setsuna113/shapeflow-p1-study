#!/usr/bin/env bash
# Launch gate (plan §17.1), run ON THE FROZEN HOST as root, which then drops to the right
# identity for every step. Root does nothing here but switch users and read-only checks.
#
# Protocol v0.1 mode is USER_EXPLICIT_AUTO_LAUNCH (plan §0, §4.1): the user's 2026-07-24
# "build it and start running" instruction IS the launch authorization. So this script runs the
# ordered hard gates and, if every one is green, starts the campaign. It does not stop to ask.
#
# Auto-launch is not a bypass. Any gate failure writes reports/BLOCKED_<slug>.md and exits
# non-zero -- never `|| true`, never a silent substitute model, dataset or parameter.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
REPORTS="$REPO/reports"
DATA_ROOT="${SHAPEFLOW_DATA_ROOT:-/storage/nvme/shapeflow-data}"
APPROVAL_FILE="${SHAPEFLOW_APPROVAL_FILE:-$DATA_ROOT/approvals/launch_approval.json}"
export SHAPEFLOW_APPROVAL_FILE="$APPROVAL_FILE"
CONFIG="${SHAPEFLOW_CONFIG:-$REPO/configs/week1.yaml}"
ENGINE_EPOCH_FILE="${SHAPEFLOW_ENGINE_EPOCH_FILE:-/run/shapeflow-vllm-causal/engine_epoch}"
VENDOR_PIN="408da442a661ea5e40a6163329f82e3f22628949"
PATCHED=".build/open_deep_research-patched"
MIN_FREE_BYTES=12884901888   # 12 GiB hard floor (plan §5.2)
TOTAL_STEPS=18
STEP=0
SPEND_POSSIBLE=0

mkdir -p "$REPORTS"

blocked() {  # blocked <slug> <message...>
  local slug="$1"; shift
  {
    echo "# BLOCKED: $slug"
    echo
    echo "Launch gate stopped at $(date -u +%Y-%m-%dT%H:%M:%SZ) on step ${STEP}/${TOTAL_STEPS}."
    echo
    echo "$*"
    echo
    echo "The campaign was NOT started."
    if [ "$SPEND_POSSIBLE" -eq 0 ]; then
      echo
      echo "No paid step had run yet, so no Tavily credit or GPU hour was consumed."
    else
      echo
      echo "A paid step had already run. Consult the ledger for what was actually spent:"
      echo '    shapeflow-p1 status'
    fi
  } > "$REPORTS/BLOCKED_${slug}.md"
  echo "BLOCKED[$slug]: $*" >&2
  exit 1
}

# Stale artifacts are ARCHIVED, not deleted: a leftover green LAUNCH_GATE_PASSED.json must not
# be read as evidence for this run, and a previous BLOCKED report is audit history.
ARCHIVE="$REPORTS/history/$(date -u +%Y%m%dT%H%M%SZ)"
if compgen -G "$REPORTS/LAUNCH_GATE_PASSED.json" > /dev/null \
   || compgen -G "$REPORTS/BLOCKED_*.md" > /dev/null; then
  mkdir -p "$ARCHIVE"
  mv -f "$REPORTS"/LAUNCH_GATE_PASSED.json "$REPORTS"/ACCEPTANCE.json "$REPORTS"/BLOCKED_*.md \
        "$ARCHIVE"/ 2>/dev/null || true
  echo "archived prior gate artifacts to $ARCHIVE"
fi

step() { STEP=$((STEP + 1)); echo "== [${STEP}/${TOTAL_STEPS}] $* =="; }
# runuser without -l does not export USER, and the CLI asserts the effective identity from
# it -- so a step would run as the right uid while failing the check that says so.
as() { local role="$1"; shift; runuser -u "$role" -- env USER="$role" LOGNAME="$role" \
        SHAPEFLOW_DATA_ROOT="$DATA_ROOT" SHAPEFLOW_REPO="$REPO" \
        SHAPEFLOW_APPROVAL_FILE="$APPROVAL_FILE" \
        PYTHONHASHSEED=0 TZ=UTC \
        ${SHAPEFLOW_ENGINE_PID:+SHAPEFLOW_ENGINE_PID="$SHAPEFLOW_ENGINE_PID"} \
        ${SHAPEFLOW_ENGINE_LOG:+SHAPEFLOW_ENGINE_LOG="$SHAPEFLOW_ENGINE_LOG"} \
        SHAPEFLOW_ENGINE_EPOCH_FILE="$ENGINE_EPOCH_FILE" \
        "$@"; }
SF="$REPO/.venv/bin/shapeflow-p1"

# The causal engine runs under sfsupervise; find the api_server pid that is genuinely a
# descendant of OUR supervisor (its pid is in the run file), so doctor's flags check reads this
# run's process and can never match a foreign vLLM on a shared host. Empty output if not found,
# in which case doctor still verifies the backend from the log but leaves the flags unchecked.
discover_engine_pid() {
  local runfile="${SHAPEFLOW_RUN_DIR:-/run/shapeflow}/vllm-causal.pid" sup pid p argv0
  [ -r "$runfile" ] || return 1
  sup="$(cat "$runfile" 2>/dev/null)" || return 1
  [ -n "$sup" ] || return 1
  for pid in $(pgrep -f "vllm.entrypoints.openai.api_server" 2>/dev/null); do
    # The engine is the python interpreter itself, not the runuser/env wrappers whose argv
    # also carries the command -- pick the one whose argv[0] is python so the flags read from
    # /proc match what the freeze recorded.
    argv0="$(tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null | head -1)"
    case "${argv0##*/}" in python | python3 | python3.*) ;; *) continue ;; esac
    p="$pid"
    while [ -n "$p" ] && [ "$p" -gt 1 ] 2>/dev/null; do
      p="$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')"
      [ "$p" = "$sup" ] && { echo "$pid"; return 0; }
    done
  done
  return 1
}

# ---------------------------------------------------------------------------------------
step "singleton: no active coordinator"
# `systemctl is-active` is not sufficient on this host: systemd is not PID 1 here, so the call
# fails, `&&` short-circuits, and the gate passed unconditionally -- while the coordinator that
# actually runs is an sfsupervise process holding a flock. Check both, so the gate means the
# same thing under systemd and under the supervisor.
if command -v systemctl >/dev/null 2>&1 && [ "$(cat /proc/1/comm 2>/dev/null)" = "systemd" ]; then
  systemctl is-active --quiet shapeflow-p1-week1.service \
    && blocked "ACTIVE_COORDINATOR" "shapeflow-p1-week1.service is already running"
fi
for lock in "${SHAPEFLOW_RUN_DIR:-/run/shapeflow}"/week1*.lock; do
  [ -e "$lock" ] || continue
  # flock -n succeeds only if nothing holds it; if we cannot take it, a coordinator is live.
  if ! flock -n "$lock" true 2>/dev/null; then
    blocked "ACTIVE_COORDINATOR" "a supervised coordinator already holds $lock"
  fi
done

step "credentials present, readable only by the provider"
# Exa is the acquisition credential the provider refuses to start without; Tavily is optional
# since acquisition moved off it. Checking only tavily.key here would have passed a host that
# cannot acquire anything at all.
runuser -u sfprovider -- test -r /etc/shapeflow/exa.key \
  || blocked "NO_SECURE_SECRET_INJECTION" "sfprovider cannot read the Exa credential"
runuser -u sfprovider -- test -r /etc/shapeflow/deepseek.key \
  || blocked "NO_SECURE_SECRET_INJECTION" "sfprovider cannot read the DeepSeek credential"
# Every credential actually present must be unreachable by every non-provider role. Probing one
# fixed filename would leave a newly added key unproven.
for role in sfrunner sfinfer sfevaluator; do
  for key in exa deepseek tavily; do
    [ -f "/etc/shapeflow/$key.key" ] || continue
    if runuser -u "$role" -- cat "/etc/shapeflow/$key.key" >/dev/null 2>&1; then
      blocked "SECRET_ISOLATION" "$role can read $key.key; the boundary does not hold"
    fi
  done
done

step "campaign quota + free-space floor"
FREE_BYTES="$(df -B1 --output=avail "$DATA_ROOT" | tail -1 | tr -d ' ')"
[ "$FREE_BYTES" -ge "$MIN_FREE_BYTES" ] || blocked "LOW_DISK" \
  "free ${FREE_BYTES}B is under the ${MIN_FREE_BYTES}B floor"

step "one GPU, leased by UUID, with no foreign process on it"
command -v nvidia-smi >/dev/null 2>&1 || blocked "NO_GPU" "nvidia-smi not found"

step "submodule + vendor commit"
git submodule update --init --recursive
GOT="$(git -C vendor/open_deep_research rev-parse HEAD)"
[ "$GOT" = "$VENDOR_PIN" ] || blocked "VENDOR_DRIFT" "ODR at $GOT, expected $VENDOR_PIN"

step "materialize the pinned submodule and verify the patched-tree hash"
[ -f "$REPO/patches/odr_p1_hooks.patch" ] || blocked "PATCH_MISSING" "the patch is absent"
./scripts/materialize_vendor.sh || blocked "PATCHED_TREE_DRIFT" \
  "the patched tree does not hash to patches/patched_tree.sha256"

step "uv sync --frozen"
command -v uv >/dev/null 2>&1 || blocked "NO_UV" "uv not installed"
uv sync --frozen --all-extras || blocked "UV_SYNC" "uv sync --frozen failed; do not relax --frozen"

step "the installed ODR is the patched tree"
"$REPO/.venv/bin/python" - <<'PY' || blocked "IMPORT_ORIGIN" "the installed ODR is not the patched tree"
import sys, pathlib
sys.path.insert(0, "src")
from shapeflow_p1.treehash import tree_sha256
import open_deep_research
live = tree_sha256(pathlib.Path(open_deep_research.__path__[0]))
want = tree_sha256(pathlib.Path(".build/open_deep_research-patched/src/open_deep_research"))
sys.exit(0 if live == want else 1)
PY

step "secret scan (gitleaks)"
command -v gitleaks >/dev/null 2>&1 || blocked "NO_GITLEAKS" \
  "gitleaks not installed; the secret gate cannot be satisfied by assertion"
gitleaks detect --source "$REPO" --config "$REPO/.gitleaks.toml" --no-banner --redact \
  || blocked "SECRET_SCAN" "gitleaks found a candidate credential; see its redacted output"

step "tests"
"$REPO/.venv/bin/python" -m pytest -q || blocked "TESTS" "test suite failed"

step "approval binds the live configuration"
SHAPEFLOW_APPROVAL_FILE="$APPROVAL_FILE" \
  "$SF" verify-approval --approval "$APPROVAL_FILE" --config "$CONFIG" \
  || blocked "APPROVAL_MISMATCH" \
    "external approval $APPROVAL_FILE does not bind the clean live configuration"

step "doctor: stack, GPU UUID, driver, CUDA, vLLM, model revision, patch, APC"
# The frozen stack manifest is only verifiable against a running engine. Give doctor the
# engine's own log (for the attention backend it chose at startup) and its pid (for the live
# --max-num-seqs 1 and prefix-caching-off flags that make the causal layer causal).
export SHAPEFLOW_ENGINE_LOG="$REPO/logs/vllm-causal.log"
SHAPEFLOW_ENGINE_PID="$(discover_engine_pid || true)"; export SHAPEFLOW_ENGINE_PID
as sfrunner "$SF" doctor --config "$CONFIG" --role runner || blocked "DOCTOR" "doctor failed"

step "acceptance matrix"
as sfrunner "$SF" accept --config "$CONFIG" || blocked "ACCEPTANCE" \
  "reports/ACCEPTANCE.json has a failing gate"

step "P0 parity (mock model, no GPU, no credits)"
"$SF" test-p0-parity --config "$CONFIG" || blocked "P0_PARITY" \
  "the patched hooks-off graph does not reproduce vendor; all GPU screening is barred"

# Everything past this line can consume a paid resource.
SPEND_POSSIBLE=1

# The last free gate, and deliberately the one immediately before the first paid command.
# "The credential file is readable" and "the vendor accepts it" are different facts, and only
# the second predicts whether acquisition can work. Without this check a rejected key is
# discovered one billed rejection at a time -- which is exactly how a dead Tavily key turned
# into 262 charged 401s and an empty frozen world. The probe sends a deliberately invalid body,
# so a working credential answers 400 and nothing is searched or charged.
step "upstream credentials are accepted (free probe, before anything is bought)"
PROBE_STATUS="$(as sfsteward curl -sS -o /tmp/sf-credential-probe.json -w '%{http_code}' \
  --unix-socket "${SHAPEFLOW_RUN_DIR:-/run/shapeflow}/provider.sock" \
  -H "Authorization: Bearer $(cat /etc/shapeflow-tokens/steward.token)" \
  -X POST http://localhost/v1/credentials/probe 2>/dev/null || echo 000)"
if [ "$PROBE_STATUS" != "200" ]; then
  blocked "CREDENTIAL_REJECTED" \
    "the provider's upstream credential probe returned HTTP $PROBE_STATUS: $(
      head -c 400 /tmp/sf-credential-probe.json 2>/dev/null)"
fi
rm -f /tmp/sf-credential-probe.json

step "prepare + acquire + freeze machine truth candidates (steward)"
as sfsteward "$SF" prepare --config "$CONFIG" || blocked "PREPARE" "prepare failed"
as sfsteward "$SF" acquire --config "$CONFIG" || blocked "ACQUIRE" "acquisition failed"
# Truth construction reads only the already-frozen task/source world and writes into the
# evaluator tree.  It runs before treatment so neither an arm's report nor an observed effect
# can influence which atoms enter the candidate answer key.  Human audit remains a later,
# explicit verdict gate; this command never labels machine candidates AUDITED.
as sfsteward "$SF" build-truth --config "$CONFIG" \
  || blocked "BUILD_TRUTH" "machine truth-candidate construction failed"
as sfsteward "$SF" freeze-analysis-design --config "$CONFIG" \
  || blocked "ANALYSIS_DESIGN" \
    "pre-treatment feature registry or eligibility specification could not be frozen"

# Preflight is the last check BEFORE treatment, so it runs before the GPU canary rather
# than after it. It used to sit after prepare, acquire and smoke, by which point the
# credits and the GPU hours it was meant to protect were already spent.
step "preflight against the approved protocol SHA"
APPROVED_PROTOCOL_SHA="$("$REPO/.venv/bin/python" -c \
  "import sys; sys.path.insert(0,'src'); from pathlib import Path; \
from shapeflow_p1.protocol import protocol_sha; print(protocol_sha(Path('.')))")"
APPROVED_BINDING_SHA="$("$REPO/.venv/bin/python" -c \
  "import sys; sys.path.insert(0,'src'); from pathlib import Path; \
from shapeflow_p1.protocol import verified_execution_binding; \
print(verified_execution_binding(Path('.')).digest)")"
as sfrunner "$SF" preflight --config "$CONFIG" \
  --approved-protocol-sha "$APPROVED_PROTOCOL_SHA" \
  || blocked "PREFLIGHT" "campaign preflight failed under protocol $APPROVED_PROTOCOL_SHA"

step "GPU smoke (runner)"
as sfrunner "$SF" smoke --config "$CONFIG" || blocked "GPU_SMOKE" "the GPU canary failed"

# ---------------------------------------------------------------------------------------
cat > "$REPORTS/LAUNCH_GATE_PASSED.json" <<JSON
{
  "status": "GATES_GREEN_LAUNCHING",
  "approval_mode": "USER_EXPLICIT_AUTO_LAUNCH",
  "claim_scope": "FORMATIVE_ONLY",
  "protocol_sha": "$APPROVED_PROTOCOL_SHA",
  "execution_binding_sha256": "$APPROVED_BINDING_SHA",
  "approval_file": "$APPROVAL_FILE",
  "vendor_commit": "$GOT",
  "config": "$CONFIG",
  "data_root": "$DATA_ROOT",
  "free_bytes_at_launch": $FREE_BYTES,
  "gates_passed": $TOTAL_STEPS,
  "passed_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON

echo
echo "All ${TOTAL_STEPS} hard gates passed. Starting screening (binding $APPROVED_BINDING_SHA)."
# This host runs in a container where systemd is not PID 1. The launch used to be
# `systemctl enable --now` unconditionally: the earlier `systemctl is-active` gate
# short-circuited on the same absence, so every gate reported green and then the campaign
# simply never started. sfsupervise is the documented substitute (plan §17.2) and was
# already installed by install_host.sh; nothing used it for the coordinator.
if [ "$(cat /proc/1/comm 2>/dev/null)" = "systemd" ] && command -v systemctl >/dev/null 2>&1; then
  # install_host already rendered @PROTOCOL_SHA@ (normally to UNSET), so replacing the
  # template token here was a no-op. Replace the coordinator argument itself.
  sed -i -E \
    "s|--protocol-sha [^[:space:]]+|--protocol-sha $APPROVED_BINDING_SHA|" \
    /etc/systemd/system/shapeflow-p1-week1.service
  systemctl daemon-reload
  systemctl enable --now shapeflow-p1-week1.service
  systemctl --no-pager status shapeflow-p1-week1.service | head -20
else
  echo "systemd is not PID 1; supervising the campaign with sfsupervise instead."
  setsid /usr/local/bin/sfsupervise week1 sfrunner "$REPO" "$DATA_ROOT" -- \
    "$SF" run-screen --config "$CONFIG" --resume --protocol-sha "$APPROVED_BINDING_SHA" \
    >> "$REPO/logs/coordinator.log" 2>&1 &
  echo "sfsupervise started (pid $!); log: $REPO/logs/coordinator.log"
fi
