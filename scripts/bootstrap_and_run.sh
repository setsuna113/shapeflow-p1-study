#!/usr/bin/env bash
# Launch gate (plan section 17.1), run ON THE FROZEN HOST (sjtu), not the authoring environment.
#
# Protocol v0.1 mode is USER_EXPLICIT_AUTO_LAUNCH (plan sections 0 and 4.1): the user's
# 2026-07-24 "build it and start running" instruction IS the launch authorization. So this
# script runs the ordered hard gates and, if every one is green, EXECS the campaign. It does
# not stop to ask. There is no gate-green-then-pause mode; a pause would be a different user
# decision that was never made.
#
# Auto-launch is not a bypass. Any gate failure writes reports/BLOCKED_<slug>.md and exits
# non-zero -- never `|| true`, never a silent substitute model/dataset/parameter (plan
# section 0, AGENTS.md section 7). Steps whose runtime is not yet implemented hard-fail here
# by design, so this can never "pass" an incomplete stack.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"   # absolute, per plan 17.1 step 1
cd "$REPO"
REPORTS="$REPO/reports"
RUN_DIR="$REPO/runs"
mkdir -p "$REPORTS" "$RUN_DIR"

CONFIG="${SHAPEFLOW_CONFIG:-$REPO/configs/week1.yaml}"
VENDOR_PIN="408da442a661ea5e40a6163329f82e3f22628949"
PATCHED=".build/open_deep_research-patched"
MIN_FREE_BYTES=12884901888   # 12 GiB hard floor (plan section 5.2)
TOTAL_STEPS=18
STEP=0
# Flipped by the first step that can consume a paid resource (GPU smoke touches the engine;
# prepare can call Tavily). Before that a BLOCKED report may say nothing was spent; after it,
# saying so would be a false claim in the audit trail, and the ledger is the only authority.
SPEND_POSSIBLE=0

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
      echo '    uv run shapeflow-p1 status'
      echo '    uv run shapeflow-p1 verify-artifacts runs/ledger.sqlite object_store'
    fi
  } > "$REPORTS/BLOCKED_${slug}.md"
  echo "BLOCKED[$slug]: $*" >&2
  exit 1
}

# Stale artifacts from an earlier attempt are ARCHIVED, not deleted: a leftover green
# LAUNCH_GATE_PASSED.json must not be read as evidence for this run, but a previous BLOCKED
# report is audit history and deleting it destroys the record of why a run was stopped.
ARCHIVE="$REPORTS/history/$(date -u +%Y%m%dT%H%M%SZ)"
if compgen -G "$REPORTS/LAUNCH_GATE_PASSED.json" > /dev/null \
   || compgen -G "$REPORTS/ACCEPTANCE.json" > /dev/null \
   || compgen -G "$REPORTS/BLOCKED_*.md" > /dev/null; then
  mkdir -p "$ARCHIVE"
  mv -f "$REPORTS"/LAUNCH_GATE_PASSED.json "$REPORTS"/ACCEPTANCE.json "$REPORTS"/BLOCKED_*.md \
        "$ARCHIVE"/ 2>/dev/null || true
  echo "archived prior gate artifacts to $ARCHIVE"
fi

step() { STEP=$((STEP + 1)); echo "== [${STEP}/${TOTAL_STEPS}] $* =="; }

# ---------------------------------------------------------------------------------------
step "singleton: no active coordinator"
if pgrep -f "shapeflow-p1 run-week1" >/dev/null 2>&1; then
  blocked "ACTIVE_COORDINATOR" "another coordinator is already running; refusing to start a second"
fi

step "singleton flock (plan 17.2)"
exec 9>"$RUN_DIR/launch.lock"
flock -n 9 || blocked "SINGLETON_LOCK" "another launch gate holds runs/launch.lock"

step "credentials present in approved backend (never printed)"
: "${TAVILY_API_KEY_FILE:?set TAVILY_API_KEY_FILE to the 0400 credential path}"
: "${DEEPSEEK_API_KEY_FILE:?set DEEPSEEK_API_KEY_FILE to the 0400 credential path}"
[ -r "$TAVILY_API_KEY_FILE" ] || blocked "NO_SECURE_SECRET_INJECTION" "Tavily credential unreadable"
[ -r "$DEEPSEEK_API_KEY_FILE" ] || blocked "NO_SECURE_SECRET_INJECTION" "DeepSeek credential unreadable"

step "budgets explicit and approved; approval hashes match live config"
[ -f "$REPO/protocol/launch_approval.json" ] || blocked "NO_APPROVAL" \
  "protocol/launch_approval.json missing"
[ -f "$CONFIG" ] || blocked "NO_CAMPAIGN_CONFIG" "campaign config $CONFIG missing"
# Verify the pinned hashes NOW, not after the paid steps. Checking mere file existence here and
# the hashes later means an edited decision threshold or budget can already have spent Tavily
# credits and GPU hours by the time the mismatch is discovered.
: "${SHAPEFLOW_PROTOCOL_SHA:?set SHAPEFLOW_PROTOCOL_SHA to the SHA of the protocol document}"
uv run shapeflow-p1 verify-approval \
  --approval "$REPO/protocol/launch_approval.json" --config "$CONFIG" \
  || blocked "APPROVAL_MISMATCH" \
     "launch_approval.json does not pin the live protocol/budget/threshold hashes; \
re-approval is required before anything is spent"

step "one GPU UUID lease (never index)"
command -v nvidia-smi >/dev/null 2>&1 || blocked "NO_GPU" "nvidia-smi not found on this host"

step "campaign quota + free-space floor"
FREE_BYTES="$(df -B1 --output=avail "$REPO" | tail -1 | tr -d ' ')"
[ "$FREE_BYTES" -ge "$MIN_FREE_BYTES" ] || blocked "LOW_DISK" \
  "free ${FREE_BYTES}B is under the ${MIN_FREE_BYTES}B floor"

step "submodule + vendor commit"
git submodule update --init --recursive
GOT="$(git -C vendor/open_deep_research rev-parse HEAD)"
[ "$GOT" = "$VENDOR_PIN" ] || blocked "VENDOR_DRIFT" "ODR at $GOT, expected $VENDOR_PIN"

step "materialize pinned submodule read-only into $PATCHED"
[ -f "$REPO/patches/odr_p1_hooks.patch" ] || blocked "PATCH_MISSING" \
  "patches/odr_p1_hooks.patch not generated yet (Block 2)"
rm -rf "$PATCHED"
mkdir -p "$PATCHED"
git -C vendor/open_deep_research archive HEAD | tar -x -C "$PATCHED"

step "apply patch and verify patched-tree hash"
git apply --directory="$PATCHED" "$REPO/patches/odr_p1_hooks.patch" \
  || blocked "PATCH_APPLY" "odr_p1_hooks.patch did not apply cleanly to the pinned tree"
PATCHED_TREE_SHA="$(find "$PATCHED" -type f -not -path '*/.git/*' -print0 \
  | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"
EXPECTED_TREE_SHA_FILE="$REPO/patches/patched_tree.sha256"
if [ -f "$EXPECTED_TREE_SHA_FILE" ]; then
  EXPECTED_TREE_SHA="$(tr -d ' \n' < "$EXPECTED_TREE_SHA_FILE")"
  [ "$PATCHED_TREE_SHA" = "$EXPECTED_TREE_SHA" ] || blocked "PATCHED_TREE_DRIFT" \
    "patched tree hashes to $PATCHED_TREE_SHA, approval expects $EXPECTED_TREE_SHA"
else
  blocked "NO_PATCHED_TREE_SHA" \
    "patches/patched_tree.sha256 missing; the patched tree is not pinned (observed $PATCHED_TREE_SHA)"
fi

step "uv sync --frozen"
command -v uv >/dev/null 2>&1 || blocked "NO_UV" "uv not installed; provision it first"
uv sync --frozen || blocked "UV_SYNC" "uv sync --frozen failed; do not relax --frozen"

step "import-origin assertion: ODR must come from the patched materialization"
uv run python -c "import open_deep_research,os,sys; p=os.path.realpath(open_deep_research.__file__); \
b=os.path.realpath('$PATCHED'); sys.exit(0 if p.startswith(b) else 1)" \
  || blocked "IMPORT_ORIGIN" "open_deep_research is not imported from $PATCHED"

step "secret scan (gitleaks)"
command -v gitleaks >/dev/null 2>&1 || blocked "NO_GITLEAKS" \
  "gitleaks not installed; the secret gate cannot be satisfied by assertion"
gitleaks detect --source "$REPO" --config "$REPO/.gitleaks.toml" --no-banner --redact \
  || blocked "SECRET_SCAN" "gitleaks found a candidate credential; see its redacted output"

step "tests"
uv run pytest -q || blocked "TESTS" "test suite failed"

step "doctor: stack, GPU UUID, driver, CUDA, vLLM, model revision, patch, APC, approval"
# --role is required: without it the identity check SKIPs, and a SKIP is not a pass, so the
# gate could never go green. The coordinator runs as the runner identity.
uv run shapeflow-p1 doctor --config "$CONFIG" --role runner \
  || blocked "DOCTOR" "doctor failed"

step "acceptance matrix (machine gate; completion claims do not count)"
uv run shapeflow-p1 accept --config "$CONFIG" || blocked "ACCEPTANCE" \
  "reports/ACCEPTANCE.json has a failing gate"

step "P0 parity (mock model, no GPU, no credits)"
uv run shapeflow-p1 test-p0-parity --config "$CONFIG" || blocked "P0_PARITY" \
  "patched hooks-off graph does not reproduce vendor; all GPU screening is barred"

# Everything past this line can consume a paid resource.
SPEND_POSSIBLE=1

step "prepare + GPU smoke (first paid steps)"
uv run shapeflow-p1 prepare --config "$CONFIG" || blocked "PREPARE" "prepare failed"
uv run shapeflow-p1 smoke --config "$CONFIG" || blocked "GPU_SMOKE" "GPU smoke failed"

step "preflight against the approved protocol SHA"
APPROVED_SHA="$SHAPEFLOW_PROTOCOL_SHA"
uv run shapeflow-p1 preflight --config "$CONFIG" --approved-protocol-sha "$APPROVED_SHA" \
  || blocked "PREFLIGHT" "campaign preflight failed under protocol $APPROVED_SHA"

# ---------------------------------------------------------------------------------------
# Every hard gate is green. Record the evidence, then start -- no pause, per protocol v0.1.
cat > "$REPORTS/LAUNCH_GATE_PASSED.json" <<JSON
{
  "status": "GATES_GREEN_LAUNCHING",
  "approval_mode": "USER_EXPLICIT_AUTO_LAUNCH",
  "protocol_sha": "$APPROVED_SHA",
  "vendor_commit": "$GOT",
  "patched_tree_sha256": "$PATCHED_TREE_SHA",
  "config": "$CONFIG",
  "free_bytes_at_launch": $FREE_BYTES,
  "gates_passed": $TOTAL_STEPS,
  "passed_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON

echo
echo "All ${TOTAL_STEPS} hard gates passed. Starting the Week-1 campaign (protocol $APPROVED_SHA)."
exec uv run shapeflow-p1 run-week1 --config "$CONFIG" --resume \
     --protocol-sha "$APPROVED_SHA"
