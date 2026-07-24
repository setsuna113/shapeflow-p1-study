#!/usr/bin/env bash
# Launch gate (plan §17.1), run ON THE FROZEN HOST (sjtu), not the authoring environment.
#
# It runs the ordered hard gates and then STOPS: the user chose gate-green-then-pause
# (2026-07-24), overriding the plan's auto-launch. On success it writes
# reports/LAUNCH_GATE_PASSED.json and waits for an explicit human go-ahead before any Tavily
# credit or GPU hour is spent. Any gate failure writes reports/BLOCKED*.md and exits non-zero.
#
# Steps that need runtime modules not yet complete (the ODR patch, the vLLM proxy, GPU smoke) are
# marked PENDING and will hard-fail until those land -- deliberately, so this never silently
# "passes" an incomplete stack.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
REPORTS="$REPO/reports"
mkdir -p "$REPORTS"

blocked() {  # blocked <slug> <message>
  local slug="$1"; shift
  {
    echo "# BLOCKED: $slug"
    echo
    echo "Launch gate stopped at $(date -u +%Y-%m-%dT%H:%M:%SZ)."
    echo
    echo "$*"
  } > "$REPORTS/BLOCKED_${slug}.md"
  echo "BLOCKED[$slug]: $*" >&2
  exit 1
}

echo "== [1/13] singleton: no active coordinator =="
if pgrep -f "shapeflow-p1 run-week1" >/dev/null 2>&1; then
  blocked "ACTIVE_COORDINATOR" "another coordinator is already running"
fi

echo "== [2/13] credentials present in approved backend (never printed) =="
: "${TAVILY_API_KEY_FILE:?set TAVILY_API_KEY_FILE to the 0400 credential path}"
: "${DEEPSEEK_API_KEY_FILE:?set DEEPSEEK_API_KEY_FILE to the 0400 credential path}"
[ -r "$TAVILY_API_KEY_FILE" ] || blocked "NO_SECURE_SECRET_INJECTION" "Tavily credential unreadable"
[ -r "$DEEPSEEK_API_KEY_FILE" ] || blocked "NO_SECURE_SECRET_INJECTION" "DeepSeek credential unreadable"

echo "== [3/13] budgets explicit and approved =="
[ -f "$REPO/protocol/launch_approval.json" ] || blocked "NO_APPROVAL" "protocol/launch_approval.json missing"

echo "== [4/13] singleton lock + one GPU UUID lease =="
# gpu_lease is enforced in-process by the coordinator; here we only assert nvidia-smi is present.
command -v nvidia-smi >/dev/null 2>&1 || blocked "NO_GPU" "nvidia-smi not found on this host"

echo "== [5/13] campaign quota + free-space floor =="
FREE_BYTES="$(df -B1 --output=avail "$REPO" | tail -1 | tr -d ' ')"
MIN_FREE=12884901888
[ "$FREE_BYTES" -ge "$MIN_FREE" ] || blocked "LOW_DISK" "free ${FREE_BYTES}B < floor ${MIN_FREE}B"

echo "== [6/13] submodule + vendor commit =="
git submodule update --init --recursive
PIN="408da442a661ea5e40a6163329f82e3f22628949"
GOT="$(git -C vendor/open_deep_research rev-parse HEAD)"
[ "$GOT" = "$PIN" ] || blocked "VENDOR_DRIFT" "ODR at $GOT, expected $PIN"

echo "== [7/13] materialize + patch vendor (PENDING: needs patches/odr_p1_hooks.patch) =="
if [ ! -f "$REPO/patches/odr_p1_hooks.patch" ]; then
  blocked "PATCH_MISSING" "patches/odr_p1_hooks.patch not yet generated (runtime task)"
fi
rm -rf .build/open_deep_research-patched
git -C vendor/open_deep_research archive HEAD | (mkdir -p .build/open_deep_research-patched && tar -x -C .build/open_deep_research-patched)
git apply --directory=.build/open_deep_research-patched "$REPO/patches/odr_p1_hooks.patch" \
  || blocked "PATCH_APPLY" "odr_p1_hooks.patch did not apply cleanly"

echo "== [8/13] uv sync --frozen =="
command -v uv >/dev/null 2>&1 || blocked "NO_UV" "uv not installed; provision it first"
uv sync --frozen || blocked "UV_SYNC" "uv sync --frozen failed"

echo "== [9/13] import-origin assertion =="
uv run python -c "import open_deep_research,os,sys; p=os.path.realpath(open_deep_research.__file__); \
b=os.path.realpath('.build/open_deep_research-patched'); sys.exit(0 if p.startswith(b) else 1)" \
  || blocked "IMPORT_ORIGIN" "open_deep_research is not imported from the patched tree"

echo "== [10/13] tests =="
uv run pytest -q || blocked "TESTS" "test suite failed"

echo "== [11/13] doctor =="
uv run shapeflow-p1 doctor || blocked "DOCTOR" "doctor failed"

echo "== [12/13] P0 parity + GPU smoke (PENDING: runtime) =="
uv run shapeflow-p1 test-p0-parity || blocked "P0_PARITY" "P0 parity failed"
uv run shapeflow-p1 smoke || blocked "GPU_SMOKE" "GPU smoke failed"

echo "== [13/13] gates green -> PAUSE for explicit go-ahead =="
cat > "$REPORTS/LAUNCH_GATE_PASSED.json" <<JSON
{
  "status": "GATES_GREEN_AWAITING_HUMAN_GO_AHEAD",
  "approval_mode": "USER_EXPLICIT_GATE_GREEN_THEN_PAUSE",
  "vendor_commit": "$GOT",
  "passed_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON
echo
echo "All hard gates passed. Per the approved gate-green-then-pause mode, NOT launching."
echo "To start the campaign (spends Tavily credits + GPU hours), a human runs:"
echo "    uv run shapeflow-p1 run-week1 --resume --protocol-sha <exact>"
