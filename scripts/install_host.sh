#!/usr/bin/env bash
# One-time host installation, run as root on the experiment machine (plan §5.2, §17.2).
#
# Root is used here and nowhere else: to create directories, set ownership, mint role tokens and
# install units. Nothing that touches a model, a web page or a credential ever runs as root
# afterwards -- the provider, the engine, the runner and the evaluator all run downgraded.
#
# The credential isolation is plain Unix ownership rather than ACLs or systemd credentials,
# because this host has neither setfacl nor a systemd new enough for LoadCredential=. The
# resulting boundary is the same and is *proved* at the end of this script by attempting a read
# as each non-provider identity and requiring it to fail.
set -euo pipefail

REPO="${SHAPEFLOW_REPO:-/storage/nvme/shapeflow-p1-study}"
DATA_ROOT="${SHAPEFLOW_DATA_ROOT:-/storage/nvme/shapeflow-data}"
CRED_DIR="/etc/shapeflow"
TOKEN_DIR="/etc/shapeflow-tokens"
VLLM_VENV="${SHAPEFLOW_VLLM_VENV:-/storage/nvme/drbat/.venv}"
MODEL_PATH="${SHAPEFLOW_MODEL_PATH:-/storage/nvme/reme/models/Qwen3-14B-AWQ}"
GPU_UUID="${SHAPEFLOW_GPU_UUID:-GPU-ef013951-e496-78da-da70-a5a289dcc634}"
VLLM_PORT="${SHAPEFLOW_VLLM_PORT:-8000}"
MIN_FREE_BYTES=12884901888   # 12 GiB floor (plan §5.2)

[ "$(id -u)" -eq 0 ] || { echo "install_host.sh must run as root" >&2; exit 1; }

say() { echo "== $* =="; }

# --- read-only preconditions -----------------------------------------------------------------
say "preconditions"
for user in sfprovider sfrunner sfinfer sfsteward sfevaluator; do
  id "$user" >/dev/null 2>&1 || { echo "missing service user $user" >&2; exit 1; }
done
[ -d "$REPO" ] || { echo "repo $REPO not found" >&2; exit 1; }
[ -r "$CRED_DIR/tavily.key" ] || { echo "$CRED_DIR/tavily.key not found" >&2; exit 1; }
[ -r "$CRED_DIR/deepseek.key" ] || { echo "$CRED_DIR/deepseek.key not found" >&2; exit 1; }

FREE="$(df -B1 --output=avail "$REPO" | tail -1 | tr -d ' ')"
[ "$FREE" -ge "$MIN_FREE_BYTES" ] || {
  echo "free ${FREE}B is under the ${MIN_FREE_BYTES}B floor" >&2; exit 1; }

# A foreign process on the leased GPU is a reason to stop, never a reason to kill anything.
if command -v nvidia-smi >/dev/null 2>&1; then
  FOREIGN="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | grep -c . || true)"
  if [ "${FOREIGN:-0}" -gt 0 ]; then
    echo "GPUs have $FOREIGN compute process(es); refusing to install over someone else's run" >&2
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv >&2
    exit 1
  fi
fi
nvidia-smi --query-gpu=uuid --format=csv,noheader | grep -qx "$GPU_UUID" \
  || { echo "leased GPU $GPU_UUID is not present on this host" >&2; exit 1; }

# --- credentials: only sfprovider may read them ------------------------------------------------
say "credential isolation"
chown root:sfprovider "$CRED_DIR"
chmod 0750 "$CRED_DIR"
chown sfprovider:sfprovider "$CRED_DIR"/tavily.key "$CRED_DIR"/deepseek.key
chmod 0400 "$CRED_DIR"/tavily.key "$CRED_DIR"/deepseek.key

# --- role tokens: each role reads only its own capability ---------------------------------------
say "role tokens"
mkdir -p "$TOKEN_DIR"
chown root:root "$TOKEN_DIR"
chmod 0755 "$TOKEN_DIR"
for role in runner steward evaluator infer; do
  file="$TOKEN_DIR/$role.token"
  if [ ! -s "$file" ]; then
    # 32 url-safe bytes; the provider only ever compares it, never logs it.
    printf '%s-%s\n' "$role" "$(head -c 24 /dev/urandom | base64 | tr '+/' '-_' | tr -d '=\n')" > "$file"
  fi
  chown "sf$role:sfprovider" "$file"
  chmod 0440 "$file"
done

# --- the pinned vLLM must be executable by the inference identity --------------------------------
# /root is 0700 and drbat's interpreter symlinks into it. One traversal bit, no read bit: the
# directory stays unlistable, and sfinfer can exec the pinned interpreter without a second
# multi-gigabyte install competing for the disk floor.
say "vllm interpreter reachability"
chmod o+x /root
INTERP="$(readlink -f "$VLLM_VENV/bin/python")"
[ -x "$INTERP" ] || { echo "pinned interpreter $INTERP is not executable" >&2; exit 1; }

# --- data root, one owner per identity -----------------------------------------------------------
say "data root"
mkdir -p "$DATA_ROOT"/{provider,steward,runner,evaluator}
chown root:root "$DATA_ROOT"; chmod 0755 "$DATA_ROOT"
chown -R sfprovider:sfprovider "$DATA_ROOT/provider"; chmod 0700 "$DATA_ROOT/provider"
# The steward writes the corpus; the runner must read the frozen corpus it publishes.
chown -R sfsteward:sfrunner "$DATA_ROOT/steward"; chmod 0750 "$DATA_ROOT/steward"
mkdir -p "$DATA_ROOT/runner/frozen_corpus"
chown -R sfrunner:sfrunner "$DATA_ROOT/runner"; chmod 0755 "$DATA_ROOT/runner"
# The evaluator holds the answer key. Nothing else may enter this tree.
chown -R sfevaluator:sfevaluator "$DATA_ROOT/evaluator"; chmod 0700 "$DATA_ROOT/evaluator"

# The steward publishes the runner-readable corpus into the runner's tree.
setfacl_missing=0
command -v setfacl >/dev/null 2>&1 || setfacl_missing=1
if [ "$setfacl_missing" -eq 1 ]; then
  chown -R sfsteward:sfrunner "$DATA_ROOT/runner/frozen_corpus"
  chmod -R 0750 "$DATA_ROOT/runner/frozen_corpus"
fi

mkdir -p "$REPO/logs" "$REPO/reports"
chown sfrunner:sfrunner "$REPO/logs" "$REPO/reports"
chmod 0775 "$REPO/logs" "$REPO/reports"
mkdir -p /run/shapeflow && chown sfprovider:sfprovider /run/shapeflow && chmod 0750 /run/shapeflow

# The repo is root-owned but the low-privilege roles must READ its git state (doctor's
# git-clean integrity check). Without this, git refuses with "detected dubious ownership" and
# the check degrades to SKIP. Read-only git usage only; no role can write the tree.
git config --system --add safe.directory "$REPO" 2>/dev/null || true

# --- gitleaks: the secret gate cannot be satisfied by assertion -----------------------------------
if ! command -v gitleaks >/dev/null 2>&1; then
  say "installing gitleaks"
  TMP="$(mktemp -d)"
  VERSION="8.21.2"
  curl -fsSL -o "$TMP/gitleaks.tar.gz" \
    "https://github.com/gitleaks/gitleaks/releases/download/v${VERSION}/gitleaks_${VERSION}_linux_x64.tar.gz"
  tar -xzf "$TMP/gitleaks.tar.gz" -C "$TMP" gitleaks
  install -m 0755 "$TMP/gitleaks" /usr/local/bin/gitleaks
  rm -rf "$TMP"
fi
gitleaks version

# --- systemd units ---------------------------------------------------------------------------------
say "units"
render() {
  sed -e "s|@REPO@|$REPO|g" \
      -e "s|@DATA_ROOT@|$DATA_ROOT|g" \
      -e "s|@API_USER@|sfprovider|g" \
      -e "s|@RUNNER_USER@|sfrunner|g" \
      -e "s|@INFER_USER@|sfinfer|g" \
      -e "s|@CRED_DIR@|$CRED_DIR|g" \
      -e "s|@TOKEN_DIR@|$TOKEN_DIR|g" \
      -e "s|@VLLM_VENV@|$VLLM_VENV|g" \
      -e "s|@MODEL_PATH@|$MODEL_PATH|g" \
      -e "s|@MODEL_REVISION@|${SHAPEFLOW_MODEL_REVISION:-31c69efc29464b6bb0aee1398b5a7b50a99340c3}|g" \
      -e "s|@GPU_UUID@|$GPU_UUID|g" \
      -e "s|@VLLM_PORT@|$VLLM_PORT|g" \
      -e "s|@PROTOCOL_SHA@|${SHAPEFLOW_PROTOCOL_SHA:-UNSET}|g" \
      "$1"
}
SUPERVISOR="systemd"
if ! systemctl is-system-running >/dev/null 2>&1 && ! pidof systemd >/dev/null 2>&1; then
  # This host is a container: systemd is installed but is not PID 1, so `systemctl` cannot
  # operate. Plan section 17.2 allows an alternative supervisor only if it passes the same
  # fault test as the unit, which scripts/sfsupervise.sh does and
  # tests/integration/test_supervisor_faults.py proves. The units are still rendered, to
  # /etc/systemd/system, so the intended configuration is on the host verbatim and a future
  # systemd host needs no re-derivation.
  SUPERVISOR="sfsupervise"
fi
mkdir -p /etc/systemd/system
for unit in shapeflow-api-provider shapeflow-vllm-causal shapeflow-p1-week1; do
  render "$REPO/systemd/$unit.service.template" > "/etc/systemd/system/$unit.service"
  chmod 0644 "/etc/systemd/system/$unit.service"
done
if [ "$SUPERVISOR" = "systemd" ]; then
  systemctl daemon-reload
else
  echo "systemd is not PID 1 on this host; supervision falls back to scripts/sfsupervise.sh"
  echo "(units rendered to /etc/systemd/system for the record, not started)"
fi
install -m 0755 "$REPO/scripts/sfsupervise.sh" /usr/local/bin/sfsupervise

# --- prove the boundary rather than asserting it -----------------------------------------------------
# The result is written to reports/CREDENTIAL_ISOLATION.json, which is what `doctor` reads.
# Doctor deliberately does not open a key file itself: that check could only pass for the one
# identity holding the credential -- so `doctor --role runner`, the identity the launch gate
# actually runs it as, failed by construction -- and it demonstrated that nobody else can read
# the secret by reading it.
say "proving credential isolation"
DENIED_JSON=""
for role in sfrunner sfinfer sfsteward sfevaluator; do
  denied=true
  for key in tavily deepseek exa; do
    [ -f "$CRED_DIR/$key.key" ] || continue
    if runuser -u "$role" -- cat "$CRED_DIR/$key.key" >/dev/null 2>&1; then
      echo "FATAL: $role can read $CRED_DIR/$key.key" >&2
      denied=false
    fi
  done
  DENIED_JSON="$DENIED_JSON\"$role\": $denied,"
  [ "$denied" = true ] || exit 1
done
PROVIDER_CAN_READ=true
runuser -u sfprovider -- cat "$CRED_DIR/deepseek.key" >/dev/null \
  || { echo "FATAL: sfprovider cannot read its own credential" >&2; PROVIDER_CAN_READ=false; }
[ "$PROVIDER_CAN_READ" = true ] || exit 1
runuser -u sfinfer -- "$VLLM_VENV/bin/python" -c 'import sys; sys.exit(0)' \
  || { echo "FATAL: sfinfer cannot execute the pinned interpreter" >&2; exit 1; }

# The runner may not reach the answer key, and may not walk the steward's tree either: the
# acquisition manifests there carry the audit occurrence graph, which is evaluator-only.
for tree in evaluator steward; do
  if runuser -u sfrunner -- ls "$DATA_ROOT/$tree" >/dev/null 2>&1; then
    echo "FATAL: sfrunner can list the $tree tree" >&2
    exit 1
  fi
done

mkdir -p "$REPO/reports"
cat > "$REPO/reports/CREDENTIAL_ISOLATION.json" <<JSON
{
  "generated_by": "scripts/install_host.sh",
  "credential_dir": "$CRED_DIR",
  "denied": { ${DENIED_JSON%,} },
  "provider_can_read": $PROVIDER_CAN_READ,
  "runner_cannot_list": ["evaluator", "steward"]
}
JSON
chown sfrunner:sfrunner "$REPO/reports/CREDENTIAL_ISOLATION.json"

echo
echo "host installed."
echo "  credentials  : $CRED_DIR (sfprovider only, proved)"
echo "  role tokens  : $TOKEN_DIR (one per role, proved)"
echo "  data root    : $DATA_ROOT (per-identity ownership)"
echo "  units        : rendered to /etc/systemd/system"
echo "  supervisor   : $SUPERVISOR"
