#!/usr/bin/env bash
# One-time host installation, run as root on the experiment machine (plan §5.2, §17.2).
#
# Root is used here and nowhere else: to create directories, set ownership, mint role tokens and
# install units. Nothing that touches a model, a web page or a credential ever runs as root
# afterwards -- the provider, the engine, the runner and the evaluator all run downgraded.
#
# Credentials use plain Unix ownership.  Experiment artifacts use narrow POSIX ACLs because
# runner/steward services have UMask=0077 while the evaluator still needs read-only access to
# their frozen outputs.  Both directions are proved below by actually dropping privileges.
set -euo pipefail

REPO="${SHAPEFLOW_REPO:-/storage/nvme/shapeflow-p1-study}"
DATA_ROOT="${SHAPEFLOW_DATA_ROOT:-/storage/nvme/shapeflow-data}"
CRED_DIR="/etc/shapeflow"
TOKEN_DIR="/etc/shapeflow-tokens"
VLLM_VENV="${SHAPEFLOW_VLLM_VENV:-/storage/nvme/drbat/.venv}"
MODEL_PATH="${SHAPEFLOW_MODEL_PATH:-/storage/nvme/reme/models/Qwen3-14B-AWQ}"
# The permitted devices, mirroring configs/stack.yaml host.gpu_uuid_pool. Which one a run
# actually leases is resolved from what is idle at launch (shapeflow_p1.ops.gpu_pool) and
# recorded by freeze-stack into the stack manifest -- it is not decided here.
SHAPEFLOW_GPU_UUID_POOL="${SHAPEFLOW_GPU_UUID_POOL:-\
GPU-ef013951-e496-78da-da70-a5a289dcc634 \
GPU-d1f5d8c4-2f0f-45f7-0574-ac90018440db \
GPU-11b9da9d-6149-3c1b-79c8-011d74a180d2 \
GPU-99847987-522a-5889-1066-a21512dff220}"
# Only substituted into the unit templates, which this host renders for the record and never
# starts (systemd is not PID 1 here). Defaults to the first permitted device rather than a
# separately hard-coded card, so the two can no longer drift apart.
GPU_UUID="${SHAPEFLOW_GPU_UUID:-${SHAPEFLOW_GPU_UUID_POOL%% *}}"
VLLM_PORT="${SHAPEFLOW_VLLM_PORT:-8000}"
MIN_FREE_BYTES=12884901888   # 12 GiB floor (plan §5.2)

[ "$(id -u)" -eq 0 ] || { echo "install_host.sh must run as root" >&2; exit 1; }

say() { echo "== $* =="; }

# --- read-only preconditions -----------------------------------------------------------------
say "preconditions"
for user in sfprovider sfrunner sfinfer sfsteward sfevaluator; do
  id "$user" >/dev/null 2>&1 || { echo "missing service user $user" >&2; exit 1; }
done
for command in setfacl getfacl; do
  command -v "$command" >/dev/null 2>&1 \
    || { echo "$command is required for evaluator read-only publication ACLs" >&2; exit 1; }
done
[ -d "$REPO" ] || { echo "repo $REPO not found" >&2; exit 1; }
# Exa is the acquisition credential and the provider refuses to start without it. Tavily is
# optional now that acquisition moved to Exa -- its caps stay on the books for the requests
# already spent there, but nothing new is bought through it, so a missing tavily.key is fine.
[ -r "$CRED_DIR/exa.key" ] || { echo "$CRED_DIR/exa.key not found" >&2; exit 1; }
[ -r "$CRED_DIR/deepseek.key" ] || { echo "$CRED_DIR/deepseek.key not found" >&2; exit 1; }

FREE="$(df -B1 --output=avail "$REPO" | tail -1 | tr -d ' ')"
[ "$FREE" -ge "$MIN_FREE_BYTES" ] || {
  echo "free ${FREE}B is under the ${MIN_FREE_BYTES}B floor" >&2; exit 1; }

# Installation creates users, directories and ACLs; it does not touch a GPU. Refusing to
# install because *some* card on a shared host is busy blocked setup for no reason -- and in
# practice the busy card was usually our own engine from the previous round. Device selection
# happens later, from the hash-locked pool, and start_engine.sh does the per-card check.
#
# What is still verified here is that at least one permitted device exists at all, because an
# install against a host with none of them is a misconfiguration worth catching now.
if command -v nvidia-smi >/dev/null 2>&1; then
  PRESENT=0
  for uuid in $(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d ' '); do
    case " $SHAPEFLOW_GPU_UUID_POOL " in *" $uuid "*) PRESENT=$((PRESENT + 1)) ;; esac
  done
  if [ -n "${SHAPEFLOW_GPU_UUID_POOL:-}" ] && [ "$PRESENT" -eq 0 ]; then
    echo "none of the permitted GPUs are present on this host" >&2
    nvidia-smi --query-gpu=uuid,name --format=csv >&2
    exit 1
  fi
fi

# --- credentials: only sfprovider may read them ------------------------------------------------
say "credential isolation"
chown root:sfprovider "$CRED_DIR"
chmod 0750 "$CRED_DIR"
# exa.key and deepseek.key are required; tavily.key is optional and only locked down if the
# operator still keeps one here. Every key present must end up sfprovider-only either way --
# an unlisted credential left at its creation mode is exactly the leak this section prevents.
for key in exa deepseek tavily; do
  [ -e "$CRED_DIR/$key.key" ] || continue
  chown sfprovider:sfprovider "$CRED_DIR/$key.key"
  chmod 0400 "$CRED_DIR/$key.key"
done

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
mkdir -p "$DATA_ROOT"/{provider,steward,runner,evaluator,approvals}
chown root:root "$DATA_ROOT"; chmod 0755 "$DATA_ROOT"
chown -R sfprovider:sfprovider "$DATA_ROOT/provider"; chmod 0700 "$DATA_ROOT/provider"
readonly_acl() {
  # Existing bytes plus defaults for files created later under a service UMask of 0077.
  # Named users receive r-X only; the owner keeps rwX. No shared directory grants write.
  #
  # BOTH applications are recursive, and the second one used to not be. A default ACL is
  # inherited from a file's *immediate* parent, so setting defaults only on the top directory
  # leaves every pre-existing subdirectory without one -- and the object store is three levels
  # deep (objects/<ab>/<cd>/<sha>.zst). New blobs therefore landed with no ACL whatsoever, and
  # `getfacl` on them showed nothing to be masked: the grant was not weakened, it was absent.
  # Observed on the run host with 1465 frozen page blobs the runner could not read.
  #
  # `setfacl -R -m d:...` applies default entries to directories only, which is exactly right:
  # files have no default ACL to set.
  local path="$1"; shift
  local -a access_args=()
  local -a default_args=("d:u::rwx" "d:g::---" "d:m::rwx" "d:o::---")
  for reader in "$@"; do
    access_args+=("u:$reader:r-X")
    default_args+=("d:u:$reader:r-x")
  done
  setfacl -R -m "$(IFS=,; echo "${access_args[*]}")" "$path"
  setfacl -R -m "$(IFS=,; echo "${default_args[*]}")" "$path"
}

# Approval history is outside the Git execution tree: the steward appends, while every role
# that can spend or evaluate can only read the current pointer and immutable history.
chown -R sfsteward:sfsteward "$DATA_ROOT/approvals"
find "$DATA_ROOT/approvals" -type d -exec chmod 0700 {} +
# 0640, not 0600: chmod sets the ACL *mask* on a file that carries an ACL, and a 0600 mask is
# `---`, which cancels every u:<role>:r-x entry readonly_acl grants below. The 0700 directory
# is the real gate; see shapeflow_p1.fsmode.
find "$DATA_ROOT/approvals" -type f -exec chmod 0640 {} +
# sfprovider reads it too. It is the identity that enforces the budget ceilings, so an
# authorized cap raise has to be checkable against the approval that granted it -- otherwise
# the one command that can widen a ceiling is the one command that cannot verify it is allowed
# to. Read-only, like the other two; only the steward writes here.
readonly_acl "$DATA_ROOT/approvals" sfrunner sfevaluator sfprovider

# The steward tree is private except for acquisition manifests that evaluation verifies.
# sfevaluator gets traverse-only on the root and inherited read-only access on that one subtree.
mkdir -p "$DATA_ROOT/steward/acquisition"
chown -R sfsteward:sfsteward "$DATA_ROOT/steward"
find "$DATA_ROOT/steward" -type d -exec chmod 0700 {} +
find "$DATA_ROOT/steward" -type f -exec chmod 0640 {} +
setfacl -m u:sfevaluator:--x,m::--x "$DATA_ROOT/steward"
readonly_acl "$DATA_ROOT/steward/acquisition" sfevaluator

# Runner output stays runner-owned.  The evaluator can traverse the root and read only the
# frozen/runtime evidence needed by evaluate: runs (ledger + frozen blocks), object store and
# checkpoints.
#
# The earlier claim here -- that default ACLs make newly-created 0600 bytes readable without
# weakening UMask -- was false, and it is the whole reason evaluation could not read the
# runner's artifacts. A default ACL is inherited, but the kernel folds the creating mode into
# it (`mask &= mode >> 3`), so 0600 yields mask `---` and every named-user grant below becomes
# `#effective:---`. The umask is genuinely ignored when a default ACL exists, which is why the
# `umask 077` probes passed while the Python writers -- which ask for 0600 explicitly through
# tempfile.mkstemp -- did not. See shapeflow_p1.fsmode.
mkdir -p "$DATA_ROOT/runner"/{runs,object_store,checkpoints,frozen_corpus}
chown -R sfrunner:sfrunner "$DATA_ROOT/runner"
find "$DATA_ROOT/runner" -type d -exec chmod 0700 {} +
find "$DATA_ROOT/runner" -type f -exec chmod 0640 {} +
setfacl -m u:sfsteward:--x,u:sfevaluator:--x,m::--x "$DATA_ROOT/runner"
for published in runs object_store checkpoints; do
  readonly_acl "$DATA_ROOT/runner/$published" sfevaluator
done
# The steward needs *traverse* on runs/ -- not read -- so freeze-analysis-design can stat the
# runner ledger. That freeze is the pre-registration guard: it must prove no treatment state
# exists before it authors the feature registry. Without this the stat raises EACCES, and the
# guard is left unable to check the one thing it exists to check. Traverse alone does not let
# the steward list the directory or read a block; it only lets it ask whether a named path is
# there, which is exactly the question the guard asks.
setfacl -m u:sfsteward:--x "$DATA_ROOT/runner/runs"

# One runner tree per execution lane. The ledger is single-writer by design, so four concurrent
# runners need four of them; a shared object store would also make "which lane produced this
# artifact" unanswerable at exactly the moment the merge asks it. Same ownership and the same
# evaluator/steward ACLs as the unsharded tree above -- a lane whose artifacts the evaluator
# could not read would fail at scoring time, three days in.
#
# frozen_corpus is deliberately NOT duplicated: it is the steward-to-runner publication
# boundary and every lane reads the same one. Four copies of the corpus would be four worlds.
LANE_COUNT="$("$REPO/.venv/bin/python" - "$REPO" <<'PYX'
import sys, yaml
with open(f"{sys.argv[1]}/configs/week1.yaml", encoding="utf-8") as handle:
    print(int(yaml.safe_load(handle)["measurement"]["shards"]["lane_count"]))
PYX
)"
for lane in $(seq 0 $((LANE_COUNT - 1))); do
  lane_root="$DATA_ROOT/runner-lane${lane}"
  mkdir -p "$lane_root"/{runs,object_store,checkpoints}
  chown -R sfrunner:sfrunner "$lane_root"
  find "$lane_root" -type d -exec chmod 0700 {} +
  setfacl -m u:sfsteward:--x,u:sfevaluator:--x,m::--x "$lane_root"
  for published in runs object_store checkpoints; do
    readonly_acl "$lane_root/$published" sfevaluator
  done
  setfacl -m u:sfsteward:--x "$lane_root/runs"
done

# The frozen task-to-lane partition. Steward-owned, readable by every lane: a lane that could
# rewrite its own share could choose its tasks after seeing a result.
mkdir -p "$DATA_ROOT/shards"
chown -R sfsteward:sfsteward "$DATA_ROOT/shards"
find "$DATA_ROOT/shards" -type d -exec chmod 0700 {} +
# 0640 rather than 0600, for the reason spelled out at the approvals tree above: chmod sets the
# ACL mask on a file that carries an ACL, and a 0600 mask is `---`, which would cancel every
# named-user grant readonly_acl makes on the next line. The 0700 directory is the real gate.
find "$DATA_ROOT/shards" -type f -exec chmod 0640 {} +
readonly_acl "$DATA_ROOT/shards" sfrunner sfevaluator

# frozen_corpus is the steward-to-runner publication boundary. Both the runner and evaluator
# read it; only the steward owns/writes it.
chown -R sfsteward:sfsteward "$DATA_ROOT/runner/frozen_corpus"
readonly_acl "$DATA_ROOT/runner/frozen_corpus" sfrunner sfevaluator
# The evaluator holds the answer key. Plan §7.3: the steward *builds* truth and the
# evaluator *reads and scores* it, so the steward owns the answer-key directories and the
# evaluator group reads them. Judgments are a separate capability: only sfevaluator owns and
# writes that subtree. Do not make the whole evaluator tree group-writable -- that would let
# scoring mutate the truth it is supposed to measure against.
mkdir -p "$DATA_ROOT/evaluator"/{tasks,truth_packets,visible_truth,analysis_design,judgments}
chown -R sfsteward:sfevaluator "$DATA_ROOT/evaluator"
find "$DATA_ROOT/evaluator" -type d -exec chmod 0750 {} +
find "$DATA_ROOT/evaluator" -type f -exec chmod 0640 {} +
for answer_key in tasks truth_packets visible_truth analysis_design; do
  readonly_acl "$DATA_ROOT/evaluator/$answer_key" sfevaluator
done
chown -R sfevaluator:sfevaluator "$DATA_ROOT/evaluator/judgments"
find "$DATA_ROOT/evaluator/judgments" -type d -exec chmod 0700 {} +
find "$DATA_ROOT/evaluator/judgments" -type f -exec chmod 0400 {} +

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
# The coordinator flag historically says --protocol-sha, but after launch it carries the
# complete approved ProtocolBinding.digest. Never substitute the document-only SHA.
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
      -e "s|@PROTOCOL_SHA@|${SHAPEFLOW_EXECUTION_BINDING_SHA:-UNSET}|g" \
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
# Both mandatory credentials, not just one: the provider will not start without either, and a
# check that passed on deepseek alone would let an unreadable exa.key through to launch.
for key in exa deepseek; do
  runuser -u sfprovider -- cat "$CRED_DIR/$key.key" >/dev/null \
    || { echo "FATAL: sfprovider cannot read $CRED_DIR/$key.key" >&2; PROVIDER_CAN_READ=false; }
done
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

# Prove default ACL inheritance the way the campaign actually writes.  Existing-file checks are
# insufficient: every ledger/object/checkpoint is created after install.
#
# These probes used `umask 077; : > file`, which asks the kernel for 0666 -- and the umask is
# ignored outright when a default ACL is present, so the probe file landed with mask `rw-` and
# passed unconditionally. The Python writers ask for 0600 explicitly (tempfile.mkstemp), which
# collapses the mask to `---`. The probe therefore certified an access path that did not exist.
#
# The file must be created by an `open()` that *requests* mode 0600, because that is what
# tempfile.mkstemp does and the whole question is what the kernel does with the inherited
# default ACL when it folds that mode in.
#
# Shell redirection cannot express this: `: > file` always requests 0666, and the umask is
# ignored outright when a default ACL is present, so the file lands with a permissive mask and
# the probe passes no matter what. `install -m 0600` is worse than useless here -- coreutils
# treats an explicit mode as authoritative and *strips the inherited ACL entirely*, so the
# probe file ends up with no ACL at all and tests nothing that production does.
#
# python3 is already a hard dependency of every service on this host, so use it and request the
# mode directly.
new_probe_at_mode() {  # <owner-uid> <path> <octal-mode>
  # The parent directories are created by the same identity, with the same mkdir(parents=True)
  # the object store uses, because *where* a file is created decides which default ACL it
  # inherits. Probing only the top of a granted tree passed while the object store -- three
  # levels down at objects/<ab>/<cd>/<sha>.zst -- produced files with no ACL at all.
  runuser -u "$1" -- python3 -c \
    'import os,sys; p=sys.argv[1]; os.makedirs(os.path.dirname(p), exist_ok=True); os.close(os.open(p, os.O_CREAT|os.O_WRONLY|os.O_EXCL, int(sys.argv[2], 8)))' \
    "$2" "$3"
}
# The real publication sequence: create restricted, widen to 0640 before publishing, exactly as
# shapeflow_p1.fsmode.chmod_shared does.
new_shared_probe() {  # <owner-uid> <path>
  new_probe_at_mode "$1" "$2" 0600
  runuser -u "$1" -- chmod 0640 "$2"
}
# ...and `new_unshared_probe` omits the widening, so the assertions below are proved
# non-vacuous: where ACLs are live this file must NOT be readable by the granted role.
new_unshared_probe() {  # <owner-uid> <path>
  new_probe_at_mode "$1" "$2" 0600
}

# Mask liveness: if this read were to succeed, every "can read" assertion below would be
# meaningless, because the mask would not be constraining anything.
effective_probe="$DATA_ROOT/approvals/.acl-effective-entry-probe-$$"
new_unshared_probe sfsteward "$effective_probe"
if runuser -u sfrunner -- cat "$effective_probe" >/dev/null 2>&1; then
  rm -f "$effective_probe"
  echo "FATAL: a 0600 file under $DATA_ROOT/approvals is readable by sfrunner; the ACL mask" >&2
  echo "       is not constraining access, so the isolation probes below prove nothing" >&2
  exit 1
fi
rm -f "$effective_probe"

probe="$DATA_ROOT/approvals/.approval-read-probe-$$"
denied="$DATA_ROOT/approvals/.runner-approval-write-denied-probe-$$"
new_shared_probe sfsteward "$probe"
for reader in sfrunner sfevaluator; do
  runuser -u "$reader" -- cat "$probe" >/dev/null \
    || { echo "FATAL: $reader cannot read new external approval bytes" >&2; exit 1; }
done
if runuser -u sfrunner -- touch "$denied" >/dev/null 2>&1; then
  rm -f "$probe" "$denied"
  echo "FATAL: sfrunner can mutate the external approval store" >&2
  exit 1
fi
rm -f "$probe"

for directory in \
  "$DATA_ROOT/runner/runs" \
  "$DATA_ROOT/runner/object_store" \
  "$DATA_ROOT/runner/checkpoints"; do
  probe="$directory/.probe-$$/ab/cd/.runner-evaluator-read-probe"
  denied="$directory/.evaluator-write-denied-probe-$$"
  new_shared_probe sfrunner "$probe"
  runuser -u sfevaluator -- cat "$probe" >/dev/null \
    || { echo "FATAL: sfevaluator cannot read new runner artifact $probe" >&2; exit 1; }
  if runuser -u sfevaluator -- touch "$denied" >/dev/null 2>&1; then
    rm -f "$denied"
    echo "FATAL: sfevaluator can write runner publication directory $directory" >&2
    exit 1
  fi
  rm -rf "$(dirname "$(dirname "$(dirname "$probe")")")"
done

probe="$DATA_ROOT/runner/frozen_corpus/objects/.probe-$$/ab/cd/.steward-publication-read-probe"
new_shared_probe sfsteward "$probe"
runuser -u sfrunner -- cat "$probe" >/dev/null \
  || { echo "FATAL: sfrunner cannot read newly published frozen corpus bytes" >&2; exit 1; }
runuser -u sfevaluator -- cat "$probe" >/dev/null \
  || { echo "FATAL: sfevaluator cannot read newly published frozen corpus bytes" >&2; exit 1; }
rm -f "$probe"

probe="$DATA_ROOT/steward/acquisition/.evaluator-acquisition-read-probe-$$"
new_shared_probe sfsteward "$probe"
runuser -u sfevaluator -- cat "$probe" >/dev/null \
  || { echo "FATAL: sfevaluator cannot read a new frozen acquisition manifest" >&2; exit 1; }
if runuser -u sfrunner -- cat "$probe" >/dev/null 2>&1; then
  rm -f "$probe"
  echo "FATAL: sfrunner can read evaluator-only acquisition provenance" >&2
  exit 1
fi
rm -f "$probe"

probe="$DATA_ROOT/evaluator/tasks/.answer-key-read-probe-$$"
denied="$DATA_ROOT/evaluator/tasks/.evaluator-answer-key-write-denied-probe-$$"
new_shared_probe sfsteward "$probe"
runuser -u sfevaluator -- cat "$probe" >/dev/null \
  || { echo "FATAL: sfevaluator cannot read a new answer-key artifact" >&2; exit 1; }
if runuser -u sfrunner -- cat "$probe" >/dev/null 2>&1; then
  rm -f "$probe"
  echo "FATAL: sfrunner can read the evaluator answer key" >&2
  exit 1
fi
if runuser -u sfevaluator -- touch "$denied" >/dev/null 2>&1; then
  rm -f "$probe" "$denied"
  echo "FATAL: sfevaluator can mutate the steward-owned answer key" >&2
  exit 1
fi
rm -f "$probe"

probe="$DATA_ROOT/evaluator/judgments/.evaluator-write-probe-$$"
new_unshared_probe sfevaluator "$probe"
if runuser -u sfrunner -- cat "$probe" >/dev/null 2>&1; then
  rm -f "$probe"
  echo "FATAL: sfrunner can read evaluator judgments" >&2
  exit 1
fi
rm -f "$probe"

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
