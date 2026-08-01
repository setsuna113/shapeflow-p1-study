#!/usr/bin/env bash
# Start the API provider under supervision, as the provider identity.
#
# This script exists because its absence was a real defect. On a host where systemd is not PID 1
# the units in systemd/ are rendered for the record but never started, so the provider has to be
# launched through sfsupervise -- and sfsupervise passes no credential environment of its own,
# it only inherits the caller's. The provider was therefore started by hand with
# TAVILY_API_KEY_FILE and DEEPSEEK_API_KEY_FILE and no EXA_API_KEY_FILE, which is exactly the
# combination that cannot acquire anything: runtime.provider_main.build_service calls
# load_exa_key unconditionally and raises without it.
#
# The credential *paths* are exported here; the values are read once by the provider itself,
# registered with the redactor, and never enter argv, a child environment, or a log line. Do not
# add `set -x` to this file.
set -euo pipefail

REPO="${SHAPEFLOW_REPO:-/storage/nvme/shapeflow-p1-study}"
DATA_ROOT="${SHAPEFLOW_DATA_ROOT:-/storage/nvme/shapeflow-data}"
CRED_DIR="${SHAPEFLOW_CRED_DIR:-/etc/shapeflow}"
CONFIG="${SHAPEFLOW_CONFIG:-$REPO/configs/week1.yaml}"
# Which execution lane this provider serves. Each lane runs its own provider bound to its own
# port and pointed at its own engine; only the lane named by measurement.shards.paid_upstream_lane
# may reach an upstream that costs money, and the provider refuses those routes on the others.
# Four providers each admitting against the full DeepSeek cap would be a four-fold budget.
LANE="${SHAPEFLOW_LANE:-}"
UNIT_NAME="provider${LANE:+-lane$LANE}"

[ "$(id -u)" -eq 0 ] || { echo "start_provider.sh switches identity and must run as root" >&2; exit 1; }

# Fail here, not three gates later inside a Python traceback. Exa and DeepSeek are both
# mandatory; Tavily is optional since acquisition moved to Exa, and is only passed when present.
for key in exa deepseek; do
  runuser -u sfprovider -- test -r "$CRED_DIR/$key.key" \
    || { echo "sfprovider cannot read $CRED_DIR/$key.key" >&2; exit 1; }
done

CREDENTIAL_ENV=(
  "EXA_API_KEY_FILE=$CRED_DIR/exa.key"
  "DEEPSEEK_API_KEY_FILE=$CRED_DIR/deepseek.key"
)
if [ -r "$CRED_DIR/tavily.key" ]; then
  CREDENTIAL_ENV+=("TAVILY_API_KEY_FILE=$CRED_DIR/tavily.key")
fi

exec setsid /usr/local/bin/sfsupervise "$UNIT_NAME" sfprovider "$REPO" "$DATA_ROOT" -- \
  env HOME=/tmp "${CREDENTIAL_ENV[@]}" ${LANE:+SHAPEFLOW_LANE="$LANE"} \
  "$REPO/.venv/bin/shapeflow" serve-provider --config "$CONFIG"
