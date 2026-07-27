#!/usr/bin/env bash
# Start one measurement layer's vLLM engine under supervision, as the inference identity.
#
# The engine flags are READ FROM configs/stack.yaml rather than written here. They were
# hard-coded, and the two copies are exactly the kind of pair that drifts: doctor compares the
# running engine against the frozen manifest, and a manifest that agrees with a config the
# launcher ignored proves nothing. `isolation.<layer>` is the single source.
#
# SHAPEFLOW_LAYER selects the layer. `causal_native` is the primary one: causal because prefix
# caching is off, native because it does NOT serialize the graph. The older `causal` layer
# admits one upstream request at a time and is now a mechanism control -- that serialization was
# never an isolation requirement, it was a requirement of the summed-service metric, and
# imposing it on a graph that summarises a result set with asyncio.gather produced 212 vendor
# timeouts on the largest pages.
#
# SHAPEFLOW_LANE selects which of the four task-sharded lanes this engine serves. Every lane gets
# its own port, epoch file and supervisor name, because a lane must be independently
# restartable: a block is paired-valid only under one engine epoch, so an engine restart may
# invalidate its own lane's in-flight task and must not touch any other lane's.
#
# Two environment settings are load-bearing on this host and are therefore recorded here rather
# than discovered again later:
#
# PATH must include the pinned venv's bin. Without it flashinfer's JIT cannot find `ninja` and
# the engine dies during KV-cache initialisation.
#
# VLLM_ATTENTION_BACKEND=FLASH_ATTN pins the prebuilt kernel. The default selection tries
# FlashInfer, which JIT-compiles against the system CUDA -- and the system CUDA here is 11,
# while torch is built for 12.9, so the compile fails with "CUDA versions below 12 are not
# supported". Pinning the backend is also the honest thing to do for the measurement: which
# attention kernel served the run is part of the frozen stack, and `freeze-stack` records it
# from this engine's own log.
#
# --enable-auto-tool-choice + --tool-call-parser are required to serve tool calls at all: the
# ODR graph binds tools (tavily_search, ConductResearch, ResearchComplete, think_tool), and
# without these vLLM answers 400 on every tool-bound request. They are serving-correctness
# flags and do not touch the causal invariant, which is prefix caching off.
set -euo pipefail

REPO="${SHAPEFLOW_REPO:-/storage/nvme/shapeflow-p1-study}"
DATA_ROOT="${SHAPEFLOW_DATA_ROOT:-/storage/nvme/shapeflow-data}"
VLLM_VENV="${SHAPEFLOW_VLLM_VENV:-/storage/nvme/drbat/.venv}"
MODEL_PATH="${SHAPEFLOW_MODEL_PATH:-/storage/nvme/reme/models/Qwen3-14B-AWQ}"
LAYER="${SHAPEFLOW_LAYER:-causal_native}"
LANE="${SHAPEFLOW_LANE:-0}"
# No default device. A hard-coded UUID meant the campaign could only ever run on one card and
# refused to start whenever that card was busy -- including when it was busy with our own
# previous engine. The caller resolves a free device from the hash-locked pool
# (shapeflow_p1.ops.gpu_pool) and passes it in.
GPU_UUID="${SHAPEFLOW_GPU_UUID:?SHAPEFLOW_GPU_UUID is required; select one from host.gpu_uuid_pool}"
PORT="${SHAPEFLOW_VLLM_PORT:-$((8000 + LANE))}"
UNIT_NAME="vllm-${LAYER//_/-}-lane${LANE}"
ENGINE_EPOCH_FILE="${SHAPEFLOW_ENGINE_EPOCH_FILE:-/run/shapeflow-${UNIT_NAME}/engine_epoch}"

case "$LANE" in ''|*[!0-9]*) echo "SHAPEFLOW_LANE must be a non-negative integer" >&2; exit 1;; esac

[ "$(id -u)" -eq 0 ] || { echo "start_engine.sh switches identity and must run as root" >&2; exit 1; }

# The frozen flags for this layer AND the engine block, emitted by the same config loader
# everything else reads. max-model-len was hard-coded here while stack.yaml declared its own
# value -- two copies of a number that must agree, which is the drift shape that has already
# cost this study three separate launches.
read -r MAX_NUM_SEQS PREFIX_CACHING CHUNKED_PREFILL MAX_MODEL_LEN GPU_MEM_UTIL <<EOF
$("${SHAPEFLOW_PYTHON:-$REPO/.venv/bin/python}" - "$REPO" "$LAYER" <<'PY'
import sys, yaml
repo, layer = sys.argv[1], sys.argv[2]
with open(f"{repo}/configs/stack.yaml", encoding="utf-8") as handle:
    stack = yaml.safe_load(handle)
isolation = stack["isolation"]
if layer not in isolation:
    raise SystemExit(f"configs/stack.yaml declares no isolation layer {layer!r}")
block = isolation[layer]
print(
    int(block["max_num_seqs"]),
    "on" if block["enable_prefix_caching"] else "off",
    "on" if block["enable_chunked_prefill"] else "off",
    int(stack["engine"]["max_model_len"]),
    stack["engine"]["gpu_memory_utilization"],
)
PY
)
EOF
[ -n "${MAX_NUM_SEQS:-}" ] && [ -n "${MAX_MODEL_LEN:-}" ] \
  || { echo "could not read engine/isolation.$LAYER from configs/stack.yaml" >&2; exit 1; }

APC_FLAG="--no-enable-prefix-caching"
[ "$PREFIX_CACHING" = "on" ] && APC_FLAG="--enable-prefix-caching"
PREFILL_FLAG="--no-enable-chunked-prefill"
[ "$CHUNKED_PREFILL" = "on" ] && PREFILL_FLAG="--enable-chunked-prefill"

# Never start over a foreign process on the LEASED card. Both halves of this were host-wide and
# therefore wrong: `--query-compute-apps` without `-i` counts processes on every GPU, so a
# neighbour's job on a card we do not want blocked us; and a bare `pgrep -f api_server` counts
# any engine of ours anywhere, so "ours" on another card looked like permission to start here.
FOREIGN="$(nvidia-smi -i "$GPU_UUID" --query-compute-apps=pid --format=csv,noheader \
  | tr -d ' ' | grep -c . || true)"
MINE="$(pgrep -u sfinfer -f "api_server.*--port $PORT" | wc -l || true)"
if [ "${FOREIGN:-0}" -gt 0 ] && [ "${MINE:-0}" -eq 0 ]; then
  echo "GPU $GPU_UUID has $FOREIGN compute process(es) that are not ours; refusing to start" >&2
  nvidia-smi -i "$GPU_UUID" --query-compute-apps=pid,used_memory --format=csv >&2
  exit 1
fi

# Manual/supervised launches need the same per-boot identity as the systemd unit.  This happens
# once immediately before the supervisor takes ownership of the engine process.
"${SHAPEFLOW_PYTHON:-$REPO/.venv/bin/python}" \
  "$REPO/scripts/write_engine_epoch.py" "$ENGINE_EPOCH_FILE"

exec setsid /usr/local/bin/sfsupervise "$UNIT_NAME" sfinfer "$REPO" "$DATA_ROOT" -- \
  env HOME=/tmp \
      PATH="$VLLM_VENV/bin:/usr/local/bin:/usr/bin:/bin" \
      CUDA_VISIBLE_DEVICES="$GPU_UUID" \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0 \
  "$VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_PATH" --served-model-name Qwen3-14B-AWQ \
      --host 127.0.0.1 --port "$PORT" \
      --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM_UTIL" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      "$APC_FLAG" \
      "$PREFILL_FLAG" \
      --enable-auto-tool-choice --tool-call-parser ${SHAPEFLOW_TOOL_PARSER:-hermes}
