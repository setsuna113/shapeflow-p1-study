#!/usr/bin/env bash
# Start the causal vLLM engine under supervision, as the inference identity.
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
# flags and do not touch the causal invariants (max-num-seqs 1, APC off, chunked prefill off).
set -euo pipefail

REPO="${SHAPEFLOW_REPO:-/storage/nvme/shapeflow-p1-study}"
DATA_ROOT="${SHAPEFLOW_DATA_ROOT:-/storage/nvme/shapeflow-data}"
VLLM_VENV="${SHAPEFLOW_VLLM_VENV:-/storage/nvme/drbat/.venv}"
MODEL_PATH="${SHAPEFLOW_MODEL_PATH:-/storage/nvme/reme/models/Qwen3-14B-AWQ}"
# No default device. A hard-coded UUID meant the campaign could only ever run on one card and
# refused to start whenever that card was busy -- including when it was busy with our own
# previous engine. The caller resolves a free device from the hash-locked pool
# (shapeflow_p1.ops.gpu_pool) and passes it in.
GPU_UUID="${SHAPEFLOW_GPU_UUID:?SHAPEFLOW_GPU_UUID is required; select one from host.gpu_uuid_pool}"
PORT="${SHAPEFLOW_VLLM_PORT:-8000}"
ENGINE_EPOCH_FILE="${SHAPEFLOW_ENGINE_EPOCH_FILE:-/run/shapeflow-vllm-causal/engine_epoch}"

[ "$(id -u)" -eq 0 ] || { echo "start_engine.sh switches identity and must run as root" >&2; exit 1; }

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

exec setsid /usr/local/bin/sfsupervise vllm-causal sfinfer "$REPO" "$DATA_ROOT" -- \
  env HOME=/tmp \
      PATH="$VLLM_VENV/bin:/usr/local/bin:/usr/bin:/bin" \
      CUDA_VISIBLE_DEVICES="$GPU_UUID" \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0 \
  "$VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_PATH" --served-model-name Qwen3-14B-AWQ \
      --host 127.0.0.1 --port "$PORT" \
      --max-model-len 16384 --gpu-memory-utilization 0.90 \
      --max-num-seqs 1 \
      --no-enable-prefix-caching \
      --no-enable-chunked-prefill \
      --enable-auto-tool-choice --tool-call-parser ${SHAPEFLOW_TOOL_PARSER:-hermes}
