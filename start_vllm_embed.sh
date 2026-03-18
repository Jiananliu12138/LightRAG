#!/usr/bin/env bash
set -euo pipefail

# Edit these values directly for your machine.
MODEL="BAAI/bge-large-en-v1.5"
HOST="127.0.0.1"
PORT="8003"
API_KEY="EMPTY"
GPU="1"
DTYPE="auto"
PYTHON_BIN="python"

export CUDA_VISIBLE_DEVICES="$GPU"
export NVIDIA_VISIBLE_DEVICES="$GPU"

echo "Starting vLLM embedding server on cuda:$GPU"
echo "  model=$MODEL"
echo "  host=$HOST"
echo "  port=$PORT"

exec "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --api-key "$API_KEY" \
  --dtype "$DTYPE" \
  --task embed
