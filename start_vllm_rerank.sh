#!/usr/bin/env bash
set -euo pipefail

# Edit these values directly for your machine.
MODEL="BAAI/bge-reranker-v2-m3"
HOST="127.0.0.1"
PORT="8002"
API_KEY="EMPTY"
GPU="1"
DTYPE="auto"
PYTHON_BIN="python"

export CUDA_VISIBLE_DEVICES="$GPU"
export NVIDIA_VISIBLE_DEVICES="$GPU"

echo "Starting vLLM rerank server on cuda:$GPU"
echo "  model=$MODEL"
echo "  host=$HOST"
echo "  port=$PORT"
echo "  route=http://$HOST:$PORT/rerank"

exec "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --api-key "$API_KEY" \
  --dtype "$DTYPE" \
  --task rerank
