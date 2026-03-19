#!/usr/bin/env bash
set -euo pipefail

# Edit these values directly for your machine.
MODEL="/data/h50056789/Rag_chunk_bench/model/BAAI/bge-reranker-v2-m3"
SERVED_MODEL_NAME="BAAI/bge-reranker-v2-m3"
HOST="127.0.0.1"
PORT="8002"
API_KEY="EMPTY"
GPU="0"
DTYPE="auto"
GPU_MEMORY_UTILIZATION="0.2"
PYTHON_BIN="python"

export CUDA_VISIBLE_DEVICES="$GPU"
export NVIDIA_VISIBLE_DEVICES="$GPU"

echo "Starting vLLM rerank server on cuda:$GPU"
echo "  model=$MODEL"
echo "  served_model_name=$SERVED_MODEL_NAME"
echo "  host=$HOST"
echo "  port=$PORT"
echo "  route=http://$HOST:$PORT/rerank"
echo "  gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"

exec "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --api-key "$API_KEY" \
  --dtype "$DTYPE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --runner pooling \
  --served-model-name "$SERVED_MODEL_NAME"
