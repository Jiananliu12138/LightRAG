#!/usr/bin/env bash
set -euo pipefail

# Edit these values directly for your machine.
MODEL="/data/h50056789/Rag_Chunking/model/Qwen/Qwen2.5-7B-Instruct"
SERVED_MODEL_NAME="Qwen2.5-7B-Instruct"
HOST="127.0.0.1"
PORT="8005"
API_KEY="EMPTY"
GPU="0"
DTYPE="auto"
GPU_MEMORY_UTILIZATION="0.5"
PYTHON_BIN="python"

export CUDA_VISIBLE_DEVICES="$GPU"
export NVIDIA_VISIBLE_DEVICES="$GPU"

echo "Starting vLLM LLM server on cuda:$GPU"
echo "  model=$MODEL"
echo "  served_model_name=$SERVED_MODEL_NAME"
echo "  host=$HOST"
echo "  port=$PORT"
echo "  gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"

exec "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --api-key "$API_KEY" \
  --dtype "$DTYPE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --served-model-name "$SERVED_MODEL_NAME"
