#!/bin/bash
# 启动 vLLM Rerank 服务器

echo "=========================================="
echo "  启动 vLLM Rerank 服务"
echo "=========================================="

# 模型路径
MODEL_PATH="BAAI/bge-reranker-v2-m3"

# 启动 vLLM 服务
vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 8000 \
  --task score \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.5

# 参数说明：
# --task score: 使用评分模式（rerank 需要）
# --max-model-len 8192: 最大序列长度
# --gpu-memory-utilization 0.5: 使用 50% GPU 内存（根据实际情况调整）
