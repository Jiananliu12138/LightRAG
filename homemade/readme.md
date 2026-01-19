Todo： 1.运行 lightrag-server 以暴露 lightrag-api for evaluation
pip install -e ".[api,evaluation]"

lightrag-server \
 --host 0.0.0.0\
 --port 9621 \
 --llm-binding ollama \
 --embedding-binding ollama \
 --ollama-llm-model qwen2.5-7b \
 --ollama-embedding-model nomic-embed

.env

# LLM 配置

LLM_BINDING=ollama
LLM_MODEL=qwen2.5-32b
OLLAMA_LLM_API_BASE=http://localhost:11434

# 嵌入模型配置

EMBEDDING_BINDING=ollama
EMBEDDING_MODEL=nomic-embed
OLLAMA_EMBEDDING_API_BASE=http://localhost:11434

# 存储配置

WORKING_DIR=./rag_storage
KV_STORAGE=JsonKVStorage
VECTOR_STORAGE=NanoVectorDBStorage
GRAPH_STORAGE=NetworkXStorage

2.可以启动 eval
在本地部署了嵌入模型，llm 模型和 rag server 之后就可以去跑评估了

3.前端
cd lightrag_webui
bun install --frozen-lockfile
bun run build
cd ..
