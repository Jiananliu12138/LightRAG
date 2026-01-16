#!/bin/bash
# ============================================================
# Ollama 模型部署脚本
# 用于部署 Qwen2.5-32B-Instruct 和 nomic-embed-text-v1.5
# ============================================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印函数
print_header() {
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR"
LLM_MODELFILE="$MODELS_DIR/Modelfile.Qwen2.5-32B-Instruct"
EMBED_MODELFILE="$MODELS_DIR/Modelfile.nomic-embed-text-v1.5"
LLM_MODEL_NAME="qwen2.5-32b"
EMBED_MODEL_NAME="nomic-embed"

echo ""
print_header "  Ollama 模型部署工具"
echo "  LLM: Qwen2.5-32B-Instruct (Q4_K_M, 5个分片)"
echo "  Embedding: nomic-embed-text-v1.5 (Q4_K_M)"
echo ""

# ============================================================
# [1/5] 检查 Ollama 安装
# ============================================================
echo "[1/5] 检查 Ollama 安装..."
if ! command -v ollama &> /dev/null; then
    print_error "未找到 Ollama 命令"
    echo ""
    echo "请先安装 Ollama:"
    echo "  macOS/Linux: curl -fsSL https://ollama.ai/install.sh | sh"
    echo "  Windows: https://ollama.ai/download"
    echo ""
    exit 1
fi
print_success "Ollama 已安装"
echo ""

# ============================================================
# [2/5] 检查 Modelfile 文件
# ============================================================
echo "[2/5] 检查 Modelfile 文件..."
if [ ! -f "$LLM_MODELFILE" ]; then
    print_error "未找到 LLM Modelfile"
    echo "   路径: $LLM_MODELFILE"
    echo ""
    echo "请先运行 download_models.py 下载模型并生成 Modelfile"
    exit 1
fi
print_success "找到 LLM Modelfile"

if [ ! -f "$EMBED_MODELFILE" ]; then
    print_error "未找到嵌入模型 Modelfile"
    echo "   路径: $EMBED_MODELFILE"
    echo ""
    echo "请先运行 download_models.py 下载模型并生成 Modelfile"
    exit 1
fi
print_success "找到嵌入模型 Modelfile"
echo ""

# ============================================================
# [3/5] 验证模型文件完整性
# ============================================================
echo "[3/5] 验证模型文件完整性..."
LLM_DIR="$MODELS_DIR/Qwen2.5-32B-Instruct"
EMBED_DIR="$MODELS_DIR/nomic-embed-text-v1.5"

# 检查 LLM 分片文件
MISSING_SHARDS=0
for i in 00001 00002 00003 00004 00005; do
    SHARD_FILE="$LLM_DIR/qwen2.5-32b-instruct-q4_k_m-${i}-of-00005.gguf"
    if [ ! -f "$SHARD_FILE" ]; then
        print_error "缺少分片: qwen2.5-32b-instruct-q4_k_m-${i}-of-00005.gguf"
        MISSING_SHARDS=1
    else
        print_success "qwen2.5-32b-instruct-q4_k_m-${i}-of-00005.gguf"
    fi
done

if [ $MISSING_SHARDS -eq 1 ]; then
    echo ""
    print_error "LLM 分片文件不完整"
    echo "请重新运行 download_models.py 下载所有分片"
    exit 1
fi

# 检查嵌入模型文件
EMBED_FILE="$EMBED_DIR/nomic-embed-text-v1.5.Q4_K_M.gguf"
if [ ! -f "$EMBED_FILE" ]; then
    print_error "嵌入模型文件不存在"
    echo "   路径: $EMBED_FILE"
    exit 1
fi
print_success "nomic-embed-text-v1.5.Q4_K_M.gguf"
echo ""

# ============================================================
# [4/5] 创建 LLM 模型
# ============================================================
echo "[4/5] 创建 Ollama LLM 模型..."
echo "执行命令: ollama create $LLM_MODEL_NAME -f \"$LLM_MODELFILE\""
echo ""

if ollama create "$LLM_MODEL_NAME" -f "$LLM_MODELFILE"; then
    echo ""
    print_success "LLM 模型创建成功: $LLM_MODEL_NAME"
else
    echo ""
    print_error "LLM 模型创建失败"
    exit 1
fi
echo ""

# ============================================================
# [5/5] 创建嵌入模型
# ============================================================
echo "[5/5] 创建 Ollama 嵌入模型..."
echo "执行命令: ollama create $EMBED_MODEL_NAME -f \"$EMBED_MODELFILE\""
echo ""

if ollama create "$EMBED_MODEL_NAME" -f "$EMBED_MODELFILE"; then
    echo ""
    print_success "嵌入模型创建成功: $EMBED_MODEL_NAME"
else
    echo ""
    print_error "嵌入模型创建失败"
    exit 1
fi
echo ""

# ============================================================
# 显示已创建的模型
# ============================================================
print_header "  部署完成!"
echo ""
echo "📋 已创建的模型:"
ollama list
echo ""

# ============================================================
# 提供测试命令
# ============================================================
print_header "  测试模型"
echo ""
echo "1. 测试 LLM 模型:"
echo "   ollama run $LLM_MODEL_NAME \"你好，请介绍一下自己\""
echo ""
echo "2. 测试嵌入模型:"
echo "   ollama run $EMBED_MODEL_NAME"
echo ""
echo "3. 在 LightRAG 中使用:"
echo "   设置环境变量:"
echo "     export LLM_MODEL=$LLM_MODEL_NAME"
echo "     export EMBEDDING_MODEL=$EMBED_MODEL_NAME"
echo ""
echo "   然后运行:"
echo "     cd /path/to/LightRAG"
echo "     python examples/lightrag_ollama_demo.py"
echo ""
print_header "============================================================"
