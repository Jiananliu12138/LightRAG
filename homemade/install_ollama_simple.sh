#!/bin/bash
# 简单直接的 Ollama 安装脚本（无需 sudo）

set -e

echo "=========================================="
echo "  安装 Ollama 到本地目录"
echo "=========================================="

# 创建目录
mkdir -p ~/ollama/bin
mkdir -p ~/ollama/data

# 下载 Ollama 可执行文件
echo ""
echo "📥 下载 Ollama..."
curl -L https://ollama.ai/download/ollama-linux-amd64 -o ~/ollama/bin/ollama

# 添加执行权限
chmod +x ~/ollama/bin/ollama

# 配置环境变量
echo ""
echo "⚙️  配置环境变量..."

if ! grep -q "export PATH=\$HOME/ollama/bin:\$PATH" ~/.bashrc; then
    echo '' >> ~/.bashrc
    echo '# Ollama' >> ~/.bashrc
    echo 'export PATH=$HOME/ollama/bin:$PATH' >> ~/.bashrc
    echo 'export OLLAMA_MODELS=$HOME/ollama/data/models' >> ~/.bashrc
    echo "   ✅ 已添加到 ~/.bashrc"
else
    echo "   ⚠️  环境变量已存在"
fi

# 临时设置（当前会话）
export PATH=$HOME/ollama/bin:$PATH
export OLLAMA_MODELS=$HOME/ollama/data/models

# 验证
echo ""
echo "🔍 验证安装..."
if ~/ollama/bin/ollama --version; then
    echo ""
    echo "=========================================="
    echo "  ✅ 安装成功！"
    echo "=========================================="
    echo ""
    echo "📋 下一步："
    echo ""
    echo "1. 激活环境变量："
    echo "   source ~/.bashrc"
    echo ""
    echo "2. 启动 Ollama 服务（新终端窗口）："
    echo "   ollama serve"
    echo ""
    echo "3. 拉取 Qwen2.5-7B 模型（另一个终端）："
    echo "   ollama pull qwen2.5:7b-instruct"
    echo ""
    echo "4. 测试模型："
    echo "   ollama run qwen2.5:7b-instruct '你好'"
    echo ""
else
    echo "❌ 安装验证失败"
    exit 1
fi
