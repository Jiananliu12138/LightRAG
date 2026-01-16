#!/usr/bin/env python3
"""
从 Hugging Face 下载 GGUF 格式模型
LLM: Qwen2.5-32B-Instruct (Q4_K_M, ~19.8GB, 5 个分片文件)
Embedding: nomic-embed-text-v1.5 (Q4_K_M, ~100MB)
专为 Ollama 部署优化

注意：LLM 模型由 5 个分片文件组成，必须下载全部分片才能使用
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download

# 设置下载目录
MODELS_DIR = Path(__file__).parent
os.makedirs(MODELS_DIR, exist_ok=True)

print(f"模型将下载到: {MODELS_DIR.absolute()}")
print("=" * 60)

# ============== 模型配置（GGUF Q4_K_M 量化）==============

# GGUF 量化格式: 使用 Q4_K_M (平衡质量和大小)
# LLM: Qwen2.5-32B-Instruct (~19.8GB, 5 个分片，必须全部下载)
# Embedding: nomic-embed-text-v1.5 (~100MB, 单文件)

LLM_CONFIG = {
    "name": "Qwen2.5-32B-Instruct",
    "repo_id": "Qwen/Qwen2.5-32B-Instruct-GGUF",
    "files": [  # 分片文件列表 (必须下载全部 5 个分片)
        "qwen2.5-32b-instruct-q4_k_m-00001-of-00005.gguf",  # ~3.96GB
        "qwen2.5-32b-instruct-q4_k_m-00002-of-00005.gguf",  # ~3.95GB
        "qwen2.5-32b-instruct-q4_k_m-00003-of-00005.gguf",  # ~3.99GB
        "qwen2.5-32b-instruct-q4_k_m-00004-of-00005.gguf",  # ~3.95GB
        "qwen2.5-32b-instruct-q4_k_m-00005-of-00005.gguf",  # 最后一个分片
    ],
    "size": "~19.8GB (5 个分片，缺一不可)",
}

EMBEDDING_CONFIG = {
    "name": "nomic-embed-text-v1.5",
    "repo_id": "nomic-ai/nomic-embed-text-v1.5-GGUF",
    "file": "nomic-embed-text-v1.5.Q4_K_M.gguf",  # ~100MB, Q4_K_M 量化
    "format": "gguf",  # GGUF 格式
    "size": "~100MB",
    "dim": 768,
    "description": "英文嵌入模型，GGUF 格式",
}


def download_llm_gguf():
    """下载 LLM 的 GGUF Q4_K_M 量化版本（所有分片文件）"""
    print("\n" + "="*60)
    print(f"下载语言模型: {LLM_CONFIG['name']}")
    print("量化格式: Q4_K_M")
    print(f"文件大小: {LLM_CONFIG['size']}")
    print(f"⚠️  注意：需要下载全部 {len(LLM_CONFIG['files'])} 个分片文件")
    print("="*60)
    
    local_dir = MODELS_DIR / LLM_CONFIG['name']
    os.makedirs(local_dir, exist_ok=True)
    
    print("\n开始下载...")
    print(f"仓库: {LLM_CONFIG['repo_id']}")
    print(f"保存到: {local_dir}\n")
    
    try:
        downloaded_files = []
        
        # 下载所有分片文件
        print(f"📥 下载 {len(LLM_CONFIG['files'])} 个分片文件:")
        print("   (每个分片约 4GB，总计约 20GB，请耐心等待)\n")
        
        for i, filename in enumerate(LLM_CONFIG['files'], 1):
            print(f"[{i}/{len(LLM_CONFIG['files'])}] 正在下载: {filename}")
            hf_hub_download(
                repo_id=LLM_CONFIG['repo_id'],
                filename=filename,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            downloaded_files.append(filename)
            print(f"    ✅ 已下载: {filename}\n")
        
        # 下载配置文件（可选但推荐）
        config_files = [
            "config.json",
            "tokenizer.json", 
            "tokenizer_config.json",
            "special_tokens_map.json",
        ]
        
        print("\n📥 下载配置文件...")
        for config_file in config_files:
            try:
                hf_hub_download(
                    repo_id=LLM_CONFIG['repo_id'],
                    filename=config_file,
                    local_dir=str(local_dir),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
                downloaded_files.append(config_file)
                print(f"✅ 已下载: {config_file}")
            except Exception:
                print(f"⚠️  跳过: {config_file}")
        
        print(f"\n✅ {LLM_CONFIG['name']} 下载完成！")
        print("   量化格式: Q4_K_M")
        print(f"   分片文件: {len(LLM_CONFIG['files'])} 个")
        print(f"   总文件数: {len(downloaded_files)} (包含配置文件)")
        return True, local_dir, "Q4_K_M"
    
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def download_embedding_model():
    """下载嵌入模型（GGUF Q4_K_M 量化版本）"""
    print("\n" + "="*60)
    print(f"下载嵌入模型: {EMBEDDING_CONFIG['name']}")
    print("量化格式: Q4_K_M")
    print("语言: 英文 (English)")
    print(f"维度: {EMBEDDING_CONFIG['dim']}, 大小: {EMBEDDING_CONFIG['size']}")
    print(f"说明: {EMBEDDING_CONFIG['description']}")
    print("="*60)
    
    filename = EMBEDDING_CONFIG['file']
    local_dir = MODELS_DIR / EMBEDDING_CONFIG['name']
    os.makedirs(local_dir, exist_ok=True)
    
    print("\n开始下载...")
    print(f"仓库: {EMBEDDING_CONFIG['repo_id']}")
    print(f"文件: {filename}")
    print(f"保存到: {local_dir}\n")
    
    try:
        # 下载单个 GGUF 文件
        print(f"📥 正在下载: {filename}")
        hf_hub_download(
            repo_id=EMBEDDING_CONFIG['repo_id'],
            filename=filename,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"✅ 已下载: {filename}")
        
        print(f"\n✅ {EMBEDDING_CONFIG['name']} 下载完成！")
        print("   量化格式: Q4_K_M")
        return True, local_dir
    
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def create_ollama_modelfile(llm_dir, embed_dir):
    """创建 Ollama Modelfile 配置"""
    print("\n" + "="*60)
    print("生成 Ollama Modelfile 配置")
    print("="*60)
    
    # 验证所有分片文件是否存在
    print("\n🔍 验证 LLM 分片文件...")
    missing_files = []
    for shard_file in LLM_CONFIG['files']:
        shard_path = llm_dir / shard_file
        if shard_path.exists():
            print(f"   ✅ {shard_file}")
        else:
            print(f"   ❌ {shard_file} (缺失)")
            missing_files.append(shard_file)
    
    if missing_files:
        print(f"\n⚠️  警告：缺少 {len(missing_files)} 个分片文件，模型可能无法正常加载！")
        print("   请重新运行下载脚本以获取所有分片。")
    else:
        print("\n✅ 所有 5 个分片文件完整！")
    
    # 生成 LLM Modelfile
    # 注意：Ollama 通过文件名模式自动识别分片，只需指向第一个分片
    llm_gguf_file = llm_dir / LLM_CONFIG['files'][0]
    llm_modelfile = MODELS_DIR / f"Modelfile.{LLM_CONFIG['name']}"
    
    llm_content = f'''# Ollama Modelfile for {LLM_CONFIG['name']}
#
# 分片模型说明：
# 此模型由 5 个分片文件组成 (00001-of-00005 至 00005-of-00005)
# Ollama 会自动检测并加载同目录下的所有分片
# 
# 分片文件列表：
# - {LLM_CONFIG['files'][0]}
# - {LLM_CONFIG['files'][1]}
# - {LLM_CONFIG['files'][2]}
# - {LLM_CONFIG['files'][3]}
# - {LLM_CONFIG['files'][4]}
#
# ⚠️ 所有分片必须在同一目录，缺一不可！

FROM {llm_gguf_file.absolute()}

TEMPLATE """{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>
"""

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.8
PARAMETER top_k 20
PARAMETER num_ctx 8192
'''
    
    with open(llm_modelfile, 'w', encoding='utf-8') as f:
        f.write(llm_content)
    
    print(f"\n✅ 已生成: {llm_modelfile}")
    if missing_files:
        print("   ⚠️  但请注意：存在缺失的分片文件！")
    
    # 验证嵌入模型文件是否存在
    print("\n🔍 验证嵌入模型文件...")
    embed_gguf_file = embed_dir / EMBEDDING_CONFIG['file']
    if embed_gguf_file.exists():
        print(f"   ✅ {EMBEDDING_CONFIG['file']}")
    else:
        print(f"   ❌ {EMBEDDING_CONFIG['file']} (缺失)")
    
    # 生成嵌入模型 Modelfile（GGUF 格式，单文件）
    embed_modelfile = MODELS_DIR / f"Modelfile.{EMBEDDING_CONFIG['name']}"
    embed_content = f'''# Ollama Modelfile for {EMBEDDING_CONFIG['name']}
# 
# 嵌入模型说明：
# 此模型为单文件 GGUF 格式 (Q4_K_M 量化)
# 文件: {EMBEDDING_CONFIG['file']}

FROM {embed_gguf_file.absolute()}

PARAMETER num_ctx 8192
'''
    
    with open(embed_modelfile, 'w', encoding='utf-8') as f:
        f.write(embed_content)
    
    print(f"\n✅ 已生成: {embed_modelfile}")
    
    return llm_modelfile, embed_modelfile


def main():
    print("\n" + "="*60)
    print("  LightRAG + Ollama 模型下载工具")
    print("  Qwen2.5-32B + Nomic Embed (GGUF Q4_K_M)")
    print("="*60)
    print("\n配置:")
    print(f"  - LLM: {LLM_CONFIG['name']} (Q4_K_M, {LLM_CONFIG['size']})")
    print(f"  - Embedding: {EMBEDDING_CONFIG['name']} (Q4_K_M, {EMBEDDING_CONFIG['size']})")
    print("="*60)
    
    # 检查依赖
    try:
        from huggingface_hub import snapshot_download  # noqa: F401
    except ImportError:
        print("\n❌ 缺少依赖，请先安装:")
        print("pip install huggingface_hub")
        return
    
    results = []
    
    # 下载 LLM
    print("\n📥 第 1 步: 下载语言模型 (GGUF 格式)")
    llm_success, llm_dir, llm_quant = download_llm_gguf()
    if llm_success:
        results.append(("LLM", LLM_CONFIG['name'], llm_dir, llm_quant))
    
    # 下载嵌入模型
    print("\n📥 第 2 步: 下载嵌入模型")
    embed_success, embed_dir = download_embedding_model()
    if embed_success:
        results.append(("Embedding", EMBEDDING_CONFIG['name'], embed_dir, "Q4_K_M"))
    
    # 生成 Modelfile
    if llm_success and embed_success:
        print("\n📝 第 3 步: 生成 Ollama 配置文件")
        llm_modelfile, embed_modelfile = create_ollama_modelfile(llm_dir, embed_dir)
    
    # 总结
    print("\n" + "="*60)
    print("  下载完成汇总")
    print("="*60)
    
    if results:
        print("\n✅ 已下载的模型:")
        for model_type, name, path, quant in results:
            print(f"\n  [{model_type}] {name}")
            print(f"    路径: {path}")
            print(f"    格式: {quant}")
        
        if llm_success and embed_success:
            print("\n" + "="*60)
            print("📋 后续步骤 - 使用 Ollama 部署:")
            print("="*60)
            print("\n1. 创建 LLM 模型:")
            print(f"   ollama create qwen2.5-32b -f {llm_modelfile}")
            
            print("\n2. 创建嵌入模型:")
            print(f"   ollama create nomic-embed -f {embed_modelfile}")
            
            print("\n3. 测试模型:")
            print("   ollama run qwen2.5-32b \"Hello, please introduce yourself\"")
            
            print("\n4. 修改 LightRAG 配置，使用本地模型:")
            print("   LLM_MODEL=qwen2.5-32b")
            print("   EMBEDDING_MODEL=nomic-embed")
            
            print("\n5. 运行 LightRAG demo:")
            print("   cd F:\\thesis\\LightRAG")
            print("   python examples\\lightrag_ollama_demo.py")
    else:
        print("\n⚠️  没有成功下载任何模型")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  下载已取消")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
